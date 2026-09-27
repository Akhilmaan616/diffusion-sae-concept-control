"""Activation storage and streaming for SAE training.

Two non-obvious things happen here, and both are the difference between an SAE
that finds features and one that finds nothing.

1. SHUFFLING. Tokens captured from one image at one timestep are enormously
   correlated -- they are neighbouring patches of the same picture. Feed them to
   the optimiser in capture order and every batch is "one image", so the
   gradient is dominated by whatever that image happens to contain and the model
   learns image-specific directions instead of reusable features. The standard
   fix is a shuffling buffer: hold N tokens in memory, yield random draws from
   it, refill as it drains. With a buffer much larger than the batch, successive
   batches mix tokens from many images and timesteps.

2. NORMALISATION. Activation scale varies by an order of magnitude across UNet
   blocks and across timesteps, and the L1 penalty / TopK threshold are both
   scale-sensitive. We rescale so that E[||x||_2] = sqrt(d), which makes a
   single hyper-parameter setting transfer across sites. The scale is *saved* --
   steering has to undo it, because the hook sees raw activations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class ShardMeta:
    site: str
    bucket: int
    d_in: int
    n_tokens: int
    dtype: str = "float16"

    def to_dict(self) -> dict:
        return asdict(self)


class ShardWriter:
    """Appends captured activations to a per-(site, bucket) binary shard."""

    def __init__(self, root: str | Path, site: str, bucket: int, d_in: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.site, self.bucket, self.d_in = site, bucket, d_in
        self.path = self.root / f"{self._slug()}.bin"
        self.meta_path = self.root / f"{self._slug()}.json"
        self._f = open(self.path, "ab")
        self.n_tokens = 0
        if self.meta_path.exists():
            self.n_tokens = json.loads(self.meta_path.read_text())["n_tokens"]

    def _slug(self) -> str:
        return f"{self.site.replace('.', '_')}__b{self.bucket}"

    def write(self, x: torch.Tensor) -> None:
        arr = x.detach().to(torch.float16).cpu().numpy()
        assert arr.shape[-1] == self.d_in, f"width mismatch {arr.shape[-1]} != {self.d_in}"
        self._f.write(arr.tobytes())
        self.n_tokens += arr.shape[0]

    def close(self) -> None:
        self._f.close()
        self.meta_path.write_text(json.dumps(
            ShardMeta(self.site, self.bucket, self.d_in, self.n_tokens).to_dict(), indent=2
        ))


class ActivationStore:
    """Memory-mapped reader with a shuffling buffer and fixed normalisation."""

    def __init__(self, shard_path: str | Path, buffer_tokens: int = 1 << 19,
                 device: str = "cpu", seed: int = 0, normalize: bool = True):
        self.path = Path(shard_path)
        meta = json.loads(self.path.with_suffix(".json").read_text())
        self.meta = ShardMeta(**meta)
        self.d_in = self.meta.d_in
        self.device = device
        self.normalize = normalize

        self.mm = np.memmap(self.path, dtype=np.float16, mode="r").reshape(-1, self.d_in)
        self.n_tokens = self.mm.shape[0]
        self.rng = np.random.default_rng(seed)

        self.buffer_tokens = min(buffer_tokens, self.n_tokens)
        self._buf: torch.Tensor | None = None
        self._ptr = 0

        self.scale = self._compute_scale() if normalize else 1.0

    # ---------------------------------------------------------------- scaling

    def _compute_scale(self, n_sample: int = 100_000) -> float:
        """Rescale so E[||x||_2] == sqrt(d_in).

        Returned as a single float, saved alongside the SAE, and applied in
        reverse at steering time. Using the mean L2 norm rather than a per-
        dimension std keeps directions intact -- a per-dimension whitening would
        rotate the space and make decoder columns no longer comparable to raw
        activation directions.
        """
        n = min(n_sample, self.n_tokens)
        idx = self.rng.choice(self.n_tokens, size=n, replace=False)
        sample = torch.from_numpy(np.asarray(self.mm[np.sort(idx)], dtype=np.float32))
        mean_norm = sample.norm(dim=-1).mean().item()
        if mean_norm < 1e-8:
            return 1.0
        return math.sqrt(self.d_in) / mean_norm

    # ---------------------------------------------------------------- streaming

    def _refill(self) -> None:
        idx = np.sort(self.rng.choice(self.n_tokens, size=self.buffer_tokens, replace=False))
        arr = np.asarray(self.mm[idx], dtype=np.float32)
        self.rng.shuffle(arr)
        buf = torch.from_numpy(arr)
        if self.normalize:
            buf = buf * self.scale
        self._buf = buf.to(self.device)
        self._ptr = 0

    def batch(self, batch_size: int) -> torch.Tensor:
        if self._buf is None or self._ptr + batch_size > self._buf.shape[0]:
            self._refill()
        out = self._buf[self._ptr:self._ptr + batch_size]
        self._ptr += batch_size
        return out

    def iter_batches(self, batch_size: int, n_steps: int):
        for _ in range(n_steps):
            yield self.batch(batch_size)

    def __repr__(self) -> str:
        return (f"ActivationStore({self.meta.site}, bucket={self.meta.bucket}, "
                f"n={self.n_tokens:,}, d={self.d_in}, scale={self.scale:.4f})")
