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
   blocks and across timesteps, and the TopK threshold / L1 penalty are both
   scale-sensitive. We rescale so that E[||x||_2] = sqrt(d), which makes a
   single hyper-parameter setting transfer across sites. The scale is *saved* --
   steering has to undo it, because the hook sees raw activations.

CRASH SAFETY (why ShardWriter checkpoints)

Free GPU sessions die: idle timeouts, preemption, the 12-hour cap. A shard is an
append-only binary file plus a small JSON recording how many tokens in it are
valid. If the process is killed mid-write, the binary file can end with a
partial batch that the JSON does not account for. ``ShardWriter`` therefore
treats the JSON as the source of truth: on open it truncates the binary file to
exactly the number of tokens the JSON vouches for, discarding any torn tail.
``checkpoint()`` makes the JSON update atomic (write a temp file, then rename),
so the JSON itself can never be half-written.

POOLING WITHOUT COPYING

The pooled-vs-bucketed experiment needs an SAE trained on all three timestep
buckets together. Physically concatenating the shards would duplicate ~8 GB of
activations, which does not fit alongside the originals in Kaggle's 20 GB
working directory. Instead a pooled shard is a JSON manifest listing source
shards and how many tokens to take from each; ``ActivationStore`` reads from the
sources directly.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import torch

BYTES_PER_VALUE = 2  # float16


@dataclass
class ShardMeta:
    site: str
    bucket: int
    d_in: int
    n_tokens: int
    dtype: str = "float16"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShardMeta":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def shard_slug(site: str, bucket) -> str:
    return f"{site.replace('.', '_')}__b{bucket}"


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def truncate_shard(bin_path: Path, d_in: int, n_tokens: int) -> None:
    """Cut a shard's binary file to exactly ``n_tokens`` rows."""
    want = n_tokens * d_in * BYTES_PER_VALUE
    if bin_path.exists() and bin_path.stat().st_size != want:
        with open(bin_path, "r+b") as f:
            f.truncate(want)


class ShardWriter:
    """Appends captured activations to a per-(site, bucket) binary shard.

    ``expected_tokens`` lets a resuming caller force the shard back to a
    known-good length (from its own progress checkpoint), which matters when
    several shards are checkpointed one after another and the process died
    between two of them.
    """

    def __init__(self, root: str | Path, site: str, bucket: int, d_in: int,
                 expected_tokens: int | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.site, self.bucket, self.d_in = site, bucket, d_in
        self.path = self.root / f"{shard_slug(site, bucket)}.bin"
        self.meta_path = self.root / f"{shard_slug(site, bucket)}.json"

        if expected_tokens is not None:
            self.n_tokens = int(expected_tokens)
        elif self.meta_path.exists():
            self.n_tokens = int(json.loads(self.meta_path.read_text())["n_tokens"])
        else:
            self.n_tokens = 0

        truncate_shard(self.path, d_in, self.n_tokens)   # drop any torn tail
        self._f = open(self.path, "ab")
        self.checkpoint()

    def write(self, x: torch.Tensor) -> None:
        arr = x.detach().to(torch.float16).cpu().numpy()
        assert arr.shape[-1] == self.d_in, f"width mismatch {arr.shape[-1]} != {self.d_in}"
        self._f.write(arr.tobytes())
        self.n_tokens += arr.shape[0]

    def checkpoint(self) -> None:
        self._f.flush()
        _atomic_write_json(self.meta_path, ShardMeta(
            self.site, self.bucket, self.d_in, self.n_tokens).to_dict())

    def close(self) -> None:
        self.checkpoint()
        self._f.close()


def find_shards(root: str | Path, include_pooled: bool = True) -> list[Path]:
    """Every trainable shard in a directory, identified by its metadata JSON."""
    out = []
    for p in sorted(Path(root).glob("*.json")):
        if p.name in ("manifest.json", "progress.json") or p.name.endswith(".tmp"):
            continue
        try:
            meta = json.loads(p.read_text())
        except Exception:
            continue
        if "d_in" not in meta or "n_tokens" not in meta:
            continue
        if int(meta.get("bucket", 0)) < 0 and not include_pooled:
            continue
        out.append(p)
    return out


class ActivationStore:
    """Memory-mapped reader with a shuffling buffer and fixed normalisation.

    Accepts a shard's ``.bin`` or ``.json`` path, or a pooled manifest JSON
    (one with a ``sources`` list), in which case it reads from several shards
    as if they were one.
    """

    def __init__(self, shard_path: str | Path, buffer_tokens: int = 1 << 19,
                 device: str = "cpu", seed: int = 0, normalize: bool = True):
        p = Path(shard_path)
        meta_path = p.with_suffix(".json") if p.suffix == ".bin" else p
        meta = json.loads(meta_path.read_text())
        self.path = meta_path
        self.meta = ShardMeta.from_dict(meta)
        self.d_in = self.meta.d_in
        self.device = device
        self.normalize = normalize

        if "sources" in meta:
            srcs = [(meta_path.parent / s["path"], int(s["n_take"])) for s in meta["sources"]]
        else:
            srcs = [(meta_path.with_suffix(".bin"), int(meta["n_tokens"]))]

        self._mms, self._ns = [], []
        for bin_path, n in srcs:
            mm = np.memmap(bin_path, dtype=np.float16, mode="r").reshape(-1, self.d_in)
            n = min(n, mm.shape[0])
            self._mms.append(mm)
            self._ns.append(n)
        self._offsets = np.cumsum([0] + self._ns)
        self.n_tokens = int(self._offsets[-1])
        self.meta.n_tokens = self.n_tokens
        self.rng = np.random.default_rng(seed)

        self.buffer_tokens = min(buffer_tokens, self.n_tokens)
        self._buf: torch.Tensor | None = None
        self._ptr = 0

        self.scale = self._compute_scale() if normalize else 1.0

    # ---------------------------------------------------------------- reading

    def _gather(self, idx: np.ndarray) -> np.ndarray:
        """Rows for sorted global indices, across however many sources."""
        idx = np.sort(idx)
        parts = []
        for k, mm in enumerate(self._mms):
            lo, hi = self._offsets[k], self._offsets[k + 1]
            sel = idx[(idx >= lo) & (idx < hi)] - lo
            if sel.size:
                parts.append(np.asarray(mm[sel], dtype=np.float32))
        return np.concatenate(parts, axis=0) if parts else np.empty((0, self.d_in), np.float32)

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
        sample = torch.from_numpy(self._gather(idx))
        mean_norm = sample.norm(dim=-1).mean().item()
        if mean_norm < 1e-8:
            return 1.0
        return math.sqrt(self.d_in) / mean_norm

    # ---------------------------------------------------------------- streaming

    def _refill(self) -> None:
        idx = self.rng.choice(self.n_tokens, size=self.buffer_tokens, replace=False)
        arr = self._gather(idx)
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
                f"n={self.n_tokens:,}, d={self.d_in}, sources={len(self._mms)}, "
                f"scale={self.scale:.4f})")
