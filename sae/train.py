"""SAE training loop.

Kept deliberately small and readable: you have to be able to explain every line
of this in an interview, and there is nothing here that needs a framework.

The metrics that actually matter, and what a bad run looks like:

  explained_variance  fraction of activation variance the SAE reconstructs.
                      Below ~0.5 the SAE is not modelling the data; above ~0.95
                      at low L0 is suspiciously good and usually means the
                      activations are low-rank (check!).
  l0                  mean number of active features per token. For topk this is
                      exactly k by construction, so it is only a sanity check;
                      for l1 it is the number you are actually tuning.
  dead_frac           fraction of latents that have not fired recently. Above
                      ~0.3 and you are wasting most of the dictionary -- lower
                      the expansion factor or raise aux_coef.
  feature_density     per-feature firing rate. The healthy shape is a broad
                      log-spaced spread. A spike of features firing on >10% of
                      tokens usually means they have absorbed the mean rather
                      than a concept.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .data import ActivationStore
from .model import SAEConfig, SparseAutoencoder


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 4096
    lr: float = 3e-4
    warmup_steps: int = 500
    # The L1 coefficient is ramped from 0 over this many steps. Starting at full
    # strength kills most latents before they learn anything -- the model finds
    # it cheaper to output zero than to represent structure it has not yet
    # discovered. Ignored in topk mode, which has no such penalty.
    l1_warmup_steps: int = 2_000
    log_every: int = 200
    eval_every: int = 2_000
    device: str = "cuda"
    grad_clip: float = 1.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    # Linear decay over the back half -- keeps the final features sharp.
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * max(0.05, 1.0 - progress)


@torch.no_grad()
def feature_density(sae: SparseAutoencoder, store: ActivationStore,
                    n_batches: int = 20, batch_size: int = 4096) -> torch.Tensor:
    """Firing rate per latent, measured over a fresh sample."""
    counts = torch.zeros(sae.cfg.d_hidden, device=next(sae.parameters()).device)
    total = 0
    for _ in range(n_batches):
        x = store.batch(batch_size).to(counts.device)
        codes = sae.encode(x)
        counts += (codes > 0).float().sum(dim=0)
        total += x.shape[0]
    return counts / max(1, total)


def train_sae(store: ActivationStore, sae_cfg: SAEConfig, cfg: TrainConfig,
              out_dir: str | Path, verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    sae = SparseAutoencoder(sae_cfg).to(device)
    store.device = device

    target_l1 = sae.cfg.l1_coef
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)

    history: list[dict] = []
    t0 = time.time()

    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        if sae.cfg.mode == "l1":
            ramp = min(1.0, (step + 1) / max(1, cfg.l1_warmup_steps))
            sae.cfg.l1_coef = target_l1 * ramp

        x = store.batch(cfg.batch_size)
        out = sae(x)

        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        if sae.cfg.normalize_decoder:
            sae.set_decoder_norm_grad_()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)
        opt.step()
        if sae.cfg.normalize_decoder:
            sae.normalize_decoder_()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            rec = {"step": step, "loss": out.loss.item(),
                   "lr": opt.param_groups[0]["lr"], **out.metrics}
            history.append(rec)
            if verbose:
                print(f"[{step:>6}] loss {rec['loss']:.5f}  "
                      f"EV {rec['explained_variance']:.3f}  "
                      f"L0 {rec['l0']:.1f}  dead {rec['dead_frac']:.3f}", flush=True)

    sae.cfg.l1_coef = target_l1
    density = feature_density(sae, store).cpu()

    sae.save(out_dir / "sae.pt")
    summary = {
        "site": store.meta.site,
        "bucket": store.meta.bucket,
        "n_tokens": store.n_tokens,
        "act_scale": store.scale,
        "sae_cfg": sae_cfg.to_dict(),
        "train_cfg": cfg.to_dict(),
        "final": history[-1] if history else {},
        "density_quantiles": {
            q: float(torch.quantile(density, q)) for q in (0.1, 0.5, 0.9, 0.99)
        },
        "n_never_fired": int((density == 0).sum()),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(density, out_dir / "feature_density.pt")

    if verbose:
        print(json.dumps(summary["final"], indent=2))
        print(f"never fired: {summary['n_never_fired']}/{sae_cfg.d_hidden}")

    return summary
