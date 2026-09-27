#!/usr/bin/env python
"""Reproduce outputs/l1_vs_topk.md. CPU, ~5 min."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import torch
from sae.model import SAEConfig
from sae.data import ActivationStore, ShardWriter
from sae.train import TrainConfig, train_sae
from test_core import make_sparse_data

x, _ = make_sparse_data(n=20000, d=32, n_feats=64, k=3, seed=2)
tc = TrainConfig(steps=1500, batch_size=1024, lr=1e-3, device="cpu",
                 log_every=10**9, warmup_steps=100, l1_warmup_steps=300)

def run(cfg):
    with tempfile.TemporaryDirectory() as td:
        w = ShardWriter(td, "s", 0, d_in=32); w.write(x); w.close()
        st = ActivationStore(Path(td) / "s__b0.bin", buffer_tokens=1 << 14)
        s = train_sae(st, cfg, tc, out_dir=Path(td) / "r", verbose=False)
        return s["final"]["explained_variance"], s["final"]["l0"]

print(f"{'mode':<10}{'setting':<14}{'EV':>8}{'L0':>8}")
ev, l0 = run(SAEConfig(d_in=32, expansion=4, mode="topk", k=4))
print(f"{'topk':<10}{'k=4':<14}{ev:>8.3f}{l0:>8.1f}")
for lam in (2e-3, 1e-2, 3e-2, 1e-1, 3e-1):
    ev, l0 = run(SAEConfig(d_in=32, expansion=4, mode="l1", l1_coef=lam))
    print(f"{'l1':<10}{f'lam={lam}':<14}{ev:>8.3f}{l0:>8.1f}")
