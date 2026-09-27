"""Unit tests on synthetic data.

No GPU and no model weights are needed. Every test here is a property that must
hold regardless of what the real activations look like, which is why they are
worth writing: they catch the plumbing bugs that would otherwise show up as
"the SAE trained but the features are garbage" three GPU-hours later.

The sparse-recovery test is the important one. We synthesise data from a known
sparse dictionary and check the SAE recovers it. If that fails, nothing
downstream is trustworthy.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.model import SAEConfig, SparseAutoencoder
from sae.data import ActivationStore, ShardWriter
from sae.train import TrainConfig, train_sae
from sae.hooks import (ActivationCapture, CaptureConfig, SteeringHook,
                       SteerSpec, bucket_of, get_module)


# ---------------------------------------------------------------- synthetic data

def make_sparse_data(n=20000, d=64, n_feats=256, k=4, seed=0):
    """Ground-truth sparse dictionary problem.

    x = D @ z + small noise, with z exactly k-sparse and non-negative.
    A working SAE should recover the columns of D up to permutation and sign.
    """
    g = torch.Generator().manual_seed(seed)
    D = F.normalize(torch.randn(n_feats, d, generator=g), dim=1)
    z = torch.zeros(n, n_feats)
    idx = torch.stack([torch.randperm(n_feats, generator=g)[:k] for _ in range(n)])
    vals = torch.rand(n, k, generator=g) * 2 + 0.5
    z.scatter_(1, idx, vals)
    x = z @ D + 0.01 * torch.randn(n, d, generator=g)
    return x, D


# ------------------------------------------------------------------------ tests

def test_shapes_and_sparsity():
    cfg = SAEConfig(d_in=64, expansion=4, mode="topk", k=8)
    sae = SparseAutoencoder(cfg)
    x = torch.randn(32, 64)
    out = sae(x)
    assert out.x_hat.shape == x.shape
    assert out.codes.shape == (32, 256)
    # TopK must give exactly k active (or fewer, if ReLU zeroed some).
    l0 = (out.codes > 0).float().sum(-1)
    assert (l0 <= cfg.k).all(), "topk exceeded k"
    print(f"  topk L0: mean {l0.mean():.2f} max {l0.max():.0f} (k={cfg.k})")


def test_decoder_unit_norm():
    sae = SparseAutoencoder(SAEConfig(d_in=32, expansion=4))
    sae.W_dec.data = torch.randn_like(sae.W_dec) * 5
    sae.normalize_decoder_()
    norms = sae.W_dec.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print(f"  decoder norms after projection: {norms.min():.6f}..{norms.max():.6f}")


def test_grad_projection_is_tangent():
    """After removing the parallel component, the gradient must be orthogonal
    to each decoder column -- that is the whole point."""
    sae = SparseAutoencoder(SAEConfig(d_in=16, expansion=4))
    sae.normalize_decoder_()
    x = torch.randn(64, 16)
    sae(x).loss.backward()
    sae.set_decoder_norm_grad_()
    dots = (sae.W_dec.grad * sae.W_dec.data).sum(dim=1).abs()
    assert dots.max() < 1e-4, f"gradient not tangent: {dots.max()}"
    print(f"  max |grad . w| after projection: {dots.max():.2e}")


def test_empty_steering_is_identity():
    """The reconstruction-error-preserving path must be an exact no-op when
    there is nothing to edit. If this fails, every steering result is confounded
    by SAE reconstruction error."""
    sae = SparseAutoencoder(SAEConfig(d_in=32, expansion=4, k=6))
    hook = SteeringHook(sae, specs=[], preserve_error=True)
    x = torch.randn(2, 10, 32)
    assert hook._hook(None, None, x) is x   # no specs -> untouched

    hook.specs = [SteerSpec(feature=0, mode="scale", value=1.0)]
    out = hook._hook(None, None, x)
    err = (out - x).abs().max()
    assert err < 1e-4, f"scale-by-1.0 changed the activation by {err}"
    print(f"  identity edit max deviation: {err:.2e}")


def test_error_preservation_matters():
    """Show the failure mode the flag guards against: without error
    preservation, a no-op edit still perturbs the activation."""
    sae = SparseAutoencoder(SAEConfig(d_in=32, expansion=4, k=6))
    x = torch.randn(2, 10, 32)
    spec = [SteerSpec(feature=0, mode="scale", value=1.0)]
    with_err = SteeringHook(sae, spec, preserve_error=True)._hook(None, None, x)
    without = SteeringHook(sae, spec, preserve_error=False)._hook(None, None, x)
    d_with = (with_err - x).norm() / x.norm()
    d_without = (without - x).norm() / x.norm()
    assert d_with < d_without
    print(f"  no-op edit relative change: preserve={d_with:.4f} "
          f"drop={d_without:.4f}  ({d_without / max(d_with, 1e-9):.0f}x worse)")


def test_projection_removes_direction():
    sae = SparseAutoencoder(SAEConfig(d_in=32, expansion=4))
    sae.normalize_decoder_()
    hook = SteeringHook(sae, [SteerSpec(feature=3, mode="project", value=1.0)])
    x = torch.randn(1, 8, 32)
    out = hook._hook(None, None, x)
    d = F.normalize(sae.W_dec[3], dim=0)
    residual = (out @ d).abs().max()
    assert residual < 1e-4, f"direction not removed: {residual}"
    print(f"  residual along erased direction: {residual:.2e}")


def test_bucketing():
    edges = (334.0, 667.0)
    assert bucket_of(10, edges) == 0
    assert bucket_of(333, edges) == 0
    assert bucket_of(334, edges) == 1
    assert bucket_of(666, edges) == 1
    assert bucket_of(667, edges) == 2
    assert bucket_of(999, edges) == 2
    print("  bucket edges correct")


def test_capture_on_fake_unet():
    """Exercise the hook plumbing against a module tree that mimics the
    diffusers naming, including the CFG doubled batch."""
    class Attn(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.lin = nn.Linear(c, c)
        def forward(self, x):
            return self.lin(x)

    class Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.attn2 = Attn(c)
        def forward(self, x):
            return self.attn2(x)

    class Attns(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([Block(c)])
        def forward(self, x):
            return self.transformer_blocks[0](x)

    class Mid(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.attentions = nn.ModuleList([Attns(c)])
        def forward(self, x):
            return self.attentions[0](x)

    class FakeUNet(nn.Module):
        def __init__(self, c=64):
            super().__init__()
            self.mid_block = Mid(c)
        def forward(self, x):
            return self.mid_block(x)

    site = "mid_block.attentions.0.transformer_blocks.0.attn2"
    unet = FakeUNet(64)
    assert isinstance(get_module(unet, site), Attn)

    cap = ActivationCapture(unet, CaptureConfig(sites=(site,), tokens_per_pass=16))
    cap.attach()
    for t in (900.0, 500.0, 100.0):
        cap.set_timestep(t)
        unet(torch.randn(4, 32, 64))     # batch 4 == CFG doubled batch of 2
    got = cap.pop()
    cap.detach()

    assert set(got.keys()) == {(site, 2), (site, 1), (site, 0)}
    for v in got.values():
        assert v.shape == (16, 64), v.shape
    print(f"  captured buckets {sorted(k[1] for k in got)}, "
          f"each {tuple(next(iter(got.values())).shape)} (cond half only)")


def test_timestep_attribution_is_exact():
    """Regression test for a bug the shard sizes exposed on a real run.

    Taking the timestep from ``callback_on_step_end`` labels every activation
    with the PREVIOUS step's t (the callback fires after the step) and the first
    forward pass with a default. On a 25-step DDIM schedule that turns the true
    bucket distribution 9/8/8 into 8/8/9 and mislabels every token by one step.

    Reading the timestep from the UNet's own forward arguments is exact. This
    test drives a fake UNet with the real SD1.5 DDIM grid and asserts the
    recovered distribution matches the schedule, not the lagged version.
    """
    class Attn(nn.Module):
        def __init__(self, c): super().__init__(); self.lin = nn.Linear(c, c)
        def forward(self, x): return self.lin(x)

    class Blk(nn.Module):
        def __init__(self, c): super().__init__(); self.attn2 = Attn(c)
        def forward(self, x): return self.attn2(x)

    class Atn(nn.Module):
        def __init__(self, c):
            super().__init__(); self.transformer_blocks = nn.ModuleList([Blk(c)])
        def forward(self, x): return self.transformer_blocks[0](x)

    class Mid(nn.Module):
        def __init__(self, c):
            super().__init__(); self.attentions = nn.ModuleList([Atn(c)])
        def forward(self, x): return self.attentions[0](x)

    class FakeUNet(nn.Module):
        """Mimics UNet2DConditionModel.forward(sample, timestep, ...)."""
        def __init__(self, c=32):
            super().__init__(); self.mid_block = Mid(c)
        def forward(self, sample, timestep, encoder_hidden_states=None):
            return self.mid_block(sample)

    site = "mid_block.attentions.0.transformer_blocks.0.attn2"
    unet = FakeUNet(32)
    cap = ActivationCapture(unet, CaptureConfig(sites=(site,), tokens_per_pass=8))
    cap.attach()

    # The real SD1.5 grids at 25 steps, steps_offset=1.
    ddim = [int(1000 / 25 * (25 - 1 - i)) + 1 for i in range(25)]
    # PNDM/PLMS (what the Hub pipeline ships) repeats the second timestep.
    pndm = ddim[:2] + [ddim[1]] + ddim[2:]
    assert len(pndm) == 26 and pndm[:4] == [961, 921, 921, 881]

    from collections import Counter
    lagged_pndm = dict(Counter(bucket_of(t, (334.0, 667.0))
                               for t in [999.0] + pndm[:-1]))
    # Pinned to the real Kaggle run: shards of 51,200 / 51,200 / 64,000 tokens
    # at 64 tokens x 100 batches = 8 / 8 / 10 passes. The lagged-PNDM model of
    # the bug must reproduce exactly that, or the diagnosis is wrong.
    assert lagged_pndm == {0: 8, 1: 8, 2: 10}, lagged_pndm

    timesteps = ddim
    for t in timesteps:
        unet(torch.randn(2, 16, 32), torch.tensor(t))       # positional
    got_pos = {k[1]: v.shape[0] // 8 for k, v in cap.pop().items()}

    for t in timesteps:
        unet(torch.randn(2, 16, 32), timestep=torch.tensor([t, t]))   # kwarg, batched
    got_kw = {k[1]: v.shape[0] // 8 for k, v in cap.pop().items()}
    cap.detach()

    expected = dict(Counter(bucket_of(t, (334.0, 667.0)) for t in timesteps))
    assert got_pos == expected, f"positional: {got_pos} != {expected}"
    assert got_kw == expected, f"kwarg: {got_kw} != {expected}"
    assert sum(expected.values()) == 25

    lagged = dict(Counter(bucket_of(t, (334.0, 667.0))
                          for t in [999.0] + timesteps[:-1]))
    assert lagged != expected, "the lagged version should differ -- test is not probing anything"
    print(f"  exact {dict(sorted(expected.items()))} vs "
          f"lagged {dict(sorted(lagged.items()))} (the bug); both arg styles OK")
    print(f"  lagged-PNDM model reproduces the observed Kaggle shards: {lagged_pndm}")


def test_store_roundtrip_and_scaling():
    with tempfile.TemporaryDirectory() as td:
        w = ShardWriter(td, "site.a", 1, d_in=32)
        data = torch.randn(5000, 32) * 7.5
        w.write(data)
        w.close()

        store = ActivationStore(Path(td) / "site_a__b1.bin", buffer_tokens=2048, seed=0)
        assert store.n_tokens == 5000
        b = store.batch(256)
        assert b.shape == (256, 32)
        mean_norm = b.norm(dim=-1).mean().item()
        assert abs(mean_norm - math.sqrt(32)) < 1.0, mean_norm
        print(f"  store: {store.n_tokens} tokens, scale {store.scale:.4f}, "
              f"batch mean-norm {mean_norm:.3f} (target {math.sqrt(32):.3f})")


def test_sae_recovers_known_dictionary():
    """The real test. Train on data generated from a known dictionary and check
    the learned decoder columns match the true ones."""
    x, D_true = make_sparse_data(n=40000, d=64, n_feats=128, k=4, seed=1)

    with tempfile.TemporaryDirectory() as td:
        w = ShardWriter(td, "synth", 0, d_in=64)
        w.write(x)
        w.close()
        store = ActivationStore(Path(td) / "synth__b0.bin", buffer_tokens=1 << 15, seed=0)

        summary = train_sae(
            store,
            SAEConfig(d_in=64, expansion=4, mode="topk", k=4, seed=0),
            TrainConfig(steps=3000, batch_size=1024, lr=1e-3, device="cpu",
                        log_every=1000, warmup_steps=100),
            out_dir=Path(td) / "run",
            verbose=False,
        )

        sae = SparseAutoencoder.load(Path(td) / "run" / "sae.pt")
        learned = F.normalize(sae.W_dec, dim=1)
        true = F.normalize(D_true, dim=1)

        # For each true feature, the best-matching learned column.
        sim = (true @ learned.T).abs()
        best = sim.max(dim=1).values
        frac_recovered = (best > 0.9).float().mean().item()

        print(f"  EV {summary['final']['explained_variance']:.3f}  "
              f"dead {summary['final']['dead_frac']:.3f}")
        print(f"  dictionary recovery: {frac_recovered * 100:.1f}% of true features "
              f"matched at cos>0.9 (median best cos {best.median():.3f})")
        assert summary["final"]["explained_variance"] > 0.9, "failed to reconstruct"
        assert frac_recovered > 0.5, f"only recovered {frac_recovered:.2f}"


def test_l1_mode_trains():
    """L1 mode works, but note the lambda it needs.

    On this synthetic problem the true sparsity is k=3. TopK with k=4 recovers
    the dictionary at EV 0.963 (see the test above). L1 cannot reach both: see
    ``outputs/l1_vs_topk.md`` for the sweep. lambda=0.1 is where L0 becomes
    comparable, and EV collapses to ~0.53 there. That gap is the empirical
    reason this repo defaults to topk, and it is worth being able to quote.
    """
    x, _ = make_sparse_data(n=20000, d=32, n_feats=64, k=3, seed=2)
    with tempfile.TemporaryDirectory() as td:
        w = ShardWriter(td, "synth", 0, d_in=32); w.write(x); w.close()
        store = ActivationStore(Path(td) / "synth__b0.bin", buffer_tokens=1 << 14)
        s = train_sae(
            store,
            SAEConfig(d_in=32, expansion=4, mode="l1", l1_coef=1e-1),
            TrainConfig(steps=1500, batch_size=1024, lr=1e-3, device="cpu",
                        log_every=1000, warmup_steps=100, l1_warmup_steps=300),
            out_dir=Path(td) / "run_l1", verbose=False,
        )
        print(f"  l1 mode: EV {s['final']['explained_variance']:.3f} "
              f"L0 {s['final']['l0']:.1f} (topk reached EV 0.963 at L0 4)")
        assert s["final"]["l0"] < 16, "l1 failed to sparsify"
        assert s["final"]["explained_variance"] > 0.4


def test_partial_checkpoint_detection_and_repair():
    """Reproduces the real local cache layout on the Windows box: unet, vae,
    text_encoder and tokenizer present, but no model_index.json and no
    scheduler/ -- so plain from_pretrained fails. The loader must detect that,
    resolve the snapshot sha directory, and be able to repair it idempotently.
    """
    from sae.loader import find_snapshot, inspect, write_missing_pipeline_files

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "models--runwayml--stable-diffusion-v1-5"
        sha = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
        snap = root / "snapshots" / sha
        for d in ("unet", "vae", "text_encoder", "tokenizer"):
            (snap / d).mkdir(parents=True)

        for entry in (root, root / "snapshots", snap):
            assert find_snapshot(entry).name == sha

        rep = inspect(root)
        assert rep["loadable_directly"] is False
        assert rep["loadable_componentwise"] is True

        written = write_missing_pipeline_files(root)
        assert len(written) == 2
        assert inspect(root)["loadable_directly"] is True
        assert write_missing_pipeline_files(root) == []      # idempotent

        cfg = json.loads((snap / "scheduler" / "scheduler_config.json").read_text())
        assert cfg["beta_schedule"] == "scaled_linear"       # not the diffusers default
        assert cfg["steps_offset"] == 1
        assert cfg["set_alpha_to_one"] is False
        print("  partial cache detected, repaired with SD1.5 scheduler config, idempotent")


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        cfg = SAEConfig(d_in=48, expansion=8, mode="topk", k=5)
        a = SparseAutoencoder(cfg)
        p = Path(td) / "sae.pt"
        a.save(p)
        b = SparseAutoencoder.load(p)
        x = torch.randn(16, 48)
        assert torch.allclose(a(x, update_stats=False).x_hat,
                              b(x, update_stats=False).x_hat, atol=1e-6)
        print("  save/load identical")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
            print("  PASS")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
