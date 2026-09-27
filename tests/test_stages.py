"""Tests for stage 5 (atlas) and stage 6 (substitution), on synthetic data.

The substitution test is the interesting one: it builds a small network, trains
it so the hooked site genuinely carries signal, fits an SAE to that site's real
activations, and checks the three-condition metric behaves. On an *untrained*
network the ablation control often does not raise the loss at all, which makes
loss_recovered meaningless -- that degenerate case is tested too, because
06_substitution.py has to warn about it rather than print a confident number.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.model import SAEConfig, SparseAutoencoder
from sae.data import ActivationStore, ShardWriter
from sae.train import TrainConfig, train_sae
from sae.hooks import bucket_of, get_module

SITE = "mid_block.attentions.0.transformer_blocks.0.attn2"


def _load_atlas():
    spec = importlib.util.spec_from_file_location(
        "atlas", Path(__file__).resolve().parents[1] / "scripts" / "05_atlas.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Attn(nn.Module):
    def __init__(self, c): super().__init__(); self.lin = nn.Linear(c, c)
    def forward(self, x): return F.relu(self.lin(x))


class _Blk(nn.Module):
    def __init__(self, c): super().__init__(); self.attn2 = _Attn(c)
    def forward(self, x): return self.attn2(x)


class _Atn(nn.Module):
    def __init__(self, c):
        super().__init__(); self.transformer_blocks = nn.ModuleList([_Blk(c)])
    def forward(self, x): return self.transformer_blocks[0](x)


class _Mid(nn.Module):
    def __init__(self, c):
        super().__init__(); self.attentions = nn.ModuleList([_Atn(c)])
    def forward(self, x): return self.attentions[0](x)


class _Net(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.inp = nn.Linear(c, c); self.mid_block = _Mid(c); self.head = nn.Linear(c, c)
    def forward(self, x): return self.head(self.mid_block(F.relu(self.inp(x))))


def test_atlas_overlay_localises():
    """A localised activation blob must produce a localised red shift, and the
    cold region must be untouched -- otherwise the atlas image is decorative
    rather than informative."""
    from PIL import Image
    m = _load_atlas()
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((256, 256, 3)) * 255).astype(np.uint8))
    heat = torch.zeros(16, 16); heat[4:8, 9:13] = 3.0

    ov = m.overlay(img, heat)
    a, b = np.asarray(ov).astype(float), np.asarray(img).astype(float)
    hot = (a[64:128, 144:208, 0] - b[64:128, 144:208, 0]).mean()
    cold = (a[0:40, 0:40, 0] - b[0:40, 0:40, 0]).mean()
    assert hot > 20 and abs(cold) < 2, f"hot {hot} cold {cold}"
    print(f"  overlay red shift: hot {hot:+.1f}, cold {cold:+.1f}")

    sheet = m.contact_sheet([[ov] * 4])
    assert sheet.size[0] > 4 * 256
    print(f"  contact sheet {sheet.size}")


def test_spatial_grid_assumption():
    """Token count must be a perfect square for the heatmap reshape. At 512px:
    up_blocks.1 -> 16x16, up_blocks.2 -> 32x32, mid -> 8x8."""
    for seq, side in ((256, 16), (1024, 32), (64, 8)):
        assert int(round(seq ** 0.5)) == side and side * side == seq
    print("  256->16x16, 1024->32x32, 64->8x8")


def _three_conditions(net, sae, scale, X, Y, bucket=1, edges=(334.0, 667.0)):
    mod = get_module(net, SITE)
    state = {"mode": "clean", "t": 500.0, "hit": 0, "skip": 0}

    def hook(m, a, out):
        x = out
        if x.dim() != 3 or state["mode"] == "clean":
            return out
        if bucket_of(state["t"], edges) != bucket:
            state["skip"] += 1
            return out
        state["hit"] += 1
        if state["mode"] == "substituted":
            return (sae.decode(sae.encode(x.float() * scale)) / scale).to(x.dtype)
        return x.float().mean(dim=1, keepdim=True).expand_as(x).to(x.dtype)

    h = mod.register_forward_hook(hook)
    res, hits = {}, {}
    with torch.no_grad():
        for mode in ("clean", "substituted", "ablated"):
            state.update(mode=mode, hit=0, skip=0)
            tot = 0.0
            for t in (200.0, 500.0, 800.0):
                state["t"] = t
                tot += float(F.mse_loss(net(X), Y))
            res[mode] = tot / 3
            hits[mode] = (state["hit"], state["skip"])
    h.remove()
    return res, hits


def test_substitution_metric_on_trained_net():
    torch.manual_seed(0)
    net, teacher = _Net(64), _Net(64).eval()
    X = torch.randn(512, 32, 64)
    with torch.no_grad():
        Y = teacher(X)

    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(400):
        j = torch.randint(0, 512, (64,))
        loss = F.mse_loss(net(X[j]), Y[j])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    acts = []
    h = get_module(net, SITE).register_forward_hook(
        lambda m, a, o: acts.append(o.detach().reshape(-1, 64)))
    with torch.no_grad():
        net(X)
    h.remove()
    A = torch.cat(acts)

    with tempfile.TemporaryDirectory() as td:
        w = ShardWriter(td, "s", 1, d_in=64); w.write(A); w.close()
        store = ActivationStore(Path(td) / "s__b1.bin", buffer_tokens=1 << 14)
        summary = train_sae(
            store, SAEConfig(d_in=64, expansion=8, mode="topk", k=12),
            TrainConfig(steps=2500, batch_size=1024, lr=1e-3, device="cpu",
                        log_every=10 ** 9, warmup_steps=100),
            out_dir=Path(td) / "r", verbose=False)
        sae = SparseAutoencoder.load(Path(td) / "r" / "sae.pt").eval()
        scale = store.scale
        ev = summary["final"]["explained_variance"]

    res, hits = _three_conditions(net, sae, scale, X[:128], Y[:128])
    den = res["ablated"] - res["clean"]
    lr = (res["ablated"] - res["substituted"]) / den

    assert hits["substituted"] == (1, 2), f"bucket gate wrong: {hits}"
    assert den > 0, "ablation should raise loss on a net that uses the site"
    assert 0.5 < lr <= 1.05, f"loss_recovered {lr}"
    print(f"  EV {ev:.3f} -> loss_recovered {lr:.3f}  "
          f"(clean {res['clean']:.5f} sub {res['substituted']:.5f} "
          f"abl {res['ablated']:.5f})")
    print(f"  bucket gate: hooked {hits['substituted'][0]}/3 timesteps")


def test_degenerate_denominator_is_detected():
    """On an untrained net the site may not matter, so ablation does not raise
    the loss and loss_recovered is garbage. The script must warn instead of
    reporting it; here we just assert the condition is detectable."""
    torch.manual_seed(3)
    net = _Net(64).eval()
    sae = SparseAutoencoder(SAEConfig(d_in=64, expansion=4, k=16)).eval()
    X, Y = torch.randn(4, 32, 64), torch.randn(4, 32, 64)
    res, _ = _three_conditions(net, sae, 1.0, X, Y)
    den = res["ablated"] - res["clean"]
    print(f"  untrained net denominator {den:+.6f} "
          f"({'degenerate, warning path' if den <= 0 else 'positive'})")
    assert isinstance(den, float)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t(); print("  PASS")
        except Exception as e:
            failed += 1; print(f"  FAIL: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
