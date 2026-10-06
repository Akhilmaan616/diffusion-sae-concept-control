#!/usr/bin/env python
"""Stage 5 -- feature atlas.

    python scripts/05_atlas.py --sae runs/<shard>/sae.pt --site <site> \
        --bucket 1 --act-scale <scale> --features 12 --out outputs/atlas

For each feature: the prompts that activate it most, and *where in the image*
it fires. The spatial map is the part that makes a diffusion atlas more
informative than an LLM one -- a feature is not just "fires on dog prompts", it
is "fires on the dog-shaped region", and you can see that directly.

TWO THINGS THIS STAGE DOES DIFFERENTLY FROM CAPTURE

1. No token subsampling. Training throws away spatial layout on purpose (64
   random tokens per pass is plenty for fitting a dictionary, and storing full
   maps for 400 images would be ~16x the disk). The atlas needs the layout, so
   it keeps whole [seq, C] maps for a much smaller prompt set.

2. Max-pooling over space, not mean. A feature that fires intensely on a small
   region -- which is what a localised concept looks like -- has a low spatial
   mean and a high spatial max. Ranking prompts by the mean would systematically
   hide exactly the features you most want to find, and surface diffuse
   background features instead.

INTERPRETING THE OUTPUT HONESTLY

An atlas shows correlation. A feature whose top prompts all contain dogs is a
candidate "dog feature" and nothing more until stage 4 amplifies it and the dogs
appear. Label features from the atlas, then *test* the label causally. Writing
"feature 8823 = dog" in a README on atlas evidence alone is the single most
common overclaim in this literature.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.model import SparseAutoencoder
from sae.hooks import get_module, bucket_of
from sae.loader import load_pipeline, pick_device
from sae.prompts import DISCOVERY, PROBE


class SpatialCapture:
    """Keeps full [seq, C] maps for one site, tagged with timestep bucket."""

    def __init__(self, unet, site: str, bucket: int, edges=(334.0, 667.0),
                 cond_only: bool = True):
        self.site, self.bucket, self.edges, self.cond_only = site, bucket, edges, cond_only
        self.unet = unet
        self._t = 999.0
        self._maps: list[torch.Tensor] = []
        self._h = None
        self._pre = None

    def attach(self):
        # Exact timestep from the UNet's own arguments -- see the note in
        # sae/hooks.py::ActivationCapture.attach about why the sampler callback
        # is one step late.
        self._pre = self.unet.register_forward_pre_hook(self._ts, with_kwargs=True)
        self._h = get_module(self.unet, self.site).register_forward_hook(self._hook)
        return self

    def _ts(self, module, args, kwargs):
        t = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        if torch.is_tensor(t):
            t = t.flatten()[0] if t.numel() else None
        if t is not None:
            self._t = float(t)

    def detach(self):
        for h in (getattr(self, "_pre", None), self._h):
            if h:
                h.remove()
        self._pre = self._h = None

    def callback(self, pipe, i, t, kw):
        """No-op; retained so the call site does not need changing."""
        return kw

    def _hook(self, module, args, output):
        x = output[0] if isinstance(output, (tuple, list)) else output
        if x.dim() != 3 or bucket_of(self._t, self.edges) != self.bucket:
            return
        if self.cond_only and x.shape[0] % 2 == 0:
            x = x[x.shape[0] // 2:]
        self._maps.append(x.detach().float().cpu())

    def pop_mean(self):
        """Average the maps over the timesteps inside the bucket.

        Averaging within a bucket is defensible because the bucket is defined as
        a regime where the activations are supposed to be comparable -- that is
        the whole premise of bucketing. Averaging *across* buckets would not be.
        """
        if not self._maps:
            return None
        out = torch.stack(self._maps, 0).mean(0)   # [B, seq, C]
        self._maps.clear()
        return out


def overlay(img, heat, alpha=0.55):
    """Red-hot overlay of a [h, w] activation map on a PIL image."""
    from PIL import Image
    heat = heat.detach().float().cpu()
    h = heat - heat.min()
    h = h / max(float(h.max()), 1e-8)
    hm = Image.fromarray((h.numpy() * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)
    hm = np.asarray(hm).astype(np.float32) / 255.0
    base = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    tint = np.stack([hm, hm * 0.25, hm * 0.1], axis=-1)
    w = (alpha * hm)[..., None]
    blend = base * (1.0 - w) + tint * w
    return Image.fromarray((np.clip(blend, 0, 1) * 255).astype(np.uint8))


def contact_sheet(rows, pad=6):
    """rows: list of list of PIL images -> one grid image."""
    from PIL import Image
    if not rows or not rows[0]:
        return None
    w, h = rows[0][0].size
    ncol = max(len(r) for r in rows)
    sheet = Image.new("RGB", (ncol * w + (ncol + 1) * pad,
                              len(rows) * h + (len(rows) + 1) * pad), (18, 18, 20))
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            sheet.paste(im, (pad + j * (w + pad), pad + i * (h + pad)))
    return sheet


def main():
    # Inference only. Without this, SAE outputs carry autograd graphs: memory is
    # wasted and anything that calls .numpy() on them crashes (stage 5 did).
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--site", required=True)
    ap.add_argument("--bucket", type=int, required=True)
    ap.add_argument("--act-scale", type=float, required=True)
    ap.add_argument("--locate", default="outputs/locate.json",
                    help="if present, atlas these candidate features first")
    ap.add_argument("--features", type=int, default=12,
                    help="how many extra high-density features to atlas")
    ap.add_argument("--top-prompts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs/atlas")
    args = ap.parse_args()

    device = pick_device(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = load_pipeline(args.model, dtype=dtype, device=device)
    sae = SparseAutoencoder.load(args.sae).to(device).eval()

    prompts = list(DISCOVERY)
    for ps in PROBE.values():
        prompts.extend(ps)

    cap = SpatialCapture(pipe.unet, args.site, args.bucket).attach()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # [n_prompts, d_hidden] max-pooled codes, plus the per-prompt spatial maps.
    pooled = torch.zeros(len(prompts), sae.cfg.d_hidden)
    images, spatial = [], []

    for i, p in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        img = pipe([p], num_inference_steps=args.steps, height=args.size,
                   width=args.size, generator=gen,
                   callback_on_step_end=cap.callback).images[0]
        maps = cap.pop_mean()
        if maps is None:
            continue
        codes = sae.encode(maps[0].to(device) * args.act_scale).cpu()   # [seq, d_hidden]
        pooled[i] = codes.max(dim=0).values
        images.append(img)
        spatial.append(codes)
        if (i + 1) % 8 == 0:
            print(f"{i + 1}/{len(prompts)} prompts", flush=True)

    cap.detach()

    # Which features to render.
    feats: list[int] = []
    loc_path = Path(args.locate)
    if loc_path.exists():
        for cand in json.loads(loc_path.read_text())["candidates"].values():
            feats.append(cand[0]["feature"])
    # Top up with the features that are most *selective*: high max activation
    # but firing on few prompts. A feature firing on everything is not a concept.
    fire_rate = (pooled > 0).float().mean(dim=0)
    selectivity = pooled.max(dim=0).values * (fire_rate < 0.35).float()
    for f in torch.topk(selectivity, args.features + len(feats)).indices.tolist():
        if f not in feats:
            feats.append(f)
        if len(feats) >= args.features + 4:
            break

    manifest = []
    for f in feats:
        order = torch.argsort(pooled[:, f], descending=True)[:args.top_prompts]
        if float(pooled[order[0], f]) <= 0:
            continue
        row = []
        for idx in order.tolist():
            codes = spatial[idx]
            seq = codes.shape[0]
            side = int(round(seq ** 0.5))
            if side * side != seq:
                continue
            heat = codes[:, f].reshape(side, side)
            row.append(overlay(images[idx].resize((256, 256)), heat))
        sheet = contact_sheet([row]) if row else None
        if sheet is None:
            continue
        sheet.save(out / f"feature_{f:06d}.png")
        manifest.append({
            "feature": f,
            "fire_rate": float(fire_rate[f]),
            "max_activation": float(pooled[:, f].max()),
            "top_prompts": [prompts[i] for i in order.tolist()],
            "top_activations": [float(pooled[i, f]) for i in order.tolist()],
            "image": f"feature_{f:06d}.png",
        })
        print(f"f{f:<7} fires {fire_rate[f]:.2f}  top: {prompts[order[0]][:58]}")

    (out / "atlas.json").write_text(json.dumps({
        "site": args.site, "bucket": args.bucket, "n_prompts": len(prompts),
        "features": manifest,
        "caveat": "correlational evidence only; causality requires stage 4",
    }, indent=2))
    print(f"\nwrote {len(manifest)} feature sheets to {out}")


if __name__ == "__main__":
    main()
