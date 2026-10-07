#!/usr/bin/env python
"""Stage 3 -- find which learned features correspond to which concept.

    python scripts/03_locate.py --sae runs/<shard>/sae.pt --acts data/acts

Method: run the pipeline on concept prompts and on the neutral discovery
prompts, encode both sets of activations with the trained SAE, and rank
features by how much better they separate the two.

The ranking statistic is **AUC**, not mean-activation difference, on purpose. A
feature that fires enormously on three of forty concept images and never
otherwise has a large mean difference but is not a concept detector -- it is an
image detector. AUC asks "how reliably does this feature order concept tokens
above neutral ones", which is the property you actually want, and it is
invariant to the activation scale so features with different magnitudes stay
comparable.

What this stage does NOT establish: that the feature is causal. A feature can
separate Van Gogh prompts perfectly because it detects thick brushstrokes,
which co-occur with Van Gogh, and amplifying it may do nothing to make an image
more Van Gogh. Stage 4 is what settles that, and until stage 4 runs these should
be called "candidate features", never "the Van Gogh feature".
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.model import SparseAutoencoder
from sae.hooks import ActivationCapture, CaptureConfig
from sae.prompts import DISCOVERY, PROBE


def auc(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Per-feature AUC via rank statistics. pos/neg are [n, d_hidden]."""
    n_pos, n_neg = pos.shape[0], neg.shape[0]
    allv = torch.cat([pos, neg], dim=0)
    ranks = allv.argsort(dim=0).argsort(dim=0).float() + 1.0
    rank_pos = ranks[:n_pos].sum(dim=0)
    return (rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def collect(pipe, cap, prompts, site, bucket, batch=4, steps=25, guidance=7.5,
            size=512, seed=0):
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    chunks = []
    for i in range(0, len(prompts), batch):
        cap.reset(); cap.set_timestep(999.0)
        # Only activations are needed here, so skip the VAE decode.
        pipe(prompts[i:i + batch], num_inference_steps=steps, guidance_scale=guidance,
             height=size, width=size, generator=gen, output_type="latent")
        got = cap.pop()
        if (site, bucket) in got:
            chunks.append(got[(site, bucket)])
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def main():
    # Inference only. Without this, SAE outputs carry autograd graphs: memory is
    # wasted and anything that calls .numpy() on them crashes (stage 5 did).
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--site", required=True)
    ap.add_argument("--bucket", type=int, required=True)
    ap.add_argument("--act-scale", type=float, required=True,
                    help="from runs/<shard>/summary.json -> act_scale")
    ap.add_argument("--out", default="outputs/locate.json")
    ap.add_argument("--repeats", type=int, default=6,
                    help="how many times to repeat each probe prompt (different seeds)")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    from sae.loader import load_pipeline, pick_device

    device = pick_device(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = load_pipeline(args.model, dtype=dtype, device=device)

    sae = SparseAutoencoder.load(args.sae).to(device).eval()
    cap = ActivationCapture(pipe.unet, CaptureConfig(sites=(args.site,),
                                                     tokens_per_pass=64)).attach()

    def codes_for(prompts):
        acts = collect(pipe, cap, prompts, args.site, args.bucket)
        if acts.numel() == 0:
            return torch.empty(0, sae.cfg.d_hidden)
        x = acts.float().to(device) * args.act_scale
        return sae.encode(x).cpu()

    print("collecting neutral baseline...", flush=True)
    neg = codes_for(DISCOVERY * 2)

    results = {}
    for concept, prompts in PROBE.items():
        print(f"collecting {concept}...", flush=True)
        pos = codes_for(prompts * args.repeats)
        if pos.numel() == 0:
            continue

        scores = auc(pos, neg)
        # Require the feature to actually fire on the concept -- a feature that
        # is always off on both sets can have AUC 0.5 with ties and still rank
        # spuriously once noise breaks them.
        fires = (pos > 0).float().mean(dim=0)
        scores = torch.where(fires > 0.02, scores, torch.zeros_like(scores))

        vals, idx = torch.topk(scores, min(args.top, scores.numel()))
        results[concept] = [
            {"feature": int(i), "auc": float(v),
             "fire_rate_concept": float(fires[i]),
             "fire_rate_neutral": float((neg[:, i] > 0).float().mean()),
             "mean_act_concept": float(pos[:, i].mean()),
             "mean_act_neutral": float(neg[:, i].mean())}
            for v, i in zip(vals, idx)
        ]
        top = results[concept][0]
        print(f"  {concept}: top feature {top['feature']} "
              f"AUC {top['auc']:.3f} "
              f"(fires {top['fire_rate_concept']:.2f} vs {top['fire_rate_neutral']:.2f})")

    cap.detach()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sae": args.sae, "site": args.site, "bucket": args.bucket,
        "act_scale": args.act_scale, "candidates": results,
        "note": "candidate features only -- causality is established in stage 4",
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
