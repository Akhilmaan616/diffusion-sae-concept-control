#!/usr/bin/env python
"""Stage 1 -- dump SD1.5 UNet activations, split by site and timestep bucket.

    python scripts/01_capture.py --out data/acts --n-images 400 --steps 25

Storage arithmetic, so you can size a run before starting it:

    tokens = n_images * steps * tokens_per_pass
    bytes  = tokens * d_in * 2          (fp16)

Defaults (400 images, 25 steps, 64 tokens/pass, d=1280) give 640k tokens per
site and ~1.6 GB across four sites -- comfortably inside a Kaggle session. Those
640k tokens are split across three buckets, so ~210k per SAE, which is enough
for a 16x expansion dictionary to converge but not enough to call it saturated.
If you have time, raise --n-images before raising anything else: more prompts
beats more tokens per prompt, because token diversity is what the shuffling
buffer needs.
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.hooks import ActivationCapture, CaptureConfig, SD15_SITES, DEFAULT_SITE
from sae.data import ShardWriter
from sae.prompts import DISCOVERY
from sae.prompt_corpus import build_discovery, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    help="runwayml/stable-diffusion-v1-5 was removed from the Hub; "
                         "this is the live mirror")
    ap.add_argument("--out", default="data/acts")
    ap.add_argument("--sites", nargs="*", default=[DEFAULT_SITE])
    ap.add_argument("--all-sites", action="store_true")
    ap.add_argument("--n-images", type=int, default=400)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tokens-per-pass", type=int, default=64)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", type=int, default=0,
                    help="use a combinatorial corpus of this many distinct prompts "
                         "instead of the 32 hand-written ones (0 = hand-written)")
    args = ap.parse_args()

    from sae.loader import load_pipeline

    sites = list(SD15_SITES) if args.all_sites else args.sites
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = load_pipeline(args.model, dtype=dtype, device=device)

    cfg = CaptureConfig(sites=tuple(sites), cond_only=True,
                        tokens_per_pass=args.tokens_per_pass, seed=args.seed)
    cap = ActivationCapture(pipe.unet, cfg).attach()

    out = Path(args.out)
    writers: dict[tuple[str, int], ShardWriter] = {}

    base = build_discovery(args.corpus, seed=args.seed) if args.corpus else DISCOVERY
    if args.corpus:
        import json as _json
        print(_json.dumps(coverage(base), indent=2))
    prompts = (base * ((args.n_images // len(base)) + 1))[:args.n_images]
    print(f"{len(set(prompts))} distinct prompts across {args.n_images} images "
          f"({args.n_images / len(set(prompts)):.1f} renders each)")
    gen = torch.Generator(device=device).manual_seed(args.seed)

    done = 0
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i + args.batch]
        cap.reset()
        cap.set_timestep(999.0)
        pipe(chunk, num_inference_steps=args.steps, guidance_scale=args.guidance,
             height=args.size, width=args.size, generator=gen,
             callback_on_step_end=cap.diffusers_callback)

        for (site, bucket), tensor in cap.pop().items():
            key = (site, bucket)
            if key not in writers:
                writers[key] = ShardWriter(out, site, bucket, d_in=tensor.shape[-1])
            writers[key].write(tensor)

        done += len(chunk)
        if done % (args.batch * 10) == 0 or done >= len(prompts):
            tot = sum(w.n_tokens for w in writers.values())
            print(f"{done}/{len(prompts)} images  {tot:,} tokens", flush=True)

    cap.detach()
    for w in writers.values():
        w.close()

    manifest = {
        "model": args.model,
        "sites": sites,
        "bucket_edges": list(cfg.bucket_edges),
        "n_images": args.n_images,
        "steps": args.steps,
        "guidance": args.guidance,
        "shards": [
            {"site": s, "bucket": b, "n_tokens": w.n_tokens, "d_in": w.d_in,
             "path": str(w.path)}
            for (s, b), w in sorted(writers.items())
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["shards"], indent=2))


if __name__ == "__main__":
    main()
