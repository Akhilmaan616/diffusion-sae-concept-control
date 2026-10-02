#!/usr/bin/env python
"""Stage 1 -- dump SD1.5 UNet activations, split by site and timestep bucket.

    python scripts/01_capture.py --out data/acts --n-images 500 --corpus 256

SIZING

Capture cost is almost entirely the diffusion forward passes. How many of the
resulting tokens you keep is nearly free, so keep them all and spend the time
budget on *images* (which is what adds diversity):

    tokens per (site, bucket) ~= n_images * tokens_per_image_per_pass * passes_in_bucket

At up_blocks.1 (16x16 = 256 tokens per image per pass) with 25 DDIM steps the
three buckets get 9 / 8 / 8 passes, so 500 images give ~1.0-1.15M tokens per
bucket: about 100 tokens per feature for a 10,240-feature dictionary. Storage is
tokens * d_in * 2 bytes, ~8 GB for that one 1280-wide site.

``--tokens-per-pass`` counts per *batch* forward pass, not per image: with
``--batch 8`` the conditional half holds 4 x 256 = 1024 tokens, so 1024 keeps
everything at this site.

SPEED

``output_type="latent"`` skips the VAE decode and safety checker. Nothing here
needs pixels, and the decoder is a noticeable slice of per-image time at 512px.

RESUMING

Free GPU sessions get killed. After every batch the shards are checkpointed and
``progress.json`` records how many images are done and how many tokens each
shard held at that moment. Re-running the same command picks up from there,
first cutting every shard back to the checkpointed length so a batch that was
half-written when the process died is discarded rather than kept twice.
Pass ``--fresh`` to throw the directory away and start over.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.hooks import ActivationCapture, CaptureConfig, SD15_SITES, DEFAULT_SITE
from sae.data import ShardWriter, shard_slug, truncate_shard
from sae.prompts import DISCOVERY
from sae.prompt_corpus import build_discovery, coverage

# Settings that must match for a resume to be valid.
RESUME_KEYS = ("model", "sites", "n_images", "steps", "batch", "tokens_per_pass",
               "guidance", "size", "seed", "corpus")


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    help="runwayml/stable-diffusion-v1-5 was removed from the Hub; "
                         "this is the live mirror")
    ap.add_argument("--out", default="data/acts")
    ap.add_argument("--sites", nargs="*", default=[DEFAULT_SITE])
    ap.add_argument("--all-sites", action="store_true")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tokens-per-pass", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", type=int, default=256,
                    help="combinatorial corpus of this many distinct prompts "
                         "(0 = the 32 hand-written ones)")
    ap.add_argument("--fresh", action="store_true", help="discard any previous capture in --out")
    args = ap.parse_args()

    sites = sorted(SD15_SITES) if args.all_sites else sorted(args.sites)
    out = Path(args.out)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    settings = {"model": args.model, "sites": sites, "n_images": args.n_images,
                "steps": args.steps, "batch": args.batch,
                "tokens_per_pass": args.tokens_per_pass, "guidance": args.guidance,
                "size": args.size, "seed": args.seed, "corpus": args.corpus}

    # ---------------------------------------------------------------- resume
    progress_path = out / "progress.json"
    images_done, shard_tokens = 0, {}
    if progress_path.exists():
        prog = json.loads(progress_path.read_text())
        mismatched = [k for k in RESUME_KEYS if prog["settings"].get(k) != settings[k]]
        if mismatched:
            raise SystemExit(
                f"{out} holds a capture made with different settings ({mismatched}). "
                f"Re-run with the same arguments to resume, or pass --fresh.")
        images_done = prog["images_done"]
        shard_tokens = prog["shard_tokens"]
        if images_done >= args.n_images:
            print(f"capture already complete ({images_done} images) -- nothing to do")
            return
        print(f"resuming at image {images_done}/{args.n_images}")
    else:
        # A directory with shards but no progress file came from an older or
        # interrupted-before-first-checkpoint run; do not silently append to it.
        if any(out.glob("*.bin")):
            raise SystemExit(f"{out} has shards but no progress.json -- pass --fresh")

    # Cut every shard back to its checkpointed length before anything appends.
    for meta_path in out.glob("*__b[0-9].json"):
        meta = json.loads(meta_path.read_text())
        slug = meta_path.stem
        n = int(shard_tokens.get(slug, 0))
        truncate_shard(meta_path.with_suffix(".bin"), meta["d_in"], n)

    # ---------------------------------------------------------------- model
    from sae.loader import load_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = load_pipeline(args.model, dtype=dtype, device=device)

    cfg = CaptureConfig(sites=tuple(sites), cond_only=True,
                        tokens_per_pass=args.tokens_per_pass, seed=args.seed)
    cap = ActivationCapture(pipe.unet, cfg).attach()

    base = build_discovery(args.corpus, seed=args.seed) if args.corpus else DISCOVERY
    if args.corpus and images_done == 0:
        print(json.dumps(coverage(base), indent=2))
    prompts = (base * ((args.n_images // len(base)) + 1))[:args.n_images]
    print(f"{len(set(prompts))} distinct prompts across {args.n_images} images "
          f"({args.n_images / len(set(prompts)):.1f} renders each)", flush=True)

    writers: dict[tuple[str, int], ShardWriter] = {}
    t0, done_this_run = time.time(), 0

    for i in range(images_done, len(prompts), args.batch):
        chunk = prompts[i:i + args.batch]
        cap.reset()
        # Seed per batch from the absolute image index -- both the diffusion
        # noise and the token subsampling -- so a resumed run produces exactly
        # the bytes an uninterrupted run would have.
        batch_seed = args.seed * 1_000_003 + i
        cap._gen.manual_seed(batch_seed)
        gen = torch.Generator(device=device).manual_seed(batch_seed)
        pipe(chunk, num_inference_steps=args.steps, guidance_scale=args.guidance,
             height=args.size, width=args.size, generator=gen, output_type="latent")

        for (site, bucket), tensor in cap.pop().items():
            key = (site, bucket)
            if key not in writers:
                slug = shard_slug(site, bucket)
                writers[key] = ShardWriter(out, site, bucket, d_in=tensor.shape[-1],
                                           expected_tokens=shard_tokens.get(slug, 0))
            writers[key].write(tensor)

        for w in writers.values():
            w.checkpoint()
        images_done = i + len(chunk)
        done_this_run += len(chunk)
        shard_tokens = {shard_slug(s, b): w.n_tokens for (s, b), w in writers.items()}
        _atomic_json(progress_path, {"settings": settings, "images_done": images_done,
                                     "shard_tokens": shard_tokens})

        if (i // args.batch) % 5 == 0 or images_done >= len(prompts):
            rate = done_this_run / max(1e-6, time.time() - t0)
            eta = (len(prompts) - images_done) / max(rate, 1e-6) / 60
            tot = sum(shard_tokens.values())
            print(f"{images_done}/{len(prompts)} images  {tot:,} tokens  "
                  f"{rate * 60:.1f} img/min  eta {eta:.0f} min", flush=True)

    cap.detach()
    for w in writers.values():
        w.close()

    manifest = {
        **settings,
        "bucket_edges": list(cfg.bucket_edges),
        "scheduler": type(pipe.scheduler).__name__,
        "shards": [
            {"site": s, "bucket": b, "n_tokens": w.n_tokens, "d_in": w.d_in,
             "path": w.path.name}
            for (s, b), w in sorted(writers.items())
        ],
    }
    _atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest["shards"], indent=2))


if __name__ == "__main__":
    main()
