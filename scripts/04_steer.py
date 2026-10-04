#!/usr/bin/env python
"""Stage 4 -- causal intervention. The stage that decides whether stage 3 found
features or correlates.

    python scripts/04_steer.py --sae runs/<shard>/sae.pt --locate outputs/locate.json \
        --site <site> --act-scale <scale> --mode amplify
    python scripts/04_steer.py ... --mode erase

Two experiments:

  amplify -- take a *neutral* prompt, amplify the candidate feature, and measure
             whether the concept appears. Uses CLIP similarity to a concept
             phrase as the outcome and CLIP similarity to the original prompt as
             the retention control. A feature is causal if concept similarity
             rises meaningfully while prompt retention holds.

  erase   -- take a prompt that *does* contain the concept, project the feature
             direction out, and measure whether the concept goes away -- then
             run the same intervention on the HOLDOUT prompts, which are near
             neighbours that must survive. Both numbers go in the report. An
             erasure score on its own is not a result.

Everything is scored with a CLIP checkpoint that is not the one inside SD1.5.
Scoring with SD's own text encoder measures how well you satisfied the model's
training objective, which is circular.
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.model import SparseAutoencoder
from sae.hooks import SteeringHook, SteerSpec
from sae.metrics import ClipScorer, erasure_report
from sae.prompts import DISCOVERY, PROBE, HOLDOUT

CONCEPT_PHRASE = {
    "van_gogh": "a painting in the style of Vincent van Gogh",
    "dog": "a dog",
    "fire": "fire and flames",
    "watercolour": "a watercolour painting",
}


def generate(pipe, prompts, seed, steps=25, guidance=7.5, size=512):
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(prompts, num_inference_steps=steps, guidance_scale=guidance,
                height=size, width=size, generator=gen).images


def main():
    # Inference only. Without this, SAE outputs carry autograd graphs: memory is
    # wasted and anything that calls .numpy() on them crashes (stage 5 did).
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--locate", default="outputs/locate.json")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
    ap.add_argument("--site", required=True)
    ap.add_argument("--act-scale", type=float, required=True)
    ap.add_argument("--mode", choices=["amplify", "erase"], default="amplify")
    ap.add_argument("--strengths", type=float, nargs="*",
                    default=[0.0, 2.0, 4.0, 8.0])
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-grids", default="outputs/grids")
    args = ap.parse_args()

    from sae.loader import load_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = load_pipeline(args.model, dtype=dtype, device=device)

    if args.clip == "__fake__":                  # tests only; see sae/fake_pipe.py
        from sae.fake_pipe import FakeClipScorer
        scorer = FakeClipScorer()
    else:
        from transformers import CLIPModel, CLIPProcessor
        scorer = ClipScorer(CLIPModel.from_pretrained(args.clip).to(device).eval(),
                            CLIPProcessor.from_pretrained(args.clip), device)

    sae = SparseAutoencoder.load(args.sae).to(device).eval()
    cand = json.loads(Path(args.locate).read_text())["candidates"]

    grid_dir = Path(args.save_grids); grid_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for concept, feats in cand.items():
        feature = feats[0]["feature"]
        phrase = CONCEPT_PHRASE.get(concept, concept.replace("_", " "))

        if args.mode == "amplify":
            prompts = DISCOVERY[:args.n_prompts]
        else:
            base = PROBE[concept]
            prompts = (base * ((args.n_prompts // len(base)) + 1))[:args.n_prompts]

        for strength in args.strengths:
            if args.mode == "amplify":
                spec = SteerSpec(feature=feature, mode="clamp", value=strength)
            else:
                spec = SteerSpec(feature=feature, mode="project", value=strength)

            hook = SteeringHook(sae, [spec] if strength != 0.0 else [],
                                preserve_error=True)
            hook.attach(pipe.unet, args.site)
            try:
                imgs = generate(pipe, prompts, seed=args.seed)
            finally:
                hook.detach()

            sim_concept = scorer.similarity(imgs, [phrase] * len(imgs)).mean().item()
            sim_prompt = scorer.similarity(imgs, prompts).mean().item()

            rows.append({"concept": concept, "feature": feature, "mode": args.mode,
                         "strength": strength, "clip_concept": sim_concept,
                         "clip_prompt": sim_prompt, "set": "target"})
            print(f"{concept:<12} f{feature:<6} s={strength:<5} "
                  f"concept {sim_concept:.4f}  prompt {sim_prompt:.4f}", flush=True)

            imgs[0].save(grid_dir / f"{concept}_f{feature}_{args.mode}_{strength}.png")

            # Collateral: same intervention, near-neighbour prompts that must survive.
            if args.mode == "erase" and strength != 0.0:
                hold = HOLDOUT[concept]
                hook = SteeringHook(sae, [spec], preserve_error=True)
                hook.attach(pipe.unet, args.site)
                try:
                    h_imgs = generate(pipe, hold, seed=args.seed)
                finally:
                    hook.detach()
                rows.append({
                    "concept": concept, "feature": feature, "mode": args.mode,
                    "strength": strength, "set": "holdout",
                    "clip_concept": scorer.similarity(h_imgs, [phrase] * len(hold)).mean().item(),
                    "clip_prompt": scorer.similarity(h_imgs, hold).mean().item(),
                })

    out = Path(args.out or f"outputs/steer_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pair the two numbers. Baseline is strength 0.
    report = {}
    for concept in cand:
        tgt = [r for r in rows if r["concept"] == concept and r["set"] == "target"]
        base = next((r for r in tgt if r["strength"] == 0.0), None)
        if base is None:
            continue
        best = max(tgt, key=lambda r: (base["clip_concept"] - r["clip_concept"])
                   if args.mode == "erase" else (r["clip_concept"] - base["clip_concept"]))
        hold = [r for r in rows if r["concept"] == concept and r["set"] == "holdout"
                and r["strength"] == best["strength"]]
        report[concept] = {
            "feature": best["feature"],
            "best_strength": best["strength"],
            "concept_delta": best["clip_concept"] - base["clip_concept"],
            "prompt_retention_delta": best["clip_prompt"] - base["clip_prompt"],
            "holdout_prompt_delta": (hold[0]["clip_prompt"] - base["clip_prompt"]) if hold else None,
        }

    out.write_text(json.dumps({"rows": rows, "report": report}, indent=2))
    print(f"\nwrote {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
