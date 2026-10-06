#!/usr/bin/env python
"""Stage 4 -- causal intervention. The stage that decides whether stage 3 found
features or correlates.

    python scripts/04_steer.py --sae runs/<shard>/sae.pt --locate outputs/locate.json \
        --site <site> --act-scale <scale> --mode amplify
    python scripts/04_steer.py ... --mode erase

Two experiments:

  amplify -- on *neutral* prompts, ADD the feature's direction at every position
             (``add`` mode), at multiples of the feature's typical active code.
             A causal feature makes CLIP similarity to the concept phrase rise
             while similarity to the original prompt holds.

  erase   -- on held-out concept prompts (EVAL, never used to find the feature),
             multiply the feature's code by gamma wherever it is active
             (gamma = 1 is the baseline, 0 removes it, negative pushes against
             it -- the same family of edit SAeUron uses). Then the same edit on
             HOLDOUT prompts: near neighbours that must survive.

WHAT RUN 1 GOT WRONG, AND WHAT CHANGED

1. ``--act-scale`` was accepted and never applied, so the SAE encoded raw
   activations at the wrong scale. SteeringHook now normalises on the way in.
2. Amplify used ``clamp``, which only touches positions where the feature is
   already active -- on neutral prompts almost none. It now uses ``add``.
3. The holdout delta compared holdout images against the *target* prompts'
   baseline: two different prompt sets, so the number was meaningless. Holdout
   now has its own unedited baseline.
4. Erase was evaluated on the prompts used to select the feature. It now uses
   EVAL prompts, disjoint from PROBE and HOLDOUT.
5. One seed per setting. Effects of ~0.001 CLIP are below seed noise, so every
   setting now runs ``--seeds`` times and each delta is a PAIRED difference
   (edited minus unedited, same seed, same prompts), reported as mean and std.

Scoring uses a CLIP checkpoint that is not the one inside SD1.5. Scoring with
SD's own text encoder measures how well you satisfied the model's training
objective, which is circular.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.model import SparseAutoencoder
from sae.hooks import SteeringHook, SteerSpec
from sae.metrics import ClipScorer
from sae.prompts import DISCOVERY, EVAL, HOLDOUT

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


def mean_std(xs):
    return (statistics.fmean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


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
    ap.add_argument("--strengths", type=float, nargs="*", default=None,
                    help="amplify: multiples of the feature's typical active code "
                         "(default 1 2 4 8); erase: gamma (default 0 -1 -3)")
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-grids", default="outputs/grids")
    args = ap.parse_args()

    from sae.loader import load_pipeline, pick_device

    device = pick_device(args.model)
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
    strengths = args.strengths or ([1.0, 2.0, 4.0, 8.0] if args.mode == "amplify"
                                   else [0.0, -1.0, -3.0])
    seeds = [args.seed + 1000 * i for i in range(args.seeds)]

    grid_dir = Path(args.save_grids); grid_dir.mkdir(parents=True, exist_ok=True)

    def run(prompts, specs, seed, phrase):
        hook = SteeringHook(sae, specs, preserve_error=True, act_scale=args.act_scale)
        hook.attach(pipe.unet, args.site)
        try:
            imgs = generate(pipe, prompts, seed=seed)
        finally:
            hook.detach()
        return (imgs,
                scorer.similarity(imgs, [phrase] * len(imgs)).mean().item(),
                scorer.similarity(imgs, prompts).mean().item())

    report, rows = {}, []
    for concept, feats in cand.items():
        top = feats[0]
        feature = top["feature"]
        phrase = CONCEPT_PHRASE.get(concept, concept.replace("_", " "))
        # Typical size of this feature's code when it fires, in normalised units:
        # mean over concept tokens divided by how often it fires.
        act_ref = top["mean_act_concept"] / max(top["fire_rate_concept"], 1e-6)

        if args.mode == "amplify":
            targets = DISCOVERY[:args.n_prompts]
            holdout = None
            make = lambda s: [SteerSpec(feature=feature, mode="add", value=s * act_ref)]
        else:
            ev = EVAL[concept]
            targets = (ev * ((args.n_prompts // len(ev)) + 1))[:args.n_prompts]
            holdout = HOLDOUT[concept]
            make = lambda g: [SteerSpec(feature=feature, mode="scale", value=g)]

        # Unedited baselines, one per seed, for targets and (erase) holdout.
        base_t, base_h = {}, {}
        for sd in seeds:
            imgs, c, p = run(targets, [], sd, phrase)
            base_t[sd] = (c, p)
            if sd == seeds[0]:
                imgs[0].save(grid_dir / f"{concept}_f{feature}_{args.mode}_base.png")
            if holdout:
                _, hc, hp = run(holdout, [], sd, phrase)
                base_h[sd] = (hc, hp)

        per_strength = {}
        for s in strengths:
            d_concept, d_prompt, d_hold = [], [], []
            for sd in seeds:
                imgs, c, p = run(targets, make(s), sd, phrase)
                d_concept.append(c - base_t[sd][0])
                d_prompt.append(p - base_t[sd][1])
                if sd == seeds[0]:
                    imgs[0].save(grid_dir / f"{concept}_f{feature}_{args.mode}_{s:g}.png")
                if holdout:
                    _, hc, hp = run(holdout, make(s), sd, phrase)
                    d_hold.append(hp - base_h[sd][1])
            mc, sc = mean_std(d_concept)
            mp, sp = mean_std(d_prompt)
            entry = {"strength": s, "concept_delta": mc, "concept_delta_std": sc,
                     "prompt_delta": mp, "prompt_delta_std": sp}
            if d_hold:
                mh, sh = mean_std(d_hold)
                entry.update(holdout_prompt_delta=mh, holdout_prompt_delta_std=sh)
            per_strength[s] = entry
            rows.append({"concept": concept, "feature": feature, **entry})
            print(f"{concept:<12} f{feature:<6} s={s:<5g} concept {mc:+.4f}±{sc:.4f}  "
                  f"prompt {mp:+.4f}±{sp:.4f}"
                  + (f"  holdout {entry['holdout_prompt_delta']:+.4f}"
                     f"±{entry['holdout_prompt_delta_std']:.4f}" if d_hold else ""),
                  flush=True)

        # "Best" = largest mean effect in the intended direction. Choosing the
        # best of several strengths on the same data inflates it a little; the
        # std and the per-strength rows are reported so that can be judged.
        sign = 1 if args.mode == "amplify" else -1
        best = max(per_strength.values(), key=lambda e: sign * e["concept_delta"])
        report[concept] = {"feature": feature, "act_ref": act_ref,
                           "best_strength": best["strength"], **{k: v for k, v in best.items()
                                                                 if k != "strength"},
                           "n_seeds": len(seeds), "n_prompts": len(targets)}

    out = Path(args.out or f"outputs/steer_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "report": report,
                               "strengths": strengths, "seeds": seeds}, indent=2))
    print(f"\nwrote {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
