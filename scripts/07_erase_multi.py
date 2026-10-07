#!/usr/bin/env python
"""Stage 7 -- multi-feature erasure: how many SAE features does it take to
remove a concept, and what does it cost?

    python scripts/07_erase_multi.py --sae runs/<shard>/sae.pt --site <site> \
        --act-scale <scale> --bucket 1 --arm pooled --out outputs/erase_multi_pooled.json

WHY THIS EXISTS

Run 2 showed that scaling ONE feature to zero changes nothing measurable
(every concept CLIP delta within noise), while ADDING one feature's direction
clearly injects the concept. The working explanation is redundancy: the concept
is spread over many features, the SAE's error term passes through untouched,
and cross-attention in every later block re-reads the concept from the prompt.
If that is right, erasure should start to bite as more features are removed.
This stage measures exactly that dose-response.

SETTINGS (per concept, all at every denoising step)

    top5   / top10 / top20   gamma 0   the N most concept-selective features (by AUC)
    top20_neg                gamma -1  same 20, pushed past zero
    random20                 gamma 0   20 features that ARE active on concept prompts
                                       but are not among the top 20 -- the control

The control answers "does deleting ANY 20 active features hurt the concept?"
If random20 drops concept CLIP as much as top20, the effect is generic damage,
not concept removal. Each setting also records how much it changed the
activations (relative L2 norm), so a big effect from a tiny edit -- or a tiny
effect from a big one -- is visible rather than hidden.

Scoring: CLIP similarity to the concept phrase on EVAL prompts (never used to
select features), CLIP similarity to the prompt itself, and the same prompt
similarity on HOLDOUT near-neighbour prompts (collateral damage), each against
its own unedited baseline, paired by seed, mean +/- std over seeds.

Feature lists come from 03_locate.py run with --top 200 on this arm's SAE. The
locate file is created on first use and reused afterwards, so a crash after
locating does not repeat it.
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.model import SparseAutoencoder
from sae.hooks import SteeringHook, SteerSpec
from sae.metrics import ClipScorer
from sae.prompts import EVAL, HOLDOUT

CONCEPT_PHRASE = {
    "van_gogh": "a painting in the style of Vincent van Gogh",
    "dog": "a dog",
    "fire": "fire and flames",
    "watercolour": "a watercolour painting",
}


class MeasuringHook(SteeringHook):
    """SteeringHook that also records how large its edit was: the mean over
    calls of ||x_new - x|| / ||x|| on the edited (conditional) half."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rel = []

    def _edit(self, x):
        out = super()._edit(x)
        num = (out.float() - x.float()).norm()
        den = x.float().norm().clamp_min(1e-8)
        self.rel.append(float(num / den))
        return out


def mean_std(xs):
    return (statistics.fmean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def strip(rows, size=192, pad=4):
    """rows: list of lists of PIL images -> one grid (rows = settings)."""
    from PIL import Image
    rows = [r for r in rows if r]
    if not rows:
        return None
    ncol = max(len(r) for r in rows)
    sheet = Image.new("RGB", (ncol * size + (ncol + 1) * pad,
                              len(rows) * size + (len(rows) + 1) * pad), (18, 18, 20))
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            sheet.paste(im.convert("RGB").resize((size, size)),
                        (pad + j * (size + pad), pad + i * (size + pad)))
    return sheet


def ensure_locate(args) -> dict:
    path = Path(args.locate)
    if not path.exists():
        cmd = [sys.executable, str(Path(__file__).with_name("03_locate.py")),
               "--model", args.model, "--sae", args.sae, "--site", args.site,
               "--bucket", str(args.bucket), "--act-scale", str(args.act_scale),
               "--repeats", str(args.locate_repeats), "--top", "200",
               "--out", str(path)]
        print("locating features for this arm:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return json.loads(path.read_text())["candidates"]


def main():
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--arm", required=True, help="label, e.g. bucketed_b1 or pooled")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
    ap.add_argument("--site", required=True)
    ap.add_argument("--bucket", type=int, default=1,
                    help="timestep bucket whose activations are used to LOCATE features")
    ap.add_argument("--act-scale", type=float, required=True)
    ap.add_argument("--locate", required=True, help="cached top-200 locate file for this arm")
    ap.add_argument("--locate-repeats", type=int, default=4)
    ap.add_argument("--ns", type=int, nargs="*", default=[5, 10, 20])
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grids", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"{out_path} exists -- skipping (delete it to rerun)")
        return

    from sae.loader import load_pipeline, pick_device
    device = pick_device(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32

    cand = ensure_locate(args)          # before loading anything big a second time

    pipe = load_pipeline(args.model, dtype=dtype, device=device)
    if args.clip == "__fake__":
        from sae.fake_pipe import FakeClipScorer
        scorer = FakeClipScorer()
    else:
        from transformers import CLIPModel, CLIPProcessor
        scorer = ClipScorer(CLIPModel.from_pretrained(args.clip).to(device).eval(),
                            CLIPProcessor.from_pretrained(args.clip), device)
    sae = SparseAutoencoder.load(args.sae).to(device).eval()

    seeds = [args.seed + 1000 * i for i in range(args.seeds)]
    nmax = max(args.ns)
    grid_dir = Path(args.grids); grid_dir.mkdir(parents=True, exist_ok=True)

    def gen(prompts, specs, seed):
        hook = MeasuringHook(sae, specs, preserve_error=True, act_scale=args.act_scale)
        hook.attach(pipe.unet, args.site)
        try:
            g = torch.Generator(device=pipe.device).manual_seed(seed)
            imgs = pipe(prompts, num_inference_steps=args.steps, guidance_scale=7.5,
                        height=512, width=512, generator=g).images
        finally:
            hook.detach()
        rel = statistics.fmean(hook.rel) if hook.rel else 0.0
        return imgs, rel

    def score(imgs, prompts, phrase):
        return (scorer.similarity(imgs, [phrase] * len(imgs)).mean().item(),
                scorer.similarity(imgs, prompts).mean().item())

    report, rows = {}, []
    for concept, feats in cand.items():
        if concept not in EVAL:
            continue
        ranked = [f for f in feats if f["auc"] > 0]
        top = [f["feature"] for f in ranked[:nmax]]
        if len(top) < nmax:
            print(f"{concept}: only {len(top)} usable features; using what there is")
        pool = [f["feature"] for f in ranked[nmax:]]
        rng = random.Random(f"{concept}-{args.seed}")      # deterministic across runs
        pool_kind = "active"
        if len(pool) < nmax:
            # Too few other features fire on this concept (only happens on tiny
            # dictionaries): pad with features from the rest of the dictionary
            # and say so in the output.
            rest = [f for f in range(sae.cfg.d_hidden) if f not in set(top) | set(pool)]
            pool = pool + rng.sample(rest, min(nmax - len(pool), len(rest)))
            pool_kind = "padded"
        rand = rng.sample(pool, min(nmax, len(pool)))

        settings = [(f"top{n}", top[:n], 0.0) for n in args.ns]
        settings.append((f"top{nmax}_neg", top, -1.0))
        settings.append((f"random{len(rand)}", rand, 0.0))

        phrase = CONCEPT_PHRASE.get(concept, concept.replace("_", " "))
        ev = EVAL[concept]
        targets = (ev * ((args.n_prompts // len(ev)) + 1))[:args.n_prompts]
        holdout = HOLDOUT[concept]

        base = {}
        for sd in seeds:
            ti, _ = gen(targets, [], sd)
            hi, _ = gen(holdout, [], sd)
            base[sd] = (score(ti, targets, phrase), score(hi, holdout, phrase), ti, hi)

        grid_rows = {"base": base[seeds[0]][2][:4]}
        hold_rows = {"base": base[seeds[0]][3][:4]}
        per = {}
        for label, fl, gamma in settings:
            specs = [SteerSpec(feature=f, mode="scale", value=gamma) for f in fl]
            dc, dp, dh, rels = [], [], [], []
            for sd in seeds:
                ti, rel_t = gen(targets, specs, sd)
                hi, _ = gen(holdout, specs, sd)
                (c0, p0), (_, hp0) = base[sd][0], base[sd][1]
                c, p = score(ti, targets, phrase)
                _, hp = score(hi, holdout, phrase)
                dc.append(c - c0); dp.append(p - p0); dh.append(hp - hp0); rels.append(rel_t)
                if sd == seeds[0]:
                    grid_rows[label] = ti[:4]
                    hold_rows[label] = hi[:4]
            e = {"setting": label, "n_features": len(fl), "gamma": gamma}
            for k, xs in (("concept_delta", dc), ("prompt_delta", dp),
                          ("holdout_prompt_delta", dh), ("edit_rel_norm", rels)):
                m, s = mean_std(xs)
                e[k], e[k + "_std"] = m, s
            per[label] = e
            rows.append({"arm": args.arm, "concept": concept, **e})
            print(f"[{args.arm}] {concept:<12} {label:<10} concept {e['concept_delta']:+.4f}"
                  f"±{e['concept_delta_std']:.4f}  prompt {e['prompt_delta']:+.4f}  "
                  f"holdout {e['holdout_prompt_delta']:+.4f}±{e['holdout_prompt_delta_std']:.4f}"
                  f"  edit {100 * e['edit_rel_norm']:.1f}%", flush=True)

        order = ["base"] + [s[0] for s in settings]
        for name, rws in (("eval", grid_rows), ("holdout", hold_rows)):
            sheet = strip([rws[k] for k in order if k in rws])
            if sheet is not None:
                sheet.save(grid_dir / f"{concept}_{name}.png")
        (grid_dir / "rows.txt").write_text(
            "rows, top to bottom: " + ", ".join(order) +
            "\ncolumns: the first 4 prompts, seed " + str(seeds[0]) + "\n")
        report[concept] = {"features_top": top, "features_random": rand,
                           "random_pool": pool_kind, "settings": per}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "arm": args.arm, "sae": args.sae, "site": args.site, "seeds": seeds,
        "ns": args.ns, "report": report, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
