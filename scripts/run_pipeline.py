#!/usr/bin/env python
"""Run the whole project end to end, unattended, and resume after a crash.

    python scripts/run_pipeline.py --root /kaggle/working/project_b

Designed for Kaggle's "Save Version -> Save & Run All", which executes the
notebook as a background job with the browser closed (no idle disconnects, up
to 12 hours). It also works in a Colab or Lightning terminal under ``nohup``.

WHAT IT DOES

    prefetch    download SD1.5 and the scoring CLIP once, before any parallel work
    capture     one capture process per GPU (two on Kaggle's T4 x2), different seeds
    merge       join the per-GPU captures into one set of shards (no copying)
    pool        define the all-timestep pooled shard (no copying)
    train       bucket SAEs + pooled SAE, split across GPUs, fp16 mixed precision
    locate      rank candidate concept features (bucket-1 SAE)
    atlas       spatial feature atlas
    substitute  loss_recovered for pooled vs bucketed SAEs in every bucket
    amplify     causal test: clamp candidate features on neutral prompts
    erase       causal test: project features out; score near-neighbour damage
    report      collect every number into outputs/REPORT.md

RESUMING

Each stage writes ``state/<stage>.done`` when it finishes and is skipped on the
next run. Capture and training are additionally resumable *inside* the stage
(per-batch checkpoints; per-SAE skip), so a crash in the middle of a 40-minute
capture costs one batch, not 40 minutes. Re-run the same command to continue.

TIME (2 x T4, defaults): capture ~35-45 min, training ~20-30 min, everything
else ~45-60 min. Roughly 2 hours of the 12-hour background limit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
ATTN2_SITE = "up_blocks.1.attentions.1.transformer_blocks.0.attn2"   # run 1
RESIDUAL_SITE = "up_blocks.1.attentions.1.transformer_blocks.0"       # run 2 default
STAGES = ["prefetch", "capture", "merge", "pool", "train", "locate", "atlas",
          "substitute", "amplify", "erase", "erase_multi", "report"]


# --------------------------------------------------------------------- utils

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pump(proc, prefix, fh):
    for line in iter(proc.stdout.readline, ""):
        fh.write(line); fh.flush()
        print(f"{prefix}{line}", end="", flush=True)


def run(cmds: list[tuple[list[str], dict, str]], log_dir: Path, name: str) -> None:
    """Run one or more commands in parallel, streaming prefixed output.

    ``cmds`` is a list of (argv, extra_env, prefix). Raises if any fails, after
    all have finished -- so one GPU failing does not orphan the other.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    procs, threads, files = [], [], []
    for i, (argv, env_extra, prefix) in enumerate(cmds):
        env = {**os.environ, **env_extra, "PYTHONUNBUFFERED": "1"}
        fh = open(log_dir / f"{name}{'' if len(cmds) == 1 else f'_{i}'}.log", "a")
        files.append(fh)
        fh.write(f"\n$ {' '.join(argv)}\n")
        p = subprocess.Popen(argv, cwd=REPO, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=_pump, args=(p, prefix, fh), daemon=True)
        t.start()
        procs.append(p); threads.append(t)
    codes = [p.wait() for p in procs]
    for t in threads:
        t.join()
    for fh in files:
        fh.close()
    if any(codes):
        raise RuntimeError(f"stage '{name}' failed (exit codes {codes}); see {log_dir}")


def gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def gpu_env(i: int, n: int) -> dict:
    return {"CUDA_VISIBLE_DEVICES": str(i)} if n > 0 else {}


def read_json(p: Path):
    return json.loads(p.read_text())


# -------------------------------------------------------------------- stages

def st_prefetch(a, P):
    """Download once, up front, so two capture processes don't race on the cache.

    Uses diffusers' own file selection: a blanket ``*.safetensors`` snapshot of
    the SD1.5 repo would also pull its multi-GB single-file checkpoints, which
    the pipeline never uses.
    """
    if a.model.startswith("__fake__") or Path(a.model).exists():
        log("prefetch: nothing to download")
        return
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download
    log(f"downloading {a.model}")
    DiffusionPipeline.download(a.model)
    log(f"downloading {a.clip}")
    snapshot_download(a.clip, allow_patterns=["*.json", "*.txt", "*.safetensors"])


def st_capture(a, P):
    import shutil
    # 256 tokens per image per pass at a 16x16 site, of which tokens_per_pass are
    # kept per batch call (the conditional half of a batch holds batch x 256).
    keep = min(1.0, a.tokens_per_pass / (a.capture_batch * 256))
    est = (a.n_images * 256 * keep * a.diffusion_steps * 1280 * 2
           * (2.2 if a.all_sites else 1.0))
    P["acts"].mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(P["acts"]).free
    log(f"capture needs ~{est / 1e9:.1f} GB; {free / 1e9:.1f} GB free at {P['acts']}")
    if est > free * 0.95:
        raise SystemExit(f"not enough disk for {a.n_images} images -- lower --n-images "
                         f"(each 100 images is ~{est / a.n_images * 100 / 1e9:.1f} GB)")
    n = max(1, min(a.gpus, 2))
    per = [a.n_images // n + (1 if i < a.n_images % n else 0) for i in range(n)]
    cmds = []
    for i in range(n):
        argv = [PY, "scripts/01_capture.py", "--model", a.model,
                "--out", str(P["acts"] / f"gpu{i}"), "--n-images", str(per[i]),
                "--steps", str(a.diffusion_steps), "--batch", str(a.capture_batch),
                "--tokens-per-pass", str(a.tokens_per_pass),
                "--corpus", str(a.corpus), "--seed", str(i)]
        argv += ["--all-sites"] if a.all_sites else ["--sites", a.site]
        cmds.append((argv, gpu_env(i, a.gpus), f"[cap{i}] "))
    run(cmds, P["logs"], "capture")
    (P["state"] / "capture_info.json").write_text(json.dumps(
        {"n_images": a.n_images, "site": a.site, "tokens_per_pass": a.tokens_per_pass,
         "corpus": a.corpus, "capture_batch": a.capture_batch}, indent=2))


def st_merge(a, P):
    inputs = sorted(str(d) for d in P["acts"].glob("gpu*") if d.is_dir())
    run([([PY, "scripts/merge_shards.py", "--inputs", *inputs,
           "--out", str(P["merged"])], {}, "")], P["logs"], "merge")


def st_pool(a, P):
    for extra in ([], ["--matched"]):
        run([([PY, "scripts/pool_shards.py", "--acts", str(P["merged"]),
               "--site", a.site, *extra], {}, "")], P["logs"], "pool")


def st_train(a, P):
    common = ["--acts", str(P["merged"]), "--out", str(P["runs"]),
              "--expansion", str(a.expansion), "--k", str(a.k),
              "--steps", str(a.train_steps), "--batch-size", str(a.train_batch),
              "--device", "cuda" if a.gpus else "cpu"]
    if a.no_amp:
        common.append("--no-amp")
    # Main-site SAEs first: these are the ones every later stage depends on.
    if a.gpus >= 2:
        split = [["0", "1", "pooled_matched"], ["2", "pooled"]]
        run([([PY, "scripts/02_train.py", *common, "--site", a.site,
               "--buckets", *b], gpu_env(i, a.gpus), f"[train{i}] ")
             for i, b in enumerate(split)], P["logs"], "train")
    else:
        run([([PY, "scripts/02_train.py", *common, "--site", a.site], {}, "")],
            P["logs"], "train")
    if a.all_sites:   # secondary sites for the cross-site comparison, if captured
        others = [[PY, "scripts/02_train.py", *common, "--buckets", "0", "1", "2"]]
        run([(others[0], gpu_env(0, a.gpus), "[train-x] ")], P["logs"], "train_other")
    # Rebuild the index from summaries (parallel writers may have raced on it).
    rows = []
    for s in sorted(P["runs"].glob("*/summary.json")):
        d = read_json(s)
        rows.append({"shard": s.parent.name, "site": d["site"],
                     "bucket": {-1: "pooled", -2: "pooled_matched"}.get(int(d["bucket"]),
                                                                       str(d["bucket"])),
                     "n_tokens": d["n_tokens"], "EV": d["final"]["explained_variance"],
                     "L0": d["final"]["l0"], "dead": d["final"]["dead_frac"],
                     "never_fired": d["n_never_fired"], "act_scale": d["act_scale"],
                     "run_dir": str(s.parent)})
    (P["runs"] / "index.json").write_text(json.dumps(rows, indent=2))


def _main_sae(a, P, bucket):
    from_slug = a.site.replace(".", "_") + f"__b{bucket}"
    run_dir = P["runs"] / from_slug
    return run_dir / "sae.pt", read_json(run_dir / "summary.json")["act_scale"]


def st_locate(a, P):
    sae, scale = _main_sae(a, P, a.analysis_bucket)
    run([([PY, "scripts/03_locate.py", "--model", a.model, "--sae", str(sae),
           "--site", a.site, "--bucket", str(a.analysis_bucket),
           "--act-scale", str(scale), "--repeats", str(a.locate_repeats),
           "--out", str(P["out"] / "locate.json")], gpu_env(0, a.gpus), "")],
        P["logs"], "locate")


def st_atlas(a, P):
    sae, scale = _main_sae(a, P, a.analysis_bucket)
    run([([PY, "scripts/05_atlas.py", "--model", a.model, "--sae", str(sae),
           "--site", a.site, "--bucket", str(a.analysis_bucket),
           "--act-scale", str(scale), "--locate", str(P["out"] / "locate.json"),
           "--features", "12", "--top-prompts", "4",
           "--out", str(P["out"] / "atlas")], gpu_env(0, a.gpus), "")],
        P["logs"], "atlas")


def st_substitute(a, P):
    table = {}
    cache = P["out"] / "substitution_latents.pt"
    for b in (0, 1, 2):
        for kind, slug_bucket in (("bucketed", b), ("pooled", "pooled"),
                                  ("pooled_matched", "pooled_matched")):
            sae, scale = _main_sae(a, P, slug_bucket)
            out = P["out"] / f"substitution_{kind}_b{b}.json"
            if not out.exists():
                run([([PY, "scripts/06_substitution.py", "--model", a.model,
                       "--sae", str(sae), "--site", a.site, "--bucket", str(b),
                       "--act-scale", str(scale), "--out", str(out),
                       "--latent-cache", str(cache)], gpu_env(0, a.gpus), "")],
                    P["logs"], "substitute")
            table[f"{kind}_b{b}"] = read_json(out)
    (P["out"] / "substitution_table.json").write_text(json.dumps(table, indent=2))


def st_amplify(a, P):
    sae, scale = _main_sae(a, P, a.analysis_bucket)
    run([([PY, "scripts/04_steer.py", "--model", a.model, "--clip", a.clip,
           "--sae", str(sae), "--site", a.site, "--act-scale", str(scale),
           "--locate", str(P["out"] / "locate.json"), "--mode", "amplify",
           "--n-prompts", str(a.steer_prompts), "--seeds", str(a.steer_seeds),
           "--out", str(P["out"] / "steer_amplify.json"),
           "--save-grids", str(P["out"] / "grids")], gpu_env(0, a.gpus), "")],
        P["logs"], "amplify")


def st_erase(a, P):
    sae, scale = _main_sae(a, P, a.analysis_bucket)
    run([([PY, "scripts/04_steer.py", "--model", a.model, "--clip", a.clip,
           "--sae", str(sae), "--site", a.site, "--act-scale", str(scale),
           "--locate", str(P["out"] / "locate.json"), "--mode", "erase",
           "--n-prompts", str(a.steer_prompts), "--seeds", str(a.steer_seeds),
           "--out", str(P["out"] / "steer_erase.json"),
           "--save-grids", str(P["out"] / "grids")], gpu_env(0, a.gpus), "")],
        P["logs"], "erase")


def st_erase_multi(a, P):
    """Dose-response erasure (top 5/10/20 features + a random-20 control), for
    two SAEs: the bucket-1 SAE used everywhere else, and the pooled SAE, which
    is in-distribution at every timestep the edit touches. With two GPUs the
    arms run side by side. Each arm locates its own top-200 features first."""
    arms = [("bucketed_b1", a.analysis_bucket), ("pooled", "pooled")]
    cmds = []
    for i, (arm, bucket) in enumerate(arms):
        sae, scale = _main_sae(a, P, bucket)
        argv = [PY, "scripts/07_erase_multi.py", "--model", a.model, "--clip", a.clip,
                "--sae", str(sae), "--arm", arm, "--site", a.site,
                "--bucket", str(a.analysis_bucket), "--act-scale", str(scale),
                "--locate", str(P["out"] / f"locate_top200_{arm}.json"),
                "--locate-repeats", str(a.locate_repeats),
                "--n-prompts", str(a.steer_prompts), "--seeds", str(a.steer_seeds),
                "--ns", *[str(n) for n in a.erase_ns],
                "--out", str(P["out"] / f"erase_multi_{arm}.json"),
                "--grids", str(P["out"] / "erase_multi" / arm)]
        cmds.append((argv, gpu_env(i % max(1, min(a.gpus, 2)), a.gpus), f"[{arm}] "))
    if a.gpus >= 2:
        run(cmds, P["logs"], "erase_multi")
    else:
        for c in cmds:
            run([c], P["logs"], "erase_multi")


def st_report(a, P):
    # Describe what was actually run (from the artifacts), not this invocation's flags.
    idx_p = P["runs"] / "index.json"
    idx = read_json(idx_p) if idx_p.exists() else []
    cfg0 = read_json(P["runs"] / idx[0]["shard"] / "summary.json")["sae_cfg"] if idx else {}
    info = P["state"] / "capture_info.json"
    n_img = (read_json(info)["n_images"] if info.exists() else
             sum(read_json(p)["images_done"] for p in P["acts"].glob("gpu*/progress.json")))
    st = P["state"] / "last_run.json"
    run_info = read_json(st) if st.exists() else {}
    lines = ["# Project B — results", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · stages run this session: "
             f"{', '.join(run_info.get('ran') or []) or 'none (all were already done)'}", ""]
    if run_info.get("failed"):
        f = run_info["failed"]
        lines += [f"> **Stage `{f['stage']}` FAILED in this run; later stages did not run.**",
                  f"> `{f['error']}`", f"> Full log: `{f['log']}`", ""]
    lines += [
             f"Model `{a.model}` · site `{a.site}` · expansion {cfg0.get('expansion', '?')}"
             f" · k {cfg0.get('k', '?')} · {n_img or 'n/a'} capture images"
             f" · analysis bucket {a.analysis_bucket}", ""]
    lines += ["## SAE training", "",
              "| shard | bucket | tokens | tokens/feature | EV | L0 | dead | never fired |",
              "|---|---|---|---|---|---|---|---|"]
    for r in idx:
        d_hidden = read_json(P["runs"] / r["shard"] / "summary.json")["sae_cfg"]["d_hidden"]
        lines.append(f"| {r['shard']} | {r['bucket']} | {r['n_tokens']:,} | "
                     f"{r['n_tokens'] / d_hidden:.0f} | {r['EV']:.3f} | {r['L0']:.1f} | "
                     f"{r['dead']:.3f} | {r['never_fired']:,} |")

    t = P["out"] / "substitution_table.json"
    if t.exists():
        tab = read_json(t)
        lines += ["", "## Faithfulness: loss recovered", "",
                  "Each cell substitutes the SAE reconstruction into the UNet at that "
                  "bucket's timesteps only. 1.0 = no loss; 0 = no better than ablating "
                  "the layer. Mean ± std over noise draws. `pooled` saw 3x the data of a "
                  "bucketed SAE; `pooled_matched` saw the same amount -- compare bucketed "
                  "with pooled_matched for the fair test.", "",
                  "| | bucket 0 (texture) | bucket 1 (structure) | bucket 2 (composition) |",
                  "|---|---|---|---|"]
        for kind in ("bucketed", "pooled", "pooled_matched"):
            cells = []
            for b in (0, 1, 2):
                c = tab.get(f"{kind}_b{b}", {})
                v = c.get("loss_recovered")
                cells.append("n/a" if v is None else
                             f"{v:.3f} ± {c.get('loss_recovered_std', 0):.3f}")
            lines.append(f"| {kind} SAE | " + " | ".join(cells) + " |")
        lines += ["", "Raw denoising losses (clean / substituted / ablated) and how much "
                      "ablating the layer costs. A small ablation cost makes the ratio noisy.",
                  "", "| cell | clean | substituted | ablated | ablation cost |",
                  "|---|---|---|---|---|"]
        for k in sorted(tab):
            c = tab[k]
            if "clean" not in c:
                continue
            lines.append(f"| {k} | {c['clean']:.5f} | {c['substituted']:.5f} | "
                         f"{c['ablated']:.5f} | "
                         f"{100 * c.get('ablation_cost_rel', float('nan')):.1f}% |")
        degenerate = [k for k, v in tab.items()
                      if v.get("ablated", 1) - v.get("clean", 0) <= 0]
        if degenerate:
            lines += ["", f"**Warning:** ablation did not raise the loss in {degenerate}; "
                          "loss_recovered is meaningless there."]

    loc = P["out"] / "locate.json"
    if loc.exists():
        lines += ["", "## Candidate concept features (correlational)", "",
                  "| concept | feature | AUC | fires on concept | fires on neutral |",
                  "|---|---|---|---|---|"]
        for c, feats in read_json(loc)["candidates"].items():
            f = feats[0]
            lines.append(f"| {c} | {f['feature']} | {f['auc']:.3f} | "
                         f"{f['fire_rate_concept']:.3f} | {f['fire_rate_neutral']:.3f} |")

    def pm(r, k):
        v = r.get(k)
        return "—" if v is None else f"{v:+.4f} ± {r.get(k + '_std', 0):.4f}"

    for mode in ("amplify", "erase"):
        p = P["out"] / f"steer_{mode}.json"
        if not p.exists():
            continue
        blob = read_json(p)
        rep = blob["report"]
        what = ("ADD the feature direction on neutral prompts; strength = multiple of the "
                "feature's typical active code" if mode == "amplify" else
                "scale the feature's code by gamma on held-out concept prompts (EVAL); "
                "holdout = near-neighbour prompts, each against its own unedited baseline")
        lines += ["", f"## Causal test: {mode}", "",
                  f"{what}. Paired CLIP deltas (edited − unedited, same seed), mean ± std "
                  f"over {len(blob.get('seeds', []))} seeds.", "",
                  "| concept | feature | best strength | concept CLIP Δ | prompt CLIP Δ | "
                  "holdout prompt Δ |", "|---|---|---|---|---|---|"]
        for c, r in rep.items():
            lines.append(f"| {c} | {r['feature']} | {r['best_strength']:g} | "
                         f"{pm(r, 'concept_delta')} | {pm(r, 'prompt_delta')} | "
                         f"{pm(r, 'holdout_prompt_delta')} |")

    em = [(arm, P["out"] / f"erase_multi_{arm}.json") for arm in ("bucketed_b1", "pooled")]
    em = [(arm, read_json(p)) for arm, p in em if p.exists()]
    if em:
        lines += ["", "## Multi-feature erasure (dose-response)", "",
                  "Scale the top-N concept features (by AUC) to gamma at every step, on "
                  "EVAL prompts. `random20` = 20 features active on the concept prompts "
                  "but not in its top 20 -- the control. `edit` = relative change to the "
                  "activations. Paired CLIP deltas, mean ± std over seeds. Success = "
                  "concept Δ clearly negative, below random20, with holdout Δ near zero."]
        for arm, blob in em:
            lines += ["", f"### SAE: {arm}", "",
                      "| concept | setting | concept CLIP Δ | prompt CLIP Δ | "
                      "holdout prompt Δ | edit |", "|---|---|---|---|---|---|"]
            for c, r in blob["report"].items():
                for lab, e in r["settings"].items():
                    lines.append(f"| {c} | {lab} | {pm(e, 'concept_delta')} | "
                                 f"{pm(e, 'prompt_delta')} | {pm(e, 'holdout_prompt_delta')} | "
                                 f"{100 * e['edit_rel_norm']:.1f}% |")
        lines += ["", f"Image grids: `{P['out'] / 'erase_multi'}` (rows: base, then each "
                      "setting; columns: first 4 prompts)."]

    lines += ["", "## Where things are", "",
              f"- SAEs: `{P['runs']}`", f"- atlas sheets: `{P['out'] / 'atlas'}`",
              f"- steering images: `{P['out'] / 'grids'}`", f"- logs: `{P['logs']}`"]
    (P["out"] / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


FUNCS = {name: globals()[f"st_{name}"] for name in STAGES}


def resume_from(src: Path, P, site: str | None = None) -> None:
    """Copy a previous run's progress into this (empty) root.

    SAEs, outputs, logs and stage markers are always copied. The ~8 GB of
    activations are copied only if training had not finished -- every stage
    after training works from the saved SAEs and never touches them.
    """
    import shutil
    if not (src / "state").is_dir():
        log(f"resume-from: {src} has no state/ -- starting fresh")
        return
    # A previous run at a DIFFERENT site is not progress for this one: its
    # markers would skip capture and every later stage. Run 1 predates
    # capture_info.json and used the attn2 site, so a missing file counts as a
    # mismatch unless that is the site requested.
    info = src / "state" / "capture_info.json"
    prev_site = read_json(info)["site"] if info.exists() else ATTN2_SITE
    if site is not None and prev_site != site:
        log(f"resume-from: {src} captured {prev_site}, this run captures {site} "
            "-- ignoring it and starting fresh")
        return
    done = sorted(p.stem for p in (src / "state").glob("*.done"))
    log(f"resuming from {src}: finished stages {done}")
    for sub in ("runs", "outputs", "state", "logs"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, P[sub if sub != "outputs" else "out"],
                            dirs_exist_ok=True)
    if "train" not in done and (src / "data").is_dir():
        log("training had not finished -- copying activations too")
        shutil.copytree(src / "data", P["acts"], dirs_exist_ok=True)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="where runs/, outputs/, state/ live")
    ap.add_argument("--acts-dir", default=None, help="activations (default: <root>/data)")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
    ap.add_argument("--site", default=RESIDUAL_SITE,
                    help="layer to capture; run 1 used the attn2 sub-layer "
                         f"({ATTN2_SITE})")
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--diffusion-steps", type=int, default=25)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--tokens-per-pass", type=int, default=1024)
    ap.add_argument("--corpus", type=int, default=512)
    ap.add_argument("--all-sites", action="store_true",
                    help="also capture/train the other three sites (adds ~3x storage)")
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=15000)
    ap.add_argument("--train-batch", type=int, default=4096)
    ap.add_argument("--analysis-bucket", type=int, default=1)
    ap.add_argument("--locate-repeats", type=int, default=4)
    ap.add_argument("--steer-prompts", type=int, default=6)
    ap.add_argument("--steer-seeds", type=int, default=3)
    ap.add_argument("--erase-ns", type=int, nargs="*", default=[5, 10, 20])
    ap.add_argument("--only", nargs="*", default=None, help="run just these stages")
    ap.add_argument("--redo", nargs="*", default=[], help="force these stages to rerun")
    ap.add_argument("--gpus", type=int, default=None, help="override detected GPU count")
    ap.add_argument("--resume-from", default=None,
                    help="a previous run's root (e.g. an attached Kaggle notebook output); "
                         "copied in if this root has no finished stages yet")
    ap.add_argument("--no-amp", action="store_true",
                    help="train in fp32 (use only if fp16 training produces NaNs)")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    acts = Path(a.acts_dir).resolve() if a.acts_dir else root / "data"
    P = {"root": root, "acts": acts, "merged": acts / "merged", "runs": root / "runs",
         "out": root / "outputs", "state": root / "state", "logs": root / "logs"}
    for k in ("runs", "out", "state", "logs"):
        P[k].mkdir(parents=True, exist_ok=True)
    a.gpus = gpu_count() if a.gpus is None else a.gpus
    log(f"root {root} | acts {acts} | GPUs {a.gpus}")

    if a.resume_from and not any(P["state"].glob("*.done")):
        resume_from(Path(a.resume_from), P, a.site)

    # If training hasn't finished but the activations have vanished (e.g. a Colab
    # VM was recycled while results live on Drive), the capture/merge/pool
    # markers are stale -- clear them so those stages run again.
    if not (P["state"] / "train.done").exists() and not P["merged"].exists():
        for s in ("capture", "merge", "pool"):
            m = P["state"] / f"{s}.done"
            if m.exists():
                log(f"activations missing -- re-running {s}")
                m.unlink()

    # A failed stage stops the stages after it, but the report is still rebuilt
    # so REPORT.md always says what happened in THIS run. Before this, a failure
    # left the previous run's REPORT.md in place, indistinguishable from a fresh one.
    status = P["state"] / "last_run.json"
    ran, skipped, failed = [], [], None
    for name in STAGES:
        if a.only and name not in a.only:
            continue
        if failed and name != "report":
            continue
        marker = P["state"] / f"{name}.done"
        if marker.exists() and name not in a.redo and name != "report":
            log(f"skip {name} (done)")
            skipped.append(name)
            continue
        if name == "report":
            status.write_text(json.dumps({
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "ran": ran,
                "skipped": skipped, "failed": failed}, indent=2))
        log(f"=== {name} ===")
        t0 = time.time()
        try:
            FUNCS[name](a, P)
        except Exception as e:                   # noqa: BLE001 -- reported below
            if name == "report":
                raise
            failed = {"stage": name, "error": f"{type(e).__name__}: {e}"[:2000],
                      "log": str(P["logs"] / f"{name}.log")}
            log(f"STAGE FAILED: {name}: {e}")
            continue
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        ran.append(name)
        log(f"{name} finished in {(time.time() - t0) / 60:.1f} min")

    if failed:
        log(f"pipeline stopped at {failed['stage']} -- REPORT.md says so; "
            f"log: {failed['log']}")
        raise SystemExit(1)
    log("pipeline complete -- see outputs/REPORT.md")


if __name__ == "__main__":
    main()
