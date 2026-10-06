#!/usr/bin/env python
"""Stage 2 -- train one SAE per (site, timestep bucket), plus pooled if defined.

    python scripts/02_train.py --acts data/acts --out runs --expansion 8 --steps 15000
    python scripts/02_train.py ... --site up_blocks.1 --buckets 0 1      # a subset
    python scripts/02_train.py ... --buckets pooled                       # just the pooled SAE

Runs whose ``sae.pt`` and ``summary.json`` already exist are skipped, so the
command is safe to re-run after a crash; ``--force`` retrains.

The cross-bucket comparison is a result in itself: if the early-timestep SAE and
the late-timestep SAE have similar explained variance but their atlases look
completely different, that is direct evidence that a diffusion UNet's activation
space means different things at different noise levels.
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.data import ActivationStore, find_shards
from sae.model import SAEConfig
from sae.train import TrainConfig, train_sae


def _bucket_label(b: int) -> str:
    return {-1: "pooled", -2: "pooled_matched"}.get(int(b), str(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/acts")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--mode", default="topk", choices=["topk", "l1"])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--l1-coef", type=float, default=5e-4)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--site", default=None,
                    help="exact site name to train (the residual-stream site is a prefix "
                         "of the attn2 site, so a substring match would catch both)")
    ap.add_argument("--buckets", nargs="*", default=None,
                    help="e.g. 0 1 2 pooled pooled_matched (default: all)")
    ap.add_argument("--only-bucket", type=int, default=None, help="(legacy) single bucket")
    ap.add_argument("--buffer-tokens", type=int, default=1 << 19)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    wanted = set(args.buckets) if args.buckets else None
    if args.only_bucket is not None:
        wanted = {_bucket_label(args.only_bucket)}

    shards = find_shards(args.acts)
    if not shards:
        raise SystemExit(f"no shards in {args.acts} -- run 01_capture.py first")

    rows = []
    for meta_path in shards:
        meta = json.loads(meta_path.read_text())
        label = _bucket_label(meta["bucket"])
        if args.site and args.site != meta["site"]:
            continue
        if wanted is not None and label not in wanted:
            continue

        run_dir = Path(args.out) / meta_path.stem
        if (run_dir / "sae.pt").exists() and (run_dir / "summary.json").exists() \
                and not args.force:
            summary = json.loads((run_dir / "summary.json").read_text())
            print(f"skip {meta_path.stem} (already trained)", flush=True)
        else:
            store = ActivationStore(meta_path, buffer_tokens=args.buffer_tokens,
                                    device=args.device)
            print(f"\n=== {store} ===", flush=True)
            summary = train_sae(
                store,
                SAEConfig(d_in=store.d_in, expansion=args.expansion, mode=args.mode,
                          k=args.k, l1_coef=args.l1_coef),
                TrainConfig(steps=args.steps, batch_size=args.batch_size,
                            lr=args.lr, device=args.device, amp=not args.no_amp),
                out_dir=run_dir,
            )

        rows.append({
            "shard": meta_path.stem, "site": meta["site"], "bucket": label,
            "n_tokens": summary["n_tokens"],
            "EV": summary["final"].get("explained_variance"),
            "L0": summary["final"].get("l0"),
            "dead": summary["final"].get("dead_frac"),
            "never_fired": summary["n_never_fired"],
            "act_scale": summary["act_scale"],
            "run_dir": str(run_dir),
        })

    Path(args.out).mkdir(parents=True, exist_ok=True)
    # Merge with any existing index so parallel processes don't clobber each other.
    idx_path = Path(args.out) / "index.json"
    existing = json.loads(idx_path.read_text()) if idx_path.exists() else []
    by_shard = {r["shard"]: r for r in existing}
    by_shard.update({r["shard"]: r for r in rows})
    idx_path.write_text(json.dumps(sorted(by_shard.values(), key=lambda r: r["shard"]),
                                   indent=2))

    print(f"\n{'shard':<58}{'bucket':>7}{'EV':>8}{'L0':>7}{'dead':>8}")
    for r in rows:
        print(f"{r['shard']:<58}{r['bucket']:>7}{r['EV']:>8.3f}"
              f"{r['L0']:>7.1f}{r['dead']:>8.3f}")


if __name__ == "__main__":
    main()
