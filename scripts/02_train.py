#!/usr/bin/env python
"""Stage 2 -- train one SAE per (site, timestep bucket).

    python scripts/02_train.py --acts data/acts --out runs/ --steps 20000

Prints a comparison table at the end. The cross-bucket comparison is a result in
itself: if the early-timestep SAE and the late-timestep SAE have similar
explained variance but their feature atlases look completely different, that is
direct evidence for the claim that a diffusion UNet's activation space means
different things at different noise levels.
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.data import ActivationStore
from sae.model import SAEConfig
from sae.train import TrainConfig, train_sae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/acts")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--mode", default="topk", choices=["topk", "l1"])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--l1-coef", type=float, default=5e-4)
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-bucket", type=int, default=None)
    args = ap.parse_args()

    acts = Path(args.acts)
    shards = sorted(acts.glob("*.bin"))
    if not shards:
        raise SystemExit(f"no shards in {acts} -- run 01_capture.py first")

    rows = []
    for shard in shards:
        meta = json.loads(shard.with_suffix(".json").read_text())
        if args.only_bucket is not None and meta["bucket"] != args.only_bucket:
            continue

        store = ActivationStore(shard, buffer_tokens=1 << 19, device=args.device)
        print(f"\n=== {store} ===", flush=True)

        run_dir = Path(args.out) / shard.stem
        summary = train_sae(
            store,
            SAEConfig(d_in=store.d_in, expansion=args.expansion, mode=args.mode,
                      k=args.k, l1_coef=args.l1_coef),
            TrainConfig(steps=args.steps, batch_size=args.batch_size,
                        lr=args.lr, device=args.device),
            out_dir=run_dir,
        )
        rows.append({
            "shard": shard.stem,
            "site": meta["site"],
            "bucket": meta["bucket"],
            "n_tokens": meta["n_tokens"],
            "EV": summary["final"]["explained_variance"],
            "L0": summary["final"]["l0"],
            "dead": summary["final"]["dead_frac"],
            "never_fired": summary["n_never_fired"],
            "run_dir": str(run_dir),
        })

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "index.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'shard':<52}{'bucket':>7}{'EV':>8}{'L0':>7}{'dead':>8}")
    for r in rows:
        print(f"{r['shard']:<52}{r['bucket']:>7}{r['EV']:>8.3f}"
              f"{r['L0']:>7.1f}{r['dead']:>8.3f}")


if __name__ == "__main__":
    main()
