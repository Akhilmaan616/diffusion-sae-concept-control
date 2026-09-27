#!/usr/bin/env python
"""Merge a site's per-bucket shards into one pooled shard.

    python scripts/pool_shards.py --acts data/acts \
        --site up_blocks.1.attentions.1.transformer_blocks.0.attn2

This exists for the experiment that makes the project distinct from prior work.

SAeUron (Cywinski & Deja, ICML 2025) trains ONE SAE on SD1.5 cross-attention
activations pooled across all denoising timesteps, and handles timestep
dependence afterwards, at feature-selection time. This repo instead trains one
SAE per timestep bucket. Which is right is an empirical question, and the way to
answer it is a 2 x 3 table:

                    bucket 0     bucket 1     bucket 2
    pooled SAE      LR_p0        LR_p1        LR_p2
    bucketed SAEs   LR_b0        LR_b1        LR_b2

where each cell is loss_recovered from 06_substitution.py, evaluated only at
that bucket's timesteps. If the bucketed row wins clearly, per-timestep
dictionaries are justified. If the rows tie, pooling is the better design (one
SAE, more data per feature) and you should say so -- that is still a result.

Token counts are balanced before pooling (each bucket truncated to the smallest)
so the pooled SAE is not dominated by whichever bucket happened to get more
passes from the scheduler.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/acts")
    ap.add_argument("--site", required=True)
    ap.add_argument("--no-balance", action="store_true")
    args = ap.parse_args()

    acts = Path(args.acts)
    slug = args.site.replace(".", "_")
    shards = sorted(acts.glob(f"{slug}__b[0-9].bin"))
    if not shards:
        raise SystemExit(f"no bucket shards for {args.site} in {acts}")

    metas = [json.loads(s.with_suffix(".json").read_text()) for s in shards]
    d_in = metas[0]["d_in"]
    assert all(m["d_in"] == d_in for m in metas)

    n_take = min(m["n_tokens"] for m in metas)
    out_bin = acts / f"{slug}__pooled.bin"
    total = 0
    with open(out_bin, "wb") as f:
        for s, m in zip(shards, metas):
            arr = np.memmap(s, dtype=np.float16, mode="r").reshape(-1, d_in)
            take = m["n_tokens"] if args.no_balance else n_take
            f.write(np.asarray(arr[:take]).tobytes())
            total += take
            print(f"  bucket {m['bucket']}: {m['n_tokens']:,} available, {take:,} used")

    (acts / f"{slug}__pooled.json").write_text(json.dumps(
        {"site": args.site, "bucket": -1, "d_in": d_in, "n_tokens": total,
         "dtype": "float16"}, indent=2))
    print(f"wrote {out_bin} ({total:,} tokens)")


if __name__ == "__main__":
    main()
