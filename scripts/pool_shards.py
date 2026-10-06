#!/usr/bin/env python
"""Define a pooled (all-timestep) shard for one site, without copying data.

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

Two details:

* The pooled shard is a manifest pointing at the bucket shards, not a copy.
  Copying would duplicate ~8 GB and blow Kaggle's 20 GB working directory.
* Each bucket contributes the same number of tokens (the smallest bucket's
  count), so the pooled SAE is not tilted toward whichever bucket the scheduler
  happened to visit more often.

A fairness caveat to state when you report the table: by default the pooled SAE
sees 3x the tokens of any single bucketed SAE, but has to spread the same number
of features across three regimes. That is the real trade-off being tested, not
a confound -- but say it explicitly.
"""

import argparse
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.data import shard_slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/acts")
    ap.add_argument("--site", required=True)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--matched", action="store_true",
                    help="size the pooled shard like ONE bucket (each bucket "
                         "contributes a third), written as bucket -2 / 'pooled_matched'. "
                         "Separates 'pooling helps' from 'more data helps': run 1's "
                         "pooled SAE saw 3x the tokens of each bucketed SAE")
    args = ap.parse_args()

    acts = Path(args.acts)
    metas = []
    for b in range(10):
        mp = acts / f"{shard_slug(args.site, b)}.json"
        if mp.exists():
            metas.append(json.loads(mp.read_text()))
    if len(metas) < 2:
        raise SystemExit(f"need >=2 bucket shards for {args.site} in {acts}, found {len(metas)}")

    d_in = metas[0]["d_in"]
    assert all(m["d_in"] == d_in for m in metas)
    n_take = min(m["n_tokens"] for m in metas)
    if args.matched:
        n_take //= len(metas)

    sources = []
    for m in metas:
        take = m["n_tokens"] if args.no_balance else n_take
        # A bucket shard is either a real file or (after a multi-GPU merge) a
        # manifest of files; expand to files and draw `take` tokens greedily.
        srcs = m.get("sources") or [{"path": f"{shard_slug(args.site, m['bucket'])}.bin",
                                     "n_take": m["n_tokens"]}]
        remaining = take
        for src in srcs:
            if remaining <= 0:
                break
            n = min(remaining, int(src["n_take"]))
            sources.append({"path": src["path"], "bucket": m["bucket"], "n_take": n})
            remaining -= n
        print(f"  bucket {m['bucket']}: {m['n_tokens']:,} available, {take:,} used "
              f"({len(srcs)} file(s))")

    tag, bucket_id = ("pooled_matched", -2) if args.matched else ("pooled", -1)
    out = acts / f"{shard_slug(args.site, tag)}.json"
    manifest = {"site": args.site, "bucket": bucket_id, "d_in": d_in,
                "n_tokens": sum(s["n_take"] for s in sources),
                "dtype": "float16", "sources": sources}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out)
    print(f"wrote {out} ({manifest['n_tokens']:,} tokens, no data copied)")


if __name__ == "__main__":
    main()
