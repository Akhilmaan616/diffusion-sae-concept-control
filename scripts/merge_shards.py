#!/usr/bin/env python
"""Combine captures made in parallel on several GPUs, without copying data.

    python scripts/merge_shards.py --inputs data/acts_gpu0 data/acts_gpu1 --out data/acts

Kaggle gives you two T4s; the capture script uses one GPU per process, so the
pipeline runs two captures side by side (different seeds, so different images)
and merges them here. Each merged shard is a small JSON manifest whose
``sources`` point at the per-GPU binary files -- ActivationStore reads them as
one shard. A physical copy would briefly need twice the disk (~16 GB for one
site), which does not fit in Kaggle's 20 GB working directory.
"""

import argparse
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict] = {}
    for d in args.inputs:
        d = Path(d)
        for mp in sorted(d.glob("*__b[0-9].json")):
            meta = json.loads(mp.read_text())
            rel = os.path.relpath(mp.with_suffix(".bin"), out)
            entry = merged.setdefault(mp.stem, {
                "site": meta["site"], "bucket": meta["bucket"], "d_in": meta["d_in"],
                "n_tokens": 0, "dtype": "float16", "sources": []})
            assert entry["d_in"] == meta["d_in"], f"width mismatch in {mp}"
            entry["sources"].append({"path": rel, "n_take": meta["n_tokens"]})
            entry["n_tokens"] += meta["n_tokens"]

    if not merged:
        raise SystemExit(f"no bucket shards found in {args.inputs}")

    for slug, entry in merged.items():
        tmp = out / f"{slug}.json.tmp"
        tmp.write_text(json.dumps(entry, indent=2))
        os.replace(tmp, out / f"{slug}.json")
        print(f"  {slug}: {entry['n_tokens']:,} tokens from {len(entry['sources'])} source(s)")

    (out / "manifest.json").write_text(json.dumps(
        {"merged_from": args.inputs,
         "shards": [{k: v for k, v in e.items() if k != "sources"} for e in merged.values()]},
        indent=2))
    print(f"merged {len(merged)} shards into {out} (no data copied)")


if __name__ == "__main__":
    main()
