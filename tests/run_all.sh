#!/usr/bin/env bash
# Run the suite twice: as-is, and with PyTorch told a GPU is visible.
#
# Why the second pass: Kaggle has GPUs, the build machine does not, and two
# Kaggle-only failures came from code that sent tensors to "cuda" while the
# test-only fake model ran on CPU. On a CPU-only machine "pretend GPU" turns any
# such leak into an immediate error, so it is caught here instead of on Kaggle.
set -e
cd "$(dirname "$0")/.."
for mode in normal pretend-gpu; do
  echo "== $mode =="
  for t in tests/test_core.py tests/test_stages.py tests/test_pipeline.py; do
    if [ "$mode" = pretend-gpu ]; then
      PYTHONPATH=tests/pretend_gpu python3 "$t" | tail -1 | sed "s|^|$t: |"
    else
      python3 "$t" | tail -1 | sed "s|^|$t: |"
    fi
  done
done
