"""Tests for the unattended pipeline: crash-safe capture, resume, merge, pooling,
skip-if-done training, and the orchestrator itself.

These use ``--model __fake__`` (sae/fake_pipe.py), a tiny stand-in with the same
module path, call pattern, CFG-doubled batch and timestep grid as SD1.5, so the
real scripts run end to end on CPU in seconds.

The headline test is ``test_killed_capture_resumes_bit_identically``: it kills a
capture part-way, corrupts the tail of a shard to mimic a half-written batch,
resumes, and requires the final shards to be byte-for-byte identical to a run
that was never interrupted. That is the property that makes a free GPU session
dying mid-capture cost one batch instead of the whole run.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sae.data import ActivationStore, ShardWriter, find_shards  # noqa: E402

SITE = "up_blocks.1.attentions.1.transformer_blocks.0.attn2"
SLUG = SITE.replace(".", "_")


def _run(args, env_extra=None, check=True):
    env = {**os.environ, **(env_extra or {})}
    env.pop("SAE_FAKE_CRASH_AFTER", None) if not env_extra else None
    r = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"{args[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def _capture(out, n=24, seed=0, extra_env=None, check=True, fresh=False):
    args = ["scripts/01_capture.py", "--model", "__fake__", "--out", str(out),
            "--n-images", str(n), "--batch", "4", "--tokens-per-pass", "1024",
            "--corpus", "64", "--seed", str(seed), "--sites", SITE]
    if fresh:
        args.append("--fresh")
    return _run(args, extra_env, check)


def _bytes(d):
    return {p.name: p.read_bytes() for p in sorted(Path(d).glob("*.bin"))}


# --------------------------------------------------------------------- tests

def test_shardwriter_discards_torn_tail():
    with tempfile.TemporaryDirectory() as td:
        w = ShardWriter(td, SITE, 0, d_in=8)
        w.write(__import__("torch").randn(10, 8)); w.checkpoint()
        good = (Path(td) / f"{SLUG}__b0.bin").read_bytes()
        w.write(__import__("torch").randn(3, 8))          # written, never checkpointed
        w._f.close()
        with open(Path(td) / f"{SLUG}__b0.bin", "ab") as f:
            f.write(b"\x00" * 7)                          # and a torn partial row
        w2 = ShardWriter(td, SITE, 0, d_in=8)
        assert w2.n_tokens == 10
        assert (Path(td) / f"{SLUG}__b0.bin").read_bytes() == good
        w2.close()
        print("  reopened shard truncated to its 10 checkpointed rows")


def test_killed_capture_resumes_bit_identically():
    with tempfile.TemporaryDirectory() as td:
        ref, res = Path(td) / "ref", Path(td) / "res"
        _capture(ref)
        reference = _bytes(ref)
        prog = json.loads((ref / "progress.json").read_text())
        assert prog["images_done"] == 24

        r = _capture(res, extra_env={"SAE_FAKE_CRASH_AFTER": "3"}, check=False)
        assert r.returncode != 0, "simulated kill did not happen"
        mid = json.loads((res / "progress.json").read_text())
        assert mid["images_done"] == 12, mid["images_done"]

        # Mimic the worst case: bytes from a batch that never got checkpointed.
        victim = sorted(res.glob("*.bin"))[0]
        with open(victim, "ab") as f:
            f.write(os.urandom(64 * 2 * 5 + 3))

        r2 = _capture(res)
        assert "resuming at image 12/24" in r2.stdout, r2.stdout[-500:]
        resumed = _bytes(res)
        assert resumed.keys() == reference.keys()
        for k in reference:
            assert resumed[k] == reference[k], f"{k} differs after resume"
        tok = {k: len(v) // (64 * 2) for k, v in reference.items()}
        print(f"  killed at 12/24, corrupted a shard tail, resumed: shards byte-identical "
              f"to the uninterrupted run {tok}")


def test_resume_refuses_changed_settings():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "a"
        _capture(out, n=8)
        r = _capture(out, n=8, seed=5, check=False)
        assert r.returncode != 0 and "different settings" in (r.stdout + r.stderr)
        _capture(out, n=8, seed=5, fresh=True)          # --fresh is the escape hatch
        print("  mismatched resume refused; --fresh starts over")


def test_merge_and_pool_are_virtual_and_correct():
    with tempfile.TemporaryDirectory() as td:
        acts = Path(td) / "data"
        _capture(acts / "gpu0", n=8, seed=0)
        _capture(acts / "gpu1", n=8, seed=1)
        _run(["scripts/merge_shards.py", "--inputs", str(acts / "gpu0"), str(acts / "gpu1"),
              "--out", str(acts / "merged")])
        _run(["scripts/pool_shards.py", "--acts", str(acts / "merged"), "--site", SITE])
        _run(["scripts/pool_shards.py", "--acts", str(acts / "merged"), "--site", SITE,
              "--matched"])

        merged = acts / "merged"
        assert not list(merged.glob("*.bin")), "merge copied data"
        size = sum(p.stat().st_size for p in merged.iterdir())
        assert size < 50_000, f"merged dir unexpectedly large: {size}"

        for b in (0, 1, 2):
            st = ActivationStore(merged / f"{SLUG}__b{b}.json", buffer_tokens=256)
            n0 = json.loads((acts / "gpu0" / f"{SLUG}__b{b}.json").read_text())["n_tokens"]
            n1 = json.loads((acts / "gpu1" / f"{SLUG}__b{b}.json").read_text())["n_tokens"]
            assert st.n_tokens == n0 + n1

        pooled = ActivationStore(merged / f"{SLUG}__bpooled.json", buffer_tokens=256)
        per = min(json.loads((merged / f"{SLUG}__b{b}.json").read_text())["n_tokens"]
                  for b in (0, 1, 2))
        assert pooled.n_tokens == 3 * per
        assert pooled.batch(64).shape == (64, 64)
        # The data-matched control: same total tokens as ONE bucket, a third from each.
        matched = ActivationStore(merged / f"{SLUG}__bpooled_matched.json", buffer_tokens=256)
        assert matched.n_tokens == 3 * (per // 3), (matched.n_tokens, per)

        # Reading through the manifest must return the same rows as the file.
        st = ActivationStore(merged / f"{SLUG}__b0.json", buffer_tokens=256, normalize=False)
        direct = np.memmap(acts / "gpu0" / f"{SLUG}__b0.bin", dtype=np.float16,
                           mode="r").reshape(-1, 64)
        got = st._gather(np.arange(5))
        assert np.allclose(got, np.asarray(direct[:5], dtype=np.float32))

        kinds = sorted(p.stem.split("__")[-1] for p in find_shards(merged))
        assert kinds == ["b0", "b1", "b2", "bpooled", "bpooled_matched"], kinds
        print(f"  merged 2 GPUs + pooled 3 buckets with {size:,} bytes of manifests; "
              f"pooled = 3 x {per} tokens")


def test_orchestrator_end_to_end_and_resume():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "proj"
        args = ["scripts/run_pipeline.py", "--root", str(root), "--model", "__fake__",
                "--site", SITE, "--gpus", "0", "--n-images", "16", "--capture-batch", "4",
                "--corpus", "32", "--expansion", "2", "--k", "4",
                "--train-steps", "60", "--train-batch", "256",
                "--only", "prefetch", "capture", "merge", "pool", "train"]
        r = _run(args)
        for s in ("prefetch", "capture", "merge", "pool", "train"):
            assert (root / "state" / f"{s}.done").exists(), s
        idx = json.loads((root / "runs" / "index.json").read_text())
        buckets = sorted(row["bucket"] for row in idx)
        assert buckets == ["0", "1", "2", "pooled", "pooled_matched"], buckets
        for row in idx:
            assert (Path(row["run_dir"]) / "sae.pt").exists()

        r2 = _run(args)
        skipped = r2.stdout.count("(done)")
        assert skipped == 5, r2.stdout[-800:]

        # Losing a stage marker re-runs only that stage, and training then skips
        # every SAE it already has.
        (root / "state" / "train.done").unlink()
        r3 = _run(args)
        assert r3.stdout.count("already trained") == 5, r3.stdout[-800:]
        print(f"  5 stages ran; rerun skipped all 5; re-entering train skipped 5 trained SAEs "
              f"({', '.join(buckets)})")


def test_lost_activations_trigger_recapture():
    """Colab case: results live on Drive but the VM (and its local activations)
    was recycled before training finished. Stale capture/merge/pool markers must
    be cleared so those stages run again instead of failing on missing files."""
    import shutil
    with tempfile.TemporaryDirectory() as td:
        root, acts = Path(td) / "proj", Path(td) / "local_acts"
        args = ["scripts/run_pipeline.py", "--root", str(root), "--acts-dir", str(acts),
                "--model", "__fake__", "--site", SITE, "--gpus", "0", "--n-images", "8",
                "--capture-batch", "4", "--corpus", "16",
                "--only", "prefetch", "capture", "merge", "pool"]
        _run(args)
        assert (root / "state" / "capture.done").exists()
        shutil.rmtree(acts)                                  # VM recycled
        r = _run(args)
        assert "activations missing -- re-running capture" in r.stdout
        assert (acts / "merged" / f"{SLUG}__bpooled.json").exists()
        print("  missing activations detected; capture/merge/pool re-ran automatically")


def test_all_eleven_stages_end_to_end():
    """Every stage, in order, through the real scripts -- including the four
    (atlas, substitute, amplify, erase) that only run after training. This is
    the test that would have caught the atlas crash on Kaggle (autograd graph
    left on SAE codes, then .numpy()) before a GPU run reached it."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "proj"
        r = _run(["scripts/run_pipeline.py", "--root", str(root), "--model", "__fake__",
                  "--clip", "__fake__", "--gpus", "0", "--n-images", "16",
                  "--capture-batch", "4", "--corpus", "32", "--expansion", "2",
                  "--k", "4", "--train-steps", "60", "--train-batch", "256",
                  "--locate-repeats", "1"])
        stages = ["prefetch", "capture", "merge", "pool", "train", "locate", "atlas",
                  "substitute", "amplify", "erase", "report"]
        missing = [s for s in stages if not (root / "state" / f"{s}.done").exists()]
        assert not missing, f"stages not completed: {missing}\n{r.stdout[-1500:]}"

        out = root / "outputs"
        atlas = json.loads((out / "atlas" / "atlas.json").read_text())
        assert atlas["features"] and all((out / "atlas" / f["image"]).exists()
                                         for f in atlas["features"])
        table = json.loads((out / "substitution_table.json").read_text())
        assert sorted(table) == sorted(f"{k}_b{b}" for k in ("bucketed", "pooled",
                                                              "pooled_matched")
                                       for b in (0, 1, 2))
        for cell in table.values():
            assert {"clean", "substituted", "ablated", "loss_recovered_std",
                    "per_draw"} <= set(cell), sorted(cell)
        idx = json.loads((root / "runs" / "index.json").read_text())
        assert {r["site"] for r in idx} == {"up_blocks.1.attentions.1.transformer_blocks.0"}
        for mode in ("amplify", "erase"):
            blob = json.loads((out / f"steer_{mode}.json").read_text())
            rep = blob["report"]
            assert rep, f"empty {mode} report"
            assert len(blob["seeds"]) == 3
            for r in rep.values():
                assert "concept_delta_std" in r and r["n_seeds"] == 3
                if mode == "erase":
                    assert "holdout_prompt_delta" in r
        report = (out / "REPORT.md").read_text()
        for section in ("## SAE training", "## Faithfulness", "## Candidate concept",
                        "## Causal test: amplify", "## Causal test: erase"):
            assert section in report, section
        assert "pooled_matched SAE" in report and "± " in report
        print(f"  all 11 stages completed (residual-stream site); atlas {len(atlas['features'])} sheets, "
              f"9-cell substitution table, both steering reports, REPORT.md complete")


def test_resume_from_previous_output_restarts_at_failed_stage():
    """The Kaggle situation: a run finished capture..locate, then failed at atlas.
    A new session seeded with --resume-from that output must skip everything up
    to locate, not copy the activations (training is done), and finish the rest."""
    import shutil
    with tempfile.TemporaryDirectory() as td:
        old, new = Path(td) / "old", Path(td) / "new"
        common = ["--model", "__fake__", "--clip", "__fake__", "--gpus", "0", "--site", SITE,
                  "--n-images", "16", "--capture-batch", "4", "--corpus", "32",
                  "--expansion", "2", "--k", "4", "--train-steps", "60",
                  "--train-batch", "256", "--locate-repeats", "1"]
        _run(["scripts/run_pipeline.py", "--root", str(old), *common,
              "--only", "prefetch", "capture", "merge", "pool", "train", "locate"])

        r = _run(["scripts/run_pipeline.py", "--root", str(new), "--resume-from", str(old),
                  *common])
        for s in ("capture", "merge", "pool", "train", "locate"):
            assert f"skip {s} (done)" in r.stdout, s
        assert "copying activations" not in r.stdout
        assert not (new / "data").exists(), "activations were copied needlessly"
        for s in ("atlas", "substitute", "amplify", "erase", "report"):
            assert (new / "state" / f"{s}.done").exists(), s
        # A run-1 output (different site) must NOT be treated as progress.
        other = Path(td) / "other"
        r = _run(["scripts/run_pipeline.py", "--root", str(other), "--resume-from", str(old),
                  *[c for c in common if c not in ("--site", SITE)],
                  "--only", "prefetch"])
        assert "ignoring it and starting fresh" in r.stdout, r.stdout[-600:]
        assert not (other / "state" / "locate.done").exists()
        print("  seeded from a run that stopped after locate: skipped 5 stages, "
              "copied no activations, finished atlas..report; a different-site "
              "source is ignored")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t(); print("  PASS")
        except Exception as e:
            failed += 1; print(f"  FAIL: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
