"""Build notebooks/project_b_unattended.ipynb: a self-contained notebook that carries
the code (sae/, scripts/, tests/) as an embedded archive, so Kaggle needs no git clone.

    python tools/make_notebook.py            # from the repo root, after committing
"""
import base64
import json
import subprocess

# Code only -- results/, assets/ and notebooks/ would bloat the payload.
_tgz = subprocess.run(["git", "archive", "--format=tar.gz", "HEAD", "sae", "scripts", "tests",
                       "outputs", "requirements.txt", "README.md", ".gitignore"],
                      check=True, capture_output=True).stdout
b64 = base64.b64encode(_tgz).decode()
# payload layout check: files must sit at the archive root (tests/..., scripts/...),
# because the notebook extracts INTO diffusion-sae/. A --prefix archive nests them.
import base64 as _b, io as _io, tarfile as _t
_names = _t.open(fileobj=_io.BytesIO(_b.b64decode(b64)), mode="r:gz").getnames()
assert "tests/test_core.py" in _names and "scripts/run_pipeline.py" in _names, _names[:5]
lines = [b64[i:i + 120] for i in range(0, len(b64), 120)]


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""# Project B — SAEs for concept control in Stable Diffusion (unattended)

The whole pipeline runs from **one cell** and doesn't need you watching it.

### On Kaggle (recommended)
1. Sidebar → **Accelerator: GPU T4 x2**, **Internet: On**.
2. Top right → **Save Version** → **Save & Run All (Commit)** → Save.
3. Close the tab. This runs as a background job, so there's no idle disconnect. The limit is 12 hours per run; run 2 takes roughly 4–6 (twice the images of run 1, plus 3 seeds for steering).
4. Come back to **your notebook → Versions → the run → Output / Logs**. `project_b/outputs/REPORT.md` has every number.

### If a stage fails
The notebook doesn't stop with an error. It finishes cleanly so the version **saves everything completed so far**. To continue from the failed stage instead of starting over:
open the notebook → **Add Input → Your Work → Notebooks →** this notebook → pick the failed version. Then **Save & Run All** again.
The configuration cell finds the attached output and resumes. Finished stages are skipped, and the activations are only copied if training hadn't finished.

It uses both T4s: captures run side by side, and the five SAEs are split across the two GPUs.

### Adding the erasure experiment to a finished run (about 1–1.5 hours)
Do **not** start from scratch. Open the notebook → **Add Input → Your Work → Notebooks →** this notebook →
pick the version that produced run 2's results. Then **Save & Run All**. Every finished stage is skipped (no
capture, no training); only `erase_multi` runs, one SAE per GPU, and `REPORT.md` is rebuilt with the new section.
If several earlier versions are attached, the one captured at the current `SITE` is used.

### On Colab
Runtime → T4 GPU → Run all. It mounts Google Drive and keeps SAEs and results there, so if the
runtime disconnects **after training**, Run all again picks up where it stopped. Activations stay on the VM's local disk. A disconnect *during* capture or training
redoes those stages, because free Colab runtimes don't survive a disconnect. You only get one GPU, so it's roughly twice as slow as Kaggle.
Free Colab also has no background execution — keep the tab open."""))

cells.append(code(
    "!pip install -q 'diffusers>=0.27' 'transformers>=4.38' accelerate safetensors\n"
    "import torch, subprocess\n"
    "print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), "
    "'| GPUs', torch.cuda.device_count())\n"
    "print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],\n"
    "                     capture_output=True, text=True).stdout)\n"))

cells.append(code(
    "import base64, io, tarfile, os, pathlib, sys\n"
    "REPO_B64 = (\n" + "".join(f'    "{l}"\n' for l in lines) + ")\n"
    "BASE = '/kaggle/working' if os.path.exists('/kaggle/working') else "
    "('/content' if os.path.exists('/content') else os.getcwd())\n"
    "repo = pathlib.Path(BASE) / 'diffusion-sae'\n"
    "repo.mkdir(parents=True, exist_ok=True)\n"
    "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(REPO_B64)), mode='r:gz') as tf:\n"
    "    try:\n"
    "        tf.extractall(repo, filter='data')\n"
    "    except TypeError:\n"
    "        tf.extractall(repo)\n"
    "os.chdir(repo); sys.path.insert(0, str(repo))\n"
    "print('repo at', repo)\n"))

cells.append(md("### Go / no-go: test suite (CPU, ~3 min)\nAll three must pass, or don't spend GPU time."))

cells.append(code(
    "import subprocess, sys\n"
    "for t in ('tests/test_core.py', 'tests/test_stages.py', 'tests/test_pipeline.py'):\n"
    "    r = subprocess.run([sys.executable, t], capture_output=True, text=True)\n"
    "    print(t, '->', r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])\n"
    "    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]\n"))

cells.append(md("### Configuration\nThe defaults are sized for Kaggle T4 x2. On Colab, `N_IMAGES` drops to fit one GPU and the VM's disk."))

cells.append(code(
    "import os, sys, pathlib\n"
    "ON_KAGGLE = os.path.exists('/kaggle/working')\n"
    "ON_COLAB = 'google.colab' in sys.modules\n"
    "\n"
    "if ON_KAGGLE:\n"
    "    ROOT = '/kaggle/working/project_b'          # saved as the version's output\n"
    "    ACTS = ROOT + '/data'\n"
    "    N_IMAGES = 1000\n"
    "elif ON_COLAB:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    ROOT = '/content/drive/MyDrive/project_b'   # survives disconnects\n"
    "    ACTS = '/content/sae_acts'                  # fast local disk; random reads from Drive are slow\n"
    "    N_IMAGES = 600\n"
    "else:\n"
    "    ROOT = str(pathlib.Path.cwd() / 'project_b_run'); ACTS = ROOT + '/data'; N_IMAGES = 500\n"
    "\n"
    "# Run 2 captures the transformer block's residual stream. Run 1 used the attn2\n"
    "# sub-layer, whose output turned out to be low-rank (82-92% dead features).\n"
    "# Set SITE = ATTN2 only to reproduce run 1.\n"
    "RESIDUAL = 'up_blocks.1.attentions.1.transformer_blocks.0'\n"
    "ATTN2 = RESIDUAL + '.attn2'\n"
    "SITE = RESIDUAL\n"
    "\n"
    "# Delete the ~8 GB of captured activations once everything has finished. Keeps\n"
    "# the saved Kaggle output small. Set True if you want to retrain later\n"
    "# without recapturing.\n"
    "KEEP_ACTS = False\n"
    "\n"
    "# Resume source: this session's own previous progress is picked up automatically\n"
    "# (same ROOT). A previous *version's* output can be attached via Add Input.\n"
    "import glob\n"
    "RESUME_FROM = None\n"
    "if ON_KAGGLE:\n"
    "    import json as _json\n"
    "    def _site(st):\n"
    "        f = os.path.join(st, 'capture_info.json')\n"
    "        return _json.load(open(f))['site'] if os.path.exists(f) else ATTN2\n"
    "    found = sorted(glob.glob('/kaggle/input/**/project_b/state', recursive=True))\n"
    "    match = [st for st in found if _site(st) == SITE]\n"
    "    # Prefer the attached run with the most finished stages at this SITE.\n"
    "    match.sort(key=lambda st: len(glob.glob(os.path.join(st, '*.done'))), reverse=True)\n"
    "    if match:\n"
    "        RESUME_FROM = os.path.dirname(match[0])\n"
    "    elif found:\n"
    "        print('attached outputs were captured at a different site -- not resuming from them')\n"
    "print(f'ROOT={ROOT}\\nACTS={ACTS}\\nSITE={SITE}\\nN_IMAGES={N_IMAGES}\\nRESUME_FROM={RESUME_FROM}')\n"))

cells.append(md("""## Run everything

One command, eleven stages. Each stage is skipped if it already finished. Capture and training also resume *within* the
stage: capture checkpoints after every batch, and training skips any SAE that's already saved.
The log prints images/min and an ETA during capture.
If the cell is interrupted, run it again."""))

cells.append(code(
    "import subprocess, sys\n"
    "cmd = [sys.executable, '-u', 'scripts/run_pipeline.py', '--root', ROOT,\n"
    "       '--acts-dir', ACTS, '--n-images', str(N_IMAGES), '--site', SITE]\n"
    "if RESUME_FROM:\n"
    "    cmd += ['--resume-from', RESUME_FROM]\n"
    "print(' '.join(cmd), flush=True)\n"
    "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
    "                        text=True, bufsize=1)\n"
    "for line in proc.stdout:\n"
    "    print(line, end='', flush=True)\n"
    "PIPELINE_OK = proc.wait() == 0\n"
    "# Deliberately no exception: a background run that errors may not save its\n"
    "# output, and that output is what lets the next run resume.\n"
    "if PIPELINE_OK:\n"
    "    print('\\nPIPELINE COMPLETE')\n"
    "else:\n"
    "    print('\\n' + '=' * 70 + '\\nPIPELINE STOPPED -- a stage failed. Everything finished so far is\\n'\n"
    "          f'saved under {ROOT}. Logs: {ROOT}/logs. To continue, attach this version\\n'\n"
    "          'as input (Add Input -> Your Work) and Save & Run All again.\\n' + '=' * 70)\n"
    "    _st = f'{ROOT}/state/last_run.json'\n"
    "    if os.path.exists(_st):\n"
    "        import json as _json\n"
    "        _f = (_json.load(open(_st)).get('failed') or {})\n"
    "        if _f.get('log') and os.path.exists(_f['log']):\n"
    "            print(f\"\\nlast 60 lines of {_f['log']}:\\n\")\n"
    "            print(''.join(open(_f['log']).readlines()[-60:]))\n"))

cells.append(md("## Results"))

cells.append(code(
    "from IPython.display import Markdown, display\n"
    "rep = f'{ROOT}/outputs/REPORT.md'\n"
    "if os.path.exists(rep):\n"
    "    display(Markdown(open(rep).read()))\n"
    "else:\n"
    "    print('no report yet -- the pipeline did not reach the report stage')\n"))

cells.append(code(
    "import json, glob\n"
    "from PIL import Image\n"
    "ap = f'{ROOT}/outputs/atlas/atlas.json'\n"
    "atlas = json.load(open(ap)) if os.path.exists(ap) else {'features': []}\n"
    "print('Atlas -- correlational evidence only; labels are tested by the causal stages')\n"
    "for f in atlas['features'][:8]:\n"
    "    print(f\"\\nfeature {f['feature']}  fires on {f['fire_rate']:.0%} of prompts\")\n"
    "    print('   ', ' | '.join(p[:40] for p in f['top_prompts'][:3]))\n"
    "    display(Image.open(f\"{ROOT}/outputs/atlas/{f['image']}\"))\n"))

cells.append(code(
    "for p in sorted(glob.glob(f'{ROOT}/outputs/grids/*.png'))[:16]:\n"
    "    print(p.split('/')[-1]); display(Image.open(p).resize((256, 256)))\n"))

cells.append(md("## Package"))

cells.append(code(
    "import shutil, subprocess\n"
    "if PIPELINE_OK and not KEEP_ACTS and os.path.exists(ACTS):\n"
    "    shutil.rmtree(ACTS); print('removed activations to keep the output small')\n"
    "parts = [d for d in ('runs', 'outputs', 'logs') if os.path.exists(f'{ROOT}/{d}')]\n"
    "if parts:\n"
    "    subprocess.run(['tar', '-czf', f'{ROOT}/project_b_results.tgz', '-C', ROOT, *parts], check=True)\n"
    "print('download:', f'{ROOT}/project_b_results.tgz')\n"))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open("notebooks/project_b_unattended.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", len(cells), "cells")
