"""Robust SD1.5 loading, including from a partial local HuggingFace cache.

WHY THIS FILE EXISTS

A HuggingFace cache directory is not automatically a loadable pipeline.
``StableDiffusionPipeline.from_pretrained`` needs ``model_index.json`` (the
manifest naming each sub-module and its class) plus a ``scheduler/`` folder. If
you previously downloaded only the components you needed -- unet, vae,
text_encoder, tokenizer -- for your own training code, both of those are
missing and ``from_pretrained`` fails with a confusing error about the repo not
being found.

That is exactly the state of the local ``models--runwayml--stable-diffusion-v1-5``
cache: 4.3 GB of real weights, no manifest, no scheduler.

So this loader tries three things in order:

    1. ``from_pretrained`` on whatever you passed (a Hub id or a full snapshot)
    2. component-wise assembly from a partial snapshot directory
    3. a clear error naming what is missing

Route 2 reconstructs the scheduler from the published SD1.5 configuration rather
than downloading it. The three values worth knowing, because getting them wrong
produces subtly wrong samples rather than a crash:

    beta_schedule  "scaled_linear" with beta 0.00085 -> 0.012. SD1.5 does not
                   use the linear schedule that is the diffusers default.
    steps_offset   1. Without it the sampler is off by one timestep, which
                   mostly shows up as slightly washed-out images.
    set_alpha_to_one  False. Controls the final alpha at t=0; True gives a
                   different (and for SD1.5, wrong) terminal signal-to-noise.

RUNWAYML NOTE: the ``runwayml/stable-diffusion-v1-5`` repo was removed from the
Hub. Cached copies still work; a fresh download needs the mirror
``stable-diffusion-v1-5/stable-diffusion-v1-5``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Published SD1.5 scheduler configuration. Reproduced here so a partial cache
# can be used fully offline.
SD15_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "trained_betas": None,
    "clip_sample": False,
    "set_alpha_to_one": False,
    "steps_offset": 1,
    "prediction_type": "epsilon",
}

HUB_MIRROR = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def find_snapshot(path: str | Path) -> Path:
    """Resolve an HF cache directory to the actual snapshot folder.

    Accepts any of:
        .../models--org--name
        .../models--org--name/snapshots
        .../models--org--name/snapshots/<sha>
        a plain directory that already contains unet/ etc.
    """
    p = Path(path)
    if (p / "unet").is_dir():
        return p
    snaps = p / "snapshots"
    if snaps.is_dir():
        subs = sorted([d for d in snaps.iterdir() if d.is_dir()])
        if subs:
            return subs[-1]
    if p.name == "snapshots":
        subs = sorted([d for d in p.iterdir() if d.is_dir()])
        if subs:
            return subs[-1]
    return p


def inspect(path: str | Path) -> dict:
    """Report what a local checkpoint has and what it is missing."""
    snap = find_snapshot(path)
    needed_dirs = ["unet", "vae", "text_encoder", "tokenizer"]
    report = {
        "snapshot": str(snap),
        "present": {d: (snap / d).is_dir() for d in needed_dirs},
        "has_model_index": (snap / "model_index.json").is_file(),
        "has_scheduler": (snap / "scheduler").is_dir(),
    }
    report["loadable_directly"] = report["has_model_index"] and report["has_scheduler"]
    report["loadable_componentwise"] = all(report["present"].values())
    return report


def load_pipeline(model: str, dtype=torch.float16, device: str = "cuda",
                  verbose: bool = True, scheduler: str = "ddim"):
    """Return a StableDiffusionPipeline from a Hub id or a (possibly partial)
    local cache directory.

    ``scheduler="ddim"`` (default) forces the same DDIM sampler on every route.
    This matters more than it looks. The Hub copy of SD1.5 ships PNDM, whose
    PLMS warm-up evaluates one timestep twice (26 UNet passes for 25 steps:
    961, 921, 921, 881, ...), while a partial local cache has no scheduler and
    gets DDIM (25 passes). Left alone, captures made on Kaggle and captures
    made locally would have different timestep grids and different per-bucket
    token splits -- 9/8/9 versus 9/8/8 -- and nothing would warn you.
    Pass ``scheduler="native"`` to keep whatever the checkpoint ships.
    """
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    def _finish(pipe):
        if scheduler == "ddim":
            pipe.scheduler = DDIMScheduler(**SD15_SCHEDULER_CONFIG)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        if verbose:
            print(f"scheduler: {type(pipe.scheduler).__name__}")
        return pipe

    is_local = Path(model).exists()

    # ---- route 1: straight from_pretrained --------------------------------
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            str(find_snapshot(model)) if is_local else model,
            torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
            local_files_only=is_local,
        )
        if verbose:
            print(f"loaded pipeline directly from {model}")
        return _finish(pipe)
    except Exception as e:
        if not is_local:
            raise
        first_error = e

    # ---- route 2: assemble from components --------------------------------
    snap = find_snapshot(model)
    info = inspect(snap)
    if not info["loadable_componentwise"]:
        missing = [k for k, v in info["present"].items() if not v]
        raise FileNotFoundError(
            f"{snap} is missing {missing}. Direct load failed with: {first_error}"
        )

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    if verbose:
        print(f"partial cache at {snap}: "
              f"model_index={info['has_model_index']} scheduler={info['has_scheduler']}")
        print("assembling pipeline component-wise with the published SD1.5 "
              "scheduler config")

    unet = UNet2DConditionModel.from_pretrained(snap / "unet", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(snap / "vae", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(snap / "text_encoder", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(snap / "tokenizer")
    scheduler = DDIMScheduler(**SD15_SCHEDULER_CONFIG)

    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    )
    return _finish(pipe)


def write_missing_pipeline_files(path: str | Path) -> list[str]:
    """Repair a partial cache in place so plain ``from_pretrained`` works later.

    Writes ``model_index.json`` and ``scheduler/scheduler_config.json`` into the
    snapshot. Only touches files that do not already exist.
    """
    snap = find_snapshot(path)
    written = []

    mi = snap / "model_index.json"
    if not mi.is_file():
        mi.write_text(json.dumps({
            "_class_name": "StableDiffusionPipeline",
            "_diffusers_version": "0.27.0",
            "feature_extractor": [None, None],
            "safety_checker": [None, None],
            "scheduler": ["diffusers", "DDIMScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }, indent=2))
        written.append(str(mi))

    sched_dir = snap / "scheduler"
    sched = sched_dir / "scheduler_config.json"
    if not sched.is_file():
        sched_dir.mkdir(exist_ok=True)
        cfg = dict(SD15_SCHEDULER_CONFIG)
        cfg["_class_name"] = "DDIMScheduler"
        cfg["_diffusers_version"] = "0.27.0"
        sched.write_text(json.dumps(cfg, indent=2))
        written.append(str(sched))

    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="inspect or repair a local SD checkpoint")
    ap.add_argument("path")
    ap.add_argument("--repair", action="store_true")
    a = ap.parse_args()
    print(json.dumps(inspect(a.path), indent=2))
    if a.repair:
        w = write_missing_pipeline_files(a.path)
        print("wrote:", w or "nothing (already complete)")
