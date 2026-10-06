#!/usr/bin/env python
"""Stage 6 -- substitution loss. The evidence that decides whether the SAE
captured what the model actually reads.

    python scripts/06_substitution.py --sae runs/<shard>/sae.pt --site <site> \
        --bucket 1 --act-scale <scale>

THE ARGUMENT

Activation-space MSE weights every direction equally. The UNet does not: some
directions in that activation space drive the output strongly and some are
nearly ignored. So an SAE can reconstruct 95% of the variance and still destroy
the 5% the model depends on, or reconstruct 80% and lose nothing that matters.
MSE cannot tell those apart.

Splicing the reconstruction back into the forward pass and measuring the model's
*own* denoising loss can. That is the diffusion analogue of "cross-entropy
recovered" in LLM interpretability work, and it is the number to lead with.

WHY THE ABLATION CONTROL IS NOT OPTIONAL

"Substituted loss = 0.153" is uninterpretable on its own. Compared to what? If
the layer barely matters, replacing it with noise also gives 0.153 and the SAE
has demonstrated nothing. So we measure three conditions and report the ratio:

    loss_recovered = (ablated - substituted) / (ablated - clean)

    1.0  the SAE loses nothing the model needed
    0.0  the SAE is no better than deleting the layer
    <0   the reconstruction is actively worse than deletion

Reporting this instead of raw MSE is the difference between "my autoencoder
works" and "the model's computation survives my autoencoder".

TIMESTEP RESTRICTION

The SAE was fit to one bucket, so it is only asked to reconstruct inside that
bucket. Substituting at timesteps it never saw measures out-of-distribution
behaviour, which is a different (and also interesting) question -- ``--all-buckets``
runs that version, and a large gap between in-bucket and out-of-bucket recovery
is direct evidence that the bucketing was necessary.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.model import SparseAutoencoder
from sae.hooks import get_module, bucket_of
from sae.loader import load_pipeline, pick_device
from sae.prompts import DISCOVERY


@torch.no_grad()
def make_latents(pipe, prompts, steps, size, seed, device):
    """Generate images and VAE-encode them back to clean latents.

    Using the model's own samples rather than real photographs keeps the
    evaluation on-distribution: we are asking whether the SAE preserves the
    computation the model performs on the data it actually produces.
    """
    lat, emb = [], []
    for p in prompts:
        g = torch.Generator(device=device).manual_seed(seed)
        img = pipe([p], num_inference_steps=steps, height=size, width=size,
                   generator=g).images[0]
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0  # HWC in [-1,1]
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, pipe.vae.dtype)
        lat.append(pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor)

        ids = pipe.tokenizer([p], padding="max_length",
                             max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(device)
        emb.append(pipe.text_encoder(ids)[0])
    return torch.cat(lat), torch.cat(emb)


def main():
    # Inference only. Without this, SAE outputs carry autograd graphs: memory is
    # wasted and anything that calls .numpy() on them crashes (stage 5 did).
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--site", required=True)
    ap.add_argument("--bucket", type=int, required=True)
    ap.add_argument("--act-scale", type=float, required=True)
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--n-timesteps", type=int, default=12)
    ap.add_argument("--all-buckets", action="store_true",
                    help="also substitute outside the SAE's own timestep bucket")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="outputs/substitution.json")
    ap.add_argument("--latent-cache", default="outputs/substitution_latents.pt",
                    help="reuse the same clean latents across runs; the pooled-vs-"
                         "bucketed table needs every cell evaluated on identical data")
    args = ap.parse_args()

    device = pick_device(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = load_pipeline(args.model, dtype=dtype, device=device)
    sae = SparseAutoencoder.load(args.sae).to(device).eval()
    mod = get_module(pipe.unet, args.site)

    prompts = DISCOVERY[:args.n_prompts]
    cache = Path(args.latent_cache) if args.latent_cache else None
    if cache is not None and cache.exists():
        blob = torch.load(cache, map_location="cpu")
        if blob.get("prompts") == prompts and blob.get("seed") == args.seed:
            latents, embeds = blob["latents"].to(dtype), blob["embeds"].to(dtype)
            print(f"reusing {len(prompts)} cached latents from {cache}", flush=True)
        else:
            latents = None
    else:
        latents = None
    if latents is None:
        print(f"encoding {len(prompts)} latents...", flush=True)
        latents, embeds = make_latents(pipe, prompts, args.steps, args.size,
                                       args.seed, device)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"prompts": prompts, "seed": args.seed,
                        "latents": latents.cpu(), "embeds": embeds.cpu()}, cache)

    sched = pipe.scheduler
    T = sched.config.num_train_timesteps
    edges = (334.0, 667.0)
    lo = 0 if args.bucket == 0 else edges[args.bucket - 1]
    hi = T if args.bucket == len(edges) else edges[args.bucket]

    state = {"mode": "clean", "t": 0.0, "n_hit": 0, "n_skip": 0}

    def hook(module, a, output):
        is_t = isinstance(output, (tuple, list))
        x = output[0] if is_t else output
        if x.dim() != 3 or state["mode"] == "clean":
            return output
        if not args.all_buckets and bucket_of(state["t"], edges) != args.bucket:
            state["n_skip"] += 1
            return output
        state["n_hit"] += 1
        if state["mode"] == "substituted":
            sae_dev = next(sae.parameters()).device
            xf = x.float().to(sae_dev) * args.act_scale
            new = (sae.decode(sae.encode(xf)) / args.act_scale).to(x.device)
        else:                                     # ablated -> spatial mean
            new = x.float().mean(dim=1, keepdim=True).expand_as(x)
        new = new.to(x.dtype)
        return (new,) + tuple(output[1:]) if is_t else new

    handle = mod.register_forward_hook(hook)

    # Fixed noise and fixed timesteps across all three conditions -- otherwise
    # the comparison is contaminated by sampling variance and the differences
    # you are looking for are smaller than the noise.
    torch.manual_seed(args.seed)
    # One device for everything the UNet consumes, whichever path produced the
    # latents (fresh VAE encode or the cache). Before this, cached latents were
    # loaded onto `device` while fresh ones stayed wherever the VAE put them, so
    # the two paths could disagree -- which crashed the second substitution run.
    unet_dev = next(pipe.unet.parameters()).device
    latents, embeds = latents.to(unet_dev), embeds.to(unet_dev)
    noises = [torch.randn_like(latents) for _ in range(args.n_timesteps)]
    if args.all_buckets:
        ts = [int(T * (i + 0.5) / args.n_timesteps) for i in range(args.n_timesteps)]
    else:
        ts = [int(lo + (hi - lo) * (i + 0.5) / args.n_timesteps)
              for i in range(args.n_timesteps)]

    results = {}
    with torch.no_grad():
        for mode in ("clean", "substituted", "ablated"):
            state["mode"] = mode
            state["n_hit"] = state["n_skip"] = 0
            total = 0.0
            for noise, t in zip(noises, ts):
                state["t"] = float(t)
                tt = torch.full((latents.shape[0],), t, device=unet_dev, dtype=torch.long)
                noisy = sched.add_noise(latents, noise, tt)
                pred = pipe.unet(noisy, tt, encoder_hidden_states=embeds).sample
                total += float(F.mse_loss(pred.float(), noise.float().to(pred.device)))
            results[mode] = total / len(ts)
            print(f"{mode:<13} loss {results[mode]:.6f}   "
                  f"(hooked {state['n_hit']}, skipped {state['n_skip']})", flush=True)

    handle.remove()

    denom = results["ablated"] - results["clean"]
    results["loss_recovered"] = (results["ablated"] - results["substituted"]) / denom \
        if abs(denom) > 1e-9 else float("nan")
    results.update({
        "site": args.site, "bucket": args.bucket,
        "timestep_range": [ts[0], ts[-1]],
        "all_buckets": args.all_buckets,
        "n_prompts": len(prompts), "n_timesteps": args.n_timesteps,
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    lr = results["loss_recovered"]
    print(f"\nloss recovered: {lr:.3f}")
    if denom <= 0:
        print("  WARNING: ablating this site did not increase loss. The site "
              "barely matters here, so loss_recovered is meaningless -- pick a "
              "site the model actually depends on before reporting this.")
    elif lr > 0.9:
        print("  strong: the model's computation largely survives the SAE")
    elif lr > 0.6:
        print("  moderate: usable, but report it honestly alongside EV")
    else:
        print("  weak: high explained variance here would be misleading -- the "
              "SAE is dropping directions the model depends on")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
