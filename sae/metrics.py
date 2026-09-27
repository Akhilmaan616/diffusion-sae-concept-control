"""Evaluation.

The headline claim of any SAE paper is "these are the model's features". The
evidence for that claim is where most projects are weak, so this module is
deliberately opinionated about what counts.

THREE LEVELS OF EVIDENCE, weakest to strongest:

1. Reconstruction (``explained_variance``). Necessary, not sufficient. PCA gets
   high explained variance and PCA components are not features. If someone asks
   "your SAE reconstructs to 95%, doesn't that mean you found the real
   features?" the answer is no -- a dense linear autoencoder does better and
   finds nothing interpretable. Reconstruction only rules out failure.

2. Substitution (``substitution_loss``). Splice the SAE's reconstruction back
   into the UNet's forward pass and measure the model's *own* denoising loss.
   This is the diffusion analogue of "cross-entropy recovered" in LLM SAE work,
   and it is strictly more informative than activation-space MSE, because MSE
   weights all directions equally while the model does not. An SAE can have low
   MSE and still destroy the directions the model actually reads.

3. Causal intervention (``steering_effect``). Amplify the feature, generate,
   and measure whether the intended concept appeared -- and whether anything
   else changed. This is the only evidence that distinguishes "feature that
   correlates with dogs" from "feature the model uses to draw dogs". A feature
   that fires on dog images but does nothing when you amplify it is a
   correlate, not a cause, and should not be described as "the dog feature".

For erasure specifically, one number is always meaningless. Erasure success
without collateral damage is trivially achievable by breaking the model, which
is exactly the failure the LAPE baseline exhibited in the detoxification work:
a 99.9% "reduction" that was the model emitting gibberish. Always report the
pair.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Level 1 -- reconstruction
# ---------------------------------------------------------------------------


@torch.no_grad()
def reconstruction_report(sae, store, n_batches: int = 20,
                          batch_size: int = 4096) -> dict:
    device = next(sae.parameters()).device
    tot_ev, tot_l0, tot_mse, tot_cos, n = 0.0, 0.0, 0.0, 0.0, 0

    for _ in range(n_batches):
        x = store.batch(batch_size).to(device)
        out = sae(x, update_stats=False)
        resid_var = (x - out.x_hat).var(dim=0, unbiased=False).sum()
        total_var = x.var(dim=0, unbiased=False).sum().clamp_min(1e-8)

        tot_ev += float(1.0 - resid_var / total_var)
        tot_l0 += float((out.codes > 0).float().sum(-1).mean())
        tot_mse += float(F.mse_loss(out.x_hat, x))
        tot_cos += float(F.cosine_similarity(out.x_hat, x, dim=-1).mean())
        n += 1

    return {
        "explained_variance": tot_ev / n,
        "l0": tot_l0 / n,
        "mse": tot_mse / n,
        "cosine_similarity": tot_cos / n,
    }


# ---------------------------------------------------------------------------
# Level 2 -- substitution
# ---------------------------------------------------------------------------


@torch.no_grad()
def substitution_loss(unet, sae, site: str, batch, scale: float,
                      noise_scheduler, text_embeds, device="cuda",
                      n_steps: int = 8) -> dict:
    """Denoising loss with and without the SAE spliced into the forward pass.

    ``batch`` is a tensor of clean latents [B, 4, H, W]. We add noise at several
    timesteps, predict, and compare the model's MSE against the true noise in
    three conditions:

        clean       -- unmodified UNet
        substituted -- site activations replaced by sae(decode(encode(x)))
        ablated     -- site activations replaced by their mean (the control)

    The number to report is *loss recovered*:

        (ablated - substituted) / (ablated - clean)

    which is 1.0 if the SAE loses nothing and 0.0 if it is no better than
    deleting the layer. Reporting substituted loss alone is uninterpretable,
    because you have no idea how much that layer mattered in the first place.
    """
    from .hooks import get_module

    mod = get_module(unet, site)
    results = {}

    def run(mode: str) -> float:
        handle = None
        if mode != "clean":
            def hook(m, args, output):
                is_t = isinstance(output, (tuple, list))
                x = output[0] if is_t else output
                if x.dim() != 3:
                    return output
                if mode == "substituted":
                    xf = x.float() * scale
                    new = sae.decode(sae.encode(xf)) / scale
                else:                                     # ablated
                    new = x.float().mean(dim=1, keepdim=True).expand_as(x)
                new = new.to(x.dtype)
                return (new,) + tuple(output[1:]) if is_t else new
            handle = mod.register_forward_hook(hook)

        total, count = 0.0, 0
        try:
            for i in range(n_steps):
                t = torch.full((batch.shape[0],),
                               int(noise_scheduler.config.num_train_timesteps * (i + 0.5) / n_steps),
                               device=device, dtype=torch.long)
                noise = torch.randn_like(batch)
                noisy = noise_scheduler.add_noise(batch, noise, t)
                pred = unet(noisy, t, encoder_hidden_states=text_embeds).sample
                total += float(F.mse_loss(pred.float(), noise.float()))
                count += 1
        finally:
            if handle is not None:
                handle.remove()
        return total / max(1, count)

    for mode in ("clean", "substituted", "ablated"):
        results[mode] = run(mode)

    denom = results["ablated"] - results["clean"]
    results["loss_recovered"] = (
        (results["ablated"] - results["substituted"]) / denom if abs(denom) > 1e-9 else float("nan")
    )
    return results


# ---------------------------------------------------------------------------
# Level 3 -- causal intervention
# ---------------------------------------------------------------------------


@dataclass
class SteeringResult:
    feature: int
    concept: str
    strength: float
    clip_target: float          # CLIP sim to the concept prompt
    clip_target_baseline: float
    clip_source: float          # CLIP sim to the original prompt (retention)
    clip_source_baseline: float
    lpips_to_baseline: float    # how much the image moved overall

    @property
    def concept_gain(self) -> float:
        return self.clip_target - self.clip_target_baseline

    @property
    def collateral(self) -> float:
        """Drop in fidelity to the original prompt. Lower is better.

        A steering result with a large concept_gain and a large collateral is
        not a targeted edit -- it is the model being pushed off-distribution.
        """
        return self.clip_source_baseline - self.clip_source

    def to_dict(self) -> dict:
        d = asdict(self)
        d["concept_gain"] = self.concept_gain
        d["collateral"] = self.collateral
        return d


class ClipScorer:
    """Thin wrapper so evaluation does not depend on how CLIP is loaded.

    Deliberately a *different* CLIP checkpoint from the one inside Stable
    Diffusion. SD1.5 conditions on ViT-L/14 text embeddings; scoring with the
    same encoder measures how well you satisfied the model's own objective,
    which is circular. Use ViT-B/32 or SigLIP for scoring.
    """

    def __init__(self, model, processor, device="cuda"):
        self.model, self.processor, self.device = model, processor, device

    @torch.no_grad()
    def similarity(self, images, texts) -> torch.Tensor:
        inputs = self.processor(text=texts, images=images, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        out = self.model(**inputs)
        img = F.normalize(out.image_embeds, dim=-1)
        txt = F.normalize(out.text_embeds, dim=-1)
        return (img * txt).sum(-1)


def erasure_report(success: list[float], collateral: list[float]) -> dict:
    """Pair the two numbers that must always be reported together."""
    s = torch.tensor(success, dtype=torch.float32)
    c = torch.tensor(collateral, dtype=torch.float32)
    return {
        "erasure_success_mean": float(s.mean()),
        "erasure_success_std": float(s.std(unbiased=False)),
        "collateral_mean": float(c.mean()),
        "collateral_std": float(c.std(unbiased=False)),
        # A single scalar for ranking methods: how much erasure you bought per
        # unit of damage. Report the pair as well -- this ratio hides the scale.
        "erasure_per_damage": float(s.mean() / c.mean().clamp_min(1e-6)),
        "n": len(success),
    }
