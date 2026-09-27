"""Forward hooks for capturing and steering Stable Diffusion UNet activations.

THE TIMESTEP PROBLEM -- read this before touching anything else.

In a language model an SAE sees one activation distribution: layer L's residual
stream, full stop. A diffusion UNet does not work that way. The same layer, with
the same weights, is evaluated ~50 times per image at different noise levels,
and it is doing a *different job* each time:

    t ~ 1000 (pure noise)   -> layout, pose, global composition
    t ~ 500                 -> object identity, structure
    t ~ 0   (nearly clean)  -> texture, high-frequency detail, colour

So "the activation distribution at up_blocks.1.attn2" is really a mixture of
several distributions indexed by t. Fit one SAE to the mixture and you get
features that are averages over incompatible regimes -- they reconstruct
acceptably and mean nothing.

Two defensible responses, both implemented here:

  * ``per_bucket`` (default) -- partition the timestep range into buckets and fit
    an independent SAE to each. Simple, and it makes the regime differences
    visible: you can literally compare the feature atlas of the high-t SAE
    against the low-t one and see composition features versus texture features.
    Cost is k SAEs and no parameter sharing.

  * ``conditioned`` -- one SAE, with a Fourier embedding of t concatenated to the
    input. Shares parameters across regimes and lets you ask whether a feature
    is timestep-invariant. Harder to interpret, because a feature's meaning can
    now drift continuously with t.

Start with per_bucket. It is the version you can explain in one sentence, and
the comparison across buckets is itself a result.

CLASSIFIER-FREE GUIDANCE -- the second thing that silently corrupts this.

Under CFG, diffusers evaluates the UNet on a batch of ``cat([uncond, cond])``.
Half of every captured tensor is the *unconditional* pass, which by construction
contains no prompt information. Include it and roughly half your training data
is concept-free noise, which mostly teaches the SAE to model the unconditional
mean. ``cond_only=True`` keeps only the conditional half. Leave it on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Default capture sites for the SD1.5 UNet (diffusers UNet2DConditionModel).
#
# attn2 is cross-attention: this is where text conditioning enters the network,
# so its output is the most concept-dense place to look. Widths differ by block,
# which is why each site gets its own SAE.
# ---------------------------------------------------------------------------

SD15_SITES = {
    # name                                            width
    "mid_block.attentions.0.transformer_blocks.0.attn2": 1280,
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2": 1280,
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2": 640,
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2": 1280,
}

# A good single-site default: deep enough to be semantic, shallow enough that
# the features still localise spatially.
DEFAULT_SITE = "up_blocks.1.attentions.1.transformer_blocks.0.attn2"


def get_module(root: nn.Module, dotted: str) -> nn.Module:
    mod = root
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def bucket_of(t: float, edges: Sequence[float]) -> int:
    """Map a timestep to a bucket index. ``edges`` are ascending cut points.

    With edges (334, 667) and the SD training range [0, 1000):
        bucket 0 = [0, 334)    late   / texture
        bucket 1 = [334, 667)  middle / structure
        bucket 2 = [667, 1000) early  / composition
    """
    b = 0
    for e in edges:
        if t >= e:
            b += 1
    return b


@dataclass
class CaptureConfig:
    sites: Sequence[str] = (DEFAULT_SITE,)
    bucket_edges: Sequence[float] = (334.0, 667.0)
    cond_only: bool = True
    # Spatial tokens per forward pass are many (e.g. 32x32 = 1024). Keeping all
    # of them explodes storage for little benefit, since neighbouring tokens are
    # highly correlated. Subsample instead.
    tokens_per_pass: int = 64
    dtype: torch.dtype = torch.float16
    seed: int = 0


class ActivationCapture:
    """Collects UNet activations, split by site and timestep bucket.

    Usage::

        cap = ActivationCapture(pipe.unet, CaptureConfig())
        cap.attach()
        for prompt in prompts:
            cap.reset()
            pipe(prompt, callback_on_step_end=cap.diffusers_callback)
            shards.extend(cap.pop())
        cap.detach()

    The callback is how we learn the current timestep: hooks fire inside the
    UNet and have no idea what t is, so the sampler has to tell us.
    """

    def __init__(self, unet: nn.Module, cfg: CaptureConfig | None = None):
        self.unet = unet
        self.cfg = cfg or CaptureConfig()
        self._handles: list = []
        self._t: float = 999.0
        self._buf: dict[tuple[str, int], list[torch.Tensor]] = {}
        self._gen = torch.Generator().manual_seed(self.cfg.seed)

    # ------------------------------------------------------------------ setup

    def attach(self) -> "ActivationCapture":
        for site in self.cfg.sites:
            mod = get_module(self.unet, site)
            self._handles.append(
                mod.register_forward_hook(self._make_hook(site))
            )
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        self._buf.clear()

    def set_timestep(self, t) -> None:
        self._t = float(t.item() if torch.is_tensor(t) else t)

    def diffusers_callback(self, pipe, step_index, timestep, callback_kwargs):
        """Hand this to ``callback_on_step_end``.

        Note it fires at the *end* of a step, so it reports the timestep that
        was just used; we set it for the next forward pass. The off-by-one is
        harmless at bucket granularity but worth knowing about.
        """
        self.set_timestep(timestep)
        return callback_kwargs

    # ------------------------------------------------------------------- hook

    def _make_hook(self, site: str) -> Callable:
        def hook(module, args, output):
            x = output[0] if isinstance(output, (tuple, list)) else output
            if x.dim() != 3:                      # expect [B, seq, C]
                return
            if self.cfg.cond_only and x.shape[0] % 2 == 0:
                x = x[x.shape[0] // 2:]           # diffusers order: [uncond, cond]

            b, seq, c = x.shape
            flat = x.reshape(b * seq, c)

            n = min(self.cfg.tokens_per_pass, flat.shape[0])
            idx = torch.randperm(flat.shape[0], generator=self._gen)[:n].to(flat.device)
            sample = flat[idx].detach().to(self.cfg.dtype).cpu()

            key = (site, bucket_of(self._t, self.cfg.bucket_edges))
            self._buf.setdefault(key, []).append(sample)

        return hook

    # ------------------------------------------------------------------ output

    def pop(self) -> dict[tuple[str, int], torch.Tensor]:
        out = {k: torch.cat(v, dim=0) for k, v in self._buf.items() if v}
        self._buf.clear()
        return out


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------


@dataclass
class SteerSpec:
    """One intervention on one SAE feature.

    mode:
      ``scale``   -- multiply the feature's code by ``value`` (0.0 = ablate,
                     2.0 = amplify). Reversible, composable, and the natural
                     operation if you believe the code magnitude is meaningful.
      ``clamp``   -- set the code to ``value`` wherever it is active. Useful for
                     forcing a concept on regardless of the prompt.
      ``project`` -- remove the component of the *raw activation* along the
                     feature's decoder direction, scaled by ``value``. This does
                     not go through the SAE codes at all, so it still works where
                     the SAE reconstructs poorly. This is the erasure workhorse.
    """
    feature: int
    mode: str = "scale"
    value: float = 0.0


class SteeringHook:
    """Applies SAE-space edits to a site's activations during generation.

    The reconstruction-error subtlety, which you will be asked about:

    A naive implementation replaces the activation with the SAE's reconstruction
    of the edited codes, ``x <- decode(edit(encode(x)))``. That silently throws
    away the SAE's reconstruction error -- typically 5-20% of the variance -- on
    *every* token, whether or not you are editing it. Image quality visibly
    degrades and you cannot tell how much of that is your intervention.

    The fix is to edit only the delta and preserve the error term:

        err   = x - decode(encode(x))            # what the SAE cannot express
        x_new = decode(edit(encode(x))) + err

    Now an empty edit list is exactly the identity, which is the property you
    want: any change you observe is attributable to the intervention, not to the
    SAE being imperfect. ``preserve_error=False`` is available so you can measure
    how much this matters -- it makes a good ablation row.
    """

    def __init__(self, sae, specs: Iterable[SteerSpec] = (),
                 preserve_error: bool = True, cond_only: bool = True):
        self.sae = sae
        self.specs = list(specs)
        self.preserve_error = preserve_error
        self.cond_only = cond_only
        self._handle = None

    def attach(self, unet: nn.Module, site: str) -> "SteeringHook":
        mod = get_module(unet, site)
        self._handle = mod.register_forward_hook(self._hook)
        return self

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _edit(self, x: torch.Tensor) -> torch.Tensor:
        dev, dt = x.device, x.dtype
        sae = self.sae.to(dev)
        xf = x.float()

        projections = [s for s in self.specs if s.mode == "project"]
        code_edits = [s for s in self.specs if s.mode != "project"]

        out = xf
        if code_edits:
            codes = sae.encode(xf)
            recon = sae.decode(codes)
            err = xf - recon

            for s in code_edits:
                if s.mode == "scale":
                    codes[..., s.feature] = codes[..., s.feature] * s.value
                elif s.mode == "clamp":
                    active = codes[..., s.feature] > 0
                    codes[..., s.feature] = torch.where(
                        active,
                        torch.full_like(codes[..., s.feature], s.value),
                        codes[..., s.feature],
                    )
                else:
                    raise ValueError(f"unknown steer mode {s.mode}")

            out = sae.decode(codes)
            if self.preserve_error:
                out = out + err

        for s in projections:
            d = torch.nn.functional.normalize(sae.W_dec[s.feature], dim=0)
            out = out - s.value * (out @ d).unsqueeze(-1) * d

        return out.to(dt)

    def _hook(self, module, args, output):
        is_tuple = isinstance(output, (tuple, list))
        x = output[0] if is_tuple else output
        if x.dim() != 3 or not self.specs:
            return output

        if self.cond_only and x.shape[0] % 2 == 0:
            # Edit only the conditional half. Editing the unconditional pass too
            # changes the CFG *direction* rather than the conditional prediction,
            # which is a different (and much blunter) intervention.
            half = x.shape[0] // 2
            edited = torch.cat([x[:half], self._edit(x[half:])], dim=0)
        else:
            edited = self._edit(x)

        if is_tuple:
            return (edited,) + tuple(output[1:])
        return edited
