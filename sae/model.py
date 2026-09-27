"""Sparse autoencoders for diffusion-model activations.

Two variants, because the choice between them is a real design decision you
should be able to defend:

  * ``l1``   -- ReLU encoder + L1 penalty on the codes. The classic formulation
                (Bricken et al. 2023). Its known pathology is *shrinkage*: the
                L1 term penalises the magnitude of every active code, so the
                optimum systematically under-estimates feature activations even
                when the feature is unambiguously present. You also have to tune
                lambda to hit a target sparsity, and the mapping from lambda to
                L0 drifts during training.

  * ``topk`` -- keep exactly k codes per token, zero the rest (Gao et al. 2024).
                Sparsity is now an exact constraint rather than a penalty, so
                there is no shrinkage and no lambda search: you *set* L0 = k.
                The cost is a non-differentiable selection step (fine in
                practice -- gradients flow through the surviving k) and a
                stronger tendency to produce dead latents, which is why the
                auxiliary loss below exists.

Default is ``topk``. If an interviewer asks why, the answer is: exact sparsity
control makes the reconstruction/sparsity trade-off curve interpretable, and
removing shrinkage matters a lot when you intend to *use* the code magnitudes
for steering rather than only for analysis.

Two structural details that are easy to miss and both matter:

  1. ``b_dec`` is subtracted from the input before encoding ("pre-encoder
     bias"). The decoder bias models the mean of the activation distribution;
     subtracting it first means the encoder sees centred data and the features
     describe deviations from the mean rather than wasting capacity on it.

  2. Decoder columns are constrained to unit norm. Without this, an L1 model can
     cheat -- halve every code and double the corresponding decoder column, and
     the reconstruction is unchanged while the L1 penalty drops. The constraint
     removes that degree of freedom and makes code magnitudes comparable across
     features (which steering depends on).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int                                  # activation width being modelled
    expansion: int = 16                        # d_hidden = d_in * expansion
    mode: Literal["topk", "l1"] = "topk"

    # topk
    k: int = 32                                # exact L0 per token
    k_aux: int = 256                           # dead latents used by the aux loss
    aux_coef: float = 1.0 / 32.0               # weight on the aux reconstruction
    dead_after_tokens: int = 10_000_000        # no fire in this many tokens => dead

    # l1
    l1_coef: float = 5e-4

    # shared
    normalize_decoder: bool = True
    tied_init: bool = True                     # init W_enc as W_dec.T
    seed: int = 0

    @property
    def d_hidden(self) -> int:
        return self.d_in * self.expansion

    def to_dict(self) -> dict:
        d = asdict(self)
        d["d_hidden"] = self.d_hidden
        return d


@dataclass
class SAEOutput:
    x_hat: torch.Tensor
    codes: torch.Tensor          # post-activation, post-sparsification
    pre_acts: torch.Tensor       # pre-activation encoder output (for diagnostics)
    loss: torch.Tensor
    recon_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    aux_loss: torch.Tensor
    metrics: dict = field(default_factory=dict)


class SparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)

        d_in, d_hidden = cfg.d_in, cfg.d_hidden

        # Kaiming-ish init on the decoder, then optionally tie the encoder to it.
        w_dec = torch.randn(d_hidden, d_in, generator=g) / math.sqrt(d_in)
        w_dec = F.normalize(w_dec, dim=1)
        self.W_dec = nn.Parameter(w_dec)                       # [d_hidden, d_in]
        self.W_enc = nn.Parameter(
            w_dec.T.clone() if cfg.tied_init
            else torch.randn(d_in, d_hidden, generator=g) / math.sqrt(d_hidden)
        )                                                      # [d_in, d_hidden]
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Dead-latent bookkeeping. Counts tokens since each latent last fired.
        self.register_buffer("tokens_since_fired", torch.zeros(d_hidden, dtype=torch.long))

    # ------------------------------------------------------------------ utils

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Project decoder columns back to the unit sphere. Call after each step."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def set_decoder_norm_grad_(self) -> None:
        """Remove the component of the decoder gradient that points along the
        decoder columns.

        Renormalising after an update silently discards part of the step, which
        makes Adam's second-moment estimates wrong for those directions. Removing
        the parallel component *before* the optimiser sees the gradient keeps the
        update tangent to the sphere, so the renormalisation is a no-op to first
        order. Skipping this still trains, just less stably.
        """
        if self.W_dec.grad is None:
            return
        w = self.W_dec.data
        g = self.W_dec.grad
        parallel = (g * w).sum(dim=1, keepdim=True) * w
        self.W_dec.grad = g - parallel

    @property
    def dead_mask(self) -> torch.Tensor:
        return self.tokens_since_fired > self.cfg.dead_after_tokens

    # -------------------------------------------------------------- encode/dec

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activation encoder output, with the pre-encoder bias applied."""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def _sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        if self.cfg.mode == "topk":
            # Exact L0 = k. Selection happens on the pre-activations; we then
            # ReLU so that a negative value selected by topk contributes zero
            # rather than a negative code (codes must be non-negative for the
            # decoder columns to be interpretable as directions).
            vals, idx = torch.topk(pre, self.cfg.k, dim=-1)
            codes = torch.zeros_like(pre)
            codes.scatter_(-1, idx, F.relu(vals))
            return codes
        return F.relu(pre)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._sparsify(self.encode_pre(x))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes @ self.W_dec + self.b_dec

    # -------------------------------------------------------------------- loss

    def _aux_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                  pre: torch.Tensor) -> torch.Tensor:
        """Make dead latents reconstruct the residual.

        TopK models kill latents permanently: once a latent stops being in the
        top-k it receives no gradient and can never come back. The fix (Gao et
        al.) is to take the k_aux highest-scoring *dead* latents, ask them to
        reconstruct whatever the live latents missed, and add that to the loss.
        Dead latents therefore get gradient signal pointed at genuinely
        unexplained structure, which is exactly where a new feature should go.
        """
        dead = self.dead_mask
        n_dead = int(dead.sum())
        if n_dead == 0:
            return x.new_zeros(())

        k_aux = min(self.cfg.k_aux, n_dead)
        pre_dead = pre.masked_fill(~dead, float("-inf"))
        vals, idx = torch.topk(pre_dead, k_aux, dim=-1)
        codes = torch.zeros_like(pre)
        codes.scatter_(-1, idx, F.relu(vals))

        residual = (x - x_hat).detach()
        residual_hat = codes @ self.W_dec           # note: no b_dec, this models the *residual*
        return F.mse_loss(residual_hat, residual)

    def forward(self, x: torch.Tensor, update_stats: bool = True) -> SAEOutput:
        pre = self.encode_pre(x)
        codes = self._sparsify(pre)
        x_hat = self.decode(codes)

        recon = F.mse_loss(x_hat, x)

        if self.cfg.mode == "l1":
            # With unit-norm decoder columns the plain L1 of the codes is already
            # the right penalty; if you disable normalisation you must weight each
            # code by its decoder norm or the model games the penalty.
            if self.cfg.normalize_decoder:
                sparsity = codes.abs().sum(dim=-1).mean()
            else:
                sparsity = (codes.abs() * self.W_dec.norm(dim=1)).sum(dim=-1).mean()
            sparsity = self.cfg.l1_coef * sparsity
            aux = x.new_zeros(())
        else:
            sparsity = x.new_zeros(())
            aux = self.cfg.aux_coef * self._aux_loss(x, x_hat, pre)

        loss = recon + sparsity + aux

        if update_stats:
            self._update_fire_stats(codes)

        with torch.no_grad():
            l0 = (codes > 0).float().sum(dim=-1).mean()
            var = x.var(dim=0, unbiased=False).sum().clamp_min(1e-8)
            ev = 1.0 - (x - x_hat).var(dim=0, unbiased=False).sum() / var
            metrics = {
                "l0": l0.item(),
                "explained_variance": ev.item(),
                "dead_frac": self.dead_mask.float().mean().item(),
                "recon_mse": recon.item(),
            }

        return SAEOutput(x_hat, codes, pre, loss, recon, sparsity, aux, metrics)

    @torch.no_grad()
    def _update_fire_stats(self, codes: torch.Tensor) -> None:
        n_tokens = codes.shape[0]
        fired = (codes > 0).any(dim=0)
        self.tokens_since_fired += n_tokens
        self.tokens_since_fired[fired] = 0

    # ------------------------------------------------------------------- io

    def save(self, path: str) -> None:
        torch.save({"cfg": self.cfg.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "SparseAutoencoder":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        cfg_d = dict(blob["cfg"])
        cfg_d.pop("d_hidden", None)
        model = cls(SAEConfig(**cfg_d))
        model.load_state_dict(blob["state_dict"])
        return model
