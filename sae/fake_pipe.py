"""A tiny stand-in for StableDiffusionPipeline, used only by tests.

It reproduces exactly the properties the capture code depends on, and nothing
else:

* the module path ``up_blocks.1.attentions.1.transformer_blocks.0.attn2``
* a UNet called as ``unet(sample, timestep, ...)`` once per sampler step, so the
  timestep pre-hook sees real timestep values
* a CFG-doubled batch ``[uncond, cond]``
* the 25-step SD1.5 DDIM timestep grid
* deterministic outputs for a given generator seed

It also provides just enough of ``vae``, ``tokenizer``, ``text_encoder`` and
``scheduler.add_noise`` for the substitution stage, returns small PIL images
for the atlas and steering stages, and ``FakeClipScorer`` (``--clip __fake__``)
stands in for CLIP -- so the orchestrator can run all eleven stages end to end
on a CPU in under a minute.

Selected with ``--model __fake__``. Setting ``SAE_FAKE_CRASH_AFTER=N`` makes the
pipeline raise on its (N+1)-th call, which is how the tests simulate a session
being killed part-way through a capture.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        return torch.tanh(self.proj(x))


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn2 = _Attn(d)

    def forward(self, x):
        return x + self.attn2(x)


class _Transformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(d)])

    def forward(self, x):
        return self.transformer_blocks[0](x)


class _UpBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attentions = nn.ModuleList([_Transformer(d), _Transformer(d)])

    def forward(self, x):
        for a in self.attentions:
            x = a(x)
        return x


class FakeUNet(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.up_blocks = nn.ModuleList([_UpBlock(d), _UpBlock(d)])
        self.temb = nn.Linear(1, d)

    def forward(self, sample, timestep, encoder_hidden_states=None):
        # Callers may hand over CUDA tensors on a GPU host; the toy runs on CPU.
        t = torch.as_tensor(timestep).detach().float().cpu().reshape(-1)[:1] / 1000.0
        x = sample.float().cpu() + self.temb(t.reshape(1, 1))
        for b in self.up_blocks:
            x = b(x)
        return SimpleNamespace(sample=x)


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=1000)

    def add_noise(self, latents, noise, t):
        a = (torch.as_tensor(t).detach().float().cpu() / 1000.0).reshape(-1, 1, 1)
        return latents.cpu() * (1 - a).sqrt() + noise.cpu() * a.sqrt()

    @staticmethod
    def grid(n):
        return [int(1000 / n * (n - 1 - i)) + 1 for i in range(n)]


class _FakeVAE:
    """Maps an image [B,3,H,W] to a latent [B, seq, d] the fake UNet accepts."""

    def __init__(self, d, seq):
        g = torch.Generator().manual_seed(7)
        self.side = int(round(seq ** 0.5))
        self.proj = torch.randn(3, d, generator=g) / 3 ** 0.5
        self.dtype = torch.float32
        self.config = SimpleNamespace(scaling_factor=0.18215)

    def encode(self, x):
        x = x.detach().float().cpu()
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.side, self.side))
        z = pooled.flatten(2).transpose(1, 2) @ self.proj          # [B, seq, d]
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=z))


class _FakeTokenizer:
    model_max_length = 77

    def __call__(self, texts, **kw):
        n = len(texts) if isinstance(texts, (list, tuple)) else 1
        return SimpleNamespace(input_ids=torch.zeros(n, 77, dtype=torch.long))


class _FakeTextEncoder:
    def __call__(self, ids):
        return (torch.zeros(ids.shape[0], 77, 8),)


class FakeClipScorer:
    """Deterministic stand-in for CLIP similarity: depends on the image's pixels
    and the text, so steering that changes the image changes the score."""

    @torch.no_grad()
    def similarity(self, images, texts):
        import zlib
        import numpy as np
        out = []
        for im, t in zip(images, texts):
            m = float(np.asarray(im, dtype=np.float32).mean()) / 255.0
            h = (zlib.crc32(t.encode()) % 1000) / 1000.0
            out.append(0.2 + 0.05 * np.cos(6.0 * m + 3.0 * h))
        return torch.tensor(out, dtype=torch.float32)


class FakePipe:
    def __init__(self, d: int = 64, seq: int = 16, seed: int = 123):
        torch.manual_seed(seed)
        self.d, self.seq = d, seq
        self.unet = FakeUNet(d).eval()
        self.scheduler = FakeScheduler()
        self.vae = _FakeVAE(d, seq)
        self.tokenizer = _FakeTokenizer()
        self.text_encoder = _FakeTextEncoder()
        self.device = torch.device("cpu")
        self._calls = 0
        self._crash_after = int(os.environ.get("SAE_FAKE_CRASH_AFTER", "-1"))

    def _prompt_vec(self, prompt: str) -> torch.Tensor:
        import zlib
        g = torch.Generator().manual_seed(zlib.crc32(prompt.encode()))
        v = torch.zeros(self.d)
        idx = torch.randperm(self.d, generator=g)[:4]
        v[idx] = 1.5
        return v

    def set_progress_bar_config(self, **kw):
        pass

    def to(self, *a, **k):
        return self

    @torch.no_grad()
    def __call__(self, prompts, num_inference_steps=25, guidance_scale=7.5,
                 height=512, width=512, generator=None, output_type="pil", **kw):
        if self._crash_after >= 0 and self._calls >= self._crash_after:
            raise RuntimeError("simulated session kill (SAE_FAKE_CRASH_AFTER)")
        self._calls += 1
        if isinstance(prompts, str):
            prompts = [prompts]
        b = len(prompts)
        # Draw noise on whatever device the caller's generator lives on -- the
        # capture script makes a CUDA generator whenever a GPU is visible -- then
        # run the toy UNet on CPU, where results are bit-deterministic.
        dev = generator.device if generator is not None else torch.device("cpu")
        x = torch.randn(2 * b, self.seq, self.d, generator=generator, device=dev).cpu()
        # Make the conditional half depend on the prompt (deterministically --
        # crc32, not hash(), which is salted per process). Without this every
        # prompt with the same seed gave identical activations, so the atlas had
        # no selective features to draw once the site moved to the residual stream.
        cond = torch.stack([self._prompt_vec(p) for p in prompts])       # [b, d]
        for t in self.scheduler.grid(num_inference_steps):
            x[b:] = x[b:] + cond[:, None, :]
            x = self.unet(x, torch.tensor(t)).sample * 0.5
        if output_type == "latent":
            return SimpleNamespace(images=x[b:])
        from PIL import Image
        import numpy as np
        side = int(round(self.seq ** 0.5))
        imgs = []
        for k in range(b):
            v = x[b + k, :, :3].reshape(side, side, 3)                  # cond half
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            arr = (v.numpy() * 255).astype(np.uint8)
            imgs.append(Image.fromarray(arr).resize((64, 64), Image.NEAREST))
        return SimpleNamespace(images=imgs)
