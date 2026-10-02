"""A tiny stand-in for StableDiffusionPipeline, used only by tests.

It reproduces exactly the properties the capture code depends on, and nothing
else:

* the module path ``up_blocks.1.attentions.1.transformer_blocks.0.attn2``
* a UNet called as ``unet(sample, timestep, ...)`` once per sampler step, so the
  timestep pre-hook sees real timestep values
* a CFG-doubled batch ``[uncond, cond]``
* the 25-step SD1.5 DDIM timestep grid
* deterministic outputs for a given generator seed

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
        t = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1)[:1] / 1000.0
        x = sample + self.temb(t.reshape(1, 1))
        for b in self.up_blocks:
            x = b(x)
        return SimpleNamespace(sample=x)


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=1000)

    @staticmethod
    def grid(n):
        return [int(1000 / n * (n - 1 - i)) + 1 for i in range(n)]


class FakePipe:
    def __init__(self, d: int = 64, seq: int = 16, seed: int = 123):
        torch.manual_seed(seed)
        self.d, self.seq = d, seq
        self.unet = FakeUNet(d).eval()
        self.scheduler = FakeScheduler()
        self.device = torch.device("cpu")
        self._calls = 0
        self._crash_after = int(os.environ.get("SAE_FAKE_CRASH_AFTER", "-1"))

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
        for t in self.scheduler.grid(num_inference_steps):
            x = self.unet(x, torch.tensor(t)).sample * 0.5
        return SimpleNamespace(images=[None] * b)
