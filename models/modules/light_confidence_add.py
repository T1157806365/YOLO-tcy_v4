"""Lightweight confidence-aware ADD (LC-ADD) for aligned RGB-T features.

F = R + (1 + max_delta*tanh(spatial_gate(R,T) + channel_logits)) * T.
Spatial gate sees 3 bounded/reasonably-scaled statistics:
  - relative TIR magnitude
  - normalized RGB/TIR difference
  - cosine similarity
The gate starts EXACTLY at zero; therefore F_init == R+T.
No learned full-channel convolutions or product-residual branch.
"""
from __future__ import annotations
import torch
from torch import nn


class LightConfidenceAdd(nn.Module):
    def __init__(self, rgb_channels: int, tir_channels: int,
                 max_delta: float = 0.25, kernel_size: int = 3,
                 channel_gate: bool = True, eps: float = 1e-6):
        super().__init__()
        c = int(rgb_channels)
        if c != int(tir_channels):
            raise ValueError(f"LC-ADD requires same channels, got RGB={c}, TIR={tir_channels}")
        if not 0.0 < float(max_delta) <= 1.0:
            raise ValueError("max_delta must be in (0, 1]")
        if int(kernel_size) not in (1, 3):
            raise ValueError("kernel_size must be 1 or 3")
        self.channels = c
        self.max_delta = float(max_delta)
        self.eps = float(eps)
        k = int(kernel_size)
        # Just one spatial output channel, three scale-invariant input maps.
        self.spatial_gate = nn.Conv2d(3, 1, k, stride=1, padding=k//2, bias=True)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        # Independent channel refinements; zero-init also preserves ADD.
        self.channel_logits = nn.Parameter(torch.zeros(c)) if channel_gate else None

    def forward(self, rgb: torch.Tensor, aligned_tir: torch.Tensor) -> torch.Tensor:
        if rgb.shape != aligned_tir.shape:
            raise ValueError(f"RGB/TIR shape mismatch: {rgb.shape} vs {aligned_tir.shape}")
        if rgb.ndim != 4 or int(rgb.shape[1]) != self.channels:
            raise ValueError(f"Expected BCHW channels={self.channels}, got {rgb.shape}")

        # Calculate stable spatial statistics in fp32 even under AMP.
        # The tensor reduction is cheap relative to the 512-channel backbone.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            r = rgb.float()
            t = aligned_tir.float()
            mag_r = r.abs().mean(dim=1, keepdim=True)
            mag_t = t.abs().mean(dim=1, keepdim=True)
            sum_mag = mag_r + mag_t + self.eps
            relative_t = mag_t / sum_mag
            disagreement = (r-t).abs().mean(dim=1, keepdim=True) / sum_mag
            dot = (r*t).sum(dim=1,keepdim=True)
            norm_r = torch.linalg.vector_norm(r,ord=2,dim=1,keepdim=True)
            norm_t = torch.linalg.vector_norm(t,ord=2,dim=1,keepdim=True)
            similarity = (dot/(norm_r*norm_t+self.eps)).clamp(-1.0,1.0)
            stats = torch.cat((relative_t, disagreement, similarity), dim=1)
            # Clamp only abnormal spikes to keep the gate's input bounded.
            stats = stats.clamp(-2.0, 2.0)
            spatial = self.spatial_gate(stats)
            if self.channel_logits is not None:
                logits = spatial + self.channel_logits.reshape(1,-1,1,1)
            else:
                logits = spatial
            tir_weight = 1.0 + self.max_delta * torch.tanh(logits)
            # Exact ADD at initialized zero weights/bias/channel logits.
            out = r + tir_weight * t
        return out.to(dtype=rgb.dtype)


__all__ = ["LightConfidenceAdd"]
