"""AGDH-v1: alignment-guided detection-head P3 residual modulation.

Input:
    neck_p3: [B,C,H,W]  -- actual Detect P3 input, *not* backbone P3
    support_3ch: [B,3,h,w] -- [RGB support, aligned-TIR support, joint support]

Zero-initialized output ensures exact identity at initialization. No GT is read
in forward() and no location is masked out (independent detection is preserved).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AGDHP3(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 16,
        max_gain: float = 0.20,
        detach_alignment_prior: bool = True,
        union_guard: bool = True,
        prior_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.max_gain = float(max_gain)
        self.detach_alignment_prior = bool(detach_alignment_prior)
        self.union_guard = bool(union_guard)
        self.prior_dropout = float(prior_dropout)
        if self.hidden_channels < 4:
            raise ValueError('hidden_channels must be >= 4')
        if not (0.0 <= self.max_gain <= 1.0):
            raise ValueError('max_gain must be in [0,1]')
        if not (0.0 <= self.prior_dropout < 1.0):
            raise ValueError('prior_dropout must be in [0,1)')

        # One channel of P3 context avoids assumptions about YOLO m/s channel
        # widths or the Detect internal layer implementation.
        # Channels: context, RGB, aligned TIR, Joint, soft-union, disagreement.
        self.stem = nn.Sequential(
            nn.Conv2d(6, self.hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                groups=self.hidden_channels,
                bias=True,
            ),
            nn.SiLU(inplace=False),
        )
        self.out = nn.Conv2d(self.hidden_channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, neck_p3: torch.Tensor, support_3ch: torch.Tensor,
                return_debug: bool = False):
        if neck_p3.ndim != 4:
            raise ValueError(f'neck_p3 must be [B,C,H,W], got {neck_p3.shape}')
        if support_3ch is None or support_3ch.ndim != 4 or support_3ch.shape[1] != 3:
            raise ValueError('support_3ch must be [B,3,H,W]')
        if support_3ch.shape[0] != neck_p3.shape[0]:
            raise ValueError('AGDH batch mismatch')

        priors = support_3ch.detach() if self.detach_alignment_prior else support_3ch
        priors = priors.to(device=neck_p3.device, dtype=neck_p3.dtype)
        if tuple(priors.shape[-2:]) != tuple(neck_p3.shape[-2:]):
            priors = F.interpolate(
                priors, size=neck_p3.shape[-2:], mode='bilinear', align_corners=False
            )
        priors = priors.clamp(0.0, 1.0)
        if self.training and self.prior_dropout > 0.0:
            keep = (
                torch.rand(priors.size(0), 1, 1, 1, device=priors.device)
                >= self.prior_dropout
            ).to(priors.dtype)
            priors = priors * keep

        rgb, tir, joint = priors.split(1, dim=1)
        union = (rgb + tir - rgb * tir).clamp(0.0, 1.0)
        disagreement = torch.abs(rgb - tir)
        context = neck_p3.mean(dim=1, keepdim=True)
        cue = torch.cat((context, rgb, tir, joint, union, disagreement), dim=1)
        logits = self.out(self.stem(cue))
        gate = torch.tanh(logits) * self.max_gain
        if self.union_guard:
            gate = gate * union
        enhanced = neck_p3 * (1.0 + gate)

        if not return_debug:
            return enhanced
        return enhanced, {
            'rgb_support': rgb,
            'tir_support': tir,
            'joint_support': joint,
            'union_support': union,
            'disagreement': disagreement,
            'raw_gate_logits': logits,
            'residual_gate': gate,
            'feature_multiplier': 1.0 + gate,
        }


__all__ = ['AGDHP3']
