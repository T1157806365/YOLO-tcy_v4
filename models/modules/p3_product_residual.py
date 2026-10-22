#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 lightweight cross-modal product/conv residual fusion.

This module is intentionally independent of the original GMLS/DMLS/BHLR code.
Both ablations have identical trainable parameter counts:
  mode='conv'    -> [R, T, 0] (Conv + residual, no multiplication)
  mode='product' -> [R, T, R*T] (Conv + product + residual)

Output identity at initialization: gamma_raw == 0 -> output == preexisting ADD.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class P3ProductResidual(nn.Module):
    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        hidden_channels: int = 32,
        mode: str = "product",
        max_gain: float = 0.25,
    ):
        super().__init__()
        self.rgb_channels = int(rgb_channels)
        self.tir_channels = int(tir_channels)
        self.hidden_channels = int(hidden_channels)
        self.mode = str(mode).strip().lower()
        self.max_gain = float(max_gain)

        if self.rgb_channels <= 0 or self.tir_channels <= 0:
            raise ValueError("Feature channel counts must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.mode not in ("conv", "product"):
            raise ValueError(f"Unknown P3 residual mode: {mode!r}")
        if self.max_gain <= 0.0 or self.max_gain > 1.0:
            raise ValueError("max_gain must lie in (0, 1]")

        def projector(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, self.hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_channels),
                nn.SiLU(inplace=True),
            )

        self.rgb_proj = projector(self.rgb_channels)
        self.tir_proj = projector(self.tir_channels)

        self.interaction = nn.Sequential(
            # Spatial modeling is depthwise to keep parameter count low.
            nn.Conv2d(
                self.hidden_channels * 3,
                self.hidden_channels * 3,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.hidden_channels * 3,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_channels * 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                self.hidden_channels * 3,
                self.rgb_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        # Gamma=0 is EXACT identity to the preexisting ADD fusion.
        # A bounded gain prevents a newly added module from overwhelming P3.
        self.gamma_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(
        self,
        fused_add: torch.Tensor,
        rgb_p3: torch.Tensor,
        tir_p3_aligned: torch.Tensor,
    ) -> torch.Tensor:
        if fused_add.ndim != 4 or rgb_p3.ndim != 4 or tir_p3_aligned.ndim != 4:
            raise ValueError("P3 features must be NCHW tensors")
        if fused_add.shape != rgb_p3.shape:
            raise ValueError(
                f"P3 residual expects ADD feature to match RGB: "
                f"{tuple(fused_add.shape)} != {tuple(rgb_p3.shape)}"
            )
        if rgb_p3.shape[1] != self.rgb_channels:
            raise ValueError("Unexpected RGB P3 channel count")
        if tir_p3_aligned.shape[1] != self.tir_channels:
            raise ValueError("Unexpected TIR P3 channel count")

        if tir_p3_aligned.shape[-2:] != rgb_p3.shape[-2:]:
            tir_p3_aligned = F.interpolate(
                tir_p3_aligned,
                size=rgb_p3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        r = self.rgb_proj(rgb_p3)
        t = self.tir_proj(tir_p3_aligned)
        interaction = r * t if self.mode == "product" else torch.zeros_like(r)
        detail = self.interaction(torch.cat((r, t, interaction), dim=1))
        gain = self.max_gain * torch.tanh(self.gamma_raw)
        return fused_add + gain * detail

    @torch.no_grad()
    def scalar_state(self) -> dict:
        return {
            "p3_residual_gain": float((self.max_gain * self.gamma_raw.tanh()).item()),
            "p3_residual_mode": self.mode,
        }


__all__ = ["P3ProductResidual"]
