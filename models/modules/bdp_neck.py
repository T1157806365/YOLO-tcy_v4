"""
BDP-Neck v1
===========
BHLR Detail-Preserving Neck for RGB-T UAV detection.

BHLR injects high-resolution RGB detail into backbone P3. Standard YOLO
FPN/PAN can dilute that increment during repeated multi-scale interactions.

BDP-Neck preserves only the increment actually introduced by BHLR:

    D3 = P3_BHLR - P3_raw

and injects it into the final Neck P3 immediately before Detect:

    P3_out = P3_neck + gain * Gate(P3_neck) * Proj(D3)

Properties
----------
- P3 only.
- No auxiliary loss.
- Optional semantic spatial gate.
- Bounded residual gain.
- gain_raw is initialized to 0 -> exact identity at initialization.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BDPNeckP3(nn.Module):
    """Semantic-guided P3 detail-preservation residual."""

    def __init__(
        self,
        detail_channels: int,
        neck_channels: int,
        semantic_gate: bool = True,
        max_gain: float = 0.25,
    ):
        super().__init__()

        detail_channels = int(detail_channels)
        neck_channels = int(neck_channels)
        max_gain = float(max_gain)

        if detail_channels <= 0:
            raise ValueError(f"detail_channels must be > 0, got {detail_channels}")
        if neck_channels <= 0:
            raise ValueError(f"neck_channels must be > 0, got {neck_channels}")
        if max_gain <= 0:
            raise ValueError(f"max_gain must be > 0, got {max_gain}")

        self.detail_channels = detail_channels
        self.neck_channels = neck_channels
        self.semantic_gate_enabled = bool(semantic_gate)
        self.max_gain = max_gain

        # BHLR P3 channels -> final Neck P3 channels.
        self.detail_proj = nn.Conv2d(
            detail_channels,
            neck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Semantic Neck P3 decides WHERE high-resolution detail is useful.
        if self.semantic_gate_enabled:
            self.semantic_gate = nn.Conv2d(
                neck_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            # Initial gate = sigmoid(0) = 0.5.
            nn.init.zeros_(self.semantic_gate.weight)
            nn.init.zeros_(self.semantic_gate.bias)
        else:
            self.semantic_gate = None

        # Exact identity initialization:
        # gain = max_gain * tanh(0) = 0.
        self.gain_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def current_gain(self) -> torch.Tensor:
        return self.max_gain * torch.tanh(self.gain_raw)

    def forward(
        self,
        detail_p3: torch.Tensor,
        neck_p3: torch.Tensor,
        return_debug: bool = False,
    ):
        if detail_p3.ndim != 4 or neck_p3.ndim != 4:
            raise ValueError(
                "BDPNeckP3 expects BCHW tensors, "
                f"got detail={tuple(detail_p3.shape)}, neck={tuple(neck_p3.shape)}"
            )

        if int(detail_p3.shape[1]) != self.detail_channels:
            raise ValueError(
                "Unexpected BHLR detail channels: "
                f"got={int(detail_p3.shape[1])}, expected={self.detail_channels}"
            )

        if int(neck_p3.shape[1]) != self.neck_channels:
            raise ValueError(
                "Unexpected Neck P3 channels: "
                f"got={int(neck_p3.shape[1])}, expected={self.neck_channels}"
            )

        if detail_p3.shape[-2:] != neck_p3.shape[-2:]:
            detail_p3 = F.interpolate(
                detail_p3,
                size=neck_p3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        projected_detail = self.detail_proj(detail_p3)

        if self.semantic_gate is not None:
            gate = torch.sigmoid(self.semantic_gate(neck_p3))
        else:
            gate = torch.ones(
                neck_p3.shape[0],
                1,
                neck_p3.shape[2],
                neck_p3.shape[3],
                device=neck_p3.device,
                dtype=neck_p3.dtype,
            )

        gain = self.current_gain().to(
            device=neck_p3.device,
            dtype=neck_p3.dtype,
        )

        injection = gain * gate * projected_detail
        out = neck_p3 + injection

        if not return_debug:
            return out

        debug: Dict[str, torch.Tensor] = {
            "detail_input": detail_p3,
            "detail_projected": projected_detail,
            "semantic_gate": gate,
            "injection": injection,
            "neck_p3_before": neck_p3,
            "neck_p3_after": out,
            "gain": gain.reshape(1),
        }
        return out, debug


__all__ = ["BDPNeckP3"]
