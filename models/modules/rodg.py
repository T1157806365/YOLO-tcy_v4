"""
Regression-Oriented Detail Guidance (RODG-v1)
=============================================

Use the high-resolution RGB detail increment only for P3 bounding-box
regression, while leaving classification untouched.

    D3 = P3_BHLR - P3_raw
    F_reg = F_P3 + gain * Gate(F_P3, D3) * Proj(D3)
    F_cls = F_P3

The residual gain is initialized to zero, so the module is an exact identity
at initialization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionOrientedDetailGuidance(nn.Module):
    def __init__(
        self,
        detail_channels: int,
        neck_channels: int,
        hidden_channels: int = 32,
        spatial_gate: bool = True,
        max_gain: float = 0.25,
    ):
        super().__init__()
        detail_channels = int(detail_channels)
        neck_channels = int(neck_channels)
        hidden_channels = int(hidden_channels)
        max_gain = float(max_gain)
        if min(detail_channels, neck_channels, hidden_channels) <= 0:
            raise ValueError("RODG channel counts must be > 0")
        if max_gain <= 0:
            raise ValueError(f"max_gain must be > 0, got {max_gain}")

        self.detail_channels = detail_channels
        self.neck_channels = neck_channels
        self.hidden_channels = hidden_channels
        self.spatial_gate = bool(spatial_gate)
        self.max_gain = max_gain

        self.detail_proj = nn.Conv2d(detail_channels, neck_channels, 1, bias=False)

        if self.spatial_gate:
            self.neck_reduce = nn.Conv2d(neck_channels, hidden_channels, 1, bias=False)
            self.detail_reduce = nn.Conv2d(detail_channels, hidden_channels, 1, bias=False)
            self.gate_predictor = nn.Sequential(
                nn.Conv2d(hidden_channels * 3, hidden_channels, 3, padding=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, 1, bias=True),
            )
            nn.init.zeros_(self.gate_predictor[-1].weight)
            nn.init.zeros_(self.gate_predictor[-1].bias)
        else:
            self.neck_reduce = None
            self.detail_reduce = None
            self.gate_predictor = None

        self.gain_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def current_gain(self) -> torch.Tensor:
        return self.max_gain * torch.tanh(self.gain_raw)

    @staticmethod
    def _resize_like(detail: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if detail.shape[-2:] == reference.shape[-2:]:
            return detail
        return F.interpolate(detail, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, detail_p3: torch.Tensor, neck_p3: torch.Tensor, return_debug: bool = False):
        if not torch.is_tensor(detail_p3) or detail_p3.ndim != 4:
            raise TypeError(f"detail_p3 must be BCHW tensor, got {type(detail_p3)}")
        if not torch.is_tensor(neck_p3) or neck_p3.ndim != 4:
            raise TypeError(f"neck_p3 must be BCHW tensor, got {type(neck_p3)}")
        if int(detail_p3.shape[1]) != self.detail_channels:
            raise ValueError(f"RODG detail channels={detail_p3.shape[1]}, expected={self.detail_channels}")
        if int(neck_p3.shape[1]) != self.neck_channels:
            raise ValueError(f"RODG neck channels={neck_p3.shape[1]}, expected={self.neck_channels}")

        detail = self._resize_like(detail_p3, neck_p3)
        projected_detail = self.detail_proj(detail)

        if self.spatial_gate:
            neck_h = self.neck_reduce(neck_p3)
            detail_h = self.detail_reduce(detail)
            relation = torch.cat([neck_h, detail_h, (neck_h - detail_h).abs()], dim=1)
            gate = torch.sigmoid(self.gate_predictor(relation))
        else:
            gate = torch.ones(
                neck_p3.shape[0], 1, neck_p3.shape[2], neck_p3.shape[3],
                device=neck_p3.device, dtype=neck_p3.dtype,
            )

        gain = self.current_gain().to(dtype=neck_p3.dtype)
        injection = gain * gate * projected_detail
        out = neck_p3 + injection

        if not return_debug:
            return out
        return out, {
            "gain": gain.detach(),
            "gain_raw": self.gain_raw.detach(),
            "gate": gate.detach(),
            "projected_detail": projected_detail.detach(),
            "injection": injection.detach(),
            "detail_resized": detail.detach(),
        }


__all__ = ["RegressionOrientedDetailGuidance"]
