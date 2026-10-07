# models/modules/adaptive_rgbt_fusion.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(
            self.bn(
                self.conv(x)
            )
        )


class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive Gated RGB-T Fusion
    ===========================

    Input:
        RGB feature R
        aligned TIR feature T

    Gate input:
        [R, T, |R-T|]

    Spatial modality weights:
        w_R = 1 + tanh(l_R)
        w_T = 1 + tanh(l_T)

    Fusion:
        F = w_R * R + w_T * T

    The last gate layer is initialized to zero:

        l_R = l_T = 0
        w_R = w_T = 1

    Therefore initialization is EXACTLY equivalent to ADD:

        F = R + T
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        out_channels: int | None = None,
        align_mode: str = "bilinear",
        hidden_ratio: float = 0.25,
        hidden_min: int = 16,
    ):
        super().__init__()

        self.rgb_channels = int(rgb_channels)
        self.tir_channels = int(tir_channels)

        self.out_channels = (
            self.rgb_channels
            if out_channels is None
            else int(out_channels)
        )

        self.align_mode = str(
            align_mode
        ).lower()

        if self.out_channels != self.rgb_channels:
            raise ValueError(
                "AdaptiveGatedFusion currently keeps RGB channel "
                f"dimension unchanged: out_channels should be "
                f"{self.rgb_channels}, got {self.out_channels}"
            )

        # -----------------------------------------------------
        # TIR channel projection
        # -----------------------------------------------------

        if self.tir_channels == self.rgb_channels:

            self.tir_proj = nn.Identity()

        else:

            self.tir_proj = ConvBNAct(
                self.tir_channels,
                self.rgb_channels,
                k=1,
            )

        hidden_channels = max(
            int(hidden_min),
            int(
                round(
                    self.rgb_channels
                    * float(hidden_ratio)
                )
            ),
        )

        # -----------------------------------------------------
        # Adaptive gate
        #
        # [R, T, |R-T|]
        #       ↓
        #  1x1 projection
        #       ↓
        # DW 3x3 context
        #       ↓
        # RGB/TIR spatial logits
        # -----------------------------------------------------

        self.gate_reduce = ConvBNAct(
            self.rgb_channels * 3,
            hidden_channels,
            k=1,
        )

        self.gate_context = ConvBNAct(
            hidden_channels,
            hidden_channels,
            k=3,
            groups=hidden_channels,
        )

        self.gate_out = nn.Conv2d(
            hidden_channels,
            2,
            kernel_size=1,
            bias=True,
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # initial AGF == ADD
        # -----------------------------------------------------

        nn.init.zeros_(
            self.gate_out.weight
        )

        nn.init.zeros_(
            self.gate_out.bias
        )

    def _resize_tir(
        self,
        tir,
        rgb,
    ):
        if (
            tir.shape[-2:]
            == rgb.shape[-2:]
        ):
            return tir

        if self.align_mode in {
            "bilinear",
            "bicubic",
        }:

            return F.interpolate(
                tir,
                size=rgb.shape[-2:],
                mode=self.align_mode,
                align_corners=False,
            )

        return F.interpolate(
            tir,
            size=rgb.shape[-2:],
            mode=self.align_mode,
        )

    def forward(
        self,
        rgb,
        tir,
    ):
        # -----------------------------------------------------
        # Spatial alignment of tensor size
        # -----------------------------------------------------

        tir = self._resize_tir(
            tir,
            rgb,
        )

        # -----------------------------------------------------
        # Channel alignment
        # -----------------------------------------------------

        tir = self.tir_proj(
            tir
        )

        # -----------------------------------------------------
        # Build modality relation
        # -----------------------------------------------------

        relation = torch.cat(
            [
                rgb,
                tir,
                torch.abs(
                    rgb - tir
                ),
            ],
            dim=1,
        )

        gate_feature = (
            self.gate_reduce(
                relation
            )
        )

        gate_feature = (
            self.gate_context(
                gate_feature
            )
        )

        logits = self.gate_out(
            gate_feature
        )

        rgb_logit = (
            logits[:, 0:1]
        )

        tir_logit = (
            logits[:, 1:2]
        )

        # -----------------------------------------------------
        # Residual adaptive weights
        #
        # range: (0, 2)
        #
        # initialization:
        #     weight = 1
        # -----------------------------------------------------

        rgb_weight = (
            1.0
            + torch.tanh(
                rgb_logit
            )
        )

        tir_weight = (
            1.0
            + torch.tanh(
                tir_logit
            )
        )

        # -----------------------------------------------------
        # Adaptive fusion
        # -----------------------------------------------------

        fused = (
            rgb_weight * rgb
            + tir_weight * tir
        )

        return fused


if __name__ == "__main__":

    torch.manual_seed(0)

    rgb = torch.randn(
        2,
        128,
        80,
        80,
    )

    tir = torch.randn(
        2,
        128,
        80,
        80,
    )

    model = AdaptiveGatedFusion(
        rgb_channels=128,
        tir_channels=128,
        out_channels=128,
    )

    model.eval()

    with torch.no_grad():

        output = model(
            rgb,
            tir,
        )

    error = (
        output
        - (
            rgb
            + tir
        )
    ).abs().max().item()

    print(
        "AGF initial ADD-equivalence "
        f"max error = {error:.10f}"
    )

    assert error < 1e-7

    print(
        "AdaptiveGatedFusion test: PASS"
    )