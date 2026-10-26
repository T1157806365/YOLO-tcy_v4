"""Figure-style dual-branch Concat -> Conv -> Add, replacing original ADD.

RGB: C([raw_rgb, bhlr_rgb]) -> lightweight Conv_R
TIR: C([raw_tir, aligned_tir]) -> lightweight Conv_T
out: Conv_R(...) + Conv_T(...)

All trainable convolutions begin at identity with respect to the second
feature of each pair, so the first forward equals original ADD exactly.
No feature product, no additional residual, no gamma scaling.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class BranchConcatConv(nn.Module):
    def __init__(self, channels: int, spatial_kernel: int = 3, channel_groups: int = 16):
        super().__init__()
        channels = int(channels)
        spatial_kernel = int(spatial_kernel)
        if channels <= 0 or spatial_kernel not in (1, 3):
            raise ValueError(f"Invalid channels/kernel: {channels}/{spatial_kernel}")

        # Per-channel 2-to-1 fusion. Channel ordering is [raw_c, processed_c]
        # for each c, allowing exact selection of the processed feature.
        self.pair_conv = nn.Conv2d(
            2 * channels, channels, kernel_size=1,
            groups=channels, bias=False,
        )
        self.spatial_conv = nn.Conv2d(
            channels, channels, kernel_size=spatial_kernel,
            padding=spatial_kernel // 2, groups=channels, bias=False,
        )
        # Cheap group-wise cross-channel mixing. Identity initialized.
        groups = math.gcd(channels, max(1, int(channel_groups)))
        self.channel_conv = nn.Conv2d(
            channels, channels, kernel_size=1, groups=groups, bias=False,
        )
        self._init_as_processed_identity()

    @torch.no_grad()
    def _init_as_processed_identity(self) -> None:
        self.pair_conv.weight.zero_()
        self.pair_conv.weight[:, 1, 0, 0] = 1.0
        self.spatial_conv.weight.zero_()
        center = self.spatial_conv.kernel_size[0] // 2
        self.spatial_conv.weight[:, 0, center, center] = 1.0
        self.channel_conv.weight.zero_()
        members = self.channel_conv.in_channels // self.channel_conv.groups
        for out_idx in range(self.channel_conv.out_channels):
            self.channel_conv.weight[out_idx, out_idx % members, 0, 0] = 1.0

    def forward(self, original: torch.Tensor, processed: torch.Tensor) -> torch.Tensor:
        if original.shape != processed.shape:
            raise ValueError(
                f"Concat->Conv requires identical feature shapes: "
                f"original={tuple(original.shape)} processed={tuple(processed.shape)}"
            )
        b, c, h, w = original.shape
        # [B, C, 2, H, W] -> [B, 2*C, H, W]
        # channel pair layout: (original_c, processed_c).
        paired = torch.stack((original, processed), dim=2).reshape(b, 2 * c, h, w)
        return self.channel_conv(self.spatial_conv(self.pair_conv(paired)))


class FigureCatConvAddFusion(nn.Module):
    """Two modality branches ending in ADD, replacing the old ADD module."""
    def __init__(self, rgb_channels: int, tir_channels: int,
                 spatial_kernel: int = 3, channel_groups: int = 16):
        super().__init__()
        if int(rgb_channels) != int(tir_channels):
            raise ValueError(
                "FigureCatConvAddFusion requires equal RGB and TIR channels; "
                f"got {rgb_channels}/{tir_channels}."
            )
        self.rgb = BranchConcatConv(rgb_channels, spatial_kernel, channel_groups)
        self.tir = BranchConcatConv(tir_channels, spatial_kernel, channel_groups)

    def forward(self, raw_rgb: torch.Tensor, bhlr_rgb: torch.Tensor,
                raw_tir: torch.Tensor, aligned_tir: torch.Tensor) -> torch.Tensor:
        rgb_out = self.rgb(raw_rgb, bhlr_rgb)
        tir_out = self.tir(raw_tir, aligned_tir)
        return rgb_out + tir_out


__all__ = ["BranchConcatConv", "FigureCatConvAddFusion"]
