"""
GTGA: Gaussian Target Geometric Alignment
=========================================

GTGA estimates ONE target geometry from a reference pyramid level (normally P3)
and propagates the same normalized geometric transform to the RGB-T fusion
pyramid.

Core:
    RGB feature -> RGB target map -> target distribution -> (mu_R, sigma_R)
    TIR feature -> TIR target map -> target distribution -> (mu_T, sigma_T)

The TIR -> RGB inverse sampling map for grid_sample is:
    q = mu_T + A (p - mu_R)

where:
    A = diag(sigma_T_x / sigma_R_x,
             sigma_T_y / sigma_R_y)

Thus one analytical warp jointly corrects translation and anisotropic scale.

GTGA also warps the TIR target map with exactly the same geometry and forms:
    joint_target_support = sqrt(A_RGB * A_TIR_aligned)

This support is passed to GTGA-guided BHLR-v2.

No correlation search, offset head, scale head, frequency branch, fine alignment,
or reliability router is used in this version.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlignment


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                int(c1),
                int(c2),
                kernel_size=int(k),
                stride=1,
                padding=int(k) // 2,
                groups=int(groups),
                bias=False,
            ),
            nn.BatchNorm2d(int(c2)),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TargetDistributionHead(nn.Module):
    """Independent per-modality UAV target-map predictor."""

    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        c = int(hidden_channels)

        self.proj = nn.Sequential(
            nn.LazyConv2d(c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.local = ConvBNAct(c, c, 3, groups=c)
        self.out = nn.Conv2d(c, 1, 1, bias=True)

        # Neutral target map at initialization: sigmoid(0)=0.5.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.local(self.proj(x)))


def normalize_target_distribution(
    targetness: torch.Tensor,
    power: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert targetness A in [0,1] to a normalized spatial target distribution.

        P(x,y) ∝ A(x,y)^power

    power > 1 reduces accumulated influence from many weak background pixels.
    """
    if targetness.ndim != 4 or targetness.shape[1] != 1:
        raise ValueError(
            "targetness must have shape [B,1,H,W], "
            f"got {tuple(targetness.shape)}"
        )

    w = targetness.float().clamp(0.0, 1.0)
    p = max(float(power), 1e-6)
    if p != 1.0:
        w = w.pow(p)

    den = w.sum(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    return w / den


def gaussian_moments(
    distribution: torch.Tensor,
    sigma_min_pixels: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate diagonal Gaussian geometry from P.

    Coordinates use normalized [0,1] image/feature coordinates.

    Returns:
        mu    [B,2] : [mu_x, mu_y]
        sigma [B,2] : [sigma_x, sigma_y]
    """
    if distribution.ndim != 4 or distribution.shape[1] != 1:
        raise ValueError(
            "distribution must have shape [B,1,H,W], "
            f"got {tuple(distribution.shape)}"
        )

    b, _, h, w = distribution.shape
    device = distribution.device
    dtype = distribution.dtype

    yy = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    xx = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")

    xx = xx.view(1, 1, h, w)
    yy = yy.view(1, 1, h, w)

    mu_x = (distribution * xx).sum(dim=(-2, -1))
    mu_y = (distribution * yy).sum(dim=(-2, -1))

    var_x = (
        distribution
        * (xx - mu_x.view(b, 1, 1, 1)).square()
    ).sum(dim=(-2, -1))
    var_y = (
        distribution
        * (yy - mu_y.view(b, 1, 1, 1)).square()
    ).sum(dim=(-2, -1))

    sigma_x = torch.sqrt(var_x.clamp_min(float(eps)))
    sigma_y = torch.sqrt(var_y.clamp_min(float(eps)))

    # Numerical protection in normalized coordinates.
    min_x = float(sigma_min_pixels) / max(w - 1, 1)
    min_y = float(sigma_min_pixels) / max(h - 1, 1)

    sigma_x = sigma_x.clamp_min(min_x)
    sigma_y = sigma_y.clamp_min(min_y)

    mu = torch.cat((mu_x, mu_y), dim=1)
    sigma = torch.cat((sigma_x, sigma_y), dim=1)

    return mu, sigma


def build_gaussian_sampling_grid(
    rgb_mu: torch.Tensor,
    rgb_sigma: torch.Tensor,
    tir_mu: torch.Tensor,
    tir_sigma: torch.Tensor,
    height: int,
    width: int,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
    dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build inverse TIR sampling grid for an RGB output grid.

        q = mu_T + A (p - mu_R)
        A = diag(sigma_T / sigma_R)

    Returns:
        grid [B,H,W,2] in grid_sample [-1,1] coordinates
        sampling_scale [B,2]
    """
    if rgb_mu.shape != tir_mu.shape or rgb_mu.ndim != 2 or rgb_mu.shape[1] != 2:
        raise ValueError("rgb_mu and tir_mu must both have shape [B,2]")
    if (
        rgb_sigma.shape != tir_sigma.shape
        or rgb_sigma.ndim != 2
        or rgb_sigma.shape[1] != 2
    ):
        raise ValueError("rgb_sigma and tir_sigma must both have shape [B,2]")

    lo = max(float(scale_min), 1e-4)
    hi = max(float(scale_max), lo)

    log_scale = (
        torch.log(tir_sigma.clamp_min(1e-8))
        - torch.log(rgb_sigma.clamp_min(1e-8))
    ).clamp(math.log(lo), math.log(hi))

    scale = torch.exp(log_scale)

    b = rgb_mu.shape[0]
    device = rgb_mu.device
    calc_dtype = rgb_mu.dtype

    yy = torch.linspace(
        0.0, 1.0, int(height),
        device=device, dtype=calc_dtype,
    )
    xx = torch.linspace(
        0.0, 1.0, int(width),
        device=device, dtype=calc_dtype,
    )
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")

    px = xx.view(1, height, width).expand(b, -1, -1)
    py = yy.view(1, height, width).expand(b, -1, -1)

    qx = (
        tir_mu[:, 0].view(b, 1, 1)
        + scale[:, 0].view(b, 1, 1)
        * (px - rgb_mu[:, 0].view(b, 1, 1))
    )
    qy = (
        tir_mu[:, 1].view(b, 1, 1)
        + scale[:, 1].view(b, 1, 1)
        * (py - rgb_mu[:, 1].view(b, 1, 1))
    )

    grid = torch.stack(
        (2.0 * qx - 1.0, 2.0 * qy - 1.0),
        dim=-1,
    )

    if dtype is not None:
        grid = grid.to(dtype=dtype)

    return grid, scale


def warp_with_grid(
    source: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    if source.ndim != 4:
        raise ValueError(
            f"source must be BCHW, got {tuple(source.shape)}"
        )

    grid = grid.to(
        device=source.device,
        dtype=source.dtype,
    )

    return F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode=str(padding_mode),
        align_corners=bool(align_corners),
    )


class GTGAUnit(nn.Module):
    """Estimate one global target Gaussian relation from one feature level."""

    def __init__(
        self,
        hidden_channels: int = 32,
        moment_power: float = 2.0,
        sigma_min_pixels: float = 0.5,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        padding_mode: str = "border",
        align_corners: bool = True,
    ):
        super().__init__()

        self.rgb_target_head = TargetDistributionHead(hidden_channels)
        self.tir_target_head = TargetDistributionHead(hidden_channels)

        self.moment_power = float(moment_power)
        self.sigma_min_pixels = float(sigma_min_pixels)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)

    @staticmethod
    def _resize_like(
        source: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if source.shape[-2:] == reference.shape[-2:]:
            return source

        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def make_grid(
        self,
        geometry: Mapping[str, torch.Tensor],
        height: int,
        width: int,
        dtype: torch.dtype,
    ):
        return build_gaussian_sampling_grid(
            rgb_mu=geometry["rgb_mu"],
            rgb_sigma=geometry["rgb_sigma"],
            tir_mu=geometry["tir_mu"],
            tir_sigma=geometry["tir_sigma"],
            height=int(height),
            width=int(width),
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            dtype=dtype,
        )

    def warp_tensor(
        self,
        source: torch.Tensor,
        geometry: Mapping[str, torch.Tensor],
        output_hw: Tuple[int, int],
    ) -> torch.Tensor:
        h, w = int(output_hw[0]), int(output_hw[1])

        grid, _ = self.make_grid(
            geometry=geometry,
            height=h,
            width=w,
            dtype=source.dtype,
        )

        return warp_with_grid(
            source,
            grid,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

    def forward(
        self,
        rgb_feat: torch.Tensor,
        tir_feat: torch.Tensor,
        return_debug: bool = False,
    ):
        # Geometry is estimated on one common P-level grid.
        tir_feat = self._resize_like(
            tir_feat,
            rgb_feat,
        )

        rgb_logits = self.rgb_target_head(rgb_feat)
        tir_logits = self.tir_target_head(tir_feat)

        rgb_target = torch.sigmoid(rgb_logits)
        tir_target = torch.sigmoid(tir_logits)

        rgb_dist = normalize_target_distribution(
            rgb_target,
            power=self.moment_power,
        )
        tir_dist = normalize_target_distribution(
            tir_target,
            power=self.moment_power,
        )

        rgb_mu, rgb_sigma = gaussian_moments(
            rgb_dist,
            sigma_min_pixels=self.sigma_min_pixels,
        )
        tir_mu, tir_sigma = gaussian_moments(
            tir_dist,
            sigma_min_pixels=self.sigma_min_pixels,
        )

        geometry = {
            "rgb_mu": rgb_mu,
            "tir_mu": tir_mu,
            "rgb_sigma": rgb_sigma,
            "tir_sigma": tir_sigma,
        }

        h, w = rgb_feat.shape[-2:]
        grid, sampling_scale = self.make_grid(
            geometry=geometry,
            height=h,
            width=w,
            dtype=tir_feat.dtype,
        )

        aligned_tir = warp_with_grid(
            tir_feat,
            grid,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        aligned_tir_target = warp_with_grid(
            tir_target,
            grid,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        ).clamp(0.0, 1.0)

        joint_target_support = torch.sqrt(
            (
                rgb_target.float()
                * aligned_tir_target.float()
            ).clamp_min(0.0)
            + 1e-8
        ).clamp(0.0, 1.0).to(rgb_target.dtype)

        info = {
            "rgb_target_logits": rgb_logits,
            "tir_target_logits": tir_logits,
            "rgb_targetness": rgb_target,
            "tir_targetness": tir_target,
            "aligned_tir_targetness": aligned_tir_target,
            "joint_target_support": joint_target_support,
            "rgb_mu": rgb_mu,
            "tir_mu": tir_mu,
            "rgb_sigma": rgb_sigma,
            "tir_sigma": tir_sigma,
            "sampling_scale": sampling_scale,
        }

        if return_debug:
            info.update(
                {
                    "rgb_distribution": rgb_dist,
                    "tir_distribution": tir_dist,
                    "sampling_grid": grid,
                    "aligned_tir_reference": aligned_tir,
                }
            )

        return aligned_tir, info


class GTGAAlignment(BaseAlignment):
    """
    Pluggable GTGA.

    A single reference scale (normally P3) estimates target geometry.
    The same normalized geometry is then propagated to selected pyramid levels.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg=cfg)

        configured_scales = tuple(int(s) for s in self.scales)
        if not configured_scales:
            configured_scales = (3,)

        self.reference_scale = int(
            self.cfg.get(
                "reference_scale",
                configured_scales[0],
            )
        )

        if len(configured_scales) > 1:
            raise ValueError(
                "Current clean GTGA estimates ONE Gaussian geometry. "
                "Use one reference scale, e.g. scales: [3]."
            )

        if self.reference_scale != configured_scales[0]:
            raise ValueError(
                "reference_scale must equal the single configured GTGA scale."
            )

        self.unit = GTGAUnit(
            hidden_channels=int(
                self.cfg.get("hidden_channels", 32)
            ),
            moment_power=float(
                self.cfg.get("moment_power", 2.0)
            ),
            sigma_min_pixels=float(
                self.cfg.get("sigma_min_pixels", 0.5)
            ),
            scale_min=float(
                self.cfg.get("scale_min", 0.5)
            ),
            scale_max=float(
                self.cfg.get("scale_max", 2.0)
            ),
            padding_mode=str(
                self.cfg.get("padding_mode", "border")
            ),
            align_corners=bool(
                self.cfg.get("align_corners", True)
            ),
        )

    def align_reference_pair(
        self,
        rgb_feat: torch.Tensor,
        tir_feat: torch.Tensor,
        return_debug: bool = False,
    ):
        return self.unit(
            rgb_feat,
            tir_feat,
            return_debug=return_debug,
        )

    def warp_tensor(
        self,
        source: torch.Tensor,
        geometry: Mapping[str, torch.Tensor],
        output_hw: Tuple[int, int],
    ):
        return self.unit.warp_tensor(
            source=source,
            geometry=geometry,
            output_hw=output_hw,
        )

    def forward(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        return_debug: bool = False,
    ):
        if self.reference_scale not in scale_to_index:
            raise KeyError(
                f"GTGA reference P{self.reference_scale} missing from "
                f"scale_to_index={dict(scale_to_index)}"
            )

        ref_idx = int(
            scale_to_index[self.reference_scale]
        )

        ref_aligned, ref_info = self.align_reference_pair(
            rgb_features[ref_idx],
            tir_features[ref_idx],
            return_debug=return_debug,
        )

        aligned = list(tir_features)
        aligned[ref_idx] = ref_aligned

        apply_scales = self.cfg.get(
            "apply_scales",
            list(scale_to_index.keys()),
        )
        apply_scales = tuple(
            sorted({int(s) for s in apply_scales})
        )

        for scale in apply_scales:
            if scale not in scale_to_index:
                raise KeyError(
                    f"GTGA apply scale P{scale} missing from scale_to_index."
                )

            idx = int(scale_to_index[scale])

            if idx == ref_idx:
                continue

            aligned[idx] = self.warp_tensor(
                source=tir_features[idx],
                geometry=ref_info,
                output_hw=rgb_features[idx].shape[-2:],
            )

        return aligned, {
            "enabled": True,
            "type": "gtga",
            "reference_scale": self.reference_scale,
            "apply_scales": list(apply_scales),
            "has_alignment": True,
            "reference_debug": ref_info,
            "scale_debug": {
                self.reference_scale: ref_info
            },
        }
