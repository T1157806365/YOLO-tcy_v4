"""
BSGA: BBox-Supervised Geometric Alignment
=========================================

Purpose
-------
Replace unsupervised / weakly constrained RGB-T geometric estimation with a
fully supervised global affine geometry predictor derived directly from paired
RGB/TIR bounding boxes during training.

This module NEVER receives boxes at inference time.  Boxes are used only to
construct training targets for the geometry head.

Geometry parameterization
-------------------------
We predict a source(TIR) -> output(RGB) axis-aligned affine transform around
image center c=(0.5, 0.5):

    p = c + s * (q - c) + t

where
    q : TIR normalized coordinate in [0,1]^2
    p : RGB normalized coordinate in [0,1]^2
    t : translation [tx, ty]
    s : anisotropic scale [sx, sy]

For a paired TIR/RGB bbox pair:

    RGB  box = (crx, cry, wr, hr)
    TIR  box = (ctx, cty, wt, ht)

an exact box-derived geometry target is:

    sx = wr / wt
    sy = hr / ht
    tx = crx - (0.5 + sx * (ctx - 0.5))
    ty = cry - (0.5 + sy * (cty - 0.5))

This is NOT a first/second-moment estimate.  The geometry target comes directly
from paired annotations.

For grid_sample, inverse sampling is used:

    q = c + (p - c - t) / s

Important stability choices
---------------------------
1) The geometry head final layer is initialized to ZERO:
       t = 0, log(s) = 0 -> exact identity transform at initialization.

2) Geometry-head inputs can be detached from detector backbones.

3) Predicted geometry can be detached before warp, so detection gradients do
   not corrupt the geometry regressor.  Geometry is then trained only by the
   explicit bbox-derived geometry loss.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlignment


class ConvBNAct(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                int(c1),
                int(c2),
                kernel_size=int(k),
                stride=int(s),
                padding=int(k) // 2,
                groups=int(groups),
                bias=False,
            ),
            nn.BatchNorm2d(int(c2)),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossModalGeometryHead(nn.Module):
    """
    Coordinate-aware RGB-T geometry regressor.

    It preserves coarse spatial layout using a 4x4 pooled grid rather than
    global average pooling to a single vector.  Normalized x/y coordinate maps
    are appended before the joint encoder so the regressor can model where
    cross-modal disagreement occurs.
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        hidden_channels: int = 32,
        pooled_size: int = 4,
        max_translation_norm: float = 0.75,
        scale_min: float = 0.25,
        scale_max: float = 4.0,
    ):
        super().__init__()

        if scale_min <= 0 or scale_max <= 0 or scale_min >= scale_max:
            raise ValueError(
                f"Invalid scale range [{scale_min}, {scale_max}]"
            )
        if not (scale_min <= 1.0 <= scale_max):
            raise ValueError("Scale range must contain identity scale 1.0")

        h = int(hidden_channels)
        self.pooled_size = int(pooled_size)
        self.max_translation_norm = float(max_translation_norm)
        self.log_scale_min = float(math.log(scale_min))
        self.log_scale_max = float(math.log(scale_max))

        self.rgb_proj = ConvBNAct(int(rgb_channels), h, 1)
        self.tir_proj = ConvBNAct(int(tir_channels), h, 1)

        # r, t, |r-t|, r*t => 4h, plus normalized x/y => +2
        self.joint = nn.Sequential(
            ConvBNAct(4 * h + 2, 2 * h, 3),
            ConvBNAct(2 * h, h, 3),
            ConvBNAct(h, h, 3, groups=h),
        )

        self.pool = nn.AdaptiveAvgPool2d((self.pooled_size, self.pooled_size))

        mlp_in = h * self.pooled_size * self.pooled_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 2 * h),
            nn.SiLU(inplace=True),
            nn.Linear(2 * h, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, 4),
        )

        # Exact identity geometry at initialization:
        # raw translation = 0 -> t = 0
        # raw scale       = 0 -> log(s) = 0 -> s = 1
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _resize_like(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if source.shape[-2:] == reference.shape[-2:]:
            return source
        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _coord_channels(reference: torch.Tensor) -> torch.Tensor:
        b, _, h, w = reference.shape
        yy = torch.linspace(
            -1.0, 1.0, h,
            device=reference.device,
            dtype=reference.dtype,
        )
        xx = torch.linspace(
            -1.0, 1.0, w,
            device=reference.device,
            dtype=reference.dtype,
        )
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        coord = torch.stack((xx, yy), dim=0).unsqueeze(0)
        return coord.expand(b, -1, -1, -1)

    def _bounded_log_scale(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Piecewise bounded log-scale with identity at raw=0.

        tanh(raw) in [-1,1]. Positive values map toward log(scale_max),
        negative values toward log(scale_min).
        """
        z = torch.tanh(raw)
        pos = z.clamp_min(0.0) * self.log_scale_max
        neg = (-z).clamp_min(0.0) * (-self.log_scale_min)
        return pos - neg

    def forward(
        self,
        rgb_feat: torch.Tensor,
        tir_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tir_feat = self._resize_like(tir_feat, rgb_feat)

        r = self.rgb_proj(rgb_feat)
        t = self.tir_proj(tir_feat)
        coord = self._coord_channels(r)

        joint = torch.cat(
            (r, t, (r - t).abs(), r * t, coord),
            dim=1,
        )
        z = self.joint(joint)
        z = self.pool(z).flatten(1)
        raw = self.mlp(z)

        translation = (
            torch.tanh(raw[:, 0:2])
            * self.max_translation_norm
        )
        log_scale = self._bounded_log_scale(raw[:, 2:4])
        scale = torch.exp(log_scale)

        return {
            "raw_geometry": raw,
            "translation": translation,
            "log_scale": log_scale,
            "scale": scale,
        }


def build_inverse_sampling_grid(
    translation: torch.Tensor,
    scale: torch.Tensor,
    height: int,
    width: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Build grid_sample inverse grid for:

        p = 0.5 + s * (q - 0.5) + t

    therefore:

        q = 0.5 + (p - 0.5 - t) / s

    p/q are normalized [0,1] coordinates.
    Returned grid uses [-1,1].
    """
    if translation.ndim != 2 or translation.shape[1] != 2:
        raise ValueError("translation must have shape [B,2]")
    if scale.ndim != 2 or scale.shape[1] != 2:
        raise ValueError("scale must have shape [B,2]")
    if translation.shape[0] != scale.shape[0]:
        raise ValueError("translation/scale batch size mismatch")

    b = int(translation.shape[0])
    device = translation.device
    calc_dtype = translation.dtype

    yy = torch.linspace(0.0, 1.0, int(height), device=device, dtype=calc_dtype)
    xx = torch.linspace(0.0, 1.0, int(width), device=device, dtype=calc_dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")

    px = xx.view(1, int(height), int(width)).expand(b, -1, -1)
    py = yy.view(1, int(height), int(width)).expand(b, -1, -1)

    tx = translation[:, 0].view(b, 1, 1)
    ty = translation[:, 1].view(b, 1, 1)
    sx = scale[:, 0].view(b, 1, 1).clamp_min(1e-6)
    sy = scale[:, 1].view(b, 1, 1).clamp_min(1e-6)

    qx = 0.5 + (px - 0.5 - tx) / sx
    qy = 0.5 + (py - 0.5 - ty) / sy

    grid = torch.stack((2.0 * qx - 1.0, 2.0 * qy - 1.0), dim=-1)
    if dtype is not None:
        grid = grid.to(dtype=dtype)
    return grid


def warp_tensor_with_geometry(
    source: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    output_hw: Tuple[int, int],
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    h, w = int(output_hw[0]), int(output_hw[1])
    grid = build_inverse_sampling_grid(
        translation=translation,
        scale=scale,
        height=h,
        width=w,
        dtype=source.dtype,
    )
    return F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode=str(padding_mode),
        align_corners=bool(align_corners),
    )


def transform_boxes_tir_to_rgb(
    tir_boxes_xywhn: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Apply predicted forward geometry to normalized TIR xywh boxes.

    Supports either:
      tir_boxes_xywhn [B,4] with translation/scale [B,2]
    or one box per geometry row.
    """
    if tir_boxes_xywhn.ndim != 2 or tir_boxes_xywhn.shape[1] != 4:
        raise ValueError("tir_boxes_xywhn must have shape [N,4]")
    if translation.ndim != 2 or translation.shape[1] != 2:
        raise ValueError("translation must have shape [N,2]")
    if scale.ndim != 2 or scale.shape[1] != 2:
        raise ValueError("scale must have shape [N,2]")
    if not (
        tir_boxes_xywhn.shape[0] == translation.shape[0] == scale.shape[0]
    ):
        raise ValueError("box/geometry row count mismatch")

    center_t = tir_boxes_xywhn[:, 0:2]
    size_t = tir_boxes_xywhn[:, 2:4]

    center_r = 0.5 + scale * (center_t - 0.5) + translation
    size_r = scale * size_t
    return torch.cat((center_r, size_r), dim=1)


def bbox_geometry_targets(
    rgb_boxes_xywhn: torch.Tensor,
    tir_boxes_xywhn: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Exact geometry targets from paired normalized bboxes.
    """
    if rgb_boxes_xywhn.shape != tir_boxes_xywhn.shape:
        raise ValueError("RGB/TIR paired boxes must have identical shape")
    if rgb_boxes_xywhn.ndim != 2 or rgb_boxes_xywhn.shape[1] != 4:
        raise ValueError("paired boxes must have shape [N,4]")

    rgb = rgb_boxes_xywhn.float()
    tir = tir_boxes_xywhn.float()

    rgb_size = rgb[:, 2:4].clamp_min(float(eps))
    tir_size = tir[:, 2:4].clamp_min(float(eps))

    scale = rgb_size / tir_size
    log_scale = torch.log(scale.clamp_min(float(eps)))

    translation = (
        rgb[:, 0:2]
        - (0.5 + scale * (tir[:, 0:2] - 0.5))
    )

    return {
        "translation": translation,
        "scale": scale,
        "log_scale": log_scale,
    }


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    c = boxes[:, 0:2]
    s = boxes[:, 2:4]
    return torch.cat((c - s / 2.0, c + s / 2.0), dim=1)


def aligned_box_iou(
    mapped_tir_xywhn: torch.Tensor,
    rgb_xywhn: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Differentiable IoU for paired normalized xywh boxes."""
    a = xywh_to_xyxy(mapped_tir_xywhn)
    b = xywh_to_xyxy(rgb_xywhn)

    inter_lt = torch.maximum(a[:, 0:2], b[:, 0:2])
    inter_rb = torch.minimum(a[:, 2:4], b[:, 2:4])
    inter_wh = (inter_rb - inter_lt).clamp_min(0.0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]

    area_a_wh = (a[:, 2:4] - a[:, 0:2]).clamp_min(0.0)
    area_b_wh = (b[:, 2:4] - b[:, 0:2]).clamp_min(0.0)
    area_a = area_a_wh[:, 0] * area_a_wh[:, 1]
    area_b = area_b_wh[:, 0] * area_b_wh[:, 1]

    union = area_a + area_b - inter
    return inter / union.clamp_min(float(eps))


class BSGAAlignment(BaseAlignment):
    """
    Pluggable bbox-supervised alignment module.

    The module predicts ONE normalized global affine geometry from a configured
    reference feature scale (default P3) and applies it consistently to selected
    pyramid scales.
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(cfg=cfg)
        self.cfg = dict(cfg or {})

        self.reference_scale = int(self.cfg.get("reference_scale", 3))
        self.apply_scales = tuple(
            sorted({int(s) for s in self.cfg.get("apply_scales", [3, 4, 5])})
        )

        self.detach_geometry_features = bool(
            self.cfg.get("detach_geometry_features", True)
        )
        self.detach_geometry_for_warp = bool(
            self.cfg.get("detach_geometry_for_warp", True)
        )

        self.padding_mode = str(self.cfg.get("padding_mode", "border"))
        self.align_corners = bool(self.cfg.get("align_corners", True))

        self.head = CrossModalGeometryHead(
            rgb_channels=int(rgb_channels),
            tir_channels=int(tir_channels),
            hidden_channels=int(self.cfg.get("hidden_channels", 32)),
            pooled_size=int(self.cfg.get("pooled_size", 4)),
            max_translation_norm=float(
                self.cfg.get("max_translation_norm", 0.75)
            ),
            scale_min=float(self.cfg.get("scale_min", 0.25)),
            scale_max=float(self.cfg.get("scale_max", 4.0)),
        )

    def predict_geometry(
        self,
        rgb_feat: torch.Tensor,
        tir_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.detach_geometry_features:
            rgb_feat = rgb_feat.detach()
            tir_feat = tir_feat.detach()
        return self.head(rgb_feat, tir_feat)

    def geometry_for_warp(
        self,
        geometry: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        translation = geometry["translation"]
        scale = geometry["scale"]

        if self.detach_geometry_for_warp:
            translation = translation.detach()
            scale = scale.detach()

        return translation, scale

    def warp_tensor(
        self,
        source: torch.Tensor,
        geometry: Mapping[str, torch.Tensor],
        output_hw: Tuple[int, int],
    ) -> torch.Tensor:
        translation, scale = self.geometry_for_warp(geometry)
        return warp_tensor_with_geometry(
            source=source,
            translation=translation,
            scale=scale,
            output_hw=output_hw,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
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
                f"BSGA reference P{self.reference_scale} not in scale_to_index"
            )

        ref_idx = int(scale_to_index[self.reference_scale])
        geometry = self.predict_geometry(
            rgb_features[ref_idx],
            tir_features[ref_idx],
        )

        aligned = list(tir_features)
        for scale_id in self.apply_scales:
            if scale_id not in scale_to_index:
                raise KeyError(f"BSGA apply scale P{scale_id} unavailable")
            idx = int(scale_to_index[scale_id])
            aligned[idx] = self.warp_tensor(
                source=tir_features[idx],
                geometry=geometry,
                output_hw=rgb_features[idx].shape[-2:],
            )

        info = {
            "enabled": True,
            "type": "bsga",
            "reference_scale": self.reference_scale,
            "translation": geometry["translation"],
            "scale": geometry["scale"],
            "log_scale": geometry["log_scale"],
            "raw_geometry": geometry["raw_geometry"],
        }

        if return_debug:
            translation_warp, scale_warp = self.geometry_for_warp(geometry)
            h, w = rgb_features[ref_idx].shape[-2:]
            info["sampling_grid_reference"] = build_inverse_sampling_grid(
                translation=translation_warp,
                scale=scale_warp,
                height=h,
                width=w,
                dtype=tir_features[ref_idx].dtype,
            )

        return aligned, info


__all__ = [
    "BSGAAlignment",
    "CrossModalGeometryHead",
    "bbox_geometry_targets",
    "transform_boxes_tir_to_rgb",
    "aligned_box_iou",
    "build_inverse_sampling_grid",
    "warp_tensor_with_geometry",
]
