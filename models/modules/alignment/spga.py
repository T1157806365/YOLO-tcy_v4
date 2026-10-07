"""
SPGA-EQ: Structural Phase Geometry Alignment with Equivariance Supervision
=========================================================================== 

Purpose
-------
Replace target-moment alignment (mu/sigma) with scene-structure phase geometry.
The module never consumes UAV boxes or targetness maps.

Pipeline
--------
RGB semantic / TIR input images
    -> modality-specific structural encoders
    -> P5 global phase correlation (translation)
    -> globally warped TIR
    -> P4 windowed phase residuals
    -> P4 dense residual flow
    -> transfer dense flow to image/P3
    -> aligned TIR + alignment confidence

Training-only equivariance supervision
--------------------------------------
A known random translation Delta is synthetically applied to TIR.  The estimated
P5 global translation must change by the same Delta relative to the estimate on
the original pair.  This does NOT assume that the original RGB-T pair is already
perfectly aligned and does not require absolute alignment ground truth.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlignment


# -----------------------------------------------------------------------------
# Structure extraction
# -----------------------------------------------------------------------------


class ConvGNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, groups: int = 1):
        super().__init__()
        pad = int(k) // 2
        g = min(8, int(c2))
        while int(c2) % g != 0 and g > 1:
            g -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                int(c1), int(c2), int(k), 1, pad,
                groups=int(groups), bias=False,
            ),
            nn.GroupNorm(g, int(c2)),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StructuralEncoder(nn.Module):
    """Modality-specific encoder that maps fixed structural cues to one map."""

    def __init__(self, hidden_channels: int = 24):
        super().__init__()
        c = int(hidden_channels)
        self.net = nn.Sequential(
            ConvGNAct(4, c, 3, 1),
            ConvGNAct(c, c, 3, c),
            ConvGNAct(c, c, 1, 1),
            nn.Conv2d(c, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        # Remove DC/contrast scale before phase geometry.
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-5)
        return (x - mean) / std


def _sobel_structure_input(x: torch.Tensor) -> torch.Tensor:
    """Return [gray, gx, gy, magnitude] structural cues."""
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW image, got {tuple(x.shape)}")

    # Works for 1-channel and 3-channel images.
    gray = x.float().mean(dim=1, keepdim=True)

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8)

    return torch.cat((gray, gx, gy, mag), dim=1)


def _resize_structure_input(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    s = _sobel_structure_input(x)
    if s.shape[-2:] == tuple(hw):
        return s
    return F.interpolate(s, size=tuple(hw), mode="bilinear", align_corners=False)


# -----------------------------------------------------------------------------
# Geometry primitives
# -----------------------------------------------------------------------------


def _base_grid(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, int(height), device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, int(width), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)
    return grid.unsqueeze(0).expand(int(batch), -1, -1, -1)


def warp_with_normalized_flow(
    source: torch.Tensor,
    flow_xy: torch.Tensor,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Warp source with flow expressed in [0,1] image-coordinate displacement.

    Positive flow_x samples further right in source; positive flow_y samples
    further down.  For a source displaced +dx/+dy relative to the reference,
    the correcting inverse-sampling flow is therefore +dx/+dy.
    """
    if source.ndim != 4:
        raise ValueError(f"source must be BCHW, got {tuple(source.shape)}")
    if flow_xy.ndim != 4 or flow_xy.shape[1] != 2:
        raise ValueError(f"flow_xy must be [B,2,H,W], got {tuple(flow_xy.shape)}")

    b, _, h, w = source.shape
    if flow_xy.shape[0] != b:
        raise ValueError("flow batch size mismatch")

    flow = flow_xy
    if flow.shape[-2:] != (h, w):
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=True)

    # grid_sample uses [-1,1]; [0,1] displacement therefore multiplies by 2.
    grid = _base_grid(b, h, w, source.device, torch.float32)
    delta = 2.0 * flow.float().permute(0, 2, 3, 1)
    grid = grid + delta

    return F.grid_sample(
        source,
        grid.to(dtype=source.dtype),
        mode="bilinear",
        padding_mode=str(padding_mode),
        align_corners=bool(align_corners),
    )


def _hann_2d(h: int, w: int, device: torch.device) -> torch.Tensor:
    wy = torch.hann_window(int(h), periodic=False, device=device, dtype=torch.float32)
    wx = torch.hann_window(int(w), periodic=False, device=device, dtype=torch.float32)
    return (wy[:, None] * wx[None, :]).view(1, 1, int(h), int(w))


def phase_correlation_shift(
    reference: torch.Tensor,
    source: torch.Tensor,
    max_shift_y: int,
    max_shift_x: int,
    temperature: float = 0.20,
    eps: float = 1e-6,
    return_correlation: bool = False,
):
    """
    Differentiable soft phase correlation.

    Returns
    -------
    shift_px : [B,2], (dx,dy) in feature-map pixels.  It describes the source
               displacement relative to reference and is directly usable as an
               inverse-sampling correction.
    confidence: [B,1] entropy/peak-based confidence.
    correlation (optional): search-window correlation map.
    """
    if reference.shape != source.shape:
        raise ValueError(
            f"Phase inputs must have identical shape, got {tuple(reference.shape)} "
            f"vs {tuple(source.shape)}"
        )
    if reference.ndim != 4:
        raise ValueError("Phase inputs must be BCHW")

    # FFT is explicitly FP32 even under autocast.
    r = reference.float()
    s = source.float()
    b, _, h, w = r.shape

    win = _hann_2d(h, w, r.device)
    r = (r - r.mean(dim=(-2, -1), keepdim=True)) * win
    s = (s - s.mean(dim=(-2, -1), keepdim=True)) * win

    fr = torch.fft.fft2(r, dim=(-2, -1))
    fs = torch.fft.fft2(s, dim=(-2, -1))

    # source * conj(reference) -> positive peak for positive source displacement.
    cps = fs * torch.conj(fr)
    cps = cps / cps.abs().clamp_min(float(eps))
    cps = cps.mean(dim=1, keepdim=True)

    corr = torch.fft.ifft2(cps, dim=(-2, -1)).real
    corr = torch.fft.fftshift(corr, dim=(-2, -1))

    cy = h // 2
    cx = w // 2
    ry = max(0, min(int(max_shift_y), max(cy - 1, 0)))
    rx = max(0, min(int(max_shift_x), max(cx - 1, 0)))

    y0, y1 = cy - ry, cy + ry + 1
    x0, x1 = cx - rx, cx + rx + 1
    window = corr[:, :, y0:y1, x0:x1]

    flat = window.flatten(1)
    z = (flat - flat.mean(dim=1, keepdim=True)) / flat.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(float(eps))

    tau = max(float(temperature), 1e-3)
    probs = torch.softmax((z / tau).clamp(-60.0, 60.0), dim=1)

    ys = torch.arange(-ry, ry + 1, device=r.device, dtype=torch.float32)
    xs = torch.arange(-rx, rx + 1, device=r.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dx = (probs * xx.reshape(1, -1)).sum(dim=1)
    dy = (probs * yy.reshape(1, -1)).sum(dim=1)
    shift = torch.stack((dx, dy), dim=1)

    n = max(probs.shape[1], 2)
    entropy = -(probs * probs.clamp_min(float(eps)).log()).sum(dim=1)
    entropy = entropy / math.log(float(n))
    entropy_conf = (1.0 - entropy).clamp(0.0, 1.0)
    peak_conf = probs.max(dim=1).values.clamp(0.0, 1.0)
    confidence = (0.75 * entropy_conf + 0.25 * peak_conf).unsqueeze(1)

    if return_correlation:
        return shift, confidence, window
    return shift, confidence


def _constant_flow_from_shift_px(
    shift_px: torch.Tensor,
    image_h: int,
    image_w: int,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """Convert input-image pixel shift [B,2] to dense [0,1] normalized flow."""
    b = shift_px.shape[0]
    sx = shift_px[:, 0] / max(int(image_w) - 1, 1)
    sy = shift_px[:, 1] / max(int(image_h) - 1, 1)
    flow = torch.stack((sx, sy), dim=1).view(b, 2, 1, 1)
    return flow.expand(b, 2, int(out_h), int(out_w))


# -----------------------------------------------------------------------------
# SPGA-EQ
# -----------------------------------------------------------------------------


class SPGAAlignment(BaseAlignment):
    """Input-guided structural phase alignment used before the TIR backbone."""

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg=cfg)
        cfg = dict(cfg or {})

        self.hidden_channels = int(cfg.get("hidden_channels", 24))
        self.p5_stride = int(cfg.get("p5_stride", 32))
        self.p4_stride = int(cfg.get("p4_stride", 16))
        self.p3_stride = int(cfg.get("p3_stride", 8))

        self.max_global_shift_px = float(cfg.get("max_global_shift_px", 128.0))
        self.max_local_shift_px = float(cfg.get("max_local_shift_px", 32.0))
        self.local_grid = int(cfg.get("local_grid", 4))
        self.temperature = float(cfg.get("temperature", 0.20))

        self.equivariance_enabled = bool(cfg.get("equivariance_enabled", True))
        self.equivariance_max_shift_px = float(
            cfg.get("equivariance_max_shift_px", 32.0)
        )

        self.padding_mode = str(cfg.get("padding_mode", "border"))
        self.align_corners = bool(cfg.get("align_corners", True))

        self.rgb_encoder = StructuralEncoder(self.hidden_channels)
        self.tir_encoder = StructuralEncoder(self.hidden_channels)

    @staticmethod
    def _feature_hw(h: int, w: int, stride: int) -> Tuple[int, int]:
        return max(8, int(round(h / stride))), max(8, int(round(w / stride)))

    def _encode_pair(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.rgb_encoder(_resize_structure_input(rgb, hw))
        t = self.tir_encoder(_resize_structure_input(tir, hw))
        return r, t

    def _estimate_global(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        return_correlation: bool = False,
    ):
        h, w = rgb.shape[-2:]
        p5_h, p5_w = self._feature_hw(h, w, self.p5_stride)
        r5, t5 = self._encode_pair(rgb, tir, (p5_h, p5_w))

        rx = int(math.ceil(self.max_global_shift_px * p5_w / max(w, 1)))
        ry = int(math.ceil(self.max_global_shift_px * p5_h / max(h, 1)))

        if return_correlation:
            shift5, conf5, corr5 = phase_correlation_shift(
                r5, t5, ry, rx, self.temperature, return_correlation=True
            )
        else:
            shift5, conf5 = phase_correlation_shift(
                r5, t5, ry, rx, self.temperature, return_correlation=False
            )
            corr5 = None

        # Convert P5 feature-cell displacement to semantic-input pixels.
        shift_px = torch.stack(
            (
                shift5[:, 0] * (float(w) / float(p5_w)),
                shift5[:, 1] * (float(h) / float(p5_h)),
            ),
            dim=1,
        )
        shift_px = torch.stack(
            (
                shift_px[:, 0].clamp(-self.max_global_shift_px, self.max_global_shift_px),
                shift_px[:, 1].clamp(-self.max_global_shift_px, self.max_global_shift_px),
            ),
            dim=1,
        )

        return {
            "rgb_structure_p5": r5,
            "tir_structure_p5": t5,
            "global_shift_feature_px": shift5,
            "global_shift_px": shift_px,
            "global_confidence": conf5,
            "phase_correlation_p5": corr5,
        }

    def _estimate_local(
        self,
        rgb: torch.Tensor,
        tir_global: torch.Tensor,
    ):
        h, w = rgb.shape[-2:]
        p4_h, p4_w = self._feature_hw(h, w, self.p4_stride)
        r4, t4 = self._encode_pair(rgb, tir_global, (p4_h, p4_w))

        g = max(1, min(self.local_grid, p4_h // 4, p4_w // 4))
        y_edges = torch.linspace(0, p4_h, g + 1, device=rgb.device).round().long()
        x_edges = torch.linspace(0, p4_w, g + 1, device=rgb.device).round().long()

        local_px_rows = []
        conf_rows = []

        for gy in range(g):
            px_row = []
            cf_row = []
            y0, y1 = int(y_edges[gy].item()), int(y_edges[gy + 1].item())
            for gx in range(g):
                x0, x1 = int(x_edges[gx].item()), int(x_edges[gx + 1].item())
                rp = r4[:, :, y0:y1, x0:x1]
                tp = t4[:, :, y0:y1, x0:x1]

                ph, pw = rp.shape[-2:]
                max_rx = int(math.ceil(self.max_local_shift_px * pw / max(w / g, 1.0)))
                max_ry = int(math.ceil(self.max_local_shift_px * ph / max(h / g, 1.0)))
                max_rx = min(max_rx, max(pw // 2 - 1, 0))
                max_ry = min(max_ry, max(ph // 2 - 1, 0))

                sh, cf = phase_correlation_shift(
                    rp, tp, max_ry, max_rx, self.temperature, return_correlation=False
                )

                # Patch feature displacement -> semantic-input pixels.
                input_per_p4_x = float(w) / float(p4_w)
                input_per_p4_y = float(h) / float(p4_h)
                sh_input = torch.stack(
                    (
                        sh[:, 0] * input_per_p4_x,
                        sh[:, 1] * input_per_p4_y,
                    ),
                    dim=1,
                )
                sh_input = torch.stack(
                    (
                        sh_input[:, 0].clamp(-self.max_local_shift_px, self.max_local_shift_px),
                        sh_input[:, 1].clamp(-self.max_local_shift_px, self.max_local_shift_px),
                    ),
                    dim=1,
                )
                px_row.append(sh_input)
                cf_row.append(cf)

            local_px_rows.append(torch.stack(px_row, dim=-1))   # [B,2,g]
            conf_rows.append(torch.stack(cf_row, dim=-1))       # [B,1,g]

        local_px = torch.stack(local_px_rows, dim=-2)            # [B,2,g,g]
        local_conf = torch.stack(conf_rows, dim=-2)              # [B,1,g,g]

        # Confidence only decides whether a LOCAL RESIDUAL is used.  It never
        # scales the already-estimated global translation.
        effective_local_px = local_px * local_conf.detach()

        # Smoothness is regularization on residual deformation only.
        smooth_terms = []
        if g > 1:
            smooth_terms.append((local_px[:, :, 1:, :] - local_px[:, :, :-1, :]).abs().mean())
            smooth_terms.append((local_px[:, :, :, 1:] - local_px[:, :, :, :-1]).abs().mean())
        local_smoothness = (
            torch.stack(smooth_terms).mean() / max(self.max_local_shift_px, 1.0)
            if smooth_terms
            else local_px.sum() * 0.0
        )

        return {
            "rgb_structure_p4": r4,
            "tir_structure_p4": t4,
            "local_residual_px": local_px,
            "effective_local_residual_px": effective_local_px,
            "local_confidence": local_conf,
            "local_smoothness": local_smoothness,
        }

    def _equivariance_loss(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        base_shift_px: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.training) or (not self.equivariance_enabled):
            zero = base_shift_px.sum() * 0.0
            return zero, torch.zeros_like(base_shift_px)

        b, _, h, w = tir.shape
        max_d = max(self.equivariance_max_shift_px, 1.0)

        delta_px = (2.0 * torch.rand(b, 2, device=tir.device) - 1.0) * max_d
        delta_flow = _constant_flow_from_shift_px(
            delta_px, h, w, h, w
        )

        # To synthetically displace source by +Delta, inverse sampling uses -Delta.
        tir_perturbed = warp_with_normalized_flow(
            tir,
            -delta_flow,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        pert = self._estimate_global(rgb, tir_perturbed, return_correlation=False)
        predicted_delta = pert["global_shift_px"] - base_shift_px

        # Normalize so the auxiliary objective stays O(1).
        loss = F.smooth_l1_loss(
            predicted_delta / max_d,
            delta_px / max_d,
            beta=0.10,
            reduction="mean",
        )
        return loss, delta_px

    def align_images(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        return_debug: bool = False,
        compute_equivariance: bool = False,
    ):
        if rgb.ndim != 4 or tir.ndim != 4:
            raise ValueError("SPGA expects BCHW RGB/TIR tensors")
        if rgb.shape[0] != tir.shape[0]:
            raise ValueError("RGB/TIR batch size mismatch")

        h, w = rgb.shape[-2:]
        if tir.shape[-2:] != (h, w):
            tir = F.interpolate(tir, size=(h, w), mode="bilinear", align_corners=False)

        global_info = self._estimate_global(rgb, tir, return_correlation=return_debug)
        global_shift_px = global_info["global_shift_px"]
        global_flow = _constant_flow_from_shift_px(global_shift_px, h, w, h, w)

        tir_global = warp_with_normalized_flow(
            tir,
            global_flow,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        local_info = self._estimate_local(rgb, tir_global)
        local_px = F.interpolate(
            local_info["effective_local_residual_px"],
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        )
        local_flow = torch.stack(
            (
                local_px[:, 0] / max(w - 1, 1),
                local_px[:, 1] / max(h - 1, 1),
            ),
            dim=1,
        )

        final_flow = global_flow + local_flow
        aligned_tir = warp_with_normalized_flow(
            tir,
            final_flow,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        p3_h, p3_w = self._feature_hw(h, w, self.p3_stride)
        local_conf_p3 = F.interpolate(
            local_info["local_confidence"],
            size=(p3_h, p3_w),
            mode="bilinear",
            align_corners=False,
        )
        global_conf = global_info["global_confidence"].view(-1, 1, 1, 1)
        # Confidence is guidance/diagnostics, not a multiplier on global geometry.
        support_p3 = (0.5 * global_conf + 0.5 * global_conf * local_conf_p3).clamp(0.0, 1.0)

        equiv_loss = final_flow.sum() * 0.0
        equiv_delta_px = torch.zeros_like(global_shift_px)
        if compute_equivariance:
            equiv_loss, equiv_delta_px = self._equivariance_loss(
                rgb, tir, base_shift_px=global_shift_px
            )

        info: Dict[str, Any] = {
            "enabled": True,
            "type": "spga",
            "global_shift_px": global_shift_px,
            "global_confidence": global_info["global_confidence"],
            "local_residual_px": local_info["local_residual_px"],
            "local_confidence": local_info["local_confidence"],
            "final_flow_normalized": final_flow,
            "support_p3": support_p3,
            "alignment_support": support_p3,
            "equivariance_loss": equiv_loss,
            "equivariance_delta_px": equiv_delta_px,
            "local_smoothness": local_info["local_smoothness"],
        }

        if return_debug:
            info.update(global_info)
            info.update(local_info)
            info["tir_global_aligned"] = tir_global
            info["tir_final_aligned"] = aligned_tir

        return aligned_tir, info

    # BaseAlignment compatibility. This new method is image-guided rather than
    # pyramid-guided, therefore wrapper code should call align_images().
    def forward(self, *args, **kwargs):
        if "rgb" in kwargs and "tir" in kwargs:
            return self.align_images(
                kwargs["rgb"], kwargs["tir"],
                return_debug=bool(kwargs.get("return_debug", False)),
                compute_equivariance=bool(kwargs.get("compute_equivariance", False)),
            )
        raise RuntimeError(
            "SPGAAlignment is input-image guided. Call align_images(rgb, tir, ...)."
        )


__all__ = [
    "SPGAAlignment",
    "StructuralEncoder",
    "phase_correlation_shift",
    "warp_with_normalized_flow",
]
