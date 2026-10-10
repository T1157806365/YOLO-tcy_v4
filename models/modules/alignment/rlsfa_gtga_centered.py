"""
GMLS v2 - GMLS-primary, bbox-supervised DMLS conditional correction
=====================================================================
Global + Multi-Local Soft Alignment for RGB-T UAV alignment.

Motivation
----------
Two empirical observations are combined:

1. Historical GLOBAL soft expectation is stable but can be pulled toward an
   invalid midpoint when distant/opposite modes coexist.
2. Single-mode DMLS is locally precise, but if its single dominant peak is
   wrong, the whole 7x7 basin can jump to a catastrophic mode.

GMLS keeps BOTH properties:
    - historical global-soft offset is retained as a stable candidate;
    - several spatially distinct modes are proposed from the full 41x41
      correlation map;
    - each mode is converted to a continuous LOCAL soft offset;
    - global + local candidates are scored with target-localized feature
      consistency and targetness consistency;
    - final coarse translation is a SOFT mixture of all candidates.

No hard candidate is used as the final geometry.

Pipeline
--------
Full 41x41 correlation
    -> global soft candidate d_g
    -> K distinct mode centers (proposal only)
    -> per-mode 7x7 local soft offsets d_1...d_K
    -> candidate evidence:
         target-localized projected-feature cosine
         + targetness soft IoU
    -> prior-aware soft candidate aggregation
    -> original Fine +/-2
    -> original Local-GTGA Moment scale
    -> ONE final center-preserving warp

Important
---------
Fine translation and Moment scale are unchanged.
No Top-K hard winner.
No Fine confidence gate.
No Quantile scale.
No scale reliability fallback.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlignment


class ConvBNAct(nn.Module):
    """Original zero-padded block retained for the proven coarse branch."""
    def __init__(self, c1: int, c2: int, k: int = 3, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, 1, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ReflectConvBNAct(nn.Module):
    """Reflection padding for GTGA target heads to reduce edge artifacts."""
    def __init__(self, c1: int, c2: int, k: int = 3, groups: int = 1):
        super().__init__()
        pad = k // 2
        self.block = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(c1, c2, k, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LazyProjection(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            nn.LazyConv2d(c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            ConvBNAct(c, c, 3, groups=c),
        )

    def forward(self, x):
        return self.net(x)


class TargetMapHead(nn.Module):
    """GTGA target-support head with reflection padding."""
    def __init__(self, channels: int = 32):
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            ReflectConvBNAct(c, c, 3),
            nn.Conv2d(c, 1, 1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class AdaptiveBoundedScalar(nn.Module):
    """Predict one bounded per-image scalar while preserving a chosen baseline.

    The final layer is zero-initialized and its bias is chosen such that the
    initial output is exactly ``init_value``.  Therefore enabling this module
    does not change the historical geometry at initialization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        min_value: float,
        max_value: float,
        init_value: float,
    ):
        super().__init__()
        lo = float(min_value)
        hi = float(max_value)
        init = float(init_value)
        if not (hi > lo):
            raise ValueError(f"max_value must exceed min_value, got {lo}, {hi}")
        if not (lo < init < hi):
            raise ValueError(
                f"init_value must lie strictly inside ({lo}, {hi}), got {init}"
            )

        h = max(4, int(hidden_channels))
        self.min_value = lo
        self.max_value = hi
        self.net = nn.Sequential(
            nn.Linear(int(in_channels), h),
            nn.SiLU(inplace=True),
            nn.Linear(h, 1),
        )

        # Exact baseline-preserving initialization.
        ratio = (init - lo) / (hi - lo)
        bias = math.log(ratio / (1.0 - ratio))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.net(descriptor.float()))
        return self.min_value + (self.max_value - self.min_value) * alpha


def _as_batch_scalar(
    value,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    min_value: float = 1e-4,
) -> torch.Tensor:
    """Convert a scalar/per-image tensor to [B,1] without breaking gradients."""
    if torch.is_tensor(value):
        t = value.to(device=device, dtype=dtype)
        if t.numel() == 1:
            t = t.reshape(1, 1).expand(batch_size, 1)
        elif t.ndim == 1 and t.shape[0] == batch_size:
            t = t[:, None]
        elif t.ndim >= 2 and t.shape[0] == batch_size:
            t = t.reshape(batch_size, -1)[:, :1]
        else:
            raise ValueError(
                f"Cannot interpret adaptive scalar with shape {tuple(t.shape)} "
                f"for batch={batch_size}"
            )
        return t.clamp_min(float(min_value))

    return torch.full(
        (batch_size, 1),
        max(float(value), float(min_value)),
        device=device,
        dtype=dtype,
    )


def correlation_statistics(
    logits: torch.Tensor,
    statistics_temperature: float = 0.10,
) -> torch.Tensor:
    """Three GT-free descriptors of correlation-map quality.

    Returns [B,3]:
      0) peak probability,
      1) normalized entropy,
      2) top1-top2 probability margin.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,K], got {tuple(logits.shape)}")

    t = max(float(statistics_temperature), 1e-4)
    prob = torch.softmax(logits.float() / t, dim=1)
    peak = prob.max(dim=1).values
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / max(math.log(max(int(prob.shape[1]), 2)), 1e-6)

    if prob.shape[1] >= 2:
        top2 = torch.topk(prob, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = torch.ones_like(peak)

    return torch.stack((peak, entropy.clamp(0, 1), margin.clamp(0, 1)), dim=1)


def softargmax_translation_from_logits(
    logits: torch.Tensor,
    search_radius: int,
    temperature,
):
    """Convert full search logits into translation using scalar/per-image T."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,K], got {tuple(logits.shape)}")
    b, k = logits.shape
    r = int(search_radius)
    side = 2 * r + 1
    if k != side * side:
        raise ValueError(
            f"Expected {(side * side)} logits for radius={r}, got {k}"
        )

    temp = _as_batch_scalar(
        temperature, b, logits.device, logits.dtype, min_value=1e-4
    )
    prob = torch.softmax(logits / temp, dim=1)

    disp_y, disp_x = torch.meshgrid(
        torch.arange(-r, r + 1, device=logits.device, dtype=logits.dtype),
        torch.arange(-r, r + 1, device=logits.device, dtype=logits.dtype),
        indexing='ij',
    )
    displacement = torch.stack(
        (disp_x.reshape(-1), disp_y.reshape(-1)), dim=1
    )
    expected = prob @ displacement

    maxp = prob.max(1, keepdim=True).values
    uniform = 1.0 / max(k, 1)
    confidence = ((maxp - uniform) / max(1.0 - uniform, 1e-6)).clamp(0, 1)
    return expected.view(-1, 2, 1, 1), confidence.view(-1, 1, 1, 1), prob


def resize_mask(mask: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    b, _, h, w = reference.shape
    if mask is None:
        return torch.ones(b, 1, h, w, device=reference.device, dtype=reference.dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[0] != b or mask.shape[1] != 1:
        raise ValueError(f"valid mask must be [B,1,H,W], got {tuple(mask.shape)}")
    mask = mask.to(device=reference.device, dtype=torch.float32).clamp(0, 1)
    if mask.shape[-2:] != (h, w):
        sh, sw = mask.shape[-2:]
        mode = 'area' if sh >= h and sw >= w else 'nearest'
        mask = F.interpolate(mask, size=(h, w), mode=mode)
    return (mask >= 0.999).to(dtype=reference.dtype)


def erode_safe_mask(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Erode valid content and remove a small outer feature border."""
    r = max(int(radius), 0)
    if r == 0:
        return mask
    inv = 1.0 - mask.float()
    dilated_invalid = F.max_pool2d(inv, kernel_size=2 * r + 1, stride=1, padding=r)
    safe = (1.0 - dilated_invalid).clamp(0, 1)
    safe[..., :r, :] = 0
    safe[..., -r:, :] = 0
    safe[..., :, :r] = 0
    safe[..., :, -r:] = 0
    return safe.to(dtype=mask.dtype)


def normalized_power_distribution(
    score: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    power: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = resize_mask(valid_mask, score).float()
    p = score.float().clamp(0, 1).pow(max(float(power), 1e-6)) * valid
    den = p.sum((-2, -1), keepdim=True)
    fallback = valid / valid.sum((-2, -1), keepdim=True).clamp_min(eps)
    return torch.where(den > eps, p / den.clamp_min(eps), fallback)


def distribution_center(distribution: torch.Tensor) -> torch.Tensor:
    b, _, h, w = distribution.shape
    ys = torch.linspace(0, 1, h, device=distribution.device, dtype=distribution.dtype)
    xs = torch.linspace(0, 1, w, device=distribution.device, dtype=distribution.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    xx = xx.view(1, 1, h, w)
    yy = yy.view(1, 1, h, w)
    cx = (distribution * xx).sum((-2, -1))[:, 0]
    cy = (distribution * yy).sum((-2, -1))[:, 0]
    return torch.stack((cx, cy), dim=1)


def gaussian_window(
    center: torch.Tensor,
    height: int,
    width: int,
    sigma=0.10,
) -> torch.Tensor:
    """Gaussian window supporting fixed or per-image sigma.

    ``sigma`` may be:
      - scalar float,
      - [B] or [B,1] isotropic tensor,
      - [B,2] tensor containing (sigma_x, sigma_y),
      - a two-scalar tuple/list.
    """
    b = int(center.shape[0])
    device = center.device
    dtype = center.dtype

    if torch.is_tensor(sigma):
        st = sigma.to(device=device, dtype=dtype)
        if st.numel() == 1:
            sx = sy = st.reshape(1, 1, 1, 1).expand(b, 1, 1, 1)
        elif st.ndim == 1 and st.shape[0] == b:
            sx = sy = st.view(b, 1, 1, 1)
        elif st.ndim >= 2 and st.shape[0] == b:
            flat = st.reshape(b, -1)
            if flat.shape[1] == 1:
                sx = sy = flat[:, 0].view(b, 1, 1, 1)
            else:
                sx = flat[:, 0].view(b, 1, 1, 1)
                sy = flat[:, 1].view(b, 1, 1, 1)
        else:
            raise ValueError(
                f"Cannot interpret sigma tensor {tuple(st.shape)} for batch={b}"
            )
    elif isinstance(sigma, (tuple, list)):
        sx = torch.full((b, 1, 1, 1), float(sigma[0]), device=device, dtype=dtype)
        sy = torch.full((b, 1, 1, 1), float(sigma[1]), device=device, dtype=dtype)
    else:
        sx = sy = torch.full(
            (b, 1, 1, 1), float(sigma), device=device, dtype=dtype
        )

    sx = sx.clamp_min(1e-4)
    sy = sy.clamp_min(1e-4)
    ys = torch.linspace(0, 1, height, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    xx = xx.view(1, 1, height, width)
    yy = yy.view(1, 1, height, width)
    cx = center[:, 0].view(b, 1, 1, 1)
    cy = center[:, 1].view(b, 1, 1, 1)
    return torch.exp(-0.5 * (((xx - cx) / sx).square() + ((yy - cy) / sy).square()))


def local_target_distribution(
    targetness: torch.Tensor,
    center: torch.Tensor,
    valid_mask: torch.Tensor,
    window_sigma,
    power: float,
    eps: float = 1e-6,
):
    h, w = targetness.shape[-2:]
    window = gaussian_window(center, h, w, window_sigma)
    localized = targetness.float().clamp(0, 1) * window * valid_mask.float()
    p = localized.pow(max(float(power), 1e-6))
    den = p.sum((-2, -1), keepdim=True)
    fallback = window * valid_mask.float()
    fallback = fallback / fallback.sum((-2, -1), keepdim=True).clamp_min(eps)
    distribution = torch.where(den > eps, p / den.clamp_min(eps), fallback)
    return distribution, window, localized


def centered_sigma(
    distribution: torch.Tensor,
    center: torch.Tensor,
    sigma_min: float = 0.005,
    eps: float = 1e-8,
) -> torch.Tensor:
    b, _, h, w = distribution.shape
    ys = torch.linspace(0, 1, h, device=distribution.device, dtype=distribution.dtype)
    xs = torch.linspace(0, 1, w, device=distribution.device, dtype=distribution.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    xx = xx.view(1, 1, h, w)
    yy = yy.view(1, 1, h, w)
    cx = center[:, 0].view(b, 1, 1, 1)
    cy = center[:, 1].view(b, 1, 1, 1)
    vx = (distribution * (xx - cx).square()).sum((-2, -1))[:, 0]
    vy = (distribution * (yy - cy).square()).sum((-2, -1))[:, 0]
    return torch.stack(
        (torch.sqrt(vx.clamp_min(eps)), torch.sqrt(vy.clamp_min(eps))), dim=1
    ).clamp_min(max(float(sigma_min), 1e-6))


def correlation_translation(
    reference: torch.Tensor,
    source: torch.Tensor,
    weight_map: torch.Tensor,
    radius: int,
    temperature: float,
    reference_valid_mask: Optional[torch.Tensor] = None,
    source_valid_mask: Optional[torch.Tensor] = None,
):
    """Original targetness-weighted explicit global translation search."""
    r = int(radius)
    if reference.shape[-2:] != source.shape[-2:]:
        source = F.interpolate(source, reference.shape[-2:], mode='bilinear', align_corners=False)
    ref = F.normalize(reference.float(), p=2, dim=1, eps=1e-6)
    src = F.normalize(source.float(), p=2, dim=1, eps=1e-6)
    ref_valid = resize_mask(reference_valid_mask, ref).float()
    src_valid = resize_mask(source_valid_mask, src).float()
    weight0 = weight_map.float().clamp_min(0)
    if weight0.shape[-2:] != ref.shape[-2:]:
        weight0 = F.interpolate(weight0, ref.shape[-2:], mode='bilinear', align_corners=False)
    weight0 = weight0 * ref_valid
    _, _, h, w = src.shape
    src_pad = F.pad(src, (r, r, r, r), mode='replicate')
    valid_pad = F.pad(src_valid, (r, r, r, r), mode='constant', value=0)
    scores, displacements = [], []
    for dy in range(-r, r + 1):
        shifted_list, valid_list = [], []
        for dx in range(-r, r + 1):
            y0, x0 = r + dy, r + dx
            shifted_list.append(src_pad[:, :, y0:y0 + h, x0:x0 + w])
            valid_list.append(valid_pad[:, :, y0:y0 + h, x0:x0 + w])
            displacements.append((float(dx), float(dy)))
        shifted = torch.stack(shifted_list, dim=2)
        shifted_valid = torch.stack(valid_list, dim=2)
        corr = (ref.unsqueeze(2) * shifted).sum(1)
        weight = (weight0.unsqueeze(2) * shifted_valid)[:, 0]
        den = weight.sum((-2, -1)).clamp_min(1e-6)
        scores.append((corr * weight).sum((-2, -1)) / den)
    logits = torch.cat(scores, dim=1)
    expected, confidence, prob = softargmax_translation_from_logits(
        logits=logits,
        search_radius=r,
        temperature=temperature,
    )
    return expected, confidence, prob, logits


def dominant_mode_local_softargmax(
    logits: torch.Tensor,
    search_radius: int,
    basin_radius: int = 3,
    temperature: float = 0.10,
):
    """Read a translation from a full correlation map using local soft-argmax.

    Parameters
    ----------
    logits:
        [B, K] correlation scores from the full global search.
        K must equal (2*search_radius+1)^2.
    search_radius:
        Global displacement radius used to build logits.
    basin_radius:
        Radius around the dominant mode in displacement-index space.
        basin_radius=3 gives a 7x7 square basin before edge clipping.
    temperature:
        Softmax temperature used INSIDE the selected basin.

    Returns
    -------
    local_offset:
        [B,2,1,1] local soft expectation in feature-pixel displacement.
    local_confidence:
        [B,1,1,1] normalized peak probability inside the basin.
        Diagnostic only; it does NOT gate the geometry.
    local_probability:
        [B,K] probability, exactly zero outside the selected basin.
    basin_mask:
        [B,K] boolean mask of the selected local basin.
    peak_offset:
        [B,2,1,1] hard dominant-mode center. Diagnostic only.
    peak_index:
        [B] flat index of the dominant mode.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,K], got {tuple(logits.shape)}")

    r = int(search_radius)
    side = 2 * r + 1
    expected_k = side * side
    if int(logits.shape[1]) != expected_k:
        raise ValueError(
            f"Expected K={expected_k} for search_radius={r}, "
            f"got K={int(logits.shape[1])}"
        )

    br = max(0, int(basin_radius))
    b = int(logits.shape[0])
    device = logits.device
    dtype = logits.dtype
    temp = _as_batch_scalar(temperature, b, device, dtype, min_value=1e-4)

    # Dominant mode is used ONLY to choose the basin center.
    peak_index = logits.argmax(dim=1)
    peak_y = torch.div(peak_index, side, rounding_mode='floor')
    peak_x = peak_index.remainder(side)

    ys = torch.arange(side, device=device).view(1, side, 1)
    xs = torch.arange(side, device=device).view(1, 1, side)
    basin_2d = (
        (ys - peak_y.view(b, 1, 1)).abs() <= br
    ) & (
        (xs - peak_x.view(b, 1, 1)).abs() <= br
    )
    basin_mask = basin_2d.reshape(b, expected_k)

    scaled = logits / temp
    neg_large = torch.finfo(scaled.dtype).min
    local_logits = scaled.masked_fill(~basin_mask, neg_large)
    local_probability = torch.softmax(local_logits, dim=1)

    disp_y, disp_x = torch.meshgrid(
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        indexing='ij',
    )
    displacement = torch.stack(
        (disp_x.reshape(-1), disp_y.reshape(-1)), dim=1
    )

    local_expected = local_probability @ displacement
    peak_offset = displacement[peak_index]

    # Diagnostic confidence only. It is never multiplied into translation.
    maxp = local_probability.max(dim=1, keepdim=True).values
    basin_size = basin_mask.sum(dim=1, keepdim=True).to(dtype=dtype).clamp_min(1)
    uniform = 1.0 / basin_size
    local_confidence = (
        (maxp - uniform) / (1.0 - uniform).clamp_min(1e-6)
    ).clamp(0, 1)

    return (
        local_expected.view(-1, 2, 1, 1),
        local_confidence.view(-1, 1, 1, 1),
        local_probability,
        basin_mask,
        peak_offset.view(-1, 2, 1, 1),
        peak_index,
    )


def distinct_local_soft_modes(
    logits: torch.Tensor,
    search_radius: int,
    topk: int = 3,
    nms_radius: int = 4,
    basin_radius: int = 3,
    temperature: float = 0.10,
):
    """Extract spatially distinct modes, then soft-average inside each basin.

    The hard peak locations are used ONLY to define candidate basins.
    Returned mode offsets are continuous local-soft expectations.

    Returns
    -------
    mode_offsets        : [B,K,2]
    mode_peak_offsets   : [B,K,2]
    mode_peak_indices   : [B,K]
    mode_probabilities  : [B,K,N]
    mode_masks          : [B,K,N] bool
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,N], got {tuple(logits.shape)}")

    r = int(search_radius)
    side = 2 * r + 1
    n = side * side
    if int(logits.shape[1]) != n:
        raise ValueError(f"Expected {n} logits, got {int(logits.shape[1])}")

    k = max(1, min(int(topk), n))
    nr = max(0, int(nms_radius))
    br = max(0, int(basin_radius))

    b = int(logits.shape[0])
    device, dtype = logits.device, logits.dtype
    temp = _as_batch_scalar(temperature, b, device, dtype, min_value=1e-4)
    score_map = logits.view(b, 1, side, side)

    if nr > 0:
        pooled = F.max_pool2d(
            score_map, kernel_size=2 * nr + 1, stride=1, padding=nr
        )
        is_peak = score_map >= pooled
        nms_scores = score_map.masked_fill(~is_peak, torch.finfo(dtype).min)
    else:
        nms_scores = score_map

    peak_values, peak_indices = torch.topk(
        nms_scores.view(b, -1), k=k, dim=1
    )

    # Fallback if pathological plateaus/NMS produce non-finite selections.
    bad = ~torch.isfinite(peak_values)
    if bad.any():
        raw_values, raw_indices = torch.topk(logits, k=k, dim=1)
        peak_values = torch.where(bad, raw_values, peak_values)
        peak_indices = torch.where(bad, raw_indices, peak_indices)

    peak_y = torch.div(peak_indices, side, rounding_mode='floor')
    peak_x = peak_indices.remainder(side)

    ys = torch.arange(side, device=device).view(1, 1, side, 1)
    xs = torch.arange(side, device=device).view(1, 1, 1, side)
    masks_2d = (
        (ys - peak_y.view(b, k, 1, 1)).abs() <= br
    ) & (
        (xs - peak_x.view(b, k, 1, 1)).abs() <= br
    )
    mode_masks = masks_2d.reshape(b, k, n)

    scaled = logits[:, None, :] / temp.view(b, 1, 1)
    neg_large = torch.finfo(dtype).min
    local_logits = scaled.masked_fill(~mode_masks, neg_large)
    mode_probabilities = torch.softmax(local_logits, dim=-1)

    disp_y, disp_x = torch.meshgrid(
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        indexing='ij',
    )
    displacement = torch.stack(
        (disp_x.reshape(-1), disp_y.reshape(-1)), dim=1
    )  # [N,2]

    mode_offsets = torch.einsum(
        'bkn,nd->bkd', mode_probabilities, displacement
    )
    mode_peak_offsets = displacement[peak_indices]

    return (
        mode_offsets,
        mode_peak_offsets,
        peak_indices,
        mode_probabilities,
        mode_masks,
    )


def _candidate_translation_scores(
    reference: torch.Tensor,
    source: torch.Tensor,
    rgb_target: torch.Tensor,
    tir_target: torch.Tensor,
    rgb_valid: torch.Tensor,
    tir_valid: torch.Tensor,
    local_weight: torch.Tensor,
    candidate_offsets: torch.Tensor,
    feature_weight: float = 0.65,
    target_weight: float = 0.35,
    padding_mode: str = 'border',
    align_corners: bool = True,
):
    """Score translation candidates using target-localized cross-modal evidence.

    Parameters
    ----------
    candidate_offsets : [B,M,2] in feature pixels.

    Scores:
      feature_score = weighted cosine similarity in [0,1]
      target_score  = weighted soft IoU of RGB/TIR targetness in [0,1]
      evidence      = wf * feature_score + wt * target_score
    """
    if candidate_offsets.ndim != 3 or candidate_offsets.shape[-1] != 2:
        raise ValueError(
            f"candidate_offsets must be [B,M,2], got {tuple(candidate_offsets.shape)}"
        )

    b, m, _ = candidate_offsets.shape
    _, _, h, w = reference.shape
    dtype = reference.dtype
    device = reference.device

    ref_n = F.normalize(reference.float(), p=2, dim=1, eps=1e-6)
    src = source
    src_valid = resize_mask(tir_valid, source).float()

    rgb_t = rgb_target.float().clamp(0, 1)
    tir_t = tir_target.float().clamp(0, 1)
    rgb_v = resize_mask(rgb_valid, rgb_target).float()

    weight0 = local_weight.float().clamp_min(0) * rgb_v

    feature_scores = []
    target_scores = []

    identity = torch.ones(b, 2, device=device, dtype=dtype)

    for j in range(m):
        off = candidate_offsets[:, j].float()
        off_norm = torch.stack(
            (
                off[:, 0] / max(w - 1, 1),
                off[:, 1] / max(h - 1, 1),
            ),
            dim=1,
        )

        warped_src = warp_with_geometry(
            src,
            identity,
            off_norm.to(dtype=dtype),
            output_hw=(h, w),
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        warped_valid = warp_with_geometry(
            src_valid,
            identity.to(dtype=src_valid.dtype),
            off_norm.to(dtype=src_valid.dtype),
            output_hw=(h, w),
            padding_mode='zeros',
            align_corners=align_corners,
        )
        warped_valid = (warped_valid >= 0.999).float()

        src_n = F.normalize(warped_src.float(), p=2, dim=1, eps=1e-6)
        cos = (ref_n * src_n).sum(1, keepdim=True)
        cos01 = ((cos + 1.0) * 0.5).clamp(0, 1)

        ww = weight0 * warped_valid
        den = ww.sum((-2, -1)).clamp_min(1e-6)
        feat = (cos01 * ww).sum((-2, -1)) / den

        warped_t = warp_with_geometry(
            tir_t,
            identity.to(dtype=tir_t.dtype),
            off_norm.to(dtype=tir_t.dtype),
            output_hw=(h, w),
            padding_mode='zeros',
            align_corners=align_corners,
        ).clamp(0, 1)

        a = rgb_t * ww
        bt = warped_t * ww
        inter = (a * bt).sum((-2, -1))
        union = (a + bt - a * bt).sum((-2, -1)).clamp_min(1e-6)
        tiou = inter / union

        feature_scores.append(feat[:, 0])
        target_scores.append(tiou[:, 0])

    feature_scores = torch.stack(feature_scores, dim=1)
    target_scores = torch.stack(target_scores, dim=1)

    wf = max(float(feature_weight), 0.0)
    wt = max(float(target_weight), 0.0)
    norm = max(wf + wt, 1e-6)
    wf, wt = wf / norm, wt / norm
    evidence = wf * feature_scores + wt * target_scores

    return evidence, feature_scores, target_scores


def global_multilocal_soft_aggregation(
    reference: torch.Tensor,
    source: torch.Tensor,
    rgb_target: torch.Tensor,
    tir_target: torch.Tensor,
    rgb_valid: torch.Tensor,
    tir_valid: torch.Tensor,
    local_weight: torch.Tensor,
    coarse_anchor: torch.Tensor,
    global_offset: torch.Tensor,
    global_probability: torch.Tensor,
    logits: torch.Tensor,
    search_radius: int,
    mode_topk: int = 3,
    mode_nms_radius: int = 4,
    local_basin_radius: int = 3,
    local_temperature: float = 0.10,
    aggregation_temperature: float = 0.12,
    global_prior: float = 0.60,
    feature_weight: float = 0.65,
    target_weight: float = 0.35,
    local_window_sigma: float = 0.10,
    padding_mode: str = 'border',
    align_corners: bool = True,
):
    """GMLS coarse estimator.

    Candidate 0 is the historical global-soft offset.
    Candidates 1..K are local-soft offsets from distinct modes.
    Final output is a soft mixture of all candidates.
    """
    (
        local_offsets,
        local_peak_offsets,
        local_peak_indices,
        local_probabilities,
        local_masks,
    ) = distinct_local_soft_modes(
        logits=logits,
        search_radius=search_radius,
        topk=mode_topk,
        nms_radius=mode_nms_radius,
        basin_radius=local_basin_radius,
        temperature=local_temperature,
    )

    b, k, _ = local_offsets.shape
    global_vec = global_offset.reshape(b, 2)
    candidates = torch.cat(
        (global_vec[:, None, :], local_offsets), dim=1
    )  # [B,K+1,2]

    _, _, h, w = reference.shape
    window = gaussian_window(
        coarse_anchor, h, w, sigma=local_window_sigma
    )
    score_weight = window * local_weight.float().clamp_min(0)

    evidence, feat_score, target_score = _candidate_translation_scores(
        reference=reference,
        source=source,
        rgb_target=rgb_target,
        tir_target=tir_target,
        rgb_valid=rgb_valid,
        tir_valid=tir_valid,
        local_weight=score_weight,
        candidate_offsets=candidates,
        feature_weight=feature_weight,
        target_weight=target_weight,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # Priors:
    # - global candidate receives a configurable stable prior mass;
    # - remaining prior mass is distributed across local modes in proportion
    #   to their historical global-soft probability mass.
    local_mass = (
        global_probability[:, None, :] * local_masks.to(global_probability.dtype)
    ).sum(dim=-1).clamp_min(1e-8)
    local_mass = local_mass / local_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)

    gp = min(max(float(global_prior), 1e-4), 1.0 - 1e-4)
    global_prior_tensor = torch.full(
        (b, 1), gp, device=logits.device, dtype=logits.dtype
    )
    local_prior = (1.0 - gp) * local_mass
    candidate_prior = torch.cat(
        (global_prior_tensor, local_prior), dim=1
    ).clamp_min(1e-8)

    temp = max(float(aggregation_temperature), 1e-4)
    aggregation_logits = evidence / temp + candidate_prior.log()
    candidate_weights = torch.softmax(aggregation_logits, dim=1)

    hybrid = (candidate_weights[..., None] * candidates).sum(dim=1)

    # Useful diagnostics.
    entropy = -(
        candidate_weights * candidate_weights.clamp_min(1e-8).log()
    ).sum(dim=1)
    max_entropy = math.log(max(int(candidate_weights.shape[1]), 2))
    aggregation_confidence = (
        1.0 - entropy / max(max_entropy, 1e-6)
    ).clamp(0, 1)

    return {
        'offset': hybrid.view(-1, 2, 1, 1),
        'candidate_offsets': candidates,
        'candidate_weights': candidate_weights,
        'candidate_evidence': evidence,
        'candidate_feature_score': feat_score,
        'candidate_target_score': target_score,
        'candidate_prior': candidate_prior,
        'aggregation_confidence': aggregation_confidence.view(-1, 1, 1, 1),
        'local_offsets': local_offsets,
        'local_peak_offsets': local_peak_offsets,
        'local_peak_indices': local_peak_indices,
        'local_probabilities': local_probabilities,
        'local_masks': local_masks,
        'local_mass': local_mass,
        'dominant_local_offset': local_offsets[:, 0].view(-1, 2, 1, 1),
        'dominant_peak_offset': local_peak_offsets[:, 0].view(-1, 2, 1, 1),
        'dominant_peak_index': local_peak_indices[:, 0],
    }


class GMLSPathQualityHead(nn.Module):
    """Shared tiny quality estimator: order [original GMLS, DMLS candidate].

    12-dimensional, GT-free inference descriptors. The last layer starts
    at zero, hence a fresh model defaults to original GMLS (router margin > 0).
    """
    def __init__(self, in_channels: int = 12, hidden_channels: int = 32):
        super().__init__()
        h = max(8, int(hidden_channels))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        b, n, c = descriptors.shape
        if n != 2 or c != 12:
            raise ValueError(f'Expected [B,2,12] descriptors, got {descriptors.shape}')
        return self.mlp(descriptors.reshape(-1, c)).reshape(b, 2)


def correct_gmls_with_dmls(
    gmls: Dict[str, torch.Tensor],
    global_soft_offset: torch.Tensor,
    quality_head: nn.Module,
    search_radius: int,
    route_logit_margin: float = 0.20,
) -> Dict[str, torch.Tensor]:
    """Preserve original GMLS as BASE; hard switch only on predicted quality.

    No second alignment unit, no extra feature warps, no absolute-coordinate
    interpolation between two competing modes. The selected coarse result is
    passed to the SAME original Fine/Scale and final one-warp geometry.

    Hard routing is trained with separate bbox-derived BCEWithLogits and
    route losses in the detector wrapper; the route decision is not
    differentiated through its boolean threshold.
    """
    d_g = gmls['offset'].reshape(-1, 2).float()
    d_d = gmls['dominant_local_offset'].reshape(-1, 2).float()
    d_global = global_soft_offset.reshape(-1, 2).float()
    w = gmls['candidate_weights'].float()                 # [B,4]
    feat = gmls['candidate_feature_score'].float()       # [B,4]
    target = gmls['candidate_target_score'].float()
    evidence = gmls['candidate_evidence'].float()
    prior = gmls['candidate_prior'].float()
    conf = gmls['aggregation_confidence'].reshape(-1).float()
    local_mass = gmls['local_mass'].float()
    if w.shape[1] < 2:
        raise RuntimeError('GMLS must contain global and dominant-local candidates.')
    radius = max(float(search_radius), 1.0)
    pair_dist = torch.linalg.vector_norm(d_g - d_d, dim=1) / (2.8284271247 * radius)
    d_g_dist_global = torch.linalg.vector_norm(d_g - d_global, dim=1) / (2.8284271247 * radius)
    d_d_dist_global = torch.linalg.vector_norm(d_d - d_global, dim=1) / (2.8284271247 * radius)
    gmls_descriptors = torch.stack((
        (w * feat).sum(1),
        (w * target).sum(1),
        (w * evidence).sum(1),
        (w * prior).sum(1),
        w.max(1).values,
        conf,
        (w[:,1:] * local_mass).sum(1),
        d_g[:,0] / radius, d_g[:,1] / radius,
        pair_dist, d_g_dist_global, torch.zeros_like(pair_dist),
    ), dim=-1)
    dmls_descriptors = torch.stack((
        feat[:,1], target[:,1], evidence[:,1], prior[:,1], w[:,1],
        conf, local_mass[:,0],
        d_d[:,0] / radius, d_d[:,1] / radius,
        pair_dist, d_d_dist_global, torch.ones_like(pair_dist),
    ), dim=-1)
    descriptors = torch.stack((gmls_descriptors, dmls_descriptors), dim=1)
    logits = quality_head(descriptors)
    # Select DMLS only when its predicted log-odds CLEARLY exceed GMLS.
    margin = max(float(route_logit_margin), 0.0)
    route_logit = logits[:, 1] - logits[:, 0] - margin
    choose_dmls = (route_logit.detach() > 0.0)
    chosen = torch.where(choose_dmls[:, None], d_d, d_g)
    return {
        'offset': chosen[:, :, None, None],
        'gmls_offset': d_g[:, :, None, None],
        'dmls_offset': d_d[:, :, None, None],
        'quality_descriptors': descriptors,
        'quality_logits': logits,
        'quality_scores': torch.sigmoid(logits),
        'route_logit': route_logit,
        'choose_dmls': choose_dmls,
        'margin': margin,
    }



def fine_residual_translation(
    reference: torch.Tensor,
    source: torch.Tensor,
    weight_map: torch.Tensor,
    coarse_offset: torch.Tensor,
    radius: int = 2,
    temperature: float = 0.05,
    reference_valid_mask: Optional[torch.Tensor] = None,
    source_valid_mask: Optional[torch.Tensor] = None,
    align_corners: bool = True,
    adaptive_temperature_head: Optional[nn.Module] = None,
    detach_statistics: bool = True,
    return_temperature: bool = False,
):
    """Small residual translation search around the coarse result.

    If ``adaptive_temperature_head`` is supplied, the correlation logits are
    computed once, three GT-free correlation descriptors are extracted, and a
    bounded per-image temperature is predicted.  With the predictor's
    baseline-preserving initialization this is exactly the historical fixed-T
    behavior at initialization.
    """
    if reference.shape[-2:] != source.shape[-2:]:
        source = F.interpolate(
            source,
            reference.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    b, _, h, w = reference.shape
    coarse = coarse_offset.float().reshape(b, 2)
    coarse_norm = torch.stack(
        (
            coarse[:, 0] / max(w - 1, 1),
            coarse[:, 1] / max(h - 1, 1),
        ),
        dim=1,
    )

    identity_scale = torch.ones(
        b, 2, device=source.device, dtype=source.dtype
    )

    coarse_source = warp_with_geometry(
        source,
        identity_scale,
        coarse_norm.to(dtype=source.dtype),
        output_hw=(h, w),
        padding_mode='border',
        align_corners=align_corners,
    )

    src_valid = resize_mask(source_valid_mask, source).float()
    coarse_valid = warp_with_geometry(
        src_valid,
        identity_scale.to(dtype=src_valid.dtype),
        coarse_norm.to(dtype=src_valid.dtype),
        output_hw=(h, w),
        padding_mode='zeros',
        align_corners=align_corners,
    )
    coarse_valid = (coarse_valid >= 0.999).to(dtype=reference.dtype)

    # Compute the expensive correlation map once.  The initial fixed-T readout
    # is retained as fallback and also gives us the raw logits.
    residual, confidence, prob, logits = correlation_translation(
        reference=reference,
        source=coarse_source,
        weight_map=weight_map,
        radius=int(radius),
        temperature=float(temperature),
        reference_valid_mask=reference_valid_mask,
        source_valid_mask=coarse_valid,
    )

    used_temperature = torch.full(
        (b, 1), float(temperature),
        device=logits.device,
        dtype=logits.dtype,
    )

    if adaptive_temperature_head is not None:
        stats = correlation_statistics(logits)
        if detach_statistics:
            stats = stats.detach()
        used_temperature = adaptive_temperature_head(stats).to(dtype=logits.dtype)
        residual, confidence, prob = softargmax_translation_from_logits(
            logits=logits,
            search_radius=int(radius),
            temperature=used_temperature,
        )

    refined = coarse_offset.float() + residual.float()
    output = (refined, residual, confidence, prob, logits)
    if return_temperature:
        return output + (used_temperature,)
    return output


def build_direct_sampling_grid(
    sampling_scale: torch.Tensor,
    sampling_translation: torch.Tensor,
    height: int,
    width: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    b = int(sampling_scale.shape[0])
    ys = torch.linspace(0, 1, height, device=sampling_scale.device, dtype=sampling_scale.dtype)
    xs = torch.linspace(0, 1, width, device=sampling_scale.device, dtype=sampling_scale.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    px = xx.view(1, height, width).expand(b, -1, -1)
    py = yy.view(1, height, width).expand(b, -1, -1)
    sx = sampling_scale[:, 0].view(b, 1, 1)
    sy = sampling_scale[:, 1].view(b, 1, 1)
    tx = sampling_translation[:, 0].view(b, 1, 1)
    ty = sampling_translation[:, 1].view(b, 1, 1)
    qx = sx * px + tx
    qy = sy * py + ty
    grid = torch.stack((2 * qx - 1, 2 * qy - 1), dim=-1)
    return grid if dtype is None else grid.to(dtype=dtype)


def warp_with_geometry(
    source: torch.Tensor,
    sampling_scale: torch.Tensor,
    sampling_translation: torch.Tensor,
    output_hw: Optional[Tuple[int, int]] = None,
    padding_mode: str = 'border',
    align_corners: bool = True,
) -> torch.Tensor:
    if output_hw is None:
        output_hw = source.shape[-2:]
    h, w = map(int, output_hw)
    grid = build_direct_sampling_grid(
        sampling_scale, sampling_translation, h, w, dtype=source.dtype
    ).to(device=source.device)
    return F.grid_sample(
        source, grid, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners
    )


class RLSFAGTGAScaleUnit(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 32,
        coarse_radius: int = 20,
        coarse_temperature: float = 0.10,
        coarse_local_soft_enabled: bool = False,
        coarse_local_radius: int = 3,
        coarse_local_temperature: Optional[float] = None,

        # GMLS: historical global-soft + multiple local-soft candidates.
        coarse_gmls_enabled: bool = False,
        coarse_gmls_topk: int = 3,
        coarse_gmls_nms_radius: int = 4,
        coarse_gmls_aggregation_temperature: float = 0.12,
        coarse_gmls_global_prior: float = 0.60,
        coarse_gmls_feature_weight: float = 0.65,
        coarse_gmls_target_weight: float = 0.35,

        # GMLS v2: original GMLS primary + DMLS conditional correction.
        gmls_v2_correction_enabled: bool = False,
        gmls_v2_quality_hidden: int = 32,
        gmls_v2_route_logit_margin: float = 0.20,

        # Adaptive versions of Fine T and Local-GTGA sigma only.
        # GMLS/DMLS coarse readout stays EXACTLY on the proven fixed parameters.
        adaptive_parameters_cfg: Optional[Dict[str, Any]] = None,

        fine_translation_enabled: bool = False,
        fine_translation_radius: int = 2,
        fine_translation_temperature: float = 0.05,
        anchor_power: float = 4.0,
        moment_power: float = 2.0,
        local_window_sigma: float = 0.10,
        moment_border_margin: int = 2,
        sigma_min: float = 0.005,
        scale_min: float = 0.50,
        scale_max: float = 2.00,
        use_targetness_for_correlation: bool = True,
        padding_mode: str = 'border',
        align_corners: bool = True,
    ):
        super().__init__()
        c = int(hidden_channels)
        self.coarse_radius = int(coarse_radius)
        self.coarse_temperature = float(coarse_temperature)

        # DMLS changes only the readout of the full global correlation map.
        self.coarse_local_soft_enabled = bool(coarse_local_soft_enabled)
        self.coarse_local_radius = max(0, int(coarse_local_radius))
        self.coarse_local_temperature = (
            float(coarse_temperature)
            if coarse_local_temperature is None
            else float(coarse_local_temperature)
        )

        self.coarse_gmls_enabled = bool(coarse_gmls_enabled)
        self.coarse_gmls_topk = max(1, int(coarse_gmls_topk))
        self.coarse_gmls_nms_radius = max(0, int(coarse_gmls_nms_radius))
        self.coarse_gmls_aggregation_temperature = float(
            coarse_gmls_aggregation_temperature
        )
        self.coarse_gmls_global_prior = float(coarse_gmls_global_prior)
        self.coarse_gmls_feature_weight = float(coarse_gmls_feature_weight)
        self.coarse_gmls_target_weight = float(coarse_gmls_target_weight)

        self.gmls_v2_correction_enabled = bool(gmls_v2_correction_enabled)
        if self.gmls_v2_correction_enabled and not self.coarse_gmls_enabled:
            raise ValueError('GMLS-v2 correction requires coarse_gmls_enabled=true.')
        if self.gmls_v2_correction_enabled and self.coarse_gmls_topk != 3:
            raise ValueError('GMLS-v2 expects 3 local modes, matching v1 baseline.')
        self.gmls_v2_quality_hidden = max(8, int(gmls_v2_quality_hidden))
        self.gmls_v2_route_logit_margin = max(0.0, float(gmls_v2_route_logit_margin))

        self.fine_translation_enabled = bool(fine_translation_enabled)
        self.fine_translation_radius = int(fine_translation_radius)
        self.fine_translation_temperature = float(fine_translation_temperature)
        self.anchor_power = float(anchor_power)
        self.moment_power = float(moment_power)
        self.local_window_sigma = float(local_window_sigma)
        self.moment_border_margin = int(moment_border_margin)
        self.sigma_min = float(sigma_min)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.use_targetness_for_correlation = bool(use_targetness_for_correlation)
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)

        # --------------------------------------------------------------
        # Adaptive geometry parameters.
        # --------------------------------------------------------------
        acfg = dict(adaptive_parameters_cfg or {})
        self.adaptive_parameters_enabled = bool(acfg.get('enabled', False))
        self.adaptive_hidden_channels = max(4, int(acfg.get('hidden_channels', 16)))
        self.adaptive_detach_statistics = bool(acfg.get('detach_statistics', True))

        fcfg = dict(acfg.get('fine_temperature', {}) or {})
        scfg = dict(acfg.get('local_window_sigma', {}) or {})

        # Deliberately disabled in this variant.  Keeping the attribute avoids
        # breaking debug/checkpoint helper code, but no parameters are created.
        # Both GMLS and DMLS use the historical fixed coarse_local_temperature.
        self.adaptive_dmls_temperature_head = None

        self.adaptive_fine_temperature_head = None
        if (
            self.adaptive_parameters_enabled
            and bool(fcfg.get('enabled', True))
            and self.fine_translation_enabled
            and self.fine_translation_radius > 0
        ):
            self.adaptive_fine_temperature_head = AdaptiveBoundedScalar(
                in_channels=3,
                hidden_channels=self.adaptive_hidden_channels,
                min_value=float(fcfg.get('min', 0.02)),
                max_value=float(fcfg.get('max', 0.10)),
                init_value=float(fcfg.get('init', self.fine_translation_temperature)),
            )

        self.adaptive_local_sigma_head = None
        if self.adaptive_parameters_enabled and bool(scfg.get('enabled', True)):
            self.adaptive_local_sigma_head = AdaptiveBoundedScalar(
                in_channels=5,
                hidden_channels=self.adaptive_hidden_channels,
                min_value=float(scfg.get('min', 0.04)),
                max_value=float(scfg.get('max', 0.16)),
                init_value=float(scfg.get('init', self.local_window_sigma)),
            )

        self.rgb_proj = LazyProjection(c)
        self.tir_proj = LazyProjection(c)
        self.rgb_target_head = TargetMapHead(c)
        self.tir_target_head = TargetMapHead(c)
        self.coarse_target_head = nn.Sequential(
            ConvBNAct(c * 3, c, 3),
            nn.Conv2d(c, 1, 1, bias=True),
        )
        nn.init.zeros_(self.coarse_target_head[-1].weight)
        nn.init.zeros_(self.coarse_target_head[-1].bias)

        # Allocated only for v2 runs. v1 is preserved when disabled.
        self.gmls_v2_quality_head = (
            GMLSPathQualityHead(hidden_channels=self.gmls_v2_quality_hidden)
            if self.gmls_v2_correction_enabled else None
        )

    @staticmethod
    def _resize_like(source, reference):
        if source.shape[-2:] == reference.shape[-2:]:
            return source
        return F.interpolate(source, reference.shape[-2:], mode='bilinear', align_corners=False)

    def estimate_geometry(self, rgb_feat, tir_feat, rgb_valid_mask=None, tir_valid_mask=None):
        tir_feat = self._resize_like(tir_feat, rgb_feat)
        r = self.rgb_proj(rgb_feat)
        t = self.tir_proj(tir_feat)
        rgb_logits = self.rgb_target_head(r)
        tir_logits = self.tir_target_head(t)
        rgb_target_raw = torch.sigmoid(rgb_logits)
        tir_target_raw = torch.sigmoid(tir_logits)
        coarse_logits = self.coarse_target_head(torch.cat((r, t, (r - t).abs()), dim=1))
        coarse_targetness = torch.sigmoid(coarse_logits)
        rgb_valid = resize_mask(rgb_valid_mask, rgb_target_raw)
        tir_valid = resize_mask(tir_valid_mask, tir_target_raw)
        rgb_safe = erode_safe_mask(rgb_valid, self.moment_border_margin)
        tir_safe = erode_safe_mask(tir_valid, self.moment_border_margin)

        coarse_anchor_distribution = normalized_power_distribution(
            coarse_targetness, rgb_valid, power=self.anchor_power
        )
        coarse_anchor = distribution_center(coarse_anchor_distribution)

        coarse_weight = coarse_targetness * rgb_valid if self.use_targetness_for_correlation else rgb_valid

        # --------------------------------------------------------------
        # Historical GLOBAL soft expectation (kept for diagnostics).
        # --------------------------------------------------------------
        (
            coarse_offset_global,
            coarse_conf_global,
            coarse_prob_global,
            coarse_corr_logits,
        ) = correlation_translation(
            r, t, coarse_weight,
            self.coarse_radius, self.coarse_temperature,
            rgb_valid, tir_valid,
        )

        # --------------------------------------------------------------
        # Fixed coarse + adaptive scale-window policy.
        #
        # IMPORTANT: GMLS and DMLS retain the EXACT proven fixed local
        # temperature.  Only the later Fine stage and Local-GTGA window are
        # adaptive in this variant.  This prevents the adaptive heads from
        # perturbing the coarse correlation curriculum / GMLS candidate set.
        # --------------------------------------------------------------
        coarse_stats = correlation_statistics(coarse_corr_logits)
        coarse_local_temperature_used = self.coarse_local_temperature

        # Target-spread descriptor is available before Local-GTGA and therefore
        # does not create a circular dependence on local_window_sigma.
        coarse_target_sigma = centered_sigma(
            coarse_anchor_distribution,
            coarse_anchor,
            sigma_min=1e-6,
        )
        sigma_descriptor = torch.cat(
            (coarse_stats, coarse_target_sigma.float()), dim=1
        )
        if self.adaptive_detach_statistics:
            sigma_descriptor = sigma_descriptor.detach()

        if self.adaptive_local_sigma_head is not None:
            local_window_sigma_used = self.adaptive_local_sigma_head(
                sigma_descriptor
            )
        else:
            local_window_sigma_used = self.local_window_sigma

        # --------------------------------------------------------------
        # Coarse readout hierarchy:
        #   GMLS (new) > DMLS (single local mode) > historical global soft.
        # --------------------------------------------------------------
        gmls_debug = None
        correction_debug = None

        # Evaluation-only switch. Training ALWAYS uses the original GMLS-v2
        # dynamic route, original quality BCE and original route BCE.
        # Supported: auto (exact v2), force_dmls, force_gmls, fast_dmls.
        # Never set this variable while running validation inside training:
        # auto is used unless the model has explicitly switched to eval().
        eval_mode = 'auto' if self.training else os.environ.get(
            'GMLS_V2_EVAL_MODE', 'auto'
        ).strip().lower()
        if eval_mode not in ('auto', 'force_dmls', 'force_gmls', 'fast_dmls'):
            raise ValueError(f'Unsupported GMLS_V2_EVAL_MODE={eval_mode!r}')

        if self.coarse_gmls_enabled and eval_mode == 'fast_dmls':
            # FAST inference only: reuse the same full correlation logits,
            # but skip expensive GMLS candidate-level warp/scoring and head.
            # Training geometry and its gradients are UNCHANGED.
            (
                coarse_offset, coarse_conf, coarse_prob,
                coarse_basin_mask, coarse_peak_offset, coarse_peak_index,
            ) = dominant_mode_local_softargmax(
                logits=coarse_corr_logits,
                search_radius=self.coarse_radius,
                basin_radius=self.coarse_local_radius,
                temperature=self.coarse_local_temperature,
            )
            coarse_offset_dmls = coarse_offset

        elif self.coarse_gmls_enabled:
            gmls_debug = global_multilocal_soft_aggregation(
                reference=r,
                source=t,
                rgb_target=rgb_target_raw,
                tir_target=tir_target_raw,
                rgb_valid=rgb_valid,
                tir_valid=tir_valid,
                local_weight=coarse_weight,
                coarse_anchor=coarse_anchor,
                global_offset=coarse_offset_global,
                global_probability=coarse_prob_global,
                logits=coarse_corr_logits,
                search_radius=self.coarse_radius,
                mode_topk=self.coarse_gmls_topk,
                mode_nms_radius=self.coarse_gmls_nms_radius,
                local_basin_radius=self.coarse_local_radius,
                # Keep GMLS EXACTLY on the proven fixed coarse settings.
                local_temperature=self.coarse_local_temperature,
                aggregation_temperature=self.coarse_gmls_aggregation_temperature,
                global_prior=self.coarse_gmls_global_prior,
                feature_weight=self.coarse_gmls_feature_weight,
                target_weight=self.coarse_gmls_target_weight,
                local_window_sigma=self.local_window_sigma,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            coarse_offset = gmls_debug['offset']
            coarse_conf = gmls_debug['aggregation_confidence']

            # Probability over the 41x41 search grid is not uniquely defined
            # after candidate-level aggregation. Keep historical probability
            # for loss/debug API compatibility; expose candidate weights below.
            coarse_prob = coarse_prob_global

            coarse_peak_offset = gmls_debug['dominant_peak_offset']
            coarse_peak_index = gmls_debug['dominant_peak_index']
            coarse_basin_mask = gmls_debug['local_masks'][:, 0]
            coarse_offset_dmls = gmls_debug['dominant_local_offset']
            if self.gmls_v2_correction_enabled:
                correction_debug = correct_gmls_with_dmls(
                    gmls=gmls_debug,
                    global_soft_offset=coarse_offset_global,
                    quality_head=self.gmls_v2_quality_head,
                    search_radius=self.coarse_radius,
                    route_logit_margin=self.gmls_v2_route_logit_margin,
                )
                # Counterfactual modes are strictly evaluation-only.
                # They recompute the ORIGINAL Fine/Scale using the selected
                # coarse, rather than post-hoc swapping an IoU column.
                if eval_mode == 'force_dmls':
                    correction_debug['choose_dmls'] = torch.ones_like(
                        correction_debug['choose_dmls'], dtype=torch.bool
                    )
                    correction_debug['offset'] = coarse_offset_dmls
                elif eval_mode == 'force_gmls':
                    correction_debug['choose_dmls'] = torch.zeros_like(
                        correction_debug['choose_dmls'], dtype=torch.bool
                    )
                    correction_debug['offset'] = gmls_debug['offset']
                coarse_offset = correction_debug['offset']
                coarse_conf = torch.where(
                    correction_debug['choose_dmls'].view(-1, 1, 1, 1),
                    gmls_debug['candidate_weights'][:, 1].view(-1, 1, 1, 1),
                    gmls_debug['aggregation_confidence'],
                )

        elif self.coarse_local_soft_enabled:
            (
                coarse_offset,
                coarse_conf,
                coarse_prob,
                coarse_basin_mask,
                coarse_peak_offset,
                coarse_peak_index,
            ) = dominant_mode_local_softargmax(
                logits=coarse_corr_logits,
                search_radius=self.coarse_radius,
                basin_radius=self.coarse_local_radius,
                temperature=self.coarse_local_temperature,
            )
            coarse_offset_dmls = coarse_offset

        else:
            coarse_offset = coarse_offset_global
            coarse_offset_dmls = coarse_offset_global
            coarse_conf = coarse_conf_global
            coarse_prob = coarse_prob_global
            coarse_basin_mask = torch.ones_like(
                coarse_prob_global, dtype=torch.bool
            )
            coarse_peak_index = coarse_corr_logits.argmax(dim=1)

            r_search = int(self.coarse_radius)
            side = 2 * r_search + 1
            py = torch.div(
                coarse_peak_index, side, rounding_mode='floor'
            ).to(dtype=coarse_prob_global.dtype)
            px = coarse_peak_index.remainder(side).to(
                dtype=coarse_prob_global.dtype
            )
            coarse_peak_offset = torch.stack(
                (px - r_search, py - r_search), dim=1
            ).view(-1, 2, 1, 1)

        # --------------------------------------------------------------
        # Coarse-to-fine translation refinement
        # --------------------------------------------------------------
        # No new network and no new loss:
        # dbg['coarse_offset'] below becomes the FINAL coarse+fine offset,
        # so the existing Smooth-L1 translation supervision trains it.
        if self.fine_translation_enabled and self.fine_translation_radius > 0:
            (
                offset, fine_offset, fine_conf, fine_prob, fine_corr_logits,
                fine_translation_temperature_used,
            ) = fine_residual_translation(
                reference=r,
                source=t,
                weight_map=coarse_weight,
                coarse_offset=coarse_offset,
                radius=self.fine_translation_radius,
                temperature=self.fine_translation_temperature,
                reference_valid_mask=rgb_valid,
                source_valid_mask=tir_valid,
                align_corners=self.align_corners,
                adaptive_temperature_head=self.adaptive_fine_temperature_head,
                detach_statistics=self.adaptive_detach_statistics,
                return_temperature=True,
            )
        else:
            offset = coarse_offset
            fine_offset = torch.zeros_like(coarse_offset)
            fine_conf = torch.ones_like(coarse_conf)
            fine_prob = None
            fine_corr_logits = None
            fine_translation_temperature_used = torch.full(
                (offset.shape[0], 1),
                self.fine_translation_temperature,
                device=offset.device,
                dtype=offset.dtype,
            )

        _, _, h, w = rgb_target_raw.shape
        dx = offset[:, 0, 0, 0]
        dy = offset[:, 1, 0, 0]
        dx_norm = dx / max(w - 1, 1)
        dy_norm = dy / max(h - 1, 1)
        offset_norm = torch.stack((dx_norm, dy_norm), dim=1)

        source_center = coarse_anchor + offset_norm
        source_center_for_stats = source_center.clamp(0.0, 1.0)

        rgb_dist, rgb_window, rgb_local_target = local_target_distribution(
            rgb_target_raw, coarse_anchor, rgb_safe,
            local_window_sigma_used, self.moment_power,
        )
        tir_dist, tir_window, tir_local_target = local_target_distribution(
            tir_target_raw, source_center_for_stats, tir_safe,
            local_window_sigma_used, self.moment_power,
        )
        rgb_sigma = centered_sigma(rgb_dist, coarse_anchor, self.sigma_min)
        tir_sigma = centered_sigma(tir_dist, source_center_for_stats, self.sigma_min)
        rgb_local_mean = distribution_center(rgb_dist)
        tir_local_mean = distribution_center(tir_dist)
        scale_raw = tir_sigma / rgb_sigma.clamp_min(self.sigma_min)
        scale = scale_raw.clamp(self.scale_min, self.scale_max)

        # The only scale pivot is the UAV anchor, never image center.
        translation = source_center - scale * coarse_anchor

        return {
            'rgb_projected': r,
            'tir_projected': t,
            'rgb_target_logits': rgb_logits,
            'tir_target_logits': tir_logits,
            'rgb_targetness_raw': rgb_target_raw,
            'tir_targetness_raw': tir_target_raw,
            'coarse_targetness_logits': coarse_logits,
            'coarse_targetness': coarse_targetness,
            'rgb_valid_mask': rgb_valid,
            'tir_valid_mask': tir_valid,
            'rgb_safe_mask': rgb_safe,
            'tir_safe_mask': tir_safe,
            'coarse_anchor_distribution': coarse_anchor_distribution,
            'coarse_anchor': coarse_anchor,

            # Existing loss reads "coarse_offset".  Keep that API but make it
            # the FINAL coarse+fine translation so no loss code must change.
            'coarse_offset': offset,
            'coarse_offset_normalized': offset_norm,

            # Extra diagnostics.
            # DMLS final coarse estimate used by Fine Translation.
            'coarse_offset_initial': coarse_offset,
            'coarse_correlation_confidence': coarse_conf,
            'coarse_probability': coarse_prob,
            'coarse_score_logits': coarse_corr_logits,

            # Historical global-soft estimate, kept ONLY for direct diagnosis.
            'coarse_offset_global_soft': coarse_offset_global,
            'coarse_correlation_confidence_global': coarse_conf_global,
            'coarse_probability_global': coarse_prob_global,

            # Dominant-mode / local-basin diagnostics.
            'coarse_local_soft_enabled': self.coarse_local_soft_enabled,
            'coarse_local_radius': self.coarse_local_radius,
            'coarse_local_temperature': self.coarse_local_temperature,
            'adaptive_parameters_enabled': self.adaptive_parameters_enabled,
            'adaptive_coarse_enabled': False,
            # Compatibility key: fixed in this variant (not adaptive).
            'adaptive_coarse_local_temperature': (
                coarse_local_temperature_used
                if torch.is_tensor(coarse_local_temperature_used)
                else torch.full(
                    (coarse_corr_logits.shape[0], 1),
                    float(coarse_local_temperature_used),
                    device=coarse_corr_logits.device,
                    dtype=coarse_corr_logits.dtype,
                )
            ),
            'adaptive_local_window_sigma': (
                local_window_sigma_used
                if torch.is_tensor(local_window_sigma_used)
                else torch.full(
                    (coarse_corr_logits.shape[0], 1),
                    float(local_window_sigma_used),
                    device=coarse_corr_logits.device,
                    dtype=coarse_corr_logits.dtype,
                )
            ),
            'coarse_correlation_statistics': coarse_stats,
            'coarse_target_sigma_for_adaptation': coarse_target_sigma,
            'coarse_basin_mask': coarse_basin_mask,
            'coarse_peak_offset': coarse_peak_offset,
            'coarse_peak_index': coarse_peak_index,
            'coarse_global_to_local_delta': (
                coarse_offset - coarse_offset_global
            ),

            # Explicit single-mode DMLS reference even when GMLS is active.
            'coarse_offset_dmls': coarse_offset_dmls,

            # GMLS diagnostics.
            'coarse_gmls_enabled': self.coarse_gmls_enabled,
            'coarse_gmls_topk': self.coarse_gmls_topk,
            'coarse_gmls_nms_radius': self.coarse_gmls_nms_radius,
            'coarse_offset_gmls': (
                None if gmls_debug is None else gmls_debug['offset']
            ),
            'coarse_gmls_candidate_offsets': (
                None if gmls_debug is None else gmls_debug['candidate_offsets']
            ),
            'coarse_gmls_candidate_weights': (
                None if gmls_debug is None else gmls_debug['candidate_weights']
            ),
            'coarse_gmls_candidate_evidence': (
                None if gmls_debug is None else gmls_debug['candidate_evidence']
            ),
            'coarse_gmls_candidate_feature_score': (
                None if gmls_debug is None else gmls_debug['candidate_feature_score']
            ),
            'coarse_gmls_candidate_target_score': (
                None if gmls_debug is None else gmls_debug['candidate_target_score']
            ),
            'coarse_gmls_candidate_prior': (
                None if gmls_debug is None else gmls_debug['candidate_prior']
            ),
            'coarse_gmls_aggregation_confidence': (
                None if gmls_debug is None else gmls_debug['aggregation_confidence']
            ),
            'coarse_gmls_local_offsets': (
                None if gmls_debug is None else gmls_debug['local_offsets']
            ),
            'coarse_gmls_local_peak_offsets': (
                None if gmls_debug is None else gmls_debug['local_peak_offsets']
            ),
            'coarse_gmls_local_mass': (
                None if gmls_debug is None else gmls_debug['local_mass']
            ),

            # v2: GMLS original and DMLS remain available as counterfactual
            # coarse diagnostics. Only the selected one drives Fine/Scale.
            'gmls_v2_correction_enabled': self.gmls_v2_correction_enabled,
            'coarse_offset_gmls_v2': (
                None if correction_debug is None else correction_debug['offset']
            ),
            'gmls_v2_quality_descriptors': (
                None if correction_debug is None else correction_debug['quality_descriptors']
            ),
            'gmls_v2_quality_logits': (
                None if correction_debug is None else correction_debug['quality_logits']
            ),
            'gmls_v2_quality_scores': (
                None if correction_debug is None else correction_debug['quality_scores']
            ),
            'gmls_v2_route_logit': (
                None if correction_debug is None else correction_debug['route_logit']
            ),
            'gmls_v2_choose_dmls': (
                None if correction_debug is None else correction_debug['choose_dmls']
            ),

            'fine_translation_enabled': self.fine_translation_enabled,
            'fine_translation_radius': self.fine_translation_radius,
            'fine_translation_temperature': self.fine_translation_temperature,
            'adaptive_fine_translation_temperature': fine_translation_temperature_used,
            'fine_offset': fine_offset,
            'fine_correlation_confidence': fine_conf,
            'fine_probability': fine_prob,
            'fine_score_logits': fine_corr_logits,
            'refined_offset': offset,
            'source_center': source_center,
            'source_center_for_stats': source_center_for_stats,
            'rgb_local_window': rgb_window,
            'tir_local_window': tir_window,
            'rgb_local_targetness': rgb_local_target,
            'tir_local_targetness': tir_local_target,
            'rgb_distribution': rgb_dist,
            'tir_distribution': tir_dist,
            'rgb_local_mean': rgb_local_mean,
            'tir_local_mean': tir_local_mean,
            'rgb_sigma': rgb_sigma,
            'tir_sigma': tir_sigma,
            'sampling_scale_raw': scale_raw,
            'sampling_scale': scale,
            'sampling_translation': translation,
        }

    def forward(self, rgb_feat, tir_feat, rgb_valid_mask=None, tir_valid_mask=None, return_debug=False):
        tir_feat = self._resize_like(tir_feat, rgb_feat)
        g = self.estimate_geometry(rgb_feat, tir_feat, rgb_valid_mask, tir_valid_mask)
        aligned_tir = warp_with_geometry(
            tir_feat, g['sampling_scale'], g['sampling_translation'],
            output_hw=rgb_feat.shape[-2:], padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        aligned_tir_local_target = warp_with_geometry(
            g['tir_local_targetness'], g['sampling_scale'], g['sampling_translation'],
            output_hw=rgb_feat.shape[-2:], padding_mode='zeros',
            align_corners=self.align_corners,
        ).clamp(0, 1)
        # --------------------------------------------------------------
        # Multi-channel alignment guidance for BHLR-v3.
        # Keep the historical joint support J for backward compatibility,
        # but DO NOT collapse modality-specific target evidence before BHLR.
        #
        #   C0: RGB target support in RGB/output coordinates
        #   C1: aligned TIR target support in RGB/output coordinates
        #   C2: historical cross-modal consensus J=sqrt(C0*C1)
        # --------------------------------------------------------------
        valid = g['rgb_valid_mask'].clamp(0, 1)
        rgb_target_support = (g['rgb_local_targetness'].clamp(0, 1) * valid).clamp(0, 1)
        tir_target_support = (aligned_tir_local_target * valid).clamp(0, 1)
        joint = torch.sqrt(
            (rgb_target_support * tir_target_support).clamp_min(0)
        ).clamp(0, 1)
        multi_target_guidance = torch.cat(
            [rgb_target_support, tir_target_support, joint],
            dim=1,
        )

        # Store the richer guidance in the geometry dictionary so
        # align_pyramid() can consume it even when return_debug=False.
        g['rgb_target_support'] = rgb_target_support
        g['aligned_tir_target_support'] = tir_target_support
        g['multi_target_guidance'] = multi_target_guidance

        debug = {
            **g,
            'aligned_tir': aligned_tir,
            'aligned_tir_local_targetness': aligned_tir_local_target,
            'rgb_target_support': rgb_target_support,
            'aligned_tir_target_support': tir_target_support,
            'joint_target_support': joint,
            'multi_target_guidance': multi_target_guidance,
            'bhlr_guidance_map': joint,
            'scale_pivot_rgb': g['coarse_anchor'],
            'scale_pivot_tir': g['source_center'],
            'center_preserving_scale': True,
            'frequency_refinement': False,
            'fine_translation': self.fine_translation_enabled,
            'fine_translation_radius': self.fine_translation_radius,
            'dominant_local_softargmax': self.coarse_local_soft_enabled,
            'dominant_local_radius': self.coarse_local_radius,
            'gmls_v2_correction': self.gmls_v2_correction_enabled,
            'global_multilocal_soft_aggregation': self.coarse_gmls_enabled,
            'gmls_topk': self.coarse_gmls_topk,
            'gmls_nms_radius': self.coarse_gmls_nms_radius,
            'scale_candidate_search': False,
        }
        if return_debug:
            return aligned_tir, joint, g, debug
        return aligned_tir, joint, g


class RLSFAGTGACenteredAlignment(BaseAlignment):
    VALID_SCALES = (3, 4, 5)

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg=cfg)
        self.reference_scale = int(self.cfg.get('reference_scale', 3))
        self.apply_scales = tuple(sorted({int(s) for s in self.cfg.get('apply_scales', [3, 4, 5])}))
        if self.reference_scale not in self.VALID_SCALES:
            raise ValueError(f'reference_scale invalid: {self.reference_scale}')
        if any(s not in self.VALID_SCALES for s in self.apply_scales):
            raise ValueError(f'apply_scales invalid: {self.apply_scales}')
        self.padding_mode = str(self.cfg.get('padding_mode', 'border'))
        self.align_corners = bool(self.cfg.get('align_corners', True))
        self.unit = RLSFAGTGAScaleUnit(
            hidden_channels=int(self.cfg.get('hidden_channels', 32)),
            coarse_radius=int(self.cfg.get('coarse_radius', 20)),
            coarse_temperature=float(self.cfg.get('coarse_temperature', 0.10)),

            # DMLS coarse readout.
            coarse_local_soft_enabled=bool(
                self.cfg.get('coarse_local_soft_enabled', False)
            ),
            coarse_local_radius=int(
                self.cfg.get('coarse_local_radius', 3)
            ),
            coarse_local_temperature=(
                None
                if self.cfg.get('coarse_local_temperature', None) is None
                else float(self.cfg.get('coarse_local_temperature'))
            ),

            coarse_gmls_enabled=bool(
                self.cfg.get('coarse_gmls_enabled', False)
            ),
            coarse_gmls_topk=int(
                self.cfg.get('coarse_gmls_topk', 3)
            ),
            coarse_gmls_nms_radius=int(
                self.cfg.get('coarse_gmls_nms_radius', 4)
            ),
            coarse_gmls_aggregation_temperature=float(
                self.cfg.get('coarse_gmls_aggregation_temperature', 0.12)
            ),
            coarse_gmls_global_prior=float(
                self.cfg.get('coarse_gmls_global_prior', 0.60)
            ),
            coarse_gmls_feature_weight=float(
                self.cfg.get('coarse_gmls_feature_weight', 0.65)
            ),
            coarse_gmls_target_weight=float(
                self.cfg.get('coarse_gmls_target_weight', 0.35)
            ),
            gmls_v2_correction_enabled=bool(
                self.cfg.get('gmls_v2_correction_enabled', False)
            ),
            gmls_v2_quality_hidden=int(self.cfg.get('gmls_v2_quality_hidden', 32)),
            gmls_v2_route_logit_margin=float(self.cfg.get('gmls_v2_route_logit_margin', 0.20)),

            adaptive_parameters_cfg=dict(
                self.cfg.get('adaptive_parameters', {}) or {}
            ),

            fine_translation_enabled=bool(self.cfg.get('fine_translation_enabled', False)),
            fine_translation_radius=int(self.cfg.get('fine_translation_radius', 2)),
            fine_translation_temperature=float(self.cfg.get('fine_translation_temperature', 0.05)),
            anchor_power=float(self.cfg.get('anchor_power', 4.0)),
            moment_power=float(self.cfg.get('moment_power', 2.0)),
            local_window_sigma=float(self.cfg.get('local_window_sigma', 0.10)),
            moment_border_margin=int(self.cfg.get('moment_border_margin', 2)),
            sigma_min=float(self.cfg.get('sigma_min', 0.005)),
            scale_min=float(self.cfg.get('scale_min', 0.50)),
            scale_max=float(self.cfg.get('scale_max', 2.00)),
            use_targetness_for_correlation=bool(self.cfg.get('use_targetness_for_correlation', True)),
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

    def warp_tensor(self, source, geometry, output_hw=None, padding_mode=None):
        return warp_with_geometry(
            source, geometry['sampling_scale'], geometry['sampling_translation'],
            output_hw=output_hw,
            padding_mode=self.padding_mode if padding_mode is None else str(padding_mode),
            align_corners=self.align_corners,
        )

    def align_pyramid(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        rgb_valid_mask: Optional[torch.Tensor] = None,
        tir_valid_mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        ref_idx = int(scale_to_index[self.reference_scale])
        rgb_ref = rgb_features[ref_idx]
        tir_ref = tir_features[ref_idx]
        if return_debug:
            aligned_ref, joint_ref, geometry, debug = self.unit(
                rgb_ref, tir_ref, rgb_valid_mask, tir_valid_mask, True
            )
        else:
            aligned_ref, joint_ref, geometry = self.unit(
                rgb_ref, tir_ref, rgb_valid_mask, tir_valid_mask, False
            )
            debug = None

        aligned = list(tir_features)
        support_by_scale = {}
        guidance_by_scale = {}

        # Reference 3-channel packet [RGB, aligned-TIR, joint].
        # The unit stores it in geometry so this path is available in both
        # normal training and debug forward passes.
        multi_ref = geometry.get('multi_target_guidance', None)
        if multi_ref is None:
            # Defensive fallback for old/custom alignment units.
            rgb_ref_support = geometry['rgb_local_targetness'].clamp(0, 1)
            tir_ref_support = geometry.get('aligned_tir_target_support', joint_ref).clamp(0, 1)
            multi_ref = torch.cat(
                [rgb_ref_support, tir_ref_support, joint_ref.clamp(0, 1)],
                dim=1,
            )

        if multi_ref.ndim != 4 or multi_ref.shape[1] != 3:
            raise RuntimeError(
                'multi_target_guidance must have shape [B,3,H,W], '
                f'got {tuple(multi_ref.shape)}'
            )

        for scale in self.apply_scales:
            idx = int(scale_to_index[scale])
            if scale == self.reference_scale:
                aligned[idx] = aligned_ref
            else:
                aligned[idx] = self.warp_tensor(
                    tir_features[idx], geometry, output_hw=rgb_features[idx].shape[-2:]
                )

            multi_scale = F.interpolate(
                multi_ref,
                size=rgb_features[idx].shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).clamp(0, 1)

            # Historical single-channel support is retained exactly as C2.
            support_by_scale[scale] = multi_scale[:, 2:3]
            guidance_by_scale[scale] = {
                'multi_target_guidance': multi_scale,
                'rgb_target_support': multi_scale[:, 0:1],
                'aligned_tir_target_support': multi_scale[:, 1:2],
                'joint_target_support': multi_scale[:, 2:3],
            }

        info = {
            'enabled': True,
            'type': 'rlsfa_gtga_centered',
            'algorithm': ('gmls_v2_original_gmls_primary+dmls_conditional_correction+fine+scale+one_warp' if self.unit.gmls_v2_correction_enabled else 'gmls_v1_global+multilocal_soft+fine+scale+one_warp'),
            'reference_scale': self.reference_scale,
            'apply_scales': list(self.apply_scales),
            'coarse_anchor': geometry['coarse_anchor'],
            'source_center': geometry['source_center'],
            'sampling_scale': geometry['sampling_scale'],
            'sampling_translation': geometry['sampling_translation'],
            'coarse_offset': geometry['coarse_offset'],
            'coarse_offset_initial': geometry['coarse_offset_initial'],
            'coarse_confidence': geometry['coarse_correlation_confidence'],

            # DMLS diagnostics at alignment-info level.
            'coarse_local_soft_enabled': bool(self.unit.coarse_local_soft_enabled),
            'coarse_local_radius': int(self.unit.coarse_local_radius),
            'adaptive_parameters_enabled': bool(self.unit.adaptive_parameters_enabled),
            'adaptive_coarse_enabled': False,
            'adaptive_coarse_local_temperature': geometry['adaptive_coarse_local_temperature'],
            'adaptive_fine_translation_temperature': geometry['adaptive_fine_translation_temperature'],
            'adaptive_local_window_sigma': geometry['adaptive_local_window_sigma'],
            'coarse_offset_global_soft': geometry['coarse_offset_global_soft'],
            'coarse_probability_global': geometry['coarse_probability_global'],
            'coarse_peak_offset': geometry['coarse_peak_offset'],
            'coarse_basin_mask': geometry['coarse_basin_mask'],

            'coarse_gmls_enabled': bool(self.unit.coarse_gmls_enabled),
            'coarse_offset_dmls': geometry['coarse_offset_dmls'],
            'coarse_offset_gmls': geometry['coarse_offset_gmls'],
            'coarse_gmls_candidate_offsets': geometry['coarse_gmls_candidate_offsets'],
            'coarse_gmls_candidate_weights': geometry['coarse_gmls_candidate_weights'],
            'coarse_gmls_candidate_evidence': geometry['coarse_gmls_candidate_evidence'],
            'coarse_gmls_candidate_feature_score': geometry['coarse_gmls_candidate_feature_score'],
            'coarse_gmls_candidate_target_score': geometry['coarse_gmls_candidate_target_score'],
            'coarse_gmls_candidate_prior': geometry['coarse_gmls_candidate_prior'],
            'coarse_gmls_aggregation_confidence': geometry['coarse_gmls_aggregation_confidence'],
            'gmls_v2_correction_enabled': bool(self.unit.gmls_v2_correction_enabled),
            'coarse_offset_gmls_v2': geometry['coarse_offset_gmls_v2'],
            'gmls_v2_quality_logits': geometry['gmls_v2_quality_logits'],
            'gmls_v2_quality_scores': geometry['gmls_v2_quality_scores'],
            'gmls_v2_route_logit': geometry['gmls_v2_route_logit'],
            'gmls_v2_choose_dmls': geometry['gmls_v2_choose_dmls'],

            'fine_translation_enabled': bool(self.unit.fine_translation_enabled),
            'fine_translation_radius': int(self.unit.fine_translation_radius),
            'fine_offset': geometry['fine_offset'],
            'fine_confidence': geometry['fine_correlation_confidence'],
            'refined_offset': geometry['refined_offset'],
            'rgb_target_support': multi_ref[:, 0:1],
            'aligned_tir_target_support': multi_ref[:, 1:2],
            'joint_target_support': multi_ref[:, 2:3],
            'multi_target_guidance': multi_ref,
            'bhlr_guidance_map': multi_ref[:, 2:3],
            'support_by_scale': support_by_scale,
            'guidance_by_scale': guidance_by_scale,
            'center_preserving_scale': True,
            'has_frequency_refinement': False,
            'has_fine_translation': bool(self.unit.fine_translation_enabled),
            'has_scale_candidate_search': False,
        }
        if return_debug:
            info['scale_debug'] = {self.reference_scale: debug}
            info['debug'] = debug
        return aligned, info

    def forward(
        self, rgb_features, tir_features, scale_to_index,
        return_debug=False, rgb_valid_mask=None, tir_valid_mask=None,
    ):
        return self.align_pyramid(
            rgb_features, tir_features, scale_to_index,
            rgb_valid_mask, tir_valid_mask, return_debug,
        )


RLSFAGTGAAlignment = RLSFAGTGACenteredAlignment

__all__ = [
    "RLSFAGTGACenteredAlignment",
    "RLSFAGTGAAlignment",
    "RLSFAGTGAScaleUnit",
    "AdaptiveBoundedScalar",
    "correlation_statistics",
    "softargmax_translation_from_logits",
    "build_direct_sampling_grid",
    "warp_with_geometry",
    "correlation_translation",
    "dominant_mode_local_softargmax",
    "distinct_local_soft_modes",
    "global_multilocal_soft_aggregation",
    "GMLSPathQualityHead",
    "correct_gmls_with_dmls",
]
