#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rich internal visualization for DUCF-Safe v2.

Target pipeline
---------------
    RGB semantic -> RGB backbone -> BHLR ----------------------+
                                                               |
    TIR -> TIR backbone -> RLSFA/Local-GTGA -> aligned TIR ----+
                                                               |
                                                               v
                                                        DUCF-Safe v2
                                                               |
                                                               v
                                                         YOLO neck/head

This script visualizes the COMPLETE pipeline: alignment + BHLR + DUCF and exports:
    1) full RLSFA + Local-GTGA alignment diagnostics
    2) full BHLR high-resolution detail diagnostics
    3) RGB / aligned-TIR features entering DUCF
    2) shared RGB/TIR embeddings
    3) bidirectional cross-modal prediction
    4) RGB/TIR innovation
    5) symmetric consensus
    6) detached alignment support J
    7) independent utility maps q_RGB / q_TIR
    8) innovation aggregation
    9) consensus / innovation residual corrections
   10) protected ADD base, final DUCF feature, actual correction
   11) ADD-baseline detection vs DUCF detection
   12) metadata.json and 99_key_overview.jpg

Important
---------
- This script is inference-only.
- It does not change checkpoint parameters.
- It supports the GRADFIX version where alignment support is detached.
- The ADD comparison keeps alignment + BHLR unchanged and only removes DUCF
  residual correction at DUCF-enabled scales.

Expected model attributes
-------------------------
    model.ducf_fusions
    model.scale_to_index
    model.ducf_scales
    model.forward_head(...)
    outputs['rgb_for_fusion'] or outputs['rgb_backbone']
    outputs['tir_for_fusion'] or outputs['aligned_tir_backbone']
    outputs['fused_backbone']
    outputs['alignment_info']

Example
-------
python visualize_ducf_safe_v2_internal.py \
  --weights runs/.../weights/best.pt \
  --split test \
  --device 0 \
  --indices 0 50 100 150 \
  --scale 3 \
  --conf 0.25
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset
from engine.evaluator import postprocess_predictions
from engine.runtime import preprocess_batch, select_device
from models.gtga_bhlr_factory import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Colors (BGR)
# ---------------------------------------------------------------------------
RGB_GT = (0, 220, 0)
TIR_GT = (0, 165, 255)
ALIGNED_GT = (255, 0, 255)
PRED_ADD = (255, 170, 0)
PRED_DUCF = (0, 0, 255)
ANCHOR = (255, 255, 0)
SOURCE_CENTER = (255, 120, 255)
TEXT = (245, 245, 245)


# ---------------------------------------------------------------------------
# Basic image utilities
# ---------------------------------------------------------------------------
def tensor_bgr(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if float(x.max()) > 1.5:
        x = x / 255.0
    x = x.clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (
        x[:3]
        .permute(1, 2, 0)
        .numpy()
        .clip(0, 1)
        * 255.0
    ).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def fmap(x: torch.Tensor, mode: str = "mean_abs") -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 2:
        return x.numpy().astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"Unsupported feature/map shape: {tuple(x.shape)}")

    if mode == "mean":
        y = x.mean(0)
    elif mode == "max":
        y = x.amax(0)
    elif mode == "l2":
        y = x.square().mean(0).sqrt()
    elif mode == "signed_mean":
        y = x.mean(0)
    else:
        y = x.abs().mean(0)

    return y.numpy().astype(np.float32)


def robust_range(
    x: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
) -> Tuple[float, float]:
    x = np.nan_to_num(np.asarray(x, np.float32))
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def heat(
    x: np.ndarray,
    hw: Tuple[int, int],
    fixed=None,
    symmetric: bool = False,
) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, np.float32))

    if fixed is not None:
        lo, hi = map(float, fixed)
    elif symmetric:
        q = max(float(np.percentile(np.abs(x), 99)), 1e-6)
        lo, hi = -q, q
    else:
        lo, hi = robust_range(x)

    z = np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)
    g = (z * 255).round().astype(np.uint8)
    h, w = map(int, hw)
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(g, cv2.COLORMAP_TURBO)


def overlay(
    base: np.ndarray,
    x: np.ndarray,
    alpha: float = 0.48,
    fixed=None,
    symmetric: bool = False,
) -> np.ndarray:
    hm = heat(x, base.shape[:2], fixed=fixed, symmetric=symmetric)
    return cv2.addWeighted(base, 1 - alpha, hm, alpha, 0)


def _put_text_outlined(
    image,
    text,
    org,
    font_scale,
    color,
    thickness=2,
    outline_color=(0, 0, 0),
    outline_thickness=4,
):
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def add_title(image, title, subtitle="", annotation=""):
    image = image.copy()
    h, w = image.shape[:2]
    lines = 1 + int(bool(subtitle)) + int(bool(annotation))
    bh = min(h - 1, 48 + (lines - 1) * 32)
    cv2.rectangle(image, (0, 0), (w - 1, bh), (14, 14, 14), -1)

    _put_text_outlined(
        image,
        title[:135],
        (14, 32),
        font_scale=0.72,
        color=(255, 255, 255),
        thickness=2,
        outline_thickness=4,
    )

    y = 64
    for text in (subtitle, annotation):
        if text:
            _put_text_outlined(
                image,
                str(text)[:165],
                (14, y),
                font_scale=0.52,
                color=(225, 225, 225),
                thickness=1,
                outline_thickness=3,
            )
            y += 30
    return image


def save(
    out_dir: Path,
    number: int,
    slug: str,
    image: np.ndarray,
    title: str,
    subtitle="",
    annotation="",
    canvas=1200,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.resize(
        image,
        (canvas, canvas),
        interpolation=cv2.INTER_LINEAR,
    )
    img = add_title(
        img,
        title,
        subtitle,
        annotation,
    )
    path = out_dir / f"{number:02d}_{slug}.jpg"
    ok = cv2.imwrite(
        str(path),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 96],
    )
    if not ok:
        raise RuntimeError(f"Failed to write {path}")
    return path


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------
def xywhn_to_xyxy(boxes, w, h):
    if torch.is_tensor(boxes):
        boxes = boxes.detach().float().cpu().numpy()
    b = np.asarray(boxes, np.float32).reshape(-1, 4)

    if len(b) == 0:
        return np.zeros((0, 4), np.float32)

    cx = b[:, 0] * w
    cy = b[:, 1] * h
    bw = b[:, 2] * w
    bh = b[:, 3] * h

    out = np.stack(
        (
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2,
        ),
        axis=1,
    )

    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h - 1)
    return out


def transform_source_boxes(boxes_xywhn, scale, translation):
    """Alignment geometry q=S*p+t -> output p=(q-t)/S."""
    if torch.is_tensor(boxes_xywhn):
        b = boxes_xywhn.detach().float().cpu().numpy()
    else:
        b = np.asarray(boxes_xywhn, np.float32)

    b = b.reshape(-1, 4).copy()
    if len(b) == 0:
        return b

    sx = max(float(scale[0]), 1e-6)
    sy = max(float(scale[1]), 1e-6)
    tx = float(translation[0])
    ty = float(translation[1])

    b[:, 0] = (b[:, 0] - tx) / sx
    b[:, 1] = (b[:, 1] - ty) / sy
    b[:, 2] = b[:, 2] / sx
    b[:, 3] = b[:, 3] / sy
    return b


def draw_boxes(
    image,
    boxes,
    color,
    label,
    scores=None,
):
    out = image.copy()
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    h, w = out.shape[:2]

    box_thickness = max(1, int(round(max(h, w) / 420.0)))
    font_scale = max(0.42, min(0.58, max(h, w) / 1500.0))

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (10, 10, 10),
            box_thickness + 1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            box_thickness,
            cv2.LINE_AA,
        )

        txt = label if len(boxes) == 1 else f"{label}#{i}"
        if scores is not None and i < len(scores):
            txt += f" {float(scores[i]):.2f}"

        (tw, th), baseline = cv2.getTextSize(
            txt,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )

        text_x = max(2, min(w - tw - 2, x1))
        if y1 - baseline - 6 >= th:
            text_y = y1 - 5
        else:
            text_y = min(h - baseline - 2, y1 + th + 4)

        _put_text_outlined(
            out,
            txt,
            (text_x, text_y),
            font_scale=font_scale,
            color=color,
            thickness=1,
            outline_thickness=2,
        )

    return out


# ---------------------------------------------------------------------------
# Alignment visualization helpers (restored from previous script)
# ---------------------------------------------------------------------------
def draw_point(image, center_norm, color, label, size=18):
    """Draw a minimal hollow center marker. No cross is drawn inside the UAV box."""
    out = image.copy()
    h, w = out.shape[:2]
    x = int(round(float(center_norm[0]) * (w - 1)))
    y = int(round(float(center_norm[1]) * (h - 1)))

    radius = max(4, int(round(size * 0.28)))
    cv2.circle(out, (x, y), radius + 1, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.circle(out, (x, y), radius, color, 2, cv2.LINE_AA)

    tx = min(max(4, x + radius + 8), max(4, w - 190))
    ty = max(24, y - radius - 8)
    _put_text_outlined(
        out, label, (tx, ty),
        font_scale=0.46, color=color,
        thickness=1, outline_thickness=2,
    )
    return out


def draw_sigma_ellipse(image, center_norm, sigma_norm, color, label):
    """Draw Gaussian extent without a center cross/marker inside the target."""
    out = image.copy()
    h, w = out.shape[:2]
    cx = int(round(float(center_norm[0]) * (w - 1)))
    cy = int(round(float(center_norm[1]) * (h - 1)))
    axes = (
        max(2, int(round(2 * float(sigma_norm[0]) * w))),
        max(2, int(round(2 * float(sigma_norm[1]) * h))),
    )
    cv2.ellipse(out, (cx, cy), axes, 0, 0, 360, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.ellipse(out, (cx, cy), axes, 0, 0, 360, color, 1, cv2.LINE_AA)

    tx = min(max(4, cx + axes[0] + 8), max(4, w - 210))
    ty = max(24, cy - axes[1] - 8)
    _put_text_outlined(
        out, label, (tx, ty),
        font_scale=0.44, color=color, thickness=1, outline_thickness=2,
    )
    return out


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(aa + bb - inter, 1e-6)


def probability_stats(prob: torch.Tensor):
    p = prob[0].detach().float().cpu().numpy().reshape(-1)
    k = len(p)
    uniform = 1.0 / k
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / math.log(k)
    order = np.argsort(-p)
    return {
        'pmax': float(p[order[0]]),
        'p2': float(p[order[1]]) if k > 1 else float(p[order[0]]),
        'max_over_uniform': float(p[order[0]] / uniform),
        'top1_over_top2': float(p[order[0]] / max(p[order[1]], 1e-12)) if k > 1 else 1.0,
        'entropy_norm': float(ent),
        'effective_candidates': float(math.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
    }


def render_probability(prob, offset, canvas=960):
    p = prob[0].detach().float().cpu().numpy().reshape(-1)
    side = int(round(math.sqrt(len(p))))
    radius = (side - 1) // 2
    grid = p.reshape(side, side)
    img = cv2.applyColorMap((grid / max(float(grid.max()), 1e-12) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img = cv2.resize(img, (canvas, canvas), interpolation=cv2.INTER_NEAREST)
    cell = canvas / side
    best = int(np.argmax(p))
    br, bc = divmod(best, side)
    cv2.rectangle(img, (int(bc * cell), int(br * cell)), (int((bc + 1) * cell) - 1, int((br + 1) * cell) - 1), (0, 0, 0), 5)
    cv2.rectangle(img, (int(bc * cell), int(br * cell)), (int((bc + 1) * cell) - 1, int((br + 1) * cell) - 1), (255, 255, 255), 3)
    dx = float(offset[0, 0, 0, 0].detach().cpu())
    dy = float(offset[0, 1, 0, 0].detach().cpu())
    px = int(round((dx + radius + 0.5) * cell))
    py = int(round((dy + radius + 0.5) * cell))
    cv2.drawMarker(img, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 22, 2)
    return img


def render_top_candidates(prob, topn=12, canvas=960):
    p = prob[0].detach().float().cpu().numpy().reshape(-1)
    side = int(round(math.sqrt(len(p))))
    radius = (side - 1) // 2
    order = np.argsort(-p)[:topn]
    img = np.full((canvas, canvas, 3), 245, np.uint8)
    y = 95
    cv2.putText(img, "Top coarse translation candidates", (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    for rank, idx in enumerate(order, 1):
        row, col = divmod(int(idx), side)
        dx, dy = col - radius, row - radius
        txt = f"#{rank:02d}   dx={dx:+3d}   dy={dy:+3d}   p={p[idx]:.7f}"
        cv2.putText(img, txt, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 1, cv2.LINE_AA)
        y += 60
    return img


def render_scale_compare(pred_scale, gt_scale, raw_scale, canvas=960):
    img = np.full((canvas, canvas, 3), 248, np.uint8)
    cv2.putText(img, "GTGA centered scale: prediction vs bbox GT", (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    labels = ['sx', 'sy']
    maxv = max(2.0, *(list(pred_scale) + list(raw_scale) + (list(gt_scale) if gt_scale is not None else [])))
    for j, lab in enumerate(labels):
        x0 = 100 + j * 410
        cv2.putText(img, lab, (x0 + 110, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)
        vals = [('raw', raw_scale[j], (120, 120, 120)), ('pred', pred_scale[j], (60, 160, 60))]
        if gt_scale is not None:
            vals.append(('GT', gt_scale[j], (40, 80, 220)))
        for k, (name, val, color) in enumerate(vals):
            y = 190 + k * 180
            width = int(min(300, max(0, float(val) / maxv * 300)))
            cv2.rectangle(img, (x0, y), (x0 + width, y + 55), color, -1)
            cv2.putText(img, f"{name}: {float(val):.4f}", (x0, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (35, 35, 35), 1, cv2.LINE_AA)
    if gt_scale is not None:
        ex = (float(pred_scale[0]) - float(gt_scale[0])) / max(float(gt_scale[0]), 1e-8) * 100
        ey = (float(pred_scale[1]) - float(gt_scale[1])) / max(float(gt_scale[1]), 1e-8) * 100
        cv2.putText(img, f"relative error: sx={ex:+.1f}%   sy={ey:+.1f}%", (110, 850), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def spatial_entropy(x: np.ndarray):
    a = np.maximum(np.asarray(x, np.float64), 0)
    p = a.reshape(-1)
    s = p.sum()
    if s <= 1e-12:
        return 1.0
    p = p / s
    h = -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(h / max(math.log(len(p)), 1e-12))


def inside_outside_stats(x: np.ndarray, boxes_xywhn):
    a = np.asarray(x, np.float32)
    h, w = a.shape
    mask = np.zeros((h, w), np.uint8)

    if torch.is_tensor(boxes_xywhn):
        b = boxes_xywhn.detach().float().cpu().numpy().reshape(-1, 4)
    else:
        b = np.asarray(boxes_xywhn, np.float32).reshape(-1, 4)

    for cx, cy, bw, bh in b:
        x1 = max(0, min(w - 1, int(math.floor((cx - bw / 2) * w))))
        y1 = max(0, min(h - 1, int(math.floor((cy - bh / 2) * h))))
        x2 = max(x1 + 1, min(w, int(math.ceil((cx + bw / 2) * w))))
        y2 = max(y1 + 1, min(h, int(math.ceil((cy + bh / 2) * h))))
        mask[y1:y2, x1:x2] = 1

    inside = (
        float(a[mask > 0].mean())
        if np.any(mask > 0)
        else float("nan")
    )
    outside = (
        float(a[mask == 0].mean())
        if np.any(mask == 0)
        else float("nan")
    )
    ratio = (
        inside / max(outside, 1e-8)
        if np.isfinite(inside) and np.isfinite(outside)
        else float("nan")
    )
    return inside, outside, ratio


def stats_text(x: np.ndarray, boxes=None):
    x = np.nan_to_num(np.asarray(x, np.float32))
    parts = [
        f"min/mean/max={x.min():.4f}/{x.mean():.4f}/{x.max():.4f}",
        f"std={x.std():.4f}",
        f"H={spatial_entropy(np.abs(x)):.3f}",
        f"q99={np.quantile(x, .99):.4f}",
    ]

    if boxes is not None:
        i, o, r = inside_outside_stats(x, boxes)
        parts += [
            f"inside={i:.4f}",
            f"outside={o:.4f}",
            f"in/out={r:.2f}",
        ]

    return "; ".join(parts)


def tensor_scalar(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().float().mean().cpu().item())
    return float(x)


def get_any(mapping: Mapping[str, Any], *keys):
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


# ---------------------------------------------------------------------------
# DUCF extraction
# ---------------------------------------------------------------------------
def _model_predict_with_debug(
    model,
    batch,
):
    kwargs = dict(
        rgb_semantic=batch["rgb_semantic_img"],
        return_features=True,
        return_bhlr_debug=True,
        return_alignment_debug=True,
    )

    sig = inspect.signature(model.predict)
    if "return_ducf_debug" in sig.parameters:
        kwargs["return_ducf_debug"] = True

    return model.predict(
        batch["rgb_img"],
        batch["tir_img"],
        **kwargs,
    )


@torch.inference_mode()
def reconstruct_ducf_internal(
    module,
    rgb_feature: torch.Tensor,
    tir_feature: torch.Tensor,
    support: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Recompute all DUCF-Safe internal tensors for visualization.

    This mirrors the current DUCFSafeFusion forward and does not modify state.
    """
    tir_feature = module._resize_like(tir_feature, rgb_feature)

    rgb32 = rgb_feature.float()
    tir32 = tir_feature.float()
    base = rgb32 + tir32

    r = module.shared_embed(rgb32)
    t = module.shared_embed(tir32)

    pred_r_from_t = module.shared_predictor(t)
    pred_t_from_r = module.shared_predictor(r)

    innovation_r = r - pred_r_from_t
    innovation_t = t - pred_t_from_r

    consensus_avg = 0.5 * (r + t)
    consensus_absdiff = (r - t).abs()
    consensus_product = r * t

    consensus = module.consensus_encoder(
        torch.cat(
            [
                consensus_avg,
                consensus_absdiff,
                consensus_product,
            ],
            dim=1,
        )
    )

    support_map = module._prepare_support(
        support,
        r,
    )

    q_rgb, logit_rgb = module.shared_utility(
        r,
        t,
        innovation_r,
        consensus,
        support_map,
    )
    q_tir, logit_tir = module.shared_utility(
        t,
        r,
        innovation_t,
        consensus,
        support_map,
    )

    enc_r = module.shared_innovation_encoder(innovation_r)
    enc_t = module.shared_innovation_encoder(innovation_t)
    innovation_mix = q_rgb * enc_r + q_tir * enc_t

    consensus_delta_raw = module.consensus_to_out(consensus)
    innovation_delta_raw = module.innovation_to_out(innovation_mix)

    consensus_delta = module._rms_match_to_base(
        consensus_delta_raw,
        base,
    )
    innovation_delta = module._rms_match_to_base(
        innovation_delta_raw,
        base,
    )

    g_consensus = (
        module.max_consensus_ratio
        * torch.tanh(module.consensus_gain_raw.float())
    )
    g_innovation = (
        module.max_innovation_ratio
        * torch.tanh(module.innovation_gain_raw.float())
    )

    correction_consensus = g_consensus * consensus_delta.float()
    correction_innovation = g_innovation * innovation_delta.float()
    correction = correction_consensus + correction_innovation
    fused = base + correction

    return {
        "rgb_input": rgb32,
        "tir_input": tir32,
        "base": base,
        "rgb_embedding": r,
        "tir_embedding": t,
        "pred_rgb_from_tir": pred_r_from_t,
        "pred_tir_from_rgb": pred_t_from_r,
        "innovation_rgb": innovation_r,
        "innovation_tir": innovation_t,
        "innovation_rgb_energy": innovation_r.square().mean(1, keepdim=True),
        "innovation_tir_energy": innovation_t.square().mean(1, keepdim=True),
        "consensus_avg": consensus_avg,
        "consensus_absdiff": consensus_absdiff,
        "consensus_product": consensus_product,
        "consensus": consensus,
        "support": support_map,
        "q_rgb": q_rgb,
        "q_tir": q_tir,
        "utility_logit_rgb": logit_rgb,
        "utility_logit_tir": logit_tir,
        "utility_difference": q_rgb - q_tir,
        "utility_sum": q_rgb + q_tir,
        "innovation_encoded_rgb": enc_r,
        "innovation_encoded_tir": enc_t,
        "innovation_mix": innovation_mix,
        "consensus_delta_raw": consensus_delta_raw,
        "innovation_delta_raw": innovation_delta_raw,
        "consensus_delta": consensus_delta,
        "innovation_delta": innovation_delta,
        "consensus_gain": g_consensus,
        "innovation_gain": g_innovation,
        "correction_consensus": correction_consensus,
        "correction_innovation": correction_innovation,
        "correction": correction,
        "fused": fused,
    }


def resolve_scale_layer(model, requested_scale: Optional[int]):
    if not hasattr(model, "ducf_fusions"):
        raise AttributeError(
            "Loaded checkpoint model has no model.ducf_fusions. "
            "This visualizer is only for DUCF checkpoints."
        )

    if len(model.ducf_fusions) == 0:
        raise RuntimeError("model.ducf_fusions is empty.")

    configured_scales = [
        int(s)
        for s in getattr(model, "ducf_scales", [])
    ]

    if not configured_scales:
        scale_to_index = dict(model.scale_to_index)
        layer_keys = {int(k) for k in model.ducf_fusions.keys()}
        configured_scales = [
            int(s)
            for s, idx in scale_to_index.items()
            if int(idx) in layer_keys
        ]

    configured_scales = sorted(set(configured_scales))
    if not configured_scales:
        raise RuntimeError(
            "Could not infer conceptual DUCF scales from model."
        )

    scale = (
        int(requested_scale)
        if requested_scale is not None
        else configured_scales[0]
    )

    if scale not in configured_scales:
        raise ValueError(
            f"Requested P{scale}, but checkpoint DUCF scales are "
            f"{configured_scales}."
        )

    layer_id = int(model.scale_to_index[scale])
    key = str(layer_id)

    if key not in model.ducf_fusions:
        raise KeyError(
            f"DUCF module missing for P{scale}/layer{layer_id}; "
            f"keys={list(model.ducf_fusions.keys())}"
        )

    return scale, layer_id, model.ducf_fusions[key], configured_scales


def get_fusion_inputs(outputs, layer_id):
    rgb_all = outputs.get(
        "rgb_for_fusion",
        outputs.get("rgb_backbone"),
    )
    tir_all = outputs.get(
        "tir_for_fusion",
        outputs.get("aligned_tir_backbone"),
    )

    if rgb_all is None:
        raise KeyError(
            "Missing outputs['rgb_for_fusion'] / outputs['rgb_backbone']."
        )
    if tir_all is None:
        raise KeyError(
            "Missing outputs['tir_for_fusion'] / outputs['aligned_tir_backbone']."
        )

    return rgb_all[layer_id], tir_all[layer_id]


def get_support(outputs, scale):
    info = outputs.get("alignment_info", {}) or {}
    support_by_scale = info.get("support_by_scale", {}) or {}

    if scale in support_by_scale:
        return support_by_scale[scale]
    if str(scale) in support_by_scale:
        return support_by_scale[str(scale)]

    return info.get(
        "joint_target_support",
        info.get("bhlr_guidance_map", None),
    )


@torch.inference_mode()
def fuse_pyramid_with_current_ducf(model, rgb_pyramid, tir_pyramid, alignment_info):
    """Fuse RGB/TIR pyramids using the *actual* current policy.

    IMPORTANT:
    The DUCF wrapper may not keep ``model.fusions`` entries for every parent
    fusion layer.  Therefore this visualization helper MUST NOT call
    ``model.fuse_backbone_features()``.

    Current policy:
      1) every configured P3/P4/P5 fusion layer starts from protected ADD
             F_base = F_RGB + F_TIR_aligned
      2) DUCF-enabled scales (currently P3) overwrite that ADD feature with
             F_DUCF = F_base + residual correction

    This is exactly the comparison needed for BHLR OFF/ON while keeping the
    same alignment and DUCF policy.
    """
    fused = list(rgb_pyramid)

    # --------------------------------------------------------------
    # 1. Protected ADD baseline for every configured fusion layer.
    #    Do this manually; do NOT depend on model.fusions[str(index)].
    # --------------------------------------------------------------
    for idx in getattr(model, "fusion_indices", []):
        idx = int(idx)
        rgb_s = rgb_pyramid[idx]
        tir_s = tir_pyramid[idx]

        if tir_s.shape[-2:] != rgb_s.shape[-2:]:
            tir_s = F.interpolate(
                tir_s,
                size=rgb_s.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if tir_s.shape[1] != rgb_s.shape[1]:
            raise RuntimeError(
                f"Manual ADD fusion channel mismatch at layer {idx}: "
                f"RGB={tuple(rgb_s.shape)}, TIR={tuple(tir_s.shape)}. "
                "The current DUCF experiment expects channel-matched ADD."
            )

        fused[idx] = rgb_s + tir_s

    # No DUCF modules -> the protected ADD pyramid is already final.
    if not hasattr(model, "ducf_fusions") or len(model.ducf_fusions) == 0:
        return fused

    # --------------------------------------------------------------
    # 2. Overwrite DUCF-enabled conceptual scales, e.g. P3.
    # --------------------------------------------------------------
    configured_scales = [
        int(s) for s in getattr(model, "ducf_scales", [])
    ]
    if not configured_scales:
        layer_keys = {int(k) for k in model.ducf_fusions.keys()}
        configured_scales = [
            int(s)
            for s, idx in model.scale_to_index.items()
            if int(idx) in layer_keys
        ]

    support_by_scale = (alignment_info or {}).get(
        "support_by_scale", {}
    ) or {}

    for s in configured_scales:
        idx = int(model.scale_to_index[s])
        key = str(idx)
        if key not in model.ducf_fusions:
            continue

        rgb_s = rgb_pyramid[idx]
        tir_s = tir_pyramid[idx]
        support = support_by_scale.get(
            s, support_by_scale.get(str(s), None)
        )

        dbg = reconstruct_ducf_internal(
            model.ducf_fusions[key],
            rgb_s,
            tir_s,
            support,
        )
        fused[idx] = dbg["fused"].to(dtype=rgb_s.dtype)

    return fused


# ---------------------------------------------------------------------------
# Overview / side-by-side helpers
# ---------------------------------------------------------------------------
def make_overview(
    paths: List[Path],
    out_path: Path,
    cols=4,
    thumb=360,
):
    imgs = []
    for p in paths:
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(
                    cv2.resize(
                        im,
                        (thumb, thumb),
                        interpolation=cv2.INTER_AREA,
                    )
                )

    if not imgs:
        return

    rows = math.ceil(len(imgs) / cols)
    canvas = np.full(
        (rows * thumb, cols * thumb, 3),
        240,
        np.uint8,
    )

    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[
            r * thumb:(r + 1) * thumb,
            c * thumb:(c + 1) * thumb,
        ] = im

    cv2.imwrite(
        str(out_path),
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )


def make_two_panel(
    left: np.ndarray,
    right: np.ndarray,
    left_title: str,
    right_title: str,
):
    h = max(left.shape[0], right.shape[0])
    w = max(left.shape[1], right.shape[1])

    left = cv2.resize(left, (w, h), interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(right, (w, h), interpolation=cv2.INTER_LINEAR)

    gap = 12
    canvas = np.full(
        (h, 2 * w + gap, 3),
        245,
        np.uint8,
    )
    canvas[:, :w] = left
    canvas[:, w + gap:] = right

    cv2.line(
        canvas,
        (w + gap // 2, 0),
        (w + gap // 2, h - 1),
        (90, 90, 90),
        2,
        cv2.LINE_AA,
    )

    _put_text_outlined(
        canvas,
        left_title,
        (12, 30),
        0.68,
        (255, 255, 255),
        thickness=2,
        outline_thickness=3,
    )
    _put_text_outlined(
        canvas,
        right_title,
        (w + gap + 12, 30),
        0.68,
        (255, 255, 255),
        thickness=2,
        outline_thickness=3,
    )
    return canvas


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def draw_prediction(
    rgb: np.ndarray,
    gt_boxes: np.ndarray,
    pred_list,
    conf: float,
    color,
    label,
):
    out = draw_boxes(
        rgb.copy(),
        gt_boxes,
        RGB_GT,
        "RGB GT",
    )

    n = 0
    kept_boxes = np.zeros((0, 4), np.float32)
    kept_scores = np.zeros((0,), np.float32)

    if pred_list:
        pred = pred_list[0]
        boxes = pred["bboxes"].detach().float().cpu().numpy()
        scores = pred["conf"].detach().float().cpu().numpy()
        keep = scores >= float(conf)
        kept_boxes = boxes[keep]
        kept_scores = scores[keep]
        n = int(keep.sum())

        out = draw_boxes(
            out,
            kept_boxes,
            color,
            label,
            kept_scores,
        )

    return out, n, kept_boxes, kept_scores


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------
@torch.inference_mode()
def visualize_one(
    index,
    model,
    dataset,
    device,
    output_root,
    conf,
    nms_iou,
    max_det,
    canvas,
    requested_scale,
):
    sample = dataset[index]
    batch = preprocess_batch(
        dataset.collate_fn([sample]),
        device,
    )

    outputs = _model_predict_with_debug(
        model,
        batch,
    )

    scale, layer_id, module, all_scales = resolve_scale_layer(
        model,
        requested_scale,
    )

    rgb_feature, tir_feature = get_fusion_inputs(
        outputs,
        layer_id,
    )
    support = get_support(
        outputs,
        scale,
    )

    d = reconstruct_ducf_internal(
        module,
        rgb_feature,
        tir_feature,
        support,
    )

    rgb_high = tensor_bgr(batch["rgb_img"])
    rgb = tensor_bgr(batch["rgb_semantic_img"])
    tir = tensor_bgr(batch["tir_img"])
    h, w = rgb.shape[:2]

    rgb_boxes = xywhn_to_xyxy(
        sample["rgb_semantic_bboxes"],
        w,
        h,
    )
    tir_boxes = xywhn_to_xyxy(
        sample["tir_bboxes"],
        w,
        h,
    )

    # Optional aligned TIR image in RGB frame.
    aligned_tir_bgr = None
    aligned_tir_boxes = None
    alignment_info = outputs.get("alignment_info", {}) or {}

    geometry_scale = alignment_info.get("sampling_scale", None)
    geometry_translation = alignment_info.get("sampling_translation", None)

    if (
        geometry_scale is not None
        and geometry_translation is not None
        and hasattr(model, "alignment")
        and hasattr(model.alignment, "warp_tensor")
    ):
        try:
            aligned_tir_img = model.alignment.warp_tensor(
                batch["tir_img"],
                {
                    "sampling_scale": geometry_scale,
                    "sampling_translation": geometry_translation,
                },
                output_hw=batch["rgb_semantic_img"].shape[-2:],
            )
            aligned_tir_bgr = tensor_bgr(aligned_tir_img)

            scale_np = (
                geometry_scale[0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            trans_np = (
                geometry_translation[0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            aligned_tir_norm = transform_source_boxes(
                sample["tir_bboxes"],
                scale_np,
                trans_np,
            )
            aligned_tir_boxes = xywhn_to_xyxy(
                aligned_tir_norm,
                w,
                h,
            )
        except Exception:
            aligned_tir_bgr = None
            aligned_tir_boxes = None

    stem = Path(sample["rgb_path"]).stem
    out_dir = (
        output_root
        / f"index_{index:06d}_{stem}_P{scale}"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = []
    n = 0

    def emit(
        slug,
        image,
        title,
        subtitle="",
        annotation="",
    ):
        nonlocal n
        p = save(
            out_dir,
            n,
            slug,
            image,
            title,
            subtitle,
            annotation,
            canvas=canvas,
        )
        paths.append(p)
        n += 1
        return p

    # ------------------------------------------------------------------
    # A. Restored alignment-module diagnostics (from previous visualizer)
    # ------------------------------------------------------------------
    scale_debug_all = (outputs.get("alignment_info", {}) or {}).get("scale_debug", {}) or {}
    ref_scale = int(getattr(model, "gtga_reference_scale", scale))
    a = scale_debug_all.get(ref_scale, scale_debug_all.get(str(ref_scale), None))

    emit(
        "rgb_high_gt",
        draw_boxes(
            cv2.resize(rgb_high, (w, h)),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "RGB high-resolution input",
    )

    emit(
        "rgb_semantic_gt",
        draw_boxes(
            rgb,
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "RGB semantic input / DUCF output frame",
    )

    emit(
        "tir_raw_gt",
        draw_boxes(
            tir,
            tir_boxes,
            TIR_GT,
            "TIR GT",
        ),
        "Raw TIR input",
    )

    if a is not None:
        anchor = a['coarse_anchor'][0].detach().float().cpu().numpy()
        source_center = a['source_center'][0].detach().float().cpu().numpy()
        gtga_scale = a['sampling_scale'][0].detach().float().cpu().numpy()
        scale_raw = a['sampling_scale_raw'][0].detach().float().cpu().numpy()
        trans = a['sampling_translation'][0].detach().float().cpu().numpy()
        offset_norm = a['coarse_offset_normalized'][0].detach().float().cpu().numpy()
        dx = float(a['coarse_offset'][0, 0, 0, 0].detach().cpu())
        dy = float(a['coarse_offset'][0, 1, 0, 0].detach().cpu())
        corr_conf = float(a['coarse_correlation_confidence'][0, 0, 0, 0].detach().cpu())

        trans_only_scale = np.array([1.0, 1.0], np.float32)
        trans_only_t = offset_norm.astype(np.float32)
        trans_only_norm = transform_source_boxes(sample['tir_bboxes'], trans_only_scale, trans_only_t)
        trans_only_boxes = xywhn_to_xyxy(trans_only_norm, w, h)

        unified_norm = transform_source_boxes(sample['tir_bboxes'], gtga_scale, trans)
        unified_boxes = xywhn_to_xyxy(unified_norm, w, h)

        trans_only_img = model.alignment.warp_tensor(
            batch['tir_img'],
            {'sampling_scale': torch.ones_like(a['sampling_scale']), 'sampling_translation': a['coarse_offset_normalized']},
            output_hw=batch['rgb_semantic_img'].shape[-2:],
        )
        unified_img = model.alignment.warp_tensor(
            batch['tir_img'],
            {'sampling_scale': a['sampling_scale'], 'sampling_translation': a['sampling_translation']},
            output_hw=batch['rgb_semantic_img'].shape[-2:],
        )
        trans_only_bgr = tensor_bgr(trans_only_img)
        unified_bgr = tensor_bgr(unified_img)

        coarse = fmap(a['coarse_targetness'], 'mean')
        cad = fmap(a['coarse_anchor_distribution'], 'mean')
        c_raw = draw_point(heat(coarse, (h, w), fixed=(0, 1)), anchor, ANCHOR, 'c_R')
        c_raw = draw_boxes(c_raw, rgb_boxes, RGB_GT, 'RGB GT')
        emit('coarse_targetness_raw', c_raw, 'RLSFA coarse targetness RAW [0,1]', stats_text(coarse, sample['rgb_semantic_bboxes']))
        cdist = draw_point(heat(cad, (h, w)), anchor, ANCHOR, 'coarse UAV anchor c_R')
        cdist = draw_boxes(cdist, rgb_boxes, RGB_GT, 'RGB GT')
        emit('coarse_anchor_distribution', cdist, 'Coarse anchor distribution', f'anchor=({anchor[0]:.4f},{anchor[1]:.4f})')

        rgb_raw = fmap(a['rgb_targetness_raw'], 'mean')
        tir_raw = fmap(a['tir_targetness_raw'], 'mean')
        rgb_safe = fmap(a['rgb_safe_mask'], 'mean')
        tir_safe = fmap(a['tir_safe_mask'], 'mean')
        rgb_local = fmap(a['rgb_local_targetness'], 'mean')
        tir_local = fmap(a['tir_local_targetness'], 'mean')
        rgb_window = fmap(a['rgb_local_window'], 'mean')
        tir_window = fmap(a['tir_local_window'], 'mean')
        rgb_dist = fmap(a['rgb_distribution'], 'mean')
        tir_dist = fmap(a['tir_distribution'], 'mean')

        emit('rgb_gtga_targetness_raw', draw_boxes(heat(rgb_raw, (h, w), fixed=(0,1)), rgb_boxes, RGB_GT, 'RGB GT'), 'RGB GTGA targetness RAW [0,1]', stats_text(rgb_raw, sample['rgb_semantic_bboxes']))
        emit('rgb_gtga_targetness_local', draw_boxes(heat(rgb_local, (h, w), fixed=(0,1)), rgb_boxes, RGB_GT, 'RGB GT'), 'RGB LOCAL GTGA targetness', 'RAW x local window x safe mask', stats_text(rgb_local, sample['rgb_semantic_bboxes']))
        emit('tir_gtga_targetness_raw', draw_boxes(heat(tir_raw, (h, w), fixed=(0,1)), tir_boxes, TIR_GT, 'TIR GT'), 'TIR GTGA targetness RAW [0,1]', stats_text(tir_raw, sample['tir_bboxes']))
        emit('tir_gtga_targetness_local', draw_boxes(heat(tir_local, (h, w), fixed=(0,1)), tir_boxes, TIR_GT, 'TIR GT'), 'TIR LOCAL GTGA targetness', 'RAW x local window x safe mask', stats_text(tir_local, sample['tir_bboxes']))

        emit('rgb_safe_mask', heat(rgb_safe, (h,w), fixed=(0,1)), 'RGB moment safe mask', 'LetterBox + outer feature border removed')
        emit('tir_safe_mask', heat(tir_safe, (h,w), fixed=(0,1)), 'TIR moment safe mask', 'Suppresses left/right boundary artifacts from scale statistics')
        emit('rgb_local_window', draw_point(heat(rgb_window, (h,w), fixed=(0,1)), anchor, ANCHOR, 'c_R'), 'RGB local scale window', f'sigma_window={model.alignment.unit.local_window_sigma:.3f}')
        emit('tir_local_window', draw_point(heat(tir_window, (h,w), fixed=(0,1)), source_center, SOURCE_CENTER, 'c_T'), 'TIR local scale window', f'center=({source_center[0]:.4f},{source_center[1]:.4f})')

        rgb_sigma = a['rgb_sigma'][0].detach().float().cpu().numpy()
        tir_sigma = a['tir_sigma'][0].detach().float().cpu().numpy()
        rd = draw_sigma_ellipse(heat(rgb_dist, (h,w)), anchor, rgb_sigma, ANCHOR, 'c_R / sigma_R')
        rd = draw_boxes(rd, rgb_boxes, RGB_GT, 'RGB GT')
        emit('rgb_local_distribution', rd, 'RGB local GTGA distribution', f'sigma_R=({rgb_sigma[0]:.4f},{rgb_sigma[1]:.4f})')
        td = draw_sigma_ellipse(heat(tir_dist, (h,w)), source_center, tir_sigma, SOURCE_CENTER, 'c_T / sigma_T')
        td = draw_boxes(td, tir_boxes, TIR_GT, 'TIR GT')
        emit('tir_local_distribution', td, 'TIR local GTGA distribution', f'sigma_T=({tir_sigma[0]:.4f},{tir_sigma[1]:.4f})')

        gt_scale = None
        if sample['rgb_semantic_bboxes'].shape[0] == 1 and sample['tir_bboxes'].shape[0] == 1:
            rb = sample['rgb_semantic_bboxes'][0].detach().float().cpu().numpy()
            tb = sample['tir_bboxes'][0].detach().float().cpu().numpy()
            gt_scale = np.array([tb[2]/max(rb[2],1e-8), tb[3]/max(rb[3],1e-8)], np.float32)
        emit('scale_pred_vs_gt', render_scale_compare(gtga_scale, gt_scale, scale_raw, canvas), 'Centered GTGA scale diagnosis')

        ps = probability_stats(a['coarse_probability'])
        emit('coarse_translation_probability', render_probability(a['coarse_probability'], a['coarse_offset'], canvas), '41x41 coarse translation probability', f'dx={dx:+.3f}, dy={dy:+.3f}, confidence={corr_conf:.6f}', f"pmax/uniform={ps['max_over_uniform']:.2f}; H={ps['entropy_norm']:.4f}; effK={ps['effective_candidates']:.1f}")
        emit('coarse_top_candidates', render_top_candidates(a['coarse_probability'], 12, canvas), 'Top coarse translation candidates')

        before = draw_boxes(rgb.copy(), rgb_boxes, RGB_GT, 'RGB GT')
        before = draw_boxes(before, tir_boxes, TIR_GT, 'raw TIR GT')
        iou_before = bbox_iou(rgb_boxes[0], tir_boxes[0]) if len(rgb_boxes)==len(tir_boxes)==1 else None
        emit('before_alignment', before, 'Before alignment', 'green=RGB GT; orange=raw TIR GT', '' if iou_before is None else f'IoU={iou_before:.4f}')

        timg = draw_boxes(trans_only_bgr, rgb_boxes, RGB_GT, 'RGB GT')
        timg = draw_boxes(timg, trans_only_boxes, ALIGNED_GT, 'TIR after translation')
        iou_trans = bbox_iou(rgb_boxes[0], trans_only_boxes[0]) if len(rgb_boxes)==len(trans_only_boxes)==1 else None
        emit('translation_only', timg, 'RLSFA coarse translation ONLY', f'dx={dx:+.3f}, dy={dy:+.3f} P3 px', '' if iou_trans is None else f'IoU={iou_trans:.4f}')

        geom = rgb.copy()
        geom = draw_point(geom, anchor, ANCHOR, 'RGB pivot c_R')
        geom = draw_point(geom, source_center, SOURCE_CENTER, 'TIR source center c_T')
        ax = (int(anchor[0]*(w-1)), int(anchor[1]*(h-1)))
        sxp = (int(source_center[0]*(w-1)), int(source_center[1]*(h-1)))
        cv2.arrowedLine(geom, ax, sxp, (10,10,10), 5, cv2.LINE_AA, tipLength=.08)
        cv2.arrowedLine(geom, ax, sxp, (255,255,255), 3, cv2.LINE_AA, tipLength=.08)
        emit('centered_geometry', geom, 'Center-preserving geometry', 'scale is anchored at c_R/c_T, not image center', f'q=c_T+S(p-c_R); sx={gtga_scale[0]:.3f}, sy={gtga_scale[1]:.3f}')

        uimg = draw_boxes(unified_bgr, rgb_boxes, RGB_GT, 'RGB GT')
        uimg = draw_boxes(uimg, unified_boxes, ALIGNED_GT, 'aligned TIR GT')
        iou_unified = bbox_iou(rgb_boxes[0], unified_boxes[0]) if len(rgb_boxes)==len(unified_boxes)==1 else None
        emit('unified_centered_alignment', uimg, 'ONE centered warp: translation + Local-GTGA scale', f'sx={gtga_scale[0]:.4f}, sy={gtga_scale[1]:.4f}', '' if iou_unified is None else f'IoU: before={iou_before:.3f}, translation={iou_trans:.3f}, unified={iou_unified:.3f}')

        blend = cv2.addWeighted(rgb, .5, unified_bgr, .5, 0)
        blend = draw_boxes(blend, rgb_boxes, RGB_GT, 'RGB GT')
        emit('rgb_aligned_tir_blend', blend, 'RGB / aligned-TIR 50:50 blend')

        emit('aligned_tir_rgb_grid', draw_boxes(draw_boxes(unified_bgr.copy(), rgb_boxes, RGB_GT, 'RGB GT'), unified_boxes, ALIGNED_GT, 'aligned TIR GT'), 'Aligned TIR image on RGB grid', 'Final TIR -> RGB geometry consumed by downstream fusion')

        aligned_target = fmap(a['aligned_tir_local_targetness'], 'mean')
        joint = fmap(a['joint_target_support'], 'mean')
        emit('aligned_tir_local_targetness', draw_boxes(heat(aligned_target,(h,w),fixed=(0,1)), rgb_boxes, RGB_GT, 'RGB GT'), 'Aligned TIR LOCAL targetness', stats_text(aligned_target, sample['rgb_semantic_bboxes']))
        emit('alignment_guidance_raw', draw_boxes(heat(joint,(h,w),fixed=(0,1)), rgb_boxes, RGB_GT, 'RGB GT'), 'Alignment guidance J RAW [0,1]', 'J=sqrt(A_R_local * A_T_local_aligned)', stats_text(joint, sample['rgb_semantic_bboxes']))
        emit('alignment_guidance_overlay', draw_boxes(overlay(rgb,joint,fixed=(0,1)), rgb_boxes, RGB_GT, 'RGB GT'), 'Alignment guidance J over RGB')

    # ------------------------------------------------------------------
    # B. High-resolution BHLR-v2 diagnostics (restored in full)
    # ------------------------------------------------------------------
    bhlr_meta = {}
    b_all = outputs.get("bhlr_debug", None)

    if b_all is not None and getattr(model, "bhlr_enabled", False):
        b_scales = [int(s) for s in getattr(model, "bhlr_scales", [])]
        if b_scales:
            b_scale = int(b_scales[0])
            b_layer_id = int(model.scale_to_index[b_scale])

            scales_dict = b_all.get("scales", {}) or {}
            layers_dict = b_all.get("layers", {}) or {}
            bdbg = scales_dict.get(
                b_scale,
                scales_dict.get(
                    str(b_scale),
                    layers_dict.get(
                        b_layer_id,
                        layers_dict.get(str(b_layer_id), None),
                    ),
                ),
            )

            if bdbg is not None:
                # ------------------------------------------------------
                # Verify alignment guidance really reaches BHLR.
                # ------------------------------------------------------
                received = get_any(
                    bdbg,
                    "targetness",
                    "joint_target_support",
                )
                support_by_scale = (
                    outputs.get("alignment_info", {}) or {}
                ).get("support_by_scale", {}) or {}
                sent = support_by_scale.get(
                    b_scale,
                    support_by_scale.get(str(b_scale), None),
                )

                transfer_mae = None
                transfer_max = None

                if received is not None and sent is not None:
                    sent_cmp = F.interpolate(
                        sent,
                        size=received.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    transfer = (sent_cmp.detach() - received.detach()).abs()
                    transfer_mae = float(transfer.mean().cpu())
                    transfer_max = float(transfer.max().cpu())

                    emit(
                        "bhlr_received_guidance",
                        draw_boxes(
                            heat(fmap(received, "mean"), (h, w), fixed=(0, 1)),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "BHLR actually received alignment guidance",
                        f"source={bdbg.get('guidance_source', 'unknown')}",
                        f"MAE={transfer_mae:.8f}, MAX={transfer_max:.8f}",
                    )

                    emit(
                        "guidance_transfer_error",
                        heat(
                            fmap(transfer, "mean"),
                            (h, w),
                            fixed=(0, max(transfer_max, 1e-6)),
                        ),
                        "|alignment guidance - BHLR received|",
                        f"MAE={transfer_mae:.8f}; near-zero = correctly wired",
                    )

                # ------------------------------------------------------
                # High-resolution lost-detail extraction.
                # ------------------------------------------------------
                resolution_residual = get_any(
                    b_all,
                    "resolution_residual",
                    "residual_high",
                )
                if resolution_residual is not None:
                    rr = fmap(resolution_residual, "mean_abs")
                    emit(
                        "highres_residual_raw",
                        draw_boxes(
                            heat(rr, (h, w)),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "High-resolution RGB lost-detail residual RAW",
                        "Difference between high-resolution RGB and reconstructed semantic RGB",
                        stats_text(rr, sample["rgb_semantic_bboxes"]),
                    )
                    emit(
                        "highres_residual_overlay",
                        draw_boxes(
                            overlay(cv2.resize(rgb_high, (w, h)), rr),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "High-resolution RGB lost-detail residual over RGB",
                    )

                detail_bank = get_any(
                    b_all,
                    "detail_bank",
                    "detail_raw",
                )
                if detail_bank is not None:
                    db = fmap(detail_bank, "mean_abs")
                    emit(
                        "detail_bank_raw",
                        draw_boxes(
                            heat(db, (h, w)),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "BHLR high-resolution detail bank RAW",
                        stats_text(db, sample["rgb_semantic_bboxes"]),
                    )
                    emit(
                        "detail_bank_overlay",
                        draw_boxes(
                            overlay(rgb, db),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "BHLR high-resolution detail bank over RGB",
                    )

                # ------------------------------------------------------
                # Guidance / selector internals.
                # ------------------------------------------------------
                optional = [
                    ("rgb_structure", "bhlr_rgb_structure", "BHLR RGB structure cue", False),
                    ("tir_structure", "bhlr_tir_structure", "BHLR aligned-TIR structure cue", False),
                    ("selector_logit", "bhlr_selector_logit", "BHLR selector logit", True),
                ]

                for key, slug, title, signed in optional:
                    v = get_any(bdbg, key)
                    if v is not None:
                        m = fmap(v, "mean")
                        emit(
                            slug,
                            draw_boxes(
                                overlay(rgb, m, symmetric=signed),
                                rgb_boxes,
                                RGB_GT,
                                "RGB GT",
                            ),
                            title,
                            stats_text(m, sample["rgb_semantic_bboxes"]),
                        )

                selector_tensor = get_any(
                    bdbg,
                    "selector_multiplier",
                    "multiplier",
                )
                if selector_tensor is not None:
                    selector = fmap(selector_tensor, "mean")
                    emit(
                        "selector_multiplier_raw",
                        draw_boxes(
                            heat(selector, (h, w), fixed=(0, 2)),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "BHLR selector multiplier RAW",
                        "1 = neutral; >1 amplify detail; <1 suppress detail",
                        stats_text(selector, sample["rgb_semantic_bboxes"]),
                    )
                    emit(
                        "selector_multiplier_overlay",
                        draw_boxes(
                            overlay(rgb, selector, fixed=(0, 2)),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "BHLR selector multiplier over RGB",
                    )

                # ------------------------------------------------------
                # Detail actually selected for injection.
                # ------------------------------------------------------
                usable_tensor = get_any(
                    bdbg,
                    "usable_detail",
                    "selected_detail",
                )
                if usable_tensor is not None:
                    usable = fmap(usable_tensor, "mean_abs")
                    emit(
                        "usable_detail_raw",
                        draw_boxes(
                            heat(usable, (h, w)),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "BHLR usable high-resolution detail RAW",
                        stats_text(usable, sample["rgb_semantic_bboxes"]),
                    )
                    emit(
                        "usable_detail_overlay",
                        draw_boxes(
                            overlay(rgb, usable),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "BHLR usable high-resolution detail over RGB",
                    )

                # ------------------------------------------------------
                # Exact feature injection: before / after / difference.
                # ------------------------------------------------------
                rgb_raw_pyramid = outputs.get("rgb_backbone_raw", None)
                enhanced = get_any(bdbg, "enhanced_rgb")

                if rgb_raw_pyramid is not None and enhanced is not None:
                    raw_feat = rgb_raw_pyramid[b_layer_id]
                    before_m = fmap(raw_feat, "mean_abs")
                    after_m = fmap(enhanced, "mean_abs")

                    lo = min(
                        float(np.percentile(before_m, 1)),
                        float(np.percentile(after_m, 1)),
                    )
                    hi = max(
                        float(np.percentile(before_m, 99)),
                        float(np.percentile(after_m, 99)),
                    )

                    emit(
                        f"rgb_p{b_scale}_before_bhlr",
                        draw_boxes(
                            heat(before_m, (h, w), fixed=(lo, hi)),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        f"RGB P{b_scale} before BHLR",
                        f"shared display [{lo:.4g}, {hi:.4g}]",
                    )

                    emit(
                        f"rgb_p{b_scale}_after_bhlr",
                        draw_boxes(
                            heat(after_m, (h, w), fixed=(lo, hi)),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        f"RGB P{b_scale} after BHLR",
                        f"shared display [{lo:.4g}, {hi:.4g}]",
                    )

                    injection_tensor = enhanced - raw_feat
                    inj_abs = fmap(injection_tensor, "mean_abs")
                    inj_signed = fmap(injection_tensor, "signed_mean")

                    emit(
                        "actual_bhlr_injection_raw",
                        draw_boxes(
                            heat(inj_abs, (h, w)),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "Actual BHLR injection magnitude",
                        "|F_after - F_before|",
                        stats_text(inj_abs, sample["rgb_semantic_bboxes"]),
                    )

                    emit(
                        "actual_bhlr_injection_signed",
                        draw_boxes(
                            overlay(rgb, inj_signed, symmetric=True),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "Actual BHLR injection signed map",
                        "channel mean of F_after - F_before",
                    )

                    emit(
                        "actual_bhlr_injection_overlay",
                        draw_boxes(
                            overlay(rgb, inj_abs),
                            rgb_boxes,
                            RGB_GT,
                            "RGB GT",
                        ),
                        "Actual BHLR high-resolution injection over RGB",
                    )

                    gamma = get_any(bdbg, "gamma", "gamma_value")
                    if gamma is None and hasattr(model, "bhlr"):
                        try:
                            adapter = model.bhlr.adapters[str(b_layer_id)]
                            if hasattr(adapter, "gamma_raw"):
                                gamma = torch.tanh(adapter.gamma_raw)
                        except Exception:
                            gamma = None

                    base_rms_b = float(raw_feat.float().square().mean().sqrt().cpu())
                    inj_rms_b = float(injection_tensor.float().square().mean().sqrt().cpu())

                    bhlr_meta.update({
                        "scale": int(b_scale),
                        "layer_id": int(b_layer_id),
                        "gamma": None if gamma is None else tensor_scalar(gamma),
                        "feature_before_rms": base_rms_b,
                        "injection_rms": inj_rms_b,
                        "injection_to_feature_rms": inj_rms_b / max(base_rms_b, 1e-12),
                        "guidance_transfer_mae": transfer_mae,
                        "guidance_transfer_max": transfer_max,
                    })

                # ------------------------------------------------------
                # Detection comparison with BHLR OFF vs ON while keeping
                # alignment and DUCF policy the same.
                # ------------------------------------------------------
                rgb_raw_pyramid = outputs.get("rgb_backbone_raw", None)
                aligned_tir_pyramid = outputs.get("aligned_tir_backbone", None)
                fused_after_bhlr = outputs.get("fused_backbone", None)

                if (
                    rgb_raw_pyramid is not None
                    and aligned_tir_pyramid is not None
                    and fused_after_bhlr is not None
                ):
                    fused_before_bhlr = fuse_pyramid_with_current_ducf(
                        model,
                        rgb_raw_pyramid,
                        aligned_tir_pyramid,
                        outputs.get("alignment_info", {}),
                    )

                    pred_before_bhlr_raw = model.forward_head(fused_before_bhlr)
                    pred_after_bhlr_raw = outputs["pred"]

                    pred_before_list = postprocess_predictions(
                        preds=pred_before_bhlr_raw,
                        model=model,
                        conf_thres=0.001,
                        iou_thres=float(nms_iou),
                        max_det=int(max_det),
                    )
                    pred_after_list = postprocess_predictions(
                        preds=pred_after_bhlr_raw,
                        model=model,
                        conf_thres=0.001,
                        iou_thres=float(nms_iou),
                        max_det=int(max_det),
                    )

                    before_det_img, before_displayed, _, _ = draw_prediction(
                        rgb,
                        rgb_boxes,
                        pred_before_list,
                        conf,
                        PRED_ADD,
                        "w/o BHLR",
                    )
                    after_det_img, after_displayed, _, _ = draw_prediction(
                        rgb,
                        rgb_boxes,
                        pred_after_list,
                        conf,
                        PRED_DUCF,
                        "BHLR",
                    )

                    emit(
                        "prediction_before_bhlr_vs_gt",
                        before_det_img,
                        "Detection BEFORE BHLR",
                        "alignment + DUCF kept; high-resolution BHLR removed",
                        f"num_det={before_displayed} @ conf>={float(conf):.2f}",
                    )
                    emit(
                        "prediction_after_bhlr_vs_gt",
                        after_det_img,
                        "Detection AFTER BHLR",
                        "alignment + BHLR + DUCF",
                        f"num_det={after_displayed} @ conf>={float(conf):.2f}",
                    )

                    bcomp = make_two_panel(
                        before_det_img,
                        after_det_img,
                        "WITHOUT BHLR",
                        "WITH BHLR",
                    )
                    emit(
                        "prediction_before_after_bhlr_compare",
                        bcomp,
                        "Detection comparison: without vs with BHLR",
                        f"without={before_displayed} det; with={after_displayed} det",
                        "Alignment and DUCF policy are kept the same",
                    )

    # ------------------------------------------------------------------
    # 2. DUCF input features
    # ------------------------------------------------------------------
    rgb_in = fmap(d["rgb_input"], "mean_abs")
    tir_in = fmap(d["tir_input"], "mean_abs")
    base_map = fmap(d["base"], "mean_abs")

    emit(
        "ducf_rgb_input_feature",
        draw_boxes(
            overlay(rgb, rgb_in),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        f"DUCF P{scale}: RGB feature input",
        stats_text(rgb_in, sample["rgb_semantic_bboxes"]),
    )

    emit(
        "ducf_tir_input_feature",
        draw_boxes(
            overlay(rgb, tir_in),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        f"DUCF P{scale}: aligned-TIR feature input",
        stats_text(tir_in, sample["rgb_semantic_bboxes"]),
    )

    emit(
        "add_base_feature",
        draw_boxes(
            overlay(rgb, base_map),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Protected ADD base feature",
        "F_base = F_RGB + F_TIR",
        stats_text(base_map, sample["rgb_semantic_bboxes"]),
    )

    # ------------------------------------------------------------------
    # 3. Shared embedding + prediction
    # ------------------------------------------------------------------
    r_embed = fmap(d["rgb_embedding"], "mean_abs")
    t_embed = fmap(d["tir_embedding"], "mean_abs")

    emit(
        "shared_rgb_embedding",
        draw_boxes(
            overlay(rgb, r_embed),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Shared embedding R = phi(F_RGB)",
        stats_text(r_embed, sample["rgb_semantic_bboxes"]),
    )

    emit(
        "shared_tir_embedding",
        draw_boxes(
            overlay(rgb, t_embed),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Shared embedding T = phi(F_TIR)",
        stats_text(t_embed, sample["rgb_semantic_bboxes"]),
    )

    pred_r = fmap(d["pred_rgb_from_tir"], "mean_abs")
    pred_t = fmap(d["pred_tir_from_rgb"], "mean_abs")

    emit(
        "pred_rgb_from_tir",
        draw_boxes(
            overlay(rgb, pred_r),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Cross-modal prediction: R_hat = P(T)",
        "RGB-representation component predictable from TIR",
    )

    emit(
        "pred_tir_from_rgb",
        draw_boxes(
            overlay(rgb, pred_t),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Cross-modal prediction: T_hat = P(R)",
        "TIR-representation component predictable from RGB",
    )

    # ------------------------------------------------------------------
    # 4. Bidirectional innovations
    # ------------------------------------------------------------------
    i_rgb_signed = fmap(
        d["innovation_rgb"],
        "signed_mean",
    )
    i_tir_signed = fmap(
        d["innovation_tir"],
        "signed_mean",
    )
    e_rgb = fmap(
        d["innovation_rgb_energy"],
        "mean",
    )
    e_tir = fmap(
        d["innovation_tir_energy"],
        "mean",
    )

    emit(
        "innovation_rgb_signed",
        draw_boxes(
            overlay(
                rgb,
                i_rgb_signed,
                symmetric=True,
            ),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "RGB innovation I_R = R - P(T)",
        "signed channel-mean map",
        stats_text(i_rgb_signed, sample["rgb_semantic_bboxes"]),
    )

    emit(
        "innovation_tir_signed",
        draw_boxes(
            overlay(
                rgb,
                i_tir_signed,
                symmetric=True,
            ),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "TIR innovation I_T = T - P(R)",
        "signed channel-mean map",
        stats_text(i_tir_signed, sample["rgb_semantic_bboxes"]),
    )

    emit(
        "innovation_rgb_energy",
        draw_boxes(
            heat(e_rgb, (h, w)),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "RGB innovation energy",
        stats_text(e_rgb, sample["rgb_semantic_bboxes"]),
    )

    emit(
        "innovation_tir_energy",
        draw_boxes(
            heat(e_tir, (h, w)),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "TIR innovation energy",
        stats_text(e_tir, sample["rgb_semantic_bboxes"]),
    )

    # ------------------------------------------------------------------
    # 5. Consensus
    # ------------------------------------------------------------------
    c_avg = fmap(d["consensus_avg"], "mean_abs")
    c_diff = fmap(d["consensus_absdiff"], "mean_abs")
    c_prod = fmap(d["consensus_product"], "mean_abs")
    consensus_map = fmap(d["consensus"], "mean_abs")

    emit(
        "consensus_input_average",
        draw_boxes(
            overlay(rgb, c_avg),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Consensus input: 0.5(R + T)",
    )

    emit(
        "consensus_input_absdiff",
        draw_boxes(
            overlay(rgb, c_diff),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Consensus input: |R - T|",
    )

    emit(
        "consensus_input_product",
        draw_boxes(
            overlay(rgb, c_prod),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Consensus input: R x T",
    )

    emit(
        "shared_consensus",
        draw_boxes(
            overlay(rgb, consensus_map),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Shared symmetric consensus C",
        stats_text(
            consensus_map,
            sample["rgb_semantic_bboxes"],
        ),
    )

    # ------------------------------------------------------------------
    # 6. Alignment support + utility gates
    # ------------------------------------------------------------------
    support_map = fmap(d["support"], "mean")
    q_rgb = fmap(d["q_rgb"], "mean")
    q_tir = fmap(d["q_tir"], "mean")
    q_diff = fmap(d["utility_difference"], "mean")
    q_sum = fmap(d["utility_sum"], "mean")

    emit(
        "detached_alignment_support",
        draw_boxes(
            heat(
                support_map,
                (h, w),
                fixed=(0, 1),
            ),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Detached alignment support J",
        "context only; no DUCF -> alignment backprop",
        stats_text(
            support_map,
            sample["rgb_semantic_bboxes"],
        ),
    )

    emit(
        "q_rgb_raw",
        draw_boxes(
            heat(
                q_rgb,
                (h, w),
                fixed=(0, 1),
            ),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "DUCF RGB utility q_RGB",
        "independent sigmoid; not softmax competition",
        stats_text(
            q_rgb,
            sample["rgb_semantic_bboxes"],
        ),
    )

    emit(
        "q_tir_raw",
        draw_boxes(
            heat(
                q_tir,
                (h, w),
                fixed=(0, 1),
            ),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "DUCF TIR utility q_TIR",
        "independent sigmoid; not softmax competition",
        stats_text(
            q_tir,
            sample["rgb_semantic_bboxes"],
        ),
    )

    emit(
        "q_rgb_minus_q_tir",
        draw_boxes(
            heat(
                q_diff,
                (h, w),
                fixed=(-1, 1),
            ),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Utility difference q_RGB - q_TIR",
        "warm/cool map is only a visualization of relative preference",
        stats_text(
            q_diff,
            sample["rgb_semantic_bboxes"],
        ),
    )

    emit(
        "q_rgb_plus_q_tir",
        draw_boxes(
            heat(
                q_sum,
                (h, w),
                fixed=(0, 2),
            ),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Joint utility q_RGB + q_TIR",
        "values can exceed 1 because gates are independent",
        stats_text(
            q_sum,
            sample["rgb_semantic_bboxes"],
        ),
    )

    utility_pair = make_two_panel(
        draw_boxes(
            overlay(
                rgb,
                q_rgb,
                fixed=(0, 1),
            ),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        draw_boxes(
            overlay(
                rgb,
                q_tir,
                fixed=(0, 1),
            ),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "q_RGB",
        "q_TIR",
    )

    emit(
        "utility_rgb_tir_compare",
        utility_pair,
        "DUCF utility comparison",
        f"mean q_RGB={q_rgb.mean():.4f}; mean q_TIR={q_tir.mean():.4f}",
    )

    # ------------------------------------------------------------------
    # 7. Innovation aggregation + residual corrections
    # ------------------------------------------------------------------
    innovation_mix = fmap(
        d["innovation_mix"],
        "mean_abs",
    )
    consensus_delta = fmap(
        d["consensus_delta"],
        "mean_abs",
    )
    innovation_delta = fmap(
        d["innovation_delta"],
        "mean_abs",
    )

    corr_c = fmap(
        d["correction_consensus"],
        "mean_abs",
    )
    corr_i = fmap(
        d["correction_innovation"],
        "mean_abs",
    )
    correction = fmap(
        d["correction"],
        "mean_abs",
    )
    fused = fmap(
        d["fused"],
        "mean_abs",
    )

    emit(
        "innovation_mix",
        draw_boxes(
            overlay(rgb, innovation_mix),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Utility-weighted innovation mix",
        "q_RGB * psi(I_R) + q_TIR * psi(I_T)",
        stats_text(
            innovation_mix,
            sample["rgb_semantic_bboxes"],
        ),
    )

    emit(
        "consensus_delta_normalized",
        draw_boxes(
            overlay(rgb, consensus_delta),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Normalized consensus residual Delta_C",
    )

    emit(
        "innovation_delta_normalized",
        draw_boxes(
            overlay(rgb, innovation_delta),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Normalized innovation residual Delta_I",
    )

    g_c = tensor_scalar(d["consensus_gain"])
    g_i = tensor_scalar(d["innovation_gain"])

    emit(
        "actual_consensus_correction",
        draw_boxes(
            overlay(rgb, corr_c),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Actual consensus correction g_C * Delta_C",
        f"g_C={g_c:.8f}",
        stats_text(
            corr_c,
            sample["rgb_semantic_bboxes"],
        ),
    )

    emit(
        "actual_innovation_correction",
        draw_boxes(
            overlay(rgb, corr_i),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Actual innovation correction g_I * Delta_I",
        f"g_I={g_i:.8f}",
        stats_text(
            corr_i,
            sample["rgb_semantic_bboxes"],
        ),
    )

    base_rms = float(
        d["base"].float().square().mean().sqrt().cpu().item()
    )
    corr_rms = float(
        d["correction"].float().square().mean().sqrt().cpu().item()
    )
    corr_ratio = corr_rms / max(base_rms, 1e-12)

    emit(
        "actual_ducf_correction",
        draw_boxes(
            overlay(rgb, correction),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Actual DUCF correction",
        "F_DUCF - F_ADD",
        f"RMS correction/base={corr_ratio:.8f}",
    )

    emit(
        "final_ducf_feature",
        draw_boxes(
            overlay(rgb, fused),
            rgb_boxes,
            RGB_GT,
            "RGB GT",
        ),
        "Final DUCF fused feature",
        "F_out = F_ADD + correction",
        stats_text(
            fused,
            sample["rgb_semantic_bboxes"],
        ),
    )

    # ------------------------------------------------------------------
    # 8. Detector comparison: ADD baseline vs DUCF
    # ------------------------------------------------------------------
    fused_add = list(outputs["fused_backbone"])

    rgb_all = outputs.get(
        "rgb_for_fusion",
        outputs.get("rgb_backbone"),
    )
    tir_all = outputs.get(
        "tir_for_fusion",
        outputs.get("aligned_tir_backbone"),
    )

    if rgb_all is None or tir_all is None:
        raise KeyError(
            "Cannot construct ADD comparison: missing fusion input pyramids."
        )

    # Remove DUCF residual correction at every DUCF-enabled scale.
    for s in all_scales:
        idx = int(model.scale_to_index[s])
        tir_s = tir_all[idx]
        rgb_s = rgb_all[idx]

        if tir_s.shape[-2:] != rgb_s.shape[-2:]:
            tir_s = F.interpolate(
                tir_s,
                size=rgb_s.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused_add[idx] = rgb_s + tir_s

    pred_add_raw = model.forward_head(fused_add)
    pred_final_raw = outputs["pred"]

    pred_add_list = postprocess_predictions(
        preds=pred_add_raw,
        model=model,
        conf_thres=0.001,
        iou_thres=float(nms_iou),
        max_det=int(max_det),
    )
    pred_final_list = postprocess_predictions(
        preds=pred_final_raw,
        model=model,
        conf_thres=0.001,
        iou_thres=float(nms_iou),
        max_det=int(max_det),
    )

    add_img, add_n, add_boxes, add_scores = draw_prediction(
        rgb,
        rgb_boxes,
        pred_add_list,
        conf,
        PRED_ADD,
        "ADD",
    )

    final_img, final_n, final_boxes, final_scores = draw_prediction(
        rgb,
        rgb_boxes,
        pred_final_list,
        conf,
        PRED_DUCF,
        "DUCF",
    )

    emit(
        "prediction_add_baseline",
        add_img,
        "Detection with protected ADD baseline",
        "alignment + BHLR kept; DUCF correction removed",
        f"num_det={add_n} @ conf>={float(conf):.2f}",
    )

    emit(
        "prediction_ducf_final",
        final_img,
        "Detection with DUCF-Safe",
        "same alignment, BHLR, neck, head",
        f"num_det={final_n} @ conf>={float(conf):.2f}",
    )

    compare = make_two_panel(
        add_img,
        final_img,
        "ADD baseline",
        "DUCF-Safe",
    )

    emit(
        "prediction_add_vs_ducf",
        compare,
        "Detection comparison: ADD vs DUCF",
        f"ADD={add_n} det; DUCF={final_n} det",
        "Only the DUCF residual fusion differs",
    )

    # ------------------------------------------------------------------
    # 9. Metadata
    # ------------------------------------------------------------------
    qrgb_i, qrgb_o, qrgb_ratio = inside_outside_stats(
        q_rgb,
        sample["rgb_semantic_bboxes"],
    )
    qtir_i, qtir_o, qtir_ratio = inside_outside_stats(
        q_tir,
        sample["rgb_semantic_bboxes"],
    )

    e_rgb_i, e_rgb_o, e_rgb_ratio = inside_outside_stats(
        e_rgb,
        sample["rgb_semantic_bboxes"],
    )
    e_tir_i, e_tir_o, e_tir_ratio = inside_outside_stats(
        e_tir,
        sample["rgb_semantic_bboxes"],
    )

    meta = {
        "index": int(index),
        "rgb_path": str(sample["rgb_path"]),
        "tir_path": str(sample["tir_path"]),
        "ducf_scale": int(scale),
        "ducf_layer_id": int(layer_id),
        "configured_ducf_scales": [
            int(s)
            for s in all_scales
        ],
        "geometry_direction": "TIR -> RGB",
        "fusion_frame": "RGB",
        "alignment_support": {
            "used": support is not None,
            "detached_context_expected": True,
            "mean": float(support_map.mean()),
            "inside_gt": float(
                inside_outside_stats(
                    support_map,
                    sample["rgb_semantic_bboxes"],
                )[0]
            ),
        },
        "utility": {
            "q_rgb_mean": float(q_rgb.mean()),
            "q_tir_mean": float(q_tir.mean()),
            "q_rgb_std": float(q_rgb.std()),
            "q_tir_std": float(q_tir.std()),
            "q_rgb_inside_gt": qrgb_i,
            "q_rgb_outside_gt": qrgb_o,
            "q_rgb_inside_outside_ratio": qrgb_ratio,
            "q_tir_inside_gt": qtir_i,
            "q_tir_outside_gt": qtir_o,
            "q_tir_inside_outside_ratio": qtir_ratio,
            "mean_difference_rgb_minus_tir": float(q_diff.mean()),
            "mean_sum": float(q_sum.mean()),
        },
        "innovation": {
            "rgb_energy_mean": float(e_rgb.mean()),
            "tir_energy_mean": float(e_tir.mean()),
            "rgb_energy_inside_gt": e_rgb_i,
            "rgb_energy_outside_gt": e_rgb_o,
            "rgb_energy_inside_outside_ratio": e_rgb_ratio,
            "tir_energy_inside_gt": e_tir_i,
            "tir_energy_outside_gt": e_tir_o,
            "tir_energy_inside_outside_ratio": e_tir_ratio,
        },
        "bhlr": bhlr_meta,
        "safe_residual": {
            "consensus_gain": g_c,
            "innovation_gain": g_i,
            "base_rms": base_rms,
            "correction_rms": corr_rms,
            "correction_to_base_rms": corr_ratio,
            "max_consensus_ratio": float(module.max_consensus_ratio),
            "max_innovation_ratio": float(module.max_innovation_ratio),
        },
        "detection": {
            "confidence_display_threshold": float(conf),
            "add_num_det": int(add_n),
            "ducf_num_det": int(final_n),
            "add_scores": [
                float(v)
                for v in add_scores.tolist()
            ],
            "ducf_scores": [
                float(v)
                for v in final_scores.tolist()
            ],
        },
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(
            meta,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 10. Key overview
    # ------------------------------------------------------------------
    key_slugs = [
        "rgb_high_gt",
        "rgb_semantic_gt",
        "tir_raw_gt",
        "coarse_targetness_raw",
        "coarse_anchor_distribution",
        "coarse_translation_probability",
        "before_alignment",
        "translation_only",
        "centered_geometry",
        "unified_centered_alignment",
        "rgb_aligned_tir_blend",
        "alignment_guidance_raw",
        "aligned_tir_rgb_grid",
        "highres_residual_overlay",
        "detail_bank_overlay",
        "bhlr_received_guidance",
        "selector_multiplier_raw",
        "usable_detail_overlay",
        "rgb_p3_before_bhlr",
        "rgb_p3_after_bhlr",
        "actual_bhlr_injection_overlay",
        "prediction_before_after_bhlr_compare",
        "ducf_rgb_input_feature",
        "ducf_tir_input_feature",
        "innovation_rgb_energy",
        "innovation_tir_energy",
        "shared_consensus",
        "detached_alignment_support",
        "q_rgb_raw",
        "q_tir_raw",
        "q_rgb_minus_q_tir",
        "innovation_mix",
        "actual_consensus_correction",
        "actual_innovation_correction",
        "actual_ducf_correction",
        "final_ducf_feature",
        "prediction_add_baseline",
        "prediction_ducf_final",
        "prediction_add_vs_ducf",
    ]

    key_paths = []
    for slug in key_slugs:
        found = list(
            out_dir.glob(
                f"*_{slug}.jpg"
            )
        )
        if found:
            key_paths.append(found[0])

    make_overview(
        key_paths,
        out_dir / "99_key_overview.jpg",
        cols=4,
        thumb=360,
    )

    print(
        f"[OK] index={index} P{scale} -> {out_dir}"
    )
    print(
        f"     qRGB mean={q_rgb.mean():.4f} | "
        f"qTIR mean={q_tir.mean():.4f}"
    )
    print(
        f"     gains: gC={g_c:.8f}, gI={g_i:.8f}"
    )
    print(
        f"     correction/base RMS={corr_ratio:.8f}"
    )
    print(
        f"     detections: ADD={add_n}, DUCF={final_n}"
    )

    return meta


def parse_indices(
    values: Optional[Sequence[str]],
    single: Optional[int],
):
    out = []

    if single is not None:
        out.append(int(single))

    if values:
        for item in values:
            for tok in str(item).split(","):
                if tok.strip():
                    out.append(int(tok.strip()))

    return list(dict.fromkeys(out))


def main(args):
    device = select_device(args.device)

    model, _, cfg = load_model_from_checkpoint(
        args.weights,
        device=device,
        fusion_override=args.fusion,
        strict=True,
        verbose=False,
    )

    if not hasattr(model, "ducf_fusions"):
        raise RuntimeError(
            "This checkpoint is not a DUCF model: model.ducf_fusions missing."
        )

    dc = cfg.get("data", {}) or {}

    rgb_yaml = args.rgb or dc.get("rgb")
    tir_yaml = args.tir or dc.get("tir")

    if rgb_yaml is None or tir_yaml is None:
        raise ValueError(
            "RGB/TIR dataset YAML unresolved."
        )

    rgb_imgsz = int(
        args.rgb_imgsz
        or dc.get("rgb_imgsz", 1920)
    )
    sem_imgsz = int(
        args.rgb_semantic_imgsz
        or dc.get("rgb_semantic_imgsz", 640)
    )
    tir_imgsz = int(
        args.tir_imgsz
        or dc.get("tir_imgsz", 640)
    )
    pair_mode = (
        args.pair_mode
        or dc.get("pair_mode", "relative")
    )

    dataset = build_dual_resolution_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=args.split,
        rgb_high_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=sem_imgsz,
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )

    indices = parse_indices(
        args.indices,
        args.index,
    )

    if not indices:
        for i in range(len(dataset)):
            s = dataset[i]
            if (
                s["rgb_semantic_bboxes"].numel()
                and s["tir_bboxes"].numel()
            ):
                indices = [i]
                break

    if not indices:
        raise RuntimeError(
            "No positive sample found."
        )

    weights = (
        Path(args.weights)
        .expanduser()
        .resolve()
    )

    output = (
        Path(args.output)
        .expanduser()
        .resolve()
        if args.output
        else weights.parent.parent
        / "internal_visualization_ducf_safe_v2"
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            print(
                f"[SKIP] index={idx} out of range "
                f"[0,{len(dataset)-1}]"
            )
            continue

        results.append(
            visualize_one(
                index=idx,
                model=model,
                dataset=dataset,
                device=device,
                output_root=output,
                conf=float(args.conf),
                nms_iou=float(args.iou),
                max_det=int(args.max_det),
                canvas=int(args.canvas),
                requested_scale=args.scale,
            )
        )

    (output / "summary.json").write_text(
        json.dumps(
            {
                "weights": str(weights),
                "indices": indices,
                "requested_scale": args.scale,
                "samples": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nDone: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Internal DUCF-Safe v2 visualization."
    )

    p.add_argument(
        "--weights",
        required=True,
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )

    p.add_argument("--rgb", default=None)
    p.add_argument("--tir", default=None)

    p.add_argument(
        "--rgb-imgsz",
        type=int,
        default=None,
    )
    p.add_argument(
        "--rgb-semantic-imgsz",
        type=int,
        default=None,
    )
    p.add_argument(
        "--tir-imgsz",
        type=int,
        default=None,
    )

    p.add_argument(
        "--pair-mode",
        default=None,
        choices=[
            "relative",
            "stem",
            "filename",
        ],
    )

    p.add_argument(
        "--indices",
        nargs="+",
        default=None,
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
    )

    p.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Conceptual DUCF scale to visualize, e.g. 3 for P3.",
    )

    p.add_argument(
        "--device",
        default="0",
    )
    p.add_argument(
        "--fusion",
        default="auto",
        choices=[
            "auto",
            "concat",
            "add",
            "weighted",
        ],
    )

    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.70,
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=300,
    )

    p.add_argument(
        "--canvas",
        type=int,
        default=1200,
    )
    p.add_argument(
        "--output",
        default=None,
    )

    main(p.parse_args())
