#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rich internal visualization for:
  RLSFA coarse UAV anchor + Local-GTGA centered scale
  + alignment-guided BHLR-v2.

The script intentionally exports many diagnostic images. It separates:
  - coarse anchor / coarse translation,
  - raw GTGA targetness / border-safe local targetness,
  - local scale statistics around UAV centers,
  - translation-only vs centered translation+scale,
  - alignment guidance sent to BHLR,
  - BHLR selector/detail injection,
  - final detection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset
from engine.evaluator import postprocess_predictions
from engine.runtime import preprocess_batch, select_device
from models.gtga_bhlr_factory import load_model_from_checkpoint


RGB_GT = (0, 220, 0)
TIR_GT = (0, 165, 255)
ALIGNED_GT = (255, 0, 255)
PRED = (0, 0, 255)
ANCHOR = (255, 255, 0)
SOURCE_CENTER = (255, 120, 255)
TEXT = (245, 245, 245)


def tensor_bgr(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if float(x.max()) > 1.5:
        x = x / 255.0
    x = x.clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x[:3].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def fmap(x: torch.Tensor, mode: str = "mean_abs") -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 2:
        return x.numpy().astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"Unsupported map shape: {tuple(x.shape)}")
    if mode == "mean":
        y = x.mean(0)
    elif mode == "max":
        y = x.amax(0)
    else:
        y = x.abs().mean(0)
    return y.numpy().astype(np.float32)


def robust_range(x: np.ndarray, low=1.0, high=99.0) -> Tuple[float, float]:
    x = np.nan_to_num(np.asarray(x, np.float32))
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def heat(x: np.ndarray, hw: Tuple[int, int], fixed=None, symmetric=False) -> np.ndarray:
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


def overlay(base: np.ndarray, x: np.ndarray, alpha=0.48, fixed=None, symmetric=False) -> np.ndarray:
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
    """High-contrast OpenCV text: dark outline + foreground."""
    cv2.putText(
        image, text, org, cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, outline_color, outline_thickness, cv2.LINE_AA
    )
    cv2.putText(
        image, text, org, cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, cv2.LINE_AA
    )


def add_title(image, title, subtitle="", annotation=""):
    image = image.copy()
    h, w = image.shape[:2]
    lines = 1 + int(bool(subtitle)) + int(bool(annotation))

    # Larger header and larger fonts for paper/zoom inspection.
    bh = min(h - 1, 48 + (lines - 1) * 32)
    cv2.rectangle(image, (0, 0), (w - 1, bh), (14, 14, 14), -1)

    _put_text_outlined(
        image, title[:135], (14, 32),
        font_scale=0.72, color=(255, 255, 255),
        thickness=2, outline_thickness=4,
    )

    y = 64
    for text in (subtitle, annotation):
        if text:
            _put_text_outlined(
                image, text[:165], (14, y),
                font_scale=0.52, color=(225, 225, 225),
                thickness=1, outline_thickness=3,
            )
            y += 30
    return image


def save(out_dir: Path, number: int, slug: str, image: np.ndarray, title: str, subtitle="", annotation="", canvas=960):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.resize(image, (canvas, canvas), interpolation=cv2.INTER_LINEAR)
    img = add_title(img, title, subtitle, annotation)
    path = out_dir / f"{number:02d}_{slug}.jpg"
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"Failed to write {path}")
    return path


def xywhn_to_xyxy(boxes, w, h):
    if torch.is_tensor(boxes):
        boxes = boxes.detach().float().cpu().numpy()
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    if len(b) == 0:
        return np.zeros((0, 4), np.float32)
    cx, cy, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
    out = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h - 1)
    return out


def transform_source_boxes(boxes_xywhn, scale, translation):
    """q=S*p+t -> p=(q-t)/S."""
    if torch.is_tensor(boxes_xywhn):
        b = boxes_xywhn.detach().float().cpu().numpy()
    else:
        b = np.asarray(boxes_xywhn, np.float32)
    b = b.reshape(-1, 4).copy()
    if len(b) == 0:
        return b
    sx, sy = max(float(scale[0]), 1e-6), max(float(scale[1]), 1e-6)
    tx, ty = float(translation[0]), float(translation[1])
    b[:, 0] = (b[:, 0] - tx) / sx
    b[:, 1] = (b[:, 1] - ty) / sy
    b[:, 2] = b[:, 2] / sx
    b[:, 3] = b[:, 3] / sy
    return b


def draw_boxes(image, boxes, color, label, scores=None):
    """Draw slimmer boxes and small outlined labels without any text background box."""
    out = image.copy()
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    h, w = out.shape[:2]

    box_thickness = max(1, int(round(max(h, w) / 420.0)))
    font_scale = max(0.42, min(0.58, max(h, w) / 1500.0))
    text_thickness = 1
    outline_thickness = 2

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))

        # Slim high-contrast box with only a light outline.
        cv2.rectangle(
            out, (x1, y1), (x2, y2), (12, 12, 12),
            box_thickness + 1, cv2.LINE_AA
        )
        cv2.rectangle(
            out, (x1, y1), (x2, y2), color,
            box_thickness, cv2.LINE_AA
        )

        txt = label if len(boxes) == 1 else f"{label}#{i}"
        if scores is not None and i < len(scores):
            txt += f" {float(scores[i]):.2f}"

        (tw, th), baseline = cv2.getTextSize(
            txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )

        # Prefer label above bbox; if not enough room, place just inside the top-left.
        text_x = max(2, min(w - tw - 2, x1))
        if y1 - baseline - 6 >= th:
            text_y = y1 - 5
        else:
            text_y = min(h - baseline - 2, y1 + th + 4)

        _put_text_outlined(
            out, txt, (text_x, text_y),
            font_scale=font_scale, color=color,
            thickness=text_thickness, outline_thickness=outline_thickness,
        )

    return out


def draw_point(image, center_norm, color, label, size=18):
    """Draw a minimal hollow center marker. No cross is drawn inside the UAV box."""
    out = image.copy()
    h, w = out.shape[:2]
    x = int(round(float(center_norm[0]) * (w - 1)))
    y = int(round(float(center_norm[1]) * (h - 1)))

    radius = max(4, int(round(size * 0.28)))
    cv2.circle(out, (x, y), radius + 1, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.circle(out, (x, y), radius, color, 2, cv2.LINE_AA)

    # Label is offset from the center so it does not cover the target itself.
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

    # Put the label near the ellipse edge, not at its center.
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
    inside = float(a[mask > 0].mean()) if np.any(mask > 0) else float('nan')
    outside = float(a[mask == 0].mean()) if np.any(mask == 0) else float('nan')
    ratio = inside / max(outside, 1e-8) if np.isfinite(inside) and np.isfinite(outside) else float('nan')
    return inside, outside, ratio


def stats_text(x: np.ndarray, boxes=None):
    x = np.asarray(x, np.float32)
    parts = [
        f"min/mean/max={x.min():.3f}/{x.mean():.3f}/{x.max():.3f}",
        f"std={x.std():.3f}",
        f"H={spatial_entropy(x):.3f}",
        f"q99={np.quantile(x, .99):.3f}",
    ]
    if boxes is not None:
        i, o, r = inside_outside_stats(x, boxes)
        parts += [f"inside={i:.3f}", f"outside={o:.3f}", f"in/out={r:.2f}"]
    return '; '.join(parts)


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
    base_y = 320
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


def get_any(mapping: Mapping[str, Any], *keys):
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


def make_overview(paths: List[Path], out_path: Path, cols=4, thumb=360):
    imgs = []
    for p in paths:
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(cv2.resize(im, (thumb, thumb), interpolation=cv2.INTER_AREA))
    if not imgs:
        return
    rows = math.ceil(len(imgs) / cols)
    canvas = np.full((rows * thumb, cols * thumb, 3), 240, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r*thumb:(r+1)*thumb, c*thumb:(c+1)*thumb] = im
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])


@torch.inference_mode()
def visualize_one(index, model, dataset, device, output_root, conf, nms_iou, max_det, canvas):
    sample = dataset[index]
    batch = preprocess_batch(dataset.collate_fn([sample]), device)
    outputs = model.predict(
        batch['rgb_img'], batch['tir_img'], rgb_semantic=batch['rgb_semantic_img'],
        return_features=True, return_bhlr_debug=True, return_alignment_debug=True,
    )

    ref_scale = int(model.gtga_reference_scale)
    dbg_map = outputs['alignment_info']['scale_debug']
    a = dbg_map[ref_scale] if ref_scale in dbg_map else dbg_map[str(ref_scale)]

    rgb_high = tensor_bgr(batch['rgb_img'])
    rgb = tensor_bgr(batch['rgb_semantic_img'])
    tir = tensor_bgr(batch['tir_img'])
    h, w = rgb.shape[:2]
    rgb_boxes = xywhn_to_xyxy(sample['rgb_semantic_bboxes'], w, h)
    tir_boxes = xywhn_to_xyxy(sample['tir_bboxes'], w, h)

    anchor = a['coarse_anchor'][0].detach().float().cpu().numpy()
    source_center = a['source_center'][0].detach().float().cpu().numpy()
    scale = a['sampling_scale'][0].detach().float().cpu().numpy()
    scale_raw = a['sampling_scale_raw'][0].detach().float().cpu().numpy()
    trans = a['sampling_translation'][0].detach().float().cpu().numpy()
    offset_norm = a['coarse_offset_normalized'][0].detach().float().cpu().numpy()
    dx = float(a['coarse_offset'][0, 0, 0, 0].detach().cpu())
    dy = float(a['coarse_offset'][0, 1, 0, 0].detach().cpu())
    corr_conf = float(a['coarse_correlation_confidence'][0, 0, 0, 0].detach().cpu())

    # Translation-only q=p+d.
    trans_only_scale = np.array([1.0, 1.0], np.float32)
    trans_only_t = offset_norm.astype(np.float32)
    trans_only_norm = transform_source_boxes(sample['tir_bboxes'], trans_only_scale, trans_only_t)
    trans_only_boxes = xywhn_to_xyxy(trans_only_norm, w, h)

    unified_norm = transform_source_boxes(sample['tir_bboxes'], scale, trans)
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

    stem = Path(sample['rgb_path']).stem
    out_dir = output_root / f"index_{index:06d}_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n = 0

    def emit(slug, image, title, subtitle='', annotation=''):
        nonlocal n
        p = save(out_dir, n, slug, image, title, subtitle, annotation, canvas=canvas)
        paths.append(p)
        n += 1
        return p

    emit('rgb_high_gt', draw_boxes(cv2.resize(rgb_high, (w, h)), rgb_boxes, RGB_GT, 'RGB GT'), 'RGB high-resolution input')
    emit('rgb_semantic_gt', draw_boxes(rgb, rgb_boxes, RGB_GT, 'RGB GT'), 'RGB semantic reference')
    emit('tir_gt', draw_boxes(tir, tir_boxes, TIR_GT, 'TIR GT'), 'Raw TIR input')

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
    emit('scale_pred_vs_gt', render_scale_compare(scale, gt_scale, scale_raw, canvas), 'Centered GTGA scale diagnosis')

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

    # Geometry/pivot diagnostic.
    geom = rgb.copy()
    geom = draw_point(geom, anchor, ANCHOR, 'RGB pivot c_R')
    geom = draw_point(geom, source_center, SOURCE_CENTER, 'TIR source center c_T')
    ax = (int(anchor[0]*(w-1)), int(anchor[1]*(h-1)))
    sxp = (int(source_center[0]*(w-1)), int(source_center[1]*(h-1)))
    cv2.arrowedLine(geom, ax, sxp, (10,10,10), 5, cv2.LINE_AA, tipLength=.08)
    cv2.arrowedLine(geom, ax, sxp, (255,255,255), 3, cv2.LINE_AA, tipLength=.08)
    emit('centered_geometry', geom, 'Center-preserving geometry', 'scale is anchored at c_R/c_T, not image center', f'q=c_T+S(p-c_R); sx={scale[0]:.3f}, sy={scale[1]:.3f}')

    uimg = draw_boxes(unified_bgr, rgb_boxes, RGB_GT, 'RGB GT')
    uimg = draw_boxes(uimg, unified_boxes, ALIGNED_GT, 'aligned TIR GT')
    iou_unified = bbox_iou(rgb_boxes[0], unified_boxes[0]) if len(rgb_boxes)==len(unified_boxes)==1 else None
    emit('unified_centered_alignment', uimg, 'ONE centered warp: translation + Local-GTGA scale', f'sx={scale[0]:.4f}, sy={scale[1]:.4f}', '' if iou_unified is None else f'IoU: before={iou_before:.3f}, translation={iou_trans:.3f}, unified={iou_unified:.3f}')

    blend = cv2.addWeighted(rgb, .5, unified_bgr, .5, 0)
    blend = draw_boxes(blend, rgb_boxes, RGB_GT, 'RGB GT')
    emit('rgb_aligned_tir_blend', blend, 'RGB / aligned-TIR 50:50 blend')

    aligned_target = fmap(a['aligned_tir_local_targetness'], 'mean')
    joint = fmap(a['joint_target_support'], 'mean')
    emit('aligned_tir_local_targetness', draw_boxes(heat(aligned_target,(h,w),fixed=(0,1)), rgb_boxes, RGB_GT, 'RGB GT'), 'Aligned TIR LOCAL targetness', stats_text(aligned_target, sample['rgb_semantic_bboxes']))
    emit('alignment_guidance_raw', draw_boxes(heat(joint,(h,w),fixed=(0,1)), rgb_boxes, RGB_GT, 'RGB GT'), 'Alignment guidance J RAW [0,1]', 'J=sqrt(A_R_local * A_T_local_aligned)', stats_text(joint, sample['rgb_semantic_bboxes']))
    emit('alignment_guidance_overlay', draw_boxes(overlay(rgb,joint,fixed=(0,1)), rgb_boxes, RGB_GT, 'RGB GT'), 'Alignment guidance J over RGB')

    # BHLR transfer verification.
    b_all = outputs['bhlr_debug']
    b_scale = int(model.bhlr_scales[0])
    bdbg = b_all['scales'][b_scale] if b_scale in b_all.get('scales',{}) else b_all['layers'][model.scale_to_index[b_scale]]
    received = bdbg['targetness']
    sent = outputs['alignment_info']['support_by_scale'][b_scale]
    sent = F.interpolate(sent, size=received.shape[-2:], mode='bilinear', align_corners=False)
    transfer = (sent - received).abs()
    mae = float(transfer.mean().detach().cpu())
    maxerr = float(transfer.max().detach().cpu())
    emit('bhlr_received_guidance', draw_boxes(heat(fmap(received,'mean'),(h,w),fixed=(0,1)), rgb_boxes, RGB_GT, 'RGB GT'), 'BHLR actually received guidance', f"source={bdbg.get('guidance_source','unknown')}", f'MAE={mae:.8f}, MAX={maxerr:.8f}')
    emit('guidance_transfer_error', heat(fmap(transfer,'mean'),(h,w),fixed=(0,max(maxerr,1e-6))), '|alignment guidance - BHLR received|', f'MAE={mae:.8f}; near-zero = correctly wired')

    # BHLR details.
    if 'resolution_residual' in b_all:
        emit('highres_residual', overlay(cv2.resize(rgb_high,(w,h)), fmap(b_all['resolution_residual'])), 'High-resolution RGB lost-detail residual')
    if 'detail_bank' in b_all:
        emit('detail_bank', overlay(rgb, fmap(b_all['detail_bank'])), 'BHLR detail bank')

    optional = [
        ('rgb_structure','bhlr_rgb_structure','BHLR RGB structure cue',False),
        ('tir_structure','bhlr_tir_structure','BHLR aligned-TIR structure cue',False),
        ('selector_logit','selector_logit','BHLR selector logit',True),
    ]
    for key, slug, title, signed in optional:
        v = get_any(bdbg, key)
        if v is not None:
            m = fmap(v, 'mean')
            emit(slug, draw_boxes(overlay(rgb,m,symmetric=signed), rgb_boxes, RGB_GT, 'RGB GT'), title)

    selector = fmap(bdbg['selector_multiplier'], 'mean')
    emit('selector_multiplier_raw', draw_boxes(heat(selector,(h,w),fixed=(0,2)), rgb_boxes, RGB_GT, 'RGB GT'), 'BHLR selector multiplier RAW', stats_text(selector, sample['rgb_semantic_bboxes']))
    emit('selector_multiplier_overlay', draw_boxes(overlay(rgb,selector,fixed=(0,2)), rgb_boxes, RGB_GT, 'RGB GT'), 'BHLR selector multiplier over RGB')

    usable = fmap(bdbg['usable_detail'])
    emit('usable_detail_raw', draw_boxes(heat(usable,(h,w)), rgb_boxes, RGB_GT, 'RGB GT'), 'BHLR usable detail RAW')
    emit('usable_detail_overlay', draw_boxes(overlay(rgb,usable), rgb_boxes, RGB_GT, 'RGB GT'), 'BHLR usable detail over RGB')

    layer_id = model.scale_to_index[b_scale]
    raw_feat = outputs['rgb_backbone_raw'][layer_id]
    enhanced = bdbg['enhanced_rgb']
    before_m = fmap(raw_feat)
    after_m = fmap(enhanced)
    lo = min(np.percentile(before_m,1), np.percentile(after_m,1))
    hi = max(np.percentile(before_m,99), np.percentile(after_m,99))
    emit('rgb_p3_before_bhlr', draw_boxes(heat(before_m,(h,w),fixed=(lo,hi)), rgb_boxes, RGB_GT, 'RGB GT'), 'RGB P3 before BHLR', f'shared display [{lo:.4g},{hi:.4g}]')
    emit('rgb_p3_after_bhlr', draw_boxes(heat(after_m,(h,w),fixed=(lo,hi)), rgb_boxes, RGB_GT, 'RGB GT'), 'RGB P3 after BHLR', f'shared display [{lo:.4g},{hi:.4g}]')
    inj = fmap(enhanced - raw_feat)
    emit('actual_bhlr_injection_raw', draw_boxes(heat(inj,(h,w)), rgb_boxes, RGB_GT, 'RGB GT'), 'Actual BHLR injection RAW', 'F_after - F_before')
    emit('actual_bhlr_injection_overlay', draw_boxes(overlay(rgb,inj), rgb_boxes, RGB_GT, 'RGB GT'), 'Actual BHLR injection over RGB')

    # Detection boxes before and after BHLR injection.
    fused_before_bhlr, _ = model.fuse_backbone_features(
        outputs['rgb_backbone_raw'], outputs['aligned_tir_backbone']
    )
    pred_before_bhlr_raw = model.forward_head(fused_before_bhlr)
    pred_before_list = postprocess_predictions(
        preds=pred_before_bhlr_raw, model=model,
        conf_thres=0.001, iou_thres=float(nms_iou), max_det=int(max_det),
    )
    pred_after_list = postprocess_predictions(
        preds=outputs['pred'], model=model,
        conf_thres=0.001, iou_thres=float(nms_iou), max_det=int(max_det),
    )

    before_det_img = draw_boxes(rgb.copy(), rgb_boxes, RGB_GT, 'RGB GT')
    after_det_img = draw_boxes(rgb.copy(), rgb_boxes, RGB_GT, 'RGB GT')
    before_displayed = 0
    after_displayed = 0

    before_boxes_kept = np.zeros((0, 4), np.float32)
    after_boxes_kept = np.zeros((0, 4), np.float32)
    before_scores_kept = np.zeros((0,), np.float32)
    after_scores_kept = np.zeros((0,), np.float32)

    if pred_before_list:
        pb = pred_before_list[0]
        bboxes = pb['bboxes'].detach().float().cpu().numpy()
        scores = pb['conf'].detach().float().cpu().numpy()
        keep = scores >= float(conf)
        before_displayed = int(keep.sum())
        before_boxes_kept = bboxes[keep]
        before_scores_kept = scores[keep]
        before_det_img = draw_boxes(before_det_img, before_boxes_kept, PRED, 'DET', before_scores_kept)

    if pred_after_list:
        pa = pred_after_list[0]
        bboxes = pa['bboxes'].detach().float().cpu().numpy()
        scores = pa['conf'].detach().float().cpu().numpy()
        keep = scores >= float(conf)
        after_displayed = int(keep.sum())
        after_boxes_kept = bboxes[keep]
        after_scores_kept = scores[keep]
        after_det_img = draw_boxes(after_det_img, after_boxes_kept, PRED, 'DET', after_scores_kept)

    emit('prediction_before_bhlr_vs_gt', before_det_img, 'Detection BEFORE BHLR',
         'green=RGB GT; red=prediction',
         f'num_det={before_displayed} @ conf>={float(conf):.2f}')
    emit('prediction_after_bhlr_vs_gt', after_det_img, 'Detection AFTER BHLR',
         'green=RGB GT; red=prediction',
         f'num_det={after_displayed} @ conf>={float(conf):.2f}')

    compare = np.full((h, w * 2 + 12, 3), 245, np.uint8)
    compare[:, :w] = before_det_img
    compare[:, w + 12:] = after_det_img
    cv2.line(compare, (w + 5, 0), (w + 5, h - 1), (90, 90, 90), 2, cv2.LINE_AA)
    _put_text_outlined(compare, 'BEFORE BHLR', (12, 26), 0.62, (255, 255, 255), thickness=1, outline_thickness=2)
    _put_text_outlined(compare, 'AFTER BHLR', (w + 24, 26), 0.62, (255, 255, 255), thickness=1, outline_thickness=2)
    emit('prediction_before_after_bhlr_compare', compare, 'Detection comparison: before vs after BHLR',
         f'before={before_displayed} det, after={after_displayed} det',
         'Same GT, same conf threshold, same NMS settings')

    final = after_det_img
    displayed = after_displayed
    emit('prediction_vs_gt', final, 'Final detection', 'green=RGB GT; red=prediction')

    # Invariance: the geometric pivot must map exactly c_R -> c_T.
    reconstructed_source_center = scale * anchor + trans
    pivot_err = float(np.linalg.norm(reconstructed_source_center - source_center))

    meta = {
        'index': int(index),
        'rgb_path': str(sample['rgb_path']),
        'tir_path': str(sample['tir_path']),
        'coarse_anchor': {
            'pred_rgb_anchor': [float(anchor[0]), float(anchor[1])],
            'source_center': [float(source_center[0]), float(source_center[1])],
            'pivot_invariance_error': pivot_err,
        },
        'coarse_translation': {
            'dx_p3': dx, 'dy_p3': dy, 'confidence': corr_conf,
            'probability': ps,
        },
        'gtga_scale': {
            'pred_sampling_scale': [float(scale[0]), float(scale[1])],
            'raw_local_sigma_ratio': [float(scale_raw[0]), float(scale_raw[1])],
            'bbox_gt_sampling_scale': None if gt_scale is None else [float(gt_scale[0]), float(gt_scale[1])],
            'rgb_sigma': [float(rgb_sigma[0]), float(rgb_sigma[1])],
            'tir_sigma': [float(tir_sigma[0]), float(tir_sigma[1])],
            'local_window_sigma': float(model.alignment.unit.local_window_sigma),
            'moment_border_margin': int(model.alignment.unit.moment_border_margin),
        },
        'alignment_effect': {
            'bbox_iou_before': iou_before,
            'bbox_iou_translation_only': iou_trans,
            'bbox_iou_unified': iou_unified,
            'translation_gain': None if iou_before is None or iou_trans is None else iou_trans - iou_before,
            'scale_additional_gain': None if iou_trans is None or iou_unified is None else iou_unified - iou_trans,
            'total_gain': None if iou_before is None or iou_unified is None else iou_unified - iou_before,
        },
        'targetness': {
            'coarse': stats_text(coarse, sample['rgb_semantic_bboxes']),
            'rgb_raw': stats_text(rgb_raw, sample['rgb_semantic_bboxes']),
            'rgb_local': stats_text(rgb_local, sample['rgb_semantic_bboxes']),
            'tir_raw': stats_text(tir_raw, sample['tir_bboxes']),
            'tir_local': stats_text(tir_local, sample['tir_bboxes']),
        },
        'alignment_guidance_to_bhlr': {
            'enabled': model.bhlr_guidance_mode == 'alignment',
            'formula': 'sqrt(A_RGB_local * A_TIR_local_aligned)',
            'source': str(bdbg.get('guidance_source','unknown')),
            'transfer_mae': mae,
            'transfer_max_abs': maxerr,
            'verified': bool(mae < 1e-5),
        },
        'displayed_predictions': displayed,
        'displayed_predictions_before_bhlr': before_displayed,
        'displayed_predictions_after_bhlr': after_displayed,
    }
    (out_dir/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')

    # Overview of key stages.
    key_slugs = [
        'rgb_semantic_gt','tir_gt','coarse_targetness_raw','coarse_anchor_distribution',
        'tir_gtga_targetness_raw','tir_gtga_targetness_local','tir_local_distribution',
        'coarse_translation_probability','before_alignment','translation_only',
        'centered_geometry','unified_centered_alignment','alignment_guidance_raw',
        'bhlr_received_guidance','selector_multiplier_raw','actual_bhlr_injection_overlay',
        'prediction_before_bhlr_vs_gt','prediction_after_bhlr_vs_gt','prediction_vs_gt'
    ]
    key_paths=[]
    for slug in key_slugs:
        found=list(out_dir.glob(f'*_{slug}.jpg'))
        if found: key_paths.append(found[0])
    make_overview(key_paths, out_dir/'99_key_overview.jpg', cols=4, thumb=360)

    print(f'[OK] index={index} -> {out_dir}')
    print(f'     c_R=({anchor[0]:.4f},{anchor[1]:.4f})  c_T=({source_center[0]:.4f},{source_center[1]:.4f})')
    print(f'     dx/dy=({dx:+.3f},{dy:+.3f}) P3   sx/sy=({scale[0]:.4f},{scale[1]:.4f})')
    print(f'     pivot invariance error={pivot_err:.10f}')
    if iou_before is not None:
        print(f'     IoU {iou_before:.4f} -> {iou_trans:.4f} -> {iou_unified:.4f}')
    print(f'     guidance transfer MAE={mae:.8f} verified={mae < 1e-5}')
    return meta


def parse_indices(values: Optional[Sequence[str]], single: Optional[int]):
    out=[]
    if single is not None: out.append(int(single))
    if values:
        for item in values:
            for tok in str(item).split(','):
                if tok.strip(): out.append(int(tok.strip()))
    return list(dict.fromkeys(out))


def main(args):
    device=select_device(args.device)
    model,_,cfg=load_model_from_checkpoint(args.weights,device=device,fusion_override=args.fusion,strict=True,verbose=False)
    dc=cfg.get('data',{}) or {}
    rgb_yaml=args.rgb or dc.get('rgb'); tir_yaml=args.tir or dc.get('tir')
    if rgb_yaml is None or tir_yaml is None: raise ValueError('RGB/TIR dataset YAML unresolved')
    rgb_imgsz=int(args.rgb_imgsz or dc.get('rgb_imgsz',1920))
    sem_imgsz=int(args.rgb_semantic_imgsz or dc.get('rgb_semantic_imgsz',640))
    tir_imgsz=int(args.tir_imgsz or dc.get('tir_imgsz',640))
    pair_mode=args.pair_mode or dc.get('pair_mode','relative')
    dataset=build_dual_resolution_dataset(
        rgb_yaml=rgb_yaml,tir_yaml=tir_yaml,split=args.split,
        rgb_high_imgsz=rgb_imgsz,rgb_semantic_imgsz=sem_imgsz,tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,augment=False,tir_channels=3,strict_pair=True,
    )
    indices=parse_indices(args.indices,args.index)
    if not indices:
        for i in range(len(dataset)):
            s=dataset[i]
            if s['rgb_semantic_bboxes'].numel() and s['tir_bboxes'].numel():
                indices=[i]; break
    if not indices: raise RuntimeError('No positive sample found')
    weights=Path(args.weights).expanduser().resolve()
    output=Path(args.output).expanduser().resolve() if args.output else weights.parent.parent/'internal_visualization_rlsfa_gtga'
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for idx in indices:
        results.append(visualize_one(idx,model,dataset,device,output,float(args.conf),float(args.iou),int(args.max_det),int(args.canvas)))
    (output/'summary.json').write_text(json.dumps({'weights':str(weights),'indices':indices,'samples':results},indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'\nDone: {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--rgb',default=None); p.add_argument('--tir',default=None)
    p.add_argument('--rgb-imgsz',type=int,default=None); p.add_argument('--rgb-semantic-imgsz',type=int,default=None); p.add_argument('--tir-imgsz',type=int,default=None)
    p.add_argument('--pair-mode',default=None,choices=['relative','stem','filename'])
    p.add_argument('--indices',nargs='+',default=None); p.add_argument('--index',type=int,default=None)
    p.add_argument('--device',default='5'); p.add_argument('--fusion',default='auto',choices=['auto','concat','add','weighted'])
    p.add_argument('--conf',type=float,default=0.25); p.add_argument('--iou',type=float,default=0.70); p.add_argument('--max-det',type=int,default=300)
    p.add_argument('--canvas',type=int,default=1200); p.add_argument('--output',default=None)
    main(p.parse_args())
