#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMLS-Align v1 dedicated evaluator
=================================

Evaluation chain:
    Raw RGB/TIR GT
      -> Historical GLOBAL soft coarse translation
      -> Single-mode DMLS local-soft coarse translation
      -> GMLS global + multi-local soft aggregated coarse translation
      -> GMLS coarse + original Fine +/-2 translation
      -> GMLS + Fine + original Local-GTGA Moment scale
      -> FINAL one-warp geometry

This evaluator intentionally removes diagnostics from the abandoned robust-v1:
    - no Top-K hard verification
    - no Fine confidence gate
    - no Quantile scale
    - no scale reliability fallback

Key questions answered
----------------------
1) Does single-mode DMLS improve historical Global Soft?
2) Does GMLS improve further over DMLS?
3) Does GMLS reduce DMLS catastrophic zero-IoU failures?
4) Is GMLS closer to GT translation than Global Soft and DMLS?
5) Does the original Fine stage help after GMLS?
6) Does the original Moment scale help or hurt after translation?

Geometry convention
-------------------
The model uses inverse sampling:
    q = S * p + t
where p is RGB/output coordinate and q is TIR/source coordinate.

For translation only:
    q = p + t
Therefore the GT sampling translation that maps the RGB center p_R to the
raw TIR center q_T is:
    t_GT = q_T - p_R

For the final centered geometry:
    q = c_T + S * (p - c_R)
      = S * p + (c_T - S*c_R)

TIR GT box -> RGB/output box:
    p_center = (q_center - t) / S
    p_size   = q_size / S
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,
    build_dual_resolution_dataloader,
)
from engine.runtime import autocast_context, preprocess_batch, select_device
from models.gtga_bhlr_factory import load_model_from_checkpoint


# =============================================================================
# Basic box geometry
# =============================================================================

def xywhn_to_xyxy(box: Sequence[float], width: int, height: int) -> np.ndarray:
    cx, cy, bw, bh = [float(v) for v in box]
    return np.array(
        [
            (cx - bw / 2.0) * width,
            (cy - bh / 2.0) * height,
            (cx + bw / 2.0) * width,
            (cy + bh / 2.0) * height,
        ],
        dtype=np.float32,
    )


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih

    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def transform_tir_box_to_rgb(
    tir_box_xywhn: np.ndarray,
    sampling_scale: np.ndarray,
    sampling_translation: np.ndarray,
) -> np.ndarray:
    """
    q = S*p + t  =>  p = (q-t)/S
    """
    b = np.asarray(tir_box_xywhn, np.float32).reshape(4)
    s = np.maximum(np.asarray(sampling_scale, np.float32).reshape(2), 1e-8)
    t = np.asarray(sampling_translation, np.float32).reshape(2)

    out = np.empty(4, dtype=np.float32)
    out[:2] = (b[:2] - t) / s
    out[2:] = b[2:] / s
    return out


def center_delta_px(
    src_xywhn: np.ndarray,
    dst_xywhn: np.ndarray,
    width: int,
    height: int,
) -> Tuple[float, float, float]:
    dx = (float(dst_xywhn[0]) - float(src_xywhn[0])) * float(width)
    dy = (float(dst_xywhn[1]) - float(src_xywhn[1])) * float(height)
    return dx, dy, float(math.hypot(dx, dy))


def pair_boxes_by_center(
    rgb_boxes: np.ndarray,
    tir_boxes: np.ndarray,
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    """
    UAV-CB is normally one object per image.
    Multi-object samples use deterministic greedy nearest-center pairing.
    """
    nr, nt = len(rgb_boxes), len(tir_boxes)
    if nr == 0 or nt == 0:
        return []
    if nr == 1 and nt == 1:
        return [(0, 0)]

    candidates = []
    for ri, rb in enumerate(rgb_boxes):
        for ti, tb in enumerate(tir_boxes):
            dx = (float(rb[0]) - float(tb[0])) * width
            dy = (float(rb[1]) - float(tb[1])) * height
            candidates.append((math.hypot(dx, dy), ri, ti))
    candidates.sort(key=lambda x: x[0])

    used_r, used_t, pairs = set(), set(), []
    for _, ri, ti in candidates:
        if ri in used_r or ti in used_t:
            continue
        used_r.add(ri)
        used_t.add(ti)
        pairs.append((ri, ti))
        if len(pairs) >= min(nr, nt):
            break
    return pairs


def stage_boxes(
    tir_boxes: np.ndarray,
    scale_xy: Optional[np.ndarray],
    trans_xy: Optional[np.ndarray],
) -> np.ndarray:
    if len(tir_boxes) == 0 or scale_xy is None or trans_xy is None:
        return np.zeros((0, 4), np.float32)
    return np.stack(
        [transform_tir_box_to_rgb(b, scale_xy, trans_xy) for b in tir_boxes],
        axis=0,
    )


# =============================================================================
# Tensor extraction
# =============================================================================

def extract_batch_scalar(x: Any, bi: int, default=float("nan")) -> float:
    if x is None:
        return float(default)
    if torch.is_tensor(x):
        y = x.detach().float().cpu()
        if y.ndim == 0:
            return float(y.item())
        return float(y[bi].reshape(-1)[0].item())
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a.item())
    return float(a[bi].reshape(-1)[0])


def extract_batch_vec2(
    x: Any,
    bi: int,
    default=(float("nan"), float("nan")),
) -> np.ndarray:
    if x is None:
        return np.asarray(default, dtype=np.float32)
    if torch.is_tensor(x):
        y = x.detach().float().cpu()[bi].reshape(-1).numpy()
    else:
        y = np.asarray(x)[bi].reshape(-1)
    if y.size < 2:
        return np.asarray(default, dtype=np.float32)
    return y[:2].astype(np.float32)


def get_ref_debug(info: Mapping[str, Any], model) -> Mapping[str, Any]:
    ref_scale = int(
        getattr(
            model,
            "gtga_reference_scale",
            getattr(getattr(model, "alignment", None), "reference_scale", 3),
        )
    )
    scale_debug = info.get("scale_debug", {}) or {}
    if ref_scale in scale_debug:
        return scale_debug[ref_scale]
    if str(ref_scale) in scale_debug:
        return scale_debug[str(ref_scale)]
    dbg = info.get("debug", None)
    return dbg if isinstance(dbg, Mapping) else info


def feature_offset_to_normalized(
    offset_xy: np.ndarray,
    feat_w: int,
    feat_h: int,
) -> np.ndarray:
    return np.array(
        [
            float(offset_xy[0]) / max(feat_w - 1, 1),
            float(offset_xy[1]) / max(feat_h - 1, 1),
        ],
        dtype=np.float32,
    )


# =============================================================================
# Visualization
# =============================================================================

RGB_GT = (0, 220, 0)
RAW_TIR = (0, 165, 255)
GLOBAL_SOFT = (255, 180, 0)
DMLS = (255, 0, 255)
GMLS = (0, 255, 255)
FINE = (255, 255, 0)
FINAL = (0, 80, 255)
TEXT = (245, 245, 245)


def tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
    x = img.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x[:3].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _clip_box(box, w, h):
    x1, y1, x2, y2 = map(float, box)
    return (
        int(round(np.clip(x1, 0, w - 1))),
        int(round(np.clip(y1, 0, h - 1))),
        int(round(np.clip(x2, 0, w - 1))),
        int(round(np.clip(y2, 0, h - 1))),
    )


def draw_box(img, box, color, label, thickness=2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (10, 10, 10), thickness + 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.putText(
        img, label, (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, label, (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA,
    )


def put_lines(img, lines, x=10, y=24):
    for i, line in enumerate(lines):
        yy = y + i * 21
        cv2.putText(
            img, line, (x + 1, yy + 1), cv2.FONT_HERSHEY_SIMPLEX,
            0.46, (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
            0.46, TEXT, 1, cv2.LINE_AA,
        )


def make_stage_panel(
    base: np.ndarray,
    rgb_boxes: np.ndarray,
    tir_boxes: np.ndarray,
    global_boxes: np.ndarray,
    dmls_boxes: np.ndarray,
    gmls_boxes: np.ndarray,
    fine_boxes: np.ndarray,
    final_boxes: np.ndarray,
    pairs: List[Tuple[int, int]],
    metrics: List[Dict[str, Any]],
) -> np.ndarray:
    panels = []
    names = [
        ("RAW", tir_boxes, RAW_TIR, "iou_before"),
        ("GLOBAL SOFT", global_boxes, GLOBAL_SOFT, "iou_coarse_global_soft"),
        ("DMLS", dmls_boxes, DMLS, "iou_coarse_dmls"),
        ("GMLS", gmls_boxes, GMLS, "iou_coarse_gmls"),
        ("GMLS + FINE", fine_boxes, FINE, "iou_translation_final"),
        ("FINAL + SCALE", final_boxes, FINAL, "iou_after"),
    ]

    h, w = base.shape[:2]

    for title, boxes, color, metric_key in names:
        im = base.copy()
        values = []
        for k, (ri, ti) in enumerate(pairs):
            if ri >= len(rgb_boxes) or ti >= len(boxes):
                continue
            draw_box(im, xywhn_to_xyxy(rgb_boxes[ri], w, h), RGB_GT, "RGB GT")
            draw_box(im, xywhn_to_xyxy(boxes[ti], w, h), color, title)
            if k < len(metrics):
                values.append(metrics[k][metric_key])

        mean_iou = float(np.mean(values)) if values else float("nan")
        put_lines(im, [title, f"mean IoU={mean_iou:.4f}"])
        panels.append(im)

    divider = np.full((h, 4, 3), 250, np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.concatenate([out, divider, p], axis=1)
    return out


# =============================================================================
# Statistics
# =============================================================================

def safe_stats(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(list(values), np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def tail_counts(values: Sequence[float]) -> Dict[str, int]:
    x = np.asarray(list(values), np.float64)
    x = x[np.isfinite(x)]
    return {
        "zero": int(np.sum(x <= 1e-12)),
        "lt_0p1": int(np.sum(x < 0.1)),
        "lt_0p3": int(np.sum(x < 0.3)),
    }


# =============================================================================
# Main evaluation
# =============================================================================

@torch.no_grad()
def evaluate(args):
    device = select_device(args.device)

    model, _, cfg = load_model_from_checkpoint(
        args.weights,
        device=device,
        fusion_override=args.fusion,
        strict=True,
        verbose=False,
    )
    model.eval()

    data_cfg = cfg.get("data", {}) or {}
    rgb_yaml = args.rgb or data_cfg.get("rgb")
    tir_yaml = args.tir or data_cfg.get("tir")
    if rgb_yaml is None or tir_yaml is None:
        raise ValueError("RGB/TIR dataset YAML could not be resolved.")

    rgb_imgsz = int(args.rgb_imgsz or data_cfg.get("rgb_imgsz", 1920))
    sem_imgsz = int(args.rgb_semantic_imgsz or data_cfg.get("rgb_semantic_imgsz", 640))
    tir_imgsz = int(args.tir_imgsz or data_cfg.get("tir_imgsz", 640))
    pair_mode = args.pair_mode or data_cfg.get("pair_mode", "relative")

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

    loader = build_dual_resolution_dataloader(
        dataset,
        batch_size=int(args.batch),
        workers=int(args.workers),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    ckpt_path = Path(args.weights).expanduser().resolve()
    save_dir = (
        Path(args.save_dir).expanduser().resolve()
        if args.save_dir
        else ckpt_path.parent.parent / f"gmls_alignment_geometry_{args.split}"
    )
    boxed_dir = save_dir / "boxed_images"
    rescued_dir = save_dir / "dmls_rescued_zero"
    created_dir = save_dir / "dmls_created_zero"
    final_zero_dir = save_dir / "final_zero_iou"
    for p in (save_dir, boxed_dir, rescued_dir, created_dir, final_zero_dir):
        p.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []
    image_index = 0

    pbar = tqdm(loader, desc=f"DMLS alignment {args.split}")

    for batch in pbar:
        batch = preprocess_batch(batch, device)

        if hasattr(model, "build_alignment_valid_masks_from_batch"):
            rgb_valid_mask, tir_valid_mask = model.build_alignment_valid_masks_from_batch(batch)
        else:
            rgb_valid_mask = tir_valid_mask = None

        with autocast_context(device, enabled=bool(args.amp)):
            outputs = model.predict(
                batch["rgb_img"],
                batch["tir_img"],
                rgb_semantic=batch["rgb_semantic_img"],
                rgb_valid_mask=rgb_valid_mask,
                tir_valid_mask=tir_valid_mask,
                return_features=True,
                return_alignment_debug=True,
            )

        info = outputs["alignment_info"]
        dbg = get_ref_debug(info, model)

        # Global / DMLS / GMLS coarse fields.
        global_soft_all = dbg.get("coarse_offset_global_soft", None)
        dmls_all = dbg.get("coarse_offset_dmls", None)
        gmls_all = dbg.get("coarse_offset_gmls", dbg.get("coarse_offset_initial", None))
        peak_all = dbg.get("coarse_peak_offset", None)
        fine_all = dbg.get("fine_offset", None)
        refined_all = dbg.get("refined_offset", dbg.get("coarse_offset", None))
        final_scale_all = dbg.get("sampling_scale", info.get("sampling_scale", None))
        final_trans_all = dbg.get("sampling_translation", info.get("sampling_translation", None))

        if global_soft_all is None:
            raise KeyError(
                "Missing 'coarse_offset_global_soft'. "
                "This evaluator requires DMLS-Align v1."
            )
        if dmls_all is None:
            raise KeyError("Missing 'coarse_offset_dmls'. This evaluator requires GMLS-Align v1.")
        if gmls_all is None or refined_all is None:
            raise KeyError("Missing GMLS coarse/refined offset.")
        if final_scale_all is None or final_trans_all is None:
            raise KeyError("Missing final sampling geometry.")

        feat_h, feat_w = dbg["rgb_targetness_raw"].shape[-2:]

        bsz = int(batch["rgb_semantic_img"].shape[0])
        h = int(batch["rgb_semantic_img"].shape[2])
        w = int(batch["rgb_semantic_img"].shape[3])

        for bi in range(bsz):
            rmask = batch["rgb_batch_idx"] == bi
            tmask = batch["tir_batch_idx"] == bi

            rgb_boxes = batch["rgb_semantic_bboxes"][rmask].detach().float().cpu().numpy()
            tir_boxes = batch["tir_bboxes"][tmask].detach().float().cpu().numpy()

            global_soft = extract_batch_vec2(global_soft_all, bi)
            dmls = extract_batch_vec2(dmls_all, bi)
            gmls = extract_batch_vec2(gmls_all, bi)
            peak = extract_batch_vec2(peak_all, bi)
            fine = extract_batch_vec2(fine_all, bi, default=(0.0, 0.0))
            refined = extract_batch_vec2(refined_all, bi)

            global_norm = feature_offset_to_normalized(global_soft, feat_w, feat_h)
            dmls_norm = feature_offset_to_normalized(dmls, feat_w, feat_h)
            gmls_norm = feature_offset_to_normalized(gmls, feat_w, feat_h)
            refined_norm = feature_offset_to_normalized(refined, feat_w, feat_h)

            unit_scale = np.ones(2, np.float32)
            global_boxes = stage_boxes(tir_boxes, unit_scale, global_norm)
            dmls_boxes = stage_boxes(tir_boxes, unit_scale, dmls_norm)
            gmls_boxes = stage_boxes(tir_boxes, unit_scale, gmls_norm)
            fine_boxes = stage_boxes(tir_boxes, unit_scale, refined_norm)

            final_scale = extract_batch_vec2(final_scale_all, bi, default=(1.0, 1.0))
            final_trans = extract_batch_vec2(final_trans_all, bi, default=(0.0, 0.0))
            final_boxes = stage_boxes(tir_boxes, final_scale, final_trans)

            global_to_dmls = dmls - global_soft
            global_to_dmls_dist = float(np.linalg.norm(global_to_dmls))
            dmls_to_gmls = gmls - dmls
            dmls_to_gmls_dist = float(np.linalg.norm(dmls_to_gmls))
            global_to_gmls = gmls - global_soft
            global_to_gmls_dist = float(np.linalg.norm(global_to_gmls))

            pairs = pair_boxes_by_center(rgb_boxes, tir_boxes, w, h)
            pair_metrics: List[Dict[str, Any]] = []

            for pair_id, (ri, ti) in enumerate(pairs):
                rb = rgb_boxes[ri]
                tb = tir_boxes[ti]

                rb_xyxy = xywhn_to_xyxy(rb, w, h)
                tb_xyxy = xywhn_to_xyxy(tb, w, h)

                gb = global_boxes[ti]
                db = dmls_boxes[ti]
                hb = gmls_boxes[ti]
                fb = fine_boxes[ti]
                ab = final_boxes[ti]

                iou_before = bbox_iou_xyxy(rb_xyxy, tb_xyxy)
                iou_global = bbox_iou_xyxy(rb_xyxy, xywhn_to_xyxy(gb, w, h))
                iou_dmls = bbox_iou_xyxy(rb_xyxy, xywhn_to_xyxy(db, w, h))
                iou_gmls = bbox_iou_xyxy(rb_xyxy, xywhn_to_xyxy(hb, w, h))
                iou_fine = bbox_iou_xyxy(rb_xyxy, xywhn_to_xyxy(fb, w, h))
                iou_after = bbox_iou_xyxy(rb_xyxy, xywhn_to_xyxy(ab, w, h))

                # Correct inverse-sampling GT translation:
                # q_TIR = p_RGB + t  =>  t = q_TIR - p_RGB.
                gt_tx_p3 = (float(tb[0]) - float(rb[0])) * max(feat_w - 1, 1)
                gt_ty_p3 = (float(tb[1]) - float(rb[1])) * max(feat_h - 1, 1)

                global_translation_error = math.hypot(
                    float(global_soft[0]) - gt_tx_p3,
                    float(global_soft[1]) - gt_ty_p3,
                )
                dmls_translation_error = math.hypot(
                    float(dmls[0]) - gt_tx_p3,
                    float(dmls[1]) - gt_ty_p3,
                )
                gmls_translation_error = math.hypot(
                    float(gmls[0]) - gt_tx_p3,
                    float(gmls[1]) - gt_ty_p3,
                )
                refined_translation_error = math.hypot(
                    float(refined[0]) - gt_tx_p3,
                    float(refined[1]) - gt_ty_p3,
                )
                peak_translation_error = math.hypot(
                    float(peak[0]) - gt_tx_p3,
                    float(peak[1]) - gt_ty_p3,
                )

                _, _, center_before = center_delta_px(tb, rb, w, h)
                _, _, center_after = center_delta_px(ab, rb, w, h)

                gt_sx = float(tb[2]) / max(float(rb[2]), 1e-8)
                gt_sy = float(tb[3]) / max(float(rb[3]), 1e-8)

                dmls_improved = int(iou_dmls > iou_global + 1e-12)
                dmls_worsened = int(iou_dmls < iou_global - 1e-12)
                rescued_zero = int(iou_global <= 1e-12 and iou_dmls > 1e-12)
                created_zero = int(iou_global > 1e-12 and iou_dmls <= 1e-12)
                gmls_improved_vs_dmls = int(iou_gmls > iou_dmls + 1e-12)
                gmls_worsened_vs_dmls = int(iou_gmls < iou_dmls - 1e-12)
                gmls_rescued_dmls_zero = int(iou_dmls <= 1e-12 and iou_gmls > 1e-12)
                gmls_created_zero_from_dmls = int(iou_dmls > 1e-12 and iou_gmls <= 1e-12)

                fine_improved = int(iou_fine > iou_gmls + 1e-12)
                fine_worsened = int(iou_fine < iou_gmls - 1e-12)
                scale_improved = int(iou_after > iou_fine + 1e-12)
                scale_worsened = int(iou_after < iou_fine - 1e-12)

                row = {
                    "image_index": image_index,
                    "pair_id": pair_id,
                    "pair_key": str(batch.get("pair_key", [""] * bsz)[bi]),
                    "rgb_path": str(batch["rgb_path"][bi]),
                    "tir_path": str(batch["tir_path"][bi]),
                    "rgb_index": ri,
                    "tir_index": ti,

                    # IoU stages
                    "iou_before": iou_before,
                    "iou_coarse_global_soft": iou_global,
                    "iou_coarse_dmls": iou_dmls,
                    "iou_coarse_gmls": iou_gmls,
                    "iou_translation_final": iou_fine,
                    "iou_after": iou_after,

                    # Incremental gains
                    "gain_global_vs_raw": iou_global - iou_before,
                    "gain_dmls_vs_global": iou_dmls - iou_global,
                    "gain_gmls_vs_dmls": iou_gmls - iou_dmls,
                    "gain_fine_vs_gmls": iou_fine - iou_gmls,
                    "gain_scale_vs_translation": iou_after - iou_fine,
                    "gain_total": iou_after - iou_before,

                    # DMLS outcome flags
                    "dmls_improved": dmls_improved,
                    "dmls_worsened": dmls_worsened,
                    "dmls_rescued_zero": rescued_zero,
                    "dmls_created_zero": created_zero,
                    "gmls_improved_vs_dmls": gmls_improved_vs_dmls,
                    "gmls_worsened_vs_dmls": gmls_worsened_vs_dmls,
                    "gmls_rescued_dmls_zero": gmls_rescued_dmls_zero,
                    "gmls_created_zero_from_dmls": gmls_created_zero_from_dmls,
                    "fine_improved": fine_improved,
                    "fine_worsened": fine_worsened,
                    "scale_improved": scale_improved,
                    "scale_worsened": scale_worsened,
                    "final_zero": int(iou_after <= 1e-12),

                    # GT translation in P3 cells
                    "gt_sampling_tx_p3": gt_tx_p3,
                    "gt_sampling_ty_p3": gt_ty_p3,

                    # Historical global-soft coarse
                    "global_soft_dx_p3": float(global_soft[0]),
                    "global_soft_dy_p3": float(global_soft[1]),
                    "global_soft_translation_error_p3": global_translation_error,

                    # Dominant peak (diagnostic only)
                    "dominant_peak_dx_p3": float(peak[0]),
                    "dominant_peak_dy_p3": float(peak[1]),
                    "dominant_peak_translation_error_p3": peak_translation_error,

                    # DMLS local-soft coarse
                    "dmls_dx_p3": float(dmls[0]),
                    "dmls_dy_p3": float(dmls[1]),
                    "dmls_translation_error_p3": dmls_translation_error,

                    # GMLS hybrid coarse
                    "gmls_dx_p3": float(gmls[0]),
                    "gmls_dy_p3": float(gmls[1]),
                    "gmls_translation_error_p3": gmls_translation_error,

                    # Differences among coarse readouts
                    "global_to_dmls_dx_p3": float(global_to_dmls[0]),
                    "global_to_dmls_dy_p3": float(global_to_dmls[1]),
                    "global_to_dmls_distance_p3": global_to_dmls_dist,
                    "dmls_to_gmls_dx_p3": float(dmls_to_gmls[0]),
                    "dmls_to_gmls_dy_p3": float(dmls_to_gmls[1]),
                    "dmls_to_gmls_distance_p3": dmls_to_gmls_dist,
                    "global_to_gmls_dx_p3": float(global_to_gmls[0]),
                    "global_to_gmls_dy_p3": float(global_to_gmls[1]),
                    "global_to_gmls_distance_p3": global_to_gmls_dist,

                    # Original Fine stage
                    "fine_dx_p3": float(fine[0]),
                    "fine_dy_p3": float(fine[1]),
                    "final_translation_dx_p3": float(refined[0]),
                    "final_translation_dy_p3": float(refined[1]),
                    "final_translation_error_p3": refined_translation_error,

                    # Original Local-GTGA Moment scale
                    "sx": float(final_scale[0]),
                    "sy": float(final_scale[1]),
                    "gt_sx": gt_sx,
                    "gt_sy": gt_sy,
                    "scale_error_log_x": abs(
                        math.log(max(float(final_scale[0]), 1e-8))
                        - math.log(max(gt_sx, 1e-8))
                    ),
                    "scale_error_log_y": abs(
                        math.log(max(float(final_scale[1]), 1e-8))
                        - math.log(max(gt_sy, 1e-8))
                    ),

                    # Final spatial error
                    "center_distance_before_px": center_before,
                    "center_distance_after_px": center_after,

                    # Original box values
                    "rgb_cx": float(rb[0]),
                    "rgb_cy": float(rb[1]),
                    "rgb_w": float(rb[2]),
                    "rgb_h": float(rb[3]),
                    "tir_cx_before": float(tb[0]),
                    "tir_cy_before": float(tb[1]),
                    "tir_w_before": float(tb[2]),
                    "tir_h_before": float(tb[3]),
                    "tir_cx_after": float(ab[0]),
                    "tir_cy_after": float(ab[1]),
                    "tir_w_after": float(ab[2]),
                    "tir_h_after": float(ab[3]),
                }

                rows.append(row)
                pair_metrics.append(row)

            # -----------------------------------------------------------------
            # Five-stage boxed visualization for every image.
            # -----------------------------------------------------------------
            rgb_bgr = tensor_to_bgr(batch["rgb_semantic_img"][bi])
            panel = make_stage_panel(
                rgb_bgr,
                rgb_boxes,
                tir_boxes,
                global_boxes,
                dmls_boxes,
                gmls_boxes,
                fine_boxes,
                final_boxes,
                pairs,
                pair_metrics,
            )

            stem = Path(str(batch["rgb_path"][bi])).stem
            out_path = boxed_dir / f"{image_index:04d}_{stem}.jpg"
            cv2.imwrite(
                str(out_path), panel,
                [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)],
            )

            has_rescued = any(r["dmls_rescued_zero"] for r in pair_metrics)
            has_created = any(r["dmls_created_zero"] for r in pair_metrics)
            has_final_zero = any(r["final_zero"] for r in pair_metrics)

            if has_rescued:
                cv2.imwrite(
                    str(rescued_dir / out_path.name), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)],
                )
            if has_created:
                cv2.imwrite(
                    str(created_dir / out_path.name), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)],
                )
            if has_final_zero:
                cv2.imwrite(
                    str(final_zero_dir / out_path.name), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)],
                )

            image_rows.append({
                "image_index": image_index,
                "pair_key": str(batch.get("pair_key", [""] * bsz)[bi]),
                "rgb_path": str(batch["rgb_path"][bi]),
                "tir_path": str(batch["tir_path"][bi]),
                "num_pairs": int(len(pair_metrics)),
                "mean_iou_before": (
                    float(np.mean([x["iou_before"] for x in pair_metrics]))
                    if pair_metrics else float("nan")
                ),
                "mean_iou_global_soft": (
                    float(np.mean([x["iou_coarse_global_soft"] for x in pair_metrics]))
                    if pair_metrics else float("nan")
                ),
                "mean_iou_dmls": (
                    float(np.mean([x["iou_coarse_dmls"] for x in pair_metrics]))
                    if pair_metrics else float("nan")
                ),
                "mean_iou_gmls": (
                    float(np.mean([x["iou_coarse_gmls"] for x in pair_metrics]))
                    if pair_metrics else float("nan")
                ),
                "mean_iou_translation_final": (
                    float(np.mean([x["iou_translation_final"] for x in pair_metrics]))
                    if pair_metrics else float("nan")
                ),
                "mean_iou_after": (
                    float(np.mean([x["iou_after"] for x in pair_metrics]))
                    if pair_metrics else float("nan")
                ),
                "dmls_rescued_zero": int(has_rescued),
                "dmls_created_zero": int(has_created),
                "final_zero": int(has_final_zero),
                "boxed_image": str(out_path),
            })

            image_index += 1

        if rows:
            pbar.set_postfix(
                global_iou=f"{np.mean([r['iou_coarse_global_soft'] for r in rows]):.3f}",
                dmls=f"{np.mean([r['iou_coarse_dmls'] for r in rows]):.3f}",
                gmls=f"{np.mean([r['iou_coarse_gmls'] for r in rows]):.3f}",
                final=f"{np.mean([r['iou_after'] for r in rows]):.3f}",
                rescued=sum(r["dmls_rescued_zero"] for r in rows),
                created=sum(r["dmls_created_zero"] for r in rows),
            )

    # =============================================================================
    # Export CSV
    # =============================================================================
    pair_csv = save_dir / "alignment_pairs.csv"
    image_csv = save_dir / "alignment_images.csv"

    if rows:
        with pair_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if image_rows:
        with image_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(image_rows)

    # =============================================================================
    # Summary
    # =============================================================================
    stage_keys = [
        "iou_before",
        "iou_coarse_global_soft",
        "iou_coarse_dmls",
        "iou_coarse_gmls",
        "iou_translation_final",
        "iou_after",
    ]
    stage_iou = {
        key: {
            **safe_stats([r[key] for r in rows]),
            "tail": tail_counts([r[key] for r in rows]),
        }
        for key in stage_keys
    }

    global_err = [r["global_soft_translation_error_p3"] for r in rows]
    dmls_err = [r["dmls_translation_error_p3"] for r in rows]
    gmls_err = [r["gmls_translation_error_p3"] for r in rows]
    refined_err = [r["final_translation_error_p3"] for r in rows]
    peak_err = [r["dominant_peak_translation_error_p3"] for r in rows]

    dmls_improved = int(sum(r["dmls_improved"] for r in rows))
    dmls_worsened = int(sum(r["dmls_worsened"] for r in rows))
    dmls_unchanged = int(len(rows) - dmls_improved - dmls_worsened)

    rescued = int(sum(r["dmls_rescued_zero"] for r in rows))
    created = int(sum(r["dmls_created_zero"] for r in rows))

    gmls_improved = int(sum(r["gmls_improved_vs_dmls"] for r in rows))
    gmls_worsened = int(sum(r["gmls_worsened_vs_dmls"] for r in rows))
    gmls_rescued = int(sum(r["gmls_rescued_dmls_zero"] for r in rows))
    gmls_created = int(sum(r["gmls_created_zero_from_dmls"] for r in rows))

    fine_improved = int(sum(r["fine_improved"] for r in rows))
    fine_worsened = int(sum(r["fine_worsened"] for r in rows))
    scale_improved = int(sum(r["scale_improved"] for r in rows))
    scale_worsened = int(sum(r["scale_worsened"] for r in rows))

    final_improved = int(sum(
        r["iou_after"] > r["iou_before"] + 1e-12 for r in rows
    ))
    final_worsened = int(sum(
        r["iou_after"] < r["iou_before"] - 1e-12 for r in rows
    ))

    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "images": int(image_index),
        "paired_instances": int(len(rows)),
        "rgb_semantic_canvas": [
            w if "w" in locals() else sem_imgsz,
            h if "h" in locals() else sem_imgsz,
        ],
        "feature_canvas": [
            feat_w if "feat_w" in locals() else None,
            feat_h if "feat_h" in locals() else None,
        ],
        "gmls_config": {
            "enabled": bool(getattr(model.alignment.unit, "coarse_gmls_enabled", False)),
            "global_search_radius": int(getattr(model.alignment.unit, "coarse_radius", -1)),
            "local_basin_radius": int(getattr(model.alignment.unit, "coarse_local_radius", -1)),
            "local_basin_size": int(2 * getattr(model.alignment.unit, "coarse_local_radius", 0) + 1),
            "global_temperature": float(getattr(model.alignment.unit, "coarse_temperature", float("nan"))),
            "local_temperature": float(getattr(model.alignment.unit, "coarse_local_temperature", float("nan"))),
            "fine_radius": int(getattr(model.alignment.unit, "fine_translation_radius", -1)),
        },
        "stage_iou": stage_iou,
        "dmls_effect": {
            "improved_instances": dmls_improved,
            "unchanged_instances": dmls_unchanged,
            "worsened_instances": dmls_worsened,
            "improved_percent": 100.0 * dmls_improved / len(rows) if rows else 0.0,
            "worsened_percent": 100.0 * dmls_worsened / len(rows) if rows else 0.0,
            "rescued_zero_instances": rescued,
            "created_zero_instances": created,
            "net_zero_rescue": rescued - created,
            "mean_gain_vs_global": float(np.mean(
                [r["gain_dmls_vs_global"] for r in rows]
            )) if rows else float("nan"),
        },
        "gmls_effect_vs_dmls": {
            "improved_instances": gmls_improved,
            "worsened_instances": gmls_worsened,
            "rescued_dmls_zero_instances": gmls_rescued,
            "created_zero_from_dmls_instances": gmls_created,
            "net_zero_rescue_vs_dmls": gmls_rescued - gmls_created,
            "mean_gain_vs_dmls": float(np.mean(
                [r["gain_gmls_vs_dmls"] for r in rows]
            )) if rows else float("nan"),
        },
        "translation_error_p3": {
            "global_soft": safe_stats(global_err),
            "dominant_peak": safe_stats(peak_err),
            "dmls": safe_stats(dmls_err),
            "gmls": safe_stats(gmls_err),
            "after_fine": safe_stats(refined_err),
            "dmls_better_than_global_instances": int(sum(
                de < ge - 1e-12 for ge, de in zip(global_err, dmls_err)
            )),
            "dmls_worse_than_global_instances": int(sum(
                de > ge + 1e-12 for ge, de in zip(global_err, dmls_err)
            )),
            "gmls_better_than_dmls_instances": int(sum(
                he < de - 1e-12 for de, he in zip(dmls_err, gmls_err)
            )),
            "gmls_worse_than_dmls_instances": int(sum(
                he > de + 1e-12 for de, he in zip(dmls_err, gmls_err)
            )),
        },
        "fine_effect": {
            "improved_instances": fine_improved,
            "worsened_instances": fine_worsened,
            "mean_gain": float(np.mean(
                [r["gain_fine_vs_gmls"] for r in rows]
            )) if rows else float("nan"),
        },
        "scale_effect": {
            "improved_instances": scale_improved,
            "worsened_instances": scale_worsened,
            "mean_gain": float(np.mean(
                [r["gain_scale_vs_translation"] for r in rows]
            )) if rows else float("nan"),
        },
        "final_effect": {
            "improved_vs_raw_instances": final_improved,
            "worsened_vs_raw_instances": final_worsened,
            "improved_percent": 100.0 * final_improved / len(rows) if rows else 0.0,
            "worsened_percent": 100.0 * final_worsened / len(rows) if rows else 0.0,
        },
        "center_distance_before_px": safe_stats(
            [r["center_distance_before_px"] for r in rows]
        ),
        "center_distance_after_px": safe_stats(
            [r["center_distance_after_px"] for r in rows]
        ),
        "outputs": {
            "pair_csv": str(pair_csv),
            "image_csv": str(image_csv),
            "boxed_images": str(boxed_dir),
            "dmls_rescued_zero_images": str(rescued_dir),
            "dmls_created_zero_images": str(created_dir),
            "final_zero_images": str(final_zero_dir),
        },
    }

    summary_path = save_dir / "alignment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # =============================================================================
    # Console report
    # =============================================================================
    print("\n" + "=" * 108)
    print("GMLS-ALIGN v1 EVALUATION")
    print("=" * 108)
    print(f"Images / paired instances       : {image_index} / {len(rows)}")
    print(
        f"Search                          : global {2*summary['gmls_config']['global_search_radius']+1}x"
        f"{2*summary['gmls_config']['global_search_radius']+1} -> "
        f"local {summary['gmls_config']['local_basin_size']}x"
        f"{summary['gmls_config']['local_basin_size']}"
    )
    print("-" * 108)

    for key in stage_keys:
        s = stage_iou[key]
        print(
            f"{key:<30} "
            f"mean={s['mean']:.6f}  med={s['median']:.6f}  "
            f"P10={s['p10']:.6f}  zero={s['tail']['zero']:3d}  "
            f"<0.1={s['tail']['lt_0p1']:3d}  <0.3={s['tail']['lt_0p3']:3d}"
        )

    print("-" * 108)
    print(f"DMLS improved / worsened Global : {dmls_improved} / {dmls_worsened}")
    print(f"DMLS rescued / created zero     : {rescued} / {created}")
    print(f"GMLS improved / worsened DMLS   : {gmls_improved} / {gmls_worsened}")
    print(f"GMLS rescued / created DMLS zero: {gmls_rescued} / {gmls_created}")
    print(f"GMLS net zero rescue vs DMLS    : {gmls_rescued - gmls_created:+d}")
    print(
        f"Mean P3 translation error       : "
        f"global={summary['translation_error_p3']['global_soft']['mean']:.4f}  "
        f"peak={summary['translation_error_p3']['dominant_peak']['mean']:.4f}  "
        f"DMLS={summary['translation_error_p3']['dmls']['mean']:.4f}  "
        f"GMLS={summary['translation_error_p3']['gmls']['mean']:.4f}  "
        f"afterFine={summary['translation_error_p3']['after_fine']['mean']:.4f}"
    )
    print(f"Fine improved / worsened        : {fine_improved} / {fine_worsened}")
    print(f"Scale improved / worsened       : {scale_improved} / {scale_worsened}")
    print("-" * 108)
    print(f"Pair CSV                        : {pair_csv}")
    print(f"Image CSV                       : {image_csv}")
    print(f"DMLS rescued-zero images        : {rescued_dir}")
    print(f"DMLS created-zero images        : {created_dir}")
    print(f"Final zero-IoU images           : {final_zero_dir}")
    print(f"Summary                         : {summary_path}")
    print("=" * 108)

    return summary


def parser():
    p = argparse.ArgumentParser(
        description="Evaluate Global Soft vs DMLS vs GMLS alignment stage by stage."
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--rgb", default=None)
    p.add_argument("--tir", default=None)
    p.add_argument("--rgb-imgsz", type=int, default=None)
    p.add_argument("--rgb-semantic-imgsz", type=int, default=None)
    p.add_argument("--tir-imgsz", type=int, default=None)
    p.add_argument("--pair-mode", default=None, choices=["relative", "stem", "filename"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--fusion", default="auto", choices=["auto", "concat", "add", "weighted"])
    p.add_argument("--save-dir", default=None)
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast. Default FP32 is recommended for geometry diagnostics.",
    )
    p.add_argument("--jpg-quality", type=int, default=95)
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())
