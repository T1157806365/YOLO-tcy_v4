#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust alignment evaluation for:
  Top-K verified coarse translation
  + confidence-gated fine translation
  + Moment/Quantile reliable scale
  + ONE final center-preserving warp.

Compared with the old evaluator, this script does NOT report only before/after.
It evaluates the GT boxes at every important geometry stage:

  raw
  -> raw coarse correlation offset
  -> Top-K verified coarse offset
  -> verified coarse + confidence-gated fine translation
  -> Moment scale
  -> Quantile scale
  -> Moment/Quantile fused scale (before reliability fallback)
  -> FINAL reliability-gated scale

This directly answers:
  1) Did Top-K verification rescue/worsen the coarse translation?
  2) Did the confidence-gated fine stage help or hurt?
  3) Was translation already correct before scale?
  4) Did Moment scale, Quantile scale, or final scale destroy a good translation?
  5) How many IoU=0 / IoU<0.1 / IoU<0.3 tail failures remain?

Geometry convention
-------------------
The model uses inverse sampling:
    q = S * p + t
where:
    p = RGB/output coordinate
    q = TIR/source coordinate

Therefore TIR GT q maps to RGB/output p by:
    p_center = (q_center - t) / S
    p_size   = q_size / S

All reported box metrics use the 640 RGB-semantic/TIR canvas.
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
# Box / geometry helpers
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
    b = np.asarray(tir_box_xywhn, dtype=np.float32).reshape(4)
    s = np.maximum(np.asarray(sampling_scale, dtype=np.float32).reshape(2), 1e-8)
    t = np.asarray(sampling_translation, dtype=np.float32).reshape(2)
    out = np.empty(4, dtype=np.float32)
    out[0:2] = (b[0:2] - t) / s
    out[2:4] = b[2:4] / s
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
    candidates.sort(key=lambda z: z[0])

    used_r, used_t, pairs = set(), set(), []
    for _, ri, ti in candidates:
        if ri in used_r or ti in used_t:
            continue
        used_r.add(ri)
        used_t.add(ti)
        pairs.append((ri, ti))
        if len(pairs) == min(nr, nt):
            break
    return pairs


def stage_boxes(
    tir_boxes: np.ndarray,
    scale_xy: Optional[np.ndarray],
    translation_xy: Optional[np.ndarray],
) -> np.ndarray:
    if scale_xy is None or translation_xy is None or len(tir_boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(
        [transform_tir_box_to_rgb(b, scale_xy, translation_xy) for b in tir_boxes],
        axis=0,
    )


def center_preserving_translation(
    anchor_xy: np.ndarray,
    source_center_xy: np.ndarray,
    scale_xy: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(source_center_xy, np.float32)
        - np.asarray(scale_xy, np.float32) * np.asarray(anchor_xy, np.float32)
    ).astype(np.float32)


# =============================================================================
# Tensor extraction helpers
# =============================================================================

def extract_batch_scalar(x: Any, bi: int, default: float = float("nan")) -> float:
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
    default: Tuple[float, float] = (float("nan"), float("nan")),
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


def extract_batch_row(x: Any, bi: int) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu()[bi].numpy()
    return np.asarray(x)[bi]


def finite_vec2(x: np.ndarray) -> bool:
    return x is not None and np.asarray(x).size >= 2 and np.isfinite(np.asarray(x)[:2]).all()


def get_ref_debug(info: Mapping[str, Any], model) -> Mapping[str, Any]:
    ref_scale = int(
        getattr(
            model,
            "gtga_reference_scale",
            getattr(getattr(model, "alignment", None), "reference_scale", 3),
        )
    )
    dbg_map = info.get("scale_debug", {}) or {}
    if ref_scale in dbg_map:
        return dbg_map[ref_scale]
    if str(ref_scale) in dbg_map:
        return dbg_map[str(ref_scale)]
    dbg = info.get("debug", None)
    return dbg if isinstance(dbg, Mapping) else info


def offset_feature_to_norm(
    offset: Optional[np.ndarray],
    feat_w: int,
    feat_h: int,
) -> Optional[np.ndarray]:
    if offset is None:
        return None
    o = np.asarray(offset, np.float32).reshape(-1)
    if o.size < 2 or not np.isfinite(o[:2]).all():
        return None
    return np.array(
        [
            float(o[0]) / max(feat_w - 1, 1),
            float(o[1]) / max(feat_h - 1, 1),
        ],
        dtype=np.float32,
    )


# =============================================================================
# Visualization
# =============================================================================

RGB_GT = (0, 255, 0)
RAW_TIR = (0, 165, 255)
FINAL_TIR = (255, 255, 0)
TEXT = (255, 255, 255)


def tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
    x = img.detach().float().cpu().clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def clip_xyxy(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    return (
        int(round(np.clip(x1, 0, width - 1))),
        int(round(np.clip(y1, 0, height - 1))),
        int(round(np.clip(x2, 0, width - 1))),
        int(round(np.clip(y2, 0, height - 1))),
    )


def draw_box(img, box, color, label, thickness=2):
    out = img
    h, w = out.shape[:2]
    x1, y1, x2, y2 = clip_xyxy(box, w, h)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.putText(
        out, label, (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
    )


def put_lines(img, lines, x=10, y=24):
    for i, line in enumerate(lines):
        yy = y + i * 21
        cv2.putText(img, line, (x + 1, yy + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, TEXT, 1, cv2.LINE_AA)


def make_final_panel(
    rgb_bgr: np.ndarray,
    rgb_boxes: np.ndarray,
    tir_boxes: np.ndarray,
    final_boxes: np.ndarray,
    pair_metrics: List[Dict[str, Any]],
    stage_info: Dict[str, float],
) -> np.ndarray:
    before = rgb_bgr.copy()
    after = rgb_bgr.copy()
    h, w = before.shape[:2]

    for k, row in enumerate(pair_metrics):
        ri, ti = int(row["rgb_index"]), int(row["tir_index"])
        draw_box(before, xywhn_to_xyxy(rgb_boxes[ri], w, h), RGB_GT, f"RGB GT #{k}")
        draw_box(before, xywhn_to_xyxy(tir_boxes[ti], w, h), RAW_TIR, f"Raw TIR #{k}")
        draw_box(after, xywhn_to_xyxy(rgb_boxes[ri], w, h), RGB_GT, f"RGB GT #{k}")
        draw_box(after, xywhn_to_xyxy(final_boxes[ti], w, h), FINAL_TIR, f"Final TIR #{k}")

    before_iou = np.mean([r["iou_before"] for r in pair_metrics]) if pair_metrics else float("nan")
    final_iou = np.mean([r["iou_after"] for r in pair_metrics]) if pair_metrics else float("nan")
    put_lines(before, ["RAW RGB/TIR GT", f"mean IoU={before_iou:.4f}"])
    put_lines(
        after,
        [
            "FINAL ROBUST ALIGNMENT",
            f"mean IoU={final_iou:.4f}",
            f"coarse verified conf={stage_info.get('coarse_conf', float('nan')):.3f}",
            f"fine gate={stage_info.get('fine_gate', float('nan')):.3f}",
            f"scale conf x/y=({stage_info.get('scale_conf_x', float('nan')):.3f},"
            f"{stage_info.get('scale_conf_y', float('nan')):.3f})",
        ],
    )
    divider = np.full((h, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([before, divider, after], axis=1)


# =============================================================================
# Statistics
# =============================================================================

def safe_stats(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": float("nan"), "median": float("nan"),
            "p10": float("nan"), "p25": float("nan"),
            "min": float("nan"), "max": float("nan"),
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
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return {
        "zero": int(np.sum(x <= 1e-12)),
        "lt_0p1": int(np.sum(x < 0.1)),
        "lt_0p3": int(np.sum(x < 0.3)),
    }


# =============================================================================
# Main
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
        raise ValueError("RGB/TIR dataset YAML unresolved.")

    rgb_imgsz = int(args.rgb_imgsz or data_cfg.get("rgb_imgsz", 1920))
    rgb_semantic_imgsz = int(args.rgb_semantic_imgsz or data_cfg.get("rgb_semantic_imgsz", 640))
    tir_imgsz = int(args.tir_imgsz or data_cfg.get("tir_imgsz", 640))
    pair_mode = args.pair_mode or data_cfg.get("pair_mode", "relative")

    dataset = build_dual_resolution_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=args.split,
        rgb_high_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=rgb_semantic_imgsz,
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
        else ckpt_path.parent.parent / f"alignment_robust_geometry_{args.split}"
    )
    image_dir = save_dir / "boxed_images"
    zero_dir = save_dir / "zero_iou_images"
    save_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    zero_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []
    global_image_index = 0

    pbar = tqdm(loader, desc=f"robust-alignment {args.split}")

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

        final_scale_all = info.get("sampling_scale", dbg.get("sampling_scale"))
        final_trans_all = info.get("sampling_translation", dbg.get("sampling_translation"))
        if final_scale_all is None or final_trans_all is None:
            raise KeyError("Final sampling_scale/sampling_translation are missing.")

        # Robust diagnostics (all optional for backward compatibility).
        coarse_raw_all = dbg.get("coarse_offset_raw", None)
        coarse_verified_all = dbg.get("coarse_offset_initial", dbg.get("coarse_offset", None))
        refined_all = dbg.get("refined_offset", dbg.get("coarse_offset", None))
        fine_raw_all = dbg.get("fine_offset_raw", None)
        fine_gated_all = dbg.get("fine_offset", None)
        coarse_conf_raw_all = dbg.get("coarse_correlation_confidence_raw", None)
        coarse_conf_all = dbg.get("coarse_correlation_confidence", None)
        fine_peak_conf_all = dbg.get("fine_peak_confidence", None)
        fine_entropy_conf_all = dbg.get("fine_entropy_confidence", None)
        fine_gate_all = dbg.get("fine_confidence_gate", None)

        candidate_offsets_all = dbg.get("coarse_candidate_offsets", None)
        verify_prob_all = dbg.get("coarse_verification_probability", None)
        verify_score_all = dbg.get("coarse_verification_score", None)
        verify_iou_all = dbg.get("coarse_verification_iou", None)
        verify_cos_all = dbg.get("coarse_verification_cosine", None)
        winner_all = dbg.get("coarse_verification_winner", None)

        anchor_all = dbg.get("coarse_anchor", info.get("coarse_anchor"))
        source_center_all = dbg.get("source_center", info.get("source_center"))

        scale_raw_all = dbg.get("sampling_scale_raw", None)
        scale_moment_all = dbg.get("sampling_scale_moment", scale_raw_all)
        scale_quantile_all = dbg.get("sampling_scale_quantile", None)
        scale_fused_all = dbg.get("sampling_scale_fused", scale_moment_all)
        scale_conf_all = dbg.get("scale_reliability_confidence", None)
        scale_agreement_all = dbg.get("scale_estimator_agreement", None)
        scale_log_dis_all = dbg.get("scale_log_disagreement", None)

        feat_h, feat_w = dbg["rgb_targetness_raw"].shape[-2:]

        bsz = int(batch["rgb_semantic_img"].shape[0])
        h = int(batch["rgb_semantic_img"].shape[2])
        w = int(batch["rgb_semantic_img"].shape[3])

        for bi in range(bsz):
            rmask = batch["rgb_batch_idx"] == bi
            tmask = batch["tir_batch_idx"] == bi
            rgb_boxes = batch["rgb_semantic_bboxes"][rmask].detach().float().cpu().numpy()
            tir_boxes = batch["tir_bboxes"][tmask].detach().float().cpu().numpy()

            # -------- translation stages --------
            coarse_raw = extract_batch_vec2(coarse_raw_all, bi)
            coarse_verified = extract_batch_vec2(coarse_verified_all, bi)
            refined = extract_batch_vec2(refined_all, bi)

            coarse_raw_norm = offset_feature_to_norm(coarse_raw, feat_w, feat_h)
            coarse_verified_norm = offset_feature_to_norm(coarse_verified, feat_w, feat_h)
            refined_norm = offset_feature_to_norm(refined, feat_w, feat_h)

            unit = np.ones(2, dtype=np.float32)
            boxes_coarse_raw = stage_boxes(tir_boxes, unit, coarse_raw_norm)
            boxes_coarse_verified = stage_boxes(tir_boxes, unit, coarse_verified_norm)
            boxes_translation = stage_boxes(tir_boxes, unit, refined_norm)

            # -------- scale stages --------
            anchor = extract_batch_vec2(anchor_all, bi)
            source_center = extract_batch_vec2(source_center_all, bi)
            scale_raw = extract_batch_vec2(scale_raw_all, bi)
            scale_moment = extract_batch_vec2(scale_moment_all, bi)
            scale_quantile = extract_batch_vec2(scale_quantile_all, bi)
            scale_fused = extract_batch_vec2(scale_fused_all, bi)
            scale_final = extract_batch_vec2(final_scale_all, bi, default=(1.0, 1.0))
            trans_final = extract_batch_vec2(final_trans_all, bi, default=(0.0, 0.0))

            def scale_stage(scale_xy):
                if not finite_vec2(scale_xy) or not finite_vec2(anchor) or not finite_vec2(source_center):
                    return np.zeros((0, 4), np.float32), None
                t = center_preserving_translation(anchor, source_center, scale_xy)
                return stage_boxes(tir_boxes, scale_xy, t), t

            boxes_moment, trans_moment = scale_stage(scale_moment)
            boxes_quantile, trans_quantile = scale_stage(scale_quantile)
            boxes_fused, trans_fused = scale_stage(scale_fused)
            boxes_final = stage_boxes(tir_boxes, scale_final, trans_final)

            # -------- mechanism diagnostics --------
            coarse_conf_raw = extract_batch_scalar(coarse_conf_raw_all, bi)
            coarse_conf = extract_batch_scalar(coarse_conf_all, bi)
            fine_peak_conf = extract_batch_scalar(fine_peak_conf_all, bi)
            fine_entropy_conf = extract_batch_scalar(fine_entropy_conf_all, bi)
            fine_gate = extract_batch_scalar(fine_gate_all, bi)
            fine_raw = extract_batch_vec2(fine_raw_all, bi, default=(0.0, 0.0))
            fine_gated = extract_batch_vec2(fine_gated_all, bi, default=(0.0, 0.0))
            scale_conf = extract_batch_vec2(scale_conf_all, bi)
            scale_agreement = extract_batch_vec2(scale_agreement_all, bi)
            scale_log_dis = extract_batch_vec2(scale_log_dis_all, bi)

            winner = int(round(extract_batch_scalar(winner_all, bi, default=-1)))
            cand_offsets = extract_batch_row(candidate_offsets_all, bi)
            verify_prob = extract_batch_row(verify_prob_all, bi)
            verify_score = extract_batch_row(verify_score_all, bi)
            verify_iou = extract_batch_row(verify_iou_all, bi)
            verify_cos = extract_batch_row(verify_cos_all, bi)

            winner_prob = float("nan")
            winner_score = float("nan")
            winner_soft_iou = float("nan")
            winner_cosine = float("nan")
            if winner >= 0:
                if verify_prob is not None and winner < len(np.asarray(verify_prob).reshape(-1)):
                    winner_prob = float(np.asarray(verify_prob).reshape(-1)[winner])
                if verify_score is not None and winner < len(np.asarray(verify_score).reshape(-1)):
                    winner_score = float(np.asarray(verify_score).reshape(-1)[winner])
                if verify_iou is not None and winner < len(np.asarray(verify_iou).reshape(-1)):
                    winner_soft_iou = float(np.asarray(verify_iou).reshape(-1)[winner])
                if verify_cos is not None and winner < len(np.asarray(verify_cos).reshape(-1)):
                    winner_cosine = float(np.asarray(verify_cos).reshape(-1)[winner])

            candidates_json = ""
            if cand_offsets is not None:
                candidates_json = json.dumps(np.asarray(cand_offsets).tolist(), ensure_ascii=False)

            pairs = pair_boxes_by_center(rgb_boxes, tir_boxes, w, h)
            pair_metrics = []

            for pair_id, (ri, ti) in enumerate(pairs):
                rb, tb = rgb_boxes[ri], tir_boxes[ti]
                rb_xyxy = xywhn_to_xyxy(rb, w, h)
                tb_xyxy = xywhn_to_xyxy(tb, w, h)

                def stage_iou(boxes):
                    if boxes is None or ti >= len(boxes):
                        return float("nan")
                    return bbox_iou_xyxy(rb_xyxy, xywhn_to_xyxy(boxes[ti], w, h))

                iou_before = bbox_iou_xyxy(rb_xyxy, tb_xyxy)
                iou_coarse_raw = stage_iou(boxes_coarse_raw)
                iou_coarse_verified = stage_iou(boxes_coarse_verified)
                iou_translation = stage_iou(boxes_translation)
                iou_moment = stage_iou(boxes_moment)
                iou_quantile = stage_iou(boxes_quantile)
                iou_fused = stage_iou(boxes_fused)
                iou_after = stage_iou(boxes_final)

                final_box = boxes_final[ti]
                gt_dx, gt_dy, dist_before = center_delta_px(tb, rb, w, h)
                app_dx, app_dy, app_dist = center_delta_px(tb, final_box, w, h)
                res_dx, res_dy, dist_after = center_delta_px(final_box, rb, w, h)

                row = {
                    "image_index": global_image_index,
                    "pair_id": pair_id,
                    "pair_key": str(batch.get("pair_key", [""] * bsz)[bi]),
                    "rgb_path": str(batch["rgb_path"][bi]),
                    "tir_path": str(batch["tir_path"][bi]),
                    "rgb_index": ri,
                    "tir_index": ti,

                    # Stage IoUs
                    "iou_before": iou_before,
                    "iou_coarse_raw": iou_coarse_raw,
                    "iou_coarse_verified": iou_coarse_verified,
                    "iou_translation_final": iou_translation,
                    "iou_scale_moment": iou_moment,
                    "iou_scale_quantile": iou_quantile,
                    "iou_scale_fused": iou_fused,
                    "iou_after": iou_after,

                    # Stage gains
                    "gain_topk_verification": iou_coarse_verified - iou_coarse_raw,
                    "gain_fine_gated": iou_translation - iou_coarse_verified,
                    "gain_final_scale": iou_after - iou_translation,
                    "iou_gain_total": iou_after - iou_before,

                    # Failure flags
                    "translation_zero": int(np.isfinite(iou_translation) and iou_translation <= 1e-12),
                    "final_zero": int(np.isfinite(iou_after) and iou_after <= 1e-12),
                    "scale_destroyed_positive_translation": int(
                        np.isfinite(iou_translation) and np.isfinite(iou_after)
                        and iou_translation > 1e-12 and iou_after <= 1e-12
                    ),
                    "topk_rescued": int(
                        np.isfinite(iou_coarse_raw) and np.isfinite(iou_coarse_verified)
                        and iou_coarse_verified > iou_coarse_raw + 1e-12
                    ),
                    "fine_worsened": int(
                        np.isfinite(iou_coarse_verified) and np.isfinite(iou_translation)
                        and iou_translation < iou_coarse_verified - 1e-12
                    ),
                    "scale_worsened": int(
                        np.isfinite(iou_translation) and np.isfinite(iou_after)
                        and iou_after < iou_translation - 1e-12
                    ),

                    # Center metrics
                    "gt_dx_before_px": gt_dx,
                    "gt_dy_before_px": gt_dy,
                    "center_distance_before_px": dist_before,
                    "applied_dx_px": app_dx,
                    "applied_dy_px": app_dy,
                    "applied_distance_px": app_dist,
                    "residual_dx_after_px": res_dx,
                    "residual_dy_after_px": res_dy,
                    "center_distance_after_px": dist_after,

                    # Translation diagnostics
                    "coarse_raw_dx_p3": float(coarse_raw[0]),
                    "coarse_raw_dy_p3": float(coarse_raw[1]),
                    "coarse_verified_dx_p3": float(coarse_verified[0]),
                    "coarse_verified_dy_p3": float(coarse_verified[1]),
                    "fine_raw_dx_p3": float(fine_raw[0]),
                    "fine_raw_dy_p3": float(fine_raw[1]),
                    "fine_gated_dx_p3": float(fine_gated[0]),
                    "fine_gated_dy_p3": float(fine_gated[1]),
                    "final_dx_p3": float(refined[0]),
                    "final_dy_p3": float(refined[1]),
                    "coarse_confidence_raw": coarse_conf_raw,
                    "coarse_confidence_verified": coarse_conf,
                    "coarse_verification_winner": winner,
                    "coarse_verification_winner_probability": winner_prob,
                    "coarse_verification_winner_score": winner_score,
                    "coarse_verification_winner_soft_iou": winner_soft_iou,
                    "coarse_verification_winner_cosine": winner_cosine,
                    "coarse_candidate_offsets_json": candidates_json,
                    "fine_peak_confidence": fine_peak_conf,
                    "fine_entropy_confidence": fine_entropy_conf,
                    "fine_confidence_gate": fine_gate,

                    # Scale diagnostics
                    "scale_raw_sx": float(scale_raw[0]),
                    "scale_raw_sy": float(scale_raw[1]),
                    "scale_moment_sx": float(scale_moment[0]),
                    "scale_moment_sy": float(scale_moment[1]),
                    "scale_quantile_sx": float(scale_quantile[0]),
                    "scale_quantile_sy": float(scale_quantile[1]),
                    "scale_fused_sx": float(scale_fused[0]),
                    "scale_fused_sy": float(scale_fused[1]),
                    "scale_final_sx": float(scale_final[0]),
                    "scale_final_sy": float(scale_final[1]),
                    "scale_confidence_x": float(scale_conf[0]),
                    "scale_confidence_y": float(scale_conf[1]),
                    "scale_agreement_x": float(scale_agreement[0]),
                    "scale_agreement_y": float(scale_agreement[1]),
                    "scale_log_disagreement_x": float(scale_log_dis[0]),
                    "scale_log_disagreement_y": float(scale_log_dis[1]),

                    # GT box details
                    "rgb_cx": float(rb[0]), "rgb_cy": float(rb[1]),
                    "rgb_w": float(rb[2]), "rgb_h": float(rb[3]),
                    "tir_cx_before": float(tb[0]), "tir_cy_before": float(tb[1]),
                    "tir_w_before": float(tb[2]), "tir_h_before": float(tb[3]),
                    "tir_cx_after": float(final_box[0]), "tir_cy_after": float(final_box[1]),
                    "tir_w_after": float(final_box[2]), "tir_h_after": float(final_box[3]),
                }
                rows.append(row)
                pair_metrics.append(row)

            rgb_bgr = tensor_to_bgr(batch["rgb_semantic_img"][bi])
            panel = make_final_panel(
                rgb_bgr,
                rgb_boxes,
                tir_boxes,
                boxes_final,
                pair_metrics,
                {
                    "coarse_conf": coarse_conf,
                    "fine_gate": fine_gate,
                    "scale_conf_x": float(scale_conf[0]),
                    "scale_conf_y": float(scale_conf[1]),
                },
            )
            stem = Path(str(batch["rgb_path"][bi])).stem
            out_path = image_dir / f"{global_image_index:04d}_{stem}.jpg"
            cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)])

            has_zero = any(r["iou_after"] <= 1e-12 for r in pair_metrics)
            if has_zero:
                cv2.imwrite(
                    str(zero_dir / out_path.name),
                    panel,
                    [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)],
                )

            image_rows.append({
                "image_index": global_image_index,
                "pair_key": str(batch.get("pair_key", [""] * bsz)[bi]),
                "rgb_path": str(batch["rgb_path"][bi]),
                "tir_path": str(batch["tir_path"][bi]),
                "num_rgb_boxes": int(len(rgb_boxes)),
                "num_tir_boxes": int(len(tir_boxes)),
                "num_pairs": int(len(pair_metrics)),
                "mean_iou_before": (
                    float(np.mean([r["iou_before"] for r in pair_metrics]))
                    if pair_metrics else float("nan")
                ),
                "mean_iou_translation_final": (
                    float(np.mean([r["iou_translation_final"] for r in pair_metrics]))
                    if pair_metrics else float("nan")
                ),
                "mean_iou_after": (
                    float(np.mean([r["iou_after"] for r in pair_metrics]))
                    if pair_metrics else float("nan")
                ),
                "coarse_confidence_verified": coarse_conf,
                "fine_confidence_gate": fine_gate,
                "scale_confidence_x": float(scale_conf[0]),
                "scale_confidence_y": float(scale_conf[1]),
                "final_sx": float(scale_final[0]),
                "final_sy": float(scale_final[1]),
                "has_final_zero_iou": int(has_zero),
                "boxed_image": str(out_path),
            })

            global_image_index += 1

        if rows:
            pbar.set_postfix(
                raw=f"{np.mean([r['iou_before'] for r in rows]):.3f}",
                trans=f"{np.nanmean([r['iou_translation_final'] for r in rows]):.3f}",
                final=f"{np.mean([r['iou_after'] for r in rows]):.3f}",
                zero=sum(r["final_zero"] for r in rows),
            )

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
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

    stage_names = [
        "iou_before",
        "iou_coarse_raw",
        "iou_coarse_verified",
        "iou_translation_final",
        "iou_scale_moment",
        "iou_scale_quantile",
        "iou_scale_fused",
        "iou_after",
    ]
    stage_summary = {
        name: {
            **safe_stats([r[name] for r in rows]),
            "tail": tail_counts([r[name] for r in rows]),
        }
        for name in stage_names
    }

    improved = sum(r["iou_after"] > r["iou_before"] + 1e-12 for r in rows)
    worsened = sum(r["iou_after"] < r["iou_before"] - 1e-12 for r in rows)
    unchanged = len(rows) - improved - worsened

    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "images": int(global_image_index),
        "paired_instances": int(len(rows)),
        "rgb_semantic_canvas": [w if "w" in locals() else rgb_semantic_imgsz,
                                h if "h" in locals() else rgb_semantic_imgsz],
        "stage_iou": stage_summary,
        "mechanism_effects": {
            "topk_rescued_instances": int(sum(r["topk_rescued"] for r in rows)),
            "fine_worsened_instances": int(sum(r["fine_worsened"] for r in rows)),
            "scale_worsened_instances": int(sum(r["scale_worsened"] for r in rows)),
            "translation_zero_instances": int(sum(r["translation_zero"] for r in rows)),
            "final_zero_instances": int(sum(r["final_zero"] for r in rows)),
            "scale_destroyed_positive_translation_instances": int(
                sum(r["scale_destroyed_positive_translation"] for r in rows)
            ),
        },
        "iou_improved_instances": int(improved),
        "iou_unchanged_instances": int(unchanged),
        "iou_worsened_instances": int(worsened),
        "iou_improved_percent": 100.0 * improved / len(rows) if rows else 0.0,
        "iou_worsened_percent": 100.0 * worsened / len(rows) if rows else 0.0,
        "center_distance_before_px": safe_stats([r["center_distance_before_px"] for r in rows]),
        "center_distance_after_px": safe_stats([r["center_distance_after_px"] for r in rows]),
        "scale_confidence_x": safe_stats([r["scale_confidence_x"] for r in rows]),
        "scale_confidence_y": safe_stats([r["scale_confidence_y"] for r in rows]),
        "fine_confidence_gate": safe_stats([r["fine_confidence_gate"] for r in rows]),
        "outputs": {
            "pair_csv": str(pair_csv),
            "image_csv": str(image_csv),
            "boxed_images": str(image_dir),
            "zero_iou_images": str(zero_dir),
        },
    }

    summary_path = save_dir / "alignment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("ROBUST ALIGNMENT EVALUATION")
    print("=" * 96)
    print(f"Images / pairs               : {global_image_index} / {len(rows)}")
    for name in stage_names:
        s = stage_summary[name]
        print(
            f"{name:<28}: mean={s['mean']:.6f}  med={s['median']:.6f}  "
            f"P10={s['p10']:.6f}  zero={s['tail']['zero']}  <0.1={s['tail']['lt_0p1']}"
        )
    me = summary["mechanism_effects"]
    print("-" * 96)
    print(f"Top-K rescued                : {me['topk_rescued_instances']}")
    print(f"Fine worsened                : {me['fine_worsened_instances']}")
    print(f"Scale worsened               : {me['scale_worsened_instances']}")
    print(f"Translation zero             : {me['translation_zero_instances']}")
    print(f"Final zero                   : {me['final_zero_instances']}")
    print(f"Scale destroyed good trans   : {me['scale_destroyed_positive_translation_instances']}")
    print(f"Pair CSV                     : {pair_csv}")
    print(f"Image CSV                    : {image_csv}")
    print(f"Zero-IoU images              : {zero_dir}")
    print(f"Summary                      : {summary_path}")
    print("=" * 96)

    return summary


def parser():
    p = argparse.ArgumentParser(
        description="Evaluate robust RLSFA/GTGA alignment stage by stage."
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
    p.add_argument("--amp", action="store_true")
    p.add_argument("--jpg-quality", type=int, default=95)
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())
