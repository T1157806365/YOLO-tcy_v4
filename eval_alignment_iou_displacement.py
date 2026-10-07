#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v3
python eval_alignment_iou_displacement.py \
  --weights /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v3/runs/rlsfa_gtga_bhlr_coarsefine/yolo26m_UAVCB_TIR2RGB_RLSFA-CoarseFine_GTGA-Scale_ADD_rgb1920_sem640_tir640_seed0/weights/best.pt \
  --split test \
  --device 5 \
  --batch 4 \
  --workers 4 \
  --save-dir runs/alignment_iou_UAV-CB

conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v3
python eval_alignment_iou_displacement.py \
  --weights /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v3/runs/rlsfa_gtga_gchrdr/yolo26m_Anti-UAV_RLSFA-Coarse_GTGA-Scale_CenteredPivot_BHLRv2-P3_rgb1920_sem640_tir640_seed0_v5_device0/weights/best.pt \
  --rgb /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v3/configs/datasets/Anti-UAV-RGB.yaml \
  --tir /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v3/configs/datasets/Anti-UAV-TIR.yaml \
  --split test \
  --rgb-imgsz 1920 \
  --rgb-semantic-imgsz 640 \
  --tir-imgsz 640 \
  --device 5 \
  --batch 4 \
  --workers 4 \
  --save-dir runs/alignment_iou_Anti-UAV
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


# -----------------------------------------------------------------------------
# Box / geometry helpers
# -----------------------------------------------------------------------------

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
    """Axis-aligned IoU. Boxes are NOT clipped to the image before IoU."""
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]

    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def transform_tir_box_to_rgb(
    tir_box_xywhn: np.ndarray,
    sampling_scale: np.ndarray,
    sampling_translation: np.ndarray,
) -> np.ndarray:
    """
    Map source TIR box q to RGB/output box p for inverse sampling geometry:
        q = S*p + t  =>  p = (q - t) / S

    Input/output boxes are normalized xywh on the 640 canvas.
    """
    b = np.asarray(tir_box_xywhn, dtype=np.float32).copy()
    s = np.asarray(sampling_scale, dtype=np.float32).reshape(2)
    t = np.asarray(sampling_translation, dtype=np.float32).reshape(2)
    s = np.maximum(s, 1e-8)

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
    """Vector from src center to dst center, in current canvas pixels."""
    dx = (float(dst_xywhn[0]) - float(src_xywhn[0])) * float(width)
    dy = (float(dst_xywhn[1]) - float(src_xywhn[1])) * float(height)
    return dx, dy, math.hypot(dx, dy)


def pair_boxes_by_center(
    rgb_boxes: np.ndarray,
    tir_boxes: np.ndarray,
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    """
    Pair boxes inside one image.

    UAV-CB used here is normally one UAV per image, so n=1 is exact.
    For unexpected multi-object samples, use deterministic greedy nearest-center
    pairing so the script still completes instead of silently using array order.
    """
    nr = int(len(rgb_boxes))
    nt = int(len(tir_boxes))
    if nr == 0 or nt == 0:
        return []
    if nr == 1 and nt == 1:
        return [(0, 0)]

    candidates: List[Tuple[float, int, int]] = []
    for ri, rb in enumerate(rgb_boxes):
        for ti, tb in enumerate(tir_boxes):
            dx = (float(rb[0]) - float(tb[0])) * width
            dy = (float(rb[1]) - float(tb[1])) * height
            candidates.append((math.hypot(dx, dy), ri, ti))
    candidates.sort(key=lambda x: x[0])

    used_r, used_t = set(), set()
    pairs: List[Tuple[int, int]] = []
    for _, ri, ti in candidates:
        if ri in used_r or ti in used_t:
            continue
        used_r.add(ri)
        used_t.add(ti)
        pairs.append((ri, ti))
        if len(pairs) == min(nr, nt):
            break
    return pairs


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

RGB_GT = (0, 255, 0)          # green
RAW_TIR = (0, 165, 255)       # orange
ALIGNED_TIR = (255, 255, 0)   # cyan in BGR
ARROW = (255, 0, 255)         # magenta
TEXT = (255, 255, 255)


def tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
    """CHW RGB float [0,1] -> uint8 BGR."""
    x = img.detach().float().cpu().clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def clip_xyxy_for_draw(box: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = int(round(np.clip(x1, 0, width - 1)))
    y1 = int(round(np.clip(y1, 0, height - 1)))
    x2 = int(round(np.clip(x2, 0, width - 1)))
    y2 = int(round(np.clip(y2, 0, height - 1)))
    return x1, y1, x2, y2


def draw_box(img: np.ndarray, box: np.ndarray, color, label: str, thickness: int = 2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_xyxy_for_draw(box, w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        cv2.putText(
            img,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )


def put_lines(img: np.ndarray, lines: Sequence[str], x: int = 10, y: int = 24):
    for i, line in enumerate(lines):
        yy = y + i * 22
        # dark backing for readability
        cv2.putText(
            img, line, (x + 1, yy + 1), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, TEXT, 1, cv2.LINE_AA,
        )


def make_before_after_panel(
    rgb_bgr: np.ndarray,
    rgb_boxes: np.ndarray,
    raw_tir_boxes: np.ndarray,
    aligned_tir_boxes: np.ndarray,
    pair_metrics: List[Dict],
    scale_xy: np.ndarray,
    reliability: float,
) -> np.ndarray:
    h, w = rgb_bgr.shape[:2]
    before = rgb_bgr.copy()
    after = rgb_bgr.copy()

    for k, m in enumerate(pair_metrics):
        ri = int(m["rgb_index"])
        ti = int(m["tir_index"])

        rb_xyxy = xywhn_to_xyxy(rgb_boxes[ri], w, h)
        tb_xyxy = xywhn_to_xyxy(raw_tir_boxes[ti], w, h)
        ab_xyxy = xywhn_to_xyxy(aligned_tir_boxes[ti], w, h)

        draw_box(before, rb_xyxy, RGB_GT, f"RGB GT #{k}")
        draw_box(before, tb_xyxy, RAW_TIR, f"Raw TIR #{k}")

        draw_box(after, rb_xyxy, RGB_GT, f"RGB GT #{k}")
        draw_box(after, ab_xyxy, ALIGNED_TIR, f"Aligned TIR #{k}")

        raw_c = (
            int(round(float(raw_tir_boxes[ti][0]) * w)),
            int(round(float(raw_tir_boxes[ti][1]) * h)),
        )
        aligned_c = (
            int(round(float(aligned_tir_boxes[ti][0]) * w)),
            int(round(float(aligned_tir_boxes[ti][1]) * h)),
        )
        cv2.arrowedLine(after, raw_c, aligned_c, ARROW, 2, cv2.LINE_AA, tipLength=0.08)

    if pair_metrics:
        mean_before = float(np.mean([m["iou_before"] for m in pair_metrics]))
        mean_after = float(np.mean([m["iou_after"] for m in pair_metrics]))
        mean_db = float(np.mean([m["center_distance_before_px"] for m in pair_metrics]))
        mean_da = float(np.mean([m["center_distance_after_px"] for m in pair_metrics]))
        mean_move = float(np.mean([m["applied_distance_px"] for m in pair_metrics]))
    else:
        mean_before = mean_after = mean_db = mean_da = mean_move = float("nan")

    put_lines(
        before,
        [
            "BEFORE ALIGNMENT",
            f"mean IoU = {mean_before:.4f}",
            f"center dist = {mean_db:.2f}px",
            "green=RGB GT; orange=raw TIR GT",
        ],
    )
    put_lines(
        after,
        [
            "AFTER ALIGNMENT",
            f"mean IoU = {mean_after:.4f}  gain={mean_after - mean_before:+.4f}",
            f"residual center dist = {mean_da:.2f}px",
            f"applied move = {mean_move:.2f}px",
            f"S=({scale_xy[0]:.3f},{scale_xy[1]:.3f})  rho={reliability:.3f}",
            "green=RGB GT; cyan=aligned TIR GT; magenta=move",
        ],
    )

    divider = np.full((h, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([before, divider, after], axis=1)


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def safe_stats(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def extract_batch_scalar(x, bi: int, default: float = 1.0) -> float:
    if x is None:
        return float(default)
    if torch.is_tensor(x):
        y = x.detach().float().cpu()
        if y.ndim == 0:
            return float(y.item())
        return float(y[bi].reshape(-1)[0].item())
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr.item())
    return float(arr[bi].reshape(-1)[0])


def extract_batch_vec2(x, bi: int, default=(1.0, 1.0)) -> np.ndarray:
    if x is None:
        return np.asarray(default, dtype=np.float32)
    if torch.is_tensor(x):
        y = x.detach().float().cpu()[bi].reshape(-1).numpy()
    else:
        y = np.asarray(x)[bi].reshape(-1)
    if y.size < 2:
        raise ValueError(f"Expected a 2-vector, got shape/value {y}")
    return y[:2].astype(np.float32)


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------

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
        raise ValueError("RGB/TIR dataset YAML could not be resolved from CLI or checkpoint config.")

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
        else ckpt_path.parent.parent / f"alignment_geometry_{args.split}"
    )
    image_dir = save_dir / "boxed_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    image_rows: List[Dict] = []
    global_image_index = 0

    pbar = tqdm(loader, desc=f"alignment {args.split}")
    for batch in pbar:
        batch = preprocess_batch(batch, device)

        # Use the same valid-content masks as the model's training path.
        if hasattr(model, "build_alignment_valid_masks_from_batch"):
            rgb_valid_mask, tir_valid_mask = model.build_alignment_valid_masks_from_batch(batch)
        else:
            rgb_valid_mask, tir_valid_mask = None, None

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
        scale_all = info.get("sampling_scale")
        trans_all = info.get("sampling_translation")
        if scale_all is None or trans_all is None:
            raise KeyError(
                "alignment_info must contain 'sampling_scale' and 'sampling_translation'. "
                "These must be the actual geometry used by the final ONE warp."
            )

        reliability_all = info.get("alignment_reliability", None)
        coarse_offset_all = info.get("coarse_offset", None)

        bsz = int(batch["rgb_semantic_img"].shape[0])
        h = int(batch["rgb_semantic_img"].shape[2])
        w = int(batch["rgb_semantic_img"].shape[3])

        for bi in range(bsz):
            rmask = batch["rgb_batch_idx"] == bi
            tmask = batch["tir_batch_idx"] == bi

            rgb_boxes = batch["rgb_semantic_bboxes"][rmask].detach().float().cpu().numpy()
            tir_boxes = batch["tir_bboxes"][tmask].detach().float().cpu().numpy()

            scale_xy = extract_batch_vec2(scale_all, bi, default=(1.0, 1.0))
            trans_xy = extract_batch_vec2(trans_all, bi, default=(0.0, 0.0))
            rho = extract_batch_scalar(reliability_all, bi, default=1.0)

            if coarse_offset_all is not None:
                coarse_xy = extract_batch_vec2(coarse_offset_all, bi, default=(0.0, 0.0))
                coarse_dx_p3 = float(coarse_xy[0])
                coarse_dy_p3 = float(coarse_xy[1])
                coarse_dist_p3 = float(math.hypot(coarse_dx_p3, coarse_dy_p3))
            else:
                coarse_dx_p3 = coarse_dy_p3 = coarse_dist_p3 = float("nan")

            aligned_tir_boxes = np.stack(
                [transform_tir_box_to_rgb(tb, scale_xy, trans_xy) for tb in tir_boxes],
                axis=0,
            ) if len(tir_boxes) else np.zeros((0, 4), dtype=np.float32)

            pairs = pair_boxes_by_center(rgb_boxes, tir_boxes, w, h)
            pair_metrics: List[Dict] = []

            for pair_id, (ri, ti) in enumerate(pairs):
                rb = rgb_boxes[ri]
                tb = tir_boxes[ti]
                ab = aligned_tir_boxes[ti]

                rb_xyxy = xywhn_to_xyxy(rb, w, h)
                tb_xyxy = xywhn_to_xyxy(tb, w, h)
                ab_xyxy = xywhn_to_xyxy(ab, w, h)

                iou_before = bbox_iou_xyxy(rb_xyxy, tb_xyxy)
                iou_after = bbox_iou_xyxy(rb_xyxy, ab_xyxy)

                # TIR -> RGB: how far it SHOULD move before alignment.
                gt_dx, gt_dy, dist_before = center_delta_px(tb, rb, w, h)

                # Raw TIR -> aligned TIR: what the model actually applied.
                applied_dx, applied_dy, applied_dist = center_delta_px(tb, ab, w, h)

                # Aligned TIR -> RGB: remaining center error after alignment.
                residual_dx, residual_dy, dist_after = center_delta_px(ab, rb, w, h)

                row = {
                    "image_index": global_image_index,
                    "pair_id": pair_id,
                    "pair_key": str(batch.get("pair_key", [""] * bsz)[bi]),
                    "rgb_path": str(batch["rgb_path"][bi]),
                    "tir_path": str(batch["tir_path"][bi]),
                    "rgb_index": ri,
                    "tir_index": ti,
                    "iou_before": iou_before,
                    "iou_after": iou_after,
                    "iou_gain": iou_after - iou_before,
                    "gt_dx_before_px": gt_dx,
                    "gt_dy_before_px": gt_dy,
                    "center_distance_before_px": dist_before,
                    "applied_dx_px": applied_dx,
                    "applied_dy_px": applied_dy,
                    "applied_distance_px": applied_dist,
                    "residual_dx_after_px": residual_dx,
                    "residual_dy_after_px": residual_dy,
                    "center_distance_after_px": dist_after,
                    "center_distance_reduction_px": dist_before - dist_after,
                    "sx": float(scale_xy[0]),
                    "sy": float(scale_xy[1]),
                    "sampling_tx_norm": float(trans_xy[0]),
                    "sampling_ty_norm": float(trans_xy[1]),
                    "alignment_reliability": rho,
                    "coarse_dx_p3": coarse_dx_p3,
                    "coarse_dy_p3": coarse_dy_p3,
                    "coarse_distance_p3": coarse_dist_p3,
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

            # Save every test image, including unexpected no-box samples.
            rgb_bgr = tensor_to_bgr(batch["rgb_semantic_img"][bi])
            panel = make_before_after_panel(
                rgb_bgr=rgb_bgr,
                rgb_boxes=rgb_boxes,
                raw_tir_boxes=tir_boxes,
                aligned_tir_boxes=aligned_tir_boxes,
                pair_metrics=pair_metrics,
                scale_xy=scale_xy,
                reliability=rho,
            )

            stem = Path(str(batch["rgb_path"][bi])).stem
            out_name = f"{global_image_index:04d}_{stem}.jpg"
            out_path = image_dir / out_name
            cv2.imwrite(str(out_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])

            if pair_metrics:
                image_rows.append({
                    "image_index": global_image_index,
                    "pair_key": str(batch.get("pair_key", [""] * bsz)[bi]),
                    "rgb_path": str(batch["rgb_path"][bi]),
                    "tir_path": str(batch["tir_path"][bi]),
                    "num_rgb_boxes": int(len(rgb_boxes)),
                    "num_tir_boxes": int(len(tir_boxes)),
                    "num_pairs": int(len(pair_metrics)),
                    "mean_iou_before": float(np.mean([x["iou_before"] for x in pair_metrics])),
                    "mean_iou_after": float(np.mean([x["iou_after"] for x in pair_metrics])),
                    "mean_iou_gain": float(np.mean([x["iou_gain"] for x in pair_metrics])),
                    "mean_center_distance_before_px": float(np.mean([x["center_distance_before_px"] for x in pair_metrics])),
                    "mean_center_distance_after_px": float(np.mean([x["center_distance_after_px"] for x in pair_metrics])),
                    "mean_applied_distance_px": float(np.mean([x["applied_distance_px"] for x in pair_metrics])),
                    "sx": float(scale_xy[0]),
                    "sy": float(scale_xy[1]),
                    "alignment_reliability": rho,
                    "boxed_image": str(out_path),
                })
            else:
                image_rows.append({
                    "image_index": global_image_index,
                    "pair_key": str(batch.get("pair_key", [""] * bsz)[bi]),
                    "rgb_path": str(batch["rgb_path"][bi]),
                    "tir_path": str(batch["tir_path"][bi]),
                    "num_rgb_boxes": int(len(rgb_boxes)),
                    "num_tir_boxes": int(len(tir_boxes)),
                    "num_pairs": 0,
                    "mean_iou_before": float("nan"),
                    "mean_iou_after": float("nan"),
                    "mean_iou_gain": float("nan"),
                    "mean_center_distance_before_px": float("nan"),
                    "mean_center_distance_after_px": float("nan"),
                    "mean_applied_distance_px": float("nan"),
                    "sx": float(scale_xy[0]),
                    "sy": float(scale_xy[1]),
                    "alignment_reliability": rho,
                    "boxed_image": str(out_path),
                })

            global_image_index += 1

        if rows:
            pbar.set_postfix(
                iou0=f"{np.mean([r['iou_before'] for r in rows]):.3f}",
                iou1=f"{np.mean([r['iou_after'] for r in rows]):.3f}",
            )

    # ------------------------------------------------------------------
    # Export tables
    # ------------------------------------------------------------------
    csv_path = save_dir / "alignment_pairs.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    image_csv_path = save_dir / "alignment_images.csv"
    if image_rows:
        with image_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(image_rows)

    iou_before = [r["iou_before"] for r in rows]
    iou_after = [r["iou_after"] for r in rows]
    iou_gain = [r["iou_gain"] for r in rows]
    db = [r["center_distance_before_px"] for r in rows]
    da = [r["center_distance_after_px"] for r in rows]
    dm = [r["applied_distance_px"] for r in rows]

    improved = int(sum(1 for r in rows if r["iou_after"] > r["iou_before"] + 1e-12))
    unchanged = int(sum(1 for r in rows if abs(r["iou_after"] - r["iou_before"]) <= 1e-12))
    worsened = int(len(rows) - improved - unchanged)

    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "images": int(global_image_index),
        "paired_instances": int(len(rows)),
        "rgb_semantic_canvas": [w if 'w' in locals() else rgb_semantic_imgsz,
                                h if 'h' in locals() else rgb_semantic_imgsz],
        "iou_before": safe_stats(iou_before),
        "iou_after": safe_stats(iou_after),
        "iou_gain": safe_stats(iou_gain),
        "center_distance_before_px": safe_stats(db),
        "center_distance_after_px": safe_stats(da),
        "applied_distance_px": safe_stats(dm),
        "iou_improved_instances": improved,
        "iou_unchanged_instances": unchanged,
        "iou_worsened_instances": worsened,
        "iou_improved_percent": (100.0 * improved / len(rows)) if rows else 0.0,
        "iou_worsened_percent": (100.0 * worsened / len(rows)) if rows else 0.0,
        "outputs": {
            "pair_csv": str(csv_path),
            "image_csv": str(image_csv_path),
            "boxed_images": str(image_dir),
        },
    }

    summary_path = save_dir / "alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Alignment geometry evaluation completed")
    print("=" * 80)
    print(f"Images                         : {global_image_index}")
    print(f"Paired instances               : {len(rows)}")
    if rows:
        print(f"Mean IoU before                : {summary['iou_before']['mean']:.6f}")
        print(f"Mean IoU after                 : {summary['iou_after']['mean']:.6f}")
        print(f"Mean IoU gain                  : {summary['iou_gain']['mean']:+.6f}")
        print(f"Mean center distance before(px): {summary['center_distance_before_px']['mean']:.3f}")
        print(f"Mean center distance after(px) : {summary['center_distance_after_px']['mean']:.3f}")
        print(f"Mean applied movement(px)      : {summary['applied_distance_px']['mean']:.3f}")
        print(f"IoU improved                   : {improved}/{len(rows)} ({summary['iou_improved_percent']:.2f}%)")
        print(f"IoU worsened                   : {worsened}/{len(rows)} ({summary['iou_worsened_percent']:.2f}%)")
    print(f"Pair CSV                       : {csv_path}")
    print(f"Image CSV                      : {image_csv_path}")
    print(f"Boxed images                   : {image_dir}")
    print(f"Summary                        : {summary_path}")
    print("=" * 80)

    return summary


def parser():
    p = argparse.ArgumentParser(
        description="Evaluate alignment IoU/displacement and export boxed images for the whole split."
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
    p.add_argument("--amp", action="store_true", help="Use CUDA autocast. Default is FP32 for geometry diagnostics.")
    p.add_argument("--jpg-quality", type=int, default=95)
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())
