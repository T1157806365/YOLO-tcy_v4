#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v3
python analyze_gt_scale_range.py \
  --rgb configs/datasets/UAV-CB-RGB.yaml \
  --tir configs/datasets/UAV-CB-TIR.yaml \
  --split test \
  --rgb-imgsz 1920 \
  --rgb-semantic-imgsz 640 \
  --tir-imgsz 640 \
  --scale-min 0.20 \
  --scale-max 1.50 \
  --output runs/scale_analysis/uavcb_test_gt_scale.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,
)


def percentile_report(name, values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        print(f"{name}: EMPTY")
        return

    percentiles = [
        0,
        0.1,
        0.5,
        1,
        2.5,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        97.5,
        99,
        99.5,
        99.9,
        100,
    ]

    p = np.percentile(
        values,
        percentiles,
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(f"Count  : {len(values)}")
    print(f"Mean   : {values.mean():.6f}")
    print(f"Std    : {values.std():.6f}")
    print(f"Min    : {values.min():.6f}")
    print(f"Max    : {values.max():.6f}")

    print("-" * 72)

    for q, v in zip(percentiles, p):
        print(
            f"P{q:<5g}: {v:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rgb",
        required=True,
        help="RGB dataset YAML",
    )

    parser.add_argument(
        "--tir",
        required=True,
        help="TIR dataset YAML",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--rgb-imgsz",
        type=int,
        default=1920,
    )

    parser.add_argument(
        "--rgb-semantic-imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--tir-imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--pair-mode",
        default="relative",
    )

    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.50,
    )

    parser.add_argument(
        "--output",
        default="gt_scale_range_test.csv",
    )

    args = parser.parse_args()

    # =========================================================
    # Dataset
    # =========================================================

    dataset = build_dual_resolution_dataset(
        rgb_yaml=args.rgb,
        tir_yaml=args.tir,
        split=args.split,

        rgb_high_imgsz=args.rgb_imgsz,
        rgb_semantic_imgsz=args.rgb_semantic_imgsz,
        tir_imgsz=args.tir_imgsz,

        pair_mode=args.pair_mode,

        augment=False,
        tir_channels=3,
        strict_pair=True,
    )

    rows = []

    sx_all = []
    sy_all = []
    s_mean_all = []
    s_geo_all = []

    unmatched_images = 0
    invalid_boxes = 0

    # =========================================================
    # Iterate samples
    # =========================================================

    for index in range(len(dataset)):

        sample = dataset[index]

        rgb_boxes = (
            sample["rgb_semantic_bboxes"]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1, 4)
        )

        tir_boxes = (
            sample["tir_bboxes"]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1, 4)
        )

        rgb_path = str(
            sample.get(
                "rgb_path",
                index,
            )
        )

        tir_path = str(
            sample.get(
                "tir_path",
                index,
            )
        )

        # UAV-CB usually has corresponding RGB/TIR objects.
        # For multi-object cases this assumes label order is paired.
        n = min(
            len(rgb_boxes),
            len(tir_boxes),
        )

        if (
            len(rgb_boxes)
            != len(tir_boxes)
        ):
            unmatched_images += 1

        for j in range(n):

            rgb_box = rgb_boxes[j]
            tir_box = tir_boxes[j]

            # normalized xywh
            rgb_w = float(rgb_box[2])
            rgb_h = float(rgb_box[3])

            tir_w = float(tir_box[2])
            tir_h = float(tir_box[3])

            if (
                rgb_w <= 1e-9
                or rgb_h <= 1e-9
                or tir_w <= 1e-9
                or tir_h <= 1e-9
            ):
                invalid_boxes += 1
                continue

            # -------------------------------------------------
            # RGB reference -> TIR source
            # q_T = S p_R + t
            # -------------------------------------------------

            sx = tir_w / rgb_w
            sy = tir_h / rgb_h

            s_mean = 0.5 * (
                sx + sy
            )

            s_geo = math.sqrt(
                sx * sy
            )

            current_ok_x = (
                args.scale_min
                <= sx
                <= args.scale_max
            )

            current_ok_y = (
                args.scale_min
                <= sy
                <= args.scale_max
            )

            current_ok_both = (
                current_ok_x
                and current_ok_y
            )

            sx_all.append(sx)
            sy_all.append(sy)
            s_mean_all.append(s_mean)
            s_geo_all.append(s_geo)

            rows.append(
                {
                    "index": index,
                    "object": j,

                    "rgb_path": rgb_path,
                    "tir_path": tir_path,

                    "rgb_cx": float(
                        rgb_box[0]
                    ),
                    "rgb_cy": float(
                        rgb_box[1]
                    ),
                    "rgb_w": rgb_w,
                    "rgb_h": rgb_h,

                    "tir_cx": float(
                        tir_box[0]
                    ),
                    "tir_cy": float(
                        tir_box[1]
                    ),
                    "tir_w": tir_w,
                    "tir_h": tir_h,

                    "scale_x": sx,
                    "scale_y": sy,

                    "scale_mean": s_mean,
                    "scale_geo": s_geo,

                    "inside_current_x":
                        int(current_ok_x),

                    "inside_current_y":
                        int(current_ok_y),

                    "inside_current_both":
                        int(current_ok_both),
                }
            )

    # =========================================================
    # Save CSV
    # =========================================================

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if rows:
        with output.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(rows)

    # =========================================================
    # Statistics
    # =========================================================

    sx_all = np.asarray(
        sx_all,
        dtype=np.float64,
    )

    sy_all = np.asarray(
        sy_all,
        dtype=np.float64,
    )

    s_mean_all = np.asarray(
        s_mean_all,
        dtype=np.float64,
    )

    s_geo_all = np.asarray(
        s_geo_all,
        dtype=np.float64,
    )

    print()
    print("=" * 72)
    print("GT RGB -> TIR SCALE ANALYSIS")
    print("=" * 72)

    print(
        f"Split                  : {args.split}"
    )

    print(
        f"Images                 : {len(dataset)}"
    )

    print(
        f"Valid paired boxes     : {len(sx_all)}"
    )

    print(
        f"Unmatched label images : {unmatched_images}"
    )

    print(
        f"Invalid boxes          : {invalid_boxes}"
    )

    print(
        f"Current scale range    : "
        f"[{args.scale_min:.4f}, "
        f"{args.scale_max:.4f}]"
    )

    percentile_report(
        "scale_x = TIR width / RGB width",
        sx_all,
    )

    percentile_report(
        "scale_y = TIR height / RGB height",
        sy_all,
    )

    percentile_report(
        "scale_mean = (sx + sy) / 2",
        s_mean_all,
    )

    percentile_report(
        "scale_geo = sqrt(sx * sy)",
        s_geo_all,
    )

    # =========================================================
    # Current range coverage
    # =========================================================

    if len(sx_all) > 0:

        x_low = (
            sx_all
            < args.scale_min
        )

        x_high = (
            sx_all
            > args.scale_max
        )

        y_low = (
            sy_all
            < args.scale_min
        )

        y_high = (
            sy_all
            > args.scale_max
        )

        both_inside = (
            (sx_all >= args.scale_min)
            &
            (sx_all <= args.scale_max)
            &
            (sy_all >= args.scale_min)
            &
            (sy_all <= args.scale_max)
        )

        print()
        print("=" * 72)
        print("CURRENT RANGE COVERAGE")
        print("=" * 72)

        print(
            f"scale_x < min : "
            f"{x_low.sum():4d} / {len(sx_all)} "
            f"({x_low.mean() * 100:.2f}%)"
        )

        print(
            f"scale_x > max : "
            f"{x_high.sum():4d} / {len(sx_all)} "
            f"({x_high.mean() * 100:.2f}%)"
        )

        print(
            f"scale_y < min : "
            f"{y_low.sum():4d} / {len(sy_all)} "
            f"({y_low.mean() * 100:.2f}%)"
        )

        print(
            f"scale_y > max : "
            f"{y_high.sum():4d} / {len(sy_all)} "
            f"({y_high.mean() * 100:.2f}%)"
        )

        print(
            f"Both inside   : "
            f"{both_inside.sum():4d} / {len(sx_all)} "
            f"({both_inside.mean() * 100:.2f}%)"
        )

        # -----------------------------------------------------
        # Suggested ranges
        # -----------------------------------------------------

        sx_p1, sx_p99 = np.percentile(
            sx_all,
            [1, 99],
        )

        sy_p1, sy_p99 = np.percentile(
            sy_all,
            [1, 99],
        )

        sx_p05, sx_p995 = np.percentile(
            sx_all,
            [0.5, 99.5],
        )

        sy_p05, sy_p995 = np.percentile(
            sy_all,
            [0.5, 99.5],
        )

        recommended_p1_min = min(
            sx_p1,
            sy_p1,
        )

        recommended_p1_max = max(
            sx_p99,
            sy_p99,
        )

        recommended_p05_min = min(
            sx_p05,
            sy_p05,
        )

        recommended_p05_max = max(
            sx_p995,
            sy_p995,
        )

        absolute_min = min(
            sx_all.min(),
            sy_all.min(),
        )

        absolute_max = max(
            sx_all.max(),
            sy_all.max(),
        )

        print()
        print("=" * 72)
        print("SUGGESTED SCALE RANGE")
        print("=" * 72)

        print(
            "Absolute GT range : "
            f"[{absolute_min:.4f}, "
            f"{absolute_max:.4f}]"
        )

        print(
            "P0.5-P99.5 range  : "
            f"[{recommended_p05_min:.4f}, "
            f"{recommended_p05_max:.4f}]"
        )

        print(
            "P1-P99 range      : "
            f"[{recommended_p1_min:.4f}, "
            f"{recommended_p1_max:.4f}]"
        )

        print()
        print("YAML candidate:")

        print(
            f"scale_min: "
            f"{recommended_p1_min:.3f}"
        )

        print(
            f"scale_max: "
            f"{recommended_p1_max:.3f}"
        )

    print()
    print(
        f"CSV saved to: "
        f"{output.resolve()}"
    )


if __name__ == "__main__":
    main()