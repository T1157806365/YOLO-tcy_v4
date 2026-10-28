#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AGDH-DRG v2 same-checkpoint ablation suite for YOLO-tcy_v4.

No retraining, no checkpoint edits.  Evaluates one checkpoint with the exact
project evaluator and captures AP50, AP55, ..., AP95 from DetMetrics.

Examples:
  python eval_agdh_drg_v2_suite.py --weights runs/.../best.pt --split val --device 3
  python eval_agdh_drg_v2_suite.py --weights runs/.../best.pt --split test --device 3 \
      --batch 4 --workers 4 --cases off cls reg clsweak full no_q
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,
    build_dual_resolution_dataloader,
)
from engine import evaluator as ev
from engine.runtime import select_device
from models.gtga_bhlr_factory import load_model_from_checkpoint


# All settings are applied at inference time to the SAME loaded checkpoint.
# (case_name, classification_gain, regression_gain, reliability_enabled)
CASES = {
    "off": (0.0, 0.0, True),
    "cls": (0.5, 0.0, True),
    "reg": (0.0, 0.05, True),
    "clsweak": (0.2, 0.05, True),
    "full": (0.5, 0.05, True),
    "no_q": (0.5, 0.05, False),
}
THRESHOLDS = [f"AP{int(round(v * 100)):02d}" for v in np.linspace(0.50, 0.95, 10)]


def get_args():
    p = argparse.ArgumentParser(
        description="AGDH-DRG v2 same-checkpoint cls/reg/reliability ablation"
    )
    p.add_argument("--weights", required=True, help="AGDH-DRG v2 best.pt")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--device", default="3")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cases", nargs="+", choices=tuple(CASES),
                   default=["off", "cls", "reg", "clsweak", "full", "no_q"])
    p.add_argument("--save-dir", default=None)
    p.add_argument("--rgb", default=None)
    p.add_argument("--tir", default=None)
    p.add_argument("--rgb-imgsz", type=int, default=None)
    p.add_argument("--rgb-semantic-imgsz", type=int, default=None)
    p.add_argument("--tir-imgsz", type=int, default=None)
    p.add_argument("--pair-mode", default=None,
                   choices=["relative", "filename", "stem"])
    p.add_argument("--check-only", action="store_true",
                   help="Validate checkpoint and settings without loading dataset")
    return p.parse_args()


def validate_model(model):
    if not hasattr(model, "agdh_drg"):
        raise TypeError("This is NOT an AGDH-DRG v2 checkpoint: model.agdh_drg is missing")
    drg = model.agdh_drg
    for key in ("cls_gain", "reg_gain", "use_reliability", "prior_dropout"):
        if not hasattr(drg, key):
            raise RuntimeError(f"AGDH-DRG v2 missing runtime control: {key}")
    if not hasattr(drg, "forward_detect"):
        raise RuntimeError("Loaded AGDH-DRG v2 module has no forward_detect")
    if hasattr(drg, "check_detect"):
        drg.check_detect(model.model[-1])
    return drg


def evaluate_with_ap_curve(model, loader, device, args):
    # Reuse the ACTUAL project's validate_metrics unchanged.  Only replace its
    # DetMetrics constructor temporarily so we can read the full per-IoU AP
    # vector after process(), which the stock result otherwise does not expose.
    original_cls = ev.DetMetrics
    captured = {}

    class CaptureDetMetrics(original_cls):
        def process(self, *a, **kw):
            result = super().process(*a, **kw)
            box = getattr(self, "box", None)
            raw = getattr(box, "all_ap", None)
            if callable(raw):
                raw = raw()
            if raw is not None:
                ap = np.asarray(raw, dtype=np.float64)
                if ap.ndim == 2 and ap.shape[1] == 10 and ap.shape[0] > 0:
                    captured["ap_per_iou"] = ap.mean(axis=0).tolist()
                    captured["per_class_count"] = int(ap.shape[0])
            return result

    ev.DetMetrics = CaptureDetMetrics
    try:
        result = ev.validate_metrics(
            model=model,
            loader=loader,
            device=device,
            amp=not args.no_amp,
            conf_thres=args.conf,
            iou_thres=args.iou,
            max_det=args.max_det,
            preview_num=0,
        )
    finally:
        ev.DetMetrics = original_cls
    return result, captured.get("ap_per_iou"), captured.get("per_class_count")


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def plot_curves(rows, path):
    if not any("ap_per_iou" in x for x in rows):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(50, 100, 5)
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in rows:
            if "ap_per_iou" in r:
                ax.plot(x, np.asarray(r["ap_per_iou"]) * 100.,
                        marker="o", label=r["case"])
        ax.set_xlabel("IoU threshold")
        ax.set_ylabel("Average Precision (%)")
        ax.set_title("AGDH-DRG v2: AP vs. IoU (same checkpoint)")
        ax.set_xticks(x)
        ax.grid(True, alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        print(f"[saved] {path}")
    except ImportError:
        print("[note] matplotlib not installed; CSV still contains AP per threshold")


def main():
    args = get_args()
    if args.batch < 1 or args.workers < 0:
        raise ValueError("--batch must be >=1 and --workers must be >=0")
    ckpt_path = Path(args.weights).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    device = select_device(args.device)
    model, _, cfg = load_model_from_checkpoint(
        str(ckpt_path), device=device, fusion_override="auto", strict=True,
        verbose=False,
    )
    drg = validate_model(model)
    model.eval()
    # For evaluation, dropout must not activate. Explicitly record this.
    initial_gains = {
        "cls_gain": float(drg.cls_gain),
        "reg_gain": float(drg.reg_gain),
        "use_reliability": bool(drg.use_reliability),
    }
    print("=" * 75)
    print("AGDH-DRG v2 same-checkpoint ablation")
    print("Checkpoint :", ckpt_path)
    print("Device     :", device)
    print("Split      :", args.split)
    print("Original   :", initial_gains)
    print("Cases      :", ", ".join(args.cases))
    print("State dict : unchanged; only runtime gains/Q are overridden")
    print("=" * 75)
    if args.check_only:
        print("[OK] model loaded strictly; AGDH-DRG v2 controls found")
        return

    data_cfg = cfg.get("data", {}) or {}
    rgb_yaml = args.rgb or data_cfg.get("rgb")
    tir_yaml = args.tir or data_cfg.get("tir")
    if not rgb_yaml or not tir_yaml:
        raise ValueError("Missing RGB/TIR dataset YAML")
    rgb_high = int(args.rgb_imgsz or data_cfg.get("rgb_imgsz", 1920))
    rgb_sem = int(args.rgb_semantic_imgsz or data_cfg.get("rgb_semantic_imgsz", 640))
    tir_size = int(args.tir_imgsz or data_cfg.get("tir_imgsz", 640))
    pair_mode = args.pair_mode or data_cfg.get("pair_mode", "relative")
    dataset = build_dual_resolution_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=args.split,
        rgb_high_imgsz=rgb_high,
        rgb_semantic_imgsz=rgb_sem,
        tir_imgsz=tir_size,
        pair_mode=pair_mode,
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )
    loader = build_dual_resolution_dataloader(
        dataset,
        batch_size=args.batch,
        workers=args.workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    if args.save_dir is None:
        save_dir = ckpt_path.parent.parent / f"agdh_drg_v2_ablation_{args.split}"
    else:
        save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for idx, case in enumerate(dict.fromkeys(args.cases), 1):
            cg, rg, use_q = CASES[case]
            drg.cls_gain = float(cg)
            drg.reg_gain = float(rg)
            drg.use_reliability = bool(use_q)
            model.eval()
            assert model.agdh_drg is drg
            print("\n" + "=" * 75)
            print(f"[{idx}/{len(args.cases)}] {case}: cls_gain={cg}, "
                  f"reg_gain={rg}, reliability={use_q}", flush=True)
            print("=" * 75)
            stats, curve, ncls = evaluate_with_ap_curve(model, loader, device, args)
            row = {
                "case": case,
                "cls_gain": cg,
                "reg_gain": rg,
                "use_reliability": use_q,
                "precision": float(stats["precision"]),
                "recall": float(stats["recall"]),
                "map50": float(stats["map50"]),
                "map75": float(stats["map75"]),
                "map5095": float(stats["map5095"]),
                "images": int(stats["images"]),
                "instances": int(stats["instances"]),
            }
            if curve is not None:
                row["ap_per_iou"] = curve
                row["ap_curve_num_classes"] = ncls
                for name, value in zip(THRESHOLDS, curve):
                    row[name] = float(value)
                if abs(float(np.mean(curve)) - row["map5095"]) > 1e-4:
                    print("[note] averaged class AP curve differs from map5095; "
                          "inspect dataset class handling")
            else:
                print("[note] per-IoU AP vector unavailable in this Ultralytics version")
            rows.append(row)
            (save_dir / f"{case}.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[RESULT] {case:8s} P={row['precision']*100:.3f}% "
                  f"R={row['recall']*100:.3f}% AP50={row['map50']*100:.3f}% "
                  f"AP75={row['map75']*100:.3f}% "
                  f"AP50:95={row['map5095']*100:.3f}%", flush=True)
    finally:
        drg.cls_gain = initial_gains["cls_gain"]
        drg.reg_gain = initial_gains["reg_gain"]
        drg.use_reliability = initial_gains["use_reliability"]

    fields = ["case", "cls_gain", "reg_gain", "use_reliability",
              "precision", "recall", "map50", "map75", "map5095",
              "images", "instances"] + THRESHOLDS
    write_csv(save_dir / "summary.csv", rows, fields)
    if any("ap_per_iou" in x for x in rows):
        write_csv(save_dir / "ap_by_iou.csv", rows,
                  ["case", "cls_gain", "reg_gain", "use_reliability"] + THRESHOLDS)
        plot_curves(rows, save_dir / "ap_by_iou.png")
    (save_dir / "summary.json").write_text(
        json.dumps({
            "checkpoint": str(ckpt_path), "split": args.split,
            "dataset": {"rgb": rgb_yaml, "tir": tir_yaml,
                        "rgb_imgsz": rgb_high, "rgb_semantic_imgsz": rgb_sem,
                        "tir_imgsz": tir_size, "pair_mode": pair_mode},
            "evaluation": {"conf": args.conf, "iou": args.iou,
                           "max_det": args.max_det, "amp": not args.no_amp},
            "cases": rows,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + "=" * 75)
    print("FINAL SUMMARY (%, higher is better)")
    print(f"{'case':10s} {'P':>8s} {'R':>8s} {'AP50':>8s} {'AP75':>8s} {'AP50:95':>10s}")
    for r in rows:
        print(f"{r['case']:10s} {r['precision']*100:8.3f} "
              f"{r['recall']*100:8.3f} {r['map50']*100:8.3f} "
              f"{r['map75']*100:8.3f} {r['map5095']*100:10.3f}")
    print("[saved]", save_dir)


if __name__ == "__main__":
    main()
