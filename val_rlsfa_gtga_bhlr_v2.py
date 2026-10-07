#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,
    build_dual_resolution_dataloader,
)
from engine.evaluator import (
    benchmark_fps,
    get_model_complexity,
    save_metrics_csv,
    validate_metrics,
)
from engine.runtime import select_device
from models.gtga_bhlr_factory import load_model_from_checkpoint


def validate(args):
    device = select_device(args.device)
    model, _, cfg = load_model_from_checkpoint(
        args.weights,
        device=device,
        fusion_override=args.fusion,
        strict=True,
        verbose=False,
    )

    data_cfg = cfg.get("data", {}) or {}
    rgb_yaml = args.rgb or data_cfg.get("rgb")
    tir_yaml = args.tir or data_cfg.get("tir")
    if rgb_yaml is None or tir_yaml is None:
        raise ValueError("RGB/TIR dataset YAML could not be resolved.")

    rgb_imgsz = int(args.rgb_imgsz or data_cfg.get("rgb_imgsz", 1920))
    rgb_semantic_imgsz = int(
        args.rgb_semantic_imgsz or data_cfg.get("rgb_semantic_imgsz", 640)
    )
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
        ckpt_path.parent.parent / f"val_{args.split}"
        if args.save_dir is None
        else Path(args.save_dir).expanduser().resolve()
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("RLSFA coarse anchor + Local-GTGA center-preserving scale + guided BHLR-v2 validation")
    print("=" * 72)
    print(f"Checkpoint            : {ckpt_path}")
    print(f"Model                 : {model.model_name}")
    print(f"Fusion                : {model.fusion_type}")
    print(f"Split                 : {args.split}")
    print(f"RGB high              : {rgb_imgsz}")
    print(f"RGB semantic          : {rgb_semantic_imgsz}")
    print(f"TIR                   : {tir_imgsz}")
    print("Translation           : coarse correlation")
    print("Scale                 : local sigma_T/sigma_R around UAV anchors")
    print("Frequency / fine      : OFF")
    print(f"BHLR guidance         : {model.bhlr_guidance_mode}")
    print(f"Device                : {device}")
    print("=" * 72 + "\n")

    params_m, flops_g = get_model_complexity(
        model=model,
        rgb_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=rgb_semantic_imgsz,
        tir_imgsz=tir_imgsz,
        device=device,
    )
    metrics = validate_metrics(
        model=model,
        loader=loader,
        device=device,
        amp=not args.no_amp,
        conf_thres=float(args.conf),
        iou_thres=float(args.iou),
        max_det=int(args.max_det),
        preview_num=int(args.preview_num) if args.plots else 0,
        preview_conf=float(args.preview_conf),
        preview_cols=int(args.preview_cols),
        preview_dir=save_dir / "previews" if args.plots else None,
    )
    fps, latency_ms = benchmark_fps(
        model=model,
        rgb_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=rgb_semantic_imgsz,
        tir_imgsz=tir_imgsz,
        device=device,
        amp=not args.no_amp,
        warmup=int(args.speed_warmup),
        iterations=int(args.speed_iters),
    )

    result = {
        "model": model.model_name,
        "fusion": model.fusion_type,
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "params_m": params_m,
        "flops_g": flops_g,
        "fps": fps,
        "latency_ms": latency_ms,
        "rgb_high_imgsz": rgb_imgsz,
        "rgb_semantic_imgsz": rgb_semantic_imgsz,
        "tir_imgsz": tir_imgsz,
        "alignment": "RLSFA coarse anchor + local GTGA center-preserving anisotropic scale",
        "bhlr_guidance": model.bhlr_guidance_mode,
        **metrics,
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_metrics_csv(
        save_dir / "metrics.csv",
        {
            "Model": model.model_name,
            "Params(M)": f"{params_m:.4f}",
            "FLOPs(G)": f"{flops_g:.4f}",
            "FPS": f"{fps:.4f}",
            "Latency(ms)": f"{latency_ms:.4f}",
            "P(%)": f"{metrics['precision'] * 100:.4f}",
            "R(%)": f"{metrics['recall'] * 100:.4f}",
            "AP50(%)": f"{metrics['map50'] * 100:.4f}",
            "AP75(%)": f"{metrics['map75'] * 100:.4f}",
            "AP50:95(%)": f"{metrics['map5095'] * 100:.4f}",
            "Split": args.split,
        },
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSave dir: {save_dir}")
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--rgb", default=None)
    p.add_argument("--tir", default=None)
    p.add_argument("--rgb-imgsz", type=int, default=None)
    p.add_argument("--rgb-semantic-imgsz", type=int, default=None)
    p.add_argument("--tir-imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="0")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--pair-mode", default=None, choices=["relative", "stem", "filename"])
    p.add_argument("--fusion", default="auto", choices=["auto", "concat", "add", "weighted"])
    p.add_argument("--save-dir", default=None)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--preview-num", type=int, default=6)
    p.add_argument("--preview-conf", type=float, default=0.25)
    p.add_argument("--preview-cols", type=int, default=3)
    p.add_argument("--speed-warmup", type=int, default=20)
    p.add_argument("--speed-iters", type=int, default=100)
    return p


if __name__ == "__main__":
    validate(parser().parse_args())
