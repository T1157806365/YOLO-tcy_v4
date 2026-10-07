#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic two-model UAV scale-stratified comparison for YOLO-tcy_v3.

This is a direct replacement for the old:
    compare_baseline_bhlr_by_uav_scale.py

The old script hard-coded:
    Model 1 = baseline RGBTDetectionModel
    Model 2 = BHLR/factory model

This version compares ANY TWO compatible RGB-T checkpoints in the current
YOLO-tcy_v3 project, as long as each model can be restored by one of:

1) A serialized nn.Module stored in checkpoint["ema"] or checkpoint["model"].
2) A project checkpoint containing a state_dict + config that can be rebuilt
   by models.gtga_bhlr_factory.build_model_from_config.
3) A standard RGBT baseline checkpoint containing a state_dict + config.
4) A raw/state_dict-style checkpoint, when the user supplies --modelX-config.

Supported state-dict keys:
    model_state_dict
    state_dict
    raw_model_state_dict
    ema_state_dict
    model      (if it is a state_dict)
    ema        (if it is a state_dict)

The comparison is symmetric:
    Model 1 vs Model 2
and every delta is:
    Model2 - Model1

Scale bins:
    <8, 8-15, 16-31, 32-63, 64-95, >=96 px

Default scale definition:
    max(width_px, height_px)

Important:
- Both models are evaluated on the SAME dataset object.
- Inference uses the formal validation settings: AMP + conf/NMS/max_det.
- AP/P/R are calculated with process_single_image + Ultralytics DetMetrics,
  matching the formal validation metric path.
- For one size bin, GTs outside that bin are removed before metric calculation.
- ALL keeps every GT and should reproduce the formal validation result when
  weights/data/conf/NMS/max_det/AMP are identical.
- One-class UAV detection is assumed, matching the current project.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,
    build_dual_resolution_dataloader,
)
from engine.evaluator import postprocess_predictions
try:
    # Use the exact project matcher when it is exported by engine.evaluator.
    from engine.evaluator import process_single_image as _engine_process_single_image
except ImportError:
    _engine_process_single_image = None

from engine.runtime import preprocess_batch, select_device
from ultralytics.utils.metrics import DetMetrics, box_iou
from models.rgbt_model import RGBTDetectionModel
from models.gtga_bhlr_factory import build_model_from_config


SCALE_BINS = [
    ("<8", 0.0, 8.0),
    ("8-15", 8.0, 16.0),
    ("16-31", 16.0, 32.0),
    ("32-63", 32.0, 64.0),
    ("64-95", 64.0, 96.0),
    (">=96", 96.0, float("inf")),
]


def strip_known_prefixes(state_dict: Dict[str, torch.Tensor]):
    out = dict(state_dict)

    for prefix in ("module.", "_orig_mod."):
        if out and all(str(k).startswith(prefix) for k in out):
            out = {
                str(k)[len(prefix):]: v
                for k, v in out.items()
            }

    return out


def _looks_like_state_dict(obj) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False

    tensor_count = 0
    for value in obj.values():
        if torch.is_tensor(value):
            tensor_count += 1
        elif isinstance(value, nn.Parameter):
            tensor_count += 1
        else:
            return False

    return tensor_count > 0


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for --modelX-config YAML files. "
                "Install it in the current environment."
            ) from exc

        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    if not isinstance(cfg, dict):
        raise TypeError(
            f"Config must be a dict, got {type(cfg)} from {p}"
        )

    return cfg


def _checkpoint_config(ckpt, override_path: Optional[str] = None):
    override = _load_config_file(override_path)
    if override:
        return override

    if isinstance(ckpt, dict):
        cfg = ckpt.get("config", None)
        if isinstance(cfg, dict):
            return cfg

        cfg = ckpt.get("cfg", None)
        if isinstance(cfg, dict):
            return cfg

    return {}


def peek_checkpoint_config(
    path: str,
    config_override: Optional[str] = None,
) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    ckpt = torch.load(
        p,
        map_location="cpu",
        weights_only=False,
    )

    cfg = _checkpoint_config(
        ckpt,
        config_override,
    )

    del ckpt
    return cfg


def _extract_serialized_module(ckpt):
    """
    Prefer EMA only when the checkpoint actually stores an nn.Module.
    This supports Ultralytics-style and other object checkpoints.
    """
    if not isinstance(ckpt, dict):
        return ckpt if isinstance(ckpt, nn.Module) else None, None

    for key in ("ema", "model"):
        obj = ckpt.get(key, None)
        if isinstance(obj, nn.Module):
            return obj, key

    return None, None


def _extract_state_dict(ckpt):
    if _looks_like_state_dict(ckpt):
        return strip_known_prefixes(dict(ckpt)), "checkpoint_root"

    if not isinstance(ckpt, dict):
        return None, None

    for key in (
        "model_state_dict",
        "state_dict",
        "raw_model_state_dict",
        "ema_state_dict",
    ):
        obj = ckpt.get(key, None)
        if _looks_like_state_dict(obj):
            return strip_known_prefixes(dict(obj)), key

    # Some checkpoints use model/ema to store a state_dict rather than Module.
    for key in ("ema", "model"):
        obj = ckpt.get(key, None)
        if _looks_like_state_dict(obj):
            return strip_known_prefixes(dict(obj)), key

    return None, None


def _is_baseline_config(cfg: Dict[str, Any]) -> bool:
    """
    Detect a plain RGB-T baseline config.

    A missing variant is considered baseline ONLY when there are no known
    alignment/BHLR architecture blocks. This avoids sending a plain baseline
    to the GTGA/BHLR factory, whose historical default variant is non-baseline.
    """
    model_cfg = cfg.get("model", {}) or {}

    variant = str(
        model_cfg.get(
            "variant",
            model_cfg.get("architecture", ""),
        )
    ).strip().lower()

    baseline_aliases = {
        "",
        "baseline",
        "rgbt",
        "rgbt_baseline",
        "rgbt-baseline",
        "base",
        "standard",
    }

    architecture_keys = {
        "rlsfa_gtga_centered",
        "rlsfa_centered_gtga",
        "rlsfa_gtga",
        "rlsfa_scale",
        "bsca",
        "bhlr",
        "alignment",
        "rsdt",
        "rsdt_scales",
    }

    if variant and variant not in baseline_aliases:
        return False

    def block_is_enabled(value) -> bool:
        if value in (None, {}, [], (), False):
            return False
        if isinstance(value, dict):
            # A configured-but-disabled block is still baseline.
            if "enabled" in value:
                return bool(value.get("enabled", False))
            return bool(value)
        return bool(value)

    if any(block_is_enabled(model_cfg.get(k, None)) for k in architecture_keys):
        return False

    return True


def _build_baseline_from_config(
    cfg: Dict[str, Any],
    ckpt: Optional[Dict[str, Any]] = None,
):
    model_cfg = cfg.get("model", {}) or {}
    ckpt = ckpt if isinstance(ckpt, dict) else {}

    fusion_indices = model_cfg.get("fusion_indices", None)
    if isinstance(fusion_indices, str):
        fusion_indices = [
            int(x.strip())
            for x in fusion_indices.replace(";", ",").split(",")
            if x.strip()
        ]

    names = ckpt.get(
        "names",
        model_cfg.get("names", {0: "UAV"}),
    )

    nc = int(
        ckpt.get(
            "nc",
            model_cfg.get("nc", 1),
        )
    )

    model = RGBTDetectionModel(
        model_name=str(
            ckpt.get(
                "model_name",
                model_cfg.get("name", "yolo26s"),
            )
        ),
        nc=nc,
        pretrained=False,
        fusion=str(
            ckpt.get(
                "fusion",
                model_cfg.get("fusion", "add"),
            )
        ),
        fusion_indices=fusion_indices,
        align_mode=str(
            model_cfg.get("align_mode", "bilinear")
        ),
        names=names,
        verbose=False,
    )

    return model


def load_any_project_checkpoint(
    path: str,
    device: torch.device,
    loader: str = "auto",
    config_override: Optional[str] = None,
    strict: bool = True,
):
    """
    Load one comparison model.

    loader:
        auto
            Serialized Module -> use directly.
            Otherwise baseline/factory is inferred from config.

        baseline
            Force RGBTDetectionModel reconstruction.

        factory
            Force models.gtga_bhlr_factory.build_model_from_config.

        module
            Require checkpoint to contain a serialized nn.Module.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    ckpt = torch.load(
        p,
        map_location="cpu",
        weights_only=False,
    )

    cfg = _checkpoint_config(
        ckpt,
        config_override,
    )

    loader = str(loader).strip().lower()
    if loader not in {
        "auto",
        "baseline",
        "factory",
        "module",
    }:
        raise ValueError(
            f"Unsupported loader={loader!r}"
        )

    module_obj, module_key = _extract_serialized_module(ckpt)

    if loader in {"auto", "module"} and module_obj is not None:
        model = module_obj
        model = model.to(device).eval()

        info = {
            "loader": "serialized_module",
            "checkpoint_key": module_key,
            "strict": None,
        }
        return model, ckpt, cfg, info

    if loader == "module":
        raise RuntimeError(
            f"{p} does not contain a serialized nn.Module under "
            "checkpoint['ema'] or checkpoint['model']."
        )

    state_dict, state_key = _extract_state_dict(ckpt)
    if state_dict is None:
        keys = (
            sorted(ckpt.keys())
            if isinstance(ckpt, dict)
            else []
        )
        raise KeyError(
            "Could not find model weights in checkpoint.\n"
            f"Checkpoint: {p}\n"
            f"Available top-level keys: {keys}\n"
            "Supported keys are model_state_dict/state_dict/"
            "raw_model_state_dict/ema_state_dict/model/ema, "
            "or a serialized nn.Module."
        )

    if not cfg and loader == "auto":
        raise RuntimeError(
            f"{p} contains a state_dict but no reconstructable config. "
            "Pass --modelX-config and, if necessary, "
            "--modelX-loader baseline|factory."
        )

    selected_loader = loader
    if loader == "auto":
        selected_loader = (
            "baseline"
            if _is_baseline_config(cfg)
            else "factory"
        )

    if selected_loader == "baseline":
        model = _build_baseline_from_config(
            cfg,
            ckpt if isinstance(ckpt, dict) else None,
        )

    elif selected_loader == "factory":
        names = (
            ckpt.get("names", {0: "UAV"})
            if isinstance(ckpt, dict)
            else {0: "UAV"}
        )

        nc = int(
            ckpt.get("nc", 1)
            if isinstance(ckpt, dict)
            else 1
        )

        fusion = (
            (
                ckpt.get(
                    "fusion",
                    (cfg.get("model", {}) or {}).get(
                        "fusion",
                        "add",
                    ),
                )
            )
            if isinstance(ckpt, dict)
            else (cfg.get("model", {}) or {}).get(
                "fusion",
                "add",
            )
        )

        model = build_model_from_config(
            cfg,
            pretrained=False,
            fusion_override=str(fusion),
            names=names,
            nc=nc,
            verbose=False,
        )

    else:
        raise AssertionError(selected_loader)

    incompatible = model.load_state_dict(
        state_dict,
        strict=bool(strict),
    )

    model = model.to(device).eval()

    info = {
        "loader": selected_loader,
        "checkpoint_key": state_key,
        "strict": bool(strict),
        "missing_keys": list(
            getattr(incompatible, "missing_keys", [])
        ),
        "unexpected_keys": list(
            getattr(incompatible, "unexpected_keys", [])
        ),
    }

    return model, ckpt, cfg, info


def _candidate_data_values(
    cfg1: Dict[str, Any],
    cfg2: Dict[str, Any],
    key: str,
):
    vals = []
    for cfg in (cfg1, cfg2):
        data_cfg = cfg.get("data", {}) or {}
        value = data_cfg.get(key, None)
        if value is not None:
            vals.append(value)
    return vals


def _pick_data_value(
    explicit,
    cfg1: Dict[str, Any],
    cfg2: Dict[str, Any],
    keys,
    default=None,
    label="value",
):
    if explicit is not None:
        return explicit

    values = []

    for cfg in (cfg1, cfg2):
        data_cfg = cfg.get("data", {}) or {}
        for key in keys:
            if key in data_cfg and data_cfg[key] is not None:
                values.append(data_cfg[key])
                break

    if not values:
        return default

    if len(values) >= 2 and str(values[0]) != str(values[1]):
        print(
            f"[WARN] Model configs disagree on {label}: "
            f"{values[0]!r} vs {values[1]!r}. "
            f"Using Model 1 value {values[0]!r}. "
            "Pass an explicit CLI argument to override."
        )

    return values[0]


def resolve_data_cfg(
    cfg1: Dict[str, Any],
    cfg2: Dict[str, Any],
    args,
):
    rgb_yaml = _pick_data_value(
        args.rgb,
        cfg1,
        cfg2,
        ["rgb"],
        default=None,
        label="RGB dataset YAML",
    )

    tir_yaml = _pick_data_value(
        args.tir,
        cfg1,
        cfg2,
        ["tir"],
        default=None,
        label="TIR dataset YAML",
    )

    if not rgb_yaml or not tir_yaml:
        raise ValueError(
            "Could not resolve RGB/TIR dataset YAML. "
            "Pass --rgb and --tir explicitly."
        )

    rgb_imgsz = int(
        _pick_data_value(
            args.rgb_imgsz,
            cfg1,
            cfg2,
            ["rgb_imgsz", "rgb_high_imgsz"],
            default=1920,
            label="RGB high imgsz",
        )
    )

    semantic_imgsz = int(
        _pick_data_value(
            args.rgb_semantic_imgsz,
            cfg1,
            cfg2,
            ["rgb_semantic_imgsz"],
            default=640,
            label="RGB semantic imgsz",
        )
    )

    tir_imgsz = int(
        _pick_data_value(
            args.tir_imgsz,
            cfg1,
            cfg2,
            ["tir_imgsz"],
            default=640,
            label="TIR imgsz",
        )
    )

    pair_mode = str(
        _pick_data_value(
            args.pair_mode,
            cfg1,
            cfg2,
            ["pair_mode"],
            default="relative",
            label="pair_mode",
        )
    )

    return (
        rgb_yaml,
        tir_yaml,
        rgb_imgsz,
        semantic_imgsz,
        tir_imgsz,
        pair_mode,
    )


def xywhn_to_xyxy(
    boxes: torch.Tensor | np.ndarray,
    width_px: int,
    height_px: int,
) -> np.ndarray:
    if torch.is_tensor(boxes):
        b = boxes.detach().float().cpu().numpy()
    else:
        b = np.asarray(boxes, np.float32)

    b = b.reshape(-1, 4).astype(np.float32)

    if len(b) == 0:
        return np.zeros((0, 4), np.float32)

    cx, cy, w, h = b.T

    return np.stack(
        [
            (cx - w / 2.0) * float(width_px),
            (cy - h / 2.0) * float(height_px),
            (cx + w / 2.0) * float(width_px),
            (cy + h / 2.0) * float(height_px),
        ],
        axis=1,
    ).astype(np.float32)


def bbox_scale_px(
    boxes_xywhn: torch.Tensor | np.ndarray,
    size_px: int,
    measure: str,
) -> np.ndarray:
    """
    Size bin only.

    The detector metric itself is always evaluated in RGB-semantic coordinates,
    exactly like the formal validation path. `size_px` only defines the bin
    labels (<8, 8-15, ...).
    """
    if torch.is_tensor(boxes_xywhn):
        b = boxes_xywhn.detach().float().cpu().numpy()
    else:
        b = np.asarray(boxes_xywhn, np.float32)

    b = b.reshape(-1, 4).astype(np.float32)

    if len(b) == 0:
        return np.zeros((0,), np.float32)

    w = b[:, 2] * float(size_px)
    h = b[:, 3] * float(size_px)

    if measure == "max_side":
        return np.maximum(w, h)
    if measure == "min_side":
        return np.minimum(w, h)
    if measure == "sqrt_area":
        return np.sqrt(np.maximum(w * h, 0.0))

    raise ValueError(measure)


def autocast_context(
    device: torch.device,
    enabled: bool,
):
    """
    Match the AMP behavior used by the formal validation script.
    """
    enabled = bool(enabled) and device.type == "cuda"

    return torch.autocast(
        device_type=device.type,
        dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.bfloat16
        ),
        enabled=enabled,
    )


def _fallback_process_single_image(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    iouv: torch.Tensor,
) -> np.ndarray:
    """
    Exact-style fallback for engine.evaluator.process_single_image().

    Matching follows the same project validation logic:
      1) build GT x prediction IoU
      2) class-consistent matches only
      3) independently for each IoU threshold 0.50:0.95
      4) sort candidate matches by IoU descending
      5) unique prediction
      6) unique GT
    """
    pred_cls = pred["cls"]
    pred_boxes = pred["bboxes"]
    gt_cls = gt["cls"]
    gt_boxes = gt["bboxes"]

    if pred_cls.shape[0] == 0:
        return np.zeros(
            (0, len(iouv)),
            dtype=bool,
        )

    if gt_cls.shape[0] == 0:
        return np.zeros(
            (pred_cls.shape[0], len(iouv)),
            dtype=bool,
        )

    iou = box_iou(
        gt_boxes,
        pred_boxes,
    )

    correct = np.zeros(
        (pred_cls.shape[0], len(iouv)),
        dtype=bool,
    )

    correct_class = (
        gt_cls[:, None]
        ==
        pred_cls[None, :]
    )

    iou = iou * correct_class

    iou_np = (
        iou.detach()
        .cpu()
        .numpy()
    )

    thresholds = (
        iouv.detach()
        .cpu()
        .tolist()
    )

    for threshold_index, threshold in enumerate(thresholds):
        matches = np.nonzero(
            iou_np >= threshold
        )

        matches = np.array(matches).T

        if matches.shape[0] == 0:
            continue

        if matches.shape[0] > 1:
            # highest IoU first
            matches = matches[
                iou_np[
                    matches[:, 0],
                    matches[:, 1],
                ].argsort()[::-1]
            ]

            # each prediction -> one GT
            matches = matches[
                np.unique(
                    matches[:, 1],
                    return_index=True,
                )[1]
            ]

            # each GT -> one prediction
            matches = matches[
                np.unique(
                    matches[:, 0],
                    return_index=True,
                )[1]
            ]

        pred_indices = matches[:, 1].astype(int)

        correct[
            pred_indices,
            threshold_index,
        ] = True

    return correct


def official_process_single_image(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    iouv: torch.Tensor,
) -> np.ndarray:
    if _engine_process_single_image is not None:
        result = _engine_process_single_image(
            pred=pred,
            gt=gt,
            iouv=iouv,
        )
        if torch.is_tensor(result):
            result = result.detach().cpu().numpy()
        return np.asarray(result, dtype=bool)

    return _fallback_process_single_image(
        pred=pred,
        gt=gt,
        iouv=iouv,
    )


def normalize_names(names) -> Dict[int, str]:
    if isinstance(names, dict):
        return {
            int(k): str(v)
            for k, v in names.items()
        }

    return {
        i: str(name)
        for i, name in enumerate(names)
    }


def metrics_for_bin_official(
    records: List[Dict[str, np.ndarray]],
    names,
    lo: float,
    hi: float,
) -> Dict[str, float]:
    """
    Calculate one scale bin with the SAME metric machinery as formal validation:

        process_single_image
            ->
        DetMetrics.update_stats
            ->
        DetMetrics.process
            ->
        DetMetrics.mean_results

    Important:
    For one size bin, GTs outside that bin are removed from that bin's GT set.
    Therefore detections of an out-of-bin UAV can be counted as false positives.
    This is intentional here because the user requested the exact validation
    calculation, not the old custom "ignore other-scale GT" AP calculator.

    For ALL, no GT is filtered, so ALL should reproduce formal validation
    (assuming the same weights/data/conf/NMS/max_det/AMP).
    """
    names = normalize_names(names)
    metrics = DetMetrics(names=names)

    iouv = torch.linspace(
        0.50,
        0.95,
        10,
        device="cpu",
    )

    total_gt = 0

    for rec in records:
        gt_scales = rec["gt_scales"]

        mask = (
            (gt_scales >= float(lo))
            &
            (gt_scales < float(hi))
        )

        gt_boxes_np = rec["gt_boxes"][mask]
        gt_cls_np = rec["gt_cls"][mask]

        pred_boxes_np = rec["pred_boxes"]
        pred_scores_np = rec["pred_scores"]
        pred_cls_np = rec["pred_cls"]

        total_gt += int(len(gt_boxes_np))

        gt = {
            "cls": torch.from_numpy(
                gt_cls_np.astype(np.float32, copy=False)
            ),
            "bboxes": torch.from_numpy(
                gt_boxes_np.astype(np.float32, copy=False)
            ),
        }

        pred = {
            "cls": torch.from_numpy(
                pred_cls_np.astype(np.float32, copy=False)
            ),
            "bboxes": torch.from_numpy(
                pred_boxes_np.astype(np.float32, copy=False)
            ),
            "conf": torch.from_numpy(
                pred_scores_np.astype(np.float32, copy=False)
            ),
        }

        tp = official_process_single_image(
            pred=pred,
            gt=gt,
            iouv=iouv,
        )

        target_cls = (
            gt["cls"]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        metrics.update_stats(
            {
                "tp": tp,
                "conf": pred_scores_np.astype(
                    np.float32,
                    copy=False,
                ),
                "pred_cls": pred_cls_np.astype(
                    np.float32,
                    copy=False,
                ),
                "target_cls": target_cls,
                "target_img": np.unique(
                    target_cls
                ),
                "im_name": rec.get(
                    "image_name",
                    "",
                ),
            }
        )

    if total_gt == 0:
        return {
            "GT": 0,
            "P": float("nan"),
            "R": float("nan"),
            "AP50": float("nan"),
            "AP75": float("nan"),
            "AP50_95": float("nan"),
        }

    metrics.process(
        plot=False
    )

    (
        precision,
        recall,
        map50,
        map5095,
    ) = metrics.mean_results()

    map75 = float(
        metrics.box.map75
    )

    return {
        "GT": int(total_gt),
        "P": float(precision),
        "R": float(recall),
        "AP50": float(map50),
        "AP75": float(map75),
        "AP50_95": float(map5095),
    }


def _callable_signature_has_parameter(fn, name: str) -> bool:
    try:
        import inspect
        sig = inspect.signature(fn)
        return name in sig.parameters
    except (TypeError, ValueError):
        return False


def infer_input_mode(model) -> str:
    """
    Infer the direct forward-call form.

    semantic:
        model(rgb_semantic, tir)

    dualres:
        model(rgb_high, tir, rgb_semantic=rgb_semantic)

    high:
        model(rgb_high, tir)
    """
    fn = model.forward if hasattr(model, "forward") else model

    if _callable_signature_has_parameter(
        fn,
        "rgb_semantic",
    ):
        return "dualres"

    return "semantic"


def model_forward(
    input_mode: str,
    model,
    batch,
):
    mode = str(
        input_mode
    ).strip().lower()

    if mode == "auto":
        mode = infer_input_mode(
            model
        )

    # Direct __call__/forward, matching formal validation.
    if mode == "semantic":
        return model(
            batch["rgb_semantic_img"],
            batch["tir_img"],
        )

    if mode == "high":
        return model(
            batch["rgb_img"],
            batch["tir_img"],
        )

    if mode == "dualres":
        return model(
            batch["rgb_img"],
            batch["tir_img"],
            rgb_semantic=batch["rgb_semantic_img"],
        )

    raise ValueError(
        f"Unsupported input mode={input_mode!r}. "
        "Use auto/semantic/high/dualres."
    )


@torch.inference_mode()
def collect_records(
    label: str,
    input_mode: str,
    model,
    dataset,
    device,
    batch_size: int,
    workers: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    scale_imgsz: int,
    scale_measure: str,
    amp: bool,
):
    """
    Run inference once and retain exactly the information needed by DetMetrics.

    Inference settings mirror the formal validation script:
        AMP
        conf=0.001 by default
        NMS IoU=0.70 by default
        max_det=300 by default
    """
    loader = build_dual_resolution_dataloader(
        dataset,
        batch_size=int(batch_size),
        workers=int(workers),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()

    records: List[
        Dict[str, np.ndarray]
    ] = []

    seen = 0

    for batch_i, raw_batch in enumerate(loader):
        batch = preprocess_batch(
            raw_batch,
            device,
        )

        with autocast_context(
            device,
            amp,
        ):
            pred_raw = model_forward(
                input_mode,
                model,
                batch,
            )

        pred_list = postprocess_predictions(
            preds=pred_raw,
            model=model,
            conf_thres=float(
                conf_thres
            ),
            iou_thres=float(
                iou_thres
            ),
            max_det=int(
                max_det
            ),
        )

        bsz = int(
            batch[
                "rgb_semantic_img"
            ].shape[0]
        )

        semantic_h = int(
            batch[
                "rgb_semantic_img"
            ].shape[2]
        )

        semantic_w = int(
            batch[
                "rgb_semantic_img"
            ].shape[3]
        )

        gt_boxes_all = batch[
            "rgb_semantic_bboxes"
        ]

        gt_cls_all = batch[
            "rgb_cls"
        ].view(-1)

        gt_batch_idx = (
            batch[
                "rgb_batch_idx"
            ]
            .long()
            .view(-1)
        )

        for bi in range(bsz):
            gt_mask = (
                gt_batch_idx == bi
            )

            gt_xywhn = (
                gt_boxes_all[
                    gt_mask
                ]
            )

            gt_cls = (
                gt_cls_all[
                    gt_mask
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )

            # Metric coordinates are ALWAYS semantic coordinates,
            # exactly like formal validation.
            gt_boxes = xywhn_to_xyxy(
                gt_xywhn,
                width_px=semantic_w,
                height_px=semantic_h,
            )

            # Scale-space is only for assigning the GT to a size bin.
            gt_scales = bbox_scale_px(
                gt_xywhn,
                scale_imgsz,
                scale_measure,
            )

            if (
                pred_list
                and bi < len(pred_list)
                and pred_list[bi] is not None
            ):
                p = pred_list[bi]

                pb = p.get(
                    "bboxes",
                    torch.zeros(
                        (0, 4),
                        device=device,
                    ),
                )

                ps = p.get(
                    "conf",
                    torch.zeros(
                        (0,),
                        device=device,
                    ),
                )

                pc = p.get(
                    "cls",
                    torch.zeros(
                        (0,),
                        device=device,
                    ),
                )

                pred_boxes = (
                    pb.detach()
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1, 4)
                    .astype(np.float32)
                )

                pred_scores = (
                    ps.detach()
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float32)
                )

                pred_cls = (
                    pc.detach()
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float32)
                )

            else:
                pred_boxes = np.zeros(
                    (0, 4),
                    np.float32,
                )

                pred_scores = np.zeros(
                    (0,),
                    np.float32,
                )

                pred_cls = np.zeros(
                    (0,),
                    np.float32,
                )

            image_name = ""

            try:
                image_name = Path(
                    batch[
                        "rgb_path"
                    ][bi]
                ).name
            except Exception:
                image_name = (
                    f"image_{seen:08d}"
                )

            records.append(
                {
                    "gt_boxes":
                        gt_boxes,
                    "gt_cls":
                        gt_cls,
                    "gt_scales":
                        gt_scales,
                    "pred_boxes":
                        pred_boxes,
                    "pred_scores":
                        pred_scores,
                    "pred_cls":
                        pred_cls,
                    "image_name":
                        image_name,
                }
            )

            seen += 1

        if (
            (batch_i + 1)
            % 20
            == 0
        ):
            print(
                f"[{label}] processed "
                f"{seen}/{len(dataset)} images"
            )

    if len(records) != len(dataset):
        raise RuntimeError(
            f"{label}: record count "
            f"{len(records)} != "
            f"dataset size {len(dataset)}"
        )

    return records



def pct(x):
    return "" if not np.isfinite(x) else round(float(x) * 100.0, 3)


def write_outputs(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "size_stratified_comparison.csv"
    json_path = out_dir / "size_stratified_comparison.json"

    fields = [
        "Scale",
        "Range_px",
        "GT_count",

        "Model1_P(%)",
        "Model1_R(%)",
        "Model1_AP50(%)",
        "Model1_AP75(%)",
        "Model1_AP50:95(%)",

        "Model2_P(%)",
        "Model2_R(%)",
        "Model2_AP50(%)",
        "Model2_AP75(%)",
        "Model2_AP50:95(%)",

        "Delta_R_Model2-Model1(pp)",
        "Delta_AP50_Model2-Model1(pp)",
        "Delta_AP50:95_Model2-Model1(pp)",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "rows": rows},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")



def main():
    parser = argparse.ArgumentParser(
        "Compare any two compatible RGB-T models by UAV pixel scale"
    )

    # Old aliases are intentionally retained so old command lines still work.
    parser.add_argument(
        "--model1-weights",
        "--baseline-weights",
        dest="model1_weights",
        required=True,
        help="Weights/checkpoint for Model 1.",
    )
    parser.add_argument(
        "--model2-weights",
        "--bhlr-weights",
        dest="model2_weights",
        required=True,
        help="Weights/checkpoint for Model 2.",
    )

    parser.add_argument(
        "--model1-name",
        default="Model1",
        help="Display name only; does not affect loading.",
    )
    parser.add_argument(
        "--model2-name",
        default="Model2",
        help="Display name only; does not affect loading.",
    )

    parser.add_argument(
        "--model1-loader",
        choices=["auto", "baseline", "factory", "module"],
        default="auto",
    )
    parser.add_argument(
        "--model2-loader",
        choices=["auto", "baseline", "factory", "module"],
        default="auto",
    )

    parser.add_argument(
        "--model1-config",
        default=None,
        help="Optional YAML/JSON config override for reconstructing Model 1.",
    )
    parser.add_argument(
        "--model2-config",
        default=None,
        help="Optional YAML/JSON config override for reconstructing Model 2.",
    )

    parser.add_argument(
        "--model1-input",
        choices=["auto", "semantic", "high", "dualres"],
        default="auto",
        help=(
            "semantic: predict(rgb_semantic,tir); "
            "high: predict(rgb_high,tir); "
            "dualres: predict(rgb_high,tir,rgb_semantic=...); "
            "auto: infer from predict signature."
        ),
    )
    parser.add_argument(
        "--model2-input",
        choices=["auto", "semantic", "high", "dualres"],
        default="auto",
    )

    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Use strict=False when loading state_dict checkpoints.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--rgb", default=None)
    parser.add_argument("--tir", default=None)
    parser.add_argument(
        "--rgb-imgsz",
        type=int,
        default=None,
        help="High-resolution RGB canvas, e.g. 1920.",
    )
    parser.add_argument(
        "--rgb-semantic-imgsz",
        type=int,
        default=None,
        help="Semantic RGB detector canvas, normally 640.",
    )
    parser.add_argument(
        "--tir-imgsz",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--pair-mode",
        default=None,
        choices=["relative", "stem", "filename"],
    )

    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="5")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", "--nms-iou", dest="iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument(
        "--scale-space",
        choices=["semantic", "high"],
        default="semantic",
    )
    parser.add_argument(
        "--scale-measure",
        choices=["max_side", "min_side", "sqrt_area"],
        default="max_side",
    )

    parser.add_argument(
        "--save-dir",
        default="runs/scale_stratified_model1_vs_model2",
    )

    args = parser.parse_args()
    device = select_device(args.device)
    strict = not args.non_strict

    # ----------------------------------------------------------
    # Read only configs first so the exact same dataset is used.
    # ----------------------------------------------------------
    cfg1 = peek_checkpoint_config(
        args.model1_weights,
        args.model1_config,
    )
    cfg2 = peek_checkpoint_config(
        args.model2_weights,
        args.model2_config,
    )

    (
        rgb_yaml,
        tir_yaml,
        rgb_imgsz,
        semantic_imgsz,
        tir_imgsz,
        pair_mode,
    ) = resolve_data_cfg(
        cfg1,
        cfg2,
        args,
    )

    scale_imgsz = (
        semantic_imgsz
        if args.scale_space == "semantic"
        else rgb_imgsz
    )

    print("\n" + "=" * 76)
    print("UAV scale-stratified comparison: arbitrary Model 1 vs Model 2")
    print("=" * 76)
    print(f"Model 1 name      : {args.model1_name}")
    print(f"Model 1 weights   : {args.model1_weights}")
    print(f"Model 1 loader    : {args.model1_loader}")
    print(f"Model 1 input     : {args.model1_input}")
    print()
    print(f"Model 2 name      : {args.model2_name}")
    print(f"Model 2 weights   : {args.model2_weights}")
    print(f"Model 2 loader    : {args.model2_loader}")
    print(f"Model 2 input     : {args.model2_input}")
    print()
    print(f"Split             : {args.split}")
    print(f"RGB high          : {rgb_imgsz}")
    print(f"RGB semantic      : {semantic_imgsz}")
    print(f"TIR               : {tir_imgsz}")
    print(
        f"Scale space       : {args.scale_space} "
        f"({scale_imgsz}px square network canvas)"
    )
    print(f"Scale measure     : {args.scale_measure}")
    print("Bins              : <8, 8-15, 16-31, 32-63, 64-95, >=96 px")
    print(f"State-dict strict : {strict}")
    print(f"Conf threshold    : {args.conf}")
    print(f"NMS IoU           : {args.iou}")
    print(f"Max detections    : {args.max_det}")
    print(f"AMP               : {not args.no_amp}")
    print("Metric engine     : process_single_image + Ultralytics DetMetrics")
    print("=" * 76 + "\n")

    dataset = build_dual_resolution_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=args.split,
        rgb_high_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=semantic_imgsz,
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    # ----------------------------------------------------------
    # Model 1
    # ----------------------------------------------------------
    print(f"\n[1/2] Evaluating {args.model1_name}...")

    model1, ckpt1, loaded_cfg1, info1 = load_any_project_checkpoint(
        args.model1_weights,
        device=device,
        loader=args.model1_loader,
        config_override=args.model1_config,
        strict=strict,
    )

    actual_input1 = (
        infer_input_mode(model1)
        if args.model1_input == "auto"
        else args.model1_input
    )

    print(
        f"[Model1] loader={info1['loader']} "
        f"checkpoint_key={info1['checkpoint_key']} "
        f"input={actual_input1}"
    )

    model1_names = normalize_names(
        getattr(model1, "names", {0: "UAV"})
    )

    model1_records = collect_records(
        args.model1_name,
        actual_input1,
        model1,
        dataset,
        device,
        args.batch,
        args.workers,
        args.conf,
        args.iou,
        args.max_det,
        scale_imgsz,
        args.scale_measure,
        amp=not args.no_amp,
    )

    del model1, ckpt1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Model 2
    # ----------------------------------------------------------
    print(f"\n[2/2] Evaluating {args.model2_name}...")

    model2, ckpt2, loaded_cfg2, info2 = load_any_project_checkpoint(
        args.model2_weights,
        device=device,
        loader=args.model2_loader,
        config_override=args.model2_config,
        strict=strict,
    )

    actual_input2 = (
        infer_input_mode(model2)
        if args.model2_input == "auto"
        else args.model2_input
    )

    print(
        f"[Model2] loader={info2['loader']} "
        f"checkpoint_key={info2['checkpoint_key']} "
        f"input={actual_input2}"
    )

    model2_names = normalize_names(
        getattr(model2, "names", {0: "UAV"})
    )

    model2_records = collect_records(
        args.model2_name,
        actual_input2,
        model2,
        dataset,
        device,
        args.batch,
        args.workers,
        args.conf,
        args.iou,
        args.max_det,
        scale_imgsz,
        args.scale_measure,
        amp=not args.no_amp,
    )

    del model2, ckpt2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Compare
    # ----------------------------------------------------------
    rows = []

    n1 = args.model1_name[:12]
    n2 = args.model2_name[:12]

    print("\nScale-stratified results")
    print("-" * 118)
    print(
        f"{'Scale':<10}"
        f"{'GT':>7}"
        f"{(n1 + ' AP50'):>14}"
        f"{(n2 + ' AP50'):>14}"
        f"{'ΔAP50':>10}"
        f"{(n1 + ' AP50:95'):>17}"
        f"{(n2 + ' AP50:95'):>17}"
        f"{'ΔAP50:95':>12}"
        f"{'ΔR':>10}"
    )
    print("-" * 118)

    eval_bins = SCALE_BINS + [
        ("ALL", 0.0, float("inf"))
    ]

    for scale_name, lo, hi in eval_bins:
        m1 = metrics_for_bin_official(
            model1_records,
            model1_names,
            lo,
            hi,
        )
        m2 = metrics_for_bin_official(
            model2_records,
            model2_names,
            lo,
            hi,
        )

        if m1["GT"] != m2["GT"]:
            raise RuntimeError(
                f"GT count mismatch for {scale_name}: "
                f"model1={m1['GT']} model2={m2['GT']}"
            )

        range_txt = (
            f"[{int(lo)},"
            f"{'inf' if math.isinf(hi) else int(hi)})"
        )

        d_r = (
            (m2["R"] - m1["R"]) * 100
            if np.isfinite(m2["R"])
            and np.isfinite(m1["R"])
            else float("nan")
        )

        d50 = (
            (m2["AP50"] - m1["AP50"]) * 100
            if np.isfinite(m2["AP50"])
            and np.isfinite(m1["AP50"])
            else float("nan")
        )

        d5095 = (
            (m2["AP50_95"] - m1["AP50_95"]) * 100
            if np.isfinite(m2["AP50_95"])
            and np.isfinite(m1["AP50_95"])
            else float("nan")
        )

        row = {
            "Scale": scale_name,
            "Range_px": range_txt,
            "GT_count": m1["GT"],

            "Model1_P(%)": pct(m1["P"]),
            "Model1_R(%)": pct(m1["R"]),
            "Model1_AP50(%)": pct(m1["AP50"]),
            "Model1_AP75(%)": pct(m1["AP75"]),
            "Model1_AP50:95(%)": pct(m1["AP50_95"]),

            "Model2_P(%)": pct(m2["P"]),
            "Model2_R(%)": pct(m2["R"]),
            "Model2_AP50(%)": pct(m2["AP50"]),
            "Model2_AP75(%)": pct(m2["AP75"]),
            "Model2_AP50:95(%)": pct(m2["AP50_95"]),

            "Delta_R_Model2-Model1(pp)":
                "" if not np.isfinite(d_r)
                else round(d_r, 3),

            "Delta_AP50_Model2-Model1(pp)":
                "" if not np.isfinite(d50)
                else round(d50, 3),

            "Delta_AP50:95_Model2-Model1(pp)":
                "" if not np.isfinite(d5095)
                else round(d5095, 3),
        }

        rows.append(row)

        def fmt(x):
            return (
                "   n/a"
                if not np.isfinite(x)
                else f"{x * 100:7.2f}"
            )

        print(
            f"{scale_name:<10}"
            f"{m1['GT']:>7}"
            f"{fmt(m1['AP50']):>14}"
            f"{fmt(m2['AP50']):>14}"
            f"{('n/a' if not np.isfinite(d50) else f'{d50:+.2f}'):>10}"
            f"{fmt(m1['AP50_95']):>17}"
            f"{fmt(m2['AP50_95']):>17}"
            f"{('n/a' if not np.isfinite(d5095) else f'{d5095:+.2f}'):>12}"
            f"{('n/a' if not np.isfinite(d_r) else f'{d_r:+.2f}'):>10}"
        )

    print("-" * 118)

    meta = {
        "model1_name": args.model1_name,
        "model2_name": args.model2_name,

        "model1_weights":
            str(Path(args.model1_weights).expanduser().resolve()),
        "model2_weights":
            str(Path(args.model2_weights).expanduser().resolve()),

        "model1_loader_requested": args.model1_loader,
        "model2_loader_requested": args.model2_loader,
        "model1_loader_used": info1["loader"],
        "model2_loader_used": info2["loader"],

        "model1_checkpoint_key": info1["checkpoint_key"],
        "model2_checkpoint_key": info2["checkpoint_key"],

        "model1_input_mode": actual_input1,
        "model2_input_mode": actual_input2,

        "delta_definition": "Model2 - Model1",

        "split": args.split,
        "dataset_size": len(dataset),

        "rgb_yaml": str(rgb_yaml),
        "tir_yaml": str(tir_yaml),

        "rgb_high_imgsz": rgb_imgsz,
        "rgb_semantic_imgsz": semantic_imgsz,
        "tir_imgsz": tir_imgsz,

        "scale_space": args.scale_space,
        "scale_imgsz": scale_imgsz,
        "scale_measure": args.scale_measure,

        "scale_bins": [
            {
                "name": name,
                "lo": lo,
                "hi": None if math.isinf(hi) else hi,
            }
            for name, lo, hi in SCALE_BINS
        ],

        "ap_iou_thresholds": [
            round(float(x), 2)
            for x in np.arange(0.50, 0.951, 0.05)
        ],

        "conf_thres": args.conf,
        "nms_iou": args.iou,
        "amp": not args.no_amp,
        "strict_state_dict": strict,

        "note": (
            "Metric calculation uses the same process_single_image + "
            "Ultralytics DetMetrics pipeline as formal validation. "
            "For each size bin, only GTs in that bin are retained; "
            "therefore detections of other-size UAVs may count as FP. "
            "ALL contains every GT and should match formal validation "
            "when conf/iou/max_det/AMP/data/model loading are identical. "
            "All deltas are Model2 - Model1."
        ),
    }

    write_outputs(
        Path(args.save_dir).expanduser().resolve(),
        rows,
        meta,
    )


if __name__ == "__main__":
    main()
