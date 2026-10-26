#!/usr/bin/env python3
"""Drop-in replacement for models/gtga_bhlr_factory.py.

SOURCE BASIS:
  T1157806365/YOLO-tcy_v4@backup-20261021_best

IMPORTANT MANUAL STEP (one-time, BEFORE replacing this file):
  cp models/gtga_bhlr_factory.py models/gtga_bhlr_factory_base20261021.py

This module delegates every historical variant to the unchanged saved factory
and builds the new Figure-style Cat->Conv + Cat->Conv -> Add variant directly.
This is a complete runnable factory file, NOT an installer or patcher.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import torch

try:
    from models import gtga_bhlr_factory_base20261021 as _baseline
except ImportError as e:
    raise ImportError(
        "Missing models/gtga_bhlr_factory_base20261021.py. Before manually "
        "replacing models/gtga_bhlr_factory.py, run: "
        "cp models/gtga_bhlr_factory.py "
        "models/gtga_bhlr_factory_base20261021.py"
    ) from e

from models.rgbt_rlsfa_gtga_centered_bhlr_fig_conv_add_model import (
    RGBTFigureCatConvAddDetectionModel,
)

FIGURE_VARIANT = "rlsfa_gtga_centered_bhlr_fig_conv_add_v1"


def build_model_from_config(
    cfg: Dict,
    pretrained: bool = True,
    fusion_override: Optional[str] = None,
    names=None,
    nc: Optional[int] = None,
    verbose: bool = True,
):
    model_cfg = cfg.get("model", {}) or {}
    variant = str(model_cfg.get("variant", model_cfg.get("architecture", ""))).strip().lower()

    if variant != FIGURE_VARIANT:
        # Keep exactly the user's existing original factory behavior, including
        # any older variants, BDP/RODG and earlier Product Residual variant.
        return _baseline.build_model_from_config(
            cfg, pretrained=pretrained, fusion_override=fusion_override,
            names=names, nc=nc, verbose=verbose,
        )

    data_cfg = cfg.get("data", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    fusion = str(fusion_override or model_cfg.get("fusion", "add")).strip().lower()
    if fusion != "add":
        raise ValueError(f"Figure fusion expects model.fusion: add, got {fusion!r}")
    if nc is None:
        nc = int(model_cfg.get("nc", 1))
    if names is None and int(nc) == 1:
        names = {0: "UAV"}

    # Use backup branch's own parser for P2-compatible fusion_indices.
    fusion_indices = _baseline._resolve_fusion_indices(model_cfg)
    align_cfg = dict(model_cfg.get(
        "rlsfa_gtga_centered", model_cfg.get("alignment", {})
    ) or {})
    align_loss_cfg = dict(loss_cfg.get(
        "alignment", loss_cfg.get("rlsfa_gtga_centered", {})
    ) or {})
    bhlr_cfg = dict(model_cfg.get("bhlr", {}) or {})
    fig_cfg = dict(model_cfg.get("figure_fusion", {}) or {})

    model = RGBTFigureCatConvAddDetectionModel(
        figure_fusion_cfg=fig_cfg,
        model_name=str(model_cfg.get("name", "yolo26m")),
        nc=int(nc),
        pretrained=bool(pretrained),
        fusion=fusion,
        fusion_indices=fusion_indices,
        align_mode=str(model_cfg.get("align_mode", "bilinear")),
        names=names,
        semantic_imgsz=int(data_cfg.get("rgb_semantic_imgsz", 640)),
        rlsfa_gtga_centered_cfg=align_cfg,
        rlsfa_gtga_centered_loss_cfg=align_loss_cfg,
        bhlr_enabled=bool(bhlr_cfg.get("enabled", True)),
        bhlr_scales=bhlr_cfg.get("scales", [3]),
        bhlr_layers=bhlr_cfg.get("layers", None),
        high_to_semantic_ratio=int(bhlr_cfg.get("high_to_semantic_ratio", 3)),
        detail_channels=int(bhlr_cfg.get("detail_channels", 48)),
        stem_channels=int(bhlr_cfg.get("stem_channels", 16)),
        guide_channels=int(bhlr_cfg.get("guide_channels", 24)),
        bhlr_target_guidance_channels=int(bhlr_cfg.get("target_guidance_channels", 3)),
        bhlr_guidance_mode=str(bhlr_cfg.get("guidance_mode", "alignment")),
        bhlr_detach_alignment_guidance=bool(bhlr_cfg.get(
            "detach_alignment_guidance", True
        )),
        bhlr_geometry_routing_cfg=dict(bhlr_cfg.get("geometry_routing", {}) or {}),
        verbose=verbose,
    )

    # Preserve historical loss weight initialization exactly.
    if hasattr(model, "args"):
        model.args.box = float(loss_cfg.get("box", 7.5))
        model.args.cls = float(loss_cfg.get("cls", 0.5))
        model.args.dfl = float(loss_cfg.get("dfl", 1.5))
        model.criterion = None
    if verbose:
        print(f"[factory] variant       : {variant}")
        print(f"[factory] fusion_indices: {list(fusion_indices) if fusion_indices is not None else 'auto'}")
    return model


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    fusion_override: str = "auto",
    strict: bool = True,
    verbose: bool = False,
):
    # Support all existing checkpoints through the unchanged original loader.
    p = Path(checkpoint_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Expected checkpoint dict with model_state_dict")
    cfg = ckpt.get("config", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    variant = str(model_cfg.get("variant", "")).strip().lower()
    if variant != FIGURE_VARIANT:
        return _baseline.load_model_from_checkpoint(
            p, device=device, fusion_override=fusion_override,
            strict=strict, verbose=verbose,
        )

    saved_fusion = str(ckpt.get("fusion", model_cfg.get("fusion", "add"))).strip().lower()
    fusion = saved_fusion if str(fusion_override).strip().lower() == "auto" else str(fusion_override).strip().lower()
    model = build_model_from_config(
        cfg, pretrained=False, fusion_override=fusion,
        names=ckpt.get("names", {0: "UAV"}),
        nc=int(ckpt.get("nc", 1)), verbose=verbose,
    )
    model.load_state_dict(
        _baseline._strip_module_prefix(ckpt["model_state_dict"]),
        strict=bool(strict),
    )
    model = model.to(device)
    model.eval()
    return model, ckpt, cfg


__all__ = ["build_model_from_config", "load_model_from_checkpoint"]
