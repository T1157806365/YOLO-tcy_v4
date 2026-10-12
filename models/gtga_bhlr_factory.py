"""
Unified model factory for YOLO-tcy_v3.

Supports:
    - rlsfa_gtga_centered_bhlr_v2
    - rlsfa_gtga_bhlr_v2
    - bsca_bhlr_v2
    - rlsfa_scale_bhlr_v2

P2 compatibility
----------------
For a P2 detector such as yolo26m-p2, the detector may expose four
backbone pyramid levels [P2, P3, P4, P5].  The alignment/BHLR/fusion
modules in the current RLSFA-GTGA-Centered design must still operate
only on P3/P4/P5.

Therefore the experiment YAML can explicitly set:

    model:
      fusion_indices: [4, 6, 10]

The factory forwards that list to RGBTDetectionModel through the
selected wrapper.  If fusion_indices is omitted, legacy automatic
inference is preserved unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import torch


# ============================================================
# Helpers
# ============================================================

def _variant(cfg: Dict) -> str:
    model_cfg = cfg.get("model", {}) or {}

    explicit = str(
        model_cfg.get(
            "variant",
            model_cfg.get("architecture", ""),
        )
    ).strip().lower()

    if explicit:
        return explicit

    alignment_cfg = model_cfg.get("alignment", {}) or {}
    alignment_type = str(
        alignment_cfg.get("type", "")
    ).strip().lower()

    # Explicit centered configuration without variant.
    if (
        "rlsfa_gtga_centered" in model_cfg
        or alignment_type == "rlsfa_gtga_centered"
    ):
        return "rlsfa_gtga_centered_bhlr_v2"

    # Legacy RLSFA-GTGA interface.
    if (
        "rlsfa_gtga" in model_cfg
        or alignment_type == "rlsfa_gtga_scale"
    ):
        return "rlsfa_gtga_bhlr_v2"

    if (
        "bsca" in model_cfg
        or alignment_type == "bsca"
    ):
        return "bsca_bhlr_v2"

    if (
        "rlsfa_scale" in model_cfg
        or alignment_type == "rlsfa_scale"
    ):
        return "rlsfa_scale_bhlr_v2"

    # Keep historical default unchanged.
    return "rlsfa_gtga_bhlr_v2"


def _resolve_fusion_indices(
    model_cfg: Dict,
) -> Optional[Sequence[int]]:
    """
    Read optional model.fusion_indices.

    Accepted examples:
        fusion_indices: [4, 6, 10]
        fusion_indices: "4,6,10"

    None preserves the base detector's legacy auto-detection.
    """

    value = model_cfg.get("fusion_indices", None)

    if value is None:
        return None

    if isinstance(value, str):
        parts = [
            p.strip()
            for p in value.replace(";", ",").split(",")
            if p.strip()
        ]
        indices = tuple(int(p) for p in parts)

    elif isinstance(value, (list, tuple)):
        indices = tuple(int(x) for x in value)

    else:
        raise TypeError(
            "model.fusion_indices must be a list/tuple/string or null, "
            f"got {type(value)}"
        )

    if not indices:
        raise ValueError(
            "model.fusion_indices cannot be empty."
        )

    if any(i < 0 for i in indices):
        raise ValueError(
            f"model.fusion_indices must be non-negative, got {indices}"
        )

    if len(set(indices)) != len(indices):
        raise ValueError(
            f"model.fusion_indices contains duplicates: {indices}"
        )

    return indices


# ============================================================
# Factory
# ============================================================

def build_model_from_config(
    cfg: Dict,
    pretrained: bool = True,
    fusion_override: Optional[str] = None,
    names=None,
    nc: Optional[int] = None,
    verbose: bool = True,
):
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}

    fusion = str(
        fusion_override
        or model_cfg.get("fusion", "add")
    ).strip().lower()

    if nc is None:
        nc = int(
            model_cfg.get("nc", 1)
        )

    if names is None:
        names = (
            {0: "UAV"}
            if int(nc) == 1
            else None
        )

    variant = _variant(cfg)

    # CRITICAL FOR P2:
    # None => keep old automatic behavior.
    # [4,6,10] => keep RGB-T fusion/alignment/BHLR on P3/P4/P5
    # while the detector head can still output P2/P3/P4/P5.
    fusion_indices = _resolve_fusion_indices(
        model_cfg
    )

    # --------------------------------------------------------
    # BDP v1. Fixed Centered RLSFA + Local-GTGA + BHLR-v2
    #         + ADD fusion + detail-preserving final P3 neck bypass.
    #
    # IMPORTANT:
    #   This is a NEW variant. The historical
    #   rlsfa_gtga_centered_bhlr_v2 branch below remains unchanged.
    # --------------------------------------------------------
    if variant in {
        "rlsfa_gtga_centered_bhlr_bdp_v1",
        "rlsfa_gtga_centered_bdp_v1",
    }:
        from models.rgbt_rlsfa_gtga_centered_bhlr_bdp_model import (
            RGBTRLSFAGTGACenteredBHLRBDPDetectionModel,
        )

        align_cfg = dict(
            model_cfg.get(
                "rlsfa_gtga_centered",
                model_cfg.get("alignment", {}),
            )
            or {}
        )

        align_loss_cfg = dict(
            loss_cfg.get(
                "alignment",
                loss_cfg.get(
                    "rlsfa_gtga_centered",
                    {},
                ),
            )
            or {}
        )

        bhlr_cfg = dict(
            model_cfg.get("bhlr", {})
            or {}
        )

        bdp_cfg = dict(
            model_cfg.get("bdp_neck", {})
            or {}
        )

        model = RGBTRLSFAGTGACenteredBHLRBDPDetectionModel(
            model_name=str(
                model_cfg.get(
                    "name",
                    "yolo26s",
                )
            ),
            nc=int(nc),
            pretrained=bool(pretrained),
            fusion=fusion,
            fusion_indices=fusion_indices,
            align_mode=str(
                model_cfg.get(
                    "align_mode",
                    "bilinear",
                )
            ),
            names=names,
            semantic_imgsz=int(
                data_cfg.get(
                    "rgb_semantic_imgsz",
                    640,
                )
            ),
            rlsfa_gtga_centered_cfg=align_cfg,
            rlsfa_gtga_centered_loss_cfg=align_loss_cfg,
            bhlr_enabled=bool(
                bhlr_cfg.get(
                    "enabled",
                    True,
                )
            ),
            bhlr_scales=bhlr_cfg.get(
                "scales",
                [3],
            ),
            bhlr_layers=bhlr_cfg.get(
                "layers",
                None,
            ),
            high_to_semantic_ratio=int(
                bhlr_cfg.get(
                    "high_to_semantic_ratio",
                    3,
                )
            ),
            detail_channels=int(
                bhlr_cfg.get(
                    "detail_channels",
                    48,
                )
            ),
            stem_channels=int(
                bhlr_cfg.get(
                    "stem_channels",
                    16,
                )
            ),
            guide_channels=int(
                bhlr_cfg.get(
                    "guide_channels",
                    24,
                )
            ),
            bhlr_guidance_mode=str(
                bhlr_cfg.get(
                    "guidance_mode",
                    "alignment",
                )
            ),
            bhlr_detach_alignment_guidance=bool(
                bhlr_cfg.get(
                    "detach_alignment_guidance",
                    True,
                )
            ),
            bdp_neck_cfg=bdp_cfg,
            verbose=verbose,
        )

    # --------------------------------------------------------
    # 1. Centered RLSFA + Local-GTGA + BHLR-v2
    # --------------------------------------------------------

    # --------------------------------------------------------
    # BDP-Neck v2
    # Fixed centered alignment + BHLR + ADD + internal P3
    # detail fusion BEFORE the native P3 refinement block.
    # --------------------------------------------------------
    elif variant in {
        "rlsfa_gtga_centered_bhlr_bdp_neck_v2",
        "rlsfa_gtga_centered_bdp_neck_v2",
    }:
        from models.rgbt_rlsfa_gtga_centered_bhlr_bdp_neck_v2_model import (
            RGBTRLSFAGTGACenteredBHLRBDPNeckV2DetectionModel,
        )

        align_cfg = dict(
            model_cfg.get(
                "rlsfa_gtga_centered",
                model_cfg.get("alignment", {}),
            )
            or {}
        )

        align_loss_cfg = dict(
            loss_cfg.get(
                "alignment",
                loss_cfg.get("rlsfa_gtga_centered", {}),
            )
            or {}
        )

        bhlr_cfg = dict(model_cfg.get("bhlr", {}) or {})
        bdp_neck_v2_cfg = dict(model_cfg.get("bdp_neck_v2", {}) or {})

        model = RGBTRLSFAGTGACenteredBHLRBDPNeckV2DetectionModel(
            model_name=str(model_cfg.get("name", "yolo26s")),
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
            bhlr_guidance_mode=str(bhlr_cfg.get("guidance_mode", "alignment")),
            bhlr_detach_alignment_guidance=bool(
                bhlr_cfg.get("detach_alignment_guidance", True)
            ),
            bdp_neck_v2_cfg=bdp_neck_v2_cfg,
            verbose=verbose,
        )

    elif variant in {
        "rlsfa_gtga_centered_bhlr_v2",
        "rlsfa_gtga_centered",
    }:
        from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
            RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
        )

        align_cfg = dict(
            model_cfg.get(
                "rlsfa_gtga_centered",
                model_cfg.get("alignment", {}),
            )
            or {}
        )

        align_loss_cfg = dict(
            loss_cfg.get(
                "alignment",
                loss_cfg.get(
                    "rlsfa_gtga_centered",
                    {},
                ),
            )
            or {}
        )

        bhlr_cfg = dict(
            model_cfg.get("bhlr", {})
            or {}
        )

        model = RGBTRLSFAGTGACenteredBHLRV2DetectionModel(
            model_name=str(
                model_cfg.get(
                    "name",
                    "yolo26s",
                )
            ),
            nc=int(nc),
            pretrained=bool(pretrained),
            fusion=fusion,

            # P2 compatibility.
            fusion_indices=fusion_indices,

            align_mode=str(
                model_cfg.get(
                    "align_mode",
                    "bilinear",
                )
            ),
            names=names,
            semantic_imgsz=int(
                data_cfg.get(
                    "rgb_semantic_imgsz",
                    640,
                )
            ),

            rlsfa_gtga_centered_cfg=align_cfg,
            rlsfa_gtga_centered_loss_cfg=align_loss_cfg,

            bhlr_enabled=bool(
                bhlr_cfg.get(
                    "enabled",
                    True,
                )
            ),
            bhlr_scales=bhlr_cfg.get(
                "scales",
                [3],
            ),
            bhlr_layers=bhlr_cfg.get(
                "layers",
                None,
            ),
            high_to_semantic_ratio=int(
                bhlr_cfg.get(
                    "high_to_semantic_ratio",
                    3,
                )
            ),
            detail_channels=int(
                bhlr_cfg.get(
                    "detail_channels",
                    48,
                )
            ),
            stem_channels=int(
                bhlr_cfg.get(
                    "stem_channels",
                    16,
                )
            ),
            guide_channels=int(
                bhlr_cfg.get(
                    "guide_channels",
                    24,
                )
            ),
            bhlr_target_guidance_channels=int(
                bhlr_cfg.get(
                    "target_guidance_channels",
                    3,
                )
            ),
            bhlr_guidance_mode=str(
                bhlr_cfg.get(
                    "guidance_mode",
                    "alignment",
                )
            ),
            bhlr_detach_alignment_guidance=bool(
                bhlr_cfg.get(
                    "detach_alignment_guidance",
                    True,
                )
            ),
            bhlr_geometry_routing_cfg=dict(
                bhlr_cfg.get(
                    "geometry_routing",
                    {},
                )
                or {}
            ),
            verbose=verbose,
        )

    # --------------------------------------------------------
    # 2. Legacy RLSFA-GTGA + BHLR-v2
    # --------------------------------------------------------
    elif variant in {
        "rlsfa_gtga_bhlr_v2",
        "rlsfa_gtga",
        "rlsfa_gtga_scale",
    }:
        from models.rgbt_rlsfa_gtga_bhlr_model import (
            RGBTRLSFAGTGABHLRV2DetectionModel,
        )

        align_cfg = dict(
            model_cfg.get(
                "rlsfa_gtga",
                model_cfg.get(
                    "alignment",
                    {},
                ),
            )
            or {}
        )

        align_loss_cfg = dict(
            loss_cfg.get(
                "alignment",
                loss_cfg.get(
                    "rlsfa_gtga",
                    {},
                ),
            )
            or {}
        )

        bhlr_cfg = dict(
            model_cfg.get(
                "bhlr",
                {},
            )
            or {}
        )

        model = RGBTRLSFAGTGABHLRV2DetectionModel(
            model_name=str(
                model_cfg.get(
                    "name",
                    "yolo26s",
                )
            ),
            nc=int(nc),
            pretrained=bool(pretrained),
            fusion=fusion,
            fusion_indices=fusion_indices,
            align_mode=str(
                model_cfg.get(
                    "align_mode",
                    "bilinear",
                )
            ),
            names=names,
            semantic_imgsz=int(
                data_cfg.get(
                    "rgb_semantic_imgsz",
                    640,
                )
            ),
            rlsfa_gtga_cfg=align_cfg,
            rlsfa_gtga_loss_cfg=align_loss_cfg,
            bhlr_enabled=bool(
                bhlr_cfg.get(
                    "enabled",
                    True,
                )
            ),
            bhlr_scales=bhlr_cfg.get(
                "scales",
                [3],
            ),
            bhlr_layers=bhlr_cfg.get(
                "layers",
                None,
            ),
            high_to_semantic_ratio=int(
                bhlr_cfg.get(
                    "high_to_semantic_ratio",
                    3,
                )
            ),
            detail_channels=int(
                bhlr_cfg.get(
                    "detail_channels",
                    48,
                )
            ),
            stem_channels=int(
                bhlr_cfg.get(
                    "stem_channels",
                    16,
                )
            ),
            guide_channels=int(
                bhlr_cfg.get(
                    "guide_channels",
                    24,
                )
            ),
            bhlr_guidance_mode=str(
                bhlr_cfg.get(
                    "guidance_mode",
                    "alignment",
                )
            ),
            bhlr_detach_alignment_guidance=bool(
                bhlr_cfg.get(
                    "detach_alignment_guidance",
                    True,
                )
            ),
            verbose=verbose,
        )

    # --------------------------------------------------------
    # 3. BSCA + BHLR-v2
    # --------------------------------------------------------
    elif variant in {
        "bsca_bhlr_v2",
        "bsca",
    }:
        from models.rgbt_bsca_bhlr_model import (
            RGBTBSCABHLRV2DetectionModel,
        )

        bsca_cfg = dict(
            model_cfg.get(
                "bsca",
                model_cfg.get(
                    "alignment",
                    {},
                ),
            )
            or {}
        )

        bsca_loss_cfg = dict(
            loss_cfg.get(
                "bsca",
                {},
            )
            or {}
        )

        bhlr_cfg = dict(
            model_cfg.get(
                "bhlr",
                {},
            )
            or {}
        )

        model = RGBTBSCABHLRV2DetectionModel(
            model_name=str(
                model_cfg.get(
                    "name",
                    "yolo26s",
                )
            ),
            nc=int(nc),
            pretrained=bool(pretrained),
            fusion=fusion,
            fusion_indices=fusion_indices,
            align_mode=str(
                model_cfg.get(
                    "align_mode",
                    "bilinear",
                )
            ),
            names=names,
            semantic_imgsz=int(
                data_cfg.get(
                    "rgb_semantic_imgsz",
                    640,
                )
            ),
            bsca_cfg=bsca_cfg,
            bsca_loss_cfg=bsca_loss_cfg,
            bhlr_enabled=bool(
                bhlr_cfg.get(
                    "enabled",
                    True,
                )
            ),
            bhlr_scales=bhlr_cfg.get(
                "scales",
                [3],
            ),
            bhlr_layers=bhlr_cfg.get(
                "layers",
                None,
            ),
            high_to_semantic_ratio=int(
                bhlr_cfg.get(
                    "high_to_semantic_ratio",
                    3,
                )
            ),
            detail_channels=int(
                bhlr_cfg.get(
                    "detail_channels",
                    48,
                )
            ),
            stem_channels=int(
                bhlr_cfg.get(
                    "stem_channels",
                    16,
                )
            ),
            guide_channels=int(
                bhlr_cfg.get(
                    "guide_channels",
                    24,
                )
            ),
            bhlr_guidance_mode=str(
                bhlr_cfg.get(
                    "guidance_mode",
                    "bsca",
                )
            ),
            bhlr_detach_bsca_guidance=bool(
                bhlr_cfg.get(
                    "detach_bsca_guidance",
                    True,
                )
            ),
            verbose=verbose,
        )

    # --------------------------------------------------------
    # 4. RLSFA-Scale compatibility branch
    # --------------------------------------------------------
    elif variant in {
        "rlsfa_scale_bhlr_v2",
        "rlsfa_scale",
    }:
        from models.rgbt_rlsfa_scale_bhlr_model import (
            RGBTRLSFAScaleBHLRV2DetectionModel,
        )

        align_cfg = dict(
            model_cfg.get(
                "rlsfa_scale",
                model_cfg.get(
                    "alignment",
                    {},
                ),
            )
            or {}
        )

        align_loss_cfg = dict(
            loss_cfg.get(
                "alignment",
                loss_cfg.get(
                    "rlsfa_scale",
                    {},
                ),
            )
            or {}
        )

        bhlr_cfg = dict(
            model_cfg.get(
                "bhlr",
                {},
            )
            or {}
        )

        model = RGBTRLSFAScaleBHLRV2DetectionModel(
            model_name=str(
                model_cfg.get(
                    "name",
                    "yolo26s",
                )
            ),
            nc=int(nc),
            pretrained=bool(pretrained),
            fusion=fusion,
            fusion_indices=fusion_indices,
            align_mode=str(
                model_cfg.get(
                    "align_mode",
                    "bilinear",
                )
            ),
            names=names,
            semantic_imgsz=int(
                data_cfg.get(
                    "rgb_semantic_imgsz",
                    640,
                )
            ),
            alignment_cfg=align_cfg,
            alignment_loss_cfg=align_loss_cfg,
            bhlr_enabled=bool(
                bhlr_cfg.get(
                    "enabled",
                    True,
                )
            ),
            bhlr_scales=bhlr_cfg.get(
                "scales",
                [3],
            ),
            high_to_semantic_ratio=int(
                bhlr_cfg.get(
                    "high_to_semantic_ratio",
                    3,
                )
            ),
            detail_channels=int(
                bhlr_cfg.get(
                    "detail_channels",
                    48,
                )
            ),
            stem_channels=int(
                bhlr_cfg.get(
                    "stem_channels",
                    16,
                )
            ),
            guide_channels=int(
                bhlr_cfg.get(
                    "guide_channels",
                    24,
                )
            ),
            verbose=verbose,
        )

    else:
        raise ValueError(
            f"Unsupported model.variant={variant!r}"
        )

    # --------------------------------------------------------
    # Standard YOLO loss weights
    # --------------------------------------------------------
    if hasattr(model, "args"):
        model.args.box = float(
            loss_cfg.get(
                "box",
                7.5,
            )
        )
        model.args.cls = float(
            loss_cfg.get(
                "cls",
                0.5,
            )
        )
        model.args.dfl = float(
            loss_cfg.get(
                "dfl",
                1.5,
            )
        )
        model.criterion = None

    if verbose:
        print(
            "[factory] variant       :",
            variant,
        )
        print(
            "[factory] fusion_indices:",
            (
                list(fusion_indices)
                if fusion_indices is not None
                else "auto"
            ),
        )

    return model


# ============================================================
# Checkpoint loading
# ============================================================

def _strip_module_prefix(
    state_dict,
):
    if (
        state_dict
        and all(
            str(k).startswith("module.")
            for k in state_dict
        )
    ):
        return {
            str(k)[len("module."):]:
            v
            for k, v
            in state_dict.items()
        }

    return state_dict


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    fusion_override: str = "auto",
    strict: bool = True,
    verbose: bool = False,
):
    path = (
        Path(checkpoint_path)
        .expanduser()
        .resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        ckpt,
        dict,
    ):
        raise TypeError(
            "Expected checkpoint dict, "
            f"got {type(ckpt)}"
        )

    if "model_state_dict" not in ckpt:
        raise KeyError(
            "Checkpoint has no model_state_dict"
        )

    cfg = (
        ckpt.get(
            "config",
            {},
        )
        or {}
    )

    saved_fusion = str(
        ckpt.get(
            "fusion",
            (
                cfg.get(
                    "model",
                    {},
                )
                or {}
            ).get(
                "fusion",
                "add",
            ),
        )
    ).strip().lower()

    fusion = (
        saved_fusion
        if str(
            fusion_override
        ).strip().lower() == "auto"
        else str(
            fusion_override
        ).strip().lower()
    )

    model = build_model_from_config(
        cfg,
        pretrained=False,
        fusion_override=fusion,
        names=ckpt.get(
            "names",
            {0: "UAV"},
        ),
        nc=int(
            ckpt.get(
                "nc",
                1,
            )
        ),
        verbose=verbose,
    )

    model.load_state_dict(
        _strip_module_prefix(
            ckpt[
                "model_state_dict"
            ]
        ),
        strict=bool(
            strict
        ),
    )

    model = model.to(
        device
    )
    model.eval()

    return (
        model,
        ckpt,
        cfg,
    )


__all__ = [
    "build_model_from_config",
    "load_model_from_checkpoint",
]
