"""
RGB-T YOLO with GTGA + GTGA-guided BHLR-v2
==========================================

Stable foundation:
    models.rgbt_model.RGBTDetectionModel

Current pipeline:
    TIR 640 -> TIR backbone (once)

    RGB semantic 640 -> RGB backbone layer-by-layer

    At the GTGA reference level (default P3):
        raw RGB feature + raw TIR feature
        -> independent target distributions
        -> diagonal Gaussian target geometry
        -> analytical one-shot TIR->RGB warp
        -> joint target support

    At selected BHLR layer(s):
        RGB-high lost-detail bank
        + current RGB semantic feature
        + GTGA-aligned TIR
        + GTGA joint target support
        -> BHLR-v2
        -> enhanced RGB feature
        -> continue through deeper RGB backbone layers

    The SAME GTGA normalized geometry is propagated to all configured RGB-T
    fusion pyramid levels, so fusion receives geometrically normalized TIR.

Direction:
    GTGA -> BHLR

There is NO high-resolution-detail feedback into GTGA in this version.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from models.rgbt_model import (
    RGBTDetectionModel,
    get_layer_input,
)
from models.modules.alignment.builder import (
    build_alignment,
)
from models.modules.alignment.gtga import (
    GTGAAlignment,
)
from models.modules.bhlr_v2 import (
    MultiScaleBHLRv2,
)

class RGBTGTGABHLRV2DetectionModel(
    RGBTDetectionModel
):
    def __init__(
        self,
        model_name: str = "yolo26n",
        nc: int = 1,
        pretrained: bool = True,
        fusion: str = "add",
        fusion_indices: Optional[
            Sequence[int]
        ] = None,
        align_mode: str = "bilinear",
        names: Optional[
            Union[
                Dict[int, str],
                Sequence[str],
            ]
        ] = None,
        semantic_imgsz: int = 640,

        # -----------------------------------------------------
        # GTGA
        # -----------------------------------------------------
        gtga_cfg: Optional[
            Dict[str, Any]
        ] = None,
        gtga_loss_cfg: Optional[
            Dict[str, Any]
        ] = None,

        # -----------------------------------------------------
        # BHLR-v2
        # -----------------------------------------------------
        bhlr_enabled: bool = True,

        # Preferred: conceptual P3/P4/P5 configuration.
        bhlr_scales: Optional[
            Iterable[int]
        ] = (3,),

        # Optional advanced interface: actual backbone layer IDs.
        bhlr_layers: Optional[
            Iterable[int]
        ] = None,

        high_to_semantic_ratio: int = 3,
        detail_channels: int = 48,
        stem_channels: int = 16,
        guide_channels: int = 24,

        bhlr_guidance_mode: str = "gtga",
        bhlr_detach_gtga_guidance: bool = True,

        verbose: bool = True,
    ):
        # =====================================================
        # Stable RGB-T dual-backbone YOLO
        # =====================================================
        super().__init__(
            model_name=model_name,
            nc=nc,
            pretrained=pretrained,
            fusion=fusion,
            fusion_indices=fusion_indices,
            align_mode=align_mode,
            names=names,
            verbose=verbose,
        )

        self.semantic_imgsz = int(
            semantic_imgsz
        )

        if len(self.fusion_indices) < 3:
            raise RuntimeError(
                "GTGA+BHLR-v2 expects at least "
                "P3/P4/P5 fusion indices."
            )

        self.scale_to_index = {
            3:
                int(
                    self.fusion_indices[0]
                ),
            4:
                int(
                    self.fusion_indices[1]
                ),
            5:
                int(
                    self.fusion_indices[2]
                ),
        }

        # Historical compatibility for val_rsdt.py.
        self.rsdt_scale_to_index = dict(
            self.scale_to_index
        )

        # =====================================================
        # GTGA
        # =====================================================
        self.alignment_cfg = dict(
            gtga_cfg or {}
        )

        # Main config defaults to ON + GTGA.
        self.alignment_cfg.setdefault(
            "enabled",
            True,
        )
        self.alignment_cfg.setdefault(
            "type",
            "gtga",
        )
        self.alignment_cfg.setdefault(
            "scales",
            [3],
        )

        self.alignment_enabled = bool(
            self.alignment_cfg.get(
                "enabled",
                True,
            )
        )

        self.alignment_use_for_bhlr = bool(
            self.alignment_cfg.get(
                "use_for_bhlr",
                True,
            )
        )

        self.alignment_use_for_fusion = bool(
            self.alignment_cfg.get(
                "use_for_fusion",
                True,
            )
        )

        self.alignment = build_alignment(
            self.alignment_cfg
        )

        if (
            self.alignment_enabled
            and not isinstance(
                self.alignment,
                GTGAAlignment,
            )
        ):
            raise TypeError(
                "This wrapper requires alignment.type='gtga' "
                "when GTGA is enabled. "
                f"Got {type(self.alignment).__name__}."
            )

        if self.alignment_enabled:
            self.gtga_reference_scale = int(
                self.alignment.reference_scale
            )
            self.gtga_reference_layer = int(
                self.scale_to_index[
                    self.gtga_reference_scale
                ]
            )
        else:
            self.gtga_reference_scale = int(
                self.alignment_cfg.get(
                    "reference_scale",
                    self.alignment_cfg.get(
                        "scales",
                        [3],
                    )[0],
                )
            )
            self.gtga_reference_layer = int(
                self.scale_to_index[
                    self.gtga_reference_scale
                ]
            )

        # -----------------------------------------------------
        # Initialize GTGA LazyConv parameters BEFORE optimizer.
        # -----------------------------------------------------
        self._initialize_gtga_lazy_modules()

        # =====================================================
        # GTGA auxiliary supervision
        # =====================================================
        self.gtga_loss_cfg = dict(
            gtga_loss_cfg or {}
        )

        self.gtga_loss_enabled = bool(
            self.gtga_loss_cfg.get(
                "enabled",
                True,
            )
        )

        if (
            self.gtga_loss_enabled
            and not self.alignment_enabled
        ):
            raise ValueError(
                "GTGA loss cannot be enabled when GTGA alignment "
                "is disabled."
            )

        self.lambda_gtga = float(
            self.gtga_loss_cfg.get(
                "lambda_gtga",
                0.2,
            )
        )

        self.lambda_target_map = float(
            self.gtga_loss_cfg.get(
                "lambda_target_map",
                1.0,
            )
        )

        self.lambda_gaussian = float(
            self.gtga_loss_cfg.get(
                "lambda_gaussian",
                5.0,
            )
        )

        self.targetness_alpha = float(
            self.gtga_loss_cfg.get(
                "targetness_alpha",
                0.75,
            )
        )

        self.targetness_gamma = float(
            self.gtga_loss_cfg.get(
                "targetness_gamma",
                2.0,
            )
        )

        # Current clean GTGA has ONE reference Gaussian geometry.
        loss_scales = self.gtga_loss_cfg.get(
            "scales",
            [self.gtga_reference_scale],
        )

        loss_scales = tuple(
            sorted(
                {
                    int(x)
                    for x in loss_scales
                }
            )
        )

        if (
            self.gtga_loss_enabled
            and loss_scales
            != (self.gtga_reference_scale,)
        ):
            raise ValueError(
                "Current GTGA loss must supervise only the single "
                f"reference scale P{self.gtga_reference_scale}; "
                f"got {list(loss_scales)}"
            )

        self.gtga_loss_scales = loss_scales

        # =====================================================
        # BHLR-v2
        # =====================================================
        self.bhlr_enabled = bool(
            bhlr_enabled
        )

        self.bhlr_guidance_mode = str(
            bhlr_guidance_mode
        ).strip().lower()

        if (
            self.bhlr_enabled
            and self.bhlr_guidance_mode == "gtga"
            and not self.alignment_enabled
        ):
            raise ValueError(
                "BHLR guidance_mode='gtga' requires GTGA enabled. "
                "For BHLR-only ablation use guidance_mode='none'."
            )

        if bhlr_layers is not None:
            resolved_layers = tuple(
                sorted(
                    {
                        int(x)
                        for x in bhlr_layers
                    }
                )
            )

            conceptual_scales = tuple(
                sorted(
                    s
                    for s, idx in self.scale_to_index.items()
                    if idx in resolved_layers
                )
            )

        else:
            conceptual_scales = tuple(
                sorted(
                    {
                        int(x)
                        for x in (
                            bhlr_scales
                            or (3,)
                        )
                    }
                )
            )

            invalid = [
                s
                for s in conceptual_scales
                if s not in self.scale_to_index
            ]

            if invalid:
                raise ValueError(
                    "bhlr_scales only supports conceptual "
                    f"P3/P4/P5, got {invalid}"
                )

            resolved_layers = tuple(
                self.scale_to_index[s]
                for s in conceptual_scales
            )

        self.bhlr_layers = (
            resolved_layers
            if self.bhlr_enabled
            else tuple()
        )

        self.bhlr_scales = (
            conceptual_scales
            if self.bhlr_enabled
            else tuple()
        )

        # val_rsdt.py compatibility.
        self.rsdt_scales = self.bhlr_scales
        self.use_guidance = (
            self.bhlr_guidance_mode
            == "gtga"
        )

        self.bhlr = None

        if self.bhlr_enabled:
            if not self.bhlr_layers:
                raise ValueError(
                    "BHLR is enabled but no BHLR layers were resolved."
                )

            # GTGA must be known before inline BHLR if BHLR uses GTGA guide.
            if (
                self.bhlr_guidance_mode == "gtga"
                and any(
                    layer
                    < self.gtga_reference_layer
                    for layer in self.bhlr_layers
                )
            ):
                raise ValueError(
                    "GTGA-guided BHLR cannot be injected BEFORE "
                    "the GTGA reference layer. "
                    f"GTGA layer={self.gtga_reference_layer}, "
                    f"BHLR layers={list(self.bhlr_layers)}"
                )

            feature_channels = (
                self._infer_any_layer_channels(
                    self.bhlr_layers
                )
            )

            rgb_channels = {
                layer:
                    int(
                        feature_channels[
                            layer
                        ][
                            "rgb"
                        ]
                    )
                for layer in self.bhlr_layers
            }

            tir_channels = {
                layer:
                    int(
                        feature_channels[
                            layer
                        ][
                            "tir"
                        ]
                    )
                for layer in self.bhlr_layers
            }

            self.bhlr = MultiScaleBHLRv2(
                rgb_channels=rgb_channels,
                tir_channels=tir_channels,
                scales=self.bhlr_layers,
                high_to_semantic_ratio=int(
                    high_to_semantic_ratio
                ),
                detail_channels=int(
                    detail_channels
                ),
                stem_channels=int(
                    stem_channels
                ),
                guide_channels=int(
                    guide_channels
                ),
                exact_identity_when_equal=True,
                guidance_mode=(
                    self.bhlr_guidance_mode
                ),
                detach_gtga_guidance=bool(
                    bhlr_detach_gtga_guidance
                ),
            )

        if verbose:
            print(
                "\n"
                "============================================================"
            )
            print(
                "GTGA + GTGA-guided BHLR-v2 Detector"
            )
            print(
                "============================================================"
            )
            print(
                f"Model                : {self.model_name}"
            )
            print(
                f"Fusion               : {self.fusion_type}"
            )
            print(
                f"RGB semantic         : {self.semantic_imgsz}"
            )
            print(
                f"GTGA enabled         : {self.alignment_enabled}"
            )
            print(
                f"GTGA reference       : "
                f"P{self.gtga_reference_scale} "
                f"(layer {self.gtga_reference_layer})"
            )
            print(
                f"GTGA -> fusion       : "
                f"{self.alignment_use_for_fusion}"
            )
            print(
                f"GTGA -> BHLR         : "
                f"{self.alignment_use_for_bhlr}"
            )
            print(
                f"GTGA loss            : "
                f"{self.gtga_loss_enabled}"
            )
            print(
                f"BHLR enabled         : "
                f"{self.bhlr_enabled}"
            )
            print(
                f"BHLR layers          : "
                f"{list(self.bhlr_layers)}"
            )
            print(
                f"BHLR guidance        : "
                f"{self.bhlr_guidance_mode}"
            )
            print(
                "GTGA geometry        : "
                "target Gaussian -> analytical one-shot warp"
            )
            print(
                "HR feedback -> GTGA  : OFF"
            )
            print(
                "============================================================\n"
            )

    # =========================================================
    # Compatibility aliases
    # =========================================================

    @property
    def rsdt(self):
        """
        val_rsdt/train_rsdt compatibility without registering the same
        nn.Module twice in state_dict.
        """
        if self.bhlr is None:
            raise AttributeError(
                "BHLR is disabled."
            )

        return self.bhlr

    # =========================================================
    # Utilities
    # =========================================================

    def _fallback_semantic_rgb(
        self,
        rgb_high: torch.Tensor,
    ) -> torch.Tensor:
        if (
            rgb_high.shape[-2:]
            == (
                self.semantic_imgsz,
                self.semantic_imgsz,
            )
        ):
            return rgb_high

        return F.interpolate(
            rgb_high,
            size=(
                self.semantic_imgsz,
                self.semantic_imgsz,
            ),
            mode="bilinear",
            align_corners=False,
        )

    @torch.no_grad()
    def _initialize_gtga_lazy_modules(
        self,
    ):
        if not self.alignment_enabled:
            return

        states = (
            self.model.training,
            self.tir_backbone.training,
            self.alignment.training,
        )

        self.model.eval()
        self.tir_backbone.eval()
        self.alignment.eval()

        device = next(
            self.model.parameters()
        ).device

        dummy = torch.zeros(
            1,
            3,
            256,
            256,
            device=device,
            dtype=torch.float32,
        )

        rgb_outputs = list(
            self.forward_rgb_backbone(
                dummy
            )
        )

        tir_outputs = list(
            self.forward_tir_backbone(
                dummy
            )
        )

        _ = self.alignment(
            rgb_features=rgb_outputs,
            tir_features=tir_outputs,
            scale_to_index=self.scale_to_index,
            return_debug=False,
        )

        self.model.train(states[0])
        self.tir_backbone.train(states[1])
        self.alignment.train(states[2])

        from torch.nn.parameter import (
            UninitializedParameter,
        )

        bad = [
            name
            for name, parameter
            in self.named_parameters()
            if isinstance(
                parameter,
                UninitializedParameter,
            )
        ]

        if bad:
            raise RuntimeError(
                "GTGA lazy parameters remain uninitialized: "
                + ", ".join(bad)
            )

    @torch.no_grad()
    def _infer_any_layer_channels(
        self,
        layer_ids: Iterable[int],
    ) -> Dict[int, Dict[str, int]]:
        layer_ids = tuple(
            int(x)
            for x in layer_ids
        )

        rgb_training = self.model.training
        tir_training = self.tir_backbone.training

        self.model.eval()
        self.tir_backbone.eval()

        device = next(
            self.model.parameters()
        ).device

        dummy = torch.zeros(
            1,
            3,
            256,
            256,
            device=device,
            dtype=torch.float32,
        )

        rgb_outputs = list(
            self.forward_rgb_backbone(
                dummy
            )
        )

        tir_outputs = list(
            self.forward_tir_backbone(
                dummy
            )
        )

        channels = {}

        for layer in layer_ids:
            if (
                layer < 0
                or layer >= self.backbone_len
            ):
                raise ValueError(
                    f"BHLR layer {layer} outside backbone "
                    f"range 0..{self.backbone_len - 1}"
                )

            r = rgb_outputs[layer]
            t = tir_outputs[layer]

            if (
                not isinstance(r, torch.Tensor)
                or not isinstance(t, torch.Tensor)
            ):
                raise TypeError(
                    f"Backbone layer {layer} must return Tensor "
                    "in both streams."
                )

            channels[layer] = {
                "rgb":
                    int(r.shape[1]),
                "tir":
                    int(t.shape[1]),
            }

        self.model.train(
            rgb_training
        )
        self.tir_backbone.train(
            tir_training
        )

        return channels

    def _geometry_from_info(
        self,
        ref_info: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            "rgb_mu":
                ref_info["rgb_mu"],
            "tir_mu":
                ref_info["tir_mu"],
            "rgb_sigma":
                ref_info["rgb_sigma"],
            "tir_sigma":
                ref_info["tir_sigma"],
        }

    # =========================================================
    # One-pass backbone execution
    # =========================================================

    def _forward_backbones(
        self,
        rgb_high: torch.Tensor,
        rgb_semantic: torch.Tensor,
        tir: torch.Tensor,
        return_bhlr_debug: bool = False,
        return_alignment_debug: bool = False,
    ):
        # -----------------------------------------------------
        # TIR backbone: exactly one pass
        # -----------------------------------------------------
        raw_tir_outputs = list(
            self.forward_tir_backbone(
                tir
            )
        )

        aligned_tir_outputs = list(
            raw_tir_outputs
        )

        # -----------------------------------------------------
        # Build HR lost-detail bank exactly once if BHLR active.
        # -----------------------------------------------------
        use_bhlr_now = bool(
            self.bhlr_enabled
            and self.bhlr is not None
            and (
                rgb_high.shape[-2:]
                != rgb_semantic.shape[-2:]
            )
        )

        detail_raw = None

        bhlr_debug = {
            "equal_resolution":
                (
                    rgb_high.shape[-2:]
                    == rgb_semantic.shape[-2:]
                ),
            "layers": {},
            "scales": {},
        }

        if use_bhlr_now:
            (
                detail_raw,
                reconstructed_high,
                residual_high,
            ) = (
                self.bhlr
                .build_lost_detail_bank(
                    rgb_high,
                    rgb_semantic,
                )
            )

            if return_bhlr_debug:
                bhlr_debug.update(
                    {
                        "reconstructed_high":
                            reconstructed_high,
                        "resolution_residual":
                            residual_high,
                        "residual_s2d":
                            F.pixel_unshuffle(
                                residual_high,
                                downscale_factor=(
                                    self.bhlr.ratio
                                ),
                            ),
                        "detail_raw":
                            detail_raw,
                        "detail_bank":
                            detail_raw,
                    }
                )

        # -----------------------------------------------------
        # RGB backbone: one pass, with GTGA before BHLR at P3.
        # -----------------------------------------------------
        rgb_outputs = []
        x = rgb_semantic

        gtga_ref_info = None

        rgb_layers = list(
            self.model
        )[
            :self.backbone_len
        ]

        for layer_id, module in enumerate(
            rgb_layers
        ):
            module_input = get_layer_input(
                module,
                x,
                rgb_outputs,
            )

            x = module(
                module_input
            )

            # -------------------------------------------------
            # 1) GTGA first: see RAW semantic RGB P3.
            # -------------------------------------------------
            if (
                self.alignment_enabled
                and layer_id
                == self.gtga_reference_layer
            ):
                (
                    aligned_ref,
                    gtga_ref_info,
                ) = (
                    self.alignment
                    .align_reference_pair(
                        rgb_feat=x,
                        tir_feat=(
                            raw_tir_outputs[
                                layer_id
                            ]
                        ),
                        return_debug=(
                            return_alignment_debug
                        ),
                    )
                )

                aligned_tir_outputs[
                    layer_id
                ] = aligned_ref

            # -------------------------------------------------
            # 2) BHLR second: GTGA -> BHLR only.
            # -------------------------------------------------
            if (
                use_bhlr_now
                and layer_id
                in self.bhlr_layers
            ):
                # If BHLR is at a deeper level, propagate the SAME
                # normalized GTGA geometry to that TIR feature first.
                if (
                    self.alignment_enabled
                    and self.alignment_use_for_bhlr
                ):
                    if gtga_ref_info is None:
                        raise RuntimeError(
                            "GTGA-guided BHLR reached a layer "
                            "before GTGA geometry was available."
                        )

                    if (
                        layer_id
                        != self.gtga_reference_layer
                    ):
                        aligned_tir_outputs[
                            layer_id
                        ] = (
                            self.alignment
                            .warp_tensor(
                                source=(
                                    raw_tir_outputs[
                                        layer_id
                                    ]
                                ),
                                geometry=(
                                    self._geometry_from_info(
                                        gtga_ref_info
                                    )
                                ),
                                output_hw=(
                                    x.shape[-2:]
                                ),
                            )
                        )

                    tir_for_bhlr = (
                        aligned_tir_outputs[
                            layer_id
                        ]
                    )

                    guidance = {
                        "joint_target_support":
                            gtga_ref_info[
                                "joint_target_support"
                            ]
                    }

                else:
                    tir_for_bhlr = (
                        raw_tir_outputs[
                            layer_id
                        ]
                    )
                    guidance = None

                if return_bhlr_debug:
                    (
                        x,
                        layer_debug,
                    ) = (
                        self.bhlr
                        .enhance_layer(
                            layer_id=layer_id,
                            detail_raw=detail_raw,
                            rgb_feat=x,
                            tir_feat=tir_for_bhlr,
                            external_guidance=guidance,
                            return_debug=True,
                        )
                    )

                    bhlr_debug[
                        "layers"
                    ][
                        layer_id
                    ] = layer_debug

                    bhlr_debug[
                        "scales"
                    ][
                        layer_id
                    ] = layer_debug

                else:
                    x = (
                        self.bhlr
                        .enhance_layer(
                            layer_id=layer_id,
                            detail_raw=detail_raw,
                            rgb_feat=x,
                            tir_feat=tir_for_bhlr,
                            external_guidance=guidance,
                            return_debug=False,
                        )
                    )

            rgb_outputs.append(
                x
            )

        # -----------------------------------------------------
        # Propagate ONE GTGA geometry to all fusion pyramid levels.
        # This keeps the alignment idea clean: one target geometry,
        # one normalized transform, pyramid-wide application.
        # -----------------------------------------------------
        if (
            self.alignment_enabled
            and self.alignment_use_for_fusion
        ):
            if gtga_ref_info is None:
                raise RuntimeError(
                    "GTGA enabled but reference geometry "
                    "was not produced."
                )

            geometry = self._geometry_from_info(
                gtga_ref_info
            )

            apply_scales = self.alignment_cfg.get(
                "apply_scales",
                [3, 4, 5],
            )

            for scale in sorted(
                {
                    int(s)
                    for s in apply_scales
                }
            ):
                if scale not in self.scale_to_index:
                    raise KeyError(
                        f"GTGA apply scale P{scale} "
                        "is not available."
                    )

                idx = self.scale_to_index[
                    scale
                ]

                if (
                    idx
                    == self.gtga_reference_layer
                ):
                    continue

                aligned_tir_outputs[
                    idx
                ] = (
                    self.alignment
                    .warp_tensor(
                        source=(
                            raw_tir_outputs[
                                idx
                            ]
                        ),
                        geometry=geometry,
                        output_hw=(
                            rgb_outputs[
                                idx
                            ].shape[-2:]
                        ),
                    )
                )

        alignment_info = {
            "enabled":
                bool(
                    self.alignment_enabled
                ),
            "type":
                (
                    "gtga"
                    if self.alignment_enabled
                    else "identity"
                ),
            "reference_scale":
                self.gtga_reference_scale,
            "reference_layer":
                self.gtga_reference_layer,
            "has_alignment":
                bool(
                    self.alignment_enabled
                ),
            "scale_debug":
                (
                    {
                        self.gtga_reference_scale:
                            gtga_ref_info
                    }
                    if gtga_ref_info
                    is not None
                    else {}
                ),
        }

        if gtga_ref_info is not None:
            alignment_info[
                "reference_debug"
            ] = gtga_ref_info

        return (
            rgb_outputs,
            raw_tir_outputs,
            aligned_tir_outputs,
            alignment_info,
            bhlr_debug,
        )

    # =========================================================
    # Features / prediction
    # =========================================================

    def forward_features(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        rgb_semantic: Optional[
            torch.Tensor
        ] = None,
        return_bhlr_debug: bool = False,
        return_alignment_debug: bool = False,
    ) -> Dict:
        if rgb_semantic is None:
            rgb_semantic = (
                self._fallback_semantic_rgb(
                    rgb
                )
            )

        (
            rgb_outputs,
            raw_tir_outputs,
            aligned_tir_outputs,
            alignment_info,
            bhlr_debug,
        ) = self._forward_backbones(
            rgb_high=rgb,
            rgb_semantic=rgb_semantic,
            tir=tir,
            return_bhlr_debug=(
                return_bhlr_debug
            ),
            return_alignment_debug=(
                return_alignment_debug
            ),
        )

        tir_for_fusion = (
            aligned_tir_outputs
            if (
                self.alignment_enabled
                and self.alignment_use_for_fusion
            )
            else raw_tir_outputs
        )

        (
            fused_outputs,
            fusion_features,
        ) = self.fuse_backbone_features(
            rgb_outputs,
            tir_for_fusion,
        )

        result = {
            "rgb_backbone":
                rgb_outputs,
            "tir_backbone":
                raw_tir_outputs,
            "aligned_tir_backbone":
                aligned_tir_outputs,
            "tir_for_fusion":
                tir_for_fusion,
            "fused_backbone":
                fused_outputs,
            "fusion_features":
                fusion_features,
            "rgb_semantic":
                rgb_semantic,
            "alignment_info":
                alignment_info,
            "alignment_enabled":
                self.alignment_enabled,
            "alignment_type":
                (
                    "gtga"
                    if self.alignment_enabled
                    else "identity"
                ),
            "bhlr_enabled":
                self.bhlr_enabled,
            "bhlr_layers":
                self.bhlr_layers,
        }

        if return_bhlr_debug:
            result[
                "bhlr_debug"
            ] = bhlr_debug

            # val/old debug compatibility.
            result[
                "rsdt_debug"
            ] = bhlr_debug

        return result

    def predict(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        rgb_semantic: Optional[
            torch.Tensor
        ] = None,
        return_features: bool = False,
        return_bhlr_debug: bool = False,
        return_alignment_debug: bool = False,
    ):
        features = self.forward_features(
            rgb,
            tir,
            rgb_semantic=rgb_semantic,
            return_bhlr_debug=(
                return_bhlr_debug
            ),
            return_alignment_debug=(
                return_alignment_debug
            ),
        )

        pred = self.forward_head(
            features[
                "fused_backbone"
            ]
        )

        if (
            return_features
            or return_bhlr_debug
            or return_alignment_debug
        ):
            return {
                "pred":
                    pred,
                **features,
            }

        return pred

    def forward(
        self,
        rgb,
        tir=None,
        rgb_semantic=None,
        return_features: bool = False,
        return_bhlr_debug: bool = False,
        return_alignment_debug: bool = False,
    ):
        if isinstance(
            rgb,
            dict,
        ):
            return self.loss(
                rgb
            )

        if tir is None:
            raise ValueError(
                "GTGA+BHLR-v2 requires both RGB and TIR."
            )

        return self.predict(
            rgb,
            tir,
            rgb_semantic=rgb_semantic,
            return_features=(
                return_features
            ),
            return_bhlr_debug=(
                return_bhlr_debug
            ),
            return_alignment_debug=(
                return_alignment_debug
            ),
        )

    # =========================================================
    # GTGA supervision
    # =========================================================

    @staticmethod
    def _build_target_mask(
        boxes: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
        h: int,
        w: int,
        device,
        dtype,
    ) -> torch.Tensor:
        mask = torch.zeros(
            int(batch_size),
            1,
            int(h),
            int(w),
            device=device,
            dtype=dtype,
        )

        for i in range(
            int(
                boxes.shape[0]
            )
        ):
            bi = int(
                batch_idx[i].item()
            )

            cx, cy, bw, bh = boxes[i]

            x1 = max(
                0,
                min(
                    w - 1,
                    int(
                        torch.floor(
                            (cx - bw / 2) * w
                        ).item()
                    ),
                ),
            )
            y1 = max(
                0,
                min(
                    h - 1,
                    int(
                        torch.floor(
                            (cy - bh / 2) * h
                        ).item()
                    ),
                ),
            )
            x2 = max(
                x1 + 1,
                min(
                    w,
                    int(
                        torch.ceil(
                            (cx + bw / 2) * w
                        ).item()
                    ),
                ),
            )
            y2 = max(
                y1 + 1,
                min(
                    h,
                    int(
                        torch.ceil(
                            (cy + bh / 2) * h
                        ).item()
                    ),
                ),
            )

            mask[
                bi,
                0,
                y1:y2,
                x1:x2,
            ] = 1.0

        return mask

    def _focal_bce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prob = torch.sigmoid(
            logits
        )

        ce = (
            F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            )
        )

        pt = (
            target * prob
            + (1.0 - target)
            * (1.0 - prob)
        )

        alpha_t = (
            self.targetness_alpha
            * target
            + (
                1.0
                - self.targetness_alpha
            )
            * (1.0 - target)
        )

        return (
            alpha_t
            * (1.0 - pt).pow(
                self.targetness_gamma
            )
            * ce
        ).mean()

    @staticmethod
    def _diag_gaussian_w2_sq(
        pred_mu: torch.Tensor,
        pred_sigma: torch.Tensor,
        gt_mu: torch.Tensor,
        gt_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Squared W2 distance between diagonal Gaussians:
            ||mu1-mu2||^2 + ||sigma1-sigma2||^2
        """
        return (
            (
                pred_mu
                - gt_mu
            ).square().sum()
            + (
                pred_sigma
                - gt_sigma
            ).square().sum()
        )

    def _reference_debug(
        self,
        alignment_info: Dict,
    ) -> Dict:
        mapping = alignment_info.get(
            "scale_debug",
            {},
        )

        scale = int(
            self.gtga_reference_scale
        )

        if scale in mapping:
            return mapping[scale]

        if str(scale) in mapping:
            return mapping[str(scale)]

        raise KeyError(
            "GTGA reference debug missing "
            f"P{scale}; available={list(mapping.keys())}"
        )

    def _gtga_auxiliary_loss(
        self,
        batch: Dict,
        alignment_info: Dict,
    ):
        required = (
            "rgb_semantic_bboxes",
            "rgb_batch_idx",
            "tir_bboxes",
            "tir_batch_idx",
        )

        for key in required:
            if key not in batch:
                raise KeyError(
                    f"GTGA supervision requires "
                    f"batch['{key}']"
                )

        dbg = self._reference_debug(
            alignment_info
        )

        rgb_logits = dbg[
            "rgb_target_logits"
        ].float()

        tir_logits = dbg[
            "tir_target_logits"
        ].float()

        bsz, _, h, w = (
            rgb_logits.shape
        )

        rgb_mask = self._build_target_mask(
            boxes=batch[
                "rgb_semantic_bboxes"
            ],
            batch_idx=batch[
                "rgb_batch_idx"
            ],
            batch_size=bsz,
            h=h,
            w=w,
            device=rgb_logits.device,
            dtype=rgb_logits.dtype,
        )

        tir_mask = self._build_target_mask(
            boxes=batch[
                "tir_bboxes"
            ],
            batch_idx=batch[
                "tir_batch_idx"
            ],
            batch_size=bsz,
            h=h,
            w=w,
            device=tir_logits.device,
            dtype=tir_logits.dtype,
        )

        rgb_target_loss = self._focal_bce(
            rgb_logits,
            rgb_mask,
        )

        tir_target_loss = self._focal_bce(
            tir_logits,
            tir_mask,
        )

        target_map_loss = (
            0.5
            * (
                rgb_target_loss
                + tir_target_loss
            )
        )

        pred_r_mu = dbg[
            "rgb_mu"
        ].float()
        pred_t_mu = dbg[
            "tir_mu"
        ].float()

        pred_r_sigma = dbg[
            "rgb_sigma"
        ].float()
        pred_t_sigma = dbg[
            "tir_sigma"
        ].float()

        gaussian_losses = []
        center_mses = []
        sigma_mses = []

        valid_pairs = 0

        for bi in range(
            bsz
        ):
            rsel = (
                batch[
                    "rgb_batch_idx"
                ]
                == bi
            )

            tsel = (
                batch[
                    "tir_batch_idx"
                ]
                == bi
            )

            # Clean version: one global Gaussian geometry = one paired UAV.
            if (
                int(
                    rsel.sum().item()
                )
                != 1
                or int(
                    tsel.sum().item()
                )
                != 1
            ):
                continue

            rbox = (
                batch[
                    "rgb_semantic_bboxes"
                ][
                    rsel
                ][0]
                .float()
            )

            tbox = (
                batch[
                    "tir_bboxes"
                ][
                    tsel
                ][0]
                .float()
            )

            gt_r_mu = rbox[
                :2
            ]
            gt_t_mu = tbox[
                :2
            ]

            # Uniform axis-aligned rectangle:
            # std_x = w / sqrt(12), std_y = h / sqrt(12).
            gt_r_sigma = (
                rbox[2:4]
                / math.sqrt(12.0)
            )
            gt_t_sigma = (
                tbox[2:4]
                / math.sqrt(12.0)
            )

            wr = (
                self
                ._diag_gaussian_w2_sq(
                    pred_r_mu[bi],
                    pred_r_sigma[bi],
                    gt_r_mu,
                    gt_r_sigma,
                )
            )

            wt = (
                self
                ._diag_gaussian_w2_sq(
                    pred_t_mu[bi],
                    pred_t_sigma[bi],
                    gt_t_mu,
                    gt_t_sigma,
                )
            )

            gaussian_losses.append(
                0.5 * (wr + wt)
            )

            center_mses.append(
                0.5
                * (
                    (
                        pred_r_mu[bi]
                        - gt_r_mu
                    ).square().mean()
                    + (
                        pred_t_mu[bi]
                        - gt_t_mu
                    ).square().mean()
                )
            )

            sigma_mses.append(
                0.5
                * (
                    (
                        pred_r_sigma[bi]
                        - gt_r_sigma
                    ).square().mean()
                    + (
                        pred_t_sigma[bi]
                        - gt_t_sigma
                    ).square().mean()
                )
            )

            valid_pairs += 1

        zero = (
            rgb_logits.sum()
            * 0.0
        )

        gaussian_w2_loss = (
            torch.stack(
                gaussian_losses
            ).mean()
            if gaussian_losses
            else zero
        )

        center_mse = (
            torch.stack(
                center_mses
            ).mean()
            if center_mses
            else zero
        )

        sigma_mse = (
            torch.stack(
                sigma_mses
            ).mean()
            if sigma_mses
            else zero
        )

        gtga_loss = (
            self.lambda_target_map
            * target_map_loss
            + self.lambda_gaussian
            * gaussian_w2_loss
        )

        return {
            "gtga_loss":
                gtga_loss,
            "gtga_target_map_loss":
                target_map_loss,
            "gtga_rgb_target_loss":
                rgb_target_loss,
            "gtga_tir_target_loss":
                tir_target_loss,
            "gtga_gaussian_w2_loss":
                gaussian_w2_loss,
            "gtga_center_mse":
                center_mse,
            "gtga_sigma_mse":
                sigma_mse,
            "gtga_valid_pairs":
                valid_pairs,
        }

    @staticmethod
    def _loss_items_to_dict(
        loss_items,
    ):
        if isinstance(
            loss_items,
            dict,
        ):
            return dict(
                loss_items
            )

        if torch.is_tensor(
            loss_items
        ):
            values = list(
                loss_items.flatten()
            )
        elif isinstance(
            loss_items,
            (list, tuple),
        ):
            values = list(
                loss_items
            )
        else:
            return {
                "loss":
                    loss_items
            }

        if len(values) == 3:
            return {
                "box_loss":
                    values[0],
                "cls_loss":
                    values[1],
                "dfl_loss":
                    values[2],
            }

        return {
            f"loss_{i}":
                value
            for i, value
            in enumerate(values)
        }

    # =========================================================
    # Loss
    # =========================================================

    def loss(
        self,
        batch: Dict,
        preds=None,
    ):
        required = (
            "rgb_img",
            "rgb_semantic_img",
            "tir_img",
            "rgb_cls",
            "rgb_semantic_bboxes",
            "rgb_batch_idx",
        )

        for key in required:
            if key not in batch:
                raise KeyError(
                    f"GTGA+BHLR batch missing: {key}"
                )

        if self.criterion is None:
            self.criterion = (
                self.init_criterion()
            )

        # Explicit preds => detection-only criterion path.
        if preds is not None:
            yolo_batch = {
                "img":
                    batch[
                        "rgb_semantic_img"
                    ],
                "cls":
                    batch[
                        "rgb_cls"
                    ],
                "bboxes":
                    batch[
                        "rgb_semantic_bboxes"
                    ],
                "batch_idx":
                    batch[
                        "rgb_batch_idx"
                    ],
            }

            return self.criterion(
                preds,
                yolo_batch,
            )

        # Detection-only ablation.
        if not self.gtga_loss_enabled:
            predictions = self.predict(
                batch[
                    "rgb_img"
                ],
                batch[
                    "tir_img"
                ],
                rgb_semantic=(
                    batch[
                        "rgb_semantic_img"
                    ]
                ),
            )

            yolo_batch = {
                "img":
                    batch[
                        "rgb_semantic_img"
                    ],
                "cls":
                    batch[
                        "rgb_cls"
                    ],
                "bboxes":
                    batch[
                        "rgb_semantic_bboxes"
                    ],
                "batch_idx":
                    batch[
                        "rgb_batch_idx"
                    ],
            }

            return self.criterion(
                predictions,
                yolo_batch,
            )

        for key in (
            "tir_bboxes",
            "tir_batch_idx",
        ):
            if key not in batch:
                raise KeyError(
                    "GTGA geometry supervision "
                    f"requires batch['{key}']"
                )

        outputs = self.predict(
            batch[
                "rgb_img"
            ],
            batch[
                "tir_img"
            ],
            rgb_semantic=(
                batch[
                    "rgb_semantic_img"
                ]
            ),
            return_features=True,
            return_alignment_debug=True,
        )

        yolo_batch = {
            "img":
                batch[
                    "rgb_semantic_img"
                ],
            "cls":
                batch[
                    "rgb_cls"
                ],
            "bboxes":
                batch[
                    "rgb_semantic_bboxes"
                ],
            "batch_idx":
                batch[
                    "rgb_batch_idx"
                ],
        }

        (
            det_loss_raw,
            det_loss_items,
        ) = self.criterion(
            outputs[
                "pred"
            ],
            yolo_batch,
        )

        stats = self._gtga_auxiliary_loss(
            batch,
            outputs[
                "alignment_info"
            ],
        )

        batch_size = int(
            batch[
                "rgb_semantic_img"
            ].shape[0]
        )

        weighted_gtga_raw = (
            self.lambda_gtga
            * stats[
                "gtga_loss"
            ]
            * float(
                batch_size
            )
        )

        if not torch.is_tensor(
            det_loss_raw
        ):
            raise TypeError(
                "Expected tensor detection loss, "
                f"got {type(det_loss_raw)}"
            )

        # Same custom-loss convention as the previous RLSFA wrapper:
        # train_rgbt.py later performs loss_raw.sum().
        total_raw = torch.cat(
            (
                det_loss_raw.reshape(
                    -1
                ),
                weighted_gtga_raw.reshape(
                    1
                ),
            ),
            dim=0,
        )

        items = (
            self
            ._loss_items_to_dict(
                det_loss_items
            )
        )

        items.update(
            {
                "gtga_loss":
                    stats[
                        "gtga_loss"
                    ].detach(),
                "gtga_target_map_loss":
                    stats[
                        "gtga_target_map_loss"
                    ].detach(),
                "gtga_rgb_target_loss":
                    stats[
                        "gtga_rgb_target_loss"
                    ].detach(),
                "gtga_tir_target_loss":
                    stats[
                        "gtga_tir_target_loss"
                    ].detach(),
                "gtga_gaussian_w2_loss":
                    stats[
                        "gtga_gaussian_w2_loss"
                    ].detach(),
                "gtga_center_mse":
                    stats[
                        "gtga_center_mse"
                    ].detach(),
                "gtga_sigma_mse":
                    stats[
                        "gtga_sigma_mse"
                    ].detach(),
                "gtga_weighted_loss":
                    (
                        self.lambda_gtga
                        * stats[
                            "gtga_loss"
                        ]
                    ).detach(),
                "gtga_valid_pairs":
                    float(
                        stats[
                            "gtga_valid_pairs"
                        ]
                    ),
            }
        )

        return (
            total_raw,
            items,
        )

    # =========================================================
    # Info
    # =========================================================

    def print_info(
        self,
    ):
        super().print_info()

        print(
            "GTGA + BHLR-v2:"
        )
        print(
            f"  GTGA enabled       : "
            f"{self.alignment_enabled}"
        )
        print(
            f"  GTGA reference     : "
            f"P{self.gtga_reference_scale}"
        )
        print(
            f"  GTGA apply scales  : "
            f"{self.alignment_cfg.get('apply_scales', [3,4,5])}"
        )
        print(
            f"  GTGA loss          : "
            f"{self.gtga_loss_enabled}"
        )
        print(
            f"  BHLR enabled       : "
            f"{self.bhlr_enabled}"
        )
        print(
            f"  BHLR layers        : "
            f"{list(self.bhlr_layers)}"
        )
        print(
            f"  BHLR guidance      : "
            f"{self.bhlr_guidance_mode}"
        )

        if self.bhlr is not None:
            state = (
                self.bhlr
                .scalar_state()
            )
            print(
                f"  BHLR gamma         : "
                f"{state['gamma']:.6f}"
            )
