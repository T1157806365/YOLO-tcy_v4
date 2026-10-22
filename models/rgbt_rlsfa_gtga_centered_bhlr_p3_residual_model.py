#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drop-in model variant for backup-20261021_best.

Keeps all original RGB/TIR backbones, GMLS v2, DMLS correction, fine
translation, GTGA scale, BHLR-v2, P4/P5 ADD fusion and native YOLO head.
Only changes the P3 feature after existing ADD fusion.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)
from models.modules.p3_product_residual import P3ProductResidual


class RGBTP3ProductResidualDetectionModel(
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel
):
    def __init__(
        self,
        p3_residual_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = dict(p3_residual_cfg or {})
        if not bool(cfg.get("enabled", True)):
            raise ValueError(
                "P3 residual variant requires model.p3_residual.enabled: true; "
                "for baseline use variant: rlsfa_gtga_centered_bhlr_v2"
            )
        if str(self.fusion_type).lower() != "add":
            raise ValueError("P3 residual variant requires model.fusion: add")

        self.p3_residual_index = int(self.scale_to_index[3])
        channels = self._infer_any_layer_channels((self.p3_residual_index,))
        rgb_ch = int(channels[self.p3_residual_index]["rgb"])
        tir_ch = int(channels[self.p3_residual_index]["tir"])

        self.p3_residual = P3ProductResidual(
            rgb_channels=rgb_ch,
            tir_channels=tir_ch,
            hidden_channels=int(cfg.get("hidden_channels", 32)),
            mode=str(cfg.get("mode", "product")),
            max_gain=float(cfg.get("max_gain", 0.25)),
        )
        self.p3_residual_cfg = cfg
        print(
            "[P3 residual] mode={mode} hidden={hidden} max_gain={gain} "
            "channels=({rgb},{tir}) layer={layer}".format(
                mode=self.p3_residual.mode,
                hidden=self.p3_residual.hidden_channels,
                gain=self.p3_residual.max_gain,
                rgb=rgb_ch,
                tir=tir_ch,
                layer=self.p3_residual_index,
            ),
            flush=True,
        )

    def fuse_backbone_features(self, rgb_outputs, tir_outputs):
        # The original ADD fusion runs unchanged at P3/P4/P5.
        fused_outputs, fusion_features = super().fuse_backbone_features(
            rgb_outputs, tir_outputs
        )
        p3 = self.p3_residual_index
        # rgb_outputs[p3] has already been enhanced by BHLR-v2.
        # tir_outputs[p3] has already been geometrically aligned.
        # For baseline-preserving init, gamma_raw == 0.
        new_p3 = self.p3_residual(
            fused_add=fused_outputs[p3],
            rgb_p3=rgb_outputs[p3],
            tir_p3_aligned=tir_outputs[p3],
        )
        fused_outputs[p3] = new_p3
        fusion_features[p3] = new_p3
        return fused_outputs, fusion_features


__all__ = ["RGBTP3ProductResidualDetectionModel"]
