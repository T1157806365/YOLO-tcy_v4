"""Figure Cat->Conv + Cat->Conv -> Add replacement for final ADD fusion.

Inherits the unchanged GMLS-v2, DMLS, Fine, Local-GTGA and BHLR-v2 model.
Does not use P3ProductResidual and does not call original ADD on replaced scales.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)
from models.modules.figure_catconv_add import FigureCatConvAddFusion


class RGBTFigureCatConvAddDetectionModel(RGBTRLSFAGTGACenteredBHLRV2DetectionModel):
    def __init__(self, figure_fusion_cfg: Optional[Dict[str, Any]] = None, **kwargs):
        # Base __init__ makes dummy forward_rgb_backbone() calls for shape
        # inference. The overrides below tolerate those calls.
        super().__init__(**kwargs)
        cfg = dict(figure_fusion_cfg or {})
        if not bool(cfg.get("enabled", True)):
            raise ValueError("Figure fusion is disabled; use the original baseline variant")
        if str(self.fusion_type).lower() != "add":
            raise ValueError("Use fusion: add for compatibility with native fusion API")

        scales = sorted({int(x) for x in cfg.get("scales", [3, 4, 5])})
        if not scales or any(s not in self.scale_to_index for s in scales):
            raise ValueError(f"Figure fusion scales must be non-empty subset of P3/P4/P5, got {scales}")
        self.figure_scales = tuple(scales)
        self.figure_layers = frozenset(self.scale_to_index[s] for s in scales)
        self.figure_fusion_cfg = cfg
        spatial_kernel = int(cfg.get("spatial_kernel", 3))
        channel_groups = int(cfg.get("channel_groups", 16))
        ch = self._infer_any_layer_channels(tuple(self.figure_layers))

        # REPLACE the actual ADD modules in the same ModuleDict (do not add a
        # separate parallel branch). This prevents unused ADD trainable params.
        for s in self.figure_scales:
            layer = self.scale_to_index[s]
            self.fusions[str(layer)] = FigureCatConvAddFusion(
                rgb_channels=ch[layer]["rgb"],
                tir_channels=ch[layer]["tir"],
                spatial_kernel=spatial_kernel,
                channel_groups=channel_groups,
            )
        self._figure_raw_rgb = None
        self._figure_raw_tir = None
        print(
            "[Figure ConvAdd REPLACE] scales={} layers={} kernel={} "
            "channel_groups={} (NO product, NO residual)".format(
                self.figure_scales, sorted(self.figure_layers),
                spatial_kernel, channel_groups,
            ), flush=True,
        )

    def forward_rgb_backbone(self, rgb):
        out = super().forward_rgb_backbone(rgb)
        self._figure_raw_rgb = out
        return out

    def forward_tir_backbone(self, tir):
        out = super().forward_tir_backbone(tir)
        self._figure_raw_tir = out
        return out

    def fuse_backbone_features(self, rgb_outputs, tir_outputs):
        # Called after GTGA and BHLR in the unchanged parent forward_features.
        # Cached originals originate from the current forward_backbone calls.
        raw_rgb = getattr(self, "_figure_raw_rgb", None)
        raw_tir = getattr(self, "_figure_raw_tir", None)
        if raw_rgb is None or raw_tir is None:
            raise RuntimeError("Missing raw backbone tensors for Figure fusion")
        fused_outputs = list(rgb_outputs)
        fusion_features = {}
        for index in self.fusion_indices:
            if index in self.figure_layers:
                fused = self.fusions[str(index)](
                    raw_rgb[index], rgb_outputs[index],
                    raw_tir[index], tir_outputs[index],
                )
            else:
                fused = self.fusions[str(index)](rgb_outputs[index], tir_outputs[index])
            fused_outputs[index] = fused
            fusion_features[index] = fused
        # Drop only the subclass's extra references; original graph is held by
        # parent's rgb_outputs/raw_rgb and forwarded as normal.
        self._figure_raw_rgb = None
        self._figure_raw_tir = None
        return fused_outputs, fusion_features


__all__ = ["RGBTFigureCatConvAddDetectionModel"]
