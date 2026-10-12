"""
Fixed-parameter RLSFA/GTGA + BHLR + ADD + BDP-Neck v2.

This version keeps the proven fixed alignment/BHLR/fusion pipeline unchanged
and changes ONLY the neck-side use of BHLR detail.

Difference from BDP-Neck v1
---------------------------
v1:
    native YOLO neck -> final P3 -> add detail -> Detect

v2:
    native top-down P3 concat -> add gated detail -> native P3 C3k2 refine
        -> PAN bottom-up path -> Detect

Thus the BHLR detail is semantically refined by the original YOLO neck before
it reaches Detect.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.rgbt_model import get_layer_input
from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)
from models.modules.bdp_neck_v2 import BDPNeckInternalP3


class RGBTRLSFAGTGACenteredBHLRBDPNeckV2DetectionModel(
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel
):
    """Fixed alignment/BHLR/ADD model + internal P3 detail-preserving neck."""

    def __init__(
        self,
        *args,
        bdp_neck_v2_cfg: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        # Parent remains the exact fixed-parameter baseline:
        # alignment -> BHLR -> ADD -> original YOLO neck/head.
        super().__init__(*args, verbose=verbose, **kwargs)

        cfg = dict(bdp_neck_v2_cfg or {})
        self.bdp_neck_v2_cfg = cfg
        self.bdp_neck_v2_enabled = bool(cfg.get("enabled", True))
        self.bdp_neck_v2_scale = int(cfg.get("scale", 3))
        self.bdp_neck_v2_hidden_channels = int(cfg.get("hidden_channels", 64))
        self.bdp_neck_v2_gate_enabled = bool(cfg.get("gate_enabled", True))
        self.bdp_neck_v2_max_gain = float(cfg.get("max_gain", 0.25))

        if self.bdp_neck_v2_scale != 3:
            raise ValueError(
                "BDP-Neck v2 is intentionally P3-only. "
                f"Set scale=3, got {self.bdp_neck_v2_scale}."
            )
        if self.bdp_neck_v2_scale not in self.scale_to_index:
            raise RuntimeError(
                f"P{self.bdp_neck_v2_scale} is unavailable in "
                f"scale_to_index={self.scale_to_index}."
            )

        self.bdp_detail_layer = int(
            self.scale_to_index[self.bdp_neck_v2_scale]
        )
        self.bdp_p3_refine_index = None
        self.bdp_neck_v2 = None

        if self.bdp_neck_v2_enabled:
            detail_channels = int(
                self._infer_any_layer_channels((self.bdp_detail_layer,))[
                    self.bdp_detail_layer
                ]["rgb"]
            )

            (
                p3_refine_index,
                neck_pre_channels,
            ) = self._infer_internal_p3_refine_spec()

            self.bdp_p3_refine_index = int(p3_refine_index)
            self.bdp_neck_v2 = BDPNeckInternalP3(
                detail_channels=detail_channels,
                neck_channels=neck_pre_channels,
                hidden_channels=self.bdp_neck_v2_hidden_channels,
                gate_enabled=self.bdp_neck_v2_gate_enabled,
                max_gain=self.bdp_neck_v2_max_gain,
            )

            if verbose:
                print("\n" + "=" * 72)
                print("BDP-Neck v2: internal P3 detail fusion enabled")
                print("=" * 72)
                print(f"Detail source          : P{self.bdp_neck_v2_scale}")
                print(f"Backbone detail layer  : {self.bdp_detail_layer}")
                print(f"Detail channels        : {detail_channels}")
                print(f"P3 refine layer index  : {self.bdp_p3_refine_index}")
                print(f"P3 pre-refine channels : {neck_pre_channels}")
                print(f"Hidden channels        : {self.bdp_neck_v2_hidden_channels}")
                print(f"Detail gate            : {self.bdp_neck_v2_gate_enabled}")
                print(f"Max residual gain      : {self.bdp_neck_v2_max_gain}")
                print("Injection position     : BEFORE native P3 C3k2 refinement")
                print("Initialization         : exact original YOLO neck (gain=0)")
                print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Infer the REAL P3 refinement location from the YOLO graph instead
    # of hard-coding layer 16.  For the current YOLO26 P3/P4/P5 model this
    # resolves to layer 16, whose input is the top-down P3 concat feature.
    # ------------------------------------------------------------------
    @staticmethod
    def _highest_resolution_slot(features) -> int:
        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise TypeError("Detect input must be a non-empty list/tuple.")

        candidates = []
        for slot, feat in enumerate(features):
            if not torch.is_tensor(feat) or feat.ndim != 4:
                raise TypeError(
                    f"Detect input[{slot}] must be BCHW tensor, got {type(feat)}"
                )
            area = int(feat.shape[-2]) * int(feat.shape[-1])
            candidates.append((area, slot))
        return max(candidates, key=lambda x: x[0])[1]

    @torch.no_grad()
    def _infer_internal_p3_refine_spec(self):
        """
        Return:
            p3_refine_global_index,
            channels of its INPUT feature.

        The final Detect layer tells us which previous global layer is P3.
        We identify the highest-resolution Detect input, then rerun the native
        neck until that P3-producing layer and inspect its module input.
        """
        model_training = self.model.training
        self.model.eval()

        try:
            device = next(self.model.parameters()).device
            dummy = torch.zeros(1, 3, 256, 256, device=device)
            backbone_outputs = list(self.forward_rgb_backbone(dummy))

            # Pass 1: locate the highest-resolution source of final Detect.
            outputs = list(backbone_outputs)
            x = outputs[self.backbone_len - 1]
            head_layers = list(self.model)[self.backbone_len:]
            if not head_layers:
                raise RuntimeError("YOLO model has no neck/head layers.")

            detect_p3_source = None
            for local_i, module in enumerate(head_layers):
                global_i = self.backbone_len + local_i
                module_input = get_layer_input(module, x, outputs)
                is_final = local_i == len(head_layers) - 1

                if is_final:
                    if not isinstance(module_input, (list, tuple)):
                        raise TypeError(
                            "Expected final Detect input to be list/tuple, "
                            f"got {type(module_input)}"
                        )
                    slot = self._highest_resolution_slot(module_input)
                    f = module.f
                    refs = [f] if isinstance(f, int) else list(f)
                    if slot >= len(refs):
                        raise RuntimeError(
                            f"Detect slot {slot} exceeds module.f={refs}."
                        )
                    detect_p3_source = int(refs[slot])
                    break

                x = module(module_input)
                outputs.append(x)

            if detect_p3_source is None:
                raise RuntimeError("Unable to identify final Detect P3 source.")
            if detect_p3_source < self.backbone_len:
                raise RuntimeError(
                    "Detected P3 source belongs to backbone, expected neck layer: "
                    f"index={detect_p3_source}, backbone_len={self.backbone_len}."
                )

            # Pass 2: capture the INPUT to that native P3 refinement layer.
            outputs = list(backbone_outputs)
            x = outputs[self.backbone_len - 1]
            for local_i, module in enumerate(head_layers):
                global_i = self.backbone_len + local_i
                module_input = get_layer_input(module, x, outputs)

                if global_i == detect_p3_source:
                    if not torch.is_tensor(module_input) or module_input.ndim != 4:
                        raise TypeError(
                            "P3 refinement input should be BCHW tensor, got "
                            f"{type(module_input)}"
                        )
                    return detect_p3_source, int(module_input.shape[1])

                # We never need to execute final Detect in this second pass.
                if local_i == len(head_layers) - 1:
                    break
                x = module(module_input)
                outputs.append(x)

            raise RuntimeError(
                f"Unable to reach P3 refinement layer {detect_p3_source}."
            )
        finally:
            self.model.train(model_training)

    def _forward_head_with_internal_bdp(
        self,
        backbone_outputs,
        detail_p3: torch.Tensor,
        return_bdp_debug: bool = False,
    ):
        """Run native YOLO neck/head, injecting detail BEFORE P3 refinement."""
        if not self.bdp_neck_v2_enabled or self.bdp_neck_v2 is None:
            return self.forward_head(backbone_outputs), None

        if len(backbone_outputs) != self.backbone_len:
            raise ValueError(
                f"backbone_outputs length={len(backbone_outputs)} "
                f"!= backbone_len={self.backbone_len}"
            )

        outputs = list(backbone_outputs)
        x = outputs[self.backbone_len - 1]
        head_layers = list(self.model)[self.backbone_len:]
        bdp_debug = None
        injected = False

        for local_i, module in enumerate(head_layers):
            global_i = self.backbone_len + local_i
            module_input = get_layer_input(module, x, outputs)

            if global_i == self.bdp_p3_refine_index:
                if not torch.is_tensor(module_input):
                    raise TypeError(
                        "Internal P3 refinement input must be Tensor, got "
                        f"{type(module_input)}"
                    )

                if return_bdp_debug:
                    module_input, bdp_debug = self.bdp_neck_v2(
                        detail_p3,
                        module_input,
                        return_debug=True,
                    )
                else:
                    module_input = self.bdp_neck_v2(
                        detail_p3,
                        module_input,
                        return_debug=False,
                    )
                injected = True

            x = module(module_input)
            outputs.append(x)

        if not injected:
            raise RuntimeError(
                "BDP-Neck v2 did not reach configured P3 refinement layer "
                f"{self.bdp_p3_refine_index}."
            )

        return x, bdp_debug

    def forward_features(
        self,
        rgb,
        tir,
        rgb_semantic=None,
        rgb_valid_mask=None,
        tir_valid_mask=None,
        return_bhlr_debug=False,
        return_alignment_debug=False,
    ):
        # Parent computes the exact baseline path:
        # fixed alignment -> BHLR -> ADD fusion.
        features = super().forward_features(
            rgb=rgb,
            tir=tir,
            rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask,
            tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )

        # D3 is deliberately measured BEFORE RGB-T ADD fusion.
        # It therefore contains only the BHLR high-resolution RGB increment.
        p3_layer = self.bdp_detail_layer
        p3_bhlr = features["rgb_backbone"][p3_layer]
        p3_raw = features["rgb_backbone_raw"][p3_layer]
        detail_p3 = p3_bhlr - p3_raw

        features["bdp_detail_p3"] = detail_p3
        features["bdp_neck_v2_enabled"] = self.bdp_neck_v2_enabled
        features["bdp_neck_v2_scale"] = self.bdp_neck_v2_scale
        features["bdp_p3_refine_index"] = self.bdp_p3_refine_index
        return features

    def predict(
        self,
        rgb,
        tir,
        rgb_semantic=None,
        rgb_valid_mask=None,
        tir_valid_mask=None,
        return_features=False,
        return_bhlr_debug=False,
        return_alignment_debug=False,
        return_bdp_debug=False,
    ):
        features = self.forward_features(
            rgb=rgb,
            tir=tir,
            rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask,
            tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )

        pred, bdp_debug = self._forward_head_with_internal_bdp(
            features["fused_backbone"],
            features["bdp_detail_p3"],
            return_bdp_debug=return_bdp_debug,
        )

        if return_bdp_debug:
            features["bdp_neck_v2_debug"] = bdp_debug

        if (
            return_features
            or return_bhlr_debug
            or return_alignment_debug
            or return_bdp_debug
        ):
            return {"pred": pred, **features}
        return pred

    def forward(
        self,
        rgb,
        tir=None,
        rgb_semantic=None,
        return_features=False,
        return_bhlr_debug=False,
        return_alignment_debug=False,
        return_bdp_debug=False,
    ):
        # Preserve the baseline training API.  The inherited loss() calls this
        # class's predict(), so the new neck is active in both train and val.
        if isinstance(rgb, dict):
            return self.loss(rgb)
        if tir is None:
            raise ValueError(
                "RLSFA-GTGA-Centered+BHLR+BDP-Neck-v2 requires RGB and TIR."
            )
        return self.predict(
            rgb=rgb,
            tir=tir,
            rgb_semantic=rgb_semantic,
            return_features=return_features,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
            return_bdp_debug=return_bdp_debug,
        )


__all__ = ["RGBTRLSFAGTGACenteredBHLRBDPNeckV2DetectionModel"]
