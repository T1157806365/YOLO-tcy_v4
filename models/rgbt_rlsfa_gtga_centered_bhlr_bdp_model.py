"""
Fixed-parameter RLSFA/GTGA + BHLR + ADD + BDP-Neck.

This wrapper deliberately inherits the existing fixed-parameter
RGBTRLSFAGTGACenteredBHLRV2DetectionModel without changing alignment,
BHLR, fusion, detection loss, or auxiliary alignment losses.

The ONLY new path is:

    D3 = P3_BHLR - P3_raw

    fused P3/P4/P5 -> original YOLO neck -> [N3, N4, N5]

    N3' = N3 + gamma * Gate(N3) * Proj(D3)

and N3' is supplied to the existing Detect layer.

The BDP residual gain is zero-initialized, so at initialization this model is
exactly the original neck behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.rgbt_model import get_layer_input
from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)
from models.modules.bdp_neck import BDPNeckP3


class RGBTRLSFAGTGACenteredBHLRBDPDetectionModel(
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel
):
    """Original fixed alignment/BHLR model + P3 detail-preserving neck bypass."""

    def __init__(
        self,
        *args,
        bdp_neck_cfg: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        # Keep every existing alignment/BHLR/fusion parameter unchanged.
        super().__init__(*args, verbose=verbose, **kwargs)

        cfg = dict(bdp_neck_cfg or {})
        self.bdp_neck_cfg = cfg
        self.bdp_neck_enabled = bool(cfg.get("enabled", True))
        self.bdp_scale = int(cfg.get("scale", 3))
        self.bdp_semantic_gate = bool(cfg.get("semantic_gate", True))
        self.bdp_max_gain = float(cfg.get("max_gain", 0.25))

        if self.bdp_scale != 3:
            raise ValueError(
                "BDP-Neck v1 is intentionally P3-only. "
                f"Set model.bdp_neck.scale: 3, got {self.bdp_scale}."
            )
        if self.bdp_scale not in self.scale_to_index:
            raise RuntimeError(
                f"P{self.bdp_scale} is not present in scale_to_index={self.scale_to_index}."
            )

        self.bdp_detail_layer = int(self.scale_to_index[self.bdp_scale])
        self.bdp_detect_slot = None
        self.bdp_neck = None

        if self.bdp_neck_enabled:
            detail_channels = int(
                self._infer_any_layer_channels((self.bdp_detail_layer,))[
                    self.bdp_detail_layer
                ]["rgb"]
            )
            neck_channels, detect_slot = self._infer_final_detect_p3_spec()
            self.bdp_detect_slot = int(detect_slot)
            self.bdp_neck = BDPNeckP3(
                detail_channels=detail_channels,
                neck_channels=neck_channels,
                semantic_gate=self.bdp_semantic_gate,
                max_gain=self.bdp_max_gain,
            )

            if verbose:
                print("\n" + "=" * 72)
                print("BDP-Neck v1 enabled")
                print("=" * 72)
                print(f"Detail source         : P{self.bdp_scale}")
                print(f"Backbone layer        : {self.bdp_detail_layer}")
                print(f"Detail channels       : {detail_channels}")
                print(f"Final Detect P3 slot  : {self.bdp_detect_slot}")
                print(f"Final Neck P3 channels: {neck_channels}")
                print(f"Semantic gate         : {self.bdp_semantic_gate}")
                print(f"Max residual gain     : {self.bdp_max_gain}")
                print("Initialization        : exact original YOLO neck (gain=0)")
                print("=" * 72 + "\n")

    @staticmethod
    def _highest_resolution_slot(features) -> int:
        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise TypeError(
                "YOLO Detect input must be a non-empty list/tuple of feature maps."
            )
        valid = []
        for i, x in enumerate(features):
            if not torch.is_tensor(x) or x.ndim != 4:
                raise TypeError(
                    f"Detect input[{i}] must be BCHW tensor, got {type(x)}"
                )
            valid.append((int(x.shape[-2]) * int(x.shape[-1]), i))
        # P3 is the largest spatial-resolution feature among P3/P4/P5.
        return max(valid, key=lambda z: z[0])[1]

    @torch.no_grad()
    def _infer_final_detect_p3_spec(self):
        """Infer final Neck-P3 channel width and its position in Detect input."""
        model_training = self.model.training
        self.model.eval()

        try:
            device = next(self.model.parameters()).device
            dummy = torch.zeros(1, 3, 256, 256, device=device)
            backbone_outputs = list(self.forward_rgb_backbone(dummy))

            outputs = list(backbone_outputs)
            x = outputs[self.backbone_len - 1]
            head_layers = list(self.model)[self.backbone_len:]
            if not head_layers:
                raise RuntimeError("YOLO model has no neck/head layers.")

            for head_i, module in enumerate(head_layers):
                module_input = get_layer_input(module, x, outputs)
                is_final_detect = head_i == len(head_layers) - 1
                if is_final_detect:
                    if not isinstance(module_input, (list, tuple)):
                        raise TypeError(
                            "Expected final Detect input to be list/tuple, "
                            f"got {type(module_input)}"
                        )
                    slot = self._highest_resolution_slot(module_input)
                    channels = int(module_input[slot].shape[1])
                    return channels, slot

                x = module(module_input)
                outputs.append(x)

            raise RuntimeError("Unable to locate final Detect input.")
        finally:
            self.model.train(model_training)

    def _forward_head_with_bdp(
        self,
        backbone_outputs,
        detail_p3: torch.Tensor,
        return_bdp_debug: bool = False,
    ):
        """Run the untouched YOLO neck, modify only final P3 before Detect."""
        if not self.bdp_neck_enabled or self.bdp_neck is None:
            pred = self.forward_head(backbone_outputs)
            return pred, None

        if len(backbone_outputs) != self.backbone_len:
            raise ValueError(
                f"backbone_outputs length {len(backbone_outputs)} != {self.backbone_len}"
            )

        outputs = list(backbone_outputs)
        x = outputs[self.backbone_len - 1]
        head_layers = list(self.model)[self.backbone_len:]

        bdp_debug = None
        for head_i, module in enumerate(head_layers):
            module_input = get_layer_input(module, x, outputs)
            is_final_detect = head_i == len(head_layers) - 1

            if is_final_detect:
                if not isinstance(module_input, (list, tuple)):
                    raise TypeError(
                        "Expected final Detect input to be list/tuple, "
                        f"got {type(module_input)}"
                    )

                detect_inputs = list(module_input)
                slot = self._highest_resolution_slot(detect_inputs)

                # Protect against architecture changes after initialization.
                if self.bdp_detect_slot is not None and slot != self.bdp_detect_slot:
                    raise RuntimeError(
                        "Final Detect P3 slot changed unexpectedly: "
                        f"init={self.bdp_detect_slot}, runtime={slot}."
                    )

                neck_p3 = detect_inputs[slot]
                if return_bdp_debug:
                    neck_p3_out, bdp_debug = self.bdp_neck(
                        detail_p3,
                        neck_p3,
                        return_debug=True,
                    )
                else:
                    neck_p3_out = self.bdp_neck(
                        detail_p3,
                        neck_p3,
                        return_debug=False,
                    )

                detect_inputs[slot] = neck_p3_out
                x = module(detect_inputs)
                outputs.append(x)
                return x, bdp_debug

            x = module(module_input)
            outputs.append(x)

        raise RuntimeError("Final Detect layer was not reached.")

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
        # Parent computes EXACT existing:
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

        p3_layer = self.bdp_detail_layer

        # IMPORTANT: this is computed BEFORE RGB-T ADD fusion.
        # Therefore D3 contains only the high-resolution RGB increment added
        # by BHLR, not TIR information.
        p3_bhlr = features["rgb_backbone"][p3_layer]
        p3_raw = features["rgb_backbone_raw"][p3_layer]
        detail_p3 = p3_bhlr - p3_raw

        features["bdp_detail_p3"] = detail_p3
        features["bdp_neck_enabled"] = self.bdp_neck_enabled
        features["bdp_scale"] = self.bdp_scale
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

        pred, bdp_debug = self._forward_head_with_bdp(
            features["fused_backbone"],
            features["bdp_detail_p3"],
            return_bdp_debug=return_bdp_debug,
        )

        if return_bdp_debug:
            features["bdp_debug"] = bdp_debug

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
        # Keep parent training API unchanged.
        if isinstance(rgb, dict):
            return self.loss(rgb)
        if tir is None:
            raise ValueError(
                "RLSFA-GTGA-Centered+BHLR+BDP requires both RGB and TIR."
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


__all__ = ["RGBTRLSFAGTGACenteredBHLRBDPDetectionModel"]
