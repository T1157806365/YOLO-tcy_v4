"""
Fixed RLSFA/GTGA + BHLR-v2 + ADD + RODG-v1.

Only the P3 bbox-regression branch receives high-resolution RGB detail.
Classification and P4/P5 remain unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.rgbt_model import get_layer_input
from models.rgbt_rlsfa_gtga_centered_bhlr_model import RGBTRLSFAGTGACenteredBHLRV2DetectionModel
from models.modules.rodg import RegressionOrientedDetailGuidance


class RGBTRLSFAGTGACenteredBHLRRODGV1DetectionModel(RGBTRLSFAGTGACenteredBHLRV2DetectionModel):
    def __init__(self, *args, rodg_cfg: Optional[Dict[str, Any]] = None, verbose: bool = True, **kwargs):
        super().__init__(*args, verbose=verbose, **kwargs)

        cfg = dict(rodg_cfg or {})
        self.rodg_cfg = cfg
        self.rodg_enabled = bool(cfg.get("enabled", True))
        self.rodg_scale = int(cfg.get("scale", 3))
        self.rodg_hidden_channels = int(cfg.get("hidden_channels", 32))
        self.rodg_spatial_gate = bool(cfg.get("spatial_gate", True))
        self.rodg_max_gain = float(cfg.get("max_gain", 0.25))
        self.rodg_regression_only = bool(cfg.get("regression_only", True))
        self.rodg_classification_unchanged = bool(cfg.get("classification_unchanged", True))

        if self.rodg_scale != 3:
            raise ValueError(f"RODG-v1 is P3-only; got scale={self.rodg_scale}")
        if not self.rodg_regression_only:
            raise ValueError("RODG-v1 requires regression_only: true")
        if not self.rodg_classification_unchanged:
            raise ValueError("RODG-v1 requires classification_unchanged: true")
        if self.rodg_scale not in self.scale_to_index:
            raise RuntimeError(f"P3 unavailable in scale_to_index={self.scale_to_index}")

        self.rodg_detail_layer = int(self.scale_to_index[self.rodg_scale])
        self.rodg_detect_index = None
        self.rodg_p3_slot = None
        self.rodg = None

        if self.rodg_enabled:
            detail_channels = int(
                self._infer_any_layer_channels((self.rodg_detail_layer,))[self.rodg_detail_layer]["rgb"]
            )
            detect_index, p3_slot, neck_channels = self._infer_detect_p3_spec()
            self.rodg_detect_index = int(detect_index)
            self.rodg_p3_slot = int(p3_slot)
            self.rodg = RegressionOrientedDetailGuidance(
                detail_channels=detail_channels,
                neck_channels=neck_channels,
                hidden_channels=self.rodg_hidden_channels,
                spatial_gate=self.rodg_spatial_gate,
                max_gain=self.rodg_max_gain,
            )

            if verbose:
                detect = self.model[self.rodg_detect_index]
                api = "dict-forward-head" if callable(getattr(detect, "forward_head", None)) else "legacy-list"
                print("\n" + "=" * 76)
                print("RODG-v1: Regression-Oriented Detail Guidance")
                print("=" * 76)
                print(f"Detail source            : P3_BHLR - P3_raw")
                print(f"Backbone detail layer    : {self.rodg_detail_layer}")
                print(f"Detect global index      : {self.rodg_detect_index}")
                print(f"Detect P3 slot           : {self.rodg_p3_slot}")
                print(f"P3 neck channels         : {neck_channels}")
                print(f"Detail channels          : {detail_channels}")
                print(f"Hidden channels          : {self.rodg_hidden_channels}")
                print(f"Spatial reliability gate : {self.rodg_spatial_gate}")
                print(f"Max signed gain          : {self.rodg_max_gain}")
                print("Classification path      : unchanged")
                print("Regression path          : RODG only on P3")
                print("P4/P5                     : unchanged")
                print("Auxiliary RODG loss       : NONE")
                print("Initialization            : exact baseline (gain=0)")
                print(f"Detect compatibility API : {api}")
                print("=" * 76 + "\n")

    @staticmethod
    def _highest_resolution_slot(features) -> int:
        if not isinstance(features, (list, tuple)) or not features:
            raise TypeError("Detect input must be non-empty list/tuple")
        candidates = []
        for slot, feat in enumerate(features):
            if not torch.is_tensor(feat) or feat.ndim != 4:
                raise TypeError(f"Detect input[{slot}] is not BCHW tensor")
            candidates.append((int(feat.shape[-2]) * int(feat.shape[-1]), slot))
        return max(candidates, key=lambda x: x[0])[1]

    @torch.no_grad()
    def _infer_detect_p3_spec(self):
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
                raise RuntimeError("YOLO model has no neck/head layers")

            for local_i, module in enumerate(head_layers):
                global_i = self.backbone_len + local_i
                module_input = get_layer_input(module, x, outputs)
                if local_i == len(head_layers) - 1:
                    if not isinstance(module_input, (list, tuple)):
                        raise TypeError(f"Final Detect input must be list/tuple, got {type(module_input)}")
                    p3_slot = self._highest_resolution_slot(module_input)
                    p3 = module_input[p3_slot]
                    required = ("cv2", "cv3", "nl", "nc", "reg_max")
                    if not all(hasattr(module, k) for k in required):
                        raise TypeError("Final head is not Detect-compatible (missing cv2/cv3/nl/nc/reg_max)")
                    if int(module.nl) != len(module_input):
                        raise RuntimeError(f"Detect nl={module.nl}, inputs={len(module_input)}")
                    return global_i, p3_slot, int(p3.shape[1])
                x = module(module_input)
                outputs.append(x)
            raise RuntimeError("Unable to locate final Detect head")
        finally:
            self.model.train(model_training)

    @staticmethod
    def _postprocess_end2end(detect, y: torch.Tensor) -> torch.Tensor:
        try:
            return detect.postprocess(y.permute(0, 2, 1))
        except TypeError:
            return detect.postprocess(y.permute(0, 2, 1), detect.max_det, detect.nc)

    @staticmethod
    def _new_api_raw_head(detect, reg_feats, cls_feats, box_head, cls_head):
        if box_head is None or cls_head is None:
            return {}
        bs = cls_feats[0].shape[0]
        boxes = torch.cat(
            [box_head[i](reg_feats[i]).view(bs, 4 * detect.reg_max, -1) for i in range(detect.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head[i](cls_feats[i]).view(bs, detect.nc, -1) for i in range(detect.nl)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": cls_feats}

    def _forward_detect_new_api(self, detect, cls_feats, reg_feats):
        one2many = self._new_api_raw_head(detect, reg_feats, cls_feats, detect.cv2, detect.cv3)
        preds = one2many

        has_one2one = (
            getattr(detect, "one2one_cv2", None) is not None
            and getattr(detect, "one2one_cv3", None) is not None
        )
        end2end = bool(getattr(detect, "end2end", False)) and has_one2one

        if end2end:
            cls_detach = [x.detach() for x in cls_feats]
            reg_detach = [x.detach() for x in reg_feats]
            one2one = self._new_api_raw_head(
                detect, reg_detach, cls_detach, detect.one2one_cv2, detect.one2one_cv3
            )
            preds = {"one2many": one2many, "one2one": one2one}

        if detect.training:
            return preds

        inference_raw = preds["one2one"] if end2end else preds
        y = detect._inference(inference_raw)
        if end2end:
            y = self._postprocess_end2end(detect, y)
        return y if detect.export else (y, preds)

    @staticmethod
    def _legacy_combined_head(detect, reg_feats, cls_feats, box_head, cls_head):
        return [
            torch.cat((box_head[i](reg_feats[i]), cls_head[i](cls_feats[i])), dim=1)
            for i in range(detect.nl)
        ]

    def _forward_detect_legacy_api(self, detect, cls_feats, reg_feats):
        has_one2one = (
            getattr(detect, "one2one_cv2", None) is not None
            and getattr(detect, "one2one_cv3", None) is not None
        )
        end2end = bool(getattr(detect, "end2end", False)) and has_one2one

        one2many = self._legacy_combined_head(detect, reg_feats, cls_feats, detect.cv2, detect.cv3)
        if end2end:
            cls_detach = [x.detach() for x in cls_feats]
            reg_detach = [x.detach() for x in reg_feats]
            one2one = self._legacy_combined_head(
                detect, reg_detach, cls_detach, detect.one2one_cv2, detect.one2one_cv3
            )
            preds = {"one2many": one2many, "one2one": one2one}
            if detect.training:
                return preds
            y = detect._inference(one2one)
            y = self._postprocess_end2end(detect, y)
            return y if detect.export else (y, preds)

        if detect.training:
            return one2many
        y = detect._inference(one2many)
        return y if detect.export else (y, one2many)

    def _forward_detect_regression_only(self, detect, detect_inputs, detail_p3, return_rodg_debug=False):
        if not isinstance(detect_inputs, (list, tuple)):
            raise TypeError(f"Detect inputs must be list/tuple, got {type(detect_inputs)}")
        cls_feats = list(detect_inputs)
        if len(cls_feats) != int(detect.nl):
            raise RuntimeError(f"Detect got {len(cls_feats)} levels, expected nl={detect.nl}")

        p3_slot = int(self.rodg_p3_slot)
        baseline_p3 = cls_feats[p3_slot]
        if return_rodg_debug:
            enhanced_p3, rodg_debug = self.rodg(detail_p3, baseline_p3, return_debug=True)
        else:
            enhanced_p3 = self.rodg(detail_p3, baseline_p3, return_debug=False)
            rodg_debug = None

        reg_feats = list(cls_feats)
        reg_feats[p3_slot] = enhanced_p3

        if callable(getattr(detect, "forward_head", None)):
            pred = self._forward_detect_new_api(detect, cls_feats, reg_feats)
        else:
            pred = self._forward_detect_legacy_api(detect, cls_feats, reg_feats)

        if return_rodg_debug and rodg_debug is not None:
            rodg_debug["p3_cls_feature"] = baseline_p3.detach()
            rodg_debug["p3_reg_feature"] = enhanced_p3.detach()
            rodg_debug["p3_reg_delta"] = (enhanced_p3 - baseline_p3).detach()
            rodg_debug["p3_slot"] = p3_slot
        return pred, rodg_debug

    def _forward_head_with_rodg(self, backbone_outputs, detail_p3, return_rodg_debug=False):
        if not self.rodg_enabled or self.rodg is None:
            return self.forward_head(backbone_outputs), None
        if len(backbone_outputs) != self.backbone_len:
            raise ValueError(f"backbone_outputs length={len(backbone_outputs)} != {self.backbone_len}")

        outputs = list(backbone_outputs)
        x = outputs[self.backbone_len - 1]
        used = False
        rodg_debug = None
        for local_i, module in enumerate(list(self.model)[self.backbone_len:]):
            global_i = self.backbone_len + local_i
            module_input = get_layer_input(module, x, outputs)
            if global_i == self.rodg_detect_index:
                x, rodg_debug = self._forward_detect_regression_only(
                    module, module_input, detail_p3, return_rodg_debug=return_rodg_debug
                )
                used = True
            else:
                x = module(module_input)
            outputs.append(x)

        if not used:
            raise RuntimeError(f"RODG did not reach Detect layer {self.rodg_detect_index}")
        return x, rodg_debug

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
        features = super().forward_features(
            rgb=rgb,
            tir=tir,
            rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask,
            tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )
        p3_layer = self.rodg_detail_layer
        features["rodg_detail_p3"] = (
            features["rgb_backbone"][p3_layer] - features["rgb_backbone_raw"][p3_layer]
        )
        features["rodg_enabled"] = self.rodg_enabled
        features["rodg_scale"] = self.rodg_scale
        features["rodg_detect_index"] = self.rodg_detect_index
        features["rodg_p3_slot"] = self.rodg_p3_slot
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
        return_rodg_debug=False,
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
        pred, rodg_debug = self._forward_head_with_rodg(
            features["fused_backbone"], features["rodg_detail_p3"], return_rodg_debug=return_rodg_debug
        )
        if return_rodg_debug:
            features["rodg_debug"] = rodg_debug
        if return_features or return_bhlr_debug or return_alignment_debug or return_rodg_debug:
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
        return_rodg_debug=False,
    ):
        if isinstance(rgb, dict):
            return self.loss(rgb)
        if tir is None:
            raise ValueError("RLSFA-GTGA-Centered+BHLR+RODG requires both RGB and TIR")
        return self.predict(
            rgb=rgb,
            tir=tir,
            rgb_semantic=rgb_semantic,
            return_features=return_features,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
            return_rodg_debug=return_rodg_debug,
        )


__all__ = ["RGBTRLSFAGTGACenteredBHLRRODGV1DetectionModel"]
