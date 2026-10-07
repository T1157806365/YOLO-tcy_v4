"""
RLSFA/Local-GTGA (TIR -> RGB) + BHLR-v2 + DUCF detector.

This wrapper keeps the current YOLO-tcy-v3 geometry and BHLR pipeline intact:

    RGB semantic -> RGB backbone ------------------------------+
                                                               |
    TIR -> TIR backbone -> RLSFA/GTGA -> aligned TIR on RGB ---+--> BHLR RGB
                                                               |
                                                               +--> DUCF
                                                                    |
                                                               YOLO neck/head

Only the final RGB-T fusion operator changes.

DUCF is NOT a mixture-of-experts model. There is one fused backbone feature
pyramid and one YOLO neck/head. RGB/TIR utility is estimated inside feature
fusion with a shared, symmetric scorer and independent non-softmax gates.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)
from models.modules.ducf import DUCFFusion


class RGBTRLSFAGTGACenteredBHLRDUCFDetectionModel(
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel
):
    """Current TIR->RGB alignment + BHLR-v2 + DUCF fusion."""

    def __init__(
        self,
        *args,
        ducf_cfg: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        # Build the proven current model first. For scales not handled by DUCF,
        # the original fusion remains active (normally ADD at P5).
        super().__init__(*args, verbose=verbose, **kwargs)

        cfg = dict(ducf_cfg or {})
        self.ducf_cfg = cfg
        self.ducf_enabled = bool(cfg.get("enabled", True))

        scales = tuple(sorted({int(s) for s in cfg.get("scales", [3, 4])}))
        invalid = [s for s in scales if s not in self.scale_to_index]
        if invalid:
            raise ValueError(
                f"DUCF scales must be conceptual P3/P4/P5, got invalid={invalid}"
            )
        self.ducf_scales = scales if self.ducf_enabled else tuple()
        self.ducf_layers = tuple(self.scale_to_index[s] for s in self.ducf_scales)
        self.ducf_index_to_scale = {
            self.scale_to_index[s]: s for s in self.ducf_scales
        }

        self.ducf_use_target_support = bool(
            cfg.get("use_target_support", True)
        )
        self.ducf_prediction_loss_weight = float(
            cfg.get("prediction_loss_weight", 0.05)
        )

        self.ducf_fusions = nn.ModuleDict()

        if self.ducf_enabled:
            channels = self._infer_any_layer_channels(self.ducf_layers)

            hidden = int(cfg.get("hidden_channels", 64))
            predictor_aux_enabled = bool(
                cfg.get("predictor_aux_enabled", True)
            )
            consensus_gain_init = float(
                cfg.get("consensus_gain_init", 0.0)
            )
            innovation_gain_init = float(
                cfg.get("innovation_gain_init", 0.0)
            )

            for scale in self.ducf_scales:
                layer_id = self.scale_to_index[scale]
                rgb_c = int(channels[layer_id]["rgb"])
                tir_c = int(channels[layer_id]["tir"])

                if rgb_c != tir_c:
                    raise RuntimeError(
                        "Strict symmetric DUCF requires equal RGB/TIR channels. "
                        f"P{scale}/layer{layer_id}: RGB={rgb_c}, TIR={tir_c}."
                    )

                self.ducf_fusions[str(layer_id)] = DUCFFusion(
                    rgb_channels=rgb_c,
                    tir_channels=tir_c,
                    hidden_channels=hidden,
                    use_target_support=self.ducf_use_target_support,
                    predictor_aux_enabled=predictor_aux_enabled,
                    consensus_gain_init=consensus_gain_init,
                    innovation_gain_init=innovation_gain_init,
                )

                # Parent built one ordinary fusion for every fusion index.
                # Remove the modules replaced by DUCF so DDP does not see
                # permanently-unused trainable parameters.
                key = str(layer_id)
                if key in self.fusions:
                    del self.fusions[key]

        if verbose:
            print("\n" + "=" * 76)
            print("DUCF: Detection-Utility Complementary Fusion")
            print("=" * 76)
            print("Geometry direction      : TIR -> RGB")
            print("Final coordinate grid   : RGB")
            print("Mixture of experts      : OFF")
            print("Shared utility scorer   : ON")
            print("Softmax competition     : OFF")
            print(f"DUCF scales             : {list(self.ducf_scales)}")
            print(f"DUCF layers             : {list(self.ducf_layers)}")
            print(
                "Alignment support input : "
                f"{'ON' if self.ducf_use_target_support else 'OFF'}"
            )
            print(
                "Predictor aux weight    : "
                f"{self.ducf_prediction_loss_weight:.4f}"
            )
            print("Initialization          : exact existing ADD fusion")
            print("=" * 76 + "\n")

    # ------------------------------------------------------------------
    # DUCF fusion
    # ------------------------------------------------------------------
    def fuse_backbone_features_ducf(
        self,
        rgb_outputs,
        tir_outputs,
        alignment_info: Optional[Dict[str, Any]] = None,
        return_debug: bool = False,
    ):
        fused_outputs = list(rgb_outputs)
        fusion_features = {}
        ducf_debug = {}

        support_by_scale = {}
        if isinstance(alignment_info, dict):
            support_by_scale = (
                alignment_info.get("support_by_scale", {}) or {}
            )

        for index in self.fusion_indices:
            rgb_feature = rgb_outputs[index]
            tir_feature = tir_outputs[index]
            key = str(index)

            if key in self.ducf_fusions:
                scale = self.ducf_index_to_scale[index]
                support = support_by_scale.get(scale, None)

                if return_debug:
                    fused, dbg = self.ducf_fusions[key](
                        rgb_feature,
                        tir_feature,
                        support=support,
                        return_debug=True,
                    )
                    ducf_debug[scale] = dbg
                else:
                    fused = self.ducf_fusions[key](
                        rgb_feature,
                        tir_feature,
                        support=support,
                        return_debug=False,
                    )
            else:
                # Keep the parent fusion on unselected scales (P5 by default).
                if key not in self.fusions:
                    raise KeyError(
                        f"No fusion module for backbone layer {index}."
                    )
                fused = self.fusions[key](
                    rgb_feature,
                    tir_feature,
                )

            fused_outputs[index] = fused
            fusion_features[index] = fused

        return fused_outputs, fusion_features, ducf_debug

    # ------------------------------------------------------------------
    # Forward features: copied from the current proven wrapper, with ONLY
    # the final fusion call replaced by DUCF-aware fusion.
    # ------------------------------------------------------------------
    def forward_features(
        self,
        rgb,
        tir,
        rgb_semantic=None,
        rgb_valid_mask=None,
        tir_valid_mask=None,
        return_bhlr_debug=False,
        return_alignment_debug=False,
        return_ducf_debug=False,
    ):
        if rgb_semantic is None:
            rgb_semantic = self._fallback_semantic_rgb(rgb)

        raw_rgb = list(self.forward_rgb_backbone(rgb_semantic))
        raw_tir = list(self.forward_tir_backbone(tir))

        if rgb_valid_mask is None:
            rgb_valid_mask = self._infer_valid_mask_from_pixels(rgb_semantic)
        if tir_valid_mask is None:
            tir_valid_mask = self._infer_valid_mask_from_pixels(tir)

        need_align_debug = bool(
            return_alignment_debug
            or return_bhlr_debug
            or return_ducf_debug
            or self.alignment_loss_enabled
        )

        # Existing RLSFA/Local-GTGA: source TIR -> RGB output grid.
        if self.alignment_enabled:
            aligned_tir, alignment_info = self.alignment.align_pyramid(
                rgb_features=raw_rgb,
                tir_features=raw_tir,
                scale_to_index=self.scale_to_index,
                rgb_valid_mask=rgb_valid_mask,
                tir_valid_mask=tir_valid_mask,
                return_debug=need_align_debug,
            )
        else:
            aligned_tir = list(raw_tir)
            alignment_info = {"enabled": False, "type": "identity"}

        # Existing BHLR path remains unchanged.
        rgb_outputs = list(raw_rgb)
        bhlr_debug = {
            "equal_resolution": rgb.shape[-2:] == rgb_semantic.shape[-2:],
            "layers": {},
            "scales": {},
        }

        use_bhlr = bool(
            self.bhlr_enabled
            and self.bhlr is not None
            and rgb.shape[-2:] != rgb_semantic.shape[-2:]
        )

        if use_bhlr:
            detail_raw, reconstructed_high, residual_high = (
                self.bhlr.build_lost_detail_bank(rgb, rgb_semantic)
            )

            if return_bhlr_debug:
                bhlr_debug.update({
                    "reconstructed_high": reconstructed_high,
                    "resolution_residual": residual_high,
                    "residual_s2d": F.pixel_unshuffle(
                        residual_high,
                        downscale_factor=self.bhlr.ratio,
                    ),
                    "detail_raw": detail_raw,
                    "detail_bank": detail_raw,
                })

            for scale in self.bhlr_scales:
                layer_id = self.scale_to_index[scale]

                if self.bhlr_guidance_mode == "alignment":
                    guidance = {
                        "joint_target_support":
                            alignment_info["support_by_scale"][scale]
                    }
                else:
                    guidance = None

                if return_bhlr_debug:
                    enhanced, dbg = self.bhlr.enhance_layer(
                        layer_id=layer_id,
                        detail_raw=detail_raw,
                        rgb_feat=rgb_outputs[layer_id],
                        tir_feat=aligned_tir[layer_id],
                        external_guidance=guidance,
                        return_debug=True,
                    )
                    rgb_outputs[layer_id] = enhanced
                    bhlr_debug["layers"][layer_id] = dbg
                    bhlr_debug["scales"][scale] = dbg
                else:
                    rgb_outputs[layer_id] = self.bhlr.enhance_layer(
                        layer_id=layer_id,
                        detail_raw=detail_raw,
                        rgb_feat=rgb_outputs[layer_id],
                        tir_feat=aligned_tir[layer_id],
                        external_guidance=guidance,
                        return_debug=False,
                    )

        # ONLY changed section: final fusion.
        fused_outputs, fusion_features, ducf_debug = (
            self.fuse_backbone_features_ducf(
                rgb_outputs,
                aligned_tir,
                alignment_info=alignment_info,
                return_debug=return_ducf_debug,
            )
        )

        result = {
            "rgb_backbone": rgb_outputs,
            "rgb_backbone_raw": raw_rgb,
            "tir_backbone": raw_tir,
            "aligned_tir_backbone": aligned_tir,
            "tir_for_bhlr": aligned_tir,
            "tir_for_fusion": aligned_tir,
            "rgb_for_fusion": rgb_outputs,
            "fused_backbone": fused_outputs,
            "fusion_features": fusion_features,
            "rgb_semantic": rgb_semantic,
            "alignment_info": alignment_info,
            "alignment_enabled": self.alignment_enabled,
            "alignment_type": (
                alignment_info.get("type", "rlsfa_gtga_centered")
                if self.alignment_enabled
                else "identity"
            ),
            "scale_to_index": dict(self.scale_to_index),
            "rsdt_scale_to_index": dict(self.scale_to_index),
            "bhlr_enabled": self.bhlr_enabled,
            "bhlr_scales": self.bhlr_scales,
            "bhlr_layers": self.bhlr_layers,
            "bhlr_guidance_mode": self.bhlr_guidance_mode,
            "ducf_enabled": self.ducf_enabled,
            "ducf_scales": self.ducf_scales,
            "ducf_layers": self.ducf_layers,
        }

        if return_bhlr_debug:
            result["bhlr_debug"] = bhlr_debug
            result["rsdt_debug"] = bhlr_debug
        if return_ducf_debug:
            result["ducf_debug"] = ducf_debug

        return result

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
        return_ducf_debug=False,
    ):
        features = self.forward_features(
            rgb,
            tir,
            rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask,
            tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
            return_ducf_debug=return_ducf_debug,
        )
        pred = self.forward_head(features["fused_backbone"])
        if (
            return_features
            or return_bhlr_debug
            or return_alignment_debug
            or return_ducf_debug
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
        return_ducf_debug=False,
    ):
        if isinstance(rgb, dict):
            return self.loss(rgb)
        if tir is None:
            raise ValueError(
                "RLSFA-GTGA-Centered+BHLR+DUCF requires both RGB and TIR."
            )
        return self.predict(
            rgb,
            tir,
            rgb_semantic=rgb_semantic,
            return_features=return_features,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
            return_ducf_debug=return_ducf_debug,
        )

    # ------------------------------------------------------------------
    # DUCF auxiliary predictor loss. Detection loss still trains the actual
    # utility gates. There are no per-modality expert heads.
    # ------------------------------------------------------------------
    def _collect_ducf_prediction_loss(self):
        if not self.ducf_fusions:
            zero = next(self.parameters()).sum() * 0.0
            return zero, {}

        losses = []
        per_scale = {}
        for key, module in self.ducf_fusions.items():
            scale = self.ducf_index_to_scale[int(key)]
            value = module.auxiliary_prediction_loss()
            losses.append(value)
            per_scale[scale] = value

        total = torch.stack(losses).mean() if losses else (
            next(self.parameters()).sum() * 0.0
        )
        return total, per_scale

    def loss(self, batch: Dict, preds=None):
        # Externally supplied predictions cannot reproduce DUCF auxiliary state;
        # retain the parent's standard behavior in that uncommon code path.
        if preds is not None:
            return super().loss(batch, preds=preds)

        base_raw, base_items = super().loss(batch, preds=None)

        if (
            not self.ducf_enabled
            or self.ducf_prediction_loss_weight <= 0.0
        ):
            return base_raw, base_items

        prediction_loss, per_scale = self._collect_ducf_prediction_loss()
        batch_size = int(batch["rgb_semantic_img"].shape[0])
        weighted = (
            self.ducf_prediction_loss_weight
            * prediction_loss
            * float(batch_size)
        )

        if not torch.is_tensor(base_raw):
            raise TypeError(f"Expected tensor base loss, got {type(base_raw)}")

        total_raw = torch.cat(
            (base_raw.reshape(-1), weighted.reshape(1)),
            dim=0,
        )

        if isinstance(base_items, dict):
            items = dict(base_items)
        else:
            items = self._loss_items_to_dict(base_items)

        items["ducf_prediction_loss"] = prediction_loss.detach()
        items["ducf_prediction_weighted_loss"] = (
            self.ducf_prediction_loss_weight * prediction_loss
        ).detach()

        for scale, value in per_scale.items():
            items[f"ducf_p{scale}_prediction_loss"] = value.detach()

            stats = self.ducf_fusions[
                str(self.scale_to_index[scale])
            ].detached_stats()
            for name, stat in stats.items():
                if torch.is_tensor(stat) and stat.numel() == 1:
                    items[f"ducf_p{scale}_{name}"] = stat.detach()

        return total_raw, items


__all__ = [
    "RGBTRLSFAGTGACenteredBHLRDUCFDetectionModel",
]
