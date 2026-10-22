"""
Detector wrapper for:
  RLSFA coarse UAV anchor + Local-GTGA centered scale
  + alignment-guided BHLR-v2 + RGB-T YOLO.

Critical forward connection:
    alignment_info["support_by_scale"][P3]
        -> {"joint_target_support": support}
        -> BHLR-v2 external_guidance
        -> BHLR selector targetness input.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgbt_model import RGBTDetectionModel
from models.modules.alignment.rlsfa_gtga_centered import RLSFAGTGACenteredAlignment
from models.modules.alignment.alignment_reliability_gate import (
    ReliabilityGatedRLSFAGTGACenteredAlignment,
)
from models.modules.bhlr_v2 import MultiScaleBHLRv2


class AdaptiveCrossModalGuidanceFusion(nn.Module):
    """
    Fuse three alignment-support channels into one BHLR guidance map.

    Input channels:
        C0: RGB local target support
        C1: aligned TIR local target support
        C2: historical joint support sqrt(C0 * C1)

    The 1x1 projection predicts a residual in logit space. Its weights and
    bias start at zero, so the module starts from the historical C2 guidance
    and learns only a residual correction.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)
        self.proj = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, guidance_3ch: torch.Tensor):
        if guidance_3ch.ndim != 4 or int(guidance_3ch.shape[1]) != 3:
            raise ValueError(
                "AdaptiveCrossModalGuidanceFusion expects [B,3,H,W], "
                f"got {tuple(guidance_3ch.shape)}"
            )

        x = guidance_3ch.clamp(0.0, 1.0)
        base = x[:, 2:3]
        base_safe = base.clamp(self.eps, 1.0 - self.eps)
        base_logit = torch.log(base_safe) - torch.log1p(-base_safe)
        residual_logit = self.proj(x)
        learned = torch.sigmoid(base_logit + residual_logit)
        return learned, residual_logit


class RGBTRLSFAGTGACenteredBHLRV2DetectionModel(RGBTDetectionModel):
    def __init__(
        self,
        model_name: str = "yolo26s",
        nc: int = 1,
        pretrained: bool = True,
        fusion: str = "add",
        fusion_indices: Optional[Sequence[int]] = None,
        align_mode: str = "bilinear",
        names: Optional[Union[Dict[int, str], Sequence[str]]] = None,
        semantic_imgsz: int = 640,
        alignment_cfg: Optional[Dict[str, Any]] = None,
        rlsfa_gtga_centered_cfg: Optional[Dict[str, Any]] = None,
        alignment_loss_cfg: Optional[Dict[str, Any]] = None,
        rlsfa_gtga_centered_loss_cfg: Optional[Dict[str, Any]] = None,
        bhlr_enabled: bool = True,
        bhlr_scales: Optional[Iterable[int]] = (3,),
        bhlr_layers: Optional[Iterable[int]] = None,
        high_to_semantic_ratio: int = 3,
        detail_channels: int = 48,
        stem_channels: int = 16,
        guide_channels: int = 24,
        bhlr_guidance_mode: str = "alignment",
        bhlr_detach_alignment_guidance: bool = True,
        verbose: bool = True,
        **legacy_kwargs,
    ):
        # Reuse the parent RGB-T fusion implementation directly.
        # For the current final configuration, YAML should use fusion: add.
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

        self.semantic_imgsz = int(semantic_imgsz)
        if len(self.fusion_indices) < 3:
            raise RuntimeError("RLSFA-GTGA-Centered+BHLR requires P3/P4/P5 fusion indices.")

        self.scale_to_index = {
            3: int(self.fusion_indices[0]),
            4: int(self.fusion_indices[1]),
            5: int(self.fusion_indices[2]),
        }
        self.rsdt_scale_to_index = dict(self.scale_to_index)

        # ------------------------------------------------------------------
        # Alignment
        # ------------------------------------------------------------------
        cfg = {}
        if rlsfa_gtga_centered_cfg:
            cfg.update(dict(rlsfa_gtga_centered_cfg))
        if alignment_cfg:
            cfg.update(dict(alignment_cfg))
        cfg.setdefault("enabled", True)
        cfg.setdefault("type", "rlsfa_gtga_centered")
        cfg.setdefault("reference_scale", 3)
        cfg.setdefault("apply_scales", [3, 4, 5])

        self.alignment_cfg = cfg
        self.alignment_enabled = bool(cfg.get("enabled", True))

        # --------------------------------------------------------------
        # Optional reliability gate.
        #
        # enabled: false -> EXACT original centered alignment class
        # enabled: true  -> same estimator + parameter-free geometry gate
        # --------------------------------------------------------------
        reliability_cfg = dict(cfg.get("reliability_gate", {}) or {})
        self.alignment_reliability_enabled = bool(
            reliability_cfg.get("enabled", False)
        )

        if self.alignment_enabled:
            alignment_cls = (
                ReliabilityGatedRLSFAGTGACenteredAlignment
                if self.alignment_reliability_enabled
                else RLSFAGTGACenteredAlignment
            )
            self.alignment = alignment_cls(cfg)
        else:
            self.alignment = None

        self.gtga_reference_scale = int(cfg.get("reference_scale", 3))
        self.gtga_reference_layer = int(self.scale_to_index[self.gtga_reference_scale])

        if self.alignment_enabled:
            self._initialize_alignment_lazy_modules()

        # ------------------------------------------------------------------
        # Auxiliary alignment losses
        # ------------------------------------------------------------------
        lcfg = {}
        if rlsfa_gtga_centered_loss_cfg:
            lcfg.update(dict(rlsfa_gtga_centered_loss_cfg))
        if alignment_loss_cfg:
            lcfg.update(dict(alignment_loss_cfg))

        self.alignment_loss_cfg = lcfg
        self.alignment_loss_enabled = bool(lcfg.get("enabled", True))
        self.lambda_alignment = float(lcfg.get("lambda_alignment", 0.20))
        self.lambda_anchor = float(lcfg.get("lambda_anchor", 1.00))
        self.lambda_coarse = float(lcfg.get("lambda_coarse", 1.00))
        self.lambda_scale = float(lcfg.get("lambda_scale", 1.00))
        self.lambda_targetness = float(lcfg.get("lambda_targetness", 0.20))
        self.smooth_l1_beta = float(lcfg.get("smooth_l1_beta", 1.0))
        self.targetness_alpha = float(lcfg.get("targetness_alpha", 0.75))
        self.targetness_gamma = float(lcfg.get("targetness_gamma", 2.0))
        self.gtga_gaussian_sigma_factor = float(lcfg.get("gtga_gaussian_sigma_factor", 0.25))
        self.gtga_gaussian_sigma_min_px = float(lcfg.get("gtga_gaussian_sigma_min_px", 1.0))
        self.clamp_gt_translation = bool(lcfg.get("clamp_gt_translation", True))

        # New GMLS-v2 bbox-supervised quality and routing losses.
        self.lambda_v2_quality = float(lcfg.get("lambda_v2_quality", 0.25))
        self.lambda_v2_route = float(lcfg.get("lambda_v2_route", 0.10))
        self.v2_quality_iou_weight = float(lcfg.get("v2_quality_iou_weight", 0.80))
        self.v2_quality_advantage = float(lcfg.get("v2_quality_advantage", 0.05))
        self.v2_route_target_temperature = float(lcfg.get("v2_route_target_temperature", 0.05))

        # ------------------------------------------------------------------
        # BHLR
        # ------------------------------------------------------------------
        self.bhlr_enabled = bool(bhlr_enabled)
        self.bhlr_guidance_mode = str(bhlr_guidance_mode).strip().lower()
        if self.bhlr_guidance_mode not in {"alignment", "none"}:
            raise ValueError(
                "bhlr_guidance_mode must be 'alignment' or 'none', "
                f"got {self.bhlr_guidance_mode!r}"
            )
        if self.bhlr_guidance_mode == "alignment" and not self.alignment_enabled:
            raise ValueError("alignment-guided BHLR requires alignment enabled.")

        if bhlr_layers is not None:
            resolved_layers = tuple(sorted({int(x) for x in bhlr_layers}))
            conceptual_scales = tuple(
                sorted(s for s, idx in self.scale_to_index.items() if idx in resolved_layers)
            )
        else:
            conceptual_scales = tuple(sorted({int(x) for x in (bhlr_scales or (3,))}))
            invalid = [s for s in conceptual_scales if s not in self.scale_to_index]
            if invalid:
                raise ValueError(f"BHLR conceptual scales must be P3/P4/P5, got {invalid}")
            resolved_layers = tuple(self.scale_to_index[s] for s in conceptual_scales)

        self.bhlr_scales = conceptual_scales if self.bhlr_enabled else tuple()
        self.bhlr_layers = resolved_layers if self.bhlr_enabled else tuple()
        self.rsdt_scales = self.bhlr_scales

        # Preserve the original one-way alignment -> BHLR policy. Alignment
        # support tensors are detached BEFORE the learnable 1x1 fusion, so the
        # new fusion is trainable while BHLR gradients do not alter alignment.
        self.bhlr_detach_alignment_guidance = bool(
            bhlr_detach_alignment_guidance
        )
        self.alignment_guidance_fusion = nn.ModuleDict()
        if self.bhlr_enabled and self.bhlr_guidance_mode == "alignment":
            self.alignment_guidance_fusion = nn.ModuleDict({
                str(scale): AdaptiveCrossModalGuidanceFusion()
                for scale in self.bhlr_scales
            })

        self.bhlr = None
        if self.bhlr_enabled:
            channels = self._infer_any_layer_channels(self.bhlr_layers)
            rgb_channels = {i: int(channels[i]["rgb"]) for i in self.bhlr_layers}
            tir_channels = {i: int(channels[i]["tir"]) for i in self.bhlr_layers}

            # Current BHLR-v2 already accepts "external" + joint_target_support.
            internal_mode = "external" if self.bhlr_guidance_mode == "alignment" else "none"
            self.bhlr = MultiScaleBHLRv2(
                rgb_channels=rgb_channels,
                tir_channels=tir_channels,
                scales=self.bhlr_layers,
                high_to_semantic_ratio=int(high_to_semantic_ratio),
                detail_channels=int(detail_channels),
                stem_channels=int(stem_channels),
                guide_channels=int(guide_channels),
                exact_identity_when_equal=True,
                guidance_mode=internal_mode,
                # IMPORTANT: do not detach here. The raw alignment supports
                # are detached before AdaptiveCrossModalGuidanceFusion, while
                # the learned 1x1 guidance itself must keep gradients.
                detach_alignment_guidance=False,
            )

            # --------------------------------------------------------------
            # DDP FIX FOR CENTERED ALIGNMENT GUIDANCE
            # --------------------------------------------------------------
            # In the current centered-alignment pipeline, BHLR receives only:
            #
            #     joint_target_support
            #
            # from the alignment module.  It does NOT receive external
            # rgb_structure / tir_structure tensors.
            #
            # Therefore BHLR intentionally uses:
            #   - fallback RGB/TIR structure extractors
            #   - external joint target support
            #
            # and the following trainable compatibility branches do not
            # participate in the loss graph:
            #
            #   1) fallback.target_head
            #      (self-targetness is replaced by joint_target_support)
            #
            #   2) external_rgb_adapter
            #   3) external_tir_adapter
            #      (no external structure tensors are supplied)
            #
            # Their parameter tensors are exactly the kind of permanently
            # unused parameters that make DDP fail when
            # find_unused_parameters=False.
            #
            # Freeze ONLY these compatibility branches.  The actually used
            # fallback structure extractors, detail projection, selector and
            # gamma remain trainable.
            # --------------------------------------------------------------
            if self.bhlr_guidance_mode == "alignment":
                for adapter in self.bhlr.adapters.values():
                    adapter.fallback.target_head.requires_grad_(False)
                    adapter.external_rgb_adapter.requires_grad_(False)
                    adapter.external_tir_adapter.requires_grad_(False)

        # Historical logging compatibility.
        self.rsdt = self.bhlr

        if verbose:
            print("\n" + "=" * 72)
            print("RLSFA coarse anchor + Local-GTGA centered scale + guided BHLR-v2")
            print("=" * 72)
            print(f"Model                 : {self.model_name}")
            print(f"Fusion                : {self.fusion_type}")
            print(f"Reference scale       : P{self.gtga_reference_scale}")
            print("Translation           : targetness-weighted coarse correlation")
            print("Scale pivot           : coarse UAV anchor (NOT image center)")
            print("Scale                 : local sigma_T / sigma_R around UAV anchors")
            print("Frequency refinement  : OFF")
            print("Fine translation      : OFF")
            print("Final warp            : ONE center-preserving unified warp")
            print(
                f"Reliability gate      : "
                f"{'ON' if self.alignment_reliability_enabled else 'OFF'}"
            )
            print(f"BHLR scales           : {list(self.bhlr_scales)}")
            print(f"BHLR guidance         : {self.bhlr_guidance_mode}")
            print("Guidance formula      : 3ch [A_RGB, A_TIR_aligned, J] -> 1x1 residual fusion")
            print("=" * 72 + "\n")

        if legacy_kwargs and verbose:
            print("[RLSFA-GTGA-Centered] ignored legacy keys:", sorted(legacy_kwargs))

    # ==================================================================
    # Utilities
    # ==================================================================
    def _fallback_semantic_rgb(self, rgb_high):
        if rgb_high.shape[-2:] == (self.semantic_imgsz, self.semantic_imgsz):
            return rgb_high
        return F.interpolate(
            rgb_high,
            size=(self.semantic_imgsz, self.semantic_imgsz),
            mode="bilinear",
            align_corners=False,
        )

    @torch.no_grad()
    def _infer_any_layer_channels(self, layer_ids):
        layer_ids = tuple(int(x) for x in layer_ids)
        rgb_training = self.model.training
        tir_training = self.tir_backbone.training
        self.model.eval()
        self.tir_backbone.eval()

        device = next(self.model.parameters()).device
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        rgb = list(self.forward_rgb_backbone(dummy))
        tir = list(self.forward_tir_backbone(dummy))
        out = {
            layer: {"rgb": int(rgb[layer].shape[1]), "tir": int(tir[layer].shape[1])}
            for layer in layer_ids
        }

        self.model.train(rgb_training)
        self.tir_backbone.train(tir_training)
        return out

    @torch.no_grad()
    def _initialize_alignment_lazy_modules(self):
        rgb_training = self.model.training
        tir_training = self.tir_backbone.training
        align_training = self.alignment.training

        self.model.eval()
        self.tir_backbone.eval()
        self.alignment.eval()

        device = next(self.model.parameters()).device
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        rgb = list(self.forward_rgb_backbone(dummy))
        tir = list(self.forward_tir_backbone(dummy))
        self.alignment.align_pyramid(
            rgb, tir, self.scale_to_index, return_debug=False
        )

        self.model.train(rgb_training)
        self.tir_backbone.train(tir_training)
        self.alignment.train(align_training)

        from torch.nn.parameter import UninitializedParameter
        bad = [
            n for n, p in self.named_parameters()
            if isinstance(p, UninitializedParameter)
        ]
        if bad:
            raise RuntimeError("Uninitialized alignment parameters: " + ", ".join(bad))

    @staticmethod
    def _build_valid_content_mask(image, original_shapes, ratio_pads):
        if original_shapes is None or ratio_pads is None:
            return None
        b, _, h, w = image.shape
        if len(original_shapes) != b or len(ratio_pads) != b:
            return None

        mask = torch.zeros(b, 1, h, w, device=image.device, dtype=torch.float32)
        for bi in range(b):
            ori_h, ori_w = original_shapes[bi]
            meta = ratio_pads[bi]
            if not isinstance(meta, Mapping):
                return None
            # YOLO_TCY_STRETCH_NO_PAD_V1: all pixels are valid after stretch.
            if meta.get("stretch", False):
                mask[bi, 0] = 1.0
                continue
            ratio = float(meta["ratio"])
            left, top = map(int, meta["pad"])
            rw = int(round(int(ori_w) * ratio))
            rh = int(round(int(ori_h) * ratio))
            x1, y1 = max(0, min(w, left)), max(0, min(h, top))
            x2, y2 = max(x1, min(w, left + rw)), max(y1, min(h, top + rh))
            mask[bi, 0, y1:y2, x1:x2] = 1.0
        return mask

    @staticmethod
    def _infer_valid_mask_from_pixels(image, pad_value=114.0 / 255.0, tolerance=1e-3):
        b, _, h, w = image.shape
        mask = torch.zeros(b, 1, h, w, device=image.device, dtype=torch.float32)
        value = torch.as_tensor(pad_value, device=image.device, dtype=image.dtype)
        for bi in range(b):
            pad = image[bi].sub(value).abs().le(tolerance).all(dim=0)
            rows = torch.nonzero(~pad.all(dim=1), as_tuple=False).flatten()
            cols = torch.nonzero(~pad.all(dim=0), as_tuple=False).flatten()
            if rows.numel() == 0 or cols.numel() == 0:
                mask[bi, 0] = 1.0
            else:
                y1, y2 = int(rows[0]), int(rows[-1]) + 1
                x1, x2 = int(cols[0]), int(cols[-1]) + 1
                mask[bi, 0, y1:y2, x1:x2] = 1.0
        return mask

    def build_alignment_valid_masks_from_batch(self, batch):
        return (
            self._build_valid_content_mask(
                batch["rgb_semantic_img"],
                batch.get("rgb_ori_shape"),
                batch.get("rgb_semantic_ratio_pad"),
            ),
            self._build_valid_content_mask(
                batch["tir_img"],
                batch.get("tir_ori_shape"),
                batch.get("tir_ratio_pad"),
            ),
        )

    @staticmethod
    def _loss_items_to_dict(loss_items):
        if isinstance(loss_items, dict):
            return dict(loss_items)
        if torch.is_tensor(loss_items):
            values = loss_items.reshape(-1)
        elif isinstance(loss_items, (list, tuple)):
            values = torch.stack([
                v if torch.is_tensor(v) else torch.tensor(float(v))
                for v in loss_items
            ])
        else:
            return {"det_loss": loss_items}

        if int(values.numel()) == 3:
            return {
                "box_loss": values[0].detach(),
                "cls_loss": values[1].detach(),
                "dfl_loss": values[2].detach(),
            }
        return {f"loss_{i}": v.detach() for i, v in enumerate(values)}

    # ==================================================================
    # Forward
    # ==================================================================
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
        if rgb_semantic is None:
            rgb_semantic = self._fallback_semantic_rgb(rgb)

        raw_rgb = list(self.forward_rgb_backbone(rgb_semantic))
        raw_tir = list(self.forward_tir_backbone(tir))

        if rgb_valid_mask is None:
            # YOLO_TCY_STRETCH_NO_PAD_V1: no padded border is present.
            rgb_valid_mask = rgb_semantic.new_ones(
                (rgb_semantic.shape[0], 1, *rgb_semantic.shape[-2:]), dtype=torch.float32
            )
        if tir_valid_mask is None:
            # YOLO_TCY_STRETCH_NO_PAD_V1: no padded border is present.
            tir_valid_mask = tir.new_ones(
                (tir.shape[0], 1, *tir.shape[-2:]), dtype=torch.float32
            )

        need_align_debug = bool(
            return_alignment_debug
            or return_bhlr_debug
            or self.alignment_loss_enabled
            or (
                self.bhlr_enabled
                and self.bhlr_guidance_mode == "alignment"
            )
        )

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
            detail_raw, reconstructed_high, residual_high = self.bhlr.build_lost_detail_bank(
                rgb, rgb_semantic
            )
            if return_bhlr_debug:
                bhlr_debug.update({
                    "reconstructed_high": reconstructed_high,
                    "resolution_residual": residual_high,
                    "residual_s2d": F.pixel_unshuffle(
                        residual_high, downscale_factor=self.bhlr.ratio
                    ),
                    "detail_raw": detail_raw,
                    "detail_bank": detail_raw,
                })

            for scale in self.bhlr_scales:
                layer_id = self.scale_to_index[scale]

                # ----------------------------------------------------------
                # 3-channel alignment -> learnable 1x1 BHLR guidance.
                # C0 = RGB local target support
                # C1 = aligned TIR local target support
                # C2 = historical joint support J=sqrt(C0*C1)
                # ----------------------------------------------------------
                if self.bhlr_guidance_mode == "alignment":
                    dbg_map = alignment_info.get("scale_debug", {})
                    ref_scale = self.gtga_reference_scale
                    ref_dbg = dbg_map.get(ref_scale, dbg_map.get(str(ref_scale), None))
                    if ref_dbg is None:
                        raise RuntimeError(
                            "3-channel BHLR guidance requires alignment scale_debug "
                            f"for reference P{ref_scale}."
                        )

                    rgb_support = ref_dbg["rgb_local_targetness"]
                    tir_support = ref_dbg["aligned_tir_local_targetness"]
                    joint_support = alignment_info["support_by_scale"][scale]

                    target_hw = rgb_outputs[layer_id].shape[-2:]
                    if rgb_support.shape[-2:] != target_hw:
                        rgb_support = F.interpolate(
                            rgb_support, size=target_hw, mode="bilinear",
                            align_corners=False,
                        )
                    if tir_support.shape[-2:] != target_hw:
                        tir_support = F.interpolate(
                            tir_support, size=target_hw, mode="bilinear",
                            align_corners=False,
                        )
                    if joint_support.shape[-2:] != target_hw:
                        joint_support = F.interpolate(
                            joint_support, size=target_hw, mode="bilinear",
                            align_corners=False,
                        )

                    rgb_support = rgb_support.clamp(0.0, 1.0)
                    tir_support = tir_support.clamp(0.0, 1.0)
                    joint_support = joint_support.clamp(0.0, 1.0)

                    if self.bhlr_detach_alignment_guidance:
                        rgb_support = rgb_support.detach()
                        tir_support = tir_support.detach()
                        joint_support = joint_support.detach()

                    guidance_3ch = torch.cat(
                        [rgb_support, tir_support, joint_support], dim=1
                    )
                    learned_support, residual_logit = (
                        self.alignment_guidance_fusion[str(scale)](guidance_3ch)
                    )

                    # Keep the historical BHLR interface unchanged: the
                    # learned single-channel map is supplied under the same key.
                    guidance = {
                        "joint_target_support": learned_support,
                        "original_joint_target_support": joint_support,
                        "rgb_target_support": rgb_support,
                        "aligned_tir_target_support": tir_support,
                        "multi_target_guidance": guidance_3ch,
                        "guidance_residual_logit": residual_logit,
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
                    if guidance is not None:
                        dbg["rgb_target_support"] = guidance["rgb_target_support"]
                        dbg["aligned_tir_target_support"] = guidance["aligned_tir_target_support"]
                        dbg["original_joint_target_support"] = guidance["original_joint_target_support"]
                        dbg["multi_target_guidance"] = guidance["multi_target_guidance"]
                        dbg["guidance_residual_logit"] = guidance["guidance_residual_logit"]
                        dbg["learned_target_support"] = guidance["joint_target_support"]
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

        fused_outputs, fusion_features = self.fuse_backbone_features(
            rgb_outputs, aligned_tir
        )

        result = {
            "rgb_backbone": rgb_outputs,
            "rgb_backbone_raw": raw_rgb,
            "tir_backbone": raw_tir,
            "aligned_tir_backbone": aligned_tir,
            "tir_for_bhlr": aligned_tir,
            "tir_for_fusion": aligned_tir,
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
        }
        if return_bhlr_debug:
            result["bhlr_debug"] = bhlr_debug
            result["rsdt_debug"] = bhlr_debug
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
    ):
        features = self.forward_features(
            rgb, tir,
            rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask,
            tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )
        pred = self.forward_head(features["fused_backbone"])
        if return_features or return_bhlr_debug or return_alignment_debug:
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
    ):
        if isinstance(rgb, dict):
            return self.loss(rgb)
        if tir is None:
            raise ValueError("RLSFA-GTGA-Centered+BHLR requires both RGB and TIR.")
        return self.predict(
            rgb, tir,
            rgb_semantic=rgb_semantic,
            return_features=return_features,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )

    # ==================================================================
    # Loss
    # ==================================================================
    @staticmethod
    def _build_target_mask(boxes, batch_idx, batch_size, h, w, device, dtype):
        mask = torch.zeros(batch_size, 1, h, w, device=device, dtype=dtype)
        for i in range(int(boxes.shape[0])):
            bi = int(batch_idx[i].item())
            cx, cy, bw, bh = boxes[i]
            x1 = max(0, min(w - 1, int(torch.floor((cx - bw / 2) * w).item())))
            y1 = max(0, min(h - 1, int(torch.floor((cy - bh / 2) * h).item())))
            x2 = max(x1 + 1, min(w, int(torch.ceil((cx + bw / 2) * w).item())))
            y2 = max(y1 + 1, min(h, int(torch.ceil((cy + bh / 2) * h).item())))
            mask[bi, 0, y1:y2, x1:x2] = 1.0
        return mask

    @staticmethod
    def _build_gaussian_target(
        boxes, batch_idx, batch_size, h, w, device, dtype,
        sigma_factor: float = 0.25, sigma_min_px: float = 1.0,
    ):
        """Soft Gaussian target for GTGA RGB/TIR target heads."""
        target = torch.zeros(batch_size, 1, h, w, device=device, dtype=dtype)
        ys = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
        xs = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
        for i in range(int(boxes.shape[0])):
            bi = int(batch_idx[i].item())
            cx, cy, bw, bh = boxes[i].to(dtype=dtype)
            cx_px = cx * max(w - 1, 1)
            cy_px = cy * max(h - 1, 1)
            sx = torch.clamp(bw * w * float(sigma_factor), min=float(sigma_min_px))
            sy = torch.clamp(bh * h * float(sigma_factor), min=float(sigma_min_px))
            g = torch.exp(-0.5 * (((xs - cx_px) / sx).square() + ((ys - cy_px) / sy).square()))
            target[bi, 0] = torch.maximum(target[bi, 0], g[0])
        return target.clamp(0, 1)

    def _focal_bce(self, logits, target):
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = target * prob + (1 - target) * (1 - prob)
        alpha_t = self.targetness_alpha * target + (1 - self.targetness_alpha) * (1 - target)
        return (alpha_t * (1 - pt).pow(self.targetness_gamma) * ce).mean()

    @staticmethod
    def _gmls_v2_candidate_quality(rbox, tbox, offsets, feat_h, feat_w,
                                   iou_weight=0.80, eps=1e-7):
        """GT-derived translation-only bbox quality for two coarse candidates.

        Model geometry: q = p + d; thus aligned TIR bbox center = c_T - d_norm.
        A small Gaussian center component avoids both qualities collapsing to
        zero when the tiny UAV boxes do not overlap at all.
        All targets are detached by the caller; labels are never inference inputs.
        """
        offsets = offsets.float().reshape(-1, 2)
        ref = rbox.float().reshape(4)
        src = tbox.float().reshape(4)
        norm = offsets.new_tensor([max(int(feat_w)-1,1), max(int(feat_h)-1,1)])
        aligned_center = src[None, :2] - offsets / norm[None, :]
        aligned_size = src[None, 2:4].expand_as(aligned_center).clamp_min(eps)
        tlo = aligned_center - aligned_size * 0.5
        thi = aligned_center + aligned_size * 0.5
        rlo = ref[:2] - ref[2:] * 0.5
        rhi = ref[:2] + ref[2:] * 0.5
        inter = (torch.minimum(thi, rhi) - torch.maximum(tlo, rlo)).clamp_min(0).prod(-1)
        union = aligned_size.prod(-1) + ref[2:].prod() - inter
        iou = (inter / union.clamp_min(eps)).clamp(0,1)
        # Smooth distance term in units of bbox width and height.
        size_ref = torch.sqrt((aligned_size * ref[None, 2:].clamp_min(eps)).clamp_min(eps))
        center_delta = (aligned_center - ref[None, :2]) / size_ref
        center_quality = torch.exp(-0.5 * center_delta.square().sum(-1)).clamp(0,1)
        alpha = min(max(float(iou_weight),0.0),1.0)
        return (alpha * iou + (1.0-alpha) * center_quality).clamp(0,1)

    def _alignment_auxiliary_loss(self, batch, alignment_info):
        dbg_map = alignment_info["scale_debug"]
        s = self.gtga_reference_scale
        dbg = dbg_map[s] if s in dbg_map else dbg_map[str(s)]

        rgb_logits = dbg["rgb_target_logits"].float()
        tir_logits = dbg["tir_target_logits"].float()
        coarse_logits = dbg["coarse_targetness_logits"].float()
        bsz, _, h, w = rgb_logits.shape

        coarse_mask = self._build_target_mask(
            batch["rgb_semantic_bboxes"], batch["rgb_batch_idx"],
            bsz, h, w, coarse_logits.device, coarse_logits.dtype,
        )
        rgb_gaussian = self._build_gaussian_target(
            batch["rgb_semantic_bboxes"], batch["rgb_batch_idx"],
            bsz, h, w, rgb_logits.device, rgb_logits.dtype,
            sigma_factor=self.gtga_gaussian_sigma_factor,
            sigma_min_px=self.gtga_gaussian_sigma_min_px,
        )
        tir_gaussian = self._build_gaussian_target(
            batch["tir_bboxes"], batch["tir_batch_idx"],
            bsz, h, w, tir_logits.device, tir_logits.dtype,
            sigma_factor=self.gtga_gaussian_sigma_factor,
            sigma_min_px=self.gtga_gaussian_sigma_min_px,
        )
        target_loss = (
            self._focal_bce(rgb_logits, rgb_gaussian)
            + self._focal_bce(tir_logits, tir_gaussian)
            + self._focal_bce(coarse_logits, coarse_mask)
        ) / 3.0

        pred_anchor = dbg["coarse_anchor"].float()
        pred_offset = dbg["coarse_offset"].float()
        pred_scale = dbg["sampling_scale"].float()
        radius = float(self.alignment.unit.coarse_radius)

        # Optional v2 quality head; baseline v1 still works with this wrapper.
        v2_logits = dbg.get("gmls_v2_quality_logits", None)
        v2_route_logit = dbg.get("gmls_v2_route_logit", None)
        v2_gmls = dbg.get("coarse_offset_gmls", None)
        v2_dmls = dbg.get("coarse_offset_dmls", None)

        anchor_losses, coarse_losses, scale_losses = [], [], []
        v2_quality_losses, v2_route_losses = [], []
        v2_gmls_gt_qualities, v2_dmls_gt_qualities = [], []
        for bi in range(bsz):
            rsel = batch["rgb_batch_idx"] == bi
            tsel = batch["tir_batch_idx"] == bi

            # Geometry supervision only where pairing is unambiguous.
            if int(rsel.sum()) != 1 or int(tsel.sum()) != 1:
                continue

            rbox = batch["rgb_semantic_bboxes"][rsel][0].float()
            tbox = batch["tir_bboxes"][tsel][0].float()

            gt_anchor = torch.stack((rbox[0], rbox[1])).view(1, 2)
            scale_xy = pred_anchor.new_tensor([max(w - 1, 1), max(h - 1, 1)]).view(1, 2)
            anchor_losses.append(
                F.smooth_l1_loss(
                    pred_anchor[bi:bi + 1] * scale_xy, gt_anchor * scale_xy,
                    beta=self.smooth_l1_beta, reduction="mean"
                )
            )

            gt_offset = torch.stack((
                (tbox[0] - rbox[0]) * max(w - 1, 1),
                (tbox[1] - rbox[1]) * max(h - 1, 1),
            )).view(1, 2, 1, 1)
            if self.clamp_gt_translation:
                gt_offset = gt_offset.clamp(-radius, radius)

            coarse_losses.append(
                F.smooth_l1_loss(
                    pred_offset[bi:bi + 1], gt_offset,
                    beta=self.smooth_l1_beta, reduction="mean"
                )
            )

            # Inverse-sampling scale q=S*p+t:
            # source TIR width / desired RGB output width.
            gt_scale = torch.stack((
                tbox[2] / rbox[2].clamp_min(1e-6),
                tbox[3] / rbox[3].clamp_min(1e-6),
            )).view(1, 2)
            gt_scale = gt_scale.clamp(
                self.alignment.unit.scale_min,
                self.alignment.unit.scale_max,
            )
            scale_losses.append(
                F.smooth_l1_loss(
                    torch.log(pred_scale[bi:bi + 1].clamp_min(1e-6)),
                    torch.log(gt_scale.clamp_min(1e-6)),
                    beta=self.smooth_l1_beta,
                    reduction="mean",
                )
            )

            if (
                v2_logits is not None and v2_route_logit is not None
                and v2_gmls is not None and v2_dmls is not None
            ):
                offsets = torch.stack((
                    v2_gmls[bi].float().reshape(2),
                    v2_dmls[bi].float().reshape(2),
                ), dim=0)
                quality = self._gmls_v2_candidate_quality(
                    rbox=rbox, tbox=tbox, offsets=offsets.detach(),
                    feat_h=h, feat_w=w,
                    iou_weight=self.v2_quality_iou_weight,
                ).detach()
                logits_i = v2_logits[bi].float().reshape(2)
                # AMP-safe: train pre-sigmoid logits, not BCE(sigmoid(x), y).
                v2_quality_losses.append(
                    F.binary_cross_entropy_with_logits(
                        logits_i, quality, reduction="mean"
                    )
                )
                advantage = quality[1] - quality[0]
                target_route = torch.sigmoid(
                    (advantage - self.v2_quality_advantage)
                    / max(self.v2_route_target_temperature, 1e-4)
                ).detach()
                v2_route_losses.append(
                    F.binary_cross_entropy_with_logits(
                        v2_route_logit[bi].float().reshape(()),
                        target_route.to(dtype=torch.float32),
                        reduction="mean",
                    )
                )
                v2_gmls_gt_qualities.append(quality[0])
                v2_dmls_gt_qualities.append(quality[1])

        # Head remains in the loss graph even for an unpaired/no-box batch;
        # essential for robust DDP with batches containing no valid 1:1 pair.
        zero = pred_offset.sum() * 0.0
        if v2_logits is not None:
            zero = zero + v2_logits.sum() * 0.0
        if v2_route_logit is not None:
            zero = zero + v2_route_logit.sum() * 0.0
        anchor_loss = torch.stack(anchor_losses).mean() if anchor_losses else zero
        coarse_loss = torch.stack(coarse_losses).mean() if coarse_losses else zero
        scale_loss = torch.stack(scale_losses).mean() if scale_losses else zero
        v2_quality_loss = torch.stack(v2_quality_losses).mean() if v2_quality_losses else zero
        v2_route_loss = torch.stack(v2_route_losses).mean() if v2_route_losses else zero

        total = (
            self.lambda_anchor * anchor_loss
            + self.lambda_coarse * coarse_loss
            + self.lambda_scale * scale_loss
            + self.lambda_targetness * target_loss
            + self.lambda_v2_quality * v2_quality_loss
            + self.lambda_v2_route * v2_route_loss
        )
        return {
            "total": total,
            "anchor": anchor_loss,
            "coarse": coarse_loss,
            "scale": scale_loss,
            "targetness": target_loss,
            "v2_quality": v2_quality_loss,
            "v2_route": v2_route_loss,
            "v2_gmls_gt_quality": (torch.stack(v2_gmls_gt_qualities).mean() if v2_gmls_gt_qualities else zero).detach(),
            "v2_dmls_gt_quality": (torch.stack(v2_dmls_gt_qualities).mean() if v2_dmls_gt_qualities else zero).detach(),
            "valid_pairs": len(coarse_losses),
        }

    def loss(self, batch: Dict, preds=None):
        required = (
            "rgb_img", "rgb_semantic_img", "tir_img",
            "rgb_cls", "rgb_semantic_bboxes", "rgb_batch_idx",
            "tir_bboxes", "tir_batch_idx",
        )
        for key in required:
            if key not in batch:
                raise KeyError(f"RLSFA-GTGA-Centered+BHLR batch missing {key!r}")

        if self.criterion is None:
            self.criterion = self.init_criterion()

        yolo_batch = {
            "img": batch["rgb_semantic_img"],
            "cls": batch["rgb_cls"],
            "bboxes": batch["rgb_semantic_bboxes"],
            "batch_idx": batch["rgb_batch_idx"],
        }
        if preds is not None:
            return self.criterion(preds, yolo_batch)

        rgb_mask, tir_mask = self.build_alignment_valid_masks_from_batch(batch)
        outputs = self.predict(
            batch["rgb_img"],
            batch["tir_img"],
            rgb_semantic=batch["rgb_semantic_img"],
            rgb_valid_mask=rgb_mask,
            tir_valid_mask=tir_mask,
            return_features=True,
            return_alignment_debug=self.alignment_loss_enabled,
        )
        det_raw, det_items = self.criterion(outputs["pred"], yolo_batch)

        if not self.alignment_loss_enabled:
            return det_raw, det_items

        stats = self._alignment_auxiliary_loss(batch, outputs["alignment_info"])
        batch_size = int(batch["rgb_semantic_img"].shape[0])
        align_raw = self.lambda_alignment * stats["total"] * float(batch_size)

        if not torch.is_tensor(det_raw):
            raise TypeError(f"Expected tensor detection loss, got {type(det_raw)}")
        total_raw = torch.cat((det_raw.reshape(-1), align_raw.reshape(1)), dim=0)

        items = self._loss_items_to_dict(det_items)
        items.update({
            "align_loss": stats["total"].detach(),
            "align_anchor_loss": stats["anchor"].detach(),
            "align_coarse_loss": stats["coarse"].detach(),
            "align_scale_loss": stats["scale"].detach(),
            "align_targetness_loss": stats["targetness"].detach(),
            "align_v2_quality_loss": stats["v2_quality"].detach(),
            "align_v2_route_loss": stats["v2_route"].detach(),
            "align_v2_gmls_gt_quality": stats["v2_gmls_gt_quality"].detach(),
            "align_v2_dmls_gt_quality": stats["v2_dmls_gt_quality"].detach(),
            "align_weighted_loss": (self.lambda_alignment * stats["total"]).detach(),
            "align_valid_pairs": float(stats["valid_pairs"]),
        })
        return total_raw, items


__all__ = [
    "AdaptiveCrossModalGuidanceFusion",
    "RGBTRLSFAGTGACenteredBHLRV2DetectionModel",
]
