"""
DUCF: Detection-Utility Complementary Fusion
===========================================

Purpose
-------
Fuse already aligned RGB/TIR backbone features without hard-coding either
modality as the primary modality.

This module is designed for the YOLO-tcy-v3 pipeline where:
    1) RLSFA/GTGA aligns TIR -> RGB feature coordinates;
    2) BHLR enhances RGB high-resolution detail;
    3) DUCF replaces the final ADD/CONCAT fusion.

Key properties
--------------
1. No mixture-of-experts / no expert routing.
   There is one fused feature and one downstream YOLO neck/head.

2. Symmetric modality treatment.
   RGB and TIR use the same embedding, predictor, innovation encoder and
   utility scorer. The scorer is called twice with swapped inputs.

3. Independent utility gates, NOT softmax competition.
   Both modalities can be simultaneously useful or simultaneously weak.

4. Exact ADD baseline at initialization.
   Utility multipliers start at 1 and all residual fusion gains start at 0:
       F_out(init) = F_RGB + F_TIR
   This matches the existing ADD fusion used by the current experiment.

5. Bidirectional cross-modal innovation.
   Let R/T be shared projected features and P(.) be one shared predictor:
       I_R = R - P(T)
       I_T = T - P(R)
   The detector can inject useful modality-specific residual information
   without declaring RGB or TIR globally stronger.

6. Optional target-support conditioning.
   RLSFA/GTGA joint_target_support can be supplied as an additional gate
   input. It is used only as context; it is NOT treated as a modality-quality
   label and does not directly force either modality weight.

The optional predictor auxiliary loss trains only the shared predictor against
stop-gradient targets. It does not backpropagate this auxiliary objective into
RGB/TIR backbones. Detection loss remains responsible for learning the final
utility gates and fusion behavior.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Return the largest <= max_groups group count that divides channels."""
    c = int(channels)
    for g in range(min(int(max_groups), c), 0, -1):
        if c % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    """Conv2d -> GroupNorm -> SiLU, avoiding modality-order-dependent BN."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)
        k = int(k)
        s = int(s)
        groups = int(groups)
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=k // 2,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(c2), c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SharedCrossModalPredictor(nn.Module):
    """
    One predictor is shared in both directions:
        P(T) -> predicted RGB embedding
        P(R) -> predicted TIR embedding

    Sharing is intentional: there is no modality-specific predictor prior.
    """

    def __init__(self, channels: int):
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            ConvGNAct(c, c, 1, 1),
            ConvGNAct(c, c, 3, 1, groups=c),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedUtilityScorer(nn.Module):
    """
    Shared detector-driven utility estimator.

    The same scorer is applied twice:
        score(R, T, I_R, C, support)
        score(T, R, I_T, C, support)

    Output is an independent spatial utility multiplier in (0, 2):
        multiplier = 2 * sigmoid(logit)

    The final conv is zero-initialized, hence multiplier starts exactly at 1.
    No softmax is used, so one modality does not have to lose weight when the
    other modality gains weight.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        h = int(hidden_channels)
        # Inputs: modality, |innovation|, shared consensus, |modality-other|,
        #         target support (1 channel).
        in_channels = 4 * h + 1
        self.body = nn.Sequential(
            ConvGNAct(in_channels, h, 1, 1),
            ConvGNAct(h, h, 3, 1, groups=h),
            nn.Conv2d(h, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self,
        modality: torch.Tensor,
        other: torch.Tensor,
        innovation: torch.Tensor,
        consensus: torch.Tensor,
        support: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat(
            (
                modality,
                innovation.abs(),
                consensus,
                (modality - other).abs(),
                support,
            ),
            dim=1,
        )
        logit = self.body(x)
        multiplier = 2.0 * torch.sigmoid(logit)
        return multiplier, logit


class DUCFFusion(nn.Module):
    """
    Detection-Utility Complementary Fusion for one pyramid scale.

    Parameters
    ----------
    rgb_channels / tir_channels:
        For strict symmetric DUCF these must be equal. In the current
        dual-backbone YOLO setup both backbones use the same architecture,
        therefore P3/P4/P5 channel widths are expected to match.

    hidden_channels:
        Compact common-space width used by the utility/innovation branch.

    use_target_support:
        If True, the RLSFA/GTGA support map is supplied to utility scorers.
        If unavailable, an all-ones map is used.

    predictor_aux_enabled:
        Enables the stop-gradient bidirectional cross-prediction auxiliary
        objective. The model wrapper controls its final loss weight.
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        hidden_channels: int = 64,
        use_target_support: bool = True,
        predictor_aux_enabled: bool = True,
        consensus_gain_init: float = 0.0,
        innovation_gain_init: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.rgb_channels = int(rgb_channels)
        self.tir_channels = int(tir_channels)
        self.hidden_channels = int(hidden_channels)
        self.use_target_support = bool(use_target_support)
        self.predictor_aux_enabled = bool(predictor_aux_enabled)
        self.eps = float(eps)

        if self.rgb_channels != self.tir_channels:
            raise ValueError(
                "DUCF strict symmetric mode requires equal RGB/TIR channels, "
                f"got RGB={self.rgb_channels}, TIR={self.tir_channels}."
            )
        if self.hidden_channels < 8:
            raise ValueError("DUCF hidden_channels should be >= 8.")

        c = self.rgb_channels
        h = self.hidden_channels

        # One embedding is shared by both modalities.
        self.shared_embed = nn.Sequential(
            ConvGNAct(c, h, 1, 1),
            ConvGNAct(h, h, 3, 1, groups=h),
        )

        # One predictor is shared in both cross-modal directions.
        self.shared_predictor = SharedCrossModalPredictor(h)

        # Symmetric consensus statistics:
        #   mean(R,T), |R-T|, R*T
        self.consensus_encoder = nn.Sequential(
            ConvGNAct(3 * h, h, 1, 1),
            ConvGNAct(h, h, 3, 1, groups=h),
        )

        # Same scorer for RGB and TIR, called with swapped inputs.
        self.shared_utility = SharedUtilityScorer(h)

        # Same innovation transform for RGB and TIR.
        self.shared_innovation_encoder = nn.Sequential(
            ConvGNAct(h, h, 1, 1),
            ConvGNAct(h, h, 3, 1, groups=h),
        )

        # Project compact residuals back to detector feature channels.
        self.consensus_to_out = nn.Conv2d(h, c, kernel_size=1, bias=False)
        self.innovation_to_out = nn.Conv2d(h, c, kernel_size=1, bias=False)

        # Residual gains are scalar and shared across modalities.
        self.consensus_gain_raw = nn.Parameter(
            torch.tensor(self._inverse_tanh_safe(consensus_gain_init), dtype=torch.float32)
        )
        self.innovation_gain_raw = nn.Parameter(
            torch.tensor(self._inverse_tanh_safe(innovation_gain_init), dtype=torch.float32)
        )

        self._last_aux_prediction_loss: Optional[torch.Tensor] = None
        self._last_debug: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _inverse_tanh_safe(value: float) -> float:
        v = max(-0.999, min(0.999, float(value)))
        return 0.5 * math.log((1.0 + v) / (1.0 - v))

    @staticmethod
    def _resize_like(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if source.shape[-2:] == reference.shape[-2:]:
            return source
        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def _prepare_support(
        self,
        support: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = reference.shape
        if (not self.use_target_support) or support is None:
            return reference.new_ones(b, 1, h, w)
        if support.ndim == 3:
            support = support.unsqueeze(1)
        if support.ndim != 4 or support.shape[0] != b:
            raise ValueError(
                "DUCF support must be [B,1,H,W] or [B,H,W], "
                f"got {tuple(support.shape)}"
            )
        if support.shape[1] != 1:
            support = support.mean(dim=1, keepdim=True)
        support = support.to(device=reference.device, dtype=reference.dtype)
        if support.shape[-2:] != (h, w):
            support = F.interpolate(
                support,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        return support.clamp(0.0, 1.0)

    def _predictor_auxiliary_loss(
        self,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if (not self.predictor_aux_enabled) or (not self.training):
            return r.sum() * 0.0

        # Train the predictor itself, but do not let this auxiliary objective
        # pull RGB/TIR backbones toward each other.
        pred_r = self.shared_predictor(t.detach())
        pred_t = self.shared_predictor(r.detach())

        target_r = r.detach()
        target_t = t.detach()

        # Normalize channel energy so the auxiliary term measures structure
        # agreement instead of raw modality amplitude.
        pred_r_n = F.normalize(pred_r, p=2, dim=1, eps=self.eps)
        pred_t_n = F.normalize(pred_t, p=2, dim=1, eps=self.eps)
        target_r_n = F.normalize(target_r, p=2, dim=1, eps=self.eps)
        target_t_n = F.normalize(target_t, p=2, dim=1, eps=self.eps)

        loss_r = F.smooth_l1_loss(pred_r_n, target_r_n, reduction="mean")
        loss_t = F.smooth_l1_loss(pred_t_n, target_t_n, reduction="mean")
        return 0.5 * (loss_r + loss_t)

    def forward(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        support: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        if rgb.ndim != 4 or tir.ndim != 4:
            raise ValueError(
                "DUCF expects BCHW tensors, "
                f"got RGB={tuple(rgb.shape)}, TIR={tuple(tir.shape)}"
            )
        if int(rgb.shape[1]) != self.rgb_channels:
            raise ValueError(
                f"DUCF RGB channels mismatch: {rgb.shape[1]} != {self.rgb_channels}"
            )
        if int(tir.shape[1]) != self.tir_channels:
            raise ValueError(
                f"DUCF TIR channels mismatch: {tir.shape[1]} != {self.tir_channels}"
            )

        # Safety only. RLSFA/GTGA should already produce TIR on the RGB grid.
        tir = self._resize_like(tir, rgb)

        # Shared common-space embeddings.
        r = self.shared_embed(rgb)
        t = self.shared_embed(tir)

        # Bidirectional predictable components and innovations.
        pred_r_from_t = self.shared_predictor(t)
        pred_t_from_r = self.shared_predictor(r)
        innovation_r = r - pred_r_from_t
        innovation_t = t - pred_t_from_r

        # Exchange-invariant consensus context.
        consensus = self.consensus_encoder(
            torch.cat(
                (
                    0.5 * (r + t),
                    (r - t).abs(),
                    r * t,
                ),
                dim=1,
            )
        )

        support_map = self._prepare_support(support, r)

        # Shared scorer, swapped arguments -> no architecture-level modality
        # preference. Multipliers are independent and do not sum to one.
        rgb_multiplier, rgb_logit = self.shared_utility(
            r,
            t,
            innovation_r,
            consensus,
            support_map,
        )
        tir_multiplier, tir_logit = self.shared_utility(
            t,
            r,
            innovation_t,
            consensus,
            support_map,
        )

        # Exact existing ADD baseline at initialization because both utility
        # multipliers start at 1.
        adaptive_add = rgb_multiplier * rgb + tir_multiplier * tir

        # Symmetric innovation correction.
        encoded_i_r = self.shared_innovation_encoder(innovation_r)
        encoded_i_t = self.shared_innovation_encoder(innovation_t)
        innovation_mix = (
            rgb_multiplier * encoded_i_r
            + tir_multiplier * encoded_i_t
        )

        consensus_delta = self.consensus_to_out(consensus)
        innovation_delta = self.innovation_to_out(innovation_mix)

        consensus_gain = torch.tanh(self.consensus_gain_raw)
        innovation_gain = torch.tanh(self.innovation_gain_raw)

        fused = (
            adaptive_add
            + consensus_gain * consensus_delta
            + innovation_gain * innovation_delta
        )

        aux_prediction_loss = self._predictor_auxiliary_loss(r, t)
        self._last_aux_prediction_loss = aux_prediction_loss

        # Detached monitoring statistics are cheap and useful for diagnosing
        # whether one dataset pushes DUCF toward RGB or TIR.
        with torch.no_grad():
            self._last_debug = {
                "rgb_multiplier_mean": rgb_multiplier.mean().detach(),
                "tir_multiplier_mean": tir_multiplier.mean().detach(),
                "rgb_multiplier_std": rgb_multiplier.std(unbiased=False).detach(),
                "tir_multiplier_std": tir_multiplier.std(unbiased=False).detach(),
                "rgb_innovation_abs_mean": innovation_r.abs().mean().detach(),
                "tir_innovation_abs_mean": innovation_t.abs().mean().detach(),
                "support_mean": support_map.mean().detach(),
                "consensus_gain": consensus_gain.detach(),
                "innovation_gain": innovation_gain.detach(),
                "prediction_aux_loss": aux_prediction_loss.detach(),
            }

        if not return_debug:
            return fused

        debug = {
            "rgb_multiplier": rgb_multiplier,
            "tir_multiplier": tir_multiplier,
            "rgb_utility_logit": rgb_logit,
            "tir_utility_logit": tir_logit,
            "rgb_innovation_energy": innovation_r.square().mean(dim=1, keepdim=True),
            "tir_innovation_energy": innovation_t.square().mean(dim=1, keepdim=True),
            "support": support_map,
            "consensus": consensus,
            "consensus_gain": consensus_gain,
            "innovation_gain": innovation_gain,
            "prediction_aux_loss": aux_prediction_loss,
        }
        return fused, debug

    def auxiliary_prediction_loss(self) -> torch.Tensor:
        """Return the most recent differentiable predictor auxiliary loss."""
        if self._last_aux_prediction_loss is not None:
            return self._last_aux_prediction_loss
        # Parameter-derived zero keeps device/dtype correct before first forward.
        return self.consensus_gain_raw.sum() * 0.0

    def detached_stats(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_debug)


__all__ = [
    "DUCFFusion",
    "SharedUtilityScorer",
    "SharedCrossModalPredictor",
]
