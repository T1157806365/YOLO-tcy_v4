"""
DUCF-Safe v2: Detection-Utility Complementary Fusion with protected ADD anchor.

Design goal
-----------
Keep the proven ADD fusion path *always present*:

    F_base = F_RGB + F_TIR

DUCF never multiplies the raw RGB/TIR features directly.  It can only add a
bounded residual correction learned from symmetric cross-modal context:

    F_out = F_base
          + g_c * N(Delta_consensus; F_base)
          + g_i * N(Delta_innovation; F_base)

where g_c/g_i are bounded scalar gains initialized at exactly zero.
Therefore:

    F_out(init) == F_RGB + F_TIR

and, unlike v1, the first detector gradient cannot immediately rescale the
raw P3/P4 features pixel-by-pixel.

The utility scorer is shared between modalities and uses independent sigmoid
scores.  It only gates the innovation residual branch; it never chooses one
expert and never forms a softmax competition.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    c = int(channels)
    for g in range(min(int(max_groups), c), 0, -1):
        if c % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, groups: int = 1):
        super().__init__()
        c1, c2, k, groups = int(c1), int(c2), int(k), int(groups)
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, 1, k // 2, groups=groups, bias=False),
            nn.GroupNorm(_group_count(c2), c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedPredictor(nn.Module):
    """One predictor shared for T->R and R->T prediction."""

    def __init__(self, channels: int):
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            ConvGNAct(c, c, 1),
            ConvGNAct(c, c, 3, groups=c),
            nn.Conv2d(c, c, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedUtilityScorer(nn.Module):
    """
    Symmetric, shared utility scorer.

    The same weights are called twice with modality order swapped.
    Scores are independent sigmoid values in (0, 1), not a softmax pair.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        h = int(hidden_channels)
        # modality, other, |innovation|, consensus, support
        self.net = nn.Sequential(
            ConvGNAct(4 * h + 1, h, 1),
            ConvGNAct(h, h, 3, groups=h),
            nn.Conv2d(h, 1, 1, bias=True),
        )
        # Neutral score = 0.5 at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        modality: torch.Tensor,
        other: torch.Tensor,
        innovation: torch.Tensor,
        consensus: torch.Tensor,
        support: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat(
            [modality, other, innovation.abs(), consensus, support],
            dim=1,
        )
        logit = self.net(x)
        return torch.sigmoid(logit), logit


class DUCFSafeFusion(nn.Module):
    """
    One-scale safe DUCF block.

    Important stability contract
    ----------------------------
    1) raw ADD path is immutable: base = rgb + tir
    2) raw modalities are never multiplied by utility masks
    3) residual gains start at 0
    4) residual gains are bounded by configurable maximum ratios
    5) residual RMS is matched to detached base RMS before gain application
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        hidden_channels: int = 64,
        use_target_support: bool = True,
        max_consensus_ratio: float = 0.15,
        max_innovation_ratio: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.rgb_channels = int(rgb_channels)
        self.tir_channels = int(tir_channels)
        self.hidden_channels = int(hidden_channels)
        self.use_target_support = bool(use_target_support)
        self.max_consensus_ratio = float(max_consensus_ratio)
        self.max_innovation_ratio = float(max_innovation_ratio)
        self.eps = float(eps)

        if self.rgb_channels != self.tir_channels:
            raise ValueError(
                "DUCF-Safe strict symmetric mode requires equal RGB/TIR channels, "
                f"got RGB={self.rgb_channels}, TIR={self.tir_channels}."
            )
        if self.hidden_channels < 8:
            raise ValueError("hidden_channels must be >= 8")
        if self.max_consensus_ratio < 0 or self.max_innovation_ratio < 0:
            raise ValueError("maximum residual ratios must be non-negative")

        c = self.rgb_channels
        h = self.hidden_channels

        # Shared representation mapping -> no modality-specific parameter prior.
        self.shared_embed = nn.Sequential(
            ConvGNAct(c, h, 1),
            ConvGNAct(h, h, 3, groups=h),
        )

        self.shared_predictor = SharedPredictor(h)

        # Exchange-invariant common context.
        self.consensus_encoder = nn.Sequential(
            ConvGNAct(3 * h, h, 1),
            ConvGNAct(h, h, 3, groups=h),
        )

        self.shared_utility = SharedUtilityScorer(h)

        self.shared_innovation_encoder = nn.Sequential(
            ConvGNAct(h, h, 1),
            ConvGNAct(h, h, 3, groups=h),
        )

        self.consensus_to_out = nn.Conv2d(h, c, 1, bias=False)
        self.innovation_to_out = nn.Conv2d(h, c, 1, bias=False)

        # Only these two scalars can move the output away from ADD initially.
        self.consensus_gain_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.innovation_gain_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))

        self._last_debug: Dict[str, torch.Tensor] = {}

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
        if support.ndim != 4 or int(support.shape[0]) != b:
            raise ValueError(
                "support must be [B,1,H,W] or [B,H,W], "
                f"got {tuple(support.shape)}"
            )
        if int(support.shape[1]) != 1:
            support = support.mean(dim=1, keepdim=True)

        # CRITICAL:
        # alignment support is context only.  Do NOT back-propagate DUCF
        # gradients through joint_target_support.
        #
        # The centered alignment builds this support as:
        #     sqrt(rgb_local_targetness * aligned_tir_local_targetness)
        # and many background pixels are exactly zero.  sqrt(0) is finite in
        # the forward pass but has an infinite derivative at zero.  If DUCF
        # keeps this tensor attached, a zero residual gain can still produce
        # the classic 0 * inf -> NaN pattern during backward and contaminate
        # the RGB/TIR backbones.
        support = support.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        )

        if support.shape[-2:] != (h, w):
            support = F.interpolate(
                support,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        # Avoid in-place modification of a detached view that may share
        # storage with alignment_info.
        return support.clamp(0.0, 1.0)

    def _rms_match_to_base(
        self,
        delta: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        """
        Match residual RMS to detached ADD-base RMS per sample.

        The subsequent bounded scalar gain is therefore interpretable as a
        maximum fraction of the current base feature energy.
        """
        dims = (1, 2, 3)
        delta_rms = delta.float().square().mean(dims, keepdim=True).add(self.eps).sqrt()
        base_rms = (
            base.detach().float().square().mean(dims, keepdim=True).add(self.eps).sqrt()
        )
        scaled = delta.float() / delta_rms * base_rms
        return scaled.to(dtype=delta.dtype)

    def forward(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        support: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        if rgb.ndim != 4 or tir.ndim != 4:
            raise ValueError(
                f"DUCF-Safe expects BCHW, got RGB={tuple(rgb.shape)}, TIR={tuple(tir.shape)}"
            )
        if int(rgb.shape[1]) != self.rgb_channels:
            raise ValueError(
                f"RGB channels mismatch: {rgb.shape[1]} != {self.rgb_channels}"
            )
        if int(tir.shape[1]) != self.tir_channels:
            raise ValueError(
                f"TIR channels mismatch: {tir.shape[1]} != {self.tir_channels}"
            )

        tir = self._resize_like(tir, rgb)

        # --------------------------------------------------------------
        # CRITICAL AMP SAFETY CONTRACT
        # --------------------------------------------------------------
        # Keep the exact original ADD anchor in the incoming feature dtype.
        # This is the feature seen by the pretrained YOLO neck at init.
        base = rgb + tir

        # All NEW DUCF computations run in FP32 even when the surrounding
        # detector is under CUDA autocast(float16).  In v2 the two scalar
        # residual gains were multiplied with FP16 feature maps.  Their
        # backward reduction spans millions of P3 elements and can overflow
        # under GradScaler, causing scaler.step(optimizer) to be skipped.
        #
        # Running only the small DUCF branch in FP32 keeps the old detector,
        # alignment, BHLR and neck under the existing AMP policy.
        if rgb.device.type == "cuda":
            amp_guard = torch.autocast(device_type="cuda", enabled=False)
        else:
            from contextlib import nullcontext
            amp_guard = nullcontext()

        with amp_guard:
            rgb32 = rgb.float()
            tir32 = tir.float()
            base32 = base.float()

            r = self.shared_embed(rgb32)
            t = self.shared_embed(tir32)

            pred_r_from_t = self.shared_predictor(t)
            pred_t_from_r = self.shared_predictor(r)
            innovation_r = r - pred_r_from_t
            innovation_t = t - pred_t_from_r

            consensus = self.consensus_encoder(
                torch.cat(
                    [0.5 * (r + t), (r - t).abs(), r * t],
                    dim=1,
                )
            )

            support_map = self._prepare_support(support, r)

            q_rgb, logit_rgb = self.shared_utility(
                r, t, innovation_r, consensus, support_map
            )
            q_tir, logit_tir = self.shared_utility(
                t, r, innovation_t, consensus, support_map
            )

            enc_r = self.shared_innovation_encoder(innovation_r)
            enc_t = self.shared_innovation_encoder(innovation_t)
            innovation_mix = q_rgb * enc_r + q_tir * enc_t

            consensus_delta = self.consensus_to_out(consensus)
            innovation_delta = self.innovation_to_out(innovation_mix)

            consensus_delta = self._rms_match_to_base(consensus_delta, base32)
            innovation_delta = self._rms_match_to_base(innovation_delta, base32)

            # Scalar gain math stays FP32.
            g_consensus = (
                self.max_consensus_ratio
                * torch.tanh(self.consensus_gain_raw.float())
            )
            g_innovation = (
                self.max_innovation_ratio
                * torch.tanh(self.innovation_gain_raw.float())
            )

            correction32 = (
                g_consensus * consensus_delta.float()
                + g_innovation * innovation_delta.float()
            )

        # Cast only the bounded correction back to the detector feature dtype.
        # At initialization both gains are exactly 0, therefore:
        #     fused == base == rgb + tir
        correction = correction32.to(dtype=base.dtype)
        fused = base + correction

        with torch.no_grad():
            base_rms = base.float().square().mean().sqrt()
            corr_rms = correction32.float().square().mean().sqrt()
            ratio = corr_rms / base_rms.clamp_min(self.eps)
            self._last_debug = {
                "q_rgb_mean": q_rgb.mean().detach(),
                "q_tir_mean": q_tir.mean().detach(),
                "q_rgb_std": q_rgb.std(unbiased=False).detach(),
                "q_tir_std": q_tir.std(unbiased=False).detach(),
                "innovation_rgb_abs_mean": innovation_r.abs().mean().detach(),
                "innovation_tir_abs_mean": innovation_t.abs().mean().detach(),
                "support_mean": support_map.mean().detach(),
                "consensus_gain": g_consensus.detach(),
                "innovation_gain": g_innovation.detach(),
                "base_rms": base_rms.detach(),
                "correction_rms": corr_rms.detach(),
                "correction_to_base_rms": ratio.detach(),
                "ducf_internal_dtype_fp32": torch.tensor(
                    1.0, device=base.device
                ),
                "support_detached": torch.tensor(
                    1.0, device=base.device
                ),
            }

        if not return_debug:
            return fused

        return fused, {
            "q_rgb": q_rgb,
            "q_tir": q_tir,
            "utility_logit_rgb": logit_rgb,
            "utility_logit_tir": logit_tir,
            "innovation_rgb_energy": innovation_r.square().mean(dim=1, keepdim=True),
            "innovation_tir_energy": innovation_t.square().mean(dim=1, keepdim=True),
            "support": support_map,
            "consensus": consensus,
            "consensus_gain": g_consensus,
            "innovation_gain": g_innovation,
            "base": base,
            "correction": correction,
        }

    def detached_stats(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_debug)


__all__ = ["DUCFSafeFusion", "SharedUtilityScorer", "SharedPredictor"]
