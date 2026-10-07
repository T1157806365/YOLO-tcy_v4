"""
Optional reliability-gated geometry for RLSFA + Local-GTGA Centered Alignment.

Purpose
-------
Keep the original geometry estimator unchanged:

    coarse anchor c_R
    source center c_T
    anisotropic sampling scale S

and only decide how strongly the estimated geometry should be executed.

Original center-preserving inverse warp:
    q = c_T + S * (p - c_R)

Reliability-gated geometry:
    c_T_eff = c_R + rho_t * (c_T - c_R)
    S_eff   = I   + rho_s * (S - I)

    q = c_T_eff + S_eff * (p - c_R)

Properties
----------
rho = 1:
    exactly the original alignment.

rho = 0:
    identity geometry:
        c_T_eff = c_R
        S_eff = I
        q = p

The reliability signal reuses the confidence that the existing explicit
coarse matcher already computes:

    geometry["coarse_correlation_confidence"]

No new trainable parameters are introduced, so this module does not add
optimizer/DDP parameters and preserves checkpoint parameter compatibility.

Recommended integration
-----------------------
Do NOT replace models/modules/alignment/rlsfa_gtga_centered.py.

Save this file as:

    models/modules/alignment/alignment_reliability_gate.py

Then let the detector wrapper select either:
    RLSFAGTGACenteredAlignment
or:
    ReliabilityGatedRLSFAGTGACenteredAlignment

through YAML.

This keeps the ablation clean:
    enabled: false -> original alignment
    enabled: true  -> original alignment + reliability gate
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rlsfa_gtga_centered import (
    RLSFAGTGACenteredAlignment,
    warp_with_geometry,
)


class GeometryReliabilityGate(nn.Module):
    """
    Parameter-free confidence gate for the estimated centered geometry.

    Config keys
    -----------
    enabled:
        bool, default True.

    power:
        Confidence calibration:
            rho = confidence ** power
        default 0.5.

        power < 1 makes a conservative confidence less suppressive.
        Example:
            confidence=0.04, power=0.5 -> rho=0.20.

    min_reliability:
        Lower floor after confidence calibration.
        default 0.0.

    max_reliability:
        Upper ceiling.
        default 1.0.

    detach_confidence:
        Detach rho from the computation graph.
        default True.

        This is recommended because rho is a reliability controller,
        while the original alignment auxiliary losses should continue
        to supervise the raw geometry estimator itself.

    gate_translation:
        Whether rho controls c_T - c_R.
        default True.

    gate_scale:
        Whether rho controls S - I.
        default True.

    translation_power:
        Optional additional exponent for translation reliability.
        default 1.0.

    scale_power:
        Optional additional exponent for scale reliability.
        default 1.0.
        Setting this > 1 makes scale correction more conservative.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None):
        super().__init__()

        cfg = dict(cfg or {})

        self.enabled = bool(cfg.get("enabled", True))

        self.power = float(cfg.get("power", 0.5))
        self.min_reliability = float(cfg.get("min_reliability", 0.0))
        self.max_reliability = float(cfg.get("max_reliability", 1.0))

        self.detach_confidence = bool(cfg.get("detach_confidence", True))

        self.gate_translation = bool(cfg.get("gate_translation", True))
        self.gate_scale = bool(cfg.get("gate_scale", True))

        self.translation_power = float(cfg.get("translation_power", 1.0))
        self.scale_power = float(cfg.get("scale_power", 1.0))

        if self.power <= 0:
            raise ValueError(
                f"reliability_gate.power must be > 0, got {self.power}"
            )

        if self.translation_power <= 0:
            raise ValueError(
                "reliability_gate.translation_power must be > 0, "
                f"got {self.translation_power}"
            )

        if self.scale_power <= 0:
            raise ValueError(
                "reliability_gate.scale_power must be > 0, "
                f"got {self.scale_power}"
            )

        if not (0.0 <= self.min_reliability <= 1.0):
            raise ValueError(
                "reliability_gate.min_reliability must be in [0,1], "
                f"got {self.min_reliability}"
            )

        if not (0.0 <= self.max_reliability <= 1.0):
            raise ValueError(
                "reliability_gate.max_reliability must be in [0,1], "
                f"got {self.max_reliability}"
            )

        if self.min_reliability > self.max_reliability:
            raise ValueError(
                "reliability_gate.min_reliability cannot exceed "
                "max_reliability."
            )

    @staticmethod
    def _as_b1(confidence: torch.Tensor) -> torch.Tensor:
        """Convert confidence to [B,1] while preserving device/dtype."""
        if confidence.ndim == 4:
            confidence = confidence[:, 0, 0, 0]
        elif confidence.ndim == 3:
            confidence = confidence[:, 0, 0]
        elif confidence.ndim == 2:
            confidence = confidence[:, 0]
        elif confidence.ndim != 1:
            raise ValueError(
                "coarse_correlation_confidence must be [B], [B,1], "
                f"[B,1,1] or [B,1,1,1], got {tuple(confidence.shape)}"
            )
        return confidence.view(-1, 1)

    def _calibrate(
        self,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self._as_b1(confidence).float().clamp(0.0, 1.0)

        if self.detach_confidence:
            raw_for_gate = raw.detach()
        else:
            raw_for_gate = raw

        rho = raw_for_gate.pow(self.power)

        if self.min_reliability > 0.0:
            rho = (
                self.min_reliability
                + (1.0 - self.min_reliability) * rho
            )

        rho = rho.clamp(
            self.min_reliability,
            self.max_reliability,
        )

        rho_t = (
            rho.pow(self.translation_power)
            if self.gate_translation
            else torch.ones_like(rho)
        )

        rho_s = (
            rho.pow(self.scale_power)
            if self.gate_scale
            else torch.ones_like(rho)
        )

        return raw, rho_t, rho_s

    def forward(
        self,
        geometry: Mapping[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        required = (
            "coarse_anchor",
            "source_center",
            "sampling_scale",
            "coarse_correlation_confidence",
        )

        missing = [
            key
            for key in required
            if key not in geometry
        ]

        if missing:
            raise KeyError(
                "GeometryReliabilityGate missing geometry keys: "
                + ", ".join(missing)
            )

        c_r = geometry["coarse_anchor"]
        c_t = geometry["source_center"]
        scale = geometry["sampling_scale"]

        if c_r.ndim != 2 or c_r.shape[1] != 2:
            raise ValueError(
                f"coarse_anchor must be [B,2], got {tuple(c_r.shape)}"
            )
        if c_t.ndim != 2 or c_t.shape[1] != 2:
            raise ValueError(
                f"source_center must be [B,2], got {tuple(c_t.shape)}"
            )
        if scale.ndim != 2 or scale.shape[1] != 2:
            raise ValueError(
                f"sampling_scale must be [B,2], got {tuple(scale.shape)}"
            )

        if not self.enabled:
            one = torch.ones(
                c_r.shape[0],
                1,
                device=c_r.device,
                dtype=c_r.dtype,
            )
            translation = c_t - scale * c_r

            gated = dict(geometry)
            gated["effective_source_center"] = c_t
            gated["effective_sampling_scale"] = scale
            gated["effective_sampling_translation"] = translation

            info = {
                "raw_confidence":
                    self._as_b1(
                        geometry["coarse_correlation_confidence"]
                    ).to(dtype=c_r.dtype),
                "translation_reliability": one,
                "scale_reliability": one,
                "reliability": one,
            }
            return gated, info

        raw, rho_t, rho_s = self._calibrate(
            geometry["coarse_correlation_confidence"]
        )

        raw = raw.to(device=c_r.device, dtype=c_r.dtype)
        rho_t = rho_t.to(device=c_r.device, dtype=c_r.dtype)
        rho_s = rho_s.to(device=c_r.device, dtype=c_r.dtype)

        # ----------------------------------------------------------
        # Translation gate
        #
        # c_T_eff = c_R + rho_t * (c_T - c_R)
        # ----------------------------------------------------------
        effective_source_center = (
            c_r
            + rho_t * (c_t - c_r)
        )

        # ----------------------------------------------------------
        # Scale gate
        #
        # S_eff = I + rho_s * (S - I)
        # ----------------------------------------------------------
        identity_scale = torch.ones_like(scale)

        effective_scale = (
            identity_scale
            + rho_s * (scale - identity_scale)
        )

        # ----------------------------------------------------------
        # Rebuild translation for the same centered inverse warp:
        #
        # q = S_eff * p + t_eff
        # t_eff = c_T_eff - S_eff * c_R
        #
        # Therefore:
        # p=c_R -> q=c_T_eff
        # ----------------------------------------------------------
        effective_translation = (
            effective_source_center
            - effective_scale * c_r
        )

        gated = dict(geometry)

        # Keep the original/raw keys untouched for auxiliary losses.
        gated["raw_sampling_scale"] = geometry["sampling_scale"]
        gated["raw_sampling_translation"] = geometry["sampling_translation"]
        gated["raw_source_center"] = geometry["source_center"]

        # Actual geometry used by the warp is stored under explicit keys.
        gated["effective_source_center"] = effective_source_center
        gated["effective_sampling_scale"] = effective_scale
        gated["effective_sampling_translation"] = effective_translation

        reliability = torch.sqrt(
            (rho_t * rho_s).clamp_min(0.0)
        )

        info = {
            "raw_confidence": raw,
            "translation_reliability": rho_t,
            "scale_reliability": rho_s,
            "reliability": reliability,
        }

        return gated, info


class ReliabilityGatedRLSFAGTGACenteredAlignment(
    RLSFAGTGACenteredAlignment
):
    """
    Drop-in alternative to RLSFAGTGACenteredAlignment.

    The original RLSFA + Local-GTGA estimator is reused unchanged.
    Only the final geometry EXECUTION is reliability gated.

    Because the gate contains no trainable parameters, the original
    alignment state_dict keys are preserved.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(cfg=cfg)

        gate_cfg = dict(
            (self.cfg.get("reliability_gate", {}) or {})
        )

        # This class is selected only when enabled by the wrapper.
        gate_cfg.setdefault("enabled", True)

        self.geometry_reliability_gate = GeometryReliabilityGate(
            gate_cfg
        )

    def _estimate_and_gate_reference(
        self,
        rgb_ref: torch.Tensor,
        tir_ref: torch.Tensor,
        rgb_valid_mask: Optional[torch.Tensor],
        tir_valid_mask: Optional[torch.Tensor],
    ):
        # Match the original unit.forward() behavior:
        # resize TIR reference to RGB reference first.
        tir_ref_resized = self.unit._resize_like(
            tir_ref,
            rgb_ref,
        )

        raw_geometry = self.unit.estimate_geometry(
            rgb_ref,
            tir_ref_resized,
            rgb_valid_mask,
            tir_valid_mask,
        )

        gated_geometry, gate_info = self.geometry_reliability_gate(
            raw_geometry
        )

        effective_scale = gated_geometry[
            "effective_sampling_scale"
        ]
        effective_translation = gated_geometry[
            "effective_sampling_translation"
        ]

        aligned_ref = warp_with_geometry(
            tir_ref_resized,
            effective_scale,
            effective_translation,
            output_hw=rgb_ref.shape[-2:],
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        aligned_tir_local_target = warp_with_geometry(
            raw_geometry["tir_local_targetness"],
            effective_scale,
            effective_translation,
            output_hw=rgb_ref.shape[-2:],
            padding_mode="zeros",
            align_corners=self.align_corners,
        ).clamp(0, 1)

        joint_ref = torch.sqrt(
            (
                raw_geometry["rgb_local_targetness"]
                .clamp(0, 1)
                * aligned_tir_local_target
            ).clamp_min(0)
        )

        joint_ref = (
            joint_ref
            * raw_geometry["rgb_valid_mask"]
        ).clamp(0, 1)

        debug = {
            # IMPORTANT:
            # Keep raw geometry keys such as sampling_scale and
            # coarse_offset for the existing auxiliary loss.
            **raw_geometry,

            "aligned_tir":
                aligned_ref,

            "aligned_tir_local_targetness":
                aligned_tir_local_target,

            "joint_target_support":
                joint_ref,

            "bhlr_guidance_map":
                joint_ref,

            "scale_pivot_rgb":
                raw_geometry["coarse_anchor"],

            "scale_pivot_tir":
                raw_geometry["source_center"],

            "effective_scale_pivot_tir":
                gated_geometry["effective_source_center"],

            "raw_sampling_scale":
                raw_geometry["sampling_scale"],

            "raw_sampling_translation":
                raw_geometry["sampling_translation"],

            "effective_sampling_scale":
                effective_scale,

            "effective_sampling_translation":
                effective_translation,

            "alignment_reliability":
                gate_info["reliability"],

            "translation_reliability":
                gate_info["translation_reliability"],

            "scale_reliability":
                gate_info["scale_reliability"],

            "raw_alignment_confidence":
                gate_info["raw_confidence"],

            "reliability_gate_enabled":
                True,

            "center_preserving_scale":
                True,

            "frequency_refinement":
                False,

            "fine_translation":
                False,

            "scale_candidate_search":
                False,
        }

        return (
            aligned_ref,
            joint_ref,
            raw_geometry,
            gated_geometry,
            gate_info,
            debug,
        )

    def align_pyramid(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        rgb_valid_mask: Optional[torch.Tensor] = None,
        tir_valid_mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        ref_idx = int(
            scale_to_index[
                self.reference_scale
            ]
        )

        rgb_ref = rgb_features[
            ref_idx
        ]

        tir_ref = tir_features[
            ref_idx
        ]

        (
            aligned_ref,
            joint_ref,
            raw_geometry,
            gated_geometry,
            gate_info,
            debug,
        ) = self._estimate_and_gate_reference(
            rgb_ref=rgb_ref,
            tir_ref=tir_ref,
            rgb_valid_mask=rgb_valid_mask,
            tir_valid_mask=tir_valid_mask,
        )

        effective_scale = gated_geometry[
            "effective_sampling_scale"
        ]

        effective_translation = gated_geometry[
            "effective_sampling_translation"
        ]

        aligned = list(
            tir_features
        )

        support_by_scale = {}

        for scale in self.apply_scales:
            idx = int(
                scale_to_index[
                    scale
                ]
            )

            if scale == self.reference_scale:
                aligned[idx] = aligned_ref

            else:
                aligned[idx] = warp_with_geometry(
                    tir_features[idx],
                    effective_scale,
                    effective_translation,
                    output_hw=rgb_features[idx].shape[-2:],
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                )

            support_by_scale[scale] = F.interpolate(
                joint_ref,
                size=rgb_features[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0, 1)

        info = {
            "enabled":
                True,

            "type":
                "rlsfa_gtga_centered_reliability_gated",

            "algorithm":
                "rlsfa_coarse_anchor+local_gtga_scale"
                "+reliability_gated_center_preserving_one_warp",

            "reference_scale":
                self.reference_scale,

            "apply_scales":
                list(self.apply_scales),

            # Raw estimator outputs.
            "coarse_anchor":
                raw_geometry["coarse_anchor"],

            "source_center":
                raw_geometry["source_center"],

            "raw_sampling_scale":
                raw_geometry["sampling_scale"],

            "raw_sampling_translation":
                raw_geometry["sampling_translation"],

            "coarse_offset":
                raw_geometry["coarse_offset"],

            "coarse_confidence":
                raw_geometry["coarse_correlation_confidence"],

            # Actual geometry used for ONE warp.
            "effective_source_center":
                gated_geometry["effective_source_center"],

            "sampling_scale":
                effective_scale,

            "sampling_translation":
                effective_translation,

            # Reliability.
            "reliability_gate_enabled":
                True,

            "alignment_reliability":
                gate_info["reliability"],

            "translation_reliability":
                gate_info["translation_reliability"],

            "scale_reliability":
                gate_info["scale_reliability"],

            "raw_alignment_confidence":
                gate_info["raw_confidence"],

            # BHLR guidance after the SAME gated geometry.
            "joint_target_support":
                joint_ref,

            "bhlr_guidance_map":
                joint_ref,

            "support_by_scale":
                support_by_scale,

            "guidance_by_scale": {
                s: {
                    "joint_target_support":
                        m
                }
                for s, m
                in support_by_scale.items()
            },

            "center_preserving_scale":
                True,

            "has_frequency_refinement":
                False,

            "has_fine_translation":
                False,

            "has_scale_candidate_search":
                False,
        }

        if return_debug:
            info["scale_debug"] = {
                self.reference_scale:
                    debug
            }
            info["debug"] = debug

        return aligned, info

    def forward(
        self,
        rgb_features,
        tir_features,
        scale_to_index,
        return_debug=False,
        rgb_valid_mask=None,
        tir_valid_mask=None,
    ):
        return self.align_pyramid(
            rgb_features=rgb_features,
            tir_features=tir_features,
            scale_to_index=scale_to_index,
            rgb_valid_mask=rgb_valid_mask,
            tir_valid_mask=tir_valid_mask,
            return_debug=return_debug,
        )


__all__ = [
    "GeometryReliabilityGate",
    "ReliabilityGatedRLSFAGTGACenteredAlignment",
]


# ----------------------------------------------------------------------
# Small self-test:
#
# python -m models.modules.alignment.alignment_reliability_gate
#
# Only tests the geometry gating equations; no detector is required.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    gate = GeometryReliabilityGate({
        "enabled": True,
        "power": 1.0,
        "min_reliability": 0.0,
        "max_reliability": 1.0,
        "detach_confidence": True,
    })

    c_r = torch.tensor([
        [0.40, 0.50],
        [0.40, 0.50],
    ])

    c_t = torch.tensor([
        [0.50, 0.60],
        [0.50, 0.60],
    ])

    scale = torch.tensor([
        [1.20, 0.80],
        [1.20, 0.80],
    ])

    translation = (
        c_t
        - scale * c_r
    )

    geometry = {
        "coarse_anchor":
            c_r,

        "source_center":
            c_t,

        "sampling_scale":
            scale,

        "sampling_translation":
            translation,

        "coarse_correlation_confidence":
            torch.tensor([
                [[[0.0]]],
                [[[1.0]]],
            ]),
    }

    gated, info = gate(
        geometry
    )

    # sample 0: rho=0 -> identity
    assert torch.allclose(
        gated[
            "effective_sampling_scale"
        ][0],
        torch.ones(2),
        atol=1e-6,
    )

    assert torch.allclose(
        gated[
            "effective_sampling_translation"
        ][0],
        torch.zeros(2),
        atol=1e-6,
    )

    # sample 1: rho=1 -> original
    assert torch.allclose(
        gated[
            "effective_sampling_scale"
        ][1],
        scale[1],
        atol=1e-6,
    )

    assert torch.allclose(
        gated[
            "effective_sampling_translation"
        ][1],
        translation[1],
        atol=1e-6,
    )

    print(
        "Alignment reliability gate self-test: PASS"
    )
