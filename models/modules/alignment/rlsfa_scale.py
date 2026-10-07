"""
RLSFA-Scale legacy filename -- COARSE-ONLY implementation
=========================================================

IMPORTANT
---------
The filename/class names are intentionally kept compatible with the existing
YOLO-tcy_v3 RLSFA-Scale builder/factory, but the internal algorithm has been
simplified to:

    RGB feature ----\
                     -> targetness-weighted explicit correlation
    TIR feature ----/                |
                                      v
                               coarse offset
                                      |
                                      v
                          warp ORIGINAL TIR once
                                      |
                                      v
                               aligned TIR

REMOVED COMPLETELY
------------------
- isotropic scale search
- spatial-structure refinement
- local/window FFT
- RGB/TIR frequency reliability
- fine descriptor matching
- fine residual translation
- joint_target_support

The only geometric degree of freedom is one GLOBAL translation per sample:
    [dx, dy] in feature pixels.

Sampling convention
-------------------
grid_sample uses:
    output(p) = source(p + delta)

Therefore, for RGB-reference alignment, the bbox-derived sampling target is:
    delta_gt = C_TIR - C_RGB
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlignment


class ConvBNAct(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                int(c1),
                int(c2),
                kernel_size=int(k),
                stride=1,
                padding=int(k) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(int(c2)),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class LazyProj(nn.Module):
    """
    1x1 lazy projection so the alignment module does not need to know the
    backbone channel count at construction time.
    """

    def __init__(
        self,
        out_channels: int,
    ):
        super().__init__()

        self.conv = nn.LazyConv2d(
            int(out_channels),
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            int(out_channels)
        )

        self.act = nn.SiLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.act(
            self.bn(
                self.conv(x)
            )
        )


def correlation_translation(
    reference: torch.Tensor,
    source: torch.Tensor,
    weight_map: torch.Tensor,
    radius: int,
    temperature: float,
):
    """
    Explicit weighted translation search.

    Parameters
    ----------
    reference:
        RGB projected feature, [B,C,H,W].
    source:
        TIR projected feature, [B,C,H,W].
    weight_map:
        RGB-grid weighting map, [B,1,H,W].
        It is used ONLY inside coarse matching.
    radius:
        Candidate dx/dy search radius in feature pixels.
    temperature:
        Softmax temperature.

    Returns
    -------
    expected_offset:
        [B,2,1,1], order = [dx, dy].
    confidence:
        [B,1,1,1], peakiness of the candidate distribution.
    probability:
        [B,K], K=(2r+1)^2.
    score_logits:
        [B,K], raw correlation scores before temperature softmax.
    """
    r = int(radius)

    if r < 0:
        raise ValueError(
            f"coarse_radius must be >= 0, got {r}"
        )

    temp = max(
        float(temperature),
        1e-4,
    )

    if reference.shape[-2:] != source.shape[-2:]:
        raise ValueError(
            "reference/source spatial size mismatch: "
            f"{tuple(reference.shape[-2:])} vs "
            f"{tuple(source.shape[-2:])}"
        )

    ref = F.normalize(
        reference.float(),
        p=2,
        dim=1,
        eps=1e-6,
    )

    src = F.normalize(
        source.float(),
        p=2,
        dim=1,
        eps=1e-6,
    )

    weight = (
        weight_map.float()
        .clamp(min=0.0)
    )

    if weight.shape[-2:] != ref.shape[-2:]:
        weight = F.interpolate(
            weight,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    den = (
        weight.sum(
            dim=(-2, -1)
        )
        .clamp(min=1e-6)
    )  # [B,1]

    _, _, h, w = src.shape

    padded = F.pad(
        src,
        (r, r, r, r),
        mode="replicate",
    )

    all_scores = []
    displacements = []

    # Only 2r+1 outer loops; dx candidates are stacked together.
    for dy in range(
        -r,
        r + 1,
    ):
        shifted_dx = []

        for dx in range(
            -r,
            r + 1,
        ):
            y0 = r + dy
            x0 = r + dx

            shifted_dx.append(
                padded[
                    :,
                    :,
                    y0:y0 + h,
                    x0:x0 + w,
                ]
            )

            displacements.append(
                (
                    float(dx),
                    float(dy),
                )
            )

        shifted = torch.stack(
            shifted_dx,
            dim=2,
        )  # [B,C,Kx,H,W]

        corr = (
            ref.unsqueeze(2)
            * shifted
        ).sum(
            dim=1
        )  # [B,Kx,H,W]

        score = (
            (corr * weight)
            .sum(
                dim=(-2, -1)
            )
            / den
        )  # [B,Kx]

        all_scores.append(
            score
        )

    score_logits = torch.cat(
        all_scores,
        dim=1,
    )  # [B,K]

    probability = torch.softmax(
        score_logits / temp,
        dim=1,
    )

    disp_tensor = torch.tensor(
        displacements,
        device=probability.device,
        dtype=probability.dtype,
    )  # [K,2]

    expected = (
        probability
        @ disp_tensor
    )  # [B,2]

    k = int(
        probability.shape[1]
    )

    max_prob = (
        probability.max(
            dim=1,
            keepdim=True,
        ).values
    )

    uniform_prob = (
        1.0
        / max(
            k,
            1,
        )
    )

    confidence = (
        (
            max_prob
            - uniform_prob
        )
        / max(
            1.0 - uniform_prob,
            1e-6,
        )
    ).clamp(
        0.0,
        1.0,
    )

    return (
        expected.view(
            -1,
            2,
            1,
            1,
        ),
        confidence.view(
            -1,
            1,
            1,
            1,
        ),
        probability,
        score_logits,
    )


class RLSFAScaleUnit(nn.Module):
    """
    Legacy class name retained for factory/checkpoint-code compatibility.

    This is now a COARSE-ONLY translation unit.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        coarse_radius: int = 20,
        coarse_temperature: float = 0.1,
        use_targetness: bool = True,
        coarse_confidence_gate: bool = False,

        # -----------------------------------------------------
        # Obsolete arguments are accepted only so an older
        # builder/config does not crash. They are NOT used.
        # -----------------------------------------------------
        scale_min=None,
        scale_max=None,
        scale_steps=None,
        scale_temperature=None,
        spatial_kernel_sizes=None,
        window_sizes=None,
        window_overlap=None,
        use_hann_window=None,
        freq_cutoff=None,
        freq_sharpness=None,
        reliability_channels=None,
        fine_radius=None,
        max_fine_offset=None,
        fine_temperature=None,
        **_ignored,
    ):
        super().__init__()

        c = int(
            hidden_channels
        )

        self.hidden_channels = c

        self.coarse_radius = int(
            coarse_radius
        )

        self.coarse_temperature = float(
            coarse_temperature
        )

        self.use_targetness = bool(
            use_targetness
        )

        self.coarse_confidence_gate = bool(
            coarse_confidence_gate
        )

        self.rgb_proj = LazyProj(
            c
        )

        self.tir_proj = LazyProj(
            c
        )

        # Important:
        # this targetness is used only to weight COARSE correlation.
        # It is NOT exported to BHLR as guidance.
        self.targetness_head = nn.Sequential(
            ConvBNAct(
                c * 3,
                c,
                3,
            ),
            nn.Conv2d(
                c,
                1,
                kernel_size=1,
                bias=True,
            ),
        )

        # Start from spatially uniform 0.5 targetness.
        nn.init.zeros_(
            self.targetness_head[
                -1
            ].weight
        )

        nn.init.zeros_(
            self.targetness_head[
                -1
            ].bias
        )

    @staticmethod
    def _resize_like(
        source: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if (
            source.shape[-2:]
            == reference.shape[-2:]
        ):
            return source

        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _expand_translation(
        offset: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        if (
            offset.shape[-2:]
            == (h, w)
        ):
            return offset

        return offset.expand(
            -1,
            -1,
            h,
            w,
        )

    @staticmethod
    def warp_translation(
        source: torch.Tensor,
        offset_px: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp source with GLOBAL [dx,dy] translation in feature pixels.

        grid_sample convention:
            output(p) = source(p + delta)
        """
        b, _, h, w = (
            source.shape
        )

        offset_px = (
            RLSFAScaleUnit
            ._expand_translation(
                offset_px,
                h,
                w,
            )
        )

        device = (
            source.device
        )

        dtype = (
            source.dtype
        )

        ys = torch.linspace(
            -1.0,
            1.0,
            h,
            device=device,
            dtype=dtype,
        )

        xs = torch.linspace(
            -1.0,
            1.0,
            w,
            device=device,
            dtype=dtype,
        )

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        base = torch.stack(
            (
                xx,
                yy,
            ),
            dim=-1,
        )[None].expand(
            b,
            -1,
            -1,
            -1,
        )

        dx = (
            2.0
            * offset_px[:, 0]
            / max(
                w - 1,
                1,
            )
        )

        dy = (
            2.0
            * offset_px[:, 1]
            / max(
                h - 1,
                1,
            )
        )

        delta = torch.stack(
            (
                dx,
                dy,
            ),
            dim=-1,
        )

        return F.grid_sample(
            source,
            base + delta,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def forward(
        self,
        rgb_feat: torch.Tensor,
        tir_feat: torch.Tensor,
        return_debug: bool = False,
    ):
        # -----------------------------------------------------
        # A. Same spatial grid for correlation.
        # -----------------------------------------------------
        tir_feat = self._resize_like(
            tir_feat,
            rgb_feat,
        )

        # -----------------------------------------------------
        # B. Lightweight semantic projections.
        # -----------------------------------------------------
        rgb_projected = self.rgb_proj(
            rgb_feat
        )

        tir_projected = self.tir_proj(
            tir_feat
        )

        # -----------------------------------------------------
        # C. Weight map ONLY for coarse matching.
        # -----------------------------------------------------
        target_input = torch.cat(
            (
                rgb_projected,
                tir_projected,
                (
                    rgb_projected
                    - tir_projected
                ).abs(),
            ),
            dim=1,
        )

        target_logits = self.targetness_head(
            target_input
        )

        if self.use_targetness:
            targetness = torch.sigmoid(
                target_logits
            )
        else:
            targetness = torch.ones_like(
                target_logits
            )

        # -----------------------------------------------------
        # D. SINGLE coarse explicit correlation search.
        # -----------------------------------------------------
        (
            coarse_offset,
            coarse_confidence,
            coarse_probability,
            coarse_score_logits,
        ) = correlation_translation(
            rgb_projected,
            tir_projected,
            targetness,
            self.coarse_radius,
            self.coarse_temperature,
        )

        # Optional safety gate.
        # Default is OFF because the user's current observation is that the
        # coarse estimate itself is already accurate.
        effective_offset = (
            coarse_offset
            * coarse_confidence
            if self.coarse_confidence_gate
            else coarse_offset
        )

        # -----------------------------------------------------
        # E. ONE warp. STOP HERE.
        # -----------------------------------------------------
        aligned_tir = self.warp_translation(
            tir_feat,
            effective_offset,
        )

        if not return_debug:
            return aligned_tir

        return (
            aligned_tir,
            {
                "coarse_targetness_logits":
                    target_logits,

                "coarse_targetness":
                    targetness,

                "coarse_correlation_confidence":
                    coarse_confidence,

                "coarse_probability":
                    coarse_probability,

                "coarse_score_logits":
                    coarse_score_logits,

                "coarse_offset":
                    coarse_offset,

                "effective_offset":
                    effective_offset,

                # Compatibility alias for old visualizers.
                # It is exactly the same as coarse_offset now.
                "total_offset":
                    effective_offset,

                "aligned_tir":
                    aligned_tir,

                # Useful aliases.
                "tir_coarse":
                    aligned_tir,

                "coarse_only":
                    True,
            },
        )


class RLSFAScaleAlignment(
    BaseAlignment
):
    """
    Drop-in replacement for the old RLSFA-Scale alignment plugin.

    Despite the legacy name, there is NO scale search anymore.
    """

    VALID_SCALES = (
        3,
        4,
        5,
    )

    def __init__(
        self,
        cfg: Dict[str, Any] | None = None,
    ):
        super().__init__(
            cfg=cfg
        )

        invalid = [
            s
            for s
            in self.scales
            if s
            not in self.VALID_SCALES
        ]

        if invalid:
            raise ValueError(
                "RLSFA coarse alignment supports "
                f"P3/P4/P5 only; invalid={invalid}"
            )

        hidden = int(
            self.cfg.get(
                "hidden_channels",
                32,
            )
        )

        self.units = nn.ModuleDict(
            {
                str(scale):
                    RLSFAScaleUnit(
                        hidden_channels=hidden,

                        coarse_radius=int(
                            self.cfg.get(
                                "coarse_radius",
                                20,
                            )
                        ),

                        coarse_temperature=float(
                            self.cfg.get(
                                "coarse_temperature",
                                0.1,
                            )
                        ),

                        use_targetness=bool(
                            self.cfg.get(
                                "use_targetness",
                                True,
                            )
                        ),

                        coarse_confidence_gate=bool(
                            self.cfg.get(
                                "coarse_confidence_gate",
                                False,
                            )
                        ),
                    )

                for scale
                in self.scales
            }
        )

    def forward(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        return_debug: bool = False,
    ):
        aligned = list(
            tir_features
        )

        info = {
            "enabled":
                True,

            # Keep legacy type so existing wrappers/factory do not reject it.
            "type":
                "rlsfa_scale",

            "algorithm":
                "coarse_translation_only",

            "scales":
                list(
                    self.scales
                ),

            "has_scale_search":
                False,

            "has_frequency_refinement":
                False,

            "has_fine_translation":
                False,

            "exports_bhlr_guidance":
                False,
        }

        debug_all = {}

        for scale in self.scales:
            if scale not in scale_to_index:
                raise KeyError(
                    f"Missing scale mapping for P{scale}; "
                    f"available={list(scale_to_index.keys())}"
                )

            idx = int(
                scale_to_index[
                    scale
                ]
            )

            unit = self.units[
                str(
                    scale
                )
            ]

            if return_debug:
                (
                    aligned_feat,
                    dbg,
                ) = unit(
                    rgb_features[
                        idx
                    ],
                    tir_features[
                        idx
                    ],
                    return_debug=True,
                )

                debug_all[
                    scale
                ] = dbg

            else:
                aligned_feat = unit(
                    rgb_features[
                        idx
                    ],
                    tir_features[
                        idx
                    ],
                    return_debug=False,
                )

            aligned[
                idx
            ] = aligned_feat

        if return_debug:
            info[
                "scale_debug"
            ] = debug_all

        return (
            aligned,
            info,
        )


# Compatibility aliases.
RLSFACoarseAlignment = RLSFAScaleAlignment
RLSFAAlignment = RLSFAScaleAlignment
