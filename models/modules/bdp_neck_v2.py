"""
BDP-Neck v2: Internal Detail Fusion before P3 refinement
========================================================

Motivation
----------
BHLR recovers high-resolution RGB detail at backbone P3.  BDP-Neck v1
re-injected the BHLR increment immediately before Detect.  That improved AP75
but reduced recall, suggesting that the detector received a detail residual
without subsequent semantic refinement.

BDP-Neck v2 moves the detail injection INSIDE the YOLO neck, immediately
before the final top-down P3 C3k2 refinement block:

    D3 = P3_BHLR - P3_raw

    top-down P3 concat feature N3_pre
              +
       gated BHLR detail D3
              |
              v
        P3 C3k2 refinement
              |
              +--> PAN bottom-up path
              |
              +--> Detect P3

The injected detail is therefore filtered by the native YOLO P3 refinement
block before it reaches Detect, and it can also participate in the following
bottom-up PAN path.

Design properties
-----------------
1. P3 only.  The high-resolution detail source remains exactly BHLR's P3
   increment; no TIR feature is mixed into D3.
2. No auxiliary loss.
3. The gate observes BOTH the neck feature and the detail feature:
       [Reduce(N), Reduce(D), |Reduce(N)-Reduce(D)|]
4. A bounded scalar residual gain is initialized to zero.  Therefore the
   initial forward pass is EXACTLY the original YOLO neck.
5. The detail projection is factorized through a compact hidden width to keep
   parameters/FLOPs low.
"""

from __future__ import annotations

from typing import Dict

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
    """Conv2d -> GroupNorm -> SiLU."""

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
        c1, c2 = int(c1), int(c2)
        k, s, groups = int(k), int(s), int(groups)
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


class BDPNeckInternalP3(nn.Module):
    """
    Fuse BHLR detail into the input of the native YOLO P3 refinement block.

    Parameters
    ----------
    detail_channels:
        Channels of D3 = P3_BHLR - P3_raw.
    neck_channels:
        Channels of the top-down P3 concat feature immediately before the
        native C3k2 P3 refinement block.
    hidden_channels:
        Compact relation-space width.
    gate_enabled:
        If True, predict a spatial gate from both neck/detail features.
    max_gain:
        Absolute bound of the learned residual gain.
    """

    def __init__(
        self,
        detail_channels: int,
        neck_channels: int,
        hidden_channels: int = 64,
        gate_enabled: bool = True,
        max_gain: float = 0.25,
    ):
        super().__init__()

        self.detail_channels = int(detail_channels)
        self.neck_channels = int(neck_channels)
        self.hidden_channels = int(hidden_channels)
        self.gate_enabled = bool(gate_enabled)
        self.max_gain = float(max_gain)

        if self.detail_channels <= 0:
            raise ValueError("detail_channels must be > 0")
        if self.neck_channels <= 0:
            raise ValueError("neck_channels must be > 0")
        if self.hidden_channels < 8:
            raise ValueError("hidden_channels must be >= 8")
        if self.max_gain <= 0:
            raise ValueError("max_gain must be > 0")

        h = self.hidden_channels

        # Compact detail representation.  It is reused both for the gate and
        # for the factorized projection back to the neck channel width.
        self.detail_reduce = ConvGNAct(
            self.detail_channels,
            h,
            k=1,
        )

        # Compact semantic representation of the current neck feature.
        self.neck_reduce = ConvGNAct(
            self.neck_channels,
            h,
            k=1,
        )

        # Factorized D3 -> neck-channel projection.
        self.detail_expand = nn.Conv2d(
            h,
            self.neck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        if self.gate_enabled:
            # The gate explicitly sees semantic context, detail context, and
            # their disagreement.  This is different from v1, whose gate saw
            # only the neck feature.
            self.gate_reduce = ConvGNAct(
                3 * h,
                h,
                k=1,
            )
            self.gate_context = ConvGNAct(
                h,
                h,
                k=3,
                groups=h,
            )
            self.gate_out = nn.Conv2d(
                h,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            # gate(init) = 0.5.  The whole branch is still exact identity
            # because gain(init) = 0.
            nn.init.zeros_(self.gate_out.weight)
            nn.init.zeros_(self.gate_out.bias)
        else:
            self.gate_reduce = None
            self.gate_context = None
            self.gate_out = None

        # Exact identity initialization:
        #   gain = max_gain * tanh(0) = 0
        self.gain_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def current_gain(self) -> torch.Tensor:
        return self.max_gain * torch.tanh(self.gain_raw)

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

    def forward(
        self,
        detail_p3: torch.Tensor,
        neck_p3_pre: torch.Tensor,
        return_debug: bool = False,
    ):
        if detail_p3.ndim != 4 or neck_p3_pre.ndim != 4:
            raise ValueError(
                "BDPNeckInternalP3 expects BCHW tensors, "
                f"got detail={tuple(detail_p3.shape)}, "
                f"neck={tuple(neck_p3_pre.shape)}"
            )

        if int(detail_p3.shape[1]) != self.detail_channels:
            raise ValueError(
                "Unexpected detail channels: "
                f"{detail_p3.shape[1]} != {self.detail_channels}"
            )
        if int(neck_p3_pre.shape[1]) != self.neck_channels:
            raise ValueError(
                "Unexpected neck channels: "
                f"{neck_p3_pre.shape[1]} != {self.neck_channels}"
            )

        detail_p3 = self._resize_like(detail_p3, neck_p3_pre)

        detail_h = self.detail_reduce(detail_p3)
        neck_h = self.neck_reduce(neck_p3_pre)
        projected_detail = self.detail_expand(detail_h)

        if self.gate_enabled:
            relation = torch.cat(
                (neck_h, detail_h, (neck_h - detail_h).abs()),
                dim=1,
            )
            gate_feature = self.gate_reduce(relation)
            gate_feature = self.gate_context(gate_feature)
            gate_logit = self.gate_out(gate_feature)
            gate = torch.sigmoid(gate_logit)
        else:
            gate_logit = None
            gate = torch.ones(
                neck_p3_pre.shape[0],
                1,
                neck_p3_pre.shape[2],
                neck_p3_pre.shape[3],
                device=neck_p3_pre.device,
                dtype=neck_p3_pre.dtype,
            )

        gain = self.current_gain().to(
            device=neck_p3_pre.device,
            dtype=neck_p3_pre.dtype,
        )

        injection = gain * gate * projected_detail
        out = neck_p3_pre + injection

        if not return_debug:
            return out

        debug: Dict[str, torch.Tensor] = {
            "detail_input": detail_p3,
            "detail_hidden": detail_h,
            "neck_hidden": neck_h,
            "detail_projected": projected_detail,
            "gate": gate,
            "injection": injection,
            "neck_before": neck_p3_pre,
            "neck_after": out,
            "gain": gain.reshape(1),
        }
        if gate_logit is not None:
            debug["gate_logit"] = gate_logit
        return out, debug


__all__ = ["BDPNeckInternalP3"]
