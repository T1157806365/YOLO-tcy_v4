from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn


class BaseAlignment(nn.Module):
    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__()
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get('enabled', False))
        self.scales = tuple(
            int(s)
            for s in self.cfg.get('scales', [3])
        )

    def forward(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        return_debug: bool = False,
    ):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f'enabled={self.enabled}, scales={list(self.scales)}'
