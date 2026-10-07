from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .base import BaseAlignment


class IdentityAlignment(BaseAlignment):
    def forward(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        return_debug: bool = False,
    ):
        del rgb_features, scale_to_index
        aligned_tir = list(tir_features)
        info = {
            'enabled': False,
            'type': 'identity',
            'scales': list(self.scales),
            'has_alignment': False,
        }
        if return_debug:
            info['message'] = 'IdentityAlignment: TIR unchanged.'
        return aligned_tir, info
