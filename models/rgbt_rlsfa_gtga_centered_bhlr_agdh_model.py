"""AGDH-v1 detector built on the *unchanged* centered GMLS+BHLR+ADD baseline.

The only change in the inference path is to modify the Neck P3 feature
immediately before the last Detect layer receives [P3, P4, P5]. Importantly,
Neck P4/P5 generation is computed from the unmodified P3, so ablation is clean.
The parent loss() and forward(dict) remain completely untouched.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from models.rgbt_model import get_layer_input
from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)
from models.modules.agdh_v1 import AGDHP3


class RGBTAGDHV1DetectionModel(RGBTRLSFAGTGACenteredBHLRV2DetectionModel):
    def __init__(self, *args, agdh_cfg: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_agdh(agdh_cfg or {})

    def configure_agdh(self, cfg):
        cfg = dict(cfg or {})
        allowed_scales = list(cfg.get('scales', [3]))
        if allowed_scales != [3]:
            raise ValueError('AGDH v1 supports Detect P3 only: scales: [3]')
        if str(cfg.get('injection_position', 'detect_input')) != 'detect_input':
            raise ValueError('AGDH v1 requires injection_position: detect_input')
        if not self.alignment_enabled:
            raise ValueError('AGDH requires enabled RLSFA/GTGA alignment')
        if str(self.fusion_type).strip().lower() != 'add':
            raise ValueError('AGDH baseline must retain fusion: add')
        self.agdh = AGDHP3(
            hidden_channels=int(cfg.get('hidden_channels', 16)),
            max_gain=float(cfg.get('max_gain', 0.2)),
            detach_alignment_prior=bool(cfg.get('detach_alignment_prior', True)),
            union_guard=bool(cfg.get('union_guard', True)),
            prior_dropout=float(cfg.get('prior_dropout', 0.0)),
        )
        self.agdh_cfg = cfg
        print('[AGDH-v1] Detect P3-only | original ADD P345 | '
              f'hidden={self.agdh.hidden_channels} | max_gain={self.agdh.max_gain} '
              f'| prior_detach={self.agdh.detach_alignment_prior}', flush=True)

    def _forward_head_agdh(self, backbone_outputs, support_packet, debug=False):
        if len(backbone_outputs) != self.backbone_len:
            raise ValueError('Wrong backbone_outputs length')
        if support_packet is None:
            raise RuntimeError('AGDH missing alignment_info.multi_target_guidance')
        if list(self.model[-1].f) != [16, 19, 22]:
            raise RuntimeError(
                'AGDH-v1 was validated for YOLO26m Detect from [16,19,22]; '
                f'actual Detect from={self.model[-1].f}. Refuse silent wrong injection.'
            )

        outputs = list(backbone_outputs)
        x = outputs[self.backbone_len - 1]
        agdh_debug = None
        n_detect = 0
        for module in list(self.model)[self.backbone_len:]:
            module_input = get_layer_input(module, x, outputs)
            if module is self.model[-1]:
                if not isinstance(module_input, (list, tuple)) or len(module_input) != 3:
                    raise RuntimeError('Expected Detect input [P3,P4,P5]')
                if debug:
                    p3_modified, agdh_debug = self.agdh(
                        module_input[0], support_packet, return_debug=True
                    )
                else:
                    p3_modified = self.agdh(module_input[0], support_packet)
                module_input = [p3_modified, module_input[1], module_input[2]]
                n_detect += 1
            x = module(module_input)
            outputs.append(x)
        if n_detect != 1:
            raise RuntimeError(f'AGDH expected one final Detect, found {n_detect}')
        return x, agdh_debug

    def predict(
        self, rgb, tir, rgb_semantic=None,
        rgb_valid_mask=None, tir_valid_mask=None,
        return_features=False, return_bhlr_debug=False,
        return_alignment_debug=False, return_agdh_debug=False,
    ):
        features = self.forward_features(
            rgb, tir,
            rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask,
            tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )
        alignment_info = features.get('alignment_info', {})
        support_packet = alignment_info.get('multi_target_guidance')
        pred, agdh_debug = self._forward_head_agdh(
            features['fused_backbone'], support_packet,
            debug=bool(return_agdh_debug),
        )
        if return_features or return_bhlr_debug or return_alignment_debug or return_agdh_debug:
            result = {'pred': pred, **features}
            if return_agdh_debug:
                result['agdh_debug'] = agdh_debug
            return result
        return pred


__all__ = ['RGBTAGDHV1DetectionModel']
