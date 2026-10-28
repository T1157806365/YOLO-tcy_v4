"""AGDH-DRG v2 detector: unchanged centered GMLS/DMLS+BHLR+ADD foundation.

Only P3 classification logits and independent P3 box-branch inputs are altered.
P3/P4/P5 backbone fusion, Neck, non-P3 Detect, and native YOLO26 loss
remain exactly as in the original detector.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from models.rgbt_model import get_layer_input
from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)
from models.modules.agdh_drg_v2 import AGDHDRGV2


class RGBTAGDHDRGV2DetectionModel(RGBTRLSFAGTGACenteredBHLRV2DetectionModel):
    def __init__(self, *args, agdh_drg_cfg: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(agdh_drg_cfg or {})
        if not self.alignment_enabled or str(self.fusion_type).lower() != 'add':
            raise RuntimeError('AGDH-DRG v2 requires alignment enabled and original ADD fusion')
        if [int(x) for x in cfg.get('scales', [3])] != [3]:
            raise RuntimeError('AGDH-DRG v2 supports Detect P3 only')
        if list(self.model[-1].f) != [16, 19, 22]:
            raise RuntimeError('Supported YOLO26m Detect.f is [16,19,22], got '
                               + str(self.model[-1].f))
        self.agdh_drg = AGDHDRGV2(
            nc=int(self.nc),
            hidden_channels=int(cfg.get('hidden_channels', 16)),
            cls_gain=float(cfg.get('cls_gain', 0.5)),
            reg_gain=float(cfg.get('reg_gain', 0.05)),
            use_reliability=bool(cfg.get('use_reliability', True)),
            detach_alignment_prior=bool(cfg.get('detach_alignment_prior', True)),
            prior_dropout=float(cfg.get('prior_dropout', 0.0)),
        )
        self.agdh_drg.check_detect(self.model[-1])
        self.agdh_drg_cfg = cfg
        print('[AGDH-DRG v2] Detect P3 | cls_logit_gain=' + str(self.agdh_drg.cls_gain)
              + ' | reg_gain=' + str(self.agdh_drg.reg_gain)
              + ' | reliability=' + str(self.agdh_drg.use_reliability)
              + ' | prior_detach=' + str(self.agdh_drg.detach_alignment_prior), flush=True)

    def _forward_head_drg(self, backbone_outputs, support, return_debug=False):
        if len(backbone_outputs) != self.backbone_len:
            raise RuntimeError('Unexpected backbone output length')
        if support is None:
            raise RuntimeError('Missing alignment_info.multi_target_guidance')
        outputs = list(backbone_outputs)
        x = outputs[self.backbone_len - 1]
        drg_debug = None
        seen_detect = 0
        for module in list(self.model)[self.backbone_len:]:
            module_input = get_layer_input(module, x, outputs)
            if module is self.model[-1]:
                if return_debug:
                    x, drg_debug = self.agdh_drg.forward_detect(
                        module, module_input, support, return_debug=True)
                else:
                    x = self.agdh_drg.forward_detect(module, module_input, support)
                seen_detect += 1
            else:
                x = module(module_input)
            outputs.append(x)
        if seen_detect != 1:
            raise RuntimeError('Expected one native Detect module')
        return x, drg_debug

    def predict(self, rgb, tir, rgb_semantic=None, rgb_valid_mask=None,
                tir_valid_mask=None, return_features=False,
                return_bhlr_debug=False, return_alignment_debug=False,
                return_agdh_debug=False, return_drg_debug=False):
        features = self.forward_features(
            rgb, tir, rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask, tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )
        support = features.get('alignment_info', {}).get('multi_target_guidance')
        pred, debug = self._forward_head_drg(
            features['fused_backbone'], support,
            return_debug=bool(return_drg_debug or return_agdh_debug),
        )
        if (return_features or return_bhlr_debug or return_alignment_debug
                or return_agdh_debug or return_drg_debug):
            output = {'pred': pred, **features}
            if return_drg_debug or return_agdh_debug:
                output['agdh_drg_debug'] = debug
            return output
        return pred


__all__ = ['RGBTAGDHDRGV2DetectionModel']
