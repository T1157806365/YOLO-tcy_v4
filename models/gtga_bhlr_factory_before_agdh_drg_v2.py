"""AGDH-aware factory for YOLO-tcy_v4.

This dispatcher preserves the exact 2026-10-21 baseline factory in
models/gtga_bhlr_factory_base20261021.py and only constructs the AGDH variant
itself. Run install_agdh_v1.py FIRST; it automatically prepares that base.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

try:
    from models import gtga_bhlr_factory_base20261021 as _base
except ImportError as exc:
    raise ImportError(
        'AGDH base factory is missing. Run python install_agdh_v1.py '
        '(this automatically restores the exact 2026-10-21 factory).'
    ) from exc

AGDH_VARIANT = 'rlsfa_gtga_centered_bhlr_agdh_v1'


def build_model_from_config(
    cfg: Dict, pretrained: bool = True, fusion_override: Optional[str] = None,
    names=None, nc: Optional[int] = None, verbose: bool = True,
):
    mcfg = cfg.get('model', {}) or {}
    variant = str(mcfg.get('variant', mcfg.get('architecture', ''))).strip().lower()
    if variant != AGDH_VARIANT:
        return _base.build_model_from_config(
            cfg, pretrained=pretrained, fusion_override=fusion_override,
            names=names, nc=nc, verbose=verbose,
        )

    from models.rgbt_rlsfa_gtga_centered_bhlr_agdh_model import (
        RGBTAGDHV1DetectionModel,
    )
    data = cfg.get('data', {}) or {}
    lcfg = cfg.get('loss', {}) or {}
    bhlr = dict(mcfg.get('bhlr', {}) or {})
    fusion = str(fusion_override or mcfg.get('fusion', 'add')).lower().strip()
    if fusion != 'add':
        raise ValueError(f'AGDH must preserve fusion: add, got {fusion!r}')
    if nc is None:
        nc = int(mcfg.get('nc', 1))
    if names is None and int(nc) == 1:
        names = {0: 'UAV'}

    model = RGBTAGDHV1DetectionModel(
        agdh_cfg=dict(mcfg.get('agdh', {}) or {}),
        model_name=str(mcfg.get('name', 'yolo26m')),
        nc=int(nc),
        pretrained=bool(pretrained),
        fusion=fusion,
        fusion_indices=_base._resolve_fusion_indices(mcfg),
        align_mode=str(mcfg.get('align_mode', 'bilinear')),
        names=names,
        semantic_imgsz=int(data.get('rgb_semantic_imgsz', 640)),
        rlsfa_gtga_centered_cfg=dict(
            mcfg.get('rlsfa_gtga_centered', mcfg.get('alignment', {})) or {}
        ),
        rlsfa_gtga_centered_loss_cfg=dict(
            lcfg.get('alignment', lcfg.get('rlsfa_gtga_centered', {})) or {}
        ),
        bhlr_enabled=bool(bhlr.get('enabled', True)),
        bhlr_scales=bhlr.get('scales', [3]),
        bhlr_layers=bhlr.get('layers', None),
        high_to_semantic_ratio=int(bhlr.get('high_to_semantic_ratio', 3)),
        detail_channels=int(bhlr.get('detail_channels', 48)),
        stem_channels=int(bhlr.get('stem_channels', 16)),
        guide_channels=int(bhlr.get('guide_channels', 24)),
        bhlr_target_guidance_channels=int(bhlr.get('target_guidance_channels', 3)),
        bhlr_guidance_mode=str(bhlr.get('guidance_mode', 'alignment')),
        bhlr_detach_alignment_guidance=bool(
            bhlr.get('detach_alignment_guidance', True)
        ),
        bhlr_geometry_routing_cfg=dict(bhlr.get('geometry_routing', {}) or {}),
        verbose=verbose,
    )
    if hasattr(model, 'args'):
        model.args.box = float(lcfg.get('box', 7.5))
        model.args.cls = float(lcfg.get('cls', 0.5))
        model.args.dfl = float(lcfg.get('dfl', 1.5))
        model.criterion = None
    if verbose:
        print('[factory] AGDH-v1 | original ADD P345 | detector P3 only')
    return model


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device,
    fusion_override: str = 'auto', strict: bool = True, verbose: bool = False,
):
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise ValueError('Expected a training checkpoint with model_state_dict')
    cfg = ckpt.get('config', {}) or {}
    variant = str((cfg.get('model', {}) or {}).get('variant', '')).strip().lower()
    if variant != AGDH_VARIANT:
        return _base.load_model_from_checkpoint(
            path, device=device, fusion_override=fusion_override,
            strict=strict, verbose=verbose,
        )
    fusion = str(fusion_override).strip().lower()
    if fusion == 'auto':
        fusion = str(ckpt.get('fusion', (cfg.get('model', {}) or {}).get('fusion', 'add')))
    model = build_model_from_config(
        cfg, pretrained=False, fusion_override=fusion,
        names=ckpt.get('names', {0: 'UAV'}),
        nc=int(ckpt.get('nc', 1)), verbose=verbose,
    )
    model.load_state_dict(
        _base._strip_module_prefix(ckpt['model_state_dict']),
        strict=bool(strict),
    )
    model = model.to(device)
    model.eval()
    return model, ckpt, cfg


__all__ = ['build_model_from_config', 'load_model_from_checkpoint']
