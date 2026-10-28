"""ADGH-DRG v2 dispatcher; all earlier variants delegate to saved pre-v2 factory."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import torch

from models import gtga_bhlr_factory_before_agdh_drg_v2 as _previous

AGDH_DRG_VARIANT = 'rlsfa_gtga_centered_bhlr_agdh_drg_v2'


def _get_variant(cfg):
    x = cfg.get('model', {}) or {}
    return str(x.get('variant', x.get('architecture', ''))).strip().lower()


def build_model_from_config(
    cfg: Dict, pretrained: bool = True, fusion_override: Optional[str] = None,
    names=None, nc: Optional[int] = None, verbose: bool = True,
):
    if _get_variant(cfg) != AGDH_DRG_VARIANT:
        return _previous.build_model_from_config(
            cfg, pretrained=pretrained, fusion_override=fusion_override,
            names=names, nc=nc, verbose=verbose)

    from models.rgbt_rlsfa_gtga_centered_bhlr_agdh_drg_v2_model import (
        RGBTAGDHDRGV2DetectionModel,
    )
    model_cfg = cfg.get('model', {}) or {}
    data_cfg = cfg.get('data', {}) or {}
    loss_cfg = cfg.get('loss', {}) or {}
    bhlr = dict(model_cfg.get('bhlr', {}) or {})
    fusion = str(fusion_override or model_cfg.get('fusion', 'add')).strip().lower()
    if fusion != 'add':
        raise ValueError('AGDH-DRG v2 requires fusion=add, not ' + fusion)
    if nc is None:
        nc = int(model_cfg.get('nc', 1))
    if names is None and int(nc) == 1:
        names = {0: 'UAV'}
    if str(model_cfg.get('name', 'yolo26m')).lower() != 'yolo26m':
        raise RuntimeError('AGDH-DRG v2 currently supports yolo26m only')
    resolve_indices = getattr(_previous, '_resolve_fusion_indices', None)
    if resolve_indices is None:
        from models import gtga_bhlr_factory_base20261021 as _base
        resolve_indices = _base._resolve_fusion_indices

    model = RGBTAGDHDRGV2DetectionModel(
        agdh_drg_cfg=dict(model_cfg.get('agdh_drg', {}) or {}),
        model_name=str(model_cfg.get('name', 'yolo26m')),
        nc=int(nc), pretrained=bool(pretrained), fusion=fusion,
        fusion_indices=resolve_indices(model_cfg),
        align_mode=str(model_cfg.get('align_mode', 'bilinear')),
        names=names,
        semantic_imgsz=int(data_cfg.get('rgb_semantic_imgsz', 640)),
        rlsfa_gtga_centered_cfg=dict(
            model_cfg.get('rlsfa_gtga_centered', model_cfg.get('alignment', {})) or {}),
        rlsfa_gtga_centered_loss_cfg=dict(
            loss_cfg.get('alignment', loss_cfg.get('rlsfa_gtga_centered', {})) or {}),
        bhlr_enabled=bool(bhlr.get('enabled', True)),
        bhlr_scales=bhlr.get('scales', [3]),
        bhlr_layers=bhlr.get('layers'),
        high_to_semantic_ratio=int(bhlr.get('high_to_semantic_ratio', 3)),
        detail_channels=int(bhlr.get('detail_channels', 48)),
        stem_channels=int(bhlr.get('stem_channels', 16)),
        guide_channels=int(bhlr.get('guide_channels', 24)),
        bhlr_guidance_mode=str(bhlr.get('guidance_mode', 'alignment')),
        bhlr_detach_alignment_guidance=bool(bhlr.get('detach_alignment_guidance', True)),
        verbose=verbose,
    )
    if hasattr(model, 'args'):
        model.args.box = float(loss_cfg.get('box', 7.5))
        model.args.cls = float(loss_cfg.get('cls', 0.5))
        model.args.dfl = float(loss_cfg.get('dfl', 1.5))
        model.criterion = None
    if verbose:
        print('[factory] AGDH-DRG v2 | original ADD P345 | Detect P3 cls+reg decoupled')
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
        raise ValueError('Expected checkpoint dict with model_state_dict')
    cfg = ckpt.get('config', {}) or {}
    if _get_variant(cfg) != AGDH_DRG_VARIANT:
        return _previous.load_model_from_checkpoint(
            path, device=device, fusion_override=fusion_override,
            strict=strict, verbose=verbose,
        )
    fusion = str(fusion_override).strip().lower()
    if fusion == 'auto':
        fusion = str(ckpt.get('fusion', cfg.get('model', {}).get('fusion', 'add')))
    model = build_model_from_config(
        cfg, pretrained=False, fusion_override=fusion,
        names=ckpt.get('names', {0: 'UAV'}),
        nc=int(ckpt.get('nc', 1)), verbose=verbose,
    )
    weights = ckpt['model_state_dict']
    if weights and all(str(k).startswith('module.') for k in weights):
        weights = {k[len('module.'):]: v for k, v in weights.items()}
    model.load_state_dict(weights, strict=bool(strict))
    model = model.to(device)
    model.eval()
    return model, ckpt, cfg


__all__ = ['build_model_from_config', 'load_model_from_checkpoint']
