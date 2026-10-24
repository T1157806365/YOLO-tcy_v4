"""Drop-in factory for LC-ADD, preserving the original 2026-10-21 factory.

BEFORE manual replacement, the repository MUST contain:
    models/gtga_bhlr_factory_base20261021.py
copied from the original `backup-20261021_best` branch.
This file does not install or patch anything.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional
import torch

from models import gtga_bhlr_factory_base20261021 as _base
from models.modules.light_confidence_add import LightConfidenceAdd

LC_VARIANT = "rlsfa_gtga_centered_bhlr_light_confadd_v1"
BASE_VARIANT = "rlsfa_gtga_centered_bhlr_v2"


def _inject_confidence_add(model, cfg):
    m = dict(cfg.get('model', {}) or {})
    opt = dict(m.get('confidence_add', {}) or {})
    if not opt.get('enabled', True):
        raise ValueError('confidence_add disabled; use baseline model variant instead')
    scales = sorted(set(int(x) for x in opt.get('scales', [3])))
    if not scales or any(x not in (3,4,5) for x in scales):
        raise ValueError('confidence_add.scales must be nonempty subset of [3,4,5]')
    if getattr(model, 'fusion_type', '').lower() != 'add':
        raise ValueError('LC-ADD expects model.fusion: add')

    cinfo = model._infer_feature_channels()
    for scale in scales:
        if scale not in model.scale_to_index:
            raise ValueError(f'Missing P{scale} from model.scale_to_index')
        idx = int(model.scale_to_index[scale])
        if idx not in model.fusion_indices:
            raise ValueError(f'P{scale} layer {idx} not in fusion_indices')
        ch = cinfo[idx]
        model.fusions[str(idx)] = LightConfidenceAdd(
            rgb_channels=int(ch['rgb']),
            tir_channels=int(ch['tir']),
            max_delta=float(opt.get('max_delta', 0.25)),
            kernel_size=int(opt.get('kernel_size', 3)),
            channel_gate=bool(opt.get('channel_gate', True)),
        )
    model.confadd_scales = tuple(scales)
    model.confadd_cfg = opt
    print('[LC-ADD REPLACE] scales={} layers={} max_delta={} kernel={} channel_gate={} (init=exact ADD)'.format(
        scales, [int(model.scale_to_index[x]) for x in scales],
        opt.get('max_delta', 0.25), opt.get('kernel_size', 3),
        opt.get('channel_gate', True)), flush=True)
    return model


def _warm_start(model, path):
    from collections.abc import Mapping
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f'Initial baseline checkpoint does not exist: {p}')
    checkpoint = torch.load(p,map_location='cpu',weights_only=False)
    source_cfg = checkpoint.get('config', {}) or {}
    old_variant = (source_cfg.get('model',{}) or {}).get('variant',None)
    if old_variant != BASE_VARIANT:
        raise ValueError(f'Expected a {BASE_VARIANT} checkpoint, got {old_variant}')
    state = _base._strip_module_prefix(checkpoint['model_state_dict'])
    if not isinstance(state, Mapping):
        raise TypeError('Invalid baseline state dict')
    incompatible = model.load_state_dict(state,strict=False)
    new_prefixes = tuple(f'fusions.{model.scale_to_index[x]}.' for x in model.confadd_scales)
    bad_missing = [x for x in incompatible.missing_keys if not x.startswith(new_prefixes)]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f'Baseline not compatible: missing={bad_missing[:20]}, unexpected={incompatible.unexpected_keys[:20]}')
    print(f'[LC-ADD warm-start] baseline weights: {p}',flush=True)
    print(f'[LC-ADD warm-start] new gates: {list(incompatible.missing_keys)}',flush=True)


def build_model_from_config(
    cfg: Dict, pretrained: bool=True, fusion_override: Optional[str]=None,
    names=None, nc: Optional[int]=None, verbose: bool=True,
):
    m = cfg.get('model',{}) or {}
    variant = str(m.get('variant','')).strip().lower()
    if variant != LC_VARIANT:
        return _base.build_model_from_config(cfg, pretrained=pretrained,
            fusion_override=fusion_override, names=names, nc=nc, verbose=verbose)
    fusion = str(fusion_override or m.get('fusion','add')).lower()
    if fusion != 'add':
        raise ValueError('Use fusion: add for LC-ADD')
    cfg_old = deepcopy(cfg)
    cfg_old['model']['variant'] = BASE_VARIANT
    initial_ckpt = m.get('init_from_baseline', None)
    # Don't initialize the backbone twice when an entire baseline checkpoint is used.
    model = _base.build_model_from_config(
        cfg_old, pretrained=(bool(pretrained) and not bool(initial_ckpt)),
        fusion_override='add', names=names, nc=nc, verbose=verbose)
    model = _inject_confidence_add(model, cfg)
    # This is TRAINING-ONLY; evaluation loader calls with pretrained=False.
    if bool(pretrained) and initial_ckpt:
        _warm_start(model, initial_ckpt)
    return model


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device,
                               fusion_override: str='auto', strict: bool=True,
                               verbose: bool=False):
    p = Path(checkpoint_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    checkpoint = torch.load(p,map_location='cpu',weights_only=False)
    cfg = checkpoint.get('config',{}) or {}
    m = cfg.get('model',{}) or {}
    if str(m.get('variant','')).strip().lower() != LC_VARIANT:
        return _base.load_model_from_checkpoint(p,device=device,
            fusion_override=fusion_override,strict=strict,verbose=verbose)
    saved_fusion = str(checkpoint.get('fusion',m.get('fusion','add'))).lower()
    fusion = saved_fusion if str(fusion_override).lower()=='auto' else str(fusion_override).lower()
    model = build_model_from_config(cfg,pretrained=False,fusion_override=fusion,
        names=checkpoint.get('names',{0:'UAV'}),nc=int(checkpoint.get('nc',1)),
        verbose=verbose)
    state = _base._strip_module_prefix(checkpoint['model_state_dict'])
    model.load_state_dict(state,strict=bool(strict))
    model = model.to(device)
    model.eval()
    return model,checkpoint,cfg

__all__=['build_model_from_config','load_model_from_checkpoint']
