#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate complete baseline/conv/product YAML files from an existing baseline.

This is intentional: your baseline YAML is the authoritative source for dataset
paths, GMLS/DMLS parameters, schedules and loss coefficients.  Generated files
contain the FULL resolved YAML and have isolated weight directories.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

BASE_VARIANT = 'rlsfa_gtga_centered_bhlr_v2'
NEW_VARIANT = 'rlsfa_gtga_centered_bhlr_p3_residual_v1'


def generate(src: Path, dst: Path, device: str | None, screen: bool = False):
    original = yaml.safe_load(src.read_text(encoding='utf-8'))
    if not isinstance(original, dict):
        raise TypeError(f'Expected YAML dict in {src}')
    model = original.get('model', {})
    if model.get('variant') != BASE_VARIANT:
        raise ValueError(
            'Expected baseline variant rlsfa_gtga_centered_bhlr_v2. '
            f'Found {model.get("variant")!r}'
        )
    if model.get('fusion') != 'add':
        raise ValueError('Baseline fusion must be add')
    if model.get('name') != 'yolo26m':
        print(f'Notice: source model={model.get("name")!r}, not yolo26m')
    dst.mkdir(parents=True, exist_ok=True)
    results = {}
    for stem, mode in [('S0_baseline', None), ('S1_conv_residual', 'conv'),
                       ('S2_product_residual', 'product')]:
        cfg = copy.deepcopy(original)
        # Old source YAML has traincc scripts for a different name/weights.
        # Drop those from GENERATED YAML to prevent copy-paste path mistakes.
        cfg.pop('traincc', None)
        m = cfg['model']
        # In old YAML 'ture' is a string whose bool(...) was True.  Make
        # semantics explicit while preserving the original actual behavior.
        pretrained = m.get('pretrained', True)
        if isinstance(pretrained, str):
            token = pretrained.strip().lower()
            if token in {'ture', 'true', 'yes', '1'}:
                pretrained = True
            elif token in {'false', 'no', '0'}:
                pretrained = False
            else:
                raise ValueError(f'Ambiguous pretrained value: {pretrained!r}')
        m['pretrained'] = bool(pretrained)
        m.pop('p3_residual', None)
        if mode is not None:
            m['variant'] = NEW_VARIANT
            m['p3_residual'] = {
                'enabled': True,
                'mode': mode,
                'hidden_channels': 32,
                'max_gain': 0.25,
            }
        t = cfg['train']
        if device is not None:
            t['device'] = device
        if screen:
            t['epochs'] = 60
            t['patience'] = 20
            t.setdefault('stage2', {})['enabled'] = False
        cfg['output']['project'] = 'runs/p3_product_residual_screen' if screen else 'runs/p3_product_residual_full'
        cfg['output']['name'] = f'yolo26m_UAVCB_best20261021_{stem}'
        path = dst / f'{stem}.yaml'
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
        results[stem] = path
        print(f'{stem:23s} {path}')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640.yaml')
    parser.add_argument('--output', default='configs/experiments/p3_product_residual')
    parser.add_argument('--device', default=None, help='GPU id(s), e.g. 0,4; default keeps baseline')
    parser.add_argument('--screen', action='store_true', help='60 epochs, no stage2 (screening only)')
    args = parser.parse_args()
    generate(Path(args.source), Path(args.output), args.device, args.screen)


if __name__ == '__main__':
    main()
