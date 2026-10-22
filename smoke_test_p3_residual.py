#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast identity/gradient/shape test; optional full model CUDA forward test."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.modules.p3_product_residual import P3ProductResidual


def test_module():
    torch.manual_seed(0)
    shape = (2, 64, 40, 40)
    rgb = torch.randn(shape, requires_grad=True)
    tir = torch.randn(shape, requires_grad=True)
    add = rgb + tir
    for mode in ('conv', 'product'):
        module = P3ProductResidual(64, 64, 16, mode, max_gain=0.25)
        module.eval()
        y = module(add, rgb, tir)
        assert torch.equal(y, add), f'Identity init broken in mode={mode}'
        loss = y.square().mean()
        loss.backward(retain_graph=True)
        assert module.gamma_raw.grad is not None
        assert module.gamma_raw.grad.isfinite().all()
        with torch.no_grad():
            module.gamma_raw.fill_(0.5)
        y2 = module(add, rgb, tir)
        assert y2.shape == add.shape
        assert torch.isfinite(y2).all()
        assert not torch.equal(y2, add)
        print(f'[OK] mode={mode} shape={tuple(y2.shape)} init_identity=True grad=True')


def test_full(cfg_file: Path, device_str: str):
    import yaml
    from models.gtga_bhlr_factory import build_model_from_config
    cfg = yaml.safe_load(cfg_file.read_text(encoding='utf-8'))
    if not torch.cuda.is_available() and device_str.startswith('cuda'):
        raise RuntimeError('CUDA unavailable; use --device cpu or skip --full')
    device = torch.device(device_str)
    model = build_model_from_config(cfg, pretrained=False, verbose=False).to(device)
    model.eval()
    with torch.inference_mode():
        rgb_high = torch.rand((1, 3, 1920, 1920), device=device)
        rgb_sem = torch.rand((1, 3, 640, 640), device=device)
        tir = torch.rand((1, 3, 640, 640), device=device)
        out = model(rgb_high, tir, rgb_semantic=rgb_sem)
    print('[OK] model forward:', type(out), 'P3 layer:', model.p3_residual_index)
    print('P3 module gain:', model.p3_residual.scalar_state())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Also create the full network and perform a forward')
    parser.add_argument('--yaml', default='configs/experiments/p3_product_residual/S2_product_residual.yaml')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    test_module()
    if args.full:
        test_full(Path(args.yaml), args.device)


if __name__ == '__main__':
    main()
