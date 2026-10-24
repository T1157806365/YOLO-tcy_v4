#!/usr/bin/env python3
"""Evaluate an existing LC-ADD checkpoint with a temporary inference-time gate scale.

No source files/checkpoints are changed. Pass regular arguments intended for
val_rlsfa_gtga_bhlr_v2.py after --gate-scale.  Example:
  python eval_lcadd_gain_scale.py --gate-scale 0 \
    --weights runs/.../best.pt --split val --device 1 --batch 4

0: replace all LC-ADD outputs with exact RGB + aligned TIR.
0.5: interpolate halfway between original ADD and trained LC-ADD.
1: original checkpoint forward, intended sanity check.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Temporary LC-ADD gate amplitude ablation (no checkpoint modifications).',
        allow_abbrev=False,
    )
    parser.add_argument('--gate-scale', type=float, required=True,
                        help='0=ADD bypass, 0.5=half-gate, 1=trained LC-ADD')
    args, remaining = parser.parse_known_args()
    if not (0 <= args.gate_scale <= 1):
        parser.error('--gate-scale must be between 0 and 1')

    from models.modules.light_confidence_add import LightConfidenceAdd

    old_forward = LightConfidenceAdd.forward
    scale = args.gate_scale
    if scale == 0:
        def ablated_forward(self, rgb, aligned_tir):
            if rgb.shape != aligned_tir.shape:
                raise ValueError('RGB/TIR feature shape mismatch')
            return rgb + aligned_tir
    elif scale == 1:
        ablated_forward = old_forward
    else:
        def ablated_forward(self, rgb, aligned_tir):
            base = rgb + aligned_tir
            trained = old_forward(self, rgb, aligned_tir)
            return base + scale * (trained - base)

    LightConfidenceAdd.forward = ablated_forward
    print(f'[LC-ADD GATE TEST] scale={scale:g} '
          '(0=ADD, 1=trained LC-ADD); saved weights untouched', flush=True)

    script = Path(__file__).resolve().parent / 'val_rlsfa_gtga_bhlr_v2.py'
    if not script.is_file():
        raise FileNotFoundError(
            f'Move this script to the same directory as val_rlsfa_gtga_bhlr_v2.py: {script}'
        )
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
