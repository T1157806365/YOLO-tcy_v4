#!/usr/bin/env python3
"""Inspect no-padding stretch and normalized bounding boxes without touching user data."""
from __future__ import annotations
import numpy as np
import torch

from datasets.rgbt_dataset import letterbox
from models.rgbt_rlsfa_gtga_centered_bhlr_model import RGBTRLSFAGTGACenteredBHLRV2DetectionModel


def inspect(tag, shape, dst):
    image = np.full((shape[0], shape[1], 3), 50, dtype=np.uint8)
    image[0, 0] = [0, 255, 255]
    bboxes = np.array([[0.35, 0.45, 0.25, 0.13]], dtype=np.float32)
    orig = bboxes.copy()
    img, out_boxes, ratio_pad = letterbox(image, bboxes, dst, scaleup=False)
    shape_dst = (dst, dst) if isinstance(dst, int) else tuple(dst)
    assert img.shape[:2] == shape_dst, (tag, img.shape)
    assert np.allclose(out_boxes, orig, atol=1e-6), (tag, orig, out_boxes)
    assert ratio_pad.get('stretch') is True, ratio_pad
    assert ratio_pad['pad'] == (0, 0), ratio_pad
    # Input is uniform except marker, so no fabricated border can occur.
    assert np.all(img[2:-2, 2:-2] == 50), (tag, 'unexpected padding')
    print(f'PASS {tag}: {shape} -> {shape_dst}, bbox unchanged, no padding')
    return ratio_pad


inspect('RGB-high', (1080, 1920), 1920)
rgb_meta = inspect('RGB-semantic', (1080, 1920), 640)
tir_meta = inspect('TIR', (512, 640), 640)

build_mask = RGBTRLSFAGTGACenteredBHLRV2DetectionModel._build_valid_content_mask
for tag, ori_shape, meta in [('RGB', (1080,1920),rgb_meta),('TIR',(512,640),tir_meta)]:
    x = torch.full((1, 3, 640, 640), 50 / 255.0, dtype=torch.float32)
    mask = build_mask(x, [ori_shape], [meta])
    assert mask is not None and torch.all(mask == 1), (tag, mask)
    print(f'PASS {tag}: full valid-content mask')
print('ALL CHECKS PASSED')
