#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch an EXISTING YOLO-tcy-v4 checkout in-place for stretch-only preprocessing.

- No dataset copies, no model architecture modifications, no YAML changes.
- Backs up changed source files.
- Checks input source signatures before modifying anything.
- Supports repeat installs safely.
"""
from __future__ import annotations

import argparse
import ast
import difflib
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys

MARKER = 'YOLO_TCY_STRETCH_NO_PAD_V1'

DATASET_BODY = '''    # YOLO_TCY_STRETCH_NO_PAD_V1
    # Stretch independently on x and y; no letterbox and no padding.
    # Normalized YOLO xywh HBB are unchanged under whole-image scaling.
    target_h, target_w = _parse_imgsz(new_shape)
    original_h, original_w = image.shape[:2]
    if min(original_h, original_w, target_h, target_w) <= 0:
        raise ValueError(
            f"Invalid image/target shape: "
            f"{(original_h, original_w)} -> {(target_h, target_w)}"
        )

    if (original_h, original_w) != (target_h, target_w):
        image = cv2.resize(
            image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    # Do NOT rescale normalized boxes; under full-image stretch,
    # normalized x/y/w/h remain the same. Copies prevent in-place mutation.
    new_boxes = boxes.copy()

    # The legacy scalar `ratio` is retained only for API compatibility.
    # Anisotropic scales are represented by the accurate `ratio_xy`.
    # The no-padding mask path below checks `stretch` first.
    ratio_pad = {
        "ratio": 1.0,
        "pad": (0, 0),
        "ratio_xy": (
            float(target_w) / float(original_w),
            float(target_h) / float(original_h),
        ),
        "stretch": True,
    }
    return image, new_boxes, ratio_pad
'''


def _line_ending(s: str) -> str:
    return '\r\n' if '\r\n' in s else '\n'


def patch_letterbox(src: str) -> tuple[str, str]:
    t = ast.parse(src)
    candidates = [node for node in t.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'letterbox']
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one top-level def letterbox, got {len(candidates)}')
    node = candidates[0]
    if MARKER in ast.get_source_segment(src, node):
        return src, 'letterbox already patched'
    argnames = {x.arg for x in node.args.args}
    required = {'image', 'boxes', 'new_shape'}
    if not required.issubset(argnames):
        raise RuntimeError('letterbox signature differs from expected. Refusing unverified modification.')
    if len(node.body) == 0:
        raise RuntimeError('Empty letterbox body')
    lines = src.splitlines(keepends=True)
    first_body_line = node.body[0].lineno
    # Preserve original function signature/annotations, replace only function BODY.
    header = ''.join(lines[node.lineno - 1: first_body_line - 1])
    if not header.endswith(('\n', '\r\n')):
        raise RuntimeError('Could not locate letterbox signature boundary')
    newline = _line_ending(src)
    body = DATASET_BODY.replace('\n', newline)
    updated = ''.join(lines[: node.lineno - 1]) + header + body + ''.join(lines[node.end_lineno:])
    ast.parse(updated)
    return updated, 'replace letterbox body with no-padding anisotropic stretch'


def patch_model_masks(src: str) -> tuple[str, str]:
    actions = []
    target = '            if not isinstance(meta, Mapping):\n                return None\n'
    add = (
        '            if not isinstance(meta, Mapping):\n'
        '                return None\n'
        '            # YOLO_TCY_STRETCH_NO_PAD_V1: all pixels are valid after stretch.\n'
        '            if meta.get("stretch", False):\n'
        '                mask[bi, 0] = 1.0\n'
        '                continue\n'
    )
    normalized = src.replace('\r\n', '\n')
    if 'if meta.get("stretch", False):' not in normalized:
        if normalized.count(target) != 1:
            raise RuntimeError('Could not locate valid-content-mask metadata test')
        normalized = normalized.replace(target, add, 1)
        actions.append('honor stretch=True metadata (train valid masks)')

    # During validation/inference, no original shape metadata is supplied.
    # Use complete valid masks, not padded-pixel heuristics, for stretched images.
    re_rgb = '        if rgb_valid_mask is None:\n            rgb_valid_mask = self._infer_valid_mask_from_pixels(rgb_semantic)\n'
    new_rgb = (
        '        if rgb_valid_mask is None:\n'
        '            # YOLO_TCY_STRETCH_NO_PAD_V1: no padded border is present.\n'
        '            rgb_valid_mask = rgb_semantic.new_ones(\n'
        '                (rgb_semantic.shape[0], 1, *rgb_semantic.shape[-2:]), dtype=torch.float32\n'
        '            )\n'
    )
    re_tir = '        if tir_valid_mask is None:\n            tir_valid_mask = self._infer_valid_mask_from_pixels(tir)\n'
    new_tir = (
        '        if tir_valid_mask is None:\n'
        '            # YOLO_TCY_STRETCH_NO_PAD_V1: no padded border is present.\n'
        '            tir_valid_mask = tir.new_ones(\n'
        '                (tir.shape[0], 1, *tir.shape[-2:]), dtype=torch.float32\n'
        '            )\n'
    )
    if re_rgb in normalized:
        normalized = normalized.replace(re_rgb, new_rgb, 1)
        actions.append('RGB val/inference all-valid mask')
    elif '# YOLO_TCY_STRETCH_NO_PAD_V1: no padded border is present.' not in normalized:
        raise RuntimeError('Could not locate RGB forward_features mask fallback')
    if re_tir in normalized:
        normalized = normalized.replace(re_tir, new_tir, 1)
        actions.append('TIR val/inference all-valid mask')
    elif 'tir_valid_mask = tir.new_ones(' not in normalized:
        raise RuntimeError('Could not locate TIR forward_features mask fallback')

    updated = normalized.replace('\n', _line_ending(src))
    ast.parse(updated)
    return updated, ', '.join(actions) if actions else 'model masks already patched'


def main() -> None:
    ap = argparse.ArgumentParser(description='Patch YOLO-tcy-v4 source data preprocessing in-place')
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--apply', action='store_true', help='Write changes; otherwise show dry run')
    ap.add_argument('--show-diff', action='store_true', help='Print unified diff')
    args = ap.parse_args()
    repo = args.repo.expanduser().resolve()
    dataset = repo / 'datasets/rgbt_dataset.py'
    model = repo / 'models/rgbt_rlsfa_gtga_centered_bhlr_model.py'
    if not dataset.exists():
        raise SystemExit(f'Missing {dataset}. Need the actual dataset source to make a safe in-place patch.')
    if not model.exists():
        raise SystemExit(f'Missing {model}. Wrong repo directory?')
    sources = {dataset: dataset.read_text(encoding='utf-8'), model: model.read_text(encoding='utf-8')}
    # Validate BOTH modifications prior to changing any file.
    target1, act1 = patch_letterbox(sources[dataset])
    target2, act2 = patch_model_masks(sources[model])
    changed = {dataset: target1, model: target2}
    print('Source modifications:')
    for path, dest in changed.items():
        status = 'MODIFIED' if dest != sources[path] else 'UNCHANGED'
        print(f'[{status}] {path.relative_to(repo)}: {act1 if path == dataset else act2}')
        if args.show_diff and dest != sources[path]:
            print(''.join(difflib.unified_diff(sources[path].splitlines(True), dest.splitlines(True), fromfile=str(path), tofile=str(path))))
    if not args.apply:
        print('\nDRY RUN ONLY. To apply: python patch_stretch_inplace.py --repo . --apply')
        return
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for path, dest in changed.items():
        if dest == sources[path]:
            continue
        backup = path.with_name(f'{path.name}.bak_before_stretch_{timestamp}')
        shutil.copy2(path, backup)
        path.write_text(dest, encoding='utf-8')
        print('Backup:', backup)
    print('Done. Validate preprocessing before training: python verify_stretch_inplace.py')


if __name__ == '__main__':
    main()
