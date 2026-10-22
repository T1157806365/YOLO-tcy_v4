#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe, idempotent installer for YOLO-tcy_v4 backup-20261021_best.

Only modifies models/gtga_bhlr_factory.py.  Copies two new complete Python
files.  Existing model/trainer/evaluator are not replaced.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import shutil
from pathlib import Path

VARIANT = "rlsfa_gtga_centered_bhlr_p3_residual_v1"
BASE_VARIANTS = '''    elif variant in {
        "rlsfa_gtga_centered_bhlr_v2",
        "rlsfa_gtga_centered",
    }:'''
NEW_VARIANTS = '''    elif variant in {
        "rlsfa_gtga_centered_bhlr_v2",
        "rlsfa_gtga_centered",
        "rlsfa_gtga_centered_bhlr_p3_residual_v1",
    }:'''
BASE_IMPORT = '''        from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
            RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
        )'''
NEW_IMPORT = '''        if variant == "rlsfa_gtga_centered_bhlr_p3_residual_v1":
            from models.rgbt_rlsfa_gtga_centered_bhlr_p3_residual_model import (
                RGBTP3ProductResidualDetectionModel as RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
            )
        else:
            from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
                RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
            )'''


def patch_factory(source: str) -> str:
    # Protect the original baseline branch by only modifying its factory path.
    start_key = BASE_VARIANTS
    end_key = '    # 2. Legacy RLSFA-GTGA + BHLR-v2'
    if source.count(start_key) == 1:
        source = source.replace(start_key, NEW_VARIANTS, 1)
    elif source.count(NEW_VARIANTS) != 1:
        raise RuntimeError("Can't find the specific 20261021 baseline factory branch")

    start = source.find(NEW_VARIANTS)
    end = source.find(end_key, start)
    if start < 0 or end < 0:
        raise RuntimeError("Can't locate baseline branch boundaries")
    branch = source[start:end]

    if BASE_IMPORT in branch:
        branch = branch.replace(BASE_IMPORT, NEW_IMPORT, 1)
    elif NEW_IMPORT not in branch:
        raise RuntimeError("Original model import changed; refusing to patch")

    cfg_insert = '''        # Extra constructor argument is supplied ONLY for the new variant.
        # The original baseline model constructor is unchanged.
        p3_extra_cfg = (
            {"p3_residual_cfg": dict(model_cfg.get("p3_residual", {}) or {})}
            if variant == "rlsfa_gtga_centered_bhlr_p3_residual_v1" else {}
        )

'''
    cfg_anchor = '''        model = RGBTRLSFAGTGACenteredBHLRV2DetectionModel('''
    if '        p3_extra_cfg = (' not in branch:
        if branch.count(cfg_anchor) != 1:
            raise RuntimeError("Unexpected model constructor anchor")
        branch = branch.replace(cfg_anchor, cfg_insert + cfg_anchor, 1)

    original_tail = '''            verbose=verbose,
        )'''
    new_tail = '''            **p3_extra_cfg,
            verbose=verbose,
        )'''
    if new_tail not in branch:
        if branch.count(original_tail) != 1:
            raise RuntimeError("Expected one baseline constructor ending")
        branch = branch.replace(original_tail, new_tail, 1)

    patched = source[:start] + branch + source[end:]
    ast.parse(patched)
    return patched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.', help='YOLO-tcy_v4 working tree')
    parser.add_argument('--apply', action='store_true', help='Actually install')
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    home = Path(__file__).resolve().parent

    factory = repo / 'models' / 'gtga_bhlr_factory.py'
    if not factory.is_file():
        raise FileNotFoundError(f'Invalid YOLO-tcy_v4 repository: {factory}')
    original = factory.read_text(encoding='utf-8')
    edited = patch_factory(original)

    items = [
        ('models/modules/p3_product_residual.py', 'new module'),
        ('models/rgbt_rlsfa_gtga_centered_bhlr_p3_residual_model.py', 'new variant'),
    ]
    for relative, title in items:
        f = home / relative
        if not f.is_file():
            raise FileNotFoundError(f'Missing installer payload: {f}')
        ast.parse(f.read_text(encoding='utf-8'))
        dest = repo / relative
        if dest.exists() and dest.read_bytes() != f.read_bytes():
            raise RuntimeError(
                f'Existing file has different content: {dest}. '
                'Refusing to modify the factory until this conflict is resolved.'
            )

    print(f'Repository: {repo}')
    print(f'Factory patched: {edited != original}')
    for relative, title in items:
        dest = repo / relative
        print(f'{title}: {relative} ({"exists" if dest.exists() else "new"})')

    if not args.apply:
        print('Dry run only. Run with --apply to install.')
        if edited != original:
            diff = difflib.unified_diff(original.splitlines(), edited.splitlines(),
                                        fromfile='before', tofile='after', lineterm='')
            print('\n'.join(list(diff)[:110]))
        return

    if edited != original:
        backup = factory.with_name('gtga_bhlr_factory.before_p3_residual.py')
        if not backup.exists():
            shutil.copy2(factory, backup)
        factory.write_text(edited, encoding='utf-8')
    for relative, _ in items:
        src, dest = home / relative, repo / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != src.read_bytes():
            raise RuntimeError(
                f'Existing file has different content: {dest}. '
                'Do not overwrite manually edited source.'
            )
        if not dest.exists():
            shutil.copy2(src, dest)

    for path in (factory, *(repo / relative for relative, _ in items)):
        ast.parse(path.read_text(encoding='utf-8'))
    print('Installed successfully; original main model and trainer not modified.')
    print('Next: python generate_p3_configs.py --source <baseline.yaml>')


if __name__ == '__main__':
    main()
