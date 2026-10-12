#!/usr/bin/env python3
"""Register rlsfa_gtga_centered_bhlr_bdp_neck_v2 in gtga_bhlr_factory.py.

The script patches the user's CURRENT factory file in place instead of
replacing it, so existing variants (including DUCF or BDP-v1) are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


VARIANT = "rlsfa_gtga_centered_bhlr_bdp_neck_v2"

BRANCH_BODY = r'''
    # --------------------------------------------------------
    # BDP-Neck v2
    # Fixed centered alignment + BHLR + ADD + internal P3
    # detail fusion BEFORE the native P3 refinement block.
    # --------------------------------------------------------
    __BRANCH_KW__ variant in {
        "rlsfa_gtga_centered_bhlr_bdp_neck_v2",
        "rlsfa_gtga_centered_bdp_neck_v2",
    }:
        from models.rgbt_rlsfa_gtga_centered_bhlr_bdp_neck_v2_model import (
            RGBTRLSFAGTGACenteredBHLRBDPNeckV2DetectionModel,
        )

        align_cfg = dict(
            model_cfg.get(
                "rlsfa_gtga_centered",
                model_cfg.get("alignment", {}),
            )
            or {}
        )

        align_loss_cfg = dict(
            loss_cfg.get(
                "alignment",
                loss_cfg.get("rlsfa_gtga_centered", {}),
            )
            or {}
        )

        bhlr_cfg = dict(model_cfg.get("bhlr", {}) or {})
        bdp_neck_v2_cfg = dict(model_cfg.get("bdp_neck_v2", {}) or {})

        model = RGBTRLSFAGTGACenteredBHLRBDPNeckV2DetectionModel(
            model_name=str(model_cfg.get("name", "yolo26s")),
            nc=int(nc),
            pretrained=bool(pretrained),
            fusion=fusion,
            fusion_indices=fusion_indices,
            align_mode=str(model_cfg.get("align_mode", "bilinear")),
            names=names,
            semantic_imgsz=int(data_cfg.get("rgb_semantic_imgsz", 640)),
            rlsfa_gtga_centered_cfg=align_cfg,
            rlsfa_gtga_centered_loss_cfg=align_loss_cfg,
            bhlr_enabled=bool(bhlr_cfg.get("enabled", True)),
            bhlr_scales=bhlr_cfg.get("scales", [3]),
            bhlr_layers=bhlr_cfg.get("layers", None),
            high_to_semantic_ratio=int(bhlr_cfg.get("high_to_semantic_ratio", 3)),
            detail_channels=int(bhlr_cfg.get("detail_channels", 48)),
            stem_channels=int(bhlr_cfg.get("stem_channels", 16)),
            guide_channels=int(bhlr_cfg.get("guide_channels", 24)),
            bhlr_guidance_mode=str(bhlr_cfg.get("guidance_mode", "alignment")),
            bhlr_detach_alignment_guidance=bool(
                bhlr_cfg.get("detach_alignment_guidance", True)
            ),
            bdp_neck_v2_cfg=bdp_neck_v2_cfg,
            verbose=verbose,
        )

'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default=".",
        help="YOLO-tcy_v4 project root (default: current directory)",
    )
    args = parser.parse_args()

    root = Path(args.project).expanduser().resolve()
    target = root / "models" / "gtga_bhlr_factory.py"
    if not target.exists():
        raise FileNotFoundError(target)

    text = target.read_text(encoding="utf-8")
    if VARIANT in text:
        print(f"Factory already supports {VARIANT}; no change needed.")
        return

    # Insert immediately before the historical centered-BHLR baseline branch.
    # The baseline may be the first branch (`if`) in older factory versions,
    # or an `elif` after DUCF/other variants in newer versions.
    needle_elif = '''    elif variant in {\n        "rlsfa_gtga_centered_bhlr_v2",'''
    needle_if = '''    if variant in {\n        "rlsfa_gtga_centered_bhlr_v2",'''

    if needle_elif in text:
        pos = text.find(needle_elif)
        branch = BRANCH_BODY.replace("__BRANCH_KW__", "elif", 1)
        patched = text[:pos] + branch + text[pos:]
    elif needle_if in text:
        pos = text.find(needle_if)
        # Our new branch becomes the first `if`; convert the old baseline to `elif`.
        branch = BRANCH_BODY.replace("__BRANCH_KW__", "if", 1)
        baseline = text[pos:].replace(needle_if, needle_elif, 1)
        patched = text[:pos] + branch + baseline
    else:
        raise RuntimeError(
            "Could not find the centered-BHLR baseline branch in "
            f"{target}. Please send the current factory file if it has changed."
        )

    backup = target.with_name("gtga_bhlr_factory_before_bdp_neck_v2.py")
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")

    print(f"Backup : {backup}")
    print(f"Patched: {target}")
    print(f"Added  : {VARIANT}")


if __name__ == "__main__":
    main()
