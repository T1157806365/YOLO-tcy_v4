#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def patch(path: Path):
    text = path.read_text(encoding="utf-8")

    if '"rlsfa_scale"' not in text:
        old = '''    VALID_GUIDANCE_MODES = {
        "gtga",
        "none",
    }'''
        new = '''    VALID_GUIDANCE_MODES = {
        "gtga",
        "rlsfa_scale",
        "none",
    }'''
        if old not in text:
            raise RuntimeError(
                "Cannot find current BHLR VALID_GUIDANCE_MODES block. "
                "Refusing to patch an unknown bhlr_v2.py."
            )
        text = text.replace(old, new, 1)

    # Only wording changes below; computation still consumes the same
    # joint_target_support key, so GTGA and RLSFA-Scale are both supported.
    text = text.replace(
        '"BHLR-v2 guidance_mode=\'gtga\' requires "\n                "GTGA external guidance."',
        'f"BHLR-v2 guidance_mode={self.guidance_mode!r} requires "\n                "external alignment guidance."',
    )
    text = text.replace(
        '"GTGA-guided BHLR-v2 requires "\n                "external_guidance[\'joint_target_support\']."',
        '"Alignment-guided BHLR-v2 requires "\n                "external_guidance[\'joint_target_support\']."',
    )

    path.write_text(text, encoding="utf-8")
    print(f"[patched] {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python patch_bhlr.py /path/to/models/modules/bhlr_v2.py")
    patch(Path(sys.argv[1]).expanduser().resolve())
