# YOLO-tcy_v4 original-code preprocessing: Stretch (no padding)

**Use on the local working tree of `backup-20261021_best`.** Does not create a dataset copy, change the network, or change YAML. All original RGB/TIR paired files remain untouched.

## Files changed

1. `datasets/rgbt_dataset.py`: replace the original `letterbox()` body with a direct `cv2.resize((target_w,target_h))`. All **normalized XYWH** boxes are unchanged under whole-image anisotropic stretching; no `copyMakeBorder` call. The dataset uses this function through `datasets/rsdt_dataset.py` and `datasets/dual_resolution_rgbt_dataset.py`.
2. `models/rgbt_rlsfa_gtga_centered_bhlr_model.py`: set alignment valid-content masks to all ones when the data are stretched, both in training (metadata) and prediction (fallback). This is mask logic, **not a network architecture change**.

The old `ratio` field is retained as a scalar compatibility placeholder (`1.0`) and the actual two-axis ratios are in `ratio_xy`, with `stretch=True`.

## Install in-place

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v4
unzip -o YOLO_tcy_v4_Stretch_InPlace.zip
python patch_stretch_inplace.py --repo . --show-diff
python patch_stretch_inplace.py --repo . --apply
python -m py_compile datasets/rgbt_dataset.py models/rgbt_rlsfa_gtga_centered_bhlr_model.py
python verify_stretch_inplace.py
```

Expected:
- RGB high: 1920×1080 → **1920×1920**, no padding
- RGB semantic: 1920×1080 → **640×640**, no padding
- TIR: 640×512 → **640×640**, no padding
- All labels stay YOLO normalized XYWH / HBB.

**Important:** The same resolution and loader code must be used for training and validation. Existing model weights were trained under Letterbox, so retraining is needed to fairly evaluate stretch preprocessing. 1920×1920 RGB changes aspect ratio, not a mere removal of black padding; if you want 1920×1080 unchanged without padding, change the model's image input shape instead.

## Restore

Before modification, installer creates timestamped backups `*.bak_before_stretch_YYYYMMDD_HHMMSS`. To restore, copy those two backups back to their original `.py` paths.

## Caveat

The backup branch on GitHub does not include `datasets/` files. This installer is validated against the `datasets/rgbt_dataset.py` and `datasets/rsdt_dataset.py` interfaces historically used by your YOLO-tcy implementation, and will abort if their expected function names or model mask logic differ. Inspect dry-run diff before applying on server.
