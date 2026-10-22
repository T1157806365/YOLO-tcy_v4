# YOLO-tcy_v4 · backup-20261021_best · P3 product residual fusion

## What changes

This is a **complete installable feature implementation**, not a replacement of the
entire original YOLO-tcy_v4 repository. It needs the existing
`backup-20261021_best` code as its base.

- **Unmodified:** RGB/TIR YOLO26m backbone, GMLS-v2, DMLS routing, Fine
  Translation, Local-GTGA, BHLR-v2, P4/P5 ADD, native YOLO Neck/Detect,
  loss functions, training engine and validator.
- **Added at P3 only:** projected RGB and aligned TIR features → (optionally)
  Hadamard product → depthwise 3x3 → 1x1 → zero-initialized residual gain.
- **Baseline-preserving initialization:** `gamma_raw=0`, so *at initialization*
  `F_new == F_add` (tested exactly in FP32).
- **Modes:** `conv` uses `[R,T,0]`; `product` uses `[R,T,R*T]` with identical
  module parameter count. This isolates the direct product interaction.
- **Pretrained:** `pretrained: true` in generated YAML. The original config
  contained `pretrained: ture` (string, truthy). Generating a proper Boolean
  preserves its effective behavior; make sure the local YOLO26m `.pt` exists.

## Files

```text
P3_PRODUCT_RESIDUAL_README.md
models/modules/p3_product_residual.py
models/rgbt_rlsfa_gtga_centered_bhlr_p3_residual_model.py
install_p3_residual.py
baseline_reference.yaml
generate_p3_configs.py
smoke_test_p3_residual.py
configs/experiments/p3_product_residual/{S0_baseline,S1_conv_residual,S2_product_residual}.yaml
configs/experiments/p3_product_residual_screen/{S0_baseline,S1_conv_residual,S2_product_residual}.yaml
scripts/{train_p3_product,train_p3_conv,validate_p3}.sh
```

## Install safely

Work in your YOLO-tcy_v4 directory. The GitHub **backup branch** should not
be overwritten. Prefer a separate feature branch/worktree if training jobs
are using the existing working tree.

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v4
unzip -o YOLO_tcy_v4_P3_ProductResidual_Full.zip

# Inspect the exact factory diff, make no changes yet
python install_p3_residual.py --repo .

# Installs new Python files and adds only the new factory variant
python install_p3_residual.py --repo . --apply

python -m py_compile \
  models/modules/p3_product_residual.py \
  models/rgbt_rlsfa_gtga_centered_bhlr_p3_residual_model.py \
  models/gtga_bhlr_factory.py
python smoke_test_p3_residual.py
```

Installer creates `models/gtga_bhlr_factory.before_p3_residual.py` on first
application. Running again is idempotent. Existing trainer and validation
scripts are not modified. The built-in packaged YAMLs target UAV-CB and
`device: '0,4'`; adjust to the GPUs actually available.

**Recommended:** regenerate the three complete YAMLs from the **local**
original config (so any differences in your saved baseline are preserved):

```bash
python generate_p3_configs.py \
  --source configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640.yaml \
  --device '0,4'

# Optional short screening experiments (60 epochs, Stage 2 disabled):
python generate_p3_configs.py \
  --source configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640.yaml \
  --device '0,4' --screen \
  --output configs/experiments/p3_product_residual_screen
```

## Mandatory GPU check before training

```bash
nvidia-smi --list-gpus
```

Change `train.device` in the generated YAML to real usable GPU IDs. If
necessary use single GPU `--device 0`, which in the YAML means `device: '0'`.

## Quick unit test + optional full-model smoke test

```bash
python smoke_test_p3_residual.py
# Only if sufficient GPU memory; may take a while and needs your repo deps:
python smoke_test_p3_residual.py --full \
  --yaml configs/experiments/p3_product_residual/S2_product_residual.yaml \
  --device cuda:0
```

`--full` builds a *fresh* non-pretrained model to check the plumbing, not
quality or correct YOLO pretrained weights; training uses YAML's true setting.

## Train (recommended sequence)

Three separate configurations prevent overwriting weights:

- `S0_baseline`: Original ADD model unchanged.
- `S1_conv_residual`: P3 convolutional residual; no Hadamard product.
- `S2_product_residual`: P3 convolutional residual plus Hadamard product.

```bash
# Test both variants with 60-epoch screening configurations
bash scripts/train_p3_conv.sh screen
bash scripts/train_p3_product.sh screen

# Run promising models with the full 300 + 200-epoch schedule
bash scripts/train_p3_product.sh full
```

Or manually:

```bash
mkdir -p logs pid
nohup python -u train_rlsfa_gtga_bhlr_v2.py \
  --cfg configs/experiments/p3_product_residual/S2_product_residual.yaml \
  > logs/p3_product_full.log 2>&1 &
echo $! > pid/p3_product_full.pid
tail -f logs/p3_product_full.log
```

## Validate

After training has produced `best.pt`:

```bash
bash scripts/validate_p3.sh S2_product_residual full val 0
bash scripts/validate_p3.sh S2_product_residual full test 0
```

This invokes the **original** `val_rlsfa_gtga_bhlr_v2.py`, which reads the new
checkpoint's saved config, reconstructs the new variant automatically, and
loads weights with `strict=True`. It reports Precision, Recall, AP50, AP75,
AP50:95, FPS, parameters and FLOPs. Existing alignment visualization commands
are compatible subject to their debug interfaces.

## Important methodological checks

1. Compare on **val** to select the final model; test only after selection.
2. Keep `rgb_imgsz: 1920`, `rgb_semantic_imgsz: 640`, `tir_imgsz: 640`,
   HBB, global batch 32, seed 0, pretrained strategy and data split unchanged.
3. Convolutional model and product model use same number of parameters. One
   zero-valued third input in conv mode is for matched architectural capacity.
4. Identity output initially does **not** imply an old baseline checkpoint can
   be loaded with `strict=True` into the **new variant**. Train a new run or
   implement an explicit compatible partial checkpoint transplant.
5. If the target installation has diverged from the inspected backup branch,
   installer will stop rather than risk corrupting your factory.
6. This package was syntax/unit/installer tested locally but **not yet trained
   or validated on your server GPU**. AP improvements are hypotheses.
