# SPGA-EQ formal experiment patch

This patch starts from the current `T1157806365/YOLO-tcy_v2` main-branch design and replaces moment-based GTGA with **SPGA-EQ**:

- no UAV target map for alignment
- no first moment / second moment
- no bbox-derived geometry
- P5 global structural phase translation
- P4 local phase residual field
- P3 geometry/confidence transfer to BHLR
- known synthetic translation equivariance loss during training
- BHLR-v2 retained as the high-resolution RGB-detail module

## Install

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model
7z x SPGA_EQ_formal_v1.zip
cd SPGA_EQ_formal_v1
chmod +x install.sh
./install.sh /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v2
```

## Train

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v2
mkdir -p logs pid

nohup python -u train_spga_bhlr_v2.py \
  --cfg /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v2/configs/experiments/spga_bhlr_v2/uavcb_spga_bhlr_v2_yolo_1920_640.yaml \
  > logs/uavcb_SPGAEQ_P5P4_BHLRv2_P3_yolo26s.log 2>&1 &

echo $! > pid/uavcb_SPGAEQ_P5P4_BHLRv2_P3_yolo26s.pid

tail -f logs/uavcb_SPGAEQ_P5P4_BHLRv2_P3_yolo26s.log
```

## Validate

```bash
python val_spga_bhlr_v2.py \
  --weights runs/spga_bhlr_v2/yolo26s_UAVCB_SPGAEQ-P5P4_BHLRv2-P3_rgb1920_sem640_tir640_seed0_add/weights/best.pt \
  --split test \
  --device 5 \
  --batch 4 \
  --workers 4 \
  --plots
```

Checkpoint loading remains `strict=True`.
