"""
RSD-T Dataset
=============
Stable dual-resolution RGB-T dataset with optional DistributedSampler support.

The data semantics are unchanged from the single-GPU version. The only DDP
extension is that build_rsdt_dataloader() now accepts sampler= and drop_last=.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.rgbt_dataset import (
    RGBTPairedDataset,
    _parse_imgsz,
    horizontal_flip,
    vertical_flip,
    letterbox,
    load_yolo_label,
)


class RSDTPairedDataset(RGBTPairedDataset):
    def __init__(
        self,
        rgb_yaml: Union[str, "Path"],
        tir_yaml: Union[str, "Path"],
        split: str = "train",
        rgb_high_imgsz: Union[int, Sequence[int]] = 1280,
        rgb_semantic_imgsz: Union[int, Sequence[int]] = 640,
        tir_imgsz: Union[int, Sequence[int]] = 640,
        pair_mode: str = "relative",
        augment: bool = False,
        fliplr: float = 0.5,
        flipud: float = 0.0,
        tir_channels: int = 3,
        scaleup: Optional[bool] = None,
        strict_pair: bool = True,
        strict_label: bool = True,
    ):
        super().__init__(
            rgb_yaml=rgb_yaml,
            tir_yaml=tir_yaml,
            split=split,
            rgb_imgsz=rgb_high_imgsz,
            tir_imgsz=tir_imgsz,
            pair_mode=pair_mode,
            augment=augment,
            fliplr=fliplr,
            flipud=flipud,
            tir_channels=tir_channels,
            scaleup=scaleup,
            strict_pair=strict_pair,
            strict_label=strict_label,
        )

        self.rgb_high_imgsz = _parse_imgsz(rgb_high_imgsz)
        self.rgb_semantic_imgsz = _parse_imgsz(rgb_semantic_imgsz)
        self.rgb_imgsz = self.rgb_high_imgsz

        print(f"RSD-T RGB high: {self.rgb_high_imgsz}")
        print(f"RSD-T RGB semantic: {self.rgb_semantic_imgsz}")

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        rgb_original = self._read_rgb(sample["rgb_path"])
        tir = self._read_tir(sample["tir_path"])

        rgb_ori_shape = rgb_original.shape[:2]
        tir_ori_shape = tir.shape[:2]

        rgb_cls, rgb_boxes_original = load_yolo_label(
            sample["rgb_label"],
            nc=1,
            strict=self.strict_label,
            target_classes=(0,),
            remap_classes=True,
        )
        tir_cls, tir_boxes = load_yolo_label(
            sample["tir_label"],
            nc=1,
            strict=self.strict_label,
            target_classes=(0,),
            remap_classes=True,
        )

        rgb_high, rgb_high_boxes, rgb_high_ratio_pad = letterbox(
            rgb_original.copy(),
            rgb_boxes_original.copy(),
            self.rgb_high_imgsz,
            scaleup=self.scaleup,
        )
        rgb_semantic, rgb_semantic_boxes, rgb_semantic_ratio_pad = letterbox(
            rgb_original.copy(),
            rgb_boxes_original.copy(),
            self.rgb_semantic_imgsz,
            scaleup=self.scaleup,
        )
        tir, tir_boxes, tir_ratio_pad = letterbox(
            tir,
            tir_boxes,
            self.tir_imgsz,
            scaleup=self.scaleup,
        )

        if self.augment:
            import random

            if random.random() < self.fliplr:
                rgb_high, rgb_high_boxes = horizontal_flip(
                    rgb_high, rgb_high_boxes
                )
                rgb_semantic, rgb_semantic_boxes = horizontal_flip(
                    rgb_semantic, rgb_semantic_boxes
                )
                tir, tir_boxes = horizontal_flip(tir, tir_boxes)

            if random.random() < self.flipud:
                rgb_high, rgb_high_boxes = vertical_flip(
                    rgb_high, rgb_high_boxes
                )
                rgb_semantic, rgb_semantic_boxes = vertical_flip(
                    rgb_semantic, rgb_semantic_boxes
                )
                tir, tir_boxes = vertical_flip(tir, tir_boxes)

        rgb_high = np.ascontiguousarray(rgb_high.transpose(2, 0, 1))
        rgb_semantic = np.ascontiguousarray(rgb_semantic.transpose(2, 0, 1))
        tir = np.ascontiguousarray(tir.transpose(2, 0, 1))

        return {
            "rgb_img": torch.from_numpy(rgb_high),
            "rgb_bboxes": torch.from_numpy(rgb_high_boxes).float(),
            "rgb_ratio_pad": rgb_high_ratio_pad,

            "rgb_semantic_img": torch.from_numpy(rgb_semantic),
            "rgb_semantic_bboxes": torch.from_numpy(rgb_semantic_boxes).float(),
            "rgb_semantic_ratio_pad": rgb_semantic_ratio_pad,

            "rgb_cls": torch.from_numpy(rgb_cls).float(),
            "rgb_path": sample["rgb_path"],
            "rgb_label_path": sample["rgb_label"],
            "rgb_ori_shape": rgb_ori_shape,

            "tir_img": torch.from_numpy(tir),
            "tir_cls": torch.from_numpy(tir_cls).float(),
            "tir_bboxes": torch.from_numpy(tir_boxes).float(),
            "tir_path": sample["tir_path"],
            "tir_label_path": sample["tir_label"],
            "tir_ori_shape": tir_ori_shape,
            "tir_ratio_pad": tir_ratio_pad,

            "pair_key": sample["key"],
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        rgb_img = torch.stack([item["rgb_img"] for item in batch], dim=0)
        rgb_semantic_img = torch.stack(
            [item["rgb_semantic_img"] for item in batch], dim=0
        )
        tir_img = torch.stack([item["tir_img"] for item in batch], dim=0)

        rgb_cls_list = []
        rgb_high_box_list = []
        rgb_semantic_box_list = []
        rgb_batch_idx_list = []

        tir_cls_list = []
        tir_box_list = []
        tir_batch_idx_list = []

        for i, item in enumerate(batch):
            n_rgb = len(item["rgb_cls"])
            if n_rgb > 0:
                rgb_cls_list.append(item["rgb_cls"])
                rgb_high_box_list.append(item["rgb_bboxes"])
                rgb_semantic_box_list.append(item["rgb_semantic_bboxes"])
                rgb_batch_idx_list.append(
                    torch.full((n_rgb,), i, dtype=torch.long)
                )

            n_tir = len(item["tir_cls"])
            if n_tir > 0:
                tir_cls_list.append(item["tir_cls"])
                tir_box_list.append(item["tir_bboxes"])
                tir_batch_idx_list.append(
                    torch.full((n_tir,), i, dtype=torch.long)
                )

        if rgb_cls_list:
            rgb_cls = torch.cat(rgb_cls_list, dim=0)
            rgb_bboxes = torch.cat(rgb_high_box_list, dim=0)
            rgb_semantic_bboxes = torch.cat(rgb_semantic_box_list, dim=0)
            rgb_batch_idx = torch.cat(rgb_batch_idx_list, dim=0)
        else:
            rgb_cls = torch.zeros((0, 1), dtype=torch.float32)
            rgb_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            rgb_semantic_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            rgb_batch_idx = torch.zeros((0,), dtype=torch.long)

        if tir_cls_list:
            tir_cls = torch.cat(tir_cls_list, dim=0)
            tir_bboxes = torch.cat(tir_box_list, dim=0)
            tir_batch_idx = torch.cat(tir_batch_idx_list, dim=0)
        else:
            tir_cls = torch.zeros((0, 1), dtype=torch.float32)
            tir_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            tir_batch_idx = torch.zeros((0,), dtype=torch.long)

        return {
            "rgb_img": rgb_img,
            "rgb_semantic_img": rgb_semantic_img,
            "tir_img": tir_img,

            "rgb_cls": rgb_cls,
            "rgb_bboxes": rgb_bboxes,
            "rgb_semantic_bboxes": rgb_semantic_bboxes,
            "rgb_batch_idx": rgb_batch_idx,

            "tir_cls": tir_cls,
            "tir_bboxes": tir_bboxes,
            "tir_batch_idx": tir_batch_idx,

            "rgb_path": [item["rgb_path"] for item in batch],
            "tir_path": [item["tir_path"] for item in batch],
            "pair_key": [item["pair_key"] for item in batch],

            "rgb_ori_shape": [item["rgb_ori_shape"] for item in batch],
            "tir_ori_shape": [item["tir_ori_shape"] for item in batch],
            "rgb_ratio_pad": [item["rgb_ratio_pad"] for item in batch],
            "rgb_semantic_ratio_pad": [
                item["rgb_semantic_ratio_pad"] for item in batch
            ],
            "tir_ratio_pad": [item["tir_ratio_pad"] for item in batch],
        }


def build_rsdt_dataset(
    rgb_yaml,
    tir_yaml,
    split="train",
    rgb_high_imgsz=1280,
    rgb_semantic_imgsz=640,
    tir_imgsz=640,
    pair_mode="relative",
    augment=False,
    fliplr=0.5,
    flipud=0.0,
    tir_channels=3,
    strict_pair=True,
):
    return RSDTPairedDataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=split,
        rgb_high_imgsz=rgb_high_imgsz,
        rgb_semantic_imgsz=rgb_semantic_imgsz,
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=augment,
        fliplr=fliplr,
        flipud=flipud,
        tir_channels=tir_channels,
        strict_pair=strict_pair,
    )


def build_rsdt_dataloader(
    dataset: RSDTPairedDataset,
    batch_size: int = 8,
    workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    sampler=None,
    drop_last: bool = False,
):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=(bool(shuffle) if sampler is None else False),
        sampler=sampler,
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        collate_fn=dataset.collate_fn,
        persistent_workers=(int(workers) > 0),
        drop_last=bool(drop_last),
    )
