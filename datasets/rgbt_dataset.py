"""
Paired RGB-T Dataset for YOLO-tcy

设计目标
--------
RGB 和 TIR 完全作为两个独立模态：

RGB image  -> RGB Backbone
TIR image  -> TIR Backbone

RGB labels -> RGB supervision
TIR labels -> TIR supervision

不进行 4-channel early fusion。

Batch 输出结构：

{
    "rgb_img":        [B, 3, H_rgb, W_rgb],
    "tir_img":        [B, 3, H_tir, W_tir],

    "rgb_cls":        [N_rgb, 1],
    "rgb_bboxes":     [N_rgb, 4],
    "rgb_batch_idx":  [N_rgb],

    "tir_cls":        [N_tir, 1],
    "tir_bboxes":     [N_tir, 4],
    "tir_batch_idx":  [N_tir],

    ...
}

其中 bbox 均为：
    normalized xywh

项目路径：
    /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ultralytics.data.utils import IMG_FORMATS, img2label_paths


# ============================================================
# Allow direct execution
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.config import (
    load_yaml,
    resolve_dataset_yaml,
)


# ============================================================
# 1. Basic utilities
# ============================================================

def _as_list(value):
    """Convert YAML field into list."""

    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def _parse_imgsz(
    imgsz: Union[int, Sequence[int]],
) -> Tuple[int, int]:
    """
    Convert imgsz into (height, width).

    Examples
    --------
    640
        -> (640, 640)

    [512, 640]
        -> (512, 640)
    """

    if isinstance(imgsz, int):
        return imgsz, imgsz

    if isinstance(imgsz, (list, tuple)):

        if len(imgsz) == 1:
            return int(imgsz[0]), int(imgsz[0])

        if len(imgsz) == 2:
            return int(imgsz[0]), int(imgsz[1])

    raise ValueError(
        f"无法解析 imgsz: {imgsz}"
    )


# ============================================================
# 2. Resolve dataset YAML paths
# ============================================================

def _dataset_root(
    yaml_file: Path,
    cfg: Dict,
) -> Path:
    """
    Resolve YAML 'path:' field.
    """

    root = cfg.get("path")

    if root is None:
        return yaml_file.parent.resolve()

    root = Path(
        os.path.expandvars(
            os.path.expanduser(
                str(root)
            )
        )
    )

    if root.is_absolute():
        return root.resolve()

    return (
        yaml_file.parent / root
    ).resolve()


def resolve_split_paths(
    yaml_file: Union[str, Path],
    split: str,
) -> List[Path]:
    """
    Resolve train / val / test entries.
    """

    yaml_file = Path(
        yaml_file
    ).resolve()

    cfg = load_yaml(
        yaml_file
    )

    if split not in cfg:
        raise KeyError(
            f"\nYAML 中不存在 split='{split}'\n"
            f"文件: {yaml_file}\n"
        )

    root = _dataset_root(
        yaml_file,
        cfg,
    )

    values = _as_list(
        cfg[split]
    )

    paths = []

    for value in values:

        value = os.path.expandvars(
            os.path.expanduser(
                str(value)
            )
        )

        p = Path(value)

        if not p.is_absolute():
            p = root / p

        paths.append(
            p.resolve()
        )

    if not paths:
        raise RuntimeError(
            f"{yaml_file}: {split} 为空"
        )

    return paths


# ============================================================
# 3. Collect images
# ============================================================

def collect_images(
    sources: Sequence[Path],
) -> List[str]:
    """
    Collect image files from:

        directory
        txt file
        single image
    """

    valid_ext = {
        "." + x.lower()
        for x in IMG_FORMATS
    }

    images: List[str] = []

    for source in sources:

        source = Path(source)

        # ----------------------------------------------------
        # Directory
        # ----------------------------------------------------
        if source.is_dir():

            for file in source.rglob("*"):

                if (
                    file.is_file()
                    and file.suffix.lower()
                    in valid_ext
                ):
                    images.append(
                        str(file.resolve())
                    )

        # ----------------------------------------------------
        # txt list
        # ----------------------------------------------------
        elif (
            source.is_file()
            and source.suffix.lower() == ".txt"
        ):

            with source.open(
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    p = Path(line)

                    if not p.is_absolute():
                        p = source.parent / p

                    p = p.resolve()

                    if (
                        p.exists()
                        and p.suffix.lower()
                        in valid_ext
                    ):
                        images.append(
                            str(p)
                        )

        # ----------------------------------------------------
        # Single image
        # ----------------------------------------------------
        elif (
            source.is_file()
            and source.suffix.lower()
            in valid_ext
        ):

            images.append(
                str(source.resolve())
            )

        else:

            raise FileNotFoundError(
                f"\n图像路径不存在:\n{source}"
            )

    images = sorted(
        set(images)
    )

    if len(images) == 0:
        raise RuntimeError(
            f"没有找到任何图像:\n{sources}"
        )

    return images


# ============================================================
# 4. Pair RGB and TIR
# ============================================================

def _source_root(
    source: Path,
) -> Path:
    """
    Directory -> itself
    txt       -> its parent
    """

    source = Path(source)

    if source.is_dir():
        return source.resolve()

    return source.parent.resolve()


def _relative_key(
    file: Union[str, Path],
    roots: Sequence[Path],
) -> Optional[str]:
    """
    Find relative path under one of split roots.

    Extension is removed so:
        xxx.jpg
        xxx.png

    can still pair.
    """

    file = Path(file).resolve()

    for root in roots:

        root = _source_root(
            root
        )

        try:

            rel = file.relative_to(
                root
            )

            return (
                rel
                .with_suffix("")
                .as_posix()
                .lower()
            )

        except ValueError:
            continue

    return None


def make_pair_key(
    file: Union[str, Path],
    mode: str,
    roots: Sequence[Path],
) -> str:
    """
    Pairing strategy.

    stem:
        image001.jpg -> image001

    filename:
        image001.jpg -> image001.jpg

    relative:
        seq01/image001.jpg
        -> seq01/image001
    """

    file = Path(file)

    mode = mode.lower()

    if mode == "stem":
        return file.stem.lower()

    if mode == "filename":
        return file.name.lower()

    if mode == "relative":

        key = _relative_key(
            file,
            roots,
        )

        if key is None:
            raise RuntimeError(
                f"无法生成 relative pair key:\n{file}"
            )

        return key

    raise ValueError(
        f"不支持 pair_mode={mode}"
    )


def build_index(
    files: Sequence[str],
    roots: Sequence[Path],
    mode: str,
    modality: str,
) -> Dict[str, str]:
    """
    Build:

        pair_key -> image path
    """

    index = {}

    duplicates = []

    for file in files:

        key = make_pair_key(
            file,
            mode,
            roots,
        )

        if key in index:

            duplicates.append(
                (
                    key,
                    index[key],
                    file,
                )
            )

        else:
            index[key] = file

    if duplicates:

        text = "\n".join(
            [
                (
                    f"{key}\n"
                    f"  1: {a}\n"
                    f"  2: {b}"
                )
                for key, a, b
                in duplicates[:10]
            ]
        )

        raise RuntimeError(
            f"\n{modality} 出现重复 pair key。\n"
            f"pair_mode={mode}\n\n"
            f"{text}\n\n"
            "建议改用 --pair-mode relative"
        )

    return index


# ============================================================
# 5. YOLO label loading
# ============================================================

def image_to_label_path(
    image_path: Union[str, Path],
) -> Path:
    """
    Use Ultralytics standard images -> labels mapping.

    Example:

        dataset/images/train/a.jpg

    ->

        dataset/labels/train/a.txt
    """

    path = img2label_paths(
        [str(image_path)]
    )[0]

    return Path(path)


def load_yolo_label(
    label_path: Union[str, Path],
    nc: Optional[int] = None,
    strict: bool = True,
    target_classes: Optional[Sequence[int]] = (0,),
    remap_classes: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read YOLO detection labels and optionally keep only selected classes.

    Original YOLO format:
        class x_center y_center width height

    Current LRDD_v3 setting:
        class 0 -> UAV       keep
        class 1 -> other     remove
        class 2 -> other     remove

    Parameters
    ----------
    label_path:
        YOLO txt label path.

    nc:
        Number of classes AFTER filtering.
        For UAV-only detection:
            nc = 1

    strict:
        Validate normalized xywh.

    target_classes:
        Original class IDs to keep.

        Example:
            (0,)       -> only UAV
            (0, 1)     -> keep classes 0 and 1
            None       -> keep all classes

    remap_classes:
        Remap selected original class IDs to consecutive IDs.

        Example:
            target_classes=(2,)
            original class 2 -> new class 0

            target_classes=(0, 2)
            original 0 -> new 0
            original 2 -> new 1

    Returns
    -------
    cls:
        [N, 1]

    boxes:
        [N, 4]

    boxes are normalized xywh.
    """

    label_path = Path(label_path)

    # ========================================================
    # Missing label = background
    # ========================================================

    if not label_path.exists():

        return (
            np.zeros(
                (0, 1),
                dtype=np.float32,
            ),
            np.zeros(
                (0, 4),
                dtype=np.float32,
            ),
        )

    # ========================================================
    # Read file
    # ========================================================

    labels = []

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:

                raise ValueError(
                    "\n当前只支持 YOLO Detection 标签。\n"
                    f"文件: {label_path}\n"
                    f"第 {line_number} 行:\n"
                    f"{line}\n"
                )

            values = [
                float(x)
                for x in parts
            ]

            labels.append(values)

    # ========================================================
    # Empty label file
    # ========================================================

    if not labels:

        return (
            np.zeros(
                (0, 1),
                dtype=np.float32,
            ),
            np.zeros(
                (0, 4),
                dtype=np.float32,
            ),
        )

    labels = np.asarray(
        labels,
        dtype=np.float32,
    )

    # ========================================================
    # IMPORTANT:
    # Filter classes BEFORE nc validation
    # ========================================================

    if target_classes is not None:

        target_classes = [
            int(c)
            for c in target_classes
        ]

        original_cls = (
            labels[:, 0]
            .astype(np.int64)
        )

        keep_mask = np.isin(
            original_cls,
            target_classes,
        )

        labels = labels[
            keep_mask
        ]

        # -----------------------------------------------
        # File contains labels, but none are UAV.
        # Then this image becomes a background sample.
        # -----------------------------------------------

        if len(labels) == 0:

            return (
                np.zeros(
                    (0, 1),
                    dtype=np.float32,
                ),
                np.zeros(
                    (0, 4),
                    dtype=np.float32,
                ),
            )

        # ====================================================
        # Remap selected classes
        #
        # For target_classes=(0,):
        #     0 -> 0
        #
        # For target_classes=(2,):
        #     2 -> 0
        # ====================================================

        if remap_classes:

            class_map = {
                old_cls: new_cls
                for new_cls, old_cls
                in enumerate(
                    target_classes
                )
            }

            labels[:, 0] = np.asarray(
                [
                    class_map[
                        int(c)
                    ]
                    for c in labels[:, 0]
                ],
                dtype=np.float32,
            )

    # ========================================================
    # Split cls / boxes
    # ========================================================

    cls = labels[:, 0:1]

    boxes = labels[:, 1:5]

    # ========================================================
    # Validate filtered class IDs
    # ========================================================

    if nc is not None and len(cls):

        if (
            cls.min() < 0
            or cls.max() >= nc
        ):

            raise ValueError(
                "\n过滤后的类别 ID 超出范围:\n"
                f"{label_path}\n"
                f"nc={nc}\n"
                f"class range="
                f"{cls.min()}~{cls.max()}\n"
            )

    # ========================================================
    # Validate normalized xywh
    # ========================================================

    if strict and len(boxes):

        if (
            np.any(boxes < 0)
            or np.any(boxes > 1)
        ):

            raise ValueError(
                "\n发现非 normalized xywh 标签:\n"
                f"{label_path}\n"
            )

        if np.any(
            boxes[:, 2:4] <= 0
        ):

            raise ValueError(
                "\n发现 width/height <= 0:\n"
                f"{label_path}\n"
            )

    return (
        cls.astype(
            np.float32
        ),
        boxes.astype(
            np.float32
        ),
    )


# ============================================================
# 6. LetterBox + bbox transformation
# ============================================================

def letterbox(
    image: np.ndarray,
    boxes: np.ndarray,
    new_shape: Union[
        int,
        Sequence[int],
    ],
    scaleup: bool = True,
    color: int = 114,
):
    """
    LetterBox one modality independently.

    boxes:
        normalized xywh in original image.

    returns:
        image
        normalized xywh in new image
        ratio_pad
    """

    new_h, new_w = _parse_imgsz(
        new_shape
    )

    h0, w0 = image.shape[:2]

    r = min(
        new_h / h0,
        new_w / w0,
    )

    if not scaleup:
        r = min(
            r,
            1.0,
        )

    resize_w = int(
        round(w0 * r)
    )

    resize_h = int(
        round(h0 * r)
    )

    if (
        resize_w != w0
        or resize_h != h0
    ):

        image = cv2.resize(
            image,
            (resize_w, resize_h),
            interpolation=cv2.INTER_LINEAR,
        )

    dw = new_w - resize_w
    dh = new_h - resize_h

    dw /= 2
    dh /= 2

    left = int(
        round(dw - 0.1)
    )

    right = int(
        round(dw + 0.1)
    )

    top = int(
        round(dh - 0.1)
    )

    bottom = int(
        round(dh + 0.1)
    )

    if image.ndim == 2:

        border_value = color

    else:

        border_value = tuple(
            [color] * image.shape[2]
        )

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )

    # --------------------------------------------------------
    # Transform bbox
    # --------------------------------------------------------
    new_boxes = boxes.copy()

    if len(new_boxes):

        # normalized xywh -> original absolute xyxy
        cx = new_boxes[:, 0] * w0
        cy = new_boxes[:, 1] * h0
        bw = new_boxes[:, 2] * w0
        bh = new_boxes[:, 3] * h0

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # resize
        x1 = x1 * r + left
        x2 = x2 * r + left

        y1 = y1 * r + top
        y2 = y2 * r + top

        # clip
        x1 = np.clip(
            x1,
            0,
            new_w,
        )

        x2 = np.clip(
            x2,
            0,
            new_w,
        )

        y1 = np.clip(
            y1,
            0,
            new_h,
        )

        y2 = np.clip(
            y2,
            0,
            new_h,
        )

        # absolute xyxy -> normalized xywh
        new_boxes[:, 0] = (
            (x1 + x2)
            / 2
            / new_w
        )

        new_boxes[:, 1] = (
            (y1 + y2)
            / 2
            / new_h
        )

        new_boxes[:, 2] = (
            (x2 - x1)
            / new_w
        )

        new_boxes[:, 3] = (
            (y2 - y1)
            / new_h
        )

    ratio_pad = {
        "ratio": r,
        "pad": (
            left,
            top,
        ),
    }

    return (
        image,
        new_boxes,
        ratio_pad,
    )


# ============================================================
# 7. Shared geometric augmentation
# ============================================================

def horizontal_flip(
    image: np.ndarray,
    boxes: np.ndarray,
):
    """
    Horizontal flip.

    boxes:
        normalized xywh
    """

    image = np.ascontiguousarray(
        image[:, ::-1]
    )

    boxes = boxes.copy()

    if len(boxes):
        boxes[:, 0] = (
            1.0 - boxes[:, 0]
        )

    return image, boxes


def vertical_flip(
    image: np.ndarray,
    boxes: np.ndarray,
):
    image = np.ascontiguousarray(
        image[::-1, :]
    )

    boxes = boxes.copy()

    if len(boxes):
        boxes[:, 1] = (
            1.0 - boxes[:, 1]
        )

    return image, boxes


# ============================================================
# 8. Paired sample
# ============================================================

class RGBTPairedDataset(Dataset):
    """
    RGB-T paired detection dataset.

    Important
    ---------
    RGB / TIR:
        independent image tensor

    RGB / TIR:
        independent YOLO labels

    Geometric augmentation:
        same random flip decision for both modalities

    Image sizes:
        may be different.
    """

    def __init__(
        self,
        rgb_yaml: Union[str, Path],
        tir_yaml: Union[str, Path],
        split: str = "train",

        rgb_imgsz: Union[
            int,
            Sequence[int],
        ] = 640,

        tir_imgsz: Union[
            int,
            Sequence[int],
        ] = 640,

        pair_mode: str = "relative",

        augment: bool = False,

        fliplr: float = 0.5,

        flipud: float = 0.0,

        tir_channels: int = 3,

        scaleup: Optional[bool] = None,

        strict_pair: bool = True,

        strict_label: bool = True,
    ):

        super().__init__()

        # ====================================================
        # Configuration
        # ====================================================

        self.rgb_yaml = (
            resolve_dataset_yaml(
                rgb_yaml
            )
        )

        self.tir_yaml = (
            resolve_dataset_yaml(
                tir_yaml
            )
        )

        self.split = split

        self.rgb_imgsz = _parse_imgsz(
            rgb_imgsz
        )

        self.tir_imgsz = _parse_imgsz(
            tir_imgsz
        )

        self.pair_mode = (
            pair_mode.lower()
        )

        self.augment = augment

        self.fliplr = float(
            fliplr
        )

        self.flipud = float(
            flipud
        )

        self.tir_channels = int(
            tir_channels
        )

        self.strict_pair = (
            strict_pair
        )

        self.strict_label = (
            strict_label
        )

        if self.tir_channels not in {
            1,
            3,
        }:

            raise ValueError(
                "tir_channels 只能为 1 或 3"
            )

        if scaleup is None:

            self.scaleup = (
                split == "train"
            )

        else:

            self.scaleup = bool(
                scaleup
            )

        # ====================================================
        # Load YAML
        # ====================================================

        self.rgb_cfg = load_yaml(
            self.rgb_yaml
        )

        self.tir_cfg = load_yaml(
            self.tir_yaml
        )

        # ====================================================
        # Class names
        # ====================================================

        self.rgb_names = (
            self.rgb_cfg.get(
                "names"
            )
        )

        self.tir_names = (
            self.tir_cfg.get(
                "names"
            )
        )

        if (
            self.rgb_names is not None
            and self.tir_names is not None
            and self.rgb_names
            != self.tir_names
        ):

            raise ValueError(
                "\nRGB 和 TIR 类别定义不一致。\n"
                f"RGB names={self.rgb_names}\n"
                f"TIR names={self.tir_names}"
            )

        self.names = (
            self.rgb_names
            if self.rgb_names
            is not None
            else self.tir_names
        )

        if self.names is None:

            self.nc = int(
                self.rgb_cfg.get(
                    "nc",
                    self.tir_cfg.get(
                        "nc",
                        1,
                    ),
                )
            )

        else:

            self.nc = len(
                self.names
            )

        # ====================================================
        # Split paths
        # ====================================================

        self.rgb_sources = (
            resolve_split_paths(
                self.rgb_yaml,
                split,
            )
        )

        self.tir_sources = (
            resolve_split_paths(
                self.tir_yaml,
                split,
            )
        )

        # ====================================================
        # Image lists
        # ====================================================

        rgb_files = collect_images(
            self.rgb_sources
        )

        tir_files = collect_images(
            self.tir_sources
        )

        # ====================================================
        # Build pair indices
        # ====================================================

        rgb_index = build_index(
            rgb_files,
            self.rgb_sources,
            self.pair_mode,
            "RGB",
        )

        tir_index = build_index(
            tir_files,
            self.tir_sources,
            self.pair_mode,
            "TIR",
        )

        rgb_keys = set(
            rgb_index.keys()
        )

        tir_keys = set(
            tir_index.keys()
        )

        common_keys = sorted(
            rgb_keys
            & tir_keys
        )

        missing_tir = sorted(
            rgb_keys - tir_keys
        )

        missing_rgb = sorted(
            tir_keys - rgb_keys
        )

        if self.strict_pair:

            if missing_tir:

                raise RuntimeError(
                    "\n部分 RGB 没有对应 TIR。\n"
                    f"数量: {len(missing_tir)}\n"
                    f"示例: {missing_tir[:10]}\n"
                )

            if missing_rgb:

                raise RuntimeError(
                    "\n部分 TIR 没有对应 RGB。\n"
                    f"数量: {len(missing_rgb)}\n"
                    f"示例: {missing_rgb[:10]}\n"
                )

        if not common_keys:

            raise RuntimeError(
                "\nRGB/TIR 没有成功配对。\n"
                f"pair_mode={self.pair_mode}\n"
            )

        self.samples = []

        for key in common_keys:

            rgb_path = rgb_index[key]

            tir_path = tir_index[key]

            rgb_label = (
                image_to_label_path(
                    rgb_path
                )
            )

            tir_label = (
                image_to_label_path(
                    tir_path
                )
            )

            self.samples.append(
                {
                    "key": key,

                    "rgb_path":
                        rgb_path,

                    "tir_path":
                        tir_path,

                    "rgb_label":
                        str(rgb_label),

                    "tir_label":
                        str(tir_label),
                }
            )

        # ====================================================
        # Information
        # ====================================================

        print(
            "\n"
            "============================================================"
        )

        print(
            "RGB-T Paired Dataset"
        )

        print(
            "============================================================"
        )

        print(
            f"Split        : {split}"
        )

        print(
            f"RGB YAML     : {self.rgb_yaml}"
        )

        print(
            f"TIR YAML     : {self.tir_yaml}"
        )

        print(
            f"RGB images   : {len(rgb_files)}"
        )

        print(
            f"TIR images   : {len(tir_files)}"
        )

        print(
            f"Pairs        : {len(self.samples)}"
        )

        print(
            f"Missing TIR  : {len(missing_tir)}"
        )

        print(
            f"Missing RGB  : {len(missing_rgb)}"
        )

        print(
            f"Pair mode    : {self.pair_mode}"
        )

        print(
            f"RGB imgsz    : {self.rgb_imgsz}"
        )

        print(
            f"TIR imgsz    : {self.tir_imgsz}"
        )

        print(
            f"TIR channels : {self.tir_channels}"
        )

        print(
            f"Classes      : {self.nc}"
        )

        print(
            "============================================================\n"
        )

    # ========================================================
    # length
    # ========================================================

    def __len__(self):

        return len(
            self.samples
        )

    # ========================================================
    # RGB read
    # ========================================================

    @staticmethod
    def _read_rgb(
        path: str,
    ) -> np.ndarray:

        image = cv2.imread(
            path,
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise FileNotFoundError(
                f"RGB 图像读取失败:\n{path}"
            )

        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        return image

    # ========================================================
    # TIR read
    # ========================================================

    def _read_tir(
        self,
        path: str,
    ) -> np.ndarray:

        tir = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE,
        )

        if tir is None:

            raise FileNotFoundError(
                f"TIR 图像读取失败:\n{path}"
            )

        if self.tir_channels == 3:

            # ------------------------------------------------
            # Repeat thermal grayscale to 3 channels.
            #
            # Advantage:
            # TIR backbone can directly use normal YOLO
            # pretrained 3-channel convolution weights.
            # ------------------------------------------------
            tir = np.stack(
                [
                    tir,
                    tir,
                    tir,
                ],
                axis=-1,
            )

        else:

            tir = tir[..., None]

        return tir

    # ========================================================
    # get item
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ) -> Dict:

        sample = self.samples[
            index
        ]

        # ====================================================
        # Read two modalities
        # ====================================================

        rgb = self._read_rgb(
            sample["rgb_path"]
        )

        tir = self._read_tir(
            sample["tir_path"]
        )

        rgb_ori_shape = (
            rgb.shape[:2]
        )

        tir_ori_shape = (
            tir.shape[:2]
        )

        # ====================================================
        # Read two independent labels
        # ====================================================

        rgb_cls, rgb_boxes = (
            load_yolo_label(
                sample["rgb_label"],

                nc=1,

                strict=self.strict_label,

                # 只保留 UAV
                target_classes=(0,),

                # 原 class 0 仍然映射到 class 0
                remap_classes=True,
            )
        )

        tir_cls, tir_boxes = (
            load_yolo_label(
                sample["tir_label"],

                nc=1,

                strict=self.strict_label,

                # 只保留 UAV
                target_classes=(0,),

                remap_classes=True,
            )
        )

        # ====================================================
        # Independent LetterBox
        #
        # RGB and TIR may have different input resolutions.
        # ====================================================

        (
            rgb,
            rgb_boxes,
            rgb_ratio_pad,
        ) = letterbox(
            rgb,
            rgb_boxes,
            self.rgb_imgsz,
            scaleup=self.scaleup,
        )

        (
            tir,
            tir_boxes,
            tir_ratio_pad,
        ) = letterbox(
            tir,
            tir_boxes,
            self.tir_imgsz,
            scaleup=self.scaleup,
        )

        # ====================================================
        # Shared geometric augmentation
        #
        # Same random decision is applied to RGB and TIR.
        # ====================================================

        if self.augment:

            # Horizontal flip
            do_fliplr = (
                random.random()
                < self.fliplr
            )

            if do_fliplr:

                rgb, rgb_boxes = (
                    horizontal_flip(
                        rgb,
                        rgb_boxes,
                    )
                )

                tir, tir_boxes = (
                    horizontal_flip(
                        tir,
                        tir_boxes,
                    )
                )

            # Vertical flip
            do_flipud = (
                random.random()
                < self.flipud
            )

            if do_flipud:

                rgb, rgb_boxes = (
                    vertical_flip(
                        rgb,
                        rgb_boxes,
                    )
                )

                tir, tir_boxes = (
                    vertical_flip(
                        tir,
                        tir_boxes,
                    )
                )

        # ====================================================
        # HWC -> CHW
        # ====================================================

        rgb = np.ascontiguousarray(
            rgb.transpose(
                2,
                0,
                1,
            )
        )

        tir = np.ascontiguousarray(
            tir.transpose(
                2,
                0,
                1,
            )
        )

        # ====================================================
        # numpy -> torch
        #
        # Keep uint8 here.
        # train_rgbt.py will normalize /255.
        # ====================================================

        rgb = torch.from_numpy(
            rgb
        )

        tir = torch.from_numpy(
            tir
        )

        rgb_cls = torch.from_numpy(
            rgb_cls
        ).float()

        rgb_boxes = torch.from_numpy(
            rgb_boxes
        ).float()

        tir_cls = torch.from_numpy(
            tir_cls
        ).float()

        tir_boxes = torch.from_numpy(
            tir_boxes
        ).float()

        # ====================================================
        # Output
        # ====================================================

        return {
            # ---------------- RGB ----------------
            "rgb_img":
                rgb,

            "rgb_cls":
                rgb_cls,

            "rgb_bboxes":
                rgb_boxes,

            "rgb_path":
                sample[
                    "rgb_path"
                ],

            "rgb_label_path":
                sample[
                    "rgb_label"
                ],

            "rgb_ori_shape":
                rgb_ori_shape,

            "rgb_ratio_pad":
                rgb_ratio_pad,

            # ---------------- TIR ----------------
            "tir_img":
                tir,

            "tir_cls":
                tir_cls,

            "tir_bboxes":
                tir_boxes,

            "tir_path":
                sample[
                    "tir_path"
                ],

            "tir_label_path":
                sample[
                    "tir_label"
                ],

            "tir_ori_shape":
                tir_ori_shape,

            "tir_ratio_pad":
                tir_ratio_pad,

            # ---------------- pair ----------------
            "pair_key":
                sample[
                    "key"
                ],
        }

    # ========================================================
    # Collate
    # ========================================================

    @staticmethod
    def collate_fn(
        batch: List[Dict],
    ) -> Dict:
        """
        Build RGB-T batch.

        Similar concept to Ultralytics YOLO collate,
        but RGB and TIR each keep independent targets.
        """

        # ====================================================
        # Images
        # ====================================================

        rgb_img = torch.stack(
            [
                item["rgb_img"]
                for item in batch
            ],
            dim=0,
        )

        tir_img = torch.stack(
            [
                item["tir_img"]
                for item in batch
            ],
            dim=0,
        )

        # ====================================================
        # RGB targets
        # ====================================================

        rgb_cls_list = []

        rgb_box_list = []

        rgb_batch_idx = []

        # ====================================================
        # TIR targets
        # ====================================================

        tir_cls_list = []

        tir_box_list = []

        tir_batch_idx = []

        # ====================================================
        # Build indices
        # ====================================================

        for i, item in enumerate(
            batch
        ):

            n_rgb = len(
                item[
                    "rgb_cls"
                ]
            )

            if n_rgb > 0:

                rgb_cls_list.append(
                    item[
                        "rgb_cls"
                    ]
                )

                rgb_box_list.append(
                    item[
                        "rgb_bboxes"
                    ]
                )

                rgb_batch_idx.append(
                    torch.full(
                        (n_rgb,),
                        i,
                        dtype=torch.long,
                    )
                )

            n_tir = len(
                item[
                    "tir_cls"
                ]
            )

            if n_tir > 0:

                tir_cls_list.append(
                    item[
                        "tir_cls"
                    ]
                )

                tir_box_list.append(
                    item[
                        "tir_bboxes"
                    ]
                )

                tir_batch_idx.append(
                    torch.full(
                        (n_tir,),
                        i,
                        dtype=torch.long,
                    )
                )

        # ====================================================
        # Concatenate RGB labels
        # ====================================================

        if rgb_cls_list:

            rgb_cls = torch.cat(
                rgb_cls_list,
                dim=0,
            )

            rgb_bboxes = torch.cat(
                rgb_box_list,
                dim=0,
            )

            rgb_batch_idx = torch.cat(
                rgb_batch_idx,
                dim=0,
            )

        else:

            rgb_cls = torch.zeros(
                (0, 1),
                dtype=torch.float32,
            )

            rgb_bboxes = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            rgb_batch_idx = torch.zeros(
                (0,),
                dtype=torch.long,
            )

        # ====================================================
        # Concatenate TIR labels
        # ====================================================

        if tir_cls_list:

            tir_cls = torch.cat(
                tir_cls_list,
                dim=0,
            )

            tir_bboxes = torch.cat(
                tir_box_list,
                dim=0,
            )

            tir_batch_idx = torch.cat(
                tir_batch_idx,
                dim=0,
            )

        else:

            tir_cls = torch.zeros(
                (0, 1),
                dtype=torch.float32,
            )

            tir_bboxes = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            tir_batch_idx = torch.zeros(
                (0,),
                dtype=torch.long,
            )

        # ====================================================
        # Batch
        # ====================================================

        return {
            # ---------------- Images ----------------
            "rgb_img":
                rgb_img,

            "tir_img":
                tir_img,

            # ---------------- RGB GT ----------------
            "rgb_cls":
                rgb_cls,

            "rgb_bboxes":
                rgb_bboxes,

            "rgb_batch_idx":
                rgb_batch_idx,

            # ---------------- TIR GT ----------------
            "tir_cls":
                tir_cls,

            "tir_bboxes":
                tir_bboxes,

            "tir_batch_idx":
                tir_batch_idx,

            # ---------------- Paths ----------------
            "rgb_path":
                [
                    item[
                        "rgb_path"
                    ]
                    for item in batch
                ],

            "tir_path":
                [
                    item[
                        "tir_path"
                    ]
                    for item in batch
                ],

            "pair_key":
                [
                    item[
                        "pair_key"
                    ]
                    for item in batch
                ],

            # ---------------- Original shapes ----------------
            "rgb_ori_shape":
                [
                    item[
                        "rgb_ori_shape"
                    ]
                    for item in batch
                ],

            "tir_ori_shape":
                [
                    item[
                        "tir_ori_shape"
                    ]
                    for item in batch
                ],

            "rgb_ratio_pad":
                [
                    item[
                        "rgb_ratio_pad"
                    ]
                    for item in batch
                ],

            "tir_ratio_pad":
                [
                    item[
                        "tir_ratio_pad"
                    ]
                    for item in batch
                ],
        }


# ============================================================
# 9. Convenient builder
# ============================================================

def build_rgbt_dataset(
    rgb_yaml: Union[str, Path],
    tir_yaml: Union[str, Path],
    split: str = "train",

    rgb_imgsz: Union[
        int,
        Sequence[int],
    ] = 640,

    tir_imgsz: Union[
        int,
        Sequence[int],
    ] = 640,

    pair_mode: str = "relative",

    augment: bool = False,

    fliplr: float = 0.5,

    flipud: float = 0.0,

    tir_channels: int = 3,

    strict_pair: bool = True,
):
    """
    Unified dataset builder.
    """

    return RGBTPairedDataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,

        split=split,

        rgb_imgsz=rgb_imgsz,
        tir_imgsz=tir_imgsz,

        pair_mode=pair_mode,

        augment=augment,

        fliplr=fliplr,
        flipud=flipud,

        tir_channels=tir_channels,

        strict_pair=strict_pair,
    )


# ============================================================
# 10. DataLoader builder
# ============================================================

def build_rgbt_dataloader(
    dataset: RGBTPairedDataset,

    batch_size: int = 8,

    workers: int = 4,

    shuffle: bool = True,

    pin_memory: bool = True,
):
    """
    Build PyTorch DataLoader.
    """

    return DataLoader(
        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=workers,

        pin_memory=pin_memory,

        collate_fn=(
            dataset.collate_fn
        ),

        persistent_workers=(
            workers > 0
        ),
    )


# ============================================================
# 11. Self-test
# ============================================================

def self_test(
    rgb_yaml: str,
    tir_yaml: str,

    split: str,

    rgb_imgsz: int,
    tir_imgsz: int,

    pair_mode: str,

    batch_size: int,

    workers: int,

    samples: int,
):
    """
    Test dataset without model.
    """

    print(
        "\n"
        "############################################################"
    )

    print(
        "RGB-T DATASET TEST"
    )

    print(
        "############################################################"
    )

    dataset = RGBTPairedDataset(
        rgb_yaml=rgb_yaml,

        tir_yaml=tir_yaml,

        split=split,

        rgb_imgsz=rgb_imgsz,

        tir_imgsz=tir_imgsz,

        pair_mode=pair_mode,

        augment=False,

        tir_channels=3,

        strict_pair=True,
    )

    print(
        f"\nDataset size = {len(dataset)}"
    )

    # ========================================================
    # Single samples
    # ========================================================

    n = min(
        samples,
        len(dataset),
    )

    for i in range(n):

        sample = dataset[i]

        print(
            "\n"
            "------------------------------------------------------------"
        )

        print(
            f"Index      : {i}"
        )

        print(
            f"Pair key   : {sample['pair_key']}"
        )

        print(
            f"RGB        : {sample['rgb_path']}"
        )

        print(
            f"TIR        : {sample['tir_path']}"
        )

        print(
            f"RGB label  : {sample['rgb_label_path']}"
        )

        print(
            f"TIR label  : {sample['tir_label_path']}"
        )

        print(
            f"RGB native : {sample['rgb_ori_shape']}"
        )

        print(
            f"TIR native : {sample['tir_ori_shape']}"
        )

        print(
            f"RGB tensor : "
            f"{tuple(sample['rgb_img'].shape)}"
        )

        print(
            f"TIR tensor : "
            f"{tuple(sample['tir_img'].shape)}"
        )

        print(
            f"RGB boxes  : "
            f"{tuple(sample['rgb_bboxes'].shape)}"
        )

        print(
            f"TIR boxes  : "
            f"{tuple(sample['tir_bboxes'].shape)}"
        )

        print(
            f"RGB classes: "
            f"{tuple(sample['rgb_cls'].shape)}"
        )

        print(
            f"TIR classes: "
            f"{tuple(sample['tir_cls'].shape)}"
        )

    # ========================================================
    # DataLoader batch test
    # ========================================================

    loader = build_rgbt_dataloader(
        dataset,

        batch_size=batch_size,

        workers=workers,

        shuffle=False,
    )

    batch = next(
        iter(loader)
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "BATCH TEST"
    )

    print(
        "============================================================"
    )

    print(
        "rgb_img       :",
        tuple(
            batch[
                "rgb_img"
            ].shape
        ),
    )

    print(
        "tir_img       :",
        tuple(
            batch[
                "tir_img"
            ].shape
        ),
    )

    print(
        "rgb_cls       :",
        tuple(
            batch[
                "rgb_cls"
            ].shape
        ),
    )

    print(
        "rgb_bboxes    :",
        tuple(
            batch[
                "rgb_bboxes"
            ].shape
        ),
    )

    print(
        "rgb_batch_idx :",
        tuple(
            batch[
                "rgb_batch_idx"
            ].shape
        ),
    )

    print(
        "tir_cls       :",
        tuple(
            batch[
                "tir_cls"
            ].shape
        ),
    )

    print(
        "tir_bboxes    :",
        tuple(
            batch[
                "tir_bboxes"
            ].shape
        ),
    )

    print(
        "tir_batch_idx :",
        tuple(
            batch[
                "tir_batch_idx"
            ].shape
        ),
    )

    # ========================================================
    # Sanity assertions
    # ========================================================

    assert (
        batch[
            "rgb_img"
        ].ndim
        == 4
    )

    assert (
        batch[
            "tir_img"
        ].ndim
        == 4
    )

    assert (
        batch[
            "rgb_img"
        ].shape[1]
        == 3
    )

    assert (
        batch[
            "tir_img"
        ].shape[1]
        in {1, 3}
    )

    assert (
        batch[
            "rgb_bboxes"
        ].shape[1]
        == 4
    )

    assert (
        batch[
            "tir_bboxes"
        ].shape[1]
        == 4
    )

    print(
        "\n"
        "############################################################"
    )

    print(
        "[OK] RGB-T Dataset test passed."
    )

    print(
        "############################################################\n"
    )


# ============================================================
# 12. CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rgb",
        type=str,
        default=(
            "/mnt/sda/taochangyong/Projects/"
            "Model/YOLO-tcy/configs/datasets/"
            "LRDD_v3-RGB.yaml"
        ),
    )

    parser.add_argument(
        "--tir",
        type=str,
        default=(
            "/mnt/sda/taochangyong/Projects/"
            "Model/YOLO-tcy/configs/datasets/"
            "LRDD_v3-TIR.yaml"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--rgb-imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--tir-imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--pair-mode",
        type=str,
        default="relative",
        choices=[
            "relative",
            "stem",
            "filename",
        ],
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    self_test(
        rgb_yaml=args.rgb,

        tir_yaml=args.tir,

        split=args.split,

        rgb_imgsz=args.rgb_imgsz,

        tir_imgsz=args.tir_imgsz,

        pair_mode=args.pair_mode,

        batch_size=args.batch,

        workers=args.workers,

        samples=args.samples,
    )
