"""
RGB-T Dual-Backbone YOLO Model
==============================

Project:
    /mnt/sda/taochangyong/Projects/Model/YOLO-tcy

Architecture
------------

RGB image
    │
    ↓
YOLO Backbone-R
    │
    ├── P3_R
    ├── P4_R
    └── P5_R
          │
          │
          ├───────────────┐
                          │
TIR image                 │
    │                     │
    ↓                     │
YOLO Backbone-T           │
    │                     │
    ├── P3_T ─────────────┤
    ├── P4_T ─────────────┤
    └── P5_T ─────────────┘
                          │
                          ↓
                Cross-modal Fusion
                  Align + Concat
                          │
                          ↓
                     1×1 Conv
                          │
                          ↓
                  YOLO Neck + Head
                          │
                          ↓
                       Detect


Main design
-----------
python models/rgbt_model.py \
    --model yolo26n \
    --nc 1 \
    --fusion concat \
    --rgb-imgsz 1280 \
    --tir-imgsz 640 \
    --device 3
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from utils.config import resolve_model

from models.modules.adaptive_rgbt_fusion import (
    AdaptiveGatedFusion,
)
# ============================================================
# 1. Basic Conv block
# ============================================================

class ConvBNAct(nn.Module):
    """
    Simple:
        Conv2d -> BatchNorm -> SiLU

    Used after RGB-T feature concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.act = nn.SiLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.act(
            self.bn(
                self.conv(x)
            )
        )


# ============================================================
# 2. Feature alignment
# ============================================================

class FeatureAlign(nn.Module):
    """
    Spatially align TIR features to RGB features.

    Important:
        Alignment happens in FEATURE space,
        NOT image space.

    Example:

        RGB P3:
            [B, C, 160, 160]

        TIR P3:
            [B, C, 80, 80]

        ↓

        TIR aligned:
            [B, C, 160, 160]
    """

    def __init__(
        self,
        mode: str = "bilinear",
    ):
        super().__init__()

        self.mode = mode

    def forward(
        self,
        source: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        source:
            TIR feature.

        reference:
            RGB feature.
        """

        target_size = (
            reference.shape[-2:]
        )

        if (
            source.shape[-2:]
            == target_size
        ):
            return source

        if self.mode in {
            "nearest",
            "area",
        }:

            return F.interpolate(
                source,
                size=target_size,
                mode=self.mode,
            )

        return F.interpolate(
            source,
            size=target_size,
            mode=self.mode,
            align_corners=False,
        )


# ============================================================
# 3. Concat Fusion
# ============================================================

class ConcatFusion(nn.Module):
    """
    Baseline RGB-T feature fusion.

        RGB feature ──────────────┐
                                 │
        TIR feature -> Align ─────┤
                                 ↓
                              Concat
                                 ↓
                            1×1 Conv
                                 ↓
                         Fused feature

    Output channels are kept equal to RGB channels.

    Therefore the original pretrained YOLO neck can continue
    receiving the same number of channels.
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        out_channels: Optional[int] = None,
        align_mode: str = "bilinear",
    ):
        super().__init__()

        if out_channels is None:
            out_channels = rgb_channels

        self.rgb_channels = rgb_channels
        self.tir_channels = tir_channels
        self.out_channels = out_channels

        self.align = FeatureAlign(
            mode=align_mode
        )

        self.fuse = ConvBNAct(
            in_channels=(
                rgb_channels
                + tir_channels
            ),
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
    ) -> torch.Tensor:

        tir = self.align(
            tir,
            rgb,
        )

        x = torch.cat(
            [
                rgb,
                tir,
            ],
            dim=1,
        )

        return self.fuse(x)


# ============================================================
# 4. Add Fusion
# ============================================================

class AddFusion(nn.Module):
    """
    Simple addition baseline.

        RGB + aligned TIR
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        align_mode: str = "bilinear",
    ):
        super().__init__()

        self.align = FeatureAlign(
            mode=align_mode
        )

        if (
            rgb_channels
            != tir_channels
        ):

            self.tir_proj = ConvBNAct(
                tir_channels,
                rgb_channels,
                kernel_size=1,
            )

        else:

            self.tir_proj = nn.Identity()

    def forward(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
    ) -> torch.Tensor:

        tir = self.align(
            tir,
            rgb,
        )

        tir = self.tir_proj(
            tir
        )

        return rgb + tir


# ============================================================
# 5. Weighted Fusion
# ============================================================

class WeightedFusion(nn.Module):
    """
    Learnable weighted fusion.

        F = alpha * RGB
          + (1-alpha) * TIR

    alpha is learned.
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        align_mode: str = "bilinear",
    ):
        super().__init__()

        self.align = FeatureAlign(
            mode=align_mode
        )

        if (
            rgb_channels
            != tir_channels
        ):

            self.tir_proj = ConvBNAct(
                tir_channels,
                rgb_channels,
                kernel_size=1,
            )

        else:

            self.tir_proj = nn.Identity()

        # sigmoid(0) = 0.5
        self.alpha = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
    ) -> torch.Tensor:

        tir = self.align(
            tir,
            rgb,
        )

        tir = self.tir_proj(
            tir
        )

        alpha = torch.sigmoid(
            self.alpha
        )

        return (
            alpha * rgb
            + (1.0 - alpha) * tir
        )


# ============================================================
# 6. Fusion factory
# ============================================================

def build_fusion_module(
    fusion: str,
    rgb_channels: int,
    tir_channels: int,
    align_mode: str,
) -> nn.Module:

    fusion = fusion.lower()

    if fusion == "concat":

        return ConcatFusion(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            out_channels=rgb_channels,
            align_mode=align_mode,
        )

    if fusion == "add":

        return AddFusion(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            align_mode=align_mode,
        )

    if fusion == "weighted":

        return WeightedFusion(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            align_mode=align_mode,
        )
    
    if fusion == "agf":

        return AdaptiveGatedFusion(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            out_channels=rgb_channels,
            align_mode=align_mode,
        )
    
    if fusion == "hra":
        return HRAFusion(
            rgb_channels=rgb_channels,

            tir_channels=tir_channels,

            out_channels=rgb_channels,

            align_mode=align_mode,

            max_offset=0.10,
        )
    
    raise ValueError(
        "\n不支持 fusion:\n"
        f"{fusion}\n\n"
        "当前支持:\n"
        "  concat\n"
        "  add\n"
        "  weighted\n"
    )


# ============================================================
# 7. Ultralytics layer execution helper
# ============================================================

def get_layer_input(
    module: nn.Module,
    x,
    outputs: List,
):
    """
    Reproduce Ultralytics 'from' connection behaviour.

    module.f may be:

        -1

        6

        [-1, 6]

        [16, 19, 22]

    This is required because YOLO neck/head is not simply
    a linear nn.Sequential pipeline.
    """

    f = module.f

    # Previous layer
    if f == -1:
        return x

    # One earlier layer
    if isinstance(
        f,
        int,
    ):
        return outputs[f]

    # Multiple earlier layers
    return [
        (
            x
            if j == -1
            else outputs[j]
        )
        for j in f
    ]


# ============================================================
# 8. Discover backbone features used by YOLO neck
# ============================================================

def discover_fusion_indices(
    model: DetectionModel,
) -> List[int]:
    """
    Automatically discover backbone outputs referenced
    by YOLO head.

    For standard YOLO11 / YOLO26 this yields:

        [4, 6, 10]

    which correspond to:

        P3
        P4
        P5

    But we do NOT hard-code these indices.
    """

    backbone_len = len(
        model.yaml[
            "backbone"
        ]
    )

    indices = set()

    # --------------------------------------------------------
    # Last backbone output is directly fed into first
    # neck layer, even if head does not explicitly use
    # a positive reference to it at that first step.
    # --------------------------------------------------------

    indices.add(
        backbone_len - 1
    )

    # --------------------------------------------------------
    # Find all head references to backbone layers
    # --------------------------------------------------------

    for module in list(
        model.model
    )[backbone_len:]:

        f = module.f

        if isinstance(
            f,
            int,
        ):

            refs = [f]

        else:

            refs = list(f)

        for ref in refs:

            if (
                isinstance(ref, int)
                and 0 <= ref < backbone_len
            ):

                indices.add(
                    ref
                )

    return sorted(
        indices
    )


# ============================================================
# 9. RGB-T Detection Model
# ============================================================

class RGBTDetectionModel(nn.Module):
    """
    Dual-backbone YOLO RGB-T detector.

    Main branch:
        RGB backbone
        +
        fused features
        +
        original YOLO neck/head

    Second branch:
        independent TIR backbone

    Both branches may start from the same pretrained YOLO
    checkpoint but their weights are independent after model
    construction.
    """

    def __init__(
        self,
        model_name: str = "yolo26n",
        nc: int = 1,

        pretrained: bool = True,

        fusion: str = "concat",

        fusion_indices: Optional[
            Sequence[int]
        ] = None,

        align_mode: str = "bilinear",

        names: Optional[
            Union[
                Dict[int, str],
                Sequence[str],
            ]
        ] = None,

        verbose: bool = True,
    ):
        super().__init__()

        self.model_name = (
            model_name.lower()
        )

        self.nc = int(
            nc
        )

        self.pretrained = bool(
            pretrained
        )

        self.fusion_type = (
            fusion.lower()
        )

        self.align_mode = (
            align_mode
        )

        # ====================================================
        # Resolve model YAML + pretrained PT
        # ====================================================

        model_info = resolve_model(
            self.model_name,
            pretrained=pretrained,
        )

        self.model_yaml_path = Path(
            model_info[
                "yaml"
            ]
        )

        self.pretrained_path = (
            Path(
                model_info[
                    "weights"
                ]
            )
            if pretrained
            else None
        )

        if verbose:

            print(
                "\n"
                "============================================================"
            )

            print(
                "Build RGB-T Dual-Backbone YOLO"
            )

            print(
                "============================================================"
            )

            print(
                f"Model       : {self.model_name}"
            )

            print(
                f"Model YAML  : {self.model_yaml_path}"
            )

            print(
                f"Pretrained  : {self.pretrained_path}"
            )

            print(
                f"Classes     : {self.nc}"
            )

            print(
                f"Fusion      : {self.fusion_type}"
            )

            print(
                f"Align       : {self.align_mode}"
            )

        # ====================================================
        # Build TWO independent YOLO models
        #
        # We later keep:
        #
        # RGB:
        #   backbone + neck + detect
        #
        # TIR:
        #   backbone only
        # ====================================================

        rgb_base = DetectionModel(
            cfg=str(
                self.model_yaml_path
            ),
            ch=3,
            nc=self.nc,
            verbose=False,
        )

        tir_base = DetectionModel(
            cfg=str(
                self.model_yaml_path
            ),
            ch=3,
            nc=self.nc,
            verbose=False,
        )

        # ====================================================
        # Load pretrained weights
        # ====================================================

        if pretrained:

            if not self.pretrained_path.exists():

                raise FileNotFoundError(
                    "\n预训练模型不存在:\n"
                    f"{self.pretrained_path}"
                )

            # -----------------------------------------------
            # IMPORTANT:
            # local .pt only
            #
            # No automatic GitHub download here because the
            # path must already exist.
            # -----------------------------------------------

            pretrained_yolo = YOLO(
                str(
                    self.pretrained_path
                )
            )

            pretrained_model = (
                pretrained_yolo.model
            )

            # -----------------------------------------------
            # Load same official pretrained knowledge into
            # both branches.
            #
            # After this point parameters are independent.
            # -----------------------------------------------

            rgb_base.load(
                pretrained_model,
                verbose=verbose,
            )

            tir_base.load(
                pretrained_model,
                verbose=verbose,
            )

            del pretrained_yolo
            del pretrained_model

        # ====================================================
        # Backbone boundary
        # ====================================================

        self.backbone_len = len(
            rgb_base.yaml[
                "backbone"
            ]
        )

        # ====================================================
        # Main model
        #
        # Keep the complete RGB YOLO:
        #
        # self.model[0 : backbone_len]
        #       RGB backbone
        #
        # self.model[backbone_len :]
        #       YOLO neck + detect
        #
        # This also preserves:
        #
        # self.model[-1] = Detect
        #
        # which is useful for Ultralytics loss.
        # ====================================================

        self.model = (
            rgb_base.model
        )

        # ====================================================
        # Second independent TIR backbone
        # ====================================================

        self.tir_backbone = (
            nn.ModuleList(
                list(
                    tir_base.model[
                        :self.backbone_len
                    ]
                )
            )
        )

        # ====================================================
        # YOLO metadata
        # ====================================================

        self.yaml = deepcopy(
            rgb_base.yaml
        )

        self.save = deepcopy(
            rgb_base.save
        )

        self.stride = (
            rgb_base.stride.clone()
        )

        self.task = "detect"

        self.inplace = getattr(
            rgb_base,
            "inplace",
            True,
        )

        self.end2end = getattr(
            rgb_base,
            "end2end",
            getattr(
                self.model[-1],
                "end2end",
                False,
            ),
        )

        # ====================================================
        # Class names
        # ====================================================

        if names is None:

            self.names = {
                i: str(i)
                for i in range(
                    self.nc
                )
            }

        elif isinstance(
            names,
            dict,
        ):

            self.names = names

        else:

            self.names = {
                i: name
                for i, name
                in enumerate(names)
            }

        # ====================================================
        # Hyperparameters expected by Ultralytics loss
        #
        # train_rgbt.py will later overwrite these with the
        # actual experiment/training config.
        # ====================================================

        self.args = SimpleNamespace(
            **deepcopy(
                DEFAULT_CFG_DICT
            )
        )

        # ====================================================
        # Discover fusion points
        # ====================================================

        if fusion_indices is None:

            self.fusion_indices = (
                discover_fusion_indices(
                    rgb_base
                )
            )

        else:

            self.fusion_indices = sorted(
                [
                    int(i)
                    for i
                    in fusion_indices
                ]
            )

        # Validate
        for index in self.fusion_indices:

            if not (
                0
                <= index
                < self.backbone_len
            ):

                raise ValueError(
                    "\nFusion index 超出 backbone 范围:\n"
                    f"{index}\n"
                    f"backbone_len={self.backbone_len}"
                )

        # ====================================================
        # Infer feature channels
        # ====================================================

        feature_channels = (
            self._infer_feature_channels()
        )

        # ====================================================
        # Build one fusion module for each scale
        # ====================================================

        self.fusions = nn.ModuleDict()

        for index in self.fusion_indices:

            rgb_channels = (
                feature_channels[
                    index
                ][
                    "rgb"
                ]
            )

            tir_channels = (
                feature_channels[
                    index
                ][
                    "tir"
                ]
            )

            self.fusions[
                str(index)
            ] = build_fusion_module(
                fusion=self.fusion_type,

                rgb_channels=(
                    rgb_channels
                ),

                tir_channels=(
                    tir_channels
                ),

                align_mode=(
                    self.align_mode
                ),
            )

        # ====================================================
        # Criterion will be initialized lazily
        # ====================================================

        self.criterion = None

        if verbose:

            print(
                f"Backbone len: {self.backbone_len}"
            )

            print(
                "Fusion indices:",
                self.fusion_indices,
            )

            for index in (
                self.fusion_indices
            ):

                print(
                    f"  layer {index}: "
                    f"RGB C="
                    f"{feature_channels[index]['rgb']}, "
                    f"TIR C="
                    f"{feature_channels[index]['tir']}"
                )

            print(
                f"YOLO stride : "
                f"{self.stride.tolist()}"
            )

            print(
                f"End2End     : "
                f"{self.end2end}"
            )

            print(
                "============================================================\n"
            )

        # Local base models no longer needed as wrappers.
        # Their modules have already been moved into this model.
        del rgb_base
        del tir_base

    # ========================================================
    # Run arbitrary YOLO layers
    # ========================================================

    @staticmethod
    def _run_layers(
        layers,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        List,
    ]:
        """
        Execute YOLO layers while respecting module.f.

        Returns:
            final x
            all intermediate outputs
        """

        outputs = []

        for module in layers:

            module_input = (
                get_layer_input(
                    module,
                    x,
                    outputs,
                )
            )

            x = module(
                module_input
            )

            outputs.append(
                x
            )

        return (
            x,
            outputs,
        )

    # ========================================================
    # Run RGB backbone
    # ========================================================

    def forward_rgb_backbone(
        self,
        rgb: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        RGB -> independent RGB backbone.
        """

        layers = list(
            self.model
        )[
            :self.backbone_len
        ]

        _, outputs = (
            self._run_layers(
                layers,
                rgb,
            )
        )

        return outputs

    # ========================================================
    # Run TIR backbone
    # ========================================================

    def forward_tir_backbone(
        self,
        tir: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        TIR -> independent TIR backbone.
        """

        _, outputs = (
            self._run_layers(
                self.tir_backbone,
                tir,
            )
        )

        return outputs

    # ========================================================
    # Infer channels
    # ========================================================

    def _infer_feature_channels(
        self,
    ) -> Dict[
        int,
        Dict[str, int],
    ]:
        """
        Infer feature channels using dummy tensors.

        This avoids hard-coding n/s/m channel dimensions.
        """

        # Save original training states
        rgb_training = (
            self.model.training
        )

        tir_training = (
            self.tir_backbone.training
        )

        self.model.eval()
        self.tir_backbone.eval()

        # 256 is sufficient for P3/P4/P5 shape inference
        rgb_dummy = torch.zeros(
            1,
            3,
            256,
            256,
        )

        tir_dummy = torch.zeros(
            1,
            3,
            256,
            256,
        )

        with torch.no_grad():

            rgb_outputs = (
                self.forward_rgb_backbone(
                    rgb_dummy
                )
            )

            tir_outputs = (
                self.forward_tir_backbone(
                    tir_dummy
                )
            )

        channels = {}

        for index in (
            self.fusion_indices
        ):

            rgb_feature = (
                rgb_outputs[
                    index
                ]
            )

            tir_feature = (
                tir_outputs[
                    index
                ]
            )

            if not isinstance(
                rgb_feature,
                torch.Tensor,
            ):

                raise TypeError(
                    f"RGB layer {index} "
                    "不是 Tensor 输出。"
                )

            if not isinstance(
                tir_feature,
                torch.Tensor,
            ):

                raise TypeError(
                    f"TIR layer {index} "
                    "不是 Tensor 输出。"
                )

            channels[index] = {
                "rgb":
                    int(
                        rgb_feature.shape[1]
                    ),

                "tir":
                    int(
                        tir_feature.shape[1]
                    ),
            }

        # Restore states
        self.model.train(
            rgb_training
        )

        self.tir_backbone.train(
            tir_training
        )

        return channels

    # ========================================================
    # Feature fusion
    # ========================================================

    def fuse_backbone_features(
        self,

        rgb_outputs: List[
            torch.Tensor
        ],

        tir_outputs: List[
            torch.Tensor
        ],

    ) -> Tuple[
        List,
        Dict[int, torch.Tensor],
    ]:
        """
        Replace selected RGB backbone outputs with
        fused RGB-T features.

        Important:
        TIR is spatially aligned to RGB feature maps.
        RGB therefore defines the fused coordinate/reference
        resolution.
        """

        fused_outputs = list(
            rgb_outputs
        )

        fusion_features = {}

        for index in (
            self.fusion_indices
        ):

            rgb_feature = (
                rgb_outputs[
                    index
                ]
            )

            tir_feature = (
                tir_outputs[
                    index
                ]
            )

            fused = self.fusions[
                str(index)
            ](
                rgb_feature,
                tir_feature,
            )

            fused_outputs[
                index
            ] = fused

            fusion_features[
                index
            ] = fused

        return (
            fused_outputs,
            fusion_features,
        )

    # ========================================================
    # YOLO Neck + Detect
    # ========================================================

    def forward_head(
        self,
        backbone_outputs: List,
    ):
        """
        Run original YOLO neck + detection head.

        backbone_outputs already contain fused P3/P4/P5.
        """

        if (
            len(backbone_outputs)
            != self.backbone_len
        ):

            raise ValueError(
                "\nbackbone_outputs 长度错误:\n"
                f"got={len(backbone_outputs)}\n"
                f"expected={self.backbone_len}"
            )

        # ----------------------------------------------------
        # outputs keeps the ORIGINAL YOLO global layer indexing
        # ----------------------------------------------------

        outputs = list(
            backbone_outputs
        )

        # The first neck layer receives final backbone feature
        x = outputs[
            self.backbone_len - 1
        ]

        head_layers = list(
            self.model
        )[
            self.backbone_len:
        ]

        for module in head_layers:

            module_input = (
                get_layer_input(
                    module,
                    x,
                    outputs,
                )
            )

            x = module(
                module_input
            )

            outputs.append(
                x
            )

        return x

    # ========================================================
    # Forward features
    # ========================================================

    def forward_features(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
    ) -> Dict:
        """
        Extract two independent backbones and fuse
        selected scales.
        """

        rgb_outputs = (
            self.forward_rgb_backbone(
                rgb
            )
        )

        tir_outputs = (
            self.forward_tir_backbone(
                tir
            )
        )

        (
            fused_outputs,
            fusion_features,
        ) = self.fuse_backbone_features(
            rgb_outputs,
            tir_outputs,
        )

        return {
            "rgb_backbone":
                rgb_outputs,

            "tir_backbone":
                tir_outputs,

            "fused_backbone":
                fused_outputs,

            "fusion_features":
                fusion_features,
        }

    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        return_features: bool = False,
    ):
        """
        RGB + TIR -> fused YOLO prediction.
        """

        features = (
            self.forward_features(
                rgb,
                tir,
            )
        )

        pred = self.forward_head(
            features[
                "fused_backbone"
            ]
        )

        if return_features:

            return {
                "pred":
                    pred,

                **features,
            }

        return pred

    # ========================================================
    # Standard forward
    # ========================================================

    def forward(
        self,
        rgb,
        tir=None,
        return_features: bool = False,
    ):
        """
        Supports:

        Inference:
            model(rgb, tir)

        Training batch:
            model(batch)

        where batch contains:

            rgb_img
            tir_img
            rgb_cls
            rgb_bboxes
            rgb_batch_idx
        """

        # ----------------------------------------------------
        # Training dict
        # ----------------------------------------------------

        if isinstance(
            rgb,
            dict,
        ):

            return self.loss(
                rgb
            )

        if tir is None:

            raise ValueError(
                "\nRGB-T 模型必须同时提供 rgb 和 tir。\n"
            )

        return self.predict(
            rgb,
            tir,
            return_features=(
                return_features
            ),
        )

    # ========================================================
    # Ultralytics loss
    # ========================================================

    def init_criterion(
        self,
    ):
        """
        Same detection-loss selection as current
        Ultralytics DetectionModel.

        YOLO26:
            end2end=True
            -> E2ELoss

        YOLO11:
            -> v8DetectionLoss
        """

        if getattr(
            self,
            "end2end",
            False,
        ):

            return E2ELoss(
                self
            )

        return v8DetectionLoss(
            self
        )

    # ========================================================
    # Main fused loss
    # ========================================================

    def loss(
        self,
        batch: Dict,
        preds=None,
    ):
        """
        Main fused detection loss.

        Current baseline:
            fused prediction is supervised using RGB labels.

        Why?
        ----
        Fused features are aligned to RGB feature coordinates.

        Therefore the final fused detection head belongs to
        RGB reference space.

        TIR labels remain available as:
            tir_cls
            tir_bboxes
            tir_batch_idx

        They will later support:
            auxiliary TIR supervision.
        """

        required = [
            "rgb_img",
            "tir_img",

            "rgb_cls",
            "rgb_bboxes",
            "rgb_batch_idx",
        ]

        for key in required:

            if key not in batch:

                raise KeyError(
                    f"RGB-T batch 缺少: {key}"
                )

        if self.criterion is None:

            self.criterion = (
                self.init_criterion()
            )

        if preds is None:

            preds = self.predict(
                batch[
                    "rgb_img"
                ],
                batch[
                    "tir_img"
                ],
            )

        # ----------------------------------------------------
        # Convert our RGBT batch to the field names expected
        # by Ultralytics detection loss.
        # ----------------------------------------------------

        yolo_batch = {
            "img":
                batch[
                    "rgb_img"
                ],

            "cls":
                batch[
                    "rgb_cls"
                ],

            "bboxes":
                batch[
                    "rgb_bboxes"
                ],

            "batch_idx":
                batch[
                    "rgb_batch_idx"
                ],
        }

        return self.criterion(
            preds,
            yolo_batch,
        )

    # ========================================================
    # Freeze helpers
    # ========================================================

    def freeze_rgb_backbone(
        self,
    ):
        """Freeze RGB backbone."""

        for module in list(
            self.model
        )[
            :self.backbone_len
        ]:

            for p in (
                module.parameters()
            ):

                p.requires_grad = False

    def freeze_tir_backbone(
        self,
    ):
        """Freeze TIR backbone."""

        for p in (
            self.tir_backbone
            .parameters()
        ):

            p.requires_grad = False

    def unfreeze_all(
        self,
    ):
        """Unfreeze entire model."""

        for p in (
            self.parameters()
        ):

            p.requires_grad = True

    # ========================================================
    # Parameter statistics
    # ========================================================

    def parameter_count(
        self,
    ) -> Dict[str, int]:

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            "total":
                total,

            "trainable":
                trainable,
        }

    # ========================================================
    # Print model information
    # ========================================================

    def print_info(
        self,
    ):

        count = self.parameter_count()

        print(
            "\n"
            "============================================================"
        )

        print(
            "RGB-T Model Information"
        )

        print(
            "============================================================"
        )

        print(
            f"YOLO model     : {self.model_name}"
        )

        print(
            f"Classes        : {self.nc}"
        )

        print(
            f"RGB Backbone   : "
            f"{self.backbone_len} blocks"
        )

        print(
            f"TIR Backbone   : "
            f"{len(self.tir_backbone)} blocks"
        )

        print(
            f"Fusion type    : "
            f"{self.fusion_type}"
        )

        print(
            f"Fusion layers  : "
            f"{self.fusion_indices}"
        )

        print(
            f"Stride         : "
            f"{self.stride.tolist()}"
        )

        print(
            f"End2End        : "
            f"{self.end2end}"
        )

        print(
            f"Parameters     : "
            f"{count['total']:,}"
        )

        print(
            f"Trainable      : "
            f"{count['trainable']:,}"
        )

        print(
            "============================================================\n"
        )


# ============================================================
# 10. Output-shape helper
# ============================================================

def print_output_structure(
    x,
    prefix: str = "",
):
    """
    Safely print YOLO11 / YOLO26 output structures.

    YOLO heads may return:
        Tensor
        list
        tuple
        dict
    """

    if isinstance(
        x,
        torch.Tensor,
    ):

        print(
            f"{prefix}Tensor "
            f"{tuple(x.shape)}"
        )

        return

    if isinstance(
        x,
        dict,
    ):

        print(
            f"{prefix}dict"
        )

        for key, value in x.items():

            print(
                f"{prefix}  [{key}]"
            )

            print_output_structure(
                value,
                prefix + "    ",
            )

        return

    if isinstance(
        x,
        (list, tuple),
    ):

        print(
            f"{prefix}"
            f"{type(x).__name__}"
            f"[{len(x)}]"
        )

        for i, value in enumerate(
            x
        ):

            print(
                f"{prefix}  [{i}]"
            )

            print_output_structure(
                value,
                prefix + "    ",
            )

        return

    print(
        f"{prefix}"
        f"{type(x).__name__}"
    )


# ============================================================
# 11. Self test
# ============================================================

def self_test(
    model_name: str,
    nc: int,
    fusion: str,
    rgb_imgsz: int,
    tir_imgsz: int,
    device: str,
):
    """
    Test dual-backbone forward only.

    No dataset and no training required.
    """

    print(
        "\n"
        "############################################################"
    )

    print(
        "RGB-T MODEL SELF TEST"
    )

    print(
        "############################################################\n"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if (
        device != "cpu"
        and torch.cuda.is_available()
    ):

        device_obj = torch.device(
            f"cuda:{device}"
        )

    else:

        device_obj = torch.device(
            "cpu"
        )

    print(
        f"Device: {device_obj}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = RGBTDetectionModel(
        model_name=model_name,

        nc=nc,

        pretrained=True,

        fusion=fusion,

        align_mode="bilinear",

        verbose=True,
    )

    model = model.to(
        device_obj
    )

    model.eval()

    model.print_info()

    # --------------------------------------------------------
    # RGB and TIR may use DIFFERENT resolutions
    # --------------------------------------------------------

    rgb = torch.randn(
        1,
        3,
        rgb_imgsz,
        rgb_imgsz,
        device=device_obj,
    )

    tir = torch.randn(
        1,
        3,
        tir_imgsz,
        tir_imgsz,
        device=device_obj,
    )

    print(
        "Input:"
    )

    print(
        "  RGB:",
        tuple(
            rgb.shape
        ),
    )

    print(
        "  TIR:",
        tuple(
            tir.shape
        ),
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            rgb,
            tir,
            return_features=True,
        )

    print(
        "\nFusion feature shapes:"
    )

    for index, feature in (
        outputs[
            "fusion_features"
        ].items()
    ):

        print(
            f"  layer {index}: "
            f"{tuple(feature.shape)}"
        )

    print(
        "\nPrediction structure:"
    )

    print_output_structure(
        outputs[
            "pred"
        ]
    )

    print(
        "\n"
        "############################################################"
    )

    print(
        "[OK] RGB-T model forward test passed."
    )

    print(
        "############################################################\n"
    )


# ============================================================
# 12. CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "RGB-T Dual-Backbone YOLO model test"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n",
    )

    parser.add_argument(
        "--nc",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--fusion",
        type=str,
        default="concat",
        choices=[
            "concat",
            "add",
            "weighted",
        ],
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
        "--device",
        type=str,
        default="0",
    )

    args = parser.parse_args()

    self_test(
        model_name=args.model,

        nc=args.nc,

        fusion=args.fusion,

        rgb_imgsz=args.rgb_imgsz,

        tir_imgsz=args.tir_imgsz,

        device=args.device,
    )

