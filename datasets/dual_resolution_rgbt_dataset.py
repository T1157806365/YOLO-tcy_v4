"""
Clean public name for the stable dual-resolution RGB-T dataset.

This module intentionally reuses the already-tested implementation in
``datasets/rsdt_dataset.py`` but removes any train_rsdt/train_rgbt dependency
from the new GTGA training/validation entry points.
"""

from __future__ import annotations

from datasets.rsdt_dataset import (
    RSDTPairedDataset,
    build_rsdt_dataset,
    build_rsdt_dataloader,
)


DualResolutionRGBTDataset = RSDTPairedDataset


def build_dual_resolution_dataset(*args, **kwargs):
    return build_rsdt_dataset(*args, **kwargs)


def build_dual_resolution_dataloader(*args, **kwargs):
    return build_rsdt_dataloader(*args, **kwargs)


__all__ = [
    'DualResolutionRGBTDataset',
    'build_dual_resolution_dataset',
    'build_dual_resolution_dataloader',
]
