#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from engine.runtime import load_config, parse_device_ids
from engine.trainer import train_experiment


def _is_torchrun_child() -> bool:
    """
    判断当前进程是否已经是 torchrun 创建的 DDP 子进程。
    """
    return (
        int(os.environ.get("WORLD_SIZE", "1")) > 1
        or "LOCAL_RANK" in os.environ
    )


def _auto_launch_ddp(cfg_path: str, cfg: dict) -> bool:
    """
    当 train.device 中配置多个 GPU 时，
    自动重新通过 torchrun 启动多卡 DDP。

    Example
    -------
    YAML:
        device: "0,2,3,4"

    自动变成:
        CUDA_VISIBLE_DEVICES=0,2,3,4
        torchrun --nproc_per_node=4 ...
    """

    # 已经处于 torchrun 子进程中，不能再次启动
    if _is_torchrun_child():
        return False

    train_cfg = cfg.get("train", {}) or {}

    device_spec = train_cfg.get(
        "device",
        "0",
    )

    device_ids = parse_device_ids(
        device_spec
    )

    # ---------------------------------------------------------
    # 单 GPU
    # ---------------------------------------------------------
    if len(device_ids) <= 1:
        return False

    # ---------------------------------------------------------
    # Multi-GPU DDP
    # ---------------------------------------------------------
    env = os.environ.copy()

    # 只向子进程暴露指定的物理 GPU
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(x)
        for x in device_ids
    )

    env["YOLO_TCY_AUTOLAUNCHED"] = "1"

    script = str(
        Path(__file__).resolve()
    )

    cfg_path = str(
        Path(cfg_path)
        .expanduser()
        .resolve()
    )

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(device_ids)}",
        script,
        "--cfg",
        cfg_path,
    ]

    print(
        "\n"
        "============================================================\n"
        "Auto-launching multi-GPU DDP\n"
        "============================================================"
    )

    print(
        f"Physical GPUs        : {device_ids}"
    )

    print(
        f"CUDA_VISIBLE_DEVICES : "
        f"{env['CUDA_VISIBLE_DEVICES']}"
    )

    print(
        f"World size           : {len(device_ids)}"
    )

    print(
        "Command              :",
        " ".join(cmd),
    )

    print(
        "============================================================\n"
    )

    subprocess.run(
        cmd,
        env=env,
        check=True,
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "RLSFA coarse + GTGA scale + "
            "guided BHLR-v2 training "
            "with single/Multi-GPU DDP support"
        )
    )

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.cfg
    )

    # ---------------------------------------------------------
    # 多 GPU:
    # 主进程自动通过 torchrun 重新启动
    # ---------------------------------------------------------
    if _auto_launch_ddp(
        args.cfg,
        cfg,
    ):
        return

    # ---------------------------------------------------------
    # 单 GPU 或 torchrun 子进程
    # ---------------------------------------------------------
    train_experiment(
        cfg
    )


if __name__ == "__main__":
    main()