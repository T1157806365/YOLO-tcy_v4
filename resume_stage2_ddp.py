#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from engine.evaluator import validate_metrics
from engine.runtime import (
    broadcast_metrics,
    cleanup_distributed,
    distributed_barrier,
    is_main_process,
    load_config,
    make_grad_scaler,
    parse_device_ids,
    set_seed,
    setup_distributed,
)
from engine.trainer import (
    _build_data,
    _resolve_batch_size,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    train_one_epoch,
    unwrap_model,
)
from models.gtga_bhlr_factory import build_model_from_config


def main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def _is_torchrun_child():
    return int(os.environ.get('WORLD_SIZE', '1')) > 1 or 'LOCAL_RANK' in os.environ


def _auto_launch_ddp(cfg_path: str, resume_path: str, cfg: dict):
    if _is_torchrun_child():
        return False
    ids = parse_device_ids((cfg.get('train', {}) or {}).get('device', '0'))
    if len(ids) <= 1:
        return False
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in ids)
    env.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    env.setdefault('TORCH_NCCL_TRACE_BUFFER_SIZE', '1048576')
    env.setdefault('NCCL_DEBUG', 'WARN')
    cmd = [
        sys.executable, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc_per_node={len(ids)}', str(Path(__file__).resolve()),
        '--cfg', str(Path(cfg_path).expanduser().resolve()),
        '--resume', str(Path(resume_path).expanduser().resolve()),
    ]
    print('Auto-launch DDP:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)
    return True


def _load_history(path: Path):
    if not path.exists():
        return []
    try:
        x = json.loads(path.read_text(encoding='utf-8'))
        return x if isinstance(x, list) else []
    except Exception:
        return []


def resume_stage2(cfg: dict, resume_path: Path):
    train_cfg = cfg.get('train', {}) or {}
    stage2_cfg = train_cfg.get('stage2', {}) or {}
    output_cfg = cfg.get('output', {}) or {}
    model_cfg = cfg.get('model', {}) or {}
    data_cfg = cfg.get('data', {}) or {}

    device, rank, local_rank, world_size = setup_distributed(train_cfg.get('device', '0'))
    distributed = world_size > 1
    seed = int(train_cfg.get('seed', 0))
    deterministic = bool(train_cfg.get('deterministic', True))
    set_seed(seed, deterministic=deterministic)

    amp = bool(train_cfg.get('amp', True))
    grad_clip = float(train_cfg.get('grad_clip', 10.0))
    local_batch, global_batch, _ = _resolve_batch_size(cfg, world_size)

    out_dir = Path(output_cfg.get('project', 'runs/gtga_bhlr_v2')) / str(output_cfg.get('name', 'gtga_bhlr_v2_exp'))
    weights_dir = out_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    _, train_loader = _build_data(
        cfg, split='train', augment=True, shuffle=True,
        batch_size=local_batch, distributed=distributed,
        rank=rank, world_size=world_size,
    )
    if is_main_process():
        _, val_loader = _build_data(
            cfg, split='val', augment=False, shuffle=False,
            batch_size=int(train_cfg.get('val_batch', local_batch)),
            distributed=False, rank=0, world_size=1,
        )
    else:
        val_loader = None

    model = build_model_from_config(
        cfg,
        pretrained=bool(model_cfg.get('pretrained', True)),
        verbose=is_main_process(),
    ).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[int(device.index)],
            output_device=int(device.index),
            broadcast_buffers=True,
            find_unused_parameters=bool(train_cfg.get('ddp_find_unused_parameters', False)),
        )

    epochs = int(stage2_cfg.get('epochs', 200))
    lr0 = float(stage2_cfg.get('lr0', 0.001))
    lrf = float(stage2_cfg.get('lrf', 0.01))
    patience = int(stage2_cfg.get('patience', 50))
    min_delta = float(stage2_cfg.get('early_stop_min_delta', 1e-6))

    optimizer = build_optimizer(model, cfg, lr0=lr0)
    scheduler = build_scheduler(optimizer, epochs=epochs, lrf=lrf)
    scaler = make_grad_scaler(device, enabled=amp)

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    if ckpt.get('stage') != 'stage2':
        raise ValueError(f"Expected stage2 checkpoint, got {ckpt.get('stage')!r}")

    unwrap_model(model).load_state_dict(ckpt['model_state_dict'], strict=True)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])

    start_epoch = int(ckpt['stage_epoch']) + 1
    stage_best = float(ckpt.get('best_map5095', -1.0))
    early_stop_counter = int(ckpt.get('early_stop_counter', 0))
    global_epoch_offset = int(ckpt.get('global_epoch', start_epoch - 1)) - int(ckpt['stage_epoch'])
    global_best = stage_best

    generic_best_path = weights_dir / 'best.pt'
    if generic_best_path.exists():
        try:
            b = torch.load(generic_best_path, map_location='cpu', weights_only=False)
            global_best = max(global_best, float(b.get('best_map5095', global_best)))
        except Exception:
            pass

    history_path = out_dir / 'history_stage2.json'
    history = _load_history(history_path) if is_main_process() else []
    set_seed(seed + rank, deterministic=deterministic)

    main_print('\n============================================================')
    main_print('Resume stage2')
    main_print('============================================================')
    main_print(f'checkpoint     : {resume_path}')
    main_print(f'next epoch     : {start_epoch + 1}/{epochs}')
    main_print(f'stage best     : {stage_best:.6f}')
    main_print(f'world size     : {world_size}')
    main_print(f'batch / GPU    : {local_batch}')
    main_print(f'global batch   : {global_batch}')
    main_print('============================================================\n')

    rgb_imgsz = int(data_cfg.get('rgb_imgsz', 1920))
    tir_imgsz = int(data_cfg.get('tir_imgsz', 640))
    stage_last_path = weights_dir / 'stage2_last.pt'
    stage_best_path = weights_dir / 'stage2_best.pt'
    generic_last_path = weights_dir / 'last.pt'

    for epoch in range(start_epoch, epochs):
        train_loss, running = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp,
            grad_clip=grad_clip,
            epoch=epoch,
            epochs=epochs,
            rgb_imgsz=rgb_imgsz,
            tir_imgsz=tir_imgsz,
        )

        distributed_barrier()
        metrics = None
        if is_main_process():
            metrics = validate_metrics(
                model=unwrap_model(model),
                loader=val_loader,
                device=device,
                amp=amp,
                conf_thres=0.001,
                iou_thres=0.7,
                max_det=300,
                preview_num=0,
            )

        # Explicit synchronization after rank-0-only validation.
        distributed_barrier()
        metrics = broadcast_metrics(metrics, device)

        current_map = float(metrics['map5095'])
        improved_stage = current_map > stage_best + min_delta
        improved_global = current_map > global_best + min_delta
        if improved_stage:
            stage_best = current_map
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if improved_global:
            global_best = current_map

        scheduler.step()
        global_epoch = int(global_epoch_offset + epoch)

        if is_main_process():
            save_checkpoint(
                stage_last_path, model, optimizer, scheduler, scaler, cfg,
                stage='stage2', stage_epoch=epoch, global_epoch=global_epoch,
                best_map5095=stage_best, early_stop_counter=early_stop_counter,
            )
            shutil.copy2(stage_last_path, generic_last_path)
            if improved_stage:
                shutil.copy2(stage_last_path, stage_best_path)
            if improved_global:
                shutil.copy2(stage_last_path, generic_best_path)

        distributed_barrier()

        row = {
            'stage': 'stage2',
            'epoch': epoch + 1,
            'global_epoch': global_epoch + 1,
            'train_loss': float(train_loss),
            'lr': float(optimizer.param_groups[0]['lr']),
            **{k: float(v) for k, v in running.items()},
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        if is_main_process():
            history.append(row)
            history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding='utf-8')

        main_print(
            f"\n[stage2] epoch {epoch+1}/{epochs} | train_loss={train_loss:.6f} | "
            f"P={metrics['precision']:.4f} | R={metrics['recall']:.4f} | "
            f"AP50={metrics['map50']:.4f} | AP75={metrics['map75']:.4f} | "
            f"AP50:95={metrics['map5095']:.4f} | stage_best={stage_best:.4f} | "
            f"global_best={global_best:.4f}\n"
        )

        if patience > 0 and early_stop_counter >= patience:
            main_print(f'[stage2] early stop after {early_stop_counter} non-improving epochs')
            break

    distributed_barrier()
    cleanup_distributed()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--resume', required=True)
    args = p.parse_args()
    cfg = load_config(args.cfg)
    resume_path = Path(args.resume).expanduser().resolve()
    if not resume_path.exists():
        raise FileNotFoundError(resume_path)
    if _auto_launch_ddp(args.cfg, str(resume_path), cfg):
        return
    resume_stage2(cfg, resume_path)


if __name__ == '__main__':
    main()
