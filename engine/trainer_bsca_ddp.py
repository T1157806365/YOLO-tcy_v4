from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset
from engine.evaluator import validate_metrics
from engine.runtime import (
    autocast_context,
    gpu_memory_gb,
    make_grad_scaler,
    preprocess_batch,
    select_device,
    set_seed,
)
from models.gtga_bhlr_factory import build_model_from_config


def _dist_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main(rank: int) -> bool:
    return int(rank) == 0


def _unwrap(model):
    return model.module if isinstance(model, DDP) else model


def _barrier():
    if _dist_active():
        dist.barrier()


def _reduce_mean_float(value: float, device: torch.device, world_size: int) -> float:
    if int(world_size) <= 1:
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(world_size)
    return float(t.item())


def _reduce_mean_dict(values: Dict[str, float], device: torch.device, world_size: int) -> Dict[str, float]:
    if int(world_size) <= 1 or not values:
        return dict(values)
    keys = sorted(values.keys())
    t = torch.tensor([float(values[k]) for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(world_size)
    return {k: float(v) for k, v in zip(keys, t.cpu().tolist())}


def _broadcast_metrics(metrics: Optional[Dict], device: torch.device, rank: int, world_size: int) -> Dict:
    keys = ("precision", "recall", "map50", "map75", "map5095")
    if int(world_size) <= 1:
        return dict(metrics or {})

    if _is_main(rank):
        values = [float((metrics or {}).get(k, 0.0)) for k in keys]
    else:
        values = [0.0] * len(keys)

    t = torch.tensor(values, device=device, dtype=torch.float64)
    dist.broadcast(t, src=0)
    out = {k: float(v) for k, v in zip(keys, t.cpu().tolist())}
    return out


def _init_runtime(train_cfg: Dict):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        if not _dist_active():
            dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = select_device(train_cfg.get("device", "0"))

    return device, rank, local_rank, world_size


def normalize_loss_items(loss_items):
    if isinstance(loss_items, dict):
        out = {}
        for key, value in loss_items.items():
            if torch.is_tensor(value):
                value = float(value.detach().mean().item())
            else:
                value = float(value)
            out[key] = value
        return out

    if torch.is_tensor(loss_items):
        values = loss_items.detach().flatten().cpu().tolist()
    elif isinstance(loss_items, (list, tuple)):
        values = [float(v.detach().item()) if torch.is_tensor(v) else float(v) for v in loss_items]
    else:
        return {"loss": float(loss_items)}

    if len(values) == 3:
        return {"box_loss": values[0], "cls_loss": values[1], "dfl_loss": values[2]}
    return {f"loss_{i}": v for i, v in enumerate(values)}


def update_running_losses(running: Dict, current: Dict, batch_i: int):
    for key, value in current.items():
        if key not in running:
            running[key] = float(value)
        else:
            running[key] = (running[key] * batch_i + float(value)) / (batch_i + 1)
    return running


def get_standard_losses(losses: Dict):
    def find(keys, default=0.0):
        for key in keys:
            if key in losses:
                return losses[key]
        return default

    return (
        find(("box_loss", "box")),
        find(("cls_loss", "cls")),
        find(("dfl_loss", "dfl", "l1_loss", "l1")),
    )


def build_optimizer(model, cfg: Dict, lr0: Optional[float] = None):
    train_cfg = cfg.get("train", {}) or {}
    name = str(train_cfg.get("optimizer", "SGD")).strip().lower()
    lr = float(train_cfg.get("lr0", 0.01) if lr0 is None else lr0)
    momentum = float(train_cfg.get("momentum", 0.937))
    weight_decay = float(train_cfg.get("weight_decay", 5e-4))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, epochs: int, lrf: float = 0.01):
    epochs = max(int(epochs), 1)
    lrf = float(lrf)

    def lf(epoch):
        return ((1.0 - math.cos(math.pi * epoch / epochs)) / 2.0 * (lrf - 1.0) + 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


def save_checkpoint(path, model, optimizer, scheduler, scaler, cfg, stage, stage_epoch, global_epoch, best_map5095, early_stop_counter):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _unwrap(model)
    checkpoint = {
        "epoch": int(stage_epoch),
        "stage": str(stage),
        "stage_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_map5095": float(best_map5095),
        "early_stop_counter": int(early_stop_counter),
        "config": cfg,
        "model_name": raw.model_name,
        "fusion": raw.fusion_type,
        "nc": raw.nc,
        "names": raw.names,
    }
    torch.save(checkpoint, path)


def load_weights_strict(path, model, device):
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"{path} has no model_state_dict")
    _unwrap(model).load_state_dict(ckpt["model_state_dict"], strict=True)
    return ckpt


def _resolve_local_batch(train_cfg: Dict, world_size: int) -> int:
    batch = int(train_cfg.get("batch", 8))
    mode = str(train_cfg.get("batch_mode", "global")).strip().lower()
    if mode not in {"global", "per_device"}:
        raise ValueError("train.batch_mode must be 'global' or 'per_device'")

    if world_size <= 1 or mode == "per_device":
        return batch

    if batch % world_size != 0:
        raise ValueError(
            f"Global batch={batch} must be divisible by world_size={world_size}. "
            "Change train.batch or use batch_mode: per_device."
        )
    return batch // world_size


def _build_data(cfg: Dict, split: str, augment: bool, shuffle: bool, rank: int, world_size: int, distributed: bool):
    data_cfg = cfg.get("data", {}) or {}
    aug_cfg = cfg.get("augment", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    dataset = build_dual_resolution_dataset(
        rgb_yaml=data_cfg["rgb"],
        tir_yaml=data_cfg["tir"],
        split=split,
        rgb_high_imgsz=int(data_cfg.get("rgb_imgsz", 1920)),
        rgb_semantic_imgsz=int(data_cfg.get("rgb_semantic_imgsz", 640)),
        tir_imgsz=int(data_cfg.get("tir_imgsz", 640)),
        pair_mode=str(data_cfg.get("pair_mode", "relative")),
        augment=bool(augment),
        fliplr=float(aug_cfg.get("fliplr", 0.5)),
        flipud=float(aug_cfg.get("flipud", 0.0)),
        tir_channels=3,
        strict_pair=True,
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(shuffle),
            seed=int(train_cfg.get("seed", 0)),
            drop_last=False,
        )

    batch_size = _resolve_local_batch(train_cfg, world_size if distributed else 1)
    workers = int(train_cfg.get("workers", 4))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(bool(shuffle) and sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        persistent_workers=(workers > 0),
    )
    return dataset, loader, sampler, batch_size


def _write_history(path: Path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def train_one_epoch(model, loader, sampler, optimizer, scaler, device, amp, grad_clip, epoch, epochs, rgb_imgsz, tir_imgsz, rank, world_size):
    model.train()
    if sampler is not None:
        sampler.set_epoch(int(epoch))

    running_total = 0.0
    running_items = {}
    main = _is_main(rank)

    if main:
        print(
            f"{'Epoch':>11}{'GPU_mem':>11}{'box_loss':>11}{'cls_loss':>11}"
            f"{'dfl_loss':>11}{'Instances':>11}{'Size':>13}{'Progress':>18}"
        )
        iterator = tqdm(
            enumerate(loader), total=len(loader), dynamic_ncols=False, ncols=145,
            leave=False, mininterval=0.5, file=sys.stdout,
            bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    else:
        iterator = enumerate(loader)

    for batch_i, batch in iterator:
        batch = preprocess_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp):
            loss_raw, loss_items = model(batch)
            local_loss = loss_raw.sum()
            # Ultralytics detection loss is batch-sum scaled. DDP averages
            # gradients across ranks, so compensate to preserve the same
            # gradient scale as a single process using the global batch.
            backward_loss = local_loss * float(world_size) if world_size > 1 else local_loss

        scaler.scale(backward_loss).backward()

        if float(grad_clip) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

        scaler.step(optimizer)
        scaler.update()

        running_total += float(local_loss.detach().item())
        current_items = normalize_loss_items(loss_items)
        running_items = update_running_losses(running_items, current_items, batch_i)

        if main:
            box_loss, cls_loss, dfl_loss = get_standard_losses(running_items)
            instances = int(batch["rgb_cls"].shape[0])
            size_str = str(rgb_imgsz) if int(rgb_imgsz) == int(tir_imgsz) else f"{rgb_imgsz}/{tir_imgsz}"
            iterator.set_description(
                f"{epoch + 1:>5}/{epochs:<5}{gpu_memory_gb(device):>10.3g}G"
                f"{box_loss:>11.4g}{cls_loss:>11.4g}{dfl_loss:>11.4g}"
                f"{instances:>11}{size_str:>13}", refresh=True,
            )

    if main:
        iterator.close()

    local_epoch_loss = running_total / max(len(loader), 1)
    mean_local_loss = _reduce_mean_float(local_epoch_loss, device, world_size)
    # Report global-batch-equivalent summed loss for comparability with the
    # previous single-process trainer.
    epoch_loss = mean_local_loss * float(world_size) if world_size > 1 else mean_local_loss
    running_items = _reduce_mean_dict(running_items, device, world_size)
    return epoch_loss, running_items


def run_stage(model, cfg, train_loader, train_sampler, val_loader, device, out_dir, stage, epochs, lr0, lrf, patience, min_delta, amp, grad_clip, rank, world_size, global_epoch_offset=0, global_best_map=-1.0, generic_best_path=None):
    optimizer = build_optimizer(model, cfg, lr0=lr0)
    scheduler = build_scheduler(optimizer, epochs=epochs, lrf=lrf)
    scaler = make_grad_scaler(device, enabled=amp)

    data_cfg = cfg.get("data", {}) or {}
    rgb_imgsz = int(data_cfg.get("rgb_imgsz", 1920))
    tir_imgsz = int(data_cfg.get("tir_imgsz", 640))

    stage_best = -1.0
    early_stop_counter = 0
    history = []

    stage_best_path = out_dir / "weights" / f"{stage}_best.pt"
    stage_last_path = out_dir / "weights" / f"{stage}_last.pt"
    generic_last_path = out_dir / "weights" / "last.pt"
    main = _is_main(rank)

    if main:
        print("\n============================================================")
        print(f"{stage} training")
        print("============================================================")
        print(f"epochs       : {epochs}")
        print(f"lr0          : {lr0}")
        print(f"lrf          : {lrf}")
        print(f"patience     : {patience}")
        print("best metric  : mAP50-95")
        print("============================================================\n")

    for epoch in range(int(epochs)):
        train_loss, running = train_one_epoch(
            model=model, loader=train_loader, sampler=train_sampler,
            optimizer=optimizer, scaler=scaler, device=device, amp=amp,
            grad_clip=grad_clip, epoch=epoch, epochs=epochs,
            rgb_imgsz=rgb_imgsz, tir_imgsz=tir_imgsz,
            rank=rank, world_size=world_size,
        )

        # All ranks finish training collectives before rank 0 validates alone.
        _barrier()

        metrics = None
        if main:
            raw = _unwrap(model)
            metrics = validate_metrics(
                model=raw,
                loader=val_loader,
                device=device,
                amp=amp,
                conf_thres=0.001,
                iou_thres=0.7,
                max_det=300,
                preview_num=0,
            )

        metrics = _broadcast_metrics(metrics, device, rank, world_size)
        current_map = float(metrics["map5095"])
        global_epoch = int(global_epoch_offset + epoch)

        improved_stage = current_map > stage_best + float(min_delta)
        improved_global = current_map > global_best_map + float(min_delta)

        if improved_stage:
            stage_best = current_map
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if improved_global:
            global_best_map = current_map

        scheduler.step()

        if main:
            save_checkpoint(
                stage_last_path, model, optimizer, scheduler, scaler, cfg,
                stage=stage, stage_epoch=epoch, global_epoch=global_epoch,
                best_map5095=stage_best, early_stop_counter=early_stop_counter,
            )
            shutil.copy2(stage_last_path, generic_last_path)
            if improved_stage:
                shutil.copy2(stage_last_path, stage_best_path)
            if improved_global and generic_best_path is not None:
                shutil.copy2(stage_last_path, generic_best_path)

            row = {
                "stage": stage,
                "epoch": epoch + 1,
                "global_epoch": global_epoch + 1,
                "train_loss": float(train_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{k: float(v) for k, v in running.items()},
                **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            }
            history.append(row)
            _write_history(out_dir / f"history_{stage}.json", history)

            print(
                f"\n[{stage}] epoch {epoch+1}/{epochs} | train_loss={train_loss:.6f} | "
                f"P={metrics['precision']:.4f} | R={metrics['recall']:.4f} | "
                f"AP50={metrics['map50']:.4f} | AP75={metrics['map75']:.4f} | "
                f"AP50:95={metrics['map5095']:.4f} | stage_best={stage_best:.4f} | "
                f"global_best={global_best_map:.4f}\n"
            )

        # Checkpoint copy must finish before any rank starts the next epoch/stage.
        _barrier()

        if int(patience) > 0 and early_stop_counter >= int(patience):
            if main:
                print(f"[{stage}] early stop: no mAP50-95 improvement for {early_stop_counter} epochs.")
            break

    epochs_ran = len(history) if main else (epoch + 1 if 'epoch' in locals() else 0)
    # All ranks have identical early-stop state; stage/global best are identical
    # because validation metrics were broadcast every epoch.
    return {
        "stage_best_path": stage_best_path,
        "stage_last_path": stage_last_path,
        "stage_best_map": stage_best,
        "global_best_map": global_best_map,
        "epochs_ran": epochs_ran,
        "history": history if main else [],
    }


def train_experiment(cfg: Dict):
    train_cfg = cfg.get("train", {}) or {}
    output_cfg = cfg.get("output", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    resume = train_cfg.get("resume", None)
    if resume not in (None, "", "null"):
        raise NotImplementedError("Set train.resume: null for this clean BSCA DDP run.")

    device, rank, local_rank, world_size = _init_runtime(train_cfg)
    main = _is_main(rank)

    try:
        seed = int(train_cfg.get("seed", 0))
        deterministic = bool(train_cfg.get("deterministic", True))
        set_seed(seed + rank, deterministic=deterministic)

        amp = bool(train_cfg.get("amp", True))
        grad_clip = float(train_cfg.get("grad_clip", 10.0))

        out_dir = Path(output_cfg.get("project", "runs/bsca_bhlr_v2")) / str(
            output_cfg.get("name", "bsca_bhlr_v2_exp")
        )
        if main:
            (out_dir / "weights").mkdir(parents=True, exist_ok=True)
        _barrier()

        local_batch = _resolve_local_batch(train_cfg, world_size)
        if main:
            print("\n============================================================")
            print("BSCA + BHLR-v2 DDP trainer")
            print("============================================================")
            print(f"Model                : {model_cfg.get('name', 'yolo26s')}")
            print(f"Fusion               : {model_cfg.get('fusion', 'add')}")
            print(f"RGB high             : {data_cfg.get('rgb_imgsz', 1920)}")
            print(f"RGB semantic         : {data_cfg.get('rgb_semantic_imgsz', 640)}")
            print(f"TIR                  : {data_cfg.get('tir_imgsz', 640)}")
            print(f"World size           : {world_size}")
            print(f"Configured devices   : {train_cfg.get('device', '0')}")
            print(f"Batch mode           : {train_cfg.get('batch_mode', 'global')}")
            print(f"Configured batch     : {train_cfg.get('batch', 8)}")
            print(f"Local batch / GPU    : {local_batch}")
            print(f"Output               : {out_dir.resolve()}")
            print("============================================================\n")

        train_dataset, train_loader, train_sampler, _ = _build_data(
            cfg, split="train", augment=True, shuffle=True,
            rank=rank, world_size=world_size, distributed=(world_size > 1),
        )

        val_dataset = None
        val_loader = None
        if main:
            val_dataset, val_loader, _, _ = _build_data(
                cfg, split="val", augment=False, shuffle=False,
                rank=0, world_size=1, distributed=False,
            )
            print(f"train samples: {len(train_dataset)}")
            print(f"val samples  : {len(val_dataset)}")
        _barrier()

        raw_model = build_model_from_config(
            cfg,
            pretrained=bool(model_cfg.get("pretrained", True)),
            verbose=main,
        ).to(device)

        if world_size > 1:
            model = DDP(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            model = raw_model

        generic_best_path = out_dir / "weights" / "best.pt"

        stage1 = run_stage(
            model=model, cfg=cfg, train_loader=train_loader, train_sampler=train_sampler,
            val_loader=val_loader, device=device, out_dir=out_dir, stage="stage1",
            epochs=int(train_cfg.get("epochs", 300)), lr0=float(train_cfg.get("lr0", 0.01)),
            lrf=float(train_cfg.get("lrf", 0.01)), patience=int(train_cfg.get("patience", 50)),
            min_delta=float(train_cfg.get("early_stop_min_delta", 1e-6)), amp=amp,
            grad_clip=grad_clip, rank=rank, world_size=world_size,
            global_epoch_offset=0, global_best_map=-1.0,
            generic_best_path=generic_best_path,
        )

        global_best = float(stage1["global_best_map"])
        global_offset = int(stage1["epochs_ran"])
        # Non-main ranks do not keep history; derive offset by broadcasting main's value.
        if world_size > 1:
            t = torch.tensor([global_offset if main else 0], device=device, dtype=torch.long)
            dist.broadcast(t, src=0)
            global_offset = int(t.item())

        stage2_cfg = train_cfg.get("stage2", {}) or {}
        stage2_result = None

        if bool(stage2_cfg.get("enabled", True)):
            _barrier()
            load_weights_strict(stage1["stage_best_path"], model, device)
            _barrier()

            stage2_result = run_stage(
                model=model, cfg=cfg, train_loader=train_loader, train_sampler=train_sampler,
                val_loader=val_loader, device=device, out_dir=out_dir, stage="stage2",
                epochs=int(stage2_cfg.get("epochs", 200)), lr0=float(stage2_cfg.get("lr0", 0.001)),
                lrf=float(stage2_cfg.get("lrf", 0.01)), patience=int(stage2_cfg.get("patience", 50)),
                min_delta=float(stage2_cfg.get("early_stop_min_delta", 1e-6)), amp=amp,
                grad_clip=grad_clip, rank=rank, world_size=world_size,
                global_epoch_offset=global_offset, global_best_map=global_best,
                generic_best_path=generic_best_path,
            )
            global_best = float(stage2_result["global_best_map"])

        if main:
            summary = {
                "output_dir": str(out_dir.resolve()),
                "best_checkpoint": str(generic_best_path.resolve()),
                "global_best_map5095": global_best,
                "stage1_best_map5095": float(stage1["stage_best_map"]),
                "stage2_best_map5095": None if stage2_result is None else float(stage2_result["stage_best_map"]),
                "world_size": int(world_size),
                "batch_mode": str(train_cfg.get("batch_mode", "global")),
                "global_batch": int(train_cfg.get("batch", 8)) if str(train_cfg.get("batch_mode", "global")).lower() == "global" else int(train_cfg.get("batch", 8)) * int(world_size),
            }
            (out_dir / "training_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("\n============================================================")
            print("Training finished")
            print("============================================================")
            for key, value in summary.items():
                print(f"{key:<24}: {value}")
            print("============================================================\n")
            return summary
        return None
    finally:
        if _dist_active():
            dist.destroy_process_group()
