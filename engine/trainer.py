from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,
    build_dual_resolution_dataloader,
)
from engine.evaluator import validate_metrics
from engine.runtime import (
    autocast_context,
    broadcast_metrics,
    cleanup_distributed,
    distributed_barrier,
    distributed_mean,
    distributed_mean_dict,
    gpu_memory_gb,
    is_main_process,
    make_grad_scaler,
    preprocess_batch,
    set_seed,
    setup_distributed,
)
from models.gtga_bhlr_factory import build_model_from_config


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


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
        values = [
            float(v.detach().item()) if torch.is_tensor(v) else float(v)
            for v in loss_items
        ]
    else:
        return {"loss": float(loss_items)}

    if len(values) == 3:
        return {
            "box_loss": values[0],
            "cls_loss": values[1],
            "dfl_loss": values[2],
        }

    return {f"loss_{i}": v for i, v in enumerate(values)}


def update_running_losses(running: Dict, current: Dict, batch_i: int):
    for key, value in current.items():
        if key not in running:
            running[key] = float(value)
        else:
            running[key] = (
                running[key] * batch_i + float(value)
            ) / (batch_i + 1)
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
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, epochs: int, lrf: float = 0.01):
    epochs = max(int(epochs), 1)
    lrf = float(lrf)

    def lf(epoch):
        return (
            (1.0 - math.cos(math.pi * epoch / epochs)) / 2.0
            * (lrf - 1.0)
            + 1.0
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    cfg: Dict,
    stage: str,
    stage_epoch: int,
    global_epoch: int,
    best_map5095: float,
    early_stop_counter: int,
):
    if not is_main_process():
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = unwrap_model(model)

    checkpoint = {
        "epoch": int(stage_epoch),
        "stage": str(stage),
        "stage_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_map5095": float(best_map5095),
        "early_stop_counter": int(early_stop_counter),
        "config": cfg,
        "model_name": raw_model.model_name,
        "fusion": raw_model.fusion_type,
        "nc": raw_model.nc,
        "names": raw_model.names,
    }
    torch.save(checkpoint, path)


def load_weights_strict(path: str | Path, model, device):
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"{path} has no model_state_dict")

    raw_model = unwrap_model(model)
    raw_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return ckpt


def _set_sampler_epoch(loader, epoch: int):
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(int(epoch))


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    amp: bool,
    grad_clip: float,
    epoch: int,
    epochs: int,
    rgb_imgsz: int,
    tir_imgsz: int,
):
    model.train()
    _set_sampler_epoch(loader, epoch)

    running_total = 0.0
    running_items = {}

    if is_main_process():
        print(
            f"{'Epoch':>11}"
            f"{'GPU_mem':>11}"
            f"{'box_loss':>11}"
            f"{'cls_loss':>11}"
            f"{'dfl_loss':>11}"
            f"{'Instances':>11}"
            f"{'Size':>13}"
            f"{'Progress':>18}"
        )

    iterator = enumerate(loader)
    if is_main_process():
        iterator = tqdm(
            iterator,
            total=len(loader),
            dynamic_ncols=False,
            ncols=145,
            leave=False,
            mininterval=0.5,
            file=sys.stdout,
            bar_format=(
                "{desc} "
                "{percentage:3.0f}%|{bar:10}| "
                "{n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

    for batch_i, batch in iterator:
        batch = preprocess_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp):
            loss_raw, loss_items = model(batch)
            loss = loss_raw.sum()

        scaler.scale(loss).backward()

        if float(grad_clip) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(grad_clip),
            )

        scaler.step(optimizer)
        scaler.update()

        running_total += float(loss.detach().item())
        current_items = normalize_loss_items(loss_items)
        running_items = update_running_losses(
            running_items,
            current_items,
            batch_i,
        )

        if is_main_process():
            box_loss, cls_loss, dfl_loss = get_standard_losses(running_items)
            instances = int(batch["rgb_cls"].shape[0])
            size_str = (
                str(rgb_imgsz)
                if int(rgb_imgsz) == int(tir_imgsz)
                else f"{rgb_imgsz}/{tir_imgsz}"
            )

            iterator.set_description(
                f"{epoch + 1:>5}/{epochs:<5}"
                f"{gpu_memory_gb(device):>10.3g}G"
                f"{box_loss:>11.4g}"
                f"{cls_loss:>11.4g}"
                f"{dfl_loss:>11.4g}"
                f"{instances:>11}"
                f"{size_str:>13}",
                refresh=True,
            )

    if is_main_process():
        iterator.close()

    epoch_loss = running_total / max(len(loader), 1)
    epoch_loss = distributed_mean(epoch_loss, device)
    running_items = distributed_mean_dict(running_items, device)

    return epoch_loss, running_items


def _resolve_batch_size(cfg: Dict, world_size: int):
    """
    Backward-compatible default:
        train.batch = PER-GPU batch size.

    Optional:
        train.batch_mode: global
        train.batch: 18

    With 3 GPUs that gives batch/GPU=6.
    """
    train_cfg = cfg.get("train", {}) or {}
    requested = int(train_cfg.get("batch", 8))
    mode = str(train_cfg.get("batch_mode", "per_gpu")).strip().lower()

    if mode in {"per_gpu", "per-gpu", "local"}:
        local_batch = requested
        global_batch = local_batch * int(world_size)
        mode = "per_gpu"
    elif mode in {"global", "total"}:
        if requested % int(world_size) != 0:
            raise ValueError(
                "train.batch_mode='global' requires train.batch to be divisible "
                f"by WORLD_SIZE. Got batch={requested}, world_size={world_size}."
            )
        global_batch = requested
        local_batch = requested // int(world_size)
        mode = "global"
    else:
        raise ValueError(
            "train.batch_mode must be 'per_gpu' or 'global', "
            f"got {mode!r}"
        )

    if local_batch < 1:
        raise ValueError(f"Invalid per-GPU batch: {local_batch}")

    return local_batch, global_batch, mode


def _build_data(
    cfg: Dict,
    split: str,
    augment: bool,
    shuffle: bool,
    *,
    batch_size: int,
    distributed: bool,
    rank: int,
    world_size: int,
):
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

    loader = build_dual_resolution_dataloader(
        dataset,
        batch_size=int(batch_size),
        workers=int(train_cfg.get("workers", 4)),
        shuffle=(bool(shuffle) if sampler is None else False),
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )

    return dataset, loader


def _write_history(path: Path, history):
    if not is_main_process():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_stage(
    model,
    cfg: Dict,
    train_loader,
    val_loader,
    device,
    out_dir: Path,
    stage: str,
    epochs: int,
    lr0: float,
    lrf: float,
    patience: int,
    min_delta: float,
    amp: bool,
    grad_clip: float,
    global_epoch_offset: int = 0,
    global_best_map: float = -1.0,
    generic_best_path: Optional[Path] = None,
):
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

    main_print("\n============================================================")
    main_print(f"{stage} training")
    main_print("============================================================")
    main_print(f"epochs       : {epochs}")
    main_print(f"lr0          : {lr0}")
    main_print(f"lrf          : {lrf}")
    main_print(f"patience     : {patience}")
    main_print("best metric  : mAP50-95")
    main_print("============================================================\n")

    for epoch in range(int(epochs)):
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
            if val_loader is None:
                raise RuntimeError("Rank 0 validation loader is missing.")

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

        metrics = broadcast_metrics(metrics, device)

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

        if is_main_process():
            save_checkpoint(
                stage_last_path,
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                stage=stage,
                stage_epoch=epoch,
                global_epoch=global_epoch,
                best_map5095=stage_best,
                early_stop_counter=early_stop_counter,
            )
            shutil.copy2(stage_last_path, generic_last_path)

            if improved_stage:
                shutil.copy2(stage_last_path, stage_best_path)

            if improved_global and generic_best_path is not None:
                shutil.copy2(stage_last_path, generic_best_path)

        distributed_barrier()

        row = {
            "stage": stage,
            "epoch": epoch + 1,
            "global_epoch": global_epoch + 1,
            "train_loss": float(train_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{k: float(v) for k, v in running.items()},
            **{
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            },
        }
        history.append(row)
        _write_history(out_dir / f"history_{stage}.json", history)

        main_print(
            f"\n[{stage}] epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f} | "
            f"AP50={metrics['map50']:.4f} | "
            f"AP75={metrics['map75']:.4f} | "
            f"AP50:95={metrics['map5095']:.4f} | "
            f"stage_best={stage_best:.4f} | "
            f"global_best={global_best_map:.4f}\n"
        )

        if int(patience) > 0 and early_stop_counter >= int(patience):
            main_print(
                f"[{stage}] early stop: no mAP50-95 improvement for "
                f"{early_stop_counter} epochs."
            )
            break

    return {
        "stage_best_path": stage_best_path,
        "stage_last_path": stage_last_path,
        "stage_best_map": stage_best,
        "global_best_map": global_best_map,
        "epochs_ran": len(history),
        "history": history,
    }


def train_experiment(cfg: Dict):
    train_cfg = cfg.get("train", {}) or {}
    output_cfg = cfg.get("output", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    resume = train_cfg.get("resume", None)
    if resume not in (None, "", "null"):
        raise NotImplementedError(
            "Clean trainer intentionally does not silently reuse the old resume path yet. "
            "Set train.resume: null."
        )

    device, rank, local_rank, world_size = setup_distributed(
        train_cfg.get("device", "0")
    )
    distributed = int(world_size) > 1

    seed = int(train_cfg.get("seed", 0))
    deterministic = bool(train_cfg.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    amp = bool(train_cfg.get("amp", True))
    grad_clip = float(train_cfg.get("grad_clip", 10.0))

    local_batch, global_batch, batch_mode = _resolve_batch_size(
        cfg,
        world_size,
    )

    out_dir = Path(
        output_cfg.get("project", "runs/gtga_bhlr_v2")
    ) / str(
        output_cfg.get("name", "gtga_bhlr_v2_exp")
    )

    if is_main_process():
        (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    distributed_barrier()

    main_print("\n============================================================")
    main_print("Clean GTGA + BHLR-v2 trainer | Single GPU / DDP")
    main_print("============================================================")
    main_print("Runtime dependencies : NO train_rsdt.py / train_rgbt.py")
    main_print(f"Model                : {model_cfg.get('name', 'yolo26s')}")
    main_print(f"Fusion               : {model_cfg.get('fusion', 'add')}")
    main_print(f"RGB high             : {data_cfg.get('rgb_imgsz', 1920)}")
    main_print(f"RGB semantic         : {data_cfg.get('rgb_semantic_imgsz', 640)}")
    main_print(f"TIR                  : {data_cfg.get('tir_imgsz', 640)}")
    main_print(f"DDP                  : {distributed}")
    main_print(f"World size           : {world_size}")
    main_print(f"Batch mode           : {batch_mode}")
    main_print(f"Batch / GPU          : {local_batch}")
    main_print(f"Global batch         : {global_batch}")
    main_print(f"Output               : {out_dir.resolve()}")
    main_print("============================================================\n")

    print(
        f"[rank {rank}/{world_size}] local_rank={local_rank} -> {device}",
        flush=True,
    )

    train_dataset, train_loader = _build_data(
        cfg,
        split="train",
        augment=True,
        shuffle=True,
        batch_size=local_batch,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    if is_main_process():
        val_dataset, val_loader = _build_data(
            cfg,
            split="val",
            augment=False,
            shuffle=False,
            batch_size=int(train_cfg.get("val_batch", local_batch)),
            distributed=False,
            rank=0,
            world_size=1,
        )
    else:
        val_dataset = None
        val_loader = None

    main_print(f"train samples        : {len(train_dataset)}")
    if is_main_process():
        main_print(f"val samples          : {len(val_dataset)}")

    model = build_model_from_config(
        cfg,
        pretrained=bool(model_cfg.get("pretrained", True)),
        verbose=is_main_process(),
    ).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[int(device.index)],
            output_device=int(device.index),
            broadcast_buffers=True,
            find_unused_parameters=bool(
                train_cfg.get("ddp_find_unused_parameters", False)
            ),
        )

    # Same model init, different rank-specific augmentation/random stream.
    set_seed(seed + rank, deterministic=deterministic)

    generic_best_path = out_dir / "weights" / "best.pt"

    stage1 = run_stage(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=out_dir,
        stage="stage1",
        epochs=int(train_cfg.get("epochs", 300)),
        lr0=float(train_cfg.get("lr0", 0.01)),
        lrf=float(train_cfg.get("lrf", 0.01)),
        patience=int(train_cfg.get("patience", 50)),
        min_delta=float(train_cfg.get("early_stop_min_delta", 1e-6)),
        amp=amp,
        grad_clip=grad_clip,
        global_epoch_offset=0,
        global_best_map=-1.0,
        generic_best_path=generic_best_path,
    )

    global_best = float(stage1["global_best_map"])
    global_offset = int(stage1["epochs_ran"])

    stage2_cfg = train_cfg.get("stage2", {}) or {}
    stage2_result = None

    if bool(stage2_cfg.get("enabled", True)):
        distributed_barrier()
        load_weights_strict(
            stage1["stage_best_path"],
            model,
            device,
        )
        distributed_barrier()

        stage2_result = run_stage(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            out_dir=out_dir,
            stage="stage2",
            epochs=int(stage2_cfg.get("epochs", 200)),
            lr0=float(stage2_cfg.get("lr0", 0.001)),
            lrf=float(stage2_cfg.get("lrf", 0.01)),
            patience=int(stage2_cfg.get("patience", 50)),
            min_delta=float(stage2_cfg.get("early_stop_min_delta", 1e-6)),
            amp=amp,
            grad_clip=grad_clip,
            global_epoch_offset=global_offset,
            global_best_map=global_best,
            generic_best_path=generic_best_path,
        )

        global_best = float(stage2_result["global_best_map"])

    summary = {
        "output_dir": str(out_dir.resolve()),
        "best_checkpoint": str(generic_best_path.resolve()),
        "global_best_map5095": global_best,
        "stage1_best_map5095": float(stage1["stage_best_map"]),
        "stage2_best_map5095": (
            None
            if stage2_result is None
            else float(stage2_result["stage_best_map"])
        ),
        "world_size": int(world_size),
        "batch_per_gpu": int(local_batch),
        "global_batch": int(global_batch),
    }

    if is_main_process():
        (out_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n============================================================")
        print("Training finished")
        print("============================================================")
        for key, value in summary.items():
            print(f"{key:<24}: {value}")
        print("============================================================\n")

    distributed_barrier()
    cleanup_distributed()
    return summary
