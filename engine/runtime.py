from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_config(path: str | Path) -> Dict:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a dict, got {type(cfg)}")
    return cfg


def set_seed(seed: int = 0, deterministic: bool = True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def parse_device_ids(device) -> List[int]:
    """Parse 3, '3', '3,4,5', [3,4,5]. CPU/-1 returns []."""
    if device is None:
        return [0]

    if isinstance(device, int):
        return [] if device < 0 else [int(device)]

    if isinstance(device, (list, tuple)):
        out = []
        for item in device:
            out.extend(parse_device_ids(item))
        seen = set()
        result = []
        for x in out:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result

    text = str(device).strip().lower()
    if text in {"cpu", "-1", "none"}:
        return []
    if text.startswith("cuda:"):
        text = text[len("cuda:"):]
    text = text.replace(";", ",")

    ids = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx >= 0:
            ids.append(idx)

    seen = set()
    result = []
    for idx in ids:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def distributed_info():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def select_device(device: str | int | Iterable[int] = "0") -> torch.device:
    ids = parse_device_ids(device)
    if not ids:
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print("[device] CUDA unavailable; falling back to CPU")
        return torch.device("cpu")

    index = int(ids[0])
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid CUDA device {index}; available=0..{torch.cuda.device_count()-1}"
        )

    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def setup_distributed(device_spec="0"):
    """Initialize NCCL DDP when launched by torchrun."""
    rank, local_rank, world_size = distributed_info()

    if world_size <= 1:
        return select_device(device_spec), 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested but CUDA is unavailable.")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    requested = parse_device_ids(device_spec)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if visible:
        # CUDA_VISIBLE_DEVICES=3,4,5 => children see cuda:0,1,2.
        cuda_index = int(local_rank)
    elif len(requested) >= world_size:
        # Manual torchrun without CUDA_VISIBLE_DEVICES.
        cuda_index = int(requested[local_rank])
    else:
        cuda_index = int(local_rank)

    if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
        raise ValueError(
            f"Rank {rank}: invalid CUDA device {cuda_index}; "
            f"torch sees 0..{torch.cuda.device_count()-1}. "
            f"requested={requested}, CUDA_VISIBLE_DEVICES={visible!r}"
        )

    torch.cuda.set_device(cuda_index)
    device = torch.device(f"cuda:{cuda_index}")
    return device, rank, local_rank, world_size


def distributed_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


def distributed_mean(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)

    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(dist.get_world_size())
    return float(tensor.item())


def distributed_mean_dict(values: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return {k: float(v) for k, v in values.items()}

    keys = sorted(values.keys())
    if not keys:
        return {}

    tensor = torch.tensor(
        [float(values[k]) for k in keys],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(dist.get_world_size())
    out = tensor.detach().cpu().tolist()
    return {k: float(v) for k, v in zip(keys, out)}


def broadcast_metrics(metrics, device: torch.device):
    """Broadcast rank-0 validation metrics to every DDP rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return metrics

    keys = (
        "precision",
        "recall",
        "map50",
        "map75",
        "map5095",
        "images",
        "instances",
    )

    rank = dist.get_rank()
    vals = [float(metrics[k]) for k in keys] if rank == 0 else [0.0] * len(keys)
    tensor = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.broadcast(tensor, src=0)
    vals = tensor.detach().cpu().tolist()

    return {
        "precision": float(vals[0]),
        "recall": float(vals[1]),
        "map50": float(vals[2]),
        "map75": float(vals[3]),
        "map5095": float(vals[4]),
        "images": int(round(vals[5])),
        "instances": int(round(vals[6])),
    }


def autocast_context(device: torch.device, enabled: bool = True):
    enabled = bool(enabled and device.type == "cuda")
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    )


def make_grad_scaler(device: torch.device, enabled: bool = True):
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def preprocess_batch(batch: Dict, device: torch.device) -> Dict:
    for key in ("rgb_img", "rgb_semantic_img", "tir_img"):
        if key not in batch:
            raise KeyError(f"batch missing image key: {key}")
        batch[key] = (
            batch[key]
            .to(device, non_blocking=True)
            .float()
            .div_(255.0)
        )

    tensor_keys = (
        "rgb_cls",
        "rgb_bboxes",
        "rgb_semantic_bboxes",
        "rgb_batch_idx",
        "tir_cls",
        "tir_bboxes",
        "tir_batch_idx",
    )

    for key in tensor_keys:
        if key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

    return batch


def gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_reserved(device) / (1024 ** 3))


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
