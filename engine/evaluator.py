from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics.utils import nms, ops
from ultralytics.utils.metrics import DetMetrics, box_iou

from engine.runtime import (
    autocast_context,
    preprocess_batch,
    synchronize,
)


def postprocess_predictions(
    preds,
    model: torch.nn.Module,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_det: int = 300,
):
    """Support YOLO26 end-to-end output and traditional YOLO output."""
    if isinstance(preds, (tuple, list)):
        if len(preds) == 0:
            raise RuntimeError('Empty model output')
        preds = preds[0]

    if isinstance(preds, dict):
        for key in ('pred', 'preds', 'output'):
            if key in preds:
                preds = preds[key]
                break
        else:
            raise TypeError(
                f'Cannot parse dict model output; keys={list(preds.keys())}'
            )
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

    if not torch.is_tensor(preds):
        raise TypeError(f'Final prediction is not Tensor: {type(preds)}')

    is_end2end_output = (
        preds.ndim == 3
        and preds.shape[-1] == 6
    )

    if getattr(model, 'end2end', False) or is_end2end_output:
        results = []
        for pred in preds:
            pred = pred[pred[:, 4] >= float(conf_thres)]
            if pred.shape[0] > 0:
                order = torch.argsort(pred[:, 4], descending=True)
                pred = pred[order]
            pred = pred[: int(max_det)]
            results.append(
                {
                    'bboxes': pred[:, :4],
                    'conf': pred[:, 4],
                    'cls': pred[:, 5],
                }
            )
        return results

    outputs = nms.non_max_suppression(
        preds,
        float(conf_thres),
        float(iou_thres),
        nc=model.nc,
        multi_label=True,
        agnostic=False,
        max_det=int(max_det),
        end2end=False,
        rotated=False,
    )

    return [
        {
            'bboxes': out[:, :4],
            'conf': out[:, 4],
            'cls': out[:, 5],
        }
        for out in outputs
    ]


def match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
):
    correct = np.zeros(
        (pred_classes.shape[0], iouv.shape[0]),
        dtype=bool,
    )

    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return torch.from_numpy(correct)

    correct_class = (
        true_classes[:, None]
        == pred_classes[None, :]
    )
    iou = iou * correct_class
    iou_np = iou.detach().cpu().numpy()

    for threshold_index, threshold in enumerate(
        iouv.detach().cpu().tolist()
    ):
        matches = np.array(
            np.nonzero(iou_np >= threshold)
        ).T

        if matches.shape[0] == 0:
            continue

        if matches.shape[0] > 1:
            matches = matches[
                iou_np[
                    matches[:, 0],
                    matches[:, 1],
                ].argsort()[::-1]
            ]
            matches = matches[
                np.unique(
                    matches[:, 1],
                    return_index=True,
                )[1]
            ]
            matches = matches[
                np.unique(
                    matches[:, 0],
                    return_index=True,
                )[1]
            ]

        correct[
            matches[:, 1].astype(int),
            threshold_index,
        ] = True

    return torch.from_numpy(correct)


def prepare_semantic_ground_truth(batch: Dict, sample_index: int):
    """Final detector coordinates are RGB-semantic coordinates."""
    mask = batch['rgb_batch_idx'] == int(sample_index)

    cls = batch['rgb_cls'][mask].squeeze(-1)
    boxes = batch['rgb_semantic_bboxes'][mask]

    h = int(batch['rgb_semantic_img'].shape[2])
    w = int(batch['rgb_semantic_img'].shape[3])

    if boxes.shape[0] > 0:
        boxes = ops.xywh2xyxy(boxes)
        scale = torch.tensor(
            [w, h, w, h],
            device=boxes.device,
            dtype=boxes.dtype,
        )
        boxes = boxes * scale

    return {
        'cls': cls,
        'bboxes': boxes,
    }


def process_single_image(pred: Dict, gt: Dict, iouv: torch.Tensor):
    pred_cls = pred['cls']
    pred_boxes = pred['bboxes']
    gt_cls = gt['cls']
    gt_boxes = gt['bboxes']

    if pred_cls.shape[0] == 0:
        return np.zeros((0, len(iouv)), dtype=bool)

    if gt_cls.shape[0] == 0:
        return np.zeros(
            (pred_cls.shape[0], len(iouv)),
            dtype=bool,
        )

    iou = box_iou(gt_boxes, pred_boxes)

    return (
        match_predictions(
            pred_classes=pred_cls,
            true_classes=gt_cls,
            iou=iou,
            iouv=iouv,
        )
        .cpu()
        .numpy()
    )


def tensor_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    image = (
        image_tensor.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    image = (image * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _draw_title_bar(image: np.ndarray, text: str):
    h, w = image.shape[:2]
    font_scale = max(0.45, min(h, w) / 1200.0)
    thickness = max(1, int(min(h, w) / 500))
    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    bar_h = max(th + baseline + 12, 28)
    cv2.rectangle(image, (0, 0), (w - 1, min(bar_h, h - 1)), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (8, min(th + 7, h - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_prediction_tile(
    image_bgr: np.ndarray,
    pred: Dict,
    image_name: str,
    preview_conf: float = 0.25,
):
    image = image_bgr.copy()
    h, w = image.shape[:2]
    thickness = max(2, int(min(h, w) / 320))
    font_scale = max(0.45, min(h, w) / 1100.0)

    boxes = pred['bboxes'].detach().float().cpu().numpy()
    confs = pred['conf'].detach().float().cpu().numpy()
    shown = 0

    for box, conf in zip(boxes, confs):
        conf = float(conf)
        if conf < float(preview_conf):
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        label = f'UAV {conf:.2f}'
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )
        shown += 1

    _draw_title_bar(
        image,
        f'PRED | {image_name} | boxes={shown} | conf>={preview_conf:.2f}',
    )
    return image


def draw_ground_truth_tile(image_bgr: np.ndarray, gt: Dict, image_name: str):
    image = image_bgr.copy()
    h, w = image.shape[:2]
    thickness = max(2, int(min(h, w) / 320))

    boxes = gt['bboxes'].detach().float().cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    _draw_title_bar(
        image,
        f'GROUND TRUTH | {image_name} | boxes={len(boxes)}',
    )
    return image


def make_image_grid(images: list[np.ndarray], cols: int = 3, gap: int = 8):
    if not images:
        raise ValueError('No images to compose')

    cols = max(int(cols), 1)
    gap = max(int(gap), 0)
    tile_h, tile_w = images[0].shape[:2]

    resized = []
    for image in images:
        if image.shape[:2] != (tile_h, tile_w):
            image = cv2.resize(
                image,
                (tile_w, tile_h),
                interpolation=cv2.INTER_LINEAR,
            )
        resized.append(image)

    rows = int(np.ceil(len(resized) / cols))
    grid_h = rows * tile_h + (rows - 1) * gap
    grid_w = cols * tile_w + (cols - 1) * gap
    grid = np.full((grid_h, grid_w, 3), 245, dtype=np.uint8)

    for i, image in enumerate(resized):
        row, col = divmod(i, cols)
        y1 = row * (tile_h + gap)
        x1 = col * (tile_w + gap)
        grid[y1:y1 + tile_h, x1:x1 + tile_w] = image

    return grid


@torch.inference_mode()
def validate_metrics(
    model,
    loader,
    device: torch.device,
    amp: bool = True,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_det: int = 300,
    preview_num: int = 0,
    preview_conf: float = 0.25,
    preview_cols: int = 3,
    preview_dir: Optional[str | Path] = None,
):
    model.eval()

    names = model.names
    if not isinstance(names, dict):
        names = {i: name for i, name in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}

    metrics = DetMetrics(names=names)
    iouv = torch.linspace(0.50, 0.95, 10, device=device)

    seen = 0
    total_instances = 0
    pred_tiles = []
    gt_tiles = []

    pbar = tqdm(loader, total=len(loader), dynamic_ncols=True, leave=True)

    for batch in pbar:
        batch_size = int(batch['rgb_img'].shape[0])
        seen += batch_size
        total_instances += int(batch['rgb_cls'].shape[0])

        batch = preprocess_batch(batch, device)

        with autocast_context(device, amp):
            raw = model(
                batch['rgb_img'],
                batch['tir_img'],
                rgb_semantic=batch['rgb_semantic_img'],
            )

        preds = postprocess_predictions(
            raw,
            model=model,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )

        for sample_index, pred in enumerate(preds):
            gt = prepare_semantic_ground_truth(batch, sample_index)
            tp = process_single_image(pred, gt, iouv)

            target_cls = (
                gt['cls'].detach().float().cpu().numpy()
            )

            if pred['cls'].shape[0] == 0:
                pred_conf = np.zeros(0, dtype=np.float32)
                pred_cls = np.zeros(0, dtype=np.float32)
            else:
                pred_conf = pred['conf'].detach().float().cpu().numpy()
                pred_cls = pred['cls'].detach().float().cpu().numpy()

            metrics.update_stats(
                {
                    'tp': tp,
                    'conf': pred_conf,
                    'pred_cls': pred_cls,
                    'target_cls': target_cls,
                    'target_img': np.unique(target_cls),
                    'im_name': Path(batch['rgb_path'][sample_index]).name,
                }
            )

            if len(pred_tiles) < int(preview_num):
                image = tensor_to_bgr(
                    batch['rgb_semantic_img'][sample_index]
                )
                image_name = Path(batch['rgb_path'][sample_index]).name
                pred_tiles.append(
                    draw_prediction_tile(
                        image,
                        pred,
                        image_name,
                        preview_conf=preview_conf,
                    )
                )
                gt_tiles.append(
                    draw_ground_truth_tile(
                        image,
                        gt,
                        image_name,
                    )
                )

        pbar.set_description(
            f'all images={seen} instances={total_instances}',
            refresh=False,
        )

    pbar.close()

    metrics.process(plot=False)
    precision, recall, map50, map5095 = metrics.mean_results()
    map75 = float(metrics.box.map75)

    result = {
        'precision': float(precision),
        'recall': float(recall),
        'map50': float(map50),
        'map75': map75,
        'map5095': float(map5095),
        'images': int(seen),
        'instances': int(total_instances),
    }

    if preview_dir is not None and pred_tiles:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(preview_dir / 'prediction_grid.jpg'),
            make_image_grid(pred_tiles, cols=preview_cols),
        )
        cv2.imwrite(
            str(preview_dir / 'ground_truth_grid.jpg'),
            make_image_grid(gt_tiles, cols=preview_cols),
        )

    return result


def get_model_complexity(
    model,
    rgb_imgsz: int,
    rgb_semantic_imgsz: int,
    tir_imgsz: int,
    device: torch.device,
):
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops_g = 0.0

    try:
        import thop

        was_training = model.training
        model.eval()

        rgb_high = torch.zeros(
            1, 3, int(rgb_imgsz), int(rgb_imgsz),
            device=device,
        )
        rgb_semantic = torch.zeros(
            1, 3, int(rgb_semantic_imgsz), int(rgb_semantic_imgsz),
            device=device,
        )
        tir = torch.zeros(
            1, 3, int(tir_imgsz), int(tir_imgsz),
            device=device,
        )

        with torch.no_grad():
            macs, _ = thop.profile(
                model,
                inputs=(rgb_high, tir, rgb_semantic),
                verbose=False,
            )
        flops_g = float(macs) * 2.0 / 1e9

        model.train(was_training)
        del rgb_high, rgb_semantic, tir
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    except ImportError:
        print(
            '[complexity] thop not installed; FLOPs=0. '
            'Install with: python -m pip install ultralytics-thop '
            '-i https://mirrors.aliyun.com/pypi/simple/'
        )
    except Exception as exc:
        print(f'[complexity] FLOPs calculation failed: {exc}')

    return float(params_m), float(flops_g)


@torch.inference_mode()
def benchmark_fps(
    model,
    rgb_imgsz: int,
    rgb_semantic_imgsz: int,
    tir_imgsz: int,
    device: torch.device,
    amp: bool = True,
    warmup: int = 20,
    iterations: int = 100,
):
    model.eval()

    rgb_high = torch.randn(
        1, 3, int(rgb_imgsz), int(rgb_imgsz),
        device=device,
    )
    rgb_semantic = torch.randn(
        1, 3, int(rgb_semantic_imgsz), int(rgb_semantic_imgsz),
        device=device,
    )
    tir = torch.randn(
        1, 3, int(tir_imgsz), int(tir_imgsz),
        device=device,
    )

    with autocast_context(device, amp):
        for _ in range(max(int(warmup), 0)):
            _ = model(
                rgb_high,
                tir,
                rgb_semantic=rgb_semantic,
            )

    synchronize(device)
    t0 = time.perf_counter()

    with autocast_context(device, amp):
        for _ in range(max(int(iterations), 1)):
            _ = model(
                rgb_high,
                tir,
                rgb_semantic=rgb_semantic,
            )

    synchronize(device)
    elapsed = time.perf_counter() - t0
    n = max(int(iterations), 1)
    latency_s = elapsed / n

    del rgb_high, rgb_semantic, tir
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return (
        (1.0 / latency_s) if latency_s > 0 else 0.0,
        latency_s * 1000.0,
    )


def save_metrics_csv(path: str | Path, row: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'Model',
        'Params(M)',
        'FLOPs(G)',
        'FPS',
        'Latency(ms)',
        'P(%)',
        'R(%)',
        'AP50(%)',
        'AP75(%)',
        'AP50:95(%)',
        'RGB_high_imgsz',
        'RGB_semantic_imgsz',
        'TIR_imgsz',
        'Split',
    ]

    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fieldnames})
