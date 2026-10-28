#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AHR2Net B2 proposal-only threshold audit (same GMLS-v2 checkpoint).

This diagnostic does not train, change weights, or use GT to propose regions.
It evaluates the same low-resolution-proxy model input as B2, then sweeps
proposal confidence thresholds on the SAME forward output. GT is only used
for candidate coverage evaluation. No final detector AP is computed.

Dependency: put alongside eval_ahr2_b2_detail_retention.py in YOLO-tcy_v4.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset, build_dual_resolution_dataloader,
)
from engine.runtime import select_device, preprocess_batch, autocast_context
from engine import evaluator as ev
from models.gtga_bhlr_factory import load_model_from_checkpoint
from eval_ahr2_b2_detail_retention import (
    gt_boxes_xyxy, grid_mask, prediction_heatmap, coverage,
)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='3')
    p.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--thresholds', type=float, nargs='+', default=[.01, .003, .001, .0001])
    p.add_argument('--budgets', type=float, nargs='+', default=[1, 3, 5])
    p.add_argument('--iou', type=float, default=.70)
    p.add_argument('--max-det', type=int, default=100)
    p.add_argument('--tile', type=int, default=8)
    p.add_argument('--min-sigma', type=float, default=7.0)
    p.add_argument('--box-margin', type=float, default=1.0)
    p.add_argument('--rgb', default=None)
    p.add_argument('--tir', default=None)
    p.add_argument('--pair-mode', default=None, choices=('relative','filename','stem'))
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-native', action='store_true',
                   help='Skip high-resolution native proposal baseline (faster, but loses proxy-vs-native diagnostic)')
    p.add_argument('--save-dir', default='runs/ahr2net/gmls_correct/proposal_threshold_audit')
    return p.parse_args()


def write_rows(path, fields, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k:r.get(k,'') for k in fields} for r in rows)


@torch.inference_mode()
def main():
    a=arguments()
    thresholds=sorted(set(a.thresholds), reverse=True)
    budgets=sorted(set(a.budgets))
    if any(not 0<t<1 for t in thresholds) or any(not 0<b<=100 for b in budgets):
        raise ValueError('thresholds must be in (0,1), budgets in (0,100]')
    device=select_device(a.device)
    weight=Path(a.weights).expanduser().resolve()
    model,_,cfg=load_model_from_checkpoint(str(weight),device=device,
                                            fusion_override='auto',strict=True,verbose=False)
    model.eval()
    dc=cfg.get('data',{}) or {}
    rgb=a.rgb or dc.get('rgb')
    tir=a.tir or dc.get('tir')
    if not rgb or not tir:
        raise ValueError('RGB/TIR YAML missing in checkpoint: supply --rgb and --tir')
    ds=build_dual_resolution_dataset(rgb_yaml=rgb,tir_yaml=tir,split=a.split,
        rgb_high_imgsz=int(dc.get('rgb_imgsz',1920)),
        rgb_semantic_imgsz=int(dc.get('rgb_semantic_imgsz',640)),
        tir_imgsz=int(dc.get('tir_imgsz',640)),
        pair_mode=a.pair_mode or dc.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    dl=build_dual_resolution_dataloader(ds,batch_size=a.batch,workers=a.workers,
                                        shuffle=False,pin_memory=device.type=='cuda')
    out=Path(a.save_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    counts=defaultdict(list)
    gt_num=defaultdict(int)
    n_img=0
    print('[WEIGHT]',weight)
    print('[METHOD] low-resolution semantic RGB upsampled to high branch (same B2 proxy)')
    print('[THRESHOLDS]',thresholds,'[BUDGETS]',budgets)
    print('[NOTE] when no proposals exist, original B2 grid_mask selects arbitrary tied zero-score tiles')
    print('[SOURCES]', ['low_proxy'] if a.no_native else ['low_proxy','native_high'])
    for batch in tqdm(dl,desc='Proposal confidence audit',dynamic_ncols=True):
        if a.max_images and n_img>=a.max_images:
            break
        batch=preprocess_batch(batch,device)
        semantic=batch['rgb_semantic_img']
        high=batch['rgb_img']
        tir_batch=batch['tir_img']
        bs,_,H,W=high.shape
        _,_,h,w=semantic.shape
        if h%a.tile or w%a.tile:
            raise ValueError('semantic canvas must divide evenly into --tile')
        proxy=F.interpolate(semantic,size=(H,W),mode='bilinear',align_corners=False)
        sources={'low_proxy':proxy}
        if not a.no_native:
            sources['native_high']=high
        for source,source_high in sources.items():
            with autocast_context(device,not a.no_amp):
                raw=model(source_high,tir_batch,rgb_semantic=semantic)
            # Same forward result at all thresholds; only postprocessing changes.
            for th in thresholds:
                proposals=ev.postprocess_predictions(raw,model=model,conf_thres=th,
                                                      iou_thres=a.iou,max_det=a.max_det)
                for i,pred in enumerate(proposals):
                    if a.max_images and n_img+i>=a.max_images:
                        continue
                    name=Path(batch['rgb_path'][i]).name
                    gt=gt_boxes_xyxy(batch,i,h,w)
                    n=len(pred['conf'])
                    counts[(source,th)].append(n)
                    gt_num[(source,th)]+=len(gt)
                    scores=pred['conf'].detach().float().cpu().numpy()
                    score_max=float(scores.max()) if len(scores) else 0.
                    detmap=prediction_heatmap(pred,h,w,a.min_sigma,a.box_margin)
                    for budget in budgets:
                        mask=grid_mask(detmap,budget,a.tile)
                        cov=coverage(mask,gt)
                        for j,stat in enumerate(cov):
                            x1,y1,x2,y2=gt[j]
                            side=max(x2-x1,y2-y1)
                            size_bin=('<8' if side<8 else '8-16' if side<16 else
                                      '16-32' if side<32 else '>=32')
                            rows.append(dict(source=source,image=name,threshold=th,budget_pct=budget,
                                proposals=n,has_proposal=int(n>0),max_conf=score_max,
                                size_bin=size_bin,**stat))
        n_img+=bs
    summary=[]
    for source in (['low_proxy'] if a.no_native else ['low_proxy','native_high']):
        for th in thresholds:
            ct=counts[(source,th)]
            for budget in budgets:
                group=[r for r in rows if r['source']==source and r['threshold']==th and r['budget_pct']==budget]
                hit=lambda field,rr:float(np.mean([v[field] for v in rr])*100) if rr else float('nan')
                valid=[r for r in group if r['has_proposal']]
                zero=[r for r in group if not r['has_proposal']]
                summary.append(dict(source=source,threshold=th,budget_pct=budget,images=len(ct),
                    zero_proposal_images=sum(c==0 for c in ct),zero_proposal_pct=100*sum(c==0 for c in ct)/max(len(ct),1),
                    mean_proposals=float(np.mean(ct)) if ct else 0.,
                    total_gt=len(group),center_pct=hit('center',group),
                    cover50_pct=hit('cover50',group),cover80_pct=hit('cover80',group),
                    center_with_proposal_pct=hit('center',valid),
                    center_zero_proposal_pct=hit('center',zero),
                    small8_16_center_pct=hit('center',[r for r in group if r['size_bin']=='8-16'])))
    detail_fields=['source','image','threshold','budget_pct','proposals','has_proposal','max_conf',
                   'size_bin','center','cover50','cover80','covered']
    summary_fields=['source','threshold','budget_pct','images','zero_proposal_images','zero_proposal_pct',
                    'mean_proposals','total_gt','center_pct','cover50_pct','cover80_pct',
                    'center_with_proposal_pct','center_zero_proposal_pct','small8_16_center_pct']
    write_rows(out/'threshold_summary.csv',summary_fields,summary)
    write_rows(out/'per_gt.csv',detail_fields,rows)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n=== THRESHOLD SUMMARY (3% budget only; percentages) ===')
    print(f"{'source':>12} {'conf':>9} {'zero_imgs':>10} {'mean_n':>8} {'center':>9} {'cover80':>9} {'small_ctr':>10}")
    for row in summary:
        if row['budget_pct'] != 3: continue
        print(f"{row['source']:>12} {row['threshold']:9g} {row['zero_proposal_images']:10d} {row['mean_proposals']:8.3f} "
              f"{row['center_pct']:9.3f} {row['cover80_pct']:9.3f} {row['small8_16_center_pct']:10.3f}")
    print('[SAVED]',out)

if __name__=='__main__':
    main()
