#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AHR2Net Exp B2: area-budgeted, no-GT local high-RGB detail retention.

Two passes per batch on a single checkpoint:
  (1) Proposal pass with 640 RGB semantic upsampled into high-RGB branch.
      This pass never sees original high-resolution image pixels.
  (2) Original RGB high branch with only chosen 1/3/5% regions kept sharp;
      remaining high RGB blurred. RGB semantic and TIR are IDENTICAL.

Methods: detector, hybrid, rgb_saliency; gt_oracle is an explicitly labeled
GT-informed diagnostic reference, NOT deployable and not a ceiling guarantee.
The trained 1920-high model is OOD in proposal pass. None of these results
are independently trained low-res-vs-high-res comparisons.

Requires: standard YOLO-tcy_v4 project code and libraries only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import zlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset, build_dual_resolution_dataloader
from engine.runtime import select_device, preprocess_batch, autocast_context
from engine import evaluator as ev
from models.gtga_bhlr_factory import load_model_from_checkpoint

MODES = ('detector', 'hybrid', 'rgb_saliency', 'random', 'gt_oracle')

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', required=True)
    p.add_argument('--split', default='val', choices=('val', 'test', 'train'))
    p.add_argument('--device', default='3')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--budgets', type=float, nargs='+', default=[1, 3, 5])
    p.add_argument('--modes', nargs='+', choices=MODES, default=['detector', 'hybrid', 'rgb_saliency', 'gt_oracle'])
    p.add_argument('--tile', type=int, default=8)
    p.add_argument('--feather', type=int, default=11,
                   help='Odd pixel-size blending kernel at high-RGB scale; 1=hard boundary')
    p.add_argument('--hybrid-weight', type=float, default=0.80)
    p.add_argument('--proposal-conf', type=float, default=0.01)
    p.add_argument('--proposal-max-det', type=int, default=100)
    p.add_argument('--min-sigma', type=float, default=7.0)
    p.add_argument('--box-margin', type=float, default=1.0)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.70)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--rgb', default=None)
    p.add_argument('--tir', default=None)
    p.add_argument('--pair-mode', default=None, choices=('relative', 'stem', 'filename'))
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--save-dir', default='runs/ahr2net/gmls_correct/exp_b2_val')
    return p.parse_args()


def robust_norm(x):
    x = np.maximum(np.nan_to_num(x, nan=0, posinf=0, neginf=0), 0)
    z = float(np.percentile(x, 99))
    return np.clip(x / max(z, 1e-6), 0, 1).astype(np.float32)


def rgb_saliency(chw):
    gray = np.asarray(chw).transpose(1, 2, 0).astype(np.float32).mean(2) / 255.0
    sm = cv2.GaussianBlur(gray, (0, 0), 1.2)
    local = cv2.GaussianBlur(gray, (0, 0), 4.5)
    wide = cv2.GaussianBlur(gray, (0, 0), 12)
    dog1 = robust_norm(np.abs(sm - local))
    dog2 = robust_norm(np.abs(local - wide))
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    edge = robust_norm(cv2.magnitude(gx, gy))
    return robust_norm(cv2.GaussianBlur(.50 * dog1 + .25 * dog2 + .25 * edge, (3,3), .6))


def prediction_heatmap(pred, h, w, min_sigma, margin):
    out = np.zeros((h, w), dtype=np.float32)
    boxes = pred['bboxes'].detach().float().cpu().numpy()
    conf = pred['conf'].detach().float().cpu().numpy()
    for box, score in zip(boxes, conf):
        x1, y1, x2, y2 = map(float, box)
        cx, cy = float(np.clip((x1+x2)/2, 0, w-1)), float(np.clip((y1+y2)/2, 0, h-1))
        sx = max(float(min_sigma), abs(x2-x1) * .5 * margin)
        sy = max(float(min_sigma), abs(y2-y1) * .5 * margin)
        xl = max(0, int(cx-3*sx)); xr = min(w, int(cx+3*sx+2))
        yl = max(0, int(cy-3*sy)); yr = min(h, int(cy+3*sy+2))
        if xl >= xr or yl >= yr: continue
        xx = np.arange(xl, xr, dtype=np.float32)
        yy = np.arange(yl, yr, dtype=np.float32)
        g = float(score) * np.exp(-.5*((xx[None,:]-cx)/sx)**2 - .5*((yy[:,None]-cy)/sy)**2)
        np.maximum(out[yl:yr, xl:xr], g, out=out[yl:yr, xl:xr])
    return robust_norm(out)


def gt_boxes_xyxy(batch, image_index, h, w):
    idx = batch['rgb_batch_idx'] == image_index
    boxes = batch['rgb_semantic_bboxes'][idx].detach().float().cpu().numpy()
    out = []
    for cx, cy, bw, bh in boxes:
        x1, x2 = (cx-bw/2)*w, (cx+bw/2)*w
        y1, y2 = (cy-bh/2)*h, (cy+bh/2)*h
        out.append((float(np.clip(x1,0,w)), float(np.clip(y1,0,h)),
                    float(np.clip(x2,0,w)), float(np.clip(y2,0,h))))
    return out


def oracle_heatmap(gt, h, w, sigma_min, margin):
    out = np.zeros((h,w), np.float32)
    for x1,y1,x2,y2 in gt:
        cx,cy = (x1+x2)*.5, (y1+y2)*.5
        sx=max(sigma_min,.5*(x2-x1)*margin);sy=max(sigma_min,.5*(y2-y1)*margin)
        xl=max(0,int(cx-3*sx));xr=min(w,int(cx+3*sx+2))
        yl=max(0,int(cy-3*sy));yr=min(h,int(cy+3*sy+2))
        if xl>=xr or yl>=yr:continue
        xx=np.arange(xl,xr,dtype=np.float32); yy=np.arange(yl,yr,dtype=np.float32)
        g=np.exp(-.5*((xx[None,:]-cx)/sx)**2-.5*((yy[:,None]-cy)/sy)**2)
        np.maximum(out[yl:yr,xl:xr],g,out=out[yl:yr,xl:xr])
    return out


def grid_mask(score, budget, tile):
    h,w=score.shape
    if h % tile or w % tile: raise ValueError(f'{w}x{h} cannot be tiled at {tile}')
    tile_scores=score.reshape(h//tile,tile,w//tile,tile).mean(axis=(1,3)).reshape(-1)
    n=max(1,min(tile_scores.size,int(math.floor(tile_scores.size*budget/100+.5))))
    idx=np.argpartition(-tile_scores,n-1)[:n]
    mask=np.zeros(tile_scores.size,np.uint8);mask[idx]=1
    return np.repeat(np.repeat(mask.reshape(h//tile,w//tile),tile,axis=0),tile,axis=1)


def seeded_random_mask(h,w,tile,budget,image_name,seed):
    n=(h//tile)*(w//tile)
    k=max(1,min(n,int(math.floor(n*budget/100+.5))))
    rng=np.random.default_rng(zlib.crc32(image_name.encode('utf8')) ^ (int(seed)&0xffffffff))
    choice=rng.choice(n,size=k,replace=False)
    grid=np.zeros(n,np.uint8);grid[choice]=1
    return np.repeat(np.repeat(grid.reshape(h//tile,w//tile),tile,axis=0),tile,axis=1)


def coverage(mask,gt):
    h,w=mask.shape
    out=[]
    for x1,y1,x2,y2 in gt:
        x1=int(np.clip(math.floor(x1),0,w-1)); y1=int(np.clip(math.floor(y1),0,h-1))
        x2=int(np.clip(math.ceil(x2),x1+1,w)); y2=int(np.clip(math.ceil(y2),y1+1,h))
        fraction=float(mask[y1:y2,x1:x2].mean())
        cx=int(np.clip((x1+x2-1)/2,0,w-1));cy=int(np.clip((y1+y2-1)/2,0,h-1))
        out.append({'center':int(mask[cy,cx]>0),'cover50':int(fraction>=.5),
                    'cover80':int(fraction>=.8),'covered':fraction})
    return out


def add_metrics(metrics, preds, batch, device):
    iouv=torch.linspace(.50,.95,10,device=device)
    for i,pred in enumerate(preds):
        gt=ev.prepare_semantic_ground_truth(batch,i)
        tp=ev.process_single_image(pred,gt,iouv)
        cl=gt['cls'].detach().float().cpu().numpy()
        metrics.update_stats({'tp':tp,
                              'conf':pred['conf'].detach().float().cpu().numpy(),
                              'pred_cls':pred['cls'].detach().float().cpu().numpy(),
                              'target_cls':cl, 'target_img':np.unique(cl),
                              'im_name':Path(batch['rgb_path'][i]).name})


def dump_csv(path,rows,keys):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:row.get(k,'') for k in keys})


@torch.inference_mode()
def main():
    args=parse_args()
    budgets=sorted(set(float(x) for x in args.budgets))
    modes=list(dict.fromkeys(args.modes))
    if not budgets or not all(0<x<=100 for x in budgets):raise ValueError('budgets must be in (0,100]')
    if args.tile<1 or args.batch<1 or args.workers<0:raise ValueError('Invalid tile/batch/workers')
    if args.feather<1 or args.feather%2!=1:raise ValueError('--feather must be positive odd')
    if not 0<=args.hybrid_weight<=1:raise ValueError('Invalid hybrid weight')
    if not 0<args.proposal_conf<=1 or not 0<=args.conf<=1:raise ValueError('Invalid conf thresholds')
    save_dir=Path(args.save_dir).expanduser().resolve();save_dir.mkdir(parents=True,exist_ok=True)
    weight_path=Path(args.weights).expanduser().resolve()
    device=select_device(args.device)
    model,_,cfg=load_model_from_checkpoint(str(weight_path),device=device,
                                           fusion_override='auto',strict=True,verbose=False)
    model.eval()
    # Explicitly keep any optional DRG gains at zero: this is not the variable under study.
    drg=getattr(model,'agdh_drg',None)
    if drg is not None:
        for key in ('cls_gain','reg_gain'):
            if hasattr(drg,key):setattr(drg,key,0.)
    data_cfg=cfg.get('data',{}) or {}
    rgb=args.rgb or data_cfg.get('rgb');tir=args.tir or data_cfg.get('tir')
    if not rgb or not tir:raise ValueError('Missing RGB or TIR YAML in checkpoint; pass --rgb --tir')
    rh=int(data_cfg.get('rgb_imgsz',1920));sem=int(data_cfg.get('rgb_semantic_imgsz',640))
    tir_size=int(data_cfg.get('tir_imgsz',640))
    ds=build_dual_resolution_dataset(rgb_yaml=rgb,tir_yaml=tir,split=args.split,
              rgb_high_imgsz=rh,rgb_semantic_imgsz=sem,tir_imgsz=tir_size,
              pair_mode=args.pair_mode or data_cfg.get('pair_mode','relative'),
              augment=False,tir_channels=3,strict_pair=True)
    dl=build_dual_resolution_dataloader(ds,batch_size=args.batch,workers=args.workers,
                                         shuffle=False,pin_memory=device.type=='cuda')
    cases=['native','global_blur'] + [f'{mode}_{budget:g}pct' for mode in modes for budget in budgets]
    names=model.names
    if not isinstance(names,dict):names={i:n for i,n in enumerate(names)}
    names={int(k):str(v) for k,v in names.items()}
    metrics={case:ev.DetMetrics(names=names) for case in cases}
    coverage_rows=[];diagnostics=[];area_rows=[]
    processed=0
    print('[WEIGHTS]',weight_path)
    print('[CASES]',cases)
    print('[PROPOSAL] no original high-resolution pixels; high branch receives upsampled RGB-semantic')
    print('[RETENTION] ONLY high-RGB input modified; RGB-semantic and TIR unchanged')
    print('[GT_ORACLE] uses GT to choose sharp pixels, diagnostic ONLY; not a deployable baseline')
    for batch in tqdm(dl,desc='AHR2 B2 area-budget AP',dynamic_ncols=True):
        if args.max_images and processed>=args.max_images:break
        batch=preprocess_batch(batch,device)
        high=batch['rgb_img'];semantic=batch['rgb_semantic_img'];tir=batch['tir_img']
        bs,_,H,W=high.shape;_,_,h,w=semantic.shape
        if h%args.tile or w%args.tile:raise ValueError('semantic image not divisible by tile')
        # No use of original RGB-high values in proposal pass.
        high_proxy=F.interpolate(semantic,size=(H,W),mode='bilinear',align_corners=False)
        with autocast_context(device,not args.no_amp):
            raw_proposal=model(high_proxy,tir,rgb_semantic=semantic)
        pred_proposal=ev.postprocess_predictions(raw_proposal,model=model,
                    conf_thres=args.proposal_conf,iou_thres=args.iou,max_det=args.proposal_max_det)
        gt_per_image=[];all_masks={case:[] for case in cases[2:]}
        for i in range(bs):
            gt=gt_boxes_xyxy(batch,i,h,w)
            gt_per_image.append(gt)
            name=str(batch['rgb_path'][i])
            det_map=prediction_heatmap(pred_proposal[i],h,w,args.min_sigma,args.box_margin)
            rgb_array=(semantic[i].detach().float().cpu().numpy()*255).clip(0,255).astype(np.uint8)
            rgb_map=rgb_saliency(rgb_array)
            combo=robust_norm(args.hybrid_weight*det_map+(1-args.hybrid_weight)*rgb_map)
            scoremaps={'detector':det_map,'hybrid':combo,'rgb_saliency':rgb_map}
            if 'gt_oracle' in modes:
                scoremaps['gt_oracle']=oracle_heatmap(gt,h,w,args.min_sigma,args.box_margin)
            for mode in modes:
                for budget in budgets:
                    case=f'{mode}_{budget:g}pct'
                    if mode=='random':mask=seeded_random_mask(h,w,args.tile,budget,name,args.seed)
                    else:mask=grid_mask(scoremaps[mode],budget,args.tile)
                    all_masks[case].append(mask)
                    for g,stat in enumerate(coverage(mask,gt)):
                        x1,y1,x2,y2=gt[g]
                        side=max(x2-x1,y2-y1)
                        size_bin=('<8' if side<8 else '8-16' if side<16 else
                                  '16-32' if side<32 else '>=32')
                        coverage_rows.append({'case':case,'image':Path(name).name,
                                              'gt_index':g,'size_bin':size_bin,**stat})
                    area_rows.append({'case':case,'image':Path(name).name,
                                      'budget_pct':budget,'selected_pct':100*float(mask.mean())})
            diagnostics.append({'image':name,'proposals':len(pred_proposal[i]['conf']),
                                'gt_count':len(gt)})
        processed+=bs
        blur=F.avg_pool2d(F.pad(high,(2,2,2,2),mode='replicate'),5,stride=1)
        def evaluate(case,modified):
            with autocast_context(device,not args.no_amp):
                raw=model(modified,tir,rgb_semantic=semantic)
            pred=ev.postprocess_predictions(raw,model=model,conf_thres=args.conf,
                                            iou_thres=args.iou,max_det=args.max_det)
            add_metrics(metrics[case],pred,batch,device)
        evaluate('native',high)
        evaluate('global_blur',blur)
        for case in cases[2:]:
            arr=np.stack(all_masks[case],axis=0)[:,None,:,:].astype(np.float32)
            alpha=torch.from_numpy(arr).to(device)
            alpha=F.interpolate(alpha,size=(H,W),mode='nearest')
            if args.feather>1:
                r=args.feather//2
                alpha=F.avg_pool2d(F.pad(alpha,(r,r,r,r),mode='replicate'),
                                   args.feather,stride=1)
            blended=high*alpha+blur*(1-alpha)
            evaluate(case,blended)
    summary=[]
    for case in cases:
        m=metrics[case];m.process(plot=False)
        p,r,ap50,ap5095=m.mean_results()
        summary.append({'case':case,'precision':float(p),'recall':float(r),'AP50':float(ap50),
                        'AP75':float(m.box.map75),'AP50:95':float(ap5095),
                        'images':processed,'instances':sum(d['gt_count'] for d in diagnostics)})
    dump_csv(save_dir/'summary.csv',summary,
             ['case','precision','recall','AP50','AP75','AP50:95','images','instances'])
    dump_csv(save_dir/'coverage.csv',coverage_rows,
             ['case','image','gt_index','size_bin','center','cover50','cover80','covered'])
    summaries=[]
    for case in cases[2:]:
        matching=[r for r in coverage_rows if r['case']==case]
        for size_bin in ('all','<8','8-16','16-32','>=32'):
            group=matching if size_bin=='all' else [r for r in matching if r['size_bin']==size_bin]
            out={'case':case,'size_bin':size_bin,'instances':len(group)}
            for fld in ('center','cover50','cover80','covered'):
                out[fld+'_pct']=(round(100*float(np.mean([r[fld] for r in group])),3)
                                  if group else '')
            summaries.append(out)
    dump_csv(save_dir/'coverage_summary.csv',summaries,
             ['case','size_bin','instances','center_pct','cover50_pct','cover80_pct','covered_pct'])
    dump_csv(save_dir/'area.csv',area_rows,['case','image','budget_pct','selected_pct'])
    dump_csv(save_dir/'proposals.csv',diagnostics,['image','proposals','gt_count'])
    (save_dir/'config.json').write_text(json.dumps(vars(args),ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nFINAL SUMMARY (AP and P/R in %)')
    print(f"{'case':<22}{'AP50':>9}{'AP75':>9}{'AP50:95':>11}{'P':>9}{'R':>9}")
    for s in summary:
        print(f"{s['case']:<22}{100*s['AP50']:9.3f}{100*s['AP75']:9.3f}"
              f"{100*s['AP50:95']:11.3f}{100*s['precision']:9.3f}{100*s['recall']:9.3f}")
    print('[PROPOSALS] mean=',round(float(np.mean([d['proposals'] for d in diagnostics])),2),
          'zero=',sum(d['proposals']==0 for d in diagnostics))
    print('[SAVED]',save_dir)

if __name__=='__main__':
    main()
