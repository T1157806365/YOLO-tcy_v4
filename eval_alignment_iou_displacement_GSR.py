#!/usr/bin/env python3
"""GSR v3 full-geometry evaluator, RGB/TIR source boxes in normalized XYWH.

Evaluates complete independent GLOBAL+Fine+Scale, complete RG+Fine+Scale,
and the hard-selected final geometry, on the same pairs. GT is metrics only.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,build_dual_resolution_dataloader)
from engine.runtime import autocast_context,preprocess_batch,select_device
from models.gtga_bhlr_factory import load_model_from_checkpoint

def xyxy(b,w,h):
    x,y,bw,bh=map(float,b)
    return np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h],np.float32)
def box_iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]))
    ih=max(0,min(a[3],b[3])-max(a[1],b[1]))
    ia=iw*ih
    ua=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-ia
    return ia/ua if ua>0 else 0.
def unwarp(box,scale,translation):
    s=np.asarray(scale,dtype=np.float64).reshape(2).clip(1e-8)
    t=np.asarray(translation,dtype=np.float64).reshape(2)
    a=np.asarray(box,dtype=np.float64).reshape(4).copy()
    a[:2]=(a[:2]-t)/s; a[2:]/=s
    return a

def vec(x,i):
    return x[i].detach().float().reshape(-1)[:2].cpu().numpy()
def stat(vals):
    x=np.asarray(vals,dtype=np.float64)
    return {'mean':float(x.mean()),'median':float(np.median(x)),
            'p10':float(np.quantile(x,0.1)),
            'zero':int((x<=1e-12).sum()),'lt_0p1':int((x<0.1).sum()),
            'lt_0p3':int((x<0.3).sum())}
def panel(img,labels,boxes,vals,colors):
    images=[]
    for title,box,val,color in zip(labels,boxes,vals,colors):
        canvas=img.copy()
        if box is not None:
            xa,ya,xb,yb=np.asarray(box).round().astype(int)
            cv2.rectangle(canvas,(xa,ya),(xb,yb),color,2)
        cv2.putText(canvas,f'{title}: {val:.3f}',(14,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4)
        cv2.putText(canvas,f'{title}: {val:.3f}',(14,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        images.append(canvas)
    return np.concatenate(images,axis=1)

def get_ref_debug(info,model):
    s=model.alignment.reference_scale
    dbg=info['scale_debug']
    return dbg[s] if s in dbg else dbg[str(s)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',required=True)
    ap.add_argument('--split',default='test')
    ap.add_argument('--device',default='0')
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--amp',action='store_true')
    ap.add_argument('--rgb',default=None)
    ap.add_argument('--tir',default=None)
    ap.add_argument('--save-dir',default='runs/alignment_iou_GSRv3')
    ap.add_argument('--preview-num',type=int,default=30)
    args=ap.parse_args()
    device=select_device(args.device)
    model,_,cfg=load_model_from_checkpoint(args.weights,device=device,
                                           fusion_override=None,strict=True,verbose=False)
    model.eval()
    if not model.alignment.gsr_enabled:
        raise RuntimeError('This checkpoint is not GSR v3 (gsr_enabled=false).')
    data=cfg['data']
    ds=build_dual_resolution_dataset(
        rgb_yaml=args.rgb or data['rgb'],tir_yaml=args.tir or data['tir'],split=args.split,
        rgb_high_imgsz=int(data.get('rgb_imgsz',1920)),
        rgb_semantic_imgsz=int(data.get('rgb_semantic_imgsz',640)),
        tir_imgsz=int(data.get('tir_imgsz',640)),pair_mode=data.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(ds,batch_size=args.batch,workers=args.workers,
                                            shuffle=False,pin_memory=(device.type=='cuda'))
    save=Path(args.save_dir); save.mkdir(parents=True,exist_ok=True)
    panels=save/'boxed_images'; panels.mkdir(exist_ok=True)
    rows=[]; index=0
    with torch.no_grad():
        for batch in tqdm(loader,desc='GSR full geometry'):
            batch=preprocess_batch(batch,device)
            if hasattr(model,'build_alignment_valid_masks_from_batch'):
                rm,tm=model.build_alignment_valid_masks_from_batch(batch)
            else: rm=tm=None
            with autocast_context(device,enabled=args.amp):
                outputs=model.predict(batch['rgb_img'],batch['tir_img'],
                    rgb_semantic=batch['rgb_semantic_img'],rgb_valid_mask=rm,
                    tir_valid_mask=tm,return_features=True,return_alignment_debug=True)
            info=outputs['alignment_info']; dbg=get_ref_debug(info,model)
            gg=info['gsr_global_geometry']; rg=info['gsr_rg_geometry']
            score=info['gsr_path_scores']; route=info['gsr_route_rg']
            bsz=int(batch['rgb_semantic_img'].shape[0]); h,w=batch['rgb_semantic_img'].shape[-2:]
            for bi in range(bsz):
                rgb_boxes=batch['rgb_semantic_bboxes'][batch['rgb_batch_idx']==bi].detach().cpu().float().numpy()
                tir_boxes=batch['tir_bboxes'][batch['tir_batch_idx']==bi].detach().cpu().float().numpy()
                # UAV-CB one-to-one box metrics; prevent silently matching wrong boxes.
                if len(rgb_boxes)!=1 or len(tir_boxes)!=1: continue
                rb,tb=rgb_boxes[0],tir_boxes[0]
                rxy=xyxy(rb,w,h); raw=box_iou(rxy,xyxy(tb,w,h))
                gbox=unwarp(tb,vec(gg['sampling_scale'],bi),vec(gg['sampling_translation'],bi))
                rgbox=unwarp(tb,vec(rg['sampling_scale'],bi),vec(rg['sampling_translation'],bi))
                selbox=unwarp(tb,vec(info['sampling_scale'],bi),vec(info['sampling_translation'],bi))
                ig=box_iou(rxy,xyxy(gbox,w,h)); ir=box_iou(rxy,xyxy(rgbox,w,h))
                isel=box_iou(rxy,xyxy(selbox,w,h)); take=bool(route[bi].item())
                assert abs(isel-(ir if take else ig))<1e-5, 'Selected full geometry mismatch.'
                row={'index':index,'rgb_path':str(batch['rgb_path'][bi]),
                     'tir_path':str(batch['tir_path'][bi]),'iou_before':raw,
                     'iou_global_full':ig,'iou_rg_full':ir,'iou_after':isel,
                     'route_rg':int(take),'q_global':float(score[bi,0]),
                     'q_rg':float(score[bi,1]),'oracle_gap':max(ig,ir)-isel,
                     'rescued_global_zero':int(ig<=1e-12 and isel>1e-12),
                     'created_zero_vs_global':int(ig>1e-12 and isel<=1e-12),
                     'gain_selected_vs_global':isel-ig}
                rows.append(row)
                if index<args.preview_num or row['rescued_global_zero'] or row['created_zero_vs_global']:
                    img=batch['rgb_semantic_img'][bi].detach().float().cpu().numpy()
                    img=np.transpose(img,(1,2,0))
                    if img.max()<=1.5: img*=255.
                    img=np.clip(img,0,255).astype(np.uint8)[:,:,::-1].copy()
                    # RGB GT in green, staged TIR box in respective colors.
                    cv2.rectangle(img,tuple(rxy[:2].astype(int)),tuple(rxy[2:].astype(int)),(0,255,0),2)
                    titles=['RAW','GLOBAL+FINE+SCALE','RG+FINE+SCALE','ROUTED FINAL']
                    boxes=[xyxy(tb,w,h),xyxy(gbox,w,h),xyxy(rgbox,w,h),xyxy(selbox,w,h)]
                    vals=[raw,ig,ir,isel]
                    colors=[(0,0,255),(255,130,0),(220,0,220),(0,255,255)]
                    cv2.imwrite(str(panels/f'{index:04d}_{"rg" if take else "global"}.jpg'),
                                panel(img,titles,boxes,vals,colors))
                index+=1
    if not rows: raise RuntimeError('No unambiguous one-to-one bbox pairs available.')
    with (save/'alignment_pairs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys())); writer.writeheader();writer.writerows(rows)
    summary={'checkpoint':str(args.weights),'split':args.split,'paired_instances':len(rows),
             'stage_iou':{k:stat([r[k] for r in rows]) for k in
                 ('iou_before','iou_global_full','iou_rg_full','iou_after')},
             'routing':{'selected_rg':sum(r['route_rg'] for r in rows),
                        'selected_global':sum(1-r['route_rg'] for r in rows),
                        'rescued_global_zero':sum(r['rescued_global_zero'] for r in rows),
                        'created_zero_vs_global':sum(r['created_zero_vs_global'] for r in rows),
                        'mean_oracle_gap':float(np.mean([r['oracle_gap'] for r in rows]))}}
    (save/'alignment_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('GSR v3 full-geometry evaluation (not coarse diagnostics):')
    for k,v in summary['stage_iou'].items():
        print(f'{k:<22} mean={v["mean"]:.6f} median={v["median"]:.6f} zero={v["zero"]}')
    print('Routing:',summary['routing'])
    print('Files:',save.resolve())
if __name__=='__main__':main()
