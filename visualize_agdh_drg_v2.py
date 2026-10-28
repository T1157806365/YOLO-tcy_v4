#!/usr/bin/env python3
"""AGDH-DRG v2 visualization: prior maps, Q, cls/reg independent corrections.

Images show GT HBB on RGB semantic reference, not final predicted boxes.
For predicted HBB, use val_rlsfa_gtga_bhlr_v2.py --plots.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset
from engine.runtime import preprocess_batch
from models.gtga_bhlr_factory import load_model_from_checkpoint


def resize_map(x,h,w):
    y=F.interpolate(x.float(),size=(h,w),mode='bilinear',align_corners=False)
    return y[0,0].detach().cpu().numpy()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--split',default='val',choices=['train','val','test'])
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--save-dir',default='runs/agdh_drg_v2_visualize')
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    device=torch.device(a.device)
    model,ckpt,cfg=load_model_from_checkpoint(a.weights,device=device,strict=True,verbose=False)
    if not hasattr(model,'agdh_drg'):
        raise RuntimeError('Given checkpoint is not AGDH-DRG v2')
    d=cfg['data']
    dataset=build_dual_resolution_dataset(
        rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=a.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),
        rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),
        pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True,
    )
    if not 0<=a.index<len(dataset):
        raise IndexError(f'--index outside 0..{len(dataset)-1}')
    batch=dataset.collate_fn([dataset[a.index]])
    batch=preprocess_batch(batch,device)
    rgb_mask,tir_mask=model.build_alignment_valid_masks_from_batch(batch)
    with torch.inference_mode():
        out=model.predict(
            batch['rgb_img'],batch['tir_img'],
            rgb_semantic=batch['rgb_semantic_img'],
            rgb_valid_mask=rgb_mask,tir_valid_mask=tir_mask,
            return_drg_debug=True,return_features=True,
        )
    pri=out['alignment_info']['multi_target_guidance'].float()
    dbg=out['agdh_drg_debug']
    image=batch['rgb_semantic_img'][0].detach().float().cpu().permute(1,2,0).numpy()
    image=np.clip(image,0,1)
    h,w=image.shape[:2]
    vals=[
        ('RGB semantic + GT HBB',image),
        ('RGB support',resize_map(pri[:,0:1],h,w)),
        ('Aligned TIR support',resize_map(pri[:,1:2],h,w)),
        ('Joint support J',resize_map(pri[:,2:3],h,w)),
        ('Union U',resize_map(dbg['union'],h,w)),
        ('Reliability Q (one-to-one)',resize_map(dbg['quality_one'],h,w)),
        ('Cls logit delta (one-to-one)',resize_map(dbg['cls_one'],h,w)),
        ('Reg gain delta (one-to-one)',resize_map(dbg['reg_one'],h,w)),
    ]
    fig,axes=plt.subplots(2,4,figsize=(19,9.8),dpi=130)
    for idx,((title,mat),ax) in enumerate(zip(vals,axes.flat)):
        if idx==0:
            ax.imshow(mat)
            boxes=batch['rgb_semantic_bboxes'].detach().float().cpu().numpy()
            for cx,cy,bw,bh in boxes:
                ax.add_patch(patches.Rectangle(((cx-bw/2)*w,(cy-bh/2)*h),bw*w,bh*h,
                                               edgecolor='lime',facecolor='none',linewidth=1.5))
        else:
            ax.imshow(image)
            if idx in (6,7):
                vmax=float(np.abs(mat).max())
                vmax=max(vmax,1e-6)
                ax.imshow(mat,cmap='coolwarm',vmin=-vmax,vmax=vmax,alpha=.70)
            else:
                ax.imshow(mat,cmap='inferno',vmin=0,vmax=1,alpha=.65)
        ax.set_title(title,fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    target=Path(a.save_dir)
    target.mkdir(parents=True,exist_ok=True)
    path=target/f'agdh_drg_{a.split}_{a.index:04d}.png'
    fig.savefig(path,bbox_inches='tight')
    plt.close(fig)
    print('[AGDH-DRG v2] saved:',path)

if __name__=='__main__':main()
