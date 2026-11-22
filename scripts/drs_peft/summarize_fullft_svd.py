"""Quantify low-rank reconstruction of source-only Full-FT updates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from scripts.drs_peft.self_reference_protocol import PRIVATE, PUBLIC

FOLDS=(0,1,2)
EPOCH=8
RANKS=(1,2,4,8,16,32,64,128)

def group(name):
    low=name.lower()
    for tag in ("q_proj","k_proj","v_proj","out_proj","fc1","fc2","fc_gate"):
        if tag in low:return tag
    if "fusion12" in low:return "bridge_fusion12"
    if "fusion30" in low:return "bridge_fusion30"
    if "head" in low:return "head"
    return "other"

def layer(name):
    marker="encoder.layers."
    if marker in name:
        return "encoder.layer."+name.split(marker,1)[1].split('.',1)[0]
    if name.startswith("fusion12"):return "bridge.fusion12"
    if name.startswith("fusion30"):return "bridge.fusion30"
    if name.startswith("head"):return "head"
    return "other"

def facts(delta):
    x=delta.detach().cpu().float()
    if x.ndim!=2:
        return None
    s=torch.linalg.svdvals(x.double())
    total=float(s.square().sum())
    if total==0:return None
    captured=torch.cumsum(s.square(),0)/total
    out={"frobenius":float(total**0.5),"rank90":int(torch.searchsorted(captured,0.90).item()+1),"rank95":int(torch.searchsorted(captured,0.95).item()+1),"rank99":int(torch.searchsorted(captured,0.99).item()+1),"numerical_rank":int((s>s.max()*1e-5).sum())}
    for r in RANKS:
        k=min(r,len(s)); out[f"residual_rank{r}"]=float(1-captured[k-1])
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--private',type=Path,default=PRIVATE); p.add_argument('--public',type=Path,default=PUBLIC/'fullft_update_audit'); a=p.parse_args(); a.public.mkdir(parents=True,exist_ok=True)
    rows=[]
    for f in FOLDS:
        root=a.private/f'fold{f}'
        warm=torch.load(root/'FULL_FT_EPOCH0.pt',map_location='cpu',weights_only=True)['state']
        final=torch.load(root/f'FULL_FT_EPOCH{EPOCH}.pt',map_location='cpu',weights_only=True)['state']
        for name in warm:
            if warm[name].ndim!=2: continue
            z=facts(final[name]-warm[name])
            if z is None: continue
            rows.append({"fold":f,"parameter":name,"group":group(name),"layer":layer(name),**z})
    frame=pd.DataFrame(rows)
    frame.to_csv(a.public/'UPDATE_SVD_RECONSTRUCTION.csv',index=False,lineterminator='\n')
    # Energy-weighted residual is more interpretable than an unweighted mean.
    summaries=[]
    for key in frame.group.unique():
        scope = "group"
        sub=frame[frame.group.eq(key)];
        total=(sub.frobenius**2).sum()
        rec={"scope":scope,"key":key,"parameters":len(sub),"total_energy":float(total),"mean_rank90":float(sub.rank90.mean()),"mean_rank95":float(sub.rank95.mean()),"mean_rank99":float(sub.rank99.mean())}
        for r in RANKS: rec[f"energy_residual_rank{r}"]=max(0.0, float((sub.frobenius**2*sub[f"residual_rank{r}"]).sum()/total))
        summaries.append(rec)
    for key in sorted(frame.layer.unique()):
        sub=frame[frame.layer.eq(key)]; total=(sub.frobenius**2).sum(); rec={"scope":"layer","key":key,"parameters":len(sub),"total_energy":float(total),"mean_rank90":float(sub.rank90.mean()),"mean_rank95":float(sub.rank95.mean()),"mean_rank99":float(sub.rank99.mean())}
        for r in RANKS: rec[f"energy_residual_rank{r}"]=max(0.0, float((sub.frobenius**2*sub[f"residual_rank{r}"]).sum()/total))
        summaries.append(rec)
    allsub=frame; total=(allsub.frobenius**2).sum(); rec={"scope":"all_2d","key":"all","parameters":len(allsub),"total_energy":float(total),"mean_rank90":float(allsub.rank90.mean()),"mean_rank95":float(allsub.rank95.mean()),"mean_rank99":float(allsub.rank99.mean())}
    for r in RANKS: rec[f"energy_residual_rank{r}"]=max(0.0, float((allsub.frobenius**2*allsub[f"residual_rank{r}"]).sum()/total))
    summaries.append(rec)
    sf=pd.DataFrame(summaries); sf.to_csv(a.public/'UPDATE_SVD_GROUP_SUMMARY.csv',index=False,lineterminator='\n')
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    (a.public/'SVD_COMPLETE.json').write_text(json.dumps({"status":"COMPLETE_FULL_FT_SVD_RECONSTRUCTION","folds":list(FOLDS),"epoch":EPOCH,"ranks":list(RANKS),"source_only":True,"target_future_labels_read":0,"outer_labels_read":0,"files":{n:sha(a.public/n) for n in ('UPDATE_SVD_RECONSTRUCTION.csv','UPDATE_SVD_GROUP_SUMMARY.csv')}},indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='')
    print(sf[sf.scope.eq('all_2d')].to_string(index=False))
if __name__=='__main__': main()
