"""Source-only functional ablation of locked Full-FT updates.

This is a teacher-update diagnostic. It performs inference from private fold
snapshots and writes only aggregate, per-cell, and receipt CSV/JSON artifacts.
No checkpoint tensors are copied into the public results directory.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import full_prediction, load_common_state, observed_mae, role_bundle
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE

EPOCH=8

def group(name:str)->str:
    low=name.lower()
    for tag in ("q_proj","k_proj","v_proj","out_proj","fc1","fc2","fc_gate"):
        if tag in low:return tag
    if "fusion12" in low:return "bridge_fusion12"
    if "fusion30" in low:return "bridge_fusion30"
    if "head" in low:return "head"
    return "other"

def is_encoder(name, indices):
    return any(f"encoder.layers.{i}." in name for i in indices)

def keep_delta(name, method):
    g=group(name)
    if method in {"WARMUP_ONLY_12"}: return False
    if method in {"FULL_FT_12"}: return True
    if method=="FFN_ONLY": return g in {"fc1","fc2","fc_gate"}
    if method=="ATTN_ONLY": return g in {"q_proj","k_proj","v_proj","out_proj"}
    if method=="FFN_ATTN": return g in {"fc1","fc2","fc_gate","q_proj","k_proj","v_proj","out_proj"}
    if method=="LATE3": return is_encoder(name,(3,4,5))
    if method=="EARLY3": return is_encoder(name,(0,1,2))
    if method=="HEAD_BRIDGE": return g in {"head","bridge_fusion12","bridge_fusion30"} or name.startswith("bridge.fusion")
    if method=="NO_FFN": return g not in {"fc1","fc2","fc_gate"}
    if method=="NO_ATTN": return g not in {"q_proj","k_proj","v_proj","out_proj"}
    if method=="NO_QK": return g not in {"q_proj","k_proj"}
    if method=="NO_VO": return g not in {"v_proj","out_proj"}
    if method=="NO_LAST_LAYER": return not is_encoder(name,(5,))
    if method=="NO_HEAD": return g!="head"
    if method=="NO_BRIDGE": return not (g in {"bridge_fusion12","bridge_fusion30"} or name.startswith("bridge.fusion"))
    raise ValueError(method)

METHODS=("WARMUP_ONLY_12","FULL_FT_12","FFN_ONLY","ATTN_ONLY","FFN_ATTN","LATE3","EARLY3","HEAD_BRIDGE","NO_FFN","NO_ATTN","NO_QK","NO_VO","NO_LAST_LAYER","NO_HEAD","NO_BRIDGE")

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_parameters(model, state):
    named=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in state.items(): named[name].copy_(value)

def metric_rows(prediction, role):
    mask=role.observed
    diff=np.where(mask, prediction-role.targets, 0.0)
    count=mask.sum(axis=1)
    mae=np.abs(diff).sum(axis=1)/count
    mse=(diff**2).sum(axis=1)/count
    rmse=np.sqrt(mse)
    pooled_mse=float(np.sum(diff**2)/np.sum(mask))
    pooled_rmse=float(np.sqrt(pooled_mse))
    return mae,mse,rmse,pooled_mse,pooled_rmse

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--private',type=Path,default=PRIVATE)
    p.add_argument('--public',type=Path,default=PUBLIC/'fullft_update_audit')
    p.add_argument('--features',type=Path,default=PRIVATE)
    p.add_argument('--data-root',type=Path,default=CACHE)
    p.add_argument('--backbone-dir',type=Path,default=BACKBONE)
    p.add_argument('--device',default='cuda')
    args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True)
    cfg,plan,*_=build_plan(args.config); device=torch.device(args.device)
    score_rows=[]; cell_rows=[]; checkpoint_hashes=[]
    for fold in plan['folds']:
        ids=list(dict.fromkeys(fold['fit_ids']+fold['tune_ids']+fold['screen_ids']))
        bundle=load_bundle(args.features/'PAIRED_PREFIXES.npz',ids)
        target,observed=load_future_rows('MATR',args.data_root,ids)
        labels=Labels(target,observed,ids)
        screen=role_bundle(bundle,fold['screen_ids'],labels,device)
        common_path=args.private/f"COMMON_FOLD{fold['fold']}.pt"
        common=torch.load(common_path,map_location='cpu',weights_only=True)
        root=args.private/f"fold{fold['fold']}"
        warm=torch.load(root/'FULL_FT_EPOCH0.pt',map_location='cpu',weights_only=True)['state']
        final=torch.load(root/f'FULL_FT_EPOCH{EPOCH}.pt',map_location='cpu',weights_only=True)['state']
        checkpoint_hashes += [{"fold":fold['fold'],"epoch":e,"file":f"FULL_FT_EPOCH{e}.pt","sha256":digest(root/f"FULL_FT_EPOCH{e}.pt")} for e in (0,EPOCH)]
        delta={name:final[name]-warm[name] for name in warm}
        model=SharedBatteryModel(args.backbone_dir).to(device)
        load_common_state(model,common); load_parameters(model,warm)
        for method in METHODS:
            state_start=time.perf_counter()
            if method=="WARMUP_ONLY_12":
                load_parameters(model,warm)
            else:
                state={name:(warm[name]+delta[name] if keep_delta(name,method) else warm[name]) for name in warm}
                load_parameters(model,state)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            prediction,seconds=full_prediction(model,screen,12,cfg['train_batch_size'],device)
            peak_memory_bytes = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            mae,mse,rmse,pooled_mse,pooled_rmse=metric_rows(prediction,screen)
            score_rows.append({"fold":fold['fold'],"method":method,"epoch":0 if method=="WARMUP_ONLY_12" else EPOCH,"screen_mae":float(mae.mean()),"screen_rmse":float(rmse.mean()),"screen_mse":float(mse.mean()),"screen_pooled_rmse":pooled_rmse,"screen_pooled_mse":pooled_mse,"screen_cells":len(screen.ids),"trainable_parameters":0,"teacher_update_parameters":13999369,"training_seconds":0.0,"inference_seconds":float(seconds),"state_apply_seconds":float(time.perf_counter()-state_start-seconds),"gpu_peak_memory_bytes":peak_memory_bytes,"gpu_peak_memory_mib":peak_memory_bytes/(1024**2),"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed']})
            for uid,cell_mae,cell_mse,cell_rmse in zip(screen.ids,mae,mse,rmse):
                cell_rows.append({"fold":fold['fold'],"method":method,"cell_uid":uid,"mae":float(cell_mae),"mse":float(cell_mse),"rmse":float(cell_rmse)})
        del model
    scores=pd.DataFrame(score_rows); cells=pd.DataFrame(cell_rows)
    scores.to_csv(args.public/'FUNCTIONAL_ABLATION_SCORES.csv',index=False,lineterminator='\n')
    cells.to_csv(args.public/'FUNCTIONAL_ABLATION_CELL.csv',index=False,lineterminator='\n')
    summary=scores.groupby('method',as_index=False).agg(screen_mae=('screen_mae','mean'),screen_rmse=('screen_rmse','mean'),screen_mse=('screen_mse','mean'),screen_pooled_rmse=('screen_pooled_rmse','mean'),screen_pooled_mse=('screen_pooled_mse','mean'),folds=('fold','nunique'),screen_cells=('screen_cells','sum'),inference_seconds=('inference_seconds','sum')).sort_values('screen_mae')
    summary.to_csv(args.public/'FUNCTIONAL_ABLATION_SUMMARY.csv',index=False,lineterminator='\n')
    receipt={"status":"COMPLETE_SOURCE_FULLFT_FUNCTIONAL_ABLATION","domain":"MATR","folds":[f['fold'] for f in plan['folds']],"methods":list(METHODS),"input_channels":12,"epoch":EPOCH,"source_roles":"fit-generated teacher updates, tune/screen evaluation","target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed'],"private_checkpoint_hashes":checkpoint_hashes,"files":{n:digest(args.public/n) for n in ('FUNCTIONAL_ABLATION_SCORES.csv','FUNCTIONAL_ABLATION_CELL.csv','FUNCTIONAL_ABLATION_SUMMARY.csv')}}
    (args.public/'FUNCTIONAL_ABLATION_COMPLETE.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='')
    print(summary.to_string(index=False))
if __name__=='__main__': main()
