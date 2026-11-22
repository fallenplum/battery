"""Replay source-selected Head-aware states and lock MATR target predictions before labels."""
from __future__ import annotations
import argparse, copy, hashlib, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from scripts.drs_peft.data import CACHE, file_hash
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import GRID, Role, full_prediction, load_common_state, role_bundle
from scripts.drs_peft.self_reference_features import build_self_reference_prefixes
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE
from scripts.drs_peft.train import configure
from scripts.drs_peft.run_head_bridge_aware_peft import METHODS, apply, install, train
from scripts.drs_peft.run_structured_peft_stage2 import install as install_lora, train_trial
from scripts.drs_peft.run_fair_fullft_tuned import train_full

TARGET_LOCK=PUBLIC.parent/'head_aware_target_v1'/'TARGET_LOCK.json'
TARGET_METHODS=("WARMUP_ONLY","LORA_12","HEAD_ONLY_TRAIN","HEAD_LORA","HEAD_BRIDGE_TRAIN","HEAD_BRIDGE_LORA","FULL_FT_TUNED")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_hash(state):
    h=hashlib.sha256()
    for n in sorted(state):
        h.update(n.encode()); h.update(state[n].detach().cpu().numpy().tobytes())
    return h.hexdigest()
def arr_hash(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def target_role(prefixes,ids,device):
    n=len(ids); q=len(GRID)
    return Role(list(ids),torch.as_tensor(prefixes.real_features,device=device),torch.as_tensor(prefixes.reference_features,device=device),torch.as_tensor(prefixes.feature_mask,device=device),torch.as_tensor(prefixes.cycles,device=device),torch.as_tensor(prefixes.real_anchor,device=device),np.zeros((n,q),np.float32),np.zeros((n,q),bool))

def source_setup(args,fold,device):
    ids=list(dict.fromkeys(fold['fit_ids']+fold['tune_ids']+fold['screen_ids']))
    bundle=load_bundle(args.features/'PAIRED_PREFIXES.npz',ids)
    target,observed=load_future_rows('MATR',args.data_root,ids)
    labels=Labels(target,observed,ids)
    fit=role_bundle(bundle,fold['fit_ids'],labels,device); tune=role_bundle(bundle,fold['tune_ids'],labels,device)
    common_path=args.private/f"COMMON_FOLD{fold['fold']}.pt"
    common=torch.load(common_path,map_location='cpu',weights_only=True)
    common_model=SharedBatteryModel(args.backbone_dir).to(device); load_common_state(common_model,common)
    return fit,tune,common_model,common_path

def train_head_method(common,method,fit,tune,cfg,fold,device):
    best_global=None; best_state=None
    for lr in cfg['adapter_rates']:
        model=copy.deepcopy(common); params,groups,count,ranks=install(model,method,cfg['seed']); best,state,trace=train(model,params,method,lr,fit,tune,cfg,fold,device)
        if best_global is None or best['tune_mae']<best_global['tune_mae'] or (best['tune_mae']==best_global['tune_mae'] and lr<best_global['lr']):
            best_global=best; best_state=state
        del model
    model=copy.deepcopy(common); params,groups,count,ranks=install(model,method,cfg['seed']); apply(params,best_state)
    return model,best_global,count,groups

def train_lora_method(common,fit,tune,cfg,fold,device):
    best_global=None; best_state=None
    for lr in cfg['adapter_rates']:
        model=copy.deepcopy(common); params,ranks,count=install_lora(model,'LORA_12',cfg['seed'])
        best,state,trace=train_trial(model,params,'LORA_12',lr,fit,tune,cfg,fold,device)
        if best_global is None or best['tune_mae']<best_global['tune_mae'] or (best['tune_mae']==best_global['tune_mae'] and lr<best_global['lr']): best_global=best; best_state=state
        del model
    model=copy.deepcopy(common); params,ranks,count=install_lora(model,'LORA_12',cfg['seed'])
    apply(params,best_state)
    return model,best_global,count

def train_full_method(common,fit,tune,cfg,fold,device):
    best_global=None; best_state=None
    for lr in cfg['adapter_rates']:
        best,state,trace=train_full(common,fit,tune,cfg,fold,lr,device)
        if best_global is None or best['tune_mae']<best_global['tune_mae'] or (best['tune_mae']==best_global['tune_mae'] and lr<best_global['lr']):
            best_global=best; best_state=state
    model=copy.deepcopy(common).to(device); model.requires_grad_(True)
    with torch.no_grad():
        named=dict(model.named_parameters())
        for n,v in best_state.items(): named[n].copy_(v)
    return model,best_global,sum(p.numel() for p in model.parameters())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--private',type=Path,default=PRIVATE); p.add_argument('--public',type=Path,default=PUBLIC.parent/'head_aware_target_v1'); p.add_argument('--features',type=Path,default=PRIVATE); p.add_argument('--data-root',type=Path,default=CACHE); p.add_argument('--backbone-dir',type=Path,default=BACKBONE); p.add_argument('--device',default='cuda'); p.add_argument('--domain',default='MATR',choices=['MATR','HUST']); p.add_argument('--task-id',default='MATR_POOLED'); args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True)
    lock=json.loads((args.public/'TARGET_LOCK.json').read_text(encoding='utf-8'))
    task=next(t for t in lock['target_tasks'] if t['task_id']==args.task_id and t['domain']==args.domain)
    target_ids=task['query_ids']; device=torch.device(args.device); cfg,plan,*_=build_plan(args.config); run_private=args.private/'head_aware_target_v1'/args.domain/args.task_id; run_private.mkdir(parents=True,exist_ok=True); target_public=args.public/args.domain/args.task_id; target_public.mkdir(parents=True,exist_ok=True)
    # Prefix-only target read. This path does not open targets or observed future masks.
    prefixes=build_self_reference_prefixes(args.domain,args.data_root,target_ids); target_role_obj=target_role(prefixes,target_ids,device)
    pred_rows=[]; state_rows=[]
    # Replay each source fold and lock one selected state per method. All source folds
    # are retained as separate prediction units for the retrospective audit.
    predictions={}
    for fold in plan['folds']:
        fit,tune,common,common_path=source_setup(args,fold,device)
        for method in TARGET_METHODS:
            if method=='WARMUP_ONLY':
                model=common; selected={'lr':0.0,'epoch':0,'tune_mae':None}; parameter_count=0
            elif method=='FULL_FT_TUNED':
                model,selected,parameter_count=train_full_method(common,fit,tune,cfg,fold['fold'],device)
            elif method=='LORA_12':
                model,selected,parameter_count=train_lora_method(common,fit,tune,cfg,fold['fold'],device)
            else:
                model,selected,parameter_count,groups=train_head_method(common,method,fit,tune,cfg,fold['fold'],device)
            model.eval(); pred,inf=full_prediction(model,target_role_obj,12,cfg['train_batch_size'],device); key=f"fold{fold['fold']}__{method}"; predictions[key]=pred
            pred_path=run_private/(key+'.npz'); np.savez_compressed(pred_path,prediction=pred,target_ids=np.asarray(target_ids)); state_rows.append({'fold':fold['fold'],'method':method,'best_lr':selected['lr'],'best_epoch':selected['epoch'],'selected_tune_mae':selected.get('tune_mae'),'parameter_count':parameter_count,'common_checkpoint_sha256':sha(common_path),'prediction_sha256':arr_hash(pred),'prediction_file_sha256':sha(pred_path),'target_future_labels_read':0,'outer_labels_read':0,'seed':cfg['seed']})
            if method != 'WARMUP_ONLY':
                del model
        del common
    state_frame=pd.DataFrame(state_rows); state_frame.to_csv(target_public/'SOURCE_STATE_PREDICTION_LEDGER.csv',index=False,lineterminator='\n')
    # Lock all prediction files before opening target future labels.
    lock_payload={'status':'PREDICTIONS_LOCKED_BEFORE_TARGET_SCORING','domain':args.domain,'task_id':args.task_id,'target_ids':target_ids,'methods':list(TARGET_METHODS),'source_state_ledger_sha256':sha(target_public/'SOURCE_STATE_PREDICTION_LEDGER.csv'),'target_future_labels_read':0,'outer_labels_read':0,'prediction_files_locked':True}
    (target_public/'PREDICTIONS_LOCKED.json').write_text(json.dumps(lock_payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='')
    # Future labels are opened only after the prediction lock exists.
    target_values,target_observed=load_future_rows(args.domain,args.data_root,target_ids,load_targets=True)
    rows=[]
    for rec in state_rows:
        pred=np.load(run_private/(f"fold{rec['fold']}__{rec['method']}.npz"),allow_pickle=False)['prediction']; diff=np.where(target_observed,pred-target_values,0.0); n=target_observed.sum(1); mae=np.abs(diff).sum(1)/n; mse=(diff**2).sum(1)/n
        rows.append({'fold':rec['fold'],'method':rec['method'],'mae':float(mae.mean()),'rmse':float(np.sqrt(mse).mean()),'mse':float(mse.mean()),'target_query_labels_used':1,'target_future_rows_loaded':len(target_ids)})
    pd.DataFrame(rows).to_csv(target_public/'TARGET_SCORES.csv',index=False,lineterminator='\n'); pd.DataFrame(rows).groupby('method',as_index=False).agg(mae=('mae','mean'),rmse=('rmse','mean'),mse=('mse','mean'),folds=('fold','nunique')).to_csv(target_public/'TARGET_SUMMARY.csv',index=False,lineterminator='\n')
    receipt={'status':'COMPLETE_HEAD_AWARE_RETROSPECTIVE_TARGET_SCORE','domain':args.domain,'task_id':args.task_id,'target_query_labels_used':1,'target_future_labels_read':1,'prediction_lock_sha256':sha(target_public/'PREDICTIONS_LOCKED.json'),'files':{n:sha(target_public/n) for n in ('SOURCE_STATE_PREDICTION_LEDGER.csv','PREDICTIONS_LOCKED.json','TARGET_SCORES.csv','TARGET_SUMMARY.csv')}}; (target_public/'TARGET_SCORE_RECEIPT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline=''); print(pd.read_csv(target_public/'TARGET_SUMMARY.csv').to_string(index=False))
if __name__=='__main__': main()
