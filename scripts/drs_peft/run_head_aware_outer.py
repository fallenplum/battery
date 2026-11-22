"""Retrospective source-outer validation for Head-aware PEFT."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from scripts.drs_peft.data import CACHE, ROOT
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, build_plan
from scripts.drs_peft.self_reference_features import build_self_reference_prefixes
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE
from scripts.drs_peft.train import configure
from scripts.drs_peft.run_head_aware_retrospective_target import (
    TARGET_METHODS, source_setup, target_role, train_head_method, train_full_method,
)
from scripts.drs_peft.self_reference_training import full_prediction

METHODS=TARGET_METHODS

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr_hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def score(pred, values, observed):
    diff=np.where(observed,pred-values,0.0); n=observed.sum(1)
    mae=np.abs(diff).sum(1)/n; mse=(diff**2).sum(1)/n
    return float(mae.mean()),float(np.sqrt(mse).mean()),float(mse.mean()),mae,mse

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--private',type=Path,default=PRIVATE); p.add_argument('--public',type=Path,default=ROOT/'results/drs_peft/head_aware_outer_v1'); p.add_argument('--features',type=Path,default=PRIVATE); p.add_argument('--data-root',type=Path,default=CACHE); p.add_argument('--backbone-dir',type=Path,default=BACKBONE); p.add_argument('--device',default='cuda'); args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True)
    e123=load_json(ROOT/'configs/drs_peft/e123_folds_v1.json'); cfg,source_plan,*_=build_plan(args.config); device=torch.device(args.device)
    outer_folds=[{"fold":int(f["fold_id"]),"ids":list(f["outer_test_ids"])} for f in e123["folds"]]
    outer_lock={"status":"HEAD_AWARE_RETROSPECTIVE_SOURCE_OUTER_LOCKED_BEFORE_PREDICTION","historical_exposure":True,"source_outer_e123_sha256":sha(ROOT/'configs/drs_peft/e123_folds_v1.json'),"source_protocol_config_sha256":sha(args.config),"methods":list(METHODS),"seed":cfg["seed"],"input_channels":12,"target_future_labels_read":0,"outer_labels_read":0,"folds":outer_folds,"prediction_lock_required_before_scoring":True}
    (args.public/'OUTER_LOCK.json').write_text(json.dumps(outer_lock,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='')
    pred_private=args.private/'head_aware_outer_v1'; pred_private.mkdir(parents=True,exist_ok=True)
    ledger=[]; rows=[]; cells=[]
    for fold,source_fold in zip(outer_folds,source_plan['folds']):
        outer_ids=fold['ids']; prefixes=build_self_reference_prefixes('MATR',args.data_root,outer_ids); role=target_role(prefixes,outer_ids,device)
        # Prefix-only construction. Future rows remain unopened until after lock.
        fit,tune,common,common_path=source_setup(args,source_fold,device)
        for method in METHODS:
            if method=='WARMUP_ONLY':
                model=common; selected={'lr':0.0,'epoch':0,'tune_mae':None}; parameter_count=0
            elif method=='FULL_FT_TUNED':
                model,selected,parameter_count=train_full_method(common,fit,tune,cfg,source_fold['fold'],device)
            else:
                model,selected,parameter_count,groups=train_head_method(common,method,fit,tune,cfg,source_fold['fold'],device)
            model.eval(); pred,_=full_prediction(model,role,12,cfg['train_batch_size'],device)
            key=f"fold{fold['fold']}__{method}"; pred_path=pred_private/(key+'.npz'); np.savez_compressed(pred_path,prediction=pred,outer_ids=np.asarray(outer_ids))
            ledger.append({"fold":fold['fold'],"method":method,"outer_ids":outer_ids,"best_lr":selected['lr'],"best_epoch":selected['epoch'],"selected_tune_mae":selected.get('tune_mae'),"parameter_count":parameter_count,"common_checkpoint_sha256":sha(common_path),"prediction_sha256":arr_hash(pred),"prediction_file_sha256":sha(pred_path),"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed']})
            if method!='WARMUP_ONLY': del model
        del common
    ledger_path=args.public/'OUTER_STATE_PREDICTION_LEDGER.csv'; pd.DataFrame(ledger).drop(columns=['outer_ids']).to_csv(ledger_path,index=False,lineterminator='\n')
    pred_lock={"status":"OUTER_PREDICTIONS_LOCKED_BEFORE_SCORING","source_outer_e123_sha256":sha(ROOT/'configs/drs_peft/e123_folds_v1.json'),"outer_folds":outer_folds,"methods":list(METHODS),"ledger_sha256":sha(ledger_path),"target_future_labels_read":0,"outer_labels_read":0,"prediction_files_locked":True}
    pred_lock_path=args.public/'OUTER_PREDICTIONS_LOCKED.json'; pred_lock_path.write_text(json.dumps(pred_lock,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='')
    # Outer labels are opened only after prediction lock.
    for rec in ledger:
        values,observed=load_future_rows('MATR',args.data_root,rec['outer_ids'],load_targets=True)
        pred=np.load(pred_private/f"fold{rec['fold']}__{rec['method']}.npz",allow_pickle=False)['prediction']; mae,rmse,mse,cm,cs=score(pred,values,observed)
        rows.append({"fold":rec['fold'],"method":rec['method'],"mae":mae,"rmse":rmse,"mse":mse,"outer_cells":len(rec['outer_ids']),"outer_labels_read":1})
        for uid,x,y in zip(rec['outer_ids'],cm,cs): cells.append({"fold":rec['fold'],"method":rec['method'],"cell_uid":uid,"mae":float(x),"mse":float(y)})
    frame=pd.DataFrame(rows); frame.to_csv(args.public/'OUTER_SCORES.csv',index=False,lineterminator='\n'); pd.DataFrame(cells).to_csv(args.public/'OUTER_CELL.csv',index=False,lineterminator='\n'); frame.groupby('method',as_index=False).agg(mae=('mae','mean'),rmse=('rmse','mean'),mse=('mse','mean'),folds=('fold','nunique'),outer_cells=('outer_cells','sum')).to_csv(args.public/'OUTER_SUMMARY.csv',index=False,lineterminator='\n')
    receipt={"status":"COMPLETE_HEAD_AWARE_RETROSPECTIVE_SOURCE_OUTER_SCORE","historical_exposure":True,"target_future_labels_read":0,"outer_labels_read":1,"prediction_lock_sha256":sha(pred_lock_path),"files":{n:sha(args.public/n) for n in ('OUTER_LOCK.json','OUTER_PREDICTIONS_LOCKED.json','OUTER_STATE_PREDICTION_LEDGER.csv','OUTER_SCORES.csv','OUTER_CELL.csv','OUTER_SUMMARY.csv')}}; (args.public/'OUTER_SCORE_RECEIPT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline=''); print(pd.read_csv(args.public/'OUTER_SUMMARY.csv').to_string(index=False))
if __name__=='__main__': main()
