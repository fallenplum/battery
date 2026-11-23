"""E0 full seven-method retrospective scoring on MICH/RWTH prefix caches."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd, torch
from scripts.drs_peft.run_head_aware_new_family_blind import load_future_from_pickles, load_prefix_role, score
from scripts.drs_peft.run_head_aware_retrospective_target import source_setup, train_head_method, train_lora_method, train_full_method
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, build_plan
from scripts.drs_peft.self_reference_training import full_prediction
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_reference import BACKBONE

METHODS = ("WARMUP_ONLY","LORA_12","HEAD_ONLY_TRAIN","HEAD_LORA","HEAD_BRIDGE_TRAIN","HEAD_BRIDGE_LORA","FULL_FT_TUNED")
WINDOWS = ((101,200),(201,500),(501,1000),(1001,5000))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--family",required=True)
    parser.add_argument("--prefix-cache",type=Path,required=True)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--source-private",type=Path,default=PRIVATE)
    parser.add_argument("--source-features",type=Path,default=PRIVATE)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--public",type=Path,required=True)
    parser.add_argument("--private",type=Path,required=True)
    parser.add_argument("--backbone-dir",type=Path,default=BACKBONE)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    configure()
    cfg,plan,*_=build_plan(args.config)
    device=torch.device(args.device)
    role=load_prefix_role(args.prefix_cache,device)
    ids=list(role.ids)
    args.public.mkdir(parents=True,exist_ok=True)
    args.private.mkdir(parents=True,exist_ok=True)
    lock={"status":"E0_LOCKED_BEFORE_TARGET_LABELS","family":args.family,"methods":list(METHODS),"target_ids":ids,"input_channels":12,"prefix_cache_sha256":sha(args.prefix_cache),"source_config_sha256":sha(args.config),"source_selection":"MATR fit/tune only","target_future_labels_read":0,"window_rule":[list(x) for x in WINDOWS]}
    (args.public/"E0_LOCK.json").write_text(json.dumps(lock,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    prediction_dir=args.private/"predictions"
    prediction_dir.mkdir(parents=True,exist_ok=True)
    ledger=[]
    for fold in plan["folds"]:
        setup_args=argparse.Namespace(features=args.source_features,private=args.source_private,data_root=args.data_root,backbone_dir=args.backbone_dir)
        fit,tune,common,common_path=source_setup(setup_args,fold,device)
        for method in METHODS:
            if method=="WARMUP_ONLY":
                model=common
                selected={"lr":0.0,"epoch":0,"tune_mae":None,"total_training_seconds":0.0,"max_gpu_peak_memory_bytes":0}
                count=0
            elif method=="LORA_12":
                model,selected,count=train_lora_method(common,fit,tune,cfg,int(fold["fold"]),device)
            elif method=="FULL_FT_TUNED":
                model,selected,count=train_full_method(common,fit,tune,cfg,int(fold["fold"]),device)
            else:
                model,selected,count,_=train_head_method(common,method,fit,tune,cfg,int(fold["fold"]),device)
            model.eval()
            prediction,inference_seconds=full_prediction(model,role,12,cfg["train_batch_size"],device)
            key=f"fold{fold['fold']}__{method}"
            path=prediction_dir/(key+".npz")
            np.savez_compressed(path,prediction=prediction,target_ids=np.asarray(ids))
            ledger.append({"fold":int(fold["fold"]),"method":method,"best_lr":float(selected["lr"]),"best_epoch":int(selected["epoch"]),"selected_tune_mae":selected.get("tune_mae"),"parameter_count":int(count),"training_seconds":float(selected.get("total_training_seconds",0.0)),"gpu_peak_memory_bytes":int(selected.get("max_gpu_peak_memory_bytes",0)),"inference_seconds":float(inference_seconds),"common_checkpoint_sha256":sha(common_path),"prediction_sha256":ah(prediction),"prediction_file_sha256":sha(path),"target_future_labels_read":0})
            if method!="WARMUP_ONLY":
                del model
        del common
    ledger_path=args.public/"E0_LEDGER.csv"
    pd.DataFrame(ledger).to_csv(ledger_path,index=False,lineterminator="\n")
    prediction_lock={"status":"E0_PREDICTIONS_LOCKED_BEFORE_TARGET_LABELS","family":args.family,"methods":list(METHODS),"target_ids":ids,"ledger_sha256":sha(ledger_path),"target_future_labels_read":0}
    prediction_lock_path=args.public/"E0_PREDICTIONS_LOCKED.json"
    prediction_lock_path.write_text(json.dumps(prediction_lock,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    values,observed=load_future_from_pickles(args.dataset_root,args.family,ids)
    score_rows=[]; cell_rows=[]; window_rows=[]
    for rec in ledger:
        prediction=np.load(prediction_dir/f"fold{rec['fold']}__{rec['method']}.npz",allow_pickle=False)["prediction"]
        mae,rmse,mse,cell_mae,cell_mse=score(prediction,values,observed)
        score_rows.append({"fold":rec["fold"],"method":rec["method"],"mae":mae,"rmse":rmse,"mse":mse,"cells":len(ids),"target_future_labels_read":1})
        for uid,x,y,n in zip(ids,cell_mae,cell_mse,observed.sum(1)):
            cell_rows.append({"fold":rec["fold"],"method":rec["method"],"cell_uid":uid,"mae":float(x),"mse":float(y),"observed_points":int(n)})
        for start,end in WINDOWS:
            sl=slice(start-101,end-100)
            mask=observed[:,sl]
            residual=np.where(mask,prediction[:,sl]-values[:,sl],0.0)
            pooled=residual[mask]
            window_rows.append({"fold":rec["fold"],"method":rec["method"],"window":f"{start}_{end}","mae":float(np.abs(pooled).mean()) if len(pooled) else np.nan,"rmse":float(np.sqrt(np.mean(pooled**2))) if len(pooled) else np.nan,"bias":float(pooled.mean()) if len(pooled) else np.nan,"cells":int((mask.sum(1)>0).sum()),"observed_points":int(mask.sum())})
    pd.DataFrame(score_rows).to_csv(args.public/"E0_SCORES.csv",index=False,lineterminator="\n")
    pd.DataFrame(cell_rows).to_csv(args.public/"E0_CELL.csv",index=False,lineterminator="\n")
    pd.DataFrame(window_rows).to_csv(args.public/"E0_WINDOW.csv",index=False,lineterminator="\n")
    pd.DataFrame(score_rows).groupby("method",as_index=False).agg(mae=("mae","mean"),rmse=("rmse","mean"),mse=("mse","mean"),folds=("fold","nunique"),cells=("cells","first")).to_csv(args.public/"E0_SUMMARY.csv",index=False,lineterminator="\n")
    receipt={"status":"COMPLETE_E0_FULL_MATRIX","family":args.family,"methods":list(METHODS),"target_future_labels_read":1,"prediction_lock_sha256":sha(prediction_lock_path),"files":{name:sha(args.public/name) for name in ("E0_LOCK.json","E0_PREDICTIONS_LOCKED.json","E0_LEDGER.csv","E0_SCORES.csv","E0_CELL.csv","E0_WINDOW.csv","E0_SUMMARY.csv")}}
    (args.public/"E0_RECEIPT.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(pd.read_csv(args.public/"E0_SUMMARY.csv").to_string(index=False))

if __name__=="__main__":
    main()

