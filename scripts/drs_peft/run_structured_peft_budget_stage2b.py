"""Stage 2b budget ladder for the selected late-layer structural PEFT."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from models.drs_peft.abba_adapters import LoRALinear
from models.drs_peft.onesided import replace_layer
from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import batches, full_prediction, load_common_state, role_bundle, training_targets
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE
from scripts.drs_peft.run_structured_peft_stage2 import EPOCHS, CHECKPOINT_EPOCHS, metrics, rank_map, train_trial

BUDGETS={"BUDGET_36K":36864,"BUDGET_140K":131328,"BUDGET_500K":503424}

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def custom_ranks(budget,names):
    if budget=="BUDGET_36K":
        return rank_map("LATE_LAYER_GROUP", names)
    ranks={n:0 for n in names}; late=[n for n in sorted(names) if any(f"encoder.layers.{i}." in n for i in (3,4,5))]
    rank=4 if budget=="BUDGET_36K" else (6 if budget=="BUDGET_140K" else 23)
    for n in late: ranks[n]=rank
    return ranks

def install_custom(model,budget,seed):
    base=model.bridge.backbone.linear_targets("qkvo_ffn"); ranks=custom_ranks(budget,list(base)); model.requires_grad_(False)
    for name,layer in base.items():
        rank=ranks[name]
        if rank<=0: continue
        adapter=LoRALinear(layer,rank,alpha=rank,dropout=0.0,seed=seed)
        for pn,p in adapter.named_parameters(): p.requires_grad_(not pn.startswith("base."))
        replace_layer(model.backbone,name,adapter)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}; count=sum(p.numel() for p in params.values())
    return params,ranks,count

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--private',type=Path,default=PRIVATE); p.add_argument('--public',type=Path,default=PUBLIC/'structured_peft_budget_v1'); p.add_argument('--features',type=Path,default=PRIVATE); p.add_argument('--data-root',type=Path,default=CACHE); p.add_argument('--backbone-dir',type=Path,default=BACKBONE); p.add_argument('--device',default='cuda'); args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True)
    cfg,plan,*_=build_plan(args.config); device=torch.device(args.device); scores=[]; traces=[]; cells=[]; budgets=[]
    for fold in plan['folds']:
        ids=list(dict.fromkeys(fold['fit_ids']+fold['tune_ids']+fold['screen_ids'])); bundle=load_bundle(args.features/'PAIRED_PREFIXES.npz',ids); target,observed=load_future_rows('MATR',args.data_root,ids); labels=Labels(target,observed,ids)
        fit=role_bundle(bundle,fold['fit_ids'],labels,device); tune=role_bundle(bundle,fold['tune_ids'],labels,device); screen=role_bundle(bundle,fold['screen_ids'],labels,device)
        common_path=args.private/f"COMMON_FOLD{fold['fold']}.pt"; common=torch.load(common_path,map_location='cpu',weights_only=True); common_model=SharedBatteryModel(args.backbone_dir).to(device); load_common_state(common_model,common); common_hash=digest(common_path)
        for budget,target_count in BUDGETS.items():
            best_global=None; best_state=None; rank_report=None
            for lr in cfg['adapter_rates']:
                model=copy.deepcopy(common_model); params,ranks,count=install_custom(model,budget,cfg['seed']); rank_report=ranks
                if abs(count-target_count)>512: raise RuntimeError(f"{budget} count {count} != {target_count}")
                best,state,trace=train_trial(model,params,budget,lr,fit,tune,cfg,fold['fold'],device); traces.extend(trace)
                if best_global is None or best['tune_mae']<best_global['tune_mae'] or (best['tune_mae']==best_global['tune_mae'] and lr<best_global['lr']): best_global=best; best_state=state
                del model
            final=copy.deepcopy(common_model); final_params,_,count=install_custom(final,budget,cfg['seed']);
            with torch.no_grad():
                for name,value in best_state.items(): final_params[name].copy_(value)
            pred,inf=full_prediction(final,screen,12,cfg['train_batch_size'],device); mae,rmse,mse,cell_mae,cell_mse=metrics(pred,screen)
            scores.append({"fold":fold['fold'],"budget":budget,"requested_parameters":target_count,"trainable_parameters":count,"best_lr":best_global['lr'],"best_epoch":best_global['epoch'],"screen_mae":mae,"screen_rmse":rmse,"screen_mse":mse,"screen_cells":len(screen.ids),"training_seconds":best_global['total_training_seconds'],"screen_inference_seconds":inf,"max_gpu_peak_memory_bytes":best_global['max_gpu_peak_memory_bytes'],"max_gpu_peak_memory_mib":best_global['max_gpu_peak_memory_bytes']/(1024**2),"common_checkpoint_sha256":common_hash,"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed']})
            for uid,x,y in zip(screen.ids,cell_mae,cell_mse): cells.append({"fold":fold['fold'],"budget":budget,"cell_uid":uid,"mae":float(x),"mse":float(y)})
            budgets.append({"budget":budget,"requested_parameters":target_count,"trainable_parameters":count,"rank":next(iter(set(ranks.values())))})
            del final
        del common_model
    sf=pd.DataFrame(scores); tf=pd.DataFrame(traces); cf=pd.DataFrame(cells); bf=pd.DataFrame(budgets).drop_duplicates('budget')
    sf.to_csv(args.public/'STRUCTURED_PEFT_BUDGET_SCORES.csv',index=False,lineterminator='\n'); tf.to_csv(args.public/'STRUCTURED_PEFT_BUDGET_TRACE.csv',index=False,lineterminator='\n'); cf.to_csv(args.public/'STRUCTURED_PEFT_BUDGET_CELL.csv',index=False,lineterminator='\n'); bf.to_csv(args.public/'STRUCTURED_PEFT_BUDGETS.csv',index=False,lineterminator='\n')
    summary=sf.groupby('budget',as_index=False).agg(screen_mae=('screen_mae','mean'),screen_rmse=('screen_rmse','mean'),screen_mse=('screen_mse','mean'),folds=('fold','nunique'),screen_cells=('screen_cells','sum'),trainable_parameters=('trainable_parameters','first'),best_epoch=('best_epoch','median'),best_lr=('best_lr','first')).sort_values('trainable_parameters'); summary.to_csv(args.public/'STRUCTURED_PEFT_BUDGET_SUMMARY.csv',index=False,lineterminator='\n')
    receipt={"status":"COMPLETE_SOURCE_STRUCTURED_PEFT_STAGE2B","domain":"MATR","folds":[f['fold'] for f in plan['folds']],"budgets":BUDGETS,"selected_structure":"LATE_LAYER_GROUP","input_channels":12,"optimizer":"AdamW","epochs":EPOCHS,"checkpoint_epochs":list(CHECKPOINT_EPOCHS),"learning_rates":cfg['adapter_rates'],"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed'],"files":{n:digest(args.public/n) for n in ('STRUCTURED_PEFT_BUDGET_SCORES.csv','STRUCTURED_PEFT_BUDGET_TRACE.csv','STRUCTURED_PEFT_BUDGET_CELL.csv','STRUCTURED_PEFT_BUDGETS.csv','STRUCTURED_PEFT_BUDGET_SUMMARY.csv')}}; (args.public/'STRUCTURED_PEFT_BUDGET_COMPLETE.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline=''); print(summary.to_string(index=False))
if __name__=='__main__': main()
