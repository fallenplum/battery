"""Stage 2 budget-matched structural LoRA source screen."""
from __future__ import annotations
import argparse, copy, hashlib, json, time
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
from scripts.drs_peft.self_reference_training import batches, full_prediction, load_common_state, masked_cell_loss, observed_mae, role_bundle, training_targets
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE

EPOCHS=8
CHECKPOINT_EPOCHS=(2,4,8)
METHODS=("LORA_12","FFN_ONLY","ATTN_ONLY","FFN_ATTN_ALL","LATE_LAYER_GROUP","GROUPWISE_RANK")

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def targets(model): return model.bridge.backbone.linear_targets("qkvo_ffn")

def rank_map(method, names):
    ordered=sorted(names)
    ranks={name:0 for name in names}
    if method=="LORA_12":
        for n in names:
            if n.endswith("self_attn.q_proj") or n.endswith("self_attn.v_proj"): ranks[n]=4
    elif method=="FFN_ONLY":
        ffn = [n for n in ordered if ".ffn." in n]
        for n in ffn: ranks[n]=1
        extra = [n for n in ffn if "encoder.layers.4." in n or "encoder.layers.5." in n]
        extra += [n for n in ffn if "encoder.layers.3." in n and (n.endswith("ffn.fc1") or n.endswith("ffn.fc2"))]
        for n in extra[:8]: ranks[n]=2
    elif method=="ATTN_ONLY":
        for n in names: ranks[n]=2
        ranks={n:r for n,r in ranks.items() if ".self_attn." in n}
    elif method=="FFN_ATTN_ALL":
        for n in names:
            if n.endswith("ffn.fc1") or n.endswith("ffn.fc2") or ".self_attn." in n:
                ranks[n]=1
        late=[n for n in ordered if "encoder.layers.5." in n and ".self_attn." in n]
        for n in late[:2]: ranks[n]=2
    elif method=="LATE_LAYER_GROUP":
        for n in names:
            if any(f"encoder.layers.{i}." in n for i in (3,4,5)): ranks[n]=2
        # Start every late-layer site at rank one, then spend the remaining
        # budget on three FFN rank-2 sites and seven attention rank-3 sites.
        for n in names:
            ranks[n] = 1 if any(f"encoder.layers.{i}." in n for i in (3,4,5)) else 0
        ffn_extra=[n for n in ordered if "encoder.layers.5." in n and ".ffn." in n][:3]
        attn_extra=[n for n in ordered if any(f"encoder.layers.{i}." in n for i in (3,4,5)) and ".self_attn." in n][:7]
        for n in ffn_extra: ranks[n]=2
        for n in attn_extra: ranks[n]=3
    elif method=="GROUPWISE_RANK":
        for n in names:
            if ".ffn." in n: ranks[n]=1
            elif n.endswith("self_attn.q_proj") or n.endswith("self_attn.v_proj"): ranks[n]=1
        attn_extra=[n for n in ordered if (n.endswith("self_attn.q_proj") or n.endswith("self_attn.v_proj"))][:3]
        for n in attn_extra: ranks[n]=2
    else: raise ValueError(method)
    return ranks

def install(model, method, seed):
    base=targets(model); ranks=rank_map(method,list(base))
    model.requires_grad_(False)
    report={}
    for name,layer in base.items():
        rank=ranks.get(name,0)
        if rank<=0: continue
        adapter=LoRALinear(layer,rank,alpha=rank,dropout=0.0,seed=seed)
        for pn,p in adapter.named_parameters(): p.requires_grad_(not pn.startswith("base."))
        replace_layer(model.backbone,name,adapter)
        report[name]=rank
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    count=sum(p.numel() for p in params.values())
    return params,report,count

def apply_state(params,state):
    with torch.no_grad():
        for name,value in state.items(): params[name].copy_(value)

def metrics(pred,role):
    mask=role.observed; diff=np.where(mask,pred-role.targets,0.0); n=mask.sum(1)
    mae=np.abs(diff).sum(1)/n; mse=(diff**2).sum(1)/n
    return float(mae.mean()),float(np.sqrt(mse).mean()),float(mse.mean()),mae,mse

def state_hash(state):
    h=hashlib.sha256()
    for n in sorted(state): h.update(n.encode()); h.update(state[n].numpy().tobytes())
    return h.hexdigest()

def train_trial(model,params,method,lr,fit,tune,cfg,fold,device):
    values=list(params.values()); opt=torch.optim.AdamW(values,lr=lr,weight_decay=0.0)
    trace=[]; best=None; started_all=time.perf_counter(); peak=0
    for epoch in range(1,EPOCHS+1):
        model.train(); started=time.perf_counter(); total=0.0; steps=0
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        for index in batches(len(fit.ids),cfg['train_batch_size'],seed=cfg['seed']+1009*epoch+fold):
            cycles,y,mask=training_targets(fit,index,device); opt.zero_grad(set_to_none=True)
            pred=model(*fit.arrays(index),cycles,channels=12); loss=masked_cell_loss(pred,y,mask,kind='mae'); loss.backward()
            torch.nn.utils.clip_grad_norm_(values,1.0,error_if_nonfinite=True); opt.step()
            total+=float(loss.detach())*len(index); steps+=1
        model.eval(); pred,inf=full_prediction(model,tune,12,cfg['train_batch_size'],device); tune_mae=float(observed_mae(pred,tune).mean())
        p=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0; peak=max(peak,p)
        row={"fold":fold,"method":method,"lr":lr,"epoch":epoch,"training_mae":total/len(fit.ids),"tune_mae":tune_mae,"training_seconds":time.perf_counter()-started,"tune_inference_seconds":inf,"optimizer_updates":steps,"gpu_peak_memory_bytes":p,"trainable_parameters":sum(p.numel() for p in values)}
        trace.append(row)
        if epoch in CHECKPOINT_EPOCHS and (best is None or tune_mae<best['tune_mae']):
            best=row; best_state={n:p.detach().cpu().clone() for n,p in params.items()}
    best=best|{"total_training_seconds":time.perf_counter()-started_all,"max_gpu_peak_memory_bytes":peak,"adapter_state_sha256":state_hash(best_state)}
    return best,best_state,trace

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--private',type=Path,default=PRIVATE); p.add_argument('--public',type=Path,default=PUBLIC/'structured_peft_v1'); p.add_argument('--features',type=Path,default=PRIVATE); p.add_argument('--data-root',type=Path,default=CACHE); p.add_argument('--backbone-dir',type=Path,default=BACKBONE); p.add_argument('--device',default='cuda'); args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True)
    cfg,plan,*_=build_plan(args.config); device=torch.device(args.device); score_rows=[]; trace_rows=[]; cell_rows=[]; budget_rows=[]
    for fold in plan['folds']:
        ids=list(dict.fromkeys(fold['fit_ids']+fold['tune_ids']+fold['screen_ids'])); bundle=load_bundle(args.features/'PAIRED_PREFIXES.npz',ids); target,observed=load_future_rows('MATR',args.data_root,ids); labels=Labels(target,observed,ids)
        fit=role_bundle(bundle,fold['fit_ids'],labels,device); tune=role_bundle(bundle,fold['tune_ids'],labels,device); screen=role_bundle(bundle,fold['screen_ids'],labels,device)
        common_path=args.private/f"COMMON_FOLD{fold['fold']}.pt"; common=torch.load(common_path,map_location='cpu',weights_only=True); common_model=SharedBatteryModel(args.backbone_dir).to(device); load_common_state(common_model,common)
        common_hash=digest(common_path)
        base_targets=targets(common_model)
        for method in METHODS:
            best_global=None; best_global_state=None; best_global_trace=None; rank_report=None
            for lr in cfg['adapter_rates']:
                model=copy.deepcopy(common_model); params,ranks,count=install(model,method,cfg['seed']); rank_report=ranks
                if abs(count-cfg['expected_adapter_parameters'])>512: raise RuntimeError(f"{method} budget {count} differs from {cfg['expected_adapter_parameters']}")
                best,state,trace=train_trial(model,params,method,lr,fit,tune,cfg,fold['fold'],device); trace_rows.extend(trace)
                if best_global is None or best['tune_mae']<best_global['tune_mae'] or (best['tune_mae']==best_global['tune_mae'] and lr<best_global['lr']): best_global=best; best_global_state=state; best_global_trace=trace
                del model
            final=copy.deepcopy(common_model); final_params,_,count=install(final,method,cfg['seed']); apply_state(final_params,best_global_state); final.eval(); pred,inf=full_prediction(final,screen,12,cfg['train_batch_size'],device); mae,rmse,mse,cell_mae,cell_mse=metrics(pred,screen)
            score_rows.append({"fold":fold['fold'],"method":method,"best_lr":best_global['lr'],"best_epoch":best_global['epoch'],"screen_mae":mae,"screen_rmse":rmse,"screen_mse":mse,"screen_cells":len(screen.ids),"trainable_parameters":count,"training_seconds":best_global['total_training_seconds'],"screen_inference_seconds":inf,"max_gpu_peak_memory_bytes":best_global['max_gpu_peak_memory_bytes'],"max_gpu_peak_memory_mib":best_global['max_gpu_peak_memory_bytes']/(1024**2),"common_checkpoint_sha256":common_hash,"adapter_state_sha256":best_global['adapter_state_sha256'],"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed']})
            for uid,x,y in zip(screen.ids,cell_mae,cell_mse): cell_rows.append({"fold":fold['fold'],"method":method,"cell_uid":uid,"mae":float(x),"mse":float(y)})
            budget_rows.append({"method":method,"trainable_parameters":count,"rank_schedule":json.dumps(rank_report,sort_keys=True)})
            del final
        del common_model
    scores=pd.DataFrame(score_rows); traces=pd.DataFrame(trace_rows); cells=pd.DataFrame(cell_rows); budgets=pd.DataFrame(budget_rows).drop_duplicates('method')
    scores.to_csv(args.public/'STRUCTURED_PEFT_SCORES.csv',index=False,lineterminator='\n'); traces.to_csv(args.public/'STRUCTURED_PEFT_TRACE.csv',index=False,lineterminator='\n'); cells.to_csv(args.public/'STRUCTURED_PEFT_CELL.csv',index=False,lineterminator='\n'); budgets.to_csv(args.public/'STRUCTURED_PEFT_BUDGETS.csv',index=False,lineterminator='\n')
    summary=scores.groupby('method',as_index=False).agg(screen_mae=('screen_mae','mean'),screen_rmse=('screen_rmse','mean'),screen_mse=('screen_mse','mean'),folds=('fold','nunique'),screen_cells=('screen_cells','sum'),trainable_parameters=('trainable_parameters','first'),best_epoch=('best_epoch','median'),best_lr=('best_lr','first')).sort_values('screen_mae'); summary.to_csv(args.public/'STRUCTURED_PEFT_SUMMARY.csv',index=False,lineterminator='\n')
    receipt={"status":"COMPLETE_SOURCE_STRUCTURED_PEFT_STAGE2","domain":"MATR","folds":[f['fold'] for f in plan['folds']],"methods":list(METHODS),"input_channels":12,"optimizer":"AdamW","epochs":EPOCHS,"checkpoint_epochs":list(CHECKPOINT_EPOCHS),"learning_rates":cfg['adapter_rates'],"expected_parameters":cfg['expected_adapter_parameters'],"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed'],"files":{n:digest(args.public/n) for n in ('STRUCTURED_PEFT_SCORES.csv','STRUCTURED_PEFT_TRACE.csv','STRUCTURED_PEFT_CELL.csv','STRUCTURED_PEFT_BUDGETS.csv','STRUCTURED_PEFT_SUMMARY.csv')}}; (args.public/'STRUCTURED_PEFT_COMPLETE.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline=''); print(summary.to_string(index=False))
if __name__=='__main__': main()
