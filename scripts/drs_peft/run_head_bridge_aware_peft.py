"""Head/Bridge-aware PEFT from the locked warmup source state."""
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
from scripts.drs_peft.self_reference_training import batches, full_prediction, load_common_state, masked_cell_loss, observed_mae, role_bundle, training_targets
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE

EPOCHS=8; CHECKPOINTS=(2,4,8)
METHODS=("HEAD_ONLY_TRAIN","BRIDGE12_ONLY_TRAIN","HEAD_BRIDGE_TRAIN","HEAD_LORA","HEAD_BRIDGE_LORA","BACKBONE_ONLY")

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def late_rank_map(targets, mode):
    ranks={n:0 for n in targets}
    if mode=="LORA_12":
        for n in targets:
            if n.endswith("self_attn.q_proj") or n.endswith("self_attn.v_proj"):
                ranks[n]=4
    elif mode=="BACKBONE_ONLY":
        # Registered Stage-2 late-layer budget, with exact 36k matching.
        for n in targets:
            if any(f"encoder.layers.{i}." in n for i in (3,4,5)): ranks[n]=1
        ffn=[n for n in sorted(targets) if ranks[n] and ".ffn." in n]
        attn=[n for n in sorted(targets) if ranks[n] and ".self_attn." in n]
        for n in ffn[:3]: ranks[n]=2
        for n in attn[:7]: ranks[n]=3
    elif mode=="HEAD_LORA":
        # Head plus 30,336 backbone parameters, staying within 36k.
        for n in targets:
            if any(f"encoder.layers.{i}." in n for i in (3,4,5)): ranks[n]=1
        attn=[n for n in sorted(targets) if ranks[n] and ".self_attn." in n]
        for n in attn[:11]: ranks[n]=2
    elif mode=="HEAD_BRIDGE_LORA":
        # Head + fusion12 plus one late FFN layer, below 64k.
        for n in targets:
            if "encoder.layers.5.ffn." in n: ranks[n]=1
    return ranks

def install(model,method,seed):
    model.requires_grad_(False)
    if method in ("HEAD_ONLY_TRAIN","HEAD_BRIDGE_TRAIN","HEAD_LORA","HEAD_BRIDGE_LORA"):
        model.head.requires_grad_(True)
    if method in ("BRIDGE12_ONLY_TRAIN","HEAD_BRIDGE_TRAIN","HEAD_BRIDGE_LORA"):
        model.fusion12.requires_grad_(True)
    if method=="BRIDGE12_ONLY_TRAIN": model.head.requires_grad_(False)
    targets=model.bridge.backbone.linear_targets("qkvo_ffn")
    ranks=late_rank_map(targets,method)
    if any(ranks.values()):
        if method in ("HEAD_LORA","HEAD_BRIDGE_LORA"):
            model.head.requires_grad_(True)
        if method=="HEAD_BRIDGE_LORA": model.fusion12.requires_grad_(True)
        for name,layer in targets.items():
            rank=ranks[name]
            if rank<=0: continue
            adapter=LoRALinear(layer,rank,alpha=rank,dropout=0.0,seed=seed)
            for pn,p in adapter.named_parameters(): p.requires_grad_(not pn.startswith("base."))
            replace_layer(model.backbone,name,adapter)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    groups={"head":sum(p.numel() for p in model.head.parameters() if p.requires_grad),"fusion12":sum(p.numel() for p in model.fusion12.parameters() if p.requires_grad),"backbone":sum(p.numel() for n,p in params.items() if n.startswith("bridge.backbone") and p.requires_grad)}
    return params,groups,sum(params.values(),torch.tensor(0)).item() if False else sum(p.numel() for p in params.values()),ranks

def metric(pred,role):
    mask=role.observed; diff=np.where(mask,pred-role.targets,0.0); n=mask.sum(1); mae=np.abs(diff).sum(1)/n; mse=(diff**2).sum(1)/n
    return float(mae.mean()),float(np.sqrt(mse).mean()),float(mse.mean()),mae,mse

def train(model,params,method,lr,fit,tune,cfg,fold,device):
    values=list(params.values()); opt=torch.optim.AdamW(values,lr=lr,weight_decay=0.0); traces=[]; best=None; best_state=None; total_start=__import__('time').perf_counter(); peak=0
    for epoch in range(1,EPOCHS+1):
        model.train(); start=__import__('time').perf_counter(); total=0.; steps=0
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        for index in batches(len(fit.ids),cfg['train_batch_size'],seed=cfg['seed']+1009*epoch+fold):
            cyc,y,mask=training_targets(fit,index,device); opt.zero_grad(set_to_none=True); pred=model(*fit.arrays(index),cyc,channels=12); loss=masked_cell_loss(pred,y,mask,kind='mae'); loss.backward(); torch.nn.utils.clip_grad_norm_(values,1.0,error_if_nonfinite=True); opt.step(); total+=float(loss.detach())*len(index); steps+=1
        model.eval(); tune_pred,inf=full_prediction(model,tune,12,cfg['train_batch_size'],device); tune_mae=float(observed_mae(tune_pred,tune).mean()); mem=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0; peak=max(peak,mem)
        row={"fold":fold,"method":method,"lr":lr,"epoch":epoch,"training_mae":total/len(fit.ids),"tune_mae":tune_mae,"training_seconds":__import__('time').perf_counter()-start,"tune_inference_seconds":inf,"gpu_peak_memory_bytes":mem,"trainable_parameters":sum(p.numel() for p in values)}; traces.append(row)
        if epoch in CHECKPOINTS and (best is None or tune_mae<best['tune_mae']): best=row; best_state={n:p.detach().cpu().clone() for n,p in params.items()}
    best=best|{"total_training_seconds":__import__('time').perf_counter()-total_start,"max_gpu_peak_memory_bytes":peak}
    return best,best_state,traces

def apply(params,state):
    with torch.no_grad():
        for n,v in state.items(): params[n].copy_(v)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--private',type=Path,default=PRIVATE); p.add_argument('--public',type=Path,default=PUBLIC/'head_bridge_peft_v1'); p.add_argument('--features',type=Path,default=PRIVATE); p.add_argument('--data-root',type=Path,default=CACHE); p.add_argument('--backbone-dir',type=Path,default=BACKBONE); p.add_argument('--device',default='cuda'); args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True); cfg,plan,*_=build_plan(args.config); device=torch.device(args.device); scores=[]; traces=[]; cells=[]; budgets=[]
    for fold in plan['folds']:
        ids=list(dict.fromkeys(fold['fit_ids']+fold['tune_ids']+fold['screen_ids'])); bundle=load_bundle(args.features/'PAIRED_PREFIXES.npz',ids); target,observed=load_future_rows('MATR',args.data_root,ids); labels=Labels(target,observed,ids); fit=role_bundle(bundle,fold['fit_ids'],labels,device); tune=role_bundle(bundle,fold['tune_ids'],labels,device); screen=role_bundle(bundle,fold['screen_ids'],labels,device)
        common_path=args.private/f"COMMON_FOLD{fold['fold']}.pt"; common=torch.load(common_path,map_location='cpu',weights_only=True); common_model=SharedBatteryModel(args.backbone_dir).to(device); load_common_state(common_model,common); common_hash=digest(common_path)
        for method in METHODS:
            best_global=None; best_state=None; last_groups=None; last_ranks=None
            for lr in cfg['adapter_rates']:
                model=copy.deepcopy(common_model); params,groups,count,ranks=install(model,method,cfg['seed']); last_groups=groups; last_ranks=ranks; best,state,tr=train(model,params,method,lr,fit,tune,cfg,fold['fold'],device); traces.extend(tr)
                if best_global is None or best['tune_mae']<best_global['tune_mae'] or (best['tune_mae']==best_global['tune_mae'] and lr<best_global['lr']): best_global=best; best_state=state
                del model
            final=copy.deepcopy(common_model); fp,groups,count,ranks=install(final,method,cfg['seed']); apply(fp,best_state); pred,inf=full_prediction(final,screen,12,cfg['train_batch_size'],device); mae,rmse,mse,cm,cs=metric(pred,screen)
            scores.append({"fold":fold['fold'],"method":method,"best_lr":best_global['lr'],"best_epoch":best_global['epoch'],"screen_mae":mae,"screen_rmse":rmse,"screen_mse":mse,"screen_cells":len(screen.ids),"trainable_parameters":count,"head_parameters":groups['head'],"fusion12_parameters":groups['fusion12'],"backbone_parameters":groups['backbone'],"training_seconds":best_global['total_training_seconds'],"screen_inference_seconds":inf,"max_gpu_peak_memory_bytes":best_global['max_gpu_peak_memory_bytes'],"max_gpu_peak_memory_mib":best_global['max_gpu_peak_memory_bytes']/(1024**2),"common_checkpoint_sha256":common_hash,"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed']})
            for uid,x,y in zip(screen.ids,cm,cs): cells.append({"fold":fold['fold'],"method":method,"cell_uid":uid,"mae":float(x),"mse":float(y)})
            budgets.append({"method":method,"trainable_parameters":count,"head_parameters":groups['head'],"fusion12_parameters":groups['fusion12'],"backbone_parameters":groups['backbone'],"rank_schedule":json.dumps(ranks,sort_keys=True)})
            del final
        del common_model
    sf=pd.DataFrame(scores); tf=pd.DataFrame(traces); cf=pd.DataFrame(cells); bf=pd.DataFrame(budgets).drop_duplicates('method'); sf.to_csv(args.public/'PEFT_SCORES.csv',index=False,lineterminator='\n'); tf.to_csv(args.public/'PEFT_TRACE.csv',index=False,lineterminator='\n'); cf.to_csv(args.public/'PEFT_CELL.csv',index=False,lineterminator='\n'); bf.to_csv(args.public/'PEFT_BUDGETS.csv',index=False,lineterminator='\n'); summary=sf.groupby('method',as_index=False).agg(screen_mae=('screen_mae','mean'),screen_rmse=('screen_rmse','mean'),screen_mse=('screen_mse','mean'),folds=('fold','nunique'),screen_cells=('screen_cells','sum'),trainable_parameters=('trainable_parameters','first'),head_parameters=('head_parameters','first'),fusion12_parameters=('fusion12_parameters','first'),backbone_parameters=('backbone_parameters','first'),best_epoch=('best_epoch','median'),best_lr=('best_lr','first')).sort_values('screen_mae'); summary.to_csv(args.public/'PEFT_SUMMARY.csv',index=False,lineterminator='\n'); receipt={"status":"COMPLETE_SOURCE_HEAD_BRIDGE_AWARE_PEFT","domain":"MATR","folds":[f['fold'] for f in plan['folds']],"methods":list(METHODS),"input_channels":12,"epochs":EPOCHS,"learning_rates":cfg['adapter_rates'],"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed'],"files":{n:digest(args.public/n) for n in ('PEFT_SCORES.csv','PEFT_TRACE.csv','PEFT_CELL.csv','PEFT_BUDGETS.csv','PEFT_SUMMARY.csv')}}; (args.public/'PEFT_COMPLETE.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline=''); print(summary.to_string(index=False))
if __name__=='__main__': main()
