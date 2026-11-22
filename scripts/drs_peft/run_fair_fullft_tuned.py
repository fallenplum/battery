"""Fair Full-FT source comparison using the same tune protocol as Head-aware PEFT."""
from __future__ import annotations
import argparse, copy, hashlib, json, time
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
from scripts.drs_peft.self_reference_training import batches, full_prediction, load_common_state, masked_cell_loss, observed_mae, role_bundle, training_targets
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE

EPOCHS=8
CHECKPOINTS=(2,4,8)
METHOD="FULL_FT_TUNED"

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metric(pred, role):
    mask=role.observed; diff=np.where(mask,pred-role.targets,0.0); n=mask.sum(1)
    mae=np.abs(diff).sum(1)/n; mse=(diff**2).sum(1)/n
    return float(mae.mean()),float(np.sqrt(mse).mean()),float(mse.mean()),mae,mse

def module_counts(model):
    out={"head":sum(p.numel() for p in model.head.parameters()),"fusion12":sum(p.numel() for p in model.fusion12.parameters()),"fusion30":sum(p.numel() for p in model.fusion30.parameters()),"backbone":sum(p.numel() for p in model.bridge.backbone.parameters())}
    out["total"] = sum(out.values())
    return out

def train_full(common, fit, tune, cfg, fold, lr, device):
    model=copy.deepcopy(common).to(device); model.requires_grad_(True)
    values=list(model.parameters()); optimizer=torch.optim.AdamW(values,lr=lr,weight_decay=0.0)
    trace=[]; best=None; best_state=None; peak=0; total_start=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train(); started=time.perf_counter(); total=0.0; steps=0
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        for index in batches(len(fit.ids),cfg['train_batch_size'],seed=cfg['seed']+1009*epoch+fold):
            cycles,y,mask=training_targets(fit,index,device); optimizer.zero_grad(set_to_none=True)
            pred=model(*fit.arrays(index),cycles,channels=12); loss=masked_cell_loss(pred,y,mask,kind='mae'); loss.backward()
            torch.nn.utils.clip_grad_norm_(values,1.0,error_if_nonfinite=True); optimizer.step()
            total += float(loss.detach())*len(index); steps+=1
        model.eval(); tune_pred,inf=full_prediction(model,tune,12,cfg['train_batch_size'],device); tune_mae=float(observed_mae(tune_pred,tune).mean())
        mem=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0; peak=max(peak,mem)
        row={"fold":fold,"method":METHOD,"lr":lr,"epoch":epoch,"training_mae":total/len(fit.ids),"tune_mae":tune_mae,"training_seconds":time.perf_counter()-started,"tune_inference_seconds":inf,"optimizer_updates":steps,"gpu_peak_memory_bytes":mem,"trainable_parameters":sum(p.numel() for p in values)}
        trace.append(row)
        if epoch in CHECKPOINTS and (best is None or tune_mae<best['tune_mae']):
            best=row; best_state={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
    best=best|{"total_training_seconds":time.perf_counter()-total_start,"max_gpu_peak_memory_bytes":peak,"selected_state_sha256":state_hash(best_state)}
    return best,best_state,trace

def state_hash(state):
    h=hashlib.sha256()
    for n in sorted(state): h.update(n.encode()); h.update(state[n].numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--private',type=Path,default=PRIVATE); p.add_argument('--public',type=Path,default=PUBLIC/'fair_fullft_v1'); p.add_argument('--features',type=Path,default=PRIVATE); p.add_argument('--data-root',type=Path,default=CACHE); p.add_argument('--backbone-dir',type=Path,default=BACKBONE); p.add_argument('--device',default='cuda'); args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True)
    cfg,plan,*_=build_plan(args.config); device=torch.device(args.device); scores=[]; traces=[]; cells=[]; budgets=[]
    for fold in plan['folds']:
        ids=list(dict.fromkeys(fold['fit_ids']+fold['tune_ids']+fold['screen_ids'])); bundle=load_bundle(args.features/'PAIRED_PREFIXES.npz',ids); target,observed=load_future_rows('MATR',args.data_root,ids); labels=Labels(target,observed,ids)
        fit=role_bundle(bundle,fold['fit_ids'],labels,device); tune=role_bundle(bundle,fold['tune_ids'],labels,device); screen=role_bundle(bundle,fold['screen_ids'],labels,device)
        common_path=args.private/f"COMMON_FOLD{fold['fold']}.pt"; stored=torch.load(common_path,map_location='cpu',weights_only=True); common=SharedBatteryModel(args.backbone_dir).to(device); load_common_state(common,stored); counts=module_counts(common); common_hash=digest(common_path)
        with torch.no_grad():
            warm_pred,inf=full_prediction(common,screen,12,cfg['train_batch_size'],device)
        wmae,wrmse,wmse,wcm,wcs=metric(warm_pred,screen)
        scores.append({"fold":fold['fold'],"method":"WARMUP_ONLY","best_lr":0.0,"best_epoch":0,"screen_mae":wmae,"screen_rmse":wrmse,"screen_mse":wmse,"screen_cells":len(screen.ids),"trainable_parameters":0,"head_parameters":0,"fusion12_parameters":0,"backbone_parameters":0,"training_seconds":0.0,"screen_inference_seconds":inf,"max_gpu_peak_memory_bytes":0,"common_checkpoint_sha256":common_hash,"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed']})
        for uid,x,y in zip(screen.ids,wcm,wcs): cells.append({"fold":fold['fold'],"method":"WARMUP_ONLY","cell_uid":uid,"mae":float(x),"mse":float(y)})
        global_best=None; global_state=None
        for lr in cfg['adapter_rates']:
            best,state,trace=train_full(common,fit,tune,cfg,fold['fold'],lr,device); traces.extend(trace)
            if global_best is None or best['tune_mae']<global_best['tune_mae'] or (best['tune_mae']==global_best['tune_mae'] and lr<global_best['lr']): global_best=best; global_state=state
            # explicit per-lr selection is recorded in trace. Reset per fold after both rates below.
            if lr==cfg['adapter_rates'][-1]:
                selected=global_best; selected_state=global_state; del global_best
                final=copy.deepcopy(common).to(device); final.requires_grad_(True)
                with torch.no_grad():
                    for name,value in selected_state.items(): dict(final.named_parameters())[name].copy_(value)
                pred,inf=full_prediction(final,screen,12,cfg['train_batch_size'],device); mae,rmse,mse,cm,cs=metric(pred,screen)
                scores.append({"fold":fold['fold'],"method":METHOD,"best_lr":selected['lr'],"best_epoch":selected['epoch'],"screen_mae":mae,"screen_rmse":rmse,"screen_mse":mse,"screen_cells":len(screen.ids),"trainable_parameters":selected['trainable_parameters'],"head_parameters":counts['head'],"fusion12_parameters":counts['fusion12']+counts['fusion30'],"backbone_parameters":counts['backbone'],"training_seconds":selected['total_training_seconds'],"screen_inference_seconds":inf,"max_gpu_peak_memory_bytes":selected['max_gpu_peak_memory_bytes'],"max_gpu_peak_memory_mib":selected['max_gpu_peak_memory_bytes']/(1024**2),"common_checkpoint_sha256":common_hash,"selected_state_sha256":selected['selected_state_sha256'],"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed']})
                for uid,x,y in zip(screen.ids,cm,cs): cells.append({"fold":fold['fold'],"method":METHOD,"cell_uid":uid,"mae":float(x),"mse":float(y)})
                del final
        budgets.append({"method":"WARMUP_ONLY","trainable_parameters":0}); budgets.append({"method":METHOD,"trainable_parameters":counts['total'],"head_parameters":counts['head'],"fusion12_parameters":counts['fusion12']+counts['fusion30'],"backbone_parameters":counts['backbone']})
        del common
    sf=pd.DataFrame(scores); tf=pd.DataFrame(traces); cf=pd.DataFrame(cells); bf=pd.DataFrame(budgets).drop_duplicates('method'); sf.to_csv(args.public/'FAIR_FT_SCORES.csv',index=False,lineterminator='\n'); tf.to_csv(args.public/'FAIR_FT_TRACE.csv',index=False,lineterminator='\n'); cf.to_csv(args.public/'FAIR_FT_CELL.csv',index=False,lineterminator='\n'); bf.to_csv(args.public/'FAIR_FT_BUDGETS.csv',index=False,lineterminator='\n'); summary=sf.groupby('method',as_index=False).agg(screen_mae=('screen_mae','mean'),screen_rmse=('screen_rmse','mean'),screen_mse=('screen_mse','mean'),folds=('fold','nunique'),screen_cells=('screen_cells','sum'),trainable_parameters=('trainable_parameters','first'),best_epoch=('best_epoch','median'),best_lr=('best_lr','first')).sort_values('screen_mae'); summary.to_csv(args.public/'FAIR_FT_SUMMARY.csv',index=False,lineterminator='\n'); receipt={"status":"COMPLETE_SOURCE_FAIR_TUNED_FULLFT","domain":"MATR","folds":[f['fold'] for f in plan['folds']],"methods":["WARMUP_ONLY",METHOD],"input_channels":12,"epochs":EPOCHS,"checkpoint_epochs":list(CHECKPOINTS),"learning_rates":cfg['adapter_rates'],"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed'],"files":{n:digest(args.public/n) for n in ('FAIR_FT_SCORES.csv','FAIR_FT_TRACE.csv','FAIR_FT_CELL.csv','FAIR_FT_BUDGETS.csv','FAIR_FT_SUMMARY.csv')}}; (args.public/'FAIR_FT_COMPLETE.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline=''); print(summary.to_string(index=False))
if __name__=='__main__': main()
