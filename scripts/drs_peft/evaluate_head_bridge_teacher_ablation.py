"""Head/Bridge-aware Full-FT teacher functional ablation."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd, torch
from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import full_prediction, load_common_state, role_bundle
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE
from scripts.drs_peft.evaluate_fullft_functional_ablation import digest, group, is_encoder, load_parameters, metric_rows

EPOCH=8
METHODS=("WARMUP_ONLY_12","FULL_FT_12","HEAD_ONLY","BRIDGE_ONLY","HEAD_BRIDGE","HEAD_LATE_BACKBONE","HEAD_ALL_BACKBONE")

def keep_delta(name, method):
    g=group(name)
    head=g=="head"
    bridge=g=="bridge_fusion12"
    backbone=name.startswith("bridge.backbone.module.")
    late=is_encoder(name,(3,4,5))
    if method=="WARMUP_ONLY_12": return False
    if method=="FULL_FT_12": return True
    if method=="HEAD_ONLY": return head
    if method=="BRIDGE_ONLY": return bridge
    if method=="HEAD_BRIDGE": return head or bridge
    if method=="HEAD_LATE_BACKBONE": return head or (backbone and late)
    if method=="HEAD_ALL_BACKBONE": return head or backbone
    raise ValueError(method)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--private',type=Path,default=PRIVATE); p.add_argument('--public',type=Path,default=PUBLIC/'head_bridge_teacher_v1'); p.add_argument('--features',type=Path,default=PRIVATE); p.add_argument('--data-root',type=Path,default=CACHE); p.add_argument('--backbone-dir',type=Path,default=BACKBONE); p.add_argument('--device',default='cuda'); args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True)
    cfg,plan,*_=build_plan(args.config); device=torch.device(args.device); scores=[]; cells=[]; hashes=[]
    for fold in plan['folds']:
        ids=list(dict.fromkeys(fold['fit_ids']+fold['tune_ids']+fold['screen_ids'])); bundle=load_bundle(args.features/'PAIRED_PREFIXES.npz',ids); target,observed=load_future_rows('MATR',args.data_root,ids); labels=Labels(target,observed,ids); screen=role_bundle(bundle,fold['screen_ids'],labels,device)
        common_path=args.private/f"COMMON_FOLD{fold['fold']}.pt"; common=torch.load(common_path,map_location='cpu',weights_only=True); root=args.private/f"fold{fold['fold']}"; warm=torch.load(root/'FULL_FT_EPOCH0.pt',map_location='cpu',weights_only=True)['state']; final=torch.load(root/f'FULL_FT_EPOCH{EPOCH}.pt',map_location='cpu',weights_only=True)['state']; delta={n:final[n]-warm[n] for n in warm}; hashes.extend([{"fold":fold['fold'],"epoch":e,"sha256":digest(root/f"FULL_FT_EPOCH{e}.pt")} for e in (0,EPOCH)])
        model=SharedBatteryModel(args.backbone_dir).to(device); load_common_state(model,common); load_parameters(model,warm)
        for method in METHODS:
            started=time.perf_counter(); state={n:(warm[n]+delta[n] if keep_delta(n,method) else warm[n]) for n in warm}; load_parameters(model,state)
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            pred,seconds=full_prediction(model,screen,12,cfg['train_batch_size'],device); peak=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0; mae,mse,rmse,pooled_mse,pooled_rmse=metric_rows(pred,screen)
            scores.append({"fold":fold['fold'],"method":method,"screen_mae":float(mae.mean()),"screen_rmse":float(rmse.mean()),"screen_mse":float(mse.mean()),"screen_pooled_rmse":pooled_rmse,"screen_pooled_mse":pooled_mse,"screen_cells":len(screen.ids),"trainable_parameters":0,"teacher_update_parameters":13999369,"training_seconds":0.0,"inference_seconds":float(seconds),"gpu_peak_memory_bytes":peak,"gpu_peak_memory_mib":peak/(1024**2),"state_apply_seconds":float(time.perf_counter()-started-seconds),"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed']})
            for uid,x,y,z in zip(screen.ids,mae,mse,rmse): cells.append({"fold":fold['fold'],"method":method,"cell_uid":uid,"mae":float(x),"mse":float(y),"rmse":float(z)})
        del model
    sf=pd.DataFrame(scores); cf=pd.DataFrame(cells); sf.to_csv(args.public/'HEAD_BRIDGE_TEACHER_SCORES.csv',index=False,lineterminator='\n'); cf.to_csv(args.public/'HEAD_BRIDGE_TEACHER_CELL.csv',index=False,lineterminator='\n'); summary=sf.groupby('method',as_index=False).agg(screen_mae=('screen_mae','mean'),screen_rmse=('screen_rmse','mean'),screen_mse=('screen_mse','mean'),screen_pooled_rmse=('screen_pooled_rmse','mean'),screen_pooled_mse=('screen_pooled_mse','mean'),folds=('fold','nunique'),screen_cells=('screen_cells','sum')).sort_values('screen_mae'); summary.to_csv(args.public/'HEAD_BRIDGE_TEACHER_SUMMARY.csv',index=False,lineterminator='\n'); receipt={"status":"COMPLETE_SOURCE_HEAD_BRIDGE_TEACHER_ABLATION","domain":"MATR","folds":[f['fold'] for f in plan['folds']],"methods":list(METHODS),"input_channels":12,"epoch":EPOCH,"definition":"HEAD_ALL_BACKBONE includes Head plus bridge.backbone.module parameters and excludes fusion12","target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed'],"private_checkpoint_hashes":hashes,"files":{n:digest(args.public/n) for n in ('HEAD_BRIDGE_TEACHER_SCORES.csv','HEAD_BRIDGE_TEACHER_CELL.csv','HEAD_BRIDGE_TEACHER_SUMMARY.csv')}}; (args.public/'HEAD_BRIDGE_TEACHER_COMPLETE.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline=''); print(summary.to_string(index=False))
if __name__=='__main__': main()
