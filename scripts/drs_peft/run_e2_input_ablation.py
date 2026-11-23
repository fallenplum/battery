"""E2 source-only ablations for response/descriptor and input normalization."""
from __future__ import annotations
import argparse, copy, hashlib, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd, torch
from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.run_head_bridge_aware_peft import apply, install, train
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import full_prediction, role_bundle, observed_mae
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE

METHODS=("RESPONSE_ONLY","DESCRIPTOR_ONLY","RESPONSE_DESCRIPTOR","SOURCE_NORMALIZED","PER_CELL_NORMALIZED","SHUFFLED")

class AblationModel(SharedBatteryModel):
    def __init__(self, backbone_dir, mode):
        super().__init__(backbone_dir); self.mode=mode
    def context(self, features, feature_mask, cycles, anchor, channels=30):
        if channels==12: features,feature_mask=features[...,:12],feature_mask[...,:12]
        cycles=cycles.masked_fill(~feature_mask.any(-1),0); values=torch.where(feature_mask,features,0.0)
        descriptors=__import__('models.drs_peft.bridges',fromlist=['prefix_descriptors']).prefix_descriptors(values,feature_mask,cycles)
        encoded,valid=self.responses(values,feature_mask,cycles); encoded=encoded.reshape(*valid.shape,-1); vm,vl=__import__('models.drs_peft.bridges',fromlist=['temporal_pool']).temporal_pool(encoded,valid)
        obs=valid.any(-1)[...,None].expand_as(vm); mean=__import__('models.drs_peft.bridges',fromlist=['masked_mean']).masked_mean(vm,obs,1); last=__import__('models.drs_peft.bridges',fromlist=['masked_mean']).masked_mean(vl,obs,1)
        if self.mode=="RESPONSE_ONLY": descriptors=torch.zeros_like(descriptors)
        if self.mode=="DESCRIPTOR_ONLY": mean=torch.zeros_like(mean); last=torch.zeros_like(last)
        return torch.cat((mean,last,descriptors,anchor[:,None]),-1)

def transform(bundle,fit_ids,method,seed):
    out={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in bundle.items()}; ids=bundle['ids'].astype(str).tolist(); rows=np.asarray([ids.index(x) for x in fit_ids]); x=bundle['real_features'].astype(np.float32); r=bundle['reference_features'].astype(np.float32)
    if method=="SOURCE_NORMALIZED":
        mu=x[rows].mean((0,1),keepdims=True); sd=x[rows].std((0,1),keepdims=True); sd=np.maximum(sd,1e-5); out['real_features']=(x-mu)/sd; out['reference_features']=(r-mu)/sd
    elif method=="PER_CELL_NORMALIZED":
        mu=x.mean(1,keepdims=True); sd=np.maximum(x.std(1,keepdims=True),1e-5); out['real_features']=(x-mu)/sd; out['reference_features']=(r-mu)/sd
    elif method=="SHUFFLED":
        rng=np.random.default_rng(seed); perm=rng.permutation(len(x)); out['real_features']=x[perm]; out['reference_features']=r[perm]; out['feature_mask']=bundle['feature_mask'][perm]; out['cycles']=bundle['cycles'][perm]
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--private',type=Path,default=PRIVATE); p.add_argument('--features',type=Path,default=PRIVATE); p.add_argument('--data-root',type=Path,default=Path('runtime/drs_peft/data_v1')); p.add_argument('--public',type=Path,default=Path('results/drs_peft/e2_input_ablation')); p.add_argument('--backbone-dir',type=Path,default=BACKBONE); p.add_argument('--device',default='cuda'); args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True); cfg,plan,*_=build_plan(args.config); dev=torch.device(args.device); scores=[]; traces=[]; cells=[]
    for fold in plan['folds']:
        ids=list(dict.fromkeys(fold['fit_ids']+fold['tune_ids']+fold['screen_ids'])); bundle=load_bundle(args.features/'PAIRED_PREFIXES.npz',ids); target,observed=load_future_rows('MATR',args.data_root,ids); labels=Labels(target,observed,ids)
        common_path=args.private/f"COMMON_FOLD{fold['fold']}.pt"; common=torch.load(common_path,map_location='cpu',weights_only=True)
        for method in METHODS:
            transformed=transform(bundle,fold['fit_ids'],method,cfg['seed']+fold['fold']); fit=role_bundle(transformed,fold['fit_ids'],labels,dev); tune=role_bundle(transformed,fold['tune_ids'],labels,dev); screen=role_bundle(transformed,fold['screen_ids'],labels,dev)
            model=AblationModel(args.backbone_dir,method if method in ('RESPONSE_ONLY','DESCRIPTOR_ONLY') else 'RESPONSE_DESCRIPTOR').to(dev); model.load_state_dict({**model.state_dict(),**{k:v for k,v in common.get(k,{}) for k,v in []}}) if False else None
            from scripts.drs_peft.self_reference_training import load_common_state
            load_common_state(model,common); best_global=None; best_state=None; groups=None; count=None
            for lr in cfg['adapter_rates']:
                trial=copy.deepcopy(model); params,groups0,count0,ranks=install(trial,'HEAD_BRIDGE_LORA',cfg['seed']); best,state,tr=train(trial,params,method,lr,fit,tune,cfg,fold['fold'],dev); traces.extend(tr)
                if best_global is None or best['tune_mae']<best_global['tune_mae'] or (best['tune_mae']==best_global['tune_mae'] and lr<best_global['lr']): best_global=best; best_state=state; groups=groups0; count=count0
                del trial
            params,groups2,count2,ranks=install(model,'HEAD_BRIDGE_LORA',cfg['seed']); apply(params,best_state); pred,_=full_prediction(model,screen,12,cfg['train_batch_size'],dev); mae=float(observed_mae(pred,screen).mean()); diff=np.where(screen.observed,pred-screen.targets,0.0); cm=np.abs(diff).sum(1)/screen.observed.sum(1)
            scores.append({'fold':fold['fold'],'method':method,'screen_mae':mae,'best_lr':best_global['lr'],'best_epoch':best_global['epoch'],'trainable_parameters':count2,'head_parameters':groups2['head'],'fusion12_parameters':groups2['fusion12'],'backbone_parameters':groups2['backbone'],'target_future_labels_read':0,'outer_labels_read':0})
            cells.extend({'fold':fold['fold'],'method':method,'cell_uid':uid,'mae':float(v)} for uid,v in zip(screen.ids,cm)); del model
        del common
    sf=pd.DataFrame(scores); sf.to_csv(args.public/'E2_SCORES.csv',index=False,lineterminator='\n'); pd.DataFrame(cells).to_csv(args.public/'E2_CELL.csv',index=False,lineterminator='\n'); sf.groupby('method',as_index=False).agg(screen_mae=('screen_mae','mean'),folds=('fold','nunique'),trainable_parameters=('trainable_parameters','first'),best_epoch=('best_epoch','median'),best_lr=('best_lr','first')).to_csv(args.public/'E2_SUMMARY.csv',index=False,lineterminator='\n'); pd.DataFrame(traces).to_csv(args.public/'E2_TRACE.csv',index=False,lineterminator='\n'); (args.public/'E2_RECEIPT.json').write_text(json.dumps({'status':'COMPLETE_E2_SOURCE_INPUT_ABLATION','methods':list(METHODS),'future_labels_read':0,'fixed_adapter':'HEAD_BRIDGE_LORA'},indent=2)+'\n',encoding='utf-8'); print(pd.read_csv(args.public/'E2_SUMMARY.csv').to_string(index=False))
if __name__=='__main__': main()
