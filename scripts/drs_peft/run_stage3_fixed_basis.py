"""Stage 3 fixed SVD-basis versus random-basis PEFT on source data."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from models.drs_peft.onesided import replace_layer
from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import full_prediction, load_common_state, role_bundle
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE
from scripts.drs_peft.run_structured_peft_stage2 import EPOCHS, CHECKPOINT_EPOCHS, metrics, train_trial

RANK=30
METHODS=("FIXED_SVD_BASIS","RANDOM_BASIS")

class FixedTwoSidedLinear(nn.Module):
    def __init__(self, base: nn.Linear, left: torch.Tensor, right: torch.Tensor):
        super().__init__()
        if not isinstance(base, nn.Linear): raise TypeError("Expected nn.Linear")
        if left.shape[0] != base.out_features or right.shape[0] != base.in_features or left.shape[1] != right.shape[1]:
            raise ValueError("Basis dimensions do not match layer")
        self.base=base.requires_grad_(False)
        self.register_buffer("U", left.detach().to(base.weight))
        self.register_buffer("V", right.detach().to(base.weight))
        self.C=nn.Parameter(base.weight.new_zeros(left.shape[1],left.shape[1]))
    def forward(self,x):
        z=F.linear(x,self.V.T)
        z=F.linear(z,self.C)
        return self.base(x)+F.linear(z,self.U)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def teacher_bases(private,fold,target_names,device,kind):
    root=Path(private)/f"fold{fold}"
    warm=torch.load(root/"FULL_FT_EPOCH0.pt",map_location="cpu",weights_only=True)["state"]
    final=torch.load(root/"FULL_FT_EPOCH8.pt",map_location="cpu",weights_only=True)["state"]
    bases={}
    for name,layer in target_names.items():
        full="bridge.backbone.module."+name+".weight"
        delta=(final[full]-warm[full]).double()
        out_dim,in_dim=delta.shape
        rank=min(RANK,out_dim,in_dim)
        if kind=="svd":
            U,S,Vh=torch.linalg.svd(delta,full_matrices=False)
            left=U[:,:rank].float(); right=Vh[:rank,:].T.float()
        else:
            seed=42+fold*1009+(sum(ord(c) for c in name)%997)
            gen=torch.Generator(device="cpu").manual_seed(seed)
            left=torch.linalg.qr(torch.randn(out_dim,rank,generator=gen,dtype=torch.float64),mode="reduced").Q.float()
            gen=torch.Generator(device="cpu").manual_seed(seed+1000003)
            right=torch.linalg.qr(torch.randn(in_dim,rank,generator=gen,dtype=torch.float64),mode="reduced").Q.float()
        bases[name]=(left,right)
    return bases,warm,final

def install_basis(model,bases):
    targets=model.bridge.backbone.linear_targets("qkvo_ffn")
    model.requires_grad_(False)
    for name,layer in targets.items():
        left,right=bases[name]
        replace_layer(model.backbone,name,FixedTwoSidedLinear(layer,left,right))
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    return params,sum(p.numel() for p in params.values())

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=CONFIG); p.add_argument("--private",type=Path,default=PRIVATE); p.add_argument("--public",type=Path,default=PUBLIC/"structured_peft_stage3_v1"); p.add_argument("--features",type=Path,default=PRIVATE); p.add_argument("--data-root",type=Path,default=CACHE); p.add_argument("--backbone-dir",type=Path,default=BACKBONE); p.add_argument("--device",default="cuda"); args=p.parse_args(); configure(); args.public.mkdir(parents=True,exist_ok=True)
    cfg,plan,*_=build_plan(args.config); device=torch.device(args.device); scores=[]; traces=[]; cells=[]; budgets=[]
    for fold in plan["folds"]:
        ids=list(dict.fromkeys(fold["fit_ids"]+fold["tune_ids"]+fold["screen_ids"])); bundle=load_bundle(args.features/"PAIRED_PREFIXES.npz",ids); target,observed=load_future_rows("MATR",args.data_root,ids); labels=Labels(target,observed,ids)
        fit=role_bundle(bundle,fold["fit_ids"],labels,device); tune=role_bundle(bundle,fold["tune_ids"],labels,device); screen=role_bundle(bundle,fold["screen_ids"],labels,device)
        common_path=args.private/f"COMMON_FOLD{fold['fold']}.pt"; common=torch.load(common_path,map_location="cpu",weights_only=True); common_model=SharedBatteryModel(args.backbone_dir).to(device); load_common_state(common_model,common); common_hash=digest(common_path); target_names=common_model.bridge.backbone.linear_targets("qkvo_ffn")
        for method in METHODS:
            kind="svd" if method=="FIXED_SVD_BASIS" else "random"; bases,_,_=teacher_bases(args.private,fold['fold'],target_names,device,kind); best_global=None; best_state=None
            for lr in cfg["adapter_rates"]:
                model=copy.deepcopy(common_model); params,count=install_basis(model,bases)
                if count!=42*RANK*RANK: raise RuntimeError(f"unexpected basis budget {count}")
                best,state,trace=train_trial(model,params,method,lr,fit,tune,cfg,fold['fold'],device); traces.extend(trace)
                if best_global is None or best['tune_mae']<best_global['tune_mae'] or (best['tune_mae']==best_global['tune_mae'] and lr<best_global['lr']): best_global=best; best_state=state
                del model
            final=copy.deepcopy(common_model); final_params,count=install_basis(final,bases)
            with torch.no_grad():
                for name,value in best_state.items(): final_params[name].copy_(value)
            pred,inf=full_prediction(final,screen,12,cfg['train_batch_size'],device); mae,rmse,mse,cell_mae,cell_mse=metrics(pred,screen)
            scores.append({"fold":fold['fold'],"method":method,"rank":RANK,"trainable_parameters":count,"best_lr":best_global['lr'],"best_epoch":best_global['epoch'],"screen_mae":mae,"screen_rmse":rmse,"screen_mse":mse,"screen_cells":len(screen.ids),"training_seconds":best_global['total_training_seconds'],"screen_inference_seconds":inf,"max_gpu_peak_memory_bytes":best_global['max_gpu_peak_memory_bytes'],"max_gpu_peak_memory_mib":best_global['max_gpu_peak_memory_bytes']/(1024**2),"common_checkpoint_sha256":common_hash,"basis_source":"source_fit_fullft_epoch8" if kind=="svd" else "source_fit_shape_random_control","target_future_labels_read":0,"outer_labels_read":0,"seed":cfg['seed']})
            for uid,x,y in zip(screen.ids,cell_mae,cell_mse): cells.append({"fold":fold['fold'],"method":method,"cell_uid":uid,"mae":float(x),"mse":float(y)})
            budgets.append({"method":method,"rank":RANK,"trainable_parameters":count,"basis_parameters":42*RANK*(384+384)})
            del final
        del common_model
    sf=pd.DataFrame(scores); tf=pd.DataFrame(traces); cf=pd.DataFrame(cells); bf=pd.DataFrame(budgets).drop_duplicates("method"); sf.to_csv(args.public/"STAGE3_SCORES.csv",index=False,lineterminator="\n"); tf.to_csv(args.public/"STAGE3_TRACE.csv",index=False,lineterminator="\n"); cf.to_csv(args.public/"STAGE3_CELL.csv",index=False,lineterminator="\n"); bf.to_csv(args.public/"STAGE3_BUDGETS.csv",index=False,lineterminator="\n"); summary=sf.groupby("method",as_index=False).agg(screen_mae=("screen_mae","mean"),screen_rmse=("screen_rmse","mean"),screen_mse=("screen_mse","mean"),folds=("fold","nunique"),screen_cells=("screen_cells","sum"),trainable_parameters=("trainable_parameters","first"),best_epoch=("best_epoch","median"),best_lr=("best_lr","first")).sort_values("screen_mae"); summary.to_csv(args.public/"STAGE3_SUMMARY.csv",index=False,lineterminator="\n")
    receipt={"status":"COMPLETE_SOURCE_STAGE3_FIXED_BASIS","domain":"MATR","folds":[f["fold"] for f in plan["folds"]],"methods":list(METHODS),"rank":RANK,"targets":"qkvo_ffn all 42 matrices","basis_source":"source fit FullFT epoch8 private snapshots for SVD, deterministic random orthogonal control","input_channels":12,"epochs":EPOCHS,"learning_rates":cfg["adapter_rates"],"target_future_labels_read":0,"outer_labels_read":0,"seed":cfg["seed"],"files":{n:digest(args.public/n) for n in ("STAGE3_SCORES.csv","STAGE3_TRACE.csv","STAGE3_CELL.csv","STAGE3_BUDGETS.csv","STAGE3_SUMMARY.csv")}}; (args.public/"STAGE3_COMPLETE.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline=""); print(summary.to_string(index=False))
if __name__=="__main__": main()
