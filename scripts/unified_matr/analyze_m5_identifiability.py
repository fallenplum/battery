"""Frozen-MATR early Q(V) identifiability: rate level, rate change and knee-proxy stability.

No certified knee labels exist. This is retrospective study on the same historically
exposed 169 cells, NOT a new blind test or proof of intrinsic non-identifiability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.unified_matr.train_seed42 import load_frozen_data

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
SPLITS = tuple(f"cell_fold_{i}" for i in range(5))
LOCAL_COLUMNS = (0, 1)  # signed residual mean, residual RMS in 8 fixed bins


def early_features(inputs):
    n = len(inputs["anchor_soh100"])
    policy = np.asarray(inputs["policy"], dtype=float)
    policy_mask = np.asarray(inputs["policy_mask"], dtype=bool)
    prior = np.column_stack((np.where(policy_mask, policy, np.nan),
                             policy_mask.astype(float),
                             np.asarray(inputs["anchor_soh100"], dtype=float)))
    physics = []
    for i in range(n):
        available = np.asarray(inputs["point_mask"][i], dtype=bool).any(-1)
        ids = np.asarray(inputs["cycle_ids"][i], dtype=float)
        if not available.any() or not ((ids[available] >= 1) & (ids[available] <= 100)).all():
            raise RuntimeError("Uncertified or empty frozen physical cycle prefix")
        global_mask = np.asarray(inputs["global_mask"][i], dtype=bool)
        local_mask = np.asarray(inputs["local_feature_mask"][i], dtype=bool)
        global_values = np.where(global_mask, inputs["global_features"][i], np.nan)
        local_values = np.where(local_mask, inputs["local_features"][i], np.nan)
        last = int(np.flatnonzero(available)[-1])
        g = global_values[last]
        l = local_values[last][:, list(LOCAL_COLUMNS)].reshape(-1)
        gm = global_mask[last].astype(float)
        lm = local_mask[last][:, list(LOCAL_COLUMNS)].reshape(-1).astype(float)
        physics.append(np.concatenate((g, l, gm, lm)))
    return prior, np.asarray(physics)


def ridge_predict(xtrain, ytrain, xval, yval, xtest):
    best = None
    for alpha in ALPHAS:
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(), Ridge(alpha=alpha))
        model.fit(xtrain, ytrain)
        error = float(np.mean(np.abs(model.predict(xval) - yval)))
        if best is None or error < best[1]:
            best = (alpha, error)
    final = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), Ridge(alpha=best[0]))
    final.fit(np.vstack((xtrain, xval)), np.r_[ytrain, yval])
    return final.predict(xtest), best


def forecast_targets(rate_targets, rate_mask):
    mid, late = rate_targets[:,0], rate_targets[:,1]
    both = rate_mask[:,0] & rate_mask[:,1]
    return {
        "level_mid_all": (mid, rate_mask[:,0]),
        "level_mid_shared": (mid, both),
        "level_late_shared": (late, both),
        "change_late_minus_mid_shared": (late-mid, both),
    }


def rate_task(inputs, identifiers, rate_targets, rate_mask, splits):
    base, phys = early_features(inputs)
    designs = {"policy_capacity":base, "policy_capacity_plus_QV":np.hstack((base,phys))}
    ids = {u:i for i,u in enumerate(identifiers)}
    rows = []
    for name,(target,supported) in forecast_targets(rate_targets,rate_mask).items():
        for fold in SPLITS:
            group = splits["splits"][fold]
            index = {role:np.array([ids[uid] for uid in group[role] if supported[ids[uid]]],dtype=int)
                     for role in ("train","validation","test")}
            tr,va,te = (index[k] for k in ("train","validation","test"))
            if len(tr)<8 or len(va)<2 or len(te)<1:
                rows.append(dict(task=name,split=fold,model="SKIPPED_LOW_SUPPORT",
                                 train=len(tr),validation=len(va),test=len(te)))
                continue
            for model_name,X in designs.items():
                pred,(alpha,valerror)=ridge_predict(X[tr],target[tr]*1000,
                                                    X[va],target[va]*1000,X[te])
                for pos,value in zip(te,pred):
                    rows.append(dict(task=name,split=fold,model=model_name,
                                     cell_uid=identifiers[pos],truth=float(target[pos]),
                                     prediction=float(value/1000),
                                     abs_error=float(abs(value/1000-target[pos])),
                                     alpha=alpha,val_mae_1000=valerror,
                                     train=len(tr),validation=len(va),test=len(te)))
    df = pd.DataFrame(rows)
    valid = df[df.model.isin(designs)]
    summary = valid.groupby(["task","model"],as_index=False).agg(
        unique_test_cells=("cell_uid","nunique"),mae=("abs_error","mean"),
        truth_sd=("truth","std"),truth_mean=("truth","mean"))
    return df,summary


def hinge_proxy(x,y):
    if len(x)<240 or x[-1]-x[0]<300:
        return None
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if len(x)>500:
        choose=np.unique(np.round(np.linspace(0,len(x)-1,500)).astype(int))
        x,y=x[choose],y[choose]
    xm,xs=x.mean(),max(x.std(),1.)
    z=(x-xm)/xs
    quad=np.column_stack((np.ones(len(x)),z,z*z))
    c=np.linalg.lstsq(quad,y,rcond=None)[0]
    quad_sse=float(np.square(y-quad@c).sum())
    left,right=int(len(x)*.2),int(len(x)*.8)
    candidates=np.unique(x[np.linspace(left,right,75).astype(int)])
    fits=[]
    for k in candidates:
        design=np.column_stack((np.ones(len(x)),z,np.maximum((x-k)/xs,0)))
        beta=np.linalg.lstsq(design,y,rcond=None)[0]
        sse=float(np.square(y-design@beta).sum())
        fits.append((float(k),sse,float(beta[1]/xs),float((beta[1]+beta[2])/xs)))
    opt=min(fits,key=lambda r:r[1])
    compatible=[r[0] for r in fits if r[1]<=opt[1]*1.05+1e-12]
    return dict(knee_proxy_center_cycle=opt[0],profile_width_5pct_SSE=max(compatible)-min(compatible),
                pre_SOHSlope=opt[2],post_SOHSlope=opt[3],
                bic_quad_minus_hinge=len(x)*np.log(max(quad_sse,1e-16)/max(opt[1],1e-16)),
                certified_physical_knee=False)


def knee_diagnostic(targets,observed,identifiers):
    rows=[]
    for i,uid in enumerate(identifiers):
        x=np.flatnonzero(observed[i])+101
        y=targets[i,observed[i]]
        for start in (101,201,301):
            subset=x>=start
            result=hinge_proxy(x[subset],y[subset])
            if result:
                rows.append(dict(cell_uid=uid,fit_start_cycle=start,**result))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("Write one immutable analysis directory per execution")
    inputs,ids,eligible,_,_,targets,observed,rates,mask,splits,_=load_frozen_data(a.data)
    if len(ids)!=169 or not eligible.all():
        raise RuntimeError("The fixed MATR169 data gate failed")
    predicted,summary=rate_task(inputs,ids,rates,mask,splits)
    knee=knee_diagnostic(targets,observed,ids)
    a.output.mkdir(parents=True)
    predicted.to_csv(a.output/"rate_predictability_oof.csv",index=False)
    summary.to_csv(a.output/"rate_predictability_summary.csv",index=False)
    knee.to_csv(a.output/"knee_proxy_stability.csv",index=False)
    contract=dict(status="RETROSPECTIVE_EXPLORATORY_NO_BLIND_TEST",
       n_original_MATR=len(ids),mid_support=int(mask[:,0].sum()),late_support=int(mask[:,1].sum()),
       paired_mid_late_support=int((mask[:,0]&mask[:,1]).sum()),
       feature_cutoff_original_cycle=100,validation_tunes_alpha_only=True,
       fixed_outer_splits=list(SPLITS),
       target_units="SOH_fraction_per_real_cycle",
       predictor="prior known policy and SOH100 anchor; plus last valid global20 and 8x2 local QV features/masks",
       knee_true_events_certified=0,
       knee_diagnostic="Hinge candidate sensitivity to suffix fit start and quadratic countermodel, not an event label",
       causal_warning="Predictive associations do not uniquely identify LLI/LAM or structural knee onset",
       scientific_limit="Failure to estimate a hinge candidate does not prove a true knee is theoretically unidentifiable")
    (a.output/"IDENTIFIABILITY_CONTRACT.json").write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(contract,ensure_ascii=False))
    print(summary.to_string(index=False))


if __name__=="__main__":
    main()
