from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.drs_peft.run_head_aware_new_family_blind import score
from scripts.drs_peft.transfer_scope_data import load_future_rows

METHODS = ("WARMUP_ONLY", "LORA_12", "HEAD_ONLY_TRAIN", "HEAD_LORA", "HEAD_BRIDGE_TRAIN", "HEAD_BRIDGE_LORA", "FULL_FT_TUNED")
WINDOWS = ((101, 200), (201, 500), (501, 1000), (1001, 5000))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--arrays", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--source-prefix", type=Path, required=True)
    a = p.parse_args()
    a.public.mkdir(parents=True, exist_ok=True)
    lock = json.loads(Path("results/drs_peft/head_aware_seed_v1/seed123/target/HUST/C_HUST/PREDICTIONS_LOCKED.json").read_text())
    ids = lock["target_ids"]
    meta = pd.read_csv(a.metadata).set_index("cell_uid").loc[ids]
    complete = json.loads((a.data_root / "HUST" / "COMPLETE.json").read_text())
    policy_columns = complete["policy_columns"]
    with np.load(a.arrays, allow_pickle=False) as z:
        all_ids = z["ids"].astype(str).tolist(); rows = [all_ids.index(x) for x in ids]
        raw = z["raw_viq"][rows]; mask = z["point_mask"][rows]; anchor = z["anchor"][rows].astype(float); policy = z["policy"][rows].astype(float)
    capacity = np.where(mask, raw[:, :, :, 2], np.nan).mean(axis=2)
    prefix = pd.DataFrame({"cell_uid": ids, "K100": anchor, "early_slope": [np.polyfit(np.arange(1,101), row, 1)[0] if np.isfinite(row).all() else np.nan for row in capacity], "capacity_drift": np.nanmean(capacity[:,-20:],1)-np.nanmean(capacity[:,:20],1), "protocol_uid": meta["protocol_uid"].to_numpy()})
    for name in ("chamber_design_temperature_C", "formation_temperature_C", "observed_prefix_discharge_final_C"):
        if name in policy_columns:
            prefix[name] = policy[:, policy_columns.index(name)]
    for c in ("K100", "early_slope", "capacity_drift"):
        prefix[c+"_group"] = pd.qcut(prefix[c], 2, labels=["low", "high"], duplicates="drop")
    values, observed = load_future_rows("HUST", a.data_root, ids, load_targets=True)
    residual_rows=[]; window_rows=[]
    for seed in (123,3407):
        for fold in range(3):
            base=a.runtime_root/f"seed{seed}"/"head_aware_target_v1"/"HUST"/"C_HUST"
            for method in METHODS:
                pred=np.load(base/f"fold{fold}__{method}.npz",allow_pickle=False)["prediction"]
                _,_,_,cm,cs=score(pred,values,observed)
                residual_rows.extend({"seed":seed,"fold":fold,"method":method,"cell_uid":uid,"mae":float(x),"mse":float(y)} for uid,x,y in zip(ids,cm,cs))
                for lo,hi in WINDOWS:
                    sl=slice(lo-101,hi-100); m=observed[:,sl]; d=np.where(m,pred[:,sl]-values[:,sl],0); n=m.sum()
                    window_rows.append({"seed":seed,"fold":fold,"method":method,"window":f"{lo}_{hi}","mae":float(np.abs(d).sum()/n),"rmse":float(np.sqrt((d*d).sum()/n)),"bias":float(d.sum()/n),"observed_points":int(n)})
    residual=pd.DataFrame(residual_rows); windows=pd.DataFrame(window_rows)
    wide=residual.groupby(["cell_uid","method"],as_index=False).mae.mean().pivot(index="cell_uid",columns="method",values="mae")
    gain=pd.DataFrame({"cell_uid":wide.index,"gain_vs_HEAD_ONLY_TRAIN":wide["HEAD_ONLY_TRAIN"]-wide["HEAD_BRIDGE_LORA"],"gain_vs_HEAD_LORA":wide["HEAD_LORA"]-wide["HEAD_BRIDGE_LORA"]}).reset_index(drop=True).merge(prefix,on="cell_uid")
    with np.load(a.source_prefix,allow_pickle=False) as z:
        sx=z["real_features"][:,:,:12].astype(float).mean(1)
    tx=np.concatenate((capacity.mean(1)[:,None],capacity.std(1)[:,None],(capacity[:,-20:].mean(1)-capacity[:,:20].mean(1))[:,None]),1)
    sx=np.concatenate((sx.mean(0)[None,:],sx.std(0)[None,:],np.zeros((1,sx.shape[1]))),1); tx=np.pad(tx,((0,0),(0,sx.shape[1]-tx.shape[1])))
    shift=np.sqrt(((tx-sx.mean(0))**2).sum(1)); gain["feature_shift"] = shift
    relation=[]
    for col in ("gain_vs_HEAD_ONLY_TRAIN","gain_vs_HEAD_LORA"):
        valid = np.isfinite(gain[col].to_numpy()) & np.isfinite(gain.feature_shift.to_numpy())
        relation.append({"gain":col,"spearman_r":float(spearmanr(gain.loc[valid, col],gain.loc[valid, "feature_shift"]).statistic),"cells":int(valid.sum())})
    prefix.to_csv(a.public/"HUST_PREFIX_GROUPS.csv",index=False); residual.to_csv(a.public/"HUST_CELL_RESIDUALS.csv",index=False); windows.to_csv(a.public/"HUST_WINDOW_RESIDUALS.csv",index=False); gain.to_csv(a.public/"HUST_BRIDGE_GAIN.csv",index=False)
    receipt={"status":"COMPLETE_E3_HUST_RETROSPECTIVE_AUDIT","target_query_count":len(ids),"eol80_events":int(meta["eol80_event_observed"].sum()),"methods":list(METHODS),"seeds":[123,3407],"feature_shift_relations":relation,"eol80_grouping":"forbidden because events are 0/77"}
    (a.public/"E3_HUST_RECEIPT.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
