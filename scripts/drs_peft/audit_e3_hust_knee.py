from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from scripts.physics_targets import knee_target
from scripts.drs_peft.transfer_scope_data import load_future_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--public", type=Path, required=True)
    a = p.parse_args()
    lock = json.loads(Path("results/drs_peft/head_aware_seed_v1/seed123/target/HUST/C_HUST/PREDICTIONS_LOCKED.json").read_text())
    ids = lock["target_ids"]
    values, observed = load_future_rows("HUST", a.data_root, ids, load_targets=True)
    cycles = np.arange(101, 5001, dtype=float)
    rows = []
    for index, uid in enumerate(ids):
        valid = observed[index]
        result = knee_target(cycles[valid], values[index, valid], after_cycle=100)
        rows.append({"cell_uid": uid, "has_candidate": result["has_candidate"], "stable": result["stable"], "knee_cycle": result["knee_cycle"], "reason": result["reason"], "observed_start": result["support"].get("observed_start"), "observed_end": result["support"].get("observed_end"), "tau_range_cycles": result.get("tau_range_cycles")})
    frame = pd.DataFrame(rows)
    gain_path = a.public / "HUST_BRIDGE_GAIN.csv"
    if gain_path.exists():
        frame = frame.merge(pd.read_csv(gain_path), on="cell_uid", how="left")
    frame.to_csv(a.public / "HUST_FUTURE_KNEE_RETRO.csv", index=False, lineterminator="\n")
    receipt = {"status":"COMPLETE_E3_HUST_FUTURE_KNEE_RETROSPECTIVE", "cells":len(frame), "stable_candidates":int(frame["stable"].sum()), "has_candidates":int(frame["has_candidate"].sum()), "eol80_events":0, "eol80_grouping":"forbidden"}
    (a.public / "HUST_FUTURE_KNEE_RECEIPT.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
