"""Prepare fold-local COMMON_WARMUP states for a registered training seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE, ROOT
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import build_plan
from scripts.drs_peft.self_reference_training import role_bundle, warmup
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure()
    cfg, plan, *_ = build_plan(args.config)
    seed = int(cfg["seed"])
    args.private.mkdir(parents=True, exist_ok=True)
    args.public.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    traces = []
    checkpoints = []

    for fold in plan["folds"]:
        ids = list(
            dict.fromkeys(fold["fit_ids"] + fold["tune_ids"] + fold["screen_ids"])
        )
        bundle = load_bundle(args.features / "PAIRED_PREFIXES.npz", ids)
        values, observed = load_future_rows("MATR", args.data_root, ids)
        labels = Labels(values, observed, ids)
        fit = role_bundle(bundle, fold["fit_ids"], labels, device)
        model = SharedBatteryModel(args.backbone_dir).to(device)
        state, trace = warmup(model, fit, cfg, fold["fold"], device)
        checkpoint = args.private / f"COMMON_FOLD{fold['fold']}.pt"
        torch.save(state, checkpoint)
        checkpoints.append(
            {
                "fold": fold["fold"],
                "file": checkpoint.name,
                "sha256": sha(checkpoint),
            }
        )
        traces.extend(
            {"seed": seed, "fold": fold["fold"], **row} for row in trace
        )
        del model

    trace_path = args.public / "WARMUP_TRACE.csv"
    import pandas as pd

    pd.DataFrame(traces).to_csv(trace_path, index=False, lineterminator="\n")
    receipt = {
        "status": "COMPLETE_HEAD_AWARE_SEED_COMMON_WARMUP",
        "seed": seed,
        "config_sha256": sha(args.config),
        "source_domain": "MATR",
        "folds": [fold["fold"] for fold in plan["folds"]],
        "input_channels": [12, 30],
        "target_future_labels_read": 0,
        "outer_labels_read": 0,
        "source_future_rows_loaded": 1,
        "checkpoints": checkpoints,
        "files": {"WARMUP_TRACE.csv": sha(trace_path)},
    }
    (args.public / "WARMUP_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
