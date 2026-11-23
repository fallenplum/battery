"""Retrospective baseline supplement for already exposed new-family labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from scripts.drs_peft.run_head_aware_new_family_blind import load_future_from_pickles, load_prefix_role, score
from scripts.drs_peft.run_head_aware_retrospective_target import source_setup, train_full_method, train_lora_method
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, build_plan
from scripts.drs_peft.self_reference_training import full_prediction
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_reference import BACKBONE


METHODS = ("WARMUP_ONLY", "LORA_12", "FULL_FT_TUNED")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--prefix-cache", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-private", type=Path, default=PRIVATE)
    parser.add_argument("--source-features", type=Path, default=PRIVATE)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure()
    cfg, source_plan, *_ = build_plan(args.config)
    device = torch.device(args.device)
    target_role = load_prefix_role(args.prefix_cache, device)
    ids = list(target_role.ids)
    args.public.mkdir(parents=True, exist_ok=True)
    args.private.mkdir(parents=True, exist_ok=True)
    lock = {
        "status": "RETROSPECTIVE_BASELINE_LOCKED_BEFORE_SCORING",
        "family": args.family,
        "methods": list(METHODS),
        "target_ids": ids,
        "input_channels": 12,
        "source_config_sha256": sha256(args.config),
        "prefix_cache_sha256": sha256(args.prefix_cache),
        "selection": "MATR source fit/tune only",
        "target_future_labels_read": 0,
        "evaluation_status": "RETROSPECTIVE_PUBLIC_LABEL_SUPPLEMENT",
    }
    (args.public / "RETRO_BASELINE_LOCK.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prediction_dir = args.private / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    ledger = []
    for fold in source_plan["folds"]:
        fit, tune, common, common_path = source_setup(
            argparse.Namespace(
                features=args.source_features,
                private=args.source_private,
                data_root=args.data_root,
                backbone_dir=args.backbone_dir,
            ),
            fold,
            device,
        )
        for method in METHODS:
            if method == "WARMUP_ONLY":
                model, selected, parameter_count = common, {"lr": 0.0, "epoch": 0, "tune_mae": None}, 0
            elif method == "LORA_12":
                model, selected, parameter_count = train_lora_method(common, fit, tune, cfg, int(fold["fold"]), device)
            else:
                model, selected, parameter_count = train_full_method(common, fit, tune, cfg, int(fold["fold"]), device)
            model.eval()
            prediction, inference_seconds = full_prediction(model, target_role, 12, cfg["train_batch_size"], device)
            key = f"fold{fold['fold']}__{method}"
            pred_path = prediction_dir / f"{key}.npz"
            np.savez_compressed(pred_path, prediction=prediction, target_ids=np.asarray(ids))
            ledger.append(
                {
                    "fold": int(fold["fold"]),
                    "method": method,
                    "best_lr": float(selected["lr"]),
                    "best_epoch": int(selected["epoch"]),
                    "selected_tune_mae": None if selected.get("tune_mae") is None else float(selected["tune_mae"]),
                    "parameter_count": int(parameter_count),
                    "common_checkpoint_sha256": sha256(common_path),
                    "prediction_sha256": array_hash(prediction),
                    "prediction_file_sha256": sha256(pred_path),
                    "inference_seconds": float(inference_seconds),
                    "target_future_labels_read": 0,
                }
            )
            if method != "WARMUP_ONLY":
                del model
        del common
    ledger_path = args.public / "RETRO_BASELINE_LEDGER.csv"
    pd.DataFrame(ledger).to_csv(ledger_path, index=False, lineterminator="\n")
    prediction_lock = {
        "status": "RETROSPECTIVE_PREDICTIONS_LOCKED_BEFORE_SCORING",
        "family": args.family,
        "methods": list(METHODS),
        "target_ids": ids,
        "ledger_sha256": sha256(ledger_path),
        "target_future_labels_read": 0,
    }
    prediction_lock_path = args.public / "RETRO_PREDICTIONS_LOCKED.json"
    prediction_lock_path.write_text(json.dumps(prediction_lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    values, observed = load_future_from_pickles(args.dataset_root, args.family, ids)
    rows, cells = [], []
    for rec in ledger:
        prediction = np.load(prediction_dir / f"fold{rec['fold']}__{rec['method']}.npz", allow_pickle=False)["prediction"]
        mae, rmse, mse, cell_mae, cell_mse = score(prediction, values, observed)
        rows.append(
            {
                "fold": rec["fold"],
                "method": rec["method"],
                "mae": mae,
                "rmse": rmse,
                "mse": mse,
                "target_cells": len(ids),
                "observed_future_points": int(observed.sum()),
                "target_future_labels_read": 1,
                "evaluation_type": "retrospective",
            }
        )
        for uid, cell_mae, cell_mse, count in zip(ids, cell_mae, cell_mse, observed.sum(axis=1)):
            cells.append({"fold": rec["fold"], "method": rec["method"], "cell_uid": uid, "mae": float(cell_mae), "mse": float(cell_mse), "observed_future_points": int(count)})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.public / "RETRO_BASELINE_SCORES.csv", index=False, lineterminator="\n")
    pd.DataFrame(cells).to_csv(args.public / "RETRO_BASELINE_CELL.csv", index=False, lineterminator="\n")
    summary = frame.groupby("method", as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"), mse=("mse", "mean"), folds=("fold", "nunique"), target_cells=("target_cells", "first"))
    summary.to_csv(args.public / "RETRO_BASELINE_SUMMARY.csv", index=False, lineterminator="\n")
    receipt = {
        "status": "COMPLETE_RETROSPECTIVE_BASELINE_SUPPLEMENT",
        "family": args.family,
        "methods": list(METHODS),
        "target_future_labels_read": 1,
        "evaluation_type": "retrospective_public_labels",
        "prediction_lock_sha256": sha256(prediction_lock_path),
        "files": {name: sha256(args.public / name) for name in ("RETRO_BASELINE_LOCK.json", "RETRO_PREDICTIONS_LOCKED.json", "RETRO_BASELINE_LEDGER.csv", "RETRO_BASELINE_SCORES.csv", "RETRO_BASELINE_CELL.csv", "RETRO_BASELINE_SUMMARY.csv")},
    }
    (args.public / "RETRO_BASELINE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

