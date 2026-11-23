"""Fixed-method blind evaluation on a new BatteryLife family."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from scripts.drs_peft.data import CACHE, ROOT
from scripts.drs_peft.run_head_aware_retrospective_target import source_setup, train_head_method
from scripts.drs_peft.run_self_reference import load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, build_plan
from scripts.drs_peft.self_reference_training import GRID, Role, full_prediction
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_reference import BACKBONE


METHODS = ("HEAD_ONLY_TRAIN", "HEAD_LORA", "HEAD_BRIDGE_LORA")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_prefix_role(cache_path: Path, device: torch.device) -> Role:
    with np.load(cache_path, allow_pickle=False) as archive:
        ids = archive["ids"].astype(str).tolist()
        features = archive["real_features"]
        reference = archive["reference_features"]
        mask = archive["feature_mask"]
        cycles = archive["cycles"]
        anchors = archive["real_anchor"]
    n = len(ids)
    targets = np.zeros((n, len(GRID)), dtype=np.float32)
    observed = np.zeros((n, len(GRID)), dtype=bool)
    return Role(
        ids,
        torch.as_tensor(features, device=device),
        torch.as_tensor(reference, device=device),
        torch.as_tensor(mask, device=device),
        torch.as_tensor(cycles, device=device),
        torch.as_tensor(anchors, device=device),
        targets,
        observed,
    )


def load_future_from_pickles(dataset_root: Path, family: str, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    lookup = {}
    for path in sorted((dataset_root / family).glob("*.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        cell_id = str(payload["cell_id"])
        if cell_id in ids:
            lookup[cell_id] = payload
    if set(ids) != set(lookup):
        raise RuntimeError("The blind cache IDs do not match the raw family files")
    values = np.zeros((len(ids), len(GRID)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=bool)
    for row, cell_id in enumerate(ids):
        payload = lookup[cell_id]
        nominal = float(payload["nominal_capacity_in_Ah"])
        for cycle in payload["cycle_data"]:
            number = int(cycle.get("cycle_number", 0))
            if not 101 <= number <= 5000:
                continue
            capacity = np.asarray(cycle.get("discharge_capacity_in_Ah", []), dtype=float)
            finite = capacity[np.isfinite(capacity)]
            if not len(finite):
                continue
            values[row, number - 101] = float(np.max(finite) / nominal)
            observed[row, number - 101] = True
    if not observed.any(axis=1).all():
        raise RuntimeError("At least one blind cell has no observed future target")
    return values, observed


def score(prediction: np.ndarray, values: np.ndarray, observed: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    diff = np.where(observed, prediction - values, 0.0)
    counts = observed.sum(axis=1)
    mae = np.abs(diff).sum(axis=1) / counts
    mse = (diff**2).sum(axis=1) / counts
    return float(mae.mean()), float(np.sqrt(mse).mean()), float(mse.mean()), mae, mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--prefix-cache", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-private", type=Path, default=PRIVATE)
    parser.add_argument("--source-features", type=Path, default=PRIVATE)
    parser.add_argument("--data-root", type=Path, default=CACHE)
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
    args.public.mkdir(parents=True, exist_ok=True)
    args.private.mkdir(parents=True, exist_ok=True)
    ids = list(target_role.ids)
    common_hashes = {
        str(fold["fold"]): sha256(args.source_private / f"COMMON_FOLD{fold['fold']}.pt")
        for fold in source_plan["folds"]
    }
    lock = {
        "status": "LOCKED_BEFORE_NEW_FAMILY_PREDICTION",
        "family": args.family,
        "methods": list(METHODS),
        "target_ids": ids,
        "input_channels": 12,
        "prediction_grid": [101, 5000],
        "source_config_sha256": sha256(args.config),
        "prefix_cache_sha256": sha256(args.prefix_cache),
        "common_checkpoint_sha256": common_hashes,
        "selection": "source MATR fit/tune only, minimum tune MAE then lower learning rate",
        "target_future_labels_read": 0,
        "prediction_status": "PENDING",
        "historical_exposure_risk": "The family is absent from the current Head-aware method-selection records. Generic BatteryLife baseline exposure is not independently ruled out.",
    }
    lock_path = args.public / "BLIND_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    predictions_dir = args.private / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
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
            model, selected, parameter_count, groups = train_head_method(
                common, method, fit, tune, cfg, int(fold["fold"]), device
            )
            model.eval()
            prediction, inference_seconds = full_prediction(
                model, target_role, 12, cfg["train_batch_size"], device
            )
            key = f"fold{fold['fold']}__{method}"
            path = predictions_dir / f"{key}.npz"
            np.savez_compressed(path, prediction=prediction, target_ids=np.asarray(ids))
            ledger.append(
                {
                    "fold": int(fold["fold"]),
                    "method": method,
                    "best_lr": float(selected["lr"]),
                    "best_epoch": int(selected["epoch"]),
                    "selected_tune_mae": float(selected["tune_mae"]),
                    "parameter_count": int(parameter_count),
                    "common_checkpoint_sha256": sha256(common_path),
                    "prediction_sha256": array_hash(prediction),
                    "prediction_file_sha256": sha256(path),
                    "inference_seconds": float(inference_seconds),
                    "target_future_labels_read": 0,
                }
            )
            del model
        del common
    ledger_path = args.public / "SOURCE_SELECTION_PREDICTION_LEDGER.csv"
    pd.DataFrame(ledger).to_csv(ledger_path, index=False, lineterminator="\n")
    prediction_lock = {
        "status": "PREDICTIONS_LOCKED_BEFORE_TARGET_LABELS",
        "family": args.family,
        "methods": list(METHODS),
        "target_ids": ids,
        "source_ledger_sha256": sha256(ledger_path),
        "prediction_files_locked": True,
        "target_future_labels_read": 0,
    }
    prediction_lock_path = args.public / "PREDICTIONS_LOCKED.json"
    prediction_lock_path.write_text(json.dumps(prediction_lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    values, observed = load_future_from_pickles(args.dataset_root, args.family, ids)
    rows, cells = [], []
    for rec in ledger:
        prediction = np.load(predictions_dir / f"fold{rec['fold']}__{rec['method']}.npz", allow_pickle=False)["prediction"]
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
            }
        )
        for uid, cell_mae_value, cell_mse_value, count in zip(ids, cell_mae, cell_mse, observed.sum(axis=1)):
            cells.append(
                {
                    "fold": rec["fold"],
                    "method": rec["method"],
                    "cell_uid": uid,
                    "mae": float(cell_mae_value),
                    "mse": float(cell_mse_value),
                    "observed_future_points": int(count),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.public / "BLIND_SCORES.csv", index=False, lineterminator="\n")
    pd.DataFrame(cells).to_csv(args.public / "BLIND_CELL.csv", index=False, lineterminator="\n")
    summary = frame.groupby("method", as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), mse=("mse", "mean"), folds=("fold", "nunique"), target_cells=("target_cells", "first")
    )
    summary.to_csv(args.public / "BLIND_SUMMARY.csv", index=False, lineterminator="\n")
    receipt = {
        "status": "COMPLETE_NEW_FAMILY_BLIND_SCORE",
        "family": args.family,
        "methods": list(METHODS),
        "target_future_labels_read": 1,
        "prediction_lock_sha256": sha256(prediction_lock_path),
        "files": {name: sha256(args.public / name) for name in ("BLIND_LOCK.json", "PREDICTIONS_LOCKED.json", "SOURCE_SELECTION_PREDICTION_LEDGER.csv", "BLIND_SCORES.csv", "BLIND_CELL.csv", "BLIND_SUMMARY.csv")},
    }
    (args.public / "BLIND_SCORE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

