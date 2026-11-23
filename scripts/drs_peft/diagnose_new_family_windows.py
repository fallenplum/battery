"""Four-window and cell-level residual diagnostics after a blind lock."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


GRID = np.arange(101, 5001, dtype=np.int64)
WINDOWS = {
    "101_200": (101, 200),
    "201_500": (201, 500),
    "501_1000": (501, 1000),
    "1001_5000": (1001, 5000),
}


def load_targets(dataset_root: Path, family: str, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    payloads = {}
    for path in sorted((dataset_root / family).glob("*.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if str(payload.get("cell_id")) in ids:
            payloads[str(payload["cell_id"])] = payload
    values = np.zeros((len(ids), len(GRID)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=bool)
    for row, cell_id in enumerate(ids):
        payload = payloads[cell_id]
        nominal = float(payload["nominal_capacity_in_Ah"])
        for cycle in payload["cycle_data"]:
            n = int(cycle.get("cycle_number", 0))
            if not 101 <= n <= 5000:
                continue
            q = np.asarray(cycle.get("discharge_capacity_in_Ah", []), dtype=float)
            q = q[np.isfinite(q)]
            if len(q):
                values[row, n - 101] = float(q.max() / nominal)
                observed[row, n - 101] = True
    return values, observed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--private", type=Path, required=True)
    args = p.parse_args()
    lock = json.loads((args.public / "BLIND_LOCK.json").read_text(encoding="utf-8"))
    prediction_lock = json.loads((args.public / "PREDICTIONS_LOCKED.json").read_text(encoding="utf-8"))
    if not prediction_lock.get("prediction_files_locked") or prediction_lock.get("target_future_labels_read") != 0:
        raise RuntimeError("Prediction lock is missing or was written after labels were opened")
    ids = lock["target_ids"]
    values, observed = load_targets(args.dataset_root, args.family, ids)
    rows, cells = [], []
    for prediction_path in sorted(args.private.glob("predictions/fold*__*.npz")):
        stem = prediction_path.stem
        fold, method = stem.split("__", 1)
        fold = int(fold.replace("fold", ""))
        prediction = np.load(prediction_path, allow_pickle=False)["prediction"]
        residual = prediction - values
        for window, (lo, hi) in WINDOWS.items():
            mask = observed & (GRID[None, :] >= lo) & (GRID[None, :] <= hi)
            count = mask.sum(axis=1)
            valid = count > 0
            mae_cell = np.abs(np.where(mask, residual, 0.0)).sum(axis=1) / np.maximum(count, 1)
            mse_cell = (np.where(mask, residual, 0.0) ** 2).sum(axis=1) / np.maximum(count, 1)
            rows.append({
                "fold": fold,
                "method": method,
                "window": window,
                "mae": float(mae_cell[valid].mean()),
                "rmse": float(np.sqrt(mse_cell[valid]).mean()),
                "mse": float(mse_cell[valid].mean()),
                "cells": int(valid.sum()),
                "observed_points": int(mask.sum()),
            })
            for row_idx, (uid, value, mse_value, n) in enumerate(zip(ids, mae_cell, mse_cell, count)):
                cells.append({
                    "fold": fold,
                    "method": method,
                    "cell_uid": uid,
                    "window": window,
                    "mae": float(value) if n else np.nan,
                    "rmse": float(np.sqrt(mse_value)) if n else np.nan,
                    "bias": float(np.where(mask[row_idx], residual[row_idx], 0.0).sum() / n) if n else np.nan,
                    "observed_points": int(n),
                })
    frame = pd.DataFrame(rows)
    cell_frame = pd.DataFrame(cells)
    frame.to_csv(args.public / "WINDOW_SCORES.csv", index=False, lineterminator="\n")
    cell_frame.to_csv(args.public / "CELL_RESIDUALS.csv", index=False, lineterminator="\n")
    frame.groupby(["method", "window"], as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), mse=("mse", "mean"), folds=("fold", "nunique"), cells=("cells", "first"), observed_points=("observed_points", "first")
    ).to_csv(args.public / "WINDOW_SUMMARY.csv", index=False, lineterminator="\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

