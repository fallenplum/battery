"""Score the frozen common warmup model without installing any adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.drs_peft.data import CACHE, file_hash, write_json
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import (
    full_prediction,
    load_common_state,
    role_bundle,
)
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE

GRID = np.arange(101, 5001, dtype=np.float32)
WINDOWS = (
    ("all", 101, 5000),
    ("101_200", 101, 200),
    ("201_500", 201, 500),
    ("501_1000", 501, 1000),
    ("1001_5000", 1001, 5000),
)


def score(prediction, role, role_name, channels):
    error = prediction.astype(np.float64) - role.targets
    rows = []
    for window, left, right in WINDOWS:
        mask = role.observed & (GRID[None, :] >= left) & (GRID[None, :] <= right)
        cells = mask.any(1)
        values = np.where(mask, error, 0.0)
        count = mask.sum(1)
        mae = np.abs(values).sum(1) / np.maximum(count, 1)
        mse = np.square(values).sum(1) / np.maximum(count, 1)
        for i, cell in enumerate(cells):
            rows.append(
                {
                    "cell_uid": role.ids[i],
                    "role": role_name,
                    "channels": channels,
                    "window": window,
                    "n_obs": int(count[i]),
                    "mae": float(mae[i]) if cell else None,
                    "mse": float(mse[i]) if cell else None,
                }
            )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--public", type=Path, default=PUBLIC)
    p.add_argument("--private", type=Path, default=PRIVATE)
    p.add_argument("--features", type=Path, default=PRIVATE)
    p.add_argument("--data-root", type=Path, default=CACHE)
    p.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    configure()
    cfg, plan, *_ = build_plan(CONFIG)
    rows = []
    for fold in plan["folds"]:
        ids = fold["tune_ids"] + fold["screen_ids"]
        bundle = load_bundle(a.features / "PAIRED_PREFIXES.npz", ids)
        vals, masks = load_future_rows("MATR", a.data_root, ids)
        labels = Labels(vals, masks, ids)
        model = (
            __import__(
                "models.drs_peft.self_reference_model", fromlist=["SharedBatteryModel"]
            )
            .SharedBatteryModel(a.backbone_dir)
            .to(a.device)
        )
        load_common_state(
            model,
            torch.load(
                a.private / f"COMMON_FOLD{fold['fold']}.pt",
                map_location="cpu",
                weights_only=True,
            ),
        )
        model.eval()
        for role_name, role_ids in (
            ("tune", fold["tune_ids"]),
            ("screen", fold["screen_ids"]),
        ):
            role = role_bundle(bundle, role_ids, labels, a.device)
            for channels in (12, 30):
                prediction, seconds = full_prediction(
                    model, role, channels, cfg["train_batch_size"], a.device
                )
                rows.extend(
                    {
                        "fold": fold["fold"],
                        "method": f"WARMUP_ONLY_{channels}",
                        "inference_seconds": seconds,
                        **row,
                    }
                    for row in score(prediction, role, role_name, channels)
                )
        del model
    result = pd.DataFrame(rows)
    a.public.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.public / "WARMUP_ONLY_SCORES.csv", index=False, lineterminator="\n")
    summary = (
        result[result.n_obs > 0]
        .groupby(["role", "method", "window"], as_index=False)
        .agg(mae=("mae", "mean"), mse=("mse", "mean"), cells=("cell_uid", "nunique"))
    )
    summary.to_csv(
        a.public / "WARMUP_ONLY_SUMMARY.csv", index=False, lineterminator="\n"
    )
    write_json(
        a.public / "WARMUP_ONLY_RECEIPT.json",
        {
            "status": "COMPLETE_SOURCE_WARMUP_ONLY",
            "future_label_scope": "source fit/tune/screen only",
            "outer_labels_loaded": 0,
            "target_labels_loaded": 0,
            "common_checkpoint_sha256": {
                str(f["fold"]): file_hash(a.private / f"COMMON_FOLD{f['fold']}.pt")
                for f in plan["folds"]
            },
            "scores_sha256": file_hash(a.public / "WARMUP_ONLY_SCORES.csv"),
            "summary_sha256": file_hash(a.public / "WARMUP_ONLY_SUMMARY.csv"),
            "rows": len(result),
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
