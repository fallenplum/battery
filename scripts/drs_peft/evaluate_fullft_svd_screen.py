"""Evaluate truncated SVD approximations of source-only Full-FT updates."""
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

from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import (
    full_prediction,
    load_common_state,
    observed_mae,
    role_bundle,
)
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE

RANKS = (1, 4, 8, 16, 32)
EPOCH = 8


def load_parameters(model, state):
    with torch.no_grad():
        for name, value in state.items():
            dict(model.named_parameters())[name].copy_(value)


def low_rank_update(delta, rank):
    if delta.ndim != 2:
        return delta
    if not torch.any(delta):
        return delta
    left, values, right = torch.linalg.svd(delta.double(), full_matrices=False)
    k = min(rank, values.numel())
    return ((left[:, :k] * values[:k]) @ right[:k, :]).to(delta.dtype)


def truncated_state(warm, final, rank, retain_non2d):
    result = {}
    for name, value in warm.items():
        delta = final[name] - value
        if delta.ndim != 2 and not retain_non2d:
            result[name] = value
        else:
            result[name] = value + low_rank_update(delta, rank)
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_fold(args, cfg, fold, device):
    ids = list(dict.fromkeys(fold["fit_ids"] + fold["tune_ids"] + fold["screen_ids"]))
    bundle = load_bundle(args.features / "PAIRED_PREFIXES.npz", ids)
    target, observed = load_future_rows("MATR", args.data_root, ids)
    labels = Labels(target, observed, ids)
    screen = role_bundle(bundle, fold["screen_ids"], labels, device)

    common = torch.load(
        args.private / f"COMMON_FOLD{fold['fold']}.pt",
        map_location="cpu",
        weights_only=True,
    )
    root = args.private / f"fold{fold['fold']}"
    warm = torch.load(root / "FULL_FT_EPOCH0.pt", map_location="cpu", weights_only=True)["state"]
    final = torch.load(root / f"FULL_FT_EPOCH{EPOCH}.pt", map_location="cpu", weights_only=True)["state"]

    base = SharedBatteryModel(args.backbone_dir).to(device)
    load_common_state(base, common)
    load_parameters(base, warm)

    rows = []
    states = [("WARMUP_ONLY_12", warm)]
    for rank in RANKS:
        states.append(
            (
                f"FULL_FT_SVD_RANK{rank}_2D_ONLY",
                truncated_state(warm, final, rank, retain_non2d=False),
            )
        )
        states.append(
            (
                f"FULL_FT_SVD_RANK{rank}_WITH_NON2D",
                truncated_state(warm, final, rank, retain_non2d=True),
            )
        )
    states.append(("FULL_FT_EXACT_EPOCH8", final))
    for method, state in states:
        model = copy.deepcopy(base)
        load_parameters(model, state)
        prediction, seconds = full_prediction(
            model, screen, 12, cfg["train_batch_size"], device
        )
        cell_mae = observed_mae(prediction, screen)
        residual = np.where(screen.observed, prediction - screen.targets, 0.0)
        screen_rmse = float(
            np.sqrt(np.sum(residual**2) / np.sum(screen.observed))
        )
        rows.append(
            {
                "fold": fold["fold"],
                "method": method,
                "rank": 0
                if method == "WARMUP_ONLY_12"
                else (
                    np.nan
                    if method == "FULL_FT_EXACT_EPOCH8"
                    else int(method.split("RANK", 1)[1].split("_", 1)[0])
                ),
                "screen_mae": float(cell_mae.mean()),
                "screen_rmse": screen_rmse,
                "screen_cells": len(screen.ids),
                "cell_mae_median": float(np.median(cell_mae)),
                "inference_seconds": seconds,
            }
        )
        del model
    print(f"fold {fold['fold']} complete", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--public", type=Path, default=PUBLIC / "fullft_update_audit")
    parser.add_argument("--features", type=Path, default=PRIVATE)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure()
    cfg, plan, *_ = build_plan(args.config)
    args.public.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    for fold in plan["folds"]:
        rows.extend(run_fold(args, cfg, fold, device))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.public / "SVD_SCREEN_SCORES.csv", index=False, lineterminator="\n")
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            screen_mae=("screen_mae", "mean"),
            screen_rmse=("screen_rmse", "mean"),
            cell_mae_median=("cell_mae_median", "mean"),
            folds=("fold", "nunique"),
            screen_cells=("screen_cells", "sum"),
        )
        .sort_values("screen_mae")
    )
    summary.to_csv(args.public / "SVD_SCREEN_SUMMARY.csv", index=False, lineterminator="\n")
    receipt = {
        "status": "COMPLETE_SOURCE_FULLFT_SVD_SCREEN_DIAGNOSTIC",
        "domain": "MATR",
        "folds": [fold["fold"] for fold in plan["folds"]],
        "epoch": EPOCH,
        "ranks": list(RANKS),
        "non_2d_update_policies": ["zero_for_2D_ONLY", "retain_exact_full_ft_delta"],
        "target_future_labels_read": 0,
        "outer_labels_read": 0,
        "files": {
            name: digest(args.public / name)
            for name in ("SVD_SCREEN_SCORES.csv", "SVD_SCREEN_SUMMARY.csv")
        },
    }
    (args.public / "SVD_SCREEN_COMPLETE.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
