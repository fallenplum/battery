"""Report warmup-only and controlled Full-FT-12 source baselines."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE, file_hash, write_json
from scripts.drs_peft.run_abba_moirai import csv
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import (
    Trial,
    batches,
    build_trial,
    fit_trial,
    full_prediction,
    load_common_state,
    observed_mae,
    role_bundle,
    training_targets,
)
from scripts.drs_peft.train import bind, configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE


class Labels:
    def __init__(self, values, observed, names):
        self.values, self.observed, self.names = values, observed, names

    def arrays(self, ids):
        rows = [self.names.index(uid) for uid in ids]
        return self.values[rows], self.observed[rows]


def load_bundle(path, ids):
    with np.load(path, allow_pickle=False) as archive:
        names = archive["ids"].astype(str).tolist()
        rows = [names.index(uid) for uid in ids]
        return {
            "ids": np.asarray(ids),
            **{
                key: archive[key][rows]
                for key in archive.files
                if key not in ("ids", "columns")
            },
        }


def fit_full(model, role, tune, cfg, fold, device):
    model.train()
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    rows, snapshots = [], {}
    for epoch in range(1, 9):
        started = time.perf_counter()
        total = 0.0
        for index in batches(
            len(role.ids),
            cfg["train_batch_size"],
            seed=cfg["seed"] + 1009 * epoch + fold,
        ):
            future, target, observed = training_targets(role, index, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*role.arrays(index), future, channels=12)
            error = torch.where(observed, prediction - target, 0.0)
            loss = (error.abs().sum(dim=1) / observed.sum(dim=1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(index)
        model.eval()
        prediction, inference_seconds = full_prediction(
            model, tune, 12, cfg["train_batch_size"], device
        )
        rows.append(
            {
                "fold": fold,
                "method": "FULL_FT_12",
                "epoch": epoch,
                "lr": 1e-4,
                "training_mae": total / len(role.ids),
                "tune_mae": float(observed_mae(prediction, tune).mean()),
                "training_seconds": time.perf_counter() - started,
                "tune_inference_seconds": inference_seconds,
                "trainable_parameters": sum(
                    value.numel() for value in model.parameters() if value.requires_grad
                ),
            }
        )
        if epoch in (2, 4, 8):
            snapshots[epoch] = copy.deepcopy(model.state_dict())
    return rows, snapshots


def run_fold(args, cfg, fold, device):
    ids = list(dict.fromkeys(fold["fit_ids"] + fold["tune_ids"] + fold["screen_ids"]))
    bundle = load_bundle(args.features / "PAIRED_PREFIXES.npz", ids)
    targets, observed = load_future_rows("MATR", args.data_root, ids)
    labels = Labels(targets, observed, ids)
    model = SharedBatteryModel(args.backbone_dir).to(device)
    fit = role_bundle(bundle, fold["fit_ids"], labels, device)
    tune = role_bundle(bundle, fold["tune_ids"], labels, device)
    screen = role_bundle(bundle, fold["screen_ids"], labels, device)
    common_path = args.private.parent / f"COMMON_FOLD{fold['fold']}.pt"
    common_state = torch.load(common_path, map_location="cpu", weights_only=True)
    load_common_state(model, common_state)
    warmup_trace = [
        {"fold": fold["fold"], "source": "locked_COMMON_FOLD", "path": str(common_path)}
    ]
    warmup_prediction, warmup_seconds = full_prediction(
        model, screen, 12, cfg["train_batch_size"], device
    )
    scores = [
        {
            "fold": fold["fold"],
            "method": "WARMUP_ONLY_12",
            "epoch": 0,
            "screen_mae": float(observed_mae(warmup_prediction, screen).mean()),
            "screen_cells": len(screen.ids),
            "inference_seconds": warmup_seconds,
            "trainable_parameters": 0,
        }
    ]
    full_model = copy.deepcopy(model)
    full_rows, snapshots = fit_full(full_model, fit, tune, cfg, fold["fold"], device)
    for row in full_rows:
        if row["epoch"] in (2, 4, 8):
            state = snapshots[row["epoch"]]
            full_model.load_state_dict(state)
            prediction, seconds = full_prediction(
                full_model, screen, 12, cfg["train_batch_size"], device
            )
            scores.append(
                {
                    "fold": fold["fold"],
                    "method": "FULL_FT_12",
                    "epoch": row["epoch"],
                    "lr": 1e-4,
                    "screen_mae": float(observed_mae(prediction, screen).mean()),
                    "screen_cells": len(screen.ids),
                    "inference_seconds": seconds,
                    "trainable_parameters": sum(
                        value.numel() for value in full_model.parameters()
                    ),
                }
            )
    # Same-warmup LoRA-12 control. Its old screen artifact predates the
    # strict 12-channel split and is intentionally not reused here.
    lora_model, lora_parameters = build_trial(
        model, Trial("LORA_12", 0.001, 0.1), {0.1: {}}, cfg, device
    )
    _, _, lora_states, _ = fit_trial(
        lora_model,
        lora_parameters,
        Trial("LORA_12", 0.001, 0.1),
        fit,
        tune,
        cfg,
        fold["fold"],
        device,
    )
    lora_state = lora_states[max(lora_states)]
    with torch.no_grad():
        for name, parameter in lora_parameters.items():
            parameter.copy_(lora_state[name])
    prediction, seconds = full_prediction(
        lora_model, screen, 12, cfg["train_batch_size"], device
    )
    scores.append(
        {
            "fold": fold["fold"],
            "method": "LORA_12_SAME_WARMUP",
            "epoch": max(lora_states),
            "lr": 0.001,
            "screen_mae": float(observed_mae(prediction, screen).mean()),
            "screen_cells": len(screen.ids),
            "inference_seconds": seconds,
            "trainable_parameters": sum(p.numel() for p in lora_parameters.values()),
        }
    )
    del lora_model
    out = args.public / f"fold{fold['fold']}"
    out.mkdir(parents=True, exist_ok=True)
    csv(out / "BASELINE_SCORES.csv", scores)
    csv(out / "FULL_FT_TRACE.csv", full_rows)
    csv(out / "WARMUP_TRACE.csv", warmup_trace)
    write_json(
        out / "COMMON_STATE.json",
        {"state_keys": len(common_state), "status": "COMPLETE"},
    )
    write_json(
        out / "COMPLETE.json",
        {
            "status": "COMPLETE_SOURCE_BASELINES",
            "fold": fold["fold"],
            "target_future_rows_loaded": 0,
            "outer_labels_loaded": 0,
            "files": {
                name: file_hash(out / name)
                for name in (
                    "BASELINE_SCORES.csv",
                    "FULL_FT_TRACE.csv",
                    "WARMUP_TRACE.csv",
                    "COMMON_STATE.json",
                )
            },
        },
    )
    print(f"baseline fold={fold['fold']} complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("register", "run"), default="run")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--public", type=Path, default=PUBLIC / "baselines")
    parser.add_argument("--private", type=Path, default=PRIVATE / "baselines")
    parser.add_argument("--features", type=Path, default=PRIVATE)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure()
    cfg, plan, *_ = build_plan(args.config)
    args.public.mkdir(parents=True, exist_ok=True)
    args.private.mkdir(parents=True, exist_ok=True)
    bind(args.public / "REGISTERED_PLAN.json", plan)
    if args.phase == "register":
        return
    device = torch.device(args.device)
    for fold in plan["folds"]:
        run_fold(args, cfg, fold, device)
    scores = pd.concat(
        [
            pd.read_csv(args.public / f"fold{fold['fold']}" / "BASELINE_SCORES.csv")
            for fold in plan["folds"]
        ],
        ignore_index=True,
    )
    csv(args.public / "BASELINE_SCORES.csv", scores)
    write_json(
        args.public / "FINAL_DECISION.json",
        {
            "status": "COMPLETE_SOURCE_BASELINE_SCREEN",
            "target_future_rows_loaded": 0,
            "methods": ["WARMUP_ONLY_12", "FULL_FT_12"],
            "files": {
                "BASELINE_SCORES.csv": file_hash(args.public / "BASELINE_SCORES.csv")
            },
        },
    )


if __name__ == "__main__":
    main()
