"""Source-only supervised screen for self-reference input projections."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.drs_peft.self_reference_adapters import (
    collect_paired_activations,
    fit_projection_bases,
)
from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE, file_hash, write_json
from scripts.drs_peft.run_abba_moirai import csv
from scripts.drs_peft.self_reference_protocol import (
    CONFIG,
    METHODS,
    PRIVATE,
    PUBLIC,
    build_plan,
)
from scripts.drs_peft.self_reference_training import (
    Trial,
    build_trial,
    fit_trial,
    full_prediction,
    observed_mae,
    role_bundle,
    warmup,
)
from scripts.drs_peft.train import bind, configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE


def load_bundle(path, ids):
    with np.load(path, allow_pickle=False) as z:
        names = z["ids"].astype(str).tolist()
        rows = [names.index(x) for x in ids]
        return {
            "ids": np.asarray(ids),
            **{k: z[k][rows] for k in z.files if k not in ("ids", "columns")},
        }


class Labels:
    def __init__(self, vals, masks, names):
        self.vals, self.masks, self.names = vals, masks, names

    def arrays(self, ids):
        rows = [self.names.index(x) for x in ids]
        return self.vals[rows], self.masks[rows]


def run_fold(args, cfg, fold, device):
    ids = fold["fit_ids"] + fold["tune_ids"] + fold["screen_ids"]
    ids = list(dict.fromkeys(ids))
    bundle = load_bundle(args.features / "PAIRED_PREFIXES.npz", ids)
    vals, masks = load_future_rows("MATR", args.data_root, ids)
    lab = Labels(vals, masks, ids)
    model = SharedBatteryModel(args.backbone_dir).to(device)
    fit = role_bundle(bundle, fold["fit_ids"], lab, device)
    tune = role_bundle(bundle, fold["tune_ids"], lab, device)
    screen = role_bundle(bundle, fold["screen_ids"], lab, device)
    state, trace = warmup(model, fit, cfg, fold["fold"], device)
    torch.save(state, args.private / f"COMMON_FOLD{fold['fold']}.pt")
    moments = collect_paired_activations(
        model.eval(), fit.features, fit.reference, fit.mask, fit.cycles
    )
    bases = {}
    audits = {}
    for ridge in cfg["self_reference_ridge_multipliers"]:
        bases[ridge], audits[str(ridge)] = fit_projection_bases(
            moments, cfg["fixed_input_rank"], ridge, cfg["seed"]
        )
    rows = []
    for method in METHODS:
        for rate in cfg["adapter_rates"]:
            ridges = (
                cfg["self_reference_ridge_multipliers"]
                if method == "SELF_REFERENCE_30"
                else [cfg["self_reference_ridge_multipliers"][1]]
            )
            for ridge in ridges:
                trial = Trial(method, rate, ridge)
                m, pars = build_trial(model, trial, bases, cfg, device)
                rec, _history, states, _preds = fit_trial(
                    m, pars, trial, fit, tune, cfg, fold["fold"], device
                )
                rows.extend(rec)
                torch.save(
                    states,
                    args.private
                    / f"fold{fold['fold']}_{trial.relative().replace('/', '_')}.pt",
                )
                del m
    frame = pd.DataFrame(rows)
    select = {}
    for method in METHODS:
        select[method] = (
            frame[frame.method.eq(method)]
            .sort_values(["tune_mae", "lr", "epoch", "ridge"])
            .iloc[0]
            .to_dict()
        )
    scores = []
    for method in METHODS:
        row = select[method]
        trial = Trial(method, float(row["lr"]), float(row["ridge"]))
        m, pars = build_trial(model, trial, bases, cfg, device)
        path = (
            args.private / f"fold{fold['fold']}_{trial.relative().replace('/', '_')}.pt"
        )
        state = torch.load(path, map_location="cpu", weights_only=True)[
            int(row["epoch"])
        ]
        with torch.no_grad():
            for n, p in pars.items():
                p.copy_(state[n])
        pred, seconds = full_prediction(
            m, screen, trial.channels, cfg["train_batch_size"], device
        )
        scores.append(
            {
                "fold": fold["fold"],
                "method": method,
                "rate": trial.rate,
                "ridge": trial.ridge,
                "epoch": int(row["epoch"]),
                "screen_mae": float(observed_mae(pred, screen).mean()),
                "screen_cells": len(screen.ids),
                "inference_seconds": seconds,
                "trainable_parameters": sum(p.numel() for p in pars.values()),
            }
        )
        del m
    out = args.public / f"fold{fold['fold']}"
    out.mkdir(parents=True, exist_ok=True)
    csv(out / "VALIDATION_TRIALS.csv", rows)
    csv(out / "SCREEN_SCORES.csv", scores)
    csv(out / "WARMUP_TRACE.csv", trace)
    write_json(out / "BASIS_AUDIT.json", audits)
    write_json(out / "SELECTION.json", select)
    write_json(
        out / "COMPLETE.json",
        {
            "status": "COMPLETE_SOURCE_FOLD",
            "fold": fold["fold"],
            "target_future_rows_loaded": 0,
            "outer_labels_loaded": 0,
            "files": {
                n: file_hash(out / n)
                for n in (
                    "VALIDATION_TRIALS.csv",
                    "SCREEN_SCORES.csv",
                    "WARMUP_TRACE.csv",
                    "BASIS_AUDIT.json",
                    "SELECTION.json",
                )
            },
        },
    )
    print(f"fold={fold['fold']} complete", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("register", "run"), default="run")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--public", type=Path, default=PUBLIC)
    p.add_argument("--private", type=Path, default=PRIVATE)
    p.add_argument("--features", type=Path, default=PRIVATE)
    p.add_argument("--data-root", type=Path, default=CACHE)
    p.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    configure()
    cfg, plan, *_ = build_plan(a.config)
    a.public.mkdir(parents=True, exist_ok=True)
    a.private.mkdir(parents=True, exist_ok=True)
    bind(a.public / "REGISTERED_PLAN.json", plan)
    if a.phase == "register":
        return
    for fold in plan["folds"]:
        run_fold(a, cfg, fold, torch.device(a.device))
    final = pd.concat(
        [
            pd.read_csv(a.public / f"fold{x['fold']}" / "SCREEN_SCORES.csv")
            for x in plan["folds"]
        ],
        ignore_index=True,
    )
    csv(a.public / "SCREEN_SCORES.csv", final)
    write_json(
        a.public / "FINAL_DECISION.json",
        {
            "status": "COMPLETE_SOURCE_SCREEN",
            "target_future_rows_loaded": 0,
            "methods": METHODS,
            "files": {"SCREEN_SCORES.csv": file_hash(a.public / "SCREEN_SCORES.csv")},
        },
    )


if __name__ == "__main__":
    main()
