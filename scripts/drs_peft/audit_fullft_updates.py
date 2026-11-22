"""Audit Full-FT update geometry against the locked warmup source state."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE, write_json
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import (
    batches,
    full_prediction,
    load_common_state,
    observed_mae,
    role_bundle,
    training_targets,
)
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE

MILESTONES = (0, 2, 4, 8)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checkpoint_manifest(private):
    """Return hashes for private weights without exposing their tensor payloads."""
    rows = []
    for fold_dir in sorted(Path(private).glob("fold[0-9]")):
        try:
            fold = int(fold_dir.name.removeprefix("fold"))
        except ValueError:
            continue
        for checkpoint in sorted(fold_dir.glob("FULL_FT_EPOCH*.pt")):
            epoch = int(checkpoint.stem.removeprefix("FULL_FT_EPOCH"))
            rows.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "file": checkpoint.name,
                    "bytes": checkpoint.stat().st_size,
                    "sha256": digest(checkpoint),
                }
            )
    return rows


def group(name):
    low = name.lower()
    for tag in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "fc_gate"):
        if tag in low:
            return tag
    if "fusion12" in low:
        return "bridge_fusion12"
    if "fusion30" in low:
        return "bridge_fusion30"
    if "head" in low:
        return "head"
    return "other"


def rank_facts(delta):
    arr = delta.detach().cpu().double()
    fro = float(torch.linalg.vector_norm(arr))
    if arr.ndim != 2:
        return {
            "shape": list(arr.shape),
            "frobenius": fro,
            "rank99": None,
            "stable_rank": None,
            "numerical_rank": None,
            "singular_values": None,
        }
    s = torch.linalg.svdvals(arr)
    energy = torch.cumsum(s.square(), 0)
    total = float(energy[-1]) if len(s) else 0
    rank99 = (
        int(torch.searchsorted(energy, 0.99 * energy[-1]).item() + 1)
        if len(s) and total > 0
        else 0
    )
    tol = float(s.max()) * 1e-5 if len(s) else 0
    return {
        "shape": list(arr.shape),
        "frobenius": fro,
        "rank99": rank99,
        "stable_rank": float(s.square().sum() / s.max().square())
        if len(s) and s.max() > 0
        else 0,
        "numerical_rank": int((s > tol).sum()),
        "singular_values": s[:32].tolist(),
    }


def train_fold(args, cfg, fold, device):
    ids = list(dict.fromkeys(fold["fit_ids"] + fold["tune_ids"] + fold["screen_ids"]))
    bundle = load_bundle(args.features / "PAIRED_PREFIXES.npz", ids)
    target, observed = load_future_rows("MATR", args.data_root, ids)
    labels = Labels(target, observed, ids)
    model = SharedBatteryModel(args.backbone_dir).to(device)
    fit = role_bundle(bundle, fold["fit_ids"], labels, device)
    tune = role_bundle(bundle, fold["tune_ids"], labels, device)
    common_path = args.private / f"COMMON_FOLD{fold['fold']}.pt"
    if not common_path.is_file():
        common_path = args.private.parent / f"COMMON_FOLD{fold['fold']}.pt"
    common = torch.load(common_path, map_location="cpu", weights_only=True)
    load_common_state(model, common)
    # The locked warmup state is frozen. Full FT must explicitly reopen every
    # parameter, including the official backbone, before creating the optimizer.
    model.requires_grad_(True)
    warmup_state = {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }
    snapshots = {
        0: {
            name: value.detach().cpu().clone()
            for name, value in model.named_parameters()
        }
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    trace = []
    for epoch in range(1, 9):
        model.train()
        total = 0.0
        steps = 0
        start = time.perf_counter()
        for index in batches(
            len(fit.ids),
            cfg["train_batch_size"],
            seed=cfg["seed"] + 1009 * epoch + fold["fold"],
        ):
            future, y, mask = training_targets(fit, index, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(*fit.arrays(index), future, channels=12)
            err = torch.where(mask, pred - y, 0.0)
            loss = (err.abs().sum(1) / mask.sum(1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(index)
            steps += 1
        model.eval()
        tune_pred, _ = full_prediction(model, tune, 12, cfg["train_batch_size"], device)
        row = {
            "fold": fold["fold"],
            "epoch": epoch,
            "training_mae": total / len(fit.ids),
            "tune_mae": float(observed_mae(tune_pred, tune).mean()),
            "optimizer_updates": steps,
            "seconds": time.perf_counter() - start,
        }
        trace.append(row)
        if epoch in MILESTONES:
            snapshots[epoch] = {
                name: value.detach().cpu().clone()
                for name, value in model.named_parameters()
            }
    out = args.private / f"fold{fold['fold']}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch, state in snapshots.items():
        facts = []
        for name, value in state.items():
            if name not in warmup_state:
                continue
            d = value - warmup_state[name]
            fact = rank_facts(d) | {
                "fold": fold["fold"],
                "epoch": epoch,
                "parameter": name,
                "group": group(name),
                "relative_to_warmup": float(
                    torch.linalg.vector_norm(d)
                    / (torch.linalg.vector_norm(warmup_state[name]) + 1e-12)
                ),
            }
            facts.append(fact)
        write_json(
            out / f"FULL_FT_EPOCH{epoch}.json",
            {"fold": fold["fold"], "epoch": epoch, "parameter_facts": facts},
        )
        rows.extend(facts)
        torch.save({"epoch": epoch, "state": state}, out / f"FULL_FT_EPOCH{epoch}.pt")
    pd.DataFrame(rows).to_csv(
        out / "UPDATE_GEOMETRY.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(trace).to_csv(
        out / "FULL_FT_TRACE.csv", index=False, lineterminator="\n"
    )
    write_json(
        out / "COMPLETE.json",
        {
            "status": "COMPLETE_FULL_FT_UPDATE_AUDIT",
            "fold": fold["fold"],
            "target_future_labels_read": 0,
            "outer_labels_read": 0,
            "milestones": list(MILESTONES),
            "files": {
                p.name: digest(p)
                for p in [out / "UPDATE_GEOMETRY.csv", out / "FULL_FT_TRACE.csv"]
            },
        },
    )
    print("fold", fold["fold"], "complete", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--public", type=Path, default=PUBLIC / "fullft_update_audit")
    # Full-FT snapshots live at the study root alongside COMMON_FOLD*.pt.
    # Keeping the default aligned with the registered warmup location prevents
    # accidentally mixing an older nested diagnostic run with the public audit.
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
    for fold in plan["folds"]:
        train_fold(a, cfg, fold, torch.device(a.device))
    frames = [
        pd.read_csv(a.private / f"fold{fold['fold']}" / "UPDATE_GEOMETRY.csv")
        for fold in plan["folds"]
    ]
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(a.public / "UPDATE_GEOMETRY.csv", index=False, lineterminator="\n")
    summary = (
        all_rows[all_rows.epoch.eq(8)]
        .groupby("group", as_index=False)
        .agg(
            frobenius=("frobenius", "mean"),
            rank99=("rank99", "mean"),
            stable_rank=("stable_rank", "mean"),
            parameters=("parameter", "nunique"),
        )
    )
    summary.to_csv(
        a.public / "UPDATE_GROUP_SUMMARY.csv", index=False, lineterminator="\n"
    )
    checkpoint_hashes = checkpoint_manifest(a.private)
    if len(checkpoint_hashes) != len(plan["folds"]) * len(MILESTONES):
        raise RuntimeError("Every source fold must provide all Full-FT milestone checkpoints")
    pd.DataFrame(checkpoint_hashes).to_csv(
        a.public / "CHECKPOINT_HASHES.csv", index=False, lineterminator="\n"
    )
    epoch0 = all_rows[all_rows.epoch.eq(0)]
    trainable_inventory = (
        epoch0.groupby("group", as_index=False)
        .agg(
            parameters=("parameter", "nunique"),
            folds=("fold", "nunique"),
            nonzero_initial_update=("frobenius", lambda values: int((values > 0).sum())),
        )
    )
    trainable_inventory.to_csv(
        a.public / "TRAINABLE_GROUP_INVENTORY.csv",
        index=False,
        lineterminator="\n",
    )
    write_json(
        a.public / "COMPLETE.json",
        {
            "status": "COMPLETE_FULL_FT_UPDATE_AUDIT",
            "folds": [0, 1, 2],
            "target_future_labels_read": 0,
            "outer_labels_read": 0,
            "milestones": list(MILESTONES),
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 1e-4,
                "weight_decay": 0,
                "epochs": 8,
                "gradient_clip_norm": 1.0,
                "trainable_parameters": 13999369,
            },
            "source_roles": {
                "domain": "MATR",
                "folds": [fold["fold"] for fold in plan["folds"]],
                "target_future_labels_read": 0,
                "outer_labels_read": 0,
            },
            "files": {
                "UPDATE_GEOMETRY.csv": digest(a.public / "UPDATE_GEOMETRY.csv"),
                "UPDATE_GROUP_SUMMARY.csv": digest(
                    a.public / "UPDATE_GROUP_SUMMARY.csv"
                ),
                "CHECKPOINT_HASHES.csv": digest(a.public / "CHECKPOINT_HASHES.csv"),
                "TRAINABLE_GROUP_INVENTORY.csv": digest(
                    a.public / "TRAINABLE_GROUP_INVENTORY.csv"
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
