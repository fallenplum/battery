"""Source-only dynamic output calibration from the locked warmup state.

The condition vector is computed only from the current battery's 100-cycle,
12-channel prefix.  The adapter changes the final CycleQueryHead function
through a rank-4, anchor-preserving correction.  Backbone, Bridge, and the
original Head weights stay frozen for the dynamic arm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.data import CACHE
from scripts.drs_peft.run_head_bridge_aware_peft import apply, install, train
from scripts.drs_peft.run_self_reference import Labels, load_bundle
from scripts.drs_peft.self_reference_protocol import CONFIG as BASE_CONFIG, PRIVATE, PUBLIC, build_plan
from scripts.drs_peft.self_reference_training import (
    GRID,
    batches,
    full_prediction,
    load_common_state,
    masked_cell_loss,
    observed_mae,
    role_bundle,
    training_targets,
)
from scripts.drs_peft.train import configure
from scripts.drs_peft.transfer_scope_data import load_future_rows
from scripts.drs_peft.transfer_scope_reference import BACKBONE


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/drs_peft/dynamic_head_v1/CONFIG.json"
PUBLIC_OUT = ROOT / "results/drs_peft/dynamic_head_v1"
PRIVATE_OUT = ROOT / "runtime/drs_peft/dynamic_head_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix_condition(features: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
    """Return z from the observed prefix only, with no future-label access."""
    values = torch.where(feature_mask, features, 0.0)
    count = feature_mask.sum(dim=1).clamp_min(1).to(values.dtype)
    mean = values.sum(dim=1) / count
    valid_time = feature_mask.any(dim=-1)
    last_index = valid_time.to(torch.int64).sum(dim=1).clamp_min(1) - 1
    rows = torch.arange(values.shape[0], device=values.device)
    last = values[rows, last_index]
    return torch.cat((mean, last), dim=-1)


class DynamicCycleHead(nn.Module):
    """Dynamic rank-r final-head correction that preserves the anchor."""

    def __init__(self, base: nn.Module, z_dim: int = 24, rank: int = 4, seed: int = 42, condition_mode: str = "dynamic"):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        if condition_mode not in ("dynamic", "constant"):
            raise ValueError("condition_mode must be dynamic or constant")
        self.condition_mode = condition_mode
        torch.manual_seed(seed)
        self.mapper = nn.Linear(z_dim, rank)
        self.basis = nn.Parameter(torch.empty(rank, 32))
        nn.init.normal_(self.basis, mean=0.0, std=1e-3)

    def forward(self, state, anchor, query, z):
        if query.ndim == 1:
            query = query[None].expand(len(state), -1)
        u = (query - 100) / 1000
        # The constant arm uses a fixed all-ones probe. This keeps the mapper
        # nominally at the same parameter count and lets every mapper weight
        # receive a gradient while the resulting coefficient stays battery
        # independent.
        condition = z if self.condition_mode == "dynamic" else torch.ones_like(z)
        alpha = torch.tanh(self.mapper(condition))
        delta_weight = alpha @ self.basis
        context = state[:, None].expand(-1, query.shape[1], -1)
        value_input = torch.cat((context, u[..., None]), -1)
        hidden = self.base.net[:4](value_input)
        value = self.base.net[4](hidden).squeeze(-1)
        ref_input = torch.cat((state, torch.zeros_like(anchor[:, None])), -1)
        ref_hidden = self.base.net[:4](ref_input)
        reference = self.base.net[4](ref_hidden).squeeze(-1)
        delta = (hidden * delta_weight[:, None, :]).sum(dim=-1)
        ref_delta = (ref_hidden * delta_weight).sum(dim=-1)
        return anchor[:, None] + 0.2 * ((value + delta) - (reference + ref_delta)[:, None])


class DynamicHeadModel(SharedBatteryModel):
    def __init__(self, common: SharedBatteryModel, rank: int, seed: int, condition_mode: str):
        super().__init__(BACKBONE)
        self.load_state_dict(common.state_dict())
        self.head = DynamicCycleHead(self.head, rank=rank, seed=seed, condition_mode=condition_mode)

    def forward(self, features, feature_mask, cycles, anchor, future_cycles, channels=30):
        context = self.context(features, feature_mask, cycles, anchor, channels)
        state = (self.fusion12 if context.shape[-1] == 829 else self.fusion30)(context)
        z = prefix_condition(features[..., :12], feature_mask[..., :12])
        return self.head(state, anchor, future_cycles, z)


def dynamic_train(model, params, method, lr, fit, tune, cfg, fold, device):
    values = list(params.values())
    optimizer = torch.optim.AdamW(values, lr=lr, weight_decay=0.0)
    rows = []
    best = None
    best_state = None
    start_total = time.perf_counter()
    for epoch in range(1, max(cfg["epochs"]) + 1):
        model.train()
        total = 0.0
        started = time.perf_counter()
        for index in batches(
            len(fit.ids),
            cfg["train_batch_size"],
            seed=cfg["seed"] + 1009 * epoch + fold,
        ):
            cycles, target, mask = training_targets(fit, index, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*fit.arrays(index), cycles, channels=12)
            loss = masked_cell_loss(prediction, target, mask, kind="mae")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(values, cfg["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        if epoch in cfg["epochs"]:
            model.eval()
            prediction, inference_seconds = full_prediction(
                model, tune, 12, cfg["train_batch_size"], device
            )
            tune_mae = float(observed_mae(prediction, tune).mean())
            row = {
                "fold": fold,
                "method": method,
                "lr": lr,
                "epoch": epoch,
                "training_mae": total / len(fit.ids),
                "tune_mae": tune_mae,
                "training_seconds": time.perf_counter() - started,
                "tune_inference_seconds": inference_seconds,
                "trainable_parameters": sum(value.numel() for value in values),
            }
            rows.append(row)
            if best is None or tune_mae < best["tune_mae"] or (
                tune_mae == best["tune_mae"] and lr < best["lr"]
            ):
                best = row
                best_state = {name: value.detach().cpu().clone() for name, value in params.items()}
    best = best | {"total_training_seconds": time.perf_counter() - start_total}
    return best, best_state, rows


def score(prediction, role):
    diff = np.where(role.observed, prediction - role.targets, 0.0)
    n = role.observed.sum(axis=1)
    mae = np.abs(diff).sum(axis=1) / n
    mse = (diff**2).sum(axis=1) / n
    return float(mae.mean()), float(np.sqrt(mse).mean()), float(mse.mean()), mae, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--public", type=Path, default=PUBLIC_OUT)
    parser.add_argument("--features", type=Path, default=PRIVATE)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    base_cfg, plan, *_ = build_plan(BASE_CONFIG)
    if cfg["seed"] != base_cfg["seed"]:
        raise RuntimeError("The minimal dynamic experiment is registered for seed42")
    args.public.mkdir(parents=True, exist_ok=True)
    args.private.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    scores, traces, cells = [], [], []
    for fold in plan["folds"]:
        ids = list(dict.fromkeys(fold["fit_ids"] + fold["tune_ids"] + fold["screen_ids"]))
        bundle = load_bundle(args.features / "PAIRED_PREFIXES.npz", ids)
        targets, observed = load_future_rows("MATR", args.data_root, ids)
        labels = Labels(targets, observed, ids)
        fit = role_bundle(bundle, fold["fit_ids"], labels, device)
        tune = role_bundle(bundle, fold["tune_ids"], labels, device)
        screen = role_bundle(bundle, fold["screen_ids"], labels, device)
        common_path = args.private / f"COMMON_FOLD{fold['fold']}.pt"
        common = torch.load(common_path, map_location="cpu", weights_only=True)
        common_model = SharedBatteryModel(args.backbone_dir).to(device)
        load_common_state(common_model, common)
        # Static Head-only comparator uses the same warmup checkpoint and source roles.
        static_selected = None
        static_count = None
        for lr in cfg["adapter_rates"]:
            model = copy.deepcopy(common_model)
            params, groups, count, _ = install(model, "HEAD_ONLY_TRAIN", cfg["seed"])
            best, state, trace = train(model, params, "HEAD_ONLY_TRAIN", lr, fit, tune, base_cfg, fold["fold"], device)
            traces.extend(trace)
            static_count = count
            if static_selected is None or best["tune_mae"] < static_selected["best"]["tune_mae"] or (
                best["tune_mae"] == static_selected["best"]["tune_mae"] and lr < static_selected["best"]["lr"]
            ):
                static_selected = {"best": best, "state": state}
            del model
        model = copy.deepcopy(common_model)
        params, groups, count, _ = install(model, "HEAD_ONLY_TRAIN", cfg["seed"])
        apply(params, static_selected["state"])
        prediction, _ = full_prediction(model, screen, 12, cfg["train_batch_size"], device)
        mae, rmse, mse, cell_mae, cell_mse = score(prediction, screen)
        scores.append({"fold": fold["fold"], "method": "HEAD_ONLY_TRAIN", "mae": mae, "rmse": rmse, "mse": mse, "parameters": count, "best_lr": static_selected["best"]["lr"], "best_epoch": static_selected["best"]["epoch"], "target_future_labels_read": 0, "outer_labels_read": 0})
        for uid, value, error in zip(screen.ids, cell_mae, cell_mse):
            cells.append({"fold": fold["fold"], "method": "HEAD_ONLY_TRAIN", "cell_uid": uid, "mae": float(value), "mse": float(error)})
        del model
        for method, mode in (("CONST_HEAD_R4", "constant"), ("DYNAMIC_HEAD_R4", "dynamic")):
            selected = None
            for lr in cfg["adapter_rates"]:
                model = DynamicHeadModel(common_model, cfg["dynamic_rank"], cfg["seed"], mode).to(device)
                model.requires_grad_(False)
                model.head.mapper.requires_grad_(True)
                model.head.basis.requires_grad_(True)
                params = {name: value for name, value in model.named_parameters() if value.requires_grad}
                best, state, trace = dynamic_train(model, params, method, lr, fit, tune, cfg, fold["fold"], device)
                if selected is None or best["tune_mae"] < selected["best"]["tune_mae"] or (
                    best["tune_mae"] == selected["best"]["tune_mae"] and lr < selected["best"]["lr"]
                ):
                    selected = {"best": best, "state": state}
                traces.extend(trace)
                del model
            model = DynamicHeadModel(common_model, cfg["dynamic_rank"], cfg["seed"], mode).to(device)
            model.requires_grad_(False)
            model.head.mapper.requires_grad_(True)
            model.head.basis.requires_grad_(True)
            params = {name: value for name, value in model.named_parameters() if value.requires_grad}
            with torch.no_grad():
                for name, value in selected["state"].items():
                    params[name].copy_(value)
            prediction, _ = full_prediction(model, screen, 12, cfg["train_batch_size"], device)
            mae, rmse, mse, cell_mae, cell_mse = score(prediction, screen)
            scores.append({"fold": fold["fold"], "method": method, "mae": mae, "rmse": rmse, "mse": mse, "parameters": sum(value.numel() for value in params.values()), "best_lr": selected["best"]["lr"], "best_epoch": selected["best"]["epoch"], "target_future_labels_read": 0, "outer_labels_read": 0})
            for uid, value, error in zip(screen.ids, cell_mae, cell_mse):
                cells.append({"fold": fold["fold"], "method": method, "cell_uid": uid, "mae": float(value), "mse": float(error)})
            del model
        del common_model
    pd.DataFrame(scores).to_csv(args.public / "DYNAMIC_HEAD_SCORES.csv", index=False, lineterminator="\n")
    pd.DataFrame(traces).to_csv(args.public / "DYNAMIC_HEAD_TRACE.csv", index=False, lineterminator="\n")
    pd.DataFrame(cells).to_csv(args.public / "DYNAMIC_HEAD_CELL.csv", index=False, lineterminator="\n")
    summary = pd.DataFrame(scores).groupby("method", as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"), mse=("mse", "mean"), folds=("fold", "nunique"), parameters=("parameters", "first"), best_epoch=("best_epoch", "median"), best_lr=("best_lr", "first"))
    summary.to_csv(args.public / "DYNAMIC_HEAD_SUMMARY.csv", index=False, lineterminator="\n")
    receipt = {"status": "COMPLETE_SOURCE_DYNAMIC_HEAD_CALIBRATION", "seed": cfg["seed"], "methods": cfg["methods"], "dynamic_rank": cfg["dynamic_rank"], "dynamic_z_dim": 24, "target_future_labels_read": 0, "outer_labels_read": 0, "files": {name: digest(args.public / name) for name in ("DYNAMIC_HEAD_SCORES.csv", "DYNAMIC_HEAD_TRACE.csv", "DYNAMIC_HEAD_CELL.csv", "DYNAMIC_HEAD_SUMMARY.csv")}}
    (args.public / "DYNAMIC_HEAD_COMPLETE.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
