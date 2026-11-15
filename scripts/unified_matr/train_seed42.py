"""Run the registered seven-arm MATR trajectory matrix using seed 42."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from scripts.run_experiment_b import execution_lock

ARMS = ("M0", "M1", "M2", "M3", "M3-NL", "M3-NP", "M3-NR")
INPUT_KEYS = (
    "raw_viq", "point_mask", "cycle_ids", "reference_viq", "policy", "policy_mask",
    "global_features", "global_mask", "local_residual", "local_mask",
    "local_features", "local_feature_mask", "anchor_soh100",
)
FEATURE_MASKS = {
    "raw_viq": "point_mask", "reference_viq": "point_mask", "policy": "policy_mask",
    "global_features": "global_mask", "local_residual": "local_mask",
    "local_features": "local_feature_mask",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def capture_random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA training checkpoint requires the original CUDA RNG devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def validate_config(config):
    if config["training_seeds"] != [42] or config["split"]["seed"] != 42 or tuple(config["models"]) != ARMS:
        raise RuntimeError("The registered matrix requires exactly seven arms and seed42")
    expected = {"epochs": 150, "batch_size": 8, "optimizer": "AdamW", "learning_rate": 0.0003,
                "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
                "gradient_clip_norm": 1.0, "early_stopping": False,
                "checkpoint_selection": "minimum validation cell-mean trajectory MAE",
                "auxiliary_rate_windows": [[200, 500], [500, 1000]],
                "auxiliary_rate_scale": 1000, "auxiliary_rate_loss_weight": 0.1,
                "persistence_soft_penalty_weight": 0.001,
                "observation_soh_bounds": [0.0, 1.2], "observation_residual_amplitude": 0.03,
                "num_workers": 0, "max_concurrent_gpu_runs": 1}
    for key, value in expected.items():
        if config["training"].get(key) != value:
            raise RuntimeError("Training setting differs from the registered seed42 budget at " + key)
    split_settings = {"outer_cell_folds": 5, "validation_fraction_of_outer_train": 0.2,
                      "protocol_sensitivity_test_fraction": 0.2}
    if any(config["split"].get(key) != value for key, value in split_settings.items()):
        raise RuntimeError("Split settings differ from the registered five-fold and protocol-heldout matrix")


def load_frozen_data(root):
    freeze = json.loads((root / "DATA_FREEZE.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "DATA_FREEZE_PASS" or freeze.get("training_seeds") != [42]:
        raise RuntimeError("Formal training requires a completed original-axis data freeze")
    required = {"dataset_manifest.json", "cells.parquet", "early_viq_qv_64.npz", "early_physics_features.parquet",
                "future_soh.parquet", "split_contract.json", "dataset_qa.json", "dataset_fingerprint.sha256"}
    required = {(root / name).resolve() for name in required}
    covered = {(root / item["path"]).resolve() for item in freeze["files"]}
    if not required.issubset(covered) or len(freeze.get("source_file_sha256", [])) != 4:
        raise RuntimeError("Data freeze omits required artifacts, original instrument mapping or four source archives")
    for item in freeze["files"]:
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError("Frozen data fingerprint mismatch at " + item["path"])
    qa = json.loads((root / "dataset_qa.json").read_text(encoding="utf-8"))
    axis = next((item for item in qa.get("checks", []) if item["id"] == "instrument_original_axis_certified"), {})
    if qa.get("status") != "PASS" or qa.get("formal_training_authorized_by_data_gate") is not True or axis.get("status") != "PASS":
        raise RuntimeError("Independent QA has not authenticated the original instrument cycle axis")
    mapping_sha = axis.get("detail", {}).get("mapping_sha256")
    if not mapping_sha or mapping_sha not in {item["sha256"] for item in freeze["files"]}:
        raise RuntimeError("The authenticated original-cycle mapping is absent from the data freeze")
    with np.load(root / "early_viq_qv_64.npz", allow_pickle=False) as archive:
        identifiers = archive["cell_uid"].astype(str).tolist()
        inputs = {key: archive[key].copy() for key in INPUT_KEYS}
        eligible = archive["input_eligible"].astype(bool)
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Repeated physical cell UID")
    for key in set(FEATURE_MASKS.values()):
        if inputs[key].dtype != np.bool_:
            raise RuntimeError("Frozen input mask must have boolean dtype at " + key)
    cells = pd.read_parquet(root / "cells.parquet").set_index("cell_uid")
    future = pd.read_parquet(root / "future_soh.parquet")
    future = future.loc[future.capacity_valid & (future.original_cycle_id > 100)].copy()
    if future.empty or not np.isfinite(future.soh_nominal).all():
        raise RuntimeError("Future supervision must contain finite observed targets")
    max_cycle = max(1000, int(future.original_cycle_id.max()))
    query = np.arange(101, max_cycle + 1, dtype=np.float32)
    targets = np.zeros((len(identifiers), len(query)), dtype=np.float32)
    observed = np.zeros_like(targets, dtype=bool)
    rate_targets = np.zeros((len(identifiers), 2), dtype=np.float32)
    rate_mask = np.zeros_like(rate_targets, dtype=bool)
    index = {uid: number for number, uid in enumerate(identifiers)}
    for row in future.itertuples(index=False):
        number = index[row.cell_uid]
        position = int(row.original_cycle_id) - 101
        if observed[number, position]:
            raise RuntimeError("Repeated original cycle in a physical cell")
        targets[number, position] = row.soh_nominal
        observed[number, position] = True
    for number, uid in enumerate(identifiers):
        for window, name in enumerate(("r200_500", "r500_1000")):
            value = cells.loc[uid, name]
            if pd.notna(value):
                if not np.isfinite(value):
                    raise RuntimeError("A supported rate target is nonfinite")
                rate_targets[number, window] = float(value)
                rate_mask[number, window] = True
    split = json.loads((root / "split_contract.json").read_text(encoding="utf-8"))
    return inputs, identifiers, eligible, cells, query, targets, observed, rate_targets, rate_mask, split, freeze


def fold_scaler(inputs, training):
    scaled = {key: value.copy() for key, value in inputs.items()}
    statistics = {}
    for key, mask_key in FEATURE_MASKS.items():
        values = np.asarray(inputs[key], dtype=np.float64)
        mask = np.asarray(inputs[mask_key], dtype=bool)
        while mask.ndim < values.ndim:
            mask = mask[..., None]
        mask = np.broadcast_to(mask, values.shape)
        if not np.isfinite(values[mask]).all():
            raise RuntimeError("Nonfinite supported frozen features at " + key)
        train_values, train_mask = values[training], mask[training]
        axes = tuple(range(values.ndim - 1))
        # Residual coordinates stay distinct. Every statistic is train-cell-only.
        if key == "local_residual":
            axes = (0, 1)
        count = train_mask.sum(axis=axes)
        mean = np.divide(np.where(train_mask, train_values, 0).sum(axis=axes), count,
                         out=np.zeros_like(count, dtype=float), where=count > 0)
        variance = np.divide(np.where(train_mask, (train_values - mean) ** 2, 0).sum(axis=axes), count,
                             out=np.zeros_like(mean), where=count > 0)
        standard_deviation = np.sqrt(variance)
        standard_deviation = np.where(standard_deviation >= 1e-6, standard_deviation, 1.0)
        scaled[key] = np.where(mask, (values - mean) / standard_deviation, 0).astype(np.float32)
        statistics[key] = {"mean": mean.tolist(), "std": standard_deviation.tolist(), "count": count.tolist()}
    return scaled, statistics


def masked_cell_mae(prediction, target, mask):
    count = mask.sum(dim=1)
    error = torch.where(mask, prediction - target, 0.0)
    per_cell = error.abs().sum(dim=1) / count.clamp_min(1)
    valid = count > 0
    return per_cell[valid].sum() / valid.sum().clamp_min(1)


def initialization_registry(inputs):
    from models.unified_matr import UnifiedMATRModel, model_registry

    models = {arm: UnifiedMATRModel(arm, policy_dim=inputs["policy"].shape[-1],
                                  global_dim=inputs["global_features"].shape[-1],
                                  local_feature_dim=inputs["local_features"].shape[-1], seed=42)
              for arm in ARMS}
    states = {arm: model.state_dict() for arm, model in models.items()}
    checks = []
    for number, left in enumerate(ARMS):
        for right in ARMS[number + 1:]:
            shared = [key for key in states[left].keys() & states[right].keys()
                      if states[left][key].shape == states[right][key].shape]
            if not all(torch.equal(states[left][key], states[right][key]) for key in shared):
                raise RuntimeError("Shared model initialization differs for " + left + " / " + right)
            checks.append({"arms": [left, right], "common_tensors": len(shared), "exact_equal": True})
    entries = {arm: model_registry(model) for arm, model in models.items()}
    full_counts = {entries[arm]["parameters"] for arm in ("M1", "M2", "M3", "M3-NP", "M3-NR")}
    if len(full_counts) != 1 or abs(entries["M3-NL"]["parameters"] - entries["M3"]["parameters"]) > 80:
        raise RuntimeError("Actual active parameter budgets differ from the registered model comparison")
    return {"models": entries, "common_initialization_checks": checks,
            "NL_parameter_difference_from_M3": entries["M3-NL"]["parameters"] - entries["M3"]["parameters"]}


def write_history(path, history):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(value, pending)
    os.replace(pending, path)


def same_selection(actual, expected):
    if {key: value for key, value in actual.items() if key != "model"} != {key: value for key, value in expected.items() if key != "model"}:
        return False
    return actual["model"].keys() == expected["model"].keys() and all(
        torch.equal(actual["model"][key].cpu(), value.cpu()) for key, value in expected["model"].items())


def to_batch(inputs, rows, device):
    return {key: torch.as_tensor(value[rows], device=device) for key, value in inputs.items()}


def predictions(model, inputs, rows, query, device, batch_size):
    collected = []
    trend = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = to_batch(inputs, rows[start:start + batch_size], device)
            output = model(**batch, query_cycles=torch.as_tensor(query, device=device))
            collected.append(output["trajectory"].cpu().numpy())
            trend.append(output["trend"].cpu().numpy())
    return np.concatenate(collected), np.concatenate(trend)


def trajectory_metrics(prediction, target, mask):
    count = mask.sum(axis=1)
    error = np.where(mask, prediction - target, 0.0)
    mae = np.divide((np.abs(error) * mask).sum(axis=1), count,
                    out=np.full(len(count), np.nan), where=count > 0)
    rmse = np.sqrt(np.divide((error ** 2 * mask).sum(axis=1), count,
                            out=np.full(len(count), np.nan), where=count > 0))
    return {"cell_mean_mae": float(np.nanmean(mae)), "cell_mean_rmse": float(np.nanmean(rmse)),
            "supported_cells": int((count > 0).sum()), "observations": int(count.sum())}, mae, rmse


def train_unit(arm, split_name, roles, data, config, data_root, run_root, device, registration):
    from models.unified_matr import UnifiedMATRModel
    from models.unified_matr.model import state_hash

    inputs, identifiers, eligible, cells, query, targets, observed, rate_targets, rate_mask, _, _ = data
    identifier_index = {uid: number for number, uid in enumerate(identifiers)}
    indices = {role: np.array([identifier_index[uid] for uid in roles[role]
                              if eligible[identifier_index[uid]]], dtype=int)
               for role in ("train", "validation", "test")}
    for role, rows in indices.items():
        if not len(rows) or not observed[rows].sum():
            raise RuntimeError("No observed supervision in split role " + role)
    unit_root = run_root / split_name / arm
    unit_root.mkdir(parents=True, exist_ok=True)
    unit_path = unit_root / "UNIT_CONTRACT.json"
    unit_contract = {"arm": arm, "seed": 42, "split_name": split_name,
                     "registration_sha256": sha256(run_root / "RUN_REGISTRATION.json"),
                     "roles": {role: [identifiers[i] for i in rows] for role, rows in indices.items()},
                     "training": config["training"],
                     "expected_updates": config["training"]["epochs"] * math.ceil(len(indices["train"]) / config["training"]["batch_size"]),
                     "initial_model": registration["initialization"]["models"][arm], "device": str(device)}
    if unit_path.exists() and json.loads(unit_path.read_text(encoding="utf-8")) != unit_contract:
        raise RuntimeError("Existing unit has a different frozen contract")
    write_json(unit_path, unit_contract)
    complete_path = unit_root / "COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "EVALUATED" or complete.get("successful_updates") != unit_contract["expected_updates"]:
            raise RuntimeError("Completed unit has an inconsistent evaluation state or update budget")
        for item in complete["files"]:
            if sha256(unit_root / item["path"]) != item["sha256"]:
                raise RuntimeError("Completed unit artifact mismatch")
        print(json.dumps({"status": "VERIFIED_COMPLETE", "split": split_name, "arm": arm}), flush=True)
        return
    scaled, scaler = fold_scaler(inputs, indices["train"])
    write_json(unit_root / "TRAIN_SCALER.json", {"training_ids": unit_contract["roles"]["train"], "statistics": scaler})
    seed_everything()
    model = UnifiedMATRModel(arm, policy_dim=inputs["policy"].shape[-1],
                             global_dim=inputs["global_features"].shape[-1],
                             local_feature_dim=inputs["local_features"].shape[-1], seed=42,
                             observation_bound=config["training"]["observation_residual_amplitude"],
                             soh_bounds=tuple(config["training"]["observation_soh_bounds"])).to(device)
    if state_hash(model) != registration["initialization"]["models"][arm]["initial_state_sha256"]:
        raise RuntimeError("Actual training model differs from the frozen common initialization")
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]),
                                 eps=settings["eps"])
    checkpoint = unit_root / "TRAIN_STATE.pt"
    best_path = unit_root / "SELECTED.pt"
    epoch_start, best_mae, successful_updates = 0, math.inf, 0
    history, selected_checkpoint = [], None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["contract"] != unit_contract:
            raise RuntimeError("Resume checkpoint contract mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch_start, best_mae, successful_updates = state["next_epoch"], state["best_mae"], state["successful_updates"]
        history, selected_checkpoint = state["history"], state["selected_checkpoint"]
        expected_updates = epoch_start * math.ceil(len(indices["train"]) / settings["batch_size"])
        if len(history) != epoch_start or successful_updates != expected_updates:
            raise RuntimeError("Resume checkpoint has an incomplete epoch or optimizer-update history")
        restore_random_state(state["random_state"])
        actual = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None
        if actual is None or not same_selection(actual, selected_checkpoint):
            if (unit_root / "SELECTION_FROZEN.json").exists():
                raise RuntimeError("A frozen selected checkpoint differs from the atomic training state")
            save_checkpoint(best_path, selected_checkpoint)
        write_history(unit_root / "training.jsonl", history)
    tensor_query = torch.as_tensor(query, device=device)
    started = time.time() - (history[-1]["elapsed_seconds"] if history else 0)
    for epoch in range(epoch_start, settings["epochs"]):
        model.train()
        epoch_rng = np.random.default_rng(42 + epoch)
        order = epoch_rng.permutation(indices["train"])
        losses = []
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            batch = to_batch(scaled, rows, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch, query_cycles=tensor_query)
            mask = torch.as_tensor(observed[rows], device=device)
            target = torch.as_tensor(targets[rows], device=device)
            loss = masked_cell_mae(output["trajectory"], target, mask)
            rate_prediction = torch.stack((
                (output["trajectory"][:, 200 - 101] - output["trajectory"][:, 500 - 101]) / 300,
                (output["trajectory"][:, 500 - 101] - output["trajectory"][:, 1000 - 101]) / 500,
            ), dim=1)
            auxiliary_mask = torch.as_tensor(rate_mask[rows], device=device)
            auxiliary_count = auxiliary_mask.sum(dim=1)
            if auxiliary_mask.any():
                auxiliary_error = torch.abs(rate_prediction - torch.as_tensor(rate_targets[rows], device=device))
                auxiliary_cell = (auxiliary_error * auxiliary_mask).sum(dim=1) / auxiliary_count.clamp_min(1)
                auxiliary_loss = auxiliary_cell[auxiliary_count > 0].mean() * settings["auxiliary_rate_scale"]
                loss = loss + settings["auxiliary_rate_loss_weight"] * auxiliary_loss
            loss = loss + settings["persistence_soft_penalty_weight"] * output["persistence_penalty"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            successful_updates += 1
            losses.append(float(loss.detach()))
        val_prediction, _ = predictions(model, scaled, indices["validation"], query, device, settings["batch_size"])
        val_metrics, _, _ = trajectory_metrics(val_prediction, targets[indices["validation"]], observed[indices["validation"]])
        if not np.isfinite(val_metrics["cell_mean_mae"]):
            raise FloatingPointError("Validation selection criterion is nonfinite")
        selected = val_metrics["cell_mean_mae"] < best_mae
        if selected:
            best_mae = val_metrics["cell_mean_mae"]
            selected_checkpoint = {"model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                                   "epoch": epoch + 1, "validation_mae": best_mae, "contract": unit_contract}
            save_checkpoint(best_path, selected_checkpoint)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                  "validation_mae": val_metrics["cell_mean_mae"], "selected": selected,
                  "successful_updates": successful_updates, "elapsed_seconds": time.time() - started}
        history.append(record)
        save_checkpoint(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "next_epoch": epoch + 1, "best_mae": best_mae,
                                    "successful_updates": successful_updates, "contract": unit_contract,
                                    "random_state": capture_random_state(), "history": history,
                                    "selected_checkpoint": selected_checkpoint})
        write_history(unit_root / "training.jsonl", history)
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"split": split_name, "arm": arm, **record}), flush=True)
    if successful_updates != unit_contract["expected_updates"]:
        raise RuntimeError("Final optimizer updates differ from the fixed epoch/batch budget")
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    # Freeze the chosen weights and validation criterion before any test prediction.
    selection = {"selected_epoch": selected["epoch"],
               "validation_mae": selected["validation_mae"], "checkpoint_sha256": sha256(best_path),
               "successful_updates": successful_updates, "seed": 42,
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
    selection_path = unit_root / "SELECTION_FROZEN.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise RuntimeError("Test-exposed checkpoint selection is immutable")
    write_json(selection_path, selection)
    test = indices["test"]
    test_prediction, test_trend = predictions(model, scaled, test, query, device, settings["batch_size"])
    metrics, per_cell_mae, per_cell_rmse = trajectory_metrics(test_prediction, targets[test], observed[test])
    rows = []
    for position, number in enumerate(test):
        uid = identifiers[number]
        valid = observed[number]
        for slot in np.flatnonzero(valid):
            rows.append({"cell_uid": uid, "original_cycle_id": int(query[slot]), "target": float(targets[number, slot]),
                         "prediction": float(test_prediction[position, slot]), "trend": float(test_trend[position, slot])})
    pd.DataFrame(rows).to_parquet(unit_root / "test_predictions.parquet", index=False)
    per_cell = [{"cell_uid": identifiers[number], "mae": float(per_cell_mae[position]),
                 "rmse": float(per_cell_rmse[position]), "observations": int(observed[number].sum()),
                 "protocol_uid": str(cells.loc[identifiers[number], "protocol_uid"])}
                for position, number in enumerate(test) if observed[number].any()]
    pd.DataFrame(per_cell).to_csv(unit_root / "test_cell_metrics.csv", index=False)
    rate_rows = []
    query_index = {int(cycle): position for position, cycle in enumerate(query)}
    for position, number in enumerate(test):
        for window, (left, right) in enumerate(((200, 500), (500, 1000))):
            first = float(test_prediction[position, query_index[left]])
            last = float(test_prediction[position, query_index[right]])
            rate_rows.append({"cell_uid": identifiers[number], "protocol_uid": str(cells.loc[identifiers[number], "protocol_uid"]),
                              "target_name": f"r{left}_{right}", "start_cycle": left, "end_cycle": right,
                              "predicted_start_soh": first, "predicted_end_soh": last,
                              "predicted_rate": (first - last) / (right - left),
                              "target_supported": bool(rate_mask[number, window]),
                              "target_rate": float(rate_targets[number, window]) if rate_mask[number, window] else None})
    pd.DataFrame(rate_rows).to_csv(unit_root / "test_rate_predictions.csv", index=False)
    eol_limit = config["reporting"]["eol80_prediction_query_max_cycle"]
    eol_queries = np.arange(100, eol_limit + 1, dtype=np.float32)
    _, eol_trends = predictions(model, scaled, test, eol_queries, device, settings["batch_size"])
    eol_rows = []
    for position, number in enumerate(test):
        crossings = np.flatnonzero(eol_trends[position] <= 0.8)
        predicted_cycle = float(eol_limit)
        if len(crossings):
            right = int(crossings[0])
            predicted_cycle = float(eol_queries[right])
            if right > 0:
                y0, y1 = map(float, eol_trends[position, right - 1:right + 1])
                if y0 > y1:
                    predicted_cycle = float(eol_queries[right - 1] + (y0 - 0.8) / (y0 - y1))
        eol_rows.append({"cell_uid": identifiers[number], "predicted_trend_eol80_cycle": predicted_cycle,
                         "no_crossing_through_registered_horizon": not bool(len(crossings)),
                         "prediction_query_max_cycle": eol_limit})
    pd.DataFrame(eol_rows).to_csv(unit_root / "test_eol_predictions.csv", index=False)
    write_json(unit_root / "test_metrics.json", {**metrics, "arm": arm, "split": split_name, "seed": 42})
    files = ["UNIT_CONTRACT.json", "TRAIN_SCALER.json", "SELECTION_FROZEN.json", "training.jsonl",
             "test_predictions.parquet", "test_cell_metrics.csv", "test_metrics.json", "test_eol_predictions.csv", "test_rate_predictions.csv",
             "SELECTED.pt", "TRAIN_STATE.pt"]
    write_json(complete_path, {"status": "EVALUATED", "seed": 42, "successful_updates": successful_updates, "files": [
        {"path": name, "sha256": sha256(unit_root / name)} for name in files]})
    print(json.dumps({"status": "EVALUATED", "split": split_name, "arm": arm, **metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/unified_matr_seed42.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    data = load_frozen_data(args.data)
    if sha256(args.config) not in {item["sha256"] for item in data[-1]["files"]}:
        raise RuntimeError("Training config is outside the verified data-freeze fingerprint")
    splits = data[-2]["splits"]
    if len(splits) != 6 or sum(name.startswith("cell_fold_") for name in splits) != 5 or "unseen_protocol_sensitivity" not in splits:
        raise RuntimeError("The registered seed42 matrix requires five cell folds and one protocol-heldout split")
    sources = sorted((REPOSITORY / "models/unified_matr").glob("*.py"))
    sources += sorted((REPOSITORY / "scripts/unified_matr").glob("*.py"))
    sources.append(REPOSITORY / "scripts/run_experiment_b.py")
    registration = {"study_id": config["study_id"], "seed": 42, "config_sha256": sha256(args.config),
                    "data_freeze_sha256": sha256(args.data / "DATA_FREEZE.json"),
                    "sources": [{"path": str(path.relative_to(REPOSITORY)).replace("\\", "/"), "sha256": sha256(path)} for path in sources],
                    "planned_units": len(splits) * len(config["models"]), "config": config,
                    "initialization": initialization_registry(data[0])}
    args.run_root.mkdir(parents=True, exist_ok=True)
    with execution_lock(args.run_root / ".matrix.lock"):
        registration_path = args.run_root / "RUN_REGISTRATION.json"
        if registration_path.exists() and json.loads(registration_path.read_text(encoding="utf-8")) != registration:
            raise RuntimeError("Run registration differs from existing source, data or settings")
        write_json(registration_path, registration)
        for split_name, roles in splits.items():
            for arm in config["models"]:
                train_unit(arm, split_name, roles, data, config, args.data, args.run_root, args.device, registration)
        write_json(args.run_root / "MATRIX_COMPLETE.json", {"status": "COMPLETE", "seed": 42,
                   "completed_units": registration["planned_units"], "registration_sha256": sha256(registration_path)})


if __name__ == "__main__":
    main()
