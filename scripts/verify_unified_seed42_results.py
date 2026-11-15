"""Independent post-completion verification of the registered MATR seed42 matrix.

The preparation mode reads only registration, configuration and frozen data.
The full mode requires every formal completion marker before opening test
tables. It performs inference and independent arithmetic without optimization.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "scripts/unified_matr"))
import train_seed42 as training
from models.unified_matr import UnifiedMATRModel
from models.unified_matr.model import state_hash


ARMS = ("M0", "M1", "M2", "M3", "M3-NL", "M3-NP", "M3-NR")
SPLITS = tuple(f"cell_fold_{number}" for number in range(5)) + ("unseen_protocol_sensitivity",)
CONTRASTS = (("M3", "M0"), ("M1", "M0"), ("M2", "M1"), ("M3", "M2"),
             ("M3", "M3-NL"), ("M3", "M3-NP"), ("M3", "M3-NR"))
HORIZONS = ((101, 200), (201, 500), (501, 1000), (1001, 1000000))
RATE_WINDOWS = (("r200_500", 200, 500), ("r500_1000", 500, 1000))
REPLICATES = 2000


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def numerical_equal(left, right, atol=1e-10, rtol=1e-7):
    first, second = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return first.shape == second.shape and bool(np.allclose(first, second, atol=atol, rtol=rtol, equal_nan=True))


class VerificationError(RuntimeError):
    pass


class Checks:
    def __init__(self):
        self.rows = []

    def require(self, name, passed, detail=None):
        self.rows.append({"id": name, "status": "PASS" if bool(passed) else "FAIL", "detail": clean(detail)})
        if not passed:
            raise VerificationError(name)

    def finding(self, name, detail):
        self.rows.append({"id": name, "status": "QUALIFIED_FINDING", "detail": clean(detail)})


def compare_table(checks, name, actual, expected, keys, atol=1e-10, rtol=1e-7):
    columns = list(expected.columns)
    missing = sorted(set(columns) - set(actual))
    checks.require(name + "_schema", not missing, {"missing": missing})
    first = actual[columns].sort_values(keys).reset_index(drop=True)
    second = expected.sort_values(keys).reset_index(drop=True)
    checks.require(name + "_identities", len(first) == len(second) and not first.duplicated(keys).any()
                   and first[keys].equals(second[keys]), {"rows": len(second), "keys": keys})
    failures = []
    for column in columns:
        if column in keys:
            continue
        if pd.api.types.is_numeric_dtype(second[column]) or pd.api.types.is_bool_dtype(second[column]):
            equal = numerical_equal(pd.to_numeric(first[column]), second[column], atol, rtol)
        else:
            equal = first[column].fillna("").astype(str).equals(second[column].fillna("").astype(str))
        if not equal:
            failures.append(column)
    checks.require(name + "_values", not failures, {"different_columns": failures})


def independent_scaler(inputs, rows):
    """Recompute train-only supported population moments in float64."""
    result = {}
    for name, mask_name in training.FEATURE_MASKS.items():
        values = np.asarray(inputs[name][rows], dtype=np.float64)
        support = np.asarray(inputs[mask_name][rows], dtype=bool)
        while support.ndim < values.ndim:
            support = support[..., None]
        support = np.broadcast_to(support, values.shape)
        coordinates = values.shape[-1]
        flat_values = values.reshape(-1, coordinates)
        flat_support = support.reshape(-1, coordinates)
        counts = np.count_nonzero(flat_support, axis=0)
        means, deviations = [], []
        for column in range(coordinates):
            selected = flat_values[flat_support[:, column], column]
            mean = float(selected.mean()) if len(selected) else 0.0
            deviation = float(np.sqrt(np.mean((selected - mean) ** 2))) if len(selected) else 0.0
            means.append(mean)
            deviations.append(deviation if deviation >= 1e-6 else 1.0)
        result[name] = {"mean": means, "std": deviations, "count": counts.tolist()}
    return result


def prepare(checks, data_root, run_root):
    registration = read_json(run_root / "RUN_REGISTRATION.json")
    config = read_json(REPOSITORY / "configs/unified_matr_seed42.json")
    checks.require("registered_config_and_seed42", registration["config"] == config
                   and registration["seed"] == 42 and registration["planned_units"] == 42
                   and tuple(config["models"]) == ARMS, registration["planned_units"])
    checks.require("registered_config_bytes", registration["config_sha256"] == digest(REPOSITORY / "configs/unified_matr_seed42.json"))
    checks.require("data_freeze_bound_to_registration", registration["data_freeze_sha256"] == digest(data_root / "DATA_FREEZE.json"))
    for source in registration["sources"]:
        checks.require("registered_source_" + source["path"], digest(REPOSITORY / source["path"]) == source["sha256"])
    data = training.load_frozen_data(data_root)
    identifiers, eligible, cells, split = data[1], data[2], data[3], data[-2]
    checks.require("fixed169_and_historical_exposure", len(identifiers) == 169 and len(set(identifiers)) == 169
                   and cells.historical_exposure.all())
    checks.require("split_contract_identical_to_preregistration", split == read_json(REPOSITORY / "configs/unified_matr_seed42_splits.json"))
    counts = Counter()
    plan = []
    for name in SPLITS:
        roles = split["splits"][name]
        role_sets = [set(roles[role]) for role in ("train", "validation", "test")]
        disjoint = all(not role_sets[a] & role_sets[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        checks.require("physical_disjoint_" + name, disjoint and set.union(*role_sets) == set(identifiers))
        executed = {role: [uid for uid in roles[role] if eligible[identifiers.index(uid)]] for role in roles}
        updates = 150 * math.ceil(len(executed["train"]) / 8)
        plan.append({"split": name, "roles": executed, "updates_per_arm": updates})
        if name.startswith("cell_fold_"):
            counts.update(executed["test"])
        else:
            groups = split["protocol_groups"]
            train_groups = {groups[uid] for uid in executed["train"] + executed["validation"]}
            checks.require("protocol_sensitivity_groups_disjoint", not train_groups & {groups[uid] for uid in executed["test"]})
    checks.require("outer_OOF_each_eligible_UID_once", counts == Counter(uid for uid, valid in zip(identifiers, eligible) if valid))
    budget = sum(row["updates_per_arm"] * len(ARMS) for row in plan)
    checks.require("complete_registered_budget", config["training"]["epochs"] == 150
                   and config["training"]["batch_size"] == 8 and budget == 87150,
                   {"units": len(plan) * len(ARMS), "epochs_per_unit": 150, "optimizer_updates": budget})
    return registration, data, plan


def completion_guard(checks, run_root, analysis_root, registration):
    missing = [str(run_root / name / arm / "COMPLETE.json") for name in SPLITS for arm in ARMS
               if not (run_root / name / arm / "COMPLETE.json").is_file()]
    checks.require("all42_completion_markers_before_test_reads", not missing and (run_root / "MATRIX_COMPLETE.json").is_file(), {"missing": missing})
    matrix = read_json(run_root / "MATRIX_COMPLETE.json")
    checks.require("matrix_completion_identity", matrix["status"] == "COMPLETE" and matrix["completed_units"] == 42
                   and matrix["registration_sha256"] == digest(run_root / "RUN_REGISTRATION.json"))
    checks.require("completed_analysis_before_comparison", (analysis_root / "ANALYSIS_COMPLETE.json").is_file())
    analysis = read_json(analysis_root / "ANALYSIS_COMPLETE.json")
    checks.require("analysis_registration_and_source_binding", analysis["registration_sha256"] == digest(run_root / "RUN_REGISTRATION.json")
                   and analysis["data_freeze_sha256"] == registration["data_freeze_sha256"]
                   and analysis["analysis_source_sha256"] == digest(REPOSITORY / "scripts/unified_matr/analyze_results.py"))
    for item in analysis["files"]:
        checks.require("analysis_file_" + item["path"], digest(analysis_root / item["path"]) == item["sha256"])


def independent_metric_rows(prediction, test_ids, protocols, scope, split, arm):
    rows = []
    for uid in test_ids:
        frame = prediction.loc[prediction.cell_uid == uid]
        for label, left, right in (("trajectory", 101, 1000000),) + tuple((f"horizon_{a}_{b}", a, b) for a, b in HORIZONS):
            selected = frame.loc[frame.original_cycle_id.between(left, right)]
            errors = selected.prediction.to_numpy(float) - selected.target.to_numpy(float)
            values = {"mae": np.mean(np.abs(errors)) if len(errors) else np.nan,
                      "rmse": np.sqrt(np.mean(np.square(errors))) if len(errors) else np.nan}
            for name, value in values.items():
                rows.append({"scope": scope, "split": split, "arm": arm, "cell_uid": uid,
                             "protocol_uid": protocols[uid], "metric": label + "_" + name,
                             "value": value, "observations": len(errors)})
    return rows


def crossing_time(values, query):
    crossing = np.flatnonzero(values <= 0.8)
    if not len(crossing):
        return float(query[-1]), True
    index = int(crossing[0])
    result = float(query[index])
    if index and float(values[index - 1]) > float(values[index]):
        result = float(query[index - 1]) + (float(values[index - 1]) - 0.8) / (float(values[index - 1]) - float(values[index]))
    return result, False


def observed_trend_diagnostics(prediction, scope, split, arm):
    rows = []
    for uid, frame in prediction.groupby("cell_uid", sort=True):
        frame = frame.sort_values("original_cycle_id")
        changes = np.diff(frame.trend.to_numpy(float))
        distances = np.diff(frame.original_cycle_id.to_numpy(float))
        rising = changes > 0
        rows.append({"scope": scope, "split": split, "arm": arm, "cell_uid": uid,
                     "trajectory_min": frame.prediction.min(), "trajectory_max": frame.prediction.max(),
                     "trend_min": frame.trend.min(), "trend_max": frame.trend.max(),
                     "observed_query_points": len(frame), "observed_adjacent_pairs": len(changes),
                     "rising_step_count": int(rising.sum()), "rising_steps_over_1e_minus6_SOH": int((changes > 1e-6).sum()),
                     "rising_pair_fraction": float(rising.mean()) if len(changes) else np.nan,
                     "largest_positive_trend_step": float(max(0.0, changes.max())) if len(changes) else 0.0,
                     "largest_positive_trend_step_per_cycle": float(max(0.0, (changes / distances).max())) if len(changes) else 0.0,
                     "positive_rate_integral_arm": arm in ("M3", "M3-NL", "M3-NP")})
    return rows


def verify_unit(checks, split, arm, roles, registration, data, run_root, device, scaled, statistics):
    inputs, ids, _, cells, query, targets, observed, _, _, _, _ = data
    root = run_root / split / arm
    contract, complete, selection = (read_json(root / name) for name in ("UNIT_CONTRACT.json", "COMPLETE.json", "SELECTION_FROZEN.json"))
    tag = split + "/" + arm
    expected_updates = 150 * math.ceil(len(roles["train"]) / 8)
    required_files = {"UNIT_CONTRACT.json", "TRAIN_SCALER.json", "SELECTION_FROZEN.json", "training.jsonl", "SELECTED.pt", "TRAIN_STATE.pt",
                      "test_predictions.parquet", "test_cell_metrics.csv", "test_metrics.json", "test_rate_predictions.csv", "test_eol_predictions.csv"}
    checks.require(tag + "/identity_and_budget", contract["roles"] == roles and contract["arm"] == arm
                   and contract["seed"] == 42 and contract["split_name"] == split
                   and contract["registration_sha256"] == digest(run_root / "RUN_REGISTRATION.json")
                   and contract["training"] == registration["config"]["training"]
                   and complete["status"] == "EVALUATED" and complete["successful_updates"] == contract["expected_updates"] == expected_updates
                   and required_files.issubset({entry["path"] for entry in complete["files"]}))
    for entry in complete["files"]:
        checks.require(tag + "/artifact/" + entry["path"], digest(root / entry["path"]) == entry["sha256"])
    saved_scaler = read_json(root / "TRAIN_SCALER.json")
    scaler_equal = saved_scaler["training_ids"] == roles["train"]
    for name, stats in statistics.items():
        scaler_equal = scaler_equal and all(numerical_equal(saved_scaler["statistics"][name][key], value, atol=1e-10, rtol=1e-10) for key, value in stats.items())
    checks.require(tag + "/independent_training_only_scaler", scaler_equal)
    history = [json.loads(line) for line in (root / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    checks.require(tag + "/150_complete_epochs_and_all_updates", [row["epoch"] for row in history] == list(range(1, 151))
                   and [row["successful_updates"] for row in history] == [epoch * math.ceil(len(roles["train"]) / 8) for epoch in range(1, 151)]
                   and all(np.isfinite(row["train_loss"]) and np.isfinite(row["validation_mae"]) for row in history))
    running_best, flags = math.inf, []
    for row in history:
        flags.append(row["validation_mae"] < running_best)
        running_best = min(running_best, row["validation_mae"])
    best = min(history, key=lambda row: row["validation_mae"])
    checks.require(tag + "/validation_only_first_minimum_selection", [row["selected"] for row in history] == flags
                   and selection["selected_epoch"] == best["epoch"] and selection["validation_mae"] == best["validation_mae"]
                   and selection["checkpoint_sha256"] == digest(root / "SELECTED.pt"))
    selected = torch.load(root / "SELECTED.pt", map_location="cpu", weights_only=True)
    # These are hash-bound, locally generated formal checkpoints, including RNG
    # and optimizer state. No external/raw pickle is deserialized here.
    state = torch.load(root / "TRAIN_STATE.pt", map_location="cpu", weights_only=False)
    states_equal = selected["model"].keys() == state["selected_checkpoint"]["model"].keys()
    states_equal = states_equal and all(torch.equal(value, state["selected_checkpoint"]["model"][key]) for key, value in selected["model"].items())
    steps = [int(value["step"].item()) for value in state["optimizer"]["state"].values()]
    checks.require(tag + "/atomic_training_checkpoint_and_optimizer_steps", state["contract"] == contract
                   and selected["contract"] == contract and state["next_epoch"] == 150
                   and state["history"] == history and state["successful_updates"] == expected_updates
                   and bool(steps) and all(step == expected_updates for step in steps) and states_equal
                   and state["best_mae"] == selection["validation_mae"]
                   and selected["epoch"] == selection["selected_epoch"] and selected["validation_mae"] == selection["validation_mae"])
    settings = registration["config"]["training"]
    groups = state["optimizer"]["param_groups"]
    checks.require(tag + "/actual_AdamW_hyperparameters", all(group["lr"] == settings["learning_rate"]
                   and group["weight_decay"] == settings["weight_decay"] and list(group["betas"]) == settings["betas"]
                   and group["eps"] == settings["eps"] for group in groups))
    del state
    model = UnifiedMATRModel(arm, inputs["policy"].shape[-1], inputs["global_features"].shape[-1],
                             inputs["local_features"].shape[-1], seed=42,
                             observation_bound=settings["observation_residual_amplitude"],
                             soh_bounds=tuple(settings["observation_soh_bounds"]))
    checks.require(tag + "/normal_model_initialization_and_active_budget", state_hash(model) == registration["initialization"]["models"][arm]["initial_state_sha256"]
                   and sum(parameter.numel() for parameter in model.parameters()) == selection["parameters"]
                   and sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == selection["trainable_parameters"])
    model.load_state_dict(selected["model"], strict=True)
    model.to(device)
    indices = {role: np.asarray([ids.index(uid) for uid in members], dtype=int) for role, members in roles.items()}
    validation, _ = training.predictions(model, scaled, indices["validation"], query, device, 8)
    val_metrics, _, _ = training.trajectory_metrics(validation, targets[indices["validation"]], observed[indices["validation"]])
    checks.require(tag + "/selected_validation_prediction_replay", numerical_equal(val_metrics["cell_mean_mae"], selection["validation_mae"], atol=1e-7, rtol=5e-6), val_metrics)
    predictions, trends = training.predictions(model, scaled, indices["test"], query, device, 8)
    saved = pd.read_parquet(root / "test_predictions.parquet")
    replay_rows = []
    for position, number in enumerate(indices["test"]):
        slots = np.flatnonzero(observed[number])
        replay_rows.extend({"cell_uid": ids[number], "original_cycle_id": int(query[slot]), "target": float(targets[number, slot]),
                            "prediction": float(predictions[position, slot]), "trend": float(trends[position, slot])} for slot in slots)
    replay = pd.DataFrame(replay_rows)
    compare_table(checks, tag + "/all_selected_test_prediction_replay", saved, replay, ["cell_uid", "original_cycle_id"], atol=2e-6, rtol=2e-5)
    sort_keys = ["cell_uid", "original_cycle_id"]
    checks.require(tag + "/frozen_float32_targets_exact", np.array_equal(saved.sort_values(sort_keys).target.to_numpy(np.float32), replay.sort_values(sort_keys).target.to_numpy(np.float32)))
    maximum_error = float(np.max(np.abs(saved.sort_values(sort_keys).prediction.to_numpy() - replay.sort_values(sort_keys).prediction.to_numpy())))
    scope = "main_oof" if split.startswith("cell_fold_") else "unseen_protocol_sensitivity"
    # Delivered, hash-bound values are the arithmetic inputs for independent
    # statistics. The selected-weight replay above is a separate equality check.
    trend_diagnostics = observed_trend_diagnostics(saved, scope, split, arm)
    checks.require(tag + "/registered_SOH_observation_bounds", all(row["trajectory_min"] >= settings["observation_soh_bounds"][0]
                   and row["trajectory_max"] <= settings["observation_soh_bounds"][1] + 1e-6 for row in trend_diagnostics))
    if arm in ("M3", "M3-NL", "M3-NP"):
        checks.require(tag + "/rate_integral_observed_trend_monotonicity", all(row["rising_steps_over_1e_minus6_SOH"] == 0 for row in trend_diagnostics), trend_diagnostics)
    protocols = cells.protocol_uid.to_dict()
    rows = independent_metric_rows(saved, roles["test"], protocols, scope, split, arm)
    rates = pd.read_csv(root / "test_rate_predictions.csv")
    rate_lookup = rates.set_index(["cell_uid", "target_name"])
    expected_rates, rate_rows = [], []
    for position, number in enumerate(indices["test"]):
        uid = ids[number]
        for name, left, right in RATE_WINDOWS:
            start, end = map(float, predictions[position, [left - 101, right - 101]])
            truth = cells.loc[uid, name]
            supported = bool(pd.notna(truth))
            predicted_rate = (start - end) / (right - left)
            entry = {"cell_uid": uid, "protocol_uid": protocols[uid], "target_name": name, "start_cycle": left, "end_cycle": right,
                     "predicted_start_soh": start, "predicted_end_soh": end, "predicted_rate": predicted_rate,
                     "target_supported": supported, "target_rate": truth}
            expected_rates.append(entry)
            actual_entry = rate_lookup.loc[(uid, name)].to_dict()
            absolute = abs(float(actual_entry["predicted_rate"]) - truth) if supported else np.nan
            rate_rows.append({"cell_uid": uid, "target_name": name, **actual_entry, "scope": scope, "split": split, "arm": arm,
                              "absolute_error": absolute, "absolute_error_per_1000_cycles": absolute * 1000})
            rows.append({"scope": scope, "split": split, "arm": arm, "cell_uid": uid, "protocol_uid": protocols[uid],
                         "metric": name + "_absolute_error", "value": absolute, "observations": 2 if supported else 0})
    compare_table(checks, tag + "/all_rate_endpoint_replay", rates, pd.DataFrame(expected_rates), ["cell_uid", "target_name"], atol=2e-8, rtol=2e-5)
    eol_query = np.arange(100, 5001, dtype=np.float32)
    _, eol_trends = training.predictions(model, scaled, indices["test"], eol_query, device, 8)
    eol_rows = []
    for position, number in enumerate(indices["test"]):
        crossing, absent = crossing_time(eol_trends[position], eol_query)
        eol_rows.append({"cell_uid": ids[number], "predicted_trend_eol80_cycle": crossing,
                         "no_crossing_through_registered_horizon": absent, "prediction_query_max_cycle": 5000})
    saved_eol = pd.read_csv(root / "test_eol_predictions.csv")
    compare_table(checks, tag + "/all_trend_EOL_replay", saved_eol, pd.DataFrame(eol_rows), ["cell_uid"], atol=0.002, rtol=1e-6)
    unit_metrics = pd.DataFrame(rows)
    primary = unit_metrics.loc[unit_metrics.metric.isin(["trajectory_mae", "trajectory_rmse"])].pivot(index="cell_uid", columns="metric", values="value")
    expected_cell = primary.rename(columns={"trajectory_mae": "mae", "trajectory_rmse": "rmse"}).reset_index()
    expected_cell["observations"] = expected_cell.cell_uid.map(replay.groupby("cell_uid").size())
    expected_cell["protocol_uid"] = expected_cell.cell_uid.map(protocols)
    compare_table(checks, tag + "/cell_metrics", pd.read_csv(root / "test_cell_metrics.csv"), expected_cell, ["cell_uid"], atol=1e-7, rtol=5e-6)
    saved_metrics = read_json(root / "test_metrics.json")
    checks.require(tag + "/cell_weighted_metrics", numerical_equal(saved_metrics["cell_mean_mae"], primary.trajectory_mae.mean(), atol=1e-7, rtol=5e-6)
                   and numerical_equal(saved_metrics["cell_mean_rmse"], primary.trajectory_rmse.mean(), atol=1e-7, rtol=5e-6)
                   and saved_metrics["observations"] == len(replay) and saved_metrics["supported_cells"] == len(primary))
    summary = {"scope": scope, "split": split, "arm": arm, "seed": 42, "epochs": 150,
               "selected_epoch": selection["selected_epoch"], "validation_mae": selection["validation_mae"],
               "successful_updates": expected_updates, "parameters": selection["parameters"],
               "trainable_parameters": selection["trainable_parameters"], "elapsed_seconds": history[-1]["elapsed_seconds"],
               **{role + "_cells": len(members) for role, members in roles.items()}}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    history_core = [{key: row[key] for key in ("epoch", "train_loss", "validation_mae", "selected", "successful_updates")} for row in history]
    return {"metrics": rows, "rates": rate_rows, "eol": saved_eol.assign(scope=scope, split=split, arm=arm),
            "training": summary, "history": history_core, "weights": selected["model"],
            "prediction": saved, "max_absolute_prediction_replay_error": maximum_error,
            "observed_trend_diagnostics": trend_diagnostics}


def bootstrap_average(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"improvement": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "bootstrap_supported": 0, "resampling_units": 0}
    generator = np.random.default_rng(42)
    draws = generator.integers(len(values), size=(REPLICATES, len(values)))
    samples = np.sum(values[draws], axis=1) / len(values)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"improvement": float(np.mean(values)), "ci95_low": low, "ci95_high": high,
            "bootstrap_supported": REPLICATES, "resampling_units": len(values)}


def aggregate_metrics(metrics):
    valid = metrics.loc[metrics.value.notna()]
    protocols = valid.groupby(["scope", "arm", "metric", "protocol_uid"], sort=True).agg(
        protocol_mean=("value", "mean"), supported_cells=("cell_uid", "nunique"), observations=("observations", "sum")).reset_index()
    summary = []
    for (scope, arm, metric), frame in metrics.groupby(["scope", "arm", "metric"], sort=True):
        usable = frame.loc[frame.value.notna()]
        groups = protocols.loc[(protocols.scope == scope) & (protocols.arm == arm) & (protocols.metric == metric)]
        summary.append({"scope": scope, "arm": arm, "metric": metric, "cell_mean": usable.value.mean(),
                        "protocol_macro": groups.protocol_mean.mean(), "execution_test_cells": len(frame),
                        "supported_cells": len(usable), "supported_protocols": len(groups), "observations": int(usable.observations.sum()),
                        "unit": "SOH/cycle" if metric.startswith("r") else "SOH"})
    folds = valid.groupby(["scope", "split", "arm", "metric"], sort=True).agg(
        cell_mean=("value", "mean"), supported_cells=("cell_uid", "nunique"), observations=("observations", "sum")).reset_index()
    return pd.DataFrame(summary), protocols, folds


def independent_contrasts(metrics):
    summaries, cell_rows, protocol_rows = [], [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        for metric in sorted(metrics.metric.unique()):
            scoped = metrics.loc[(metrics.scope == scope) & (metrics.metric == metric)]
            indexed = scoped.pivot(index=["cell_uid", "protocol_uid"], columns="arm", values="value")
            for candidate, reference in CONTRASTS:
                pair = indexed[[reference, candidate]].rename(columns={reference: "reference_error", candidate: "candidate_error"}).reset_index()
                if not pair.reference_error.notna().equals(pair.candidate_error.notna()):
                    raise VerificationError("Cross-arm support differs")
                pair["improvement"] = pair.reference_error - pair.candidate_error
                pair["relative_improvement_percent"] = np.divide(pair.improvement.to_numpy() * 100, pair.reference_error.to_numpy(), out=np.full(len(pair), np.nan), where=pair.reference_error.to_numpy() > 0)
                pair["supported"] = pair.improvement.notna()
                contrast = candidate + "_vs_" + reference
                pair = pair.assign(scope=scope, metric=metric, contrast=contrast, candidate=candidate, reference=reference)
                cell_rows.extend(pair.to_dict("records"))
                supported = pair.loc[pair.supported].sort_values("cell_uid")
                groups = supported.groupby("protocol_uid", sort=True).agg(improvement=("improvement", "mean"), reference_error=("reference_error", "mean"), candidate_error=("candidate_error", "mean"), supported_cells=("cell_uid", "size")).reset_index()
                protocol_rows.extend(groups.assign(scope=scope, metric=metric, contrast=contrast).to_dict("records"))
                for estimator, unit, sample in (("cell_mean", "paired_cell", supported), ("protocol_macro", "protocol_group", groups)):
                    first, last = sample.reference_error.mean(), sample.candidate_error.mean()
                    statistics = bootstrap_average(sample.improvement)
                    summaries.append({"scope": scope, "contrast": contrast, "candidate": candidate, "reference": reference, "metric": metric,
                                      "estimator": estimator, "sampling_unit": unit, "reference_metric": first, "candidate_metric": last, **statistics,
                                      "relative_improvement_percent": (first - last) / first * 100 if first > 0 else np.nan,
                                      "paired_cells": len(supported), "protocol_groups": len(groups), "bootstrap_requested": REPLICATES,
                                      "bootstrap_seed": 42, "bootstrap_identifiable": statistics["resampling_units"] > 1,
                                      "improved_cells": int(supported.improvement.gt(0).sum()), "worsened_cells": int(supported.improvement.lt(0).sum()),
                                      "tied_cells": int(supported.improvement.eq(0).sum()), "direction": "reference_error_minus_candidate_error"})
    return pd.DataFrame(summaries), pd.DataFrame(cell_rows), pd.DataFrame(protocol_rows)


def concordance_pairs(times, events, predictions):
    early, late = [], []
    for first in range(len(times)):
        for second in range(len(times)):
            if events[first] and times[first] < times[second]:
                early.append(first)
                late.append(second)
    early, late = np.asarray(early, dtype=int), np.asarray(late, dtype=int)
    scores = (predictions[:, early] < predictions[:, late]).astype(float)
    scores += (predictions[:, early] == predictions[:, late]) * 0.5
    return early, late, scores


def eol_statistics(eol, cells):
    truth = cells[["protocol_uid", "eol80_event_observed", "eol80_first_observed_crossing_cycle", "last_observed_valid_cycle"]].copy()
    truth["event"] = truth.eol80_event_observed.astype(bool)
    truth["outcome_cycle"] = np.where(truth.event, truth.eol80_first_observed_crossing_cycle, truth.last_observed_valid_cycle)
    truth["at_risk100"] = truth.outcome_cycle > 100
    merged = eol.merge(truth, left_on="cell_uid", right_index=True, validate="many_to_one")
    summaries, contrasts = [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        scoped = merged.loc[merged.scope == scope]
        risk = scoped.loc[scoped.at_risk100]
        wide = risk.pivot(index="cell_uid", columns="arm", values="predicted_trend_eol80_cycle").sort_index().reindex(columns=ARMS)
        outcome = truth.loc[wide.index]
        predictions = wide.to_numpy(float).T
        early, late, scores = concordance_pairs(outcome.outcome_cycle.to_numpy(), outcome.event.to_numpy(), predictions)
        absolute = scores.mean(axis=1) if len(early) else np.full(len(ARMS), np.nan)
        for number, arm in enumerate(ARMS):
            arm_rows = scoped.loc[scoped.arm == arm]
            concordant, ties = int(np.sum(scores[number] == 1)), int(np.sum(scores[number] == 0.5))
            summaries.append({"scope": scope, "arm": arm, "harrell_c": absolute[number], "comparable_pairs": len(early),
                              "concordant_pairs": concordant, "prediction_ties": ties, "discordant_pairs": len(early) - concordant - ties,
                              "executed_test_cells": len(arm_rows), "at_risk100_cells": len(outcome), "observed_events": int(outcome.event.sum()),
                              "right_censored": int((~outcome.event).sum()), "events_at_or_before100_excluded": int((~arm_rows.at_risk100 & arm_rows.event).sum()),
                              "no_observed_followup_after100_excluded": int((~arm_rows.at_risk100 & ~arm_rows.event).sum()),
                              "predicted_no_crossing_through5000": int(arm_rows.no_crossing_through_registered_horizon.sum()), "prediction_query_max_cycle": 5000})
        groups, inverse = np.unique(outcome.protocol_uid.to_numpy(str), return_inverse=True)
        for unit, size in (("cell", len(outcome)), ("protocol_group", len(groups))):
            draws = np.full((REPLICATES, len(ARMS)), np.nan)
            if size:
                generator = np.random.default_rng(42)
                multiplicities = generator.multinomial(size, np.full(size, 1 / size), size=REPLICATES)
                if unit == "protocol_group":
                    multiplicities = multiplicities[:, inverse]
                for start in range(0, REPLICATES, 64):
                    weights = multiplicities[start:start + 64]
                    pairs = weights[:, early] * weights[:, late]
                    denominators = pairs.sum(axis=1)
                    numerators = pairs @ scores.T
                    draws[start:start + 64] = np.divide(numerators, denominators[:, None], out=np.full_like(numerators, np.nan), where=denominators[:, None] > 0)
            for candidate, reference in CONTRASTS:
                first, last = ARMS.index(reference), ARMS.index(candidate)
                delta = draws[:, last] - draws[:, first]
                supported = delta[np.isfinite(delta)]
                low, high = np.percentile(supported, [2.5, 97.5]) if len(supported) else (np.nan, np.nan)
                contrasts.append({"scope": scope, "contrast": candidate + "_vs_" + reference, "candidate": candidate, "reference": reference,
                                  "metric": "eol80_harrell_c", "estimator": "harrell_concordance", "sampling_unit": unit,
                                  "reference_metric": absolute[first], "candidate_metric": absolute[last], "improvement": absolute[last] - absolute[first],
                                  "ci95_low": low, "ci95_high": high, "bootstrap_requested": REPLICATES, "bootstrap_supported": len(supported),
                                  "bootstrap_seed": 42, "paired_cells": len(outcome), "protocol_groups": len(groups),
                                  "direction": "candidate_concordance_minus_reference_concordance"})
    return pd.DataFrame(summaries), pd.DataFrame(contrasts)


def verify_analysis(checks, results, data, analysis_root):
    metrics = pd.DataFrame([row for result in results for row in result["metrics"]])
    rates = pd.DataFrame([row for result in results for row in result["rates"]])
    eol = pd.concat([result["eol"] for result in results], ignore_index=True)
    summary, protocols, folds = aggregate_metrics(metrics)
    contrasts, pairs, protocol_pairs = independent_contrasts(metrics)
    eol_summary, eol_contrasts = eol_statistics(eol, data[3])
    tables = {
        "cell_metrics.csv": (metrics, ["scope", "arm", "cell_uid", "metric"]),
        "rate_predictions_and_errors.csv": (rates, ["scope", "arm", "cell_uid", "target_name"]),
        "protocol_metrics.csv": (protocols, ["scope", "arm", "metric", "protocol_uid"]),
        "split_metrics.csv": (folds, ["scope", "split", "arm", "metric"]),
        "paired_cell_contrasts.csv": (pairs, ["scope", "metric", "contrast", "cell_uid"]),
        "paired_protocol_contrasts.csv": (protocol_pairs, ["scope", "metric", "contrast", "protocol_uid"]),
        "eol80_summary.csv": (eol_summary, ["scope", "arm"]),
        "eol80_fixed_contrasts.csv": (eol_contrasts, ["scope", "contrast", "sampling_unit"]),
        "training_units.csv": (pd.DataFrame([row["training"] for row in results]), ["split", "arm"]),
    }
    for scope, metric_file, contrast_file in (("main_oof", "main_oof_metrics.csv", "main_oof_fixed_contrasts.csv"),
                                               ("unseen_protocol_sensitivity", "protocol_heldout_metrics.csv", "protocol_heldout_fixed_contrasts.csv")):
        tables[metric_file] = (summary.loc[summary.scope == scope], ["scope", "arm", "metric"])
        tables[contrast_file] = (contrasts.loc[contrasts.scope == scope], ["scope", "contrast", "metric", "estimator"])
    for name, (expected, keys) in tables.items():
        compare_table(checks, "independent_analysis/" + name, pd.read_csv(analysis_root / name), expected, keys, atol=2e-8, rtol=2e-5)
    analysis_json = read_json(analysis_root / "ANALYSIS_SUMMARY.json")
    checks.require("analysis_JSON_training_budget_and_denominators", analysis_json["completed_training_units"] == 42
                   and analysis_json["actual_optimizer_updates"] == 87150 and analysis_json["qualification_cells"] == 169
                   and analysis_json["early_input_eligible_cells"] == 169 and analysis_json["training_seed"] == 42)
    for field, expected, keys in (("metric_summary", summary, ["scope", "arm", "metric"]),
                                   ("fixed_metric_contrasts", contrasts, ["scope", "contrast", "metric", "estimator"]),
                                   ("eol80_summary", eol_summary, ["scope", "arm"]),
                                   ("eol80_contrasts", eol_contrasts, ["scope", "contrast", "sampling_unit"])):
        compare_table(checks, "independent_analysis_JSON/" + field, pd.DataFrame(analysis_json[field]), expected, keys, atol=2e-8, rtol=2e-5)
    qualification = data[3].reset_index()
    qualification_columns = ["cell_uid", "protocol_uid", "input_eligible"] + [name for name in (
        "historical_exposure", "early_curve_valid_count", "admission_status", "failure_reasons",
        "eol80_event_observed", "eol80_first_observed_crossing_cycle", "last_observed_valid_cycle") if name in qualification]
    qualification = qualification[qualification_columns].copy()
    qualification["future_valid_observations"] = qualification.cell_uid.map({uid: int(data[6][position].sum()) for position, uid in enumerate(data[1])})
    compare_table(checks, "independent_analysis/qualification_denominators.csv", pd.read_csv(analysis_root / "qualification_denominators.csv"), qualification, ["cell_uid"])
    negative = pairs.loc[pairs.improvement < 0]
    failures = pd.read_csv(analysis_root / "all_failures.csv")
    published = failures.loc[failures.code == "candidate_error_larger"]
    negative_keys = ["scope", "metric", "contrast", "cell_uid"]
    checks.require("all_negative_cell_results_retained", set(map(tuple, negative[negative_keys].to_numpy())) == set(map(tuple, published[negative_keys].to_numpy())))
    checks.require("no_unresolved_integrity_errors", not failures.severity.eq("ERROR").any())
    main = metrics.loc[(metrics.scope == "main_oof") & (metrics.metric == "trajectory_mae")]
    checks.require("main_OOF169_once_per_arm", all(len(main.loc[main.arm == arm]) == 169 and main.loc[main.arm == arm, "cell_uid"].nunique() == 169 for arm in ARMS))
    return {"metric_rows": len(metrics), "trajectory_summary": summary.loc[summary.metric == "trajectory_mae"].to_dict("records"),
            "contrast_rows": len(contrasts), "eol_contrast_rows": len(eol_contrasts), "negative_cell_contrasts": len(negative)}


def run_verification(checks, args):
    registration, data, plan = prepare(checks, args.data, args.run_root)
    preparation = {"units": 42, "epochs_per_unit": 150, "optimizer_updates": 87150,
                   "replay": "all42 selected checkpoints via normal model/predictions entrypoints",
                   "splits": [{"split": row["split"], "role_sizes": {role: len(ids) for role, ids in row["roles"].items()}, "updates_per_arm": row["updates_per_arm"]} for row in plan],
                   "test_results_read": False}
    if args.prepare_only:
        return preparation
    completion_guard(checks, args.run_root, args.analysis_root, registration)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise VerificationError("Requested replay CUDA device is unavailable")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    results, replay_receipts, duplicates = [], [], []
    for split_plan in plan:
        split, roles = split_plan["split"], split_plan["roles"]
        indices = np.asarray([data[1].index(uid) for uid in roles["train"]], dtype=int)
        independent = independent_scaler(data[0], indices)
        scaled, canonical = training.fold_scaler(data[0], indices)
        checks.require(split + "/normal_scaler_equals_independent_supported_moments", all(numerical_equal(canonical[name][key], value, 1e-10, 1e-10) for name, statistics in independent.items() for key, value in statistics.items()))
        previous = None
        for arm in ARMS:
            result = verify_unit(checks, split, arm, roles, registration, data, args.run_root, device, scaled, independent)
            if arm == "M2":
                previous = result
            if arm == "M3-NR":
                weights_equal = previous["weights"].keys() == result["weights"].keys() and all(torch.equal(value, result["weights"][key]) for key, value in previous["weights"].items())
                rate_columns = ["cell_uid", "target_name", "predicted_start_soh", "predicted_end_soh", "predicted_rate", "target_supported", "target_rate"]
                first_rates = pd.DataFrame(previous["rates"])[rate_columns].sort_values(["cell_uid", "target_name"]).reset_index(drop=True)
                last_rates = pd.DataFrame(result["rates"])[rate_columns].sort_values(["cell_uid", "target_name"]).reset_index(drop=True)
                duplicate_checks = {"trajectory_and_trend_equal": previous["prediction"].equals(result["prediction"]),
                                    "rate_endpoints_equal": first_rates.equals(last_rates),
                                    "eol_risk_scores_equal": previous["eol"].drop(columns="arm").equals(result["eol"].drop(columns="arm")),
                                    "training_history_equal_excluding_elapsed": previous["history"] == result["history"],
                                    "selected_weight_tensors_equal": weights_equal,
                                    "selected_epoch_equal": previous["training"]["selected_epoch"] == result["training"]["selected_epoch"]}
                equal = all(duplicate_checks.values())
                if equal:
                    checks.require(split + "/M2_NR_exact_selected_weights_history_and_predictions", True, duplicate_checks)
                else:
                    checks.finding(split + "/M2_NR_exact_selected_weights_history_and_predictions", duplicate_checks)
                duplicates.append({"split": split, **duplicate_checks, "compared_weight_tensors": len(result["weights"]), "reproduced": equal})
            replay_receipts.append({"split": split, "arm": arm, "max_absolute_prediction_error": result["max_absolute_prediction_replay_error"], "selected_epoch": result["training"]["selected_epoch"]})
            results.append(result)
            print(json.dumps({"verification": "selected_checkpoint_replay", "completed": len(results), "required": 42, "split": split, "arm": arm}), flush=True)
        for result in results:
            result.pop("weights", None)
    statistical = verify_analysis(checks, results, data, args.analysis_root)
    compare_table(checks, "independent_analysis/duplicate_reproduction.csv", pd.read_csv(args.analysis_root / "duplicate_reproduction.csv"), pd.DataFrame(duplicates), ["split"])
    analysis_json = read_json(args.analysis_root / "ANALYSIS_SUMMARY.json")
    reproduced = sum(item["reproduced"] for item in duplicates)
    expected_analysis_status = "ANALYZED" if reproduced == len(SPLITS) else "ANALYZED_WITH_REPRODUCTION_FAILURE"
    checks.require("duplicate_failure_disclosed_in_sealed_analysis", analysis_json["M2_NR_reproduced_splits"] == reproduced
                   and analysis_json["status"] == expected_analysis_status,
                   {"reproduced_splits": reproduced, "required_splits": len(SPLITS), "analysis_status": analysis_json["status"]})
    trend_cells = pd.DataFrame([row for result in results for row in result["observed_trend_diagnostics"]])
    trend_arms = []
    for (scope, arm), frame in trend_cells.groupby(["scope", "arm"], sort=True):
        total_pairs = int(frame.observed_adjacent_pairs.sum())
        total_rises = int(frame.rising_step_count.sum())
        trend_arms.append({"scope": scope, "arm": arm, "cells": len(frame),
                           "trajectory_min": frame.trajectory_min.min(), "trajectory_max": frame.trajectory_max.max(),
                           "trend_min": frame.trend_min.min(), "trend_max": frame.trend_max.max(),
                           "observed_adjacent_pairs": total_pairs, "rising_step_count": total_rises,
                           "rising_pair_fraction": total_rises / total_pairs if total_pairs else np.nan,
                           "cells_with_rises": int(frame.rising_step_count.gt(0).sum()),
                           "cell_fraction_with_rises": float(frame.rising_step_count.gt(0).mean()),
                           "largest_positive_trend_step": frame.largest_positive_trend_step.max(),
                           "largest_positive_trend_step_per_cycle": frame.largest_positive_trend_step_per_cycle.max()})
    total = sum(result["training"]["successful_updates"] for result in results)
    checks.require("all42_actual_optimizer_updates87150", total == 87150)
    freeze = read_json(args.data / "DATA_FREEZE.json")
    checks.require("frozen44_and_registration_unchanged_after_replay", all(digest(args.data / item["path"]) == item["sha256"] for item in freeze["files"])
                   and read_json(args.run_root / "RUN_REGISTRATION.json") == registration
                   and digest(args.data / "DATA_FREEZE.json") == registration["data_freeze_sha256"])
    return {**preparation, "test_results_read": True, "selected_checkpoint_replays": replay_receipts,
            "M2_NR_duplicate_reproduction": duplicates, "independent_statistics": statistical,
            "M2_NR_reproduced_splits": reproduced, "registered_duplicate_reproduction_complete": reproduced == len(SPLITS),
            "actual_optimizer_updates": total, "inference_device": str(device),
            "history_verification_limit": "All150 history rows, per-epoch cumulative updates, final optimizer states, atomic training checkpoint and first validation argmin are verified. Only actually saved selected weights are independently replayed. Historical unsaved epoch checkpoints are unavailable for separate inference replay.",
            "normal_entrypoint_replay_scope": "The registered model and normal prediction entrypoints are replayed from selected weights. Architecture is not independently reimplemented. Metrics, supported moments, pairing and bootstrap arithmetic are independently recomputed from immutable delivered predictions.",
            "observed_trend_by_arm": trend_arms, "observed_trend_by_cell": trend_cells.to_dict("records"),
            "trend_interpretation": "M3/M3-NL/M3-NP use positive-rate integrated loss. SOH clamping permits flat segments. M0/M1/M2/M3-NR retain the registered legacy analytic flow without a mathematical monotonicity guarantee. Rises are counted on consecutive actual observed native test queries, with actual-cycle-normalized magnitudes, and remain a reported limitation without model or metric edits."}


def selfcheck(checks):
    checks.require("bootstrap_reproducible", bootstrap_average([1, -1, 2]) == bootstrap_average([1, -1, 2]))
    checks.require("bootstrap_empty_preserves_missing", bootstrap_average([])["bootstrap_supported"] == 0)
    values = np.asarray([0.9, 0.85, 0.75], np.float32)
    when, absent = crossing_time(values, np.arange(100, 103, dtype=np.float32))
    checks.require("crossing_linear_interpolation", numerical_equal(when, 101.5, atol=1e-6) and not absent)
    checks.require("no_crossing_fixed_horizon", crossing_time(np.asarray([1.0, 0.9]), np.asarray([100, 5000])) == (5000.0, True))
    early, late, scores = concordance_pairs(np.asarray([1, 2, 2, 3]), np.asarray([True, True, False, True]), np.asarray([[1, 2, 2, 3], [3, 2, 2, 1]]))
    checks.require("concordance_event_censor_and_time_ties", len(early) == 4 and scores[0].mean() == 1 and scores[1].mean() == 0)
    prediction = pd.DataFrame({"cell_uid": ["a", "b", "b"], "original_cycle_id": [101, 101, 201], "prediction": [1., 2., 4.], "target": [0., 0., 0.]})
    rows = pd.DataFrame(independent_metric_rows(prediction, ["a", "b", "c"], {"a": "p", "b": "q", "c": "q"}, "main_oof", "cell_fold_0", "M0"))
    selected = rows.loc[rows.metric == "trajectory_mae"].set_index("cell_uid")
    checks.require("cell_mean_and_unsupported_denominator", selected.loc["a", "value"] == 1 and selected.loc["b", "value"] == 3 and pd.isna(selected.loc["c", "value"]))
    gapped = pd.DataFrame({"cell_uid": ["a"] * 3, "original_cycle_id": [101, 103, 110],
                           "prediction": [0.9, 0.91, 0.89], "trend": [0.9, 0.91, 0.89]})
    diagnostic = observed_trend_diagnostics(gapped, "main_oof", "cell_fold_0", "M0")[0]
    checks.require("actual_observed_query_trend_rises_and_cycle_distances", diagnostic["rising_step_count"] == 1
                   and diagnostic["observed_adjacent_pairs"] == 2 and diagnostic["rising_pair_fraction"] == 0.5
                   and numerical_equal(diagnostic["largest_positive_trend_step_per_cycle"], 0.005))
    truth = pd.DataFrame({"cell_uid": ["a", "b", "c", "d"], "protocol_uid": ["p", "p", "q", "q"],
                          "eol80_event_observed": [True, True, False, True],
                          "eol80_first_observed_crossing_cycle": [50., 200., np.nan, 400.],
                          "last_observed_valid_cycle": [100., 250., 350., 450.]}).set_index("cell_uid")
    eol = pd.DataFrame([{"scope": scope, "split": "fixture", "arm": arm, "cell_uid": uid,
                         "predicted_trend_eol80_cycle": 5000., "no_crossing_through_registered_horizon": True,
                         "prediction_query_max_cycle": 5000} for scope in ("main_oof", "unseen_protocol_sensitivity") for arm in ARMS for uid in truth.index])
    eol_summary, eol_contrasts = eol_statistics(eol, truth)
    checks.require("EOL_at_risk_event_censor_and_prediction_ties", eol_summary.at_risk100_cells.eq(3).all()
                   and eol_summary.harrell_c.eq(0.5).all() and eol_summary.comparable_pairs.eq(2).all())
    checks.require("EOL_paired_cell_protocol_bootstrap_empty_draws_retained", len(eol_contrasts) == 28
                   and eol_contrasts.improvement.eq(0).all() and eol_contrasts.bootstrap_supported.between(1, 2000).all())
    generator = np.random.default_rng(42)
    inputs = {"raw_viq": generator.normal(size=(2, 2, 3)), "reference_viq": generator.normal(size=(2, 2, 3)),
              "point_mask": np.asarray([[True, False], [True, True]]), "policy": generator.normal(size=(2, 15)),
              "policy_mask": np.ones((2, 15), bool), "global_features": generator.normal(size=(2, 2, 20)),
              "global_mask": np.ones((2, 2, 20), bool), "local_residual": generator.normal(size=(2, 2, 64)),
              "local_mask": np.ones((2, 2, 64), bool), "local_features": generator.normal(size=(2, 2, 8, 26)),
              "local_feature_mask": np.ones((2, 2, 8, 26), bool)}
    inputs["global_mask"][0, :, 0] = False
    inputs["global_features"][0, :, 0] = np.nan
    independent = independent_scaler(inputs, [0])
    _, canonical = training.fold_scaler(inputs, [0])
    checks.require("independent_scaler_matches_normal_supported_moments", all(numerical_equal(canonical[name][key], value, 1e-10, 1e-10) for name, stats in independent.items() for key, value in stats.items()))
    for name in training.FEATURE_MASKS:
        inputs[name][1] = 1e20
    checks.require("independent_scaler_excludes_heldout_cell", independent_scaler(inputs, [0]) == independent)
    # Synthetic comparisons exercise the independent pair-loop implementation
    # against the registered matrix implementation without formal test data.
    import analyze_results as canonical_analysis

    synthetic_cells = 19
    labels = [f"synthetic_{index:02d}" for index in range(synthetic_cells)]
    first_crossing = generator.integers(80, 1500, synthetic_cells).astype(float)
    events = generator.random(synthetic_cells) > 0.25
    source = pd.DataFrame({"cell_uid": labels, "protocol_uid": [f"protocol_{index % 5}" for index in range(synthetic_cells)],
                           "eol80_event_observed": events, "eol80_first_observed_crossing_cycle": np.where(events, first_crossing, np.nan),
                           "last_observed_valid_cycle": first_crossing + generator.integers(0, 100, synthetic_cells)}).set_index("cell_uid", drop=False)
    synthetic_eol = pd.DataFrame([{"scope": scope, "split": "synthetic", "arm": arm, "cell_uid": uid,
                                   "predicted_trend_eol80_cycle": float(generator.integers(100, 3000)),
                                   "no_crossing_through_registered_horizon": False, "prediction_query_max_cycle": 5000}
                                  for scope in ("main_oof", "unseen_protocol_sensitivity") for arm in ARMS for uid in labels])
    independent_eol, independent_intervals = eol_statistics(synthetic_eol, source)
    reference_eol, reference_intervals, _ = canonical_analysis.eol_analysis(synthetic_eol, source)
    compare_table(checks, "synthetic_nonconstant_EOL_pair_loop_vs_matrix", reference_eol, independent_eol, ["scope", "arm"], atol=1e-12, rtol=1e-12)
    compare_table(checks, "synthetic_nonconstant_EOL_cell_and_cluster_bootstrap", reference_intervals, independent_intervals,
                  ["scope", "contrast", "sampling_unit"], atol=1e-12, rtol=1e-12)
    return {"test_results_read": False, "scope": "synthetic verification mathematics only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPOSITORY / "data/unified_matr_seed42")
    parser.add_argument("--run-root", type=Path, default=REPOSITORY.parent / "private/unified_matr_seed42/runs")
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and not args.selfcheck and args.analysis_root is None:
        parser.error("Full verification requires --analysis-root")
    checks, summary, error = Checks(), {}, None
    try:
        summary = selfcheck(checks) if args.selfcheck else run_verification(checks, args)
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    status = "FAIL" if error else "PREPARED" if args.prepare_only else "PASS"
    if not error and summary.get("registered_duplicate_reproduction_complete") is False:
        status = "VERIFIED_WITH_REPRODUCTION_FAILURE"
    result = {"status": status,
              "verification_source_sha256": digest(Path(__file__)), "checks": checks.rows, "summary": summary,
              "error": error, "optimizer_updates_performed_by_verifier": 0, "model_fits_performed_by_verifier": 0,
              "historical_exposure": "All fixed169 cells participated in historical exploration. Training-seed uncertainty is unmeasured under seed42 only."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".pending")
    temporary.write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": result["status"], "checks": len(checks.rows), "error": error}, ensure_ascii=False))
    if error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
