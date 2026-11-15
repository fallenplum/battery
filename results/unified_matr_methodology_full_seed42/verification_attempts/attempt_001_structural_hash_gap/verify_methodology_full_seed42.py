"""Replay all six selected M4 checkpoints and independently verify arithmetic.

No optimizer is constructed. Historical references are read from saved results.
The model replay uses its normal registered entry point. The statistical
calculations below are independently implemented from the analysis program.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts.unified_matr import train_methodology_full as training
from scripts import verify_unified_seed42_results as independent
from scripts import analyze_methodology_full_seed42 as analysis

ARM = training.ARM
SPLITS = training.SPLITS
REFERENCES = ("M1", "M2", "M3")


def tensor_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def require_same(checks, name, first, last, atol=1e-7, rtol=2e-6):
    a, b = np.asarray(first), np.asarray(last)
    error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size and a.shape == b.shape else None
    checks.require(name, independent.numerical_equal(a, b, atol=atol, rtol=rtol),
                   {"shape_actual": list(a.shape), "shape_expected": list(b.shape),
                    "max_absolute_error": error, "absolute_tolerance": atol, "relative_tolerance": rtol})


def check_mechanism_math(checks, prefix, diagnostic, cycle_ids, model):
    persistence = diagnostic["persistence"]
    values = {key: value.detach().cpu().numpy() for key, value in persistence.items()}
    valid, rho, state = values["valid"], values["rho"], values["state"]
    gap, previous = values["gap"], values["previous_valid_index"]
    current = values["current_response"]
    expected = np.zeros_like(state, dtype=np.float64)
    expected_internal = np.zeros_like(state, dtype=np.float64)
    expected_write = np.zeros_like(state, dtype=np.float64)
    expected_drift = np.zeros_like(state, dtype=np.float64)
    transition_count = 0
    for cell in range(len(valid)):
        carried = np.zeros(state.shape[-1], dtype=float)
        prior = -1
        for step in range(valid.shape[1]):
            if valid[cell, step]:
                expected_gap = max(float(cycle_ids[cell, step] - (cycle_ids[cell, prior] if prior >= 0 else 0)), 1)
                checks.require(prefix + f"/gap/{cell}/{step}", float(gap[cell, step]) == expected_gap and int(previous[cell, step]) == prior)
                write = 1.0 if prior < 0 else 1 - np.power(rho[cell, step].astype(float), expected_gap)
                before = carried.copy()
                carried = np.asarray(write) * current[cell, step] + (1 - np.asarray(write)) * carried
                expected[cell, step] = carried
                expected_write[cell, step] = write
                if prior >= 0:
                    expected_drift[cell, step] = (carried - before) / expected_gap
                    transition_count += 1
                prior = step
            expected_internal[cell, step] = carried
    require_same(checks, prefix + "/independent_sequential_dynamic_recurrence", state, expected, atol=2e-5, rtol=2e-5)
    require_same(checks, prefix + "/invalid_internal_hold", values["internal_state"], expected_internal, atol=2e-5, rtol=2e-5)
    require_same(checks, prefix + "/first_valid_and_actual_gap_write", values["write_weight"], expected_write, atol=1e-6, rtol=2e-6)
    require_same(checks, prefix + "/state_change_per_actual_cycle", values["drift_per_cycle"], expected_drift, atol=2e-5, rtol=2e-5)
    checks.require(prefix + "/rho_bounds_and_finiteness", np.isfinite(rho).all() and bool((rho >= .01).all() and (rho <= .99).all()))
    checks.require(prefix + "/invalid_output_and_write_zero", np.all(state[~valid] == 0) and np.all(values["write_weight"][~valid] == 0))
    for layer, history in enumerate(diagnostic["history"]):
        original = history["B"].detach().cpu().numpy()
        modified = history["Bprime"].detach().cpu().numpy()
        alteration = history["write_modulation"].detach().cpu().numpy()
        supported = history["modulation_valid"].detach().cpu().numpy()
        gate = history["gate"].detach().cpu().numpy()
        weights = model.history[layer].mixer.write_projection.weight.detach().cpu().numpy()
        projected = state @ weights.T
        expected_modulation = np.where(supported[..., None], gate * np.tanh(projected), 0)
        require_same(checks, prefix + f"/B_layer{layer}_write_formula", alteration, expected_modulation, atol=1e-6, rtol=1e-5)
        require_same(checks, prefix + f"/B_layer{layer}_actual_modulation", modified, original + alteration)
        checks.require(prefix + f"/B_layer{layer}_invalid_no_modulation", np.all(alteration[~supported] == 0))
    return float(np.square(expected_drift).sum() / (max(transition_count, 1) * state.shape[-1]))


def independent_mechanism_summary(output, dense, sparse, sparse_query, identifiers, anchors):
    persistence = {name: value.detach().cpu().numpy() for name, value in output["diagnostics"]["persistence"].items()}
    rows = []
    for local, uid in enumerate(identifiers):
        valid = persistence["valid"][local]
        rho, write = persistence["rho"][local][valid], persistence["write_weight"][local][valid]
        drift = persistence["drift_per_cycle"][local][valid]
        trend = dense["trend"][local].detach().cpu().numpy()
        raw = dense["raw_trend"][local].detach().cpu().numpy()
        trajectory = output["trajectory"][local].detach().cpu().numpy()
        rate = dense["rate_per_1000_cycles"][local].detach().cpu().numpy()
        row = {"cell_uid": uid, "valid_prefix_cycles": int(np.count_nonzero(valid)),
               "rho_min": float(np.min(rho)), "rho_max": float(np.max(rho)), "rho_mean": float(np.mean(rho)),
               "write_mean": float(np.mean(write)), "max_actual_cycle_gap": float(np.max(persistence["gap"][local][valid])),
               "drift_mean_abs_per_cycle": float(np.mean(np.abs(drift))), "drift_positive_fraction": float(np.mean(drift > 0)),
               "rate_min_per1000": float(np.min(rate)), "rate_max_per1000": float(np.max(rate)),
               "trend_min": float(np.min(trend)), "trend_max": float(np.max(trend)),
               "trajectory_min": float(np.min(trajectory)), "trajectory_max": float(np.max(trajectory)),
               "trend_increase_count_over_1e_6": int(np.count_nonzero(np.diff(trend) > 1e-6)),
               "anchor_abs_error": float(abs(trend[0] - anchors[local])),
               "raw_trend_min": float(np.min(raw)), "raw_trend_max": float(np.max(raw)),
               "raw_trend_increase_count_over_1e_6": int(np.count_nonzero(np.diff(raw) > 1e-6))}
        for key, column in (("trend", "cross_query_max_abs_error"),
                            ("raw_trend", "cross_query_raw_trend_max_abs_error"),
                            ("integrated_loss", "cross_query_integrated_loss_max_abs_error")):
            full = dense[key][local].detach().cpu().numpy()[sparse_query.astype(int) - 100]
            few = sparse[key][local].detach().cpu().numpy()
            row[column] = float(np.max(np.abs(full - few)))
        robust = persistence["robust_drift_mask"][local]
        row["robust_drift_supported_coordinates"] = int(np.count_nonzero(robust))
        row["robust_drift_direction_consistency_mean"] = float(np.mean(persistence["drift_direction_consistency"][local][robust])) if robust.any() else 0.0
        row["robust_drift_positive_fraction"] = float(np.mean(persistence["robust_drift_per_cycle"][local][robust] > 0)) if robust.any() else 0.0
        for layer, history in enumerate(output["diagnostics"]["history"]):
            first = history["B"][local].detach().cpu().numpy()[valid]
            last = history["Bprime"][local].detach().cpu().numpy()[valid]
            for name, value in (("B", first), ("Bprime", last), ("B_delta", last - first)):
                row[f"history{layer}_{name}_l2_mean"] = float(np.sqrt(np.square(value).sum(-1)).mean())
        rows.append(row)
    return rows


def independent_query_region_rows(uid, scope, split, queries, observed, followup, output):
    raw = output["raw_trend"]
    preclip = np.add(raw, output["observation_residual"])
    rate = output["rate_per_1000_cycles"]
    rows = []
    selections = {"all_queries_100_5000": np.arange(len(queries)),
                  "observed_future": np.flatnonzero(observed),
                  "beyond_cell_last_followup": np.flatnonzero(queries > followup)}
    for region, selected in selections.items():
        count = len(selected)
        row = {"scope": scope, "split": split, "arm": ARM, "cell_uid": uid, "query_region": region,
               "query_count": count, "query_min_cycle": float(queries[selected[0]]) if count else np.nan,
               "query_max_cycle": float(queries[selected[-1]]) if count else np.nan,
               "last_observed_valid_cycle": followup}
        conditions = {"raw_trend_below_zero": raw[selected] < 0,
                      "trend_clipped": (raw[selected] < 0) | (raw[selected] > 1.2),
                      "observation_clipped": (preclip[selected] < 0) | (preclip[selected] > 1.2)}
        for name, condition in conditions.items():
            affected = int(np.count_nonzero(condition))
            row[name + "_count"] = affected
            row[name + "_fraction"] = affected / count if count else np.nan
        for name, values in (("raw_trend", raw), ("unclamped_observation", preclip),
                             ("rate", rate)):
            first = float(np.min(values[selected])) if count else np.nan
            last = float(np.max(values[selected])) if count else np.nan
            if name == "rate":
                row["rate_min_per1000_cycles"], row["rate_max_per1000_cycles"] = first, last
                row["rate_min_per_cycle"], row["rate_max_per_cycle"] = first / 1000, last / 1000
            else:
                row[name + "_min"], row[name + "_max"] = first, last
        rows.append(row)
    return rows


def independent_query_summary(frame):
    rows = []
    minima = ("query_min_cycle", "raw_trend_min", "unclamped_observation_min", "rate_min_per1000_cycles", "rate_min_per_cycle")
    maxima = ("query_max_cycle", "raw_trend_max", "unclamped_observation_max", "rate_max_per1000_cycles", "rate_max_per_cycle")
    for (scope, region), group in frame.groupby(["scope", "query_region"], sort=True):
        count = int(np.sum(group.query_count.to_numpy()))
        row = {"scope": scope, "query_region": region, "supported_cells": int(np.count_nonzero(group.query_count.to_numpy())), "query_count": count}
        for name in minima:
            row[name] = np.nanmin(group[name].to_numpy())
        for name in maxima:
            row[name] = np.nanmax(group[name].to_numpy())
        for name in ("raw_trend_below_zero", "trend_clipped", "observation_clipped"):
            affected = group[name + "_count"].to_numpy()
            row[name + "_count"] = int(np.sum(affected))
            row[name + "_fraction"] = float(np.sum(affected) / count) if count else np.nan
            row[name + "_affected_cells"] = int(np.count_nonzero(affected))
        rows.append(row)
    return pd.DataFrame(rows)


def replay_unit(checks, unit, data, contract, device):
    inputs, identifiers, eligible, cells, query, targets, observed, rate_targets, rate_mask, splits, _ = data
    split, root = unit["split"], unit["root"]
    prefix = split + "/" + ARM
    if unit.get("verification_context"):
        prefix = unit["verification_context"] + "/" + prefix
    roles = splits["splits"][split]
    index = {uid: number for number, uid in enumerate(identifiers)}
    indices = {role: np.asarray([index[uid] for uid in roles[role] if eligible[index[uid]]], dtype=int)
               for role in ("train", "validation", "test")}
    expected_roles = {role: [identifiers[number] for number in rows] for role, rows in indices.items()}
    checks.require(prefix + "/fixed_split_and_seed", unit["contract"]["roles"] == expected_roles and unit["contract"]["seed"] == 42)
    expected_updates = 150 * math.ceil(len(indices["train"]) / 8)
    checks.require(prefix + "/fixed_settings_and_budget", unit["contract"]["training"] == contract["training"]
                   and unit["contract"]["expected_updates"] == expected_updates)
    statistics = independent.independent_scaler(inputs, indices["train"])
    scaled, _ = training.fold_scaler(inputs, indices["train"])
    saved_scaler = independent.read_json(root / "TRAIN_SCALER.json")
    checks.require(prefix + "/scaler_training_ids", saved_scaler["training_ids"] == expected_roles["train"])
    for key, item in statistics.items():
        for name, values in item.items():
            require_same(checks, prefix + "/train_only_scaler/" + key + "/" + name,
                         saved_scaler["statistics"][key][name], values, atol=1e-12, rtol=1e-12)
    history = [json.loads(line) for line in (root / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    checks.require(prefix + "/150_epochs", len(history) == 150 and [row["epoch"] for row in history] == list(range(1, 151)))
    best = math.inf
    for epoch, row in enumerate(history, 1):
        chosen = row["validation_mae"] < best
        checks.require(prefix + f"/epoch{epoch}_strict_first_min_and_updates", bool(row["selected"]) == chosen
                       and row["successful_updates"] == epoch * math.ceil(len(indices["train"]) / 8)
                       and np.isfinite(row["train_loss"]) and np.isfinite(row["validation_mae"]))
        best = min(best, row["validation_mae"])
    minimum = min(history, key=lambda row: row["validation_mae"])
    selection = unit["selection"]
    checks.require(prefix + "/selection_only_validation_first_min", selection["selected_epoch"] == minimum["epoch"]
                   and selection["validation_mae"] == minimum["validation_mae"])
    final = torch.load(root / "TRAIN_STATE.pt", map_location="cpu", weights_only=False)
    checks.require(prefix + "/recoverable_final_optimizer_budget", final["next_epoch"] == 150
                   and final["successful_updates"] == expected_updates and final["history"] == history
                   and final["contract"] == unit["contract"])
    for group_number, group in enumerate(final["optimizer"]["param_groups"]):
        expected_optimizer = {"lr": .0003, "weight_decay": .01, "betas": (.9, .999), "eps": 1e-8}
        checks.require(prefix + f"/AdamW_group{group_number}_fixed_hyperparameters",
                       all(group[key] == value for key, value in expected_optimizer.items()))
    steps = [int(item["step"].item()) for item in final["optimizer"]["state"].values() if "step" in item]
    checks.require(prefix + "/AdamW_saved_step_counts", bool(steps) and set(steps) == {expected_updates})
    selected = torch.load(root / "SELECTED.pt", map_location="cpu", weights_only=False)
    checks.require(prefix + "/atomic_selected_checkpoint", training.same_selection(selected, final["selected_checkpoint"])
                   and selected["contract"] == unit["contract"])
    checks.require(prefix + "/selected_float32_parameters", all(value.dtype == torch.float32
                   for value in selected["model"].values() if value.is_floating_point()))
    del final
    model = training.make_model(inputs, contract["training"]).to(device)
    model.load_state_dict(selected["model"])
    validation, _ = training.predictions(model, scaled, indices["validation"], query, device, 8)
    validation_error = np.where(observed[indices["validation"]], np.abs(validation - targets[indices["validation"]]), 0)
    criterion = np.mean(validation_error.sum(1) / observed[indices["validation"]].sum(1))
    require_same(checks, prefix + "/selected_validation_entrypoint_replay", criterion, selection["validation_mae"], atol=1e-7, rtol=2e-6)
    test = indices["test"]
    prediction, trend = training.predictions(model, scaled, test, query, device, 8)
    saved = pd.read_parquet(root / "test_predictions.parquet")
    expected = []
    for local, number in enumerate(test):
        for slot in np.flatnonzero(observed[number]):
            expected.append({"cell_uid": identifiers[number], "original_cycle_id": int(query[slot]),
                             "target": float(targets[number, slot]), "prediction": float(prediction[local, slot]),
                             "trend": float(trend[local, slot])})
    expected = pd.DataFrame(expected)
    independent.compare_table(checks, prefix + "/normal_test_checkpoint_replay", saved, expected,
                              ["cell_uid", "original_cycle_id"], atol=1e-7, rtol=2e-6)
    protocols = cells.protocol_uid.astype(str).to_dict()
    scope = "main_oof" if split.startswith("cell_fold_") else "unseen_protocol_sensitivity"
    metric_rows = independent.independent_metric_rows(saved, expected_roles["test"], protocols, scope, split, ARM)
    rates = pd.read_csv(root / "test_rate_predictions.csv")
    checks.require(prefix + "/every_test_cell_two_rate_windows", len(rates) == len(test) * 2
                   and not rates.duplicated(["cell_uid", "target_name"]).any()
                   and set(zip(rates.cell_uid, rates.target_name)) == {
                       (identifiers[number], name) for number in test for name in ("r200_500", "r500_1000")})
    rate_rows = []
    for row in rates.itertuples(index=False):
        number = index[row.cell_uid]
        position = list(test).index(number)
        window = 0 if row.target_name == "r200_500" else 1
        left, right = ((200, 500), (500, 1000))[window]
        endpoints = prediction[position, [left - 101, right - 101]]
        expected_rate = float((float(endpoints[0]) - float(endpoints[1])) / (right - left))
        require_same(checks, prefix + "/rate_endpoints/" + row.cell_uid + "/" + row.target_name,
                     [row.predicted_start_soh, row.predicted_end_soh, row.predicted_rate],
                     [endpoints[0], endpoints[1], expected_rate], atol=1e-9, rtol=2e-6)
        checks.require(prefix + "/rate_mask/" + row.cell_uid + "/" + row.target_name,
                       bool(row.target_supported) == bool(rate_mask[number, window]))
        truth = float(rate_targets[number, window]) if rate_mask[number, window] else np.nan
        if rate_mask[number, window]:
            require_same(checks, prefix + "/rate_target/" + row.cell_uid + "/" + row.target_name, row.target_rate, truth, atol=1e-10)
        error = abs(row.predicted_rate - truth) if rate_mask[number, window] else np.nan
        metric_rows.append({"scope": scope, "split": split, "arm": ARM, "cell_uid": row.cell_uid,
                            "protocol_uid": protocols[row.cell_uid], "metric": row.target_name + "_absolute_error",
                            "value": error, "observations": 2 if rate_mask[number, window] else 0})
        rate_rows.append({**row._asdict(), "scope": scope, "split": split, "arm": ARM,
                          "absolute_error": error, "absolute_error_per_1000_cycles": error * 1000})
    full_query = np.arange(100, 5001, dtype=np.float32)
    _, eol_trend = training.predictions(model, scaled, test, full_query, device, 8)
    saved_eol = pd.read_csv(root / "test_eol_predictions.csv")
    checks.require(prefix + "/every_test_cell_EOl", len(saved_eol) == len(test)
                   and not saved_eol.cell_uid.duplicated().any() and set(saved_eol.cell_uid) == set(expected_roles["test"]))
    expected_eol = []
    for local, number in enumerate(test):
        crossing, censored = independent.crossing_time(eol_trend[local], full_query)
        expected_eol.append({"cell_uid": identifiers[number], "predicted_trend_eol80_cycle": crossing,
                             "no_crossing_through_registered_horizon": censored, "prediction_query_max_cycle": 5000})
    independent.compare_table(checks, prefix + "/EOL5000_normal_replay", saved_eol, pd.DataFrame(expected_eol), ["cell_uid"], atol=1e-5, rtol=2e-6)
    diagnostics = {}
    mechanism_rows = []
    query_range_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test), 8):
            part = test[start:start + 8]
            batch = training.to_batch(scaled, part, device)
            output = model(**batch, query_cycles=torch.as_tensor(query, device=device), return_diagnostics=True)
            penalty = check_mechanism_math(checks, prefix + f"/mechanism_batch{start // 8}", output["diagnostics"], scaled["cycle_ids"][part], model)
            require_same(checks, prefix + f"/penalty_units_batch{start // 8}",
                         float(output["persistence_penalty"].cpu()), penalty, atol=1e-7, rtol=2e-5)
            flat = training.flatten_tensors(output["diagnostics"])
            for key, value in flat.items():
                if value.ndim and value.shape[0] == len(part):
                    diagnostics.setdefault(key, []).append(value)
            sparse_query = np.asarray([100, 101, 200, 500, 1000, 2238, 5000], dtype=np.float32)
            sparse = model(**batch, query_cycles=torch.as_tensor(sparse_query, device=device))
            dense = model(**batch, query_cycles=torch.as_tensor(full_query, device=device))
            mechanism_rows.extend(independent_mechanism_summary(
                output, dense, sparse, sparse_query, [identifiers[number] for number in part], inputs["anchor_soh100"][part]))
            dense_arrays = {key: dense[key].cpu().numpy() for key in ("raw_trend", "observation_residual", "rate_per_1000_cycles")}
            for local, number in enumerate(part):
                native_support = np.zeros(len(full_query), dtype=bool)
                native_support[1:1 + observed.shape[1]] = observed[number]
                uid = identifiers[number]
                query_range_rows.extend(independent_query_region_rows(
                    uid, scope, split, full_query, native_support, int(cells.loc[uid, "last_observed_valid_cycle"]),
                    {key: value[local] for key, value in dense_arrays.items()}))
            for key in ("trend", "raw_trend", "integrated_loss"):
                require_same(checks, prefix + f"/query_consistency_batch{start // 8}/" + key,
                             sparse[key].cpu().numpy(), dense[key][:, torch.as_tensor(sparse_query.astype(int) - 100, device=device)].cpu().numpy(),
                             atol=2e-6, rtol=2e-6)
            raw = dense["raw_trend"].cpu().numpy()
            rate = dense["rate_per_cycle"].cpu().numpy()
            checks.require(prefix + f"/far_raw_trend_monotonic_batch{start // 8}", np.isfinite(raw).all()
                           and np.isfinite(rate).all() and (rate >= 0).all() and (np.diff(raw, axis=1) <= 1e-6).all())
            require_same(checks, prefix + f"/SOH100_anchor_batch{start // 8}", raw[:, 0], inputs["anchor_soh100"][part])
            integral = np.concatenate((np.zeros((len(part), 1)), np.cumsum((rate[:, 1:].astype(float) + rate[:, :-1]) * .5, axis=1)), axis=1)
            require_same(checks, prefix + f"/independent_rate_trapezoid_units_batch{start // 8}",
                         dense["integrated_loss"].cpu().numpy(), integral, atol=2e-6, rtol=2e-5)
            require_same(checks, prefix + f"/raw_trend_anchor_minus_integral_batch{start // 8}", raw,
                         inputs["anchor_soh100"][part, None] - dense["integrated_loss"].cpu().numpy())
    with np.load(root / "test_mechanism_diagnostics.npz", allow_pickle=False) as archive:
        checks.require(prefix + "/mechanism_saved_UID_order", archive["cell_uid"].astype(str).tolist() == expected_roles["test"])
        checks.require(prefix + "/mechanism_saved_tensor_schema", set(archive.files) == set(diagnostics) | {"cell_uid"})
        for key, parts in diagnostics.items():
            require_same(checks, prefix + "/mechanism_saved_replay/" + key, archive[key], np.concatenate(parts), atol=2e-6, rtol=2e-6)
    independent.compare_table(checks, prefix + "/independent_mechanism_summary", pd.read_csv(root / "test_mechanism_summary.csv"),
                              pd.DataFrame(mechanism_rows), ["cell_uid"], atol=2e-6, rtol=2e-5)
    saved_sorted = saved.sort_values(["cell_uid", "original_cycle_id"])
    replay_sorted = expected.sort_values(["cell_uid", "original_cycle_id"])
    maximum = float(np.max(np.abs(saved_sorted.prediction.to_numpy() - replay_sorted.prediction.to_numpy())))
    observed_diagnostics = independent.observed_trend_diagnostics(saved, scope, split, ARM)
    for row in observed_diagnostics:
        row["positive_rate_integral_arm"] = True
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": metric_rows, "rates": rate_rows,
            "eol": saved_eol.assign(scope=scope, split=split, arm=ARM),
            "training": {"scope": scope, "split": split, "arm": ARM, "seed": 42,
                         "epochs": 150, "selected_epoch": selection["selected_epoch"],
                         "validation_mae": selection["validation_mae"], "successful_updates": expected_updates,
                         "parameters": selection["parameters"], "trainable_parameters": selection["trainable_parameters"],
                         "final_epoch_train_loss": history[-1]["train_loss"],
                         "selected_epoch_train_loss": history[selection["selected_epoch"] - 1]["train_loss"],
                         "elapsed_seconds": history[-1]["elapsed_seconds"],
                         **{role + "_cells": len(rows) for role, rows in indices.items()}},
            "observed_diagnostics": observed_diagnostics,
            "mechanism_rows": pd.DataFrame(mechanism_rows).assign(split=split, arm=ARM),
            "query_range_rows": pd.DataFrame(query_range_rows),
            "max_absolute_prediction_replay_error": maximum}


def nested_fingerprint(value):
    """Hash every tensor/array/scalar with type and shape in a stable tree."""
    digest = hashlib.sha256()
    tensors = 0
    def visit(item):
        nonlocal tensors
        digest.update(type(item).__name__.encode() + b"\0")
        if isinstance(item, torch.Tensor):
            tensors += 1
            digest.update(tensor_digest(item.detach().cpu().numpy()).encode())
        elif isinstance(item, np.ndarray):
            digest.update(tensor_digest(item).encode())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return {"sha256": digest.hexdigest(), "tensor_leaves": tensors}


def verify_full_reproduction(checks, output_root, contract):
    formal = output_root / "runs" / SPLITS[0] / ARM
    repeated = output_root / "training_reproduction" / SPLITS[0] / ARM
    root = output_root / "training_reproduction"
    marker = independent.read_json(root / "REPRODUCTION_COMPLETE.json")
    complete = independent.read_json(repeated / "COMPLETE.json")
    checks.require("complete_reproduction_registration_and_budget", marker["status"] == "EVALUATED"
                   and marker["additional_optimizer_updates"] == 2100
                   and marker["registration_sha256"] == training.sha256(root / "RUN_REGISTRATION.json")
                   and independent.read_json(root / "RUN_REGISTRATION.json") == contract
                   and complete["status"] == "EVALUATED" and complete["successful_updates"] == 2100)
    for entry in complete["files"]:
        checks.require("reproduction_hash/" + entry["path"], training.sha256(repeated / entry["path"]) == entry["sha256"])
    state = [torch.load(path / "TRAIN_STATE.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
    results = []
    fingerprints = {}
    def record(name, equal):
        results.append({"comparison": name, "exact_equal": bool(equal)})
    core = ("epoch", "train_loss", "validation_mae", "selected", "successful_updates")
    histories = [[{key: row[key] for key in core} for row in item["history"]] for item in state]
    record("all150_epoch_history_core", nested_fingerprint(histories[0]) == nested_fingerprint(histories[1]))
    for key in ("model", "optimizer", "random_state", "selected_checkpoint", "next_epoch", "best_mae", "successful_updates", "contract"):
        hashes = [nested_fingerprint(item[key]) for item in state]
        record("final_" + key, hashes[0] == hashes[1])
        fingerprints[key] = {"formal": hashes[0], "repeated": hashes[1]}
    selected = [torch.load(path / "SELECTED.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
    record("selected_checkpoint_all_tensors_and_metadata", nested_fingerprint(selected[0]) == nested_fingerprint(selected[1]))
    for name in ("UNIT_CONTRACT.json", "TRAIN_SCALER.json", "test_metrics.json"):
        record(name, nested_fingerprint(independent.read_json(formal / name)) == nested_fingerprint(independent.read_json(repeated / name)))
    selections = [{k: v for k, v in independent.read_json(path / "SELECTION_FROZEN.json").items() if k != "checkpoint_sha256"} for path in (formal, repeated)]
    record("selection_validation_and_epoch", nested_fingerprint(selections[0]) == nested_fingerprint(selections[1]))
    for name in ("test_predictions.parquet", "test_cell_metrics.csv", "test_rate_predictions.csv", "test_eol_predictions.csv", "test_mechanism_summary.csv"):
        frames = [pd.read_parquet(path / name) if name.endswith(".parquet") else pd.read_csv(path / name) for path in (formal, repeated)]
        record(name, frames[0].shape == frames[1].shape and frames[0].columns.equals(frames[1].columns)
               and frames[0].dtypes.equals(frames[1].dtypes) and frames[0].fillna("__MISSING__").to_dict("list") == frames[1].fillna("__MISSING__").to_dict("list"))
    with np.load(formal / "test_mechanism_diagnostics.npz", allow_pickle=False) as first, np.load(repeated / "test_mechanism_diagnostics.npz", allow_pickle=False) as second:
        record("all_saved_mechanism_tensors", sorted(first.files) == sorted(second.files) and all(tensor_digest(first[key]) == tensor_digest(second[key]) for key in first.files))
    published = independent.read_json(output_root / "analysis/TRAINING_REPRODUCTION.json")
    independent.compare_table(checks, "complete_training_reproduction_arithmetic", pd.DataFrame(published["comparison_checks"]), pd.DataFrame(results), ["comparison"], atol=0, rtol=0)
    status = "PASS" if all(item["exact_equal"] for item in results) else "FAIL"
    checks.require("complete_training_reproduction_reported_status", published["status"] == status
                   and published["epochs_per_fit"] == 150 and published["additional_optimizer_updates"] == 2100)
    divergence = next((i + 1 for i, (first, last) in enumerate(zip(*histories)) if first != last), None)
    checks.require("complete_training_reproduction_first_divergent_epoch", published["first_divergent_epoch"] == divergence)
    if status == "PASS":
        checks.require("complete_fold0_training_reproduced_exactly", True, results)
    else:
        checks.finding("complete_fold0_training_reproduction_failure", results)
    unit = {"split": SPLITS[0], "arm": ARM, "root": repeated,
            "contract": independent.read_json(repeated / "UNIT_CONTRACT.json"),
            "selection": independent.read_json(repeated / "SELECTION_FROZEN.json"),
            "verification_context": "training_reproduction"}
    return unit, {"status": status, "comparison_checks": results, "full_state_fingerprints": fingerprints,
                  "formal_optimizer_updates": 12450, "additional_reproduction_optimizer_updates": 2100}


def independent_comparisons(metrics):
    rows, pairs = [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        for metric in sorted(metrics.metric.unique()):
            scoped = metrics.loc[(metrics.scope == scope) & (metrics.metric == metric)]
            pivot = scoped.pivot(index=["cell_uid", "protocol_uid"], columns="arm", values="value")
            for reference in REFERENCES:
                paired = pivot[[reference, ARM]].reset_index().rename(columns={reference: "reference_error", ARM: "candidate_error"})
                paired["improvement"] = paired.reference_error - paired.candidate_error
                paired = paired.assign(scope=scope, metric=metric, reference=reference, candidate=ARM)
                pairs.extend(paired.to_dict("records"))
                valid = paired.loc[paired.improvement.notna()].sort_values("cell_uid")
                grouped = valid.groupby("protocol_uid", sort=True)[["reference_error", "candidate_error", "improvement"]].mean()
                for estimator, values in (("cell_mean", valid), ("protocol_macro", grouped)):
                    statistics = independent.bootstrap_average(values.improvement.to_numpy())
                    first, last = values.reference_error.mean(), values.candidate_error.mean()
                    rows.append({"scope": scope, "metric": metric, "candidate": ARM, "reference": reference,
                                 "estimator": estimator, "reference_metric": first, "candidate_metric": last,
                                 **statistics, "relative_improvement_percent": (first - last) / first * 100 if first > 0 else np.nan,
                                 "paired_cells": len(valid), "protocol_groups": len(grouped),
                                 "improved_cells": int(valid.improvement.gt(0).sum()), "worsened_cells": int(valid.improvement.lt(0).sum()),
                                 "tied_cells": int(valid.improvement.eq(0).sum()), "bootstrap_seed": 42, "bootstrap_replicates": 2000,
                                 "comparison_status": "historical_same_data_exposed_internal_single_seed"})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def independent_eol(eol, cells):
    truth = cells[["protocol_uid", "eol80_event_observed", "eol80_first_observed_crossing_cycle", "last_observed_valid_cycle"]].copy()
    truth["outcome_cycle"] = np.where(truth.eol80_event_observed, truth.eol80_first_observed_crossing_cycle, truth.last_observed_valid_cycle)
    merged = eol.merge(truth, left_on="cell_uid", right_index=True, validate="many_to_one")
    summary, contrasts = [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        frame = merged.loc[merged.scope == scope]
        risk = frame.loc[frame.outcome_cycle > 100]
        wide = risk.pivot(index="cell_uid", columns="arm", values="predicted_trend_eol80_cycle").sort_index().reindex(columns=(ARM,) + REFERENCES)
        outcomes = truth.loc[wide.index]
        early, late, scores = independent.concordance_pairs(outcomes.outcome_cycle.to_numpy(), outcomes.eol80_event_observed.to_numpy(bool), wide.to_numpy(float).T)
        values = scores.mean(1) if len(early) else np.full(4, np.nan)
        for number, arm in enumerate((ARM,) + REFERENCES):
            arm_rows = frame.loc[frame.arm == arm]
            concordant, ties = int((scores[number] == 1).sum()), int((scores[number] == .5).sum())
            summary.append({"scope": scope, "arm": arm, "harrell_c": values[number], "comparable_pairs": len(early),
                            "concordant_pairs": concordant, "prediction_ties": ties, "discordant_pairs": len(early) - concordant - ties,
                            "executed_test_cells": len(arm_rows), "at_risk100_cells": len(outcomes),
                            "observed_events": int(outcomes.eol80_event_observed.sum()), "right_censored": int((~outcomes.eol80_event_observed).sum()),
                            "predicted_no_crossing_through5000": int(arm_rows.no_crossing_through_registered_horizon.sum())})
        groups, inverse = np.unique(outcomes.protocol_uid.to_numpy(str), return_inverse=True)
        for unit, size in (("cell", len(outcomes)), ("protocol_group", len(groups))):
            rng = np.random.default_rng(42)
            multiplicities = rng.multinomial(size, np.full(size, 1 / size), size=2000)
            if unit == "protocol_group":
                multiplicities = multiplicities[:, inverse]
            draws = np.full((2000, 4), np.nan)
            for start in range(0, 2000, 64):
                weights = multiplicities[start:start + 64]
                pairs = weights[:, early] * weights[:, late]
                denominator = pairs.sum(1)
                numerator = pairs @ scores.T
                draws[start:start + 64] = np.divide(numerator, denominator[:, None], out=np.full_like(numerator, np.nan), where=denominator[:, None] > 0)
            for number, reference in enumerate(REFERENCES, 1):
                delta = draws[:, 0] - draws[:, number]
                valid = delta[np.isfinite(delta)]
                low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
                contrasts.append({"scope": scope, "candidate": ARM, "reference": reference, "sampling_unit": unit,
                                  "reference_metric": values[number], "candidate_metric": values[0],
                                  "improvement": values[0] - values[number], "ci95_low": low, "ci95_high": high,
                                  "bootstrap_supported": len(valid), "bootstrap_requested": 2000, "bootstrap_seed": 42, "paired_cells": len(outcomes)})
    return pd.DataFrame(summary), pd.DataFrame(contrasts)


def verify(args, checks):
    training.configure_execution()
    contract, units = analysis.completed_units(args.output_root, args.data)
    data = training.load_frozen_data(args.data)
    smoke = independent.read_json(args.output_root / "SMOKE_QA.json")
    checks.require("registered_engineering_smoke_PASS", smoke["status"] == "PASS"
                   and smoke["model_contract_sha256"] == training.sha256(args.output_root / "MODEL_CONTRACT.json")
                   and all(smoke["checks"].values()) and smoke["formal_optimizer_updates"] == 0
                   and smoke["test_observations_consumed"] == 0 and smoke["engineering_optimizer_updates"] == 14)
    for entry in smoke["files"]:
        checks.require("smoke_hash/" + entry["path"], training.sha256(args.output_root / entry["path"]) == entry["sha256"])
    first_batch = independent.read_json(args.output_root / "engineering_smoke/FIRST_BATCH_REPRODUCIBILITY.json")
    checks.require("fresh_seed42_first_batch_all_gradients_exact", first_batch["repeats"][0] == first_batch["repeats"][1]
                   and all(first_batch["checks"].values()) and first_batch["optimizer_updates"] == 0)
    checks.require("complete_fold0_duplicate_preregistered", contract["full_duplicate_training_planned"] is True
                   and contract["full_duplicate_training_scope"]["additional_optimizer_updates"] == 2100
                   and contract["planned_optimizer_updates"] == 12450)
    checks.require("exact_MATR169_and_original_six_splits", len(data[1]) == 169 and data[2].all()
                   and tuple(data[-2]["splits"]) == SPLITS)
    input_fingerprints = {key: tensor_digest(value) for key, value in data[0].items()}
    input_fingerprints.update({"targets": tensor_digest(data[5]), "observed": tensor_digest(data[6]),
                               "query": tensor_digest(data[4]), "rate_targets": tensor_digest(data[7]), "rate_mask": tensor_digest(data[8])})
    results = []
    for unit in units:
        print(json.dumps({"status": "REPLAYING", "split": unit["split"]}), flush=True)
        results.append(replay_unit(checks, unit, data, contract, args.device))
    repeated_unit, reproduction = verify_full_reproduction(checks, args.output_root, contract)
    reproduction_replay = replay_unit(checks, repeated_unit, data, contract, args.device)
    checks.require("six_units_150epochs_12450_optimizer_updates", len(results) == 6
                   and sum(row["training"]["successful_updates"] for row in results) == 12450)
    main = Counter(uid for name in SPLITS[:5] for uid in data[-2]["splits"][name]["test"])
    checks.require("OOF_each_physical_cell_exactly_once", main == Counter({uid: 1 for uid in data[1]}))
    # Historical loader needs source tables for support validation, never new fitting.
    cells, future, split = analysis.historical.frozen_tables(args.data)
    old, history_bindings = analysis.historical_units(args.historical_runs, cells, future, split, contract)
    metrics = pd.DataFrame([row for result in results + old for row in result["metrics"]])
    summary, protocols, folds = independent.aggregate_metrics(metrics)
    comparisons, pairs = independent_comparisons(metrics)
    eol = pd.concat([row["eol"] for row in results + old], ignore_index=True)
    eol_scores, eol_comparisons = independent_eol(eol, data[3])
    rate_rows = pd.DataFrame([row for result in results + old for row in result["rates"]])
    query_range_rows = pd.concat([row["query_range_rows"] for row in results], ignore_index=True)
    tables = {
        "cell_metrics.csv": (metrics, ["scope", "arm", "cell_uid", "metric"]),
        "summary_metrics.csv": (summary, ["scope", "arm", "metric"]),
        "protocol_metrics.csv": (protocols, ["scope", "arm", "metric", "protocol_uid"]),
        "split_metrics.csv": (folds, ["scope", "split", "arm", "metric"]),
        "historical_paired_comparisons.csv": (comparisons, ["scope", "metric", "reference", "estimator"]),
        "historical_paired_cells.csv": (pairs, ["scope", "metric", "reference", "cell_uid"]),
        "rate_predictions_and_errors.csv": (rate_rows, ["scope", "arm", "cell_uid", "target_name"]),
        "eol80_predictions.csv": (eol, ["scope", "arm", "cell_uid"]),
        "eol80_summary.csv": (eol_scores, ["scope", "arm"]),
        "eol80_historical_comparisons.csv": (eol_comparisons, ["scope", "reference", "sampling_unit"]),
        "training_units.csv": (pd.DataFrame([row["training"] for row in results]), ["split", "arm"]),
        "mechanism_summary.csv": (pd.concat([row["mechanism_rows"] for row in results], ignore_index=True), ["split", "arm", "cell_uid"]),
        "query_range_cell_diagnostics.csv": (query_range_rows, ["scope", "cell_uid", "query_region"]),
        "query_range_summary.csv": (independent_query_summary(query_range_rows), ["scope", "query_region"]),
    }
    analysis_root = args.output_root / "analysis"
    receipt = independent.read_json(analysis_root / "ANALYSIS_COMPLETE.json")
    for entry in receipt["files"]:
        checks.require("analysis_hash/" + entry["path"], training.sha256(analysis_root / entry["path"]) == entry["sha256"])
    checks.require("analysis_source_binding", receipt["analysis_source"]["sha256"] == training.sha256(REPOSITORY / receipt["analysis_source"]["path"]))
    checks.require("all_compute_counted_12450_plus2100_plus14", receipt["formal_optimizer_updates"] == 12450
                   and receipt["full_reproduction_optimizer_updates"] == 2100
                   and receipt["engineering_optimizer_updates"] == 14 and receipt["total_executed_optimizer_updates"] == 14564)
    for name, (expected, keys) in tables.items():
        independent.compare_table(checks, "independent_statistics/" + name, pd.read_csv(analysis_root / name), expected, keys, atol=2e-8, rtol=2e-5)
    diagnostics = pd.DataFrame([row for result in results for row in result["observed_diagnostics"]])
    checks.require("observed_future_trend_no_rise_over_1e_6", not diagnostics.rising_steps_over_1e_minus6_SOH.any())
    status = "VERIFIED" if reproduction["status"] == "PASS" else "VERIFIED_WITH_REPRODUCTION_FAILURE"
    return {"status": status, "completed_units": 6, "normal_selected_checkpoint_replays": 6,
            "normal_reproduction_checkpoint_replays": 1, "training_reproduction": reproduction,
            "reproduction_selected_checkpoint_replay_max_abs_error": reproduction_replay["max_absolute_prediction_replay_error"],
            "successful_updates": 12450, "seed": 42,
            "formal_optimizer_updates": 12450, "full_reproduction_optimizer_updates": 2100,
            "engineering_optimizer_updates": 14, "total_executed_optimizer_updates": 14564,
            "model_contract_sha256": training.sha256(args.output_root / "MODEL_CONTRACT.json"),
            "replay_device": str(args.device), "torch_version": torch.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "verification_dependencies": [
                {"path": name, "sha256": training.sha256(REPOSITORY / name)} for name in (
                    "scripts/verify_unified_seed42_results.py", "scripts/analyze_methodology_full_seed42.py",
                    "scripts/unified_matr/analyze_results.py")],
            "data_freeze_sha256": training.sha256(args.data / "DATA_FREEZE.json"),
            "input_label_tensor_fingerprints": input_fingerprints,
            "historical_reference_bindings": history_bindings,
            "max_absolute_prediction_replay_error": max(row["max_absolute_prediction_replay_error"] for row in results),
            "observed_trend_diagnostics": diagnostics.to_dict("records"),
            "limitations": ["Saved selected checkpoints were replayed through the registered normal model entry point.",
                            "Independent arithmetic verifies statistics. The architecture is reused for checkpoint replay.",
                            "150 logged epochs, final optimizer state and saved selected checkpoints are checked. Unsaved epoch weights are unavailable.",
                            "One additional complete fold0 training was repeated from seed42. This checks one registered split/environment and does not measure cross-seed variability.",
                            "The first M4 engineering first-batch check had exact predictions and loss with25 parameter-gradient mismatches. Failed code/contract/receipts and subsequent engineering changes/re-registration are preserved in the failure history.",
                            "Historical169-cell exposure and single-seed variability limitations remain.",
                            "The earlier seven-arm M2/NR complete-training exact reproduction remains0/6 with unresolved cause."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical-runs", type=Path, default=REPOSITORY / "results/unified_matr_seed42/runs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output_root / "RUN_VERIFICATION.json"
    if output.exists():
        previous = independent.read_json(output)
        if previous.get("status", "").startswith("VERIFIED"):
            raise RuntimeError("The completed passing verification receipt is immutable")
        history = args.output_root / "verification_attempts"
        history.mkdir(parents=True, exist_ok=True)
        preserved = history / ("RUN_VERIFICATION_" + training.sha256(output) + ".json")
        if not preserved.exists():
            shutil.copy2(output, preserved)
    checks = independent.Checks()
    try:
        result = verify(args, checks)
    except Exception as exc:
        result = {"status": "FAIL", "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    result["checks"] = checks.rows
    result["counts"] = dict(Counter(row["status"] for row in checks.rows))
    result["verifier_source_sha256"] = training.sha256(__file__)
    training.write_json(output, independent.clean(result))
    print(json.dumps({"status": result["status"], "counts": result["counts"]}), flush=True)
    if not result["status"].startswith("VERIFIED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
