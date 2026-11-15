"""Independent arithmetic and normal-entry replay for24 ablation fits and4 repeats."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from scipy.special import erf
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts.unified_matr import train_methodology_ablations as training
from scripts import analyze_methodology_ablations_seed42 as analysis
from scripts import verify_methodology_full_seed42 as full
from scripts import verify_unified_seed42_results as independent
from models.unified_matr.model import state_hash

ARMS, SPLITS = training.ARMS, training.SPLITS
ARM = "M4_FULL_METHOD"
REFERENCES = ARMS
require_same = full.require_same
independent_mechanism_summary = full.independent_mechanism_summary
independent_query_region_rows = full.independent_query_region_rows
nested_fingerprint = full.nested_fingerprint


def frame_fingerprint(frame):
    columns = {}
    for name in frame.columns:
        values = frame[name].to_numpy()
        columns[name] = values.tolist() if values.dtype.hasobject else values
    return nested_fingerprint({"names": frame.columns.tolist(), "dtypes": [str(value) for value in frame.dtypes], "columns": columns})


def strict_comparator_controls(checks):
    nan1 = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    nan2 = struct.unpack(">d", bytes.fromhex("7ff8000000000002"))[0]
    cases = (
        ("float_signed_zero", 0., -0., False),
        ("tensor_signed_zero", torch.tensor([0.]), torch.tensor([-0.]), False),
        ("array_signed_zero", np.array([0.]), np.array([-0.]), False),
        ("numpy_scalar_signed_zero", np.float64(0), np.float64(-0.), False),
        ("tree_structure", [[], []], [[[]]], False),
        ("list_tuple", [1], (1,), False),
        ("typed_mapping_key", {1: "a"}, {"1": "a"}, False),
        ("signed_zero_mapping_key", {0.: "a"}, {-0.: "a"}, False),
        ("same_nan_payload", nan1, nan1, True),
        ("different_nan_payload", nan1, nan2, False),
        ("same_tensor_nan", torch.tensor([nan1], dtype=torch.float64), torch.tensor([nan1], dtype=torch.float64), True),
        ("mapping_order", {"a": 1, "b": (2, None)}, {"b": (2, None), "a": 1}, True),
    )
    for name, first, last, expected in cases:
        direct = analysis.strict_nested_equal(first, last)
        hashed = nested_fingerprint(first) == nested_fingerprint(last)
        checks.require("strict_comparison_boundary/" + name, direct == hashed == expected)
    for name, first, last, expected in (
            ("frame_signed_zero", pd.DataFrame({"x": [0.]}), pd.DataFrame({"x": [-0.]}), False),
            ("frame_same_nan", pd.DataFrame({"x": [nan1]}), pd.DataFrame({"x": [nan1]}), True)):
        checks.require("strict_comparison_boundary/" + name,
                       analysis.strict_frame_equal(first, last) == (frame_fingerprint(first) == frame_fingerprint(last)) == expected)


def legacy_flow_oracle(checks, prefix, model, diagnostic, query, output, anchors):
    """Evaluate the analytic decoder and its signed derivative in NumPy float64."""
    head = model.flow_head
    state = diagnostic["state"].detach().cpu().numpy().astype(float)
    condition = diagnostic["condition"].detach().cpu().numpy().astype(float)
    def affine(value, layer):
        result = value @ layer.weight.detach().cpu().numpy().astype(float).T
        return result + layer.bias.detach().cpu().numpy().astype(float) if layer.bias is not None else result
    def gelu(value):
        return .5 * value * (1 + erf(value / math.sqrt(2)))
    generated = affine(gelu(affine(condition, head.generator[0])), head.generator[2])
    rate = np.logaddexp(0, generated[:, :16]) + 1e-4
    mean = np.tanh(generated[:, 16:])
    tau = (np.asarray(query, dtype=float) - 100) / 1000
    decay = np.exp(-rate[:, None] * tau[None, :, None])
    future = decay * state[:, None] - np.expm1(-rate[:, None] * tau[None, :, None]) * mean[:, None]
    before = affine(future, head.decoder[0])
    decoded = affine(gelu(before), head.decoder[2]).squeeze(-1)
    baseline = affine(gelu(affine(state, head.decoder[0])), head.decoder[2]).squeeze(-1)
    correction = .2 * (decoded - baseline[:, None])
    correction[:, tau == 0] = 0
    tangent = -rate[:, None] * decay * (state[:, None] - mean[:, None])
    projected = tangent @ head.decoder[0].weight.detach().cpu().numpy().astype(float).T
    derivative = .5 * (1 + erf(before / math.sqrt(2))) + before * np.exp(-.5 * before ** 2) / math.sqrt(2 * math.pi)
    signed_rate = -.2 * (derivative * projected) @ head.decoder[2].weight.detach().cpu().numpy().astype(float).T
    require_same(checks, prefix + "/legacy_decoder_float64_oracle", output["raw_trend"].cpu().numpy(), anchors[:, None] + correction, atol=2e-6, rtol=2e-5)
    require_same(checks, prefix + "/legacy_signed_effective_loss", output["integrated_loss"].cpu().numpy(), -correction, atol=2e-6, rtol=2e-5)
    require_same(checks, prefix + "/legacy_signed_derivative_float64_oracle", output["rate_per_1000_cycles"].cpu().numpy(), signed_rate.squeeze(-1), atol=2e-5, rtol=2e-4)


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
        if not hasattr(model.history[layer].mixer, "write_projection"):
            checks.require(prefix + f"/A2_layer{layer}_Bprime_identity", np.array_equal(original, modified) and np.all(alteration == 0))
            continue
        weights = model.history[layer].mixer.write_projection.weight.detach().cpu().numpy()
        projected = state @ weights.T
        expected_modulation = np.where(supported[..., None], gate * np.tanh(projected), 0)
        require_same(checks, prefix + f"/B_layer{layer}_write_formula", alteration, expected_modulation, atol=1e-6, rtol=1e-5)
        require_same(checks, prefix + f"/B_layer{layer}_actual_modulation", modified, original + alteration)
        checks.require(prefix + f"/B_layer{layer}_invalid_no_modulation", np.all(alteration[~supported] == 0))
    return float(np.square(expected_drift).sum() / (max(transition_count, 1) * state.shape[-1]))



def replay_unit(checks, unit, data, contract, device):
    inputs, identifiers, eligible, cells, query, targets, observed, rate_targets, rate_mask, splits, _ = data
    ARM = unit["arm"]
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
    model = training.make_model(ARM, inputs, contract["training"]).to(device)
    checks.require(prefix + "/registered_fresh_seed42_initialization", state_hash(model) == contract["initialization"]["models"][ARM]["initial_state_sha256"])
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
            if ARM == ARMS[2]:
                rho = output["diagnostics"]["persistence"]["rho"].cpu().numpy()
                checks.require(prefix + f"/static_channel_rho_batch{start // 8}", np.array_equal(rho, np.broadcast_to(rho[:1, :1], rho.shape)))
            if ARM == ARMS[3]:
                quality = output["diagnostics"]["persistence"]["quality"]
                checks.require(prefix + f"/A4_no_local_quality_batch{start // 8}", bool((quality == 0).all()))
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
            checks.require(prefix + f"/far_query_finite_batch{start // 8}", np.isfinite(raw).all() and np.isfinite(rate).all())
            require_same(checks, prefix + f"/SOH100_anchor_batch{start // 8}", raw[:, 0], inputs["anchor_soh100"][part])
            if ARM == ARMS[0]:
                legacy_flow_oracle(checks, prefix + f"/batch{start // 8}", model, output["diagnostics"], sparse_query,
                                   sparse, inputs["anchor_soh100"][part])
            else:
                checks.require(prefix + f"/nonnegative_rate_monotonic_batch{start // 8}", (rate >= 0).all() and (np.diff(raw, axis=1) <= 1e-6).all())
                integral = np.concatenate((np.zeros((len(part), 1)), np.cumsum((rate[:, 1:].astype(float) + rate[:, :-1]) * .5, axis=1)), axis=1)
                require_same(checks, prefix + f"/independent_rate_trapezoid_units_batch{start // 8}",
                             dense["integrated_loss"].cpu().numpy(), integral, atol=2e-6, rtol=2e-5)
            require_same(checks, prefix + f"/raw_trend_anchor_minus_effective_loss_batch{start // 8}", raw,
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
        row["positive_rate_integral_arm"] = ARM != ARMS[0]
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
            "query_range_rows": pd.DataFrame(query_range_rows).assign(arm=ARM),
            "max_absolute_prediction_replay_error": maximum}



def independent_comparisons(metrics):
    rows, pairs = [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        for metric in sorted(metrics.metric.unique()):
            scoped = metrics.loc[(metrics.scope == scope) & (metrics.metric == metric)]
            pivot = scoped.pivot(index=["cell_uid", "protocol_uid"], columns="arm", values="value")
            for reference in REFERENCES:
                paired = pivot[[reference, ARM]].reset_index().rename(columns={reference: "reference_error", ARM: "candidate_error"})
                paired["improvement"] = paired.reference_error - paired.candidate_error
                paired = paired.assign(scope=scope, metric=metric, reference=reference, candidate=ARM, contrast=ARM + "_vs_" + reference)
                pairs.extend(paired.to_dict("records"))
                valid = paired.loc[paired.improvement.notna()].sort_values("cell_uid")
                grouped = valid.groupby("protocol_uid", sort=True)[["reference_error", "candidate_error", "improvement"]].mean()
                for estimator, values in (("cell_mean", valid), ("protocol_macro", grouped)):
                    statistics = independent.bootstrap_average(values.improvement.to_numpy())
                    first, last = values.reference_error.mean(), values.candidate_error.mean()
                    rows.append({"scope": scope, "metric": metric, "candidate": ARM, "reference": reference,
                                 "estimator": estimator, "reference_metric": first, "candidate_metric": last,
                                 **statistics, "relative_full_improvement_percent": (first - last) / first * 100 if first > 0 else np.nan,
                                 "paired_cells": len(valid), "protocol_groups": len(grouped),
                                 "full_better_cells": int(valid.improvement.gt(0).sum()), "ablation_better_cells": int(valid.improvement.lt(0).sum()),
                                 "tied_cells": int(valid.improvement.eq(0).sum()), "bootstrap_seed": 42, "bootstrap_replicates": 2000,
                                 "direction": "ablation_error_minus_full_error_positive_favors_full",
                                 "contrast": ARM + "_vs_" + reference})
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
        values = scores.mean(1) if len(early) else np.full(len(REFERENCES) + 1, np.nan)
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
            draws = np.full((2000, len(REFERENCES) + 1), np.nan)
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
                                  "bootstrap_supported": len(valid), "bootstrap_requested": 2000, "bootstrap_seed": 42, "paired_cells": len(outcomes), "direction": "full_C_minus_ablation_C_positive_favors_full"})
    return pd.DataFrame(summary), pd.DataFrame(contrasts)


def verify_reproductions(checks, output_root, contract):
    root = output_root / "training_reproduction"
    marker = independent.read_json(root / "REPRODUCTION_COMPLETE.json")
    checks.require("four_complete_reproductions_registered", marker["status"] == "EVALUATED"
                   and marker["additional_optimizer_updates"] == 8400 and tuple(marker["arms"]) == ARMS
                   and independent.read_json(root / "RUN_REGISTRATION.json") == contract
                   and marker["registration_sha256"] == training.sha256(root / "RUN_REGISTRATION.json"))
    published = {item["arm"]: item for item in independent.read_json(output_root / "analysis/TRAINING_REPRODUCTIONS.json")["arms"]}
    units, summaries = [], []
    for arm in ARMS:
        formal, repeated = output_root / "runs" / SPLITS[0] / arm, root / SPLITS[0] / arm
        complete = independent.read_json(repeated / "COMPLETE.json")
        checks.require(arm + "/repeat_complete_budget", complete["status"] == "EVALUATED" and complete["successful_updates"] == 2100)
        for item in complete["files"]:
            checks.require(arm + "/repeat_hash/" + item["path"], training.sha256(repeated / item["path"]) == item["sha256"])
        states = [torch.load(path / "TRAIN_STATE.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
        rows, fingerprints = [], {}
        def record(name, first, last):
            values = [nested_fingerprint(first), nested_fingerprint(last)]
            rows.append({"comparison": name, "exact_equal": values[0] == values[1]})
            fingerprints[name] = {"formal": values[0], "repeated": values[1]}
        core = ("epoch", "train_loss", "validation_mae", "selected", "successful_updates")
        histories = [[{key: row[key] for key in core} for row in state["history"]] for state in states]
        record("all150_epoch_history_core", *histories)
        for key in ("model", "optimizer", "random_state", "selected_checkpoint", "next_epoch", "best_mae", "successful_updates", "contract"):
            record("final_" + key, states[0][key], states[1][key])
        selected = [torch.load(path / "SELECTED.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
        record("selected_checkpoint_all_tensors_and_metadata", *selected)
        for name in ("UNIT_CONTRACT.json", "TRAIN_SCALER.json", "test_metrics.json"):
            record(name, independent.read_json(formal / name), independent.read_json(repeated / name))
        selections = [{key: value for key, value in independent.read_json(path / "SELECTION_FROZEN.json").items() if key != "checkpoint_sha256"} for path in (formal, repeated)]
        record("selection_validation_and_epoch", *selections)
        for name in ("test_predictions.parquet", "test_cell_metrics.csv", "test_rate_predictions.csv", "test_eol_predictions.csv", "test_mechanism_summary.csv"):
            tables = [pd.read_parquet(path / name) if name.endswith("parquet") else pd.read_csv(path / name) for path in (formal, repeated)]
            equal = frame_fingerprint(tables[0]) == frame_fingerprint(tables[1])
            rows.append({"comparison": name, "exact_equal": equal})
        with np.load(formal / "test_mechanism_diagnostics.npz", allow_pickle=False) as first, np.load(repeated / "test_mechanism_diagnostics.npz", allow_pickle=False) as second:
            equal = sorted(first.files) == sorted(second.files) and all(full.tensor_digest(first[key]) == full.tensor_digest(second[key]) for key in first.files)
            rows.append({"comparison": "all_saved_mechanism_tensors", "exact_equal": equal})
        independent.compare_table(checks, arm + "/reproduction_twenty_comparisons", pd.DataFrame(published[arm]["comparison_checks"]), pd.DataFrame(rows), ["comparison"], atol=0, rtol=0)
        status = "PASS" if all(row["exact_equal"] for row in rows) else "FAIL"
        divergence = next((i + 1 for i, (a, b) in enumerate(zip(*histories)) if nested_fingerprint(a) != nested_fingerprint(b)), None)
        checks.require(arm + "/reproduction_status_and_epoch", published[arm]["status"] == status and published[arm]["first_divergent_epoch"] == divergence)
        if status == "PASS":
            checks.require(arm + "/complete_training_exact_repeat", True)
        else:
            checks.finding(arm + "/complete_training_repeat_failed", rows)
        units.append({"arm": arm, "split": SPLITS[0], "root": repeated,
                      "contract": independent.read_json(repeated / "UNIT_CONTRACT.json"),
                      "selection": independent.read_json(repeated / "SELECTION_FROZEN.json"),
                      "verification_context": "full_training_reproduction"})
        summaries.append({"arm": arm, "status": status, "comparisons": rows, "state_fingerprints": fingerprints})
    return units, summaries


def independent_monotonic_rows(unit):
    saved = pd.read_parquet(unit["root"] / "test_predictions.parquet")
    rows = []
    with np.load(unit["root"] / "test_mechanism_diagnostics.npz", allow_pickle=False) as archive:
        lookup = {uid: i for i, uid in enumerate(archive["cell_uid"].astype(str))}
        for uid in sorted(saved.cell_uid.unique()):
            frame = saved.loc[saved.cell_uid == uid].sort_values("original_cycle_id")
            positions = frame.original_cycle_id.to_numpy(int) - 101
            number = lookup[uid]
            row = {"arm": unit["arm"], "split": unit["split"], "cell_uid": uid,
                   "scope": "main_oof" if unit["split"].startswith("cell_fold_") else "unseen_protocol_sensitivity",
                   "observed_queries": len(frame), "observed_adjacent_pairs": len(frame) - 1,
                   "negative_signed_rate_points": int(np.count_nonzero(archive["flow__rate_per_1000_cycles"][number, positions] < 0)),
                   "nonnegative_rate_structure": unit["arm"] != ARMS[0]}
            for name, values in (("raw_trend", archive["flow__raw_trend"][number, positions]),
                                 ("trend", frame.trend.to_numpy()), ("prediction", frame.prediction.to_numpy())):
                change = values[1:] - values[:-1]
                row[name + "_rise_count"] = int(np.count_nonzero(change > 0))
                row[name + "_rise_over1e6_count"] = int(np.count_nonzero(change > 1e-6))
                row[name + "_largest_positive_step"] = float(max(np.max(change, initial=0), 0))
            rows.append(row)
    return rows


def verify(args, checks):
    training.configure_execution()
    full.fingerprint_controls(checks)
    strict_comparator_controls(checks)
    contract, units = analysis.completed_units(args.output_root, args.data)
    data = training.load_frozen_data(args.data)
    checks.require("frozen169_original_six_splits", len(data[1]) == 169 and data[2].all() and tuple(data[-2]["splits"]) == SPLITS)
    config = independent.read_json(REPOSITORY / "configs/unified_matr_ablations_seed42.json")
    checks.require("registered_config_byte_hash", contract["config_sha256"] == training.sha256(REPOSITORY / "configs/unified_matr_ablations_seed42.json") and contract["config"] == config)
    checks.require("formal_arm_first_order_registered", contract["execution_order"] == [{"arm": arm, "split": split} for arm in ARMS for split in SPLITS])
    smoke = independent.read_json(args.output_root / "SMOKE_QA.json")
    checks.require("four_engineering_smokes_budget", smoke["status"] == "PASS" and smoke["engineering_optimizer_updates"] == 56
                   and smoke["formal_optimizer_updates"] == 0 and smoke["test_observations_consumed"] == 0
                   and smoke["model_contract_sha256"] == training.sha256(args.output_root / "MODEL_CONTRACT.json")
                   and smoke["training_source_hashes_verified_before_formal_launch"] is True)
    controls = smoke["independent_engineering_controls"]
    checks.require("independent_engineering_CPU_CUDA_receipts_present", {item["path"] for item in controls} == {
        "INDEPENDENT_ABLATION_QA_CPU_FINAL.json", "INDEPENDENT_ABLATION_QA_CUDA_FINAL.json",
        "engineering/MODEL_ENGINEERING_CPU.json", "engineering/MODEL_ENGINEERING_CUDA.json"})
    for item in controls:
        path = args.output_root / item["path"]
        receipt = independent.read_json(path)
        checks.require("independent_engineering/" + item["path"], training.sha256(path) == item["sha256"]
                       and receipt["status"] == "PASS" and receipt["optimizer_updates"] == 0
                       and receipt.get("test_observations_consumed", 0) == 0)
        if "contract_sha256" in receipt:
            checks.require("independent_engineering_contract/" + item["path"], receipt["contract_sha256"] == contract["ablation_contract"]["sha256"])
    arm_smokes = {item["arm"]: item["sha256"] for item in smoke["arm_smoke_receipts"]}
    for arm in ARMS:
        root = args.output_root / "engineering_smoke" / arm
        receipt = independent.read_json(root / "SMOKE_QA.json")
        checks.require(arm + "/smoke_aggregate_binding", arm_smokes.get(arm) == training.sha256(root / "SMOKE_QA.json"))
        checks.require(arm + "/engineering_smoke_PASS", receipt["status"] == "PASS" and all(receipt["checks"].values())
                       and receipt["engineering_optimizer_updates"] == 14 and receipt["model_contract_sha256"] == training.sha256(args.output_root / "MODEL_CONTRACT.json"))
        for item in receipt["files"]:
            checks.require(arm + "/smoke_hash/" + item["path"], training.sha256(args.output_root / item["path"]) == item["sha256"])
        first = independent.read_json(root / "FIRST_BATCH_REPRODUCIBILITY.json")
        checks.require(arm + "/fresh_first_batch_gradient_exact", first["repeats"][0] == first["repeats"][1] and first["optimizer_updates"] == 0)
    input_hashes = {key: full.tensor_digest(value) for key, value in data[0].items()}
    input_hashes.update({name: full.tensor_digest(data[position]) for name, position in (("query", 4), ("targets", 5), ("observed", 6), ("rate_targets", 7), ("rate_mask", 8))})
    results = []
    for unit in units:
        print(json.dumps({"status": "REPLAYING", "arm": unit["arm"], "split": unit["split"]}), flush=True)
        results.append(replay_unit(checks, unit, data, contract, args.device))
    repeat_units, reproductions = verify_reproductions(checks, args.output_root, contract)
    repeat_replays = [replay_unit(checks, unit, data, contract, args.device) for unit in repeat_units]
    checks.require("complete24_formal_updates49800", len(results) == 24 and sum(row["training"]["successful_updates"] for row in results) == 49800)
    checks.require("complete4_repeat_updates8400", sum(row["training"]["successful_updates"] for row in repeat_replays) == 8400)
    counts = Counter(uid for split in SPLITS[:5] for uid in data[-2]["splits"][split]["test"])
    checks.require("each_arm_full_OOF169_once", counts == Counter({uid: 1 for uid in data[1]}))
    cells, future, split = analysis.statistics.frozen_tables(args.data)
    reference, reference_root = analysis.load_m4_reference(contract, args.data, cells, future, split)
    combined = results + reference
    metrics = pd.DataFrame([row for result in combined for row in result["metrics"]])
    summary, protocols, folds = independent.aggregate_metrics(metrics)
    comparisons, pairs = independent_comparisons(metrics)
    eol = pd.concat([row["eol"] for row in combined], ignore_index=True)
    eol_summary, eol_comparisons = independent_eol(eol, data[3])
    query_rows = pd.concat([row["query_range_rows"] for row in results] + [pd.read_csv(reference_root / "analysis/query_range_cell_diagnostics.csv")], ignore_index=True)
    query_summary = pd.concat([full.independent_query_summary(group).assign(arm=arm) for arm, group in query_rows.groupby("arm", sort=True)], ignore_index=True)
    monotonic = pd.DataFrame([row for unit in units for row in independent_monotonic_rows(unit)])
    constrained = monotonic.loc[monotonic.arm != ARMS[0]]
    checks.require("A2_A3_A4_observed_raw_trend_nonincreasing", not constrained.raw_trend_rise_over1e6_count.any())
    known = metrics.loc[(metrics.cell_uid == "MATR_b1c2") & metrics.metric.isin(("trajectory_mae", "trajectory_rmse"))]
    known = known.pivot(index=["scope", "split", "arm", "cell_uid"], columns="metric", values="value").reset_index()
    known = known.merge(query_rows.loc[(query_rows.cell_uid == "MATR_b1c2") & (query_rows.query_region == "observed_future")], on=["scope", "split", "arm", "cell_uid"], validate="one_to_one")
    tables = {
        "cell_metrics.csv": (metrics, ["scope", "arm", "cell_uid", "metric"]),
        "summary_metrics.csv": (summary, ["scope", "arm", "metric"]),
        "protocol_metrics.csv": (protocols, ["scope", "arm", "metric", "protocol_uid"]),
        "split_metrics.csv": (folds, ["scope", "split", "arm", "metric"]),
        "full_vs_ablation_paired_comparisons.csv": (comparisons, ["scope", "metric", "reference", "estimator"]),
        "full_vs_ablation_paired_cells.csv": (pairs, ["scope", "metric", "reference", "cell_uid"]),
        "rate_predictions_and_errors.csv": (pd.DataFrame([row for result in combined for row in result["rates"]]), ["scope", "arm", "cell_uid", "target_name"]),
        "eol80_predictions.csv": (eol, ["scope", "arm", "cell_uid"]),
        "eol80_summary.csv": (eol_summary, ["scope", "arm"]),
        "eol80_paired_comparisons.csv": (eol_comparisons, ["scope", "reference", "sampling_unit"]),
        "training_units.csv": (pd.DataFrame([row["training"] for row in results]), ["arm", "split"]),
        "mechanism_summary.csv": (pd.concat([row["mechanism_rows"] for row in results], ignore_index=True), ["arm", "split", "cell_uid"]),
        "observed_monotonic_diagnostics.csv": (monotonic, ["arm", "split", "cell_uid"]),
        "query_range_cell_diagnostics.csv": (query_rows, ["scope", "arm", "cell_uid", "query_region"]),
        "query_range_summary.csv": (query_summary, ["scope", "arm", "query_region"]),
        "known_counterexample_MATR_b1c2.csv": (known, ["scope", "arm", "cell_uid"]),
    }
    root = args.output_root / "analysis"
    receipt = independent.read_json(root / "ANALYSIS_COMPLETE.json")
    checks.require("analysis_source_fingerprint", receipt["analysis_source_sha256"] == training.sha256(REPOSITORY / "scripts/analyze_methodology_ablations_seed42.py"))
    checks.require("all_compute58256_accounted", receipt["formal_optimizer_updates"] == 49800
                   and receipt["full_reproduction_optimizer_updates"] == 8400 and receipt["engineering_optimizer_updates"] == 56
                   and receipt["total_executed_optimizer_updates"] == 58256)
    for item in receipt["files"]:
        checks.require("analysis_hash/" + item["path"], training.sha256(root / item["path"]) == item["sha256"])
    for name, (expected, keys) in tables.items():
        independent.compare_table(checks, "independent_statistics/" + name, pd.read_csv(root / name), expected, keys, atol=2e-8, rtol=2e-5)
    return {"status": "VERIFIED" if all(row["status"] == "PASS" for row in reproductions) else "VERIFIED_WITH_REPRODUCTION_FAILURE",
            "arms": list(ARMS), "seed": 42, "formal_units": 24, "normal_formal_checkpoint_replays": 24,
            "normal_reproduction_checkpoint_replays": 4, "reproductions": reproductions,
            "formal_optimizer_updates": 49800, "full_reproduction_optimizer_updates": 8400,
            "engineering_optimizer_updates": 56, "total_executed_optimizer_updates": 58256,
            "max_absolute_formal_prediction_replay_error": max(row["max_absolute_prediction_replay_error"] for row in results),
            "max_absolute_repeat_prediction_replay_error": max(row["max_absolute_prediction_replay_error"] for row in repeat_replays),
            "input_label_fingerprints": input_hashes, "model_contract_sha256": training.sha256(args.output_root / "MODEL_CONTRACT.json"),
            "A1_monotonic_diagnostics": monotonic.loc[monotonic.arm == ARMS[0]].to_dict("records"),
            "limitations": ["The169-cell cohort and saved M4 comparison were already exposed. This is internal exploratory single-seed evidence.",
                            "Paired bootstrap conditions on fitted predictions and does not measure training-seed variability.",
                            "Each complete repeat covers fold0 at seed42 under one registered environment.",
                            "150 history rows, final optimizer states and saved selected weights were checked. Unsaved epoch weights are unavailable.",
                            "Normal-entry replay reuses the registered architecture. Statistical arithmetic and selected operator checks are independent.",
                            "A1 signed decoder loss is an interface compatibility field and its rate is an analytic derivative, with rises permitted.",
                            "Queries beyond each cell followup lack same-range physical calibration."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.data, args.output_root = args.data.resolve(), args.output_root.resolve()
    path = args.output_root / "RUN_VERIFICATION.json"
    if path.exists():
        old = independent.read_json(path)
        if old.get("status", "").startswith("VERIFIED"):
            raise RuntimeError("The passing verification receipt is immutable")
        history = args.output_root / "verification_attempts"
        history.mkdir(parents=True, exist_ok=True)
        preserved = history / ("RUN_VERIFICATION_" + training.sha256(path) + ".json")
        if not preserved.exists():
            shutil.copy2(path, preserved)
    checks = independent.Checks()
    try:
        result = verify(args, checks)
    except Exception as error:
        result = {"status": "FAIL", "exception": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    result["checks"] = checks.rows
    result["counts"] = dict(Counter(row["status"] for row in checks.rows))
    result["verifier_source_sha256"] = training.sha256(__file__)
    result["verification_dependencies"] = [{"path": name, "sha256": training.sha256(REPOSITORY / name)} for name in (
        "scripts/verify_methodology_full_seed42.py", "scripts/verify_unified_seed42_results.py",
        "scripts/analyze_methodology_ablations_seed42.py", "scripts/analyze_methodology_full_seed42.py",
        "scripts/unified_matr/analyze_results.py")]
    training.write_json(path, independent.clean(result))
    print(json.dumps({"status": result["status"], "counts": result["counts"]}), flush=True)
    if not result["status"].startswith("VERIFIED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

