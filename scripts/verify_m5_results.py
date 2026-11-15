"""Independent replay and arithmetic verification of all selected M5 fits.

The full-entrypoint replay is adapted from the sealed M4 verifier. M5 uses an
independent relative-intensity integral and exponential trend oracle.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from scipy.special import erf, expit
from threadpoolctl import threadpool_limits

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts.unified_matr import train_methodology_full as training
from scripts.unified_matr import train_m5
from scripts import verify_unified_seed42_results as independent
from scripts import analyze_m5_results as analysis
from scripts.verify_methodology_full_seed42 import (
    require_same, check_mechanism_math, independent_mechanism_summary,
    independent_query_region_rows as m4_query_rows, independent_query_summary,
    nested_fingerprint, fingerprint_controls, tensor_digest)
from scripts.verify_m5_engineering import numpy_head

ARM = train_m5.ARM
SPLITS = training.SPLITS
REFERENCES = analysis.REFERENCES


def independent_query_region_rows(*args):
    rows = m4_query_rows(*args)
    for row in rows:
        row['arm'] = ARM
    return rows


def frame_fingerprint(frame):
    values=[]
    for name in frame.columns:
        column=frame[name].to_numpy()
        values.append((name,str(frame[name].dtype),column.tolist() if column.dtype.hasobject else column))
    return nested_fingerprint((frame.shape,values))



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
    model = train_m5.build_model(inputs, contract["training"]).to(device)
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
    soft_rows = []
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
            dense = model(**batch, query_cycles=torch.as_tensor(full_query, device=device), return_diagnostics=True)
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
            checks.require(prefix + f"/nonnegative_raw_trend_batch{start // 8}", bool((raw >= 0).all()))
            flow = dense['diagnostics']['flow']
            h, lam = numpy_head(model.flow_head,
                                dense['diagnostics']['state'].cpu().numpy().astype(float),
                                dense['diagnostics']['condition'].cpu().numpy().astype(float),
                                np.broadcast_to(full_query.astype(float),(len(part),len(full_query))))
            expected_raw = inputs['anchor_soh100'][part,None].astype(float)*np.exp(-h)
            require_same(checks,prefix+f'/independent_relative_integral_batch{start // 8}',
                         flow['relative_hazard_integral'].cpu().numpy(),h,atol=2e-6,rtol=2e-5)
            require_same(checks,prefix+f'/independent_bounded_intensity_batch{start // 8}',
                         flow['relative_hazard_per_1000'].cpu().numpy(),lam,atol=2e-6,rtol=2e-5)
            require_same(checks,prefix+f'/independent_exponential_trend_batch{start // 8}',raw,expected_raw,atol=2e-6,rtol=2e-5)
            require_same(checks,prefix+f'/absolute_rate_units_batch{start // 8}',
                         dense['rate_per_1000_cycles'].cpu().numpy(),expected_raw*lam,atol=2e-6,rtol=2e-5)
            require_same(checks,prefix+f'/per_cycle_units_batch{start // 8}',rate,expected_raw*lam/1000,atol=2e-9,rtol=2e-5)
            require_same(checks,prefix+f'/absolute_capacity_loss_batch{start // 8}',
                         dense['integrated_loss'].cpu().numpy(),inputs['anchor_soh100'][part,None]-expected_raw,atol=2e-6,rtol=2e-5)
            require_same(checks,prefix+f'/observation_bound_and_clamp_batch{start // 8}',
                         dense['trajectory'].cpu().numpy(),np.clip(raw+dense['observation_residual'].cpu().numpy(),0,1.2),atol=0,rtol=0)
            checks.require(prefix+f'/relative_intensity_and_observation_bounds_batch{start // 8}',
                           bool(((lam>0)&(lam<2)).all()) and bool((np.abs(dense['observation_residual'].cpu().numpy())<=.03).all()))
            first, last = model.flow_head.net[0], model.flow_head.net[2]
            context = np.concatenate((dense['diagnostics']['state'].cpu().numpy(),
                                      dense['diagnostics']['condition'].cpu().numpy()),axis=1).astype(float)
            hidden = context@first.weight.detach().cpu().numpy().T+first.bias.detach().cpu().numpy()
            hidden = .5*hidden*(1+erf(hidden/np.sqrt(2)))
            parameters = hidden@last.weight.detach().cpu().numpy().T+last.bias.detach().cpu().numpy()
            for local, number in enumerate(part):
                soft_rows.append(dict(scope=scope,split=split,arm=ARM,cell_uid=identifiers[number],
                                      soft_transition_center_cycle=100+1000*(.1+3.5*expit(parameters[local,4])),
                                      soft_transition_width_cycles=1000*(.05+.8*expit(parameters[local,5])),
                                      soft_transition_signed_amplitude=2*np.tanh(parameters[local,3]),
                                      relative_intensity_min_per1000=lam[local].min(),
                                      relative_intensity_max_per1000=lam[local].max(),certified_physical_knee=False))
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
            "soft_rows": pd.DataFrame(soft_rows),
            "max_absolute_prediction_replay_error": maximum}


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
        record(name, frame_fingerprint(frames[0]) == frame_fingerprint(frames[1]))
    with np.load(formal / "test_mechanism_diagnostics.npz", allow_pickle=False) as first, np.load(repeated / "test_mechanism_diagnostics.npz", allow_pickle=False) as second:
        record("all_saved_mechanism_tensors", sorted(first.files) == sorted(second.files) and all(tensor_digest(first[key]) == tensor_digest(second[key]) for key in first.files))
    published = independent.read_json(output_root / "analysis/TRAINING_REPRODUCTION.json")
    independent.compare_table(checks, "complete_training_reproduction_arithmetic", pd.DataFrame(published["comparison_checks"]), pd.DataFrame(results), ["comparison"], atol=0, rtol=0)
    status = "PASS" if all(item["exact_equal"] for item in results) else "FAIL"
    checks.require("complete_training_reproduction_reported_status", published["status"] == status
                   and published["epochs_per_fit"] == 150 and published["additional_optimizer_updates"] == 2100)
    divergence = next((i + 1 for i, (first, last) in enumerate(zip(*histories)) if nested_fingerprint(first) != nested_fingerprint(last)), None)
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
    fingerprint_controls(checks)
    contract, units = analysis.completed_units(args.output, args.data)
    data = training.load_frozen_data(args.data)
    checks.require('single_seed42_M5_only', contract['arm']==ARM and contract['seed']==42
                   and contract['config']['models']==[ARM] and contract['planned_optimizer_updates']==12450)
    checks.require('registered_method_hash', training.sha256(REPOSITORY/contract['methodology_contract']['path'])
                   ==contract['methodology_contract']['sha256'])
    smoke = independent.read_json(args.output/'SMOKE_QA.json')
    checks.require('smoke_pass_before_formal',smoke['status']=='PASS' and all(smoke['checks'].values())
                   and smoke['formal_optimizer_updates']==0 and smoke['engineering_optimizer_updates']==14
                   and smoke['test_observations_consumed']==0
                   and smoke['model_contract_sha256']==training.sha256(args.output/'MODEL_CONTRACT.json'))
    for item in smoke['files']:
        checks.require('smoke_hash/'+item['path'], training.sha256(args.output/item['path'])==item['sha256'])
    for device in ('CPU','CUDA'):
        receipt=independent.read_json(args.output/('ENGINEERING_'+device+'.json'))
        checks.require(device+'_independent_engineering',receipt['status']=='PASS' and receipt['optimizer_updates']==0
                       and receipt['test_observations_consumed']==0)
    main=Counter(uid for fold in SPLITS[:5] for uid in data[-2]['splits'][fold]['test'])
    checks.require('OOF169_once',len(data[1])==169 and data[2].all() and main==Counter({uid:1 for uid in data[1]}))
    results=[]
    for unit in units:
        print(json.dumps({'status':'REPLAYING','split':unit['split']}),flush=True)
        results.append(replay_unit(checks,unit,data,contract,args.device))
    repeated, reproduction=verify_full_reproduction(checks,args.output,contract)
    repeat=replay_unit(checks,repeated,data,contract,args.device)
    checks.require('all_seven_selected_predictions_exact',all(row['max_absolute_prediction_replay_error']==0 for row in results+[repeat]))
    checks.require('12450_plus2100_plus14',sum(row['training']['successful_updates'] for row in results)==12450
                   and repeat['training']['successful_updates']==2100 and contract['training_budget']['total_optimizer_updates']==14564)
    cells,future,split=analysis.historical.frozen_tables(args.data)
    history,bound=analysis.historical_results(contract,cells,future,split)
    metrics=pd.DataFrame([row for result in results+history for row in result['metrics']])
    summary,protocols,folds=independent.aggregate_metrics(metrics)
    comparisons,pairs=independent_comparisons(metrics)
    eol=pd.concat([row['eol'] for row in results+history],ignore_index=True)
    eol_stats,eol_pairs=independent_eol(eol,cells)
    query=pd.concat([row['query_range_rows'] for row in results],ignore_index=True)
    tables={
        'cell_metrics.csv':(metrics,['scope','arm','cell_uid','metric']),
        'metric_summary.csv':(summary,['scope','arm','metric']),
        'protocol_metrics.csv':(protocols,['scope','arm','metric','protocol_uid']),
        'fold_metrics.csv':(folds,['scope','split','arm','metric']),
        'paired_comparisons.csv':(comparisons,['scope','metric','reference','estimator']),
        'paired_cells.csv':(pairs,['scope','metric','reference','cell_uid']),
        'eol_predictions.csv':(eol,['scope','arm','cell_uid']),
        'eol_summary.csv':(eol_stats,['scope','arm']),
        'eol_paired_comparisons.csv':(eol_pairs,['scope','reference','sampling_unit']),
        'query_range_cells.csv':(query,['scope','cell_uid','query_region']),
        'query_range_summary.csv':(independent_query_summary(query),['scope','query_region']),
        'mechanism_summary.csv':(pd.concat([row['mechanism_rows'] for row in results],ignore_index=True),['split','cell_uid']),
        'MATR_b1c2_counterexample.csv':(metrics[(metrics.cell_uid=='MATR_b1c2')&(metrics.scope=='unseen_protocol_sensitivity')],['arm','metric'])}
    destination=args.output/'analysis'
    complete=independent.read_json(destination/'ANALYSIS_COMPLETE.json')
    checks.require('analysis_source_and_contract',complete['source_sha256']==training.sha256(REPOSITORY/'scripts/analyze_m5_results.py')
                   and complete['model_contract_sha256']==training.sha256(args.output/'MODEL_CONTRACT.json'))
    for item in complete['files']:
        checks.require('analysis_hash/'+item['path'],training.sha256(destination/item['path'])==item['sha256'])
    for name,(expected,keys) in tables.items():
        independent.compare_table(checks,'independent_statistics/'+name,pd.read_csv(destination/name),expected,keys,atol=2e-8,rtol=2e-5)
    checks.require('historical_complete_bindings',independent.read_json(destination/'HISTORICAL_REFERENCES.json')['units']==bound)
    soft=pd.read_csv(destination/'soft_transition_parameters.csv')
    independent.compare_table(checks,'independent_all_soft_parameters_and_intensity_extrema',soft,
                              pd.concat([row['soft_rows'] for row in results],ignore_index=True),
                              ['split','cell_uid'],atol=2e-4,rtol=2e-5)
    soft_expected=[]
    monotonic=[]
    for unit in units:
        with np.load(unit['root']/'test_mechanism_diagnostics.npz',allow_pickle=False) as archive:
            ids=archive['cell_uid'].astype(str).tolist()
            scope='main_oof' if unit['split'].startswith('cell_fold_') else 'unseen_protocol_sensitivity'
            saved=pd.read_parquet(unit['root']/'test_predictions.parquet')
            for i,uid in enumerate(ids):
                soft_expected.append(dict(cell_uid=uid,split=unit['split'],scope=scope,arm=ARM,
                                          **{key:float(archive['flow__'+key][i]) for key in
                                             ('soft_transition_center_cycle','soft_transition_width_cycles','soft_transition_signed_amplitude')},
                                          certified_physical_knee=False))
                part=saved[saved.cell_uid==uid].sort_values('original_cycle_id')
                slots=part.original_cycle_id.to_numpy(int)-101
                row=dict(scope=scope,split=unit['split'],arm=ARM,cell_uid=uid,observed_queries=len(part),observed_adjacent_pairs=max(len(part)-1,0))
                for name,values in (('raw_trend',archive['flow__raw_trend'][i,slots]),('trend',part.trend.to_numpy()),('prediction',part.prediction.to_numpy())):
                    differences=np.diff(values)
                    row[name+'_rise_count']=int(np.count_nonzero(differences>0))
                    row[name+'_rise_over1e6_count']=int(np.count_nonzero(differences>1e-6))
                    row[name+'_largest_positive_step']=float(differences.max(initial=0))
                monotonic.append(row)
    independent.compare_table(checks,'soft_centers_saved_and_replayed',soft,pd.DataFrame(soft_expected),['split','cell_uid'])
    checks.require('soft_parameters_unvalidated_and_within_bounds',len(soft)==215 and not soft.certified_physical_knee.any()
                   and soft.soft_transition_center_cycle.between(200,3700).all()
                   and soft.soft_transition_width_cycles.between(50,850).all()
                   and soft.relative_intensity_min_per1000.gt(0).all() and soft.relative_intensity_max_per1000.lt(2).all())
    independent.compare_table(checks,'observed_monotonicity_counts',pd.read_csv(destination/'observed_monotonicity.csv'),pd.DataFrame(monotonic),['split','cell_uid'],atol=2e-7)
    checks.require('latent_no_negative_and_no_increase',not query.raw_trend_below_zero_count.any()
                   and not pd.DataFrame(monotonic).raw_trend_rise_count.any())
    return dict(status='VERIFIED' if reproduction['status']=='PASS' else 'VERIFIED_WITH_REPRODUCTION_FAILURE',
                completed_units=6,normal_selected_checkpoint_replays=7,max_absolute_prediction_replay_error=0.,
                training_reproduction=reproduction,formal_optimizer_updates=12450,full_reproduction_optimizer_updates=2100,
                engineering_optimizer_updates=14,total_executed_optimizer_updates=14564,seed=42,replay_device=args.device,
                model_contract_sha256=training.sha256(args.output/'MODEL_CONTRACT.json'),
                analysis_complete_sha256=training.sha256(destination/'ANALYSIS_COMPLETE.json'),
                verification_dependencies=[{'path':name,'sha256':training.sha256(REPOSITORY/name)} for name in
                                           ('scripts/verify_methodology_full_seed42.py','scripts/verify_unified_seed42_results.py',
                                            'scripts/verify_m5_engineering.py','scripts/analyze_m5_results.py')],
                input_label_tensor_fingerprints={**{key:tensor_digest(value) for key,value in data[0].items()},
                                                **{key:tensor_digest(data[index]) for key,index in
                                                   (('query',4),('targets',5),('observed',6),('rates',7),('rate_mask',8))}},
                historical_reference_bindings=bound,
                limitations=['One seed42 and one full fold0 duplicate in one environment.',
                             'All169 cells were historically exposed. Numerical extrapolation beyond follow-up has no observed target.',
                             'Soft transition coordinates have no certified physical Knee truth.',
                             'Unstored epoch weights cannot be independently replayed. Selected validation, test and complete final optimizer state are checked.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    args=parser.parse_args()
    checks=independent.Checks()
    receipt=args.output/'RUN_VERIFICATION.json'
    if receipt.exists():
        previous=receipt.read_bytes()
        history=args.output/'verification_attempts'
        history.mkdir(exist_ok=True)
        (history/('RUN_VERIFICATION_'+training.sha256(receipt)+'.json')).write_bytes(previous)
    try:
        with threadpool_limits(limits=1):
            result=verify(args,checks)
    except Exception:
        result=dict(status='FAIL',traceback=traceback.format_exc())
    result['checks']=checks.rows
    result['counts']=dict(Counter(row['status'] for row in checks.rows))
    result['verifier_source_sha256']=training.sha256(__file__)
    training.write_json(receipt,result)
    print(json.dumps({key:result[key] for key in ('status','counts')}),flush=True)
    if result['status']=='FAIL':
        print(result['traceback'],flush=True)
        raise SystemExit(1)


if __name__=='__main__':
    main()
