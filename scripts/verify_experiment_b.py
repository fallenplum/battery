"""Read-only experiment B replay from frozen coefficients and stored predictions.

No estimator is fitted and the registered test entry point is never invoked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_true(value):
    return value is True or value == "True"


def matches(left, right):
    if left in (None, "") or right in (None, ""):
        return left in (None, "") and right in (None, "")
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-13)


def metric_values(truth, prediction):
    y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    error = p - y
    total = np.sum((y - y.mean()) ** 2)
    return {"n": len(y), "MAE": np.abs(error).mean(), "RMSE": np.sqrt(np.mean(error ** 2)),
            "R2": 1 - np.sum(error ** 2) / total if len(y) >= 2 and total > 0 else None,
            "Spearman": spearmanr(y, p).statistic if len(y) >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prior-metadata", type=Path)
    args = parser.parse_args()
    repo = args.repository
    output = repo / "results/experiment_b"
    contract = read(output / "CONTRACT.json")
    selections = read(output / "SELECTIONS.json")
    completed = read(output / "EVALUATED.json")
    opened = read(output / "TEST_OPENED.json")
    split = csv_rows(output / "split.csv")
    eligible = {row["cell_id"]: row for row in csv_rows(repo / "results/d0/eligible_cells.csv") if row["family"] == "HUST"}
    feature_rows = csv_rows(repo / "results/d0/early_observed_features.csv")
    features = {row["cell_id"]: row for row in feature_rows if row["family"] == "HUST" and int(row["prefix_cutoff_supplied_cycle_number"]) == 100}
    targets = {row["cell_id"]: row for row in csv_rows(repo / "results/d0/HUST_RAW_CAPACITY_TARGET_SUPPORT.csv")}
    legacy_json = read(repo / "data/legacy_576_ids.json")
    legacy = set(legacy_json if isinstance(legacy_json, list) else legacy_json.get("ids", legacy_json.get("cells", {})))
    legacy = {name.removesuffix(".pkl") for name in legacy}
    metadata = {}
    for row in csv_rows(repo / "data/cell_metadata.csv"):
        if row["family"] == "HUST":
            item = json.loads(row["metadata"])
            metadata[item["cell_id"].removesuffix(".pkl")] = item
    rates = {cell: [step["rate_in_C"] for step in item["discharge_protocol"][:3]] for cell, item in metadata.items()}
    policies = {row["cell_id"].removesuffix(".pkl"): row for row in csv_rows(repo / "results/d0/policy_scheduled_evidence.csv")}
    role = {row["cell_id"]: row["role"] for row in split}
    roles = {name: sorted(cell for cell, value in role.items() if value == name) for name in ("train", "val", "test")}
    checks = {}
    checks["unique76_cells_and33train11val32test"] = len(split) == len(role) == 76 and {key: len(value) for key, value in roles.items()} == {"train": 33, "val": 11, "test": 32}
    checks["cell_id_roles_pairwise_disjoint"] = not (set(roles["train"]) & set(roles["val"]) or set(roles["train"]) & set(roles["test"]) or set(roles["val"]) & set(roles["test"]))
    checks["exactly7_5_excluded"] = set(eligible) - set(role) == {"HUST_7-5"} and csv_rows(output / "excluded_cells.csv") == [{"cell_id": "HUST_7-5", "reasons": "CYCLES_1_THROUGH_100_INCOMPLETE"}]
    checks["legacy_roles_match_actual576_identity"] = all((cell in legacy) == (value in ("train", "val")) for cell, value in role.items()) and all(is_true(row["legacy_576_member"]) == (row["cell_id"] in legacy) for row in split)
    historical = sorted((cell for cell in role if cell in legacy), key=lambda cell: (hashlib.sha256(("physics-B-20261115|" + cell).encode()).hexdigest(), cell))
    checks["registered_hash_split_recomputed"] = set(roles["train"]) == set(historical[:33]) and set(roles["val"]) == set(historical[33:]) and len(historical) == 44
    checks["all76_meet_common_registered_intersection"] = all(is_true(eligible[cell]["physical_cycle_axis_authenticated"]) and is_true(eligible[cell]["prefix_K100_numbers_complete"]) and int(eligible[cell]["curve_valid_records_K100"]) >= 80 for cell in role)
    checks["prior_context_exact_first3_rates_only"] = all(json.loads(row["protocol_group"]) == rates[row["cell_id"]] for row in split) and contract["prior_context"]["last_stage"] == "excluded_due_to_incorrect_hardcoded_rate"
    checks["context_rate_ranges_match_actual_source"] = all(2 <= rates[cell][0] <= 5 and 1 <= rates[cell][1] <= 5 and 1 <= rates[cell][2] <= 5 for cell in role)
    context_confirmed = []
    for cell in role:
        observed = json.loads(policies[cell]["raw_stage_current_A_per_cycle_median_ranges"])
        declared = json.loads(policies[cell]["discharge_C_rates_declared"])[:3]
        context_confirmed.append(declared == rates[cell] and all(abs(rates[cell][index] + observed[f"Constant current discharge_{index}"]["median"] / 1.1) <= .05 for index in range(3)))
    checks["all76_three_prior_rates_match_original_source_current_medians"] = all(context_confirmed)
    checks["raw_primary_secondary_targets_registered"] = contract["targets"]["primary"] == "r500_1000" and contract["targets"]["secondary"] == "r200_500" and contract["inputs"]["target"]["sha256"] == sha(repo / "results/d0/HUST_RAW_CAPACITY_TARGET_SUPPORT.csv")
    input_paths = {"eligible": repo / "results/d0/eligible_cells.csv", "features": repo / "results/d0/early_observed_features.csv", "target": repo / "results/d0/HUST_RAW_CAPACITY_TARGET_SUPPORT.csv", "metadata": repo / "data/cell_metadata.csv"}
    checks["all4_registered_input_hashes_and_lengths_match"] = all(sha(path) == contract["inputs"][key]["sha256"] and path.stat().st_size == contract["inputs"][key]["bytes"] for key, path in input_paths.items())
    if args.prior_metadata is not None:
        checks["published_metadata_byte_identical_to_prior_dataset_audit"] = (repo / "data/cell_metadata.csv").read_bytes() == args.prior_metadata.read_bytes()
    checks["registered_implementation_hash_matches"] = sha(repo / "scripts/run_experiment_b.py") == contract["implementation_sha256"]
    contract_sha, selection_sha = sha(output / "CONTRACT.json"), sha(output / "SELECTIONS.json")
    checks["contract_selection_test_marker_hash_chain"] = selections["contract_sha256"] == opened["contract_sha256"] == completed["contract_sha256"] == contract_sha and opened["selection_sha256"] == completed["selection_sha256"] == selection_sha and (output / "SELECTIONS.sha256").read_text().strip() == selection_sha
    checks["all_completed_artifact_hashes_match"] = all((output / name).is_file() and sha(output / name) == value for name, value in completed["artifacts_sha256"].items())
    checks["split_payload_hash_matches"] = sha(output / "split.csv") == contract["split_sha256"]
    checks["stored_selection_train_val_ids_match"] = selections["train_ids"] == roles["train"] and selections["validation_ids"] == roles["val"]
    prediction_paths = list((output / "predictions").glob("*.csv"))
    chronology = [output / "CONTRACT.json", output / "SELECTIONS.json", output / "TEST_OPENED.json"]
    checks["selection_freeze_precedes_test_publication"] = chronology[0].stat().st_mtime_ns <= chronology[1].stat().st_mtime_ns <= chronology[2].stat().st_mtime_ns <= min(path.stat().st_mtime_ns for path in prediction_paths) <= (output / "EVALUATED.json").stat().st_mtime_ns
    values = {cell: {key: float(value) for key, value in features[cell].items() if key not in ("cell_id", "family") and value not in (None, "")} for cell in role}
    for cell in role:
        values[cell].update(dict(zip(contract["prior_context"]["columns"], rates[cell])))
    model_checks, prediction_groups = [], {}
    largest_normal_residual = 0.0
    for target, models in selections["models"].items():
        y_train = np.array([float(targets[cell][target]) for cell in roles["train"]])
        y_val = np.array([float(targets[cell][target]) for cell in roles["val"]])
        for model_name, model in models.items():
            columns = model["features"]
            x_train = np.array([[values[cell][column] for column in columns] for cell in roles["train"]])
            x_val = np.array([[values[cell][column] for column in columns] for cell in roles["val"]])
            x_test = np.array([[values[cell][column] for column in columns] for cell in roles["test"]])
            item = {"target": target, "model": model_name, "columns_match": columns == contract["models"][model_name]}
            if model["kind"] == "mean":
                predicted_val = np.full(len(y_val), model["mean"])
                predicted_test = np.full(len(x_test), model["mean"])
                item["mean_train_only"] = matches(model["mean"], y_train.mean())
            else:
                mean = np.array(model["scaler_mean"])
                scale = np.array(model["scaler_scale"])
                beta = np.array(model["coef_scaled"])
                expected_scale = x_train.std(axis=0)
                expected_scale[expected_scale == 0] = 1
                item["scaler_train_mean_std_exact"] = np.allclose(mean, x_train.mean(axis=0), rtol=1e-12, atol=1e-14) and np.allclose(scale, expected_scale, rtol=1e-12, atol=1e-14)
                z = (x_train - mean) / scale
                centered = z - z.mean(axis=0)
                residual = (centered.T @ centered + model["alpha"] * np.eye(len(columns))) @ beta - centered.T @ (y_train - y_train.mean())
                residual_max = float(np.max(np.abs(residual)))
                largest_normal_residual = max(largest_normal_residual, residual_max)
                item["frozen_coefficients_satisfy_train_ridge_equations"] = residual_max < 1e-11
                item["frozen_intercept_train_only"] = matches(model["intercept"], y_train.mean() - z.mean(axis=0) @ beta)
                predicted_val = ((x_val - mean) / scale) @ beta + model["intercept"]
                predicted_test = ((x_test - mean) / scale) @ beta + model["intercept"]
                grid = model["validation_grid"]
                selected = min(grid, key=lambda row: row["validation_MAE"])
                item["alpha_is_registered_validation_argmin"] = [row["alpha"] for row in grid] == contract["selection"]["alphas"] and model["alpha"] == selected["alpha"] and matches(model["validation_MAE"], selected["validation_MAE"])
            item["selected_validation_MAE_recomputed"] = matches(model["validation_MAE"], np.mean(np.abs(predicted_val - y_val)))
            path = output / "predictions" / f"{target}__{model_name}.csv"
            saved = csv_rows(path)
            receipt = read(path.with_suffix(".json"))
            item["prediction_cells_exact_test32"] = [row["cell_id"] for row in saved] == roles["test"]
            item["prediction_from_frozen_coefficients"] = np.allclose(predicted_test, [float(row["prediction"]) for row in saved], rtol=1e-12, atol=1e-14)
            item["prediction_truth_matches_raw_targets"] = all(matches(row["truth"], targets[row["cell_id"]][target]) and matches(row["absolute_error"], abs(float(row["prediction"]) - float(row["truth"]))) for row in saved)
            item["prediction_hash_receipt_and_freeze_chain"] = receipt["prediction_sha256"] == sha(path) and receipt["contract_sha256"] == contract_sha and receipt["selection_sha256"] == selection_sha and all(row["contract_sha256"] == contract_sha and row["selection_sha256"] == selection_sha for row in saved)
            model_checks.append(item)
            prediction_groups[target, model_name] = saved
    checks["all10_frozen_models_predictions_validation_checks"] = len(model_checks) == 10 and all(all(bool(value) for key, value in row.items() if key not in ("target", "model")) for row in model_checks)
    stored_metrics = {(row["target"], row["model"]): row for row in csv_rows(output / "metrics.csv")}
    protocol_metrics = {(row["target"], row["model"], row["protocol_group"]): row for row in csv_rows(output / "protocol_metrics.csv")}
    metric_checks = []
    grouped_predictions = defaultdict(list)
    for key, rows in prediction_groups.items():
        calculated = metric_values([row["truth"] for row in rows], [row["prediction"] for row in rows])
        metric_checks.append(all(matches(value, stored_metrics[key][name]) for name, value in calculated.items()))
        for row in rows:
            grouped_predictions[key + (row["protocol_group"],)].append(row)
    protocol_checks = []
    for key, rows in grouped_predictions.items():
        calculated = metric_values([row["truth"] for row in rows], [row["prediction"] for row in rows])
        protocol_checks.append(all(matches(value, protocol_metrics[key][name]) for name, value in calculated.items()))
    checks["all10_metric_groups_independently_recomputed"] = len(stored_metrics) == 10 and all(metric_checks)
    checks["all320_protocol_metric_groups_recomputed"] = len(protocol_metrics) == 320 and all(protocol_checks)
    aggregate = csv_rows(output / "predictions.csv")
    flat_predictions = [row for rows in prediction_groups.values() for row in rows]
    keys = ("cell_id", "target", "model")
    checks["aggregate320_predictions_match_group_files"] = len(aggregate) == 320 and {tuple(row[key] for key in keys): row for row in aggregate} == {tuple(row[key] for key in keys): row for row in flat_predictions}
    paired = {row["target"]: row for row in csv_rows(output / "paired_mae_improvement.csv")}
    confidence = []
    negatives = []
    for target in (contract["targets"]["primary"], contract["targets"]["secondary"]):
        control = prediction_groups[target, "context_capacity"]
        candidate = prediction_groups[target, "combined"]
        improvement = np.array([float(left["absolute_error"]) - float(right["absolute_error"]) for left, right in zip(control, candidate)])
        rng = np.random.default_rng(contract["paired_comparison"]["seed"])
        resamples = improvement[rng.integers(0, 32, size=(2000, 32))].mean(axis=1)
        low, high = np.quantile(resamples, [.025, .975])
        saved = paired[target]
        okay = matches(saved["MAE_improvement"], improvement.mean()) and matches(saved["CI95_low"], low) and matches(saved["CI95_high"], high) and int(saved["bootstrap_resamples"]) == 2000 and int(saved["bootstrap_seed"]) == 20261115
        confidence.append({"target": target, "recomputed": bool(okay), "MAE_improvement": float(improvement.mean()), "CI95_low": float(low), "CI95_high": float(high), "crosses_zero": bool(low <= 0 <= high)})
        negatives.extend((left["cell_id"], target, -float(gain)) for left, gain in zip(control, improvement) if gain < 0)
    saved_negative = csv_rows(output / "negative_cases.csv")
    checks["both_paired_bootstrap_intervals_recomputed"] = len(confidence) == 2 and all(row["recomputed"] for row in confidence)
    checks["both_intervals_include_zero"] = all(row["crosses_zero"] for row in confidence)
    checks["all27_negative_cases_preserved"] = len(negatives) == len(saved_negative) == 27 and {(cell, target) for cell, target, _ in negatives} == {(row["cell_id"], row["target"]) for row in saved_negative} and all(matches(gain, next(row["absolute_error_increase"] for row in saved_negative if row["cell_id"] == cell and row["target"] == target)) for cell, target, gain in negatives)
    checks["historical_exploratory_scope_disclosed"] = completed["status"] == "COMPLETE_EXPLORATORY_HOLDOUT" and any("does not establish fresh independent" in value for value in contract["limitations"])
    checks = {name: bool(value) for name, value in checks.items()}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "role_counts": {key: len(value) for key, value in roles.items()}, "model_checks": model_checks,
              "maximum_selected_ridge_normal_equation_residual": largest_normal_residual,
              "paired_intervals": confidence, "contract_sha256": contract_sha, "selection_sha256": selection_sha,
              "target_source": "HUST_RAW_CAPACITY_TARGET_SUPPORT.csv original net-counter capacities",
              "context_source": "Only first three source-confirmed prior discharge rates, final incorrect1C excluded",
              "fits_or_test_entrypoints_invoked": 0,
              "validation_grid_check_scope": "Recorded registered alpha grid argmin and frozen selected-model validation MAE, candidate models not refit",
              "conclusion": "Historical exploratory holdout with positive point-estimate improvement, both paired intervals include zero",
              "code_sha256": sha(Path(__file__))}
    (repo / "results/FINAL_EXPERIMENT_QA.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [key for key, value in checks.items() if not value]}))
    if not all(checks.values()):
        raise AssertionError("Experiment B cross-check failed")


if __name__ == "__main__":
    main()
