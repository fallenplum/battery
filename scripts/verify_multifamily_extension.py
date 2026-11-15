"""Independently replay frozen extension artifacts without fitting an estimator.

The verifier reads CSV inputs, stored coefficients and completed artifacts.
It imports neither experiment entry point and computes predictions by matrix
multiplication only. Historical files are checked against their published Git
snapshot, with the extension's README index update explicitly allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


BASELINE_COMMIT = "e8a7295d5199775006f7a0b750f58cb96488d5df"
FAMILIES = {"MATR": "matr", "Stanford_2": "stanford"}
TARGETS = ("r200_500", "r500_1000")
CAPACITY = ["early_capacity_level", "early_capacity_slope", "early_capacity_std"]
SHAPE = ["qv_window_capacity_start_Ah", "qv_window_capacity_end_Ah",
         "delta_qv_mean_Ah", "delta_qv_std_Ah", "delta_qv_log10_variance",
         "delta_qv_range_Ah", "delta_qv_rms_Ah"]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 20261115
RESAMPLES = 2000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def true(value):
    return str(value).lower() == "true"


def finite(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def context_number(value):
    normalized = str(value).lower()
    if normalized in ("true", "false"):
        return float(normalized == "true")
    return finite(value)


def matches(left, right):
    if left in (None, "") or right in (None, ""):
        return left in (None, "") and right in (None, "")
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-13)


def indexed(rows):
    result = {row["cell_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate cell IDs in an input or frozen split")
    return result


def metric_values(rows):
    truth = np.asarray([float(row["truth"]) for row in rows])
    prediction = np.asarray([float(row["prediction"]) for row in rows])
    error = prediction - truth
    total = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - np.sum(error ** 2) / total if len(truth) >= 2 and total > 0 else None
    correlation = None
    if len(truth) >= 2 and np.ptp(truth) > 0 and np.ptp(prediction) > 0:
        correlation = spearmanr(truth, prediction).statistic
    return {"n": len(rows), "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))), "R2": r2,
            "Spearman": correlation,
            "R2_reason": "AVAILABLE" if r2 is not None else "INSUFFICIENT_OR_CONSTANT_TARGET_SUPPORT",
            "Spearman_reason": "AVAILABLE" if correlation is not None else "INSUFFICIENT_OR_CONSTANT_VECTOR_SUPPORT"}


def metrics_match(rows, stored):
    expected = metric_values(rows)
    return all(stored[key] == value if key.endswith("_reason") else matches(stored[key], value)
               for key, value in expected.items())


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def artifact_hashes_match(directory, receipt):
    return all((directory / name).is_file() and sha(directory / name) == digest
               for name, digest in receipt["artifacts_sha256"].items())


def expected_artifact_names(directory, family_root):
    excluded = {directory / "EVALUATED.json"}
    return {path.relative_to(directory).as_posix() for path in directory.rglob("*")
            if path.is_file() and path.name != "RUN.lock" and path not in excluded
            and (family_root or path.name != "EVALUATED.json")}


def historical_snapshot(repo):
    process = subprocess.run(["git", "show", BASELINE_COMMIT + ":results/DELIVERY_MANIFEST.json"],
                             cwd=repo, check=True, capture_output=True)
    manifest_bytes = process.stdout
    manifest = json.loads(manifest_bytes)
    checked, failures = [], []
    for item in manifest["files"]:
        if item["path"] == "README.md":
            continue
        path = repo / item["path"]
        okay = path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
        checked.append(item["path"])
        if not okay:
            failures.append(item["path"])
    if (repo / "results/DELIVERY_MANIFEST.json").read_bytes() != manifest_bytes:
        failures.append("results/DELIVERY_MANIFEST.json")
    checked.append("results/DELIVERY_MANIFEST.json")
    return {"baseline_commit": BASELINE_COMMIT, "checked_files": len(checked),
            "checked_paths": checked, "changed_or_missing": failures,
            "allowed_versioned_document_update": "README.md extension index"}


def verify_prior_sources(repo, matr_policy_root, stanford_parameters):
    metadata = {row["family"]: {} for row in csv_rows(repo / "data/cell_metadata.csv")}
    for row in csv_rows(repo / "data/cell_metadata.csv"):
        item = json.loads(row["metadata"])
        metadata[row["family"]][item["cell_id"].removesuffix(".pkl")] = item
    matr = indexed(csv_rows(repo / "results/extension_matr/prior_context.csv"))
    matr_report = read_json(repo / "results/extension_matr/source_qualification.json")
    original, fingerprints, static_only = {}, [], []
    for item in matr_report["source_files"]:
        path = matr_policy_root / item["file"]
        payload = read_json(path)
        fingerprints.append({"file_name": path.name, "sha256": sha(path)})
        static_only.append(payload["static_policy_only"] is True and payload["summary_curves_cycle_life_targets_or_model_metrics_read"] is False and sha(path) == item["sha256"])
        for policy in payload["policies"]:
            identifier = policy["processed_cell_id"].removesuffix(".pkl")
            if identifier in matr:
                original[identifier] = policy
    checks = {"matr_three_policy_archive_hashes_static_only": all(static_only) and len(static_only) == 3,
              "matr_124_original_policy_IDs_recovered": len(original) == 124}
    expected_b4 = {(3.6, 6.0, 5.6, 4.755), (4.4, 5.6, 5.2, 4.252),
                   (4.8, 5.2, 5.2, 4.160), (5.2, 5.2, 4.8, 4.160),
                   (6.0, 5.6, 4.4, 3.834), (7.0, 4.8, 4.8, 3.652),
                   (8.0, 4.4, 4.4, 3.940), (8.0, 6.0, 4.8, 3.000),
                   (8.0, 7.0, 5.2, 2.680)}
    matr_matches, b4 = [], Counter()
    for identifier, row in matr.items():
        provided = metadata["MATR"][identifier]["charge_protocol"]
        rates = [float(stage["rate_in_C"]) for stage in provided]
        if identifier in original:
            text = original[identifier]["policy_original"]
            match = re.fullmatch(r"([0-9.]+)C\(([0-9.]+)%\)-([0-9.]+)C?(?:-newstructure)?", text)
            if match is None:
                raise ValueError("Unsupported original policy syntax in independent verifier")
            rate1, percent, rate2 = map(float, match.groups())
            soc = float(provided[0]["end_soc"])
            soc = soc / 100 if soc > 1 else soc
            original_matches = rates == [rate1, rate2] and matches(soc, percent / 100) and row["source_policy_string"] == text
            ends = [percent / 100, 0.8]
        else:
            b4[tuple(rates)] += 1
            original_matches = identifier.startswith("MATR_b4") and tuple(rates) in expected_b4
            ends = [0.2, 0.4, 0.6, 0.8]
            original_matches = original_matches and len(provided) == len(ends) and all(matches(stage["end_soc"], end) for stage, end in zip(provided, ends))
        expected = {"prior_charge_stage_count": len(rates), "prior_charge_first_switch_soc": ends[0],
                    "prior_charge_first_switch_soc_missing": 0, "prior_fast_charge_end_soc": 0.8,
                    "prior_common_topoff_rate_C": 1, "prior_discharge_rate_C": 4}
        for index in range(4):
            expected["prior_charge_rate_" + str(index + 1) + "_C"] = rates[index] if index < len(rates) else 0
            expected["prior_charge_rate_" + str(index + 1) + "_missing"] = int(index >= len(rates))
        group = json.loads(row["protocol_group"])
        matr_matches.append(original_matches and all(matches(row[key], value) for key, value in expected.items()) and set(expected) == {key for key in row if key.startswith("prior_")} and group == {"fast_charge_rates_C": rates, "fast_charge_end_soc": ends, "discharge_rate_C": 4})
    checks["matr_all169_prior_values_SOC_masks_and_protocols_match_design_sources"] = len(matr_matches) == 169 and all(matr_matches)
    checks["matr_b4_nine_published_profiles_five_each"] = set(b4) == expected_b4 and set(b4.values()) == {5}
    checks["matr_context_contains_no_measured_future_values"] = all(row["source_future_values_used"] == "False" and row["future_policy_status"] == "observed_only" for row in matr.values())
    stanford = indexed(csv_rows(repo / "results/extension_stanford/prior_context.csv"))
    stanford_report = read_json(repo / "results/extension_stanford/source_qualification.json")
    source = {int(row["seq_num"]): row for row in csv_rows(stanford_parameters)}
    parameter_sha = sha(stanford_parameters)
    checks["stanford_author_parameter_sha_matches_source_report"] = parameter_sha == stanford_report["evidence"]["formation_parameter_csv"]["sha256"]
    stanford_matches, hardware_hashes = [], []
    formation_names = ("formation_temperature", "formation_charge_current_1", "formation_cutoff_voltage_1",
                       "formation_charge_current_2", "formation_verification_repeat", "ocv_time")
    for identifier, row in stanford.items():
        sequence = int(identifier.rsplit("_", 1)[1])
        author = source[sequence]
        nominal = float(metadata["Stanford_2"][identifier]["nominal_capacity_in_Ah"])
        parameters = [float(author[name]) for name in formation_names]
        hardware_sha = hashlib.sha256(author["cell_id"].encode()).hexdigest()
        hardware_hashes.append(hardware_sha)
        expected = {"prior_charge_rate_C": float(author["regular_charge_current"]) / nominal,
                    "prior_discharge_rate_C": float(author["regular_discharge_current"]) / nominal,
                    "prior_nominal_capacity_Ah": nominal,
                    "prior_upper_voltage_V": float(author["regular_charge_cutoff_voltage"]),
                    "prior_lower_voltage_V": float(author["regular_discharge_cutoff_voltage"]),
                    "prior_aging_chamber_setpoint_C": 30,
                    "prior_formation_temperature_C": parameters[0],
                    "prior_formation_charge1_C": parameters[1] / nominal,
                    "prior_formation_transition_voltage_V": parameters[2],
                    "prior_formation_charge2_C": parameters[3] / nominal,
                    "prior_verification_cycles": parameters[4],
                    "prior_postformation_rest_hours": parameters[5],
                    "prior_missing_aging_protocol": 0, "prior_missing_formation_protocol": 0}
        group = json.loads(row["protocol_group"].removeprefix("formation_tuple:"))
        stanford_matches.append(all(matches(row[key], value) for key, value in expected.items()) and set(expected) == {key for key in row if key.startswith("prior_")} and group == parameters and row["source_device_id_sha256"] == hardware_sha and row["source_parameter_sha256"] == parameter_sha and int(row["source_seq_num"]) == sequence and author["regular_project_name"] == row["source_project_name"] == "Nova_Regular")
    checks["stanford_all181_source_formation_and_aging_design_priors_recomputed"] = len(stanford_matches) == 181 and all(stanford_matches) and len(set(hardware_hashes)) == 181
    checks["stanford_all_context_available_before_aging_and_design_scope_retained"] = all(true(row["metadata_context_available_before_aging"]) and row["future_policy_status"] == "observed_only" and row["future_policy_design_status"] == "documented_fixed_aging_design" for row in stanford.values())
    checks = {name: bool(value) for name, value in checks.items()}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "matr_policy_archives": fingerprints, "stanford_author_parameter_sha256": parameter_sha,
            "scope": "All prior numeric values matched to assigned designs or pre-aging formation parameters; fixed aging chamber30C and rest-time units retain primary-source interpretation from the qualification report",
            "physical_binding_limit": "Batch4 design-vector match retained; individual original-archive mapping remains unverified"}


def verify_family(repo, family, slug, legacy):
    output = repo / ("results/experiment_b_" + slug)
    context_path = repo / ("results/extension_" + slug + "/prior_context.csv")
    contract = read_json(output / "CONTRACT.json")
    selections = read_json(output / "SELECTIONS.json")
    completed = read_json(output / "EVALUATED.json")
    opened = read_json(output / "TEST_OPENED.json")
    split_rows = csv_rows(output / "split.csv")
    split = indexed(split_rows)
    eligible = indexed([row for row in csv_rows(repo / "results/d0/eligible_cells.csv") if row["family"] == family])
    feature = indexed([row for row in csv_rows(repo / "results/d0/early_observed_features.csv")
                       if row["family"] == family and finite(row["prefix_cutoff_supplied_cycle_number"]) == 100])
    target_rows = indexed([row for row in csv_rows(repo / "results/d0/future_target_support.csv") if row["family"] == family])
    contexts = indexed([row for row in csv_rows(context_path) if row["family"] == family])
    prior_columns = sorted({name for row in contexts.values() for name in row if name.startswith("prior_")})
    values, expected_excluded = {}, []
    for identifier, row in sorted(eligible.items()):
        reasons = []
        if not true(row["prefix_K100_numbers_complete"]):
            reasons.append("SUPPLIED_CYCLES_1_THROUGH_100_INCOMPLETE")
        count = finite(row["curve_valid_records_K100"])
        if count is None or count < 80:
            reasons.append("FEWER_THAN_80_VALID_K100_OBSERVED_CURVES")
        prefix = {name: finite(feature.get(identifier, {}).get(name)) for name in CAPACITY + SHAPE}
        missing = [name for name, number in prefix.items() if number is None]
        if missing:
            reasons.append("MISSING_PREFIX_FEATURES=" + ",".join(missing))
        prior = {name: context_number(contexts.get(identifier, {}).get(name)) for name in prior_columns}
        missing = [name for name, number in prior.items() if number is None]
        if missing:
            reasons.append("MISSING_REGISTERED_PRIOR_CONTEXT=" + ",".join(missing))
        if reasons:
            expected_excluded.append({"cell_id": identifier, "reasons": "|".join(reasons)})
        else:
            values[identifier] = {**prefix, **prior}
    roles = {role: sorted(cell for cell, row in split.items() if row["role"] == role)
             for role in ("train", "val", "test")}
    historical = sorted((cell for cell in values if cell in legacy), key=lambda cell:
                        (hashlib.sha256(("physics-B-extension-20261115|" + cell).encode()).hexdigest(), cell))
    training_count = math.floor(0.75 * len(historical))
    expected_models = {"mean": prior_columns, "context_only": prior_columns,
                       "context_capacity": prior_columns + CAPACITY,
                       "context_qv_shape": prior_columns + SHAPE,
                       "combined": prior_columns + CAPACITY + SHAPE}
    inputs = {"eligible": repo / "results/d0/eligible_cells.csv",
              "features": repo / "results/d0/early_observed_features.csv",
              "target": repo / "results/d0/future_target_support.csv", "context": context_path}
    contract_sha, selection_sha = sha(output / "CONTRACT.json"), sha(output / "SELECTIONS.json")
    checks = {}
    checks["family_and_fixed_supplied_coordinate"] = contract["family"] == family and contract["coordinate"] == "supplied_cycle_number" and contract["prefix"] == 100
    checks["base_admission_recomputed_without_future_target_filter"] = set(split) == set(values) and csv_rows(output / "excluded_cells.csv") == expected_excluded
    checks["base_legacy_roles_match_actual576_ids"] = all(true(row["legacy_576_member"]) == (cell in legacy) == (row["role"] in ("train", "val")) for cell, row in split.items())
    checks["hash_split_independent_of_target_support"] = set(roles["train"]) == set(historical[:training_count]) and set(roles["val"]) == set(historical[training_count:]) and set(roles["test"]) == set(values) - set(historical)
    checks["base_roles_pairwise_disjoint_complete"] = set().union(*(set(items) for items in roles.values())) == set(split) and sum(map(len, roles.values())) == len(split)
    checks["role_counts_and_split_sha_match"] = contract["role_counts"] == {key: len(items) for key, items in roles.items()} and contract["split_sha256"] == sha(output / "split.csv")
    checks["protocol_groups_come_from_pre_prediction_context"] = all(row["protocol_group"] == contexts[cell]["protocol_group"] for cell, row in split.items())
    checks["context_all_finite_prior_columns_only"] = prior_columns == contract["context"]["input_columns"] and all(math.isfinite(values[cell][name]) for cell in values for name in prior_columns) and not contract["context"]["protocol_group_used_as_numeric_input"]
    checks["all_models_share_same_prior_budget"] = contract["models"] == expected_models
    checks["registered_all_input_hashes_lengths_match"] = all(sha(path) == contract["inputs"][name]["sha256"] and path.stat().st_size == contract["inputs"][name]["bytes"] for name, path in inputs.items())
    checks["runner_and_frozen_hust_core_sha_match"] = sha(repo / "scripts/run_multifamily_b.py") == contract["implementation_sha256"] and sha(repo / "scripts/run_experiment_b.py") == contract["shared_hust_implementation_sha256"]
    checks["family_freeze_hash_chain"] = selections["contract_sha256"] == completed["contract_sha256"] == opened["contract_sha256"] == contract_sha and opened["all_target_selection_sha256"] == completed["all_target_selection_sha256"] == selection_sha and (output / "SELECTIONS.sha256").read_text().strip() == selection_sha
    checks["all_family_completed_artifact_hashes_match"] = artifact_hashes_match(output, completed)
    checks["family_receipt_covers_every_output_file"] = set(completed["artifacts_sha256"]) == expected_artifact_names(output, True)
    checks["all_target_selections_frozen_before_family_test_open"] = (output / "CONTRACT.json").stat().st_mtime_ns <= (output / "SELECTIONS.json").stat().st_mtime_ns <= (output / "TEST_OPENED.json").stat().st_mtime_ns
    checks["physical_axis_evidence_preserved_without_admission_exclusion"] = contract["evidence"]["strict_physical_axis_authenticated_base_cells"] == sum(true(eligible[cell]["physical_cycle_axis_authenticated"]) for cell in split) and all(true(row["physical_cycle_axis_authenticated"]) == true(eligible[cell]["physical_cycle_axis_authenticated"]) for cell, row in split.items())
    checks["registered_alpha_scaler_refit_policy"] = contract["selection"]["alphas"] == ALPHAS and contract["selection"]["scaler_fit"] == "target-specific train cells only" and contract["selection"]["selected_refit_on_validation"] is False
    checks["historical_exploratory_limits_preserved"] = completed["status"] == "COMPLETE_EXPLORATORY_HOLDOUT" and any("Historical-data exploratory" in text for text in contract["limitations"]) and any("does not certify" in text for text in contract["limitations"])
    support_saved = {(row["cell_id"], row["target"]): row for row in csv_rows(output / "target_cell_support.csv")}
    all_predictions, all_metric_rows, all_protocol_rows = [], [], []
    all_paired_rows, all_negative_rows = [], []
    models_checked, target_results, protocol_role_support = [], {}, []
    maximum_residual = 0.0
    completed_targets = []
    for target in TARGETS:
        supported = {role: [cell for cell in identifiers if finite(target_rows.get(cell, {}).get(target)) is not None]
                     for role, identifiers in roles.items()}
        available = sorted(cell for identifiers in supported.values() for cell in identifiers)
        available_set = set(available)
        missing = sorted(set(split) - available_set)
        support = contract["target_support"][target]
        counts = {role: len(identifiers) for role, identifiers in supported.items()}
        ready = counts["train"] >= 2 and counts["val"] >= 1 and counts["test"] >= 1
        prefix = target + "__"
        checks[prefix + "target_specific_support_and_fixed_base_roles"] = support["role_counts"] == counts and support["available_cell_ids"] == available and support["missing_cell_ids"] == missing and (support["status"] == "READY") == ready
        checks[prefix + "all_cell_target_masks_and_reasons_recomputed"] = all((cell, target) in support_saved and true(support_saved[cell, target]["observed_target"]) == (cell in available_set) and support_saved[cell, target]["role"] == split[cell]["role"] and support_saved[cell, target]["source_reason"] == target_rows.get(cell, {}).get(target + "_reason", "UNAVAILABLE") for cell in split)
        group_counts = defaultdict(Counter)
        for role, identifiers in supported.items():
            for cell in identifiers:
                group_counts[split[cell]["protocol_group"]][role] += 1
        protocol_role_support.extend({"target": target, "protocol_group": group, **{role: counts[role] for role in roles},
                                      "test_group_seen_in_train": counts["train"] > 0}
                                     for group, counts in sorted(group_counts.items()))
        if not ready:
            target_results[target] = {"status": "INSUFFICIENT_SUPPORT", "role_counts": counts}
            continue
        completed_targets.append(target)
        directory = output / "targets" / target
        selected = selections["targets"][target]
        saved_selection = read_json(directory / "SELECTIONS.json")
        target_selection_sha = sha(directory / "SELECTIONS.json")
        target_opened = read_json(directory / "TEST_OPENED.json")
        target_complete = read_json(directory / "EVALUATED.json")
        checks[prefix + "target_selection_and_hash_chain"] = saved_selection == selected and selected["contract_sha256"] == target_opened["contract_sha256"] == target_complete["contract_sha256"] == contract_sha and target_opened["selection_sha256"] == target_complete["selection_sha256"] == target_selection_sha and (directory / "SELECTIONS.sha256").read_text().strip() == target_selection_sha
        checks[prefix + "target_receipt_all_output_hashes"] = artifact_hashes_match(directory, target_complete) and set(target_complete["artifacts_sha256"]) == expected_artifact_names(directory, False)
        checks[prefix + "source_train_val_ids_exact_target_mask"] = selected["train_ids"] == supported["train"] and selected["validation_ids"] == supported["val"]
        checks[prefix + "five_models_exact_registered_set"] = set(selected["models"][target]) == set(expected_models)
        y_train = np.asarray([float(target_rows[cell][target]) for cell in supported["train"]])
        y_val = np.asarray([float(target_rows[cell][target]) for cell in supported["val"]])
        prediction_groups = {}
        for name, model in selected["models"][target].items():
            columns = model["features"]
            matrices = {role: np.asarray([[values[cell][column] for column in columns] for cell in identifiers], dtype=float)
                        for role, identifiers in supported.items()}
            item = {"target": target, "model": name, "features_match_contract": columns == expected_models[name]}
            if model["kind"] == "mean":
                predicted_val = np.full(len(y_val), model["mean"])
                predicted_test = np.full(len(supported["test"]), model["mean"])
                item["mean_uses_target_train_only"] = matches(model["mean"], y_train.mean())
            else:
                train = matrices["train"]
                mean = np.asarray(model["scaler_mean"])
                scale = np.asarray(model["scaler_scale"])
                beta = np.asarray(model["coef_scaled"])
                variance = train.var(axis=0)
                expected_scale = np.sqrt(variance)
                eps = np.finfo(float).eps
                constant_bound = len(train) * eps * variance + (len(train) * train.mean(axis=0) * eps) ** 2
                expected_scale[variance <= constant_bound] = 1.0
                item["scaler_recomputed_from_target_training_only"] = model["scaler_fit_role"] == "train_only" and np.allclose(mean, train.mean(axis=0), rtol=1e-12, atol=1e-14) and np.allclose(scale, expected_scale, rtol=1e-12, atol=1e-14)
                z = (train - mean) / scale
                centered = z - z.mean(axis=0)
                residual = (centered.T @ centered + model["alpha"] * np.eye(len(columns))) @ beta - centered.T @ (y_train - y_train.mean())
                maximum_residual = max(maximum_residual, float(np.max(np.abs(residual))))
                item["stored_coefficients_satisfy_ridge_normal_equations"] = np.max(np.abs(residual)) < 1e-10
                item["stored_intercept_uses_training_only"] = matches(model["intercept"], y_train.mean() - z.mean(axis=0) @ beta)
                predicted_val = ((matrices["val"] - mean) / scale) @ beta + model["intercept"]
                predicted_test = ((matrices["test"] - mean) / scale) @ beta + model["intercept"]
                grid = model["validation_grid"]
                best = min(grid, key=lambda row: row["validation_MAE"])
                item["alpha_matches_registered_validation_argmin"] = [row["alpha"] for row in grid] == ALPHAS and model["alpha"] == best["alpha"] and matches(model["validation_MAE"], best["validation_MAE"])
            item["selected_validation_MAE_recomputed"] = matches(model["validation_MAE"], np.mean(np.abs(predicted_val - y_val)))
            prediction_path = directory / "predictions" / (target + "__" + name + ".csv")
            saved = csv_rows(prediction_path)
            receipt = read_json(prediction_path.with_suffix(".json"))
            item["test_IDs_exact_target_mask_for_every_model"] = [row["cell_id"] for row in saved] == supported["test"]
            item["test_prediction_replayed_from_frozen_coefficients"] = len(saved) == len(predicted_test) and np.allclose(predicted_test, [float(row["prediction"]) for row in saved], rtol=1e-12, atol=1e-14)
            item["truth_errors_protocol_and_role_match_source"] = all(matches(row["truth"], target_rows[row["cell_id"]][target]) and matches(row["absolute_error"], abs(float(row["truth"]) - float(row["prediction"]))) and row["protocol_group"] == split[row["cell_id"]]["protocol_group"] and row["role"] == "test" and row["target"] == target and row["model"] == name for row in saved)
            item["prediction_receipt_and_freeze_sha_chain"] = receipt["prediction_sha256"] == sha(prediction_path) and receipt["contract_sha256"] == contract_sha and receipt["selection_sha256"] == target_selection_sha and receipt["cells"] == len(saved) and all(row["contract_sha256"] == contract_sha and row["selection_sha256"] == target_selection_sha for row in saved)
            item["test_prediction_publication_after_all_source_selections"] = (output / "SELECTIONS.json").stat().st_mtime_ns <= (output / "TEST_OPENED.json").stat().st_mtime_ns <= (directory / "TEST_OPENED.json").stat().st_mtime_ns <= prediction_path.stat().st_mtime_ns <= (output / "EVALUATED.json").stat().st_mtime_ns
            models_checked.append({key: bool(value) if key not in ("target", "model") else value for key, value in item.items()})
            prediction_groups[name] = saved
        aggregate = csv_rows(directory / "predictions.csv")
        flat = [row for rows in prediction_groups.values() for row in rows]
        key = lambda row: (row["cell_id"], row["target"], row["model"])
        checks[prefix + "target_aggregate_contains_all_prediction_arrays_once"] = len(aggregate) == len(flat) and {key(row): row for row in aggregate} == {key(row): row for row in flat}
        metrics = csv_rows(directory / "metrics.csv")
        checks[prefix + "all_model_metrics_independently_recomputed"] = len(metrics) == len(expected_models) and {row["model"] for row in metrics} == set(expected_models) and all(row["target"] == target and metrics_match(prediction_groups[row["model"]], row) for row in metrics)
        protocol_metrics = csv_rows(directory / "protocol_metrics.csv")
        grouped = defaultdict(list)
        for name, rows in prediction_groups.items():
            for row in rows:
                grouped[name, row["protocol_group"]].append(row)
        checks[prefix + "all_protocol_metrics_support_and_reasons_recomputed"] = len(protocol_metrics) == len(grouped) and set(grouped) == {(row["model"], row["protocol_group"]) for row in protocol_metrics} and all(metrics_match(grouped[row["model"], row["protocol_group"]], row) for row in protocol_metrics)
        baseline, candidate = prediction_groups["context_capacity"], prediction_groups["combined"]
        gains = np.asarray([float(left["absolute_error"]) - float(right["absolute_error"]) for left, right in zip(baseline, candidate)])
        low, high = bootstrap(gains)
        paired = csv_rows(directory / "paired_mae_improvement.csv")
        saved = paired[0]
        checks[prefix + "paired_cell_bootstrap_2000_recomputed"] = len(paired) == 1 and saved["target"] == target and matches(saved["MAE_improvement"], gains.mean()) and matches(saved["CI95_low"], low) and matches(saved["CI95_high"], high) and int(saved["n_test_cells"]) == len(gains) and int(saved["bootstrap_resamples"]) == RESAMPLES and int(saved["bootstrap_seed"]) == SEED
        negatives = csv_rows(directory / "negative_cases.csv")
        expected_negative = {left["cell_id"]: (left, right, -float(gain)) for left, right, gain in zip(baseline, candidate, gains) if gain < 0}
        checks[prefix + "all_negative_cases_and_magnitudes_preserved"] = len(negatives) == len(expected_negative) and {row["cell_id"] for row in negatives} == set(expected_negative) and all(matches(row["absolute_error_increase"], expected_negative[row["cell_id"]][2]) and matches(row["truth"], expected_negative[row["cell_id"]][0]["truth"]) and matches(row["context_capacity_prediction"], expected_negative[row["cell_id"]][0]["prediction"]) and matches(row["combined_prediction"], expected_negative[row["cell_id"]][1]["prediction"]) and row["protocol_group"] == expected_negative[row["cell_id"]][0]["protocol_group"] for row in negatives)
        all_predictions.extend(aggregate)
        all_metric_rows.extend(metrics)
        all_protocol_rows.extend(protocol_metrics)
        all_paired_rows.extend(paired)
        all_negative_rows.extend(negatives)
        target_results[target] = {"status": "COMPLETE_EXPLORATORY_HOLDOUT", "role_counts": counts,
                                  "missing_target_by_base_role": {role: len(roles[role]) - counts[role] for role in roles},
                                  "test_IDs": supported["test"], "negative_cases": len(negatives),
                                  "cell_paired_MAE_improvement": float(gains.mean()),
                                  "cell_bootstrap_CI95": [low, high], "test_protocol_groups": len({row["protocol_group"] for row in baseline})}
    checks["cell_target_support_rows_complete_unique"] = len(support_saved) == len(split) * len(TARGETS) == len(csv_rows(output / "target_cell_support.csv"))
    checks["all_stored_model_prediction_and_validation_checks"] = len(models_checked) == len(completed_targets) * len(expected_models) and all(all(value for key, value in item.items() if key not in ("target", "model")) for item in models_checked)
    checks["family_completed_target_list_and_prediction_array_counts"] = completed["completed_targets"] == completed_targets and completed["test_prediction_arrays"] == len(completed_targets) * len(expected_models)
    for name, rows in (("predictions.csv", all_predictions), ("metrics.csv", all_metric_rows),
                       ("protocol_metrics.csv", all_protocol_rows), ("paired_mae_improvement.csv", all_paired_rows),
                       ("negative_cases.csv", all_negative_rows)):
        checks["family_aggregate_matches_target_outputs__" + name] = csv_rows(output / name) == [{"family": family, **row} for row in rows]
    macro_rows = {row["target"]: row for row in csv_rows(output / "protocol_macro_paired_improvement.csv")}
    group_rows = {(row["target"], row["protocol_group"]): row for row in csv_rows(output / "protocol_pair_support.csv")}
    expected_group_keys = set()
    macro_checks = []
    for target in completed_targets:
        baseline = {row["cell_id"]: row for row in all_predictions if row["target"] == target and row["model"] == "context_capacity"}
        candidate = {row["cell_id"]: row for row in all_predictions if row["target"] == target and row["model"] == "combined"}
        groups = defaultdict(list)
        for cell, left in baseline.items():
            right = candidate[cell]
            groups[left["protocol_group"]].append((float(left["absolute_error"]), float(right["absolute_error"])))
        group_gains = []
        support_okay = True
        for group, pairs in sorted(groups.items()):
            pairs = np.asarray(pairs)
            gain = float((pairs[:, 0] - pairs[:, 1]).mean())
            group_gains.append(gain)
            expected_group_keys.add((target, group))
            saved = group_rows[target, group]
            support_okay = support_okay and int(saved["n_test_cells"]) == len(pairs) and matches(saved["context_capacity_MAE"], pairs[:, 0].mean()) and matches(saved["combined_MAE"], pairs[:, 1].mean()) and matches(saved["MAE_improvement"], gain)
        low, high = bootstrap(group_gains) if len(group_gains) >= 2 else (None, None)
        saved = macro_rows[target]
        okay = support_okay and int(saved["protocol_groups"]) == len(group_gains) and int(saved["test_cells"]) == len(baseline) and matches(saved["protocol_macro_MAE_improvement"], np.mean(group_gains)) and matches(saved["CI95_low"], low) and matches(saved["CI95_high"], high) and saved["bootstrap_unit"] == "protocol_group" and int(saved["bootstrap_resamples"]) == RESAMPLES and int(saved["bootstrap_seed"]) == SEED
        macro_checks.append(okay)
        target_results[target]["protocol_macro_MAE_improvement"] = float(np.mean(group_gains))
        target_results[target]["protocol_group_bootstrap_CI95"] = [low, high]
    checks["protocol_macro_equal_weight_and_group_bootstrap_2000_recomputed"] = all(macro_checks) and set(macro_rows) == set(completed_targets) and set(group_rows) == expected_group_keys
    checks = {name: bool(value) for name, value in checks.items()}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "base_role_counts": {role: len(items) for role, items in roles.items()},
            "context_numeric_columns": len(prior_columns), "target_results": target_results,
            "model_checks": models_checked, "protocol_role_support": protocol_role_support,
            "maximum_stored_ridge_normal_equation_residual": maximum_residual,
            "contract_sha256": contract_sha, "selection_sha256": selection_sha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--matr-policy-root", type=Path)
    parser.add_argument("--stanford-parameters", type=Path)
    args = parser.parse_args()
    repo = args.repository
    legacy_source = read_json(repo / "data/legacy_576_ids.json")
    legacy = legacy_source if isinstance(legacy_source, list) else legacy_source.get("ids", legacy_source.get("cells", {}))
    legacy = {identifier.removesuffix(".pkl") for identifier in legacy}
    families = {family: verify_family(repo, family, slug, legacy) for family, slug in FAMILIES.items()}
    historical = historical_snapshot(repo)
    matr_policy_root = args.matr_policy_root or repo.parents[2] / "BatteryLife/dataset/MATR_policy_metadata_20261029"
    stanford_parameters = args.stanford_parameters or repo.parent / "private/stanford_source_metadata/Formation_2022-Parameter.csv"
    source_context = verify_prior_sources(repo, matr_policy_root, stanford_parameters)
    result = {"status": "PASS" if all(item["status"] == "PASS" for item in families.values()) and source_context["status"] == "PASS" and not historical["changed_or_missing"] else "FAIL",
              "families": families, "historical_snapshot": historical,
              "source_context": source_context,
              "fits_or_test_entrypoints_invoked": 0,
              "prediction_replay": "Stored scaler/coefficient matrix multiplication only; experiment modules never imported",
              "alpha_validation_check": "Registered ascending candidate-grid argmin and independent selected-model validation MAE; no candidate refit",
              "bootstrap_replay": {"cell_resamples": RESAMPLES, "protocol_group_resamples": RESAMPLES, "seed": SEED},
              "scope": "Historical exploratory supplied-coordinate cohorts; physical-axis qualification remains in source reports",
              "code_sha256": sha(Path(__file__))}
    destination = repo / "results/MULTIFAMILY_EXTENSION_QA.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    failed = {family: [key for key, value in item["checks"].items() if not value] for family, item in families.items()}
    failed["source_context"] = [key for key, value in source_context["checks"].items() if not value]
    print(json.dumps({"status": result["status"], "checks": sum(len(item["checks"]) for item in families.values()) + len(source_context["checks"]),
                      "failed": failed, "historical_files_checked": historical["checked_files"],
                      "historical_changed_or_missing": historical["changed_or_missing"]}, ensure_ascii=False))
    if result["status"] != "PASS":
        raise AssertionError("Independent multifamily extension verification failed")


if __name__ == "__main__":
    main()
