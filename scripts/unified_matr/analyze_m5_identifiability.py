"""Frozen-prefix rate regression and explicitly unvalidated Hinge diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.unified_matr.train_seed42 import execution_lock, load_frozen_data, sha256, write_json
from scripts.unified_matr.physics_features import GLOBAL_COLUMNS, LOCAL_COLUMNS
from scripts.unified_matr.analyze_results import paired_mean_bootstrap

ALPHAS = (.01, .1, 1., 10., 100.)
SPLITS = tuple(f"cell_fold_{i}" for i in range(5))
LOCAL_INDICES = (0, 1)
SOURCE_COMMIT = "85c6ff92301a2e14f9ae72299f672f34a26bebe3"


def feature_names():
    prior = ([f"policy_value_{i}" for i in range(15)]
             + [f"policy_mask_{i}" for i in range(15)] + ["anchor_soh100"])
    values = list(GLOBAL_COLUMNS) + [
        f"bin{b}_{LOCAL_COLUMNS[j]}" for b in range(8) for j in LOCAL_INDICES]
    physics = values + [name + "_available" for name in values]
    return {"C0": prior, "C1": prior + physics}


def early_features(inputs, identifiers):
    policy = np.asarray(inputs["policy"], float)
    policy_mask = np.asarray(inputs["policy_mask"], bool)
    prior = np.column_stack((np.where(policy_mask, policy, np.nan),
                             policy_mask.astype(float),
                             np.asarray(inputs["anchor_soh100"], float)))
    physics, audit = [], []
    for i, uid in enumerate(identifiers):
        available = np.asarray(inputs["point_mask"][i], bool).any(-1)
        cycles = np.asarray(inputs["cycle_ids"][i], float)
        native = cycles[available]
        if (not len(native) or not np.isfinite(native).all() or
                not ((native >= 1) & (native <= 100) & (native == np.floor(native))).all()):
            raise RuntimeError("Prefix physical-cycle certification failed")
        gm = np.asarray(inputs["global_mask"][i], bool) & available[:, None]
        lm = np.asarray(inputs["local_feature_mask"][i], bool) & available[:, None, None]
        qualified = available & (gm.any(-1) | lm.any((1, 2)))
        if qualified.any():
            last = int(np.flatnonzero(qualified)[-1])
            gmask = gm[last]
            lmask = lm[last][:, list(LOCAL_INDICES)].reshape(-1)
            global_value = np.where(gmask, inputs["global_features"][i, last], np.nan)
            local_value = np.where(lmask, inputs["local_features"][i, last][:, list(LOCAL_INDICES)].reshape(-1), np.nan)
            values = np.concatenate((global_value, local_value))
            masks = np.concatenate((gmask, lmask))
            if not np.isfinite(values[masks]).all():
                raise RuntimeError("A supported early physical feature is nonfinite")
            selected_cycle = float(cycles[last])
        else:
            values, masks, selected_cycle = np.full(36, np.nan), np.zeros(36, bool), None
        physics.append(np.r_[values, masks.astype(float)])
        audit.append({"cell_uid": uid, "selected_physics_cycle": selected_cycle,
                      "qualified_prefix_records": int(qualified.sum()),
                      "global_feature_support": int(masks[:20].sum()),
                      "local_mean_rms_support": int(masks[20:].sum()),
                      "prefix_feature_cutoff": 100})
    designs = {"C0": prior, "C1": np.column_stack((prior, np.asarray(physics)))}
    if designs["C0"].shape[1] != 31 or designs["C1"].shape[1] != 103:
        raise RuntimeError("Registered feature dimensions changed")
    return designs, pd.DataFrame(audit)


def targets_by_task(rates, mask):
    both = mask[:, 0] & mask[:, 1]
    return {
        "level_mid_all": (rates[:, 0].astype(float), mask[:, 0]),
        "level_mid_shared": (rates[:, 0].astype(float), both),
        "level_late_shared": (rates[:, 1].astype(float), both),
        "change_late_minus_mid_shared": (rates[:, 1].astype(float) - rates[:, 0], both),
    }


def fit_selected_ridge(xtrain, ytrain, xval, yval, xtest):
    selected, best, trials = None, float("inf"), []
    for alpha in ALPHAS:
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(), Ridge(alpha=alpha))
        model.fit(xtrain, ytrain)
        val_mae = float(np.abs(model.predict(xval) - yval).mean())
        trials.append({"alpha": alpha, "validation_mae_per1000": val_mae})
        if val_mae < best:
            best, selected = val_mae, model
    imputer, scaler, ridge = selected.steps[0][1], selected.steps[1][1], selected.steps[2][1]
    fitted = {
        "alpha": float(ridge.alpha), "validation_mae_per1000": best, "trials": trials,
        "imputer_medians": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "ridge_coef_per1000": ridge.coef_.tolist(), "ridge_intercept_per1000": float(ridge.intercept_),
        "preprocessing_fit_roles": ["train"], "coefficient_fit_roles": ["train"],
        "validation_role": "alpha_selection_only", "refit_train_plus_validation": False,
    }
    return selected.predict(xtest), fitted


def safe_spearman(truth, prediction):
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        return np.nan
    return float(spearmanr(truth, prediction).statistic)


def rate_regression(designs, identifiers, rates, mask, split, cells):
    lookup = {uid: number for number, uid in enumerate(identifiers)}
    rows, models = [], []
    expected = {"level_mid_all": 151, "level_mid_shared": 42,
                "level_late_shared": 42, "change_late_minus_mid_shared": 42}
    for task, (target, support) in targets_by_task(rates, mask).items():
        if int(support.sum()) != expected[task]:
            raise RuntimeError("Frozen rate support changed for " + task)
        for fold in SPLITS:
            roles = {role: np.array([lookup[uid] for uid in split["splits"][fold][role]
                                    if support[lookup[uid]]], int)
                     for role in ("train", "validation", "test")}
            train, validation, test = (roles[key] for key in ("train", "validation", "test"))
            if min(len(train), len(validation), len(test)) == 0:
                raise RuntimeError("An actual rate task has an empty role")
            training_mean = float(target[train].mean())
            for name, design in designs.items():
                pred, fitted = fit_selected_ridge(design[train], target[train] * 1000,
                                                 design[validation], target[validation] * 1000, design[test])
                model_id = task + "/" + fold + "/" + name
                models.append({"model_id": model_id, "task": task, "split": fold, "model": name,
                               "roles": {role: [identifiers[i] for i in ids] for role, ids in roles.items()},
                               "training_target_mean": training_mean, **fitted})
                for i, value in zip(test, pred / 1000):
                    truth = float(target[i])
                    rows.append({"task": task, "split": fold, "model": name, "model_id": model_id,
                                 "cell_uid": identifiers[i],
                                 "protocol_uid": str(cells.loc[identifiers[i], "protocol_uid"]),
                                 "truth": truth, "prediction": float(value),
                                 "signed_error": float(value - truth), "abs_error": float(abs(value - truth)),
                                 "training_mean_prediction": training_mean,
                                 "training_mean_abs_error": abs(training_mean - truth),
                                 "alpha": fitted["alpha"], "val_mae_per1000": fitted["validation_mae_per1000"],
                                 "train_cells": len(train), "validation_cells": len(validation), "test_cells": len(test)})
    return pd.DataFrame(rows), models


def metric_record(frame):
    baseline = float(frame.training_mean_abs_error.mean())
    error = float(frame.abs_error.mean())
    return {"unique_test_cells": frame.cell_uid.nunique(), "observations": len(frame),
            "mae": error, "signed_bias": float(frame.signed_error.mean()),
            "training_mean_mae": baseline,
            "skill_vs_training_mean": 1 - error / baseline if baseline > 0 else np.nan,
            "spearman": safe_spearman(frame.truth.to_numpy(), frame.prediction.to_numpy())}


def regression_summaries(predictions):
    summary, protocols = [], []
    for (task, model), frame in predictions.groupby(["task", "model"], sort=True):
        group_rows = []
        for protocol, group in frame.groupby("protocol_uid", sort=True):
            row = {"task": task, "model": model, "protocol_uid": protocol, **metric_record(group)}
            protocols.append(row)
            group_rows.append(row)
        summary.append({"task": task, "model": model, **metric_record(frame),
                        "protocol_macro_mae": float(np.mean([row["mae"] for row in group_rows])),
                        "supported_protocols": len(group_rows)})
    left = predictions.loc[predictions.model == "C0"]
    right = predictions.loc[predictions.model == "C1"]
    pairs = left.merge(right, on=["task", "split", "cell_uid", "protocol_uid"],
                       suffixes=("_C0", "_C1"), validate="one_to_one")
    if not np.array_equal(pairs.truth_C0, pairs.truth_C1):
        raise RuntimeError("Paired labels disagree")
    pairs["improvement_abs_error"] = pairs.abs_error_C0 - pairs.abs_error_C1
    pairs["difference_signed_error"] = pairs.signed_error_C1 - pairs.signed_error_C0
    comparisons = []
    for task, frame in pairs.groupby("task", sort=True):
        for unit, values in (("cell", frame.improvement_abs_error.to_numpy()),
                             ("protocol_group", frame.groupby("protocol_uid").improvement_abs_error.mean().to_numpy())):
            comparisons.append({"task": task, "sampling_unit": unit,
                                "paired_cells": len(frame), "protocol_groups": frame.protocol_uid.nunique(),
                                "direction": "C0_abs_error_minus_C1_abs_error_positive_favors_QV",
                                **paired_mean_bootstrap(values),
                                "bootstrap_seed": 42, "bootstrap_replicates": 2000,
                                "scope": "conditional_on_fitted_predictions_not_training_seed_variance"})
    return pd.DataFrame(summary), pd.DataFrame(protocols), pairs, pd.DataFrame(comparisons)


def hinge_proxy(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    count = len(x)
    span = float(x[-1] - x[0]) if count else 0.
    result = {"status": "INSUFFICIENT_SUPPORT", "raw_observations": count, "raw_cycle_span": span,
              "certified_physical_knee": False, "conditional_linear_coefficients": 3,
              "free_hinge_parameter_count": 4, "quadratic_parameter_count": 3}
    if count < 240 or span < 300:
        return result
    if count > 500:
        choose = np.unique(np.round(np.linspace(0, count - 1, 500)).astype(int))
        x, y = x[choose], y[choose]
    center, scale = x.mean(), max(float(x.std()), 1.)
    z = (x - center) / scale
    quadratic = np.column_stack((np.ones(len(x)), z, z * z))
    beta = np.linalg.lstsq(quadratic, y, rcond=None)[0]
    quadratic_sse = float(np.square(y - quadratic @ beta).sum())
    left, right = int(len(x) * .2), int(len(x) * .8)
    candidates = np.unique(x[np.linspace(left, right, 75).astype(int)])
    fits = []
    for candidate in candidates:
        design = np.column_stack((np.ones(len(x)), z, np.maximum((x - candidate) / scale, 0)))
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        sse = float(np.square(y - design @ coefficients).sum())
        fits.append((float(candidate), sse, float(coefficients[1] / scale),
                     float((coefficients[1] + coefficients[2]) / scale)))
    selected = min(fits, key=lambda row: row[1])
    compatible = [row[0] for row in fits if row[1] <= selected[1] * 1.05 + 1e-12]
    ratio = np.log(max(quadratic_sse, 1e-16) / max(selected[1], 1e-16))
    result.update(status="FITTED_UNVALIDATED_PROXY", fitted_observations=len(x),
                  kappa_proxy=selected[0], knee_proxy_center_cycle=selected[0],
                  profile_lower_cycle=min(compatible), profile_upper_cycle=max(compatible),
                  profile_width_5pct_SSE=max(compatible) - min(compatible),
                  candidate_count=len(candidates), candidate_grid_min=float(candidates[0]),
                  candidate_grid_max=float(candidates[-1]),
                  median_candidate_spacing_cycles=float(np.median(np.diff(candidates))),
                  profile_reaches_left_grid=bool(min(compatible) == candidates[0]),
                  profile_reaches_right_grid=bool(max(compatible) == candidates[-1]),
                  hinge_sse=selected[1], quadratic_sse=quadratic_sse,
                  pre_SOHSlope=selected[2], post_SOHSlope=selected[3],
                  conditional_three_coefficient_log_sse_ratio_n=float(len(x) * ratio),
                  bic_quad_minus_free_hinge=float(len(x) * ratio - np.log(len(x))),
                  bic_quad_minus_hinge=float(len(x) * ratio - np.log(len(x))))
    return result


def knee_diagnostic(targets, observed, identifiers):
    rows = []
    for i, uid in enumerate(identifiers):
        cycles = np.flatnonzero(observed[i]) + 101
        values = targets[i, observed[i]]
        for start in (101, 201, 301):
            use = cycles >= start
            rows.append({"cell_uid": uid, "fit_start_cycle": start, **hinge_proxy(cycles[use], values[use])})
    return pd.DataFrame(rows)


def smooth_negative_controls():
    rows = []
    cycles = np.arange(101, 2001, dtype=float)
    functions = {"smooth_quadratic": 1.02 - 2e-5 * (cycles - 100) - 6e-8 * (cycles - 100)**2,
                 "smooth_exponential": 1.02 * np.exp(-.00015 * (cycles - 100))}
    for name, values in functions.items():
        for start in (101, 201, 301):
            use = cycles >= start
            rows.append({"synthetic_source": name, "fit_start_cycle": start,
                         "synthetic_true_knee_event": False, **hinge_proxy(cycles[use], values[use])})
    return pd.DataFrame(rows)


def write_csv(path, frame):
    temporary = path.with_suffix(path.suffix + ".pending")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    data = load_frozen_data(args.data)
    inputs, ids, eligible, cells, query, targets, observed, rates, mask, split, freeze = data
    if len(ids) != 169 or not eligible.all():
        raise RuntimeError("Frozen MATR169 qualification failed")
    designs, audit = early_features(inputs, ids)
    supported = mask[:, 0] & mask[:, 1]
    mid, late = rates[supported, 0].astype(float), rates[supported, 1].astype(float)
    contract = {
        "status": "REGISTERED_RETROSPECTIVE_EXPLORATORY",
        "upstream_source_commit": SOURCE_COMMIT, "source_sha256": sha256(__file__),
        "data_freeze_sha256": sha256(args.data / "DATA_FREEZE.json"), "frozen_files": freeze["files"],
        "split_sha256": sha256(args.data / "split_contract.json"), "training_seed": 42,
        "fixed_outer_splits": list(SPLITS), "n_original_MATR": 169,
        "feature_cutoff_original_cycle": 100, "feature_names": feature_names(),
        "feature_dimensions": {key: value.shape[1] for key, value in designs.items()},
        "target_units": "SOH_fraction_per_real_cycle", "regression_fitting_target_multiplier": 1000,
        "alphas": list(ALPHAS), "selection": "strict_first_min_validation_MAE",
        "preprocessing_fit_roles": ["train"], "final_coefficient_fit_roles": ["train"],
        "validation_tunes_alpha_only": True, "train_plus_validation_refit": False,
        "planned_ridge_candidate_fits": 200, "planned_selected_ridge_models": 40,
        "skill_definition": "1 - OOF_model_MAE / OOF_MAE_of_each_fold_training_target_mean",
        "shared_cohort_warning": "The42 shared-label cells are selected by follow-up long enough to observe1000cycles.",
        "target_support": {task: int(support.sum()) for task, (_, support) in targets_by_task(rates, mask).items()},
        "shared_label_statistics": {"positive_late_minus_mid": int((late > mid).sum()),
                                    "median_late_over_mid": float(np.median(late / mid)),
                                    "pearson_mid_late": float(pearsonr(mid, late).statistic)},
        "knee_true_events_certified": 0, "knee_fitting_starts": [101, 201, 301],
        "knee_minimum_observations": 240, "knee_minimum_span_cycles": 300,
        "knee_maximum_subsampled_points": 500, "knee_candidate_count": 75,
        "knee_candidate_quantile_range": [.2, .8], "knee_profile_relative_SSE_limit": 1.05,
        "knee_profile_absolute_SSE_allowance": 1e-12,
        "hinge_dof_correction": "Fixed-position hinge has3 linear coefficients; searching position adds1. Quadratic has3. Corrected BIC includes log(n) penalty difference.",
        "knee_semantics": "Unsupervised future-trajectory proxy, never supplied to M5 or early-rate regressions.",
        "M5_results_used_for_selection": False, "formal_deep_model_fits": 0,
        "implementation_corrections": ["train-only preprocessing and coefficients after alpha choice",
                                      "all507 cell/start qualification rows retained",
                                      "paired bias, training-mean skill, Spearman and protocol metrics",
                                      "explicit breakpoint-search degree of freedom"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "BLAS_threads": 1},
        "scientific_limit": "Failure of this feature/model pair does not establish physical non-identifiability.",
        "historical_exposure": "Same169 historically exposed cells, single-seed internal retrospective study.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with execution_lock(args.output / ".identifiability.lock"):
        contract_path = args.output / "IDENTIFIABILITY_CONTRACT.json"
        if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Existing registered analysis has different source/data/settings")
        write_json(contract_path, contract)
        completion = args.output / "ANALYSIS_COMPLETE.json"
        if completion.exists():
            previous = json.loads(completion.read_text(encoding="utf-8"))
            for item in previous["files"]:
                if sha256(args.output / item["path"]) != item["sha256"]:
                    raise RuntimeError("Completed identifiability artifact changed")
            print(json.dumps({"status": "VERIFIED_EXISTING_COMPLETE"}), flush=True)
            return
        with threadpool_limits(limits=1):
            predictions, models = rate_regression(designs, ids, rates, mask, split, cells)
            summary, protocols, pairs, comparisons = regression_summaries(predictions)
            knee = knee_diagnostic(targets, observed, ids)
            controls = smooth_negative_controls()
        tables = {"rate_predictability_oof.csv": predictions, "rate_predictability_summary.csv": summary,
                  "rate_predictability_protocol_metrics.csv": protocols,
                  "rate_predictability_paired_cells.csv": pairs,
                  "rate_predictability_paired_summary.csv": comparisons,
                  "early_feature_audit.csv": audit, "knee_proxy_stability.csv": knee,
                  "knee_proxy_negative_controls.csv": controls}
        for name, table in tables.items():
            write_csv(args.output / name, table)
        write_json(args.output / "SELECTED_RIDGE_MODELS.json", {"models": models, "candidate_fits": len(models) * len(ALPHAS)})
        write_json(completion, {"status": "ANALYZED", "source_sha256": sha256(__file__),
                                "contract_sha256": sha256(contract_path), "ridge_candidate_fits": len(models) * len(ALPHAS),
                                "selected_ridge_models": len(models), "knee_proxy_rows": len(knee),
                                "files": [{"path": name, "sha256": sha256(args.output / name)} for name in
                                          list(tables) + ["SELECTED_RIDGE_MODELS.json"]]})
    print(json.dumps({"status": "ANALYZED", "ridge_candidate_fits": len(models) * len(ALPHAS),
                      "knee_proxy_rows": len(knee)}), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
