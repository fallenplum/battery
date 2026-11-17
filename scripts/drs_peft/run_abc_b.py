"""Source-only, protocol-held-out diagnostics of early feature information."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from filelock import FileLock
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from models.drs_peft.diagnostic_dualview import (
    SourceViewScaler,
    prefix_view_descriptors,
)
from scripts.drs_peft.data import ROOT, file_hash, load_dataset, write_json
from scripts.drs_peft.e123_protocol import load_protocol, validate_against_data
from scripts.drs_peft.episodes import load_registration

ARMS = ("baseline", "absolute", "relative", "dual", "shuffled_dual")
MODELS = ("ridge", "gbr")
ROLES = ("fit", "inner_validation", "outer_test")
BASELINE_NAMES = (
    "mean_prefix_Qmax_SOH_cycles1_10",
    "mean_prefix_Qmax_SOH_cycles81_100",
    "late_minus_early_prefix_Qmax_SOH",
    "prefix_Qmax_SOH_slope_per100cycles",
    "prefix_Qmax_SOH_std",
    "registered_prefix_anchor_SOH",
)
DEFAULT_SETTINGS = {
    "windows": [[200, 500], [500, 1000]],
    "minimum_observation_fraction": 0.8,
    "edge_width": 20,
    "minimum_edge_observations": 10,
    "arms": list(ARMS),
    "models": list(MODELS),
    "training_seeds": [42, 123, 2026],
    "ridge": {"alpha": 1.0},
    "gbr": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 3,
        "subsample": 1.0,
        "loss": "squared_error",
    },
}


def payload_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Frozen B identity changed at {path}")
    else:
        write_json(path, value)


def read_settings(config):
    settings = read_json(config)["B"]
    for key, expected in DEFAULT_SETTINGS.items():
        if settings.get(key) != expected:
            raise ValueError(f"B registration differs from implemented setting {key}")
    return settings


def role_derangement(ids, seed, fold_id, role):
    """Return one donor per cell, without self-donors or label-dependent choices."""
    ids = list(ids)
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError(
            "Feature shuffling requires at least two unique physical cells"
        )
    ordering = sorted(
        range(len(ids)),
        key=lambda row: hashlib.sha256(
            f"abc-state-shuffle|{seed}|{fold_id}|{role}|{ids[row]}".encode()
        ).digest(),
    )
    donors = np.empty(len(ids), dtype=np.int64)
    for index, row in enumerate(ordering):
        donors[row] = ordering[(index + 1) % len(ordering)]
    return donors


def rate_support(observed, query, lower, upper, settings):
    """Select coverage using only observation masks and physical cycle numbers."""
    observed = np.asarray(observed, dtype=bool)
    query = np.asarray(query, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != len(query):
        raise ValueError("Rate observation masks must match the cycle grid")
    if len(np.unique(query)) != len(query) or np.any(np.diff(query) <= 0):
        raise ValueError("Rate cycle coordinates must be unique and increasing")
    width = int(settings["edge_width"])
    whole = (query >= lower) & (query <= upper)
    early = (query >= lower) & (query < lower + width)
    late = (query > upper - width) & (query <= upper)
    if not whole.any() or not early.any() or not late.any():
        raise ValueError("The registered rate window is outside the cycle grid")
    coverage = observed[:, whole].sum(1)
    minimum = int(
        np.ceil(float(settings["minimum_observation_fraction"]) * whole.sum())
    )
    minimum_edge = int(settings["minimum_edge_observations"])
    eligible = (
        (coverage >= minimum)
        & (observed[:, early].sum(1) >= minimum_edge)
        & (observed[:, late].sum(1) >= minimum_edge)
    )
    return eligible, early, late


def degradation_rates(targets, observed, query, lower, upper, settings):
    """Compute positive SOH decline per cycle from observed endpoint bands."""
    targets = np.asarray(targets, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    query = np.asarray(query, dtype=float)
    if targets.shape != observed.shape:
        raise ValueError("Rate targets must match observation masks")
    eligible, early, late = rate_support(observed, query, lower, upper, settings)
    result = np.full(len(targets), np.nan)
    rows = np.flatnonzero(eligible)
    if not rows.size:
        return result, eligible
    band_values, band_cycles = [], []
    for band in (early, late):
        supported = observed[rows][:, band]
        values = targets[rows][:, band]
        if not np.isfinite(values[supported]).all():
            raise ValueError("A supported rate endpoint is nonfinite")
        counts = supported.sum(1)
        band_values.append(np.where(supported, values, 0).sum(1) / counts)
        band_cycles.append((supported * query[band]).sum(1) / counts)
    duration = band_cycles[1] - band_cycles[0]
    if np.any(duration <= 0):
        raise ValueError("Rate endpoint bands do not have positive time separation")
    result[rows] = (band_values[0] - band_values[1]) / duration
    return result, eligible


def baseline_prefix_features(x, mask, cycle_ids, anchor):
    """Six early capacity summaries, using only supported prefix cycles 1..100."""
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    cycles = np.asarray(cycle_ids, dtype=float)
    anchor = np.asarray(anchor, dtype=float).reshape(-1)
    if x.ndim != 4 or x.shape[1:3] != (100, 64) or x.shape[-1] < 3:
        raise ValueError("Baseline inputs require [physical_cell,100,64,>=3]")
    if mask.shape != x.shape[:-1] or cycles.shape != x.shape[:2]:
        raise ValueError("Baseline prefix coordinates or masks have wrong dimensions")
    valid = mask.any(2)
    if np.any((cycles[valid] < 1) | (cycles[valid] > 100)):
        raise ValueError("Baseline features may use only cycles 1 through 100")
    capacity = np.where(valid, np.where(mask, x[..., 2], -np.inf).max(2), 0)
    if not np.isfinite(capacity[valid]).all() or not np.isfinite(anchor).all():
        raise ValueError("A supported early capacity or anchor is nonfinite")
    columns = []
    for lower, upper in ((1, 10), (81, 100)):
        selected = valid & (cycles >= lower) & (cycles <= upper)
        count = selected.sum(1)
        if np.any(count == 0):
            raise ValueError("An early baseline capacity band has no observations")
        columns.append(np.where(selected, capacity, 0).sum(1) / count)
    delta = columns[1] - columns[0]
    counts = valid.sum(1)
    if np.any(counts < 2):
        raise ValueError("Prefix capacity trends require two observed cycles")
    mean_cycle = np.where(valid, cycles, 0).sum(1) / counts
    centered = np.where(valid, cycles - mean_cycle[:, None], 0)
    denominator = (centered**2).sum(1)
    if np.any(denominator <= 0):
        raise ValueError("Prefix capacity trend cycle variance must be positive")
    slope = (centered * capacity).sum(1) / denominator * 100
    mean_capacity = np.where(valid, capacity, 0).sum(1) / counts
    variance = (
        np.where(valid, (capacity - mean_capacity[:, None]) ** 2, 0).sum(1) / counts
    )
    return np.column_stack((*columns, delta, slope, np.sqrt(variance), anchor))


def rate_audit(data, protocol, settings):
    rows = data.rows(protocol["source_ids"])
    ids = list(protocol["source_ids"])
    groups = {uid: protocol["source_qualification"][uid]["protocol_uid"] for uid in ids}
    windows = {}
    for lower, upper in settings["windows"]:
        eligible, _, _ = rate_support(
            data.observed[rows], data.query, lower, upper, settings
        )
        eligible_ids = {uid for uid, included in zip(ids, eligible) if included}
        folds = []
        for fold in protocol["folds"]:
            roles = {}
            for role in ROLES:
                selected = [uid for uid in fold[role + "_ids"] if uid in eligible_ids]
                roles[role] = {
                    "ids": selected,
                    "physical_cells": len(selected),
                    "protocol_groups": len({groups[uid] for uid in selected}),
                }
            folds.append({"fold_id": fold["fold_id"], "roles": roles})
        windows[f"{lower}_{upper}"] = {
            "source_physical_cells": len(eligible_ids),
            "source_protocol_groups": len({groups[uid] for uid in eligible_ids}),
            "folds": folds,
        }
    return {
        "status": "MASK_ONLY_QUALIFICATION",
        "rate_units": "nominal_SOH_per_original_cycle; positive_is_degradation",
        "rate_definition": "(mean_SOH_first20 - mean_SOH_last20)/(mean_cycle_last20 - mean_cycle_first20)",
        "interpolated_future_values": 0,
        "source_future_SOH_values_read": 0,
        "final_target_query_labels_used": 0,
        "windows": windows,
    }


def make_designs(data, fold, roles, seed):
    """Fit all standardization on the exact same eligible fit cells."""
    ids = [uid for role in ROLES for uid in roles[role]["ids"]]
    rows = data.rows(ids)
    descriptors = prefix_view_descriptors(
        data.x[rows], data.point_mask[rows], data.cycle_ids[rows]
    )
    fit_count = len(roles["fit"]["ids"])
    view_scaler = SourceViewScaler().fit(
        descriptors, np.arange(fit_count), roles["fit"]["ids"]
    )
    views = view_scaler.transform(descriptors)
    baseline = baseline_prefix_features(
        data.x[rows], data.point_mask[rows], data.cycle_ids[rows], data.anchor[rows]
    )
    baseline_scaler = StandardScaler().fit(baseline[:fit_count])
    base = baseline_scaler.transform(baseline)
    combined = np.column_stack((views["absolute"], views["relative"]))
    output, permutations, offset = {}, {}, 0
    for role in ROLES:
        role_ids = roles[role]["ids"]
        count = len(role_ids)
        interval = slice(offset, offset + count)
        current = combined[interval]
        donors = role_derangement(role_ids, seed, fold["fold_id"], role)
        extra = {
            "baseline": np.zeros_like(current),
            "absolute": np.column_stack((current[:, :12], np.zeros((count, 12)))),
            "relative": np.column_stack((np.zeros((count, 12)), current[:, 12:])),
            "dual": current,
            "shuffled_dual": current[donors],
        }
        output[role] = {
            arm: np.column_stack((base[interval], extra[arm])) for arm in ARMS
        }
        permutations[role] = {
            "recipient_ids": role_ids,
            "donor_ids": [role_ids[index] for index in donors],
        }
        offset += count
    manifest = {
        "training_ids": roles["fit"]["ids"],
        "baseline_feature_names": list(BASELINE_NAMES),
        "baseline_capacity_recipe": "maximum_supported_prefix_discharge_Q(V)/nominal_capacity_per_cycle; never_zero_filled_endpoint",
        "baseline_scaler_mean": baseline_scaler.mean_.tolist(),
        "baseline_scaler_scale": baseline_scaler.scale_.tolist(),
        "view_scaler": view_scaler.to_dict(),
        "nominal_design_columns": 30,
        "zero_padding": "absolute_only_or_relative_only_to24; baseline_only_to30",
        "shuffle": permutations,
        "shuffle_mapping_sha256": payload_hash(permutations),
        "input_sha256": {
            role: {
                arm: hashlib.sha256(value.tobytes()).hexdigest()
                for arm, value in designs.items()
            }
            for role, designs in output.items()
        },
        "final_target_descriptors_computed": 0,
    }
    return output, manifest


def make_estimator(name, seed, settings):
    if name == "ridge":
        return Ridge(alpha=float(settings["ridge"]["alpha"]), solver="svd")
    if name == "gbr":
        return GradientBoostingRegressor(random_state=int(seed), **settings["gbr"])
    raise ValueError("Unknown preregistered B model")


def complexity(model, fit_x):
    centered = fit_x - fit_x.mean(0)
    result = {
        "input_columns": int(fit_x.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(centered)),
    }
    if isinstance(model, Ridge):
        singular = np.linalg.svd(centered, compute_uv=False)
        result["ridge_effective_df_including_intercept"] = float(
            1 + np.sum(singular**2 / (singular**2 + model.alpha))
        )
        result["nominal_linear_parameters_including_intercept"] = int(
            fit_x.shape[1] + 1
        )
    else:
        result["fitted_trees"] = len(model.estimators_)
        result["realized_total_leaves"] = int(
            sum(tree[0].tree_.n_leaves for tree in model.estimators_)
        )
        result["maximum_total_leaves"] = int(model.n_estimators * (2**model.max_depth))
    return result


def unit_plan(protocol, settings):
    return [
        {
            "fold_id": fold["fold_id"],
            "seed": seed,
            "window": f"{lower}_{upper}",
            "model": model,
            "arm": arm,
        }
        for fold in protocol["folds"]
        for seed in settings["training_seeds"]
        for lower, upper in settings["windows"]
        for model in MODELS
        for arm in ARMS
    ]


def unit_path(root, unit):
    return (
        Path(root)
        / f"fold{unit['fold_id']}"
        / f"seed{unit['seed']}"
        / unit["window"]
        / unit["model"]
        / unit["arm"]
    )


def metrics(targets, predicted):
    error = np.asarray(predicted) - np.asarray(targets)
    if not error.size or not np.isfinite(error).all():
        raise ValueError("B metrics require finite, nonempty paired observations")
    return {
        "cells": len(error),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
    }


def checked_complete(folder, identity=None):
    receipt = read_json(folder / "TRAIN_COMPLETE.json")
    if identity is not None and receipt["identity"] != identity:
        raise RuntimeError("B training provenance differs from current registration")
    for record in receipt["files"]:
        if file_hash(folder / record["path"]) != record["sha256"]:
            raise RuntimeError("A completed B training file changed")
    if receipt["outer_labels_used_for_training_or_selection"] != 0:
        raise RuntimeError("B fit consumed outer labels")
    return receipt


def train_models(args, protocol, settings, data, audit, plan, identity):
    for fold in protocol["folds"]:
        if fold["fold_id"] not in args.fold_ids:
            continue
        for lower, upper in settings["windows"]:
            window = f"{lower}_{upper}"
            roles = audit["windows"][window]["folds"][fold["fold_id"]]["roles"]
            fit_rows = data.rows(roles["fit"]["ids"])
            inner_rows = data.rows(roles["inner_validation"]["ids"])
            targets = {}
            for role, rows in (("fit", fit_rows), ("inner_validation", inner_rows)):
                rate, eligible = degradation_rates(
                    data.targets[rows],
                    data.observed[rows],
                    data.query,
                    lower,
                    upper,
                    settings,
                )
                if not eligible.all():
                    raise RuntimeError("Registered B rate qualification changed")
                targets[role] = rate
            for seed in settings["training_seeds"]:
                designs, manifest = make_designs(data, fold, roles, seed)
                manifest_path = (
                    args.public_root
                    / f"fold{fold['fold_id']}"
                    / f"seed{seed}"
                    / window
                    / "INPUT_MANIFEST.json"
                )
                bind(manifest_path, manifest)
                for unit in plan:
                    if (unit["fold_id"], unit["seed"], unit["window"]) != (
                        fold["fold_id"],
                        seed,
                        window,
                    ):
                        continue
                    folder = unit_path(args.private_root / "models", unit)
                    public_folder = unit_path(args.public_root / "models", unit)
                    lock_key = payload_hash(unit)
                    lock_path = args.private_root / "_locks" / (lock_key + ".lock")
                    lock_path.parent.mkdir(parents=True, exist_ok=True)
                    with FileLock(str(lock_path)):
                        bound = dict(
                            identity,
                            unit=unit,
                            input_manifest_sha256=file_hash(manifest_path),
                        )
                        if (folder / "TRAIN_COMPLETE.json").exists():
                            checked_complete(folder, bound)
                            continue
                        folder.mkdir(parents=True, exist_ok=True)
                        fit_x = designs["fit"][unit["arm"]]
                        inner_x = designs["inner_validation"][unit["arm"]]
                        model = make_estimator(unit["model"], seed, settings)
                        with threadpool_limits(limits=2):
                            model.fit(fit_x, targets["fit"])
                            inner_prediction = model.predict(inner_x)
                        temporary = folder / "MODEL.pending"
                        joblib.dump(model, temporary)
                        os.replace(temporary, folder / "MODEL.joblib")
                        np.savez_compressed(
                            folder / "INNER_PREDICTIONS.npz",
                            ids=np.asarray(roles["inner_validation"]["ids"]),
                            target=targets["inner_validation"],
                            prediction=inner_prediction,
                        )
                        record = {
                            "identity": bound,
                            "metrics_inner": metrics(
                                targets["inner_validation"], inner_prediction
                            ),
                            "complexity": complexity(model, fit_x),
                            "hyperparameter_search_trials": 0,
                            "outer_labels_used_for_training_or_selection": 0,
                            "final_target_query_labels_used": 0,
                            "files": [
                                {"path": name, "sha256": file_hash(folder / name)}
                                for name in ("MODEL.joblib", "INNER_PREDICTIONS.npz")
                            ],
                        }
                        write_json(folder / "TRAIN_COMPLETE.json", record)
                        write_json(public_folder / "TRAIN_COMPLETE.json", record)
                        print(
                            json.dumps(
                                {
                                    "completed_B_training": unit,
                                    "fit_cells": len(fit_rows),
                                    "inner_cells": len(inner_rows),
                                }
                            ),
                            flush=True,
                        )


def score_models(args, protocol, settings, data, audit, plan, identity):
    completed = []
    for unit in plan:
        folder = unit_path(args.private_root / "models", unit)
        receipt = checked_complete(folder)
        for key, expected in identity.items():
            if receipt["identity"].get(key) != expected:
                raise RuntimeError(
                    "B outer score used a different training implementation"
                )
        completed.append(
            {"unit": unit, "receipt_sha256": file_hash(folder / "TRAIN_COMPLETE.json")}
        )
    bind(
        args.public_root / "OUTER_OPENED.json",
        {
            "training_units": completed,
            "outer_selection_trials": 0,
            "final_target_query_labels_used": 0,
        },
    )
    all_rows = []
    for fold in protocol["folds"]:
        for lower, upper in settings["windows"]:
            window = f"{lower}_{upper}"
            roles = audit["windows"][window]["folds"][fold["fold_id"]]["roles"]
            rows = data.rows(roles["outer_test"]["ids"])
            target, eligible = degradation_rates(
                data.targets[rows],
                data.observed[rows],
                data.query,
                lower,
                upper,
                settings,
            )
            if not eligible.all():
                raise RuntimeError("B outer qualification changed")
            for seed in settings["training_seeds"]:
                designs, manifest = make_designs(data, fold, roles, seed)
                manifest_path = (
                    args.public_root
                    / f"fold{fold['fold_id']}"
                    / f"seed{seed}"
                    / window
                    / "INPUT_MANIFEST.json"
                )
                if read_json(manifest_path) != manifest:
                    raise RuntimeError(
                        "B descriptor or shuffle inputs changed before outer scoring"
                    )
                for unit in plan:
                    if (unit["fold_id"], unit["seed"], unit["window"]) != (
                        fold["fold_id"],
                        seed,
                        window,
                    ):
                        continue
                    folder = unit_path(args.private_root / "models", unit)
                    receipt = checked_complete(folder)
                    if receipt["identity"]["input_manifest_sha256"] != file_hash(
                        manifest_path
                    ):
                        raise RuntimeError("B training manifest binding changed")
                    model = joblib.load(folder / "MODEL.joblib")
                    prediction = model.predict(designs["outer_test"][unit["arm"]])
                    private_outer = unit_path(args.private_root / "outer", unit)
                    private_outer.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        private_outer / "PREDICTIONS.npz",
                        ids=np.asarray(roles["outer_test"]["ids"]),
                        target=target,
                        prediction=prediction,
                    )
                    public_outer = unit_path(args.public_root / "outer", unit)
                    write_json(
                        public_outer / "OUTER_COMPLETE.json",
                        {
                            "unit": unit,
                            "training_receipt_sha256": file_hash(
                                folder / "TRAIN_COMPLETE.json"
                            ),
                            "input_manifest_sha256": file_hash(manifest_path),
                            "private_prediction_sha256": file_hash(
                                private_outer / "PREDICTIONS.npz"
                            ),
                            "metrics": metrics(target, prediction),
                            "optimization_updates": 0,
                            "final_target_query_labels_used": 0,
                        },
                    )
                    for index, uid in enumerate(roles["outer_test"]["ids"]):
                        all_rows.append(
                            dict(
                                unit,
                                cell_uid=uid,
                                protocol_uid=protocol["source_qualification"][uid][
                                    "protocol_uid"
                                ],
                                target_rate=float(target[index]),
                                predicted_rate=float(prediction[index]),
                                absolute_error=float(
                                    abs(target[index] - prediction[index])
                                ),
                            )
                        )
    frame = pd.DataFrame(all_rows)
    private_csv = args.private_root / "B_CELL_PREDICTIONS.csv"
    frame.to_csv(private_csv, index=False)
    public_frame = frame.drop(columns=["target_rate", "predicted_rate"])
    public_frame.to_csv(args.public_root / "B_CELL_ERRORS.csv", index=False)
    write_json(
        args.public_root / "B_SCORE_COMPLETE.json",
        {
            "status": "COMPLETE",
            "training_units": len(plan),
            "outer_units": len(plan),
            "cell_error_rows": len(frame),
            "private_predictions_sha256": file_hash(private_csv),
            "public_errors_sha256": file_hash(args.public_root / "B_CELL_ERRORS.csv"),
            "final_target_query_labels_used": 0,
        },
    )


def summarize(args, protocol, settings, plan, identity):
    complete = read_json(args.public_root / "B_SCORE_COMPLETE.json")
    frame = pd.read_csv(args.private_root / "B_CELL_PREDICTIONS.csv")
    if (
        file_hash(args.private_root / "B_CELL_PREDICTIONS.csv")
        != complete["private_predictions_sha256"]
    ):
        raise RuntimeError("B outer prediction CSV changed")
    independent = []
    for unit in plan:
        receipt = read_json(
            unit_path(args.public_root / "outer", unit) / "OUTER_COMPLETE.json"
        )
        prediction_path = (
            unit_path(args.private_root / "outer", unit) / "PREDICTIONS.npz"
        )
        if file_hash(prediction_path) != receipt["private_prediction_sha256"]:
            raise RuntimeError("B stored outer predictions changed")
        with np.load(prediction_path, allow_pickle=False) as archive:
            recomputed = metrics(archive["target"], archive["prediction"])
        if recomputed != receipt["metrics"]:
            raise RuntimeError("B metrics failed independent prediction replay")
        independent.append({"unit": unit, "metrics": recomputed})
    fold_summary = frame.groupby(
        ["window", "model", "arm", "fold_id", "seed"], as_index=False
    ).agg(
        MAE=("absolute_error", "mean"),
        physical_cells=("cell_uid", "nunique"),
        protocol_groups=("protocol_uid", "nunique"),
    )
    fold_summary.to_csv(args.public_root / "B_FOLD_SEED_SUMMARY.csv", index=False)
    pooled = frame.groupby(["window", "model", "arm"], as_index=False).agg(
        MAE=("absolute_error", "mean"),
        physical_cells=("cell_uid", "nunique"),
        protocol_groups=("protocol_uid", "nunique"),
    )
    pooled.to_csv(args.public_root / "B_MODEL_SUMMARY.csv", index=False)
    contrasts = []
    for window, model in (
        frame[["window", "model"]].drop_duplicates().itertuples(index=False)
    ):
        subset = frame[(frame.window == window) & (frame.model == model)]
        wide = subset.pivot(
            index=["fold_id", "seed", "cell_uid", "protocol_uid"],
            columns="arm",
            values="absolute_error",
        ).reset_index()
        for candidate in ("absolute", "relative", "dual"):
            for reference in ("baseline", "shuffled_dual"):
                difference = wide[reference] - wide[candidate]
                comparison = wide.assign(difference=difference)
                fold_directions = comparison.groupby("fold_id").difference.mean()
                seed_directions = comparison.groupby("seed").difference.mean()
                group_directions = comparison.groupby("protocol_uid").difference.mean()
                contrasts.append(
                    {
                        "window": window,
                        "model": model,
                        "candidate": candidate,
                        "reference": reference,
                        "reference_MAE": float(wide[reference].mean()),
                        "candidate_MAE": float(wide[candidate].mean()),
                        "MAE_difference": float(difference.mean()),
                        "relative_improvement_percent": float(
                            100 * difference.mean() / wide[reference].mean()
                        ),
                        "fold_MAE_differences": {
                            str(key): float(value)
                            for key, value in fold_directions.items()
                        },
                        "seed_MAE_differences": {
                            str(key): float(value)
                            for key, value in seed_directions.items()
                        },
                        "protocol_group_equal_weight_MAE_difference": float(
                            group_directions.mean()
                        ),
                        "all_fold_directions_positive": bool(
                            (fold_directions > 0).all()
                        ),
                        "all_seed_directions_positive": bool(
                            (seed_directions > 0).all()
                        ),
                    }
                )
    write_json(
        args.public_root / "B_CONTRASTS.json",
        {
            "comparisons": contrasts,
            "uncertainty_intervals_computed": 0,
            "historical_source_exposure": True,
            "independent_physical_samples_not_seed_replicates": True,
        },
    )
    dual = [value for value in contrasts if value["candidate"] == "dual"]
    consistent = all(
        value["all_fold_directions_positive"]
        and value["all_seed_directions_positive"]
        and value["protocol_group_equal_weight_MAE_difference"] > 0
        for value in dual
    )
    gate = {
        "B_dual_incremental_information_consistent": bool(consistent),
        "definition": "dual_beats_baseline_and_shuffled_in_each_fold_and_each_seed_with_positive_group_equal_weight_gain_for_both_windows_and_both_models",
        "evidence_scope": "diagnostic_retrospective_source_group_cv",
        "n_models": len(plan),
        "final_target_query_labels_used": 0,
    }
    write_json(args.public_root / "B_GATE.json", gate)
    write_json(
        args.public_root / "B_INDEPENDENT_METRIC_AUDIT.json",
        {
            "status": "PASSED",
            "outer_units": len(independent),
            "recomputed_cell_predictions": len(frame),
            "outer_metrics": independent,
            "identity": identity,
        },
    )
    outputs = [
        "B_CELL_ERRORS.csv",
        "B_FOLD_SEED_SUMMARY.csv",
        "B_MODEL_SUMMARY.csv",
        "B_CONTRASTS.json",
        "B_GATE.json",
        "B_INDEPENDENT_METRIC_AUDIT.json",
    ]
    write_json(
        args.public_root / "B_SUMMARY_COMPLETE.json",
        {
            "status": "COMPLETE",
            "identity": identity,
            "gate": gate,
            "files": [
                {"path": name, "sha256": file_hash(args.public_root / name)}
                for name in outputs
            ],
            "final_target_query_labels_used": 0,
        },
    )
    print(json.dumps(gate), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/drs_peft/abc_v1.json"
    )
    parser.add_argument(
        "--folds-file", type=Path, default=ROOT / "configs/drs_peft/e123_folds_v1.json"
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/abc_v1/B"
    )
    parser.add_argument(
        "--public-root", type=Path, default=ROOT / "results/drs_peft/abc_v1/B"
    )
    parser.add_argument(
        "--phase", choices=("audit", "train", "score", "summarize"), required=True
    )
    parser.add_argument(
        "--fold-ids", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2]
    )
    args = parser.parse_args()
    settings = read_settings(args.config)
    protocol = load_protocol(args.folds_file)
    plan = unit_plan(protocol, settings)
    identity = {
        "registration_sha256": file_hash(args.config),
        "protocol_file_sha256": file_hash(args.folds_file),
        "protocol_payload_sha256": protocol["protocol_sha256"],
        "implementation_sha256": file_hash(__file__),
        "feature_implementation_sha256": file_hash(
            ROOT / "models/drs_peft/diagnostic_dualview.py"
        ),
        "data_adapter_implementation_sha256": file_hash(
            ROOT / "scripts/drs_peft/data.py"
        ),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "scientific_source_commit": os.environ.get("BATTERY_SOURCE_COMMIT"),
        "data_complete_sha256": file_hash(args.data_root / "MATR/COMPLETE.json"),
        "formal_training_units": len(plan),
        "new_hyperparameter_search_trials": 0,
        "final_target_query_labels_used": 0,
    }
    args.public_root.mkdir(parents=True, exist_ok=True)
    args.private_root.mkdir(parents=True, exist_ok=True)
    bind(
        args.public_root / "B_REGISTRATION.json",
        {"identity": identity, "settings": settings, "plan": plan},
    )
    if args.phase == "summarize":
        summarize(args, protocol, settings, plan, identity)
        return
    data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
    validate_against_data(protocol, data, load_registration())
    audit = rate_audit(data, protocol, settings)
    bind(args.public_root / "B_MASK_SUPPORT_AUDIT.json", audit)
    if args.phase == "audit":
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    elif args.phase == "train":
        train_models(args, protocol, settings, data, audit, plan, identity)
    else:
        score_models(args, protocol, settings, data, audit, plan, identity)


if __name__ == "__main__":
    main()
