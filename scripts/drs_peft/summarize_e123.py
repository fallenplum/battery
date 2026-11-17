"""Independently score sealed E1--E3 source outer folds and frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .e123_protocol import WINDOWS, load_protocol

ROOT = Path(__file__).resolve().parents[2]
MODEL_KEYS = ["stage", "method", "mode", "radius", "evaluation_mode"]
PAIR_KEYS = ["fold_id", "seed", "cell_uid", "protocol_uid"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def statistic_seed(base, label):
    return int(hashlib.sha256(f"{base}|{label}".encode()).hexdigest()[:16], 16)


def unit_relative_path(unit):
    path = Path(f"fold{unit['fold_id']}") / f"seed{unit['seed']}" / unit["stage"]
    if unit["stage"] != "warmup":
        path /= unit["method"]
    if unit["stage"] == "condition":
        radius = format(float(unit["radius"]), "g").replace(".", "p")
        path = path / ("radius" + radius) / unit["mode"]
    return path


def expected_outer_units(config, protocol):
    result = []
    for fold in protocol["folds"]:
        for seed in config["training_seeds"]:
            for method in config["methods"]:
                base = {"fold_id": fold["fold_id"], "seed": seed, "method": method}
                result.append((dict(base, stage="static"), "static"))
                radii = [config["primary_radius"]]
                if method == "grad_core":
                    radii.extend(config["grad_radius_sensitivity"])
                for radius in radii:
                    for mode in config["modes"]:
                        unit = dict(
                            base, stage="condition", radius=float(radius), mode=mode
                        )
                        result.extend(
                            (unit, evaluation)
                            for evaluation in config["evaluation_modes"]
                        )
    return result


def check_complete_outer(args, config, protocol):
    if config["fold_payload_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("The experiment config and frozen fold payload differ")
    opened = read_json(args.public_root / "OUTER_EVALUATION_OPENED.json")
    if opened["final_target_query_labels_used"] != 0:
        raise RuntimeError("Final target labels entered the outer evaluation barrier")
    expected_training = config["expected_training_units"]["total"]
    if len(opened["source_units"]) != expected_training:
        raise RuntimeError(
            "Outer evaluation lacks the full registered training closure"
        )
    expected = expected_outer_units(config, protocol)
    expected_source_units = {
        json.dumps(unit, sort_keys=True): unit for unit, _ in expected
    }
    for fold in protocol["folds"]:
        for seed in config["training_seeds"]:
            unit = {"stage": "warmup", "fold_id": fold["fold_id"], "seed": seed}
            expected_source_units[json.dumps(unit, sort_keys=True)] = unit
    actual_source_units = {
        json.dumps(item["unit"], sort_keys=True): item
        for item in opened["source_units"]
    }
    if set(expected_source_units) != set(actual_source_units):
        raise RuntimeError("The outer barrier omits registered source unit identities")
    actual_updates = 0
    for key, unit in expected_source_units.items():
        folder = args.private_root / unit_relative_path(unit)
        marker = folder / "SOURCE_DEVELOPMENT_COMPLETE.json"
        entry = actual_source_units[key]
        receipt = read_json(marker)
        run_config = read_json(folder / "RUN_CONFIG.json")
        fold = protocol["folds"][unit["fold_id"]]
        expected_epochs = config["epochs"][unit["stage"]]
        expected_updates = expected_epochs * math.ceil(
            len(fold["fit_ids"]) / config["batch_size"]
        )
        if (
            file_hash(marker) != entry["source_receipt_sha256"]
            or file_hash(folder / "CHECKPOINT_SELECTED.pt")
            != entry["checkpoint_sha256"]
            or run_config["training_ids"] != fold["fit_ids"]
            or run_config["validation_ids"] != fold["inner_validation_ids"]
            or receipt["target_query_labels_used"] != 0
            or receipt["status"] != "SOURCE_DEVELOPMENT_COMPLETE"
            or receipt["epochs"] != expected_epochs
            or receipt["updates"] != expected_updates
        ):
            raise RuntimeError(
                "Fold-local source closure or fit/inner provenance changed"
            )
        actual_updates += receipt["updates"]
    if actual_updates != config["expected_optimizer_updates"]:
        raise RuntimeError("The source optimizer update budget remains incomplete")
    expected_count = (
        config["expected_training_units"]["static"]
        + 2 * config["expected_training_units"]["condition"]
    )
    if len(expected) != expected_count:
        raise RuntimeError("Outer matrix differs from the frozen source experiment")
    config_sha = file_hash(args.config)
    records = []
    for unit, evaluation in expected:
        folder = args.private_root / "outer" / unit_relative_path(unit) / evaluation
        marker = folder / "OUTER_COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError("Outer matrix remains incomplete at " + str(folder))
        receipt = read_json(marker)
        if (
            receipt["unit"] != unit
            or receipt["evaluation_mode"] != evaluation
            or receipt["final_target_query_labels_used"] != 0
            or receipt["optimizer_updates"] != 0
            or receipt["registration_sha256"] != config_sha
        ):
            raise RuntimeError("An outer receipt violates its registered identity")
        fold = protocol["folds"][unit["fold_id"]]
        if (
            receipt["outer_ids"] != fold["outer_test_ids"]
            or receipt["outer_groups"] != fold["outer_test_groups"]
        ):
            raise RuntimeError("Outer scoring identities differ from the group freeze")
        for item in receipt["files"]:
            if file_hash(folder / item["path"]) != item["sha256"]:
                raise RuntimeError("A completed outer artifact changed")
        published = args.public_root / "outer" / unit_relative_path(unit) / evaluation
        if read_json(published / "OUTER_COMPLETE.json") != receipt:
            raise RuntimeError("Public and private outer receipts differ")
        records.append(
            {
                "unit": unit,
                "evaluation_mode": evaluation,
                "folder": folder,
                "marker_sha256": file_hash(marker),
            }
        )
    return records


def load_source_reference(data_root, protocol):
    cache = Path(data_root) / "MATR"
    if (
        file_hash(cache / "COMPLETE.json")
        != protocol["provenance"]["source_cache_complete_sha256"]
    ):
        raise RuntimeError("The frozen source data cache changed")
    with np.load(cache / "arrays.npz", allow_pickle=False) as archive:
        ids = archive["ids"].astype(str).tolist()
        lookup = {uid: row for row, uid in enumerate(ids)}
        rows = np.asarray([lookup[uid] for uid in protocol["source_ids"]])
        reference = {
            "ids": protocol["source_ids"],
            "query": archive["query"],
            "target": archive["targets"][rows],
            "observed": archive["observed"][rows],
            "protocols": archive["protocol_ids"].astype(str)[rows],
        }
    reference["lookup"] = {uid: row for row, uid in enumerate(reference["ids"])}
    return reference


def recompute_unit_metrics(record, reference):
    folder, unit = record["folder"], record["unit"]
    with np.load(folder / "OUTER_PREDICTIONS.npz", allow_pickle=False) as archive:
        ids = archive["ids"].astype(str).tolist()
        if ids != read_json(folder / "OUTER_COMPLETE.json")["outer_ids"]:
            raise RuntimeError("Outer prediction NPZ omits registered physical cells")
        query, prediction, target, observed = (
            archive[name] for name in ("query", "prediction", "target", "observed")
        )
        positions = np.asarray([reference["lookup"][uid] for uid in ids])
        if (
            not np.array_equal(query, reference["query"])
            or not np.array_equal(
                target, reference["target"][positions], equal_nan=True
            )
            or not np.array_equal(observed, reference["observed"][positions])
            or not np.isfinite(prediction).all()
            or not np.isfinite(target[observed]).all()
        ):
            raise RuntimeError(
                "Outer NPZ supervision differs from its frozen source cache"
            )
        records = []
        for index, uid in enumerate(ids):
            for label, lower, upper in WINDOWS:
                valid = observed[index] & (query >= lower) & (query <= upper)
                errors = prediction[index, valid].astype(np.float64) - target[
                    index, valid
                ].astype(np.float64)
                records.append(
                    {
                        "cell_uid": uid,
                        "protocol_uid": str(reference["protocols"][positions[index]]),
                        "window": label,
                        "n_obs": int(valid.sum()),
                        "mae": float(np.abs(errors).mean()) if len(errors) else np.nan,
                        "rmse": float(np.sqrt(np.mean(errors**2)))
                        if len(errors)
                        else np.nan,
                        "bias": float(errors.mean()) if len(errors) else np.nan,
                    }
                )
    actual = pd.DataFrame(records)
    reported = pd.read_csv(folder / "OUTER_CELL_METRICS.csv")
    paired = actual.merge(
        reported,
        on=["cell_uid", "protocol_uid", "window"],
        suffixes=("_actual", "_reported"),
        validate="one_to_one",
    )
    if (
        len(paired) != len(actual)
        or len(paired) != len(reported)
        or not paired.n_obs_actual.eq(paired.n_obs_reported).all()
    ):
        raise RuntimeError(
            "Outer CSV has missing physical-cell windows or altered masks"
        )
    max_difference = 0.0
    for metric in ("mae", "rmse", "bias"):
        left, right = (
            paired[metric + "_actual"].to_numpy(),
            paired[metric + "_reported"].to_numpy(),
        )
        if not np.allclose(left, right, atol=1e-12, rtol=1e-12, equal_nan=True):
            raise RuntimeError("Independently recomputed outer metric differs")
        finite = np.isfinite(left)
        if finite.any():
            max_difference = max(
                max_difference, float(np.abs(left[finite] - right[finite]).max())
            )
    for key, value in unit.items():
        actual[key] = value
    actual["mode"] = unit.get("mode", "static")
    actual["radius"] = float(unit.get("radius", 0.0))
    actual["evaluation_mode"] = record["evaluation_mode"]
    return actual, max_difference


def audit_update_norms(record):
    if record["unit"]["stage"] == "static":
        return None
    frame = pd.read_csv(record["folder"] / "UPDATE_DIAGNOSTICS.csv")
    expected = (
        len(read_json(record["folder"] / "OUTER_COMPLETE.json")["outer_ids"]) * 12
    )
    outer_ids = read_json(record["folder"] / "OUTER_COMPLETE.json")["outer_ids"]
    if (
        len(frame) != expected
        or frame.duplicated(["cell_uid", "layer"]).any()
        or set(frame.cell_uid) != set(outer_ids)
        or frame.groupby("cell_uid").layer.nunique().ne(12).any()
    ):
        raise RuntimeError("Missing cell-layer update-norm diagnostics")
    if (
        not np.isfinite(frame[["increment_norm", "radius", "norm_reference"]])
        .all()
        .all()
    ):
        raise RuntimeError("Nonfinite physical update-norm diagnostics")
    if (
        frame.increment_norm.lt(0).any()
        or frame.radius.le(0).any()
        or frame.norm_reference.le(0).any()
    ):
        raise RuntimeError(
            "Physical update norms or registered radii have invalid signs"
        )
    matched = record["evaluation_mode"] == "matched_norm"
    unattainable = int(frame.matched_norm_unattainable.astype(bool).sum())
    excess = frame.increment_norm.to_numpy() - frame.radius.to_numpy()
    tolerance = 1e-7 + 1e-5 * frame.radius.to_numpy()
    valid = excess <= tolerance
    if matched:
        valid = (
            np.abs(excess) <= tolerance
        ) & ~frame.matched_norm_unattainable.to_numpy(dtype=bool)
    control = bool(valid.all())
    return dict(
        **record["unit"],
        evaluation_mode=record["evaluation_mode"],
        physical_cell_layers=len(frame),
        valid_norm_control_cell_layers=int(valid.sum()),
        valid_norm_control_coverage=float(valid.mean()),
        norm_control_passed=control,
        matched_norm_unattainable=unattainable,
        max_increment_radius_absolute_difference=float(np.abs(excess).max()),
        max_increment_radius_excess=float(excess.max()),
        mean_increment_norm=float(frame.increment_norm.mean()),
        mean_total_update_norm=float(frame.total_update_norm.mean()),
    )


def aggregate_metrics(table, extra_keys=()):
    keys = [*MODEL_KEYS, *extra_keys, "window"]
    valid = table[table.n_obs.gt(0)]
    result = valid.groupby(keys, as_index=False).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        bias=("bias", "mean"),
        physical_cells=("cell_uid", "nunique"),
        protocol_groups=("protocol_uid", "nunique"),
        training_seeds=("seed", "nunique"),
        outer_folds=("fold_id", "nunique"),
        cell_seed_records=("cell_uid", "size"),
        observation_records=("n_obs", "sum"),
    )
    group_keys = list(dict.fromkeys([*keys, "seed", "protocol_uid"]))
    group = valid.groupby(group_keys, as_index=False).mae.mean()
    macro = (
        group.groupby(keys, as_index=False)
        .mae.mean()
        .rename(columns={"mae": "protocol_macro_mae"})
    )
    distinct = valid.drop_duplicates([*keys, "cell_uid"])
    observations = (
        distinct.groupby(keys, as_index=False)
        .n_obs.sum()
        .rename(columns={"n_obs": "unique_observed_points"})
    )
    return result.merge(macro, on=keys, validate="one_to_one").merge(
        observations, on=keys, validate="one_to_one"
    )


def paired_rows(table, method, radius, evaluation, candidate, window):
    subset = table[
        (table.stage == "condition")
        & table.method.eq(method)
        & table.radius.eq(radius)
        & table.evaluation_mode.eq(evaluation)
        & table.window.eq(window)
        & table.n_obs.gt(0)
    ]
    left = subset[subset["mode"].eq(candidate)]
    right = subset[subset["mode"].eq("constant")]
    pair = left.merge(
        right, on=PAIR_KEYS, suffixes=("_candidate", "_constant"), validate="one_to_one"
    )
    if len(pair) != len(left) or len(pair) != len(right) or not len(pair):
        raise RuntimeError("Missing paired candidate/constant physical-cell scores")
    if not pair.n_obs_candidate.eq(pair.n_obs_constant).all():
        raise RuntimeError("Paired model scores use different observed support")
    pair["improvement"] = pair.mae_constant - pair.mae_candidate
    return pair


def effect_matrix(pair, value, unit):
    if unit == "protocol_uid":
        values = pair.groupby(["seed", unit])[value].mean().unstack(unit)
    else:
        values = pair.pivot(index="seed", columns=unit, values=value)
    values = values.sort_index().sort_index(axis=1)
    if values.isna().any().any():
        raise RuntimeError("Crossed resampling requires complete seed-by-unit pairing")
    return (
        values.to_numpy(dtype=np.float64),
        values.index.tolist(),
        values.columns.tolist(),
    )


def resampling_indices(shape, rng, draws):
    return (
        rng.integers(0, shape[0], size=(draws, shape[0])),
        rng.integers(0, shape[1], size=(draws, shape[1])),
    )


def sampled_means(matrix, seed_indices, unit_indices):
    return matrix[seed_indices[:, :, None], unit_indices[:, None, :]].mean((1, 2))


def crossed_interval(matrix, rng, draws, confidence):
    selected_seeds, selected_units = resampling_indices(matrix.shape, rng, draws)
    values = sampled_means(matrix, selected_seeds, selected_units)
    tail = (1 - confidence) / 2
    return np.quantile(values, [tail, 1 - tail]).tolist()


def relative_gain(reference, candidate):
    denominator = float(np.mean(reference))
    if not denominator > 0:
        raise RuntimeError("Relative MAE gains need a positive constant baseline")
    return float((np.mean(reference) - np.mean(candidate)) / denominator)


def summarize_contrast(
    table, method, radius, evaluation, candidate, window, statistics
):
    pair = paired_rows(table, method, radius, evaluation, candidate, window)
    label = f"{method}|{radius}|{evaluation}|{candidate}|{window}"
    result = {
        "method": method,
        "radius": radius,
        "evaluation_mode": evaluation,
        "candidate": candidate,
        "reference": "constant",
        "window": window,
        "physical_cells": int(pair.cell_uid.nunique()),
        "protocol_groups": int(pair.protocol_uid.nunique()),
        "candidate_mae": float(pair.mae_candidate.mean()),
        "constant_mae": float(pair.mae_constant.mean()),
        "mean_absolute_MAE_improvement": float(pair.improvement.mean()),
        "relative_MAE_improvement": relative_gain(
            pair.mae_constant, pair.mae_candidate
        ),
        "confidence_level": statistics["primary_confidence_level"],
        "bootstrap_draws": statistics["bootstrap_draws"],
    }
    for column, name in (("fold_id", "per_fold"), ("seed", "per_seed")):
        result[name + "_relative_MAE_improvement"] = {
            str(key): relative_gain(value.mae_constant, value.mae_candidate)
            for key, value in pair.groupby(column)
        }
    result["per_fold_seed_relative_MAE_improvement"] = {
        f"fold{fold}|seed{seed}": relative_gain(value.mae_constant, value.mae_candidate)
        for (fold, seed), value in pair.groupby(["fold_id", "seed"])
    }
    for unit in ("cell_uid", "protocol_uid"):
        matrix, seeds, identifiers = effect_matrix(pair, "improvement", unit)
        if len(seeds) != 3:
            raise RuntimeError("All three training seeds are required for inference")
        interval = crossed_interval(
            matrix,
            np.random.default_rng(statistic_seed(statistics["seed"], label + unit)),
            statistics["bootstrap_draws"],
            statistics["primary_confidence_level"],
        )
        result[unit + "_interval"] = interval
        result[unit + "_resampling_units"] = len(identifiers)
    return result


def equivalence_interval(first, second, unit, statistics, label):
    matrices = []
    names = None
    for pair in (first, second):
        for value in ("mae_constant", "mae_candidate"):
            matrix, seeds, identifiers = effect_matrix(pair, value, unit)
            identity = (seeds, identifiers)
            if names is not None and names != identity:
                raise RuntimeError(
                    "Subspace-gain equivalence requires paired seed/unit identities"
                )
            names = identity
            matrices.append(matrix)
    selected_seeds, selected_units = resampling_indices(
        matrices[0].shape,
        np.random.default_rng(statistic_seed(statistics["seed"], label + unit)),
        statistics["bootstrap_draws"],
    )
    means = [
        sampled_means(matrix, selected_seeds, selected_units) for matrix in matrices
    ]
    if (means[0] <= 0).any() or (means[2] <= 0).any():
        raise RuntimeError("An equivalence bootstrap has a zero reference loss")
    values = (1 - means[1] / means[0]) - (1 - means[3] / means[2])
    tail = (1 - statistics["primary_confidence_level"]) / 2
    return np.quantile(values, [tail, 1 - tail]).tolist()


def summarize_equivalence(table, evaluation, candidate, radius, statistics, margin):
    gradient = paired_rows(table, "grad_core", radius, evaluation, candidate, "all")
    weight = paired_rows(table, "weight_svd_core", radius, evaluation, candidate, "all")
    point = relative_gain(
        gradient.mae_constant, gradient.mae_candidate
    ) - relative_gain(weight.mae_constant, weight.mae_candidate)
    result = {
        "candidate": candidate,
        "evaluation_mode": evaluation,
        "radius": radius,
        "gradient_minus_weight_relative_gain": point,
        "equivalence_margin": margin,
        "confidence_level": statistics["primary_confidence_level"],
        "bootstrap_draws": statistics["bootstrap_draws"],
    }
    label = f"equivalence|{evaluation}|{candidate}|{radius}"
    for unit in ("cell_uid", "protocol_uid"):
        interval = equivalence_interval(gradient, weight, unit, statistics, label)
        result[unit + "_interval"] = interval
        result[unit + "_equivalent"] = interval[0] >= -margin and interval[1] <= margin
    result["equivalent"] = (
        result["cell_uid_equivalent"] and result["protocol_uid_equivalent"]
    )
    return result


def scoped_norm_controls(config, candidate, norm_records):
    primary = float(config["primary_radius"])
    definitions = {
        "E1": (
            {"grad_core"},
            {primary, *map(float, config["grad_radius_sensitivity"])},
        ),
        "E2": (set(config["methods"]), {primary}),
        "E3": ({"grad_core", "weight_svd_core"}, {primary}),
    }
    result = {}
    for experiment, (methods, radii) in definitions.items():
        relevant = [
            item
            for item in norm_records
            if item["method"] in methods
            and item["radius"] in radii
            and item["mode"] in ("constant", candidate)
        ]
        expected = (
            len(methods)
            * len(radii)
            * 2
            * len(config["evaluation_modes"])
            * 3
            * len(config["training_seeds"])
        )
        if len(relevant) != expected:
            raise RuntimeError("Incomplete candidate-specific amplitude diagnostics")
        matched = [
            item for item in relevant if item["evaluation_mode"] == "matched_norm"
        ]
        result[experiment] = {
            "valid": all(item["norm_control_passed"] for item in relevant),
            "evaluated_outer_units": len(relevant),
            "physical_cell_layers": sum(
                item["physical_cell_layers"] for item in relevant
            ),
            "valid_norm_control_cell_layers": sum(
                item["valid_norm_control_cell_layers"] for item in relevant
            ),
            "valid_norm_control_coverage": sum(
                item["valid_norm_control_cell_layers"] for item in relevant
            )
            / sum(item["physical_cell_layers"] for item in relevant),
            "matched_norm_control_coverage": sum(
                item["valid_norm_control_cell_layers"] for item in matched
            )
            / sum(item["physical_cell_layers"] for item in matched),
            "matched_norm_unattainable_cell_layers": sum(
                item["matched_norm_unattainable"]
                for item in relevant
                if item["evaluation_mode"] == "matched_norm"
            ),
            "failed_units": [
                {
                    key: item[key]
                    for key in (
                        "fold_id",
                        "seed",
                        "method",
                        "mode",
                        "radius",
                        "evaluation_mode",
                        "matched_norm_unattainable",
                    )
                }
                for item in relevant
                if not item["norm_control_passed"]
            ],
        }
    return result


def candidate_gates(config, candidate, contrasts, equivalence, norm_controls=None):
    primary = float(config["primary_radius"])
    lookup = {
        (
            item["method"],
            float(item["radius"]),
            item["evaluation_mode"],
            item["candidate"],
            item["window"],
        ): item
        for item in contrasts
    }
    equivalence_lookup = {
        (item["candidate"], item["evaluation_mode"]): item for item in equivalence
    }
    e1, e2, e3 = [], [], []
    for evaluation in config["evaluation_modes"]:
        full = lookup[("grad_core", primary, evaluation, candidate, "all")]
        e1.append(
            {
                "evaluation_mode": evaluation,
                "minimum_relative_gain": full["relative_MAE_improvement"]
                >= config["success"]["minimum_relative_full_MAE_improvement"],
                "all_fold_full_positive": all(
                    value > 0
                    for value in full["per_fold_relative_MAE_improvement"].values()
                ),
                "all_seed_full_positive": all(
                    value > 0
                    for value in full["per_seed_relative_MAE_improvement"].values()
                ),
                "cell_interval_positive": full["cell_uid_interval"][0] > 0,
                "protocol_interval_positive": full["protocol_uid_interval"][0] > 0,
                "extra_radii_full_positive": all(
                    lookup[("grad_core", float(radius), evaluation, candidate, "all")][
                        "relative_MAE_improvement"
                    ]
                    > 0
                    for radius in config["grad_radius_sensitivity"]
                ),
            }
        )
        for method in config["methods"]:
            value = lookup[(method, primary, evaluation, candidate, "all")]
            e2.append(
                {
                    "method": method,
                    "evaluation_mode": evaluation,
                    "full_gain_positive": value["relative_MAE_improvement"] > 0,
                }
            )
        e2.append(
            {
                "evaluation_mode": evaluation,
                "gradient_weight_gain_equivalence": equivalence_lookup[
                    (candidate, evaluation)
                ]["equivalent"],
            }
        )
        for method in ("grad_core", "weight_svd_core"):
            full = lookup[(method, primary, evaluation, candidate, "all")]
            tail = lookup[(method, primary, evaluation, candidate, "1001_plus")]
            e3.append(
                {
                    "method": method,
                    "evaluation_mode": evaluation,
                    "all_fold_full_positive": all(
                        value > 0
                        for value in full["per_fold_relative_MAE_improvement"].values()
                    ),
                    "all_fold_tail_positive": all(
                        value > 0
                        for value in tail["per_fold_relative_MAE_improvement"].values()
                    ),
                    "pooled_tail_positive": tail["relative_MAE_improvement"] > 0,
                    "tail_cell_interval_positive": tail["cell_uid_interval"][0] > 0,
                    "tail_protocol_interval_positive": tail["protocol_uid_interval"][0]
                    > 0,
                }
            )

    def passes(records, experiment):
        norm_valid = norm_controls is None or norm_controls[experiment]["valid"]
        return norm_valid and all(
            value
            for record in records
            for value in record.values()
            if isinstance(value, bool)
        )

    result = {
        "candidate": candidate,
        "norm_controls": norm_controls,
        "E1_supported": passes(e1, "E1"),
        "E2_supported": passes(e2, "E2"),
        "E3_supported": passes(e3, "E3"),
        "E1_criteria": e1,
        "E2_criteria": e2,
        "E3_criteria": e3,
    }
    result["TARGET_ADVANCE_ALLOWED"] = (
        result["E1_supported"] and result["E2_supported"] and result["E3_supported"]
    )
    return result


def markdown_table(frame):
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                f"{value:.6f}" if isinstance(value, float) else str(value)
                for value in row
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    path,
    config,
    protocol,
    summary,
    contrasts,
    equivalence,
    gate,
    metric_audit,
    norm_audit,
):
    primary = float(config["primary_radius"])
    outer_sizes = "、".join(
        str(len(fold["outer_test_ids"])) for fold in protocol["folds"]
    )
    tail_sizes = "、".join(
        str(fold["outer_test_support"]["windows"]["1001_plus"]["physical_cells"])
        for fold in protocol["folds"]
    )
    tail_ids = [
        uid
        for uid, value in protocol["source_qualification"].items()
        if value["window_observations"]["1001_plus"] > 0
    ]
    tail_groups = {
        protocol["source_qualification"][uid]["protocol_uid"] for uid in tail_ids
    }
    main = (
        summary[
            (summary.stage == "condition")
            & summary.radius.eq(primary)
            & summary.window.isin(["all", "1001_plus"])
        ]
        .pivot(
            index=["method", "mode", "evaluation_mode"], columns="window", values="mae"
        )
        .reset_index()
    )
    main.columns = ["子空间参数化", "视角", "幅度评价", "1001+ MAE", "完整未来MAE"]
    primary_contrasts = [
        item
        for item in contrasts
        if item["candidate"] in config["primary_candidate_modes"]
        and item["method"] == "grad_core"
        and item["radius"] == primary
        and item["window"] in ("all", "1001_plus")
    ]
    comparison = pd.DataFrame(
        [
            {
                "候选": item["candidate"],
                "幅度评价": item["evaluation_mode"],
                "窗口": item["window"],
                "改善百分比": 100 * item["relative_MAE_improvement"],
                "电芯CI下界": item["cell_uid_interval"][0],
                "电芯CI上界": item["cell_uid_interval"][1],
                "工况CI下界": item["protocol_uid_interval"][0],
                "工况CI上界": item["protocol_uid_interval"][1],
            }
            for item in primary_contrasts
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                key: value
                for key, value in item.items()
                if key
                in (
                    "candidate",
                    "E1_supported",
                    "E2_supported",
                    "E3_supported",
                    "TARGET_ADVANCE_ALLOWED",
                )
            }
            for item in gate["candidates"]
        ]
    )
    equivalence_table = pd.DataFrame(
        [
            {
                "候选": item["candidate"],
                "幅度评价": item["evaluation_mode"],
                "梯度减权重来源收益差": item["gradient_minus_weight_relative_gain"],
                "电芯CI下界": item["cell_uid_interval"][0],
                "电芯CI上界": item["cell_uid_interval"][1],
                "工况CI下界": item["protocol_uid_interval"][0],
                "工况CI上界": item["protocol_uid_interval"][1],
                "等效判据": item["equivalent"],
            }
            for item in equivalence
            if item["candidate"] in config["primary_candidate_modes"]
        ]
    )
    lines = [
        "# E1–E3 工况组交叉验证结果",
        "",
        "## 完成状态与总判定",
        "",
        f"**本轮总判定为 {gate['status']}，目标校准推进条件为 {gate['TARGET_ADVANCE_ALLOWED']}。**",
        "",
        markdown_table(decisions),
        "",
        "absolute与dual在训练前登记为两个主要候选。全部E1、E2、E3判据必须由同一个候选满足，才允许推进正式1/3/5-shot目标校准及原封存跨域评价。本报告只评价源内部外层工况。",
        "",
        "## 数据、训练与评价范围",
        "",
        f"源池共{len(protocol['source_ids'])}颗物理电芯、{len(protocol['source_groups'])}个exact protocol groups，三外层折、三个训练种子。外层电芯数为{outer_sizes}，1001+电芯数为{tail_sizes}。合并OOF完整未来支持{len(protocol['source_ids'])}颗、{len(protocol['source_groups'])}组，1001+支持{len(tail_ids)}颗、{len(tail_groups)}组。所有源电芯在每种配置和种子下只外层评价一次。",
        "",
        f"登记训练单元为{config['expected_training_units']['total']}，其中warmup9、静态核心27、条件化180，正式优化更新预算{config['expected_optimizer_updates']}。全部387个外层结果完成后才运行本汇总。模型、超参数、幅度及结构在外层评分前冻结，选权仅用每折inner-validation。最终MATR目标54颗及跨域Query的未来标签使用量为0。",
        "",
        "历史源池已参与前期研究，本轮属于retrospective嵌套group CV。各折fit有部分重叠，不能将三个折或三个种子视为九个完全独立数据集。归一化、梯度基、Bridge/Head与静态核心均按本折fit/inner重建。",
        "",
        "## 主幅度结果",
        "",
        markdown_table(main),
        "",
        "每种条件化方法仅训练192个delta-core和1248个门参数，共1440。Bridge、Head及静态C0冻结。所有视角从同一参数化的同一C0和单位门开始。绝对与相对视角的标准化仅使用fit，固定电压网格使各视角四个V统计恒定，参数数目相同仍存在有效函数自由度差异。",
        "",
        "trained_ball限制每cell×layer的物理权重增量，matched_norm对同一已选checkpoint的非零增量进行固定范数评价。增量以静态C0为起点，包含核心和门的联合效应。radius1对应每层官方W0 Frobenius范数的1%，Gradient-Core额外训练radius0.25与4。",
        "",
        "## E1 状态条件化与幅度控制",
        "",
        markdown_table(comparison),
        "",
        "区间对象为常数门MAE减去候选MAE，正值对应候选改善。主点估计采用电芯等权，工况区间先组内平均再组等权。置信水平为97.5%，两个主要候选预登记，各区间使用2000次训练seed与电芯或工况的交叉重抽样。区间属于固定训练设置下的描述，保留历史暴露和训练重叠的解释边界。",
        "",
        "E1要求同一候选在Gradient-Core主幅度、两种评价方式均改善至少1%，三个fold及三个seed边际方向均正，两类区间下界均正，额外两个幅度的完整未来方向均正。九个fold×seed的逐项方向保留在E123_CONTRASTS.json。",
        "",
        "## E2 子空间来源与收益等效",
        "",
        markdown_table(equivalence_table),
        "",
        "收益差定义为Gradient-Core的相对改善减去orthogonal weight-SVD核心的相对改善，每种改善由同次重抽样中constant和候选MAE的均值比计算。两类97.5%区间均完全落在±0.02内才满足等效判据，额外要求三参数化、两种评价方式均有正完整未来收益。",
        "",
        "LoRA-XS与orthogonal weight-SVD拥有相同的权重SVD U/V空间，XS吸收Σ作为坐标缩放。三分支覆盖梯度与权重两类方向来源，并包含XS参数化对照。相关收益不能解释为三个独立方向空间的验证。",
        "",
        "## E3 独立工况方向一致性",
        "",
        "E3分别检查Gradient-Core与orthogonal weight-SVD，两种幅度评价下每个外层fold的完整未来与1001+方向均需为正。合并1001+改善与两类区间下界也均需为正。每fold×seed、每组和全部分段结果保存于CSV和逐项gate JSON，负方向保留。",
        "",
        "## 独立核验",
        "",
        f"重新从私有NPZ计算{metric_audit['recomputed_cell_window_rows']}条电芯窗口MAE/RMSE/bias，最大数值差{metric_audit['maximum_metric_difference']:.3g}。原始监督及观察mask与冻结缓存逐值匹配。物理增量控制通过状态为{norm_audit['all_norm_controls_passed']}，matched-norm不可达到的cell×layer数为{norm_audit['matched_norm_unattainable_cell_layers']}。",
        "",
        "若与候选相关配置的某个增量为零，matched-norm无法获得指定半径，该行完整保留并使对应实验的幅度隔离支持判定失败。局部判定按候选、方向来源与实验范围执行，其count、分布与有效覆盖保存在norm审计和gate JSON。计算所有已登记外层结果后统一统计，不以结果选半径或选候选。",
        "",
        "## 优势、风险与置信度",
        "",
        "优势是工况组隔离、折内来源重建、固定适配器起点、相同可训练参数量以及物理权重空间幅度控制，可以共同约束参数容量与优化尺度解释。风险包括115颗源数据历史暴露、各折训练重叠、仅21颗远期支持、恒定视角维和不同参数化的训练动力学。",
        "",
        "执行与核验完整性的置信度高，依据完整训练闭环、387个外层结果、独立NPZ重算与来源摘要。物理因果与外域泛化的解释取决于本轮全部gate支持情况和后续封存目标评价。",
        "",
        "## 拓展思考",
        "",
        "E2若支持两类方向来源获得等效收益，后续机制定位应围绕状态条件化，并保留梯度构基作为来源选择对照。E1/E3的失败项需要与其幅度和工况分布一起保留。",
        "",
        "## 你可能想知道",
        "",
        "本汇总执行0次训练和0次优化更新。最终目标Query未来轨迹在本轮保持封存，目标推进取决于同一主要候选同时满足三个登记判据。",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/drs_peft/e123_v1.json"
    )
    parser.add_argument(
        "--folds-file", type=Path, default=ROOT / "configs/drs_peft/e123_folds_v1.json"
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/e123_v1"
    )
    parser.add_argument(
        "--public-root", type=Path, default=ROOT / "results/drs_peft/e123_v1"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "report/drs_peft/e123/E123_RESULTS_ZH.md"
    )
    args = parser.parse_args()
    config, protocol = read_json(args.config), load_protocol(args.folds_file)
    records = check_complete_outer(args, config, protocol)
    reference = load_source_reference(args.data_root, protocol)
    metrics, norm_records, max_difference = [], [], 0.0
    for record in records:
        unit, difference = recompute_unit_metrics(record, reference)
        metrics.append(unit)
        max_difference = max(max_difference, difference)
        norm = audit_update_norms(record)
        if norm is not None:
            norm_records.append(norm)
    table = pd.concat(metrics, ignore_index=True)
    summary = aggregate_metrics(table)
    contrasts = []
    for method in config["methods"]:
        radii = [float(config["primary_radius"])]
        if method == "grad_core":
            radii.extend(float(value) for value in config["grad_radius_sensitivity"])
        for radius in radii:
            for evaluation in config["evaluation_modes"]:
                for candidate in ("absolute", "relative", "dual"):
                    for window, _, _ in WINDOWS:
                        contrasts.append(
                            summarize_contrast(
                                table,
                                method,
                                radius,
                                evaluation,
                                candidate,
                                window,
                                config["statistics"],
                            )
                        )
    equivalence = [
        summarize_equivalence(
            table,
            evaluation,
            candidate,
            float(config["primary_radius"]),
            config["statistics"],
            config["success"]["E2_gain_equivalence_margin"],
        )
        for evaluation in config["evaluation_modes"]
        for candidate in ("absolute", "relative", "dual")
    ]
    norm_audit = {
        "status": "COMPLETE",
        "outer_conditioned_results": len(norm_records),
        "all_norm_controls_passed": all(
            item["norm_control_passed"] for item in norm_records
        ),
        "matched_norm_unattainable_cell_layers": sum(
            item["matched_norm_unattainable"]
            for item in norm_records
            if item["evaluation_mode"] == "matched_norm"
        ),
        "optimizer_updates": 0,
        "final_target_query_labels_used": 0,
    }
    candidates = [
        candidate_gates(
            config,
            candidate,
            contrasts,
            equivalence,
            scoped_norm_controls(config, candidate, norm_records),
        )
        for candidate in config["primary_candidate_modes"]
    ]
    advance = any(item["TARGET_ADVANCE_ALLOWED"] for item in candidates)
    gate = {
        "study_id": config["study_id"],
        "status": "SUPPORTED_SOURCE_E123" if advance else "NOT_SUPPORTED_SOURCE_E123",
        "candidates": candidates,
        "TARGET_ADVANCE_ALLOWED": advance,
        "source_outer_cells": len(protocol["source_ids"]),
        "source_outer_groups": len(protocol["source_groups"]),
        "completed_training_units": config["expected_training_units"]["total"],
        "completed_outer_results": len(records),
        "optimizer_updates_in_summarizer": 0,
        "final_target_query_labels_used": 0,
        "config_sha256": file_hash(args.config),
        "protocol_file_sha256": file_hash(args.folds_file),
        "summary_source_sha256": file_hash(__file__),
    }
    audit = {
        "status": "PASSED",
        "recomputed_outer_results": len(records),
        "recomputed_cell_window_rows": len(table),
        "maximum_metric_difference": max_difference,
        "supervision_matches_frozen_source": True,
        "optimizer_updates": 0,
        "final_target_query_labels_used": 0,
        "outer_receipts": [
            {
                "unit": record["unit"],
                "evaluation_mode": record["evaluation_mode"],
                "sha256": record["marker_sha256"],
            }
            for record in records
        ],
    }
    outputs = []
    for name, frame in (
        ("E123_CELL_METRICS.csv", table),
        ("E123_MODEL_SUMMARY.csv", summary),
        ("E123_SEED_SUMMARY.csv", aggregate_metrics(table, ["seed"])),
        ("E123_FOLD_SUMMARY.csv", aggregate_metrics(table, ["fold_id"])),
        ("E123_FOLD_SEED_SUMMARY.csv", aggregate_metrics(table, ["fold_id", "seed"])),
        ("E123_PROTOCOL_SUMMARY.csv", aggregate_metrics(table, ["protocol_uid"])),
        ("E123_UPDATE_NORMS.csv", pd.DataFrame(norm_records)),
    ):
        path = args.public_root / name
        frame.to_csv(path, index=False)
        outputs.append(path)
    for name, value in (
        ("E123_CONTRASTS.json", contrasts),
        ("E123_EQUIVALENCE.json", equivalence),
        ("E123_GATE.json", gate),
        ("E123_INDEPENDENT_METRIC_AUDIT.json", audit),
        ("E123_UPDATE_NORM_AUDIT.json", norm_audit),
    ):
        path = args.public_root / name
        save_json(path, value)
        outputs.append(path)
    write_report(
        args.report,
        config,
        protocol,
        summary,
        contrasts,
        equivalence,
        gate,
        audit,
        norm_audit,
    )
    outputs.append(args.report)
    save_json(
        args.public_root / "E123_SUMMARY_MANIFEST.json",
        {
            "status": "COMPLETE",
            "config_sha256": file_hash(args.config),
            "protocol_sha256": protocol["protocol_sha256"],
            "implementation_sha256": file_hash(__file__),
            "outputs": [
                {
                    "path": str(path.resolve()),
                    "sha256": file_hash(path),
                    "bytes": path.stat().st_size,
                }
                for path in outputs
            ],
            "final_target_query_labels_used": 0,
            "optimizer_updates": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": gate["status"],
                "TARGET_ADVANCE_ALLOWED": advance,
                "report": str(args.report),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
