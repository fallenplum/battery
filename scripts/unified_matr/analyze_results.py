"""Frozen seed42 MATR OOF and protocol-heldout analysis after all42 formal runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from scripts.run_experiment_b import execution_lock, write_once

ARMS = ("M0", "M1", "M2", "M3", "M3-NL", "M3-NP", "M3-NR")
CONTRASTS = (("M3", "M0"), ("M1", "M0"), ("M2", "M1"), ("M3", "M2"),
             ("M3", "M3-NL"), ("M3", "M3-NP"), ("M3", "M3-NR"))
SPLITS = tuple(f"cell_fold_{number}" for number in range(5)) + ("unseen_protocol_sensitivity",)
HORIZONS = ((101, 200), (201, 500), (501, 1000), (1001, 1000000))
RATES = (("r200_500", 200, 500), ("r500_1000", 500, 1000))
BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED = 2000, 42
ISSUE_COLUMNS = ("severity", "category", "scope", "split", "arm", "cell_uid", "metric", "contrast", "code", "detail")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_bytes(value):
    return (json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def issue(issues, code, detail, severity="ERROR", category="integrity", **context):
    issues.append({**{key: "" for key in ISSUE_COLUMNS}, **context, "severity": severity,
                   "category": category, "code": code,
                   "detail": json.dumps(json_ready(detail), ensure_ascii=False, sort_keys=True)})


class AnalysisBlocked(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def boolean(values):
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    mapped = values.map({True: True, False: False, 1: True, 0: False, "True": True, "False": False,
                         "true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("An evidence flag contains unsupported or missing values")
    return mapped.astype(bool)


def matrix_delivery(run_root, data_root, issues):
    """Inspect completion and hashes before opening any scientific test table."""
    matrix_path, registration_path = run_root / "MATRIX_COMPLETE.json", run_root / "RUN_REGISTRATION.json"
    if not matrix_path.is_file() or not registration_path.is_file():
        issue(issues, "formal_matrix_incomplete", "All42 formal units and their matrix completion marker are required", category="pending")
        raise AnalysisBlocked("Formal matrix completion is pending")
    matrix, registration = read_json(matrix_path), read_json(registration_path)
    registration_sha = sha256(registration_path)
    if (matrix.get("status") != "COMPLETE" or matrix.get("completed_units") != 42 or matrix.get("seed") != 42
            or matrix.get("registration_sha256") != registration_sha):
        issue(issues, "matrix_completion_contract", matrix)
    if (registration.get("seed") != 42 or registration.get("planned_units") != 42
            or tuple(registration["config"]["models"]) != ARMS):
        issue(issues, "registered_matrix_contract", "Exactly seven arms and six splits under seed42 are required")
    reporting = registration["config"]["reporting"]
    if (reporting["bootstrap_replicates"] != BOOTSTRAP_REPLICATES or reporting["bootstrap_seed"] != BOOTSTRAP_SEED
            or tuple(map(tuple, reporting["horizon_bins"])) != HORIZONS
            or reporting.get("eol80_prediction_query_max_cycle") != 5000):
        issue(issues, "registered_analysis_contract", reporting)
    freeze_path = data_root / "DATA_FREEZE.json"
    if not freeze_path.is_file() or sha256(freeze_path) != registration.get("data_freeze_sha256"):
        issue(issues, "data_freeze_binding", "Run registration must bind the supplied frozen dataset")
    else:
        freeze = read_json(freeze_path)
        if freeze.get("status") != "DATA_FREEZE_PASS" or freeze.get("training_seeds") != [42]:
            issue(issues, "data_freeze_state", freeze.get("status"))
        for entry in freeze["files"]:
            path = data_root / entry["path"]
            if not path.is_file() or sha256(path) != entry["sha256"]:
                issue(issues, "frozen_artifact_hash", entry["path"])
    for source in registration["sources"]:
        path = REPOSITORY / source["path"]
        if not path.is_file() or sha256(path) != source["sha256"]:
            severity = "WARNING" if path.resolve() == Path(__file__).resolve() else "ERROR"
            issue(issues, "registered_source_hash_changed", source["path"], severity=severity)
    units = []
    required = {"UNIT_CONTRACT.json", "TRAIN_SCALER.json", "SELECTION_FROZEN.json", "training.jsonl",
                "test_predictions.parquet", "test_cell_metrics.csv", "test_metrics.json", "test_rate_predictions.csv",
                "test_eol_predictions.csv", "SELECTED.pt", "TRAIN_STATE.pt"}
    # Completion presence for the entire matrix precedes reading unit result hashes.
    missing = [(split, arm) for split in SPLITS for arm in ARMS if not (run_root / split / arm / "COMPLETE.json").is_file()]
    for split, arm in missing:
        issue(issues, "unit_completion_missing", "Formal unit is incomplete", category="pending", split=split, arm=arm)
    if missing or any(row["severity"] == "ERROR" for row in issues):
        raise AnalysisBlocked("Formal matrix or frozen identities are incomplete")
    for split in SPLITS:
        for arm in ARMS:
            root = run_root / split / arm
            complete, contract = read_json(root / "COMPLETE.json"), read_json(root / "UNIT_CONTRACT.json")
            files = {entry["path"]: entry for entry in complete["files"]}
            if (complete.get("status") != "EVALUATED" or complete.get("seed") != 42
                    or contract.get("arm") != arm or contract.get("split_name") != split
                    or contract.get("registration_sha256") != registration_sha
                    or complete.get("successful_updates") != contract.get("expected_updates")
                    or not required.issubset(files)):
                issue(issues, "unit_completion_contract", "Unit identity, fixed update budget or required outputs differ", split=split, arm=arm)
            for entry in complete["files"]:
                path = root / entry["path"]
                if not path.is_file() or sha256(path) != entry["sha256"]:
                    issue(issues, "unit_artifact_hash", entry["path"], split=split, arm=arm)
            selection = read_json(root / "SELECTION_FROZEN.json")
            if selection.get("checkpoint_sha256") != files.get("SELECTED.pt", {}).get("sha256"):
                issue(issues, "selected_weight_binding", "Selected weights differ from the pre-test selection", split=split, arm=arm)
            units.append({"split": split, "arm": arm, "root": root, "contract": contract,
                          "selection": selection, "complete_sha256": sha256(root / "COMPLETE.json")})
    if any(row["severity"] == "ERROR" for row in issues):
        raise AnalysisBlocked("At least one formal unit failed integrity verification")
    return registration, registration_sha, units


def paired_mean_bootstrap(values, replicates=BOOTSTRAP_REPLICATES):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"improvement": None, "ci95_low": None, "ci95_high": None, "bootstrap_supported": 0, "resampling_units": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"improvement": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high),
            "bootstrap_supported": replicates, "resampling_units": len(values)}


def concordance_matrices(outcome_time, event, predicted_time):
    outcome_time, predicted_time = np.asarray(outcome_time, float), np.asarray(predicted_time, float)
    event = np.asarray(event, bool)
    comparable = event[:, None] & (outcome_time[:, None] < outcome_time[None, :])
    score = (predicted_time[:, None] < predicted_time[None, :]).astype(float)
    score += 0.5 * (predicted_time[:, None] == predicted_time[None, :])
    return comparable.astype(float), score * comparable


def harrell_concordance(outcome_time, event, predicted_time):
    comparable, score = concordance_matrices(outcome_time, event, predicted_time)
    pairs = int(comparable.sum())
    return {"harrell_c": float(score.sum() / pairs) if pairs else None, "comparable_pairs": pairs,
            "concordant_pairs": int((score == 1.0).sum()), "prediction_ties": int((score == 0.5).sum()),
            "discordant_pairs": int(pairs - (score == 1.0).sum() - (score == 0.5).sum())}


def eol_bootstrap(outcome_time, event, predictions, protocols, replicates=BOOTSTRAP_REPLICATES):
    n = len(outcome_time)
    groups, inverse = np.unique(np.asarray(protocols, str), return_inverse=True)
    result = {}
    for unit in ("cell", "protocol_group"):
        size = n if unit == "cell" else len(groups)
        if not size:
            result[unit] = {arm: np.full(replicates, np.nan) for arm in predictions}
            continue
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        weights = rng.multinomial(size, np.full(size, 1 / size), size=replicates).astype(float)
        if unit == "protocol_group":
            weights = weights[:, inverse]
        scores = {}
        for arm, predicted in predictions.items():
            comparable, score = concordance_matrices(outcome_time, event, predicted)
            denominator = np.einsum("bi,ij,bj->b", weights, comparable, weights, optimize=True)
            numerator = np.einsum("bi,ij,bj->b", weights, score, weights, optimize=True)
            scores[arm] = np.divide(numerator, denominator, out=np.full(replicates, np.nan), where=denominator > 0)
        result[unit] = scores
    return result


def metric_rows(prediction, test_ids, protocols, scope, split, arm):
    rows = []
    grouped = {uid: frame for uid, frame in prediction.groupby("cell_uid", sort=False)}
    for uid in test_ids:
        frame = grouped.get(uid, prediction.iloc[:0])
        error = frame.prediction.to_numpy(float) - frame.target.to_numpy(float)
        for label, selected in [("trajectory", np.ones(len(frame), bool))] + [
                (f"horizon_{left}_{right}", frame.original_cycle_id.between(left, right).to_numpy()) for left, right in HORIZONS]:
            values = error[selected]
            for name, value in (("mae", float(np.abs(values).mean()) if len(values) else None),
                                ("rmse", float(np.sqrt((values ** 2).mean())) if len(values) else None)):
                rows.append({"scope": scope, "split": split, "arm": arm, "cell_uid": uid,
                             "protocol_uid": protocols[uid], "metric": label + "_" + name,
                             "value": value, "observations": len(values)})
    return rows


def frozen_tables(data_root):
    cells = pd.read_parquet(data_root / "cells.parquet").copy()
    if len(cells) != 169 or cells.cell_uid.duplicated().any():
        raise ValueError("The fixed169 physical-cell qualification denominator differs")
    cells["cell_uid"] = cells.cell_uid.astype(str)
    cells["protocol_uid"] = cells.protocol_uid.astype(str)
    cells["input_eligible"] = boolean(cells.input_eligible)
    cells = cells.set_index("cell_uid", drop=False)
    future = pd.read_parquet(data_root / "future_soh.parquet")
    future = future.loc[boolean(future.capacity_valid) & (future.original_cycle_id > 100),
                        ["cell_uid", "original_cycle_id", "soh_nominal"]].copy()
    future.cell_uid = future.cell_uid.astype(str)
    future.original_cycle_id = future.original_cycle_id.astype(np.int64)
    if future.duplicated(["cell_uid", "original_cycle_id"]).any() or not np.isfinite(future.soh_nominal).all():
        raise ValueError("Frozen future supervision contains repeated coordinates or nonfinite targets")
    coordinates = future.set_index(["cell_uid", "original_cycle_id"]).soh_nominal
    for uid, row in cells.iterrows():
        for name, left, right in RATES:
            supported = (uid, left) in coordinates and (uid, right) in coordinates
            if bool(pd.notna(row[name])) != supported:
                raise ValueError("Frozen rate mask differs from exact observed endpoint support for " + uid + " / " + name)
            if supported:
                expected = (coordinates.loc[(uid, left)] - coordinates.loc[(uid, right)]) / (right - left)
                if not np.isclose(row[name], expected, atol=1e-10, rtol=1e-6):
                    raise ValueError("Frozen rate target differs from its two original endpoints")
    split = read_json(data_root / "split_contract.json")
    if tuple(split["splits"].keys()) != SPLITS and set(split["splits"]) != set(SPLITS):
        raise ValueError("The fixed five-fold and protocol-heldout split names differ")
    counts = Counter(uid for name in SPLITS[:5] for uid in split["splits"][name]["test"])
    if counts != Counter({uid: 1 for uid in cells.index}):
        raise ValueError("Every fixed physical UID must occur in exactly one outer test fold")
    return cells, future, split


def load_unit_results(unit, cells, future, split_contract, registration):
    split, arm, root = unit["split"], unit["arm"], unit["root"]
    scope = "main_oof" if split.startswith("cell_fold_") else "unseen_protocol_sensitivity"
    protocols = cells.protocol_uid.to_dict()
    eligible = set(cells.index[cells.input_eligible])
    expected_roles = {role: [uid for uid in split_contract["splits"][split][role] if uid in eligible]
                      for role in ("train", "validation", "test")}
    if unit["contract"]["roles"] != expected_roles:
        raise ValueError("Execution roles differ from the fixed all-cell split and early input mask")
    test_ids = expected_roles["test"]
    prediction = pd.read_parquet(root / "test_predictions.parquet")
    columns = ["cell_uid", "original_cycle_id", "target", "prediction", "trend"]
    if not set(columns).issubset(prediction):
        raise ValueError("Trajectory prediction columns differ from the registered schema")
    prediction = prediction[columns].copy()
    prediction.cell_uid = prediction.cell_uid.astype(str)
    if not np.isfinite(prediction[columns[1:]].to_numpy(float)).all():
        raise ValueError("Trajectory contains nonfinite coordinates, targets or predictions")
    if not np.equal(prediction.original_cycle_id, np.floor(prediction.original_cycle_id)).all():
        raise ValueError("Prediction original-cycle coordinate is noninteger")
    prediction.original_cycle_id = prediction.original_cycle_id.astype(np.int64)
    prediction = prediction.sort_values(columns[:2]).reset_index(drop=True)
    if prediction.duplicated(columns[:2]).any():
        raise ValueError("Repeated physical-cell original-cycle prediction")
    expected = future.loc[future.cell_uid.isin(test_ids)].sort_values(columns[:2]).reset_index(drop=True)
    if not prediction[columns[:2]].equals(expected[columns[:2]]):
        raise ValueError("Prediction support differs from all valid observed future cycles of the execution test cells")
    if not np.array_equal(prediction.target.to_numpy(np.float32), expected.soh_nominal.to_numpy(np.float32)):
        raise ValueError("Prediction target values differ from frozen native SOH labels")
    rows = metric_rows(prediction, test_ids, protocols, scope, split, arm)
    recomputed = pd.DataFrame(rows)
    primary = recomputed.loc[recomputed.metric.isin(("trajectory_mae", "trajectory_rmse"))]
    wide = primary.pivot(index="cell_uid", columns="metric", values="value").dropna()
    saved = pd.read_csv(root / "test_cell_metrics.csv", dtype={"cell_uid": str, "protocol_uid": str}).set_index("cell_uid")
    if saved.index.duplicated().any() or set(saved.index) != set(wide.index):
        raise ValueError("Saved cell-metric support differs from trajectory replay")
    saved = saved.loc[wide.index]
    if (not np.allclose(saved.mae, wide.trajectory_mae, atol=1e-7, rtol=5e-6)
            or not np.allclose(saved.rmse, wide.trajectory_rmse, atol=1e-7, rtol=5e-6)
            or any(saved.loc[uid, "protocol_uid"] != protocols[uid] for uid in saved.index)
            or not np.array_equal(saved.observations.to_numpy(), prediction.groupby("cell_uid").size().reindex(saved.index).to_numpy())):
        raise ValueError("Saved cell metrics differ from recomputed errors, counts or protocol identities")
    saved_metric = read_json(root / "test_metrics.json")
    if (saved_metric["supported_cells"] != len(wide) or saved_metric["observations"] != len(prediction)
            or not np.isclose(saved_metric["cell_mean_mae"], wide.trajectory_mae.mean(), atol=1e-7, rtol=5e-6)
            or not np.isclose(saved_metric["cell_mean_rmse"], wide.trajectory_rmse.mean(), atol=1e-7, rtol=5e-6)):
        raise ValueError("Saved aggregate metrics differ from independent per-cell replay")
    rates = pd.read_csv(root / "test_rate_predictions.csv", dtype={"cell_uid": str, "protocol_uid": str})
    required_rate = {"cell_uid", "protocol_uid", "target_name", "start_cycle", "end_cycle", "predicted_start_soh",
                     "predicted_end_soh", "predicted_rate", "target_supported", "target_rate"}
    if not required_rate.issubset(rates) or rates.duplicated(["cell_uid", "target_name"]).any():
        raise ValueError("Rate prediction schema or unique endpoint identities differ")
    if set(zip(rates.cell_uid, rates.target_name)) != {(uid, name) for uid in test_ids for name, _, _ in RATES}:
        raise ValueError("Rate table omits a test cell or a registered window")
    rates.target_supported = boolean(rates.target_supported)
    pred_lookup = prediction.set_index(["cell_uid", "original_cycle_id"])
    rate_rows = []
    rate_windows = {name: (left, right) for name, left, right in RATES}
    for row in rates.itertuples(index=False):
        left, right = rate_windows[row.target_name]
        supported = bool(pd.notna(cells.loc[row.cell_uid, row.target_name]))
        numerical = np.array([row.predicted_start_soh, row.predicted_end_soh, row.predicted_rate], float)
        if (row.start_cycle != left or row.end_cycle != right or row.protocol_uid != protocols[row.cell_uid]
                or row.target_supported != supported or not np.isfinite(numerical).all()
                or not np.isclose(row.predicted_rate, (row.predicted_start_soh - row.predicted_end_soh) / (right - left), atol=1e-12, rtol=1e-7)):
            raise ValueError("Rate prediction values, identity or exact-endpoint formula differ")
        for cycle, predicted in ((left, row.predicted_start_soh), (right, row.predicted_end_soh)):
            if (row.cell_uid, cycle) in pred_lookup.index and not np.isclose(predicted, pred_lookup.loc[(row.cell_uid, cycle), "prediction"], atol=1e-10, rtol=1e-8):
                raise ValueError("Saved rate endpoint differs from the same model trajectory prediction")
        truth = float(cells.loc[row.cell_uid, row.target_name]) if supported else None
        if supported and not np.isclose(row.target_rate, truth, atol=1e-10, rtol=1e-6):
            raise ValueError("Rate target differs from frozen exact-endpoint label")
        error = abs(row.predicted_rate - truth) if supported else None
        rate_rows.append({**row._asdict(), "scope": scope, "split": split, "arm": arm,
                          "absolute_error": error, "absolute_error_per_1000_cycles": error * 1000 if supported else None})
        rows.append({"scope": scope, "split": split, "arm": arm, "cell_uid": row.cell_uid,
                     "protocol_uid": protocols[row.cell_uid], "metric": row.target_name + "_absolute_error",
                     "value": error, "observations": 2 if supported else 0})
    eol = pd.read_csv(root / "test_eol_predictions.csv", dtype={"cell_uid": str})
    if (not {"cell_uid", "predicted_trend_eol80_cycle", "no_crossing_through_registered_horizon", "prediction_query_max_cycle"}.issubset(eol)
            or eol.cell_uid.duplicated().any() or set(eol.cell_uid) != set(test_ids)):
        raise ValueError("EOL prediction table differs from every executed test cell")
    eol.no_crossing_through_registered_horizon = boolean(eol.no_crossing_through_registered_horizon)
    if (not np.isfinite(eol.predicted_trend_eol80_cycle).all()
            or not eol.prediction_query_max_cycle.eq(5000).all()
            or not eol.predicted_trend_eol80_cycle.between(100, 5000).all()
            or not eol.loc[eol.no_crossing_through_registered_horizon, "predicted_trend_eol80_cycle"].eq(5000).all()):
        raise ValueError("EOL risk scores violate the pre-registered5000-cycle prediction horizon")
    eol = eol.assign(scope=scope, split=split, arm=arm)
    history = [json.loads(line) for line in (root / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    epochs = registration["config"]["training"]["epochs"]
    selection = unit["selection"]
    best = min(history, key=lambda row: row["validation_mae"])
    if ([row["epoch"] for row in history] != list(range(1, epochs + 1))
            or history[-1]["successful_updates"] != unit["contract"]["expected_updates"]
            or selection["selected_epoch"] != best["epoch"]
            or selection["validation_mae"] != best["validation_mae"]
            or selection["parameters"] != registration["initialization"]["models"][arm]["parameters"]):
        raise ValueError("Epoch count, update budget, minimum-validation selection or parameter count differs")
    training_row = {"scope": scope, "split": split, "arm": arm, "seed": 42,
                    "epochs": len(history), "selected_epoch": selection["selected_epoch"],
                    "validation_mae": selection["validation_mae"], "successful_updates": history[-1]["successful_updates"],
                    "parameters": selection["parameters"], "trainable_parameters": selection["trainable_parameters"],
                    "elapsed_seconds": history[-1]["elapsed_seconds"],
                    **{role + "_cells": len(ids) for role, ids in expected_roles.items()}}
    history_columns = ("epoch", "train_loss", "validation_mae", "selected", "successful_updates")
    reproduced_history = [{key: row[key] for key in history_columns} for row in history]
    return {"metrics": rows, "rates": rate_rows, "eol": eol, "training": training_row,
            "prediction": prediction, "history": reproduced_history,
            "rate_frame": rates.sort_values(["cell_uid", "target_name"]).reset_index(drop=True),
            "eol_frame": eol.sort_values("cell_uid").drop(columns=["arm"]).reset_index(drop=True)}


def metric_summaries(metrics):
    valid = metrics.loc[metrics.value.notna()].copy()
    protocols = valid.groupby(["scope", "arm", "metric", "protocol_uid"], sort=True).agg(
        protocol_mean=("value", "mean"), supported_cells=("cell_uid", "nunique"), observations=("observations", "sum")).reset_index()
    rows = []
    for (scope, arm, metric), frame in metrics.groupby(["scope", "arm", "metric"], sort=True):
        supported = frame.loc[frame.value.notna()]
        group_values = protocols.loc[(protocols.scope == scope) & (protocols.arm == arm) & (protocols.metric == metric)]
        rows.append({"scope": scope, "arm": arm, "metric": metric,
                     "cell_mean": supported.value.mean(), "protocol_macro": group_values.protocol_mean.mean(),
                     "execution_test_cells": len(frame), "supported_cells": len(supported),
                     "supported_protocols": len(group_values), "observations": int(supported.observations.sum()),
                     "unit": "SOH/cycle" if metric.startswith("r") else "SOH"})
    folds = valid.groupby(["scope", "split", "arm", "metric"], sort=True).agg(
        cell_mean=("value", "mean"), supported_cells=("cell_uid", "nunique"), observations=("observations", "sum")).reset_index()
    return pd.DataFrame(rows), protocols, folds


def selected_weight_equality(first_path, second_path):
    import torch

    first = torch.load(first_path, map_location="cpu", weights_only=True)["model"]
    second = torch.load(second_path, map_location="cpu", weights_only=True)["model"]
    equal = first.keys() == second.keys() and all(torch.equal(value, second[key]) for key, value in first.items())
    return equal, len(first)


def numeric_contrasts(metrics, issues):
    summaries, paired_rows, group_rows = [], [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        scoped = metrics.loc[metrics.scope == scope]
        for candidate, reference in CONTRASTS:
            contrast = candidate + "_vs_" + reference
            for metric in sorted(scoped.metric.unique()):
                selected = scoped.loc[scoped.metric == metric]
                indexed = selected.pivot(index=["cell_uid", "protocol_uid"], columns="arm", values="value")
                if not indexed[candidate].notna().equals(indexed[reference].notna()):
                    raise ValueError("Paired metric support differs across models for " + contrast + " / " + metric)
                pair = indexed[[reference, candidate]].rename(columns={reference: "reference_error", candidate: "candidate_error"}).reset_index()
                pair["improvement"] = pair.reference_error - pair.candidate_error
                pair["relative_improvement_percent"] = np.divide(pair.improvement.to_numpy() * 100, pair.reference_error.to_numpy(),
                                                               out=np.full(len(pair), np.nan), where=pair.reference_error.to_numpy() > 0)
                pair["supported"] = pair.improvement.notna()
                pair = pair.assign(scope=scope, metric=metric, contrast=contrast, candidate=candidate, reference=reference)
                paired_rows.extend(pair.to_dict("records"))
                supported = pair.loc[pair.supported].sort_values("cell_uid")
                groups = supported.groupby("protocol_uid", sort=True).agg(
                    improvement=("improvement", "mean"), reference_error=("reference_error", "mean"),
                    candidate_error=("candidate_error", "mean"), supported_cells=("cell_uid", "size")).reset_index()
                group_rows.extend(groups.assign(scope=scope, metric=metric, contrast=contrast).to_dict("records"))
                for estimator, sampling_unit, values, first, last in (
                    ("cell_mean", "paired_cell", supported.improvement, supported.reference_error.mean(), supported.candidate_error.mean()),
                    ("protocol_macro", "protocol_group", groups.improvement, groups.reference_error.mean(), groups.candidate_error.mean()),
                ):
                    statistics = paired_mean_bootstrap(values)
                    summaries.append({"scope": scope, "contrast": contrast, "candidate": candidate, "reference": reference,
                                      "metric": metric, "estimator": estimator, "sampling_unit": sampling_unit,
                                      "reference_metric": first, "candidate_metric": last, **statistics,
                                      "relative_improvement_percent": (first - last) / first * 100 if first > 0 else None,
                                      "paired_cells": len(supported), "protocol_groups": len(groups),
                                      "bootstrap_requested": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                                      "bootstrap_identifiable": statistics["resampling_units"] > 1,
                                      "improved_cells": int((supported.improvement > 0).sum()),
                                      "worsened_cells": int((supported.improvement < 0).sum()),
                                      "tied_cells": int((supported.improvement == 0).sum()),
                                      "direction": "reference_error_minus_candidate_error"})
                for row in supported.loc[supported.improvement < 0].itertuples(index=False):
                    issue(issues, "candidate_error_larger", {"reference_error": row.reference_error,
                          "candidate_error": row.candidate_error, "improvement": row.improvement},
                          severity="RESULT", category="negative_result", scope=scope, cell_uid=row.cell_uid,
                          metric=metric, contrast=contrast, arm=candidate)
    return pd.DataFrame(summaries), pd.DataFrame(paired_rows), pd.DataFrame(group_rows)


def eol_analysis(eol_predictions, cells):
    required = ("eol80_event_observed", "eol80_first_observed_crossing_cycle", "last_observed_valid_cycle")
    if not set(required).issubset(cells):
        raise ValueError("Frozen EOL event, first-crossing or right-censor outcome fields are missing")
    executed = sorted(set(eol_predictions.cell_uid))
    truth = cells.loc[executed, ["cell_uid", "protocol_uid", *required]].copy().reset_index(drop=True)
    truth["event"] = boolean(truth.eol80_event_observed)
    truth["outcome_cycle"] = np.where(truth.event, truth.eol80_first_observed_crossing_cycle, truth.last_observed_valid_cycle).astype(float)
    if (not np.isfinite(truth.outcome_cycle).all() or (truth.outcome_cycle < 1).any()
            or (truth.loc[truth.event, "eol80_first_observed_crossing_cycle"] > truth.loc[truth.event, "last_observed_valid_cycle"]).any()):
        raise ValueError("EOL event or censor outcome times are invalid")
    truth["at_risk100"] = truth.outcome_cycle > 100
    per_cell = eol_predictions.merge(truth, on="cell_uid", validate="many_to_one")
    summaries, contrasts = [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        scoped = per_cell.loc[per_cell.scope == scope]
        risk = scoped.loc[scoped.at_risk100]
        wide = risk.pivot(index="cell_uid", columns="arm", values="predicted_trend_eol80_cycle").reindex(columns=ARMS).sort_index()
        if wide.isna().any().any() or set(wide.columns) != set(ARMS):
            raise ValueError("EOL at-risk comparison does not share the same executed cells")
        outcome = truth.set_index("cell_uid").loc[wide.index]
        predictions = {arm: wide[arm].to_numpy(float) for arm in ARMS}
        absolute = {}
        for arm in ARMS:
            statistics = harrell_concordance(outcome.outcome_cycle, outcome.event, predictions[arm])
            absolute[arm] = statistics["harrell_c"]
            arm_rows = scoped.loc[scoped.arm == arm]
            summaries.append({"scope": scope, "arm": arm, **statistics,
                              "executed_test_cells": len(arm_rows), "at_risk100_cells": len(outcome),
                              "observed_events": int(outcome.event.sum()), "right_censored": int((~outcome.event).sum()),
                              "events_at_or_before100_excluded": int((~arm_rows.at_risk100 & arm_rows.event).sum()),
                              "no_observed_followup_after100_excluded": int((~arm_rows.at_risk100 & ~arm_rows.event).sum()),
                              "predicted_no_crossing_through5000": int(arm_rows.no_crossing_through_registered_horizon.sum()),
                              "prediction_query_max_cycle": 5000})
        samples = eol_bootstrap(outcome.outcome_cycle, outcome.event, predictions, outcome.protocol_uid)
        for candidate, reference in CONTRASTS:
            for unit, draws in samples.items():
                delta = draws[candidate] - draws[reference]
                supported = delta[np.isfinite(delta)]
                low, high = np.quantile(supported, [0.025, 0.975]) if len(supported) else (None, None)
                contrasts.append({"scope": scope, "contrast": candidate + "_vs_" + reference,
                                  "candidate": candidate, "reference": reference, "metric": "eol80_harrell_c",
                                  "estimator": "harrell_concordance", "sampling_unit": unit,
                                  "reference_metric": absolute[reference], "candidate_metric": absolute[candidate],
                                  "improvement": absolute[candidate] - absolute[reference] if absolute[candidate] is not None else None,
                                  "ci95_low": low, "ci95_high": high,
                                  "bootstrap_requested": BOOTSTRAP_REPLICATES, "bootstrap_supported": len(supported),
                                  "bootstrap_seed": BOOTSTRAP_SEED, "paired_cells": len(outcome),
                                  "protocol_groups": outcome.protocol_uid.nunique(),
                                  "direction": "candidate_concordance_minus_reference_concordance"})
    return pd.DataFrame(summaries), pd.DataFrame(contrasts), per_cell


def chinese_report(summary, contrasts, eol, reproduction, training, qualification, issues):
    lines = ["# MATR seed42 统一轨迹实验结果", "", "## 分析范围", "",
             "五个 cell-disjoint 外折合并为主 OOF 分析。每颗固定物理电芯在外 test 中恰好出现一次，执行时应用早期输入资格掩码。协议外推敏感性单独报告。全部七组使用同一 seed42，训练随机性重复证据仍待后续研究。", "",
             f"固定资格分母 {len(qualification)} 颗，早期输入合格 {int(qualification.input_eligible.sum())} 颗。", "",
             "## 主 OOF 轨迹", "", "| 模型 | cell-mean MAE | cell-mean RMSE | protocol-macro MAE | 支持电芯 | 有效观测点 |", "|---|---:|---:|---:|---:|---:|"]
    for scope, title in (("main_oof", None), ("unseen_protocol_sensitivity", "协议外推敏感性")):
        if title:
            lines += ["", "## " + title, "", "| 模型 | cell-mean MAE | cell-mean RMSE | protocol-macro MAE | 支持电芯 | 有效观测点 |", "|---|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            rows = summary.loc[(summary.scope == scope) & (summary.arm == arm)].set_index("metric")
            mae, rmse = rows.loc["trajectory_mae"], rows.loc["trajectory_rmse"]
            lines.append(f"| {arm} | {mae.cell_mean:.6f} | {rmse.cell_mean:.6f} | {mae.protocol_macro:.6f} | {int(mae.supported_cells)} | {int(mae.observations)} |")
    lines += ["", "## 分时段轨迹支持", "", "单元格依次给出 cell-mean MAE、RMSE 与支持电芯数 n。", "",
              "| 范围 | 模型 | 101–200 | 201–500 | 501–1000 | 1001 以后 |", "|---|---|---|---|---|---|"]
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        for arm in ARMS:
            indexed = summary.loc[(summary.scope == scope) & (summary.arm == arm)].set_index("metric")
            values = []
            for left, right in HORIZONS:
                mae, rmse = indexed.loc[f"horizon_{left}_{right}_mae"], indexed.loc[f"horizon_{left}_{right}_rmse"]
                values.append(f"{mae.cell_mean:.6f} / {rmse.cell_mean:.6f} (n={int(mae.supported_cells)})" if mae.supported_cells else "缺支持 (n=0)")
            lines.append("| " + " | ".join([scope, arm, *values]) + " |")
    lines += ["", "## 两段容量速率", "", "速率标签仅由两端均有效的原始观测构成。表内 MAE 的单位为 SOH/1000圈，各模型共享同一目标支持。", "",
              "| 范围 | 模型 | r200_500 MAE | 支持电芯 | r500_1000 MAE | 支持电芯 |", "|---|---|---:|---:|---:|---:|"]
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        for arm in ARMS:
            indexed = summary.loc[(summary.scope == scope) & (summary.arm == arm)].set_index("metric")
            first, second = indexed.loc["r200_500_absolute_error"], indexed.loc["r500_1000_absolute_error"]
            one = f"{first.cell_mean * 1000:.6f}" if first.supported_cells else "缺支持"
            two = f"{second.cell_mean * 1000:.6f}" if second.supported_cells else "缺支持"
            lines.append(f"| {scope} | {arm} | {one} | {int(first.supported_cells)} | {two} | {int(second.supported_cells)} |")
    lines += ["", "## 预设主对照", "", "正的 improvement 表示候选误差更低。cell 行重采样物理电芯，protocol-macro 行先在各协议内取电芯均值，再等权重采样协议。每项使用 seed42 的 2000 次配对 bootstrap，区间表示当前电芯与协议样本的不确定性。", "",
              "| 对照 | 估计量 | MAE 降低 | 95% 区间 | 配对电芯 | 协议组 |", "|---|---|---:|---|---:|---:|"]
    selected = contrasts.loc[(contrasts.scope == "main_oof") & (contrasts.metric == "trajectory_mae")]
    for candidate, reference in CONTRASTS:
        for row in selected.loc[selected.contrast == candidate + "_vs_" + reference].itertuples(index=False):
            interval = f"[{row.ci95_low:.6f}, {row.ci95_high:.6f}]" if pd.notna(row.ci95_low) else "支持不足"
            lines.append(f"| {row.contrast} | {row.estimator} | {row.improvement:.6f} | {interval} | {row.paired_cells} | {row.protocol_groups} |")
    lines += ["", "## EOL80 附属指标", "", "以仍处于 cycle100 风险集的电芯计算 Harrell C。已观察事件时间严格早于另一颗电芯事件或删失时间时形成可比较对，结局同时间对排除，预测同分贡献 0.5。所有预测均使用预设至 cycle5000 的 trend 首次 0.8 crossing，未 crossing 的风险分数固定为 5000。", "",
              "| 范围 | 模型 | Harrell C | 事件 | 右删失 | 可比较对 |", "|---|---|---:|---:|---:|---:|"]
    for row in eol.itertuples(index=False):
        score = f"{row.harrell_c:.6f}" if pd.notna(row.harrell_c) else "支持不足"
        lines.append(f"| {row.scope} | {row.arm} | {score} | {row.observed_events} | {row.right_censored} | {row.comparable_pairs} |")
    lines += ["", "## 预算与复现", "", f"已核验 {len(training)} 个训练单元，累计 optimizer updates {int(training.successful_updates.sum())}。逐单元 epoch、选中 epoch、参数数与累计耗时保存在 training_units.csv。",
              f"M2 与 M3-NR 六个 split 的逐值复现检查通过 {int(reproduction.reproduced.sum())}/{len(reproduction)}，详情保存在 duplicate_reproduction.csv。", "",
              "## 优势与风险", "", "全部预设对照、四个 horizon、两段 exact-endpoint rate、每电芯与每协议支持都予以保留。主指标按电芯平均，降低长轨迹电芯对总指标的隐式加权。", "",
              "固定169颗已参与历史探索，本轮属于内部探索性对照。单 seed 的训练波动尚未测量。协议样本数较小或仅一颗时，协议内重复方差证据有限。M0 包含单放电、已知策略与共同观测层的任务适配，NL 为完整全局信息消融。", "",
              f"all_failures.csv 共保留 {len(issues)} 条完整性、支持缺口或负向 cell 对照记录。分时段、rate、协议以及 EOL 配对区间见结构化结果。", "",
              "## 拓展思考", "", "M2 与 M3-NR 的重复一致性为计算图、输入、预算与断点链路提供直接核对。", "",
              "## 你可能想知道", "", "bootstrap 未覆盖训练 seed 的随机性。该范围已在本次报告与结果元数据中登记。"]
    return "\n".join(lines) + "\n"


def analyze(run_root, data_root):
    issues = []
    try:
        registration, registration_sha, units = matrix_delivery(run_root, data_root, issues)
        cells, future, split_contract = frozen_tables(data_root)
    except AnalysisBlocked as error:
        error.issues = issues
        raise
    except (KeyError, ValueError, OSError) as error:
        issue(issues, "frozen_analysis_inputs", {"exception": type(error).__name__, "message": str(error)})
        blocked = AnalysisBlocked("Frozen analysis inputs failed verification")
        blocked.issues = issues
        raise blocked from error
    metrics, rate_rows, eol_frames, training_rows, reproduction = [], [], [], [], []
    previous = {}
    for unit in units:
        try:
            result = load_unit_results(unit, cells, future, split_contract, registration)
        except (KeyError, ValueError, OSError, RuntimeError) as error:
            issue(issues, "unit_independent_replay", {"exception": type(error).__name__, "message": str(error)},
                  split=unit["split"], arm=unit["arm"])
            continue
        metrics.extend(result["metrics"])
        rate_rows.extend(result["rates"])
        eol_frames.append(result["eol"])
        training_rows.append(result["training"])
        if unit["arm"] == "M2":
            previous[unit["split"]] = (result, unit["root"] / "SELECTED.pt")
        if unit["arm"] == "M3-NR" and unit["split"] in previous:
            first, first_weights = previous.pop(unit["split"])
            weights_equal, tensors = selected_weight_equality(first_weights, unit["root"] / "SELECTED.pt")
            checks = {"trajectory_and_trend_equal": first["prediction"].equals(result["prediction"]),
                      "rate_endpoints_equal": first["rate_frame"].equals(result["rate_frame"]),
                      "eol_risk_scores_equal": first["eol_frame"].equals(result["eol_frame"]),
                      "training_history_equal_excluding_elapsed": first["history"] == result["history"],
                      "selected_weight_tensors_equal": weights_equal,
                      "selected_epoch_equal": first["training"]["selected_epoch"] == result["training"]["selected_epoch"]}
            row = {"split": unit["split"], **checks, "compared_weight_tensors": tensors, "reproduced": all(checks.values())}
            reproduction.append(row)
            if not row["reproduced"]:
                issue(issues, "M2_NR_reproduction_failed", checks, severity="WARNING", category="duplicate_reproduction", split=unit["split"])
    if any(row["severity"] == "ERROR" for row in issues):
        blocked = AnalysisBlocked("Formal outputs failed independent replay; aggregate analysis was withheld")
        blocked.issues = issues
        raise blocked
    metrics, rates, eol_predictions = pd.DataFrame(metrics), pd.DataFrame(rate_rows), pd.concat(eol_frames, ignore_index=True)
    if (metrics.duplicated(["scope", "arm", "cell_uid", "metric"]).any()
            or eol_predictions.duplicated(["scope", "arm", "cell_uid"]).any()):
        raise ValueError("A physical cell is repeated in combined OOF or heldout analysis")
    eligible = set(cells.index[cells.input_eligible])
    for arm in ARMS:
        main_ids = metrics.loc[(metrics.scope == "main_oof") & (metrics.arm == arm) & (metrics.metric == "trajectory_mae"), "cell_uid"]
        if set(main_ids) != eligible or len(main_ids) != len(eligible):
            raise ValueError("Main OOF does not contain every early-input-eligible physical cell exactly once")
    for row in metrics.loc[metrics.value.isna()].itertuples(index=False):
        issue(issues, "target_support_missing", {"observations": row.observations}, severity="INFO", category="support",
              scope=row.scope, split=row.split, arm=row.arm, cell_uid=row.cell_uid, metric=row.metric)
    summary, protocols, folds = metric_summaries(metrics)
    contrasts, paired, paired_protocols = numeric_contrasts(metrics, issues)
    eol_summary, eol_contrasts, eol_cells = eol_analysis(eol_predictions, cells)
    for row in eol_summary.loc[eol_summary.comparable_pairs == 0].itertuples(index=False):
        issue(issues, "EOL_no_comparable_pairs", {"events": row.observed_events, "censored": row.right_censored},
              severity="INFO", category="support", scope=row.scope, arm=row.arm, metric="eol80_harrell_c")
    columns = ["cell_uid", "protocol_uid", "input_eligible"] + [
        name for name in ("historical_exposure", "early_curve_valid_count", "admission_status", "failure_reasons",
                          "eol80_event_observed", "eol80_first_observed_crossing_cycle", "last_observed_valid_cycle") if name in cells]
    qualification = cells[columns].reset_index(drop=True).copy()
    qualification["future_valid_observations"] = qualification.cell_uid.map(future.groupby("cell_uid").size()).fillna(0).astype(int)
    for row in qualification.loc[~qualification.input_eligible].itertuples(index=False):
        issue(issues, "early_input_ineligible", "UID remains in the fixed169 qualification denominator and fixed split", severity="INFO",
              category="qualification", cell_uid=row.cell_uid)
    reproduction, training = pd.DataFrame(reproduction), pd.DataFrame(training_rows)
    state = "ANALYZED" if reproduction.reproduced.all() else "ANALYZED_WITH_REPRODUCTION_FAILURE"
    tables = {"cell_metrics.csv": metrics, "rate_predictions_and_errors.csv": rates,
              "main_oof_metrics.csv": summary.loc[summary.scope == "main_oof"],
              "protocol_heldout_metrics.csv": summary.loc[summary.scope == "unseen_protocol_sensitivity"],
              "protocol_metrics.csv": protocols, "split_metrics.csv": folds,
              "main_oof_fixed_contrasts.csv": contrasts.loc[contrasts.scope == "main_oof"],
              "protocol_heldout_fixed_contrasts.csv": contrasts.loc[contrasts.scope == "unseen_protocol_sensitivity"],
              "paired_cell_contrasts.csv": paired,
              "paired_protocol_contrasts.csv": paired_protocols, "eol80_summary.csv": eol_summary,
              "eol80_fixed_contrasts.csv": eol_contrasts, "eol80_cell_outcomes.csv": eol_cells,
              "training_units.csv": training, "duplicate_reproduction.csv": reproduction,
              "qualification_denominators.csv": qualification,
              "all_failures.csv": pd.DataFrame(issues, columns=ISSUE_COLUMNS)}
    summary_json = {"status": state, "training_seed": 42, "primary_scope": "main_oof",
                    "sensitivity_scope": "unseen_protocol_sensitivity", "fixed_contrasts": CONTRASTS,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                    "qualification_cells": len(cells), "early_input_eligible_cells": len(eligible),
                    "completed_training_units": len(training), "actual_optimizer_updates": int(training.successful_updates.sum()),
                    "M2_NR_reproduced_splits": int(reproduction.reproduced.sum()),
                    "metric_summary": summary.to_dict("records"), "fixed_metric_contrasts": contrasts.to_dict("records"),
                    "eol80_summary": eol_summary.to_dict("records"), "eol80_contrasts": eol_contrasts.to_dict("records"),
                    "issues_by_severity": dict(Counter(row["severity"] for row in issues)),
                    "interpretation": {"positive_error_improvement": "reference_error_minus_candidate_error",
                                       "positive_EOL_improvement": "candidate_concordance_minus_reference_concordance",
                                       "cell_interval": "paired physical-cell percentile bootstrap",
                                       "protocol_interval": "protocol-macro mean error contrast; EOL uses paired protocol-cluster concordance resampling",
                                       "training_seed_variability": "unmeasured under the single-seed first-round design",
                                       "historical_exposure": "All fixed169 cells participated in previous exploration"},
                    "analysis_environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
    contract = {"registration_sha256": registration_sha, "data_freeze_sha256": sha256(data_root / "DATA_FREEZE.json"),
                "analysis_source_sha256": sha256(Path(__file__)), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED, "contrasts": CONTRASTS,
                "units": [{"split": unit["split"], "arm": unit["arm"], "complete_sha256": unit["complete_sha256"]} for unit in units]}
    report = chinese_report(summary, contrasts, eol_summary, reproduction, training, qualification, issues)
    return contract, summary_json, tables, report


def selfcheck():
    records = []

    def require(condition, name):
        if not condition:
            raise AssertionError(name)
        records.append({"check": name, "status": "PASS"})

    with tempfile.TemporaryDirectory(prefix="analysis_QA_", dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory).resolve()
        if root.parent != Path(__file__).resolve().parent:
            raise RuntimeError("Engineering temporary directory escaped the verified workspace parent")
        problems = []
        try:
            matrix_delivery(root, root, problems)
        except AnalysisBlocked:
            rejected = True
        else:
            rejected = False
        require(rejected and problems[0]["category"] == "pending", "missing_formal_matrix_blocks_before_any_scientific_table_read")
        import torch

        first, second = root / "first.pt", root / "second.pt"
        torch.save({"model": {"engineering_tensor": torch.tensor([1.0, 2.0])}, "contract": {"arm": "M2"}}, first)
        torch.save({"model": {"engineering_tensor": torch.tensor([1.0, 2.0])}, "contract": {"arm": "M3-NR"}}, second)
        equal, tensors = selected_weight_equality(first, second)
        require(equal and tensors == 1, "selected_weights_compared_as_safe_tensor_values_with_arm_metadata_separate")
        torch.save({"model": {"engineering_tensor": torch.tensor([1.0, 3.0])}}, second)
        require(not selected_weight_equality(first, second)[0], "selected_weight_tensor_change_detected")
    frame = pd.DataFrame({"cell_uid": ["a", "b", "b"], "original_cycle_id": [101, 101, 201],
                          "target": [0.0, 0.0, 0.0], "prediction": [1.0, 2.0, 4.0]})
    rows = pd.DataFrame(metric_rows(frame, ["a", "b", "c"], {"a": "p1", "b": "p2", "c": "p2"}, "main_oof", "cell_fold_0", "M0"))
    primary = rows.loc[rows.metric == "trajectory_mae"].set_index("cell_uid")
    require(primary.loc["a", "value"] == 1 and primary.loc["b", "value"] == 3 and pd.isna(primary.loc["c", "value"]), "cell_weighted_primary_with_unobserved_cell_retained")
    require(float(primary.value.mean()) == 2.0, "primary_is_mean_of_cell_errors")
    horizon = rows.loc[(rows.metric == "horizon_101_200_mae") & (rows.cell_uid == "b")].iloc[0]
    require(horizon.value == 2 and horizon.observations == 1, "horizon_masks_use_registered_inclusive_coordinates")
    interval = paired_mean_bootstrap([1.0, 2.0, -1.0], replicates=20)
    require(interval == paired_mean_bootstrap([1.0, 2.0, -1.0], replicates=20), "paired_bootstrap_seed42_exact_replay")
    require(interval["bootstrap_supported"] == 20 and interval["resampling_units"] == 3, "paired_bootstrap_counts_recorded")
    require(paired_mean_bootstrap([], replicates=20)["improvement"] is None, "unsupported_metric_returns_no_scientific_estimate")
    score = harrell_concordance([110, 120, 130], [True, True, False], [111, 121, 5000])
    require(score["harrell_c"] == 1 and score["comparable_pairs"] == 3, "Harrell_event_before_other_event_or_censor")
    score = harrell_concordance([110, 110, 130], [True, True, False], [5000, 5000, 5000])
    require(score["harrell_c"] == 0.5 and score["comparable_pairs"] == 2 and score["prediction_ties"] == 2, "Harrell_outcome_ties_excluded_prediction_ties_half")
    require(harrell_concordance([110, 120], [False, False], [1, 2])["harrell_c"] is None, "all_censored_no_comparable_pairs_unsupported")
    scores = eol_bootstrap([110, 120, 130], [True, True, False], {"M0": [111, 121, 5000], "M3": [111, 121, 5000]}, ["p1", "p1", "p2"], replicates=20)
    require(all(np.array_equal(value["M0"], value["M3"], equal_nan=True) for value in scores.values()), "paired_E0L_resampling_uses_identical_cell_and_protocol_draws")
    weights = np.random.default_rng(42).multinomial(3, [1 / 3] * 3, size=20)[0]
    indices = np.repeat(np.arange(3), weights)
    direct = harrell_concordance(np.array([110, 120, 130])[indices], np.array([True, True, False])[indices], np.array([111, 121, 5000])[indices])
    require(scores["cell"]["M0"][0] == direct["harrell_c"], "weighted_pair_E0L_bootstrap_matches_explicit_duplicate_cell_resample")
    fixture = pd.DataFrame({"scope": ["main_oof"] * 3, "arm": ["M0"] * 3, "metric": ["trajectory_mae"] * 3,
                            "protocol_uid": ["p1", "p1", "p2"], "cell_uid": ["a", "b", "c"],
                            "split": ["cell_fold_0"] * 3, "value": [1.0, 3.0, 9.0], "observations": [1, 2, 1]})
    aggregate, group, _ = metric_summaries(fixture)
    require(np.isclose(aggregate.cell_mean.iloc[0], 13 / 3) and aggregate.protocol_macro.iloc[0] == 5.5, "protocol_macro_equal_group_weight_and_cell_mean_preserved")
    require(group.supported_cells.tolist() == [2, 1], "protocol_support_counts_retained")
    numeric_rows = []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        for number, arm in enumerate(ARMS):
            for uid, protocol, value in (("a", "p1", 1.0), ("b", "p1", 3.0), ("c", "p2", 9.0)):
                for metric, observed in (("trajectory_mae", value + number / 10), ("horizon_1001_1000000_mae", None)):
                    numeric_rows.append({"scope": scope, "arm": arm, "cell_uid": uid, "protocol_uid": protocol,
                                         "metric": metric, "value": observed, "observations": int(observed is not None)})
    problems = []
    comparison, pairs, groups = numeric_contrasts(pd.DataFrame(numeric_rows), problems)
    require(len(comparison) == 2 * 7 * 2 * 2 and len(pairs) == 2 * 7 * 2 * 3, "every_fixed_contrast_metric_scope_and_estimator_retained")
    require(bool((comparison.loc[comparison.metric == "horizon_1001_1000000_mae", "bootstrap_supported"] == 0).all()), "unsupported_horizon_keeps_zero_support_and_null_intervals")
    require(any(row["category"] == "negative_result" for row in problems), "negative_cell_contrasts_retained_without_winner_selection")
    require(set(comparison.loc[comparison.metric == "trajectory_mae", "sampling_unit"]) == {"paired_cell", "protocol_group"}, "paired_cell_and_equal_protocol_macro_sampling_both_present")
    eol_truth = pd.DataFrame({"cell_uid": ["a", "b", "c", "d"], "protocol_uid": ["p1", "p1", "p2", "p2"],
                              "eol80_event_observed": [True, True, False, True],
                              "eol80_first_observed_crossing_cycle": [110.0, 110.0, np.nan, 90.0],
                              "last_observed_valid_cycle": [200.0, 200.0, 130.0, 150.0]}).set_index("cell_uid", drop=False)
    predictions = pd.DataFrame([{"scope": scope, "split": "fixture", "arm": arm, "cell_uid": uid,
                                 "predicted_trend_eol80_cycle": 5000.0, "no_crossing_through_registered_horizon": True,
                                 "prediction_query_max_cycle": 5000}
                                for scope in ("main_oof", "unseen_protocol_sensitivity") for arm in ARMS for uid in eol_truth.index])
    eol_summary, eol_delta, outcomes = eol_analysis(predictions, eol_truth)
    require(eol_summary.at_risk100_cells.eq(3).all() and eol_summary.events_at_or_before100_excluded.eq(1).all(), "E0L_risk_set_excludes_prefix_events")
    require(eol_summary.harrell_c.eq(0.5).all() and eol_summary.comparable_pairs.eq(2).all(), "E0L_outcome_and_prediction_ties_follow_registered_definition")
    require(outcomes.predicted_trend_eol80_cycle.eq(5000).all(), "E0L_no_crossing_risk_score_keeps_registered_horizon_independent_of_observation_stop")
    require(len(eol_delta) == 2 * 7 * 2 and bool((eol_delta.bootstrap_supported < 2000).all()), "E0L_paired_bootstrap_records_noncomparable_resamples")
    before100 = eol_truth.copy()
    before100["last_observed_valid_cycle"] = 90.0
    before100.loc[before100.eol80_event_observed, "eol80_first_observed_crossing_cycle"] = 50.0
    unsupported, _, _ = eol_analysis(predictions, before100)
    require(unsupported.at_risk100_cells.eq(0).all() and unsupported.harrell_c.isna().all(), "empty_E0L_risk_set_is_reported_unsupported")
    require(len(CONTRASTS) == 7 and len(SPLITS) * len(ARMS) == 42, "fixed_seven_contrasts_and42_units")
    return {"status": "PASS", "scope": "synthetic_analysis_mathematics_and_guards_only", "formal_results_read": False,
            "optimizer_updates": 0, "checks": len(records), "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        result = selfcheck()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(json_bytes(result))
        print(json.dumps({key: result[key] for key in ("status", "scope", "checks", "formal_results_read")}))
        return
    if args.data is None or args.run_root is None:
        parser.error("Formal analysis requires --data and --run-root")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output / "RESULTS_REPORT_ZH.md"
    # Reuse the private training lock while verifying its completed snapshot.
    with execution_lock(args.run_root / ".matrix.lock"):
        complete_path = args.output / "ANALYSIS_COMPLETE.json"
        if complete_path.exists():
            complete = read_json(complete_path)
            if (complete["registration_sha256"] != sha256(args.run_root / "RUN_REGISTRATION.json")
                    or complete["analysis_source_sha256"] != sha256(Path(__file__))
                    or complete["data_freeze_sha256"] != sha256(args.data / "DATA_FREEZE.json")):
                raise RuntimeError("Existing completed analysis has a different frozen source or input contract")
            for entry in complete["files"]:
                if sha256(args.output / entry["path"]) != entry["sha256"]:
                    raise RuntimeError("Completed analysis artifact fingerprint differs")
            if args.report is not None:
                original_report = args.output / complete["report_path"]
                write_once(args.report, original_report.read_bytes())
            print(json.dumps({"status": "VERIFIED_EXISTING_ANALYSIS", "files": len(complete["files"])}))
            return
        try:
            contract, summary, tables, report = analyze(args.run_root, args.data)
        except (AnalysisBlocked, KeyError, ValueError, OSError, RuntimeError) as error:
            problems = getattr(error, "issues", [])
            if not problems:
                issue(problems, "analysis_calculation_error", {"exception": type(error).__name__, "message": str(error)})
            status = {"status": "PENDING_OR_INVALID_FORMAL_RESULTS", "scientific_analysis_complete": False,
                      "message": str(error), "issues": problems}
            temporary = args.output / "ANALYSIS_STATUS.json.pending"
            temporary.write_bytes(json_bytes(status))
            os.replace(temporary, args.output / "ANALYSIS_STATUS.json")
            print(json.dumps({"status": status["status"], "message": str(error)}))
            raise SystemExit(2)
        write_once(args.output / "ANALYSIS_CONTRACT.json", json_bytes(contract))
        paths = [args.output / "ANALYSIS_CONTRACT.json"]
        for name, table in tables.items():
            path = args.output / name
            write_once(path, table.to_csv(index=False, lineterminator="\n").encode("utf-8"))
            paths.append(path)
        summary_path = args.output / "ANALYSIS_SUMMARY.json"
        write_once(summary_path, json_bytes(summary))
        paths.append(summary_path)
        write_once(report_path, report.encode("utf-8"))
        paths.append(report_path)
        complete = {"status": summary["status"], "registration_sha256": contract["registration_sha256"],
                    "analysis_source_sha256": contract["analysis_source_sha256"],
                    "data_freeze_sha256": contract["data_freeze_sha256"], "training_seed": 42,
                    "report_path": os.path.relpath(report_path.resolve(), args.output.resolve()).replace("\\", "/"),
                    "files": [{"path": os.path.relpath(path.resolve(), args.output.resolve()).replace("\\", "/"),
                               "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths]}
        write_once(complete_path, json_bytes(complete))
        pending = args.output / "ANALYSIS_STATUS.json"
        if pending.exists():
            pending.unlink()
        print(json.dumps({"status": summary["status"], "files": len(paths), "seed": 42}))


if __name__ == "__main__":
    main()
