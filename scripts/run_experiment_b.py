"""Registered HUST early-feature prediction of observed future capacity rates.

Registration fixes the shared cell cohort, input columns and split before any
model selection. Device-context parameters are additional prior inputs. The
historical-cell split supports exploratory holdout evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TARGETS = ("r500_1000", "r200_500")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CONTEXT = tuple(f"prior_discharge_rate_{stage}_C" for stage in (1, 2, 3))
CAPACITY = ("early_capacity_level", "early_capacity_slope", "early_capacity_std")
SHAPE = ("qv_window_capacity_start_Ah", "qv_window_capacity_end_Ah",
         "delta_qv_mean_Ah", "delta_qv_std_Ah", "delta_qv_log10_variance",
         "delta_qv_range_Ah", "delta_qv_rms_Ah")
MODELS = {
    "mean": CONTEXT,
    "context_only": CONTEXT,
    "context_capacity": CONTEXT + CAPACITY,
    "context_qv_shape": CONTEXT + SHAPE,
    "combined": CONTEXT + CAPACITY + SHAPE,
}
SPLIT_PREFIX = "physics-B-20261115|"
BOOTSTRAP_SEED = 20261115


def canonical_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def write_once(path, payload):
    """Publish a complete file atomically, refusing differing existing content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Immutable artifact differs from registered content: {path.name}")
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with temporary.open("xb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        try:
            if os.name == "nt":
                # Windows rename publishes atomically and refuses an existing target.
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(f"Concurrent artifact has different content: {path.name}")
    finally:
        temporary.unlink(missing_ok=True)


def csv_bytes(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


@contextlib.contextmanager
def execution_lock(path):
    """An OS-held lock releases automatically if the process terminates."""
    with path.open("a+b") as file:
        file.seek(0, 2)
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("An experiment process already holds this output lock") from error
        try:
            yield
        finally:
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def boolean(value, field, cell_id):
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"Unrecognized {field} for {cell_id}: {value!r}")


def canonical_id(value):
    return str(value).strip().removesuffix(".pkl")


def index_rows(rows, label):
    result = {}
    for row in rows:
        identifier = canonical_id(row.get("cell_id", ""))
        if not identifier or identifier in result:
            raise ValueError(f"Missing or duplicate cell_id in {label}: {identifier!r}")
        result[identifier] = row
    return result


def load_inputs(args):
    tables, identities = {}, {}
    for name in ("eligible", "features", "target", "metadata"):
        path = getattr(args, name + "_csv")
        payload = path.read_bytes()
        tables[name] = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        identities[name] = {"file_name": path.name, "bytes": len(payload), "sha256": sha256(payload)}
    eligible = index_rows([row for row in tables["eligible"] if row.get("family") == "HUST"], "eligible")
    feature_rows = [row for row in tables["features"] if row.get("family") == "HUST" and finite_number(row.get("prefix_cutoff_supplied_cycle_number")) == 100]
    features = index_rows(feature_rows, "K100 features")
    targets = index_rows(tables["target"], "raw targets")
    if len(targets) != 77 or any(not name.startswith("HUST_") for name in targets):
        raise ValueError(f"Expected the official raw-counter target table for 77 HUST cells, received {len(targets)} rows")
    metadata = {}
    for row in tables["metadata"]:
        if row.get("family") != "HUST":
            continue
        item = json.loads(row["metadata"])
        identifier = canonical_id(item["cell_id"])
        if identifier in metadata:
            raise ValueError(f"Duplicate HUST metadata for {identifier}")
        metadata[identifier] = item
    accepted, excluded = [], []
    for identifier, row in sorted(eligible.items()):
        reasons = []
        if not boolean(row.get("physical_cycle_axis_authenticated"), "physical_cycle_axis_authenticated", identifier):
            reasons.append("RAW_CYCLE_AXIS_UNAUTHENTICATED")
        if not boolean(row.get("prefix_K100_numbers_complete"), "prefix_K100_numbers_complete", identifier):
            reasons.append("CYCLES_1_THROUGH_100_INCOMPLETE")
        curve_count = finite_number(row.get("curve_valid_records_K100"))
        if curve_count is None or curve_count < 80:
            reasons.append("FEWER_THAN_80_VALID_K100_CURVES")
        legacy = boolean(row.get("legacy_576_member"), "legacy_576_member", identifier)
        feature = features.get(identifier, {})
        target = targets.get(identifier, {})
        values = {name: finite_number(feature.get(name)) for name in CAPACITY + SHAPE}
        target_values = {name: finite_number(target.get(name)) for name in TARGETS}
        missing = [name for name, value in values.items() if value is None]
        if missing:
            reasons.append("MISSING_REQUIRED_PREFIX_FEATURES=" + ",".join(missing))
        missing_targets = [name for name, value in target_values.items() if value is None]
        if missing_targets:
            reasons.append("MISSING_SHARED_TARGETS=" + ",".join(missing_targets))
        protocol = metadata.get(identifier, {}).get("discharge_protocol")
        rates = []
        if isinstance(protocol, list) and len(protocol) >= 3:
            rates = [finite_number(step.get("rate_in_C")) if isinstance(step, dict) else None for step in protocol[:3]]
        if len(rates) != 3 or any(value is None or value <= 0 for value in rates):
            reasons.append("FIRST_THREE_PRIOR_DISCHARGE_RATES_UNAVAILABLE")
        if reasons:
            excluded.append({"cell_id": identifier, "reasons": "|".join(reasons)})
            continue
        values.update(dict(zip(CONTEXT, rates)))
        accepted.append({"cell_id": identifier, "legacy_576_member": legacy,
                         "protocol_group": json.dumps(rates, separators=(",", ":")),
                         "features": values, "targets": target_values})
    return accepted, excluded, identities, {name: len(rows) for name, rows in tables.items()}


def register(args):
    cells, excluded, identities, input_counts = load_inputs(args)
    historical = sorted([row for row in cells if row["legacy_576_member"]],
                        key=lambda row: (sha256((SPLIT_PREFIX + row["cell_id"]).encode()), row["cell_id"]))
    train_count = math.floor(0.75 * len(historical))
    train_ids = {row["cell_id"] for row in historical[:train_count]}
    validation_ids = {row["cell_id"] for row in historical[train_count:]}
    for row in cells:
        row["role"] = "train" if row["cell_id"] in train_ids else "val" if row["cell_id"] in validation_ids else "test"
    split = [{"cell_id": row["cell_id"], "role": row["role"], "legacy_576_member": row["legacy_576_member"], "protocol_group": row["protocol_group"]} for row in cells]
    split_payload = csv_bytes(split, ("cell_id", "role", "legacy_576_member", "protocol_group"))
    counts = dict(Counter(row["role"] for row in cells))
    unsupported = []
    for role, minimum in (("train", 2), ("val", 1), ("test", 1)):
        if counts.get(role, 0) < minimum:
            unsupported.append(f"{role} has {counts.get(role,0)} cells, requires at least {minimum}")
    contract = {
        "version": "HUST_EARLY_FEATURE_FUTURE_RATE_EXPERIMENT_B_20261115",
        "status": "READY" if not unsupported else "INSUFFICIENT_SUPPORT",
        "inputs": identities, "input_row_counts": input_counts,
        "implementation_sha256": sha256(Path(__file__).read_bytes()),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "targets": {"primary": TARGETS[0], "secondary": TARGETS[1], "measurement": "caller_supplied_official_HUST_raw_discharge_counter_capacity_targets", "unit": "SOH_ratio_per_original_cycle"},
        "cohort_rule": "HUST original cycle axis authenticated, cycles1..100 complete, at least80 valid K100 curves, every required input finite, and both registered targets observed. All targets and models use this same intersection.",
        "cell_id_normalization": "Remove optional .pkl suffix only",
        "prefix": 100, "models": {name: list(columns) for name, columns in MODELS.items()},
        "mean_control": "Receives the same prior context availability, predicts the source-train target mean and ignores covariates",
        "prior_context": {"source": "metadata JSON discharge_protocol[0:3].rate_in_C", "columns": list(CONTEXT), "role": "additional_source_confirmed_prior_device_context", "last_stage": "excluded_due_to_incorrect_hardcoded_rate"},
        "temperature": {"chamber_setpoint_C": 30.0, "actual_cell_temperature_measured": False, "used_as_model_input": False},
        "split_rule": {"historical_members": "train or validation", "new_to_576_members": "all test", "sort_key": "sha256('physics-B-20261115|'+cell_id)", "train_count": "floor(0.75*historical_eligible_count)", "same_exact_cells_for_both_targets": True},
        "split_sha256": sha256(split_payload), "role_counts": counts,
        "selection": {"alphas": list(ALPHAS), "metric": "validation_MAE", "ties": "first alpha in ascending registered order", "scaler_fit": "train_only", "selected_model_refit_on_validation": False},
        "test_policy": "Freeze selected coefficients and alpha before opening test predictions. Cache each model prediction atomically. Completed EVALUATED outputs reject reevaluation.",
        "paired_comparison": {"candidate": "combined", "control": "context_capacity", "improvement": "abs_error(control)-abs_error(candidate)", "bootstrap_unit": "test_cell", "resamples": 2000, "seed": BOOTSTRAP_SEED, "interval": "percentile_95"},
        "limitations": ["Historical data exploratory holdout. New-to576 does not establish fresh independent confirmation.", "Test cells may share source/device/protocol structures with train cells. This is not a held-out-protocol study.", "The three prior device-context rates extend the prior VIQ-only information contract.", "Chamber30C is design metadata, actual cell temperature is unmeasured.", "Registered stable hinge candidates are shape diagnostics. Knee supervised targets are excluded.", "Bootstrap intervals condition on the fitted models and the selected test-cell cohort."],
        "excluded_cells": excluded, "unsupported_reasons": unsupported,
    }
    write_once(args.output / "split.csv", split_payload)
    write_once(args.output / "excluded_cells.csv", csv_bytes(excluded, ("cell_id", "reasons")))
    write_once(args.output / "CONTRACT.json", canonical_json(contract))
    return cells, contract, sha256((args.output / "CONTRACT.json").read_bytes())


def matrix(rows, columns):
    return np.asarray([[row["features"][column] for column in columns] for row in rows], dtype=float)


def fit_selection(cells, contract_digest):
    train = [row for row in cells if row["role"] == "train"]
    validation = [row for row in cells if row["role"] == "val"]
    selected = {}
    for target in TARGETS:
        y_train = np.asarray([row["targets"][target] for row in train])
        y_validation = np.asarray([row["targets"][target] for row in validation])
        selected[target] = {}
        for name, columns in MODELS.items():
            if name == "mean":
                mean = float(y_train.mean())
                selected[target][name] = {"kind": "mean", "features": list(columns), "mean": mean,
                                          "validation_MAE": float(np.mean(np.abs(y_validation - mean)))}
                continue
            scaler = StandardScaler().fit(matrix(train, columns))
            x_train = scaler.transform(matrix(train, columns))
            x_validation = scaler.transform(matrix(validation, columns))
            grid, best = [], None
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha).fit(x_train, y_train)
                mae = float(np.mean(np.abs(model.predict(x_validation) - y_validation)))
                grid.append({"alpha": alpha, "validation_MAE": mae})
                if best is None or mae < best[0]:
                    best = (mae, alpha, model)
            mae, alpha, model = best
            selected[target][name] = {"kind": "ridge", "features": list(columns), "alpha": alpha,
                                      "validation_MAE": mae, "validation_grid": grid,
                                      "scaler_fit_role": "train_only", "scaler_mean": scaler.mean_.tolist(),
                                      "scaler_scale": scaler.scale_.tolist(), "coef_scaled": model.coef_.tolist(),
                                      "intercept": float(model.intercept_)}
    return {"contract_sha256": contract_digest, "models": selected,
            "train_ids": [row["cell_id"] for row in train], "validation_ids": [row["cell_id"] for row in validation]}


def selected_prediction(rows, selected):
    if selected["kind"] == "mean":
        return np.full(len(rows), selected["mean"], dtype=float)
    x = matrix(rows, selected["features"])
    scaled = (x - np.asarray(selected["scaler_mean"])) / np.asarray(selected["scaler_scale"])
    return scaled @ np.asarray(selected["coef_scaled"]) + selected["intercept"]


def metric_values(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = prediction - truth
    variance = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float(1 - np.sum(error ** 2) / variance) if len(truth) >= 2 and variance > 0 else None
    correlation = None
    if len(truth) >= 2 and np.ptp(truth) > 0 and np.ptp(prediction) > 0:
        correlation = float(spearmanr(truth, prediction).statistic)
    return {"n": len(truth), "MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "R2": r2, "Spearman": correlation,
            "R2_reason": "AVAILABLE" if r2 is not None else "INSUFFICIENT_OR_CONSTANT_TARGET_SUPPORT",
            "Spearman_reason": "AVAILABLE" if correlation is not None else "INSUFFICIENT_OR_CONSTANT_VECTOR_SUPPORT"}


def evaluate(cells, selections, contract_digest, output):
    selection_digest = sha256((output / "SELECTIONS.json").read_bytes())
    write_once(output / "TEST_OPENED.json", canonical_json({"contract_sha256": contract_digest, "selection_sha256": selection_digest}))
    test = sorted([row for row in cells if row["role"] == "test"], key=lambda row: row["cell_id"])
    predictions, metrics, protocols, comparisons, negative = [], [], [], [], []
    for target in TARGETS:
        truth = np.asarray([row["targets"][target] for row in test])
        model_predictions = {}
        for name in MODELS:
            path = output / "predictions" / f"{target}__{name}.csv"
            if path.exists():
                cached = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
                if [row["cell_id"] for row in cached] != [row["cell_id"] for row in test] or any(row["contract_sha256"] != contract_digest or row["selection_sha256"] != selection_digest for row in cached):
                    raise RuntimeError(f"Cached test prediction identity differs for {target}/{name}")
                observed_truth = np.asarray([float(row["truth"]) for row in cached])
                if not np.array_equal(observed_truth, truth):
                    raise RuntimeError(f"Cached test target differs for {target}/{name}")
                prediction = np.asarray([float(row["prediction"]) for row in cached])
            else:
                prediction = selected_prediction(test, selections["models"][target][name])
                if not np.isfinite(prediction).all():
                    raise RuntimeError(f"Nonfinite selected prediction for {target}/{name}")
                cached = [{"cell_id": cell["cell_id"], "role": "test", "target": target, "model": name,
                           "protocol_group": cell["protocol_group"], "truth": float(y), "prediction": float(p),
                           "absolute_error": float(abs(y - p)), "contract_sha256": contract_digest,
                           "selection_sha256": selection_digest} for cell, y, p in zip(test, truth, prediction)]
                write_once(path, csv_bytes(cached, tuple(cached[0])))
            receipt_path = path.with_suffix(".json")
            write_once(receipt_path, canonical_json({"prediction_sha256": sha256(path.read_bytes()), "contract_sha256": contract_digest, "selection_sha256": selection_digest, "cells": len(test)}))
            model_predictions[name] = prediction
            predictions.extend(cached)
            metrics.append({"target": target, "model": name, **metric_values(truth, prediction)})
            for group in sorted({row["protocol_group"] for row in test}):
                selected = np.asarray([row["protocol_group"] == group for row in test])
                protocols.append({"target": target, "model": name, "protocol_group": group, **metric_values(truth[selected], prediction[selected])})
        baseline = model_predictions["context_capacity"]
        combined = model_predictions["combined"]
        improvement = np.abs(truth - baseline) - np.abs(truth - combined)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = improvement[rng.integers(0, len(test), size=(2000, len(test)))].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        comparisons.append({"target": target, "comparison": "combined_vs_context_capacity", "n_test_cells": len(test),
                            "MAE_improvement": float(improvement.mean()), "CI95_low": float(low), "CI95_high": float(high),
                            "bootstrap_resamples": 2000, "bootstrap_seed": BOOTSTRAP_SEED,
                            "positive_direction": "combined has lower absolute error"})
        for cell, y, base, prediction, gain in zip(test, truth, baseline, combined, improvement):
            if gain < 0:
                negative.append({"cell_id": cell["cell_id"], "target": target, "protocol_group": cell["protocol_group"],
                                 "truth": float(y), "context_capacity_prediction": float(base), "combined_prediction": float(prediction),
                                 "absolute_error_increase": float(-gain)})
    outputs = {
        "predictions.csv": (predictions, ("cell_id", "role", "target", "model", "protocol_group", "truth", "prediction", "absolute_error", "contract_sha256", "selection_sha256")),
        "metrics.csv": (metrics, ("target", "model", "n", "MAE", "RMSE", "R2", "Spearman", "R2_reason", "Spearman_reason")),
        "protocol_metrics.csv": (protocols, ("target", "model", "protocol_group", "n", "MAE", "RMSE", "R2", "Spearman", "R2_reason", "Spearman_reason")),
        "paired_mae_improvement.csv": (comparisons, ("target", "comparison", "n_test_cells", "MAE_improvement", "CI95_low", "CI95_high", "bootstrap_resamples", "bootstrap_seed", "positive_direction")),
        "negative_cases.csv": (sorted(negative, key=lambda row: row["absolute_error_increase"], reverse=True), ("cell_id", "target", "protocol_group", "truth", "context_capacity_prediction", "combined_prediction", "absolute_error_increase")),
    }
    for name, (rows, columns) in outputs.items():
        write_once(output / name, csv_bytes(rows, columns))
    artifacts = {str(path.relative_to(output)).replace("\\", "/"): sha256(path.read_bytes()) for path in sorted(output.rglob("*")) if path.is_file() and path.name not in ("RUN.lock", "EVALUATED.json")}
    write_once(output / "EVALUATED.json", canonical_json({"status": "COMPLETE_EXPLORATORY_HOLDOUT", "contract_sha256": contract_digest,
                "selection_sha256": selection_digest, "test_cells": len(test), "targets": list(TARGETS),
                "models": list(MODELS), "test_prediction_arrays": len(TARGETS) * len(MODELS), "artifacts_sha256": artifacts}))
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with execution_lock(args.output / "RUN.lock"):
        if not args.register_only and (args.output / "EVALUATED.json").exists():
            raise RuntimeError("EVALUATED.json exists. Completed test evaluation is sealed.")
        cells, contract, digest = register(args)
        if args.register_only:
            print(json.dumps({"status": "REGISTERED", "contract_status": contract["status"], "contract_sha256": digest, "role_counts": contract["role_counts"], "excluded": len(contract["excluded_cells"]), "test_predictions_computed": 0}, ensure_ascii=False))
            return 0
        if contract["unsupported_reasons"]:
            raise RuntimeError("Insufficient registered support: " + " | ".join(contract["unsupported_reasons"]))
        selection_path = args.output / "SELECTIONS.json"
        if selection_path.exists():
            selections = json.loads(selection_path.read_text(encoding="utf-8"))
            if selections["contract_sha256"] != digest:
                raise RuntimeError("Frozen source selection has a different contract identity")
        else:
            selections = fit_selection(cells, digest)
            write_once(selection_path, canonical_json(selections))
        selection_digest = sha256(selection_path.read_bytes())
        write_once(args.output / "SELECTIONS.sha256", (selection_digest + "\n").encode())
        comparisons = evaluate(cells, selections, digest, args.output)
        print(json.dumps({"status": "COMPLETE_EXPLORATORY_HOLDOUT", "role_counts": contract["role_counts"], "paired_comparisons": comparisons}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
