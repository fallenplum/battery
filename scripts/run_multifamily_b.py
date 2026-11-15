"""Registered MATR/Stanford observed-curve prediction of future capacity rates.

The existing HUST implementation supplies atomic artifacts, locks, Ridge
selection and evaluation. Each target uses its observed support within one
previously fixed base-cohort split. Physical-axis evidence is recorded alongside
the supplied-coordinate inference task.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

import run_experiment_b as shared


TARGETS = ("r200_500", "r500_1000")
SPLIT_PREFIX = "physics-B-extension-20261115|"
SOURCE = Path(__file__).with_name("run_experiment_b.py")
REQUIRED = shared.CAPACITY + shared.SHAPE


def read_csv(path):
    payload = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    identity = {"file_name": path.name, "bytes": len(payload), "sha256": shared.sha256(payload)}
    return rows, identity


def make_models(context_columns):
    context = tuple(context_columns)
    return {
        "mean": context,
        "context_only": context,
        "context_capacity": context + shared.CAPACITY,
        "context_qv_shape": context + shared.SHAPE,
        "combined": context + shared.CAPACITY + shared.SHAPE,
    }


def context_number(value):
    flag = str(value).strip().lower()
    if flag in ("true", "false"):
        return float(flag == "true")
    return shared.finite_number(value)


def load_inputs(args):
    tables, identities = {}, {}
    for name in ("eligible", "features", "target"):
        tables[name], identities[name] = read_csv(getattr(args, name + "_csv"))
    eligible = shared.index_rows([row for row in tables["eligible"] if row.get("family") == args.family], "eligible family")
    features = shared.index_rows([row for row in tables["features"] if row.get("family") == args.family and shared.finite_number(row.get("prefix_cutoff_supplied_cycle_number")) == 100], "K100 family features")
    targets = shared.index_rows([row for row in tables["target"] if row.get("family") == args.family], "family targets")
    contexts, context_columns = {}, ()
    if args.context_csv is not None:
        tables["context"], identities["context"] = read_csv(args.context_csv)
        matching = [row for row in tables["context"] if row.get("family") == args.family]
        contexts = shared.index_rows(matching, "family prior context")
        context_columns = tuple(sorted({name for row in matching for name in row if name.startswith("prior_")}))
        if not context_columns:
            raise ValueError("The supplied context CSV has no explicit prior_ numeric input columns")
    accepted, excluded = [], []
    for identifier, row in sorted(eligible.items()):
        reasons = []
        if not shared.boolean(row.get("prefix_K100_numbers_complete"), "prefix_K100_numbers_complete", identifier):
            reasons.append("SUPPLIED_CYCLES_1_THROUGH_100_INCOMPLETE")
        count = shared.finite_number(row.get("curve_valid_records_K100"))
        if count is None or count < 80:
            reasons.append("FEWER_THAN_80_VALID_K100_OBSERVED_CURVES")
        feature_row = features.get(identifier, {})
        values = {name: shared.finite_number(feature_row.get(name)) for name in REQUIRED}
        missing = [name for name, value in values.items() if value is None]
        if missing:
            reasons.append("MISSING_PREFIX_FEATURES=" + ",".join(missing))
        context_row = contexts.get(identifier, {})
        if args.context_csv is not None:
            context_values = {name: context_number(context_row.get(name)) for name in context_columns}
            missing_context = [name for name, value in context_values.items() if value is None]
            if missing_context:
                reasons.append("MISSING_REGISTERED_PRIOR_CONTEXT=" + ",".join(missing_context))
            values.update(context_values)
        if reasons:
            excluded.append({"cell_id": identifier, "reasons": "|".join(reasons)})
            continue
        target_row = targets.get(identifier, {})
        accepted.append({
            "cell_id": identifier,
            "family": args.family,
            "legacy_576_member": shared.boolean(row.get("legacy_576_member"), "legacy_576_member", identifier),
            "protocol_group": context_row.get("protocol_group", "UNAVAILABLE"),
            "physical_cycle_axis_authenticated": shared.boolean(row.get("physical_cycle_axis_authenticated"), "physical_cycle_axis_authenticated", identifier),
            "target_coordinate": target_row.get("target_coordinate", "supplied_cycle_number"),
            "features": values,
            "targets": {name: shared.finite_number(target_row.get(name)) for name in TARGETS},
            "target_reasons": {name: target_row.get(name + "_reason", "UNAVAILABLE") for name in TARGETS},
        })
    return accepted, excluded, identities, {name: len(rows) for name, rows in tables.items()}, context_columns


def assign_base_roles(cells):
    historical = sorted([row for row in cells if row["legacy_576_member"]], key=lambda row: (shared.sha256((SPLIT_PREFIX + row["cell_id"]).encode()), row["cell_id"]))
    count = math.floor(0.75 * len(historical))
    train = {row["cell_id"] for row in historical[:count]}
    validation = {row["cell_id"] for row in historical[count:]}
    for row in cells:
        row["role"] = "train" if row["cell_id"] in train else "val" if row["cell_id"] in validation else "test"


def target_support(cells, target):
    available = [row for row in cells if row["targets"][target] is not None]
    counts = {role: sum(row["role"] == role for row in available) for role in ("train", "val", "test")}
    reasons = [f"{role} has {counts[role]} observed target cells, requires {minimum}" for role, minimum in (("train", 2), ("val", 1), ("test", 1)) if counts[role] < minimum]
    return {"status": "READY" if not reasons else "INSUFFICIENT_SUPPORT", "role_counts": counts,
            "available_cell_ids": [row["cell_id"] for row in available],
            "missing_cell_ids": [row["cell_id"] for row in cells if row["targets"][target] is None],
            "unsupported_reasons": reasons}


def register(args):
    cells, excluded, identities, row_counts, context = load_inputs(args)
    assign_base_roles(cells)
    split_columns = ("cell_id", "family", "role", "legacy_576_member", "protocol_group", "physical_cycle_axis_authenticated", "target_coordinate")
    split = [{name: row[name] for name in split_columns} for row in cells]
    split_payload = shared.csv_bytes(split, split_columns)
    support = {target: target_support(cells, target) for target in TARGETS}
    ready = [target for target, details in support.items() if details["status"] == "READY"]
    models = make_models(context)
    contract = {
        "version": "MATR_STANFORD_OBSERVED_CURVE_EXPERIMENT_B_EXTENSION_V1_20261115",
        "family": args.family, "status": "READY" if ready else "INSUFFICIENT_SUPPORT",
        "inputs": identities, "input_row_counts": row_counts,
        "implementation_sha256": shared.sha256(Path(__file__).read_bytes()),
        "shared_hust_implementation_sha256": shared.sha256(SOURCE.read_bytes()),
        "software": {"numpy": np.__version__, "scipy": shared.scipy.__version__, "sklearn": shared.sklearn.__version__},
        "prefix": 100,
        "coordinate": "supplied_cycle_number",
        "measurement": "provided phase-confirmed discharge maximum minus finite whole-record baseline, divided by supplied nominal capacity",
        "targets": {"primary": TARGETS[0], "secondary": TARGETS[1], "unit": "capacity_ratio_per_supplied_cycle"},
        "cohort_rule": "Base cells have supplied cycles1..100 complete, at least80 qualified observed curves and finite required prefix/context inputs. Physical-axis authentication is evidence, outside model admission rules.",
        "target_mask_rule": "Fix the base split first. Use each target's observed cells within those roles. Do not resplit or take the both-target intersection. Every model for a target uses identical cells.",
        "role_counts": dict(Counter(row["role"] for row in cells)),
        "split_rule": {"historical_members": "train or validation", "other_members": "test", "sort_key": "sha256('physics-B-extension-20261115|'+cell_id)", "train_count": "floor(0.75*historical_base_count)"},
        "split_sha256": shared.sha256(split_payload),
        "target_support": support,
        "models": {name: list(columns) for name, columns in models.items()},
        "context": {"enabled": args.context_csv is not None, "input_columns": list(context), "missingness": "Context generator supplies explicit missing flags and finite placeholders. Boolean CSV flags map to zero/one. Runner never estimates missing context from later records.", "protocol_group_used_as_numeric_input": False, "context_only_without_context": "equal to the source-train target mean"},
        "selection": {"alphas": list(shared.ALPHAS), "metric": "validation_MAE", "ties": "first ascending alpha", "scaler_fit": "target-specific train cells only", "selected_refit_on_validation": False},
        "test_policy": "Freeze all supported-target selections before any test prediction. Atomically cache each target/model array. Completed family evaluation rejects rescoring.",
        "paired_comparison": {"candidate": "combined", "control": "context_capacity", "unit": "target-specific test cell", "resamples": 2000, "seed": shared.BOOTSTRAP_SEED, "improvement": "abs_error(control)-abs_error(candidate)", "interval": "percentile95"},
        "protocol_comparison": {"enabled": args.context_csv is not None, "metric": "equal-weight mean of each observed protocol group's paired cell MAE improvement", "bootstrap_unit": "protocol group", "resamples": 2000, "seed": shared.BOOTSTRAP_SEED, "minimum_groups_for_interval": 2, "group_labels": "context CSV protocol_group, retained for stratification"},
        "evidence": {"strict_physical_axis_authenticated_base_cells": sum(row["physical_cycle_axis_authenticated"] for row in cells), "base_cells": len(cells)},
        "limitations": [
            "Historical-data exploratory holdout with previous benchmark exposure.",
            "Supplied-coordinate prediction does not certify original physical-cycle numbering or recover deleted cycles.",
            "MATR data include multiple charging policies. Stanford source curves were interpolated. Per-family source reports retain those differences.",
            "Target availability differs across roles and targets. Report denominators and missing support.",
            "Context parameters extend the VIQ-only input budget equally for all compared models when enabled.",
            "Bootstrap intervals condition on fitted models and the observed test cohort.",
        ],
        "excluded_cells": excluded,
    }
    shared.write_once(args.output / "split.csv", split_payload)
    shared.write_once(args.output / "excluded_cells.csv", shared.csv_bytes(excluded, ("cell_id", "reasons")))
    support_rows = [{"cell_id": row["cell_id"], "role": row["role"], "target": target,
                     "observed_target": row["targets"][target] is not None,
                     "source_reason": row["target_reasons"][target]} for row in cells for target in TARGETS]
    shared.write_once(args.output / "target_cell_support.csv", shared.csv_bytes(support_rows, ("cell_id", "role", "target", "observed_target", "source_reason")))
    shared.write_once(args.output / "CONTRACT.json", shared.canonical_json(contract))
    return cells, contract, shared.sha256((args.output / "CONTRACT.json").read_bytes())


def engine_for(target, models):
    # Separate module namespaces keep the frozen HUST globals and each target isolated.
    spec = importlib.util.spec_from_file_location("_registered_hust_engine_" + target, SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TARGETS = (target,)
    module.MODELS = {name: tuple(columns) for name, columns in models.items()}
    return module


def select_all(cells, contract, digest):
    selected = {"contract_sha256": digest, "targets": {}}
    for target, support in contract["target_support"].items():
        if support["status"] != "READY":
            continue
        observed = [row for row in cells if row["targets"][target] is not None]
        fitting_models = dict(contract["models"])
        context_absent = not fitting_models["context_only"]
        if context_absent:
            del fitting_models["context_only"]
        target_selection = engine_for(target, fitting_models).fit_selection(observed, digest)
        if context_absent:
            target_selection["models"][target]["context_only"] = dict(target_selection["models"][target]["mean"])
            target_selection["models"][target]["context_only"]["reason"] = "NO_CONTEXT_COLUMNS_TRAIN_MEAN_EQUIVALENT"
        selected["targets"][target] = target_selection
    return selected


def protocol_comparisons(predictions, enabled):
    summaries, groups = [], []
    for target in TARGETS:
        baseline = {row["cell_id"]: row for row in predictions if row["target"] == target and row["model"] == "context_capacity"}
        combined = {row["cell_id"]: row for row in predictions if row["target"] == target and row["model"] == "combined"}
        if not baseline:
            continue
        if set(baseline) != set(combined):
            raise RuntimeError("Protocol comparison models have different test-cell support")
        values = {}
        for identifier, row in baseline.items():
            group = row["protocol_group"]
            other = combined[identifier]
            if row["truth"] != other["truth"] or group != other["protocol_group"]:
                raise RuntimeError("Protocol comparison cell target/group mismatch")
            truth = float(row["truth"])
            base_error = abs(truth - float(row["prediction"]))
            candidate_error = abs(truth - float(other["prediction"]))
            values.setdefault(group, []).append((base_error, candidate_error))
        improvements = []
        for group, pairs in sorted(values.items()):
            array = np.asarray(pairs)
            mean_improvement = float(np.mean(array[:, 0] - array[:, 1]))
            improvements.append(mean_improvement)
            groups.append({"target": target, "protocol_group": group, "n_test_cells": len(pairs),
                           "context_capacity_MAE": float(array[:, 0].mean()), "combined_MAE": float(array[:, 1].mean()),
                           "MAE_improvement": mean_improvement})
        low, high = None, None
        reason = "CONTEXT_PROTOCOL_GROUPS_UNAVAILABLE"
        if enabled:
            reason = "INSUFFICIENT_PROTOCOL_GROUPS_FOR_GROUP_BOOTSTRAP"
            if len(improvements) >= 2:
                array = np.asarray(improvements)
                rng = np.random.default_rng(shared.BOOTSTRAP_SEED)
                samples = array[rng.integers(0, len(array), size=(2000, len(array)))].mean(axis=1)
                low, high = (float(value) for value in np.quantile(samples, [0.025, 0.975]))
                reason = "AVAILABLE"
        summaries.append({"target": target, "comparison": "combined_vs_context_capacity", "protocol_groups": len(improvements),
                          "test_cells": len(baseline), "protocol_macro_MAE_improvement": float(np.mean(improvements)),
                          "CI95_low": low, "CI95_high": high, "bootstrap_unit": "protocol_group",
                          "bootstrap_resamples": 2000, "bootstrap_seed": shared.BOOTSTRAP_SEED, "reason": reason})
    return summaries, groups


def evaluate(cells, contract, selections, digest, output):
    selection_digest = shared.sha256((output / "SELECTIONS.json").read_bytes())
    shared.write_once(output / "TEST_OPENED.json", shared.canonical_json({"contract_sha256": digest, "all_target_selection_sha256": selection_digest}))
    completed_targets, support_rows = [], []
    summary_names = ("predictions.csv", "metrics.csv", "protocol_metrics.csv", "paired_mae_improvement.csv", "negative_cases.csv")
    merged = {name: [] for name in summary_names}
    headers = {}
    for target in TARGETS:
        support = contract["target_support"][target]
        support_rows.append({"target": target, "status": support["status"], **support["role_counts"], "reason": " | ".join(support["unsupported_reasons"])})
        if support["status"] != "READY":
            continue
        target_output = output / "targets" / target
        target_output.mkdir(parents=True, exist_ok=True)
        target_selection = selections["targets"][target]
        shared.write_once(target_output / "SELECTIONS.json", shared.canonical_json(target_selection))
        target_digest = shared.sha256((target_output / "SELECTIONS.json").read_bytes())
        shared.write_once(target_output / "SELECTIONS.sha256", (target_digest + "\n").encode())
        evaluated_path = target_output / "EVALUATED.json"
        if evaluated_path.exists():
            receipt = json.loads(evaluated_path.read_text(encoding="utf-8"))
            if receipt["contract_sha256"] != digest or receipt["selection_sha256"] != target_digest:
                raise RuntimeError("Completed target identity differs from frozen family selection")
            for name, expected in receipt["artifacts_sha256"].items():
                if shared.sha256((target_output / name).read_bytes()) != expected:
                    raise RuntimeError("Completed target artifact hash mismatch")
        else:
            observed = [row for row in cells if row["targets"][target] is not None]
            engine_for(target, contract["models"]).evaluate(observed, target_selection, digest, target_output)
        for name in summary_names:
            rows, _ = read_csv(target_output / name)
            with (target_output / name).open(encoding="utf-8-sig", newline="") as file:
                headers[name] = ("family",) + tuple(csv.DictReader(file).fieldnames)
            merged[name].extend({"family": contract["family"], **row} for row in rows)
        completed_targets.append(target)
    shared.write_once(output / "target_evaluation_status.csv", shared.csv_bytes(support_rows, ("target", "status", "train", "val", "test", "reason")))
    for name, rows in merged.items():
        if name in headers:
            shared.write_once(output / name, shared.csv_bytes(rows, headers[name]))
    macro, group_support = protocol_comparisons(merged["predictions.csv"], contract["protocol_comparison"]["enabled"])
    shared.write_once(output / "protocol_macro_paired_improvement.csv", shared.csv_bytes(macro, ("target", "comparison", "protocol_groups", "test_cells", "protocol_macro_MAE_improvement", "CI95_low", "CI95_high", "bootstrap_unit", "bootstrap_resamples", "bootstrap_seed", "reason")))
    shared.write_once(output / "protocol_pair_support.csv", shared.csv_bytes(group_support, ("target", "protocol_group", "n_test_cells", "context_capacity_MAE", "combined_MAE", "MAE_improvement")))
    artifacts = {path.relative_to(output).as_posix(): shared.sha256(path.read_bytes()) for path in sorted(output.rglob("*")) if path.is_file() and path.name not in ("RUN.lock",) and path != output / "EVALUATED.json"}
    receipt = {"status": "COMPLETE_EXPLORATORY_HOLDOUT" if len(completed_targets) == len(TARGETS) else "COMPLETE_WITH_UNSUPPORTED_TARGETS",
               "family": contract["family"], "contract_sha256": digest, "all_target_selection_sha256": selection_digest,
               "completed_targets": completed_targets, "unsupported_targets": [target for target in TARGETS if target not in completed_targets],
               "target_support": contract["target_support"], "models": list(contract["models"]),
               "test_prediction_arrays": len(completed_targets) * len(contract["models"]), "artifacts_sha256": artifacts}
    shared.write_once(output / "EVALUATED.json", shared.canonical_json(receipt))
    return receipt


def _self_check():
    rows = [{"cell_id": f"synthetic_{i}", "legacy_576_member": i < 8,
             "targets": {TARGETS[0]: float(i), TARGETS[1]: None if i in (0, 2, 5, 8) else float(i)}} for i in range(11)]
    assign_base_roles(rows)
    roles = {row["cell_id"]: row["role"] for row in rows}
    first = target_support(rows, TARGETS[0])
    second = target_support(rows, TARGETS[1])
    assert len(first["available_cell_ids"]) == 11
    assert len(second["available_cell_ids"]) == 7
    assert roles == {row["cell_id"]: row["role"] for row in rows}
    assert len({row["cell_id"] for row in rows}) == len(rows)
    assert tuple(make_models(())) == ("mean", "context_only", "context_capacity", "context_qv_shape", "combined")
    unsupported = target_support([row for row in rows if row["role"] != "val"], TARGETS[0])
    assert unsupported["status"] == "INSUFFICIENT_SUPPORT"
    assert first["status"] == "READY"
    assert context_number("True") == 1 and context_number("false") == 0
    synthetic_predictions = []
    for target in TARGETS:
        for identifier, group, truth in (("a", "g1", 1.0), ("b", "g1", 2.0), ("c", "g2", 3.0)):
            for model, offset in (("context_capacity", 0.5), ("combined", 0.25)):
                synthetic_predictions.append({"cell_id": identifier, "target": target, "model": model, "protocol_group": group,
                                              "truth": str(truth), "prediction": str(truth + offset)})
    summaries, groups = protocol_comparisons(synthetic_predictions, True)
    assert len(summaries) == 2 and len(groups) == 4
    assert all(row["protocol_macro_MAE_improvement"] == 0.25 for row in summaries)
    assert all(row["CI95_low"] == 0.25 and row["CI95_high"] == 0.25 for row in summaries)
    singleton = [row for row in synthetic_predictions if row["protocol_group"] == "g1"]
    single_summary, _ = protocol_comparisons(singleton, True)
    assert all(row["CI95_low"] is None for row in single_summary)
    return {"status": "PASS", "kind": "synthetic_split_target_support_protocol_macro_selfcheck", "checks": 12, "models_fitted": 0, "test_predictions": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("MATR", "Stanford_2"), required=True)
    parser.add_argument("--eligible-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--context-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with shared.execution_lock(args.output / "RUN.lock"):
        if not args.register_only and (args.output / "EVALUATED.json").exists():
            raise RuntimeError("Completed family evaluation is sealed")
        cells, contract, digest = register(args)
        if args.register_only:
            print(json.dumps({"status": "REGISTERED", "family": args.family, "contract_sha256": digest, "base_role_counts": contract["role_counts"], "target_support": contract["target_support"], "test_predictions_computed": 0}, ensure_ascii=False))
            return 0
        selection_path = args.output / "SELECTIONS.json"
        if selection_path.exists():
            selections = json.loads(selection_path.read_text(encoding="utf-8"))
            if selections["contract_sha256"] != digest:
                raise RuntimeError("Source selection is bound to a different family contract")
        else:
            selections = select_all(cells, contract, digest)
            shared.write_once(selection_path, shared.canonical_json(selections))
        shared.write_once(args.output / "SELECTIONS.sha256", (shared.sha256(selection_path.read_bytes()) + "\n").encode())
        receipt = evaluate(cells, contract, selections, digest, args.output)
        print(json.dumps({"status": receipt["status"], "family": args.family, "completed_targets": receipt["completed_targets"], "test_prediction_arrays": receipt["test_prediction_arrays"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
