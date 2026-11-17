"""Independently replay source-only A/B/C scores and paired comparisons."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.data import ROOT, file_hash
from scripts.drs_peft.e123_protocol import WINDOWS, load_protocol
from scripts.drs_peft.run_abc_a import build_plan as a_plan
from scripts.drs_peft.run_abc_c import build_plan as c_plan
from scripts.drs_peft.run_e123 import unit_path as legacy_unit_path
from scripts.drs_peft.summarize_e123 import (
    audit_update_norms,
    crossed_interval,
    effect_matrix,
    load_source_reference,
    recompute_unit_metrics,
    statistic_seed,
)

PAIR_KEYS = ["fold_id", "seed", "cell_uid", "protocol_uid"]
MODEL_KEYS = ["study", "regime", "mode", "evaluation_mode", "window"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_frozen_bytes(path, payload):
    """Resume an identical summary without replacing an existing byte prefix."""
    path = Path(path)
    if path.exists():
        require(
            path.read_bytes() == payload, "Frozen ABC summary differs at " + str(path)
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_bytes(payload)
    os.replace(pending, path)


def write_frozen_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_frozen_bytes(path, payload.encode("utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def unit_key(unit):
    return json.dumps(unit, sort_keys=True, separators=(",", ":"))


def verify_files(folder, receipt):
    for record in receipt["files"]:
        require(
            file_hash(Path(folder) / record["path"]) == record["sha256"],
            "A bound ABC artifact changed at " + str(folder / record["path"]),
        )


def study_plan(config, protocol, study):
    return c_plan(config, protocol) if study == "C" else a_plan(config, protocol, study)


def source_folder(args, study, unit):
    if study == "C":
        return (
            args.private_root
            / study
            / f"fold{unit['fold_id']}"
            / f"seed{unit['seed']}"
            / unit["arm"]
        )
    return legacy_unit_path(args.private_root / study / unit["regime"], unit)


def is_reused(study, unit):
    return (
        study in ("A", "B")
        and unit["regime"] == "frozen"
        and unit["mode"] != "shuffled_dual"
    )


def check_source_unit(args, config, protocol, study, unit):
    folder = source_folder(args, study, unit)
    receipt = read(folder / "SOURCE_DEVELOPMENT_COMPLETE.json")
    verify_files(folder, receipt)
    run = read(folder / "RUN_CONFIG.json")
    spec = read(folder / f"ABC_{study}_UNIT_SPEC.json")
    fold = protocol["folds"][unit["fold_id"]]
    expected_epochs = int(
        config["C"]["epochs"] if study == "C" else config["epochs"]["condition"]
    )
    expected_updates = expected_epochs * math.ceil(len(fold["fit_ids"]) / 8)
    static = (
        legacy_unit_path(
            args.e123_private_root,
            {
                "stage": "static",
                "fold_id": unit["fold_id"],
                "seed": unit["seed"],
                "method": "grad_core",
            },
        )
        / "CHECKPOINT_SELECTED.pt"
    )
    start_key = (
        "source_C0_checkpoint_sha256" if study == "C" else "static_checkpoint_sha256"
    )
    require(spec["unit"] == unit, "ABC source unit differs from its registered role")
    for label, role in (
        ("fit_ids", "fit"),
        ("inner_validation_ids", "inner_validation"),
        ("outer_ids", "outer_test"),
    ):
        require(
            spec[label] == fold[role + "_ids"],
            "ABC source specification crosses fold roles",
        )
    require(
        run["training_ids"] == fold["fit_ids"]
        and run["validation_ids"] == fold["inner_validation_ids"]
        and run["seed"] == unit["seed"]
        and run["config_sha256"] == file_hash(args.config)
        and spec[start_key] == file_hash(static)
        and run["initial_checkpoint_sha256"] == file_hash(static)
        and run["target_query_labels_used"] == 0,
        "ABC source fit, initialization, or registration binding changed",
    )
    expected_method = (
        "ABC_C_" + unit["arm"] if study == "C" else f"ABC_{study}_{unit['regime']}"
    )
    expected_parameters = (
        config["C"]["trainable_parameters"][unit["arm"]]
        if study == "C"
        else config["A"]["joint_total_trainable_parameters"]
        if unit["regime"] == "joint"
        else config["A"]["core_gate_trainable_parameters"]
    )
    require(
        run["method"] == expected_method
        and run["trainable_parameters"] == expected_parameters,
        "ABC source parameterization or trainable budget changed",
    )
    history = read(folder / "TRAIN_LOG.json")
    best = min(history, key=lambda value: value["validation_mae"])
    require(
        receipt["status"] == "SOURCE_DEVELOPMENT_COMPLETE"
        and receipt["target_query_labels_used"] == 0
        and receipt["epochs"] == run["epochs"] == len(history) == expected_epochs
        and receipt["updates"]
        == run["expected_updates"]
        == history[-1]["updates"]
        == expected_updates
        and receipt["best_epoch"] == best["epoch"],
        "ABC training budget or first strict inner minimum is incomplete",
    )
    checkpoint = folder / "CHECKPOINT_SELECTED.pt"
    require(
        file_hash(checkpoint) == receipt["selected_checkpoint_sha256"],
        "ABC selected checkpoint changed",
    )
    for record in read(folder / "SOURCE_SNAPSHOT.json")["files"]:
        require(
            file_hash(ROOT / record["path"]) == record["sha256"],
            "Executed ABC training source changed",
        )
    return {
        "study": study,
        "unit": unit,
        "folder": str(folder),
        "checkpoint_sha256": file_hash(checkpoint),
        "source_receipt_sha256": file_hash(folder / "SOURCE_DEVELOPMENT_COMPLETE.json"),
        "updates": expected_updates,
    }


def verify_reuse(args, protocol, study, plan):
    path = args.public_root / study / f"ABC_{study}_FROZEN_REUSE.json"
    receipt = read(path)
    expected = [unit for unit in plan["units"] if is_reused(study, unit)]
    require(
        receipt["status"] == "VERIFIED_FROZEN_REUSE"
        and receipt["plan_sha256"] == plan["plan_sha256"]
        and receipt["new_optimizer_updates"] == 0
        and receipt["final_target_query_labels_used"] == 0
        and len(receipt["units"]) == len(expected),
        "ABC frozen reuse matrix is incomplete",
    )
    actual = {unit_key(record["unit"]): record for record in receipt["units"]}
    require(
        set(actual) == {unit_key(unit) for unit in expected},
        "ABC reuse identities differ from the plan",
    )
    records = []
    for unit in expected:
        record = actual[unit_key(unit)]
        old_unit = {key: value for key, value in unit.items() if key != "regime"}
        folder = legacy_unit_path(args.e123_private_root, old_unit)
        source = read(folder / "SOURCE_DEVELOPMENT_COMPLETE.json")
        verify_files(folder, source)
        spec = read(folder / "E123_UNIT_SPEC.json")
        fold = protocol["folds"][unit["fold_id"]]
        require(
            spec["unit"] == old_unit
            and spec["fit_ids"] == fold["fit_ids"]
            and spec["inner_validation_ids"] == fold["inner_validation_ids"]
            and spec["outer_ids"] == fold["outer_test_ids"],
            "Reused E123 source fold or view changed",
        )
        require(
            file_hash(ROOT / record["source_receipt_path"])
            == record["source_receipt_sha256"]
            and file_hash(ROOT / record["selected_checkpoint_path"])
            == record["selected_checkpoint_sha256"]
            == source["selected_checkpoint_sha256"]
            and record["new_optimizer_updates"] == 0,
            "A reused E123 source checkpoint changed",
        )
        records.append(record)
    return records


def helper_unit(study, unit):
    if study == "C":
        return dict(
            unit,
            study=study,
            stage="condition",
            method="grad_core",
            mode=unit["arm"],
            regime="frozen",
            radius=1.0,
        )
    return dict(unit, study=study)


def collect_records(args, config, protocol):
    records, closure, reused = [], [], []
    config_sha = file_hash(args.config)
    for study in args.studies:
        plan = study_plan(config, protocol, study)
        stored = read(
            args.public_root
            / study
            / ("C_PLAN.json" if study == "C" else f"ABC_{study}_PLAN.json")
        )
        for key, value in plan.items():
            require(stored[key] == value, "ABC plan changed after registration")
        opened = read(args.public_root / study / "OUTER_EVALUATION_OPENED.json")
        require(
            opened["plan_sha256"] == plan["plan_sha256"]
            and opened["final_target_query_labels_used"] == 0,
            "ABC outer evaluation barrier changed",
        )
        entries = {unit_key(value["unit"]): value for value in opened["source_units"]}
        new_units = [unit for unit in plan["units"] if not is_reused(study, unit)]
        require(
            set(entries) == {unit_key(unit) for unit in new_units}
            and len(entries) == len(opened["source_units"]),
            "ABC outer barrier omits source units",
        )
        new_records = [
            check_source_unit(args, config, protocol, study, unit) for unit in new_units
        ]
        closure.extend(new_records)
        reuse_records = (
            verify_reuse(args, protocol, study, plan) if study != "C" else []
        )
        reused.extend({"study": study, **value} for value in reuse_records)
        by_unit = {unit_key(value["unit"]): value for value in new_records}
        by_reuse = {unit_key(value["unit"]): value for value in reuse_records}
        for unit in plan["units"]:
            fold = protocol["folds"][unit["fold_id"]]
            for evaluation in config["evaluation_modes"]:
                if is_reused(study, unit):
                    reuse = by_reuse[unit_key(unit)]
                    reference = next(
                        value
                        for value in reuse["outer"]
                        if value["evaluation_mode"] == evaluation
                    )
                    folder = (ROOT / reference["receipt_path"]).parent
                    require(
                        file_hash(folder / "OUTER_COMPLETE.json")
                        == reference["receipt_sha256"],
                        "Reused outer receipt changed",
                    )
                    expected_checkpoint = reuse["selected_checkpoint_sha256"]
                    actual_unit = {
                        key: value for key, value in unit.items() if key != "regime"
                    }
                else:
                    source = by_unit[unit_key(unit)]
                    if study == "C":
                        folder = (
                            args.private_root
                            / study
                            / "outer"
                            / f"fold{unit['fold_id']}"
                            / f"seed{unit['seed']}"
                            / unit["arm"]
                        )
                    else:
                        folder = (
                            args.private_root
                            / study
                            / unit["regime"]
                            / "outer"
                            / legacy_unit_path(Path("."), unit)
                        )
                    folder /= evaluation
                    expected_checkpoint = source["checkpoint_sha256"]
                    entry = entries[unit_key(unit)]
                    entry_checkpoint = entry.get(
                        "checkpoint_sha256", entry.get("selected_checkpoint_sha256")
                    )
                    require(
                        entry_checkpoint == expected_checkpoint,
                        "ABC outer barrier uses another selected weight",
                    )
                    if "source_receipt_sha256" in entry:
                        require(
                            entry["source_receipt_sha256"]
                            == source["source_receipt_sha256"],
                            "ABC barrier source receipt changed",
                        )
                    actual_unit = unit
                receipt = read(folder / "OUTER_COMPLETE.json")
                require(
                    receipt["unit"] == actual_unit
                    and receipt["evaluation_mode"] == evaluation
                    and receipt["checkpoint_sha256"] == expected_checkpoint
                    and receipt["outer_ids"] == fold["outer_test_ids"]
                    and receipt["outer_groups"] == fold["outer_test_groups"]
                    and receipt["final_target_query_labels_used"] == 0
                    and receipt["optimizer_updates"] == 0,
                    "ABC outer receipt differs from the registered cell/weight/evaluation",
                )
                if not is_reused(study, unit):
                    require(
                        receipt["registration_sha256"] == config_sha,
                        "ABC outer scoring registration changed",
                    )
                verify_files(folder, receipt)
                records.append(
                    {
                        "unit": helper_unit(study, unit),
                        "evaluation_mode": evaluation,
                        "folder": folder,
                        "receipt_sha256": file_hash(folder / "OUTER_COMPLETE.json"),
                        "reused": is_reused(study, unit),
                    }
                )
    expected_new = sum(
        config["expected_new_neural_training_units"][
            study if study != "B" else "B_extra"
        ]
        for study in args.studies
    )
    require(len(closure) == expected_new, "ABC new source closure count changed")
    require(
        len({str(record["folder"]) for record in records}) == len(records),
        "Shared A/B comparisons accidentally duplicate outer artifacts",
    )
    if set(args.studies) == {"A", "B", "C"}:
        require(
            len(closure) == 99 and len(reused) == 36 and len(records) == 270,
            "The full ABC matrix is incomplete",
        )
        require(
            sum(value["updates"] for value in closure)
            == config["expected_new_neural_optimizer_updates"],
            "ABC source optimization ledger is incomplete",
        )
    return records, {
        "new_training_units": len(closure),
        "reused_training_units": len(reused),
        "new_optimizer_updates": sum(value["updates"] for value in closure),
        "records": closure,
        "reused": reused,
        "final_target_query_labels_used": 0,
    }


def paired_scores(table, candidate, reference, selectors):
    selected = table[table.n_obs.gt(0)].copy()
    for key, value in selectors.items():
        selected = selected[selected[key].eq(value)]
    candidate_rows = selected[selected["mode"].eq(candidate)]
    reference_rows = selected[selected["mode"].eq(reference)]
    pair = candidate_rows.merge(
        reference_rows,
        on=PAIR_KEYS,
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    require(
        len(pair) == len(candidate_rows) == len(reference_rows) and len(pair) > 0,
        "ABC paired comparison omits a physical cell or seed",
    )
    require(
        pair.n_obs_candidate.eq(pair.n_obs_reference).all(),
        "ABC paired models have different observation support",
    )
    pair["improvement"] = pair.mae_reference - pair.mae_candidate
    return pair


def summarize_pair(pair, label, statistics, interaction=False):
    result = {
        **label,
        "physical_cells": int(pair.cell_uid.nunique()),
        "protocol_groups": int(pair.protocol_uid.nunique()),
        "training_seeds": int(pair.seed.nunique()),
        "mean_absolute_MAE_improvement": float(pair.improvement.mean()),
        "confidence_level": float(statistics["confidence_level"]),
        "bootstrap_draws": int(statistics["bootstrap_draws"]),
    }
    if not interaction:
        require(
            pair.mae_reference.mean() > 0,
            "Relative gain requires positive reference MAE",
        )
        result.update(
            reference_mae=float(pair.mae_reference.mean()),
            candidate_mae=float(pair.mae_candidate.mean()),
            relative_MAE_improvement_percent=float(
                100 * pair.improvement.mean() / pair.mae_reference.mean()
            ),
        )
    for columns, name in (
        (["fold_id"], "per_fold"),
        (["seed"], "per_seed"),
        (["fold_id", "seed"], "per_fold_seed"),
        (["protocol_uid"], "per_protocol"),
    ):
        point = {}
        for key, group in pair.groupby(columns):
            key = key if isinstance(key, tuple) else (key,)
            identity = "|".join(map(str, key))
            point[identity] = {
                "mean_absolute_MAE_improvement": float(group.improvement.mean()),
                "physical_cells": int(group.cell_uid.nunique()),
            }
            if not interaction:
                point[identity]["relative_MAE_improvement_percent"] = float(
                    100 * group.improvement.mean() / group.mae_reference.mean()
                )
        result[name] = point
    result["all_fold_directions_positive"] = all(
        value["mean_absolute_MAE_improvement"] > 0
        for value in result["per_fold"].values()
    )
    result["all_seed_directions_positive"] = all(
        value["mean_absolute_MAE_improvement"] > 0
        for value in result["per_seed"].values()
    )
    for unit in ("cell_uid", "protocol_uid"):
        matrix, seeds, identities = effect_matrix(pair, "improvement", unit)
        require(
            len(seeds) == 3, "All three registered seeds are required for ABC intervals"
        )
        result[unit + "_interval"] = crossed_interval(
            matrix,
            np.random.default_rng(
                statistic_seed(statistics["seed"], unit_key(label) + unit)
            ),
            int(statistics["bootstrap_draws"]),
            float(statistics["confidence_level"]),
        )
        result[unit + "_resampling_units"] = len(identities)
    return result


def neural_contrasts(table, config, studies):
    contrasts, pairs = [], {}
    if "A" in studies:
        a = table[table.study.eq("A")]
        for regime, evaluation, (window, _, _), candidate in itertools.product(
            ("frozen", "joint"),
            config["evaluation_modes"],
            WINDOWS,
            ("absolute", "dual"),
        ):
            selectors = {
                "regime": regime,
                "evaluation_mode": evaluation,
                "window": window,
            }
            pair = paired_scores(a, candidate, "constant", selectors)
            label = dict(
                study="A",
                comparison="gating_within_regime",
                candidate=candidate,
                reference="constant",
                **selectors,
            )
            contrasts.append(summarize_pair(pair, label, config["statistics"]))
            pairs[(regime, evaluation, window, candidate)] = pair
        for evaluation, (window, _, _), candidate in itertools.product(
            config["evaluation_modes"], WINDOWS, ("absolute", "dual")
        ):
            frozen = pairs[("frozen", evaluation, window, candidate)]
            joint = pairs[("joint", evaluation, window, candidate)]
            pair = joint[PAIR_KEYS + ["improvement"]].merge(
                frozen[PAIR_KEYS + ["improvement"]],
                on=PAIR_KEYS,
                suffixes=("_joint", "_frozen"),
                validate="one_to_one",
            )
            require(
                len(pair) == len(joint) == len(frozen),
                "A factorial interaction has unmatched cells",
            )
            pair["improvement"] = pair.improvement_joint - pair.improvement_frozen
            label = {
                "study": "A",
                "comparison": "joint_gate_gain_minus_frozen_gate_gain",
                "candidate": candidate,
                "reference": "constant",
                "regime": "interaction",
                "evaluation_mode": evaluation,
                "window": window,
            }
            contrasts.append(
                summarize_pair(pair, label, config["statistics"], interaction=True)
            )
    if "B" in studies:
        require("A" in studies, "B neural contrasts require the shared A views")
        b = pd.concat(
            (table[table.study.eq("A")], table[table.study.eq("B")]), ignore_index=True
        )
        comparisons = [
            (candidate, "shuffled_dual")
            for candidate in ("absolute", "relative", "dual")
        ] + [("dual", reference) for reference in ("absolute", "relative")]
        for regime, evaluation, (window, _, _), (
            candidate,
            reference,
        ) in itertools.product(
            ("frozen", "joint"), config["evaluation_modes"], WINDOWS, comparisons
        ):
            selectors = {
                "regime": regime,
                "evaluation_mode": evaluation,
                "window": window,
            }
            pair = paired_scores(b, candidate, reference, selectors)
            contrasts.append(
                summarize_pair(
                    pair,
                    dict(
                        study="B",
                        comparison="neural_feature_information",
                        candidate=candidate,
                        reference=reference,
                        **selectors,
                    ),
                    config["statistics"],
                )
            )
    if "C" in studies:
        c = table[table.study.eq("C")]
        comparisons = [
            ("rank4_plus_small", "fixed_rank4"),
            ("matched_residual", "matched_fixed"),
            ("matched_lora", "matched_fixed"),
            ("matched_residual", "matched_lora"),
            ("matched_fixed", "fixed_rank4"),
        ]
        for evaluation, (window, _, _), (candidate, reference) in itertools.product(
            config["evaluation_modes"], WINDOWS, comparisons
        ):
            selectors = {
                "regime": "frozen",
                "evaluation_mode": evaluation,
                "window": window,
            }
            pair = paired_scores(c, candidate, reference, selectors)
            label = dict(
                study="C",
                comparison="matched_capacity"
                if candidate.startswith("matched") and reference.startswith("matched")
                else "capacity_different_secondary",
                candidate=candidate,
                reference=reference,
                **selectors,
            )
            contrasts.append(summarize_pair(pair, label, config["statistics"]))
    return contrasts


def aggregate(table, extra_keys=()):
    valid = table[table.n_obs.gt(0)]
    keys = [*MODEL_KEYS, *extra_keys]
    summary = valid.groupby(keys, as_index=False).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        bias=("bias", "mean"),
        physical_cells=("cell_uid", "nunique"),
        protocol_groups=("protocol_uid", "nunique"),
        training_seeds=("seed", "nunique"),
        cell_seed_rows=("cell_uid", "size"),
    )
    grouped = valid.groupby(
        list(dict.fromkeys([*keys, "seed", "protocol_uid"])), as_index=False
    ).mae.mean()
    macro = (
        grouped.groupby(keys, as_index=False)
        .mae.mean()
        .rename(columns={"mae": "protocol_macro_mae"})
    )
    return summary.merge(macro, on=keys, validate="one_to_one")


def independent_endpoint_rates(reference, ids, lower, upper):
    rates = []
    query = reference["query"].astype(float)
    for uid in ids:
        row = reference["lookup"][uid]
        supported = reference["observed"][row]
        target = reference["target"][row].astype(float)
        first = supported & (query >= lower) & (query < lower + 20)
        last = supported & (query > upper - 20) & (query <= upper)
        require(
            first.sum() >= 10 and last.sum() >= 10,
            "B rate endpoints have insufficient source-cache support",
        )
        rates.append(
            (target[first].mean() - target[last].mean())
            / (query[last].mean() - query[first].mean())
        )
    return np.asarray(rates)


def audit_simple_b_predictions(args, config, protocol, reference):
    root = args.public_root / "B"
    support = read(root / "B_MASK_SUPPORT_AUDIT.json")
    frame = pd.read_csv(root / "B_CELL_ERRORS.csv")
    actual_rows = []
    source_receipts = []
    for fold, seed, (lower, upper), model, arm in itertools.product(
        protocol["folds"],
        config["training_seeds"],
        config["B"]["windows"],
        config["B"]["models"],
        config["B"]["arms"],
    ):
        window = f"{lower}_{upper}"
        suffix = Path(f"fold{fold['fold_id']}") / f"seed{seed}" / window / model / arm
        training = args.private_root / "B/models" / suffix
        source = read(training / "TRAIN_COMPLETE.json")
        verify_files(training, source)
        unit = {
            "fold_id": fold["fold_id"],
            "seed": seed,
            "window": window,
            "model": model,
            "arm": arm,
        }
        require(
            source["identity"]["unit"] == unit
            and source["identity"]["registration_sha256"] == file_hash(args.config)
            and source["outer_labels_used_for_training_or_selection"] == 0
            and source["final_target_query_labels_used"] == 0,
            "Simple B training source identity changed",
        )
        folder = args.private_root / "B/outer" / suffix
        receipt = read(root / "outer" / suffix / "OUTER_COMPLETE.json")
        require(
            receipt["unit"] == unit
            and receipt["training_receipt_sha256"]
            == file_hash(training / "TRAIN_COMPLETE.json")
            and receipt["final_target_query_labels_used"] == 0
            and receipt["optimization_updates"] == 0
            and file_hash(folder / "PREDICTIONS.npz")
            == receipt["private_prediction_sha256"],
            "Simple B outer lineage changed",
        )
        expected_ids = support["windows"][window]["folds"][fold["fold_id"]]["roles"][
            "outer_test"
        ]["ids"]
        with np.load(folder / "PREDICTIONS.npz", allow_pickle=False) as archive:
            ids = archive["ids"].astype(str).tolist()
            target, predicted = archive["target"], archive["prediction"]
        expected_target = independent_endpoint_rates(
            reference, expected_ids, lower, upper
        )
        require(
            ids == expected_ids
            and np.allclose(target, expected_target, rtol=1e-12, atol=1e-14)
            and np.isfinite(predicted).all(),
            "Simple B NPZ differs from independently derived source-cache rates",
        )
        error = np.abs(predicted - expected_target)
        require(
            abs(error.mean() - receipt["metrics"]["MAE"]) < 1e-12,
            "Simple B reported MAE failed independent recomputation",
        )
        for index, uid in enumerate(ids):
            actual_rows.append(
                dict(
                    unit,
                    cell_uid=uid,
                    protocol_uid=protocol["source_qualification"][uid]["protocol_uid"],
                    absolute_error=float(error[index]),
                )
            )
        source_receipts.append(
            {
                "unit": unit,
                "source_receipt_sha256": file_hash(training / "TRAIN_COMPLETE.json"),
                "outer_receipt_sha256": file_hash(
                    root / "outer" / suffix / "OUTER_COMPLETE.json"
                ),
            }
        )
    actual = pd.DataFrame(actual_rows)
    keys = ["fold_id", "seed", "window", "model", "arm", "cell_uid", "protocol_uid"]
    joined = actual.merge(
        frame, on=keys, suffixes=("_actual", "_reported"), validate="one_to_one"
    )
    require(
        len(joined) == len(actual) == len(frame)
        and np.allclose(
            joined.absolute_error_actual,
            joined.absolute_error_reported,
            rtol=1e-12,
            atol=1e-14,
        ),
        "Simple B public cell errors failed independent NPZ/cache recomputation",
    )
    return {
        "status": "PASSED",
        "training_units": len(source_receipts),
        "outer_units": len(source_receipts),
        "cell_predictions": len(actual),
        "rates_independently_derived_from_source_cache": True,
        "receipts": source_receipts,
        "final_target_query_labels_used": 0,
    }


def simple_b_intervals(args, config):
    root = args.public_root / "B"
    complete = read(root / "B_SUMMARY_COMPLETE.json")
    for record in complete["files"]:
        require(
            file_hash(root / record["path"]) == record["sha256"],
            "Original simple B numerical output changed",
        )
    require(
        complete["gate"]["n_models"]
        == config["expected_new_simple_regressor_fits"]
        == 180,
        "Simple B fit matrix remains incomplete",
    )
    frame = pd.read_csv(root / "B_CELL_ERRORS.csv")
    contrasts = []
    for window, model in (
        frame[["window", "model"]].drop_duplicates().itertuples(index=False)
    ):
        subset = frame[(frame.window == window) & (frame.model == model)].rename(
            columns={"absolute_error": "mae", "arm": "mode"}
        )
        subset["n_obs"] = 1
        for candidate, reference in [
            (mode, ref)
            for mode in ("absolute", "relative", "dual")
            for ref in ("baseline", "shuffled_dual")
        ] + [("dual", "absolute"), ("dual", "relative")]:
            pair = paired_scores(subset, candidate, reference, {})
            label = {
                "study": "B_simple",
                "model": model,
                "window": str(window),
                "candidate": candidate,
                "reference": reference,
                "rate_units": "SOH_per_original_cycle",
            }
            contrasts.append(summarize_pair(pair, label, config["statistics"]))
    return contrasts


def gradient_summaries(args, config, protocol):
    frames, receipts = [], []
    for fold in protocol["folds"]:
        for seed in config["training_seeds"]:
            folder = (
                args.private_root
                / "C/heldout_gradients"
                / f"fold{fold['fold_id']}"
                / f"seed{seed}"
            )
            marker = folder / "HELDOUT_GRADIENT_COMPLETE.json"
            receipt = read(marker)
            verify_files(folder, receipt)
            require(
                receipt["fold_id"] == fold["fold_id"]
                and receipt["seed"] == seed
                and receipt["outer_ids"] == fold["outer_test_ids"]
                and receipt["registration_sha256"] == file_hash(args.config)
                and receipt["optimizer_updates"] == 0
                and receipt["heldout_gradients_used_for_directions_or_selection"] == 0
                and receipt["final_target_query_labels_used"] == 0,
                "C held-out gradient provenance changed",
            )
            public = (
                args.public_root
                / "C/heldout_gradients"
                / f"fold{fold['fold_id']}"
                / f"seed{seed}"
            )
            require(
                read(public / marker.name) == receipt,
                "C public and private gradient receipts differ",
            )
            frame = pd.read_csv(folder / "HELDOUT_GRADIENT_BLOCKS.csv")
            require(
                len(frame) == 12 * len(fold["outer_test_ids"])
                and set(frame.cell_uid) == set(fold["outer_test_ids"])
                and frame.fold_id.eq(fold["fold_id"]).all()
                and frame.seed.eq(seed).all()
                and not frame.duplicated(["cell_uid", "layer"]).any(),
                "C held-out gradient coverage is incomplete",
            )
            require(
                frame.groupby("cell_uid").layer.nunique().eq(12).all(),
                "C gradient layer family is incomplete",
            )
            expected_groups = frame.cell_uid.map(
                lambda uid: protocol["source_qualification"][uid]["protocol_uid"]
            )
            require(
                frame.protocol_uid.eq(expected_groups).all(),
                "C gradient protocol labels changed",
            )
            nonzero = frame.gradient_norm.gt(0)
            energies = frame.loc[
                nonzero,
                [
                    "fixed_energy_ratio",
                    "left_only_energy_ratio",
                    "right_only_energy_ratio",
                    "double_perp_energy_ratio",
                    "residual_energy_ratio",
                ],
            ]
            require(
                np.isfinite(energies).all().all()
                and (energies >= -1e-8).all().all()
                and frame.block_energy_reconstruction_error.le(1e-5).all(),
                "C gradient energy decomposition failed",
            )
            frames.append(frame)
            receipts.append(
                {
                    "fold_id": fold["fold_id"],
                    "seed": seed,
                    "receipt_sha256": file_hash(marker),
                }
            )
    frame = pd.concat(frames, ignore_index=True)
    require(
        len(frame) == len(protocol["source_ids"]) * len(config["training_seeds"]) * 12,
        "C held-out gradient matrix is incomplete",
    )
    values = [
        "fixed_energy_ratio",
        "left_only_energy_ratio",
        "right_only_energy_ratio",
        "double_perp_energy_ratio",
        "residual_energy_ratio",
        "residual_to_fit_warmup_residual_cosine",
        "full_to_fit_warmup_mean_cosine",
    ]
    grouped = frame.groupby(["seed", "protocol_uid", "layer"], as_index=False)[
        values
    ].mean()
    macro = grouped.groupby("layer", as_index=False)[values].mean()
    return (
        frame,
        macro,
        {
            "status": "PASSED",
            "gradient_cell_layer_rows": len(frame),
            "physical_cells": int(frame.cell_uid.nunique()),
            "protocol_groups": int(frame.protocol_uid.nunique()),
            "new_gradient_evaluations": len(protocol["source_ids"])
            * len(config["training_seeds"]),
            "endpoint_replay_evaluations": 2 * len(receipts),
            "receipts": receipts,
            "heldout_gradients_used_for_directions_or_selection": 0,
            "final_target_query_labels_used": 0,
        },
    )


def markdown(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
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
    studies,
    summary,
    contrasts,
    simple,
    closure,
    audit,
    norms,
    simple_support=None,
    gradient_macro=None,
    simple_models=None,
):
    sections = [
        "# A–C 源工况诊断实验结果",
        "## 执行范围",
        f"本报告覆盖{'、'.join(studies)}。新增真实神经训练{closure['new_training_units']}项，复用已完成冻结训练{closure['reused_training_units']}项，新增优化更新{closure['new_optimizer_updates']}次，独立重算{audit['outer_units']}项外层结果与{audit['cell_window_rows']}条电芯窗口误差。最终目标Query未来标签使用量为0。",
        "源池为115颗物理电芯、62个工况组，沿用三折与42、123、2026三个种子。源池和工况划分存在历史研究暴露，各折训练集重叠，当前证据属于回顾性源内部诊断。",
        "## 模型完整未来与远期误差",
        markdown(
            summary[summary.window.isin(["all", "1001_plus"])][
                [
                    "study",
                    "regime",
                    "mode",
                    "evaluation_mode",
                    "window",
                    "mae",
                    "physical_cells",
                    "protocol_groups",
                ]
            ]
        ),
    ]
    if "A" in studies:
        sections.extend(
            [
                "## A 门控与联合优化的交互",
                "所有组从同折同种子的相同静态C0与Bridge/Head参数出发。适配核心和门参数均为1440，joint额外训练相同的63169个Bridge/Head参数。冻结与联合组使用相同训练轮数、选权方式、损失和实际适配权重增量半径。",
                "下表的收益为对应constant的MAE减去候选MAE。交互项为joint门控收益减去frozen门控收益，正值对应联合优化增强门控收益。",
                contrast_table(contrasts, "A"),
            ]
        )
    if "B" in studies:
        sections.extend(
            [
                "## B 状态特征的信息诊断",
                "神经对照保留absolute、relative、dual与整电芯配对打乱视角，并分别评价冻结和联合Bridge/Head。打乱在fit、inner与outer角色内部独立执行无自配置换，使用前100圈特征。",
                contrast_table(contrasts, "B"),
                "### 退化速率的简单模型诊断",
                "Ridge固定alpha为1。GBR固定100棵树、深度2、最小叶样本数3、学习率0.05。所有组使用相同的六维早期容量参照，附加视角统一为24列。零填充产生的有效秩差异、Ridge有效自由度和实际树叶数保留于各训练凭据。",
                "平均退化速率由窗首尾各20圈的实测SOH均值差除以对应实测循环均值差计算。每窗要求至少80%实测覆盖，首尾各至少10个实测点。",
                simple_support_text(simple_support),
                markdown(simple_models),
                simple_table(simple),
                "### 退化速率对照的逐折与逐种子方向",
                direction_table(simple),
            ]
        )
    if "C" in studies:
        sections.extend(
            [
                "## C 固定子空间外的更新",
                "三组主比较各训练9204个有效参数，分别为扩展固定子空间、原rank4加双侧正交残差，以及去除尺度冗余的rank1 BA。另保留192参数原rank4与300参数小残差补充的次级嵌套对照。Bridge/Head与C0全部冻结，新增方向只由本折fit梯度构造。",
                "匹配容量比较涉及方向分配变化。192与300参数比较包含容量差。残差适配覆盖双侧正交块，完整G减去P_U G P_V还包含单侧交叉块，独立梯度诊断分别保留这些部分。",
                contrast_table(contrasts, "C"),
                "### 独立工况梯度的能量分解",
                "下表先在每个工况内平均，再对工况与训练seed等权平均。G_perp包含双侧正交和两种单侧交叉块，保留其相对能量与训练梯度余弦。",
                markdown(gradient_macro),
            ]
        )
    sections.extend(
        [
            "## 完整未来与远期对照的逐折与逐种子方向",
            direction_table(
                [
                    value
                    for value in contrasts
                    if value["window"] in ("all", "1001_plus")
                ]
            ),
            "## 独立核验与解释范围",
            f"全部NPZ监督与冻结源缓存逐值对齐，电芯窗口MAE、RMSE和bias独立重算最大差为{audit['maximum_metric_difference']:.3g}。实际增量CSV覆盖{sum(value['physical_cell_layers'] for value in norms)}条cell×layer记录，范数控制全部通过为{all(value['norm_control_passed'] for value in norms)}。",
            "所有95%区间采用训练seed与电芯或工况的配对交叉重抽样，抽样次数2000。区间用于描述当前训练设置，各折训练重叠和源历史暴露限制独立确认性解释。每折、每种子、每个工况的反方向结果完整保留。",
            "## 优势与风险",
            "优势为共同起点、折内来源、同组预算及独立数值核验。风险包括历史源暴露、有限远期样本、常数电压统计、模型的不同有效自由度、范数匹配评价的分布变化与LoRA零B起点的Jacobian退化。",
            "## 拓展思考",
            "门控收益与Bridge/Head交互、特征的增量预测信息和子空间外更新收益分别以配对结果解释，单项改善的因果归属还取决于其他对照。",
            "## 你可能想知道",
            "本汇总执行0次训练和0次优化更新，正式目标校准次数为0。执行完整性置信度由来源闭环、逐样本重算和预算审计支持。",
        ]
    )
    write_frozen_bytes(path, ("\n\n".join(sections) + "\n").encode("utf-8"))


def simple_support_text(audit):
    require(audit is not None, "Simple B support audit is required for reporting")
    parts = []
    for window, record in audit["windows"].items():
        fit = [value["roles"]["fit"]["physical_cells"] for value in record["folds"]]
        inner = [
            value["roles"]["inner_validation"]["physical_cells"]
            for value in record["folds"]
        ]
        parts.append(
            f"{window.replace('_', '–')}圈支持{record['source_physical_cells']}颗电芯、{record['source_protocol_groups']}个工况组，各fold fit为{'、'.join(map(str, fit))}颗，inner为{'、'.join(map(str, inner))}颗。"
        )
    return "".join(parts)


def direction_table(contrasts):
    rows = []
    for value in contrasts:
        rows.append(
            {
                "实验": value["study"],
                "模型或组": value.get("model", value.get("regime", "")),
                "评价": value.get("evaluation_mode", "rate"),
                "窗口": value["window"],
                "候选": value["candidate"],
                "参照": value["reference"],
                "各折MAE收益": " / ".join(
                    f"{key}={point['mean_absolute_MAE_improvement']:+.7g}"
                    for key, point in value["per_fold"].items()
                ),
                "各种子MAE收益": " / ".join(
                    f"{key}={point['mean_absolute_MAE_improvement']:+.7g}"
                    for key, point in value["per_seed"].items()
                ),
            }
        )
    return markdown(pd.DataFrame(rows))


def contrast_table(contrasts, study):
    rows = []
    for value in contrasts:
        if value["study"] != study or value["window"] not in ("all", "1001_plus"):
            continue
        rows.append(
            {
                "对照": value["comparison"],
                "组": value["regime"],
                "评价": value["evaluation_mode"],
                "窗口": value["window"],
                "候选": value["candidate"],
                "参照": value["reference"],
                "MAE收益": value["mean_absolute_MAE_improvement"],
                "改善%": value.get("relative_MAE_improvement_percent", "—"),
                "电芯CI": str(value["cell_uid_interval"]),
                "工况CI": str(value["protocol_uid_interval"]),
                "各折均正": value["all_fold_directions_positive"],
            }
        )
    return markdown(pd.DataFrame(rows))


def simple_table(contrasts):
    return markdown(
        pd.DataFrame(
            [
                {
                    "窗口": value["window"],
                    "模型": value["model"],
                    "候选": value["candidate"],
                    "参照": value["reference"],
                    "参照MAE": value["reference_mae"],
                    "候选MAE": value["candidate_mae"],
                    "改善%": value["relative_MAE_improvement_percent"],
                    "各折均正": value["all_fold_directions_positive"],
                    "电芯CI": str(value["cell_uid_interval"]),
                }
                for value in contrasts
            ]
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/drs_peft/abc_v1.json"
    )
    parser.add_argument(
        "--folds-file", type=Path, default=ROOT / "configs/drs_peft/e123_folds_v1.json"
    )
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/abc_v1"
    )
    parser.add_argument(
        "--public-root", type=Path, default=ROOT / "results/drs_peft/abc_v1"
    )
    parser.add_argument(
        "--e123-private-root", type=Path, default=ROOT / "runtime/drs_peft/e123_v1"
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument(
        "--studies", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"]
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    require(
        args.studies == sorted(set(args.studies)),
        "Study scope must be unique and canonically ordered",
    )
    config, protocol = read(args.config), load_protocol(args.folds_file)
    require(
        config["fold_payload_sha256"] == protocol["protocol_sha256"],
        "ABC registration uses another fold payload",
    )
    records, closure = collect_records(args, config, protocol)
    reference = load_source_reference(args.data_root, protocol)
    tables, norms, maximum = [], [], 0.0
    for record in records:
        frame, difference = recompute_unit_metrics(record, reference)
        tables.append(frame)
        maximum = max(maximum, difference)
        norm = audit_update_norms(record)
        norms.append(norm)
    table = pd.concat(tables, ignore_index=True)
    require(
        not table.duplicated([*MODEL_KEYS, *PAIR_KEYS]).any(),
        "ABC repeated a cell-window model score",
    )
    summary = aggregate(table)
    contrasts = neural_contrasts(table, config, args.studies)
    simple = simple_b_intervals(args, config) if "B" in args.studies else []
    simple_metric_audit = (
        audit_simple_b_predictions(args, config, protocol, reference)
        if "B" in args.studies
        else None
    )
    gradient_frame, gradient_macro, gradient_audit = (
        gradient_summaries(args, config, protocol)
        if "C" in args.studies
        else (None, None, None)
    )
    scope = "".join(args.studies)
    prefix = "ABC_" + scope
    report = args.report or ROOT / "report/drs_peft/abc" / (
        "ABC_RESULTS_ZH.md" if scope == "ABC" else f"ABC_{scope}_RESULTS_ZH.md"
    )
    outputs = []
    for suffix, frame in (
        ("CELL_METRICS.csv", table),
        ("MODEL_SUMMARY.csv", summary),
        ("FOLD_SUMMARY.csv", aggregate(table, ("fold_id",))),
        ("SEED_SUMMARY.csv", aggregate(table, ("seed",))),
        ("FOLD_SEED_SUMMARY.csv", aggregate(table, ("fold_id", "seed"))),
        ("PROTOCOL_SUMMARY.csv", aggregate(table, ("protocol_uid",))),
        ("UPDATE_NORMS.csv", pd.DataFrame(norms)),
    ):
        path = args.public_root / (prefix + "_" + suffix)
        write_frozen_bytes(path, frame.to_csv(index=False).encode("utf-8"))
        outputs.append(path)
    audit = {
        "status": "PASSED",
        "outer_units": len(records),
        "cell_window_rows": len(table),
        "maximum_metric_difference": maximum,
        "source_supervision_exactly_matched": True,
        "final_target_query_labels_used": 0,
    }
    payloads = {
        "SOURCE_CLOSURE.json": closure,
        "CONTRASTS.json": {"comparisons": contrasts},
        "B_SIMPLE_INTERVALS.json": {"comparisons": simple},
        "INDEPENDENT_METRIC_AUDIT.json": audit,
        "NORM_AUDIT.json": {
            "all_norm_controls_passed": all(
                value["norm_control_passed"] for value in norms
            ),
            "matched_norm_unattainable_cell_layers": sum(
                value["matched_norm_unattainable"]
                for value in norms
                if value["evaluation_mode"] == "matched_norm"
            ),
            "outer_units": len(norms),
        },
    }
    if simple_metric_audit is not None:
        payloads["B_SIMPLE_INDEPENDENT_AUDIT.json"] = simple_metric_audit
    for suffix, value in payloads.items():
        path = args.public_root / (prefix + "_" + suffix)
        write_frozen_json(path, value)
        outputs.append(path)
    simple_support = (
        read(args.public_root / "B/B_MASK_SUPPORT_AUDIT.json")
        if "B" in args.studies
        else None
    )
    simple_models = (
        pd.read_csv(args.public_root / "B/B_MODEL_SUMMARY.csv")
        if "B" in args.studies
        else None
    )
    if gradient_frame is not None:
        for suffix, frame in (
            ("GRADIENT_CELL_LAYERS.csv", gradient_frame),
            ("GRADIENT_PROTOCOL_MACRO.csv", gradient_macro),
        ):
            path = args.public_root / (prefix + "_" + suffix)
            write_frozen_bytes(path, frame.to_csv(index=False).encode("utf-8"))
            outputs.append(path)
        path = args.public_root / (prefix + "_GRADIENT_AUDIT.json")
        write_frozen_json(path, gradient_audit)
        outputs.append(path)
    write_report(
        report,
        args.studies,
        summary,
        contrasts,
        simple,
        closure,
        audit,
        norms,
        simple_support,
        gradient_macro,
        simple_models,
    )
    outputs.append(report)
    write_frozen_json(
        args.public_root / (prefix + "_SUMMARY_COMPLETE.json"),
        {
            "status": "COMPLETE",
            "studies": args.studies,
            "implementation_sha256": file_hash(__file__),
            "config_sha256": file_hash(args.config),
            "folds_sha256": file_hash(args.folds_file),
            "outer_records": [
                {
                    "unit": record["unit"],
                    "evaluation_mode": record["evaluation_mode"],
                    "receipt_sha256": record["receipt_sha256"],
                }
                for record in records
            ],
            "files": [
                {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path)}
                for path in outputs
            ],
            "optimizer_updates": 0,
            "formal_target_calibration_fits": 0,
            "final_target_query_labels_used": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "studies": args.studies,
                "new_training_units": closure["new_training_units"],
                "reused_training_units": closure["reused_training_units"],
                "outer_units": len(records),
                "cell_window_rows": len(table),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
