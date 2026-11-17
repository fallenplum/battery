"""Aggregate already-scored one-sided few-shot cells without outcome loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.e123_protocol import WINDOWS
from scripts.drs_peft.onesided_protocol import (
    DEFAULT_PATH,
    ROOT,
    SOURCE_METHODS,
    load_registration,
)

STAGES = ("D2", "D3", "D4")
METHOD_ALIASES = {
    "ZERO": "ZERO",
    "CAL2": "CAL2",
    "CORE": "CORE",
    "OUT": "OUT",
    "IN": "IN",
    "NORMAL": "NORMAL",
    "LORA": "LoRA",
    "LORA_FA": "LoRA_FA",
    "LORA-FA": "LoRA_FA",
    "AUTO": "AUTO",
    "CROSS": "CROSS",
    "RANDOM": "RANDOM",
}
METHOD_ALIASES.update(
    {name + "_COMMON": name + "_COMMON" for name in ("CORE", "OUT", "IN", "NORMAL")}
)
METRIC_COLUMNS = (
    "stage",
    "domain",
    "task_id",
    "fold_id",
    "seed",
    "method",
    "K",
    "draw",
    "adaptation_id",
    "is_duplicate",
    "cell_uid",
    "protocol_uid",
    "window",
    "n_obs",
    "mae",
    "rmse",
    "bias",
    "zero_mae",
    "gain_mae",
    "negative_transfer",
    "same_protocol_support_count",
)
GROUP_KEYS = ["stage", "domain", "method", "K", "window"]
SLOT_KEYS = ["stage", "task_id", "seed", "method", "K", "draw"]
TERMINAL_FAILURES = {
    "NOT_ESTIMABLE",
    "NOT_FEASIBLE",
    "NUMERICAL_FAILURE",
    "FAILED",
    "ERROR",
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def method_name(value):
    key = str(value).upper()
    if key not in METHOD_ALIASES:
        raise ValueError("Unknown method name in scored results " + str(value))
    return METHOD_ALIASES[key]


def parse_boolean(value, allow_missing=False):
    if pd.isna(value) and allow_missing:
        return False
    if value in (True, 1, "True", "true", "1"):
        return True
    if value in (False, 0, "False", "false", "0"):
        return False
    raise ValueError("An outcome boolean has an unsupported representation")


def normalize_table(frame):
    missing = set(METRIC_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Scored cells omit columns " + ", ".join(sorted(missing)))
    result = frame.copy()
    for name in (
        "stage",
        "domain",
        "task_id",
        "adaptation_id",
        "cell_uid",
        "protocol_uid",
        "window",
    ):
        if result[name].isna().any():
            raise ValueError("A scored cell has a missing identity field " + name)
        result[name] = result[name].astype(str)
    result["method"] = result.method.map(method_name)
    for name in ("seed", "K", "draw", "n_obs", "same_protocol_support_count"):
        numeric = pd.to_numeric(result[name], errors="raise")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("A scored integer field is invalid " + name)
        result[name] = numeric.astype(int)
    result["fold_id"] = (
        result.fold_id.fillna("final").astype(str).str.removesuffix(".0")
    )
    result["is_duplicate"] = result.is_duplicate.map(parse_boolean)
    result["negative_transfer"] = result.negative_transfer.map(
        lambda value: parse_boolean(value, True)
    )
    for name in ("mae", "rmse", "bias", "zero_mae", "gain_mae"):
        result[name] = pd.to_numeric(result[name], errors="raise")
    return result


def normalize_receipts(records):
    """Support the runner's nested receipts and the original flat schema."""
    normalized = []
    for original in records:
        record = dict(original)
        details = record.get("details", {})
        parameters = details.get("parameter_report", {})
        counters = record.get("new_execution_counters", {})
        aliases = {
            "SIDs": record.get("support_ids"),
            "QIDs": record.get("query_ids"),
            "actual_optsteps": counters.get("nonlinear_optimizer_updates"),
            "state_optimizer_steps": details.get("optimizer_updates"),
            "cache_solver": details.get("solver"),
            "logtime": record.get("elapsed_seconds"),
            "dictbytes": parameters.get("frozen_dictionary_bytes"),
            "frozen_A_bytes": parameters.get("frozen_A_bytes"),
            "linearization_gap": details.get("query_gap_MAE"),
            "linearization_gap_scope": details.get("query_diagnostic_scope"),
            "support_gap_MAE": details.get("support_gap_MAE"),
            "support_mae_before": details.get("support_mae_before"),
            "support_mae_after": details.get("support_mae_after"),
            "peak_gpu_bytes": record.get("peak_gpu_allocated_bytes"),
            "trainable_parameters": parameters.get("adaptation_parameters"),
            "actual_q": details.get("dimension"),
        }
        for name, value in aliases.items():
            if name not in record and value is not None:
                record[name] = value
        normalized.append(record)
    return normalized


def task_index(episodes):
    source = {task["task_id"]: task for task in episodes["source_tasks"]["outer"]}
    target = {task["task_id"]: task for task in episodes["target_tasks"]}
    return source, target


def validate_scored_cells(frame, episodes):
    """Validate identities, masks and score arithmetic using scored facts only."""
    if frame.empty:
        return frame.copy()
    frame = normalize_table(frame)
    if not set(frame.stage).issubset(STAGES):
        raise ValueError("The cell table contains an unregistered evaluation stage")
    if not set(frame.window).issubset(label for label, _, _ in WINDOWS):
        raise ValueError("The cell table contains an unregistered trajectory window")
    if (frame[["n_obs", "K", "draw", "same_protocol_support_count"]] < 0).any().any():
        raise ValueError("Counts and shot budgets must be nonnegative")
    unique = SLOT_KEYS + ["cell_uid", "window"]
    if frame.duplicated(unique).any():
        raise ValueError(
            "A physical Query cell-window is repeated within an evaluation slot"
        )
    valid = frame.n_obs > 0
    numeric = frame.loc[
        valid, ["mae", "rmse", "bias", "zero_mae", "gain_mae"]
    ].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("An observed Query window has a nonfinite score")
    if (frame.loc[valid, ["mae", "rmse", "zero_mae"]] < 0).any().any():
        raise ValueError("Absolute errors must be nonnegative")
    if not np.allclose(
        frame.loc[valid, "gain_mae"],
        frame.loc[valid, "zero_mae"] - frame.loc[valid, "mae"],
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("Adaptation gain differs from ZERO minus adapted MAE")
    if not np.array_equal(
        frame.loc[valid, "negative_transfer"], frame.loc[valid, "gain_mae"] < 0
    ):
        raise ValueError("Negative-transfer flags disagree with the signed gain")
    if (
        frame.loc[~valid, ["mae", "rmse", "bias", "zero_mae", "gain_mae"]]
        .notna()
        .any()
        .any()
    ):
        raise ValueError("An unsupported trajectory window must retain NA metrics")
    if frame.groupby(["task_id", "cell_uid", "window"]).n_obs.nunique().gt(1).any():
        raise ValueError(
            "Query observation support changed across methods, draws or seeds"
        )
    if (
        frame.groupby(["task_id", "seed", "cell_uid", "window"])
        .zero_mae.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("The shared f0 reference changed across adaptation methods")
    source, target = task_index(episodes)
    for (stage, task_id, seed, method, k, draw), rows in frame.groupby(
        SLOT_KEYS, sort=False
    ):
        tasks = target if stage == "D4" else source
        if task_id not in tasks:
            raise ValueError(
                "An evaluation task escaped its registered source or target role"
            )
        task = tasks[task_id]
        if rows.domain.nunique() != 1 or rows.domain.iloc[0] != task["domain"]:
            raise ValueError("An evaluation task changed its data domain")
        expected_fold = str(task["fold_id"]) if stage != "D4" else "final"
        if not rows.fold_id.eq(expected_fold).all():
            raise ValueError("An evaluation task changed its registered source fold")
        allowed = (
            set(SOURCE_METHODS)
            | {name + "_COMMON" for name in ("CORE", "OUT", "IN", "NORMAL")}
            if stage == "D2"
            else {"AUTO", "CROSS", "RANDOM"}
            if stage == "D3"
            else {"CAL2", "OUT", "LoRA", "LoRA_FA"}
        )
        if method != "ZERO" and method not in allowed:
            raise ValueError(
                "An adaptation method escaped its registered evaluation stage"
            )
        if rows.adaptation_id.nunique() != 1 or rows.adaptation_id.iloc[0] == "":
            raise ValueError("One evaluation slot must identify one adapted state")
        if seed not in episodes["training_seeds"] or (
            stage == "D3" and seed not in episodes["D3_training_seeds"]
        ):
            raise ValueError("An evaluation used an unregistered model seed")
        if k == 0:
            if method != "ZERO" or draw != 0 or rows.is_duplicate.any():
                raise ValueError("K0 must be one unduplicated shared ZERO reference")
            if not np.allclose(
                rows.loc[rows.n_obs > 0, "mae"],
                rows.loc[rows.n_obs > 0, "zero_mae"],
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError("The K0 score differs from the shared ZERO reference")
        else:
            if method == "ZERO" or k not in episodes["shots"][1:]:
                raise ValueError("A nonzero adaptation method has invalid shot count")
            shots = next(item for item in task["shots"] if item["k"] == k)
            requested = next(
                (item for item in shots["draws"] if item["draw"] == draw), None
            )
            if requested is None:
                raise ValueError("A support draw was not preregistered")
            if rows.is_duplicate.nunique() != 1 or bool(rows.is_duplicate.iloc[0]) != (
                requested["duplicate_of_draw"] is not None
            ):
                raise ValueError(
                    "Duplicate Support sets were not reported consistently"
                )
        expected_ids = set(task["query_ids"])
        if not set(rows.cell_uid).issubset(expected_ids):
            raise ValueError("A Query identity changed with method, K or support draw")
        protocols = task.get("protocol_by_id", episodes["source_protocols"])
        for uid, group in rows.groupby("cell_uid", sort=False):
            if (
                group.protocol_uid.nunique() != 1
                or group.protocol_uid.iloc[0] != protocols[uid]
            ):
                raise ValueError(
                    "A scored Query changed its descriptive protocol identity"
                )
            support = [] if k == 0 else requested["support_ids"]
            count = sum(protocols[item] == protocols[uid] for item in support)
            if not group.same_protocol_support_count.eq(count).all():
                raise ValueError("Same-protocol Support coverage was miscounted")
    return frame


def expected_slots(episodes, stages, seeds):
    source, target = task_index(episodes)
    result = []
    for stage in stages:
        tasks = list(target.values()) if stage == "D4" else list(source.values())
        if stage == "D4":
            tasks = [
                task for task in tasks if task["selection_role"] == "primary_target"
            ]
        active_seeds = [
            seed
            for seed in seeds
            if stage != "D3" or seed in episodes["D3_training_seeds"]
        ]
        families = (
            list(SOURCE_METHODS)
            if stage == "D2"
            else ["AUTO", "CROSS", "RANDOM"]
            if stage == "D3"
            else ["CAL2", "GENERIC", "OUT"]
        )
        for task in tasks:
            for seed in active_seeds:
                if stage != "D3":
                    result.append(
                        {
                            "stage": stage,
                            "task_id": task["task_id"],
                            "domain": task["domain"],
                            "seed": seed,
                            "method": "ZERO",
                            "K": 0,
                            "draw": 0,
                            "query_ids": task["query_ids"],
                        }
                    )
                for family in families:
                    for shot in task["shots"]:
                        if shot["k"] == 0 or shot["status"] != "FEASIBLE":
                            continue
                        for draw in shot["draws"]:
                            result.append(
                                {
                                    "stage": stage,
                                    "task_id": task["task_id"],
                                    "domain": task["domain"],
                                    "seed": seed,
                                    "method": family,
                                    "K": shot["k"],
                                    "draw": draw["draw"],
                                    "query_ids": task["query_ids"],
                                }
                            )
    return result


def family_matches(actual, family):
    return (
        method_name(actual) in ("LoRA", "LoRA_FA")
        if family == "GENERIC"
        else method_name(actual) == family
    )


def terminal_record(slot, records):
    for record in records:
        if str(record.get("status", "")).upper() not in TERMINAL_FAILURES:
            continue
        if (
            record.get("stage") != slot["stage"]
            or record.get("task_id") != slot["task_id"]
        ):
            continue
        if int(record.get("seed", -1)) != slot["seed"] or not family_matches(
            record.get("method", "ZERO"), slot["method"]
        ):
            continue
        if record.get("K") is not None and int(record["K"]) != slot["K"]:
            continue
        if record.get("draw") is not None and int(record["draw"]) != slot["draw"]:
            continue
        return record
    return None


def slot_coverage(frame, episodes, records, stages, seeds):
    expected = expected_slots(episodes, stages, seeds)
    output = []
    labels = {label for label, _, _ in WINDOWS}
    actual_slots = {key: rows for key, rows in frame.groupby(SLOT_KEYS, sort=False)}
    for slot in expected:
        methods = (
            ("LoRA", "LoRA_FA") if slot["method"] == "GENERIC" else (slot["method"],)
        )
        groups = [
            actual_slots[key]
            for method in methods
            if (
                key := (
                    slot["stage"],
                    slot["task_id"],
                    slot["seed"],
                    method,
                    slot["K"],
                    slot["draw"],
                )
            )
            in actual_slots
        ]
        selected = pd.concat(groups, ignore_index=True) if groups else frame.iloc[:0]
        if selected.method.nunique() > 1:
            raise ValueError(
                "A target generic slot contains multiple selected baselines"
            )
        expected_pairs = {
            (uid, window) for uid in slot["query_ids"] for window in labels
        }
        actual_pairs = set(zip(selected.cell_uid, selected.window))
        complete = actual_pairs == expected_pairs
        failure = terminal_record(slot, records)
        if complete and failure is not None:
            raise ValueError("A fully scored slot also claims terminal failure")
        status = (
            "SCORED"
            if complete
            else str(failure["status"]).upper()
            if failure is not None
            else "PENDING"
        )
        output.append(
            {key: value for key, value in slot.items() if key != "query_ids"}
            | {
                "status": status,
                "expected_query_cells": len(slot["query_ids"]),
                "expected_cell_window_rows": len(expected_pairs),
                "scored_cell_window_rows": len(actual_pairs),
                "adaptation_id": selected.adaptation_id.iloc[0]
                if len(selected) and selected.adaptation_id.nunique() == 1
                else None,
                "failure_reason": failure.get("error", failure.get("reason"))
                if failure is not None
                else None,
            }
        )
    return pd.DataFrame(output)


def aggregate(frame, extra=()):
    columns = [*GROUP_KEYS, *extra]
    output = []
    for key, rows in frame.groupby(columns, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        valid = rows[rows.n_obs > 0]
        groups = valid.groupby("protocol_uid").mae.mean()
        cells = valid.groupby("cell_uid").mae.mean()
        output.append(
            dict(zip(columns, key, strict=True))
            | {
                "mae": float(valid.mae.mean()) if len(valid) else np.nan,
                "rmse_cell_mean": float(valid.rmse.mean()) if len(valid) else np.nan,
                "bias_cell_mean": float(valid.bias.mean()) if len(valid) else np.nan,
                "zero_mae": float(valid.zero_mae.mean()) if len(valid) else np.nan,
                "gain_mae": float(valid.gain_mae.mean()) if len(valid) else np.nan,
                "relative_gain_percent": float(
                    100 * valid.gain_mae.mean() / valid.zero_mae.mean()
                )
                if len(valid) and valid.zero_mae.mean() > 0
                else np.nan,
                "negative_transfer_ratio": float(valid.negative_transfer.mean())
                if len(valid)
                else np.nan,
                "protocol_macro_mae": float(groups.mean()) if len(groups) else np.nan,
                "worst_cell_uid": cells.idxmax() if len(cells) else None,
                "worst_cell_mae": float(cells.max()) if len(cells) else np.nan,
                "worst_protocol_uid": groups.idxmax() if len(groups) else None,
                "worst_protocol_mae": float(groups.max()) if len(groups) else np.nan,
                "physical_query_cells": int(valid.cell_uid.nunique()),
                "protocol_groups": int(valid.protocol_uid.nunique()),
                "cell_seed_draw_rows": len(valid),
                "observations_with_repeated_evaluation": int(valid.n_obs.sum()),
                "model_seeds": int(rows.seed.nunique()),
                "support_draw_slots": int(rows.draw.nunique()),
                "unique_adapted_states": int(rows.adaptation_id.nunique()),
                "same_protocol_support_coverage": float(
                    valid.same_protocol_support_count.gt(0).mean()
                )
                if len(valid)
                else np.nan,
                "unsupported_cell_window_rows": int((rows.n_obs == 0).sum()),
            }
        )
    return pd.DataFrame(output)


def paired_contrasts(frame):
    """Compare registered methods only on identical physical-cell seed draws."""
    output = []
    pairs = {
        "D2": [
            (method, "OUT")
            for method in ("CORE", "IN", "NORMAL", "CAL2", "LoRA", "LoRA_FA")
        ],
        "D3": [("AUTO", "CROSS"), ("RANDOM", "CROSS")],
        "D4": [(method, "OUT") for method in ("CAL2", "LoRA", "LoRA_FA")],
    }
    columns = [
        "stage",
        "domain",
        "K",
        "window",
        "reference_method",
        "candidate_method",
        "reference_mae",
        "candidate_mae",
        "candidate_gain_over_reference",
        "physical_query_cells",
        "paired_cell_seed_draw_rows",
        "candidate_improved_fraction",
    ]
    valid = frame[(frame.n_obs > 0) & (frame.K > 0)]
    identity = ["task_id", "seed", "draw", "cell_uid"]
    for (stage, domain, k, window), rows in valid.groupby(
        ["stage", "domain", "K", "window"], sort=True
    ):
        wide = rows.pivot(index=identity, columns="method", values="mae")
        for reference, candidate in pairs[stage]:
            if reference not in wide or candidate not in wide:
                continue
            paired = wide[[reference, candidate]].dropna()
            if paired.empty:
                continue
            gains = paired[reference] - paired[candidate]
            output.append(
                {
                    "stage": stage,
                    "domain": domain,
                    "K": k,
                    "window": window,
                    "reference_method": reference,
                    "candidate_method": candidate,
                    "reference_mae": float(paired[reference].mean()),
                    "candidate_mae": float(paired[candidate].mean()),
                    "candidate_gain_over_reference": float(gains.mean()),
                    "physical_query_cells": paired.index.get_level_values(
                        "cell_uid"
                    ).nunique(),
                    "paired_cell_seed_draw_rows": len(paired),
                    "candidate_improved_fraction": float(gains.gt(0).mean()),
                }
            )
    return pd.DataFrame(output, columns=columns)


def separated_variability(frame):
    state = aggregate(frame, ("task_id", "seed", "draw"))
    output = []
    for key, rows in state.groupby(GROUP_KEYS, sort=True):
        valid = rows[rows.physical_query_cells > 0]
        # Every physical Query cell has equal weight within its seed. The same
        # cell can recur across nested draws, which are a repeated axis.
        seed_means = (
            valid.assign(total_mae=valid.mae * valid.physical_query_cells)
            .groupby("seed")[["total_mae", "physical_query_cells"]]
            .sum()
        )
        seed_means = seed_means.total_mae / seed_means.physical_query_cells
        draw_std = valid.groupby(["task_id", "seed"]).mae.std(ddof=1)
        output.append(
            dict(zip(GROUP_KEYS, key, strict=True))
            | {
                "seed_std_after_averaging_draws": float(seed_means.std(ddof=1))
                if len(seed_means) > 1
                else np.nan,
                "mean_within_seed_task_draw_std": float(draw_std.mean())
                if draw_std.notna().any()
                else np.nan,
                "seeds": len(seed_means),
                "draw_variance_includes_explicit_repeated_support_slots": True,
                "independent_units": "physical_cells_and_known_protocols;seed_and_draw_are_repeated_evaluation_axes",
            }
        )
    return pd.DataFrame(output)


def validate_shared_source(records):
    sources = {}
    identities = {}
    for record in records:
        adaptation_id = record.get("adaptation_id")
        identity = (
            record.get("task_id"),
            record.get("seed"),
            method_name(record.get("method", "ZERO")),
            record.get("K"),
        )
        if adaptation_id:
            if adaptation_id in identities and identities[adaptation_id] != identity:
                raise ValueError(
                    "An adaptation identity was reused for a different task or method"
                )
            identities[adaptation_id] = identity
        checkpoint = record.get(
            "sourcecheckpoint_sha", record.get("source_checkpoint_sha256")
        )
        if not checkpoint:
            continue
        key = (
            "TARGET_FINAL" if record.get("stage") == "D4" else record.get("task_id"),
            record.get("seed"),
        )
        if key in sources and sources[key] != checkpoint:
            raise ValueError("An adaptation task changed its shared source checkpoint")
        sources[key] = checkpoint


def validate_receipt_identities(frame, records, episodes):
    """Check any supplied Support and Query identity lists without loading data."""
    source, target = task_index(episodes)
    scored = {
        identity: rows for identity, rows in frame.groupby("adaptation_id", sort=False)
    }
    for record in records:
        task = (target if record.get("stage") == "D4" else source).get(
            record.get("task_id")
        )
        if task is None:
            raise ValueError(
                "An adaptation receipt has an unregistered evaluation task"
            )
        if "QIDs" in record and set(record["QIDs"]) != set(task["query_ids"]):
            raise ValueError("An adaptation receipt changed its fixed Query identities")
        if record.get("K") is not None and "SIDs" in record:
            k = int(record["K"])
            shot = next((shot for shot in task["shots"] if shot["k"] == k), None)
            if shot is None or not any(
                set(draw["support_ids"]) == set(record["SIDs"])
                for draw in shot["draws"]
            ):
                raise ValueError(
                    "An adaptation receipt used an unregistered Support set"
                )
            if record.get("draw") is not None:
                draw = next(
                    (
                        draw
                        for draw in shot["draws"]
                        if draw["draw"] == int(record["draw"])
                    ),
                    None,
                )
                if draw is None or set(draw["support_ids"]) != set(record["SIDs"]):
                    raise ValueError(
                        "An adaptation receipt changed the preregistered Support draw"
                    )
        identity = record.get("adaptation_id")
        if identity not in scored:
            continue
        rows = scored[identity]
        expected = (
            record.get("task_id"),
            int(record.get("seed", -1)),
            method_name(record.get("method", "ZERO")),
            int(record.get("K", -1)),
        )
        actual = rows[["task_id", "seed", "method", "K"]].drop_duplicates()
        if len(actual) != 1 or tuple(actual.iloc[0]) != expected:
            raise ValueError(
                "A scored adapted state disagrees with its adaptation receipt"
            )


def adaptation_costs(records):
    """Deduplicate receipts while preserving missing cost measurements as NA."""
    unique = {}
    for record in records:
        identity = record.get("adaptation_id")
        if not identity:
            continue
        previous = unique.get(identity, {})
        current = previous | record
        for key in ("actual_optsteps", "logtime", "dictbytes"):
            values = [
                item[key] for item in (previous, record) if item.get(key) is not None
            ]
            if values:
                current[key] = max(values)
        unique[identity] = current
    rows = []
    for identity, record in unique.items():
        rows.append(
            {
                key: record.get(key)
                for key in (
                    "stage",
                    "task_id",
                    "seed",
                    "K",
                    "status",
                    "actual_optsteps",
                    "cache_solver",
                    "logtime",
                    "dictbytes",
                    "linearization_gap",
                    "support_mae_before",
                    "support_mae_after",
                    "peak_gpu_bytes",
                    "trainable_parameters",
                    "actual_q",
                    "frozen_A_bytes",
                    "state_optimizer_steps",
                    "linearization_gap_scope",
                    "support_gap_MAE",
                    "cache_reused",
                )
            }
            | {
                "adaptation_id": identity,
                "method": method_name(record.get("method", "ZERO")),
                "sourcecheckpoint_sha": record.get(
                    "sourcecheckpoint_sha", record.get("source_checkpoint_sha256")
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "stage",
            "task_id",
            "seed",
            "method",
            "K",
            "adaptation_id",
            "status",
            "actual_optsteps",
            "cache_solver",
            "logtime",
            "dictbytes",
            "linearization_gap",
            "support_mae_before",
            "support_mae_after",
            "peak_gpu_bytes",
            "trainable_parameters",
            "actual_q",
            "frozen_A_bytes",
            "state_optimizer_steps",
            "linearization_gap_scope",
            "support_gap_MAE",
            "cache_reused",
            "sourcecheckpoint_sha",
        ],
    )


def ledger(frame, coverage, records, episodes):
    pending = int(coverage.status.eq("PENDING").sum()) if len(coverage) else 0
    failures = (
        int((~coverage.status.isin(["SCORED", "PENDING"])).sum())
        if len(coverage)
        else 0
    )
    # Explicit dictionary IDs prevent repeated Support draws from inflating fit counts.
    unique = {}
    for record in records:
        identity = record.get("adaptation_id")
        if identity:
            previous = unique.get(identity, {})
            steps = max(
                int(record.get("actual_optsteps", 0) or 0),
                int(previous.get("actual_optsteps", 0) or 0),
            )
            unique[identity] = record | {"actual_optsteps": steps}
    geometric = {"CORE", "OUT", "IN", "NORMAL", "AUTO", "CROSS", "RANDOM", "CAL2"}
    geometric |= {name + "_COMMON" for name in ("CORE", "OUT", "IN", "NORMAL")}
    active = [
        record
        for record in unique.values()
        if int(record.get("K", 0) or 0) > 0
        and str(record.get("status", "COMPLETE")).upper()
        in {"COMPLETE", "SCORED", "SUCCESS"}
    ]
    generic = frame[(frame.stage == "D4") & frame.method.isin(["LoRA", "LoRA_FA"])]
    scored_nonzero = set(frame.loc[frame.K > 0, "adaptation_id"])
    missing_receipts = sorted(scored_nonzero - set(unique))
    missing_checkpoint = sorted(
        identity
        for identity in scored_nonzero & set(unique)
        if not unique[identity].get(
            "sourcecheckpoint_sha", unique[identity].get("source_checkpoint_sha256")
        )
    )
    counters = {}
    for record in records:
        for name, value in record.get("new_execution_counters", {}).items():
            counters[name] = counters.get(name, 0) + int(value)
    return {
        "status": "PARTIAL"
        if pending or missing_receipts or missing_checkpoint
        else "COMPLETE_WITH_FAILURES"
        if failures
        else "COMPLETE",
        "requested_nonzero_slots": int((coverage.K > 0).sum()) if len(coverage) else 0,
        "requested_zero_reference_slots": int(coverage.K.eq(0).sum())
        if len(coverage)
        else 0,
        "scored_slots": int(coverage.status.eq("SCORED").sum()) if len(coverage) else 0,
        "pending_slots": pending,
        "terminal_failed_or_not_estimable_slots": failures,
        "scored_cell_window_rows": len(frame),
        "unique_scored_adaptation_ids": int(
            frame.loc[frame.K > 0, "adaptation_id"].nunique()
        ),
        "unique_recorded_nonzero_adaptations": len(active),
        "scored_adaptation_ids_missing_receipts": missing_receipts,
        "scored_adaptation_receipts_missing_sourcecheckpoint_sha": missing_checkpoint,
        "nonlinear_adapter_fits": sum(
            method_name(record.get("method", "ZERO")) in ("LoRA", "LoRA_FA")
            for record in active
        ),
        "regularized_local_solver_states": sum(
            method_name(record.get("method", "ZERO")) in geometric for record in active
        ),
        "actual_adapter_optimizer_steps": sum(
            record["actual_optsteps"] for record in active
        ),
        "reported_unit_execution_counters": counters,
        "adapter_compute_cost_scope": "selected_stage_adaptation_unit_receipts;inner_hyperparameter_search_and_source_reference_training_accounted_separately",
        "source_reference_fits": "recorded_separately_in_REFERENCE_MODELS.csv;never_inferred_from_adaptation_states",
        "source_D2_requested_all_seed_slots": episodes[
            "source_outer_adaptation_states_all_seeds"
        ],
        "source_D3_requested_seed42_slots": episodes["D3_spectrum_adaptation_states"],
        "target_primary_requested_slots": episodes[
            "D4_primary_nonzero_adaptation_states"
        ],
        "target_primary_zero_references": episodes[
            "D4_primary_zero_reference_task_seed_states"
        ],
        "target_generic_methods_by_K": {
            str(k): sorted(rows.method.unique()) for k, rows in generic.groupby("K")
        },
        "supplementary_target_cell_window_rows": int(
            frame.task_id.eq("B_Stanford_2").sum()
        ),
        "source_common_dimension_supplement_cell_window_rows": int(
            frame.method.str.endswith("_COMMON").sum()
        ),
        "requested_slot_coverage_scope": "all_registered_D2_D3_source_slots_and_D4_primary_target_slots;Stanford_supplementary_separate",
        "MATR_primary_support_budget": "K_per_exact_protocol_task;four_tasks_aggregate_4K_for_eight_fixed_Query_cells",
        "HUST_support_budget": "K_from_fixed_pooled_support_set_for39_fixed_Query_cells",
        "target_Query_exposure_in_this_table": "EXPOSED"
        if frame.stage.eq("D4").any()
        else "NOT_SCORED_HERE",
        "new_model_forwards_in_summarizer": 0,
        "new_gradient_evaluations_in_summarizer": 0,
        "target_future_label_arrays_loaded_by_summarizer": 0,
        "bootstrap_draws": 0,
        "historical_source_exposure": True,
    }


def project_costs(results_root, stages, seeds):
    """Sum one current cumulative receipt per process and completed reference."""
    engines = []
    counter_totals = {}
    for path in sorted((results_root / "ENGINE_COSTS").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("seed") not in seeds or not any(
            str(record.get("role", "")).startswith(stage + "_") for stage in stages
        ):
            continue
        engines.append(
            {
                "path": path.relative_to(results_root).as_posix(),
                "sha256": digest(path.read_bytes()),
                "role": record.get("role"),
                "fold_id": record.get("fold_id"),
                "seed": record.get("seed"),
            }
        )
        for name, value in record["counters"].items():
            if int(value) < 0:
                raise ValueError("A cumulative execution counter became negative")
            counter_totals[name] = counter_totals.get(name, 0) + int(value)
    references = []
    for path in sorted(
        (results_root / "references").glob("*/seed*/REFERENCE_COMPLETE.json")
    ):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("seed") not in seeds:
            continue
        final = str(record["fold_id"]) == "final"
        if (final and "D4" not in stages) or (
            not final and not set(stages) & {"D2", "D3"}
        ):
            continue
        references.append(
            {
                "fold_id": record["fold_id"],
                "seed": record["seed"],
                "reference_checkpoint_sha256": record["reference_checkpoint_sha256"],
                "optimizer_updates": int(record["reference_training_updates"]),
                "source_fits": 2,
            }
        )
    identity = [(record["fold_id"], record["seed"]) for record in references]
    if len(set(identity)) != len(identity):
        raise ValueError("A completed source reference was counted more than once")
    return {
        "engine_counter_scope": "one_current_cumulative_ENGINE_COSTS_file_per_process;D2_D3_D4_selected_scope_includes_inner_and_evaluation;unit_counters_overlap_and_are_separate",
        "engine_receipts": engines,
        "actual_inner_and_evaluation_counters": counter_totals,
        "completed_source_reference_models": len(references),
        "completed_source_training_fits": sum(
            record["source_fits"] for record in references
        ),
        "completed_source_reference_optimizer_updates": sum(
            record["optimizer_updates"] for record in references
        ),
        "source_reference_costs": references,
        "shared_reference_counting": "each_reference_once_in_current_scope;separate_stage_ledgers_share_CV_references_and_must_not_be_added",
    }


def markdown(frame):
    if frame.empty:
        return "当前无已评分结果。"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            "NA"
            if pd.isna(value)
            else f"{value:.6g}"
            if isinstance(value, float)
            else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_latest(path, payload, history):
    path = Path(path)
    if path.exists() and path.read_bytes() != payload:
        old = path.read_bytes()
        snapshot = history / (path.stem + "_" + digest(old)[:16] + path.suffix)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and snapshot.read_bytes() != old:
            raise RuntimeError("An immutable summary history snapshot changed")
        snapshot.write_bytes(old)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_bytes(payload)
    os.replace(pending, path)


def report(summary, variation, run, coverage, costs, contrasts):
    main = (
        summary[summary.window.isin(["all", "1001_plus"])] if len(summary) else summary
    )
    display = [
        "stage",
        "domain",
        "method",
        "K",
        "window",
        "mae",
        "gain_mae",
        "negative_transfer_ratio",
        "physical_query_cells",
        "protocol_groups",
    ]
    failed = (
        coverage[~coverage.status.isin(["SCORED", "PENDING"])]
        if len(coverage)
        else coverage
    )
    sections = [
        "# 单侧方向少样本迁移实验复核",
        "## 当前执行范围",
        f"状态为{run['status']}。已评分{run['scored_slots']}个请求槽，待完成{run['pending_slots']}个，失败或不可估计{run['terminal_failed_or_not_estimable_slots']}个。实际适配优化更新{run['actual_adapter_optimizer_steps']}次。",
        "输入合同为VIQ+earlySOH / no-policy。native_multivariate使用每圈12个统计描述量。源模型从policy维度0初始化，SOH锚点及前缀单位来源单列于输入审计。",
        "## 完整未来与远期K曲线",
        markdown(main[display] if len(main) else main),
        "MATR主任务由原54颗目标池中的4个共同可行精确工况组成，固定8颗Query。每个工况获得K颗Support，聚合信息预算为4K。HUST固定39颗Query，Support预算为跨工况池中的K颗。Stanford仅按供应循环坐标列为附加协议。",
        "## 同Query方法差异",
        markdown(
            contrasts[contrasts.window.isin(["all", "1001_plus"])]
            if len(contrasts)
            else contrasts
        ),
        "candidate_gain_over_reference为逐cell、同seed及draw的参考方法MAE减去候选方法MAE，正值支持候选改善。每行明确当前已评分的配对数量。D3仅使用seed42，三训练种子证据由D2和D4分别提供。",
        "## 抽样与训练种子波动",
        markdown(
            variation[variation.window.eq("all")] if len(variation) else variation
        ),
        "draw波动在同task与seed内计算，seed波动先对draw平均后计算。重复Support集合与相同adaptation_id保留，独立样本数按物理电芯与工况计数。",
        "## 适配成本",
        f"按唯一adaptation_id计，非线性适配器拟合{run['nonlinear_adapter_fits']}次，局部正规化求解{run['regularized_local_solver_states']}次。ADAPTATION_COSTS.csv保留每个唯一状态的实际优化步数、cache_solver、logtime、字典字节与线性化差距，未测量字段保留NA。",
        f"当前共有{len(costs)}条唯一状态回执。源参考模型训练次数在REFERENCE_MODELS.csv单列。",
        f"所选范围的累计engine计数包含inner与评价，实际非线性优化更新{run.get('project_costs', {}).get('actual_inner_and_evaluation_counters', {}).get('nonlinear_optimizer_updates', 'NA')}次。已完成源参考模型{run.get('project_costs', {}).get('completed_source_reference_models', 0)}组、训练阶段拟合{run.get('project_costs', {}).get('completed_source_training_fits', 0)}次，累计源优化更新{run.get('project_costs', {}).get('completed_source_reference_optimizer_updates', 0)}次。不同阶段汇总共享的参考训练只能计一次。",
        "## 失败与不可估计",
        markdown(
            failed[
                [
                    "stage",
                    "task_id",
                    "seed",
                    "method",
                    "K",
                    "draw",
                    "status",
                    "failure_reason",
                ]
            ]
            if len(failed)
            else pd.DataFrame()
        ),
        "远期窗口没有观测时保留NA。Support下降、Query恶化作为负迁移保留，源开发负结果继续进入已锁定的目标评价流程。",
        "## 优势与风险",
        "优势为共同f0、固定Query、嵌套物理Support、fold-local构基及真实非线性前向评分。四种几何臂与通用LoRA基线分别登记维度、冻结存储和优化成本。",
        "风险包括源115颗与三折的历史暴露、训练集重叠、有限远期支持、局部线性近似误差，以及Support协议覆盖不足。source空间方向不赋予未验证的电化学含义。",
        "现有数字的解释以实际覆盖为限。汇总程序只读取已评分表，未读取目标未来标签数组，也未新增bootstrap或模型运行。数值汇总核验置信度取决于请求槽覆盖与算术一致性。",
        "## 拓展思考",
        "结合K曲线、最强通用基线、CAL2、负迁移与成本判断下一步，单独的梯度捕获比例不作为方法成功依据。",
        "## 你可能想知道",
        "输入或超参数选择只使用源内部inner任务。目标一经评分登记为已暴露，当前结果不能重新作为未见目标测试。",
    ]
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/drs_peft/onesided_fewshot_v1",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    episodes = load_registration(args.episodes)
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(
        episodes["training_seeds"]
    ):
        raise ValueError(
            "The summary seed scope must be a distinct subset of registered seeds"
        )
    output = args.output_root or args.results_root
    history = output / "summary_history"
    table_path = args.results_root / "CELL_METRICS.csv"
    frame = (
        pd.read_csv(table_path)
        if table_path.exists()
        else pd.DataFrame(columns=METRIC_COLUMNS)
    )
    if len(frame):
        frame = frame[frame.stage.isin(stages) & frame.seed.isin(args.seeds)].copy()
    frame = validate_scored_cells(frame, episodes)
    records_path = args.results_root / "ADAPTATION_RECORDS.jsonl"
    records = (
        [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if records_path.exists()
        else []
    )
    records = [
        record
        for record in records
        if record.get("stage") in stages and record.get("seed") in args.seeds
    ]
    records = normalize_receipts(records)
    validate_shared_source(records)
    validate_receipt_identities(frame, records, episodes)
    costs = adaptation_costs(records)
    contrasts = paired_contrasts(frame)
    coverage = slot_coverage(frame, episodes, records, stages, args.seeds)
    summary = (
        aggregate(frame)
        if len(frame)
        else pd.DataFrame(
            columns=[*GROUP_KEYS, "mae", "gain_mae", "physical_query_cells"]
        )
    )
    variation = (
        separated_variability(frame) if len(frame) else pd.DataFrame(columns=GROUP_KEYS)
    )
    run = ledger(frame, coverage, records, episodes)
    run["project_costs"] = project_costs(args.results_root, stages, args.seeds)
    names = {
        "D2": "SOURCE_FEWSHOT_RESULTS.csv",
        "D3": "TRANSFER_SPECTRUM_RESULTS.csv",
        "D4": "TARGET_FEWSHOT_RESULTS.csv",
    }
    written = []
    for stage in stages:
        path = output / names[stage]
        write_latest(
            path,
            summary[summary.stage.eq(stage)].to_csv(index=False).encode("utf-8"),
            history,
        )
        written.append(path)
    stem = "" if args.stage == "all" else args.stage + "_"
    for name, data in (
        (
            stem + "NEGATIVE_TRANSFER.csv",
            frame[frame.n_obs.gt(0) & frame.K.gt(0)][list(METRIC_COLUMNS)],
        ),
        (stem + "SLOT_COVERAGE.csv", coverage),
        (stem + "SEED_DRAW_VARIABILITY.csv", variation),
        (
            stem + "TASK_SEED_DRAW_RESULTS.csv",
            aggregate(frame, ("task_id", "seed", "draw"))
            if len(frame)
            else pd.DataFrame(),
        ),
        (stem + "ADAPTATION_COSTS.csv", costs),
        (stem + "PAIRED_METHOD_CONTRASTS.csv", contrasts),
    ):
        path = output / name
        write_latest(path, data.to_csv(index=False).encode("utf-8"), history)
        written.append(path)
    path = output / (stem + "RUN_LEDGER.json")
    write_latest(
        path,
        (json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
        history,
    )
    written.append(path)
    report_path = args.report or ROOT / "report/drs_peft/onesided_fewshot_v1" / (
        "FINAL_REVIEW_ZH.md" if args.stage == "all" else args.stage + "_REVIEW_ZH.md"
    )
    write_latest(
        report_path,
        report(summary, variation, run, coverage, costs, contrasts).encode("utf-8"),
        history,
    )
    written.append(report_path)
    manifest = {
        "status": run["status"],
        "stages": stages,
        "seeds": args.seeds,
        "episode_contract_sha256": episodes["episode_contract_sha256"],
        "selected_scored_cell_rows_sha256": digest(
            frame.to_csv(index=False).encode("utf-8")
        ),
        "summarizer_sha256": digest(Path(__file__).read_bytes()),
        "files": [
            {"path": str(path), "sha256": digest(path.read_bytes())} for path in written
        ],
        "new_model_runs": 0,
        "target_future_label_arrays_loaded": 0,
        "bootstrap_draws": 0,
    }
    write_latest(
        output / (stem + "SUMMARY_MANIFEST.json"),
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        history,
    )
    print(json.dumps(run, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
