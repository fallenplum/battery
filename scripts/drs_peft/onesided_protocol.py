"""Outcome-independent physical-cell episodes for one-sided few-shot transfer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.e123_protocol import load_protocol
from scripts.drs_peft.episodes import load_registration as load_parent_episodes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "configs/drs_peft/onesided_fewshot_v1/EPISODES.json"
SHOTS = (0, 1, 3, 5)
SOURCE_METHODS = ("CORE", "OUT", "IN", "NORMAL", "CAL2", "LoRA", "LoRA_FA")


def payload_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ordered_ids(ids, seed, role):
    values = list(ids)
    if len(values) != len(set(values)):
        raise ValueError("Episode physical identities must be unique")
    return sorted(
        values,
        key=lambda uid: (
            hashlib.sha256(f"{seed}|onesided|{role}|{uid}".encode()).digest(),
            uid,
        ),
    )


def coverage(query_ids, support_ids, protocols):
    support_groups = {protocols[uid] for uid in support_ids}
    per_cell = {uid: protocols[uid] in support_groups for uid in query_ids}
    return {
        "query_cells": len(query_ids),
        "covered_query_cells": sum(per_cell.values()),
        "query_same_protocol_support": per_cell,
        "support_protocol_groups": sorted(support_groups),
    }


def nested_shots(pool, query, protocols, *, seed, role, draws=5):
    pool, query = list(pool), list(query)
    if (
        set(pool) & set(query)
        or len(pool) != len(set(pool))
        or len(query) != len(set(query))
    ):
        raise ValueError("Support and Query must be unique disjoint physical cells")
    permutations = [
        ordered_ids(pool, seed, f"{role}|draw{draw}") for draw in range(draws)
    ]
    shots = []
    for k in SHOTS:
        if len(pool) < k:
            shots.append(
                {
                    "k": k,
                    "status": "NOT_FEASIBLE",
                    "query_ids": query,
                    "draws": [],
                    "available_support_cells": len(pool),
                }
            )
            continue
        seen, records = {}, []
        for draw in range(1 if k == 0 else draws):
            ids = sorted(permutations[draw][:k])
            identity = payload_hash(ids)
            duplicate = seen.get(identity)
            if duplicate is None:
                seen[identity] = draw
            records.append(
                {
                    "draw": draw,
                    "support_ids": ids,
                    "query_ids": query,
                    "support_set_sha256": identity,
                    "duplicate_of_draw": duplicate,
                    "same_protocol_coverage": coverage(query, ids, protocols),
                }
            )
        shots.append(
            {
                "k": k,
                "status": "FEASIBLE",
                "query_ids": query,
                "available_support_cells": len(pool),
                "requested_draws": 1 if k == 0 else draws,
                "unique_support_sets": len(seen),
                "draws": records,
            }
        )
    return shots


def source_task(ids, protocols, fold_id, role, *, seed, query_count, draws):
    order = ordered_ids(ids, seed, f"source|fold{fold_id}|{role}|query")
    if len(order) <= query_count:
        raise ValueError("Registered source task needs a nonempty Support pool")
    query, pool = order[:query_count], order[query_count:]
    task_id = f"SOURCE_{role.upper()}_FOLD{fold_id}"
    return {
        "task_id": task_id,
        "domain": "MATR",
        "fold_id": fold_id,
        "role": role,
        "protocol": "POOLED_UNSEEN_SOURCE_PROTOCOL_SET",
        "support_budget_scope": "K_physical_cells_from_pooled_unseen_protocol_Support_pool_per_fold_task",
        "status": "READY",
        "query_ids": query,
        "support_pool_ids": pool,
        "query_protocol_groups": sorted({protocols[uid] for uid in query}),
        "support_pool_protocol_groups": sorted({protocols[uid] for uid in pool}),
        "shots": nested_shots(
            pool, query, protocols, seed=seed, role=task_id, draws=draws
        ),
    }


def fit_gradient_tasks(fit_ids, protocols, fold_id, seed):
    groups = {}
    for uid in fit_ids:
        groups.setdefault(protocols[uid], []).append(uid)
    pairs, singletons = [], []
    for protocol_uid, ids in sorted(groups.items()):
        if len(ids) < 2:
            singletons.extend(ids)
            continue
        values = sorted(ids)
        pairs.append(
            {
                "task_id": f"FIT_FOLD{fold_id}_{protocol_uid}",
                "protocol_uid": protocol_uid,
                "cell_ids": values,
                "ordered_pair_count": len(values) * (len(values) - 1),
                "ordered_pairs": [
                    {"support_ids": [left], "query_ids": [right]}
                    for left in values
                    for right in values
                    if left != right
                ],
                "physical_cells": len(values),
            }
        )
    return {
        "fold_id": fold_id,
        "fit_ids": sorted(fit_ids),
        "grouping": "known_exact_protocol_before_outcome_access",
        "candidate_block": "OUT",
        "task_weighting": "equal_exact_protocol_task",
        "within_task_pair_weighting": "equal_ordered_distinct_cell_pair",
        "pair_rule": "all_ordered_distinct_physical_cell_pairs_within_same_known_exact_protocol",
        "cross_formula": "(sum_g_sum_g_T-minus-sum_gi_gi_T)/(n*(n-1));then_equal_pairable_protocol_weight",
        "pairs": pairs,
        "pairable_tasks": len(pairs),
        "paired_physical_cells": sum(row["physical_cells"] for row in pairs),
        "ordered_pair_count": sum(row["ordered_pair_count"] for row in pairs),
        "singleton_ids": sorted(singletons),
        "future_outcome_values_used_for_partition": 0,
    }


def inherited_target_task(task, protocols):
    result = copy.deepcopy(task)
    result["parent_task_id"] = task["task_id"]
    result["identity_rule"] = "unchanged_parent_Query_pool_and_nested_Support_draws"
    result["coordinate_scope"] = (
        "supplied_cycle_number_additional_protocol"
        if task["domain"] == "Stanford_2"
        else "original_instrument_cycle"
    )
    result["selection_role"] = (
        "supplementary" if task["domain"] == "Stanford_2" else "primary_target"
    )
    result["support_budget_scope"] = (
        "K_per_exact_protocol_task;four_MATR_tasks_aggregate_4K_Support_cells"
        if task["domain"] == "MATR"
        else "K_physical_cells_from_fixed_pooled_cross_domain_Support_pool"
    )
    result["protocol_by_id"] = {
        uid: protocols[uid] for uid in task["query_ids"] + task["support_pool_ids"]
    }
    result["shots"] = []
    for old in task["shots"]:
        if old["k"] not in SHOTS:
            continue
        shot = copy.deepcopy(old)
        if old["status"] == "FEASIBLE":
            shot["draws"] = shot["draws"][:1] if old["k"] == 0 else shot["draws"]
            for draw in shot["draws"]:
                draw["same_protocol_coverage"] = coverage(
                    task["query_ids"], draw["support_ids"], protocols
                )
            shot["requested_draws"] = len(shot["draws"])
            shot["unique_support_sets"] = len(
                {payload_hash(row["support_ids"]) for row in shot["draws"]}
            )
        result["shots"].append(shot)
    return result


def build_registration(
    root=ROOT, *, episode_seed=42, outer_query_count=20, inner_query_count=8, draws=5
):
    root = Path(root)
    fold_path = root / "configs/drs_peft/e123_folds_v1.json"
    parent_path = root / "configs/drs_peft/episodes_v1.json"
    lock_path = root / "research/drs_peft/DATA_SPLIT_LOCK.json"
    folds, parent = load_protocol(fold_path), load_parent_episodes(parent_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["source_cell_roles"] != parent["source_cell_roles"]:
        raise ValueError("The source role lock and existing target episodes differ")
    if lock["target_tasks"] != parent["target_tasks"]:
        raise ValueError("The original target Support/Query identity lock changed")
    protocols = {
        uid: group for group, ids in parent["source_groups"].items() for uid in ids
    }
    source_ids = sorted(
        parent["source_cell_roles"]["source_train"]
        + parent["source_cell_roles"]["source_meta_validation"]
    )
    if source_ids != sorted(folds["source_ids"]):
        raise ValueError("The one-sided source cohort differs from E1-E3")
    source_tasks, gradient_tasks = {"inner": [], "outer": []}, []
    for fold in folds["folds"]:
        for role, key, count in (
            ("inner", "inner_validation_ids", inner_query_count),
            ("outer", "outer_test_ids", outer_query_count),
        ):
            source_tasks[role].append(
                source_task(
                    fold[key],
                    protocols,
                    fold["fold_id"],
                    role,
                    seed=episode_seed,
                    query_count=count,
                    draws=draws,
                )
            )
        gradient_tasks.append(
            fit_gradient_tasks(
                fold["fit_ids"], protocols, fold["fold_id"], episode_seed
            )
        )
    target_tasks, excluded, target_protocol_provenance = [], [], {}
    for task in parent["target_tasks"]:
        if task["domain"] == "MATR":
            possible = all(
                any(
                    shot["k"] == k and shot["status"] == "FEASIBLE"
                    for shot in task.get("shots", [])
                )
                for k in SHOTS
            )
            if not possible:
                excluded.append(
                    {
                        "task_id": task["task_id"],
                        "domain": "MATR",
                        "status": "NOT_COMMON_K0_K1_K3_K5_FEASIBLE",
                        "query_ids": task["query_ids"],
                        "support_pool_ids": task["support_pool_ids"],
                        "available_cells": task["available_cells"],
                    }
                )
                continue
        task_protocols = dict(protocols)
        if task["domain"] != "MATR":
            path = root / parent["target_identity_audit"][task["domain"]]["file"]
            if task["domain"] == "HUST":
                table = pd.read_parquet(path, columns=["cell_uid", "protocol_uid"])
                mapping = dict(
                    zip(
                        table.cell_uid.astype(str),
                        table.protocol_uid.astype(str),
                        strict=True,
                    )
                )
            else:
                table = pd.read_csv(path, usecols=["cell_id", "protocol_group"])
                mapping = dict(
                    zip(
                        table.cell_id.astype(str),
                        table.protocol_group.astype(str),
                        strict=True,
                    )
                )
            task_protocols.update(mapping)
            target_protocol_provenance[task["domain"]] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_hash(path),
                "columns_read": ["physical_uid", "known_protocol_group"],
                "future_outcome_values_read": 0,
            }
        target_tasks.append(inherited_target_task(task, task_protocols))
    result = {
        "study_id": "DRS_PEFT_ONESIDED_FEWSHOT_V1_20261117",
        "episode_seed": episode_seed,
        "training_seeds": [42, 123, 2026],
        "shots": list(SHOTS),
        "support_draws": draws,
        "source_outer_query_count": outer_query_count,
        "source_inner_query_count": inner_query_count,
        "source_methods": list(SOURCE_METHODS),
        "method_display_names": {"LoRA_FA": "LoRA-FA"},
        "source_ids": source_ids,
        "sealed_final_MATR_ids": sorted(parent["source_cell_roles"]["target_test"]),
        "source_protocols": {uid: protocols[uid] for uid in source_ids},
        "source_tasks": source_tasks,
        "fit_gradient_tasks": gradient_tasks,
        "target_tasks": target_tasks,
        "excluded_from_common_MATR_curve": excluded,
        "source_outer_adaptation_states_per_seed": len(SOURCE_METHODS)
        * len(folds["folds"])
        * (len(SHOTS) - 1)
        * draws,
        "source_outer_adaptation_states_all_seeds": len(SOURCE_METHODS)
        * len(folds["folds"])
        * (len(SHOTS) - 1)
        * draws
        * 3,
        "D3_spectrum_adaptation_states": 3
        * len(folds["folds"])
        * (len(SHOTS) - 1)
        * draws,
        "D3_training_seeds": [42],
        "D4_primary_nonzero_methods": ["CAL2", "INNER_SELECTED_GENERIC", "OUT"],
        "D4_primary_nonzero_adaptation_states": 3 * 3 * 5 * (len(SHOTS) - 1) * draws,
        "D4_primary_zero_reference_task_seed_states": 3 * 5,
        "zero_shot_rule": "one_f0_prediction_per_task_seed_reused_for_all_arms_and_draws",
        "adaptation_state_count_scope": "requested_nonzero_slots_including_repeated_Support_sets;actual_unique_fits_reuse_identical_Support_cache_identity",
        "nested_rule": "same_hash_order_per_draw_first_K_physical_cells;methods_and_K_share_fixed_Query",
        "source_protocol_scope": "pooled_unseen_protocol_set;individual_Query_same_protocol_Support_coverage_reported",
        "input_scope": "VIQ+earlySOH / no-policy",
        "policy_model_dimension": 0,
        "input_allowed": [
            "prefix_VIQ",
            "prefix_mask",
            "prefix_cycle_coordinates",
            "legal_prefix_SOH_anchor",
        ],
        "input_forbidden": [
            "policy",
            "policy_mask",
            "temperature",
            "material_label",
            "protocol_id",
            "dataset_id",
            "future_control",
            "Query_future_mask",
            "Query_lifetime",
            "EOL",
        ],
        "fixed_prediction_grid": copy.deepcopy(parent["fixed_prediction_grid"]),
        "support_observation_sampling": {
            "maximum_points_per_physical_cell": 512,
            "rule": "deterministic_evenly_spaced_supported_indices_including_endpoints",
            "weighting": "equal_physical_cell_then_equal_sampled_observation",
        },
        "selection_uses": "physical_UID_known_protocol_and_fixed_hash_only",
        "future_outcome_values_read_for_registration": 0,
        "final_target_Query_future_label_uses_during_registration": 0,
        "target_release_rule": "after_all_input_method_and_hyperparameter_choices_are_source_locked;real_D4_required_even_for_negative_source_results",
        "target_exposure_after_scoring": "register_as_exposed;no_later_claim_of_unseen_test",
        "historical_exposure": "source115_and_same_fold_assignment_have_prior_exposure;retrospective_development",
        "target_historical_exposure": parent["historical_exposure"],
        "provenance": {
            "parent_episodes_file": parent_path.relative_to(root).as_posix(),
            "parent_episodes_sha256": file_hash(parent_path),
            "parent_episode_contract_sha256": parent["episode_contract_sha256"],
            "source_folds_file": fold_path.relative_to(root).as_posix(),
            "source_folds_sha256": file_hash(fold_path),
            "source_fold_payload_sha256": folds["protocol_sha256"],
            "split_lock_file": lock_path.relative_to(root).as_posix(),
            "split_lock_sha256": file_hash(lock_path),
            "target_protocol_metadata": target_protocol_provenance,
        },
    }
    result["episode_contract_sha256"] = payload_hash(result)
    return validate_registration(result, folds, parent)


def _validate_task(task, protocols, draws):
    query, pool = task["query_ids"], task["support_pool_ids"]
    if (
        len(query) != len(set(query))
        or len(pool) != len(set(pool))
        or set(query) & set(pool)
    ):
        raise ValueError("A task repeats or crosses physical identities")
    if {row["k"] for row in task["shots"]} != set(SHOTS):
        raise ValueError("The registered shot family changed")
    by_k = {}
    for shot in task["shots"]:
        k = shot["k"]
        if shot["query_ids"] != query:
            raise ValueError("Query identities changed across shot counts")
        if shot["status"] != "FEASIBLE":
            if len(pool) >= k or shot["draws"]:
                raise ValueError("Support feasibility was reported incorrectly")
            continue
        expected = 1 if k == 0 else draws
        if len(shot["draws"]) != expected:
            raise ValueError("K0 or nonzero support draw multiplicity changed")
        by_k[k] = {}
        seen = {}
        for draw in shot["draws"]:
            ids = draw["support_ids"]
            if (
                len(ids) != k
                or len(ids) != len(set(ids))
                or not set(ids).issubset(pool)
                or draw["query_ids"] != query
            ):
                raise ValueError("A draw has invalid Support or Query identities")
            if k == 0 and draw["draw"] != 0:
                raise ValueError("K0 cannot create repeated independent draws")
            identity = payload_hash(sorted(ids))
            if draw["support_set_sha256"] != identity:
                raise ValueError("A support-set identity changed")
            duplicate = seen.get(identity)
            if draw["duplicate_of_draw"] != duplicate:
                raise ValueError("A duplicate Support draw was not identified")
            if duplicate is None:
                seen[identity] = draw["draw"]
            if draw["same_protocol_coverage"] != coverage(query, ids, protocols):
                raise ValueError("Query same-protocol Support coverage changed")
            if draw["draw"] in by_k[k]:
                raise ValueError("Draw indices must be unique")
            by_k[k][draw["draw"]] = set(ids)
    for draw in range(draws):
        previous = set()
        for k in SHOTS[1:]:
            if k in by_k:
                values = by_k[k][draw]
                if not previous.issubset(values):
                    raise ValueError("Support sets are not nested across K")
                previous = values


def validate_registration(value, folds=None, parent=None):
    payload = {
        key: item for key, item in value.items() if key != "episode_contract_sha256"
    }
    if payload_hash(payload) != value["episode_contract_sha256"]:
        raise ValueError("The one-sided episode payload fingerprint changed")
    if value["shots"] != list(SHOTS) or value["source_methods"] != list(SOURCE_METHODS):
        raise ValueError("Registered method or shot families changed")
    if (
        value["policy_model_dimension"] != 0
        or value["input_scope"] != "VIQ+earlySOH / no-policy"
    ):
        raise ValueError("The registered source input scope changed")
    if (
        value["future_outcome_values_read_for_registration"]
        or value["final_target_Query_future_label_uses_during_registration"]
    ):
        raise ValueError("Future values entered episode registration")
    protocols = value["source_protocols"]
    if (
        len(value["source_ids"]) != 115
        or len(set(value["source_ids"])) != 115
        or set(protocols) != set(value["source_ids"])
    ):
        raise ValueError("The physical source denominator changed")
    if len(value["sealed_final_MATR_ids"]) != 54 or set(
        value["sealed_final_MATR_ids"]
    ) & set(value["source_ids"]):
        raise ValueError("The sealed MATR target cohort changed or overlaps source")
    for role in ("inner", "outer"):
        if len(value["source_tasks"][role]) != 3:
            raise ValueError("Every source fold must have an adaptation task")
        for task in value["source_tasks"][role]:
            expected_query_count = (
                value["source_inner_query_count"]
                if role == "inner"
                else value["source_outer_query_count"]
            )
            if len(task["query_ids"]) != expected_query_count:
                raise ValueError("The fixed source Query count changed")
            _validate_task(task, protocols, value["support_draws"])
            if folds is not None:
                fold = folds["folds"][task["fold_id"]]
                key = "inner_validation_ids" if role == "inner" else "outer_test_ids"
                if set(task["query_ids"] + task["support_pool_ids"]) != set(fold[key]):
                    raise ValueError("A source task escaped its registered fold role")
                if set(task["query_ids"] + task["support_pool_ids"]) & set(
                    fold["fit_ids"]
                ):
                    raise ValueError("A source task overlaps its source fit cells")
    for task in value["target_tasks"]:
        active_protocols = task["protocol_by_id"]
        _validate_task(task, active_protocols, value["support_draws"])
        if set(task["query_ids"] + task["support_pool_ids"]) & set(value["source_ids"]):
            raise ValueError("A target task overlaps the source cohort")
        if parent is not None:
            original = next(
                row
                for row in parent["target_tasks"]
                if row["task_id"] == task["parent_task_id"]
            )
            for key in ("query_ids", "support_pool_ids"):
                if task[key] != original[key]:
                    raise ValueError("A final target identity was resampled")
            for shot in task["shots"]:
                old = next(row for row in original["shots"] if row["k"] == shot["k"])
                for draw in shot["draws"]:
                    original_draw = next(
                        row for row in old["draws"] if row["draw"] == draw["draw"]
                    )
                    if draw["support_ids"] != original_draw["support_ids"]:
                        raise ValueError("A final target Support draw changed")
    for row in value["fit_gradient_tasks"]:
        available = set(row["fit_ids"])
        if not available.issubset(value["source_ids"]) or len(available) != len(
            row["fit_ids"]
        ):
            raise ValueError("A fit gradient role repeats or escapes source cells")
        if folds is not None and available != set(
            folds["folds"][row["fold_id"]]["fit_ids"]
        ):
            raise ValueError("A fit gradient role changed its registered fold")
        seen = set()
        for pair in row["pairs"]:
            ids = pair["cell_ids"]
            cells = set(ids)
            if (
                len(cells) < 2
                or len(cells) != len(ids)
                or not cells.issubset(available)
                or cells & seen
            ):
                raise ValueError(
                    "A CROSS source-gradient task has invalid physical roles"
                )
            if any(protocols[uid] != pair["protocol_uid"] for uid in cells):
                raise ValueError("A CROSS pair crosses a registered known task")
            expected = {(left, right) for left in ids for right in ids if left != right}
            actual = {
                (item["support_ids"][0], item["query_ids"][0])
                for item in pair["ordered_pairs"]
                if len(item["support_ids"]) == len(item["query_ids"]) == 1
            }
            if (
                actual != expected
                or len(pair["ordered_pairs"]) != len(expected)
                or pair["ordered_pair_count"] != len(expected)
            ):
                raise ValueError(
                    "CROSS pairs must enumerate all ordered distinct physical cells"
                )
            seen.update(cells)
        if not set(row["singleton_ids"]).issubset(available) or seen & set(
            row["singleton_ids"]
        ):
            raise ValueError("Singleton gradient tasks overlap paired tasks")
        if seen | set(row["singleton_ids"]) != available:
            raise ValueError("Fit gradient tasks omit physical source cells")
    return value


def load_registration(path=DEFAULT_PATH):
    return validate_registration(json.loads(Path(path).read_text(encoding="utf-8")))


def deterministic_observation_positions(observed, maximum=512):
    observed = np.asarray(observed)
    if observed.ndim != 1 or observed.dtype != bool or maximum < 1:
        raise ValueError(
            "Support observation sampling requires a boolean vector and positive maximum"
        )
    indices = np.flatnonzero(observed)
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)
    return indices[positions]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    value = build_registration(args.root)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.output.exists():
        if load_registration(args.output) != value:
            raise RuntimeError("An existing few-shot episode lock differs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "status": "REGISTERED_NO_OUTCOME_VALUES",
                "source_outer_adaptation_states": value[
                    "source_outer_adaptation_states_all_seeds"
                ],
                "targets": [
                    {
                        "task_id": row["task_id"],
                        "query": len(row["query_ids"]),
                        "support_pool": len(row["support_pool_ids"]),
                    }
                    for row in value["target_tasks"]
                ],
                "episode_contract_sha256": value["episode_contract_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
