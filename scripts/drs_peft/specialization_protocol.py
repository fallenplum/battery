"""Register domain-total-K target episodes using existing identity metadata."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.drs_peft.onesided_protocol import (
    ROOT,
    load_registration,
    nested_shots,
    payload_hash,
)
from scripts.drs_peft.summarize_onesided import digest

DEFAULT_CONFIG = ROOT / "configs/drs_peft/specialization_comparison_v1/CONFIG.json"
DEFAULT_EPISODES = DEFAULT_CONFIG.parent / "EPISODES_TOTAL_K.json"
DEFAULT_TARGET_CONTRACT = DEFAULT_CONFIG.parent / "TARGET_TOTAL_K.json"


def build_total_k_episodes(config, seed=42):
    parent = load_registration(ROOT / config["episodes_file"])
    if parent["episode_contract_sha256"] != config["episode_contract_sha256"]:
        raise ValueError("The original source and target episode identities changed")
    by_id = {task["task_id"]: task for task in parent["target_tasks"]}
    groups = [
        by_id[name]
        for name in config["target"]["task_ids"]
        if by_id[name]["domain"] == "MATR"
    ]
    query = [uid for task in groups for uid in task["query_ids"]]
    pool = [uid for task in groups for uid in task["support_pool_ids"]]
    if (
        len(query) != 8
        or len(set(query)) != 8
        or len(pool) != 24
        or len(set(pool)) != 24
    ):
        raise ValueError(
            "The pooled MATR task must preserve eight Query and 24 Support candidates"
        )
    protocols = {
        uid: group for task in groups for uid, group in task["protocol_by_id"].items()
    }
    matr = {
        "task_id": "MATR_POOLED",
        "domain": "MATR",
        "selection_role": "primary_target",
        "parent_task_ids": [task["task_id"] for task in groups],
        "protocol": "POOLED_ORIGINAL_FOUR_MATR_TARGET_PROTOCOLS",
        "query_ids": query,
        "support_pool_ids": pool,
        "protocol_by_id": protocols,
        "support_budget_scope": "K_total_physical_Support_cells_per_domain_shared_across_all_eight_Query",
        "identity_rule": "fixed_ordered_parent_Query_union_and_Support_pool_union",
        "coordinate_scope": "original_instrument_cycle",
        "state_scope": "one_adapted_state_for_all_Query_cells_per_method_seed_K_draw",
        "sampling_uses": "seed_and_physical_UID_only;protocol_labels_used_for_scoring_and_description",
        "shots": nested_shots(
            pool,
            query,
            protocols,
            seed=seed,
            role="specialization_total_K|MATR_POOLED",
            draws=5,
        ),
    }
    hust = copy.deepcopy(by_id["C_HUST"])
    tasks = [matr, hust]
    unique = {
        task["task_id"]: len(
            {
                tuple(sorted(draw["support_ids"]))
                for shot in task["shots"]
                if shot["k"] > 0
                for draw in shot["draws"]
            }
        )
        for task in tasks
    }
    result = {
        "study_id": "BATTERY_SPECIALIZATION_DOMAIN_TOTAL_K_20261118_v1",
        "parent_episode_contract_sha256": parent["episode_contract_sha256"],
        "training_seeds": config["training_seeds"],
        "episode_seed": seed,
        "shots": [0, 1, 3, 5],
        "support_draws": 5,
        "draw_hash_recipe": "sha256(f'{seed}|onesided|specialization_total_K|MATR_POOLED|draw{draw}|{uid}');ascending_digest_then_UID;first_K_then_sorted_UIDs",
        "target_tasks": tasks,
        "unique_nonzero_support_sets_per_task": unique,
        "expected_nonzero_slots": 360,
        "expected_family_zero_slots": 12,
        "expected_unique_nonzero_states": sum(unique.values())
        * len(config["methods"])
        * len(config["training_seeds"]),
        "expected_new_unique_nonzero_states": unique["MATR_POOLED"] * 4 * 3
        + unique["C_HUST"] * 2 * 3,
        "future_outcome_values_read_for_registration": 0,
        "protocol_groups_forced_into_Support": False,
        "HUST_parent_task_and_draws_preserved_exactly": True,
        "MATR_nonzero_parent_four_task_prediction_reuse": False,
        "MATR_ZERO_parent_prediction_union_allowed": True,
    }
    result["episode_contract_sha256"] = payload_hash(result)
    return result


def validate_total_k_episodes(value, config):
    actual_hash = payload_hash(
        {key: item for key, item in value.items() if key != "episode_contract_sha256"}
    )
    if actual_hash != value["episode_contract_sha256"]:
        raise ValueError("The total-K episode payload changed")
    expected = build_total_k_episodes(config, seed=value["episode_seed"])
    if value != expected:
        raise ValueError(
            "The total-K episode identities differ from their frozen UID-only recipe"
        )
    return value


def load_target_contract(path, config_path=DEFAULT_CONFIG):
    path, config_path = Path(path), Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    override = json.loads(path.read_text(encoding="utf-8"))
    if digest(config_path.read_bytes()) != override["parent_config_sha256"]:
        raise ValueError("The source-inner parent configuration changed")
    if (
        digest((ROOT / override["parent_master_file"]).read_bytes())
        != override["parent_master_sha256"]
    ):
        raise ValueError("The source-inner parent implementation changed")
    episode_path = ROOT / override["episodes_file"]
    if digest(episode_path.read_bytes()) != override["episodes_file_sha256"]:
        raise ValueError("The total-K episode file changed")
    episodes = validate_total_k_episodes(
        json.loads(episode_path.read_text(encoding="utf-8")), config
    )
    if episodes["episode_contract_sha256"] != override["episode_contract_sha256"]:
        raise ValueError("The target override points to a different episode payload")
    if override["target"]["task_ids"] != [
        task["task_id"] for task in episodes["target_tasks"]
    ]:
        raise ValueError("The target override changed its task identities")
    if (
        override["methods"] != config["methods"]
        or override["training_seeds"] != config["training_seeds"]
    ):
        raise ValueError("The target override changed methods or source seeds")
    for name in (
        "expected_nonzero_slots",
        "expected_family_zero_slots",
        "expected_unique_nonzero_states",
        "expected_new_unique_nonzero_states",
    ):
        if override["target"][name] != episodes[name]:
            raise ValueError("The target override changed a registered matrix count")
    return config, override, episodes
