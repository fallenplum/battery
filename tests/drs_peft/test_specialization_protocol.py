"""Domain-total-K registration preserves fixed physical identities."""

import copy
import json

import numpy as np
import pytest

from scripts.drs_peft.onesided_protocol import ROOT, load_registration, payload_hash
from scripts.drs_peft.specialization_protocol import (
    DEFAULT_CONFIG,
    DEFAULT_TARGET_CONTRACT,
    build_total_k_episodes,
    load_target_contract,
    validate_total_k_episodes,
)


@pytest.fixture(scope="module")
def contract():
    return load_target_contract(DEFAULT_TARGET_CONTRACT)


def test_registration_reads_no_numeric_archives(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Target registration loaded a numeric archive")

    monkeypatch.setattr(np, "load", forbidden)
    config, override, episodes = load_target_contract(DEFAULT_TARGET_CONTRACT)
    assert episodes["future_outcome_values_read_for_registration"] == 0
    assert override["parent_config_sha256"].startswith("ad49fe5b")
    assert build_total_k_episodes(config) == episodes


def test_matr_ordered_query_and_support_unions(contract):
    config, _, episodes = contract
    parent = load_registration(ROOT / config["episodes_file"])
    by_id = {task["task_id"]: task for task in parent["target_tasks"]}
    groups = [
        by_id[name]
        for name in config["target"]["task_ids"]
        if by_id[name]["domain"] == "MATR"
    ]
    matr = episodes["target_tasks"][0]
    assert matr["task_id"] == "MATR_POOLED"
    assert matr["query_ids"] == [uid for task in groups for uid in task["query_ids"]]
    assert matr["support_pool_ids"] == [
        uid for task in groups for uid in task["support_pool_ids"]
    ]
    assert len(matr["query_ids"]) == 8
    assert len(matr["support_pool_ids"]) == 24
    assert not set(matr["query_ids"]) & set(matr["support_pool_ids"])


def test_hust_task_and_every_draw_exactly_preserved(contract):
    config, _, episodes = contract
    parent = load_registration(ROOT / config["episodes_file"])
    hust = next(task for task in parent["target_tasks"] if task["task_id"] == "C_HUST")
    assert episodes["target_tasks"][1] == hust


def test_nested_draws_use_domain_total_k_and_report_repeats(contract):
    _, override, episodes = contract
    matr = episodes["target_tasks"][0]
    shots = {shot["k"]: shot for shot in matr["shots"]}
    assert [shots[k]["unique_support_sets"] for k in (1, 3, 5)] == [3, 5, 5]
    assert [draw["duplicate_of_draw"] for draw in shots[1]["draws"]] == [
        None,
        None,
        None,
        0,
        0,
    ]
    for draw in range(5):
        supports = [set(shots[k]["draws"][draw]["support_ids"]) for k in (1, 3, 5)]
        assert [len(ids) for ids in supports] == [1, 3, 5]
        assert supports[0] <= supports[1] <= supports[2]
    assert episodes["unique_nonzero_support_sets_per_task"] == {
        "MATR_POOLED": 13,
        "C_HUST": 15,
    }
    assert override["target"]["expected_nonzero_slots"] == 360
    assert override["target"]["expected_family_zero_slots"] == 12
    assert override["target"]["expected_unique_nonzero_states"] == 336
    assert override["target"]["expected_new_unique_nonzero_states"] == 246


def test_protocol_labels_do_not_change_uid_sampling(contract, monkeypatch):
    config, _, episodes = contract
    parent = copy.deepcopy(load_registration(ROOT / config["episodes_file"]))
    for task in parent["target_tasks"]:
        if task["domain"] == "MATR":
            task["protocol_by_id"] = {
                uid: "different-scoring-only-group" for uid in task["protocol_by_id"]
            }
    monkeypatch.setattr(
        "scripts.drs_peft.specialization_protocol.load_registration",
        lambda path: parent,
    )
    changed = build_total_k_episodes(config)
    for old_shot, new_shot in zip(
        episodes["target_tasks"][0]["shots"],
        changed["target_tasks"][0]["shots"],
        strict=True,
    ):
        assert [draw["support_ids"] for draw in old_shot["draws"]] == [
            draw["support_ids"] for draw in new_shot["draws"]
        ]


def test_rehashed_identity_change_still_rejected(contract):
    config, _, episodes = contract
    changed = copy.deepcopy(episodes)
    changed["target_tasks"][0]["query_ids"].reverse()
    changed["episode_contract_sha256"] = payload_hash(
        {
            key: value
            for key, value in changed.items()
            if key != "episode_contract_sha256"
        }
    )
    with pytest.raises(ValueError, match="UID-only recipe"):
        validate_total_k_episodes(changed, config)


def test_frozen_source_config_retains_original_identity():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["source_inner"]["expected_scored_trial_slots"] == 3240
    assert config["source_inner"]["new_source_outer_experiments"] == 0
    assert config["target"]["expected_nonzero_slots"] == 900
