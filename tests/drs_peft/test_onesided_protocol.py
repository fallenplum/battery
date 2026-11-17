"""Regression checks for physical-cell few-shot identity and budget locks."""

import copy
import json

import numpy as np
import pytest

from scripts.drs_peft.e123_protocol import load_protocol
from scripts.drs_peft.episodes import load_registration as load_parent
from scripts.drs_peft.onesided_protocol import (
    ROOT,
    SHOTS,
    build_registration,
    deterministic_observation_positions,
    fit_gradient_tasks,
    load_registration,
    nested_shots,
    payload_hash,
    validate_registration,
)


@pytest.fixture(scope="module")
def registration():
    return build_registration()


def rehash(value):
    value["episode_contract_sha256"] = payload_hash(
        {key: item for key, item in value.items() if key != "episode_contract_sha256"}
    )
    return value


def test_build_reads_no_npz_future_arrays(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Episode registration attempted to read a numeric archive")

    monkeypatch.setattr(np, "load", forbidden)
    actual = build_registration()
    assert actual["future_outcome_values_read_for_registration"] == 0
    assert actual["final_target_Query_future_label_uses_during_registration"] == 0


def test_source_query_counts_and_original_fold_partition(registration):
    folds = load_protocol()
    for role, sizes in (("outer", [19, 18, 18]), ("inner", [7, 7, 8])):
        for task, size in zip(registration["source_tasks"][role], sizes, strict=True):
            assert len(task["query_ids"]) == (20 if role == "outer" else 8)
            assert len(task["support_pool_ids"]) == size
            fold = folds["folds"][task["fold_id"]]
            parent_key = "outer_test_ids" if role == "outer" else "inner_validation_ids"
            assert set(task["query_ids"] + task["support_pool_ids"]) == set(
                fold[parent_key]
            )
            assert not set(task["query_ids"] + task["support_pool_ids"]) & set(
                fold["fit_ids"]
            )


def test_state_counts_and_no_policy_scope(registration):
    assert registration["source_outer_adaptation_states_per_seed"] == 315
    assert registration["source_outer_adaptation_states_all_seeds"] == 945
    assert registration["D3_spectrum_adaptation_states"] == 135
    assert registration["D3_training_seeds"] == [42]
    assert registration["D4_primary_nonzero_adaptation_states"] == 675
    assert registration["D4_primary_zero_reference_task_seed_states"] == 15
    assert registration["input_scope"] == "VIQ+earlySOH / no-policy"
    assert registration["policy_model_dimension"] == 0


def test_k0_singleton_and_nested_nonzero_draws(registration):
    for task in (
        registration["source_tasks"]["inner"]
        + registration["source_tasks"]["outer"]
        + registration["target_tasks"]
    ):
        by_k = {shot["k"]: shot for shot in task["shots"]}
        assert len(by_k[0]["draws"]) == 1
        assert by_k[0]["draws"][0]["support_ids"] == []
        for draw in range(5):
            sets = [set(by_k[k]["draws"][draw]["support_ids"]) for k in SHOTS[1:]]
            assert sets[0] <= sets[1] <= sets[2]
            assert all(
                by_k[k]["draws"][draw]["query_ids"] == task["query_ids"]
                for k in SHOTS[1:]
            )


def test_original_target_tasks_are_preserved(registration):
    parent = {task["task_id"]: task for task in load_parent()["target_tasks"]}
    main = [
        task
        for task in registration["target_tasks"]
        if task["selection_role"] == "primary_target"
    ]
    matr = [task for task in main if task["domain"] == "MATR"]
    hust = next(task for task in main if task["domain"] == "HUST")
    assert len(matr) == 4
    assert sum(len(task["query_ids"]) for task in matr) == 8
    assert sorted(len(task["support_pool_ids"]) for task in matr) == [5, 6, 6, 7]
    assert (len(hust["query_ids"]), len(hust["support_pool_ids"])) == (39, 38)
    for task in registration["target_tasks"]:
        original = parent[task["parent_task_id"]]
        assert task["query_ids"] == original["query_ids"]
        assert task["support_pool_ids"] == original["support_pool_ids"]
        for shot in task["shots"]:
            old = next(row for row in original["shots"] if row["k"] == shot["k"])
            for draw in shot["draws"]:
                assert draw["support_ids"] == old["draws"][draw["draw"]]["support_ids"]
    validate_registration(registration, load_protocol(), load_parent())


def test_stanford_coordinate_scope_is_supplementary(registration):
    task = next(
        task for task in registration["target_tasks"] if task["domain"] == "Stanford_2"
    )
    assert task["selection_role"] == "supplementary"
    assert task["coordinate_scope"] == "supplied_cycle_number_additional_protocol"
    assert (len(task["query_ids"]), len(task["support_pool_ids"])) == (91, 90)


def test_known_protocol_coverage_uses_real_hust_groups(registration):
    task = next(
        task for task in registration["target_tasks"] if task["domain"] == "HUST"
    )
    assert len(set(task["protocol_by_id"].values())) > 1
    draw = next(shot for shot in task["shots"] if shot["k"] == 1)["draws"][0]
    support_group = task["protocol_by_id"][draw["support_ids"][0]]
    expected = sum(
        task["protocol_by_id"][uid] == support_group for uid in task["query_ids"]
    )
    assert draw["same_protocol_coverage"]["covered_query_cells"] == expected


def test_d3_ordered_pairs_use_every_distinct_cell_once(registration):
    assert [row["pairable_tasks"] for row in registration["fit_gradient_tasks"]] == [
        14,
        12,
        14,
    ]
    assert [
        row["ordered_pair_count"] for row in registration["fit_gradient_tasks"]
    ] == [118, 122, 104]
    for fold in registration["fit_gradient_tasks"]:
        for task in fold["pairs"]:
            ids = task["cell_ids"]
            expected = {(left, right) for left in ids for right in ids if left != right}
            actual = {
                (pair["support_ids"][0], pair["query_ids"][0])
                for pair in task["ordered_pairs"]
            }
            assert expected == actual
            assert len(actual) == task["ordered_pair_count"]
        assert not set(fold["singleton_ids"]) & {
            uid for task in fold["pairs"] for uid in task["cell_ids"]
        }


def test_d3_singletons_are_explicitly_ineligible():
    tasks = fit_gradient_tasks(["a", "b", "c"], {"a": "p", "b": "p", "c": "q"}, 0, 42)
    assert tasks["singleton_ids"] == ["c"]
    assert tasks["pairable_tasks"] == 1
    assert tasks["ordered_pair_count"] == 2


def test_repeated_support_sets_are_marked():
    protocols = {"a": "p", "b": "p", "q": "p"}
    shots = nested_shots(["a", "b"], ["q"], protocols, seed=42, role="tiny", draws=5)
    one = next(row for row in shots if row["k"] == 1)
    assert one["unique_support_sets"] <= 2
    assert sum(row["duplicate_of_draw"] is not None for row in one["draws"]) >= 3
    assert all(row["status"] == "NOT_FEASIBLE" for row in shots if row["k"] in (3, 5))


def test_load_roundtrip_and_fingerprint_guard(registration, tmp_path):
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps(registration), encoding="utf-8")
    assert load_registration(path) == registration
    changed = copy.deepcopy(registration)
    changed["support_draws"] = 6
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_registration(path)


def test_future_value_flag_is_rejected_even_after_rehash(registration):
    changed = copy.deepcopy(registration)
    changed["future_outcome_values_read_for_registration"] = 1
    with pytest.raises(ValueError, match="Future values"):
        validate_registration(rehash(changed))


def test_support_query_identity_overlap_is_rejected(registration):
    changed = copy.deepcopy(registration)
    task = changed["source_tasks"]["outer"][0]
    task["support_pool_ids"][0] = task["query_ids"][0]
    with pytest.raises(ValueError, match="physical identities"):
        validate_registration(rehash(changed))


def test_final_target_resampling_is_rejected(registration):
    changed = copy.deepcopy(registration)
    task = next(task for task in changed["target_tasks"] if task["domain"] == "HUST")
    task["query_ids"] = list(reversed(task["query_ids"]))
    for shot in task["shots"]:
        shot["query_ids"] = task["query_ids"]
        for draw in shot["draws"]:
            draw["query_ids"] = task["query_ids"]
    with pytest.raises(ValueError, match="resampled"):
        validate_registration(rehash(changed), load_protocol(), load_parent())


@pytest.mark.parametrize("length", [0, 1, 100, 512, 513, 4900])
def test_observation_sampling_counts_and_endpoints(length):
    observed = np.ones(length, dtype=bool)
    positions = deterministic_observation_positions(observed)
    assert len(positions) == min(length, 512)
    assert len(np.unique(positions)) == len(positions)
    if length:
        assert positions[0] == 0 and positions[-1] == length - 1


def test_sampling_uses_only_supported_positions():
    observed = np.zeros(4900, dtype=bool)
    observed[::3] = True
    positions = deterministic_observation_positions(observed)
    assert len(positions) == 512
    assert observed[positions].all()
    assert positions[-1] == np.flatnonzero(observed)[-1]


def test_registration_is_deterministic(registration):
    assert build_registration(ROOT) == registration
