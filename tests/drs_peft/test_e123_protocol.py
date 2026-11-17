"""Leakage and immutability checks for source-internal grouped CV."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.drs_peft import e123_protocol as protocol_module
from scripts.drs_peft.e123_protocol import (
    ROOT,
    _digest,
    _support,
    balanced_group_partition,
    build_protocol,
    fold_episode_view,
    load_protocol,
    validate_against_data,
    validate_protocol,
)


@pytest.fixture(scope="module")
def registration():
    return build_protocol()


@pytest.fixture(scope="module")
def parent():
    return json.loads(
        (ROOT / "configs/drs_peft/episodes_v1.json").read_text(encoding="utf-8")
    )


def reseal(value):
    value["protocol_sha256"] = _digest(
        {key: item for key, item in value.items() if key != "protocol_sha256"}
    )
    return value


def test_formal_source_counts_and_once_only_outer_coverage(registration, parent):
    validate_protocol(registration, parent)
    assert len(registration["source_ids"]) == 115
    assert len(registration["source_groups"]) == 62
    assert len(registration["sealed_target_ids"]) == 54
    counts = Counter(
        uid for fold in registration["folds"] for uid in fold["outer_test_ids"]
    )
    assert set(counts) == set(registration["source_ids"])
    assert set(counts.values()) == {1}
    assert [
        fold["outer_test_support"]["windows"]["1001_plus"]["physical_cells"]
        for fold in registration["folds"]
    ] == [7, 7, 7]


def test_balanced_partition_is_independent_of_mapping_insertion_order():
    groups = {
        f"p{index}": {"ids": [f"c{index}"], "tail_cells": int(index < 3)}
        for index in range(9)
    }
    expected = balanced_group_partition(groups, 3, 42, "test")
    reversed_mapping = dict(reversed(list(groups.items())))
    assert balanced_group_partition(reversed_mapping, 3, 42, "test") == expected
    assert sorted(name for part in expected for name in part) == sorted(groups)
    with pytest.raises(ValueError):
        balanced_group_partition(groups, 10, 42, "test")


def test_protocol_roundtrip_and_tamper_rejection(tmp_path, registration):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(registration), encoding="utf-8")
    assert load_protocol(path) == registration
    changed = copy.deepcopy(registration)
    changed["folds"][0]["fit_ids"].append("unauthorized")
    with pytest.raises(RuntimeError, match="identity changed"):
        validate_protocol(changed)


def test_sealed_target_and_duplicate_ids_fail_after_valid_hash(registration):
    changed = copy.deepcopy(registration)
    changed["folds"][0]["fit_ids"].append(changed["sealed_target_ids"][0])
    with pytest.raises(RuntimeError, match="sealed target"):
        validate_protocol(reseal(changed))
    changed = copy.deepcopy(registration)
    changed["folds"][0]["fit_ids"].append(changed["folds"][0]["fit_ids"][0])
    with pytest.raises(RuntimeError, match="repeated"):
        validate_protocol(reseal(changed))


def test_partial_protocol_movement_is_rejected(registration):
    changed = copy.deepcopy(registration)
    fold = changed["folds"][0]
    qualification = changed["source_qualification"]
    fit_groups = Counter(qualification[uid]["protocol_uid"] for uid in fold["fit_ids"])
    moved = next(
        uid
        for uid in fold["fit_ids"]
        if fit_groups[qualification[uid]["protocol_uid"]] >= 2
    )
    fold["fit_ids"].remove(moved)
    fold["outer_test_ids"].append(moved)
    for prefix in ("fit", "outer_test"):
        ids = fold[prefix + "_ids"]
        fold[prefix + "_groups"] = sorted(
            {qualification[uid]["protocol_uid"] for uid in ids}
        )
        fold[prefix + "_support"] = _support(ids, qualification)
    with pytest.raises(RuntimeError, match="protocol groups cross"):
        validate_protocol(reseal(changed))


def test_trainer_view_exposes_fit_and_inner_only_for_training(registration, parent):
    fold = registration["folds"][0]
    view = fold_episode_view(parent, fold)
    roles = view["source_cell_roles"]
    assert roles["source_train"] == fold["fit_ids"]
    assert roles["source_meta_validation"] == fold["inner_validation_ids"]
    assert roles["source_outer_validation"] == fold["outer_test_ids"]
    assert roles["target_test"] == parent["source_cell_roles"]["target_test"]
    assert view["episode_contract_sha256"] != parent["episode_contract_sha256"]
    changed = copy.deepcopy(fold)
    changed["fit_ids"].append(changed["fit_ids"][0])
    with pytest.raises(RuntimeError, match="repeats"):
        fold_episode_view(parent, changed)


def test_split_never_reads_soh_and_ignores_final_target_masks(
    monkeypatch, registration
):
    original_load = np.load
    source = set(registration["source_ids"])
    accessed = []

    class CheckedArchive:
        def __init__(self, path, **kwargs):
            self.archive = original_load(path, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            assert key in {"ids", "query", "observed", "protocol_ids"}
            accessed.append(key)
            result = self.archive[key]
            if key == "observed":
                result = result.copy()
                for row, uid in enumerate(self.archive["ids"].astype(str)):
                    if uid not in source:
                        result[row] = ~result[row]
            return result

    monkeypatch.setattr(protocol_module.np, "load", CheckedArchive)
    assert build_protocol() == registration
    assert "targets" not in accessed


def test_parent_pool_change_rejected(registration, parent):
    changed = copy.deepcopy(parent)
    changed["source_cell_roles"]["source_train"].pop()
    with pytest.raises(RuntimeError, match="authorized parent"):
        validate_protocol(registration, changed)


def test_live_cache_binding_catches_protocol_and_support_drift(registration, parent):
    with np.load(
        ROOT / "runtime/drs_peft/data_v1/MATR/arrays.npz", allow_pickle=False
    ) as archive:
        data = SimpleNamespace(
            domain="MATR",
            ids=archive["ids"].astype(str).tolist(),
            query=archive["query"],
            observed=archive["observed"],
            protocol_ids=archive["protocol_ids"].astype(str),
        )
    assert validate_against_data(registration, data, parent)["status"] == "PASSED"
    row = data.ids.index(registration["source_ids"][0])
    old_group = data.protocol_ids[row]
    data.protocol_ids[row] = "invalid"
    with pytest.raises(RuntimeError, match="protocol identities changed"):
        validate_against_data(registration, data, parent)
    data.protocol_ids[row] = old_group
    data.observed[row, 0] = ~data.observed[row, 0]
    with pytest.raises(RuntimeError, match="observation support changed"):
        validate_against_data(registration, data, parent)
