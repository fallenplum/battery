"""Fold-local tuning and target sealing regression checks."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.drs_peft import run_onesided as runner
from scripts.drs_peft.data import ROOT, file_hash
from scripts.drs_peft.train import read_json


def inner_trials():
    return pd.DataFrame(
        [
            {
                "role": "inner",
                "fold_id": 0,
                "seed": 42,
                "method": "OUT",
                "K": 1,
                "trial": trial,
                "draw": draw,
                "mae": score,
            }
            for trial, score in ((0, 0.1), (1, 0.2))
            for draw in range(5)
        ]
    )


def test_selection_uses_requested_fold_seed_method_shot_and_first_trial_tie():
    table = inner_trials()
    table.loc[table.trial.eq(1), "mae"] = 0.1
    unrelated = table.copy()
    unrelated["fold_id"], unrelated["mae"] = 2, -999.0
    combined = pd.concat([unrelated, table], ignore_index=True)
    trial, score = runner.select_own_fold(combined, 0, 42, "OUT", 1)
    assert trial == 0
    assert score == pytest.approx(0.1)
    with pytest.raises(ValueError, match="own-fold"):
        runner.select_own_fold(combined, 1, 42, "OUT", 1)


def test_selection_rejects_outer_query_scores():
    table = inner_trials()
    table.loc[0, "role"] = "outer"
    with pytest.raises(ValueError, match="inner Query"):
        runner.select_own_fold(table, 0, 42, "OUT", 1)


@pytest.mark.parametrize(
    "corruption", ["missing_draw", "nonfinite", "duplicate_draw", "wrong_draw_range"]
)
def test_selection_rejects_incomplete_or_corrupted_registered_trials(corruption):
    table = inner_trials()
    if corruption == "missing_draw":
        table = table.drop(table.index[-1])
    elif corruption == "nonfinite":
        table.loc[0, "mae"] = np.nan
    elif corruption == "duplicate_draw":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    else:
        table["draw"] += 1
    with pytest.raises(ValueError):
        runner.select_own_fold(table, 0, 42, "OUT", 1)


def target_setup(tmp_path):
    config = read_json(runner.CONFIG)
    registration = read_json(ROOT / config["episodes_file"])
    public = tmp_path / "public"
    public.mkdir()
    lock = {"best_generic": "LoRA"}
    (public / "TARGET_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    return SimpleNamespace(public_root=public), config, registration


def test_target_units_cover_all_seeds_tasks_methods_and_registered_draws(tmp_path):
    args, config, registration = target_setup(tmp_path)
    units = runner.expected_target_units(args, config, registration)
    assert len(units) == len(set(units)) == 690
    assert sum(path.name == "K0_draw0" for path in units) == 15
    assert sum(path.parent.name != "ZERO" for path in units) == 675
    assert all(
        any(f"seed{seed}" in path.parts for path in units) for seed in (42, 123, 2026)
    )
    assert {path.parent.name for path in units} == {"ZERO", "CAL2", "OUT", "LoRA"}


def test_target_units_reject_missing_registered_draw(tmp_path):
    args, config, registration = target_setup(tmp_path)
    registration = copy.deepcopy(registration)
    registration["target_tasks"][0]["shots"][1]["draws"].pop()
    with pytest.raises(RuntimeError, match="675 adaptations"):
        runner.expected_target_units(args, config, registration)


def test_target_units_reject_duplicate_draw_paths(tmp_path):
    args, config, registration = target_setup(tmp_path)
    registration = copy.deepcopy(registration)
    draws = registration["target_tasks"][0]["shots"][1]["draws"]
    draws[-1]["draw"] = draws[0]["draw"]
    with pytest.raises(RuntimeError):
        runner.expected_target_units(args, config, registration)


def test_inner_cache_rejects_another_source_before_support_outcome_access(
    tmp_path, monkeypatch
):
    args, config, registration = target_setup(tmp_path)
    args.config = runner.CONFIG
    args.private_root = tmp_path / "private"
    args.reference_root = tmp_path / "references"
    cache = args.private_root / "inner_trials/D2/fold0/seed42/OUT/K1/trial0_draw0.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps({"identity": {"reference_checkpoint_sha256": "old_source"}}),
        encoding="utf-8",
    )
    identity = {
        "reference_checkpoint_sha256": "current_source",
        "dictionary_sha256": "current_dictionary",
    }
    monkeypatch.setattr(runner, "prepare_engine", lambda *args: (None, None, identity))
    monkeypatch.setattr(runner, "register_methods", lambda *args: ["OUT"])
    accesses = []
    monkeypatch.setattr(
        runner, "make_support", lambda *args, **kwargs: accesses.append(args)
    )
    with pytest.raises(RuntimeError, match="inner trial cache"):
        runner.tune_inner(args, config, registration, None, 0, 42)
    assert not accesses


def scoring_setup(tmp_path):
    args, config, registration = target_setup(tmp_path)
    config["training_seeds"] = [42]
    registration["target_tasks"] = registration["target_tasks"][:1]
    args.reference_root = tmp_path / "references"
    args.config = tmp_path / "CONFIG.json"
    args.config.write_text(json.dumps(config), encoding="utf-8")
    lock = {
        "best_generic": "LoRA",
        "config_sha256": file_hash(args.config),
        "episodes_sha256": registration["episode_contract_sha256"],
        "selected_hyperparameters": {
            method: {
                str(k): runner.hyperparameters(config, method)[0] for k in (1, 3, 5)
            }
            for method in ("CAL2", "OUT", "LoRA")
        },
    }
    (args.public_root / "TARGET_LOCK.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    reference = runner.reference_folder(args.reference_root, "final", 42)
    (reference / "geometry").mkdir(parents=True)
    dictionary = reference / "geometry/DICTIONARIES.pt"
    dictionary.write_bytes(b"private dictionary fixture")
    identity = {"reference_checkpoint_sha256": "source_only_checkpoint"}
    (reference / "REFERENCE_COMPLETE.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    identity["dictionary_sha256"] = file_hash(dictionary)
    task = registration["target_tasks"][0]
    folder = runner.unit_folder(args, "D4", task, 42, "ZERO", 0, 0)
    folder.mkdir(parents=True)
    payload = tmp_path / "prediction.npz"
    np.savez_compressed(
        payload,
        ids=np.asarray(task["query_ids"]),
        query=np.arange(101, 5001, dtype=np.float32),
        prediction=np.ones((len(task["query_ids"]), 4900), dtype=np.float32),
    )
    state = tmp_path / "state.pt"
    state.write_bytes(b"private state fixture")
    receipt = {
        **runner.prediction_identity("D4", task, 42, "ZERO", 0, 0, [], {}, identity),
        "status": "PREDICTIONS_LOCKED",
        "prediction_path": str(payload),
        "prediction_sha256": file_hash(payload),
        "state_path": str(state),
        "state_sha256": file_hash(state),
    }
    (folder / "RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
    missing = runner.unit_folder(args, "D4", task, 42, "CAL2", 1, 0)
    return args, config, registration, folder, missing, receipt


def test_target_scoring_requires_every_prediction_before_loading_outcomes(
    tmp_path, monkeypatch
):
    args, config, registration, complete, missing, _ = scoring_setup(tmp_path)
    monkeypatch.setattr(
        runner, "expected_target_units", lambda *args: [complete, missing]
    )
    reads = []
    monkeypatch.setattr(
        runner, "load_dataset", lambda *args, **kwargs: reads.append(args)
    )
    with pytest.raises(FileNotFoundError):
        runner.target_score(args, config, registration)
    assert not reads
    assert not (args.public_root / "TARGET_QUERY_EXPOSURE.json").exists()


@pytest.mark.parametrize(
    "corruption", ["prediction_hash", "state_hash", "query_identity", "source_identity"]
)
def test_target_scoring_rejects_changed_prediction_before_loading_outcomes(
    tmp_path, monkeypatch, corruption
):
    args, config, registration, folder, _, receipt = scoring_setup(tmp_path)
    if corruption == "prediction_hash":
        receipt["prediction_sha256"] = "old_sha"
    elif corruption == "state_hash":
        receipt["state_sha256"] = "old_sha"
    elif corruption == "query_identity":
        receipt["query_ids"] = ["some_other_battery"]
    else:
        receipt["source_checkpoint_sha256"] = "another_source_model"
    (folder / "RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(runner, "expected_target_units", lambda *args: [folder])
    reads = []
    monkeypatch.setattr(
        runner, "load_dataset", lambda *args, **kwargs: reads.append(args)
    )
    with pytest.raises(RuntimeError):
        runner.target_score(args, config, registration)
    assert not reads
    assert not (args.public_root / "TARGET_QUERY_EXPOSURE.json").exists()
