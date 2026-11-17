"""CPU regression checks for paired reference families and sealed target scoring."""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.drs_peft import run_specialization as runner
from scripts.drs_peft.data import ROOT, file_hash

CONFIG = ROOT / "configs/drs_peft/specialization_comparison_v1/CONFIG.json"


def setup(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registration = json.loads(
        (ROOT / config["episodes_file"]).read_text(encoding="utf-8")
    )
    args = SimpleNamespace(
        config=CONFIG,
        public_root=tmp_path / "public",
        private_root=tmp_path / "private",
        reference_root=tmp_path / "references",
    )
    return args, config, registration


def test_reference_family_and_zero_baseline_remain_explicit(tmp_path):
    _, config, _ = setup(tmp_path)
    for method in ("PRETRAINED_LORA", "WARMUP_ZERO"):
        assert runner.reference_family(method, config) == "WARMUP"
        assert runner.family_zero_method(method, config) == "WARMUP_ZERO"
    for method in (
        "SPECIALIZED_LORA",
        "SPECIALIZED_XS",
        "SPECIALIZED_OUT",
        "SPECIALIZED_ZERO",
    ):
        assert runner.reference_family(method, config) == "SPECIALIZED"
        assert runner.family_zero_method(method, config) == "SPECIALIZED_ZERO"
    with pytest.raises((KeyError, ValueError)):
        runner.reference_family("ambiguous_ZERO", config)


def test_target_matrix_preserves_two_zeros_and_four_distinct_adapters(tmp_path):
    args, config, registration = setup(tmp_path)
    specs = runner.target_units(args, config, registration)
    assert len(specs) == len({spec["folder"] for spec in specs}) == 930
    zeros = [spec for spec in specs if spec["K"] == 0]
    nonzero = [spec for spec in specs if spec["K"] > 0]
    assert len(zeros) == 30
    assert {spec["method"] for spec in zeros} == {"WARMUP_ZERO", "SPECIALIZED_ZERO"}
    assert all(spec["draw"] == 0 and not spec["support_ids"] for spec in zeros)
    assert len(nonzero) == 900
    unique = {
        (
            spec["task"]["task_id"],
            spec["seed"],
            spec["method"],
            spec["K"],
            tuple(sorted(spec["support_ids"])),
        )
        for spec in nonzero
    }
    assert len(unique) == 732
    assert sum(key[2] in config["source_inner"]["new_methods"] for key in unique) == 366
    assert {spec["task"]["domain"] for spec in specs} == {"MATR", "HUST"}


def inner_table(config):
    records = []
    winners = {1: 0, 3: 5, 5: 11}
    for method, fold, seed, k, trial, draw in itertools.product(
        config["source_inner"]["new_methods"],
        (0, 1, 2),
        (42, 123, 2026),
        (1, 3, 5),
        range(12),
        range(5),
    ):
        error = (
            abs(trial - winners[k]) * 0.01
            + fold * 0.001
            + (seed % 7) * 0.0001
            + draw * 0.00001
        )
        records.append(
            {
                "method": method,
                "fold_id": fold,
                "seed": seed,
                "K": k,
                "trial": trial,
                "draw": draw,
                "role": "inner",
                "mae": error,
            }
        )
    return pd.DataFrame(records)


def test_source_inner_aggregates_all_3240_slots_with_registered_order(tmp_path):
    _, config, _ = setup(tmp_path)
    grid = runner.trial_grid(config)
    assert grid == [
        {"learning_rate": lr, "steps": steps, "eta": eta}
        for lr, steps, eta in itertools.product(
            (1e-4, 1e-3), (10, 30), (0.25, 0.5, 1.0)
        )
    ]
    table = inner_table(config)
    assert len(table) == 3240
    selected = runner.aggregate_inner_trials(table, config)
    assert len(selected["selection_records"]) == 6
    for method in config["source_inner"]["new_methods"]:
        assert selected["selected_hyperparameters"][method] == {
            "1": grid[0],
            "3": grid[5],
            "5": grid[11],
        }
    table.loc[table.K.eq(1) & table.trial.eq(1), "mae"] = table.loc[
        table.K.eq(1) & table.trial.eq(0), "mae"
    ].to_numpy()
    tied = runner.aggregate_inner_trials(table, config)
    assert all(
        tied["selected_hyperparameters"][method]["1"] == grid[0]
        for method in config["source_inner"]["new_methods"]
    )


@pytest.mark.parametrize(
    "corruption",
    ["outer_score", "missing_slot", "duplicate_slot", "nonfinite", "other_method"],
)
def test_source_inner_rejects_leakage_and_incomplete_feedback(corruption, tmp_path):
    _, config, _ = setup(tmp_path)
    table = inner_table(config)
    if corruption == "outer_score":
        table.loc[0, "role"] = "outer"
    elif corruption == "missing_slot":
        table = table.drop(index=0)
    elif corruption == "duplicate_slot":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    elif corruption == "nonfinite":
        table.loc[0, "mae"] = np.nan
    else:
        table.loc[0, "method"] = "SPECIALIZED_LORA"
    with pytest.raises((ValueError, RuntimeError)):
        runner.aggregate_inner_trials(table, config)


def parent_fixture(tmp_path, monkeypatch, method="SPECIALIZED_LORA"):
    args, config, registration = setup(tmp_path)
    config["target"]["parent_reuse_root"] = "p"
    args.public_root = tmp_path / "u"
    args.reference_root = tmp_path / "r"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "source_header", lambda *args: {"study_id": "cpu_fixture"}
    )
    specs = runner.target_units(args, config, registration)
    spec = next(
        spec
        for spec in specs
        if spec["method"] == method
        and spec["seed"] == 42
        and spec["draw"] == 0
        and spec["K"] == (0 if method == "SPECIALIZED_ZERO" else 1)
    )
    reference = runner.reference_folder(args.reference_root, "final", 42)
    for stage in ("warmup", "static"):
        folder = reference / stage
        folder.mkdir(parents=True)
        torch.save(
            {
                "model": {
                    "bridge.policy_scale": torch.zeros(0),
                    "head.fixture": torch.tensor([1.0]),
                },
                "stage": stage,
            },
            folder / "CHECKPOINT_SELECTED.pt",
        )
    (reference / "geometry").mkdir()
    dictionary = reference / "geometry/DICTIONARIES.pt"
    dictionary.write_bytes(b"frozen OUT dictionary fixture")
    reference_receipt = {
        "warmup_checkpoint_sha256": file_hash(
            reference / "warmup/CHECKPOINT_SELECTED.pt"
        ),
        "reference_checkpoint_sha256": file_hash(
            reference / "static/CHECKPOINT_SELECTED.pt"
        ),
    }
    (reference / "REFERENCE_COMPLETE.json").write_text(
        json.dumps(reference_receipt), encoding="utf-8"
    )
    for name in ("run_onesided.py", "onesided_engine.py"):
        path = tmp_path / "scripts/drs_peft" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen parent fixture\n", encoding="utf-8")
    hyper = (
        {}
        if not spec["K"]
        else (
            {"lambda_multiplier": 0.1, "eta": 0.5, "epsilon": 1e-12}
            if method == "SPECIALIZED_OUT"
            else runner.trial_grid(config)[0]
        )
    )
    lock = {"selected_hyperparameters": {method: {"1": hyper}}}
    monkeypatch.setattr(runner, "read_target_lock", lambda *args: lock)
    identity = runner.unit_identity(args, config, registration, spec, hyper)
    old_method = {
        "SPECIALIZED_LORA": "LoRA",
        "SPECIALIZED_OUT": "OUT",
        "SPECIALIZED_ZERO": "ZERO",
    }[method]
    old_folder = (
        tmp_path
        / config["target"]["parent_reuse_root"]
        / "units/D4"
        / spec["task"]["task_id"]
        / "seed42"
        / old_method
        / f"K{spec['K']}_draw0"
    )
    old_folder.mkdir(parents=True)
    private = tmp_path / "private_parent"
    private.mkdir()
    prediction = private / "PREDICTIONS.npz"
    np.savez_compressed(
        prediction,
        ids=np.asarray(identity["query_ids"]),
        query=np.arange(101, 5001, dtype=np.float32),
        prediction=np.full((len(identity["query_ids"]), 4900), 0.8, dtype=np.float32),
    )
    state = private / "STATE.pt"
    torch.save(
        {
            "state": {
                "kind": "lowrank",
                "selected_state": {"head.fixture": torch.tensor([1.0])},
            }
        },
        state,
    )
    parent = {
        **identity,
        "stage": "D4",
        "method": old_method,
        "config_sha256": config["parent_config_sha256"],
        "fold_id": "final",
        "status": "SCORED",
        "execution_sha256": file_hash(tmp_path / "scripts/drs_peft/run_onesided.py"),
        "engine_sha256": file_hash(tmp_path / "scripts/drs_peft/onesided_engine.py"),
        "dictionary_sha256": file_hash(dictionary),
        "prediction_path": prediction.relative_to(tmp_path).as_posix(),
        "prediction_sha256": file_hash(prediction),
        "state_path": state.relative_to(tmp_path).as_posix(),
        "state_sha256": file_hash(state),
        "details": {"optimizer_updates": 30},
    }
    rows = []
    for uid in identity["query_ids"]:
        for window in ("all", "101_200", "201_500", "501_1000", "1001_plus"):
            rows.append(
                {
                    **{
                        key: parent[key]
                        for key in (
                            "stage",
                            "domain",
                            "task_id",
                            "fold_id",
                            "seed",
                            "method",
                            "K",
                            "draw",
                        )
                    },
                    "cell_uid": uid,
                    "window": window,
                    "n_obs": 1,
                    "mae": 0.12,
                    "rmse": 0.13,
                    "zero_mae": 0.20,
                    "gain_mae": 0.08,
                    "negative_transfer": False,
                }
            )
    pd.DataFrame(rows).to_csv(old_folder / "CELL_METRICS.csv", index=False)
    parent["cell_metrics_sha256"] = file_hash(old_folder / "CELL_METRICS.csv")
    (old_folder / "RECEIPT.json").write_text(json.dumps(parent), encoding="utf-8")
    return args, config, registration, spec, parent, old_folder


@pytest.mark.parametrize(
    "method", ["SPECIALIZED_LORA", "SPECIALIZED_OUT", "SPECIALIZED_ZERO"]
)
def test_parent_reuse_keeps_authentic_predictions_states_and_numeric_metrics(
    tmp_path, monkeypatch, method
):
    args, config, registration, spec, parent, old_folder = parent_fixture(
        tmp_path, monkeypatch, method
    )
    reads = []
    monkeypatch.setattr(
        runner, "load_dataset", lambda *args, **kwargs: reads.append(args)
    )
    receipt = runner.reuse_parent_unit(args, config, registration, spec)
    assert receipt["method"] == method
    assert receipt["reference_family"] == "SPECIALIZED"
    assert receipt["parent_reused"] is True
    assert receipt["prediction_path"] == parent["prediction_path"]
    assert receipt["prediction_sha256"] == parent["prediction_sha256"]
    assert receipt["state_path"] == parent["state_path"]
    assert receipt["state_sha256"] == parent["state_sha256"]
    assert not any(receipt["new_execution_counters"].values())
    assert receipt["target_Query_labels_used_for_adaptation_or_selection"] == 0
    original = pd.read_csv(old_folder / "CELL_METRICS.csv")
    reused = pd.read_csv(spec["folder"] / "CELL_METRICS.csv")
    assert reused.method.eq(method).all()
    assert reused.reference_family.eq("SPECIALIZED").all()
    for column in ("mae", "rmse", "zero_mae", "gain_mae", "n_obs"):
        np.testing.assert_array_equal(original[column], reused[column])
    assert not reads


@pytest.mark.parametrize(
    "corruption", ["prediction", "state", "metrics", "source", "support"]
)
def test_parent_reuse_rejects_changed_artifacts_before_any_query_label_read(
    tmp_path, monkeypatch, corruption
):
    args, config, registration, spec, parent, old_folder = parent_fixture(
        tmp_path, monkeypatch
    )
    if corruption in ("prediction", "state"):
        (tmp_path / parent[corruption + "_path"]).write_bytes(b"changed bytes")
    elif corruption == "metrics":
        (old_folder / "CELL_METRICS.csv").write_text("changed", encoding="utf-8")
    else:
        parent[
            "source_checkpoint_sha256" if corruption == "source" else "support_ids"
        ] = "another_source" if corruption == "source" else ["another_support"]
        (old_folder / "RECEIPT.json").write_text(json.dumps(parent), encoding="utf-8")
    reads = []
    monkeypatch.setattr(
        runner, "load_dataset", lambda *args, **kwargs: reads.append(args)
    )
    with pytest.raises(RuntimeError):
        runner.reuse_parent_unit(args, config, registration, spec)
    assert not reads


def test_all_930_target_predictions_are_required_before_query_label_loading(
    tmp_path, monkeypatch
):
    args, config, registration = setup(tmp_path)
    specs = runner.target_units(args, config, registration)
    monkeypatch.setattr(
        runner, "read_target_lock", lambda *args: {"selected_hyperparameters": {}}
    )
    monkeypatch.setattr(runner, "unit_identity", lambda *args: {})
    monkeypatch.setattr(runner, "load_prediction", lambda *args: np.zeros((1, 4900)))
    for spec in specs[:-1]:
        spec["folder"].mkdir(parents=True, exist_ok=True)
        (spec["folder"] / "RECEIPT.json").write_text(
            json.dumps({"status": "PREDICTIONS_LOCKED"}), encoding="utf-8"
        )
    lock = {
        "selected_hyperparameters": {
            method: {str(k): {} for k in (1, 3, 5)} for method in config["methods"]
        }
    }
    monkeypatch.setattr(runner, "read_target_lock", lambda *args: lock)
    reads = []
    monkeypatch.setattr(
        runner, "load_dataset", lambda *args, **kwargs: reads.append(args)
    )
    with pytest.raises(FileNotFoundError):
        runner.target_score(args, config, registration)
    assert not reads
    assert not (args.public_root / "TARGET_QUERY_EXPOSURE.json").exists()


@pytest.mark.parametrize(
    "method,zero_value,expected_gain",
    [("PRETRAINED_LORA", 0.3, 0.1), ("SPECIALIZED_OUT", 0.1, -0.1)],
)
def test_family_metrics_use_the_own_reference_zero_and_commit_complete_rows(
    tmp_path, method, zero_value, expected_gain
):
    _, config, _ = setup(tmp_path)
    data = SimpleNamespace(
        ids=["q"],
        protocol_ids=np.array(["protocol"]),
        query=np.arange(101, 5001, dtype=np.float32),
        targets=np.zeros((1, 4900), dtype=np.float32),
        observed=np.ones((1, 4900), dtype=bool),
        rows=lambda identifiers: np.array([0]),
    )
    receipt = {
        "query_ids": ["q"],
        "support_ids": [],
        "stage": "SPECIALIZATION",
        "task_id": "test_task",
        "domain": "MATR",
        "fold_id": "final",
        "seed": 42,
        "method": method,
        "K": 1,
        "draw": 0,
        "adaptation_id": "fixture",
        "is_duplicate": False,
        "reference_family": runner.reference_family(method, config),
        "status": "PREDICTIONS_LOCKED",
    }
    prediction = np.full((1, 4900), 0.2, dtype=np.float32)
    zero = np.full((1, 4900), zero_value, dtype=np.float32)
    runner.write_family_metrics(data, tmp_path, receipt, prediction, zero)
    table = pd.read_csv(tmp_path / "CELL_METRICS.csv")
    np.testing.assert_allclose(table.zero_mae, zero_value, atol=1e-7)
    np.testing.assert_allclose(table.gain_mae, expected_gain, atol=1e-7)
    assert table.reference_family.eq(receipt["reference_family"]).all()
    assert table.negative_transfer.eq(expected_gain < 0).all()
    saved = json.loads((tmp_path / "RECEIPT.json").read_text(encoding="utf-8"))
    assert saved["status"] == "SCORED"
    assert saved["cell_metrics_sha256"] == file_hash(tmp_path / "CELL_METRICS.csv")
    assert not (tmp_path / "CELL_METRICS.pending.csv").exists()
