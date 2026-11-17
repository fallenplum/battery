"""Provenance, role-isolation and matrix checks for A and shuffled-feature PEFT."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.drs_peft import run_abc_a


def registration():
    return {
        "study_id": "fixture_abc",
        "a_modes": ["constant", "absolute", "dual"],
        "training_seeds": [42, 123, 2026],
        "evaluation_modes": ["trained_ball", "matched_norm"],
        "primary_radius": 1.0,
        "reference_weight_fraction": 0.01,
    }


def protocol():
    return {
        "folds": [{"fold_id": value} for value in range(3)],
        "protocol_sha256": "fixture",
    }


def test_registered_factorial_has_54_units_with_27_new_joint_trainings():
    plan = run_abc_a.build_plan(registration(), protocol())
    assert plan["total_factorial_units"] == 54
    assert plan["new_training_units"] == 27
    assert plan["reused_frozen_units"] == 27
    assert {unit["mode"] for unit in plan["units"]} == {"constant", "absolute", "dual"}
    assert (
        len(
            {
                (unit["regime"], str(run_abc_a.unit_path(Path("fixture"), unit)))
                for unit in plan["units"]
            }
        )
        == 54
    )
    assert plan == run_abc_a.build_plan(registration(), protocol())


def test_relative_and_shuffled_peft_add_27_new_and_9_reused_units():
    plan = run_abc_a.build_plan(registration(), protocol(), "B")
    assert plan["new_training_units"] == 27
    assert plan["total_factorial_units"] == 36
    assert plan["reused_frozen_units"] == 9
    assert {unit["mode"] for unit in plan["units"]} == {"relative", "shuffled_dual"}
    assert {unit["regime"] for unit in plan["units"]} == {"frozen", "joint"}


@pytest.mark.parametrize(
    "key,value",
    [("primary_radius", 4), ("reference_weight_fraction", 0.1), ("a_modes", ["dual"])],
)
def test_reuse_controls_cannot_change_without_a_new_contract(key, value):
    config = registration()
    config[key] = value
    with pytest.raises(ValueError):
        run_abc_a.build_plan(config, protocol())


def test_external_lock_is_outside_scanned_unit_tree(tmp_path):
    folder = tmp_path / "fold0/seed42/joint"
    folder.mkdir(parents=True)
    lock = run_abc_a.external_lock(tmp_path, folder)
    assert Path(lock.lock_file).parent == tmp_path / "_locks"
    with lock:
        assert not list(folder.iterdir())


def test_outer_stays_sealed_until_all_new_training_units_complete(tmp_path):
    args = SimpleNamespace(private_root=tmp_path, study="A")
    with pytest.raises(RuntimeError, match="stays sealed"):
        run_abc_a.require_training_complete(
            args, run_abc_a.build_plan(registration(), protocol())
        )


def test_shuffle_preserves_role_rows_and_paired_absolute_relative_features(monkeypatch):
    from scripts.drs_peft import run_abc_b

    monkeypatch.setattr(
        run_abc_b,
        "role_derangement",
        lambda ids, seed, fold_id, role: np.roll(np.arange(len(ids)), -1),
    )
    ids = ["f0", "f1", "f2", "i0", "i1", "o0", "o1", "target"]
    data = SimpleNamespace(
        ids=ids, rows=lambda names: np.asarray([ids.index(uid) for uid in names])
    )
    absolute = np.arange(96).reshape(8, 12).astype(np.float32)
    inputs = {
        "diagnostic_absolute": absolute,
        "diagnostic_relative": absolute + 1000,
        "x": np.zeros((8, 2)),
    }
    fold = {
        "fold_id": 0,
        "fit_ids": ids[:3],
        "inner_validation_ids": ids[3:5],
        "outer_test_ids": ids[5:7],
    }
    result, manifest = run_abc_a.shuffled_condition_inputs(data, inputs, fold, 42)
    np.testing.assert_array_equal(
        result["diagnostic_relative"] - result["diagnostic_absolute"],
        np.full((8, 12), 1000),
    )
    np.testing.assert_array_equal(result["diagnostic_absolute"][7], absolute[7])
    np.testing.assert_array_equal(inputs["diagnostic_absolute"], absolute)
    assert all(
        row["cell_uid"] != row["feature_donor_uid"] for row in manifest["mapping"]
    )
    assert {row["role"] for row in manifest["mapping"]} == {
        "fit",
        "inner_validation",
        "outer_test",
    }
    assert manifest["label_values_used"] == 0
    assert manifest["views_shuffled_as_paired_rows"]


def test_frozen_outer_reuse_rejects_checkpoint_changes(tmp_path):
    checkpoint = tmp_path / "CHECKPOINT_SELECTED.pt"
    checkpoint.write_bytes(b"initial")
    run_abc_a.bind(
        tmp_path / "OUTER_COMPLETE.json",
        {
            "checkpoint_sha256": "another_checkpoint",
            "outer_ids": ["outer"],
            "outer_groups": ["group"],
            "evaluation_mode": "trained_ball",
            "final_target_query_labels_used": 0,
            "optimizer_updates": 0,
        },
    )
    fold = {"outer_test_ids": ["outer"], "outer_test_groups": ["group"]}
    with pytest.raises(RuntimeError, match="registered source"):
        run_abc_a.checked_outer_reference(tmp_path, checkpoint, fold, "trained_ball")


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", 123),
        ("epochs", 2),
        ("batch_size", 16),
        ("learning_rate", 0.01),
        ("adapter_learning_rate", 0.01),
        ("weight_decay", 0),
        ("gradient_clip", 2),
        ("trainable_parameters", 1450),
        ("target_query_labels_used", 1),
    ],
)
def test_training_controls_reject_changed_reuse_budgets(key, value):
    config = {
        "epochs": {"condition": 120},
        "bridge_head_lr": 0.0003,
        "adapter_lr": 0.001,
    }
    identity = {
        "seed": 42,
        "epochs": 120,
        "batch_size": 8,
        "rank": 4,
        "learning_rate": 0.0003,
        "adapter_learning_rate": 0.001,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "trainable_parameters": 1440,
        "target_query_labels_used": 0,
    }
    run_abc_a.validate_training_controls(identity, config, {"seed": 42}, 1440)
    identity[key] = value
    with pytest.raises(RuntimeError, match="controls differ"):
        run_abc_a.validate_training_controls(identity, config, {"seed": 42}, 1440)


def test_static_start_guard_checks_c0_and_bridge_head_tensors():
    import hashlib

    state = {
        f"bridge.backbone.module.layer{index}.core": torch.ones(4, 4) * index
        for index in range(12)
    }
    state["bridge.fusion.weight"] = torch.ones(2, 3)
    state["head.net.weight"] = torch.ones(2, 3)
    start = {
        "initial_state_sha256": "a" * 64,
        "bridge_head_state_sha256": run_abc_a.bridge_head_state_hash(state),
        "source_core_sha256": {
            f"layer{index}": hashlib.sha256(
                state[f"bridge.backbone.module.layer{index}.core"].numpy().tobytes()
            ).hexdigest()
            for index in range(12)
        },
    }
    run_abc_a.validate_static_start(start, state)
    state["head.net.weight"][0, 0] += 1
    with pytest.raises(RuntimeError, match="initial state differs"):
        run_abc_a.validate_static_start(start, state)


def test_outer_guard_rejects_another_mode_receipt_with_the_same_config(
    tmp_path, monkeypatch
):
    unit = {
        "stage": "condition",
        "fold_id": 0,
        "seed": 42,
        "method": "grad_core",
        "mode": "constant",
        "radius": 1,
        "regime": "joint",
    }
    folder = run_abc_a.unit_path(tmp_path / "joint", unit)
    folder.mkdir(parents=True)
    config = tmp_path / "config.json"
    run_abc_a.bind(config, registration())
    run_abc_a.bind(folder / "SOURCE_DEVELOPMENT_COMPLETE.json", {})
    run_abc_a.bind(folder / "ABC_A_UNIT_SPEC.json", {"unit": dict(unit, mode="dual")})
    checkpoint = tmp_path / "dummy.pt"
    checkpoint.write_bytes(b"dummy")
    identity = {"config_sha256": run_abc_a.file_hash(config)}
    monkeypatch.setattr(
        run_abc_a,
        "checked_training_checkpoint",
        lambda path: ({"identity": identity, "model": {}}, checkpoint),
    )
    fold = {
        "fold_id": 0,
        "fit_ids": ["fit"],
        "inner_validation_ids": ["inner"],
        "outer_test_ids": ["outer"],
    }
    monkeypatch.setattr(run_abc_a, "load_protocol", lambda path: {"folds": [fold]})
    args = SimpleNamespace(
        private_root=tmp_path,
        study="A",
        config=config,
        folds=tmp_path / "folds.json",
        e123_private_root=tmp_path / "old",
    )
    with pytest.raises(RuntimeError, match="changed registration"):
        run_abc_a.require_training_complete(args, {"units": [unit]})
