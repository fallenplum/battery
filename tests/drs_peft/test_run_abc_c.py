"""Checks for C allocation, fit-only directions, and complete residual blocks."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.drs_peft import run_abc_c
from scripts.drs_peft.train import ROOT, read_json


def test_registered_C_plan_has_45_units_and_two_equal_budget_comparisons():
    registration = read_json(ROOT / "configs/drs_peft/abc_v1.json")
    protocol = read_json(ROOT / "configs/drs_peft/e123_folds_v1.json")
    plan = run_abc_c.build_plan(registration, protocol)
    assert plan["expected_training_units"] == 45
    assert plan["expected_optimizer_updates"] == 43200
    assert plan["expected_outer_evaluations"] == 90
    assert (
        len({run_abc_c.unit_path(Path("test"), unit) for unit in plan["units"]}) == 45
    )
    assert (
        plan["trainable_parameters"]["matched_fixed"]
        == plan["trainable_parameters"]["matched_residual"]
        == plan["trainable_parameters"]["matched_lora"]
        == 9204
    )


def test_C_rejects_padding_or_capacity_reallocation_changes():
    registration = read_json(ROOT / "configs/drs_peft/abc_v1.json")
    invalid = copy.deepcopy(registration)
    invalid["C"]["trainable_parameters"]["matched_fixed"] += 1
    with pytest.raises(ValueError, match="budgets"):
        run_abc_c.validate_registration(invalid)


def test_gradient_residual_includes_both_cross_blocks():
    u, v = np.eye(4)[:, :1], np.eye(4)[:, :1]
    gradient = np.zeros((4, 4))
    gradient[0, 0], gradient[0, 1], gradient[1, 0], gradient[1, 1] = 1, 2, 3, 4
    blocks = run_abc_c.decompose_gradient(gradient, u, v)
    np.testing.assert_array_equal(blocks["residual"], gradient - blocks["fixed"])
    assert np.square(blocks["left_only"]).sum() == 4
    assert np.square(blocks["right_only"]).sum() == 9
    assert np.square(blocks["double_perp"]).sum() == 16
    np.testing.assert_array_equal(
        blocks["left_only"] + blocks["right_only"] + blocks["double_perp"],
        blocks["residual"],
    )


def test_projection_complement_uses_covariance_without_original_frame_leakage():
    rng = np.random.default_rng(42)
    existing, _ = np.linalg.qr(rng.normal(size=(20, 4)))
    covariance = rng.normal(size=(20, 20))
    covariance = covariance @ covariance.T
    complement, eigenvalues = run_abc_c._projection_complement(covariance, existing, 6)
    np.testing.assert_allclose(existing.T @ complement, 0, atol=1e-12)
    np.testing.assert_allclose(complement.T @ complement, np.eye(6), atol=1e-12)
    assert np.all(eigenvalues > 0)


def test_layer_pair_indices_follow_QV_six_transformer_layers():
    assert run_abc_c.layer_pair_index("encoder.layers.0.self_attn.q_proj") == 0
    assert run_abc_c.layer_pair_index("encoder.layers.5.self_attn.v_proj") == 5
    with pytest.raises(ValueError):
        run_abc_c.layer_pair_index("encoder.layers.6.self_attn.v_proj")


def test_C_outer_requires_all_registered_training(tmp_path):
    args = SimpleNamespace(
        private_root=tmp_path,
        config=ROOT / "configs/drs_peft/abc_v1.json",
        folds=ROOT / "configs/drs_peft/e123_folds_v1.json",
    )
    plan = {
        "units": [{"fold_id": 0, "seed": 42, "arm": "fixed_rank4"}],
        "protocol_sha256": read_json(args.folds)["protocol_sha256"],
    }
    with pytest.raises(FileNotFoundError):
        run_abc_c.require_all_training(args, plan)


def test_active_C_lock_stays_outside_artifact_subtree(tmp_path):
    subtree = tmp_path / "fold0" / "seed42" / "fixed_rank4"
    lock = run_abc_c.external_lock(tmp_path, subtree)
    with lock:
        assert Path(lock.lock_file).parent == tmp_path / "_locks"
        assert not subtree.exists()


def test_C_source_guard_rejects_rehashed_cross_arm_checkpoint_with_same_budget(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        private_root=tmp_path / "private",
        parent_private_root=tmp_path / "parent",
        config=ROOT / "configs/drs_peft/abc_v1.json",
        folds=ROOT / "configs/drs_peft/e123_folds_v1.json",
    )
    registration = read_json(args.config)
    protocol = read_json(args.folds)
    fold = protocol["folds"][0]
    unit = {"fold_id": 0, "seed": 42, "arm": "matched_fixed"}
    folder = run_abc_c.unit_path(args.private_root, unit)
    folder.mkdir(parents=True)
    parent_checkpoint = tmp_path / "parent.pt"
    parent_checkpoint.write_bytes(b"authentic_source_start_fixture")
    monkeypatch.setattr(
        run_abc_c, "checked_training_checkpoint", lambda path: ({}, parent_checkpoint)
    )
    parent_hash = run_abc_c.file_hash(parent_checkpoint)
    sources = [
        {"path": name, "sha256": run_abc_c.file_hash(ROOT / name)}
        for name in (
            "scripts/drs_peft/run_abc_c.py",
            "models/drs_peft/abc_subspace.py",
            args.config.relative_to(ROOT).as_posix(),
        )
    ]
    run = {
        "stage": "ABC_C_FOLD_LOCAL_SOURCE",
        "method": "ABC_C_matched_fixed",
        "bridge": "native_multivariate",
        "seed": 42,
        "epochs": registration["C"]["epochs"],
        "batch_size": 8,
        "rank": 4,
        "training_ids": fold["fit_ids"],
        "validation_ids": fold["inner_validation_ids"],
        "expected_updates": 960,
        "target_query_labels_used": 0,
        "initial_checkpoint_sha256": parent_hash,
        "config_sha256": run_abc_c.file_hash(args.config),
        "trainable_parameters": 9204,
        "source_snapshot": sources,
    }
    spec = {
        "unit": unit,
        "config_sha256": run["config_sha256"],
        "folds_sha256": run_abc_c.file_hash(args.folds),
        "fold_payload_sha256": protocol["protocol_sha256"],
        "source_C0_checkpoint_sha256": parent_hash,
        "fit_ids": fold["fit_ids"],
        "fit_groups": fold["fit_groups"],
        "inner_validation_ids": fold["inner_validation_ids"],
        "inner_validation_groups": fold["inner_validation_groups"],
        "outer_ids": fold["outer_test_ids"],
        "outer_groups": fold["outer_test_groups"],
        "trainable_parameters": 9204,
        "outer_labels_used_for_training_or_selection": 0,
        "final_target_query_labels_used": 0,
    }
    names = (
        "RUN_CONFIG.json",
        "SOURCE_SNAPSHOT.json",
        "ABC_C_UNIT_SPEC.json",
        "FIXED_ADAPTER_START_CHECK.json",
        "SOURCE_SCALER.json",
        "TRAINABLE_PARAMETER_REPORT.json",
    )
    for name in names:
        run_abc_c.write_json(folder / name, {})
    run_abc_c.write_json(folder / "ABC_C_UNIT_SPEC.json", spec)

    def seal():
        run_abc_c.write_json(folder / "RUN_CONFIG.json", run)
        torch.save(
            {"identity": run, "best_epoch": 1, "validation": {"mae": 0.1}},
            folder / "CHECKPOINT_SELECTED.pt",
        )
        files = [
            {"path": path.name, "sha256": run_abc_c.file_hash(path)}
            for path in folder.iterdir()
            if path.name != "SOURCE_DEVELOPMENT_COMPLETE.json"
        ]
        receipt = {
            key: run[key]
            for key in (
                "stage",
                "method",
                "bridge",
                "seed",
                "epochs",
                "trainable_parameters",
            )
        }
        receipt.update(
            files=files,
            selected_checkpoint_sha256=run_abc_c.file_hash(
                folder / "CHECKPOINT_SELECTED.pt"
            ),
            best_epoch=1,
            validation={"mae": 0.1},
            updates=960,
            target_query_labels_used=0,
        )
        run_abc_c.write_json(folder / "SOURCE_DEVELOPMENT_COMPLETE.json", receipt)

    plan = {"units": [unit], "protocol_sha256": protocol["protocol_sha256"]}
    seal()
    assert len(run_abc_c.require_all_training(args, plan)) == 1
    run["method"] = "ABC_C_matched_residual"
    seal()
    with pytest.raises(RuntimeError, match="checkpoint method"):
        run_abc_c.require_all_training(args, plan)
