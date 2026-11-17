"""Specific regression checks for the preregistered E1--E3 statistics."""

import copy
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.drs_peft.e123_protocol import DEFAULT_PATH, load_protocol
from scripts.drs_peft.summarize_e123 import (
    ROOT,
    aggregate_metrics,
    candidate_gates,
    check_complete_outer,
    crossed_interval,
    equivalence_interval,
    expected_outer_units,
    paired_rows,
    relative_gain,
    render_report_only,
    report_detail_tables,
    scoped_norm_controls,
    summarize_contrast,
    summarize_equivalence,
    unit_relative_path,
)


@pytest.fixture(scope="module")
def config():
    return json.loads(
        (ROOT / "configs/drs_peft/e123_v1.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def observations(config):
    rows = []
    for method in config["methods"]:
        radii = [1.0, 0.25, 4.0] if method == "grad_core" else [1.0]
        for radius in radii:
            for evaluation in config["evaluation_modes"]:
                for seed in config["training_seeds"]:
                    for cell in range(6):
                        baseline = 0.1 + cell * 0.005
                        for window in ("all", "1001_plus"):
                            for mode, offset in (
                                ("constant", 0),
                                ("absolute", -0.01),
                                ("relative", 0.001),
                                ("dual", -0.008),
                            ):
                                mae = baseline + offset
                                rows.append(
                                    {
                                        "stage": "condition",
                                        "method": method,
                                        "radius": radius,
                                        "evaluation_mode": evaluation,
                                        "seed": seed,
                                        "fold_id": cell % 3,
                                        "cell_uid": f"cell{cell}",
                                        "protocol_uid": f"p{cell % 3}",
                                        "window": window,
                                        "mode": mode,
                                        "n_obs": 100,
                                        "mae": mae,
                                        "rmse": mae + 0.01,
                                        "bias": mae / 2,
                                    }
                                )
    return pd.DataFrame(rows)


def test_expected_outer_matrix_is_complete_and_unique(config):
    expected = expected_outer_units(config, load_protocol(DEFAULT_PATH))
    assert len(expected) == 387
    keys = [json.dumps([unit, mode], sort_keys=True) for unit, mode in expected]
    assert len(set(keys)) == len(expected)


def test_crossed_interval_uses_registered_two_candidate_confidence():
    actual = crossed_interval(np.ones((3, 6)), np.random.default_rng(42), 2000, 0.975)
    np.testing.assert_array_equal(actual, [1, 1])


def test_relative_gain_uses_ratio_of_means():
    assert relative_gain([1, 100], [0.5, 99]) == pytest.approx(1.5 / 101)
    with pytest.raises(RuntimeError, match="positive constant"):
        relative_gain([0, 0], [1, 1])


def test_paired_contrast_preserves_cell_fold_seed_units(observations, config):
    result = summarize_contrast(
        observations,
        "grad_core",
        1,
        "trained_ball",
        "absolute",
        "all",
        config["statistics"],
    )
    assert result["physical_cells"] == 6
    assert result["protocol_groups"] == 3
    assert len(result["per_fold_seed_relative_MAE_improvement"]) == 9
    np.testing.assert_allclose(result["cell_uid_interval"], [0.01, 0.01], atol=1e-16)
    np.testing.assert_allclose(
        result["protocol_uid_interval"], [0.01, 0.01], atol=1e-16
    )
    assert all(
        value > 0 for value in result["per_fold_relative_MAE_improvement"].values()
    )


def test_paired_missing_cells_and_support_drift_fail(observations):
    changed = observations.drop(observations.index[0])
    with pytest.raises(RuntimeError, match="Missing paired"):
        paired_rows(changed, "grad_core", 1, "trained_ball", "absolute", "all")
    changed = observations.copy()
    selected = changed.index[
        (changed.method == "grad_core")
        & changed.radius.eq(1)
        & changed.evaluation_mode.eq("trained_ball")
        & changed["mode"].eq("absolute")
        & changed.window.eq("all")
    ][0]
    changed.loc[selected, "n_obs"] += 1
    with pytest.raises(RuntimeError, match="different observed"):
        paired_rows(changed, "grad_core", 1, "trained_ball", "absolute", "all")


def test_equivalence_uses_paired_gain_ratios_and_is_scale_invariant(
    observations, config
):
    first = paired_rows(observations, "grad_core", 1, "trained_ball", "absolute", "all")
    second = first.copy()
    second[["mae_candidate", "mae_constant"]] *= 20
    for unit in ("cell_uid", "protocol_uid"):
        actual = equivalence_interval(first, second, unit, config["statistics"], "test")
        np.testing.assert_allclose(actual, [0, 0], atol=1e-15)


def test_extra_seed_and_protocol_group_summaries_avoid_duplicate_groupers(observations):
    for extra in (["seed"], ["fold_id", "seed"], ["protocol_uid"]):
        result = aggregate_metrics(observations, extra)
        assert not result.empty
        assert result.protocol_macro_mae.notna().all()
        assert result.unique_observed_points.le(result.observation_records).all()


def test_norm_failure_only_blocks_relevant_candidate_scope(config):
    records = []
    for unit, evaluation in expected_outer_units(config, load_protocol(DEFAULT_PATH)):
        if unit["stage"] == "condition":
            records.append(
                dict(
                    unit,
                    evaluation_mode=evaluation,
                    norm_control_passed=unit["mode"] != "relative",
                    matched_norm_unattainable=int(unit["mode"] == "relative"),
                    physical_cell_layers=12,
                    valid_norm_control_cell_layers=11
                    if unit["mode"] == "relative"
                    else 12,
                )
            )
    absolute = scoped_norm_controls(config, "absolute", records)
    assert all(value["valid"] for value in absolute.values())
    relative = scoped_norm_controls(config, "relative", records)
    assert not any(value["valid"] for value in relative.values())


def test_candidate_gates_require_same_candidate_all_three_experiments(
    observations, config
):
    contrasts = []
    for method in config["methods"]:
        for radius in [1, 0.25, 4] if method == "grad_core" else [1]:
            for evaluation in config["evaluation_modes"]:
                for window in ("all", "1001_plus"):
                    contrasts.append(
                        summarize_contrast(
                            observations,
                            method,
                            radius,
                            evaluation,
                            "absolute",
                            window,
                            config["statistics"],
                        )
                    )
    equivalence = [
        summarize_equivalence(
            observations, evaluation, "absolute", 1, config["statistics"], 0.02
        )
        for evaluation in config["evaluation_modes"]
    ]
    supported = candidate_gates(config, "absolute", contrasts, equivalence)
    assert (
        supported["E1_supported"]
        and supported["E2_supported"]
        and supported["E3_supported"]
    )
    assert supported["TARGET_ADVANCE_ALLOWED"]
    changed = copy.deepcopy(contrasts)
    selected = next(
        item
        for item in changed
        if item["method"] == "weight_svd_core" and item["window"] == "1001_plus"
    )
    selected["per_fold_relative_MAE_improvement"]["0"] = -0.1
    failed = candidate_gates(config, "absolute", changed, equivalence)
    assert not failed["E3_supported"]
    assert not failed["TARGET_ADVANCE_ALLOWED"]


def test_outer_guard_stops_before_partial_statistics(tmp_path, config):
    root = tmp_path / "public"
    root.mkdir()
    (root / "OUTER_EVALUATION_OPENED.json").write_text(
        json.dumps({"final_target_query_labels_used": 0, "source_units": []}),
        encoding="utf-8",
    )
    args = SimpleNamespace(public_root=root)
    with pytest.raises(RuntimeError, match="training closure"):
        check_complete_outer(args, config, load_protocol(DEFAULT_PATH))
    assert not (root / "E123_GATE.json").exists()


def source_binding_fixture(root):
    """Minimal complete closure for testing provenance guards without model data."""
    root.mkdir()
    public, private = root / "p", root / "r"
    public.mkdir()
    config = {
        "fold_payload_sha256": "engineering_fold",
        "training_seeds": [42],
        "methods": ["grad_core"],
        "modes": ["constant"],
        "primary_radius": 1.0,
        "grad_radius_sensitivity": [],
        "evaluation_modes": ["trained_ball", "matched_norm"],
        "epochs": {"warmup": 1, "static": 1, "condition": 1},
        "batch_size": 8,
        "expected_optimizer_updates": 3,
        "expected_training_units": {"total": 3, "static": 1, "condition": 1},
    }
    fold = {
        "fold_id": 0,
        "fit_ids": ["fit"],
        "inner_validation_ids": ["inner"],
        "outer_test_ids": ["outer"],
        "outer_test_groups": ["outer_group"],
    }
    protocol = {"protocol_sha256": "engineering_fold", "folds": [fold]}
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha = sha256(config_path.read_bytes()).hexdigest()
    expected = expected_outer_units(config, protocol)
    training = {json.dumps(unit, sort_keys=True): unit for unit, _ in expected}
    warmup = {"stage": "warmup", "fold_id": 0, "seed": 42}
    training[json.dumps(warmup, sort_keys=True)] = warmup
    closures = []
    for unit in training.values():
        folder = private / unit_relative_path(unit)
        folder.mkdir(parents=True)
        checkpoint = folder / "CHECKPOINT_SELECTED.pt"
        checkpoint.write_text("ENGINEERING_ONLY_" + unit["stage"], encoding="utf-8")
        run = {
            "training_ids": fold["fit_ids"],
            "validation_ids": fold["inner_validation_ids"],
            "config_sha256": config_sha,
        }
        (folder / "RUN_CONFIG.json").write_text(json.dumps(run), encoding="utf-8")
        marker = folder / "SOURCE_DEVELOPMENT_COMPLETE.json"
        marker.write_text(
            json.dumps(
                {
                    "status": "SOURCE_DEVELOPMENT_COMPLETE",
                    "epochs": 1,
                    "updates": 1,
                    "target_query_labels_used": 0,
                }
            ),
            encoding="utf-8",
        )
        closures.append(
            {
                "unit": unit,
                "source_receipt_sha256": sha256(marker.read_bytes()).hexdigest(),
                "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
    (public / "OUTER_EVALUATION_OPENED.json").write_text(
        json.dumps({"source_units": closures, "final_target_query_labels_used": 0}),
        encoding="utf-8",
    )
    for unit, evaluation in expected:
        folder = private / "outer" / unit_relative_path(unit) / evaluation
        folder.mkdir(parents=True)
        selected = next(entry for entry in closures if entry["unit"] == unit)
        receipt = {
            "unit": unit,
            "evaluation_mode": evaluation,
            "optimizer_updates": 0,
            "final_target_query_labels_used": 0,
            "registration_sha256": config_sha,
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "outer_ids": fold["outer_test_ids"],
            "outer_groups": fold["outer_test_groups"],
            "files": [],
        }
        for target in (
            folder,
            public / "outer" / unit_relative_path(unit) / evaluation,
        ):
            target.mkdir(parents=True, exist_ok=True)
            (target / "OUTER_COMPLETE.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
    args = SimpleNamespace(public_root=public, private_root=private, config=config_path)
    return args, config, protocol, expected


def test_source_config_sha_tamper(tmp_path):
    args, config, protocol, expected = source_binding_fixture(tmp_path.parent / "sha_a")
    assert len(check_complete_outer(args, config, protocol)) == 3
    unit = expected[0][0]
    run_path = args.private_root / unit_relative_path(unit) / "RUN_CONFIG.json"
    run = json.loads(run_path.read_text())
    run["config_sha256"] = "another_registration"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source registration config changed"):
        check_complete_outer(args, config, protocol)


def test_outer_checkpoint_sha_tamper(tmp_path):
    args, config, protocol, expected = source_binding_fixture(tmp_path.parent / "sha_b")
    assert len(check_complete_outer(args, config, protocol)) == 3
    unit, evaluation = expected[0]
    for target in (args.private_root, args.public_root):
        marker = (
            target
            / "outer"
            / unit_relative_path(unit)
            / evaluation
            / "OUTER_COMPLETE.json"
        )
        receipt = json.loads(marker.read_text())
        receipt["checkpoint_sha256"] = "another_checkpoint"
        marker.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen source selection"):
        check_complete_outer(args, config, protocol)


@pytest.fixture(scope="module")
def report_fixture(observations, config):
    contrasts = []
    for method in config["methods"]:
        for radius in [1, 0.25, 4] if method == "grad_core" else [1]:
            for evaluation in config["evaluation_modes"]:
                for candidate in config["primary_candidate_modes"]:
                    for window in ("all", "1001_plus"):
                        contrasts.append(
                            summarize_contrast(
                                observations,
                                method,
                                radius,
                                evaluation,
                                candidate,
                                window,
                                config["statistics"],
                            )
                        )
    protocol = {
        "protocol_sha256": "engineering_report",
        "source_ids": [f"cell{i}" for i in range(6)],
        "source_groups": [f"p{i}" for i in range(3)],
        "source_qualification": {
            f"cell{i}": {
                "protocol_uid": f"p{i % 3}",
                "window_observations": {"1001_plus": 100},
            }
            for i in range(6)
        },
        "folds": [],
    }
    for fold in range(3):
        protocol["folds"].append(
            {
                "fold_id": fold,
                "outer_test_ids": [f"cell{i}" for i in range(6) if i % 3 == fold],
                "outer_test_support": {
                    "windows": {
                        "all": {"physical_cells": 2, "protocol_groups": 1},
                        "1001_plus": {"physical_cells": 2, "protocol_groups": 1},
                    }
                },
            }
        )
    return contrasts, protocol


def test_report_tables_bind_registered_radius_and_fold_metrics(report_fixture, config):
    contrasts, protocol = copy.deepcopy(report_fixture)
    selected = next(
        item
        for item in contrasts
        if item["method"] == "weight_svd_core"
        and item["candidate"] == "dual"
        and item["evaluation_mode"] == "trained_ball"
        and item["window"] == "1001_plus"
    )
    selected["per_fold_relative_MAE_improvement"]["1"] = -0.031
    amplitudes, gains, folds = report_detail_tables(config, protocol, contrasts)
    assert len(amplitudes) == 12 and len(gains) == 12
    assert len(folds["trained_ball"]) == 12 and len(folds["matched_norm"]) == 12
    radius_record = next(
        item
        for item in contrasts
        if item["method"] == "grad_core"
        and item["radius"] == 4
        and item["candidate"] == "absolute"
        and item["evaluation_mode"] == "matched_norm"
        and item["window"] == "all"
    )
    row = amplitudes[
        amplitudes["候选"].eq("absolute")
        & amplitudes.R.eq(4)
        & amplitudes["幅度评价"].eq("matched_norm")
    ].iloc[0]
    assert row["完整未来改善%"] == pytest.approx(
        100 * radius_record["relative_MAE_improvement"]
    )
    negative = folds["trained_ball"]
    row = negative[
        negative["方向来源"].eq("weight_svd_core")
        & negative["候选"].eq("dual")
        & negative["折"].eq(1)
    ].iloc[0]
    assert row["1001+改善%"] == pytest.approx(-3.1)
    assert row["完整/远期电芯"] == "2/2"


def test_report_only_preserves_numerical_artifacts_and_is_idempotent(
    tmp_path, config, observations, report_fixture
):
    contrasts, protocol = report_fixture
    root = tmp_path / "render"
    root.mkdir()
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    static = observations[
        (observations["mode"] == "constant")
        & observations.radius.eq(1)
        & observations.evaluation_mode.eq("trained_ball")
    ].copy()
    static["stage"], static["mode"], static["radius"], static["evaluation_mode"] = (
        "static",
        "static",
        0.0,
        "static",
    )
    static["mae"] *= 0.99
    summary = aggregate_metrics(pd.concat([observations, static], ignore_index=True))
    summary.to_csv(root / "E123_MODEL_SUMMARY.csv", index=False)
    equivalence = [
        summarize_equivalence(
            observations, evaluation, candidate, 1, config["statistics"], 0.02
        )
        for evaluation in config["evaluation_modes"]
        for candidate in config["primary_candidate_modes"]
    ]
    gate = {
        "status": "ENGINEERING_ONLY",
        "TARGET_ADVANCE_ALLOWED": False,
        "candidates": [
            candidate_gates(config, candidate, contrasts, equivalence)
            for candidate in config["primary_candidate_modes"]
        ],
    }
    values = {
        "E123_CONTRASTS.json": contrasts,
        "E123_EQUIVALENCE.json": equivalence,
        "E123_GATE.json": gate,
        "E123_INDEPENDENT_METRIC_AUDIT.json": {
            "recomputed_cell_window_rows": len(observations),
            "maximum_metric_difference": 0,
        },
        "E123_UPDATE_NORM_AUDIT.json": {
            "all_norm_controls_passed": True,
            "matched_norm_unattainable_cell_layers": 0,
        },
    }
    paths = [root / "E123_MODEL_SUMMARY.csv"]
    for name, value in values.items():
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    report = root / "REPORT.md"
    report.write_text("engineering original report", encoding="utf-8")
    numeric_hashes = {
        str(path): sha256(path.read_bytes()).hexdigest() for path in paths
    }
    manifest_path = root / "E123_SUMMARY_MANIFEST.json"
    manifest = {
        "implementation_sha256": "a" * 64,
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "protocol_sha256": protocol["protocol_sha256"],
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in [*paths, report]
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    history = root / "history.json"
    history.write_text(
        json.dumps(
            [
                {
                    "candidate": "dual",
                    "reference": "static",
                    "window": "all",
                    "relative_mae_improvement": 0.08188735,
                }
            ]
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(public_root=root, config=config_path, report=report)
    render_report_only(args, config, protocol, history)
    first_report, first_manifest = report.read_bytes(), manifest_path.read_bytes()
    assert "8.19%" in report.read_text(encoding="utf-8")
    assert "### 主半径 matched_norm" in report.read_text(encoding="utf-8")
    assert all(
        sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in numeric_hashes.items()
    )
    after = json.loads(first_manifest)
    assert after["implementation_sha256"] == "a" * 64
    assert after["report_rendering"]["statistics_source_sha256"] == "a" * 64
    render_report_only(args, config, protocol, history)
    assert (
        report.read_bytes() == first_report
        and manifest_path.read_bytes() == first_manifest
    )
