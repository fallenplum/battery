"""Specific regression checks for the preregistered E1--E3 statistics."""

import copy
import json
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
    scoped_norm_controls,
    summarize_contrast,
    summarize_equivalence,
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
