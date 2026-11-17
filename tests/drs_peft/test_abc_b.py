"""Specific leakage, rate, shuffling, and complexity checks for experiment B."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.drs_peft.run_abc_b import (
    ARMS,
    DEFAULT_SETTINGS,
    baseline_prefix_features,
    complexity,
    degradation_rates,
    make_designs,
    make_estimator,
    metrics,
    rate_audit,
    rate_support,
    read_settings,
    role_derangement,
    score_models,
    summarize,
    train_models,
    unit_plan,
)


def prefix_fixture(count=6):
    cycle = np.tile(np.arange(1, 101), (count, 1))
    mask = np.ones((count, 100, 64), dtype=bool)
    x = np.zeros((count, 100, 64, 4), dtype=float)
    x[..., 0] = np.linspace(3.4, 4.1, 64)
    for row in range(count):
        x[row, ..., 1] = -1 - row * 0.1
        x[row, ..., 2] = (1 + row * 0.02 - cycle[row] * 0.0001)[:, None] * np.linspace(
            0, 1, 64
        )[None]
    return x, mask, cycle


def test_derangement_is_role_local_unique_and_uid_order_invariant():
    ids = ["a", "b", "c", "d", "e"]
    donor = role_derangement(ids, 42, 1, "fit")
    assert sorted(donor.tolist()) == list(range(5))
    assert np.all(donor != np.arange(5))
    mapping = {uid: ids[index] for uid, index in zip(ids, donor)}
    reverse = list(reversed(ids))
    other = role_derangement(reverse, 42, 1, "fit")
    assert mapping == {uid: reverse[index] for uid, index in zip(reverse, other)}
    np.testing.assert_array_equal(donor, role_derangement(ids, 42, 1, "fit"))


@pytest.mark.parametrize("ids", [[], ["a"], ["a", "a"]])
def test_derangement_rejects_degenerate_recipient_sets(ids):
    with pytest.raises(ValueError, match="unique physical"):
        role_derangement(ids, 42, 0, "fit")


def test_rates_use_observed_endpoint_bands_and_physical_duration():
    query = np.arange(101, 1201)
    values = np.stack((1 - query * 0.0002, 1 + query * 0.0001))
    observed = np.ones_like(values, dtype=bool)
    rate, included = degradation_rates(
        values, observed, query, 200, 500, DEFAULT_SETTINGS
    )
    assert included.tolist() == [True, True]
    np.testing.assert_allclose(rate, [0.0002, -0.0001], atol=1e-16)
    observed[0, (query >= 200) & (query < 208)] = False
    values[~observed] = np.nan
    rate, included = degradation_rates(
        values, observed, query, 200, 500, DEFAULT_SETTINGS
    )
    assert included.all()
    np.testing.assert_allclose(rate, [0.0002, -0.0001], atol=1e-16)


def test_rate_support_requires_coverage_and_both_endpoint_bands():
    query = np.arange(101, 1201)
    observed = np.ones((3, len(query)), dtype=bool)
    observed[0, (query > 490) & (query <= 500)] = False
    observed[1, (query >= 250) & (query <= 350)] = False
    included, _, _ = rate_support(observed, query, 200, 500, DEFAULT_SETTINGS)
    assert included.tolist() == [True, False, True]
    observed[0, query == 490] = False
    included, _, _ = rate_support(observed, query, 200, 500, DEFAULT_SETTINGS)
    assert included.tolist() == [False, False, True]


def test_rate_ignores_all_values_outside_qualified_endpoint_bands():
    query = np.arange(101, 1201)
    values = 1 - query[None] * 0.0002
    observed = np.ones_like(values, dtype=bool)
    expected, _ = degradation_rates(
        values, observed, query, 500, 1000, DEFAULT_SETTINGS
    )
    changed = values.copy()
    changed[:, (query < 500) | (query > 1000) | ((query >= 520) & (query <= 980))] = (
        10000
    )
    actual, _ = degradation_rates(changed, observed, query, 500, 1000, DEFAULT_SETTINGS)
    np.testing.assert_array_equal(expected, actual)


def test_prefix_capacity_baseline_uses_masks_and_rejects_future_coordinates():
    x, mask, cycles = prefix_fixture(2)
    expected = baseline_prefix_features(x, mask, cycles, [1.0, 1.1])
    np.testing.assert_allclose(expected[:, 3], -0.01, atol=1e-12)
    mask[:, :, 0] = False
    x[:, :, 0, 2] = 100000
    actual = baseline_prefix_features(x, mask, cycles, [1.0, 1.1])
    np.testing.assert_array_equal(actual, expected)
    cycles[0, -1] = 101
    with pytest.raises(ValueError, match="only cycles"):
        baseline_prefix_features(x, mask, cycles, [1.0, 1.1])


def design_fixture():
    x, mask, cycles = prefix_fixture()
    ids = [f"c{index}" for index in range(6)]
    data = SimpleNamespace(
        x=x,
        point_mask=mask,
        cycle_ids=cycles,
        anchor=np.arange(6) * 0.01 + 1,
        rows=lambda values: np.asarray([ids.index(uid) for uid in values]),
    )
    roles = {
        "fit": {"ids": ids[:2]},
        "inner_validation": {"ids": ids[2:4]},
        "outer_test": {"ids": ids[4:]},
    }
    return data, roles


def test_designs_pair_shuffled_views_and_keep_common_capacity_baseline():
    data, roles = design_fixture()
    designs, manifest = make_designs(data, {"fold_id": 0}, roles, 42)
    for role, group in designs.items():
        for arm in ARMS:
            assert group[arm].shape == (2, 30)
            np.testing.assert_array_equal(group[arm][:, :6], group["baseline"][:, :6])
        np.testing.assert_array_equal(
            group["shuffled_dual"][:, 6:], group["dual"][::-1, 6:]
        )
        assert not np.any(group["baseline"][:, 6:])
        assert not np.any(group["absolute"][:, 18:])
        assert not np.any(group["relative"][:, 6:18])
        assert manifest["shuffle"][role]["donor_ids"] == list(
            reversed(roles[role]["ids"])
        )


def test_design_scalers_are_fit_local_under_arbitrary_outer_prefix_changes():
    data, roles = design_fixture()
    original, original_manifest = make_designs(data, {"fold_id": 0}, roles, 42)
    data.x[4:, ..., 1:3] *= 10
    data.anchor[4:] *= 10
    changed, changed_manifest = make_designs(data, {"fold_id": 0}, roles, 42)
    assert original_manifest["view_scaler"] == changed_manifest["view_scaler"]
    assert (
        original_manifest["baseline_scaler_mean"]
        == changed_manifest["baseline_scaler_mean"]
    )
    assert (
        original_manifest["baseline_scaler_scale"]
        == changed_manifest["baseline_scaler_scale"]
    )
    for role in ("fit", "inner_validation"):
        for arm in ARMS:
            np.testing.assert_array_equal(original[role][arm], changed[role][arm])


def test_support_audit_never_accesses_future_target_values():
    class MaskOnlyData:
        query = np.arange(101, 1201)
        observed = np.ones((6, len(query)), dtype=bool)

        def rows(self, ids):
            return np.asarray([int(uid[1:]) for uid in ids])

        @property
        def targets(self):
            raise AssertionError("Mask qualification accessed future SOH")

    ids = [f"c{index}" for index in range(6)]
    protocol = {
        "source_ids": ids,
        "source_qualification": {uid: {"protocol_uid": uid} for uid in ids},
        "folds": [
            {
                "fold_id": 0,
                "fit_ids": ids[:2],
                "inner_validation_ids": ids[2:4],
                "outer_test_ids": ids[4:],
            }
        ],
    }
    audit = rate_audit(MaskOnlyData(), protocol, DEFAULT_SETTINGS)
    assert audit["source_future_SOH_values_read"] == 0
    assert audit["windows"]["500_1000"]["source_physical_cells"] == 6


@pytest.mark.parametrize("model_name", ["ridge", "gbr"])
def test_estimator_budget_is_fixed_and_complexity_is_measured(model_name):
    generator = np.random.default_rng(5)
    x = generator.normal(size=(20, 30))
    y = x[:, 0] * 0.001
    model = make_estimator(model_name, 42, DEFAULT_SETTINGS)
    model.fit(x, y)
    report = complexity(model, x)
    assert report["input_columns"] == 30
    assert report["design_rank"] <= 19
    if model_name == "ridge":
        assert 1 <= report["ridge_effective_df_including_intercept"] <= 20
        assert report["nominal_linear_parameters_including_intercept"] == 31
    else:
        assert report["fitted_trees"] == 100
        assert report["realized_total_leaves"] <= report["maximum_total_leaves"] == 400


def test_plan_contains180_cpu_fits_and_no_outer_hyperparameter_search():
    protocol = {"folds": [{"fold_id": fold} for fold in range(3)]}
    plan = unit_plan(protocol, DEFAULT_SETTINGS)
    assert len(plan) == 180
    assert len({json.dumps(unit, sort_keys=True) for unit in plan}) == 180


def test_registration_fails_closed_for_unimplemented_design_changes(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"B": DEFAULT_SETTINGS}), encoding="utf-8")
    assert read_settings(config) == DEFAULT_SETTINGS
    changed = copy.deepcopy(DEFAULT_SETTINGS)
    changed["gbr"]["max_depth"] = 5
    config.write_text(json.dumps({"B": changed}), encoding="utf-8")
    with pytest.raises(ValueError, match="gbr"):
        read_settings(config)


def test_metric_identity_matches_manual_paired_errors():
    actual = metrics([1, 2, 3], [2, 2, 1])
    assert actual["cells"] == 3
    assert actual["MAE"] == 1
    assert actual["bias"] == pytest.approx(-1 / 3)
    with pytest.raises(ValueError):
        metrics([], [])


def test_full180_fit_score_summary_surface_on_synthetic_arrays(tmp_path):
    data, _roles = design_fixture()
    data.query = np.arange(101, 1201)
    data.targets = np.stack(
        [1 - data.query * (0.0001 + index * 0.00001) for index in range(6)]
    )
    data.observed = np.ones_like(data.targets, dtype=bool)
    ids = [f"c{index}" for index in range(6)]
    protocol = {
        "source_ids": ids,
        "source_qualification": {uid: {"protocol_uid": uid} for uid in ids},
        "folds": [
            {
                "fold_id": fold,
                **{
                    role + "_ids": [
                        ids[(index + 2 * fold) % 6] for index in range(start, start + 2)
                    ]
                    for role, start in (
                        ("fit", 0),
                        ("inner_validation", 2),
                        ("outer_test", 4),
                    )
                },
            }
            for fold in range(3)
        ],
    }
    args = SimpleNamespace(
        private_root=tmp_path / "private",
        public_root=tmp_path / "public",
        fold_ids=[0, 1, 2],
    )
    args.public_root.mkdir()
    audit = rate_audit(data, protocol, DEFAULT_SETTINGS)
    plan = unit_plan(protocol, DEFAULT_SETTINGS)
    identity = {"source": "synthetic_integration_fixture"}
    with pytest.raises(FileNotFoundError):
        score_models(args, protocol, DEFAULT_SETTINGS, data, audit, plan, identity)
    assert not (args.public_root / "OUTER_OPENED.json").exists()
    train_models(args, protocol, DEFAULT_SETTINGS, data, audit, plan, identity)
    assert len(list((args.public_root / "models").rglob("TRAIN_COMPLETE.json"))) == 180
    score_models(args, protocol, DEFAULT_SETTINGS, data, audit, plan, identity)
    summarize(args, protocol, DEFAULT_SETTINGS, plan, identity)
    completion = json.loads((args.public_root / "B_SUMMARY_COMPLETE.json").read_text())
    audit = json.loads(
        (args.public_root / "B_INDEPENDENT_METRIC_AUDIT.json").read_text()
    )
    assert completion["status"] == "COMPLETE"
    assert audit["outer_units"] == 180
    assert audit["recomputed_cell_predictions"] == 360
    assert completion["gate"]["final_target_query_labels_used"] == 0
