"""Numerical contracts for source-scaled and Query-local calibration."""

import numpy as np
import pytest

from scripts.drs_peft.transfer_scope_geometry import (
    calibration_design,
    cell_normal_equations,
    fit_prefix_geometry,
    fit_scales,
    neighbor_statistics,
    solve_calibration,
)


def test_fit_geometry_uses_positive_distances_and_source_statistics():
    source = np.array([[0.0, 1.0, 7.3], [0.0, 1.0, 7.3], [2.0, 3.0, 7.3]])
    geometry = fit_prefix_geometry(source)
    np.testing.assert_array_equal(geometry.mask, [True, True, False])
    np.testing.assert_allclose(geometry.source_std, source.std(axis=0))
    expected = 2.0 / source[:, 0].std()
    assert geometry.d0 == pytest.approx(expected)

    support = source[1:]
    query = np.array([[0.0, 1.0, 99.0], [8.0, 9.0, -99.0]])
    weights = geometry.weights(support, query, 1.0)
    np.testing.assert_allclose(weights[0], [1.0, np.exp(-0.5)])
    np.testing.assert_allclose(weights[1], np.exp(-0.5 * np.array([4.0, 3.0]) ** 2))
    np.testing.assert_array_equal(weights[0:1], geometry.weights(support, query[0], 1.0))
    assert weights[1].sum() < 0.02


def test_constant_source_has_auditable_global_fallback():
    geometry = fit_prefix_geometry(np.full((5, 64), 0.1))
    assert not geometry.distinguishable
    assert geometry.d0 == 0.0
    assert geometry.mask.sum() == 0
    np.testing.assert_array_equal(
        geometry.weights(np.zeros((3, 64)), np.ones((2, 64)), 0.5), np.ones((2, 3))
    )


def test_floor_applies_only_to_nonconstant_dimensions():
    geometry = fit_prefix_geometry([[1.0, 7.0], [1.0 + 1e-10, 7.0]], std_floor=1e-8)
    assert geometry.mask.tolist() == [True, False]
    assert geometry.d0 == pytest.approx((geometry.source_std[0] * 2) / 1e-8)


def test_neighbor_diagnostics_stable_at_tiny_and_zero_weights():
    result = neighbor_statistics([[0.2, 0.4], [0, 0], [1e-300, 1e-300]])
    np.testing.assert_allclose(result["mean_weight"], [0.3, 0.0, 1e-300], atol=0)
    np.testing.assert_allclose(result["max_weight"], [0.4, 0.0, 1e-300], atol=0)
    np.testing.assert_allclose(result["effective_neighbors"], [1.8, 0.0, 2.0])


def test_cal2_design_uses_frozen_reference_and_preserves_anchor():
    reference = np.array([[0.9, 0.8, 0.7], [1.0, 0.9, 0.75]])
    design = calibration_design(reference, [0.9, 1.0], [100, 200, 900])
    np.testing.assert_array_equal(design[:, 0], np.zeros((2, 2)))
    np.testing.assert_allclose(design[..., 0], [[0, -0.1, -0.2], [0, -0.1, -0.25]])
    np.testing.assert_allclose(design[..., 1], [[0, 0.1, 0.8], [0, 0.1, 0.8]])


def test_local_solve_matches_explicit_cell_mean_augmented_least_squares():
    rng = np.random.default_rng(42)
    source = np.stack([np.diag([5.0, 1.0] + [4.0] * 16), np.eye(18)])
    scales = fit_scales(source)
    assert scales == (2.0, 2.5)
    design = [rng.normal(size=(count, 18)) for count in (1, 4, 20)]
    residuals = [rng.normal(size=len(matrix)) for matrix in design]
    grams, rhs = zip(*(cell_normal_equations(x, y) for x, y in zip(design, residuals)))
    weights = np.array([[0.03, 0.1, 0.6], [0.8, 0.2, 0.0]])
    ridge = 0.1 * np.array([scales[0]] * 2 + [scales[1]] * 16)
    expected = []
    for row in weights:
        factors = [np.sqrt(w / (len(row) * len(x))) for w, x in zip(row, design)]
        matrix = np.vstack([*(f * x for f, x in zip(factors, design)), np.diag(np.sqrt(ridge))])
        error = np.concatenate([*(f * y for f, y in zip(factors, residuals)), np.zeros(18)])
        expected.append(0.5 * np.linalg.lstsq(matrix, error, rcond=None)[0])
    for solver in ("cholesky", "schur"):
        result = solve_calibration(
            grams, rhs, weights, lambda_multiplier=0.1, scales=scales, eta=0.5,
            support_ids=["c", "a", "b"], solver=solver,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)


def test_equal_weights_match_global_and_small_weights_shrink_with_fixed_ridge():
    grams = np.stack([np.eye(2), 3.0 * np.eye(2)])
    rhs = np.array([[2.0, 3.0], [4.0, 6.0]])
    settings = dict(lambda_multiplier=1.0, scales=(0.5, None), eta=1.0)
    global_result = solve_calibration(grams, rhs, [1.0, 1.0], **settings)
    result = solve_calibration(
        grams, rhs, [[1.0, 1.0], [1e-9, 1e-9], [0.0, 0.0]], **settings
    )
    np.testing.assert_array_equal(result[0], global_result[0])
    np.testing.assert_allclose(global_result[0], rhs.mean(axis=0) / 2.5)
    assert np.linalg.norm(result[1]) < 1e-7
    np.testing.assert_array_equal(result[2], [0.0, 0.0])


def test_support_order_is_deterministic_and_duplicate_ids_are_rejected():
    rng = np.random.default_rng(123)
    designs = rng.normal(size=(4, 25, 18))
    residuals = rng.normal(size=(4, 25))
    grams, rhs = map(np.asarray, zip(*(cell_normal_equations(x, y) for x, y in zip(designs, residuals))))
    weights = rng.uniform(size=(2, 4))
    ids = np.array(["b", "z", "a", "c"])
    order = [2, 3, 1, 0]
    settings = dict(lambda_multiplier=1.0, scales=(1.0, 1.0), eta=1.0)
    result = solve_calibration(grams, rhs, weights, support_ids=ids, **settings)
    permuted = solve_calibration(
        grams[order], rhs[order], weights[:, order], support_ids=ids[order], **settings
    )
    np.testing.assert_array_equal(result, permuted)
    with pytest.raises(ValueError, match="distinct physical cells"):
        solve_calibration(grams, rhs, weights, support_ids=["b", "z", "a", "a"], **settings)


def test_zero_eta_empty_support_and_inactive_column_limits():
    rng = np.random.default_rng(2026)
    all_columns = rng.normal(size=(25, 18))
    residual = rng.normal(size=25)
    scales = (1.3, 1.7)
    settings = dict(lambda_multiplier=1.0, scales=scales)
    for active in (slice(0, 2), slice(2, 18)):
        design = np.zeros_like(all_columns)
        design[:, active] = all_columns[:, active]
        gram, rhs = cell_normal_equations(design, residual)
        actual = solve_calibration([gram], [rhs], [1.0], eta=1.0, solver="schur", **settings)[0]
        expected = np.zeros(18)
        block_dimension = len(expected[active])
        block_scales = (scales[0] if block_dimension == 2 else scales[1]) * np.eye(block_dimension)
        expected[active] = np.linalg.solve(gram[active, active] + block_scales, rhs[active])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    no_update = solve_calibration([gram], [rhs], [1.0], eta=0.0, **settings)
    np.testing.assert_array_equal(no_update, np.zeros((1, 18)))
    no_support = solve_calibration(
        np.empty((0, 18, 18)), np.empty((0, 18)), np.empty((2, 0)),
        eta=1.0, support_ids=[], **settings,
    )
    np.testing.assert_array_equal(no_support, np.zeros((2, 18)))


@pytest.mark.parametrize("invalid", [[-0.01], [1.01], [np.nan], [np.inf]])
def test_invalid_weights_rejected(invalid):
    with pytest.raises(ValueError):
        solve_calibration(
            [np.eye(2)], [[1.0, 2.0]], invalid, lambda_multiplier=1.0,
            scales=(1.0, None), eta=0.5,
        )


def test_zero_source_gram_has_strict_positive_fallback():
    assert fit_scales(np.zeros((1, 18, 18))) == (1e-12, 1e-12)
    assert fit_scales(np.stack([1e-14 * np.eye(2)])) == (1e-14, None)
