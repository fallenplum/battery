import numpy as np
import pytest
from scipy.signal import savgol_filter

from scripts.drs_peft.spectral_rotation_features import (
    COLUMNS,
    VOLTAGE_GRID_V,
    _describe_cell,
    _shared_grid_curves,
    summarize_signed_channels,
)


def _viq(voltage, capacity):
    return np.stack((np.broadcast_to(voltage, capacity.shape), np.full_like(capacity, -2.2), capacity), axis=-1)


def test_shared_voltage_coordinates_do_not_restore_missing_measurements():
    native = np.linspace(2.8, 3.3, 64)
    q_native = 1.1 - .25 * native
    raw = _viq(native, np.tile(q_native, (3, 1)))
    mask = np.ones((3, 64), dtype=bool)
    mask[1, 12] = False
    mask[2, -1] = False
    shared, qualified = _shared_grid_curves(raw, mask, native)
    assert qualified.tolist() == [True, False, True]
    assert np.isnan(shared[1]).all()
    np.testing.assert_allclose(shared[qualified], np.tile(1.1 - .25 * VOLTAGE_GRID_V, (2, 1)), atol=1e-14)
    with pytest.raises(ValueError, match="exceeds native measurement support"):
        _shared_grid_curves(raw, mask, np.linspace(2.8, 3.09, 64))


def test_signed_summaries_preserve_reversal_and_actual_cycle_slopes():
    cycles = np.array([11, 15, 19, 23, 29, 81, 84, 89, 97, 100])
    response = np.column_stack((-.003 * cycles, .007 * cycles))
    signed = summarize_signed_channels(response, cycles).reshape(2, 3)
    np.testing.assert_allclose(signed[:, 2], [-.3, .7], atol=1e-14)
    assert signed[0, 1] < 0 < signed[1, 1]
    contaminated = np.vstack((response, [1e30, -1e30]))
    np.testing.assert_array_equal(signed.ravel(), summarize_signed_channels(contaminated, np.r_[cycles, 101]))
    missing = summarize_signed_channels(response[1:], cycles[1:]).reshape(2, 3)
    assert np.isnan(missing[:, 1]).all()
    with pytest.raises(ValueError, match="unique integer"):
        summarize_signed_channels(response, np.r_[cycles[:-1], cycles[-2]])


def test_signed_descriptors_recover_the_known_loaded_response_signs():
    grid = VOLTAGE_GRID_V
    cycles = np.arange(1, 101)
    reference = .6 - .13 * (grid - 2.8) + .021 * np.sin(12 * (grid - 2.8))
    derivative = np.gradient(savgol_filter(reference, 5, 2), grid, edge_order=2)
    actual_a = np.where(cycles > 10, -.002 * cycles / 100., 0.)
    actual_b = np.where(cycles > 10, .003 * cycles / 100., 0.)
    raw = _viq(grid, reference + actual_a[:, None] + actual_b[:, None] * derivative)
    features, info = _describe_cell(raw, np.ones((100, 64), bool), cycles, 2.0, grid)
    assert features.shape == (len(COLUMNS),)
    assert info["decomposed_curves"] == 100
    assert info["reconstruction_max_abs_Ah"] < 1e-14
    np.testing.assert_allclose(features[0:3], [-.000905, -.0007, -.001], atol=1e-11)
    np.testing.assert_allclose(features[3:6], [.002715, .0021, .003], atol=1e-11)
    np.testing.assert_allclose(features[6:30], 0., atol=1e-11)
