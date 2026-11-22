import numpy as np
import pytest

from scripts.drs_peft.self_reference_features import (
    COLUMNS,
    _signed_channels,
    capacity_reference,
    paired_electrical_values,
)
from scripts.unified_matr.physics_features import PhysicsConfig, extract_prefix_features


def test_probe_uses_exact_early_observed_median_and_keeps_same_vi():
    cycles = np.arange(1, 101)[None]
    voltage = np.array([2.8, 3.1, 3.3], dtype=np.float64)
    q = cycles[..., None] + np.arange(3)[None, None, :] / 4
    raw = np.stack((np.broadcast_to(voltage, q.shape), np.full_like(q, -3.4), q), axis=-1)
    mask = np.ones(q.shape, dtype=bool)
    mask[0, 0, 1] = False
    mask[0, 4, 2] = False
    raw[~mask, 2] = np.nan
    real, reference, template, counts, drift = paired_electrical_values(raw, mask, cycles, [2.0])
    np.testing.assert_array_equal(counts, [[10, 9, 9]])
    np.testing.assert_allclose(template[0], [5.5, 6.25, 6.5])
    np.testing.assert_array_equal(real[..., :2], reference[..., :2])
    assert not reference[..., 3].any()
    np.testing.assert_allclose(reference[0, -1, :, 2], template[0] / 2)
    np.testing.assert_allclose(real[..., 3], drift / 2)
    assert not reference[~mask].any()
    contaminated = raw.copy()
    contaminated[:, 10:, :, 2] += 1e6
    later = paired_electrical_values(contaminated, mask, cycles, [2.0])
    np.testing.assert_array_equal(later[1], reference)
    np.testing.assert_array_equal(later[2], template)


def test_missing_reference_fails_closed_without_changing_real_mask():
    q = np.zeros((1, 100, 2))
    cycles = np.arange(1, 101)[None]
    mask = np.ones_like(q, dtype=bool)
    mask[0, :6, 1] = False
    with pytest.raises(ValueError, match="fewer than five early reference curves"):
        capacity_reference(q, mask, cycles)
    mask[0, :6, 1] = True
    cycles[0, 98] = cycles[0, 99]
    with pytest.raises(ValueError, match="observed cycle ID is repeated"):
        capacity_reference(q, mask, cycles)


def test_native_physics_close_against_exact_template():
    n = 100
    grid = np.linspace(2.8, 3.3, 64)
    time = np.arange(1, n + 1)
    base = .63 - .17 * (grid - 2.8) + .016 * np.sin(14 * (grid - 2.8))
    variation = np.maximum(time - 10, 0) / 100 * -.001
    q = base[None] + variation[:, None]
    masks = np.ones((1, n, 64), bool)
    raw = np.stack((np.broadcast_to(grid, q.shape), np.full_like(q, -3.0), q), -1)[None]
    features = extract_prefix_features(q, masks[0].all(1), time, grid, PhysicsConfig())
    fields = {
        key: features[key][None].copy()
        for key in ("global_features", "global_mask", "local_features", "local_feature_mask", "local_residual", "local_mask")
    }
    fields.update(raw_viq=raw, point_mask=masks, nominal_capacity_Ah=np.array([2.]))
    _, _, template, _, drift = paired_electrical_values(raw, masks, time[None], [2.])
    values, available, _, _ = _signed_channels(fields, grid, template, drift)
    assert values.shape == available.shape == (1, 100, 18)
    assert len(COLUMNS) == 30
    assert available.all() and values[0, -1, 0] < 0
    np.testing.assert_allclose(values[0, -1, 0], variation[-1] / 2, atol=1e-9)
    fields["local_features"][0, -1, 0, 7] += .001
    with pytest.raises(ValueError, match="different capacity reference"):
        _signed_channels(fields, grid, template, drift)
