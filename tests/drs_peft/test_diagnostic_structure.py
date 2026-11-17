"""Regressions for source diagnostic coordinate geometry."""

import numpy as np
import torch
from torch import nn

from models.drs_peft.peft import SubspaceLinear
from scripts.drs_peft.diagnostic_structure import (
    normalized_weight_jacobian,
    spectral_summary,
    update_singular_values,
    zero_cores,
)


def test_compact_singular_spectrum_matches_dense_update():
    rng = np.random.default_rng(42)
    u = rng.normal(size=(9, 3))
    v = rng.normal(size=(7, 3))
    core = rng.normal(size=(3, 3))
    expected = np.linalg.svd(u @ core @ v.T, compute_uv=False)[:3]
    np.testing.assert_allclose(update_singular_values(u, core, v), expected, atol=1e-12)


def test_normalized_jacobian_removes_xs_sigma_rescaling():
    u = np.eye(4)[:, :2]
    v = np.eye(5)[:, :2]
    sigma = np.array([3.0, 0.2])
    reference = np.array([[1.0, 2.0], [4.0, 5.0]])
    transformed = normalized_weight_jacobian(
        reference * sigma[None], u, v * sigma[None]
    )
    np.testing.assert_allclose(np.linalg.norm(transformed), np.linalg.norm(reference))


def test_general_coordinate_jacobian_matches_dense_directional_derivative():
    rng = np.random.default_rng(123)
    u = rng.normal(size=(7, 3))
    v = rng.normal(size=(9, 3))
    dense_gradient = rng.normal(size=(7, 9))
    qu = np.linalg.qr(u, mode="reduced")[0]
    qv = np.linalg.qr(v, mode="reduced")[0]
    core_gradient = u.T @ dense_gradient @ v
    actual = normalized_weight_jacobian(core_gradient, u, v)
    np.testing.assert_allclose(actual, qu.T @ dense_gradient @ qv, atol=1e-12)


def test_zero_spectrum_is_finite_and_full_flat_spectrum_has_full_rank():
    zero = spectral_summary(np.zeros(4))
    assert zero["entropy_rank"] == zero["stable_rank"] == zero["rank99"] == 0
    full = spectral_summary(np.ones(4))
    assert full["entropy_rank"] == full["stable_rank"] == full["rank99"] == 4


def test_zero_core_context_restores_state_even_when_body_raises():
    layer = SubspaceLinear(nn.Linear(3, 3), torch.eye(3)[:, :2], torch.eye(3)[:, :2])
    with torch.no_grad():
        layer.core.fill_(0.2)
    original = layer.core.detach().clone()
    try:
        with zero_cores({"layer": layer}):
            assert not bool(layer.core.any())
            raise ValueError("intentional")
    except ValueError:
        pass
    assert torch.equal(layer.core, original)
