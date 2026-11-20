import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.drs_peft.peft import SubspaceLinear
from scripts.drs_peft.spectral_rotation import (
    SpectralRotationModel,
    build_rotation_bases,
    masked_cell_mse,
)


class SyntheticBattery(nn.Module):
    input_keys = ("anchor", "x")

    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            for _ in range(6):
                attention = nn.Module()
                for name in ("q_proj", "v_proj"):
                    base = nn.Linear(10, 10, dtype=torch.float64)
                    u = torch.linalg.qr(torch.randn(10, 4, dtype=torch.float64)).Q
                    v = torch.linalg.qr(torch.randn(10, 4, dtype=torch.float64)).Q
                    c = torch.randn(4, 4, dtype=torch.float64) * 0.02
                    setattr(attention, name, SubspaceLinear(base, u, v, c))
                self.backbone.layers.append(attention)

    def forward(self, early, query):
        x = early["x"]
        for pair in self.backbone.layers:
            x = x + 0.14 * (pair.q_proj(x).tanh() + pair.v_proj(x).tanh())
        return early["anchor"][:, None] + 0.4 * x.mean(dim=(-2, -1))[:, None] * query


@pytest.fixture(scope="module")
def synthetic():
    model = SyntheticBattery().eval()
    bases = build_rotation_bases(model)
    dtype = torch.float64
    generator = torch.Generator().manual_seed(15)
    early = {
        "x": torch.randn(3, 4, 10, generator=generator, dtype=dtype),
        "anchor": torch.ones(3, dtype=dtype),
    }
    query = torch.arange(1, 5, dtype=dtype) * 0.4
    descriptor = torch.tensor(((1.0, 0.0), (-0.4, 1.7), (0.2, 0.2)), dtype=dtype)
    return model, bases, early, query, descriptor


@pytest.mark.parametrize(
    "mode,rank",
    (
        ("static", 4),
        ("old_static", 4),
        ("constant", 4),
        ("conditioned", 4),
        ("shuffled", 4),
        ("fixed_half", 4),
        ("drs", 4),
        ("gain", 4),
        ("joint", 2),
    ),
)
def test_static_reference_and_equal_budget(synthetic, mode, rank):
    original, bases, early, query, descriptor = synthetic
    model = SpectralRotationModel(original, bases, mode, 2)
    torch.testing.assert_close(
        model(early, query, descriptor, model.initial_cores()), original(early, query)
    )
    assert model.budget()["episode_parameters"] == 192
    assert model.budget()["maximum_matrix_rank"] == rank
    assert model.budget()["source_scales"] == [1.0]
    mapper = {
        "constant": 144,
        "conditioned": 144,
        "shuffled": 144,
        "gain": 144,
        "drs": 288,
    }
    assert model.budget()["controller_parameters"] == mapper.get(mode, 0)
    if mode == "constant":
        assert model.budget()["constant_identifiable_parameters"] == 48
        torch.testing.assert_close(
            model.modulation(3, descriptor), model.modulation(3, descriptor.flip(0))
        )
    assert not any(value.requires_grad for value in model.reference.parameters())
    shape = (8, 2) if mode == "joint" else (4, 4)
    assert all(value.shape == shape for value in model.initial_cores().values())
    for name, basis in bases.items():
        layer = original.backbone.get_submodule(name)
        assert torch.equal(basis.u0, layer.U) and torch.equal(basis.v, layer.V)
        assert basis.audit()["left_cross_gram_max_abs"] < 1e-12


@pytest.mark.parametrize("mode", ("conditioned", "constant", "drs", "gain", "joint"))
def test_matrix_free_forward_and_norm_match_dense(synthetic, mode):
    original, bases, early, query, descriptor = synthetic
    model = SpectralRotationModel(original, bases, mode, 2)
    generator = torch.Generator().manual_seed(6)
    cores = {
        name: torch.randn(value.shape, dtype=value.dtype, generator=generator) * 0.07
        for name, value in model.initial_cores().items()
    }
    if model.controller is not None:
        with torch.no_grad():
            model.controller.weight.normal_(std=0.4, generator=generator)
    mod = model.modulation(len(descriptor), descriptor)
    dense = {}
    dense_norms = []
    for index, (name, layer) in enumerate(model.layers.items()):
        delta = layer.delta_weight(cores[name], None if mod is None else mod[:, index])
        dense[name] = (
            delta if delta.ndim == 3 else delta.expand(len(descriptor), -1, -1)
        )
        dense_norms.append(dense[name].norm(dim=(-2, -1)))
    x = early["x"]
    for index in range(6):
        paths = []
        for side in ("q_proj", "v_proj"):
            name = f"layers.{index}.{side}"
            base = model.layers[name].base
            paths.append(
                torch.stack(
                    [
                        F.linear(
                            x[row], base.weight + dense[name][row], base.bias
                        ).tanh()
                        for row in range(len(x))
                    ]
                )
            )
        x = x + 0.14 * (paths[0] + paths[1])
    expected = early["anchor"][:, None] + 0.4 * x.mean(dim=(-2, -1))[:, None] * query
    torch.testing.assert_close(
        model(early, query, descriptor, cores), expected, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        model.update_frobenius(cores, descriptor, per_layer=True),
        torch.stack(dense_norms, dim=-1),
    )
    scaled = model.update_frobenius(model.scale_cores(cores, 0.25), descriptor)
    torch.testing.assert_close(scaled, model.update_frobenius(cores, descriptor) * 0.25)

    if mode == "conditioned":
        for key, layer in model.layers.items():
            swapped = layer.delta_weight(cores[key], mod[:, layer.index].flip(0))
            expected = torch.linalg.svdvals(cores[key]).expand(len(descriptor), -1)
            torch.testing.assert_close(
                torch.linalg.svdvals(dense[key])[:, :4], expected
            )
            torch.testing.assert_close(torch.linalg.svdvals(swapped)[:, :4], expected)
        moved = model(early, query, descriptor.flip(0), cores)
        assert not torch.allclose(moved, model(early, query, descriptor, cores))


def test_masked_loss_uses_equal_cell_weight_and_ignores_missing_labels():
    predicted = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor(
        [[1.0, 1.0, 1.0], [3.0, math.nan, math.nan]], dtype=predicted.dtype
    )
    observed = torch.isfinite(labels)
    loss = masked_cell_mse(predicted, labels, observed)
    assert float(loss) == 5.0
    derivative = torch.autograd.grad(loss, predicted)[0]
    torch.testing.assert_close(
        derivative, predicted.new_tensor([[-1.0 / 3] * 3, [-3.0, 0.0, 0.0]])
    )


def test_exact_inner_gradient_preserves_support_to_mapper_chain(synthetic):
    original, bases, early, query, descriptor = synthetic
    model = SpectralRotationModel(original, bases, "conditioned", 2)
    query_descriptor = torch.zeros_like(descriptor)
    with torch.no_grad():
        target = model(early, query) + 0.35
        query_target = model(early, query) - 0.30
    observed = torch.ones_like(target, dtype=torch.bool)

    def outer(differentiable):
        cores = model.adapt_episode(
            early,
            query,
            target,
            observed,
            descriptor,
            steps=3,
            lr=0.5,
            clip_norm=0.008,
            differentiable=differentiable,
        )
        return masked_cell_mse(
            model(early, query, query_descriptor, cores), query_target, observed
        )

    detached = torch.autograd.grad(outer(False), model.controller.weight)[0]
    exact = torch.autograd.grad(outer(True), model.controller.weight)[0]
    assert torch.count_nonzero(detached) == 0
    index = tuple(torch.unravel_index(exact.abs().argmax(), exact.shape))
    assert exact[index].abs() > 1e-10
    epsilon = 1e-4
    with torch.no_grad():
        model.controller.weight[index] = epsilon
    upper = float(outer(False).detach())
    with torch.no_grad():
        model.controller.weight[index] = -epsilon
    lower = float(outer(False).detach())
    with torch.no_grad():
        model.controller.weight[index] = 0.0
    torch.testing.assert_close(
        exact[index],
        exact.new_tensor((upper - lower) / (2 * epsilon)),
        atol=5e-12,
        rtol=2e-5,
    )
