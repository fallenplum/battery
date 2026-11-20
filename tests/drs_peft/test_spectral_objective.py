import math

import pytest
import torch
from torch import nn

from scripts.drs_peft.spectral_objective import masked_cell_loss, unroll_support
from scripts.drs_peft.spectral_rotation import SpectralRotationModel


@pytest.mark.parametrize(
    ("kind", "value", "gradient"),
    (
        ("mse", 16 / 3, [[-1 / 3, 2 / 3, 0], [3, 0, 0]]),
        ("mae", 2.0, [[-1 / 6, 1 / 6, 0], [0.5, 0, 0]]),
    ),
)
def test_equal_cell_loss_with_unequal_lengths_and_zero_residual(kind, value, gradient):
    prediction = torch.tensor(
        [[0.0] * 3, [3.0, 1.0, -2.0]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor(
        [[1.0, -2.0, 0.0], [0.0, math.nan, math.nan]], dtype=prediction.dtype
    )
    observed = torch.isfinite(target)
    loss = masked_cell_loss(prediction, target, observed, kind)
    assert float(loss) == pytest.approx(value, abs=1e-15)
    actual = torch.autograd.grad(loss, prediction)[0]
    torch.testing.assert_close(
        actual, prediction.new_tensor(gradient), atol=1e-15, rtol=0
    )


class ToyMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.controller = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        nn.init.zeros_(self.controller.weight)

    def initial_cores(self):
        return {"small_core": self.controller.weight.new_zeros(2, requires_grad=True)}

    def forward(self, early, query, descriptor, cores):
        action = self.controller(descriptor)
        gate = torch.cat((1 + action, 1 - 0.3 * action), dim=-1)
        activated = ((early * gate) @ cores["small_core"]).tanh()
        return 0.85 + activated[:, None] * query[None, :]


@pytest.fixture
def toy_episode():
    model = ToyMapper()
    dtype = model.controller.weight.dtype
    early = torch.tensor([[0.2, 1.6], [1.1, -0.4], [0.8, 0.55]], dtype=dtype)
    query = torch.tensor([0.25, 0.4, 0.8], dtype=dtype)
    labels = torch.tensor(
        [[0.45, 0.46, 0.47], [0.63, 0.65, math.nan], [0.30, 0.33, 0.35]], dtype=dtype
    )
    descriptor = torch.tensor([[1.2], [-1.6], [0.45]], dtype=dtype)
    return model, early, query, labels, labels.isfinite(), descriptor


def test_unroll_numeric_parity_with_v1_and_detached_episode(toy_episode):
    model, early, query, target, observed, descriptor = toy_episode
    kwargs = {"lr": 0.1, "steps": 3, "clip_norm": 0.10}
    exact, trace = unroll_support(
        model, early, query, target, observed, descriptor, differentiable=True, **kwargs
    )
    detached, detached_trace = unroll_support(
        model,
        early,
        query,
        target,
        observed,
        descriptor,
        differentiable=False,
        **kwargs,
    )
    original = SpectralRotationModel.adapt_episode(
        model, early, query, target, observed, descriptor, differentiable=True, **kwargs
    )
    assert exact["small_core"].grad_fn is not None
    assert detached["small_core"].grad_fn is None
    assert torch.equal(exact["small_core"], detached["small_core"])
    assert torch.equal(exact["small_core"], original["small_core"])
    assert torch.equal(
        model(early, query, descriptor, exact),
        model(early, query, descriptor, detached),
    )
    assert torch.equal(torch.stack(trace), torch.stack(detached_trace))
    assert all(not value.requires_grad for value in trace)
    assert all(float(row[2]) <= kwargs["clip_norm"] for row in trace)
    assert any(float(row[3]) < 1.0 for row in trace)


def test_exact_outer_mae_hypergradient_and_disconnection(toy_episode):
    model, early, query, target, observed, descriptor = toy_episode
    test_data = early.flip(0) + 0.20
    # A zero Query descriptor removes the direct contribution of the mapper.
    no_direct_descriptor = torch.zeros_like(descriptor)
    query_labels = torch.full_like(target, 0.15)

    def outer(differentiable):
        state, _ = unroll_support(
            model,
            early,
            query,
            target,
            observed,
            descriptor,
            steps=3,
            lr=0.1,
            clip_norm=0.10,
            differentiable=differentiable,
        )
        prediction = model(test_data, query, no_direct_descriptor, state)
        assert float((prediction.detach() - query_labels).abs().min()) > 0.2
        return masked_cell_loss(prediction, query_labels, observed, kind="mae")

    parameter = model.controller.weight
    exact = torch.autograd.grad(outer(differentiable=True), parameter)[0].item()
    disconnected = torch.autograd.grad(outer(differentiable=False), parameter)[0].item()
    assert disconnected == 0.0
    assert abs(exact) > 1e-6
    epsilon = 1e-5
    with torch.no_grad():
        parameter.fill_(epsilon)
    upper = float(outer(differentiable=False).detach())
    with torch.no_grad():
        parameter.fill_(-epsilon)
    lower = float(outer(differentiable=False).detach())
    with torch.no_grad():
        parameter.zero_()
    assert exact == pytest.approx((upper - lower) / (2 * epsilon), rel=1e-6, abs=1e-10)
