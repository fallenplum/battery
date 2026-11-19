import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.drs_peft.abba_adapters import make_adapter


@pytest.mark.parametrize("method", ("lora", "hira", "abba"))
def test_zero_step_freezes_base_and_matches_parameter_budget(method):
    source = nn.Linear(19, 23, dtype=torch.float64)
    values = torch.linspace(-2, 2, 6 * 19, dtype=source.weight.dtype).reshape(2, 3, 19)
    expected = source(values).detach()
    rng_before = torch.random.get_rng_state()

    adapter = make_adapter(source, method, 8, seed=12)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(expected, adapter(values))
    assert torch.count_nonzero(adapter.delta_weight()) == 0
    assert not any(p.requires_grad for p in source.parameters())
    assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == 8 * (19 + 23)

    adapter(values).square().mean().backward()
    for name, value in adapter.named_parameters():
        if name.startswith("base."):
            assert value.grad is None
        elif name in ("lora_b2", "lora_b.weight"):
            assert torch.count_nonzero(value.grad) > 0
        else:
            assert torch.count_nonzero(value.grad) == 0


def test_abba_reproduces_source_randomized_svd_and_attains_its_rank_ceiling():
    generator = torch.Generator().manual_seed(11)
    source = nn.Linear(19, 23, bias=False, dtype=torch.float64)
    with torch.no_grad():
        spectrum = torch.diag(torch.logspace(1, -1, 19, dtype=source.weight.dtype))
        u = torch.linalg.qr(torch.randn(23, 19, generator=generator, dtype=source.weight.dtype)).Q
        v = torch.linalg.qr(torch.randn(19, 19, generator=generator, dtype=source.weight.dtype)).Q
        source.weight.copy_(u @ spectrum @ v.T)

    layer = make_adapter(source, "abba", (4, 4), alpha=2.0, seed=42)
    assert layer.alpha1 == layer.alpha2 == layer.scale == 1.0
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        u, s, v = torch.svd_lowrank(source.weight.float(), q=4, niter=10)
        a1 = (v.T * s.sqrt().unsqueeze(1)).contiguous()
        b1 = (u * s.sqrt()).contiguous()
        a2 = nn.init.kaiming_uniform_(torch.randn_like(a1))
    torch.testing.assert_close(layer.lora_a1, a1.double(), atol=0, rtol=0)
    torch.testing.assert_close(layer.lora_b1, b1.double(), atol=0, rtol=0)
    torch.testing.assert_close(layer.lora_a2, a2.double(), atol=0, rtol=0)

    u, s, vh = torch.linalg.svd(source.weight, full_matrices=False)
    oracle = (u[:, :4] * s[:4]) @ vh[:4]
    approx = layer.lora_b1 @ layer.lora_a1
    torch.testing.assert_close(approx, oracle, atol=2e-3, rtol=2e-3)

    with torch.no_grad():
        layer.lora_b2.normal_(generator=generator)
    assert torch.linalg.matrix_rank(layer.delta_weight()).item() == 16


@pytest.mark.parametrize("method", ("lora", "hira", "abba"))
def test_nonzero_forward_and_gradients_match_explicit_update_and_merge(method):
    generator = torch.Generator().manual_seed(13)
    source = nn.Linear(19, 23, dtype=torch.float64)
    alpha = {"lora": 11, "hira": None, "abba": (9, 16)}[method]
    adapter = make_adapter(source, method, (4, 4) if method == "abba" else 8, alpha=alpha)
    frozen = source.weight.detach().clone()
    values = torch.randn(2, 3, 19, generator=generator, dtype=frozen.dtype, requires_grad=True)
    with torch.no_grad():
        for param in adapter.parameters():
            if param.requires_grad:
                param.normal_(std=0.15, generator=generator)

    if method == "abba":
        expected_delta = 12 * (adapter.lora_b1 @ adapter.lora_a1) * (
            adapter.lora_b2 @ adapter.lora_a2
        )
    elif method == "hira":
        expected_delta = source.weight * (adapter.lora_b.weight @ adapter.lora_a.weight)
    else:
        expected_delta = (11 / 8) * (adapter.lora_b.weight @ adapter.lora_a.weight)
    oracle = F.linear(values, frozen + expected_delta, source.bias)
    actual = adapter(values)
    torch.testing.assert_close(adapter.delta_weight(), expected_delta)
    torch.testing.assert_close(actual, oracle)
    torch.testing.assert_close(adapter.merged_linear()(values), oracle)
    assert torch.equal(source.weight, frozen)

    parameters = tuple(param for param in adapter.parameters() if param.requires_grad)
    expected_grad = torch.autograd.grad(oracle.square().mean(), (values, *parameters))
    actual_grad = torch.autograd.grad(actual.square().mean(), (values, *parameters))
    for actual_part, expected_part in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_part, expected_part)


def test_abba_refuses_unimplemented_asymmetric_initialization():
    source = nn.Linear(19, 23)
    with pytest.raises(ValueError, match="equal factor ranks"):
        make_adapter(source, "abba", (3, 5))


def test_matched_initializations_use_same_lo_ra_and_hira_kaiming_draw():
    source = nn.Linear(19, 23, dtype=torch.float64)
    lora = make_adapter(copy.deepcopy(source), "lora", 8, seed=18)
    hira = make_adapter(copy.deepcopy(source), "hira", 8, seed=18)
    assert torch.equal(lora.lora_a.weight, hira.lora_a.weight)
    assert lora.lora_a.weight.abs().max() <= 1.0 / math.sqrt(19)
