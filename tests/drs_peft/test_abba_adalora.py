"""CPU regression checks for the AdaLoRA operator and allocation schedule."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.drs_peft.abba_adalora import (
    AdaLoRALinear,
    AdaLoRARankAllocator,
    adalora_orthogonal_regularization,
)


def test_adalora_affine_update_and_initial_identity():
    base = nn.Linear(7, 5, dtype=torch.float64)
    x = torch.linspace(-0.5, 0.5, 28, dtype=torch.float64).reshape(2, 2, 7)
    initial = base(x).detach()
    state = torch.get_rng_state().clone()
    layer = AdaLoRALinear(base, rank=3, alpha=6, seed=42)
    torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
    torch.testing.assert_close(layer(x), initial, atol=0, rtol=0)
    assert all(not param.requires_grad for param in (base.weight, base.bias, layer.ranknum))
    assert layer.ranknum.item() == 3

    with torch.no_grad():
        layer.lora_E[:, 0] = torch.tensor([0.5, 0.0, -0.75])
    expected = layer.lora_B @ torch.diag(layer.lora_E.flatten()) @ layer.lora_A
    expected = expected * (6 / (3 + 1e-5))
    torch.testing.assert_close(layer.delta_weight(), expected)
    torch.testing.assert_close(layer(x), initial + F.linear(x, expected))
    torch.testing.assert_close(layer.eval()(x), initial + F.linear(x, expected))


def test_adalora_zero_based_cubic_budget_and_fixed_final_pattern():
    model = nn.Sequential(
        AdaLoRALinear(nn.Linear(8, 7, dtype=torch.float64), rank=3, seed=42),
        AdaLoRALinear(nn.Linear(7, 5, dtype=torch.float64), rank=3, seed=43),
    )
    allocator = AdaLoRARankAllocator(
        model, target_rank=1, total_steps=12, initial_warmup=1, final_warmup=4, mask_interval=2
    )
    expected = [(6, False), (6, False), (4, True), (3, False), (2, True), (2, False)]
    expected.extend([(2, True), (2, False), (2, True), (2, True), (2, True), (2, True)])
    assert [allocator.schedule_threshold(step) for step in range(12)] == expected

    x = torch.ones(4, 8, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    final_patterns = []
    for step, (expected_rank, masked) in enumerate(expected):
        loss = (model(x) - 1).square().mean() + adalora_orthogonal_regularization(model)
        loss.backward()
        optimizer.step()
        budget, threshold = allocator.update_and_mask(step)
        if step == 0:
            singular_values = model[0].lora_E
            sensitivity = (singular_values.detach() * singular_values.grad).abs()
            assert sensitivity.count_nonzero() > 0
            torch.testing.assert_close(allocator._mean_sensitivity[singular_values], 0.15 * sensitivity)
        optimizer.zero_grad(set_to_none=True)
        assert budget == expected_rank
        assert (threshold is not None) == masked
        assert all(layer.ranknum.item() == 3 for layer in model)
        if masked:
            assert sum(allocator.rank_pattern.values()) == budget
            assert sum(layer.lora_E.count_nonzero().item() for layer in model) == budget
        if step >= 8:
            final_patterns.append((threshold, allocator.rank_pattern.copy()))
    assert final_patterns == [final_patterns[0]] * 4


def test_adalora_threshold_removes_all_ties():
    layer = AdaLoRALinear(nn.Linear(4, 5), rank=3, seed=42)
    with torch.no_grad():
        layer.lora_E.fill_(1)
    allocator = AdaLoRARankAllocator(
        layer, target_rank=1, total_steps=5, initial_warmup=0, final_warmup=2, mask_interval=1
    )
    with pytest.raises(RuntimeError, match="needs gradients"):
        allocator.update_and_mask(0)
    for factor in (layer.lora_A, layer.lora_B, layer.lora_E):
        factor.grad = torch.zeros_like(factor)
    assert allocator.update_and_mask(0) == (3, None)
    assert allocator.update_and_mask(1) == (1, 0.0)
    assert allocator.rank_pattern == {"": 0}
    assert layer.lora_E.count_nonzero() == 0
