"""Regression checks for fixed-start and physical-amplitude-controlled gates."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from models.drs_peft.e123_conditioning import (
    BoundedConditionedLinear,
    ConditionedBatteryTSFM,
)
from models.drs_peft.peft import SubspaceLinear, random_orthogonal_bases
from models.drs_peft.trajectory import BatteryTSFM, CycleQueryHead
from scripts.drs_peft.train import load_selected, selected_state


def layer_fixture(scaled=False):
    torch.manual_seed(42)
    u, v = random_orthogonal_bases(7, 7, 4, 42)
    if scaled:
        v = v * torch.tensor([1.0, 3.0, 10.0, 40.0])
    return SubspaceLinear(
        nn.Linear(7, 7),
        u,
        v,
        core=torch.randn(4, 4) * 0.1,
        require_orthogonal=not scaled,
        method="lora_xs" if scaled else "grad_core",
    )


class ToyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.module = nn.Module()
        for index in range(12):
            self.backbone.module.add_module(f"layer{index}", layer_fixture())
        self.fusion = nn.Linear(7, 64)

    def encode(self, x, point_mask, cycle_ids, policy, policy_mask, anchor):
        values = torch.cat((x.mean((1, 2)), policy), dim=-1)
        return self.fusion(
            sum(
                layer(values)
                for layer in self.backbone.module.children()
                if isinstance(layer, SubspaceLinear)
            )
        )

    def parameter_budget(self):
        return {}


def static_model():
    torch.manual_seed(42)
    model = BatteryTSFM.__new__(BatteryTSFM)
    nn.Module.__init__(model)
    model.bridge = ToyBridge()
    model.head = CycleQueryHead()
    model.bridge_kind = "toy"
    model.official_backbone_identity = {"toy": True}
    return model


def early_batch(batch=3):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(batch, 100, 64, 4)).astype(np.float32)
    return {
        "x": torch.tensor(x),
        "point_mask": torch.ones(batch, 100, 64, dtype=torch.bool),
        "cycle_ids": torch.arange(1, 101).repeat(batch, 1),
        "policy": torch.ones(batch, 3),
        "policy_mask": torch.ones(batch, 3, dtype=torch.bool),
        "anchor": torch.ones(batch),
        "diagnostic_absolute": torch.randn(batch, 12),
        "diagnostic_relative": torch.randn(batch, 12),
    }


@pytest.mark.parametrize("mode", ("constant", "absolute", "relative", "dual"))
@pytest.mark.parametrize("radius", (0.25, 1.0, 4.0))
def test_fixed_static_start_is_bitwise_identical_for_all_modes_and_radii(mode, radius):
    base = static_model().eval()
    reference = copy.deepcopy(base)
    model = ConditionedBatteryTSFM(base, mode, radius).eval()
    early = early_batch()
    query = torch.tensor([100.0, 101.0, 500.0, 5000.0])
    assert torch.equal(model(early, query), reference(early, query))
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        == 1440
    )
    assert all(not parameter.requires_grad for parameter in model.head.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.bridge.fusion.parameters()
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert ".delta_core" in name or ".core_gates." in name
    assert all(
        result["zero_increment"].all()
        for result in model.update_diagnostics(early).values()
    )


@pytest.mark.parametrize("scaled", (False, True))
def test_ball_controls_full_gated_increment_in_physical_weight_coordinates(scaled):
    source = layer_fixture(scaled)
    layer = BoundedConditionedLinear(source, 0.25, source_norm_reference=0.05).eval()
    with torch.no_grad():
        layer.delta_core.fill_(1.0)
    absolute = torch.tensor([[0.5, 1.5, 0.8, 1.2], [1.5, 1.0, 1.0, 0.5]])
    relative = absolute.flip(1)
    increment, result = layer.conditioned_increment(2, absolute, relative)
    actual = (
        torch.linalg.matrix_norm(
            layer.U.double()[None] @ increment.double() @ layer.V.double().T[None],
            dim=(-2, -1),
            ord="fro",
        )
        * layer.scale
    )
    torch.testing.assert_close(result["increment_norm"], actual, atol=1e-10, rtol=1e-7)
    torch.testing.assert_close(
        actual, torch.full_like(actual, 0.0125), atol=1e-9, rtol=2e-6
    )
    assert result["cap_active"].all()
    assert not torch.equal(layer.source_core, layer.source_core + layer.delta_core)


def test_matched_norm_is_exact_for_nonzero_updates_and_reports_zero_denominator():
    layer = BoundedConditionedLinear(
        layer_fixture(True), 4, source_norm_reference=0.02
    ).eval()
    layer.evaluation_mode = "matched_norm"
    _, zero = layer.conditioned_increment(2)
    assert zero["matched_norm_unattainable"].all()
    assert zero["increment_norm"].eq(0).all()
    with torch.no_grad():
        layer.delta_core.fill_(1e-5)
    _, nonzero = layer.conditioned_increment(2)
    torch.testing.assert_close(
        nonzero["increment_norm"],
        torch.full_like(nonzero["increment_norm"], 0.08),
        atol=1e-8,
        rtol=2e-6,
    )
    assert not nonzero["matched_norm_unattainable"].any()
    layer.train()
    with pytest.raises(RuntimeError, match="model.eval"):
        layer.conditioned_increment(2)


def test_zero_radius_keeps_fixed_start_and_zero_update():
    source = layer_fixture()
    reference = copy.deepcopy(source).eval()
    layer = BoundedConditionedLinear(source, 0, source_norm_reference=0.05).eval()
    with torch.no_grad():
        layer.delta_core.fill_(1)
    value = torch.randn(3, 9, 7)
    assert torch.equal(layer(value), reference(value))
    assert layer.conditioned_increment(3)[1]["increment_norm"].eq(0).all()
    layer(value).sum().backward()
    assert layer.delta_core.grad.eq(0).all()


def test_zero_initial_increment_first_backward_is_finite():
    model = ConditionedBatteryTSFM(static_model(), "dual", 1)
    output = model(early_batch(), torch.tensor([500.0, 1000.0]))
    output.square().mean().backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable
    )
    assert any(
        parameter.grad.abs().sum() > 0
        for parameter in model.backbone.core_gates.parameters()
    )


def test_selected_state_restores_all_controls_without_replacing_optimizer_parameters():
    model = ConditionedBatteryTSFM(static_model(), "dual", 1).eval()
    with torch.no_grad():
        for layer in model.backbone.modules():
            if isinstance(layer, BoundedConditionedLinear):
                layer.delta_core.fill_(0.01)
        for parameter in model.backbone.core_gates.parameters():
            parameter.fill_(0.03)
    saved = selected_state(model)
    for key in model.state_dict():
        if any(token in key for token in ("core_", "source_core", "delta_core")):
            assert key in saved
    restored = ConditionedBatteryTSFM(static_model(), "dual", 1).eval()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in restored.parameters() if parameter.requires_grad]
    )
    identities = [
        id(parameter) for parameter in restored.parameters() if parameter.requires_grad
    ]
    load_selected(restored, saved)
    assert identities == [
        id(parameter) for parameter in restored.parameters() if parameter.requires_grad
    ]
    assert identities == [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    early = early_batch()
    query = torch.tensor([101.0, 500.0, 5000.0])
    assert torch.equal(model(early, query), restored(early, query))
    assert all(
        not bool(layer.target_active)
        for layer in restored.backbone.modules()
        if isinstance(layer, BoundedConditionedLinear)
    )


def test_training_updates_delta_and_gates_with_frozen_start_and_bridge_head():
    model = ConditionedBatteryTSFM(static_model(), "dual", 1)
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    with torch.no_grad():
        for layer in model.backbone.modules():
            if isinstance(layer, BoundedConditionedLinear):
                layer.delta_core.fill_(0.01)
    early = early_batch()
    output = model(early, torch.tensor([500.0, 1000.0]))
    output.square().mean().backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable
    )
    assert any(
        parameter.grad.abs().sum() > 0
        for parameter in model.backbone.core_gates.parameters()
    )
    optimizer = torch.optim.AdamW(trainable, lr=0.001)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name])


def test_reference_is_point_zero_one_base_weight_norm_and_modes_validate():
    source = layer_fixture()
    layer = BoundedConditionedLinear(source, 1)
    assert float(layer.core_norm_reference) == pytest.approx(
        0.01 * float(source.base.weight.double().norm())
    )
    for radius in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            BoundedConditionedLinear(source, radius)
    with pytest.raises(ValueError):
        BoundedConditionedLinear(source, 1, source_norm_reference=0)
    model = ConditionedBatteryTSFM(static_model(), "constant", 1)
    with pytest.raises(ValueError):
        model.set_evaluation_mode("unknown")
