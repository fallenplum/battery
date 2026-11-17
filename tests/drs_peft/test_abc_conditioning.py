"""Regression checks for the Bridge/Head by conditioning factorial."""

import copy

import pytest
import torch
from torch import nn

from models.drs_peft.abc_conditioning import FactorialConditionedBatteryTSFM
from models.drs_peft.e123_conditioning import ConditionedBatteryTSFM
from models.drs_peft.peft import SubspaceLinear, random_orthogonal_bases
from models.drs_peft.trajectory import BatteryTSFM, CycleQueryHead
from scripts.drs_peft.train import load_selected, selected_state


class ToyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.module = nn.Module()
        for index in range(12):
            u, v = random_orthogonal_bases(7, 7, 4, index)
            self.backbone.module.add_module(
                f"layer{index}",
                SubspaceLinear(nn.Linear(7, 7), u, v, core=torch.randn(4, 4) * 0.1),
            )
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
    return {
        "x": torch.randn(batch, 100, 64, 4),
        "point_mask": torch.ones(batch, 100, 64, dtype=torch.bool),
        "cycle_ids": torch.arange(1, 101).repeat(batch, 1),
        "policy": torch.ones(batch, 3),
        "policy_mask": torch.ones(batch, 3, dtype=torch.bool),
        "anchor": torch.ones(batch),
        "diagnostic_absolute": torch.randn(batch, 12),
        "diagnostic_relative": torch.randn(batch, 12),
    }


@pytest.mark.parametrize("mode", ("constant", "absolute", "dual", "shuffled_dual"))
def test_joint_and_frozen_start_with_identical_predictions_and_adapter_budget(mode):
    base = static_model()
    frozen = FactorialConditionedBatteryTSFM(copy.deepcopy(base), mode, 1).eval()
    joint = FactorialConditionedBatteryTSFM(
        copy.deepcopy(base), mode, 1, train_bridge_head=True
    ).eval()
    early = early_batch()
    query = torch.tensor([100.0, 101.0, 500.0, 5000.0])
    assert torch.equal(frozen(early, query), joint(early, query))
    assert torch.equal(joint(early, query), base.eval()(early, query))
    for name in selected_state(frozen):
        assert torch.equal(selected_state(frozen)[name], selected_state(joint)[name])
    for model in (frozen, joint):
        assert (
            sum(
                p.numel()
                for name, p in model.named_parameters()
                if name.startswith("bridge.backbone.module.") and p.requires_grad
            )
            == 1440
        )
    assert frozen.parameter_report()["bridge_head_trainable_parameters"] == 0
    assert joint.parameter_report()["bridge_head_trainable_parameters"] > 0


@pytest.mark.parametrize("mode", ("constant", "absolute", "dual"))
def test_frozen_operator_is_exactly_the_prior_e123_model(mode):
    base = static_model()
    prior = ConditionedBatteryTSFM(copy.deepcopy(base), mode, 1)
    current = FactorialConditionedBatteryTSFM(copy.deepcopy(base), mode, 1)
    with torch.no_grad():
        for layer in prior.backbone.modules():
            if isinstance(layer, SubspaceLinear):
                layer.delta_core.fill_(0.001)
        for parameter in prior.backbone.core_gates.parameters():
            parameter.fill_(0.01)
    load_selected(current, selected_state(prior))
    early, query = early_batch(), torch.tensor([200.0, 500.0, 1000.0])
    for evaluation in ("trained_ball", "matched_norm"):
        prior.eval().set_evaluation_mode(evaluation)
        current.eval().set_evaluation_mode(evaluation)
        assert torch.equal(prior(early, query), current(early, query))


def test_joint_training_updates_bridge_head_and_adapters_with_fixed_backbone_and_c0():
    model = FactorialConditionedBatteryTSFM(
        static_model(), "dual", 1, train_bridge_head=True
    )
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    model(early_batch(), torch.tensor([300.0, 600.0])).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    optimizer.step()
    changed = {
        name for name, p in model.named_parameters() if not torch.equal(before[name], p)
    }
    assert any(name.startswith("head.") for name in changed)
    assert any(name.startswith("bridge.fusion.") for name in changed)
    assert any(".delta_core" in name for name in changed)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(before[name], parameter)


def test_joint_serialization_restores_common_and_conditional_updates():
    model = FactorialConditionedBatteryTSFM(
        static_model(), "dual", 1, train_bridge_head=True
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)
    state = selected_state(model)
    restored = FactorialConditionedBatteryTSFM(
        static_model(), "dual", 1, train_bridge_head=True
    )
    optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad])
    before = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    load_selected(restored, state)
    assert before == [
        id(parameter) for parameter in restored.parameters() if parameter.requires_grad
    ]
    early, query = early_batch(), torch.tensor([200.0, 500.0, 1000.0])
    assert torch.equal(model.eval()(early, query), restored.eval()(early, query))
