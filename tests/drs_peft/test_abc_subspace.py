"""Regression checks for source function, capacity and physical C increments."""

import copy

import pytest
import torch
from torch import nn

from models.drs_peft.abc_subspace import (
    ARMS,
    EXPECTED_PARAMETERS,
    MATCHED_PERP_SHAPES,
    MATCHED_RANKS,
    SubspaceProbeLinear,
    SubspaceProbeTSFM,
)
from models.drs_peft.peft import SubspaceLinear, random_orthogonal_bases
from models.drs_peft.trajectory import BatteryTSFM, CycleQueryHead
from scripts.drs_peft.train import load_selected, selected_state

torch.set_num_threads(2)


def layer_fixture(index=0):
    torch.manual_seed(100 + index)
    u, v = random_orthogonal_bases(384, 384, 40, seed=100 + index)
    source = SubspaceLinear(
        nn.Linear(384, 384), u[:, :4], v[:, :4], core=torch.randn(4, 4) * 0.03
    )
    pair = index // 2
    rank = MATCHED_RANKS[pair]
    left, right = MATCHED_PERP_SHAPES[pair]
    bases = {
        "fixed_u": u[:, :rank],
        "fixed_v": v[:, :rank],
        "perp_u": u[:, 4 : 4 + left],
        "perp_v": v[:, 4 : 4 + right],
        "lora_a": v[:, 0],
    }
    return source, bases


class ToyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.module = nn.Module()
        self.input_projection = nn.Linear(7, 384)
        self.fusion = nn.Linear(384, 64)
        self.probe_bases = {}
        for index in range(12):
            layer, bases = layer_fixture(index)
            name = f"layer{index}"
            self.backbone.module.add_module(name, layer)
            self.probe_bases[name] = bases

    def encode(self, x, point_mask, cycle_ids, policy, policy_mask, anchor):
        values = self.input_projection(torch.cat((x.mean((1, 2)), policy), dim=-1))
        encoded = sum(layer(values) for layer in self.backbone.module.children())
        return self.fusion(encoded)

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


def early_batch():
    return {
        "x": torch.randn(2, 5, 3, 4),
        "point_mask": torch.ones(2, 5, 3, dtype=torch.bool),
        "cycle_ids": torch.arange(1, 6).repeat(2, 1),
        "policy": torch.ones(2, 3),
        "policy_mask": torch.ones(2, 3, dtype=torch.bool),
        "anchor": torch.ones(2),
    }


@pytest.mark.parametrize("arm", ARMS)
def test_source_prediction_is_bitwise_equal_with_registered_active_budget(arm):
    source = static_model().eval()
    reference = copy.deepcopy(source)
    model = SubspaceProbeTSFM(source, arm, source.bridge.probe_bases).eval()
    batch, query = early_batch(), torch.tensor([100.0, 500.0, 1000.0])
    assert torch.equal(model(batch, query), reference(batch, query))
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        == EXPECTED_PARAMETERS[arm]
    )
    assert all(not parameter.requires_grad for parameter in model.head.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.bridge.fusion.parameters()
    )
    assert all(
        value["zero_increment"].all()
        for value in model.update_diagnostics(batch).values()
    )


@pytest.mark.parametrize("arm", ARMS)
def test_actual_weight_increment_respects_common_radius(arm):
    source, bases = layer_fixture()
    layer = SubspaceProbeLinear(source, arm, bases, reference_fraction=0.01).eval()
    with torch.no_grad():
        layer.delta_core.fill_(2.0)
        if layer.core_perp is not None:
            layer.core_perp.fill_(3.0)
    expected = 0.01 * source.base.weight.double().norm()
    actual = layer.increment_matrix().norm()
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=2e-6)
    layer.evaluation_mode = "matched_norm"
    torch.testing.assert_close(
        layer.increment_matrix().norm(), expected, atol=1e-8, rtol=2e-6
    )


def test_residual_increment_is_orthogonal_to_original_fixed_space():
    source, bases = layer_fixture()
    layer = SubspaceProbeLinear(source, "matched_residual", bases).eval()
    with torch.no_grad():
        layer.core_perp.normal_()
    matrix = layer.increment_matrix()
    projection = source.U.double().T @ matrix @ source.V.double()
    assert projection.abs().max() < 1e-12
    assert matrix.norm() > 0


def test_gauge_fixed_lora_has_767_parameters_per_matrix_and_rank_one_output():
    source, bases = layer_fixture()
    layer = SubspaceProbeLinear(source, "matched_lora", bases).eval()
    assert (
        sum(
            parameter.numel()
            for parameter in layer.parameters()
            if parameter.requires_grad
        )
        == 767
    )
    assert layer.lora_a()[layer.core_anchor] == 1
    with torch.no_grad():
        layer.delta_core.normal_()
    singular = torch.linalg.svdvals(layer.increment_matrix())
    assert singular[0] > 0
    assert singular[1] < singular[0] * 1e-12


@pytest.mark.parametrize("arm", ARMS)
def test_zero_increment_has_finite_first_backward_and_lora_A_starts_zero_sensitive(arm):
    source, bases = layer_fixture()
    layer = SubspaceProbeLinear(source, arm, bases)
    layer(torch.randn(2, 3, 384)).square().mean().backward()
    for parameter in layer.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert layer.delta_core.grad.abs().sum() > 0
    if arm == "matched_lora":
        assert layer.core_a_free.grad.eq(0).all()


@pytest.mark.parametrize("arm", ARMS)
def test_selected_state_replay_preserves_parameters_and_all_controls(arm):
    source = static_model()
    model = SubspaceProbeTSFM(source, arm, source.bridge.probe_bases).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.001)
    saved = selected_state(model)
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }.issubset(saved)
    source_again = static_model()
    restored = SubspaceProbeTSFM(
        source_again, arm, source_again.bridge.probe_bases
    ).eval()
    parameters = [
        parameter for parameter in restored.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters)
    identities = [id(parameter) for parameter in parameters]
    load_selected(restored, saved)
    assert identities == [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    batch, query = early_batch(), torch.tensor([101.0, 500.0])
    assert torch.equal(model(batch, query), restored(batch, query))


def test_matched_norm_zero_increment_is_reported_and_training_rejected():
    source, bases = layer_fixture()
    layer = SubspaceProbeLinear(source, "fixed_rank4", bases).eval()
    layer.evaluation_mode = "matched_norm"
    assert layer.update_diagnostics(1)["matched_norm_unattainable"].all()
    layer.train()
    with pytest.raises(RuntimeError, match="model.eval"):
        layer(torch.randn(2, 384))
