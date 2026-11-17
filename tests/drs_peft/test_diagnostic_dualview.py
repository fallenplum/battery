"""Regression checks for the new source-only dual-view diagnostic."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from models.drs_peft.diagnostic_dualview import (
    CoreGates,
    DualViewDiagnosticTSFM,
    SourceViewScaler,
    prefix_view_descriptors,
)
from models.drs_peft.peft import SubspaceLinear, random_orthogonal_bases
from models.drs_peft.trajectory import BatteryTSFM, CycleQueryHead
from scripts.drs_peft.diagnostic_dualview import inputs_for_mode
from scripts.drs_peft.train import load_selected, selected_state


def synthetic_prefix(batch=4):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(batch, 100, 64, 4)).astype(np.float32)
    mask = np.ones(x.shape[:-1], dtype=bool)
    cycle = np.repeat(np.arange(1, 101, dtype=np.float32)[None], batch, axis=0)
    return x, mask, cycle


def test_descriptors_ignore_fourth_channel_and_masked_values():
    x, mask, cycles = synthetic_prefix()
    mask[:, 2] = False
    expected = prefix_view_descriptors(x, mask, cycles)
    x[..., 3] = 1e25
    x[:, 2] = np.nan
    actual = prefix_view_descriptors(x, mask, cycles)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert actual["absolute"].shape == actual["relative"].shape == (4, 12)


def test_relative_view_invariant_to_per_cell_constant_channel_offsets():
    x, mask, cycles = synthetic_prefix()
    x = x.astype(np.float64)
    before = prefix_view_descriptors(x, mask, cycles)
    x[..., :3] += np.arange(12, dtype=np.float64).reshape(4, 1, 1, 3)
    after = prefix_view_descriptors(x, mask, cycles)
    np.testing.assert_allclose(
        after["relative"], before["relative"], rtol=1e-6, atol=1e-7
    )
    assert not np.array_equal(after["absolute"], before["absolute"])


def test_reference_support_mask_and_prefix_guard():
    x, mask, cycles = synthetic_prefix()
    mask[:, :7] = False
    descriptors = prefix_view_descriptors(x, mask, cycles)
    assert not descriptors["relative_mask"].any()
    assert (descriptors["relative"] == 0).all()
    cycles[0, -1] = 101
    with pytest.raises(ValueError, match="prefix cycle"):
        prefix_view_descriptors(x, mask, cycles)


def test_scaler_uses_source_rows_only():
    x, mask, cycles = synthetic_prefix()
    descriptors = prefix_view_descriptors(x, mask, cycles)
    source = SourceViewScaler().fit(descriptors, [0, 1], ["s0", "s1"]).to_dict()
    descriptors["absolute"][2:] = 1e20
    descriptors["relative"][2:] = -1e20
    repeated = SourceViewScaler().fit(descriptors, [0, 1], ["s0", "s1"]).to_dict()
    assert repeated == source
    assert json.loads(json.dumps(source)) == source


class ToyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.module = nn.Module()
        for index in range(12):
            u, v = random_orthogonal_bases(7, 7, 4, index)
            self.backbone.module.add_module(
                f"layer{index}", SubspaceLinear(nn.Linear(7, 7), u, v)
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


def toy_model(mode=None):
    torch.manual_seed(42)
    model = BatteryTSFM.__new__(BatteryTSFM)
    nn.Module.__init__(model)
    model.bridge = ToyBridge()
    model.head = CycleQueryHead()
    model.bridge_kind = "toy"
    model.official_backbone_identity = {"toy": True}
    return DualViewDiagnosticTSFM(model, mode) if mode else model


def toy_batch(batch=3):
    x, mask, cycles = synthetic_prefix(batch)
    return {
        "x": torch.tensor(x),
        "point_mask": torch.tensor(mask),
        "cycle_ids": torch.tensor(cycles),
        "policy": torch.ones(batch, 3),
        "policy_mask": torch.ones(batch, 3, dtype=torch.bool),
        "anchor": torch.ones(batch),
        "diagnostic_absolute": torch.randn(batch, 12),
        "diagnostic_relative": torch.randn(batch, 12),
    }


@pytest.mark.parametrize(
    "mode", ["constant", "absolute", "relative", "dual", "shuffled_dual"]
)
def test_initial_identity_parameter_match_and_selected_roundtrip(mode):
    expected = toy_model()
    actual = toy_model(mode)
    batch = toy_batch()
    query = torch.tensor([100.0, 500.0, 5000.0])
    torch.testing.assert_close(
        actual(batch, query), expected(batch, query), rtol=0, atol=0
    )
    assert (
        sum(value.numel() for value in actual.backbone.core_gates.parameters()) == 1248
    )
    state = selected_state(actual)
    assert sum(".core_gates." in key for key in state) == 48
    for key in actual.state_dict():
        if ".core_gates." in key:
            assert key in state
    restored = toy_model(mode)
    load_selected(restored, state)
    torch.testing.assert_close(
        restored(batch, query), actual(batch, query), rtol=0, atol=0
    )


def test_constant_arm_has_effective_weight_gradients_and_remains_static():
    model = toy_model("constant")
    for layer in model.backbone.modules():
        if isinstance(layer, SubspaceLinear):
            with torch.no_grad():
                layer.core.copy_(torch.eye(4) * 0.1)
    batch = toy_batch()
    query = torch.tensor([500.0, 1000.0])
    output = model(batch, query)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.backbone.core_gates.parameters()
    )
    other = copy.deepcopy(batch)
    other["diagnostic_absolute"].mul_(1e6)
    other["diagnostic_relative"].add_(1e6)
    torch.testing.assert_close(model(other, query), output, rtol=0, atol=0)


def test_role_local_shuffle_is_bijective_and_target_features_untouched():
    data = SimpleNamespace(ids=["s0", "s1", "s2", "m0", "m1", "t0"])
    data.rows = lambda ids: np.array([data.ids.index(uid) for uid in ids])
    episodes = {
        "source_cell_roles": {
            "source_train": data.ids[:3],
            "source_meta_validation": data.ids[3:5],
        }
    }
    arrays = {
        view: np.arange(72, dtype=np.float32).reshape(6, 12)
        for view in ("absolute", "relative")
    }
    result, donors = inputs_for_mode({}, arrays, episodes, data, "shuffled_dual", 42)
    for role in episodes["source_cell_roles"]:
        assert set(donors[role]) == set(donors[role].values())
        assert all(uid != donor for uid, donor in donors[role].items())
    np.testing.assert_array_equal(
        result["diagnostic_absolute"][-1], arrays["absolute"][-1]
    )


def test_gates_are_bounded_and_zero_initialization_exact():
    bank = CoreGates(["first", "second"])
    values = torch.randn(3, 12)
    absolute, relative = bank(values, values)
    assert all(
        torch.equal(gate, torch.ones_like(gate))
        for gate in [*absolute.values(), *relative.values()]
    )
    with torch.no_grad():
        for parameter in bank.parameters():
            parameter.normal_(std=10)
    absolute, relative = bank(values, values)
    assert all(
        bool(((gate >= 0.5) & (gate <= 1.5)).all())
        for gate in [*absolute.values(), *relative.values()]
    )
