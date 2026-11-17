"""Regressions for zero-start, no-policy and few-shot appended updates."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from models.drs_peft.onesided import (
    GEOMETRIC_ARMS,
    CalibrationBatteryTSFM,
    DictionaryBatteryTSFM,
    DictionaryLinear,
    LowRankBatteryTSFM,
    NoPolicyBatteryTSFM,
    calibration_jacobian,
    coordinate_context,
    dictionary_jacobian,
)
from models.drs_peft.peft import SubspaceLinear, random_orthogonal_bases
from models.drs_peft.trajectory import BatteryTSFM, CycleQueryHead
from scripts.drs_peft.onesided_calibration import (
    actual_local_predictions,
    cache_local_design,
    observed_positions,
    ridge_solve,
    train_lowrank_support,
)
from scripts.drs_peft.train import load_selected, selected_state


class TinyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.module = nn.Sequential()
        self.policy_dim = 4
        for index in range(3):
            left, right = random_orthogonal_bases(8, 8, 2, seed=17 + index)
            layer = SubspaceLinear(
                nn.Linear(8, 8), left, right, torch.randn(2, 2) * 0.03
            )
            self.backbone.module.add_module(f"layer{index}", layer)
        self.fusion = nn.Linear(8, 64)

    def encode(self, x, point_mask, cycle_ids, policy, policy_mask, anchor):
        values = torch.cat((x.mean((1, 2)), torch.where(policy_mask, policy, 0)), dim=1)
        for layer in self.backbone.module:
            values = torch.tanh(layer(values))
        return self.fusion(values)

    def parameter_budget(self):
        return {}


def model_fixture():
    torch.manual_seed(42)
    source = BatteryTSFM.__new__(BatteryTSFM)
    nn.Module.__init__(source)
    source.bridge = TinyBridge()
    source.head = CycleQueryHead()
    source.bridge_kind = "native_multivariate"
    source.official_backbone_identity = {"toy": True}
    return NoPolicyBatteryTSFM(source)


def inputs():
    torch.manual_seed(9)
    return {
        "x": torch.randn(1, 5, 3, 4),
        "point_mask": torch.ones(1, 5, 3, dtype=torch.bool),
        "cycle_ids": torch.arange(1, 6)[None],
        "policy": torch.randn(1, 4),
        "policy_mask": torch.ones(1, 4, dtype=torch.bool),
        "anchor": torch.tensor([0.97]),
    }


def dictionary(source, arm, q=2):
    generator = torch.Generator().manual_seed(34)
    layers = {}
    for name, layer in source.backbone.named_modules():
        if not isinstance(layer, SubspaceLinear):
            continue
        shape = {"CORE": (2, 2), "OUT": (8, 2), "IN": (2, 8), "NORMAL": (8, 8)}[arm]
        specification = {
            "kind": arm,
            "directions": torch.randn(q, *shape, generator=generator),
        }
        if arm in ("CORE", "IN"):
            specification["left"] = layer.U
        if arm in ("CORE", "OUT"):
            specification["right"] = layer.V
        layers[name] = specification
    return {"dimension": q, "layers": layers}


def test_policy_values_and_masks_are_erased_before_source_encoding():
    model = model_fixture().eval()
    early = inputs()
    query = torch.tensor([100.0, 200.0, 1000.0])
    first = model(early, query)
    changed = dict(
        early,
        policy=torch.full_like(early["policy"], float("nan")),
        policy_mask=torch.zeros_like(early["policy_mask"]),
    )
    assert torch.equal(first, model(changed, query))
    altered_anchor = dict(early, anchor=early["anchor"] - 0.02)
    assert not torch.equal(first, model(altered_anchor, query))


@pytest.mark.parametrize("arm", GEOMETRIC_ARMS)
def test_dictionary_uses_one_shared_vector_and_exact_zero_function(arm):
    source = model_fixture().eval()
    reference = copy.deepcopy(source)
    model = DictionaryBatteryTSFM(source, dictionary(source, arm)).eval()
    early, query = inputs(), torch.tensor([100.0, 200.0, 1000.0])
    assert torch.equal(model(early, query), reference(early, query))
    assert (
        sum(value.numel() for value in model.parameters() if value.requires_grad) == 2
    )
    assert [
        name for name, value in model.named_parameters() if value.requires_grad
    ] == ["bridge.backbone.module.core_coordinates"]


@pytest.mark.parametrize("arm", GEOMETRIC_ARMS)
def test_factor_forward_equals_dense_increment_and_merge(arm):
    source = model_fixture().eval()
    model = DictionaryBatteryTSFM(source, dictionary(source, arm)).eval()
    with torch.no_grad():
        model.coordinates.copy_(torch.tensor([0.003, -0.002]))
    for layer in model.backbone.modules():
        if isinstance(layer, DictionaryLinear):
            x = torch.randn(1, 7, 8)
            with coordinate_context(model.coordinates):
                expected = layer.source(x) + nn.functional.linear(
                    x, layer.increment_matrix(model.coordinates)
                )
                torch.testing.assert_close(layer(x), expected)
    early, query = inputs(), torch.tensor([100.0, 200.0, 1000.0])
    merged = model.merged_copy()
    torch.testing.assert_close(
        model(early, query), merged(early, query), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("arm", GEOMETRIC_ARMS)
def test_jvp_chain_matches_direct_derivative_and_finite_difference(arm):
    source = model_fixture().double().eval()
    model = DictionaryBatteryTSFM(source, dictionary(source, arm)).double().eval()
    early = {
        name: value.double() if value.is_floating_point() else value
        for name, value in inputs().items()
    }
    query = torch.tensor([100.0, 200.0, 1000.0], dtype=torch.float64)
    prediction, jacobian = dictionary_jacobian(model, early, query)
    for index in range(2):
        step = torch.zeros_like(model.coordinates)
        step[index] = 1e-4
        difference = (
            model(early, query, coordinates=step)
            - model(early, query, coordinates=-step)
        )[0] / 2e-4
        torch.testing.assert_close(jacobian[:, index], difference, rtol=1e-5, atol=1e-8)
    assert torch.equal(prediction, model(early, query)[0])
    assert jacobian[0].abs().max() < 1e-12


@pytest.mark.parametrize("method", ["LoRA", "LoRA-FA"])
def test_real_low_rank_branches_retain_source_and_match_trainable_budget(method):
    source = model_fixture().eval()
    reference = copy.deepcopy(source)
    model = LowRankBatteryTSFM(source, method).eval()
    early, query = inputs(), torch.tensor([100.0, 200.0, 1000.0])
    assert torch.equal(model(early, query), reference(early, query))
    assert (
        sum(value.numel() for value in model.parameters() if value.requires_grad) == 48
    )
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=0.01
    )
    for _ in range(2):
        optimizer.zero_grad()
        model(early, query).sum().backward()
        optimizer.step()
    assert not torch.equal(model(early, query), reference(early, query))
    torch.testing.assert_close(
        model(early, query), model.merged_copy()(early, query), rtol=1e-6, atol=1e-7
    )


def test_cal2_preserves_zero_start_anchor_and_exact_design():
    source = model_fixture().eval()
    reference = copy.deepcopy(source)
    model = CalibrationBatteryTSFM(source).eval()
    early, query = inputs(), torch.tensor([100.0, 200.0, 1000.0])
    assert torch.equal(model(early, query), reference(early, query))
    prediction, jacobian = calibration_jacobian(model, early, query)
    with torch.no_grad():
        model.coordinates.copy_(torch.tensor([0.5, -0.01]))
    torch.testing.assert_close(
        model(early, query)[0], prediction + jacobian @ model.coordinates
    )
    assert torch.equal(model(early, query)[:, 0], early["anchor"])


@pytest.mark.parametrize(
    "method", ["CORE", "OUT", "IN", "NORMAL", "LoRA", "LoRA-FA", "CAL2"]
)
def test_selected_state_reconstructs_all_appended_parameters(method):
    def construct():
        source = model_fixture()
        if method in GEOMETRIC_ARMS:
            return DictionaryBatteryTSFM(source, dictionary(source, method))
        if method == "CAL2":
            return CalibrationBatteryTSFM(source)
        return LowRankBatteryTSFM(source, method)

    model = construct().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.002)
    reconstructed = construct().eval()
    load_selected(reconstructed, selected_state(model))
    early, query = inputs(), torch.tensor([100.0, 200.0, 1000.0])
    assert torch.equal(model(early, query), reconstructed(early, query))


def test_observed_sampling_preserves_physical_support_and_boundaries():
    mask = np.ones(4900, dtype=bool)
    mask[200:500] = False
    positions = observed_positions(mask)
    assert len(positions) == 512
    assert positions[0] == 0 and positions[-1] == 4899
    assert len(np.unique(positions)) == 512
    assert mask[positions].all()
    np.testing.assert_array_equal(positions, observed_positions(mask))
    np.testing.assert_array_equal(observed_positions(mask[:80]), np.arange(80))
    with pytest.raises(ValueError):
        observed_positions(np.zeros(30, dtype=bool))


def test_ridge_exact_cell_weighting_and_dimensionless_penalty():
    first = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    second = np.asarray([[2.0, 0.0]])
    errors = [np.asarray([1.0, 2.0]), np.asarray([4.0])]
    result = ridge_solve([first, second], errors, 0.01, eta=0.5)
    gram = (first.T @ first / 2 + second.T @ second) / 2
    rhs = (first.T @ errors[0] / 2 + second.T @ errors[1]) / 2
    ridge = 0.01 * np.trace(gram) / 2
    expected = np.linalg.solve(gram + ridge * np.eye(2), rhs)
    np.testing.assert_allclose(result["gram"], gram, atol=1e-15)
    np.testing.assert_allclose(result["rhs"], rhs, atol=1e-15)
    np.testing.assert_allclose(result["a"], 0.5 * expected, atol=1e-15)
    assert result["lambda_actual"] == ridge
    replicated = ridge_solve(
        [np.repeat(first, 100, axis=0), second],
        [np.repeat(errors[0], 100), errors[1]],
        0.01,
        eta=0.5,
    )
    np.testing.assert_allclose(replicated["a"], result["a"], atol=1e-15)


def test_zero_design_has_finite_registered_floor_and_zero_solution():
    result = ridge_solve([np.zeros((3, 4))], [np.ones(3)], 1e-4)
    np.testing.assert_array_equal(result["a"], np.zeros(4))
    assert result["lambda_actual"] == 1e-16
    for invalid in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(ValueError):
            ridge_solve([np.zeros((3, 4))], [np.ones(3)], invalid)


@pytest.mark.parametrize("method", ["LoRA", "LoRA_FA"])
def test_nonlinear_support_steps_and_whole_increment_shrinkage(method):
    source = model_fixture().eval()
    reference = copy.deepcopy(source)
    model = LowRankBatteryTSFM(source, method)
    early, query = inputs(), torch.tensor([100.0, 200.0, 1000.0])
    with torch.no_grad():
        base = reference(early, query)
    target = base[0] + (query - 100) / 1000 * 0.02
    result = train_lowrank_support(
        model,
        [{"early": early, "query": query, "target": target}],
        0.01,
        save_steps=(1, 2),
    )
    assert result["optimizer_updates"] == 2
    assert set(result["snapshots"]) == {1, 2}
    assert all(value["weight_decay"] == 0 for value in result["snapshots"].values())
    model.set_adaptation_scale(0)
    assert torch.equal(model(early, query), base)
    model.set_adaptation_scale(0.5)
    half = {
        name: layer.increment_matrix().detach().clone()
        for name, layer in model.backbone.named_modules()
        if hasattr(layer, "lora_b")
    }
    model.set_adaptation_scale(1)
    for name, layer in model.backbone.named_modules():
        if name in half:
            torch.testing.assert_close(half[name], 0.5 * layer.increment_matrix())


def test_real_nonlinear_predictions_and_approximation_are_separately_retained():
    source = model_fixture().eval()
    model = DictionaryBatteryTSFM(source, dictionary(source, "CORE")).eval()
    early, query = inputs(), torch.tensor([100.0, 200.0, 1000.0])
    cache = cache_local_design(model, early, query)
    result = actual_local_predictions(
        model, early, query, np.asarray([0.1, -0.1]), cache
    )
    with torch.no_grad():
        actual = model(early, query, coordinates=torch.tensor([0.1, -0.1]))[0].numpy()
    np.testing.assert_array_equal(result["actual"], actual)
    torch.testing.assert_close(model.coordinates, torch.tensor([0.1, -0.1]))
    assert result["gap_MAE"] > 0
