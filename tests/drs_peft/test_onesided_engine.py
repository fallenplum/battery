"""CPU regressions for label-separated support fits and reusable prediction states."""

import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from models.drs_peft.onesided import NoPolicyBatteryTSFM
from models.drs_peft.peft import SubspaceLinear, random_orthogonal_bases
from models.drs_peft.trajectory import BatteryTSFM, CycleQueryHead
from scripts.drs_peft.gradient_audit import state_fingerprint
from scripts.drs_peft.onesided_engine import FewShotEngine


class SmallBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_dim = 0
        self.input_projection = nn.Linear(12, 8)
        self.backbone = nn.Module()
        self.backbone.module = nn.Sequential()
        for index in range(12):
            left, right = random_orthogonal_bases(8, 8, 2, seed=index + 20)
            self.backbone.module.add_module(
                "layer" + str(index),
                SubspaceLinear(nn.Linear(8, 8), left, right, torch.randn(2, 2) * 0.02),
            )
        self.fusion = nn.Linear(8, 64)

    def encode(self, x, point_mask, cycle_ids, policy, policy_mask, anchor):
        values = self.input_projection(x.mean((1, 2)))
        for layer in self.backbone.module:
            values = values + 0.1 * torch.tanh(layer(values))
        return self.fusion(values)

    def parameter_budget(self):
        return {}


def reference_and_inputs():
    torch.manual_seed(42)
    source = BatteryTSFM.__new__(BatteryTSFM)
    nn.Module.__init__(source)
    source.bridge = SmallBridge()
    source.head = CycleQueryHead()
    source.bridge_kind = "native_multivariate"
    source.official_backbone_identity = {"engineering_only": True}
    model = NoPolicyBatteryTSFM(source).eval()
    inputs = {
        "ids": ["cell-A", "cell-B", "cell-C"],
        "x": torch.randn(3, 5, 3, 12).numpy(),
        "point_mask": np.ones((3, 5, 3), dtype=bool),
        "cycle_ids": np.tile(np.arange(1, 6), (3, 1)),
        "policy": np.empty((3, 0), dtype=np.float32),
        "policy_mask": np.empty((3, 0), dtype=bool),
        "anchor": np.array([0.97, 0.96, 0.95], dtype=np.float32),
    }
    return model, inputs


def payload(model, arm="CORE"):
    generator = torch.Generator().manual_seed(77)
    layers = {}
    for name, layer in model.backbone.named_modules():
        if not isinstance(layer, SubspaceLinear):
            continue
        shape = {"CORE": (2, 2), "OUT": (8, 2), "IN": (2, 8), "NORMAL": (8, 8)}[arm]
        specification = {
            "kind": arm,
            "directions": torch.randn(2, *shape, generator=generator) * 0.2,
        }
        if arm in ("CORE", "IN"):
            specification["left"] = layer.U
        if arm in ("CORE", "OUT"):
            specification["right"] = layer.V
        layers[name] = specification
    return {"dimension": 2, "layers": layers}


def engine_fixture(tmp_path):
    model, inputs = reference_and_inputs()
    query = np.array([100, 200, 400, 700, 1000], dtype=np.float32)
    engine = FewShotEngine(
        model,
        inputs,
        query,
        {"reference_checkpoint_sha256": "engineering-checkpoint"},
        tmp_path,
        device="cpu",
    )
    for arm in ("CORE", "OUT", "IN", "NORMAL"):
        engine.register_method(arm, payload(model, arm))
    for name in ("CAL2", "LoRA", "LoRA_FA"):
        engine.register_method(name)
    return engine, model


def support_for(engine, ids=("cell-A", "cell-B")):
    output = []
    for uid in ids:
        early = engine._early([uid])
        positions = np.array([1, 3, 4])
        query = torch.as_tensor(engine.query[positions])
        with torch.no_grad():
            target = engine.f0(early, query)[0].numpy() - np.array(
                [0.005, 0.012, 0.02], dtype=np.float32
            )
        output.append(
            {
                "uid": uid,
                "early": early,
                "query": query,
                "target": target,
                "positions": positions,
            }
        )
    return output


@pytest.mark.parametrize("method", ("CORE", "OUT", "IN", "NORMAL", "CAL2"))
def test_geometric_actual_predictions_cache_and_roundtrip(tmp_path, method):
    engine, original = engine_fixture(tmp_path)
    before = state_fingerprint(original)
    zero = engine.baseline_prediction(["cell-C"])
    support = support_for(engine)
    state, details = engine.geometry_trial(
        method, support, {"lambda_multiplier": 1.0, "eta": 0.5}
    )
    json.dumps(details, allow_nan=False)
    assert details["dimension"] == 2
    assert state["source_checkpoint_sha256"] == "engineering-checkpoint"
    assert engine.counters["jacobian_builds"] == 2
    actual = engine.predict_state(method, state, ["cell-C"])
    assert actual.shape == zero.shape == (1, 5)
    assert np.array_equal(actual[:, 0], zero[:, 0])
    assert not np.array_equal(actual, zero)
    support_metrics = engine.support_diagnostics(method, state, support)
    json.dumps(support_metrics, allow_nan=False)
    assert support_metrics["support_mae_before"] > 0
    assert support_metrics["support_mae_after"] >= 0
    assert support_metrics["evaluation"] == "actual_nonlinear_cell_mean"
    diagnosis = engine.query_diagnostics(method, state, ["cell-C"], actual)
    json.dumps(diagnosis, allow_nan=False)
    assert diagnosis["query_labels_read"] == 0
    assert diagnosis["query_gap_MAE"] >= 0
    replay = engine.replay_and_merge(method, state, ["cell-C"])
    assert replay["serialized_state_replay_maximum_difference"] == 0
    assert replay["serialized_adaptation_state_bytes"] > 0
    assert (
        replay["merged_prediction_maximum_difference"] is None
        or replay["merged_prediction_maximum_difference"] < 1e-6
    )
    repeated, repeated_details = engine.geometry_trial(
        method, list(reversed(support)), {"lambda_multiplier": 1.0, "eta": 0.5}
    )
    np.testing.assert_array_equal(state["a"], repeated["a"])
    assert repeated_details == details
    assert engine.counters["geometry_cache_hits"] == 1
    assert state_fingerprint(original) == before
    assert np.array_equal(zero, engine.baseline_prediction(["cell-C"]))


def test_target_change_resolves_state_and_reuses_label_free_jacobian(tmp_path):
    engine, _ = engine_fixture(tmp_path)
    support = support_for(engine, ("cell-A",))
    first, _ = engine.geometry_trial("CAL2", support, {"lambda_multiplier": 1})
    support[0]["target"] = support[0]["target"] - 0.01
    second, _ = engine.geometry_trial("CAL2", support, {"lambda_multiplier": 1})
    assert engine.counters["jacobian_builds"] == 1
    assert engine.counters["jacobian_cache_hits"] == 1
    assert engine.counters["geometry_solves"] == 2
    assert not np.array_equal(first["a"], second["a"])
    for path in (tmp_path / "jacobians").glob("*.json"):
        assert "target" not in path.read_text()


@pytest.mark.parametrize("method", ("LoRA", "LoRA_FA"))
def test_nonlinear_fresh_start_shared_steps_shrink_and_cache(tmp_path, method):
    engine, original = engine_fixture(tmp_path)
    before = state_fingerprint(original)
    zero = engine.baseline_prediction(["cell-C"])
    support = support_for(engine)
    result = engine.nonlinear_trials(method, support, 1e-3, steps=(1, 2))
    assert set(result) == {1, 2}
    assert engine.counters["nonlinear_optimizer_updates"] == 2
    json.dumps(result[2]["details"], allow_nan=False)
    state = copy.deepcopy(result[2]["state"])
    actual = engine.predict_state(method, state, ["cell-C"])
    assert not np.array_equal(actual, zero)
    state["adaptation_scale"] = 0.0
    np.testing.assert_array_equal(engine.predict_state(method, state, ["cell-C"]), zero)
    support_metrics = engine.support_diagnostics(method, state, support)
    json.dumps(support_metrics, allow_nan=False)
    assert support_metrics["support_mae_before"] == support_metrics["support_mae_after"]
    assert support_metrics["support_mse_before"] == support_metrics["support_mse_after"]
    state["adaptation_scale"] = 0.5
    replay = engine.replay_and_merge(method, state, ["cell-C"])
    assert replay["serialized_state_replay_maximum_difference"] == 0
    assert replay["merged_prediction_maximum_difference"] < 1e-6
    repeated = engine.nonlinear_trials(
        method, list(reversed(support)), 1e-3, steps=(1, 2)
    )
    assert engine.counters["nonlinear_optimizer_updates"] == 2
    assert engine.counters["nonlinear_cache_hits"] == 1
    assert repeated[2]["details"] == result[2]["details"]
    assert state_fingerprint(original) == before
    assert np.array_equal(zero, engine.baseline_prediction(["cell-C"]))


def test_prefix_tampering_and_state_identity_changes_are_rejected(tmp_path):
    engine, _ = engine_fixture(tmp_path)
    support = support_for(engine, ("cell-A",))
    support[0]["early"]["x"] = support[0]["early"]["x"] + 0.01
    with pytest.raises(RuntimeError, match="prefix differs"):
        engine.geometry_trial("CAL2", support, {"lambda_multiplier": 1})
    state, _ = engine.geometry_trial(
        "CAL2", support_for(engine, ("cell-A",)), {"lambda_multiplier": 1}
    )
    state["source_checkpoint_sha256"] = "another-source"
    with pytest.raises(RuntimeError, match="another source"):
        engine.predict_state("CAL2", state, ["cell-C"])


def test_corrupted_jacobian_cache_fails_closed(tmp_path):
    engine, _ = engine_fixture(tmp_path)
    support = support_for(engine, ("cell-A",))
    engine.geometry_trial("CAL2", support, {"lambda_multiplier": 1})
    array_path = next((tmp_path / "jacobians").glob("*.npz"))
    array_path.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="Jacobian identity changed"):
        engine.geometry_trial("CAL2", support, {"lambda_multiplier": 2})


def test_support_grid_and_duplicate_physical_ids_are_rejected(tmp_path):
    engine, _ = engine_fixture(tmp_path)
    support = support_for(engine, ("cell-A",))
    with pytest.raises(ValueError, match="distinct"):
        engine.geometry_trial("CAL2", support * 2, {"lambda_multiplier": 1})
    support[0]["positions"] = np.array([1, 3, 99])
    with pytest.raises(ValueError, match="fixed-grid positions"):
        engine.geometry_trial("CAL2", support, {"lambda_multiplier": 1})
