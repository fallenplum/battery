"""Zero-start singular-value-scaled LoRA-XS and warmup-reference regressions."""

import copy
import io
import json

import numpy as np
import pytest
import torch
from torch import nn

from models.drs_peft.onesided import NoPolicyBatteryTSFM, replace_layer, source_layers
from models.drs_peft.peft import SubspaceLinear, random_orthogonal_bases
from models.drs_peft.specialization import XSCalibrationModel, XSLinear
from models.drs_peft.trajectory import BatteryTSFM, CycleQueryHead
from scripts.drs_peft import specialization_engine as specialization
from scripts.drs_peft.data import file_hash
from scripts.drs_peft.gradient_audit import state_fingerprint
from scripts.drs_peft.train import load_selected, selected_state


class TinyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_dim = 0
        self.backbone = nn.Module()
        self.backbone.module = nn.Sequential()
        for index in range(12):
            left, right = random_orthogonal_bases(8, 8, 4, seed=index + 42)
            self.backbone.module.add_module(
                "layer" + str(index),
                SubspaceLinear(nn.Linear(8, 8), left, right, torch.randn(4, 4) * 0.2),
            )
        self.fusion = nn.Linear(8, 64)

    def encode(self, x, point_mask, cycle_ids, policy, policy_mask, anchor):
        values = x.mean((1, 2))
        for layer in self.backbone.module:
            values = values + 0.1 * torch.tanh(layer(values))
        return self.fusion(values)

    def parameter_budget(self):
        return {}


def source_fixture(bare=False):
    torch.manual_seed(42)
    model = BatteryTSFM.__new__(BatteryTSFM)
    nn.Module.__init__(model)
    model.bridge = TinyBridge()
    model.head = CycleQueryHead()
    model.bridge_kind = "native_multivariate"
    model.official_backbone_identity = {"engineering_only": True}
    model = NoPolicyBatteryTSFM(model).eval()
    if bare:
        for name, layer in source_layers(model):
            replace_layer(model.backbone, name, layer.base)
    return model


def early_fixture():
    torch.manual_seed(32)
    return {
        "x": torch.randn(1, 5, 3, 8),
        "point_mask": torch.ones(1, 5, 3, dtype=torch.bool),
        "cycle_ids": torch.arange(1, 6)[None],
        "anchor": torch.tensor([0.97]),
        "policy": torch.empty((1, 0)),
        "policy_mask": torch.empty((1, 0), dtype=torch.bool),
    }


def test_payload_uses_complete_effective_source_weight_and_cpu_double_svd():
    source = source_fixture()
    payload, construction = specialization.build_xs_payload(source)
    assert construction["new_SVD_evaluations"] == 12
    assert payload["source_state_sha256"] == state_fingerprint(source)
    for name, layer in source_layers(source):
        factor = payload["layers"][name]
        expected = (
            layer.base.weight.double()
            + layer.scale * layer.U.double() @ layer.core.double() @ layer.V.double().T
        )
        singular = torch.linalg.svdvals(expected)[:4]
        torch.testing.assert_close(factor["singular"], singular)
        assert not torch.allclose(
            factor["singular"], torch.linalg.svdvals(layer.base.weight.double())[:4]
        )
        assert factor["U"].dtype == torch.float64
        assert factor["U"].device.type == "cpu"
        torch.testing.assert_close(
            factor["U"].T @ factor["U"], torch.eye(4, dtype=torch.float64)
        )


def test_xs_zero_start_parameter_budget_and_singular_forward_merge_and_save():
    source = source_fixture()
    initial = state_fingerprint(source)
    payload, _ = specialization.build_xs_payload(source)
    model = XSCalibrationModel(copy.deepcopy(source), payload).eval()
    early, query = early_fixture(), torch.tensor([100.0, 200.0, 500.0, 1000.0])
    assert torch.equal(model(early, query), source(early, query))
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    assert len(names) == 12
    assert all(name.endswith(".core_xs_delta") for name in names)
    assert (
        sum(value.numel() for value in model.parameters() if value.requires_grad) == 192
    )
    assert model.parameter_report()["singular_values_retained"] is True
    with torch.no_grad():
        for layer in model.backbone.modules():
            if isinstance(layer, XSLinear):
                layer.core_xs_delta.copy_(torch.randn_like(layer.core_xs_delta) * 0.01)
                x = torch.randn(3, 7, 8)
                torch.testing.assert_close(
                    layer(x),
                    layer.source(x) + nn.functional.linear(x, layer.increment_matrix()),
                    atol=2e-7,
                    rtol=1e-6,
                )
                without_sigma = (
                    layer.core_xs_U @ layer.core_xs_delta @ layer.core_xs_V.T
                )
                assert not torch.allclose(layer.increment_matrix(), without_sigma)
    model.set_adaptation_scale(0.5)
    expected = model(early, query)
    serialized = io.BytesIO()
    torch.save(selected_state(model), serialized)
    serialized.seek(0)
    loaded = XSCalibrationModel(copy.deepcopy(source), payload).eval()
    load_selected(loaded, torch.load(serialized, weights_only=False))
    assert torch.equal(expected, loaded(early, query))
    merged = model.merged_copy()
    torch.testing.assert_close(expected, merged(early, query), atol=1e-7, rtol=1e-6)
    model.set_adaptation_scale(0)
    assert torch.equal(model(early, query), source(early, query))
    assert state_fingerprint(source) == initial


def test_xs_svd_cache_constructs_once_and_rejects_another_source(tmp_path, monkeypatch):
    source = source_fixture()
    path = tmp_path / "XS.pt"
    first, construction = specialization.cache_xs_payload(source, path)
    assert construction["new_SVD_evaluations"] == 12
    monkeypatch.setattr(
        torch.linalg,
        "svd",
        lambda *args, **kwargs: pytest.fail("A cached reference must not repeat SVD"),
    )
    second, reused = specialization.cache_xs_payload(source, path)
    assert first["payload_sha256"] == second["payload_sha256"]
    assert reused["new_SVD_evaluations"] == 0
    assert reused["cache_reused"] is True
    XSCalibrationModel(copy.deepcopy(source), second)
    with torch.no_grad():
        source.head.net[0].weight.add_(0.01)
    with pytest.raises(RuntimeError, match="another source"):
        specialization.cache_xs_payload(source, path)


def support_fixture(source):
    query = torch.tensor([200.0, 400.0, 700.0, 1000.0])
    early = early_fixture()
    with torch.no_grad():
        baseline = source(early, query)[0]
    return [
        {"early": early, "query": query[:2], "target": baseline[:2].numpy() - 0.01},
        {"early": early, "query": query, "target": baseline.numpy() - 0.03},
    ]


def test_xs_training_uses_exact_cellmean_objective_and_only_delta_updates():
    source = source_fixture()
    payload, _ = specialization.build_xs_payload(source)
    model = XSCalibrationModel(copy.deepcopy(source), payload)
    manual = copy.deepcopy(model)
    support = support_fixture(source)
    optimizer = torch.optim.AdamW(
        [p for p in manual.parameters() if p.requires_grad], lr=1e-3, weight_decay=0
    )
    loss = torch.stack(
        [
            (manual(cell["early"], cell["query"])[0] - torch.as_tensor(cell["target"]))
            .square()
            .mean()
            for cell in support
        ]
    ).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in manual.parameters() if p.requires_grad], 1
    )
    optimizer.step()
    frozen = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }
    result = specialization.train_xs_support(model, support, 1e-3, save_steps=(1,))
    assert result["history"][0]["cell_mean_MSE_before_update"] == pytest.approx(
        float(loss)
    )
    assert result["optimizer_updates"] == 1
    for (name, value), (_, expected) in zip(
        model.named_parameters(), manual.named_parameters(), strict=True
    ):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


def test_pretrained_loader_zero_core_matches_bare_warmup(tmp_path, monkeypatch):
    folder = tmp_path
    (folder / "bases").mkdir()
    torch.save({"bases": {}}, folder / "bases/SOURCE_BASES.pt")
    checkpoint = folder / "WARMUP.pt"
    checkpoint.write_bytes(b"warmup")
    warmup = {"model": selected_state(source_fixture(bare=True))}
    monkeypatch.setattr(specialization, "reference_folder", lambda *args: folder)
    monkeypatch.setattr(
        specialization, "checked_checkpoint", lambda *args: (warmup, checkpoint)
    )
    monkeypatch.setattr(
        specialization,
        "read_json",
        lambda *args: {"warmup_checkpoint_sha256": file_hash(checkpoint)},
    )
    monkeypatch.setattr(
        specialization, "static_factory", lambda *args: lambda: source_fixture()
    )
    monkeypatch.setattr(
        specialization,
        "warmup_factory",
        lambda *args: lambda: source_fixture(bare=True),
    )
    model = specialization.load_pretrained_reference(tmp_path, "official", 0, 42)
    assert all(
        torch.count_nonzero(layer.core) == 0 for _, layer in source_layers(model)
    )
    assert model.specialization_identity["source_checkpoint_sha256"] == file_hash(
        checkpoint
    )
    audit = specialization.audit_warmup_replay(
        model,
        tmp_path,
        "official",
        0,
        42,
        early_fixture(),
        torch.tensor([100.0, 200.0, 1000.0]),
    )
    assert audit["bit_exact"] is True
    assert audit["source_core_nonzero_entries"] == 0


def test_specialization_engine_caches_xs_trials_and_preserves_source(tmp_path):
    source = source_fixture()
    initial = state_fingerprint(source)
    payload, _ = specialization.build_xs_payload(source)
    early = early_fixture()
    inputs = {
        key: value.repeat((2,) + (1,) * (value.ndim - 1)).numpy()
        for key, value in early.items()
    }
    inputs["ids"] = ["A", "B"]
    query = np.array([100, 200, 400, 700, 1000], dtype=np.float32)
    engine = specialization.SpecializationEngine(
        source,
        inputs,
        query,
        {"source_checkpoint_sha256": "specialized"},
        tmp_path,
        device="cpu",
    )
    engine.register_method("LoRA-XS", payload)
    engine.register_method("LoRA")
    support = []
    for uid in inputs["ids"]:
        prefix = engine._early([uid])
        with torch.no_grad():
            target = source(prefix, torch.from_numpy(query[1:]))[0].numpy() - 0.01
        support.append(
            {
                "uid": uid,
                "early": prefix,
                "query": query[1:],
                "target": target,
                "positions": np.arange(1, 5),
            }
        )
    trials = engine.nonlinear_trials("LoRA-XS", support, 1e-3, steps=(1, 2))
    assert engine.counters["nonlinear_optimizer_updates"] == 2
    json.dumps(trials[2]["details"], allow_nan=False)
    state = copy.deepcopy(trials[2]["state"])
    state["adaptation_scale"] = 0
    np.testing.assert_array_equal(
        engine.predict_state("LoRA-XS", state, ["B"]), engine.baseline_prediction(["B"])
    )
    state["adaptation_scale"] = 0.5
    replay = engine.replay_and_merge("LoRA-XS", state, ["B"])
    assert replay["serialized_state_replay_maximum_difference"] == 0
    engine.nonlinear_trials("LoRA-XS", list(reversed(support)), 1e-3, steps=(1, 2))
    assert engine.counters["nonlinear_cache_hits"] == 1
    assert engine.counters["nonlinear_optimizer_updates"] == 2
    engine.nonlinear_trials("LoRA", support, 1e-3, steps=(1,))
    assert engine.counters["nonlinear_optimizer_updates"] == 3
    assert state_fingerprint(source) == initial
