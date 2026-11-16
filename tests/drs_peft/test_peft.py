"""Tensor-level engineering evidence, excluded from scientific metrics."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.drs_peft.peft import (
    LoRALinear,
    LoRAGALinear,
    LoRAXSLinear,
    SubspaceLinear,
    adaptation_parameters,
    begin_target_adaptation,
    install_peft,
    peft_context,
    random_orthogonal_bases,
    svd_bases,
    trainable_parameter_report,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(7, 5).double()
        self.v_proj = nn.Linear(7, 5).double()
        self.k_proj = nn.Linear(7, 5).double()
        self.head = nn.Linear(5, 1).double()

    def forward(self, x):
        return self.head(self.q_proj(x) + self.v_proj(x) + self.k_proj(x))


def test_standard_lora_initial_identity_counts_and_gradients():
    model = Toy()
    x = torch.randn(3, 4, 7, dtype=torch.float64)
    expected = model(x).detach()
    rng = torch.get_rng_state().clone()
    records = install_peft(model, "lora", rank=2, alpha=4, seed=42)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert len(records) == 2
    assert trainable_parameter_report(model)["trainable_parameters"] == 2 * 2 * (7 + 5)
    model(x).square().mean().backward()
    assert model.q_proj.lora_b.weight.grad.norm() > 0
    assert model.q_proj.base.weight.grad is None
    assert not model.head.weight.requires_grad
    assert not model.k_proj.weight.requires_grad


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 8])
def test_dynamic_core_matches_explicit_matrix_and_has_gate_gradients(
    device, batch_size
):
    base = nn.Linear(7, 5).double().to(device)
    u, v = random_orthogonal_bases(5, 7, 3, 19, dtype=torch.float64, device=device)
    layer = SubspaceLinear(
        base, u, v, core=torch.randn(3, 3, dtype=torch.float64, device=device), alpha=6
    ).to(device)
    x = torch.randn(batch_size, 4, 7, dtype=torch.float64, device=device)
    absolute = torch.randn(
        batch_size, 3, dtype=torch.float64, device=device, requires_grad=True
    )
    drift = torch.randn(
        batch_size, 3, dtype=torch.float64, device=device, requires_grad=True
    )
    actual = layer(x, abs_gate=absolute, drift_gate=drift)
    expected = []
    for i in range(batch_size):
        update = (
            layer.scale
            * u
            @ torch.diag(drift[i])
            @ layer.core
            @ torch.diag(absolute[i])
            @ v.T
        )
        expected.append(F.linear(x[i], base.weight + update, base.bias))
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-12, atol=1e-12)
    actual.square().mean().backward()
    assert layer.core.grad.norm() > 0
    assert absolute.grad.norm() > 0 and drift.grad.norm() > 0
    assert not layer.U.requires_grad and not layer.V.requires_grad
    assert base.weight.grad is None


def test_zero_core_identity_and_target_delta_scope_and_reload():
    model = Toy()
    x = torch.randn(3, 4, 7, dtype=torch.float64)
    expected = model(x).detach()
    original = copy.deepcopy(model.state_dict())
    install_peft(model, "random_core", rank=3, seed=42)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    model(x).square().mean().backward()
    assert model.q_proj.core.grad.norm() > 0
    with torch.no_grad():
        model.q_proj.core.add_(torch.randn_like(model.q_proj.core) * 0.01)
        model.v_proj.core.add_(torch.randn_like(model.v_proj.core) * 0.01)
    source_prediction = model(x).detach()
    source_q = model.q_proj.core.detach().clone()
    report = begin_target_adaptation(model)
    assert report["trainable_parameters"] == 2 * 3 * 3
    assert all(
        name.endswith("delta_core")
        for name, p in model.named_parameters()
        if p.requires_grad
    )
    assert (model.q_proj.delta_core == 0).all()
    torch.testing.assert_close(model(x), source_prediction, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(adaptation_parameters(model), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    optimizer.step()
    torch.testing.assert_close(model.q_proj.source_core, source_q, rtol=0, atol=0)
    torch.testing.assert_close(
        model.q_proj.base.weight, original["q_proj.weight"], rtol=0, atol=0
    )
    saved_delta = model.q_proj.delta_core.detach().clone()
    begin_target_adaptation(model)
    torch.testing.assert_close(model.q_proj.delta_core, saved_delta, rtol=0, atol=0)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = Toy()
    install_peft(restored, "random_core", rank=3, seed=42)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    assert all(
        name.endswith("delta_core")
        for name, p in restored.named_parameters()
        if p.requires_grad
    )


def test_context_nesting_batch_permutation_and_failure_cleanup():
    model = Toy()
    install_peft(model, "weight_svd_core", rank=3)
    with torch.no_grad():
        model.q_proj.core.fill_(0.1)
        model.v_proj.core.fill_(0.2)
    x = torch.randn(4, 2, 7, dtype=torch.float64)
    a = torch.arange(1, 13, dtype=torch.float64).reshape(4, 3)
    d = torch.flip(a, [1]) * 0.1
    no_context = model(x).detach()
    with peft_context(model, a, d):
        before = model(x).detach()
        with peft_context(model, torch.ones_like(a), torch.ones_like(d)):
            assert not torch.equal(before, model(x))
        torch.testing.assert_close(model(x), before, rtol=0, atol=0)
    torch.testing.assert_close(model(x), no_context, rtol=0, atol=0)
    permutation = torch.tensor([2, 0, 3, 1])
    with peft_context(model, a[permutation], d[permutation]):
        torch.testing.assert_close(
            model(x[permutation]), before[permutation], rtol=1e-13, atol=1e-13
        )
    with pytest.raises(ValueError):
        with peft_context(model, torch.ones(5, 3), torch.ones(5, 3)):
            model(x)
    torch.testing.assert_close(model(x), no_context, rtol=0, atol=0)


def test_author_style_lora_xs_includes_sigma_and_small_gaussian_core():
    base = nn.Linear(7, 5).double()
    original = base.weight.detach().clone()
    layer = LoRAXSLinear(base, 3, alpha=6, seed=13)
    x = torch.randn(4, 7, dtype=torch.float64)
    u, v, singular = svd_bases(original, 3)
    expected = F.linear(
        x, original + 2 * u @ layer.core @ torch.diag(singular) @ v.T, base.bias
    )
    torch.testing.assert_close(layer(x), expected, rtol=1e-12, atol=1e-12)
    assert 0 < layer.core.norm() < 1e-3
    assert not layer.orthogonal_bases
    torch.testing.assert_close(
        layer.V.T @ layer.V, torch.diag(singular.square()), rtol=1e-12, atol=1e-12
    )
    assert trainable_parameter_report(layer)["trainable_parameters"] == 9


@pytest.mark.parametrize("direction", ["ArB2r", "A2rBr", "ArBr"])
def test_lora_ga_compensation_and_source_grad_initialization(direction):
    base = nn.Linear(8, 8).double()
    original = base.weight.detach().clone()
    gradient = torch.randn(8, 8, dtype=torch.float64)
    layer = LoRAGALinear(base, 2, gradient, alpha=4, seed=42, direction=direction)
    x = torch.randn(3, 2, 8, dtype=torch.float64)
    torch.testing.assert_close(
        layer(x), F.linear(x, original, base.bias), rtol=0, atol=0
    )
    torch.testing.assert_close(base.weight, original, rtol=0, atol=0)
    layer(x).square().mean().backward()
    assert layer.lora_a.weight.grad.norm() > 0 and layer.lora_b.weight.grad.norm() > 0
    assert not layer.initial_a.requires_grad and not layer.initial_b.requires_grad
    assert trainable_parameter_report(layer)["fixed_compensation_elements"] == 32


def test_failed_installation_is_atomic_and_random_generators_are_isolated():
    model = Toy()
    before = copy.deepcopy(model.state_dict())
    u, v = random_orthogonal_bases(5, 7, 2)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Missing source-only basis"):
        install_peft(model, "grad_core", 2, bases={"q_proj": (u, v)})
    assert isinstance(model.q_proj, nn.Linear) and isinstance(model.v_proj, nn.Linear)
    assert all(p.requires_grad for p in model.parameters())
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_real_dimension_parameter_budget_matches_requested_curves():
    for rank, expected in [(2, 1536), (4, 3072), (8, 6144)]:
        layer = LoRALinear(nn.Linear(384, 384, bias=False), rank)
        assert trainable_parameter_report(layer)["trainable_parameters"] == expected
    for rank, expected in [(40, 1600), (56, 3136), (80, 6400)]:
        u, v = random_orthogonal_bases(384, 384, rank)
        layer = SubspaceLinear(nn.Linear(384, 384, bias=False), u, v)
        assert trainable_parameter_report(layer)["trainable_parameters"] == expected


@pytest.mark.parametrize("zero_gradient", [False, True])
def test_p1_source_only_gradient_cache_replay_and_state_restoration(
    tmp_path, zero_gradient
):
    import json
    from types import SimpleNamespace

    import numpy as np

    from scripts.drs_peft.gradient_audit import run_audit, state_fingerprint

    class GuardedOutcomes(np.ndarray):
        def __getitem__(self, key):
            first = key[0] if isinstance(key, tuple) else key
            selected = np.arange(self.shape[0])[first]
            if np.asarray(selected).size and np.asarray(selected).max() >= 6:
                raise AssertionError("Final target Query labels were indexed")
            return super().__getitem__(key)

    class Trajectory(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.q_proj = nn.Linear(8, 8).double()
            self.backbone.v_proj = nn.Linear(8, 8).double()
            self.head = nn.Linear(8, 1).double()

        def forward(self, early, query):
            prefix = early["x"].mean(1)
            hidden = self.backbone.q_proj(prefix).tanh() + self.backbone.v_proj(prefix)
            slope = self.head(hidden)
            if zero_gradient:
                slope = slope * 0
            return early["anchor"].reshape(-1, 1) + slope * (query - 100)[None, :] / 10

    rng = np.random.default_rng(19)
    data = SimpleNamespace(
        ids=[f"synthetic-{i}" for i in range(8)],
        protocol_ids=np.asarray(["a", "a", "b", "b", "c", "d", "target", "target"]),
        nominal_capacity_Ah=np.full(8, 1.1),
        query=np.arange(101, 106, dtype=np.float64),
        targets=rng.normal(0.8, 0.1, (8, 5)).view(GuardedOutcomes),
        observed=np.ones((8, 5), dtype=bool),
    )
    early = dict(
        x=rng.normal(size=(8, 2, 8)),
        point_mask=np.ones((8, 2), dtype=bool),
        cycle_ids=np.tile([1.0, 100.0], (8, 1)),
        policy=np.zeros((8, 1)),
        policy_mask=np.ones((8, 1), dtype=bool),
        anchor=np.full(8, 0.95),
    )
    model = Trajectory()
    model.backbone.v_proj.weight.requires_grad_(False)
    model.backbone.v_proj.train(False)
    flags = [p.requires_grad for p in model.parameters()]
    training = [m.training for m in model.modules()]
    fingerprint = state_fingerprint(model)
    config = dict(
        ranks=[1, 2, 3],
        primary_rank=2,
        random_replicates=2,
        bootstrap_draws=50,
        basis_bootstrap_draws=2,
        expected_target_count=2,
        cache_root=str(tmp_path / "runtime"),
        analysis_threads=1,
    )
    output = tmp_path / "artifacts"
    record = run_audit(
        model,
        data,
        early,
        [0, 1, 2, 3],
        [4, 5],
        output,
        seed=42,
        device="cpu",
        config=config,
    )
    assert record["status"] == "COMPLETE_SOURCE_GRADIENT_DIAGNOSTIC"
    assert record["physical_cell_gradient_evaluations"] == 16
    assert record["independent_endpoint_replay_layer_gradients"] == 20
    assert record["optimizer_updates"] == 0
    assert not record["target_Query_future_labels_accessed"]
    assert record["H1_status"].startswith("PARTIAL")
    if zero_gradient:
        assert record["mean_basis_bootstrap_overlap"] is None
        assert not record["single_seed_capture_criterion_passed"]
    assert state_fingerprint(model) == fingerprint
    assert [p.requires_grad for p in model.parameters()] == flags
    assert [m.training for m in model.modules()] == training
    assert all(p.grad is None for p in model.parameters())
    bases = torch.load(output / "P1_BASES.pt", weights_only=False)
    for u, v in bases["bases"].values():
        torch.testing.assert_close(u.T @ u, torch.eye(3), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v.T @ v, torch.eye(3), atol=1e-6, rtol=1e-6)
    replay = json.loads(
        (output / "P1_GRADIENT_REPLAY.json").read_text(encoding="utf-8")
    )
    assert all(row["passed"] for row in replay["records"])
    data.targets.view(np.ndarray)[6:] = 1e20
    repeated = run_audit(
        model,
        data,
        early,
        [0, 1, 2, 3],
        [4, 5],
        output,
        seed=42,
        device="cpu",
        config=config,
    )
    assert repeated == record
    assert state_fingerprint(model) == fingerprint


def test_p1_three_seed_gate_and_permutation_boundary(tmp_path):
    import json

    import pandas as pd

    from scripts.drs_peft.gradient_audit import aggregate_audits
    from scripts.drs_peft.data import file_hash

    def close_fixture(folder):
        files = [
            dict(path=name, sha256=file_hash(folder / name))
            for name in (
                "P1_REGISTRATION.json",
                "P1_RESULT.json",
                "P1_CAPTURE.csv",
                "P1_BASIS_STABILITY.csv",
            )
        ]
        (folder / "P1_PUBLIC_COMPLETE.json").write_text(
            json.dumps(dict(public_files=files)), encoding="utf-8"
        )

    folders = []
    for seed in [42, 123, 2026]:
        folder = tmp_path / f"seed{seed}"
        folder.mkdir()
        registration = dict(
            source_ids=["fixture-source"],
            meta_validation_ids=["fixture-meta"],
            source_protocols=["source"],
            meta_validation_protocols=["meta-a", "meta-b"],
            source_data_sha256="synthetic fixture",
            meta_data_sha256="synthetic fixture",
            query_grid=[101],
            config=dict(primary_rank=4),
            implementation_sha256={"fixture": "synthetic"},
        )
        result = dict(
            seed=seed,
            status="COMPLETE_SOURCE_GRADIENT_DIAGNOSTIC",
            source_model_before_sha256="synthetic",
            source_model_after_sha256="synthetic",
            optimizer_updates=0,
            target_Query_future_labels_accessed=False,
            total_gradient_forward_backward_evaluations=0,
        )
        for name, value in [
            ("P1_REGISTRATION.json", registration),
            ("P1_RESULT.json", result),
        ]:
            (folder / name).write_text(json.dumps(value), encoding="utf-8")
        captures = [
            dict(
                seed=seed,
                protocol_uid=group,
                layer=layer,
                statistic="group_mean",
                rank=4,
                method=method,
                capture=value,
            )
            for group in ["meta-a", "meta-b"]
            for layer in ["q_proj", "v_proj"]
            for method, value in [
                ("grad_cov", 0.8),
                ("random_00", 0.02),
                ("random_01", 0.04),
                ("weight_svd", 0.1),
                ("shuffled_cov", 0.8),
            ]
        ]
        pd.DataFrame(captures).to_csv(folder / "P1_CAPTURE.csv", index=False)
        pd.DataFrame(
            [
                dict(seed=seed, side="left", overlap=0.9),
                dict(seed=seed, side="right", overlap=0.8),
            ]
        ).to_csv(folder / "P1_BASIS_STABILITY.csv", index=False)
        close_fixture(folder)
        folders.append(folder)
    partial = aggregate_audits(
        folders[:1], tmp_path / "partial", dict(bootstrap_draws=50)
    )
    assert partial["status"] == "PARTIAL"
    positive = aggregate_audits(
        folders, tmp_path / "positive", dict(bootstrap_draws=50)
    )
    assert positive["status"] == "PASSED"
    assert positive["P2_large_rank_curve_allowed"]
    assert not positive["permutation_capture_weakening_observed"]
    assert positive["battery_specificity_claim"].startswith("UNESTABLISHED")
    for folder in folders:
        capture = pd.read_csv(folder / "P1_CAPTURE.csv")
        capture.loc[capture.method == "weight_svd", "capture"] = 0.9
        capture.to_csv(folder / "P1_CAPTURE.csv", index=False)
        close_fixture(folder)
    negative = aggregate_audits(
        folders, tmp_path / "negative", dict(bootstrap_draws=50)
    )
    assert negative["status"] == "NEGATIVE"
    assert negative["P2_minimal_controls_allowed"]
    assert not negative["P3_P4_large_expansion_allowed"]
    with pytest.raises(ValueError, match="Repeated source seeds"):
        aggregate_audits([folders[0], folders[0]], tmp_path / "duplicate")


def test_p0_gate_allows_completed_negative_and_p2_registered_branches(tmp_path):
    import json
    from pathlib import Path

    from scripts.drs_peft.run_p1 import require_gate
    from scripts.drs_peft.run_p2 import build_plan

    gate = tmp_path / "P0_GATE.json"
    gate.write_text(
        json.dumps(dict(status="NEGATIVE", execution_complete=True)), encoding="utf-8"
    )
    assert require_gate(gate, "P0")["status"] == "NEGATIVE"
    gate.write_text(
        json.dumps(dict(status="PARTIAL", execution_complete=False)), encoding="utf-8"
    )
    with pytest.raises(RuntimeError):
        require_gate(gate, "P0")
    config = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/drs_peft/execution_v1.json"
        ).read_text(encoding="utf-8")
    )
    negative = build_plan(config, dict(status="NEGATIVE"))
    assert len(negative["search"]) == 18
    assert len(negative["formal"]) == 30
    assert {unit["rank"] for unit in negative["formal"]} == {4, 56}
    positive = build_plan(config, dict(status="PASSED"))
    assert len(positive["formal"]) == 66
    assert sum(unit["rank"] == 56 for unit in positive["formal"]) == 12
    with pytest.raises(RuntimeError):
        build_plan(config, dict(status="PARTIAL"))


def test_p2_failure_classification_and_step_observer_restoration(monkeypatch):
    from scripts.drs_peft.run_p2 import (
        _observe_optimizer_steps,
        _recoverable_trial_error,
    )

    assert _recoverable_trial_error(FloatingPointError("bad objective")) == "NONFINITE"
    assert (
        _recoverable_trial_error(torch.cuda.OutOfMemoryError("synthetic OOM"))
        == "CUDA_OOM"
    )
    assert (
        _recoverable_trial_error(RuntimeError("Registered execution identity changed"))
        is None
    )
    assert _recoverable_trial_error(ValueError("missing basis")) is None

    def fake_step(optimizer, fail=False):
        if fail:
            raise FloatingPointError("synthetic fixture")
        return 17

    monkeypatch.setattr(torch.optim.AdamW, "step", fake_step)
    with _observe_optimizer_steps() as count:
        assert torch.optim.AdamW.step(None) == 17
        with pytest.raises(FloatingPointError):
            torch.optim.AdamW.step(None, fail=True)
        assert count == dict(completed=1, entered=2)
    assert torch.optim.AdamW.step is fake_step


def test_p2_two_budget_summary_preserves_h1_stop(tmp_path):
    import json
    from pathlib import Path

    import pandas as pd

    from scripts.drs_peft.data import file_hash
    from scripts.drs_peft.run_p2 import _unit_path, build_plan, summarize_p2

    execution = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/drs_peft/execution_v1.json"
        ).read_text(encoding="utf-8")
    )
    execution["statistics"]["bootstrap_draws"] = 50
    plan = build_plan(execution, dict(status="NEGATIVE"))
    for unit in plan["formal"]:
        folder = _unit_path(tmp_path, unit)
        folder.mkdir(parents=True)
        value = 0.08 if unit["method"] == "grad_core" else 0.1
        metrics = pd.DataFrame(
            [
                dict(
                    window="all", n_obs=20, mae=value, cell_uid=uid, protocol_uid=group
                )
                for uid, group in [("synthetic-meta-1", "a"), ("synthetic-meta-2", "b")]
            ]
        )
        metrics.to_csv(folder / "SOURCE_VALIDATION_CELL_METRICS.csv", index=False)
        adapter = 12 * (
            unit["rank"] * 768
            if unit["method"] in ("lora", "lora_ga")
            else unit["rank"] ** 2
        )
        parameter = dict(
            head_trainable=10,
            bridge_trainable=20,
            peft_detail=dict(
                adapter_trainable_parameters=adapter,
                frozen_basis_elements=0,
                fixed_compensation_elements=0,
            ),
        )
        (folder / "TRAINABLE_PARAMETER_REPORT.json").write_text(
            json.dumps(parameter), encoding="utf-8"
        )
        receipt = dict(
            validation=dict(mae=value),
            updates=1320,
            trainable_parameters=adapter + 30,
            selected_checkpoint_sha256="synthetic",
            public_files=[
                dict(path=name, sha256=file_hash(folder / name))
                for name in (
                    "SOURCE_VALIDATION_CELL_METRICS.csv",
                    "TRAINABLE_PARAMETER_REPORT.json",
                )
            ],
        )
        (folder / "SOURCE_DEVELOPMENT_COMPLETE.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
    decision = summarize_p2(tmp_path, plan, execution)
    assert decision["status"] == "PASSED"
    assert decision["complete_formal_units"] == 30
    assert decision["parameter_matched_curves"] == "COMPLETE"
    assert decision["additional_rank_curves"] == "NOT_RUN_H1_NEGATIVE_GATE"
    assert decision["H1_stop_remains_active"]
    assert not decision["P3_P4_large_expansion_allowed"]
    assert len(decision["primary_controls"]) == 4


def test_p1_candidate_family_streaming_is_secondary_and_state_preserving(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from scripts.drs_peft.gradient_audit import (
        _candidate_family_diagnostic,
        state_fingerprint,
    )

    class CandidateModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.ModuleDict(
                dict(q_proj=nn.Linear(8, 8).double(), v_proj=nn.Linear(8, 8).double())
            )
            self.head = nn.Linear(8, 1).double()
            self.bridge = SimpleNamespace(
                backbone=SimpleNamespace(
                    linear_targets=lambda family: dict(self.backbone.items())
                )
            )

        def forward(self, early, query):
            x = early["x"].mean(1)
            value = self.head(
                self.backbone["q_proj"](x).tanh() + self.backbone["v_proj"](x)
            )
            return early["anchor"][:, None] + value * (query[None] - 100) / 10

    rng = np.random.default_rng(19)
    data = SimpleNamespace(
        ids=[f"synthetic-{i:02d}" for i in range(20)],
        protocol_ids=np.asarray(
            [f"source-{i // 2}" if i < 12 else f"meta-{i}" for i in range(20)]
        ),
        targets=rng.normal(0.8, 0.1, (20, 5)),
        observed=np.ones((20, 5), bool),
        query=np.arange(101, 106, dtype=np.float64),
        nominal_capacity_Ah=np.full(20, 1.1),
    )
    early = dict(
        x=rng.normal(size=(20, 2, 8)),
        point_mask=np.ones((20, 2), bool),
        cycle_ids=np.tile([1.0, 100.0], (20, 1)),
        policy=np.zeros((20, 1)),
        policy_mask=np.ones((20, 1), bool),
        anchor=np.full(20, 0.95),
    )
    model = CandidateModel()
    model.backbone["v_proj"].weight.requires_grad_(False)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    fingerprint = state_fingerprint(model)
    result = _candidate_family_diagnostic(
        model, data, early, list(range(12)), list(range(12, 20)), tmp_path, 42, "cpu"
    )
    assert result["extra_gradient_forward_backward_evaluations"] == 20
    assert result["optimizer_updates"] == 0
    assert not result["main_QV_target_set_changed"]
    assert not result["rank_or_target_selection_permitted"]
    assert len(pd.read_csv(tmp_path / "P1_CANDIDATE_CELL_CHECKS.csv")) == 40
    assert state_fingerprint(model) == fingerprint
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
