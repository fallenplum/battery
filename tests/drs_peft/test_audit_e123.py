"""Independent-audit regressions for denominator tampering and norm controls."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from models.drs_peft.e123_conditioning import BoundedConditionedLinear
from models.drs_peft.peft import SubspaceLinear
from scripts.drs_peft import audit_e123
from scripts.drs_peft.audit_e123 import (
    ROOT,
    audit_norm_table,
    condition_norms,
    read,
    recompute_metrics,
    units,
)
from scripts.drs_peft.train import evaluate_arrays


def test_independent_matrix_contains_registered_216_units_and_207360_updates():
    registration = read(ROOT / "configs/drs_peft/e123_v1.json")
    protocol = read(ROOT / "configs/drs_peft/e123_folds_v1.json")
    matrix = units(registration, protocol)
    assert len(matrix) == 216
    assert sum(unit["stage"] == "warmup" for unit in matrix) == 9
    assert sum(unit["stage"] == "static" for unit in matrix) == 27
    assert sum(unit["stage"] == "condition" for unit in matrix) == 180
    fits = {fold["fold_id"]: len(fold["fit_ids"]) for fold in protocol["folds"]}
    updates = sum(
        registration["epochs"][unit["stage"]] * int(np.ceil(fits[unit["fold_id"]] / 8))
        for unit in matrix
    )
    assert updates == registration["expected_optimizer_updates"] == 207360


def test_npz_metric_recomputation_rejects_changed_observation_masks(tmp_path):
    ids = ["a", "b"]
    query = np.array([101, 200, 501, 1001])
    target = np.array([[1.0, 0.9, 0.7, 0.5], [1.0, 0.8, 0.6, 0.4]], np.float32)
    prediction = np.array(
        [[0.99, 0.88, 0.75, 0.55], [1.01, 0.83, 0.58, 0.5]], np.float32
    )
    observed = np.array([[True, True, True, True], [True, True, False, False]])
    data = SimpleNamespace(
        query=query,
        targets=target,
        observed=observed,
        protocol_ids=np.array(["g1", "g2"]),
    )
    data.rows = lambda selected: np.array([ids.index(uid) for uid in selected])
    table, _ = evaluate_arrays(
        prediction, target, observed, query, ids, data.protocol_ids, "original_cycle_id"
    )
    path, csv = tmp_path / "arrays.npz", tmp_path / "metrics.csv"
    table.to_csv(csv, index=False)
    np.savez(
        path,
        ids=ids,
        query=query,
        prediction=prediction,
        target=target,
        observed=observed,
    )
    _, maximum, count = recompute_metrics(path, csv, ids, data)
    assert count == 10 and maximum < 1e-12
    changed = observed.copy()
    changed[1, -1] = True
    np.savez(
        path,
        ids=ids,
        query=query,
        prediction=prediction,
        target=target,
        observed=changed,
    )
    with pytest.raises(AssertionError):
        recompute_metrics(path, csv, ids, data)


def norm_table():
    return pd.DataFrame(
        {
            "cell_uid": ["a", "b"],
            "layer": ["q", "q"],
            "raw_increment_norm": [2.0, 0.0],
            "increment_norm": [1.0, 0.0],
            "radius": [1.0, 1.0],
            "norm_reference": [1.0, 1.0],
            "total_update_norm": [1.1, 0.1],
            "baseline_update_norm": [0.1, 0.1],
            "matched_norm_unattainable": [False, True],
            "zero_increment": [False, True],
            "cap_active": [True, False],
        }
    )


def test_norm_audit_preserves_zero_denominator_support_and_rejects_cap_violation():
    table = norm_table()
    result = audit_norm_table(table, "matched_norm")
    assert result["unattainable_rows"] == 1
    table.loc[0, "increment_norm"] = 1.01
    with pytest.raises(RuntimeError, match="exceeds"):
        audit_norm_table(table, "trained_ball")


@pytest.mark.parametrize("scaled", (False, True))
@pytest.mark.parametrize("evaluation_mode", ("trained_ball", "matched_norm"))
def test_numpy_norm_reconstruction_matches_actual_dense_weight_updates(
    scaled, evaluation_mode
):
    generator = np.random.default_rng(42)
    ids = ["a", "b", "c"]
    features = {
        view: {uid: generator.normal(size=12).astype(np.float32) for uid in ids}
        for view in ("absolute", "relative")
    }
    state, expected = {}, []
    for index in range(12):
        prefix = f"bridge.backbone.module.layer{index}"
        u = np.linalg.qr(generator.normal(size=(7, 4)), mode="reduced")[0].astype(
            np.float32
        )
        v = np.linalg.qr(generator.normal(size=(7, 4)), mode="reduced")[0].astype(
            np.float32
        )
        if scaled:
            v *= np.array([1.0, 3.0, 10.0, 40.0], dtype=np.float32)
        core = generator.normal(size=(4, 4)).astype(np.float32) * 0.1
        delta = generator.normal(size=(4, 4)).astype(np.float32) * 0.02
        source = SubspaceLinear(
            nn.Linear(7, 7),
            torch.tensor(u),
            torch.tensor(v),
            torch.tensor(core),
            require_orthogonal=not scaled,
        )
        layer = BoundedConditionedLinear(source, 1, source_norm_reference=0.05).eval()
        layer.evaluation_mode = evaluation_mode
        with torch.no_grad():
            layer.delta_core.copy_(torch.tensor(delta))
        for name in (
            "U",
            "V",
            "source_core",
            "delta_core",
            "core_norm_reference",
            "core_radius_multiplier",
        ):
            state[prefix + "." + name] = getattr(layer, name).detach().clone()
        gates = []
        for view in ("absolute", "relative"):
            weight = generator.normal(size=(4, 12)).astype(np.float32) * 0.1
            bias = generator.normal(size=4).astype(np.float32) * 0.02
            gate_prefix = f"bridge.backbone.module.core_gates.{view}.{index}"
            state[gate_prefix + ".weight"] = torch.tensor(weight)
            state[gate_prefix + ".bias"] = torch.tensor(bias)
            values = torch.tensor(np.stack([features[view][uid] for uid in ids]))
            gates.append(
                1
                + 0.5 * torch.tanh(values @ torch.tensor(weight).T + torch.tensor(bias))
            )
        increment, _ = layer.conditioned_increment(3, *gates)
        dense = layer.U.double()[None] @ increment.double() @ layer.V.double().T[None]
        norms = (
            torch.linalg.matrix_norm(dense, ord="fro", dim=(-2, -1)).detach().numpy()
        )
        expected.extend(norms.tolist())
    actual = condition_norms(state, features, ids, "dual", evaluation_mode)
    np.testing.assert_allclose(
        actual.independent_increment_norm, expected, rtol=2e-5, atol=2e-7
    )


def test_completed_optimizer_recovery_preserves_prefix_and_rejects_extra_updates(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit_e123, "ROOT", tmp_path)
    public = tmp_path / "results"
    (public / "io_invocations").mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    amendment = {
        "configuration_sha256": audit_e123.digest(config),
        "training_operator_changes": 0,
        "final_target_query_labels_used": 0,
        "lock_root": str(tmp_path / "_locks"),
    }
    for name in ("driver", "launcher", "trainer"):
        path = tmp_path / (name + ".py")
        path.write_text("pass\n", encoding="utf-8")
        amendment[name] = {"path": path.name, "sha256": audit_e123.digest(path)}
    amendment_path = public / "IO_LOCK_AMENDMENT.json"
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
    unit = tmp_path / "private/fold0/seed42/warmup"
    unit.mkdir(parents=True)
    history = [{"epoch": index + 1, "updates": (index + 1) * 8} for index in range(120)]
    state = {
        "updates": 960,
        "history": history,
        "optimizer": {"state": {0: {"step": torch.tensor(960)}}},
    }
    torch.save(state, unit / "TRAIN_STATE.pt")
    (unit / "TRAIN_LOG.json").write_text(json.dumps(history), encoding="utf-8")
    before = {
        "updates": 960,
        "was_epoch_complete": True,
        "original_files": {
            "RUN_CONFIG.json": "config",
            "SOURCE_SNAPSHOT.json": "source",
            "TRAIN_STATE.pt": audit_e123.digest(unit / "TRAIN_STATE.pt"),
        },
    }
    invocation = {
        "status": "COMPLETE",
        "amendment_sha256": audit_e123.digest(amendment_path),
        "requested_lock_mappings": [],
        "before_states": {str(unit): before},
        "after_states": {str(unit): before},
        "new_optimizer_updates": 0,
        "resumed_state_checks": [
            {
                "unit": str(unit),
                "new_optimizer_updates": 0,
                "completed_optimizer_state_preserved": True,
            }
        ],
    }
    invocation_path = public / "io_invocations/call.json"
    invocation_path.write_text(json.dumps(invocation), encoding="utf-8")
    args = SimpleNamespace(public_root=public, config=config)
    result = audit_e123.recovery_provenance(args)
    assert result["recovered_warmups"][0]["optimizer_updates"] == 960
    assert result["recovered_warmups"][0]["finalization_optimizer_updates"] == 0
    after = {**before, "updates": 961}
    invocation["after_states"] = {str(unit): after}
    invocation["new_optimizer_updates"] = 1
    invocation["resumed_state_checks"][0]["new_optimizer_updates"] = 1
    invocation_path.write_text(json.dumps(invocation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="repeated training updates"):
        audit_e123.recovery_provenance(args)
