"""Regression checks for matrix freezing, role isolation and source replay guards."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.drs_peft import run_e123


def configuration():
    return {
        "study_id": "unit_e123",
        "training_seeds": [42, 123, 2026],
        "methods": list(run_e123.METHODS),
        "modes": list(run_e123.MODES),
        "evaluation_modes": list(run_e123.EVALUATION_MODES),
        "primary_radius": 1.0,
        "grad_radius_sensitivity": [0.25, 4.0],
    }


def test_exact_registered_matrix_and_amplitude_scope():
    protocol = {
        "folds": [{"fold_id": value} for value in range(3)],
        "protocol_sha256": "fixture",
    }
    plan = run_e123.build_plan(configuration(), protocol)
    assert [len(plan[name]) for name in ("warmup", "static", "condition")] == [
        9,
        27,
        180,
    ]
    assert plan["total_training_units"] == 216
    assert (
        len(
            {
                str(run_e123.unit_path(Path("fixture"), unit))
                for unit in plan["condition"]
            }
        )
        == 180
    )
    assert {
        unit["radius"] for unit in plan["condition"] if unit["method"] == "grad_core"
    } == {0.25, 1.0, 4.0}
    assert all(
        unit["radius"] == 1.0
        for unit in plan["condition"]
        if unit["method"] != "grad_core"
    )
    assert plan == run_e123.build_plan(configuration(), protocol)


@pytest.mark.parametrize("radii", [[0], [-1], [1], [0.25, 0.25]])
def test_amplitude_registration_rejects_invalid_or_duplicate_trials(radii):
    config = configuration()
    config["grad_radius_sensitivity"] = radii
    with pytest.raises(ValueError):
        run_e123.build_plan(config, {"folds": [], "protocol_sha256": "fixture"})


def test_outer_scores_stay_sealed_until_all_registered_units_complete(tmp_path):
    args = SimpleNamespace(private_root=tmp_path)
    plan = {
        "warmup": [{"stage": "warmup", "fold_id": 0, "seed": 42}],
        "static": [],
        "condition": [],
    }
    with pytest.raises(RuntimeError, match="remain incomplete"):
        run_e123.require_all_training_complete(args, plan)


def test_selected_checkpoint_guard_detects_modified_completion_artifact(tmp_path):
    artifact = tmp_path / "CHECKPOINT_SELECTED.pt"
    torch.save({"model": {"test": torch.ones(1)}}, artifact)
    record = {"path": artifact.name, "sha256": run_e123.file_hash(artifact)}
    receipt = {
        "files": [record],
        "selected_checkpoint_sha256": record["sha256"],
        "target_query_labels_used": 0,
    }
    run_e123.write_json(tmp_path / "SOURCE_DEVELOPMENT_COMPLETE.json", receipt)
    checkpoint, path = run_e123.checked_training_checkpoint(tmp_path)
    assert path == artifact and torch.equal(checkpoint["model"]["test"], torch.ones(1))
    artifact.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact changed"):
        run_e123.checked_training_checkpoint(tmp_path)


def test_fold_gate_scaler_uses_only_fit_cells_and_ignores_target_prefixes():
    rng = np.random.default_rng(5)
    data = SimpleNamespace(ids=["fit0", "fit1", "inner", "outer", "final"])
    data.rows = lambda ids: np.asarray([data.ids.index(uid) for uid in ids])
    inputs = {
        "x": rng.normal(size=(5, 100, 64, 4)).astype(np.float32),
        "point_mask": np.ones((5, 100, 64), bool),
        "cycle_ids": np.tile(np.arange(1, 101), (5, 1)),
    }
    fold = {
        "fit_ids": data.ids[:2],
        "inner_validation_ids": ["inner"],
        "outer_test_ids": ["outer"],
    }
    original, manifest = run_e123.prepare_condition_inputs(data, inputs, fold)
    altered = copy.deepcopy(inputs)
    altered["x"][2:4] *= 10000
    altered["x"][4] = np.nan
    repeated, second = run_e123.prepare_condition_inputs(data, altered, fold)
    assert manifest["normalization"] == second["normalization"]
    for view in ("absolute", "relative"):
        np.testing.assert_array_equal(
            original["diagnostic_" + view][:2], repeated["diagnostic_" + view][:2]
        )
        assert not repeated["diagnostic_" + view][4].any()
    assert manifest["normalization"]["source_ids"] == ["fit0", "fit1"]


def test_update_diagnostics_preserve_values_ids_and_types():
    model = SimpleNamespace(
        update_diagnostics=lambda batch: {
            "layer": {
                "increment_norm": torch.tensor([0.1, 0.2]),
                "zero_increment": torch.tensor([False, True]),
            }
        }
    )
    rows = run_e123.update_diagnostic_rows(model, {}, ["a", "b"])
    assert len(rows) == 2
    assert rows[0]["cell_uid"] == "a"
    assert rows[0]["increment_norm"] == pytest.approx(0.1)
    assert rows[1]["zero_increment"] is True
    json.dumps(rows, allow_nan=False)


def test_cached_basis_publication_repairs_partial_copy(tmp_path):
    private, public = tmp_path / "private", tmp_path / "public"
    private.mkdir()
    public.mkdir()
    filenames = (
        "GRADIENT_CACHE_MANIFEST.json",
        "GRADIENT_REPLAY.json",
        "GRADIENT_SPECTRA.csv",
    )
    for name in filenames:
        (private / name).write_bytes(name.encode())
    run_e123.write_json(private / "BASES_COMPLETE.json", {"status": "fixture"})
    (public / filenames[0]).write_bytes(b"incomplete_previous_copy")
    run_e123.publish_bases(private, public)
    before = {name: (public / name).read_bytes() for name in filenames}
    assert all(before[name] == (private / name).read_bytes() for name in filenames)
    assert not list(public.glob("*.pending"))
    run_e123.publish_bases(private, public)
    assert before == {name: (public / name).read_bytes() for name in filenames}


def test_condition_factory_uses_registered_weight_reference_fraction(
    tmp_path, monkeypatch
):
    from models.drs_peft import e123_conditioning

    class FakeConditioned(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.delta = torch.nn.Parameter(torch.zeros(1440))

    captured = {}

    def construct(model, mode, radius, reference_weight_fraction):
        captured["fraction"] = reference_weight_fraction
        return FakeConditioned()

    path = tmp_path / "registration.json"
    run_e123.write_json(path, {"reference_weight_fraction": 0.02})
    monkeypatch.setattr(e123_conditioning, "ConditionedBatteryTSFM", construct)
    monkeypatch.setattr(
        run_e123, "static_factory", lambda *args: lambda: SimpleNamespace()
    )
    monkeypatch.setattr(run_e123, "load_selected", lambda *args: None)
    monkeypatch.setattr(
        run_e123, "selected_state", lambda model: {"delta": model.delta}
    )
    unit = {
        "stage": "condition",
        "fold_id": 0,
        "seed": 42,
        "method": "grad_core",
        "mode": "constant",
        "radius": 1.0,
    }
    factory = run_e123.condition_factory(
        SimpleNamespace(config=path), {}, unit, None, None, {"model": {}}
    )
    factory()
    assert captured["fraction"] == 0.02
