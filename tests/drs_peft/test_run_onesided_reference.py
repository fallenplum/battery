"""CPU checks for no-policy inputs and CV-fixed final source refits."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts.drs_peft import run_onesided_reference as runner
from scripts.drs_peft.train import read_json, seed_all


class ToyData:
    def __init__(self):
        self.ids = ["source_a", "source_b", "sealed_target"]
        self.x = np.ones((3, 2, 2, 4), dtype=np.float32)
        self.x[1] *= 1.1
        self.point_mask = np.ones(self.x.shape, dtype=bool)
        self.cycle_ids = np.tile([1, 100], (3, 1))
        self.anchor = np.array([1.0, 0.98, 0.97], dtype=np.float32)
        self.query = np.array([101, 200, 500], dtype=np.float32)
        self.targets = self.anchor[:, None] - 0.1 * (self.query - 100) / 1000
        self.targets[2] = np.nan
        self.observed = np.array(
            [[True, True, True], [True, True, False], [False, False, False]]
        )

    def rows(self, ids):
        return np.array([self.ids.index(uid) for uid in ids], dtype=int)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1, bias=False)
        self.fixed = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.official_backbone_identity = {"engineering_only": True}

    def forward(self, early, query):
        prefix = early["x"].mean((1, 2, 3))[:, None]
        rate = self.head(prefix) * self.fixed
        return early["anchor"][:, None] + rate * (query - 100) / 1000

    def parameter_report(self):
        return {"trainable": 1}


def toy_factory():
    seed_all(42)
    return ToyModel()


def arguments(tmp_path):
    config_file = tmp_path / "CONFIG.json"
    config_file.write_text("{}", encoding="utf-8")
    return SimpleNamespace(config=config_file, device="cpu", stop_after=None)


def test_no_policy_inputs_keep_physical_arrays_and_never_fit_metadata():
    data = ToyData()
    data.policy = np.full((3, 28), np.nan)
    data.policy_mask = np.ones((3, 28), dtype=bool)
    scaler = runner.NoPolicyInputs().fit(data, [0, 1])
    inputs = scaler.transform(data)
    assert inputs["x"] is data.x
    assert inputs["anchor"] is data.anchor
    assert inputs["policy"].shape == inputs["policy_mask"].shape == (3, 0)
    assert inputs["policy"].dtype == np.float32
    assert inputs["policy_mask"].dtype == bool
    receipt = scaler.to_dict()
    assert receipt["training_ids"] == ["source_a", "source_b"]
    assert receipt["learned_input_normalization_parameters"] == 0
    assert receipt["policy_dimension"] == 0
    with pytest.raises(ValueError):
        runner.NoPolicyInputs().fit(data, [0, 0])


def test_reference_paths_distinguish_cv_and_all_source_refits(tmp_path):
    assert runner.reference_folder(tmp_path, 2, 42) == tmp_path / "fold2/seed42"
    assert runner.reference_folder(tmp_path, "final", 123) == tmp_path / "final/seed123"


def test_final_epoch_budget_is_stagewise_source_cv_median(tmp_path):
    for fold, epochs in enumerate(((1, 11), (4, 7), (10, 9))):
        for stage, epoch in zip(("warmup", "static"), epochs, strict=True):
            path = runner.reference_folder(tmp_path, fold, 42) / stage
            path.mkdir(parents=True)
            (path / "SOURCE_DEVELOPMENT_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "status": "SOURCE_DEVELOPMENT_COMPLETE",
                        "seed": 42,
                        "target_query_labels_used": 0,
                        "best_epoch": epoch,
                    }
                ),
                encoding="utf-8",
            )
    assert runner.final_epoch_budgets(tmp_path, 42) == {"warmup": 4, "static": 9}
    bad = (
        runner.reference_folder(tmp_path, 0, 42)
        / "warmup/SOURCE_DEVELOPMENT_COMPLETE.json"
    )
    payload = read_json(bad)
    payload["target_query_labels_used"] = 1
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-only"):
        runner.final_epoch_budgets(tmp_path, 42)


def test_final_fixed_refit_selects_last_epoch_and_resumes_exactly(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "source_snapshot", lambda *args, **kwargs: [])
    data, args = ToyData(), arguments(tmp_path)
    fit_ids = data.ids[:2]
    scaler = runner.NoPolicyInputs().fit(data, data.rows(fit_ids))
    inputs = scaler.transform(data)
    config = {"reference": {"lr": {"warmup": 0.01}}}

    def train(folder):
        return runner.train_fixed_budget_unit(
            args,
            config,
            data,
            inputs,
            scaler,
            fit_ids,
            tmp_path / folder,
            toy_factory,
            42,
            "warmup",
            3,
        )

    direct = train("direct")
    assert direct["status"] == "SOURCE_DEVELOPMENT_COMPLETE"
    assert direct["best_epoch"] == direct["epochs"] == direct["updates"] == 3
    assert direct["validation"] is None
    identity = read_json(tmp_path / "direct/RUN_CONFIG.json")
    assert identity["validation_ids"] == []
    assert identity["training_ids"] == fit_ids
    assert identity["target_query_labels_used"] == 0
    assert "last_epoch" in identity["epoch_selection"]
    assert direct["prediction_replay_max_absolute_error"] == 0

    args.stop_after = 1
    assert train("resumed")["status"] == "ENGINEERING_INTERRUPTED"
    args.stop_after = None
    resumed = train("resumed")
    assert resumed["updates"] == 3
    checkpoint = torch.load(
        tmp_path / "direct/CHECKPOINT_SELECTED.pt",
        map_location="cpu",
        weights_only=False,
    )
    replay = torch.load(
        tmp_path / "resumed/CHECKPOINT_SELECTED.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["identity"] == replay["identity"]
    assert all(
        torch.equal(value, replay["model"][key])
        for key, value in checkpoint["model"].items()
    )
    train("resumed")


def test_small_artifacts_exclude_weights_and_detect_changed_public_copy(tmp_path):
    private, public = tmp_path / "private", tmp_path / "public"
    private.mkdir()
    (private / "marker.json").write_text("{}", encoding="utf-8")
    (private / "weights.pt").write_bytes(b"private")
    runner.publish_small_artifacts(private, public)
    assert sorted(path.name for path in public.iterdir()) == ["marker.json"]
    (public / "marker.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="published"):
        runner.publish_small_artifacts(private, public)


def test_input_audit_preserves_anchor_and_ignores_policy():
    data = ToyData()
    inputs = runner.NoPolicyInputs().fit(data, [0, 1]).transform(data)
    audit = runner.input_scope_audit(toy_factory(), inputs, np.array([0, 1]), "cpu")
    assert audit["status"] == "PASSED"
    assert audit["future_labels_used"] == audit["anchor_max_absolute_error"] == 0
