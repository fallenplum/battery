"""Exercise the experiment gates that protect source-outer scoring."""

from argparse import Namespace

import pytest

from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.spectral_objective_engine import (
    authenticate_trial,
    stage_a_decision,
)
from scripts.drs_peft.spectral_objective_outer import (
    load_hyperparameter_lock,
    selected_mapper_trials,
)
from scripts.drs_peft.spectral_objective_protocol import (
    CANDIDATE,
    CONFIG,
    MATCHED_RATE,
    META_METHODS,
    META_OBJECTIVES,
)
from scripts.drs_peft.spectral_rotation_engine import Labels
from scripts.drs_peft.train import read_json


def test_outer_requires_stage_a_and_unchanged_hyperparameters(tmp_path):
    cfg = read_json(CONFIG)
    original = {
        str(fold): {
            CANDIDATE: {"alpha": 0.01, "validation_mae": score},
            "K0": {"validation_mae": 0.02},
        }
        for fold, score in enumerate((0.01999, 0.01998, 0.02001))
    }
    decision = stage_a_decision(cfg, original) | {"selected": original}
    assert decision["status"] == "PASS_STAGE_A"
    args = Namespace(public=tmp_path)
    write_json(tmp_path / "STAGE_A_DECISION.json", decision)
    identity = "identity"
    with pytest.raises(FileNotFoundError):
        load_hyperparameter_lock(args, cfg, decision, identity)
    hyper_path = tmp_path / "HYPERPARAMETER_LOCK.json"
    content = {
        "status": "LOCKED_AFTER_STAGE_A",
        "identity_sha256": identity,
        "stage_a_sha256": file_hash(tmp_path / "STAGE_A_DECISION.json"),
        "selected": original,
    }
    write_json(hyper_path, content)
    assert load_hyperparameter_lock(args, cfg, decision, identity) == (
        hyper_path,
        original,
    )
    content["identity_sha256"] = "different"
    write_json(hyper_path, content)
    with pytest.raises(RuntimeError):
        load_hyperparameter_lock(args, cfg, decision, identity)
    original["0"][CANDIDATE]["validation_mae"] = 0.02
    assert stage_a_decision(cfg, original)["status"] == "STOP_NO_VALIDATION_SIGNAL"


def test_mapper_tamper_and_unauthorized_future_rows(tmp_path, monkeypatch):
    mapper = tmp_path / "MAPPER.pt"
    mapper.write_bytes(b"immutable controller")
    identity = {"alpha": 0.01}
    write_json(
        tmp_path / "COMPLETE.json",
        {
            "status": "COMPLETE_MAPPER",
            "identity": identity,
            "files": {"MAPPER.pt": file_hash(mapper)},
            "cost": {"source_mapping_updates": 32, "source_inner_updates": 96},
        },
    )
    authenticate_trial(tmp_path, identity)
    with pytest.raises(RuntimeError):
        authenticate_trial(tmp_path, {"alpha": 0.001})
    mapper.write_bytes(b"changed")
    with pytest.raises(RuntimeError):
        authenticate_trial(tmp_path, identity)

    def inaccessible(*args, **kwargs):
        pytest.fail("The raw archive must not be opened for an unauthorized UID")

    monkeypatch.setattr(
        "scripts.drs_peft.spectral_rotation_engine.load_future_rows", inaccessible
    )
    with pytest.raises(RuntimeError):
        Labels.read(tmp_path, {"sealed"}, {"source"}, {"sealed"})
    with pytest.raises(RuntimeError):
        Labels.read(tmp_path, {"sealed"}, {"sealed"}, {"sealed"})


def test_matched_rate_reuses_only_identical_training_assignment():
    selected = {
        f"{method}_{tag}": {"alpha": 0.01}
        for method in META_METHODS
        for tag in META_OBJECTIVES
    }
    entries = selected_mapper_trials(selected, seed=43)
    assert entries[MATCHED_RATE] == entries["ROTATION_TRUE_MM"]
    assert len(set(entries.values())) == 8
    selected[CANDIDATE]["alpha"] = 0.001
    entries = selected_mapper_trials(selected, seed=43)
    assert len(set(entries.values())) == 9
    assert entries[MATCHED_RATE].objective == "MM"
    assert entries[MATCHED_RATE].alpha == entries[CANDIDATE].alpha
    assert entries[MATCHED_RATE].training_seed == 43
