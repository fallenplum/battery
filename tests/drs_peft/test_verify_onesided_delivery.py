"""Prevent slot/unique-state miscounting and accidental scientific stop gates."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from scripts.drs_peft.onesided_protocol import payload_hash
from scripts.drs_peft.run_onesided import CONFIG, ROOT
from scripts.drs_peft.verify_onesided_delivery import (
    DeliveryVerifier,
    matrix_counts,
    slot_specs,
)


def setup(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registration = json.loads(
        (ROOT / config["episodes_file"]).read_text(encoding="utf-8")
    )
    args = SimpleNamespace(
        config=CONFIG,
        public_root=tmp_path / "public",
        private_root=tmp_path / "private",
        reference_root=tmp_path / "private/references",
    )
    return DeliveryVerifier(args, config, registration)


def test_slots_preserve_draw_rows_and_distinct_physical_state_counts(tmp_path):
    verifier = setup(tmp_path)
    source = slot_specs(
        "D2",
        verifier.registration["source_tasks"]["outer"],
        [42, 123, 2026],
        ["CAL2", "CORE", "OUT", "IN", "NORMAL", "LoRA", "LoRA_FA"],
    )
    assert matrix_counts(source) == {
        "slots": 954,
        "unique_slot_keys": 954,
        "zero_references": 9,
        "nonzero_slots": 945,
        "unique_nonzero_support_states": 861,
    }
    target = slot_specs(
        "D4",
        [
            task
            for task in verifier.registration["target_tasks"]
            if task["domain"] in verifier.config["D4"]["domains"]
        ],
        [42, 123, 2026],
        ["CAL2", "OUT", "LoRA"],
    )
    assert matrix_counts(target) == {
        "slots": 690,
        "unique_slot_keys": 690,
        "zero_references": 15,
        "nonzero_slots": 675,
        "unique_nonzero_support_states": 549,
    }
    spectrum = slot_specs(
        "D3",
        verifier.registration["source_tasks"]["outer"],
        [42],
        ["AUTO", "CROSS", "RANDOM"],
    )
    assert matrix_counts(spectrum)["nonzero_slots"] == 135
    assert matrix_counts(source + source[:1])["unique_slot_keys"] == 954


def test_pending_delivery_is_partial_and_never_loads_dataset(tmp_path):
    verifier = setup(tmp_path)
    report = verifier.run(["target-lock"])
    assert report["status"] == "PARTIAL"
    assert report["errors"] == 0
    assert report["pending_artifacts"] == 1
    assert (
        report["Query_future_labels_read"]
        == report["dataset_loads"]
        == report["model_initializations"]
        == 0
    )


def test_hash_change_is_failure_while_absence_is_partial(tmp_path):
    verifier = setup(tmp_path)
    path = tmp_path / "state.pt"
    path.write_bytes(b"actual")
    verifier.hash_check(path, "different")
    report = verifier.run([])
    assert report["status"] == "FAILED"
    assert report["errors"] == 1


def test_unestimable_d3_does_not_require_padded_135_states(tmp_path, monkeypatch):
    verifier = setup(tmp_path)

    def no_directions(fold, seed, spectrum=False):
        assert spectrum
        key = (str(fold), seed, True)
        if key not in verifier.dictionary_metadata:
            verifier.na.append(
                {
                    "fold_id": fold,
                    "seed": seed,
                    "reason": "NOT_ESTIMABLE_NO_POSITIVE_CROSS_DIRECTION",
                }
            )
            verifier.dictionary_metadata[key] = {
                "dimensions": {},
                "status": "NOT_ESTIMABLE_NO_POSITIVE_CROSS_DIRECTION",
                "common_dimension": 0,
            }
        return verifier.dictionary_metadata[key]

    monkeypatch.setattr(verifier, "dictionary", no_directions)
    monkeypatch.setattr(verifier, "verify_inventory", lambda *args: None)
    monkeypatch.setattr(
        verifier,
        "verify_unit",
        lambda spec, *args: {"K": 0, "adaptation_id": f"zero{spec['task']['fold_id']}"},
    )
    report = verifier.run(["D3"])
    assert report["status"] == "COMPLETE"
    assert len(report["D3_not_estimable"]) == 3
    assert report["counts"]["D3"]["nonzero_slots"] == 0
    assert report["counts"]["D3"]["zero_references"] == 3
    assert report["scientific_performance_stopping_gates"] == 0


def test_actual_coordinates_and_frozen_bridge_head_are_checked(tmp_path):
    verifier = setup(tmp_path)
    expected = {
        "method": "OUT",
        "draw": 0,
        "support_ids": ["cell_b", "cell_a"],
        "source_checkpoint_sha256": "source",
    }
    identity = {key: value for key, value in expected.items() if key != "draw"}
    identity["support_ids"] = sorted(identity["support_ids"])
    receipt = {"adaptation_id": payload_hash(identity)}
    path = tmp_path / "STATE.pt"
    saved = {
        "identity": identity,
        "state": {"a": np.zeros(16), "source_checkpoint_sha256": "source"},
        "details": {
            "parameter_report": {
                "adaptation_parameters": 16,
                "bridge_trainable": 0,
                "head_trainable": 0,
            }
        },
    }
    torch.save(saved, path)
    assert verifier.verify_state(path, expected, receipt) == 16
    assert not verifier.issues
    saved["state"]["a"] = np.zeros(17)
    saved["details"]["parameter_report"]["head_trainable"] = 1
    torch.save(saved, path)
    verifier.verify_state(path, expected, receipt)
    assert len(verifier.issues) == 2
