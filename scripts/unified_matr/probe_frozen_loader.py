"""Read-only normal-entry loader probe through a temporary freeze certificate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
DATA_FILES = ("dataset_manifest.json", "cells.parquet", "early_viq_qv_64.npz",
              "early_physics_features.parquet", "future_soh.parquet", "raw_record_audit.parquet",
              "instrument_cycle_mapping.parquet", "instrument_identity_bindings.json", "split_contract.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def probe_training_loader(data_root: Path, qa_result: dict) -> dict:
    base = {"scope": "normal_training_loader_read_only_probe", "optimizer_updates": 0,
            "model_instantiated": False, "predictions_scored": False}
    checks = qa_result.get("checks", [])
    if (qa_result.get("status") != "PASS" or qa_result.get("formal_training_authorized_by_data_gate") is not True
            or not checks or any(item.get("status") != "PASS" for item in checks)):
        return {**base, "status": "PENDING", "reason": "ALL_INDEPENDENT_SEMANTIC_QA_MUST_PASS_FIRST", "loader_executed": False}
    source = Path(data_root).resolve()
    missing = [name for name in DATA_FILES if not (source / name).is_file()]
    if missing:
        return {**base, "status": "PENDING", "reason": "COMPLETE169_ARTIFACTS_UNAVAILABLE", "missing": missing, "loader_executed": False}
    summary = qa_result.get("summary", {})
    targets = summary.get("targets", {})
    qualification = targets.get("qualification", [])
    mat_receipts = summary.get("source_receipts", {}).get("released_mat_archives", [])
    axis = next((item for item in checks if item.get("id") == "instrument_original_axis_certified"), {})
    mapping_sha = sha256(source / "instrument_cycle_mapping.parquet")
    if (len(qualification) != 169 or len(mat_receipts) != 4 or axis.get("status") != "PASS"
            or axis.get("detail", {}).get("mapping_sha256") != mapping_sha):
        return {**base, "status": "FAIL", "reason": "QA_IDENTITY_AXIS_OR_QUALIFICATION_SCHEMA_MISMATCH", "loader_executed": False}
    for name in ("future_observations", "rate_support"):
        if name not in targets:
            return {**base, "status": "FAIL", "reason": "QA_TARGET_COUNTS_MISSING_" + name, "loader_executed": False}
    original_freeze = source / "DATA_FREEZE.json"
    original_sha = sha256(original_freeze) if original_freeze.exists() else None
    parent = source.parent.resolve()
    linked, copied = [], []
    with tempfile.TemporaryDirectory(prefix=source.name + "_loader_probe_", dir=parent) as directory:
        temporary = Path(directory).resolve()
        if temporary.parent != parent or temporary == source:
            raise RuntimeError("Temporary loader probe escaped its verified data-parent directory")
        for name in DATA_FILES:
            try:
                os.link(source / name, temporary / name)
                linked.append(name)
            except OSError:
                shutil.copy2(source / name, temporary / name)
                copied.append(name)
        write_json(temporary / "dataset_qa.json", copy.deepcopy(qa_result))
        entries = [{"path": name, "sha256": sha256(temporary / name), "bytes": (temporary / name).stat().st_size}
                   for name in (*DATA_FILES, "dataset_qa.json")]
        fingerprint = temporary / "dataset_fingerprint.sha256"
        fingerprint.write_text("".join(entry["sha256"] + "  " + entry["path"] + "\n" for entry in entries), encoding="utf-8")
        entries.append({"path": fingerprint.name, "sha256": sha256(fingerprint), "bytes": fingerprint.stat().st_size})
        write_json(temporary / "DATA_FREEZE.json", {"status": "DATA_FREEZE_PASS", "training_seeds": [42],
                   "files": entries, "dataset_fingerprint_sha256": sha256(fingerprint),
                   "source_file_sha256": mat_receipts, "source_SHA_and_axis_receipts": summary["source_receipts"],
                   "temporary_loader_probe_only": True})
        sys.path.insert(0, str(REPOSITORY))
        from scripts.unified_matr.train_seed42 import load_frozen_data

        loaded = load_frozen_data(temporary)
        _, identifiers, eligible, _, query, _, observed, _, rate_mask, _, _ = loaded
        expected_ids = [row["cell_uid"] for row in qualification]
        actual_rates = {name: int(rate_mask[:, index].sum()) for index, name in enumerate(("r200_500", "r500_1000"))}
        actual_observed = int(observed.sum())
        per_cell = {uid: int(observed[index].sum()) for index, uid in enumerate(identifiers)}
        passed = {"all169_unique_UIDs_and_order": len(identifiers) == len(set(identifiers)) == 169 and identifiers == expected_ids,
                  "all_observed_future_points": actual_observed == int(targets["future_observations"]),
                  "per_cell_observed_future_points": all(per_cell.get(row["cell_uid"]) == int(row["valid_future_observations"]) for row in qualification),
                  "both_rate_supports": actual_rates == {key: int(value) for key, value in targets["rate_support"].items()},
                  "early_qualification_count": int(eligible.sum()) == int(summary["input_eligible_cells"])}
        result = {**base, "status": "PASS" if all(passed.values()) else "FAIL", "loader_executed": True,
                  "checks": passed, "cells": len(identifiers), "eligible_cells": int(eligible.sum()),
                  "observed_future_points": actual_observed, "rate_support": actual_rates,
                  "query_points": len(query), "temporary_parent_verified": True,
                  "hardlinked_artifacts": linked, "copied_artifacts": copied}
        del loaded
    unchanged = (sha256(original_freeze) if original_freeze.exists() else None) == original_sha
    if not unchanged:
        raise RuntimeError("Real freeze certificate changed during the read-only loader probe")
    result.update(temporary_files_cleaned=not temporary.exists(), real_DATA_FREEZE_unchanged=True)
    return result


def selfcheck():
    source = Path(__file__).resolve().parent
    blocked = probe_training_loader(source, {})
    absent = probe_training_loader(source, {"status": "PASS", "formal_training_authorized_by_data_gate": True,
                                          "checks": [{"id": "engineering_guard_only", "status": "PASS"}]})
    assert blocked["status"] == absent["status"] == "PENDING"
    assert not blocked["loader_executed"] and not absent["loader_executed"]
    return {"status": "PASS", "scope": "engineering_guards_only", "checks": 2,
            "actual169_loader_probe_executed": False, "optimizer_updates": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(selfcheck()))
