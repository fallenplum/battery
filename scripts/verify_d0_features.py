"""Regression checks for prefix causality, phase direction and D0 cache binding.

The original cell is read once and remains read-only. Counterfactual suffixes
are injected in memory at the pickle-loader boundary, with an explicit test
identity. This separates numerical causality from whole-file authentication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import run_d0


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inspect_memory(payload, source_path, dataset_root):
    original_loader = run_d0.load_existing_pickle
    test_identity = "in_memory_counterfactual_authentication_boundary"
    run_d0.load_existing_pickle = lambda _: (payload, test_identity)
    try:
        return run_d0.inspect_cell(source_path, dataset_root, {"sha256": test_identity}, set())
    finally:
        run_d0.load_existing_pickle = original_loader


def feature_values(result, prefix):
    row = next(row for row in result["features"] if row["prefix_cutoff_supplied_cycle_number"] == prefix)
    values = {key: value for key, value in row.items() if value is None or isinstance(value, (int, float, bool))}
    json.dumps(values, allow_nan=False)
    return values


def perturb_payload(payload, prefix, mode):
    result = dict(payload)
    early = [cycle for cycle in payload["cycle_data"] if int(cycle["cycle_number"]) <= prefix]
    suffix = [cycle for cycle in payload["cycle_data"] if int(cycle["cycle_number"]) > prefix]
    if mode == "delete":
        result["cycle_data"] = early
        return result
    if mode == "reverse_records":
        result["cycle_data"] = early + list(reversed(suffix))
        return result
    changed = []
    for cycle in suffix:
        copy = dict(cycle)
        for key in ("current_in_A", "voltage_in_V", "charge_capacity_in_Ah", "discharge_capacity_in_Ah", "time_in_s"):
            array = np.asarray(cycle[key])
            copy[key] = np.full(array.shape, 1e6, dtype=np.float64)
        changed.append(copy)
    result["cycle_data"] = early + changed
    return result


def direction_cycle(direction):
    count = 101
    charge = np.linspace(0, 0.2, count)
    is_charge = direction == "charge"
    return {
        "cycle_number": 1,
        "voltage_in_V": np.linspace(3.0, 3.5, count) if is_charge else np.linspace(3.5, 3.0, count),
        "current_in_A": np.full(count, 1.0 if is_charge else -1.0),
        "charge_capacity_in_Ah": charge if is_charge else np.zeros(count),
        "discharge_capacity_in_Ah": np.zeros(count) if is_charge else charge,
    }


def cache_regressions(source_path, source_digest, output_parent):
    checks = []
    with tempfile.TemporaryDirectory(prefix="d0_feature_regression_", dir=output_parent) as temporary_name:
        temporary = Path(temporary_name).resolve()
        if not temporary.is_relative_to(output_parent.resolve()):
            raise RuntimeError("Temporary verification directory escaped the selected output parent")
        dataset = temporary / "dataset"
        cell = dataset / "HUST" / source_path.name
        cell.parent.mkdir(parents=True)
        shutil.copyfile(source_path, cell)
        audit = temporary / "audit"
        audit.mkdir()
        (audit / "standardized_cells.jsonl").write_text(json.dumps({"relative_path": "HUST/" + source_path.name, "sha256": source_digest}) + "\n", encoding="utf-8")
        legacy = temporary / "legacy.json"
        legacy.write_text('{"cells": {}}', encoding="utf-8")
        cache = temporary / "cache"
        command = [sys.executable, str(Path(run_d0.__file__).resolve()),
                   "--dataset-root", str(dataset), "--prior-audit", str(audit),
                   "--legacy-manifest", str(legacy), "--output", str(temporary / "output"),
                   "--private-cache", str(cache)]
        initial = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        checks.append({"name": "real_single_cell_cache_fixture", "pass": initial.returncode == 0,
                       "returncode": initial.returncode, "error": initial.stderr[-1500:]})
        if initial.returncode:
            return checks
        signature_path = cache / "RUN_SIGNATURE.json"
        original_signature = signature_path.read_text(encoding="utf-8")
        signature = json.loads(original_signature)
        signature["target_code_sha256"] = "0" * 64
        signature_path.write_text(json.dumps(signature), encoding="utf-8")
        stale_code = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        checks.append({"name": "stale_code_cache_rejected", "pass": stale_code.returncode != 0 and "Cache is bound to different code or data" in stale_code.stderr,
                       "returncode": stale_code.returncode})
        signature_path.write_text(original_signature, encoding="utf-8")
        cache_path = cache / "d0_cells.jsonl"
        cached = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])
        cached["eligible"]["provided_file_sha256"] = "0" * 64
        cache_path.write_text(json.dumps(cached, allow_nan=False) + "\n", encoding="utf-8")
        stale_source = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        checks.append({"name": "cached_source_sha_mismatch_rejected", "pass": stale_source.returncode != 0 and "Cached source hash differs" in stale_source.stderr,
                       "returncode": stale_source.returncode})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-cache-check", action="store_true")
    args = parser.parse_args()
    source_path = args.cell_file.resolve()
    if source_path.parent.name != "HUST":
        raise ValueError("This real-data regression requires an existing HUST cell")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = source_path.parent.parent
    before = source_path.stat()
    payload, source_digest = run_d0.load_existing_pickle(source_path)
    baseline = inspect_memory(payload, source_path, dataset_root)
    checks = []
    for prefix in (10, 50, 100):
        original = feature_values(baseline, prefix)
        for mode in ("extreme_values", "reverse_records", "delete"):
            try:
                changed = inspect_memory(perturb_payload(payload, prefix, mode), source_path, dataset_root)
                actual = feature_values(changed, prefix)
                checks.append({"name": f"K{prefix}_suffix_{mode}_invariance", "pass": original == actual,
                               "numeric_feature_columns": len(original),
                               "differing_keys": [key for key in set(original) | set(actual) if original.get(key) != actual.get(key)]})
            except Exception as error:
                checks.append({"name": f"K{prefix}_suffix_{mode}_invariance", "pass": False, "error": f"{type(error).__name__}: {error}"})
    missing = dict(payload)
    missing["cycle_data"] = [cycle for cycle in payload["cycle_data"] if int(cycle["cycle_number"]) not in (1, 2)]
    missing_result = inspect_memory(missing, source_path, dataset_root)
    eligible = missing_result["eligible"]
    checks.append({"name": "missing_cycles_1_2_do_not_certify_complete_K100",
                   "pass": eligible["prefix_K100_records"] == 98 and not eligible["prefix_K100_numbers_complete"] and not eligible["physical_cycle_source_supported"],
                   "retained_prefix_records": eligible["prefix_K100_records"],
                   "complete": eligible["prefix_K100_numbers_complete"],
                   "source_supported": eligible["physical_cycle_source_supported"]})
    for direction in ("charge", "discharge"):
        curve, reason, _ = run_d0.observed_curve(direction_cycle(direction), {"direction": direction, "voltage_window": [3.1, 3.4]}, 1.0, 1.0)
        valid = bool(curve is not None and np.isfinite(curve).all() and np.isclose(curve[0], 0) and np.isclose(curve[-1], 0.12) and np.all(np.diff(curve) > 0))
        checks.append({"name": direction + "_oriented_window_capacity_sign", "pass": valid, "reason": reason})
    reset = direction_cycle("charge")
    reset["charge_capacity_in_Ah"] = reset["charge_capacity_in_Ah"].copy()
    reset["charge_capacity_in_Ah"][50:] -= 0.10
    curve, reason, _ = run_d0.observed_curve(reset, {"direction": "charge", "voltage_window": [3.1, 3.4]}, 1.0, 1.0)
    checks.append({"name": "single_large_Q_counter_reset_rejected", "pass": curve is None and reason == "capacity_reset_or_reversal", "reason": reason})
    wrong_phase = direction_cycle("charge")
    wrong_phase["discharge_capacity_in_Ah"] = np.linspace(0, 1.0, 101)
    wrong_capacity = run_d0.capacity(wrong_phase, "discharge_capacity_in_Ah", 1.0)
    checks.append({"name": "missing_discharge_phase_rejects_positive_Qd_column", "pass": wrong_capacity is None, "value": wrong_capacity})
    # Document the residual assumption that the supplied capacity column has the
    # intended stage semantics, in addition to the verified phase-presence gate.
    contaminated = direction_cycle("discharge")
    contaminated["current_in_A"] = np.r_[contaminated["current_in_A"], np.ones(101)]
    contaminated["discharge_capacity_in_Ah"] = np.r_[contaminated["discharge_capacity_in_Ah"], np.full(101, 1.2)]
    contaminated_value = run_d0.capacity(contaminated, "discharge_capacity_in_Ah", 1.0)
    phase_diagnostic = {"name": "inactive_phase_capacity_column_contamination", "correct_active_phase_span": 0.2,
                        "returned_value": contaminated_value,
                        "rejected_or_active_phase_only": bool(contaminated_value is None or np.isclose(contaminated_value, 0.2)),
                        "scope": "Malformed capacity-column counterexample. The current dataset requires an independent source-column semantics check."}
    if not args.skip_cache_check:
        checks.extend(cache_regressions(source_path, source_digest, args.output.parent))
    after = source_path.stat()
    checks.append({"name": "original_source_bytes_and_stats_unchanged",
                   "pass": before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns and file_sha256(source_path) == source_digest})
    result = {"status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
              "source_file_name": source_path.name, "source_sha256": source_digest,
              "scanner_sha256": file_sha256(Path(run_d0.__file__).resolve()),
              "checks": checks, "additional_phase_semantics_diagnostic": phase_diagnostic,
              "interpretation": "Prefix numerical causality and listed regression checks only. Full physical-cycle, original-device and future-policy certification are separate."}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [row["name"] for row in checks if not row["pass"]],
                      "phase_contamination_rejected": phase_diagnostic["rejected_or_active_phase_only"]}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
