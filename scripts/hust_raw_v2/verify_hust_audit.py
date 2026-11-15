"""Cross-check the new full-source audit with frozen source-derived labels."""

import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from physics_targets import endpoint_rate_target

OUT = REPO / "results" / "hust_raw_idea_audit_v2"
PRIVATE = REPO.parent / "private" / "raw_sources"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def main():
    audit = json.loads((OUT / "HUST_RAW_AUDIT.json").read_text(encoding="utf-8"))
    current = read_csv(PRIVATE / "HUST_IDEA_V2" / "all_cycle_quality.csv")
    old_trajectory_path = PRIVATE / "HUST" / "raw_discharge_capacity_trajectory.csv"
    old = {(r["cell_id"], int(r["raw_cycle_number"])): float(r["discharge_capacity_Ah"]) for r in read_csv(old_trajectory_path)}
    cycles = defaultdict(list)
    max_error = 0.0
    failures = []
    for row in current:
        key = row["cell_id"], int(row["cycle"])
        if key not in old:
            failures.append(["MISSING_OLD_CAPACITY_KEY", list(key)])
            continue
        value = float(row["net_discharge_mAh"]) / 1000
        error = abs(value - old[key])
        max_error = max(max_error, error)
        if error > 1e-12:
            failures.append(["OLD_CAPACITY_DISAGREES", list(key), error])
        cycles[key[0]].append((key[1], value / 1.1))
    old_summary_path = REPO / "results" / "d0" / "HUST_RAW_CAPACITY_AUDIT_SUMMARY.json"
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    target_path = REPO / "results" / "d0" / "HUST_RAW_CAPACITY_TARGET_SUPPORT.csv"
    targets = read_csv(target_path)
    target_checks = []
    for row in targets:
        xy = np.asarray(sorted(cycles[row["cell_id"]]))
        for name, first, last in (("r200_500", 200, 500), ("r500_1000", 500, 1000)):
            result = endpoint_rate_target(xy[:, 0], xy[:, 1], first, last)
            difference = abs(result["value"] - float(row[name]))
            target_checks.append({"cell_id": row["cell_id"], "target": name, "recomputed": result["value"], "frozen": float(row[name]), "absolute_difference": difference, "available": result["available"]})
            if difference > 1e-14 or not result["available"]:
                failures.append(["OLD_TARGET_DISAGREES", row["cell_id"], name, difference])
    trajectory_quality = []
    for cell_id, values in cycles.items():
        xy = np.asarray(sorted(values))
        delta = np.diff(xy[:, 1])
        trajectory_quality.append({"cell_id": cell_id, "cycles": len(xy), "first_SOH": float(xy[0, 1]), "last_SOH": float(xy[-1, 1]),
            "min_SOH": float(xy[:, 1].min()), "max_SOH": float(xy[:, 1].max()),
            "adjacent_capacity_increases_gt1e_6mAh": int((delta > 1e-6 / 1100).sum()),
            "adjacent_capacity_decreases_gt1e_6mAh": int((delta < -1e-6 / 1100).sum()),
            "adjacent_capacity_ties_within1e_6mAh": int((np.abs(delta) <= 1e-6 / 1100).sum()),
            "largest_single_cycle_SOH_increase": float(delta.max()), "largest_single_cycle_SOH_drop": float(-delta.min()),
            "first100_capacity_increases_gt1e_6mAh": int((delta[:99] > 1e-6 / 1100).sum()),
            "first100_SOH_min": float(xy[:100, 1].min()), "first100_SOH_max": float(xy[:100, 1].max())})
    with (OUT / "degradation_trajectory_quality.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory_quality[0]))
        writer.writeheader()
        writer.writerows(trajectory_quality)
    with (OUT / "future_target_replay.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(target_checks[0]))
        writer.writeheader()
        writer.writerows(target_checks)
    source_code = Path(r"D:\develop\patchtst\BatteryLife\process_scripts\preprocess_HUST.py")
    tree = ast.parse(source_code.read_text(encoding="utf-8"))
    rates = None
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(v, ast.Name) and v.id == "DISCHARGE_RATES" for v in statement.targets):
            rates = ast.literal_eval(statement.value)
    if rates is None:
        raise ValueError("Expected declared HUST rate table")
    phase_rows = read_csv(OUT / "phase_quality.csv")
    rate_rows, phase_global = [], {}
    closure = read_csv(OUT / "capacity_integration_closure.csv")
    windows = read_csv(OUT / "early_window_support.csv")
    window_current = {}
    for view in dict.fromkeys(row["view"] for row in windows):
        selected = [row for row in windows if row["view"] == view]
        deviations = [float(row["current_max_abs_deviation_from_declared_mA"]) for row in selected]
        window_current[view] = {"prefix_cycles": len(selected), "maximum_single_point_current_deviation_mA": max(deviations),
            "all_point_current_within0p05C_cycles": sum(value <= 55 for value in deviations),
            "unbracketed_source_cycles": [[row["cell_id"], int(row["raw_cycle"])] for row in selected if row["phase_brackets_both_endpoints"] != "True"]}
    phase_names = list(dict.fromkeys(r["phase"] for r in phase_rows))
    for phase in phase_names:
        selected = [r for r in phase_rows if r["phase"] == phase]
        dt = Counter()
        for row in selected:
            dt.update({int(k): v for k, v in json.loads(row["dt"]).items()})
        phase_global[phase] = {"cycles": sum(int(r["cycles"]) for r in selected), "points": sum(int(r["points"]) for r in selected),
            "current_min_mA": min(float(r["current_min"]) for r in selected), "current_max_mA": max(float(r["current_max"]) for r in selected),
            "cycle_median_min_mA": min(json.loads(r["current_cycle_median_stats_mA"])["min"] for r in selected),
            "cycle_median_max_mA": max(json.loads(r["current_cycle_median_stats_mA"])["max"] for r in selected),
            "voltage_min_V": min(float(r["voltage_min"]) for r in selected), "voltage_max_V": max(float(r["voltage_max"]) for r in selected),
            "capacity_direction_violations": sum(int(r["q_wrong_direction"]) for r in selected),
            "voltage_direction_violations": sum(int(r["v_wrong_direction"]) for r in selected),
            "voltage_direction_violations_gt10mV": sum(int(r["v_wrong_gt10mV"]) for r in selected), "within_phase_dt_counts_s": dict(sorted(dt.items()))}
    for row in phase_rows:
        phase = row["phase"]
        if not phase.startswith("Constant current discharge_"):
            continue
        step = int(phase[-1])
        cell_key = row["cell_id"].removeprefix("HUST_")
        declared_C = float(rates[cell_key][step]) if step < 3 else 1.0
        stats = json.loads(row["current_cycle_median_stats_mA"])
        observed = -stats["median"] / 1100
        rate_rows.append({"cell_id": row["cell_id"], "phase": phase, "BatteryLife_declared_C": declared_C,
            "observed_cycle_median_C": observed, "observed_min_cycle_median_C": -stats["max"] / 1100,
            "observed_max_cycle_median_C": -stats["min"] / 1100, "deviation_gt0p05C": abs(observed - declared_C) > .05,
            "paper_terminal_C": 1 if step == 3 else "individual protocol in source table"})
    with (OUT / "declared_vs_observed_protocol.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rate_rows[0]))
        writer.writeheader()
        writer.writerows(rate_rows)
    closure_global = []
    for phase in phase_names:
        for grid in ("recorded_dt", "fixed_5s"):
            for rule in ("left", "right", "trapezoid"):
                selected = [r for r in closure if r["phase"] == phase and r["time_grid"] == grid and r["quadrature"] == rule]
                n = sum(int(r["steps"]) for r in selected)
                closure_global.append({"phase": phase, "time_grid": grid, "quadrature": rule, "steps": n,
                    "residual_MAE_mAh": sum(float(r["residual_MAE_mAh"]) * int(r["steps"]) for r in selected) / n,
                    "residual_RMSE_mAh": np.sqrt(sum(float(r["residual_RMSE_mAh"]) ** 2 * int(r["steps"]) for r in selected) / n),
                    "residual_mean_mAh": sum(float(r["residual_mean_mAh"]) * int(r["steps"]) for r in selected) / n,
                    "residual_max_abs_mAh": max(float(r["residual_max_abs_mAh"]) for r in selected),
                    "within_1e_9_mAh": sum(int(r["within_1e_9_mAh"]) for r in selected),
                    "within_1e_6_mAh": sum(int(r["within_1e_6_mAh"]) for r in selected),
                    "within_1e_3_mAh": sum(int(r["within_1e_3_mAh"]) for r in selected)})
    with (OUT / "capacity_integration_closure_global.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(closure_global[0]))
        writer.writeheader()
        writer.writerows(closure_global)
    check = {"status": "PASS" if not failures else "FAIL", "failures": failures, "current_members": audit["cells"], "current_cycle_rows": len(current),
        "old_cycle_rows": len(old), "all_current_old_cycle_keys_equal": set(old) == {(r["cell_id"], int(r["cycle"])) for r in current},
        "all_current_capacity_values_match_old": max_error <= 1e-12, "maximum_capacity_difference_Ah": max_error,
        "archive_hash_matches_old_target_summary": audit["current_archive_identity"]["inner_sha256"] == old_summary["source_archive_sha256"],
        "future_rate_values_replayed": len(target_checks), "max_future_rate_absolute_difference": max(r["absolute_difference"] for r in target_checks),
        "old_target_support_sha256": sha256(target_path), "old_summary_sha256": sha256(old_summary_path), "old_private_trajectory_sha256": sha256(old_trajectory_path),
        "phase_aggregate": phase_global, "protocol_deviation_counts": {"first_three_steps": sum(r["deviation_gt0p05C"] for r in rate_rows if r["phase"][-1] != "3"),
            "terminal_step": sum(r["deviation_gt0p05C"] for r in rate_rows if r["phase"][-1] == "3")},
        "early_window_all_point_current_support": window_current,
        "charge_capacity_starts_exactly_zero_cycles": sum(float(r["charge_first_capacity_mAh"]) == 0 for r in current),
        "charge_time_starts_exactly_zero_cycles": sum(float(r["charge_first_time_s"]) == 0 for r in current),
        "phase_boundary_gap_min_s": min(float(r["boundary_gap_min_s"]) for r in current),
        "phase_boundary_gap_max_s": max(float(r["boundary_gap_max_s"]) for r in current),
        "exact_observed_EOL80_events_cells": sum(any(y <= .8 for _, y in values) for values in cycles.values()),
        "exact_observed_EOL80_right_censored_cells": sum(all(y > .8 for _, y in values) for values in cycles.values()),
        "raw_SOH_min": min(y for values in cycles.values() for _, y in values), "raw_SOH_max": max(y for values in cycles.values() for _, y in values),
        "cross_cycle_capacity_increase_steps_gt1e_6mAh": sum(r["adjacent_capacity_increases_gt1e_6mAh"] for r in trajectory_quality),
        "cells_with_cross_cycle_capacity_increase_steps": sum(r["adjacent_capacity_increases_gt1e_6mAh"] > 0 for r in trajectory_quality),
        "true_physical_knee_annotation_count": 0, "raw_cycle_axis_matches_column": audit["counts"]["cycle_column_mismatch_cycles"] == 0,
        "measurement_independence_certified": False, "source_capture_algorithm_certified": False,
        "future_protocol_known_at_prediction_time_certified": False, "model_training_or_scoring_executed": False}
    if not check["all_current_old_cycle_keys_equal"] or not check["archive_hash_matches_old_target_summary"]:
        check["status"] = "FAIL"
    (OUT / "FULL_AUDIT_VERIFICATION.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in check.items() if k not in ("phase_aggregate",)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
