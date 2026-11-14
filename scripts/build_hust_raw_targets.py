"""Observed HUST future targets from the original net-capacity counter.

Raw trajectories remain private. Source pickle GLOBALs are held symbolically.
The provided-integral comparison uses the current D0's recorded capacity cache.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from hust_safe_pickle import dataframe_columns, parse_pickle_data
from physics_targets import TARGET_CONTRACT, future_targets


PHASES = ("Constant current discharge_0", "Constant current discharge_1",
          "Constant current discharge_2", "Constant current discharge_3")
NOMINAL_AH = 1.1
COUNTER_TOLERANCE_MAH = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {"count": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max()), "p05": float(np.quantile(values, .05)),
            "p95": float(np.quantile(values, .95))} if len(values) else {"count": 0}


def capacity_record(columns, cycle_number):
    status = columns["Status"]
    q = np.asarray(columns["Capacity (mAh)"], dtype=float)
    current_A = np.asarray(columns["Current (mA)"], dtype=float) / 1000
    time_s = np.asarray(columns["Time (s)"], dtype=float)
    boundaries = np.r_[0, np.flatnonzero(status[1:] != status[:-1]) + 1, len(status)]
    segments = {str(status[left]): (int(left), int(right)) for left, right in zip(boundaries[:-1], boundaries[1:])}
    row = {"raw_cycle_number": cycle_number, "raw_record_points": len(q),
           "discharge_capacity_Ah": None, "raw_soh_nominal": None, "raw_capacity_valid": False,
           "invalid_reason": "MISSING_DISCHARGE_PHASE", "phase_boundary_counter_changes_mAh": [],
           "counter_upward_steps": 0, "counter_large_drop_gt10pct_nominal_steps": 0,
           "nonfinite_discharge_points": 0, "negative_discharge_time_steps": 0,
           "raw_counter_start_mAh": None, "raw_counter_end_mAh": None,
           "discharge_current_integral_Ah": None, "raw_capacity_over_current_integral": None}
    if not all(phase in segments for phase in PHASES):
        return row
    starts = [segments[phase][0] for phase in PHASES]
    ends = [segments[phase][1] for phase in PHASES]
    if any(ends[index] != starts[index + 1] for index in range(3)):
        row["invalid_reason"] = "NONCONTIGUOUS_DISCHARGE_PHASES"
        return row
    left, right = starts[0], ends[-1]
    selected_q, selected_i, selected_t = q[left:right], current_A[left:right], time_s[left:right]
    row["discharge_points"] = right - left
    nonfinite = ~(np.isfinite(selected_q) & np.isfinite(selected_i) & np.isfinite(selected_t))
    row["nonfinite_discharge_points"] = int(np.count_nonzero(nonfinite))
    row["phase_boundary_counter_changes_mAh"] = [float(q[starts[index]] - q[ends[index-1] - 1]) for index in range(1, 4)]
    if np.any(nonfinite) or right - left < 15:
        row["invalid_reason"] = "NONFINITE_OR_INSUFFICIENT_DISCHARGE_POINTS"
        return row
    changes = np.diff(selected_q)
    delta_t = np.diff(selected_t)
    row["counter_upward_steps"] = int(np.count_nonzero(changes > COUNTER_TOLERANCE_MAH))
    row["counter_large_drop_gt10pct_nominal_steps"] = int(np.count_nonzero(changes < -0.1 * NOMINAL_AH * 1000))
    row["negative_discharge_time_steps"] = int(np.count_nonzero(delta_t < 0))
    row["raw_counter_start_mAh"] = float(selected_q[0])
    row["raw_counter_end_mAh"] = float(selected_q[-1])
    discharge_Ah = float(selected_q[0] - selected_q[-1]) / 1000
    row["discharge_capacity_Ah"] = discharge_Ah
    row["raw_soh_nominal"] = discharge_Ah / NOMINAL_AH
    sign_valid = all(float(np.median(current_A[a:b])) < 0 for a, b in zip(starts, ends))
    if not row["negative_discharge_time_steps"]:
        clipped = np.maximum(-selected_i, 0)
        integrated = float(np.sum((clipped[:-1] + clipped[1:]) * 0.5 * delta_t) / 3600)
        row["discharge_current_integral_Ah"] = integrated
        if integrated > 0:
            row["raw_capacity_over_current_integral"] = discharge_Ah / integrated
    reasons = []
    if row["counter_upward_steps"]:
        reasons.append("NONMONOTONE_DISCHARGE_COUNTER")
    if row["counter_large_drop_gt10pct_nominal_steps"]:
        reasons.append("LARGE_COUNTER_DISCONTINUITY_REQUIRES_REVIEW")
    if row["negative_discharge_time_steps"]:
        reasons.append("DECREASING_DISCHARGE_TIME")
    if not sign_valid:
        reasons.append("DISCHARGE_STATUS_CURRENT_SIGN_CONFLICT")
    if not 0 < row["raw_soh_nominal"] < 1.5:
        reasons.append("CAPACITY_NONPOSITIVE_OR_EXCEEDS_1_5_NOMINAL")
    row["invalid_reason"] = ";".join(reasons) if reasons else "AVAILABLE"
    row["raw_capacity_valid"] = not reasons
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--provided-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provided = {}
    for line in args.provided_cache.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["eligible"]["family"] == "HUST":
            trajectory = record["capacity_trajectory"]
            provided[record["eligible"]["cell_id"]] = dict(zip(trajectory["cycle_number"], trajectory["soh"]))
    if len(provided) != 77:
        raise ValueError("All 77 current provided-integral trajectories are required")
    recovery = json.loads((args.source_root / "SOURCE_RECOVERY_RECEIPT.json").read_text(encoding="utf-8"))
    targets, comparison, all_rows = [], [], []
    started = time.monotonic()
    trajectory_path = args.source_root / "raw_discharge_capacity_trajectory.csv"
    private_target_details = args.source_root / "RAW_CAPACITY_TARGET_DETAILS.jsonl"
    csv_fields = None
    with trajectory_path.open("w", encoding="utf-8-sig", newline="") as file, private_target_details.open("w", encoding="utf-8") as detail_file:
        writer = None
        for index, member in enumerate(recovery["members"], 1):
            path = Path(member["path"])
            if sha256(path) != member["sha256"]:
                raise ValueError("Raw member identity changed")
            obj, _ = parse_pickle_data(path.read_bytes())
            identifier = "HUST_" + path.stem
            source = obj[path.stem]["data"]
            rows = []
            for number in sorted(source):
                row = capacity_record(dataframe_columns(source[number]), number)
                row = {"cell_id": identifier, **row}
                supplied_soh = provided[identifier].get(number)
                row["provided_integral_soh"] = supplied_soh
                row["provided_integral_capacity_Ah"] = supplied_soh * NOMINAL_AH if supplied_soh is not None else None
                raw_q = row["discharge_capacity_Ah"]
                row["provided_over_raw_capacity"] = supplied_soh * NOMINAL_AH / raw_q if supplied_soh is not None and raw_q and raw_q > 0 else None
                row["provided_minus_raw_capacity_Ah"] = supplied_soh * NOMINAL_AH - raw_q if supplied_soh is not None and raw_q is not None else None
                rows.append(row)
                if writer is None:
                    csv_fields = list(row)
                    writer = csv.DictWriter(file, fieldnames=csv_fields)
                    writer.writeheader()
                writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
            file.flush()
            numbers = np.asarray([row["raw_cycle_number"] for row in rows])
            values = np.asarray([row["raw_soh_nominal"] if row["raw_soh_nominal"] is not None else np.nan for row in rows])
            valid = np.asarray([row["raw_capacity_valid"] for row in rows])
            target = future_targets(numbers, values, comparable=valid)
            target.update(cell_id=identifier, family="HUST", target_coordinate="original_official_cycle_number",
                          strict_physical_target_authenticated=True,
                          capacity_definition="raw_net_capacity_at_first_discharge_point_minus_last_discharge_point_div1000_then_div1_1Ah",
                          raw_capacity_valid_cycles=int(valid.sum()), raw_capacity_invalid_cycles=int((~valid).sum()))
            targets.append(target)
            ratios = [row["provided_over_raw_capacity"] for row in rows if row["raw_capacity_valid"] and row["provided_over_raw_capacity"] is not None]
            differences = [row["provided_minus_raw_capacity_Ah"] for row in rows if row["raw_capacity_valid"] and row["provided_minus_raw_capacity_Ah"] is not None]
            counts = Counter(row["invalid_reason"] for row in rows)
            review = {"cell_id": identifier, "family": "HUST", "original_cycles": len(rows),
                      "valid_raw_capacity_cycles": int(valid.sum()), "invalid_raw_capacity_cycles": int((~valid).sum()),
                      "invalid_reasons": dict(counts), "paired_capacity_records": len(ratios),
                      "provided_over_raw_ratio": distribution(ratios), "provided_minus_raw_Ah": distribution(differences),
                      "raw_over_current_integral": distribution([row["raw_capacity_over_current_integral"] for row in rows if row["raw_capacity_over_current_integral"] is not None]),
                      "upward_counter_steps": sum(row["counter_upward_steps"] for row in rows),
                      "large_counter_drop_gt10pct_nominal_steps": sum(row["counter_large_drop_gt10pct_nominal_steps"] for row in rows),
                      "boundary_counter_change_mAh": distribution([value for row in rows for value in row["phase_boundary_counter_changes_mAh"]]),
                      "raw_r200_500": target.get("r200_500"), "raw_r500_1000": target.get("r500_1000"),
                      "raw_EOL_event": target["eol_event"], "raw_EOL_censoring": target["eol_censoring"]}
            comparison.append(review)
            detail_file.write(json.dumps({"cell_id": identifier, "source_sha256": member["sha256"], "target": target, "capacity_review": review}, ensure_ascii=False) + "\n")
            detail_file.flush()
            all_rows.extend({"invalid_reason": row["invalid_reason"], "valid": row["raw_capacity_valid"]} for row in rows)
            del obj, source, rows
            gc.collect()
            print(json.dumps({"completed": index, "total": 77, "cell": identifier,
                              "raw_valid_cycles": review["valid_raw_capacity_cycles"],
                              "raw_capacity_ratio_median": review["provided_over_raw_ratio"].get("median"),
                              "eol_event": target["eol_event"], "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    write_csv(args.output / "HUST_RAW_CAPACITY_TARGET_SUPPORT.csv", targets)
    write_csv(args.output / "HUST_RAW_VS_PROVIDED_CAPACITY.csv", comparison)
    summary = {"status": "COMPLETE_ORIGINAL_RAW_COUNTER_TARGET_AUDIT", "cells": len(targets),
               "raw_cycle_rows": len(all_rows), "valid_raw_capacity_cycles": sum(row["valid"] for row in all_rows),
               "invalid_reasons": dict(Counter(row["invalid_reason"] for row in all_rows)),
               "N_mid_slope": sum(row.get("r200_500") is not None for row in targets),
               "N_late_slope": sum(row.get("r500_1000") is not None for row in targets),
               "N_EOL_observed_event": sum(row["eol_event"] is True for row in targets),
               "N_EOL_right_censored": sum(row["eol_censoring"] == "right" for row in targets),
               "N_knee_stable_candidate": sum(row["knee_stable_candidate"] for row in targets),
               "all_counter_boundary_changes_zero_within_tolerance": all(abs(row["boundary_counter_change_mAh"].get("min", 0)) <= COUNTER_TOLERANCE_MAH and abs(row["boundary_counter_change_mAh"].get("max", 0)) <= COUNTER_TOLERANCE_MAH for row in comparison),
               "per_cell_ratio_median_distribution": distribution([row["provided_over_raw_ratio"]["median"] for row in comparison if row["provided_over_raw_ratio"].get("count")]),
               "counter_monotonic_tolerance_mAh": COUNTER_TOLERANCE_MAH,
               "large_drop_review_threshold_fraction_nominal": 0.1, "SOH_capacity_denominator_Ah": NOMINAL_AH,
               "endpoint_sampling_scope": "First observed D0 sample through final observed D3 sample; unobserved boundary intervals are not extrapolated",
               "counter_origin_scope": "Capacity column from original authors' public record, not independent instrument calibration",
               "raw_private_trajectory": str(trajectory_path), "provided_cache_sha256": sha256(args.provided_cache),
               "target_contract": TARGET_CONTRACT,
               "target_code_sha256": sha256(Path(__file__).with_name("physics_targets.py")),
               "source_archive_sha256": recovery["actual_sha256"],
               "unsafe_pickle_global_invocations": 0, "rul_dq_used_as_labels": False,
               "early_features_or_models_fitted": False, "elapsed_seconds": round(time.monotonic() - started, 2)}
    (args.output / "HUST_RAW_CAPACITY_AUDIT_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
