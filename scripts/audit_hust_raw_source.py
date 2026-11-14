"""Audit every recovered official HUST cell using a non-executing pickle parser."""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from hust_safe_pickle import dataframe_columns, parse_pickle_data


CHARGE = "Constant current charge"
CV = "Constant current-constant voltage charge"
DISCHARGE = tuple(f"Constant current discharge_{index}" for index in range(4))
EXPECTED_PHASES = (CHARGE, CV, *DISCHARGE)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def summarize(values):
    return {"count": len(values), "min": float(min(values)), "median": float(np.median(values)), "max": float(max(values))} if values else {"count": 0}


def source_rates(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "DISCHARGE_RATES" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("DISCHARGE_RATES source mapping unavailable")


def audit_cell(member, processed_root, rates):
    path = Path(member["path"])
    if sha256(path) != member["sha256"]:
        raise ValueError("Recovered source member SHA256 mismatch")
    obj, globals_seen = parse_pickle_data(path.read_bytes())
    name = path.stem
    if list(obj) != [name]:
        raise ValueError("Source battery ID and member name differ")
    cell = obj[name]
    data = cell["data"]
    keys = sorted(data)
    row = {"family": "HUST", "cell_id": f"HUST_{name}.pkl", "original_device_record_id": name,
           "source_archive_member": member["member"], "source_member_sha256": member["sha256"],
           "source_member_crc32_verified": member["CRC32_verified"],
           "original_cycle_count": len(keys), "original_cycle_min": min(keys), "original_cycle_max": max(keys),
           "original_cycle_keys_contiguous": keys == list(range(min(keys), max(keys) + 1))}
    processed_path = processed_root / row["cell_id"]
    row["processed_file_sha256"] = sha256(processed_path)
    with processed_path.open("rb") as file:
        processed, _ = parse_pickle_data(file, capture_cycle_prefix=100)
    prefix = processed["cycle_data"]
    prefix_ids = [int(cycle["cycle_number"]) for cycle in prefix]
    voltage_exact = current_exact = 0
    for cycle, number in zip(prefix, prefix_ids):
        original = dataframe_columns(data[number])
        voltage_exact += int(np.array_equal(original["Voltage (V)"], np.asarray(cycle["voltage_in_V"])))
        current_exact += int(np.array_equal(original["Current (mA)"] / 1000, np.asarray(cycle["current_in_A"])))
    row.update(processed_prefix_records=len(prefix), processed_prefix_cycle_numbers=prefix_ids,
               processed_prefix_voltage_exact_cycles=voltage_exact, processed_prefix_current_exact_cycles=current_exact,
               processed_prefix_vi_exact=voltage_exact == current_exact == len(prefix),
               provided_prefix_starts_at_raw_cycle=prefix_ids[0], source_declared_skip_first2=name == "7-5")
    issues = Counter()
    phase_patterns = Counter()
    status_point_counts = Counter()
    stage_currents = defaultdict(list)
    complete_early = []
    starts = []
    all_columns = Counter()
    channel_ranges = {}
    numeric_nonfinite = Counter()
    expected_rates = dict(zip(DISCHARGE, [*rates[name], 1.0]))
    deviations = Counter()
    for number in keys:
        columns = dataframe_columns(data[number])
        all_columns[json.dumps(list(columns))] += 1
        numbers = columns["Cycle number"]
        issues["original_cycle_column_mismatch_cycles"] += int(not np.all(numbers == number))
        time_values = columns["Time (s)"]
        delta = np.diff(time_values)
        issues["negative_time_step_cycles"] += int(np.any(delta < 0))
        issues["repeated_time_steps"] += int(np.count_nonzero(delta == 0))
        starts.append(float(time_values[0]))
        for channel in ("Current (mA)", "Voltage (V)", "Capacity (mAh)", "Time (s)", "Cycle number"):
            values = columns[channel]
            numeric_nonfinite[channel] += int(np.count_nonzero(~np.isfinite(values)))
            finite = values[np.isfinite(values)]
            if finite.size:
                previous = channel_ranges.setdefault(channel, [float(finite.min()), float(finite.max())])
                previous[0] = min(previous[0], float(finite.min()))
                previous[1] = max(previous[1], float(finite.max()))
        status = columns["Status"]
        bounds = np.r_[0, np.flatnonzero(status[1:] != status[:-1]) + 1, len(status)]
        sequence = tuple(str(status[index]) for index in bounds[:-1])
        phase_patterns[json.dumps(sequence)] += 1
        complete = sequence == EXPECTED_PHASES
        issues["incomplete_or_changed_status_sequence_cycles"] += int(not complete)
        if number <= 100 and complete and np.all(numbers == number):
            complete_early.append(number)
        for left, right, phase in zip(bounds[:-1], bounds[1:], sequence):
            status_point_counts[phase] += int(right - left)
            current_median = float(np.median(columns["Current (mA)"][left:right])) / 1000
            if not np.isfinite(current_median):
                deviations[phase] += 1
                issues["nonfinite_stage_current_medians"] += 1
                continue
            stage_currents[phase].append(current_median)
            if phase == CHARGE:
                deviations[phase] += int(abs(current_median - 5 * 1.1) > 0.05 * 1.1)
            elif phase in expected_rates:
                deviations[phase] += int(abs(current_median + expected_rates[phase] * 1.1) > 0.05 * 1.1)
            elif phase == CV:
                deviations[phase] += int(current_median < 0 or current_median > 1.05 * 1.1)
    row.update(original_cycle_column_matches_key=issues["original_cycle_column_mismatch_cycles"] == 0,
               original_cycle_10_present=10 in data, original_cycle_50_present=50 in data, original_cycle_100_present=100 in data,
               raw_first100_cycles_with_complete_charge_discharge=len(complete_early),
               raw_first100_complete_cycle_numbers=complete_early, numeric_nonfinite_points=dict(numeric_nonfinite),
               original_cycle_relative_time_start=summarize(starts), original_time_is_calendar=False,
               raw_original_cycle_axis_verified=issues["original_cycle_column_mismatch_cycles"] == 0 and row["original_cycle_keys_contiguous"],
               raw_first100_complete_axis_and_phases=len(complete_early) == 100,
               raw_status_sequence_variants=len(phase_patterns), raw_stage_patterns=dict(phase_patterns),
               raw_control_median_deviation_cycles=dict(deviations), raw_issue_counts=dict(issues),
               schema_variants=[{"columns": json.loads(key), "cycles": count} for key, count in all_columns.items()])
    policy = {"family": "HUST", "cell_id": row["cell_id"],
              "official_design_evidence_url": "https://data.mendeley.com/datasets/nsc7hnsg4s/2",
              "official_design_charge_shared": True, "official_chamber_setpoint_C": 30,
              "chamber_setpoint_is_measured_cell_temperature": False,
              "charge_C_rates_declared": [5.0, 1.0], "discharge_C_rates_declared": [*rates[name], 1.0],
              "discharge_rate_source": "BatteryLife preprocess_HUST.py DISCHARGE_RATES, citing official paper supplementary table",
              "original_cycles_checked": len(keys), "status_sequence_variants": len(phase_patterns),
              "original_complete_status_cycles": len(keys) - issues["incomplete_or_changed_status_sequence_cycles"],
              "raw_stage_current_A_per_cycle_median_ranges": {key: summarize(value) for key, value in stage_currents.items()},
              "stage_median_deviation_cycles_tolerance_0_05C": dict(deviations),
              "future_policy_status_candidate": "known_fixed" if issues["incomplete_or_changed_status_sequence_cycles"] == 0 and not any(deviations.values()) else "observed_only",
              "status_candidate_scope": "Registered laboratory current schedule and observed stage medians, no guarantee of future actual temperature or controller trace",
              "future_controls_used_as_early_features": False}
    detail = {**row, "source_globals_held_symbolic": globals_seen, "source_top_keys": list(cell),
              "source_channel_ranges": channel_ranges, "source_status_point_counts": dict(status_point_counts),
              "source_stage_current_A_medians": {key: summarize(value) for key, value in stage_currents.items()},
              "untrusted_pickle_global_invocations": 0, "source_rul_dq_values_used_as_early_features_or_eol_event": False}
    return row, policy, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--protocol-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.source_root / "SOURCE_RECOVERY_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["actual_sha256"] != receipt["official_sha256"]:
        raise ValueError("Archive identity certificate failed")
    rates = source_rates(args.protocol_source)
    rows, policies = [], []
    members = receipt["members"][:args.limit] if args.limit else receipt["members"]
    start = time.monotonic()
    detail_path = args.source_root / "HUST_RAW_IDENTITY_AND_POLICY_DETAILS.jsonl"
    with detail_path.open("w", encoding="utf-8") as file:
        for index, member in enumerate(members, 1):
            row, policy, detail = audit_cell(member, args.processed_root, rates)
            rows.append(row)
            policies.append(policy)
            file.write(json.dumps(detail, ensure_ascii=False) + "\n")
            file.flush()
            gc.collect()
            print(json.dumps({"completed": index, "total": len(members), "cell": row["cell_id"],
                              "prefix_VI_exact": row["processed_prefix_vi_exact"],
                              "raw_axis_verified": row["raw_original_cycle_axis_verified"],
                              "first100_complete": row["raw_first100_complete_axis_and_phases"],
                              "elapsed_s": round(time.monotonic() - start, 1)}), flush=True)
    write_csv(args.output / "source_identity_alignment.csv", rows)
    write_csv(args.output / "policy_scheduled_evidence.csv", policies)
    summary = {"status": "COMPLETE_OFFICIAL_HUST_RAW_SOURCE_AUDIT", "cells": len(rows),
               "archive_sha256": receipt["actual_sha256"], "source_recovery_receipt_sha256": sha256(receipt_path),
               "source_cycle_total": sum(row["original_cycle_count"] for row in rows),
               "original_cycle_axes_verified_cells": sum(row["raw_original_cycle_axis_verified"] for row in rows),
               "prefix_VI_exact_cells": sum(row["processed_prefix_vi_exact"] for row in rows),
               "first100_complete_phase_cells": sum(row["raw_first100_complete_axis_and_phases"] for row in rows),
               "known_fixed_candidate_cells": sum(row["future_policy_status_candidate"] == "known_fixed" for row in policies),
               "unsafe_pickle_global_invocations": 0, "early_features_generated": 0, "rul_dq_decoded_as_labels": False,
               "full_lifecycle_controls_scope": "Cycle ID, stage sequence, finite counts and per-stage observed current medians for source/condition certification",
               "raw_time_coordinate": "Original source per-cycle relative seconds, calendar time unavailable",
               "source_detail_private_path": str(detail_path),
               "parser_sha256": sha256(Path(__file__).with_name("hust_safe_pickle.py")),
               "audit_script_sha256": sha256(Path(__file__)), "elapsed_seconds": round(time.monotonic() - start, 2)}
    (args.output / "HUST_SOURCE_AUDIT_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
