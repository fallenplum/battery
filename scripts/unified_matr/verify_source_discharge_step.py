"""Read-only source-protocol audit of completed final native-phase records."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


LOCATION = ("cell_uid", "source_segment_id", "source_record_index", "run_index", "instrument_cycle_id",
            "original_cycle_id", "source_row_start", "source_row_stop_exclusive", "discharge_step_index",
            "discharge_point_start", "discharge_point_stop_exclusive", "constant_current_point_start",
            "constant_current_point_stop_exclusive", "counter_endpoint_point_index", "measurement_phase",
            "instrument_discharge_counter_zero_Ah", "discharge_current_median_A", "discharge_current_min_A",
            "discharge_current_max_A", "discharge_start_voltage_V", "discharge_end_voltage_V", "capacity_Ah",
            "adjacent_stage_counter_increment_Ah", "time_valid", "integration_valid", "failure_reasons")
NUMERIC = ("instrument_discharge_counter_zero_Ah", "discharge_current_median_A", "discharge_current_min_A",
           "discharge_current_max_A", "discharge_start_voltage_V", "discharge_end_voltage_V", "capacity_Ah",
           "adjacent_stage_counter_increment_Ah")


def sha256(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def main_steps(schedule):
    result = {}
    for source in schedule["source_links"]:
        rows = [step for step in source["discharge_labeled_steps"] if step["label"] == "Discharge"
                and step["control_type"] == "CCCV" and float(step["control_value_original"]) == -4.4
                and float(step["ext_control_value1_original"]) == 2.0]
        if len(rows) != 1:
            raise ValueError("Original source must have exactly one declared main 4C discharge")
        definition = schedule["unique_schedules_by_sha256"][source["schedule_sha256"]]
        all_steps = {int(step["source_step_index"]): step for step in definition["steps"]}
        result[source["source_segment_id"]] = {**rows[0], "schedule_sha256": source["schedule_sha256"],
                                                "schedule_file_id": source["schedule_file_id"],
                                                "all_source_step_definitions": all_steps}
    if len(result) != 174:
        raise ValueError("Every fixed-cohort source segment requires a main discharge definition")
    return result


def snapshot(args, expected, settings, cache):
    files = []
    for path in sorted(args.cache.glob("*_audit.parquet")):
        metadata_path = args.cache / (path.name.removesuffix("_audit.parquet") + ".json")
        if metadata_path.is_file():
            files.append(path)
    for path in files:
        if path.name not in cache:
            frame = pd.read_parquet(path)
            cache[path.name] = (frame, sha256(path))
    frames = [cache[path.name][0] for path in files]
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frames:
        return {"status": "WAITING_FOR_COMPLETED_CELL_CACHE", "completed_cells": 0}
    regular = all_rows["record_role"].eq("regular_cycle")
    selected = all_rows.loc[regular & all_rows["capacity_valid"]].copy()
    selected["expected_source_step"] = selected["source_segment_id"].map(lambda segment: expected[segment]["source_step_index"])
    exact_step = selected["discharge_step_index"].eq(selected["expected_source_step"])
    nominal = settings["nominal_capacity_Ah"] * settings["design_rate_C"]
    current_tolerance = nominal * settings["current_tolerance_fraction"]
    numeric_checks = {"all_stored_main_phase_numeric_fields_finite": np.isfinite(selected[list(NUMERIC)]).all(axis=1),
                      "counter_zero_within_registered_tolerance": selected["instrument_discharge_counter_zero_Ah"].abs().le(settings["capacity_zero_tolerance_Ah"]),
                      "median_current_within_registered_4C_tolerance": (selected["discharge_current_median_A"] + nominal).abs().le(current_tolerance),
                      "negative_excitation_within_registered_4C_tolerance": selected["discharge_current_min_A"].ge(-nominal - current_tolerance),
                      "positive_current_within_registered_noise_tolerance": selected["discharge_current_max_A"].le(settings["minimum_discharge_current_A"]),
                      "start_voltage_within_registered_full_phase_requirement": selected["discharge_start_voltage_V"].ge(settings["full_discharge_min_start_voltage_V"]),
                      "end_voltage_within_registered_full_phase_requirement": selected["discharge_end_voltage_V"].le(settings["full_discharge_max_end_voltage_V"]),
                      "capacity_within_registered_instrument_range": selected["capacity_Ah"].gt(0) & selected["capacity_Ah"].le(1.5 * settings["nominal_capacity_Ah"])}
    step_mismatches = []
    for record in selected.loc[~exact_step, list(LOCATION) + ["expected_source_step"]].to_dict("records"):
        step = expected[record["source_segment_id"]]
        record.update(expected_label=step["label"], expected_control_type=step["control_type"],
                      expected_control_value_original=step["control_value_original"],
                      expected_ext_control_value1_original=step["ext_control_value1_original"],
                      schedule_sha256=step["schedule_sha256"])
        observed = step["all_source_step_definitions"].get(int(record["discharge_step_index"]), {})
        record.update(observed_index_declared_label=observed.get("label"),
                      observed_index_declared_control_type=observed.get("control_type"),
                      observed_index_declared_control_value_original=observed.get("control_value_original"),
                      observed_index_declared_ext_control_value1_original=observed.get("ext_control_value1_original"))
        step_mismatches.append(record)
    numeric_failures = []
    for name, passed in numeric_checks.items():
        for record in selected.loc[~passed, list(LOCATION)].to_dict("records"):
            numeric_failures.append({"check": name, **record})
    source_counts = []
    for source, rows in selected.groupby("source_segment_id", sort=True):
        source_counts.append({"source_segment_id": source, "capacity_valid_regular_records": len(rows),
                              "expected_main_discharge_step": expected[source]["source_step_index"],
                              "observed_main_phase_steps": dict(Counter(int(value) for value in rows["discharge_step_index"])),
                              "exact_step_matches": int(rows["discharge_step_index"].eq(rows["expected_source_step"]).sum())})
    ranges = {name: {"min": float(selected[name].min()), "max": float(selected[name].max()),
                     "finite_count": int(np.isfinite(selected[name]).sum())} for name in NUMERIC}
    completed = len(files) == 169
    step_pass = bool(exact_step.all())
    numeric_pass = all(bool(check.all()) for check in numeric_checks.values())
    status = "IN_PROGRESS" if step_pass else "IN_PROGRESS_STAGE_IDENTITY_EXCEPTIONS"
    if completed:
        status = "PASS" if step_pass else "COMPLETE_WITH_SOURCE_STEP_LABEL_EXCEPTIONS"
        if not numeric_pass:
            status = "FAIL_REGISTERED_NUMERIC_REQUIREMENTS"
    result = {"status": status, "coverage_complete": completed,
              "required_physical_cells": 169, "completed_cells": len(files), "required_source_segments": 174,
              "audited_source_segments": int(all_rows["source_segment_id"].nunique()),
              "all_native_records": len(all_rows), "all_regular_cycle_records": int(regular.sum()),
              "capacity_valid_regular_records": len(selected),
              "capacity_invalid_regular_records": int((regular & ~all_rows["capacity_valid"]).sum()),
              "nonregular_records": int((~regular).sum()),
              "capacity_valid_nonregular_records": int((~regular & all_rows["capacity_valid"]).sum()),
              "exact_source_step_match_records": int(exact_step.sum()),
              "source_step_mismatch_records": len(step_mismatches),
              "source_step_mismatch_cells": len({row["cell_uid"] for row in step_mismatches}),
              "source_step_mismatch_prefix_records": sum(row["original_cycle_id"] <= 100 for row in step_mismatches),
              "stage_mismatch_pairs": dict(Counter(f"expected{int(row['expected_source_step'])}_observed{int(row['discharge_step_index'])}" for row in step_mismatches)),
              "mismatched_observed_index_declared_labels": dict(Counter(row["observed_index_declared_label"] for row in step_mismatches)),
              "checks": {"all174_source_protocol_main_discharge_definitions_unique": len(expected) == 174,
                         "all_capacity_valid_records_are_regular": not bool((~regular & all_rows["capacity_valid"]).any()),
                         "all_capacity_valid_regular_steps_equal_original_protocol": step_pass,
                         **{name: bool(check.all()) for name, check in numeric_checks.items()}},
              "registered_numeric_requirements": settings, "numeric_ranges": ranges,
              "per_source_coverage": source_counts,
              "stage_identity_exceptions": step_mismatches, "registered_numeric_requirement_exceptions": numeric_failures,
              "audited_completed_cache_files": [{"file_name": path.name, "sha256": cache[path.name][1], "records": len(cache[path.name][0])} for path in files],
              "schedule_metadata_audit_sha256": sha256(args.schedule), "registered_config_sha256": sha256(args.config),
              "original_source_bindings_sha256": sha256(args.schedule.parent / "instrument_bindings.json"),
              "whole_native_source_SHA_validation_artifact_sha256": sha256(args.schedule.parent / "native_channel_audit_QA.json"),
              "whole_native_source_files_rehashed_by_this_script": False,
              "cache_recipe_sha256": sha256(args.cache / "RECIPE.json"), "code_sha256": sha256(Path(__file__)),
              "interpretation": "Actual phase certification uses recorded I/V/Q with the original CCCV(-4.4A,2V) source definition and existing registered measurement requirements. Exact source Step_Index differences remain explicit warnings. No claim that all original Step labels are correct. A three-record native replay supports a source-label-lag interpretation; original control-system causation remains unverified.",
              "qualification_label_model_or_measurement_changes": 0, "model_fits": 0}
    replay_path = args.schedule.parent / "SOURCE_DISCHARGE_STEP_NATIVE_REPLAY.json"
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        result["native_counterexample_replay_sha256"] = sha256(replay_path)
        result["native_counterexample_replay_status"] = replay["status"]
        result["native_counterexample_replay_records"] = len(replay["examples"])
        result["native_counterexample_all_replayed_quantities_equal_stored_fields"] = all(row["all_replayed_quantities_equal_stored_fields"] for row in replay["examples"])
        result["native_counterexample_records"] = replay["examples"]
        result["all6780_raw_exception_records_individually_replayed"] = False
    if args.final_audit is not None and args.final_audit.is_file() and completed:
        final = pd.read_parquet(args.final_audit)
        expected_all = all_rows.sort_values(["cell_uid", "source_segment_id", "source_record_index"]).reset_index(drop=True)
        actual_all = final[expected_all.columns].sort_values(["cell_uid", "source_segment_id", "source_record_index"]).reset_index(drop=True)
        result["final_audit_matches_completed_cache_all_fields"] = actual_all.equals(expected_all)
        result["final_audit_sha256"] = sha256(args.final_audit)
        result["final_audit_records"] = len(final)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("../private/unified_matr_seed42/cell_cache_v4_final_native_phases"))
    parser.add_argument("--schedule", type=Path, default=Path("results/unified_matr/schedule_metadata_audit.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/unified_matr_seed42.json"))
    parser.add_argument("--output", type=Path, default=Path("results/unified_matr/SOURCE_DISCHARGE_STEP_QA.json"))
    parser.add_argument("--final-audit", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))["discharge"]
    expected = main_steps(json.loads(args.schedule.read_text(encoding="utf-8")))
    cache = {}
    previous = None
    while True:
        result = snapshot(args, expected, settings, cache)
        write_json(args.output, result)
        progress = {key: result.get(key) for key in ("status", "completed_cells", "capacity_valid_regular_records", "source_step_mismatch_records", "source_step_mismatch_prefix_records")}
        if progress != previous:
            print(json.dumps(progress), flush=True)
            previous = progress
        if not args.watch or result.get("coverage_complete"):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
