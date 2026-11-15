"""Independent native-CSV DATA-FREEZE verification, without fitting models."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_utils import channel_identity, json_value, policy_context, sha256, supplied_identity_index, write_json
from instrument_csv import iter_instrument_cycles
from measurements import measure_cycle
from physics_features import extract_prefix_features
from scan_instrument_restarts import scan_native_index_runs
from verify_instrument_csv import run_synthetic_checks
from verify_physics_features import run_checks


REPOSITORY = Path(__file__).resolve().parents[2]
ARTIFACTS = (
    "dataset_manifest.json", "cells.parquet", "early_viq_qv_64.npz",
    "early_physics_features.parquet", "future_soh.parquet", "raw_record_audit.parquet",
    "instrument_cycle_mapping.parquet", "instrument_identity_bindings.json", "split_contract.json",
)
FEATURE_KEYS = ("global_features", "global_mask", "local_residual", "local_mask", "local_features", "local_feature_mask")
CONTINUATIONS = {"b2c7": "b1c0", "b2c8": "b1c1", "b2c9": "b1c2", "b2c15": "b1c3", "b2c16": "b1c4"}


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, condition, details):
        self.checks.append({"id": name, "status": "PASS" if bool(condition) else "FAIL", "detail": json_value(details)})
        return bool(condition)


def same_numeric(left, right, atol=1e-12, rtol=1e-10):
    first, second = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return first.shape == second.shape and bool(np.allclose(first, second, atol=atol, rtol=rtol, equal_nan=True))


def feature_comparison(left, right):
    result = {}
    for key in FEATURE_KEYS:
        first, second = np.asarray(left[key]), np.asarray(right[key])
        result[key] = bool(np.array_equal(first, second)) if "mask" in key else same_numeric(first, second, atol=2e-6, rtol=2e-5)
    return result


def verify_sources(audit, bindings, manifest, mappings, raw, mat_root):
    receipts = []
    sources = {source["source_segment_id"]: source for source in manifest["source_instrument_csv"]}
    role_path = REPOSITORY / "configs/unified_matr_native_run_roles.json"
    run_roles = json.loads(role_path.read_text(encoding="utf-8"))["additional_runs"]
    evidence_valid = all(sha256(REPOSITORY / role["evidence_report"]) == role["evidence_report_sha256"] for role in run_roles)
    audit.check("source_run_protocol_evidence_fingerprints", evidence_valid, {"configuration_sha256": sha256(role_path), "additional_runs": run_roles})
    for cell in bindings["cells"]:
        previous_last, previous_end = 0, None
        first_segment = cell["source_segments"][0]
        for position, segment in enumerate(cell["source_segments"]):
            name, path = segment["source_segment_id"], Path(segment["csv_path"])
            digest = sha256(path)
            authentic = segment["download_status"] == "COMPLETE" and segment["identity_certified"]
            authentic = authentic and path.stat().st_size == segment["bytes"] and digest == segment["sha256"] == sources[name]["sha256"]
            audit.check("native_source_auth_" + name, authentic, {"bytes": path.stat().st_size, "sha256": digest})
            scanned = scan_native_index_runs(path)
            independent = scanned["groups"]
            saved = mappings.loc[mappings.source_segment_id == name].sort_values("source_record_index")
            columns = ("run_index", "instrument_run_cycle_offset", "instrument_cycle_id", "source_row_start", "source_row_stop_exclusive", "source_data_point_first", "source_data_point_last")
            equal = scanned["status"] == "PASS_NATIVE_RESTART_SCAN" and len(saved) == len(independent) and all(same_numeric(saved[column], independent[column]) for column in columns)
            equal = equal and np.array_equal(saved.source_record_index, np.arange(len(saved)))
            native = independent.instrument_cycle_id.to_numpy()
            offset = previous_last if position else 0
            expected = np.full(len(independent), np.nan)
            total_offsets = np.full(len(independent), np.nan)
            aging_offset = 0
            roles_agree = True
            for run_index, run_frame in independent.groupby("run_index", sort=False):
                if run_index == 0:
                    role_name, include = "declared_original_aging_protocol", True
                else:
                    candidates = [role for role in run_roles if role["cell_uid"] == cell["cell_uid"] and role["source_file_id"] == segment["file_id"] and role["run_index"] == run_index]
                    if len(candidates) != 1 or candidates[0]["source_csv_sha256"] != digest:
                        raise ValueError("Unregistered or unauthenticated additional source run role")
                    role_name, include = candidates[0]["role"], candidates[0]["include_in_aging_axis"]
                indices = run_frame.index.to_numpy()
                total_offsets[indices] = offset + aging_offset
                local = run_frame.instrument_cycle_id.to_numpy()
                expected[indices] = np.where((local > 0) & include, local + offset + aging_offset, np.nan)
                saved_run = saved.iloc[indices]
                roles_agree = roles_agree and (saved_run.source_run_protocol_role == role_name).all() and (saved_run.include_run_in_aging_axis == include).all()
                if include:
                    aging_offset += int(local[-1])
            equal = equal and same_numeric(saved.original_cycle_id, expected) and same_numeric(saved.global_original_cycle_offset, total_offsets)
            equal = equal and (saved.file_regular_cycle_offset == offset).all()
            equal = equal and roles_agree and same_numeric(saved.cumulative_instrument_loop_id, offset + independent.instrument_run_cycle_offset.to_numpy() + native)
            zero = saved.instrument_cycle_id == 0
            equal = equal and saved.loc[zero, "original_cycle_id"].isna().all() and (saved.loc[zero, "record_role"] == "initialization_or_restart_conditioning").all()
            equal = equal and (saved.cell_uid == cell["cell_uid"]).all() and (saved.source_file_sha256 == digest).all()
            for column, value in (("source_file_id", segment["file_id"]), ("source_test_id", segment["test_id"]), ("source_batch", segment["source_batch"])):
                equal = equal and (saved[column] == value).all()
            audit.check("all_native_cycle_mapping_" + name, equal, {"native_groups": len(saved), "runs": scanned["run_count"], "restart_evidence": scanned["drops"], "rows": int(independent.source_row_stop_exclusive.iloc[-1]), "first_native": int(native[0]), "last_native": int(native[-1]), "file_global_offset": offset})
            if position:
                alias = CONTINUATIONS.get(name) == cell["source_segments"][position - 1]["source_segment_id"]
                identity = segment["barcode_normalized_sha256"] == first_segment["barcode_normalized_sha256"] and str(segment["source_channel"]) == str(first_segment["source_channel"])
                chronology = segment["first_native_datetime_unix"] > previous_end
                initial = raw.loc[(raw.source_segment_id == name) & (raw.run_index == 0) & (raw.instrument_cycle_id == 0)]
                excluded = len(initial) == 1 and not initial.capacity_valid.any() and not initial.curve_valid.any() and initial.original_cycle_id.isna().all()
                audit.check("continuation_regular_cycle_join_" + name, alias and identity and chronology and native[0] == 0 and excluded, {"same_identity": identity, "calendar_order": chronology, "offset_actual_native_last": offset, "restart_zero_excluded": excluded})
            previous_last = offset + aging_offset
            previous_end = segment["last_native_datetime_unix"]
            receipts.append({"source_segment_id": name, "bytes": segment["bytes"], "sha256": digest, "native_groups": len(saved)})
            if len(receipts) % 25 == 0:
                print(json.dumps({"verification": "source_SHA_and_native_axis", "segments": len(receipts), "total": 174}), flush=True)
    audit.check("all174_native_sources", len(receipts) == 174 and len({item["sha256"] for item in receipts}) == 174, {"segments": len(receipts)})
    mat_receipts = []
    for source in manifest["source_mat_archives"]:
        path = mat_root / source["file_name"]
        digest = sha256(path)
        audit.check("source_mat_auth_" + source["file_name"], path.stat().st_size == source["bytes"] and digest == source["sha256"], digest)
        mat_receipts.append({"file_name": path.name, "bytes": path.stat().st_size, "sha256": digest})
    audit.check("all4_released_archives_retained", len(mat_receipts) == 4, mat_receipts)
    return {"native_instrument_csv": receipts, "released_mat_archives": mat_receipts}


def verify_instrument_metadata(audit, bindings, mat_root):
    audit_path = REPOSITORY / "results/unified_matr/instrument_metadata_audit.json"
    table_path = audit_path.with_suffix(".csv")
    metadata = json.loads(audit_path.read_text(encoding="utf-8"))
    segments = {segment["source_segment_id"]: segment for cell in bindings["cells"] for segment in cell["source_segments"]}
    fields, devices, identities = set(), set(), set()
    checks, receipts = [], []
    original_values = {}
    for row in metadata["source_rows"]:
        segment = segments[row["source_segment_id"]]
        path = mat_root / "instrument_csv" / row["metadata_source_path"]
        authentic = path.stat().st_size == row["metadata_bytes"] and sha256(path) == row["metadata_sha256"]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        if len(records) != 1:
            raise ValueError("Original instrument Metadata.csv must bind one complete test row")
        source = {"source_dataframe_index" if key == "" else key: value for key, value in records[0].items()}
        fields.update(source)
        device, test, channel = source["device_id"], source["test_id"], source["iv_ch_id"]
        devices.add(device)
        identities.add((device, test, channel))
        barcode = hashlib.sha256(source["item_id"].strip().upper().encode()).hexdigest()
        equal = authentic and row["official_edp_test_id"] == segment["test_id"]
        equal = equal and barcode == segment["barcode_normalized_sha256"] and int(channel) + 1 == int(segment["source_channel"])
        redacted = {"item_id", "databases", "canconfig_filename", "m_ncanconfigmd5"}
        for field, value in source.items():
            original_values.setdefault(field, []).append(value)
            public_name = "native_metadata_" + field + ("_sha256" if field in redacted else "")
            expected = hashlib.sha256(value.encode()).hexdigest() if field in redacted and value else value
            equal = equal and expected == row[public_name]
        checks.append(equal)
        receipts.append({"source_segment_id": row["source_segment_id"], "metadata_file_id": row["metadata_file_id"], "sha256": row["metadata_sha256"], "bytes": row["metadata_bytes"]})
    passed = metadata["status"] == "PASS" and len(checks) == len(identities) == 174 and all(checks)
    passed = passed and len(fields) == metadata["source_field_count"] and sha256(table_path) == metadata["csv_sha256"]
    for field, values in original_values.items():
        statistics = metadata["field_audit"][field]
        populated = [value for value in values if value]
        passed = passed and statistics["column_present"] == len(values)
        passed = passed and statistics["empty_count"] == len(values) - len(populated)
        passed = passed and statistics["populated_count"] == len(populated)
        passed = passed and statistics["distinct_populated_count"] == len(set(populated))
    passed = passed and metadata["model_input_columns_added"] == 0
    audit.check("all174_full_metadata_and_instrument_device_identity", passed, {"metadata_files": len(checks), "original_field_union": len(fields), "device_ids_preserved_as_strings": sorted(devices), "source_audit_sha256": sha256(audit_path), "source_csv_sha256": sha256(table_path), "model_input_columns_added": 0})
    return receipts


def verify_source_audit_packages(audit, bindings, raw, mat_root, root):
    location = REPOSITORY / "results/unified_matr"
    schedules = json.loads((location / "schedule_metadata_audit.json").read_text(encoding="utf-8"))
    schedule_qa = json.loads((location / "schedule_metadata_QA.json").read_text(encoding="utf-8"))
    segments = {segment["source_segment_id"]: segment for cell in bindings["cells"] for segment in cell["source_segments"]}
    schedule_checks = []
    for source in schedules["source_links"]:
        segment = segments[source["source_segment_id"]]
        path = mat_root / "instrument_csv" / source["schedule_source_path"]
        schedule_checks.append(path.stat().st_size == source["schedule_bytes"]
                               and sha256(path) == source["schedule_sha256"]
                               and source["official_edp_test_id"] == segment["test_id"])
    schedule_pass = len(schedule_checks) == 174 and all(schedule_checks)
    schedule_pass = schedule_pass and schedule_qa["status"] == "PASS" and all(schedule_qa["checks"].values())
    schedule_pass = schedule_pass and schedule_qa["schedule_metadata_audit_sha256"] == sha256(location / "schedule_metadata_audit.json")
    schedule_pass = schedule_pass and schedules["schedule_static_fields_parquet_sha256"] == sha256(location / "schedule_static_fields.parquet")
    audit.check("all174_original_schedule_sources_and_full_static_fields", schedule_pass,
                {"source_files": len(schedule_checks), "unique_content_hashes": schedules["unique_schedule_content_hashes"],
                 "static_field_rows": schedules["unique_schedule_field_rows"], "source_steps": schedule_qa["original_main_discharge_step_counts"]})
    native = json.loads((location / "native_channel_audit_summary.json").read_text(encoding="utf-8"))
    native_qa = json.loads((location / "native_channel_audit_QA.json").read_text(encoding="utf-8"))
    native_pass = native["status"] == "COMPLETE" and native_qa["status"] == "PASS" and all(native_qa["checks"].values())
    native_pass = native_pass and native_qa["summary_sha256"] == sha256(location / "native_channel_audit_summary.json")
    native_pass = native_pass and native_qa["parquet_sha256"] == sha256(location / "native_channel_audit.parquet")
    native_pass = native_pass and native["complete_source_segments"] == native["numeric_complete_source_segments"] == native["transition_complete_source_segments"] == 174
    audit.check("all_native_columns_complete_missingness_ranges_and_transitions", native_pass,
                {"native_rows": native["native_data_rows"], "field_rows": native["audited_field_rows"], "transition_totals": native_qa["transition_totals"]})
    step_path = location / "SOURCE_DISCHARGE_STEP_QA.json"
    step = json.loads(step_path.read_text(encoding="utf-8"))
    exact_name = "all_capacity_valid_regular_steps_equal_original_protocol"
    numeric_pass = all(value for key, value in step["checks"].items() if key != exact_name)
    valid = raw.loc[raw.original_cycle_id.notna() & raw.capacity_valid]
    expected_steps = {source["source_segment_id"]: next(item["source_step_index"] for item in source["discharge_labeled_steps"] if item["label"] == "Discharge") for source in schedules["source_links"]}
    mismatches = valid.discharge_step_index.ne(valid.source_segment_id.map(expected_steps))
    replay_path = location / "SOURCE_DISCHARGE_STEP_NATIVE_REPLAY.json"
    step_pass = step["coverage_complete"] and numeric_pass and step["capacity_valid_regular_records"] == len(valid)
    step_pass = step_pass and step["source_step_mismatch_records"] == int(mismatches.sum())
    step_pass = step_pass and step["source_step_mismatch_cells"] == valid.loc[mismatches, "cell_uid"].nunique()
    step_pass = step_pass and step["source_step_mismatch_prefix_records"] == int((mismatches & valid.original_cycle_id.le(100)).sum())
    step_pass = step_pass and step["final_audit_matches_completed_cache_all_fields"] and step["final_audit_sha256"] == sha256(root / "raw_record_audit.parquet")
    step_pass = step_pass and step["schedule_metadata_audit_sha256"] == sha256(location / "schedule_metadata_audit.json")
    step_pass = step_pass and step["native_counterexample_replay_sha256"] == sha256(replay_path) and step["native_counterexample_all_replayed_quantities_equal_stored_fields"]
    detail = {key: step[key] for key in ("status", "source_step_mismatch_records", "source_step_mismatch_cells", "source_step_mismatch_prefix_records", "stage_mismatch_pairs", "interpretation", "all6780_raw_exception_records_individually_replayed")}
    detail["exact_source_step_identity"] = step["checks"][exact_name]
    audit.check("physical_phase_evidence_with_source_step_label_exceptions_preserved", step_pass, detail)
    return {"schedules": len(schedule_checks), "source_step_label_exceptions": detail, "native_field_rows": native["audited_field_rows"]}


def verify_metadata_enrichment(audit, root, cells, manifest):
    location = REPOSITORY / "results/unified_matr"
    static = json.loads((location / "source_static_field_audit.json").read_text(encoding="utf-8"))
    instrument = json.loads((location / "instrument_metadata_audit.json").read_text(encoding="utf-8"))
    schedule = json.loads((location / "schedule_metadata_audit.json").read_text(encoding="utf-8"))
    static_rows = {row["source_segment_id"]: row for row in static["cells"]}
    instrument_rows = {row["source_segment_id"]: row for row in instrument["source_rows"]}
    schedule_rows = {row["source_segment_id"]: row for row in schedule["source_links"]}
    supplied = pd.read_csv(REPOSITORY / "data/cell_metadata.csv")
    supplied_rows = {json.loads(row.metadata)["cell_id"]: json.loads(row.metadata) for row in supplied.itertuples() if row.family == "MATR"}
    checks = []
    for cell in cells.itertuples(index=False):
        names = json.loads(cell.source_segments)
        checks.append(json.loads(cell.supplied_metadata_json) == supplied_rows[cell.cell_uid]
                      and json.loads(cell.original_release_static_metadata_json) == [static_rows[name] for name in names]
                      and json.loads(cell.instrument_metadata_json) == [instrument_rows[name] for name in names]
                      and json.loads(cell.source_schedule_metadata_json) == [schedule_rows[name] for name in names]
                      and cell.source_instrument_device_id == instrument_rows[names[0]]["native_metadata_device_id"])
    enrichment = manifest["metadata_enrichment"]
    authentic = enrichment["supplied_metadata_snapshot_sha256"] == sha256(REPOSITORY / "data/cell_metadata.csv")
    authentic = authentic and enrichment["original_release_static_audit_sha256"] == sha256(location / "source_static_field_audit.json")
    authentic = authentic and enrichment["instrument_metadata_audit_sha256"] == sha256(location / "instrument_metadata_audit.json")
    authentic = authentic and enrichment["schedule_metadata_audit_sha256"] == sha256(location / "schedule_metadata_audit.json")
    comparisons = pd.read_parquet(root / "official_release_label_comparison.parquet")
    comparisons_ok = len(comparisons) == enrichment["official_label_comparison_rows"] == 174 and not comparisons.official_label_used_as_model_target.any()
    audit.check("all169_complete_supplied_release_instrument_and_schedule_metadata", len(checks) == 169 and all(checks) and authentic and comparisons_ok,
                {"cells": len(checks), "official_label_comparison_rows": len(comparisons), "official_label_used_as_model_target": False})


def verify_targets(audit, cells, raw, future, arrays, config):
    nominal = config["discharge"]["nominal_capacity_Ah"]
    regular = raw.loc[raw.original_cycle_id.notna()].copy()
    valid = regular.loc[regular.capacity_valid].copy()
    audit.check("unique_regular_physical_cycles", not regular.duplicated(["cell_uid", "original_cycle_id"]).any(), {"regular_records": len(regular)})
    audit.check("initialization_absent_from_regular_targets", not raw.loc[raw.original_cycle_id.isna(), "capacity_valid"].any() and not raw.loc[raw.original_cycle_id.isna(), "curve_valid"].any(), int(raw.original_cycle_id.isna().sum()))
    audit.check("valid_capacity_instrument_range", valid.capacity_Ah.between(0, nominal * 1.5, inclusive="right").all(), valid.capacity_Ah.agg(["min", "max"]).to_dict())
    settings = config["discharge"]
    nominal_current = nominal * settings["design_rate_C"]
    tolerance = nominal_current * settings["current_tolerance_fraction"]
    phase_valid = (valid.capacity_source == "phase_confirmed_instrument_discharge_counter").all()
    phase_valid = phase_valid and (valid.instrument_discharge_counter_zero_Ah.abs() <= settings["capacity_zero_tolerance_Ah"]).all()
    phase_valid = phase_valid and (valid.discharge_start_voltage_V >= settings["full_discharge_min_start_voltage_V"]).all()
    phase_valid = phase_valid and (valid.discharge_end_voltage_V <= settings["full_discharge_max_end_voltage_V"]).all()
    phase_valid = phase_valid and (np.abs(valid.discharge_current_median_A + nominal_current) <= tolerance).all()
    phase_valid = phase_valid and (valid.discharge_current_min_A >= -nominal_current - tolerance).all()
    phase_valid = phase_valid and (valid.discharge_current_max_A <= settings["minimum_discharge_current_A"]).all()
    phase_valid = phase_valid and (valid.constant_current_points >= settings["min_contiguous_points"]).all()
    phase_valid = phase_valid and (valid.discharge_point_start <= valid.constant_current_point_start).all()
    phase_valid = phase_valid and (valid.discharge_point_stop_exclusive >= valid.constant_current_point_stop_exclusive).all()
    audit.check("all_counter_zeros_and_anchored_complete_4C_phases", phase_valid, {"valid_capacity_rows": len(valid), "discharge_step_counts": valid.discharge_step_index.value_counts().to_dict(), "registered_current_A": nominal_current})
    invalid_time = ~raw.time_valid.astype(bool)
    time_isolated = not raw.loc[invalid_time, "integration_valid"].any() and raw.loc[invalid_time, "integration_counter_ratio"].isna().all()
    valid_integrals = raw.loc[raw.integration_valid]
    time_isolated = time_isolated and valid_integrals.time_valid.all() and np.isfinite(valid_integrals.integration_counter_ratio).all()
    audit.check("time_failures_are_separate_from_counter_QV_support", time_isolated, {"time_invalid_records": int(invalid_time.sum()), "capacity_valid_despite_time_fault": int((invalid_time & raw.capacity_valid).sum()), "curve_valid_despite_time_fault": int((invalid_time & raw.curve_valid).sum())})
    count_future = future.loc[future.capacity_valid].groupby("cell_uid").size().to_dict()
    maximum = int(cells.original_cycle_last.max())
    audit.check("dense_future_query_grid", len(future) == len(cells) * (maximum - 100) and not future.duplicated(["cell_uid", "original_cycle_id"]).any(), {"rows": len(future), "global_maximum": maximum})
    all_checks, denominators = [], []
    rates = Counter()
    for index, cell in enumerate(cells.itertuples(index=False)):
        observed = valid.loc[valid.cell_uid == cell.cell_uid].set_index("original_cycle_id").capacity_Ah.to_dict()
        labels = future.loc[future.cell_uid == cell.cell_uid].sort_values("original_cycle_id")
        query = np.arange(101, maximum + 1)
        expected_capacity = np.asarray([observed.get(number, np.nan) for number in query])
        expected_mask = np.isfinite(expected_capacity)
        row_ids = regular.loc[regular.cell_uid == cell.cell_uid, "original_cycle_id"]
        initial_values = [value for number, value in observed.items() if 1 <= number <= 10]
        initial = float(np.median(initial_values)) if initial_values else np.nan
        anchor_values = [value for number, value in observed.items() if 96 <= number <= 100]
        anchor = float(np.median(anchor_values) / nominal) if anchor_values else np.nan
        last = max(observed, default=np.nan)
        censor = query > last if np.isfinite(last) else np.ones(len(query), bool)
        equal = np.array_equal(labels.original_cycle_id, query) and np.array_equal(labels.capacity_valid, expected_mask)
        equal = equal and np.array_equal(labels.observed_mask, expected_mask) and np.array_equal(labels.raw_record_present, np.isin(query, row_ids))
        equal = equal and np.array_equal(labels.censor_mask, censor)
        equal = equal and same_numeric(labels.capacity_Ah, expected_capacity) and same_numeric(labels.soh_nominal, expected_capacity / nominal)
        equal = equal and same_numeric(labels.soh_initial, expected_capacity / initial)
        equal = equal and same_numeric(cell.initial_observed_capacity_Ah, initial) and same_numeric(cell.anchor_soh100, anchor)
        equal = equal and same_numeric(arrays["anchor_soh100"][index], anchor, atol=1e-7, rtol=1e-6)
        equal = equal and same_numeric(cell.last_observed_valid_cycle, last)
        expected_eligible = bool(arrays["curve_valid_mask"][index].any()) and int(arrays["curve_valid_mask"][index, :10].sum()) >= 5 and np.isfinite(anchor)
        equal = equal and bool(cell.input_eligible) == expected_eligible and bool(arrays["input_eligible"][index]) == expected_eligible
        events = [number for number, value in observed.items() if value / nominal <= 0.8]
        equal = equal and bool(cell.eol80_event_observed) == bool(events) and bool(cell.right_censored) == (not events)
        equal = equal and same_numeric(cell.eol80_first_observed_crossing_cycle, min(events) if events else np.nan)
        equal = equal and cell.future_observed_capacity_count == int(expected_mask.sum()) == count_future.get(cell.cell_uid, 0)
        for left, right in ((200, 500), (500, 1000)):
            rate = (observed[left] - observed[right]) / nominal / (right - left) if left in observed and right in observed else np.nan
            equal = equal and same_numeric(getattr(cell, f"r{left}_{right}"), rate)
            rates[f"r{left}_{right}"] += int(np.isfinite(rate))
        all_checks.append(equal)
        denominators.append({"cell_uid": cell.cell_uid, "input_eligible": bool(cell.input_eligible), "prefix_curves": int(arrays["curve_valid_mask"][index].sum()), "valid_future_observations": int(expected_mask.sum()), "failure_reasons": cell.input_failure_reasons})
    audit.check("all_observed_targets_masks_censoring_and_endpoint_rates", all(all_checks), {"cells": len(cells), "target_support": dict(rates)})
    audit.check("all169_qualification_denominator_retained", len(cells) == 169 and len(denominators) == 169, denominators)
    return {"qualification": denominators, "rate_support": dict(rates), "future_observations": int(future.capacity_valid.sum())}


def verify_features(audit, cells, arrays, features, manifest):
    n = len(cells)
    valid = arrays["curve_valid_mask"].astype(bool)
    audit.check("complete_global_local_feature_dimensions", arrays["global_features"].shape == (n, 100, 20) and arrays["local_residual"].shape == (n, 100, 64) and arrays["local_features"].shape == (n, 100, 8, 26), {"global_columns": manifest["global_columns"], "local_columns": manifest["local_columns"]})
    audit.check("fixed_physical_prefix_grid", arrays["q_absolute"].shape == (n, 100, 64) and np.array_equal(arrays["voltage_grid_V"], np.linspace(2.8, 3.3, 64)) and np.array_equal(arrays["cycle_ids"], np.tile(np.arange(1, 101), (n, 1))), {"cells": n})
    audit.check("curve_point_support_and_charge_origins", np.array_equal(np.isfinite(arrays["q_absolute"]).all(axis=2), valid) and np.array_equal(arrays["point_mask"], np.broadcast_to(valid[:, :, None], (n, 100, 64))) and same_numeric(arrays["q_shape"], arrays["q_absolute"] - arrays["q_absolute"][:, :, -1, None], atol=1e-7), {"valid_curves": int(valid.sum())})
    viq = arrays["raw_viq"]
    grid = np.broadcast_to(arrays["voltage_grid_V"], (n, 100, 64))
    viq_correct = viq.shape == (n, 100, 64, 3) and same_numeric(viq[..., 2], arrays["q_absolute"].astype(viq.dtype), atol=0, rtol=0)
    viq_correct = viq_correct and same_numeric(viq[..., 0][arrays["point_mask"]], grid[arrays["point_mask"]], atol=1e-6)
    viq_correct = viq_correct and np.isfinite(viq[arrays["point_mask"]]).all()
    policy_mask = arrays["policy_mask"]
    audit.check("all_model_VIQ_channels_and_known_policy_masks", viq_correct and policy_mask.shape == (n, 15) and policy_mask.dtype == np.bool_ and policy_mask.all() and np.isfinite(arrays["policy"]).all(), {"VIQ_shape": viq.shape, "policy_shape": arrays["policy"].shape})
    for values, mask in (("global_features", "global_mask"), ("local_residual", "local_mask"), ("local_features", "local_feature_mask")):
        audit.check("finite_feature_masks_" + values, arrays[values].shape == arrays[mask].shape and np.array_equal(np.isfinite(arrays[values]), arrays[mask]), {"shape": arrays[values].shape})
    complete = len(features) == n * 100 and not features.duplicated(["cell_uid", "original_cycle_id"]).any()
    comparisons, reference_checks = [], []
    descriptions = {item["cell"]["cell_uid"]: item for item in manifest["reference_cells"]}
    for index, uid in enumerate(cells.cell_uid):
        rows = features.loc[features.cell_uid == uid].sort_values("original_cycle_id")
        equal = len(rows) == 100
        for position, name in enumerate(manifest["global_columns"]):
            equal = equal and same_numeric(rows[name], arrays["global_features"][index, :, position], atol=0, rtol=0)
            equal = equal and np.array_equal(rows[name + "_valid"], arrays["global_mask"][index, :, position])
        for local_bin in range(8):
            for position, name in enumerate(manifest["local_columns"]):
                column = f"local_w{local_bin}_{name}"
                equal = equal and same_numeric(rows[column], arrays["local_features"][index, :, local_bin, position], atol=0, rtol=0)
                equal = equal and np.array_equal(rows[column + "_valid"], arrays["local_feature_mask"][index, :, local_bin, position])
        equal = equal and same_numeric(rows[[f"residual_v{point:02d}_Ah" for point in range(64)]], arrays["local_residual"][index], atol=0, rtol=0)
        equal = equal and same_numeric(rows[[f"irls_weight_v{point:02d}" for point in range(64)]], arrays["fit_weights"][index], atol=0, rtol=0)
        comparisons.append(equal)
        description = descriptions[uid]
        reference = np.asarray(description["reference_cycles"])
        diagnostics = description["diagnostics"]
        projection = np.asarray(diagnostics["weighted_normalized_projection"], dtype=float)
        finite = np.isfinite(projection)
        reconstruction = np.asarray(diagnostics["reconstruction_max_absolute_error_Ah"], dtype=float)
        finite_reconstruction = np.isfinite(reconstruction)
        reference_ok = np.all((reference >= 1) & (reference <= 10))
        reference_ok = reference_ok and np.array_equal(reference, np.flatnonzero(valid[index, :10]) + 1)
        reference_ok = reference_ok and (not finite.any() or np.abs(projection[finite]).max() < 1e-6)
        reference_ok = reference_ok and (not finite_reconstruction.any() or np.abs(reconstruction[finite_reconstruction]).max() < 1e-10)
        spans = [np.quantile(arrays["q_absolute"][index, int(number) - 1], 0.95) - np.quantile(arrays["q_absolute"][index, int(number) - 1], 0.05) for number in reference]
        scale = np.median(spans) if spans else np.nan
        reference_ok = reference_ok and same_numeric(scale, cells.iloc[index].reference_qspan_95_05_Ah, atol=1e-7)
        expected_view = arrays["raw_viq"][index].astype(np.float64)
        if np.isfinite(scale) and scale > 1e-8:
            expected_view[..., 1:] /= scale
        reference_ok = reference_ok and same_numeric(expected_view, arrays["reference_viq"][index], atol=5e-6, rtol=2e-6)
        reference_checks.append(reference_ok)
    audit.check("all_prespecified_features_parquet_NPZ_agree", complete and all(comparisons), {"rows": len(features), "global_columns": len(manifest["global_columns"]), "local_columns": len(manifest["local_columns"])})
    audit.check("prefix_reference_weighted_orthogonality_and_reconstruction", len(descriptions) == n and all(reference_checks), {"checked_cells": len(reference_checks)})
    optional = ("loaded_ica_valid", "loaded_dva_valid", "dynamic_resistance_valid", "polarization_valid")
    audit.check("optional_uncertified_physics_missing", all(column in features for column in optional) and not features[list(optional)].to_numpy().astype(bool).any(), list(optional))
    anchors = ("anchor_observed_cycles", "anchor_support", "anchor_global_features", "anchor_global_mask", "anchor_local_features", "anchor_local_mask")
    audit.check("actual_cycle_physics_anchors_saved", all(key in arrays for key in anchors) and arrays["fit_weights"].shape == (n, 100, 64), list(anchors))
    anchor_checks = []
    original = np.arange(1, 101)
    for index in range(n):
        for position, query in enumerate((10, 50, 100)):
            rows = np.flatnonzero(arrays["decomposition_valid"][index] & (original >= query - 4) & (original <= query))
            last = rows[-1] if len(rows) else None
            expected_cycle = original[last] if last is not None else np.nan
            correct = arrays["anchor_support"][index, position] == len(rows) and same_numeric(arrays["anchor_observed_cycles"][index, position], expected_cycle)
            for values, masks, source_values, source_masks in (("anchor_global_features", "anchor_global_mask", "global_features", "global_mask"), ("anchor_local_features", "anchor_local_mask", "local_features", "local_feature_mask")):
                expected_values = arrays[source_values][index, last] if last is not None else np.full_like(arrays[values][index, position], np.nan)
                expected_masks = arrays[source_masks][index, last] if last is not None else np.zeros_like(arrays[masks][index, position])
                correct = correct and same_numeric(arrays[values][index, position], expected_values, atol=0, rtol=0) and np.array_equal(arrays[masks][index, position], expected_masks)
            anchor_checks.append(correct)
    audit.check("all_K10_K50_K100_anchor_values_masks_and_actual_support", all(anchor_checks), {"anchors": len(anchor_checks), "maximum_lookback_cycles": 4})


def verify_splits_and_scaler(audit, cells, arrays, split, config):
    registered = json.loads((REPOSITORY / "configs/unified_matr_seed42_splits.json").read_text(encoding="utf-8"))
    audit.check("pre_registered_split_exact", split == registered, split.get("split_fingerprint"))
    identifiers = set(cells.cell_uid)
    groups = cells.set_index("cell_uid").protocol_uid.to_dict()
    counts, disjoint = Counter(), True
    for name, roles in split["splits"].items():
        role_sets = [set(roles[key]) for key in ("train", "validation", "test")]
        disjoint = disjoint and set.union(*role_sets) == identifiers and not any(role_sets[left] & role_sets[right] for left, right in ((0, 1), (0, 2), (1, 2)))
        if name.startswith("cell_fold_"):
            counts.update(roles["test"])
        else:
            disjoint = disjoint and not {groups[uid] for uid in roles["train"] + roles["validation"]} & {groups[uid] for uid in roles["test"]}
    audit.check("physical_UID_protocol_split_and_exposure", disjoint and counts == Counter(identifiers) and split["protocol_groups"] == groups and cells.historical_exposure.all(), {"splits": len(split["splits"]), "historical_exposure": split["historical_exposure"]})
    audit.check("seed42_only_and_seven_registered_arms", config["training_seeds"] == [42] and config["models"] == ["M0", "M1", "M2", "M3", "M3-NL", "M3-NP", "M3-NR"], config["training_seeds"])
    from train_seed42 import FEATURE_MASKS, INPUT_KEYS, fold_scaler
    inputs = {key: arrays[key].copy() for key in INPUT_KEYS}
    train_uids = set(split["splits"]["cell_fold_0"]["train"])
    training = np.asarray([index for index, uid in enumerate(cells.cell_uid) if uid in train_uids and bool(cells.iloc[index].input_eligible)], dtype=int)
    heldout = np.setdiff1d(np.arange(len(cells)), training)
    original_scaled, statistics = fold_scaler(inputs, training)
    for key in FEATURE_MASKS:
        inputs[key][heldout] = np.where(np.isfinite(inputs[key][heldout]), inputs[key][heldout] * 1e6 + 1e6, np.nan)
    perturbed_scaled, perturbed_statistics = fold_scaler(inputs, training)
    invariant = statistics == perturbed_statistics and all(np.array_equal(original_scaled[key][training], perturbed_scaled[key][training]) for key in FEATURE_MASKS)
    audit.check("actual_training_fold_scaler_excludes_validation_test", invariant, {"training_cells": len(training), "perturbed_heldout_cells": len(heldout), "model_fitted": False})


def replay_prefixes(audit, bindings, cells, arrays, raw, supplied_root, config):
    requested = ("MATR_b1c0", "MATR_b1c18", "MATR_b2c12", "MATR_b2c44", "MATR_b3c0", "MATR_b4c0")
    by_uid = {cell["cell_uid"]: cell for cell in bindings["cells"]}
    positions = {uid: index for index, uid in enumerate(cells.cell_uid)}
    receipts = []
    for uid in requested:
        cell = by_uid[uid]
        segment = cell["source_segments"][0]
        q, viq = np.full((100, 64), np.nan), np.full((100, 64, 3), np.nan)
        valid = np.zeros(100, bool)
        supplied, supplied_components = supplied_identity_index(supplied_root / (uid + ".pkl"), include_components=True)
        seen, mapping_agrees, matched_records = set(), True, 0
        for record in iter_instrument_cycles(Path(segment["csv_path"]), {**segment, "cell_uid": uid}):
            local_cycle = record["instrument_cycle_id"]
            number = local_cycle + record["instrument_run_cycle_offset"] if local_cycle > 0 else 0
            if number > 100:
                break
            measured = measure_cycle(record, config["discharge"], arrays["voltage_grid_V"])
            fingerprint = measured["raw_fingerprint"]
            if fingerprint is not None and fingerprint in seen:
                measured["capacity_valid"] = measured["curve_valid"] = False
            if number > 0 and fingerprint is not None:
                seen.add(fingerprint)
            if number <= 0:
                continue
            saved = raw.loc[(raw.cell_uid == uid) & (raw.original_cycle_id == number)].iloc[0]
            mapping_agrees = mapping_agrees and saved.source_record_index == record["source_record_index"] and saved.source_row_start == record["source_row_start"]
            mapping_agrees = mapping_agrees and bool(saved.capacity_valid) == measured["capacity_valid"] and bool(saved.curve_valid) == measured["curve_valid"]
            if measured["capacity_valid"]:
                mapping_agrees = mapping_agrees and same_numeric(saved.capacity_Ah, measured["capacity_Ah"])
            identity = channel_identity(record["V"], record["I"] / 1.1, record["Qc"], record["Qd"])
            candidates = supplied.get(identity, [])
            expected_processed = candidates[0] if len(candidates) == 1 else np.nan
            expected_start, expected_stop = (0, len(record["V"])) if len(candidates) == 1 else (np.nan, np.nan)
            expected_status = "UNIQUE_NUMERIC_VIQ_MATCH" if len(candidates) == 1 else "ABSENT_OR_MULTIPLE_SUPPLIED_MATCHES"
            components = supplied_components.get(identity, []) if len(candidates) != 1 else []
            if len(components) == 1:
                component = components[0]
                expected_processed = component["processed_cycle_id"]
                expected_start, expected_stop = component["source_point_start"], component["source_point_stop_exclusive"]
                expected_status = "COMPOSITE_SUPPLIED_COMPONENT_VIQ_MATCH"
            mapping_agrees = mapping_agrees and same_numeric(saved.processed_cycle_id, expected_processed)
            mapping_agrees = mapping_agrees and saved.processed_mapping_status == expected_status
            mapping_agrees = mapping_agrees and same_numeric(saved.processed_source_point_start, expected_start) and same_numeric(saved.processed_source_point_stop_exclusive, expected_stop)
            if measured["curve_valid"]:
                q[number - 1], viq[number - 1], valid[number - 1] = measured["q_absolute"], measured["raw_viq"], True
            matched_records += 1
        index = positions[uid]
        features = extract_prefix_features(q, valid, np.arange(1, 101), arrays["voltage_grid_V"])
        comparisons = feature_comparison({key: arrays[key][index] for key in FEATURE_KEYS}, features)
        equal = mapping_agrees and np.array_equal(valid, arrays["curve_valid_mask"][index])
        equal = equal and same_numeric(q, arrays["q_absolute"][index], atol=5e-8, rtol=1e-6) and same_numeric(viq, arrays["raw_viq"][index], atol=5e-7, rtol=1e-6) and all(comparisons.values())
        audit.check("native_prefix_replay_" + uid, equal, {"records": matched_records, "mapping_and_measurement_agree": mapping_agrees, "features": comparisons})
        extended = extract_prefix_features(np.vstack((q, np.full((20, 64), 1e20))), np.r_[valid, np.ones(20, bool)], np.arange(1, 121), arrays["voltage_grid_V"])
        causal = feature_comparison(features, extended)
        audit.check("native_prefix_future_perturbation_" + uid, all(causal.values()), causal)
        receipts.append({"cell_uid": uid, "source_sha256": segment["sha256"], "regular_prefix_records_read": matched_records, "passed": equal})
    return receipts


def verify_data(audit, root, supplied_root, mat_root, config):
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    bindings = json.loads((root / "instrument_identity_bindings.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split_contract.json").read_text(encoding="utf-8"))
    cells, raw = pd.read_parquet(root / "cells.parquet"), pd.read_parquet(root / "raw_record_audit.parquet")
    mappings, future = pd.read_parquet(root / "instrument_cycle_mapping.parquet"), pd.read_parquet(root / "future_soh.parquet")
    features = pd.read_parquet(root / "early_physics_features.parquet")
    with np.load(root / "early_viq_qv_64.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    identifiers = [cell["cell_uid"] for cell in bindings["cells"]]
    audit.check("all169_fixed_UIDs_and_order", len(identifiers) == 169 and len(set(identifiers)) == 169 and cells.cell_uid.tolist() == identifiers == arrays["cell_uid"].astype(str).tolist(), {"cells": len(cells)})
    mask_keys = ("point_mask", "policy_mask", "global_mask", "local_mask", "local_feature_mask", "curve_valid_mask", "decomposition_valid")
    boolean_masks = all(arrays[key].dtype == np.bool_ for key in mask_keys)
    boolean_masks = boolean_masks and arrays["point_mask"].shape == (len(cells), 100, 64) and arrays["policy_mask"].shape == (len(cells), 15)
    if not audit.check("model_masks_boolean_dtype_before_indexing", boolean_masks, {key: {"dtype": str(arrays[key].dtype), "shape": arrays[key].shape} for key in mask_keys}):
        raise ValueError("Invalid model mask dtype/geometry, unsafe indexing skipped")
    contract = REPOSITORY / "contracts/SRSR_Physics_统一研究问题_一次性数据处理与模型实验总合同.md"
    code_matches = all(sha256(Path(__file__).parent / name) == digest for name, digest in manifest["recipe"]["files"].items())
    audit.check("exact_contract_config_and_reconstruction_recipe", manifest["contract_sha256"] == sha256(contract) and manifest["config_sha256"] == sha256(REPOSITORY / "configs/unified_matr_seed42.json") and code_matches, manifest["recipe"])
    audit.check("native_units_and_actual_axis_provenance", manifest["measurement_units"]["Current"] == "A" and manifest["measurement_units"]["Test_Time"] == "s" and mappings.instrument_axis_certified.all(), manifest["original_axis"])
    mapping_columns = ("cell_uid", "source_segment_id", "source_record_index", "instrument_cycle_id", "original_cycle_id", "source_file_sha256", "source_row_start", "source_row_stop_exclusive")
    equal = len(raw) == len(mappings) and all(raw[column].equals(mappings[column]) for column in mapping_columns)
    audit.check("raw_observations_instrument_mapping_one_to_one", equal, {"raw_records": len(raw), "mapping_records": len(mappings)})
    for index, cell in enumerate(cells.itertuples(index=False)):
        policy, group, _ = policy_context(cell.policy_original)
        audit.check("known_policy_source_" + cell.cell_uid, group == cell.protocol_uid and np.array_equal(policy, arrays["policy"][index]) and np.isfinite(policy).all(), {"protocol_uid": group})
    source_receipts = verify_sources(audit, bindings, manifest, mappings, raw, mat_root)
    source_receipts["instrument_metadata"] = verify_instrument_metadata(audit, bindings, mat_root)
    source_receipts["metadata_and_protocol_audits"] = verify_source_audit_packages(audit, bindings, raw, mat_root, root)
    verify_metadata_enrichment(audit, root, cells, manifest)
    axis_checks = [item for item in audit.checks if item["id"].startswith(("all_native_cycle_mapping_", "native_source_auth_", "continuation_regular_cycle_join_"))]
    axis_valid = len(source_receipts["native_instrument_csv"]) == 174 and all(item["status"] == "PASS" for item in axis_checks)
    audit.check("instrument_original_axis_certified", axis_valid and equal, {"mapping_sha256": sha256(root / "instrument_cycle_mapping.parquet"), "source_records": len(mappings), "axis_scope": manifest["original_axis"]})
    targets = verify_targets(audit, cells, raw, future, arrays, config)
    verify_features(audit, cells, arrays, features, manifest)
    verify_splits_and_scaler(audit, cells, arrays, split, config)
    replay = replay_prefixes(audit, bindings, cells, arrays, raw, supplied_root, config)
    return {"cells": len(cells), "input_eligible_cells": int(cells.input_eligible.sum()), "valid_prefix_curves": int(arrays["curve_valid_mask"].sum()), "source_receipts": source_receipts, "targets": targets, "native_prefix_replays": replay}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--supplied-root", type=Path, required=True)
    parser.add_argument("--mat-root", type=Path, default=REPOSITORY.parent / "private/raw_sources/MATR")
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/unified_matr_seed42.json")
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_root / "DATA_FREEZE.json"
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("status") != "DATA_FREEZE_PASS" or freeze.get("training_seeds") != [42] or not freeze.get("files"):
            raise RuntimeError("Existing freeze certificate is incomplete")
        for item in freeze["files"]:
            if sha256(args.data_root / item["path"]) != item["sha256"]:
                raise RuntimeError("Existing frozen artifact changed: " + item["path"])
        print(json.dumps({"status": "VERIFIED_EXISTING_DATA_FREEZE", "files": len(freeze["files"])}))
        return
    audit = Audit()
    math_qa, parser_qa = run_checks(), run_synthetic_checks()
    audit.check("independent_mathematical_physics_QA", math_qa["status"] == "PASS", math_qa["checks"])
    audit.check("independent_streaming_parser_QA", all(parser_qa.values()), parser_qa)
    missing = [name for name in ARTIFACTS if not (args.data_root / name).is_file()]
    audit.check("all_complete_reconstruction_artifacts", not missing, {"missing": missing})
    summary = {}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not missing:
        try:
            summary = verify_data(audit, args.data_root, args.supplied_root, args.mat_root, config)
        except Exception as error:
            audit.check("all_semantic_QA_completed", False, {"exception": type(error).__name__, "message": str(error)})
    passed = all(item["status"] == "PASS" for item in audit.checks)
    result = {"status": "PASS" if passed else "PENDING_OR_FAILED", "DATA_FREEZE_allowed": passed,
              "formal_training_authorized_by_data_gate": passed,
              "checks": audit.checks, "summary": summary, "training_seeds": [42], "model_optimizer_updates": 0,
              "verifier_sha256": sha256(Path(__file__)), "historical_exposure": "All fixed169 cells participated in historical exploration."}
    if passed:
        from probe_frozen_loader import probe_training_loader
        try:
            probe = probe_training_loader(args.data_root, result)
            audit.check("normal_training_loader_native_schema_probe", probe["status"] == "PASS", probe)
            summary["training_loader_probe"] = probe
        except Exception as error:
            audit.check("normal_training_loader_native_schema_probe", False, {"exception": type(error).__name__, "message": str(error)})
        passed = all(item["status"] == "PASS" for item in audit.checks)
        result.update(status="PASS" if passed else "PENDING_OR_FAILED", DATA_FREEZE_allowed=passed,
                      formal_training_authorized_by_data_gate=passed)
    qa_path = args.data_root / "dataset_qa.json"
    write_json(qa_path, result)
    lines = ["# MATR seed42 数据冻结独立审计", "", f"状态 **{result['status']}**。结果来自实际执行的数学、来源、仪器轴、阶段计量、目标、掩码与划分核验。", "", "## 逐项核验", "", "| 项目 | 结果 |", "|---|---|"]
    lines += [f"| {item['id']} | {item['status']} |" for item in audit.checks]
    package_details = summary.get("source_receipts", {}).get("metadata_and_protocol_audits", {})
    lines += ["", "## 逐电芯资格与目标分母", "", json.dumps(json_value(summary.get("targets", {})), ensure_ascii=False, indent=2), "", "## 完整来源元数据与阶段标签异常", "", json.dumps(package_details, ensure_ascii=False, indent=2), "", "原始 Step_Index 与源 SDU 声明的文字阶段编号存在差异，实际阶段认证按已登记的原始 I/V/Q、完整4C放电、counter零点及2 V终止证据执行。全部例外保留可定位行界。三条原生日志反例完成精确重放，全部例外尚未逐条重读，控制系统根因尚未认证。Metadata 的起止时间和原始 CSV DateTime 的跨字段时钟对齐尚未认证，数值差值只用于来源审计。", "", "## 真实源前缀重放", "", json.dumps(summary.get("native_prefix_replays", []), ensure_ascii=False, indent=2), "", "## 优势与风险", "", "固定资格分母、仪器轴、缺圈、初始化/重启角色及字段失败原因均可追溯，标准化仅训练折拟合。169颗队列继承历史选择和多轮暴露。本轮仅seed42，训练随机性尚未测量。全部所列核验来自可复核断言，通过状态的置信度为高。物理机制归因及模型效果须由后续正式实验评估。", "", "## 待完成项目", ""]
    failures = [item for item in audit.checks if item["status"] != "PASS"]
    lines += [f"- {item['id']}，{json.dumps(item['detail'], ensure_ascii=False)}" for item in failures] or ["全部所列数据核验通过。"]
    report_path = args.data_root / "dataset_qa_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    frozen_files = []
    if passed:
        files = [args.data_root / name for name in ARTIFACTS] + [qa_path, report_path, args.config,
                 REPOSITORY / "configs/unified_matr_seed42_splits.json",
                 REPOSITORY / "configs/unified_matr_native_run_roles.json",
                 REPOSITORY / "results/unified_matr/instrument_metadata_audit.json",
                 REPOSITORY / "results/unified_matr/instrument_metadata_audit.csv",
                 args.data_root / "official_release_label_comparison.parquet",
                 REPOSITORY / "data/cell_metadata.csv",
                 REPOSITORY / "configs/unified_matr_seed42_requirements.json",
                 REPOSITORY / "contracts/SRSR_Physics_统一研究问题_一次性数据处理与模型实验总合同.md"]
        files += [REPOSITORY / "results/unified_matr" / name for name in (
            "source_static_field_audit.json", "instrument_metadata_QA.json",
            "schedule_metadata_audit.json", "schedule_metadata_QA.json", "schedule_static_fields.parquet",
            "native_channel_audit_summary.json", "native_channel_audit_QA.json", "native_channel_audit.parquet",
            "SOURCE_DISCHARGE_STEP_QA.json", "SOURCE_DISCHARGE_STEP_NATIVE_REPLAY.json")]
        role_config = json.loads((REPOSITORY / "configs/unified_matr_native_run_roles.json").read_text(encoding="utf-8"))
        files += [REPOSITORY / role["evidence_report"] for role in role_config["additional_runs"]]
        files += [Path(__file__).parent / name for name in ("build_dataset.py", "dataset_utils.py", "instrument_csv.py", "measurements.py", "physics_features.py", "splits.py", "enrich_metadata.py", "verify_dataset.py", "verify_physics_features.py", "verify_instrument_csv.py", "scan_instrument_restarts.py", "probe_frozen_loader.py")]
        frozen_files = [{"path": os.path.relpath(path.resolve(), args.data_root.resolve()).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size} for path in files]
        fingerprint = args.data_root / "dataset_fingerprint.sha256"
        fingerprint.write_text("".join(item["sha256"] + "  " + item["path"] + "\n" for item in frozen_files), encoding="utf-8")
        frozen_files.append({"path": fingerprint.name, "sha256": sha256(fingerprint), "bytes": fingerprint.stat().st_size})
        write_json(freeze_path, {"status": "DATA_FREEZE_PASS", "files": frozen_files,
                   "dataset_fingerprint_sha256": sha256(fingerprint), "source_SHA_and_axis_receipts": summary["source_receipts"],
                   "source_file_sha256": summary["source_receipts"]["released_mat_archives"],
                   "training_seeds": [42], "checks_passed": len(audit.checks), "historical_exposure": result["historical_exposure"]})
    print(json.dumps({"status": result["status"], "checks": len(audit.checks), "failed": len(failures), "frozen_files": len(frozen_files)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
