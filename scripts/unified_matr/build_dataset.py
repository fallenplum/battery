"""Build all registered arrays once from original instrument CSV cycle groups."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from dataset_utils import POLICY_COLUMNS, channel_identity, json_value, policy_context, sha256, supplied_identity_index, write_json
from instrument_csv import iter_instrument_cycles
from measurements import measure_cycle
from physics_features import DEFAULT_CONFIG, extract_prefix_features
from splits import build_splits


REPOSITORY = Path(__file__).resolve().parents[2]
FEATURE_KEYS = ("global_features", "global_mask", "local_residual", "local_mask", "local_features", "local_feature_mask")
EXTRA_KEYS = ("fit_weights", "decomposition_valid", "failure_code", "reference_q", "reference_derivative_Ah_per_V",
              "anchor_query_cycles", "anchor_observed_cycles", "anchor_support", "anchor_global_features",
              "anchor_global_mask", "anchor_local_features", "anchor_local_mask")


def cell_run_roles(cell):
    source_ids = {segment["file_id"] for segment in cell["source_segments"]}
    configuration = json.loads((REPOSITORY / "configs/unified_matr_native_run_roles.json").read_text(encoding="utf-8"))
    return [row for row in configuration["additional_runs"] if row["source_file_id"] in source_ids]


def source_signature(cell):
    return [{key: segment.get(key) for key in ("source_segment_id", "file_id", "sha256", "bytes", "identity_certified",
                                              "barcode_normalized_sha256", "source_channel")}
            for segment in cell["source_segments"]]


def build_cell(cell, supplied_root, settings):
    uid = cell["cell_uid"]
    supplied_path = supplied_root / (uid + ".pkl")
    supplied, supplied_components = supplied_identity_index(supplied_path, include_components=True)
    policy, protocol_uid, policy_definition = policy_context(cell["policy_original"])
    voltage = np.linspace(2.8, 3.3, 64)
    q = np.full((100, 64), np.nan)
    viq = np.full((100, 64, 3), np.nan)
    curve_valid = np.zeros(100, dtype=bool)
    audit_rows = []
    mappings = []
    capacities = {}
    fingerprints = {}
    run_roles = cell_run_roles(cell)
    segment_summaries = []
    offset = 0
    previous_end_datetime = None
    for segment_number, segment in enumerate(cell["source_segments"]):
        path = Path(segment["csv_path"])
        if not segment["identity_certified"] or path.stat().st_size != segment["bytes"] or sha256(path) != segment["sha256"]:
            raise RuntimeError("Original source identity, byte length or SHA mismatch for " + segment["source_segment_id"])
        if segment_number and segment["first_native_datetime_unix"] <= previous_end_datetime:
            raise RuntimeError("Continuation source dates overlap for " + uid)
        if segment_number and segment["barcode_normalized_sha256"] != cell["source_segments"][0]["barcode_normalized_sha256"]:
            raise RuntimeError("Continuation source changes physical barcode for " + uid)
        native_cycles = []
        first_info = None
        last_info = None
        current_run = -1
        current_role = None
        include_aging_run = False
        within_file_aging_offset = 0
        previous_run_native_last = 0
        for record in iter_instrument_cycles(path, {**segment, "cell_uid": uid}):
            local_cycle = record["instrument_cycle_id"]
            native_cycles.append(local_cycle)
            if record["run_index"] != current_run:
                if current_run >= 0 and include_aging_run:
                    within_file_aging_offset += previous_run_native_last
                current_run = record["run_index"]
                if current_run == 0:
                    current_role, include_aging_run = "declared_original_aging_protocol", True
                else:
                    role = next((row for row in run_roles if row["source_file_id"] == segment["file_id"] and row["run_index"] == current_run), None)
                    if role is None or role["source_csv_sha256"] != segment["sha256"]:
                        raise RuntimeError("Unregistered source protocol run requires actual audit for " + uid + " run" + str(current_run))
                    current_role, include_aging_run = role["role"], role["include_in_aging_axis"]
            previous_run_native_last = local_cycle
            # Cycle_Index0 is retained as initialization/restart evidence. It is
            # excluded from the registered regular-cycle axis and model inputs.
            run_offset = record["instrument_run_cycle_offset"]
            global_cycle = local_cycle + offset + within_file_aging_offset if local_cycle > 0 and include_aging_run else None
            measured = measure_cycle(record, settings, voltage)
            if first_info is None:
                first_info = {"native_cycle": local_cycle, "points": len(record["V"]),
                              "first_datetime": float(record["date_time"][0]), "last_datetime": float(record["date_time"][-1]),
                              "capacity_valid": measured["capacity_valid"], "failure_reasons": measured["failure_reasons"]}
            last_info = {"native_cycle": local_cycle, "points": len(record["V"]),
                         "first_datetime": float(record["date_time"][0]), "last_datetime": float(record["date_time"][-1]),
                         "capacity_valid": measured["capacity_valid"], "failure_reasons": measured["failure_reasons"]}
            identity = channel_identity(record["V"], record["I"] / 1.1, record["Qc"], record["Qd"])
            supplied_ids = supplied.get(identity, [])
            processed_cycle = supplied_ids[0] if len(supplied_ids) == 1 else None
            component_candidates = supplied_components.get(identity, []) if processed_cycle is None else []
            component = component_candidates[0] if len(component_candidates) == 1 else None
            mapping_status = "UNIQUE_NUMERIC_VIQ_MATCH" if processed_cycle is not None else "ABSENT_OR_MULTIPLE_SUPPLIED_MATCHES"
            if component is not None:
                processed_cycle = component["processed_cycle_id"]
                supplied_ids = [processed_cycle]
                mapping_status = "COMPOSITE_SUPPLIED_COMPONENT_VIQ_MATCH"
            duplicate = fingerprints.get(measured["raw_fingerprint"]) if measured["raw_fingerprint"] is not None else None
            if global_cycle is not None and measured["raw_fingerprint"] is not None:
                fingerprints.setdefault(measured["raw_fingerprint"], global_cycle)
            if duplicate is not None:
                measured["capacity_valid"] = measured["curve_valid"] = False
                measured["failure_reasons"].append("LATER_EXACT_WAVEFORM_DUPLICATE")
            if global_cycle is not None and global_cycle in capacities:
                raise RuntimeError("Two regular instrument records map to one physical cycle for " + uid)
            if global_cycle is not None:
                capacities[global_cycle] = measured["capacity_Ah"] if measured["capacity_valid"] else None
            if global_cycle is not None and 1 <= global_cycle <= 100 and measured["curve_valid"]:
                viq[global_cycle - 1] = measured["raw_viq"]
                q[global_cycle - 1] = measured["q_absolute"]
                curve_valid[global_cycle - 1] = True
            reasons = measured["failure_reasons"]
            if global_cycle is None:
                reasons = reasons + ["INITIALIZATION_OR_RESTART_CONDITIONING_EXCLUDED" if local_cycle == 0 else "NONAGING_CHARACTERIZATION_EXCLUDED"]
            reason = "|".join(reasons)
            mapping = {"cell_uid": uid, "source_segment_id": segment["source_segment_id"],
                       "source_batch": segment["source_batch"], "source_file_sha256": segment["sha256"],
                       "source_file_id": segment["file_id"], "source_test_id": segment["test_id"],
                       "source_record_index": record["source_record_index"], "instrument_cycle_id": local_cycle,
                       "run_index": record["run_index"], "instrument_run_cycle_offset": run_offset,
                       "file_regular_cycle_offset": offset,
                       "original_cycle_id": global_cycle, "global_original_cycle_offset": offset + within_file_aging_offset,
                       "cumulative_instrument_loop_id": offset + run_offset + local_cycle,
                       "source_run_protocol_role": current_role, "include_run_in_aging_axis": include_aging_run,
                       "run_boundary_evidence": json.dumps(record["run_boundary_evidence"], sort_keys=True),
                       "source_row_start": record["source_row_start"], "source_row_stop_exclusive": record["source_row_stop_exclusive"],
                       "source_data_point_first": record["source_data_point_first"], "source_data_point_last": record["source_data_point_last"],
                       "record_role": "regular_cycle" if global_cycle is not None else
                                      "initialization_or_restart_conditioning" if local_cycle == 0 else current_role,
                       "raw_record_present": True, "instrument_axis_certified": True,
                       "processed_cycle_id": processed_cycle, "processed_matching_candidates": json.dumps(supplied_ids),
                       "processed_mapping_status": mapping_status,
                       "processed_source_point_start": component["source_point_start"] if component is not None else 0 if processed_cycle is not None else None,
                       "processed_source_point_stop_exclusive": component["source_point_stop_exclusive"] if component is not None else len(record["V"]) if processed_cycle is not None else None,
                       "processed_mapping_channel_identity_sha256": identity,
                       "processed_mapping_absolute_decimal_precision": 10,
                       "native_current_unit": "A", "native_time_unit": "s", "native_capacity_unit": "Ah",
                       "protocol_uid": protocol_uid}
            mappings.append(mapping)
            audit_rows.append({**mapping, "measurement_phase": "complete_original_4C_CC_CV_discharge" if measured["capacity_valid"] else "no_qualified_original_4C_discharge",
                               "capacity_valid": bool(measured["capacity_valid"]) and global_cycle is not None,
                               "curve_valid": bool(measured["curve_valid"]) and global_cycle is not None,
                               "capacity_Ah": measured["capacity_Ah"] if global_cycle is not None else None,
                               "failure_reasons": reason,
                               "measurement_warnings": "|".join(measured["warning_reasons"]),
                               "time_valid": measured["time_valid"], "integration_valid": measured["integration_valid"],
                               "time_reversal_count": measured["time_reversal_count"],
                               "capacity_source": measured["capacity_source"],
                               "instrument_discharge_counter_zero_Ah": measured.get("instrument_discharge_counter_zero_Ah"),
                               "discharge_current_median_A": measured.get("discharge_current_median_A"),
                               "discharge_current_min_A": measured.get("discharge_current_min_A"),
                               "discharge_current_max_A": measured.get("discharge_current_max_A"),
                               "discharge_step_index": measured["discharge_step_index"],
                               "phase_boundary_basis": measured["phase_boundary_basis"],
                               "discharge_start_voltage_V": measured.get("discharge_start_voltage_V"),
                               "discharge_end_voltage_V": measured.get("discharge_end_voltage_V"),
                               "constant_current_start_voltage_V": measured.get("constant_current_start_voltage_V"),
                               "constant_current_end_voltage_V": measured.get("constant_current_end_voltage_V"),
                               "adjacent_stage_counter_increment_Ah": measured.get("adjacent_stage_counter_increment_Ah"),
                               "discharge_points": measured["discharge_points"], "window_points": measured["window_points"],
                               "constant_current_points": measured.get("constant_current_points"),
                               "constant_current_point_start": measured.get("constant_current_point_start"),
                               "constant_current_point_stop_exclusive": measured.get("constant_current_point_stop_exclusive"),
                               "full_discharge_points": measured.get("full_discharge_points"),
                               "discharge_point_start": measured.get("discharge_point_start"),
                               "discharge_point_stop_exclusive": measured.get("discharge_point_stop_exclusive"),
                               "counter_endpoint_point_index": measured.get("counter_endpoint_point_index"),
                               "integration_counter_ratio": measured["integration_counter_ratio"],
                               "duplicate_of_original_cycle": duplicate})
        if not native_cycles or native_cycles[0] != segment["first_native_cycle"] or native_cycles[-1] != segment["last_native_cycle"]:
            raise RuntimeError("Full instrument scan disagrees with completed-source boundaries")
        segment_summaries.append({"source_segment_id": segment["source_segment_id"], "source_sha256": segment["sha256"],
                                  "native_cycle_count": len(native_cycles), "native_cycle_first": min(native_cycles),
                                  "native_cycle_last": native_cycles[-1], "native_cycle_max": max(native_cycles),
                                  "native_missing_indices": int(np.maximum(np.diff(native_cycles) - 1, 0).sum()),
                                  "source_run_count": record["run_index"] + 1,
                                  "regular_cycle_last_within_file": within_file_aging_offset + (native_cycles[-1] if include_aging_run else 0),
                                  "global_regular_cycle_offset": offset, "first_record": first_info, "last_record": last_info})
        offset += within_file_aging_offset + (native_cycles[-1] if include_aging_run else 0)
        previous_end_datetime = segment["last_native_datetime_unix"]
    features = extract_prefix_features(q, curve_valid, np.arange(1, 101), voltage)
    reference_cycles = features["reference_cycles"]
    spans = [np.quantile(q[int(cycle) - 1], 0.95) - np.quantile(q[int(cycle) - 1], 0.05) for cycle in reference_cycles]
    reference_scale = float(np.median(spans)) if spans else np.nan
    reference_viq = viq.copy()
    if np.isfinite(reference_scale) and reference_scale > 1e-8:
        reference_viq[..., 1:] /= reference_scale
    qualified = {cycle: value for cycle, value in capacities.items() if value is not None}
    initial = [value for cycle, value in qualified.items() if 1 <= cycle <= 10]
    initial_capacity = float(np.median(initial)) if initial else None
    anchor = [value for cycle, value in qualified.items() if 96 <= cycle <= 100]
    anchor_soh = float(np.median(anchor) / settings["nominal_capacity_Ah"]) if anchor else np.nan
    input_failures = []
    if not curve_valid.any():
        input_failures.append("NO_QUALIFIED_REGULAR_DISCHARGE_CURVE_WITHIN_ORIGINAL1_TO100")
    if len(reference_cycles) < 5:
        input_failures.append("EARLY_REFERENCE_HAS_FEWER_THAN5_QUALIFIED_CURVES")
    if not np.isfinite(anchor_soh):
        input_failures.append("NO_OBSERVED_CAPACITY_ANCHOR_IN_ORIGINAL96_TO100")
    events = [cycle for cycle, value in qualified.items() if value / settings["nominal_capacity_Ah"] <= 0.8]
    last_valid = max(qualified, default=None)
    cell_row = {"cell_uid": uid, "chemistry": "LFP/graphite", "protocol_uid": protocol_uid,
                "policy_original": cell["policy_original"], "nominal_capacity_Ah": settings["nominal_capacity_Ah"],
                "initial_observed_capacity_Ah": initial_capacity, "anchor_soh100": anchor_soh,
                "anchor_observed_cycles": json.dumps([cycle for cycle in qualified if 96 <= cycle <= 100]),
                "source_segments": json.dumps([segment["source_segment_id"] for segment in cell["source_segments"]]),
                "source_segment_count": len(cell["source_segments"]),
                "raw_cycle_certified": True, "original_cycle_last": max(capacities),
                "last_observed_valid_cycle": last_valid, "raw_records": len(audit_rows),
                "prefix_curve_valid_count": int(curve_valid.sum()),
                "future_observed_capacity_count": sum(cycle > 100 for cycle in qualified),
                "input_eligible": not input_failures, "input_failure_reasons": "|".join(input_failures),
                "main_soh_denominator": "nominal_capacity_1.1Ah", "historical_exposure": True,
                "eol80_event_observed": bool(events), "eol80_first_observed_crossing_cycle": min(events) if events else None,
                "right_censored": not events, "reference_qspan_95_05_Ah": reference_scale,
                "loaded_ica_valid": False, "loaded_dva_valid": False,
                "dynamic_resistance_ohm": None, "dynamic_resistance_valid": False,
                "polarization_parameter": None, "polarization_valid": False,
                "supplied_file_sha256": sha256(supplied_path)}
    for left, right in ((200, 500), (500, 1000)):
        cell_row[f"r{left}_{right}"] = ((qualified[left] - qualified[right]) / settings["nominal_capacity_Ah"] / (right - left)
                                         if left in qualified and right in qualified else None)
    arrays = {"raw_viq": viq.astype(np.float32), "reference_viq": reference_viq.astype(np.float32),
              "point_mask": np.broadcast_to(curve_valid[:, None], (100, 64)).copy(),
              "cycle_ids": np.arange(1, 101, dtype=np.float32), "q_absolute": q.copy(),
              "q_shape": q - q[:, -1:], "curve_valid_mask": curve_valid,
              "policy": policy, "policy_mask": np.ones(15, dtype=bool),
              "anchor_soh100": np.float32(anchor_soh), "input_eligible": np.bool_(not input_failures)}
    for key in FEATURE_KEYS + EXTRA_KEYS:
        value = features[key]
        arrays[key] = np.asarray(value, dtype=bool if "mask" in key or key == "decomposition_valid" else
                                 np.int16 if key in ("failure_code", "anchor_support", "anchor_query_cycles") else
                                 np.float64 if key in ("fit_weights", "reference_q", "reference_derivative_Ah_per_V") else np.float32)
    metadata = {"cell": cell_row, "source_signature": source_signature(cell),
                "run_role_signature": run_roles,
                "policy_definition": policy_definition, "reference_cycles": reference_cycles,
                "global_columns": features["global_columns"], "local_columns": features["local_columns"],
                "diagnostics": features["diagnostics"], "segments": segment_summaries,
                "measurement_failure_counts": dict(Counter(reason for row in audit_rows for reason in row["failure_reasons"].split("|") if reason))}
    return arrays, json_value(metadata), audit_rows, mappings


def write_feature_table(output, arrays, metadata):
    rows = []
    for values, description in zip(arrays, metadata):
        for cycle in range(100):
            row = {"cell_uid": description["cell"]["cell_uid"], "original_cycle_id": cycle + 1,
                   "curve_valid_mask": bool(values["curve_valid_mask"][cycle]),
                   "decomposition_valid": bool(values["decomposition_valid"][cycle]),
                   "decomposition_failure_code": int(values["failure_code"][cycle]),
                   "loaded_ica_valid": False, "loaded_dva_valid": False,
                   "loaded_ica_peak": np.nan, "loaded_dva_peak": np.nan,
                   "dynamic_resistance_ohm": np.nan, "dynamic_resistance_valid": False,
                   "polarization_parameter": np.nan, "polarization_valid": False}
            for number, name in enumerate(description["global_columns"]):
                row[name] = float(values["global_features"][cycle, number])
                row[name + "_valid"] = bool(values["global_mask"][cycle, number])
            for window in range(8):
                for number, name in enumerate(description["local_columns"]):
                    column = f"local_w{window}_{name}"
                    row[column] = float(values["local_features"][cycle, window, number])
                    row[column + "_valid"] = bool(values["local_feature_mask"][cycle, window, number])
            for point in range(64):
                row[f"residual_v{point:02d}_Ah"] = float(values["local_residual"][cycle, point])
                row[f"irls_weight_v{point:02d}"] = float(values["fit_weights"][cycle, point])
            rows.append(row)
    pd.DataFrame(rows).to_parquet(output / "early_physics_features.parquet", index=False)


def finalize(output, cache, cells, config, recipe, registry):
    arrays, metadata, audit, mappings = [], [], [], []
    for cell in cells:
        uid = cell["cell_uid"]
        with np.load(cache / (uid + ".npz"), allow_pickle=False) as archive:
            arrays.append({key: archive[key].copy() for key in archive.files})
        metadata.append(json.loads((cache / (uid + ".json")).read_text(encoding="utf-8")))
        audit.append(pd.read_parquet(cache / (uid + "_audit.parquet")))
        mappings.append(pd.read_parquet(cache / (uid + "_mapping.parquet")))
    cell_rows = [item["cell"] for item in metadata]
    frame = pd.DataFrame(cell_rows)
    frame.to_parquet(output / "cells.parquet", index=False)
    audit_frame = pd.concat(audit, ignore_index=True)
    audit_frame.to_parquet(output / "raw_record_audit.parquet", index=False)
    pd.concat(mappings, ignore_index=True).to_parquet(output / "instrument_cycle_mapping.parquet", index=False)
    observed = audit_frame[audit_frame.original_cycle_id.notna() & (audit_frame.original_cycle_id > 100)].copy()
    maximum = int(frame.original_cycle_last.max())
    future_rows = []
    for cell in cell_rows:
        by_cycle = observed[observed.cell_uid == cell["cell_uid"]].set_index("original_cycle_id").to_dict("index")
        for cycle in range(101, maximum + 1):
            source = by_cycle.get(cycle)
            valid = bool(source["capacity_valid"]) if source is not None else False
            capacity = source["capacity_Ah"] if valid else np.nan
            future_rows.append({"cell_uid": cell["cell_uid"], "original_cycle_id": cycle,
                                "protocol_uid": cell["protocol_uid"], "capacity_valid": valid,
                                "raw_record_present": source is not None, "observed_mask": valid,
                                "censor_mask": cell["last_observed_valid_cycle"] is None or cycle > cell["last_observed_valid_cycle"],
                                "soh_nominal": capacity / cell["nominal_capacity_Ah"],
                                "soh_initial": capacity / cell["initial_observed_capacity_Ah"] if cell["initial_observed_capacity_Ah"] else np.nan,
                                "capacity_Ah": capacity,
                                "failure_reasons": source["failure_reasons"] if source is not None else "UNOBSERVED_QUERY_CYCLE",
                                "source_segment_id": source["source_segment_id"] if source is not None else None,
                                "source_batch": source["source_batch"] if source is not None else None,
                                "measurement_phase": source["measurement_phase"] if source is not None else None,
                                "source_file_sha256": source["source_file_sha256"] if source is not None else None,
                                "source_record_index": source["source_record_index"] if source is not None else None,
                                "processed_cycle_id": source["processed_cycle_id"] if source is not None else None})
    pd.DataFrame(future_rows).to_parquet(output / "future_soh.parquet", index=False)
    stacked = {key: np.stack([item[key] for item in arrays]) for key in arrays[0]}
    stacked["cell_uid"] = frame.cell_uid.to_numpy(dtype=str)
    stacked["voltage_grid_V"] = np.linspace(2.8, 3.3, 64)
    np.savez_compressed(output / "early_viq_qv_64.npz", **stacked)
    write_feature_table(output, arrays, metadata)
    split = build_splits(cell_rows, config["split"])
    registered_split = json.loads((REPOSITORY / "configs/unified_matr_seed42_splits.json").read_text(encoding="utf-8"))
    if split != registered_split:
        raise RuntimeError("Data reconstruction changed the pre-registered physical-cell split")
    write_json(output / "split_contract.json", split)
    write_json(output / "instrument_identity_bindings.json", registry)
    role_path = REPOSITORY / "configs/unified_matr_native_run_roles.json"
    write_json(output / "native_run_roles.json", json.loads(role_path.read_text(encoding="utf-8")))
    manifest = {"study_id": config["study_id"], "status": "COMPLETE_RECONSTRUCTION_PENDING_INDEPENDENT_QA",
                "fixed_physical_cells": len(frame), "input_eligible_cells": int(frame.input_eligible.sum()),
                "source_mat_archives": json.loads((REPOSITORY / "results/unified_matr/source_registry.json").read_text(encoding="utf-8"))["source_files"],
                "source_instrument_csv": [{key: segment.get(key) for key in ("source_segment_id", "source_batch", "file_id", "test_id", "bytes", "sha256", "source_url", "source_channel", "barcode_normalized_sha256")}
                                          for cell in cells for segment in cell["source_segments"]],
                "contract_sha256": sha256(REPOSITORY / "contracts/SRSR_Physics_统一研究问题_一次性数据处理与模型实验总合同.md"),
                "config_sha256": sha256(REPOSITORY / "configs/unified_matr_seed42.json"),
                "native_run_roles_sha256": sha256(role_path),
                "recipe": recipe, "policy_columns": POLICY_COLUMNS,
                "global_columns": metadata[0]["global_columns"], "local_columns": metadata[0]["local_columns"],
                "measurement_units": {"Current": "A", "Test_Time": "s", "Qd": "Ah", "V": "V", "Temperature": "degC"},
                "capacity_measurement": "Device discharge counter at complete sign-consistent CC+CV phase end, with verified pre-phase zero. Original CSV contains no summary capacity.",
                "SOH_primary": "phase-qualified capacity divided by1.1Ah", "SOH_secondary": "same capacity divided by observed first10 median",
                "rate_measurement": "Exact observed endpoint SOH difference divided by original cycle separation, both endpoints required",
                "reference_view": "V unchanged; native A/Ah I,Q divided by each cell first1..10 qualified Q95-Q05 span median. Independent fold-train scalers follow.",
                "original_axis": "Native instrument Cycle_Index within source protocol runs, preserving gaps; authenticated aging continuations use preceding actual last-cycle offsets. Init0 and source-certified nonaging characterization retain native identity only in the audit and receive no canonical aging-cycle ID.",
                "future_mask": "Dense fixed original-cycle queries101..global_max. Actual measured valid points supervised. Unobserved queries after each cell last valid observation are right-censored.",
                "reference_cells": metadata, "historical_exposure": config["split"]["historical_exposure"],
                "training_seeds": [42], "formal_training_started": False}
    write_json(output / "dataset_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "cells": len(frame), "input_eligible": int(frame.input_eligible.sum()),
                      "instrument_records": len(audit_frame), "future_valid_capacity_points": int(observed.capacity_valid.sum())}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--supplied-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--wait-for-sources", action="store_true")
    args = parser.parse_args()
    if (args.output / "DATA_FREEZE.json").exists():
        raise RuntimeError("A frozen dataset already exists. Rebuild is disabled.")
    config = json.loads((REPOSITORY / "configs/unified_matr_seed42.json").read_text(encoding="utf-8"))
    recipe_files = ("build_dataset.py", "dataset_utils.py", "instrument_csv.py", "measurements.py", "physics_features.py", "splits.py")
    recipe = {"files": {name: sha256(Path(__file__).parent / name) for name in recipe_files},
              "config_sha256": sha256(REPOSITORY / "configs/unified_matr_seed42.json")}
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    cache_contract = args.cache / "RECIPE.json"
    if cache_contract.exists() and json.loads(cache_contract.read_text(encoding="utf-8")) != recipe:
        raise RuntimeError("Existing reconstruction cache uses different source code or settings")
    write_json(cache_contract, recipe)
    completed = set()
    while True:
        registry = json.loads(args.bindings.read_text(encoding="utf-8"))
        for cell in registry["cells"]:
            uid = cell["cell_uid"]
            if uid in completed or any(segment["download_status"] != "COMPLETE" for segment in cell["source_segments"]):
                continue
            metadata_path = args.cache / (uid + ".json")
            if metadata_path.exists():
                cached = json.loads(metadata_path.read_text(encoding="utf-8"))
                if cached["source_signature"] != source_signature(cell) or cached["run_role_signature"] != cell_run_roles(cell):
                    raise RuntimeError("Completed cell cache source changed for " + uid)
            else:
                print(json.dumps({"status": "CELL_RECONSTRUCTION_STARTED", "cell_uid": uid}), flush=True)
                arrays, metadata, audit, mappings = build_cell(cell, args.supplied_root, config["discharge"])
                np.savez_compressed(args.cache / (uid + ".npz"), **arrays)
                pd.DataFrame(audit).to_parquet(args.cache / (uid + "_audit.parquet"), index=False)
                pd.DataFrame(mappings).to_parquet(args.cache / (uid + "_mapping.parquet"), index=False)
                write_json(metadata_path, metadata)
            completed.add(uid)
            print(json.dumps({"status": "CELL_RECONSTRUCTED", "cell_uid": uid, "completed": len(completed), "total": 169}), flush=True)
        write_json(args.output / "RECONSTRUCTION_PROGRESS.json", {"status": "BUILDING" if len(completed) < 169 else "READY_FOR_FINALIZATION",
                   "completed_cells": sorted(completed), "completed_count": len(completed), "fixed_cells": 169,
                   "source_csv_complete": registry["complete_csv_files"], "formal_training_started": False})
        if len(completed) == 169:
            break
        if not args.wait_for_sources:
            print(json.dumps({"status": "WAITING_FOR_INSTRUMENT_SOURCES", "completed_cells": len(completed)}), flush=True)
            return
        time.sleep(30)
    finalize(args.output, args.cache, registry["cells"], config, recipe, registry)


if __name__ == "__main__":
    main()
