"""One-CPU, complete-column CSV audit with one cached scan per source SHA."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import psutil


VENDOR = "https://www.arbin.com/what-can-be-included-in-the-export-file-eis-data-statistics-coulombic-efficiency.html"
AUTHOR = "https://github.com/chueh-ermon/BMS-autoanalysis/blob/aa54406731665e3ef1f77ad43dbc55ff7c763fd3/cell_analysis.m"
PROJECT = "https://data.matr.io/1/api/v1/edp/projects/5c48dd2bc625d700019f3204"
NUMERIC_SCHEMA_VERSION = "native_all_numeric_fields_v1"
ORIGINAL_NUMERIC_IMPLEMENTATION = "a87eb6d87ccce2e6a40bf15e80af1abb18052420d996f662a9f005b4caea9513"
TRANSITION_COLUMNS = ["Cycle_Index", "Data_Point", "DateTime", "Test_Time", "Step_Index",
                      "Step_Time", "Current", "Voltage", "Charge_Capacity", "Discharge_Capacity"]
UNITS = {
    "Data_Point": ("index", "export_record_counter", VENDOR),
    "Test_Time": ("s", "vendor_time_unit_and_author_division_by60", VENDOR + " | " + AUTHOR),
    "DateTime": ("Unix_s", "observed_numeric_epoch_matches_official_test_dates_timezone_unverified", PROJECT),
    "Step_Time": ("s", "vendor_export_field_unit", VENDOR),
    "Step_Index": ("index", "native_schedule_step_ordinal", VENDOR),
    "Cycle_Index": ("index", "native_controller_cycle_identifier_with_resets_and_gaps_retained", AUTHOR),
    "Current": ("A", "vendor_export_unit_and_author_current_division_by1.1", VENDOR + " | " + AUTHOR),
    "Voltage": ("V", "vendor_export_field_unit", VENDOR),
    "Charge_Capacity": ("Ah", "vendor_export_field_unit", VENDOR),
    "Discharge_Capacity": ("Ah", "vendor_export_field_unit", VENDOR),
    "Charge_Energy": ("Wh", "vendor_export_field_unit", VENDOR),
    "Discharge_Energy": ("Wh", "vendor_export_field_unit", VENDOR),
    "dV/dt": ("V/s", "dimension_of_native_voltage_per_time_derivative_filter_and_bandwidth_unverified", VENDOR),
    "Internal_Resistance": ("ohm", "native_instrument_resistance_field_physical_measurement_unverified", AUTHOR),
    "Temperature": ("degC", "official_typeT_thermocouple_field_contact_and_calibration_not_authenticated", PROJECT),
    "Aux_Voltage": ("V", "native_auxiliary_voltage_field_wiring_identity_unverified", VENDOR),
}


class HashingReader(io.RawIOBase):
    def __init__(self, path):
        super().__init__()
        self.handle = Path(path).open("rb")
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self):
        return True

    def read(self, size=-1):
        block = self.handle.read(size)
        self.digest.update(block)
        self.bytes_read += len(block)
        return block

    def readinto(self, buffer):
        count = self.handle.readinto(buffer)
        self.digest.update(memoryview(buffer)[:count])
        self.bytes_read += count
        return count

    def close(self):
        self.handle.close()
        super().close()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def empty_state():
    return {"count": 0, "finite_count": 0, "missing_count": 0, "nonfinite_count": 0,
            "positive_inf_count": 0, "negative_inf_count": 0, "zero_count": 0,
            "min": None, "max": None, "mean": None}


def update_state(state, values):
    finite = np.isfinite(values)
    valid = values[finite]
    state["count"] += len(values)
    state["missing_count"] += int(np.isnan(values).sum())
    state["nonfinite_count"] += int((~finite).sum())
    state["positive_inf_count"] += int(np.isposinf(values).sum())
    state["negative_inf_count"] += int(np.isneginf(values).sum())
    state["zero_count"] += int((values == 0).sum())
    if not len(valid):
        return
    old_count = state["finite_count"]
    total = old_count + len(valid)
    chunk_mean = float(np.sum(valid / len(valid)))
    state["mean"] = chunk_mean if old_count == 0 else state["mean"] * (old_count / total) + chunk_mean * (len(valid) / total)
    state["finite_count"] = total
    lower, upper = float(valid.min()), float(valid.max())
    state["min"] = lower if state["min"] is None else min(state["min"], lower)
    state["max"] = upper if state["max"] is None else max(state["max"], upper)


def transition_state():
    kinds = ("cycle_index_change", "cycle_index_decrease", "cycle_index_forward_jump",
             "data_point_nonincrease", "datetime_reverse", "test_time_reverse")
    return {"rows": 0, "previous": None, "counts": {name: 0 for name in kinds},
            "first_examples": {name: [] for name in kinds}}


def row_context(values):
    return {name: float(value) if np.isfinite(value) else None for name, value in zip(TRANSITION_COLUMNS, values)}


def update_transitions(state, chunk):
    values = chunk[TRANSITION_COLUMNS].to_numpy(dtype=float)
    if not len(values):
        return
    previous = state["previous"]
    base_row = state["rows"]
    joined = values
    if previous is not None:
        joined = np.vstack((previous, values))
        base_row -= 1
    differences = np.diff(joined[:, :4], axis=0)
    finite_pairs = np.isfinite(joined[:-1, :4]) & np.isfinite(joined[1:, :4])
    events = {"cycle_index_change": finite_pairs[:, 0] & (differences[:, 0] != 0),
              "cycle_index_decrease": finite_pairs[:, 0] & (differences[:, 0] < 0),
              "cycle_index_forward_jump": finite_pairs[:, 0] & (differences[:, 0] > 1),
              "data_point_nonincrease": finite_pairs[:, 1] & (differences[:, 1] <= 0),
              "datetime_reverse": finite_pairs[:, 2] & (differences[:, 2] < 0),
              "test_time_reverse": finite_pairs[:, 3] & (differences[:, 3] < 0)}
    for kind, mask in events.items():
        state["counts"][kind] += int(mask.sum())
        remaining = 3 - len(state["first_examples"][kind])
        for index in np.flatnonzero(mask)[:remaining]:
            state["first_examples"][kind].append({"row_before": base_row + int(index),
                                                   "row_after": base_row + int(index) + 1,
                                                   "before": row_context(joined[index]),
                                                   "after": row_context(joined[index + 1])})
    state["rows"] += len(values)
    state["previous"] = values[-1].copy()


def numeric_cache_valid(audit):
    return audit["status"] == "PASS" and (audit.get("numeric_schema_version") == NUMERIC_SCHEMA_VERSION
                                           or audit["implementation_sha256"] == ORIGINAL_NUMERIC_IMPLEMENTATION)


def scan_file(segment, cell_uid, chunk_rows, implementation_sha, existing=None):
    states, error = {}, None
    transitions = transition_state()
    raw = HashingReader(segment["csv_path"])
    try:
        with pd.read_csv(raw, chunksize=chunk_rows, dtype=np.float64, low_memory=False,
                         encoding="utf-8-sig", na_values=["", "nan", "NaN", "NA", "N/A", "null", "None"],
                         usecols=TRANSITION_COLUMNS if existing is not None else None) as reader:
            for chunk in reader:
                update_transitions(transitions, chunk)
                if existing is None:
                    for column in chunk:
                        state = states.setdefault(column, empty_state())
                        update_state(state, chunk[column].to_numpy(dtype=float))
    except (ValueError, OSError, pd.errors.ParserError) as exception:
        error = {"type": type(exception).__name__, "detail": str(exception)}
    finally:
        raw.close()
    digest = raw.digest.hexdigest()
    complete = error is None and digest == segment["sha256"] and raw.bytes_read == segment["bytes"]
    transitions.pop("previous")
    transitions.update(status="PASS" if complete else "FAIL", actual_sha256=digest,
                       bytes_read=raw.bytes_read, implementation_sha256=implementation_sha)
    if existing is not None:
        result = dict(existing)
        result["numeric_schema_version"] = NUMERIC_SCHEMA_VERSION
        result["transitions"] = transitions
        result["transition_error"] = error
        if transitions["rows"] != existing["rows"][0]["count"]:
            transitions["status"] = "FAIL"
            result["transition_error"] = {"type": "NumericTransitionRowCountMismatch"}
        return result
    rows = []
    for column, state in states.items():
        unit, scope, source = UNITS.get(column, ("unverified", "unrecognized_native_field_retained", "source CSV header"))
        rows.append({"cell_uid": cell_uid, "source_segment_id": segment["source_segment_id"],
                     "source_batch": segment["source_batch"], "source_channel": segment["source_channel"],
                     "file_id": segment["file_id"], "source_sha256": segment["sha256"],
                     "field": column, "unit": unit, "unit_scope": scope, "unit_source": source,
                     **state})
    return {"status": "PASS" if complete else "FAIL", "cell_uid": cell_uid,
            "source_segment_id": segment["source_segment_id"], "file_id": segment["file_id"],
            "source_sha256": segment["sha256"], "actual_sha256": digest,
            "source_bytes": segment["bytes"], "bytes_read": raw.bytes_read,
            "implementation_sha256": implementation_sha, "chunk_rows": chunk_rows,
            "numeric_schema_version": NUMERIC_SCHEMA_VERSION, "transitions": transitions,
            "columns": len(rows), "rows": rows, "error": error}


def publish(cache, destination, segment_count, implementation_sha, affinity):
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(cache.glob("*.json"))]
    audits = [audit for audit in audits if numeric_cache_valid(audit) or audit["implementation_sha256"] == implementation_sha]
    rows = [row for audit in audits for row in audit["rows"]]
    if rows:
        table = pd.DataFrame(rows)
        temporary = destination / "native_channel_audit.parquet.tmp"
        table.to_parquet(temporary, index=False)
        os.replace(temporary, destination / "native_channel_audit.parquet")
    numeric_complete = sum(audit["status"] == "PASS" for audit in audits)
    total_complete = sum(audit["status"] == "PASS" and audit.get("transitions", {}).get("status") == "PASS" for audit in audits)
    failures = [{"source_segment_id": audit["source_segment_id"], "file_id": audit["file_id"],
                 "error": audit.get("transition_error") or audit["error"],
                 "actual_sha256": audit.get("transitions", {}).get("actual_sha256", audit["actual_sha256"]),
                 "source_sha256": audit["source_sha256"]}
                for audit in audits if audit["status"] != "PASS" or audit.get("transitions", {}).get("status") == "FAIL"]
    transition_totals = {}
    per_file_transitions = []
    for audit in audits:
        transitions = audit.get("transitions")
        if transitions is None:
            continue
        for kind, count in transitions["counts"].items():
            transition_totals[kind] = transition_totals.get(kind, 0) + count
        examples = {}
        for kind, contexts in transitions["first_examples"].items():
            examples[kind] = [{"row_before": pair["row_before"], "row_after": pair["row_after"],
                               "cycle_before": pair["before"]["Cycle_Index"], "cycle_after": pair["after"]["Cycle_Index"],
                               "step_before": pair["before"]["Step_Index"], "step_after": pair["after"]["Step_Index"]}
                              for pair in contexts]
        per_file_transitions.append({"source_segment_id": audit["source_segment_id"], "file_id": audit["file_id"],
                                     "source_sha256": audit["source_sha256"], "rows": transitions["rows"],
                                     "counts": transitions["counts"], "first_examples": examples,
                                     "full_context_private_cache": audit["source_sha256"] + ".json"})
    channels = {}
    for row in rows:
        field = row["field"]
        item = channels.setdefault(field, {"files": 0, "count": 0, "finite_count": 0, "missing_count": 0,
                                           "nonfinite_count": 0, "zero_count": 0, "min": None, "max": None,
                                           "unit": row["unit"], "unit_scope": row["unit_scope"], "unit_source": row["unit_source"]})
        item["files"] += 1
        for name in ("count", "finite_count", "missing_count", "nonfinite_count", "zero_count"):
            item[name] += row[name]
        if row["min"] is not None:
            item["min"] = row["min"] if item["min"] is None else min(item["min"], row["min"])
            item["max"] = row["max"] if item["max"] is None else max(item["max"], row["max"])
    summary = {"status": "FAIL" if failures else "COMPLETE" if total_complete == segment_count else "RUNNING",
               "required_source_segments": segment_count, "complete_source_segments": total_complete,
               "numeric_complete_source_segments": numeric_complete,
               "transition_complete_source_segments": sum(audit.get("transitions", {}).get("status") == "PASS" for audit in audits),
               "audited_field_rows": len(rows), "numeric_column_scan_bytes": sum(audit["bytes_read"] for audit in audits),
               "transition_column_scan_bytes": sum(audit.get("transitions", {}).get("bytes_read", 0) for audit in audits),
               "native_data_rows": sum(audit["rows"][0]["count"] for audit in audits if audit["rows"]),
               "channels": channels, "failures": failures, "cpu_affinity": affinity,
               "transition_totals": transition_totals, "per_file_transitions": per_file_transitions,
               "numeric_schema_version": NUMERIC_SCHEMA_VERSION,
               "numeric_implementation_sha256_used": sorted({audit["implementation_sha256"] for audit in audits}),
               "transition_implementation_sha256_used": sorted({audit["transitions"]["implementation_sha256"] for audit in audits if "transitions" in audit}),
               "scan_schedule": "Numeric v1 caches are reused per source SHA. Sources scanned before transition auditing was requested receive a selected-column transition scan. Later sources compute both audits in one stream.",
               "implementation_sha256": implementation_sha,
               "definitions": {"count": "All parsed native rows", "missing_count": "Parsed NaN including empty and declared NA tokens",
                               "nonfinite_count": "NaN and positive or negative infinity", "min_max_mean": "Finite numeric values only",
                               "zero_count": "Exact parsed numeric zero, retained without interpreting it as a physical zero or missing value"},
               "IR_scope": "Native Internal_Resistance stored-row values audited. Independent diagnostic count, paired SOC, temperature, pulse-stage timing and dynamic resistance truth remain unauthenticated. R features retain the main contract's missing status.",
               "input_and_mask_changes": 0, "model_fits": 0}
    write_json(destination / "native_channel_audit_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=32768)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--only-file-id")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    allowed = process.cpu_affinity()
    process.cpu_affinity([allowed[-1]])
    affinity = process.cpu_affinity()
    implementation_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    while True:
        bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
        required = sum(len(cell["source_segments"]) for cell in bindings["cells"])
        for cell in bindings["cells"]:
            for segment in cell["source_segments"]:
                if segment["download_status"] != "COMPLETE" or (args.only_file_id and segment["file_id"] != args.only_file_id):
                    continue
                path = args.cache / (segment["sha256"] + ".json")
                if path.exists():
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if numeric_cache_valid(existing) and existing.get("transitions", {}).get("status") == "PASS":
                        continue
                else:
                    existing = None
                reuse_numeric = existing if existing is not None and numeric_cache_valid(existing) else None
                audit = scan_file(segment, cell["cell_uid"], args.chunk_rows, implementation_sha, reuse_numeric)
                write_json(path, audit)
                summary = publish(args.cache, args.output, required, implementation_sha, affinity)
                print(json.dumps({"status": audit["status"], "source_segment_id": segment["source_segment_id"],
                                  "columns": audit["columns"], "complete_files": summary["complete_source_segments"]}), flush=True)
        summary = publish(args.cache, args.output, required, implementation_sha, affinity)
        if not args.watch or summary["complete_source_segments"] + len(summary["failures"]) >= required:
            break
        time.sleep(30)
    print(json.dumps({"status": summary["status"], "complete_source_segments": summary["complete_source_segments"],
                      "audited_field_rows": summary["audited_field_rows"]}), flush=True)
    if summary["failures"]:
        raise AssertionError("Native channel audit has unverified source files")


if __name__ == "__main__":
    main()
