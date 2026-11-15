"""Certify explicit regular-cycle joins from the five native restart logs."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from instrument_csv import iter_instrument_cycles
from measurements import measure_cycle


CONTINUATIONS = {"b1c0": "b2c7", "b1c1": "b2c8", "b1c2": "b2c9", "b1c3": "b2c15", "b1c4": "b2c16"}
REPOSITORY = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_record(record, settings):
    measured = measure_cycle(record, settings, np.linspace(2.8, 3.3, 64))
    return {
        "instrument_cycle_id": record["instrument_cycle_id"], "points": len(record["V"]),
        "source_row_start": record["source_row_start"], "source_row_stop_exclusive": record["source_row_stop_exclusive"],
        "first_datetime_unix": float(record["date_time"][0]), "last_datetime_unix": float(record["date_time"][-1]),
        "first_data_point": float(record["data_point"][0]), "last_data_point": float(record["data_point"][-1]),
        "time_reversal_count": record["time_reversal_count"],
        "current_min_A": float(np.min(record["I"])), "current_max_A": float(np.max(record["I"])),
        "step_indices": np.unique(record["step_index"]).tolist(),
        "voltage_first_V": float(record["V"][0]), "voltage_last_V": float(record["V"][-1]),
        "Qd_first_Ah": float(record["Qd"][0]), "Qd_last_Ah": float(record["Qd"][-1]),
        "capacity_valid": measured["capacity_valid"], "curve_valid": measured["curve_valid"],
        "capacity_Ah": measured["capacity_Ah"], "failure_reasons": measured["failure_reasons"],
    }


def scan_log(segment, uid, settings):
    path = Path(segment["csv_path"])
    digest = sha256(path)
    if digest != segment["source_sha256"] or path.stat().st_size != segment["bytes"]:
        raise ValueError("Native continuation source SHA/length mismatch")
    first_records, last_records = [], collections.deque(maxlen=3)
    native_ids = []
    rows = 0
    previous_datetime = None
    temporal_reversals = 0
    nonfinite_datetimes = 0
    metadata = {**segment, "cell_uid": uid}
    for record in iter_instrument_cycles(path, metadata):
        native_ids.append(record["instrument_cycle_id"])
        rows += len(record["V"])
        dates = record["date_time"]
        nonfinite_datetimes += int(np.sum(~np.isfinite(dates)))
        temporal_reversals += int(np.sum(np.diff(dates) < 0))
        temporal_reversals += int(previous_datetime is not None and dates[0] < previous_datetime)
        previous_datetime = dates[-1]
        if len(first_records) < 2:
            first_records.append(summarize_record(record, settings))
        last_records.append(record)
    last_summaries = [summarize_record(record, settings) for record in last_records]
    gaps = [{"after_native_cycle": left, "before_native_cycle": right, "missing_count": right - left - 1}
            for left, right in zip(native_ids[:-1], native_ids[1:]) if right - left > 1]
    return {
        "source_segment_id": segment["source_segment_id"], "source_file_sha256": digest,
        "bytes": path.stat().st_size, "native_groups": len(native_ids), "source_rows": rows,
        "first_native_cycle": min(native_ids), "last_native_cycle": max(native_ids),
        "missing_native_cycle_intervals": gaps, "DateTime_reverse_count": temporal_reversals,
        "DateTime_nonfinite_count": nonfinite_datetimes,
        "first_records": first_records, "last_records": last_summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/unified_matr_seed42.json")
    args = parser.parse_args()
    bindings_bytes = args.bindings.read_bytes()
    bindings = json.loads(bindings_bytes)
    settings = json.loads(args.config.read_text(encoding="utf-8"))["discharge"]
    signature = {"parser_sha256": sha256(Path(__file__).with_name("instrument_csv.py")),
                 "measurements_sha256": sha256(Path(__file__).with_name("measurements.py")),
                 "discharge_config_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
                 "auditor_sha256": sha256(Path(__file__))}
    cache_path = args.output.with_name("CONTINUATION_NATIVE_LOG_SCANS.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {"signature": signature, "segments": {}}
    if cache.get("signature") != signature:
        cache = {"signature": signature, "segments": {}}
    by_uid = {cell["cell_uid"]: cell for cell in bindings["cells"]}
    results = []
    for predecessor, continuation in CONTINUATIONS.items():
        uid = "MATR_" + predecessor
        cell = by_uid[uid]
        segments = {item["source_segment_id"]: item for item in cell["source_segments"]}
        first, following = segments[predecessor], segments[continuation]
        if any(segment.get("download_status") != "COMPLETE" for segment in (first, following)):
            results.append({"cell_uid": uid, "status": "PENDING_NATIVE_LOG_RECOVERY", "source_segments": [predecessor, continuation]})
            continue
        for segment in (first, following):
            name = segment["source_segment_id"]
            saved = cache["segments"].get(name)
            if saved is None or saved["source_file_sha256"] != segment["source_sha256"]:
                cache["segments"][name] = scan_log(segment, uid, settings)
                print(json.dumps({"source_segment": name, "native_groups": cache["segments"][name]["native_groups"]}), flush=True)
        before, after = cache["segments"][predecessor], cache["segments"][continuation]
        before_last = before["last_records"][-1]
        after_first = after["first_records"][0]
        identity = first["barcode_normalized_sha256"] == following["barcode_normalized_sha256"] and str(first["source_channel"]) == str(following["source_channel"])
        chronological = after_first["first_datetime_unix"] > before_last["last_datetime_unix"]
        reset = after["first_native_cycle"] == 0
        # A valid primary terminal discharge is retained. Restart-zero groups
        # have a separately audited conditioning role. Internal clock failures
        # remain measurement masks and do not alter observed instrument IDs.
        boundary_conflict = after_first["capacity_valid"]
        checks = {"same_barcode_and_channel": identity,
                  "official_continuation_alias": following.get("continuation_of") == predecessor,
                  "successor_chronologically_later": chronological, "successor_resets_native_counter_to_zero": reset,
                  "restart_zero_has_no_comparable_regular_discharge": not boundary_conflict,
                  "cross_test_boundary_datetimes_finite": bool(np.isfinite(before_last["last_datetime_unix"]) and np.isfinite(after_first["first_datetime_unix"]))}
        certified = all(checks.values())
        results.append({
            "cell_uid": uid, "status": "CANDIDATE_JOIN_EVIDENCE_PASS" if certified else "BOUNDARY_REQUIRES_REVIEW",
            "checks": checks, "source_segments": [predecessor, continuation],
            "regular_cycle_offsets": {predecessor: 0, continuation: before["last_native_cycle"]} if certified else None,
            "predecessor_last_native_cycle": before["last_native_cycle"],
            "predecessor_last_record": before_last, "successor_initial_record": after_first,
            "successor_first_regular_record": after["first_records"][1],
            "calendar_gap_seconds": after_first["first_datetime_unix"] - before_last["last_datetime_unix"],
            "global_regular_cycle_rule": "predecessor native ID unchanged, successor regular ID=predecessor last native ID+successor native ID, both native0 groups excluded from regular query labels",
            "primary_terminal_measurement_valid": before_last["capacity_valid"],
            "terminal_rule": "Keep the actual terminal native index. Preserve a valid terminal measurement, otherwise preserve its masked slot.",
            "internal_clock_anomaly_counts": {
                predecessor: {"DateTime_reverse_count": before["DateTime_reverse_count"], "DateTime_nonfinite_count": before["DateTime_nonfinite_count"]},
                continuation: {"DateTime_reverse_count": after["DateTime_reverse_count"], "DateTime_nonfinite_count": after["DateTime_nonfinite_count"]}},
            "axis_scope": "Explicit cumulative regular-cycle reconstruction across a documented instrument reset",
            "source_sha256": {predecessor: before["source_file_sha256"], continuation: after["source_file_sha256"]},
        })
    complete = all(item["status"] == "CANDIDATE_JOIN_EVIDENCE_PASS" for item in results)
    result = {"status": "ALL_FIVE_CANDIDATE_JOINS_PASS" if complete else "PARTIAL_CONTINUATION_AXIS_AUDIT",
              "results": results, "signature": signature,
              "bindings_sha256": hashlib.sha256(bindings_bytes).hexdigest(),
              "cycle_axis_scope": "Cumulative native regular-cycle index with explicit restart conditioning and retained partial-attempt/gap slots",
              "DATA_FREEZE_allowed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passing_pairs": sum(item["status"] == "CANDIDATE_JOIN_EVIDENCE_PASS" for item in results)}))


if __name__ == "__main__":
    main()
