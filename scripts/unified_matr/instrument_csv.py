"""Streaming reader for the original MATR instrument CSV records.

Native Current is A and Test_Time is s. Every observed Cycle_Index is retained,
including zero and the terminal cycle. No interpolation, time repair, summary
capacity, continuation offset or cell identity is inferred by this reader.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "Data_Point", "Test_Time", "DateTime", "Step_Time", "Step_Index",
    "Cycle_Index", "Current", "Voltage", "Charge_Capacity", "Discharge_Capacity",
)
ARRAY_COLUMNS = {
    "V": "Voltage", "I": "Current", "Qc": "Charge_Capacity", "Qd": "Discharge_Capacity",
    "t": "Test_Time", "T": "Temperature", "date_time": "DateTime",
    "data_point": "Data_Point", "step_index": "Step_Index", "step_time": "Step_Time",
    "internal_resistance": "Internal_Resistance", "charge_energy": "Charge_Energy",
    "discharge_energy": "Discharge_Energy", "voltage_time_derivative": "dV/dt",
    "aux_voltage": "Aux_Voltage",
}


def _record(frame, metadata, ordinal, start_row, closed_by_next, file_complete,
            run_offsets, run_evidence, run_terminal):
    cycle_index = int(frame["Cycle_Index"].iloc[0])
    run_index = int(frame["__run_index"].iloc[0])
    output = {key: frame[column].to_numpy(dtype=np.float64, copy=True) if column in frame else np.full(len(frame), np.nan)
              for key, column in ARRAY_COLUMNS.items()}
    time = output["t"]
    data_point = output["data_point"]
    output.update({
        "cell_uid": metadata.get("cell_uid"), "source_batch": metadata.get("source_batch"),
        "source_cell_index": metadata.get("source_cell_index"),
        "source_segment_id": metadata.get("source_segment_id", metadata.get("test_id")),
        "source_test_id": metadata.get("test_id"),
        "source_file_sha256": metadata.get("sha256", metadata.get("source_file_sha256")),
        "source_cell_barcode_sha256": metadata.get("source_cell_barcode_sha256", metadata.get("barcode_sha256")),
        "source_channel": metadata.get("source_channel"),
        "source_kind": "instrument_csv", "source_record_index": ordinal,
        "source_record_cycle": ordinal + 1, "instrument_cycle_id": cycle_index,
        "run_index": run_index, "instrument_run_cycle_offset": run_offsets[run_index],
        "run_boundary_evidence": run_evidence.get(run_index) if cycle_index == 0 else None,
        "run_terminal_cycle": bool(run_terminal),
        "instrument_original_cycle_id": cycle_index, "original_local_cycle": cycle_index,
        "original_cycle_id": None,
        "source_row_start": start_row, "source_row_stop_exclusive": start_row + len(frame),
        "source_data_point_first": float(data_point[0]),
        "source_data_point_last": float(data_point[-1]),
        "source_cycle_closed_by_next": bool(closed_by_next),
        "source_file_complete": bool(file_complete),
        "source_terminal_cycle": not closed_by_next,
        "initial_discharge_candidate": cycle_index == 0,
        "formation_identity_certified": False,
        "raw_cycle_certified": False,
        "cycle_axis_scope": "native instrument Cycle_Index, global physical UID/continuation mapping assigned separately",
        "t_unit": "s", "current_unit": "A", "capacity_unit": "Ah",
        "source_current_scale_to_A": 1.0, "source_time_scale_to_seconds": 1.0,
        "summary_QD_Ah": None, "summary_QC_Ah": None,
        "waveform_point_counts": {key: len(output[key]) for key in ARRAY_COLUMNS},
        "source_field_names": [column for column in frame.columns if not column.startswith("__")],
        "time_reversal_count": int(np.sum(np.diff(time) < -1e-8)),
        "time_first_s": float(time[0]), "time_last_s": float(time[-1]),
        "data_point_nonincreasing_count": int(np.sum(np.diff(data_point) <= 0)),
    })
    return output


def _restart_evidence(previous, current, source_row):
    before_cycle, after_cycle = float(previous["Cycle_Index"]), float(current["Cycle_Index"])
    initial_step = float(current["Step_Index"]) in (0.0, 1.0)
    point_order = float(current["Data_Point"]) > float(previous["Data_Point"])
    date_order = float(current["DateTime"]) > float(previous["DateTime"])
    authenticated = before_cycle > 0 and after_cycle == 0 and initial_step and point_order and date_order
    evidence = {"source_row_before": source_row - 1, "source_row_after": source_row,
                "previous_instrument_cycle_id": int(before_cycle), "next_instrument_cycle_id": int(after_cycle),
                "previous_Step_Index": float(previous["Step_Index"]), "next_Step_Index": float(current["Step_Index"]),
                "previous_Data_Point": float(previous["Data_Point"]), "next_Data_Point": float(current["Data_Point"]),
                "previous_DateTime": float(previous["DateTime"]), "next_DateTime": float(current["DateTime"]),
                "previous_Test_Time": float(previous["Test_Time"]), "next_Test_Time": float(current["Test_Time"]),
                "restart_rule_authenticated": authenticated}
    if not authenticated:
        raise ValueError("Cycle_Index decrease has insufficient zero/init-Step/Data_Point/DateTime restart evidence: " + str(evidence))
    return evidence


def iter_instrument_cycles(
    csv_path: str | Path,
    source_metadata: dict,
    chunksize: int = 50000,
    incomplete_prefix: bool = False,
):
    """Yield complete observed groups across chunk boundaries in source order.

    An instrument index gap stays a gap. A decrease creates a new run only with
    zero/init-Step/ordered Data_Point/ordered DateTime restart evidence. Other
    decreases and noninteger/missing indices raise a source-axis error.
    A terminal group is yielded with an explicit
    terminal flag, allowing the measurement layer to inspect its completeness.
    For a deliberately downloaded prefix sample, the final unfinished text line
    is discarded and source_file_complete stays False for every returned record.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    path = Path(csv_path)
    source = path
    if incomplete_prefix:
        payload = path.read_bytes()
        if not payload.endswith(b"\n"):
            payload = payload[:payload.rfind(b"\n") + 1]
        source = io.StringIO(payload.decode("utf-8-sig"))
    reader = pd.read_csv(source, chunksize=chunksize, dtype=np.float64, float_precision="round_trip", encoding="utf-8-sig")
    carry = None
    carry_start = 0
    ordinal = 0
    rows_seen = 0
    previous_row = None
    current_run = 0
    run_offsets, run_evidence = {0: 0}, {0: None}
    receipt_complete = source_metadata.get("status", source_metadata.get("download_status")) == "COMPLETE"
    file_complete = bool(source_metadata.get("file_complete", receipt_complete)) and not incomplete_prefix
    try:
        for chunk in reader:
            if not set(REQUIRED_COLUMNS).issubset(chunk.columns):
                raise ValueError("Missing original CSV columns: " + str(sorted(set(REQUIRED_COLUMNS) - set(chunk.columns))))
            if not len(chunk):
                continue
            ids = chunk["Cycle_Index"].to_numpy()
            if not np.isfinite(ids).all() or np.any(ids != np.floor(ids)):
                raise ValueError(f"Instrument Cycle_Index is missing/noninteger near data row {rows_seen}")
            decreases = list(np.flatnonzero(np.diff(ids) < 0) + 1)
            if previous_row is not None and ids[0] < previous_row["Cycle_Index"]:
                decreases.insert(0, 0)
            run_values = np.empty(len(chunk), dtype=np.int64)
            start = 0
            for position in decreases:
                before = previous_row if position == 0 else chunk.iloc[position - 1]
                evidence = _restart_evidence(before, chunk.iloc[position], rows_seen + int(position))
                run_values[start:position] = current_run
                cumulative_offset = run_offsets[current_run] + int(before["Cycle_Index"])
                current_run += 1
                run_offsets[current_run] = cumulative_offset
                run_evidence[current_run] = evidence
                start = int(position)
            run_values[start:] = current_run
            chunk["__run_index"] = run_values
            previous_row = chunk.iloc[-1].copy()
            if carry is not None:
                chunk = pd.concat((carry, chunk), ignore_index=True)
                start_row = carry_start
            else:
                start_row = rows_seen
            rows_seen += len(ids)
            combined_ids = chunk["Cycle_Index"].to_numpy()
            combined_runs = chunk["__run_index"].to_numpy()
            boundaries = np.r_[0, np.flatnonzero((np.diff(combined_ids) != 0) | (np.diff(combined_runs) != 0)) + 1, len(chunk)]
            for left, right in zip(boundaries[:-2], boundaries[1:-1]):
                group = chunk.iloc[left:right]
                terminal = combined_runs[left] != combined_runs[right]
                yield _record(group, source_metadata, ordinal, start_row + int(left), True, file_complete,
                              run_offsets, run_evidence, terminal)
                ordinal += 1
            last_start = int(boundaries[-2])
            carry = chunk.iloc[last_start:].copy()
            carry_start = start_row + last_start
        if carry is not None and len(carry):
            yield _record(carry, source_metadata, ordinal, carry_start, False, file_complete,
                          run_offsets, run_evidence, True)
    finally:
        reader.close()


def compare_instrument_release(instrument_record, released_record):
    """Independent numeric binding evidence without using summary cycle order.

    The public author export scales I by 1.1 and time by 60, resetting each
    cycle's time origin and sometimes applying its published negative-time fix.
    Full V/I/Qc/Qd/t matching supplies a record binding. A singleton/ambiguous
    binding is still resolved at the source identity registry layer.
    """
    if len(instrument_record["V"]) != len(released_record["V"]):
        return {"matches": False, "reason": "point_count_difference", "instrument_points": len(instrument_record["V"]), "release_points": len(released_record["V"])}
    raw_time = np.asarray(instrument_record["t"], dtype=float)
    exported_time = (raw_time - raw_time[0]) / 60
    if exported_time[-1] < 0:
        negative = np.flatnonzero(exported_time < 0)
        first = int(negative[0])
        if first == 0:
            return {"matches": False, "reason": "published_negative_time_fix_undefined"}
        exported_time[first:] += exported_time[first - 1] - exported_time[first]
    expected = {"V": instrument_record["V"], "I": np.asarray(instrument_record["I"]) / 1.1,
                "Qc": instrument_record["Qc"], "Qd": instrument_record["Qd"], "t": exported_time}
    differences = {}
    for field, values in expected.items():
        target = np.asarray(released_record[field], dtype=float)
        source = np.asarray(values, dtype=float)
        equal = np.allclose(source, target, rtol=1e-9, atol=1e-10, equal_nan=True)
        finite = np.isfinite(source) & np.isfinite(target)
        differences[field] = {"matches": bool(equal), "max_abs_difference": float(np.max(np.abs(source[finite] - target[finite]))) if finite.any() else None}
    return {"matches": all(item["matches"] for item in differences.values()),
            "instrument_cycle_id": int(instrument_record["instrument_cycle_id"]),
            "release_source_record_index": int(released_record["source_record_index"]),
            "fields": differences, "identity_scope": "One numeric record binding, global uniqueness checked separately"}
