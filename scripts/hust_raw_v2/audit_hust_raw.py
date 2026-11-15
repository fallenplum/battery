"""Audit official HUST bytes as a symbolic graph, without pickle execution."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from hust_safe_pickle import dataframe_columns, parse_pickle_data, symbol_name

OUT = REPO / "results" / "hust_raw_idea_audit_v2"
PRIVATE = REPO.parent / "private" / "raw_sources" / "HUST_IDEA_V2"
OLD_SOURCE = PRIVATE.parent / "HUST"
PHASES = (
    "Constant current charge",
    "Constant current-constant voltage charge",
    "Constant current discharge_0",
    "Constant current discharge_1",
    "Constant current discharge_2",
    "Constant current discharge_3",
)
FIELDS = ("Status", "Cycle number", "Current (mA)", "Voltage (V)", "Capacity (mAh)", "Time (s)")
UNITS = ("phase label", "source cycle count", "mA", "V", "mAh, source net coordinate", "s, within-cycle coordinate")
VIEWS = (
    ("D3_2p8_3p1", PHASES[5], 2.8, 3.1, -2200.0),
    ("CC_3p1_3p4", PHASES[0], 3.1, 3.4, 5500.0),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def scalar(value):
    if isinstance(value, (int, float, bool)):
        return value
    if symbol_name(value) not in ("numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar"):
        raise ValueError("Unsupported scalar constructor")
    dtype_symbol, payload = value.args
    if symbol_name(dtype_symbol) != "numpy.dtype":
        raise ValueError("Scalar dtype must be explicit")
    dtype = np.dtype(dtype_symbol.args[0]).newbyteorder(dtype_symbol.state[1])
    if dtype.kind not in "biuf" or dtype.hasobject or not isinstance(payload, bytes) or len(payload) != dtype.itemsize:
        raise ValueError("Unsupported scalar payload")
    return np.frombuffer(payload, dtype=dtype)[0].item()


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "min": float(values.min()), "p05": float(np.quantile(values, .05)),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def empty_quality():
    return {"points": 0, "missing": 0, "min": float("inf"), "max": float("-inf"), "dtypes": set()}


def add_quality(record, values):
    record["points"] += len(values)
    record["dtypes"].add(str(values.dtype))
    if values.dtype.kind == "O":
        record["missing"] += sum(v is None or v == "" for v in values)
        return
    good = np.isfinite(values)
    record["missing"] += int((~good).sum())
    if good.any():
        record["min"] = min(record["min"], float(values[good].min()))
        record["max"] = max(record["max"], float(values[good].max()))


def closure_accumulator():
    return {"n": 0, "sum_abs": 0.0, "sum_square": 0.0, "sum_signed": 0.0, "max_abs": 0.0,
            "within_1e_9": 0, "within_1e_6": 0, "within_1e_3": 0}


def add_closure(accumulator, residual):
    residual = residual[np.isfinite(residual)]
    absolute = np.abs(residual)
    accumulator["n"] += len(residual)
    accumulator["sum_abs"] += float(absolute.sum())
    accumulator["sum_square"] += float(np.dot(residual, residual))
    accumulator["sum_signed"] += float(residual.sum())
    if len(absolute):
        accumulator["max_abs"] = max(accumulator["max_abs"], float(absolute.max()))
    for threshold, key in ((1e-9, "within_1e_9"), (1e-6, "within_1e_6"), (1e-3, "within_1e_3")):
        accumulator[key] += int((absolute <= threshold).sum())


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    identity = json.loads((OUT / "CURRENT_ARCHIVE_IDENTITY.json").read_text(encoding="utf-8"))
    cache = Path(identity["replay_inner_cache_path"])
    if sha256(cache) != identity["inner_sha256"]:
        raise ValueError("Current-inner replay identity changed")
    receipt = json.loads((OLD_SOURCE / "SOURCE_RECOVERY_RECEIPT.json").read_text(encoding="utf-8"))
    old_members = {v["member"]: v for v in receipt["members"]}
    source_stat = Path(identity["outer_path"]).stat()
    if source_stat.st_size != identity["outer_bytes"] or source_stat.st_mtime_ns != identity["outer_mtime_ns"]:
        raise ValueError("Current outer archive changed after its full SHA/CRC verification")
    cells, field_rows, phase_rows, closure_rows, window_rows, member_rows = [], [], [], [], [], []
    globals_all, root_schemas, frame_schemas, attrs_schemas = set(), Counter(), Counter(), Counter()
    duplicate_hashes, duplicates = {}, []
    cycle_path = PRIVATE / "all_cycle_quality.csv"
    cycle_fields = ["cell_id", "cycle", "points", "phase_count", "phase_order_valid", "cycle_column_valid", "rul", "dq_mAh", "net_discharge_mAh", "dq_residual_mAh", "charge_capacity_mAh", "charge_first_capacity_mAh", "charge_first_time_s", "end_voltage_V", "duration_s", "negative_time_steps", "duplicate_time_steps", "boundary_capacity_max_abs_mAh", "boundary_gap_min_s", "boundary_gap_max_s"]
    with zipfile.ZipFile(cache) as archive, cycle_path.open("w", encoding="utf-8-sig", newline="") as cycle_stream:
        writer = csv.DictWriter(cycle_stream, fieldnames=cycle_fields)
        writer.writeheader()
        members = sorted((v for v in archive.infolist() if not v.is_dir()), key=lambda v: tuple(int(s) for s in Path(v.filename).stem.split("-")))
        if set(v.filename for v in members) != set(old_members):
            raise ValueError("Archive member set differs from official receipt")
        for member_index, member in enumerate(members, 1):
            payload = archive.read(member)  # zipfile verifies this member's CRC at EOF.
            digest = hashlib.sha256(payload).hexdigest()
            old = old_members[member.filename]
            if digest != old["sha256"] or len(payload) != old["bytes"]:
                raise ValueError("Member content differs from official receipt")
            member_rows.append({"member": member.filename, "bytes": len(payload), "CRC32": format(member.CRC, "08x"), "CRC32_verified": True, "sha256": digest, "matches_old_official": True})
            graph, globals_seen = parse_pickle_data(payload)
            stem = Path(member.filename).stem
            if list(graph) != [stem]:
                raise ValueError("Source root identity differs from filename")
            source = graph[stem]
            globals_all.update(globals_seen)
            root_schemas[json.dumps(list(source), sort_keys=True)] += 1
            data = source["data"]
            numbers = sorted(data)
            n_cycles = len(numbers)
            if numbers != list(range(1, n_cycles + 1)):
                raise ValueError("Nonconsecutive original source cycle axis")
            if set(source["dq"]) != set(data) or set(source["rul"]) != set(data):
                raise ValueError("Source label keys are not aligned")
            cell_id = "HUST_" + stem
            quality = {name: empty_quality() for name in (*FIELDS, "rul", "dq")}
            phase = {name: {"cycles": 0, "points": 0, "current_medians": [], "current_min": float("inf"), "current_max": float("-inf"), "voltage_min": float("inf"), "voltage_max": float("-inf"), "capacity_min": float("inf"), "capacity_max": float("-inf"), "dt": Counter(), "q_wrong_direction": 0, "v_wrong_direction": 0, "v_wrong_gt10mV": 0, "nonfinite": 0, "effective_dt_cycle_medians": [], "effective_minus_actual_dt_abs_sum": 0.0, "effective_minus5_abs_sum": 0.0, "effective_dt_n": 0} for name in PHASES}
            closure = {(name, grid, rule): closure_accumulator() for name in PHASES for grid in ("recorded_dt", "fixed_5s") for rule in ("left", "right", "trapezoid")}
            counts = Counter()
            raw_qd, label_dq, label_rul, durations, boundary_gaps, charge_q0, charge_qend, end_v, dq_errors = [], [], [], [], [], [], [], [], []
            for number in numbers:
                columns = dataframe_columns(data[number])
                if tuple(columns) != FIELDS:
                    raise ValueError("Unexpected raw field schema")
                frame_schemas[json.dumps({name: str(value.dtype) for name, value in columns.items()}, sort_keys=True)] += 1
                frame_state = data[number].state
                attrs_schemas[json.dumps({k: v for k, v in frame_state.items() if k != "_mgr"}, sort_keys=True)] += 1
                status, axis, current, voltage, capacity, clock = (columns[name] for name in FIELDS)
                for name, values in columns.items():
                    add_quality(quality[name], values)
                rul, dq_label = scalar(source["rul"][number]), scalar(source["dq"][number])
                add_quality(quality["rul"], np.asarray([rul]))
                add_quality(quality["dq"], np.asarray([dq_label]))
                label_rul.append(rul)
                label_dq.append(dq_label)
                bounds = np.r_[0, np.flatnonzero(status[1:] != status[:-1]) + 1, len(status)]
                observed_phases = tuple(str(status[left]) for left in bounds[:-1])
                order_valid = observed_phases == PHASES
                counts["phase_order_invalid_cycles"] += int(not order_valid)
                counts["cycle_column_mismatch_cycles"] += int(not np.all(axis == number))
                if not order_valid:
                    raise ValueError("Unexpected source phase order")
                dt_all = np.diff(clock)
                negative_t, duplicate_t = int((dt_all < 0).sum()), int((dt_all == 0).sum())
                counts["negative_time_steps"] += negative_t
                counts["duplicate_time_steps"] += duplicate_t
                counts["rul_not_N_minus_cycle_plus1"] += int(rul != n_cycles - number + 1)
                boundaries = bounds[1:-1]
                boundary_changes = capacity[boundaries] - capacity[boundaries - 1]
                gaps = clock[boundaries] - clock[boundaries - 1]
                boundary_gaps.extend(gaps.tolist())
                counts["boundary_capacity_changes_gt1e_6"] += int((np.abs(boundary_changes) > 1e-6).sum())
                qd = float(capacity[bounds[2]] - capacity[-1])
                dq_error = float(dq_label - qd)
                dq_errors.append(dq_error)
                counts["dq_not_net_discharge_within1e_6"] += int(abs(dq_error) > 1e-6)
                counts["dq_below_or_equal_0p8_nominal"] += int(qd <= 880.0)
                raw_qd.append(qd)
                durations.append(int(clock[-1] - clock[0]))
                charge_q0.append(float(capacity[0]))
                charge_qend.append(float(capacity[bounds[2] - 1]))
                end_v.append(float(voltage[-1]))
                digest_cycle = hashlib.sha256()
                for values in (current, voltage, capacity, clock):
                    digest_cycle.update(values.tobytes())
                digest_cycle.update(json.dumps(observed_phases).encode("utf-8"))
                digest_cycle.update(np.diff(bounds).tobytes())
                signature = digest_cycle.hexdigest()
                if signature in duplicate_hashes:
                    duplicates.append({"cell_id": cell_id, "cycle": number, "same_as": duplicate_hashes[signature]})
                else:
                    duplicate_hashes[signature] = [cell_id, number]
                for name, left, right in zip(observed_phases, bounds[:-1], bounds[1:]):
                    record = phase[name]
                    i, v, q, t = current[left:right], voltage[left:right], capacity[left:right], clock[left:right]
                    dt, dcap = np.diff(t), np.diff(q)
                    record["cycles"] += 1
                    record["points"] += len(i)
                    record["current_medians"].append(float(np.median(i)))
                    record["current_min"] = min(record["current_min"], float(i.min()))
                    record["current_max"] = max(record["current_max"], float(i.max()))
                    record["voltage_min"] = min(record["voltage_min"], float(v.min()))
                    record["voltage_max"] = max(record["voltage_max"], float(v.max()))
                    record["capacity_min"] = min(record["capacity_min"], float(q.min()))
                    record["capacity_max"] = max(record["capacity_max"], float(q.max()))
                    values, frequencies = np.unique(dt, return_counts=True)
                    record["dt"].update(dict(zip((int(x) for x in values), (int(x) for x in frequencies))))
                    sign = 1 if name in PHASES[:2] else -1
                    record["q_wrong_direction"] += int((sign * dcap < -1e-6).sum())
                    record["v_wrong_direction"] += int((sign * np.diff(v) < -1e-9).sum())
                    record["v_wrong_gt10mV"] += int((sign * np.diff(v) < -.01).sum())
                    record["nonfinite"] += int((~(np.isfinite(i) & np.isfinite(v) & np.isfinite(q) & np.isfinite(t))).sum())
                    currents = {"left": i[:-1], "right": i[1:], "trapezoid": (i[:-1] + i[1:]) * .5}
                    for grid in ("recorded_dt", "fixed_5s"):
                        seconds = dt if grid == "recorded_dt" else 5.0
                        for rule, input_current in currents.items():
                            add_closure(closure[(name, grid, rule)], dcap - input_current * seconds / 3600)
                    average = currents["trapezoid"]
                    valid_effective = np.isfinite(dcap) & (np.abs(average) > .01)
                    effective = 3600 * dcap[valid_effective] / average[valid_effective]
                    if len(effective):
                        record["effective_dt_cycle_medians"].append(float(np.median(effective)))
                        record["effective_minus_actual_dt_abs_sum"] += float(np.abs(effective - dt[valid_effective]).sum())
                        record["effective_minus5_abs_sum"] += float(np.abs(effective - 5).sum())
                        record["effective_dt_n"] += len(effective)
                if number <= 100:
                    for view, selected_phase, lo, hi, expected_current in VIEWS:
                        phase_index = PHASES.index(selected_phase)
                        left, right = bounds[phase_index:phase_index + 2]
                        pv, pi, pq, pt = voltage[left:right], current[left:right], capacity[left:right], clock[left:right]
                        inside = (pv >= lo) & (pv <= hi)
                        sv, si, sq, st = pv[inside], pi[inside], pq[inside], pt[inside]
                        sign = 1 if expected_current > 0 else -1
                        window_rows.append({"cell_id": cell_id, "raw_cycle": number, "view": view, "phase": selected_phase, "lo_V": lo, "hi_V": hi,
                            "native_points": int(inside.sum()), "unique_voltages": len(np.unique(sv)), "phase_brackets_both_endpoints": bool(pv.min() <= lo and pv.max() >= hi),
                            "phase_min_V": float(pv.min()), "phase_max_V": float(pv.max()), "window_observed_min_V": float(sv.min()) if len(sv) else None,
                            "window_observed_max_V": float(sv.max()) if len(sv) else None, "current_median_mA": float(np.median(si)) if len(si) else None,
                            "current_max_abs_deviation_from_declared_mA": float(np.abs(si - expected_current).max()) if len(si) else None,
                            "current_median_within_0p05C": bool(len(si) and abs(np.median(si) - expected_current) <= 55),
                            "voltage_direction_violations": int((sign * np.diff(sv) < -1e-9).sum()), "voltage_direction_violations_gt10mV": int((sign * np.diff(sv) < -.01).sum()),
                            "capacity_direction_violations": int((sign * np.diff(sq) < -1e-6).sum()), "nonpositive_time_steps": int((np.diff(st) <= 0).sum()),
                            "observed_capacity_span_mAh": float(np.abs(sq[-1] - sq[0])) if len(sq) else None})
                writer.writerow(dict(zip(cycle_fields, [cell_id, number, len(status), len(observed_phases), order_valid, bool(np.all(axis == number)), rul, dq_label, qd, dq_error,
                    capacity[bounds[2]-1] - capacity[0], capacity[0], clock[0], voltage[-1], clock[-1]-clock[0], negative_t, duplicate_t,
                    float(np.abs(boundary_changes).max()), int(gaps.min()), int(gaps.max())])))
            for name, unit in zip((*FIELDS, "rul", "dq"), (*UNITS, "remaining recorded cycles inclusive", "mAh, net discharge")):
                record = quality[name]
                field_rows.append({"cell_id": cell_id, "field": name, "unit_semantics": unit, "points": record["points"], "missing_or_nonfinite": record["missing"],
                    "dtype": sorted(record["dtypes"]), "min": record["min"] if np.isfinite(record["min"]) else None, "max": record["max"] if np.isfinite(record["max"]) else None})
            for name, record in phase.items():
                medians = record.pop("current_medians")
                effective_medians = record.pop("effective_dt_cycle_medians")
                n_effective = record.pop("effective_dt_n")
                effective_actual_abs = record.pop("effective_minus_actual_dt_abs_sum")
                effective_five_abs = record.pop("effective_minus5_abs_sum")
                phase_rows.append({"cell_id": cell_id, "phase": name, **record, "current_cycle_median_stats_mA": stats(medians),
                    "effective_dt_cycle_median_stats_s": stats(effective_medians), "effective_dt_steps": n_effective,
                    "effective_minus_actual_dt_MAE_s": effective_actual_abs / n_effective if n_effective else None,
                    "effective_minus5_MAE_s": effective_five_abs / n_effective if n_effective else None})
            for (name, grid, rule), acc in closure.items():
                n = acc["n"]
                closure_rows.append({"cell_id": cell_id, "phase": name, "time_grid": grid, "quadrature": rule, "steps": n,
                    "residual_MAE_mAh": acc["sum_abs"] / n if n else None, "residual_RMSE_mAh": np.sqrt(acc["sum_square"] / n) if n else None,
                    "residual_mean_mAh": acc["sum_signed"] / n if n else None, "residual_max_abs_mAh": acc["max_abs"],
                    "within_1e_9_mAh": acc["within_1e_9"], "within_1e_6_mAh": acc["within_1e_6"], "within_1e_3_mAh": acc["within_1e_3"]})
            cells.append({"cell_id": cell_id, "member": member.filename, "raw_cycles": n_cycles, "first_cycle": numbers[0], "last_cycle": numbers[-1],
                "raw_axis_contiguous": True, "data_rul_dq_keys_aligned": True, "K100_source_cycles": 100, "raw_points": quality["Status"]["points"],
                **dict(counts), "rul_stats": stats(label_rul), "dq_stats_mAh": stats(label_dq), "net_discharge_stats_mAh": stats(raw_qd),
                "dq_minus_net_discharge_max_abs_mAh": float(np.abs(dq_errors).max()), "charge_start_stats_mAh": stats(charge_q0),
                "charge_end_stats_mAh": stats(charge_qend), "cycle_duration_stats_s": stats(durations), "boundary_gap_stats_s": stats(boundary_gaps),
                "last_discharge_voltage_stats_V": stats(end_v)})
            cycle_stream.flush()
            print(json.dumps({"cells_done": member_index, "cells_total": len(members), "cell_id": cell_id, "cycles": n_cycles, "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
            del graph, payload, data, source, columns
            gc.collect()
    for filename, rows in (("raw_cell_metadata.csv", cells), ("raw_field_quality.csv", field_rows), ("phase_quality.csv", phase_rows),
                           ("capacity_integration_closure.csv", closure_rows), ("early_window_support.csv", window_rows), ("archive_member_verification.csv", member_rows)):
        write_csv(OUT / filename, rows)
    support = {}
    for name, *_ in VIEWS:
        selected = [row for row in window_rows if row["view"] == name]
        support[name] = {"prefix_cycles": len(selected), "native_points": stats([r["native_points"] for r in selected]),
            "bracketed_cycles": sum(r["phase_brackets_both_endpoints"] for r in selected),
            "current_median_within_0p05C_cycles": sum(r["current_median_within_0p05C"] for r in selected),
            "voltage_direction_violations": sum(r["voltage_direction_violations"] for r in selected),
            "voltage_direction_violations_gt10mV": sum(r["voltage_direction_violations_gt10mV"] for r in selected),
            "capacity_direction_violations": sum(r["capacity_direction_violations"] for r in selected),
            "nonpositive_time_steps": sum(r["nonpositive_time_steps"] for r in selected),
            "threshold_support": {str(threshold): {"cycles": sum(r["native_points"] >= threshold for r in selected),
                "cells_all100": sum(all(r["native_points"] >= threshold for r in selected if r["cell_id"] == cell["cell_id"]) for cell in cells)} for threshold in (2, 5, 8, 10, 15, 20, 25, 30)}}
    summary = {"status": "FULL_CURRENT_ARCHIVE_REPLAY_COMPLETE", "current_archive_identity": identity, "safe_parser_sha256": sha256(REPO / "scripts" / "hust_safe_pickle.py"),
        "audit_script_sha256": sha256(Path(__file__)), "read_strategy": "Current outer ZIP inner stream fully CRC/SHA verified, replay byte-identical old inner cache, all 77 member bytes CRC/SHA verified again and symbolically parsed",
        "cells": len(cells), "cycles": sum(r["raw_cycles"] for r in cells), "points": sum(r["raw_points"] for r in cells),
        "all_members_match_old_official": all(r["matches_old_official"] for r in member_rows), "all_member_CRC32_rechecked": True,
        "globals_seen_symbolically_never_executed": sorted(globals_all), "root_schemas": dict(root_schemas), "dataframe_schemas": dict(frame_schemas), "dataframe_attrs_states": dict(attrs_schemas),
        "counts": {key: sum(r.get(key, 0) for r in cells) for key in ("phase_order_invalid_cycles", "cycle_column_mismatch_cycles", "negative_time_steps", "duplicate_time_steps", "rul_not_N_minus_cycle_plus1", "boundary_capacity_changes_gt1e_6", "dq_not_net_discharge_within1e_6", "dq_below_or_equal_0p8_nominal")},
        "missing_or_nonfinite_numeric_and_labels": sum(r["missing_or_nonfinite"] for r in field_rows),
        "exact_duplicate_cycle_content_count_excluding_cycle_label": len(duplicates), "duplicate_examples": duplicates[:100], "early_window_support": support,
        "native_field_capacity": "Source-provided signed/net Capacity (mAh). Independent device-measured provenance and hidden sampling/integration algorithm are unauthenticated.",
        "rul_semantics": "N_recorded_cycles - original_cycle + 1, inclusive administrative remaining record count. This formula alone does not establish an observed 80%-nominal event.",
        "dq_semantics": "Original source label equals Capacity at first D0 sample minus Capacity at last D3 sample, in mAh, checked at tolerance 1e-6mAh.",
        "missing_raw_metadata": ["measured temperature", "calendar timestamps", "instrument export headers and calibration", "rest-only rows", "formation/diagnostic flags", "electrode loading and geometry", "electrolyte", "impedance"],
        "external_declarations": {"chemistry": "LFP/graphite", "cell_model": "A123 APR18650M1A", "nominal_Ah": 1.1, "nominal_V": 3.3, "chamber_C": 30,
            "tester": "80-channel Neware CT-4008", "charge": "5C to80%SOC, 1C to3.6V, CV toC/20", "terminal_discharge_paper_C": 1,
            "terminal_discharge_observed_C": 2, "rest_paper_s": 30, "initial_cells_paper": 80, "fault_exclusions_paper": 3,
            "sources": ["https://data.mendeley.com/datasets/nsc7hnsg4s/2", "https://pubs.rsc.org/en/content/articlehtml/2022/ee/d2ee01676a"]},
        "private_cycle_quality_path": str(cycle_path), "private_cycle_quality_sha256": sha256(cycle_path), "elapsed_s": time.monotonic() - started}
    (OUT / "HUST_RAW_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "cells": summary["cells"], "cycles": summary["cycles"], "points": summary["points"], "counts": summary["counts"], "elapsed_s": summary["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
