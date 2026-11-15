"""Read released MATR source-record prefixes without asserting physical K100.

The author discarded instrument Cycle_Index during export. This audit keeps
source record order explicit and performs source-unit/phase/counter/Q(V) QA.
It generates no model input cohort, target, DATA_FREEZE or formal model score.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from measurements import measure_cycle
from raw_reader import cell_metadata, iter_raw_cycles


REPOSITORY = Path(__file__).resolve().parents[2]
SAMPLES = {
    1: {0: [], 1: [], 18: []},
    2: {0: [], 7: [], 12: list(range(248, 257)), 44: list(range(243, 252))},
    3: {0: []},
    4: {0: []},
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def inspect_record(record, measured, settings):
    current_native = np.asarray(record["I"], dtype=float)
    qd = np.asarray(record["Qd"], dtype=float)
    voltage = np.asarray(record["V"], dtype=float)
    time_native = np.asarray(record["t"], dtype=float)
    fields = [np.asarray(record[name], dtype=float) for name in ("V", "I", "Qd", "Qc", "t", "T")]
    negative = current_native < 0
    output = {
        "source_batch": record["source_batch"], "source_segment_id": record["source_segment_id"],
        "physical_cell_uid": record["cell_uid"], "source_record_index": record["source_record_index"],
        "source_record_cycle": record["source_record_cycle"],
        "summary_cycle_sequence": record["summary_cycle_sequence"],
        "instrument_original_cycle_id": None, "raw_cycle_certified": False,
        "all_fields_same_length": len({len(field) for field in fields}) == 1,
        "all_fields_finite": all(np.isfinite(field).all() for field in fields),
        "waveform_point_count": len(voltage),
        "native_negative_current_median": np.median(current_native[negative]) if negative.any() else np.nan,
        "restored_negative_current_median_A": np.median(current_native[negative]) * settings["source_current_scale_to_A"] if negative.any() else np.nan,
        "native_t_duration_minutes": time_native[-1] - time_native[0] if len(time_native) else np.nan,
        "restored_t_duration_seconds": (time_native[-1] - time_native[0]) * settings["source_time_scale_to_seconds"] if len(time_native) else np.nan,
        "native_time_reversal_count": int(np.sum(np.diff(time_native) < -1e-8)),
        "summary_QD_Ah": record["summary_QD_Ah"],
        "raw_QD_max_Ah": np.max(qd) if len(qd) else np.nan,
        "summary_minus_raw_max_QD_Ah": record["summary_QD_Ah"] - np.max(qd) if len(qd) else np.nan,
        "capacity_valid": measured["capacity_valid"], "curve_valid": measured["curve_valid"],
        "phase_capacity_Ah": measured["capacity_Ah"],
        "summary_counter_difference_Ah": measured["summary_counter_difference_Ah"],
        "discharge_counter_zero_Ah": measured.get("instrument_discharge_counter_zero_Ah"),
        "integration_counter_ratio": measured.get("integration_counter_ratio"),
        "time_valid": measured.get("time_valid", False),
        "integration_valid": measured.get("integration_valid", False),
        "discharge_point_start": measured.get("discharge_point_start"),
        "discharge_point_stop_exclusive": measured.get("discharge_point_stop_exclusive"),
        "phase_boundary_basis": measured.get("phase_boundary_basis"),
        "measurement_warnings": "|".join(measured.get("warning_reasons", [])),
        "discharge_points": measured["discharge_points"], "window_points": measured["window_points"],
        "q_absolute_at_2p8V_Ah": measured["q_absolute"][0] if measured["curve_valid"] else np.nan,
        "q_absolute_at_3p3V_Ah": measured["q_absolute"][-1] if measured["curve_valid"] else np.nan,
        "q_shape_window_span_Ah": measured["q_absolute"][0] - measured["q_absolute"][-1] if measured["curve_valid"] else np.nan,
        "q_grid_nondecreasing_with_discharge": bool(np.all(np.diff(measured["q_absolute"]) <= settings["capacity_reset_tolerance_Ah"])) if measured["curve_valid"] else False,
        "failure_reasons": "|".join(measured["failure_reasons"]),
        "source_record_waveform_sha256": measured["raw_fingerprint"],
    }
    # Independently recompute SI arithmetic on the explicitly selected phase.
    # Phase selection itself is the measurement module's result, so this check
    # does not independently certify the phase detector.
    independent_ratio = np.nan
    restored_current = current_native * settings["source_current_scale_to_A"]
    restored_time = time_native * settings["source_time_scale_to_seconds"]
    phase_start = measured.get("discharge_point_start")
    phase_stop = measured.get("discharge_point_stop_exclusive")
    if (measured.get("time_valid", False)
            and phase_start is not None and phase_stop is not None
            and phase_stop > phase_start
            and len({len(current_native), len(qd), len(time_native)}) == 1):
        span = np.arange(phase_start, phase_stop)
        delta_q = qd[span[-1]] - qd[span[0]]
        if delta_q > 0:
            independent_ratio = np.trapz(-restored_current[span], restored_time[span]) / 3600 / delta_q
    output["independent_integration_counter_ratio"] = independent_ratio
    output["integration_replay_scope"] = "independent SI arithmetic conditional on stored phase bounds"
    stored_ratio = measured.get("integration_counter_ratio")
    comparable = stored_ratio is not None and np.isfinite(stored_ratio) and np.isfinite(independent_ratio)
    output["integration_replay_comparable"] = comparable
    output["integration_replay_agrees"] = bool(np.isclose(independent_ratio, stored_ratio, atol=1e-12)) if comparable else None
    return output


def audit_sources(source_root, config):
    module_paths = {name: Path(__file__).with_name(name) for name in ("measurements.py", "raw_reader.py")}
    module_hashes_before = {name: sha256(path) for name, path in module_paths.items()}
    settings = config["discharge"]
    voltage = np.linspace(2.8, 3.3, 64)
    all_rows, cells, unavailable = [], [], []
    for batch, sample_cells in SAMPLES.items():
        matches = sorted(source_root.glob({1: "2017-05-12", 2: "2017-06-30", 3: "2018-04-12", 4: "2019-01-24"}[batch] + "*.mat"))
        if len(matches) != 1:
            unavailable.append({"batch": batch, "reason": "complete original archive unavailable"})
            continue
        path = matches[0]
        receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "COMPLETE" or path.stat().st_size != receipt["bytes"]:
            unavailable.append({"batch": batch, "reason": "archive receipt incomplete"})
            continue
        for source_cell_index, extra_indices in sample_cells.items():
            meta, summary = cell_metadata(path, source_cell_index, batch)
            selected = set(range(100)) | set(extra_indices)
            rows = []
            for record in iter_raw_cycles(path, source_cell_index, meta, summary):
                record_index = record["source_record_index"]
                if record_index > max(selected):
                    break
                if record_index not in selected:
                    continue
                measured = measure_cycle(record, settings, voltage)
                row = inspect_record(record, measured, settings)
                row["source_file_sha256"] = receipt["sha256"]
                row["audit_region"] = "first100_source_records" if record_index < 100 else "legacy_shift_neighborhood"
                rows.append(row)
            frame = pd.DataFrame(rows)
            reasons = collections.Counter(reason for row in rows for reason in row["failure_reasons"].split("|") if reason)
            warnings = collections.Counter(reason for row in rows for reason in row["measurement_warnings"].split("|") if reason)
            cell_summary = {
                "source_cell_id": meta["source_cell_id"], "cell_uid": meta["cell_uid"],
                "source_file_sha256": receipt["sha256"], "source_batch": batch,
                "source_record_count": meta["raw_records"], "records_read": len(rows),
                "first100_records_read": int((frame.audit_region == "first100_source_records").sum()),
                "extra_legacy_shift_records_read": int((frame.audit_region == "legacy_shift_neighborhood").sum()),
                "summary_waveform_count_aligned": meta["summary_waveform_record_count_aligned"],
                "source_summary_sequence_first": meta["summary_sequence_first"],
                "source_summary_sequence_last": meta["summary_sequence_last"],
                "instrument_cycle_axis_authenticated": False, "raw_cycle_certified": False,
                "capacity_valid_count": int(frame.capacity_valid.sum()),
                "curve_valid_count": int(frame.curve_valid.sum()),
                "time_valid_count": int(frame.time_valid.sum()),
                "integration_valid_count": int(frame.integration_valid.sum()),
                "warning_counts": dict(warnings),
                "native_negative_current_median": float(frame.native_negative_current_median.median()),
                "restored_negative_current_median_A": float(frame.restored_negative_current_median_A.median()),
                "integration_counter_ratio_quantiles": frame.integration_counter_ratio.quantile([0, 0.5, 1]).to_dict(),
                "valid_capacity_integration_ratio_quantiles": frame.loc[frame.capacity_valid, "integration_counter_ratio"].quantile([0, 0.5, 1]).to_dict(),
                "independent_integration_replay_comparable_count": int(frame.integration_replay_comparable.sum()),
                "independent_integration_replay_all_comparable_agree": bool(frame.loc[frame.integration_replay_comparable, "integration_replay_agrees"].all()),
                "summary_minus_raw_max_QD_absmax_Ah": float(frame.summary_minus_raw_max_QD_Ah.abs().max()),
                "q_absolute_at_2p8V_range_Ah": frame.q_absolute_at_2p8V_Ah.agg(["min", "max"]).to_dict(),
                "q_absolute_at_3p3V_range_Ah": frame.q_absolute_at_3p3V_Ah.agg(["min", "max"]).to_dict(),
                "failure_counts": dict(reasons),
            }
            cells.append(cell_summary)
            all_rows.extend(rows)
            print(json.dumps({"source_cell": meta["source_cell_id"], "source_records_read": len(rows), "curve_valid": int(frame.curve_valid.sum()), "instrument_axis_certified": False}), flush=True)
    complete = not unavailable and len(cells) == sum(len(sample) for sample in SAMPLES.values())
    module_hashes_after = {name: sha256(path) for name, path in module_paths.items()}
    code_stable = module_hashes_before == module_hashes_after
    result = {
        "status": "SOURCE_RECORD_AUDIT_COMPLETE" if complete and code_stable else "PARTIAL_SOURCE_RECORD_AUDIT",
        "scope": "Released source-record prefixes and legacy-shift neighborhoods, independent of instrument physical cycle certification",
        "instrument_axis_certified": False, "DATA_FREEZE_allowed": False,
        "units": {"current": "source I *1.1 -> A", "time": "source t *60 -> seconds", "capacity": "device Qd in Ah"},
        "author_source_url": "https://raw.githubusercontent.com/chueh-ermon/BMS-autoanalysis/master/cell_analysis.m",
        "measurement_module_sha256": module_hashes_before["measurements.py"],
        "reader_module_sha256": module_hashes_before["raw_reader.py"],
        "source_modules_unchanged_during_audit": code_stable,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "cells": cells, "records_read": len(all_rows), "unavailable_source_batches": unavailable,
        "raw_data_unchanged": True,
    }
    return result, pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/unified_matr_seed42.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result, records = audit_sources(args.source_root, config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_root / "source_record_measurement_audit.csv", index=False)
    (args.output_root / "SOURCE_ORDER_PREFIX_AUDIT.json").write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cells": len(result["cells"]), "records": result["records_read"]}))


if __name__ == "__main__":
    main()
