"""Read all HUST original cycles1..100 for the fixed two-view support audit."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from scripts.hust_safe_pickle import dataframe_columns, parse_pickle_data
from scripts.hust_raw_v2.measurements import VIEWS, measure_view, phase_audit, view_definitions


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            values = json_ready(row)
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in values.items()})


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max()), "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recovery_path = args.source_root / "SOURCE_RECOVERY_RECEIPT.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if recovery["actual_sha256"] != recovery["official_sha256"] or len(recovery["members"]) != 77:
        raise RuntimeError("The prefix audit requires the authenticated77-member source")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "PREFIX_SUPPORT_COMPLETE.json").exists():
        raise RuntimeError("Prefix support audit already completed")
    rows, stages, sources = [], [], []
    for index, member in enumerate(recovery["members"], 1):
        path = Path(member["path"])
        if sha256(path) != member["sha256"]:
            raise RuntimeError("Recovered source member changed")
        parsed, globals_seen = parse_pickle_data(path.read_bytes())
        data = parsed[path.stem]["data"]
        uid = "HUST_" + path.stem
        for cycle in range(1, 101):
            if cycle not in data:
                raise RuntimeError(f"Source prefix missing original cycle {uid}/{cycle}")
            columns = dataframe_columns(data[cycle])
            stages.append({"cell_uid": uid, "original_cycle_id": cycle, "phases": phase_audit(columns)})
            for view in VIEWS:
                row, _ = measure_view(columns, cycle, view)
                rows.append({"cell_uid": uid, **row})
        sources.append({"cell_uid": uid, "source_member": member["member"], "source_sha256": member["sha256"],
                        "symbolic_globals_count": len(globals_seen), "unsafe_global_invocations": 0})
        del parsed, data
        gc.collect()
        print(json.dumps({"status": "PREFIX_AUDITING", "completed_cells": index, "total_cells": 77, "cell_uid": uid}), flush=True)
    cell_rows, summary = [], {}
    for view in VIEWS:
        selected = [row for row in rows if row["view"] == view.name]
        for source in sources:
            cell = [row for row in selected if row["cell_uid"] == source["cell_uid"]]
            valid = [row for row in cell if row["curve_valid"]]
            reference = [row for row in valid if row["original_cycle_id"] <= 10]
            anchor = [row for row in valid if row["original_cycle_id"] >= 96]
            cell_rows.append({"cell_uid": source["cell_uid"], "view": view.name,
                              "prefix_curves": len(valid), "reference_curves1_to10": len(reference),
                              "anchor_curve_support96_to100": len(anchor), "coverage_ge80": len(valid) >= 80,
                              "reference_support_ge5": len(reference) >= 5,
                              "failure_reasons": dict(Counter(reason for row in cell for reason in row["failure_reasons"]))})
        view_cells = [row for row in cell_rows if row["view"] == view.name]
        valid = [row for row in selected if row["curve_valid"]]
        summary[view.name] = {"definition": view_definitions()[VIEWS.index(view)], "cells": 77,
                              "prefix_rows": len(selected), "valid_curves": len(valid),
                              "cells_coverage_ge80": sum(row["coverage_ge80"] for row in view_cells),
                              "cells_reference_ge5": sum(row["reference_support_ge5"] for row in view_cells),
                              "cells_anchor_support": sum(row["anchor_curve_support96_to100"] > 0 for row in view_cells),
                              "failure_reasons": dict(Counter(reason for row in selected for reason in row["failure_reasons"])),
                              "window_native_points": distribution([row["window_native_points"] for row in valid]),
                              "window_distinct_voltages": distribution([row["window_distinct_voltage_points"] for row in valid]),
                              "maximum_native_spacing_V": distribution([row["window_native_spacing_max_V"] for row in valid]),
                              "window_capacity_span_Ah": distribution([row["window_capacity_span_Ah"] for row in valid]),
                              "current_max_deviation_A": distribution([row["current_max_deviation_from_registered_A"] for row in selected]),
                              "waveform_prefix_current_max_deviation_A": distribution([row["waveform_prefix_current_max_deviation_A"] for row in selected]),
                              "window_current_max_deviation_A": distribution([row["window_current_max_deviation_A"] for row in selected]),
                              "mixed_current_phases": sum(row["phase_mixed_current"] for row in selected),
                              "source_capacity_over_trapezoid_integral": distribution([row["source_capacity_over_trapezoid_integral"] for row in selected])}
    write_csv(args.output / "prefix_view_support.csv", rows)
    write_csv(args.output / "prefix_cell_support.csv", cell_rows)
    write_csv(args.output / "prefix_stage_audit.csv", stages)
    files = [args.output / name for name in ("prefix_view_support.csv", "prefix_cell_support.csv", "prefix_stage_audit.csv")]
    complete = {"status": "PREFIX_SUPPORT_COMPLETE", "source_cells": 77, "original_prefix_cycles": 7700,
                "future_capacity_values_used": 0, "future_prediction_results_used": 0,
                "source_archive_sha256": recovery["actual_sha256"], "source_recovery_sha256": sha256(recovery_path),
                "sources": sources, "views": summary,
                "files": [{"path": path.name, "sha256": sha256(path)} for path in files],
                "code": [{"path": str(path.relative_to(REPOSITORY)).replace("\\", "/"), "sha256": sha256(path)}
                         for path in (Path(__file__), Path(__file__).with_name("measurements.py"), REPOSITORY / "scripts/hust_safe_pickle.py")],
                "capacity_provenance": "Source-provided net Capacity(mAh). Generation algorithm and independent instrument calibration remain uncertified.",
                "time_provenance": "Public per-cycle seconds with phase-boundary inheritance and incomplete rest representation. Integration is an audit field and never changes Q.",
                "independence": "Two separately qualified phase views.64 interpolated coordinates are not64 independent measurements. Coverage_ge80 is descriptive only."}
    write_json(args.output / "PREFIX_SUPPORT_COMPLETE.json", complete)
    print(json.dumps(json_ready({"status": complete["status"], "views": summary}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
