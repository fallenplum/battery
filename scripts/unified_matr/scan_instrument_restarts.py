"""Independent column-wise CSV restart and native-index scan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ("Cycle_Index", "Data_Point", "DateTime", "Step_Index", "Test_Time")


def scan_native_index_runs(path, chunksize=250000):
    groups, drops = [], []
    source_rows, run_index, offset = 0, 0, 0
    previous = None
    for chunk in pd.read_csv(path, usecols=list(COLUMNS), dtype=np.float64,
                             chunksize=chunksize, float_precision="round_trip"):
        if not len(chunk):
            continue
        cycles = chunk.Cycle_Index.to_numpy()
        if not np.isfinite(cycles).all() or np.any(cycles != np.floor(cycles)):
            raise ValueError("Missing or noninteger native Cycle_Index in independent scan")
        positions = list(np.flatnonzero(np.diff(cycles) < 0) + 1)
        if previous is not None and cycles[0] < previous["Cycle_Index"]:
            positions.insert(0, 0)
        run_ids = np.empty(len(chunk), dtype=np.int64)
        run_offsets = np.empty(len(chunk), dtype=np.int64)
        start = 0
        for position in positions:
            before = previous if position == 0 else chunk.iloc[position - 1]
            after = chunk.iloc[position]
            valid = after["Cycle_Index"] == 0 and before["Cycle_Index"] > 0
            valid = valid and after["Step_Index"] in (0, 1)
            valid = valid and after["Data_Point"] > before["Data_Point"] and after["DateTime"] > before["DateTime"]
            evidence = {"source_row_before": source_rows + int(position) - 1,
                        "source_row_after": source_rows + int(position),
                        "previous_native_cycle": int(before["Cycle_Index"]), "next_native_cycle": int(after["Cycle_Index"]),
                        "previous_Step_Index": float(before["Step_Index"]), "next_Step_Index": float(after["Step_Index"]),
                        "previous_Data_Point": float(before["Data_Point"]), "next_Data_Point": float(after["Data_Point"]),
                        "previous_DateTime": float(before["DateTime"]), "next_DateTime": float(after["DateTime"]),
                        "previous_Test_Time": float(before["Test_Time"]), "next_Test_Time": float(after["Test_Time"]),
                        "authenticated_restart": bool(valid), "new_run_index": run_index + 1}
            drops.append(evidence)
            if not valid:
                return {"status": "UNAUTHENTICATED_COUNTER_DROP", "groups": pd.DataFrame(groups), "drops": drops,
                        "source_rows_scanned": source_rows + int(position), "run_count": run_index + 1}
            run_ids[start:position], run_offsets[start:position] = run_index, offset
            offset += int(before["Cycle_Index"])
            run_index += 1
            start = int(position)
        run_ids[start:], run_offsets[start:] = run_index, offset
        boundaries = np.r_[0, np.flatnonzero((np.diff(cycles) != 0) | (np.diff(run_ids) != 0)) + 1, len(chunk)]
        points = chunk.Data_Point.to_numpy()
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            group = {"run_index": int(run_ids[left]), "instrument_cycle_id": int(cycles[left]),
                     "instrument_run_cycle_offset": int(run_offsets[left]),
                     "source_row_start": source_rows + int(left), "source_row_stop_exclusive": source_rows + int(right),
                     "source_data_point_first": float(points[left]), "source_data_point_last": float(points[right - 1])}
            if groups and groups[-1]["run_index"] == group["run_index"] and groups[-1]["instrument_cycle_id"] == group["instrument_cycle_id"]:
                groups[-1]["source_row_stop_exclusive"] = group["source_row_stop_exclusive"]
                groups[-1]["source_data_point_last"] = group["source_data_point_last"]
            else:
                groups.append(group)
        previous = chunk.iloc[-1].copy()
        source_rows += len(chunk)
    return {"status": "PASS_NATIVE_RESTART_SCAN", "groups": pd.DataFrame(groups), "drops": drops,
            "source_rows_scanned": source_rows, "run_count": run_index + 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bindings_bytes = args.bindings.read_bytes()
    bindings = json.loads(bindings_bytes)
    scanner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cached = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    old = {entry["source_segment_id"]: entry for entry in cached.get("files", [])} if cached.get("scanner_sha256") == scanner_sha else {}
    files, pending = [], []
    for cell in bindings["cells"]:
        for segment in cell["source_segments"]:
            if segment["download_status"] != "COMPLETE":
                pending.append(segment["source_segment_id"])
                continue
            saved = old.get(segment["source_segment_id"])
            if saved is not None and saved["source_file_sha256"] == segment["sha256"]:
                files.append(saved)
                continue
            scanned = scan_native_index_runs(Path(segment["csv_path"]))
            groups = scanned.pop("groups")
            runs = [{"run_index": int(number), "native_first": int(frame.instrument_cycle_id.iloc[0]),
                     "native_last": int(frame.instrument_cycle_id.iloc[-1]), "groups": len(frame),
                     "instrument_run_cycle_offset": int(frame.instrument_run_cycle_offset.iloc[0])}
                    for number, frame in groups.groupby("run_index", sort=False)]
            files.append({"cell_uid": cell["cell_uid"], "source_segment_id": segment["source_segment_id"],
                          "source_file_sha256": segment["sha256"], "source_bytes": segment["bytes"],
                          **scanned, "groups": len(groups), "runs": runs})
            print(json.dumps({"source_segment": segment["source_segment_id"], "runs": scanned["run_count"], "drops": len(scanned["drops"]), "status": scanned["status"]}), flush=True)
    all_valid = all(entry["status"] == "PASS_NATIVE_RESTART_SCAN" for entry in files)
    result = {"status": "ALL_COMPLETED_FILES_RESTART_SCAN_PASS" if all_valid else "UNAUTHENTICATED_RESTART_FOUND",
              "all_required_sources_complete": not pending, "completed_files_scanned": len(files),
              "pending_source_segments": pending, "files_with_restart": sum(entry["run_count"] > 1 for entry in files),
              "total_restarts": sum(len(entry["drops"]) for entry in files), "files": files,
              "scanner_sha256": scanner_sha, "bindings_sha256": hashlib.sha256(bindings_bytes).hexdigest(),
              "source_sha_scope": "Complete recovery receipts. DATA-FREEZE verifier independently rehashes each complete source.",
              "DATA_FREEZE_allowed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "scanned": len(files), "restarts": result["total_restarts"], "pending": len(pending)}))


if __name__ == "__main__":
    main()
