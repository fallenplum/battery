"""Independently bind native CSV groups to released MATLAB waveforms."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from instrument_csv import compare_instrument_release, iter_instrument_cycles
from raw_reader import cell_metadata, iter_raw_cycles


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-cell-index", type=int, required=True)
    parser.add_argument("--source-batch", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.csv.with_suffix(args.csv.suffix + ".receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETE" or args.csv.stat().st_size != receipt["bytes"] or sha256(args.csv) != receipt["sha256"]:
        raise ValueError("Native CSV receipt, actual bytes and whole-file SHA must agree")
    meta, summary = cell_metadata(args.release, args.release_cell_index, args.source_batch)
    identity = meta["source_barcode_sha256"] == receipt["source_cell_barcode_sha256"] and str(meta["source_channel_id"]) == str(receipt["source_channel"])
    if not identity:
        raise ValueError("Decoded release identity differs from original EDP CSV barcode/channel")
    release_iterator = iter_raw_cycles(args.release, args.release_cell_index, meta, summary)
    queue = collections.deque()
    release_exhausted = False
    matches, raw_rows, unmatched_release = [], [], []
    field_maxima = collections.defaultdict(float)
    for native in iter_instrument_cycles(args.csv, {**receipt, "cell_uid": meta["cell_uid"], "source_segment_id": meta["source_segment_id"]}):
        while len(queue) < 3 and not release_exhausted:
            try:
                queue.append(next(release_iterator))
            except StopIteration:
                release_exhausted = True
        candidates = [(index, compare_instrument_release(native, record)) for index, record in enumerate(queue)]
        valid = [(index, result) for index, result in candidates if result["matches"]]
        row = {"cell_uid": meta["cell_uid"], "source_segment_id": meta["source_segment_id"],
               "instrument_cycle_id": native["instrument_cycle_id"], "source_row_start": native["source_row_start"],
               "source_row_stop_exclusive": native["source_row_stop_exclusive"],
               "point_count": len(native["V"]), "time_reversal_count": native["time_reversal_count"],
               "source_terminal_cycle": native["source_terminal_cycle"],
               "initial_discharge_candidate": native["initial_discharge_candidate"],
               "matching_release_records_in_lookahead": len(valid), "release_source_record_index": None,
               "release_summary_sequence": None, "binding": "NO_NUMERIC_RELEASE_MATCH"}
        if len(valid) == 1:
            position, result = valid[0]
            for _ in range(position):
                skipped = queue.popleft()
                unmatched_release.append({"source_record_index": skipped["source_record_index"], "point_count": len(skipped["V"]), "reason": "No chronological native CSV match before later matched record"})
            released = queue.popleft()
            row["release_source_record_index"] = released["source_record_index"]
            row["release_summary_sequence"] = released["summary_cycle_sequence"]
            row["binding"] = "V_I_Qc_Qd_t_MATCH"
            for field, detail in result["fields"].items():
                field_maxima[field] = max(field_maxima[field], detail["max_abs_difference"] or 0.0)
                row[field + "_max_abs_difference"] = detail["max_abs_difference"]
            matches.append(result)
        elif len(valid) > 1:
            row["binding"] = "AMBIGUOUS_NUMERIC_RELEASE_MATCH"
        raw_rows.append(row)
    for record in queue:
        unmatched_release.append({"source_record_index": record["source_record_index"], "reason": "No numeric native CSV match by end of source log"})
    for record in release_iterator:
        unmatched_release.append({"source_record_index": record["source_record_index"], "reason": "Remaining release record after native source log ended"})
    frame = pd.DataFrame(raw_rows)
    matched = frame.loc[frame.binding == "V_I_Qc_Qd_t_MATCH"]
    coverage = len(matches) == meta["raw_records"] and not unmatched_release and not matched.release_source_record_index.duplicated().any()
    result = {
        "status": "PASS" if coverage else "PARTIAL_NUMERIC_BINDING",
        "source_cell_id": meta["source_cell_id"], "cell_uid": meta["cell_uid"],
        "identity_barcode_channel_exact_match": identity,
        "barcode_sha256": meta["source_barcode_sha256"], "source_channel": meta["source_channel_id"],
        "native_source_sha256": receipt["sha256"], "native_source_bytes": receipt["bytes"],
        "native_instrument_groups": len(frame), "native_instrument_cycle_min": int(frame.instrument_cycle_id.min()),
        "native_instrument_cycle_max": int(frame.instrument_cycle_id.max()),
        "native_source_rows": int(frame.point_count.sum()),
        "release_records": meta["raw_records"], "numeric_bindings": len(matches),
        "field_max_absolute_differences": dict(field_maxima),
        "native_groups_absent_from_release": frame.loc[frame.binding != "V_I_Qc_Qd_t_MATCH", ["instrument_cycle_id", "point_count", "source_terminal_cycle", "initial_discharge_candidate", "binding"]].to_dict("records"),
        "unmatched_release_records": unmatched_release,
        "native_time_reversal_cycles": frame.loc[frame.time_reversal_count > 0, ["instrument_cycle_id", "time_reversal_count", "release_source_record_index"]].to_dict("records"),
        "source_cycle_ids_are_instrument_observations": True,
        "global_continuation_axis_certified": False,
        "scope": "Independent single-test identity and native-to-release waveform binding. Native groups omitted by the old export remain in the source table. Continuation/global-cycle joins are a separate certification.",
        "parser_sha256": sha256(Path(__file__).with_name("instrument_csv.py")),
        "reader_sha256": sha256(Path(__file__).with_name("raw_reader.py")),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = meta["source_cell_id"]
    frame.to_csv(args.output_root / (stem + "_instrument_release_bindings.csv"), index=False)
    (args.output_root / (stem + "_INSTRUMENT_BINDING_QA.json")).write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "native_groups": len(frame), "numeric_bindings": len(matches), "release_records": meta["raw_records"]}), flush=True)


if __name__ == "__main__":
    main()
