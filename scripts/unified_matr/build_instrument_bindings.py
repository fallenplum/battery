"""Bind released source cells to official instrument CSVs by barcode and channel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path


def barcode_hash(value):
    return hashlib.sha256(str(value).strip().upper().encode()).hexdigest()


@lru_cache(maxsize=256)
def native_edges(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
        columns = reader.fieldnames
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 16384))
        last_line = handle.read().decode("utf-8-sig").splitlines()[-1]
    fields = next(csv.reader([last_line]))
    if len(fields) != len(columns):
        raise ValueError("The instrument CSV final row differs from its original header")
    last = dict(zip(columns, fields))
    return {"first_native_cycle": int(float(first["Cycle_Index"])),
            "last_native_cycle": int(float(last["Cycle_Index"])),
            "first_native_datetime_unix": float(first["DateTime"]),
            "last_native_datetime_unix": float(last["DateTime"]),
            "first_native_data_point": int(float(first["Data_Point"])),
            "last_native_data_point": int(float(last["Data_Point"])),
            "native_cycle_scan_status": "FIRST_AND_LAST_NATIVE_CSV_ROWS_READ_FULL_MAPPING_PENDING"}


def refresh_bindings(source_registry, csv_root, output):
    registry = json.loads(Path(source_registry).read_text(encoding="utf-8"))
    inventory = json.loads((Path(csv_root) / "INSTRUMENT_CSV_INVENTORY.json").read_text(encoding="utf-8"))
    if not inventory.get("actual_sizes_verified_with_HTTP_range"):
        raise ValueError("Instrument source sizes require actual HTTP Content-Range verification")
    indexed = {}
    for row in inventory["files"]:
        key = (row["source_batch"], str(int(row["source_channel"])), barcode_hash(row["source_cell_barcode"]))
        indexed.setdefault(key, []).append(row)
    cells, missing, ambiguous, selected_ids = [], [], [], []
    complete_count, total_bytes = 0, 0
    for cell in registry["cells"]:
        segments = []
        for source in cell["source_segments"]:
            key = (source["source_batch"], str(int(source["source_channel_id"])), source["source_barcode_normalized_sha256"])
            candidates = indexed.get(key, [])
            segment = {"source_segment_id": source["source_segment_id"], "source_batch": source["source_batch"],
                       "source_cell_index": source["source_cell_index"], "source_channel": key[1],
                       "barcode_sha256": source["source_barcode_sha256"],
                       "barcode_normalized_sha256": key[2], "continuation_of": source["continuation_of"],
                       "identity_certified": len(candidates) == 1,
                       "identity_basis": "MATLAB MCOS decoded source barcode and channel both equal official EDP test metadata",
                       "instrument_cycle_mapping_certified": False, "raw_cycle_certified": False,
                       "first_native_cycle": None, "last_native_cycle": None,
                       "native_cycle_scan_status": "PENDING_FULL_INSTRUMENT_SCAN"}
            if len(candidates) != 1:
                reason = "NO_MATCHING_OFFICIAL_BARCODE_CHANNEL" if not candidates else "MULTIPLE_OFFICIAL_BARCODE_CHANNEL_MATCHES"
                segment["identity_failure_reason"] = reason
                (missing if not candidates else ambiguous).append(source["source_segment_id"])
            else:
                row = candidates[0]
                selected_ids.append(row["file_id"])
                relative = f"batch{row['source_batch']}/{row['file_id']}_{Path(row['file_name']).name}"
                csv_path = Path(csv_root) / relative
                receipt_path = csv_path.with_suffix(csv_path.suffix + ".receipt.json")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
                complete = receipt.get("status") == "COMPLETE" and csv_path.is_file() and csv_path.stat().st_size == row["bytes"]
                complete_count += int(complete)
                total_bytes += row["bytes"]
                segment.update(test_id=row["test_id"], file_id=row["file_id"],
                               csv_file_name=row["file_name"], csv_relative_path=relative,
                               csv_path=str(csv_path.resolve()), source_url=row["source_url"],
                               bytes=row["bytes"], publisher_metadata_bytes=row["publisher_metadata_bytes"],
                               source_etag=row["etag"], source_sha256=receipt.get("sha256") if complete else None,
                               sha256=receipt.get("sha256") if complete else None,
                               download_status="COMPLETE" if complete else "PENDING_OR_PARTIAL")
                if complete:
                    segment.update(native_edges(str(csv_path)))
            segments.append(segment)
        cells.append({"cell_uid": cell["cell_uid"], "policy_original": cell["source_segments"][0]["policy_original"],
                      "source_segments": segments,
                      "identity_certified": all(row["identity_certified"] for row in segments),
                      "raw_cycle_certified": False})
    status = "IDENTITY_BINDINGS_COMPLETE_DOWNLOAD_PENDING"
    if missing or ambiguous:
        status = "IDENTITY_BINDINGS_INCOMPLETE"
    elif complete_count == len(selected_ids):
        status = "IDENTITY_AND_CSV_RECOVERY_COMPLETE_MAPPING_QA_PENDING"
    result = {"status": status, "cells": cells, "fixed_cells": len(cells),
              "matched_source_segments": len(selected_ids), "complete_csv_files": complete_count,
              "required_csv_bytes": total_bytes, "required_file_ids": selected_ids,
              "missing_source_segments": missing, "ambiguous_source_segments": ambiguous,
              "official_instrument_test_inventory_count": inventory["file_count"],
              "official_instrument_test_inventory_bytes": inventory["total_bytes"],
              "scope": "Identity binding and completed whole-file source SHA only. Original instrument cycle and waveform mapping require independent full CSV scan.",
              "source_registry_sha256": hashlib.sha256(Path(source_registry).read_bytes()).hexdigest(),
              "csv_inventory_sha256": hashlib.sha256((Path(csv_root) / "INSTRUMENT_CSV_INVENTORY.json").read_bytes()).hexdigest()}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = refresh_bindings(args.source_registry, args.csv_root, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ("cells", "required_file_ids")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
