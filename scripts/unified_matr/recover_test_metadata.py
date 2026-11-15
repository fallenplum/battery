"""Recover and audit every fixed-cohort original instrument metadata CSV."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://data.matr.io/1/api/v1/file/"
REDACTED_FIELDS = {"item_id", "databases", "canconfig_filename", "m_ncanconfigmd5"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def retrieve(source, root):
    file_id = source["test_metadata_file_id"]
    target = root / (file_id + ".metadata.csv")
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    if target.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        body = target.read_bytes()
        if receipt["sha256"] != sha256(body) or receipt["bytes"] != len(body):
            raise ValueError("Existing metadata differs from its complete source receipt")
        return receipt
    with requests.Session() as session:
        session.trust_env = False
        adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)))
        session.mount("https://", adapter)
        response = session.get(BASE + file_id, timeout=(20, 60))
        response.raise_for_status()
        official = response.json()
        write_json(root / (file_id + ".file.json"), official)
        response = session.get(BASE + file_id + "/download", timeout=(20, 60))
        response.raise_for_status()
        body = response.content
        receipt = {"status": "COMPLETE", "file_id": file_id, "file_name": official["name"],
                   "source_url": BASE + file_id + "/download", "resolved_url": response.url,
                   "bytes": len(body), "publisher_metadata_bytes": official["size"],
                   "publisher_metadata_size_disagrees": len(body) != official["size"],
                   "etag": response.headers.get("ETag"), "sha256": sha256(body),
                   "content_type": response.headers.get("Content-Type"), "path": str(target.resolve())}
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(body)
    os.replace(temporary, target)
    write_json(receipt_path, receipt)
    return receipt


def read_metadata(receipt):
    body = Path(receipt["path"]).read_bytes()
    if sha256(body) != receipt["sha256"]:
        raise ValueError("Metadata SHA changed between recovery and audit")
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    if len(rows) != 1 or len(reader.fieldnames) != len(set(reader.fieldnames)) or None in rows[0]:
        raise ValueError("Expected exactly one complete instrument metadata row with unique source columns")
    return {"source_dataframe_index" if key == "" else key: value for key, value in rows[0].items()}


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def public_value(field, value):
    if field in REDACTED_FIELDS:
        return sha256(value.encode()) if value else ""
    return value


def audit(bindings, inventory, root, output, workers):
    lookup = {row["file_id"]: row for row in inventory["files"]}
    selected = [(cell["cell_uid"], segment, lookup[segment["file_id"]])
                for cell in bindings["cells"] for segment in cell["source_segments"]]
    if len(selected) != 174 or len({row[2]["test_metadata_file_id"] for row in selected}) != 174:
        raise ValueError("Fixed 169-cell cohort must have 174 unique instrument metadata files")
    root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        receipts = list(executor.map(lambda row: retrieve(row[2], root), selected))
    public, private, checks = [], [], []
    fields = set()
    for (uid, segment, source), receipt in zip(selected, receipts):
        values = read_metadata(receipt)
        fields.update(values)
        item = values.get("item_id", "")
        identity = sha256(item.strip().upper().encode()) == segment["barcode_normalized_sha256"]
        channel = number(values.get("iv_ch_id"))
        channel_ok = channel is not None and channel + 1 == int(segment["source_channel"])
        device_ok = number(values.get("device_id")) is not None
        test_ok = number(values.get("test_id")) is not None
        row = {"cell_uid": uid, "source_segment_id": segment["source_segment_id"],
               "source_batch": segment["source_batch"], "source_channel_one_based": segment["source_channel"],
               "official_edp_test_id": segment["test_id"], "metadata_file_id": receipt["file_id"],
               "metadata_sha256": receipt["sha256"], "metadata_bytes": receipt["bytes"],
               "metadata_source_url": receipt["source_url"],
               "metadata_source_path": str(Path(receipt["path"]).relative_to(root.parent.resolve())),
               "barcode_identity_match": identity, "zero_based_channel_match": channel_ok,
               "instrument_device_id_present": device_ok, "instrument_test_id_present": test_ok,
               "qa_native_first_minus_metadata_start_seconds": segment["first_native_datetime_unix"] - float(values["first_start_datetime"]),
               "qa_metadata_end_minus_native_last_seconds": float(values["last_end_datetime"]) - segment["last_native_datetime_unix"],
               "model_input_columns_added": 0}
        for field, value in values.items():
            suffix = "_sha256" if field in REDACTED_FIELDS else ""
            row["native_metadata_" + field + suffix] = public_value(field, value)
        public.append(row)
        private.append({"source_segment_id": segment["source_segment_id"], "receipt": receipt, "raw_fields": values})
        checks.append(identity and channel_ok and device_ok and test_ok)
    field_audit = {}
    for field in sorted(fields):
        observed = [item["raw_fields"].get(field) for item in private]
        populated = [value for value in observed if value not in (None, "")]
        numeric = [number(value) for value in populated]
        all_numeric = bool(numeric) and all(value is not None for value in numeric)
        field_audit[field] = {"source_files": len(observed), "column_present": sum(value is not None for value in observed),
                              "empty_count": sum(value == "" for value in observed), "populated_count": len(populated),
                              "distinct_populated_count": len(set(populated)), "all_populated_values_finite_numeric": all_numeric,
                              "numeric_min": min(numeric) if all_numeric else None,
                              "numeric_max": max(numeric) if all_numeric else None,
                              "public_representation": "sha256_preserving_empty" if field in REDACTED_FIELDS else "original_text_preserving_empty"}
    columns = list(dict.fromkeys(key for row in public for key in row))
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "instrument_metadata_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(public)
    write_json(root / "METADATA_FULL_FIELD_AUDIT_PRIVATE.json", private)
    result = {"status": "PASS" if all(checks) else "FAIL", "fixed_physical_cells": bindings["fixed_cells"],
              "source_segments": len(public), "complete_metadata_files": len(receipts),
              "total_metadata_bytes": sum(item["bytes"] for item in receipts),
              "all174_barcode_and_zero_based_channel_bindings_match": all(checks),
              "source_field_count": len(fields), "field_audit": field_audit, "source_rows": public,
              "instrument_device_counts_preserving_source_strings": dict(Counter(row["native_metadata_device_id"] for row in public)),
              "unique_device_test_channel_tuples": len({(row["native_metadata_device_id"], row["native_metadata_test_id"], row["native_metadata_iv_ch_id"]) for row in public}),
              "native_first_minus_metadata_start_seconds_range": [min(row["qa_native_first_minus_metadata_start_seconds"] for row in public), max(row["qa_native_first_minus_metadata_start_seconds"] for row in public)],
              "metadata_end_before_native_last_by_over60_seconds": sum(row["qa_metadata_end_minus_native_last_seconds"] < -60 for row in public),
              "csv_sha256": sha256(csv_path.read_bytes()), "code_sha256": sha256(Path(__file__).read_bytes()),
              "timestamp_semantics": "Original numeric timestamps retained for source QA. Cross-field timezone/export semantics unverified. Numeric metadata-to-CSV deltas are QA observations and do not authorize log truncation. End/resume and database fields have no model-input authorization.",
              "instrument_identity_semantics": "Original Arbin device_id and numeric test_id plus zero-based iv_ch_id. EDP test_id is recorded separately.",
              "missing_semantics": "Original empty strings remain empty. Numeric 0 retained and not converted to missing.",
              "redaction": "Manufacturer item_id, database strings and CAN config strings remain private; public rows retain SHA and missingness.",
              "model_input_columns_added": 0}
    write_json(output / "instrument_metadata_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, default=Path("results/unified_matr/instrument_bindings.json"))
    parser.add_argument("--csv-root", type=Path, default=Path("../private/raw_sources/MATR/instrument_csv"))
    parser.add_argument("--output", type=Path, default=Path("results/unified_matr"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    inventory = json.loads((args.csv_root / "INSTRUMENT_CSV_INVENTORY.json").read_text(encoding="utf-8"))
    result = audit(bindings, inventory, args.csv_root / "instrument_metadata", args.output, args.workers)
    print(json.dumps({key: result[key] for key in ("status", "source_segments", "complete_metadata_files", "total_metadata_bytes", "source_field_count")}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
