"""Independently verify recovered instrument identity and all metadata fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


REDACTED_FIELDS = {"item_id", "databases", "canconfig_filename", "m_ncanconfigmd5"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", type=Path, default=Path("../private/raw_sources/MATR/instrument_csv"))
    parser.add_argument("--output", type=Path, default=Path("results/unified_matr"))
    args = parser.parse_args()
    audit_path = args.output / "instrument_metadata_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    bindings = json.loads((args.output / "instrument_bindings.json").read_text(encoding="utf-8"))
    inventory = json.loads((args.csv_root / "INSTRUMENT_CSV_INVENTORY.json").read_text(encoding="utf-8"))
    sources = {row["file_id"]: row for row in inventory["files"]}
    published = {row["source_segment_id"]: row for row in audit["source_rows"]}
    original_fields, bodies, identity, metadata, source_fields, time_deltas, instrument_ids = {}, [], [], [], [], [], []
    original_barcodes = []
    for cell in bindings["cells"]:
        for segment in cell["source_segments"]:
            source = sources[segment["file_id"]]
            file_id = source["test_metadata_file_id"]
            path = args.csv_root / "instrument_metadata" / (file_id + ".metadata.csv")
            receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text(encoding="utf-8"))
            body = path.read_bytes()
            reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig"), newline=""))
            values = list(reader)
            if len(values) != 1 or None in values[0]:
                raise ValueError("Metadata source does not contain one complete original row")
            fields = {"source_dataframe_index" if key == "" else key: value for key, value in values[0].items()}
            row = published[segment["source_segment_id"]]
            identity.append(digest(fields["item_id"].strip().upper().encode()) == segment["barcode_normalized_sha256"]
                            and int(fields["iv_ch_id"]) + 1 == int(segment["source_channel"]))
            instrument_ids.append((fields["device_id"], fields["test_id"], fields["iv_ch_id"]))
            time_deltas.append(row["qa_native_first_minus_metadata_start_seconds"] == segment["first_native_datetime_unix"] - float(fields["first_start_datetime"])
                               and row["qa_metadata_end_minus_native_last_seconds"] == float(fields["last_end_datetime"]) - segment["last_native_datetime_unix"])
            metadata.append(row["metadata_sha256"] == receipt["sha256"] == digest(body)
                            and row["metadata_bytes"] == receipt["bytes"] == len(body)
                            and row["metadata_file_id"] == file_id
                            and row["official_edp_test_id"] == source["test_id"] == segment["test_id"])
            for field, value in fields.items():
                original_fields.setdefault(field, []).append(value)
                public_field = "native_metadata_" + field + ("_sha256" if field in REDACTED_FIELDS else "")
                expected = digest(value.encode()) if field in REDACTED_FIELDS and value else value
                source_fields.append(row.get(public_field) == expected)
            original_barcodes.append(fields["item_id"])
            bodies.append(len(body))
    field_coverage = []
    for field, values in original_fields.items():
        stats = audit["field_audit"][field]
        populated = [value for value in values if value]
        field_coverage.append(stats["column_present"] == len(values)
                              and stats["populated_count"] == len(populated)
                              and stats["empty_count"] == sum(value == "" for value in values)
                              and stats["distinct_populated_count"] == len(set(populated)))
    csv_path = args.output / "instrument_metadata_audit.csv"
    csv_body = csv_path.read_bytes()
    csv_rows = list(csv.DictReader(io.StringIO(csv_body.decode("utf-8"), newline="")))
    csv_values_match = len(csv_rows) == len(published) and all(
        all(str(value) == row.get(key, "") for key, value in published[row["source_segment_id"]].items())
        for row in csv_rows)
    public_text = audit_path.read_text(encoding="utf-8") + csv_body.decode("utf-8")
    checks = {"fixed169_physical_cells174_source_segments": bindings["fixed_cells"] == 169 and len(published) == 174,
              "all174_original_metadata_files_complete_SHA_and_EDP_join": len(metadata) == 174 and all(metadata),
              "all174_original_barcode_and_zero_based_channel_match": len(identity) == 174 and all(identity),
              "all174_original_instrument_device_test_channel_tuples_unique": len(set(instrument_ids)) == audit["unique_device_test_channel_tuples"] == 174,
              "all_numeric_metadata_to_native_clock_deltas_recomputed": all(time_deltas),
              "all_original_fields_preserved_with_declared_redaction": all(source_fields),
              "all_field_support_and_distinct_counts_recomputed": all(field_coverage),
              "all_source_field_names_covered": set(original_fields) == set(audit["field_audit"]),
              "csv_contents_equal_JSON_rows": csv_values_match,
              "csv_complete_SHA_matches_JSON": digest(csv_body) == audit["csv_sha256"],
              "no_manufacturer_barcodes_in_public_outputs": all(barcode not in public_text for barcode in original_barcodes),
              "zero_new_model_input_columns": audit["model_input_columns_added"] == 0,
              "metadata_byte_and_file_coverage_recomputed": sum(bodies) == audit["total_metadata_bytes"] and len(bodies) == audit["complete_metadata_files"]}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "source_files": len(metadata), "metadata_bytes": sum(bodies), "source_fields": len(original_fields),
              "metadata_audit_sha256": digest(audit_path.read_bytes()), "code_sha256": digest(Path(__file__).read_bytes()),
              "model_fits_or_input_mask_changes": 0}
    (args.output / "instrument_metadata_QA.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [key for key, passed in checks.items() if not passed]}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
