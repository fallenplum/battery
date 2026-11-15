"""Independently replay every original schedule field and source identity link."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


PRIVATE_FIELDS = {"m_szOriginalInfo", "m_szCreator", "m_szComments", "Signature"}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def read_original_fields(body):
    try:
        body.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        encoding = "latin-1"
    records = []
    section = None
    for line in re.split(rb"\r\n|\n|\r", body):
        if not line:
            continue
        if line[:1] == b"[" and line[-1:] == b"]":
            section = line[1:-1].decode(encoding)
        else:
            key, value = line.split(b"=", 1)
            records.append((section, key.decode(encoding), value.decode(encoding)))
    return records


def basename(value):
    return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", type=Path, default=Path("../private/raw_sources/MATR/instrument_csv"))
    parser.add_argument("--output", type=Path, default=Path("results/unified_matr"))
    args = parser.parse_args()
    audit_path = args.output / "schedule_metadata_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metadata = json.loads((args.output / "instrument_metadata_audit.json").read_text(encoding="utf-8"))
    inventory = json.loads((args.csv_root / "INSTRUMENT_CSV_INVENTORY.json").read_text(encoding="utf-8"))
    bindings = json.loads((args.output / "instrument_bindings.json").read_text(encoding="utf-8"))
    source_by_file = {row["file_id"]: row for row in inventory["files"]}
    metadata_by_segment = {row["source_segment_id"]: row for row in metadata["source_rows"]}
    native_sources = {segment["source_segment_id"]: segment for cell in bindings["cells"] for segment in cell["source_segments"]}
    links, sha_checks, basename_checks, declared_checks = [], [], [], []
    all_receipts, expected_fields, limits, discharge_counts = {}, {}, [], {}
    for row in audit["source_links"]:
        segment = native_sources[row["source_segment_id"]]
        source = source_by_file[segment["file_id"]]
        links.append(row["schedule_file_id"] == source["schedule_file_id"]
                     and row["official_edp_test_id"] == segment["test_id"] == source["test_id"])
        basename_checks.append(basename(row["schedule_file_name"]) == basename(metadata_by_segment[row["source_segment_id"]]["native_metadata_schedule_file_name"]))
        path = args.csv_root / row["schedule_source_path"]
        receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text(encoding="utf-8"))
        body = path.read_bytes()
        source_sha = digest(body)
        sha_checks.append(source_sha == row["schedule_sha256"] == receipt["sha256"]
                          and len(body) == row["schedule_bytes"] == receipt["bytes"])
        all_receipts[row["schedule_file_id"]] = receipt
        original = read_original_fields(body)
        sections = {}
        for section, field, value in original:
            sections.setdefault(section, {})[field] = value
            public_value = digest(value.encode()) if field in PRIVATE_FIELDS and value else value
            expected_fields[(source_sha, section, field)] = (public_value, field in PRIVATE_FIELDS, value == "")
        declared = int(sections["Schedule"]["m_uStepNum"])
        actual = sorted(int(match.group(1)) for name in sections if (match := re.fullmatch(r"Schedule_Step([0-9]+)", name)))
        declared_checks.append(actual == list(range(declared)) and declared == row["declared_step_count"])
        for index in actual:
            section = "Schedule_Step" + str(index)
            count = len([name for name in sections if re.fullmatch(re.escape(section) + r"_Limit[0-9]+", name)])
            limits.append(count == int(sections[section]["m_uLimitNum"]))
            if sections[section].get("m_szLabel") == "Discharge":
                discharge_counts[index] = discharge_counts.get(index, 0) + 1
        published = audit["unique_schedules_by_sha256"][source_sha]["sections"]
        for section, fields in sections.items():
            for field, value in fields.items():
                public_field = field + "_sha256" if field in PRIVATE_FIELDS else field
                expected = digest(value.encode()) if field in PRIVATE_FIELDS and value else value
                if published[section].get(public_field) != expected:
                    raise ValueError("Published schedule field differs from its original source")
        if len(original) != sum(len(fields) for fields in published.values()):
            raise ValueError("Published schedule field coverage differs from original source")
    parquet = args.output / "schedule_static_fields.parquet"
    table = pd.read_parquet(parquet)
    actual_fields = {(row.schedule_sha256, row.source_section, row.source_field):
                     (row.source_value, row.source_value_redacted_to_sha256, row.source_value_missing)
                     for row in table.itertuples(index=False)}
    checks = {"174_source_segment_links_and_files_complete": len(links) == len(all_receipts) == 174,
              "all174_EDP_source_schedule_identifiers_join": all(links),
              "all174_original_source_whole_file_SHA_and_bytes_match": all(sha_checks),
              "all174_schedule_names_equal_original_Metadata_names": all(basename_checks),
              "all174_declared_steps_and_contiguous_indexes_match": all(declared_checks),
              "all_step_limit_section_counts_match_original_declarations": all(limits),
              "all_source_fields_replayed_exactly_into_unique_schedule_JSON": True,
              "all_unique_source_fields_replayed_exactly_into_parquet": expected_fields == actual_fields and len(actual_fields) == len(table),
              "unique_schedule_SHA_and_field_counts_match": len({item["sha256"] for item in all_receipts.values()}) == audit["unique_schedule_content_hashes"] and len(table) == audit["unique_schedule_field_rows"],
              "source_bytes_recomputed_from_complete_receipts": sum(item["bytes"] for item in all_receipts.values()) == audit["total_schedule_source_bytes"],
              "parquet_SHA_matches_JSON": digest(parquet.read_bytes()) == audit["schedule_static_fields_parquet_sha256"],
              "source_discharge_steps9_10_12_coverage_recomputed": discharge_counts == {9: 4, 10: 127, 12: 43},
              "zero_model_input_or_measurement_threshold_changes": audit["model_input_columns_added"] == audit["measurement_threshold_changes"] == 0}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "source_segments": len(links), "source_files": len(all_receipts),
              "source_bytes": sum(item["bytes"] for item in all_receipts.values()),
              "unique_schedule_contents": audit["unique_schedule_content_hashes"], "unique_field_rows": len(table),
              "original_main_discharge_step_counts": discharge_counts,
              "schedule_metadata_audit_sha256": digest(audit_path.read_bytes()), "code_sha256": digest(Path(__file__).read_bytes()),
              "native_step_index_replay_performed": False, "model_fits_or_input_mask_changes": 0}
    (args.output / "schedule_metadata_QA.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [key for key, passed in checks.items() if not passed]}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
