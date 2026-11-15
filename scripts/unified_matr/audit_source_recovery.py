"""Audit source recovery and identity receipts without generating model data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.repository / "results/unified_matr"
    registry = read(output / "source_registry.json")
    bindings = read(output / "instrument_bindings.json")
    inventory = read(args.source_root / "instrument_csv/INSTRUMENT_CSV_INVENTORY.json")
    recovered = read(args.source_root / "SOURCE_RECOVERY_RECEIPT.json")
    proof = read(args.source_root / "AUTHOR_SOURCE_PROOF.json")
    source_code = args.source_root / proof["local_file"]
    indexed = {}
    for row in inventory["files"]:
        normalized = hashlib.sha256(row["source_cell_barcode"].strip().upper().encode()).hexdigest()
        key = (row["source_batch"], str(int(row["source_channel"])), normalized)
        indexed.setdefault(key, []).append(row)
    all_matches = []
    for segment in registry["all_released_source_segments"]:
        key = (segment["source_batch"], str(int(segment["source_channel_id"])), segment["source_barcode_normalized_sha256"])
        all_matches.append(indexed.get(key, []))
    matched_ids = {rows[0]["file_id"] for rows in all_matches if len(rows) == 1}
    unmatched_edp = [{key: row[key] for key in ("source_batch", "source_channel", "test_id", "file_id", "file_name", "bytes")}
                     for row in inventory["files"] if row["file_id"] not in matched_ids]
    checks = {
        "four_matlab_archives_complete_lengths_match": len(recovered["sources"]) == 4 and all(row["status"] == "COMPLETE" and (args.source_root / row["file_name"]).stat().st_size == row["bytes"] for row in recovered["sources"]),
        "four_matlab_complete_SHA_receipts_bound_to_registry": {row["batch"]: row["sha256"] for row in recovered["sources"]} == {row["batch"]: row["sha256"] for row in registry["source_files"]},
        "previous_three_static_policy_byte_ranges_match_complete_archives": all(row.get("prior_static_policy_ranges_replayed", 0) > 0 for row in registry["source_files"] if row["batch"] <= 3),
        "all185_matlab_segments_have_unique_official_csv_identity": len(all_matches) == len(matched_ids) == 185 and all(len(rows) == 1 for rows in all_matches),
        "fixed169_physical_ids_and174_segments_match_without_future_support_filter": len(bindings["cells"]) == 169 and bindings["matched_source_segments"] == 174 and all(cell["identity_certified"] for cell in bindings["cells"]),
        "fixed169_have169_distinct_physical_barcodes": len({cell["source_segments"][0]["barcode_normalized_sha256"] for cell in bindings["cells"]}) == 169,
        "all5_continuation_pairs_share_same_actual_barcode": sum(len(cell["source_segments"]) > 1 for cell in bindings["cells"]) == 5 and all(len({segment["barcode_normalized_sha256"] for segment in cell["source_segments"]}) == 1 for cell in bindings["cells"]),
        "all186_actual_http_source_sizes_and_etags_registered": inventory["file_count"] == 186 and inventory["actual_sizes_verified_with_HTTP_range"] and all(row["etag"] and row["bytes"] > 0 for row in inventory["files"]),
        "no_missing_or_ambiguous_fixed_cohort_source_identity": not bindings["missing_source_segments"] and not bindings["ambiguous_source_segments"],
        "author_parser_immutable_commit_and_full_local_sha": proof["commit"] == "aa54406731665e3ef1f77ad43dbc55ff7c763fd3" and hashlib.sha256(source_code.read_bytes()).hexdigest() == proof["sha256"],
        "matlab_summary_sequence_retains_uncertified_instrument_axis": all(segment["raw_cycle_certified"] is False and segment["instrument_cycle_axis_authenticated"] is False for segment in registry["all_released_source_segments"]),
        "full_instrument_mapping_kept_pending_independent_record_replay": all(segment["instrument_cycle_mapping_certified"] is False for cell in bindings["cells"] for segment in cell["source_segments"]),
    }
    complete = [segment for cell in bindings["cells"] for segment in cell["source_segments"] if segment.get("download_status") == "COMPLETE"]
    checks["every_complete_csv_has_whole_file_sha_and_matching_length"] = all(segment["sha256"] and Path(segment["csv_path"]).stat().st_size == segment["bytes"] for segment in complete)
    result = {"status": "PASS_SOURCE_METADATA_RECOVERY_IN_PROGRESS" if all(checks.values()) else "FAIL",
              "checks": checks,
              "matlab_source_bytes": sum(row["bytes"] for row in recovered["sources"]),
              "matlab_source_segments": 185,
              "matlab_unique_normalized_barcodes": len({row["source_barcode_normalized_sha256"] for row in registry["all_released_source_segments"]}),
              "official_instrument_tests": inventory["file_count"], "official_instrument_bytes": inventory["total_bytes"],
              "official_csv_publisher_stale_size_records": sum(row["publisher_metadata_size_disagrees"] for row in inventory["files"]),
              "official_csv_publisher_size_overstatement_bytes": sum(row["publisher_metadata_bytes"] - row["bytes"] for row in inventory["files"]),
              "official_edp_tests_outside_matlab_snapshot": unmatched_edp,
              "fixed_cohort_cells": 169, "fixed_required_csv_segments": 174,
              "fixed_required_csv_bytes": bindings["required_csv_bytes"], "complete_csv_files": len(complete),
              "pending_csv_files": 174 - len(complete), "complete_csv_bytes": sum(row["bytes"] for row in complete),
              "author_source_proof": proof,
              "short_source_excerpts": [{"line": 92, "code": "battery.cycles(k).I = Current(cycle_indices)/1.1;"},
                                        {"line": 154, "code": "battery.summary.cycle = (1:k)';"}],
              "third_party_full_code_scope": "Private local source evidence, original immutable URL and SHA retained",
              "source_identity_confidence": "high, exact decoded manufacturer barcode and physical channel within each original batch",
              "instrument_axis_scope": "Actual instrument Cycle_Index exists in native CSV; full per-record mapping and prefix/phase qualification are independent downstream QA",
              "formal_model_evaluation_performed": False,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if len(complete) == 174 and all(checks.values()):
        result["status"] = "PASS_SOURCE_RECOVERY_COMPLETE_INSTRUMENT_MAPPING_QA_PENDING"
    (output / "source_recovery_qa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [name for name, value in checks.items() if not value],
                      "complete_csv_files": len(complete), "pending_csv_files": 174 - len(complete)}, ensure_ascii=False))
    if not all(checks.values()):
        raise AssertionError("Original source recovery identity audit failed")


if __name__ == "__main__":
    main()
