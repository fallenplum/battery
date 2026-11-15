"""Cross-check complete-column audit coverage, arithmetic and native contexts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    output = args.repository / "results/unified_matr"
    bindings = read(output / "instrument_bindings.json")
    summary = read(output / "native_channel_audit_summary.json")
    table = pd.read_parquet(output / "native_channel_audit.parquet")
    segments = {segment["file_id"]: segment for cell in bindings["cells"] for segment in cell["source_segments"]}
    checks = {"174_unique_required_sources_complete": len(segments) == 174 and all(segment["download_status"] == "COMPLETE" for segment in segments.values()),
              "summary_complete_numeric_and_transition_coverage": summary["status"] == "COMPLETE" and summary["complete_source_segments"] == summary["numeric_complete_source_segments"] == summary["transition_complete_source_segments"] == 174,
              "parquet_file_field_keys_unique": not table.duplicated(["file_id", "field"]).any(),
              "all174_parquet_source_identities_covered": set(table.file_id) == set(segments),
              "all2695_native_file_field_rows_covered": len(table) == summary["audited_field_rows"] == 2695,
              "all_perfield_support_partitions_valid": bool((table["count"] == table.finite_count + table.nonfinite_count).all() and (table.nonfinite_count == table.missing_count + table.positive_inf_count + table.negative_inf_count).all()),
              "numeric_zero_support_bounded_by_finite_values": bool((table.zero_count <= table.finite_count).all()),
              "all174_numeric_and_transition_SHA_receipts_match": True,
              "all_actual_native_header_columns_audited": True,
              "all_fields_in_each_file_have_equal_native_row_count": True,
              "all_native_contexts_obey_kind_and_row_boundary": True,
              "all_context_counts_and_finite_statistics_consistent": True}
    expected_transition_totals = {name: 0 for name in summary["transition_totals"]}
    total_rows, total_bytes = 0, 0
    for file_id, segment in segments.items():
        rows = table.loc[table.file_id == file_id]
        with Path(segment["csv_path"]).open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        checks["all_actual_native_header_columns_audited"] &= set(rows.field) == set(header) and len(rows) == len(header)
        checks["all_fields_in_each_file_have_equal_native_row_count"] &= rows["count"].nunique() == 1
        audit = read(args.cache / (segment["sha256"] + ".json"))
        transitions = audit["transitions"]
        checks["all174_numeric_and_transition_SHA_receipts_match"] &= audit["actual_sha256"] == transitions["actual_sha256"] == segment["sha256"] and audit["bytes_read"] == transitions["bytes_read"] == segment["bytes"] and audit["status"] == transitions["status"] == "PASS"
        checks["all_fields_in_each_file_have_equal_native_row_count"] &= int(rows["count"].iloc[0]) == transitions["rows"]
        total_rows += transitions["rows"]
        total_bytes += segment["bytes"]
        for kind, count in transitions["counts"].items():
            expected_transition_totals[kind] += count
            examples = transitions["first_examples"][kind]
            checks["all_context_counts_and_finite_statistics_consistent"] &= len(examples) == min(3, count)
            for pair in examples:
                before, after = pair["before"], pair["after"]
                predicate = {"cycle_index_change": after["Cycle_Index"] != before["Cycle_Index"],
                             "cycle_index_decrease": after["Cycle_Index"] < before["Cycle_Index"],
                             "cycle_index_forward_jump": after["Cycle_Index"] > before["Cycle_Index"] + 1,
                             "data_point_nonincrease": after["Data_Point"] <= before["Data_Point"],
                             "datetime_reverse": after["DateTime"] < before["DateTime"],
                             "test_time_reverse": after["Test_Time"] < before["Test_Time"]}[kind]
                checks["all_native_contexts_obey_kind_and_row_boundary"] &= predicate and 0 <= pair["row_before"] < pair["row_after"] < transitions["rows"] and pair["row_after"] == pair["row_before"] + 1
        for row in rows.to_dict("records"):
            if row["finite_count"] == 0:
                valid = all(pd.isna(row[name]) for name in ("min", "max", "mean"))
            else:
                tolerance = 1e-12 * max(1, abs(row["min"]), abs(row["max"]))
                valid = all(math.isfinite(row[name]) for name in ("min", "max", "mean")) and row["min"] - tolerance <= row["mean"] <= row["max"] + tolerance
            checks["all_context_counts_and_finite_statistics_consistent"] &= valid
    checks["all_transition_macro_counts_recomputed_from_per_source_caches"] = expected_transition_totals == summary["transition_totals"]
    checks["total_native_rows_and_bytes_recomputed"] = total_rows == summary["native_data_rows"] == 143396900 and total_bytes == bindings["required_csv_bytes"] == 21837210616
    checks["temperature_IR_and_auxiliary_fields_retained"] = all(field in set(table.field) for field in ("Temperature", "Internal_Resistance", "Step_Time", "Test_Time", "dV/dt", "Aux_Voltage"))
    checks["only_b1c18_embedded_cycle_reset_in_full_inventory"] = [row["source_segment_id"] for row in summary["per_file_transitions"] if row["counts"]["cycle_index_decrease"]] == ["b1c18"]
    checks = {name: bool(value) for name, value in checks.items()}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "source_segments": len(segments), "field_rows": len(table), "native_rows": total_rows,
              "source_bytes": total_bytes, "transition_totals": expected_transition_totals,
              "parquet_sha256": hashlib.sha256((output / "native_channel_audit.parquet").read_bytes()).hexdigest(),
              "summary_sha256": hashlib.sha256((output / "native_channel_audit_summary.json").read_bytes()).hexdigest(),
              "model_fits_or_input_mask_changes": 0,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "native_channel_audit_QA.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [name for name, value in checks.items() if not value]}, ensure_ascii=False))
    if not all(checks.values()):
        raise AssertionError("Native full-column audit verification failed")


if __name__ == "__main__":
    main()
