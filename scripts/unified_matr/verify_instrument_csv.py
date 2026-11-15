"""Streaming/parser controls and real four-batch CSV-prefix schema checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from instrument_csv import ARRAY_COLUMNS, compare_instrument_release, iter_instrument_cycles


def run_synthetic_checks():
    cycle_ids = [0] * 3 + [1] * 7 + [4] * 4
    rows = []
    for index, cycle in enumerate(cycle_ids):
        rows.append({"Data_Point": index, "Test_Time": -100 + index * 5,
                     "DateTime": 1000000 + index * 5, "Step_Time": index * 5,
                     "Step_Index": 2, "Cycle_Index": cycle, "Current": -4.4,
                     "Voltage": 3.6 - 0.1 * index, "Charge_Capacity": 0,
                     "Discharge_Capacity": 0.01 * index, "Temperature": 30,
                     "Internal_Resistance": np.nan})
    metadata = {"cell_uid": "synthetic", "source_batch": 1, "test_id": "test", "status": "COMPLETE"}
    checks = {}
    with tempfile.TemporaryDirectory(prefix="matr_csv_parser_") as temporary:
        path = Path(temporary) / "native.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        baseline = list(iter_instrument_cycles(path, metadata, chunksize=50000))
        assert [record["instrument_cycle_id"] for record in baseline] == [0, 1, 4]
        assert [len(record["V"]) for record in baseline] == [3, 7, 4]
        checks["native_index_gaps_and_first_last_preserved"] = True
        for chunk_size in (1, 2, 3, 5, 8):
            streamed = list(iter_instrument_cycles(path, metadata, chunksize=chunk_size))
            assert len(streamed) == len(baseline)
            for expected, actual in zip(baseline, streamed):
                assert expected["instrument_cycle_id"] == actual["instrument_cycle_id"]
                assert expected["source_row_start"] == actual["source_row_start"]
                assert expected["source_row_stop_exclusive"] == actual["source_row_stop_exclusive"]
                for key in ARRAY_COLUMNS:
                    assert np.array_equal(expected[key], actual[key], equal_nan=True)
        checks["chunk_boundary_exact_invariance_5_sizes"] = True
        assert baseline[0]["t"][0] == -100
        assert baseline[0]["I"][0] == -4.4
        assert np.isnan(baseline[0]["internal_resistance"]).all()
        assert baseline[0]["initial_discharge_candidate"]
        assert baseline[0]["source_cycle_closed_by_next"]
        assert baseline[-1]["source_terminal_cycle"]
        assert not baseline[-1]["source_cycle_closed_by_next"]
        assert all(not record["raw_cycle_certified"] for record in baseline)
        assert all(record["original_cycle_id"] is None for record in baseline)
        checks["native_units_missing_optional_and_no_global_axis_inference"] = True
        changed = pd.DataFrame(rows)
        changed.loc[11:, "Cycle_Index"] = 0
        changed.to_csv(path, index=False)
        try:
            list(iter_instrument_cycles(path, metadata, chunksize=4))
        except ValueError:
            checks["instrument_counter_reset_rejected_without_reindexing"] = True
        else:
            raise AssertionError("Counter reversal was silently accepted")
        changed = pd.DataFrame(rows)
        changed["Cycle_Index"] = changed["Cycle_Index"].astype(float)
        changed.loc[8, "Cycle_Index"] = 1.5
        changed.to_csv(path, index=False)
        try:
            list(iter_instrument_cycles(path, metadata, chunksize=4))
        except ValueError:
            checks["noninteger_instrument_index_rejected"] = True
        else:
            raise AssertionError("Noninteger instrument index was accepted")
        changed = pd.DataFrame(rows)
        changed.to_csv(path, index=False)
        path.write_bytes(path.read_bytes() + b"999,100,1000,20,2,4,-4")
        prefix = list(iter_instrument_cycles(path, metadata, chunksize=3, incomplete_prefix=True))
        assert sum(len(record["V"]) for record in prefix) == len(rows)
        assert all(not record["source_file_complete"] for record in prefix)
        checks["partial_download_trailing_text_trimmed_and_flagged"] = True
        restarted_rows = rows.copy()
        for extra, native in enumerate((0, 0, 0, 1, 1)):
            index = len(rows) + extra
            restarted_rows.append({**rows[extra], "Data_Point": index,
                                   "DateTime": 1000000 + index * 5, "Test_Time": extra * 5,
                                   "Step_Index": 0 if native == 0 else 2, "Cycle_Index": native})
        pd.DataFrame(restarted_rows).to_csv(path, index=False)
        for chunk_size in (1, 3, 7, 14):
            restarted = list(iter_instrument_cycles(path, metadata, chunksize=chunk_size))
            assert [record["instrument_cycle_id"] for record in restarted] == [0, 1, 4, 0, 1]
            assert [record["run_index"] for record in restarted] == [0, 0, 0, 1, 1]
            assert [record["instrument_run_cycle_offset"] for record in restarted] == [0, 0, 0, 4, 4]
            assert [len(record["V"]) for record in restarted] == [3, 7, 4, 3, 2]
            assert restarted[2]["run_terminal_cycle"] and not restarted[2]["source_terminal_cycle"]
            assert restarted[3]["run_boundary_evidence"]["source_row_after"] == 14
            assert restarted[3]["run_boundary_evidence"]["restart_rule_authenticated"]
        checks["authenticated_restart_separates_same_local_IDs_across_4_chunk_sizes"] = True
        for field, value in (("Step_Index", 2), ("Data_Point", 13), ("DateTime", 1000000)):
            broken = pd.DataFrame(restarted_rows)
            broken.loc[14, field] = value
            broken.to_csv(path, index=False)
            try:
                list(iter_instrument_cycles(path, metadata, chunksize=7))
            except ValueError:
                checks["restart_without_valid_" + field + "_evidence_rejected"] = True
            else:
                raise AssertionError("Uncertified restart was silently accepted for " + field)
    native = baseline[1]
    release = {key: native[key].copy() for key in ("V", "I", "Qc", "Qd", "t")}
    release["I"] /= 1.1
    release["t"] = (native["t"] - native["t"][0]) / 60
    release["source_record_index"] = 55
    binding = compare_instrument_release(native, release)
    assert binding["matches"] and binding["instrument_cycle_id"] == 1
    release["V"][3] += 0.01
    assert not compare_instrument_release(native, release)["matches"]
    checks["independent_native_to_release_binding_and_negative_control"] = True
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = run_synthetic_checks()
    real = []
    for batch in range(1, 5):
        schema_path = args.sample_root / f"CSV_SCHEMA_BATCH{batch}.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        path = args.sample_root / schema["sample_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == schema["prefix_sha256"]
        records = list(iter_instrument_cycles(path, schema, chunksize=113, incomplete_prefix=True))
        assert all(not record["raw_cycle_certified"] and not record["source_file_complete"] for record in records)
        real.append({"source_batch": batch, "test_id": schema["test_id"],
                     "sample_sha256": schema["prefix_sha256"], "source_field_names": records[0]["source_field_names"],
                     "instrument_cycle_ids_seen": [record["instrument_cycle_id"] for record in records],
                     "rows_read": sum(len(record["V"]) for record in records),
                     "first_Test_Time_s": float(records[0]["t"][0]),
                     "first_current_A": float(records[0]["I"][0]),
                     "source_file_complete": False, "raw_cycle_certified": False})
    result = {"status": "PASS", "synthetic_checks": checks, "real_four_batch_prefix_checks": real,
              "scope": "Streaming properties and genuine 64KiB schema samples. Whole-log SHA, identity, full instrument-cycle coverage and continuation QA remain separate.",
              "module_sha256": hashlib.sha256(Path(__file__).with_name("instrument_csv.py").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "synthetic_checks": len(checks), "actual_source_batches": len(real)}))


if __name__ == "__main__":
    main()
