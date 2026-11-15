"""Inspect a complete native CSV through the actual measurement/feature APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from instrument_csv import iter_instrument_cycles
from measurements import measure_cycle
from physics_features import extract_prefix_features


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cell-uid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    settings = json.loads((repository / "configs/unified_matr_seed42.json").read_text(encoding="utf-8"))["discharge"]
    source_metadata = {"cell_uid": args.cell_uid, "source_segment_id": args.cell_uid.removeprefix("MATR_"),
                       "status": "COMPLETE", "file_complete": True}
    voltage = np.linspace(2.8, 3.3, 64)
    q = np.full((100, 64), np.nan)
    valid = np.zeros(100, dtype=bool)
    rows = []
    for record in iter_instrument_cycles(args.csv, source_metadata):
        measurement = measure_cycle(record, settings, voltage)
        cycle = record["instrument_cycle_id"]
        if 1 <= cycle <= 100 and measurement["curve_valid"]:
            q[cycle - 1] = measurement["q_absolute"]
            valid[cycle - 1] = True
        rows.append({"instrument_cycle_id": cycle, "capacity_valid": measurement["capacity_valid"],
                     "curve_valid": measurement["curve_valid"], "capacity_Ah": measurement["capacity_Ah"],
                     "ratio": measurement["integration_counter_ratio"],
                     "reasons": measurement["failure_reasons"],
                     "source_row_start": record["source_row_start"],
                     "source_row_stop_exclusive": record["source_row_stop_exclusive"]})
    features = extract_prefix_features(q, valid, np.arange(1, 101), voltage)
    extended = extract_prefix_features(np.vstack((q, np.full((20, 64), 1e20))),
                                       np.r_[valid, np.ones(20, bool)], np.arange(1, 121), voltage)
    causal = all(np.array_equal(features[key], extended[key], equal_nan=True)
                 for key in ("global_features", "global_mask", "local_residual", "local_mask",
                             "local_features", "local_feature_mask", "reference_q"))
    ratios = [row["ratio"] for row in rows if row["capacity_valid"] and row["ratio"] is not None]
    ids = np.asarray([row["instrument_cycle_id"] for row in rows])
    digest = hashlib.sha256()
    with args.csv.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    report = {"status": "REAL_INSTRUMENT_ENGINEERING_PILOT", "cell_uid": args.cell_uid,
              "source_csv_sha256": digest.hexdigest(), "source_bytes": args.csv.stat().st_size,
              "native_cycle_count": len(rows), "native_cycle_first": int(ids.min()), "native_cycle_last": int(ids.max()),
              "native_cycle_gaps": int(np.maximum(np.diff(ids) - 1, 0).sum()),
              "valid_capacities": sum(row["capacity_valid"] for row in rows),
              "qualified_original_1_to100_curves": int(valid.sum()),
              "reference_cycles": features["reference_cycles"],
              "global_valid_values": int(features["global_mask"].sum()),
              "local_valid_values": int(features["local_feature_mask"].sum()),
              "integration_counter_ratio_median": float(np.median(ratios)) if ratios else None,
              "failure_counts": dict(Counter(reason for row in rows for reason in row["reasons"])),
              "real_prefix_future_perturbation_invariant": causal,
              "first_rows": rows[:3], "last_rows": rows[-3:],
              "formal_training_runs": 0, "optimizer_updates": 0,
              "scope": "One complete original instrument CSV. Source identity/continuation registry and full-cohort data freeze remain separate requirements."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(clean(report), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
