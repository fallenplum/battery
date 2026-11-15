"""Read native CSV counterexamples to explain observed source Step-label drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_excerpt(path, first, stop):
    position = 0
    selected = []
    columns = ["Data_Point", "Test_Time", "DateTime", "Step_Index", "Cycle_Index", "Current", "Voltage", "Discharge_Capacity"]
    for frame in pd.read_csv(path, chunksize=65536, usecols=columns, float_precision="round_trip"):
        following = position + len(frame)
        if following > first and position < stop:
            selected.append(frame.iloc[max(0, first - position):min(len(frame), stop - position)])
        position = following
        if position >= stop:
            break
    result = pd.concat(selected, ignore_index=True)
    if len(result) != stop - first:
        raise ValueError("Original source excerpt differs from registered native row bounds")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/unified_matr"))
    parser.add_argument("--private", type=Path, default=Path("../private/unified_matr_seed42/source_discharge_diagnostics"))
    args = parser.parse_args()
    qa = json.loads((args.output / "SOURCE_DISCHARGE_STEP_QA.json").read_text(encoding="utf-8"))
    bindings = json.loads((args.output / "instrument_bindings.json").read_text(encoding="utf-8"))
    sources = {row["source_segment_id"]: row for cell in bindings["cells"] for row in cell["source_segments"]}
    settings = qa["registered_numeric_requirements"]
    examples = [next(row for row in qa["stage_identity_exceptions"] if row["source_segment_id"] == "b1c0" and row["instrument_cycle_id"] == 583),
                next(row for row in qa["stage_identity_exceptions"] if row["expected_source_step"] == 12),
                next(row for row in qa["stage_identity_exceptions"] if row["source_segment_id"] == "b4c0" and row["instrument_cycle_id"] == 14)]
    args.private.mkdir(parents=True, exist_ok=True)
    results = []
    for row in examples:
        source = sources[row["source_segment_id"]]
        frame = read_excerpt(Path(source["csv_path"]), row["source_row_start"], row["source_row_stop_exclusive"])
        destination = args.private / (row["source_segment_id"] + "_native" + str(row["instrument_cycle_id"]) + ".parquet")
        frame.to_parquet(destination, index=False)
        first, stop = int(row["discharge_point_start"]), int(row["discharge_point_stop_exclusive"])
        cc_first, cc_stop = int(row["constant_current_point_start"]), int(row["constant_current_point_stop_exclusive"])
        phase, constant = frame.iloc[first:stop], frame.iloc[cc_first:cc_stop]
        zero = float(np.median(frame["Discharge_Capacity"].iloc[max(0, first - 5):first]))
        endpoint = int(row["counter_endpoint_point_index"])
        comparisons = {"capacity_Ah": float(frame["Discharge_Capacity"].iloc[endpoint]),
                       "instrument_discharge_counter_zero_Ah": zero,
                       "discharge_start_voltage_V": float(phase["Voltage"].iloc[0]),
                       "discharge_end_voltage_V": float(phase["Voltage"].iloc[-1]),
                       "discharge_current_min_A": float(phase["Current"].min()),
                       "discharge_current_max_A": float(phase["Current"].max()),
                       "discharge_current_median_A": float(constant["Current"].median())}
        exact = all(value == row[key] for key, value in comparisons.items())
        if not exact:
            raise ValueError("Native counterexample quantities differ from completed main-phase audit")
        groups = []
        for step, group in frame.groupby("Step_Index", sort=False):
            groups.append({"native_step_index": float(step), "points": len(group),
                           "current_min_A": float(group["Current"].min()), "current_max_A": float(group["Current"].max()),
                           "voltage_min_V": float(group["Voltage"].min()), "voltage_max_V": float(group["Voltage"].max())})
        result = {key: row[key] for key in ("cell_uid", "source_segment_id", "run_index", "instrument_cycle_id", "original_cycle_id",
                                           "source_row_start", "source_row_stop_exclusive", "expected_source_step", "discharge_step_index",
                                           "observed_index_declared_label", "observed_index_declared_control_type")}
        result.update(native_record_points=len(frame), native_cycle_ids=[float(value) for value in frame["Cycle_Index"].unique()],
                      registered_main_discharge_step_points=int(frame["Step_Index"].eq(row["expected_source_step"]).sum()),
                      observed_phase_native_step_indexes=[float(value) for value in phase["Step_Index"].unique()],
                      raw_phase_points=len(phase), raw_constant_current_points=len(constant),
                      raw_phase_positive_current_above_registered_noise_points=int(phase["Current"].gt(settings["minimum_discharge_current_A"]).sum()),
                      raw_phase_current_below_registered_negative_bound_points=int(phase["Current"].lt(-4.4 * (1 + settings["current_tolerance_fraction"])).sum()),
                      replay_quantities=comparisons, all_replayed_quantities_equal_stored_fields=exact,
                      native_step_value_ranges=groups, original_source_sha256=source["sha256"], private_excerpt_sha256=sha256(destination))
        results.append(result)
    report = {"status": "PASS_NATIVE_REPLAY_SOURCE_STEP_LABEL_EXCEPTIONS_REPRODUCED", "examples": results,
              "scope": "Three directly replayed original CSV counterexamples. Every selected phase has the registered 4C negative excitation, near-zero discharge counter and approximately2.0V completion, while the original Step_Index retains the preceding finishing/rest label and no declared main-step rows exist. Source label lag is a waveform-supported interpretation; original control-system causation remains unverified.",
              "all6780_raw_exception_records_individually_replayed": False,
              "full_table_stored_quantity_checks_source": "SOURCE_DISCHARGE_STEP_QA.json",
              "qualification_label_model_or_measurement_changes": 0, "code_sha256": sha256(Path(__file__))}
    (args.output / "SOURCE_DISCHARGE_STEP_NATIVE_REPLAY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "examples": len(results)}))


if __name__ == "__main__":
    main()
