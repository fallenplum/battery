"""Record small source-native anomaly witnesses after full-cohort audit."""

import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from hust_safe_pickle import dataframe_columns, parse_pickle_data

OUT = REPO / "results" / "hust_raw_idea_audit_v2"


def describe(columns, phase, left, right):
    i = columns["Current (mA)"][left:right]
    q = columns["Capacity (mAh)"][left:right]
    v = columns["Voltage (V)"][left:right]
    t = columns["Time (s)"][left:right]
    return {"phase": phase, "source_row_position_0based_left_inclusive": int(left), "source_row_position_0based_right_exclusive": int(right), "points": len(i),
        "current_min_mA": float(i.min()), "current_max_mA": float(i.max()), "current_median_mA": float(np.median(i)),
        "capacity_first_mAh": float(q[0]), "capacity_last_mAh": float(q[-1]), "zero_capacity_delta_steps_within1e_9mAh": int((np.abs(np.diff(q)) <= 1e-9).sum()),
        "voltage_min_V": float(v.min()), "voltage_max_V": float(v.max()), "time_first_s": int(t[0]), "time_last_s": int(t[-1])}


def point(columns, index):
    return {"source_row_position_0based": int(index), **{name: values[index].item() if isinstance(values[index], np.generic) else values[index] for name, values in columns.items()}}


def write_cell_flags(result):
    with (OUT / "raw_cell_metadata.csv").open(encoding="utf-8-sig") as stream:
        cells = list(csv.DictReader(stream))
    flags = []
    for cell in cells:
        identifier = cell["cell_id"]
        gaps = [row for row in result["long_or_duplicate_time_gaps"] if row["cell_id"] == identifier]
        interruptions = [row["raw_cycle"] for row in gaps if row["delta_time_s"] > 30]
        flags.append({"cell_id": identifier, "source_capacity_target_retained": True,
            "source_capacity_independent_measurement_authenticated": False,
            "capacity_target_measurement_comparability_review_cycles": json.dumps(interruptions),
            "recorded_time_integration_unsupported_cycles": json.dumps(sorted({row["raw_cycle"] for row in gaps})),
            "full_phase_charge_current_deviation_cycles": json.dumps([row["raw_cycle"] for row in result["charge_protocol_deviations"] if row["cell_id"] == identifier]),
            "D3_strict_bracket_missing_prefix_cycles": json.dumps([int(row["raw_cycle"]) for row in result["early_window_failures"] if row["cell_id"] == identifier and row["view"] == "D3_2p8_3p1"]),
            "CC_fewer_than2_inside_points_prefix_cycles": json.dumps([int(row["raw_cycle"]) for row in result["early_window_failures"] if row["cell_id"] == identifier and row["view"] == "CC_3p1_3p4"]),
            "cell_excluded_using_suffix_anomalies": False, "future_policy_known_at_inference_authenticated": False})
    with (OUT / "cell_qualification_flags.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flags[0]))
        writer.writeheader()
        writer.writerows(flags)


def main():
    identity = json.loads((OUT / "CURRENT_ARCHIVE_IDENTITY.json").read_text(encoding="utf-8"))
    with (OUT / "early_window_support.csv").open(encoding="utf-8-sig") as stream:
        windows = [row for row in csv.DictReader(stream) if row["phase_brackets_both_endpoints"] == "False" or int(row["native_points"]) < 2]
    window_pairs = {(row["cell_id"], int(row["raw_cycle"])) for row in windows}
    selected = {"1-8", "7-7", "4-5", "9-5", "9-7"} | {r["cell_id"].removeprefix("HUST_") for r in windows}
    result = {"early_window_failures": windows, "early_window_source_phases": [], "long_or_duplicate_time_gaps": [], "charge_protocol_deviations": [],
        "gap_rest_identity_certified": False, "source_capture_algorithm_certified": False,
        "qualification_scope": "Time-gap QC qualifies recorded-time integration independently from source-provided Capacity/Q(V). Suffix deviations are not used to select prefix cells."}
    with zipfile.ZipFile(identity["replay_inner_cache_path"]) as archive:
        for stem in sorted(selected):
            graph, _ = parse_pickle_data(archive.read("our_data/" + stem + ".pkl"))
            cell_id = "HUST_" + stem
            for number, frame in graph[stem]["data"].items():
                columns = dataframe_columns(frame)
                status = columns["Status"]
                bounds = np.r_[0, np.flatnonzero(status[1:] != status[:-1]) + 1, len(status)]
                phases = [(str(status[left]), left, right) for left, right in zip(bounds[:-1], bounds[1:])]
                if (cell_id, number) in window_pairs:
                    result["early_window_source_phases"].append({"cell_id": cell_id, "raw_cycle": number, "is_prefix100": number <= 100,
                        "phases": [describe(columns, phase, left, right) for phase, left, right in phases]})
                phase, left, right = phases[0]
                current = columns["Current (mA)"][left:right]
                if abs(np.median(current) - 5500) > 55 or np.any(np.abs(current - 5500) > 55):
                    after_hi = np.flatnonzero(columns["Voltage (V)"][left:right] >= 3.4)
                    bracket_end = left + int(after_hi[0]) + 1 if len(after_hi) else right
                    result["charge_protocol_deviations"].append({"cell_id": cell_id, "raw_cycle": number, "is_prefix100": number <= 100,
                        **describe(columns, phase, left, right), "points_outside5C_plusminus0p05C": int((np.abs(current - 5500) > 55).sum()),
                        "charge_begin_through_first3p4V_upper_bracket": describe(columns, phase, left, bracket_end),
                        "major_current_transitions": [{"before": point(columns, left + int(index)), "after": point(columns, left + int(index) + 1)} for index in np.flatnonzero(np.abs(np.diff(current)) > 55)]})
                for phase, left, right in phases:
                    t = columns["Time (s)"][left:right]
                    for relative_index in np.flatnonzero((np.diff(t) > 30) | (np.diff(t) <= 0)):
                        index = left + int(relative_index)
                        delta_t = int(columns["Time (s)"][index + 1] - columns["Time (s)"][index])
                        delta_q = float(columns["Capacity (mAh)"][index + 1] - columns["Capacity (mAh)"][index])
                        current_pair = columns["Current (mA)"][index:index + 2]
                        trapezoid_mAh = float(np.mean(current_pair) * delta_t / 3600)
                        result["long_or_duplicate_time_gaps"].append({"cell_id": cell_id, "raw_cycle": number, "is_prefix100": number <= 100,
                            **describe(columns, phase, left, right), "delta_time_s": delta_t, "delta_capacity_mAh": delta_q,
                            "current_trapezoid_recorded_time_mAh": trapezoid_mAh, "capacity_minus_trapezoid_mAh": delta_q - trapezoid_mAh,
                            "recorded_time_integration_supported_for_this_step": False, "source_capacity_target_retained": True,
                            "source_capacity_measurement_comparability_review_flag": "UNAUTHENTICATED_SOURCE_INTERRUPTION" if delta_t > 30 else "REPEATED_INTEGER_TIME",
                            "prefix_cell_exclusion_due_to_this_event": False,
                            "before": point(columns, index), "after": point(columns, index + 1)})
            print(json.dumps({"targeted_source_cell_completed": cell_id}), flush=True)
    (OUT / "source_anomaly_details.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_cell_flags(result)
    print(json.dumps({"window_failures": len(result["early_window_failures"]), "charge_deviations": len(result["charge_protocol_deviations"]), "time_gap_events": len(result["long_or_duplicate_time_gaps"])}))


if __name__ == "__main__":
    main()
