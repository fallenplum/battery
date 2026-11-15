"""Phase-qualified original-instrument MATR capacity and discharge response."""

from __future__ import annotations

import hashlib

import numpy as np


def longest_contiguous(indices):
    if not len(indices):
        return np.asarray([], dtype=int)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return max(groups, key=len)


def measure_cycle(record, settings, voltage_grid):
    """Keep original counters and field-level reasons alongside every mask."""
    output = {"capacity_valid": False, "curve_valid": False, "capacity_Ah": None,
              "summary_counter_difference_Ah": None, "summary_counter_agrees": False,
              "capacity_source": "phase_confirmed_instrument_discharge_counter",
              "source_record_index": record["source_record_index"],
              "source_record_cycle": record["source_record_cycle"],
              "source_segment_id": record["source_segment_id"], "failure_reasons": [],
              "warning_reasons": [], "time_valid": False, "integration_valid": False,
              "time_reversal_count": 0,
              "discharge_step_index": None, "phase_boundary_basis": None,
              "discharge_points": 0, "window_points": 0, "raw_fingerprint": None,
              "integration_counter_ratio": None,
              "q_absolute": np.full(len(voltage_grid), np.nan),
              "raw_viq": np.full((len(voltage_grid), 3), np.nan)}
    reasons = output["failure_reasons"]
    fields = {name: np.asarray(record[name], dtype=float) for name in ("V", "I", "Qd")}
    counts = {name: len(values) for name, values in fields.items()}
    if len(set(counts.values())) != 1 or counts["V"] < settings["min_contiguous_points"]:
        reasons.append("VIQ_POINT_COUNTS_DIFFER_OR_TOO_SHORT")
        return output
    if not all(np.isfinite(values).all() for values in fields.values()):
        reasons.append("NONFINITE_DISCHARGE_VIQ")
        return output
    voltage, qd = fields["V"], fields["Qd"]
    current = fields["I"] * record.get("source_current_scale_to_A", settings["source_current_scale_to_A"])
    time = np.asarray(record.get("t", []), dtype=float) * record.get("source_time_scale_to_seconds", settings["source_time_scale_to_seconds"])
    charge_counter = np.asarray(record.get("Qc", []), dtype=float)
    if len(time) != len(voltage):
        output["warning_reasons"].append("TIME_POINT_COUNT_DIFFERS")
    elif not np.isfinite(time).all():
        output["warning_reasons"].append("NONFINITE_TIME")
    else:
        output["time_reversal_count"] = int((np.diff(time) < -1e-8).sum())
        output["time_valid"] = output["time_reversal_count"] == 0
        if not output["time_valid"]:
            output["warning_reasons"].append("TIME_COUNTER_REVERSES_WITHIN_CYCLE")
    charge_counter_valid = len(charge_counter) == len(voltage) and np.isfinite(charge_counter).all()
    if not charge_counter_valid:
        output["warning_reasons"].append("CHARGE_COUNTER_UNAVAILABLE_FOR_AUDIT")
    if record.get("source_kind") == "instrument_csv":
        data_point = np.asarray(record.get("data_point", []), dtype=float)
        if len(data_point) != len(voltage) or not np.isfinite(data_point).all() or (np.diff(data_point) <= 0).any():
            reasons.append("INSTRUMENT_ACQUISITION_ORDER_UNCERTIFIED")
            return output
    digest = hashlib.sha256()
    for name in ("V", "I", "Qd"):
        digest.update(np.asarray(fields[name], dtype="<f8").tobytes())
    if charge_counter_valid:
        digest.update(np.asarray(charge_counter, dtype="<f8").tobytes())
    if output["time_valid"]:
        digest.update(np.asarray(time - time[0], dtype="<f8").tobytes())
    output["raw_fingerprint"] = digest.hexdigest()
    nominal_current = settings["nominal_capacity_Ah"] * settings["design_rate_C"]
    tolerance = nominal_current * settings["current_tolerance_fraction"]
    phase = longest_contiguous(np.flatnonzero(np.abs(current + nominal_current) <= tolerance))
    output["discharge_points"] = len(phase)
    if len(phase) < settings["min_contiguous_points"]:
        reasons.append("INSUFFICIENT_CONTIGUOUS_4C_DISCHARGE")
        return output
    output["constant_current_points"] = int(len(phase))
    output["constant_current_point_start"] = int(phase[0])
    output["constant_current_point_stop_exclusive"] = int(phase[-1] + 1)
    v, q, i = voltage[phase], qd[phase], current[phase]
    output["discharge_current_median_A"] = float(np.median(i))
    output["constant_current_start_voltage_V"] = float(v[0])
    output["constant_current_end_voltage_V"] = float(v[-1])
    if (np.diff(q) < -settings["capacity_reset_tolerance_Ah"]).any():
        reasons.append("DISCHARGE_COUNTER_RESETS_INSIDE_PHASE")
    step_index = np.asarray(record.get("step_index", []), dtype=float)
    anchor = int(phase[len(phase) // 2])
    if len(step_index) == len(current) and np.isfinite(step_index).all():
        step = step_index[anchor]
        before = np.flatnonzero(step_index[:anchor] != step)
        after = np.flatnonzero(step_index[anchor + 1:] != step)
        left = int(before[-1] + 1) if len(before) else 0
        right = int(anchor + after[0]) if len(after) else len(current) - 1
        output["discharge_step_index"] = float(step)
        output["phase_boundary_basis"] = "contiguous instrument step containing the longest4C constant-current segment"
    else:
        positive = np.flatnonzero(current > settings["minimum_discharge_current_A"])
        before, after = positive[positive < phase[0]], positive[positive > phase[-1]]
        left = int(before[-1] + 1) if len(before) else 0
        right = int(after[0] - 1) if len(after) else len(current) - 1
        output["phase_boundary_basis"] = "nonpositive/rest block anchored to the longest4C constant-current segment"
    block = np.arange(left, right + 1)
    full_negative = block[current[block] < -settings["minimum_discharge_current_A"]]
    full_phase = np.arange(full_negative[0], full_negative[-1] + 1)
    output["full_discharge_points"] = int(len(full_phase))
    output["discharge_point_start"] = int(full_phase[0])
    output["discharge_point_stop_exclusive"] = int(full_phase[-1] + 1)
    output["discharge_start_voltage_V"] = float(voltage[full_phase[0]])
    output["discharge_end_voltage_V"] = float(voltage[full_phase[-1]])
    output["discharge_current_min_A"] = float(current[full_phase].min())
    output["discharge_current_max_A"] = float(current[full_phase].max())
    if (current[full_phase] < -nominal_current - tolerance).any():
        reasons.append("DISCHARGE_EXCITATION_EXCEEDS_REGISTERED_4C_TOLERANCE")
    if (current[full_phase] > settings["minimum_discharge_current_A"]).any():
        reasons.append("DISCHARGE_PHASE_CONTAINS_POSITIVE_CURRENT")
    if voltage[full_phase[0]] < settings["full_discharge_min_start_voltage_V"]:
        reasons.append("DISCHARGE_START_VOLTAGE_INCOMPLETE")
    if voltage[full_phase[-1]] > settings["full_discharge_max_end_voltage_V"]:
        reasons.append("DISCHARGE_END_VOLTAGE_INCOMPLETE")
    preceding = qd[:full_phase[0]]
    zero = float(np.median(preceding[-5:])) if len(preceding) else float(q[0])
    output["instrument_discharge_counter_zero_Ah"] = zero
    if abs(zero) > settings["capacity_zero_tolerance_Ah"]:
        reasons.append("DISCHARGE_COUNTER_ZERO_UNCERTIFIED")
    end = int(full_phase[-1])
    if end + 1 < len(qd) and current[end + 1] <= settings["minimum_discharge_current_A"]:
        end += 1
    capacity = float(qd[end])
    output["counter_endpoint_point_index"] = end
    output["adjacent_stage_counter_increment_Ah"] = float(qd[end] - qd[full_phase[-1]])
    if (np.diff(qd[full_phase]) < -settings["capacity_reset_tolerance_Ah"]).any():
        reasons.append("DISCHARGE_COUNTER_RESETS_INSIDE_CC_CV_PHASE")
    summary_capacity = record["summary_QD_Ah"]
    if summary_capacity is None and record.get("source_kind") == "instrument_csv":
        output["summary_counter_agrees"] = None
    elif summary_capacity is None or not np.isfinite(summary_capacity):
        reasons.append("SUMMARY_DISCHARGE_CAPACITY_NONFINITE")
    else:
        difference = float(summary_capacity - capacity)
        output["summary_counter_difference_Ah"] = difference
        output["summary_counter_agrees"] = abs(difference) <= settings["max_summary_counter_difference_Ah"]
        if not output["summary_counter_agrees"]:
            reasons.append("SUMMARY_WAVEFORM_DISCHARGE_COUNTER_DISAGREES")
    if not 0 < capacity <= 1.5 * settings["nominal_capacity_Ah"]:
        reasons.append("DISCHARGE_CAPACITY_OUT_OF_INSTRUMENT_RANGE")
    counter_delta = float(qd[full_phase[-1]] - qd[full_phase[0]])
    if output["time_valid"] and counter_delta > 0:
        integrated_Ah = float(np.trapz(-current[full_phase], time[full_phase]) / 3600)
        output["integration_counter_ratio"] = integrated_Ah / counter_delta
        output["integration_valid"] = True
    # The device counter remains the measurement. Integral closure is an audit field.
    output["capacity_valid"] = not reasons
    if output["capacity_valid"]:
        output["capacity_Ah"] = capacity
    support = (v >= voltage_grid[0]) & (v <= voltage_grid[-1])
    output["window_points"] = int(support.sum())
    curve_reasons = []
    if v.max() < voltage_grid[-1] or v.min() > voltage_grid[0]:
        curve_reasons.append("WINDOW_FULL_SUPPORT_MISSING")
    window_indices = np.flatnonzero(support)
    window_voltage = v[window_indices[0]:window_indices[-1] + 1] if len(window_indices) else np.asarray([])
    if (np.diff(window_voltage) > settings["max_voltage_reversal_V"]).any():
        curve_reasons.append("VOLTAGE_REVERSAL_EXCEEDS_TOLERANCE")
    if support.sum() < settings["min_contiguous_points"]:
        curve_reasons.append("WINDOW_SAMPLING_TOO_SPARSE")
    if not output["capacity_valid"]:
        curve_reasons.append("PARENT_CAPACITY_MEASUREMENT_INVALID")
    if curve_reasons:
        reasons.extend(curve_reasons)
        return output
    # Keep a strictly decreasing subsequence in original acquisition order.
    # Omitted sub-tolerance reversals remain counted in the audit.
    keep = [0]
    for position in range(1, len(v)):
        if v[position] < v[keep[-1]] - 1e-8:
            keep.append(position)
    keep = np.asarray(keep, dtype=int)
    selected_voltage = v[keep][::-1]
    if selected_voltage[0] > voltage_grid[0] or selected_voltage[-1] < voltage_grid[-1]:
        reasons.append("MONOTONE_SUBSEQUENCE_WINDOW_SUPPORT_MISSING")
        return output
    q_grid = np.interp(voltage_grid, selected_voltage, q[keep][::-1])
    i_grid = np.interp(voltage_grid, selected_voltage, i[keep][::-1])
    output["q_absolute"] = q_grid
    output["raw_viq"] = np.column_stack((voltage_grid, i_grid, q_grid))
    output["curve_valid"] = True
    output["voltage_reversal_points_omitted"] = int(len(v) - len(keep))
    return output


def selfcheck():
    import json
    from pathlib import Path

    repository = Path(__file__).resolve().parents[2]
    settings = json.loads((repository / "configs/unified_matr_seed42.json").read_text(encoding="utf-8"))["discharge"]
    voltage_grid = np.linspace(2.8, 3.3, 64)
    voltage = np.r_[np.full(20, 3.6), np.linspace(3.5, 2.0, 301)]
    current = np.r_[np.full(20, 1.1), np.full(301, -4.4)]
    charge = np.r_[np.zeros(20), np.linspace(0, 1.1, 301)]
    record = {"V": voltage, "I": current / settings["source_current_scale_to_A"], "Qd": charge, "Qc": np.zeros(len(voltage)),
              "t": np.arange(len(voltage)) * (15 / 300), "summary_QD_Ah": 1.1,
              "source_record_index": 0, "source_record_cycle": 1, "source_segment_id": "selfcheck"}
    clean = measure_cycle(record, settings, voltage_grid)
    assert clean["capacity_valid"] and clean["curve_valid"]
    assert np.isclose(clean["q_absolute"][-1], 1.1 * (3.5 - 3.3) / 1.5)
    assert np.isclose(clean["integration_counter_ratio"], 1)
    changed = {**record, "Qd": charge.copy()}
    changed["Qd"][150:] -= 0.05
    assert not measure_cycle(changed, settings, voltage_grid)["capacity_valid"]
    changed = {**record, "summary_QD_Ah": 0.5}
    assert "SUMMARY_WAVEFORM_DISCHARGE_COUNTER_DISAGREES" in measure_cycle(changed, settings, voltage_grid)["failure_reasons"]
    changed = {**record, "V": voltage + 1}
    assert not measure_cycle(changed, settings, voltage_grid)["curve_valid"]
    changed = {**record, "t": record["t"].copy()}
    changed["t"][170:] -= 100
    clock_fault = measure_cycle(changed, settings, voltage_grid)
    assert clock_fault["capacity_valid"] and clock_fault["curve_valid"]
    assert not clock_fault["time_valid"] and not clock_fault["integration_valid"]
    assert clock_fault["integration_counter_ratio"] is None
    assert "TIME_COUNTER_REVERSES_WITHIN_CYCLE" in clock_fault["warning_reasons"]
    missing_time = measure_cycle({**record, "t": np.array([])}, settings, voltage_grid)
    assert missing_time["capacity_valid"] and missing_time["curve_valid"]
    missing_charge = measure_cycle({**record, "Qc": np.array([])}, settings, voltage_grid)
    assert missing_charge["capacity_valid"] and missing_charge["curve_valid"]
    micro_pulse = {**record, "I": record["I"].copy()}
    micro_pulse["I"][2] = -0.008 / settings["source_current_scale_to_A"]
    assert measure_cycle(micro_pulse, settings, voltage_grid)["capacity_valid"]
    step_case = {**record, "step_index": np.r_[np.full(20, 3), np.full(301, 10)]}
    for key in ("V", "I", "Qd", "Qc", "t", "step_index"):
        tail = {"V": 2.9, "I": -0.008 / settings["source_current_scale_to_A"],
                "Qd": 1.1, "Qc": 0, "t": 100, "step_index": 11}[key]
        step_case[key] = np.r_[step_case[key], np.full(20, tail)]
    scoped = measure_cycle(step_case, settings, voltage_grid)
    assert scoped["capacity_valid"] and scoped["curve_valid"] and scoped["discharge_step_index"] == 10
    ir_pulse = {**step_case, "I": step_case["I"].copy()}
    ir_pulse["I"][2] = -13.4 / settings["source_current_scale_to_A"]
    assert measure_cycle(ir_pulse, settings, voltage_grid)["capacity_valid"]
    overcurrent = {**step_case, "I": step_case["I"].copy()}
    overcurrent["I"][50] = -13.4 / settings["source_current_scale_to_A"]
    assert not measure_cycle(overcurrent, settings, voltage_grid)["capacity_valid"]
    print(json.dumps({"status": "PASS", "checks": 17, "scope": "mathematical measurement controls; no scientific data evaluation"}))


if __name__ == "__main__":
    selfcheck()
