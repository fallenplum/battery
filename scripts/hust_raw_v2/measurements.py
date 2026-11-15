"""HUST source-provided Capacity views with explicit native support audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


CHARGE = "Constant current charge"
DISCHARGE = tuple(f"Constant current discharge_{number}" for number in range(4))
EXPECTED_PHASES = (CHARGE, "Constant current-constant voltage charge", *DISCHARGE)
NOMINAL_AH = 1.1
COUNTER_TOLERANCE_MAH = 1e-6


@dataclass(frozen=True)
class ViewConfig:
    name: str
    phase: str
    direction: str
    current_A: float
    voltage_min_V: float
    voltage_max_V: float
    grid_size: int = 64
    current_tolerance_A: float = 0.055
    minimum_window_native_points: int = 15
    minimum_distinct_voltage_points: int = 12
    maximum_voltage_reversal_V: float = 0.02
    minimum_window_charge_Ah: float = 0.011


VIEWS = (
    ViewConfig("actual2C_loaded_response", DISCHARGE[-1], "discharge", -2.2, 2.8, 3.1),
    ViewConfig("common5C_charge_response", CHARGE, "charge", 5.5, 3.1, 3.4),
)


def phase_blocks(columns):
    status = np.asarray(columns["Status"])
    boundaries = np.r_[0, np.flatnonzero(status[1:] != status[:-1]) + 1, len(status)]
    return [(str(status[left]), int(left), int(right))
            for left, right in zip(boundaries[:-1], boundaries[1:])]


def phase_audit(columns):
    output = []
    for name, left, right in phase_blocks(columns):
        voltage = np.asarray(columns["Voltage (V)"][left:right], dtype=float)
        current = np.asarray(columns["Current (mA)"][left:right], dtype=float) / 1000
        capacity = np.asarray(columns["Capacity (mAh)"][left:right], dtype=float)
        output.append({"phase": name, "start_point": left, "stop_point_exclusive": right,
                       "native_points": right - left, "current_median_A": float(np.median(current)),
                       "voltage_start_V": float(voltage[0]), "voltage_end_V": float(voltage[-1]),
                       "voltage_min_V": float(voltage.min()), "voltage_max_V": float(voltage.max()),
                       "capacity_start_mAh": float(capacity[0]), "capacity_end_mAh": float(capacity[-1]),
                       "voltage_up_steps_gt_0p02V": int(np.count_nonzero(np.diff(voltage) > .02))})
    return output


def full_discharge_measurement(columns):
    """Measure the first-D0 to last-D3 source-provided net Capacity difference."""
    blocks = phase_blocks(columns)
    selected = [entry for entry in blocks if entry[0] in DISCHARGE]
    reasons = []
    result = {"capacity_valid": False, "capacity_Ah": np.nan, "soh_nominal": np.nan,
              "origin_point": None, "origin_capacity_mAh": np.nan, "end_point": None,
              "end_capacity_mAh": np.nan, "failure_reasons": reasons}
    if tuple(entry[0] for entry in selected) != DISCHARGE:
        reasons.append("FOUR_UNIQUE_ORDERED_DISCHARGE_PHASES_MISSING")
        return result
    if any(first[2] != last[1] for first, last in zip(selected[:-1], selected[1:])):
        reasons.append("DISCHARGE_PHASES_NONCONTIGUOUS")
        return result
    left, right = selected[0][1], selected[-1][2]
    capacity = np.asarray(columns["Capacity (mAh)"][left:right], dtype=float)
    current = np.asarray(columns["Current (mA)"][left:right], dtype=float) / 1000
    result.update(origin_point=left, origin_capacity_mAh=float(capacity[0]),
                  end_point=right - 1, end_capacity_mAh=float(capacity[-1]))
    if not np.isfinite(capacity).all() or not np.isfinite(current).all():
        reasons.append("NONFINITE_DISCHARGE_CAPACITY_OR_CURRENT")
        return result
    changes = np.diff(capacity)
    if np.any(changes > COUNTER_TOLERANCE_MAH):
        reasons.append("SOURCE_NET_CAPACITY_RISES_WITHIN_DISCHARGE")
    if np.any(changes < -0.1 * NOMINAL_AH * 1000):
        reasons.append("SOURCE_CAPACITY_LARGE_DISCONTINUITY")
    if np.any(current >= 0):
        reasons.append("DISCHARGE_CURRENT_SIGN_CONFLICT")
    amount = float(capacity[0] - capacity[-1]) / 1000
    if not 0 < amount < 1.5 * NOMINAL_AH:
        reasons.append("SOURCE_CAPACITY_OUTSIDE_REGISTERED_RANGE")
    result.update(capacity_valid=not reasons, capacity_Ah=amount if not reasons else np.nan,
                  soh_nominal=amount / NOMINAL_AH if not reasons else np.nan)
    return result


def measure_view(columns, original_cycle, config):
    """Resample one contiguous constant-current phase in acquisition order."""
    grid = np.linspace(config.voltage_min_V, config.voltage_max_V, config.grid_size)
    reasons = []
    row = {"original_cycle_id": int(original_cycle), "view": config.name,
           "curve_valid": False, "failure_reasons": reasons, "phase": config.phase,
           "phase_occurrences": 0, "phase_points": 0, "phase_start_point": None,
           "phase_stop_point_exclusive": None, "window_native_points": 0,
           "window_distinct_voltage_points": 0, "interpolation_native_points": 0,
           "interpolation_distinct_voltage_points": 0,
           "voltage_reversal_points_omitted": 0, "repeated_voltage_steps": 0,
           "maximum_reverse_voltage_step_V": np.nan, "full_window_bracket": False,
           "current_median_A": np.nan, "current_min_A": np.nan, "current_max_A": np.nan,
           "current_max_deviation_from_registered_A": np.nan,
           "phase_mixed_current": False, "window_current_max_deviation_A": np.nan,
           "waveform_prefix_current_max_deviation_A": np.nan,
           "interpolation_support_current_max_deviation_A": np.nan,
           "current_dependency_stop_point_exclusive": None,
           "window_native_spacing_min_V": np.nan, "window_native_spacing_median_V": np.nan,
           "window_native_spacing_max_V": np.nan, "window_capacity_span_Ah": np.nan,
           "phase_capacity_start_mAh": np.nan, "phase_capacity_end_mAh": np.nan,
           "q_origin_point": None, "q_origin_capacity_mAh": np.nan,
           "negative_time_steps": 0, "zero_time_steps": 0,
           "native_delta_time_median_s": np.nan, "counter_equivalent_interval_median_s": np.nan,
           "phase_source_capacity_change_Ah": np.nan, "phase_trapezoid_current_integral_Ah": np.nan,
           "source_capacity_over_trapezoid_integral": np.nan,
           "source_capacity_algorithm_certified": False, "temporal_sampling_certified": False}
    arrays = {"q_absolute": np.full(config.grid_size, np.nan),
              "q_native_Ah": np.full(config.grid_size, np.nan),
              "raw_viq": np.full((config.grid_size, 3), np.nan), "voltage_grid_V": grid}
    blocks = [entry for entry in phase_blocks(columns) if entry[0] == config.phase]
    row["phase_occurrences"] = len(blocks)
    if len(blocks) != 1:
        reasons.append("PHASE_MISSING_OR_REPEATED")
        return row, arrays
    _, left, right = blocks[0]
    row.update(phase_start_point=left, phase_stop_point_exclusive=right, phase_points=right - left)
    voltage = np.asarray(columns["Voltage (V)"][left:right], dtype=float)
    current = np.asarray(columns["Current (mA)"][left:right], dtype=float) / 1000
    native_q = np.asarray(columns["Capacity (mAh)"], dtype=float)
    phase_q = native_q[left:right]
    cycle_values = np.asarray(columns["Cycle number"][left:right], dtype=float)
    if not (np.isfinite(voltage).all() and np.isfinite(current).all() and np.isfinite(phase_q).all()):
        reasons.append("NONFINITE_PHASE_V_I_CAPACITY")
        return row, arrays
    if not np.all(cycle_values == original_cycle):
        reasons.append("SOURCE_CYCLE_COLUMN_DIFFERS_FROM_KEY")
    sign = 1.0 if config.direction == "charge" else -1.0
    row.update(current_median_A=float(np.median(current)), current_min_A=float(current.min()),
               current_max_A=float(current.max()),
               current_max_deviation_from_registered_A=float(np.max(np.abs(current - config.current_A))),
               phase_capacity_start_mAh=float(phase_q[0]), phase_capacity_end_mAh=float(phase_q[-1]))
    row["phase_mixed_current"] = row["current_max_deviation_from_registered_A"] > config.current_tolerance_A
    if np.any(sign * np.diff(phase_q) < -COUNTER_TOLERANCE_MAH):
        reasons.append("SOURCE_CAPACITY_DIRECTION_OR_RESET_CONFLICT")
    if np.any(np.abs(np.diff(phase_q)) > 0.1 * NOMINAL_AH * 1000):
        reasons.append("SOURCE_CAPACITY_LARGE_DISCONTINUITY")
    if config.direction == "charge":
        row.update(q_origin_point=left, q_origin_capacity_mAh=float(phase_q[0]))
        if abs(phase_q[0]) > COUNTER_TOLERANCE_MAH:
            reasons.append("CHARGE_NATIVE_ZERO_UNCERTIFIED")
        absolute_q = phase_q / 1000
    else:
        discharge = full_discharge_measurement(columns)
        row.update(q_origin_point=discharge["origin_point"], q_origin_capacity_mAh=discharge["origin_capacity_mAh"])
        if not discharge["capacity_valid"]:
            reasons.append("FULL_DISCHARGE_CAPACITY_ORIGIN_INVALID")
        absolute_q = (discharge["origin_capacity_mAh"] - phase_q) / 1000
    time_values = np.asarray(columns["Time (s)"][left:right], dtype=float)
    if len(time_values) == len(current) and np.isfinite(time_values).all() and len(time_values) > 1:
        dt = np.diff(time_values)
        row.update(negative_time_steps=int(np.count_nonzero(dt < 0)), zero_time_steps=int(np.count_nonzero(dt == 0)),
                   native_delta_time_median_s=float(np.median(dt)))
        equivalent = np.abs(np.diff(phase_q) / 1000) * 3600 / np.maximum(np.abs(current[1:]), 1e-12)
        row["counter_equivalent_interval_median_s"] = float(np.median(equivalent))
        source_change = sign * float(phase_q[-1] - phase_q[0]) / 1000
        row["phase_source_capacity_change_Ah"] = source_change
        if not np.any(dt < 0):
            integral = sign * float(np.trapz(current, time_values)) / 3600
            row["phase_trapezoid_current_integral_Ah"] = integral
            if integral > 0:
                row["source_capacity_over_trapezoid_integral"] = source_change / integral
    window = (voltage >= grid[0]) & (voltage <= grid[-1])
    row["window_native_points"] = int(window.sum())
    row["window_distinct_voltage_points"] = int(len(np.unique(voltage[window])))
    if window.any():
        row["window_current_max_deviation_A"] = float(np.max(np.abs(current[window] - config.current_A)))
    row["full_window_bracket"] = bool(voltage.min() <= grid[0] and voltage.max() >= grid[-1])
    if not row["full_window_bracket"]:
        reasons.append("REGISTERED_WINDOW_FULL_BRACKET_MISSING")
    if row["window_native_points"] < config.minimum_window_native_points:
        reasons.append("REGISTERED_WINDOW_NATIVE_POINTS_INSUFFICIENT")
    if row["window_distinct_voltage_points"] < config.minimum_distinct_voltage_points:
        reasons.append("REGISTERED_WINDOW_DISTINCT_VOLTAGES_INSUFFICIENT")
    positions = np.flatnonzero(window)
    if len(positions):
        window_voltage = voltage[positions[0]:positions[-1] + 1]
        reversed_steps = -sign * np.diff(window_voltage)
        row["maximum_reverse_voltage_step_V"] = float(max(0.0, reversed_steps.max())) if len(reversed_steps) else 0.0
        row["repeated_voltage_steps"] = int(np.count_nonzero(np.diff(window_voltage) == 0))
        if row["maximum_reverse_voltage_step_V"] > config.maximum_voltage_reversal_V:
            reasons.append("REGISTERED_WINDOW_VOLTAGE_REVERSAL_TOO_LARGE")
    keep = [0]
    for position in range(1, len(voltage)):
        if sign * (voltage[position] - voltage[keep[-1]]) > 1e-8:
            keep.append(position)
    keep = np.asarray(keep, dtype=int)
    row["voltage_reversal_points_omitted"] = int(len(voltage) - len(keep))
    if config.direction == "discharge":
        keep = keep[::-1]
    support_v = voltage[keep]
    inside = (support_v >= grid[0]) & (support_v <= grid[-1])
    row["interpolation_native_points"] = int(inside.sum())
    row["interpolation_distinct_voltage_points"] = int(len(np.unique(support_v[inside])))
    if row["interpolation_native_points"] < config.minimum_window_native_points:
        reasons.append("RETAINED_WINDOW_NATIVE_POINTS_INSUFFICIENT")
    if row["interpolation_distinct_voltage_points"] < config.minimum_distinct_voltage_points:
        reasons.append("RETAINED_WINDOW_DISTINCT_VOLTAGES_INSUFFICIENT")
    spacing = np.diff(support_v[inside])
    if len(spacing):
        row.update(window_native_spacing_min_V=float(spacing.min()), window_native_spacing_median_V=float(np.median(spacing)),
                   window_native_spacing_max_V=float(spacing.max()))
    if support_v[0] > grid[0] or support_v[-1] < grid[-1]:
        reasons.append("ACQUISITION_ORDER_MONOTONE_WINDOW_BRACKET_MISSING")
    bracket_left = max(0, int(np.searchsorted(support_v, grid[0], side="right")) - 1)
    bracket_right = min(len(keep) - 1, int(np.searchsorted(support_v, grid[-1], side="left")))
    interpolation_support = keep[bracket_left:bracket_right + 1]
    dependency_stop = int(interpolation_support.max()) + 1
    row["current_dependency_stop_point_exclusive"] = left + dependency_stop
    row["interpolation_support_current_max_deviation_A"] = float(np.max(np.abs(current[interpolation_support] - config.current_A)))
    row["waveform_prefix_current_max_deviation_A"] = float(np.max(np.abs(current[:dependency_stop] - config.current_A)))
    if row["waveform_prefix_current_max_deviation_A"] > config.current_tolerance_A:
        reasons.append("WAVEFORM_PREFIX_CURRENT_EXCEEDS_REGISTERED_TOLERANCE")
    if reasons:
        return row, arrays
    q_grid = np.interp(grid, support_v, absolute_q[keep])
    native_grid = np.interp(grid, support_v, phase_q[keep] / 1000)
    current_grid = np.interp(grid, support_v, current[keep])
    row["window_capacity_span_Ah"] = float(np.ptp(q_grid))
    if row["window_capacity_span_Ah"] < config.minimum_window_charge_Ah:
        reasons.append("REGISTERED_WINDOW_CHARGE_SPAN_INSUFFICIENT")
        return row, arrays
    arrays.update(q_absolute=q_grid, q_native_Ah=native_grid,
                  raw_viq=np.column_stack((grid, current_grid, q_grid)))
    row["curve_valid"] = True
    return row, arrays


def view_definitions():
    return [asdict(view) for view in VIEWS]
