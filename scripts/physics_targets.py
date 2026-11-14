"""Observed future targets with explicit support, censoring and knee sensitivity.

The caller supplies an authenticated cycle coordinate and a comparability mask.
These label functions may read the observed suffix. Their outputs are labels and
support metadata, and must be kept outside inference-time feature extraction.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.ndimage import median_filter


TARGET_CONTRACT = {
    "version": "SRSR_PHYSICS_OBSERVED_TARGETS_V1_20261115",
    "rates": {"r200_500": [200, 500], "r500_1000": [500, 1000]},
    "endpoint_radius_cycles": 5,
    "minimum_endpoint_observations": 5,
    "eol_threshold": 0.8,
    "knee_median_windows_observations": [5, 11, 21],
    "knee_primary_median_window": 11,
    "knee_maximum_grid_points": 201,
    "knee_minimum_side_observations": 30,
    "knee_minimum_side_span_cycles": 30.0,
    "knee_minimum_side_fraction_of_cycle_span": 0.15,
    "knee_minimum_bic_gain": 10.0,
    "knee_minimum_relative_sse_improvement": 0.10,
    "knee_maximum_tau_range_fraction_of_cycle_span": 0.10,
    "linear_parameter_count": 2,
    "hinge_parameter_count_including_tau": 4,
}


def _optional_number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _prepare_series(cycle_number, soh, comparable=None):
    cycles = np.asarray(cycle_number, dtype=np.float64)
    values = np.asarray(soh, dtype=np.float64)
    if cycles.ndim != 1 or values.ndim != 1 or cycles.shape != values.shape:
        raise ValueError("cycle_number and soh must be equally sized one-dimensional arrays")
    mask = np.ones(len(cycles), dtype=bool)
    if comparable is not None:
        supplied_mask = np.asarray(comparable)
        if supplied_mask.shape != cycles.shape or not np.isin(supplied_mask, (False, True)).all():
            raise ValueError("comparable must be a binary mask matching cycle_number")
        mask = supplied_mask.astype(bool)
    finite = np.isfinite(cycles) & np.isfinite(values)
    valid = mask & finite & (cycles >= 0)
    x, y = cycles[valid], values[valid]
    support = {
        "input_observations": len(cycles),
        "valid_observations": len(x),
        "nonfinite_observations": int(np.count_nonzero(~finite)),
        "incomparable_observations": int(np.count_nonzero(~mask)),
        "negative_coordinate_observations": int(np.count_nonzero(finite & (cycles < 0))),
        "observed_start": float(x[0]) if len(x) else None,
        "observed_end": float(x[-1]) if len(x) else None,
        "supported": bool(len(x)),
        "reason": "AVAILABLE" if len(x) else "NO_COMPARABLE_OBSERVATIONS",
    }
    if len(x) > 1 and np.any(np.diff(x) <= 0):
        support.update(supported=False, reason="NONINCREASING_OR_DUPLICATE_CYCLE_COORDINATE")
    return x, y, support


def endpoint_rate_target(
    cycle_number,
    soh,
    start_cycle,
    end_cycle,
    comparable=None,
    *,
    endpoint_radius=5,
    minimum_endpoint_observations=5,
):
    """Rate from medians in fixed physical-coordinate endpoint neighborhoods."""
    parameters = (start_cycle, end_cycle, endpoint_radius, minimum_endpoint_observations)
    if not all(math.isfinite(v) for v in parameters) or end_cycle <= start_cycle or endpoint_radius < 0 or minimum_endpoint_observations < 1:
        raise ValueError("Rate endpoints, radius and minimum support are inconsistent")
    x, y, support = _prepare_series(cycle_number, soh, comparable)
    result = {
        "available": False,
        "value": None,
        "start_cycle": float(start_cycle),
        "end_cycle": float(end_cycle),
        "endpoint_radius_cycles": float(endpoint_radius),
        "minimum_endpoint_observations": int(minimum_endpoint_observations),
        "support": support,
        "reason": support["reason"],
        "endpoint_method": "median_of_comparable_observations_in_inclusive_cycle_window",
    }
    if not support["supported"]:
        return result
    endpoints = {}
    for name, center in (("start", start_cycle), ("end", end_cycle)):
        selected = np.abs(x - center) <= endpoint_radius
        count = int(np.count_nonzero(selected))
        endpoints[name] = {
            "observations": count,
            "median_soh": float(np.median(y[selected])) if count else None,
            "first_observed_cycle": float(x[selected][0]) if count else None,
            "last_observed_cycle": float(x[selected][-1]) if count else None,
        }
    result["endpoints"] = endpoints
    if endpoints["start"]["observations"] < minimum_endpoint_observations:
        result["reason"] = "START_ENDPOINT_WINDOW_INSUFFICIENT_SUPPORT"
        return result
    if endpoints["end"]["observations"] < minimum_endpoint_observations:
        result["reason"] = "END_ENDPOINT_WINDOW_INSUFFICIENT_SUPPORT"
        return result
    rate = (endpoints["start"]["median_soh"] - endpoints["end"]["median_soh"]) / (end_cycle - start_cycle)
    result.update(available=True, value=float(rate), reason="AVAILABLE")
    return result


def eol80_target(cycle_number, soh, comparable=None, *, official_extrapolated_label=None):
    """First observed valid SOH at or below 0.8, with observation-gap bounds."""
    x, y, support = _prepare_series(cycle_number, soh, comparable)
    result = {
        "available": support["supported"],
        "has_event": False,
        "event_cycle": None,
        "event_soh": None,
        "threshold": 0.8,
        "censoring": "unknown",
        "censor_cycle": None,
        "crossing_interval_lower": None,
        "crossing_interval_upper": None,
        "observed_end": support["observed_end"],
        "official_extrapolated_label": _optional_number(official_extrapolated_label),
        "official_label_used_for_event": False,
        "reason": support["reason"],
        "support": support,
    }
    if not support["supported"]:
        return result
    below = np.flatnonzero(y <= 0.8)
    if not len(below):
        result.update(censoring="right", censor_cycle=float(x[-1]), reason="NO_OBSERVED_THRESHOLD_CROSSING")
        return result
    index = int(below[0])
    result.update(
        has_event=True,
        event_cycle=float(x[index]),
        event_soh=float(y[index]),
        crossing_interval_upper=float(x[index]),
        reason="FIRST_VALID_OBSERVED_SOH_AT_OR_BELOW_0_8",
    )
    if index == 0:
        result.update(censoring="left", censor_cycle=float(x[index]), reason="FIRST_COMPARABLE_OBSERVATION_ALREADY_BELOW_THRESHOLD")
        return result
    result["crossing_interval_lower"] = float(x[index - 1])
    if x[index] - x[index - 1] > 1:
        result.update(censoring="interval", reason="FIRST_BELOW_THRESHOLD_AFTER_UNOBSERVED_CYCLE_GAP")
    else:
        result["censoring"] = "none"
    return result


def _hinge_sensitivity(cycles, values, candidates, median_window):
    """Compare one continuous hinge against a linear trajectory on the same data."""
    smoothed = median_filter(values, size=median_window, mode="nearest")
    origin, span = float(cycles[0]), float(cycles[-1] - cycles[0])
    coordinate = (cycles - origin) / span
    linear_design = np.column_stack((np.ones(len(cycles)), coordinate))
    linear_coefficients = np.linalg.lstsq(linear_design, smoothed, rcond=None)[0]
    linear_residual = smoothed - linear_design @ linear_coefficients
    linear_sse = float(linear_residual @ linear_residual)
    numerical_floor = np.finfo(np.float64).eps * max(1.0, float(smoothed @ smoothed))
    result = {
        "median_window_observations": median_window,
        "has_candidate": False,
        "candidate_cycle": None,
        "best_grid_cycle": None,
        "linear_sse": linear_sse,
        "hinge_sse": None,
        "relative_sse_improvement": None,
        "bic_gain": None,
        "pre_lossrate": None,
        "post_lossrate": None,
        "reason": "LINEAR_TRAJECTORY_AT_NUMERICAL_PRECISION",
    }
    if linear_sse <= numerical_floor:
        return result
    best = None
    for tau in candidates:
        hinge = np.maximum(coordinate - (tau - origin) / span, 0)
        design = np.column_stack((linear_design, hinge))
        coefficients = np.linalg.lstsq(design, smoothed, rcond=None)[0]
        residual = smoothed - design @ coefficients
        sse = float(residual @ residual)
        if best is None or sse < best[0]:
            best = (sse, float(tau), coefficients)
    sse, tau, coefficients = best
    count = len(cycles)
    linear_bic = count * math.log(max(linear_sse / count, np.finfo(float).tiny)) + 2 * math.log(count)
    hinge_bic = count * math.log(max(sse / count, np.finfo(float).tiny)) + 4 * math.log(count)
    bic_gain = float(linear_bic - hinge_bic)
    improvement = float((linear_sse - sse) / linear_sse)
    pre_lossrate = float(-coefficients[1] / span)
    post_lossrate = float(-(coefficients[1] + coefficients[2]) / span)
    checks = {
        "bic_gain_pass": bic_gain >= TARGET_CONTRACT["knee_minimum_bic_gain"],
        "sse_improvement_pass": improvement >= TARGET_CONTRACT["knee_minimum_relative_sse_improvement"],
        "accelerating_loss_pass": post_lossrate > pre_lossrate >= 0,
    }
    qualified = all(checks.values())
    result.update(
        has_candidate=qualified,
        candidate_cycle=tau if qualified else None,
        best_grid_cycle=tau,
        hinge_sse=sse,
        relative_sse_improvement=improvement,
        bic_gain=bic_gain,
        pre_lossrate=pre_lossrate,
        post_lossrate=post_lossrate,
        checks=checks,
        reason="QUALIFIED_CANDIDATE" if qualified else "HINGE_IMPROVEMENT_OR_ACCELERATION_CRITERION_FAILED",
    )
    return result


def knee_target(cycle_number, soh, comparable=None, *, after_cycle=100):
    """Stable future hinge candidate with explicit no-event and uncertainty states."""
    if not math.isfinite(after_cycle):
        raise ValueError("after_cycle must be a finite cycle coordinate")
    x, y, support = _prepare_series(cycle_number, soh, comparable)
    result = {
        "has_candidate": False,
        "stable": False,
        "available": False,
        "candidate_cycle": None,
        "knee_cycle": None,
        "knee_onset": None,
        "knee_onset_reason": "SEPARATE_ONSET_DEFINITION_NOT_REGISTERED",
        "support": support,
        "reason": support["reason"],
        "sensitivity": [],
        "candidate_after_cycle": float(after_cycle),
    }
    if not support["supported"]:
        return result
    span = float(x[-1] - x[0])
    minimum_side = 30
    minimum_side_span = max(30.0, 0.15 * span)
    result.update(observed_span_cycles=span, minimum_side_observations=minimum_side,
                  minimum_side_span_cycles=minimum_side_span)
    if len(x) < 2 * minimum_side:
        result["reason"] = "INSUFFICIENT_PRE_AND_POST_SIDE_OBSERVATIONS"
        return result
    candidates = x[minimum_side - 1:len(x) - minimum_side]
    candidates = candidates[(candidates > after_cycle) &
                            (candidates - x[0] >= minimum_side_span) &
                            (x[-1] - candidates >= minimum_side_span)]
    if not len(candidates):
        result["reason"] = "NO_SUPPORTED_FUTURE_BREAKPOINT_GRID"
        return result
    grid_indices = np.linspace(0, len(candidates) - 1, min(201, len(candidates))).round().astype(int)
    candidates = candidates[np.unique(grid_indices)]
    result["candidate_grid"] = candidates.tolist()
    sensitivity = [_hinge_sensitivity(x, y, candidates, window) for window in (5, 11, 21)]
    result["sensitivity"] = sensitivity
    primary = sensitivity[1]
    result.update(has_candidate=primary["has_candidate"], candidate_cycle=primary["candidate_cycle"])
    if not primary["has_candidate"]:
        result["reason"] = "NO_QUALIFIED_PRIMARY_HINGE_CANDIDATE"
        return result
    if not all(row["has_candidate"] for row in sensitivity):
        result["reason"] = "CANDIDATE_NOT_REPRODUCED_ACROSS_MEDIAN_WINDOWS"
        return result
    tau_values = [row["candidate_cycle"] for row in sensitivity]
    tau_range = float(max(tau_values) - min(tau_values))
    result["tau_range_cycles"] = tau_range
    result["maximum_tau_range_cycles"] = 0.10 * span
    if tau_range > 0.10 * span:
        result["reason"] = "BREAKPOINT_UNSTABLE_ACROSS_MEDIAN_WINDOWS"
        return result
    result.update(stable=True, available=True, knee_cycle=primary["candidate_cycle"], reason="STABLE_HINGE_CANDIDATE")
    return result


def compute_future_targets(cycle_number, soh, comparable=None, *, prefix_cycle=100, official_extrapolated_label=None):
    """Return the fixed D0 target bundle without fitting an inference-time model."""
    _, _, support = _prepare_series(cycle_number, soh, comparable)
    return {
        "contract": dict(TARGET_CONTRACT),
        "support": support,
        "observed_end": support["observed_end"],
        "official_extrapolated_label": _optional_number(official_extrapolated_label),
        "r200_500": endpoint_rate_target(cycle_number, soh, 200, 500, comparable),
        "r500_1000": endpoint_rate_target(cycle_number, soh, 500, 1000, comparable),
        "eol80": eol80_target(cycle_number, soh, comparable, official_extrapolated_label=official_extrapolated_label),
        "knee": knee_target(cycle_number, soh, comparable, after_cycle=prefix_cycle),
    }


def future_targets(cycle_number, soh, comparable=None, *, prefix_cycle=100, official_extrapolated_label=None):
    """Scalar-column interface for the D0 eligibility and coverage tables."""
    bundle = compute_future_targets(cycle_number, soh, comparable,
                                    prefix_cycle=prefix_cycle,
                                    official_extrapolated_label=official_extrapolated_label)
    eol, knee = bundle["eol80"], bundle["knee"]
    output = {
        "observed_end": bundle["observed_end"],
        "official_extrapolated_label": _optional_number(official_extrapolated_label),
        "eol_event": eol["has_event"] if eol["available"] else None,
        "eol_cycle": eol["event_cycle"],
        "eol_censoring": eol["censoring"],
        "eol_censor_cycle": eol["censor_cycle"],
        "eol_interval_lower": eol["crossing_interval_lower"],
        "eol_interval_upper": eol["crossing_interval_upper"],
        "eol_reason": eol["reason"],
        "knee_has_candidate": knee["has_candidate"],
        "knee_stable_candidate": knee["stable"],
        "knee_cycle": knee["knee_cycle"],
        "knee_candidate_cycle": knee["candidate_cycle"],
        "knee_reason": knee["reason"],
        "knee_sensitivity_json": json.dumps(knee["sensitivity"], ensure_ascii=False, allow_nan=False),
    }
    for name in ("r200_500", "r500_1000"):
        rate = bundle[name]
        output[name] = rate["value"]
        output[name + "_reason"] = rate["reason"]
        for endpoint in ("start", "end"):
            details = rate.get("endpoints", {}).get(endpoint, {})
            output[name + "_" + endpoint + "_observations"] = details.get("observations", 0)
    return output


def _self_check():
    # Legacy extrapolated life labels must never turn an unobserved event into an event.
    cycles = np.arange(1, 201)
    censored = eol80_target(cycles, 1 - 0.0005 * cycles, official_extrapolated_label=500)
    assert not censored["has_event"] and censored["censoring"] == "right"
    assert censored["observed_end"] == 200 and censored["official_extrapolated_label"] == 500
    cycles = np.arange(1, 1501)
    linear = 1 - 0.0001 * cycles
    rate = endpoint_rate_target(cycles, linear, 200, 500)
    assert rate["available"] and np.isclose(rate["value"], 0.0001)
    assert not knee_target(cycles, linear)["has_candidate"]
    hinge = 1 - 0.00005 * cycles - 0.0003 * np.maximum(cycles - 700, 0)
    candidate = knee_target(cycles, hinge)
    assert candidate["stable"] and abs(candidate["knee_cycle"] - 700) < 10
    return {"status": "PASS", "kind": "synthetic_contract_selfcheck", "checks": 4}


if __name__ == "__main__":
    print(json.dumps(_self_check(), ensure_ascii=False))
