"""Fixed, prefix-only loaded Q(V) response features for the MATR contract.

Q must retain the instrument discharge-counter origin. The caller certifies the
phase, current, units and physical cycle mapping before setting curve_valid.
No fitted population statistic or future observation is used in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class PhysicsConfig:
    prefix_max_cycle: int = 100
    reference_max_cycle: int = 10
    reference_min_curves: int = 5
    grid_size: int = 64
    voltage_min: float = 2.8
    voltage_max: float = 3.3
    bins: int = 8
    smooth_window: int = 5
    smooth_degree: int = 2
    huber_delta: float = 1.345
    irls_max_iterations: int = 20
    coefficient_tolerance: float = 1e-10
    noise_floor_Ah: float = 1e-6
    derivative_floor_Ah_per_V: float = 1e-8
    max_scaled_basis_condition: float = 1e6
    dynamic_observations: int = 10
    anchor_max_age_cycles: int = 4


DEFAULT_CONFIG = PhysicsConfig()
DYNAMIC_COLUMNS = (
    "causal_median", "transient", "causal_MAD", "robust_slope_per_cycle",
    "last_slope_per_cycle", "slope_change_per_cycle_squared",
    "signed_direction", "direction_consistency", "support_count",
)
LOCAL_BASE_COLUMNS = (
    "residual_signed_mean_Ah", "residual_rms_Ah",
    "residual_squared_integral_Ah2_V", "absolute_residual_centroid_V",
    "absolute_residual_peak_V", "availability_fraction",
    "high_frequency_noise_proxy_Ah", "delta_q_signed_mean_Ah",
)
GLOBAL_COLUMNS = ("a_Ah", "b_V") + tuple(
    f"{channel}_{stat}" for channel in ("a_Ah", "b_V") for stat in DYNAMIC_COLUMNS
)
LOCAL_COLUMNS = LOCAL_BASE_COLUMNS + tuple(
    f"{channel}_{stat}"
    for channel in ("residual_signed_mean_Ah", "residual_rms_Ah")
    for stat in DYNAMIC_COLUMNS
)
FAILURE_REASONS = {
    0: "valid", 1: "source_curve_not_valid", 2: "nonfinite_grid_point",
    3: "insufficient_reference_curves_within_original_cycles_1_to_10",
    4: "unidentifiable_reference_derivative", 5: "ill_conditioned_weighted_basis",
}


def _mad(values: np.ndarray) -> float:
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _dynamics(values: np.ndarray, cycles: np.ndarray, count: int) -> np.ndarray:
    """Causal rolling statistics with genuine cycle-distance denominators."""
    output = np.full((len(values), len(DYNAMIC_COLUMNS)), np.nan)
    observed = []
    previous_slope = np.nan
    previous_slope_cycle = np.nan
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        observed.append(index)
        selected = np.asarray(observed[-count:], dtype=int)
        x = cycles[selected]
        y = values[selected]
        median = float(np.median(y))
        output[index, [0, 1, 2, 8]] = [median, value - median, _mad(y), len(y)]
        if len(y) < 2:
            continue
        differences = np.diff(y) / np.diff(x)
        signed = float(np.mean(np.sign(differences)))
        output[index, [4, 6, 7]] = [differences[-1], signed, abs(signed)]
        if len(y) < 3:
            continue
        left, right = np.triu_indices(len(y), 1)
        slope = float(np.median((y[right] - y[left]) / (x[right] - x[left])))
        output[index, 3] = slope
        if np.isfinite(previous_slope):
            output[index, 5] = (slope - previous_slope) / (
                cycles[index] - previous_slope_cycle
            )
        previous_slope = slope
        previous_slope_cycle = cycles[index]
    return output


def _robust_fit(
    response: np.ndarray, basis: np.ndarray, config: PhysicsConfig
) -> dict:
    """Huber IRLS. Returned weights are exactly those in the final solve."""
    weights = np.ones(len(response))
    previous = np.full(basis.shape[1], np.inf)
    condition = np.inf
    coefficients = np.zeros(basis.shape[1])
    iterations = 0
    converged = False
    for iteration in range(config.irls_max_iterations):
        weighted_basis = basis * np.sqrt(weights)[:, None]
        condition = float(np.linalg.cond(weighted_basis))
        if not np.isfinite(condition) or condition > config.max_scaled_basis_condition:
            return {"valid": False, "condition": condition}
        coefficients = np.linalg.lstsq(
            weighted_basis, response * np.sqrt(weights), rcond=None
        )[0]
        residual = response - basis @ coefficients
        iterations = iteration + 1
        converged = bool(np.max(np.abs(coefficients - previous)) <= config.coefficient_tolerance)
        if converged or iterations == config.irls_max_iterations:
            break
        previous = coefficients.copy()
        scale = max(_mad(residual), config.noise_floor_Ah)
        standardized = np.abs(residual) / (config.huber_delta * scale)
        weights = 1.0 / np.maximum(standardized, 1.0)
    residual = response - basis @ coefficients
    scale = max(_mad(residual), config.noise_floor_Ah)
    residual_norm = max(
        float(np.sqrt(np.sum(weights * residual**2))),
        config.noise_floor_Ah * float(np.sqrt(np.sum(weights))),
    )
    weighted_projection = (basis.T @ (weights * residual)) / (
        np.sqrt(np.sum(weights[:, None] * basis**2, axis=0)) * residual_norm
    )
    return {
        "valid": True, "coefficients": coefficients, "residual": residual,
        "weights": weights.copy(), "condition": condition,
        "robust_scale_Ah": scale, "iterations": iterations, "converged": converged,
        "weighted_normalized_projection": weighted_projection,
        "unweighted_projection": basis.T @ residual / len(residual),
    }


def extract_prefix_features(
    q_absolute: np.ndarray,
    curve_valid: np.ndarray,
    actualcycleids: np.ndarray,
    vgrid: np.ndarray,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> dict:
    """Return masked arrays for every supplied physical prefix row.

    Rows after original cycle 100 are discarded before reading their curves or
    computing statistics. Retained rows are sorted by actual cycle ID. Invalid
    feature values are NaN with an identically shaped boolean mask. The returned
    input_row_index makes ordering explicit. No cell is dropped for missing data.

    Main array shapes are [N,20] global, [N,64] local residual and [N,8,26]
    local features. N is the count of provided rows at original cycles 1..100.
    A fixed 100-row tensor can include missing cycles as NaN with curve_valid=False.
    """
    q = np.asarray(q_absolute, dtype=np.float64)
    valid = np.asarray(curve_valid, dtype=bool)
    cycles = np.asarray(actualcycleids, dtype=np.float64)
    voltage = np.asarray(vgrid, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != config.grid_size:
        raise ValueError("q_absolute must have shape [rows, 64]")
    if valid.shape != (len(q),) or cycles.shape != (len(q),):
        raise ValueError("curve_valid and actualcycleids must match the curve rows")
    expected_grid = np.linspace(config.voltage_min, config.voltage_max, config.grid_size)
    if voltage.shape != expected_grid.shape or not np.allclose(voltage, expected_grid, atol=1e-12, rtol=0):
        raise ValueError("vgrid must be the registered increasing 64-point 2.8-3.3 V grid")
    selected = np.flatnonzero((cycles >= 1) & (cycles <= config.prefix_max_cycle))
    selected = selected[np.argsort(cycles[selected], kind="stable")]
    q, valid, cycles = q[selected], valid[selected], cycles[selected]
    if np.any(cycles != np.floor(cycles)) or len(np.unique(cycles)) != len(cycles):
        raise ValueError("Physical prefix cycle IDs must be unique positive integers")
    size = len(q)
    complete = np.isfinite(q).all(axis=1)
    qualified = valid & complete
    reference_rows = np.flatnonzero(qualified & (cycles <= config.reference_max_cycle))
    global_features = np.full((size, len(GLOBAL_COLUMNS)), np.nan)
    local_features = np.full((size, config.bins, len(LOCAL_COLUMNS)), np.nan)
    residuals = np.full_like(q, np.nan)
    delta_q = np.full_like(q, np.nan)
    weights = np.full_like(q, np.nan)
    weighted_projection = np.full((size, 2), np.nan)
    unweighted_projection = np.full((size, 2), np.nan)
    conditions = np.full(size, np.nan)
    scales = np.full(size, np.nan)
    iterations = np.zeros(size, dtype=np.int16)
    converged = np.zeros(size, dtype=bool)
    decomposition_valid = np.zeros(size, dtype=bool)
    reasons = np.where(valid, 2, 1).astype(np.int16)
    reasons[qualified] = 3
    reference_q = np.full(config.grid_size, np.nan)
    derivative = np.full(config.grid_size, np.nan)
    derivative_rms = np.nan
    basis_condition = np.nan
    enough_reference = len(reference_rows) >= config.reference_min_curves
    bin_edges = np.linspace(config.voltage_min, config.voltage_max, config.bins + 1)
    bin_index = np.minimum(np.searchsorted(bin_edges, voltage, side="right") - 1, config.bins - 1)
    reference_valid = False
    if enough_reference:
        reference_q = np.median(q[reference_rows], axis=0)
        smooth_reference = savgol_filter(reference_q, config.smooth_window, config.smooth_degree)
        derivative = np.gradient(smooth_reference, voltage, edge_order=2)
        derivative_rms = float(np.sqrt(np.mean(derivative**2)))
        if derivative_rms >= config.derivative_floor_Ah_per_V:
            basis = np.column_stack((np.ones(config.grid_size), derivative / derivative_rms))
            basis_condition = float(np.linalg.cond(basis))
            reference_valid = np.isfinite(basis_condition) and basis_condition <= config.max_scaled_basis_condition
        reasons[qualified] = 4
        if reference_valid:
            for index in np.flatnonzero(qualified):
                response = q[index] - reference_q
                fitted = _robust_fit(response, basis, config)
                conditions[index] = fitted["condition"]
                if not fitted["valid"]:
                    reasons[index] = 5
                    continue
                a, scaled_b = fitted["coefficients"]
                global_features[index, :2] = [a, scaled_b / derivative_rms]
                delta_q[index] = response
                residuals[index] = fitted["residual"]
                weights[index] = fitted["weights"]
                weighted_projection[index] = fitted["weighted_normalized_projection"]
                unweighted_projection[index] = fitted["unweighted_projection"]
                scales[index] = fitted["robust_scale_Ah"]
                iterations[index] = fitted["iterations"]
                converged[index] = fitted["converged"]
                decomposition_valid[index] = True
                reasons[index] = 0
                smooth_q = savgol_filter(q[index], config.smooth_window, config.smooth_degree)
                high_frequency = q[index] - smooth_q
                for local_bin in range(config.bins):
                    points = bin_index == local_bin
                    r, v = residuals[index, points], voltage[points]
                    amplitude = np.abs(r)
                    centroid = np.nan
                    peak = np.nan
                    if float(np.max(amplitude)) > config.noise_floor_Ah:
                        centroid = float(np.sum(amplitude * v) / np.sum(amplitude))
                        peak = float(v[np.argmax(amplitude)])
                    local_features[index, local_bin, :8] = [
                        np.mean(r), np.sqrt(np.mean(r**2)), np.trapz(r**2, v),
                        centroid, peak, 1.0,
                        max(_mad(high_frequency[points]), config.noise_floor_Ah),
                        np.mean(response[points]),
                    ]
    for channel in range(2):
        start = 2 + channel * len(DYNAMIC_COLUMNS)
        global_features[:, start:start + len(DYNAMIC_COLUMNS)] = _dynamics(
            global_features[:, channel], cycles, config.dynamic_observations
        )
    for local_bin in range(config.bins):
        for channel in range(2):
            start = len(LOCAL_BASE_COLUMNS) + channel * len(DYNAMIC_COLUMNS)
            local_features[:, local_bin, start:start + len(DYNAMIC_COLUMNS)] = _dynamics(
                local_features[:, local_bin, channel], cycles, config.dynamic_observations
            )
    anchor_cycles = np.full(3, np.nan)
    anchor_global = np.full((3, len(GLOBAL_COLUMNS)), np.nan)
    anchor_local = np.full((3, config.bins, len(LOCAL_COLUMNS)), np.nan)
    anchor_support = np.zeros(3, dtype=np.int16)
    for anchor_index, anchor in enumerate((10, 50, 100)):
        rows = np.flatnonzero(decomposition_valid & (cycles <= anchor) & (cycles >= anchor - config.anchor_max_age_cycles))
        anchor_support[anchor_index] = len(rows)
        if len(rows):
            last = rows[-1]
            anchor_cycles[anchor_index] = cycles[last]
            anchor_global[anchor_index] = global_features[last]
            anchor_local[anchor_index] = local_features[last]
    q_zero = q - q[:, -1:]
    q_zero[~qualified] = np.nan
    reconstruction_error = np.full(size, np.nan)
    if reference_valid:
        rows = np.flatnonzero(decomposition_valid)
        reconstructed = global_features[rows, 0, None] + global_features[rows, 1, None] * derivative + residuals[rows]
        reconstruction_error[rows] = np.max(np.abs(delta_q[rows] - reconstructed), axis=1)
    return {
        "original_cycle_id": cycles.astype(np.int64), "input_row_index": selected,
        "global_features": global_features, "global_mask": np.isfinite(global_features),
        "local_residual": residuals, "local_mask": np.isfinite(residuals),
        "local_features": local_features, "local_feature_mask": np.isfinite(local_features),
        "reference_q": reference_q, "reference_cycles": cycles[reference_rows].astype(np.int64),
        "reference_derivative_Ah_per_V": derivative,
        "q_absolute": q.copy(), "q_zero_at_3p3V": q_zero, "delta_q": delta_q,
        "decomposition_valid": decomposition_valid, "failure_code": reasons,
        "fit_weights": weights, "bin_edges_V": bin_edges, "bin_index": bin_index,
        "anchor_query_cycles": np.asarray([10, 50, 100]),
        "anchor_observed_cycles": anchor_cycles, "anchor_support": anchor_support,
        "anchor_global_features": anchor_global, "anchor_global_mask": np.isfinite(anchor_global),
        "anchor_local_features": anchor_local, "anchor_local_mask": np.isfinite(anchor_local),
        "global_columns": list(GLOBAL_COLUMNS), "local_columns": list(LOCAL_COLUMNS),
        "diagnostics": {
            "config": asdict(config), "reference_valid": bool(reference_valid),
            "reference_scaled_basis_condition": basis_condition,
            "reference_derivative_rms_Ah_per_V": derivative_rms,
            "weighted_normalized_projection": weighted_projection,
            "unweighted_projection": unweighted_projection,
            "weighted_scaled_basis_condition": conditions, "robust_scale_Ah": scales,
            "irls_iterations": iterations, "irls_converged": converged,
            "reconstruction_max_absolute_error_Ah": reconstruction_error,
            "failure_reasons": FAILURE_REASONS,
            "orthogonality": "IRLS final-weight inner product, residual orthogonal to constant and normalized reference derivative",
            "unweighted_projection_basis": "[1, derivative / derivative_rms], mean projection in Ah, recorded without enforcing zero",
            "reference": "pointwise median of all qualified original cycles 1..10, at least 5",
            "derivative": "Savitzky-Golay reference smoothing, window 5 degree 2, then numpy.gradient on physical voltage",
            "b_sign": "positive b multiplies positive reference dQ/dV in dQ=a+b*dQref/dV+r",
            "shape_zero": "instrument Q(V) minus instrument Q(3.3V), discharge-counter sign retained",
            "dynamic_window": "last 10 valid observations, all differences divided by actual original-cycle distances",
            "anchors": "latest valid original cycle in [K-4,K], no future borrowing, query K=10/50/100",
            "energy": "integral of residual squared with voltage, Ah^2 V, loaded shape statistic",
            "noise": "MAD of Q minus SG(5,2) within each bin, floored at 1e-6Ah, mixed signal/interpolation noise proxy",
            "ica_dva_certified": False, "resistance_polarization_certified": False,
        },
    }
