"""Executable mathematical and causal checks for the fixed physics module."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from physics_features import DEFAULT_CONFIG, GLOBAL_COLUMNS, extract_prefix_features


def run_checks() -> dict:
    voltage = np.linspace(2.8, 3.3, 64)
    x = (voltage - 2.8) / 0.5
    reference = 0.15 + 0.18 * (1 - x) + 0.025 * np.sin(2 * np.pi * x)
    derivative = np.gradient(savgol_filter(reference, 5, 2), voltage, edge_order=2)
    cycles = np.arange(1, 101)
    q = np.tile(reference, (100, 1))
    true_a = np.zeros(100)
    true_b = np.zeros(100)
    true_a[10:] = 0.002 * cycles[10:] / 100
    true_b[10:] = 0.003 * cycles[10:] / 100
    q += true_a[:, None] + true_b[:, None] * derivative
    extracted = extract_prefix_features(q, np.ones(100, bool), cycles, voltage)
    checks = {}
    assert extracted["global_features"].shape == (100, 20)
    assert extracted["local_features"].shape == (100, 8, 26)
    assert extracted["local_residual"].shape == (100, 64)
    assert np.array_equal(extracted["reference_cycles"], np.arange(1, 11))
    checks["registered_shapes_and_reference"] = True
    recovered = extracted["global_features"][:, :2]
    coefficient_error = float(np.max(np.abs(recovered - np.column_stack((true_a, true_b)))))
    assert coefficient_error < 1e-10
    checks["known_a_Ah_b_V_max_error"] = coefficient_error
    reconstruction = float(np.nanmax(extracted["diagnostics"]["reconstruction_max_absolute_error_Ah"]))
    assert reconstruction < 1e-12
    checks["reconstruction_max_absolute_error_Ah"] = reconstruction
    weighted = float(np.nanmax(np.abs(extracted["diagnostics"]["weighted_normalized_projection"])))
    assert weighted < 1e-8
    checks["weighted_normalized_projection_max"] = weighted

    contaminated_q = q.copy()
    contaminated_q[-1, 35] += 0.1
    contaminated = extract_prefix_features(contaminated_q, np.ones(100, bool), cycles, voltage)
    basis = np.column_stack((np.ones(64), derivative))
    response = contaminated_q[-1] - reference
    ols = np.linalg.lstsq(basis, response, rcond=None)[0]
    robust = contaminated["global_features"][-1, :2]
    truth = np.array([true_a[-1], true_b[-1]])
    robust_error = float(np.linalg.norm(robust - truth))
    ols_error = float(np.linalg.norm(ols - truth))
    assert robust_error < ols_error * 0.1
    assert np.min(contaminated["fit_weights"][-1]) < 0.01
    assert np.max(np.abs(contaminated["diagnostics"]["weighted_normalized_projection"][-1])) < 1e-8
    unweighted = float(np.max(np.abs(contaminated["diagnostics"]["unweighted_projection"][-1])))
    assert unweighted > 1e-5
    checks["outlier_robust_coefficient_error"] = robust_error
    checks["outlier_OLS_control_coefficient_error"] = ols_error
    checks["outlier_unweighted_projection_Ah"] = unweighted

    future_q = np.vstack((q, np.full((40, 64), 1e20)))
    extended = extract_prefix_features(future_q, np.ones(140, bool), np.arange(1, 141), voltage)
    for key in ("global_features", "local_features", "local_residual", "reference_q", "anchor_global_features"):
        assert np.array_equal(extracted[key], extended[key], equal_nan=True)
    checks["future_suffix_perturbation_exact_invariance"] = True
    truncated = extract_prefix_features(q[:10], np.ones(10, bool), cycles[:10], voltage)
    assert np.array_equal(truncated["global_features"], extracted["global_features"][:10], equal_nan=True)
    checks["K10_reference_and_dynamics_causality"] = True

    sparse_cycles = np.r_[np.arange(1, 11), 20, 40, 100]
    sparse_q = reference[None, :] + sparse_cycles[:, None] * 0.0001
    sparse = extract_prefix_features(sparse_q, np.ones(len(sparse_cycles), bool), sparse_cycles, voltage)
    slope_index = GLOBAL_COLUMNS.index("a_Ah_robust_slope_per_cycle")
    sparse_slope = float(sparse["global_features"][-1, slope_index])
    assert abs(sparse_slope - 0.0001) < 1e-12
    assert np.isnan(sparse["anchor_observed_cycles"][1])
    checks["gapped_actual_cycle_slope_Ah_per_cycle"] = sparse_slope
    checks["missing_K50_anchor_mask_without_future_borrowing"] = True

    bad_q = q.copy()
    bad_q[50, 7] = np.nan
    source_valid = np.ones(100, bool)
    source_valid[51] = False
    bad = extract_prefix_features(bad_q, source_valid, cycles, voltage)
    assert bad["failure_code"][50] == 2 and bad["failure_code"][51] == 1
    assert not bad["global_mask"][[50, 51]].any()
    assert not bad["local_mask"][[50, 51]].any()
    checks["nonfinite_and_source_failure_masks"] = True
    short_reference = np.ones(100, bool)
    short_reference[:10] = False
    short_reference[:4] = True
    missing_reference = extract_prefix_features(q, short_reference, cycles, voltage)
    assert not missing_reference["decomposition_valid"].any()
    assert not missing_reference["diagnostics"]["reference_valid"]
    checks["four_reference_curves_produce_missing_decomposition"] = True

    for title, degenerate_reference in (
        ("constant", np.full(64, 0.2)),
        ("linear_constant_derivative", 0.3 - 0.2 * x),
    ):
        invalid = extract_prefix_features(
            np.tile(degenerate_reference, (100, 1)), np.ones(100, bool), cycles, voltage
        )
        assert not invalid["decomposition_valid"].any()
        assert np.all(invalid["failure_code"] == 4)
        checks[title + "_derivative_identifiability_mask"] = True
    permutation = np.random.default_rng(42).permutation(100)
    reordered = extract_prefix_features(q[permutation], np.ones(100, bool), cycles[permutation], voltage)
    assert np.array_equal(reordered["global_features"], extracted["global_features"], equal_nan=True)
    assert np.array_equal(reordered["input_row_index"], np.argsort(permutation))
    checks["physical_id_ordering_and_explicit_row_mapping"] = True
    try:
        duplicate_cycles = cycles.copy()
        duplicate_cycles[-1] = 99
        extract_prefix_features(q, np.ones(100, bool), duplicate_cycles, voltage)
    except ValueError:
        checks["ambiguous_duplicate_physical_id_rejected"] = True
    else:
        raise AssertionError("Duplicate physical cycles were accepted")
    assert np.allclose(extracted["q_zero_at_3p3V"][:, -1], 0)
    assert not extracted["diagnostics"]["ica_dva_certified"]
    assert not extracted["diagnostics"]["resistance_polarization_certified"]
    checks["explicit_shape_zero_and_optional_physics_missing"] = True
    module = Path(__file__).with_name("physics_features.py")
    return {
        "status": "PASS", "scope": "synthetic mathematical and causal properties only",
        "real_source_QA": "pending caller integration with original HDF5 phase/axis certification",
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "checks": checks,
        "config": extracted["diagnostics"]["config"],
        "IRLS_orthogonality": "Final IRLS weights define the primary inner product. Unweighted projection is recorded. OLS is a mathematical test control only.",
        "global_columns": extracted["global_columns"],
        "local_columns": extracted["local_columns"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(result["checks"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
