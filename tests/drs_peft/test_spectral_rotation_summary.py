"""Check scale separation, physical-cell rollups, and the locked screen gate."""

import numpy as np
import pytest

from scripts.drs_peft.summarize_spectral_rotation import (
    score_episode,
    summarize_results,
)


def test_output_and_parameter_matched_forwards_are_scored_separately():
    base = np.ones((1, 1))
    candidate, control = 0.6 * base, 0.7 * base
    pair = ("ROTATION_TRUE", "ROTATION_CONST")
    _, _, contrasts = score_episode(
        {"fold": 0, "seed": 42, "K": 3, "draw": 0},
        query_ids=["cell"],
        grid=[101],
        observed=base.astype(bool),
        targets=np.zeros_like(base),
        baseline=base,
        predictions={pair[0]: candidate, pair[1]: control},
        anchor_baseline=base,
        anchor_predictions={pair[0]: 0 * base, pair[1]: 0.75 * base},
        matched_pairs=[pair],
        parameter_matched={pair: (0.65 * base, 0.68 * base, 3.0)},
    )
    scored = next(row for row in contrasts if row["window"] == "all")
    assert scored["candidate_eta"] == 0.25
    assert scored["reference_eta"] == 1.0
    assert scored["raw_candidate_minus_reference_mae"] == pytest.approx(-0.1)
    assert scored["matched_candidate_minus_reference_mae"] == pytest.approx(0.2)
    assert scored["parameter_matched_candidate_minus_reference_mae"] == pytest.approx(
        -0.03
    )
    assert scored["common_weight_frobenius"] == 3.0
    no_observations = next(row for row in contrasts if row["window"] == "1001_plus")
    assert (
        no_observations["n_obs"] == 0
        and no_observations["matched_candidate_mae"] is None
    )


def _registered_screen(omit_parameter=False):
    base = np.ones((2, 1))
    candidate = "ROTATION_TRUE"
    control = "ROTATION_CONST"
    pair = candidate, control
    all_rows = ([], [], [])
    for fold in range(3):
        for draw in range(3):
            left = np.array([[1.05 if draw == 0 else 0.55], [1.01]])
            right = base * 1.02
            matched = (
                {}
                if omit_parameter and (fold, draw) == (0, 0)
                else {pair: (left, right, 2.0)}
            )
            tables = score_episode(
                {"fold": fold, "seed": 42, "K": 3, "draw": draw},
                query_ids=[f"{fold}a", f"{fold}b"],
                grid=[101],
                observed=base.astype(bool),
                targets=np.zeros_like(base),
                baseline=base,
                predictions={candidate: left, control: right},
                anchor_baseline=base,
                anchor_predictions={candidate: 0.9 * base, control: right},
                parameter_matched=matched,
            )
            for all_table, case_table in zip(all_rows, tables, strict=True):
                all_table.extend(case_table)
    return summarize_results(
        *all_rows,
        required_references=[control],
        parameter_references=[control],
        expected_cells_per_fold=2,
    )


def test_gate_requires_parameter_coverage_and_preserves_physical_negative_transfer():
    passed = _registered_screen()
    assert (
        passed["decision"]["direction"] == "ELIGIBLE_FOR_SEPARATE_SOURCE_CONFIRMATION"
    )
    scores = passed["method_scores"]
    whole = scores[scores.method.eq("ROTATION_TRUE") & scores.window.eq("all")].iloc[0]
    assert whole.query_cells == 6 and whole.cell_draws == 18
    assert whole.negative_cells == 3 and whole.negative_cell_draws == 12
    assert whole.negative_cell_rate == 0.5
    failed = _registered_screen(omit_parameter=True)
    assert failed["decision"]["direction"] == "STOP_INCOMPLETE_COMPARISON"
