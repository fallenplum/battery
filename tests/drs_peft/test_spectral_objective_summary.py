"""Check Stage C weighting and failure gates with staggered observations."""

import numpy as np
import pytest

from scripts.drs_peft.summarize_spectral_objective import (
    CANDIDATE,
    DETERMINISTIC_METHODS,
    METHODS,
    PARAMETER_REFERENCES,
    IncompleteComparison,
    score_episode,
    summarize_results,
)


def synthetic_ledgers(*, unstable_seed=False):
    baseline = np.ones((2, 4))
    mask = np.array([[True, True, False, True], [True, False, False, False]])
    comparisons = [
        (CANDIDATE, name) for name in METHODS if name not in {"K0", CANDIDATE}
    ]
    combined = ([], [], [])
    for fold in range(3):
        for seed in (42, 43, 44):
            for draw in range(3):
                true = np.array([[0.8, 0.6, 0.99, 0.4], [0.7, 1.0, 1.0, 1.0]])
                if unstable_seed and seed == 44:
                    true = 1.1 * baseline
                predictions = {name: 0.9 * baseline for name in METHODS}
                predictions["K0"] = baseline
                predictions[CANDIDATE] = true
                meta = {
                    method: {
                        "is_deterministic": method in DETERMINISTIC_METHODS,
                        "reused_prediction": method in DETERMINISTIC_METHODS
                        and seed != 42,
                        "source_prediction_hash": "a" * 64,
                    }
                    for method in METHODS
                }
                ids = [f"cell{fold}a", f"cell{fold}b"]
                rows = score_episode(
                    {"fold": fold, "training_seed": seed, "K": 3, "draw": draw},
                    query_ids=ids,
                    grid=[101, 150, 220, 650],
                    observed=mask,
                    targets=np.zeros_like(baseline),
                    baseline=baseline,
                    predictions=predictions,
                    anchor_baseline=baseline,
                    anchor_predictions=predictions,
                    query_protocols={uid: uid for uid in ids},
                    method_metadata=meta,
                    matched_pairs=comparisons,
                    parameter_matched={
                        (CANDIDATE, method): (true, predictions[method], 0.1)
                        for method in PARAMETER_REFERENCES
                    },
                )
                for combined_rows, episode_rows in zip(combined, rows, strict=True):
                    combined_rows.extend(episode_rows)
    return combined


def test_physical_window_contributions_recover_primary_mae():
    tables, decision = summarize_results(
        *synthetic_ledgers(),
        expected_cells_per_fold=2,
        expected_protocol_count=6,
        selected_rates={0: {CANDIDATE: {"alpha": 0.0001}}},
    )
    assert decision["direction"] == "ELIGIBLE_FOR_SEPARATE_SOURCE_CONFIRMATION"
    assert decision["labels"] == [
        "ELIGIBLE_FOR_SEPARATE_SOURCE_CONFIRMATION",
        "RATE_BOUNDARY",
    ]
    assert (
        decision["physical_query_cells"] == 6
        and decision["outer_mae_contribution_supported"]
    )
    assert (
        decision["deterministic_prediction_reuse"]["registered_deterministic_positions"]
        == 81
    )
    assert (
        decision["deterministic_prediction_reuse"]["different_prediction_positions"]
        == 27
    )
    method = tables["METHOD_SCORES.csv"]
    main = method.loc[method.method.eq(CANDIDATE) & method.window.eq("all")].iloc[0]
    assert main.mae == pytest.approx(0.65)
    contrib = tables["WINDOW_CONTRIBUTION_SUMMARY.csv"]
    baseline = contrib.loc[contrib.method.eq("K0")].set_index("window")
    assert baseline.loc[
        "101_200", "candidate_minus_method_mae_contribution"
    ] == pytest.approx(-0.25)
    assert baseline.loc[
        "501_1000", "candidate_minus_method_mae_contribution"
    ] == pytest.approx(-0.10)
    assert baseline.loc["1001_plus", "candidate_minus_method_mae_contribution"] == 0
    assert np.isnan(baseline.loc["1001_plus", "window_macro_mae"])
    assert baseline["candidate_minus_method_mae_contribution"].sum() == pytest.approx(
        -0.35
    )
    assert max(decision["window_reconstruction_error"].values()) < 1e-12


def test_seed_instability_and_missing_pairs_block_confirmation():
    cells, episodes, pairs = synthetic_ledgers(unstable_seed=True)
    _, decision = summarize_results(
        cells,
        episodes,
        pairs,
        expected_cells_per_fold=2,
        expected_protocol_count=6,
    )
    assert not decision["eligible_for_separate_source_confirmation"]
    assert "INCONCLUSIVE_SEED_INSTABILITY" in decision["labels"]
    assert not decision["outer_mae_contribution_supported"]
    pairs = [
        row
        for row in pairs
        if not (
            row["reference_method"] == "ROTATION_CONST_MA"
            and row["training_seed"] == 43
        )
    ]
    with pytest.raises(IncompleteComparison, match="parameter-matched roster"):
        summarize_results(
            cells, episodes, pairs, expected_cells_per_fold=2, expected_protocol_count=6
        )
