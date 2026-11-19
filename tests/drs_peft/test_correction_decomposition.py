"""Verify correction arithmetic, sign diagnoses, and physical-cell weighting."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.drs_peft.correction_decomposition import decompose_corrections, target_query


def test_unequal_observation_counts_give_cells_equal_weight():
    grid = np.array([[101, 101, 101], [101, 102, 1001]])
    observed = np.array([[True, False, False], [True, True, True]])
    base = np.array([[2.0, np.nan, np.nan], [3.0, 3.0, 3.0]])
    adapted = np.array([[1.0, np.nan, np.nan], [4.0, 4.0, 4.0]])
    target = np.zeros_like(base)
    cells, episodes = decompose_corrections(
        ["a", "b"], grid, observed, base, adapted, target, metadata={"K": 2}
    )
    complete = episodes[0]
    assert complete["K"] == 2
    assert complete["n_obs"] == 4
    assert complete["n_cells"] == 2
    assert complete["dot_ed"] == 0.5
    assert complete["energy_dd"] == 1.0
    assert complete["cross_term"] == 1.0
    assert complete["norm_term"] == 1.0
    assert complete["delta_mse"] == complete["delta_mse_direct"] == 2.0
    assert complete["mae_base"] == complete["mae_eta"] == 2.5
    classifications = {(r["cell_uid"], r["window"]): r["classification"] for r in cells}
    assert classifications[("a", "all")] == "beneficial"
    assert classifications[("b", "all")] == "wrong_direction"
    assert classifications[("a", "1001_plus")] == "no_observations"
    assert episodes[-1]["n_obs"] == episodes[-1]["n_cells"] == 1


def test_sign_cases_and_oracle_stay_distinct_from_fitted_scale():
    ids = ["overshoot", "neutral", "orthogonal", "no_op", "beneficial"]
    base = np.array([[1.0], [1.0], [0.0], [1.0], [4.0]])
    adapted = np.array([[-2.0], [-1.0], [1.0], [1.0], [2.0]])
    observed = np.ones_like(base, dtype=bool)
    targets = np.zeros_like(base)
    cells, _ = decompose_corrections(ids, [101], observed, base, adapted, targets)
    cases = [r for r in cells if r["window"] == "all"]
    assert [r["classification"] for r in cases] == [
        "overshoot", "neutral_update", "orthogonal_update", "no_op", "beneficial"
    ]
    assert cases[0]["dot_sign_exact"] == -1
    assert cases[0]["delta_mse_sign_exact"] == 1
    assert cases[0]["oracle_eta"] == pytest.approx(1 / 3)
    assert cases[-1]["oracle_eta"] == 2.0
    assert cases[-1]["oracle_eta_clipped_0_1"] == 1.0
    assert cases[-1]["mae_eta"] == cases[-1]["mae_adapted"] == 2.0
    scaled, _ = decompose_corrections(ids, [101], observed, base, adapted, targets, eta=0)
    assert scaled[-5]["mae_adapted"] == 2.0
    assert scaled[-5]["mae_eta"] == 4.0


def test_empty_windows_and_invalid_observed_grid():
    observed = np.zeros((1, 2), dtype=bool)
    values = np.zeros((1, 2))
    cells, episodes = decompose_corrections(["missing"], [101, 101], observed, values, values, values)
    assert all(r["classification"] == "no_observations" and r["oracle_eta"] is None for r in cells + episodes)
    with pytest.raises(ValueError, match="increasing"):
        decompose_corrections(["duplicate"], [101, 101], ~observed, values, values, values)


def test_target_reader_requires_source_artifact_hashes(tmp_path, monkeypatch):
    from scripts.drs_peft import transfer_scope_data

    def unexpected_read(*args, **kwargs):
        pytest.fail("Read target outcomes before authenticating the source gate")

    monkeypatch.setattr(transfer_scope_data, "load_future_rows", unexpected_read)
    args = SimpleNamespace(output=tmp_path)
    with pytest.raises(RuntimeError, match="Source-direction completion"):
        target_query(args, None, None, None)

    (tmp_path / "SOURCE_DIRECTION_COMPLETE.json").write_text(
        '{"status":"COMPLETE","scope":"SOURCE_DEVELOPMENT_FIRST",'
        '"target_rows_read":0,"source_cells_sha256":"expected"}', encoding="utf-8"
    )
    (tmp_path / "SOURCE_CELL.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Source direction artifact changed"):
        target_query(args, None, None, None)
