"""Stage C scoring for the locked spectral-objective source experiment."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.e123_protocol import WINDOWS
from scripts.drs_peft.summarize_spectral_rotation import (
    anchor_function_radius,
)
from scripts.drs_peft.summarize_spectral_rotation import (
    score_episode as score_rotation_episode,
)

CANDIDATE = "ROTATION_TRUE_MA"
MATCHED_RATE = "ROTATION_TRUE_MM_AT_MA_RATE"
SAME_CORE = "ROTATION_TRUE_MA_SAME_CORE_SHUFFLED"
DETERMINISTIC_METHODS = ("K0", "STATIC_CORE", "CAL2")
SIMPLE_REFERENCES = ("K0", "LORA_SMALL", "CAL2", "STATIC_CORE")
STRUCTURAL_REFERENCES = ("GAIN_MA", "ROTATION_CONST_MA", "ROTATION_SHUFFLED_MA")
RAW_REFERENCES = SIMPLE_REFERENCES + STRUCTURAL_REFERENCES
PARAMETER_REFERENCES = ("ROTATION_CONST_MA", "ROTATION_SHUFFLED_MA")
METHODS = (
    *SIMPLE_REFERENCES,
    "ROTATION_TRUE_MM",
    CANDIDATE,
    MATCHED_RATE,
    *STRUCTURAL_REFERENCES,
    "GAIN_MM",
    "ROTATION_CONST_MM",
    "ROTATION_SHUFFLED_MM",
    SAME_CORE,
)
CASE_AXES = ["fold", "training_seed", "K", "draw"]
CELL_AXES = ["fold", "K", "cell_uid", "protocol_uid", "window"]
WINDOW_NAMES = tuple(name for name, _, _ in WINDOWS)
PHYSICAL_METRICS = ("mae", "mse", "dot_ed", "energy_dd", "correction_rms")
TOLERANCE = 1e-12


class IncompleteComparison(ValueError):
    """A Stage C ledger lacks an exact registered comparison."""


def method_objective(method):
    """Return the registered meta-Query objective of a method."""
    if method == MATCHED_RATE or method.endswith("_MM"):
        return "MM"
    if method == SAME_CORE or method.endswith("_MA"):
        return "MA"
    return "BASELINE"


def score_episode(
    metadata,
    *,
    query_ids,
    grid,
    observed,
    targets,
    baseline,
    predictions,
    anchor_baseline,
    anchor_predictions,
    query_protocols,
    method_metadata=None,
    costs=None,
    matched_pairs=(),
    parameter_matched=None,
):
    """Score actual episode predictions and both label-free scale diagnostics."""
    method_metadata = method_metadata or {}
    if set(method_metadata) - set(predictions):
        raise ValueError("Method metadata does not match this episode")
    cells, episodes, pairs = score_rotation_episode(
        metadata,
        query_ids=query_ids,
        grid=grid,
        observed=observed,
        targets=targets,
        baseline=baseline,
        predictions=predictions,
        anchor_baseline=anchor_baseline,
        anchor_predictions=anchor_predictions,
        query_protocols=query_protocols,
        costs=costs,
        matched_pairs=matched_pairs,
        parameter_matched=parameter_matched,
    )
    query_rms = {
        method: anchor_function_radius(prediction, baseline)
        for method, prediction in predictions.items()
    }
    facts = {
        method: {
            "objective": method_objective(method),
            **method_metadata.get(method, {}),
        }
        for method in predictions
    }
    cells = [facts[row["method"]] | row for row in cells]
    episodes = [
        facts[row["method"]] | row | {"query_rms_full_grid": query_rms[row["method"]]}
        for row in episodes
    ]
    pairs = [
        row
        | {
            "objective": facts[row["method"]]["objective"],
            "reference_objective": facts[row["reference_method"]]["objective"],
        }
        for row in pairs
    ]
    return cells, episodes, pairs


def _expect(condition, message):
    if not condition:
        raise IncompleteComparison(message)


def _roster(
    cells, episodes, *, folds, seeds, draws, methods, cells_per_fold, protocol_count
):
    keys = CASE_AXES + ["method", "objective"]
    _expect(not cells.empty and not episodes.empty, "No Stage C scores")
    _expect(
        not cells.duplicated(keys + ["cell_uid", "window"]).any(),
        "Duplicate cell scores",
    )
    _expect(
        not episodes.duplicated(keys + ["window"]).any(), "Duplicate episode scores"
    )
    _expect(cells["K"].eq(3).all(), "Stage C requires K=3")
    _expect(set(cells.fold) == set(folds), "Missing or unexpected source fold")
    _expect(
        set(cells.training_seed) == set(seeds), "Missing or unexpected training seed"
    )
    _expect(set(cells.draw) == set(draws), "Missing or unexpected Support draw")
    _expect(set(cells.method) == set(methods), "Missing or unexpected method")
    _expect(
        cells.objective.eq(cells.method.map(method_objective)).all(),
        "Method objective changes among pairing positions",
    )
    _expect(set(cells.window) == set(WINDOW_NAMES), "Source forecast windows changed")
    n_expected = (
        len(seeds) * len(draws) * len(methods) * cells_per_fold * len(WINDOW_NAMES)
    )
    for _, part in cells.groupby("fold"):
        _expect(
            part.cell_uid.nunique() == cells_per_fold and len(part) == n_expected,
            "A source fold lacks registered cell/draw/seed/method observations",
        )
    registry = cells[["fold", "cell_uid", "protocol_uid"]].drop_duplicates()
    _expect(
        registry.cell_uid.is_unique and not registry.protocol_uid.isna().any(),
        "A held-out cell appears in multiple folds or protocols",
    )
    _expect(
        registry.groupby("protocol_uid").fold.nunique().eq(1).all(),
        "A held-out protocol appears in multiple folds",
    )
    _expect(
        protocol_count is None or registry.protocol_uid.nunique() == protocol_count,
        "The held-out protocol count changed",
    )
    variation = cells.groupby(["cell_uid", "window"]).n_obs.nunique(dropna=False)
    _expect(
        variation.eq(1).all(),
        "A cell's observation mask differs across methods, draws or seeds",
    )
    _expect(
        cells.n_obs.ge(0).all() and cells.n_obs.mod(1).eq(0).all(),
        "Invalid observation count",
    )
    count = cells.drop_duplicates(["cell_uid", "window"]).pivot(
        index="cell_uid", columns="window", values="n_obs"
    )
    _expect(
        count["all"].gt(0).all()
        and count["all"].eq(count[list(WINDOW_NAMES[1:])].sum(axis=1)).all(),
        "Whole-window observations must be the union of disjoint windows",
    )
    valid = cells.n_obs.gt(0)
    _expect(
        np.isfinite(cells.loc[valid, ["mae", "mse"]].to_numpy(dtype=float)).all()
        and cells.loc[~valid, ["mae", "mse"]].isna().to_numpy().all(),
        "Observed versus empty-window metrics are inconsistent",
    )
    axis = keys + ["window"]
    _expect(
        set(cells[axis].itertuples(index=False, name=None))
        == set(episodes[axis].itertuples(index=False, name=None)),
        "The episode and cell ledgers differ",
    )


def _reused_predictions(
    cells, episodes, deterministic_methods, selection_seed, required
):
    """Authenticate deterministic scoring broadcasts and count physical fits once."""
    columns = {"is_deterministic", "reused_prediction", "source_prediction_hash"}
    if not columns <= set(episodes):
        _expect(not required, "Stage C requires deterministic reuse lineage")
        return {
            "registered_deterministic_positions": None,
            "different_prediction_positions": None,
        }
    full = episodes.loc[episodes.window.eq("all")].copy()
    for name in ("is_deterministic", "reused_prediction"):
        _expect(full[name].isin([True, False]).all(), f"Invalid reuse flag: {name}")
    _expect(
        full["is_deterministic"].eq(full.method.isin(deterministic_methods)).all(),
        "Deterministic method flags changed",
    )
    deterministic = full.loc[full.is_deterministic.astype(bool)]
    _expect(
        deterministic.source_prediction_hash.astype(str)
        .str.fullmatch("[0-9a-f]{64}")
        .all(),
        "A deterministic broadcast lacks authenticated prediction provenance",
    )
    groups = deterministic.groupby(["fold", "K", "draw", "method", "objective"])
    for _, cases in groups:
        _expect(
            cases.source_prediction_hash.nunique() == 1,
            "Deterministic across-seed predictions differ",
        )
        _expect(
            cases.loc[
                cases.training_seed.ne(selection_seed), "reused_prediction"
            ].all(),
            "A copied deterministic prediction is counted as a new repeat",
        )
    cell_keys = ["fold", "K", "draw", "method", "objective", "cell_uid", "window"]
    source_cells = cells.loc[cells.method.isin(deterministic_methods)]
    _expect(
        source_cells.groupby(cell_keys)[["mae", "mse"]]
        .nunique(dropna=False)
        .eq(1)
        .to_numpy()
        .all(),
        "Deterministic prediction reuse changes a physical cell score",
    )
    return {
        "registered_deterministic_positions": len(deterministic),
        "different_prediction_positions": int(groups.ngroups),
        "deterministic_reused_positions": int(deterministic.reused_prediction.sum()),
        "distinct_prediction_sha256_values": int(
            deterministic.source_prediction_hash.nunique()
        ),
    }


def _reduce_draws_and_seeds(cells):
    usable = [name for name in PHYSICAL_METRICS if name in cells]
    by = CELL_AXES + ["method", "objective"]
    aggregation = {name: (name, "mean") for name in usable} | {
        "n_obs": ("n_obs", "first")
    }
    seeded = cells.groupby(by + ["training_seed"], dropna=False, as_index=False).agg(
        **aggregation
    )
    physical = seeded.groupby(by, dropna=False, as_index=False).agg(**aggregation)
    return seeded, physical


def _paired(frame, candidate, seeded=False):
    axes = CELL_AXES + (["training_seed"] if seeded else [])
    reference = frame.loc[frame.method.eq(candidate), axes + ["mae"]].rename(
        columns={"mae": "candidate_mae"}
    )
    baseline = frame.loc[frame.method.eq("K0"), axes + ["mae"]].rename(
        columns={"mae": "k0_mae"}
    )
    paired = frame.merge(reference, on=axes, validate="many_to_one").merge(
        baseline, on=axes, validate="many_to_one"
    )
    _expect(len(paired) == len(frame), "Lack of a matched physical cell")
    paired["candidate_minus_method_mae"] = paired.candidate_mae - paired.mae
    paired["method_minus_k0_mae"] = paired.mae - paired.k0_mae
    paired["candidate_harms_cell_vs_method"] = paired.candidate_minus_method_mae.gt(0)
    paired["negative_transfer_vs_k0"] = paired.method_minus_k0_mae.gt(0)
    return paired


def _rollup(frame, axes):
    measures = ["mae", "mse", "candidate_minus_method_mae", "method_minus_k0_mae"]
    output = frame.groupby(
        axes + ["K", "method", "objective", "window"], as_index=False, dropna=False
    ).agg(
        **{name: (name, "mean") for name in measures},
        physical_cells=("mae", "count"),
        physical_observations=("n_obs", "sum"),
        harmed_cells_vs_method=("candidate_harms_cell_vs_method", "sum"),
        negative_transfer_cells_vs_k0=("negative_transfer_vs_k0", "sum"),
    )
    output["negative_transfer_rate_vs_k0"] = output.negative_transfer_cells_vs_k0.div(
        output.physical_cells.where(output.physical_cells.gt(0))
    )
    return output


def _window_contributions(physical):
    axes = ["fold", "K", "cell_uid", "protocol_uid", "method", "objective"]
    fields = ["mae", "candidate_minus_method_mae", "method_minus_k0_mae"]
    full = physical.loc[physical.window.eq("all"), axes + ["n_obs", *fields]].rename(
        columns={name: f"full_{name}" for name in ["n_obs", *fields]}
    )
    windows = physical.loc[physical.window.ne("all")].merge(
        full, on=axes, validate="many_to_one"
    )
    windows["fraction_of_cell_observations"] = windows.n_obs / windows.full_n_obs
    residual = {}
    for field in fields:
        name = f"{field}_contribution"
        windows[name] = windows[field].fillna(0) * windows.fraction_of_cell_observations
        reconstructed = windows.groupby(axes, dropna=False)[name].sum()
        difference = reconstructed - full.set_index(axes)[f"full_{field}"]
        maximum = float(difference.abs().max())
        _expect(
            math.isfinite(maximum) and maximum <= TOLERANCE,
            f"Window contributions fail to recover the physical-cell {field}",
        )
        residual[field] = maximum
    aggregate = windows.groupby(
        ["K", "method", "objective", "window"], as_index=False, dropna=False
    ).agg(
        **{
            f"{field}_contribution": (f"{field}_contribution", "mean")
            for field in fields
        },
        window_macro_mae=("mae", "mean"),
        observed_cells=("mae", "count"),
        observed_points=("n_obs", "sum"),
        all_physical_cells=("cell_uid", "size"),
    )
    keys = ["K", "method", "objective"]
    for field in fields:
        rebuilt = aggregate.groupby(keys)[f"{field}_contribution"].sum()
        direct = full.groupby(keys)[f"full_{field}"].mean()
        maximum = float((rebuilt - direct).abs().max())
        _expect(
            math.isfinite(maximum) and maximum <= TOLERANCE,
            f"Window contributions fail to recover aggregate {field}",
        )
        residual[f"aggregate_{field}"] = maximum
    return windows, aggregate, residual


def _direction_check(
    reference, overall, fold_means, seed_means, *, margin, require_seeds
):
    """Record every threshold so failed gate combinations remain visible."""
    folds = {str(key): float(value) for key, value in fold_means.items()}
    seeds = {str(key): float(value) for key, value in seed_means.items()}
    wins_folds = sum(value < 0 for value in folds.values())
    wins_seeds = sum(value < 0 for value in seeds.values())
    improves = overall < 0 and (margin is None or overall <= -margin)
    minimum = 2
    return {
        "reference_method": reference,
        "mean_candidate_minus_reference_mae": float(overall),
        "margin": margin,
        "improvement_meets_mean_requirement": bool(improves),
        "fold_deltas": folds,
        "winning_folds": wins_folds,
        "folds_required": minimum,
        "seed_deltas": seeds,
        "winning_seeds": wins_seeds,
        "all_seeds_improve": wins_seeds == len(seeds) and bool(seeds),
        "passed": bool(
            improves
            and wins_folds >= minimum
            and (not require_seeds or wins_seeds == len(seeds) and bool(seeds))
        ),
    }


def _raw_check(physical, seeded, reference, *, margin=None, require_seeds=True):
    cells = physical.loc[physical.window.eq("all") & physical.method.eq(reference)]
    repeats = seeded.loc[seeded.window.eq("all") & seeded.method.eq(reference)]
    _expect(len(cells) > 0 and len(repeats) > 0, f"Missing raw comparator {reference}")
    col = "candidate_minus_method_mae"
    _expect(
        cells[col].notna().all() and repeats[col].notna().all(),
        f"Incomplete raw MAE comparator {reference}",
    )
    return _direction_check(
        reference,
        cells[col].mean(),
        cells.groupby("fold")[col].mean(),
        repeats.groupby("training_seed")[col].mean(),
        margin=margin,
        require_seeds=require_seeds,
    )


def _scale_tables(
    pairs, episodes, *, folds, seeds, draws, candidate, parameter_references
):
    _expect(not pairs.empty, "Stage C requires parameter- and output-matched ledgers")
    axis = CASE_AXES + [
        "method",
        "objective",
        "reference_method",
        "reference_objective",
        "window",
    ]
    _expect(not pairs.duplicated(axis).any(), "Duplicate scale-matched comparisons")
    _expect(
        set(pairs.fold) <= set(folds)
        and set(pairs.training_seed) <= set(seeds)
        and set(pairs.draw) <= set(draws),
        "Unknown scale-matched pairing position",
    )
    _expect(
        set(pairs.window) <= set(WINDOW_NAMES) and pairs.K.eq(3).all(),
        "Unknown scale scoring axis",
    )
    for method_field, objective_field in (
        ("method", "objective"),
        ("reference_method", "reference_objective"),
    ):
        _expect(
            pairs[objective_field].eq(pairs[method_field].map(method_objective)).all(),
            "Scale-matched objective differs from the method",
        )
    columns = [
        "raw_candidate_minus_reference_mae",
        "matched_candidate_minus_reference_mae",
        "parameter_matched_candidate_minus_reference_mae",
    ]
    matched = pairs.copy()
    expected = episodes.set_index(CASE_AXES + ["method", "window"])[
        ["n_obs", "n_cells", "mae"]
    ]
    for side, method_field in (
        ("candidate", "method"),
        ("reference", "reference_method"),
    ):
        lookup = pd.MultiIndex.from_frame(matched[CASE_AXES + [method_field, "window"]])
        original = expected.reindex(lookup)
        _expect(
            np.array_equal(matched.n_obs.to_numpy(), original.n_obs.to_numpy())
            and np.array_equal(matched.n_cells.to_numpy(), original.n_cells.to_numpy()),
            "The scale comparison changes Query membership or n_obs",
        )
        _expect(
            np.allclose(
                matched[f"raw_{side}_mae"].astype(float),
                original.mae.astype(float),
                atol=TOLERANCE,
                rtol=0,
                equal_nan=True,
            ),
            "Scale comparison carries a different raw score",
        )
    for name in columns:
        matched[name] = pd.to_numeric(matched[name])
    anchor = matched.anchor_match_resolved.astype(bool)
    for estimate, radius in (
        ("candidate_eta", "candidate_anchor_radius"),
        ("reference_eta", "reference_anchor_radius"),
    ):
        values = matched.loc[anchor, estimate]
        _expect(
            values.gt(0).all()
            and values.le(1).all()
            and np.allclose(
                values * matched.loc[anchor, radius],
                matched.loc[anchor, "common_anchor_radius"],
                atol=TOLERANCE,
                rtol=0,
            ),
            "A function-scale comparison is not a shared contraction",
        )

    raw_col, output_col, param_col = columns
    scores = matched.groupby(
        [
            "fold",
            "training_seed",
            "K",
            "method",
            "objective",
            "reference_method",
            "reference_objective",
            "window",
        ],
        as_index=False,
        dropna=False,
    ).agg(
        raw_delta_mae=(raw_col, "mean"),
        raw_evaluated_draws=(raw_col, "count"),
        output_matched_delta_mae=(output_col, "mean"),
        output_matched_draws=(output_col, "count"),
        parameter_matched_delta_mae=(param_col, "mean"),
        parameter_matched_draws=(param_col, "count"),
    )
    expected_cases = {
        (fold, seed, draw) for fold in folds for seed in seeds for draw in draws
    }
    gates = []
    for reference in parameter_references:
        full = matched.loc[
            matched.method.eq(candidate)
            & matched.reference_method.eq(reference)
            & matched.window.eq("all")
        ]
        registered = set(
            full[["fold", "training_seed", "draw"]].itertuples(index=False, name=None)
        )
        _expect(
            registered == expected_cases,
            f"Incomplete parameter-matched roster for {reference}",
        )
        valid = full.parameter_match_resolved.astype(bool) & full[param_col].notna()
        _expect(
            full.loc[valid, "common_weight_frobenius"].gt(0).all()
            and np.isfinite(full.loc[valid, param_col]).all(),
            "Invalid matched parameter score",
        )
        if not valid.all():
            gates.append(
                {
                    "reference_method": reference,
                    "passed": False,
                    "evaluated_positions": int(valid.sum()),
                    "expected_positions": len(expected_cases),
                    "reason": "UNRESOLVED_PARAMETER_NORM",
                    "mean_candidate_minus_reference_mae": None,
                }
            )
        else:
            gate = _direction_check(
                reference,
                full[param_col].mean(),
                full.groupby("fold")[param_col].mean(),
                full.groupby("training_seed")[param_col].mean(),
                margin=None,
                require_seeds=False,
            )
            gates.append(
                gate
                | {
                    "evaluated_positions": len(full),
                    "expected_positions": len(expected_cases),
                }
            )
    return matched, scores, gates


def _rate_boundary_rows(selected_rates, config):
    """Flag learning-rate bounds without requesting a larger search grid."""
    if selected_rates is None:
        return []
    core_rates = config.get("core_rates", (0.0001, 0.001, 0.01, 0.1))
    lora_rates = config.get("lora_rates", (1e-7, 1e-6, 1e-5, 3e-5))
    boundary = []
    for fold, methods in selected_rates.items():
        for method, selection in methods.items():
            field, grid = (
                ("lr", lora_rates) if method == "LORA_SMALL" else ("alpha", core_rates)
            )
            if field not in selection:
                continue
            value = float(selection[field])
            _expect(value in grid, f"Unregistered {field} for {method} fold {fold}")
            if value in {min(grid), max(grid)}:
                boundary.append(
                    {
                        "fold": int(fold),
                        "method": method,
                        "field": field,
                        "value": value,
                        "boundary": "minimum" if value == min(grid) else "maximum",
                    }
                )
    return boundary


def _costs(episodes, training_cost_rows):
    outer = episodes.loc[episodes.window.eq("all")].copy()
    columns = [
        *CASE_AXES,
        "method",
        "objective",
        "is_deterministic",
        "reused_prediction",
        "source_prediction_hash",
        "adapt_seconds",
        "adapt_updates",
        "inference_seconds",
        "adapt_peak_allocated_bytes",
        "target_parameters",
        "source_mapper_parameters",
        "source_mapper_effective_constant_parameters",
        "basis_storage_bytes",
        "support_mse_before",
        "support_mse_after",
        "support_full_mse_before",
        "support_full_mse_after",
        "weight_frobenius_rms",
        "anchor_radius",
        "query_rms_full_grid",
        "correction_rms",
    ]
    outer = outer[[name for name in columns if name in outer]].assign(
        scope="outer_episode"
    )
    if "reused_prediction" in outer:
        for name in ("adapt_seconds", "adapt_updates", "inference_seconds"):
            if name in outer:
                outer[f"actual_{name}"] = outer[name].where(
                    ~outer.reused_prediction.eq(True), 0
                )
    training = (
        pd.DataFrame(training_cost_rows)
        if training_cost_rows is not None
        else pd.DataFrame()
    )
    if training.empty:
        return outer
    if "scope" not in training:
        training = training.assign(scope="source_mapper_fit")
    return pd.concat((training, outer), ignore_index=True)


def _decision(raw, matched_rate, same_core, parameter, scale, boundary, margin):
    gates = {item["reference_method"]: item for item in raw}
    failed = {name for name, item in gates.items() if not item["passed"]}
    tags = []
    if "K0" in failed or scale["all_query_positions_near_zero"]:
        tags.append("STOP_NO_ADAPTATION_GAIN")
    if failed & (set(SIMPLE_REFERENCES) - {"K0"}):
        tags.append("STOP_SIMPLE_BASELINE")
    if not failed & set(SIMPLE_REFERENCES) and failed & set(STRUCTURAL_REFERENCES):
        tags.append("OPTIMIZATION_GAIN_WITHOUT_STRUCTURE_EVIDENCE")
    if any(0 < item["winning_seeds"] < len(item["seed_deltas"]) for item in raw):
        tags.append("INCONCLUSIVE_SEED_INSTABILITY")
    if any(
        item["mean_candidate_minus_reference_mae"] < 0 and item["winning_folds"] < 2
        for item in raw
    ):
        tags.append("INCONCLUSIVE_FOLD_INSTABILITY")
    if not same_core["passed"] or any(not item["passed"] for item in parameter):
        tags.append("STOP_STRUCTURE_EVIDENCE")
    eligible = (
        not failed
        and same_core["passed"]
        and all(item["passed"] for item in parameter)
        and not scale["all_query_positions_near_zero"]
    )
    if scale["all_anchor_positions_near_zero"]:
        tags.append("NEAR_ZERO_ON_ANCHORS")
    if scale["all_query_positions_near_zero"]:
        tags.append("NEAR_ZERO_ON_QUERY")
    if boundary:
        tags.append("RATE_BOUNDARY")
    direction = "ELIGIBLE_FOR_SEPARATE_SOURCE_CONFIRMATION" if eligible else tags[0]
    return {
        "direction": direction,
        "labels": ([direction] if eligible else []) + tags,
        "eligible_for_separate_source_confirmation": bool(eligible),
        "primary_candidate": CANDIDATE,
        "mae_margin": margin,
        "raw_gates": raw,
        "same_core_gate": same_core,
        "parameter_match_gates": parameter,
        "outer_mae_contribution_supported": matched_rate["passed"],
        "matched_rate_check": matched_rate,
        "scale": scale,
        "rate_boundaries": boundary,
        "claim_scope": "fixed_reference_source_development_screen",
        "target_future_rows_loaded": 0,
    }


def summarize_results(
    cell_rows,
    episode_rows,
    pair_rows,
    *,
    config=None,
    methods=METHODS,
    selected_rates=None,
    training_cost_rows=None,
    expected_folds=(0, 1, 2),
    expected_draws=(0, 1, 2),
    expected_cells_per_fold=20,
    expected_protocol_count=39,
    require_reuse_lineage=True,
):
    """Build Stage C cell-macro scores and apply every registered gate."""
    config = config or {}
    folds, draws = tuple(expected_folds), tuple(expected_draws)
    seeds = tuple(config.get("meta_training_seeds", (42, 43, 44)))
    margin = float(config.get("mae_margin", 1e-5))
    radius = float(config.get("near_zero_radius", 1e-5))
    _expect(
        math.isfinite(margin) and margin > 0 and math.isfinite(radius) and radius >= 0,
        "Invalid registered decision threshold",
    )
    cells, episodes, pairs = (
        pd.DataFrame(rows) for rows in (cell_rows, episode_rows, pair_rows)
    )
    _roster(
        cells,
        episodes,
        folds=folds,
        seeds=seeds,
        draws=draws,
        methods=methods,
        cells_per_fold=expected_cells_per_fold,
        protocol_count=expected_protocol_count,
    )
    reused = _reused_predictions(
        cells,
        episodes,
        DETERMINISTIC_METHODS,
        config.get("selection_seed", 42),
        require_reuse_lineage,
    )
    seeded, physical = _reduce_draws_and_seeds(cells)
    seeded = _paired(seeded, CANDIDATE, seeded=True)
    physical = _paired(physical, CANDIDATE)
    contributions, contribution_summary, reconstruction_error = _window_contributions(
        physical
    )
    paired_scales, scale_summary, param_gates = _scale_tables(
        pairs,
        episodes,
        folds=folds,
        seeds=seeds,
        draws=draws,
        candidate=CANDIDATE,
        parameter_references=PARAMETER_REFERENCES,
    )
    raw_gates = [
        _raw_check(physical, seeded, method, margin=margin) for method in RAW_REFERENCES
    ]
    same_core = _raw_check(physical, seeded, SAME_CORE, require_seeds=False)
    matched_rate = _raw_check(physical, seeded, MATCHED_RATE)
    full = episodes.loc[episodes.window.eq("all") & episodes.method.eq(CANDIDATE)]
    _expect(
        np.isfinite(
            full[["anchor_radius", "query_rms_full_grid", "correction_rms"]].to_numpy(
                dtype=float
            )
        ).all(),
        "Main candidate lacks authenticated correction magnitudes",
    )
    scale = {
        "near_zero_threshold": radius,
        "all_anchor_positions_near_zero": bool(full.anchor_radius.le(radius).all()),
        "all_query_positions_near_zero": bool(
            full.query_rms_full_grid.le(radius).all()
        ),
        "near_zero_anchor_positions": int(full.anchor_radius.le(radius).sum()),
        "near_zero_query_positions": int(full.query_rms_full_grid.le(radius).sum()),
        "candidate_positions": len(full),
        "maximum_anchor_rms": float(full.anchor_radius.max()),
        "maximum_query_rms_full_grid": float(full.query_rms_full_grid.max()),
        "maximum_query_rms_observed": float(full.correction_rms.max()),
    }
    boundary = _rate_boundary_rows(selected_rates, config)
    decision = _decision(
        raw_gates, matched_rate, same_core, param_gates, scale, boundary, margin
    )
    decision |= {
        "physical_query_cells": int(cells.cell_uid.nunique()),
        "physical_query_protocols": int(cells.protocol_uid.nunique()),
        "paired_prediction_positions": len(folds) * len(seeds) * len(draws),
        "deterministic_prediction_reuse": reused,
        "window_reconstruction_error": reconstruction_error,
    }
    decision["gain_mm_vs_candidate"] = _raw_check(physical, seeded, "GAIN_MM")
    protocol = _rollup(physical, ["fold", "protocol_uid"])
    protocol_macro = protocol.groupby(
        ["K", "method", "objective", "window"], as_index=False
    ).agg(
        protocol_macro_mae=("mae", "mean"),
        protocol_macro_candidate_minus_method_mae=(
            "candidate_minus_method_mae",
            "mean",
        ),
        observed_protocols=("mae", "count"),
    )
    method_scores = _rollup(physical, []).merge(
        protocol_macro, on=["K", "method", "objective", "window"]
    )
    cases = ["fold", "protocol_uid", "cell_uid", "K", "method", "objective", "window"]
    worst_cell = (
        physical.loc[physical.window.eq("all")]
        .sort_values(
            ["method", "candidate_minus_method_mae", "cell_uid"],
            ascending=[True, False, True],
        )
        .drop_duplicates(["method"])
    )
    worst_protocol = (
        protocol.loc[protocol.window.eq("all")]
        .sort_values(
            ["method", "candidate_minus_method_mae", "protocol_uid"],
            ascending=[True, False, True],
        )
        .drop_duplicates(["method"])
    )
    tables = {
        "CELL_SCORES.csv": cells,
        "EPISODE_SCORES.csv": episodes,
        "PAIRED_CELL_SEED_SCORES.csv": seeded,
        "CELL_MEAN_SCORES.csv": physical,
        "FOLD_SCORES.csv": _rollup(physical, ["fold"]),
        "SEED_SCORES.csv": _rollup(seeded, ["training_seed"]),
        "FOLD_SEED_SCORES.csv": _rollup(seeded, ["fold", "training_seed"]),
        "METHOD_SCORES.csv": method_scores,
        "PROTOCOL_SCORES.csv": protocol,
        "MATCHED_RATE_SCORES.csv": seeded.loc[seeded.method.eq(MATCHED_RATE)],
        "SCALE_MATCHED_EPISODES.csv": paired_scales,
        "SCALE_MATCHED_SUMMARY.csv": scale_summary,
        "WINDOW_CONTRIBUTIONS.csv": contributions,
        "WINDOW_CONTRIBUTION_SUMMARY.csv": contribution_summary,
        "WORST_CELLS.csv": worst_cell[
            cases + ["candidate_minus_method_mae", "method_minus_k0_mae"]
        ],
        "WORST_PROTOCOLS.csv": worst_protocol,
        "COSTS.csv": _costs(episodes, training_cost_rows),
    }
    return tables, decision


def write_summary(directory, cell_rows, episode_rows, pair_rows, **kwargs):
    """Write actual Stage C tables, or an explicit missing-comparison stop."""
    from scripts.drs_peft.data import file_hash, write_json
    from scripts.drs_peft.summarize_abba import atomic_csv

    directory = Path(directory)
    try:
        tables, decision = summarize_results(
            cell_rows, episode_rows, pair_rows, **kwargs
        )
    except IncompleteComparison as exc:
        receipt = {
            "status": "INCOMPLETE",
            "direction": "STOP_INCOMPLETE_COMPARISON",
            "labels": ["STOP_INCOMPLETE_COMPARISON"],
            "reason": str(exc),
            "files_sha256": {},
            "target_future_rows_loaded": 0,
        }
    else:
        hashes = {}
        for name, frame in tables.items():
            path = directory / name
            atomic_csv(path, frame)
            hashes[name] = file_hash(path)
        receipt = decision | {"status": "COMPLETE", "files_sha256": hashes}
    write_json(directory / "FINAL_DECISION.json", receipt)
    return receipt
