"""Score paired source-only spectral adaptation without opening future data."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.correction_decomposition import decompose_corrections
from scripts.drs_peft.e123_protocol import WINDOWS

RAW_REFERENCES = (
    "K0",
    "CAL2",
    "LORA_SMALL",
    "STATIC_CORE",
    "DRS",
    "GAIN",
    "JOINT2R",
    "ROTATION_CONST",
    "ROTATION_SHUFFLED",
    "ROTATION_SAME_CORE_SHUFFLED",
)
PARAMETER_REFERENCES = ("ROTATION_CONST", "ROTATION_SHUFFLED")
COST_FIELDS = (
    "fit_seconds",
    "adaptation_seconds",
    "adaptation_time_seconds",
    "training_seconds",
    "forward_seconds",
    "runtime_seconds",
    "adapt_seconds",
    "inference_seconds",
    "optimizer_steps",
    "optimizer_updates",
    "adapt_updates",
    "trainable_parameters",
    "active_parameters",
    "adapter_parameters",
    "peak_memory_bytes",
    "peak_gpu_memory_bytes",
    "fit_peak_allocated_bytes",
    "fit_peak_reserved_bytes",
    "adapt_peak_allocated_bytes",
    "target_parameters",
    "source_mapper_parameters",
    "basis_storage_bytes",
    "source_fit_seconds",
    "source_peak_allocated_bytes",
    "source_mapping_updates",
    "source_inner_updates",
    "weight_frobenius_rms",
    "anchor_mean_abs_angle_rad",
    "anchor_mean_angle_sd_rad",
    "anchor_max_abs_angle_rad",
    "model_forward_calls",
    "parameter_weight_frobenius",
    "support_mse_before",
    "support_mse_after",
    "support_mae_before",
    "support_mae_after",
    "support_mse",
    "support_mae",
    "support_sampled_mse",
    "support_full_mse",
)
WORK_FIELDS = (
    "fit_seconds",
    "adaptation_seconds",
    "adaptation_time_seconds",
    "training_seconds",
    "forward_seconds",
    "runtime_seconds",
    "adapt_seconds",
    "inference_seconds",
    "optimizer_steps",
    "optimizer_updates",
    "adapt_updates",
    "model_forward_calls",
)
STOP_EXPLANATIONS = {
    "STOP_INCOMPLETE_COMPARISON": "至少一个必要对照缺少完整的登记配对。",
    "STOP_NEAR_ZERO_CORRECTION": "所有候选任务的源锚输出变化均在登记的近零范围内。",
    "STOP_SIMPLE_BASELINE": "候选未通过未适配参考、CAL2 或小步 LoRA 的全窗 MAE 要求。",
    "STOP_PHYSICAL_CONDITIONING": "候选未通过同核心描述置乱或独立训练置乱的全窗 MAE 要求。",
    "STOP_ROTATION_CONTROL": "候选未通过至少一个静态、增益、联合空间或常数角对照。",
    "STOP_PARAMETER_MATCH_DIRECTION": "参数等幅后，相对常数角或独立训练置乱的改善方向不足。",
    "ELIGIBLE_FOR_SEPARATE_SOURCE_CONFIRMATION": "本轮源筛选通过，仍需独立队列确认。",
}


def anchor_function_radius(prediction, baseline):
    """RMS function displacement on the shared, label-free source anchor."""
    original = np.asarray(baseline, dtype=np.float64)
    adapted = np.asarray(prediction, dtype=np.float64)
    if original.shape != adapted.shape or not original.size:
        raise ValueError("Anchor predictions must share a nonempty registered grid")
    if not np.isfinite(original).all() or not np.isfinite(adapted).all():
        raise ValueError("Anchor prediction contains nonfinite values")
    return float(np.sqrt(np.mean(np.square(adapted - original))))


def _annotate_scores(rows, method, radius):
    for row in rows:
        if row["n_obs"]:
            residual_norm = math.sqrt(row["mse_base"] * row["energy_dd"])
            alignment = -row["dot_ed"] / residual_norm if residual_norm else None
            if alignment is not None:
                alignment = min(1.0, max(-1.0, alignment))
            row.update(
                mae=row["mae_eta"],
                mse=row["mse_eta"],
                mae_gain=-row["delta_mae"],
                negative_transfer=row["delta_mae"] > 0,
                correction_rms=row["eta"] * math.sqrt(row["energy_dd"]),
                alignment_with_negative_residual=alignment,
            )
        else:
            row.update(
                dict.fromkeys(
                    (
                        "mae",
                        "mse",
                        "mae_gain",
                        "negative_transfer",
                        "correction_rms",
                        "alignment_with_negative_residual",
                    )
                )
            )
        row.update(
            method=method,
            anchor_radius=radius,
            scaled_anchor_radius=row["eta"] * radius if radius is not None else None,
        )
    return rows


def _prefixed_metrics(row, prefix):
    fields = (
        "mae",
        "mse",
        "dot_ed",
        "cross_term",
        "norm_term",
        "delta_mse",
        "correction_rms",
        "alignment_with_negative_residual",
    )
    return {
        f"{prefix}_{field}": row[field] if row is not None else None for field in fields
    }


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
    costs=None,
    matched_pairs=(),
    parameter_matched=None,
    radius_floor=1e-12,
    query_protocols=None,
):
    """Score a shared registered episode and contract pairs to identical anchor RMS.

    ``predictions`` and ``anchor_predictions`` map the same method names to
    arrays. Include the frozen reference as a method when it must be reported.
    Each matched pair is ``(candidate, reference)``. It is resolved only when
    both measured anchor corrections exceed ``radius_floor``. Matching only
    contracts predictions toward the baseline, and cannot use Query outcomes.
    ``parameter_matched`` supplies pairs of separately re-evaluated predictions
    and their common weight Frobenius norm. These are scored separately from
    the anchor-based contraction. Return three lists of dictionaries for cells,
    method/windows, and contrast/windows.
    """
    if not predictions or set(predictions) != set(anchor_predictions):
        raise ValueError(
            "Every registered method needs both Query and anchor predictions"
        )
    if not math.isfinite(radius_floor) or radius_floor < 0:
        raise ValueError("Anchor radius floor must be finite and nonnegative")
    costs = costs or {}
    parameter_matched = parameter_matched or {}
    if set(costs) - set(predictions):
        raise ValueError("Cost fields must identify registered methods")
    radii = {
        method: anchor_function_radius(value, anchor_baseline)
        for method, value in anchor_predictions.items()
    }
    cell_rows, episode_rows, pair_rows = [], [], []
    evaluated = {}
    for method, prediction in predictions.items():
        cells, episodes = decompose_corrections(
            query_ids, grid, observed, baseline, prediction, targets, metadata=metadata
        )
        cells = _annotate_scores(cells, method, radii[method])
        episodes = _annotate_scores(episodes, method, radii[method])
        if query_protocols is not None:
            for row in cells:
                row["protocol_uid"] = query_protocols[row["cell_uid"]]
        cell_rows.extend(cells)
        episode_rows.extend(costs.get(method, {}) | row for row in episodes)
        evaluated[(method, 1.0)] = {row["window"]: row for row in episodes}

    for method, reference in dict.fromkeys((*matched_pairs, *parameter_matched)):
        if (
            method == reference
            or method not in predictions
            or reference not in predictions
        ):
            raise ValueError("Matching requires two distinct registered methods")
        radius = min(radii[method], radii[reference])
        match = radius > radius_floor
        weights = (
            [radius / radii[side] for side in (method, reference)]
            if match
            else [None, None]
        )
        for side, weight in zip((method, reference), weights, strict=True):
            if weight is not None and (side, weight) not in evaluated:
                _, episodes = decompose_corrections(
                    query_ids,
                    grid,
                    observed,
                    baseline,
                    predictions[side],
                    targets,
                    metadata=metadata,
                    eta=weight,
                )
                evaluated[(side, weight)] = {
                    row["window"]: row
                    for row in _annotate_scores(episodes, side, radii[side])
                }
        parameter_eval = None
        common_norm = None
        if (method, reference) in parameter_matched:
            left_prediction, right_prediction, common_norm = parameter_matched[
                method, reference
            ]
            common_norm = float(common_norm)
            if not math.isfinite(common_norm) or common_norm < 0:
                raise ValueError(
                    "Matched parameter norm must be finite and nonnegative"
                )
            if common_norm > 0:
                parameter_eval = []
                for side, prediction in (
                    (method, left_prediction),
                    (reference, right_prediction),
                ):
                    _, episodes = decompose_corrections(
                        query_ids,
                        grid,
                        observed,
                        baseline,
                        prediction,
                        targets,
                        metadata=metadata,
                    )
                    parameter_eval.append(
                        {
                            row["window"]: row
                            for row in _annotate_scores(episodes, side, None)
                        }
                    )
        for window, raw_left in evaluated[(method, 1.0)].items():
            raw_right = evaluated[(reference, 1.0)][window]
            matched_left = evaluated[(method, weights[0])][window] if match else None
            matched_right = (
                evaluated[(reference, weights[1])][window] if match else None
            )
            raw_delta = (
                raw_left["mae"] - raw_right["mae"] if raw_left["n_obs"] else None
            )
            matched_delta = (
                matched_left["mae"] - matched_right["mae"]
                if match and raw_left["n_obs"]
                else None
            )
            parameter_left, parameter_right = (
                (view[window] for view in parameter_eval)
                if parameter_eval
                else (None, None)
            )
            parameter_delta = (
                parameter_left["mae"] - parameter_right["mae"]
                if parameter_eval and raw_left["n_obs"]
                else None
            )
            pair_rows.append(
                dict(metadata)
                | {
                    "method": method,
                    "reference_method": reference,
                    "window": window,
                    "n_obs": raw_left["n_obs"],
                    "n_cells": raw_left["n_cells"],
                    "anchor_match_resolved": match,
                    "common_anchor_radius": radius if match else None,
                    "candidate_anchor_radius": radii[method],
                    "reference_anchor_radius": radii[reference],
                    "candidate_eta": weights[0],
                    "reference_eta": weights[1],
                    "parameter_match_resolved": parameter_eval is not None,
                    "common_weight_frobenius": common_norm,
                    "raw_candidate_minus_reference_mae": raw_delta,
                    "matched_candidate_minus_reference_mae": matched_delta,
                    "parameter_matched_candidate_minus_reference_mae": parameter_delta,
                    "scale_adjustment_in_pair_difference": raw_delta - matched_delta
                    if matched_delta is not None
                    else None,
                }
                | _prefixed_metrics(raw_left, "raw_candidate")
                | _prefixed_metrics(raw_right, "raw_reference")
                | _prefixed_metrics(matched_left, "matched_candidate")
                | _prefixed_metrics(matched_right, "matched_reference")
                | _prefixed_metrics(parameter_left, "parameter_matched_candidate")
                | _prefixed_metrics(parameter_right, "parameter_matched_reference")
            )
    return cell_rows, episode_rows, pair_rows


def _check_outer_roster(
    cells, episodes, expected_folds, expected_draws, expected_cells_per_fold
):
    case = ["fold", "K", "draw", "method"]
    if (
        cells.duplicated(case + ["cell_uid", "window"]).any()
        or episodes.duplicated(case + ["window"]).any()
    ):
        raise ValueError("Duplicate scores on the registered fold/draw/cell axes")
    if set(cells.fold) != set(expected_folds) or set(cells.draw) != set(expected_draws):
        raise ValueError("Missing or unexpected fold/draw in source validation")
    if cells.K.nunique() != 1 or cells.seed.nunique() != 1 or (cells.K <= 0).any():
        raise ValueError("Validation uses one fixed positive K and one shared seed")
    methods = set(cells.method)
    windows = {name for name, _, _ in WINDOWS}
    physical_cells = set()
    for fold in expected_folds:
        frame = cells[cells.fold.eq(fold)]
        ids = set(frame.cell_uid)
        if len(ids) != expected_cells_per_fold or ids & physical_cells:
            raise ValueError(
                "Held-out physical roster is incomplete or reused across folds"
            )
        physical_cells.update(ids)
        if set(frame.method) != methods or set(frame.window) != windows:
            raise ValueError(
                "Each source fold must contain the same methods and windows"
            )
        n_expected = len(ids) * len(windows) * len(methods) * len(expected_draws)
        if set(frame.draw) != set(expected_draws) or len(frame) != n_expected:
            raise ValueError("A declared method/draw is missing held-out cell scores")
        if not frame.groupby(["cell_uid", "window"]).n_obs.nunique().eq(1).all():
            raise ValueError(
                "Physical observation counts differ between methods or draws"
            )
        if frame.loc[frame.window.eq("all"), "n_obs"].le(0).any():
            raise ValueError(
                "A held-out source cell lacks full-window scoring observations"
            )
    keys = case + ["window"]
    cell_cases = set(cells[keys].itertuples(index=False, name=None))
    episode_cases = set(episodes[keys].itertuples(index=False, name=None))
    if cell_cases != episode_cases:
        raise ValueError("Method/window score coverage differs from the cell ledger")
    valid = cells.n_obs.gt(0)
    metrics = ["mae", "mse", "mae_gain"]
    if (
        not np.isfinite(cells.loc[valid, metrics].to_numpy(dtype=float)).all()
        or not cells.loc[~valid, metrics].isna().to_numpy().all()
    ):
        raise ValueError("Cell error coverage differs from its observation mask")


def _score_rollups(cells):
    identifiers = ["fold", "K", "method", "window", "cell_uid"]
    metrics = [
        "mae",
        "mse",
        "mae_gain",
        "dot_ed",
        "energy_dd",
        "correction_rms",
        "alignment_with_negative_residual",
    ]
    per_cell = cells.groupby(identifiers, as_index=False, sort=True).agg(
        **{name: (name, "mean") for name in metrics},
        n_obs=("n_obs", "first"),
        valid_draws=("mae", "count"),
        negative_cell_draws=("negative_transfer", lambda values: values.eq(True).sum()),
    )
    per_cell["negative_cell"] = per_cell.mae_gain.lt(0).where(
        per_cell.valid_draws.gt(0)
    )

    def rollup(keys):
        result = per_cell.groupby(keys, as_index=False, sort=True).agg(
            **{name: (name, "mean") for name in metrics},
            query_cells=("mae", "count"),
            physical_observations=("n_obs", "sum"),
            cell_draws=("valid_draws", "sum"),
            negative_cell_draws=("negative_cell_draws", "sum"),
            negative_cells=("negative_cell", lambda values: values.eq(True).sum()),
        )
        result["negative_cell_rate"] = result.negative_cells.div(
            result.query_cells.where(result.query_cells.gt(0))
        )
        result["negative_cell_draw_rate"] = result.negative_cell_draws.div(
            result.cell_draws.where(result.cell_draws.gt(0))
        )
        return result

    return (
        per_cell,
        rollup(["fold", "K", "method", "window"]),
        rollup(["K", "method", "window"]),
    )


def _pair_rollups(pairs, expected_folds, expected_draws):
    if pairs.empty:
        return pd.DataFrame()
    keys = ["fold", "K", "method", "reference_method", "window"]
    if (
        pairs.duplicated(keys + ["draw"]).any()
        or not set(pairs.fold) <= set(expected_folds)
        or not set(pairs.draw) <= set(expected_draws)
    ):
        raise ValueError("Invalid paired contrast on fold/draw/window axis")
    raw = "raw_candidate_minus_reference_mae"
    output = "matched_candidate_minus_reference_mae"
    weights = "parameter_matched_candidate_minus_reference_mae"
    pairs = pairs.copy()
    for name in (raw, output, weights):
        pairs[name] = pd.to_numeric(pairs[name])
    return pairs.groupby(keys, as_index=False, sort=True).agg(
        registered_draws=("draw", "size"),
        evaluated_draws=(raw, "count"),
        raw_delta_mae=(raw, "mean"),
        output_matched_draws=(output, "count"),
        output_matched_delta_mae=(output, "mean"),
        parameter_matched_draws=(weights, "count"),
        parameter_matched_delta_mae=(weights, "mean"),
    )


def _cost_rollup(episodes, near_zero_anchor_radius):
    full = episodes[episodes.window.eq("all")].assign(
        near_zero=lambda frame: frame.anchor_radius.le(near_zero_anchor_radius)
    )
    fields = [name for name in COST_FIELDS if name in full]
    aggregations = {f"mean_{name}": (name, "mean") for name in fields}
    aggregations.update({f"max_{name}": (name, "max") for name in fields})
    aggregations.update(
        {
            f"sum_{name}": (name, lambda values: values.sum(min_count=1))
            for name in WORK_FIELDS
            if name in full
        }
    )
    return full.groupby(["method", "K"], as_index=False, sort=True).agg(
        episode_count=("draw", "size"),
        mean_anchor_radius=("anchor_radius", "mean"),
        max_anchor_radius=("anchor_radius", "max"),
        near_zero_anchor_radius_count=("near_zero", "sum"),
        **aggregations,
    )


def _gate_row(reference, delta_by_fold, overall, margin, minimum_folds):
    if not delta_by_fold:
        return {
            "reference_method": reference,
            "complete": False,
            "passed": False,
            "mean_delta_mae": None,
            "winning_folds": 0,
            "fold_deltas": [],
        }
    winning = sum(delta < 0 for _, delta in delta_by_fold)
    return {
        "reference_method": reference,
        "complete": True,
        "passed": bool(overall < 0 and overall <= -margin and winning >= minimum_folds),
        "mean_delta_mae": overall,
        "winning_folds": winning,
        "fold_deltas": [
            {"fold": int(fold), "delta_mae": float(delta)}
            for fold, delta in delta_by_fold
        ],
    }


def _stop_reason(raw, parameter, candidate_near_zero):
    if any(not record["complete"] for record in raw + parameter):
        return "STOP_INCOMPLETE_COMPARISON"
    if candidate_near_zero:
        return "STOP_NEAR_ZERO_CORRECTION"
    failed = {record["reference_method"] for record in raw if not record["passed"]}
    if failed & {"K0", "CAL2", "LORA_SMALL"}:
        return "STOP_SIMPLE_BASELINE"
    if failed & {"ROTATION_SHUFFLED", "ROTATION_SAME_CORE_SHUFFLED"}:
        return "STOP_PHYSICAL_CONDITIONING"
    if failed:
        return "STOP_ROTATION_CONTROL"
    if any(not record["passed"] for record in parameter):
        return "STOP_PARAMETER_MATCH_DIRECTION"
    return "ELIGIBLE_FOR_SEPARATE_SOURCE_CONFIRMATION"


def _matched_direction_checks(
    pair_scores,
    candidate,
    references,
    scope,
    expected_folds,
    expected_draws,
    required_winning_folds,
):
    outputs = []
    for reference in references:
        selected = (
            pair_scores[
                pair_scores.method.eq(candidate)
                & pair_scores.reference_method.eq(reference)
                & pair_scores.window.eq("all")
            ]
            if not pair_scores.empty
            else pd.DataFrame()
        )
        counts, differences = f"{scope}_matched_draws", f"{scope}_matched_delta_mae"
        covered = (
            set(selected.fold) == set(expected_folds)
            and selected[counts].eq(len(expected_draws)).all()
            if not selected.empty
            else False
        )
        fold_deltas = (
            list(zip(selected.fold, selected[differences], strict=True))
            if covered
            else []
        )
        overall = float(selected[differences].mean()) if covered else None
        outputs.append(
            _gate_row(reference, fold_deltas, overall, 0.0, required_winning_folds)
        )
    return outputs


def summarize_results(
    cell_rows,
    episode_rows,
    pair_rows,
    *,
    candidate="ROTATION_TRUE",
    required_references=RAW_REFERENCES,
    parameter_references=PARAMETER_REFERENCES,
    expected_folds=(0, 1, 2),
    expected_draws=(0, 1, 2),
    expected_cells_per_fold=20,
    margin=1e-5,
    required_winning_folds=2,
    near_zero_anchor_radius=1e-5,
):
    """Aggregate outer validation scores by draw, physical cell and fold."""
    cells, episodes, pairs = (
        pd.DataFrame(rows) for rows in (cell_rows, episode_rows, pair_rows)
    )
    if cells.empty or episodes.empty:
        raise ValueError("Source validation scores must be present before stopping")
    required_references = tuple(required_references)
    if "GAIN" in set(cells.method) and "GAIN" not in required_references:
        required_references += ("GAIN",)
    _check_outer_roster(
        cells, episodes, expected_folds, expected_draws, expected_cells_per_fold
    )
    per_cell, fold_scores, method_scores = _score_rollups(cells)
    pair_scores = _pair_rollups(pairs, expected_folds, expected_draws)
    costs = _cost_rollup(episodes, near_zero_anchor_radius)
    full = per_cell[per_cell.window.eq("all")].pivot(
        index=["fold", "cell_uid"], columns="method", values="mae"
    )
    raw_gates = []
    for reference in required_references:
        delta = (
            (full[candidate] - full[reference])
            if {candidate, reference} <= set(full.columns)
            else None
        )
        fold_deltas = (
            list(delta.groupby("fold").mean().items()) if delta is not None else []
        )
        overall = float(delta.mean()) if delta is not None else None
        raw_gates.append(
            _gate_row(reference, fold_deltas, overall, margin, required_winning_folds)
        )
    param = _matched_direction_checks(
        pair_scores,
        candidate,
        parameter_references,
        "parameter",
        expected_folds,
        expected_draws,
        required_winning_folds,
    )
    output = _matched_direction_checks(
        pair_scores,
        candidate,
        required_references,
        "output",
        expected_folds,
        expected_draws,
        required_winning_folds,
    )
    candidate_cost = costs[costs.method.eq(candidate)]
    near_zero = bool(
        len(candidate_cost)
        and candidate_cost.max_anchor_radius.le(near_zero_anchor_radius).all()
    )
    direction = _stop_reason(raw_gates, param, near_zero)
    decision = {
        "direction": direction,
        "reason_zh": STOP_EXPLANATIONS[direction],
        "raw_mae_margin": margin,
        "minimum_winning_folds": required_winning_folds,
        "near_zero_anchor_radius": near_zero_anchor_radius,
        "all_candidate_episodes_near_zero": near_zero,
        "raw_mae_gates": raw_gates,
        "parameter_match_gates": param,
        "output_match_direction_diagnostics": output,
        "fold_count": len(expected_folds),
        "draw_count_per_fold": len(expected_draws),
        "physical_query_cells": int(full.shape[0]),
        "target_future_rows_loaded": 0,
        "evidence_scope": "source_development_screen_only",
    }
    return {
        "fold_scores": fold_scores,
        "method_scores": method_scores,
        "pair_scores": pair_scores,
        "costs": costs,
        "cell_draw_means": per_cell,
        "decision": decision,
    }


def write_summary(directory, cell_rows, episode_rows, pair_rows, **kwargs):
    """Write complete source validation tables and a hash-bound decision."""
    from scripts.drs_peft.data import file_hash, write_json
    from scripts.drs_peft.summarize_abba import atomic_csv

    directory = Path(directory)
    result = summarize_results(cell_rows, episode_rows, pair_rows, **kwargs)
    threshold = result["decision"]["near_zero_anchor_radius"]
    episodes = pd.DataFrame(episode_rows).assign(
        near_zero_anchor=lambda frame: frame.anchor_radius.le(threshold)
    )
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.assign(
            near_zero_candidate_anchor=lambda frame: frame.candidate_anchor_radius.le(
                threshold
            ),
            near_zero_reference_anchor=lambda frame: frame.reference_anchor_radius.le(
                threshold
            ),
        )
    tables = {
        "CELL_SCORES.csv": pd.DataFrame(cell_rows),
        "EPISODE_SCORES.csv": episodes,
        "PAIRED_EPISODE_SCORES.csv": pairs,
    }
    names = ("fold_scores", "method_scores", "pair_scores", "costs", "cell_draw_means")
    tables.update({f"{name.upper()}.csv": result[name] for name in names})
    file_sha = {}
    for name, frame in tables.items():
        path = directory / name
        atomic_csv(path, frame)
        file_sha[name] = file_hash(path)
    receipt = result["decision"] | {"status": "COMPLETE", "files_sha256": file_sha}
    write_json(directory / "SOURCE_VALIDATION_DECISION.json", receipt)
    return receipt
