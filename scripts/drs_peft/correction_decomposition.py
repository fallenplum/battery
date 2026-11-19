"""Decompose the output corrections of frozen ABBA-family predictions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.e123_protocol import WINDOWS


CLASSIFICATION_TOLERANCE = 1e-14
MOMENTS = (
    "dot_ed", "energy_dd", "mse_base", "mse_adapted", "mse_eta",
    "mae_base", "mae_adapted", "mae_eta",
)
DERIVED = (
    "cross_term", "norm_term", "delta_mse", "delta_mse_direct",
    "decomposition_residual", "delta_mae", "dot_sign_exact", "delta_mse_sign_exact",
    "oracle_eta", "oracle_eta_clipped_0_1",
)


def _statistics(moments, eta):
    if moments is None:
        return dict.fromkeys(MOMENTS + DERIVED) | {"classification": "no_observations"}
    dot, energy = moments["dot_ed"], moments["energy_dd"]
    cross_term = 2 * eta * dot
    norm_term = eta**2 * energy
    difference = cross_term + norm_term
    direct = moments["mse_eta"] - moments["mse_base"]
    if not math.isclose(difference, direct, rel_tol=1e-12, abs_tol=1e-12):
        raise ArithmeticError("Observed MSE difference differs from its correction decomposition")

    tolerance = CLASSIFICATION_TOLERANCE
    if not eta or not energy:
        classification = "no_op"
    elif dot > tolerance:
        classification = "wrong_direction"
    elif difference < -tolerance:
        classification = "beneficial"
    elif difference <= tolerance:
        classification = "neutral_update"
    elif dot < -tolerance:
        classification = "overshoot"
    else:
        classification = "orthogonal_update"
    oracle = max(0.0, -dot / energy) if energy else 0.0
    return moments | dict(
        cross_term=cross_term, norm_term=norm_term, delta_mse=difference,
        delta_mse_direct=direct, decomposition_residual=direct - difference,
        delta_mae=moments["mae_eta"] - moments["mae_base"],
        dot_sign_exact=int(np.sign(dot)), delta_mse_sign_exact=int(np.sign(difference)),
        oracle_eta=oracle, oracle_eta_clipped_0_1=min(1.0, oracle),
        classification=classification,
    )


def decompose_corrections(ids, grid, observed, base, adapted, targets, *, metadata=None, eta=1.0):
    """Return observed-window cell rows and equal-cell episode rows as dictionaries.

    A shared one-dimensional cycle grid and a per-cell two-dimensional grid
    are both accepted. The latter can contain padding outside observed points.
    MSE/MAE fields ending in ``adapted`` use the saved model output at eta=1.
    Fields ending in ``eta`` and the decomposition use the requested scale.
    Oracle scales are descriptive and never used to alter any prediction.
    """
    eta = float(eta)
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("Correction eta must be finite and nonnegative")
    ids = tuple(ids)
    observed = np.asarray(observed)
    if observed.dtype != bool or observed.ndim != 2 or observed.shape[0] != len(ids):
        raise ValueError("Observed must be a boolean cell-by-cycle mask")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected a nonempty roster of distinct physical cells")
    grid = np.asarray(grid, dtype=np.float64)
    if grid.shape == (observed.shape[1],):
        grid = np.broadcast_to(grid, observed.shape)
    if grid.shape != observed.shape:
        raise ValueError("Cycle grid must match the observations")
    arrays = [np.asarray(value, dtype=np.float64) for value in (base, adapted, targets)]
    if any(value.shape != observed.shape for value in arrays):
        raise ValueError("Baseline, adapted and target arrays must match observations")

    common = dict(metadata or {}, eta=eta)
    cell_rows, macro_rows = [], []
    by_window = {name: [] for name, _, _ in WINDOWS}
    for index, cell_uid in enumerate(ids):
        mask = observed[index]
        cycles = grid[index, mask]
        if (not np.isfinite(cycles).all() or np.any(np.diff(cycles) <= 0)
                or np.any((cycles < WINDOWS[0][1]) | (cycles > WINDOWS[0][2]))):
            raise ValueError(f"Observed cycles must be finite, in range and increasing at {cell_uid}")
        baseline, prediction, truth = [value[index, mask] for value in arrays]
        if not np.isfinite([baseline, prediction, truth]).all():
            raise ValueError(f"Nonfinite observed value at {cell_uid}")
        error = baseline - truth
        correction = prediction - baseline
        adapted_error = prediction - truth
        scaled_error = adapted_error if eta == 1 else error + eta * correction
        for name, start, end in WINDOWS:
            keep = (cycles >= start) & (cycles <= end)
            count = int(keep.sum())
            moments = None
            if count:
                e, d, a, s = (value[keep] for value in (error, correction, adapted_error, scaled_error))
                moments = dict(
                    dot_ed=float(np.mean(e * d)), energy_dd=float(np.mean(d**2)),
                    mse_base=float(np.mean(e**2)), mse_adapted=float(np.mean(a**2)),
                    mse_eta=float(np.mean(s**2)), mae_base=float(np.mean(np.abs(e))),
                    mae_adapted=float(np.mean(np.abs(a))), mae_eta=float(np.mean(np.abs(s))),
                )
                by_window[name].append((count, moments))
            cell_rows.append(common | {"cell_uid": cell_uid, "window": name, "n_obs": count}
                             | _statistics(moments, eta))

    for name, entries in by_window.items():
        count = len(entries)
        moments = None
        if count:
            moments = {key: math.fsum(value[key] for _, value in entries) / count for key in MOMENTS}
        macro_rows.append(common | {"window": name, "n_cells": count, "n_obs": sum(n for n, _ in entries)}
                          | _statistics(moments, eta))
    return cell_rows, macro_rows


def _authenticated_abba(args):
    from scripts.drs_peft import run_abba_moirai as runner
    from scripts.drs_peft import summarize_abba as audit

    config, _, episodes = runner.parents(args)
    lock_path = args.public_root / "TARGET_LOCK.json"
    lock = audit.read_json(lock_path)
    lock_sha = audit.sha256(lock_path)
    identity = runner.code_identity(args)
    if any(lock[key] != value for key, value in identity.items()):
        raise ValueError("Original ABBA execution code differs from its target lock")
    audit.validate_lock(args.public_root, config, episodes, identity["config_sha256"],
                        audit.sha256(audit.EPISODES), lock)
    registered = audit.registered_slots(episodes)
    barrier_sha = audit.validate_prediction_barrier(args.public_root, registered, config, episodes, lock, lock_sha)
    return registered, lock_sha, barrier_sha


def _parent_predictions(args, registered):
    from scripts.drs_peft import run_abba_moirai as runner
    from scripts.drs_peft import summarize_abba as audit

    for key, (task, draw) in registered.items():
        folder = key.folder(args.public_root)
        receipt = audit.read_json(folder / "PREDICTION_LOCKED.json")
        scored = audit.read_json(folder / "RECEIPT.json")
        if scored["status"] != "SCORED" or any(
            scored[name] != value for name, value in receipt.items() if name != "status"
        ):
            raise ValueError(f"Original score evidence differs from its prediction at {key}")
        if audit.sha256(folder / "METRICS.csv") != scored["metrics_sha256"]:
            raise ValueError(f"Original metrics file hash changed at {key}")
        yield key, task, draw, folder, receipt, runner.read_prediction(args, receipt)


def _verify_original_mae(rows, folder, key, task, draw):
    from scripts.drs_peft.summarize_abba import read_metrics

    prior = read_metrics(folder / "METRICS.csv", key, task, draw)
    actual = pd.DataFrame(rows)
    axes = ["cell_uid", "window", "n_obs"]
    if not prior[axes].equals(actual[axes]):
        raise ValueError(f"Observed cell/window axis changed at {key}")
    for field, expected in (("mae_base", "zero_mae"), ("mae_adapted", "mae")):
        if not np.allclose(prior[expected].astype(float), actual[field].astype(float),
                           rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"Recomputed {field} differs from the original score at {key}")


def target_query(args, registered, lock_sha, barrier_sha):
    from scripts.drs_peft.data import file_hash, write_json
    from scripts.drs_peft.summarize_abba import atomic_csv, read_json
    from scripts.drs_peft.transfer_scope_data import load_future_rows

    source_gate_path = args.output / "SOURCE_DIRECTION_COMPLETE.json"
    if not source_gate_path.is_file():
        raise RuntimeError("Source-direction completion is required before target Query outcomes are read")
    source_gate = read_json(source_gate_path)
    if (source_gate.get("status") != "COMPLETE"
            or source_gate.get("scope") != "SOURCE_DEVELOPMENT_FIRST"
            or source_gate.get("target_rows_read") != 0):
        raise RuntimeError("Source direction evidence did not preserve the research order")
    for name, field in (("SOURCE_CELL.csv", "source_cells_sha256"),
                        ("SOURCE_EPISODE.csv", "source_episodes_sha256")):
        if file_hash(args.output / name) != source_gate[field]:
            raise ValueError(f"Source direction artifact changed before target scoring: {name}")
    provenance = {
        "source_direction_sha256": file_hash(source_gate_path),
        "abba_target_lock_sha256": lock_sha,
        "abba_prediction_barrier_sha256": barrier_sha,
        "implementation_sha256": file_hash(Path(__file__)),
    }
    marker = args.output / "TARGET_QUERY_COMPLETE.json"
    if marker.is_file():
        completed = read_json(marker)
        if completed["provenance"] != provenance or any(
            file_hash(args.output / name) != sha for name, sha in completed["artifacts"].items()
        ):
            raise ValueError("Existing Query decomposition evidence differs from its locked inputs")
        return completed

    grid = np.arange(101, 5001, dtype=np.float32)
    targets_by_domain, baselines = {}, {}
    cell_rows, episode_rows = [], []
    for key, task, draw, folder, receipt, adapted in _parent_predictions(args, registered):
        ids = task["query_ids"]
        if key.domain not in targets_by_domain:
            targets, observed = load_future_rows(key.domain, args.data_root, ids)
            targets_by_domain[key.domain] = ids, targets, observed
        expected_ids, targets, observed = targets_by_domain[key.domain]
        if ids != expected_ids:
            raise ValueError("A fixed target-domain Query roster changed")
        baseline_key = (key.domain, key.seed)
        if key.K == 0:
            baselines[baseline_key] = adapted
        metadata = key._asdict() | {
            "stage": "TARGET", "population": "QUERY", "support_label_view": "observed_full",
            "task_id": receipt["task_id"],
            "is_duplicate": receipt["is_duplicate"], "adaptation_id": receipt["adaptation_id"],
            "support_set_sha256": draw["support_set_sha256"],
        }
        cells, episodes = decompose_corrections(ids, grid, observed, baselines[baseline_key],
                                               adapted, targets, metadata=metadata)
        _verify_original_mae(cells, folder, key, task, draw)
        protocols = task["protocol_by_id"]
        cell_rows.extend(row | {"protocol_uid": protocols[row["cell_uid"]]} for row in cells)
        episode_rows.extend(episodes)

    artifacts = {}
    for name, rows in (("TARGET_QUERY_CELL.csv", cell_rows), ("TARGET_QUERY_EPISODE.csv", episode_rows)):
        path = args.output / name
        atomic_csv(path, pd.DataFrame(rows))
        artifacts[name] = file_hash(path)
    evidence = {
        "status": "COMPLETE", "provenance": provenance, "artifacts": artifacts,
        "registered_prediction_slots": len(registered), "cell_window_rows": len(cell_rows),
        "episode_window_rows": len(episode_rows), "domain_suffix_reads": len(targets_by_domain),
        "target_query_future_rows_loaded": sum(len(row[0]) for row in targets_by_domain.values()),
        "historically_exposed_target_query": True, "target_query_used_for_tuning": False,
        "new_optimizer_steps": 0, "new_model_forwards": 0,
        "per_cell_error_weighting": "equal_cells_with_at_least_one_observation_in_window",
        "classification_absolute_tolerance": CLASSIFICATION_TOLERANCE,
        "oracle_eta_use": "descriptive_mse_optimum_only",
        "max_decomposition_residual_abs": max(
            abs(row["decomposition_residual"]) for row in cell_rows + episode_rows if row["n_obs"]
        ),
    }
    write_json(marker, evidence)
    return evidence


def main():
    from scripts.drs_peft import run_abba_moirai as runner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("verify-parents", "target-query"), required=True)
    parser.add_argument("--config", type=Path, default=runner.CONFIG)
    parser.add_argument("--public-root", type=Path, default=runner.PUBLIC)
    parser.add_argument("--private-root", type=Path, default=runner.PRIVATE)
    parser.add_argument("--data-root", type=Path, default=runner.CACHE)
    parser.add_argument("--output", type=Path,
                        default=runner.ROOT / "results/drs_peft/correction_transfer_v1/e1")
    args = parser.parse_args()
    registered, lock_sha, barrier_sha = _authenticated_abba(args)
    if args.phase == "verify-parents":
        verified = sum(1 for _ in _parent_predictions(args, registered))
        print(f"Verified {verified} saved ABBA prediction slots without reading future observations")
    else:
        evidence = target_query(args, registered, lock_sha, barrier_sha)
        print(f"Decomposed {evidence['registered_prediction_slots']} saved target Query predictions")


if __name__ == "__main__":
    main()
