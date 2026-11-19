"""Pair exact MSE decompositions across Support and Query."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.run_correction_replay import PUBLIC, csv
from scripts.drs_peft.train import read_json

ATOL = 1e-14


def join_episodes(frame, keys, source):
    views = {
        "sampled": frame[
            frame.population.eq("SUPPORT")
            & frame.support_label_view.eq("sampled_train")
        ],
        "full": frame[
            frame.population.eq("SUPPORT")
            & frame.support_label_view.eq("observed_full")
        ],
        "query": frame[frame.population.eq("QUERY")],
    }
    fields = [
        "dot_ed",
        "energy_dd",
        "delta_mse",
        "mse_base",
        "mse_adapted",
        "mae_base",
        "mae_adapted",
        "classification",
    ]
    tables = []
    for view, table in views.items():
        data = table[keys + fields].copy()
        tables.append(data.rename(columns={f: f"{view}_{f}" for f in fields}))
    result = (
        tables[0]
        .merge(tables[1], on=keys, validate="one_to_one")
        .merge(tables[2], on=keys, validate="one_to_one")
    )
    if len(result) != len(tables[2]):
        raise RuntimeError(
            "Each Query decomposition needs both Support views in the same window"
        )
    result["fit_beneficial_query_wrong_direction"] = result.sampled_dot_ed.lt(
        -ATOL
    ) & result.query_dot_ed.gt(ATOL)
    result["fit_improved_query_mse_worse"] = result.sampled_delta_mse.lt(
        -ATOL
    ) & result.query_delta_mse.gt(ATOL)
    result["query_overshoot"] = result.query_dot_ed.lt(
        -ATOL
    ) & result.query_delta_mse.gt(ATOL)
    result["support_direction_sign_flips_sample_vs_full"] = (
        result.sampled_dot_ed.gt(ATOL) & result.full_dot_ed.lt(-ATOL)
    ) | (result.sampled_dot_ed.lt(-ATOL) & result.full_dot_ed.gt(ATOL))
    result["support_gain_sign_flips_sample_vs_full"] = (
        result.sampled_delta_mse.gt(ATOL) & result.full_delta_mse.lt(-ATOL)
    ) | (result.sampled_delta_mse.lt(-ATOL) & result.full_delta_mse.gt(ATOL))
    result["stage"] = "SOURCE" if source else "TARGET"
    return result


def aggregate_frame(frame):
    columns = [
        "dot_ed",
        "energy_dd",
        "delta_mse",
        "mse_base",
        "mse_adapted",
        "mae_base",
        "mae_adapted",
    ]
    named = {key: (key, "mean") for key in columns}
    keys = [
        "stage",
        "domain",
        "method",
        "K",
        "window",
        "population",
        "support_label_view",
    ]
    valid = frame.loc[frame.energy_dd.notna()].copy()
    valid["direction_negative"] = valid.dot_ed.lt(-ATOL)
    valid["direction_positive"] = valid.dot_ed.gt(ATOL)
    valid["overshoot"] = valid.dot_ed.lt(-ATOL) & valid.delta_mse.gt(ATOL)
    valid["mse_worse"] = valid.delta_mse.gt(ATOL)
    result = (
        valid.groupby(keys, dropna=False)
        .agg(
            **named,
            eligible_rows=("delta_mse", "size"),
            direction_negative=("direction_negative", "sum"),
            direction_positive=("direction_positive", "sum"),
            overshoot=("overshoot", "sum"),
            mse_worse=("mse_worse", "sum"),
        )
        .reset_index()
    )
    result["function_displacement_rms"] = np.sqrt(result.energy_dd)
    result["delta_mae"] = result.mae_adapted - result.mae_base
    return result


def sign_variation(rows, episode_keys):
    valid = rows.loc[rows.K.gt(0) & rows.n_obs.gt(0)].copy()
    if "is_duplicate" in valid:
        valid = valid.loc[~valid.is_duplicate]
    valid["direction"] = np.select(
        [valid.dot_ed.lt(-ATOL), valid.dot_ed.gt(ATOL)], [-1, 1], default=0
    )
    tables = []
    for axes, scope, views in (
        (["window"], "across_cells_in_window", valid),
        (
            ["cell_uid"],
            "across_disjoint_windows_in_cell",
            valid.loc[valid.window.ne("all")],
        ),
    ):
        signs = (
            views.groupby(episode_keys + axes)
            .direction.agg(["min", "max"])
            .reset_index()
        )
        signs["both_directions"] = signs["min"].lt(0) & signs["max"].gt(0)
        keys = ["domain", "method", "K"] + (
            ["window"] if scope == "across_cells_in_window" else []
        )
        summaries = (
            signs.groupby(keys).both_directions.agg(["size", "sum"]).reset_index()
        )
        tables.append(
            summaries.rename(
                columns={"size": "units", "sum": "units_with_both_signs"}
            ).assign(scope=scope)
        )
    return pd.concat(tables, ignore_index=True)


def summarize(root):
    source_meta, target_support_meta, target_query_meta = [
        read_json(root / f"{name}.json")
        for name in (
            "SOURCE_DIRECTION_COMPLETE",
            "TARGET_SUPPORT_COMPLETE",
            "TARGET_QUERY_COMPLETE",
        )
    ]
    checks = {
        "SOURCE_CELL.csv": source_meta["source_cells_sha256"],
        "SOURCE_EPISODE.csv": source_meta["source_episodes_sha256"],
        "TARGET_SUPPORT_CELL.csv": target_support_meta["cell_sha256"],
        "TARGET_SUPPORT_EPISODE.csv": target_support_meta["episode_sha256"],
        **target_query_meta["artifacts"],
    }
    if any(
        record["status"] != "COMPLETE"
        for record in (source_meta, target_support_meta, target_query_meta)
    ) or any(file_hash(root / path) != digest for path, digest in checks.items()):
        raise RuntimeError("Source or target decomposition input failed authentication")
    source, support, query = [
        pd.read_csv(root / f"{title}_EPISODE.csv")
        for title in ("SOURCE", "TARGET_SUPPORT", "TARGET_QUERY")
    ]
    source["stage"] = "SOURCE"
    support["stage"] = query["stage"] = "TARGET"
    if "population" not in query:
        query["population"] = "QUERY"
    if "support_label_view" not in query:
        query["support_label_view"] = "observed_full"
    source_keys = ["domain", "fold", "seed", "method", "K", "draw", "window"]
    target_keys = ["domain", "seed", "method", "K", "draw", "window"]
    tgt = pd.concat([support, query.loc[query.K.gt(0)]], ignore_index=True)
    raw = pd.concat([source, tgt], ignore_index=True)
    source_pairs = join_episodes(source, source_keys, True)
    target_pairs = join_episodes(tgt, target_keys, False)
    duplicate = query[target_keys + ["is_duplicate"]].drop_duplicates(target_keys)
    target_pairs = target_pairs.merge(duplicate, on=target_keys, validate="one_to_one")
    pair = pd.concat([source_pairs, target_pairs], ignore_index=True)
    pair["is_duplicate"] = pair["is_duplicate"].fillna(False).astype(bool)
    duplicate_axes = pair.loc[~pair.is_duplicate & pair.query_energy_dd.notna()]
    counts = [
        "fit_beneficial_query_wrong_direction",
        "fit_improved_query_mse_worse",
        "query_overshoot",
        "support_direction_sign_flips_sample_vs_full",
        "support_gain_sign_flips_sample_vs_full",
    ]
    grouped = (
        duplicate_axes.groupby(["stage", "domain", "method", "K", "window"])
        .agg(
            eligible_episodes=("query_delta_mse", "size"),
            **{k: (k, "sum") for k in counts},
        )
        .reset_index()
    )
    csv(root / "PAIRED_DIRECTION_EPISODES.csv", pair)
    csv(root / "DIRECTION_COUNTS_UNIQUE.csv", grouped)
    csv(root / "DECOMPOSITION_SUMMARY.csv", aggregate_frame(raw))
    variation = pd.concat(
        [
            sign_variation(
                pd.read_csv(root / "SOURCE_CELL.csv").query("population == 'QUERY'"),
                source_keys[:-1],
            ).assign(stage="SOURCE"),
            sign_variation(
                pd.read_csv(root / "TARGET_QUERY_CELL.csv"), target_keys[:-1]
            ).assign(stage="TARGET"),
        ],
        ignore_index=True,
    )
    csv(root / "QUERY_SIGN_VARIATION.csv", variation)
    names = (
        "PAIRED_DIRECTION_EPISODES.csv",
        "DIRECTION_COUNTS_UNIQUE.csv",
        "DECOMPOSITION_SUMMARY.csv",
        "QUERY_SIGN_VARIATION.csv",
    )
    write_json(
        root / "E1_SUMMARY.json",
        {
            "status": "COMPLETE",
            "source_episode_windows": len(source_pairs),
            "target_episode_windows": len(target_pairs),
            "target_unique_nonzero_episodes": int(
                (~target_pairs.is_duplicate).sum() / 5
            ),
            "classification_tolerance": ATOL,
            "oracle_scope": "Retrospective minimizer for the same evaluated labels; never an inference rule",
            "input_sha256": checks,
            "files": {name: file_hash(root / name) for name in names},
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-root", type=Path, default=PUBLIC)
    summarize(p.parse_args().public_root)
