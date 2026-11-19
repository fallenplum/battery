"""Audit the locked target metrics and summarize the ABBA capacity diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from scripts.drs_peft.e123_protocol import WINDOWS
from scripts.drs_peft.onesided_protocol import payload_hash


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "results/drs_peft/abba_moirai_v1"
CONFIG = ROOT / "configs/drs_peft/abba_moirai_v1/CONFIG.json"
EPISODES = ROOT / "configs/drs_peft/specialization_comparison_v1/EPISODES_TOTAL_K.json"
REPORT = ROOT / "report/drs_peft/abba_moirai_v1/ABBA_P2_P3_SUMMARY_ZH.md"
HISTORICAL = ROOT / "results/drs_peft/transfer_scope_v2/HISTORICAL_ANCHORS.json"
PRIMARY = ("ABBA_R4X4", "LORA_R8", "HIRA_R8", "ADALORA_R12_TO_R8")
METHODS = (*PRIMARY, "LORA_R2", "LORA_R16")
CONTRASTS = (
    ("ABBA_R4X4", "LORA_R8"), ("ABBA_R4X4", "HIRA_R8"),
    ("ABBA_R4X4", "ADALORA_R12_TO_R8"), ("LORA_R8", "LORA_R2"),
    ("LORA_R16", "LORA_R8"), ("LORA_R16", "LORA_R2"),
)
KEY = ["domain", "seed", "method", "K", "draw"]
MATCH = ["domain", "seed", "K", "draw", "window"]
AXES = ["cell_uid", "window"]
MEASURES = ["mae", "rmse", "bias", "zero_mae", "gain_mae"]
WINDOW_NAMES = [name for name, _, _ in WINDOWS]
SPECTRUM_COLUMNS = (
    "module", "rows", "cols", "frobenius", "relative_frobenius", "spectral_norm",
    "numerical_rank_rtol_1e_5", "stable_rank", "entropy_rank", "energy_rank_99",
    "singular_values_json", "allocated_rank", "rank_bound",
)


class Slot(NamedTuple):
    domain: str
    seed: int
    method: str
    K: int
    draw: int

    def folder(self, public):
        return public / "target" / self.domain / f"s{self.seed}" / self.method / f"{self.K}_{self.draw}"


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, *, tol=1e-10):
    return np.allclose(a, b, rtol=0, atol=tol, equal_nan=True)


def boolean(value):
    check(value in (True, False, "True", "False", "true", "false", "0", "1"),
          f"Invalid boolean {value!r}")
    return value in (True, "True", "true", "1")


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(value, encoding="utf-8")
    pending.replace(path)


def atomic_csv(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    table.to_csv(pending, index=False, encoding="utf-8")
    pending.replace(path)


def registered_slots(episodes):
    slots = {}
    seeds = episodes["training_seeds"]
    check(episodes["shots"] == [0, 1, 3, 5] and episodes["support_draws"] == 5,
          "Unexpected frozen K or draw axis")
    for task in episodes["target_tasks"]:
        query = task["query_ids"]
        pool = task["support_pool_ids"]
        check(len(set(query)) == len(query) and len(set(pool)) == len(pool) and not set(query) & set(pool),
              "Target Query and Support pool are not disjoint physical sets")
        check([shot["k"] for shot in task["shots"]] == episodes["shots"], "Invalid task shot axis")
        smaller = {}
        for shot in task["shots"]:
            k = shot["k"]
            seen = {}
            members = {}
            records = shot["draws"]
            check([row["draw"] for row in records] == list(range(5 if k else 1)), "Invalid draw axis")
            for row in records:
                draw = row["draw"]
                support = row["support_ids"]
                support_key = tuple(sorted(support))
                check(len(set(support)) == len(support) == k and set(support) <= set(pool)
                      and row["query_ids"] == query and smaller.get(draw, set()) <= set(support),
                      "Invalid target total-K Support/Query or nesting")
                check(row["duplicate_of_draw"] == seen.get(support_key), "Invalid repeat-draw identity")
                seen.setdefault(support_key, draw)
                members[draw] = set(support)
                for seed, method in product(seeds, METHODS if k else ("ZERO",)):
                    key = Slot(task["domain"], seed, method, k, draw)
                    check(key not in slots, f"Duplicate registered slot {key}")
                    slots[key] = task, row
            check(len(seen) == shot["unique_support_sets"], "Unexpected distinct Support-set count")
            smaller = members
    check({key.domain for key in slots} == {"MATR", "HUST"}, "Unexpected target domains")
    return slots


def read_metrics(path, key, task, row):
    data = pd.read_csv(path, float_precision="round_trip")
    for field, value in zip(KEY, key, strict=True):
        check(data[field].eq(value).all(), f"Metric identity differs at {key} {field}")
    query = task["query_ids"]
    expected = pd.MultiIndex.from_product([query, WINDOW_NAMES], names=AXES)
    data = data.set_index(AXES, verify_integrity=True)
    check(data.index.difference(expected).empty and expected.difference(data.index).empty,
          f"Wrong cell/window roster at {key}")
    data = data.reindex(expected).reset_index()
    check(data["coordinate"].eq("original_cycle_id").all()
          and data["reference_family"].eq("SPECIALIZED").all()
          and data["protocol_uid"].eq(data.cell_uid.map(task["protocol_by_id"])).all()
          and data["is_duplicate"].map(boolean).eq(row["duplicate_of_draw"] is not None).all(),
          f"Wrong metric metadata at {key}")
    max_points = data.window.map({name: end - start + 1 for name, start, end in WINDOWS})
    check(data.n_obs.between(0, max_points).all() and data.n_obs.eq(np.floor(data.n_obs)).all(),
          f"Invalid counts at {key}")
    n_obs = data.pivot(index="cell_uid", columns="window", values="n_obs")
    check(n_obs["all"].eq(n_obs[WINDOW_NAMES[1:]].sum(axis=1)).all(), f"Window coverage differs at {key}")
    valid = data.n_obs.gt(0)
    check(np.isfinite(data.loc[valid, MEASURES]).to_numpy().all()
          and data.loc[~valid, MEASURES].isna().to_numpy().all()
          and data.loc[valid, ["mae", "rmse", "zero_mae"]].ge(0).to_numpy().all()
          and close(data.zero_mae - data.mae, data.gain_mae), f"Invalid errors at {key}")
    if "negative_transfer" in data:
        check(data.loc[valid, "negative_transfer"].map(boolean).eq(data.loc[valid, "gain_mae"].lt(0)).all(),
              f"Incorrect negative transfer at {key}")
    return data


def spectrum_statistics(value):
    s = np.array(json.loads(value.singular_values_json), dtype=np.float64)
    check(s.shape == (min(value.rows, value.cols),) and np.isfinite(s).all()
          and (s >= 0).all() and (np.diff(s) <= 0).all(), f"Invalid SVD for {value.module}")
    energy = np.square(s).sum()
    if energy == 0:
        return 0., 0., 0, 0., 0., 0
    p = s[s > 0] / s.sum()
    return (math.sqrt(energy), s[0], int((s > 1e-5 * s[0]).sum()), energy / s[0] ** 2,
            float(np.exp(-(p * np.log(p)).sum())), int(np.searchsorted(np.cumsum(np.square(s)) / energy, .99) + 1))


def read_spectrum(path, key, expected_modules):
    spectrum = pd.read_csv(path, float_precision="round_trip")
    check(set(SPECTRUM_COLUMNS) <= set(spectrum.columns), f"Spectrum fields missing at {key}")
    if not key.K:
        check(spectrum.empty, f"K0 reported nonzero spectral records at {key}")
        return spectrum
    check(len(spectrum) == expected_modules and not spectrum.module.duplicated().any(),
          f"Wrong layer coverage at {key}")
    numeric = list(SPECTRUM_COLUMNS[1:10]) + ["allocated_rank", "rank_bound"]
    check(np.isfinite(spectrum[numeric]).to_numpy().all() and spectrum[numeric].ge(0).to_numpy().all(),
          f"Invalid spectral summary at {key}")
    for value in spectrum.itertuples(index=False):
        numbers = spectrum_statistics(value)
        observed = [value.frobenius, value.spectral_norm, value.numerical_rank_rtol_1e_5,
                    value.stable_rank, value.entropy_rank, value.energy_rank_99]
        check(np.allclose(numbers, observed, rtol=1e-5, atol=1e-10)
              and value.numerical_rank_rtol_1e_5 <= value.rank_bound <= min(value.rows, value.cols)
              and value.energy_rank_99 <= value.rank_bound and value.allocated_rank <= value.rank_bound,
              f"Spectrum and reported ranks differ at {key} {value.module}")
    return spectrum


def unit_fields(key, task, row, lock, lock_sha):
    return dict(zip(KEY, key, strict=True)) | {
        "lock_sha256": lock_sha, "task_id": task["task_id"], "reference_family": "SPECIALIZED",
        "support_ids": sorted(row["support_ids"]), "query_ids": task["query_ids"],
        "reference_sha256": lock["specialized_reference_sha256_by_seed"][str(key.seed)],
        "learning_rate": lock["learning_rates"][key.method][str(key.K)] if key.K else None,
        "is_duplicate": row["duplicate_of_draw"] is not None,
    }


def validate_prediction_barrier(public, registered, config, episodes, lock, lock_sha):
    path = public / "TARGET_PREDICTIONS_COMPLETE.json"
    barrier = read_json(path)
    ordered = []
    unique = set()
    optimizer_steps = 0
    for task, seed, method in product(episodes["target_tasks"], config["target"]["seeds"], ("ZERO", *config["methods"])):
        task_keys = (Slot(task["domain"], seed, method, k, draw) for k in ((1, 3, 5) if method != "ZERO" else (0,))
                     for draw in range(5 if k else 1))
        for key in task_keys:
            _, row = registered[key]
            receipt = read_json(key.folder(public) / "PREDICTION_LOCKED.json")
            identity = unit_fields(key, task, row, lock, lock_sha)
            canonical = {name: value for name, value in identity.items() if name not in {"draw", "is_duplicate"}}
            check(all(receipt[name] == value for name, value in identity.items())
                  and receipt["status"] == "PREDICTIONS_LOCKED"
                  and receipt["canonical_identity"] == canonical
                  and receipt["adaptation_id"] == payload_hash(canonical)
                  and receipt["target_query_future_rows_loaded"] == 0,
                  f"Prediction was not authenticated before target labels at {key}")
            ordered.append(receipt["prediction_sha256"])
            if key.K:
                unique.add(receipt["adaptation_id"])
                optimizer_steps += receipt["actual_optimizer_steps"]
    check(barrier == {
        "status": "ALL_AUTHENTICATED_BEFORE_QUERY_LABEL_READ", "lock_sha256": lock_sha,
        "slots": len(ordered), "unique_nonzero": len(unique), "prediction_manifest_sha256": payload_hash(ordered),
        "new_target_optimizer_updates": optimizer_steps, "bootstrap_draws": 0,
    } and len(ordered) == config["target"]["expected_scored_slots"]
          and len(unique) == config["target"]["expected_unique_adaptations"],
          "Prediction barrier identity, counts or manifest hash changed")
    return sha256(path)


def read_slot(folder, key, task, row, lock, lock_sha, module_count):
    receipt = read_json(folder / "RECEIPT.json")
    prediction = read_json(folder / "PREDICTION_LOCKED.json")
    identity = unit_fields(key, task, row, lock, lock_sha)
    check(all(receipt[name] == value for name, value in identity.items()) and receipt["status"] == "SCORED"
          and all(receipt[name] == value for name, value in prediction.items() if name != "status"),
          f"Score identity differs from its frozen predictions at {key}")
    for path, field in (("METRICS.csv", "metrics_sha256"), ("UPDATE_SPECTRUM.csv", "update_spectrum_sha256")):
        check(sha256(folder / path) == receipt[field], f"Artifact checksum mismatch at {key} {path}")
    metrics = read_metrics(folder / "METRICS.csv", key, task, row)
    check(metrics.adaptation_id.eq(receipt["adaptation_id"]).all()
          and len(metrics) == receipt["scored_cell_window_rows"],
          f"Wrong adaptation state scope at {key}")
    spectrum = read_spectrum(folder / "UPDATE_SPECTRUM.csv", key, module_count)
    values = {name: receipt[name] for name in ("trainable_parameters", "active_parameters", "optimizer_steps",
                                             "support_mse_before", "support_mse_after")}
    for name in ("trainable_parameters", "active_parameters", "optimizer_steps"):
        check(type(values[name]) is int and values[name] >= 0, f"Invalid {name} at {key}")
    duplicate = row["duplicate_of_draw"] is not None
    actual = receipt["actual_optimizer_steps"]
    check(type(actual) is int and actual == (0 if duplicate else values["optimizer_steps"])
          and (key.K == 0) == (values["optimizer_steps"] == 0)
          and values["active_parameters"] <= values["trainable_parameters"], f"Invalid optimizer/budget at {key}")
    values["actual_optimizer_steps"] = actual
    for name in ("support_mse_before", "support_mse_after"):
        value = values[name]
        check((key.K == 0 and value is None) or (key.K and isinstance(value, (float, int))
              and math.isfinite(value) and value >= 0), f"Invalid {name} at {key}")
    values["support_mse_gain"] = values["support_mse_before"] - values["support_mse_after"] if key.K else np.nan
    if key.K:
        size = spectrum.rows + spectrum.cols + int(key.method == "ADALORA_R12_TO_R8")
        check(int((size * spectrum.allocated_rank).sum()) == values["active_parameters"],
              f"Active dimension disagrees with per-layer allocation at {key}")
        if key.method == "ADALORA_R12_TO_R8":
            check(spectrum.allocated_rank.sum() <= module_count * 8
                  and (spectrum.allocated_rank <= 12).all()
                  and int((size * 12).sum()) == values["trainable_parameters"],
                  f"AdaLoRA rank budget exceeds its locked schedule at {key}")
    else:
        check(values["trainable_parameters"] == values["active_parameters"] == 0, f"K0 has parameters at {key}")
    values["update_frobenius"] = np.linalg.norm(spectrum.frobenius.to_numpy())
    values["update_spectral_norm_max"] = spectrum.spectral_norm.max() if key.K else 0.
    for name in ("relative_frobenius", "numerical_rank_rtol_1e_5", "stable_rank", "entropy_rank", "energy_rank_99"):
        values[f"mean_layer_{name}"] = spectrum[name].mean() if key.K else 0.
    values["support_set_sha256"] = row["support_set_sha256"]
    values["is_duplicate"] = duplicate
    values["adaptation_id"] = receipt["adaptation_id"]
    values["learning_rate"] = receipt["learning_rate"]
    return metrics, spectrum, dict(zip(KEY, key, strict=True)) | values


def collect(public, lock, lock_sha, registered, *, module_count=12):
    metrics, spectra, diagnostics, coverage = [], [], [], []
    zeros, support_values, layer_axes, draw_cache = {}, {}, {}, {}
    expected = {key.folder(public) / "RECEIPT.json" for key in registered}
    found = set((public / "target").glob("*/*/*/*/RECEIPT.json"))
    check(found <= expected, "Unexpected scored Target slot receipt under result root")
    for key, (task, row) in registered.items():
        folder = key.folder(public)
        status = dict(zip(KEY, key, strict=True))
        if folder / "RECEIPT.json" not in found:
            coverage.append(status | {"status": "MISSING"})
            continue
        data, spectrum, evidence = read_slot(folder, key, task, row, lock, lock_sha, module_count)
        zero_key = key.domain, key.seed
        if not key.K:
            zeros[zero_key] = data
        base = zeros.get(zero_key)
        check(base is not None and data.n_obs.equals(base.n_obs) and close(data.zero_mae, base.mae),
              f"Zero baseline or observed scoring mask differs at {key}")
        if key.K:
            support_key = key.domain, key.seed, key.K, key.draw
            before = support_values.setdefault(support_key, evidence["support_mse_before"])
            check(close(evidence["support_mse_before"], before), f"Initial Support MSE differs at {key}")
            axes = spectrum.set_index("module")[["rows", "cols"]].sort_index()
            check(layer_axes.setdefault(key.seed, axes).equals(axes), f"Frozen target layers differ at {key}")
            axes_match = data[[*AXES, "n_obs", *MEASURES]].copy()
            repeated = row["duplicate_of_draw"]
            if repeated is not None:
                old_data, old_spectrum, old_after, old_state = draw_cache[key._replace(draw=repeated)]
                check(axes_match.equals(old_data) and spectrum.equals(old_spectrum)
                      and evidence["support_mse_after"] == old_after and evidence["adaptation_id"] == old_state,
                      f"Repeated draw result differs at {key}")
            else:
                draw_cache[key] = axes_match, spectrum, evidence["support_mse_after"], evidence["adaptation_id"]
        data["negative_transfer"] = data.gain_mae.lt(0).where(data.n_obs.gt(0))
        data["support_set_sha256"] = evidence["support_set_sha256"]
        metrics.append(data)
        diagnostics.append(evidence)
        if key.K:
            spectra.append(spectrum.assign(**status))
        coverage.append(status | {"status": "SCORED", "receipt_sha256": sha256(folder / "RECEIPT.json")})
    return metrics, spectra, diagnostics, pd.DataFrame(coverage)


def check_budgets(diagnostics):
    active = diagnostics[diagnostics.K.gt(0)]
    for _, rows in active.groupby(["domain", "seed", "K", "draw"], sort=False):
        primary = rows[rows.method.isin(PRIMARY)]
        control = rows[rows.method.str.startswith("LORA_")].set_index("method")
        exact = primary[~primary.method.eq("ADALORA_R12_TO_R8")]
        check(rows.optimizer_steps.nunique() <= 1 and exact.active_parameters.nunique() <= 1
              and exact.trainable_parameters.eq(exact.active_parameters).all(), "Primary budget differs across methods")
        if {"LORA_R2", "LORA_R8", "LORA_R16"} <= set(control.index):
            base = control.loc["LORA_R8", "active_parameters"]
            check(4 * control.loc["LORA_R2", "active_parameters"] == base
                  and control.loc["LORA_R16", "active_parameters"] == 2 * base
                  and control.learning_rate.nunique() == 1, "LoRA capacity comparison differs")


def query_episodes(metrics):
    """Keep unsupported windows and omit their errors from paired denominators."""
    metrics = metrics.assign(negative_cell=metrics.negative_transfer.eq(True))
    return metrics.groupby([*KEY, "window", "is_duplicate", "support_set_sha256"], as_index=False).agg(
        query_cells=("n_obs", lambda n: n.gt(0).sum()), query_points=("n_obs", "sum"),
        query_mae=("mae", "mean"), query_rmse_cell_mean=("rmse", "mean"),
        query_bias=("bias", "mean"), zero_mae=("zero_mae", "mean"),
        query_mae_gain=("gain_mae", "mean"), negative_cells=("negative_cell", "sum"),
    )


def summarize_query(episodes):
    rows = []
    for policy, frame in (("registered_draws", episodes), ("unique_support_sets", episodes[~episodes.is_duplicate])):
        for key, group in frame.groupby(["domain", "method", "K", "window"], sort=True):
            valid = group[group.query_cells.gt(0)]
            seed_means = valid.groupby("seed").query_mae_gain.mean()
            row = dict(zip(("domain", "method", "K", "window"), key, strict=True))
            cells = valid.query_cells.sum()
            negative_episodes = valid.query_mae_gain.lt(0).sum()
            rows.append(row | dict(
                draw_policy=policy, query_mae=valid.query_mae.mean(),
                query_rmse_cell_mean=valid.query_rmse_cell_mean.mean(), query_bias=valid.query_bias.mean(),
                zero_mae=valid.zero_mae.mean(), query_mae_gain=valid.query_mae_gain.mean(),
                physical_query_cells=int(valid.query_cells.max()) if len(valid) else 0,
                physical_observations=int(valid.query_points.max()) if len(valid) else 0,
                evaluated_cell_seed_draws=cells, negative_cell_seed_draws=valid.negative_cells.sum(),
                negative_cell_rate=valid.negative_cells.sum() / cells if cells else np.nan,
                scored_seed_draws=len(valid), negative_seed_draws=negative_episodes,
                negative_seed_draw_rate=negative_episodes / len(valid) if len(valid) else np.nan,
                model_seeds=len(seed_means), improved_seed_means=seed_means.gt(0).sum(),
            ))
    return pd.DataFrame(rows)


def make_contrasts(episodes):
    outcomes = []
    columns = [*MATCH, "support_set_sha256", "is_duplicate", "query_cells", "query_points",
               "query_mae", "negative_cells", "support_mse_after", "update_frobenius", "mean_layer_energy_rank_99"]
    for method, reference in CONTRASTS:
        left = episodes.loc[episodes.method.eq(method), columns]
        right = episodes.loc[episodes.method.eq(reference), columns]
        pair = left.merge(right, on=MATCH, suffixes=("", "_reference"), validate="one_to_one")
        for identity in ("support_set_sha256", "is_duplicate", "query_cells", "query_points"):
            check(pair[identity].eq(pair[f"{identity}_reference"]).all(), f"Pair identity differs for {method}/{reference}")
            pair = pair.drop(columns=[f"{identity}_reference"])
        pair["method"] = method
        pair["reference"] = reference
        pair["query_mae_improvement"] = pair.query_mae_reference - pair.query_mae
        pair["support_mse_improvement"] = pair.support_mse_after_reference - pair.support_mse_after
        pair["update_frobenius_difference"] = pair.update_frobenius - pair.update_frobenius_reference
        pair["energy_rank_99_difference"] = pair.mean_layer_energy_rank_99 - pair.mean_layer_energy_rank_99_reference
        outcomes.append(pair)
    return pd.concat(outcomes, ignore_index=True)


def summarize_pairs(contrasts):
    output = []
    keys = ["domain", "K", "window", "method", "reference"]
    for policy, frame in (("registered_draws", contrasts), ("unique_support_sets", contrasts[~contrasts.is_duplicate])):
        for group, rows in frame.groupby(keys, sort=True):
            rows = rows[rows.query_cells.gt(0)]
            seed_means = rows.groupby("seed").query_mae_improvement.mean()
            output.append(dict(zip(keys, group, strict=True)) | dict(
                draw_policy=policy, seed_draw_pairs=len(rows), model_seeds=len(seed_means),
                improved_seed_draws=rows.query_mae_improvement.gt(0).sum(), improved_seeds=seed_means.gt(0).sum(),
                query_mae_improvement=rows.query_mae_improvement.mean(),
                support_mse_improvement=rows.support_mse_improvement.mean(),
                energy_rank_99_difference=rows.energy_rank_99_difference.mean(),
                update_frobenius_difference=rows.update_frobenius_difference.mean(),
            ))
    return pd.DataFrame(output)


def summarize_tradeoff(episodes):
    rows = []
    for policy, frame in (("registered_draws", episodes), ("unique_support_sets", episodes[~episodes.is_duplicate])):
        frame = frame[frame.K.gt(0) & frame.query_cells.gt(0)]
        for key, group in frame.groupby(["domain", "method", "K", "window"], sort=True):
            support_better = group.support_mse_gain.gt(0)
            query_worse = group.query_mae_gain.lt(0)
            conflict = support_better & query_worse
            conflict_by_seed = group.assign(conflict=conflict).groupby("seed").conflict.mean()
            values = dict(zip(("domain", "method", "K", "window"), key, strict=True))
            rows.append(values | dict(
                draw_policy=policy, episodes=len(group), support_better=int(support_better.sum()),
                query_worse=int(query_worse.sum()), support_better_query_worse=int(conflict.sum()),
                support_better_query_strictly_better=int((support_better & group.query_mae_gain.gt(0)).sum()),
                support_better_query_worse_rate=conflict.mean(),
                seeds_with_majority_conflict=int(conflict_by_seed.gt(.5).sum()), model_seeds=len(conflict_by_seed),
            ))
    return pd.DataFrame(rows)


def decision(pair_summary, tradeoff, *, complete):
    if not complete:
        return dict(direction="PENDING", rank_checks=[], conflict_checks=[])
    rank_checks = pair_summary[
        pair_summary.draw_policy.eq("unique_support_sets") & pair_summary.window.eq("all")
        & pair_summary.method.eq("LORA_R16") & pair_summary.reference.eq("LORA_R2")
        & pair_summary.K.isin((3, 5))
    ].copy()
    rank_checks["pass"] = (
        rank_checks.query_mae_improvement.gt(0) & rank_checks.improved_seeds.eq(rank_checks.model_seeds)
        & rank_checks.improved_seed_draws.gt(rank_checks.seed_draw_pairs / 2)
        & rank_checks.energy_rank_99_difference.gt(0) & rank_checks.support_mse_improvement.gt(0)
    )
    conflict_checks = tradeoff[
        tradeoff.draw_policy.eq("unique_support_sets") & tradeoff.window.eq("all") & tradeoff.method.isin(PRIMARY)
    ].copy()
    conflict_checks["pass"] = conflict_checks.seeds_with_majority_conflict.ge(2)
    conflict_by_domain_k = conflict_checks.groupby(["domain", "K"])['pass'].sum()
    conflict_by_domain = conflict_by_domain_k.ge(2).groupby("domain").sum()
    check(len(rank_checks) == 4, "Incomplete rank-ladder decision coverage")
    if rank_checks["pass"].all():
        direction = "ADAPTIVE_RANK_EXPLORATION"
    elif conflict_by_domain.ge(2).any():
        direction = "RELIABILITY_CONSTRAINT_EXPLORATION"
    else:
        direction = "NO_ABBA_CENTERED_INNOVATION_SUPPORT"
    return dict(direction=direction, rank_checks=rank_checks.to_dict("records"),
                conflict_checks=conflict_checks.to_dict("records"))


def count_string(num, denom):
    return f"{num}/{denom}" if denom else "—"


def fmt(value, digits=6, signed=False):
    return format(value, f"{ '+' if signed else ''}.{digits}f") if pd.notna(value) else "—"


def markdown_table(headers, body):
    rows = [headers, ["---"] * len(headers), *body]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def historical_context(path, episodes_path, query, lock):
    anchor = read_json(path)
    check(anchor["status"] == "HISTORICAL_ANCHORS_IDENTITY_VERIFIED"
          and anchor["total_K_episodes_sha256"] == sha256(episodes_path)
          and all(lock["specialized_reference_sha256_by_seed"][str(receipt["seed"])] == receipt["checkpoint_sha256"]
                  for receipt in anchor["HUST_CAL2_receipts"]), "Historical CAL2 protocol differs")
    prior = anchor["historical_HUST_CAL2_K5_full_mean_mae"]
    current = query[query.domain.eq("HUST") & query.K.eq(5) & query.window.eq("all")
                    & query.method.isin(PRIMARY) & query.draw_policy.eq("registered_draws")
                    & query.scored_seed_draws.eq(15)]
    rows = [f"HUST 历史 CAL2 的 K=5 完整轨迹 MAE 为 {fmt(prior)}。既有审计核对了相同的域总 K 身份"
            "与专门化参考函数。其正则和适配目标沿用历史方案。"]
    if len(current):
        values = "，".join(f"{row.method} {fmt(row.query_mae - prior, signed=True)}" for row in current.itertuples())
        rows.append(f"本轮各已齐套方法的 MAE 减历史 CAL2 为 {values}。")
    return "\n\n".join(rows)


def abba_hypothesis_readout(pair_summary):
    primary = pair_summary[pair_summary.draw_policy.eq("unique_support_sets")
                           & pair_summary.method.eq("ABBA_R4X4") & pair_summary.reference.eq("LORA_R8")]
    full = primary[primary.window.eq("all")]
    late = primary[primary.window.eq("1001_plus")].set_index(["domain", "K"])
    lines = []
    for row in full.itertuples(index=False):
        late_improvement = late.loc[(row.domain, row.K), "query_mae_improvement"]
        lines.append(
            f"- {row.domain} K={row.K}，ABBA 减 LoRA-r8 的能量秩差为 {fmt(row.energy_rank_99_difference, 3, True)}，"
            f"更新范数差 {fmt(row.update_frobenius_difference, signed=True)}。"
            f"ABBA 在同一 Support 上的 MSE 改善为 {fmt(row.support_mse_improvement, signed=True)}，"
            f"Query 完整窗的配对改善为 {fmt(row.query_mae_improvement, signed=True)}，"
            f"远期为 {fmt(late_improvement, signed=True)}。"
        )
    high_rank = full[full.energy_rank_99_difference.gt(0)]
    if len(high_rank):
        gains = high_rank.query_mae_improvement.gt(0).sum()
        lines.append(f"- {len(high_rank)} 个域与 K 组合观测到 ABBA 能量秩更高，其中 {gains} 个组合完整窗 MAE 同时降低。"
                     "该比较反映共同预算下的总体差异，参数化同时改变秩、尺度和优化方向。")
    return "\n".join(lines)


def render_report(query, pair_summary, diagnostics, tradeoff, p0, lock, evidence, coverage, historical):
    status = evidence["status"]
    complete = status == "COMPLETE"
    direction = evidence["decision"]["direction"]
    explanation = {
        "PENDING": "存在未评分槽，暂缓 P3 判定。",
        "ADAPTIVE_RANK_EXPLORATION": "配对秩阶梯满足容量探索判据。可在新的独立工况上检验自适应 Rank。",
        "RELIABILITY_CONSTRAINT_EXPLORATION": "容量判据未通过，支持拟合和 Query 泛化的冲突达到探索判据。优先检验可靠性约束。",
        "NO_ABBA_CENTERED_INNOVATION_SUPPORT": "容量和拟合冲突均未达到预定探索判据。当前证据不足以把 ABBA 作为创新主线。",
    }[direction]
    report = [
        "# Moirai 上的 ABBA、真实少样本迁移与更新秩诊断",
        f"**{explanation}**",
        "## 评估范围",
        f"本轮已验证 {evidence['scored_slots']}/{len(coverage)} 个目标评分槽，状态 **{status}**。"
        "MATR 和 HUST 分别采用固定的 8 颗和 39 颗 Query，每次仅使用全域总 K=0、1、3、5 颗 Support。"
        "MATR K1 的 5 个登记 draw 包含 3 组不同 Support。全部方法共享专门化 K0、Bridge、Head 和原有前 100 圈输入。"
        "每颗电芯只按真实观测点评分，再取电芯、种子和 draw 的平均。目标 Query 既往已用于评分，本轮是固定队列上的回顾性对照。",
        "本轮复现公开算法的更新参数化和必要训练机制，并在同一 Moirai 适配协议下比较。"
        "[ABBA 原论文](https://arxiv.org/html/2505.14238v4)、"
        "[HiRA 原论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/48c368f105e8145b945227b73255635a-Paper-Conference.pdf)、"
        "[AdaLoRA 原论文](https://openreview.net/pdf?id=lq62uWRJjiY)",
        "## P0 真实 Moirai 复现",
        f"源核心合并前后的最大预测差为 {p0['merged_reference_max_abs']:.3g}。四种主方法均完成实际反向传播、"
        "同一 Moirai 的适配器保存回放和层权重合并校验。"
        f"源内锁定使用 {lock['source_score_slots']} 个评分槽。完整评估未重拟合 Bridge、Head 或共同主干。",
        markdown_table(["方法", "初始预测最大差", "状态回放最大差", "P0 适配后最大数值秩", "P0 末步活跃参数"], [
            [method, f"{p0['methods'][method]['initial_prediction_max_abs']:.3g}",
             f"{p0['methods'][method]['restored_prediction_max_abs']:.3g}",
             p0['methods'][method]['maximum_update_rank'], p0['methods'][method]['active_parameters']]
            for method in METHODS
        ]),
        "## P1 总 K 学习曲线",
        "ΔK 为同种子 K0 MAE 减适配后 MAE。负迁移采用严格增加。电芯记录和 episode 分母分别报告，重复记录共享物理电芯。"
        "远期窗为第 1001–5000 圈的真实观测点。其他分时距结果见 QUERY_SUMMARY.csv。",
    ]
    registered = query[query.draw_policy.eq("registered_draws")]
    if len(registered):
        far = registered[registered.window.eq("1001_plus")].set_index(["domain", "method", "K"])
        full = registered[registered.window.eq("all")]
        rows = []
        for row in full.itertuples():
            late = far.loc[(row.domain, row.method, row.K)]
            rows.append([
                row.domain, row.method, row.K, fmt(row.query_mae), fmt(row.query_mae_gain, signed=True),
                count_string(row.negative_cell_seed_draws, row.evaluated_cell_seed_draws),
                count_string(row.negative_seed_draws, row.scored_seed_draws), fmt(late.query_mae),
                fmt(late.query_mae_gain, signed=True), f"{late.physical_query_cells} / {late.physical_observations}",
            ])
        report.append(markdown_table([
            "域", "方法", "K", "完整 MAE", "完整 ΔK", "电芯记录负迁移", "episode 负迁移",
            "远期 MAE", "远期 ΔK", "远期电芯 / 观测点",
        ], rows))
    report.extend([historical, "## P2 更新容量与泛化"])
    report.append("ABBA、LoRA-r8 和 HiRA 的适配参数数目相同。AdaLoRA 单独列优化器中的物理参数和末步活跃参数，"
                  "其末步预算含至多 96 个奇异值参数，阈值同分删减可能使活跃预算低于上限。LoRA-r2 和 r16 检验预算梯度。"
                  "逐层秩从实际 ΔW 的奇异值计算，数值秩采用相对阈值 10⁻⁵，能量秩使用 99% Frobenius 能量，"
                  "stable rank 为平方 Frobenius 范数除以平方谱范数，entropy rank 使用归一化奇异值。"
                  "全模型 ΔW 范数为逐层 Frobenius 范数的平方和开根。")
    active = diagnostics[diagnostics.K.gt(0)] if len(diagnostics) else diagnostics
    if len(active):
        rows = []
        for key, group in active.groupby(["domain", "method", "K"]):
            active_min, active_max = group.active_parameters.min(), group.active_parameters.max()
            budget = str(active_min) if active_min == active_max else f"{active_min}–{active_max}"
            rows.append([*key, f"{group.trainable_parameters.min()} / {budget}",
                         fmt(group.mean_layer_numerical_rank_rtol_1e_5.mean(), 2),
                         fmt(group.mean_layer_energy_rank_99.mean(), 2), fmt(group.mean_layer_stable_rank.mean(), 2),
                         fmt(group.update_frobenius.mean()), fmt(group.support_mse_before.mean()),
                         fmt(group.support_mse_after.mean()), fmt(group.support_mse_gain.mean(), signed=True)])
        report.append(markdown_table([
            "域", "方法", "K", "优化 / 活跃参数", "平均数值秩", "平均能量秩", "平均 stable rank",
            "更新范数", "初始 Support MSE", "适配后 Support MSE", "Support MSE 改善",
        ], rows))
    if len(pair_summary):
        pairs = pair_summary[pair_summary.draw_policy.eq("unique_support_sets") & pair_summary.window.eq("all")]
        rows = [[row.domain, row.K, row.method, row.reference, fmt(row.query_mae_improvement, signed=True),
                 count_string(row.improved_seed_draws, row.seed_draw_pairs), count_string(row.improved_seeds, row.model_seeds),
                 fmt(row.energy_rank_99_difference, 3, True), fmt(row.update_frobenius_difference, signed=True)]
                for row in pairs.itertuples()]
        report.append("配对改善为对照 MAE 减候选 MAE。此表每个种子内只计不同 Support 集合，登记 draw、其余窗口和每个配对数值均有独立 CSV。")
        report.append(markdown_table([
            "域", "K", "候选", "对照", "配对完整 MAE 改善", "改善 seed/draw", "改善种子",
            "能量秩差", "更新范数差",
        ], rows))
        report.append(abba_hypothesis_readout(pair_summary))
    if len(tradeoff):
        conflicts = tradeoff[tradeoff.draw_policy.eq("unique_support_sets") & tradeoff.window.eq("all")]
        rows = [[row.domain, row.method, row.K, row.support_better, row.query_worse,
                 count_string(row.support_better_query_worse, row.episodes),
                 count_string(row.seeds_with_majority_conflict, row.model_seeds)] for row in conflicts.itertuples()]
        report.append(markdown_table([
            "域", "方法", "K", "Support MSE 降", "Query MAE 升", "同时出现的 episode", "多数 draw 冲突的种子",
        ], rows))
    report.extend([
        "## P3 研究方向",
        "容量探索要求 R16 比 R2 在两域的 K3、K5 完整窗中均降低 Query MAE，各种子的平均方向一致，"
        "过半不同 Support 集合的配对同向，同时观测到实际能量秩增加及 Support MSE 下降。"
        "若容量判据未通过，至少一个域的两个 K 中须各有两种主方法，在至少两个种子上多数不同 Support 集合出现拟合冲突，才优先可靠性。",
        "LoRA 秩阶梯沿用 r8 的源内选定学习率。秩、参数量、优化几何与更新范数可能共变，数值秩更高本身无法解释预测收益。"
        "同预算配对可核对表达能力与误差是否同时改善。固定 Query、共享 Support 及有限种子约束了跨队列外推。"
        + ("当前数值核验的置信度高，机制归因和新队列外推的置信度低。" if complete else "当前阶段仅报告已核验的槽。"),
    ])
    report.append(f"源内选择表的 SHA-256 为 `{lock['source_selection_sha256']}`。")
    return "\n\n".join(report) + "\n"


def validate_lock(public, config, episodes, config_sha, episodes_sha, lock):
    check(tuple(config["methods"]) == METHODS
          and config["target"]["seeds"] == episodes["training_seeds"]
          and config["target"]["shots"] == episodes["shots"], "Config axes differ from this analysis")
    check(lock["status"] == "LOCKED_BEFORE_TARGET_EVALUATION" and lock["config_sha256"] == config_sha
          and lock["episodes_sha256"] == episodes_sha and lock["target_tasks"] == episodes["target_tasks"]
          and lock["episode_contract_sha256"] == episodes["episode_contract_sha256"]
          and lock["target_seeds"] == episodes["training_seeds"]
          and lock["historical_target_query_exposure"] is True and lock["target_query_selection_uses"] == 0,
          "Target lock does not bind the registered evaluation")
    check(lock["source_score_slots"] == config["source"]["expected_score_slots"], "Source screen coverage changed")
    selection_path = public / "SOURCE_SELECTION.csv"
    check(sha256(selection_path) == lock["source_selection_sha256"], "Source selection checksum mismatch")
    scores = pd.read_csv(selection_path, float_precision="round_trip")
    candidates = scores.groupby(["method", "K"]).learning_rate.apply(set)
    check(candidates.index.tolist() == sorted(product(PRIMARY, (1, 3, 5)))
          and candidates.map(lambda rates: rates == set(config["source"]["learning_rates"])).all()
          and not scores.duplicated(["method", "K", "learning_rate"]).any()
          and scores["size"].sum() == lock["source_score_slots"] and np.isfinite(scores["mean"]).all(),
          "Source selection candidate matrix changed")
    for method, k in product(METHODS, (1, 3, 5)):
        compare = method if method in PRIMARY else "LORA_R8"
        ordered = scores[scores.method.eq(compare) & scores.K.eq(k)].sort_values(["mean", "learning_rate"])
        check(lock["learning_rates"][method][str(k)] == ordered.learning_rate.iloc[0],
              f"Target settings differ from source-only selection for {method} K={k}")
    paths = [f"fold{fold}/{method}" for fold, method in product(config["source"]["folds"], PRIMARY)]
    check(set(lock["source_receipt_sha256"]) == set(paths), "Source receipt axis differs")
    for path in paths:
        receipt_path = public / "source" / path / "COMPLETE.json"
        receipt = read_json(receipt_path)
        check(sha256(receipt_path) == lock["source_receipt_sha256"][path]
              and sha256(receipt_path.with_name("TRIALS.csv")) == receipt["scores_sha256"]
              and receipt["target_label_rows_loaded"] == 0, f"Invalid source-only receipt {path}")
    p0_path = public / "P0_MOIRAI_REPRODUCTION.json"
    p0 = read_json(p0_path)
    check(sha256(p0_path) == lock["p0_sha256"] and p0["status"] == "PASS_REAL_MOIRAI"
          and p0["implementation"] == {name: lock[name] for name in ("config_sha256", "code")}
          and set(METHODS) <= set(p0["methods"]) and p0["target_label_rows_loaded"] == 0,
          "P0 receipt or source-only implementation changed")
    return p0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=PUBLIC)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--episodes", type=Path, default=EPISODES)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--historical-anchors", type=Path, default=HISTORICAL)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    episodes = read_json(args.episodes)
    registered = registered_slots(episodes)
    lock_path = args.public_root / "TARGET_LOCK.json"
    lock = read_json(lock_path)
    lock_sha = sha256(lock_path)
    episodes_sha = sha256(args.episodes)
    config_sha = sha256(args.config)
    p0 = validate_lock(args.public_root, config, episodes, config_sha, episodes_sha, lock)
    barrier_sha = validate_prediction_barrier(args.public_root, registered, config, episodes, lock, lock_sha)
    metrics, spectra, episode_rows, coverage = collect(args.public_root, lock, lock_sha, registered)
    complete = len(metrics) == len(registered)
    check(complete or args.allow_partial, f"Scored coverage is {len(metrics)}/{len(registered)}")
    check(metrics, "No authenticated target score artifacts to summarize")
    diag = pd.DataFrame(episode_rows)
    check_budgets(diag)
    cells = pd.concat(metrics, ignore_index=True)
    per_episode = query_episodes(cells).merge(diag, on=KEY + ["is_duplicate", "support_set_sha256"],
                                             validate="many_to_one")
    query = summarize_query(per_episode)
    pairs = make_contrasts(per_episode)
    pair_summary = summarize_pairs(pairs)
    tradeoff = summarize_tradeoff(per_episode)
    evidence = {
        "status": "COMPLETE" if complete else "PARTIAL",
        "expected_slots": len(registered), "scored_slots": len(metrics),
        "scored_cell_window_rows": len(cells), "audited_spectrum_rows": sum(map(len, spectra)),
        "actual_target_optimizer_steps": int(diag.actual_optimizer_steps.sum()),
        "config_sha256": config_sha, "target_lock_sha256": lock_sha, "episodes_sha256": episodes_sha,
        "target_prediction_barrier_sha256": barrier_sha,
        "summary_code_sha256": sha256(Path(__file__)), "new_Query_label_array_reads": 0,
        "study_id": config.get("study_id"),
        "decision": decision(pair_summary, tradeoff, complete=complete),
    }
    filenames = {
        "TARGET_SLOT_COVERAGE.csv": coverage,
        "QUERY_SUMMARY.csv": query,
        "EPISODE_DIAGNOSTICS.csv": per_episode,
        "PAIRED_CONTRASTS.csv": pairs,
        "PAIRED_SUMMARY.csv": pair_summary,
        "SUPPORT_QUERY_TRADEOFF.csv": tradeoff,
    }
    evidence["artifacts_sha256"] = {}
    for name, table in filenames.items():
        path = args.public_root / name
        atomic_csv(path, table)
        evidence["artifacts_sha256"][name] = sha256(path)
    historical = historical_context(args.historical_anchors, args.episodes, query, lock) if args.historical_anchors.exists() else ""
    report = render_report(query, pair_summary, diag, tradeoff, p0, lock, evidence, coverage, historical)
    atomic_text(args.report, report)
    evidence["artifacts_sha256"][args.report.name] = sha256(args.report)
    atomic_text(args.public_root / "SUMMARY_VERIFICATION.json",
                json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"ABBA {evidence['status']}: {len(metrics)}/{len(registered)} slots. P3 {evidence['decision']['direction']}")


if __name__ == "__main__":
    main()
