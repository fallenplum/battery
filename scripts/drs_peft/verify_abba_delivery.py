"""Independently reconstruct ABBA predictions' published metrics from scored cell rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "results/drs_peft/abba_moirai_v1"
LEGACY = ROOT / "results/drs_peft/transfer_scope_v2"
PRIMARY = ("ABBA_R4X4", "LORA_R8", "HIRA_R8", "ADALORA_R12_TO_R8")
METHODS = (*PRIMARY, "LORA_R2", "LORA_R16")
PAIRS = (
    ("ABBA_R4X4", "LORA_R8"), ("ABBA_R4X4", "HIRA_R8"),
    ("ABBA_R4X4", "ADALORA_R12_TO_R8"),
    ("LORA_R8", "LORA_R2"), ("LORA_R16", "LORA_R8"), ("LORA_R16", "LORA_R2"),
)
WINDOWS = ("all", "101_200", "201_500", "501_1000", "1001_plus")
FOCUS = ("all", "1001_plus")


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def compare(value, captured, label):
    expected = float(captured) if captured else None
    if value is None:
        check(expected is None, f"Unexpected supported statistic {label}")
    else:
        check(expected is not None and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12),
              f"Different independently reconstructed statistic {label}")


def mean(values):
    values = list(values)
    return fmean(values) if values else None


def spectrum(path):
    numerical, energy, frobenius = [], [], []
    for row in records(path):
        singular_values = json.loads(row["singular_values_json"])
        weights = [s * s for s in singular_values]
        total = math.fsum(weights)
        numerical.append(sum(s > singular_values[0] * 1e-5 for s in singular_values))
        used = energy_count = 0
        if total:
            for value in weights:
                used += value
                energy_count += 1
                if used >= total * .99:
                    break
        energy.append(energy_count)
        frobenius.append(total)
    return {
        "layers": len(numerical),
        "mean_numerical_rank": mean(numerical),
        "mean_energy_rank_99": mean(energy),
        "update_frobenius": math.sqrt(math.fsum(frobenius)),
    }


def raw_slots(public, lock):
    by_key, checksums = {}, []
    observations = cells = 0
    seed_axis = lock["target_seeds"]
    for task in lock["target_tasks"]:
        domain = task["domain"]
        for seed, shot in product(seed_axis, task["shots"]):
            k = shot["k"]
            for draw_record, method in product(shot["draws"], METHODS if k else ("ZERO",)):
                draw = draw_record["draw"]
                key = domain, seed, method, k, draw
                folder = public / "target" / domain / f"s{seed}" / method / f"{k}_{draw}"
                receipt = read_json(folder / "RECEIPT.json")
                raw = folder / "METRICS.csv"
                spectral = folder / "UPDATE_SPECTRUM.csv"
                duplicate = draw_record["duplicate_of_draw"] is not None
                check(receipt["status"] == "SCORED" and receipt["query_ids"] == task["query_ids"]
                      and sorted(receipt["support_ids"]) == sorted(draw_record["support_ids"])
                      and receipt["is_duplicate"] == duplicate and sha256(raw) == receipt["metrics_sha256"]
                      and sha256(spectral) == receipt["update_spectrum_sha256"], f"Bad raw cell receipt {key}")
                checksums.append(receipt["metrics_sha256"])
                slots = defaultdict(dict)
                for row in records(raw):
                    uid, window = row["cell_uid"], row["window"]
                    check(uid not in slots[window] and uid in task["query_ids"], f"Unexpected cell row {key}")
                    n = int(row["n_obs"])
                    mae = float(row["mae"]) if n else None
                    zero = float(row["zero_mae"]) if n else None
                    slots[window][uid] = n, mae, zero
                    observations += n
                    cells += 1
                check(set(slots) == set(WINDOWS) and all(set(rows) == set(task["query_ids"]) for rows in slots.values()),
                      f"Missing physical cells or windows {key}")
                ranks = spectrum(spectral)
                check(ranks["layers"] == (12 if k else 0), f"Missing update layer {key}")
                by_key[key] = dict(
                    receipt, windows=slots, **ranks,
                    support_gain=(receipt["support_mse_before"] - receipt["support_mse_after"]) if k else None,
                )
    return by_key, {"slot_count": len(by_key), "cell_window_count": cells, "observation_repetitions": observations,
                   "metric_checksums_sha256": hashlib.sha256("".join(checksums).encode("ascii")).hexdigest()}


def tables(raw):
    output = defaultdict(list)
    for (domain, seed, method, k, draw), episode in raw.items():
        base = raw[domain, seed, "ZERO", 0, 0]
        for window, cells in episode["windows"].items():
            errors, gains, points = [], [], 0
            for cell, (n, mae, zero) in cells.items():
                count, reference, _ = base["windows"][window][cell]
                check(count == n and reference == zero, "Score mask or cell baseline changed")
                if n:
                    errors.append(mae)
                    gains.append(reference - mae)
                    points += n
            for draw_policy in ("registered_draws", "unique_support_sets"):
                if episode["is_duplicate"] and draw_policy == "unique_support_sets":
                    continue
                output[domain, method, k, window, draw_policy].append(dict(
                    seed=seed, draw=draw, mae=mean(errors), gain=mean(gains), points=points,
                    query_cells=len(errors), negative_cells=sum(gain < 0 for gain in gains),
                    support_gain=episode["support_gain"], numerical_rank=episode["mean_numerical_rank"],
                    energy_rank_99=episode["mean_energy_rank_99"], norm=episode["update_frobenius"],
                    support_mse_after=episode["support_mse_after"], optimizer_steps=episode["actual_optimizer_steps"],
                ))
    return output


def winning_seeds(pairs):
    groups = defaultdict(list)
    for seed, score in pairs:
        groups[seed].append(score)
    return sum(fmean(values) > 0 for values in groups.values()), len(groups)


def verify_curves(public, groups):
    summary = {}
    checked = set()
    for row in records(public / "QUERY_SUMMARY.csv"):
        domain, method, k, window, policy = row["domain"], row["method"], int(row["K"]), row["window"], row["draw_policy"]
        key = domain, method, k, window, policy
        check(key not in checked, f"Duplicate summary group {key}")
        checked.add(key)
        episodes = [item for item in groups[key] if item["query_cells"]]
        wins, seeds = winning_seeds((item["seed"], item["gain"]) for item in episodes)
        info = dict(mae=mean(item["mae"] for item in episodes), gain=mean(item["gain"] for item in episodes),
                    negative_cells=sum(item["negative_cells"] for item in episodes),
                    negative_episodes=sum(item["gain"] < 0 for item in episodes),
                    episodes=len(episodes), winning_episodes=sum(item["gain"] > 0 for item in episodes),
                    winning_seeds=wins, seeds=seeds, query_cells=max((item["query_cells"] for item in episodes), default=0))
        for metric, column in dict(mae="query_mae", gain="query_mae_gain", negative_cells="negative_cell_seed_draws",
                                   negative_episodes="negative_seed_draws", episodes="scored_seed_draws",
                                   winning_seeds="improved_seed_means", seeds="model_seeds", query_cells="physical_query_cells").items():
            compare(info[metric], row[column], (key, metric))
        if window in FOCUS and policy == "registered_draws":
            summary[f"{domain} K{k} {method} {window}"] = info
    check(checked == set(groups), "Raw group and delivered curve rosters differ")
    return summary, len(checked)


def verify_pairs(public, groups):
    output = {}
    checked = set()
    for row in records(public / "PAIRED_SUMMARY.csv"):
        domain, k, window, policy = row["domain"], int(row["K"]), row["window"], row["draw_policy"]
        method, baseline = row["method"], row["reference"]
        key = domain, k, window, policy, method, baseline
        check((method, baseline) in PAIRS and key not in checked, f"Invalid pair group {key}")
        checked.add(key)
        left = groups[domain, method, k, window, policy]
        right = {(item["seed"], item["draw"]): item for item in groups[domain, baseline, k, window, policy]}
        effects = []
        for item in left:
            base = right[item["seed"], item["draw"]]
            check(item["query_cells"] == base["query_cells"], "Paired Query coverage differs")
            if item["query_cells"]:
                effects.append((item["seed"], base["mae"] - item["mae"],
                                base["support_mse_after"] - item["support_mse_after"],
                                item["energy_rank_99"] - base["energy_rank_99"], item["norm"] - base["norm"]))
        wins, seeds = winning_seeds((seed, delta) for seed, delta, *_ in effects)
        info = dict(improvement=mean(item[1] for item in effects), wins=sum(item[1] > 0 for item in effects),
                    pairs=len(effects), winning_seeds=wins, seeds=seeds, support_mse_delta=mean(item[2] for item in effects),
                    energy_rank_delta=mean(item[3] for item in effects), update_frobenius_delta=mean(item[4] for item in effects))
        for metric, column in dict(improvement="query_mae_improvement", wins="improved_seed_draws", pairs="seed_draw_pairs",
                                   winning_seeds="improved_seeds", seeds="model_seeds", support_mse_delta="support_mse_improvement",
                                   energy_rank_delta="energy_rank_99_difference", update_frobenius_delta="update_frobenius_difference").items():
            compare(info[metric], row[column], (key, metric))
        if window in FOCUS and policy == "unique_support_sets":
            output[f"{domain} K{k} {method} vs {baseline} {window}"] = info
    check(len(checked) == 2 * 3 * len(WINDOWS) * 2 * len(PAIRS), "Delivered pair groups have missing combinations")
    return output, len(checked)


def verify_conflicts(public, groups):
    result = {}
    checked = set()
    for row in records(public / "SUPPORT_QUERY_TRADEOFF.csv"):
        domain, method, k, window, policy = row["domain"], row["method"], int(row["K"]), row["window"], row["draw_policy"]
        key = domain, method, k, window, policy
        check(key not in checked and k > 0, f"Invalid conflict group {key}")
        checked.add(key)
        episodes = [item for item in groups[key] if item["query_cells"]]
        by_seed = defaultdict(list)
        for item in episodes:
            by_seed[item["seed"]].append(item["support_gain"] > 0 and item["gain"] < 0)
        info = dict(episodes=len(episodes), support_better=sum(item["support_gain"] > 0 for item in episodes),
                    query_worse=sum(item["gain"] < 0 for item in episodes),
                    support_better_query_worse=sum(sum(values) for values in by_seed.values()),
                    seeds_with_majority_conflict=sum(fmean(values) > .5 for values in by_seed.values()))
        for item, value in info.items():
            compare(value, row[item], (key, item))
        if window == "all" and policy == "unique_support_sets":
            result[f"{domain} K{k} {method}"] = info | dict(
                mean_numerical_rank=mean(item["numerical_rank"] for item in episodes),
                mean_energy_rank_99=mean(item["energy_rank_99"] for item in episodes),
                mean_update_frobenius=mean(item["norm"] for item in episodes),
            )
    check(len(checked) == 2 * 3 * len(WINDOWS) * 2 * len(METHODS), "Missing conflict groups")
    return result, len(checked)


def verify_episode_diagnostics(public, raw):
    checked = set()
    for row in records(public / "EPISODE_DIAGNOSTICS.csv"):
        key = row["domain"], int(row["seed"]), row["method"], int(row["K"]), int(row["draw"])
        window_key = key, row["window"]
        check(window_key not in checked, f"Duplicate episode diagnostics {window_key}")
        checked.add(window_key)
        episode = raw[key]
        counts = dict(
            mean_layer_numerical_rank_rtol_1e_5=episode["mean_numerical_rank"] or 0,
            mean_layer_energy_rank_99=episode["mean_energy_rank_99"] or 0,
            update_frobenius=episode["update_frobenius"], support_mse_before=episode["support_mse_before"],
            support_mse_after=episode["support_mse_after"], support_mse_gain=episode["support_gain"],
            trainable_parameters=episode["trainable_parameters"], active_parameters=episode["active_parameters"],
            optimizer_steps=episode["optimizer_steps"], actual_optimizer_steps=episode["actual_optimizer_steps"],
        )
        for field, value in counts.items():
            compare(value, row[field], (window_key, field))
    check(len(checked) == len(raw) * len(WINDOWS), "Missing diagnostic rows")
    return len(checked)


def history(lock, curves):
    task = next(task for task in lock["target_tasks"] if task["domain"] == "HUST")
    legacy = read_json(LEGACY / "HUST_LEGACY_CAL2_REUSE.json")
    identities = {(shot["k"], draw["draw"]): draw["support_ids"] for shot in task["shots"] for draw in shot["draws"]}
    check(len(legacy["slots"]) == 45 and all(
        slot["support_ids"] == identities[slot["K"], slot["draw"]] and slot["query_ids"] == task["query_ids"]
        and slot["source_checkpoint_sha256"] == lock["specialized_reference_sha256_by_seed"][str(slot["seed"])]
        for slot in legacy["slots"]), "Historical HUST CAL2 and current reference differ")
    raw = LEGACY / "METRICS.csv"
    k5 = [float(row["mae"]) for row in records(raw) if row["level"] == "cell" and row["method"] == "HUST_LEGACY_CAL2"
          and row["window"] == "all" and float(row["K"]) == 5]
    check(len(k5) == 585, "Historical HUST CAL2 coverage changed")
    anchor = fmean(k5)
    return dict(k5_query_mae=anchor, cell_seed_draws=len(k5), metrics_sha256=sha256(raw),
                current_minus_historical={m: curves[f"HUST K5 {m} all"]["mae"] - anchor for m in METHODS})


def reconstruct_direction(pairs, conflicts):
    conditions = []
    for domain, k in product(("MATR", "HUST"), (3, 5)):
        evidence = pairs[f"{domain} K{k} LORA_R16 vs LORA_R2 all"]
        conditions.append(evidence["improvement"] > 0 and evidence["winning_seeds"] == evidence["seeds"]
                          and evidence["wins"] * 2 > evidence["pairs"] and evidence["energy_rank_delta"] > 0
                          and evidence["support_mse_delta"] > 0)
    if all(conditions):
        return "ADAPTIVE_RANK_EXPLORATION"
    for domain in ("MATR", "HUST"):
        affected_shots = sum(sum(conflicts[f"{domain} K{k} {m}"]["seeds_with_majority_conflict"] >= 2
                                 for m in PRIMARY) >= 2 for k in (1, 3, 5))
        if affected_shots >= 2:
            return "RELIABILITY_CONSTRAINT_EXPLORATION"
    return "NO_ABBA_CENTERED_INNOVATION_SUPPORT"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=PUBLIC)
    args = parser.parse_args()
    public = args.public_root
    lock_path = public / "TARGET_LOCK.json"
    verification = public / "SUMMARY_VERIFICATION.json"
    summary = read_json(verification)
    check(summary["status"] == "COMPLETE" and summary["target_lock_sha256"] == sha256(lock_path),
          "A complete delivered summary with a matching target lock is required")
    for name in ("QUERY_SUMMARY.csv", "PAIRED_SUMMARY.csv", "SUPPORT_QUERY_TRADEOFF.csv", "EPISODE_DIAGNOSTICS.csv"):
        check(summary["artifacts_sha256"][name] == sha256(public / name), f"Delivered summary changed {name}")
    lock = read_json(lock_path)
    raw, coverage = raw_slots(public, lock)
    check(coverage["slot_count"] == summary["scored_slots"] == 546
          and coverage["cell_window_count"] == summary["scored_cell_window_rows"]
          and sum(item["layers"] for item in raw.values()) == summary["audited_spectrum_rows"]
          and sum(item["actual_optimizer_steps"] for item in raw.values()) == summary["actual_target_optimizer_steps"],
          "Delivered raw coverage changed")
    diagnostic_count = verify_episode_diagnostics(public, raw)
    groups = tables(raw)
    curves, curve_count = verify_curves(public, groups)
    paired, pair_count = verify_pairs(public, groups)
    conflicts, conflict_count = verify_conflicts(public, groups)
    direction = reconstruct_direction(paired, conflicts)
    check(summary["decision"]["direction"] == direction, "Independent direction and main report differ")
    hardware = defaultdict(set)
    for (domain, seed, _, k, _), episode in raw.items():
        if k:
            hardware[f"{domain} seed{seed}"].add(episode["hardware"])
    evidence = dict(
        status="VERIFIED_INDEPENDENT_RAW_CELL_RECONSTRUCTION", coverage=coverage,
        target_lock_sha256=sha256(lock_path), summary_verification_sha256=sha256(verification),
        independent_code_sha256=sha256(Path(__file__)), curve_groups_verified=curve_count,
        paired_groups_verified=pair_count, support_query_groups_verified=conflict_count,
        episode_window_diagnostics_verified=diagnostic_count,
        learning_curves_full_and_tail_registered_draws=curves, paired_comparisons_full_and_tail_unique_sets=paired,
        full_window_fit_conflict_and_spectra_unique_sets=conflicts, historical_hust_cal2=history(lock, curves),
        direction=direction, new_query_label_array_reads=0, optimizer_updates=0,
        target_hardware_by_domain_seed={key: sorted(value) for key, value in hardware.items()},
        claim_boundary="固定 Query 队列已有历史评分。不同种子和抽样重复评价相同电芯。"
                       "秩、更新幅度与优化过程可能共变，源内选参和固定 30 步的证据限于本轮矩阵。"
                       "Support MSE 每颗电芯最多采用 512 个观测点，并按电芯等权。",
    )
    target = public / "INDEPENDENT_REVIEW.json"
    pending = target.with_suffix(".pending.json")
    pending.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(target)
    print(f"Verified {len(raw)} slots, {curve_count} curves, {pair_count} contrasts, {conflict_count} tradeoffs. {direction}")


if __name__ == "__main__":
    main()
