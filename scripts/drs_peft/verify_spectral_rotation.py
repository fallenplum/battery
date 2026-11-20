"""Independently check the locked source spectral-rotation delivery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = "drs_peft/spectral_rotation_v1"
WINDOWS = (
    ("all", 101, 5000),
    ("101_200", 101, 200),
    ("201_500", 201, 500),
    ("501_1000", 501, 1000),
    ("1001_plus", 1001, 5000),
)
WINDOW_NAMES = tuple(name for name, _, _ in WINDOWS)
CANDIDATE = "ROTATION_TRUE"
SAME_CORE = "ROTATION_SAME_CORE_SHUFFLED"
MAPPERS = {"DRS", "GAIN", "ROTATION_CONST", CANDIDATE, "ROTATION_SHUFFLED"}
PARAMETER_VIEWS = {
    "STATIC_CORE",
    "DRS",
    "GAIN",
    "JOINT2R",
    "ROTATION_CONST",
    "ROTATION_SHUFFLED",
}
TABLES = {"CELL_SCORES.csv", "EPISODE_SCORES.csv", "PAIR_SCORES.csv"}
SUMMARY_FILES = {
    "CELL_SCORES.csv",
    "EPISODE_SCORES.csv",
    "PAIRED_EPISODE_SCORES.csv",
    "FOLD_SCORES.csv",
    "METHOD_SCORES.csv",
    "PAIR_SCORES.csv",
    "COSTS.csv",
    "CELL_DRAW_MEANS.csv",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_bytes())


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authenticate(root, inventory, required):
    check(set(inventory) == set(required), f"Unexpected file set in {root}")
    for name, expected in inventory.items():
        check(
            digest(root / name) == expected, f"Fingerprint mismatch for {root / name}"
        )


def number(value):
    return None if value in (None, "") else float(value)


def compare(result, value, location):
    expected = number(value)
    if result is None:
        check(expected is None, f"Unexpected observed metric at {location}")
    else:
        check(
            expected is not None
            and math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12),
            f"Different independently reconstructed metric at {location}",
        )


def yes(value):
    check(value in ("True", "False"), "Unsupported Boolean score field")
    return value == "True"


def mean(values):
    values = list(values)
    return fmean(values) if values else None


def role_ids(episodes):
    return {
        uid for row in episodes for name in ("support", "query") for uid in row[name]
    }


def verify_plan(plan, cfg, registration):
    source, sealed = map(
        set, (registration["source_ids"], registration["sealed_final_MATR_ids"])
    )
    protocols = registration["source_protocols"]
    check(
        len(source) == 115
        and not source & sealed
        and cfg["domain"] == "MATR"
        and cfg["seed"] == 42
        and cfg["shot"] == 3
        and cfg["folds"] == [0, 1, 2]
        and len(cfg["methods"]) == len(set(cfg["methods"])) == 11
        and not cfg["target_label_access"]
        and not cfg["outer_label_access_before_lock"],
        "Source experiment axes changed",
    )
    check(
        plan["historically_used_source_population"] is True
        and plan["status"] == "REGISTERED_WITHOUT_OUTER_LABELS"
        and [row["fold"] for row in plan["folds"]] == cfg["folds"],
        "Bad registered source plan",
    )
    unique_cells, unique_protocols, registered = set(), set(), {}
    for row in plan["folds"]:
        fold, fit = row["fold"], set(row["fit_ids"])
        check(
            len(fit) == len(row["fit_ids"])
            and len(row["train"]) == cfg["train_episodes"]
            and role_ids(row["train"]) <= fit
            and len(row["anchors"]) == cfg["anchor_cells"]
            and len(set(row["anchors"])) == len(row["anchors"])
            and set(row["anchors"]) <= fit,
            f"Invalid development role in fold {fold}",
        )
        role_sets = [fit, role_ids(row["validation"]), role_ids(row["outer"])]
        check(
            all(ids <= source and not ids & sealed for ids in role_sets),
            f"Sealed cell in fold {fold}",
        )
        group_sets = [{protocols[uid] for uid in ids} for ids in role_sets]
        check(
            all(not l & r for l, r in combinations(group_sets, 2)),
            f"Fold-local protocol crossover {fold}",
        )
        for name in ("train", "validation", "outer"):
            for episode in row[name]:
                support, query = map(set, (episode["support"], episode["query"]))
                check(
                    len(support) == len(episode["support"]) == cfg["shot"]
                    and len(query) == len(episode["query"])
                    and not support & query
                    and not {protocols[uid] for uid in support}
                    & {protocols[uid] for uid in query},
                    f"Bad Support/Query axis in {name} fold {fold}",
                )
        group = next(
            task
            for task in registration["source_tasks"]["outer"]
            if task["fold_id"] == fold
        )
        query = set(group["query_ids"])
        query_groups = {protocols[uid] for uid in query}
        check(
            len(query) == 20
            and not query & unique_cells
            and not query_groups & unique_protocols
            and [episode["draw"] for episode in row["outer"]] == [0, 1, 2]
            and len({tuple(episode["support"]) for episode in row["outer"]}) == 3,
            f"Invalid outer fold roster {fold}",
        )
        unique_cells.update(query)
        unique_protocols.update(query_groups)
        for episode in row["outer"]:
            check(
                episode["query"] == group["query_ids"]
                and set(episode["support"]) <= set(group["support_pool_ids"]),
                "Outer fold episode differs from its parent role",
            )
            registered[fold, episode["draw"]] = episode, row["anchors"]
    check(
        len(registered) == 9
        and len(unique_cells) == plan["outer_source_query_cells"] == 60
        and len(unique_protocols) == plan["outer_source_query_protocols"],
        "Different source screen coverage",
    )
    return registered, unique_cells, protocols


def verify_lock(args, cfg, plan, identity, completion):
    public, private = args.public, args.private
    lock_path = public / "DEVELOPMENT_LOCK.json"
    lock = read_json(lock_path)
    lock_sha = digest(lock_path)
    check(
        lock["status"] == "LOCKED_ALL_FOLDS_BEFORE_OUTER_ROLE_SCORING"
        and lock["run_identity"] == identity
        and lock["outer_role_future_label_reads_before_lock"]
        == lock["target_future_label_reads"]
        == 0
        and completion["development_lock_sha256"] == lock_sha,
        "Bad lock ancestry",
    )
    dev_hashes = []
    for fold_plan in plan["folds"]:
        fold = fold_plan["fold"]
        root = public / "development" / f"fold{fold}"
        storage = private / "development" / f"fold{fold}"
        receipt = read_json(root / "COMPLETE.json")
        check(
            receipt["status"] == "COMPLETE_SOURCE_DEVELOPMENT_FOLD"
            and receipt["fold"] == fold
            and receipt["run_identity"] == identity
            and receipt["outer_role_label_rows_loaded"]
            == receipt["target_label_rows_loaded"]
            == 0
            and receipt["fit_label_ids"] == sorted(fold_plan["fit_ids"])
            and receipt["validation_label_ids"]
            == sorted(role_ids(fold_plan["validation"]))
            and receipt["validation_draws"] == len(fold_plan["validation"]),
            f"Bad development receipt fold {fold}",
        )
        authenticate(
            root,
            receipt["files"],
            {
                "META_TRACE.csv",
                "VALIDATION_TRIALS.csv",
                "BASIS_AUDIT.json",
                "DESCRIPTOR_SCALER.json",
            },
        )
        check(
            set(receipt["state_sha256"]) == MAPPERS,
            f"Wrong development mapper roster fold {fold}",
        )
        authenticate(
            storage,
            {f"{name}.pt": sha for name, sha in receipt["state_sha256"].items()},
            {f"{name}.pt" for name in MAPPERS},
        )
        scaler = read_json(root / "DESCRIPTOR_SCALER.json")
        check(
            scaler["fit_ids"] == fold_plan["fit_ids"],
            f"Scaler role differs at fold {fold}",
        )
        dev_hashes.append({"fold": fold, "sha256": digest(root / "COMPLETE.json")})
    check(
        lock["source_development_receipts"] == dev_hashes,
        "Development lock does not bind all fold receipts",
    )
    return lock_sha


def read_predictions(path, methods, episode, anchors):
    wanted = episode["support"] + episode["query"] + anchors
    with np.load(path, allow_pickle=False) as archive:
        check(
            archive["ids"].tolist() == wanted
            and archive["methods"].tolist() == methods
            and np.array_equal(archive["cycles"], np.arange(101, 5001)),
            f"Bad prediction axes {path}",
        )
        predictions = archive["predictions"]
        pair_names = archive["parameter_matched_methods"].tolist()
        pairs = archive["parameter_matched_predictions"]
        radii = archive["parameter_matched_radii"]
    check(
        predictions.shape == (len(methods), len(wanted), 4900)
        and np.isfinite(predictions).all()
        and len(pair_names) == len(set(pair_names))
        and set(pair_names) <= PARAMETER_VIEWS
        and radii.shape == (len(pair_names),)
        and np.isfinite(radii).all()
        and (radii > 0).all(),
        f"Invalid prediction array {path}",
    )
    if pair_names:
        check(
            pairs.shape == (len(pair_names), 2, len(wanted), 4900)
            and np.isfinite(pairs).all(),
            f"Invalid matched prediction array {path}",
        )
    else:
        check(pairs.size == 0, "Unexpected matched prediction values")
    by_method = dict(zip(methods, predictions, strict=True))
    k = len(episode["support"])
    check(
        np.array_equal(by_method[CANDIDATE][:k], by_method[SAME_CORE][:k]),
        f"Same-core shuffle affected Support {path}",
    )
    return by_method, {
        name: (value, float(radius))
        for name, value, radius in zip(pair_names, pairs, radii, strict=True)
    }


def identity(row, cfg):
    check(
        int(row["seed"]) == cfg["seed"] and int(row["K"]) == cfg["shot"],
        "Bad scoring seed or shot",
    )
    return int(row["fold"]), int(row["draw"]), row["method"], row["window"]


def index_rows(tables, cfg):
    cells, episodes, pairs = {}, {}, {}
    for row in tables["CELL_SCORES.csv"]:
        key = (*identity(row, cfg), row["cell_uid"])
        check(key not in cells, f"Repeated cell score {key}")
        cells[key] = row
    for row in tables["EPISODE_SCORES.csv"]:
        key = identity(row, cfg)
        check(key not in episodes, f"Repeated episode score {key}")
        episodes[key] = row
    for row in tables["PAIR_SCORES.csv"]:
        key = (*identity(row, cfg), row["reference_method"])
        check(key not in pairs, f"Repeated pair score {key}")
        pairs[key] = row
    return cells, episodes, pairs


def verify_outer(args, cfg, plan, completion, lock_sha, registered):
    collected = {name: [] for name in TABLES}
    private_paths = {}
    check(
        set(completion["outer_receipts_sha256"]) == set(map(str, cfg["folds"])),
        "Outer receipt axis differs",
    )
    for row in plan["folds"]:
        fold = row["fold"]
        root = args.public / "outer" / f"fold{fold}"
        store = args.private / "outer" / f"fold{fold}"
        path = root / "COMPLETE.json"
        receipt = read_json(path)
        check(
            digest(path) == completion["outer_receipts_sha256"][str(fold)]
            and receipt["status"] == "COMPLETE_LOCKED_SOURCE_OUTER_FOLD"
            and receipt["fold"] == fold
            and receipt["development_lock_sha256"] == lock_sha
            and receipt["target_label_rows_loaded"] == 0
            and receipt["outer_draws"] == 3
            and receipt["outer_role_label_ids"] == sorted(role_ids(row["outer"])),
            f"Bad locked outer receipt {fold}",
        )
        authenticate(root, receipt["files"], TABLES)
        predict_names = {f"DRAW{draw}_PREDICTIONS.npz" for draw in (0, 1, 2)}
        authenticate(store, receipt["private_predictions"], predict_names)
        for name in TABLES:
            group = list(records(root / name))
            check(
                all(int(value["fold"]) == fold for value in group),
                f"Cross-fold score rows {name}",
            )
            collected[name].extend(group)
        for draw in (0, 1, 2):
            private_paths[fold, draw] = store / f"DRAW{draw}_PREDICTIONS.npz"
    check(
        set(private_paths) == set(registered),
        "Private predictions do not cover the registered episodes",
    )
    return index_rows(collected, cfg), private_paths


def read_source_truth(data_root, uids, registration):
    allowed = {
        uid
        for task in registration["source_tasks"]["outer"]
        for uid in task["query_ids"]
    }
    check(
        set(uids) == allowed
        and len(uids) == len(allowed) == 60
        and not allowed & set(registration["sealed_final_MATR_ids"]),
        "Replay allowlist differs from the registered source Query",
    )
    from scripts.drs_peft.transfer_scope_data import load_future_rows

    truth = {}
    for task in registration["source_tasks"]["outer"]:
        ids = task["query_ids"]
        check(
            len(set(ids)) == len(ids) == 20 and not set(ids) & set(truth),
            "Bad replay fold roster",
        )
        target, observed = load_future_rows("MATR", data_root, ids)
        truth.update(zip(ids, zip(target, observed, strict=True), strict=True))
    return truth


def window_mae(prediction, ids, truth, lower, upper):
    errors = []
    for index, uid in enumerate(ids):
        labels, mask = truth[uid]
        select = slice(lower - 101, upper - 100)
        observed = mask[select]
        error = np.asarray(prediction[index, select], dtype=np.float64) - labels[select]
        errors.append(
            (
                int(observed.sum()),
                float(np.abs(error[observed]).mean()) if observed.any() else None,
            )
        )
    return errors


def verify_cases(cfg, registered, protocols, tables, prediction_paths, truth):
    cells, episodes, pairs = tables
    expected_cells, expected_episodes, expected_pairs = set(), set(), set()
    correction_count = 0
    for (fold, draw), (ep, anchors) in registered.items():
        predictions, matched = read_predictions(
            prediction_paths[fold, draw], cfg["methods"], ep, anchors
        )
        query = ep["query"]
        starts, stops = len(ep["support"]), len(ep["support"]) + len(query)
        slab = slice(starts, stops)
        ref = predictions["K0"].astype(np.float64)
        radii = {
            name: float(
                np.sqrt(np.mean((value[stops:].astype(np.float64) - ref[stops:]) ** 2))
            )
            for name, value in predictions.items()
        }
        for window, low, high in WINDOWS:
            for name, prediction in predictions.items():
                key = fold, draw, name, window
                expected_episodes.add(key)
                summary = episodes[key]
                compare(radii[name], summary["anchor_radius"], (*key, "anchor"))
                values, counts = [], []
                replay = (
                    window_mae(prediction[slab], query, truth, low, high)
                    if truth
                    else None
                )
                for index, uid in enumerate(query):
                    expected_cells.add((*key, uid))
                    row = cells[*key, uid]
                    n, mae = int(row["n_obs"]), number(row["mae"])
                    check(
                        row["protocol_uid"] == protocols[uid]
                        and 0 <= n <= high - low + 1
                        and bool(n) == (mae is not None)
                        and (window != "all" or n > 0),
                        f"Bad observation axis {key} {uid}",
                    )
                    frozen = cells[fold, draw, "K0", window, uid]
                    check(
                        n == int(frozen["n_obs"]),
                        "Common observation mask counts changed",
                    )
                    if n:
                        check(math.isfinite(mae) and mae >= 0, "Invalid scored MAE")
                        compare(
                            float(frozen["mae"]) - mae,
                            row["mae_gain"],
                            (*key, uid, "gain"),
                        )
                        values.append(mae)
                        counts.append(n)
                    if replay:
                        check(
                            replay[index][0] == n,
                            f"Independent observation count differs at {key} {uid}",
                        )
                        compare(replay[index][1], row["mae"], (*key, uid, "replay"))
                check(
                    sum(counts) == int(summary["n_obs"])
                    and len(values) == int(summary["n_cells"]),
                    f"Episode counts differ at {key}",
                )
                compare(mean(values), summary["mae"], (*key, "macro"))
            for name in (
                method for method in cfg["methods"] if method not in {"K0", CANDIDATE}
            ):
                key = fold, draw, CANDIDATE, window, name
                expected_pairs.add(key)
                pair = pairs[key]
                left, right = (
                    episodes[fold, draw, CANDIDATE, window],
                    episodes[fold, draw, name, window],
                )
                count = int(left["n_obs"])
                check(
                    int(pair["n_obs"]) == count
                    and int(pair["n_cells"]) == int(left["n_cells"]),
                    f"Pair denominator differs at {key}",
                )
                compare(
                    float(left["mae"]) - float(right["mae"]) if count else None,
                    pair["raw_candidate_minus_reference_mae"],
                    key,
                )
                for prefix, candidate in (
                    ("raw_candidate", left),
                    ("raw_reference", right),
                ):
                    compare(
                        number(candidate["mae"]), pair[f"{prefix}_mae"], (*key, prefix)
                    )
                radius = min(radii[CANDIDATE], radii[name])
                check(
                    yes(pair["anchor_match_resolved"]) == (radius > 1e-12)
                    and yes(pair["parameter_match_resolved"]) == (name in matched),
                    f"Bad matched axes at {key}",
                )
                compare(
                    radius if radius > 1e-12 else None,
                    pair["common_anchor_radius"],
                    (*key, "common anchor"),
                )
                compare(
                    matched[name][1] if name in matched else None,
                    pair["common_weight_frobenius"],
                    (*key, "common weight"),
                )
                for side, method in (("candidate", CANDIDATE), ("reference", name)):
                    eta = radius / radii[method] if radius > 1e-12 else None
                    compare(eta, pair[f"{side}_eta"], (*key, side))
                    if truth and eta is not None:
                        prediction = ref[slab] + eta * (
                            predictions[method][slab] - ref[slab]
                        )
                        observed = window_mae(prediction, query, truth, low, high)
                        compare(
                            mean(mae for n, mae in observed if n),
                            pair[f"matched_{side}_mae"],
                            (*key, "anchor matched replay", side),
                        )
                if truth and name in matched:
                    for side, prediction in zip(
                        ("candidate", "reference"), matched[name][0], strict=True
                    ):
                        observed = window_mae(prediction[slab], query, truth, low, high)
                        compare(
                            mean(mae for n, mae in observed if n),
                            pair[f"parameter_matched_{side}_mae"],
                            (*key, "parameter matched replay", side),
                        )
                for prefix, field in (
                    ("matched", "matched_candidate_minus_reference_mae"),
                    (
                        "parameter_matched",
                        "parameter_matched_candidate_minus_reference_mae",
                    ),
                ):
                    candidate, baseline = [
                        number(pair[f"{prefix}_{side}_mae"])
                        for side in ("candidate", "reference")
                    ]
                    check(
                        (candidate is None) == (baseline is None),
                        f"Asymmetric pair observations {key}",
                    )
                    compare(
                        candidate - baseline if candidate is not None else None,
                        pair[field],
                        (*key, prefix, "delta"),
                    )
                correction_count += int(radius > 1e-12 and count > 0) + int(
                    name in matched and count > 0
                )
    check(
        set(cells) == expected_cells
        and set(episodes) == expected_episodes
        and set(pairs) == expected_pairs,
        "Delivered score rows differ from the frozen factorial roster",
    )
    return correction_count


def verify_rollups(args, cfg, tables, decision):
    cells, episodes, pairs = tables
    physical = defaultdict(list)
    for (fold, _, method, window, uid), row in cells.items():
        if int(row["n_obs"]):
            physical[fold, method, window, uid].append(float(row["mae"]))
    values = {key: fmean(errors) for key, errors in physical.items()}
    physical_table = list(records(args.public / "summary/CELL_DRAW_MEANS.csv"))
    check(
        len(physical_table) == 60 * 11 * len(WINDOWS),
        "Wrong physical-cell summary coverage",
    )
    for row in physical_table:
        key = int(row["fold"]), row["method"], row["window"], row["cell_uid"]
        compare(values.get(key), row["mae"], key)
    totals = defaultdict(list)
    for (_, method, window, _), value in values.items():
        totals[method, window].append(value)
    summary = list(records(args.public / "summary/METHOD_SCORES.csv"))
    check(len(summary) == 11 * len(WINDOWS), "Wrong method score coverage")
    for row in summary:
        key = row["method"], row["window"]
        compare(mean(totals[key]), row["mae"], key)
        check(int(row["query_cells"]) == len(totals[key]), "Wrong method Query count")
    for field, head in (
        ("parameter_matched_candidate_minus_reference_mae", "parameter_matched"),
        ("matched_candidate_minus_reference_mae", "output_matched"),
        ("raw_candidate_minus_reference_mae", "raw"),
    ):
        comparison = defaultdict(list)
        for (fold, _, method, window, reference), row in pairs.items():
            value = number(row[field])
            if value is not None:
                comparison[fold, method, window, reference].append(value)
        for row in records(args.public / "summary/PAIR_SCORES.csv"):
            key = (
                int(row["fold"]),
                row["method"],
                row["window"],
                row["reference_method"],
            )
            compare(mean(comparison[key]), row[f"{head}_delta_mae"], key)
    near_zero = (
        max(
            float(episode["anchor_radius"])
            for (_, _, method, _), episode in episodes.items()
            if method == CANDIDATE
        )
        <= cfg["near_zero_anchor_radius"]
    )
    check(
        near_zero == decision["all_candidate_episodes_near_zero"],
        "Near-zero conclusion differs",
    )
    raw_marks = {row["reference_method"]: row for row in decision["raw_mae_gates"]}
    check(
        set(raw_marks) == set(cfg["methods"]) - {CANDIDATE},
        "Raw decision comparison roster differs",
    )
    failed = set()
    for method, mark in raw_marks.items():
        delta = defaultdict(list)
        for fold, arm, window, uid in values:
            if arm == CANDIDATE and window == "all":
                delta[fold].append(
                    values[fold, CANDIDATE, "all", uid]
                    - values[fold, method, "all", uid]
                )
        average = fmean(value for group in delta.values() for value in group)
        winning = sum(fmean(group) < 0 for group in delta.values())
        compare(average, mark["mean_delta_mae"], (method, "raw gate"))
        for item in mark["fold_deltas"]:
            compare(
                fmean(delta[item["fold"]]), item["delta_mae"], (method, item["fold"])
            )
        passed = average <= -cfg["mae_margin"] and winning >= cfg["winning_folds"]
        check(
            mark["complete"]
            and mark["passed"] == passed
            and mark["winning_folds"] == winning,
            f"Wrong raw gate {method}",
        )
        if not passed:
            failed.add(method)
    matched_marks = {
        row["reference_method"]: row for row in decision["parameter_match_gates"]
    }
    check(
        set(matched_marks) == {"ROTATION_CONST", "ROTATION_SHUFFLED"},
        "Wrong parameter gate roster",
    )
    for method, mark in matched_marks.items():
        effect = [
            [
                number(
                    pairs[fold, draw, CANDIDATE, "all", method][
                        "parameter_matched_candidate_minus_reference_mae"
                    ]
                )
                for draw in (0, 1, 2)
            ]
            for fold in cfg["folds"]
        ]
        complete = all(value is not None for group in effect for value in group)
        check(mark["complete"] == complete, f"Wrong parameter-match coverage {method}")
        if complete:
            by_fold = [fmean(group) for group in effect]
            average = fmean(by_fold)
            wins = sum(value < 0 for value in by_fold)
            compare(average, mark["mean_delta_mae"], (method, "matched gate"))
            check(
                mark["passed"] == (average < 0 and wins >= cfg["winning_folds"])
                and wins == mark["winning_folds"],
                f"Wrong parameter-match decision {method}",
            )
    if any(not mark["complete"] for mark in matched_marks.values()):
        expected = "STOP_INCOMPLETE_COMPARISON"
    elif near_zero:
        expected = "STOP_NEAR_ZERO_CORRECTION"
    elif failed & {"K0", "CAL2", "LORA_SMALL"}:
        expected = "STOP_SIMPLE_BASELINE"
    elif failed & {"ROTATION_SHUFFLED", SAME_CORE}:
        expected = "STOP_PHYSICAL_CONDITIONING"
    elif failed:
        expected = "STOP_ROTATION_CONTROL"
    elif any(not mark["passed"] for mark in matched_marks.values()):
        expected = "STOP_PARAMETER_MATCH_DIRECTION"
    else:
        expected = "ELIGIBLE_FOR_SEPARATE_SOURCE_CONFIRMATION"
    check(
        expected == decision["direction"],
        "A different source study conclusion was delivered",
    )
    return expected, {method: mean(totals[method, "all"]) for method in cfg["methods"]}


def authenticate_inputs(args, cfg, plan, identity, registration):
    root = args.public
    check(
        plan["config_sha256"] == digest(args.config)
        and plan["parent_episodes_sha256"] == digest(ROOT / cfg["source_episodes"])
        and identity["plan_sha256"] == digest(root / "REGISTERED_EPISODES.json")
        and identity["lora_config_sha256"] == digest(ROOT / cfg["lora_parent_config"]),
        "Registered plan or code configuration differs",
    )
    code_names = {
        path.name for path in Path(__file__).parent.glob("spectral_rotation*.py")
    }
    authenticate(
        Path(__file__).parent,
        identity["code_sha256"],
        code_names | {"run_spectral_rotation.py", "summarize_spectral_rotation.py"},
    )
    marker = args.features / "FEATURES_RECEIPT.json"
    receipt = read_json(marker)
    check(
        digest(marker) == identity["feature_receipt_sha256"]
        and digest(root / "FEATURES_RECEIPT.json") == identity["feature_receipt_sha256"]
        and receipt["status"] == "COMPLETE_SIGNED_SOURCE_PREFIX_FEATURES"
        and receipt["implementation_sha256"]
        == identity["code_sha256"]["spectral_rotation_features.py"]
        and receipt["prefix_cache_manifest_sha256"]
        == identity["prefix_cache_manifest_sha256"]
        and receipt["prefix_cache_manifest_sha256"]
        == digest(args.data_root / "MATR/COMPLETE.json")
        and receipt["features_sha256"] == digest(args.features / "FEATURES.npz")
        and receipt["audit"]["post_prefix_supervision_rows_read"] == 0,
        "Feature or prefix cache receipts changed",
    )
    with np.load(args.features / "FEATURES.npz", allow_pickle=False) as archive:
        ids = archive["ids"].tolist()
        values, mask = archive["values"], archive["mask"]
        check(
            ids == receipt["source_ids"]
            and set(ids) == set(registration["source_ids"])
            and values.shape == mask.shape == (115, len(archive["columns"]))
            and np.array_equal(mask, np.isfinite(values)),
            "Source descriptor array identity changed",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=ROOT / "results" / STUDY)
    parser.add_argument("--private", type=Path, default=ROOT / "runtime" / STUDY)
    parser.add_argument("--features", type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / STUDY / "CONFIG.json"
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Re-decode only locked MATR source-outer Query rows after checking all receipts",
    )
    args = parser.parse_args()
    args.features = args.features or args.private
    public = args.public
    final_path = public / "COMPLETE.json"
    completion = read_json(final_path)
    cfg, identity, plan = map(
        read_json,
        (
            args.config,
            public / "RUN_IDENTITY.json",
            public / "REGISTERED_EPISODES.json",
        ),
    )
    registration = read_json(ROOT / cfg["source_episodes"])
    check(
        completion["status"] == "COMPLETE_SOURCE_PROTOCOL_SCREEN"
        and completion["run_identity"] == identity
        and completion["target_label_rows_loaded"] == completion["new_target_fits"] == 0
        and completion["physical_query_cells"] == 60
        and completion["scored_episodes"] == 9
        and completion["methods"] == cfg["methods"]
        and completion["scored_folds"] == cfg["folds"]
        and completion["physical_query_protocols"]
        == plan["outer_source_query_protocols"],
        "Bad final source receipt",
    )
    authenticate_inputs(args, cfg, plan, identity, registration)
    registered, query, protocols = verify_plan(plan, cfg, registration)
    lock_sha = verify_lock(args, cfg, plan, identity, completion)
    tables, paths = verify_outer(args, cfg, plan, completion, lock_sha, registered)
    marker = public / "summary/SOURCE_VALIDATION_DECISION.json"
    decision = read_json(marker)
    check(
        digest(marker) == completion["summary_decision_sha256"]
        and decision["status"] == "COMPLETE"
        and decision["physical_query_cells"] == 60
        and decision["target_future_rows_loaded"] == 0
        and decision["evidence_scope"] == "source_development_screen_only"
        and decision["direction"] == completion["decision"],
        "Bad summary decision receipt",
    )
    authenticate(marker.parent, decision["files_sha256"], SUMMARY_FILES)
    truth = (
        read_source_truth(args.data_root, query, registration) if args.replay else None
    )
    matched_count = verify_cases(cfg, registered, protocols, tables, paths, truth)
    conclusion, method_mae = verify_rollups(args, cfg, tables, decision)
    review = {
        "status": "VERIFIED_WITH_SOURCE_QUERY_REPLAY"
        if args.replay
        else "VERIFIED_WITHOUT_FUTURE_ARRAY_READS",
        "completion_sha256": digest(final_path),
        "development_lock_sha256": lock_sha,
        "summary_decision_sha256": digest(marker),
        "independent_code_sha256": digest(Path(__file__)),
        "private_prediction_npz_files_verified": len(paths),
        "private_frozen_mapper_files_verified": len(cfg["folds"]) * len(MAPPERS),
        "registered_episodes": len(registered),
        "score_methods": len(cfg["methods"]),
        "source_query_cells": len(query),
        "source_query_protocols": plan["outer_source_query_protocols"],
        "reconstructed_cell_window_rows": len(tables[0]),
        "reconstructed_episode_window_rows": len(tables[1]),
        "reconstructed_pair_window_rows": len(tables[2]),
        "matched_contrasts_recomputed": matched_count,
        "source_query_future_label_rows_read_by_verifier": len(query)
        if args.replay
        else 0,
        "target_future_rows_loaded": 0,
        "new_optimizer_updates": 0,
        "full_trajectory_macro_MAE": method_mae,
        "decision": conclusion,
        "evidence_scope": "historically_exposed_source_only",
    }
    output = public / "INDEPENDENT_AUDIT.json"
    pending = output.with_suffix(".pending.json")
    pending.write_text(
        json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(output)
    print(
        f"{review['status']}. {len(query)} source cells, {len(tables[0])} score rows. {conclusion}"
    )


if __name__ == "__main__":
    main()
