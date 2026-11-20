"""Recompute locked spectral-objective scores from saved source predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from statistics import fmean

import numpy as np

from scripts.drs_peft.verify_spectral_rotation import WINDOWS, read_source_truth, window_mae


ROOT = Path(__file__).resolve().parents[2]
STUDY = "drs_peft/spectral_objective_v1"
V1 = "drs_peft/spectral_rotation_v1"
METHODS = ("ROTATION_TRUE", "ROTATION_CONST", "ROTATION_SHUFFLED", "GAIN")
OBJECTIVES = {"MM": "mse", "MA": "mae"}
CANDIDATE = "ROTATION_TRUE_MA"
MATCHED_RATE = "ROTATION_TRUE_MM_AT_MA_RATE"
SAME_CORE = "ROTATION_TRUE_MA_SAME_CORE_SHUFFLED"
BASELINES = ("K0", "STATIC_CORE", "LORA_SMALL", "CAL2")
SIMPLE_REFERENCES = ("K0", "LORA_SMALL", "CAL2", "STATIC_CORE")
STRUCTURAL_REFERENCES = ("GAIN_MA", "ROTATION_CONST_MA", "ROTATION_SHUFFLED_MA")
RAW_REFERENCES = (*SIMPLE_REFERENCES, *STRUCTURAL_REFERENCES)
PARAMETER_REFERENCES = {"ROTATION_CONST_MA", "ROTATION_SHUFFLED_MA"}
OUTER_METHODS = (*BASELINES, *(f"{method}_{obj}" for method in METHODS for obj in OBJECTIVES), MATCHED_RATE, SAME_CORE)
DETERMINISTIC = {"K0", "CAL2", "STATIC_CORE"}
TABLES = {"VALIDATION_TRIALS.csv", "META_TRACE.csv", "COSTS.csv"}
OUTER_TABLES = {"CELL_SCORES.csv", "EPISODE_SCORES.csv", "PAIR_SCORES.csv"}
SUMMARY_FILES = {
    "CELL_SCORES.csv", "EPISODE_SCORES.csv", "PAIRED_CELL_SEED_SCORES.csv", "CELL_MEAN_SCORES.csv",
    "FOLD_SCORES.csv", "SEED_SCORES.csv", "FOLD_SEED_SCORES.csv", "METHOD_SCORES.csv", "PROTOCOL_SCORES.csv",
    "MATCHED_RATE_SCORES.csv", "SCALE_MATCHED_EPISODES.csv", "SCALE_MATCHED_SUMMARY.csv", "WINDOW_CONTRIBUTIONS.csv",
    "WINDOW_CONTRIBUTION_SUMMARY.csv", "WORST_CELLS.csv", "WORST_PROTOCOLS.csv", "COSTS.csv",
}
CODE_FILES = {
    "run_spectral_objective.py", "spectral_objective.py", "spectral_objective_engine.py", "spectral_objective_protocol.py",
    "spectral_rotation.py", "spectral_rotation_engine.py", "run_spectral_rotation.py", "transfer_scope_data.py",
}
SETTING_FIELDS = {
    "K0": (), "STATIC_CORE": ("alpha",), "LORA_SMALL": ("steps", "lr"), "CAL2": ("eta", "ridge"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_files(root, manifest, expected):
    require(set(manifest) == set(expected), f"Different authenticated artifact inventory in {root}")
    for name, digest in manifest.items():
        require(sha(root / name) == digest, f"Artifact changed {root / name}")


def close(actual, expected, where):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12),
            f"Independent result differs at {where}")


def role_ids(episodes):
    return {uid for ep in episodes for role in ("support", "query") for uid in ep[role]}


def reproduced_donors(ids, seed, fold, phase, draw, role):
    salt = ((seed, fold, phase, draw), role)
    ordered = sorted(ids, key=lambda uid: (hashlib.sha256(f"spectral_rotation_v1|{salt}|{uid}".encode()).digest(), uid))
    donors = dict(zip(ordered, ordered[1:] + ordered[:1], strict=True))
    return [donors[uid] for uid in ids]


def verify_episode(ep, seed, fold, phase, protocols, allow, anchors=()):
    support, query = ep["support"], ep["query"]
    wanted = set(support) | set(query)
    require(len(support) == len(set(support)) == 3 and len(query) == len(set(query))
            and wanted <= allow and not set(support) & set(query)
            and not {protocols[uid] for uid in support} & {protocols[uid] for uid in query},
            f"Bad frozen source episode {fold}/{phase}/{ep['draw']}")
    roles = {"support": support, "query": query}
    if anchors:
        roles["anchor"] = anchors
    require(set(ep["donors"]) == set(roles), "Different descriptor roles")
    for role, ids in roles.items():
        require(ep["donors"][role] == reproduced_donors(ids, seed, fold, phase, ep["draw"], role)
                and all(a != b for a, b in zip(ep["donors"][role], ids, strict=True)), "Frozen descriptor donor changed")


def authentication(args):
    config = read_json(args.config)
    paths = {kind: args.public / name for kind, name in (
        ("identity", "RUN_IDENTITY.json"), ("plan", "REGISTERED_EPISODES.json"), ("stage", "STAGE_A_DECISION.json")
    )}
    identity, plan, decision = (read_json(paths[name]) for name in ("identity", "plan", "stage"))
    identity_sha = sha(paths["identity"])
    parent_plan, population = (read_json(ROOT / config[name]) for name in ("parent_plan", "source_episodes"))
    receipt_path = args.features / "FEATURES_RECEIPT.json"
    features = read_json(receipt_path)
    require(identity["config_sha256"] == plan["config_sha256"] == sha(args.config)
            and identity["plan_sha256"] == sha(paths["plan"]) and decision["identity_sha256"] == identity_sha
            and identity["lora_config_sha256"] == sha(ROOT / config["lora_parent_config"])
            and plan["parent_plan_sha256"] == config["parent_plan_sha256"] == sha(ROOT / config["parent_plan"])
            and plan["parent_completion_sha256"] == sha(ROOT / config["parent_completion"])
            and plan["source_episodes_sha256"] == sha(ROOT / config["source_episodes"])
            and identity["feature_receipt_sha256"] == sha(receipt_path)
            and features["features_sha256"] == sha(args.features / "FEATURES.npz")
            and identity["prefix_cache_manifest_sha256"] == features["prefix_cache_manifest_sha256"] == sha(args.data_root / "MATR/COMPLETE.json")
            and features["implementation_sha256"] == sha(Path(__file__).with_name("spectral_rotation_features.py")),
            "Stage A input identity drifted")
    require(plan["status"] == "REGISTERED_WITHOUT_OUTER_LABELS" and plan["historically_used_source_population"]
            and plan["source_outer_future_label_reads"] == plan["target_future_label_reads"] == 0
            and features["audit"]["post_prefix_supervision_rows_read"] == 0
            and decision["outer_role_labels_read"] == decision["target_labels_read"] == 0,
            "Stage A data scope differs")
    check_files(Path(__file__).parent, identity["code_sha256"], CODE_FILES)
    require(config["domain"] == "MATR" and config["folds"] == [0, 1, 2] and config["shot"] == 3
            and config["reference_seed"] == config["evaluation_plan_seed"] == config["selection_seed"] == 42
            and config["meta_training_seeds"] == [42, 43, 44] and config["mapper_methods"] == list(METHODS)
            and config["meta_query_objectives"] == list(OBJECTIVES.values()) and config["baselines"] == list(BASELINES)
            and config["inner_objective"] == "mse" and config["core_steps"] == 3 and config["train_episodes"] == 32
            and config["core_rates"] == [1e-4, 1e-3, 1e-2, 1e-1] and config["lora_rates"] == [1e-7, 1e-6, 1e-5, 3e-5]
            and config["lora_steps"] == [1, 10] and config["cal2_ridge"] == [1e-4, .01, 1.0]
            and config["cal2_eta"] == [.01, .1, .25, 1.] and config["mae_margin"] == 1e-5 and config["winning_folds"] == 2
            and not config["target_label_access"]
            and not config["outer_label_access_before_lock"], "Different frozen design")
    require([record["fold"] for record in plan["folds"]] == config["folds"], "Different fold roster")
    return config, plan, decision, parent_plan, population, identity_sha


def verify_folds(args, cfg, plan, parent, population):
    source, sealed = (set(population[name]) for name in ("source_ids", "sealed_final_MATR_ids"))
    protocols = population["source_protocols"]
    require(len(source) == 115 and not source & sealed, "Source and target identities overlap")
    all_query = set()
    for fold, old in zip(plan["folds"], parent["folds"], strict=True):
        index, anchors = fold["fold"], fold["anchors"]
        fit = set(fold["fit_ids"])
        require(all(fold[key] == old[key] for key in ("fold", "anchors", "fit_ids", "reference_sha256"))
                and fold["reference_seed"] == 42 and len(anchors) == len(set(anchors)) == 8 and set(anchors) <= fit,
                "Original fit or reference ancestry changed")
        root = args.reference_root / f"fold{index}/seed42"
        receipt = read_json(root / "REFERENCE_COMPLETE.json")
        require(receipt["fit_ids"] == fold["fit_ids"] and receipt["final_target_query_labels_used"] == 0
                and receipt["reference_checkpoint_sha256"] == fold["reference_sha256"] == sha(root / "static/CHECKPOINT_SELECTED.pt"),
                "Frozen source checkpoint changed")
        roles = (fit, role_ids(fold["validation"]), role_ids(fold["outer"]))
        role_groups = [{protocols[uid] for uid in roster} for roster in roles]
        require(all(roster <= source and not roster & sealed for roster in roles)
                and all(not left & right for left, right in combinations(role_groups, 2)), "Nested source roles cross")
        for phase, name, n_queries, n_draws in (("validation", "inner", 8, (3, 1, 3)[index]), ("outer", "outer", 20, 3)):
            require(len(fold[phase]) == n_draws, "Different validation or outer draw count")
            parent_role = next(unit for unit in population["source_tasks"][name] if unit["fold_id"] == index)
            for ep, before in zip(fold[phase], old[phase], strict=True):
                require({key: ep[key] for key in ("draw", "support", "query")} == before
                        and ep["query"] == parent_role["query_ids"]
                        and len(ep["query"]) == n_queries and set(ep["support"]) <= set(parent_role["support_pool_ids"]),
                        "Inherited evaluation identity changed")
                verify_episode(ep, 42, index, phase, protocols, source, anchors)
        query = set(fold["outer"][0]["query"])
        require(not query & all_query, "An outer physical cell repeats across folds")
        all_query.update(query)
        schedules = []
        for seed, schedule in fold["train_by_seed"].items():
            require([ep["draw"] for ep in schedule] == list(range(32)), "Different fit-only meta draw axis")
            for item in schedule:
                verify_episode(item, int(seed), index, "meta", protocols, fit)
                require(len(item["query"]) == 3 and len({protocols[uid] for uid in item["support"] + item["query"]}) == 6,
                        "A meta episode no longer has six disjoint physical-cell protocols")
            schedules.append({(tuple(sorted(ep["support"])), tuple(sorted(ep["query"]))) for ep in schedule})
        require(set(fold["train_by_seed"]) == {"42", "43", "44"}
                and [{key: ep[key] for key in ("draw", "support", "query")} for ep in fold["train_by_seed"]["42"]] == old["train"]
                and all(len(schedule) == 32 for schedule in schedules)
                and all(left != right for left, right in combinations(schedules, 2)), "Meta repeat episode plans changed")
    require(len(all_query) == plan["outer_source_query_cells"] == 60
            and len({protocols[uid] for uid in all_query}) == plan["outer_source_query_protocols"] == 39, "Wrong outer census")


def expected_trials(cfg):
    trials = []
    for method, objective, rate in product(METHODS, OBJECTIVES, cfg["core_rates"]):
        trials.append((f"{method}_{objective}", {"alpha": rate}))
    trials.append(("K0", {}))
    trials.extend(("STATIC_CORE", {"alpha": rate}) for rate in cfg["core_rates"])
    trials.extend(("LORA_SMALL", {"steps": steps, "lr": rate}) for rate, steps in product(cfg["lora_rates"], cfg["lora_steps"]))
    trials.extend(("CAL2", {"eta": eta, "ridge": ridge}) for ridge, eta in product(cfg["cal2_ridge"], cfg["cal2_eta"]))
    return {(model, tuple(settings.items())) for model, settings in trials}


def check_trace(trace, fold, ident):
    model, objective, rate = (ident[name] for name in ("method", "objective", "alpha"))
    planned = fold["train_by_seed"][str(ident["training_seed"])]
    require(len(trace) == len(planned) == 32, "Missing source optimization steps")
    for row, episode in zip(trace, planned, strict=True):
        expected_uid = episode.get("episode_uid", f"fold{ident['fold']}/seed{ident['training_seed']}/draw{episode['draw']}")
        require(all(row[key] == ident[key] for key in ("method", "objective", "alpha", "training_seed", "fold"))
                and row["query_objective"] == OBJECTIVES[objective] and row["meta_episode"] == episode["draw"]
                and row["episode_uid"] == expected_uid
                and row["support_ids"] == "|".join(episode["support"]) and row["query_ids"] == "|".join(episode["query"]),
                "Training trace differs from its registered objective or episode")
        for role in ("support", "query"):
            expected = episode["donors"][role] if model == "ROTATION_SHUFFLED" else episode[role]
            expected = "" if model == "ROTATION_CONST" else "|".join(expected)
            require(row[f"{role}_descriptor_ids"] == expected, "Trace descriptor assignment differs")
        for key in ("train_query_mae", "train_query_mse", "mapper_grad_norm", "mapper_grad_norm_after_clip",
                    *(f"support_step_{step}_{measure}" for step in (1, 2, 3) for measure in ("mse", "grad_norm", "grad_norm_after_clip"))):
            require(math.isfinite(row[key]) and row[key] >= 0, f"Invalid tracked training quantity {key}")
        require(0 < row["support_observations"] <= 3 * 512 and 0 < row["meta_query_observations"] <= 3 * 512, "Map fit observation budget changed")
    require(any(row["mapper_grad_norm"] > 0 for row in trace) and rate > 0, "No measured mapper gradient")


def read_features(path, population):
    with np.load(path, allow_pickle=False) as archive:
        ids, columns = archive["ids"].tolist(), archive["columns"].tolist()
        values, mask = archive["values"], archive["mask"]
    require(len(ids) == len(set(ids)) == 115 and set(ids) == set(population)
            and len(columns) == len(set(columns)) == 54 and values.shape == mask.shape == (115, 54)
            and mask.dtype == bool and np.array_equal(mask, np.isfinite(values)), "Provenance differs in the signed source feature grid")
    return ids, columns, values, mask


def verify_scaler(receipt, fold, inputs):
    ids, columns, values, mask = inputs
    fit = fold["fit_ids"]
    axis = [ids.index(uid) for uid in fit]
    observed = mask[axis]
    counts = observed.sum(axis=0)
    active = counts >= 2
    known = np.where(observed, values[axis], np.nan)
    means, scales = np.zeros(len(columns)), np.ones(len(columns))
    means[active] = np.nanmean(known[:, active], axis=0)
    scales[active] = np.maximum(np.nanstd(known[:, active], axis=0), 1e-8)
    require(receipt["fit_ids"] == fit and receipt["features"] == columns
            and receipt["train_valid_counts"] == counts.tolist() and receipt["active_columns"] == int(active.sum())
            and np.allclose(means, receipt["train_mean"], atol=1e-12, rtol=0)
            and np.allclose(scales, receipt["train_scale"], atol=1e-12, rtol=0),
            "Scaler statistics do not replay from the fold's fit-only prefixes")


def parse_trials(args, cfg, plan, decision, identity_sha):
    public, private = args.public, args.private
    check_files(public, decision["files"], TABLES)
    require(set(decision["fold_receipts"]) == {"0", "1", "2"}, "Missing Stage A fold receipt")
    expected = expected_trials(cfg)
    prepared, all_paths = [], set()
    inputs = read_features(args.features / "FEATURES.npz", read_json(ROOT / cfg["source_episodes"])["source_ids"])
    for fold in plan["folds"]:
        index = fold["fold"]
        root = public / "stage_a" / f"fold{index}"
        marker = root / "COMPLETE.json"
        receipt = read_json(marker)
        require(sha(marker) == decision["fold_receipts"][str(index)]
                and receipt["status"] == "COMPLETE_STAGE_A_FOLD" and receipt["fold"] == index
                and receipt["identity_sha256"] == identity_sha and receipt["fit_label_ids"] == sorted(fold["fit_ids"])
                and receipt["validation_label_ids"] == sorted(role_ids(fold["validation"]))
                and receipt["outer_role_labels_read"] == receipt["target_labels_read"] == 0
                and len(receipt["mapper_trials"]) == 32 and len(receipt["baseline_trials"]) == 25
                and receipt["meta_rows"] == 1024 and receipt["validation_rows"] == 57 * len(fold["validation"]),
                f"Invalid Stage A fold receipt {index}")
        check_files(root, receipt["files"], TABLES | {"BASIS_AUDIT.json", "DESCRIPTOR_SCALER.json"})
        verify_scaler(read_json(root / "DESCRIPTOR_SCALER.json"), fold, inputs)
        require(read_json(root / "BASIS_AUDIT.json")["reference_sha256"] == fold["reference_sha256"], "Fold basis scope differs")
        enrolled = set()
        for arm in (*receipt["mapper_trials"], *receipt["baseline_trials"]):
            ident = arm["identity"]
            model = ident["model"]
            trained = model not in BASELINES
            require(ident["identity_sha256"] == identity_sha and ident["fold"] == index and ident["training_seed"] == 42,
                    "Saved prediction or mapper is outside Stage A")
            if trained:
                method, objective, alpha = (ident[key] for key in ("method", "objective", "alpha"))
                setting = {"alpha": alpha}
                require(method in METHODS and objective in OBJECTIVES and model == f"{method}_{objective}"
                        and ident["reference_sha256"] == fold["reference_sha256"] and ident["stage"] == "A",
                        "Invalid mapper arm identity")
                relative = Path("mapper_trials") / f"fold{index}/seed42" / model / f"alpha{alpha:.9g}"
                require(arm["cost"]["source_mapping_updates"] == 32 and arm["cost"]["source_inner_updates"] == 96,
                        "Mapper budget differs from Stage A")
            else:
                require(ident["method"] == model and set(ident["setting"]) == set(SETTING_FIELDS[model]),
                        "Invalid frozen baseline hyperparameters")
                setting = {field: ident["setting"][field] for field in SETTING_FIELDS[model]}
                relative = Path("baseline_trials") / f"fold{index}" / "__".join(
                    [model] + [f"{key}_{value:.9g}" for key, value in setting.items()]
                )
            require(arm["path"] == relative.as_posix() and relative not in all_paths, "Missing or duplicated trial state path")
            all_paths.add(relative)
            key = model, tuple(setting.items())
            require(key in expected and key not in enrolled, f"Unexpected or duplicate Stage A grid entry {index}/{key}")
            enrolled.add(key)
            folder = private / relative
            files = {"PREDICTIONS.npz"} | ({"MAPPER.pt", "META_TRACE.json"} if trained else set())
            check_files(folder, arm["files"], files)
            if trained:
                inner = read_json(folder / "COMPLETE.json")
                require(inner["status"] == "COMPLETE_MAPPER" and inner["identity"] == ident and inner["files"] == arm["files"]
                        and inner["cost"] == arm["cost"] and len(inner["validation"]) == len(fold["validation"]),
                        "Private mapper receipt changed")
                check_trace(read_json(folder / "META_TRACE.json")["rows"], fold, ident)
            prepared.append((fold, ident, key, folder))
        require(enrolled == expected, f"Incomplete Stage A validation grid fold {index}")
    require(len(prepared) == 171 and decision["mapper_fits"] == 96 and decision["mapper_adam_updates"] == 3072
            and decision["differentiable_support_sgd_updates"] == 9216 and decision["validation_rows"] == 399,
            "Wrong Stage A work totals")
    return prepared


def read_validation_truth(data_root, plan, population):
    """Open only inherited source-validation roles, fold by fold."""
    from scripts.drs_peft.transfer_scope_data import load_future_rows

    truth = {}
    source, sealed = (set(population[name]) for name in ("source_ids", "sealed_final_MATR_ids"))
    for fold in plan["folds"]:
        ids = sorted(role_ids(fold["validation"]))
        require(ids and set(ids) <= source and not set(ids) & (sealed | set(fold["fit_ids"]) | role_ids(fold["outer"])),
                "A Stage A replay read would exceed its scope")
        y, mask = load_future_rows("MATR", data_root, ids)
        require(y.shape == mask.shape == (len(ids), 4900) and mask.any(axis=1).all(), "Wrong validation observation axes")
        truth[fold["fold"]] = dict(zip(ids, zip(y, mask, strict=True), strict=True))
    return truth


def validation_predictions(folder, fold):
    with np.load(folder / "PREDICTIONS.npz", allow_pickle=False) as data:
        predictions = data["predictions"]
        ids = data["ids"].tolist()
        draws = data["draws"].tolist()
    episodes = fold["validation"]
    require(ids == [ep["support"] + ep["query"] + fold["anchors"] for ep in episodes]
            and draws == [ep["draw"] for ep in episodes] and predictions.shape == (len(episodes), 19, 4900)
            and np.isfinite(predictions).all(), f"Prediction pack does not preserve evaluation axes {folder}")
    return predictions


def error_on_observations(predictions, cell_ids, truth, power, maximum=None):
    values = [truth[uid] for uid in cell_ids]
    y, observed = [np.stack(array) for array in zip(*values, strict=True)]
    if maximum is not None:
        observed = observed.copy()
        for mask in observed:
            indices = np.flatnonzero(mask)
            if len(indices) > maximum:
                selected = indices[np.rint(np.linspace(0, len(indices) - 1, num=maximum)).astype(np.int64)]
                mask[:] = False
                mask[selected] = True
    error = np.abs(np.where(observed, predictions.astype(np.float64) - y, 0.0))
    return float(np.mean((error**power).sum(axis=1) / observed.sum(axis=1)))


def prepare_csv_index(args, cfg):
    def key(row):
        model = row["model"]
        fields = SETTING_FIELDS.get(model, ("alpha",))
        config = tuple((field, float(row[field])) for field in fields)
        require(int(row["training_seed"]) == cfg["selection_seed"], "Validation table uses an unlocked seed")
        return int(row["fold"]), model, config, int(row["draw"])

    published = rows(args.public / "VALIDATION_TRIALS.csv")
    by_case = {key(row): row for row in published}
    require(len(by_case) == len(published) == 399, "Duplicated or incomplete aggregate validation axis")
    merged_meta = []
    merged_cost = []
    for fold in cfg["folds"]:
        folder = args.public / "stage_a" / f"fold{fold}"
        local = rows(folder / "VALIDATION_TRIALS.csv")
        fold_cases = {case for case in by_case if case[0] == fold}
        require(len(local) == len(fold_cases) and {key(row) for row in local} == fold_cases, "Fold rows differ from aggregate")
        for row in local:
            compare = by_case[key(row)]
            for measure in ("query_mae", "query_mse", "support_sampled_mse", "support_full_mse", "anchor_rms"):
                close(row[measure], compare[measure], (*key(row), measure))
        merged_meta.extend(rows(folder / "META_TRACE.csv"))
        merged_cost.extend(rows(folder / "COSTS.csv"))
    public_meta = rows(args.public / "META_TRACE.csv")
    public_cost = rows(args.public / "COSTS.csv")
    require(len(merged_meta) == len(public_meta) == 3072 and len(merged_cost) == len(public_cost) == 495,
            "Aggregate trace or cost axis differs")
    groups = defaultdict(list)
    for row in public_meta:
        group = int(row["fold"]), row["method"], row["objective"], float(row["alpha"])
        groups[group].append(int(row["meta_episode"]))
    require(len(groups) == 96 and all(sorted(draws) == list(range(32)) for draws in groups.values()), "Public source trace axis differs")
    require(Counter(row["cost_kind"] for row in public_cost) == {"SOURCE_META_TRAINING": 96, "VALIDATION_PROFILE": 399},
            "Public cost accounting differs")
    return by_case


def replay_validation(args, cfg, trials, truth):
    scores = defaultdict(dict)
    by_case = prepare_csv_index(args, cfg)
    baseline = {fold["fold"]: validation_predictions(folder, fold)
                for fold, _, (model, _), folder in trials if model == "K0"}
    consumed = set()
    for fold, ident, (model, setting), folder in trials:
        index = fold["fold"]
        prediction = baseline[index] if model == "K0" else validation_predictions(folder, fold)
        for position, episode in enumerate(fold["validation"]):
            draw = episode["draw"]
            check = (index, model, setting, draw)
            row = by_case[check]
            require(row["method"] == ident["method"] and row["objective"] == ident.get("objective", "BASELINE"),
                    "Prediction provenance and public validation objective differ")
            q_prediction = prediction[position, 3:11]
            s_prediction = prediction[position, :3]
            observed = truth[index]
            replay = {
                "query_mae": error_on_observations(q_prediction, episode["query"], observed, 1),
                "query_mse": error_on_observations(q_prediction, episode["query"], observed, 2),
                "support_sampled_mse": error_on_observations(s_prediction, episode["support"], observed, 2, cfg["support_max_observed"]),
                "support_full_mse": error_on_observations(s_prediction, episode["support"], observed, 2),
            }
            difference = prediction[position, 11:].astype(np.float64) - baseline[index][position, 11:]
            replay["anchor_rms"] = float(np.sqrt(np.mean(difference**2)))
            for metric, value in replay.items():
                close(value, row[metric], (*check, metric))
            consumed.add(check)
            scores[index, model, setting][draw] = replay["query_mae"]
    require(consumed == set(by_case), "Incomplete independent validation reconstruction")
    return scores


def recompute_gate(args, cfg, scores, decision):
    by_model = defaultdict(dict)
    for (fold, model, setting), by_draw in scores.items():
        by_model[fold, model][setting] = fmean(by_draw.values())
    selected = {str(fold): {} for fold in cfg["folds"]}
    for (fold, model), choices in by_model.items():
        chosen = min(choices, key=lambda option: (choices[option], option))
        record = dict(chosen) | {"validation_mae": choices[chosen]}
        sealed = decision["selected"][str(fold)][model]
        fold_receipt = read_json(args.public / "stage_a" / f"fold{fold}" / "COMPLETE.json")
        require(sealed == fold_receipt["selection"][model] and set(sealed) == set(record)
                and all(sealed[field] == value for field, value in chosen), "Different fold-local selected setting")
        close(record["validation_mae"], sealed["validation_mae"], (fold, model, "selected MAE"))
        selected[str(fold)][model] = record
    checks = []
    for index, expected in zip(cfg["folds"], decision["validation_folds"], strict=True):
        require(expected["fold"] == index, "Wrong Stage A decision fold")
        best = selected[str(index)][CANDIDATE]
        base = selected[str(index)]["K0"]["validation_mae"]
        row = {"fold": index, "alpha": best["alpha"], "k0_mae": base, "candidate_mae": best["validation_mae"],
               "delta_vs_k0": best["validation_mae"] - base,
               "passed": best["validation_mae"] <= base - cfg["mae_margin"]}
        require(expected["passed"] == row["passed"] and expected["alpha"] == row["alpha"], "Different Stage A fold gate")
        for name in ("k0_mae", "candidate_mae", "delta_vs_k0"):
            close(row[name], expected[name], (index, name))
        checks.append(row)
    status = "PASS_STAGE_A" if sum(row["passed"] for row in checks) >= 2 else "STOP_NO_VALIDATION_SIGNAL"
    marker = args.public / "STAGE_A_DECISION.json"
    require(decision["status"] == status and decision["primary_candidate"] == CANDIDATE
            and decision["mae_margin"] == cfg["mae_margin"], "Different independent Stage A outcome")
    if status == "STOP_NO_VALIDATION_SIGNAL":
        final = read_json(args.public / "FINAL_DECISION.json")
        expected_roles = {"seed43_44_training", "outer_scoring", "matched_rate_controls", "window_contributions"}
        require(final["status"] == status and final["stage_a_sha256"] == sha(marker)
                and final["outer_role_labels_read"] == final["target_labels_read"] == 0
                and final["later_stages"] == dict.fromkeys(expected_roles, "NOT_RUN_BY_STAGE_A_GATE")
                and decision["remaining_stages"] == "NOT_RUN_BY_STAGE_A_GATE", "Invalid Stage A stop record")
        require(not (args.public / "HYPERPARAMETER_LOCK.json").exists() and not (args.public / "DEVELOPMENT_LOCK.json").exists(),
                "A later phase was opened after Stage A stop")
    else:
        lock = read_json(args.public / "HYPERPARAMETER_LOCK.json")
        require(lock["status"] == "LOCKED_AFTER_STAGE_A" and lock["stage_a_sha256"] == sha(marker)
                and lock["identity_sha256"] == decision["identity_sha256"] and lock["selected"] == decision["selected"]
                and decision["remaining_stages"] == "PENDING", "Hyperparameter lock differs")
    return status, checks, selected


def verify_development_lock(args, cfg, plan, decision, identity_sha):
    public, private = args.public, args.private
    audit = read_json(public / "INDEPENDENT_AUDIT.json")
    verified_a = audit.get("stage_a", audit)
    hyper = public / "HYPERPARAMETER_LOCK.json"
    repeats_path = public / "stage_b/COMPLETE.json"
    lock_path = public / "DEVELOPMENT_LOCK.json"
    require(verified_a["status"] == "VERIFIED_STAGE_A_VALIDATION_PREDICTION_REPLAY"
            and verified_a["stage_a_decision_sha256"] == sha(public / "STAGE_A_DECISION.json")
            and verified_a["independent_decision"] == decision["status"] == "PASS_STAGE_A"
            and verified_a["run_identity_sha256"] == identity_sha and read_json(hyper)["selected"] == decision["selected"]
            and verified_a["target_label_rows_read"] == verified_a["outer_label_rows_read"] == 0,
            "An independently verified Stage A pass is required before the Stage C replay")
    lock, repeats = read_json(lock_path), read_json(repeats_path)
    require(lock["status"] == "LOCKED_BEFORE_SOURCE_OUTER_LABELS" and repeats["status"] == "COMPLETE_STAGE_B"
            and lock["identity_sha256"] == repeats["identity_sha256"] == identity_sha
            and lock["hyperparameters_sha256"] == sha(hyper) and lock["stage_b_sha256"] == sha(repeats_path)
            and lock["outer_role_labels_read"] == lock["target_labels_read"] == repeats["outer_role_labels_read"] == repeats["target_labels_read"] == 0
            and repeats["fold_fit_roles"] == {str(fold["fold"]): fold["fit_ids"] for fold in plan["folds"]},
            "Stage B or the development barrier does not authenticate source-only fitting")
    check_files(Path(__file__).parent, lock["scoring_code_sha256"],
                {"spectral_objective_outer.py", "summarize_spectral_objective.py", "summarize_spectral_rotation.py"})
    check_files(repeats_path.parent, repeats["files"], {"META_TRACE.csv", "COSTS.csv"})
    require(set(lock["mapper_artifacts"]) == {"0", "1", "2"}, "Locked mapper fold axis differs")
    unique_b = set()
    for fold in plan["folds"]:
        index = str(fold["fold"])
        lock_fold = lock["mapper_artifacts"][index]
        selected = decision["selected"][index]
        expected_names = {f"{method}_{objective}": (method, objective, selected[f"{method}_{objective}"]["alpha"])
                          for method, objective in product(METHODS, OBJECTIVES)}
        expected_names[MATCHED_RATE] = "ROTATION_TRUE", "MM", selected[CANDIDATE]["alpha"]
        require(set(lock_fold) == {f"{seed}:{name}" for seed, name in product(cfg["meta_training_seeds"], expected_names)},
                "Selected mapper aliases or training seeds differ")
        for seed, (name, (method, objective, alpha)) in product(cfg["meta_training_seeds"], expected_names.items()):
            entry = lock_fold[f"{seed}:{name}"]
            model = f"{method}_{objective}"
            path = Path("mapper_trials") / f"fold{index}/seed{seed}" / model / f"alpha{alpha:.9g}"
            require(entry["folder"] == path.as_posix() and entry["alpha"] == alpha, "Mapper alias uses a nonselected alpha")
            root = private / path
            receipt = read_json(root / "COMPLETE.json")
            ident = {"method": method, "objective": objective, "model": model, "alpha": alpha, "training_seed": seed,
                     "fold": fold["fold"], "identity_sha256": identity_sha, "stage": "A" if seed == 42 else "B",
                     "reference_sha256": fold["reference_sha256"]}
            require(receipt["status"] == "COMPLETE_MAPPER" and receipt["identity"] == ident
                    and entry["receipt_sha256"] == sha(root / "COMPLETE.json")
                    and entry["mapper_sha256"] == receipt["files"]["MAPPER.pt"], "Private frozen state provenance differs")
            checks = {"MAPPER.pt", "META_TRACE.json"} | ({"PREDICTIONS.npz"} if seed == 42 else set())
            check_files(root, receipt["files"], checks)
            check_trace(read_json(root / "META_TRACE.json")["rows"], fold, ident)
            if seed != 42:
                require(receipt["validation"] == [], "A repetition used inner validation to select a mapper")
                unique_b.add(path)
    actual_b = len(unique_b)
    total = 96 + actual_b
    require(48 <= actual_b <= 54 and repeats["unique_mapper_fits"] == actual_b
            and repeats["matched_rate_extra_mapper_fits"] == actual_b - 48 and repeats["total_mapper_fits"] == total
            and repeats["total_mapper_adam_updates"] == 32 * total and repeats["total_differentiable_sgd_updates"] == 96 * total,
            "Stage B matched-rate budget differs")
    return lock, sha(lock_path), repeats


def expected_score_objective(method):
    if method in BASELINES:
        return "BASELINE"
    if method == SAME_CORE:
        return "MA"
    return "MM" if method == MATCHED_RATE else method.split("_")[-1]


def outer_tables_and_predictions(args, plan, lock_sha, final):
    required = set(map(str, (0, 1, 2)))
    require(set(final["outer_receipts"]) == required, "Missing outer fold completion")
    tables = {name: {} for name in OUTER_TABLES}
    predictions = {}
    for fold in plan["folds"]:
        index = fold["fold"]
        root = args.public / "outer" / f"fold{index}"
        path = root / "COMPLETE.json"
        receipt = read_json(path)
        require(sha(path) == final["outer_receipts"][str(index)] and receipt["status"] == "COMPLETE_LOCKED_SOURCE_OUTER_FOLD"
                and receipt["fold"] == index and receipt["development_lock_sha256"] == lock_sha
                and receipt["target_label_rows_loaded"] == 0 and receipt["prediction_slots"] == len(receipt["predictions"]) == 9
                and receipt["source_outer_role_label_ids"] == sorted(role_ids(fold["outer"])), "Wrong outer role receipts")
        check_files(root, receipt["files"], OUTER_TABLES)
        for name in OUTER_TABLES:
            for row in rows(root / name):
                key = int(row["fold"]), int(row["training_seed"]), int(row["draw"]), row["method"], row["window"]
                require(key[0] == index and int(row["K"]) == 3 and int(row["reference_seed"]) == 42
                        and row["objective"] == expected_score_objective(row["method"]), "An outer score crossed identity axes")
                if name == "CELL_SCORES.csv":
                    key += (row["cell_uid"],)
                if name == "PAIR_SCORES.csv":
                    require(row["reference_objective"] == expected_score_objective(row["reference_method"]), "Wrong pair objective")
                    key += (row["reference_method"],)
                require(key not in tables[name], "Duplicated outer score row")
                tables[name][key] = row
        seen = set()
        for record in receipt["predictions"]:
            seed, draw = record["training_seed"], record["draw"]
            relative = Path(f"outer/fold{index}/seed{seed}/DRAW{draw}_PREDICTIONS.npz")
            require(record["path"] == relative.as_posix() and (seed, draw) not in seen
                    and sha(args.private / relative) == record["sha256"], "Unexpected outer prediction payload")
            seen.add((seed, draw))
            predictions[index, seed, draw] = args.private / relative
        require(seen == set(product((42, 43, 44), (0, 1, 2))), "Missing outer prediction payload")
    return tables, predictions


def compare_optional(actual, expected, context):
    if actual is None:
        require(expected in ("", None), f"Unexpected reported metric {context}")
    else:
        close(actual, expected, context)


def score_mean(cells):
    observed = [value for count, value in cells if count]
    return fmean(observed) if observed else None


def replay_outer(args, cfg, plan, lock, tables, paths, labels, protocols):
    raw_cells, episodes, pairs = (tables[name] for name in ("CELL_SCORES.csv", "EPISODE_SCORES.csv", "PAIR_SCORES.csv"))
    keys, episode_keys, pair_keys = set(), set(), set()
    errors, parameter, magnitudes = {}, defaultdict(dict), {}
    deterministic_hashes = {}
    references = (set(RAW_REFERENCES) - {"K0"}) | {MATCHED_RATE, SAME_CORE}
    plan_by_fold = {fold["fold"]: fold for fold in plan["folds"]}
    scale_pairs = Counter()
    for case, path in paths.items():
        fold, seed, draw = case
        fold_plan = plan_by_fold[fold]
        episode = fold_plan["outer"][draw]
        query = episode["query"]
        with np.load(path, allow_pickle=False) as bundle:
            require(bundle["ids"].tolist() == episode["support"] + query + fold_plan["anchors"]
                    and tuple(bundle["methods"].tolist()) == OUTER_METHODS
                    and np.array_equal(bundle["cycles"], np.arange(101, 5001)), f"Prediction axes differ in case {case}")
            predictions = bundle["predictions"]
            names, arrays, norm = (bundle[key] for key in ("parameter_matched_methods", "parameter_matched_predictions",
                                                           "parameter_matched_radii"))
        require(predictions.shape == (len(OUTER_METHODS), 31, 4900) and np.isfinite(predictions).all(),
                "Malformed outer predictions")
        require(len(names) == len(set(names)) and set(names.tolist()) <= PARAMETER_REFERENCES
                and norm.shape == (len(names),) and np.isfinite(norm).all() and (norm > 0).all(),
                "Malformed parameter-matched envelope")
        require((arrays.shape == (len(names), 2, 31, 4900) if len(names) else arrays.size == 0)
                and np.isfinite(arrays).all(), "Malformed parameter-matched prediction arrays")
        matched = dict(zip(names.tolist(), zip(arrays, norm, strict=True), strict=True))
        arms = dict(zip(OUTER_METHODS, predictions, strict=True))
        require(np.array_equal(arms[CANDIDATE][:3], arms[SAME_CORE][:3]), "Same-core control changed Support")
        base = arms["K0"].astype(np.float64)
        summaries = {}
        radii, query_radii = {}, {}
        source_hashes = {}
        for method, prediction in arms.items():
            radii[method] = float(np.sqrt(np.mean((prediction[23:].astype(np.float64) - base[23:])**2)))
            query_radii[method] = float(np.sqrt(np.mean((prediction[3:23].astype(np.float64) - base[3:23])**2)))
            source_hashes[method] = hashlib.sha256(np.ascontiguousarray(prediction, dtype="<f8").tobytes()).hexdigest()
            if method in DETERMINISTIC:
                history = deterministic_hashes.setdefault((fold, draw, method), source_hashes[method])
                require(history == source_hashes[method], f"Deterministic reuse changed predictions {case}/{method}")
        magnitudes[case] = radii[CANDIDATE], query_radii[CANDIDATE]
        for window, start, end in WINDOWS:
            for method, prediction in arms.items():
                key = *case, method, window
                episode_keys.add(key)
                stored = episodes[key]
                recovered = window_mae(prediction[3:23], query, labels, start, end)
                summaries[method, window] = score_mean(recovered)
                compare_optional(summaries[method, window], stored["mae"], key)
                close(radii[method], stored["anchor_radius"], (*key, "anchor RMS"))
                close(query_radii[method], stored["query_rms_full_grid"], (*key, "Query RMS"))
                require(stored["source_prediction_hash"] == source_hashes[method]
                        and stored["is_deterministic"] == str(method in DETERMINISTIC)
                        and (method not in DETERMINISTIC or stored["reused_prediction"] == str(seed != 42)), "Wrong prediction lineage")
                alias = CANDIDATE if method == SAME_CORE else method
                if alias not in BASELINES:
                    entry = lock["mapper_artifacts"][str(fold)][f"{seed}:{alias}"]
                    require(stored["source_mapper_sha256"] == entry["mapper_sha256"], "A different mapper produced predictions")
                    close(entry["alpha"], stored["alpha"], (*key, "adaptation alpha"))
                for uid, (count, value) in zip(query, recovered, strict=True):
                    cell = (*key, uid)
                    keys.add(cell)
                    record = raw_cells[cell]
                    require(record["protocol_uid"] == protocols[uid] and int(record["n_obs"]) == count,
                            f"Different source Query observations at {cell}")
                    compare_optional(value, record["mae"], cell)
                    errors[cell] = count, value
                require(sum(count for count, _ in recovered) == int(stored["n_obs"])
                        and sum(count > 0 for count, _ in recovered) == int(stored["n_cells"]), "Wrong episode MAE denominator")
            for method in references:
                key = *case, CANDIDATE, window, method
                pair_keys.add(key)
                pair = pairs[key]
                left, right = summaries[CANDIDATE, window], summaries[method, window]
                raw_delta = left - right if left is not None else None
                compare_optional(raw_delta, pair["raw_candidate_minus_reference_mae"], (*key, "raw"))
                holder = min(radii[CANDIDATE], radii[method])
                usable = holder > 1e-12
                require(pair["anchor_match_resolved"] == str(usable)
                        and pair["parameter_match_resolved"] == str(method in matched), "Stored scale matching availability differs")
                compare_optional(holder if usable else None, pair["common_anchor_radius"], (*key, "anchor radius"))
                output_mae, pressure_mae = [], []
                for side, name in (("candidate", CANDIDATE), ("reference", method)):
                    shrink = holder / radii[name] if usable else None
                    compare_optional(shrink, pair[f"{side}_eta"], (*key, side))
                    contracted = base[3:23] + shrink * (arms[name][3:23] - base[3:23]) if usable else None
                    error = score_mean(window_mae(contracted, query, labels, start, end)) if usable else None
                    compare_optional(error, pair[f"matched_{side}_mae"], (*key, "output matched", side))
                    output_mae.append(error)
                if method in matched:
                    (candidate, reference), radius = matched[method]
                    close(radius, pair["common_weight_frobenius"], (*key, "parameter radius"))
                    expected = min(float(episodes[(*case, name, window)]["weight_frobenius_rms"]) for name in (CANDIDATE, method))
                    close(radius, expected, (*key, "minimum anchor weight radius"))
                    for side, prediction in (("candidate", candidate), ("reference", reference)):
                        value = score_mean(window_mae(prediction[3:23], query, labels, start, end))
                        compare_optional(value, pair[f"parameter_matched_{side}_mae"], (*key, "parameter matched", side))
                        pressure_mae.append(value)
                for scope, values, name in (
                    ("output", output_mae, "matched_candidate_minus_reference_mae"),
                    ("parameter", pressure_mae, "parameter_matched_candidate_minus_reference_mae"),
                ):
                    delta = values[0] - values[1] if values and values[0] is not None else None
                    compare_optional(delta, pair[name], (*key, scope))
                    scale_pairs[scope] += int(delta is not None)
                    if scope == "parameter" and window == "all" and method in PARAMETER_REFERENCES:
                        parameter[method][case] = delta
    require(keys == set(raw_cells) and episode_keys == set(episodes) and pair_keys == set(pairs), "Outer physical scoring grid differs")
    return errors, parameter, magnitudes, dict(scale_pairs)


def average_repetitions(values):
    groups = defaultdict(list)
    for (fold, seed, _, method, window, uid), value in values.items():
        groups[fold, seed, method, window, uid].append(value)

    def average(rows):
        require(len(rows) == 3 and len({count for count, _ in rows}) == 1, "Physical coverage changes across paired repeats")
        count = rows[0][0]
        return count, fmean(val for _, val in rows) if count else None

    seeded = {key: average(rows) for key, rows in groups.items()}
    physical = defaultdict(list)
    for (fold, _, method, window, uid), value in seeded.items():
        physical[fold, method, window, uid].append(value)
    return seeded, {key: average(rows) for key, rows in physical.items()}


def verify_replayed_tables(root, seeded, physical, protocols):
    readouts = []
    for filename, memory, seeded_file in (("CELL_MEAN_SCORES.csv", physical, False),
                                          ("PAIRED_CELL_SEED_SCORES.csv", seeded, True),
                                          ("MATCHED_RATE_SCORES.csv", seeded, True)):
        seen = set()
        for row in rows(root / filename):
            identity = (int(row["fold"]),)
            if seeded_file:
                identity += (int(row["training_seed"]),)
            key = (*identity, row["method"], row["window"], row["cell_uid"])
            expected_count, expected_mae = memory[key]
            _, candidate_mae = memory[*identity, CANDIDATE, row["window"], row["cell_uid"]]
            require(key not in seen and int(row["n_obs"]) == expected_count and protocols[row["cell_uid"]] == row["protocol_uid"],
                    "Summary changes physical cell coverage")
            compare_optional(expected_mae, row["mae"], (filename, key))
            compare_optional(candidate_mae - expected_mae if expected_count else None, row["candidate_minus_method_mae"],
                             (filename, key, "candidate minus method"))
            seen.add(key)
        expected = {key for key in memory if key[-3] == MATCHED_RATE} if filename == "MATCHED_RATE_SCORES.csv" else set(memory)
        require(seen == expected, "Summary loses physical cell coverage")
        readouts.append(len(seen))

    fields = (("METHOD_SCORES.csv", (), False), ("FOLD_SCORES.csv", ("fold",), False),
              ("SEED_SCORES.csv", ("training_seed",), True), ("FOLD_SEED_SCORES.csv", ("fold", "training_seed"), True),
              ("PROTOCOL_SCORES.csv", ("fold", "protocol_uid"), False))
    for filename, axes, use_seed in fields:
        grouped = defaultdict(list)
        for key, pair in (seeded if use_seed else physical).items():
            fold, method, window, uid = (key[0], *key[-3:])
            identities = {"fold": fold, "training_seed": key[1] if use_seed else None, "protocol_uid": protocols[uid]}
            grouped[(*(identities[k] for k in axes), method, window)].append(pair)
        for row in rows(root / filename):
            key = (*(int(row[k]) if k in {"fold", "training_seed"} else row[k] for k in axes), row["method"], row["window"])
            group = grouped.pop(key)
            compare_optional(score_mean(group), row["mae"], (filename, key))
            require(int(row["physical_cells"]) == sum(n > 0 for n, _ in group)
                    and int(row["physical_observations"]) == sum(n for n, _ in group), "Wrong window coverage after macro averaging")
        require(not grouped, "A score rollup omits a registered group")
    return readouts


def verify_windows(root, physical, decision):
    windows = {name for name, _, _ in WINDOWS} - {"all"}
    columns = ("mae", "candidate_minus_method_mae", "method_minus_k0_mae")
    macros, rebuild = defaultdict(list), defaultdict(list)
    expected = {key for key in physical if key[2] in windows}
    confirmed = set()
    for row in rows(root / "WINDOW_CONTRIBUTIONS.csv"):
        fold, method, window, uid = int(row["fold"]), row["method"], row["window"], row["cell_uid"]
        key = fold, method, window, uid
        count, mae = physical[key]
        size, full_mae = physical[fold, method, "all", uid]
        _, candidate = physical[fold, CANDIDATE, window, uid]
        _, baseline = physical[fold, "K0", window, uid]
        parts = (mae, candidate - mae, mae - baseline) if count else (0., 0., 0.)
        require(size > 0 and key not in confirmed and key in expected and int(row["n_obs"]) == count
                and int(row["full_n_obs"]) == size, "Invalid contribution denominator")
        close(count / size, row["fraction_of_cell_observations"], (key, "physical window fraction"))
        for field, value in zip(columns, parts, strict=True):
            contribution = value * count / size
            close(contribution, row[f"{field}_contribution"], (*key, field))
            macros[method, window, field].append(contribution)
            rebuild[fold, method, uid, field].append(contribution)
        confirmed.add(key)
    require(confirmed == expected, "Window contribution rows do not cover every cell and method")
    for (fold, method, uid, field), parts in rebuild.items():
        _, mae = physical[fold, method, "all", uid]
        _, candidate = physical[fold, CANDIDATE, "all", uid]
        _, baseline = physical[fold, "K0", "all", uid]
        direct = {"mae": mae, "candidate_minus_method_mae": candidate - mae, "method_minus_k0_mae": mae - baseline}
        close(math.fsum(parts), direct[field], (fold, method, uid, field, "all-window reconstruction"))
    allowed_summary = set(product(OUTER_METHODS, windows))
    found = set()
    for row in rows(root / "WINDOW_CONTRIBUTION_SUMMARY.csv"):
        key = row["method"], row["window"]
        require(key not in found and key in allowed_summary and int(row["all_physical_cells"]) == 60, "Bad window contribution aggregate coverage")
        for name in columns:
            exact = fmean(macros[*key, name])
            close(exact, row[f"{name}_contribution"], (*key, name, "macro"))
        found.add(key)
    require(found == allowed_summary and all(value <= 1e-12 for value in decision["window_reconstruction_error"].values()),
            "Final window reconstruction certificate differs")
    return len(confirmed)


def verify_direction(mark, deltas, by_fold, by_seed, margin, require_seeds, method):
    average = fmean(deltas)
    fold = {str(i): fmean(values) for i, values in by_fold.items()}
    seed = {str(i): fmean(values) for i, values in by_seed.items()}
    wins_folds, wins_seeds = sum(v < 0 for v in fold.values()), sum(v < 0 for v in seed.values())
    improves = average < 0 and (margin is None or average <= -margin)
    passed = improves and wins_folds >= 2 and (not require_seeds or wins_seeds == len(seed))
    require(mark["reference_method"] == method and mark["margin"] == margin
            and mark["winning_folds"] == wins_folds and mark["winning_seeds"] == wins_seeds
            and mark["improvement_meets_mean_requirement"] == improves and mark["passed"] == passed,
            f"A different directional gate was reported for {method}")
    close(average, mark["mean_candidate_minus_reference_mae"], (method, "mean difference"))
    for computed, stored in ((fold, mark["fold_deltas"]), (seed, mark["seed_deltas"])):
        require(set(computed) == set(stored), "Incomplete gate repeat axis")
        for repeat, value in computed.items():
            close(value, stored[repeat], (method, repeat, "paired difference"))
    return passed


def verify_outer_gate(cfg, final, physical, seeded, parameter, magnitudes, selected):
    def raw(mark, reference, margin=None, require_seeds=True):
        folding, seeding = defaultdict(list), defaultdict(list)
        for fold, method, window, uid in physical:
            if method == CANDIDATE and window == "all":
                folding[fold].append(physical[fold, CANDIDATE, "all", uid][1] - physical[fold, reference, "all", uid][1])
        for fold, seed, method, window, uid in seeded:
            if method == CANDIDATE and window == "all":
                seeding[seed].append(seeded[fold, seed, CANDIDATE, "all", uid][1] - seeded[fold, seed, reference, "all", uid][1])
        return verify_direction(mark, [d for v in folding.values() for d in v], folding, seeding, margin, require_seeds, reference)

    raw_gates = {mark["reference_method"]: mark for mark in final["raw_gates"]}
    require(set(raw_gates) == set(RAW_REFERENCES), "Raw comparator gates are incomplete")
    failed = {reference for reference, gate in raw_gates.items() if not raw(gate, reference, cfg["mae_margin"])}
    same_core_passed = raw(final["same_core_gate"], SAME_CORE, require_seeds=False)
    matched_passed = raw(final["matched_rate_check"], MATCHED_RATE)
    raw(final["gain_mm_vs_candidate"], "GAIN_MM")
    require(final["outer_mae_contribution_supported"] == matched_passed, "Different matched-rate interpretation")
    param_gates = {item["reference_method"]: item for item in final["parameter_match_gates"]}
    require(set(param_gates) == PARAMETER_REFERENCES, "Missing equal-parameter norm gate")
    param_passed = []
    for name, mark in param_gates.items():
        recovered = parameter[name]
        require(set(recovered) == set(magnitudes), "Incomplete raw matched-parameter grid")
        observed = sum(value is not None for value in recovered.values())
        require(observed == mark["evaluated_positions"] and mark["expected_positions"] == 27, "Wrong matched sample denominator")
        if observed < 27:
            require(mark["passed"] is False and mark["reason"] == "UNRESOLVED_PARAMETER_NORM", "An unresolved norm reopened the gate")
            param_passed.append(False)
            continue
        fold, seed = defaultdict(list), defaultdict(list)
        for (fold_id, seed_id, _), effect in recovered.items():
            fold[fold_id].append(effect)
            seed[seed_id].append(effect)
        param_passed.append(verify_direction(mark, list(recovered.values()), fold, seed, None, False, name))
    scale = final["scale"]
    anchor, query = (list(values) for values in zip(*magnitudes.values(), strict=True))
    radius = cfg["near_zero_radius"]
    all_anchor, all_query = max(anchor) <= radius, max(query) <= radius
    require(scale["candidate_positions"] == 27 and scale["near_zero_threshold"] == radius
            and scale["all_anchor_positions_near_zero"] == all_anchor and scale["all_query_positions_near_zero"] == all_query
            and scale["near_zero_anchor_positions"] == sum(v <= radius for v in anchor)
            and scale["near_zero_query_positions"] == sum(v <= radius for v in query), "Wrong correction magnitude bounds")
    close(max(anchor), scale["maximum_anchor_rms"], "maximum anchor update")
    close(max(query), scale["maximum_query_rms_full_grid"], "maximum Query update")
    tags = []
    if "K0" in failed or all_query:
        tags.append("STOP_NO_ADAPTATION_GAIN")
    if failed & (set(SIMPLE_REFERENCES) - {"K0"}):
        tags.append("STOP_SIMPLE_BASELINE")
    if not failed & set(SIMPLE_REFERENCES) and failed & set(STRUCTURAL_REFERENCES):
        tags.append("OPTIMIZATION_GAIN_WITHOUT_STRUCTURE_EVIDENCE")
    if any(0 < mark["winning_seeds"] < 3 for mark in raw_gates.values()):
        tags.append("INCONCLUSIVE_SEED_INSTABILITY")
    if any(mark["mean_candidate_minus_reference_mae"] < 0 and mark["winning_folds"] < 2 for mark in raw_gates.values()):
        tags.append("INCONCLUSIVE_FOLD_INSTABILITY")
    if not same_core_passed or not all(param_passed):
        tags.append("STOP_STRUCTURE_EVIDENCE")
    eligible = not failed and same_core_passed and all(param_passed) and not all_query
    if all_anchor:
        tags.append("NEAR_ZERO_ON_ANCHORS")
    if all_query:
        tags.append("NEAR_ZERO_ON_QUERY")
    boundary = []
    for fold, methods in selected.items():
        for method, record in methods.items():
            name, grid = ("lr", cfg["lora_rates"]) if method == "LORA_SMALL" else ("alpha", cfg["core_rates"])
            if name in record and record[name] in (min(grid), max(grid)):
                boundary.append({"fold": int(fold), "method": method, "field": name, "value": record[name],
                                 "boundary": "minimum" if record[name] == min(grid) else "maximum"})
    require(boundary == final["rate_boundaries"], "Different locked learning-rate boundaries")
    if boundary:
        tags.append("RATE_BOUNDARY")
    status = "ELIGIBLE_FOR_SEPARATE_SOURCE_CONFIRMATION" if eligible else tags[0]
    require(final["direction"] == status and final["eligible_for_separate_source_confirmation"] == eligible
            and final["labels"] == ([status] if eligible else []) + tags, "A different full source decision was delivered")
    return status


def verify_a(args, cfg, plan, decision, population, identity_sha, trials):
    truth = read_validation_truth(args.data_root, plan, population)
    scores = replay_validation(args, cfg, trials, truth)
    status, checks, selected = recompute_gate(args, cfg, scores, decision)
    return {
        "status": "VERIFIED_STAGE_A_VALIDATION_PREDICTION_REPLAY",
        "independent_decision": status,
        "stage_a_decision_sha256": sha(args.public / "STAGE_A_DECISION.json"),
        "run_identity_sha256": identity_sha,
        "verifier_sha256": sha(Path(__file__)),
        "mapper_states_sha_verified": 96,
        "mapper_trace_episodes_verified": 3072,
        "support_updates_verified": 9216,
        "validation_prediction_npz_sha_verified": len(trials),
        "validation_prediction_rows_recomputed": sum(len(facts) for facts in scores.values()),
        "validation_role_label_reads_by_fold": {str(fold): len(values) for fold, values in truth.items()},
        "validation_role_physical_cells": len({cell for group in truth.values() for cell in group}),
        "new_mapper_optimizer_steps": 0,
        "outer_label_rows_read": 0,
        "target_label_rows_read": 0,
        "validation_folds": checks,
        "selected": selected,
    }


def verify_c(args, cfg, plan, decision, population, identity_sha):
    lock, lock_sha, repeats = verify_development_lock(args, cfg, plan, decision, identity_sha)
    final_path = args.public / "FINAL_DECISION.json"
    summary_path = args.public / "summary/FINAL_DECISION.json"
    final, summary = read_json(final_path), read_json(summary_path)
    require(final["status"] == summary["status"] == "COMPLETE" and final["development_lock_sha256"] == lock_sha
            and final["summary_receipt_sha256"] == sha(summary_path) and final["stage_b"] == repeats
            and final["target_future_rows_loaded"] == summary["target_future_rows_loaded"] == 0
            and all(final[key] == value for key, value in summary.items())
            and summary["physical_query_cells"] == 60 and summary["physical_query_protocols"] == 39
            and summary["paired_prediction_positions"] == 27, "The Stage C completion receipt changed")
    check_files(summary_path.parent, summary["files_sha256"], SUMMARY_FILES)
    tables, paths = outer_tables_and_predictions(args, plan, lock_sha, final)
    queries = {uid for fold in plan["folds"] for ep in fold["outer"] for uid in ep["query"]}
    truth = read_source_truth(args.data_root, queries, population)
    protocols = population["source_protocols"]
    cell_errors, parameters, magnitude, matches = replay_outer(args, cfg, plan, lock, tables, paths, truth, protocols)
    seeded, physical = average_repetitions(cell_errors)
    physical_count, repeat_count, matched_count = verify_replayed_tables(summary_path.parent, seeded, physical, protocols)
    window_count = verify_windows(summary_path.parent, physical, summary)
    outcome = verify_outer_gate(cfg, summary, physical, seeded, parameters, magnitude, decision["selected"])
    method_mae = {
        method: fmean(value for (fold, name, window, uid), (_, value) in physical.items() if name == method and window == "all")
        for method in OUTER_METHODS
    }
    review = {
        "status": "VERIFIED_STAGE_C_SOURCE_OUTER_PREDICTION_REPLAY",
        "independent_decision": outcome,
        "development_lock_sha256": lock_sha,
        "final_decision_sha256": sha(final_path),
        "summary_decision_sha256": sha(summary_path),
        "verifier_sha256": sha(Path(__file__)),
        "replay_helpers_sha256": sha(Path(__file__).with_name("verify_spectral_rotation.py")),
        "source_query_physical_cells": len(truth),
        "source_query_protocols": len({protocols[uid] for uid in truth}),
        "source_query_future_label_rows_read": len(truth),
        "target_label_rows_read": 0,
        "outer_prediction_npz_sha_verified": len(paths),
        "outer_cell_window_records_replayed": len(cell_errors),
        "source_mapper_fits_verified": repeats["total_mapper_fits"],
        "physical_cell_window_records_replayed": physical_count,
        "seed_cell_window_records_replayed": repeat_count,
        "matched_rate_seed_cell_window_records": matched_count,
        "window_contribution_records_replayed": window_count,
        "scale_matched_scores_count": matches,
        "meta_query_mae_contribution_supported": summary["outer_mae_contribution_supported"],
        "full_trajectory_macro_MAE": method_mae,
        "new_optimizer_updates": 0,
    }
    prior = read_json(args.public / "INDEPENDENT_AUDIT.json")
    return {"status": "VERIFIED_STAGE_A_AND_C_PREDICTION_REPLAY", "stage_a": prior.get("stage_a", prior), "stage_c": review}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("A", "C"), default="A", help="A replays only locked inner validation. C requires an independently verified A pass and a complete outer receipt.")
    parser.add_argument("--public", type=Path, default=ROOT / "results" / STUDY)
    parser.add_argument("--private", type=Path, default=ROOT / "runtime" / STUDY)
    parser.add_argument("--features", type=Path, default=ROOT / "runtime" / V1)
    parser.add_argument("--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "runtime/drs_peft/onesided_fewshot_v1/references")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / STUDY / "CONFIG.json")
    args = parser.parse_args()
    cfg, plan, decision, parent, population, identity_sha = authentication(args)
    verify_folds(args, cfg, plan, parent, population)
    trials = parse_trials(args, cfg, plan, decision, identity_sha)
    path = args.public / "INDEPENDENT_AUDIT.json"
    if args.phase == "C":
        receipt = verify_c(args, cfg, plan, decision, population, identity_sha)
        status = receipt["stage_c"]["independent_decision"]
        count = receipt["stage_c"]["outer_cell_window_records_replayed"]
    else:
        receipt = verify_a(args, cfg, plan, decision, population, identity_sha, trials)
        status = receipt["independent_decision"]
        count = receipt["validation_prediction_rows_recomputed"]
        if path.exists():
            prior = read_json(path)
            if prior["status"] == "VERIFIED_STAGE_A_AND_C_PREDICTION_REPLAY":
                require(prior["stage_a"]["stage_a_decision_sha256"] == receipt["stage_a_decision_sha256"],
                        "Stage A no longer matches the combined audit")
                receipt = prior | {"stage_a": receipt}
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)
    print(f"{receipt['status']}: {status}, {count} replayed evaluations.")


if __name__ == "__main__":
    main()
