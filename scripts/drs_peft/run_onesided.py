"""Execute registered source and target battery few-shot experiments."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.drs_peft.data import ROOT, file_hash, load_dataset, write_json
from scripts.drs_peft.onesided_basis import truncate_dictionary
from scripts.drs_peft.onesided_calibration import observed_positions
from scripts.drs_peft.onesided_protocol import load_registration, payload_hash
from scripts.drs_peft.run_onesided_reference import (
    DEFAULT_PRIVATE as REFERENCES,
)
from scripts.drs_peft.run_onesided_reference import (
    NoPolicyInputs,
    external_lock,
    load_reference_model,
    reference_folder,
)
from scripts.drs_peft.train import (
    bind,
    configure,
    early_batch,
    evaluate_arrays,
    read_json,
    save_torch,
)

CONFIG = ROOT / "configs/drs_peft/onesided_fewshot_v1/CONFIG.json"
PRIVATE = ROOT / "runtime/drs_peft/onesided_fewshot_v1"
PUBLIC = ROOT / "results/drs_peft/onesided_fewshot_v1"
GEOMETRY = ("CORE", "OUT", "IN", "NORMAL")
GENERIC = ("LoRA", "LoRA_FA")


def hyperparameters(config, method):
    adaptation = config["adaptation"]
    if method in GENERIC:
        specification = adaptation["lowrank"]
        return [
            {"learning_rate": lr, "steps": steps, "eta": eta}
            for lr, steps, eta in itertools.product(
                specification["learning_rates"],
                specification["steps"],
                specification["etas"],
            )
        ]
    specification = adaptation["ridge"]
    return [
        {"lambda_multiplier": ridge, "eta": eta, "epsilon": specification["epsilon"]}
        for ridge, eta in itertools.product(
            specification["lambda_multipliers"], specification["etas"]
        )
    ]


def task_for(registration, role, fold_id):
    matches = [
        task
        for task in registration["source_tasks"][role]
        if task["fold_id"] == int(fold_id)
    ]
    if len(matches) != 1:
        raise ValueError("Expected one own-fold registered source task")
    return matches[0]


def shot_for(task, k):
    matches = [shot for shot in task["shots"] if shot["k"] == int(k)]
    if len(matches) != 1 or matches[0]["status"] != "FEASIBLE":
        raise ValueError("The requested physical Support budget is infeasible")
    return matches[0]


def make_support(data, inputs, identifiers, device, maximum=512):
    """This is the sole transfer of future labels into an adaptation function."""
    result = []
    for uid in identifiers:
        row = int(data.rows([uid])[0])
        positions = observed_positions(data.observed[row], maximum)
        result.append(
            {
                "uid": uid,
                "early": early_batch(inputs, [row], device),
                "query": torch.as_tensor(data.query[positions], device=device),
                "target": data.targets[row, positions].copy(),
                "positions": positions,
            }
        )
    return result


def score_predictions(data, identifiers, predictions):
    """Read Query outcomes only after full fixed-grid predictions exist."""
    rows = data.rows(identifiers)
    table, summary = evaluate_arrays(
        predictions,
        data.targets[rows],
        data.observed[rows],
        data.query,
        identifiers,
        data.protocol_ids[rows],
        "original_cycle_id",
    )
    return table, summary


def prepare_engine(args, data, fold_id, seed, scope):
    from scripts.drs_peft.onesided_engine import FewShotEngine

    reference = reference_folder(args.reference_root, fold_id, seed)
    identity = read_json(reference / "REFERENCE_COMPLETE.json")
    geometry_name = "TRANSFER_SPECTRUM.pt" if scope == "D3" else "DICTIONARIES.pt"
    identity["dictionary_sha256"] = file_hash(reference / "geometry" / geometry_name)
    inputs = NoPolicyInputs().transform(data)
    inputs["ids"] = list(data.ids)
    model = load_reference_model(
        args.reference_root, args.backbone_dir, fold_id, seed, "cpu"
    )
    engine = FewShotEngine(
        model,
        inputs,
        data.query,
        identity,
        args.private_root / "engine" / str(scope) / f"fold{fold_id}" / f"seed{seed}",
        device=args.device,
        seed=seed,
    )
    return engine, inputs, identity


def register_methods(engine, reference, spectrum=False):
    if spectrum:
        spectral = torch.load(
            reference / "geometry/TRANSFER_SPECTRUM.pt",
            map_location="cpu",
            weights_only=False,
        )
        for method, payload in spectral["dictionaries"].items():
            engine.register_method(method, payload)
        return list(spectral["dictionaries"])
    bundle = torch.load(
        reference / "geometry/DICTIONARIES.pt", map_location="cpu", weights_only=False
    )
    methods = ["CAL2", *bundle["arms"], *GENERIC]
    for method in methods:
        engine.register_method(method, bundle["arms"].get(method))
    common = bundle["common_dimension"]
    if common and any(
        payload["dimension"] != common for payload in bundle["arms"].values()
    ):
        for method, payload in bundle["arms"].items():
            name = method + "_COMMON"
            engine.register_method(name, truncate_dictionary(payload, int(common)))
            methods.append(name)
    return methods


def adapted_state(engine, method, support, hyper):
    if method in GENERIC:
        trials = engine.nonlinear_trials(
            method, support, hyper["learning_rate"], steps=(10, 30)
        )
        selected = trials[int(hyper["steps"])]
        state = copy.deepcopy(selected["state"])
        state["adaptation_scale"] = float(hyper["eta"])
        details = dict(selected["details"])
        details["eta"] = float(hyper["eta"])
        return state, details
    return engine.geometry_trial(method, support, hyper)


def save_engine_cost(args, engine, role, fold_id, seed):
    path = (
        args.public_root
        / "ENGINE_COSTS"
        / f"{role}_fold{fold_id}_seed{seed}_pid{os.getpid()}.json"
    )
    write_json(
        path,
        {
            "role": role,
            "fold_id": fold_id,
            "seed": seed,
            "counters": dict(engine.counters),
            "counting_scope": "cumulative_actual_new_execution_in_this_process_engine;not_sum_of_selected_state_step_budgets",
            "source_checkpoint_sha256": engine.source_sha256,
            "source_commit": os.environ.get("BATTERY_SOURCE_COMMIT"),
        },
    )


def select_own_fold(trials, fold_id, seed, method, k):
    """Fail closed on cross-fold tuning and incomplete draw coverage."""
    selected = trials[
        trials.fold_id.eq(int(fold_id))
        & trials.seed.eq(int(seed))
        & trials.method.eq(method)
        & trials.K.eq(int(k))
    ]
    if selected.empty:
        raise ValueError("No own-fold inner trials are available")
    if set(selected.role) != {"inner"}:
        raise ValueError("Only inner Query scores can select adaptation settings")
    if not np.isfinite(selected.mae.to_numpy(dtype=float)).all():
        raise ValueError("Every inner draw must have a finite Query error")
    for _, group in selected.groupby("trial", sort=True):
        if len(group) != 5 or set(group.draw) != set(range(5)):
            raise ValueError(
                "Every inner trial requires exactly draws zero through four"
            )
    means = selected.groupby("trial", sort=True).agg(
        mae=("mae", "mean"), draws=("draw", "nunique")
    )
    if means.empty or not np.isfinite(means.mae).all():
        raise ValueError("An inner hyperparameter trial lacks five finite draws")
    return int(means.mae.idxmin()), float(means.mae.min())


def tune_inner(args, config, registration, data, fold_id, seed, spectrum=False):
    phase = "D3" if spectrum else "D2"
    task = task_for(registration, "inner", fold_id)
    reference = reference_folder(args.reference_root, fold_id, seed)
    engine, inputs, identity = prepare_engine(args, data, fold_id, seed, phase)
    methods = register_methods(engine, reference, spectrum)
    destination = args.public_root / "inner" / phase / f"fold{fold_id}" / f"seed{seed}"
    for method in methods:
        for k in (1, 3, 5):
            folder = destination / method / f"K{k}"
            marker = folder / "SELECTED.json"
            expected = {
                "config_sha256": file_hash(args.config),
                "reference_checkpoint_sha256": identity["reference_checkpoint_sha256"],
                "dictionary_sha256": identity["dictionary_sha256"],
                "execution_sha256": file_hash(__file__),
                "engine_sha256": file_hash(
                    ROOT / "scripts/drs_peft/onesided_engine.py"
                ),
                "episode_sha256": registration["episode_contract_sha256"],
                "fold_id": fold_id,
                "seed": seed,
                "method": method,
                "K": k,
                "role": "own_fold_inner_only",
                "query_ids": task["query_ids"],
            }
            if marker.exists():
                selected = read_json(marker)
                if (
                    selected["identity"] != expected
                    or file_hash(folder / "TRIALS.csv") != selected["trials_csv_sha256"]
                ):
                    raise RuntimeError("An inner selection changed identity")
                continue
            folder.mkdir(parents=True, exist_ok=True)
            records = []
            grid = hyperparameters(config, method)
            for trial, hyper in enumerate(grid):
                for draw in shot_for(task, k)["draws"]:
                    cache = (
                        args.private_root
                        / "inner_trials"
                        / phase
                        / f"fold{fold_id}"
                        / f"seed{seed}"
                        / method
                        / f"K{k}"
                        / f"trial{trial}_draw{draw['draw']}.json"
                    )
                    cache_identity = {
                        **expected,
                        "trial": trial,
                        "draw": draw["draw"],
                        "hyperparameters": hyper,
                        "support_ids": draw["support_ids"],
                    }
                    if cache.exists():
                        row = read_json(cache)
                        if row.get("identity") != cache_identity:
                            raise RuntimeError(
                                "An inner trial cache changed source or experimental identity"
                            )
                    else:
                        support = make_support(
                            data, inputs, sorted(draw["support_ids"]), args.device
                        )
                        state, details = adapted_state(engine, method, support, hyper)
                        prediction = engine.predict_state(
                            method, state, task["query_ids"]
                        )
                        _, summary = score_predictions(
                            data, task["query_ids"], prediction
                        )
                        row = {
                            "stage": phase,
                            "role": "inner",
                            "fold_id": fold_id,
                            "seed": seed,
                            "method": method,
                            "K": k,
                            "draw": draw["draw"],
                            "trial": trial,
                            "mae": summary["mae"],
                            "hyperparameters": hyper,
                            "support_ids": draw["support_ids"],
                            "query_ids": task["query_ids"],
                            "details": details,
                            "identity": cache_identity,
                        }
                        write_json(cache, row)
                    records.append(row)
            table = pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        not in (
                            "details",
                            "support_ids",
                            "query_ids",
                            "hyperparameters",
                            "identity",
                        )
                    }
                    for row in records
                ]
            )
            trial, score = select_own_fold(table, fold_id, seed, method, k)
            table.to_csv(folder / "TRIALS.csv", index=False)
            write_json(
                marker,
                {
                    "identity": expected,
                    "selected_trial": trial,
                    "hyperparameters": grid[trial],
                    "inner_mae": score,
                    "trial_grid": grid,
                    "target_query_label_uses": 0,
                    "trials_csv_sha256": file_hash(folder / "TRIALS.csv"),
                },
            )
            print(
                json.dumps(
                    {
                        "event": "INNER_SELECTED",
                        "stage": phase,
                        "fold": fold_id,
                        "seed": seed,
                        "method": method,
                        "K": k,
                        "mae": score,
                        "hyperparameters": grid[trial],
                    }
                ),
                flush=True,
            )
            save_engine_cost(args, engine, phase + "_inner", fold_id, seed)


def unit_folder(args, phase, task, seed, method, k, draw):
    return (
        args.public_root
        / "units"
        / phase
        / task["task_id"]
        / f"seed{seed}"
        / method
        / f"K{k}_draw{draw}"
    )


def prediction_identity(phase, task, seed, method, k, draw, support, hyper, reference):
    return {
        "stage": phase,
        "task_id": task["task_id"],
        "domain": task["domain"],
        "fold_id": task.get("fold_id", "final"),
        "seed": seed,
        "method": method,
        "K": k,
        "draw": draw,
        "support_ids": list(support),
        "query_ids": task["query_ids"],
        "hyperparameters": hyper,
        "source_checkpoint_sha256": reference["reference_checkpoint_sha256"],
        "dictionary_sha256": reference["dictionary_sha256"],
        "execution_sha256": file_hash(__file__),
        "engine_sha256": file_hash(ROOT / "scripts/drs_peft/onesided_engine.py"),
    }


def write_cell_metrics(args, data, folder, receipt, prediction, zero):
    identifiers = receipt["query_ids"]
    table, _ = score_predictions(data, identifiers, prediction)
    baseline, _ = score_predictions(data, identifiers, zero)
    if not table[["cell_uid", "window", "n_obs"]].equals(
        baseline[["cell_uid", "window", "n_obs"]]
    ):
        raise RuntimeError(
            "Adapted and ZERO predictions use different physical Query scoring"
        )
    table["zero_mae"] = baseline.mae
    table["gain_mae"] = table.zero_mae - table.mae
    table["negative_transfer"] = table.gain_mae.lt(0).where(table.n_obs.gt(0))
    for key in (
        "stage",
        "task_id",
        "domain",
        "fold_id",
        "seed",
        "method",
        "K",
        "draw",
        "adaptation_id",
        "is_duplicate",
    ):
        table[key] = receipt[key]
    lookup = dict(zip(data.ids, data.protocol_ids, strict=True))
    counts = {
        uid: sum(lookup[uid] == lookup[support] for support in receipt["support_ids"])
        for uid in identifiers
    }
    table["same_protocol_support_count"] = table.cell_uid.map(counts)
    table.to_csv(folder / "CELL_METRICS.csv", index=False)
    receipt["status"] = "SCORED"
    receipt["cell_metrics_sha256"] = file_hash(folder / "CELL_METRICS.csv")
    receipt["query_cells"] = len(identifiers)
    receipt["metrics_rows"] = len(table)
    write_json(folder / "RECEIPT.json", receipt)


def execute_task(
    args,
    config,
    engine,
    inputs,
    data,
    identity,
    phase,
    task,
    seed,
    methods,
    selections,
    defer_score=False,
):
    identifiers = task["query_ids"]
    zero = engine.baseline_prediction(identifiers)
    configurations = [("ZERO", 0, shot_for(task, 0)["draws"][0], {})]
    configurations.extend(
        (method, k, draw, selections[(method, k)])
        for method, k in itertools.product(methods, (1, 3, 5))
        for draw in shot_for(task, k)["draws"]
    )
    for method, k, draw, hyper in configurations:
        folder = unit_folder(args, phase, task, seed, method, k, draw["draw"])
        receipt_path = folder / "RECEIPT.json"
        expected = prediction_identity(
            phase,
            task,
            seed,
            method,
            k,
            draw["draw"],
            draw["support_ids"],
            hyper,
            identity,
        )
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if any(receipt[key] != value for key, value in expected.items()):
                raise RuntimeError(
                    "An existing adaptation changed source, Support, Query or selection"
                )
            if receipt["status"] == "SCORED" or defer_score:
                continue
        started = time.perf_counter()
        counters_before = dict(engine.counters)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        state_identity = {
            key: value for key, value in expected.items() if key != "draw"
        }
        state_identity["support_ids"] = sorted(state_identity["support_ids"])
        adaptation_id = payload_hash(state_identity)
        private = args.private_root / "adaptations" / adaptation_id
        private.mkdir(parents=True, exist_ok=True)
        state_path, prediction_path = private / "STATE.pt", private / "PREDICTIONS.npz"
        cached = prediction_path.exists() and state_path.exists()
        if cached:
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            if saved["identity"] != state_identity:
                raise RuntimeError("A cached adaptation state changed identity")
            with np.load(prediction_path, allow_pickle=False) as archive:
                if archive["ids"].astype(
                    str
                ).tolist() != identifiers or not np.array_equal(
                    archive["query"], data.query
                ):
                    raise RuntimeError("Cached predictions use another fixed Query")
                prediction = archive["prediction"].copy()
            details, replay = saved["details"], saved["replay"]
        else:
            query_diagnostics = None
            if method == "ZERO":
                state, prediction, details, replay = (
                    None,
                    zero,
                    {"trainable_parameters": 0, "optimizer_updates": 0},
                    {"shared_reference": True},
                )
            else:
                support = make_support(
                    data,
                    inputs,
                    state_identity["support_ids"],
                    args.device,
                    config["adaptation"]["maximum_support_observations"],
                )
                state, details = adapted_state(engine, method, support, hyper)
                prediction = engine.predict_state(method, state, identifiers)
                replay = engine.replay_and_merge(method, state, identifiers[:1])
                details.update(engine.support_diagnostics(method, state, support))
                if method not in GENERIC:
                    query_diagnostics = engine.query_diagnostics(
                        method, state, identifiers[:1], prediction=prediction[:1]
                    )
                    details["query_diagnostic_cells"] = identifiers[:1]
                    details["query_diagnostic_scope"] = (
                        "first_fixed_Query_cell_full_4900_grid_per_adaptation"
                    )
                    details["query_gap_MAE"] = query_diagnostics["query_gap_MAE"]
            if not np.isfinite(prediction).all():
                raise FloatingPointError(
                    "An adapted fixed-grid prediction is nonfinite"
                )
            save_torch(
                state_path,
                {
                    "identity": state_identity,
                    "state": state,
                    "details": details,
                    "replay": replay,
                    "query_diagnostics": query_diagnostics,
                },
            )
            np.savez_compressed(
                prediction_path,
                ids=np.asarray(identifiers),
                query=data.query,
                prediction=prediction,
            )
        folder.mkdir(parents=True, exist_ok=True)
        receipt = {
            **expected,
            "status": "PREDICTIONS_LOCKED",
            "adaptation_id": adaptation_id,
            "is_duplicate": draw.get("duplicate_of_draw") is not None,
            "duplicate_of_draw": draw.get("duplicate_of_draw"),
            "prediction_path": prediction_path.relative_to(ROOT).as_posix(),
            "prediction_sha256": file_hash(prediction_path),
            "state_path": state_path.relative_to(ROOT).as_posix(),
            "state_sha256": file_hash(state_path),
            "cache_reused": cached,
            "details": {
                key: value
                for key, value in details.items()
                if key
                not in (
                    "support_actual_predictions",
                    "support_linear_predictions",
                    "support_linearization_gap",
                    "support_predictions",
                    "support_before_predictions",
                    "support_after_predictions",
                )
            },
            "serialized_replay_and_merge": replay,
            "elapsed_seconds": time.perf_counter() - started,
            "new_execution_counters": {
                key: value - counters_before[key]
                for key, value in engine.counters.items()
            },
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if args.device.startswith("cuda")
            else 0,
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved())
            if args.device.startswith("cuda")
            else 0,
            "final_target_query_labels_used_for_selection_or_adaptation": 0,
            "source_commit": os.environ.get("BATTERY_SOURCE_COMMIT"),
            "config_sha256": file_hash(args.config),
        }
        write_json(receipt_path, receipt)
        if not defer_score:
            write_cell_metrics(args, data, folder, receipt, prediction, zero)
        print(
            json.dumps(
                {
                    "event": "ADAPTATION_COMPLETE",
                    "stage": phase,
                    "task": task["task_id"],
                    "seed": seed,
                    "method": method,
                    "K": k,
                    "draw": draw["draw"],
                    "cached": cached,
                    "seconds": receipt["elapsed_seconds"],
                }
            ),
            flush=True,
        )
        save_engine_cost(
            args,
            engine,
            phase + "_" + task["domain"] + "_evaluation",
            task.get("fold_id", "final"),
            seed,
        )


def execute_source(args, config, registration, data, fold_id, seed, spectrum=False):
    phase = "D3" if spectrum else "D2"
    reference = reference_folder(args.reference_root, fold_id, seed)
    engine, inputs, identity = prepare_engine(args, data, fold_id, seed, phase)
    methods = register_methods(engine, reference, spectrum)
    selections = {}
    for method, k in itertools.product(methods, (1, 3, 5)):
        selection_folder = (
            args.public_root
            / "inner"
            / phase
            / f"fold{fold_id}"
            / f"seed{seed}"
            / method
            / f"K{k}"
        )
        selected = read_json(selection_folder / "SELECTED.json")
        if (
            selected["identity"]["fold_id"] != fold_id
            or selected["identity"]["seed"] != seed
            or selected["identity"]["method"] != method
            or selected["identity"]["K"] != k
            or selected["identity"]["config_sha256"] != file_hash(args.config)
            or selected["identity"]["reference_checkpoint_sha256"]
            != identity["reference_checkpoint_sha256"]
            or selected["identity"]["dictionary_sha256"]
            != identity["dictionary_sha256"]
            or selected["identity"]["episode_sha256"]
            != registration["episode_contract_sha256"]
            or selected["identity"]["execution_sha256"] != file_hash(__file__)
            or selected["trials_csv_sha256"]
            != file_hash(selection_folder / "TRIALS.csv")
        ):
            raise RuntimeError("An outer task received another fold's hyperparameters")
        selections[(method, k)] = selected["hyperparameters"]
    task = task_for(registration, "outer", fold_id)
    execute_task(
        args,
        config,
        engine,
        inputs,
        data,
        identity,
        phase,
        task,
        seed,
        methods,
        selections,
    )


def target_lock(args, config, registration):
    selections, scores, fingerprints = {}, {}, []
    for method in ("CAL2", "OUT", *GENERIC):
        selections[method], scores[method] = {}, {}
        for k in (1, 3, 5):
            tables = []
            for fold_id, seed in itertools.product((0, 1, 2), config["training_seeds"]):
                folder = (
                    args.public_root
                    / "inner/D2"
                    / f"fold{fold_id}"
                    / f"seed{seed}"
                    / method
                    / f"K{k}"
                )
                selected = read_json(folder / "SELECTED.json")
                reference = reference_folder(args.reference_root, fold_id, seed)
                source = read_json(reference / "REFERENCE_COMPLETE.json")
                required = {
                    "config_sha256": file_hash(args.config),
                    "reference_checkpoint_sha256": source[
                        "reference_checkpoint_sha256"
                    ],
                    "dictionary_sha256": file_hash(
                        reference / "geometry/DICTIONARIES.pt"
                    ),
                    "episode_sha256": registration["episode_contract_sha256"],
                    "execution_sha256": file_hash(__file__),
                    "engine_sha256": file_hash(
                        ROOT / "scripts/drs_peft/onesided_engine.py"
                    ),
                    "fold_id": fold_id,
                    "seed": seed,
                    "method": method,
                    "K": k,
                    "role": "own_fold_inner_only",
                    "query_ids": task_for(registration, "inner", fold_id)["query_ids"],
                }
                if selected["identity"] != required:
                    raise RuntimeError(
                        "A target hyperparameter source selection changed identity"
                    )
                path = folder / "TRIALS.csv"
                if file_hash(path) != selected["trials_csv_sha256"]:
                    raise RuntimeError("A source-inner tuning record changed")
                table = pd.read_csv(path)
                if any(
                    set(table[key]) != {value}
                    for key, value in {
                        "fold_id": fold_id,
                        "seed": seed,
                        "method": method,
                        "K": k,
                        "role": "inner",
                    }.items()
                ) or set(table.trial) != set(
                    range(len(hyperparameters(config, method)))
                ):
                    raise RuntimeError(
                        "Final source aggregation includes another task or incomplete hyperparameter grid"
                    )
                select_own_fold(table, fold_id, seed, method, k)
                if len(table) != len(hyperparameters(config, method)) * 5:
                    raise RuntimeError(
                        "Target selection requires every registered source-inner trial"
                    )
                tables.append(table)
                fingerprints.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": file_hash(path),
                    }
                )
            full = pd.concat(tables, ignore_index=True)
            means = full.groupby("trial", sort=True).mae.mean()
            winner = int(means.idxmin())
            selections[method][str(k)] = hyperparameters(config, method)[winner]
            scores[method][str(k)] = float(means.loc[winner])
    generic = min(
        GENERIC, key=lambda method: float(np.mean(list(scores[method].values())))
    )
    expected = {
        "status": "SOURCE_ONLY_TARGET_CHOICES_LOCKED",
        "candidate": "OUT",
        "best_generic": generic,
        "selected_hyperparameters": selections,
        "source_inner_selected_mae": scores,
        "source_inner_files": fingerprints,
        "training_seeds": config["training_seeds"],
        "config_sha256": file_hash(args.config),
        "episodes_sha256": registration["episode_contract_sha256"],
        "target_query_label_uses_for_selection": 0,
        "method_selection_source": "source_inner_only_equal_folds_seeds_draws_and_K",
        "target_methods": ["ZERO", "CAL2", "OUT", generic],
        "MATR_support_budget": "K_per_task;4K_across_four_tasks",
        "target_exposure_after_scoring": "fixed_Query_exposed_do_not_reuse_as_blind_selection",
    }
    bind(args.public_root / "TARGET_LOCK.json", expected)
    return expected


def target_predict(args, config, registration):
    lock = read_json(args.public_root / "TARGET_LOCK.json")
    if (
        lock["config_sha256"] != file_hash(args.config)
        or lock["episodes_sha256"] != registration["episode_contract_sha256"]
    ):
        raise RuntimeError("Target choices refer to another frozen study")
    methods = ["CAL2", "OUT", lock["best_generic"]]
    selections = {
        (method, k): lock["selected_hyperparameters"][method][str(k)]
        for method, k in itertools.product(methods, (1, 3, 5))
    }
    for seed in args.seeds:
        for domain in ("MATR", "HUST"):
            data = load_dataset(
                domain, cache_root=args.data_root, build_if_missing=False
            )
            engine, inputs, identity = prepare_engine(
                args, data, "final", seed, "D4_" + domain
            )
            reference = reference_folder(args.reference_root, "final", seed)
            register_methods(engine, reference)
            for task in registration["target_tasks"]:
                if task["domain"] == domain:
                    execute_task(
                        args,
                        config,
                        engine,
                        inputs,
                        data,
                        identity,
                        "D4",
                        task,
                        seed,
                        methods,
                        selections,
                        defer_score=True,
                    )


def expected_target_units(args, config, registration):
    lock = read_json(args.public_root / "TARGET_LOCK.json")
    units = []
    for task in registration["target_tasks"]:
        if task["domain"] not in config["D4"]["domains"]:
            continue
        for seed in config["training_seeds"]:
            units.append(unit_folder(args, "D4", task, seed, "ZERO", 0, 0))
            for method, k in itertools.product(
                ("CAL2", "OUT", lock["best_generic"]), (1, 3, 5)
            ):
                units.extend(
                    unit_folder(args, "D4", task, seed, method, k, draw["draw"])
                    for draw in shot_for(task, k)["draws"]
                )
    if len(units) != 690 or len(set(units)) != 690:
        raise RuntimeError(
            "The primary target matrix must contain 675 adaptations and 15 references"
        )
    return units


def target_score(args, config, registration):
    units = expected_target_units(args, config, registration)
    lock = read_json(args.public_root / "TARGET_LOCK.json")
    expected = {}
    for task in registration["target_tasks"]:
        if task["domain"] not in config["D4"]["domains"]:
            continue
        for seed in config["training_seeds"]:
            reference = reference_folder(args.reference_root, "final", seed)
            identity = read_json(reference / "REFERENCE_COMPLETE.json")
            identity["dictionary_sha256"] = file_hash(
                reference / "geometry/DICTIONARIES.pt"
            )
            specifications = [("ZERO", 0, shot_for(task, 0)["draws"][0], {})]
            specifications.extend(
                (method, k, draw, lock["selected_hyperparameters"][method][str(k)])
                for method, k in itertools.product(
                    ("CAL2", "OUT", lock["best_generic"]), (1, 3, 5)
                )
                for draw in shot_for(task, k)["draws"]
            )
            for method, k, draw, hyper in specifications:
                folder = unit_folder(args, "D4", task, seed, method, k, draw["draw"])
                expected[folder] = prediction_identity(
                    "D4",
                    task,
                    seed,
                    method,
                    k,
                    draw["draw"],
                    draw["support_ids"],
                    hyper,
                    identity,
                )
    receipts = []
    for folder in units:
        receipt = read_json(folder / "RECEIPT.json")
        if any(receipt.get(key) != value for key, value in expected[folder].items()):
            raise RuntimeError(
                "A target receipt differs from locked source, Query, Support or hyperparameters"
            )
        if receipt["status"] not in ("PREDICTIONS_LOCKED", "SCORED"):
            raise RuntimeError(
                "All target predictions must be locked before Query scoring"
            )
        if file_hash(ROOT / receipt["prediction_path"]) != receipt["prediction_sha256"]:
            raise RuntimeError("A target prediction changed before scoring")
        if file_hash(ROOT / receipt["state_path"]) != receipt["state_sha256"]:
            raise RuntimeError("A target adapted state changed before scoring")
        with np.load(ROOT / receipt["prediction_path"], allow_pickle=False) as archive:
            if (
                archive["ids"].astype(str).tolist() != receipt["query_ids"]
                or not np.array_equal(
                    archive["query"], np.arange(101, 5001, dtype=np.float32)
                )
                or archive["prediction"].shape != (len(receipt["query_ids"]), 4900)
                or not np.isfinite(archive["prediction"]).all()
            ):
                raise RuntimeError(
                    "A target prediction has incompatible identities or fixed coordinates"
                )
        receipts.append(receipt)
    exposure = args.public_root / "TARGET_QUERY_EXPOSURE.json"
    bind(
        exposure,
        {
            "status": "FIXED_TARGET_QUERY_EXPOSED_BY_REGISTERED_SCORING",
            "target_lock_sha256": file_hash(args.public_root / "TARGET_LOCK.json"),
            "all_target_prediction_units_locked_before_scoring": len(units),
            "prediction_receipts_sha256": payload_hash(
                [
                    {
                        "adaptation_id": item["adaptation_id"],
                        "prediction_sha256": item["prediction_sha256"],
                    }
                    for item in receipts
                ]
            ),
            "query_ids": {
                domain: sorted(
                    {
                        uid
                        for task in registration["target_tasks"]
                        if task["domain"] == domain
                        for uid in task["query_ids"]
                    }
                )
                for domain in config["D4"]["domains"]
            },
            "query_labels_used_for_adaptation_or_selection": 0,
            "reuse_as_blind_confirmation": False,
        },
    )
    for domain in config["D4"]["domains"]:
        data = load_dataset(domain, cache_root=args.data_root, build_if_missing=False)
        zeros = {}
        for folder, receipt in zip(units, receipts, strict=True):
            if receipt["domain"] == domain and receipt["method"] == "ZERO":
                with np.load(
                    ROOT / receipt["prediction_path"], allow_pickle=False
                ) as archive:
                    zeros[(receipt["task_id"], receipt["seed"])] = archive[
                        "prediction"
                    ].copy()
        for folder, receipt in zip(units, receipts, strict=True):
            if receipt["domain"] != domain or receipt["status"] == "SCORED":
                continue
            with np.load(
                ROOT / receipt["prediction_path"], allow_pickle=False
            ) as archive:
                prediction = archive["prediction"].copy()
            write_cell_metrics(
                args,
                data,
                folder,
                receipt,
                prediction,
                zeros[(receipt["task_id"], receipt["seed"])],
            )
    print(
        json.dumps(
            {
                "event": "PRIMARY_TARGET_QUERY_SCORING_COMPLETE",
                "units": len(units),
                "domains": config["D4"]["domains"],
            }
        ),
        flush=True,
    )


def collect_results(args):
    tables, receipts = [], []
    for path in sorted((args.public_root / "units").glob("**/RECEIPT.json")):
        receipt = read_json(path)
        receipts.append(receipt)
        metrics = path.parent / "CELL_METRICS.csv"
        if receipt["status"] == "SCORED":
            if file_hash(metrics) != receipt["cell_metrics_sha256"]:
                raise RuntimeError("A scored Query metric table changed")
            tables.append(pd.read_csv(metrics))
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(
            args.public_root / "CELL_METRICS.csv", index=False
        )
    args.public_root.mkdir(parents=True, exist_ok=True)
    (args.public_root / "ADAPTATION_RECORDS.jsonl").write_text(
        "".join(
            json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n"
            for receipt in receipts
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "RESULT_TABLES_COLLECTED",
                "receipts": len(receipts),
                "scored": len(tables),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "inner",
            "source",
            "spectrum-inner",
            "spectrum",
            "target-lock",
            "target-predict",
            "target-score",
            "collect",
        ),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--private-root", type=Path, default=PRIVATE)
    parser.add_argument("--public-root", type=Path, default=PUBLIC)
    parser.add_argument("--reference-root", type=Path, default=REFERENCES)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fold-ids", type=int, nargs="+", choices=(0, 1, 2), default=[0, 1, 2]
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", choices=(42, 123, 2026), default=[42, 123, 2026]
    )
    args = parser.parse_args()
    configure()
    config = read_json(args.config)
    registration = load_registration(ROOT / config["episodes_file"])
    if args.phase == "target-lock":
        target_lock(args, config, registration)
    elif args.phase == "target-predict":
        target_predict(args, config, registration)
    elif args.phase == "target-score":
        target_score(args, config, registration)
    elif args.phase == "collect":
        collect_results(args)
    else:
        data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
        for seed in args.seeds:
            if (
                args.phase.startswith("spectrum")
                and seed not in config["D3"]["training_seeds"]
            ):
                continue
            for fold_id in args.fold_ids:
                unit = (
                    args.private_root
                    / "execution"
                    / args.phase
                    / f"fold{fold_id}_seed{seed}"
                )
                with external_lock(args.private_root, unit):
                    if args.phase in ("inner", "spectrum-inner"):
                        tune_inner(
                            args,
                            config,
                            registration,
                            data,
                            fold_id,
                            seed,
                            args.phase == "spectrum-inner",
                        )
                    else:
                        execute_source(
                            args,
                            config,
                            registration,
                            data,
                            fold_id,
                            seed,
                            args.phase == "spectrum",
                        )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
