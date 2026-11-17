"""Run locked warmup/specialized-backbone comparisons using saved source references."""

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

from scripts.drs_peft.data import ROOT, file_hash, load_dataset, write_json
from scripts.drs_peft.onesided_protocol import load_registration, payload_hash
from scripts.drs_peft.run_onesided import (
    make_support,
    score_predictions,
    select_own_fold,
    shot_for,
    task_for,
)
from scripts.drs_peft.run_onesided_reference import (
    NoPolicyInputs,
    checked_checkpoint,
    external_lock,
    load_reference_model,
    reference_folder,
)
from scripts.drs_peft.specialization_engine import (
    SpecializationEngine,
    cache_xs_payload,
    load_pretrained_reference,
)
from scripts.drs_peft.train import bind, configure, read_json, save_torch

CONFIG = ROOT / "configs/drs_peft/specialization_comparison_v1/CONFIG.json"
NEW_METHODS = ("PRETRAINED_LORA", "SPECIALIZED_XS")
METHODS = ("PRETRAINED_LORA", "SPECIALIZED_LORA", "SPECIALIZED_XS", "SPECIALIZED_OUT")
ZEROS = ("WARMUP_ZERO", "SPECIALIZED_ZERO")
IMPLEMENTATIONS = (
    "scripts/drs_peft/run_specialization.py",
    "scripts/drs_peft/specialization_engine.py",
    "models/drs_peft/specialization.py",
    "scripts/drs_peft/onesided_engine.py",
    "scripts/drs_peft/onesided_calibration.py",
    "models/drs_peft/onesided.py",
    "scripts/drs_peft/run_onesided.py",
)


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def reference_family(method, config):
    for family, definition in config["reference_families"].items():
        if method == definition["zero_method"]:
            return family
    return config["method_specs"][method]["reference_family"]


def family_zero_method(method, config):
    return config["reference_families"][reference_family(method, config)]["zero_method"]


def trial_grid(config):
    adaptation = config["adaptation"]
    return [
        {"learning_rate": float(lr), "steps": int(steps), "eta": float(eta)}
        for lr, steps, eta in itertools.product(
            sorted(adaptation["learning_rates"]),
            sorted(adaptation["steps"]),
            sorted(adaptation["etas"]),
        )
    ]


def source_header(args, config, registration):
    return {
        "study_id": config["study_id"],
        "config_sha256": file_hash(args.config),
        "parent_config_sha256": config["parent_config_sha256"],
        "episodes_file_sha256": config["episodes_file_sha256"],
        "episode_contract_sha256": registration["episode_contract_sha256"],
        "implementation_sha256": {
            name: file_hash(ROOT / name) for name in IMPLEMENTATIONS
        },
    }


def load_contract(args):
    config = read_json(args.config)
    for path_key, hash_key in (
        ("parent_config_file", "parent_config_sha256"),
        ("episodes_file", "episodes_file_sha256"),
    ):
        if file_hash(ROOT / config[path_key]) != config[hash_key]:
            raise RuntimeError("An exact parent contract or episode file changed")
    registration = load_registration(ROOT / config["episodes_file"])
    if registration["episode_contract_sha256"] != config["episode_contract_sha256"]:
        raise RuntimeError("The episode payload differs from the registered study")
    parent_lock = ROOT / config["hyperparameter_selection"]["parent_target_lock_file"]
    if (
        file_hash(parent_lock)
        != config["hyperparameter_selection"]["parent_target_lock_sha256"]
        or read_json(parent_lock)["best_generic"] != "LoRA"
    ):
        raise RuntimeError("The inherited parent target lock changed")
    if (
        len(trial_grid(config)) != 12
        or config["references"]["new_source_reference_fits"] != 0
    ):
        raise ValueError(
            "The contrast requires twelve adaptation trials and no new reference fits"
        )
    return config, registration


def reference_identity(args, fold, seed, family):
    folder = reference_folder(args.reference_root, fold, seed)
    identity = read_json(folder / "REFERENCE_COMPLETE.json")
    key = (
        "warmup_checkpoint_sha256"
        if family == "WARMUP"
        else "reference_checkpoint_sha256"
    )
    stage = "warmup" if family == "WARMUP" else "static"
    if file_hash(folder / stage / "CHECKPOINT_SELECTED.pt") != identity[key]:
        raise RuntimeError("A frozen source checkpoint changed")
    return {
        "source_checkpoint_sha256": identity[key],
        "reference_family": family,
        "fold_id": fold,
        "seed": int(seed),
    }


def audit_common_reference(args, fold, seed):
    folder = reference_folder(args.reference_root, fold, seed)
    warmup, warmup_path = checked_checkpoint(folder / "warmup")
    specialized, specialized_path = checked_checkpoint(folder / "static")
    names = [
        name
        for name in warmup["model"]
        if not name.startswith("bridge.backbone.module.")
    ]
    if not names or any(
        name not in specialized["model"]
        or not torch.equal(warmup["model"][name], specialized["model"][name])
        for name in names
    ):
        raise RuntimeError(
            "The two reference families must share exact Bridge/Head tensors"
        )
    receipt = {
        "fold_id": fold,
        "seed": int(seed),
        "Bridge_Head_bit_exact": True,
        "Bridge_Head_parameters": 59585,
        "warmup_checkpoint_sha256": file_hash(warmup_path),
        "reference_checkpoint_sha256": file_hash(specialized_path),
        "new_source_reference_fits": 0,
    }
    audit_path = args.public_root / "reference_audits" / f"fold{fold}_seed{seed}.json"
    with external_lock(args.private_root, audit_path):
        bind(audit_path, receipt)


def prepare_engine(args, config, data, fold, seed, method, role):
    family = reference_family(method, config)
    audit_common_reference(args, fold, seed)
    identity = reference_identity(args, fold, seed, family)
    if family == "WARMUP":
        model = load_pretrained_reference(
            args.reference_root, args.backbone_dir, fold, seed
        )
    else:
        model = load_reference_model(args.reference_root, args.backbone_dir, fold, seed)
    inputs = NoPolicyInputs().transform(data)
    inputs["ids"] = list(data.ids)
    code = "P" if family == "WARMUP" else "X"
    engine = SpecializationEngine(
        model,
        inputs,
        data.query,
        identity,
        args.private_root / "e" / role / f"f{fold}s{seed}" / code,
        device=args.device,
        seed=seed,
    )
    engine.execution_id = f"{os.getpid()}_{time.time_ns()}"
    internal = "LoRA" if family == "WARMUP" else "LoRA-XS"
    if internal == "LoRA-XS":
        payload, construction = cache_xs_payload(
            model, args.private_root / "xs" / f"f{fold}s{seed}.pt"
        )
        engine.register_method(internal, payload)
        write_json(
            args.public_root
            / "XS_COSTS"
            / f"f{fold}s{seed}_{engine.execution_id}.json",
            {
                **construction,
                "execution_id": engine.execution_id,
                "role": role,
                "fold_id": fold,
                "seed": int(seed),
                "method": method,
                "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
                "new_SVD_seconds": float(construction["SVD_seconds"])
                if construction["new_SVD_evaluations"] > 0
                else 0.0,
                "counting_scope": "actual_new_factorizations_in_this_engine_construction;cached_reference_cost_retained_for_provenance",
            },
        )
    else:
        engine.register_method(internal)
    return engine, inputs, internal


def cost_snapshot(args, engine, role, fold, seed, method):
    write_json(
        args.public_root
        / "ENGINE_COSTS"
        / f"{role}_f{fold}s{seed}_{method}_{engine.execution_id}.json",
        {
            "role": role,
            "fold_id": fold,
            "seed": int(seed),
            "method": method,
            "counters": dict(engine.counters),
            "source_checkpoint_sha256": engine.source_sha256,
            "implementation_sha256": engine.implementation_sha256,
            "counting_scope": "cumulative_actual_new_execution_in_this_process_engine;selected_state_steps_counted_separately",
        },
    )


def adapted_state(engine, internal, support, hyper):
    selected = engine.nonlinear_trials(
        internal, support, hyper["learning_rate"], steps=(10, 30)
    )[int(hyper["steps"])]
    state = copy.deepcopy(selected["state"])
    state["adaptation_scale"] = float(hyper["eta"])
    details = dict(
        selected["details"],
        selected_optimizer_steps=int(hyper["steps"]),
        eta=float(hyper["eta"]),
    )
    return state, details


def inner_identity(args, config, registration, fold, seed, method, k):
    task = task_for(registration, "inner", fold)
    return {
        **source_header(args, config, registration),
        **reference_identity(args, fold, seed, reference_family(method, config)),
        "method": method,
        "K": int(k),
        "role": "inner",
        "query_ids": task["query_ids"],
    }


def tune_inner(args, config, registration, data, fold, seed, method):
    task = task_for(registration, "inner", fold)
    destination = args.public_root / "inner" / f"fold{fold}" / f"seed{seed}" / method
    with external_lock(args.private_root, destination):
        engine = None
        for k in (1, 3, 5):
            folder = destination / f"K{k}"
            identity = inner_identity(args, config, registration, fold, seed, method, k)
            marker = folder / "COMPLETE.json"
            if marker.exists():
                complete = read_json(marker)
                if (
                    complete["identity"] != identity
                    or file_hash(folder / "TRIALS.csv") != complete["trials_csv_sha256"]
                ):
                    raise RuntimeError(
                        "An inner trial table changed source or execution identity"
                    )
                continue
            if engine is None:
                engine, inputs, internal = prepare_engine(
                    args, config, data, fold, seed, method, "i"
                )
            records = []
            for trial, hyper in enumerate(trial_grid(config)):
                for draw in shot_for(task, k)["draws"]:
                    cache = (
                        args.private_root
                        / "i"
                        / f"f{fold}s{seed}"
                        / method
                        / f"K{k}"
                        / f"t{trial}d{draw['draw']}.json"
                    )
                    expected = dict(
                        identity,
                        trial=trial,
                        draw=draw["draw"],
                        hyperparameters=hyper,
                        support_ids=draw["support_ids"],
                    )
                    if cache.exists():
                        record = read_json(cache)
                        if record["identity"] != expected:
                            raise RuntimeError(
                                "An inner trial cache changed source or experimental identity"
                            )
                    else:
                        before = dict(engine.counters)
                        support = make_support(
                            data,
                            inputs,
                            sorted(draw["support_ids"]),
                            args.device,
                            config["adaptation"][
                                "maximum_support_observations_per_physical_cell"
                            ],
                        )
                        state, _details = adapted_state(
                            engine, internal, support, hyper
                        )
                        prediction = engine.predict_state(
                            internal, state, task["query_ids"]
                        )
                        _, summary = score_predictions(
                            data, task["query_ids"], prediction
                        )
                        record = {
                            "stage": "SPECIALIZATION",
                            "role": "inner",
                            "fold_id": fold,
                            "seed": int(seed),
                            "method": method,
                            "reference_family": reference_family(method, config),
                            "K": k,
                            "trial": trial,
                            "draw": draw["draw"],
                            "mae": float(summary["mae"]),
                            "hyperparameters": hyper,
                            "identity": expected,
                            "selected_optimizer_steps": int(hyper["steps"]),
                            "new_execution_counters": {
                                key: engine.counters[key] - before[key]
                                for key in before
                            },
                        }
                        write_json(cache, record)
                        cost_snapshot(args, engine, "inner", fold, seed, method)
                    records.append(
                        {
                            key: value
                            for key, value in record.items()
                            if key
                            not in (
                                "identity",
                                "hyperparameters",
                                "new_execution_counters",
                            )
                        }
                    )
            table = pd.DataFrame(records)
            select_own_fold(table, fold, seed, method, k)
            if len(table) != 60 or set(table.trial) != set(range(12)):
                raise RuntimeError(
                    "Every inner block requires twelve trials and five draws"
                )
            folder.mkdir(parents=True, exist_ok=True)
            table.to_csv(folder / "TRIALS.csv", index=False)
            write_json(
                marker,
                {
                    "identity": identity,
                    "trials_csv_sha256": file_hash(folder / "TRIALS.csv"),
                    "rows": len(table),
                },
            )
            print(
                json.dumps(
                    {
                        "event": "SPECIALIZATION_INNER_COMPLETE",
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "K": k,
                        "rows": len(table),
                    }
                ),
                flush=True,
            )


def aggregate_inner_trials(table, config):
    selections, records = {}, []
    required_methods = config["source_inner"]["new_methods"]
    expected_slots = config["source_inner"]["expected_scored_trial_slots"]
    if (
        len(table) != expected_slots
        or set(table.method) != set(required_methods)
        or set(table.role) != {"inner"}
    ):
        raise RuntimeError(
            "Only the complete registered source-inner matrix may select target settings"
        )
    grid = trial_grid(config)
    for method, k in itertools.product(required_methods, (1, 3, 5)):
        selected = table[table.method.eq(method) & table.K.eq(k)]
        for fold, seed in itertools.product(
            config["source_inner"]["fold_ids"], config["training_seeds"]
        ):
            block = selected[selected.fold_id.eq(fold) & selected.seed.eq(seed)]
            select_own_fold(block, fold, seed, method, k)
            if len(block) != 60 or set(block.trial) != set(range(12)):
                raise RuntimeError(
                    "Source-inner aggregation lacks a complete fold/seed/trial block"
                )
        means = selected.groupby("trial", sort=True).mae.mean()
        winner = int(means.idxmin())
        selections.setdefault(method, {})[str(k)] = grid[winner]
        records.append(
            {
                "method": method,
                "K": k,
                "selected_trial": winner,
                "source_inner_mae": float(means.loc[winner]),
                "folds": 3,
                "seeds": 3,
                "draws": 5,
            }
        )
    return {"selected_hyperparameters": selections, "selection_records": records}


def target_lock(args, config, registration):
    tables, files = [], []
    for method, fold, seed, k in itertools.product(
        config["source_inner"]["new_methods"],
        config["source_inner"]["fold_ids"],
        config["training_seeds"],
        (1, 3, 5),
    ):
        folder = (
            args.public_root
            / "inner"
            / f"fold{fold}"
            / f"seed{seed}"
            / method
            / f"K{k}"
        )
        complete = read_json(folder / "COMPLETE.json")
        if (
            complete["identity"]
            != inner_identity(args, config, registration, fold, seed, method, k)
            or file_hash(folder / "TRIALS.csv") != complete["trials_csv_sha256"]
        ):
            raise RuntimeError("The source-inner selection inputs changed")
        table = pd.read_csv(folder / "TRIALS.csv")
        if any(
            set(table[key]) != {value}
            for key, value in {
                "method": method,
                "fold_id": fold,
                "seed": seed,
                "K": k,
                "role": "inner",
            }.items()
        ):
            raise RuntimeError("A source-inner file contains another method or role")
        tables.append(table)
        files.append(
            {
                "path": relative(folder / "TRIALS.csv"),
                "sha256": file_hash(folder / "TRIALS.csv"),
            }
        )
    selected = aggregate_inner_trials(pd.concat(tables, ignore_index=True), config)
    parent_path = ROOT / config["hyperparameter_selection"]["parent_target_lock_file"]
    parent = read_json(parent_path)
    if (
        file_hash(parent_path)
        != config["hyperparameter_selection"]["parent_target_lock_sha256"]
        or parent["best_generic"] != "LoRA"
    ):
        raise RuntimeError("The inherited target choices changed")
    for method, old_method in config["hyperparameter_selection"][
        "reused_parent_methods"
    ].items():
        selected["selected_hyperparameters"][method] = parent[
            "selected_hyperparameters"
        ][old_method]
    lock = dict(
        source_header(args, config, registration),
        **selected,
        status="SOURCE_INNER_ONLY_CHOICES_LOCKED",
        source_inner_files=files,
        parent_target_lock_sha256=file_hash(parent_path),
        target_Query_selection_uses=0,
        source_outer_selection_uses=0,
        target_exposure=config["target"]["exposure_description"],
        blind_confirmation=False,
    )
    bind(args.public_root / "TARGET_LOCK.json", lock)
    return lock


def read_target_lock(args, config, registration):
    lock = read_json(args.public_root / "TARGET_LOCK.json")
    if any(
        lock.get(key) != value
        for key, value in source_header(args, config, registration).items()
    ):
        raise RuntimeError(
            "The target lock belongs to another source contract or implementation"
        )
    return lock


def target_units(args, config, registration):
    result = []
    for task in registration["target_tasks"]:
        if task["task_id"] not in config["target"]["task_ids"]:
            continue
        for seed in config["training_seeds"]:
            specifications = [
                (zero, 0, shot_for(task, 0)["draws"][0])
                for zero in config["target"]["zero_methods"]
            ]
            specifications.extend(
                (method, k, draw)
                for method, k in itertools.product(config["methods"], (1, 3, 5))
                for draw in shot_for(task, k)["draws"]
            )
            for method, k, draw in specifications:
                result.append(
                    {
                        "task": task,
                        "seed": int(seed),
                        "method": method,
                        "K": k,
                        "draw": draw["draw"],
                        "support_ids": draw["support_ids"],
                        "duplicate_of_draw": draw.get("duplicate_of_draw"),
                        "folder": args.public_root
                        / "units"
                        / task["task_id"]
                        / f"seed{seed}"
                        / method
                        / f"K{k}_draw{draw['draw']}",
                    }
                )
    expected = (
        config["target"]["expected_nonzero_slots"]
        + config["target"]["expected_family_zero_slots"]
    )
    if len(result) != expected or len({spec["folder"] for spec in result}) != expected:
        raise RuntimeError(
            "The target matrix must include all 900 adaptations and 30 family references"
        )
    keys = {
        (
            spec["method"],
            spec["task"]["task_id"],
            spec["seed"],
            spec["K"],
            tuple(sorted(spec["support_ids"])),
        )
        for spec in result
        if spec["K"]
    }
    new_keys = {
        key
        for key in keys
        if key[0] in config["target"]["new_methods_requiring_target_adaptation"]
    }
    if (
        len(keys) != config["target"]["expected_unique_nonzero_states"]
        or len(new_keys) != config["target"]["expected_new_unique_nonzero_states"]
    ):
        raise RuntimeError(
            "Registered duplicate Support sets differ from the locked state counts"
        )
    return result


def unit_identity(args, config, registration, specification, hyper):
    family = reference_family(specification["method"], config)
    return {
        **source_header(args, config, registration),
        **reference_identity(args, "final", specification["seed"], family),
        "stage": "SPECIALIZATION",
        "domain": specification["task"]["domain"],
        "task_id": specification["task"]["task_id"],
        "method": specification["method"],
        "K": specification["K"],
        "draw": specification["draw"],
        "support_ids": specification["support_ids"],
        "query_ids": specification["task"]["query_ids"],
        "hyperparameters": hyper,
    }


def state_identity(identity):
    result = {key: value for key, value in identity.items() if key != "draw"}
    result["support_ids"] = sorted(result["support_ids"])
    return result


def load_prediction(receipt):
    path = ROOT / receipt["prediction_path"]
    if (
        file_hash(path) != receipt["prediction_sha256"]
        or file_hash(ROOT / receipt["state_path"]) != receipt["state_sha256"]
    ):
        raise RuntimeError("A saved target state or prediction changed")
    with np.load(path, allow_pickle=False) as archive:
        if archive["ids"].astype(str).tolist() != receipt[
            "query_ids"
        ] or not np.array_equal(
            archive["query"], np.arange(101, 5001, dtype=np.float32)
        ):
            raise RuntimeError("A saved prediction uses another Query identity or grid")
        prediction = archive["prediction"].copy()
    if (
        prediction.shape != (len(receipt["query_ids"]), 4900)
        or not np.isfinite(prediction).all()
    ):
        raise RuntimeError(
            "Every target prediction must cover the full finite 4900-point grid"
        )
    return prediction


def zero_counters():
    return {
        "jacobian_builds": 0,
        "jacobian_cache_hits": 0,
        "geometry_solves": 0,
        "geometry_cache_hits": 0,
        "nonlinear_optimizer_updates": 0,
        "nonlinear_cache_hits": 0,
        "baseline_cell_forwards": 0,
        "adapted_cell_forwards": 0,
    }


def predict_unit(
    args, config, registration, engine, inputs, internal, data, specification, hyper
):
    identity = unit_identity(args, config, registration, specification, hyper)
    folder = specification["folder"]
    marker = folder / "RECEIPT.json"
    if marker.exists():
        receipt = read_json(marker)
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise RuntimeError("An existing target prediction changed locked identity")
        load_prediction(receipt)
        return
    started = time.perf_counter()
    before = dict(engine.counters)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    canonical = state_identity(identity)
    adaptation_id = payload_hash(canonical)
    private = args.private_root / "a" / adaptation_id
    private.mkdir(parents=True, exist_ok=True)
    state_path, prediction_path = private / "STATE.pt", private / "PREDICTIONS.npz"
    cached = state_path.exists() and prediction_path.exists()
    if cached:
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved["identity"] != canonical or saved["prediction_sha256"] != file_hash(
            prediction_path
        ):
            raise RuntimeError("The shared Support state cache changed identity")
        details, replay = saved["details"], saved["replay"]
    else:
        if specification["K"] == 0:
            state = None
            prediction = engine.baseline_prediction(identity["query_ids"])
            details, replay = (
                {"selected_optimizer_steps": 0, "trainable_parameters": 0},
                {"shared_family_reference": True},
            )
        else:
            support = make_support(
                data,
                inputs,
                sorted(identity["support_ids"]),
                args.device,
                config["adaptation"]["maximum_support_observations_per_physical_cell"],
            )
            state, details = adapted_state(engine, internal, support, hyper)
            prediction = engine.predict_state(internal, state, identity["query_ids"])
            replay = engine.replay_and_merge(internal, state, identity["query_ids"][:1])
            details.update(engine.support_diagnostics(internal, state, support))
        if not np.isfinite(prediction).all():
            raise FloatingPointError("The nonlinear target prediction is nonfinite")
        large = {key: details.pop(key) for key in list(details) if "predictions" in key}
        np.savez_compressed(
            prediction_path,
            ids=np.asarray(identity["query_ids"]),
            query=data.query,
            prediction=prediction,
        )
        save_torch(
            state_path,
            {
                "identity": canonical,
                "state": state,
                "details": details,
                "replay": replay,
                "support_predictions": large,
                "prediction_sha256": file_hash(prediction_path),
            },
        )
    receipt = {
        **identity,
        "status": "PREDICTIONS_LOCKED",
        "adaptation_id": adaptation_id,
        "prediction_path": relative(prediction_path),
        "prediction_sha256": file_hash(prediction_path),
        "state_path": relative(state_path),
        "state_sha256": file_hash(state_path),
        "is_duplicate": specification["duplicate_of_draw"] is not None,
        "duplicate_of_draw": specification["duplicate_of_draw"],
        "cache_reused": cached,
        "parent_reused": False,
        "details": details,
        "serialized_replay_and_merge": replay,
        "new_execution_counters": {
            key: engine.counters[key] - before[key] for key in before
        },
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if args.device.startswith("cuda")
        else None,
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved())
        if args.device.startswith("cuda")
        else None,
        "gpu_memory_scope": "current_process_context_unit_peak",
        "elapsed_seconds": time.perf_counter() - started,
        "target_Query_labels_used_for_adaptation_or_selection": 0,
        "historical_target_Query_exposure": True,
        "blind_confirmation": False,
        "source_commit": os.environ.get("BATTERY_SOURCE_COMMIT"),
    }
    load_prediction(receipt)
    write_json(marker, receipt)
    print(
        json.dumps(
            {
                "event": "SPECIALIZATION_TARGET_PREDICTION_LOCKED",
                "task": identity["task_id"],
                "seed": identity["seed"],
                "method": identity["method"],
                "K": identity["K"],
                "draw": identity["draw"],
                "cache_reused": cached,
            }
        ),
        flush=True,
    )


def target_predict(args, config, registration):
    lock = read_target_lock(args, config, registration)
    plans = target_units(args, config, registration)
    methods = [method for method in args.methods if method in NEW_METHODS]
    if "PRETRAINED_LORA" in methods:
        methods = ["WARMUP_ZERO", *methods]
    for seed, domain, method in itertools.product(
        args.seeds, config["target"]["domains"], methods
    ):
        selected = [
            spec
            for spec in plans
            if spec["seed"] == seed
            and spec["task"]["domain"] == domain
            and spec["method"] == method
        ]
        if not selected:
            continue
        group_path = (
            args.private_root / "execution" / f"target_{method}_{domain}_{seed}"
        )
        with external_lock(args.private_root, group_path):
            if all((spec["folder"] / "RECEIPT.json").exists() for spec in selected):
                for spec in selected:
                    hyper = (
                        {}
                        if not spec["K"]
                        else lock["selected_hyperparameters"][method][str(spec["K"])]
                    )
                    receipt = read_json(spec["folder"] / "RECEIPT.json")
                    if any(
                        receipt.get(key) != value
                        for key, value in unit_identity(
                            args, config, registration, spec, hyper
                        ).items()
                    ):
                        raise RuntimeError("An existing target group changed identity")
                    load_prediction(receipt)
                continue
            data = load_dataset(
                domain, cache_root=args.data_root, build_if_missing=False
            )
            engine, inputs, internal = prepare_engine(
                args, config, data, "final", seed, method, "t" + domain[0]
            )
            for spec in selected:
                hyper = (
                    {}
                    if not spec["K"]
                    else lock["selected_hyperparameters"][method][str(spec["K"])]
                )
                predict_unit(
                    args,
                    config,
                    registration,
                    engine,
                    inputs,
                    internal,
                    data,
                    spec,
                    hyper,
                )
                cost_snapshot(args, engine, "target_" + domain, "final", seed, method)


def reuse_parent_unit(args, config, registration, specification):
    started = time.perf_counter()
    method = specification["method"]
    old_method = (
        "ZERO"
        if method == "SPECIALIZED_ZERO"
        else config["method_specs"][method]["reuse_parent_target_method"]
    )
    parent_root = ROOT / config["target"]["parent_reuse_root"]
    old_folder = (
        parent_root
        / "units/D4"
        / specification["task"]["task_id"]
        / f"seed{specification['seed']}"
        / old_method
        / f"K{specification['K']}_draw{specification['draw']}"
    )
    parent = read_json(old_folder / "RECEIPT.json")
    lock = read_target_lock(args, config, registration)
    hyper = (
        {}
        if not specification["K"]
        else lock["selected_hyperparameters"][method][str(specification["K"])]
    )
    identity = unit_identity(args, config, registration, specification, hyper)
    expected_parent = {
        "stage": "D4",
        "task_id": identity["task_id"],
        "domain": identity["domain"],
        "seed": identity["seed"],
        "method": old_method,
        "K": identity["K"],
        "draw": identity["draw"],
        "support_ids": identity["support_ids"],
        "query_ids": identity["query_ids"],
        "hyperparameters": hyper,
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "config_sha256": config["parent_config_sha256"],
        "fold_id": "final",
        "execution_sha256": file_hash(ROOT / "scripts/drs_peft/run_onesided.py"),
        "engine_sha256": file_hash(ROOT / "scripts/drs_peft/onesided_engine.py"),
        "dictionary_sha256": file_hash(
            reference_folder(args.reference_root, "final", specification["seed"])
            / "geometry/DICTIONARIES.pt"
        ),
    }
    if parent["status"] != "SCORED" or any(
        parent.get(key) != value for key, value in expected_parent.items()
    ):
        raise RuntimeError(
            "The parent result differs in source, Support, Query, choices or completion"
        )
    load_prediction(parent)
    metrics_path = old_folder / "CELL_METRICS.csv"
    if file_hash(metrics_path) != parent["cell_metrics_sha256"]:
        raise RuntimeError("The authenticated parent Query metrics changed")
    folder, canonical = specification["folder"], state_identity(identity)
    adaptation_id = payload_hash(canonical)
    marker = folder / "RECEIPT.json"
    if marker.exists():
        existing = read_json(marker)
        if any(
            existing.get(key) != value for key, value in identity.items()
        ) or existing.get("parent_receipt_sha256") != file_hash(
            old_folder / "RECEIPT.json"
        ):
            raise RuntimeError("A reused parent unit changed identity")
        load_prediction(existing)
        if file_hash(folder / "CELL_METRICS.csv") != existing["cell_metrics_sha256"]:
            raise RuntimeError("Reused parent metrics changed")
        return existing
    table = pd.read_csv(metrics_path)
    for key in ("stage", "domain", "task_id", "fold_id", "seed", "method", "K", "draw"):
        if key in table and any(table[key].astype(str) != str(parent[key])):
            raise RuntimeError("Parent metric rows differ from the parent receipt")
        table[key] = identity[key]
    table["reference_family"] = "SPECIALIZED"
    table["adaptation_id"] = adaptation_id
    table["is_duplicate"] = specification["duplicate_of_draw"] is not None
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "CELL_METRICS.csv", index=False)
    receipt = dict(
        identity,
        status="PREDICTIONS_LOCKED",
        adaptation_id=adaptation_id,
        parent_reused=True,
        parent_receipt_path=relative(old_folder / "RECEIPT.json"),
        parent_receipt_sha256=file_hash(old_folder / "RECEIPT.json"),
        parent_state_identity_preserved=True,
        prediction_path=parent["prediction_path"],
        prediction_sha256=parent["prediction_sha256"],
        state_path=parent["state_path"],
        state_sha256=parent["state_sha256"],
        parent_cell_metrics_sha256=parent["cell_metrics_sha256"],
        cell_metrics_sha256=file_hash(folder / "CELL_METRICS.csv"),
        is_duplicate=specification["duplicate_of_draw"] is not None,
        duplicate_of_draw=specification["duplicate_of_draw"],
        details=dict(
            parent.get("details", {}),
            selected_optimizer_steps=0
            if not specification["K"] or old_method == "OUT"
            else hyper["steps"],
        ),
        new_execution_counters=zero_counters(),
        selected_state_budget_reused=True,
        elapsed_seconds=time.perf_counter() - started,
        new_adaptation_seconds=0.0,
        elapsed_scope="parent_authentication_and_metric_copy_without_model_forward_or_adaptation",
        peak_gpu_allocated_bytes=0,
        peak_gpu_reserved_bytes=0,
        target_Query_labels_used_for_adaptation_or_selection=0,
        historical_target_Query_exposure=True,
        blind_confirmation=False,
    )
    write_json(marker, receipt)
    return receipt


def reuse_parent(args, config, registration):
    read_target_lock(args, config, registration)
    for spec in target_units(args, config, registration):
        if (
            spec["seed"] not in args.seeds
            or spec["method"] not in config["target"]["parent_exact_reuse_methods"]
        ):
            continue
        if spec["method"] not in args.methods and spec["method"] != "SPECIALIZED_ZERO":
            continue
        with external_lock(args.private_root, spec["folder"]):
            reuse_parent_unit(args, config, registration, spec)


def write_family_metrics(data, folder, receipt, prediction, zero):
    identifiers = receipt["query_ids"]
    table, _ = score_predictions(data, identifiers, prediction)
    baseline, _ = score_predictions(data, identifiers, zero)
    if not table[["cell_uid", "window", "n_obs"]].equals(
        baseline[["cell_uid", "window", "n_obs"]]
    ):
        raise RuntimeError("Adapted and own-family ZERO scoring axes differ")
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
        "reference_family",
    ):
        table[key] = receipt[key]
    lookup = dict(zip(data.ids, data.protocol_ids, strict=True))
    counts = {
        uid: sum(lookup[uid] == lookup[support] for support in receipt["support_ids"])
        for uid in identifiers
    }
    table["same_protocol_support_count"] = table.cell_uid.map(counts)
    pending = folder / "CELL_METRICS.pending.csv"
    table.to_csv(pending, index=False)
    pending.replace(folder / "CELL_METRICS.csv")
    receipt.update(
        status="SCORED",
        cell_metrics_sha256=file_hash(folder / "CELL_METRICS.csv"),
        query_cells=len(identifiers),
        metrics_rows=len(table),
    )
    write_json(folder / "RECEIPT.json", receipt)


def target_score(args, config, registration):
    lock = read_target_lock(args, config, registration)
    specifications = target_units(args, config, registration)
    receipts = []
    for spec in specifications:
        receipt = read_json(spec["folder"] / "RECEIPT.json")
        hyper = (
            {}
            if not spec["K"]
            else lock["selected_hyperparameters"][spec["method"]][str(spec["K"])]
        )
        expected = unit_identity(args, config, registration, spec, hyper)
        if any(receipt.get(key) != value for key, value in expected.items()) or receipt[
            "status"
        ] not in ("PREDICTIONS_LOCKED", "SCORED"):
            raise RuntimeError(
                "A target unit differs from its locked source/family/Support/Query"
            )
        load_prediction(receipt)
        if receipt.get("parent_reused") and (
            file_hash(ROOT / receipt["parent_receipt_path"])
            != receipt["parent_receipt_sha256"]
            or file_hash(spec["folder"] / "CELL_METRICS.csv")
            != receipt["cell_metrics_sha256"]
        ):
            raise RuntimeError("A reused parent result changed before scoring")
        receipts.append(receipt)
    bind(
        args.public_root / "TARGET_QUERY_EXPOSURE.json",
        {
            "status": "ALL_930_TARGET_PREDICTIONS_LOCKED_BEFORE_NEW_QUERY_SCORING",
            "units_locked": len(receipts),
            "target_lock_sha256": file_hash(args.public_root / "TARGET_LOCK.json"),
            "predictions_sha256": payload_hash(
                [
                    {
                        key: receipt[key]
                        for key in (
                            "adaptation_id",
                            "prediction_sha256",
                            "reference_family",
                        )
                    }
                    for receipt in receipts
                ]
            ),
            "historical_target_Query_exposure": config["target"][
                "exposure_description"
            ],
            "blind_confirmation": False,
            "target_Query_selection_uses": 0,
        },
    )
    for domain in config["target"]["domains"]:
        relevant = [
            (spec, receipt)
            for spec, receipt in zip(specifications, receipts, strict=True)
            if receipt["domain"] == domain
        ]
        zeros = {
            (
                receipt["task_id"],
                receipt["seed"],
                receipt["reference_family"],
            ): load_prediction(receipt)
            for _, receipt in relevant
            if receipt["K"] == 0
        }
        need_new = any(
            receipt["status"] != "SCORED" and not receipt.get("parent_reused")
            for _, receipt in relevant
        )
        data = (
            load_dataset(domain, cache_root=args.data_root, build_if_missing=False)
            if need_new
            else None
        )
        for spec, receipt in relevant:
            if receipt["status"] == "SCORED":
                if (
                    file_hash(spec["folder"] / "CELL_METRICS.csv")
                    != receipt["cell_metrics_sha256"]
                ):
                    raise RuntimeError("A scored target table changed")
                continue
            if receipt.get("parent_reused"):
                receipt["status"] = "SCORED"
                receipt["query_cells"] = len(receipt["query_ids"])
                receipt["metrics_rows"] = len(
                    pd.read_csv(spec["folder"] / "CELL_METRICS.csv")
                )
                write_json(spec["folder"] / "RECEIPT.json", receipt)
                continue
            write_family_metrics(
                data,
                spec["folder"],
                receipt,
                load_prediction(receipt),
                zeros[
                    (receipt["task_id"], receipt["seed"], receipt["reference_family"])
                ],
            )


def collect(args, config, registration):
    tables, receipts = [], []
    for spec in target_units(args, config, registration):
        marker = spec["folder"] / "RECEIPT.json"
        if not marker.exists():
            continue
        receipt = read_json(marker)
        receipts.append(receipt)
        if receipt["status"] == "SCORED":
            path = spec["folder"] / "CELL_METRICS.csv"
            if file_hash(path) != receipt["cell_metrics_sha256"]:
                raise RuntimeError("A collected target Query metric table changed")
            tables.append(pd.read_csv(path))
    args.public_root.mkdir(parents=True, exist_ok=True)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(
            args.public_root / "CELL_METRICS.csv", index=False
        )
    (args.public_root / "ADAPTATION_RECORDS.jsonl").write_text(
        "".join(
            json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n"
            for receipt in receipts
        ),
        encoding="utf-8",
    )
    families = []
    for path in sorted((args.public_root / "reference_audits").glob("*.json")):
        audit = read_json(path)
        for family, definition in config["reference_families"].items():
            families.append(
                dict(
                    audit,
                    reference_family=family,
                    zero_method=definition["zero_method"],
                    source_core_parameters=definition["source_core_parameters"],
                    source_checkpoint_sha256=audit[
                        definition["checkpoint_identity_field"]
                    ],
                )
            )
    if families:
        pd.DataFrame(families).to_csv(
            args.public_root / "REFERENCE_FAMILIES.csv", index=False
        )
    pd.DataFrame(
        [
            dict(
                method=method,
                **{
                    key: value
                    for key, value in definition.items()
                    if key
                    in (
                        "reference_family",
                        "trainable_parameters",
                        "rank",
                        "alpha",
                        "scale",
                        "singular_values_retained",
                    )
                },
            )
            for method, definition in config["method_specs"].items()
        ]
    ).to_csv(args.public_root / "ADAPTER_BUDGETS.csv", index=False)
    costs = [
        read_json(path)
        for path in sorted((args.public_root / "ENGINE_COSTS").glob("*.json"))
    ]
    xs_events = [
        read_json(path)
        for path in sorted((args.public_root / "XS_COSTS").glob("*.json"))
    ]
    ledger = {
        **source_header(args, config, registration),
        "expected_target_units": 930,
        "target_units": len(receipts),
        "scored_units": len(tables),
        "parent_reused_units": sum(
            bool(receipt.get("parent_reused")) for receipt in receipts
        ),
        "unique_nonzero_states": len(
            {receipt["adaptation_id"] for receipt in receipts if receipt["K"]}
        ),
        "new_source_reference_fits": 0,
        "actual_new_optimizer_updates": sum(
            cost["counters"]["nonlinear_optimizer_updates"] for cost in costs
        ),
        "cost_counting_scope": "engine_process_cumulative_costs_not_selected_step_sum",
        "engine_cost_records": costs,
        "actual_new_SVD_evaluations": sum(
            event["new_SVD_evaluations"] for event in xs_events
        ),
        "actual_new_SVD_seconds": sum(event["new_SVD_seconds"] for event in xs_events),
        "XS_construction_events": xs_events,
        "target_exposure": config["target"]["exposure_description"],
        "blind_confirmation": False,
    }
    write_json(args.public_root / "EXECUTION_LEDGER.json", ledger)
    write_json(args.public_root / "RUN_LEDGER.json", ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "inner",
            "target-lock",
            "target-predict",
            "target-score",
            "reuse-parent",
            "collect",
        ),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--private-root",
        type=Path,
        default=ROOT / "runtime/drs_peft/specialization_comparison_v1",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=ROOT / "results/drs_peft/specialization_comparison_v1",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=ROOT / "runtime/drs_peft/onesided_fewshot_v1/references",
    )
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
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    configure()
    config, registration = load_contract(args)
    if args.phase == "inner":
        data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
        for fold, seed, method in itertools.product(
            args.fold_ids, args.seeds, args.methods
        ):
            if method in NEW_METHODS:
                tune_inner(args, config, registration, data, fold, seed, method)
    elif args.phase == "target-lock":
        target_lock(args, config, registration)
    elif args.phase == "target-predict":
        target_predict(args, config, registration)
    elif args.phase == "target-score":
        target_score(args, config, registration)
    elif args.phase == "reuse-parent":
        reuse_parent(args, config, registration)
    else:
        collect(args, config, registration)


if __name__ == "__main__":
    main()
