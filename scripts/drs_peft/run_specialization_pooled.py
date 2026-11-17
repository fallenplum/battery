"""Execute target calibration with one shared adapter and total-domain K support cells."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scripts.drs_peft import run_specialization as source
from scripts.drs_peft.data import ROOT, file_hash, load_dataset, write_json
from scripts.drs_peft.onesided_protocol import payload_hash
from scripts.drs_peft.run_onesided import make_support, shot_for
from scripts.drs_peft.run_onesided_reference import (
    NoPolicyInputs,
    external_lock,
    load_reference_model,
    reference_folder,
)
from scripts.drs_peft.specialization_engine import (
    SpecializationEngine,
    cache_xs_payload,
    load_pretrained_reference,
)
from scripts.drs_peft.specialization_protocol import load_target_contract
from scripts.drs_peft.train import bind, configure, read_json, save_torch

TARGET_CONTRACT = (
    ROOT / "configs/drs_peft/specialization_comparison_v1/TARGET_TOTAL_K.json"
)
STAGE = "SPECIALIZATION_TOTAL_K"
SOURCE_TRAINING_COMMIT = "421a3d81605d54456e50091564733dbab5821cba"


def source_args(args):
    return SimpleNamespace(
        config=args.config,
        public_root=args.source_public_root,
        private_root=args.source_private_root,
        reference_root=args.reference_root,
    )


def load_contract(args):
    _, override, episodes = load_target_contract(args.target_contract, args.config)
    config, original = source.load_contract(source_args(args))
    target_units(args, config, override, episodes)
    matr = [task for task in episodes["target_tasks"] if task["domain"] == "MATR"]
    hust = [task for task in episodes["target_tasks"] if task["domain"] == "HUST"]
    old_matr = [task for task in original["target_tasks"] if task["domain"] == "MATR"]
    old_hust = [task for task in original["target_tasks"] if task["domain"] == "HUST"]
    if len(matr) != 1 or len(hust) != 1 or hust != old_hust:
        raise RuntimeError("Expected one pooled MATR task and the unchanged HUST task")
    if set(matr[0]["query_ids"]) != {
        uid for task in old_matr for uid in task["query_ids"]
    } or set(matr[0]["support_pool_ids"]) != {
        uid for task in old_matr for uid in task["support_pool_ids"]
    }:
        raise RuntimeError(
            "MATR must retain the original eight Queries and 24 Support candidates"
        )
    return config, original, override, episodes


def header(args, config, original, override):
    return {
        **source.source_header(source_args(args), config, original),
        "source_training_commit": SOURCE_TRAINING_COMMIT,
        "target_study_id": override["study_id"],
        "target_override_file": source.relative(args.target_contract),
        "target_override_sha256": file_hash(args.target_contract),
        "target_episode_file_sha256": override["episodes_file_sha256"],
        "target_episode_contract_sha256": override["episode_contract_sha256"],
        "target_execution_sha256": file_hash(__file__),
        "target_protocol_code_sha256": file_hash(
            ROOT / "scripts/drs_peft/specialization_protocol.py"
        ),
        "support_budget_scope": "total_domain_K_one_shared_adapter",
    }


def target_units(args, config, override, episodes):
    result = []
    if len(episodes["target_tasks"]) != 2 or {
        task["task_id"] for task in episodes["target_tasks"]
    } != set(override["target"]["task_ids"]):
        raise RuntimeError(
            "The total-domain target contract requires exactly two tasks"
        )
    for task in episodes["target_tasks"]:
        pool, queries = set(task["support_pool_ids"]), set(task["query_ids"])
        if (
            pool & queries
            or len(pool) != len(task["support_pool_ids"])
            or len(queries) != len(task["query_ids"])
        ):
            raise ValueError(
                "Physical Support and Query pools must be unique and disjoint"
            )
        if task["domain"] == "MATR" and (len(pool), len(queries)) != (24, 8):
            raise ValueError(
                "MATR needs 24 pooled candidates and eight fixed Query cells"
            )
        by_shot = {k: shot_for(task, k)["draws"] for k in (1, 3, 5)}
        for k, draws in by_shot.items():
            if len(draws) != 5 or {draw["draw"] for draw in draws} != set(range(5)):
                raise RuntimeError("Each pooled Support budget requires all five draws")
            for draw in draws:
                if (
                    len(draw["support_ids"]) != k
                    or len(set(draw["support_ids"])) != k
                    or not set(draw["support_ids"]).issubset(pool)
                    or draw["query_ids"] != task["query_ids"]
                ):
                    raise RuntimeError(
                        "Each shared adapter must use exactly K total-domain Support cells"
                    )
        for draw_id in range(5):
            sets = [
                {
                    *next(draw for draw in by_shot[k] if draw["draw"] == draw_id)[
                        "support_ids"
                    ]
                }
                for k in (1, 3, 5)
            ]
            if not sets[0] <= sets[1] <= sets[2]:
                raise RuntimeError("Support draws must stay nested over K=1,3,5")
        for seed in config["training_seeds"]:
            specifications = [
                (method, 0, shot_for(task, 0)["draws"][0]) for method in source.ZEROS
            ]
            specifications.extend(
                (method, k, draw)
                for method, k in itertools.product(config["methods"], (1, 3, 5))
                for draw in by_shot[k]
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
        override["target"]["expected_nonzero_slots"]
        + override["target"]["expected_family_zero_slots"]
    )
    if (
        len(result) != 372
        or expected != 372
        or len({spec["folder"] for spec in result}) != 372
    ):
        raise RuntimeError(
            "The pooled matrix requires 360 nonzero units and twelve family references"
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
        (
            spec["method"],
            spec["task"]["task_id"],
            spec["seed"],
            spec["K"],
            tuple(sorted(spec["support_ids"])),
        )
        for spec in result
        if spec["K"] and not is_parent_reuse(spec)
    }
    if (
        len(keys) != override["target"]["expected_unique_nonzero_states"]
        or len(new_keys) != override["target"]["expected_new_unique_nonzero_states"]
    ):
        raise RuntimeError("The registered unique pooled Support states changed")
    return result


def target_lock(args, config, original, override):
    choice_path = args.source_public_root / "TARGET_LOCK.json"
    if choice_path.exists():
        choices = source.read_target_lock(source_args(args), config, original)
        for record in choices["source_inner_files"]:
            if file_hash(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError("A locked source-inner choice input changed")
    else:
        choices = source.target_lock(source_args(args), config, original)
    lock = {
        **header(args, config, original, override),
        "status": "SOURCE_CHOICES_REUSED_FOR_TOTAL_DOMAIN_K_TARGET",
        "source_choice_file": source.relative(choice_path),
        "source_choice_sha256": file_hash(choice_path),
        "selected_hyperparameters": choices["selected_hyperparameters"],
        "source_inner_trial_slots_reused": config["source_inner"][
            "expected_scored_trial_slots"
        ],
        "new_source_reference_fits": 0,
        "new_source_inner_trials": 0,
        "target_Query_selection_uses": 0,
        "MATR_support_budget": "one_shared_adapter_K_total_over_24_candidates",
        "HUST_support_budget": "unchanged_one_shared_adapter_K_total_over_38_candidates",
        "old_MATR_protocol_specific_nonzero_predictions_reused": False,
        "final_OUT_dictionary_sha256": {
            str(seed): file_hash(
                reference_folder(args.reference_root, "final", seed)
                / "geometry/DICTIONARIES.pt"
            )
            for seed in config["training_seeds"]
        },
        "historical_target_Query_exposure": True,
        "blind_confirmation": False,
    }
    bind(args.public_root / "TARGET_LOCK.json", lock)
    return lock


def read_target_lock(args, config, original, override):
    lock = read_json(args.public_root / "TARGET_LOCK.json")
    if (
        any(
            lock.get(key) != value
            for key, value in header(args, config, original, override).items()
        )
        or file_hash(ROOT / lock["source_choice_file"]) != lock["source_choice_sha256"]
    ):
        raise RuntimeError("The source choices or total-domain target override changed")
    return lock


def prepare_engine(args, config, data, seed, method, domain, lock):
    family = source.reference_family(method, config)
    source.audit_common_reference(args, "final", seed)
    identity = source.reference_identity(args, "final", seed, family)
    model = (
        load_pretrained_reference(args.reference_root, args.backbone_dir, "final", seed)
        if family == "WARMUP"
        else load_reference_model(args.reference_root, args.backbone_dir, "final", seed)
    )
    inputs = NoPolicyInputs().transform(data)
    inputs["ids"] = list(data.ids)
    method_code = {
        "PRETRAINED_LORA": "P",
        "SPECIALIZED_LORA": "L",
        "SPECIALIZED_XS": "X",
        "SPECIALIZED_OUT": "O",
        "WARMUP_ZERO": "W0",
        "SPECIALIZED_ZERO": "S0",
    }[method]
    engine = SpecializationEngine(
        model,
        inputs,
        data.query,
        identity,
        args.private_root / "e" / domain[0] / str(seed) / method_code,
        device=args.device,
        seed=seed,
    )
    engine.execution_id = f"{os.getpid()}_{time.time_ns()}"
    if method in source.ZEROS:
        return engine, inputs, None
    internal = {
        "PRETRAINED_LORA": "LoRA",
        "SPECIALIZED_LORA": "LoRA",
        "SPECIALIZED_XS": "LoRA-XS",
        "SPECIALIZED_OUT": "OUT",
    }[method]
    if internal == "LoRA-XS":
        payload, construction = cache_xs_payload(
            model, args.private_root / "xs" / f"s{seed}.pt"
        )
        engine.register_method(internal, payload)
        write_json(
            args.public_root / "XS_COSTS" / f"s{seed}_{engine.execution_id}.json",
            dict(
                construction,
                execution_id=engine.execution_id,
                domain=domain,
                seed=int(seed),
                method=method,
                new_SVD_seconds=float(construction["SVD_seconds"])
                if construction["new_SVD_evaluations"]
                else 0.0,
            ),
        )
    elif internal == "OUT":
        path = (
            reference_folder(args.reference_root, "final", seed)
            / "geometry/DICTIONARIES.pt"
        )
        if file_hash(path) != lock["final_OUT_dictionary_sha256"][str(seed)]:
            raise RuntimeError("The original frozen OUT dictionary changed")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if bundle["arms"]["OUT"]["dimension"] != 16:
            raise RuntimeError(
                "The registered specialized OUT arm requires sixteen coordinates"
            )
        engine.register_method(internal, bundle["arms"]["OUT"])
    else:
        engine.register_method(internal)
    return engine, inputs, internal


def unit_identity(args, config, original, override, spec, hyper, lock):
    return {
        **header(args, config, original, override),
        **source.reference_identity(
            args, "final", spec["seed"], source.reference_family(spec["method"], config)
        ),
        "source_choice_sha256": lock["source_choice_sha256"],
        "stage": STAGE,
        "domain": spec["task"]["domain"],
        "task_id": spec["task"]["task_id"],
        "method": spec["method"],
        "K": spec["K"],
        "draw": spec["draw"],
        "support_ids": spec["support_ids"],
        "query_ids": spec["task"]["query_ids"],
        "hyperparameters": hyper,
        "adapter_dictionary_sha256": lock["final_OUT_dictionary_sha256"][
            str(spec["seed"])
        ]
        if spec["method"] == "SPECIALIZED_OUT"
        else None,
    }


def is_parent_reuse(spec):
    return spec["task"]["domain"] == "HUST" and spec["method"] in (
        "SPECIALIZED_LORA",
        "SPECIALIZED_OUT",
        "SPECIALIZED_ZERO",
    )


def adapted_state(engine, internal, support, hyper):
    if internal == "OUT":
        state, details = engine.geometry_trial(internal, support, hyper)
        return state, dict(details, selected_optimizer_steps=0)
    return source.adapted_state(engine, internal, support, hyper)


def predict_unit(
    args, config, original, override, data, engine, inputs, internal, spec, hyper, lock
):
    identity = unit_identity(args, config, original, override, spec, hyper, lock)
    marker = spec["folder"] / "RECEIPT.json"
    if marker.exists():
        receipt = read_json(marker)
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise RuntimeError("An existing pooled target unit changed identity")
        source.load_prediction(receipt)
        return
    started, before = time.perf_counter(), dict(engine.counters)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    canonical = source.state_identity(identity)
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
            raise RuntimeError("The pooled Support state cache changed")
        details, replay = saved["details"], saved["replay"]
    else:
        query_diagnostics = None
        if spec["K"] == 0:
            state, prediction = None, engine.baseline_prediction(identity["query_ids"])
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
            if internal == "OUT":
                query_diagnostics = engine.query_diagnostics(
                    internal, state, identity["query_ids"][:1], prediction[:1]
                )
                details["query_gap_MAE"] = query_diagnostics["query_gap_MAE"]
                details["query_diagnostic_scope"] = "first_Query_cell_full_4900_grid"
        if not np.isfinite(prediction).all():
            raise FloatingPointError(
                "The real nonlinear pooled target prediction is nonfinite"
            )
        large = {
            key: details.pop(key)
            for key in list(details)
            if "predictions" in key or "linearization_gap" in key
        }
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
                "support_diagnostics": large,
                "query_diagnostics": query_diagnostics,
                "prediction_sha256": file_hash(prediction_path),
            },
        )
    receipt = dict(
        identity,
        status="PREDICTIONS_LOCKED",
        adaptation_id=adaptation_id,
        prediction_path=source.relative(prediction_path),
        prediction_sha256=file_hash(prediction_path),
        state_path=source.relative(state_path),
        state_sha256=file_hash(state_path),
        is_duplicate=spec["duplicate_of_draw"] is not None,
        duplicate_of_draw=spec["duplicate_of_draw"],
        cache_reused=cached,
        parent_reused=False,
        details=details,
        serialized_replay_and_merge=replay,
        new_execution_counters={
            key: engine.counters[key] - before[key] for key in before
        },
        peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated())
        if args.device.startswith("cuda")
        else None,
        peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved())
        if args.device.startswith("cuda")
        else None,
        gpu_memory_scope="current_process_context_unit_peak",
        elapsed_seconds=time.perf_counter() - started,
        target_Query_labels_used_for_adaptation_or_selection=0,
        historical_target_Query_exposure=True,
        blind_confirmation=False,
    )
    source.load_prediction(receipt)
    write_json(marker, receipt)
    print(
        json.dumps(
            {
                "event": "POOLED_TARGET_PREDICTION_LOCKED",
                "domain": identity["domain"],
                "seed": identity["seed"],
                "method": identity["method"],
                "K": identity["K"],
                "draw": identity["draw"],
            }
        ),
        flush=True,
    )


def target_predict(args, config, original, override, episodes):
    lock = read_target_lock(args, config, original, override)
    plans = target_units(args, config, override, episodes)
    methods = list(
        dict.fromkeys(
            [
                *(source.family_zero_method(method, config) for method in args.methods),
                *args.methods,
            ]
        )
    )
    for seed, domain, method in itertools.product(args.seeds, args.domains, methods):
        selected = [
            spec
            for spec in plans
            if spec["seed"] == seed
            and spec["task"]["domain"] == domain
            and spec["method"] == method
            and not is_parent_reuse(spec)
        ]
        if not selected:
            continue
        with external_lock(
            args.private_root,
            args.private_root / "execution" / f"{domain}_{method}_{seed}",
        ):
            remaining = []
            for spec in selected:
                hyper = (
                    {}
                    if not spec["K"]
                    else lock["selected_hyperparameters"][method][str(spec["K"])]
                )
                marker = spec["folder"] / "RECEIPT.json"
                if marker.exists():
                    receipt = read_json(marker)
                    if any(
                        receipt.get(key) != value
                        for key, value in unit_identity(
                            args, config, original, override, spec, hyper, lock
                        ).items()
                    ):
                        raise RuntimeError(
                            "A resumed pooled target group changed identity"
                        )
                    source.load_prediction(receipt)
                else:
                    remaining.append((spec, hyper))
            if not remaining:
                continue
            data = load_dataset(
                domain, cache_root=args.data_root, build_if_missing=False
            )
            engine, inputs, internal = prepare_engine(
                args, config, data, seed, method, domain, lock
            )
            for spec, hyper in remaining:
                predict_unit(
                    args,
                    config,
                    original,
                    override,
                    data,
                    engine,
                    inputs,
                    internal,
                    spec,
                    hyper,
                    lock,
                )
                source.cost_snapshot(
                    args, engine, "pooled_target_" + domain, "final", seed, method
                )


def reuse_parent_unit(args, config, original, override, spec, lock):
    started = time.perf_counter()
    if not is_parent_reuse(spec):
        raise ValueError(
            "Only unchanged HUST specialized outputs can reuse parent results"
        )
    old_method = (
        "ZERO"
        if spec["method"] == "SPECIALIZED_ZERO"
        else config["method_specs"][spec["method"]]["reuse_parent_target_method"]
    )
    folder = (
        ROOT
        / config["target"]["parent_reuse_root"]
        / "units/D4"
        / spec["task"]["task_id"]
        / f"seed{spec['seed']}"
        / old_method
        / f"K{spec['K']}_draw{spec['draw']}"
    )
    parent = read_json(folder / "RECEIPT.json")
    hyper = (
        {}
        if not spec["K"]
        else lock["selected_hyperparameters"][spec["method"]][str(spec["K"])]
    )
    identity = unit_identity(args, config, original, override, spec, hyper, lock)
    expected = {
        key: identity[key]
        for key in (
            "domain",
            "task_id",
            "seed",
            "K",
            "draw",
            "support_ids",
            "query_ids",
            "hyperparameters",
            "source_checkpoint_sha256",
        )
    }
    expected.update(
        stage="D4",
        fold_id="final",
        method=old_method,
        config_sha256=config["parent_config_sha256"],
        execution_sha256=file_hash(ROOT / "scripts/drs_peft/run_onesided.py"),
        engine_sha256=file_hash(ROOT / "scripts/drs_peft/onesided_engine.py"),
        dictionary_sha256=lock["final_OUT_dictionary_sha256"][str(spec["seed"])],
    )
    if parent["status"] != "SCORED" or any(
        parent.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(
            "The HUST parent source, Support, Query or adaptation choice changed"
        )
    source.load_prediction(parent)
    metrics = folder / "CELL_METRICS.csv"
    if file_hash(metrics) != parent["cell_metrics_sha256"]:
        raise RuntimeError("The authenticated HUST parent metrics changed")
    marker = spec["folder"] / "RECEIPT.json"
    if marker.exists():
        receipt = read_json(marker)
        if any(receipt.get(key) != value for key, value in identity.items()) or receipt[
            "parent_receipt_sha256"
        ] != file_hash(folder / "RECEIPT.json"):
            raise RuntimeError("A reused HUST unit changed identity")
        source.load_prediction(receipt)
        if (
            file_hash(spec["folder"] / "CELL_METRICS.csv")
            != receipt["cell_metrics_sha256"]
        ):
            raise RuntimeError("Reused HUST metric rows changed")
        return
    table = pd.read_csv(metrics)
    for key in ("stage", "domain", "task_id", "fold_id", "seed", "method", "K", "draw"):
        if key not in table or not table[key].astype(str).eq(str(parent[key])).all():
            raise RuntimeError("HUST parent metric rows differ from their receipt")
        table[key] = identity[key]
    adaptation_id = payload_hash(source.state_identity(identity))
    table["reference_family"], table["adaptation_id"] = "SPECIALIZED", adaptation_id
    table["is_duplicate"] = spec["duplicate_of_draw"] is not None
    spec["folder"].mkdir(parents=True, exist_ok=True)
    table.to_csv(spec["folder"] / "CELL_METRICS.csv", index=False)
    receipt = dict(
        identity,
        status="PREDICTIONS_LOCKED",
        adaptation_id=adaptation_id,
        parent_reused=True,
        parent_receipt_path=source.relative(folder / "RECEIPT.json"),
        parent_receipt_sha256=file_hash(folder / "RECEIPT.json"),
        prediction_path=parent["prediction_path"],
        prediction_sha256=parent["prediction_sha256"],
        state_path=parent["state_path"],
        state_sha256=parent["state_sha256"],
        parent_cell_metrics_sha256=parent["cell_metrics_sha256"],
        cell_metrics_sha256=file_hash(spec["folder"] / "CELL_METRICS.csv"),
        is_duplicate=spec["duplicate_of_draw"] is not None,
        duplicate_of_draw=spec["duplicate_of_draw"],
        details=dict(
            parent.get("details", {}), selected_optimizer_steps=hyper.get("steps", 0)
        ),
        new_execution_counters=source.zero_counters(),
        new_adaptation_seconds=0.0,
        elapsed_seconds=time.perf_counter() - started,
        elapsed_scope="HUST_parent_authentication_and_metric_copy_without_new_forward_or_adaptation",
        peak_gpu_allocated_bytes=0,
        peak_gpu_reserved_bytes=0,
        target_Query_labels_used_for_adaptation_or_selection=0,
        historical_target_Query_exposure=True,
        blind_confirmation=False,
    )
    write_json(marker, receipt)


def reuse_parent(args, config, original, override, episodes):
    lock = read_target_lock(args, config, original, override)
    for spec in target_units(args, config, override, episodes):
        if (
            not is_parent_reuse(spec)
            or spec["seed"] not in args.seeds
            or "HUST" not in args.domains
            or (
                spec["method"] not in args.methods
                and spec["method"] != "SPECIALIZED_ZERO"
            )
        ):
            continue
        with external_lock(args.private_root, spec["folder"]):
            reuse_parent_unit(args, config, original, override, spec, lock)


def target_score(args, config, original, override, episodes):
    lock = read_target_lock(args, config, original, override)
    plans = target_units(args, config, override, episodes)
    receipts = []
    for spec in plans:
        receipt = read_json(spec["folder"] / "RECEIPT.json")
        hyper = (
            {}
            if not spec["K"]
            else lock["selected_hyperparameters"][spec["method"]][str(spec["K"])]
        )
        if (
            any(
                receipt.get(key) != value
                for key, value in unit_identity(
                    args, config, original, override, spec, hyper, lock
                ).items()
            )
            or receipt["status"] not in ("PREDICTIONS_LOCKED", "SCORED")
            or bool(receipt.get("parent_reused")) != is_parent_reuse(spec)
        ):
            raise RuntimeError(
                "A pooled target unit changed source, family, budget or prediction origin"
            )
        source.load_prediction(receipt)
        if receipt.get("parent_reused") and (
            file_hash(ROOT / receipt["parent_receipt_path"])
            != receipt["parent_receipt_sha256"]
            or file_hash(spec["folder"] / "CELL_METRICS.csv")
            != receipt["cell_metrics_sha256"]
        ):
            raise RuntimeError("A reused HUST parent result changed before scoring")
        receipts.append(receipt)
    bind(
        args.public_root / "TARGET_QUERY_EXPOSURE.json",
        {
            "status": "ALL_372_TOTAL_DOMAIN_K_PREDICTIONS_LOCKED_BEFORE_SCORING",
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
            "support_budget_scope": "total_domain_K_one_shared_adapter",
            "historical_target_Query_exposure": True,
            "blind_confirmation": False,
            "target_Query_selection_uses": 0,
        },
    )
    for domain in ("MATR", "HUST"):
        relevant = [
            (spec, receipt)
            for spec, receipt in zip(plans, receipts, strict=True)
            if receipt["domain"] == domain
        ]
        zeros = {
            (receipt["seed"], receipt["reference_family"]): source.load_prediction(
                receipt
            )
            for _, receipt in relevant
            if not receipt["K"]
        }
        data = (
            load_dataset(domain, cache_root=args.data_root, build_if_missing=False)
            if any(
                receipt["status"] != "SCORED" and not receipt.get("parent_reused")
                for _, receipt in relevant
            )
            else None
        )
        for spec, receipt in relevant:
            if receipt["status"] == "SCORED":
                if (
                    file_hash(spec["folder"] / "CELL_METRICS.csv")
                    != receipt["cell_metrics_sha256"]
                ):
                    raise RuntimeError("A scored pooled target table changed")
            elif receipt.get("parent_reused"):
                receipt.update(
                    status="SCORED",
                    query_cells=len(receipt["query_ids"]),
                    metrics_rows=len(pd.read_csv(spec["folder"] / "CELL_METRICS.csv")),
                )
                write_json(spec["folder"] / "RECEIPT.json", receipt)
            else:
                source.write_family_metrics(
                    data,
                    spec["folder"],
                    receipt,
                    source.load_prediction(receipt),
                    zeros[(receipt["seed"], receipt["reference_family"])],
                )


def collect(args, config, original, override, episodes):
    receipts, tables = [], []
    for spec in target_units(args, config, override, episodes):
        marker = spec["folder"] / "RECEIPT.json"
        if not marker.exists():
            continue
        receipt = read_json(marker)
        receipts.append(receipt)
        if receipt["status"] == "SCORED":
            path = spec["folder"] / "CELL_METRICS.csv"
            if file_hash(path) != receipt["cell_metrics_sha256"]:
                raise RuntimeError("A pooled target Query metric table changed")
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
    costs = [
        read_json(path)
        for path in sorted((args.public_root / "ENGINE_COSTS").glob("*.json"))
    ]
    events = [
        read_json(path)
        for path in sorted((args.public_root / "XS_COSTS").glob("*.json"))
    ]
    source_costs = [
        read_json(path)
        for path in sorted((args.source_public_root / "ENGINE_COSTS").glob("*.json"))
    ]
    source_ledger = args.source_public_root / "EXECUTION_LEDGER.json"
    ledger = dict(
        header(args, config, original, override),
        expected_target_units=372,
        target_units=len(receipts),
        scored_units=len(tables),
        parent_reused_units=sum(
            bool(receipt.get("parent_reused")) for receipt in receipts
        ),
        actual_new_optimizer_updates=sum(
            cost["counters"]["nonlinear_optimizer_updates"] for cost in costs
        ),
        engine_cost_records=costs,
        actual_new_SVD_evaluations=sum(
            event["new_SVD_evaluations"] for event in events
        ),
        actual_new_SVD_seconds=sum(event["new_SVD_seconds"] for event in events),
        XS_construction_events=events,
        new_source_reference_fits=0,
        new_source_inner_trials=0,
        source_choice_file=source.relative(
            args.source_public_root / "TARGET_LOCK.json"
        ),
        source_choice_sha256=file_hash(args.source_public_root / "TARGET_LOCK.json"),
        source_actual_new_optimizer_updates=sum(
            cost["counters"]["nonlinear_optimizer_updates"] for cost in source_costs
        ),
        source_engine_cost_records=source_costs,
        source_execution_ledger=source.relative(source_ledger)
        if source_ledger.exists()
        else None,
        source_execution_ledger_sha256=file_hash(source_ledger)
        if source_ledger.exists()
        else None,
        historical_target_Query_exposure=True,
        blind_confirmation=False,
    )
    write_json(args.public_root / "EXECUTION_LEDGER.json", ledger)
    write_json(args.public_root / "RUN_LEDGER.json", ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "target-lock",
            "target-predict",
            "reuse-parent",
            "target-score",
            "collect",
        ),
        required=True,
    )
    parser.add_argument("--target-contract", type=Path, default=TARGET_CONTRACT)
    parser.add_argument("--config", type=Path, default=source.CONFIG)
    parser.add_argument(
        "--source-public-root",
        type=Path,
        default=ROOT / "results/drs_peft/specialization_comparison_v1",
    )
    parser.add_argument(
        "--source-private-root",
        type=Path,
        default=ROOT / "runtime/drs_peft/specialization_comparison_v1",
    )
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--private-root", type=Path)
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
        "--seeds", nargs="+", type=int, choices=(42, 123, 2026), default=[42, 123, 2026]
    )
    parser.add_argument(
        "--methods", nargs="+", choices=source.METHODS, default=list(source.METHODS)
    )
    parser.add_argument(
        "--domains", nargs="+", choices=("MATR", "HUST"), default=["MATR", "HUST"]
    )
    args = parser.parse_args()
    override = read_json(args.target_contract)
    args.public_root = args.public_root or ROOT / override["outputs"]["public_root"]
    args.private_root = args.private_root or ROOT / override["outputs"]["private_root"]
    configure()
    config, original, override, episodes = load_contract(args)
    if args.phase == "target-lock":
        target_lock(args, config, original, override)
    elif args.phase == "target-predict":
        target_predict(args, config, original, override, episodes)
    elif args.phase == "reuse-parent":
        reuse_parent(args, config, original, override, episodes)
    elif args.phase == "target-score":
        target_score(args, config, original, override, episodes)
    else:
        collect(args, config, original, override, episodes)


if __name__ == "__main__":
    main()
