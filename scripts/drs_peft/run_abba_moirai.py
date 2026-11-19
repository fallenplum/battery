"""Reproduce ABBA on Moirai and evaluate locked physical-battery few-shot episodes."""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.drs_peft.abba_engine import (
    AdapterExperiment,
    merged_reference,
    restore,
    snapshot,
    support_batch,
)
from scripts.drs_peft.data import CACHE, ROOT, file_hash, write_json
from scripts.drs_peft.onesided_protocol import load_registration, payload_hash
from scripts.drs_peft.run_onesided import task_for
from scripts.drs_peft.run_onesided_reference import (
    load_reference_model,
    reference_folder,
)
from scripts.drs_peft.run_transfer_scope import draws
from scripts.drs_peft.specialization_protocol import load_target_contract
from scripts.drs_peft.train import (
    bind,
    configure,
    evaluate_arrays,
    predict,
    read_json,
    save_torch,
)
from scripts.drs_peft.transfer_scope_data import load_future_rows, load_prefixes
from scripts.drs_peft.transfer_scope_reference import (
    BACKBONE,
    REFERENCES,
    old_zero,
    save_npz,
)

CONFIG = ROOT / "configs/drs_peft/abba_moirai_v1/CONFIG.json"
PUBLIC = ROOT / "results/drs_peft/abba_moirai_v1"
PRIVATE = ROOT / "runtime/drs_peft/abba_moirai_v1"
REPORT = ROOT / "report/drs_peft/abba_moirai_v1"
ZERO = "ZERO"


def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.csv")
    pd.DataFrame(rows).to_csv(pending, index=False, lineterminator="\n")
    pending.replace(path)


def parents(args):
    cfg = read_json(args.config)
    src = load_registration(ROOT / cfg["source_episodes"])
    _, _, target = load_target_contract(ROOT / cfg["target_contract"])
    if (
        target["shots"] != cfg["target"]["shots"]
        or target["training_seeds"] != cfg["target"]["seeds"]
    ):
        raise RuntimeError(
            "The experiment must retain the frozen target K and seed protocol"
        )
    if set(src["source_ids"]) & set(src["sealed_final_MATR_ids"]):
        raise RuntimeError("Source and held-out MATR roles overlap")
    return cfg, src, target


def code_identity(args):
    files = (
        "models/drs_peft/abba_adapters.py",
        "models/drs_peft/abba_adalora.py",
        "scripts/drs_peft/abba_engine.py",
        "scripts/drs_peft/run_abba_moirai.py",
        "scripts/drs_peft/transfer_scope_data.py",
    )
    return {
        "config_sha256": file_hash(args.config),
        "code": {p: file_hash(ROOT / p) for p in files},
    }


def checkpoint(args, fold, seed):
    root = reference_folder(args.reference_root, fold, seed)
    item = read_json(root / "REFERENCE_COMPLETE.json")
    path = root / "static/CHECKPOINT_SELECTED.pt"
    if (
        item["final_target_query_labels_used"] != 0
        or file_hash(path) != item["reference_checkpoint_sha256"]
    ):
        raise RuntimeError("Frozen source ancestry changed")
    return item


def models(args, fold, seed):
    model = load_reference_model(
        args.reference_root, args.backbone_dir, fold, seed, "cpu"
    )
    return model, merged_reference(model)


def qpredict(model, data, ids, device):
    return predict(model, data.inputs, data.rows(ids), data.query, device)


def p0(args, cfg, source):
    marker = args.public_root / "P0_MOIRAI_REPRODUCTION.json"
    code = code_identity(args)
    if marker.exists():
        if read_json(marker)["implementation"] != code:
            raise RuntimeError(
                "Existing P0 operator evidence belongs to another implementation"
            )
        return
    data = load_prefixes("MATR", args.data_root)
    task = task_for(source, "inner", 0)
    ids = task["query_ids"][:2]
    support_ids = next(d["support_ids"] for k, d in draws(task) if k == 3)
    if set(support_ids + ids) - set(source["source_ids"]):
        raise RuntimeError("P0 uses registered source inner only")
    original, merged = models(args, 0, 42)
    original_prediction = qpredict(original.to(args.device), data, ids, args.device)
    original.cpu()
    merged_prediction = qpredict(merged.to(args.device), data, ids, args.device)
    merged.cpu()
    if not np.allclose(original_prediction, merged_prediction, atol=2e-6, rtol=0):
        raise RuntimeError("Source core merge failed real Moirai equivalence")
    from models.drs_peft.abba_adapters import UPSTREAM_ADAPTERS

    evidence = {}
    for method, options in cfg["methods"].items():
        operator = AdapterExperiment(merged, method, 42, cfg, args.device)
        initial = qpredict(operator.model, data, ids, args.device)
        if not np.allclose(initial, merged_prediction, atol=2e-6, rtol=0):
            raise RuntimeError(f"Initial output changed for {method}")
        batch = support_batch(data, args.data_root, support_ids, args.device, 512)
        metrics = operator.fit(batch, cfg["source"]["learning_rates"][-1])
        pred = qpredict(operator.model, data, ids, args.device)
        if not np.any(initial != pred):
            raise RuntimeError(
                f"Real Moirai fitting did not change predictions for {method}"
            )
        saved = snapshot(operator.model)
        rows, active = operator.spectrum(metrics)
        restore(operator.model, operator.initial)
        restore(operator.model, saved)
        reload = qpredict(operator.model, data, ids, args.device)
        errors = [
            float(np.max(np.abs(initial - merged_prediction))),
            float(np.max(np.abs(pred - reload))),
        ]
        for name, layer in operator.layers.items():
            with torch.no_grad():
                weight = layer.base.weight + layer.delta_weight()
                x = torch.linspace(-1, 1, weight.shape[1], device=args.device).reshape(
                    1, -1
                )
                torch.testing.assert_close(
                    layer(x),
                    torch.nn.functional.linear(x, weight, layer.base.bias),
                    atol=3e-6,
                    rtol=3e-6,
                )
        if errors[1] > 2e-6:
            raise RuntimeError("Adapter-only save/load failed")
        evidence[method] = dict(
            metrics,
            active_parameters=active,
            maximum_update_rank=max(r["numerical_rank_rtol_1e_5"] for r in rows),
            initial_prediction_max_abs=errors[0],
            restored_prediction_max_abs=errors[1],
        )
        print(
            f"P0 {method} passes real Moirai, support MSE {metrics['support_mse_before']:.6g}->{metrics['support_mse_after']:.6g}",
            flush=True,
        )
        del operator
    write_json(
        marker,
        {
            "status": "PASS_REAL_MOIRAI",
            "implementation": code,
            "source_checkpoint_sha256": checkpoint(args, 0, 42)[
                "reference_checkpoint_sha256"
            ],
            "upstream": UPSTREAM_ADAPTERS,
            "merged_reference_max_abs": float(
                np.max(np.abs(original_prediction - merged_prediction))
            ),
            "source_support_ids": support_ids,
            "source_prefix_query_ids": ids,
            "target_label_rows_loaded": 0,
            "new_reference_fits": 0,
            "real_optimizer_updates": len(evidence) * cfg["optimization"]["steps"],
            "methods": evidence,
        },
    )


def source_path(args, fold, method):
    return args.public_root / "source" / f"fold{fold}" / method


def source_run(args, cfg, source):
    primary = [m for m in args.methods if not cfg["methods"][m].get("diagnostic_only")]
    prefixes = load_prefixes("MATR", args.data_root)
    source_uids = set(source["source_ids"])
    for fold in args.folds:
        seed = cfg["source"]["seeds"][0]
        task = task_for(source, "inner", fold)
        query = task["query_ids"]
        if set(query + task["support_pool_ids"]) - source_uids:
            raise RuntimeError("Source selection cannot read held-out outcomes")
        y, mask = load_future_rows("MATR", args.data_root, query)
        if not mask.any(axis=1).all():
            raise RuntimeError("Each source Query requires observed future capacity")
        pending = [
            m
            for m in primary
            if not (source_path(args, fold, m) / "COMPLETE.json").exists()
        ]
        if not pending:
            continue
        _, merged = models(args, fold, seed)
        reference = checkpoint(args, fold, seed)
        if set(query + task["support_pool_ids"]) & set(reference["fit_ids"]):
            raise RuntimeError("Source inner and reference fit roles overlap")
        for method in pending:
            folder = source_path(args, fold, method)
            operator = AdapterExperiment(merged, method, seed, cfg, args.device)
            identity = dict(
                code_identity(args),
                method=method,
                fold=fold,
                seed=seed,
                query_ids=query,
                reference_sha256=reference["reference_checkpoint_sha256"],
                data_manifest_sha256=prefixes.cache_manifest_sha256,
            )
            cache, records = {}, []
            for k, draw in draws(task):
                ids = sorted(draw["support_ids"])
                support = support_batch(
                    prefixes,
                    args.data_root,
                    ids,
                    args.device,
                    cfg["input"]["maximum_observations_per_support_cell"],
                )
                for rate in cfg["source"]["learning_rates"]:
                    key = (tuple(ids), rate)
                    if key not in cache:
                        trial = (
                            folder
                            / "trials"
                            / (payload_hash({"ids": ids, "rate": rate}) + ".json")
                        )
                        signature = dict(identity, support_ids=ids, learning_rate=rate)
                        if trial.exists():
                            record = read_json(trial)
                            if record["identity"] != signature:
                                raise RuntimeError(
                                    "A source trial changed its input identity"
                                )
                        else:
                            diagnostics = operator.fit(support, rate)
                            pred = qpredict(
                                operator.model, prefixes, query, args.device
                            )
                            mae = float(
                                np.mean(
                                    [
                                        np.abs(p[m].astype(float) - t[m]).mean()
                                        for p, t, m in zip(pred, y, mask, strict=True)
                                    ]
                                )
                            )
                            record = dict(identity=signature, mae=mae, **diagnostics)
                            write_json(trial, record)
                        cache[key] = record
                    records.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "K": k,
                            "draw": draw["draw"],
                            "method": method,
                            "learning_rate": rate,
                            "mae": cache[key]["mae"],
                            "support_ids_json": list(ids),
                        }
                    )
                print(
                    f"source {method} fold={fold} K{k} draw={draw['draw']}", flush=True
                )
            csv(folder / "TRIALS.csv", records)
            write_json(
                folder / "COMPLETE.json",
                dict(
                    identity=identity,
                    scores_sha256=file_hash(folder / "TRIALS.csv"),
                    slots=len(records),
                    unique_fits=len(cache),
                    optimizer_updates=sum(
                        r["actual_optimizer_steps"] for r in cache.values()
                    ),
                    fit_seconds=sum(r["fit_seconds"] for r in cache.values()),
                    target_label_rows_loaded=0,
                ),
            )
            del operator


def selection(args, cfg, target, *, bind_lock):
    tables, sources, parents_sha = [], {}, {}
    experiment = code_identity(args)
    p0_path = args.public_root / "P0_MOIRAI_REPRODUCTION.json"
    if read_json(p0_path)["implementation"] != experiment:
        raise RuntimeError("P0 does not authenticate the active PEFT operators")
    for method, options in cfg["methods"].items():
        if options.get("diagnostic_only"):
            continue
        for fold in cfg["source"]["folds"]:
            base = source_path(args, fold, method)
            row = read_json(base / "COMPLETE.json")
            if any(row["identity"][k] != v for k, v in experiment.items()) or row[
                "scores_sha256"
            ] != file_hash(base / "TRIALS.csv"):
                raise RuntimeError("Source selection scores or implementation changed")
            frame = pd.read_csv(base / "TRIALS.csv")
            if (
                len(frame) != 30
                or set(frame.method) != {method}
                or set(frame.fold) != {fold}
            ):
                raise RuntimeError("Incomplete source screening matrix")
            tables.append(frame)
            sources[f"fold{fold}/{method}"] = file_hash(base / "COMPLETE.json")
    for seed in cfg["target"]["seeds"]:
        parents_sha[str(seed)] = checkpoint(args, "final", seed)[
            "reference_checkpoint_sha256"
        ]
    facts = pd.concat(tables, ignore_index=True)
    keys = ["fold", "seed", "method", "K", "draw", "learning_rate"]
    if (
        len(facts) != cfg["source"]["expected_score_slots"]
        or facts.duplicated(keys).any()
    ):
        raise RuntimeError("Source search counts changed")
    scores = (
        facts.groupby(["method", "K", "learning_rate"])
        .mae.agg(["mean", "size"])
        .reset_index()
    )
    if not scores["size"].eq(15).all():
        raise RuntimeError(
            "Every source learning rate requires all 15 registered slots"
        )
    choices = {}
    for method, k in itertools.product(cfg["methods"], (1, 3, 5)):
        comparison = (
            "LORA_R8" if cfg["methods"][method].get("diagnostic_only") else method
        )
        ordered = scores[(scores.method == comparison) & (scores.K == k)].sort_values(
            ["mean", "learning_rate"]
        )
        choices.setdefault(method, {})[str(k)] = float(ordered.iloc[0].learning_rate)
    selection_path = args.public_root / "SOURCE_SELECTION.csv"
    if bind_lock:
        csv(selection_path, scores)
    episode_path = ROOT / read_json(ROOT / cfg["target_contract"])["episodes_file"]
    lock = dict(
        experiment,
        status="LOCKED_BEFORE_TARGET_EVALUATION",
        episode_contract_sha256=target["episode_contract_sha256"],
        episodes_sha256=file_hash(episode_path),
        target_tasks=target["target_tasks"],
        target_seeds=cfg["target"]["seeds"],
        specialized_reference_sha256_by_seed=parents_sha,
        source_receipt_sha256=sources,
        p0_sha256=file_hash(p0_path),
        source_score_slots=len(facts),
        source_selection_sha256=file_hash(selection_path),
        learning_rates=choices,
        historical_target_query_exposure=True,
        target_query_selection_uses=0,
    )
    if bind_lock:
        bind(args.public_root / "TARGET_LOCK.json", lock)
        print("Source selection and both target cohorts locked", flush=True)
    elif read_json(args.public_root / "TARGET_LOCK.json") != lock:
        raise RuntimeError("Active execution does not match the sealed target plan")
    return lock


def target_slots(args, cfg, episodes):
    for task, seed in itertools.product(
        episodes["target_tasks"], cfg["target"]["seeds"]
    ):
        domain = task["domain"]
        for method in [ZERO, *cfg["methods"]]:
            prefix = args.public_root / "target" / domain / f"s{seed}" / method
            if method == ZERO:
                yield (
                    prefix / "0_0",
                    task,
                    seed,
                    method,
                    0,
                    {"draw": 0, "support_ids": []},
                )
            else:
                for k, draw in draws(task):
                    yield prefix / f"{k}_{draw['draw']}", task, seed, method, k, draw


def unit_identity(lock_sha, lock, task, seed, method, k, draw):
    return {
        "lock_sha256": lock_sha,
        "domain": task["domain"],
        "task_id": task["task_id"],
        "seed": seed,
        "method": method,
        "K": k,
        "draw": draw["draw"],
        "support_ids": sorted(draw["support_ids"]),
        "query_ids": task["query_ids"],
        "reference_family": "SPECIALIZED",
        "reference_sha256": lock["specialized_reference_sha256_by_seed"][str(seed)],
        "learning_rate": lock["learning_rates"][method][str(k)] if k else None,
        "is_duplicate": draw.get("duplicate_of_draw") is not None,
    }


def read_prediction(args, receipt):
    path = args.private_root / receipt["prediction_path"]
    if file_hash(path) != receipt["prediction_sha256"]:
        raise RuntimeError("Locked prediction was modified")
    if (
        receipt["state_path"]
        and file_hash(args.private_root / receipt["state_path"])
        != receipt["state_sha256"]
    ):
        raise RuntimeError("Locked adapter state was modified")
    with np.load(path, allow_pickle=False) as data:
        prediction = data["prediction"]
        if data["ids"].tolist() != receipt["query_ids"] or not np.array_equal(
            data["grid"], np.arange(101, 5001, dtype=np.float32)
        ):
            raise RuntimeError("Prediction cell or cycle axis changed")
    if (
        prediction.shape != (len(receipt["query_ids"]), 4900)
        or not np.isfinite(prediction).all()
    ):
        raise RuntimeError("Nonfinite or incomplete physical Query prediction")
    return prediction


def target_predict(args, cfg, episodes, lock):
    lock_sha = file_hash(args.public_root / "TARGET_LOCK.json")
    plans = [
        (folder, task, seed, method, k, draw)
        for folder, task, seed, method, k, draw in target_slots(args, cfg, episodes)
        if task["domain"] in args.domains
        and seed in args.seeds
        and method in args.methods
    ]
    for task in episodes["target_tasks"]:
        domain = task["domain"]
        relevant = [plan for plan in plans if plan[1]["domain"] == domain]
        if not relevant:
            continue
        data = load_prefixes(domain, args.data_root)
        for seed in args.seeds:
            seed_plans = [
                plan
                for plan in relevant
                if plan[2] == seed and not (plan[0] / "PREDICTION_LOCKED.json").exists()
            ]
            if not seed_plans:
                continue
            original, merged = models(args, "final", seed)
            for method in args.methods:
                method_plans = [plan for plan in seed_plans if plan[3] == method]
                if not method_plans:
                    continue
                if method == ZERO:
                    original.to(args.device)
                    merged.to(args.device)
                    fresh = qpredict(original, data, task["query_ids"], args.device)
                    joined = qpredict(merged, data, task["query_ids"], args.device)
                    original.cpu()
                    merged.cpu()
                    prediction, zero = old_zero(task, seed)
                    if (
                        zero["source_checkpoint_sha256"]
                        != lock["specialized_reference_sha256_by_seed"][str(seed)]
                        or not np.allclose(fresh, prediction, atol=2e-6, rtol=0)
                        or not np.allclose(joined, prediction, atol=2e-6, rtol=0)
                    ):
                        raise RuntimeError(
                            "Authenticated parent ZERO is not the loaded reference"
                        )
                else:
                    operator = AdapterExperiment(merged, method, seed, cfg, args.device)
                for folder, _, _, _, k, draw in method_plans:
                    identity = unit_identity(
                        lock_sha, lock, task, seed, method, k, draw
                    )
                    canonical = {
                        key: value
                        for key, value in identity.items()
                        if key not in {"draw", "is_duplicate"}
                    }
                    state_id = payload_hash(canonical)
                    run = Path("adaptations") / state_id[:24]
                    state_dir = args.private_root / run
                    record_path = state_dir / "COMPLETE.json"
                    cached = record_path.exists()
                    if cached:
                        facts = read_json(record_path)
                        if facts["canonical_identity"] != canonical:
                            raise RuntimeError(
                                "Cached target state belongs to another episode"
                            )
                        read_prediction(args, dict(identity, **facts))
                    else:
                        started = time.perf_counter()
                        if method == ZERO:
                            detail = dict(
                                trainable_parameters=0,
                                active_parameters=0,
                                optimizer_steps=0,
                                actual_optimizer_steps=0,
                                support_mse_before=None,
                                support_mse_after=None,
                                original_replay_max_abs=float(
                                    np.max(np.abs(fresh - prediction))
                                ),
                                merged_replay_max_abs=float(
                                    np.max(np.abs(joined - prediction))
                                ),
                                parent_prediction_sha256=zero["prediction_sha256"],
                            )
                            spectra = pd.DataFrame(
                                columns=[
                                    "module",
                                    "rows",
                                    "cols",
                                    "frobenius",
                                    "relative_frobenius",
                                    "spectral_norm",
                                    "numerical_rank_rtol_1e_5",
                                    "stable_rank",
                                    "entropy_rank",
                                    "energy_rank_99",
                                    "singular_values_json",
                                    "allocated_rank",
                                    "rank_bound",
                                    "nonzero_svd_triplets",
                                ]
                            )
                            state_file = None
                        else:
                            ids = canonical["support_ids"]
                            batch = support_batch(
                                data,
                                args.data_root,
                                ids,
                                args.device,
                                cfg["input"]["maximum_observations_per_support_cell"],
                            )
                            detail = operator.fit(batch, identity["learning_rate"])
                            prediction = qpredict(
                                operator.model, data, task["query_ids"], args.device
                            )
                            spectra, active = operator.spectrum(detail)
                            detail["active_parameters"] = active
                            state_file = run / "ADAPTER.pt"
                            save_torch(
                                args.private_root / state_file,
                                dict(
                                    identity=canonical,
                                    adapter_state=snapshot(operator.model),
                                    allocated_rank_by_module=detail[
                                        "allocated_rank_by_module"
                                    ],
                                ),
                            )
                            del batch
                        pred_file = run / "PREDICTIONS.npz"
                        save_npz(
                            args.private_root / pred_file,
                            ids=np.asarray(task["query_ids"]),
                            grid=data.query,
                            prediction=prediction,
                        )
                        csv(state_dir / "UPDATE_SPECTRUM.csv", spectra)
                        facts = dict(
                            detail,
                            canonical_identity=canonical,
                            adaptation_id=state_id,
                            prediction_path=pred_file.as_posix(),
                            prediction_sha256=file_hash(args.private_root / pred_file),
                            state_path=state_file.as_posix() if state_file else None,
                            state_sha256=file_hash(args.private_root / state_file)
                            if state_file
                            else None,
                            update_spectrum_sha256=file_hash(
                                state_dir / "UPDATE_SPECTRUM.csv"
                            ),
                            runtime_seconds=time.perf_counter() - started,
                            hardware=torch.cuda.get_device_name()
                            if args.device.startswith("cuda")
                            else "CPU",
                            target_query_future_rows_loaded=0,
                        )
                        write_json(record_path, facts)
                    folder.mkdir(parents=True, exist_ok=True)
                    table = state_dir / "UPDATE_SPECTRUM.csv"
                    if file_hash(table) != facts["update_spectrum_sha256"]:
                        raise RuntimeError("Cached update spectrum changed")
                    (folder / "UPDATE_SPECTRUM.csv").write_bytes(table.read_bytes())
                    receipt = dict(identity, **facts, status="PREDICTIONS_LOCKED")
                    receipt["actual_optimizer_steps"] = (
                        0
                        if identity["is_duplicate"]
                        else facts["actual_optimizer_steps"]
                    )
                    receipt["prediction_from_existing_cache"] = cached
                    write_json(folder / "PREDICTION_LOCKED.json", receipt)
                    print(
                        f"target {domain} seed={seed} {method} K{k} draw={draw['draw']} cached={cached}",
                        flush=True,
                    )
                if method != ZERO:
                    del operator
            del original, merged


def score(args, cfg, episodes, lock):
    lock_sha = file_hash(args.public_root / "TARGET_LOCK.json")
    records = []
    for folder, task, seed, method, k, draw in target_slots(args, cfg, episodes):
        receipt = read_json(folder / "PREDICTION_LOCKED.json")
        if any(
            receipt[key] != value
            for key, value in unit_identity(
                lock_sha, lock, task, seed, method, k, draw
            ).items()
        ):
            raise RuntimeError(
                "Target envelope does not match the locked physical evaluation"
            )
        read_prediction(args, receipt)
        if (
            file_hash(folder / "UPDATE_SPECTRUM.csv")
            != receipt["update_spectrum_sha256"]
            or receipt["target_query_future_rows_loaded"]
        ):
            raise RuntimeError(
                "Target prediction integrity or Query visibility violation"
            )
        records.append((folder, task, receipt))
    nonzero = [record for _, _, record in records if record["K"]]
    if (
        len(records) != cfg["target"]["expected_scored_slots"]
        or len({x["adaptation_id"] for x in nonzero})
        != cfg["target"]["expected_unique_adaptations"]
    ):
        raise RuntimeError("Missing or unregistered adaptation identities")
    bind(
        args.public_root / "TARGET_PREDICTIONS_COMPLETE.json",
        {
            "status": "ALL_AUTHENTICATED_BEFORE_QUERY_LABEL_READ",
            "lock_sha256": lock_sha,
            "slots": len(records),
            "unique_nonzero": len({x["adaptation_id"] for x in nonzero}),
            "prediction_manifest_sha256": payload_hash(
                [x["prediction_sha256"] for _, _, x in records]
            ),
            "new_target_optimizer_updates": sum(
                x["actual_optimizer_steps"] for x in nonzero
            ),
            "bootstrap_draws": 0,
        },
    )
    scoring = {}
    for folder, task, receipt in records:
        marker = folder / "RECEIPT.json"
        if marker.exists():
            result = read_json(marker)
            if (
                result["metrics_sha256"] != file_hash(folder / "METRICS.csv")
                or result["prediction_sha256"] != receipt["prediction_sha256"]
            ):
                raise RuntimeError("Existing scores no longer match locked predictions")
            continue
        domain, seed = receipt["domain"], receipt["seed"]
        uids = receipt["query_ids"]
        if domain not in scoring:
            scoring[domain] = load_future_rows(domain, args.data_root, uids)
        y, mask = scoring[domain]
        q = np.arange(101, 5001, dtype=np.float32)
        prediction = read_prediction(args, receipt)
        protocols = task["protocol_by_id"]
        metrics, _ = evaluate_arrays(
            prediction,
            y,
            mask,
            q,
            uids,
            [protocols[uid] for uid in uids],
            "original_cycle_id",
        )
        zero_path = (
            args.public_root / "target" / domain / f"s{seed}/ZERO/0_0/METRICS.csv"
        )
        if receipt["method"] == ZERO:
            metrics["zero_mae"] = metrics["mae"]
        else:
            baseline = pd.read_csv(zero_path, float_precision="round_trip")
            if not baseline[["cell_uid", "window", "n_obs"]].equals(
                metrics[["cell_uid", "window", "n_obs"]]
            ):
                raise RuntimeError(
                    "ZERO scoring uses another physical observation mask"
                )
            metrics["zero_mae"] = baseline["mae"]
        metrics["gain_mae"] = metrics["zero_mae"] - metrics["mae"]
        metrics["negative_transfer"] = (
            metrics["gain_mae"].lt(0).where(metrics["n_obs"].gt(0))
        )
        metrics["same_protocol_support_count"] = metrics.cell_uid.map(
            {
                uid: sum(protocols[uid] == protocols[s] for s in receipt["support_ids"])
                for uid in uids
            }
        )
        for k in (
            "domain",
            "seed",
            "method",
            "K",
            "draw",
            "is_duplicate",
            "adaptation_id",
            "reference_family",
            "task_id",
        ):
            metrics[k] = receipt[k]
        metrics["prediction_scope"] = "HISTORICALLY_EXPOSED_FIXED_TARGET_QUERY"
        csv(folder / "METRICS.csv", metrics)
        write_json(
            marker,
            dict(
                receipt,
                status="SCORED",
                metrics_sha256=file_hash(folder / "METRICS.csv"),
                scored_cell_window_rows=len(metrics),
            ),
        )
    print(f"Scored {len(records)} registered target slots", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=["p0", "source", "lock", "target-predict", "target-score"],
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--public-root", type=Path, default=PUBLIC)
    parser.add_argument("--private-root", type=Path, default=PRIVATE)
    parser.add_argument("--reference-root", type=Path, default=REFERENCES)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg, source, episodes = parents(args)
    args.folds = args.folds or cfg["source"]["folds"]
    args.seeds = args.seeds or cfg["target"]["seeds"]
    args.domains = args.domains or cfg["target"]["domains"]
    args.methods = args.methods or [ZERO, *cfg["methods"]]
    if (
        not set(args.folds) <= set(cfg["source"]["folds"])
        or not set(args.seeds) <= set(cfg["target"]["seeds"])
        or not set(args.domains) <= set(cfg["target"]["domains"])
        or not set(args.methods) <= {ZERO, *cfg["methods"]}
    ):
        raise ValueError("Run filters may only narrow the registered matrix")
    configure()
    with threadpool_limits(limits=2):
        if args.phase == "p0":
            p0(args, cfg, source)
        elif args.phase == "source":
            if read_json(args.public_root / "P0_MOIRAI_REPRODUCTION.json")[
                "implementation"
            ] != code_identity(args):
                raise RuntimeError(
                    "Verify the active operators through P0 before source selection"
                )
            args.methods = [m for m in args.methods if m != ZERO]
            source_run(args, cfg, source)
        elif args.phase == "lock":
            selection(args, cfg, episodes, bind_lock=True)
        else:
            lock = selection(args, cfg, episodes, bind_lock=False)
            if args.phase == "target-predict":
                target_predict(args, cfg, episodes, lock)
            else:
                score(args, cfg, episodes, lock)


if __name__ == "__main__":
    main()
