"""Run the source-only, nested protocol screen of spectrum-preserving rotation."""

from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.drs_peft.data import CACHE, ROOT, file_hash, write_json
from scripts.drs_peft.run_abba_moirai import csv, models
from scripts.drs_peft.spectral_rotation import build_rotation_bases
from scripts.drs_peft.spectral_rotation_engine import (
    Descriptors,
    Labels,
    core_model,
    fit_cal2,
    fit_core,
    fit_mapper,
    frozen_mapper,
    lora_operator,
    lora_snapshots,
    predict_arrays,
    predict_core,
    prepare_episode,
)
from scripts.drs_peft.spectral_rotation_protocol import (
    CANDIDATE,
    CONFIG,
    CORE_MODES,
    META_METHODS,
    PARAMETER_MATCHED,
    PRIVATE,
    PUBLIC,
    SAME_CORE,
    build_plan,
    role_ids,
)
from scripts.drs_peft.summarize_spectral_rotation import score_episode, write_summary
from scripts.drs_peft.train import bind, configure, read_json, save_torch, seed_all
from scripts.drs_peft.transfer_scope_data import load_prefixes
from scripts.drs_peft.transfer_scope_reference import BACKBONE, REFERENCES, save_npz


def input_features(args, source_ids, prefixes):
    feature_path = args.features / "FEATURES.npz"
    receipt_path = args.features / "FEATURES_RECEIPT.json"
    receipt = read_json(receipt_path)
    if (
        receipt["status"] != "COMPLETE_SIGNED_SOURCE_PREFIX_FEATURES"
        or receipt["features_sha256"] != file_hash(feature_path)
        or set(receipt["source_ids"]) != source_ids
        or receipt["prefix_cache_manifest_sha256"] != prefixes.cache_manifest_sha256
        or receipt["implementation_sha256"]
        != file_hash(Path(__file__).with_name("spectral_rotation_features.py"))
    ):
        raise RuntimeError("The source-prefix descriptor receipt is invalid")
    with np.load(feature_path, allow_pickle=False) as feature_file:
        features = dict(feature_file)
    if features["ids"].astype(str).tolist() != receipt["source_ids"]:
        raise RuntimeError("The registered descriptor row order changed")
    return features, receipt_path


def frozen_inputs(args, fold_cfg, features, data, cfg):
    fold = fold_cfg["fold"]
    original, reference = models(args, fold, cfg["seed"])
    started = time.perf_counter()
    bases = build_rotation_bases(original)
    elapsed = time.perf_counter() - started
    descriptors = Descriptors(features, fold_cfg["fit_ids"])
    basis_audit = {
        "seconds_per_fold": elapsed,
        "layers": {name: basis.audit() for name, basis in bases.items()},
        "reference_sha256": fold_cfg["reference_sha256"],
        "host_device": "cpu",
    }
    return original, reference, bases, descriptors, basis_audit


def initial_audit(original, reference, data, fold_cfg, device):
    ids = fold_cfg["anchors"][:2]
    pred, _ = predict_arrays(original.to(device), data, ids, device)
    original.cpu()
    merged, _ = predict_arrays(reference, data, ids, device)
    maximum = float(np.max(np.abs(pred - merged)))
    if maximum > 2e-6:
        raise RuntimeError("The merged frozen reference prediction changed")
    return maximum


def model_accounting(model):
    budget = model.budget()
    frozen_bases = sum(
        buffer.numel() * buffer.element_size()
        for layer in model.layers.values()
        for buffer in layer.buffers(recurse=False)
    )
    return {
        "target_parameters": budget["episode_parameters"],
        "source_mapper_parameters": budget["controller_parameters"],
        "source_mapper_effective_constant_parameters": budget[
            "constant_identifiable_parameters"
        ],
        "basis_storage_bytes": frozen_bases,
        "matrix_rank_cap": budget["maximum_matrix_rank"],
    }


def validation_row(method, fold, ep, setting, prediction, cost):
    return {
        "method": method,
        "fold": fold,
        "draw": ep.draw,
        **setting,
        **cost,
        **ep.scores(prediction),
    }


def select_trial(rows, method, fields, ep_count):
    selected = {}
    groups = {}
    for row in rows:
        if row["method"] == method:
            groups.setdefault(tuple(row[field] for field in fields), []).append(row)
    if not groups or any(len(group) != ep_count for group in groups.values()):
        raise RuntimeError(f"Incomplete source-validation grid for {method}")
    for settings, scores in groups.items():
        selected[settings] = math.fsum(row["query_mae"] for row in scores) / ep_count
    winner = min(selected, key=lambda key: (selected[key], key))
    return dict(zip(fields, winner, strict=True)) | {
        "validation_query_mae": selected[winner]
    }


def dev_fold(args, cfg, plan, data, features, sealed, identity, lora_parent):
    fold = plan["fold"]
    path = args.public / "development" / f"fold{fold}"
    storage = args.private / "development" / f"fold{fold}"
    complete = path / "COMPLETE.json"
    if complete.exists():
        receipt = read_json(complete)
        if receipt["run_identity"] != identity:
            raise RuntimeError("Existing development fold belongs to another design")
        verify_development(receipt, path, storage)
        return receipt

    started = time.perf_counter()
    original, reference, bases, phi, evidence = frozen_inputs(
        args, plan, features, data, cfg
    )
    fit_ids = set(plan["fit_ids"])
    validation_ids = role_ids(plan["validation"])
    fit_labels = Labels.read(args.data_root, fit_ids, fit_ids, sealed)
    val_labels = Labels.read(args.data_root, validation_ids, validation_ids, sealed)
    reference.to(args.device)
    evidence["original_vs_merged_zero_max"] = initial_audit(
        original, reference, data, plan, args.device
    )
    del original
    eps = [
        prepare_episode(
            data,
            val_labels,
            item,
            plan["anchors"],
            reference,
            phi,
            cfg,
            args.device,
            (cfg["seed"], fold, "validation", item["draw"]),
        )
        for item in plan["validation"]
    ]
    reference.cpu()
    calibration, lora, trials, meta, costs, states = {}, {}, [], [], {}, {}
    for ep in eps:
        trials.append(validation_row("K0", fold, ep, {}, ep.baseline, {}))
        for rate, eta in itertools.product(cfg["cal2_ridge"], cfg["cal2_eta"]):
            prediction, cost = fit_cal2(ep, data, rate, eta)
            trials.append(
                validation_row(
                    "CAL2", fold, ep, {"eta": eta, "rate": rate}, prediction, cost
                )
            )
    calibration = select_trial(trials, "CAL2", ("eta", "rate"), len(eps))
    operator = lora_operator(reference, lora_parent, cfg, args.device)
    for ep, rate in itertools.product(eps, cfg["lora_rates"]):
        for step, prediction, cost in lora_snapshots(
            operator, ep, data, args.device, rate, cfg["lora_steps"]
        ):
            trials.append(
                validation_row(
                    "LORA_SMALL",
                    fold,
                    ep,
                    {"steps": step, "rate": rate},
                    prediction,
                    cost,
                )
            )
    lora = select_trial(trials, "LORA_SMALL", ("steps", "rate"), len(eps))
    del operator
    selection = {"CAL2": calibration, "LORA_SMALL": lora}
    for method in CORE_MODES:
        model = core_model(reference, bases, method, phi, cfg, args.device)
        costs[method] = model_accounting(model)
        if method in META_METHODS:
            trace, cost = fit_mapper(
                model,
                method,
                plan["train"],
                data,
                fit_labels,
                phi,
                cfg,
                fold,
                args.device,
            )
            meta.extend(trace)
            costs[method].update(cost)
            destination = storage / f"{method}.pt"
            save_torch(
                destination,
                {k: p.detach().cpu() for k, p in model.controller.state_dict().items()},
            )
            states[method] = file_hash(destination)
        frozen_mapper(model)
        for ep, rate in itertools.product(eps, cfg["core_rates"]):
            _, prediction, cost = fit_core(
                model, ep, data, method, cfg, rate, args.device
            )
            trials.append(
                validation_row(method, fold, ep, {"rate": rate}, prediction, cost)
            )
        selection[method] = select_trial(trials, method, ("rate",), len(eps))
        print(f"dev fold={fold} {method} ready", flush=True)
        del model
    outputs = {
        "VALIDATION_TRIALS.csv": trials,
        "META_TRACE.csv": meta,
    }
    for name, rows in outputs.items():
        csv(path / name, rows)
    write_json(path / "BASIS_AUDIT.json", evidence)
    write_json(path / "DESCRIPTOR_SCALER.json", phi.audit)
    receipt = {
        "status": "COMPLETE_SOURCE_DEVELOPMENT_FOLD",
        "run_identity": identity,
        "fold": fold,
        "validation_draws": len(eps),
        "fit_label_ids": sorted(fit_ids),
        "validation_label_ids": sorted(validation_ids),
        "outer_role_label_rows_loaded": 0,
        "target_label_rows_loaded": 0,
        "selection": selection,
        "costs": costs,
        "state_sha256": states,
        "wall_seconds": time.perf_counter() - started,
        "files": {
            name: file_hash(path / name)
            for name in (*outputs, "BASIS_AUDIT.json", "DESCRIPTOR_SCALER.json")
        },
    }
    write_json(complete, receipt)
    return receipt


def verify_development(receipt, path, storage):
    if receipt["status"] != "COMPLETE_SOURCE_DEVELOPMENT_FOLD":
        raise RuntimeError("Missing completed source-development fold")
    for name, expected in receipt["files"].items():
        if file_hash(path / name) != expected:
            raise RuntimeError(f"Development artifact changed: {name}")
    for method, expected in receipt["state_sha256"].items():
        if file_hash(storage / f"{method}.pt") != expected:
            raise RuntimeError(f"Locked mapper changed: {method}")


def dev_lock(args, cfg, plan, identity, *, create):
    files = []
    selection = []
    for entry in plan["folds"]:
        fold = entry["fold"]
        path = args.public / "development" / f"fold{fold}"
        marker = path / "COMPLETE.json"
        receipt = read_json(marker)
        if receipt["run_identity"] != identity or receipt["fold"] != fold:
            raise RuntimeError("Dev receipt identity differs from the frozen plan")
        if (
            receipt["outer_role_label_rows_loaded"]
            or receipt["target_label_rows_loaded"]
        ):
            raise RuntimeError("Development run opened a sealed role")
        verify_development(receipt, path, args.private / "development" / f"fold{fold}")
        files.append({"fold": fold, "sha256": file_hash(marker)})
        selection.append(receipt)
    lock = {
        "status": "LOCKED_ALL_FOLDS_BEFORE_OUTER_ROLE_SCORING",
        "run_identity": identity,
        "source_development_receipts": files,
        "outer_role_future_label_reads_before_lock": 0,
        "target_future_label_reads": 0,
    }
    marker = args.public / "DEVELOPMENT_LOCK.json"
    if create:
        bind(marker, lock)
    elif read_json(marker) != lock:
        raise RuntimeError("All folds must be locked before source-outer scoring")
    return selection, file_hash(marker)


def replay_at_norm(model, ep, data, method, state, norm, radius, device):
    phi = ep.descriptors["shuffled" if method == "ROTATION_SHUFFLED" else "true"]
    scale = radius / norm
    scaled = model.scale_cores(state, scale)
    output, facts = predict_core(model, ep, data, phi, scaled, device)
    if not math.isclose(
        facts["weight_frobenius_rms"], radius, rel_tol=1e-5, abs_tol=1e-10
    ):
        raise RuntimeError(
            "The parameter-matched actual rollout missed its common radius"
        )
    return output


def core_protocol_scores(args, cfg, ep, data, operators, selection):
    outputs, state, facts = {}, {}, {}
    for method, model in operators.items():
        model.to(args.device)
        state[method], outputs[method], facts[method] = fit_core(
            model, ep, data, method, cfg, selection[method]["rate"], args.device
        )
        state[method] = {
            key: value.detach().cpu() for key, value in state[method].items()
        }
        if method == CANDIDATE:
            k = ep.query_start
            phi = torch.cat(
                (ep.descriptors["true"][:k], ep.descriptors["shuffled"][k:])
            )
            # The physical model and C stay identical. Only the Query/anchor descriptors change.
            device_state = {
                key: value.to(args.device) for key, value in state[method].items()
            }
            outputs[SAME_CORE], replay = predict_core(
                model, ep, data, phi, device_state, args.device
            )
            if not np.array_equal(
                outputs[method][:k], outputs[SAME_CORE][:k]
            ) or not math.isclose(
                facts[method]["weight_frobenius_rms"],
                replay["weight_frobenius_rms"],
                rel_tol=1e-6,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Same-core derangement changed the Support or parameter scale"
                )
            facts[SAME_CORE] = (
                facts[method]
                | replay
                | {
                    "reuses_conditioned_core": True,
                    "adapt_seconds": 0.0,
                    "adapt_updates": 0,
                }
            )
        model.cpu()
    pairs = {}
    for control in PARAMETER_MATCHED:
        first, other = [
            facts[name]["weight_frobenius_rms"] for name in (CANDIDATE, control)
        ]
        radius = min(first, other)
        if radius <= 1e-12:
            continue
        values = []
        for name, norm in zip((CANDIDATE, control), (first, other), strict=True):
            if norm == radius:
                values.append(outputs[name])
                continue
            model = operators[name].to(args.device)
            device_state = {
                key: value.to(args.device) for key, value in state[name].items()
            }
            values.append(
                replay_at_norm(
                    model,
                    ep,
                    data,
                    name,
                    device_state,
                    norm,
                    radius,
                    args.device,
                )
            )
            model.cpu()
        pairs[CANDIDATE, control] = (*values, radius)
    return outputs, facts, pairs


def locked_outer_fold(
    args, cfg, plan, data, features, sealed, dev, lock_hash, protocols, lora_parent
):
    fold = plan["fold"]
    private_path = args.private / "outer" / f"fold{fold}"
    path = args.public / "outer" / f"fold{fold}"
    marker = path / "COMPLETE.json"
    if marker.exists():
        result = read_json(marker)
        if result["development_lock_sha256"] != lock_hash or result["fold"] != fold:
            raise RuntimeError(
                "Existing outer scores belong to another development lock"
            )
        assert_outputs(result, path, private_path)
        return result
    original, reference, bases, phi, _ = frozen_inputs(args, plan, features, data, cfg)
    del original
    ids = role_ids(plan["outer"])
    labels = Labels.read(args.data_root, ids, ids, sealed)
    reference.to(args.device)
    eps = [
        prepare_episode(
            data,
            labels,
            item,
            plan["anchors"],
            reference,
            phi,
            cfg,
            args.device,
            (cfg["seed"], fold, "outer", item["draw"]),
        )
        for item in plan["outer"]
    ]
    reference.cpu()
    selection = dev["selection"]
    controls = {
        method: core_model(reference, bases, method, phi, cfg, "cpu")
        for method in CORE_MODES
    }
    for name, model in controls.items():
        if name in META_METHODS:
            state = args.private / "development" / f"fold{fold}" / f"{name}.pt"
            if file_hash(state) != dev["state_sha256"][name]:
                raise RuntimeError("The frozen source mapping changed after the lock")
            model.controller.load_state_dict(
                torch.load(state, weights_only=True, map_location="cpu")
            )
        frozen_mapper(model)
    started = time.perf_counter()
    all_cells, all_episodes, all_pairs, prediction_sha256 = [], [], [], {}
    for ep in eps:
        outputs, cost, norm_pairs = core_protocol_scores(
            args, cfg, ep, data, controls, selection
        )
        for name in CORE_MODES:
            cost[name].update(dev["costs"][name])
        cost[SAME_CORE].update(dev["costs"][CANDIDATE])
        cost[SAME_CORE]["source_fit_shared_with"] = CANDIDATE
        outputs["K0"] = ep.baseline
        cost["K0"] = {
            "target_parameters": 0,
            "source_mapper_parameters": 0,
            "basis_storage_bytes": 0,
            "adapt_seconds": 0.0,
            "adapt_updates": 0,
            "inference_seconds": ep.base_predict_seconds,
        }
        setting = selection["CAL2"]
        outputs["CAL2"], cost["CAL2"] = fit_cal2(
            ep, data, setting["rate"], setting["eta"]
        )
        setting = selection["LORA_SMALL"]
        operator = lora_operator(reference, lora_parent, cfg, args.device)
        for _, output, cost_lora in lora_snapshots(
            operator, ep, data, args.device, setting["rate"], [setting["steps"]]
        ):
            outputs["LORA_SMALL"], cost["LORA_SMALL"] = output, cost_lora
        del operator
        if set(outputs) != set(cfg["methods"]):
            raise RuntimeError("Incomplete outer method comparison")
        support_base = ep.scores(ep.baseline)
        for name, prediction in outputs.items():
            stats = ep.scores(prediction)
            cost[name].update(
                {
                    "support_mse_before": support_base["support_sampled_mse"],
                    "support_mse_after": stats["support_sampled_mse"],
                    "support_full_mse_before": support_base["support_full_mse"],
                    "support_full_mse_after": stats["support_full_mse"],
                }
            )
        k, a = ep.query_start, ep.anchor_start
        cells, moments, pairs = score_episode(
            {"fold": fold, "seed": cfg["seed"], "K": cfg["shot"], "draw": ep.draw},
            query_ids=ep.query,
            grid=data.query,
            observed=ep.query_observed,
            targets=ep.query_y,
            baseline=ep.baseline[k:a],
            predictions={name: output[k:a] for name, output in outputs.items()},
            anchor_baseline=ep.baseline[a:],
            anchor_predictions={name: output[a:] for name, output in outputs.items()},
            query_protocols={uid: protocols[uid] for uid in ep.query},
            costs=cost,
            matched_pairs=[
                (CANDIDATE, name)
                for name in cfg["methods"]
                if name not in {"K0", CANDIDATE}
            ],
            parameter_matched={
                pair: (first[k:a], second[k:a], norm)
                for pair, (first, second, norm) in norm_pairs.items()
            },
        )
        all_cells.extend(cells)
        all_episodes.extend(moments)
        all_pairs.extend(pairs)
        destination = private_path / f"DRAW{ep.draw}_PREDICTIONS.npz"
        save_npz(
            destination,
            ids=np.array(ep.ids),
            cycles=data.query,
            methods=np.array(cfg["methods"]),
            predictions=np.stack([outputs[name] for name in cfg["methods"]]),
            parameter_matched_methods=np.array([name for _, name in norm_pairs]),
            parameter_matched_predictions=np.array(
                [pair[:2] for pair in norm_pairs.values()]
            ),
            parameter_matched_radii=np.array([pair[2] for pair in norm_pairs.values()]),
        )
        prediction_sha256[destination.name] = file_hash(destination)
        print(f"outer fold={fold} draw={ep.draw} complete", flush=True)
    output = {
        "CELL_SCORES.csv": all_cells,
        "EPISODE_SCORES.csv": all_episodes,
        "PAIR_SCORES.csv": all_pairs,
    }
    for name, rows in output.items():
        csv(path / name, rows)
    result = {
        "status": "COMPLETE_LOCKED_SOURCE_OUTER_FOLD",
        "fold": fold,
        "development_lock_sha256": lock_hash,
        "outer_role_label_ids": sorted(ids),
        "target_label_rows_loaded": 0,
        "outer_draws": len(eps),
        "score_seconds": time.perf_counter() - started,
        "files": {name: file_hash(path / name) for name in output},
        "private_predictions": prediction_sha256,
    }
    write_json(marker, result)
    return result


def assert_outputs(result, public, private):
    if (
        result["status"] != "COMPLETE_LOCKED_SOURCE_OUTER_FOLD"
        or result["target_label_rows_loaded"]
    ):
        raise RuntimeError("Incomplete or contaminated outer fold")
    for root, entries in (
        (public, result["files"]),
        (private, result["private_predictions"]),
    ):
        for name, expected in entries.items():
            if file_hash(root / name) != expected:
                raise RuntimeError(f"Outer artifact changed: {name}")


def prepare_features(args, source_ids, prefixes):
    from scripts.drs_peft.spectral_rotation_features import load_signed_deformation

    feature_path = args.features / "FEATURES.npz"
    receipt = args.features / "FEATURES_RECEIPT.json"
    if receipt.exists():
        input_features(args, source_ids, prefixes)
        return
    start = time.perf_counter()
    extracted = load_signed_deformation("MATR", args.data_root, sorted(source_ids))
    save_npz(
        feature_path,
        ids=np.array(extracted.ids),
        values=extracted.values,
        mask=extracted.mask,
        columns=np.array(extracted.columns),
    )
    write_json(
        receipt,
        {
            "status": "COMPLETE_SIGNED_SOURCE_PREFIX_FEATURES",
            "source_ids": list(extracted.ids),
            "prefix_cache_manifest_sha256": prefixes.cache_manifest_sha256,
            "features_sha256": file_hash(feature_path),
            "implementation_sha256": file_hash(
                Path(__file__).with_name("spectral_rotation_features.py")
            ),
            "target_label_rows_loaded": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "audit": extracted.audit,
        },
    )


def combine_outer(args, cfg, plan, results, identity, lock_hash):
    tables = []
    for table in ("CELL_SCORES.csv", "EPISODE_SCORES.csv", "PAIR_SCORES.csv"):
        tables.append(
            pd.concat(
                [
                    pd.read_csv(
                        args.public / "outer" / f"fold{fold}" / table,
                        float_precision="round_trip",
                    )
                    for fold in cfg["folds"]
                ],
                ignore_index=True,
            )
        )
    required = [name for name in cfg["methods"] if name != CANDIDATE]
    decision = write_summary(
        args.public / "summary",
        *tables,
        candidate=CANDIDATE,
        required_references=required,
        expected_folds=tuple(cfg["folds"]),
        expected_draws=tuple(range(cfg["heldout_draws"])),
        margin=cfg["mae_margin"],
        required_winning_folds=cfg["winning_folds"],
        near_zero_anchor_radius=cfg["near_zero_anchor_radius"],
    )
    write_json(
        args.public / "COMPLETE.json",
        {
            "status": "COMPLETE_SOURCE_PROTOCOL_SCREEN",
            "decision": decision["direction"],
            "run_identity": identity,
            "development_lock_sha256": lock_hash,
            "scored_folds": cfg["folds"],
            "methods": cfg["methods"],
            "physical_query_cells": plan["outer_source_query_cells"],
            "physical_query_protocols": plan["outer_source_query_protocols"],
            "scored_episodes": sum(row["outer_draws"] for row in results),
            "target_label_rows_loaded": 0,
            "new_target_fits": 0,
            "summary_decision_sha256": file_hash(
                args.public / "summary" / "SOURCE_VALIDATION_DECISION.json"
            ),
            "outer_receipts_sha256": {
                str(fold): file_hash(
                    args.public / "outer" / f"fold{fold}" / "COMPLETE.json"
                )
                for fold in cfg["folds"]
            },
        },
    )
    print(f"Complete source protocol screen: {decision['direction']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("register", "features", "dev", "outer", "run"),
        default="run",
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--features", type=Path, default=PRIVATE)
    parser.add_argument("--reference-root", type=Path, default=REFERENCES)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure()
    with threadpool_limits(limits=1):
        cfg, plan, protocols, source_ids, sealed = build_plan(args)
        seed_all(cfg["seed"])
        plan_path = args.public / "REGISTERED_EPISODES.json"
        bind(plan_path, plan)
        if args.phase == "register":
            print(
                f"Registered {plan['outer_source_query_cells']} source Query cells",
                flush=True,
            )
            return
        data = load_prefixes("MATR", args.data_root)
        if args.phase == "features":
            prepare_features(args, source_ids, data)
            return
        features, feature_receipt = input_features(args, source_ids, data)
        lora_path = ROOT / cfg["lora_parent_config"]
        lora_parent = read_json(lora_path)
        relevant_code = sorted(Path(__file__).parent.glob("spectral_rotation*.py")) + [
            Path(__file__),
            Path(__file__).with_name("summarize_spectral_rotation.py"),
        ]
        identity = {
            "plan_sha256": file_hash(plan_path),
            "lora_config_sha256": file_hash(lora_path),
            "feature_receipt_sha256": file_hash(feature_receipt),
            "prefix_cache_manifest_sha256": data.cache_manifest_sha256,
            "code_sha256": {item.name: file_hash(item) for item in relevant_code},
            "hardware": torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda")
            else "CPU",
            "torch_version": str(torch.__version__),
        }
        bind(args.public / "RUN_IDENTITY.json", identity)
        if args.phase in {"run", "dev"}:
            for fold in plan["folds"]:
                dev_fold(args, cfg, fold, data, features, sealed, identity, lora_parent)
            dev_lock(args, cfg, plan, identity, create=True)
            print("All source development folds locked", flush=True)
        if args.phase in {"run", "outer"}:
            development, lock_hash = dev_lock(args, cfg, plan, identity, create=False)
            results = [
                locked_outer_fold(
                    args,
                    cfg,
                    fold,
                    data,
                    features,
                    sealed,
                    receipt,
                    lock_hash,
                    protocols,
                    lora_parent,
                )
                for fold, receipt in zip(plan["folds"], development, strict=True)
            ]
            combine_outer(args, cfg, plan, results, identity, lock_hash)


if __name__ == "__main__":
    main()
