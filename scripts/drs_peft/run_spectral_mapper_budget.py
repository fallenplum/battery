"""Replay locked MA mappers and extend the old seed-42 schedule at fixed rates."""

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

from scripts.drs_peft.abba_engine import synchronized
from scripts.drs_peft.data import CACHE, ROOT, file_hash, write_json
from scripts.drs_peft.run_abba_moirai import csv
from scripts.drs_peft.run_spectral_rotation import input_features
from scripts.drs_peft.spectral_objective import (
    _masked_residual,
    _meta_roles,
    unroll_support,
)
from scripts.drs_peft.spectral_objective_engine import (
    MapperTrial,
    authenticate_trial,
    fold_inputs,
    mapper_trial_path,
    prepare_episodes,
    trial_identity,
)
from scripts.drs_peft.spectral_objective_protocol import (
    CONFIG,
    build_plan,
    donor_episode,
    role_ids,
)
from scripts.drs_peft.spectral_objective_protocol import (
    PRIVATE as OLD_PRIVATE,
)
from scripts.drs_peft.spectral_objective_protocol import (
    PUBLIC as OLD_PUBLIC,
)
from scripts.drs_peft.spectral_rotation_engine import (
    Labels,
    core_model,
    fit_cal2,
    fit_core,
    lora_operator,
    lora_snapshots,
    sampled,
)
from scripts.drs_peft.train import bind, configure, read_json, save_torch, seed_all
from scripts.drs_peft.transfer_scope_data import load_prefixes
from scripts.drs_peft.transfer_scope_reference import BACKBONE, REFERENCES, save_npz

STUDY = "spectral_mapper_budget_v1"
METHODS = ("ROTATION_TRUE", "ROTATION_CONST", "ROTATION_SHUFFLED", "GAIN")
CONTROLS = ("K0", "STATIC_CORE", "CAL2", "LORA_SMALL")
MILESTONES = (0, 32, 64, 128, 256)


def register(args):
    cfg, plan, _, source, sealed = build_plan(args)
    old_plan = args.old_public / "REGISTERED_EPISODES.json"
    run_id = args.old_public / "RUN_IDENTITY.json"
    hyper = args.old_public / "HYPERPARAMETER_LOCK.json"
    develop = args.old_public / "DEVELOPMENT_LOCK.json"
    identity, selection, lock = map(read_json, (run_id, hyper, develop))
    old_sha = file_hash(run_id)
    if (
        read_json(old_plan) != plan
        or identity["plan_sha256"] != file_hash(old_plan)
        or identity["config_sha256"] != file_hash(args.config)
        or selection["identity_sha256"] != old_sha
        or lock["identity_sha256"] != old_sha
        or lock["hyperparameters_sha256"] != file_hash(hyper)
        or lock["status"] != "LOCKED_BEFORE_SOURCE_OUTER_LABELS"
    ):
        raise RuntimeError("The old source mapper identity changed")
    for name, expected in identity["code_sha256"].items():
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise RuntimeError(f"Old mapper dependency changed: {name}")
    if (
        args.device != identity["device"]
        or torch.cuda.get_device_name(args.device) != identity["hardware"]
        or str(torch.__version__) != identity["torch_version"]
    ):
        raise RuntimeError(
            "From-zero bitwise replay requires the same training runtime"
        )
    probes = {}
    for fold in plan["folds"]:
        first = next(
            row
            for row in fold["train_by_seed"]["42"]
            if not set(row["support"] + row["query"]) & set(fold["anchors"])
        )
        probe = donor_episode(first, 42, fold["fold"], "meta", fold["anchors"])
        if any(
            probe["donors"][role] != first["donors"][role]
            for role in ("support", "query")
        ):
            raise RuntimeError("Fit probe changed the original descriptor permutation")
        probes[str(fold["fold"])] = probe
    contract = {
        "status": "REGISTERED_FIT_AND_INNER_VALIDATION_ONLY",
        "study_id": STUDY,
        "implementation_sha256": file_hash(Path(__file__)),
        "old_run_identity_sha256": old_sha,
        "old_plan_sha256": file_hash(old_plan),
        "old_hyperparameters_sha256": file_hash(hyper),
        "old_development_lock_sha256": file_hash(develop),
        "meta_training_seed": 42,
        "outer_objective": "exact_masked_cell_macro_MAE",
        "support_sgd_steps": 3,
        "old_schedule_repetitions": 8,
        "schedule": "Repeat the original ordered 32 episodes eight times with one uninterrupted Adam",
        "milestones": list(MILESTONES),
        "methods": list(METHODS),
        "fit_probe_episodes": probes,
        "output_fit_and_inner_validation_only": True,
    }
    bind(args.public / "REGISTERED_PLAN.json", contract)
    return cfg, plan, source, sealed, identity, selection["selected"], lock, contract


def old_trial(args, fold, method, alpha, lock, contract):
    trial = MapperTrial(method, "MA", alpha, 42)
    folder = mapper_trial_path(args.old_private, fold["fold"], trial)
    expected = trial_identity(trial, fold, contract["old_run_identity_sha256"], "A")
    receipt = authenticate_trial(folder, expected)
    entry = lock["mapper_artifacts"][str(fold["fold"])][f"42:{trial.name}"]
    if entry != {
        "folder": folder.relative_to(args.old_private).as_posix(),
        "alpha": alpha,
        "receipt_sha256": file_hash(folder / "COMPLETE.json"),
        "mapper_sha256": receipt["files"]["MAPPER.pt"],
    }:
        raise RuntimeError("The selected old MA controller is not the locked one")
    saved = torch.load(folder / "MAPPER.pt", map_location="cpu", weights_only=True)
    if saved["identity"] != expected:
        raise RuntimeError("The saved old MA controller identity changed")
    with np.load(folder / "PREDICTIONS.npz", allow_pickle=False) as values:
        predictions = dict(values)
    return saved["controller"], predictions, receipt["files"]


def step_batches(plans, data, labels, phi, cfg, device):
    result = []
    for item in plans:
        support, query, donors = _meta_roles(item)
        sb = sampled(data, labels, support, cfg["support_max_observed"], device)
        qb = sampled(data, labels, query, cfg["meta_query_max_observed"], device)
        result.append(
            (
                sb,
                qb,
                phi.tensor(support, device),
                phi.tensor(query, device),
                phi.tensor(donors["support"], device),
                phi.tensor(donors["query"], device),
            )
        )
    return result


def meta_update(model, parameters, optimizer, batch, shuffled, alpha):
    sb, qb, phi_s, phi_q, donor_s, donor_q = batch
    optimizer.zero_grad(set_to_none=True)
    cores, inner_trace = unroll_support(
        model,
        *sb.batch(),
        donor_s if shuffled else phi_s,
        steps=3,
        lr=alpha,
        clip_norm=1.0,
        differentiable=True,
    )
    prediction = model(
        qb.early, qb.cycles, descriptor=donor_q if shuffled else phi_q, cores=cores
    )
    residual, counts = _masked_residual(prediction, qb.targets, qb.observed)
    query_mae = (residual.abs().sum(dim=1) / counts).mean()
    query_mse = (residual.square().sum(dim=1) / counts).mean()
    query_mae.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    metrics = (
        torch.cat(
            (
                torch.stack((query_mae.detach(), query_mse.detach(), norm.detach())),
                torch.stack(inner_trace).flatten(),
            )
        )
        .cpu()
        .tolist()
    )
    if not all(math.isfinite(value) for value in metrics):
        raise FloatingPointError("Nonfinite continuation objective or gradient")
    return {
        "train_query_sampled_mae": metrics[0],
        "train_query_sampled_mse": metrics[1],
        "mapper_grad_norm": metrics[2],
        **{
            f"support_mse_before_step_{step + 1}": metrics[3 + 4 * step]
            for step in range(3)
        },
        **{
            f"support_grad_norm_before_step_{step + 1}": metrics[4 + 4 * step]
            for step in range(3)
        },
    }


def cell_metrics(ep, prediction, metadata):
    pred = prediction[ep.query_start : ep.anchor_start].astype(np.float64)
    counts = ep.query_observed.sum(axis=1)
    residual = np.where(ep.query_observed, pred - ep.query_y, 0)
    return [
        dict(
            metadata,
            cell_uid=uid,
            n_obs=int(n),
            mae=float(np.abs(row).sum() / n),
            mse=float(np.square(row).sum() / n),
        )
        for uid, n, row in zip(ep.query, counts, residual, strict=True)
    ]


def controller_saturation(model, ep, shuffled):
    features = ep.descriptors["shuffled" if shuffled else "true"]
    scope = features[ep.query_start : ep.anchor_start]
    with torch.no_grad():
        modulation = model.modulation(len(scope), scope)
        relative = (
            (modulation - 1).abs() / 0.5
            if model.mode == "gain"
            else modulation.abs() / model.angle_max
        )
        if not bool(torch.isfinite(relative).all()):
            raise FloatingPointError("Invalid mapper saturation probe")
        return {
            "modulation_fraction_ge_0_95": float(
                relative.ge(0.95).to(torch.float64).mean()
            ),
            "modulation_mean_fraction_of_bound": float(relative.mean()),
            "modulation_max_fraction_of_bound": float(relative.max()),
        }


def snapshot(
    args,
    cfg,
    data,
    eps,
    model,
    fold,
    method,
    alpha,
    step,
    training_time,
    directory,
    saved32,
):
    model.eval()
    model.controller.requires_grad_(False)
    arm = f"{method}_MA"
    weights = {
        key: value.detach().cpu().clone()
        for key, value in model.controller.state_dict().items()
    }
    zero_delta = math.sqrt(
        sum(float(x.double().square().sum()) for x in weights.values())
    )
    after32 = (
        math.sqrt(
            sum(
                float((x.double() - saved32[key]).square().sum())
                for key, x in weights.items()
            )
        )
        if step >= 32
        else None
    )
    diagnostic, cells, predictions = [], [], {key: [] for key in eps}
    for scope, episodes in eps.items():
        for ep in episodes:
            _, prediction, cost = fit_core(
                model, ep, data, method, cfg, alpha, args.device
            )
            metadata = {
                "fold": fold,
                "training_seed": 42,
                "method": arm,
                "alpha": alpha,
                "meta_steps": step,
                "scope": scope,
                "draw": ep.draw,
            }
            diagnostic.append(
                metadata
                | ep.scores(prediction)
                | cost
                | controller_saturation(model, ep, method == "ROTATION_SHUFFLED")
                | {
                    "mapper_frobenius_from_zero": zero_delta,
                    "mapper_frobenius_from_step32": after32,
                    "cumulative_meta_training_seconds": training_time,
                }
            )
            cells.extend(cell_metrics(ep, prediction, metadata))
            predictions[scope].append(prediction)
    model.controller.requires_grad_(True)
    model.train()
    save_torch(
        directory / f"STEP{step:03d}_MAPPER.pt",
        {"meta_steps": step, "controller": weights},
    )
    save_npz(
        directory / f"STEP{step:03d}_PREDICTIONS.npz",
        **{f"{key}_predictions": np.stack(value) for key, value in predictions.items()},
        **{
            f"{key}_ids": np.array([ep.ids for ep in value])
            for key, value in eps.items()
        },
    )
    return weights, diagnostic, cells, predictions


def certify_step32(weights, predictions, old_state, old_predictions, validation):
    if set(weights) != set(old_state):
        raise RuntimeError("Replayed mapper state fields changed")
    weight_error = max(
        float((weights[name] - previous).abs().max())
        for name, previous in old_state.items()
    )
    replay = np.stack(predictions["validation"])
    archived = old_predictions["predictions"]
    if (
        replay.shape != archived.shape
        or not np.array_equal(
            old_predictions["ids"], np.asarray([ep.ids for ep in validation])
        )
        or not np.array_equal(
            old_predictions["draws"], np.asarray([ep.draw for ep in validation])
        )
    ):
        raise RuntimeError("The original validation episode or axis changed")
    prediction_error = float(np.max(np.abs(replay.astype(np.float64) - archived)))
    if not all(
        torch.equal(weights[name], expected) for name, expected in old_state.items()
    ) or not np.array_equal(replay, archived):
        raise RuntimeError(
            f"Step-32 replay differs from the archived experiment: mapper max={weight_error}, prediction max={prediction_error}"
        )
    return {
        "status": "EXACT_STEP32_REPLAY",
        "mapper_max_abs_error": weight_error,
        "validation_prediction_max_abs_error": prediction_error,
    }


def verify_files(receipt, public, private, expected):
    if receipt["identity"] != expected:
        raise RuntimeError("Existing budget experiment belongs to a different contract")
    for root, files in (
        (public, receipt["public_files"]),
        (private, receipt["private_files"]),
    ):
        for name, digest in files.items():
            if file_hash(root / name) != digest:
                raise RuntimeError(f"Budget experiment artifact changed: {name}")
    return receipt


def run_mapper(args, cfg, fold, setup, method, alpha, lock, contract):
    reference, bases, phi, data, eps, batches = setup
    index = fold["fold"]
    public = args.public / f"fold{index}" / method
    private = args.private / f"fold{index}" / method
    marker = public / "COMPLETE.json"
    expected = {
        "fold": index,
        "method": f"{method}_MA",
        "alpha": alpha,
        "contract_sha256": file_hash(args.public / "REGISTERED_PLAN.json"),
    }
    if marker.exists():
        return verify_files(read_json(marker), public, private, expected)
    old_state, old_predictions, old_files = old_trial(
        args, fold, method, alpha, lock, contract
    )
    model = core_model(reference, bases, method, phi, cfg, args.device)
    model.train()
    parameters = tuple(model.controller.parameters())
    if any(bool(torch.count_nonzero(p)) for p in parameters):
        raise RuntimeError("Mapper continuation does not start from zero")
    optimizer = torch.optim.Adam(parameters, lr=cfg["meta_lr"])
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    train_seconds, timeline, metrics, cells = 0.0, [], [], []
    max_allocated = max_reserved = 0
    for step in range(MILESTONES[-1] + 1):
        if step:
            slot = (step - 1) % len(batches)
            begin = time.perf_counter()
            update = meta_update(
                model,
                parameters,
                optimizer,
                batches[slot],
                method == "ROTATION_SHUFFLED",
                alpha,
            )
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            timeline.append(
                expected
                | {
                    "meta_step": step,
                    "cycle": (step - 1) // len(batches) + 1,
                    "original_meta_draw": slot,
                    "update_seconds": elapsed,
                }
                | update
            )
        if step not in MILESTONES:
            continue
        if step and any(
            int(state["step"]) != step for state in optimizer.state.values()
        ):
            raise RuntimeError("One uninterrupted Adam must own all mapper iterations")
        max_allocated = max(max_allocated, torch.cuda.max_memory_allocated(args.device))
        max_reserved = max(max_reserved, torch.cuda.max_memory_reserved(args.device))
        weights, scores, rows, preds = snapshot(
            args,
            cfg,
            data,
            eps,
            model,
            index,
            method,
            alpha,
            step,
            train_seconds,
            private,
            old_state,
        )
        if step == 32:
            replay = certify_step32(
                weights, preds, old_state, old_predictions, eps["validation"]
            )
            replay.update(
                old_archive_sha256={
                    name: old_files[name] for name in ("MAPPER.pt", "PREDICTIONS.npz")
                }
            )
            write_json(public / "STEP32_REPLAY.json", replay)
        metrics.extend(scores)
        cells.extend(rows)
        print(
            f"budget fold={index} {method} alpha={alpha:g} meta_steps={step}",
            flush=True,
        )
    synchronized(args.device)
    outputs = {
        "EPISODE_METRICS.csv": metrics,
        "CELL_METRICS.csv": cells,
        "META_TRACE.csv": timeline,
    }
    for name, values in outputs.items():
        csv(public / name, values)
    receipt = {
        "status": "COMPLETE_MAPPER_BUDGET",
        "identity": expected,
        "cost": {
            "mapper_adam_updates": len(timeline),
            "differentiable_support_updates": 3 * len(timeline),
            "training_seconds": train_seconds,
            "wall_seconds": time.perf_counter() - started,
            "meta_training_peak_allocated_bytes": max_allocated,
            "meta_training_peak_reserved_bytes": max_reserved,
        },
        "public_files": {
            name: file_hash(public / name) for name in (*outputs, "STEP32_REPLAY.json")
        },
        "private_files": {
            item.name: file_hash(item) for item in sorted(private.glob("STEP*"))
        },
    }
    write_json(marker, receipt)
    return receipt


def run_control(args, cfg, fold, setup, selection, control):
    reference, bases, phi, data, eps, _ = setup
    index = fold["fold"]
    public = args.public / f"fold{index}" / control
    private = args.private / f"fold{index}" / control
    marker = public / "COMPLETE.json"
    expected = {
        "fold": index,
        "method": control,
        "contract_sha256": file_hash(args.public / "REGISTERED_PLAN.json"),
    }
    if marker.exists():
        return verify_files(read_json(marker), public, private, expected)
    option = selection[control]
    model = (
        core_model(reference, bases, control, phi, cfg, args.device)
        if control == "STATIC_CORE"
        else None
    )
    operator = (
        lora_operator(
            reference,
            read_json(ROOT / cfg["lora_parent_config"]),
            cfg | {"seed": 42},
            args.device,
        )
        if control == "LORA_SMALL"
        else None
    )
    rows, predictions, episodes = [], {key: [] for key in eps}, []
    for scope, values in eps.items():
        for ep in values:
            if control == "K0":
                pred, cost = ep.baseline, {"adapt_updates": 0}
            elif control == "STATIC_CORE":
                _, pred, cost = fit_core(
                    model, ep, data, control, cfg, option["alpha"], args.device
                )
            elif control == "CAL2":
                pred, cost = fit_cal2(ep, data, option["ridge"], option["eta"])
            else:
                _, pred, cost = next(
                    lora_snapshots(
                        operator, ep, data, args.device, option["lr"], [option["steps"]]
                    )
                )
            core = {
                "fold": index,
                "training_seed": 42,
                "method": control,
                "meta_steps": 0,
                "scope": scope,
                "draw": ep.draw,
            }
            episodes.append(core | ep.scores(pred) | cost)
            rows.extend(cell_metrics(ep, pred, core))
            predictions[scope].append(pred)
    files = {"CELL_METRICS.csv": rows, "EPISODE_METRICS.csv": episodes}
    for name, values in files.items():
        csv(public / name, values)
    archive = private / "PREDICTIONS.npz"
    save_npz(archive, **{key: np.stack(value) for key, value in predictions.items()})
    record = {
        "status": "COMPLETE_FIXED_CONTROL",
        "identity": expected,
        "public_files": {name: file_hash(public / name) for name in files},
        "private_files": {archive.name: file_hash(archive)},
    }
    write_json(marker, record)
    return record


def pair_tables(cells):
    grouping = ["scope", "fold", "draw"]
    metrics = []
    for key, block in cells.groupby(grouping, sort=True):
        lookup = {
            (name, int(step)): values.set_index("cell_uid").sort_index()
            for (name, step), values in block.groupby(["method", "meta_steps"])
        }
        for method, step in itertools.product(
            (f"{name}_MA" for name in METHODS), MILESTONES
        ):
            candidate = lookup[(method, step)]
            comparators = [(name, 0) for name in CONTROLS] + [
                (f"{name}_MA", step) for name in METHODS if f"{name}_MA" != method
            ]
            for reference, ref_step in comparators:
                comparator = lookup[(reference, ref_step)]
                if not candidate.index.equals(
                    comparator.index
                ) or not candidate.n_obs.equals(comparator.n_obs):
                    raise RuntimeError(
                        "Source comparison changed the observed cells or denominators"
                    )
                diff = candidate.mae - comparator.mae
                metrics.append(
                    dict(zip(grouping, key, strict=True))
                    | {
                        "method": method,
                        "meta_steps": step,
                        "reference": reference,
                        "reference_meta_steps": ref_step,
                        "n_cells": len(diff),
                        "mae_delta": float(diff.mean()),
                        "mse_delta": float((candidate.mse - comparator.mse).mean()),
                        "cell_wins": int((diff < 0).sum()),
                        "cell_ties": int((diff == 0).sum()),
                        "cell_losses": int((diff > 0).sum()),
                    }
                )
    return metrics


def finalize(args, records, contract):
    roots = [
        args.public
        / f"fold{row['identity']['fold']}"
        / row["identity"]["method"].removesuffix("_MA")
        for row in records
    ]
    cells = pd.concat(
        [
            pd.read_csv(root / "CELL_METRICS.csv", float_precision="round_trip")
            for root in roots
        ],
        ignore_index=True,
    )
    episodes = pd.concat(
        [
            pd.read_csv(root / "EPISODE_METRICS.csv", float_precision="round_trip")
            for root in roots
        ],
        ignore_index=True,
    )
    costs = [
        row["identity"] | row["cost"]
        for row in records
        if row["status"] == "COMPLETE_MAPPER_BUDGET"
    ]
    if len(costs) != 12 or sum(row["mapper_adam_updates"] for row in costs) != 3072:
        raise RuntimeError("The twelve registered continuations are incomplete")
    outputs = {
        "CELL_METRICS.csv": cells,
        "EPISODE_METRICS.csv": episodes,
        "PAIRED_WINS.csv": pair_tables(cells),
        "COSTS.csv": costs,
    }
    for name, values in outputs.items():
        csv(args.public / name, values)
    write_json(
        args.public / "COMPLETE.json",
        {
            "status": "COMPLETE_SOURCE_TRAINING_BUDGET_DIAGNOSTIC",
            "contract": contract,
            "mapper_trials": len(costs),
            "total_mapper_adam_updates": 3072,
            "total_differentiable_support_updates": 9216,
            "replayed_step32_trials": 12,
            "source_outer_future_labels_read": 0,
            "target_future_labels_read": 0,
            "artifact_sha256": {
                name: file_hash(args.public / name) for name in outputs
            },
            "trial_receipts_sha256": {
                (root / "COMPLETE.json").relative_to(args.public).as_posix(): file_hash(
                    root / "COMPLETE.json"
                )
                for root in roots
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--old-public", type=Path, default=OLD_PUBLIC)
    parser.add_argument("--old-private", type=Path, default=OLD_PRIVATE)
    parser.add_argument(
        "--public", type=Path, default=ROOT / "results/drs_peft" / STUDY
    )
    parser.add_argument(
        "--private", type=Path, default=ROOT / "runtime/drs_peft" / STUDY
    )
    parser.add_argument(
        "--features", type=Path, default=ROOT / "runtime/drs_peft/spectral_rotation_v1"
    )
    parser.add_argument("--reference-root", type=Path, default=REFERENCES)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure()
    with threadpool_limits(limits=1):
        cfg, plan, source, sealed, identity, selection, lock, contract = register(args)
        seed_all(42)
        data = load_prefixes("MATR", args.data_root)
        features, receipt = input_features(args, source, data)
        if (
            file_hash(receipt) != identity["feature_receipt_sha256"]
            or data.cache_manifest_sha256 != identity["prefix_cache_manifest_sha256"]
        ):
            raise RuntimeError("The old signed feature or prefix identity changed")
        records = []
        for fold in plan["folds"]:
            index = str(fold["fold"])
            reference, bases, phi = fold_inputs(args, cfg, fold, data, features)
            heldout = sealed | role_ids(fold["outer"])
            source_fit = set(fold["fit_ids"])
            val = role_ids(fold["validation"])
            fit_labels = Labels.read(args.data_root, source_fit, source_fit, heldout)
            val_labels = Labels.read(args.data_root, val, val, heldout)
            reference.to(args.device)
            eps = {
                "fit_probe": prepare_episodes(
                    cfg,
                    data,
                    fit_labels,
                    [contract["fit_probe_episodes"][index]],
                    fold["anchors"],
                    reference,
                    phi,
                    args.device,
                ),
                "validation": prepare_episodes(
                    cfg,
                    data,
                    val_labels,
                    fold["validation"],
                    fold["anchors"],
                    reference,
                    phi,
                    args.device,
                ),
            }
            reference.cpu()
            batches = step_batches(
                fold["train_by_seed"]["42"], data, fit_labels, phi, cfg, args.device
            )
            setup = reference, bases, phi, data, eps, batches
            for method in METHODS:
                alpha = selection[index][f"{method}_MA"]["alpha"]
                records.append(
                    run_mapper(args, cfg, fold, setup, method, alpha, lock, contract)
                )
            for control in CONTROLS:
                records.append(
                    run_control(args, cfg, fold, setup, selection[index], control)
                )
            del setup, reference, eps, batches
        finalize(args, records, contract)
    print("Complete source fit/validation budget diagnostic", flush=True)


if __name__ == "__main__":
    main()
