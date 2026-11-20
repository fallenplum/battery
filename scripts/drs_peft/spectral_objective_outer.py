"""Train locked repetitions and open source-outer rows only after authentication."""

from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.run_abba_moirai import csv
from scripts.drs_peft.run_spectral_rotation import model_accounting, replay_at_norm
from scripts.drs_peft.spectral_objective_engine import (
    MapperTrial,
    authenticate_trial,
    fit_mapper_trial,
    fold_inputs,
    load_mapper,
    mapper_trial_path,
    prepare_episodes,
    trial_identity,
)
from scripts.drs_peft.spectral_objective_protocol import (
    CANDIDATE,
    MATCHED_RATE,
    META_METHODS,
    META_OBJECTIVES,
    OUTER_METHODS,
    PARAMETER_REFERENCES,
    SAME_CORE,
)
from scripts.drs_peft.spectral_rotation_engine import (
    Labels,
    core_model,
    fit_cal2,
    fit_core,
    lora_operator,
    lora_snapshots,
    predict_core,
)
from scripts.drs_peft.spectral_rotation_protocol import role_ids
from scripts.drs_peft.summarize_spectral_objective import (
    RAW_REFERENCES,
    score_episode,
    write_summary,
)
from scripts.drs_peft.train import bind, read_json
from scripts.drs_peft.transfer_scope_reference import save_npz

DETERMINISTIC = ("K0", "CAL2", "STATIC_CORE")


def selected_mapper_trials(selected, seed):
    primary = {
        f"{method}_{tag}": MapperTrial(
            method, tag, selected[f"{method}_{tag}"]["alpha"], seed
        )
        for method, tag in itertools.product(META_METHODS, META_OBJECTIVES)
    }
    return primary | {
        MATCHED_RATE: MapperTrial(
            "ROTATION_TRUE", "MM", selected[CANDIDATE]["alpha"], seed
        )
    }


def load_hyperparameter_lock(args, cfg, decision, identity_sha):
    path = args.public / "HYPERPARAMETER_LOCK.json"
    stage_a = args.public / "STAGE_A_DECISION.json"
    hyper = read_json(path)
    if (
        decision["status"] != "PASS_STAGE_A"
        or read_json(stage_a) != decision
        or sum(row["passed"] for row in decision["validation_folds"]) < 2
        or hyper
        != {
            "status": "LOCKED_AFTER_STAGE_A",
            "identity_sha256": identity_sha,
            "stage_a_sha256": file_hash(stage_a),
            "selected": decision["selected"],
        }
        or cfg["selection_seed"] != 42
    ):
        raise RuntimeError("The source-development selection is not authorized")
    return path, hyper["selected"]


def model_receipts_for_fold(args, cfg, fold, selected, identity_sha):
    result = {}
    for seed in cfg["meta_training_seeds"]:
        aliases = selected_mapper_trials(selected, seed)
        for name, trial in aliases.items():
            path = mapper_trial_path(args.private, fold["fold"], trial)
            stage = "A" if seed == cfg["selection_seed"] else "B"
            receipt = authenticate_trial(
                path, trial_identity(trial, fold, identity_sha, stage)
            )
            result[f"{seed}:{name}"] = {
                "folder": path.relative_to(args.private).as_posix(),
                "receipt_sha256": file_hash(path / "COMPLETE.json"),
                "mapper_sha256": receipt["files"]["MAPPER.pt"],
                "alpha": trial.alpha,
            }
    return result


def run_repetitions(args, cfg, plan, data, features, sealed, identity_sha, selected):
    trace, costs, fit_count = [], [], 0
    for fold in plan["folds"]:
        index = fold["fold"]
        reference, bases, phi = fold_inputs(args, cfg, fold, data, features)
        fit_ids = set(fold["fit_ids"])
        labels = Labels.read(args.data_root, fit_ids, fit_ids, sealed)
        fold_data = reference, bases, phi, data, labels
        for seed in cfg["meta_training_seeds"]:
            if seed == cfg["selection_seed"]:
                continue
            trials = selected_mapper_trials(selected[str(index)], seed)
            for trial in dict.fromkeys(trials.values()):
                receipt = fit_mapper_trial(
                    args, cfg, fold, fold_data, trial, identity_sha
                )
                folder = mapper_trial_path(args.private, index, trial)
                trace.extend(read_json(folder / "META_TRACE.json")["rows"])
                costs.append(receipt["identity"] | receipt["cost"])
                fit_count += 1
    if not 48 <= fit_count <= cfg["maximum_mapper_fits"] - 96:
        raise RuntimeError("Locked repetition count exceeds the registered budget")
    path = args.public / "stage_b"
    csv(path / "META_TRACE.csv", trace)
    csv(path / "COSTS.csv", costs)
    return {
        "status": "COMPLETE_STAGE_B",
        "identity_sha256": identity_sha,
        "fold_fit_roles": {
            str(fold["fold"]): fold["fit_ids"] for fold in plan["folds"]
        },
        "target_labels_read": 0,
        "outer_role_labels_read": 0,
        "unique_mapper_fits": fit_count,
        "matched_rate_extra_mapper_fits": fit_count - 48,
        "total_mapper_fits": 96 + fit_count,
        "total_mapper_adam_updates": (96 + fit_count) * cfg["train_episodes"],
        "total_differentiable_sgd_updates": (96 + fit_count)
        * cfg["train_episodes"]
        * cfg["core_steps"],
        "files": {
            name: file_hash(path / name) for name in ("META_TRACE.csv", "COSTS.csv")
        },
    }


def development_lock(args, cfg, plan, identity_sha, hyper_path, selected, *, create):
    marker = args.public / "DEVELOPMENT_LOCK.json"
    repeats_path = args.public / "stage_b" / "COMPLETE.json"
    repeats = read_json(repeats_path)
    if (
        repeats["status"] != "COMPLETE_STAGE_B"
        or repeats["identity_sha256"] != identity_sha
        or repeats["target_labels_read"] != 0
        or repeats["outer_role_labels_read"] != 0
    ):
        raise RuntimeError("The repetition receipt is missing or not source-only")
    for name, sha in repeats["files"].items():
        if file_hash(repeats_path.parent / name) != sha:
            raise RuntimeError("A Stage B public artifact changed")
    code = (
        Path(__file__),
        Path(__file__).with_name("summarize_spectral_objective.py"),
        Path(__file__).with_name("summarize_spectral_rotation.py"),
    )
    models = {
        str(fold["fold"]): model_receipts_for_fold(
            args,
            cfg,
            fold,
            selected[str(fold["fold"])],
            identity_sha,
        )
        for fold in plan["folds"]
    }
    expected_b = {
        model["folder"]
        for fold_models in models.values()
        for name, model in fold_models.items()
        if not name.startswith(f"{cfg['selection_seed']}:")
    }
    if len(expected_b) != repeats["unique_mapper_fits"]:
        raise RuntimeError("Mapper repeats and the locked paths differ")
    lock = {
        "status": "LOCKED_BEFORE_SOURCE_OUTER_LABELS",
        "identity_sha256": identity_sha,
        "hyperparameters_sha256": file_hash(hyper_path),
        "stage_b_sha256": file_hash(repeats_path),
        "scoring_code_sha256": {path.name: file_hash(path) for path in code},
        "mapper_artifacts": models,
        "target_labels_read": 0,
        "outer_role_labels_read": 0,
    }
    if create:
        bind(marker, lock)
    elif read_json(marker) != lock:
        raise RuntimeError(
            "The complete development lock or one of its dependencies changed"
        )
    return lock, file_hash(marker)


def digest_prediction(prediction):
    return hashlib.sha256(
        np.ascontiguousarray(prediction, dtype="<f8").tobytes()
    ).hexdigest()


def deterministic_outputs(args, cfg, data, ep, reference, bases, phi, selected):
    outputs = {"K0": ep.baseline}
    costs = {
        "K0": {
            "target_parameters": 0,
            "source_mapper_parameters": 0,
            "basis_storage_bytes": 0,
            "adapt_seconds": 0.0,
            "adapt_updates": 0,
            "inference_seconds": ep.base_predict_seconds,
        }
    }
    cal = selected["CAL2"]
    outputs["CAL2"], costs["CAL2"] = fit_cal2(ep, data, cal["ridge"], cal["eta"])
    static = core_model(reference, bases, "STATIC_CORE", phi, cfg, args.device)
    _, outputs["STATIC_CORE"], observed = fit_core(
        static,
        ep,
        data,
        "STATIC_CORE",
        cfg,
        selected["STATIC_CORE"]["alpha"],
        args.device,
    )
    costs["STATIC_CORE"] = model_accounting(static) | observed
    return outputs, costs


def frozen_mappers(
    args, cfg, fold, reference, bases, phi, selected, seed, identity_sha, mapper_lock
):
    aliases = selected_mapper_trials(selected, seed)
    models = {}
    for name, trial in aliases.items():
        if trial in models:
            continue
        entry = mapper_lock[f"{seed}:{name}"]
        path = args.private / entry["folder"]
        expected = trial_identity(
            trial, fold, identity_sha, "A" if seed == cfg["selection_seed"] else "B"
        )
        if (
            entry["alpha"] != trial.alpha
            or entry["receipt_sha256"] != file_hash(path / "COMPLETE.json")
            or entry["mapper_sha256"] != file_hash(path / "MAPPER.pt")
        ):
            raise RuntimeError("The locked mapper changed before inference")
        model = core_model(reference, bases, trial.method, phi, cfg, "cpu")
        budget = model_accounting(model)
        load_mapper(model, path, expected)
        models[trial] = model, budget, entry["mapper_sha256"]
    return aliases, models


def fitted_outputs(args, cfg, data, ep, aliases, models):
    outputs, costs, facts, states, seen = {}, {}, {}, {}, {}
    for name, trial in aliases.items():
        model, budget, state_sha = models[trial]
        if trial in seen:
            alias = seen[trial]
            outputs[name] = outputs[alias]
            costs[name] = costs[alias]
            facts[name] = facts[alias] | {
                "reused_prediction": True,
                "reused_from_method": alias,
            }
            continue
        seen[trial] = name
        model.to(args.device)
        state, outputs[name], cost = fit_core(
            model,
            ep,
            data,
            trial.method,
            cfg,
            trial.alpha,
            args.device,
        )
        states[name] = {key: value.detach().cpu() for key, value in state.items()}
        costs[name] = budget | cost
        facts[name] = {
            "alpha": trial.alpha,
            "source_mapper_sha256": state_sha,
            "reused_prediction": False,
        }
        if name == CANDIDATE:
            k = ep.query_start
            new_phi = torch.cat(
                (ep.descriptors["true"][:k], ep.descriptors["shuffled"][k:])
            )
            outputs[SAME_CORE], replay = predict_core(
                model, ep, data, new_phi, state, args.device
            )
            if not np.array_equal(
                outputs[name][:k], outputs[SAME_CORE][:k]
            ) or not math.isclose(
                cost["weight_frobenius_rms"],
                replay["weight_frobenius_rms"],
                rel_tol=1e-6,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Same-core Query perturbation changed the Support or core norm"
                )
            costs[SAME_CORE] = (
                costs[name] | replay | {"adapt_seconds": 0.0, "adapt_updates": 0}
            )
            facts[SAME_CORE] = facts[name] | {"same_ephemeral_core": True}
        model.cpu()

    norms = {name: value["weight_frobenius_rms"] for name, value in costs.items()}
    pairs = {}
    for reference in PARAMETER_REFERENCES:
        names = (CANDIDATE, reference)
        radius = min(norms[name] for name in names)
        if radius <= 1e-12:
            continue
        matched = []
        for name in names:
            trial = aliases[name]
            if radius == norms[name]:
                matched.append(outputs[name])
                continue
            model = models[trial][0].to(args.device)
            state = {key: value.to(args.device) for key, value in states[name].items()}
            matched.append(
                replay_at_norm(
                    model,
                    ep,
                    data,
                    trial.method,
                    state,
                    norms[name],
                    radius,
                    args.device,
                )
            )
            model.cpu()
        pairs[CANDIDATE, reference] = (*matched, radius)
    return outputs, costs, facts, pairs


def outer_slot(
    args,
    cfg,
    fold,
    seed,
    ep,
    data,
    protocols,
    selected,
    aliases,
    models,
    lora,
    deterministic,
):
    outputs, costs, facts, parameter_pairs = fitted_outputs(
        args, cfg, data, ep, aliases, models
    )
    baselines, baseline_costs = deterministic
    outputs |= baselines
    costs |= baseline_costs
    for method in DETERMINISTIC:
        facts[method] = {"reused_prediction": seed != cfg["selection_seed"]}
    setting = selected["LORA_SMALL"]
    for _, prediction, stats in lora_snapshots(
        lora,
        ep,
        data,
        args.device,
        setting["lr"],
        [setting["steps"]],
    ):
        outputs["LORA_SMALL"], costs["LORA_SMALL"] = prediction, stats
    facts["LORA_SMALL"] = {"reused_prediction": False, **setting}
    before = ep.scores(ep.baseline)
    for name, prediction in outputs.items():
        facts[name].update(
            {
                "is_deterministic": name in DETERMINISTIC,
                "source_prediction_hash": digest_prediction(prediction),
            }
        )
        after = ep.scores(prediction)
        costs[name] = costs[name] | {
            "support_mse_before": before["support_sampled_mse"],
            "support_mse_after": after["support_sampled_mse"],
            "support_full_mse_before": before["support_full_mse"],
            "support_full_mse_after": after["support_full_mse"],
        }
    if set(outputs) != set(OUTER_METHODS):
        raise RuntimeError("An outer slot lacks a registered comparison")
    k, a = ep.query_start, ep.anchor_start
    scores = score_episode(
        {
            "fold": fold,
            "training_seed": seed,
            "reference_seed": cfg["reference_seed"],
            "K": cfg["shot"],
            "draw": ep.draw,
        },
        query_ids=ep.query,
        grid=data.query,
        observed=ep.query_observed,
        targets=ep.query_y,
        baseline=ep.baseline[k:a],
        predictions={name: value[k:a] for name, value in outputs.items()},
        anchor_baseline=ep.baseline[a:],
        anchor_predictions={name: value[a:] for name, value in outputs.items()},
        query_protocols={uid: protocols[uid] for uid in ep.query},
        method_metadata=facts,
        costs=costs,
        matched_pairs=[
            (CANDIDATE, name)
            for name in (*RAW_REFERENCES, MATCHED_RATE, SAME_CORE)
            if name != "K0"
        ],
        parameter_matched={
            pair: (first[k:a], second[k:a], norm)
            for pair, (first, second, norm) in parameter_pairs.items()
        },
    )
    private = (
        args.private
        / "outer"
        / f"fold{fold}"
        / f"seed{seed}"
        / f"DRAW{ep.draw}_PREDICTIONS.npz"
    )
    save_npz(
        private,
        ids=np.array(ep.ids),
        cycles=data.query,
        methods=np.array(OUTER_METHODS),
        predictions=np.stack([outputs[method] for method in OUTER_METHODS]),
        parameter_matched_methods=np.array([name for _, name in parameter_pairs]),
        parameter_matched_predictions=np.array(
            [value[:2] for value in parameter_pairs.values()]
        ),
        parameter_matched_radii=np.array(
            [value[2] for value in parameter_pairs.values()]
        ),
    )
    return scores, {
        "path": private.relative_to(args.private).as_posix(),
        "sha256": file_hash(private),
    }


def verify_outer_fold(args, directory, receipt, index, lock_sha):
    if (
        receipt["status"] != "COMPLETE_LOCKED_SOURCE_OUTER_FOLD"
        or receipt["fold"] != index
        or receipt["development_lock_sha256"] != lock_sha
        or receipt["target_label_rows_loaded"] != 0
        or len(receipt["predictions"]) != 9
    ):
        raise RuntimeError("The existing source-outer fold belongs to another study")
    for name, expected in receipt["files"].items():
        if file_hash(directory / name) != expected:
            raise RuntimeError(f"Source-outer table changed: {name}")
    for item in receipt["predictions"]:
        if file_hash(args.private / item["path"]) != item["sha256"]:
            raise RuntimeError("A registered source prediction changed")
    return receipt


def run_outer_fold(
    args,
    cfg,
    fold,
    data,
    features,
    sealed,
    identity_sha,
    selected,
    lock_sha,
    mapper_lock,
    lora_parent,
    protocols,
):
    index = fold["fold"]
    directory = args.public / "outer" / f"fold{index}"
    marker = directory / "COMPLETE.json"
    if marker.exists():
        return verify_outer_fold(args, directory, read_json(marker), index, lock_sha)

    reference, bases, phi = fold_inputs(args, cfg, fold, data, features)
    # The caller authenticates the complete development lock and all mapper hashes.
    if file_hash(args.public / "DEVELOPMENT_LOCK.json") != lock_sha:
        raise RuntimeError("The development lock changed before label access")
    ids = role_ids(fold["outer"])
    labels = Labels.read(args.data_root, ids, ids, sealed)
    episodes = prepare_episodes(
        cfg,
        data,
        labels,
        fold["outer"],
        fold["anchors"],
        reference.to(args.device),
        phi,
        args.device,
    )
    reference.cpu()
    deterministic = [
        deterministic_outputs(args, cfg, data, ep, reference, bases, phi, selected)
        for ep in episodes
    ]
    cells, scores, paired, predictions = [], [], [], []
    for seed in cfg["meta_training_seeds"]:
        aliases, models = frozen_mappers(
            args,
            cfg,
            fold,
            reference,
            bases,
            phi,
            selected,
            seed,
            identity_sha,
            mapper_lock,
        )
        lora = lora_operator(reference, lora_parent, {"seed": seed}, args.device)
        for ep, cached in zip(episodes, deterministic, strict=True):
            (cell_rows, score_rows, pair_rows), artifact = outer_slot(
                args,
                cfg,
                index,
                seed,
                ep,
                data,
                protocols,
                selected,
                aliases,
                models,
                lora,
                cached,
            )
            cells.extend(cell_rows)
            scores.extend(score_rows)
            paired.extend(pair_rows)
            predictions.append(artifact | {"training_seed": seed, "draw": ep.draw})
            print(f"C fold={index} seed={seed} draw={ep.draw}", flush=True)
        del lora, models
    files = ("CELL_SCORES.csv", "EPISODE_SCORES.csv", "PAIR_SCORES.csv")
    for name, rows in zip(files, (cells, scores, paired), strict=True):
        csv(directory / name, rows)
    receipt = {
        "status": "COMPLETE_LOCKED_SOURCE_OUTER_FOLD",
        "fold": index,
        "development_lock_sha256": lock_sha,
        "source_outer_role_label_ids": sorted(ids),
        "target_label_rows_loaded": 0,
        "prediction_slots": len(predictions),
        "predictions": predictions,
        "files": {name: file_hash(directory / name) for name in files},
    }
    write_json(marker, receipt)
    return verify_outer_fold(args, directory, receipt, index, lock_sha)


def run_locked_stages(
    args,
    cfg,
    plan,
    data,
    features,
    sealed,
    identity_sha,
    lora_parent,
    protocols,
    decision,
):
    hyper_path, selected = load_hyperparameter_lock(args, cfg, decision, identity_sha)
    stage_b = args.public / "stage_b" / "COMPLETE.json"
    if not stage_b.exists():
        write_json(
            stage_b,
            run_repetitions(
                args, cfg, plan, data, features, sealed, identity_sha, selected
            ),
        )
    marker = args.public / "DEVELOPMENT_LOCK.json"
    lock, lock_sha = development_lock(
        args,
        cfg,
        plan,
        identity_sha,
        hyper_path,
        selected,
        create=not marker.exists(),
    )
    records = {}
    for fold in plan["folds"]:
        index = str(fold["fold"])
        records[index] = run_outer_fold(
            args,
            cfg,
            fold,
            data,
            features,
            sealed,
            identity_sha,
            selected[index],
            lock_sha,
            lock["mapper_artifacts"][index],
            lora_parent,
            protocols,
        )
    tables = [
        pd.concat(
            [
                pd.read_csv(
                    args.public / "outer" / f"fold{fold}" / name,
                    float_precision="round_trip",
                )
                for fold in cfg["folds"]
            ],
            ignore_index=True,
        )
        for name in ("CELL_SCORES.csv", "EPISODE_SCORES.csv", "PAIR_SCORES.csv")
    ]
    cost_a = pd.read_csv(args.public / "COSTS.csv")
    cost_b = pd.read_csv(stage_b.parent / "COSTS.csv")
    costs = pd.concat(
        [cost_a[cost_a.cost_kind == "SOURCE_META_TRAINING"], cost_b], ignore_index=True
    )
    summary = args.public / "summary"
    final = write_summary(
        summary,
        *tables,
        methods=OUTER_METHODS,
        selected_rates=selected,
        config=cfg,
        training_cost_rows=costs,
    )
    write_json(
        args.public / "FINAL_DECISION.json",
        final
        | {
            "development_lock_sha256": lock_sha,
            "summary_receipt_sha256": file_hash(summary / "FINAL_DECISION.json"),
            "outer_receipts": {
                index: file_hash(
                    args.public / "outer" / f"fold{index}" / "COMPLETE.json"
                )
                for index in records
            },
            "stage_b": read_json(stage_b),
        },
    )
    print(f"Stage C {final['direction']}", flush=True)
