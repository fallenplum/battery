"""Select shrinkage of frozen LoRA fits and authenticate the legacy HUST CAL2 anchor."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scripts.drs_peft import run_onesided as old
from scripts.drs_peft import run_specialization as specialization
from scripts.drs_peft.data import ROOT, file_hash, write_json
from scripts.drs_peft.onesided_engine import FewShotEngine, _fingerprint
from scripts.drs_peft.onesided_protocol import payload_hash
from scripts.drs_peft.run_onesided_reference import load_reference_model, reference_folder
from scripts.drs_peft.specialization_engine import (
    SpecializationEngine,
    load_pretrained_reference,
)
from scripts.drs_peft.specialization_protocol import load_target_contract
from scripts.drs_peft.train import configure, read_json, save_torch
from scripts.drs_peft.transfer_scope_data import SPLIT_LOCK, load_future_rows, load_prefixes


BASE = ROOT / "results/drs_peft/specialization_comparison_v1"
BASE_PRIVATE = ROOT / "runtime/drs_peft/specialization_comparison_v1"
TARGET_BASE = BASE / "target_total_K"
TARGET_CONTRACT = specialization.CONFIG.with_name("TARGET_TOTAL_K.json")
METHODS = {"PRETRAINED_LORA": "WARMUP", "SPECIALIZED_LORA": "SPECIALIZED"}
FOLDS, SEEDS, SHOTS = (0, 1, 2), (42, 123, 2026), (1, 3, 5)
ETAS = (0.0, 0.1, 0.25, 0.5, 1.0)
SUFFIX = "_ZERO_RETAINING"
STAGE = "TRANSFER_SCOPE_V2"


def contracts(args):
    context = SimpleNamespace(config=specialization.CONFIG)
    config, registration = specialization.load_contract(context)
    _, override, episodes = load_target_contract(TARGET_CONTRACT, specialization.CONFIG)
    parent_path = ROOT / config["hyperparameter_selection"]["parent_target_lock_file"]
    source_lock = read_json(BASE / "TARGET_LOCK.json")
    target_lock = read_json(TARGET_BASE / "TARGET_LOCK.json")
    if (
        target_lock["source_choice_sha256"] != file_hash(BASE / "TARGET_LOCK.json")
        or target_lock["target_episode_contract_sha256"]
        != override["episode_contract_sha256"]
    ):
        raise RuntimeError("The previous source choices or target episodes changed")
    for method in METHODS:
        choices = target_lock["selected_hyperparameters"][method]
        validate_grid(choices)
        if choices != source_lock["selected_hyperparameters"][method]:
            raise RuntimeError("A target LoRA choice differs from the original source lock")
    return config, registration, episodes, target_lock, read_json(parent_path)


def validate_grid(choices):
    if set(choices) != set(map(str, SHOTS)) or any(
        float(choice["eta"]) not in ETAS[2:]
        or int(choice["steps"]) not in (10, 30)
        or float(choice["learning_rate"]) not in (1e-4, 1e-3)
        for choice in choices.values()
    ):
        raise RuntimeError("The frozen LoRA fits are outside their registered budget")


def score_data(args, domain, ids, protocols, *, protocol_uids=None):
    targets, observed = load_future_rows(domain, args.data_root, ids)
    lookup = {uid: index for index, uid in enumerate(ids)}
    protocol_uids = ids if protocol_uids is None else protocol_uids
    return SimpleNamespace(
        ids=protocol_uids, targets=targets, observed=observed,
        query=np.arange(101, 5001, dtype=np.float32),
        protocol_ids=np.asarray([protocols[uid] for uid in protocol_uids]),
        rows=lambda requested: np.asarray([lookup[uid] for uid in requested], dtype=np.int64),
    )


def source_data(args, registration):
    ids = sorted({
        uid for fold in args.fold_ids
        for task in [old.task_for(registration, "inner", fold)]
        for uid in (*task["query_ids"], *task["support_pool_ids"])
    })
    split = read_json(SPLIT_LOCK)
    roles = split["source_cell_roles"]
    if not set(ids) <= set(roles["source_train"]) | set(roles["source_meta_validation"]):
        raise RuntimeError("The source labels must come from registered source cells")
    if set(ids) & set(roles["target_test"]):
        raise RuntimeError("Source shrinkage cannot decode target outcomes")
    protocols = {uid: p for p, group in split["source_groups"].items() for uid in group}
    data = score_data(args, "MATR", ids, protocols)
    prefix = load_prefixes("MATR", args.data_root)
    prefix_rows = prefix.rows(ids)
    inputs = {key: value[prefix_rows] for key, value in prefix.inputs.items() if key != "ids"}
    return data, dict(inputs, ids=ids)


def setup_engine(args, method, fold, seed, inputs, grid, parent_implementation):
    identity = specialization.reference_identity(args, fold, seed, METHODS[method])
    model = (
        load_pretrained_reference(args.reference_root, args.backbone_dir, fold, seed)
        if method == "PRETRAINED_LORA"
        else load_reference_model(args.reference_root, args.backbone_dir, fold, seed)
    )
    engine_class = FewShotEngine if parent_implementation else SpecializationEngine
    engine = engine_class(
        model, inputs, grid, identity,
        args.private_root / "e0" / "engine" / f"f{fold}s{seed}" / METHODS[method],
        device=args.device, seed=seed,
    )
    engine.register_method("LoRA")
    return engine


def source_state(engine, legacy_cache, support, hyper):
    """Authenticate the existing unit-scale snapshot without entering the optimizer."""
    _, identity_support = engine._support(support)
    identity = dict(
        engine._method_identity("LoRA"),
        support=identity_support,
        lr=float(hyper["learning_rate"]),
        steps=[10, 30],
        kind="nonlinear_support",
    )
    path = legacy_cache / "adaptation_states" / f"{_fingerprint(identity)}.pt"
    receipt = path.with_suffix(".json")
    snapshots = engine._load_trial(path, receipt, identity)
    state = snapshots[int(hyper["steps"])]["state"]
    if state.get("adaptation_scale") != 1.0:
        raise RuntimeError("The saved source snapshot is not at unit adapter scale")
    if any(state.get(key) != value for key, value in engine._method_identity("LoRA").items()):
        raise RuntimeError("The source snapshot is incompatible with this engine")
    return state, path, file_hash(path)


def source_run_identity(args):
    files = (
        Path(__file__), TARGET_CONTRACT, TARGET_BASE / "TARGET_LOCK.json",
        BASE / "TARGET_LOCK.json", old.PUBLIC / "TARGET_LOCK.json",
        args.data_root / "MATR/COMPLETE.json", SPLIT_LOCK,
        Path(__file__).with_name("transfer_scope_data.py"),
    )
    return {specialization.relative(path): file_hash(path) for path in files}


def source_scores(args, config, registration, target_lock, data, inputs, fold, seed, method):
    task = old.task_for(registration, "inner", fold)
    query_ids = task["query_ids"]
    stream = args.public_root / "e0/source" / f"f{fold}s{seed}" / f"{method}.json"
    run_identity = source_run_identity(args)
    if stream.exists():
        cached = read_json(stream)
        if cached["identity"] != run_identity or cached["query_ids"] != query_ids:
            raise RuntimeError("A V2 source shrinkage block changed identity")
        return
    pretrained = method == "PRETRAINED_LORA"
    engine = setup_engine(args, method, fold, seed, inputs, data.query, not pretrained)
    if pretrained:
        legacy_cache = BASE_PRIVATE / "e/i" / f"f{fold}s{seed}" / "P"
        legacy_results = BASE / "inner" / f"fold{fold}" / f"seed{seed}" / method
        grid = specialization.trial_grid(config)
        marker_name = "COMPLETE.json"
    else:
        legacy_cache = old.PRIVATE / "engine/D2" / f"fold{fold}" / f"seed{seed}"
        legacy_results = old.PUBLIC / "inner/D2" / f"fold{fold}" / f"seed{seed}" / "LoRA"
        grid = old.hyperparameters(read_json(old.CONFIG), "LoRA")
        marker_name = "SELECTED.json"
    _, zero_summary = old.score_predictions(data, query_ids, engine.baseline_prediction(query_ids))
    zero_mae = float(zero_summary["mae"])
    rows, old_scores_reused, new_prediction_count, cache_hits = [], 0, 0, 0
    for k in SHOTS:
        legacy = legacy_results / f"K{k}"
        table_path = legacy / "TRIALS.csv"
        marker = read_json(legacy / marker_name)
        table_sha = file_hash(table_path)
        expected = marker["identity"]
        checkpoint_key = "source_checkpoint_sha256" if pretrained else "reference_checkpoint_sha256"
        if (
            table_sha != marker["trials_csv_sha256"]
            or expected[checkpoint_key] != engine.source_sha256
            or expected["query_ids"] != query_ids
            or expected["K"] != k or expected["fold_id"] != fold or expected["seed"] != seed
            or expected["method"] != (method if pretrained else "LoRA")
            or expected.get("episode_contract_sha256", expected.get("episode_sha256"))
            != registration["episode_contract_sha256"]
        ):
            raise RuntimeError("An old LoRA score table uses another source task")
        table = pd.read_csv(table_path).set_index(["trial", "draw"], verify_integrity=True)
        if len(table) != 60:
            raise RuntimeError("Every inherited LoRA grid has 12 trials times 5 draws")
        locked = target_lock["selected_hyperparameters"][method][str(k)]
        recomputed = {}
        for draw in old.shot_for(task, k)["draws"]:
            draw_index, uids = int(draw["draw"]), tuple(draw["support_ids"])
            if len(set(uids)) != k or set(uids).intersection(query_ids):
                raise RuntimeError("Source Support identities conflict with the registered split")
            if uids in recomputed:
                mae_01, state_path, state_sha = recomputed[uids]
            else:
                support = old.make_support(data, inputs, sorted(uids), args.device)
                state, state_path, state_sha = source_state(engine, legacy_cache, support, locked)
                prediction = engine.predict_state("LoRA", dict(state, adaptation_scale=0.1), query_ids)
                _, summary = old.score_predictions(data, query_ids, prediction)
                mae_01 = float(summary["mae"])
                recomputed[uids] = mae_01, state_path, state_sha
                new_prediction_count += 1
                cache_hits += 1
            for eta in ETAS:
                if not eta:
                    mae, origin = zero_mae, "SOURCE_OWN_FAMILY_ZERO"
                elif eta == 0.1:
                    mae, origin = mae_01, "NEW_NONLINEAR_FORWARD_SAVED_UNIT_SCALE_STATE"
                else:
                    trial = grid.index(dict(locked, eta=eta))
                    mae, origin = float(table.loc[(trial, draw_index), "mae"]), "AUTHENTICATED_SOURCE_TRIAL"
                    old_scores_reused += 1
                if not np.isfinite(mae):
                    raise RuntimeError("The source LoRA shrinkage score is nonfinite")
                rows.append(dict(
                    fold=fold, seed=seed, method=method, K=k, draw=draw_index, eta=eta,
                    mae=mae, origin=origin, support_ids=list(uids),
                    legacy_trials_sha256=table_sha,
                    state_path=specialization.relative(state_path), state_sha256=state_sha,
                ))
    if engine.counters["nonlinear_optimizer_updates"]:
        raise RuntimeError("Source shrinkage requested an unauthorized new fit")
    write_json(stream, dict(
        status="COMPLETE", identity=run_identity, query_ids=query_ids,
        rows=rows, old_score_slots_reused=old_scores_reused,
        new_nonzero_prediction_states=new_prediction_count,
        snapshot_cache_hits=cache_hits, counters=dict(engine.counters),
        supervised_source_uids=data.ids, target_future_rows_decoded=0,
    ))


def select_source(args, target_lock):
    run_identity = source_run_identity(args)
    frames, blocks = [], []
    for fold, seed, method in product(FOLDS, SEEDS, METHODS):
        path = args.public_root / "e0/source" / f"f{fold}s{seed}" / f"{method}.json"
        block = read_json(path)
        if (
            block["status"] != "COMPLETE" or block["identity"] != run_identity
            or block["counters"]["nonlinear_optimizer_updates"] != 0
            or len(block["rows"]) != 75
        ):
            raise RuntimeError("Source shrinkage selection needs an intact 75-slot block")
        frames.extend(block["rows"])
        blocks.append(dict(
            path=specialization.relative(path), sha256=file_hash(path),
            old_score_slots_reused=block["old_score_slots_reused"],
            new_nonzero_prediction_states=block["new_nonzero_prediction_states"],
            snapshot_cache_hits=block["snapshot_cache_hits"], counters=block["counters"],
        ))
    rows = pd.DataFrame(frames)
    if set(rows.method) != set(METHODS) or rows.duplicated(["fold", "seed", "method", "K", "draw", "eta"]).any():
        raise RuntimeError("The source shrinkage candidates do not form a unique matrix")
    selection, scores = {}, []
    for method in METHODS:
        selection[method + SUFFIX] = {}
        for k in SHOTS:
            candidates = rows.loc[(rows.method == method) & (rows.K == k)]
            grouped = candidates.groupby("eta").mae.agg(["mean", "size"]).reset_index()
            if tuple(grouped.eta) != ETAS or not grouped["size"].eq(45).all():
                raise RuntimeError("Every source shrinkage eta requires 45 registered slots")
            winner = grouped.sort_values(["mean", "eta"]).iloc[0]
            selection[method + SUFFIX][str(k)] = dict(
                target_lock["selected_hyperparameters"][method][str(k)], eta=float(winner.eta)
            )
            scores.extend(dict(method=method, K=k, **item) for item in grouped.to_dict("records"))
    result = {
        "status": "SOURCE_ONLY_SHRINKAGE_SELECTED",
        "identity": run_identity,
        "selected_hyperparameters": selection,
        "candidate_scores": scores,
        "source_blocks": blocks,
        "selection_rule": "Full observed Query cell-mean MAE over 3 folds, 3 seeds, 5 draws. Ties use smaller eta.",
        "new_optimizer_updates": 0,
        "target_Query_labels_used": 0,
        "target_future_rows_decoded": 0,
    }
    write_json(args.public_root / "LORA_SOURCE_SELECTION.json", result)
    return result


def parent_folder(base, task, seed, method, k, draw):
    return base / task / f"seed{seed}" / method / f"K{k}_draw{draw}"


def authenticated_parent(folder, expected):
    marker = folder / "RECEIPT.json"
    receipt = read_json(marker)
    if receipt["status"] != "SCORED" or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(f"An inherited target receipt changed identity: {marker}")
    if file_hash(folder / "CELL_METRICS.csv") != receipt["cell_metrics_sha256"]:
        raise RuntimeError(f"An inherited target metrics file changed: {folder}")
    prediction = specialization.load_prediction(receipt)
    return receipt, prediction


def expected_parent(args, task, seed, support, method, k, draw, hyper):
    family = "WARMUP" if method == "WARMUP_ZERO" else METHODS.get(method, "SPECIALIZED")
    reference = specialization.reference_identity(
        args, "final", seed, family
    )
    return {
        "domain": task["domain"], "task_id": task["task_id"], "fold_id": "final",
        "seed": seed, "support_ids": support, "query_ids": task["query_ids"],
        "method": method, "K": k, "draw": draw, "hyperparameters": hyper,
        "source_checkpoint_sha256": reference["source_checkpoint_sha256"],
    }


def audit_hust_cal2(args, config, registration, episodes, target_lock, parent_lock):
    """Verify the distinct, legacy CAL2 anchor against the identical HUST episodes."""
    task, = [item for item in episodes["target_tasks"] if item["domain"] == "HUST"]
    original, = [item for item in registration["target_tasks"] if item["domain"] == "HUST"]
    if task != original:
        raise RuntimeError("HUST requires the unchanged registered target task")
    rows = []
    for seed, k in product(SEEDS, SHOTS):
        for draw in old.shot_for(task, k)["draws"]:
            index, support = draw["draw"], draw["support_ids"]
            hyper = parent_lock["selected_hyperparameters"]["CAL2"][str(k)]
            expected = expected_parent(args, task, seed, support, "CAL2", k, index, hyper)
            dictionary = reference_folder(args.reference_root, "final", seed) / "geometry/DICTIONARIES.pt"
            dictionary_sha = file_hash(dictionary)
            if dictionary_sha != target_lock["final_OUT_dictionary_sha256"][str(seed)]:
                raise RuntimeError("The legacy HUST reference dictionary changed")
            expected.update(
                stage="D4", config_sha256=config["parent_config_sha256"],
                execution_sha256=file_hash(Path(old.__file__)),
                engine_sha256=file_hash(ROOT / "scripts/drs_peft/onesided_engine.py"),
                dictionary_sha256=dictionary_sha,
            )
            folder = parent_folder(old.PUBLIC / "units/D4", task["task_id"], seed, "CAL2", k, index)
            receipt, _ = authenticated_parent(folder, expected)
            state = read_parent_state(receipt, receipt)
            if (
                state["kind"] != "coordinates" or state["method"] != "CAL2"
                or np.asarray(state["a"]).shape != (2,)
                or not np.isfinite(state["a"]).all()
            ):
                raise RuntimeError("Legacy CAL2 did not save an authenticated two-coordinate state")
            pooled_zero = parent_folder(TARGET_BASE / "units", task["task_id"], seed, "SPECIALIZED_ZERO", 0, 0)
            old_zero = parent_folder(old.PUBLIC / "units/D4", task["task_id"], seed, "ZERO", 0, 0)
            same_zero = read_json(pooled_zero / "RECEIPT.json")
            legacy_zero = read_json(old_zero / "RECEIPT.json")
            if (
                same_zero["parent_receipt_sha256"] != file_hash(old_zero / "RECEIPT.json")
                or same_zero["prediction_sha256"] != legacy_zero["prediction_sha256"]
                or same_zero["source_checkpoint_sha256"] != expected["source_checkpoint_sha256"]
            ):
                raise RuntimeError("Legacy CAL2 and pooled HUST do not use the same reference function")
            rows.append(dict(
                seed=seed, K=k, draw=index, hyperparameters=hyper,
                support_ids=support, query_ids=task["query_ids"],
                source_checkpoint_sha256=expected["source_checkpoint_sha256"],
                receipt_path=specialization.relative(folder / "RECEIPT.json"),
                receipt_sha256=file_hash(folder / "RECEIPT.json"),
                prediction_sha256=receipt["prediction_sha256"], state_sha256=receipt["state_sha256"],
                cell_metrics_sha256=receipt["cell_metrics_sha256"],
            ))
    write_json(args.public_root / "HUST_LEGACY_CAL2_REUSE.json", dict(
        status="AUTHENTICATED_LEGACY_ANCHOR", slots=rows,
        source_fit_and_target_identity_match=True,
        V2_fixed_source_ridge_identity_match=False,
        registered_scope="Legacy frozen CAL2 anchor. Its Support-Gram-scaled ridge predates V2 G-CAL2.",
        new_optimizer_updates=0, new_model_forwards=0, target_Query_label_arrays_read=0,
    ))


def read_parent_state(receipt, envelope_receipt):
    saved = torch.load(ROOT / receipt["state_path"], map_location="cpu", weights_only=False)
    state_identity = saved["identity"]
    canonical = specialization.state_identity(envelope_receipt)
    if any(canonical.get(key) != value for key, value in state_identity.items()):
        raise RuntimeError("The serialized adaptation state belongs to another episode")
    if (
        saved.get("prediction_sha256", receipt["prediction_sha256"]) != receipt["prediction_sha256"]
        or saved["state"]["source_checkpoint_sha256"] != receipt["source_checkpoint_sha256"]
    ):
        raise RuntimeError("The serialized adaptation state reports another source or prediction")
    return saved["state"]


def open_lora_state(receipt):
    envelope = receipt
    if receipt.get("parent_reused"):
        path = ROOT / receipt["parent_receipt_path"]
        if file_hash(path) != receipt["parent_receipt_sha256"]:
            raise RuntimeError("The HUST LoRA parent receipt changed")
        envelope = read_json(path)
        for key in (
            "state_sha256", "prediction_sha256", "source_checkpoint_sha256", "support_ids",
            "query_ids", "seed", "hyperparameters", "K", "draw",
        ):
            if receipt[key] != envelope[key]:
                raise RuntimeError("The HUST LoRA parent envelope differs from the pooled receipt")
    state = read_parent_state(receipt, envelope)
    if (
        state["kind"] != "lowrank" or state["method"] != "LoRA"
        or state["adaptation_scale"] != receipt["hyperparameters"]["eta"]
    ):
        raise RuntimeError("Legacy target LoRA must retain unscaled factors and one scale")
    factors = state["selected_state"]
    scale_keys = [key for key in factors if key.endswith("core_adaptation_scale")]
    if len(scale_keys) != 12 or any(float(factors[key]) != 1.0 for key in scale_keys):
        raise RuntimeError("The saved LoRA factors were not captured at unit scale")
    return state


def target_run(args, episodes, old_lock, phase):
    from scripts.drs_peft.run_transfer_scope import read_target_lock

    lock = read_target_lock(args)
    selection_path = args.public_root / "LORA_SOURCE_SELECTION.json"
    selection = read_json(selection_path)
    if selection["status"] != "SOURCE_ONLY_SHRINKAGE_SELECTED" or selection["identity"] != source_run_identity(args):
        raise RuntimeError("The source-only LoRA shrinkage selection changed")
    for method, chosen in selection["selected_hyperparameters"].items():
        if lock["selected_hyperparameters"].get(method) != chosen:
            raise RuntimeError("The unified V2 target lock omits a source LoRA choice")
    lock_sha = file_hash(args.public_root / "TARGET_TOTAL_K_LOCK.json")
    if phase == "target-score":
        for task in episodes["target_tasks"]:
            if task["domain"] not in args.domains:
                continue
            for seed, method, k in product(args.seeds, args.methods, (0, *SHOTS)):
                for draw in old.shot_for(task, k)["draws"]:
                    folder = parent_folder(
                        args.public_root / "e0/units", task["task_id"], seed,
                        method + SUFFIX, k, draw["draw"],
                    )
                    receipt = read_json(folder / "RECEIPT.json")
                    if receipt["target_lock_sha256"] != lock_sha or receipt["status"] not in ("PREDICTIONS_LOCKED", "SCORED"):
                        raise RuntimeError("Complete locked LoRA predictions are required before scoring")
                    specialization.load_prediction(receipt)
    for task in episodes["target_tasks"]:
        domain, task_id = task["domain"], task["task_id"]
        if domain not in args.domains:
            continue
        data = score_data(
            args, domain, task["query_ids"], task["protocol_by_id"],
            protocol_uids=[*task["query_ids"], *task["support_pool_ids"]],
        ) if phase == "target-score" else None
        prefix = None
        for seed, method in product(args.seeds, args.methods):
            family = METHODS[method]
            zero_method = "WARMUP_ZERO" if family == "WARMUP" else "SPECIALIZED_ZERO"
            zero_folder = parent_folder(TARGET_BASE / "units", task_id, seed, zero_method, 0, 0)
            zero_identity = expected_parent(args, task, seed, [], zero_method, 0, 0, {})
            zero_receipt, zero = authenticated_parent(zero_folder, zero_identity)
            choice = selection["selected_hyperparameters"][method + SUFFIX]
            engine = None
            for k in (0, *SHOTS):
                old_hyper = old_lock["selected_hyperparameters"][method][str(k)] if k else {}
                hyper = choice[str(k)] if k else {}
                if k and dict(hyper, eta=old_hyper["eta"]) != old_hyper:
                    raise RuntimeError("V2 shrinkage may change only eta")
                for draw in old.shot_for(task, k)["draws"]:
                    index, support = int(draw["draw"]), draw["support_ids"]
                    name = method + SUFFIX
                    folder = parent_folder(args.public_root / "e0/units", task_id, seed, name, k, index)
                    receipt_path = folder / "RECEIPT.json"
                    expected = expected_parent(args, task, seed, support, method, k, index, hyper)
                    expected.update(
                        method=name, stage=STAGE, reference_family=family,
                        is_duplicate=draw.get("duplicate_of_draw") is not None,
                        duplicate_of_draw=draw.get("duplicate_of_draw"),
                        target_lock_sha256=lock_sha,
                        target_Query_labels_used_for_adaptation_or_selection=0,
                    )
                    if receipt_path.exists():
                        receipt = read_json(receipt_path)
                        if any(receipt.get(key) != value for key, value in expected.items()):
                            raise RuntimeError("A V2 LoRA shrinkage receipt changed identity")
                        prediction = specialization.load_prediction(receipt)
                        if phase == "target-score" and receipt["status"] != "SCORED":
                            specialization.write_family_metrics(data, folder, receipt, prediction, zero)
                        elif receipt["status"] == "SCORED" and file_hash(folder / "CELL_METRICS.csv") != receipt["cell_metrics_sha256"]:
                            raise RuntimeError("V2 LoRA scored facts changed")
                        continue
                    if phase == "target-score":
                        raise RuntimeError("Predictions must be locked before target scores")
                    if len(set(support)) != k or set(support).intersection(task["query_ids"]):
                        raise RuntimeError("The target LoRA slot violates the fixed Support and Query identities")
                    state = None
                    original_receipt = zero_receipt
                    if k:
                        original_folder = parent_folder(TARGET_BASE / "units", task_id, seed, method, k, index)
                        original_identity = expected_parent(args, task, seed, support, method, k, index, old_hyper)
                        original_receipt, fitted = authenticated_parent(original_folder, original_identity)
                        state = open_lora_state(original_receipt)
                    counters = {"new_optimizer_updates": 0, "new_model_forwards": 0}
                    if not k or hyper["eta"] == 0.0:
                        saved = zero_receipt
                        origin = "AUTHENTICATED_OWN_FAMILY_ZERO"
                    elif hyper["eta"] == old_hyper["eta"]:
                        saved = original_receipt
                        origin = "AUTHENTICATED_LEGACY_LORA_PREDICTION"
                    else:
                        if prefix is None:
                            prefix = load_prefixes(domain, args.data_root)
                        if engine is None:
                            engine = setup_engine(args, method, "final", seed, prefix.inputs, prefix.query, bool(original_receipt.get("parent_reused")))
                        original_state = dict(state)
                        state["adaptation_scale"] = hyper["eta"]
                        cache_identity = specialization.state_identity(expected)
                        cache_identity.pop("duplicate_of_draw")
                        cache_identity.pop("is_duplicate")
                        identifier = payload_hash(cache_identity)
                        private = args.private_root / "e0/states" / identifier[:24]
                        state_path, pred_path = private / "STATE.pt", private / "PREDICTIONS.npz"
                        marker = private / "COMPLETE.json"
                        if marker.exists():
                            saved = read_json(marker)
                            if saved["identity"] != cache_identity:
                                raise RuntimeError("A cached rescaled LoRA state changed identity")
                            specialization.load_prediction(dict(saved, query_ids=task["query_ids"]))
                            origin = "REUSED_DUPLICATE_RESCALED_LORA_PREDICTION"
                        else:
                            private.mkdir(parents=True, exist_ok=True)
                            if engine.counters["adapted_cell_forwards"] == 0:
                                replay = engine.predict_state("LoRA", original_state, task["query_ids"][:1])
                                if not np.allclose(replay, fitted[:1], rtol=1e-5, atol=1e-6):
                                    raise RuntimeError("The loaded LoRA fit failed original-scale replay")
                                counters["new_model_forwards"] += 1
                            prediction = engine.predict_state("LoRA", state, task["query_ids"])
                            np.savez_compressed(pred_path, ids=np.asarray(task["query_ids"]), query=prefix.query, prediction=prediction)
                            save_torch(state_path, dict(identity=cache_identity, state=state, prediction_sha256=file_hash(pred_path)))
                            saved = dict(
                                identity=cache_identity,
                                state_path=specialization.relative(state_path), state_sha256=file_hash(state_path),
                                prediction_path=specialization.relative(pred_path), prediction_sha256=file_hash(pred_path),
                            )
                            write_json(marker, saved)
                            counters["new_model_forwards"] += len(task["query_ids"])
                            origin = "NEW_NONLINEAR_FORWARD_SAVED_LORA_FACTORS"
                    references = {key: saved[key] for key in ("state_path", "state_sha256", "prediction_path", "prediction_sha256")}
                    receipt = dict(
                        expected, **references, status="PREDICTIONS_LOCKED",
                        adaptation_id=payload_hash(dict(
                            domain=domain, seed=seed, method=name, K=k, support_ids=sorted(support),
                            hyperparameters=hyper, target_lock_sha256=lock_sha,
                        )),
                        reused_state_sha256=original_receipt["state_sha256"],
                        origin=origin, details=dict(trainable_parameters=9216 if k and hyper["eta"] else 0),
                        new_execution_counters=counters,
                        historical_target_Query_exposure=True,
                    )
                    write_json(receipt_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("source", "source-select", "audit-reuse", "target-predict", "target-score"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/drs_peft/transfer_scope_v2/CONFIG.json")
    parser.add_argument("--public-root", type=Path, default=ROOT / "results/drs_peft/transfer_scope_v2")
    parser.add_argument("--private-root", type=Path, default=ROOT / "runtime/drs_peft/transfer_scope_v2")
    parser.add_argument("--reference-root", type=Path, default=old.REFERENCES)
    parser.add_argument("--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1")
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-ids", nargs="+", type=int, choices=FOLDS, default=FOLDS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=tuple(METHODS))
    parser.add_argument("--domains", nargs="+", choices=("MATR", "HUST"), default=("MATR", "HUST"))
    args = parser.parse_args()
    configure()
    config, registration, episodes, target_lock, parent_lock = contracts(args)
    if args.phase == "source":
        data, inputs = source_data(args, registration)
        for fold, seed, method in product(args.fold_ids, args.seeds, args.methods):
            source_scores(args, config, registration, target_lock, data, inputs, fold, seed, method)
            print(f"LoRA source shrinkage complete: fold={fold} seed={seed} method={method}", flush=True)
    elif args.phase == "source-select":
        print(select_source(args, target_lock)["selected_hyperparameters"], flush=True)
    elif args.phase == "audit-reuse":
        audit_hust_cal2(args, config, registration, episodes, target_lock, parent_lock)
        print("45 historical HUST CAL2 receipts and their predictions verified", flush=True)
    else:
        target_run(args, episodes, target_lock, args.phase)


if __name__ == "__main__":
    main()
