"""Run the MATR source experiment for the aligned Query objective."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from scripts.drs_peft.data import CACHE, ROOT, file_hash, write_json
from scripts.drs_peft.run_abba_moirai import csv
from scripts.drs_peft.run_spectral_rotation import input_features
from scripts.drs_peft.spectral_objective_engine import (
    V1,
    MapperTrial,
    baseline_key,
    baseline_trials,
    fit_mapper_trial,
    fold_inputs,
    mapper_trial_path,
    prepare_episodes,
    stage_a_decision,
    trial_entry,
    validation_selection,
)
from scripts.drs_peft.spectral_objective_protocol import (
    CONFIG,
    META_METHODS,
    META_OBJECTIVES,
    PRIVATE,
    PUBLIC,
    build_plan,
)
from scripts.drs_peft.spectral_rotation_engine import Labels
from scripts.drs_peft.spectral_rotation_protocol import role_ids
from scripts.drs_peft.train import bind, configure, read_json, seed_all
from scripts.drs_peft.transfer_scope_data import load_prefixes
from scripts.drs_peft.transfer_scope_reference import BACKBONE, REFERENCES


def identity_for(args, cfg, plan_path, receipt, data):
    files = (
        "run_spectral_objective.py",
        "spectral_objective.py",
        "spectral_objective_engine.py",
        "spectral_objective_protocol.py",
        "spectral_rotation.py",
        "spectral_rotation_engine.py",
        "run_spectral_rotation.py",
        "transfer_scope_data.py",
    )
    return {
        "config_sha256": file_hash(args.config),
        "plan_sha256": file_hash(plan_path),
        "lora_config_sha256": file_hash(ROOT / cfg["lora_parent_config"]),
        "feature_receipt_sha256": file_hash(receipt),
        "prefix_cache_manifest_sha256": data.cache_manifest_sha256,
        "code_sha256": {
            name: file_hash(Path(__file__).with_name(name)) for name in files
        },
        "hardware": torch.cuda.get_device_name(args.device)
        if args.device.startswith("cuda")
        else "CPU",
        "torch_version": str(torch.__version__),
        "device": args.device,
    }


def check_fold_files(args, record, identity_sha):
    folder = args.public / "stage_a" / f"fold{record['fold']}"
    if (
        record["identity_sha256"] != identity_sha
        or record["status"] != "COMPLETE_STAGE_A_FOLD"
        or record["outer_role_labels_read"] != 0
        or record["target_labels_read"] != 0
        or len(record["mapper_trials"]) != 32
        or len(record["baseline_trials"]) != 25
    ):
        raise RuntimeError("Stage A fold receipt is stale or incomplete")
    for name, sha in record["files"].items():
        if file_hash(folder / name) != sha:
            raise RuntimeError(f"Stage A public file changed: {name}")
    for entry in (*record["mapper_trials"], *record["baseline_trials"]):
        for name, sha in entry["files"].items():
            if file_hash(args.private / entry["path"] / name) != sha:
                raise RuntimeError(
                    f"Stage A private artifact changed: {entry['path']}/{name}"
                )
    return record


def run_stage_a_fold(
    args, cfg, fold, data, features, sealed, identity_sha, lora_parent
):
    index = fold["fold"]
    output = args.public / "stage_a" / f"fold{index}"
    marker = output / "COMPLETE.json"
    if marker.exists():
        return check_fold_files(args, read_json(marker), identity_sha)
    reference, bases, phi = fold_inputs(
        args, cfg, fold, data, features, check_reference=True
    )
    fit_ids = set(fold["fit_ids"])
    validation_ids = role_ids(fold["validation"])
    fit_labels = Labels.read(args.data_root, fit_ids, fit_ids, sealed)
    val_labels = Labels.read(args.data_root, validation_ids, validation_ids, sealed)
    validation = prepare_episodes(
        cfg,
        data,
        val_labels,
        fold["validation"],
        fold["anchors"],
        reference.to(args.device),
        phi,
        args.device,
    )
    reference.cpu()
    fold_data = reference, bases, phi, data, fit_labels
    records, traces, validation_rows, costs = [], [], [], []
    for method, objective, rate in itertools.product(
        META_METHODS, META_OBJECTIVES, cfg["core_rates"]
    ):
        trial = MapperTrial(method, objective, rate, cfg["selection_seed"])
        receipt = fit_mapper_trial(
            args,
            cfg,
            fold,
            fold_data,
            trial,
            identity_sha,
            validation=validation,
        )
        folder = mapper_trial_path(args.private, index, trial)
        records.append(trial_entry(args.private, folder, receipt))
        traces.extend(read_json(folder / "META_TRACE.json")["rows"])
        validation_rows.extend(receipt["validation"])
        costs.append(
            receipt["identity"]
            | {"cost_kind": "SOURCE_META_TRAINING"}
            | receipt["cost"]
        )

    baseline_records = []
    for receipt in baseline_trials(
        args, cfg, index, fold_data, validation, lora_parent, identity_sha
    ):
        key = baseline_key(
            receipt["identity"]["method"], receipt["identity"]["setting"]
        )
        folder = args.private / "baseline_trials" / f"fold{index}" / key
        baseline_records.append(trial_entry(args.private, folder, receipt))
        validation_rows.extend(receipt["validation"])
    selected = validation_selection(cfg, validation_rows, index)
    costs.extend(row | {"cost_kind": "VALIDATION_PROFILE"} for row in validation_rows)
    files = ("VALIDATION_TRIALS.csv", "META_TRACE.csv", "COSTS.csv")
    for name, rows in zip(files, (validation_rows, traces, costs), strict=True):
        csv(output / name, rows)
    record = {
        "status": "COMPLETE_STAGE_A_FOLD",
        "identity_sha256": identity_sha,
        "fold": index,
        "fit_label_ids": sorted(fit_ids),
        "validation_label_ids": sorted(validation_ids),
        "outer_role_labels_read": 0,
        "target_labels_read": 0,
        "validation_rows": len(validation_rows),
        "meta_rows": len(traces),
        "selection": selected,
        "mapper_trials": records,
        "baseline_trials": baseline_records,
        "files": {
            name: file_hash(output / name)
            for name in (*files, "BASIS_AUDIT.json", "DESCRIPTOR_SCALER.json")
        },
    }
    write_json(marker, record)
    return check_fold_files(args, record, identity_sha)


def finalize_a(args, cfg, identity_sha):
    import pandas as pd

    folds = [
        check_fold_files(
            args,
            read_json(args.public / "stage_a" / f"fold{fold}" / "COMPLETE.json"),
            identity_sha,
        )
        for fold in cfg["folds"]
    ]
    selected = {str(rec["fold"]): rec["selection"] for rec in folds}
    published = ("VALIDATION_TRIALS.csv", "META_TRACE.csv", "COSTS.csv")
    for name in published:
        combined = pd.concat(
            [
                pd.read_csv(
                    args.public / "stage_a" / f"fold{rec['fold']}" / name,
                    float_precision="round_trip",
                )
                for rec in folds
            ],
            ignore_index=True,
        )
        csv(args.public / name, combined)
    decision = stage_a_decision(cfg, selected) | {
        "selected": selected,
        "identity_sha256": identity_sha,
        "mapper_fits": sum(len(rec["mapper_trials"]) for rec in folds),
        "validation_rows": sum(rec["validation_rows"] for rec in folds),
        "mapper_adam_updates": sum(rec["meta_rows"] for rec in folds),
        "differentiable_support_sgd_updates": sum(rec["meta_rows"] for rec in folds)
        * cfg["core_steps"],
        "fold_receipts": {
            str(fold): file_hash(
                args.public / "stage_a" / f"fold{fold}" / "COMPLETE.json"
            )
            for fold in cfg["folds"]
        },
        "files": {name: file_hash(args.public / name) for name in published},
    }
    if (
        decision["mapper_fits"],
        decision["mapper_adam_updates"],
        decision["validation_rows"],
    ) != (96, 3072, 399):
        raise RuntimeError("Stage A exact registered work count mismatch")
    marker = args.public / "STAGE_A_DECISION.json"
    bind(marker, decision)
    if decision["status"] == "STOP_NO_VALIDATION_SIGNAL":
        bind(
            args.public / "FINAL_DECISION.json",
            {
                "status": decision["status"],
                "stage_a_sha256": file_hash(marker),
                "later_stages": {
                    stage: "NOT_RUN_BY_STAGE_A_GATE"
                    for stage in (
                        "seed43_44_training",
                        "outer_scoring",
                        "matched_rate_controls",
                        "window_contributions",
                    )
                },
                "outer_role_labels_read": 0,
                "target_labels_read": 0,
            },
        )
    else:
        bind(
            args.public / "HYPERPARAMETER_LOCK.json",
            {
                "status": "LOCKED_AFTER_STAGE_A",
                "identity_sha256": identity_sha,
                "stage_a_sha256": file_hash(marker),
                "selected": selected,
            },
        )
    print(
        f"{decision['status']}  ({sum(r['passed'] for r in decision['validation_folds'])}/3 folds)",
        flush=True,
    )
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("register", "dev", "outer", "run"), default="run"
    )
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--features", type=Path, default=ROOT / "runtime/drs_peft" / V1)
    parser.add_argument("--reference-root", type=Path, default=REFERENCES)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.fold is not None and args.phase != "dev":
        parser.error("--fold is only supported for independent Stage A development")
    configure()
    with threadpool_limits(limits=1):
        cfg, plan, protocols, source_ids, sealed = build_plan(args)
        seed_all(cfg["selection_seed"])
        plan_path = args.public / "REGISTERED_EPISODES.json"
        bind(plan_path, plan)
        data = load_prefixes("MATR", args.data_root)
        features, receipt = input_features(args, source_ids, data)
        identity = identity_for(args, cfg, plan_path, receipt, data)
        marker = args.public / "RUN_IDENTITY.json"
        bind(marker, identity)
        identity_sha = file_hash(marker)
        if args.phase == "register":
            print(f"Registered source-only run {identity_sha}", flush=True)
            return
        lora_parent = read_json(ROOT / cfg["lora_parent_config"])
        if args.phase in {"dev", "run"}:
            for fold in plan["folds"]:
                if args.fold is None or args.fold == fold["fold"]:
                    run_stage_a_fold(
                        args,
                        cfg,
                        fold,
                        data,
                        features,
                        sealed,
                        identity_sha,
                        lora_parent,
                    )
            if args.fold is not None:
                return
        decision = finalize_a(args, cfg, identity_sha)
        if args.phase in {"outer", "run"} and decision["status"] == "PASS_STAGE_A":
            from scripts.drs_peft.spectral_objective_outer import run_locked_stages

            run_locked_stages(
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
            )


if __name__ == "__main__":
    main()
