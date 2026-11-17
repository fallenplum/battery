"""Train shared no-policy references and build same-point fit dictionaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from threadpoolctl import threadpool_limits

from models.drs_peft.onesided import NoPolicyBatteryTSFM
from models.drs_peft.peft import (
    SubspaceLinear,
    install_peft,
    trainable_parameter_report,
)
from scripts.drs_peft.data import ROOT, file_hash, load_dataset, write_json
from scripts.drs_peft.e123_protocol import fold_episode_view, load_protocol
from scripts.drs_peft.episodes import load_registration
from scripts.drs_peft.gradient_audit import _covariances, _top, state_fingerprint
from scripts.drs_peft.onesided_basis import (
    build_dictionary_bundle,
    collect_same_point_gradients,
    physical_cell_weights,
    source_layer_bases,
    transfer_spectrum,
    write_dictionary_artifacts,
)
from scripts.drs_peft.run_p2 import _publish_unit
from scripts.drs_peft.train import (
    bind,
    cell_loss,
    configure,
    early_batch,
    frozen_parameter_fingerprint,
    load_selected,
    predict,
    read_json,
    restore_rng,
    rng_state,
    save_torch,
    seed_all,
    selected_state,
    source_snapshot,
    tensor_fingerprint,
    train_source_unit,
)

DEFAULT_CONFIG = ROOT / "configs/drs_peft/onesided_fewshot_v1/CONFIG.json"
DEFAULT_PRIVATE = ROOT / "runtime/drs_peft/onesided_fewshot_v1/references"
DEFAULT_PUBLIC = ROOT / "results/drs_peft/onesided_fewshot_v1/references"
EXTRA_SOURCES = (
    "models/drs_peft/onesided.py",
    "scripts/drs_peft/onesided_basis.py",
    "scripts/drs_peft/run_onesided_reference.py",
    "scripts/drs_peft/e123_protocol.py",
)


class NoPolicyInputs:
    """Expose the existing physical VIQ and legal prefix anchor without metadata."""

    def fit(self, data, rows):
        rows = np.asarray(rows, dtype=int)
        if not len(rows) or len(rows) != len(set(rows.tolist())):
            raise ValueError("Reference fit roles need unique physical cells")
        self.training_ids = [data.ids[index] for index in rows]
        return self

    def transform(self, data):
        return {
            "x": data.x,
            "point_mask": data.point_mask,
            "cycle_ids": data.cycle_ids,
            "anchor": data.anchor,
            "policy": np.zeros((len(data.ids), 0), dtype=np.float32),
            "policy_mask": np.zeros((len(data.ids), 0), dtype=bool),
        }

    def to_dict(self):
        return {
            "training_ids": self.training_ids,
            "input_scope": "VIQ+earlySOH / no-policy",
            "learned_input_normalization_parameters": 0,
            "canonical_x_units": [
                "V",
                "signed_I/declared_nominal_Ah",
                "Q/declared_nominal_Ah",
                "true_prefix_drift/declared_nominal_Ah",
            ],
            "nominal_capacity_source": "declared_cell_metadata; no_full_lifetime_capacity_maximum",
            "drift_reference": "pointwise_median_Q_of_qualified_prefix_cycles1..10",
            "early_anchor_source": "registered_prefix_only_anchor_derived_before_cycle101",
            "x_is_z_scored": False,
            "policy_dimension": 0,
            "policy_values_zero_from_initialization": True,
            "policy_masks_false_from_initialization": True,
            "native_interface": "twelve_descriptors_per_prefix_cycle",
        }


def reference_folder(root, fold_id, seed):
    label = "final" if fold_id == "final" else f"fold{int(fold_id)}"
    return Path(root) / label / f"seed{int(seed)}"


def external_lock(root, path):
    identity = os.path.normcase(str(Path(path).resolve()))
    folder = Path(root) / "_locks"
    folder.mkdir(parents=True, exist_ok=True)
    return FileLock(
        str(folder / (hashlib.sha256(identity.encode()).hexdigest() + ".lock"))
    )


def publish_small_artifacts(private, public):
    public.mkdir(parents=True, exist_ok=True)
    for path in private.iterdir():
        if path.is_file() and path.suffix in (".json", ".csv"):
            target = public / path.name
            if target.exists() and file_hash(target) != file_hash(path):
                raise RuntimeError("A published reference artifact changed")
            target.write_bytes(path.read_bytes())


def validate_config(config):
    reference = config["reference"]
    if (
        reference["bridge"] != "native_multivariate"
        or reference["policy_dim"] != 0
        or reference["rank"] != 4
    ):
        raise ValueError(
            "The registered source must use native, policy-dimension-zero rank4"
        )
    if (
        reference["batch_size"] != 8
        or reference["weight_decay"] != 0.01
        or reference["gradient_clip"] != 1.0
    ):
        raise ValueError(
            "Reference optimizer settings differ from the shared source trainer"
        )
    if reference["existing_policy_checkpoint_reuse"]:
        raise ValueError(
            "Policy-conditioned reference weights cannot qualify for this study"
        )


def training_config(config, stage):
    legacy = read_json(ROOT / "configs/drs_peft/execution_v1.json")
    legacy["study_id"] = config["study_id"]
    legacy["training"]["learning_rate"] = config["reference"]["lr"][stage]
    legacy["training"]["batch_size"] = 8
    return legacy


def warmup_factory(backbone_dir, seed):
    def build():
        seed_all(seed)
        model = NoPolicyBatteryTSFM(
            backbone_dir, 0, "native_multivariate", pretrained=True
        )
        model.official_backbone_identity["actual_initial_tensor_sha256"] = (
            tensor_fingerprint(model.backbone)
        )
        return model

    return build


def static_factory(backbone_dir, seed, warmup, source_bases):
    def build():
        model = warmup_factory(backbone_dir, seed)()
        load_selected(model, warmup["model"])
        install_peft(
            model.backbone,
            "grad_core",
            rank=4,
            alpha=4,
            dropout=0.0,
            seed=seed,
            bases=source_bases,
        )
        model.requires_grad_(False)
        for layer in model.backbone.modules():
            if isinstance(layer, SubspaceLinear):
                layer.core.requires_grad_(True)
        if (
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            != 192
        ):
            raise RuntimeError("Static reference fitting must train 192 core entries")
        return model

    return build


def checked_checkpoint(folder):
    folder = Path(folder)
    receipt = read_json(folder / "SOURCE_DEVELOPMENT_COMPLETE.json")
    if (
        receipt["status"] != "SOURCE_DEVELOPMENT_COMPLETE"
        or receipt["target_query_labels_used"] != 0
    ):
        raise RuntimeError("The reference source fit is incomplete or exposed targets")
    checkpoint = folder / "CHECKPOINT_SELECTED.pt"
    if file_hash(checkpoint) != receipt["selected_checkpoint_sha256"]:
        raise RuntimeError("The source reference checkpoint changed")
    run = read_json(folder / "RUN_CONFIG.json")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["identity"] != run:
        raise RuntimeError("Reference checkpoint and execution identity differ")
    return saved, checkpoint


def load_reference_model(reference_root, backbone_dir, fold_id, seed, device="cpu"):
    folder = reference_folder(reference_root, fold_id, seed)
    complete = read_json(folder / "REFERENCE_COMPLETE.json")
    warmup, warmup_path = checked_checkpoint(folder / "warmup")
    selected, selected_path = checked_checkpoint(folder / "static")
    bases = torch.load(
        folder / "bases/SOURCE_BASES.pt", map_location="cpu", weights_only=False
    )
    if complete["warmup_checkpoint_sha256"] != file_hash(warmup_path) or complete[
        "reference_checkpoint_sha256"
    ] != file_hash(selected_path):
        raise RuntimeError(
            "The reference completion marker points to another source model"
        )
    model = static_factory(backbone_dir, seed, warmup, bases["bases"])()
    load_selected(model, selected["model"])
    if state_fingerprint(model) != complete["reference_state_sha256"]:
        raise RuntimeError("The reconstructed no-policy f0 state differs")
    return model.to(device).eval()


def input_scope_audit(model, inputs, rows, device):
    model = model.to(device).eval()
    batch = early_batch(inputs, rows[:2], device)
    query = torch.tensor([100.0, 150.0, 500.0, 1500.0, 5000.0], device=device)
    altered = dict(batch)
    altered["policy"] = torch.full((len(batch["x"]), 28), 987.0, device=device)
    altered["policy_mask"] = torch.ones(
        (len(batch["x"]), 28), dtype=torch.bool, device=device
    )
    with torch.no_grad():
        reference = model(batch, query)
        perturbed = model(altered, query)
    if not torch.equal(reference, perturbed):
        raise RuntimeError("No-policy f0 depends on policy values or masks")
    anchor_error = float((reference[:, 0] - batch["anchor"]).abs().max().cpu())
    if anchor_error > 1e-7:
        raise RuntimeError("The legal t100 anchor changed")
    return {
        "status": "PASSED",
        "input_scope": "VIQ+earlySOH / no-policy",
        "policy_dimension": 0,
        "policy_perturbation_max_prediction_change": 0.0,
        "anchor_max_absolute_error": anchor_error,
        "prefix_rows": len(rows[:2]),
        "future_labels_used": 0,
    }


def build_source_bases(
    args, config, data, inputs, fit_ids, folder, public_folder, seed
):
    directory = folder / "bases"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "SOURCE_BASES_COMPLETE.json"
    warmup, checkpoint = checked_checkpoint(folder / "warmup")
    expected = {
        "fit_ids": fit_ids,
        "warmup_checkpoint_sha256": file_hash(checkpoint),
        "config_sha256": file_hash(args.config),
    }
    if marker.exists():
        receipt = read_json(marker)
        if receipt["identity"] != expected:
            raise RuntimeError("The no-policy source projection basis changed identity")
        for record in receipt["files"]:
            if file_hash(directory / record["path"]) != record["sha256"]:
                raise RuntimeError("A no-policy projection basis artifact changed")
        publish_small_artifacts(directory, public_folder / "bases")
        return torch.load(
            directory / "SOURCE_BASES.pt", map_location="cpu", weights_only=False
        )["bases"]
    model = warmup_factory(args.backbone_dir, seed)().to(args.device)
    load_selected(model, warmup["model"])
    gradients, gradient_identity = collect_same_point_gradients(
        model,
        data,
        inputs,
        fit_ids,
        checkpoint,
        directory / "warmup_gradients",
        seed,
        checkpoint_stage="no_policy_warmup",
    )
    weights = physical_cell_weights(
        data.protocol_ids[data.rows(fit_ids)], config["reference"]["basis_weighting"]
    )
    bases, records = {}, []
    for name, values in gradients.items():
        left, right, _, _ = _covariances(np.asarray(values, dtype=np.float64), weights)
        u, left_values = _top(left, 4)
        v, right_values = _top(right, 4)
        bases[name] = (
            torch.from_numpy(u.astype(np.float32)),
            torch.from_numpy(v.astype(np.float32)),
        )
        for side, eigenvalues in (("left", left_values), ("right", right_values)):
            records.extend(
                {
                    "layer": name,
                    "side": side,
                    "direction": index + 1,
                    "eigenvalue": float(value),
                }
                for index, value in enumerate(eigenvalues)
            )
    save_torch(
        directory / "SOURCE_BASES.pt",
        {"identity": expected, "bases": bases, "gradient_identity": gradient_identity},
    )
    pd.DataFrame(records).to_csv(directory / "SOURCE_BASIS_SPECTRA.csv", index=False)
    receipt = {
        "status": "COMPLETE_NO_POLICY_WARMUP_SOURCE_BASES",
        "identity": expected,
        "gradient_checkpoint": "no_policy_warmup",
        "new_gradient_evaluations": len(fit_ids),
        "endpoint_replay_evaluations": 2,
        "optimizer_updates": 0,
        "files": [
            {"path": name, "sha256": file_hash(directory / name)}
            for name in ("SOURCE_BASES.pt", "SOURCE_BASIS_SPECTRA.csv")
        ],
    }
    write_json(marker, receipt)
    publish_small_artifacts(directory, public_folder / "bases")
    del model, gradients
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return bases


def train_fixed_budget_unit(
    args,
    config,
    data,
    inputs,
    scaler,
    fit_ids,
    destination,
    factory,
    seed,
    stage,
    epochs,
    parent_checkpoint=None,
):
    """Refit all source cells to a CV-fixed last epoch without validation labels."""
    destination.mkdir(parents=True, exist_ok=True)
    if epochs < 1 or len(fit_ids) != len(set(fit_ids)) or not fit_ids:
        raise ValueError("Final source refits need unique cells and positive epochs")
    source_rows = data.rows(fit_ids)
    model = factory().to(args.device)
    frozen = frozen_parameter_fingerprint(model)
    identity = {
        "stage": "ONESIDED_FINAL_REFERENCE_" + stage,
        "method": "ONESIDED_" + stage,
        "seed": seed,
        "epochs": int(epochs),
        "batch_size": 8,
        "training_ids": fit_ids,
        "validation_ids": [],
        "learning_rate": config["reference"]["lr"][stage],
        "adapter_learning_rate": config["reference"]["lr"][stage],
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "expected_updates": int(epochs) * math.ceil(len(source_rows) / 8),
        "input_scope": "VIQ+earlySOH / no-policy",
        "config_sha256": file_hash(args.config),
        "source_snapshot": source_snapshot(args.config, extra_sources=EXTRA_SOURCES),
        "source_commit": os.environ.get("BATTERY_SOURCE_COMMIT"),
        "initial_checkpoint_sha256": file_hash(parent_checkpoint)
        if parent_checkpoint
        else None,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "target_query_labels_used": 0,
        "epoch_selection": "last_epoch_at_per_seed_median_CV_budget_no_validation",
    }
    for name, value in (
        ("RUN_CONFIG.json", identity),
        ("SOURCE_SNAPSHOT.json", {"files": identity["source_snapshot"]}),
        ("SOURCE_SCALER.json", scaler.to_dict()),
        ("BACKBONE_IDENTITY.json", model.official_backbone_identity),
        (
            "TRAINABLE_PARAMETER_REPORT.json",
            dict(
                model.parameter_report(), peft_detail=trainable_parameter_report(model)
            ),
        ),
    ):
        bind(destination / name, value)
    marker = destination / "SOURCE_DEVELOPMENT_COMPLETE.json"
    if marker.exists():
        checked_checkpoint(destination)
        return read_json(marker)
    common, adapters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (adapters if name.startswith("bridge.backbone.module.") else common).append(
                parameter
            )
    groups = [{"params": common, "lr": config["reference"]["lr"][stage]}]
    if adapters:
        groups.append({"params": adapters, "lr": config["reference"]["lr"][stage]})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    state_path = destination / "TRAIN_STATE.pt"
    history, updates, start_epoch, elapsed = [], 0, 0, 0.0
    if state_path.exists():
        saved = torch.load(state_path, map_location=args.device, weights_only=False)
        if saved["identity"] != identity:
            raise RuntimeError("Final source refit resume identity changed")
        load_selected(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["random"])
        history, updates, start_epoch, elapsed = (
            saved[key] for key in ("history", "updates", "next_epoch", "elapsed")
        )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        order = np.random.default_rng(seed + epoch * 1009).permutation(source_rows)
        total = 0.0
        for start in range(0, len(order), 8):
            rows = order[start : start + 8]
            positions = np.flatnonzero(data.observed[rows].any(0))
            query = torch.as_tensor(data.query[positions], device=args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(early_batch(inputs, rows, args.device), query)
            loss = cell_loss(
                prediction,
                torch.as_tensor(data.targets[rows][:, positions], device=args.device),
                torch.as_tensor(data.observed[rows][:, positions], device=args.device),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Final source refit loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            updates += 1
            total += float(loss.detach().cpu()) * len(rows)
        history.append(
            {
                "epoch": epoch + 1,
                "train_mae": total / len(source_rows),
                "updates": updates,
            }
        )
        saved = {
            "model": selected_state(model),
            "optimizer": optimizer.state_dict(),
            "random": rng_state(),
            "history": history,
            "updates": updates,
            "next_epoch": epoch + 1,
            "elapsed": elapsed + time.perf_counter() - started,
            "identity": identity,
        }
        save_torch(state_path, saved)
        if epoch == start_epoch or (epoch + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "event": "FINAL_SOURCE_REFERENCE_TRAINING",
                        "stage": stage,
                        "seed": seed,
                        **history[-1],
                    }
                ),
                flush=True,
            )
        if args.stop_after is not None and epoch + 1 >= args.stop_after:
            return {"status": "ENGINEERING_INTERRUPTED", "updates": updates}
    if (
        updates != identity["expected_updates"]
        or frozen_parameter_fingerprint(model) != frozen
    ):
        raise RuntimeError("Final refit budget or frozen source state changed")
    selected = {
        "model": selected_state(model),
        "identity": identity,
        "best_epoch": epochs,
        "validation": None,
        "backbone_id": model.official_backbone_identity,
    }
    save_torch(destination / "CHECKPOINT_SELECTED.pt", selected)
    write_json(destination / "TRAIN_LOG.json", history)
    write_json(
        destination / "CHECKPOINT_SELECTION.json",
        {"rule": identity["epoch_selection"], "best_epoch": epochs, "validation": None},
    )
    reconstructed = factory().to(args.device)
    load_selected(
        reconstructed,
        torch.load(
            destination / "CHECKPOINT_SELECTED.pt",
            map_location="cpu",
            weights_only=False,
        )["model"],
    )
    sample = source_rows[:2]
    first, second = (
        predict(model, inputs, sample, data.query, args.device),
        predict(reconstructed, inputs, sample, data.query, args.device),
    )
    if not np.array_equal(first, second):
        raise RuntimeError("Final source checkpoint serialized replay failed")
    receipt = {
        "status": "SOURCE_DEVELOPMENT_COMPLETE",
        "stage": identity["stage"],
        "method": identity["method"],
        "seed": seed,
        "epochs": epochs,
        "updates": updates,
        "best_epoch": epochs,
        "validation": None,
        "selection_rule": identity["epoch_selection"],
        "trainable_parameters": identity["trainable_parameters"],
        "elapsed_seconds": saved["elapsed"],
        "selected_checkpoint_sha256": file_hash(destination / "CHECKPOINT_SELECTED.pt"),
        "target_query_labels_used": 0,
        "prediction_replay_max_absolute_error": 0.0,
        "frozen_parameters_unchanged": True,
        "frozen_parameter_sha256": frozen,
        "peak_gpu_memory_bytes": {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        }
        if args.device.startswith("cuda")
        else {"allocated": 0, "reserved": 0},
        "files": [
            {"path": path.name, "sha256": file_hash(path)}
            for path in sorted(destination.iterdir())
            if path.is_file() and path.name != marker.name
        ],
    }
    write_json(marker, receipt)
    del model, reconstructed, optimizer
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return receipt


def final_epoch_budgets(reference_root, seed):
    budgets = {}
    for stage in ("warmup", "static"):
        records = [
            read_json(
                reference_folder(reference_root, fold, seed)
                / stage
                / "SOURCE_DEVELOPMENT_COMPLETE.json"
            )
            for fold in (0, 1, 2)
        ]
        if any(
            record["status"] != "SOURCE_DEVELOPMENT_COMPLETE"
            or record["seed"] != seed
            or record["target_query_labels_used"] != 0
            for record in records
        ):
            raise RuntimeError(
                "Final source budgets require three completed source-only CV fits"
            )
        epochs = [record["best_epoch"] for record in records]
        budgets[stage] = int(np.median(epochs))
        if budgets[stage] < 1:
            raise ValueError("Source CV produced no legal final reference epoch budget")
    return budgets


def write_reference_index(private_root, public_root):
    records, audits = [], []
    for path in sorted(Path(private_root).glob("*/seed*/REFERENCE_COMPLETE.json")):
        reference = read_json(path)
        warmup = read_json(path.parent / "warmup/SOURCE_DEVELOPMENT_COMPLETE.json")
        static = read_json(path.parent / "static/SOURCE_DEVELOPMENT_COMPLETE.json")
        records.append(
            {
                "fold_id": reference["fold_id"],
                "seed": reference["seed"],
                "input_scope": reference["input_scope"],
                "policy_dimension": 0,
                "source_fit_cells": len(reference["fit_ids"]),
                "source_inner_validation_cells": len(reference["inner_validation_ids"]),
                "warmup_selected_epoch": warmup["best_epoch"],
                "static_selected_epoch": static["best_epoch"],
                "warmup_trainable_parameters": warmup["trainable_parameters"],
                "static_trainable_parameters": static["trainable_parameters"],
                "source_optimizer_updates": reference["reference_training_updates"],
                "reference_checkpoint_sha256": reference["reference_checkpoint_sha256"],
                "reference_state_sha256": reference["reference_state_sha256"],
                "target_query_labels_used": 0,
            }
        )
        audits.append(
            {
                "fold_id": reference["fold_id"],
                "seed": reference["seed"],
                **read_json(path.parent / "INPUT_SCOPE_AUDIT.json"),
            }
        )
    for root in (Path(private_root), Path(public_root)):
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(root / "REFERENCE_MODELS.csv", index=False)
        write_json(
            root / "INPUT_SCOPE_AUDIT.json",
            {
                "status": "PASSED" if records else "NO_COMPLETED_REFERENCES",
                "completed_references": len(records),
                "references": audits,
                "target_query_labels_used": 0,
            },
        )


def train_reference(args, config, data, fit_ids, episodes, fold_id, seed, epochs):
    folder, public_folder = (
        reference_folder(args.private_root, fold_id, seed),
        reference_folder(args.public_root, fold_id, seed),
    )
    folder.mkdir(parents=True, exist_ok=True)
    scaler = NoPolicyInputs().fit(data, data.rows(fit_ids))
    inputs = scaler.transform(data)
    with external_lock(args.private_root, folder):
        for stage in ("warmup", "static"):
            parent_checkpoint = None
            if stage == "warmup":
                factory = warmup_factory(args.backbone_dir, seed)
            else:
                warmup, parent_checkpoint = checked_checkpoint(folder / "warmup")
                bases = build_source_bases(
                    args, config, data, inputs, fit_ids, folder, public_folder, seed
                )
                factory = static_factory(args.backbone_dir, seed, warmup, bases)
            destination = folder / stage
            if fold_id == "final":
                receipt = train_fixed_budget_unit(
                    args,
                    config,
                    data,
                    inputs,
                    scaler,
                    fit_ids,
                    destination,
                    factory,
                    seed,
                    stage,
                    epochs[stage],
                    parent_checkpoint,
                )
            else:
                receipt = train_source_unit(
                    data,
                    inputs,
                    scaler,
                    episodes,
                    training_config(config, stage),
                    args.config,
                    args.backbone_dir,
                    destination,
                    "native_multivariate",
                    "ONESIDED_REFERENCE_" + stage,
                    seed,
                    epochs[stage],
                    args.device,
                    rank=4,
                    adapter_lr=config["reference"]["lr"][stage],
                    stop_after=args.stop_after,
                    stage="ONESIDED_CV_REFERENCE_" + stage,
                    model_factory=factory,
                    initial_checkpoint_sha256=file_hash(parent_checkpoint)
                    if parent_checkpoint
                    else None,
                    extra_sources=EXTRA_SOURCES,
                )
            if receipt["status"] == "ENGINEERING_INTERRUPTED":
                print(
                    json.dumps(
                        {
                            "event": "REFERENCE_ENGINEERING_INTERRUPTED",
                            "fold_id": fold_id,
                            "seed": seed,
                            "stage": stage,
                            **receipt,
                        }
                    ),
                    flush=True,
                )
                return False
            _publish_unit(destination, public_folder / stage)
        selected, checkpoint_path = checked_checkpoint(folder / "static")
        warmup, warmup_path = checked_checkpoint(folder / "warmup")
        model = static_factory(args.backbone_dir, seed, warmup, bases)().to(args.device)
        load_selected(model, selected["model"])
        audit = input_scope_audit(model, inputs, data.rows(fit_ids), args.device)
        bind(folder / "INPUT_SCOPE_AUDIT.json", audit)
        bind(public_folder / "INPUT_SCOPE_AUDIT.json", audit)
        receipt = {
            "status": "COMPLETE_NO_POLICY_REFERENCE",
            "fold_id": fold_id,
            "seed": seed,
            "fit_ids": fit_ids,
            "inner_validation_ids": episodes["source_cell_roles"][
                "source_meta_validation"
            ]
            if fold_id != "final"
            else [],
            "input_scope": "VIQ+earlySOH / no-policy",
            "policy_dimension": 0,
            "warmup_checkpoint_sha256": file_hash(warmup_path),
            "reference_checkpoint_sha256": file_hash(checkpoint_path),
            "reference_state_sha256": state_fingerprint(model),
            "reference_epochs": epochs,
            "reference_training_updates": sum(
                read_json(folder / stage / "SOURCE_DEVELOPMENT_COMPLETE.json")[
                    "updates"
                ]
                for stage in ("warmup", "static")
            ),
            "config_sha256": file_hash(args.config),
            "source_commit": os.environ.get("BATTERY_SOURCE_COMMIT"),
            "final_target_query_labels_used": 0,
        }
        bind(folder / "REFERENCE_COMPLETE.json", receipt)
        bind(public_folder / "REFERENCE_COMPLETE.json", receipt)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "NO_POLICY_REFERENCE_COMPLETE",
                    "fold_id": fold_id,
                    "seed": seed,
                    "updates": receipt["reference_training_updates"],
                }
            ),
            flush=True,
        )
        write_reference_index(args.private_root, args.public_root)
    return True


def build_reference_dictionary(args, config, data, fold_id, seed):
    folder, public_folder = (
        reference_folder(args.private_root, fold_id, seed),
        reference_folder(args.public_root, fold_id, seed),
    )
    reference = read_json(folder / "REFERENCE_COMPLETE.json")
    fit_ids = reference["fit_ids"]
    scaler = NoPolicyInputs().fit(data, data.rows(fit_ids))
    inputs = scaler.transform(data)
    directory = folder / "geometry"
    with external_lock(args.private_root, directory):
        marker = directory / "DICTIONARIES_COMPLETE.json"
        if marker.exists():
            receipt = read_json(marker)
            if (
                receipt["identity"]["reference_checkpoint_sha256"]
                != reference["reference_checkpoint_sha256"]
            ):
                raise RuntimeError(
                    "A cached dictionary belongs to another no-policy f0"
                )
            if (
                receipt["implementation_sha256"]
                != file_hash(ROOT / "scripts/drs_peft/onesided_basis.py")
                or receipt["requested_dimension"]
                != config["geometry"]["requested_dimension"]
                or receipt["relative_rank_tolerance"]
                != config["geometry"]["gram_eigenvalue_relative_tolerance"]
            ):
                raise RuntimeError("A cached dictionary uses another construction")
            for record in receipt["files"]:
                if file_hash(directory / record["path"]) != record["sha256"]:
                    raise RuntimeError("A completed reference dictionary changed")
            if (
                fold_id != "final"
                and seed in config["D3"]["training_seeds"]
                and not (directory / "TRANSFER_SPECTRUM.json").exists()
            ):
                bundle = torch.load(
                    directory / "DICTIONARIES.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                transfer = transfer_spectrum(
                    bundle,
                    seed=seed,
                    maximum_dimension=config["D3"]["requested_dimension"],
                )
                save_torch(directory / "TRANSFER_SPECTRUM.pt", transfer)
                write_json(
                    directory / "TRANSFER_SPECTRUM.json",
                    {
                        key: value
                        for key, value in transfer.items()
                        if key not in ("dictionaries", "rotations")
                    },
                )
            publish_small_artifacts(directory, public_folder / "geometry")
            return
        model = load_reference_model(
            args.private_root, args.backbone_dir, fold_id, seed, args.device
        )
        gradients, identity = collect_same_point_gradients(
            model,
            data,
            inputs,
            fit_ids,
            folder / "static/CHECKPOINT_SELECTED.pt",
            directory / "fit_gradients",
            seed,
        )
        bundle, blocks, spectra = build_dictionary_bundle(
            gradients,
            source_layer_bases(model),
            fit_ids,
            data.protocol_ids[data.rows(fit_ids)],
            requested_dimension=config["geometry"]["requested_dimension"],
            weighting=config["reference"]["gradient_weighting"],
            relative_rank_tolerance=config["geometry"][
                "gram_eigenvalue_relative_tolerance"
            ],
            identity=identity,
        )
        if fold_id == "final":
            bundle["arms"] = {"OUT": bundle["arms"]["OUT"]}
            bundle["fit_coordinates"] = {"OUT": bundle["fit_coordinates"]["OUT"]}
            bundle["common_dimension"] = bundle["arms"]["OUT"]["dimension"]
        write_dictionary_artifacts(directory, bundle, blocks, spectra)
        if fold_id != "final" and seed in config["D3"]["training_seeds"]:
            transfer = transfer_spectrum(
                bundle, seed=seed, maximum_dimension=config["D3"]["requested_dimension"]
            )
            save_torch(directory / "TRANSFER_SPECTRUM.pt", transfer)
            write_json(
                directory / "TRANSFER_SPECTRUM.json",
                {
                    key: value
                    for key, value in transfer.items()
                    if key not in ("dictionaries", "rotations")
                },
            )
        publish_small_artifacts(directory, public_folder / "geometry")
        del model, gradients, bundle
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("reference", "basis", "final-reference", "final-basis"),
        required=True,
    )
    parser.add_argument(
        "--fold-ids", type=int, nargs="+", choices=(0, 1, 2), default=[0, 1, 2]
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", choices=(42, 123, 2026), default=[42, 123, 2026]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    configure()
    config = read_json(args.config)
    validate_config(config)
    data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
    parent = load_registration()
    protocol = load_protocol(ROOT / config["folds_file"])
    for seed in args.seeds:
        if args.phase.startswith("final"):
            fit_ids = sorted(
                parent["source_cell_roles"]["source_train"]
                + parent["source_cell_roles"]["source_meta_validation"]
            )
            if len(fit_ids) != 115 or set(fit_ids) & set(
                parent["source_cell_roles"]["target_test"]
            ):
                raise RuntimeError("Final f0 source identities include target cells")
            if args.phase == "final-reference":
                budgets = final_epoch_budgets(args.private_root, seed)
                if not train_reference(
                    args, config, data, fit_ids, None, "final", seed, budgets
                ):
                    return
            else:
                build_reference_dictionary(args, config, data, "final", seed)
            continue
        for fold in protocol["folds"]:
            if fold["fold_id"] not in args.fold_ids:
                continue
            if args.phase == "reference":
                episodes = fold_episode_view(parent, fold)
                if not train_reference(
                    args,
                    config,
                    data,
                    fold["fit_ids"],
                    episodes,
                    fold["fold_id"],
                    seed,
                    config["reference"]["epochs"],
                ):
                    return
            else:
                build_reference_dictionary(args, config, data, fold["fold_id"], seed)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
