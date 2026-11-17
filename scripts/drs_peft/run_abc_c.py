"""Fit-only subspace supplements and held-out gradient diagnostics for C."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from threadpoolctl import threadpool_limits

from models.drs_peft.abc_subspace import (
    ARMS,
    EXPECTED_PARAMETERS,
    MATCHED_PERP_SHAPES,
    MATCHED_RANKS,
    SubspaceProbeTSFM,
)
from models.drs_peft.peft import SubspaceLinear
from scripts.drs_peft.data import file_hash, load_dataset, write_json
from scripts.drs_peft.e123_protocol import load_protocol, validate_against_data
from scripts.drs_peft.episodes import load_registration
from scripts.drs_peft.execute_e123 import mapped_lock_path
from scripts.drs_peft.gradient_audit import (
    _collect,
    _covariances,
    _group_balanced_weights,
    _group_indices,
    _replay_gradients,
    _top,
    state_fingerprint,
)
from scripts.drs_peft.run_e123 import (
    check_initialization,
    checked_bases,
    checked_training_checkpoint,
    matching_unit,
    payload_hash,
    prepare_fold,
    publish_outer,
    static_factory,
    update_diagnostic_rows,
)
from scripts.drs_peft.run_e123 import (
    unit_path as parent_unit_path,
)
from scripts.drs_peft.run_p2 import _publish_unit
from scripts.drs_peft.train import (
    ROOT,
    bind,
    configure,
    coordinate_name,
    early_batch,
    evaluate_arrays,
    load_selected,
    predict,
    read_json,
    save_torch,
    selected_state,
    train_source_unit,
)

DEFAULT_CONFIG = ROOT / "configs/drs_peft/abc_v1.json"
EXTRA_SOURCES = (
    "models/drs_peft/abc_subspace.py",
    "scripts/drs_peft/run_abc_c.py",
    "scripts/drs_peft/run_e123.py",
    "scripts/drs_peft/e123_protocol.py",
    "scripts/drs_peft/execute_e123.py",
    "configs/drs_peft/e123_folds_v1.json",
    "configs/drs_peft/execution_v1.json",
)


def external_lock(private_root, path):
    folder = Path(private_root) / "_locks"
    folder.mkdir(parents=True, exist_ok=True)
    return FileLock(str(mapped_lock_path(path, folder)))


def bind_shared(args, path, value):
    with external_lock(args.private_root, path):
        bind(path, value)


def unit_path(root, unit):
    return Path(root) / f"fold{unit['fold_id']}" / f"seed{unit['seed']}" / unit["arm"]


def basis_path(root, fold_id, seed):
    return Path(root) / f"fold{fold_id}" / f"seed{seed}" / "bases"


def validate_registration(registration):
    config = registration["C"]
    if config["arms"] != list(ARMS):
        raise ValueError("All five C arms must remain registered")
    if config["matched_ranks"] != list(MATCHED_RANKS):
        raise ValueError("The shared effective budget fixes the six pair ranks")
    if config["matched_perp_shapes"] != [list(shape) for shape in MATCHED_PERP_SHAPES]:
        raise ValueError("Residual allocations differ from the capacity control")
    if config["small_perp_shape"] != [3, 3]:
        raise ValueError(
            "The direct-addition probe requires nine residual coefficients"
        )
    if config["trainable_parameters"] != EXPECTED_PARAMETERS:
        raise ValueError("C parameter budgets changed")
    if not config["freeze_bridge_head"] or not config["freeze_source_C0"]:
        raise ValueError("The C shared source function must stay frozen")
    if config["evaluation_modes"] != ["trained_ball", "matched_norm"]:
        raise ValueError("Both physical update evaluations are registered")
    return config


def build_plan(registration, protocol):
    config = validate_registration(registration)
    units = [
        {"fold_id": fold["fold_id"], "seed": seed, "arm": arm}
        for fold in protocol["folds"]
        for seed in registration["training_seeds"]
        for arm in ARMS
    ]
    updates = sum(
        config["epochs"]
        * math.ceil(len(fold["fit_ids"]) / 8)
        * len(ARMS)
        * len(registration["training_seeds"])
        for fold in protocol["folds"]
    )
    result = {
        "study_id": registration["study_id"],
        "experiment": "C",
        "units": units,
        "expected_training_units": len(units),
        "expected_optimizer_updates": updates,
        "expected_outer_evaluations": len(units) * len(config["evaluation_modes"]),
        "trainable_parameters": EXPECTED_PARAMETERS,
        "protocol_sha256": protocol["protocol_sha256"],
        "final_target_query_labels_used": 0,
        "nested_addition_capacity_difference": {"baseline": 192, "supplemented": 300},
        "matched_comparison": "capacity_reallocation_among_distinct_fit_only_spaces",
    }
    result["plan_sha256"] = payload_hash(result)
    return result


def _projection_complement(covariance, existing, count):
    """Find fit-covariance directions orthogonal to the actual saved frame."""
    existing = np.asarray(existing, dtype=np.float64)
    projector = np.eye(len(covariance)) - existing @ np.linalg.solve(
        existing.T @ existing, existing.T
    )
    projected = projector @ covariance @ projector
    vectors, eigenvalues = _top(projected, count)
    vectors = projector @ vectors
    vectors, triangular = np.linalg.qr(vectors, mode="reduced")
    vectors *= np.where(np.diag(triangular) < 0, -1.0, 1.0)
    largest = np.argmax(np.abs(vectors), axis=0)
    vectors *= np.where(vectors[largest, np.arange(count)] < 0, -1.0, 1.0)
    return vectors, eigenvalues


def decompose_gradient(gradient, u, v):
    """Retain both one-sided blocks in G - P_U G P_V."""
    gradient, u, v = (np.asarray(value, dtype=np.float64) for value in (gradient, u, v))
    left = u @ (u.T @ gradient)
    right = (gradient @ v) @ v.T
    fixed = u @ (u.T @ gradient @ v) @ v.T
    return {
        "fixed": fixed,
        "left_only": left - fixed,
        "right_only": right - fixed,
        "double_perp": gradient - left - right + fixed,
        "residual": gradient - fixed,
    }


def _parent_artifacts(args, fold, seed):
    parent_args = SimpleNamespace(
        private_root=args.parent_private_root, backbone_dir=args.backbone_dir
    )
    warmup_unit = matching_unit("warmup", fold["fold_id"], seed)
    static_unit = matching_unit("static", fold["fold_id"], seed, "grad_core")
    warmup, warmup_path = checked_training_checkpoint(
        parent_unit_path(args.parent_private_root, warmup_unit)
    )
    static, static_path = checked_training_checkpoint(
        parent_unit_path(args.parent_private_root, static_unit)
    )
    bundle = checked_bases(parent_args, fold["fold_id"], seed, fold["fit_ids"])
    if file_hash(warmup_path) != bundle["warmup_checkpoint_sha256"]:
        raise RuntimeError("C fit bases are bound to another fold warmup")
    return parent_args, warmup, static, static_path, bundle


def _checked_parent_gradients(args, fold, seed, bundle):
    folder = basis_path(args.parent_private_root, fold["fold_id"], seed)
    receipt = read_json(folder / "GRADIENT_CACHE_MANIFEST.json")
    identity = receipt["identity"]
    if (
        receipt["status"] != "COMPLETE_SOURCE_GRADIENT_DIAGNOSTIC"
        or identity["role"] != "E123_FIT"
        or identity["variant"] != "nominal"
        or identity["seed"] != seed
    ):
        raise RuntimeError("C fit gradients differ from the authenticated parent role")
    if (
        identity["ids"] != fold["fit_ids"]
        or identity["outcome_donors"] != fold["fit_ids"]
    ):
        raise RuntimeError("C supplements used gradients from outside fit")
    if identity["model_sha256"] != bundle["source_checkpoint_sha256"]:
        raise RuntimeError("C gradient arrays belong to another source model")
    cache = Path(receipt["folder"])
    if not cache.resolve().is_relative_to((folder / "gradients").resolve()):
        raise RuntimeError("The gradient cache escaped its authenticated parent")
    for record in receipt["files"]:
        if file_hash(cache / record["path"]) != record["sha256"]:
            raise RuntimeError("A parent fit gradient cache changed")
    gradients = {
        name: np.load(cache / record["path"], mmap_mode="r")
        for name, record in receipt["layer_files"].items()
    }
    if set(gradients) != set(bundle["bases"]):
        raise RuntimeError("The C gradient target layer family changed")
    if any(
        values.shape != (len(fold["fit_ids"]), 384, 384)
        for values in gradients.values()
    ):
        raise RuntimeError("C gradients differ from the registered matrix dimensions")
    return gradients, receipt


def layer_pair_index(name):
    match = re.search(r"(?:layers|layer)\.(\d+)\.", name)
    if match is None or not 0 <= int(match.group(1)) < 6:
        raise ValueError("The six transformer Q/V layer pairs must be explicit")
    return int(match.group(1))


def _verify_files(folder, receipt):
    for record in receipt["files"]:
        if file_hash(Path(folder) / record["path"]) != record["sha256"]:
            raise RuntimeError("A completed C artifact changed")


def publish_bases(folder, destination, receipt):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("BASE_SUMMARY.csv", "BASES_COMPLETE.json"):
        source = folder / name
        target = destination / name
        if target.exists() and file_hash(target) != file_hash(source):
            raise RuntimeError("A published C basis receipt changed")
        shutil.copyfile(source, target)
    bind(destination / "BASES_COMPLETE.json", receipt)


def collect_bases(args, registration, data, fold, seed, bundle, static_path):
    folder = basis_path(args.private_root, fold["fold_id"], seed)
    destination = basis_path(args.public_root, fold["fold_id"], seed)
    folder.mkdir(parents=True, exist_ok=True)
    identity = {
        "fold_id": fold["fold_id"],
        "seed": seed,
        "fit_ids": fold["fit_ids"],
        "parent_static_checkpoint_sha256": file_hash(static_path),
        "parent_basis_sha256": file_hash(
            basis_path(args.parent_private_root, fold["fold_id"], seed) / "BASES.pt"
        ),
        "config_sha256": file_hash(args.config),
        "implementation_sha256": file_hash(__file__),
        "outer_labels_used": 0,
        "final_target_query_labels_used": 0,
    }
    with external_lock(args.private_root, folder):
        marker = folder / "BASES_COMPLETE.json"
        if marker.exists():
            receipt = read_json(marker)
            if receipt["identity"] != identity:
                raise RuntimeError("The completed C basis identity changed")
            _verify_files(folder, receipt)
            publish_bases(folder, destination, receipt)
            return
        gradients, parent_receipt = _checked_parent_gradients(args, fold, seed, bundle)
        rows = data.rows(fold["fit_ids"])
        weights = _group_balanced_weights(
            _group_indices(data.protocol_ids[rows]), len(rows)
        )
        bases, summaries = {}, []
        fit_means = {}
        with threadpool_limits(limits=2):
            for name, values in gradients.items():
                index = layer_pair_index(name)
                source_u, source_v = (
                    value.double().numpy() for value in bundle["bases"][name]
                )
                g = np.asarray(values, dtype=np.float64)
                left, right, _, _ = _covariances(g, weights)
                rank = MATCHED_RANKS[index]
                extra_u, _ = _projection_complement(left, source_u, rank - 4)
                extra_v, _ = _projection_complement(right, source_v, rank - 4)
                fixed_u = np.concatenate((source_u, extra_u), axis=1)
                fixed_v = np.concatenate((source_v, extra_v), axis=1)
                gv = g @ source_v
                utg = source_u.T[None] @ g
                left_cross = np.tensordot(weights, gv @ gv.transpose(0, 2, 1), axes=1)
                right_cross = np.tensordot(
                    weights, utg.transpose(0, 2, 1) @ utg, axes=1
                )
                perp_shape = MATCHED_PERP_SHAPES[index]
                perp_u, left_eigenvalues = _projection_complement(
                    left - left_cross, source_u, perp_shape[0]
                )
                perp_v, right_eigenvalues = _projection_complement(
                    right - right_cross, source_v, perp_shape[1]
                )
                bases[name] = {
                    "fixed_u": torch.from_numpy(fixed_u.astype(np.float32)),
                    "fixed_v": torch.from_numpy(fixed_v.astype(np.float32)),
                    "perp_u": torch.from_numpy(perp_u.astype(np.float32)),
                    "perp_v": torch.from_numpy(perp_v.astype(np.float32)),
                    "lora_a": torch.from_numpy(source_v[:, 0].astype(np.float32)),
                }
                mean = np.tensordot(weights, g, axes=1)
                fit_means[name] = mean.astype(np.float32)
                summaries.append(
                    {
                        "layer": name,
                        "pair_index": index,
                        "fixed_rank": rank,
                        "fixed_coefficients": rank * rank,
                        "residual_left_rank": perp_shape[0],
                        "residual_right_rank": perp_shape[1],
                        "residual_coefficients": perp_shape[0] * perp_shape[1],
                        "core_plus_residual_coefficients": 16
                        + perp_shape[0] * perp_shape[1],
                        "gauge_fixed_rank1_coefficients": 767,
                        "left_perp_min_eigenvalue": float(left_eigenvalues[-1]),
                        "right_perp_min_eigenvalue": float(right_eigenvalues[-1]),
                        "left_source_cross_gram_max": float(
                            np.abs(source_u.T @ perp_u).max()
                        ),
                        "right_source_cross_gram_max": float(
                            np.abs(source_v.T @ perp_v).max()
                        ),
                    }
                )
        if sum(record["fixed_coefficients"] for record in summaries) != 9204:
            raise RuntimeError("The matched fixed capacity differs from 9204")
        if (
            sum(record["core_plus_residual_coefficients"] for record in summaries)
            != 9204
        ):
            raise RuntimeError("The residual allocation capacity differs from 9204")
        save_torch(folder / "BASES.pt", {"identity": identity, "bases": bases})
        np.savez_compressed(folder / "FIT_GRADIENT_MEANS.npz", **fit_means)
        pd.DataFrame(summaries).to_csv(folder / "BASE_SUMMARY.csv", index=False)
        receipt = {
            "status": "COMPLETE_FIT_ONLY_C_BASES",
            "identity": identity,
            "parent_gradient_cache_receipt_sha256": file_hash(
                basis_path(args.parent_private_root, fold["fold_id"], seed)
                / "GRADIENT_CACHE_MANIFEST.json"
            ),
            "parent_gradient_model_sha256": parent_receipt["identity"]["model_sha256"],
            "new_gradient_evaluations": 0,
            "optimizer_updates": 0,
            "source_commit": os.environ.get("BATTERY_SOURCE_COMMIT"),
            "files": [
                {"path": name, "sha256": file_hash(folder / name)}
                for name in ("BASES.pt", "FIT_GRADIENT_MEANS.npz", "BASE_SUMMARY.csv")
            ],
        }
        write_json(marker, receipt)
        publish_bases(folder, destination, receipt)
        print(
            json.dumps(
                {
                    "event": "ABC_C_BASES_COMPLETE",
                    "fold_id": fold["fold_id"],
                    "seed": seed,
                }
            ),
            flush=True,
        )


def load_probe_bases(args, fold, seed):
    folder = basis_path(args.private_root, fold["fold_id"], seed)
    receipt = read_json(folder / "BASES_COMPLETE.json")
    _verify_files(folder, receipt)
    if (
        receipt["identity"]["fit_ids"] != fold["fit_ids"]
        or receipt["identity"]["fold_id"] != fold["fold_id"]
        or receipt["identity"]["seed"] != seed
    ):
        raise RuntimeError("C basis provenance differs from fold fit")
    if receipt["identity"]["config_sha256"] != file_hash(args.config):
        raise RuntimeError("C basis registration changed")
    saved = torch.load(folder / "BASES.pt", map_location="cpu", weights_only=False)
    if saved["identity"] != receipt["identity"]:
        raise RuntimeError("C serialized bases differ from their provenance receipt")
    return saved["bases"]


def model_factory(
    args, registration, inputs, unit, parent_args, warmup, static, bundle, probe_bases
):
    static_unit = matching_unit("static", unit["fold_id"], unit["seed"], "grad_core")
    construct = static_factory(parent_args, inputs, static_unit, warmup, bundle)

    def build():
        source = construct()
        load_selected(source, static["model"])
        model = SubspaceProbeTSFM(
            source,
            unit["arm"],
            probe_bases,
            registration["C"]["reference_weight_fraction"],
            registration["C"]["radius"],
        )
        saved = selected_state(model)
        names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not names.issubset(saved):
            raise RuntimeError("The inherited serializer omitted a C parameter")
        return model

    return build, construct


def training_config(registration):
    config = read_json(ROOT / "configs/drs_peft/execution_v1.json")
    config["study_id"] = registration["study_id"]
    config["training"]["learning_rate"] = float(registration["C"]["adapter_lr"])
    config["training"]["batch_size"] = 8
    return config


def train_unit(
    args,
    registration,
    data,
    inputs,
    scaler,
    episodes,
    fold,
    unit,
    factory,
    static_build,
    static,
    static_path,
):
    folder = unit_path(args.private_root, unit)
    folder.mkdir(parents=True, exist_ok=True)
    with external_lock(args.private_root, folder):
        initial = factory()
        initial_identity = {
            "unit": unit,
            "config_sha256": file_hash(args.config),
            "folds_sha256": file_hash(args.folds),
            "fold_payload_sha256": load_protocol(args.folds)["protocol_sha256"],
            "source_C0_checkpoint_sha256": file_hash(static_path),
            "initial_model_sha256": state_fingerprint(initial),
            "initial_head_sha256": state_fingerprint(initial.head),
            "fit_ids": fold["fit_ids"],
            "fit_groups": fold["fit_groups"],
            "inner_validation_ids": fold["inner_validation_ids"],
            "inner_validation_groups": fold["inner_validation_groups"],
            "outer_ids": fold["outer_test_ids"],
            "outer_groups": fold["outer_test_groups"],
            "trainable_parameters": EXPECTED_PARAMETERS[unit["arm"]],
            "outer_labels_used_for_training_or_selection": 0,
            "final_target_query_labels_used": 0,
        }
        del initial
        bind(folder / "ABC_C_UNIT_SPEC.json", initial_identity)
        initial_path = folder / "FIXED_ADAPTER_START_CHECK.json"
        if not initial_path.exists():

            def loaded_static():
                source = static_build()
                load_selected(source, static["model"])
                return source

            bind(
                initial_path,
                check_initialization(
                    factory,
                    loaded_static,
                    inputs,
                    data.rows(fold["fit_ids"]),
                    args.device,
                ),
            )
        receipt = train_source_unit(
            data,
            inputs,
            scaler,
            episodes,
            training_config(registration),
            args.config,
            args.backbone_dir,
            folder,
            "native_multivariate",
            "ABC_C_" + unit["arm"],
            unit["seed"],
            registration["C"]["epochs"],
            args.device,
            rank=4,
            adapter_lr=registration["C"]["adapter_lr"],
            stop_after=args.stop_after,
            stage="ABC_C_FOLD_LOCAL_SOURCE",
            model_factory=factory,
            initial_checkpoint_sha256=file_hash(static_path),
            extra_sources=EXTRA_SOURCES,
        )
        if receipt["status"] == "ENGINEERING_INTERRUPTED":
            print(
                json.dumps(
                    dict(event="ABC_C_ENGINEERING_INTERRUPTED", unit=unit, **receipt)
                ),
                flush=True,
            )
            return False
        expected = registration["C"]["epochs"] * math.ceil(len(fold["fit_ids"]) / 8)
        if (
            receipt["updates"] != expected
            or receipt["trainable_parameters"] != EXPECTED_PARAMETERS[unit["arm"]]
        ):
            raise RuntimeError("C training budget differs from registration")
        _publish_unit(folder, unit_path(args.public_root / "units", unit))
        print(
            json.dumps(
                {
                    "event": "ABC_C_TRAINING_COMPLETE",
                    "unit": unit,
                    "updates": receipt["updates"],
                    "inner_validation": receipt["validation"],
                }
            ),
            flush=True,
        )
        return True


def require_all_training(args, plan):
    registration = read_json(args.config)
    protocol = load_protocol(args.folds)
    folds = {fold["fold_id"]: fold for fold in protocol["folds"]}
    if protocol["protocol_sha256"] != plan["protocol_sha256"]:
        raise RuntimeError("C outer scoring received a different fold payload")
    result = []
    for unit in plan["units"]:
        folder = unit_path(args.private_root, unit)
        receipt = read_json(folder / "SOURCE_DEVELOPMENT_COMPLETE.json")
        _verify_files(folder, receipt)
        mandatory_files = {
            "RUN_CONFIG.json",
            "SOURCE_SNAPSHOT.json",
            "ABC_C_UNIT_SPEC.json",
            "CHECKPOINT_SELECTED.pt",
            "FIXED_ADAPTER_START_CHECK.json",
            "SOURCE_SCALER.json",
            "TRAINABLE_PARAMETER_REPORT.json",
        }
        if not mandatory_files.issubset(
            {record["path"] for record in receipt["files"]}
        ):
            raise RuntimeError("The C source receipt omitted required provenance files")
        checkpoint = folder / "CHECKPOINT_SELECTED.pt"
        if file_hash(checkpoint) != receipt["selected_checkpoint_sha256"]:
            raise RuntimeError("A selected C checkpoint changed")
        run = read_json(folder / "RUN_CONFIG.json")
        spec = read_json(folder / "ABC_C_UNIT_SPEC.json")
        fold = folds[unit["fold_id"]]
        static_unit = matching_unit(
            "static", unit["fold_id"], unit["seed"], "grad_core"
        )
        _, static_path = checked_training_checkpoint(
            parent_unit_path(args.parent_private_root, static_unit)
        )
        expected_spec = {
            "unit": unit,
            "config_sha256": file_hash(args.config),
            "folds_sha256": file_hash(args.folds),
            "fold_payload_sha256": protocol["protocol_sha256"],
            "source_C0_checkpoint_sha256": file_hash(static_path),
            "fit_ids": fold["fit_ids"],
            "fit_groups": fold["fit_groups"],
            "inner_validation_ids": fold["inner_validation_ids"],
            "inner_validation_groups": fold["inner_validation_groups"],
            "outer_ids": fold["outer_test_ids"],
            "outer_groups": fold["outer_test_groups"],
            "trainable_parameters": EXPECTED_PARAMETERS[unit["arm"]],
            "outer_labels_used_for_training_or_selection": 0,
            "final_target_query_labels_used": 0,
        }
        if any(spec.get(key) != value for key, value in expected_spec.items()):
            raise RuntimeError("C unit roles, source start, or registration differ")
        expected_updates = registration["C"]["epochs"] * math.ceil(
            len(fold["fit_ids"]) / 8
        )
        expected_run = {
            "stage": "ABC_C_FOLD_LOCAL_SOURCE",
            "method": "ABC_C_" + unit["arm"],
            "bridge": "native_multivariate",
            "seed": unit["seed"],
            "epochs": registration["C"]["epochs"],
            "batch_size": 8,
            "rank": 4,
            "training_ids": fold["fit_ids"],
            "validation_ids": fold["inner_validation_ids"],
            "expected_updates": expected_updates,
            "target_query_labels_used": 0,
            "initial_checkpoint_sha256": file_hash(static_path),
        }
        if any(run.get(key) != value for key, value in expected_run.items()):
            raise RuntimeError("C source checkpoint method, roles, or budget differ")
        if (
            run["config_sha256"] != file_hash(args.config)
            or run["trainable_parameters"] != EXPECTED_PARAMETERS[unit["arm"]]
        ):
            raise RuntimeError("C source identity differs from the frozen plan")
        mandatory_sources = {
            "scripts/drs_peft/run_abc_c.py",
            "models/drs_peft/abc_subspace.py",
            Path(args.config).relative_to(ROOT).as_posix(),
        }
        if not mandatory_sources.issubset(
            {record["path"] for record in run["source_snapshot"]}
        ):
            raise RuntimeError("The C source snapshot omitted its implementation")
        for record in run["source_snapshot"]:
            if file_hash(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(
                    "A C source implementation file changed after training"
                )
        for key in (
            "stage",
            "method",
            "bridge",
            "seed",
            "epochs",
            "trainable_parameters",
        ):
            if receipt[key] != run[key]:
                raise RuntimeError("C source receipt identity differs from its run")
        if (
            receipt["updates"] != expected_updates
            or receipt["target_query_labels_used"] != 0
        ):
            raise RuntimeError("C source receipt update or label budget differs")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            saved["identity"] != run
            or saved["best_epoch"] != receipt["best_epoch"]
            or saved["validation"] != receipt["validation"]
        ):
            raise RuntimeError(
                "C selected checkpoint identity differs from its source run"
            )
        result.append(
            {"unit": unit, "selected_checkpoint_sha256": file_hash(checkpoint)}
        )
    return result


def score_unit(args, data, inputs, fold, unit, factory, checkpoint_path, mode):
    folder = unit_path(args.private_root / "outer", unit) / mode
    destination = unit_path(args.public_root / "outer", unit) / mode
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "OUTER_COMPLETE.json"
    with external_lock(args.private_root, folder):
        if marker.exists():
            receipt = read_json(marker)
            expected = {
                "unit": unit,
                "evaluation_mode": mode,
                "checkpoint_sha256": file_hash(checkpoint_path),
                "outer_ids": fold["outer_test_ids"],
                "outer_groups": fold["outer_test_groups"],
                "registration_sha256": file_hash(args.config),
                "source_sha256": file_hash(__file__),
                "final_target_query_labels_used": 0,
                "optimizer_updates": 0,
            }
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise RuntimeError("C outer results point to another checkpoint")
            _verify_files(folder, receipt)
            publish_outer(folder, destination, receipt)
            return
        rows = data.rows(fold["outer_test_ids"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = factory().to(args.device)
        load_selected(model, checkpoint["model"])
        model.set_evaluation_mode(mode)
        before = state_fingerprint(model)
        prediction = predict(model, inputs, rows, data.query, args.device)
        reconstructed = factory().to(args.device)
        load_selected(reconstructed, checkpoint["model"])
        reconstructed.set_evaluation_mode(mode)
        replay = predict(reconstructed, inputs, rows, data.query, args.device)
        if not np.array_equal(prediction, replay) or state_fingerprint(model) != before:
            raise RuntimeError("C serialized outer replay failed")
        table, metrics = evaluate_arrays(
            prediction,
            data.targets[rows],
            data.observed[rows],
            data.query,
            fold["outer_test_ids"],
            data.protocol_ids[rows],
            coordinate_name(data),
        )
        table["prediction_scope"] = "SOURCE_OUTER_GROUP_CROSS_VALIDATION"
        table.to_csv(folder / "OUTER_CELL_METRICS.csv", index=False)
        np.savez_compressed(
            folder / "OUTER_PREDICTIONS.npz",
            ids=np.asarray(fold["outer_test_ids"]),
            query=data.query,
            prediction=prediction,
            target=data.targets[rows],
            observed=data.observed[rows],
        )
        write_json(
            folder / "OUTER_METRICS.json",
            dict(unit=unit, evaluation_mode=mode, **metrics),
        )
        diagnostics = []
        for start in range(0, len(rows), 8):
            batch_rows = rows[start : start + 8]
            batch = early_batch(inputs, batch_rows, args.device)
            diagnostics.extend(
                update_diagnostic_rows(
                    model, batch, [data.ids[row] for row in batch_rows]
                )
            )
        pd.DataFrame(diagnostics).to_csv(folder / "UPDATE_DIAGNOSTICS.csv", index=False)
        files = [
            {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in sorted(folder.iterdir())
            if path.is_file() and path.name != marker.name
        ]
        receipt = {
            "status": "COMPLETE_SOURCE_OUTER_EVALUATION",
            "unit": unit,
            "evaluation_mode": mode,
            "checkpoint_sha256": file_hash(checkpoint_path),
            "outer_ids": fold["outer_test_ids"],
            "outer_groups": fold["outer_test_groups"],
            "prediction_replay_max_absolute_error": 0.0,
            "files": files,
            "metrics": metrics,
            "final_target_query_labels_used": 0,
            "optimizer_updates": 0,
            "registration_sha256": file_hash(args.config),
            "source_sha256": file_hash(__file__),
        }
        write_json(marker, receipt)
        publish_outer(folder, destination, receipt)
        print(
            json.dumps(
                {
                    "event": "ABC_C_OUTER_COMPLETE",
                    "unit": unit,
                    "evaluation_mode": mode,
                    "metrics": metrics,
                }
            ),
            flush=True,
        )
        del model, reconstructed, checkpoint
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def _cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.sum(left * right) / denominator) if denominator else None


def heldout_gradient_diagnostic(
    args, data, inputs, fold, seed, static_build, static, bundle
):
    folder = (
        args.private_root
        / "heldout_gradients"
        / f"fold{fold['fold_id']}"
        / f"seed{seed}"
    )
    destination = (
        args.public_root
        / "heldout_gradients"
        / f"fold{fold['fold_id']}"
        / f"seed{seed}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "HELDOUT_GRADIENT_COMPLETE.json"
    with external_lock(args.private_root, folder):
        if marker.exists():
            receipt = read_json(marker)
            _verify_files(folder, receipt)
        else:
            model = static_build().to(args.device)
            load_selected(model, static["model"])
            before = state_fingerprint(model)
            layers = {
                name: layer.base
                for name, layer in model.backbone.named_modules()
                if isinstance(layer, SubspaceLinear)
            }
            rows = data.rows(fold["outer_test_ids"])
            gradients, cache_receipt = _collect(
                model,
                data,
                inputs,
                rows,
                rows,
                layers,
                folder / "raw",
                seed,
                "ABC_C_OUTER_DIAGNOSTIC",
                "nominal",
                args.device,
                before,
            )
            replay = _replay_gradients(
                model,
                data,
                inputs,
                [("ABC_C_OUTER_DIAGNOSTIC", "nominal", rows, rows)],
                {("ABC_C_OUTER_DIAGNOSTIC", "nominal"): gradients},
                layers,
                args.device,
            )
            if state_fingerprint(model) != before:
                raise RuntimeError(
                    "Held-out gradient diagnosis changed the source model"
                )
            means = np.load(
                basis_path(args.private_root, fold["fold_id"], seed)
                / "FIT_GRADIENT_MEANS.npz"
            )
            records = []
            for name, values in gradients.items():
                u, v = (tensor.double().numpy() for tensor in bundle["bases"][name])
                fit_blocks = decompose_gradient(means[name], u, v)
                for index, uid in enumerate(fold["outer_test_ids"]):
                    gradient = np.asarray(values[index], dtype=np.float64)
                    blocks = decompose_gradient(gradient, u, v)
                    total = float(np.square(gradient).sum())
                    energies = {
                        key: float(np.square(value).sum())
                        for key, value in blocks.items()
                    }
                    records.append(
                        {
                            "fold_id": fold["fold_id"],
                            "seed": seed,
                            "cell_uid": uid,
                            "protocol_uid": str(data.protocol_ids[rows[index]]),
                            "layer": name,
                            "gradient_norm": math.sqrt(total),
                            **{
                                key + "_energy_ratio": value / total if total else None
                                for key, value in energies.items()
                            },
                            "residual_to_fit_warmup_residual_cosine": _cosine(
                                blocks["residual"], fit_blocks["residual"]
                            ),
                            "full_to_fit_warmup_mean_cosine": _cosine(
                                gradient, means[name]
                            ),
                            "block_energy_reconstruction_error": abs(
                                sum(
                                    energies[key]
                                    for key in (
                                        "fixed",
                                        "left_only",
                                        "right_only",
                                        "double_perp",
                                    )
                                )
                                - total
                            )
                            / total
                            if total
                            else 0.0,
                        }
                    )
            pd.DataFrame(records).to_csv(
                folder / "HELDOUT_GRADIENT_BLOCKS.csv", index=False
            )
            write_json(folder / "GRADIENT_CACHE_MANIFEST.json", cache_receipt)
            write_json(folder / "GRADIENT_REPLAY.json", replay)
            receipt = {
                "status": "COMPLETE_POST_TRAIN_HELDOUT_GRADIENT_DIAGNOSTIC",
                "fold_id": fold["fold_id"],
                "seed": seed,
                "outer_ids": fold["outer_test_ids"],
                "checkpoint_state_sha256": before,
                "fit_reference_checkpoint": "parent_fold_warmup_at_which_U4_V4_were_constructed",
                "diagnostic_checkpoint": "shared_parent_fold_static_C0",
                "new_outer_gradient_evaluations": len(rows),
                "endpoint_replay_evaluations": 2,
                "optimizer_updates": 0,
                "final_target_query_labels_used": 0,
                "heldout_gradients_used_for_directions_or_selection": 0,
                "registration_sha256": file_hash(args.config),
                "implementation_sha256": file_hash(__file__),
                "files": [
                    {"path": name, "sha256": file_hash(folder / name)}
                    for name in (
                        "HELDOUT_GRADIENT_BLOCKS.csv",
                        "GRADIENT_CACHE_MANIFEST.json",
                        "GRADIENT_REPLAY.json",
                    )
                ],
            }
            write_json(marker, receipt)
            del model, gradients
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        destination.mkdir(parents=True, exist_ok=True)
        for record in receipt["files"]:
            shutil.copyfile(folder / record["path"], destination / record["path"])
        bind(destination / marker.name, receipt)
        print(
            json.dumps(
                {
                    "event": "ABC_C_HELDOUT_GRADIENT_COMPLETE",
                    "fold_id": fold["fold_id"],
                    "seed": seed,
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--folds-file",
        dest="folds",
        type=Path,
        default=ROOT / "configs/drs_peft/e123_folds_v1.json",
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument(
        "--parent-private-root", type=Path, default=ROOT / "runtime/drs_peft/e123_v1"
    )
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/abc_v1/C"
    )
    parser.add_argument(
        "--public-root", type=Path, default=ROOT / "results/drs_peft/abc_v1/C"
    )
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("bases", "train", "score", "gradients"), required=True
    )
    parser.add_argument(
        "--fold-ids", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2]
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=(42, 123, 2026), default=[42, 123, 2026]
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure()
    registration, protocol = read_json(args.config), load_protocol(args.folds)
    plan = build_plan(registration, protocol)
    if args.dry_run:
        print(json.dumps(plan, indent=2), flush=True)
        return
    bind_shared(
        args,
        args.public_root / "C_PLAN.json",
        dict(
            plan,
            config_sha256=file_hash(args.config),
            folds_sha256=file_hash(args.folds),
        ),
    )
    data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
    parent_episodes = load_registration()
    bind_shared(
        args,
        args.public_root / "DATA_PROTOCOL_VALIDATION.json",
        validate_against_data(protocol, data, parent_episodes),
    )
    if args.phase in ("score", "gradients"):
        completed = require_all_training(args, plan)
        bind_shared(
            args,
            args.public_root / "OUTER_EVALUATION_OPENED.json",
            {
                "plan_sha256": plan["plan_sha256"],
                "source_units": completed,
                "final_target_query_labels_used": 0,
            },
        )
    for fold in protocol["folds"]:
        if fold["fold_id"] not in args.fold_ids:
            continue
        inputs, scaler, episodes = prepare_fold(data, parent_episodes, fold)
        for seed in args.seeds:
            if seed not in registration["training_seeds"]:
                raise ValueError("The requested seed is outside registration")
            parent_args, warmup, static, static_path, bundle = _parent_artifacts(
                args, fold, seed
            )
            if args.phase in ("bases", "train"):
                collect_bases(args, registration, data, fold, seed, bundle, static_path)
            if args.phase == "bases":
                continue
            probe_bases = load_probe_bases(args, fold, seed)
            if args.phase == "gradients":
                static_unit = matching_unit(
                    "static", fold["fold_id"], seed, "grad_core"
                )
                static_build = static_factory(
                    parent_args, inputs, static_unit, warmup, bundle
                )
                heldout_gradient_diagnostic(
                    args, data, inputs, fold, seed, static_build, static, bundle
                )
                continue
            for arm in args.arms:
                unit = {"fold_id": fold["fold_id"], "seed": seed, "arm": arm}
                factory, static_build = model_factory(
                    args,
                    registration,
                    inputs,
                    unit,
                    parent_args,
                    warmup,
                    static,
                    bundle,
                    probe_bases,
                )
                if args.phase == "train":
                    if not train_unit(
                        args,
                        registration,
                        data,
                        inputs,
                        scaler,
                        episodes,
                        fold,
                        unit,
                        factory,
                        static_build,
                        static,
                        static_path,
                    ):
                        return
                else:
                    checkpoint = (
                        unit_path(args.private_root, unit) / "CHECKPOINT_SELECTED.pt"
                    )
                    for mode in registration["C"]["evaluation_modes"]:
                        score_unit(
                            args, data, inputs, fold, unit, factory, checkpoint, mode
                        )
    print(
        json.dumps(
            {
                "status": "PHASE_COMPLETE",
                "experiment": "C",
                "phase": args.phase,
                "final_target_query_labels_used": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
