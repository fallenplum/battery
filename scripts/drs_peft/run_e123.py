"""Run fold-local source experiments with sealed outer and final-target roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from threadpoolctl import threadpool_limits

from models.drs_peft.diagnostic_dualview import (
    SourceViewScaler,
    prefix_view_descriptors,
)
from models.drs_peft.peft import SubspaceLinear, install_peft
from scripts.drs_peft.data import SourceOnlyScaler, file_hash, load_dataset, write_json
from scripts.drs_peft.e123_protocol import (
    fold_episode_view,
    load_protocol,
    validate_against_data,
)
from scripts.drs_peft.episodes import load_registration
from scripts.drs_peft.gradient_audit import (
    _collect,
    _covariances,
    _group_balanced_weights,
    _group_indices,
    _replay_gradients,
    _top,
    state_fingerprint,
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
    make_model,
    predict,
    read_json,
    save_torch,
    selected_state,
    train_source_unit,
)

METHODS = ("grad_core", "lora_xs", "weight_svd_core")
MODES = ("constant", "absolute", "relative", "dual")
EVALUATION_MODES = ("trained_ball", "matched_norm")
DEFAULT_CONFIG = ROOT / "configs/drs_peft/e123_v1.json"
EXTRA_SOURCES = (
    "scripts/drs_peft/run_e123.py",
    "scripts/drs_peft/e123_protocol.py",
    "models/drs_peft/e123_conditioning.py",
    "models/drs_peft/diagnostic_dualview.py",
    "configs/drs_peft/e123_folds_v1.json",
    "configs/drs_peft/execution_v1.json",
)


def payload_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def bind_shared(path, value):
    """Serialize shared registrations across fold/seed worker processes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        bind(path, value)


def radius_name(value):
    return format(float(value), "g").replace(".", "p")


def build_plan(registration, protocol):
    """Expand registered comparisons without using any outcome or score."""
    if list(registration["methods"]) != list(METHODS):
        raise ValueError("E2 requires all three registered adapter parameterizations")
    if list(registration["modes"]) != list(MODES):
        raise ValueError("E1 requires the four matched conditioning arms")
    if list(registration["evaluation_modes"]) != list(EVALUATION_MODES):
        raise ValueError("Both registered update-norm evaluation modes are required")
    primary = float(registration["primary_radius"])
    if primary <= 0:
        raise ValueError("A positive primary physical update radius is required")
    sensitivity = [float(value) for value in registration["grad_radius_sensitivity"]]
    if any(value <= 0 or value == primary for value in sensitivity):
        raise ValueError("Additional radii must be positive and distinct from primary")
    if len(set(sensitivity)) != len(sensitivity):
        raise ValueError("Duplicate amplitude comparisons would change the budget")
    warmup, static, conditioned = [], [], []
    for fold in protocol["folds"]:
        for seed in registration["training_seeds"]:
            base = {"fold_id": int(fold["fold_id"]), "seed": int(seed)}
            warmup.append(dict(base, stage="warmup"))
            for method in METHODS:
                static.append(dict(base, stage="static", method=method))
                radii = [primary, *sensitivity] if method == "grad_core" else [primary]
                for radius in radii:
                    for mode in MODES:
                        conditioned.append(
                            dict(
                                base,
                                stage="condition",
                                method=method,
                                mode=mode,
                                radius=float(radius),
                            )
                        )
    plan = {
        "study_id": registration["study_id"],
        "warmup": warmup,
        "static": static,
        "condition": conditioned,
        "outer_evaluation_modes": list(EVALUATION_MODES),
        "total_training_units": len(warmup) + len(static) + len(conditioned),
        "final_target_query_labels_used": 0,
        "outer_score_policy": "all registered source units must complete before outer scoring",
        "protocol_sha256": protocol["protocol_sha256"],
    }
    plan["plan_sha256"] = payload_hash(plan)
    return plan


def unit_path(root, unit):
    folder = (
        Path(root) / f"fold{unit['fold_id']}" / f"seed{unit['seed']}" / unit["stage"]
    )
    if unit["stage"] in ("static", "condition"):
        folder /= unit["method"]
    if unit["stage"] == "condition":
        folder /= "radius" + radius_name(unit["radius"])
        folder /= unit["mode"]
    return folder


def matching_unit(stage, fold_id, seed, method=None):
    result = {"stage": stage, "fold_id": int(fold_id), "seed": int(seed)}
    if method is not None:
        result["method"] = method
    return result


def checked_training_checkpoint(folder):
    folder = Path(folder)
    receipt = read_json(folder / "SOURCE_DEVELOPMENT_COMPLETE.json")
    for record in receipt["files"]:
        if file_hash(folder / record["path"]) != record["sha256"]:
            raise RuntimeError("A completed fold-local training artifact changed")
    checkpoint = folder / "CHECKPOINT_SELECTED.pt"
    if file_hash(checkpoint) != receipt["selected_checkpoint_sha256"]:
        raise RuntimeError("A selected fold-local source checkpoint changed")
    if receipt["target_query_labels_used"] != 0:
        raise RuntimeError("Final-target outcomes entered a source training unit")
    return torch.load(checkpoint, map_location="cpu", weights_only=False), checkpoint


def training_config(registration):
    config = read_json(ROOT / "configs/drs_peft/execution_v1.json")
    config["study_id"] = registration["study_id"]
    config["training"]["learning_rate"] = float(registration["bridge_head_lr"])
    config["training"]["batch_size"] = 8
    return config


def prepare_fold(data, parent_episodes, fold):
    episodes = fold_episode_view(parent_episodes, fold)
    fit = data.rows(fold["fit_ids"])
    scaler = SourceOnlyScaler().fit(data, fit)
    if scaler.to_dict()["training_ids"] != fold["fit_ids"]:
        raise RuntimeError("Fold preprocessing used identities outside fit")
    inputs = scaler.transform(data)
    return inputs, scaler, episodes


def prepare_condition_inputs(data, inputs, fold):
    """Extract source-pool prefixes while fitting gate normalization on fit only."""
    role_ids = [
        fold[key] for key in ("fit_ids", "inner_validation_ids", "outer_test_ids")
    ]
    selected_ids = [uid for ids in role_ids for uid in ids]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Conditioning descriptor roles repeat a physical cell")
    rows = data.rows(selected_ids)
    descriptors = prefix_view_descriptors(
        inputs["x"][rows], inputs["point_mask"][rows], inputs["cycle_ids"][rows]
    )
    source_scaler = SourceViewScaler().fit(
        descriptors, np.arange(len(fold["fit_ids"])), fold["fit_ids"]
    )
    normalized = source_scaler.transform(descriptors)
    result = dict(inputs)
    for view in ("absolute", "relative"):
        values = np.zeros((len(data.ids), 12), dtype=np.float32)
        values[rows] = normalized[view]
        result["diagnostic_" + view] = values
    manifest = {
        "fit_ids": fold["fit_ids"],
        "inner_validation_ids": fold["inner_validation_ids"],
        "outer_prefix_ids": fold["outer_test_ids"],
        "normalization": source_scaler.to_dict(),
        "final_target_descriptors_computed": 0,
        "descriptor_sha256": {
            view: hashlib.sha256(result["diagnostic_" + view].tobytes()).hexdigest()
            for view in ("absolute", "relative")
        },
    }
    return result, manifest


def warmup_factory(args, inputs, seed):
    return lambda: make_model(
        args.backbone_dir,
        inputs["policy"].shape[-1],
        "native_multivariate",
        "B0_pretrained_frozen",
        seed,
    )


def basis_folder(args, fold_id, seed):
    return args.private_root / f"fold{fold_id}" / f"seed{seed}" / "bases"


def checked_bases(args, fold_id, seed, fit_ids):
    folder = basis_folder(args, fold_id, seed)
    receipt = read_json(folder / "BASES_COMPLETE.json")
    if receipt["fit_ids"] != fit_ids or receipt["final_target_query_labels_used"] != 0:
        raise RuntimeError("Fold-local gradient basis provenance differs")
    for record in receipt["files"]:
        if file_hash(folder / record["path"]) != record["sha256"]:
            raise RuntimeError("The completed fold-local basis changed")
    return torch.load(folder / "BASES.pt", map_location="cpu", weights_only=False)


def static_factory(args, inputs, unit, warmup, bundle):
    method, seed = unit["method"], unit["seed"]
    bases = bundle["bases"] if method == "grad_core" else None

    def construct():
        model = warmup_factory(args, inputs, seed)()
        load_selected(model, warmup["model"])
        if state_fingerprint(model) != bundle["source_checkpoint_sha256"]:
            raise RuntimeError(
                "Static core source differs from the fold gradient checkpoint"
            )
        install_peft(
            model.backbone, method, rank=4, alpha=4, dropout=0.0, seed=seed, bases=bases
        )
        model.requires_grad_(False)
        layers = [
            layer
            for layer in model.backbone.modules()
            if isinstance(layer, SubspaceLinear)
        ]
        if len(layers) != 12:
            raise RuntimeError(
                "E2 static target set differs from the twelve Q/V matrices"
            )
        for layer in layers:
            layer.core.requires_grad_(True)
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != 192:
            raise RuntimeError(
                "Static source fitting must update only 192 core parameters"
            )
        return model

    return construct


def condition_factory(args, inputs, unit, warmup, bundle, static_checkpoint):
    from models.drs_peft.e123_conditioning import ConditionedBatteryTSFM

    static_unit = matching_unit("static", unit["fold_id"], unit["seed"], unit["method"])
    construct_static = static_factory(args, inputs, static_unit, warmup, bundle)
    reference_fraction = float(read_json(args.config)["reference_weight_fraction"])

    def construct():
        model = construct_static()
        load_selected(model, static_checkpoint["model"])
        conditioned = ConditionedBatteryTSFM(
            model,
            unit["mode"],
            unit["radius"],
            reference_weight_fraction=reference_fraction,
        )
        count = sum(p.numel() for p in conditioned.parameters() if p.requires_grad)
        if count != 1440:
            raise RuntimeError(
                "Every conditioning arm must train 1440 adapter/gate parameters"
            )
        saved = selected_state(conditioned)
        names = {name for name, p in conditioned.named_parameters() if p.requires_grad}
        if not names.issubset(saved):
            raise RuntimeError("The frozen trainer would omit a conditioning parameter")
        return conditioned

    return construct


def train_unit(
    args,
    registration,
    data,
    inputs,
    scaler,
    episodes,
    unit,
    factory,
    parent_checkpoint=None,
    extra_identity=None,
):
    destination = unit_path(args.private_root, unit)
    published = unit_path(args.public_root / "units", unit)
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "RUN.lock")):
        identity = {
            "unit": unit,
            "registration_sha256": file_hash(args.config),
            "protocol_file_sha256": file_hash(args.folds),
            "fit_ids": episodes["source_cell_roles"]["source_train"],
            "inner_validation_ids": episodes["source_cell_roles"][
                "source_meta_validation"
            ],
            "outer_ids": episodes["source_cell_roles"]["source_outer_validation"],
            "outer_labels_used_for_training_or_selection": 0,
            "final_target_query_labels_used": 0,
            "parent_checkpoint_sha256": file_hash(parent_checkpoint)
            if parent_checkpoint
            else None,
            "extra": extra_identity or {},
        }
        bind(destination / "E123_UNIT_SPEC.json", identity)
        epochs = int(registration["epochs"][unit["stage"]])
        receipt = train_source_unit(
            data,
            inputs,
            scaler,
            episodes,
            training_config(registration),
            args.config,
            args.backbone_dir,
            destination,
            "native_multivariate",
            "E123_" + unit["stage"],
            unit["seed"],
            epochs,
            args.device,
            rank=4,
            adapter_lr=float(registration["adapter_lr"]),
            stop_after=args.stop_after,
            stage="E123_FOLD_LOCAL_SOURCE",
            model_factory=factory,
            initial_checkpoint_sha256=identity["parent_checkpoint_sha256"],
            extra_sources=EXTRA_SOURCES,
        )
        if receipt["status"] == "ENGINEERING_INTERRUPTED":
            print(
                json.dumps(
                    dict(event="E123_ENGINEERING_INTERRUPTED", unit=unit, **receipt)
                ),
                flush=True,
            )
            return False
        expected = epochs * math.ceil(len(identity["fit_ids"]) / 8)
        if receipt["updates"] != expected or receipt["target_query_labels_used"] != 0:
            raise RuntimeError(
                "E123 training receipt violated its registered update or label budget"
            )
        _publish_unit(destination, published)
        print(
            json.dumps(
                {
                    "event": "E123_TRAINING_UNIT_COMPLETE",
                    "unit": unit,
                    "updates": receipt["updates"],
                    "inner_validation": receipt["validation"],
                }
            ),
            flush=True,
        )
        return True


def publish_bases(folder, destination):
    """Complete or repair interrupted publication from verified private bases."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "GRADIENT_CACHE_MANIFEST.json",
        "GRADIENT_REPLAY.json",
        "GRADIENT_SPECTRA.csv",
    ):
        target = destination / name
        pending = target.with_suffix(target.suffix + ".pending")
        shutil.copyfile(folder / name, pending)
        os.replace(pending, target)
    bind(destination / "BASES_COMPLETE.json", read_json(folder / "BASES_COMPLETE.json"))


def collect_bases(args, data, inputs, episodes, fold_id, seed):
    folder = basis_folder(args, fold_id, seed)
    folder.mkdir(parents=True, exist_ok=True)
    fit_ids = episodes["source_cell_roles"]["source_train"]
    rows = data.rows(fit_ids)
    warmup_unit = matching_unit("warmup", fold_id, seed)
    checkpoint, checkpoint_path = checked_training_checkpoint(
        unit_path(args.private_root, warmup_unit)
    )
    with FileLock(str(folder / "RUN.lock")):
        if (folder / "BASES_COMPLETE.json").exists():
            checked_bases(args, fold_id, seed, fit_ids)
            publish_bases(
                folder, args.public_root / "bases" / f"fold{fold_id}" / f"seed{seed}"
            )
            return
        model = warmup_factory(args, inputs, seed)().to(args.device)
        load_selected(model, checkpoint["model"])
        before = state_fingerprint(model)
        layers = {
            name: layer
            for name, layer in model.backbone.named_modules()
            if isinstance(layer, torch.nn.Linear)
            and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj")
        }
        if len(layers) != 12:
            raise RuntimeError("Fold gradient target family changed")
        gradient, receipt = _collect(
            model,
            data,
            inputs,
            rows,
            rows,
            layers,
            folder / "gradients",
            seed,
            "E123_FIT",
            "nominal",
            args.device,
            before,
        )
        replay = _replay_gradients(
            model,
            data,
            inputs,
            [("E123_FIT", "nominal", rows, rows)],
            {("E123_FIT", "nominal"): gradient},
            layers,
            args.device,
        )
        if state_fingerprint(model) != before:
            raise RuntimeError("Fold gradient collection changed its source model")
        weights = _group_balanced_weights(
            _group_indices(data.protocol_ids[rows]), len(rows)
        )
        bases, spectra = {}, []
        with threadpool_limits(limits=2):
            for name, values in gradient.items():
                g = np.asarray(values, dtype=np.float64)
                left, right, _, _ = _covariances(g, weights)
                u, le = _top(left, 4)
                v, re = _top(right, 4)
                bases[name] = (
                    torch.from_numpy(u.astype(np.float32)),
                    torch.from_numpy(v.astype(np.float32)),
                )
                for side, eigenvalues in (("left", le), ("right", re)):
                    spectra.extend(
                        {
                            "layer": name,
                            "side": side,
                            "direction": i + 1,
                            "eigenvalue": float(value),
                        }
                        for i, value in enumerate(eigenvalues)
                    )
        save_torch(
            folder / "BASES.pt",
            {
                "rank": 4,
                "bases": bases,
                "fit_ids": fit_ids,
                "source_checkpoint_sha256": before,
                "warmup_checkpoint_sha256": file_hash(checkpoint_path),
            },
        )
        write_json(folder / "GRADIENT_CACHE_MANIFEST.json", receipt)
        write_json(folder / "GRADIENT_REPLAY.json", replay)
        pd.DataFrame(spectra).to_csv(folder / "GRADIENT_SPECTRA.csv", index=False)
        files = [
            {"path": name, "sha256": file_hash(folder / name)}
            for name in (
                "BASES.pt",
                "GRADIENT_CACHE_MANIFEST.json",
                "GRADIENT_REPLAY.json",
                "GRADIENT_SPECTRA.csv",
            )
        ]
        complete = {
            "status": "COMPLETE_FIT_ONLY_BASES",
            "fit_ids": fit_ids,
            "fold_id": fold_id,
            "seed": seed,
            "files": files,
            "source_checkpoint_sha256": before,
            "new_gradient_evaluations": len(rows),
            "endpoint_replay_evaluations": 2,
            "optimizer_updates": 0,
            "outer_labels_used": 0,
            "final_target_query_labels_used": 0,
            "registration_sha256": file_hash(args.config),
            "source_sha256": file_hash(__file__),
        }
        write_json(folder / "BASES_COMPLETE.json", complete)
        public = args.public_root / "bases" / f"fold{fold_id}" / f"seed{seed}"
        publish_bases(folder, public)
        del model, checkpoint, gradient, bases
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {"event": "E123_FIT_BASES_COMPLETE", "fold_id": fold_id, "seed": seed}
            ),
            flush=True,
        )


def check_initialization(factory, static_factory, inputs, fit_rows, device):
    model, reference = factory().to(device).eval(), static_factory().to(device).eval()
    batch = early_batch(inputs, fit_rows[:2], device, model.input_keys)
    query = torch.tensor(
        [100, 150, 500, 1500, 5000], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        actual, expected = model(batch, query), reference(batch, query)
    if not torch.equal(actual, expected):
        raise RuntimeError(
            "Conditioning changed the fixed source-adapter starting prediction"
        )
    result = {
        "status": "PASSED",
        "initial_prediction_max_difference": 0.0,
        "query_cycles": query.cpu().tolist(),
        "fit_cells": len(fit_rows[:2]),
        "final_target_query_labels_used": 0,
    }
    del model, reference
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def require_all_training_complete(args, plan):
    """Open outer evaluation only after the entire registered source matrix closes."""
    records = []
    for stage in ("warmup", "static", "condition"):
        for unit in plan[stage]:
            folder = unit_path(args.private_root, unit)
            if not (folder / "SOURCE_DEVELOPMENT_COMPLETE.json").exists():
                raise RuntimeError(
                    "Outer evaluation is sealed while source units remain incomplete: "
                    + str(unit)
                )
            _, path = checked_training_checkpoint(folder)
            records.append(
                {
                    "unit": unit,
                    "checkpoint_sha256": file_hash(path),
                    "source_receipt_sha256": file_hash(
                        folder / "SOURCE_DEVELOPMENT_COMPLETE.json"
                    ),
                }
            )
    return records


def publish_outer(folder, destination, receipt):
    destination.mkdir(parents=True, exist_ok=True)
    for record in receipt["files"]:
        source = folder / record["path"]
        if source.suffix in (".json", ".csv"):
            target = destination / source.name
            if target.exists() and file_hash(target) != record["sha256"]:
                raise RuntimeError("A published outer diagnostic artifact changed")
            shutil.copyfile(source, target)
    bind(destination / "OUTER_COMPLETE.json", receipt)


def update_diagnostic_rows(model, batch, ids):
    """Materialize per-cell physical norms without changing the model state."""
    rows = []
    for layer, values in model.update_diagnostics(batch).items():
        arrays = {name: value.detach().cpu().numpy() for name, value in values.items()}
        if any(array.shape != (len(ids),) for array in arrays.values()):
            raise RuntimeError("Update diagnostics must retain the physical-cell axis")
        for index, uid in enumerate(ids):
            rows.append(
                {
                    "cell_uid": uid,
                    "layer": layer,
                    **{name: array[index].item() for name, array in arrays.items()},
                }
            )
    return rows


def score_outer_unit(
    args, data, inputs, fold, unit, factory, checkpoint_path, evaluation_mode
):
    suffix = unit_path(Path("."), unit)
    folder = args.private_root / "outer" / suffix / evaluation_mode
    destination = args.public_root / "outer" / suffix / evaluation_mode
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "OUTER_COMPLETE.json"
    if marker.exists():
        receipt = read_json(marker)
        if receipt["checkpoint_sha256"] != file_hash(checkpoint_path):
            raise RuntimeError(
                "Completed outer evaluation points to another source checkpoint"
            )
        for record in receipt["files"]:
            if file_hash(folder / record["path"]) != record["sha256"]:
                raise RuntimeError("A completed outer result changed")
        publish_outer(folder, destination, receipt)
        return
    rows = data.rows(fold["outer_test_ids"])
    model = factory().to(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_selected(model, checkpoint["model"])
    if unit["stage"] == "condition":
        model.set_evaluation_mode(evaluation_mode)
    before = state_fingerprint(model)
    prediction = predict(model, inputs, rows, data.query, args.device)
    reconstructed = factory().to(args.device)
    load_selected(reconstructed, checkpoint["model"])
    if unit["stage"] == "condition":
        reconstructed.set_evaluation_mode(evaluation_mode)
    replay = predict(reconstructed, inputs, rows, data.query, args.device)
    if not np.array_equal(prediction, replay) or state_fingerprint(model) != before:
        raise RuntimeError("Outer serialized replay or source-state protection failed")
    table, metrics = evaluate_arrays(
        prediction,
        data.targets[rows],
        data.observed[rows],
        data.query,
        [data.ids[row] for row in rows],
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
        dict(unit=unit, evaluation_mode=evaluation_mode, **metrics),
    )
    diagnostics = []
    if unit["stage"] == "condition":
        for start in range(0, len(rows), 8):
            batch_rows = rows[start : start + 8]
            batch = early_batch(inputs, batch_rows, args.device, model.input_keys)
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
        "evaluation_mode": evaluation_mode,
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
    del model, reconstructed, checkpoint
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "event": "E123_OUTER_UNIT_COMPLETE",
                "unit": unit,
                "evaluation_mode": evaluation_mode,
                "metrics": metrics,
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
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/e123_v1"
    )
    parser.add_argument(
        "--public-root", type=Path, default=ROOT / "results/drs_peft/e123_v1"
    )
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("warmup", "bases", "static", "condition", "score", "all"),
        required=True,
    )
    parser.add_argument(
        "--fold-ids", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2]
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=(42, 123, 2026), default=[42, 123, 2026]
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
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
    args.public_root.mkdir(parents=True, exist_ok=True)
    bind_shared(
        args.public_root / "E123_PLAN.json",
        dict(
            plan,
            registration_sha256=file_hash(args.config),
            folds_file_sha256=file_hash(args.folds),
        ),
    )
    data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
    parent_episodes = load_registration()
    validation = validate_against_data(protocol, data, parent_episodes)
    bind_shared(args.public_root / "DATA_PROTOCOL_VALIDATION.json", validation)
    selected_folds = [
        fold for fold in protocol["folds"] if fold["fold_id"] in args.fold_ids
    ]
    if args.phase == "score":
        completed = require_all_training_complete(args, plan)
        bind_shared(
            args.public_root / "OUTER_EVALUATION_OPENED.json",
            {
                "plan_sha256": plan["plan_sha256"],
                "source_units": completed,
                "selection_rule": "own inner-validation first strict minimum; all outer settings frozen",
                "final_target_query_labels_used": 0,
            },
        )
    for fold in selected_folds:
        inputs, scaler, episodes = prepare_fold(data, parent_episodes, fold)
        conditioned_inputs, view_manifest = prepare_condition_inputs(data, inputs, fold)
        bind_shared(
            args.public_root / f"fold{fold['fold_id']}" / "FOLD_VIEW_MANIFEST.json",
            view_manifest,
        )
        for seed in args.seeds:
            warmup_unit = matching_unit("warmup", fold["fold_id"], seed)
            if args.phase in ("warmup", "all") and not train_unit(
                args,
                registration,
                data,
                inputs,
                scaler,
                episodes,
                warmup_unit,
                warmup_factory(args, inputs, seed),
            ):
                return
            if args.phase == "warmup":
                continue
            if args.phase in ("bases", "all"):
                collect_bases(args, data, inputs, episodes, fold["fold_id"], seed)
            if args.phase == "bases":
                continue
            warmup, warmup_path = checked_training_checkpoint(
                unit_path(args.private_root, warmup_unit)
            )
            bundle = checked_bases(args, fold["fold_id"], seed, fold["fit_ids"])
            for method in args.methods:
                static_unit = matching_unit("static", fold["fold_id"], seed, method)
                static_build = static_factory(args, inputs, static_unit, warmup, bundle)
                if args.phase in ("static", "all") and not train_unit(
                    args,
                    registration,
                    data,
                    inputs,
                    scaler,
                    episodes,
                    static_unit,
                    static_build,
                    warmup_path,
                ):
                    return
                if args.phase == "static":
                    continue
                static, static_path = checked_training_checkpoint(
                    unit_path(args.private_root, static_unit)
                )
                if args.phase == "score":
                    score_outer_unit(
                        args,
                        data,
                        inputs,
                        fold,
                        static_unit,
                        static_build,
                        static_path,
                        "static",
                    )
                units = [
                    unit
                    for unit in plan["condition"]
                    if unit["fold_id"] == fold["fold_id"]
                    and unit["seed"] == seed
                    and unit["method"] == method
                    and unit["mode"] in args.modes
                ]
                for unit in units:
                    factory = condition_factory(
                        args, inputs, unit, warmup, bundle, static
                    )
                    if args.phase in ("condition", "all"):
                        folder = unit_path(args.private_root, unit)
                        initial_path = folder / "FIXED_ADAPTER_START_CHECK.json"
                        if not initial_path.exists():
                            fixed = partial(_loaded_static, static_build, static)
                            bind(
                                initial_path,
                                check_initialization(
                                    factory,
                                    fixed,
                                    conditioned_inputs,
                                    data.rows(fold["fit_ids"]),
                                    args.device,
                                ),
                            )
                        if not train_unit(
                            args,
                            registration,
                            data,
                            conditioned_inputs,
                            scaler,
                            episodes,
                            unit,
                            factory,
                            static_path,
                            extra_identity={
                                "view_manifest_sha256": payload_hash(view_manifest)
                            },
                        ):
                            return
                    if args.phase == "score":
                        _, selected = checked_training_checkpoint(
                            unit_path(args.private_root, unit)
                        )
                        for evaluation_mode in EVALUATION_MODES:
                            score_outer_unit(
                                args,
                                data,
                                conditioned_inputs,
                                fold,
                                unit,
                                factory,
                                selected,
                                evaluation_mode,
                            )
    print(
        json.dumps(
            {
                "status": "PHASE_COMPLETE",
                "phase": args.phase,
                "final_target_query_labels_used": 0,
            }
        ),
        flush=True,
    )


def _loaded_static(factory, checkpoint):
    model = factory()
    load_selected(model, checkpoint["model"])
    return model


if __name__ == "__main__":
    main()
