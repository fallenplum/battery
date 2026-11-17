"""Independent CPU-only provenance, metric and update-norm audit for E1--E3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from models.drs_peft.diagnostic_dualview import prefix_view_descriptors
from scripts.drs_peft.data import POLICY_PHYSICAL_SCALES, load_dataset

ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA = "369ca882403cbff7e4fe414d7bc55b72ac7fea34d583f684ad36d732fbdc9e9c"
METHODS = ("grad_core", "lora_xs", "weight_svd_core")
MODES = ("constant", "absolute", "relative", "dual")
WINDOWS = (
    ("all", 101, 5000),
    ("101_200", 101, 200),
    ("201_500", 201, 500),
    ("501_1000", 501, 1000),
    ("1001_plus", 1001, 5000),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_files(folder, records):
    for record in records:
        require(
            digest(folder / record["path"]) == record["sha256"],
            "Completed artifact bytes changed at " + str(folder / record["path"]),
        )


def units(registration, protocol):
    result = []
    for fold in protocol["folds"]:
        for seed in registration["training_seeds"]:
            base = {"fold_id": fold["fold_id"], "seed": seed}
            result.append(dict(base, stage="warmup"))
            for method in METHODS:
                result.append(dict(base, stage="static", method=method))
                radii = [registration["primary_radius"]]
                if method == "grad_core":
                    radii += registration["grad_radius_sensitivity"]
                for radius in radii:
                    for mode in MODES:
                        result.append(
                            dict(
                                base,
                                stage="condition",
                                method=method,
                                radius=float(radius),
                                mode=mode,
                            )
                        )
    return result


def unit_path(root, unit):
    folder = root / f"fold{unit['fold_id']}" / f"seed{unit['seed']}" / unit["stage"]
    if unit["stage"] in ("static", "condition"):
        folder = folder / unit["method"]
    if unit["stage"] == "condition":
        name = format(float(unit["radius"]), "g").replace(".", "p")
        folder = folder / ("radius" + name) / unit["mode"]
    return folder


def bridge_head_hash(state):
    value = hashlib.sha256()
    for key in sorted(state):
        if not key.startswith("bridge.backbone.module."):
            tensor = state[key].detach().cpu().contiguous()
            value.update(key.encode())
            value.update(str(tensor.dtype).encode())
            value.update(str(tuple(tensor.shape)).encode())
            value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def audit_static_bases(state, method, bundle, official, svd_cache):
    prefixes = [key.removesuffix(".core") for key in state if key.endswith(".core")]
    require(len(prefixes) == 12, "Static fit must contain exactly twelve Q/V cores")
    for prefix in prefixes:
        name = prefix.removeprefix("bridge.backbone.module.")
        left, right = (
            state[prefix + ".U"].double().numpy(),
            state[prefix + ".V"].double().numpy(),
        )
        if method == "grad_core":
            expected_left, expected_right = bundle["bases"][name]
            np.testing.assert_array_equal(left, expected_left.double().numpy())
            np.testing.assert_array_equal(right, expected_right.double().numpy())
            continue
        if name not in svd_cache:
            u, singular, vh = np.linalg.svd(
                official[name + ".weight"].double().numpy(), full_matrices=False
            )
            svd_cache[name] = (u[:, :4], vh[:4].T, singular[:4])
        expected_left, expected_right, singular = svd_cache[name]
        for actual, expected in ((left, expected_left), (right, expected_right)):
            orthogonal = np.linalg.qr(actual, mode="reduced")[0]
            overlap = np.square(
                np.linalg.svd(orthogonal.T @ expected, compute_uv=False)
            ).mean()
            require(
                overlap > 1 - 1e-5,
                "Weight-SVD/XS source directions differ from official W0",
            )
        scales = np.linalg.norm(right, axis=0)
        np.testing.assert_allclose(
            scales, singular if method == "lora_xs" else 1, rtol=2e-6, atol=2e-6
        )


def recovery_provenance(args):
    amendment_path = args.public_root / "IO_LOCK_AMENDMENT.json"
    if not amendment_path.exists():
        return {
            "amendment_present": False,
            "completed_invocations": 0,
            "recovered_warmups": [],
        }
    amendment = read(amendment_path)
    for key in ("driver", "launcher", "trainer"):
        record = amendment[key]
        require(
            digest(ROOT / record["path"]) == record["sha256"],
            "Recovery execution source changed",
        )
    require(
        amendment["configuration_sha256"] == digest(args.config),
        "Recovery configuration changed",
    )
    require(
        amendment["training_operator_changes"]
        == amendment["final_target_query_labels_used"]
        == 0,
        "The lock amendment changes a training operator or target role",
    )
    lock_root = Path(amendment["lock_root"]).resolve()
    invocations, recovered = [], {}
    running = []
    for path in sorted((args.public_root / "io_invocations").glob("*.json")):
        invocation = read(path)
        require(
            invocation["amendment_sha256"] == digest(amendment_path),
            "Launcher invocation binds another amendment",
        )
        for mapping in invocation["requested_lock_mappings"]:
            identity = os.path.normcase(str(Path(mapping["requested"]).resolve()))
            expected = lock_root / (
                hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".lock"
            )
            require(
                Path(mapping["actual"]).resolve() == expected,
                "Active lock escaped the registered relocation mapping",
            )
        if invocation["status"] == "RUNNING":
            running.append(path.name)
            continue
        require(
            invocation["status"] in ("COMPLETE", "FAILED"),
            "Unknown launcher completion state",
        )
        record = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "status": invocation["status"],
        }
        if invocation["status"] == "COMPLETE":
            before, after = invocation["before_states"], invocation["after_states"]
            require(
                set(before).issubset(after), "A resumed optimizer state disappeared"
            )
            calculated = sum(item["updates"] for item in after.values()) - sum(
                item["updates"] for item in before.values()
            )
            require(
                calculated == invocation["new_optimizer_updates"],
                "Launcher optimizer increment ledger differs",
            )
            checks = {item["unit"]: item for item in invocation["resumed_state_checks"]}
            require(
                set(checks) == set(before),
                "Resume audit omitted an existing optimizer prefix",
            )
            for unit, previous in before.items():
                current = after[unit]
                require(
                    previous["original_files"]["RUN_CONFIG.json"]
                    == current["original_files"]["RUN_CONFIG.json"]
                    and previous["original_files"]["SOURCE_SNAPSHOT.json"]
                    == current["original_files"]["SOURCE_SNAPSHOT.json"],
                    "Recovery rewrote an original scientific execution identity",
                )
                added = current["updates"] - previous["updates"]
                require(
                    added == checks[unit]["new_optimizer_updates"],
                    "Resumed unit update increment differs",
                )
                if previous["was_epoch_complete"]:
                    require(
                        added == 0
                        and previous["original_files"] == current["original_files"],
                        "Finalization recovery repeated training updates",
                    )
                    require(
                        checks[unit]["completed_optimizer_state_preserved"],
                        "Completed optimizer preservation check failed",
                    )
                folder = Path(unit)
                if (
                    folder.name == "warmup"
                    and folder.parent.name == "seed42"
                    and folder.parent.parent.name in ("fold0", "fold1")
                    and previous["was_epoch_complete"]
                ):
                    train_state = torch.load(
                        folder / "TRAIN_STATE.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    history = read(folder / "TRAIN_LOG.json")
                    steps = [
                        int(item["step"])
                        for item in train_state["optimizer"]["state"].values()
                        if "step" in item
                    ]
                    require(
                        train_state["updates"] == 960
                        and len(history) == 120
                        and train_state["history"] == history
                        and steps
                        and all(step == 960 for step in steps),
                        "The two original completed warmup prefixes changed",
                    )
                    require(
                        digest(folder / "TRAIN_STATE.pt")
                        == previous["original_files"]["TRAIN_STATE.pt"],
                        "Recovered completed warmup state bytes changed",
                    )
                    recovered[unit] = {
                        "fold_id": int(folder.parent.parent.name.removeprefix("fold")),
                        "seed": 42,
                        "optimizer_updates": 960,
                        "history_epochs": 120,
                        "optimizer_step_min": min(steps),
                        "optimizer_step_max": max(steps),
                        "finalization_optimizer_updates": 0,
                        "train_state_sha256": digest(folder / "TRAIN_STATE.pt"),
                    }
            record["new_optimizer_updates"] = calculated
        invocations.append(record)
    return {
        "amendment_present": True,
        "amendment_sha256": digest(amendment_path),
        "launcher_sha256": amendment["launcher"]["sha256"],
        "driver_sha256": amendment["driver"]["sha256"],
        "trainer_sha256": amendment["trainer"]["sha256"],
        "configuration_sha256": amendment["configuration_sha256"],
        "completed_invocations": sum(
            record["status"] == "COMPLETE" for record in invocations
        ),
        "failed_invocations": sum(
            record["status"] == "FAILED" for record in invocations
        ),
        "running_invocations": running,
        "invocations": invocations,
        "recovered_warmups": list(recovered.values()),
        "training_operator_changes": 0,
    }


def independent_moments(values, masks, axes, floor):
    values, masks = np.asarray(values, np.float64), np.asarray(masks, bool)
    count = masks.sum(axis=axes)
    mean = np.where(masks, values, 0).sum(axis=axes) / np.maximum(count, 1)
    variance = np.where(masks, (values - mean) ** 2, 0).sum(axis=axes) / np.maximum(
        count, 1
    )
    return mean, np.maximum(np.sqrt(variance), floor), count


def audit_scaler(record, data, fold):
    require(
        record["training_ids"] == fold["fit_ids"],
        "Scaler identities differ from fold fit",
    )
    rows = data.rows(fold["fit_ids"])
    for key, value, mask, axes, floor in (
        ("x", data.x[rows], data.feature_mask[rows], (0, 1, 2), [0.1, 1.0, 0.05, 0.01]),
        (
            "policy",
            data.policy[rows],
            data.policy_mask[rows],
            (0,),
            POLICY_PHYSICAL_SCALES,
        ),
    ):
        mean, std, count = independent_moments(value, mask, axes, floor)
        actual = record["statistics"][key]
        np.testing.assert_allclose(actual["mean"], mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual["std"], std, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual["supported"], count)


def recompute_metrics(npz_path, csv_path, expected_ids, data):
    with np.load(npz_path, allow_pickle=False) as stored:
        ids = stored["ids"].tolist()
        prediction = stored["prediction"].astype(np.float64)
        target, observed, query = stored["target"], stored["observed"], stored["query"]
    require(ids == expected_ids, "Prediction identities differ from their frozen role")
    rows = data.rows(ids)
    np.testing.assert_array_equal(query, data.query)
    np.testing.assert_array_equal(target, data.targets[rows])
    np.testing.assert_array_equal(observed, data.observed[rows])
    require(np.isfinite(prediction).all(), "Nonfinite predicted source trajectory")
    table = pd.read_csv(csv_path)
    require(
        not table.duplicated(["cell_uid", "window"]).any(),
        "Repeated physical-cell metrics",
    )
    require(len(table) == len(ids) * len(WINDOWS), "Incomplete cell/window metric grid")
    table = table.set_index(["cell_uid", "window"])
    values, differences = [], []
    for position, uid in enumerate(ids):
        for window, lower, upper in WINDOWS:
            valid = observed[position] & (query >= lower) & (query <= upper)
            error = prediction[position, valid] - target[position, valid].astype(
                np.float64
            )
            actual = table.loc[(uid, window)]
            require(
                int(actual.n_obs) == len(error), "Observed metric denominator changed"
            )
            require(
                str(actual.protocol_uid) == str(data.protocol_ids[rows[position]]),
                "Metric protocol identity changed",
            )
            expected = (
                {
                    "mae": np.abs(error).mean(),
                    "rmse": np.sqrt(np.square(error).mean()),
                    "bias": error.mean(),
                }
                if len(error)
                else {"mae": None, "rmse": None, "bias": None}
            )
            for key, value in expected.items():
                if value is None:
                    require(
                        pd.isna(actual[key]), "An unsupported window contains a score"
                    )
                else:
                    difference = abs(float(actual[key]) - float(value))
                    require(
                        difference < 1e-11,
                        "Independently recomputed source metric differs",
                    )
                    differences.append(difference)
            if window == "all":
                values.append(
                    dict(
                        cell_uid=uid,
                        protocol_uid=str(actual.protocol_uid),
                        n_obs=len(error),
                        **expected,
                    )
                )
    full = pd.DataFrame(values)
    summary = {
        "cells": len(full),
        "observations": int(full.n_obs.sum()),
        "mae": float(full.mae.mean()),
        "rmse": float(full.rmse.mean()),
        "protocol_macro_mae": float(full.groupby("protocol_uid").mae.mean().mean()),
    }
    return summary, max(differences, default=0.0), len(table)


def verify_protocol(registration, protocol, data):
    require(
        digest(ROOT / "models/drs_peft/e123_conditioning.py") == MODEL_SHA,
        "The frozen E1--E3 conditioning implementation changed",
    )
    payload = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
    actual = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    require(
        actual == protocol["protocol_sha256"] == registration["fold_payload_sha256"],
        "Fold registration payload changed",
    )
    parent = read(ROOT / "configs/drs_peft/episodes_v1.json")["source_cell_roles"]
    source = set(parent["source_train"]) | set(parent["source_meta_validation"])
    sealed = set(parent["target_test"])
    require(
        source == set(protocol["source_ids"]) and len(source) == 115,
        "Source CV cohort differs from the registered 115 cells",
    )
    require(
        sealed == set(protocol["sealed_target_ids"]) and len(sealed) == 54,
        "Final-target identity exclusion changed",
    )
    outer = []
    for fold in protocol["folds"]:
        roles = [
            fold[name] for name in ("fit_ids", "inner_validation_ids", "outer_test_ids")
        ]
        require(
            set().union(*(set(role) for role in roles)) == source,
            "Fold source roles do not cover the source pool",
        )
        require(sum(map(len, roles)) == len(source), "Fold source identities overlap")
        groups = [set(map(str, data.protocol_ids[data.rows(role)])) for role in roles]
        require(
            sum(map(len, groups)) == len(set().union(*groups)),
            "Fold protocol groups overlap",
        )
        require(not source & sealed, "Final-target identity entered a source fold")
        outer += fold["outer_test_ids"]
    require(
        len(outer) == 115 and set(outer) == source,
        "Outer CV does not score each source cell once",
    )


def basis_audit(args, protocol):
    completed, missing = [], []
    for fold in protocol["folds"]:
        for seed in (42, 123, 2026):
            folder = (
                args.private_root / f"fold{fold['fold_id']}" / f"seed{seed}" / "bases"
            )
            marker = folder / "BASES_COMPLETE.json"
            if not marker.exists():
                missing.append(str(folder))
                continue
            receipt = read(marker)
            verify_files(folder, receipt["files"])
            require(
                receipt["fit_ids"] == fold["fit_ids"],
                "Gradient basis saw a non-fit role",
            )
            require(
                receipt["outer_labels_used"]
                == receipt["final_target_query_labels_used"]
                == 0,
                "Gradient basis used held-out labels",
            )
            warmup = (
                args.private_root
                / f"fold{fold['fold_id']}"
                / f"seed{seed}"
                / "warmup/CHECKPOINT_SELECTED.pt"
            )
            bundle = torch.load(
                folder / "BASES.pt", map_location="cpu", weights_only=False
            )
            require(
                bundle["warmup_checkpoint_sha256"] == digest(warmup),
                "Basis warmup lineage changed",
            )
            require(
                bundle["fit_ids"] == fold["fit_ids"],
                "Serialized basis fit identities changed",
            )
            require(
                bundle["source_checkpoint_sha256"]
                == receipt["source_checkpoint_sha256"],
                "Serialized basis checkpoint fingerprint differs",
            )
            cache = read(folder / "GRADIENT_CACHE_MANIFEST.json")
            require(
                cache["identity"]["ids"]
                == cache["identity"]["outcome_donors"]
                == fold["fit_ids"],
                "Per-cell gradient collection used a held-out role",
            )
            verify_files(Path(cache["folder"]), cache["files"])
            replay = read(folder / "GRADIENT_REPLAY.json")
            require(
                replay["status"] == "PASSED"
                and replay["compared_layer_gradients"] == 24,
                "Gradient endpoint replay failed",
            )
            completed.append(
                {
                    "fold_id": fold["fold_id"],
                    "seed": seed,
                    "fit_cells": len(fold["fit_ids"]),
                    "checkpoint_sha256": digest(warmup),
                    "basis_sha256": digest(folder / "BASES.pt"),
                }
            )
    return completed, missing


def source_audit(args, registration, protocol, data, official):
    records, missing = [], []
    svd_cache, basis_cache = {}, {}
    folds = {fold["fold_id"]: fold for fold in protocol["folds"]}
    for unit in units(registration, protocol):
        folder = unit_path(args.private_root, unit)
        marker = folder / "SOURCE_DEVELOPMENT_COMPLETE.json"
        if not marker.exists():
            missing.append(unit)
            continue
        fold = folds[unit["fold_id"]]
        receipt, spec, config = (
            read(marker),
            read(folder / "E123_UNIT_SPEC.json"),
            read(folder / "RUN_CONFIG.json"),
        )
        backbone = read(folder / "BACKBONE_IDENTITY.json")
        require(
            backbone["weight_sha256"] == args.official_weight_sha256
            and backbone["official_pretrained_weights_loaded"]
            and not backbone["random_control"],
            "Official pretrained backbone identity changed",
        )
        verify_files(folder, receipt["files"])
        require(spec["unit"] == unit, "Unit metadata differs from the frozen matrix")
        require(
            spec["fit_ids"] == config["training_ids"] == fold["fit_ids"],
            "Training role changed",
        )
        require(
            spec["inner_validation_ids"]
            == config["validation_ids"]
            == fold["inner_validation_ids"],
            "Inner selection role changed",
        )
        require(
            spec["outer_ids"] == fold["outer_test_ids"],
            "Outer identity declaration changed",
        )
        require(
            spec["outer_labels_used_for_training_or_selection"]
            == spec["final_target_query_labels_used"]
            == 0,
            "Held-out labels entered source fitting",
        )
        require(
            spec["registration_sha256"] == digest(args.config)
            and spec["protocol_file_sha256"] == digest(args.folds),
            "Source unit registration bytes changed",
        )
        require(
            receipt["target_query_labels_used"] == 0
            and receipt["frozen_parameters_unchanged"],
            "Source frozen-state or target-use check failed",
        )
        require(
            receipt["prediction_replay_max_absolute_error"] == 0,
            "Source checkpoint replay failed",
        )
        expected_updates = registration["epochs"][unit["stage"]] * math.ceil(
            len(fold["fit_ids"]) / 8
        )
        require(
            receipt["updates"] == expected_updates,
            "Source optimizer-update allowance changed",
        )
        expected_parameters = {"warmup": 63169, "static": 192, "condition": 1440}[
            unit["stage"]
        ]
        require(
            receipt["trainable_parameters"]
            == config["trainable_parameters"]
            == expected_parameters,
            "Source trainable parameter budget changed",
        )
        audit_scaler(read(folder / "SOURCE_SCALER.json"), data, fold)
        for record in read(folder / "SOURCE_SNAPSHOT.json")["files"]:
            require(
                digest(ROOT / record["path"]) == record["sha256"],
                "An executed source file changed",
            )
        checkpoint_path = folder / "CHECKPOINT_SELECTED.pt"
        require(
            digest(checkpoint_path) == receipt["selected_checkpoint_sha256"],
            "Selected checkpoint bytes changed",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(
            checkpoint["identity"] == config, "Serialized execution identity differs"
        )
        history = read(folder / "TRAIN_LOG.json")
        best = min(
            range(len(history)), key=lambda index: history[index]["validation_mae"]
        )
        require(
            history[best]["epoch"] == checkpoint["best_epoch"] == receipt["best_epoch"],
            "Selected epoch is not the first strict inner minimum",
        )
        require(
            len(history) == registration["epochs"][unit["stage"]]
            and history[-1]["updates"] == expected_updates,
            "Incomplete source epoch history",
        )
        state = checkpoint["model"]
        bh_hash = bridge_head_hash(state)
        if unit["stage"] != "warmup":
            layers = read(folder / "TRAINABLE_PARAMETER_REPORT.json")["peft_detail"][
                "layers"
            ]
            require(
                len(layers) == 12
                and all(
                    layer["rank"] == 4
                    and layer["weight_shape"] == [384, 384]
                    and layer["scale"] == 1.0
                    for layer in layers
                ),
                "The registered physical update coordinate scale changed",
            )
        warmup_path = (
            args.private_root
            / f"fold{unit['fold_id']}"
            / f"seed{unit['seed']}"
            / "warmup/CHECKPOINT_SELECTED.pt"
        )
        if unit["stage"] != "warmup":
            warmup = torch.load(warmup_path, map_location="cpu", weights_only=False)
            require(
                bh_hash == bridge_head_hash(warmup["model"]),
                "Bridge/Head changed after fold-local warmup",
            )
            parent_path = warmup_path
            if unit["stage"] == "static":
                basis_key = (unit["fold_id"], unit["seed"])
                if basis_key not in basis_cache:
                    basis_cache[basis_key] = torch.load(
                        args.private_root
                        / f"fold{unit['fold_id']}"
                        / f"seed{unit['seed']}"
                        / "bases/BASES.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                audit_static_bases(
                    state, unit["method"], basis_cache[basis_key], official, svd_cache
                )
            if unit["stage"] == "condition":
                parent_path = (
                    args.private_root
                    / f"fold{unit['fold_id']}"
                    / f"seed{unit['seed']}"
                    / "static"
                    / unit["method"]
                    / "CHECKPOINT_SELECTED.pt"
                )
                static = torch.load(
                    parent_path, map_location="cpu", weights_only=False
                )["model"]
                prefixes = [
                    key.removesuffix(".delta_core")
                    for key in state
                    if key.endswith(".delta_core")
                ]
                require(
                    len(prefixes) == 12,
                    "Conditional source requires twelve delta matrices",
                )
                for prefix in prefixes:
                    for suffix in (".U", ".V", ".core"):
                        require(
                            torch.equal(
                                state[prefix + suffix], static[prefix + suffix]
                            ),
                            "Static adapter start changed",
                        )
                    require(
                        torch.equal(
                            state[prefix + ".source_core"], static[prefix + ".core"]
                        ),
                        "The frozen C0 differs across view or radius",
                    )
                    require(
                        not bool(state[prefix + ".target_active"]),
                        "Target adaptation flag entered source fitting",
                    )
                    weight = official[
                        prefix.removeprefix("bridge.backbone.module.") + ".weight"
                    ]
                    reference = registration["reference_weight_fraction"] * float(
                        weight.double().norm()
                    )
                    np.testing.assert_allclose(
                        float(state[prefix + ".core_norm_reference"]),
                        reference,
                        rtol=1e-12,
                        atol=1e-14,
                    )
                    require(
                        float(state[prefix + ".core_radius_multiplier"])
                        == unit["radius"],
                        "Physical update radius changed",
                    )
                    for side in ("left", "right"):
                        factor = state[prefix + f".core_{side}_factor"].double()
                        basis = state[
                            prefix + (".U" if side == "left" else ".V")
                        ].double()
                        torch.testing.assert_close(
                            factor.T @ factor, basis.T @ basis, rtol=1e-10, atol=1e-10
                        )
            require(
                spec["parent_checkpoint_sha256"] == digest(parent_path),
                "Source continuation parent changed",
            )
        score, maximum, count = recompute_metrics(
            folder / "SOURCE_VALIDATION_PREDICTIONS.npz",
            folder / "SOURCE_VALIDATION_CELL_METRICS.csv",
            fold["inner_validation_ids"],
            data,
        )
        for key, value in score.items():
            require(
                abs(float(receipt["validation"][key]) - value) < 1e-11,
                "Inner summary differs from raw NPZ",
            )
        train_state = torch.load(
            folder / "TRAIN_STATE.pt", map_location="cpu", weights_only=False
        )
        require(
            train_state["updates"] == expected_updates,
            "On-disk optimizer state update count differs",
        )
        steps = [
            int(value["step"])
            for value in train_state["optimizer"]["state"].values()
            if "step" in value
        ]
        require(
            steps and max(steps) == expected_updates,
            "The optimizer step counter lacks actual training",
        )
        if unit["stage"] != "warmup":
            require(
                bridge_head_hash(train_state["model"]) == bh_hash,
                "Final source Bridge/Head changed",
            )
        records.append(
            {
                "unit": unit,
                "updates": expected_updates,
                "trainable_parameters": expected_parameters,
                "checkpoint_sha256": digest(checkpoint_path),
                "bridge_head_sha256": bh_hash,
                "inner_cell_window_metrics_recomputed": count,
                "maximum_metric_difference": maximum,
            }
        )
        del checkpoint, train_state, state
    bases, missing_bases = basis_audit(args, protocol)
    total = sum(record["updates"] for record in records)
    complete = not missing and not missing_bases
    require(
        complete or not (args.public_root / "OUTER_EVALUATION_OPENED.json").exists(),
        "Outer evaluation was opened before the registered source matrix completed",
    )
    if complete:
        require(
            total == registration["expected_optimizer_updates"] == 207360,
            "Whole source update ledger differs",
        )
    if args.require_complete:
        require(complete, "The source matrix is incomplete")
    return {
        "status": "PASSED" if complete else "PARTIAL",
        "completed_units": len(records),
        "expected_units": 216,
        "completed_bases": len(bases),
        "expected_bases": 9,
        "optimizer_updates": total,
        "expected_optimizer_updates": 207360,
        "missing_units": missing,
        "missing_bases": missing_bases,
        "records": records,
        "bases": bases,
        "final_target_query_labels_used": 0,
        "outer_labels_used_for_training_or_selection": 0,
    }


def normalized_views(data, fold, manifest):
    require(
        manifest["fit_ids"]
        == manifest["normalization"]["source_ids"]
        == fold["fit_ids"],
        "Conditioning normalization used a non-fit identity",
    )
    selected = fold["fit_ids"] + fold["inner_validation_ids"] + fold["outer_test_ids"]
    rows = data.rows(selected)
    descriptors = prefix_view_descriptors(
        data.x[rows], data.point_mask[rows], data.cycle_ids[rows]
    )
    result = {}
    for view in ("absolute", "relative"):
        fit_values = descriptors[view][: len(fold["fit_ids"])]
        fit_masks = descriptors[view + "_mask"][: len(fold["fit_ids"])]
        mean, std, _ = independent_moments(
            fit_values, fit_masks, (0,), 1e-3 if view == "absolute" else 1e-4
        )
        stored = manifest["normalization"]["statistics"][view]
        np.testing.assert_allclose(stored["mean"], mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(stored["scale"], std, rtol=1e-12, atol=1e-12)
        values = np.where(
            descriptors[view + "_mask"], (descriptors[view] - mean) / std, 0
        ).astype(np.float32)
        result[view] = {uid: values[index] for index, uid in enumerate(selected)}
    return result


def condition_norms(state, features, ids, mode, evaluation_mode):
    """Reconstruct physical update norms from saved tensors using NumPy only."""
    output = []
    absolute = np.stack([features["absolute"][uid] for uid in ids])
    relative = np.stack([features["relative"][uid] for uid in ids])
    if mode in ("constant", "relative"):
        absolute = np.ones_like(absolute)
    if mode in ("constant", "absolute"):
        relative = np.ones_like(relative)
    prefixes = [
        key.removesuffix(".delta_core") for key in state if key.endswith(".delta_core")
    ]
    require(len(prefixes) == 12, "Conditional norm audit requires twelve adapters")
    for index, prefix in enumerate(prefixes):
        left = state[prefix + ".U"].numpy().astype(np.float64)
        right = state[prefix + ".V"].numpy().astype(np.float64)
        ru, rv = (
            np.linalg.qr(left, mode="reduced")[1],
            np.linalg.qr(right, mode="reduced")[1],
        )
        core = state[prefix + ".source_core"].numpy()
        delta = state[prefix + ".delta_core"].numpy()
        gates = []
        for name, values in (("absolute", absolute), ("relative", relative)):
            base = f"bridge.backbone.module.core_gates.{name}.{index}"
            weight, bias = (
                state[base + ".weight"].numpy(),
                state[base + ".bias"].numpy(),
            )
            gates.append(1 + np.float32(0.5) * np.tanh(values @ weight.T + bias))
        raw = (
            (core + delta)[None] * gates[0][:, None, :] * gates[1][:, :, None]
        ).astype(np.float64) - core.astype(np.float64)[None]
        raw_norm = np.linalg.norm(ru @ raw @ rv.T, axis=(-2, -1))
        radius = float(
            state[prefix + ".core_norm_reference"]
            * state[prefix + ".core_radius_multiplier"]
        )
        factor = np.ones_like(raw_norm)
        if evaluation_mode == "matched_norm":
            np.divide(radius, raw_norm, out=factor, where=raw_norm > 0)
            factor[raw_norm == 0] = 0
        else:
            np.divide(radius, raw_norm, out=factor, where=raw_norm > radius)
            if radius == 0:
                factor.fill(0)
        increment = (raw * factor[:, None, None]).astype(np.float32)
        actual = np.linalg.norm(ru @ increment.astype(np.float64) @ rv.T, axis=(-2, -1))
        layer = prefix.removeprefix("bridge.backbone.module.")
        output.extend(
            {
                "cell_uid": uid,
                "layer": layer,
                "independent_increment_norm": float(actual[position]),
                "independent_raw_increment_norm": float(raw_norm[position]),
                "radius": radius,
            }
            for position, uid in enumerate(ids)
        )
    return pd.DataFrame(output)


def audit_norm_table(table, evaluation_mode):
    require(
        not table.duplicated(["cell_uid", "layer"]).any(),
        "Repeated update-norm cell/layer",
    )
    required = [
        "raw_increment_norm",
        "increment_norm",
        "radius",
        "norm_reference",
        "total_update_norm",
        "baseline_update_norm",
    ]
    require(
        np.isfinite(table[required].to_numpy()).all(),
        "Nonfinite physical update diagnostic",
    )
    for name in ("matched_norm_unattainable", "zero_increment", "cap_active"):
        require(
            table[name].isin([True, False]).all(), "Invalid boolean update support flag"
        )
    require(
        (table[required] >= 0).all().all() and table.radius.gt(0).all(),
        "Negative or zero physical norm allowance",
    )
    require(
        (table.increment_norm <= table.radius * (1 + 3e-6) + 1e-10).all(),
        "Actual update exceeds its physical radius",
    )
    if evaluation_mode == "matched_norm":
        attainable = ~table.matched_norm_unattainable.astype(bool)
        np.testing.assert_allclose(
            table.loc[attainable, "increment_norm"],
            table.loc[attainable, "radius"],
            rtol=3e-6,
            atol=1e-10,
        )
        require(
            table.loc[~attainable, "increment_norm"].eq(0).all(),
            "Unattainable update was filled artificially",
        )
    return {
        "rows": len(table),
        "unattainable_rows": int(table.matched_norm_unattainable.sum()),
        "actual_zero_rows": int(table.zero_increment.sum()),
        "cap_active_rows": int(table.cap_active.sum()),
        "maximum_radius_ratio": float((table.increment_norm / table.radius).max()),
    }


def outer_audit(args, registration, protocol, data):
    records, missing = [], []
    for fold in protocol["folds"]:
        manifest = read(
            args.public_root / f"fold{fold['fold_id']}/FOLD_VIEW_MANIFEST.json"
        )
        features = normalized_views(data, fold, manifest)
        for unit in units(registration, {"folds": [fold]}):
            if unit["stage"] == "warmup":
                continue
            modes = (
                ("static",)
                if unit["stage"] == "static"
                else ("trained_ball", "matched_norm")
            )
            selected = unit_path(args.private_root, unit) / "CHECKPOINT_SELECTED.pt"
            for mode in modes:
                folder = unit_path(args.private_root / "outer", unit) / mode
                marker = folder / "OUTER_COMPLETE.json"
                if not marker.exists():
                    missing.append({"unit": unit, "evaluation_mode": mode})
                    continue
                receipt = read(marker)
                verify_files(folder, receipt["files"])
                require(
                    receipt["unit"] == unit and receipt["evaluation_mode"] == mode,
                    "Outer configuration changed",
                )
                require(
                    receipt["outer_ids"] == fold["outer_test_ids"]
                    and receipt["outer_groups"] == fold["outer_test_groups"],
                    "Outer physical/group role changed",
                )
                require(
                    receipt["checkpoint_sha256"] == digest(selected),
                    "Outer model differs from inner-selected checkpoint",
                )
                require(
                    receipt["final_target_query_labels_used"]
                    == receipt["optimizer_updates"]
                    == 0,
                    "Outer evaluation adapted or used final targets",
                )
                require(
                    receipt["prediction_replay_max_absolute_error"] == 0,
                    "Outer serialized replay failed",
                )
                summary, maximum, count = recompute_metrics(
                    folder / "OUTER_PREDICTIONS.npz",
                    folder / "OUTER_CELL_METRICS.csv",
                    fold["outer_test_ids"],
                    data,
                )
                for key, value in summary.items():
                    require(
                        abs(value - receipt["metrics"][key]) < 1e-11,
                        "Outer summary differs from raw NPZ",
                    )
                norm = None
                if unit["stage"] == "condition":
                    table = pd.read_csv(folder / "UPDATE_DIAGNOSTICS.csv")
                    require(
                        len(table) == 12 * len(fold["outer_test_ids"]),
                        "Incomplete outer physical norm grid",
                    )
                    require(
                        set(table.cell_uid) == set(fold["outer_test_ids"]),
                        "Norm diagnostics used another role",
                    )
                    norm = audit_norm_table(table, mode)
                    state = torch.load(
                        selected, map_location="cpu", weights_only=False
                    )["model"]
                    independent = condition_norms(
                        state, features, fold["outer_test_ids"], unit["mode"], mode
                    )
                    compared = table.merge(
                        independent,
                        on=["cell_uid", "layer"],
                        validate="one_to_one",
                        suffixes=("", "_independent"),
                    )
                    np.testing.assert_allclose(
                        compared.increment_norm,
                        compared.independent_increment_norm,
                        rtol=5e-4,
                        atol=2e-7,
                    )
                    np.testing.assert_allclose(
                        compared.raw_increment_norm,
                        compared.independent_raw_increment_norm,
                        rtol=5e-4,
                        atol=2e-7,
                    )
                    np.testing.assert_allclose(
                        compared.radius,
                        compared.radius_independent,
                        rtol=1e-12,
                        atol=1e-14,
                    )
                    norm["independent_actual_norm_maximum_difference"] = float(
                        (compared.increment_norm - compared.independent_increment_norm)
                        .abs()
                        .max()
                    )
                records.append(
                    {
                        "unit": unit,
                        "evaluation_mode": mode,
                        "cell_window_metrics_recomputed": count,
                        "maximum_metric_difference": maximum,
                        "norm": norm,
                    }
                )
    require(
        not missing and len(records) == 387,
        "The registered outer evaluation matrix is incomplete",
    )
    return {
        "status": "PASSED",
        "completed_outer_units": len(records),
        "expected_outer_units": 387,
        "final_target_query_labels_used": 0,
        "optimizer_updates": 0,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("source", "outer", "all"), default="source")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/drs_peft/e123_v1.json"
    )
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
    parser.add_argument(
        "--backbone-dir",
        type=Path,
        default=ROOT.parents[1] / ".cache_drs_peft/moirai_small",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    registration, protocol = read(args.config), read(args.folds)
    data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
    require(
        digest(args.data_root / "MATR/COMPLETE.json")
        == protocol["provenance"]["source_cache_complete_sha256"],
        "Frozen source-cache manifest changed",
    )
    verify_protocol(registration, protocol, data)
    args.official_weight_sha256 = digest(args.backbone_dir / "model.safetensors")
    official = load_file(str(args.backbone_dir / "model.safetensors"), device="cpu")
    source = source_audit(args, registration, protocol, data, official)
    recovery = recovery_provenance(args)
    if source["status"] == "PASSED" and recovery["amendment_present"]:
        require(
            not recovery["running_invocations"],
            "A launcher invocation is still writing completion provenance",
        )
    result = {
        "source": source,
        "source_sha256": digest(__file__),
        "model_sha256": MODEL_SHA,
        "config_sha256": digest(args.config),
        "folds_sha256": digest(args.folds),
        "audit_device": "CPU",
        "io_recovery": recovery,
    }
    if args.phase in ("outer", "all"):
        require(
            source["status"] == "PASSED",
            "Outer audit requires the complete source training matrix",
        )
        result["outer"] = outer_audit(args, registration, protocol, data)
    result["status"] = source["status"]
    output = args.public_root / f"INDEPENDENT_{args.phase.upper()}_AUDIT.json"
    if source["status"] == "PARTIAL":
        output = args.private_root / "audit" / "PARTIAL_SOURCE_AUDIT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_source_units": source["completed_units"],
                "completed_bases": source["completed_bases"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
