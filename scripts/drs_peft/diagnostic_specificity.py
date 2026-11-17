"""D1 source-only supervision and gradient-subspace specificity diagnostic.

The original P1 arrays remain read-only. Capacity targets are diagnostic
pseudo-supervision. Source-meta capture always uses actual future labels.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from threadpoolctl import threadpool_limits

from .data import ROOT, file_hash, write_json
from .gradient_audit import (
    _collect,
    _covariances,
    _group_balanced_weights,
    _group_indices,
    _overlap,
    _replay_gradients,
    _role_fingerprint,
    _svd,
    _top,
    state_fingerprint,
)
from .run_p1 import load_p0_checkpoint
from .train import (
    bind,
    configure,
    load_selected,
    load_source,
    make_model,
    predict,
    read_json,
)

PREFIX_AUDIT = Path("data/unified_matr_seed42/raw_record_audit.parquet")
PREFIX_AUDIT_SHA = "ddd4278741c14d5ed4d3db1e1161fc18b2fb25e92baaa910191176601b6acb0f"
NEW_VARIANTS = ("true_common", "shuffled_common", "flat_full", "linear_full")
PAIRS = (
    ("true_full", "shuffled_full"),
    ("true_common", "shuffled_common"),
    ("true_full", "flat_full"),
    ("true_full", "linear_full"),
    ("flat_full", "linear_full"),
    ("true_full", "true_common"),
    ("shuffled_full", "shuffled_common"),
)


def cosine(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.sum(left * right) / denominator) if denominator > 0 else None


def prefix_slopes(data, rows, audit_path=ROOT / PREFIX_AUDIT):
    """Read certified source prefixes through an Arrow predicate before fitting."""
    audit_path = Path(audit_path)
    if file_hash(audit_path) != PREFIX_AUDIT_SHA:
        raise RuntimeError("Frozen raw-record capacity audit changed")
    ids = [data.ids[row] for row in rows]
    columns = [
        "cell_uid",
        "original_cycle_id",
        "capacity_valid",
        "capacity_Ah",
        "instrument_axis_certified",
    ]
    frame = pd.read_parquet(
        audit_path,
        columns=columns,
        filters=[
            ("cell_uid", "in", ids),
            ("original_cycle_id", ">=", 1),
            ("original_cycle_id", "<=", 100),
        ],
    )
    if (
        not set(frame.cell_uid).issubset(ids)
        or not frame.original_cycle_id.between(1, 100).all()
    ):
        raise RuntimeError("Prefix-capacity reader crossed its source/prefix boundary")
    if frame.duplicated(["cell_uid", "original_cycle_id"]).any():
        raise RuntimeError("Ambiguous repeated physical prefix capacities")
    if not frame.instrument_axis_certified.all():
        raise RuntimeError("Prefix-capacity coordinate authentication failed")
    records = []
    for row in rows:
        selected = frame.loc[frame.cell_uid.eq(data.ids[row]) & frame.capacity_valid]
        x = selected.original_cycle_id.to_numpy(float)
        y = selected.capacity_Ah.to_numpy(float) / float(data.nominal_capacity_Ah[row])
        slope = fit_prefix_slope(x, y)
        records.append(
            {
                "cell_uid": data.ids[row],
                "observations": len(x),
                "minimum_cycle": float(x.min()),
                "maximum_cycle": float(x.max()),
                "slope_SOH_per_cycle": slope,
                "anchor_soh100": float(data.anchor[row]),
            }
        )
    return pd.DataFrame(records), {
        "path": str(audit_path.resolve()),
        "sha256": PREFIX_AUDIT_SHA,
        "parquet_filters": [
            ["cell_uid", "in", ids],
            ["original_cycle_id", ">=", 1],
            ["original_cycle_id", "<=", 100],
        ],
        "loaded_prefix_records": len(frame),
        "valid_prefix_records": int(frame.capacity_valid.sum()),
        "model_input_usage": "none; capacities only create diagnostic pseudo-targets",
    }


def fit_prefix_slope(cycles, capacity):
    cycles, capacity = np.asarray(cycles, float), np.asarray(capacity, float)
    if (
        len(cycles) < 80
        or not np.isfinite(cycles).all()
        or not np.isfinite(capacity).all()
    ):
        raise ValueError(
            "Prefix OLS requires at least 80 finite certified observations"
        )
    if not ((cycles >= 1) & (cycles <= 100)).all() or len(np.unique(cycles)) != len(
        cycles
    ):
        raise ValueError("Prefix OLS requires unique original cycles in 1..100")
    centered = cycles - cycles.mean()
    return float(
        np.dot(centered, capacity - capacity.mean()) / np.dot(centered, centered)
    )


def supervision_arrays(data, rows, donors, slopes):
    """Build receiver-indexed targets, keeping the original arrays immutable."""
    truth = np.asarray(data.targets[rows])
    recipient = np.asarray(data.observed[rows])
    donated = np.asarray(data.targets[donors])
    donor_mask = np.asarray(data.observed[donors])
    common = recipient & donor_mask
    if not common.any(1).all():
        raise ValueError("A matched-label pair has no common actual observation")
    flat = np.broadcast_to(data.anchor[rows, None], truth.shape).copy()
    linear = flat + np.asarray(slopes)[:, None] * (data.query[None] - 100)
    return {
        "true_full": (truth, recipient),
        "shuffled_full": (donated, donor_mask),
        "true_common": (truth, common),
        "shuffled_common": (donated, common),
        "flat_full": (flat, recipient),
        "linear_full": (linear.astype(np.float32), recipient),
    }


def coefficients(prediction, target, observed):
    """Exact derivative of a physical-cell mean observed MAE wrt its outputs."""
    counts = np.asarray(observed).sum(1)
    if not (counts > 0).all():
        raise ValueError("Output coefficients need nonempty actual observation masks")
    return np.where(observed, np.sign(prediction - target) / counts[:, None], 0)


def output_diagnostics(data, rows, donors, prediction, supervision, seed):
    derivatives = {
        name: coefficients(prediction, target, mask)
        for name, (target, mask) in supervision.items()
    }
    records = []
    for left, right in PAIRS:
        left_target, left_mask = supervision[left]
        right_target, right_mask = supervision[right]
        for i, row in enumerate(rows):
            common = left_mask[i] & right_mask[i]
            exclusive = left_mask[i] ^ right_mask[i]
            delta = derivatives[right][i] - derivatives[left][i]
            denominator = float(np.square(delta).sum())
            records.append(
                {
                    "seed": seed,
                    "cell_uid": data.ids[row],
                    "protocol_uid": str(data.protocol_ids[row]),
                    "outcome_donor_uid": data.ids[donors[i]],
                    "left": left,
                    "right": right,
                    "left_observations": int(left_mask[i].sum()),
                    "right_observations": int(right_mask[i].sum()),
                    "common_observations": int(common.sum()),
                    "exact_mask_match": bool(
                        np.array_equal(left_mask[i], right_mask[i])
                    ),
                    "output_coefficients_identical": bool(
                        np.array_equal(derivatives[left][i], derivatives[right][i])
                    ),
                    "output_coefficient_cosine": cosine(
                        derivatives[left][i], derivatives[right][i]
                    ),
                    "common_residual_sign_agreement": float(
                        np.mean(
                            np.sign(prediction[i, common] - left_target[i, common])
                            == np.sign(prediction[i, common] - right_target[i, common])
                        )
                    ),
                    "coefficient_delta_common_energy_fraction": (
                        float(np.square(delta[common]).sum() / denominator)
                        if denominator
                        else 0.0
                    ),
                    "coefficient_delta_exclusive_mask_energy_fraction": (
                        float(np.square(delta[exclusive]).sum() / denominator)
                        if denominator
                        else 0.0
                    ),
                    "left_label_mean": float(left_target[i, left_mask[i]].mean()),
                    "right_label_mean": float(right_target[i, right_mask[i]].mean()),
                }
            )
    return pd.DataFrame(records)


def checked_p1_arrays(p1_root, seed, source_ids, meta_ids):
    path = Path(p1_root) / f"seed{seed}" / "P1_CACHE_MANIFEST.json"
    manifest = read_json(path)
    arrays, identities = {}, {}
    for key, variant, expected in (
        ("source/nominal", "true_full", source_ids),
        ("source_shuffled/nominal", "shuffled_full", source_ids),
        ("meta_validation/nominal", "meta_true", meta_ids),
    ):
        receipt = manifest[key]
        if receipt["identity"]["ids"] != expected:
            raise RuntimeError(
                "P1 gradient identity/order differs from frozen source roles"
            )
        folder = Path(receipt["folder"])
        complete = read_json(folder / "COMPLETE.json")
        if (
            complete["identity"] != receipt["identity"]
            or complete["files"] != receipt["files"]
        ):
            raise RuntimeError(
                "P1 compact manifest differs from its complete cache receipt"
            )
        for record in receipt["files"]:
            if file_hash(folder / record["path"]) != record["sha256"]:
                raise RuntimeError("Read-only P1 gradient cache bytes changed")
        arrays[variant] = {
            layer: np.load(folder / record["path"], mmap_mode="r")
            for layer, record in complete["layer_files"].items()
        }
        identities[variant] = receipt
    return arrays, identities, {"path": str(path.resolve()), "sha256": file_hash(path)}


def analyze(data, rows, meta_rows, arrays, seed, output):
    groups = _group_indices(data.protocol_ids[rows])
    weights = _group_balanced_weights(groups, len(rows))
    names = list(arrays["true_full"])
    gradient_records, overlap_records, capture_records = [], [], []
    variants = ["true_full", "shuffled_full", *NEW_VARIANTS]
    for layer_index, layer in enumerate(names):
        values = {
            key: np.asarray(arrays[key][layer], dtype=np.float64) for key in arrays
        }
        for left, right in PAIRS:
            for i, row in enumerate(rows):
                a, b = values[left][i], values[right][i]
                gradient_records.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "cell_uid": data.ids[row],
                        "protocol_uid": str(data.protocol_ids[row]),
                        "left": left,
                        "right": right,
                        "gradient_cosine": cosine(a, b),
                        "left_norm": float(np.linalg.norm(a)),
                        "right_norm": float(np.linalg.norm(b)),
                        "right_to_left_norm_ratio": (
                            float(np.linalg.norm(b) / np.linalg.norm(a))
                            if np.linalg.norm(a) > 0
                            else None
                        ),
                    }
                )
        bases = {}
        for variant in variants:
            gradient = values[variant]
            left, right, _, _ = _covariances(gradient, weights)
            u, _ = _top(left, 56)
            v, _ = _top(right, 56)
            mean = np.tensordot(weights, gradient, axes=1)
            um, vm, _ = _svd(mean, 56)
            bases[(variant, "gradient_cov")] = (u, v)
            bases[(variant, "gradient_mean_svd")] = (um, vm)
        for kind in ("gradient_cov", "gradient_mean_svd"):
            for left, right in combinations(variants, 2):
                for rank in (4, 56):
                    u1, v1 = bases[(left, kind)]
                    u2, v2 = bases[(right, kind)]
                    lo, lm, lx = _overlap(u1[:, :rank], u2[:, :rank])
                    ro, rm, rx = _overlap(v1[:, :rank], v2[:, :rank])
                    overlap_records.append(
                        {
                            "seed": seed,
                            "layer": layer,
                            "basis_kind": kind,
                            "left": left,
                            "right": right,
                            "rank": rank,
                            "left_overlap": lo,
                            "right_overlap": ro,
                            "tensor_product_overlap": lo * ro,
                            "left_mean_principal_angle_deg": lm,
                            "right_mean_principal_angle_deg": rm,
                            "left_max_principal_angle_deg": lx,
                            "right_max_principal_angle_deg": rx,
                        }
                    )
            for variant in variants:
                u, v = bases[(variant, kind)]
                for role, role_rows in (
                    ("source", rows),
                    ("meta_validation", meta_rows),
                ):
                    gradient = values["true_full" if role == "source" else "meta_true"]
                    denominator = np.square(gradient).sum(axis=(1, 2))
                    projection = np.matmul(np.matmul(u.T[None], gradient), v[None])
                    for rank in (4, 56):
                        captured = np.square(projection[:, :rank, :rank]).sum(
                            axis=(1, 2)
                        )
                        for i, row in enumerate(role_rows):
                            capture_records.append(
                                {
                                    "seed": seed,
                                    "layer": layer,
                                    "role": role,
                                    "cell_uid": data.ids[row],
                                    "protocol_uid": str(data.protocol_ids[row]),
                                    "basis_supervision": variant,
                                    "basis_kind": kind,
                                    "rank": rank,
                                    "capture": (
                                        float(captured[i] / denominator[i])
                                        if denominator[i] > 0
                                        else None
                                    ),
                                }
                            )
        print(
            json.dumps(
                {
                    "event": "D1_LAYER_ANALYZED",
                    "seed": seed,
                    "completed": layer_index + 1,
                    "total": len(names),
                }
            ),
            flush=True,
        )
    for name, records in (
        ("D1_GRADIENT_PAIR_METRICS.csv", gradient_records),
        ("D1_SUBSPACE_OVERLAP.csv", overlap_records),
        ("D1_TRUE_GRADIENT_CAPTURE.csv", capture_records),
    ):
        pd.DataFrame(records).to_csv(output / name, index=False)


def run_seed(args, seed):
    output = args.output_root / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "D1_COMPLETE.json"
    registration = read_json(args.registration)
    data, early, _, episodes = load_source(args.data_root)
    source_ids = episodes["source_cell_roles"]["source_train"]
    meta_ids = episodes["source_cell_roles"]["source_meta_validation"]
    rows, meta_rows = data.rows(source_ids), data.rows(meta_ids)
    arrays, receipts, p1_manifest = checked_p1_arrays(
        args.p1_root, seed, source_ids, meta_ids
    )
    donors = data.rows(receipts["shuffled_full"]["identity"]["outcome_donors"])
    if (rows == donors).any() or set(rows) != set(donors):
        raise RuntimeError("Frozen P1 shuffle must be a source-only derangement")
    slopes, capacity_manifest = prefix_slopes(data, rows)
    supervision = supervision_arrays(
        data, rows, donors, slopes.slope_SOH_per_cycle.to_numpy()
    )
    bridge = read_json(args.results_root / "P0/BRIDGE_SELECTION.json")[
        "selected_bridge"
    ]
    checkpoint, checkpoint_path = load_p0_checkpoint(args.results_root, bridge, seed)
    identity = {
        "stage": "D1_specificity",
        "seed": seed,
        "registration": {
            "path": str(args.registration.resolve()),
            "sha256": file_hash(args.registration),
        },
        "registration_content": registration,
        "implementation_sha256": file_hash(__file__),
        "source_ids": source_ids,
        "source_meta_validation_ids": meta_ids,
        "source_protocols": sorted(set(data.protocol_ids[rows].tolist())),
        "source_meta_protocols": sorted(set(data.protocol_ids[meta_rows].tolist())),
        "source_data_sha256": _role_fingerprint(data, early, rows, rows),
        "source_checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": file_hash(checkpoint_path),
        },
        "p1_cache_manifest": p1_manifest,
        "capacity_prefix_source": capacity_manifest,
        "new_supervision": list(NEW_VARIANTS),
        "ranks": [4, 56],
        "optimizer_updates": 0,
        "final_target_Query_future_labels_accessed": False,
        "pseudo_target_formulas": {
            "flat_full": "anchor_soh100",
            "linear_full": "anchor_soh100 + OLS(source certified cycles1..100 capacity_Ah/Qnom, cycle)*(query_cycle-100); no clipping",
            "true_common": "receiver real future on receiver/donor observed-mask intersection",
            "shuffled_common": "P1 deranged donor real future on same receiver/donor mask intersection",
        },
    }
    bind(output / "D1_REGISTRATION.json", identity)
    source_files = [
        Path(__file__).resolve(),
        args.registration.resolve(),
        ROOT / "scripts/drs_peft/gradient_audit.py",
        ROOT / "scripts/drs_peft/run_p1.py",
        ROOT / "scripts/drs_peft/train.py",
        ROOT / "scripts/drs_peft/data.py",
        ROOT / "models/drs_peft/backbone.py",
        ROOT / "models/drs_peft/bridges.py",
        ROOT / "models/drs_peft/trajectory.py",
    ]
    bind(
        output / "D1_SOURCE_SNAPSHOT.json",
        {
            "files": [
                {"path": str(path), "sha256": file_hash(path)} for path in source_files
            ],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda")
            else "cpu",
        },
    )
    if marker.exists():
        receipt = read_json(marker)
        for record in receipt["files"]:
            if file_hash(output / record["path"]) != record["sha256"]:
                raise RuntimeError("Completed D1 public result bytes changed")
        for receipt in read_json(output / "D1_NEW_CACHE_MANIFEST.json").values():
            for record in receipt["files"]:
                if (
                    file_hash(Path(receipt["folder"]) / record["path"])
                    != record["sha256"]
                ):
                    raise RuntimeError("Completed D1 gradient cache bytes changed")
        return
    started = time.monotonic()
    model = make_model(
        args.backbone_dir,
        early["policy"].shape[-1],
        bridge,
        "B0_pretrained_frozen",
        seed,
    ).to(args.device)
    load_selected(model, checkpoint["model"])
    state = state_fingerprint(model)
    if state != receipts["true_full"]["identity"]["model_sha256"]:
        raise RuntimeError("D1 model differs from immutable P1 B0 checkpoint")
    layers = {
        name: module
        for name, module in model.backbone.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj")
    }
    if list(layers) != list(arrays["true_full"]) or len(layers) != 12:
        raise RuntimeError("D1 target matrices differ from frozen P1 QV family")
    prediction = predict(model, early, rows, data.query, args.device, batch_size=1)
    output_diagnostics(data, rows, donors, prediction, supervision, seed).to_csv(
        output / "D1_OUTPUT_COEFFICIENTS.csv", index=False
    )
    slopes.to_csv(output / "D1_PREFIX_CAPACITY_SLOPES.csv", index=False)
    new_receipts, replay = {}, {}
    for variant in NEW_VARIANTS:
        target, mask = supervision[variant]
        target_array = np.zeros_like(data.targets)
        observed_array = np.zeros_like(data.observed)
        target_array[rows], observed_array[rows] = target, mask
        diagnostic_data = replace(data, targets=target_array, observed=observed_array)
        gradient, receipt = _collect(
            model,
            diagnostic_data,
            early,
            rows,
            rows,
            layers,
            args.private_root / f"seed{seed}",
            seed,
            "D1_" + variant,
            "nominal",
            args.device,
            state,
        )
        arrays[variant] = gradient
        new_receipts[variant] = dict(
            receipt,
            target_storage_semantics="receiver-indexed diagnostic targets; internal donor index refers to receiver storage",
            actual_label_origin_uids=[
                data.ids[row]
                for row in (donors if variant == "shuffled_common" else rows)
            ],
        )
        replay[variant] = _replay_gradients(
            model,
            diagnostic_data,
            early,
            [("D1_" + variant, "nominal", rows, rows)],
            {("D1_" + variant, "nominal"): gradient},
            layers,
            args.device,
        )
    write_json(output / "D1_NEW_CACHE_MANIFEST.json", new_receipts)
    write_json(output / "D1_GRADIENT_REPLAY.json", replay)
    if state_fingerprint(model) != state:
        raise RuntimeError("D1 changed the audited model state")
    del model, checkpoint
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    with threadpool_limits(limits=2):
        analyze(data, rows, meta_rows, arrays, seed, output)
    files = [
        {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != marker.name
    ]
    write_json(
        marker,
        {
            "status": "COMPLETE_SOURCE_ONLY_DIAGNOSTIC",
            "files": files,
            "new_physical_cell_autograd_evaluations": len(rows) * len(NEW_VARIANTS),
            "endpoint_replay_autograd_evaluations": 2 * len(NEW_VARIANTS),
            "new_forward_only_physical_cells": len(rows),
            "optimizer_updates": 0,
            "runtime_seconds": time.monotonic() - started,
            "final_target_Query_future_labels_accessed": False,
        },
    )


def aggregate(output_root):
    folders = [output_root / f"seed{seed}" for seed in (42, 123, 2026)]
    registrations, receipts = [], []
    for folder in folders:
        if not (folder / "D1_COMPLETE.json").exists():
            raise RuntimeError(
                "All three registered D1 seeds must finish before aggregation"
            )
        receipt = read_json(folder / "D1_COMPLETE.json")
        for record in receipt["files"]:
            if file_hash(folder / record["path"]) != record["sha256"]:
                raise RuntimeError(
                    "D1 aggregation encountered changed completed outputs"
                )
        registrations.append(read_json(folder / "D1_REGISTRATION.json"))
        receipts.append(receipt)
    reference = registrations[0]
    for item in registrations[1:]:
        for key in (
            "source_ids",
            "source_meta_validation_ids",
            "source_protocols",
            "source_meta_protocols",
            "registration",
            "implementation_sha256",
        ):
            if item[key] != reference[key]:
                raise RuntimeError(
                    "D1 aggregate seeds differ in their frozen study identity"
                )
    metrics = {}
    definitions = (
        (
            "D1_OUTPUT_COEFFICIENTS.csv",
            ["seed", "left", "right"],
            [
                "output_coefficient_cosine",
                "common_residual_sign_agreement",
                "coefficient_delta_exclusive_mask_energy_fraction",
            ],
        ),
        (
            "D1_GRADIENT_PAIR_METRICS.csv",
            ["seed", "left", "right"],
            ["gradient_cosine", "right_to_left_norm_ratio"],
        ),
        (
            "D1_SUBSPACE_OVERLAP.csv",
            ["seed", "basis_kind", "left", "right", "rank"],
            [
                "left_overlap",
                "right_overlap",
                "tensor_product_overlap",
                "left_mean_principal_angle_deg",
                "right_mean_principal_angle_deg",
            ],
        ),
        (
            "D1_TRUE_GRADIENT_CAPTURE.csv",
            ["seed", "role", "basis_supervision", "basis_kind", "rank"],
            ["capture"],
        ),
    )
    for filename, grouping, columns in definitions:
        frame = pd.concat(
            [pd.read_csv(folder / filename) for folder in folders], ignore_index=True
        )
        if "cell_uid" in frame:
            cell = frame.groupby(
                [*grouping, "cell_uid", "protocol_uid"], as_index=False
            )[columns].mean()
            mean = cell.groupby(grouping, as_index=False)[columns].mean()
            protocol = cell.groupby([*grouping, "protocol_uid"], as_index=False)[
                columns
            ].mean()
            macro = protocol.groupby(grouping, as_index=False)[columns].mean()
            macro = macro.rename(
                columns={column: column + "_protocol_macro" for column in columns}
            )
            mean = mean.merge(macro, on=grouping)
        else:
            mean = frame.groupby(grouping, as_index=False)[columns].mean()
        output = filename.replace(".csv", "_SUMMARY.csv")
        mean.to_csv(output_root / output, index=False)
        metrics[filename] = {"rows": len(frame), "summary_rows": len(mean)}
    coefficients_table = pd.concat(
        [pd.read_csv(folder / "D1_OUTPUT_COEFFICIENTS.csv") for folder in folders],
        ignore_index=True,
    )
    gradient_table = pd.concat(
        [pd.read_csv(folder / "D1_GRADIENT_PAIR_METRICS.csv") for folder in folders],
        ignore_index=True,
    )
    joined = gradient_table.merge(
        coefficients_table,
        on=["seed", "cell_uid", "protocol_uid", "left", "right"],
        validate="many_to_one",
    )
    association_records = []
    for keys, group in joined.groupby(["seed", "left", "right", "layer"]):
        first, second = group.output_coefficient_cosine, group.gradient_cosine
        finite = first.notna() & second.notna()
        variable = first[finite].nunique() > 1 and second[finite].nunique() > 1
        association_records.append(
            {
                "seed": keys[0],
                "left": keys[1],
                "right": keys[2],
                "layer": keys[3],
                "physical_cells": int(finite.sum()),
                "coefficient_gradient_cosine_spearman": (
                    float(first[finite].corr(second[finite], method="spearman"))
                    if variable
                    else None
                ),
                "coefficient_gradient_cosine_pearson": (
                    float(first[finite].corr(second[finite])) if variable else None
                ),
                "coefficient_cosine_unique_values": int(first[finite].nunique()),
                "gradient_cosine_unique_values": int(second[finite].nunique()),
            }
        )
    pd.DataFrame(association_records).to_csv(
        output_root / "D1_SIGN_DIRECTION_ASSOCIATION.csv", index=False
    )
    cell_mean = joined.groupby(
        [
            "seed",
            "left",
            "right",
            "cell_uid",
            "protocol_uid",
            "output_coefficients_identical",
            "exact_mask_match",
        ],
        as_index=False,
    )[
        ["gradient_cosine", "right_to_left_norm_ratio", "output_coefficient_cosine"]
    ].mean()
    conditioned = (
        cell_mean.groupby(
            [
                "seed",
                "left",
                "right",
                "output_coefficients_identical",
                "exact_mask_match",
            ]
        )
        .agg(
            physical_cells=("cell_uid", "nunique"),
            protocol_groups=("protocol_uid", "nunique"),
            mean_gradient_cosine=("gradient_cosine", "mean"),
            mean_right_to_left_norm_ratio=("right_to_left_norm_ratio", "mean"),
            mean_output_coefficient_cosine=("output_coefficient_cosine", "mean"),
        )
        .reset_index()
    )
    conditioned.to_csv(
        output_root / "D1_SIGN_CONDITIONED_GRADIENT_SUMMARY.csv", index=False
    )
    write_json(
        output_root / "D1_SUMMARY.json",
        {
            "status": "COMPLETE_SOURCE_ONLY_DIAGNOSTIC",
            "seeds": [42, 123, 2026],
            "source_cells": len(reference["source_ids"]),
            "source_protocol_groups": len(reference["source_protocols"]),
            "source_meta_validation_cells": len(
                reference["source_meta_validation_ids"]
            ),
            "source_meta_protocol_groups": len(reference["source_meta_protocols"]),
            "new_autograd_evaluations": sum(
                receipt["new_physical_cell_autograd_evaluations"]
                + receipt["endpoint_replay_autograd_evaluations"]
                for receipt in receipts
            ),
            "new_forward_only_physical_cells": sum(
                receipt["new_forward_only_physical_cells"] for receipt in receipts
            ),
            "optimizer_updates": 0,
            "final_target_Query_future_labels_accessed": False,
            "outputs": metrics,
            "inferential_scope": "descriptive mechanism diagnosis on repeatedly used source development cohort; no causal attribution or final target performance claim",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/drs_peft")
    parser.add_argument("--p1-root", type=Path, default=ROOT / "runtime/drs_peft/P1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/drs_peft/diagnostics_v1/D1_specificity",
    )
    parser.add_argument(
        "--private-root",
        type=Path,
        default=ROOT / "runtime/drs_peft/diagnostics_v1/D1_specificity",
    )
    parser.add_argument("--backbone-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    configure()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        if args.backbone_dir is None:
            parser.error("--backbone-dir is required for gradient collection")
        for seed in [args.seed] if args.seed is not None else [42, 123, 2026]:
            with FileLock(str(args.output_root / f"seed{seed}.execution.lock")):
                run_seed(args, seed)
    if args.aggregate_only or args.seed is None:
        aggregate(args.output_root)


if __name__ == "__main__":
    main()
