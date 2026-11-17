"""Run a registered, source-only fixed-core dual-view modulation diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.drs_peft.diagnostic_dualview import (
    MODES,
    DualViewDiagnosticTSFM,
    SourceViewScaler,
    prefix_view_descriptors,
)
from models.drs_peft.peft import install_peft
from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.gradient_audit import state_fingerprint
from scripts.drs_peft.run_p1 import load_p0_checkpoint
from scripts.drs_peft.run_p2 import _basis_bundle, _publish_unit
from scripts.drs_peft.train import (
    ROOT,
    bind,
    configure,
    early_batch,
    load_selected,
    load_source,
    make_model,
    read_json,
    selected_state,
    train_source_unit,
)

SEEDS = (42, 123, 2026)
ALL_MODES = ("static",) + MODES
EXTRA_SOURCES = (
    "models/drs_peft/diagnostic_dualview.py",
    "scripts/drs_peft/diagnostic_dualview.py",
    "configs/drs_peft/diagnostics_v1.json",
)


def prepare_views(data, inputs, episodes):
    roles = episodes["source_cell_roles"]
    source = data.rows(roles["source_train"])
    meta = data.rows(roles["source_meta_validation"])
    selected = np.concatenate((source, meta))
    views = prefix_view_descriptors(
        inputs["x"][selected],
        inputs["point_mask"][selected],
        inputs["cycle_ids"][selected],
    )
    scaler = SourceViewScaler().fit(
        views, np.arange(len(source)), [data.ids[row] for row in source]
    )
    normalized = scaler.transform(views)
    arrays = {}
    for view in ("absolute", "relative"):
        # Target rows are never processed by the gate descriptor extractor.
        arrays[view] = np.zeros((len(data.ids), 12), dtype=np.float32)
        arrays[view][selected] = normalized[view]
    audit = {}
    for view in arrays:
        audit[view] = {}
        for role, segment in (
            ("source", slice(0, len(source))),
            ("meta", slice(len(source), None)),
        ):
            values = views[view][segment]
            masks = views[view + "_mask"][segment]
            audit[view][role] = {
                "mask_coverage": masks.mean(0).tolist(),
                "zero_fraction_of_supported": (
                    np.logical_and(masks, values == 0).sum(0)
                    / np.maximum(masks.sum(0), 1)
                ).tolist(),
                "unique_supported_values": [
                    len(np.unique(values[masks[:, column], column]))
                    for column in range(12)
                ],
                "constant_supported_features": [
                    bool(len(np.unique(values[masks[:, column], column])) <= 1)
                    for column in range(12)
                ],
            }
    manifest = {
        "source_ids": [data.ids[row] for row in source],
        "validation_ids": [data.ids[row] for row in meta],
        "final_target_descriptors_computed": 0,
        "source_only_scaler": scaler.to_dict(),
        "descriptor_sha256": {
            name: hashlib.sha256(value.tobytes()).hexdigest()
            for name, value in arrays.items()
        },
        "supported_features": {
            view: {
                role: views[view + "_mask"][
                    slice(0, len(source))
                    if role == "source"
                    else slice(len(source), None)
                ]
                .sum(0)
                .tolist()
                for role in ("source", "meta")
            }
            for view in arrays
        },
        "feature_audit": audit,
    }
    return arrays, manifest


def inputs_for_mode(inputs, arrays, episodes, data, mode, seed):
    result = dict(inputs)
    donors = np.arange(len(data.ids))
    mappings = {}
    if mode == "shuffled_dual":
        for role in ("source_train", "source_meta_validation"):
            rows = data.rows(episodes["source_cell_roles"][role])
            # A fixed nonzero circular shift of a seeded role-local permutation
            # is a bijection with no self-donors, including validation cells.
            order = np.random.default_rng(
                seed + (10001 if role == "source_train" else 20003)
            ).permutation(rows)
            donors[order] = np.roll(order, 1)
            mappings[role] = {data.ids[row]: data.ids[donors[row]] for row in rows}
            if any(donors[row] == row for row in rows):
                raise RuntimeError("Shuffled diagnostic retained a cell's own state")
    for view in ("absolute", "relative"):
        result["diagnostic_" + view] = arrays[view][donors]
    return result, mappings


def model_factory(backbone_dir, policy_dim, bridge, mode, seed, checkpoint, bundle):
    bases = {name: (u[:, :4], v[:, :4]) for name, (u, v) in bundle["bases"].items()}

    def construct():
        model = make_model(
            backbone_dir, policy_dim, bridge, "B0_pretrained_frozen", seed
        )
        load_selected(model, checkpoint["model"])
        if state_fingerprint(model) != bundle["source_checkpoint_sha256"]:
            raise RuntimeError("D3 model differs from the frozen P1 source checkpoint")
        install_peft(
            model.backbone,
            "grad_core",
            rank=4,
            alpha=4,
            dropout=0.0,
            bases=bases,
            seed=seed,
        )
        if mode == "static":
            return model
        model = DualViewDiagnosticTSFM(model, mode)
        saved = selected_state(model)
        gate_keys = {
            name
            for name in model.state_dict()
            if name.startswith("bridge.backbone.module.core_gates.")
        }
        if not gate_keys or not gate_keys.issubset(saved):
            raise RuntimeError("The frozen trainer would omit diagnostic gate state")
        gate_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if ".core_gates." in name
        ]
        if sum(
            parameter.numel() for _, parameter in gate_parameters
        ) != 1248 or not all(
            name.startswith("bridge.backbone.module.") for name, _ in gate_parameters
        ):
            raise RuntimeError("Gate parameter count or optimizer group changed")
        return model

    return construct


def initialization_check(factory, static_factory, inputs, rows, device):
    model = factory().to(device).eval()
    reference = static_factory().to(device).eval()
    batch = early_batch(inputs, rows[:2], device, model.input_keys)
    query = torch.tensor(
        [100, 101, 500, 1000, 2500, 5000], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        actual = model(batch, query)
        expected = reference(batch, query)
    if not torch.equal(actual, expected):
        raise RuntimeError("Unit-gate initialization changed the common static model")
    state = selected_state(model)
    reconstructed = factory().to(device).eval()
    load_selected(reconstructed, state)
    with torch.no_grad():
        replay = reconstructed(batch, query)
    if not torch.equal(actual, replay):
        raise RuntimeError("Diagnostic gates failed selected-state reconstruction")
    result = {
        "status": "PASSED",
        "initial_static_prediction_max_difference": 0.0,
        "serialization_prediction_max_difference": 0.0,
        "selected_state_gate_tensors": sum(".core_gates." in key for key in state),
        "target_query_labels_used": 0,
    }
    del model, reference, reconstructed
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _crossed_interval(table, rng, draws, grouping):
    seeds = sorted(table.seed.unique())
    if grouping == "cell_uid":
        matrix = (
            table.pivot(index="seed", columns="cell_uid", values="improvement")
            .loc[seeds]
            .to_numpy()
        )
    else:
        means = table.groupby(["seed", "protocol_uid"]).improvement.mean().unstack()
        matrix = means.loc[seeds].to_numpy()
    if not np.isfinite(matrix).all():
        raise RuntimeError("Incomplete paired source diagnostic matrix")
    values = np.empty(draws)
    for draw in range(draws):
        selected_seeds = rng.integers(0, matrix.shape[0], matrix.shape[0])
        selected_units = rng.integers(0, matrix.shape[1], matrix.shape[1])
        values[draw] = matrix[np.ix_(selected_seeds, selected_units)].mean()
    return np.quantile(values, [0.025, 0.975]).tolist()


def paired_comparison(table, candidate, reference, window, draws=2000):
    selected = table[(table.window == window) & table.n_obs.gt(0)]
    candidate_rows = selected[selected["mode"] == candidate]
    reference_rows = selected[selected["mode"] == reference]
    paired = candidate_rows.merge(
        reference_rows,
        on=["seed", "cell_uid", "protocol_uid"],
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(candidate_rows) or len(paired) != len(reference_rows):
        raise RuntimeError("Diagnostic comparison is missing paired physical cells")
    paired["improvement"] = paired.mae_reference - paired.mae_candidate
    per_seed = paired.groupby("seed")[["mae_reference", "mae_candidate"]].mean()
    seed_gain = (
        per_seed.mae_reference - per_seed.mae_candidate
    ) / per_seed.mae_reference
    reference_mean = float(paired.mae_reference.mean())
    key = candidate + "/" + reference + "/" + window
    rng_seed = (42 + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)) % (2**32 - 1)
    return {
        "candidate": candidate,
        "reference": reference,
        "window": window,
        "cells": int(paired.cell_uid.nunique()),
        "protocol_groups": int(paired.protocol_uid.nunique()),
        "candidate_mae": float(paired.mae_candidate.mean()),
        "reference_mae": reference_mean,
        "relative_mae_improvement": float(paired.improvement.mean() / reference_mean),
        "per_seed_relative_improvement": {
            str(seed): float(value) for seed, value in seed_gain.items()
        },
        "all_seeds_improved": set(seed_gain.index) == set(SEEDS)
        and bool((seed_gain > 0).all()),
        "crossed_seed_cell_interval": _crossed_interval(
            paired, np.random.default_rng(rng_seed), draws, "cell_uid"
        ),
        "crossed_seed_protocol_interval": _crossed_interval(
            paired, np.random.default_rng(rng_seed + 1), draws, "protocol_uid"
        ),
        "bootstrap_draws": draws,
    }


def summarize(private_root, public_root, p2_private):
    private_root, public_root, p2_private = map(
        Path, (private_root, public_root, p2_private)
    )
    metrics, cells, lineage = [], [], []
    for mode in ALL_MODES:
        for seed in SEEDS:
            folder = (
                p2_private / "formal/grad_core/rank4" / f"seed{seed}"
                if mode == "static"
                else private_root / mode / f"seed{seed}"
            )
            complete = folder / "SOURCE_DEVELOPMENT_COMPLETE.json"
            if not complete.exists():
                raise RuntimeError(f"D3 remains incomplete at {mode}/seed{seed}")
            receipt = read_json(complete)
            for item in receipt["files"]:
                if file_hash(folder / item["path"]) != item["sha256"]:
                    raise RuntimeError("A source diagnostic training artifact changed")
            if receipt["updates"] != 1320 or receipt["target_query_labels_used"] != 0:
                raise RuntimeError("Unexpected D3 update budget or final Query access")
            unit = pd.read_csv(folder / "SOURCE_VALIDATION_CELL_METRICS.csv")
            unit["mode"], unit["seed"] = mode, seed
            cells.append(unit)
            metrics.append(
                dict(
                    mode=mode,
                    seed=seed,
                    **receipt["validation"],
                    updates=receipt["updates"],
                    reused_P2=mode == "static",
                    trainable_parameters=receipt["trainable_parameters"],
                    best_epoch=receipt["best_epoch"],
                    elapsed_seconds=receipt["elapsed_seconds"],
                )
            )
            lineage.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "receipt_sha256": file_hash(complete),
                    "checkpoint_sha256": receipt["selected_checkpoint_sha256"],
                    "source_validation_prediction_sha256": file_hash(
                        folder / "SOURCE_VALIDATION_PREDICTIONS.npz"
                    ),
                }
            )
    table = pd.concat(cells, ignore_index=True)
    public_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(public_root / "D3_CELL_METRICS.csv", index=False)
    pd.DataFrame(metrics).to_csv(public_root / "D3_METRICS.csv", index=False)
    comparisons = [
        paired_comparison(table, "dual", reference, window)
        for window in ("all", "1001_plus")
        for reference in ("static", "constant", "absolute", "relative", "shuffled_dual")
    ]
    write_json(public_root / "D3_COMPARISONS.json", comparisons)
    main = [item for item in comparisons if item["window"] == "all"]

    def supports(reference):
        full = next(item for item in main if item["reference"] == reference)
        tail = next(
            item
            for item in comparisons
            if item["window"] == "1001_plus" and item["reference"] == reference
        )
        return (
            full["relative_mae_improvement"] >= 0.01
            and full["all_seeds_improved"]
            and full["crossed_seed_cell_interval"][0] > 0
            and full["crossed_seed_protocol_interval"][0] > 0
            and tail["relative_mae_improvement"] >= 0
        )

    state_support = supports("constant") and supports("shuffled_dual")
    complementarity = state_support and supports("absolute") and supports("relative")
    status = "SUPPORTED_SOURCE_CORE_CONDITIONING" if state_support else "NOT_SUPPORTED"
    result = {
        "status": status,
        "completed_units": 18,
        "newly_trained_units": 15,
        "reused_static_units": 3,
        "new_optimizer_updates": 19800,
        "final_target_query_labels_used": 0,
        "state_conditioning_supported": state_support,
        "complementary_views_supported": complementarity,
        "contrast_support": {
            reference: supports(reference)
            for reference in (
                "static",
                "constant",
                "absolute",
                "relative",
                "shuffled_dual",
            )
        },
        "rotating_dynamic_subspace_established": False,
        "interpretation": "All gates remain inside the same frozen U/V linear update span. Improvement supports state-conditioned core coefficients only.",
        "lineage": lineage,
    }
    bind(public_root / "D3_GATE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/drs_peft/execution_v1.json"
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "configs/drs_peft/diagnostics_v1.json",
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument("--prior-public", type=Path, default=ROOT / "results/drs_peft")
    parser.add_argument("--p1-private", type=Path, default=ROOT / "runtime/drs_peft/P1")
    parser.add_argument("--p2-private", type=Path, default=ROOT / "runtime/drs_peft/P2")
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/diagnostics_v1/D3"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results/drs_peft/diagnostics_v1/D3"
    )
    parser.add_argument("--backbone-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS)
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        print(summarize(args.private_root, args.output_root, args.p2_private))
        return
    if args.backbone_dir is None:
        parser.error("--backbone-dir is required for source training")
    if args.epochs != 120:
        parser.error("Formal diagnostic epochs are frozen at 120")
    if not args.registration.exists():
        raise RuntimeError(
            "The new source-only diagnostic registration must be frozen first"
        )
    configure()
    config = read_json(args.config)
    data, inputs, scaler, episodes = load_source(args.data_root)
    bridge = read_json(args.prior_public / "P0/BRIDGE_SELECTION.json")[
        "selected_bridge"
    ]
    arrays, feature_manifest = prepare_views(data, inputs, episodes)
    plan = {
        "stage": "D3_SOURCE_DIAGNOSTIC",
        "seeds": list(SEEDS),
        "modes": list(ALL_MODES),
        "rank": 4,
        "formal_epochs": 120,
        "batch_size": 8,
        "source_train_cells": 85,
        "source_meta_cells": 30,
        "adapter_and_gate_learning_rate": 0.001,
        "bridge_head_learning_rate": 0.0003,
        "new_units": 15,
        "expected_new_updates": 15 * 120 * math.ceil(85 / 8),
        "registration_sha256": file_hash(args.registration),
        "execution_config_sha256": file_hash(args.config),
        "frozen_basis_origin": "P1 source covariance U/V at the same formal P0 B0 state",
        "final_target_query_labels_used": 0,
        "parameter_match": "Five gate arms each contain 1248 nominal trainable gate parameters. Constant inputs reduce effective functional degrees of freedom.",
        "selection": "first strict source-meta MAE minimum, fixed 120 epochs and no new learning-rate search",
    }
    if args.dry_run:
        print(plan)
        return
    bind(args.output_root / "D3_FEATURE_MANIFEST.json", feature_manifest)
    bind(args.output_root / "D3_PLAN.json", plan)
    source_ids = episodes["source_cell_roles"]["source_train"]
    source_rows = data.rows(source_ids)
    for seed in args.seeds:
        checkpoint, checkpoint_path = load_p0_checkpoint(
            args.prior_public, bridge, seed
        )
        bundle = _basis_bundle(args.prior_public, args.p1_private, seed, source_ids)
        static_factory = model_factory(
            args.backbone_dir,
            inputs["policy"].shape[-1],
            bridge,
            "static",
            seed,
            checkpoint,
            bundle,
        )
        for mode in args.modes:
            mode_inputs, donors = inputs_for_mode(
                inputs, arrays, episodes, data, mode, seed
            )
            factory = model_factory(
                args.backbone_dir,
                inputs["policy"].shape[-1],
                bridge,
                mode,
                seed,
                checkpoint,
                bundle,
            )
            destination = args.private_root / mode / f"seed{seed}"
            public = args.output_root / "units" / mode / f"seed{seed}"
            bind(destination / "D3_SHUFFLE_DONORS.json", donors)
            check_path = destination / "D3_INITIALIZATION_CHECK.json"
            if not check_path.exists():
                bind(
                    check_path,
                    initialization_check(
                        factory, static_factory, mode_inputs, source_rows, args.device
                    ),
                )
            receipt = train_source_unit(
                data,
                mode_inputs,
                scaler,
                episodes,
                config,
                args.config,
                args.backbone_dir,
                destination,
                bridge,
                "D3_" + mode,
                seed,
                120,
                args.device,
                rank=4,
                adapter_lr=0.001,
                stop_after=args.stop_after,
                stage="D3_SOURCE_DIAGNOSTIC",
                model_factory=factory,
                initial_checkpoint_sha256=file_hash(checkpoint_path),
                extra_sources=EXTRA_SOURCES,
            )
            if receipt["status"] == "ENGINEERING_INTERRUPTED":
                print(receipt, flush=True)
                return
            _publish_unit(destination, public)
            print(
                {
                    "event": "D3_UNIT_COMPLETE",
                    "mode": mode,
                    "seed": seed,
                    "updates": receipt["updates"],
                    "validation": receipt["validation"],
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
