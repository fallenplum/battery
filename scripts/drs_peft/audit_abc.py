"""Independently audit A/B/C source states and physical update norms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.drs_peft.audit_e123 import (
    audit_norm_table,
    bridge_head_hash,
    condition_norms,
    normalized_views,
    verify_files,
)
from scripts.drs_peft.data import ROOT, file_hash, load_dataset, write_json
from scripts.drs_peft.e123_protocol import load_protocol
from scripts.drs_peft.run_abc_a import build_plan as factorial_plan
from scripts.drs_peft.run_abc_a import reused_from_e123
from scripts.drs_peft.run_abc_c import build_plan as subspace_plan
from scripts.drs_peft.run_abc_c import unit_path as subspace_path
from scripts.drs_peft.run_e123 import matching_unit, unit_path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registered_units(config, protocol, studies):
    records = []
    for study in studies:
        plan = (
            subspace_plan(config, protocol)
            if study == "C"
            else factorial_plan(config, protocol, study)
        )
        for unit in plan["units"]:
            reused = study != "C" and reused_from_e123(unit, study)
            records.append({"study": study, "unit": unit, "reused": reused})
    return records


def source_folder(args, record):
    unit = record["unit"]
    if record["reused"]:
        old = {key: value for key, value in unit.items() if key != "regime"}
        return unit_path(args.e123_root, old)
    if record["study"] == "C":
        return subspace_path(args.private_root / "C", unit)
    return unit_path(args.private_root / record["study"] / unit["regime"], unit)


def expected_parameters(config, record):
    unit = record["unit"]
    if record["study"] == "C":
        return config["C"]["trainable_parameters"][unit["arm"]]
    return (
        config["A"]["joint_total_trainable_parameters"]
        if unit["regime"] == "joint"
        else 1440
    )


def audit_source(args, config, protocol, records):
    checked, starts, new_updates = [], {}, 0
    for record in records:
        study, unit = record["study"], record["unit"]
        folder = source_folder(args, record)
        receipt = read(folder / "SOURCE_DEVELOPMENT_COMPLETE.json")
        verify_files(folder, receipt["files"])
        checkpoint_path = folder / "CHECKPOINT_SELECTED.pt"
        assert file_hash(checkpoint_path) == receipt["selected_checkpoint_sha256"]
        run = read(folder / "RUN_CONFIG.json")
        fold = protocol["folds"][unit["fold_id"]]
        assert run["training_ids"] == fold["fit_ids"]
        assert run["validation_ids"] == fold["inner_validation_ids"]
        assert run["seed"] == unit["seed"]
        assert run["trainable_parameters"] == expected_parameters(config, record)
        assert run["epochs"] == 120
        assert run["batch_size"] == 8
        assert run["adapter_learning_rate"] == 0.001
        assert run["learning_rate"] == (0.001 if study == "C" else 0.0003)
        assert run["weight_decay"] == 0.01 and run["gradient_clip"] == 1.0
        assert (
            receipt["target_query_labels_used"] == run["target_query_labels_used"] == 0
        )
        assert receipt["frozen_parameters_unchanged"] is True
        history = read(folder / "TRAIN_LOG.json")
        assert len(history) == 120
        assert [row["epoch"] for row in history] == list(range(1, 121))
        updates = 120 * math.ceil(len(fold["fit_ids"]) / 8)
        assert receipt["updates"] == run["expected_updates"] == updates
        selected_epoch = int(np.argmin([row["validation_mae"] for row in history])) + 1
        assert selected_epoch == receipt["best_epoch"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["identity"] == run
        assert checkpoint["best_epoch"] == selected_epoch
        state = checkpoint["model"]
        if study == "C":
            counted = sum(
                value.numel()
                for key, value in state.items()
                if key.endswith((".delta_core", ".core_perp", ".core_a_free"))
            )
            assert counted == expected_parameters(config, record)
        else:
            counted = sum(
                value.numel()
                for key, value in state.items()
                if key.endswith(".delta_core")
                or key.startswith("bridge.backbone.module.core_gates.")
            )
            assert counted == config["A"]["core_gate_trainable_parameters"]
        static_folder = unit_path(
            args.e123_root,
            matching_unit("static", unit["fold_id"], unit["seed"], "grad_core"),
        )
        static_path = static_folder / "CHECKPOINT_SELECTED.pt"
        assert run["initial_checkpoint_sha256"] == file_hash(static_path)
        static = torch.load(static_path, map_location="cpu", weights_only=False)[
            "model"
        ]
        prefixes = [
            key.removesuffix(".source_core")
            for key in state
            if key.endswith(".source_core")
        ]
        assert len(prefixes) == 12
        for prefix in prefixes:
            assert torch.equal(state[prefix + ".source_core"], static[prefix + ".core"])
            assert torch.equal(state[prefix + ".core"], static[prefix + ".core"])
            for name in ("U", "V"):
                assert torch.equal(
                    state[prefix + "." + name], static[prefix + "." + name]
                )
        if study == "C" or unit["regime"] == "frozen":
            assert bridge_head_hash(state) == bridge_head_hash(static)
        if not record["reused"]:
            assert run["config_sha256"] == file_hash(args.config)
            for entry in run["source_snapshot"]:
                assert file_hash(ROOT / entry["path"]) == entry["sha256"], entry["path"]
            optimizer = torch.load(
                folder / "TRAIN_STATE.pt", map_location="cpu", weights_only=False
            )
            assert optimizer["identity"] == run
            assert optimizer["updates"] == updates and optimizer["next_epoch"] == 120
            steps = [
                float(value["step"])
                for value in optimizer["optimizer"]["state"].values()
                if "step" in value
            ]
            assert steps and all(value == updates for value in steps)
            spec = read(folder / f"ABC_{study}_UNIT_SPEC.json")
            assert spec["unit"] == unit
            assert spec["fit_ids"] == fold["fit_ids"]
            assert spec["inner_validation_ids"] == fold["inner_validation_ids"]
            assert spec["outer_ids"] == fold["outer_test_ids"]
            assert spec["outer_labels_used_for_training_or_selection"] == 0
            assert spec["final_target_query_labels_used"] == 0
            if study == "C":
                assert spec["config_sha256"] == file_hash(args.config)
                assert spec["folds_sha256"] == file_hash(args.folds)
                assert spec["source_C0_checkpoint_sha256"] == file_hash(static_path)
            else:
                assert spec["registration_sha256"] == file_hash(args.config)
                assert spec["protocol_file_sha256"] == file_hash(args.folds)
                assert spec["static_checkpoint_sha256"] == file_hash(static_path)
                key = (unit["fold_id"], unit["seed"])
                start = spec["start"]
                identity = (
                    start["initial_state_sha256"],
                    start["bridge_head_state_sha256"],
                    start["source_core_sha256"],
                )
                if key in starts:
                    assert starts[key] == identity
                starts[key] = identity
            new_updates += updates
            del optimizer
        checked.append(
            {
                "study": study,
                "unit": unit,
                "reused": record["reused"],
                "checkpoint_sha256": file_hash(checkpoint_path),
                "updates": updates,
            }
        )
        del checkpoint, state, static
    expected_new = sum(not record["reused"] for record in records)
    assert new_updates == expected_new * 960
    return {
        "status": "PASSED",
        "checked_source_units": len(checked),
        "new_training_units": expected_new,
        "reused_training_units": len(checked) - expected_new,
        "new_optimizer_updates": new_updates,
        "source_units": checked,
        "final_target_query_labels_used": 0,
    }


def independent_subspace_norms(state, ids):
    records = []
    prefixes = [
        key.removesuffix(".delta_core") for key in state if key.endswith(".delta_core")
    ]
    assert len(prefixes) == 12
    for prefix in prefixes:
        delta = state[prefix + ".delta_core"].double().numpy()
        if prefix + ".core_a_free" in state:
            right = np.ones(384)
            indices = state[prefix + ".core_a_indices"].numpy()
            right[indices] = state[prefix + ".core_a_free"].double().numpy()
            raw = delta[:, None] * right[None]
        else:
            coefficients = delta
            if prefix + ".core_perp" in state:
                complement = state[prefix + ".core_perp"].double().numpy()
                coefficients = np.zeros(
                    (
                        delta.shape[0] + complement.shape[0],
                        delta.shape[1] + complement.shape[1],
                    )
                )
                coefficients[: delta.shape[0], : delta.shape[1]] = delta
                coefficients[delta.shape[0] :, delta.shape[1] :] = complement
            left = state[prefix + ".core_increment_u"].double().numpy()
            right = state[prefix + ".core_increment_v"].double().numpy()
            raw = left @ coefficients @ right.T
        raw_norm = float(np.linalg.norm(raw))
        radius = float(state[prefix + ".core_radius"])
        baseline = (
            state[prefix + ".U"].double().numpy()
            @ state[prefix + ".source_core"].double().numpy()
            @ state[prefix + ".V"].double().numpy().T
        )
        ball_factor = min(1.0, radius / raw_norm) if raw_norm > 0 else 1.0
        matched_factor = radius / raw_norm if raw_norm > 0 else 0.0
        for uid in ids:
            records.append(
                {
                    "cell_uid": uid,
                    "layer": prefix.removeprefix("bridge.backbone.module."),
                    "independent_raw_increment_norm": raw_norm,
                    "independent_baseline_norm": float(np.linalg.norm(baseline)),
                    "independent_ball_total_norm": float(
                        np.linalg.norm(baseline + raw * ball_factor)
                    ),
                    "independent_matched_total_norm": float(
                        np.linalg.norm(baseline + raw * matched_factor)
                    ),
                    "radius": radius,
                }
            )
    return pd.DataFrame(records)


def verify_derangement(mapping, fold, seed):
    assert {row["role"] for row in mapping} == {"fit", "inner_validation", "outer_test"}
    assert len(mapping) == sum(
        len(fold[key]) for key in ("fit_ids", "inner_validation_ids", "outer_test_ids")
    )
    for role, key in (
        ("fit", "fit_ids"),
        ("inner_validation", "inner_validation_ids"),
        ("outer_test", "outer_test_ids"),
    ):
        ids = fold[key]
        order = sorted(
            ids,
            key=lambda uid: hashlib.sha256(
                f"abc-state-shuffle|{seed}|{fold['fold_id']}|{role}|{uid}".encode()
            ).digest(),
        )
        active = [row for row in mapping if row["role"] == role]
        assert len(active) == len(ids)
        assert len({row["cell_uid"] for row in active}) == len(active)
        assert {row["cell_uid"] for row in active} == set(ids)
        assert {row["feature_donor_uid"] for row in active} == set(order)
        assert all(row["cell_uid"] != row["feature_donor_uid"] for row in active)
        expected = {
            uid: order[(index + 1) % len(order)] for index, uid in enumerate(order)
        }
        assert {row["cell_uid"]: row["feature_donor_uid"] for row in active} == expected


def audit_outer(args, config, protocol, records):
    data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
    features, summaries, checked_rows = {}, [], 0
    for record in records:
        study, unit = record["study"], record["unit"]
        source = source_folder(args, record)
        checkpoint_path = source / "CHECKPOINT_SELECTED.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        fold = protocol["folds"][unit["fold_id"]]
        if study != "C":
            key = (study, unit["fold_id"])
            if key not in features:
                manifest = read(
                    args.public_root
                    / study
                    / f"fold{unit['fold_id']}"
                    / "FOLD_VIEW_MANIFEST.json"
                )
                features[key] = normalized_views(data, fold, manifest)
            active_features = features[key]
            if unit["mode"] == "shuffled_dual":
                mapping = read(
                    args.public_root
                    / study
                    / f"fold{unit['fold_id']}"
                    / f"seed{unit['seed']}"
                    / "SHUFFLE_MANIFEST.json"
                )["mapping"]
                verify_derangement(mapping, fold, unit["seed"])
                active_features = {
                    view: {
                        row["cell_uid"]: values[row["feature_donor_uid"]]
                        for row in mapping
                    }
                    for view, values in active_features.items()
                }
        else:
            expected_c = independent_subspace_norms(state, fold["outer_test_ids"])
        for evaluation in config["evaluation_modes"]:
            if record["reused"]:
                old_unit = {
                    key: value for key, value in unit.items() if key != "regime"
                }
                folder = (
                    args.e123_root
                    / "outer"
                    / unit_path(Path("."), old_unit)
                    / evaluation
                )
            elif study == "C":
                folder = subspace_path(args.private_root / "C/outer", unit) / evaluation
            else:
                folder = (
                    unit_path(
                        args.private_root / study / unit["regime"] / "outer", unit
                    )
                    / evaluation
                )
            receipt = read(folder / "OUTER_COMPLETE.json")
            verify_files(folder, receipt["files"])
            assert receipt["checkpoint_sha256"] == file_hash(checkpoint_path)
            assert receipt["outer_ids"] == fold["outer_test_ids"]
            assert receipt["outer_groups"] == fold["outer_test_groups"]
            assert receipt["evaluation_mode"] == evaluation
            assert receipt["unit"] == (old_unit if record["reused"] else unit)
            registration = (
                ROOT / "configs/drs_peft/e123_v1.json"
                if record["reused"]
                else args.config
            )
            implementation = (
                ROOT / "scripts/drs_peft/run_abc_c.py"
                if study == "C"
                else ROOT / "scripts/drs_peft/run_e123.py"
            )
            assert receipt["registration_sha256"] == file_hash(registration)
            assert receipt["source_sha256"] == file_hash(implementation)
            assert (
                receipt["final_target_query_labels_used"]
                == receipt["optimizer_updates"]
                == 0
            )
            table = pd.read_csv(folder / "UPDATE_DIAGNOSTICS.csv")
            audit_norm_table(table, evaluation)
            if study == "C":
                independent = expected_c.copy()
                raw = independent.independent_raw_increment_norm.to_numpy()
                radius = independent.radius.to_numpy()
                actual = np.minimum(raw, radius)
                if evaluation == "matched_norm":
                    actual = np.where(raw > 0, radius, 0)
                independent["independent_increment_norm"] = actual
            else:
                independent = condition_norms(
                    state,
                    active_features,
                    fold["outer_test_ids"],
                    "dual" if unit["mode"] == "shuffled_dual" else unit["mode"],
                    evaluation,
                )
            paired = table.merge(
                independent,
                on=["cell_uid", "layer"],
                validate="one_to_one",
                suffixes=("", "_independent"),
            )
            assert len(paired) == len(table) == len(fold["outer_test_ids"]) * 12
            for actual_name, expected_name in (
                ("raw_increment_norm", "independent_raw_increment_norm"),
                ("increment_norm", "independent_increment_norm"),
            ):
                np.testing.assert_allclose(
                    paired[actual_name], paired[expected_name], rtol=3e-5, atol=3e-8
                )
            if study == "C":
                total_column = (
                    "independent_matched_total_norm"
                    if evaluation == "matched_norm"
                    else "independent_ball_total_norm"
                )
                np.testing.assert_allclose(
                    paired.baseline_update_norm,
                    paired.independent_baseline_norm,
                    rtol=3e-5,
                    atol=3e-8,
                )
                np.testing.assert_allclose(
                    paired.total_update_norm, paired[total_column], rtol=3e-5, atol=3e-8
                )
            np.testing.assert_allclose(
                paired.radius, paired.radius_independent, rtol=1e-12, atol=1e-12
            )
            checked_rows += len(table)
            summaries.append(
                {
                    "study": study,
                    "unit": unit,
                    "evaluation_mode": evaluation,
                    "rows": len(table),
                    "matched_norm_unattainable": int(
                        table.matched_norm_unattainable.sum()
                    ),
                    "maximum_norm_difference": float(
                        np.abs(
                            paired.increment_norm - paired.independent_increment_norm
                        ).max()
                    ),
                }
            )
        del checkpoint, state
    return {
        "status": "PASSED",
        "independently_checked_cell_layer_norms": checked_rows,
        "checked_outer_units": len(summaries),
        "units": summaries,
        "final_target_query_labels_used": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/drs_peft/abc_v1.json"
    )
    parser.add_argument(
        "--folds", type=Path, default=ROOT / "configs/drs_peft/e123_folds_v1.json"
    )
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/abc_v1"
    )
    parser.add_argument(
        "--public-root", type=Path, default=ROOT / "results/drs_peft/abc_v1"
    )
    parser.add_argument(
        "--e123-root", type=Path, default=ROOT / "runtime/drs_peft/e123_v1"
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument(
        "--studies", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"]
    )
    parser.add_argument("--phase", choices=("source", "outer"), required=True)
    args = parser.parse_args()
    config, protocol = read(args.config), load_protocol(args.folds)
    assert config["fold_payload_sha256"] == protocol["protocol_sha256"]
    records = registered_units(config, protocol, args.studies)
    output = (
        audit_source(args, config, protocol, records)
        if args.phase == "source"
        else audit_outer(args, config, protocol, records)
    )
    label = "".join(args.studies)
    output.update(
        config_sha256=file_hash(args.config),
        implementation_sha256=file_hash(__file__),
        studies=args.studies,
    )
    write_json(
        args.public_root / f"INDEPENDENT_{label}_{args.phase.upper()}_AUDIT.json",
        output,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in output.items()
                if key not in ("source_units", "units")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
