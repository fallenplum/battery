"""Check every locked TRUE-MA rotation frame in both outer descriptor views."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from threadpoolctl import threadpool_limits

from scripts.drs_peft.data import CACHE, ROOT, file_hash
from scripts.drs_peft.run_onesided_reference import load_reference_model
from scripts.drs_peft.run_spectral_rotation import input_features
from scripts.drs_peft.spectral_objective_engine import (
    MapperTrial,
    authenticate_trial,
    load_mapper,
    mapper_trial_path,
    trial_identity,
)
from scripts.drs_peft.spectral_objective_protocol import (
    CANDIDATE,
    CONFIG,
    PRIVATE,
    PUBLIC,
    build_plan,
)
from scripts.drs_peft.spectral_rotation import RANK, build_rotation_bases
from scripts.drs_peft.spectral_rotation_engine import Descriptors, core_model
from scripts.drs_peft.train import bind, read_json
from scripts.drs_peft.transfer_scope_reference import BACKBONE, REFERENCES

GRAM_TOLERANCE = 2e-5


def _verify_locked_inputs(args):
    marker = args.public / "DEVELOPMENT_LOCK.json"
    if not marker.is_file():
        raise FileNotFoundError("Audit requires the completed development lock")
    lock = read_json(marker)
    lock_sha = file_hash(marker)
    identity_path = args.public / "RUN_IDENTITY.json"
    identity = read_json(identity_path)
    identity_sha = file_hash(identity_path)
    hyper_path = args.public / "HYPERPARAMETER_LOCK.json"
    hyper = read_json(hyper_path)
    if (
        lock["status"] != "LOCKED_BEFORE_SOURCE_OUTER_LABELS"
        or lock["identity_sha256"] != identity_sha
        or lock["hyperparameters_sha256"] != file_hash(hyper_path)
        or lock["stage_b_sha256"] != file_hash(args.public / "stage_b/COMPLETE.json")
        or lock["outer_role_labels_read"]
        or lock["target_labels_read"]
        or hyper["status"] != "LOCKED_AFTER_STAGE_A"
        or hyper["identity_sha256"] != identity_sha
    ):
        raise RuntimeError(
            "The frozen development or source selection identity changed"
        )
    for name, expected in (
        identity["code_sha256"] | lock["scoring_code_sha256"]
    ).items():
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise RuntimeError(f"The locked production operator changed: {name}")
    cfg, plan, _, source_ids, sealed = build_plan(args)
    plan_path = args.public / "REGISTERED_EPISODES.json"
    if (
        file_hash(plan_path) != identity["plan_sha256"]
        or read_json(plan_path) != plan
        or identity["config_sha256"] != file_hash(args.config)
        or identity["prefix_cache_manifest_sha256"]
        != file_hash(args.data_root / "MATR/COMPLETE.json")
    ):
        raise RuntimeError("The registered source-only role or data identity changed")
    # This reads the already qualified signed prefix descriptor file, never the trajectory cache.
    prefix_identity = SimpleNamespace(
        cache_manifest_sha256=identity["prefix_cache_manifest_sha256"]
    )
    features, receipt_path = input_features(args, source_ids, prefix_identity)
    feature_receipt = read_json(receipt_path)
    if (
        file_hash(receipt_path) != identity["feature_receipt_sha256"]
        or feature_receipt["audit"]["post_prefix_supervision_rows_read"]
    ):
        raise RuntimeError("Signed prefix features differ from the locked run")
    return cfg, plan, features, sealed, lock, lock_sha, identity, identity_sha, hyper


def _load_locked_controller(args, cfg, fold, seed, hyper, lock, identity_sha, model):
    index = str(fold["fold"])
    alpha = hyper["selected"][index][CANDIDATE]["alpha"]
    trial = MapperTrial("ROTATION_TRUE", "MA", alpha, seed)
    folder = mapper_trial_path(args.private, fold["fold"], trial)
    entry = lock["mapper_artifacts"][index][f"{seed}:{CANDIDATE}"]
    stage = "A" if seed == cfg["selection_seed"] else "B"
    expected = trial_identity(trial, fold, identity_sha, stage)
    receipt = authenticate_trial(folder, expected)
    if entry != {
        "folder": folder.relative_to(args.private).as_posix(),
        "alpha": alpha,
        "receipt_sha256": file_hash(folder / "COMPLETE.json"),
        "mapper_sha256": receipt["files"]["MAPPER.pt"],
    }:
        raise RuntimeError("A TRUE-MA mapper differs from its development lock entry")
    load_mapper(model, folder, expected)
    return entry


def _gram_error(frame):
    # Assemble U(theta) with its actual FP32 factors, accrue the Gram in FP64.
    frame = frame.double()
    identity = torch.eye(frame.shape[-1], dtype=frame.dtype)
    return (frame.transpose(-2, -1) @ frame - identity).abs().amax(dim=(-2, -1))


@torch.no_grad()
def _inspect_views(model, fold, descriptors, sealed):
    stats = {
        name: {
            "v_gram_max_abs": float(_gram_error(layer.right)),
            "scale": layer.scale,
            "max_angle_difference_rad": 0.0,
            "cells_by_role": {"query": 0, "anchor": 0},
            "cells_with_different_angles": 0,
            "u_theta_gram_max_abs": {
                view: {"query": 0.0, "anchor": 0.0} for view in ("true", "deranged")
            },
        }
        for name, layer in model.layers.items()
    }
    for episode in fold["outer"]:
        for kind, ids in (("query", episode["query"]), ("anchor", fold["anchors"])):
            donors = episode["donors"][kind]
            if (
                set(ids) & sealed
                or len(set(ids)) != len(ids)
                or set(donors) != set(ids)
                or any(uid == donor for uid, donor in zip(ids, donors, strict=True))
            ):
                raise RuntimeError(
                    "A registered role-local descriptor derangement changed"
                )
            angles = {
                view: model.modulation(len(ids), descriptors.tensor(roster, "cpu"))
                for view, roster in (("true", ids), ("deranged", donors))
            }
            if any(not bool(torch.isfinite(theta).all()) for theta in angles.values()):
                raise FloatingPointError("The locked mapper produced nonfinite angles")
            for name, layer in model.layers.items():
                metadata = stats[name]
                metadata["cells_by_role"][kind] += len(ids)
                difference = (
                    angles["true"][:, layer.index] - angles["deranged"][:, layer.index]
                ).abs()
                metadata["max_angle_difference_rad"] = max(
                    metadata["max_angle_difference_rad"], float(difference.max())
                )
                metadata["cells_with_different_angles"] += int(
                    difference.amax(dim=-1).gt(0).sum()
                )
                for view, theta in angles.items():
                    theta = theta[:, layer.index].unsqueeze(-2)
                    frame = layer.left * theta.cos() + layer.normal * theta.sin()
                    residual = _gram_error(frame)
                    if len(residual) != len(ids) or not bool(
                        torch.isfinite(residual).all()
                    ):
                        raise RuntimeError("Invalid per-cell Gram residual")
                    previous = metadata["u_theta_gram_max_abs"][view][kind]
                    metadata["u_theta_gram_max_abs"][view][kind] = max(
                        previous, float(residual.max())
                    )
    return stats


def _relative_bound(left_error, right_error):
    # ||G-I||_2 <= r * max_{ij} |G_ij-delta_ij| for the symmetric r x r Gram.
    left, right = RANK * left_error, RANK * right_error
    lower = math.sqrt((1 - left) * (1 - right))
    upper = math.sqrt((1 + left) * (1 + right))
    return max(1 - lower, upper - 1)


def audit(args):
    cfg, plan, features, sealed, lock, lock_sha, identity, identity_sha, hyper = (
        _verify_locked_inputs(args)
    )
    records = []
    max_u = max_v = max_spectrum_relative_error = 0.0
    tested_frames = 0
    for fold in plan["folds"]:
        index = fold["fold"]
        original = load_reference_model(
            args.reference_root, args.backbone_dir, index, cfg["reference_seed"], "cpu"
        )
        frames = build_rotation_bases(original)
        phi = Descriptors(features, fold["fit_ids"])
        scaler = args.public / "stage_a" / f"fold{index}" / "DESCRIPTOR_SCALER.json"
        if phi.audit != read_json(scaler):
            raise RuntimeError(
                "The scaler differs from fit-only source standardization"
            )
        model = core_model(original, frames, "ROTATION_TRUE", phi, cfg, "cpu")
        del original
        for seed in cfg["meta_training_seeds"]:
            entry = _load_locked_controller(
                args, cfg, fold, seed, hyper, lock, identity_sha, model
            )
            layers = _inspect_views(model, fold, phi, sealed)
            if len(layers) != 12:
                raise RuntimeError("The locked frame does not cover all 12 Q/V modules")
            for row in layers.values():
                if row["cells_by_role"] != {"query": 60, "anchor": 24}:
                    raise RuntimeError("An outer descriptor view or cell is missing")
                right = row["v_gram_max_abs"]
                left = max(
                    value
                    for view in row["u_theta_gram_max_abs"].values()
                    for value in view.values()
                )
                if not math.isfinite(right) or max(left, right) > GRAM_TOLERANCE:
                    raise RuntimeError("The tested rotation frame lost orthonormality")
                bound = _relative_bound(left, right)
                row["fixed_core_relative_singular_value_error_bound"] = bound
                tested_frames += 2 * sum(row["cells_by_role"].values())
                max_u, max_v = max(max_u, left), max(max_v, right)
                max_spectrum_relative_error = max(max_spectrum_relative_error, bound)
            records.append(
                {
                    "fold": index,
                    "reference_sha256": fold["reference_sha256"],
                    "scaler_sha256": file_hash(scaler),
                    "training_seed": seed,
                    "mapper": entry,
                    "layers": layers,
                }
            )
        del model
    if (
        len(records) != 9
        or tested_frames != 18144
        or file_hash(args.public / "DEVELOPMENT_LOCK.json") != lock_sha
    ):
        raise RuntimeError("The registered 3 x 3 frame audit is incomplete or changed")
    return {
        "status": "VERIFIED_LOCKED_TRUE_MA_FIXED_CORE_SPECTRUM",
        "method": CANDIDATE,
        "development_lock_sha256": lock_sha,
        "run_identity_sha256": identity_sha,
        "registered_episodes_sha256": identity["plan_sha256"],
        "feature_receipt_sha256": identity["feature_receipt_sha256"],
        "audit_implementation_sha256": file_hash(Path(__file__)),
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "future_label_rows_read": 0,
        "trajectory_prediction_forwards": 0,
        "registered_mapper_fits_checked": len(records),
        "per_cell_left_frames_checked_both_views": tested_frames,
        "gram_max_abs_tolerance": GRAM_TOLERANCE,
        "max_u_theta_gram_abs_error": max_u,
        "max_v_gram_abs_error": max_v,
        "right_frame_is_shared_across_views": True,
        "fixed_core_max_relative_singular_value_error_bound": max_spectrum_relative_error,
        "fixed_core_true_vs_deranged_singular_value_difference_bound_relative_to_core": 2
        * max_spectrum_relative_error,
        "scope": "Nonzero singular values of s * U_theta * C * V^T at fixed per-layer C and s",
        "fold_seed_records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument(
        "--features", type=Path, default=ROOT / "runtime/drs_peft/spectral_rotation_v1"
    )
    parser.add_argument("--reference-root", type=Path, default=REFERENCES)
    parser.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    parser.add_argument("--data-root", type=Path, default=CACHE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        receipt = audit(args)
    destination = args.public / "SPECTRUM_AUDIT.json"
    bind(destination, receipt)
    print(
        f"Verified {receipt['per_cell_left_frames_checked_both_views']} locked frames. "
        f"Max Gram errors Uθ={receipt['max_u_theta_gram_abs_error']:.8g}, "
        f"V={receipt['max_v_gram_abs_error']:.8g}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
