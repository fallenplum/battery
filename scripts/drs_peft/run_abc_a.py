"""Execute the matched Bridge/Head by state-gating source-only factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from filelock import FileLock
from threadpoolctl import threadpool_limits

from models.drs_peft.abc_conditioning import FactorialConditionedBatteryTSFM
from models.drs_peft.peft import SubspaceLinear
from scripts.drs_peft.data import file_hash, load_dataset
from scripts.drs_peft.e123_protocol import load_protocol, validate_against_data
from scripts.drs_peft.episodes import load_registration
from scripts.drs_peft.gradient_audit import state_fingerprint
from scripts.drs_peft.run_e123 import (
    _loaded_static,
    check_initialization,
    checked_bases,
    checked_training_checkpoint,
    matching_unit,
    payload_hash,
    prepare_condition_inputs,
    prepare_fold,
    score_outer_unit,
    static_factory,
    unit_path,
)
from scripts.drs_peft.run_p2 import _publish_unit
from scripts.drs_peft.train import (
    ROOT,
    bind,
    configure,
    load_selected,
    read_json,
    selected_state,
    train_source_unit,
)

MODES = ("constant", "absolute", "dual")
REGIMES = ("frozen", "joint")
B_MODES = ("relative", "shuffled_dual")
EVALUATION_MODES = ("trained_ball", "matched_norm")
EXTRA_SOURCES = (
    "scripts/drs_peft/run_abc_a.py",
    "models/drs_peft/abc_conditioning.py",
    "scripts/drs_peft/run_e123.py",
    "scripts/drs_peft/e123_protocol.py",
    "models/drs_peft/e123_conditioning.py",
    "models/drs_peft/diagnostic_dualview.py",
    "configs/drs_peft/e123_folds_v1.json",
    "configs/drs_peft/execution_v1.json",
)


def external_lock(private_root, path):
    digest = hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()
    folder = Path(private_root) / "_locks"
    folder.mkdir(parents=True, exist_ok=True)
    return FileLock(str(folder / (digest + ".lock")))


def bind_shared(args, path, value):
    with external_lock(args.private_root, path):
        bind(path, value)


def build_plan(registration, protocol, study="A"):
    modes = registration.get("a_modes", list(MODES))
    if list(modes) != list(MODES):
        raise ValueError("A requires the constant, absolute, and dual matched arms")
    if list(registration["evaluation_modes"]) != list(EVALUATION_MODES):
        raise ValueError("A requires both registered update-norm evaluations")
    if float(registration["primary_radius"]) != 1.0:
        raise ValueError("A reuses the E1--E3 primary radius of one")
    if float(registration["reference_weight_fraction"]) != 0.01:
        raise ValueError("A reuses the E1--E3 official-weight reference fraction")
    if registration.get("gate_lr", registration.get("adapter_lr")) != registration.get(
        "adapter_lr"
    ):
        raise ValueError("The common adapter optimizer requires equal core/gate rates")
    selected_modes = MODES if study == "A" else B_MODES
    units = [
        {
            "stage": "condition",
            "fold_id": int(fold["fold_id"]),
            "seed": int(seed),
            "method": "grad_core",
            "radius": 1.0,
            "mode": mode,
            "regime": regime,
        }
        for fold in protocol["folds"]
        for seed in registration["training_seeds"]
        for regime in REGIMES
        for mode in selected_modes
    ]
    plan = {
        "study_id": registration["study_id"],
        "experiment": study,
        "units": units,
        "new_training_units": sum(not reused_from_e123(unit, study) for unit in units),
        "reused_frozen_units": sum(reused_from_e123(unit, study) for unit in units),
        "total_factorial_units": len(units),
        "outer_evaluation_modes": list(EVALUATION_MODES),
        "protocol_sha256": protocol["protocol_sha256"],
        "final_target_query_labels_used": 0,
    }
    plan["plan_sha256"] = payload_hash(plan)
    return plan


def reused_from_e123(unit, study):
    return unit["regime"] == "frozen" and (study == "A" or unit["mode"] == "relative")


def legacy_args(args):
    return SimpleNamespace(
        private_root=args.e123_private_root, backbone_dir=args.backbone_dir
    )


def regime_args(args, regime):
    values = vars(args).copy()
    values["private_root"] = args.private_root / regime
    values["public_root"] = args.public_root / regime
    return SimpleNamespace(**values)


def load_start(args, inputs, fold, seed):
    old = legacy_args(args)
    warmup, warmup_path = checked_training_checkpoint(
        unit_path(
            args.e123_private_root, matching_unit("warmup", fold["fold_id"], seed)
        )
    )
    bundle = checked_bases(old, fold["fold_id"], seed, fold["fit_ids"])
    static_unit = matching_unit("static", fold["fold_id"], seed, "grad_core")
    checkpoint, checkpoint_path = checked_training_checkpoint(
        unit_path(args.e123_private_root, static_unit)
    )
    construct = static_factory(old, inputs, static_unit, warmup, bundle)
    return construct, checkpoint, checkpoint_path, warmup_path


def factorial_factory(construct_static, static_checkpoint, unit, registration):
    def construct():
        model = construct_static()
        load_selected(model, static_checkpoint["model"])
        result = FactorialConditionedBatteryTSFM(
            model,
            unit["mode"],
            unit["radius"],
            train_bridge_head=unit["regime"] == "joint",
            reference_weight_fraction=float(registration["reference_weight_fraction"]),
        )
        names = {name for name, p in result.named_parameters() if p.requires_grad}
        if not names.issubset(selected_state(result)):
            raise RuntimeError(
                "The selected checkpoint would omit trainable parameters"
            )
        if "A" in registration:
            report = result.parameter_report()
            expected_common = (
                int(registration["A"]["joint_bridge_head_trainable_parameters"])
                if unit["regime"] == "joint"
                else 0
            )
            expected_total = (
                int(registration["A"]["joint_total_trainable_parameters"])
                if unit["regime"] == "joint"
                else int(registration["A"]["core_gate_trainable_parameters"])
            )
            if (
                report["bridge_head_trainable_parameters"] != expected_common
                or report["trainable"] != expected_total
            ):
                raise RuntimeError(
                    "The factorial parameter budget differs from registration"
                )
        return result

    return construct


def start_identity(factory):
    model = factory()
    common_sha256 = bridge_head_state_hash(selected_state(model))
    cores = {
        name: hashlib.sha256(
            layer.source_core.detach().cpu().numpy().tobytes()
        ).hexdigest()
        for name, layer in model.backbone.named_modules()
        if isinstance(layer, SubspaceLinear)
    }
    result = {
        "initial_state_sha256": state_fingerprint(model),
        "bridge_head_state_sha256": common_sha256,
        "source_core_sha256": cores,
        "parameter_report": model.parameter_report(),
    }
    del model
    return result


def bridge_head_state_hash(state):
    common = {
        name: value
        for name, value in state.items()
        if not name.startswith("bridge.backbone.module.")
    }
    digest = hashlib.sha256()
    for name, value in sorted(common.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_training_controls(identity, registration, unit, expected_parameters):
    expected = {
        "seed": unit["seed"],
        "epochs": int(registration["epochs"]["condition"]),
        "batch_size": int(registration.get("batch_size", 8)),
        "rank": 4,
        "learning_rate": float(registration["bridge_head_lr"]),
        "adapter_learning_rate": float(registration["adapter_lr"]),
        "weight_decay": float(registration.get("weight_decay", 0.01)),
        "gradient_clip": float(registration.get("gradient_clip", 1.0)),
        "trainable_parameters": expected_parameters,
        "target_query_labels_used": 0,
    }
    mismatched = [key for key, value in expected.items() if identity.get(key) != value]
    if mismatched:
        raise RuntimeError(
            "Factorial training controls differ: " + ", ".join(mismatched)
        )


def validate_static_start(start, static_state):
    prefix = "bridge.backbone.module."
    cores = {
        name[len(prefix) : -len(".core")]: hashlib.sha256(
            value.cpu().numpy().tobytes()
        ).hexdigest()
        for name, value in static_state.items()
        if name.startswith(prefix) and name.endswith(".core")
    }
    if (
        len(cores) != 12
        or start["source_core_sha256"] != cores
        or start["bridge_head_state_sha256"] != bridge_head_state_hash(static_state)
        or len(start["initial_state_sha256"]) != 64
    ):
        raise RuntimeError(
            "The factorial initial state differs from its static C0/Bridge/Head"
        )


def checked_outer_reference(folder, checkpoint_path, fold, evaluation_mode):
    receipt_path = folder / "OUTER_COMPLETE.json"
    receipt = read_json(receipt_path)
    if (
        receipt["checkpoint_sha256"] != file_hash(checkpoint_path)
        or receipt["outer_ids"] != fold["outer_test_ids"]
        or receipt["outer_groups"] != fold["outer_test_groups"]
        or receipt["evaluation_mode"] != evaluation_mode
        or receipt["final_target_query_labels_used"] != 0
        or receipt["optimizer_updates"] != 0
    ):
        raise RuntimeError(
            "Frozen outer reuse differs from its registered source/roles"
        )
    for record in receipt["files"]:
        if file_hash(folder / record["path"]) != record["sha256"]:
            raise RuntimeError("A frozen outer reuse artifact changed")
    return {
        "evaluation_mode": evaluation_mode,
        "receipt_path": str(receipt_path.relative_to(ROOT).as_posix()),
        "receipt_sha256": file_hash(receipt_path),
        "cell_metrics_path": str(
            (folder / "OUTER_CELL_METRICS.csv").relative_to(ROOT).as_posix()
        ),
        "prediction_path": str(
            (folder / "OUTER_PREDICTIONS.npz").relative_to(ROOT).as_posix()
        ),
        "metrics": receipt["metrics"],
    }


def register_frozen_reuse(args, plan, protocol):
    records = []
    registration = read_json(args.config)
    old_config = ROOT / "configs/drs_peft/e123_v1.json"
    old_config_sha256 = file_hash(old_config)
    old_registration = read_json(old_config)
    if (
        old_registration["primary_radius"] != registration["primary_radius"]
        or old_registration["reference_weight_fraction"]
        != registration["reference_weight_fraction"]
    ):
        raise RuntimeError(
            "Frozen reuse would change its physical update-norm operator"
        )
    folds = {fold["fold_id"]: fold for fold in protocol["folds"]}
    for unit in plan["units"]:
        if not reused_from_e123(unit, args.study):
            continue
        old_unit = {key: value for key, value in unit.items() if key != "regime"}
        source = unit_path(args.e123_private_root, old_unit)
        selected, checkpoint = checked_training_checkpoint(source)
        spec = read_json(source / "E123_UNIT_SPEC.json")
        fold = folds[unit["fold_id"]]
        _, static_checkpoint = checked_training_checkpoint(
            unit_path(
                args.e123_private_root,
                matching_unit("static", unit["fold_id"], unit["seed"], "grad_core"),
            )
        )
        if (
            spec["fit_ids"] != fold["fit_ids"]
            or spec["inner_validation_ids"] != fold["inner_validation_ids"]
            or spec["outer_ids"] != fold["outer_test_ids"]
            or spec["unit"] != old_unit
            or spec["parent_checkpoint_sha256"] != file_hash(static_checkpoint)
            or selected["identity"]["epochs"]
            != int(read_json(args.config)["epochs"]["condition"])
            or spec["registration_sha256"] != old_config_sha256
            or selected["identity"]["config_sha256"] != old_config_sha256
            or spec["protocol_file_sha256"] != file_hash(args.folds)
        ):
            raise RuntimeError("Frozen source reuse used another protocol or arm")
        validate_training_controls(selected["identity"], registration, unit, 1440)
        if (
            selected["identity"]["training_ids"] != fold["fit_ids"]
            or selected["identity"]["validation_ids"] != fold["inner_validation_ids"]
        ):
            raise RuntimeError(
                "Frozen source training roles differ from the current fold"
            )
        for source_record in selected["identity"]["source_snapshot"]:
            if file_hash(ROOT / source_record["path"]) != source_record["sha256"]:
                raise RuntimeError(
                    "A source dependency of the reused frozen operator changed"
                )
        references = [
            checked_outer_reference(
                args.e123_private_root
                / "outer"
                / unit_path(Path("."), old_unit)
                / evaluation,
                checkpoint,
                fold,
                evaluation,
            )
            for evaluation in EVALUATION_MODES
        ]
        records.append(
            {
                "unit": unit,
                "selected_checkpoint_path": str(
                    checkpoint.relative_to(ROOT).as_posix()
                ),
                "selected_checkpoint_sha256": file_hash(checkpoint),
                "source_receipt_path": str(
                    (source / "SOURCE_DEVELOPMENT_COMPLETE.json")
                    .relative_to(ROOT)
                    .as_posix()
                ),
                "source_receipt_sha256": file_hash(
                    source / "SOURCE_DEVELOPMENT_COMPLETE.json"
                ),
                "static_checkpoint_sha256": file_hash(static_checkpoint),
                "outer": references,
                "new_optimizer_updates": 0,
            }
        )
    value = {
        "status": "VERIFIED_FROZEN_REUSE",
        "plan_sha256": plan["plan_sha256"],
        "units": records,
        "reused_training_units": len(records),
        "new_optimizer_updates": 0,
        "final_target_query_labels_used": 0,
    }
    bind_shared(args, args.public_root / f"ABC_{args.study}_FROZEN_REUSE.json", value)
    return value


def training_config(registration):
    result = read_json(ROOT / "configs/drs_peft/execution_v1.json")
    result["study_id"] = registration["study_id"]
    result["training"]["learning_rate"] = float(registration["bridge_head_lr"])
    result["training"]["batch_size"] = 8
    return result


def train_factorial_unit(
    args, registration, data, inputs, scaler, episodes, unit, factory, start_path, start
):
    branch_args = regime_args(args, unit["regime"])
    destination = unit_path(branch_args.private_root, unit)
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        "unit": unit,
        "registration_sha256": file_hash(args.config),
        "protocol_file_sha256": file_hash(args.folds),
        "static_checkpoint_sha256": file_hash(start_path),
        "fit_ids": episodes["source_cell_roles"]["source_train"],
        "inner_validation_ids": episodes["source_cell_roles"]["source_meta_validation"],
        "outer_ids": episodes["source_cell_roles"]["source_outer_validation"],
        "start": start,
        "conditioning_input_sha256": {
            view: hashlib.sha256(inputs["diagnostic_" + view].tobytes()).hexdigest()
            for view in ("absolute", "relative")
        },
        "outer_labels_used_for_training_or_selection": 0,
        "final_target_query_labels_used": 0,
    }
    with external_lock(args.private_root, destination):
        bind(destination / f"ABC_{args.study}_UNIT_SPEC.json", identity)
        epochs = int(registration["epochs"]["condition"])
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
            f"ABC_{args.study}_{unit['regime']}",
            unit["seed"],
            epochs,
            args.device,
            rank=4,
            adapter_lr=float(registration["adapter_lr"]),
            stop_after=args.stop_after,
            stage=f"ABC_{args.study}_FOLD_LOCAL_SOURCE",
            model_factory=factory,
            initial_checkpoint_sha256=file_hash(start_path),
            extra_sources=(*EXTRA_SOURCES, "scripts/drs_peft/run_abc_b.py")
            if args.study == "B"
            else EXTRA_SOURCES,
        )
        if receipt["status"] == "ENGINEERING_INTERRUPTED":
            print(
                json.dumps(
                    dict(event="ABC_A_ENGINEERING_INTERRUPTED", unit=unit, **receipt)
                ),
                flush=True,
            )
            return False
        expected = epochs * math.ceil(len(identity["fit_ids"]) / 8)
        if receipt["updates"] != expected or receipt["target_query_labels_used"] != 0:
            raise RuntimeError(
                "A violated the registered training budget or target seal"
            )
        _publish_unit(destination, unit_path(branch_args.public_root / "units", unit))
    print(
        json.dumps(
            {
                "event": "ABC_A_TRAINING_UNIT_COMPLETE",
                "unit": unit,
                "updates": receipt["updates"],
                "inner_validation": receipt["validation"],
            }
        ),
        flush=True,
    )
    return True


def require_training_complete(args, plan):
    records = []
    starts = {}
    for unit in plan["units"]:
        if reused_from_e123(unit, args.study):
            continue
        folder = unit_path(args.private_root / unit["regime"], unit)
        if not (folder / "SOURCE_DEVELOPMENT_COMPLETE.json").exists():
            raise RuntimeError(
                "A outer scoring stays sealed while registered joint training remains incomplete"
            )
        checkpoint, path = checked_training_checkpoint(folder)
        registration = read_json(args.config)
        fold = next(
            item
            for item in load_protocol(args.folds)["folds"]
            if item["fold_id"] == unit["fold_id"]
        )
        spec = read_json(folder / f"ABC_{args.study}_UNIT_SPEC.json")
        static, static_path = checked_training_checkpoint(
            unit_path(
                args.e123_private_root,
                matching_unit("static", unit["fold_id"], unit["seed"], "grad_core"),
            )
        )
        identity = checkpoint["identity"]
        if (
            identity["config_sha256"] != file_hash(args.config)
            or spec["unit"] != unit
            or spec["registration_sha256"] != file_hash(args.config)
            or spec["protocol_file_sha256"] != file_hash(args.folds)
            or spec["fit_ids"] != fold["fit_ids"]
            or spec["inner_validation_ids"] != fold["inner_validation_ids"]
            or spec["outer_ids"] != fold["outer_test_ids"]
            or identity["training_ids"] != fold["fit_ids"]
            or identity["validation_ids"] != fold["inner_validation_ids"]
            or identity["method"] != f"ABC_{args.study}_{unit['regime']}"
            or identity["initial_checkpoint_sha256"] != file_hash(static_path)
            or spec["static_checkpoint_sha256"] != file_hash(static_path)
            or spec["outer_labels_used_for_training_or_selection"] != 0
            or spec["final_target_query_labels_used"] != 0
        ):
            raise RuntimeError("A checkpoint belongs to a changed registration")
        expected_parameters = int(registration["A"]["core_gate_trainable_parameters"])
        if unit["regime"] == "joint":
            expected_parameters = int(
                registration["A"]["joint_total_trainable_parameters"]
            )
        validate_training_controls(identity, registration, unit, expected_parameters)
        validate_static_start(spec["start"], static["model"])
        key = (unit["fold_id"], unit["seed"])
        fingerprint = spec["start"]["initial_state_sha256"]
        if key in starts and starts[key] != fingerprint:
            raise RuntimeError("Factorial arms started from different model tensors")
        starts[key] = fingerprint
        check = read_json(folder / "FIXED_ADAPTER_START_CHECK.json")
        if (
            check["status"] != "PASSED"
            or check["initial_prediction_max_difference"] != 0
        ):
            raise RuntimeError("The factorial static-start replay check failed")
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


def shuffled_condition_inputs(data, inputs, fold, seed):
    """Break cell-state alignment while retaining paired view row distributions."""
    from scripts.drs_peft.run_abc_b import role_derangement

    result = dict(inputs)
    for view in ("absolute", "relative"):
        result["diagnostic_" + view] = inputs["diagnostic_" + view].copy()
    records = []
    for role, key in (
        ("fit", "fit_ids"),
        ("inner_validation", "inner_validation_ids"),
        ("outer_test", "outer_test_ids"),
    ):
        ids = fold[key]
        donor_indices = role_derangement(ids, seed, fold["fold_id"], role)
        donors = [ids[index] for index in donor_indices]
        if len(donors) != len(ids) or set(donors) != set(ids):
            raise RuntimeError(
                "State-feature shuffle must be a within-role permutation"
            )
        if any(uid == donor for uid, donor in zip(ids, donors, strict=True)):
            raise RuntimeError("State-feature shuffle must have no self-donors")
        rows, donor_rows = data.rows(ids), data.rows(donors)
        for view in ("absolute", "relative"):
            key = "diagnostic_" + view
            result[key][rows] = inputs[key][donor_rows]
        records.extend(
            {"role": role, "cell_uid": uid, "feature_donor_uid": donor}
            for uid, donor in zip(ids, donors, strict=True)
        )
    manifest = {
        "fold_id": fold["fold_id"],
        "seed": seed,
        "mapping": records,
        "mapping_sha256": payload_hash(records),
        "label_values_used": 0,
        "views_shuffled_as_paired_rows": True,
        "final_target_query_labels_used": 0,
    }
    return result, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/drs_peft/abc_v1.json"
    )
    parser.add_argument("--study", choices=("A", "B"), default="A")
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
        "--e123-private-root", type=Path, default=ROOT / "runtime/drs_peft/e123_v1"
    )
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("train", "score", "verify-reuse"), required=True
    )
    parser.add_argument(
        "--fold-ids", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2]
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=(42, 123, 2026), default=[42, 123, 2026]
    )
    parser.add_argument("--modes", nargs="+", choices=(*MODES, *B_MODES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.private_root = (
        args.private_root or ROOT / "runtime/drs_peft/abc_v1" / args.study
    )
    args.public_root = args.public_root or ROOT / "results/drs_peft/abc_v1" / args.study
    args.modes = args.modes or list(MODES if args.study == "A" else B_MODES)
    if not set(args.modes).issubset(MODES if args.study == "A" else B_MODES):
        parser.error("The requested modes do not belong to the selected study")
    configure()
    registration, protocol = read_json(args.config), load_protocol(args.folds)
    plan = build_plan(registration, protocol, args.study)
    if args.dry_run:
        print(json.dumps(plan, indent=2), flush=True)
        return
    bind_shared(
        args,
        args.public_root / f"ABC_{args.study}_PLAN.json",
        dict(
            plan,
            registration_sha256=file_hash(args.config),
            folds_file_sha256=file_hash(args.folds),
        ),
    )
    reuse = register_frozen_reuse(args, plan, protocol)
    if args.phase == "verify-reuse":
        print(
            json.dumps(
                {
                    "status": "FROZEN_REUSE_COMPLETE",
                    "units": len(reuse["units"]) if reuse else 0,
                    "new_optimizer_updates": 0,
                }
            ),
            flush=True,
        )
        return
    if args.phase == "score":
        completed = require_training_complete(args, plan)
        bind_shared(
            args,
            args.public_root / "OUTER_EVALUATION_OPENED.json",
            {
                "plan_sha256": plan["plan_sha256"],
                "source_units": completed,
                "frozen_reuse_sha256": file_hash(
                    args.public_root / f"ABC_{args.study}_FROZEN_REUSE.json"
                )
                if reuse
                else None,
                "final_target_query_labels_used": 0,
            },
        )
    data = load_dataset("MATR", cache_root=args.data_root, build_if_missing=False)
    parent_episodes = load_registration()
    bind_shared(
        args,
        args.public_root / "DATA_PROTOCOL_VALIDATION.json",
        validate_against_data(protocol, data, parent_episodes),
    )
    for fold in protocol["folds"]:
        if fold["fold_id"] not in args.fold_ids:
            continue
        inputs, scaler, episodes = prepare_fold(data, parent_episodes, fold)
        conditioned_inputs, view_manifest = prepare_condition_inputs(data, inputs, fold)
        bind_shared(
            args,
            args.public_root / f"fold{fold['fold_id']}" / "FOLD_VIEW_MANIFEST.json",
            view_manifest,
        )
        for seed in args.seeds:
            shuffled_inputs = None
            if args.study == "B":
                shuffled_inputs, shuffle_manifest = shuffled_condition_inputs(
                    data, conditioned_inputs, fold, seed
                )
                bind_shared(
                    args,
                    args.public_root
                    / f"fold{fold['fold_id']}"
                    / f"seed{seed}"
                    / "SHUFFLE_MANIFEST.json",
                    shuffle_manifest,
                )
            construct_static, checkpoint, start_path, _ = load_start(
                args, inputs, fold, seed
            )
            units = [
                unit
                for unit in plan["units"]
                if not reused_from_e123(unit, args.study)
                and unit["fold_id"] == fold["fold_id"]
                and unit["seed"] == seed
                and unit["mode"] in args.modes
            ]
            for unit in units:
                active_inputs = (
                    shuffled_inputs
                    if unit["mode"] == "shuffled_dual"
                    else conditioned_inputs
                )
                factory = factorial_factory(
                    construct_static, checkpoint, unit, registration
                )
                if args.phase == "train":
                    start = start_identity(factory)
                    folder = unit_path(args.private_root / unit["regime"], unit)
                    initial_path = folder / "FIXED_ADAPTER_START_CHECK.json"
                    if not initial_path.exists():
                        bind(
                            initial_path,
                            check_initialization(
                                factory,
                                partial(_loaded_static, construct_static, checkpoint),
                                active_inputs,
                                data.rows(fold["fit_ids"]),
                                args.device,
                            ),
                        )
                    if not train_factorial_unit(
                        args,
                        registration,
                        data,
                        active_inputs,
                        scaler,
                        episodes,
                        unit,
                        factory,
                        start_path,
                        start,
                    ):
                        return
                else:
                    _, selected_path = checked_training_checkpoint(
                        unit_path(args.private_root / unit["regime"], unit)
                    )
                    for evaluation_mode in EVALUATION_MODES:
                        score_outer_unit(
                            regime_args(args, unit["regime"]),
                            data,
                            active_inputs,
                            fold,
                            unit,
                            factory,
                            selected_path,
                            evaluation_mode,
                        )
    print(
        json.dumps(
            {
                "status": "ABC_A_PHASE_COMPLETE",
                "phase": args.phase,
                "final_target_query_labels_used": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
