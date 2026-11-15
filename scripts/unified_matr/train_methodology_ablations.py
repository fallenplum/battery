"""Run A1 through A4 as independent single-change derivatives of frozen M4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from scripts.unified_matr import train_methodology_full as full
from scripts.unified_matr.train_seed42 import (
    capture_random_state, execution_lock, fold_scaler, load_frozen_data,
    predictions, restore_random_state, same_selection, save_checkpoint,
    seed_everything, sha256, to_batch, trajectory_metrics, write_history, write_json,
)

ARMS = ("A1_M4_LEGACY_FLOW", "A2_M4_NO_B_WRITE", "A3_M4_STATIC_RHO", "A4_M4_NO_LOCAL_PHYSICS")
SPLITS = full.SPLITS
DEPENDENCIES = full.DEPENDENCIES + (
    "models/unified_matr/methodology_ablations.py",
    "scripts/unified_matr/train_methodology_ablations.py",
)
training_loss = full.training_loss
mechanism_diagnostics = full.mechanism_diagnostics
flatten_tensors = full.flatten_tensors
configure_execution = full.configure_execution
read_json = full.read_json
bind_contract = full.bind_contract


def make_model(arm, inputs, settings):
    from models.unified_matr.methodology_ablations import MethodologyAblationModel

    if arm not in ARMS:
        raise ValueError("Unregistered ablation arm")
    return MethodologyAblationModel(
        arm, policy_dim=inputs["policy"].shape[-1],
        global_dim=inputs["global_features"].shape[-1],
        local_feature_dim=inputs["local_features"].shape[-1], seed=42,
        observation_bound=settings["observation_residual_amplitude"],
        soh_bounds=tuple(settings["observation_soh_bounds"]),
    )


def model_contract(data, data_root, config_path, ablation_contract_path):
    from models.unified_matr.methodology_ablations import model_registry

    config = read_json(config_path)
    original = read_json(REPOSITORY / config["base_training_config"])
    full.baseline.validate_config(original)
    if sha256(REPOSITORY / config["base_training_config"]) not in {row["sha256"] for row in data[-1]["files"]}:
        raise RuntimeError("The reused base settings must be covered by the original data freeze")
    if (tuple(config["models"]) != ARMS or config["training_seeds"] != [42]
            or config["training"] != original["training"] or config["reporting"] != original["reporting"]
            or config["split"] != original["split"] or config["execution_order"] != "arm_first_then_original_six_splits"):
        raise RuntimeError("Ablations must retain the registered training and split settings")
    if config["full_training_reproduction"].get("root_decision_pending"):
        raise RuntimeError("The implementation must record the root reproduction-budget decision before registration")
    if (config["formal_units"] != 24 or config["formal_optimizer_updates"] != 49800
            or config["engineering_optimizer_updates_total"] != 56
            or config["total_executed_optimizer_updates"] != 58256
            or not config["full_training_reproduction"]["enabled"]
            or tuple(config["full_training_reproduction"]["arms"]) != ARMS
            or config["full_training_reproduction"]["optimizer_updates_total"] != 8400):
        raise RuntimeError("Formal, reproduction and engineering budgets must retain the complete58256 update accounting")
    splits = data[-2]["splits"]
    if tuple(splits) != SPLITS or len(data[1]) != 169 or not data[2].all():
        raise RuntimeError("Ablations retain the frozen eligible169 cohort and original six splits")
    update_by_split = {name: 150 * math.ceil(len(roles["train"]) / 8) for name, roles in splits.items()}
    if sum(update_by_split.values()) * len(ARMS) != 49800:
        raise RuntimeError("The formal24-unit budget must be49800 optimizer updates")
    reference = REPOSITORY / config["m4_reference_root"]
    m4_contract = read_json(reference / "MODEL_CONTRACT.json")
    m4_verification = read_json(reference / "RUN_VERIFICATION.json")
    if (m4_contract["data_freeze_sha256"] != sha256(data_root / "DATA_FREEZE.json")
            or m4_verification["status"] != "VERIFIED"):
        raise RuntimeError("The saved full M4 reference must be completed and verified on the same data")
    for source in m4_contract["sources"]:
        if sha256(REPOSITORY / source["path"]) != source["sha256"]:
            raise RuntimeError("Sealed M4 source changed at " + source["path"])
    registry = {}
    for arm in ARMS:
        seed_everything()
        registry[arm] = model_registry(make_model(arm, data[0], config["training"]))
    sources = [{"path": name, "sha256": sha256(REPOSITORY / name)} for name in DEPENDENCIES]
    return {
        "study_id": config["study_id"], "arms": list(ARMS), "seed": 42,
        "planned_units": 24, "planned_optimizer_updates": 49800,
        "expected_updates_by_split": update_by_split,
        "engineering_optimizer_updates": 56,
        "full_training_reproduction": config["full_training_reproduction"],
        "config_sha256": sha256(config_path), "config": config,
        "training": config["training"], "reporting": config["reporting"],
        "data_freeze_sha256": sha256(data_root / "DATA_FREEZE.json"),
        "data_fingerprint_sha256": sha256(data_root / "dataset_fingerprint.sha256"),
        "split_contract_sha256": sha256(data_root / "split_contract.json"),
        "frozen_files": data[-1]["files"], "sources": sources,
        "initialization": {"models": registry},
        "m4_reference": {"root": config["m4_reference_root"],
                         "model_contract_sha256": sha256(reference / "MODEL_CONTRACT.json"),
                         "verification_sha256": sha256(reference / "RUN_VERIFICATION.json"),
                         "analysis_sha256": sha256(reference / "analysis/ANALYSIS_COMPLETE.json")},
        "ablation_contract": {"path": str(ablation_contract_path.relative_to(REPOSITORY)).replace("\\", "/"),
                              "sha256": sha256(ablation_contract_path)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "torch_num_threads": torch.get_num_threads(), "precision": "float32"},
        "execution_order": [{"arm": arm, "split": split} for arm in ARMS for split in SPLITS],
        "historical_exposure": config["historical_exposure"],
    }


def real_smoke(arm, data, config, registration, output_root, device):
    inputs, identifiers, eligible, _, query, targets, observed, rate_targets, rate_mask, splits, _ = data
    roles = splits["splits"][SPLITS[0]]
    index = {uid: number for number, uid in enumerate(identifiers)}
    training = np.asarray([index[uid] for uid in roles["train"] if eligible[index[uid]]])
    scaled, _ = fold_scaler(inputs, training)
    rows = np.random.default_rng(42).permutation(training)[:8]
    repeats, initial_delta = [], []
    for _ in range(2):
        seed_everything()
        model = make_model(arm, inputs, config["training"]).to(device).train()
        output = model(**to_batch(scaled, rows, device), query_cycles=torch.as_tensor(query, device=device), return_diagnostics=True)
        loss = training_loss(output, rows, targets, observed, rate_targets, rate_mask, config["training"], device)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite ablation first-batch loss")
        loss.backward()
        initial_delta.append([float((layer["Bprime"] - layer["B"]).abs().max()) for layer in output["diagnostics"]["history"]])
        repeats.append({"loss": float(loss.detach()),
                        "prediction_sha256": hashlib.sha256(output["trajectory"].detach().cpu().numpy().tobytes()).hexdigest(),
                        "gradient": full.gradient_receipt(model)})
        del model, output
    gradients = repeats[0]["gradient"]["parameters"]
    write = {name: value for name, value in gradients.items() if "write_projection" in name}
    B_write_expected = arm != "A2_M4_NO_B_WRITE"
    checks = {
        "fresh_seed42_forward_backward_exact": repeats[0] == repeats[1],
        "all_gradients_finite": all(item["finite"] for item in gradients.values()),
        "initial_B_modulation_zero": all(value == 0 for layer in initial_delta for value in layer),
        "B_write_branch_matches_ablation": bool(write) == B_write_expected,
        "active_B_write_gradients_nonzero": all(item["present"] and item["l2"] > 0 for item in write.values()),
    }
    smoke_root = output_root / "engineering_smoke" / arm
    smoke_root.mkdir(parents=True, exist_ok=True)
    write_json(smoke_root / "FIRST_BATCH_REPRODUCIBILITY.json", {
        "arm": arm, "checks": checks, "repeats": repeats, "initial_B_delta_max": initial_delta,
        "cell_uid": [identifiers[number] for number in rows], "optimizer_updates": 0,
        "observed_targets": int(observed[rows].sum()), "rate_target_support": rate_mask[rows].sum(0).tolist(),
    })
    if not all(checks.values()):
        raise RuntimeError("Ablation first-batch smoke failed for " + arm)
    smoke_registration = {**registration, "phase": "engineering_smoke", "arm": arm, "epochs": 1}
    bind_contract(smoke_root / "RUN_REGISTRATION.json", smoke_registration)
    smoke_config = {**config, "training": {**config["training"], "epochs": 1}}
    train_unit(arm, SPLITS[0], roles, data, smoke_config, None, smoke_root, device,
               smoke_registration, evaluate_test=False)
    paths = sorted(path for path in smoke_root.rglob("*") if path.is_file() and path.name != "SMOKE_QA.json")
    receipt = {"status": "PASS", "arm": arm, "checks": checks,
               "model_contract_sha256": sha256(output_root / "MODEL_CONTRACT.json"),
               "engineering_epochs": 1, "engineering_optimizer_updates": 14,
               "formal_optimizer_updates": 0, "test_observations_consumed": 0,
               "fresh_formal_seed42_initialization_required": True,
               "files": [{"path": str(path.relative_to(output_root)).replace("\\", "/"), "sha256": sha256(path)} for path in paths]}
    write_json(smoke_root / "SMOKE_QA.json", receipt)
    return receipt


def smoke_receipts(output_root):
    receipts = []
    for arm in ARMS:
        path = output_root / "engineering_smoke" / arm / "SMOKE_QA.json"
        if not path.exists():
            return None
        receipt = read_json(path)
        if (receipt["status"] != "PASS" or not all(receipt["checks"].values())
                or receipt["model_contract_sha256"] != sha256(output_root / "MODEL_CONTRACT.json")):
            raise RuntimeError("A passing current smoke is required for " + arm)
        for entry in receipt["files"]:
            if sha256(output_root / entry["path"]) != entry["sha256"]:
                raise RuntimeError("Ablation smoke artifact changed")
        receipts.append(receipt)
    return receipts


def train_unit(arm, split_name, roles, data, config, data_root, run_root, device, registration, evaluate_test=True):
    from models.unified_matr.model import state_hash

    inputs, identifiers, eligible, cells, query, targets, observed, rate_targets, rate_mask, _, _ = data
    identifier_index = {uid: number for number, uid in enumerate(identifiers)}
    indices = {role: np.array([identifier_index[uid] for uid in roles[role]
                              if eligible[identifier_index[uid]]], dtype=int)
               for role in ("train", "validation", "test")}
    for role, rows in indices.items():
        if not len(rows) or not observed[rows].sum():
            raise RuntimeError("No observed supervision in split role " + role)
    unit_root = run_root / split_name / arm
    unit_root.mkdir(parents=True, exist_ok=True)
    unit_path = unit_root / "UNIT_CONTRACT.json"
    unit_contract = {"arm": arm, "seed": 42, "split_name": split_name,
                     "registration_sha256": sha256(run_root / "RUN_REGISTRATION.json"),
                     "roles": {role: [identifiers[i] for i in rows] for role, rows in indices.items()},
                     "training": config["training"],
                     "expected_updates": config["training"]["epochs"] * math.ceil(len(indices["train"]) / config["training"]["batch_size"]),
                     "initial_model": registration["initialization"]["models"][arm], "device": str(device)}
    if unit_path.exists() and json.loads(unit_path.read_text(encoding="utf-8")) != unit_contract:
        raise RuntimeError("Existing unit has a different frozen contract")
    write_json(unit_path, unit_contract)
    complete_path = unit_root / "COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "EVALUATED" or complete.get("successful_updates") != unit_contract["expected_updates"]:
            raise RuntimeError("Completed unit has an inconsistent evaluation state or update budget")
        for item in complete["files"]:
            if sha256(unit_root / item["path"]) != item["sha256"]:
                raise RuntimeError("Completed unit artifact mismatch")
        print(json.dumps({"status": "VERIFIED_COMPLETE", "split": split_name, "arm": arm}), flush=True)
        return
    scaled, scaler = fold_scaler(inputs, indices["train"])
    write_json(unit_root / "TRAIN_SCALER.json", {"training_ids": unit_contract["roles"]["train"], "statistics": scaler})
    seed_everything()
    model = make_model(arm, inputs, config["training"]).to(device)
    if state_hash(model) != registration["initialization"]["models"][arm]["initial_state_sha256"]:
        raise RuntimeError("Actual training model differs from the frozen common initialization")
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]),
                                 eps=settings["eps"])
    checkpoint = unit_root / "TRAIN_STATE.pt"
    best_path = unit_root / "SELECTED.pt"
    epoch_start, best_mae, successful_updates = 0, math.inf, 0
    history, selected_checkpoint = [], None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["contract"] != unit_contract:
            raise RuntimeError("Resume checkpoint contract mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch_start, best_mae, successful_updates = state["next_epoch"], state["best_mae"], state["successful_updates"]
        history, selected_checkpoint = state["history"], state["selected_checkpoint"]
        expected_updates = epoch_start * math.ceil(len(indices["train"]) / settings["batch_size"])
        if len(history) != epoch_start or successful_updates != expected_updates:
            raise RuntimeError("Resume checkpoint has an incomplete epoch or optimizer-update history")
        restore_random_state(state["random_state"])
        actual = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None
        if actual is None or not same_selection(actual, selected_checkpoint):
            if (unit_root / "SELECTION_FROZEN.json").exists():
                raise RuntimeError("A frozen selected checkpoint differs from the atomic training state")
            save_checkpoint(best_path, selected_checkpoint)
        write_history(unit_root / "training.jsonl", history)
    tensor_query = torch.as_tensor(query, device=device)
    started = time.time() - (history[-1]["elapsed_seconds"] if history else 0)
    for epoch in range(epoch_start, settings["epochs"]):
        model.train()
        epoch_rng = np.random.default_rng(42 + epoch)
        order = epoch_rng.permutation(indices["train"])
        losses = []
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            batch = to_batch(scaled, rows, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch, query_cycles=tensor_query)
            loss = training_loss(output, rows, targets, observed, rate_targets, rate_mask, settings, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            successful_updates += 1
            losses.append(float(loss.detach()))
        val_prediction, _ = predictions(model, scaled, indices["validation"], query, device, settings["batch_size"])
        val_metrics, _, _ = trajectory_metrics(val_prediction, targets[indices["validation"]], observed[indices["validation"]])
        if not np.isfinite(val_metrics["cell_mean_mae"]):
            raise FloatingPointError("Validation selection criterion is nonfinite")
        selected = val_metrics["cell_mean_mae"] < best_mae
        if selected:
            best_mae = val_metrics["cell_mean_mae"]
            selected_checkpoint = {"model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                                   "epoch": epoch + 1, "validation_mae": best_mae, "contract": unit_contract}
            save_checkpoint(best_path, selected_checkpoint)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                  "validation_mae": val_metrics["cell_mean_mae"], "selected": selected,
                  "successful_updates": successful_updates, "elapsed_seconds": time.time() - started}
        history.append(record)
        save_checkpoint(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "next_epoch": epoch + 1, "best_mae": best_mae,
                                    "successful_updates": successful_updates, "contract": unit_contract,
                                    "random_state": capture_random_state(), "history": history,
                                    "selected_checkpoint": selected_checkpoint})
        write_history(unit_root / "training.jsonl", history)
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"split": split_name, "arm": arm, **record}), flush=True)
    if successful_updates != unit_contract["expected_updates"]:
        raise RuntimeError("Final optimizer updates differ from the fixed epoch/batch budget")
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    # Freeze the chosen weights and validation criterion before any test prediction.
    selection = {"selected_epoch": selected["epoch"],
               "validation_mae": selected["validation_mae"], "checkpoint_sha256": sha256(best_path),
               "successful_updates": successful_updates, "seed": 42,
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
    selection_path = unit_root / "SELECTION_FROZEN.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise RuntimeError("Test-exposed checkpoint selection is immutable")
    write_json(selection_path, selection)
    if not evaluate_test:
        write_json(unit_root / "ENGINEERING_EPOCH_COMPLETE.json", {
            "status": "ENGINEERING_ONLY", "epochs": settings["epochs"],
            "successful_updates": successful_updates, "validation_mae": selected["validation_mae"],
            "test_evaluation_performed": False, "selected_checkpoint_sha256": sha256(best_path),
        })
        return
    test = indices["test"]
    test_prediction, test_trend = predictions(model, scaled, test, query, device, settings["batch_size"])
    metrics, per_cell_mae, per_cell_rmse = trajectory_metrics(test_prediction, targets[test], observed[test])
    rows = []
    for position, number in enumerate(test):
        uid = identifiers[number]
        valid = observed[number]
        for slot in np.flatnonzero(valid):
            rows.append({"cell_uid": uid, "original_cycle_id": int(query[slot]), "target": float(targets[number, slot]),
                         "prediction": float(test_prediction[position, slot]), "trend": float(test_trend[position, slot])})
    pd.DataFrame(rows).to_parquet(unit_root / "test_predictions.parquet", index=False)
    per_cell = [{"cell_uid": identifiers[number], "mae": float(per_cell_mae[position]),
                 "rmse": float(per_cell_rmse[position]), "observations": int(observed[number].sum()),
                 "protocol_uid": str(cells.loc[identifiers[number], "protocol_uid"])}
                for position, number in enumerate(test) if observed[number].any()]
    pd.DataFrame(per_cell).to_csv(unit_root / "test_cell_metrics.csv", index=False)
    rate_rows = []
    query_index = {int(cycle): position for position, cycle in enumerate(query)}
    for position, number in enumerate(test):
        for window, (left, right) in enumerate(((200, 500), (500, 1000))):
            first = float(test_prediction[position, query_index[left]])
            last = float(test_prediction[position, query_index[right]])
            rate_rows.append({"cell_uid": identifiers[number], "protocol_uid": str(cells.loc[identifiers[number], "protocol_uid"]),
                              "target_name": f"r{left}_{right}", "start_cycle": left, "end_cycle": right,
                              "predicted_start_soh": first, "predicted_end_soh": last,
                              "predicted_rate": (first - last) / (right - left),
                              "target_supported": bool(rate_mask[number, window]),
                              "target_rate": float(rate_targets[number, window]) if rate_mask[number, window] else None})
    pd.DataFrame(rate_rows).to_csv(unit_root / "test_rate_predictions.csv", index=False)
    eol_limit = config["reporting"]["eol80_prediction_query_max_cycle"]
    eol_queries = np.arange(100, eol_limit + 1, dtype=np.float32)
    _, eol_trends = predictions(model, scaled, test, eol_queries, device, settings["batch_size"])
    eol_rows = []
    for position, number in enumerate(test):
        crossings = np.flatnonzero(eol_trends[position] <= 0.8)
        predicted_cycle = float(eol_limit)
        if len(crossings):
            right = int(crossings[0])
            predicted_cycle = float(eol_queries[right])
            if right > 0:
                y0, y1 = map(float, eol_trends[position, right - 1:right + 1])
                if y0 > y1:
                    predicted_cycle = float(eol_queries[right - 1] + (y0 - 0.8) / (y0 - y1))
        eol_rows.append({"cell_uid": identifiers[number], "predicted_trend_eol80_cycle": predicted_cycle,
                         "no_crossing_through_registered_horizon": not bool(len(crossings)),
                         "prediction_query_max_cycle": eol_limit})
    pd.DataFrame(eol_rows).to_csv(unit_root / "test_eol_predictions.csv", index=False)
    write_json(unit_root / "test_metrics.json", {**metrics, "arm": arm, "split": split_name, "seed": 42})
    mechanism_diagnostics(model, scaled, test, identifiers, query, device, settings["batch_size"], unit_root)
    files = ["test_mechanism_diagnostics.npz", "test_mechanism_summary.csv", "UNIT_CONTRACT.json", "TRAIN_SCALER.json", "SELECTION_FROZEN.json", "training.jsonl",
             "test_predictions.parquet", "test_cell_metrics.csv", "test_metrics.json", "test_eol_predictions.csv", "test_rate_predictions.csv",
             "SELECTED.pt", "TRAIN_STATE.pt"]
    write_json(complete_path, {"status": "EVALUATED", "seed": 42, "successful_updates": successful_updates, "files": [
        {"path": name, "sha256": sha256(unit_root / name)} for name in files]})
    print(json.dumps({"status": "EVALUATED", "split": split_name, "arm": arm, **metrics}), flush=True)


def verify_complete_files(run_root, arms=ARMS):
    for arm in arms:
        for split in SPLITS:
            root = run_root / split / arm
            complete = read_json(root / "COMPLETE.json")
            if complete["status"] != "EVALUATED":
                raise RuntimeError("Formal evaluation is incomplete")
            for entry in complete["files"]:
                if sha256(root / entry["path"]) != entry["sha256"]:
                    raise RuntimeError("Completed artifact changed before reproduction")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("register", "smoke", "train", "reproduce"), required=True)
    parser.add_argument("--arm", choices=ARMS, help="Optional individual engineering smoke arm")
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/unified_matr_ablations_seed42.json")
    parser.add_argument("--ablation-contract", type=Path,
                        default=REPOSITORY / "research/M4_ABLATIONS_SEED42_CONTRACT_ZH.md")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.arm and args.mode != "smoke":
        raise ValueError("Formal and complete-reproduction runs follow the fixed arm-first order")
    args.data, args.output_root = args.data.resolve(), args.output_root.resolve()
    args.config, args.ablation_contract = args.config.resolve(), args.ablation_contract.resolve()
    configure_execution()
    config = read_json(args.config)
    data = load_frozen_data(args.data)
    registration = model_contract(data, args.data, args.config, args.ablation_contract)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with execution_lock(args.output_root / ".ablations.lock"):
        bind_contract(args.output_root / "MODEL_CONTRACT.json", registration)
        if args.mode == "register":
            print(json.dumps({"status": "REGISTERED", "units": 24, "formal_updates": 49800,
                              "reproduction_updates": 8400, "engineering_updates": 56,
                              "total_optimizer_updates": 58256}), flush=True)
            return
        if args.mode == "smoke":
            if any((args.output_root / "runs").glob("*/*/TRAIN_STATE.pt")):
                raise RuntimeError("Engineering smoke precedes formal optimization")
            for arm in (args.arm,) if args.arm else ARMS:
                real_smoke(arm, data, config, registration, args.output_root, args.device)
            receipts = smoke_receipts(args.output_root)
            if receipts is not None:
                write_json(args.output_root / "SMOKE_QA.json", {
                    "status": "PASS", "arms": list(ARMS), "engineering_optimizer_updates": 56,
                    "formal_optimizer_updates": 0, "test_observations_consumed": 0,
                    "model_contract_sha256": sha256(args.output_root / "MODEL_CONTRACT.json"),
                    "arm_smoke_receipts": [{"arm": arm, "sha256": sha256(args.output_root / "engineering_smoke" / arm / "SMOKE_QA.json")} for arm in ARMS]})
            return
        if smoke_receipts(args.output_root) is None or read_json(args.output_root / "SMOKE_QA.json")["status"] != "PASS":
            raise RuntimeError("All four separate one-epoch engineering smokes must pass")
        run_root = args.output_root / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        bind_contract(run_root / "RUN_REGISTRATION.json", registration)
        if args.mode == "reproduce":
            matrix = read_json(run_root / "MATRIX_COMPLETE.json")
            if (matrix["status"] != "COMPLETE" or matrix["completed_units"] != 24
                    or matrix["successful_updates"] != 49800
                    or matrix["registration_sha256"] != sha256(run_root / "RUN_REGISTRATION.json")):
                raise RuntimeError("All24 formal fits precede the registered complete reproductions")
            verify_complete_files(run_root)
            plan = registration["full_training_reproduction"]
            if not plan["enabled"] or tuple(plan["arms"]) != ARMS:
                raise RuntimeError("Complete reproductions must match their preregistered four-arm budget")
            reproduction = args.output_root / "training_reproduction"
            reproduction.mkdir(parents=True, exist_ok=True)
            bind_contract(reproduction / "RUN_REGISTRATION.json", registration)
            for arm in ARMS:
                train_unit(arm, SPLITS[0], data[-2]["splits"][SPLITS[0]], data, config, args.data,
                           reproduction, args.device, registration)
            write_json(reproduction / "REPRODUCTION_COMPLETE.json", {
                "status": "EVALUATED", "arms": list(ARMS), "split": SPLITS[0], "seed": 42,
                "additional_optimizer_updates": 8400, "formal_optimizer_updates": 49800,
                "comparison_status": "pending_independent_complete_state_history_and_prediction_comparison",
                "registration_sha256": sha256(reproduction / "RUN_REGISTRATION.json")})
            return
        for arm in ARMS:
            for split in SPLITS:
                train_unit(arm, split, data[-2]["splits"][split], data, config, args.data,
                           run_root, args.device, registration)
        write_json(run_root / "MATRIX_COMPLETE.json", {
            "status": "COMPLETE", "seed": 42, "arms": list(ARMS), "completed_units": 24,
            "successful_updates": 49800, "registration_sha256": sha256(run_root / "RUN_REGISTRATION.json"),
            "model_contract_sha256": sha256(args.output_root / "MODEL_CONTRACT.json")})


if __name__ == "__main__":
    main()

