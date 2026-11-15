"""Exercise seed42 runner guards and resume helpers without training a model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from scripts.unified_matr import train_seed42 as runner
from .selfcheck import check, make_batch


def expect_rejection(function, records, name):
    try:
        function()
    except (RuntimeError, FileNotFoundError):
        rejected = True
    else:
        rejected = False
    check(rejected, name, records)


def run_checks():
    torch.set_num_threads(1)
    records = []
    config_path = runner.REPOSITORY / "configs/unified_matr_seed42.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runner.validate_config(config)
    check(True, "registered_seed42_budget_accepted", records)
    for field, value in (("epochs", 149), ("batch_size", 16), ("learning_rate", 0.001), ("early_stopping", True)):
        altered = copy.deepcopy(config)
        altered["training"][field] = value
        expect_rejection(lambda: runner.validate_config(altered), records, "changed_budget_rejected_" + field)
    altered = copy.deepcopy(config)
    altered["training_seeds"] = [42, 43]
    expect_rejection(lambda: runner.validate_config(altered), records, "additional_seed_rejected")
    altered = copy.deepcopy(config)
    altered["models"] = altered["models"][:-1]
    expect_rejection(lambda: runner.validate_config(altered), records, "incomplete_arm_matrix_rejected")
    temporary_parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="runner_QA_", dir=temporary_parent) as directory:
        root = Path(directory)
        if root.resolve().parent != temporary_parent:
            raise RuntimeError("Engineering temporary directory escaped its verified workspace parent")
        expect_rejection(lambda: runner.load_frozen_data(root), records, "missing_DATA_FREEZE_rejected_before_data_access")
        (root / "DATA_FREEZE.json").write_text(json.dumps({"status": "PENDING", "training_seeds": [42]}), encoding="utf-8")
        expect_rejection(lambda: runner.load_frozen_data(root), records, "pending_DATA_FREEZE_rejected")
        (root / "DATA_FREEZE.json").write_text(json.dumps({"status": "DATA_FREEZE_PASS", "training_seeds": [42],
                                                        "files": [], "source_file_sha256": ["dummy"] * 4}), encoding="utf-8")
        expect_rejection(lambda: runner.load_frozen_data(root), records, "status_string_without_original_axis_evidence_rejected")
        checkpoint = root / "checkpoint.pt"
        selected = {"model": {"engineering_tensor": torch.tensor([1.0, 2.0])}, "epoch": 1,
                    "validation_mae": 0.5, "contract": {"seed": 42}}
        runner.save_checkpoint(checkpoint, selected)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        check(runner.same_selection(restored, selected), "atomic_selected_snapshot_roundtrip", records)
        altered = copy.deepcopy(restored)
        altered["model"]["engineering_tensor"][0] += 1.0
        check(not runner.same_selection(altered, selected), "selected_tensor_change_detected", records)
        journal = root / "training.jsonl"
        runner.write_history(journal, [{"epoch": 1}, {"epoch": 2}])
        runner.write_history(journal, [{"epoch": 1}])
        check(journal.read_text(encoding="utf-8").splitlines() == ['{"epoch": 1}'], "history_rolls_back_to_atomic_checkpoint_without_duplicate_epochs", records)
        locked = root / ".matrix.lock"
        with runner.execution_lock(locked):
            expect_rejection(lambda: runner.execution_lock(locked).__enter__(), records, "concurrent_matrix_writer_rejected")
        with runner.execution_lock(locked):
            check(True, "released_matrix_lock_reusable", records)
    batch = make_batch()
    inputs = {key: value.cpu().numpy() for key, value in batch.items() if key in runner.INPUT_KEYS}
    scaled, statistics = runner.fold_scaler(inputs, np.array([0]))
    altered = {key: value.copy() for key, value in inputs.items()}
    for key in runner.FEATURE_MASKS:
        altered[key][1] = altered[key][1] * 100 + 10
    _, other_statistics = runner.fold_scaler(altered, np.array([0]))
    check(statistics == other_statistics, "all_scaler_statistics_train_cell_only", records)
    check(np.array_equal(scaled["raw_viq"][..., 0], scaled["reference_viq"][..., 0]), "reference_voltage_preserves_original_view_exactly", records)
    altered = {key: value.copy() for key, value in inputs.items()}
    altered["global_features"][0, 0, 0] = np.nan
    expect_rejection(lambda: runner.fold_scaler(altered, np.array([0])), records, "supported_nonfinite_feature_rejected")
    prediction = torch.tensor([[1.0, 9.0, 9.0], [2.0, 4.0, 9.0]], requires_grad=True)
    target = torch.tensor([[0.0, float("nan"), float("nan")], [0.0, 0.0, float("nan")]])
    mask = torch.tensor([[True, False, False], [True, True, False]])
    loss = runner.masked_cell_mae(prediction, target, mask)
    check(float(loss) == 2.0, "trajectory_loss_equal_cell_weight_with_nan_missing_targets", records)
    loss.backward()
    check(bool(torch.isfinite(prediction.grad).all()), "masked_trajectory_loss_finite_gradient", records)
    prediction = torch.ones(2, 3, requires_grad=True)
    loss = runner.masked_cell_mae(prediction, torch.full((2, 3), float("nan")), torch.zeros(2, 3, dtype=torch.bool))
    loss.backward()
    check(float(loss) == 0.0 and bool(torch.isfinite(prediction.grad).all()), "all_missing_batch_zero_loss_preserves_fixed_updates", records)
    runner.seed_everything()
    random_state = runner.capture_random_state()
    expected = (random.random(), np.random.random(3), torch.rand(3))
    runner.restore_random_state(random_state)
    actual = (random.random(), np.random.random(3), torch.rand(3))
    check(expected[0] == actual[0] and np.array_equal(expected[1], actual[1]) and torch.equal(expected[2], actual[2]),
          "resume_python_numpy_torch_rng_exact_replay", records)
    if torch.cuda.is_available():
        random_state = runner.capture_random_state()
        expected = torch.rand(5, device="cuda")
        runner.restore_random_state(random_state)
        actual = torch.rand(5, device="cuda")
        check(torch.equal(expected, actual), "resume_CUDA_rng_exact_replay", records)
    registry = runner.initialization_registry(inputs)
    check(len(registry["models"]) == 7 and len(registry["common_initialization_checks"]) == 21,
          "actual_P15_G20_L26_initialization_registry_all_pairs", records)
    return {"status": "PASS", "scope": "runner_guard_loss_scaler_resume_engineering_only", "checks": len(records),
            "formal_training": False, "optimizer_updates": 0, "scientific_predictions": 0,
            "records": records, "initialization": registry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks()
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "scope", "checks", "optimizer_updates")}))


if __name__ == "__main__":
    main()
