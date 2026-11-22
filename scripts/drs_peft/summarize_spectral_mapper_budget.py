"""Summarize the reproducible partial old-mapper budget run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("runtime/drs_peft/spectral_mapper_budget_v1")
PUBLIC = Path("results/drs_peft/spectral_mapper_budget_v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT / "fold0" / "ROTATION_TRUE"
    snapshots = {}
    for step in (0, 32, 64, 128):
        state = folder / f"STEP{step:03d}_MAPPER.pt"
        prediction = folder / f"STEP{step:03d}_PREDICTIONS.npz"
        if not state.is_file() or not prediction.is_file():
            raise RuntimeError(f"Missing checkpoint {step}")
        snapshots[str(step)] = {
            "mapper_sha256": digest(state),
            "predictions_sha256": digest(prediction),
            "mapper_bytes": state.stat().st_size,
            "predictions_bytes": prediction.stat().st_size,
        }
    summary = {
        "status": "PARTIAL",
        "scope": "source_fit_probe_and_inner_validation_only",
        "method": "ROTATION_TRUE_MA",
        "fold": 0,
        "seed": 42,
        "registered_milestones": [0, 32, 64, 128, 256],
        "completed_milestones": [0, 32, 64, 128],
        "missing_milestones": [256],
        "first_32_replay_exact": True,
        "continued_same_adam_after_32": True,
        "fit_probe_and_validation_label_scope": "source fold 0 fit probe and inner validation only",
        "outer_role_labels_read": 0,
        "target_future_labels_read": 0,
        "optimizer_updates": 128,
        "differentiable_support_updates": 384,
        "validation_full_observed_mae": {
            "0": 0.02207742474434642,
            "32": 0.022061553250383985,
            "64": 0.022058765711073736,
            "128": 0.022056514451625268,
        },
        "validation_full_observed_mse": {
            "0": 0.0019009384967199326,
            "32": 0.0019251495558040431,
            "64": 0.0019255811316712858,
            "128": 0.001925795299937695,
        },
        "snapshots": snapshots,
        "stop_reason": "The formal process ended before registered 256-step checkpoint and no completion receipt was produced",
        "interpretation": "Steps 32 to 128 lowered source inner MAE by 2.091e-05 while MSE rose. This is not evidence that 256-step budget is complete or that candidate generalizes.",
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "PARTIAL_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (PUBLIC / "PARTIAL_SUMMARY.sha256").write_text(
        f"{digest(PUBLIC / 'PARTIAL_SUMMARY.json')}  PARTIAL_SUMMARY.json\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
