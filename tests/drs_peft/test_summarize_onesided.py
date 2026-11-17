"""Scored-fact aggregation preserves fixed Query, NA and repeated axes."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.drs_peft.e123_protocol import WINDOWS
from scripts.drs_peft.onesided_protocol import load_registration
from scripts.drs_peft.summarize_onesided import (
    METRIC_COLUMNS,
    adaptation_costs,
    aggregate,
    expected_slots,
    ledger,
    main,
    normalize_receipts,
    paired_contrasts,
    project_costs,
    separated_variability,
    slot_coverage,
    validate_receipt_identities,
    validate_scored_cells,
    validate_shared_source,
    write_latest,
)


@pytest.fixture(scope="module")
def episodes():
    return load_registration()


def scored_slot(
    episodes, task=None, *, method="OUT", k=1, draw=0, seed=42, stage="D2", mae=0.08
):
    task = task or episodes["source_tasks"]["outer"][0]
    requested = next(shot for shot in task["shots"] if shot["k"] == k)["draws"][draw]
    protocols = task.get("protocol_by_id", episodes["source_protocols"])
    rows = []
    for i, uid in enumerate(task["query_ids"]):
        for window, _, _ in WINDOWS:
            n_obs = 0 if i == 0 and window == "1001_plus" else 20
            error = 0.1 if k == 0 else mae
            gain = 0.1 - error
            rows.append(
                {
                    "stage": stage,
                    "domain": task["domain"],
                    "task_id": task["task_id"],
                    "fold_id": task.get("fold_id", "final"),
                    "seed": seed,
                    "method": method,
                    "K": k,
                    "draw": draw,
                    "adaptation_id": f"{stage}-{task['task_id']}-{seed}-{method}-{k}-{requested['support_set_sha256']}",
                    "is_duplicate": requested["duplicate_of_draw"] is not None,
                    "cell_uid": uid,
                    "protocol_uid": protocols[uid],
                    "window": window,
                    "n_obs": n_obs,
                    "mae": error if n_obs else np.nan,
                    "rmse": error if n_obs else np.nan,
                    "bias": -error if n_obs else np.nan,
                    "zero_mae": 0.1 if n_obs else np.nan,
                    "gain_mae": gain if n_obs else np.nan,
                    "negative_transfer": gain < 0 if n_obs else None,
                    "same_protocol_support_count": sum(
                        protocols[sid] == protocols[uid]
                        for sid in requested["support_ids"]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def test_expected_primary_state_counts(episodes):
    slots = pd.DataFrame(expected_slots(episodes, ["D2", "D3", "D4"], [42, 123, 2026]))
    nonzero = slots[slots.K > 0].groupby("stage").size().to_dict()
    assert nonzero == {"D2": 945, "D3": 135, "D4": 675}
    assert slots[(slots.stage == "D4") & (slots.K == 0)].shape[0] == 15
    target = slots[
        (slots.stage == "D4")
        & (slots.method == "OUT")
        & (slots.seed == 42)
        & (slots.K == 1)
        & (slots.draw == 0)
    ]
    matr = target[target.domain == "MATR"]
    assert len(matr) == 4
    assert sum(map(len, matr.query_ids)) == 8


def test_complete_slot_and_pending_slots(episodes):
    frame = validate_scored_cells(scored_slot(episodes), episodes)
    coverage = slot_coverage(frame, episodes, [], ["D2"], [42])
    assert coverage.status.eq("SCORED").sum() == 1
    assert coverage.status.eq("PENDING").sum() == 317
    run = ledger(frame, coverage, [], episodes)
    assert run["status"] == "PARTIAL"
    assert run["terminal_failed_or_not_estimable_slots"] == 0
    assert run["requested_nonzero_slots"] == 315


def test_terminal_failures_count_as_failures_without_negative_count(episodes):
    frame = pd.DataFrame(columns=METRIC_COLUMNS)
    task = episodes["source_tasks"]["outer"][0]
    records = [
        {
            "stage": "D3",
            "task_id": task["task_id"],
            "seed": 42,
            "method": "CROSS",
            "status": "NOT_ESTIMABLE",
            "reason": "no positive eigenvalue",
        }
    ]
    coverage = slot_coverage(frame, episodes, records, ["D3"], [42])
    run = ledger(frame, coverage, records, episodes)
    assert run["terminal_failed_or_not_estimable_slots"] == 15
    assert run["pending_slots"] == 120
    assert run["status"] == "PARTIAL"


def test_query_and_shared_zero_and_gain_guards(episodes):
    frame = scored_slot(episodes)
    wrong = frame.copy()
    wrong.loc[0, "cell_uid"] = "unregistered"
    with pytest.raises(ValueError, match="Query identity"):
        validate_scored_cells(wrong, episodes)
    wrong = frame.copy()
    wrong.loc[0, "gain_mae"] = 9
    with pytest.raises(ValueError, match="gain differs"):
        validate_scored_cells(wrong, episodes)
    changed = scored_slot(episodes, method="CAL2")
    changed.loc[0, "zero_mae"] = 0.2
    changed.loc[0, "gain_mae"] = 0.12
    with pytest.raises(ValueError, match="shared f0"):
        validate_scored_cells(pd.concat([frame, changed]), episodes)


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("fold_id", 2, "registered source fold"),
        ("method", "AUTO", "registered evaluation stage"),
        ("same_protocol_support_count", 900, "coverage was miscounted"),
        ("negative_transfer", True, "flags disagree"),
        ("is_duplicate", True, "Duplicate Support"),
        ("adaptation_id", "different-state", "one adapted state"),
    ],
)
def test_metadata_and_state_guards(episodes, column, value, match):
    frame = scored_slot(episodes)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        validate_scored_cells(frame, episodes)


def test_zero_one_state_and_na_windows(episodes):
    frame = validate_scored_cells(scored_slot(episodes, method="ZERO", k=0), episodes)
    summary = aggregate(frame)
    tail = summary[summary.window == "1001_plus"].iloc[0]
    assert tail.unsupported_cell_window_rows == 1
    assert tail.physical_query_cells == 19
    assert tail.gain_mae == 0
    wrong = frame.copy()
    wrong.loc[wrong.n_obs == 0, "mae"] = 0
    with pytest.raises(ValueError, match="retain NA"):
        validate_scored_cells(wrong, episodes)
    wrong = scored_slot(episodes, method="ZERO", k=0, draw=0)
    wrong["draw"] = 1
    with pytest.raises(ValueError, match="K0"):
        validate_scored_cells(wrong, episodes)


def test_negative_transfer_preserved(episodes):
    frame = validate_scored_cells(scored_slot(episodes, mae=0.12), episodes)
    summary = aggregate(frame)
    assert np.allclose(summary.gain_mae, -0.02)
    assert summary.negative_transfer_ratio.eq(1).all()


def test_contrasts_use_paired_physical_cell_seed_draws(episodes):
    frame = pd.concat(
        [
            scored_slot(episodes, method="CORE", mae=0.1),
            scored_slot(episodes, method="OUT", mae=0.08),
        ]
    )
    frame = validate_scored_cells(frame, episodes)
    row = paired_contrasts(frame).query("window == 'all'").iloc[0]
    assert row.candidate_gain_over_reference == pytest.approx(0.02)
    assert row.physical_query_cells == 20
    assert row.paired_cell_seed_draw_rows == 20
    summary = aggregate(frame)
    assert summary.worst_cell_mae.notna().all()
    assert summary.worst_protocol_mae.notna().all()


def test_receipts_cross_check_only_registered_identity_metadata(episodes):
    task = episodes["source_tasks"]["outer"][0]
    frame = validate_scored_cells(scored_slot(episodes), episodes)
    record = {
        "stage": "D2",
        "task_id": task["task_id"],
        "seed": 42,
        "method": "OUT",
        "K": 1,
        "draw": 0,
        "adaptation_id": frame.adaptation_id.iloc[0],
        "SIDs": task["shots"][1]["draws"][0]["support_ids"],
        "QIDs": task["query_ids"],
    }
    validate_receipt_identities(frame, [record], episodes)
    wrong = record | {"SIDs": ["unregistered"]}
    with pytest.raises(ValueError, match="unregistered Support"):
        validate_receipt_identities(frame, [wrong], episodes)
    wrong = record | {"method": "CAL2"}
    with pytest.raises(ValueError, match="disagrees"):
        validate_receipt_identities(frame, [wrong], episodes)


def test_duplicate_support_slots_reuse_adaptation_id(episodes):
    first = scored_slot(episodes, draw=1)
    duplicate = scored_slot(episodes, draw=4)
    frame = validate_scored_cells(pd.concat([first, duplicate]), episodes)
    assert frame.adaptation_id.nunique() == 1
    summary = aggregate(frame)
    assert summary.unique_adapted_states.eq(1).all()
    assert summary.support_draw_slots.eq(2).all()
    records = [
        {"adaptation_id": "shared", "method": "LoRA", "K": 1, "actual_optsteps": 10},
        {"adaptation_id": "shared", "method": "LoRA", "K": 1, "actual_optsteps": 10},
    ]
    run = ledger(
        frame, slot_coverage(frame, episodes, [], ["D2"], [42]), records, episodes
    )
    assert run["actual_adapter_optimizer_steps"] == 10
    assert run["nonlinear_adapter_fits"] == 1


def test_seed_and_draw_variation_are_separate(episodes):
    frames = [
        scored_slot(episodes, seed=seed, draw=draw, mae=error)
        for seed, draw, error in [
            (42, 0, 0.06),
            (42, 1, 0.08),
            (123, 0, 0.08),
            (123, 1, 0.1),
        ]
    ]
    frame = validate_scored_cells(pd.concat(frames), episodes)
    row = separated_variability(frame).query("window == 'all'").iloc[0]
    assert row.seeds == 2
    assert row.seed_std_after_averaging_draws == pytest.approx(
        np.std([0.07, 0.09], ddof=1)
    )
    assert row.mean_within_seed_task_draw_std == pytest.approx(
        np.std([0.06, 0.08], ddof=1)
    )


def test_target_generic_alias_and_fixed_budget(episodes):
    task = next(task for task in episodes["target_tasks"] if task["domain"] == "HUST")
    frame = validate_scored_cells(
        scored_slot(episodes, task, method="LoRA-FA", stage="D4"), episodes
    )
    assert set(frame.method) == {"LoRA_FA"}
    coverage = slot_coverage(frame, episodes, [], ["D4"], [42])
    assert coverage.status.eq("SCORED").sum() == 1
    other = validate_scored_cells(
        scored_slot(episodes, task, method="LoRA", stage="D4"), episodes
    )
    with pytest.raises(ValueError, match="multiple selected baselines"):
        slot_coverage(pd.concat([frame, other]), episodes, [], ["D4"], [42])


def test_record_source_checkpoint_and_cost_deduplication():
    records = [
        {
            "stage": "D4",
            "task_id": "a",
            "seed": 42,
            "method": "OUT",
            "K": 1,
            "adaptation_id": "a",
            "sourcecheckpoint_sha": "one",
            "actual_optsteps": 0,
            "dictbytes": 1024,
        },
        {
            "stage": "D4",
            "task_id": "b",
            "seed": 42,
            "method": "OUT",
            "K": 1,
            "adaptation_id": "b",
            "sourcecheckpoint_sha": "two",
        },
    ]
    with pytest.raises(ValueError, match="shared source checkpoint"):
        validate_shared_source(records)
    records[1]["sourcecheckpoint_sha"] = "one"
    validate_shared_source(records)
    costs = adaptation_costs(records + [records[0]])
    assert len(costs) == 2
    assert costs.loc[costs.adaptation_id == "b", "dictbytes"].isna().all()


def test_nested_runner_receipts_keep_actual_and_selected_steps_separate():
    original = {
        "method": "LoRA_FA",
        "K": 1,
        "adaptation_id": "one",
        "support_ids": ["support"],
        "query_ids": ["query"],
        "new_execution_counters": {"nonlinear_optimizer_updates": 30},
        "details": {
            "optimizer_updates": 10,
            "parameter_report": {
                "adaptation_parameters": 9216,
                "frozen_A_bytes": 36864,
            },
            "support_mae_before": 0.1,
            "support_mae_after": 0.08,
        },
        "elapsed_seconds": 4,
        "peak_gpu_allocated_bytes": 1024,
    }
    records = normalize_receipts([original])
    assert records[0]["actual_optsteps"] == 30
    assert records[0]["state_optimizer_steps"] == 10
    assert records[0]["SIDs"] == ["support"]
    costs = adaptation_costs(records).iloc[0]
    assert costs.actual_optsteps == 30
    assert costs.state_optimizer_steps == 10
    assert costs.trainable_parameters == 9216
    assert costs.frozen_A_bytes == 36864
    assert costs.support_mae_after == 0.08


def test_project_costs_sum_current_process_counters_once(tmp_path):
    engine_root = tmp_path / "ENGINE_COSTS"
    engine_root.mkdir()
    path = engine_root / "D2_inner_fold0_seed42_pid1.json"
    record = {
        "role": "D2_inner",
        "fold_id": 0,
        "seed": 42,
        "counters": {"nonlinear_optimizer_updates": 30, "geometry_solves": 2},
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    record["counters"]["nonlinear_optimizer_updates"] = 60
    path.write_text(json.dumps(record), encoding="utf-8")
    marker = tmp_path / "references/fold0/seed42/REFERENCE_COMPLETE.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "fold_id": 0,
                "seed": 42,
                "reference_checkpoint_sha256": "reference",
                "reference_training_updates": 15,
            }
        ),
        encoding="utf-8",
    )
    result = project_costs(tmp_path, ["D2", "D3"], [42])
    assert (
        result["actual_inner_and_evaluation_counters"]["nonlinear_optimizer_updates"]
        == 60
    )
    assert result["completed_source_reference_models"] == 1
    assert result["completed_source_training_fits"] == 2
    assert result["completed_source_reference_optimizer_updates"] == 15


def test_common_dimension_supplement_preserved(episodes):
    frame = validate_scored_cells(scored_slot(episodes, method="OUT_COMMON"), episodes)
    assert set(frame.method) == {"OUT_COMMON"}
    coverage = slot_coverage(frame, episodes, [], ["D2"], [42])
    assert not coverage.status.eq("SCORED").any()
    run = ledger(frame, coverage, [], episodes)
    assert run["source_common_dimension_supplement_cell_window_rows"] == len(frame)


def test_existing_summary_history_preserves_previous_content(tmp_path):
    path = tmp_path / "RUN_LEDGER.json"
    history = tmp_path / "history"
    write_latest(path, b"first", history)
    write_latest(path, b"second", history)
    assert path.read_bytes() == b"second"
    assert [file.read_bytes() for file in history.iterdir()] == [b"first"]


def test_main_empty_increment_and_stage_does_not_load_arrays(
    episodes, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("The summarizer loaded a numeric archive")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--results-root",
            str(tmp_path),
            "--stage",
            "D2",
            "--seeds",
            "42",
            "--report",
            str(tmp_path / "REVIEW.md"),
        ],
    )
    main()
    run = json.loads((tmp_path / "D2_RUN_LEDGER.json").read_text(encoding="utf-8"))
    assert run["requested_nonzero_slots"] == 315
    assert run["pending_slots"] == 318
    assert run["target_future_label_arrays_loaded_by_summarizer"] == 0
    scored_slot(episodes).to_csv(tmp_path / "CELL_METRICS.csv", index=False)
    main()
    run = json.loads((tmp_path / "D2_RUN_LEDGER.json").read_text(encoding="utf-8"))
    assert run["scored_slots"] == 1
    assert (tmp_path / "summary_history").is_dir()
    assert not (tmp_path / "TARGET_FEWSHOT_RESULTS.csv").exists()
