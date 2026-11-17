"""Family references and fixed physical Query pairings for specialization."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.drs_peft.e123_protocol import WINDOWS
from scripts.drs_peft.onesided_protocol import load_registration
from scripts.drs_peft.summarize_specialization import (
    DEFAULT_CONFIG,
    adapter_budgets,
    expected_slots,
    main,
    method_families,
    paired_differences,
    slot_coverage,
    target_tasks,
    validate_cells,
    validate_receipts,
)


@pytest.fixture(scope="module")
def contract():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return config, load_registration()


def cells(contract, method="SPECIALIZED_OUT", k=1, draw=0, error=0.06):
    config, episodes = contract
    task = next(iter(target_tasks(config, episodes).values()))
    family = method_families(config)[method]
    zero = 0.2 if family == "WARMUP" else 0.1
    requested = next(shot for shot in task["shots"] if shot["k"] == k)["draws"][draw]
    protocols = task["protocol_by_id"]
    rows = []
    for i, uid in enumerate(task["query_ids"]):
        for window, _, _ in WINDOWS:
            observed = not (i == 0 and window == "1001_plus")
            score = zero if k == 0 else error
            rows.append(
                {
                    "stage": "D4" if family == "SPECIALIZED" else "NEW_TARGET",
                    "domain": task["domain"],
                    "task_id": task["task_id"],
                    "fold_id": "final",
                    "seed": 42,
                    "method": method,
                    "reference_family": family,
                    "K": k,
                    "draw": draw,
                    "adaptation_id": method + str(k) + requested["support_set_sha256"],
                    "is_duplicate": requested["duplicate_of_draw"] is not None,
                    "cell_uid": uid,
                    "protocol_uid": protocols[uid],
                    "window": window,
                    "n_obs": 20 if observed else 0,
                    "mae": score if observed else np.nan,
                    "rmse": score if observed else np.nan,
                    "bias": score if observed else np.nan,
                    "zero_mae": zero if observed else np.nan,
                    "gain_mae": zero - score if observed else np.nan,
                    "negative_transfer": score > zero if observed else None,
                    "same_protocol_support_count": sum(
                        protocols[sid] == protocols[uid]
                        for sid in requested["support_ids"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_registered_matrix_counts_and_families(contract):
    config, episodes = contract
    slots = pd.DataFrame(expected_slots(config, episodes, [42, 123, 2026]))
    assert len(slots) == 930
    assert slots.K.gt(0).sum() == 900
    assert slots.K.eq(0).sum() == 30
    assert set(method_families(config).values()) == {"WARMUP", "SPECIALIZED"}


def test_family_specific_zero_and_requested_paired_differences(contract):
    config, episodes = contract
    frame = pd.concat(
        [
            cells(contract, "PRETRAINED_LORA", error=0.15),
            cells(contract, "SPECIALIZED_LORA", error=0.08),
            cells(contract, "SPECIALIZED_XS", error=0.07),
            cells(contract),
            cells(contract, "WARMUP_ZERO", k=0),
            cells(contract, "SPECIALIZED_ZERO", k=0),
        ]
    )
    frame = validate_cells(frame, config, episodes)
    assert set(frame.stage) == {"TARGET"}
    assert set(frame.input_stage) == {"D4", "NEW_TARGET"}
    contrasts = paired_differences(frame).query("window == 'all'")
    ours_lora = contrasts[
        (contrasts.left_method == "SPECIALIZED_OUT")
        & (contrasts.right_method == "SPECIALIZED_LORA")
    ].iloc[0]
    assert ours_lora.difference_mae == pytest.approx(-0.02)
    assert ours_lora.delta_K_difference == pytest.approx(0.02)
    specialization = contrasts[
        (contrasts.left_method == "SPECIALIZED_LORA")
        & (contrasts.right_method == "PRETRAINED_LORA")
    ].iloc[0]
    assert specialization.difference_mae == pytest.approx(-0.07)
    assert specialization.delta_K_difference == pytest.approx(-0.03)
    zero = contrasts[contrasts.K == 0].iloc[0]
    assert zero.difference_mae == pytest.approx(-0.1)


def test_wrong_reference_family_is_rejected(contract):
    config, episodes = contract
    frame = cells(contract)
    frame["reference_family"] = "WARMUP"
    with pytest.raises(ValueError, match="reference family"):
        validate_cells(frame, config, episodes)


def test_wrong_specialized_baseline_is_rejected(contract):
    config, episodes = contract
    wrong = cells(contract)
    wrong.loc[wrong.n_obs.gt(0), "zero_mae"] = 0.2
    wrong.loc[wrong.n_obs.gt(0), "gain_mae"] = 0.14
    with pytest.raises(ValueError, match="family ZERO reference"):
        validate_cells(
            pd.concat([wrong, cells(contract, "SPECIALIZED_ZERO", k=0)]),
            config,
            episodes,
        )


def test_unsupported_window_keeps_na(contract):
    config, episodes = contract
    frame = cells(contract)
    frame.loc[frame.n_obs.eq(0), "mae"] = 0
    with pytest.raises(ValueError, match="keep NA"):
        validate_cells(frame, config, episodes)


def test_duplicate_supports_keep_draws_and_unique_state(contract):
    config, episodes = contract
    task = next(iter(target_tasks(config, episodes).values()))
    draws = next(shot for shot in task["shots"] if shot["k"] == 1)["draws"]
    duplicate = next(draw for draw in draws if draw["duplicate_of_draw"] is not None)
    original = duplicate["duplicate_of_draw"]
    frame = validate_cells(
        pd.concat(
            [cells(contract, draw=original), cells(contract, draw=duplicate["draw"])]
        ),
        config,
        episodes,
    )
    assert frame.adaptation_id.nunique() == 1
    coverage = slot_coverage(frame, [], config, episodes, [42])
    assert coverage.status.eq("SCORED").sum() == 2
    assert len(coverage) == 310


def test_checkpoint_identity_is_shared_within_each_family(contract):
    config, episodes = contract
    task = next(iter(target_tasks(config, episodes).values()))
    frame = validate_cells(cells(contract), config, episodes)
    base = {"task_id": task["task_id"], "seed": 42, "K": 1}
    records = [
        base | {"method": "SPECIALIZED_OUT", "source_checkpoint_sha256": "static"},
        base | {"method": "SPECIALIZED_XS", "source_checkpoint_sha256": "other"},
    ]
    with pytest.raises(ValueError, match="shared reference checkpoint"):
        validate_receipts(records, frame, config, episodes)
    records[1] = base | {
        "method": "PRETRAINED_LORA",
        "source_checkpoint_sha256": "warmup",
    }
    validate_receipts(records, frame, config, episodes)


def test_parameter_budget_measures_actual_target_scope(contract):
    config, _ = contract
    record = {
        "method": "SPECIALIZED_XS",
        "task_id": "task",
        "seed": 42,
        "K": 1,
        "adaptation_id": "xs",
        "trainable_parameters": 192,
        "details": {"parameter_report": {"rank": 4, "frozen_xs_factor_bytes": 147648}},
    }
    budget = adapter_budgets([record], config).iloc[0]
    assert budget.trainable_parameters == 192
    assert budget.adapter_rank == 4
    assert budget.frozen_adapter_bytes == 147648
    with pytest.raises(ValueError, match="registered budget"):
        adapter_budgets([record | {"trainable_parameters": 384}], config)


def test_receipt_support_draw_and_unique_identity_are_checked(contract):
    config, episodes = contract
    frame = validate_cells(cells(contract), config, episodes)
    task = next(iter(target_tasks(config, episodes).values()))
    draws = next(shot for shot in task["shots"] if shot["k"] == 1)["draws"]
    other = next(
        draw for draw in draws if draw["support_ids"] != draws[0]["support_ids"]
    )
    record = {
        "method": "SPECIALIZED_OUT",
        "task_id": task["task_id"],
        "seed": 42,
        "K": 1,
        "draw": 0,
        "SIDs": other["support_ids"],
    }
    with pytest.raises(ValueError, match="Support draw"):
        validate_receipts([record], frame, config, episodes)
    record.pop("SIDs")
    record["adaptation_id"] = "shared"
    with pytest.raises(ValueError, match="different roles"):
        validate_receipts(
            [record, record | {"method": "SPECIALIZED_XS"}], frame, config, episodes
        )


def test_main_partial_summary_never_loads_numeric_archives(
    contract, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Specialization summary loaded a numeric archive")

    monkeypatch.setattr(np, "load", forbidden)
    execution = {"actual_new_optimizer_updates": 123, "inner_optimizer_updates": 90}
    (tmp_path / "EXECUTION_LEDGER.json").write_text(
        json.dumps(execution), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary",
            "--results-root",
            str(tmp_path),
            "--seeds",
            "42",
            "--report",
            str(tmp_path / "REVIEW.md"),
        ],
    )
    main()
    ledger = json.loads((tmp_path / "RUN_LEDGER.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "PARTIAL"
    assert ledger["requested_nonzero_slots"] == 300
    assert ledger["requested_family_zero_slots"] == 10
    assert ledger["pending_slots"] == 310
    assert ledger["future_label_arrays_loaded"] == 0
    assert ledger["execution_ledger"] == execution
    assert ledger["execution_ledger_sha256"] is not None
