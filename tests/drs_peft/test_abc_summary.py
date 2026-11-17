"""Pairing, interaction, coverage, and output-grain regression checks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.drs_peft import summarize_abc as summary
from scripts.drs_peft.data import ROOT, file_hash, write_json
from scripts.drs_peft.e123_protocol import load_protocol


def fixture_config():
    return {
        "evaluation_modes": ["trained_ball", "matched_norm"],
        "statistics": {"confidence_level": 0.95, "bootstrap_draws": 2000, "seed": 42},
    }


def fixture_table():
    values = {
        "A": {
            "frozen": {"constant": 0.1, "absolute": 0.099, "dual": 0.098},
            "joint": {"constant": 0.08, "absolute": 0.075, "dual": 0.07},
        },
        "B": {
            "frozen": {"relative": 0.0995, "shuffled_dual": 0.1},
            "joint": {"relative": 0.079, "shuffled_dual": 0.08},
        },
        "C": {
            "frozen": {
                "fixed_rank4": 0.11,
                "rank4_plus_small": 0.109,
                "matched_fixed": 0.105,
                "matched_residual": 0.104,
                "matched_lora": 0.103,
            }
        },
    }
    records = []
    for study, regimes in values.items():
        for regime, modes in regimes.items():
            for mode, base in modes.items():
                for evaluation in ("trained_ball", "matched_norm"):
                    for window, _, _ in summary.WINDOWS:
                        for fold in range(3):
                            for seed in (42, 123, 2026):
                                for cell in range(fold * 2, fold * 2 + 2):
                                    mae = base + cell * 0.00001
                                    records.append(
                                        {
                                            "study": study,
                                            "regime": regime,
                                            "mode": mode,
                                            "evaluation_mode": evaluation,
                                            "window": window,
                                            "fold_id": fold,
                                            "seed": seed,
                                            "cell_uid": f"cell{cell}",
                                            "protocol_uid": f"group{cell // 2}",
                                            "n_obs": 1,
                                            "mae": mae,
                                            "rmse": mae,
                                            "bias": -mae,
                                        }
                                    )
    return pd.DataFrame(records)


def test_factorial_contrasts_separate_gate_gain_and_joint_interaction():
    contrasts = summary.neural_contrasts(fixture_table(), fixture_config(), ["A"])
    assert len(contrasts) == 60
    interaction = [
        value
        for value in contrasts
        if value["comparison"] == "joint_gate_gain_minus_frozen_gate_gain"
        and value["candidate"] == "dual"
    ]
    assert len(interaction) == 10
    for value in interaction:
        assert value["mean_absolute_MAE_improvement"] == pytest.approx(0.008)
        assert value["cell_uid_interval"] == pytest.approx([0.008, 0.008])
        assert value["protocol_uid_interval"] == pytest.approx([0.008, 0.008])
        assert value["physical_cells"] == 6
        assert value["protocol_groups"] == 3
        assert "relative_MAE_improvement_percent" not in value


def test_shared_a_views_are_available_to_b_without_repeating_main_cell_table():
    table = fixture_table()
    contrasts = summary.neural_contrasts(table, fixture_config(), ["A", "B", "C"])
    assert len(contrasts) == 60 + 100 + 50
    first = next(
        value
        for value in contrasts
        if value["study"] == "B"
        and value["candidate"] == "dual"
        and value["reference"] == "shuffled_dual"
        and value["regime"] == "joint"
    )
    assert first["mean_absolute_MAE_improvement"] == pytest.approx(0.01)
    assert not table.duplicated([*summary.MODEL_KEYS, *summary.PAIR_KEYS]).any()


def test_missing_seed_cell_pair_fails_instead_of_changing_comparison_denominator():
    table = fixture_table()
    missing = table[
        ~(
            table.study.eq("A")
            & table["mode"].eq("dual")
            & table.seed.eq(42)
            & table.cell_uid.eq("cell0")
        )
    ]
    with pytest.raises(RuntimeError, match="omits a physical cell"):
        summary.neural_contrasts(missing, fixture_config(), ["A"])


def test_observation_support_mismatch_fails_for_paired_models():
    table = fixture_table()
    table.loc[table["mode"].eq("dual"), "n_obs"] = 2
    with pytest.raises(RuntimeError, match="different observation support"):
        summary.neural_contrasts(table, fixture_config(), ["A"])


@pytest.mark.parametrize(
    "grain", [(), ("fold_id",), ("seed",), ("fold_id", "seed"), ("protocol_uid",)]
)
def test_aggregation_preserves_unique_physical_counts_and_group_macro(grain):
    frame = summary.aggregate(fixture_table(), grain)
    assert frame.physical_cells.max() == (6 if not grain or grain == ("seed",) else 2)
    assert np.allclose(frame.mae, frame.protocol_macro_mae)
    assert not frame.duplicated([*summary.MODEL_KEYS, *grain]).any()


def test_interval_seed_is_deterministic_and_keeps_negative_effects():
    table = fixture_table()
    pair = summary.paired_scores(
        table[table.study.eq("A")],
        "constant",
        "dual",
        {"regime": "frozen", "window": "all", "evaluation_mode": "trained_ball"},
    )
    first = summary.summarize_pair(
        pair, {"name": "negative_fixture"}, fixture_config()["statistics"]
    )
    second = summary.summarize_pair(
        pair, {"name": "negative_fixture"}, fixture_config()["statistics"]
    )
    assert first == second
    assert first["mean_absolute_MAE_improvement"] < 0
    assert not first["all_fold_directions_positive"]
    assert first["cell_uid_interval"][1] < 0


def test_partial_scope_cannot_open_without_registered_plan_files(tmp_path):
    args = SimpleNamespace(
        studies=["A"], public_root=tmp_path, config=tmp_path / "missing.json"
    )
    config = {
        "a_modes": ["constant", "absolute", "dual"],
        "training_seeds": [42, 123, 2026],
        "evaluation_modes": ["trained_ball", "matched_norm"],
        "primary_radius": 1.0,
        "reference_weight_fraction": 0.01,
        "study_id": "fixture",
    }
    protocol = {
        "folds": [{"fold_id": index} for index in range(3)],
        "protocol_sha256": "fixture",
    }
    with pytest.raises(FileNotFoundError):
        summary.collect_records(args, config, protocol)
    assert not list(tmp_path.iterdir())


def test_c_helper_preserves_actual_arm_and_supplies_common_metric_schema():
    actual = {"fold_id": 1, "seed": 42, "arm": "matched_lora"}
    transformed = summary.helper_unit("C", actual)
    assert transformed["arm"] == transformed["mode"] == "matched_lora"
    assert transformed["stage"] == "condition"
    assert transformed["regime"] == "frozen"
    assert actual == {"fold_id": 1, "seed": 42, "arm": "matched_lora"}


def test_matched_and_capacity_different_c_comparisons_are_explicit():
    contrasts = summary.neural_contrasts(fixture_table(), fixture_config(), ["C"])
    by_pair = {
        (row["candidate"], row["reference"]): row["comparison"] for row in contrasts
    }
    assert by_pair[("matched_residual", "matched_fixed")] == "matched_capacity"
    assert (
        by_pair[("rank4_plus_small", "fixed_rank4")] == "capacity_different_secondary"
    )
    assert by_pair[("matched_fixed", "fixed_rank4")] == "capacity_different_secondary"


def test_summary_resume_preserves_identical_bytes_and_rejects_changed_values(tmp_path):
    path = tmp_path / "summary.json"
    summary.write_frozen_json(path, {"value": -0.01})
    original = path.read_bytes()
    timestamp = path.stat().st_mtime_ns
    summary.write_frozen_json(path, {"value": -0.01})
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == timestamp
    with pytest.raises(RuntimeError, match="Frozen ABC summary differs"):
        summary.write_frozen_json(path, {"value": 0.01})
    assert path.read_bytes() == original


def test_full270_outer_matrix_and99_new36_reused_closure_schema(tmp_path, monkeypatch):
    config_path = ROOT / "configs/drs_peft/abc_v1.json"
    config, protocol = summary.read(config_path), load_protocol()
    args = SimpleNamespace(
        studies=["A", "B", "C"],
        private_root=tmp_path / "private",
        public_root=tmp_path / "public",
        e123_private_root=tmp_path / "parent",
        config=config_path,
    )
    fake_sources, fake_reuse = {}, {}
    for study in args.studies:
        plan = summary.study_plan(config, protocol, study)
        write_json(
            args.public_root
            / study
            / ("C_PLAN.json" if study == "C" else f"ABC_{study}_PLAN.json"),
            plan,
        )
        source_entries = []
        fake_reuse[study] = []
        for unit in plan["units"]:
            key = summary.unit_key(unit)
            checkpoint_sha = "fixture_checkpoint_" + key
            fold = protocol["folds"][unit["fold_id"]]
            reused = summary.is_reused(study, unit)
            if reused:
                reuse = {
                    "unit": unit,
                    "selected_checkpoint_sha256": checkpoint_sha,
                    "outer": [],
                }
                fake_reuse[study].append(reuse)
            else:
                source = {
                    "study": study,
                    "unit": unit,
                    "folder": str(summary.source_folder(args, study, unit)),
                    "checkpoint_sha256": checkpoint_sha,
                    "source_receipt_sha256": "fixture_source_receipt",
                    "updates": 960,
                }
                fake_sources[(study, key)] = source
                source_entries.append(
                    {
                        "unit": unit,
                        "checkpoint_sha256": checkpoint_sha,
                        "source_receipt_sha256": "fixture_source_receipt",
                    }
                )
            for evaluation in config["evaluation_modes"]:
                if reused:
                    actual_unit = {
                        name: value for name, value in unit.items() if name != "regime"
                    }
                    folder = (
                        args.e123_private_root
                        / "outer"
                        / summary.legacy_unit_path(Path("."), actual_unit)
                        / evaluation
                    )
                elif study == "C":
                    actual_unit = unit
                    folder = (
                        args.private_root
                        / study
                        / "outer"
                        / f"fold{unit['fold_id']}"
                        / f"seed{unit['seed']}"
                        / unit["arm"]
                        / evaluation
                    )
                else:
                    actual_unit = unit
                    folder = (
                        args.private_root
                        / study
                        / unit["regime"]
                        / "outer"
                        / summary.legacy_unit_path(Path("."), unit)
                        / evaluation
                    )
                receipt = {
                    "unit": actual_unit,
                    "evaluation_mode": evaluation,
                    "checkpoint_sha256": checkpoint_sha,
                    "outer_ids": fold["outer_test_ids"],
                    "outer_groups": fold["outer_test_groups"],
                    "final_target_query_labels_used": 0,
                    "optimizer_updates": 0,
                    "registration_sha256": file_hash(config_path),
                    "files": [],
                }
                write_json(folder / "OUTER_COMPLETE.json", receipt)
                if reused:
                    reuse["outer"].append(
                        {
                            "evaluation_mode": evaluation,
                            "receipt_path": str(folder / "OUTER_COMPLETE.json"),
                            "receipt_sha256": file_hash(folder / "OUTER_COMPLETE.json"),
                        }
                    )
        write_json(
            args.public_root / study / "OUTER_EVALUATION_OPENED.json",
            {
                "plan_sha256": plan["plan_sha256"],
                "final_target_query_labels_used": 0,
                "source_units": source_entries,
            },
        )
    monkeypatch.setattr(
        summary,
        "check_source_unit",
        lambda args, config, protocol, study, unit: fake_sources[
            (study, summary.unit_key(unit))
        ],
    )
    monkeypatch.setattr(
        summary, "verify_reuse", lambda args, protocol, study, plan: fake_reuse[study]
    )
    records, closure = summary.collect_records(args, config, protocol)
    assert len(records) == len({str(value["folder"]) for value in records}) == 270
    assert closure["new_training_units"] == 99
    assert closure["reused_training_units"] == 36
    assert closure["new_optimizer_updates"] == 95040
    barrier_path = args.public_root / "C/OUTER_EVALUATION_OPENED.json"
    barrier = summary.read(barrier_path)
    barrier["source_units"].pop()
    write_json(barrier_path, barrier)
    with pytest.raises(RuntimeError, match="barrier omits source units"):
        summary.collect_records(args, config, protocol)


def test_independent_endpoint_rates_keep_physical_units_and_missing_endpoints():
    query = np.arange(101, 1201, dtype=float)
    target = np.stack((1 - 0.0002 * query, 1 + 0.0001 * query))
    observed = np.ones_like(target, dtype=bool)
    observed[:, (query >= 200) & (query <= 207)] = False
    reference = {
        "query": query,
        "lookup": {"a": 0, "b": 1},
        "target": target,
        "observed": observed,
    }
    result = summary.independent_endpoint_rates(reference, ["a", "b"], 200, 500)
    np.testing.assert_allclose(result, [0.0002, -0.0001], atol=1e-16)
    observed[0, (query >= 208) & (query <= 210)] = False
    with pytest.raises(RuntimeError, match="insufficient source-cache support"):
        summary.independent_endpoint_rates(reference, ["a", "b"], 200, 500)


def test_heldout_gradient_summary_requires_all_cells_seeds_and_energy_blocks(tmp_path):
    config_path = tmp_path / "config.json"
    config = {"training_seeds": [42, 123, 2026]}
    write_json(config_path, config)
    args = SimpleNamespace(
        private_root=tmp_path / "private",
        public_root=tmp_path / "public",
        config=config_path,
    )
    ids = [f"c{index}" for index in range(6)]
    protocol = {
        "source_ids": ids,
        "source_qualification": {
            uid: {"protocol_uid": f"g{index // 2}"} for index, uid in enumerate(ids)
        },
        "folds": [
            {"fold_id": fold, "outer_test_ids": ids[2 * fold : 2 * fold + 2]}
            for fold in range(3)
        ],
    }
    for fold in protocol["folds"]:
        for seed in config["training_seeds"]:
            folder = (
                args.private_root
                / "C/heldout_gradients"
                / f"fold{fold['fold_id']}"
                / f"seed{seed}"
            )
            folder.mkdir(parents=True)
            rows = []
            for uid in fold["outer_test_ids"]:
                for layer in range(12):
                    rows.append(
                        {
                            "fold_id": fold["fold_id"],
                            "seed": seed,
                            "cell_uid": uid,
                            "protocol_uid": protocol["source_qualification"][uid][
                                "protocol_uid"
                            ],
                            "layer": f"l{layer}",
                            "gradient_norm": 1.0,
                            "fixed_energy_ratio": 0.2,
                            "left_only_energy_ratio": 0.1,
                            "right_only_energy_ratio": 0.3,
                            "double_perp_energy_ratio": 0.4,
                            "residual_energy_ratio": 0.8,
                            "residual_to_fit_warmup_residual_cosine": 0.25,
                            "full_to_fit_warmup_mean_cosine": 0.5,
                            "block_energy_reconstruction_error": 0,
                        }
                    )
            csv = folder / "HELDOUT_GRADIENT_BLOCKS.csv"
            pd.DataFrame(rows).to_csv(csv, index=False)
            receipt = {
                "fold_id": fold["fold_id"],
                "seed": seed,
                "outer_ids": fold["outer_test_ids"],
                "registration_sha256": file_hash(config_path),
                "optimizer_updates": 0,
                "heldout_gradients_used_for_directions_or_selection": 0,
                "final_target_query_labels_used": 0,
                "files": [{"path": csv.name, "sha256": file_hash(csv)}],
            }
            write_json(folder / "HELDOUT_GRADIENT_COMPLETE.json", receipt)
            public = (
                args.public_root
                / "C/heldout_gradients"
                / f"fold{fold['fold_id']}"
                / f"seed{seed}"
            )
            write_json(public / "HELDOUT_GRADIENT_COMPLETE.json", receipt)
    frame, macro, audit = summary.gradient_summaries(args, config, protocol)
    assert len(frame) == 216
    assert len(macro) == 12
    assert np.allclose(macro.residual_energy_ratio, 0.8)
    assert audit["physical_cells"] == 6
    assert audit["new_gradient_evaluations"] == 18
    assert audit["endpoint_replay_evaluations"] == 18
    marker = (
        args.private_root
        / "C/heldout_gradients/fold0/seed42/HELDOUT_GRADIENT_COMPLETE.json"
    )
    receipt = summary.read(marker)
    receipt["heldout_gradients_used_for_directions_or_selection"] = 1
    write_json(marker, receipt)
    with pytest.raises(RuntimeError, match="gradient provenance changed"):
        summary.gradient_summaries(args, config, protocol)


def test_stage_a_report_replays_identically_and_keeps_fold_seed_directions(tmp_path):
    table = fixture_table()
    table = table[table.study.eq("A")]
    contrasts = summary.neural_contrasts(table, fixture_config(), ["A"])
    report = tmp_path / "A_RESULTS.md"
    closure = {
        "new_training_units": 27,
        "reused_training_units": 27,
        "new_optimizer_updates": 25920,
    }
    audit = {
        "outer_units": 108,
        "cell_window_rows": len(table),
        "maximum_metric_difference": 0.0,
    }
    norms = [{"physical_cell_layers": 24, "norm_control_passed": True}]
    summary.write_report(
        report, ["A"], summary.aggregate(table), contrasts, [], closure, audit, norms
    )
    original = report.read_bytes()
    summary.write_report(
        report, ["A"], summary.aggregate(table), contrasts, [], closure, audit, norms
    )
    assert report.read_bytes() == original
    text = report.read_text(encoding="utf-8")
    assert "joint_gate_gain_minus_frozen_gate_gain" in text
    assert "各折MAE收益" in text
    assert "各种子MAE收益" in text
    assert "0.008000" in text
