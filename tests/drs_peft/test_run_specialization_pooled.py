"""Total-domain support budgets, frozen-source reuse and pooled target guards."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.drs_peft import run_specialization_pooled as pooled
from scripts.drs_peft.data import file_hash, write_json


def setup(tmp_path):
    config = pooled.read_json(pooled.source.CONFIG)
    override = pooled.read_json(pooled.TARGET_CONTRACT)
    episodes = pooled.read_json(pooled.ROOT / override["episodes_file"])
    original = pooled.read_json(pooled.ROOT / config["episodes_file"])
    args = SimpleNamespace(
        config=pooled.source.CONFIG,
        target_contract=pooled.TARGET_CONTRACT,
        source_public_root=tmp_path / "s",
        source_private_root=tmp_path / "v",
        public_root=tmp_path / "u",
        private_root=tmp_path / "r",
        reference_root=tmp_path / "f",
        backbone_dir=tmp_path / "official",
        device="cpu",
        seeds=[42, 123, 2026],
        methods=list(pooled.source.METHODS),
        domains=["MATR", "HUST"],
    )
    return args, config, original, override, episodes


def test_contract_keeps_frozen_source_and_total_domain_target_matrix(tmp_path):
    args, config, _, override, episodes = setup(tmp_path)
    loaded, _, _, _ = pooled.load_contract(args)
    assert loaded == config
    units = pooled.target_units(args, config, override, episodes)
    assert len(units) == 372
    assert sum(spec["K"] > 0 for spec in units) == 360
    assert sum(pooled.is_parent_reuse(spec) for spec in units) == 93
    matr = [spec for spec in units if spec["task"]["domain"] == "MATR"]
    assert len(matr) == 186
    assert all(len(spec["task"]["query_ids"]) == 8 for spec in matr)
    assert all(len(spec["support_ids"]) == spec["K"] for spec in matr)
    assert not any(pooled.is_parent_reuse(spec) for spec in matr)
    assert not any(
        "inner" == phase
        for phase in (
            "target-lock",
            "target-predict",
            "reuse-parent",
            "target-score",
            "collect",
        )
    )


def test_four_protocol_support_budget_cannot_enter_pooled_plan(tmp_path):
    args, config, _, override, episodes = setup(tmp_path)
    episodes = copy.deepcopy(episodes)
    task = episodes["target_tasks"][0]
    task["shots"][1]["draws"][0]["support_ids"] = task["support_pool_ids"][:4]
    with pytest.raises(RuntimeError, match="exactly K total-domain"):
        pooled.target_units(args, config, override, episodes)


@pytest.mark.parametrize(
    "method,expected_internal,expected_family",
    [
        ("PRETRAINED_LORA", "LoRA", "WARMUP"),
        ("SPECIALIZED_LORA", "LoRA", "SPECIALIZED"),
        ("SPECIALIZED_XS", "LoRA-XS", "SPECIALIZED"),
        ("SPECIALIZED_OUT", "OUT", "SPECIALIZED"),
        ("WARMUP_ZERO", None, "WARMUP"),
        ("SPECIALIZED_ZERO", None, "SPECIALIZED"),
    ],
)
def test_prepare_engine_preserves_reference_and_adapter_mapping(
    tmp_path, monkeypatch, method, expected_internal, expected_family
):
    args, config, _, _, _ = setup(tmp_path)
    factory_calls, registrations, svd_calls = [], [], []
    monkeypatch.setattr(pooled.source, "audit_common_reference", lambda *args: None)
    monkeypatch.setattr(
        pooled.source,
        "reference_identity",
        lambda *args: {"source_checkpoint_sha256": "source"},
    )
    monkeypatch.setattr(
        pooled,
        "load_pretrained_reference",
        lambda *args: factory_calls.append("WARMUP") or object(),
    )
    monkeypatch.setattr(
        pooled,
        "load_reference_model",
        lambda *args: factory_calls.append("SPECIALIZED") or object(),
    )
    monkeypatch.setattr(
        pooled, "NoPolicyInputs", lambda: SimpleNamespace(transform=lambda data: {})
    )
    monkeypatch.setattr(
        pooled,
        "SpecializationEngine",
        lambda *args, **kwargs: SimpleNamespace(
            register_method=lambda *values: registrations.append(values)
        ),
    )
    monkeypatch.setattr(
        pooled,
        "cache_xs_payload",
        lambda *args: (
            svd_calls.append(1) or {},
            {"SVD_seconds": 0.2, "new_SVD_evaluations": 12},
        ),
    )
    dictionary = (
        pooled.reference_folder(args.reference_root, "final", 42)
        / "geometry/DICTIONARIES.pt"
    )
    dictionary.parent.mkdir(parents=True)
    torch.save({"arms": {"OUT": {"dimension": 16, "layers": {}}}}, dictionary)
    lock = {"final_OUT_dictionary_sha256": {"42": file_hash(dictionary)}}
    _, _, internal = pooled.prepare_engine(
        args,
        config,
        SimpleNamespace(ids=["a"], query=np.arange(101, 5001, dtype=np.float32)),
        42,
        method,
        "MATR",
        lock,
    )
    assert factory_calls == [expected_family]
    assert internal == expected_internal
    assert [call[0] for call in registrations] == (
        [] if internal is None else [internal]
    )
    assert len(svd_calls) == (1 if method == "SPECIALIZED_XS" else 0)
    if method == "SPECIALIZED_OUT":
        assert registrations[0][1]["dimension"] == 16


def test_out_uses_existing_dictionary_and_geometric_support_solve(
    tmp_path, monkeypatch
):
    args, config, _, _, _ = setup(tmp_path)
    monkeypatch.setattr(pooled.source, "audit_common_reference", lambda *args: None)
    monkeypatch.setattr(pooled.source, "reference_identity", lambda *args: {})
    monkeypatch.setattr(pooled, "load_reference_model", lambda *args: object())
    monkeypatch.setattr(
        pooled, "NoPolicyInputs", lambda: SimpleNamespace(transform=lambda data: {})
    )
    monkeypatch.setattr(
        pooled, "SpecializationEngine", lambda *args, **kwargs: SimpleNamespace()
    )
    dictionary = (
        pooled.reference_folder(args.reference_root, "final", 42)
        / "geometry/DICTIONARIES.pt"
    )
    dictionary.parent.mkdir(parents=True)
    dictionary.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="frozen OUT dictionary changed"):
        pooled.prepare_engine(
            args,
            config,
            SimpleNamespace(ids=["a"], query=np.arange(101, 5001, dtype=np.float32)),
            42,
            "SPECIALIZED_OUT",
            "MATR",
            {"final_OUT_dictionary_sha256": {"42": "original"}},
        )
    calls = []
    engine = SimpleNamespace(
        geometry_trial=lambda *values: calls.append(values) or ({"a": np.zeros(16)}, {})
    )
    support, hyper = [object()], {"lambda_multiplier": 1, "eta": 0.25}
    state, details = pooled.adapted_state(engine, "OUT", support, hyper)
    assert calls == [("OUT", support, hyper)]
    assert state["a"].shape == (16,)
    assert details["selected_optimizer_steps"] == 0


def test_target_lock_reuses_only_original_source_choices(tmp_path, monkeypatch):
    args, config, original, override, _ = setup(tmp_path)
    args.source_public_root.mkdir()
    source_choice = args.source_public_root / "TARGET_LOCK.json"
    inner_file = tmp_path / "SOURCE_TRIALS.csv"
    inner_file.write_bytes(b"frozen_source_scores")
    choices = {
        "selected_hyperparameters": {"marker": "source"},
        "source_inner_files": [
            {"path": str(inner_file), "sha256": file_hash(inner_file)}
        ],
    }
    calls = []

    def source_lock(*values):
        calls.append(values)
        write_json(source_choice, choices)
        return choices

    monkeypatch.setattr(pooled.source, "target_lock", source_lock)
    monkeypatch.setattr(pooled.source, "read_target_lock", lambda *args: choices)
    monkeypatch.setattr(pooled, "header", lambda *args: {})
    monkeypatch.setattr(pooled.source, "relative", lambda path: str(path))
    for seed in config["training_seeds"]:
        path = (
            pooled.reference_folder(args.reference_root, "final", seed)
            / "geometry/DICTIONARIES.pt"
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(b"unchanged-original-dictionary")
    lock = pooled.target_lock(args, config, original, override)
    assert len(calls) == 1
    assert lock["selected_hyperparameters"] == {"marker": "source"}
    assert lock["source_choice_sha256"] == file_hash(source_choice)
    assert lock["new_source_inner_trials"] == 0
    assert lock["new_source_reference_fits"] == 0
    assert lock["old_MATR_protocol_specific_nonzero_predictions_reused"] is False
    assert pooled.target_lock(args, config, original, override) == lock
    assert len(calls) == 1
    inner_file.write_bytes(b"changed_source_scores")
    with pytest.raises(RuntimeError, match="source-inner choice input changed"):
        pooled.target_lock(args, config, original, override)


def test_matr_parent_reuse_is_rejected_before_any_artifact_or_label_read(tmp_path):
    args, config, original, override, episodes = setup(tmp_path)
    units = pooled.target_units(args, config, override, episodes)
    spec = next(
        spec
        for spec in units
        if spec["task"]["domain"] == "MATR" and spec["method"] == "SPECIALIZED_OUT"
    )
    with pytest.raises(ValueError, match="Only unchanged HUST"):
        pooled.reuse_parent_unit(args, config, original, override, spec, {})


def test_all_372_predictions_are_locked_before_target_query_scoring(
    tmp_path, monkeypatch
):
    args, config, original, override, episodes = setup(tmp_path)
    units = pooled.target_units(args, config, override, episodes)
    lock = {
        "selected_hyperparameters": {
            method: {str(k): {} for k in (1, 3, 5)} for method in config["methods"]
        }
    }
    monkeypatch.setattr(pooled, "read_target_lock", lambda *args: lock)
    monkeypatch.setattr(pooled, "unit_identity", lambda *args: {})
    monkeypatch.setattr(
        pooled.source, "load_prediction", lambda *args: np.zeros((1, 4900))
    )
    monkeypatch.setattr(pooled, "file_hash", lambda *args: "fixture")
    for spec in units[:-1]:
        write_json(
            spec["folder"] / "RECEIPT.json",
            {
                "status": "PREDICTIONS_LOCKED",
                "parent_reused": pooled.is_parent_reuse(spec),
                "parent_receipt_path": "fixture",
                "parent_receipt_sha256": "fixture",
                "cell_metrics_sha256": "fixture",
            },
        )
    reads = []
    monkeypatch.setattr(
        pooled, "load_dataset", lambda *args, **kwargs: reads.append(args)
    )
    with pytest.raises(FileNotFoundError):
        pooled.target_score(args, config, original, override, episodes)
    assert not reads
    assert not (args.public_root / "TARGET_QUERY_EXPOSURE.json").exists()
