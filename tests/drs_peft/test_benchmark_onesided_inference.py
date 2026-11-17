"""Timing structure, label-free input loading and saved-state replay checks."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts.drs_peft import benchmark_onesided_inference as benchmark
from scripts.drs_peft.data import file_hash, write_json


def test_timing_contains_one_warmup_and_three_complete_forwards(monkeypatch):
    calls = []
    clock = iter((0.0, 0.01, 1.0, 1.02, 2.0, 2.03))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))

    def predict():
        calls.append(1)
        return np.ones((3, 4900), dtype=np.float32)

    report = benchmark.timed_prediction(predict, "cpu", 3, 4900)
    assert len(calls) == 4
    assert report["warmup_repetitions"] == 1
    assert report["timed_repetitions"] == 3
    np.testing.assert_allclose(report["repetition_seconds"], (0.01, 0.02, 0.03))
    assert report["milliseconds_per_physical_cell_mean"] == pytest.approx(20 / 3)
    assert report["peak_gpu_allocated_bytes"] is None
    assert report["peak_gpu_reserved_bytes"] is None


def test_cuda_timing_synchronizes_and_reports_process_peaks(monkeypatch):
    synchronization, reset = [], []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: synchronization.append(str(device))
    )
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: reset.append(str(device))
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1234)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 5678)
    report = benchmark.timed_prediction(lambda: np.ones((1, 4900)), "cuda:0", 1, 4900)
    assert len(synchronization) == 7
    assert reset == ["cuda:0"]
    assert report["peak_gpu_allocated_bytes"] == 1234
    assert report["peak_gpu_reserved_bytes"] == 5678


def test_timing_rejects_nonfinite_or_incomplete_prediction():
    with pytest.raises(RuntimeError, match="incomplete or nonfinite"):
        benchmark.timed_prediction(lambda: np.full((1, 3), np.nan), "cpu", 1, 3)


def test_early_cache_loader_never_decodes_future_arrays(tmp_path):
    folder = tmp_path / "MATR"
    folder.mkdir()
    arrays = {
        "ids": np.array(["cell0"]),
        "raw_viq": np.ones((1, 2, 3, 3), dtype=np.float32),
        "point_mask": np.ones((1, 2, 3), dtype=bool),
        "cycle_ids": np.array([[1, 2]]),
        "drift_q": np.ones((1, 2, 3), dtype=np.float32),
        "drift_mask": np.ones((1, 2, 3), dtype=bool),
        "anchor": np.array([0.97], dtype=np.float32),
        "nominal_capacity_Ah": np.array([2.0], dtype=np.float32),
        "query": np.arange(101, 5001, dtype=np.float32),
        "targets": np.array([object()], dtype=object),
        "observed": np.array([object()], dtype=object),
    }
    np.savez(folder / "arrays.npz", **arrays)
    write_json(
        folder / "COMPLETE.json",
        {"files": [{"path": "arrays.npz", "sha256": file_hash(folder / "arrays.npz")}]},
    )
    data = benchmark.load_early_only("MATR", tmp_path)
    assert not hasattr(data, "targets")
    assert not hasattr(data, "observed")
    np.testing.assert_array_equal(data.x[..., 0], 1)
    np.testing.assert_array_equal(data.x[..., 1:], 0.5)
    assert data.x.shape == (1, 2, 3, 4)


def test_representatives_follow_fixed_source_scenario_and_locked_target_generic(
    tmp_path,
):
    write_json(tmp_path / "TARGET_LOCK.json", {"best_generic": "LoRA_FA"})
    plans = benchmark.representative_units(
        SimpleNamespace(config=benchmark.CONFIG, public_root=tmp_path)
    )
    assert len(plans) == 12
    assert {plan["seed"] for plan in plans} == {42}
    assert {plan["draw"] for plan in plans} == {0}
    source = [plan for plan in plans if plan["stage"] == "D2"]
    target = [plan for plan in plans if plan["stage"] == "D4"]
    assert len(source) == 8
    assert {plan["task"]["task_id"] for plan in source} == {"SOURCE_OUTER_FOLD0"}
    assert {plan["method"] for plan in target} == {"ZERO", "CAL2", "OUT", "LoRA_FA"}
    assert {plan["task"]["task_id"] for plan in target} == {"C_HUST"}
    assert all(plan["K"] == (0 if plan["method"] == "ZERO" else 3) for plan in plans)


def saved_fixture(tmp_path, method="ZERO"):
    query = np.arange(101, 5001, dtype=np.float32)
    task = {"task_id": "TEST_SOURCE", "domain": "MATR", "query_ids": ["cell0", "cell1"]}
    plan = {
        "stage": "D2",
        "task": task,
        "seed": 42,
        "method": method,
        "K": 0 if method == "ZERO" else 3,
        "draw": 0,
        "receipt_path": tmp_path / "RECEIPT.json",
    }
    prediction = np.full((2, len(query)), 0.95, dtype=np.float32)
    identity = {
        "stage": "D2",
        "task_id": task["task_id"],
        "domain": "MATR",
        "query_ids": task["query_ids"],
        "seed": 42,
        "method": method,
        "K": plan["K"],
        "source_checkpoint_sha256": "source",
    }
    state = None if method == "ZERO" else {"kind": "coordinates", "a": np.zeros(2)}
    torch.save({"identity": identity, "state": state}, tmp_path / "STATE.pt")
    np.savez(
        tmp_path / "PREDICTIONS.npz",
        ids=np.array(task["query_ids"]),
        query=query,
        prediction=prediction,
    )
    receipt = {
        **identity,
        "draw": 0,
        "status": "SCORED",
        "state_path": str(tmp_path / "STATE.pt"),
        "prediction_path": str(tmp_path / "PREDICTIONS.npz"),
        "state_sha256": file_hash(tmp_path / "STATE.pt"),
        "prediction_sha256": file_hash(tmp_path / "PREDICTIONS.npz"),
    }
    write_json(plan["receipt_path"], receipt)
    return plan, query, prediction


def test_saved_unit_checks_hashes_and_fixed_prediction_axes(tmp_path):
    plan, query, expected = saved_fixture(tmp_path)
    receipt, state, prediction = benchmark.load_saved_unit(plan, query)
    assert receipt["method"] == "ZERO" and state is None
    np.testing.assert_array_equal(prediction, expected)
    with pytest.raises(RuntimeError, match="fixed Query axis"):
        benchmark.load_saved_unit(plan, query[:-1])
    (tmp_path / "STATE.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="state or prediction changed"):
        benchmark.load_saved_unit(plan, query)


def test_replay_reports_exact_and_rounded_agreement_and_rejects_drift():
    expected = np.ones((1, 10), dtype=np.float32)
    assert benchmark.assert_replay(expected, expected)["replay_bit_exact"] is True
    near = expected.astype(np.float64) + 1e-7
    assert benchmark.assert_replay(near, expected)["replay_bit_exact"] is False
    with pytest.raises(RuntimeError, match="FP32 tolerance"):
        benchmark.assert_replay(expected + 0.01, expected)


class TinyModel(nn.Module):
    def __init__(self, merged=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.register_buffer("dictionary", torch.empty(0) if merged else torch.ones(4))
        self.merged = merged

    def parameter_report(self):
        return {
            "adaptation_parameters": 2,
            "frozen_dictionary_bytes": 0 if self.merged else 16,
        }

    def merged_copy(self):
        return TinyModel(merged=True)


def test_benchmark_measures_appended_and_merged_states_with_full_replay(tmp_path):
    plan, query, expected = saved_fixture(tmp_path, "OUT")
    model = TinyModel()
    predictions = []

    def predict(current, ids):
        assert ids == plan["task"]["query_ids"]
        predictions.append(current.merged)
        return expected.copy()

    engine = SimpleNamespace(
        device="cpu",
        source_sha256="source",
        active_model=model,
        active_method="OUT",
        _install_state=lambda method, state: model,
        _predict=predict,
    )
    initial = copy.deepcopy(model.state_dict())
    rows = benchmark.benchmark_unit(engine, plan, query)
    assert len(rows) == 2
    assert [row["representation"] for row in rows] == ["appended", "merged"]
    assert rows[0]["actual_batch_sizes"] == [2]
    assert rows[0]["frozen_dictionary_bytes"] == 16
    assert rows[1]["frozen_dictionary_bytes"] == 0
    assert rows[1]["merge_to_appended_maximum_absolute_difference"] == 0
    assert predictions == [False] * 5 + [True] * 5
    assert engine.active_model is None
    for key, value in initial.items():
        torch.testing.assert_close(model.state_dict()[key], value)
