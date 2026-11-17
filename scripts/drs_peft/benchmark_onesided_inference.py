"""Replay saved few-shot states and benchmark representative fixed-grid inference."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.drs_peft.data import ROOT, BatteryData, file_hash, write_json
from scripts.drs_peft.run_onesided import (
    CONFIG,
    GENERIC,
    GEOMETRY,
    PRIVATE,
    PUBLIC,
    REFERENCES,
    prepare_engine,
    register_methods,
    unit_folder,
)
from scripts.drs_peft.run_onesided_reference import reference_folder
from scripts.drs_peft.train import configure, read_json


@dataclass
class EarlyOnlyData:
    ids: list[str]
    raw_viq: np.ndarray
    point_mask: np.ndarray
    cycle_ids: np.ndarray
    drift_q: np.ndarray
    drift_mask: np.ndarray
    anchor: np.ndarray
    nominal_capacity_Ah: np.ndarray
    query: np.ndarray

    x = BatteryData.x
    feature_mask = BatteryData.feature_mask


def load_early_only(domain, data_root):
    """Decode only the established prefix fields from the frozen NPZ cache."""
    folder = Path(data_root) / domain
    manifest = read_json(folder / "COMPLETE.json")
    for row in manifest["files"]:
        if file_hash(folder / row["path"]) != row["sha256"]:
            raise RuntimeError("The frozen data cache differs from its manifest")
    names = (
        "raw_viq",
        "point_mask",
        "cycle_ids",
        "drift_q",
        "drift_mask",
        "anchor",
        "nominal_capacity_Ah",
        "query",
    )
    with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in names}
        ids = archive["ids"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Early-input physical identities must be unique")
    if not np.array_equal(values["query"], np.arange(101, 5001, dtype=np.float32)):
        raise ValueError("The registered prediction grid must contain 4900 points")
    return EarlyOnlyData(ids=ids, **values)


def representative_units(args):
    config = read_json(args.config)
    registration = read_json(ROOT / config["episodes_file"])
    source = [
        task
        for task in registration["source_tasks"]["outer"]
        if task["task_id"] == "SOURCE_OUTER_FOLD0"
    ]
    target = [task for task in registration["target_tasks"] if task["domain"] == "HUST"]
    lock = read_json(args.public_root / "TARGET_LOCK.json")
    if len(source) != 1 or len(target) != 1 or lock["best_generic"] not in GENERIC:
        raise ValueError("Expected one source task, one HUST task and a locked generic")
    selections = (
        ("D2", source[0], ("ZERO", "CAL2", *GEOMETRY, *GENERIC)),
        ("D4", target[0], ("ZERO", "CAL2", "OUT", lock["best_generic"])),
    )
    result = []
    for stage, task, methods in selections:
        for method in methods:
            k = 0 if method == "ZERO" else 3
            result.append(
                {
                    "stage": stage,
                    "task": task,
                    "seed": 42,
                    "method": method,
                    "K": k,
                    "draw": 0,
                    "receipt_path": unit_folder(args, stage, task, 42, method, k, 0)
                    / "RECEIPT.json",
                }
            )
    return result


def load_saved_unit(specification, query):
    receipt = read_json(specification["receipt_path"])
    expected = {
        "stage": specification["stage"],
        "task_id": specification["task"]["task_id"],
        "domain": specification["task"]["domain"],
        "seed": specification["seed"],
        "method": specification["method"],
        "K": specification["K"],
        "draw": specification["draw"],
        "query_ids": specification["task"]["query_ids"],
    }
    if receipt["status"] != "SCORED" or any(
        receipt[key] != value for key, value in expected.items()
    ):
        raise RuntimeError("A representative experiment is incomplete or changed")
    state_path = ROOT / receipt["state_path"]
    prediction_path = ROOT / receipt["prediction_path"]
    if (
        file_hash(state_path) != receipt["state_sha256"]
        or file_hash(prediction_path) != receipt["prediction_sha256"]
    ):
        raise RuntimeError("A saved adaptation state or prediction changed")
    saved = torch.load(state_path, map_location="cpu", weights_only=False)
    identity = saved["identity"]
    for key in expected.keys() - {"draw"}:
        if identity[key] != expected[key]:
            raise RuntimeError("The saved state belongs to another experiment")
    if identity["source_checkpoint_sha256"] != receipt["source_checkpoint_sha256"]:
        raise RuntimeError("The saved state source differs from its receipt")
    with np.load(prediction_path, allow_pickle=False) as archive:
        if archive["ids"].astype(str).tolist() != receipt[
            "query_ids"
        ] or not np.array_equal(archive["query"], query):
            raise RuntimeError("Saved predictions use another fixed Query axis")
        prediction = archive["prediction"].copy()
    if (
        prediction.shape != (len(receipt["query_ids"]), len(query))
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Saved predictions must be finite on the complete grid")
    if (saved["state"] is None) != (receipt["method"] == "ZERO"):
        raise RuntimeError("ZERO and adapted states have incompatible representations")
    return receipt, saved["state"], prediction


def assert_replay(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise RuntimeError("Inference did not reproduce the full finite Query grid")
    difference = float(np.abs(actual.astype(float) - expected.astype(float)).max())
    if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise RuntimeError("Inference replay exceeds the registered FP32 tolerance")
    return {
        "replay_maximum_absolute_difference": difference,
        "replay_bit_exact": bool(np.array_equal(actual, expected)),
        "replay_absolute_tolerance": 1e-6,
        "replay_relative_tolerance": 1e-6,
    }


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def timed_prediction(predict, device, physical_cells, grid_points):
    """One warmup and three synchronized, complete fixed-grid forwards."""
    if physical_cells < 1 or grid_points < 1:
        raise ValueError("A timing requires physical cells and a complete grid")
    predict()
    synchronize(device)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    elapsed = []
    for _ in range(3):
        synchronize(device)
        started = time.perf_counter()
        output = predict()
        synchronize(device)
        elapsed.append(time.perf_counter() - started)
        if (
            output.shape != (physical_cells, grid_points)
            or not np.isfinite(output).all()
        ):
            raise RuntimeError("A timed prediction is incomplete or nonfinite")
    mean_seconds = float(np.mean(elapsed))
    return {
        "warmup_repetitions": 1,
        "timed_repetitions": 3,
        "repetition_seconds": elapsed,
        "seconds_full_query_mean": mean_seconds,
        "seconds_full_query_minimum": min(elapsed),
        "seconds_full_query_maximum": max(elapsed),
        "milliseconds_per_physical_cell_mean": 1000 * mean_seconds / physical_cells,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if cuda
        else None,
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if cuda
        else None,
    }


def hardware_environment(device):
    cuda = torch.device(device).type == "cuda"
    result = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if cuda:
        properties = torch.cuda.get_device_properties(device)
        result.update(
            gpu_name=properties.name,
            gpu_total_memory_bytes=properties.total_memory,
            gpu_compute_capability=[properties.major, properties.minor],
        )
        try:
            driver = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,pci.bus_id,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            result["nvidia_smi_gpu_driver_bus_memory_mib"] = (
                driver.stdout.strip().splitlines()
            )
        except (OSError, subprocess.SubprocessError) as error:
            result["nvidia_smi_error"] = type(error).__name__
    return result


def benchmark_unit(engine, specification, query):
    receipt, state, expected = load_saved_unit(specification, query)
    method = receipt["method"]
    if receipt["source_checkpoint_sha256"] != engine.source_sha256:
        raise RuntimeError("The replay engine source differs from the selected state")
    ids = receipt["query_ids"]
    model = (
        engine._active("ZERO")
        if method == "ZERO"
        else engine._install_state(method, state)
    )
    prediction = engine._predict(model, ids)
    replay = assert_replay(prediction, expected)
    parameter_report = model.parameter_report()
    provenance = {
        key: receipt[key]
        for key in ("stage", "task_id", "domain", "seed", "method", "K", "draw")
    }
    provenance.update(
        query_physical_cells=len(ids),
        prediction_grid_points=len(query),
        prediction_grid_minimum=float(query[0]),
        prediction_grid_maximum=float(query[-1]),
        configured_batch_size=8,
        actual_batch_sizes=[
            min(8, len(ids) - start) for start in range(0, len(ids), 8)
        ],
        source_checkpoint_sha256=receipt["source_checkpoint_sha256"],
        dictionary_sha256=receipt.get("dictionary_sha256"),
        saved_state_sha256=receipt["state_sha256"],
        saved_prediction_sha256=receipt["prediction_sha256"],
        adaptation_parameters=parameter_report.get("adaptation_parameters", 0),
        frozen_dictionary_bytes=parameter_report.get("frozen_dictionary_bytes", 0),
        frozen_lora_A_bytes=parameter_report.get("frozen_A_bytes", 0),
        model_parameter_bytes=sum(
            p.numel() * p.element_size() for p in model.parameters()
        ),
        model_buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()),
        parameter_report=parameter_report,
    )
    rows = [
        {
            **provenance,
            "representation": "appended" if method != "ZERO" else "shared_reference",
            **replay,
            **timed_prediction(
                lambda: engine._predict(model, ids), engine.device, len(ids), len(query)
            ),
        }
    ]
    if not hasattr(model, "merged_copy"):
        rows[0]["merged_benchmark_status"] = (
            "NA_CAL2" if method == "CAL2" else "NA_shared_reference"
        )
        del model
    else:
        merged = model.merged_copy().eval()
        merged_prediction = engine._predict(merged, ids)
        merge_replay = assert_replay(merged_prediction, expected)
        merge_to_appended = assert_replay(merged_prediction, prediction)
        engine.active_model, engine.active_method = None, None
        del model
        gc.collect()
        if torch.device(engine.device).type == "cuda":
            torch.cuda.empty_cache()
        rows[0]["merged_benchmark_status"] = "MEASURED"
        rows.append(
            {
                **provenance,
                "representation": "merged",
                "frozen_dictionary_bytes": 0,
                "frozen_lora_A_bytes": 0,
                "model_parameter_bytes": sum(
                    p.numel() * p.element_size() for p in merged.parameters()
                ),
                "model_buffer_bytes": sum(
                    b.numel() * b.element_size() for b in merged.buffers()
                ),
                "parameter_report": merged.parameter_report(),
                "merged_benchmark_status": "MEASURED",
                "merge_to_appended_maximum_absolute_difference": merge_to_appended[
                    "replay_maximum_absolute_difference"
                ],
                **merge_replay,
                **timed_prediction(
                    lambda: engine._predict(merged, ids),
                    engine.device,
                    len(ids),
                    len(query),
                ),
            }
        )
        del merged
    engine.active_model, engine.active_method = None, None
    gc.collect()
    if torch.device(engine.device).type == "cuda":
        torch.cuda.empty_cache()
    return rows


def execute(args):
    configure()
    plans = representative_units(args)
    records = []
    for stage in ("D2", "D4"):
        selected = [plan for plan in plans if plan["stage"] == stage]
        task = selected[0]["task"]
        data = load_early_only(task["domain"], args.data_root)
        fold = task.get("fold_id", "final")
        scope = "D2" if stage == "D2" else "D4_" + task["domain"]
        engine, _, _ = prepare_engine(args, data, fold, 42, scope)
        register_methods(engine, reference_folder(args.reference_root, fold, 42))
        for plan in selected:
            rows = benchmark_unit(engine, plan, data.query)
            for row in rows:
                row["early_data_manifest_sha256"] = file_hash(
                    args.data_root / task["domain"] / "COMPLETE.json"
                )
            records.extend(rows)
            print(
                json.dumps(
                    {
                        "event": "INFERENCE_BENCHMARK_COMPLETE",
                        "stage": stage,
                        "method": plan["method"],
                    }
                ),
                flush=True,
            )
        del engine, data
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "status": "COMPLETE_REPRESENTATIVE_INFERENCE_BENCHMARK",
        "scope": "SOURCE_OUTER_FOLD0_seed42_K3_draw0_all8_and_C_HUST_seed42_K3_draw0_locked4_ZERO_K0",
        "interpretation": "Three timing repeats characterize these representative saved states and hardware, without statistical inference or dataset-wide speed claims.",
        "timing_boundary": "Complete fixed-grid prediction using batch8, input batch transfer, model forward and CPU NumPy output transfer; model loading, state loading, adaptation and merging excluded.",
        "peak_memory_scope": "This process's allocated/reserved peaks after one warmup, including resident inference model and transient prediction storage.",
        "future_label_arrays_decoded": 0,
        "future_label_arrays_used": 0,
        "optimizer_updates": 0,
        "gradient_evaluations": 0,
        "hyperparameter_selection": False,
        "saved_experiment_state_modified": False,
        "hardware_environment": hardware_environment(args.device),
        "benchmark_script_sha256": file_hash(__file__),
        "target_lock_sha256": file_hash(args.public_root / "TARGET_LOCK.json"),
        "records": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "INFERENCE_BENCHMARK.csv"
    json_path = args.output_root / "INFERENCE_BENCHMARK.json"
    flat = []
    for record in records:
        row = dict(record)
        row.pop("parameter_report")
        for key in ("actual_batch_sizes", "repetition_seconds"):
            row[key] = json.dumps(row[key])
        flat.append(row)
    pending_csv = csv_path.with_suffix(".pending.csv")
    pd.DataFrame(flat).to_csv(pending_csv, index=False)
    pending_csv.replace(csv_path)
    output["csv_sha256"] = file_hash(csv_path)
    write_json(json_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--private-root", type=Path, default=PRIVATE)
    parser.add_argument("--public-root", type=Path, default=PUBLIC)
    parser.add_argument("--reference-root", type=Path, default=REFERENCES)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument("--output-root", type=Path, default=PUBLIC)
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
