"""Independently compare a D2 CPU replay with its preserved GPU measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import file_hash, write_json
from .train import ROOT, read_json

FILES = (
    "D2_UPDATE_GEOMETRY.csv",
    "D2_UPDATE_SPECTRA.csv",
    "D2_SOURCE_WINDOW_ERRORS.csv",
    "D2_PREVIOUS_COUNTEREXAMPLES.csv",
    "D2_JACOBIAN.csv",
    "D2_LAYER_ACTIVATIONS.csv",
    "D2_CORE_PREDICTION_EFFECT.csv",
)


def verify(primary, replay, original_source):
    records = []
    for name in FILES:
        reference = pd.read_csv(primary / name)
        actual = pd.read_csv(replay / name)
        if reference.shape != actual.shape or list(reference.columns) != list(
            actual.columns
        ):
            raise RuntimeError("D2 replay schema differs at " + name)
        column_records = []
        for column in reference.columns:
            left, right = reference[column], actual[column]
            if pd.api.types.is_numeric_dtype(left):
                left_array = left.to_numpy(dtype=float)
                right_array = right.to_numpy(dtype=float)
                finite = np.isfinite(left_array)
                if not np.array_equal(finite, np.isfinite(right_array)):
                    raise RuntimeError(
                        "D2 replay finite support differs at " + name + "/" + column
                    )
                difference = np.abs(left_array[finite] - right_array[finite])
                if not np.allclose(
                    left_array, right_array, atol=1e-4, rtol=1e-3, equal_nan=True
                ):
                    raise RuntimeError(
                        "D2 cross-device tolerance failed at " + name + "/" + column
                    )
                column_records.append(
                    {
                        "column": column,
                        "maximum_absolute_difference": float(difference.max())
                        if len(difference)
                        else 0.0,
                    }
                )
            elif not left.fillna("<NA>").equals(right.fillna("<NA>")):
                raise RuntimeError(
                    "D2 replay row identity differs at " + name + "/" + column
                )
        records.append(
            {
                "file": name,
                "rows": len(reference),
                "columns": len(reference.columns),
                "primary_sha256": file_hash(primary / name),
                "replay_sha256": file_hash(replay / name),
                "byte_identical": file_hash(primary / name) == file_hash(replay / name),
                "numeric_column_checks": column_records,
            }
        )
    gpu = read_json(primary / "D2_SENSITIVITY_COMPLETE.json")
    cpu = read_json(replay / "D2_SENSITIVITY_COMPLETE.json")
    if gpu["source_sha256"] != file_hash(original_source):
        raise RuntimeError("Original GPU executed source snapshot SHA differs")
    return {
        "status": "PASSED",
        "primary_device": gpu["device"],
        "replay_device": cpu["device"],
        "original_gpu_source_sha256": gpu["source_sha256"],
        "cpu_replay_source_sha256": cpu["source_sha256"],
        "current_source_sha256": file_hash(
            ROOT / "scripts/drs_peft/diagnostic_structure.py"
        ),
        "source_revision_scope": "import and dict formatting, explicit synchronous hook binding, additional coverage and summary outputs",
        "cross_device_absolute_tolerance": 1e-4,
        "cross_device_relative_tolerance": 1e-3,
        "primary_scalar_query_backward_evaluations": gpu["scalar_backward_evaluations"],
        "replay_scalar_query_backward_evaluations": cpu["scalar_backward_evaluations"],
        "total_scalar_query_backward_evaluations": gpu["scalar_backward_evaluations"]
        + cpu["scalar_backward_evaluations"],
        "optimizer_updates": 0,
        "final_target_query_labels_used": 0,
        "files": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT / "results/drs_peft/diagnostics_v1/D2_structure",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=ROOT / "runtime/drs_peft/diagnostics_v1/D2_CPU_REPLAY",
    )
    parser.add_argument(
        "--original-source",
        type=Path,
        default=ROOT
        / "runtime/drs_peft/diagnostics_v1/D2_original_gpu_execution/diagnostic_structure.py",
    )
    args = parser.parse_args()
    result = verify(args.primary, args.replay, args.original_source)
    write_json(args.primary / "D2_INDEPENDENT_REPLAY.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
