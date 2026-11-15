"""Inspect original cycle fields and waveform/summary alignment without relabeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def bounds(values):
    valid = np.asarray(values).reshape(-1)
    valid = valid[np.isfinite(valid)]
    return [float(valid.min()), float(valid.max())] if len(valid) else None


def inspect(path, batch_number):
    cells = []
    with h5py.File(path, "r") as handle:
        batch = handle["batch"]
        for index, reference in enumerate(batch["cycles"][()].reshape(-1)):
            waveforms = handle[reference]
            summary = handle[batch["summary"][()].reshape(-1)[index]]
            summary_values = {name: np.asarray(value[()]).reshape(-1)
                              for name, value in summary.items()}
            cycle_ids = summary_values["cycle"]
            summary_qd = summary_values["QDischarge"]
            waveform_counts = {name: int(value.size) for name, value in waveforms.items()}
            count = waveform_counts["I"]
            samples = []
            for record in sorted({0, 1, 2, 3, 4, count - 1}):
                if record < 0 or record >= count:
                    continue
                arrays = {name: np.asarray(handle[waveforms[name][()].reshape(-1)[record]][()]).reshape(-1)
                          for name in ("V", "I", "Qd", "Qc", "t")}
                lengths = {name: len(values) for name, values in arrays.items()}
                equal = len(set(lengths.values())) == 1
                discharge = arrays["I"] < -0.05
                active_qd = arrays["Qd"][discharge] if equal else np.asarray([])
                device_max = float(np.nanmax(active_qd)) if len(active_qd) else None
                differences = {}
                for shift in (-1, 0, 1):
                    position = record + shift
                    if device_max is not None and 0 <= position < len(summary_qd):
                        differences[str(shift)] = float(device_max - summary_qd[position])
                samples.append({"waveform_record_index": record, "VIQ_t_lengths": lengths,
                                "lengths_equal": equal,
                                "corresponding_summary_cycle_sequence": float(cycle_ids[record]) if record < len(cycle_ids) else None,
                                "V_range_V": bounds(arrays["V"]), "I_range_A": bounds(arrays["I"]),
                                "Qd_range_Ah": bounds(arrays["Qd"]), "t_range_original_units": bounds(arrays["t"]),
                                "active_discharge_points": int(discharge.sum()),
                                "active_discharge_Qd_max_Ah": device_max,
                                "device_Qd_max_minus_summary_Qd_at_index_plus_shift": differences})
            policy_ref = batch["policy_readable"][()].reshape(-1)[index]
            policy = "".join(chr(int(code)) for code in handle[policy_ref][()].reshape(-1))
            cells.append({"source_cell_id": f"b{batch_number}c{index}",
                          "summary_fields": list(summary_values),
                          "summary_field_lengths": {name: len(value) for name, value in summary_values.items()},
                          "summary_cycle_first10": cycle_ids[:10].tolist(),
                          "summary_cycle_last5": cycle_ids[-5:].tolist(),
                          "summary_cycle_finite_integer": bool(np.isfinite(cycle_ids).all() and np.equal(cycle_ids, np.round(cycle_ids)).all()),
                          "summary_cycle_strictly_increasing": bool((np.diff(cycle_ids) > 0).all()),
                          "summary_cycle_gap_count": int((np.diff(cycle_ids) != 1).sum()),
                          "waveform_field_record_counts": waveform_counts,
                          "summary_cycle_Qd_waveform_count_equal": len(cycle_ids) == len(summary_qd) == count,
                          "policy_original": policy, "waveform_samples": samples})
    return {"file_name": path.name, "bytes": path.stat().st_size, "batch": batch_number,
            "raw_cells": len(cells), "cells": cells,
            "scope": "Author-regenerated summary.cycle and sampled released waveform alignment; instrument Cycle_Index is unavailable"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.source, args.batch)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"file_name": args.source.name, "raw_cells": result["raw_cells"],
                      "all_summary_waveform_counts_equal": all(row["summary_cycle_Qd_waveform_count_equal"] for row in result["cells"]),
                      "all_summary_cycle_axes_monotonic": all(row["summary_cycle_strictly_increasing"] for row in result["cells"]),
                      "first_cell": result["cells"][0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
