"""Audit all released MATR static fields, retaining official labels as QA data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from raw_reader import readable_policy, static_source_field
from recover_sources import SOURCES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archives, cells = [], []
    for number, filename, _, _ in SOURCES:
        with h5py.File(args.source_root / filename, "r") as handle:
            batch = handle["batch"]
            archives.append({"source_batch": number, "root_fields": list(handle),
                             "batch_fields": list(batch), "static_field_shapes": {name: list(value.shape) for name, value in batch.items()}})
            for index in range(batch["cycles"].size):
                barcode = str(static_source_field(handle, batch, "barcode", index))
                life = np.asarray(handle[batch["cycle_life"][index, 0]][()]).reshape(-1)
                grid = np.asarray(handle[batch["Vdlin"][index, 0]][()]).reshape(-1)
                row = {"source_segment_id": f"b{number}c{index}", "source_batch": number,
                       "source_cell_index": index, "source_barcode_sha256": hashlib.sha256(barcode.encode()).hexdigest(),
                       "source_channel": str(static_source_field(handle, batch, "channel_id", index)),
                       "policy_source_text": static_source_field(handle, batch, "policy", index),
                       "policy_readable_source_text": readable_policy(handle, batch, index),
                       "qa_only_official_release_cycle_life": float(life[0]) if len(life) == 1 and np.isfinite(life[0]) else None,
                       "official_release_label_finite_scalar": bool(len(life) == 1 and np.isfinite(life[0])),
                       "official_release_label_coordinate": "Author release label, source parser uses generated summary positions and special-case instrument j; separate from authenticated native trajectory outcome",
                       "derived_Vdlin_points": len(grid), "derived_Vdlin_first_V": float(grid[0]) if len(grid) else None,
                       "derived_Vdlin_last_V": float(grid[-1]) if len(grid) else None,
                       "derived_Vdlin_all_finite": bool(np.isfinite(grid).all()),
                       "derived_Vdlin_monotonic": bool((np.diff(grid) > 0).all() or (np.diff(grid) < 0).all()),
                       "derived_Vdlin_raw_numeric_sha256": hashlib.sha256(grid.tobytes()).hexdigest()}
                cells.append(row)
    checks = {"all185_static_records_audited": len(cells) == 185,
              "all_policy_source_fields_present": all(row["policy_source_text"] is not None and row["policy_readable_source_text"] for row in cells),
              "all185_derived_grid_finite_monotonic": all(row["derived_Vdlin_all_finite"] and row["derived_Vdlin_monotonic"] for row in cells)}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "archives": archives, "cells": cells,
              "official_release_label_finite_cells": sum(row["official_release_label_finite_scalar"] for row in cells),
              "derived_grid_point_counts": dict(Counter(row["derived_Vdlin_points"] for row in cells)),
              "scope": "All released static metadata and separate official-label QA. Source VIQ is reconstructed from original CSV under the unified contract.",
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "checks", "official_release_label_finite_cells", "derived_grid_point_counts")}, ensure_ascii=False))
    if not all(checks.values()):
        raise AssertionError("Released source static-field audit failed")


if __name__ == "__main__":
    main()
