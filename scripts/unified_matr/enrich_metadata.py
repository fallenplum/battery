"""Attach complete supplied metadata and author-label comparisons before freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dataset_utils import sha256, write_json


REPOSITORY = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    if (root / "DATA_FREEZE.json").exists():
        raise RuntimeError("Metadata enrichment must precede the immutable data freeze")
    metadata_path = REPOSITORY / "data/cell_metadata.csv"
    static_path = REPOSITORY / "results/unified_matr/source_static_field_audit.json"
    instrument_path = REPOSITORY / "results/unified_matr/instrument_metadata_audit.json"
    schedule_path = REPOSITORY / "results/unified_matr/schedule_metadata_audit.json"
    original = pd.read_csv(metadata_path)
    supplied = {json.loads(row.metadata)["cell_id"]: row.metadata for row in original.itertuples() if row.family == "MATR"}
    static = json.loads(static_path.read_text(encoding="utf-8"))
    static_by_segment = {row["source_segment_id"]: row for row in static["cells"]}
    instrument = json.loads(instrument_path.read_text(encoding="utf-8"))
    instrument_by_segment = {row["source_segment_id"]: row for row in instrument["source_rows"]}
    schedules = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_by_segment = {row["source_segment_id"]: row for row in schedules["source_links"]}
    cells = pd.read_parquet(root / "cells.parquet")
    if set(supplied) != set(cells.cell_uid) or len(cells) != 169:
        raise RuntimeError("Complete metadata must cover the same fixed169 identities")
    comparisons = []
    source_metadata = []
    instrument_metadata = []
    schedule_metadata = []
    primary_devices = []
    for cell in cells.itertuples():
        segments = json.loads(cell.source_segments)
        records = [static_by_segment[segment] for segment in segments]
        source_metadata.append(json.dumps(records, ensure_ascii=False, sort_keys=True))
        hardware = [instrument_by_segment[segment] for segment in segments]
        instrument_metadata.append(json.dumps(hardware, ensure_ascii=False, sort_keys=True))
        schedule_metadata.append(json.dumps([schedule_by_segment[segment] for segment in segments], ensure_ascii=False, sort_keys=True))
        primary_devices.append(str(hardware[0]["native_metadata_device_id"]))
        for record in records:
            comparisons.append({"cell_uid": cell.cell_uid, "source_segment_id": record["source_segment_id"],
                                "qa_only_official_release_cycle_life": record["qa_only_official_release_cycle_life"],
                                "official_release_label_coordinate": record["official_release_label_coordinate"],
                                "authenticated_eol80_event_observed": cell.eol80_event_observed,
                                "authenticated_eol80_first_observed_crossing_cycle": cell.eol80_first_observed_crossing_cycle,
                                "authenticated_last_valid_cycle": cell.last_observed_valid_cycle,
                                "official_label_used_as_model_target": False})
    cells["supplied_metadata_json"] = cells.cell_uid.map(supplied)
    cells["original_release_static_metadata_json"] = source_metadata
    cells["instrument_metadata_json"] = instrument_metadata
    cells["source_schedule_metadata_json"] = schedule_metadata
    cells["source_instrument_device_id"] = primary_devices
    cells["eol80_right_censored"] = cells.right_censored
    cells["trajectory_observation_end_cycle"] = cells.last_observed_valid_cycle
    cells["official_release_labels_role"] = "QA comparison only; source-specific author coordinates remain explicit"
    cells.to_parquet(root / "cells.parquet", index=False)
    pd.DataFrame(comparisons).to_parquet(root / "official_release_label_comparison.parquet", index=False)
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata_enrichment"] = {
        "code_sha256": sha256(Path(__file__)), "supplied_metadata_snapshot_sha256": sha256(metadata_path),
        "original_release_static_audit_sha256": sha256(static_path), "cells": len(cells),
        "instrument_metadata_audit_sha256": sha256(instrument_path),
        "schedule_metadata_audit_sha256": sha256(schedule_path),
        "official_label_comparison_rows": len(comparisons),
        "model_input_effect": "None. The registered model input keys remain the frozen prefix arrays and known design policy.",
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "METADATA_ENRICHMENT_COMPLETE", "cells": len(cells), "source_segment_label_comparisons": len(comparisons)}))


if __name__ == "__main__":
    main()
