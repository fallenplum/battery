"""Independent fixed-cohort identity join against the full EDP inventory."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


CONTINUATIONS = {"b2c7": "b1c0", "b2c8": "b1c1", "b2c9": "b1c2", "b2c15": "b1c3", "b2c16": "b1c4"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bindings_bytes = args.bindings.read_bytes()
    registry_bytes = args.source_registry.read_bytes()
    inventory_bytes = args.inventory.read_bytes()
    bindings = json.loads(bindings_bytes)
    registry = json.loads(registry_bytes)
    inventory = json.loads(inventory_bytes)
    bindings_sha = hashlib.sha256(bindings_bytes).hexdigest()
    registry_sha = hashlib.sha256(registry_bytes).hexdigest()
    inventory_sha = hashlib.sha256(inventory_bytes).hexdigest()
    checks = {}
    identities = collections.defaultdict(list)
    for item in inventory["files"]:
        barcode = hashlib.sha256(item["source_cell_barcode"].strip().upper().encode()).hexdigest()
        identities[(item["source_batch"], barcode, str(item["source_channel"]))].append(item)
    cells = {cell["cell_uid"]: cell for cell in bindings["cells"]}
    source_cells = {cell["cell_uid"]: cell for cell in registry["cells"]}
    checks["fixed169_UID_set"] = len(cells) == 169 and set(cells) == set(source_cells)
    segments = {segment["source_segment_id"]: segment for cell in bindings["cells"] for segment in cell["source_segments"]}
    originals = {segment["source_segment_id"]: segment for cell in registry["cells"] for segment in cell["source_segments"]}
    checks["all174_segment_names_unique"] = len(segments) == 174 and set(segments) == set(originals)
    details = []
    for name, segment in segments.items():
        original = originals[name]
        key = (original["source_batch"], original["source_barcode_normalized_sha256"], str(original["source_channel_id"]))
        candidates = identities[key]
        unique = len(candidates) == 1
        official = candidates[0] if unique else {}
        equal = unique and all(segment[field] == official[field] for field in ("test_id", "file_id", "source_channel"))
        equal = equal and segment["source_cell_index"] == original["source_cell_index"] and segment["source_batch"] == original["source_batch"]
        equal = equal and segment["barcode_normalized_sha256"] == original["source_barcode_normalized_sha256"]
        details.append({"source_segment_id": name, "official_candidates": len(candidates), "binding_matches": equal, "file_id": segment["file_id"]})
    checks["independent_MCOS_EDP_unique_identity_join"] = all(row["binding_matches"] for row in details)
    checks["distinct_datafiles_for_distinct_source_segments"] = len({segment["file_id"] for segment in segments.values()}) == 174
    checks["policies_trace_to_first_source_segment"] = all(cell["policy_original"] == source_cells[uid]["source_segments"][0]["policy_original"] for uid, cell in cells.items())
    aliases = []
    for later, predecessor in CONTINUATIONS.items():
        parts = cells["MATR_" + predecessor]["source_segments"]
        correct = {part["source_segment_id"] for part in parts} == {later, predecessor}
        correct = correct and segments[later]["barcode_normalized_sha256"] == segments[predecessor]["barcode_normalized_sha256"]
        correct = correct and str(segments[later]["source_channel"]) == str(segments[predecessor]["source_channel"])
        aliases.append({"predecessor": predecessor, "continuation": later, "same_physical_identity": correct,
                        "cycle_offset_certified": False})
    checks["all_five_continuations_share_UID_barcode_channel"] = all(row["same_physical_identity"] for row in aliases)
    checks["historical_selection_denominator"] = registry["raw_source_segments"] == 185 and registry["original_physical_cell_count"] == 180 and len(registry["outside_fixed169"]) == 11
    checks["source_registry_fingerprint_bound"] = bindings["source_registry_sha256"] == registry_sha
    checks["official_inventory_fingerprint_bound"] = bindings["csv_inventory_sha256"] == inventory_sha
    result = {"status": "PASS_IDENTITY_BINDINGS" if all(checks.values()) else "FAIL_IDENTITY_BINDINGS",
              "checks": checks, "fixed_cells": len(cells), "source_segments": len(segments),
              "original_source_physical_cells": registry["original_physical_cell_count"],
              "historically_excluded_physical_cells": len(registry["outside_fixed169"]),
              "complete_csv_files_at_review": bindings["complete_csv_files"],
              "scope": "Independent static identity join. Whole-source native-cycle and continuation-boundary evidence remain separate.",
              "barcode_normalization": "strip surrounding whitespace and uppercase ASCII, identical to declared source registry",
              "continuations": aliases, "segments": details,
              "bindings_sha256": bindings_sha, "source_registry_sha256": registry_sha,
              "inventory_sha256": inventory_sha, "DATA_FREEZE_allowed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [key for key, value in checks.items() if not value]}))


if __name__ == "__main__":
    main()
