"""Publish original archive and physical-cell segment bindings for the fixed169."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py

from raw_reader import CONTINUATIONS, PREDECESSORS, readable_policy, static_source_field
from recover_sources import SOURCES


AUTHOR_IDENTITY_URL = "https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb"
OUTSIDE_FIXED = {"b1c8": "AUTHOR_EXCLUDED_NO_OBSERVED_EOL80",
                 "b1c10": "AUTHOR_EXCLUDED_NO_OBSERVED_EOL80",
                 "b1c12": "AUTHOR_EXCLUDED_NO_OBSERVED_EOL80",
                 "b1c13": "AUTHOR_EXCLUDED_NO_OBSERVED_EOL80",
                 "b1c22": "AUTHOR_EXCLUDED_NO_OBSERVED_EOL80",
                 "b3c2": "AUTHOR_EXCLUDED_NOISY_CHANNEL", "b3c23": "AUTHOR_EXCLUDED_NOISY_CHANNEL",
                 "b3c32": "AUTHOR_EXCLUDED_NOISY_CHANNEL", "b3c37": "AUTHOR_EXCLUDED_NOISY_CHANNEL",
                 "b3c42": "AUTHOR_EXCLUDED_NOISY_CHANNEL", "b3c43": "AUTHOR_EXCLUDED_NOISY_CHANNEL"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--policy-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy_root = args.policy_root or args.repository.parents[2] / "BatteryLife/dataset/MATR_policy_metadata_20261029"
    eligible_path = args.repository / "results/d0/eligible_cells.csv"
    with eligible_path.open(encoding="utf-8-sig", newline="") as handle:
        fixed = {row["cell_id"] for row in csv.DictReader(handle) if row["family"] == "MATR"}
    if len(fixed) != 169:
        raise ValueError("The registered main cohort must retain its fixed169 identities")
    source_files, all_segments = [], []
    official_metadata_path = args.source_root / "OFFICIAL_FILE_METADATA.json"
    official_metadata = json.loads(official_metadata_path.read_text(encoding="utf-8"))
    publisher_files = {item["batch"]: item for item in official_metadata["sources"]}
    for number, filename, _, _ in SOURCES:
        path = args.source_root / filename
        receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text(encoding="utf-8"))
        if receipt["status"] != "COMPLETE" or path.stat().st_size != receipt["bytes"]:
            raise ValueError("Complete source recovery is required for the source registry")
        source_identity = {key: receipt[key] for key in ("batch", "file_name", "bytes", "sha256", "source_url", "resolved_url", "etag")}
        publisher = publisher_files[number]
        details = publisher["official_metadata"]
        if publisher["http_status"] != 200 or details["size"] != receipt["bytes"] or details["name"] != filename:
            raise ValueError("Publisher API identity differs from the recovered complete archive")
        source_identity["official_file_metadata"] = details
        source_identity["sha256_scope"] = "Computed from complete recovered file; publisher API has no sha512 field value"
        if number <= 3:
            policy_path = policy_root / f"batch{number}_original_policies.json"
            policy_archive = json.loads(policy_path.read_text(encoding="utf-8"))
            with path.open("rb") as handle:
                for interval in policy_archive["range_source"]["ranges"]:
                    handle.seek(interval["start"])
                    actual = hashlib.sha256(handle.read(interval["bytes"])).hexdigest()
                    if actual != interval["sha256"]:
                        raise ValueError("Recovered complete source differs from the prior original-policy byte ranges")
            source_identity["prior_static_policy_metadata_sha256"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            source_identity["prior_static_policy_ranges_replayed"] = len(policy_archive["range_source"]["ranges"])
            source_identity["prior_static_policy_range_bytes_replayed"] = policy_archive["range_source"]["downloaded_range_bytes"]
        source_files.append(source_identity)
        with h5py.File(path, "r") as handle:
            batch = handle["batch"]
            count = int(batch["cycles"].size)
            containers = receipt["structure"]["cycle_containers"]
            for index in range(count):
                identifier = f"b{number}c{index}"
                barcode = str(static_source_field(handle, batch, "barcode", index))
                channel = static_source_field(handle, batch, "channel_id", index)
                counts = containers[index]["field_record_counts"]
                meta = {"source_file_name": filename, "source_batch": number,
                        "source_cell_index": index, "source_cell_id": identifier,
                        "source_segment_id": identifier,
                        "cell_uid": "MATR_" + CONTINUATIONS.get(identifier, identifier),
                        "continuation_of": CONTINUATIONS.get(identifier),
                        "continuation_segment": PREDECESSORS.get(identifier),
                        "source_barcode_sha256": hashlib.sha256(barcode.encode()).hexdigest(),
                        "source_barcode_normalized_sha256": hashlib.sha256(barcode.strip().upper().encode()).hexdigest(),
                        "source_barcode_normalization": "strip surrounding whitespace and uppercase ASCII barcode",
                        "source_channel_id": channel,
                        "policy_original": readable_policy(handle, batch, index),
                        "raw_records": counts["I"], "waveform_field_record_counts": counts,
                        "summary_cycle_provenance": "Author cell_analysis.m assigns summary.cycle=(1:k), original instrument Cycle_Index absent",
                        "instrument_cycle_axis_authenticated": False, "raw_cycle_certified": False}
                meta["source_file_sha256"] = receipt["sha256"]
                meta["fixed169_member"] = meta["cell_uid"] in fixed
                all_segments.append(meta)
                print(json.dumps({"batch": number, "source_cell_index": index, "status": "SOURCE_IDENTITY_READ"}), flush=True)
    grouped = {identifier: [] for identifier in sorted(fixed)}
    outside = []
    for segment in all_segments:
        if segment["fixed169_member"]:
            grouped[segment["cell_uid"]].append(segment)
        else:
            outside.append({"source_cell_id": segment["source_cell_id"], "cell_uid": segment["cell_uid"],
                            "reason": OUTSIDE_FIXED.get(segment["source_cell_id"], "UNEXPLAINED_OUTSIDE_FIXED_COHORT")})
    cells = []
    for identifier, segments in grouped.items():
        if not segments:
            raise ValueError("Fixed cohort identity has no source segment")
        segments.sort(key=lambda item: item["source_batch"])
        cells.append({"cell_uid": identifier, "provided_cell_id": identifier,
                      "chemistry": "LFP/graphite", "source_segments": segments,
                      "raw_cycle_certified": False,
                      "certification_pending": "Original instrument Cycle_Index or no-gap source proof and full phase measurement QA"})
    expected = len(all_segments) == 185 and sum(map(len, grouped.values())) == 174 and len(outside) == 11
    if not expected or {row["source_cell_id"] for row in outside} != set(OUTSIDE_FIXED):
        raise ValueError("Original source inventory differs from the fixed-cohort identity contract")
    result = {"status": "COMPLETE_RELEASE_SOURCE_IDENTITY_REGISTRY_PENDING_INSTRUMENT_CYCLE_MAPPING",
              "source_files": source_files, "raw_source_segments": len(all_segments),
              "fixed169_cells": len(cells), "fixed169_source_segments": 174,
              "original_physical_cell_count": 180,
              "historical_cohort_selection": "The inherited fixed169 excludes11 original physical cells; retain that selection history. Future-label support is outside admission rules for the fixed cohort.",
              "outside_fixed169": outside, "cells": cells,
              "all_released_source_segments": all_segments,
              "source_identity_evidence": AUTHOR_IDENTITY_URL,
              "legacy_eligible_sha256": hashlib.sha256(eligible_path.read_bytes()).hexdigest(),
              "scope": "Released archive identities and author-regenerated summary coordinates. Original instrument cycle mapping remains unauthenticated."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "fixed169_cells": len(cells),
                      "source_segments": len(all_segments), "outside_fixed169": len(outside)}))


if __name__ == "__main__":
    main()
