"""Read-only source inventory for MATR/Stanford reuse in the HUST idea audit."""

from __future__ import annotations

import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = REPOSITORY.parent / "private"
DATASET = Path(r"D:\develop\patchtst\BatteryLife\dataset")
OUTPUT = REPOSITORY / "results/hust_raw_idea_audit_v2/OTHER_SOURCE_INVENTORY.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def incomplete_downloads():
    return [{"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
             "source_identity": "unconfirmed", "complete_source_usable": False}
            for path in sorted(DATASET.glob("*.crdownload"))]


def check_source(path, expected_bytes, expected_sha256, identity):
    before = path.stat()
    actual_sha = sha256(path)
    after = path.stat()
    return {**identity, "path": str(path), "actual_bytes": after.st_size,
            "expected_bytes": expected_bytes, "actual_sha256": actual_sha,
            "expected_sha256": expected_sha256,
            "complete_bytes_and_sha256_match": after.st_size == expected_bytes and actual_sha == expected_sha256,
            "unchanged_during_read": before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns}


def main():
    before_downloads = incomplete_downloads()
    data_root = REPOSITORY / "data/unified_matr_seed42"
    manifest = read_json(data_root / "dataset_manifest.json")
    freeze = read_json(data_root / "DATA_FREEZE.json")
    bindings = read_json(data_root / "instrument_identity_bindings.json")
    matr_root = PRIVATE / "raw_sources/MATR"
    mat = []
    for source in manifest["source_mat_archives"]:
        mat.append(check_source(matr_root / source["file_name"], source["bytes"], source["sha256"],
                                {"source_batch": source["batch"], "source_url": source["source_url"]}))
        print(json.dumps({"stage": "MATR_full_SHA256", "mat_files": len(mat), "required": 4}), flush=True)
    source_manifest = {row["source_segment_id"]: row for row in manifest["source_instrument_csv"]}
    native = []
    for cell in bindings["cells"]:
        for segment in cell["source_segments"]:
            source = source_manifest[segment["source_segment_id"]]
            entry = check_source(Path(segment["csv_path"]), source["bytes"], source["sha256"],
                                 {"cell_uid": cell["cell_uid"], "source_segment_id": source["source_segment_id"],
                                  "source_url": source["source_url"]})
            entry["binding_sha256_matches_frozen_manifest"] = segment["sha256"] == source["sha256"]
            native.append(entry)
            if len(native) % 25 == 0:
                print(json.dumps({"stage": "MATR_full_SHA256", "native_files": len(native), "required": 174}), flush=True)
    metadata = read_json(REPOSITORY / "results/unified_matr/instrument_metadata_audit.json")
    schedules = read_json(REPOSITORY / "results/unified_matr/schedule_metadata_audit.json")
    metadata_checks = [check_source(matr_root / "instrument_csv" / row["metadata_source_path"], row["metadata_bytes"], row["metadata_sha256"],
                                    {"source_segment_id": row["source_segment_id"]}) for row in metadata["source_rows"]]
    schedule_checks = [check_source(matr_root / "instrument_csv" / row["schedule_source_path"], row["schedule_bytes"], row["schedule_sha256"],
                                    {"source_segment_id": row["source_segment_id"]}) for row in schedules["source_links"]]
    frozen_checks = [{"path": entry["path"], "sha256_matches": sha256(data_root / entry["path"]) == entry["sha256"]} for entry in freeze["files"]]
    metadata_root = PRIVATE / "stanford_source_metadata"
    stanford_metadata = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(metadata_root.iterdir()) if path.is_file()]
    parameter_file = metadata_root / "Formation_2022-Parameter.csv"
    with parameter_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        parameter_fields = reader.fieldnames
        parameter_rows = sum(1 for _ in reader)
    roots = [DATASET / "DATA", DATASET / "external", DATASET.parent / "external", REPOSITORY.parents[1] / "external",
             REPOSITORY / "external", PRIVATE]
    scanned = []
    raw_candidates = []
    for root in roots:
        files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
        candidates = [path for path in files if path.name.lower() in ("data.tar.gz", "data_03032025.tgz")
                      or "maccor" in str(path).lower() or ("nova" in path.name.lower() and ".json" in path.name.lower())]
        raw_candidates.extend(str(path) for path in candidates)
        scanned.append({"root": str(root), "exists": root.exists(), "file_count": len(files),
                        "total_bytes": sum(path.stat().st_size for path in files),
                        "extensions": dict(collections.Counter(path.suffix for path in files)),
                        "stanford_raw_candidates": [str(path) for path in candidates]})
    official = read_json(PRIVATE / "new_raw_idea_sources/Stanford/OFFICIAL_REMOTE_ARCHIVES.json")
    progress_path = PRIVATE / "new_raw_idea_sources/Stanford/BROWSER_HEADER_OBSERVATIONS.json"
    progress = read_json(progress_path) if progress_path.is_file() else {}
    all_source = mat + native + metadata_checks + schedule_checks
    matr_ok = len(mat) == 4 and len(native) == len(metadata_checks) == len(schedule_checks) == 174
    matr_ok = matr_ok and all(item["complete_bytes_and_sha256_match"] and item["unchanged_during_read"] for item in all_source)
    matr_ok = matr_ok and all(item["binding_sha256_matches_frozen_manifest"] for item in native)
    matr_ok = matr_ok and freeze["status"] == "DATA_FREEZE_PASS" and all(item["sha256_matches"] for item in frozen_checks)
    disk = shutil.disk_usage(DATASET)
    result = {
        "status": "MATR_COMPLETE_REUSE_STANFORD_BROWSER_BUNDLE_IN_PROGRESS" if matr_ok and progress else "MATR_COMPLETE_REUSE_STANFORD_RAW_NOT_YET_LOCAL" if matr_ok and not raw_candidates else "SOURCE_INVENTORY_REVIEW_REQUIRED",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "script_sha256": sha256(Path(__file__)),
        "scope": "MATR and Stanford raw-source availability for the new HUST-focused idea audit. HUST native processing is audited separately.",
        "disk_bytes": dict(zip(("total", "used", "free"), disk)),
        "MATR": {"ready_for_reuse": matr_ok, "new_download_required": False if matr_ok else None,
                 "physical_cells": len(bindings["cells"]), "mat_files": mat, "mat_total_bytes": sum(row["actual_bytes"] for row in mat),
                 "native_csv_files": native, "native_csv_total_bytes": sum(row["actual_bytes"] for row in native),
                 "metadata_files": metadata_checks, "schedule_files": schedule_checks,
                 "metadata_total_bytes": sum(row["actual_bytes"] for row in metadata_checks),
                 "schedule_total_bytes": sum(row["actual_bytes"] for row in schedule_checks),
                 "data_freeze_sha256": sha256(data_root / "DATA_FREEZE.json"), "frozen_file_checks": frozen_checks,
                 "frozen_source_receipts": freeze["source_SHA_and_axis_receipts"],
                 "source_registry_historical_status": read_json(REPOSITORY / "results/unified_matr/source_registry.json")["status"],
                 "latest_axis_evidence": "The completed DATA_FREEZE and authenticated native mapping supersede the early registry's pending status label."},
        "Stanford": {"searched_locations": scanned, "full_raw_archive_or_maccor_found": bool(raw_candidates),
                     "in_progress_bundle_header_observation": progress,
                     "in_progress_bundle_is_complete_source": False,
                     "existing_parameter_and_helper_files": stanford_metadata, "formation_parameter_rows": parameter_rows,
                     "formation_parameter_schema": parameter_fields, "official_source_metadata": official,
                     "official_source_semantics": "The current official bundle includes raw formation records and BEEP-structured/interpolated aging cycles plus summaries. Native aging acquisition records require member/schema confirmation.",
                     "assistant_initiated_full_archive_download": False,
                     "download_decision": "Only the official small source page and HTTP metadata were recovered. The current HUST-focused objective is independently supported. A Stanford-specific physical-evidence need can trigger the already authorized download after checking existing browser transfers to avoid duplication."},
        "dataset_DATA_identity": {"source": "Discovery Learning accelerates battery design evaluation", "arxiv": "2508.06985",
                                  "readme_reported_cells": 123, "readme_reported_groups": 37, "readme_update": "2026-06-05",
                                  "inventory_note": "This DATA directory has its own source identity and is not classified as Stanford."},
        "incomplete_downloads_before": before_downloads, "incomplete_downloads_after": incomplete_downloads(),
        "incomplete_download_action": "read_only_stat_observation", "original_source_mutations": 0,
        "registered_or_frozen_artifact_mutations": 0, "model_fits": 0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "MATR_ready": matr_ok, "MATR_mat_bytes": result["MATR"]["mat_total_bytes"],
                      "MATR_native_bytes": result["MATR"]["native_csv_total_bytes"], "Stanford_raw_candidates": len(raw_candidates), "output": str(OUTPUT)}, ensure_ascii=False))
    if not matr_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
