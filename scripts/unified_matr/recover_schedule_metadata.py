"""Recover original Arbin schedules and preserve every static protocol field."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://data.matr.io/1/api/v1/file/"
PRIVATE_FIELDS = {"m_szOriginalInfo", "m_szCreator", "m_szComments", "Signature"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def retrieve(file_id, root):
    path = root / (file_id + ".sdu")
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        body = path.read_bytes()
        if len(body) != receipt["bytes"] or digest(body) != receipt["sha256"]:
            raise ValueError("Recovered schedule differs from its whole-file source receipt")
        return receipt
    with requests.Session() as session:
        session.trust_env = False
        session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1,
                                                               status_forcelist=(429, 500, 502, 503, 504))))
        response = session.get(BASE + file_id, timeout=(20, 60))
        response.raise_for_status()
        official = response.json()
        write_json(root / (file_id + ".file.json"), official)
        response = session.get(BASE + file_id + "/download", timeout=(20, 60))
        response.raise_for_status()
        body = response.content
        receipt = {"status": "COMPLETE", "file_id": file_id, "file_name": official["name"],
                   "source_url": BASE + file_id + "/download", "resolved_url": response.url,
                   "bytes": len(body), "publisher_metadata_bytes": official["size"],
                   "publisher_metadata_size_disagrees": len(body) != official["size"],
                   "etag": response.headers.get("ETag"), "sha256": digest(body), "path": str(path.resolve())}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    os.replace(temporary, path)
    write_json(receipt_path, receipt)
    return receipt


def parse_schedule(body):
    """Parse exact key/value text with byte-reversible decoding and no coercion."""
    try:
        text = body.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = body.decode("latin-1")
        encoding = "latin-1-byte-preserving"
    sections, current = {}, None
    for line_number, line in enumerate(re.split(r"\r\n|\n|\r", text), 1):
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            if current in sections:
                raise ValueError(f"Duplicate original schedule section at line {line_number}")
            sections[current] = {}
        else:
            if current is None or "=" not in line:
                raise ValueError(f"Unparsed original schedule line {line_number}")
            key, value = line.split("=", 1)
            if key in sections[current]:
                raise ValueError(f"Duplicate original schedule key at line {line_number}")
            sections[current][key] = value
    if "Schedule" not in sections or "Version Section" not in sections:
        raise ValueError("Missing original Arbin schedule/version section")
    return sections, encoding


def summarize_schedule(sections):
    steps = []
    for section, fields in sections.items():
        match = re.fullmatch(r"Schedule_Step([0-9]+)", section)
        if match is None:
            continue
        limits = {name: values for name, values in sections.items()
                  if name.startswith(section + "_Limit")}
        steps.append({"source_step_index": int(match.group(1)), "source_section": section,
                      "label": fields.get("m_szLabel"), "control_type": fields.get("m_szStepCtrlType"),
                      "control_value_original": fields.get("m_szCtrlValue"),
                      "ext_control_value1_original": fields.get("m_szExtCtrlValue1"),
                      "ext_control_value2_original": fields.get("m_szExtCtrlValue2"),
                      "unit_type_original": fields.get("m_szUnitType"),
                      "ext_unit_type1_original": fields.get("m_szExUnitType1"),
                      "declared_limit_count_original": fields.get("m_uLimitNum"),
                      "parsed_limit_count": len(limits), "limits": limits})
    steps.sort(key=lambda step: step["source_step_index"])
    declared = int(sections["Schedule"]["m_uStepNum"])
    if declared != len(steps) or [step["source_step_index"] for step in steps] != list(range(declared)):
        raise ValueError("Original schedule step count or contiguous section indexes differ")
    if any(int(step["declared_limit_count_original"]) != step["parsed_limit_count"] for step in steps):
        raise ValueError("Original schedule limit count differs from parsed sections")
    return steps


def audit(bindings, inventory, csv_root, output, workers):
    lookup = {row["file_id"]: row for row in inventory["files"]}
    selected = [(cell["cell_uid"], segment, lookup[segment["file_id"]])
                for cell in bindings["cells"] for segment in cell["source_segments"]]
    file_ids = sorted({source["schedule_file_id"] for _, _, source in selected})
    if len(selected) != 174 or any(file_id is None for file_id in file_ids):
        raise ValueError("All 174 fixed-cohort segments require an original schedule file identity")
    root = csv_root / "instrument_schedules"
    root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        receipts = dict(zip(file_ids, executor.map(lambda file_id: retrieve(file_id, root), file_ids)))
    unique, parsed, long_rows = {}, {}, []
    for file_id in file_ids:
        receipt = receipts[file_id]
        body = Path(receipt["path"]).read_bytes()
        if digest(body) != receipt["sha256"]:
            raise ValueError("Original schedule changed between recovery and parsing")
        sections, encoding = parse_schedule(body)
        steps = summarize_schedule(sections)
        parsed[file_id] = (sections, steps)
        source_sha = receipt["sha256"]
        if source_sha in unique:
            unique[source_sha]["official_file_ids"].append(file_id)
            continue
        public_sections = {}
        for section, fields in sections.items():
            public_sections[section] = {}
            for field, value in fields.items():
                redacted = field in PRIVATE_FIELDS
                public_field = field + "_sha256" if redacted else field
                public_value = digest(value.encode()) if redacted and value else value
                public_sections[section][public_field] = public_value
                long_rows.append({"schedule_sha256": source_sha, "source_section": section,
                                  "source_field": field, "source_value": public_value,
                                  "source_value_redacted_to_sha256": redacted,
                                  "source_value_missing": value == ""})
        unique[source_sha] = {"official_file_ids": [file_id], "source_bytes": len(body),
                              "decoding": encoding, "sections": public_sections, "steps": steps}
    links = []
    for uid, segment, source in selected:
        receipt = receipts[source["schedule_file_id"]]
        sections, steps = parsed[receipt["file_id"]]
        discharge_steps = [step for step in steps if "discharge" in (step["label"] or "").lower()]
        links.append({"cell_uid": uid, "source_segment_id": segment["source_segment_id"],
                      "source_batch": segment["source_batch"], "official_edp_test_id": segment["test_id"],
                      "schedule_file_id": receipt["file_id"], "schedule_file_name": receipt["file_name"],
                      "schedule_sha256": receipt["sha256"], "schedule_bytes": receipt["bytes"],
                      "schedule_source_url": receipt["source_url"],
                      "schedule_source_path": "instrument_schedules/" + receipt["file_id"] + ".sdu",
                      "declared_step_count": len(steps),
                      "discharge_labeled_steps": [{key: step[key] for key in ("source_step_index", "label", "control_type", "control_value_original", "ext_control_value1_original", "unit_type_original")} for step in discharge_steps]})
    output.mkdir(parents=True, exist_ok=True)
    parquet = output / "schedule_static_fields.parquet"
    pd.DataFrame(long_rows).to_parquet(parquet, index=False)
    result = {"status": "PASS", "fixed_physical_cells": bindings["fixed_cells"],
              "source_segment_links": len(links), "complete_schedule_files": len(receipts),
              "total_schedule_source_bytes": sum(item["bytes"] for item in receipts.values()),
              "unique_schedule_content_hashes": len(unique), "unique_schedule_field_rows": len(long_rows),
              "source_segment_step_count_distribution": dict(Counter(link["declared_step_count"] for link in links)),
              "schedule_static_fields_parquet_sha256": digest(parquet.read_bytes()),
              "source_links": links, "unique_schedules_by_sha256": unique,
              "field_semantics": "Exact original strings, including empty values, units, display factors, labels, control values, limits, equations and destinations. No numeric coercion or unit-factor application.",
              "native_step_mapping": "Original Schedule_Step section indexes retained. Native Step_Index identity and actual stage membership require native row replay.",
              "redaction": "Original machine/user/path metadata, creator/comments and signature values remain private. All static Step and Limit fields preserved verbatim.",
              "model_input_columns_added": 0, "measurement_threshold_changes": 0,
              "code_sha256": digest(Path(__file__).read_bytes())}
    write_json(output / "schedule_metadata_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", type=Path, default=Path("../private/raw_sources/MATR/instrument_csv"))
    parser.add_argument("--output", type=Path, default=Path("results/unified_matr"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    bindings = json.loads((args.output / "instrument_bindings.json").read_text(encoding="utf-8"))
    inventory = json.loads((args.csv_root / "INSTRUMENT_CSV_INVENTORY.json").read_text(encoding="utf-8"))
    result = audit(bindings, inventory, args.csv_root, args.output, args.workers)
    print(json.dumps({key: result[key] for key in ("status", "source_segment_links", "complete_schedule_files", "total_schedule_source_bytes", "unique_schedule_content_hashes", "unique_schedule_field_rows")}))


if __name__ == "__main__":
    main()
