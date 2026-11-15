"""Recover complete public MATR MATLAB archives with resumable byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import requests


SOURCES = (
    (1, "2017-05-12_batchdata_updated_struct_errorcorrect.mat", "5c86c0b5fa2ede00015ddf66", 3025320241),
    (2, "2017-06-30_batchdata_updated_struct_errorcorrect.mat", "5c86bf13fa2ede00015ddd82", 2007331155),
    (3, "2018-04-12_batchdata_updated_struct_errorcorrect.mat", "5c86bd64fa2ede00015ddbb2", 3236690412),
    (4, "2019-01-24_batchdata_updated_struct_errorcorrect.mat", "5dcef152110002c7215b2c90", None),
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(source):
    batch, filename, identifier, expected = source
    url = "https://data.matr.io/1/api/v1/file/" + identifier + "/download"
    with requests.get(url, headers={"Range": "bytes=0-4095", "Accept-Encoding": "identity"},
                      timeout=(20, 60), stream=True) as response:
        response.raise_for_status()
        match = re.fullmatch(r"bytes 0-4095/([0-9]+)", response.headers.get("Content-Range", ""))
        if response.status_code != 206 or match is None:
            raise RuntimeError("Official archive did not honor the range probe")
        size = int(match.group(1))
        prefix = response.raw.read(4096)
        if len(prefix) != 4096 or b"MATLAB 7.3 MAT-file" not in prefix[:128]:
            raise RuntimeError("Official archive probe lacks the MATLAB7.3 header")
        if expected is not None and size != expected:
            raise RuntimeError("Official archive size changed relative to the policy recovery")
        return {"batch": batch, "file_name": filename, "source_url": url,
                "resolved_url": response.url, "bytes": size,
                "etag": response.headers.get("ETag"), "range_supported": True,
                "first4096_sha256": hashlib.sha256(prefix).hexdigest(),
                "status": "OFFICIAL_SOURCE_RANGE_PROBED"}


def inspect_archive(path):
    with h5py.File(path, "r") as handle:
        batch = handle["batch"]
        result = {"root_fields": list(handle.keys()), "batch_fields": list(batch.keys()),
                  "batch_field_shapes": {name: list(value.shape) for name, value in batch.items()
                                         if isinstance(value, h5py.Dataset)}}
        cell_count = int(batch["cycles"].size)
        result["raw_cell_count"] = cell_count
        result["cycle_containers"] = []
        for index, reference in enumerate(batch["cycles"][()].reshape(-1)):
            cell = handle[reference]
            fields = list(cell.keys())
            field_counts = {name: int(value.size) for name, value in cell.items()
                            if isinstance(value, h5py.Dataset)}
            result["cycle_containers"].append({"source_cell_index": index,
                                                "cycle_fields": fields,
                                                "field_record_counts": field_counts})
        return result


def download(metadata, output):
    destination = output / metadata["file_name"]
    partial = destination.with_suffix(destination.suffix + ".partial")
    receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
    expected = metadata["bytes"]
    if destination.exists():
        if destination.stat().st_size != expected:
            raise RuntimeError("Existing completed archive has an unexpected byte length")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") == "COMPLETE" and receipt.get("sha256") == file_sha(destination):
                print(json.dumps({"batch": metadata["batch"], "status": "VERIFIED_EXISTING_COMPLETE"}), flush=True)
                return receipt
    else:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise RuntimeError("Partial archive is larger than the official source")
        range_size = 32 * 1024 * 1024
        with requests.Session() as session, partial.open("ab") as handle:
            while offset < expected:
                end = min(offset + range_size, expected) - 1
                for attempt in range(5):
                    try:
                        headers = {"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"}
                        with session.get(metadata["resolved_url"], headers=headers,
                                         timeout=(20, 90), stream=True) as response:
                            response.raise_for_status()
                            expected_range = f"bytes {offset}-{end}/{expected}"
                            if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                                raise RuntimeError("Range response has a different byte interval")
                            block = response.content
                            if len(block) != end - offset + 1:
                                raise RuntimeError("Truncated range response")
                            handle.write(block)
                            handle.flush()
                            offset += len(block)
                            break
                    except (requests.RequestException, RuntimeError):
                        if attempt == 4:
                            raise
                        time.sleep(min(2 ** attempt, 16))
                progress = {**metadata, "status": "DOWNLOADING", "downloaded_bytes": offset,
                            "updated_utc": datetime.now(timezone.utc).isoformat()}
                write_json(receipt_path, progress)
                print(json.dumps({"batch": metadata["batch"], "bytes": offset, "total": expected}), flush=True)
        os.replace(partial, destination)
    structure = inspect_archive(destination)
    receipt = {**metadata, "status": "COMPLETE", "sha256": file_sha(destination),
               "structure": structure, "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.discovery_only:
        routes = ["file/" + source[2] for source in SOURCES]
        routes += ["batch/5dcef1fe110002c7215b2c94", "batches/5dcef1fe110002c7215b2c94",
                   "project/5d80e633f405260001c0b60a", "projects/5d80e633f405260001c0b60a"]
        records = []
        for route in routes:
            url = "https://data.matr.io/1/api/v1/" + route
            response = requests.get(url, timeout=(20, 60))
            record = {"url": url, "http_status": response.status_code}
            try:
                record["payload"] = response.json()
            except ValueError:
                record["text_prefix"] = response.text[:2000]
            records.append(record)
            payload = record.get("payload", {})
            keys = list(payload) if isinstance(payload, dict) else "LIST"
            print(json.dumps({"route": route, "status": response.status_code, "payload_keys": keys}), flush=True)
        write_json(args.output / "OFFICIAL_DISCOVERY.json", {"records": records})
        return
    if args.metadata_only:
        records = []
        for batch, filename, identifier, _ in SOURCES:
            url = "https://data.matr.io/1/api/v1/file/" + identifier
            response = requests.get(url, timeout=(20, 60))
            record = {"batch": batch, "file_name": filename, "url": url,
                      "http_status": response.status_code}
            if response.ok:
                payload = response.json()
                record["official_metadata"] = {name: payload.get(name) for name in
                                                ("name", "size", "sha512", "mimeType", "created", "updated")}
            records.append(record)
        write_json(args.output / "OFFICIAL_FILE_METADATA.json", {"sources": records})
        print(json.dumps(records), flush=True)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        metadata = list(executor.map(probe, SOURCES))
    write_json(args.output / "SOURCE_AVAILABILITY.json", {"status": "ALL4_OFFICIAL_ARCHIVES_ACCESSIBLE",
               "total_source_bytes": sum(item["bytes"] for item in metadata), "sources": metadata})
    print(json.dumps({"status": "ALL4_OFFICIAL_ARCHIVES_ACCESSIBLE", "sources": metadata}), flush=True)
    if args.probe_only:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        receipts = list(executor.map(lambda item: download(item, args.output), metadata))
    write_json(args.output / "SOURCE_RECOVERY_RECEIPT.json", {"status": "COMPLETE", "sources": receipts,
               "total_bytes": sum(item["bytes"] for item in receipts),
               "scope": "Official original MATLAB archives, preservation before any cycle reconstruction"})
    print(json.dumps({"status": "COMPLETE", "files": len(receipts)}), flush=True)


if __name__ == "__main__":
    main()
