"""Inventory and recover the official original per-test instrument CSV sources."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from recover_sources import file_sha


BASE = "https://data.matr.io/1/api/v1/"
BATCHES = ((1, "5c48dd2bc625d700019f3204", "5c86c0b5fa2ede00015ddf67"),
           (2, "5c48dd2bc625d700019f3204", "5c86bf14fa2ede00015ddd83"),
           (3, "5c48dd2bc625d700019f3204", "5c86bd64fa2ede00015ddbb3"),
           (4, "5d80e633f405260001c0b60a", "5dcef1fe110002c7215b2c94"))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def metadata(test, cache=None):
    path = cache / (test["dataFileId"] + ".json") if cache is not None else None
    if path is not None and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)))
    session.mount("https://", adapter)
    response = session.get(BASE + "file/" + test["dataFileId"], timeout=(20, 60))
    response.raise_for_status()
    file = response.json()
    source_url = BASE + "file/" + file["_id"] + "/download"
    with session.get(source_url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
                      timeout=(20, 60), stream=True) as probe:
        probe.raise_for_status()
        match = re.fullmatch(r"bytes 0-0/([0-9]+)", probe.headers.get("Content-Range", ""))
        if probe.status_code != 206 or match is None:
            raise RuntimeError("Original CSV did not honor the one-byte source-size probe")
        actual_size = int(match.group(1))
        resolved_url = probe.url
        etag = probe.headers.get("ETag")
    result = {"source_batch": test["source_batch"], "test_id": test["_id"],
            "source_channel": test["channel"], "source_cell_barcode": test.get("cellId"),
            "source_cell_barcode_sha256": hashlib.sha256(str(test.get("cellId")).encode()).hexdigest(),
            "test_name": test.get("name"), "file_id": file["_id"],
            "file_name": file["name"], "bytes": actual_size,
            "publisher_metadata_bytes": file["size"],
            "publisher_metadata_size_disagrees": actual_size != file["size"],
            "source_url": source_url, "resolved_url": resolved_url, "etag": etag,
            "metadata_url": BASE + "file/" + file["_id"],
            "test_metadata_file_id": test.get("metaDataFileId"),
            "schedule_file_id": test.get("scheduleFileId")}
    session.close()
    if path is not None:
        write_json(path, result)
    return result


def inventory(output, workers):
    tests = []
    for number, project, batch in BATCHES:
        url = BASE + f"edp/projects/{project}/batches/{batch}/tests"
        response = requests.get(url, timeout=(20, 60))
        response.raise_for_status()
        rows = response.json()
        write_json(output / f"official_batch{number}_tests.json", {"url": url, "tests": rows})
        tests.extend({**row, "source_batch": number} for row in rows if row.get("dataFileId"))
    cache = output / "source_metadata_cache"
    cache.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        files = list(executor.map(lambda test: metadata(test, cache), tests))
    result = {"status": "ORIGINAL_CSV_INVENTORY_COMPLETE", "files": files,
              "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files),
              "actual_sizes_verified_with_HTTP_range": True,
              "batch_counts": {str(number): sum(row["source_batch"] == number for row in files) for number, _, _ in BATCHES},
              "batch_bytes": {str(number): sum(row["bytes"] for row in files if row["source_batch"] == number) for number, _, _ in BATCHES}}
    write_json(output / "INSTRUMENT_CSV_INVENTORY.json", result)
    return result


def probe_sample(row, output):
    with requests.get(row["source_url"], headers={"Range": "bytes=0-65535", "Accept-Encoding": "identity"},
                      stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith("bytes 0-"):
            raise RuntimeError("Instrument CSV range probe was not honored")
        prefix = response.content
        path = output / ("sample_batch" + str(row["source_batch"]) + "_" + row["file_id"] + ".csv.prefix")
        path.write_bytes(prefix)
        text = prefix.decode("utf-8-sig", errors="replace")
        lines = text.splitlines()
        schema = {"source_batch": row["source_batch"], "test_id": row["test_id"],
                  "file_id": row["file_id"], "file_name": row["file_name"], "bytes": row["bytes"],
                  "source_channel": row["source_channel"], "source_cell_barcode_sha256": row["source_cell_barcode_sha256"],
                  "resolved_url": response.url, "etag": response.headers.get("ETag"),
                  "range_supported": True, "content_range": response.headers.get("Content-Range"),
                  "sample_file": path.name, "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                  "first8_lines": lines[:8]}
        write_json(output / ("CSV_SCHEMA_BATCH" + str(row["source_batch"]) + ".json"), schema)
        return schema


def download(row, output):
    directory = output / ("batch" + str(row["source_batch"]))
    directory.mkdir(exist_ok=True)
    name = row["file_id"] + "_" + Path(row["file_name"]).name
    destination = directory / name
    partial = destination.with_suffix(destination.suffix + ".partial")
    receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
    if destination.exists() and destination.stat().st_size != row["bytes"]:
        raise RuntimeError("Existing completed CSV has an unexpected byte length")
    if destination.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "COMPLETE" and destination.stat().st_size == row["bytes"]:
            if receipt.get("sha256") != file_sha(destination):
                raise RuntimeError("Existing completed CSV differs from its full-file SHA receipt")
            return receipt
    if not destination.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > row["bytes"]:
            raise RuntimeError("Partial CSV is larger than its official source")
        with requests.Session() as session, partial.open("ab") as handle:
            while offset < row["bytes"]:
                end = min(offset + 16 * 1024 * 1024, row["bytes"]) - 1
                for attempt in range(5):
                    try:
                        with session.get(row.get("resolved_url", row["source_url"]), headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
                                         timeout=(20, 90)) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{row['bytes']}":
                                raise RuntimeError("Instrument CSV response has a different byte range")
                            if row.get("etag") and response.headers.get("ETag") != row["etag"]:
                                raise RuntimeError("Instrument CSV entity changed after its registered range probe")
                            block = response.content
                            if len(block) != end - offset + 1:
                                raise RuntimeError("Instrument CSV response is truncated")
                            handle.write(block)
                            handle.flush()
                            offset += len(block)
                            break
                    except (requests.RequestException, RuntimeError):
                        if attempt == 4:
                            raise
                        time.sleep(min(2 ** attempt, 16))
                write_json(receipt_path, {**row, "status": "DOWNLOADING", "downloaded_bytes": offset})
        os.replace(partial, destination)
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    receipt = {**row, "status": "COMPLETE", "sha256": digest.hexdigest(),
               "relative_path": destination.relative_to(output).as_posix()}
    write_json(receipt_path, receipt)
    print(json.dumps({"status": "CSV_COMPLETE", "source_batch": row["source_batch"],
                      "source_channel": row["source_channel"], "bytes": row["bytes"],
                      "relative_path": receipt["relative_path"]}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--file-ids", nargs="+")
    parser.add_argument("--refresh-inventory", action="store_true")
    parser.add_argument("--skip-schema-probes", action="store_true")
    parser.add_argument("--source-registry", type=Path)
    parser.add_argument("--bindings-output", type=Path)
    parser.add_argument("--prioritize-continuations", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output / "INSTRUMENT_CSV_INVENTORY.json"
    result = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() and not args.refresh_inventory else inventory(args.output, args.workers)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}), flush=True)
    samples = [next(row for row in result["files"] if row["source_batch"] == number) for number, _, _ in BATCHES]
    for row in ([] if args.skip_schema_probes else samples):
        schema = probe_sample(row, args.output)
        print(json.dumps({key: value for key, value in schema.items() if key != "first8_lines"}), flush=True)
        print(json.dumps({"source_batch": schema["source_batch"], "schema_lines": schema["first8_lines"][:2]}), flush=True)
    if not args.download:
        return
    files = [row for row in result["files"] if args.file_ids is None or row["file_id"] in args.file_ids]
    if args.source_registry is not None:
        from build_instrument_bindings import refresh_bindings
        bindings = refresh_bindings(args.source_registry, args.output, args.bindings_output)
        required = set(bindings["required_file_ids"])
        files = [row for row in files if row["file_id"] in required]
        if args.prioritize_continuations:
            priority = [segment["file_id"] for cell in bindings["cells"] if len(cell["source_segments"]) > 1
                        for segment in cell["source_segments"]]
            for batch_number in (3, 4):
                first = next(segment for cell in bindings["cells"] for segment in cell["source_segments"]
                             if segment["source_batch"] == batch_number)
                priority.append(first["file_id"])
            rank = {identifier: index for index, identifier in enumerate(priority)}
            files.sort(key=lambda row: rank.get(row["file_id"], len(rank)))
    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, row, args.output) for row in files]
        for future in concurrent.futures.as_completed(futures):
            receipts.append(future.result())
            if args.source_registry is not None:
                refresh_bindings(args.source_registry, args.output, args.bindings_output)
    write_json(args.output / "INSTRUMENT_CSV_RECOVERY_RECEIPT.json", {"status": "COMPLETE", "sources": receipts,
               "total_bytes": sum(row["bytes"] for row in receipts)})


if __name__ == "__main__":
    main()
