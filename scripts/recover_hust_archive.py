"""Recover the official HUST archive with resume, source SHA and ZIP CRC checks."""

from __future__ import annotations

import hashlib
import argparse
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


REPOSITORY = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-metadata", type=Path, default=REPOSITORY / "configs/hust_source_registry.json")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))
    source = metadata["files"][0]
    expected_size = source["size"]
    expected_sha = source["content_details"]["sha256_hash"]
    url = source["content_details"]["download_url"]
    destination = args.destination.resolve()
    if destination.is_relative_to(REPOSITORY.resolve()):
        raise ValueError("Raw source destination must be outside the publication repository")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "our_data.zip"
    part = destination / "our_data.zip.part"
    started = time.monotonic()
    attempts = []
    if not archive.exists():
        for attempt in range(1, 7):
            offset = part.stat().st_size if part.exists() else 0
            if offset == expected_size:
                break
            if offset > expected_size:
                raise RuntimeError("Existing partial archive exceeds source length")
            headers = {"Range": f"bytes={offset}-{expected_size - 1}", "Accept-Encoding": "identity"}
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise RuntimeError("Resume request did not return HTTP 206")
                    expected_range = f"bytes {offset}-{expected_size - 1}/{expected_size}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Source Content-Range mismatch")
                    progress = offset
                    last_reported = offset
                    with part.open("ab") as file:
                        for block in response.iter_content(chunk_size=2 * 1024 * 1024):
                            if block:
                                file.write(block)
                                progress += len(block)
                            if progress - last_reported >= 64 * 1024 * 1024:
                                file.flush()
                                print(json.dumps({"downloaded_bytes": progress, "expected_bytes": expected_size,
                                                  "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
                                last_reported = progress
                    attempts.append({"attempt": attempt, "offset": offset, "status": "complete"})
            except (requests.RequestException, OSError) as exc:
                attempts.append({"attempt": attempt, "offset": offset, "error": type(exc).__name__})
                if attempt == 6:
                    raise
                time.sleep(min(2 ** attempt, 30))
        if not part.exists() or part.stat().st_size != expected_size:
            raise RuntimeError("Official archive download remains incomplete")
        actual_sha = sha256(part)
        if actual_sha != expected_sha:
            raise RuntimeError("Official archive SHA256 mismatch")
        part.replace(archive)
    else:
        actual_sha = sha256(archive)
        if archive.stat().st_size != expected_size or actual_sha != expected_sha:
            raise RuntimeError("Existing official archive has incompatible identity")

    members = []
    destination_resolved = destination
    with zipfile.ZipFile(archive) as zipped:
        expected_members = {entry["name"]: entry for entry in metadata["zip_members"]}
        if {item.filename for item in zipped.infolist()} != set(expected_members):
            raise RuntimeError("ZIP member list changed")
        for item in zipped.infolist():
            if item.is_dir():
                continue
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination_resolved) or target.suffix != ".pkl":
                raise RuntimeError("Unexpected ZIP member destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            temporary = target.with_suffix(target.suffix + ".tmp")
            with zipped.open(item) as source_file, temporary.open("wb") as output:
                for block in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
            if temporary.stat().st_size != item.file_size:
                raise RuntimeError("ZIP member length mismatch")
            temporary.replace(target)
            members.append({"member": item.filename, "bytes": item.file_size, "CRC32": f"{item.CRC:08x}",
                            "CRC32_verified": True, "sha256": digest.hexdigest(), "path": str(target)})
    receipt = {"status": "OFFICIAL_HUST_SOURCE_RECOVERED", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "official_record": "https://data.mendeley.com/datasets/nsc7hnsg4s/2", "source_url": url,
               "archive": str(archive), "archive_bytes": archive.stat().st_size,
               "official_sha256": expected_sha, "actual_sha256": actual_sha, "all_members_CRC32_verified": True,
               "members": members, "download_attempts": attempts,
               "numeric_deserialization_executed": False,
               "analysis_method": "Source files recovered and checksum verified. Any subsequent parser must avoid arbitrary pickle globals.",
               "elapsed_seconds": round(time.monotonic() - started, 2)}
    (destination / "SOURCE_RECOVERY_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "members": len(members), "archive_sha256": actual_sha,
                      "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
