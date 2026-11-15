"""Hash the current outer archive and CRC-check its full inner ZIP stream."""

import hashlib
import json
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path(r"D:\develop\patchtst\BatteryLife\dataset\hust_raw.zip")
CACHE = REPO.parent / "private" / "raw_sources" / "HUST" / "our_data.zip"
OUT = REPO / "results" / "hust_raw_idea_audit_v2"


def hash_stream(stream):
    digest = hashlib.sha256()
    for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(data)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    before = SOURCE.stat()
    with SOURCE.open("rb") as stream:
        outer_hash = hash_stream(stream)
    with zipfile.ZipFile(SOURCE) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != "nsc7hnsg4s-2/our_data.zip":
            raise ValueError("Current HUST wrapper membership changed")
        member = members[0]
        with archive.open(member) as stream:
            inner_hash = hash_stream(stream)
    with CACHE.open("rb") as stream:
        cache_hash = hash_stream(stream)
    receipt = json.loads((CACHE.parent / "SOURCE_RECOVERY_RECEIPT.json").read_text(encoding="utf-8"))
    official_hash = receipt["official_sha256"]
    after = SOURCE.stat()
    unchanged = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    if not unchanged or inner_hash != cache_hash or inner_hash != official_hash:
        raise ValueError("Current wrapper does not close to the authenticated official cache")
    result = {"outer_path": str(SOURCE), "outer_bytes": before.st_size, "outer_mtime_ns": before.st_mtime_ns,
        "outer_sha256": outer_hash, "outer_member_count": len(members), "inner_member_name": member.filename,
        "inner_bytes": member.file_size, "inner_crc32_verified": True, "inner_crc32": format(member.CRC, "08x"),
        "inner_sha256": inner_hash, "old_official_inner_sha256": official_hash, "inner_equals_old_official": inner_hash == official_hash,
        "replay_inner_cache_path": str(CACHE), "replay_cache_sha256": cache_hash, "replay_cache_matches_current_inner": cache_hash == inner_hash,
        "source_stat_unchanged": unchanged, "elapsed_s": time.monotonic() - started}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CURRENT_ARCHIVE_IDENTITY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
