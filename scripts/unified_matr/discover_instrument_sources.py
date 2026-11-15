"""Inspect official archive containers for original instrument-cycle records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

from recover_sources import SOURCES


BASE = "https://data.matr.io/1/api/v1/"
AUTHOR_SOURCE = "https://raw.githubusercontent.com/chueh-ermon/BMS-autoanalysis/master/cell_analysis.m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--routes", nargs="+")
    parser.add_argument("--author-commit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.author_commit:
        url = "https://raw.githubusercontent.com/chueh-ermon/BMS-autoanalysis/" + args.author_commit + "/cell_analysis.m"
        response = requests.get(url, timeout=(20, 60))
        response.raise_for_status()
        path = args.output / ("author_cell_analysis_" + args.author_commit + ".m")
        path.write_bytes(response.content)
        receipt = {"url": url, "commit": args.author_commit, "local_file": path.name,
                   "bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest(),
                   "source_lines": {"instrument_cycle_input": 27, "cycle_selection": [78, 82],
                                    "native_time_to_minutes": 91, "current_normalization": 92,
                                    "author_generated_summary_sequence": 154}}
        (args.output / "AUTHOR_SOURCE_PROOF.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt), flush=True)
        return
    if args.routes:
        destination = args.output / "PUBLIC_ROUTE_DISCOVERY.json"
        records = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else []
        with requests.Session() as session:
            for route in args.routes:
                url = route if route.startswith("https://") else BASE + route
                response = session.get(url, timeout=(20, 60))
                record = {"url": response.url, "requested_url": url, "http_status": response.status_code}
                try:
                    payload = response.json()
                    record["payload"] = payload
                except ValueError:
                    payload = None
                    record["text_prefix"] = response.text[:1000]
                records.append(record)
                print(json.dumps({"requested_url": url, "final_url": response.url,
                                  "http_status": response.status_code, "payload_type": type(payload).__name__,
                                  "keys_or_count": list(payload) if isinstance(payload, dict) else len(payload) if isinstance(payload, list) else None}), flush=True)
        destination.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if args.frontend_only:
        response = requests.get("https://data.matr.io/1/", timeout=(20, 60))
        response.raise_for_status()
        (args.output / "official_app.html").write_bytes(response.content)
        scripts = re.findall(r'<script[^>]*src=["\']([^"\']+)', response.text)
        frontend = []
        for source in scripts:
            url = urljoin(response.url, source)
            content = requests.get(url, timeout=(20, 60))
            content.raise_for_status()
            name = "official_" + url.rsplit("/", 1)[1]
            (args.output / name).write_bytes(content.content)
            frontend.append({"url": url, "local_file": name, "bytes": len(content.content),
                             "sha256": hashlib.sha256(content.content).hexdigest()})
        (args.output / "OFFICIAL_FRONTEND_SOURCES.json").write_text(json.dumps(frontend, indent=2), encoding="utf-8")
        print(json.dumps(frontend), flush=True)
        return
    records, folders = [], {}
    with requests.Session() as session:
        def get(route):
            url = route if route.startswith("https://") else BASE + route
            response = session.get(url, timeout=(20, 60))
            record = {"url": response.url, "http_status": response.status_code}
            try:
                payload = response.json()
                record["payload"] = payload
            except ValueError:
                payload = None
                record["content_type"] = response.headers.get("Content-Type")
                record["text_prefix"] = response.text[:1000]
            records.append(record)
            (args.output / "INSTRUMENT_SOURCE_DISCOVERY.partial.json").write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"url": response.url, "http_status": response.status_code,
                              "payload_type": type(payload).__name__}), flush=True)
            return payload

        for _, _, identifier, _ in SOURCES:
            file = get("file/" + identifier)
            item = get("item/" + file["itemId"])
            if not isinstance(item, dict):
                item = get("https://data.matr.io/api/v1/item/" + file["itemId"])
            if not isinstance(item, dict):
                continue
            get("item/" + item["_id"] + "/files?limit=0")
            folders[item["folderId"]] = 0
        for depth in range(4):
            current = [identifier for identifier, level in list(folders.items()) if level == depth]
            for identifier in current:
                folder = get("folder/" + identifier)
                get("item?folderId=" + identifier + "&limit=0")
                get("folder?parentType=folder&parentId=" + identifier + "&limit=0")
                if isinstance(folder, dict) and folder.get("parentCollection") == "folder":
                    parent = folder["parentId"]
                    if parent not in folders:
                        folders[parent] = depth + 1
        response = session.get(AUTHOR_SOURCE, timeout=(20, 60))
        response.raise_for_status()
        author_path = args.output / "author_cell_analysis.m"
        author_path.write_bytes(response.content)
    result = {"records": records, "author_source": {"url": AUTHOR_SOURCE,
              "local_file": author_path.name, "bytes": len(response.content),
              "sha256": hashlib.sha256(response.content).hexdigest()},
              "scope": "Four original official file containers and their parent-folder listings, plus original author parser"}
    destination = args.output / "INSTRUMENT_SOURCE_DISCOVERY.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for record in records:
        payload = record.get("payload")
        if isinstance(payload, list):
            print(json.dumps({"url": record["url"], "entries": [{key: row.get(key) for key in ("_id", "name", "size", "folderId", "parentId")} for row in payload]}, ensure_ascii=False))
        elif isinstance(payload, dict):
            print(json.dumps({"url": record["url"], **{key: payload.get(key) for key in ("name", "size", "folderId", "parentId", "parentCollection")}}, ensure_ascii=False))
    print(json.dumps({"author_source": result["author_source"]}))


if __name__ == "__main__":
    main()
