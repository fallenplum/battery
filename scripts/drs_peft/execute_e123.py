"""Relocate active file locks while executing the frozen E1--E3 driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = ROOT / "scripts/drs_peft/run_e123.py"
FROZEN_DRIVER_SHA256 = (
    "5e30fd70377480459a4f79771160093ab3621b94b0af589c25ee23a8669e6fef"
)
LOCK_ROOT = ROOT / "runtime/drs_peft/e123_v1/_locks"


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, path)


def mapped_lock_path(requested, lock_root=LOCK_ROOT):
    identity = os.path.normcase(str(Path(requested).resolve()))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Path(lock_root) / (digest + ".lock")


@contextmanager
def relocated_locks(driver, lock_root=LOCK_ROOT, ledger=None, lock_class=FileLock):
    """Preserve FileLock options and restore the original driver binding."""
    lock_root = Path(lock_root)
    lock_root.mkdir(parents=True, exist_ok=True)
    previous = driver.FileLock

    def relocated(requested, *args, **kwargs):
        target = mapped_lock_path(requested, lock_root)
        if ledger is not None:
            ledger.append(
                {
                    "requested": str(Path(requested).resolve()),
                    "actual": str(target.resolve()),
                }
            )
        return lock_class(str(target), *args, **kwargs)

    driver.FileLock = relocated
    try:
        yield
    finally:
        driver.FileLock = previous


def state_snapshots(private_root, fold_ids, seeds):
    """Record optimizer counters and original source identities on CPU."""
    import torch

    records = {}
    for fold_id in fold_ids:
        for seed in seeds:
            folder = Path(private_root) / f"fold{fold_id}" / f"seed{seed}"
            for state_path in sorted(folder.rglob("TRAIN_STATE.pt")):
                unit = state_path.parent
                config = read_json(unit / "RUN_CONFIG.json")
                driver_records = [
                    item
                    for item in config["source_snapshot"]
                    if item["path"] == "scripts/drs_peft/run_e123.py"
                ]
                if (
                    len(driver_records) != 1
                    or driver_records[0]["sha256"] != FROZEN_DRIVER_SHA256
                ):
                    raise RuntimeError(
                        "Existing training state belongs to another driver version"
                    )
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                if state["identity"] != config:
                    raise RuntimeError(
                        "Existing optimizer state and source execution identity differ"
                    )
                steps = [
                    float(value["step"])
                    for value in state["optimizer"]["state"].values()
                    if "step" in value
                ]
                records[str(unit.resolve())] = {
                    "updates": int(state["updates"]),
                    "next_epoch": int(state["next_epoch"]),
                    "history_rows": len(state["history"]),
                    "epochs": int(config["epochs"]),
                    "optimizer_step_min": min(steps) if steps else None,
                    "optimizer_step_max": max(steps) if steps else None,
                    "source_commit": config["source_commit"],
                    "was_epoch_complete": int(state["next_epoch"])
                    >= int(config["epochs"]),
                    "original_files": {
                        name: file_hash(unit / name)
                        for name in (
                            "RUN_CONFIG.json",
                            "SOURCE_SNAPSHOT.json",
                            "TRAIN_STATE.pt",
                        )
                    },
                }
                del state
    return records


def compare_snapshots(before, after):
    results = []
    for unit, original in before.items():
        current = after[unit]
        if (
            original["original_files"]["RUN_CONFIG.json"]
            != current["original_files"]["RUN_CONFIG.json"]
        ):
            raise RuntimeError("Lock relocation changed an original RUN_CONFIG")
        if (
            original["original_files"]["SOURCE_SNAPSHOT.json"]
            != current["original_files"]["SOURCE_SNAPSHOT.json"]
        ):
            raise RuntimeError("Lock relocation changed an original SOURCE_SNAPSHOT")
        added = current["updates"] - original["updates"]
        if original["was_epoch_complete"] and (
            added != 0 or current["original_files"] != original["original_files"]
        ):
            raise RuntimeError("Finalization replay changed completed optimizer state")
        results.append(
            {
                "unit": unit,
                "new_optimizer_updates": added,
                "epoch_complete_before_invocation": original["was_epoch_complete"],
                "updates_before": original["updates"],
                "updates_after": current["updates"],
                "completed_optimizer_state_preserved": original["was_epoch_complete"]
                and added == 0,
            }
        )
    return results


def launcher_arguments(arguments):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/e123_v1"
    )
    parser.add_argument(
        "--public-root", type=Path, default=ROOT / "results/drs_peft/e123_v1"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/drs_peft/e123_v1.json"
    )
    parser.add_argument("--fold-ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    args, _ = parser.parse_known_args(arguments)
    return args


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if file_hash(DRIVER_PATH) != FROZEN_DRIVER_SHA256:
        raise RuntimeError("The frozen E1--E3 driver source changed")
    from scripts.drs_peft import run_e123 as driver

    if "--help" in arguments or "-h" in arguments or "--dry-run" in arguments:
        previous = sys.argv
        try:
            sys.argv = [str(DRIVER_PATH), *arguments]
            driver.main()
        finally:
            sys.argv = previous
        return
    args = launcher_arguments(arguments)
    amendment = {
        "amendment_id": "E123_WINDOWS_LOCK_LOCATION_20261117_v1",
        "driver": {
            "path": "scripts/drs_peft/run_e123.py",
            "sha256": FROZEN_DRIVER_SHA256,
        },
        "launcher": {
            "path": "scripts/drs_peft/execute_e123.py",
            "sha256": file_hash(__file__),
        },
        "trainer": {
            "path": "scripts/drs_peft/train.py",
            "sha256": file_hash(ROOT / "scripts/drs_peft/train.py"),
        },
        "configuration_sha256": file_hash(args.config),
        "lock_root": str(LOCK_ROOT),
        "lock_mapping": "sha256(normcase(resolve(requested_path))) followed by .lock",
        "reason": "Windows exclusive RUN.lock prevented unit receipt hashing",
        "training_operator_changes": 0,
        "source_commit_environment": os.environ.get("BATTERY_SOURCE_COMMIT"),
        "final_target_query_labels_used": 0,
    }
    path = args.public_root / "IO_LOCK_AMENDMENT.json"
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(mapped_lock_path(path, LOCK_ROOT))):
        if path.exists():
            if read_json(path) != amendment:
                raise RuntimeError("Existing lock-location amendment identity changed")
        else:
            write_json(path, amendment)
    before = state_snapshots(args.private_root, args.fold_ids, args.seeds)
    invocation = (
        args.public_root / "io_invocations" / f"{time.time_ns()}_{os.getpid()}.json"
    )
    record = {
        "status": "RUNNING",
        "amendment_sha256": file_hash(path),
        "arguments": arguments,
        "before_states": before,
        "requested_lock_mappings": [],
    }
    write_json(invocation, record)
    previous = sys.argv
    try:
        sys.argv = [str(DRIVER_PATH), *arguments]
        with relocated_locks(driver, ledger=record["requested_lock_mappings"]):
            driver.main()
        after = state_snapshots(args.private_root, args.fold_ids, args.seeds)
        record.update(
            status="COMPLETE",
            after_states=after,
            resumed_state_checks=compare_snapshots(before, after),
            new_optimizer_updates=sum(item["updates"] for item in after.values())
            - sum(item["updates"] for item in before.values()),
        )
    except BaseException as error:
        record.update(
            status="FAILED", failure_type=type(error).__name__, failure=str(error)
        )
        raise
    finally:
        sys.argv = previous
        write_json(invocation, record)
    print(
        json.dumps(
            {
                "event": "E123_IO_LAUNCHER_COMPLETE",
                "invocation": str(invocation),
                "new_optimizer_updates": record["new_optimizer_updates"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
