"""Regression tests for external locks and completed-state finalization."""

from types import SimpleNamespace

import pytest

from scripts.drs_peft import execute_e123


def test_lock_mapping_is_absolute_path_stable_and_collision_separated(tmp_path):
    root = tmp_path / "locks"
    one = tmp_path / "one" / "RUN.lock"
    assert execute_e123.mapped_lock_path(one, root) == execute_e123.mapped_lock_path(
        one.parent / "." / one.name, root
    )
    assert execute_e123.mapped_lock_path(one, root) != execute_e123.mapped_lock_path(
        tmp_path / "two" / "RUN.lock", root
    )
    assert execute_e123.mapped_lock_path(one, root).parent == root


def test_active_windows_exclusive_lock_stays_outside_receipt_scan(tmp_path):
    locked = set()

    class WindowsExclusiveLock:
        def __init__(self, path, **options):
            self.path = path
            self.options = options

        def __enter__(self):
            from pathlib import Path

            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_bytes(b"held")
            locked.add(self.path)
            return self

        def __exit__(self, *error):
            locked.remove(self.path)

    out = tmp_path / "unit"
    out.mkdir()
    residual = out / "RUN.lock"
    residual.write_bytes(b"old_unlocked_residual")
    (out / "RESULT.json").write_bytes(b"{}")
    before = residual.read_bytes()
    driver = SimpleNamespace(FileLock=WindowsExclusiveLock)
    original = driver.FileLock
    ledger = []
    with (
        execute_e123.relocated_locks(
            driver, tmp_path / "_locks", ledger, WindowsExclusiveLock
        ),
        driver.FileLock(str(residual), timeout=3) as active,
    ):
        assert active.options == {"timeout": 3}
        for path in out.iterdir():
            if str(path) in locked:
                raise PermissionError("Windows exclusive file cannot be hashed")
            execute_e123.file_hash(path)
        assert residual.read_bytes() == before
        assert active.path not in [str(path) for path in out.iterdir()]
    assert driver.FileLock is original
    assert len(ledger) == 1


def test_driver_lock_binding_is_restored_after_failure(tmp_path):
    original = object()
    driver = SimpleNamespace(FileLock=original)
    with (
        pytest.raises(RuntimeError, match="fixture"),
        execute_e123.relocated_locks(driver, tmp_path),
    ):
        raise RuntimeError("fixture")
    assert driver.FileLock is original


def test_completed_optimizer_state_requires_zero_new_updates_and_original_bytes():
    before = {
        "unit": {
            "updates": 960,
            "was_epoch_complete": True,
            "original_files": {
                "RUN_CONFIG.json": "a",
                "SOURCE_SNAPSHOT.json": "b",
                "TRAIN_STATE.pt": "c",
            },
        }
    }
    after = {
        "unit": {
            "updates": 960,
            "original_files": dict(before["unit"]["original_files"]),
        }
    }
    rows = execute_e123.compare_snapshots(before, after)
    assert rows[0]["new_optimizer_updates"] == 0
    assert rows[0]["completed_optimizer_state_preserved"] is True
    after["unit"]["updates"] = 961
    with pytest.raises(RuntimeError, match="completed optimizer"):
        execute_e123.compare_snapshots(before, after)


def test_incomplete_state_can_resume_without_source_identity_change():
    before = {
        "unit": {
            "updates": 16,
            "was_epoch_complete": False,
            "original_files": {
                "RUN_CONFIG.json": "a",
                "SOURCE_SNAPSHOT.json": "b",
                "TRAIN_STATE.pt": "c",
            },
        }
    }
    after = {
        "unit": {
            "updates": 960,
            "original_files": {
                "RUN_CONFIG.json": "a",
                "SOURCE_SNAPSHOT.json": "b",
                "TRAIN_STATE.pt": "new",
            },
        }
    }
    assert (
        execute_e123.compare_snapshots(before, after)[0]["new_optimizer_updates"] == 944
    )
    after["unit"]["original_files"]["RUN_CONFIG.json"] = "changed"
    with pytest.raises(RuntimeError, match="RUN_CONFIG"):
        execute_e123.compare_snapshots(before, after)
