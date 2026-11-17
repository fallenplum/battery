"""Check the V2 label-access boundary and immutable cache identity."""

import json
from zipfile import ZipFile

import numpy as np
import pytest

from scripts.drs_peft import transfer_scope_data as v2


@pytest.fixture
def frozen_cache(tmp_path, monkeypatch):
    folder = tmp_path / "MATR"
    folder.mkdir()
    ids = ["one", "two", "three", "four"]
    targets = np.arange(4 * 4900, dtype=np.float32).reshape(4, 4900)
    observed = targets % 3 == 1
    mask = np.ones((4, 100, 2), dtype=bool)
    raw = np.ones((4, 100, 2, 3), dtype=np.float32)
    archive = folder / "arrays.npz"
    np.savez_compressed(
        archive, ids=ids, query=np.arange(101, 5001, dtype=np.float32),
        anchor=np.ones(4, dtype=np.float32), nominal_capacity_Ah=np.full(4, 2.0),
        raw_viq=raw, point_mask=mask, cycle_ids=np.tile(np.arange(1, 101), (4, 1)),
        drift_q=raw[..., 0], drift_mask=mask, targets=targets, observed=observed,
        policy=np.ones((4, 2), dtype=np.float32),
    )
    marker = folder / "COMPLETE.json"
    marker.write_text(json.dumps({
        "status": "QUALIFIED_FROZEN_DATA", "domain": "MATR", "physical_cells": 4,
        "prefix_shape": list(raw.shape), "future_shape": list(targets.shape),
        "files": [{"path": archive.name, "sha256": v2.file_hash(archive)}],
    }), encoding="utf-8")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "data_caches": {"MATR": {"manifest_sha256": v2.file_hash(marker)}},
    }), encoding="utf-8")
    monkeypatch.setattr(v2, "SPLIT_LOCK", lock)
    return tmp_path, ids, targets, observed


def test_prefix_and_mask_only_do_not_open_label_member(frozen_cache, monkeypatch):
    root, ids, _, expected = frozen_cache
    opened = []
    open_zip = ZipFile.open

    def record(archive, member, *args, **kwargs):
        opened.append(getattr(member, "filename", member))
        return open_zip(archive, member, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", record)
    prefix = v2.load_prefixes("MATR", root)
    assert prefix.ids == ids
    assert prefix.inputs["ids"] == ids
    assert prefix.inputs["policy"].shape == prefix.inputs["policy_mask"].shape == (4, 0)
    assert not {"targets.npy", "observed.npy", "policy.npy"}.intersection(opened)
    np.testing.assert_allclose(prefix.inputs["x"][0, 0, 0], [1.0, 0.5, 0.5, 0.5])
    chosen = [ids[3], ids[1]]
    no_labels, observed = v2.load_future_rows("MATR", root, chosen, load_targets=False)
    assert no_labels is None
    assert "observed.npy" in opened and "targets.npy" not in opened
    np.testing.assert_array_equal(prefix.rows(chosen), [3, 1])
    np.testing.assert_array_equal(observed, expected[[3, 1]])


def test_future_reader_preserves_selected_order_and_rejects_bad_ids(frozen_cache):
    root, ids, expected_targets, expected_mask = frozen_cache
    targets, masks = v2.load_future_rows("MATR", root, [ids[3], ids[0]])
    assert targets.shape == masks.shape == (2, 4900)
    np.testing.assert_array_equal(targets, expected_targets[[3, 0]])
    np.testing.assert_array_equal(masks, expected_mask[[3, 0]])
    with pytest.raises(ValueError, match="distinct physical cells"):
        v2.load_future_rows("MATR", root, [ids[0], ids[0]])
    with pytest.raises(ValueError, match="outside the frozen data cache"):
        v2.load_future_rows("MATR", root, ["unknown"])


def test_tampered_cache_and_completion_marker_fail_closed(frozen_cache):
    root, ids, _, _ = frozen_cache
    with (root / "MATR/arrays.npz").open("ab") as stream:
        stream.write(b"changed archive")
    with pytest.raises(ValueError, match="Frozen data cache changed"):
        v2.load_future_rows("MATR", root, [ids[0]])
    with (root / "MATR/COMPLETE.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="COMPLETE identity"):
        v2.load_prefixes("MATR", root)
