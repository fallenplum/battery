"""Read registered V2 prefixes and allowlisted supervision rows independently."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from numpy.lib import format as npy_format

from scripts.drs_peft.data import ROOT, BatteryData, file_hash
from scripts.drs_peft.run_onesided_reference import NoPolicyInputs


SPLIT_LOCK = ROOT / "research/drs_peft/DATA_SPLIT_LOCK.json"
PREFIX_FIELDS = (
    "raw_viq", "point_mask", "cycle_ids", "drift_q", "drift_mask",
    "anchor", "nominal_capacity_Ah", "query",
)


@dataclass(frozen=True)
class PrefixData:
    domain: str
    ids: list[str]
    query: np.ndarray
    anchor: np.ndarray
    inputs: dict
    cache_manifest_sha256: str

    def rows(self, ids):
        return _registered_rows(self.ids, ids)


def _registered_rows(registered_ids, requested_ids):
    requested = list(requested_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("Requested rows must name distinct physical cells")
    lookup = {uid: index for index, uid in enumerate(registered_ids)}
    if not set(requested).issubset(lookup):
        raise ValueError("Requested rows include a UID outside the frozen data cache")
    return np.asarray([lookup[uid] for uid in requested], dtype=np.int64)


def _verified_archive(domain, data_root):
    if domain not in {"MATR", "HUST"}:
        raise ValueError("The V2 scope uses MATR and HUST")
    folder = Path(data_root) / domain
    manifest_path = folder / "COMPLETE.json"
    identity = file_hash(manifest_path)
    lock = json.loads(SPLIT_LOCK.read_text(encoding="utf-8"))["data_caches"][domain]
    if identity != lock["manifest_sha256"]:
        raise ValueError("Data COMPLETE identity differs from the frozen split lock")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["domain"] != domain or manifest["status"] != "QUALIFIED_FROZEN_DATA":
        raise ValueError("The qualified data manifest is inconsistent")
    for entry in manifest["files"]:
        if file_hash(folder / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen data cache changed: {entry['path']}")
    return folder / "arrays.npz", manifest, identity


def _cell_ids(archive, physical_cells):
    values = archive["ids"]
    if values.ndim != 1 or len(values) != physical_cells:
        raise ValueError("Cell identity axis differs from the manifest")
    ids = values.astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("The frozen data cache contains repeated physical UIDs")
    return ids


def load_prefixes(domain, data_root):
    """Load only registered early input fields, using canonical source units."""
    path, manifest, identity = _verified_archive(domain, data_root)
    with np.load(path, allow_pickle=False) as archive:
        ids = _cell_ids(archive, manifest["physical_cells"])
        fields = {name: archive[name] for name in PREFIX_FIELDS}
    if not np.array_equal(fields["query"], np.arange(101, 5001, dtype=np.float32)):
        raise ValueError("Prediction grid differs from the registered 101–5000 cycles")
    if list(fields["raw_viq"].shape) != manifest["prefix_shape"]:
        raise ValueError("Prefix shape differs from the registered cache manifest")
    data = BatteryData(
        domain=domain, ids=ids, **fields,
        **dict.fromkeys((
            "policy", "policy_mask", "targets", "observed", "voltage_grid_V",
            "protocol_ids", "physics", "cell_metadata", "provenance",
        )),
    )
    inputs = NoPolicyInputs().transform(data)
    inputs["ids"] = ids.copy()
    return PrefixData(domain, ids, data.query, data.anchor, inputs, identity)


def _read_npy_rows(archive, key, rows, expected_shape, dtype):
    with archive.open(key + ".npy") as stream:
        version = npy_format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, actual_dtype = npy_format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, actual_dtype = npy_format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported frozen NPY member format: {version}")
        if fortran_order or shape != expected_shape or actual_dtype != np.dtype(dtype):
            raise ValueError(f"Frozen member {key} has an incompatible row layout")
        offset = stream.tell()
        row_size = shape[1] * actual_dtype.itemsize
        values = np.empty((len(rows), shape[1]), dtype=actual_dtype)
        for destination in np.argsort(rows, kind="stable"):
            stream.seek(offset + int(rows[destination]) * row_size)
            payload = stream.read(row_size)
            if len(payload) != row_size:
                raise ValueError(f"Truncated row in frozen member {key}")
            values[destination] = np.frombuffer(payload, dtype=actual_dtype)
    return values


def load_future_rows(domain, data_root, ids, *, load_targets=True):
    """Decode only requested UID rows and leave targets unopened for mask-only reads."""
    path, manifest, _ = _verified_archive(domain, data_root)
    with np.load(path, allow_pickle=False) as archive:
        registered_ids = _cell_ids(archive, manifest["physical_cells"])
    rows = _registered_rows(registered_ids, ids)
    shape = tuple(manifest["future_shape"])
    if shape != (len(registered_ids), 4900):
        raise ValueError("Future row shape differs from the registered grid and IDs")
    with ZipFile(path) as archive:
        observed = _read_npy_rows(archive, "observed", rows, shape, bool)
        targets = (
            _read_npy_rows(archive, "targets", rows, shape, np.float32)
            if load_targets else None
        )
    if targets is not None and not np.isfinite(targets[observed]).all():
        raise ValueError("Requested observations contain nonfinite targets")
    return targets, observed
