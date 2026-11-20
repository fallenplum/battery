"""Signed loaded-response descriptors on a shared, observed physical voltage grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from numpy.lib import format as npy_format

from scripts.drs_peft.transfer_scope_data import _cell_ids, _registered_rows, _verified_archive
from scripts.unified_matr.physics_features import LOCAL_COLUMNS, PhysicsConfig, extract_prefix_features


PHYSICS = PhysicsConfig(voltage_min=2.8, voltage_max=3.1)
VOLTAGE_GRID_V = np.linspace(PHYSICS.voltage_min, PHYSICS.voltage_max, PHYSICS.grid_size)
EARLY = (11, 30)
LATE = (81, 100)
MIN_BAND_CYCLES = 5
CHANNELS = (
    "global_capacity_offset_over_nominal",
    "global_derivative_coefficient_V",
    *(f"residual_signed_over_nominal_bin_{index + 1}" for index in range(PHYSICS.bins)),
    *(f"delta_q_signed_over_nominal_bin_{index + 1}" for index in range(PHYSICS.bins)),
)
STATISTICS = ("late_median", "late_minus_early_median", "theil_sen_per_100_cycles")
COLUMNS = tuple(f"{channel}__{statistic}" for channel in CHANNELS for statistic in STATISTICS)
_RESIDUAL = LOCAL_COLUMNS.index("residual_signed_mean_Ah")
_DELTA_Q = LOCAL_COLUMNS.index("delta_q_signed_mean_Ah")


@dataclass(frozen=True)
class SignedDeformation:
    ids: tuple[str, ...]
    values: np.ndarray
    mask: np.ndarray
    columns: tuple[str, ...]
    audit: dict


def _row_selection(archive, field, rows, expected_shape):
    """Decode registered prefix rows without opening outcome members."""
    with archive.open(field + ".npy") as stream:
        version = npy_format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = npy_format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = npy_format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Invalid NumPy header for qualified prefix member {field}")
        if shape != expected_shape or fortran or dtype.kind not in "bfiu":
            raise ValueError(f"Wrong prefix row shape or type for {field}")
        offset = stream.tell()
        size = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        result = np.empty((len(rows), *shape[1:]), dtype=dtype)
        for destination in np.argsort(rows, kind="stable"):
            stream.seek(offset + int(rows[destination]) * size)
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError(f"Truncated prefix member {field}")
            result[destination] = np.frombuffer(payload, dtype=dtype).reshape(shape[1:])
    return result


def _load_prefix_rows(domain, data_root, ids):
    path, manifest, manifest_sha = _verified_archive(domain, data_root)
    n_cells = manifest["physical_cells"]
    with np.load(path, allow_pickle=False) as archive:
        registered = _cell_ids(archive, n_cells)
        native_grid = archive["voltage_grid_V"]
    rows = _registered_rows(registered, ids)
    with ZipFile(path) as archive:
        raw = _row_selection(archive, "raw_viq", rows, (n_cells, 100, 64, 3))
        point_mask = _row_selection(archive, "point_mask", rows, (n_cells, 100, 64))
        cycles = _row_selection(archive, "cycle_ids", rows, (n_cells, 100))
        nominal = _row_selection(archive, "nominal_capacity_Ah", rows, (n_cells,))
    if point_mask.dtype != bool:
        raise ValueError("Qualified point mask must be Boolean")
    return raw, point_mask, cycles, nominal, native_grid, manifest_sha


def _shared_grid_curves(raw, point_mask, native_grid):
    native = np.asarray(native_grid, dtype=np.float64)
    if native.shape != (64,) or not np.isfinite(native).all() or np.any(np.diff(native) <= 0):
        raise ValueError("Invalid native loaded-response voltage coordinate")
    tolerance = 5e-7
    if native[0] > VOLTAGE_GRID_V[0] + tolerance or native[-1] < VOLTAGE_GRID_V[-1] - tolerance:
        raise ValueError("Registered shared voltage grid exceeds native measurement support")
    observed_v = np.broadcast_to(native, point_mask.shape)[point_mask]
    if not np.allclose(raw[..., 0][point_mask], observed_v, atol=tolerance, rtol=0):
        raise ValueError("Observed voltage rows disagree with the registered native grid")
    if not np.isfinite(raw[..., 2][point_mask]).all():
        raise ValueError("Supported loaded Q(V) values must be finite")
    left = max(0, np.searchsorted(native, VOLTAGE_GRID_V[0] + tolerance, side="right") - 1)
    right = min(len(native) - 1, np.searchsorted(native, VOLTAGE_GRID_V[-1] - tolerance, side="left"))
    selection = slice(left, right + 1)
    qualified = point_mask[:, selection].all(axis=1)
    q = np.full((len(raw), len(VOLTAGE_GRID_V)), np.nan, dtype=np.float64)
    if np.allclose(native, VOLTAGE_GRID_V, atol=tolerance, rtol=0):
        q[qualified] = raw[qualified, :, 2]
    else:
        # Boundary clipping only reconciles the same physical endpoint as stored in float32.
        voltage = np.clip(VOLTAGE_GRID_V, native[left], native[right])
        for index in np.flatnonzero(qualified):
            q[index] = np.interp(voltage, native[selection], raw[index, selection, 2])
    return q, qualified


def summarize_signed_channels(values, cycles):
    """Keep sign in both band changes and robust slopes on original cycle coordinates."""
    series = np.asarray(values, dtype=np.float64)
    time = np.asarray(cycles, dtype=np.float64)
    if series.ndim != 2 or series.shape[0] != len(time) or time.ndim != 1:
        raise ValueError("Expected per-cycle signed response rows")
    prefix = (time >= EARLY[0]) & (time <= LATE[1])
    series, time = series[prefix], time[prefix]
    if np.any(time != np.floor(time)) or len(np.unique(time)) != len(time):
        raise ValueError("Signed summaries require unique integer physical cycle IDs")
    output = np.full((series.shape[1], len(STATISTICS)), np.nan)
    for index, values in enumerate(series.T):
        present = np.isfinite(values)
        early = present & (time <= EARLY[1])
        late = present & (time >= LATE[0])
        if late.sum() >= MIN_BAND_CYCLES:
            output[index, 0] = np.median(values[late])
            if early.sum() >= MIN_BAND_CYCLES:
                output[index, 1] = output[index, 0] - np.median(values[early])
        if present.sum() >= MIN_BAND_CYCLES:
            x, y = time[present], values[present]
            left, right = np.triu_indices(len(x), 1)
            output[index, 2] = 100 * np.median((y[right] - y[left]) / (x[right] - x[left]))
    return output.ravel()


def _describe_cell(raw, mask, cycles, nominal, native_grid):
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError("A signed capacity coefficient requires declared nominal Ah")
    if not np.isfinite(cycles).all() or np.any((cycles < 1) | (cycles > 100)):
        raise ValueError("Frozen prefix exceeds registered original cycles 1 through 100")
    q, qualified = _shared_grid_curves(raw, mask, native_grid)
    result = extract_prefix_features(q, qualified, cycles, VOLTAGE_GRID_V, PHYSICS)
    global_columns = result["global_features"][:, :2].copy()
    global_columns[:, 0] /= nominal
    local = result["local_features"]
    signed = np.column_stack((global_columns, local[:, :, _RESIDUAL] / nominal, local[:, :, _DELTA_Q] / nominal))
    summary = summarize_signed_channels(signed, result["original_cycle_id"])
    residual = result["diagnostics"]["reconstruction_max_absolute_error_Ah"]
    reconstruction_max = float(np.nanmax(residual)) if np.isfinite(residual).any() else None
    audit = {
        "eligible_Q_V_curves": int(qualified.sum()),
        "decomposed_curves": int(result["decomposition_valid"].sum()),
        "independent_reference_curves": len(result["reference_cycles"]),
        "reconstruction_max_abs_Ah": reconstruction_max,
    }
    return summary, audit


def load_signed_deformation(domain: str, data_root: Path, ids) -> SignedDeformation:
    """Read only requested UIDs' signed electrical prefixes on 2.8–3.1 V."""
    names = tuple(ids)
    if not names:
        raise ValueError("Signed descriptors require registered physical UIDs")
    raw, mask, cycles, nominal, native, manifest_sha = _load_prefix_rows(domain, data_root, names)
    values, rows = [], {}
    for index, uid in enumerate(names):
        row, details = _describe_cell(raw[index], mask[index], cycles[index], nominal[index], native)
        values.append(row)
        rows[uid] = details
    values = np.stack(values)
    measured = np.isfinite(values)
    standard_deviation = []
    for values_column, observed in zip(values.T, measured.T, strict=True):
        standard_deviation.append(float(values_column[observed].std()) if observed.any() else None)
    audit = {
        "domain": domain,
        "cache_manifest_sha256": manifest_sha,
        "native_voltage_range_V": [float(native[0]), float(native[-1])],
        "shared_voltage_range_V": [PHYSICS.voltage_min, PHYSICS.voltage_max],
        "voltage_bins_V": np.linspace(PHYSICS.voltage_min, PHYSICS.voltage_max, PHYSICS.bins + 1).tolist(),
        "voltage_shifts_use_signed_reference_derivative_basis": True,
        "capacity_units": "instrument discharge Ah divided by declared nominal Ah",
        "physical_scope": "Loaded-response Q(V) on the qualified native discharge phase",
        "early_original_cycles": list(EARLY),
        "late_original_cycles": list(LATE),
        "reference_original_cycles": [1, PHYSICS.reference_max_cycle],
        "minimum_valid_cycles_per_band": MIN_BAND_CYCLES,
        "post_prefix_supervision_rows_read": 0,
        "policy_fields_read": 0,
        "per_cell": rows,
        "finite_cells_by_column": dict(zip(COLUMNS, measured.sum(axis=0).tolist(), strict=True)),
        "measured_standard_deviation_by_column": dict(zip(COLUMNS, standard_deviation, strict=True)),
        "positive_cells_by_column": dict(zip(COLUMNS, (values > 0).sum(axis=0).tolist(), strict=True)),
        "negative_cells_by_column": dict(zip(COLUMNS, (values < 0).sum(axis=0).tolist(), strict=True)),
    }
    return SignedDeformation(names, values, measured, COLUMNS, audit)
