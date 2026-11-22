"""Construct matched physical-history and self-reference views from prefix data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from scipy.signal import savgol_filter

from models.drs_peft.bridges import FEATURE_NAMES, cycle_statistics
from scripts.drs_peft.spectral_rotation_features import CHANNELS, _row_selection
from scripts.drs_peft.transfer_scope_data import (
    _cell_ids,
    _registered_rows,
    _verified_archive,
)
from scripts.unified_matr.physics_features import GLOBAL_COLUMNS, LOCAL_COLUMNS

REFERENCE_LAST_CYCLE = 10
MIN_REFERENCE_CURVES = 5
COLUMNS = (*FEATURE_NAMES, *CHANNELS)
_RESIDUAL = LOCAL_COLUMNS.index("residual_signed_mean_Ah")
_LOCAL_DRIFT = LOCAL_COLUMNS.index("delta_q_signed_mean_Ah")
_GLOBAL_CAPACITY_OFFSET = GLOBAL_COLUMNS.index("a_Ah")
_GLOBAL_VOLTAGE_SHIFT = GLOBAL_COLUMNS.index("b_V")


@dataclass(frozen=True)
class PairedPrefixes:
    ids: tuple[str, ...]
    real_features: np.ndarray
    reference_features: np.ndarray
    feature_mask: np.ndarray
    cycles: np.ndarray
    real_anchor: np.ndarray
    reference_anchor: np.ndarray
    audit: dict

    def rows(self, ids):
        return _registered_rows(self.ids, ids)


def _prefix_rows(domain, data_root, ids):
    path, manifest, content_hash = _verified_archive(domain, data_root)
    cells = manifest["physical_cells"]
    with np.load(path, allow_pickle=False) as archive:
        voltage = archive["voltage_grid_V"]
        full_axis = _cell_ids(archive, cells)
    indices = _registered_rows(full_axis, ids)
    axis = (cells, 100)
    global_shape = (*axis, len(GLOBAL_COLUMNS))
    local_shape = (*axis, 8, len(LOCAL_COLUMNS))
    shapes = {
        "raw_viq": (*axis, 64, 3),
        "point_mask": (*axis, 64),
        "cycle_ids": axis,
        "nominal_capacity_Ah": (cells,),
        "anchor": (cells,),
        "drift_q": (*axis, 64),
        "drift_mask": (*axis, 64),
        "global_features": global_shape,
        "global_mask": global_shape,
        "local_features": local_shape,
        "local_feature_mask": local_shape,
        "local_residual": (*axis, 64),
        "local_mask": (*axis, 64),
    }
    with ZipFile(path) as archive:
        fields = {
            name: _row_selection(archive, name, indices, shape)
            for name, shape in shapes.items()
        }
    return fields, voltage, content_hash


def capacity_reference(raw_q, point_mask, cycles):
    """Use observed samples of the same physical coordinate in cycles 1–10."""
    q, mask, cycles = (
        np.asarray(raw_q, np.float64),
        np.asarray(point_mask),
        np.asarray(cycles),
    )
    if (
        q.ndim != 3
        or q.shape != mask.shape
        or cycles.shape != q.shape[:2]
        or mask.dtype != bool
        or not mask.any(axis=(1, 2)).all()
        or not np.isfinite(q[mask]).all()
    ):
        raise ValueError("Invalid masked electrical prefix axes")
    measured_cycles = cycles[mask.any(axis=-1)]
    if not np.all(
        (measured_cycles >= 1)
        & (measured_cycles <= 100)
        & (measured_cycles == np.floor(measured_cycles))
    ):
        raise ValueError("A physical prefix sample is outside cycles 1 through 100")
    for cycle_row, valid in zip(cycles, mask.any(axis=-1), strict=True):
        values = cycle_row[valid]
        if len(values) != len(np.unique(values)):
            raise ValueError("An original observed cycle ID is repeated")
    early = mask & ((cycles >= 1) & (cycles <= REFERENCE_LAST_CYCLE))[..., None]
    count = early.sum(axis=1)
    usable = mask.any(axis=1)
    if np.any(usable & (count < MIN_REFERENCE_CURVES)):
        raise ValueError(
            "A measured voltage coordinate has fewer than five early reference curves"
        )
    reference = np.zeros((len(q), q.shape[-1]), np.float64)
    for index, available in enumerate(usable):
        source = np.where(early[index], q[index], np.nan)
        reference[index, available] = np.nanmedian(source[:, available], axis=0)
    return reference, count


def paired_electrical_values(raw, mask, cycles, nominal):
    """Replace Q by its exact same-cell reference and rebuild its own drift."""
    native = np.asarray(raw)
    nominal = np.asarray(nominal, dtype=np.float32)
    if (
        native.shape != (*mask.shape, 3)
        or nominal.shape != (len(native),)
        or not np.isfinite(nominal).all()
        or (nominal <= 0).any()
        or not np.isfinite(native[mask]).all()
    ):
        raise ValueError(
            "Invalid voltage/current/capacity samples or declared nominal Ah"
        )
    reference, count = capacity_reference(native[..., 2], mask, cycles)
    exact_delta = native[..., 2].astype(np.float64) - reference[:, None, :]
    drift = np.where(mask, exact_delta, 0).astype(np.float32)
    real = np.concatenate((native.astype(np.float32), drift[..., None]), axis=-1)
    synthetic = real.copy()
    synthetic[..., 2] = reference[:, None, :]
    synthetic[..., 3] = 0.0
    for view in (real, synthetic):
        view[..., 1:4] /= nominal[:, None, None, None]
        view[~mask] = 0.0
    if not np.array_equal(real[..., :2], synthetic[..., :2]):
        raise AssertionError("The reference probe changed voltage or current")
    return real, synthetic, reference, count, drift


def _signed_channels(fields, voltage, reference, drift):
    g, gm = fields["global_features"], fields["global_mask"]
    local, lm = fields["local_features"], fields["local_feature_mask"]
    r, rm = fields["local_residual"], fields["local_mask"]
    mask = fields["point_mask"]
    if any(value.dtype != bool for value in (gm, lm, rm, mask)):
        raise ValueError("Original prefix feature masks must be Boolean")
    if not all(np.isfinite(v[m]).all() for v, m in ((g, gm), (local, lm), (r, rm))):
        raise ValueError("A supported native Q(V) coefficient is nonfinite")
    if (
        voltage.shape != (64,)
        or not np.isfinite(voltage).all()
        or not (np.diff(voltage) > 0).all()
    ):
        raise ValueError("Invalid frozen physical voltage coordinate")
    observed_voltage = np.broadcast_to(voltage, mask.shape)[mask]
    if not np.allclose(
        fields["raw_viq"][..., 0][mask], observed_voltage, atol=5e-7, rtol=0
    ):
        raise ValueError("Observed VIQ samples disagree with the original voltage axis")
    edges = np.linspace(voltage[0], voltage[-1], 9)
    bins = np.minimum(np.searchsorted(edges, voltage, side="right") - 1, 7)
    if set(bins) != set(range(8)):
        raise ValueError("Original loaded voltage segments differ")
    max_local_error = 0.0
    for b in range(8):
        points = bins == b
        supported = lm[..., b, _LOCAL_DRIFT]
        if not (mask[:, :, points].all(axis=-1) | ~supported).all():
            raise ValueError(
                "Cached signed Q(V) local features cover an incomplete voltage segment"
            )
        recomputed = drift[:, :, points].mean(axis=-1, dtype=np.float64)
        difference = np.abs(
            recomputed[supported] - local[..., b, _LOCAL_DRIFT][supported]
        )
        max_local_error = max(max_local_error, float(np.max(difference, initial=0)))
    if max_local_error > 2e-7:
        raise ValueError(
            "Original signed Q(V) features use a different capacity reference"
        )
    a = g[..., _GLOBAL_CAPACITY_OFFSET]
    b = g[..., _GLOBAL_VOLTAGE_SHIFT]
    sample_valid = gm[..., _GLOBAL_CAPACITY_OFFSET] & gm[..., _GLOBAL_VOLTAGE_SHIFT]
    reconstruction_support = rm & mask & sample_valid[..., None]
    derivative = np.gradient(
        savgol_filter(reference, 5, 2, axis=-1), voltage, axis=-1, edge_order=2
    )
    recomposed = a[..., None] + b[..., None] * derivative[:, None, :] + r
    max_reconstruction = float(
        np.max(
            np.abs(recomposed[reconstruction_support] - drift[reconstruction_support]),
            initial=0,
        )
    )
    if max_reconstruction > 2e-7:
        raise ValueError(
            "Original Q(V) decomposition does not reconstruct the same signed drift"
        )
    nominal = fields["nominal_capacity_Ah"][:, None]
    values = np.concatenate(
        (
            (a / nominal)[..., None],
            b[..., None],
            local[..., _RESIDUAL] / nominal[..., None],
            local[..., _LOCAL_DRIFT] / nominal[..., None],
        ),
        axis=-1,
    )
    support = np.concatenate(
        (
            gm[..., _GLOBAL_CAPACITY_OFFSET, None],
            gm[..., _GLOBAL_VOLTAGE_SHIFT, None],
            lm[..., _RESIDUAL],
            lm[..., _LOCAL_DRIFT],
        ),
        axis=-1,
    )
    if np.any(support & ~mask.any(axis=-1)[..., None]):
        raise ValueError(
            "A signed feature exists on a wholly unobserved electrical cycle"
        )
    return (
        np.where(support, values, 0.0).astype(np.float32),
        support,
        max_local_error,
        max_reconstruction,
    )


def _fingerprint(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def build_self_reference_prefixes(domain: str, data_root: Path, ids) -> PairedPrefixes:
    """Build identical supports and token-ready [cells, 100, 30] view arrays."""
    names = tuple(ids)
    if not names:
        raise ValueError("A paired prefix needs a registered physical cell roster")
    fields, voltage, manifest = _prefix_rows(domain, data_root, names)
    cycles, mask = fields["cycle_ids"], fields["point_mask"]
    if (
        mask.dtype != bool
        or fields["drift_mask"].dtype != bool
        or not np.array_equal(fields["drift_mask"], mask)
    ):
        raise ValueError(
            "Paired viable voltage points must have the original legal drift reference coverage"
        )
    real_x, ref_x, qref, reference_count, drift = paired_electrical_values(
        fields["raw_viq"], mask, cycles, fields["nominal_capacity_Ah"]
    )
    if not np.array_equal(drift[mask], fields["drift_q"][mask]):
        raise ValueError(
            "Recomputed early drift differs from the registered native input"
        )
    anchor = fields["anchor"].astype(np.float32)
    if not np.isfinite(anchor).all():
        raise ValueError(
            "Original legal K100 anchor differs from the paired prefix roster"
        )
    signed, signed_mask, local_error, reconstruction_error = _signed_channels(
        fields, voltage, qref, drift
    )
    point_mask = torch.from_numpy(mask)
    with torch.no_grad():
        real_old, old_mask = cycle_statistics(torch.from_numpy(real_x), point_mask)
        ref_old, ref_mask = cycle_statistics(torch.from_numpy(ref_x), point_mask)
    if not torch.equal(old_mask, ref_mask) or not torch.equal(
        real_old[..., [0, 1, 4, 5]], ref_old[..., [0, 1, 4, 5]]
    ):
        raise AssertionError(
            "Same voltage/current coordinates yield different native electrical context"
        )
    real = np.concatenate((real_old.numpy(), signed), axis=-1)
    reference = np.concatenate((ref_old.numpy(), np.zeros_like(signed)), axis=-1)
    common = np.concatenate((old_mask.numpy(), signed_mask), axis=-1)
    observed_cycle = mask.any(axis=-1)
    live_cycles = np.where(observed_cycle, cycles, 0).astype(np.int64)
    audit = {
        "domain": domain,
        "cache_manifest_sha256": manifest,
        "physical_uids": list(names),
        "source_fit_only_preprocessing": "Cell-wise measurements only. No population parameters are fitted.",
        "reference_rule": "Pointwise median in original cycles 1–10, at least 5 measured curves per supported voltage.",
        "native_voltage_range_V": [float(voltage[0]), float(voltage[-1])],
        "native_signed_features": "Frozen native-grid Q(V) decomposition, distinct from the later common-grid 54-dimensional summaries.",
        "capacity_scaling": "Declared nominal Ah. Global derivative coefficient retains volts.",
        "architecture_input_columns": list(COLUMNS),
        "matching_input_support": True,
        "matching_voltage_current_measurements": True,
        "reference_signed_channels_exact_zero_on_common_support": True,
        "physical_context_anchor_rule": "Same measured K100 full-discharge anchor in both views. The substituted local Q window does not determine a full-discharge reference anchor.",
        "supported_native_positions": int(mask.sum()),
        "supported_tokens_by_channel": common.sum(axis=(0, 1)).tolist(),
        "minimum_reference_measurements": int(reference_count[mask.any(axis=1)].min()),
        "max_native_segment_delta_q_discrepancy_Ah": local_error,
        "max_native_qv_reconstruction_discrepancy_Ah": reconstruction_error,
        "real_feature_sha256": _fingerprint(real),
        "reference_feature_sha256": _fingerprint(reference),
        "feature_mask_sha256": _fingerprint(common),
        "policy_fields_read": 0,
        "post_prefix_supervision_rows_read": 0,
    }
    return PairedPrefixes(
        names, real, reference, common, live_cycles, anchor, anchor.copy(), audit
    )
