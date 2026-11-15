"""Serialization, known-policy context and supplied-record identity matching."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np


POLICY_COLUMNS = ["charge_stage_count", "charge_rate_1_C", "charge_rate_2_C", "charge_rate_3_C", "charge_rate_4_C",
                  "charge_rate_1_missing", "charge_rate_2_missing", "charge_rate_3_missing", "charge_rate_4_missing",
                  "first_switch_soc", "first_switch_soc_missing", "fast_charge_end_soc", "topoff_rate_C",
                  "discharge_rate_C", "chamber_design_temperature_C"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(pending, path)


def policy_context(text):
    stages = [part.strip() for part in text.split("-") if "new" not in part.strip().lower()]
    switch = 0.0
    if len(stages) == 2:
        match = re.fullmatch(r"([0-9.]+)C\(([0-9.]+)%\)", stages[0])
        if match is None:
            raise ValueError("Unrecognized original two-stage charging policy " + text)
        rate1, percent = map(float, match.groups())
        rates, switch = [rate1, float(stages[1].removesuffix("C"))], percent / 100
    elif len(stages) == 4:
        rates = [float(part.removesuffix("C")) for part in stages]
    else:
        raise ValueError("Unrecognized original charging policy " + text)
    count = len(rates)
    values = [count] + rates + [0.0] * (4 - count) + [0.0] * count + [1.0] * (4 - count)
    values += [switch, float(count == 4), 0.8, 1.0, 4.0, 30.0]
    canonical = {"rates_C": rates, "stage_end_soc": [switch, 0.8] if count == 2 else [0.2, 0.4, 0.6, 0.8],
                 "topoff_C": 1.0, "discharge_C": 4.0, "chamber_C": 30.0}
    uid = "protocol_" + hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]
    return np.asarray(values, dtype=np.float32), uid, canonical


class LocalSuppliedUnpickler(pickle.Unpickler):
    """Allow only primitive containers and numpy scalar/dtype in existing PKLs."""

    def find_class(self, module, name):
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            return np.core.multiarray.scalar
        raise pickle.UnpicklingError("Unsupported supplied metadata class " + module + "." + name)


def channel_identity(voltage, normalized_current, charge_capacity, discharge_capacity):
    digest = hashlib.sha256()
    for values in (voltage, normalized_current, charge_capacity, discharge_capacity):
        array = np.asarray(values, dtype=np.float64)
        digest.update(len(array).to_bytes(8, "little"))
        digest.update(np.round(array, decimals=10).astype("<f8").tobytes())
    return digest.hexdigest()


def supplied_identity_index(path, include_components=False):
    with Path(path).open("rb") as handle:
        payload = LocalSuppliedUnpickler(handle).load()
    identities = {}
    components = {}
    for cycle in payload["cycle_data"]:
        fingerprint = channel_identity(cycle["voltage_in_V"], cycle["current_in_A"],
                                       cycle["charge_capacity_in_Ah"], cycle["discharge_capacity_in_Ah"])
        identities.setdefault(fingerprint, []).append(int(cycle["cycle_number"]))
        if include_components:
            discharge = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=float)
            boundaries = np.r_[0, np.flatnonzero(np.diff(discharge) < -0.001) + 1, len(discharge)]
            if len(boundaries) > 2:
                for left, right in zip(boundaries[:-1], boundaries[1:]):
                    component = channel_identity(np.asarray(cycle["voltage_in_V"])[left:right],
                                                 np.asarray(cycle["current_in_A"])[left:right],
                                                 np.asarray(cycle["charge_capacity_in_Ah"])[left:right], discharge[left:right])
                    components.setdefault(component, []).append({"processed_cycle_id": int(cycle["cycle_number"]),
                                                                  "source_point_start": int(left), "source_point_stop_exclusive": int(right)})
    return (identities, components) if include_components else identities
