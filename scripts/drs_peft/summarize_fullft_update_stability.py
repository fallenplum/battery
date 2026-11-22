"""Summarize cross-fold Full-FT update geometry without reading new labels.

The public geometry receipt contains per-parameter update norms and singular
value ranks. It does not contain update vectors, so this script reports rank and
norm-profile stability and explicitly leaves directional cosine unavailable.
It never loads future labels or runs a model.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.self_reference_protocol import PRIVATE, PUBLIC


FOLDS = (0, 1, 2)
EPOCH = 8


def group(name: str) -> str:
    low = name.lower()
    for tag in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "fc_gate"):
        if tag in low:
            return tag
    if "fusion12" in low:
        return "bridge_fusion12"
    if "fusion30" in low:
        return "bridge_fusion30"
    if "head" in low:
        return "head"
    return "other"


def layer(name: str) -> str:
    marker = "encoder.layers."
    if marker in name:
        tail = name.split(marker, 1)[1]
        return "encoder.layer." + tail.split(".", 1)[0]
    if name.startswith("bridge.backbone.module"):
        return "backbone.other"
    if name.startswith("fusion12"):
        return "bridge.fusion12"
    if name.startswith("fusion30"):
        return "bridge.fusion30"
    if name.startswith("head"):
        return "head"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--public", type=Path, default=PUBLIC / "fullft_update_audit")
    args = parser.parse_args()
    args.public.mkdir(parents=True, exist_ok=True)

    source_geometry = args.public / "UPDATE_GEOMETRY.csv"
    if not source_geometry.is_file():
        raise FileNotFoundError(source_geometry)
    geometry_frame = pd.read_csv(source_geometry)
    geometry_frame = geometry_frame[geometry_frame["epoch"].eq(EPOCH)].copy()
    geometry_frame["layer"] = geometry_frame["parameter"].map(layer)
    geometry_frame.to_csv(args.public / "UPDATE_LAYER_GEOMETRY.csv", index=False, lineterminator="\n")

    # Layer and operation-family energy. A norm is formed from all tensors in
    # the cell, preserving the relative contribution of each layer.
    layer_rows = []
    for (fold, layer_name, family), frame in geometry_frame.groupby(["fold", "layer", "group"], sort=True):
        energy = float(np.square(frame["frobenius"].to_numpy()).sum())
        layer_rows.append(
            {
                "fold": int(fold),
                "layer": layer_name,
                "group": family,
                "frobenius": math.sqrt(energy),
                "rank99_mean": float(frame["rank99"].dropna().mean()) if frame["rank99"].notna().any() else None,
                "rank99_median": float(frame["rank99"].dropna().median()) if frame["rank99"].notna().any() else None,
                "parameters": int(len(frame)),
            }
        )
    layer_frame = pd.DataFrame(layer_rows)
    layer_frame.to_csv(args.public / "UPDATE_LAYER_SUMMARY.csv", index=False, lineterminator="\n")

    # Full vectors are deliberately not part of the public receipt. Compare the
    # fold-wise per-parameter Frobenius profiles instead and label that quantity
    # separately. It is not an update-direction cosine.
    stability = []
    groups = sorted(geometry_frame["group"].unique())
    for family in groups:
        family_frame = geometry_frame[geometry_frame["group"].eq(family)]
        family_names = sorted(family_frame["parameter"].unique())
        profiles = {
            fold: family_frame[family_frame["fold"].eq(fold)]
            .set_index("parameter")
            .reindex(family_names)["frobenius"]
            .to_numpy(dtype=float)
            for fold in FOLDS
        }
        for left, right in itertools.combinations(FOLDS, 2):
            left_vec, right_vec = profiles[left], profiles[right]
            left_norm = float(np.linalg.norm(left_vec))
            right_norm = float(np.linalg.norm(right_vec))
            if left_norm == 0 or right_norm == 0:
                cosine = None
                angle = None
            else:
                cosine = float(np.dot(left_vec, right_vec) / (left_norm * right_norm))
                angle = math.degrees(math.acos(max(-1.0, min(1.0, abs(cosine)))))
            stability.append(
                {
                    "group": family,
                    "fold_left": left,
                    "fold_right": right,
                    "norm_left": left_norm,
                    "norm_right": right_norm,
                    "norm_profile_cosine": cosine,
                    "norm_profile_angle_deg_abs": angle,
                    "update_direction_cosine": None,
                    "update_direction_angle_deg_abs": None,
                }
            )
    stability_frame = pd.DataFrame(stability)
    stability_frame.to_csv(args.public / "UPDATE_STABILITY.csv", index=False, lineterminator="\n")

    summary = {
        "status": "COMPLETE_FULL_FT_UPDATE_STABILITY",
        "folds": list(FOLDS),
        "full_ft_epoch": EPOCH,
        "snapshot_source": "public UPDATE_GEOMETRY.csv, epoch 8 source-fit Full FT minus locked warmup",
        "target_future_labels_read": 0,
        "outer_labels_read": 0,
        "cross_fold_angle_definition": "acos(abs(cosine)) on per-parameter Frobenius-norm profiles within each operation family",
        "update_direction_stability": "UNAVAILABLE_PUBLIC_VECTOR_RECEIPT",
        "files": {
            name: file_hash(args.public / name)
            for name in (
                "UPDATE_LAYER_GEOMETRY.csv",
                "UPDATE_LAYER_SUMMARY.csv",
                "UPDATE_STABILITY.csv",
            )
        },
    }
    write_json(args.public / "STABILITY_COMPLETE.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
