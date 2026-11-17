"""Frozen, source-internal grouped CV for the E1--E3 mechanism experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "configs/drs_peft/e123_folds_v1.json"
WINDOWS = (
    ("all", 101, 5000),
    ("101_200", 101, 200),
    ("201_500", 201, 500),
    ("501_1000", 501, 1000),
    ("1001_plus", 1001, 5000),
)


def _digest(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _tie(seed, role, identifier):
    return hashlib.sha256(f"{seed}|{role}|{identifier}".encode()).hexdigest()


def balanced_group_partition(groups, bins, seed, role):
    """Balance source mask support, then cells and groups, without SOH values.

    Each group record contains its physical-cell identities and the number with
    actual 1001+ observations. The group enters one bin as an indivisible unit.
    """
    if bins < 2 or len(groups) < bins:
        raise ValueError("Grouped CV needs at least one group per bin")
    ordered = sorted(
        groups,
        key=lambda name: (
            -groups[name]["tail_cells"],
            -len(groups[name]["ids"]),
            _tie(seed, role, name),
        ),
    )
    assigned = [[] for _ in range(bins)]
    tails, cells, counts = [0] * bins, [0] * bins, [0] * bins
    for name in ordered:

        def score(index, name=name):
            trial_tails, trial_cells, trial_counts = (
                tails.copy(),
                cells.copy(),
                counts.copy(),
            )
            trial_tails[index] += groups[name]["tail_cells"]
            trial_cells[index] += len(groups[name]["ids"])
            trial_counts[index] += 1
            return (
                max(trial_tails) - min(trial_tails),
                max(trial_cells) - min(trial_cells),
                max(trial_counts) - min(trial_counts),
                _tie(seed, role, f"{name}|{index}"),
            )

        index = min(range(bins), key=score)
        assigned[index].append(name)
        tails[index] += groups[name]["tail_cells"]
        cells[index] += len(groups[name]["ids"])
        counts[index] += 1
    return [sorted(values) for values in assigned]


def _support(ids, qualification):
    result = {
        "cells": len(ids),
        "protocol_groups": len({qualification[uid]["protocol_uid"] for uid in ids}),
    }
    result["windows"] = {}
    for label, _, _ in WINDOWS:
        supported = [
            uid for uid in ids if qualification[uid]["window_observations"][label] > 0
        ]
        result["windows"][label] = {
            "physical_cells": len(supported),
            "protocol_groups": len(
                {qualification[uid]["protocol_uid"] for uid in supported}
            ),
            "observations": sum(
                qualification[uid]["window_observations"][label] for uid in ids
            ),
        }
    return result


def build_protocol(root=ROOT, split_seed=20261117):
    root = Path(root)
    parent_path = root / "configs/drs_peft/episodes_v1.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    roles = parent["source_cell_roles"]
    source_ids = sorted(roles["source_train"] + roles["source_meta_validation"])
    sealed_ids = sorted(roles["target_test"])
    if (
        len(source_ids) != 115
        or len(set(source_ids)) != 115
        or set(source_ids) & set(sealed_ids)
    ):
        raise RuntimeError(
            "E1--E3 requires the frozen 115-cell source pool and sealed targets"
        )
    cache = root / "runtime/drs_peft/data_v1/MATR"
    metadata_path = cache / "cell_metadata.csv"
    metadata = pd.read_csv(metadata_path, usecols=["cell_uid", "protocol_uid"])
    if metadata.cell_uid.duplicated().any():
        raise RuntimeError("The source identity table contains repeated physical cells")
    mapping = dict(
        zip(metadata.cell_uid.astype(str), metadata.protocol_uid.astype(str))
    )
    with np.load(cache / "arrays.npz", allow_pickle=False) as arrays:
        ids = arrays["ids"].astype(str).tolist()
        lookup = {uid: i for i, uid in enumerate(ids)}
        rows = np.asarray([lookup[uid] for uid in source_ids], dtype=np.int64)
        query = arrays["query"]
        observed = arrays["observed"][rows]
        protocols = arrays["protocol_ids"].astype(str)[rows]
    if any(mapping[uid] != group for uid, group in zip(source_ids, protocols)):
        raise RuntimeError("Cache and metadata disagree on a source protocol")
    qualification = {}
    for row, uid in enumerate(source_ids):
        counts = {
            label: int(observed[row, (query >= lower) & (query <= upper)].sum())
            for label, lower, upper in WINDOWS
        }
        qualification[uid] = {
            "protocol_uid": mapping[uid],
            "window_observations": counts,
        }
    groups = {}
    for uid in source_ids:
        record = groups.setdefault(mapping[uid], {"ids": [], "tail_cells": 0})
        record["ids"].append(uid)
        record["tail_cells"] += (
            qualification[uid]["window_observations"]["1001_plus"] > 0
        )
    for record in groups.values():
        record["tail_cells"] = int(record["tail_cells"])
    if len(groups) != 62:
        raise RuntimeError("The source pool must retain its frozen 62 exact protocols")
    outer = balanced_group_partition(groups, 3, split_seed, "outer")
    folds = []
    for index, test_groups in enumerate(outer):
        remaining = {
            name: record for name, record in groups.items() if name not in test_groups
        }
        inner_bins = balanced_group_partition(
            remaining, 5, split_seed, f"outer{index}|inner"
        )
        inner_groups = inner_bins[0]
        fit_groups = sorted(set(remaining) - set(inner_groups))
        fold = {
            "fold_id": index,
            "fold_name": f"outer{index}",
            "fit_groups": fit_groups,
            "inner_validation_groups": inner_groups,
            "outer_test_groups": test_groups,
        }
        for prefix in ("fit", "inner_validation", "outer_test"):
            selected = sorted(
                uid
                for group in fold[prefix + "_groups"]
                for uid in groups[group]["ids"]
            )
            fold[prefix + "_ids"] = selected
            fold[prefix + "_support"] = _support(selected, qualification)
        folds.append(fold)
    result = {
        "study_id": "DRS_PEFT_E1_E2_E3_SOURCE_GROUP_CV_20261117_v1",
        "split_seed": split_seed,
        "outer_folds": 3,
        "inner_group_partition_bins": 5,
        "inner_validation_bin": 0,
        "training_seeds": [42, 123, 2026],
        "source_ids": source_ids,
        "source_groups": sorted(groups),
        "sealed_target_ids": sealed_ids,
        "source_qualification": qualification,
        "folds": folds,
        "split_rule": "Indivisible exact groups, descending 1001+ supported-cell count then size; greedily minimize tail-cell range, cell range, group range; SHA256 ties. Apply same rule to five inner bins, use bin0.",
        "information_budget": {
            "source_observation_masks_used_for_stratification": True,
            "SOH_values_read_for_split": 0,
            "model_predictions_read_for_split": 0,
            "final_target_mask_statistics_computed": False,
            "final_target_labels_used": 0,
            "only_source_rows_contribute_to_split_statistics": True,
        },
        "history": "Retrospective source-internal nested group CV. Every source cell has historical study exposure; outer roles are excluded from this fold's fitting, basis construction, checkpoint selection and hyperparameter selection.",
        "initialization_rule": "Official W0 plus newly trained fold-local bridge/head. Fit scaler, source bases and source adapters using this fold's fit role; select checkpoints using inner validation only. Outer scores are released after all registered units finish.",
        "provenance": {
            "parent_episode_file": "configs/drs_peft/episodes_v1.json",
            "parent_episode_file_sha256": _file_digest(parent_path),
            "parent_episode_payload_sha256": parent["episode_contract_sha256"],
            "source_cache_complete_sha256": _file_digest(cache / "COMPLETE.json"),
            "source_metadata_sha256": _file_digest(metadata_path),
            "source_observed_mask_sha256": hashlib.sha256(
                observed.tobytes()
            ).hexdigest(),
            "source_qualification_sha256": _digest(qualification),
        },
    }
    result["protocol_sha256"] = _digest(result)
    validate_protocol(result, parent)
    return result


def validate_protocol(protocol, parent_episodes=None):
    payload = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
    if protocol.get("protocol_sha256") != _digest(payload):
        raise RuntimeError("The E1--E3 protocol identity changed")
    source = set(protocol["source_ids"])
    sealed = set(protocol["sealed_target_ids"])
    qualification = protocol["source_qualification"]
    if (
        len(source) != len(protocol["source_ids"])
        or source & sealed
        or set(qualification) != source
    ):
        raise RuntimeError("Ambiguous source identities or sealed-target overlap")
    if parent_episodes is not None:
        roles = parent_episodes["source_cell_roles"]
        if source != set(
            roles["source_train"] + roles["source_meta_validation"]
        ) or sealed != set(roles["target_test"]):
            raise RuntimeError(
                "The grouped CV pool differs from its authorized parent roles"
            )
    if len(protocol["folds"]) != protocol["outer_folds"]:
        raise RuntimeError("Missing registered outer folds")
    outer_counts = Counter()
    for fold in protocol["folds"]:
        role_sets, group_sets = [], []
        for prefix in ("fit", "inner_validation", "outer_test"):
            ids = fold[prefix + "_ids"]
            role = set(ids)
            if not role or len(role) != len(ids) or role & sealed or not role <= source:
                raise RuntimeError(
                    "A fold role is empty, repeated or contains a sealed target"
                )
            groups = {qualification[uid]["protocol_uid"] for uid in ids}
            if sorted(groups) != fold[prefix + "_groups"]:
                raise RuntimeError(
                    "Fold protocol membership differs from its physical cells"
                )
            if _support(ids, qualification) != fold[prefix + "_support"]:
                raise RuntimeError("Fold support counts changed")
            role_sets.append(role)
            group_sets.append(groups)
        if set.union(*role_sets) != source:
            raise RuntimeError("Fold roles do not cover the source pool")
        if any(
            role_sets[i] & role_sets[j] or group_sets[i] & group_sets[j]
            for i, j in ((0, 1), (0, 2), (1, 2))
        ):
            raise RuntimeError(
                "Physical cells or exact protocol groups cross fold roles"
            )
        outer_counts.update(fold["outer_test_ids"])
    if set(outer_counts) != source or any(
        count != 1 for count in outer_counts.values()
    ):
        raise RuntimeError(
            "Each source physical cell must receive exactly one outer evaluation"
        )
    return protocol


def load_protocol(path=DEFAULT_PATH):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_protocol(protocol)


def validate_against_data(protocol, data, parent_episodes):
    """Bind registered fold identities and support masks to the live source cache."""
    validate_protocol(protocol, parent_episodes)
    if data.domain != "MATR" or len(data.ids) != len(set(data.ids)):
        raise RuntimeError("Grouped CV requires unique physical MATR identities")
    lookup = {uid: row for row, uid in enumerate(data.ids)}
    if not set(protocol["source_ids"]).issubset(lookup):
        raise RuntimeError("The current source cache omits registered physical cells")
    query = np.asarray(data.query)
    for uid in protocol["source_ids"]:
        row = lookup[uid]
        registered = protocol["source_qualification"][uid]
        if str(data.protocol_ids[row]) != registered["protocol_uid"]:
            raise RuntimeError("Current source exact protocol identities changed")
        for label, lower, upper in WINDOWS:
            count = int(
                np.asarray(data.observed[row])[
                    (query >= lower) & (query <= upper)
                ].sum()
            )
            if count != registered["window_observations"][label]:
                raise RuntimeError("Current source observation support changed")
    return {
        "status": "PASSED",
        "source_physical_cells": len(protocol["source_ids"]),
        "source_protocol_groups": len(protocol["source_groups"]),
        "outer_physical_cell_evaluations": len(protocol["source_ids"]),
        "source_SOH_values_read": 0,
        "final_target_SOH_values_read": 0,
    }


def fold_episode_view(parent_episodes, fold):
    """Expose fit/inner roles to the unchanged trainer, with outer scoring separate."""
    parent_roles = parent_episodes["source_cell_roles"]
    authorized = set(
        parent_roles["source_train"] + parent_roles["source_meta_validation"]
    )
    roles = {
        "source_train": list(fold["fit_ids"]),
        "source_meta_validation": list(fold["inner_validation_ids"]),
        "source_outer_validation": list(fold["outer_test_ids"]),
        "target_test": list(parent_roles["target_test"]),
    }
    active = [
        set(roles[name])
        for name in (
            "source_train",
            "source_meta_validation",
            "source_outer_validation",
        )
    ]
    if any(len(set(values)) != len(values) for values in roles.values()):
        raise RuntimeError("Fold trainer view repeats a physical cell")
    if set.union(*active) != authorized or any(
        active[i] & active[j] for i, j in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("Fold trainer view has invalid source roles")
    if any(values & set(roles["target_test"]) for values in active):
        raise RuntimeError("Fold trainer view includes a sealed final target")
    result = {
        "study_id": "DRS_PEFT_E123_FOLD_LOCAL_SOURCE",
        "fold_id": fold["fold_id"],
        "parent_episode_contract_sha256": parent_episodes["episode_contract_sha256"],
        "source_cell_roles": roles,
        "role_purpose": "fit trains; inner selects; outer is scored after freeze",
    }
    result["episode_contract_sha256"] = _digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    protocol = build_protocol()
    if args.output.exists():
        if load_protocol(args.output) != protocol:
            raise RuntimeError("An existing grouped CV registration differs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": "REGISTERED",
                "protocol_sha256": protocol["protocol_sha256"],
                "folds": [
                    {
                        key: fold[key]
                        for key in (
                            "fold_id",
                            "fit_support",
                            "inner_validation_support",
                            "outer_test_support",
                        )
                    }
                    for fold in protocol["folds"]
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
