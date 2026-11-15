"""Seed42 physical-cell splits using only cell UID and exact known protocol."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


HASH_NAMESPACE = "unified-matr-seed42-splits-v1"
ROLES = ("train", "validation", "test")


def hash_order(value, purpose):
    return hashlib.sha256(f"{HASH_NAMESPACE}|{purpose}|{value}".encode("utf-8")).hexdigest()


def validation_cells(pool, protocols, fraction, purpose):
    if len(pool) < 2:
        raise ValueError("A split needs at least two non-test cells for training and validation")
    groups = defaultdict(list)
    for uid in pool:
        groups[protocols[uid]].append(uid)
    target = min(len(pool) - 1, max(1, round(len(pool) * fraction)))
    allocation = {group: math.floor(len(members) * fraction) for group, members in groups.items()}
    remainders = sorted(groups, key=lambda group: (-(len(groups[group]) * fraction - allocation[group]),
                                                  hash_order(group, purpose + "/quota")))
    for group in remainders[:target - sum(allocation.values())]:
        allocation[group] += 1
    selected = set()
    for group, members in groups.items():
        order = sorted(members, key=lambda uid: hash_order(uid, purpose + "/" + group))
        selected.update(order[:allocation[group]])
    return selected


def role_partition(identifiers, test, protocols, validation_fraction, purpose):
    pool = set(identifiers) - test
    validation = validation_cells(pool, protocols, validation_fraction, purpose)
    return {"train": sorted(pool - validation), "validation": sorted(validation), "test": sorted(test)}


def validate_partitions(result, identifiers):
    expected = set(identifiers)
    test_counts = Counter()
    for name, roles in result["splits"].items():
        role_sets = {role: set(roles[role]) for role in ROLES}
        if any(len(role_sets[role]) != len(roles[role]) or not role_sets[role] for role in ROLES):
            raise ValueError("Repeated or empty cell role in " + name)
        if (role_sets["train"] & role_sets["validation"] or role_sets["train"] & role_sets["test"]
                or role_sets["validation"] & role_sets["test"]):
            raise ValueError("A physical cell crosses roles in " + name)
        if set.union(*role_sets.values()) != expected:
            raise ValueError("A split omits cells from the fixed qualification cohort")
        if name.startswith("cell_fold_"):
            test_counts.update(roles["test"])
    if test_counts != Counter({uid: 1 for uid in identifiers}):
        raise ValueError("Every physical cell must occur in exactly one outer test fold")
    sensitivity = result["splits"]["unseen_protocol_sensitivity"]
    protocol = result["protocol_groups"]
    training_groups = {protocol[uid] for uid in sensitivity["train"] + sensitivity["validation"]}
    testing_groups = {protocol[uid] for uid in sensitivity["test"]}
    if training_groups & testing_groups:
        raise ValueError("An unseen-protocol test group occurs in training or validation")


def build_splits(cells_rows, configsplit):
    """Assign all fixed physical UIDs before any input or target masking.

    Only ``cell_uid`` and ``protocol_uid`` are read from each input row.
    Continuation segments must already share their single physical UID.
    """
    settings = {"seed": configsplit.get("seed", 42),
                "outer_cell_folds": configsplit.get("outer_cell_folds", 5),
                "validation_fraction_of_outer_train": configsplit.get("validation_fraction_of_outer_train", 0.2),
                "protocol_sensitivity_test_fraction": configsplit.get("protocol_sensitivity_test_fraction", 0.2)}
    if settings != {"seed": 42, "outer_cell_folds": 5, "validation_fraction_of_outer_train": 0.2,
                    "protocol_sensitivity_test_fraction": 0.2}:
        raise ValueError("This registered split version requires seed42, five folds and both fractions0.2")
    protocols = {}
    for row in cells_rows:
        uid, group = row["cell_uid"], row["protocol_uid"]
        if not isinstance(uid, str) or not uid or not isinstance(group, str) or not group:
            raise ValueError("Physical cell UID and exact known protocol must be nonempty strings")
        if uid in protocols:
            raise ValueError("Duplicate physical cell UID; merge its continuation segments before splitting")
        protocols[uid] = group
    if len(protocols) < 5:
        raise ValueError("Five nonempty outer test folds require at least five physical cells")
    identifiers = sorted(protocols)
    grouped = defaultdict(list)
    for uid in identifiers:
        grouped[protocols[uid]].append(uid)
    if len(grouped) < 2:
        raise ValueError("Protocol-heldout sensitivity requires at least two exact known protocol groups")
    outer_tests = [set() for _ in range(5)]
    for group in sorted(grouped, key=lambda value: hash_order(value, "outer/protocol_order")):
        members = sorted(grouped[group], key=lambda uid: hash_order(uid, "outer/cell_order/" + group))
        # Each protocol differs by at most one cell across folds. The remainder
        # goes to the smallest current folds, preserving overall count balance.
        fold_order = sorted(range(5), key=lambda fold: (len(outer_tests[fold]), hash_order(str(fold), "outer/tie/" + group)))
        for number, uid in enumerate(members):
            outer_tests[fold_order[number % 5]].add(uid)
    splits = {f"cell_fold_{fold}": role_partition(identifiers, test, protocols, 0.2, f"cell_fold_{fold}/validation")
              for fold, test in enumerate(outer_tests)}
    protocol_order = sorted(grouped, key=lambda group: hash_order(group, "unseen_protocol/test"))
    group_count = min(len(grouped) - 1, max(1, round(len(grouped) * 0.2)))
    heldout_groups = set(protocol_order[:group_count])
    heldout_cells = {uid for uid in identifiers if protocols[uid] in heldout_groups}
    splits["unseen_protocol_sensitivity"] = role_partition(identifiers, heldout_cells, protocols, 0.2, "unseen_protocol/validation")
    result = {"seed": 42, "splits": splits, "protocol_groups": {uid: protocols[uid] for uid in identifiers},
              "protocol_group_sizes": {group: len(grouped[group]) for group in sorted(grouped)},
              "settings": {**settings, "hash_algorithm": "sha256", "hash_namespace": HASH_NAMESPACE,
                           "assignment_fields": ["cell_uid", "protocol_uid"],
                           "validation_allocation": "cell quotas proportional to exact protocol with largest remainders",
                           "heldout_fraction_unit": "exact protocol groups",
                           "protocol_groups_held_out": sorted(heldout_groups),
                           "cohort_cells": len(identifiers), "input_eligibility_used_for_assignment": False,
                           "future_targets_used_for_assignment": False},
              "identity_unit": "one unique physical cell UID including all continuation segments",
              "historical_exposure": "The fixed cohort participated in previous exploratory analyses; single-seed internal comparisons retain this exposure.",
              "singleton_protocol_limitation": "Singleton groups participate in splits; within-protocol replicate variance is unavailable for them."}
    validate_partitions(result, identifiers)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["split_fingerprint"] = hashlib.sha256(payload).hexdigest()
    return result


def selfcheck():
    counts = [1, 2, 3, 4, 7, 15, 20, 30, 87]
    rows = []
    for group, count in enumerate(counts):
        for number in range(count):
            rows.append({"cell_uid": f"cell_{len(rows):03d}", "protocol_uid": f"protocol_{group}",
                         "input_eligible": number % 2 == 0, "historical_exposure": True,
                         "source_segments": ["original", "continuation"] if not rows else ["original"]})
    settings = {"seed": 42, "outer_cell_folds": 5, "validation_fraction_of_outer_train": 0.2,
                "protocol_sensitivity_test_fraction": 0.2}
    result = build_splits(rows, settings)
    records = []

    def require(condition, name):
        if not condition:
            raise AssertionError(name)
        records.append({"check": name, "status": "PASS"})

    require(result == build_splits(list(reversed(rows)), settings), "row_order_invariance")
    perturbed = [{**row, "input_eligible": not row["input_eligible"], "historical_exposure": False,
                  "future_target": float("nan"), "future_length": 1000000, "source_segments": ["different_alias"]}
                 for row in rows]
    require(result == build_splits(perturbed, settings), "input_mask_future_labels_and_continuation_metadata_do_not_change_assignment")
    counts = [len(result["splits"][f"cell_fold_{fold}"]["test"]) for fold in range(5)]
    require(max(counts) - min(counts) <= 1, "outer_cell_counts_balanced_within_one")
    for group in result["protocol_group_sizes"]:
        counts = [sum(result["protocol_groups"][uid] == group for uid in result["splits"][f"cell_fold_{fold}"]["test"]) for fold in range(5)]
        require(max(counts) - min(counts) <= 1, "exact_protocol_outer_stratification_" + group)
    require(len(result["splits"]) == 6, "five_outer_folds_plus_one_protocol_sensitivity")
    require(all(set.union(*(set(roles[role]) for role in ROLES)) == {row["cell_uid"] for row in rows}
                for roles in result["splits"].values()), "all169_retained_before_input_execution_mask")
    require(sum("cell_000" in result["splits"][f"cell_fold_{fold}"]["test"] for fold in range(5)) == 1,
            "continuation_physical_UID_outer_test_once")
    require(len(result["settings"]["protocol_groups_held_out"]) == 2, "fixed_rounded20percent_protocol_group_holdout")
    for field, value in (("seed", 43), ("outer_cell_folds", 4), ("validation_fraction_of_outer_train", 0.25)):
        try:
            build_splits(rows, {**settings, field: value})
        except ValueError:
            rejected = True
        else:
            rejected = False
        require(rejected, "unregistered_split_setting_rejected_" + field)
    try:
        build_splits(rows + rows[:1], settings)
    except ValueError:
        rejected = True
    else:
        rejected = False
    require(rejected, "duplicate_physical_UID_rejected")
    return {"status": "PASS", "scope": "synthetic_split_engineering_only", "optimizer_updates": 0,
            "checks": len(records), "records": records, "fixture_cells": len(rows),
            "fixture_split_fingerprint": result["split_fingerprint"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = selfcheck()
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "checks", "scope")}))


if __name__ == "__main__":
    main()
