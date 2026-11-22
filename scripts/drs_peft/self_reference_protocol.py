"""Register independent source protocol roles for the no-Support experiment."""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

from scripts.drs_peft.data import ROOT, file_hash
from scripts.drs_peft.train import read_json

NAME = "self_reference_v1"
CONFIG = ROOT / "configs/drs_peft" / NAME / "CONFIG.json"
PUBLIC = ROOT / "results/drs_peft" / NAME
PRIVATE = ROOT / "runtime/drs_peft" / NAME
METHODS = ("LORA_12", "LORA_30", "RANDOM_30", "VARIANCE_30", "SELF_REFERENCE_30")


def fit_and_tune(fit, groups, fold, seed, minimum):
    candidates = {groups[uid] for uid in fit}
    order = sorted(
        candidates,
        key=lambda group: hashlib.sha256(
            f"{NAME}|{seed}|{fold}|{group}".encode()
        ).digest(),
    )
    tune = set()
    for group in order:
        tune.update(uid for uid in fit if groups[uid] == group)
        if len(tune) >= minimum:
            break
    if len(tune) < minimum or len(set(fit) - tune) < minimum:
        raise ValueError("Cannot reserve whole tuning protocols with adequate coverage")
    return sorted(set(fit) - tune), sorted(tune)


def build_plan(config=CONFIG):
    cfg = read_json(config)
    if (
        cfg["domain"] != "MATR"
        or cfg["folds"] != [0, 1, 2]
        or tuple(cfg["methods"]) != METHODS
        or cfg["target_future_label_access"]
        or cfg["supports_at_inference"] != 0
    ):
        raise ValueError("This screen is restricted to registered MATR source roles")
    parent_path = ROOT / cfg["parent_plan"]
    registry_path = ROOT / cfg["source_registry"]
    parent, registry = read_json(parent_path), read_json(registry_path)
    group = registry["source_protocols"]
    source = set(registry["source_ids"])
    sealed = set(registry["sealed_final_MATR_ids"])
    folds = []
    for old in parent["folds"]:
        fold = old["fold"]
        fit, tune = fit_and_tune(
            old["fit_ids"], group, fold, cfg["seed"], cfg["minimum_tuning_cells"]
        )
        screen = sorted(
            {
                uid
                for item in old["validation"]
                for role in ("query", "support")
                for uid in item[role]
            }
        )
        # Include the parent inner role's full qualified source roster. These cells
        # are each independently evaluated, with no designated inference Support.
        parent_task = next(
            row
            for row in registry["source_tasks"]["inner"]
            if int(row["fold_id"]) == fold
        )
        complete_inner = set(parent_task["query_ids"] + parent_task["support_pool_ids"])
        if set(screen) - complete_inner:
            raise ValueError("Parent inner roster does not match its frozen task")
        screen = sorted(complete_inner)
        roles = [set(fit), set(tune), set(screen)]
        if any(role - source or role & sealed for role in roles):
            raise RuntimeError("Only qualified source cells may enter this experiment")
        protocols = [{group[uid] for uid in role} for role in roles]
        if any(left & right for left, right in combinations(protocols, 2)):
            raise RuntimeError(
                "A fold shares protocol identity across fit, tuning or screen"
            )
        parent_outer = {
            uid
            for item in old["outer"]
            for role in ("support", "query")
            for uid in item[role]
        }
        if set.union(*roles) & parent_outer:
            raise RuntimeError(
                "Parent outer role overlapped the new source screen in a fold"
            )
        folds.append(
            {
                "fold": fold,
                "fit_ids": fit,
                "tune_ids": tune,
                "screen_ids": screen,
                "role_protocols": {
                    name: sorted(value)
                    for name, value in zip(
                        ("fit", "tune", "screen"), protocols, strict=True
                    )
                },
            }
        )
    return (
        cfg,
        {
            "status": "REGISTERED_SOURCE_SCREEN_WITHOUT_FUTURE_LABELS",
            "study_id": cfg["study_id"],
            "config_sha256": file_hash(Path(config)),
            "parent_plan_sha256": file_hash(parent_path),
            "source_registry_sha256": file_hash(registry_path),
            "scope": "historically_exposed_source_internal_protocol_screen",
            "folds": folds,
        },
        group,
        source,
        sealed,
    )
