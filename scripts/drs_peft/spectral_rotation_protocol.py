"""Freeze MATR source episode roles before any response arrays are decoded."""

from __future__ import annotations

import hashlib
from itertools import combinations

from scripts.drs_peft.data import ROOT, file_hash
from scripts.drs_peft.onesided_protocol import load_registration
from scripts.drs_peft.run_abba_moirai import checkpoint, task_for
from scripts.drs_peft.train import read_json

STUDY = "spectral_rotation_v1"
PUBLIC = ROOT / "results/drs_peft" / STUDY
PRIVATE = ROOT / "runtime/drs_peft" / STUDY
CONFIG = ROOT / "configs/drs_peft" / STUDY / "CONFIG.json"
CORE_MODES = {
    "STATIC_CORE": "static",
    "DRS": "drs",
    "GAIN": "gain",
    "JOINT2R": "joint",
    "ROTATION_CONST": "constant",
    "ROTATION_TRUE": "conditioned",
    "ROTATION_SHUFFLED": "shuffled",
}
META_METHODS = ("DRS", "GAIN", "ROTATION_CONST", "ROTATION_TRUE", "ROTATION_SHUFFLED")
SAME_CORE = "ROTATION_SAME_CORE_SHUFFLED"
CANDIDATE = "ROTATION_TRUE"
PARAMETER_MATCHED = (
    "STATIC_CORE",
    "DRS",
    "GAIN",
    "JOINT2R",
    "ROTATION_CONST",
    "ROTATION_SHUFFLED",
)


def ordered(items, salt):
    def order_key(value):
        key = str(value)
        return hashlib.sha256(f"{STUDY}|{salt}|{key}".encode()).digest(), key

    return sorted(items, key=order_key)


def different_descriptors(uids, salt):
    """Derange a fixed role's descriptor identities and preserve its marginals."""
    if len(uids) < 2 or len(uids) != len(set(uids)):
        raise ValueError("A shuffled descriptor role needs distinct physical cells")
    roster = ordered(uids, salt)
    donor = dict(zip(roster, roster[1:] + roster[:1], strict=True))
    return [donor[uid] for uid in uids]


def diverse_take(uids, count, protocols, salt):
    selected, seen = [], set()
    for uid in ordered(uids, salt):
        protocol = protocols[uid]
        if protocol not in seen:
            selected.append(uid)
            seen.add(protocol)
        if len(selected) == count:
            return selected
    raise ValueError("Too few different source training protocols for K3")


def train_episodes(ids, protocols, cfg, fold):
    episodes = []
    for draw in range(cfg["train_episodes"]):
        key = (cfg["seed"], fold, "meta", draw)
        query = diverse_take(ids, cfg["train_query_cells"], protocols, (key, "query"))
        excluded = {protocols[uid] for uid in query}
        support = diverse_take(
            [uid for uid in ids if protocols[uid] not in excluded],
            cfg["shot"],
            protocols,
            (key, "support"),
        )
        episodes.append({"draw": draw, "support": support, "query": query})
    return episodes


def evaluation_episodes(task, cfg, protocols, limit):
    query = task["query_ids"]
    excluded = {protocols[uid] for uid in query}
    pool = sorted(
        uid for uid in task["support_pool_ids"] if protocols[uid] not in excluded
    )
    candidates = ordered(
        combinations(pool, cfg["shot"]), (cfg["seed"], task["task_id"])
    )
    if not candidates:
        raise ValueError("No strictly cross-protocol K3 Support set")
    return [
        {"draw": i, "support": list(support), "query": query}
        for i, support in enumerate(candidates[:limit])
    ]


def role_ids(episodes):
    return {
        uid for row in episodes for role in ("support", "query") for uid in row[role]
    }


def validate_fold(fit, validation, outer, protocols, source_ids, sealed):
    physical_groups = (set(fit), role_ids(validation), role_ids(outer))
    if any(group - source_ids or group & sealed for group in physical_groups):
        raise RuntimeError(
            "A non-source battery was admitted into the source-only screen"
        )
    protocol_groups = tuple(
        {protocols[uid] for uid in group} for group in physical_groups
    )
    for first, second in combinations(protocol_groups, 2):
        if first & second:
            raise RuntimeError(
                "Fit, validation or scoring protocols overlap within a fold"
            )
    for row in (*validation, *outer):
        if {protocols[uid] for uid in row["support"]} & {
            protocols[uid] for uid in row["query"]
        }:
            raise RuntimeError("Support and scoring Query have the same protocol")


def build_plan(args):
    cfg = read_json(args.config)
    if (
        cfg["domain"] != "MATR"
        or cfg["folds"] != [0, 1, 2]
        or cfg["shot"] != 3
        or cfg["outer_label_access_before_lock"]
        or cfg["target_label_access"]
        or cfg["methods"] != ["K0", "CAL2", "LORA_SMALL", *CORE_MODES, SAME_CORE]
    ):
        raise RuntimeError("Protected input and method contract changed")
    episodes_path = ROOT / cfg["source_episodes"]
    parent = load_registration(episodes_path)
    protocols = parent["source_protocols"]
    source_ids = set(parent["source_ids"])
    sealed = set(parent["sealed_final_MATR_ids"])
    folds, all_query, all_query_protocols = [], set(), set()
    for fold in cfg["folds"]:
        reference = checkpoint(args, fold, cfg["seed"])
        fit = reference["fit_ids"]
        inner = task_for(parent, "inner", fold)
        outside = task_for(parent, "outer", fold)
        if set(reference["inner_validation_ids"]) != set(
            inner["query_ids"] + inner["support_pool_ids"]
        ):
            raise RuntimeError("The parent reference or validation roles changed")
        validation = evaluation_episodes(
            inner, cfg, protocols, cfg["validation_max_draws"]
        )
        outer = evaluation_episodes(outside, cfg, protocols, cfg["heldout_draws"])
        validate_fold(fit, validation, outer, protocols, source_ids, sealed)
        query = set(outer[0]["query"])
        query_protocols = {protocols[uid] for uid in query}
        reference_validation_protocols = {
            protocols[uid] for uid in reference["inner_validation_ids"]
        }
        if (
            len(outer) != cfg["heldout_draws"]
            or len(query) != 20
            or all_query & query
            or all_query_protocols & query_protocols
            or reference_validation_protocols & query_protocols
        ):
            raise RuntimeError("Expected distinct 20-cell Query across each outer fold")
        all_query.update(query)
        all_query_protocols.update(query_protocols)
        folds.append(
            {
                "fold": fold,
                "reference_sha256": reference["reference_checkpoint_sha256"],
                "fit_ids": fit,
                "anchors": ordered(fit, (cfg["seed"], fold, "anchor"))[
                    : cfg["anchor_cells"]
                ],
                "train": train_episodes(fit, protocols, cfg, fold),
                "validation": validation,
                "outer": outer,
            }
        )
    plan = {
        "status": "REGISTERED_WITHOUT_OUTER_LABELS",
        "config_sha256": file_hash(args.config),
        "parent_episodes_sha256": file_hash(episodes_path),
        "historically_used_source_population": True,
        "outer_source_query_cells": len(all_query),
        "outer_source_query_protocols": len(all_query_protocols),
        "folds": folds,
    }
    return cfg, plan, protocols, source_ids, sealed
