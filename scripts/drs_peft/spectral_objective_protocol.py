"""Register the same source evaluation and three fit-only meta episode schedules."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from scripts.drs_peft import spectral_rotation_protocol as rotation
from scripts.drs_peft.data import ROOT, file_hash
from scripts.drs_peft.onesided_protocol import load_registration
from scripts.drs_peft.run_abba_moirai import checkpoint, task_for
from scripts.drs_peft.train import bind, read_json
from scripts.drs_peft.transfer_scope_reference import REFERENCES


STUDY = "spectral_objective_v1"
CONFIG = ROOT / "configs/drs_peft" / STUDY / "CONFIG.json"
PUBLIC = ROOT / "results/drs_peft" / STUDY
PRIVATE = ROOT / "runtime/drs_peft" / STUDY
CANDIDATE = "ROTATION_TRUE_MA"
MATCHED_RATE = "ROTATION_TRUE_MM_AT_MA_RATE"
SAME_CORE = "ROTATION_TRUE_MA_SAME_CORE_SHUFFLED"
META_METHODS = ("ROTATION_TRUE", "ROTATION_CONST", "ROTATION_SHUFFLED", "GAIN")
META_OBJECTIVES = {"MM": "mse", "MA": "mae"}
BASELINES = ("K0", "STATIC_CORE", "LORA_SMALL", "CAL2")
CORE_MODES = {method: rotation.CORE_MODES[method] for method in ("STATIC_CORE", *META_METHODS)}
REQUIRED_RAW_REFERENCES = (*BASELINES, "GAIN_MA", "ROTATION_CONST_MA", "ROTATION_SHUFFLED_MA")
PARAMETER_REFERENCES = ("ROTATION_CONST_MA", "ROTATION_SHUFFLED_MA")
OUTER_METHODS = (*BASELINES, *(f"{method}_{tag}" for method in META_METHODS for tag in META_OBJECTIVES),
                 MATCHED_RATE, SAME_CORE)
role_ids = rotation.role_ids


def validate_config(cfg):
    fixed = {
        "study_id": "MATR_SPECTRAL_OBJECTIVE_ALIGNMENT_V1",
        "parent_revision": "87ad1c8c403503f15833f825be91c748bbf3a7bd",
        "domain": "MATR", "folds": [0, 1, 2], "shot": 3,
        "reference_seed": 42, "evaluation_plan_seed": 42, "meta_training_seeds": [42, 43, 44],
        "selection_seed": 42, "primary_candidate": CANDIDATE, "train_episodes": 32, "train_query_cells": 3,
        "validation_max_draws": 3, "heldout_draws": 3, "anchor_cells": 8,
        "support_max_observed": 512, "meta_query_max_observed": 512,
        "angle_max_radians": 0.5235987755982988, "meta_lr": 0.01, "inner_objective": "mse",
        "meta_query_objectives": list(META_OBJECTIVES.values()), "core_steps": 3,
        "core_rates": [1e-4, 1e-3, 1e-2, 1e-1], "lora_rates": [1e-7, 1e-6, 1e-5, 3e-5],
        "lora_steps": [1, 10], "cal2_ridge": [1e-4, .01, 1.0], "cal2_eta": [.01, .1, .25, 1.0],
        "mapper_methods": list(META_METHODS), "baselines": list(BASELINES), "mae_margin": 1e-5,
        "winning_folds": 2, "near_zero_radius": 1e-5, "maximum_mapper_fits": 150,
        "outer_label_access_before_lock": False, "target_label_access": False,
    }
    changed = [name for name, value in fixed.items() if cfg.get(name) != value]
    if changed:
        raise ValueError(f"Frozen spectral-objective configuration changed: {changed}")


def donor_episode(episode, seed, fold, phase, anchors=None):
    """Materialize v1 role-local donor identities in their recipient order."""
    salt = (seed, fold, phase, episode["draw"])
    roles = {"support": episode["support"], "query": episode["query"]}
    if anchors is not None:
        roles["anchor"] = anchors
    donors = {}
    for role, ids in roles.items():
        donor = rotation.different_descriptors(ids, (salt, role))
        if set(ids) != set(donor) or any(a == b for a, b in zip(ids, donor, strict=True)):
            raise ValueError(f"Invalid role-local descriptor derangement {fold}/{phase}/{role}")
        donors[role] = donor
    return {"draw": episode["draw"], "support": list(episode["support"]), "query": list(episode["query"]), "donors": donors}


def verify_meta_schedule(episodes, fit_ids, protocols, cfg):
    if [ep["draw"] for ep in episodes] != list(range(cfg["train_episodes"])):
        raise ValueError("Meta training draw axis differs")
    physical = []
    for ep in episodes:
        support, query = [set(ep[role]) for role in ("support", "query")]
        groups = {protocols[uid] for uid in support | query}
        if (
            len(support) != cfg["shot"] or len(query) != cfg["train_query_cells"]
            or len(ep["support"]) != len(support) or len(ep["query"]) != len(query)
            or not support | query <= fit_ids or support & query or len(groups) != len(support) + len(query)
        ):
            raise ValueError("A meta episode uses an unregistered cell or overlaps protocols")
        physical.append((tuple(sorted(support)), tuple(sorted(query))))
    return frozenset(physical)


def parent_plan(cfg, parent):
    path = ROOT / cfg["parent_plan"]
    marker = ROOT / cfg["parent_completion"]
    receipt = read_json(marker)
    digest = file_hash(path)
    if (
        receipt["status"] != "COMPLETE_SOURCE_PROTOCOL_SCREEN" or receipt["decision"] != "STOP_SIMPLE_BASELINE"
        or receipt["target_label_rows_loaded"] != 0
        or digest != cfg["parent_plan_sha256"] or digest != receipt["run_identity"]["plan_sha256"]
        or file_hash(Path(rotation.__file__)) != receipt["run_identity"]["code_sha256"]["spectral_rotation_protocol.py"]
    ):
        raise ValueError("V1 evaluation or donor provenance differs from the locked study")
    previous = read_json(path)
    if (
        previous["status"] != "REGISTERED_WITHOUT_OUTER_LABELS" or not previous["historically_used_source_population"]
        or file_hash(ROOT / cfg["source_episodes"]) != previous["parent_episodes_sha256"]
        or previous["outer_source_query_cells"] != 60 or previous["outer_source_query_protocols"] != 39
        or len(parent["source_ids"]) != 115
    ):
        raise ValueError("V1 data roles differ from the source registration")
    return previous, file_hash(marker)


def build_plan(args):
    cfg = read_json(args.config)
    validate_config(cfg)
    source = load_registration(ROOT / cfg["source_episodes"])
    previous, completion_sha = parent_plan(cfg, source)
    protocols = source["source_protocols"]
    source_ids = set(source["source_ids"])
    sealed = set(source["sealed_final_MATR_ids"])
    refs = SimpleNamespace(reference_root=getattr(args, "reference_root", REFERENCES))
    setup = []
    all_query, all_query_protocols = set(), set()
    for fold in cfg["folds"]:
        old = next(item for item in previous["folds"] if item["fold"] == fold)
        ref = checkpoint(refs, fold, cfg["reference_seed"])
        fit_ids, anchors = old["fit_ids"], old["anchors"]
        reference_id = old["reference_sha256"]
        if (reference_id != ref["reference_checkpoint_sha256"] or fit_ids != ref["fit_ids"]
                or len(anchors) != len(set(anchors)) or len(anchors) != cfg["anchor_cells"] or not set(anchors) <= set(fit_ids)):
            raise ValueError("The original reference or fit-only anchors changed")
        expected_validation_draws = (3, 1, 3)[fold]
        if len(old["validation"]) != expected_validation_draws or len(old["outer"]) != cfg["heldout_draws"]:
            raise ValueError("The inherited evaluation draw roster differs")
        evaluations = {}
        for phase, role, number in (("validation", "inner", 8), ("outer", "outer", 20)):
            inherited = task_for(source, role, fold)
            evaluations[phase] = []
            for ep in old[phase]:
                if (ep["query"] != inherited["query_ids"] or len(ep["query"]) != number
                        or not set(ep["support"]) <= set(inherited["support_pool_ids"])):
                    raise ValueError(f"V1 {phase} role changed relative to the original registration")
                evaluations[phase].append(donor_episode(ep, cfg["evaluation_plan_seed"], fold, phase, anchors))
        rotation.validate_fold(fit_ids, evaluations["validation"], evaluations["outer"], protocols, source_ids, sealed)
        query = set(evaluations["outer"][0]["query"])
        query_protocols = {protocols[uid] for uid in query}
        if all_query & query or all_query_protocols & query_protocols:
            raise ValueError("Outer source Query cells or protocols repeat across folds")
        all_query.update(query)
        all_query_protocols.update(query_protocols)
        by_seed, schedules = {}, {}
        for seed in cfg["meta_training_seeds"]:
            # Replaying v1's salt keeps the historical seed42 ordering and donor rule exact.
            episodes = old["train"] if seed == 42 else rotation.train_episodes(fit_ids, protocols, cfg | {"seed": seed}, fold)
            tagged = [donor_episode(ep, seed, fold, "meta") for ep in episodes]
            schedules[seed] = verify_meta_schedule(tagged, set(fit_ids), protocols, cfg)
            by_seed[str(seed)] = tagged
        if any(left == right for left, right in combinations(schedules.values(), 2)):
            raise ValueError("Meta repeat seeds must change physical Support/Query episode membership")
        setup.append({"fold": fold, "reference_seed": cfg["reference_seed"], "reference_sha256": reference_id,
                      "fit_ids": fit_ids, "anchors": anchors, "train_by_seed": by_seed, **evaluations})
    if len(all_query) != 60 or len(all_query_protocols) != 39:
        raise ValueError("Frozen evaluation must cover the same 60 source cells and 39 protocols")
    plan = {
        "status": "REGISTERED_WITHOUT_OUTER_LABELS",
        "config_sha256": file_hash(args.config),
        "parent_plan_sha256": cfg["parent_plan_sha256"],
        "parent_completion_sha256": completion_sha,
        "source_episodes_sha256": file_hash(ROOT / cfg["source_episodes"]),
        "historically_used_source_population": True,
        "reference_seed": cfg["reference_seed"],
        "evaluation_plan_seed": cfg["evaluation_plan_seed"],
        "selection_seed": cfg["selection_seed"],
        "meta_training_seeds": cfg["meta_training_seeds"],
        "outer_source_query_cells": len(all_query),
        "outer_source_query_protocols": len(all_query_protocols),
        "target_future_label_reads": 0,
        "source_outer_future_label_reads": 0,
        "folds": setup,
    }
    return cfg, plan, protocols, source_ids, sealed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--reference-root", type=Path, default=REFERENCES)
    args = parser.parse_args()
    _, plan, _, _, _ = build_plan(args)
    bind(args.public / "REGISTERED_EPISODES.json", plan)
    print(f"Registered {plan['outer_source_query_cells']} unique MATR source Query cells.")


if __name__ == "__main__":
    main()
