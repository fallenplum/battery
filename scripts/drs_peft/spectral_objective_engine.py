"""Execute and authenticate the registered source-only mapper and baseline trials."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.drs_peft.data import file_hash, write_json
from scripts.drs_peft.run_spectral_rotation import (
    frozen_inputs,
    initial_audit,
    model_accounting,
)
from scripts.drs_peft.spectral_objective import fit_mapper
from scripts.drs_peft.spectral_objective_protocol import (
    CANDIDATE,
    META_METHODS,
    META_OBJECTIVES,
)
from scripts.drs_peft.spectral_rotation_engine import (
    Episode,
    core_model,
    fit_cal2,
    fit_core,
    frozen_mapper,
    lora_operator,
    lora_snapshots,
    predict_arrays,
    sampled,
)
from scripts.drs_peft.train import read_json, save_torch
from scripts.drs_peft.transfer_scope_reference import save_npz

V1 = "spectral_rotation_v1"
BASELINES = ("K0", "STATIC_CORE", "LORA_SMALL", "CAL2")


@dataclass(frozen=True)
class MapperTrial:
    method: str
    objective: str
    alpha: float
    training_seed: int

    @property
    def name(self):
        return f"{self.method}_{self.objective}"

    @property
    def relative(self):
        return Path(f"seed{self.training_seed}") / self.name / f"alpha{self.alpha:.9g}"

    def metadata(self):
        return {
            "method": self.method,
            "objective": self.objective,
            "model": self.name,
            "alpha": self.alpha,
            "training_seed": self.training_seed,
        }


def prepare_episodes(cfg, data, labels, plans, anchors, reference, phi, device):
    episodes = []
    for plan in plans:
        support, query = plan["support"], plan["query"]
        uids = support + query + anchors
        donors = [
            uid
            for role in ("support", "query", "anchor")
            for uid in plan["donors"][role]
        ]
        if len(donors) != len(uids):
            raise RuntimeError("The frozen descriptor permutation changed")
        prediction, seconds = predict_arrays(reference, data, uids, device)
        sy, sm = labels.arrays(support)
        qy, qm = labels.arrays(query)
        episodes.append(
            Episode(
                support,
                query,
                anchors,
                plan["draw"],
                sampled(data, labels, support, cfg["support_max_observed"], device),
                sy,
                sm,
                qy,
                qm,
                prediction,
                {
                    "true": phi.tensor(uids, device),
                    "shuffled": phi.tensor(donors, device),
                },
                seconds,
            )
        )
    return episodes


def fold_inputs(args, cfg, fold, data, features, *, check_reference=False):
    original, reference, bases, phi, audit = frozen_inputs(
        args, fold, features, data, {"seed": cfg["reference_seed"]}
    )
    if check_reference:
        reference.to(args.device)
        audit["original_vs_merged_zero_max"] = initial_audit(
            original, reference, data, fold, args.device
        )
        reference.cpu()
        root = args.public / "stage_a" / f"fold{fold['fold']}"
        write_json(root / "BASIS_AUDIT.json", audit)
        write_json(root / "DESCRIPTOR_SCALER.json", phi.audit)
    return reference, bases, phi


def mapper_trial_path(private, fold, trial):
    return private / "mapper_trials" / f"fold{fold}" / trial.relative


def authenticate_trial(folder, expected):
    receipt = read_json(folder / "COMPLETE.json")
    if receipt["identity"] != expected or receipt["status"] != "COMPLETE_MAPPER":
        raise RuntimeError(f"Mapper identity differs at {folder}")
    if (
        receipt["cost"]["source_mapping_updates"] != 32
        or receipt["cost"]["source_inner_updates"] != 96
    ):
        raise RuntimeError(f"Mapper update counts differ at {folder}")
    for name, sha in receipt["files"].items():
        if file_hash(folder / name) != sha:
            raise RuntimeError(f"Mapper artifact changed: {folder / name}")
    return receipt


def trial_identity(trial, fold_plan, identity_sha, stage):
    return trial.metadata() | {
        "fold": fold_plan["fold"],
        "identity_sha256": identity_sha,
        "stage": stage,
        "reference_sha256": fold_plan["reference_sha256"],
    }


def fit_mapper_trial(
    args, cfg, fold_plan, fold_data, trial, identity_sha, *, validation=None
):
    fold = fold_plan["fold"]
    reference, bases, phi, data, labels = fold_data
    stage = "A" if trial.training_seed == cfg["selection_seed"] else "B"
    if (stage == "A") != (validation is not None):
        raise RuntimeError("Only seed42 Stage A mappers may run inner validation")
    expected = trial_identity(trial, fold_plan, identity_sha, stage)
    folder = mapper_trial_path(args.private, fold, trial)
    if (folder / "COMPLETE.json").exists():
        return authenticate_trial(folder, expected)

    model = core_model(reference, bases, trial.method, phi, cfg, args.device)
    budget = model_accounting(model)
    trace, cost = fit_mapper(
        model,
        trial.method,
        fold_plan["train_by_seed"][str(trial.training_seed)],
        data,
        labels,
        phi,
        cfg,
        fold,
        args.device,
        objective=trial.objective,
        core_lr=trial.alpha,
        training_seed=trial.training_seed,
    )
    if not trace or any(
        row["objective"] != trial.objective or row["alpha"] != trial.alpha
        for row in trace
    ):
        raise RuntimeError(
            "Mapper trace does not confirm the registered objective and rate"
        )
    save_torch(
        folder / "MAPPER.pt",
        {
            "identity": expected,
            "controller": {
                key: value.detach().cpu()
                for key, value in model.controller.state_dict().items()
            },
        },
    )
    write_json(folder / "META_TRACE.json", {"rows": trace})
    files = ["MAPPER.pt", "META_TRACE.json"]
    rows = []
    if validation is not None:
        frozen_mapper(model)
        predictions = []
        for ep in validation:
            _, prediction, adapt = fit_core(
                model, ep, data, trial.method, cfg, trial.alpha, args.device
            )
            rows.append(validation_fact(expected, ep, prediction, adapt))
            predictions.append(prediction)
        save_validation(folder, validation, predictions)
        files.append("PREDICTIONS.npz")
    write_json(
        folder / "COMPLETE.json",
        {
            "identity": expected,
            "status": "COMPLETE_MAPPER",
            "cost": budget | cost,
            "validation": rows,
            "files": {name: file_hash(folder / name) for name in files},
        },
    )
    print(
        f"{stage} fold={fold} seed={trial.training_seed} {trial.name} alpha={trial.alpha:.4g}",
        flush=True,
    )
    return authenticate_trial(folder, expected)


def load_mapper(model, path, expected):
    state = torch.load(path / "MAPPER.pt", map_location="cpu", weights_only=True)
    if state["identity"] != expected:
        raise RuntimeError(
            "The controller would be deployed at a different objective/rate"
        )
    model.controller.load_state_dict(state["controller"])
    frozen_mapper(model)


def validation_fact(metadata, episode, prediction, cost):
    return metadata | {"draw": episode.draw} | cost | episode.scores(prediction)


def save_validation(folder, episodes, predictions):
    save_npz(
        folder / "PREDICTIONS.npz",
        ids=np.array([ep.ids for ep in episodes]),
        predictions=np.stack(predictions),
        draws=np.array([ep.draw for ep in episodes]),
    )


def baseline_key(method, setting):
    return "__".join(
        [method] + [f"{key}_{value:.9g}" for key, value in setting.items()]
    )


def baseline_grid(cfg):
    yield "K0", {}
    for rate in cfg["core_rates"]:
        yield "STATIC_CORE", {"alpha": rate}
    for rate, steps in itertools.product(cfg["lora_rates"], cfg["lora_steps"]):
        yield "LORA_SMALL", {"steps": steps, "lr": rate}
    for ridge, eta in itertools.product(cfg["cal2_ridge"], cfg["cal2_eta"]):
        yield "CAL2", {"eta": eta, "ridge": ridge}


def baseline_trials(args, cfg, fold, fold_data, validation, lora_parent, identity_sha):
    reference, bases, phi, data, _ = fold_data
    seed = cfg["selection_seed"]
    outputs = {
        baseline_key(method, setting): (method, setting, [], [])
        for method, setting in baseline_grid(cfg)
    }

    def record(method, setting, ep, prediction, cost):
        _, _, rows, predictions = outputs[baseline_key(method, setting)]
        common = {
            "fold": fold,
            "method": method,
            "model": method,
            "objective": "BASELINE",
            "training_seed": seed,
            **setting,
        }
        rows.append(validation_fact(common, ep, prediction, cost))
        predictions.append(prediction)

    static = core_model(reference, bases, "STATIC_CORE", phi, cfg, args.device)
    budget = model_accounting(static)
    for ep in validation:
        record(
            "K0",
            {},
            ep,
            ep.baseline,
            {
                "target_parameters": 0,
                "source_mapper_parameters": 0,
                "basis_storage_bytes": 0,
                "adapt_seconds": 0.0,
                "adapt_updates": 0,
                "inference_seconds": ep.base_predict_seconds,
            },
        )
        for rate in cfg["core_rates"]:
            _, prediction, cost = fit_core(
                static, ep, data, "STATIC_CORE", cfg, rate, args.device
            )
            record("STATIC_CORE", {"alpha": rate}, ep, prediction, budget | cost)
        for ridge, eta in itertools.product(cfg["cal2_ridge"], cfg["cal2_eta"]):
            prediction, cost = fit_cal2(ep, data, ridge, eta)
            record("CAL2", {"eta": eta, "ridge": ridge}, ep, prediction, cost)
    del static
    lora = lora_operator(reference, lora_parent, {"seed": seed}, args.device)
    for ep, rate in itertools.product(validation, cfg["lora_rates"]):
        for step, prediction, cost in lora_snapshots(
            lora, ep, data, args.device, rate, cfg["lora_steps"]
        ):
            record("LORA_SMALL", {"steps": step, "lr": rate}, ep, prediction, cost)
    summaries = []
    for key, (method, setting, rows, predictions) in outputs.items():
        folder = args.private / "baseline_trials" / f"fold{fold}" / key
        save_validation(folder, validation, predictions)
        summaries.append(
            {
                "identity": {
                    "fold": fold,
                    "identity_sha256": identity_sha,
                    "method": method,
                    "model": method,
                    "setting": setting,
                    "training_seed": seed,
                },
                "validation": rows,
                "files": {"PREDICTIONS.npz": file_hash(folder / "PREDICTIONS.npz")},
            }
        )
    return summaries


def trial_entry(private, folder, receipt):
    return {
        "path": folder.relative_to(private).as_posix(),
        "identity": receipt["identity"],
        "files": receipt["files"],
        "cost": receipt.get("cost"),
    }


def validation_selection(cfg, rows, fold):
    selected = {}
    by_key = {}
    for row in rows:
        model = row["model"]
        axes = {
            "K0": (),
            "STATIC_CORE": ("alpha",),
            "LORA_SMALL": ("steps", "lr"),
            "CAL2": ("eta", "ridge"),
        }.get(model, ("alpha",))
        key = tuple((name, row[name]) for name in axes)
        by_key.setdefault(model, {}).setdefault(key, []).append(row)
    expected = {
        "K0": 1,
        "STATIC_CORE": len(cfg["core_rates"]),
        "LORA_SMALL": len(cfg["lora_rates"]) * len(cfg["lora_steps"]),
        "CAL2": len(cfg["cal2_ridge"]) * len(cfg["cal2_eta"]),
    }
    expected.update(
        {
            f"{method}_{objective}": len(cfg["core_rates"])
            for method, objective in itertools.product(META_METHODS, META_OBJECTIVES)
        }
    )
    if set(by_key) != set(expected):
        raise RuntimeError(f"Incomplete validation methods, fold {fold}")
    n_draws = (3, 1, 3)[fold]
    for model, groups in by_key.items():
        if len(groups) != expected[model] or any(
            sorted(row["draw"] for row in group) != list(range(n_draws))
            for group in groups.values()
        ):
            raise RuntimeError(
                f"Incomplete or repeated validation grid: {model}, fold {fold}"
            )
        means = {
            key: math.fsum(row["query_mae"] for row in group) / n_draws
            for key, group in groups.items()
        }
        key = min(means, key=lambda setting: (means[setting], setting))
        selected[model] = dict(key) | {"validation_mae": means[key]}
    return selected


def stage_a_decision(cfg, selected):
    checks = []
    for fold in cfg["folds"]:
        best = selected[str(fold)][CANDIDATE]
        base = selected[str(fold)]["K0"]["validation_mae"]
        checks.append(
            {
                "fold": fold,
                "k0_mae": base,
                "candidate_mae": best["validation_mae"],
                "alpha": best["alpha"],
                "delta_vs_k0": best["validation_mae"] - base,
                "passed": best["validation_mae"] <= base - cfg["mae_margin"],
            }
        )
    passed = sum(row["passed"] for row in checks) >= 2
    return {
        "status": "PASS_STAGE_A" if passed else "STOP_NO_VALIDATION_SIGNAL",
        "primary_candidate": CANDIDATE,
        "mae_margin": cfg["mae_margin"],
        "validation_folds": checks,
        "outer_role_labels_read": 0,
        "target_labels_read": 0,
        "remaining_stages": "PENDING" if passed else "NOT_RUN_BY_STAGE_A_GATE",
    }
