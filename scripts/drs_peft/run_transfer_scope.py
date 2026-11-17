"""Run V2 transfer-scope calibration on registered physical-cell episodes."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.drs_peft.data import CACHE, ROOT, file_hash, write_json
from scripts.drs_peft.onesided_calibration import observed_positions
from scripts.drs_peft.onesided_engine import _fingerprint
from scripts.drs_peft.onesided_protocol import load_registration, payload_hash
from scripts.drs_peft.run_onesided import shot_for, task_for
from scripts.drs_peft.specialization_protocol import load_target_contract
from scripts.drs_peft.train import bind, configure, read_json
from scripts.drs_peft.transfer_scope_data import load_future_rows, load_prefixes
from scripts.drs_peft.transfer_scope_geometry import (
    calibration_design,
    cell_normal_equations,
    neighbor_statistics,
    solve_calibration,
)
from scripts.drs_peft.transfer_scope_reference import (
    BACKBONE,
    REFERENCES,
    Reference,
    old_zero,
    save_npz,
)

CONFIG = ROOT / "configs/drs_peft/transfer_scope_v2/CONFIG.json"
PUBLIC = ROOT / "results/drs_peft/transfer_scope_v2"
PRIVATE = ROOT / "runtime/drs_peft/transfer_scope_v2"
LORA = ("PRETRAINED_LORA_ZERO_RETAINING", "SPECIALIZED_LORA_ZERO_RETAINING")


def contract(args):
    path = Path(getattr(args, "config", None) or CONFIG)
    cfg = read_json(path)
    if file_hash(ROOT / cfg["protocol"]) != cfg["protocol_sha256"]:
        raise RuntimeError("V2 protocol changed")
    source = load_registration(ROOT / cfg["original_episodes"])
    _, _, target = load_target_contract(ROOT / cfg["target_contract"])
    if target["training_seeds"] != cfg["seeds"] or target["shots"] != cfg["shots"]:
        raise RuntimeError("Inherited model seeds or K changed")
    return cfg, source, target


def draws(task):
    for k in (1, 3, 5):
        batches = shot_for(task, k)["draws"]
        if sorted(row["draw"] for row in batches) != list(range(5)):
            raise RuntimeError("Expected five registered draws")
        for row in batches:
            ids = row["support_ids"]
            if (
                len(ids) != k
                or len(set(ids)) != k
                or set(ids) & set(task["query_ids"])
                or not set(ids) <= set(task["support_pool_ids"])
                or row["query_ids"] != task["query_ids"]
            ):
                raise RuntimeError("Physical Support/Query episode changed")
            yield k, row


def fit_record(args, source, ref, prefixes):
    folds = read_json(ROOT / "configs/drs_peft/e123_folds_v1.json")["folds"]
    expected = (
        source["source_ids"] if ref.fold == "final" else folds[int(ref.fold)]["fit_ids"]
    )
    if sorted(expected) != sorted(ref.identity["fit_ids"]):
        raise RuntimeError("Source fit identities differ from the reference")
    bind(
        args.public_root / "fit" / f"f{ref.fold}s{ref.seed}.json",
        ref.freeze_fit(prefixes),
    )


class Episode:
    def __init__(self, args, ref, task, target=False):
        self.ref, self.task = ref, task
        self.q = task["query_ids"]
        self.draws = list(draws(task))
        self.support = sorted(
            {uid for _, row in self.draws for uid in row["support_ids"]}
        )
        prefixes = ref.prefixes
        self.grid = prefixes.query
        uids = self.support + self.q
        embedding, base = ref.embedding_and_base(uids, predict=not target)
        self.z = dict(zip(uids, embedding, strict=True))
        if target:
            self.base, receipt = old_zero(task, ref.seed)
            if receipt["source_checkpoint_sha256"] != ref.checkpoint_sha:
                raise RuntimeError("Target ZERO and active reference differ")
            self.zero_sha = receipt["prediction_sha256"]
        else:
            self.base = base[len(self.support) :]
            self.zero_sha = None
        self.b = calibration_design(
            self.base, prefixes.anchor[prefixes.rows(self.q)], self.grid
        )
        self.args, self.domain, self.target = args, prefixes.domain, target
        targets, masks = load_future_rows(
            prefixes.domain, args.data_root, self.support, load_targets=not target
        )
        positions = [observed_positions(mask) for mask in masks]
        cached = ref.designs(self.support, positions)
        self.pos, self.y, self.f0, self.a, self.gram, self.rhs = {}, {}, {}, {}, {}, {}
        anchors = prefixes.anchor[prefixes.rows(self.support)]
        for i, (uid, pos) in enumerate(zip(self.support, positions, strict=True)):
            f0, jac = cached[uid]
            b = calibration_design(f0, anchors[i], self.grid[pos])
            a = np.column_stack((b, jac))
            self.pos[uid], self.f0[uid], self.a[uid] = pos, f0, a
        if not target:
            self.load_support_residuals(self.support, targets)
        self.jac = None
        self.solutions = {}

    def load_support_residuals(self, ids, targets=None):
        self.y.clear()
        self.gram.clear()
        self.rhs.clear()
        if targets is None:
            targets, _ = load_future_rows(self.domain, self.args.data_root, ids)
        for uid, target in zip(ids, targets, strict=True):
            y = target[self.pos[uid]].astype(np.float64)
            self.y[uid] = y
            self.gram[uid], self.rhs[uid] = cell_normal_equations(
                self.a[uid], y - self.f0[uid]
            )

    def solve(self, ids, method, hyper, cfg):
        ids = sorted(ids)
        options = cfg["methods"][method]
        d = options["dimension"]
        zs, zq = (np.stack([self.z[uid] for uid in names]) for names in (ids, self.q))
        distance = self.ref.geometry.squared_distances(zs, zq)
        weights = (
            self.ref.geometry.weights(zs, zq, hyper["tau_ratio"])
            if options["local"]
            else np.ones((1, len(ids)), dtype=np.float64)
        )
        key = (tuple(ids), method, hyper["lambda_multiplier"], hyper["tau_ratio"])
        eta = hyper["eta"]
        if eta and key not in self.solutions:
            self.solutions[key] = solve_calibration(
                np.stack([self.gram[uid][:d, :d] for uid in ids]),
                np.stack([self.rhs[uid][:d] for uid in ids]),
                weights,
                lambda_multiplier=hyper["lambda_multiplier"],
                scales=self.ref.scales,
                eta=1.0,
                support_ids=ids,
            )
            self.ref.cost["ridge_solves"] += len(weights)
        coeff = (
            eta * self.solutions[key]
            if eta
            else np.zeros((len(weights), d), dtype=np.float64)
        )
        return coeff, weights, distance, ids

    def linear(self, coeff):
        c = np.broadcast_to(coeff, (len(self.q), coeff.shape[1]))
        update = np.einsum("ntd,nd->nt", self.b, c[:, :2])
        if c.shape[1] == 18 and c[:, 2:].any():
            if self.jac is None:
                values = self.ref.designs(
                    self.q, [np.arange(len(self.grid))] * len(self.q)
                )
                self.jac = np.stack([values[uid][1] for uid in self.q]).astype(
                    np.float64
                )
            update += np.einsum("ntd,nd->nt", self.jac, c[:, 2:])
        return self.base.astype(np.float64) + update

    def actual(self, coeff):
        return self.ref.actual(self.q, self.base, self.b, coeff)

    def support_error(self, ids, coeff):
        result = []
        model = (
            self.ref.engine._active("OUT")
            if len(coeff) == 18 and coeff[2:].any()
            else None
        )
        for uid in ids:
            f0, a, y = self.f0[uid], self.a[uid][:, : len(coeff)], self.y[uid]
            proxy = np.mean((y - f0 - a @ coeff) ** 2)
            if model is None:
                actual = proxy
            else:
                early = self.ref.engine._early([uid])
                cycles = torch.as_tensor(
                    self.grid[self.pos[uid]], device=self.ref.args.device
                )
                coordinates = torch.as_tensor(
                    coeff[2:], dtype=model.coordinates.dtype, device=cycles.device
                )
                with torch.no_grad():
                    pred = (
                        model(early, cycles, coordinates=coordinates).cpu().numpy()[0]
                    )
                actual = np.mean((y - pred - a[:, :2] @ coeff[:2]) ** 2)
                self.ref.cost["nonlinear_support_cell_forwards"] += 1
            result.append((np.mean((y - f0) ** 2), proxy, actual))
        return np.mean(result, axis=0)


def csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pending.csv")
    pd.DataFrame(records).to_csv(temp, index=False)
    temp.replace(path)


def grid(cfg, method):
    rows = [
        dict(eta=0.0, lambda_multiplier=max(cfg["ridge_multipliers"]), tau_ratio=1.0)
    ]
    for eta, ridge, tau in itertools.product(
        [eta for eta in cfg["etas"] if eta],
        cfg["ridge_multipliers"],
        cfg["bandwidth_ratios"] if cfg["methods"][method]["local"] else [1.0],
    ):
        rows.append(dict(eta=eta, lambda_multiplier=ridge, tau_ratio=tau))
    return rows


def source_folder(args, fold, seed):
    return args.public_root / "source" / f"f{fold}s{seed}"


def source_run(args, cfg, source, nonlinear=False):
    prefixes = load_prefixes("MATR", args.data_root)
    shortlist = (
        read_json(args.public_root / "SOURCE_SHORTLIST.json") if nonlinear else None
    )
    phase = "NONLINEAR" if nonlinear else "SCREEN"
    for fold, seed in itertools.product(args.folds, args.seeds):
        folder = source_folder(args, fold, seed)
        path, marker = folder / f"{phase}.csv", folder / f"{phase}_RECEIPT.json"
        if marker.exists():
            if read_json(marker)["sha256"] != file_hash(path):
                raise RuntimeError("Source scoring artifact changed")
            continue
        ref = Reference(args, fold, seed)
        fit_record(args, source, ref, prefixes)
        ref.set_data(prefixes)
        task = task_for(source, "inner", fold)
        if set(task["query_ids"]) & set(ref.identity["fit_ids"]):
            raise RuntimeError("Inner Query leaked into fit statistics")
        episode = Episode(args, ref, task)
        y, obs = load_future_rows("MATR", args.data_root, episode.q)
        if not obs.any(axis=1).all():
            raise RuntimeError("Inner Query without any scoreable observations")
        rows, cache = [], {}
        for k, draw in episode.draws:
            for method in cfg["methods"]:
                candidates = (
                    shortlist["parameters"][method][str(k)]
                    if nonlinear
                    else grid(cfg, method)
                )
                for hyper in candidates:
                    key = (
                        tuple(sorted(draw["support_ids"])),
                        method,
                        tuple(sorted(hyper.items())),
                    )
                    if key not in cache:
                        coeff, _, _, _ = episode.solve(
                            draw["support_ids"], method, hyper, cfg
                        )
                        prediction = (
                            episode.actual(coeff)
                            if nonlinear
                            else episode.linear(coeff)
                        )
                        cache[key] = float(
                            np.mean(
                                [
                                    np.abs(p[m] - t[m]).mean()
                                    for p, t, m in zip(prediction, y, obs, strict=True)
                                ]
                            )
                        )
                    rows.append(
                        dict(
                            fold=fold,
                            seed=seed,
                            K=k,
                            method=method,
                            draw=draw["draw"],
                            mae=cache[key],
                            **hyper,
                        )
                    )
        csv(path, rows)
        write_json(
            marker,
            dict(
                sha256=file_hash(path),
                rows=len(rows),
                costs=ref.cost,
                parent_jacobians=ref.reused,
            ),
        )
        print(
            f"source {phase} fold={fold} seed={seed} rows={len(rows)} builds={ref.cost['jacobian_builds']}",
            flush=True,
        )
        del episode, ref


def select(args, cfg, shortlist):
    phase = "SCREEN" if shortlist else "NONLINEAR"
    tables = []
    for fold, seed in itertools.product(cfg["source_folds"], cfg["seeds"]):
        folder = source_folder(args, fold, seed)
        path = folder / f"{phase}.csv"
        if read_json(folder / f"{phase}_RECEIPT.json")["sha256"] != file_hash(path):
            raise RuntimeError("Source score hash changed")
        tables.append(pd.read_csv(path))
    table = pd.concat(tables, ignore_index=True)
    axes = ["method", "K", "eta", "lambda_multiplier", "tau_ratio"]
    if table.duplicated(axes + ["fold", "seed", "draw"]).any():
        raise RuntimeError("Duplicated source trial")
    aggregate = (
        table.groupby(axes)
        .mae.agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "mae"})
    )
    if not aggregate["size"].eq(45).all():
        raise RuntimeError(
            "All source candidates require three folds, three seeds and five draws"
        )
    aggregate_path = args.public_root / f"SOURCE_{phase}_AGGREGATE.csv"
    csv(aggregate_path, aggregate)
    parameters = {}
    for method in cfg["methods"]:
        parameters[method] = {}
        for k in (1, 3, 5):
            rows = aggregate[(aggregate.method == method) & (aggregate.K == k)]
            rows = rows.sort_values(
                ["mae", "eta", "lambda_multiplier", "tau_ratio"],
                ascending=[True, True, False, True],
            )
            expected = len(grid(cfg, method)) if shortlist else 4
            if len(rows) != expected:
                raise RuntimeError(
                    "Source selection does not cover its registered grid"
                )
            if shortlist:
                rows = pd.concat([rows[rows.eta.gt(0)].head(3), rows[rows.eta.eq(0)]])
            candidates = rows[["eta", "lambda_multiplier", "tau_ratio"]].to_dict(
                "records"
            )
            parameters[method][str(k)] = candidates if shortlist else candidates[0]
    result = dict(
        status="COMPLETE_SOURCE_ONLY",
        parameters=parameters,
        phase=phase,
        target_labels_read=0,
        candidate_scores_sha256=file_hash(aggregate_path),
    )
    if not shortlist:
        result.update(
            selected_hyperparameters=parameters,
            source_shortlist_sha256=file_hash(
                args.public_root / "SOURCE_SHORTLIST.json"
            ),
        )
    bind(
        args.public_root
        / ("SOURCE_SHORTLIST.json" if shortlist else "SOURCE_SELECTION.json"),
        result,
    )


def lock_payload(args, target):
    root = args.public_root
    paths = [
        root / name for name in ("SOURCE_SELECTION.json", "LORA_SOURCE_SELECTION.json")
    ]
    fits = sorted((root / "fit").glob("*.json"))
    expected_fits = {
        f"f{fold}s{seed}.json"
        for fold, seed in itertools.product(
            (0, 1, 2, "final"), target["training_seeds"]
        )
    }
    if {path.name for path in fits} != expected_fits:
        raise RuntimeError(
            "Target requires all nine fold and three final-source scale records"
        )
    code = (
        "run_transfer_scope.py",
        "transfer_scope_reference.py",
        "transfer_scope_data.py",
        "transfer_scope_geometry.py",
        "transfer_scope_lora.py",
        "transfer_scope_scoring.py",
    )
    return dict(
        status="TOTAL_K_LOCKED_BEFORE_TARGET_SCORING",
        config_sha256=file_hash(Path(getattr(args, "config", None) or CONFIG)),
        code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in code},
        source_selection_shas={path.name: file_hash(path) for path in paths},
        fit_sha256={path.name: file_hash(path) for path in fits},
        episode_contract_sha256=target["episode_contract_sha256"],
        target_tasks=target["target_tasks"],
        support_budget="strict_domain_total_K",
        historical_target_query_exposure=True,
        target_selection_uses=0,
        selected_hyperparameters={
            k: v
            for path in paths
            for k, v in read_json(path)["selected_hyperparameters"].items()
        },
    )


def read_target_lock(args):
    cfg, _, target = contract(args)
    record = read_json(args.public_root / "TARGET_TOTAL_K_LOCK.json")
    if record != lock_payload(args, target) or set(
        record["selected_hyperparameters"]
    ) != set(cfg["methods"]) | set(LORA):
        raise RuntimeError("Target execution no longer matches the locked scope")
    return record


def fit_final(args, source):
    prefixes = load_prefixes("MATR", args.data_root)
    for seed in args.seeds:
        folder = args.public_root / "fit_cost" / f"s{seed}"
        if (folder / "COST.json").exists():
            continue
        ref = Reference(args, "final", seed)
        fit_record(args, source, ref, prefixes)
        write_json(folder / "COST.json", dict(ref.cost, parent_jacobians=ref.reused))
        print(
            f"final source fit seed={seed} builds={ref.cost['jacobian_builds']}",
            flush=True,
        )
        del ref


def target_plan(args, cfg, target):
    return [
        (args.public_root / "target" / task["domain"] / f"s{seed}", task, seed)
        for task, seed in itertools.product(target["target_tasks"], cfg["seeds"])
    ]


def predict(args, cfg, target):
    lock = read_target_lock(args)
    lock_sha = file_hash(args.public_root / "TARGET_TOTAL_K_LOCK.json")
    for folder, task, seed in target_plan(args, cfg, target):
        domain = task["domain"]
        marker = folder / "PREDICTIONS_LOCKED.json"
        if domain not in args.domains or seed not in args.seeds or marker.exists():
            continue
        ref = Reference(args, "final", seed)
        fitted = ref.freeze_fit(load_prefixes("MATR", args.data_root))
        if fitted != read_json(args.public_root / "fit" / f"ffinals{seed}.json"):
            raise RuntimeError(
                "Private calibrated scales differ from the locked source records"
            )
        ref.set_data(load_prefixes(domain, args.data_root))
        epi = Episode(args, ref, task, target=True)
        _, replay = ref.embedding_and_base(epi.q)
        if not np.allclose(replay, epi.base, atol=2e-6, rtol=0):
            raise RuntimeError("Active source f0 differs from parent ZERO")
        model = ref.engine._active("OUT")
        early = ref.engine._early(epi.q[:1])
        with torch.no_grad():
            anchor = model(
                early,
                torch.tensor([100.0], device=args.device),
                coordinates=torch.ones_like(model.coordinates),
            )
        if not torch.equal(anchor[:, 0], early["anchor"]):
            raise RuntimeError("Nonlinear OUT changed the early anchor")
        cache, diagnostics, slots = {}, [], []
        ref.cost["nonlinear_replay_cell_forwards"] = 0
        ref.cost["prediction_wall_seconds"] = 0.0
        replay_errors = {}
        for k, draw in epi.draws:
            epi.load_support_residuals(sorted(draw["support_ids"]))
            for method in cfg["methods"]:
                begin = time.perf_counter()
                hyper = lock["selected_hyperparameters"][method][str(k)]
                coeff, weights, distance, ids = epi.solve(
                    draw["support_ids"], method, hyper, cfg
                )
                identity = dict(
                    lock_sha256=lock_sha,
                    domain=domain,
                    seed=seed,
                    method=method,
                    K=k,
                    checkpoint_sha256=ref.checkpoint_sha,
                    dictionary_sha256=ref.dictionary_sha,
                    support_ids=ids,
                    query_ids=epi.q,
                    hyperparameters=hyper,
                    query_prefixes=[ref.engine._prefix_sha(uid) for uid in epi.q],
                    weights_sha256=_fingerprint(weights),
                    parent_zero_sha256=epi.zero_sha,
                )
                state_id = payload_hash(identity)
                cache_hit = state_id in cache
                if state_id not in cache:
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    predict_started = time.perf_counter()
                    prediction = epi.actual(coeff)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    predict_seconds = time.perf_counter() - predict_started
                    ref.cost["prediction_wall_seconds"] += predict_seconds
                    linear = epi.linear(coeff)
                    if (
                        not np.isfinite(prediction).all()
                        or prediction.shape != epi.base.shape
                    ):
                        raise RuntimeError("Invalid nonlinear prediction")
                    if not hyper["eta"] and not np.array_equal(epi.base, prediction):
                        raise RuntimeError(
                            "Eta zero must recover the exact parent ZERO"
                        )
                    gap, shift = (
                        np.abs(prediction - linear),
                        np.abs(prediction - epi.base),
                    )
                    error = np.stack([epi.support_error(ids, x) for x in coeff])
                    private = args.private_root / "target" / state_id[:24]
                    payload = private / "PREDICTION.npz"
                    save_npz(
                        payload,
                        ids=np.asarray(epi.q),
                        grid=epi.grid,
                        prediction=prediction,
                        gap=gap,
                        coefficients=coeff,
                        weights=weights,
                    )
                    sha = file_hash(payload)
                    write_json(
                        private / "STATE.json",
                        dict(identity=identity, state_id=state_id, sha256=sha),
                    )
                    if method not in replay_errors and coeff.any():
                        with np.load(payload, allow_pickle=False) as archive:
                            restored = archive["coefficients"]
                        before = ref.cost["nonlinear_query_cell_forwards"]
                        reproduced = epi.actual(restored)
                        added = ref.cost["nonlinear_query_cell_forwards"] - before
                        ref.cost["nonlinear_query_cell_forwards"] = before
                        ref.cost["nonlinear_replay_cell_forwards"] += added
                        difference = float(np.abs(reproduced - prediction).max())
                        if difference > 2e-6:
                            raise RuntimeError(
                                "Saved target coordinates failed nonlinear replay"
                            )
                        replay_errors[method] = difference
                    cache[state_id] = payload, sha, gap, shift, error, predict_seconds
                payload, sha, gap, shift, error, predict_seconds = cache[state_id]
                slots.append(
                    dict(
                        method=method,
                        K=k,
                        draw=draw["draw"],
                        support_ids=ids,
                        is_duplicate=draw.get("duplicate_of_draw") is not None,
                        adaptation_id=state_id,
                        path=payload.relative_to(ROOT).as_posix(),
                        sha256=sha,
                        coefficient_states=len(coeff),
                        active_nonzero_coefficient_states=int(
                            np.any(coeff != 0, axis=1).sum()
                        ),
                        prediction_cache_hit=cache_hit,
                        prediction_seconds=0.0 if cache_hit else predict_seconds,
                        seconds=time.perf_counter() - begin,
                    )
                )
                stats = neighbor_statistics(weights)
                for i, uid in enumerate(epi.q):
                    j = i if len(weights) > 1 else 0
                    diagnostics.append(
                        dict(
                            domain=domain,
                            seed=seed,
                            method=method,
                            K=k,
                            draw=draw["draw"],
                            cell_uid=uid,
                            **hyper,
                            **{name: value[j] for name, value in stats.items()},
                            distance_d0=ref.geometry.d0,
                            mean_distance=float(np.sqrt(distance[i]).mean()),
                            min_distance=float(np.sqrt(distance[i]).min()),
                            calibration_norm=float(np.linalg.norm(coeff[j, :2])),
                            out_norm=float(np.linalg.norm(coeff[j, 2:])),
                            linearization_gap_mae_full_grid=float(gap[i].mean()),
                            actual_prediction_shift_mae_full_grid=float(
                                shift[i].mean()
                            ),
                            support_mse_before=error[j, 0],
                            support_proxy_mse_after=error[j, 1],
                            support_mse_after=error[j, 2],
                        )
                    )
        diagnostic_path = folder / "DIAGNOSTICS_PREFIX_ONLY.csv"
        csv(diagnostic_path, diagnostics)
        write_json(
            folder / "PREDICT_COST.json",
            dict(
                ref.cost,
                parent_jacobians=ref.reused,
                request_slots=len(slots),
                unique_episode_states=len(cache),
            ),
        )
        write_json(
            marker,
            dict(
                status="PREDICTIONS_LOCKED",
                domain=domain,
                seed=seed,
                task_id=task["task_id"],
                lock_sha256=lock_sha,
                query_ids=epi.q,
                slots=slots,
                diagnostic_sha256=file_hash(diagnostic_path),
                saved_coordinate_replay_max_abs_by_method=replay_errors,
                target_query_future_rows_loaded=0,
                target_support_rows_decoded_per_draw=[k for k, _ in epi.draws],
            ),
        )
        print(f"target predicted {domain} seed={seed} slots={len(slots)}", flush=True)
        del epi, ref


def validated_predictions(args, cfg, target):
    lock_sha = file_hash(args.public_root / "TARGET_TOTAL_K_LOCK.json")
    records = []
    for folder, task, seed in target_plan(args, cfg, target):
        record = read_json(folder / "PREDICTIONS_LOCKED.json")
        expected = {
            (k, row["draw"], method): sorted(row["support_ids"])
            for k, row in draws(task)
            for method in cfg["methods"]
        }
        actual = {
            (row["K"], row["draw"], row["method"]): row["support_ids"]
            for row in record["slots"]
        }
        if (
            record["lock_sha256"] != lock_sha
            or record["query_ids"] != task["query_ids"]
            or record["seed"] != seed
            or len(record["slots"]) != 60
            or actual != expected
            or record["diagnostic_sha256"]
            != file_hash(folder / "DIAGNOSTICS_PREFIX_ONLY.csv")
        ):
            raise RuntimeError("Target predictions lack locked geometry coverage")
        for slot in record["slots"]:
            path = ROOT / slot["path"]
            state = read_json(path.with_name("STATE.json"))
            if (
                file_hash(path) != slot["sha256"]
                or state["sha256"] != slot["sha256"]
                or payload_hash(state["identity"]) != slot["adaptation_id"]
                or state["state_id"] != slot["adaptation_id"]
                or state["identity"]["lock_sha256"] != lock_sha
            ):
                raise RuntimeError("Private state or prediction changed")
        records.append((folder, task, seed, record))
    return records


def score(args, cfg, target):
    from scripts.drs_peft.run_specialization import load_prediction
    from scripts.drs_peft.transfer_scope_lora import METHODS, SUFFIX, parent_folder
    from scripts.drs_peft.transfer_scope_scoring import score_geometry

    read_target_lock(args)
    records = validated_predictions(args, cfg, target)
    lock_sha = file_hash(args.public_root / "TARGET_TOTAL_K_LOCK.json")
    paths = []
    base = args.public_root / "e0/units"
    for task, seed, method in itertools.product(
        target["target_tasks"], cfg["seeds"], METHODS
    ):
        for shot in task["shots"]:
            for draw in shot["draws"]:
                path = (
                    parent_folder(
                        base,
                        task["task_id"],
                        seed,
                        method + SUFFIX,
                        shot["k"],
                        draw["draw"],
                    )
                    / "RECEIPT.json"
                )
                receipt = read_json(path)
                if (
                    receipt["status"] not in ("PREDICTIONS_LOCKED", "SCORED")
                    or receipt["target_lock_sha256"] != lock_sha
                    or sorted(receipt["support_ids"]) != sorted(draw["support_ids"])
                    or receipt["query_ids"] != task["query_ids"]
                ):
                    raise RuntimeError(
                        "E0 prediction differs from the frozen physical episode"
                    )
                load_prediction(receipt)
                paths.append(path)
    if len(set(paths)) != 192:
        raise RuntimeError("All 192 E0 slots must exist before scoring")
    bind(
        args.public_root / "TARGET_PREDICTION_BARRIER.json",
        dict(
            status="ALL_TARGET_PREDICTIONS_VERIFIED_BEFORE_QUERY_SCORING",
            target_lock_sha256=lock_sha,
            geometry_slots=sum(len(row["slots"]) for _, _, _, row in records),
            lora_slots=len(paths),
            historical_target_query_exposure=True,
            target_selection_uses=0,
        ),
    )
    score_geometry(args, records)


def lora_stage(args, phase):
    from scripts.drs_peft import transfer_scope_lora as lora

    args.fold_ids, args.methods = args.folds, tuple(lora.METHODS)
    cfg, source, episodes, old_lock, parent_lock = lora.contracts(args)
    if phase == "source":
        data, inputs = lora.source_data(args, source)
        for fold, seed, method in itertools.product(
            args.folds, args.seeds, args.methods
        ):
            lora.source_scores(
                args, cfg, source, old_lock, data, inputs, fold, seed, method
            )
        lora.select_source(args, old_lock)
        lora.audit_hust_cal2(args, cfg, source, episodes, old_lock, parent_lock)
    else:
        lora.target_run(args, episodes, old_lock, phase)


def collect(args, cfg, target):
    from scripts.drs_peft import summarize_transfer_scope as summary
    from scripts.drs_peft.transfer_scope_scoring import collect_execution_ledger

    read_target_lock(args)
    collect_execution_ledger(args, validated_predictions(args, cfg, target))
    return summary.main(
        ["--public", str(args.public_root), "--report", str(args.report_root)]
    )


def run_all(args, cfg, source, target):
    if (
        args.folds != cfg["source_folds"]
        or args.seeds != cfg["seeds"]
        or args.domains != cfg["domains"]
    ):
        raise ValueError(
            "The end-to-end command requires all registered folds, seeds, and domains"
        )
    lora_stage(args, "source")
    source_run(args, cfg, source)
    select(args, cfg, True)
    source_run(args, cfg, source, nonlinear=True)
    select(args, cfg, False)
    fit_final(args, source)
    bind(args.public_root / "TARGET_TOTAL_K_LOCK.json", lock_payload(args, target))
    lora_stage(args, "target-predict")
    predict(args, cfg, target)
    score(args, cfg, target)
    lora_stage(args, "target-score")
    return collect(args, cfg, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        required=True,
        choices=[
            "run",
            "source-screen",
            "source-shortlist",
            "source-nonlinear",
            "source-select",
            "fit-final",
            "lock",
            "target-predict",
            "target-score",
            "collect",
        ],
    )
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--data-root", type=Path, default=CACHE)
    p.add_argument("--reference-root", type=Path, default=REFERENCES)
    p.add_argument("--backbone-dir", type=Path, default=BACKBONE)
    p.add_argument("--public-root", type=Path, default=PUBLIC)
    p.add_argument("--private-root", type=Path, default=PRIVATE)
    p.add_argument(
        "--report-root", type=Path, default=ROOT / "report/drs_peft/transfer_scope_v2"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--jvp-batch", type=int, default=8)
    p.add_argument("--folds", type=int, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--domains", nargs="+")
    args = p.parse_args()
    cfg, source, target = contract(args)
    args.folds, args.seeds, args.domains = (
        args.folds or cfg["source_folds"],
        args.seeds or cfg["seeds"],
        args.domains or cfg["domains"],
    )
    if (
        not set(args.folds) <= set(cfg["source_folds"])
        or not set(args.seeds) <= set(cfg["seeds"])
        or not set(args.domains) <= set(cfg["domains"])
    ):
        raise ValueError("Filters must remain within the experiment contract")
    configure()
    with threadpool_limits(limits=2):
        match args.phase:
            case "run":
                return run_all(args, cfg, source, target)
            case "source-screen":
                source_run(args, cfg, source)
            case "source-shortlist":
                select(args, cfg, True)
            case "source-nonlinear":
                source_run(args, cfg, source, True)
            case "source-select":
                select(args, cfg, False)
            case "fit-final":
                fit_final(args, source)
            case "lock":
                bind(
                    args.public_root / "TARGET_TOTAL_K_LOCK.json",
                    lock_payload(args, target),
                )
            case "target-predict":
                predict(args, cfg, target)
            case "target-score":
                score(args, cfg, target)
            case "collect":
                return collect(args, cfg, target)


if __name__ == "__main__":
    raise SystemExit(main())
