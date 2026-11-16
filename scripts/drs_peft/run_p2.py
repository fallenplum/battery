"""Source-only P2 controls with common B0 starts and preregistered gate branches."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from models.drs_peft.peft import install_peft
from .data import ROOT, file_hash, write_json
from .gradient_audit import _interval, _seed, state_fingerprint
from .run_p1 import load_p0_checkpoint, require_gate
from .train import (
    DEFAULT_CONFIG,
    bind,
    configure,
    load_selected,
    load_source,
    make_model,
    read_json,
    train_source_unit,
)


CORE_METHODS = ("lora_xs", "weight_svd_core", "grad_core", "random_core")


@contextmanager
def _observe_optimizer_steps():
    """Count completed AdamW steps while delegating the original implementation."""
    original = torch.optim.AdamW.step
    counts = {"completed": 0, "entered": 0}

    def observed(optimizer, *args, **kwargs):
        counts["entered"] += 1
        result = original(optimizer, *args, **kwargs)
        counts["completed"] += 1
        return result

    torch.optim.AdamW.step = observed
    try:
        yield counts
    finally:
        torch.optim.AdamW.step = original


def _recoverable_trial_error(error):
    if isinstance(error, FloatingPointError):
        return "NONFINITE"
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "CUDA_OOM"
    if isinstance(error, RuntimeError):
        text = str(error)
        if "CUDA out of memory" in text or "CUDA error: out of memory" in text:
            return "CUDA_OOM"
        if "The total norm" in text and "non-finite" in text:
            return "NONFINITE_GRADIENT_NORM"
    return None


def build_plan(execution, p1_gate):
    """Select the frozen positive/negative branch before P2 optimization."""
    if p1_gate.get("status") not in ("PASSED", "NEGATIVE"):
        raise RuntimeError(
            "P2 waits for the completed registered three-seed P1 decision"
        )
    p2 = execution["P2"]
    methods = p2["methods"]
    primary = int(p2["primary_rank"])
    seeds = execution["training_seeds"]
    search = [
        dict(
            stage="search",
            method=method,
            rank=primary,
            seed=execution["pilot_seed"],
            trial=index,
            adapter_lr=lr,
            epochs=p2["pilot_epochs"],
        )
        for method in methods
        for index, lr in enumerate(p2["learning_rate_candidates"][method])
    ]
    if any(len(p2["learning_rate_candidates"][method]) != 3 for method in methods):
        raise ValueError(
            "Each P2 method must receive exactly the frozen three LR trials"
        )
    groups = [("primary", method, primary) for method in methods]
    groups.extend(
        ("parameter_matched", method, int(p2["primary_matched_core_rank"]))
        for method in methods
        if method in CORE_METHODS
    )
    if p1_gate["status"] == "PASSED":
        groups.extend(
            ("rank_curve", method, int(rank))
            for rank in p2["rank_curve"]
            if rank != primary
            for method in methods
        )
    formal = [
        dict(
            stage="formal",
            setting=setting,
            method=method,
            rank=rank,
            seed=seed,
            epochs=p2["formal_epochs"],
        )
        for setting, method, rank in groups
        for seed in seeds
    ]
    return dict(
        stage="P2",
        P1_status=p1_gate["status"],
        branch="FULL_REGISTERED_PRIMARY_AND_PARAMETER_CURVES"
        if p1_gate["status"] == "PASSED"
        else "H1_NEGATIVE_TWO_BUDGET_CONTROLS_NO_RANK_CURVE",
        common_initialization="matching formal source-seed P0 B0 selected bridge/head and official W0",
        methods=methods,
        source_seeds=seeds,
        source_query_labels_only=True,
        target_Query_future_labels_accessed=False,
        rank=primary,
        source_train_cells=85,
        source_batches_per_epoch=11,
        parameter_matched_core_rank=p2["primary_matched_core_rank"],
        search=search,
        formal=formal,
        search_updates_per_unit=p2["pilot_epochs"] * 11,
        formal_updates_per_unit=p2["formal_epochs"] * 11,
        search_rule="lowest source-meta-validation cell mean MAE; listed LR order breaks ties",
        formal_rule=f"all three source seeds receive the same selected per-method LR and {p2['formal_epochs']}-epoch update allowance",
        rank_curve_rule="Both H1 branches retain primary r4 and actual-parameter matched core56; P1 NEGATIVE stops extra ranks2/8",
        hyperparameter_selection_and_reporting="source-meta-validation is reused and intervals are conditional on selected settings; final target Query remains sealed",
        P3_P4_large_expansion_allowed=False,
    )


def _basis_bundle(public_root, private_p1, seed, source_ids):
    public = read_json(
        Path(public_root) / "P1" / f"seed{seed}" / "P1_PUBLIC_COMPLETE.json"
    )
    path = Path(private_p1) / f"seed{seed}" / "P1_BASES.pt"
    if file_hash(path) != public["private_basis"]["sha256"]:
        raise RuntimeError("P1 compact basis bytes changed")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle["source_ids"] != source_ids or int(bundle["seed"]) != seed:
        raise RuntimeError(
            "P1 basis source identities differ from the frozen episode roles"
        )
    if "mean_gradients" not in bundle:
        raise RuntimeError(
            "P1 bundle lacks the registered source-only mean gradient for faithful LoRA-GA"
        )
    return bundle


def _factory(backbone_dir, policy_dim, bridge, method, seed, rank, checkpoint, bundle):
    bases = {
        name: (u[:, :rank], v[:, :rank]) for name, (u, v) in bundle["bases"].items()
    }
    if rank > bundle["rank"]:
        raise ValueError("P2 rank exceeds the registered P1 basis")

    def construct():
        model = make_model(
            backbone_dir, policy_dim, bridge, "B0_pretrained_frozen", seed
        )
        load_selected(model, checkpoint["model"])
        if state_fingerprint(model) != bundle["source_checkpoint_sha256"]:
            raise RuntimeError("P2 common B0 state differs from the model used by P1")
        installed = install_peft(
            model.backbone,
            method,
            rank,
            alpha=rank,
            dropout=0.0,
            seed=seed,
            bases=bases if method == "grad_core" else None,
            gradients=bundle["mean_gradients"] if method == "lora_ga" else None,
        )
        expected_per_matrix = (
            rank * 768 if method in ("lora", "lora_ga") else rank * rank
        )
        if len(installed) != 12 or any(
            record["weight_shape"] != [384, 384]
            or record["trainable"] != expected_per_matrix
            for record in installed
        ):
            raise RuntimeError(
                "P2 adapter target set or actual parameter budget differs from the registered Q/V comparison"
            )
        return model

    return construct


def _publish_unit(private, public):
    private, public = Path(private), Path(public)
    public.mkdir(parents=True, exist_ok=True)
    files = []
    for source in sorted(private.iterdir()):
        if (
            source.suffix.lower() not in (".json", ".jsonl", ".csv")
            or source.name == "SOURCE_DEVELOPMENT_COMPLETE.json"
        ):
            continue
        destination = public / source.name
        if destination.exists() and file_hash(destination) != file_hash(source):
            raise RuntimeError("P2 published source diagnostic bytes changed")
        shutil.copyfile(source, destination)
        files.append(
            dict(
                path=destination.name,
                sha256=file_hash(destination),
                bytes=destination.stat().st_size,
            )
        )
    receipt = read_json(private / "SOURCE_DEVELOPMENT_COMPLETE.json")
    result = {key: value for key, value in receipt.items() if key != "files"}
    result.update(
        private_training_folder=str(private.resolve()),
        public_files=files,
        private_selected_checkpoint=dict(
            path=str((private / "CHECKPOINT_SELECTED.pt").resolve()),
            sha256=receipt["selected_checkpoint_sha256"],
        ),
    )
    bind(public / "SOURCE_DEVELOPMENT_COMPLETE.json", result)
    return result


def _publish_failure(private, public):
    private, public = Path(private), Path(public)
    public.mkdir(parents=True, exist_ok=True)
    for source in sorted(private.iterdir()):
        if source.suffix.lower() in (".json", ".jsonl", ".csv"):
            destination = public / source.name
            if destination.exists() and file_hash(destination) != file_hash(source):
                raise RuntimeError("Published failed-trial evidence changed")
            shutil.copyfile(source, destination)


def _unit_path(root, unit):
    path = Path(root) / unit["stage"] / unit["method"] / f"rank{unit['rank']}"
    if unit["stage"] == "search":
        path = path / f"trial{unit['trial']}"
    return path / f"seed{unit['seed']}"


def _verify_search_complete(public):
    marker = read_json(Path(public) / "P2_SEARCH_COMPLETE.json")
    if marker["status"] != "PASSED":
        raise RuntimeError(
            "P2 formal units require every method's completed registered LR selection"
        )
    for record in marker["files"]:
        if file_hash(Path(public) / record["path"]) != record["sha256"]:
            raise RuntimeError(
                "Frozen P2 LR selection or search evidence bytes changed"
            )
    return marker


def _comparison(
    table, candidate, reference, candidate_rank, reference_rank, seeds, setting, draws
):
    left = table[(table.method == candidate) & (table["rank"] == candidate_rank)]
    right = table[(table.method == reference) & (table["rank"] == reference_rank)]
    paired = left.merge(
        right,
        on=["seed", "cell_uid", "protocol_uid"],
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError("P2 source evaluation roles are not exactly paired")
    paired["improvement"] = paired.mae_reference - paired.mae_candidate
    per_seed = paired.groupby("seed")[["mae_reference", "mae_candidate"]].mean()
    relative = (
        per_seed.mae_reference - per_seed.mae_candidate
    ) / per_seed.mae_reference
    cells = paired.groupby(["cell_uid", "protocol_uid"]).improvement.mean()
    groups = cells.groupby("protocol_uid").mean()
    low_cell, high_cell = _interval(
        cells.to_numpy(),
        np.random.default_rng(_seed(42, setting + reference + "/cell")),
        draws,
    )
    low_group, high_group = _interval(
        groups.to_numpy(),
        np.random.default_rng(_seed(42, setting + reference + "/group")),
        draws,
    )
    mean_ref = float(paired.mae_reference.mean())
    gain = (
        float((mean_ref - paired.mae_candidate.mean()) / mean_ref)
        if mean_ref > 0
        else None
    )
    return dict(
        setting=setting,
        candidate=candidate,
        candidate_rank=candidate_rank,
        reference=reference,
        reference_rank=reference_rank,
        paired_cells=int(len(cells)),
        paired_protocol_groups=int(len(groups)),
        mean_relative_MAE_improvement=gain,
        paired_cell_CI_low=low_cell,
        paired_cell_CI_high=high_cell,
        paired_protocol_CI_low=low_group,
        paired_protocol_CI_high=high_group,
        per_seed_relative_MAE_improvement={
            str(int(seed)): float(value) for seed, value in relative.items()
        },
        all_three_seed_directions_positive=set(relative.index) == set(seeds)
        and bool((relative > 0).all()),
    )


def summarize_p2(output_root, plan, execution):
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    completed = []
    for unit in plan["formal"]:
        folder = _unit_path(output, unit)
        marker = folder / "SOURCE_DEVELOPMENT_COMPLETE.json"
        if not marker.exists():
            continue
        receipt = read_json(marker)
        for record in receipt["public_files"]:
            if file_hash(folder / record["path"]) != record["sha256"]:
                raise RuntimeError(
                    "P2 public source audit bytes changed after completion"
                )
        parameter_report = read_json(folder / "TRAINABLE_PARAMETER_REPORT.json")
        detail = parameter_report["peft_detail"]
        adapter_parameters = detail["adapter_trainable_parameters"]
        expected_adapter = 12 * (
            unit["rank"] * 768
            if unit["method"] in ("lora", "lora_ga")
            else unit["rank"] ** 2
        )
        if adapter_parameters != expected_adapter:
            raise RuntimeError(
                "P2 actual trainable adapter parameter count differs from the preregistered curve"
            )
        metrics = pd.read_csv(folder / "SOURCE_VALIDATION_CELL_METRICS.csv")
        metrics = metrics[(metrics.window == "all") & (metrics.n_obs > 0)].copy()
        metrics["method"], metrics["rank"], metrics["seed"] = (
            unit["method"],
            unit["rank"],
            unit["seed"],
        )
        rows.append(metrics)
        completed.append(
            dict(
                **unit,
                validation_mae=receipt["validation"]["mae"],
                updates=receipt["updates"],
                trainable_parameters=receipt["trainable_parameters"],
                adapter_trainable_parameters=adapter_parameters,
                head_trainable_parameters=parameter_report["head_trainable"],
                bridge_trainable_parameters=parameter_report["bridge_trainable"],
                frozen_basis_elements=detail["frozen_basis_elements"],
                fixed_compensation_elements=detail["fixed_compensation_elements"],
                checkpoint_sha256=receipt["selected_checkpoint_sha256"],
            )
        )
    pd.DataFrame(completed).to_csv(output / "P2_FORMAL_UNITS.csv", index=False)
    if len(completed) != len(plan["formal"]):
        failures = [
            dict(
                unit=unit,
                failure_file=str(_unit_path(output, unit) / "FAILED_TRIAL.json"),
            )
            for unit in plan["formal"]
            if (_unit_path(output, unit) / "FAILED_TRIAL.json").exists()
        ]
        result = dict(
            stage="P2",
            status="BLOCKED" if failures else "PARTIAL",
            complete_formal_units=len(completed),
            required_formal_units=len(plan["formal"]),
            P1_status=plan["P1_status"],
            P3_P4_large_expansion_allowed=False,
            target_Query_future_labels_accessed=False,
            required_formal_units_failed=failures,
        )
        write_json(output / "P2_GATE.json", result)
        return result
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(output / "P2_SOURCE_VALIDATION_CELLS.csv", index=False)
    primary = execution["P2"]["primary_rank"]
    matched = execution["P2"]["primary_matched_core_rank"]
    comparisons = []
    for method in ("lora_xs", "random_core", "weight_svd_core", "lora", "lora_ga"):
        comparisons.append(
            _comparison(
                table,
                "grad_core",
                method,
                primary,
                primary,
                execution["training_seeds"],
                "rank_matched_primary",
                execution["statistics"]["bootstrap_draws"],
            )
        )
        reference_rank = primary if method in ("lora", "lora_ga") else matched
        comparisons.append(
            _comparison(
                table,
                "grad_core",
                method,
                matched,
                reference_rank,
                execution["training_seeds"],
                "actual_parameter_matched_primary",
                execution["statistics"]["bootstrap_draws"],
            )
        )
    write_json(output / "P2_PAIRED_COMPARISONS.json", comparisons)
    primary_controls = [
        row for row in comparisons if row["reference"] in ("lora_xs", "random_core")
    ]
    controlled_gain = all(
        row["mean_relative_MAE_improvement"] is not None
        and row["mean_relative_MAE_improvement"] >= 0.01
        and row["paired_cell_CI_low"] is not None
        and row["paired_cell_CI_low"] > 0
        and row["paired_protocol_CI_low"] is not None
        and row["paired_protocol_CI_low"] > 0
        and row["all_three_seed_directions_positive"]
        for row in primary_controls
    )
    passed = len(primary_controls) == 4 and controlled_gain
    expansion_allowed = passed and plan["P1_status"] == "PASSED"
    result = dict(
        stage="P2",
        status="PASSED" if passed else "NEGATIVE",
        P1_status=plan["P1_status"],
        branch=plan["branch"],
        complete_formal_units=len(completed),
        primary_controls=primary_controls,
        registered_two_budget_control_gain=bool(controlled_gain),
        parameter_matched_curves="COMPLETE",
        additional_rank_curves="COMPLETE"
        if plan["P1_status"] == "PASSED"
        else "NOT_RUN_H1_NEGATIVE_GATE",
        H1_stop_remains_active=plan["P1_status"] != "PASSED",
        P3_P4_large_expansion_allowed=bool(expansion_allowed),
        target_Query_future_labels_accessed=False,
        uncertainty_scope="paired source-meta-validation cells/protocols, conditional on selected LRs; source seeds reported separately",
        novelty_scope="P2 uses established frozen U C V.T and gradient-basis constructions; source specificity remains an empirical claim",
    )
    write_json(output / "P2_GATE.json", result)
    (output / "P2_SOURCE_AUDIT_ZH.md").write_text(
        f"""# P2 源内受控比较

本阶段为{plan["branch"]}，正式单元{len(completed)}个。P2门槛状态为**{result["status"]}**。所有方法从对应seed的正式P0 B0共用bridge/head与官方W0启动。每方法接受seed42的三LR、每次{execution["P2"]["pilot_epochs"]}epoch搜索，选定LR后所有三个seed各训练{execution["P2"]["formal_epochs"]}epoch。搜索与正式更新分别记录。

同rank主比较为r{primary}，实际参数匹配core rank为{matched}。实际参数曲线状态为{result["parameter_matched_curves"]}，额外rank曲线状态为{result["additional_rank_curves"]}。P3/P4大矩阵放行状态为{expansion_allowed}。W-SVD、LoRA与LoRA-GA对照保留在配对比较中，XS含原方法Sigma和Gaussian核心初始化。

主指标为每physical cell完整已观测未来MAE后对cell平均。配对区间以cell及exact protocol分别bootstrap，source seed方向单列。source-meta-validation同时用于LR选择与本阶段报告，区间以本次选择为条件。最终目标Query未来标签访问为0。

优势为共同B0起点、相同优化更新与LR试验次数、真实参数曲线和完整阴性记录。风险为源数据规模、选择条件化的不确定性及外域子空间失配。当前结果的置信度须结合三seed方向、区间和外域冻结后结果判断。
""",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/drs_peft")
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/P2"
    )
    parser.add_argument(
        "--p1-private-root", type=Path, default=ROOT / "runtime/drs_peft/P1"
    )
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument("--p0-gate", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--phase", choices=["all", "search", "formal", "summarize"], default="all"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal-seeds", type=int, nargs="+")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    execution = read_json(args.config)
    if args.formal_seeds is not None:
        if args.phase != "formal":
            parser.error("--formal-seeds is only valid with --phase formal")
        if len(set(args.formal_seeds)) != len(args.formal_seeds) or not set(
            args.formal_seeds
        ).issubset(execution["training_seeds"]):
            parser.error("--formal-seeds must contain unique registered source seeds")
    if args.no_summary and args.phase == "summarize":
        parser.error("--no-summary conflicts with --phase summarize")
    gate = read_json(args.output_root / "P1/P1_GATE.json")
    plan = build_plan(execution, gate)
    if args.dry_run:
        print(json.dumps(plan, indent=2), flush=True)
        return
    require_gate(args.p0_gate or args.output_root / "P0/P0_GATE.json", "P0")
    selection = read_json(args.output_root / "P0/BRIDGE_SELECTION.json")
    bridge = selection["selected_bridge"]
    public = args.output_root / "P2"
    public.mkdir(parents=True, exist_ok=True)
    registration = dict(
        plan=plan,
        config_sha256=file_hash(args.config),
        P1_gate_sha256=file_hash(args.output_root / "P1/P1_GATE.json"),
        driver_sha256=file_hash(__file__),
    )
    registration_path = public / "P2_REGISTRATION.json"
    if args.phase == "formal":
        if read_json(registration_path) != registration:
            raise RuntimeError(
                "Formal worker registration differs from the completed shared P2 search"
            )
    else:
        bind(registration_path, registration)
    if args.phase == "summarize":
        _verify_search_complete(public)
        print(summarize_p2(public, plan, execution), flush=True)
        return
    configure()
    data, inputs, scaler, episodes = load_source(args.data_root)
    source_ids = episodes["source_cell_roles"]["source_train"]
    if len(source_ids) != plan["source_train_cells"]:
        raise RuntimeError(
            "Registered P2 update budget and source physical-cell count differ"
        )

    def run(unit, lr):
        checkpoint, checkpoint_path = load_p0_checkpoint(
            args.output_root, bridge, unit["seed"]
        )
        bundle = _basis_bundle(
            args.output_root, args.p1_private_root, unit["seed"], source_ids
        )
        factory = _factory(
            args.backbone_dir,
            inputs["policy"].shape[-1],
            bridge,
            unit["method"],
            unit["seed"],
            unit["rank"],
            checkpoint,
            bundle,
        )
        private = _unit_path(args.private_root, unit)
        train_source_unit(
            data,
            inputs,
            scaler,
            episodes,
            execution,
            args.config,
            args.backbone_dir,
            private,
            bridge,
            unit["method"],
            unit["seed"],
            unit["epochs"],
            args.device,
            rank=unit["rank"],
            adapter_lr=lr,
            stage="P2",
            initial_checkpoint_sha256=file_hash(checkpoint_path),
            model_factory=factory,
            extra_sources=(
                "scripts/drs_peft/run_p2.py",
                "scripts/drs_peft/run_p1.py",
                "scripts/drs_peft/gradient_audit.py",
                "configs/drs_peft/stage_gates_v1.json",
            ),
        )
        result = _publish_unit(private, _unit_path(public, unit))
        del checkpoint, bundle, factory
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return result

    def attempt_locked(unit, lr):
        private = _unit_path(args.private_root, unit)
        private.mkdir(parents=True, exist_ok=True)
        failure = private / "FAILED_TRIAL.json"
        if failure.exists():
            old = read_json(failure)
            if old["unit"] != unit or old["adapter_lr"] != lr:
                raise RuntimeError(
                    "Failed-trial registration differs from the requested unit"
                )
            for record in old["state_files"]:
                if file_hash(private / record["path"]) != record["sha256"]:
                    raise RuntimeError("Failed-trial checkpoint or log bytes changed")
            _publish_failure(private, _unit_path(public, unit))
            return None
        resumed_updates = 0
        resumed_elapsed = 0.0
        state_path = private / "TRAIN_STATE.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            resumed_updates = int(state["updates"])
            resumed_elapsed = float(state["elapsed"])
            del state
        error_record = None
        started = time.monotonic()
        with _observe_optimizer_steps() as count:
            try:
                return run(unit, lr)
            except (FloatingPointError, RuntimeError) as error:
                reason = _recoverable_trial_error(error)
                if reason is None:
                    raise
                saved_updates = None
                if state_path.exists():
                    state = torch.load(
                        state_path, map_location="cpu", weights_only=False
                    )
                    saved_updates = int(state["updates"])
                    del state
                error_record = dict(
                    status="FAILED_REGISTERED_TRIAL",
                    unit=unit,
                    adapter_lr=lr,
                    failure_category=reason,
                    exception_type=type(error).__name__,
                    error=str(error),
                    resumed_completed_updates=resumed_updates,
                    optimizer_steps_entered_this_invocation=count["entered"],
                    optimizer_steps_completed_this_invocation=count["completed"],
                    optimizer_steps_completed_total=resumed_updates
                    + count["completed"],
                    persisted_checkpoint_completed_updates=saved_updates,
                    elapsed_this_invocation_seconds=time.monotonic() - started,
                    elapsed_total_seconds=resumed_elapsed + time.monotonic() - started,
                    unfinished_optimizer_step_may_have_partial_unsaved_mutation=count[
                        "entered"
                    ]
                    > count["completed"],
                    state_files=[
                        dict(
                            path=path.name,
                            sha256=file_hash(path),
                            bytes=path.stat().st_size,
                        )
                        for path in sorted(private.iterdir())
                        if path.is_file() and path.name != "FAILED_TRIAL.json"
                    ],
                    target_query_labels_used=0,
                    retry_policy="registered trial is retained as failed; no unregistered LR or dummy score",
                )
                write_json(failure, error_record)
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        _publish_failure(private, _unit_path(public, unit))
        print(
            json.dumps(
                dict(
                    event="P2_REGISTERED_TRIAL_FAILED",
                    unit=unit,
                    reason=error_record["failure_category"],
                    completed_updates=error_record["optimizer_steps_completed_total"],
                )
            ),
            flush=True,
        )
        return None

    def attempt(unit, lr):
        private = _unit_path(args.private_root, unit)
        lock = private / ".locks/.execution.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock)):
            return attempt_locked(unit, lr)

    if args.phase in ("all", "search"):
        receipts = [
            (unit, attempt(unit, unit["adapter_lr"])) for unit in plan["search"]
        ]
        trial_rows = []
        for unit, receipt in receipts:
            if receipt is not None:
                trial_rows.append(
                    dict(
                        **unit,
                        status="SOURCE_DEVELOPMENT_COMPLETE",
                        updates=receipt["updates"],
                        validation_mae=receipt["validation"]["mae"],
                        elapsed_seconds=receipt["elapsed_seconds"],
                    )
                )
            else:
                failure = read_json(
                    _unit_path(args.private_root, unit) / "FAILED_TRIAL.json"
                )
                trial_rows.append(
                    dict(
                        **unit,
                        status=failure["status"],
                        updates=failure["optimizer_steps_completed_total"],
                        validation_mae=None,
                        elapsed_seconds=failure["elapsed_total_seconds"],
                        failure_category=failure["failure_category"],
                    )
                )
        pd.DataFrame(trial_rows).to_csv(public / "P2_SEARCH_TRIALS.csv", index=False)
        lr_selection = {}
        failed_methods = []
        for method in execution["P2"]["methods"]:
            candidates = [
                (unit, receipt)
                for unit, receipt in receipts
                if unit["method"] == method and receipt is not None
            ]
            if not candidates:
                failed_methods.append(method)
                continue
            selected = min(candidates, key=lambda pair: pair[1]["validation"]["mae"])
            lr_selection[method] = dict(
                adapter_lr=selected[0]["adapter_lr"],
                trial=selected[0]["trial"],
                seed=selected[0]["seed"],
                validation_mae=selected[1]["validation"]["mae"],
                candidates=[
                    dict(
                        adapter_lr=unit["adapter_lr"],
                        trial=unit["trial"],
                        validation_mae=receipt["validation"]["mae"],
                        checkpoint_sha256=receipt["selected_checkpoint_sha256"],
                        updates=receipt["updates"],
                    )
                    for unit, receipt in candidates
                ],
            )
        bind(
            public / "P2_LR_SELECTION.json",
            dict(
                rule=plan["search_rule"],
                methods=lr_selection,
                methods_with_all_registered_trials_failed=failed_methods,
                attempted_registered_trials=len(receipts),
                target_Query_future_labels_accessed=False,
            ),
        )
        search_files = [
            public / "P2_LR_SELECTION.json",
            public / "P2_SEARCH_TRIALS.csv",
        ]
        search_files.extend(
            _unit_path(public, unit)
            / (
                "SOURCE_DEVELOPMENT_COMPLETE.json"
                if receipt is not None
                else "FAILED_TRIAL.json"
            )
            for unit, receipt in receipts
        )
        bind(
            public / "P2_SEARCH_COMPLETE.json",
            dict(
                status="BLOCKED" if failed_methods else "PASSED",
                attempted_registered_trials=len(receipts),
                files=[
                    dict(
                        path=path.relative_to(public).as_posix(), sha256=file_hash(path)
                    )
                    for path in search_files
                ],
                target_query_labels_used=0,
            ),
        )
        if failed_methods:
            write_json(
                public / "P2_GATE.json",
                dict(
                    stage="P2",
                    status="BLOCKED",
                    reason="ALL_THREE_REGISTERED_LR_TRIALS_FAILED_FOR_REQUIRED_METHOD",
                    failed_methods=failed_methods,
                    attempted_registered_trials=len(receipts),
                    P3_P4_large_expansion_allowed=False,
                    target_query_labels_used=0,
                ),
            )
            return
    if args.phase in ("all", "formal"):
        _verify_search_complete(public)
        chosen = read_json(public / "P2_LR_SELECTION.json")
        if chosen.get("methods_with_all_registered_trials_failed"):
            raise RuntimeError(
                "P2 formal comparison is blocked by a required method with all registered LR trials failed"
            )
        for unit in plan["formal"]:
            if args.formal_seeds is not None and unit["seed"] not in args.formal_seeds:
                continue
            attempt(unit, chosen["methods"][unit["method"]]["adapter_lr"])
        if not args.no_summary:
            print(summarize_p2(public, plan, execution), flush=True)


if __name__ == "__main__":
    main()
