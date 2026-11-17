"""Summarize scored target facts against each method's own reference family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.e123_protocol import WINDOWS
from scripts.drs_peft.onesided_protocol import ROOT, load_registration
from scripts.drs_peft.specialization_protocol import load_target_contract
from scripts.drs_peft.summarize_onesided import (
    METRIC_COLUMNS,
    aggregate,
    digest,
    markdown,
    normalize_receipts,
    parse_boolean,
    separated_variability,
    write_latest,
)

DEFAULT_CONFIG = ROOT / "configs/drs_peft/specialization_comparison_v1/CONFIG.json"
SLOT_KEYS = ["task_id", "seed", "method", "K", "draw"]
FAILURES = {"FAILED", "ERROR", "NUMERICAL_FAILURE", "NOT_ESTIMABLE", "NOT_FEASIBLE"}


def method_families(config):
    mapping = {
        name: item["reference_family"] for name, item in config["method_specs"].items()
    }
    mapping.update(
        {
            item["zero_method"]: family
            for family, item in config["reference_families"].items()
        }
    )
    return mapping


def target_tasks(config, episodes):
    registered = {task["task_id"]: task for task in episodes["target_tasks"]}
    return {name: registered[name] for name in config["target"]["task_ids"]}


def normalize_cells(frame, config):
    missing = set(METRIC_COLUMNS) - set(frame)
    if missing:
        raise ValueError(
            "The scored table omits required columns " + ", ".join(sorted(missing))
        )
    result = frame.copy()
    mapping = method_families(config)
    if result.method.isna().any() or not set(result.method).issubset(mapping):
        raise ValueError(
            "An unregistered specialization method entered the result table"
        )
    expected_family = result.method.map(mapping)
    if (
        "reference_family" in result
        and not result.reference_family.eq(expected_family).all()
    ):
        raise ValueError("A method changed its reference family")
    result["reference_family"] = expected_family
    result["input_stage"] = result.stage
    result["stage"] = "TARGET"
    for name in (
        "domain",
        "task_id",
        "adaptation_id",
        "cell_uid",
        "protocol_uid",
        "window",
    ):
        if result[name].isna().any() or result[name].astype(str).eq("").any():
            raise ValueError("A scored identity is missing " + name)
        result[name] = result[name].astype(str)
    for name in ("seed", "K", "draw", "n_obs", "same_protocol_support_count"):
        numeric = pd.to_numeric(result[name], errors="raise")
        if (
            numeric.isna().any()
            or not np.equal(numeric, np.floor(numeric)).all()
            or numeric.lt(0).any()
        ):
            raise ValueError("A count or budget is invalid " + name)
        result[name] = numeric.astype(int)
    result["fold_id"] = (
        result.fold_id.fillna("final").astype(str).str.removesuffix(".0")
    )
    result["is_duplicate"] = result.is_duplicate.map(parse_boolean)
    result["negative_transfer"] = result.negative_transfer.map(
        lambda value: parse_boolean(value, True)
    )
    for name in ("mae", "rmse", "bias", "zero_mae", "gain_mae"):
        result[name] = pd.to_numeric(result[name], errors="raise")
    return result


def validate_cells(frame, config, episodes):
    frame = normalize_cells(frame, config)
    if frame.empty:
        return frame
    if frame.duplicated(SLOT_KEYS + ["cell_uid", "window"]).any():
        raise ValueError("A Query cell-window repeats within a specialization slot")
    if not set(frame.window).issubset(name for name, _, _ in WINDOWS):
        raise ValueError("An unregistered future window entered the result table")
    observed = frame.n_obs.gt(0)
    metrics = ["mae", "rmse", "bias", "zero_mae", "gain_mae"]
    if not np.isfinite(frame.loc[observed, metrics].to_numpy()).all():
        raise ValueError("A supported window has a nonfinite score")
    if frame.loc[~observed, metrics].notna().any().any():
        raise ValueError("An unsupported window must keep NA scores")
    if frame.loc[observed, ["mae", "rmse", "zero_mae"]].lt(0).any().any():
        raise ValueError("Absolute errors must be nonnegative")
    if not np.allclose(
        frame.loc[observed, "gain_mae"],
        frame.loc[observed, "zero_mae"] - frame.loc[observed, "mae"],
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("Delta K differs from own-family ZERO minus adapted MAE")
    if not np.array_equal(
        frame.loc[observed, "negative_transfer"], frame.loc[observed, "gain_mae"].lt(0)
    ):
        raise ValueError("Negative transfer disagrees with own-family Delta K")
    if frame.groupby(["task_id", "cell_uid", "window"]).n_obs.nunique().gt(1).any():
        raise ValueError("Query observation support changed across reference families")
    if (
        frame.groupby(["task_id", "seed", "reference_family", "cell_uid", "window"])
        .zero_mae.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("A family ZERO reference changed across methods or draws")
    tasks = target_tasks(config, episodes)
    zero_methods = set(config["target"]["zero_methods"])
    for (task_id, seed, method, k, draw), rows in frame.groupby(SLOT_KEYS, sort=False):
        if task_id not in tasks or seed not in config["training_seeds"]:
            raise ValueError("A scored task or seed escaped the registered target role")
        task = tasks[task_id]
        if (
            not rows.domain.eq(task["domain"]).all()
            or not rows.fold_id.eq("final").all()
        ):
            raise ValueError("A target task changed its domain or final reference role")
        shot = next((item for item in task["shots"] if item["k"] == k), None)
        requested = (
            next((item for item in shot["draws"] if item["draw"] == draw), None)
            if shot
            else None
        )
        if requested is None or (k == 0) != (method in zero_methods):
            raise ValueError("A method used an unregistered shot or draw")
        if rows.adaptation_id.nunique() != 1:
            raise ValueError("One slot must identify one adapted state")
        duplicate = requested["duplicate_of_draw"] is not None
        if not rows.is_duplicate.eq(duplicate).all():
            raise ValueError("A repeated Support set changed its duplicate marker")
        if not set(rows.cell_uid).issubset(task["query_ids"]):
            raise ValueError("A method changed the fixed target Query identities")
        protocols = task["protocol_by_id"]
        for uid, cell in rows.groupby("cell_uid", sort=False):
            if not cell.protocol_uid.eq(protocols[uid]).all():
                raise ValueError("A Query changed its descriptive protocol identity")
            count = sum(
                protocols[sid] == protocols[uid] for sid in requested["support_ids"]
            )
            if not cell.same_protocol_support_count.eq(count).all():
                raise ValueError(
                    "Same-protocol Support coverage differs from the episode"
                )
        if k == 0 and not np.allclose(
            rows.loc[rows.n_obs.gt(0), "mae"],
            rows.loc[rows.n_obs.gt(0), "zero_mae"],
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("A family ZERO score differs from its own reference")
    return frame


def expected_slots(config, episodes, seeds):
    result = []
    for task in target_tasks(config, episodes).values():
        for seed in seeds:
            for method in config["target"]["zero_methods"]:
                result.append(
                    {
                        "task_id": task["task_id"],
                        "domain": task["domain"],
                        "seed": seed,
                        "method": method,
                        "K": 0,
                        "draw": 0,
                        "query_ids": task["query_ids"],
                    }
                )
            for method in config["methods"]:
                for shot in task["shots"]:
                    if shot["k"] == 0:
                        continue
                    for draw in shot["draws"]:
                        result.append(
                            {
                                "task_id": task["task_id"],
                                "domain": task["domain"],
                                "seed": seed,
                                "method": method,
                                "K": shot["k"],
                                "draw": draw["draw"],
                                "query_ids": task["query_ids"],
                            }
                        )
    return result


def slot_coverage(frame, records, config, episodes, seeds):
    actual = {key: rows for key, rows in frame.groupby(SLOT_KEYS, sort=False)}
    labels = [name for name, _, _ in WINDOWS]
    output = []
    families = method_families(config)
    for slot in expected_slots(config, episodes, seeds):
        key = tuple(slot[name] for name in SLOT_KEYS)
        rows = actual.get(key, frame.iloc[:0])
        expected = {(uid, window) for uid in slot["query_ids"] for window in labels}
        complete = set(zip(rows.cell_uid, rows.window)) == expected
        failures = [
            record
            for record in records
            if str(record.get("status", "")).upper() in FAILURES
            and all(record.get(name) == slot[name] for name in SLOT_KEYS)
        ]
        if complete and failures:
            raise ValueError("A fully scored slot also claims terminal failure")
        output.append(
            {name: value for name, value in slot.items() if name != "query_ids"}
            | {
                "status": "SCORED"
                if complete
                else failures[0]["status"]
                if failures
                else "PENDING",
                "expected_cell_window_rows": len(expected),
                "scored_cell_window_rows": len(rows),
                "expected_query_cells": len(slot["query_ids"]),
                "reference_family": families[slot["method"]],
                "failure_reason": failures[0].get("error", failures[0].get("reason"))
                if failures
                else None,
            }
        )
    return pd.DataFrame(output)


def paired_differences(frame, group_keys=("domain", "K", "window")):
    comparisons = [
        ("SPECIALIZED_OUT", "SPECIALIZED_LORA"),
        ("SPECIALIZED_OUT", "SPECIALIZED_XS"),
        ("SPECIALIZED_OUT", "PRETRAINED_LORA"),
        ("SPECIALIZED_LORA", "PRETRAINED_LORA"),
        ("SPECIALIZED_ZERO", "WARMUP_ZERO"),
    ]
    columns = [
        *group_keys,
        "left_method",
        "right_method",
        "left_mae",
        "right_mae",
        "difference_mae",
        "delta_K_difference",
        "left_improved_fraction",
        "physical_query_cells",
        "paired_cell_seed_draw_rows",
    ]
    output = []
    valid = frame[frame.n_obs.gt(0)]
    for key, rows in valid.groupby(list(group_keys), sort=True):
        identity = ["task_id", "seed", "draw", "cell_uid"]
        wide = rows.pivot(index=identity, columns="method", values=["mae", "gain_mae"])
        for left, right in comparisons:
            if ("mae", left) not in wide or ("mae", right) not in wide:
                continue
            paired = wide.loc[
                :,
                [
                    ("mae", left),
                    ("mae", right),
                    ("gain_mae", left),
                    ("gain_mae", right),
                ],
            ].dropna()
            if paired.empty:
                continue
            difference = paired["mae", left] - paired["mae", right]
            gain_difference = paired["gain_mae", left] - paired["gain_mae", right]
            output.append(
                dict(zip(group_keys, key, strict=True))
                | {
                    "left_method": left,
                    "right_method": right,
                    "left_mae": float(paired["mae", left].mean()),
                    "right_mae": float(paired["mae", right].mean()),
                    "difference_mae": float(difference.mean()),
                    "delta_K_difference": float(gain_difference.mean()),
                    "left_improved_fraction": float(difference.lt(0).mean()),
                    "physical_query_cells": paired.index.get_level_values(
                        "cell_uid"
                    ).nunique(),
                    "paired_cell_seed_draw_rows": len(paired),
                }
            )
    return pd.DataFrame(output, columns=columns)


def validate_receipts(records, frame, config, episodes):
    families = method_families(config)
    tasks = target_tasks(config, episodes)
    checkpoints = {}
    identities = {}
    for record in records:
        method = record["method"]
        if method not in families or record["task_id"] not in tasks:
            raise ValueError(
                "An adaptation receipt escaped registered method or task roles"
            )
        family = families[method]
        identity = record.get("adaptation_id")
        support_identity = tuple(sorted(record["SIDs"])) if "SIDs" in record else None
        role = tuple(
            record.get(name) for name in ("task_id", "seed", "method", "K")
        ) + (support_identity,)
        if identity and identities.get(identity, role) != role:
            raise ValueError(
                "One adapted-state identity was reused across different roles"
            )
        if identity:
            identities[identity] = role
        if record.get("reference_family", family) != family:
            raise ValueError("An adaptation receipt changed its reference family")
        checkpoint = record.get(
            "sourcecheckpoint_sha", record.get("source_checkpoint_sha256")
        )
        key = (record["seed"], family)
        if checkpoint and checkpoints.get(key, checkpoint) != checkpoint:
            raise ValueError("A family changed its shared reference checkpoint")
        if checkpoint:
            checkpoints[key] = checkpoint
        task = tasks[record["task_id"]]
        if parse_boolean(
            record.get("parent_reused", False)
        ) and not parent_reuse_allowed(
            config, task["domain"], method, int(record["K"])
        ):
            raise ValueError(
                "A target prediction was reused outside its registered domain budget"
            )
        if "QIDs" in record and set(record["QIDs"]) != set(task["query_ids"]):
            raise ValueError("A receipt changed fixed Query identities")
        if "SIDs" in record:
            shot = next(item for item in task["shots"] if item["k"] == int(record["K"]))
            if not any(
                set(draw["support_ids"]) == set(record["SIDs"])
                for draw in shot["draws"]
            ):
                raise ValueError("A receipt used an unregistered Support set")
            if record.get("draw") is not None:
                draw = next(
                    (
                        draw
                        for draw in shot["draws"]
                        if draw["draw"] == int(record["draw"])
                    ),
                    None,
                )
                if draw is None or set(draw["support_ids"]) != set(record["SIDs"]):
                    raise ValueError("A receipt changed its preregistered Support draw")
        scored = frame[frame.adaptation_id.eq(record.get("adaptation_id"))]
        if len(scored) and any(
            not scored[name].eq(record[name]).all()
            for name in ("task_id", "seed", "method", "K")
        ):
            raise ValueError("A scored state disagrees with its receipt identity")


def parent_reuse_allowed(config, domain, method, k):
    target = config["target"]
    if domain == "MATR" and "MATR_POOLED" in target["task_ids"]:
        return k == 0 and target.get("MATR_ZERO_parent_prediction_union_allowed", False)
    domains = target.get("parent_exact_reuse_domains")
    return method in target["parent_exact_reuse_methods"] and (
        domains is None or domain in domains
    )


def adapter_budgets(records, config):
    unique = {}
    for record in records:
        identity = record.get("adaptation_id")
        if not identity:
            continue
        old = unique.get(identity, {})
        merged = old | record
        for name in ("actual_optsteps", "logtime"):
            values = [
                item[name] for item in (old, record) if item.get(name) is not None
            ]
            if values:
                merged[name] = max(values)
        unique[identity] = merged
    families = method_families(config)
    output = []
    for identity, record in unique.items():
        method = record["method"]
        measured = record.get("trainable_parameters")
        expected = config["method_specs"].get(method, {}).get("trainable_parameters", 0)
        if measured is not None and int(measured) != expected:
            raise ValueError(
                "Measured target parameter count differs from the registered budget"
            )
        parameters = record.get("details", {}).get("parameter_report", {})
        domain = record.get(
            "domain",
            config["target"].get("task_domain_by_id", {}).get(record["task_id"]),
        )
        frozen_bytes = record.get(
            "dictbytes",
            parameters.get(
                "frozen_xs_factor_bytes", parameters.get("frozen_svd_bytes")
            ),
        )
        if frozen_bytes is None and method in (
            "PRETRAINED_LORA",
            "SPECIALIZED_LORA",
            "WARMUP_ZERO",
            "SPECIALIZED_ZERO",
        ):
            frozen_bytes = 0
        output.append(
            {
                "method": method,
                "domain": domain,
                "reference_family": families[method],
                "task_id": record["task_id"],
                "seed": record["seed"],
                "K": record["K"],
                "adaptation_id": identity,
                "status": record.get("status"),
                "trainable_parameters": measured,
                "adapter_rank": parameters.get("rank"),
                "global_coordinate_dimension": record.get("actual_q")
                if method == "SPECIALIZED_OUT"
                else None,
                "frozen_adapter_bytes": frozen_bytes,
                "actual_optsteps": record.get("actual_optsteps"),
                "selected_state_optimizer_steps": record.get("state_optimizer_steps"),
                "elapsed_seconds": record.get("logtime"),
                "peak_gpu_bytes": record.get("peak_gpu_bytes"),
                "cache_solver": record.get("cache_solver"),
                "registered_parent_reuse": parent_reuse_allowed(
                    config, domain, method, int(record["K"])
                ),
                "parent_reused": parse_boolean(record.get("parent_reused", False)),
                "parent_reuse_receipt": record.get(
                    "parent_receipt_path", record.get("parent_reused")
                ),
                "reference_checkpoint_sha256": record.get(
                    "sourcecheckpoint_sha", record.get("source_checkpoint_sha256")
                ),
            }
        )
    return pd.DataFrame(
        output,
        columns=[
            "method",
            "domain",
            "reference_family",
            "task_id",
            "seed",
            "K",
            "adaptation_id",
            "status",
            "trainable_parameters",
            "adapter_rank",
            "global_coordinate_dimension",
            "frozen_adapter_bytes",
            "actual_optsteps",
            "selected_state_optimizer_steps",
            "elapsed_seconds",
            "peak_gpu_bytes",
            "cache_solver",
            "registered_parent_reuse",
            "parent_reused",
            "parent_reuse_receipt",
            "reference_checkpoint_sha256",
        ],
    )


def review(summary, differences, run):
    main = (
        summary[summary.window.isin(["all", "1001_plus"])] if len(summary) else summary
    )
    display = [
        "domain",
        "method",
        "reference_family",
        "K",
        "window",
        "mae",
        "zero_mae",
        "delta_K",
        "negative_transfer_ratio",
        "physical_query_cells",
    ]
    paired = (
        differences[differences.window.isin(["all", "1001_plus"])]
        if len(differences)
        else differences
    )
    budget_text = (
        "MATR从24颗候选Support中取全域总共K颗，为固定8颗Query使用同一适配状态。HUST从38颗候选池取全域总K颗，固定39颗Query。"
        if run["support_budget_mode"] == "domain_total_K"
        else "MATR四个任务分别使用K颗Support，聚合预算为4K，固定8颗Query。HUST从既有池取K颗Support，固定39颗Query。"
    )
    return (
        "\n\n".join(
            [
                "# 源域专门化与目标少样本适配复核",
                "## 覆盖与输入",
                f"状态为{run['status']}。已评分{run['scored_slots']}槽，待完成{run['pending_slots']}槽，失败或不可估计{run['failed_slots']}槽。",
                "输入为VIQ+earlySOH / no-policy。两类参考共享源训练的59585参数Bridge/Head，SPECIALIZED额外包含192参数冻结源核心。warmup参考使用官方预训练主干。目标Query属于已有历史暴露的固定池。",
                "## 同Query的K曲线",
                markdown(main[display] if len(main) else main),
                "delta_K为各方法自己的family ZERO MAE减去适配MAE，正值表示改善。"
                + budget_text,
                "## 配对方法差异",
                markdown(paired),
                "difference_mae为left减去right，负值支持left。delta_K_difference为两方法各自family内增益之差。配对使用相同task、seed、K、draw和物理电芯，远期无观测保留NA。",
                "## 优势与风险",
                "适配预算与每状态成本保存在ADAPTER_BUDGETS.csv，RUN_LEDGER.json的execution_ledger保留collect产生的inner与评价完整执行成本，新增目标更新数按PRETRAINED_LORA和XS回执另列。",
                "共同Bridge/Head、固定Query与物理Support，以及独立family ZERO保留了基线位移与目标校准贡献。LoRA为9216参数，XS为192参数，OUT为16个全局共享坐标，预算与求解方式分别报告。",
                "风险包括目标历史暴露、源inner历史使用、有限远期观测及抽样波动。XS采用专门化有效权重的SVD、保留Σ、零增量核心，方法身份为LoRA-XS zero-increment variant。数值核验置信度以实际覆盖为据，方法推广判断还需结合三seed与跨域一致性。",
                "## 拓展思考",
                "结合WARMUP_ZERO与SPECIALIZED_ZERO的差异，以及各自Delta K，可以检验源专门化对迁移起点和Support利用的不同影响。",
                "## 你可能想知道",
                "本汇总只读取公开已评分表与回执，新增模型推理、未来标签数组读取和bootstrap均为0。",
            ]
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target-contract", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.target_contract:
        parent, override, episodes = load_target_contract(
            args.target_contract, args.config
        )
        config = parent | {
            "target": parent["target"] | override["target"],
            "outputs": parent["outputs"] | override["outputs"],
            "episodes_file": override["episodes_file"],
            "episode_contract_sha256": override["episode_contract_sha256"],
        }
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(
        config["training_seeds"]
    ):
        raise ValueError("Summary seeds must be a distinct registered subset")
    if not args.target_contract:
        episodes = load_registration(ROOT / config["episodes_file"])
    if episodes["episode_contract_sha256"] != config["episode_contract_sha256"]:
        raise ValueError("The specialization episode contract changed")
    inputs = args.results_root or ROOT / config["outputs"]["public_root"]
    output = args.output_root or inputs
    history = output / "summary_history"
    table = inputs / "CELL_METRICS.csv"
    frame = (
        pd.read_csv(table) if table.exists() else pd.DataFrame(columns=METRIC_COLUMNS)
    )
    frame = validate_cells(frame[frame.seed.isin(args.seeds)].copy(), config, episodes)
    receipts = inputs / "ADAPTATION_RECORDS.jsonl"
    records = (
        [
            json.loads(line)
            for line in receipts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if receipts.exists()
        else []
    )
    records = normalize_receipts(
        [record for record in records if record.get("seed") in args.seeds]
    )
    validate_receipts(records, frame, config, episodes)
    budgets = adapter_budgets(records, config)
    coverage = slot_coverage(frame, records, config, episodes, args.seeds)
    summary = (
        aggregate(frame, ("reference_family",))
        if len(frame)
        else pd.DataFrame(columns=["window", "mae"])
    )
    summary = summary.rename(columns={"gain_mae": "delta_K"})
    differences = paired_differences(frame)
    known = set(budgets.adaptation_id)
    missing = sorted(set(frame.loc[frame.K.gt(0), "adaptation_id"]) - known)
    missing_checkpoints = sorted(
        set(frame.loc[frame.K.gt(0), "adaptation_id"])
        & set(budgets.loc[budgets.reference_checkpoint_sha256.isna(), "adaptation_id"])
    )
    pending = int(coverage.status.eq("PENDING").sum())
    failed = int((~coverage.status.isin(["PENDING", "SCORED"])).sum())
    new = budgets[
        budgets.method.isin(config["target"]["new_methods_requiring_target_adaptation"])
        & ~budgets.parent_reused
    ]
    run = {
        "status": "PARTIAL"
        if pending or missing or missing_checkpoints
        else "COMPLETE_WITH_FAILURES"
        if failed
        else "COMPLETE",
        "requested_nonzero_slots": int(coverage.K.gt(0).sum()),
        "requested_family_zero_slots": int(coverage.K.eq(0).sum()),
        "scored_slots": int(coverage.status.eq("SCORED").sum()),
        "pending_slots": pending,
        "failed_slots": failed,
        "unique_scored_nonzero_states": int(
            frame.loc[frame.K.gt(0), "adaptation_id"].nunique()
        ),
        "expected_unique_nonzero_states_in_seed_scope": config["target"][
            "expected_unique_nonzero_states"
        ]
        * len(args.seeds)
        // len(config["training_seeds"]),
        "missing_nonzero_receipts": missing,
        "scored_adaptations_missing_reference_checkpoint": missing_checkpoints,
        "new_method_unique_recorded_states": len(new),
        "new_target_optimizer_updates_from_unit_receipts": int(
            new.actual_optsteps.sum()
        ),
        "new_method_states_missing_optimizer_measurement": int(
            new.actual_optsteps.isna().sum()
        ),
        "new_target_cost_scope": config["target"].get(
            "new_target_computation_scope",
            "PRETRAINED_LORA_and_SPECIALIZED_XS;registered_parent_reuse_costs_retained_separately",
        ),
        "target_Query_exposure": "REUSED_HISTORICALLY_EXPOSED_QUERY_POOL",
        "support_budget_mode": "domain_total_K"
        if "MATR_POOLED" in config["target"]["task_ids"]
        else "MATR_per_protocol_K",
        "MATR_support_budget": config["target"]["MATR_support_budget"],
        "HUST_support_budget": config["target"]["HUST_support_budget"],
        "new_model_forwards": 0,
        "future_label_arrays_loaded": 0,
        "bootstrap_draws": 0,
    }
    execution_path = inputs / "EXECUTION_LEDGER.json"
    run["execution_ledger"] = (
        json.loads(execution_path.read_text(encoding="utf-8"))
        if execution_path.exists()
        else None
    )
    run["execution_ledger_sha256"] = (
        digest(execution_path.read_bytes()) if execution_path.exists() else None
    )
    tables = {
        "TARGET_FEWSHOT_RESULTS.csv": summary,
        "PAIRED_METHOD_DIFFERENCES.csv": differences,
        "PAIRED_METHOD_DIFFERENCES_BY_SEED_DRAW.csv": paired_differences(
            frame, ("domain", "K", "window", "seed", "draw")
        ),
        "ADAPTER_BUDGETS.csv": budgets,
        "SLOT_COVERAGE.csv": coverage,
        "NEGATIVE_TRANSFER.csv": frame[frame.K.gt(0) & frame.n_obs.gt(0)],
        "TASK_SEED_DRAW_RESULTS.csv": aggregate(
            frame, ("reference_family", "task_id", "seed", "draw")
        )
        if len(frame)
        else pd.DataFrame(),
        "SEED_DRAW_VARIABILITY.csv": separated_variability(frame)
        if len(frame)
        else pd.DataFrame(),
    }
    written = []
    for name, data in tables.items():
        path = output / name
        write_latest(path, data.to_csv(index=False).encode("utf-8"), history)
        written.append(path)
    ledger_path = output / "RUN_LEDGER.json"
    write_latest(
        ledger_path,
        (json.dumps(run, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        history,
    )
    written.append(ledger_path)
    report_path = (
        args.report or ROOT / config["outputs"]["report_root"] / "FINAL_REVIEW_ZH.md"
    )
    write_latest(
        report_path, review(summary, differences, run).encode("utf-8"), history
    )
    written.append(report_path)
    manifest = {
        "status": run["status"],
        "seeds": args.seeds,
        "config_sha256": digest(args.config.read_bytes()),
        "target_contract_sha256": digest(args.target_contract.read_bytes())
        if args.target_contract
        else None,
        "episode_contract_sha256": episodes["episode_contract_sha256"],
        "selected_scored_rows_sha256": digest(
            frame.to_csv(index=False).encode("utf-8")
        ),
        "summarizer_sha256": digest(Path(__file__).read_bytes()),
        "files": [
            {"path": str(path), "sha256": digest(path.read_bytes())} for path in written
        ],
        "new_model_runs": 0,
        "future_label_arrays_loaded": 0,
    }
    write_latest(
        output / "SUMMARY_MANIFEST.json",
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        history,
    )
    print(json.dumps(run, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
