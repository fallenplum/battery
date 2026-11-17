"""Verify delivered few-shot artifacts without opening a dataset or Query labels."""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.drs_peft.data import ROOT, file_hash, write_json
from scripts.drs_peft.onesided_protocol import load_registration, payload_hash
from scripts.drs_peft.run_onesided import (
    CONFIG,
    GENERIC,
    PRIVATE,
    PUBLIC,
    hyperparameters,
    prediction_identity,
    select_own_fold,
    shot_for,
    task_for,
)
from scripts.drs_peft.run_onesided_reference import reference_folder

STAGES = (
    "cv-references",
    "final-references",
    "basis",
    "inner",
    "D2",
    "D3",
    "target-lock",
    "target-predictions",
    "target",
)
GRID = np.arange(101, 5001, dtype=np.float32)
WINDOWS = {"all", "101_200", "201_500", "501_1000", "1001_plus"}


def slot_specs(stage, tasks, seeds, methods):
    """Keep draw slots while identifying repeated physical Support sets."""
    records = []
    for task in tasks:
        for seed in seeds:
            zero = shot_for(task, 0)["draws"][0]
            records.append(
                {
                    "stage": stage,
                    "task": task,
                    "seed": seed,
                    "method": "ZERO",
                    "K": 0,
                    "draw": zero,
                }
            )
            for method in methods:
                for k in (1, 3, 5):
                    records.extend(
                        {
                            "stage": stage,
                            "task": task,
                            "seed": seed,
                            "method": method,
                            "K": k,
                            "draw": draw,
                        }
                        for draw in shot_for(task, k)["draws"]
                    )
    return records


def slot_key(spec):
    return (
        spec["stage"],
        spec["task"]["task_id"],
        spec["seed"],
        spec["method"],
        spec["K"],
        spec["draw"]["draw"],
    )


def unique_support_key(spec):
    return (*slot_key(spec)[:-1], tuple(sorted(spec["draw"]["support_ids"])))


def matrix_counts(specifications):
    keys = [slot_key(spec) for spec in specifications]
    nonzero = [spec for spec in specifications if spec["K"] > 0]
    return {
        "slots": len(keys),
        "unique_slot_keys": len(set(keys)),
        "zero_references": len(keys) - len(nonzero),
        "nonzero_slots": len(nonzero),
        "unique_nonzero_support_states": len(
            {unique_support_key(spec) for spec in nonzero}
        ),
    }


class DeliveryVerifier:
    def __init__(self, args, config, registration):
        self.args, self.config, self.registration = args, config, registration
        self.issues, self.counts, self.references = [], {}, {}
        self.hashes, self.dictionary_metadata, self.unit_summaries = {}, {}, {}
        self.na, self.parameter_records = [], []
        self.reference_bh = {}
        self.folds = json.loads(
            (ROOT / config["folds_file"]).read_text(encoding="utf-8")
        )["folds"]

    def issue(self, kind, path, message):
        self.issues.append({"kind": kind, "path": str(path), "message": message})

    def check(self, condition, path, message):
        if not condition:
            self.issue("ERROR", path, message)
        return bool(condition)

    def read(self, path):
        path = Path(path)
        if not path.is_file():
            self.issue("MISSING", path, "Required artifact is pending")
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def digest(self, path):
        path = Path(path)
        if not path.is_file():
            self.issue("MISSING", path, "Required private or public file is pending")
            return None
        key = str(path.resolve())
        if key not in self.hashes:
            self.hashes[key] = file_hash(path)
        return self.hashes[key]

    def hash_check(self, path, expected):
        actual = self.digest(path)
        if actual is not None:
            self.check(
                actual == expected,
                path,
                "Artifact bytes differ from the completion receipt",
            )
        return actual

    def guard(self, label, function, *args):
        try:
            return function(*args)
        except (
            KeyError,
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
            AssertionError,
            EOFError,
            IndexError,
            AttributeError,
            pickle.UnpicklingError,
            zipfile.BadZipFile,
        ) as error:
            self.issue("ERROR", label, f"{type(error).__name__}: {error}")
            return None

    def reference(self, fold, seed):
        key = (str(fold), seed)
        if key not in self.references:
            folder = reference_folder(self.args.reference_root, fold, seed)
            self.references[key] = self.read(folder / "REFERENCE_COMPLETE.json")
        return self.references[key]

    def verify_fit(self, fold, seed, stage):
        private = reference_folder(self.args.reference_root, fold, seed) / stage
        public = (
            reference_folder(self.args.public_root / "references", fold, seed) / stage
        )
        receipt = self.read(public / "SOURCE_DEVELOPMENT_COMPLETE.json")
        if receipt is None:
            return False
        run = self.read(public / "RUN_CONFIG.json")
        report = self.read(public / "TRAINABLE_PARAMETER_REPORT.json")
        if run is None or report is None:
            return False
        self.check(
            receipt["status"] == "SOURCE_DEVELOPMENT_COMPLETE"
            and receipt["target_query_labels_used"] == 0,
            public,
            "Source fit must be complete and target-label-free",
        )
        for record in receipt.get("public_files", []):
            self.hash_check(public / record["path"], record["sha256"])
        checkpoint = private / "CHECKPOINT_SELECTED.pt"
        if self.hash_check(checkpoint, receipt["selected_checkpoint_sha256"]) is None:
            return False
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        stored_bh = sum(
            value.numel()
            for name, value in saved["model"].items()
            if not name.startswith("bridge.backbone.module.")
        )
        self.check(
            stored_bh == 59585 and saved["model"]["bridge.policy_scale"].numel() == 0,
            checkpoint,
            "Stored no-policy Bridge/Head tensors differ from the 59585-parameter architecture",
        )
        if stage == "static":
            self.check(
                sum(
                    value.numel()
                    for name, value in saved["model"].items()
                    if name.endswith(".core")
                )
                == 192,
                checkpoint,
                "Stored source rank-four cores differ from 192 entries",
            )
        self.check(
            saved["identity"] == run and saved["best_epoch"] == receipt["best_epoch"],
            checkpoint,
            "Checkpoint identity or selected epoch differs",
        )
        expected_parameters = 59585 if stage == "warmup" else 192
        self.check(
            receipt["trainable_parameters"]
            == run["trainable_parameters"]
            == report["trainable"]
            == expected_parameters,
            public,
            "Actual source trainable budget differs",
        )
        self.check(
            report["bridge"] + report["head"] == 59585,
            public,
            "The shared Bridge/Head parameter budget differs",
        )
        self.check(
            report["policy_dimension"] == 0
            and report["policy_values_forced_zero"]
            and report["policy_masks_forced_zero"],
            public,
            "The source architecture must remove both policy channels",
        )
        self.check(
            run["config_sha256"] == file_hash(self.args.config) and run["seed"] == seed,
            public,
            "Source fit config or seed differs",
        )
        self.check(
            set(run["training_ids"]).isdisjoint(run["validation_ids"]),
            public,
            "Source fit and validation identities overlap",
        )
        self.check(
            run["expected_updates"] == receipt["updates"]
            and receipt["frozen_parameters_unchanged"],
            public,
            "Source update budget or frozen state differs",
        )
        if fold == "final":
            expected_ids = self.registration["source_ids"]
            self.check(
                sorted(run["training_ids"]) == sorted(expected_ids)
                and not run["validation_ids"],
                public,
                "Final fit must use all 115 source cells without validation",
            )
            epochs = []
            for cv_fold in range(3):
                cv = self.read(
                    reference_folder(
                        self.args.public_root / "references", cv_fold, seed
                    )
                    / stage
                    / "SOURCE_DEVELOPMENT_COMPLETE.json"
                )
                if cv is not None:
                    epochs.append(cv["best_epoch"])
            if len(epochs) == 3:
                self.check(
                    receipt["epochs"]
                    == receipt["best_epoch"]
                    == int(np.median(epochs)),
                    public,
                    "Final fit must select the last epoch at the stagewise CV median",
                )
        else:
            role = next(item for item in self.folds if item["fold_id"] == fold)
            self.check(
                run["training_ids"] == role["fit_ids"]
                and run["validation_ids"] == role["inner_validation_ids"],
                public,
                "Reference source roles differ from the registered fold",
            )
        if stage == "static":
            warmup = self.read(
                private.parent / "warmup/SOURCE_DEVELOPMENT_COMPLETE.json"
            )
            if warmup is not None:
                self.check(
                    run["initial_checkpoint_sha256"]
                    == warmup["selected_checkpoint_sha256"],
                    public,
                    "Static fitting did not start at its own warmup",
                )
        stored_parameters = (
            stored_bh
            if stage == "warmup"
            else sum(
                value.numel()
                for name, value in saved["model"].items()
                if name.endswith(".core")
            )
        )
        self.parameter_records.append(
            {
                "fold_id": fold,
                "seed": seed,
                "method": stage,
                "actual_trainable_parameters": stored_parameters,
            }
        )
        return True

    def verify_references(self, final=False):
        folds = ["final"] if final else range(3)
        fits, references = 0, 0
        for fold in folds:
            for seed in self.config["training_seeds"]:
                for stage in ("warmup", "static"):
                    fits += bool(
                        self.guard(
                            f"reference/{fold}/{seed}/{stage}",
                            self.verify_fit,
                            fold,
                            seed,
                            stage,
                        )
                    )
                receipt = self.reference(fold, seed)
                public = reference_folder(
                    self.args.public_root / "references", fold, seed
                )
                published = self.read(public / "REFERENCE_COMPLETE.json")
                audit = self.read(public / "INPUT_SCOPE_AUDIT.json")
                if receipt is None or published is None or audit is None:
                    continue
                self.check(
                    receipt == published,
                    public,
                    "Private and public f0 receipts differ",
                )
                self.check(
                    receipt["policy_dimension"] == 0
                    and receipt["final_target_query_labels_used"] == 0,
                    public,
                    "Reference input or outcome scope differs",
                )
                self.check(
                    audit["status"] == "PASSED"
                    and audit["future_labels_used"] == 0
                    and audit["policy_perturbation_max_prediction_change"] == 0,
                    public,
                    "No-policy perturbation audit did not pass",
                )
                private = reference_folder(self.args.reference_root, fold, seed)
                self.hash_check(
                    private / "warmup/CHECKPOINT_SELECTED.pt",
                    receipt["warmup_checkpoint_sha256"],
                )
                self.hash_check(
                    private / "static/CHECKPOINT_SELECTED.pt",
                    receipt["reference_checkpoint_sha256"],
                )
                references += 1
        self.counts["final_references" if final else "cv_references"] = {
            "source_fits": fits,
            "expected_source_fits": 6 if final else 18,
            "references": references,
            "expected_references": 3 if final else 9,
        }
        index_path = self.args.public_root / "references/REFERENCE_MODELS.csv"
        if self.digest(index_path) is not None:
            index = pd.read_csv(index_path, dtype={"fold_id": str})
            self.check(
                not index.duplicated(["fold_id", "seed"]).any(),
                index_path,
                "Reference model index contains duplicate fold/seed keys",
            )
            indexed = set(zip(index.fold_id, index.seed, strict=True))
            completed = {
                key for key, value in self.references.items() if value is not None
            }
            self.check(
                completed.issubset(indexed),
                index_path,
                "Completed reference functions are absent from the public index",
            )

    def dictionary(self, fold, seed, spectrum=False):
        key = (str(fold), seed, spectrum)
        if key in self.dictionary_metadata:
            return self.dictionary_metadata[key]
        folder = reference_folder(self.args.reference_root, fold, seed) / "geometry"
        name = "TRANSFER_SPECTRUM" if spectrum else "DICTIONARIES"
        receipt = self.read(
            folder / (name + (".json" if spectrum else "_COMPLETE.json"))
        )
        if receipt is None:
            return None
        path = folder / (name + ".pt")
        if not spectrum:
            for record in receipt["files"]:
                self.hash_check(folder / record["path"], record["sha256"])
                if Path(record["path"]).suffix in (".json", ".csv"):
                    published = (
                        reference_folder(
                            self.args.public_root / "references", fold, seed
                        )
                        / "geometry"
                        / record["path"]
                    )
                    self.hash_check(published, record["sha256"])
        receipt_name = name + (".json" if spectrum else "_COMPLETE.json")
        published_receipt = (
            reference_folder(self.args.public_root / "references", fold, seed)
            / "geometry"
            / receipt_name
        )
        self.hash_check(published_receipt, self.digest(folder / receipt_name))
        if self.digest(path) is None:
            return None
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        reference = self.reference(fold, seed)
        if reference is not None:
            identity = (
                receipt["source_checkpoint_identity"]
                if spectrum
                else receipt["identity"]
            )
            self.check(
                identity["reference_checkpoint_sha256"]
                == reference["reference_checkpoint_sha256"],
                path,
                "Dictionary gradients belong to another f0",
            )
        if spectrum:
            self.check(
                bundle["status"] == receipt["status"]
                and bundle["dimension"] == receipt["dimension"],
                path,
                "D3 private and public spectrum receipts differ",
            )
            payloads = bundle["dictionaries"]
            if receipt["status"].startswith("NOT_ESTIMABLE"):
                self.check(
                    not payloads and receipt["dimension"] == 0,
                    path,
                    "Unestimable CROSS must have no padded directions",
                )
                self.na.append(
                    {
                        "fold_id": fold,
                        "seed": seed,
                        "stage": "D3",
                        "reason": receipt["status"],
                    }
                )
            else:
                self.check(
                    set(payloads) == {"AUTO", "CROSS", "RANDOM"}
                    and 0 < receipt["dimension"] <= 8,
                    path,
                    "D3 must share its actual supported dimension",
                )
        else:
            self.check(
                bundle["identity"]["gradient_checkpoint"] == "final_static_f0",
                path,
                "Final f0 gradients are required",
            )
            self.check(
                bundle["requested_dimension"] == 16,
                path,
                "The registered dictionary request is q16",
            )
            payloads = bundle["arms"]
        dimensions = {}
        for method, payload in payloads.items():
            q = int(payload["dimension"])
            dimensions[method] = q
            shapes = [
                layer["directions"].shape[0] for layer in payload["layers"].values()
            ]
            actual_bytes = sum(
                value.numel() * value.element_size()
                for layer in payload["layers"].values()
                for value in layer.values()
                if torch.is_tensor(value)
            )
            self.check(
                len(shapes) == 12 and all(value == q for value in shapes),
                path,
                "Each global direction must span the same twelve layers",
            )
            self.check(
                actual_bytes == payload["frozen_buffer_bytes"],
                path,
                "Frozen dictionary storage budget differs",
            )
        common = bundle.get("common_dimension")
        result = {
            "sha256": self.digest(path),
            "dimensions": dimensions,
            "common_dimension": common,
            "status": receipt["status"],
        }
        self.dictionary_metadata[key] = result
        return result

    def verify_basis(self):
        for fold in (*range(3), "final"):
            for seed in self.config["training_seeds"]:
                self.guard(f"basis/{fold}/{seed}", self.dictionary, fold, seed)
                if fold != "final" and seed in self.config["D3"]["training_seeds"]:
                    self.guard(
                        f"spectrum/{fold}/{seed}", self.dictionary, fold, seed, True
                    )
        self.counts["geometry_artifact_groups"] = len(self.dictionary_metadata)

    def selection(self, phase, fold, seed, method, k):
        folder = (
            self.args.public_root
            / "inner"
            / phase
            / f"fold{fold}"
            / f"seed{seed}"
            / method
            / f"K{k}"
        )
        selected = self.read(folder / "SELECTED.json")
        if selected is None:
            return None
        reference, dictionary = (
            self.reference(fold, seed),
            self.dictionary(fold, seed, phase == "D3"),
        )
        identity = selected["identity"]
        task = task_for(self.registration, "inner", fold)
        expected = {
            "fold_id": fold,
            "seed": seed,
            "method": method,
            "K": k,
            "role": "own_fold_inner_only",
            "query_ids": task["query_ids"],
            "episode_sha256": self.registration["episode_contract_sha256"],
            "config_sha256": file_hash(self.args.config),
        }
        if reference is not None:
            expected["reference_checkpoint_sha256"] = reference[
                "reference_checkpoint_sha256"
            ]
        if dictionary is not None:
            expected["dictionary_sha256"] = dictionary["sha256"]
        expected["execution_sha256"] = file_hash(
            ROOT / "scripts/drs_peft/run_onesided.py"
        )
        expected["engine_sha256"] = file_hash(
            ROOT / "scripts/drs_peft/onesided_engine.py"
        )
        self.check(
            identity == expected and selected["target_query_label_uses"] == 0,
            folder,
            "Inner selection information identity differs",
        )
        trials_path = folder / "TRIALS.csv"
        if self.hash_check(trials_path, selected["trials_csv_sha256"]) is None:
            return None
        table = pd.read_csv(trials_path)
        grid = hyperparameters(self.config, method)
        self.check(
            set(table.trial) == set(range(len(grid))) and len(table) == len(grid) * 5,
            trials_path,
            "The registered hyperparameter grid is incomplete",
        )
        self.check(
            all(
                table[key].eq(value).all()
                for key, value in (
                    ("fold_id", fold),
                    ("seed", seed),
                    ("method", method),
                    ("K", k),
                    ("role", "inner"),
                )
            ),
            trials_path,
            "Cross-fold or outer records entered inner selection",
        )
        trial, score = select_own_fold(table, fold, seed, method, k)
        self.check(
            selected["selected_trial"] == trial
            and selected["hyperparameters"] == grid[trial]
            and selected["trial_grid"] == grid
            and np.isclose(selected["inner_mae"], score, rtol=0, atol=1e-12),
            folder,
            "Selected trial differs from first-minimum own-inner scores",
        )
        return selected

    def methods(self, fold, seed, spectrum=False):
        metadata = self.dictionary(fold, seed, spectrum)
        if metadata is None:
            return []
        if spectrum:
            return list(metadata["dimensions"])
        methods = ["CAL2", *metadata["dimensions"], *GENERIC]
        common = metadata["common_dimension"]
        if common and any(q != common for q in metadata["dimensions"].values()):
            methods.extend(method + "_COMMON" for method in metadata["dimensions"])
        return methods

    def verify_inner(self):
        complete = 0
        for phase, seeds in (
            ("D2", self.config["training_seeds"]),
            ("D3", self.config["D3"]["training_seeds"]),
        ):
            for fold in range(3):
                for seed in seeds:
                    for method in self.methods(fold, seed, phase == "D3"):
                        for k in (1, 3, 5):
                            complete += (
                                self.guard(
                                    f"inner/{phase}/{fold}/{seed}/{method}/{k}",
                                    self.selection,
                                    phase,
                                    fold,
                                    seed,
                                    method,
                                    k,
                                )
                                is not None
                            )
        self.counts["verified_inner_selections"] = complete

    def lock(self):
        lock = self.read(self.args.public_root / "TARGET_LOCK.json")
        if lock is None:
            return None
        self.check(
            lock["config_sha256"] == file_hash(self.args.config)
            and lock["episodes_sha256"] == self.registration["episode_contract_sha256"]
            and lock["target_query_label_uses_for_selection"] == 0,
            "TARGET_LOCK",
            "Target lock configuration or information budget differs",
        )
        self.check(
            lock["candidate"] == "OUT" and lock["best_generic"] in GENERIC,
            "TARGET_LOCK",
            "The registered candidate or generic method differs",
        )
        for record in lock["source_inner_files"]:
            self.hash_check(ROOT / record["path"], record["sha256"])
        self.check(
            len(lock["source_inner_files"])
            == len({record["path"] for record in lock["source_inner_files"]})
            == 108,
            "TARGET_LOCK",
            "Target choices must bind all 108 distinct source-inner tuning tables",
        )
        return lock

    def verify_target_lock(self):
        lock = self.lock()
        if lock is None:
            return
        scores = {}
        for method in ("CAL2", "OUT", *GENERIC):
            scores[method] = {}
            for k in (1, 3, 5):
                tables = []
                for fold in range(3):
                    for seed in self.config["training_seeds"]:
                        selected = self.guard(
                            f"target-lock/{fold}/{seed}/{method}/{k}",
                            self.selection,
                            "D2",
                            fold,
                            seed,
                            method,
                            k,
                        )
                        if selected is not None:
                            path = (
                                self.args.public_root
                                / "inner/D2"
                                / f"fold{fold}"
                                / f"seed{seed}"
                                / method
                                / f"K{k}/TRIALS.csv"
                            )
                            tables.append(pd.read_csv(path))
                if len(tables) != 9:
                    continue
                means = pd.concat(tables).groupby("trial", sort=True).mae.mean()
                trial = int(means.idxmin())
                scores[method][str(k)] = float(means.loc[trial])
                self.check(
                    lock["selected_hyperparameters"][method][str(k)]
                    == hyperparameters(self.config, method)[trial],
                    "TARGET_LOCK",
                    "Final target hyperparameters differ from source-inner aggregation",
                )
        if all(len(value) == 3 for value in scores.values()):
            generic = min(
                GENERIC, key=lambda method: np.mean(list(scores[method].values()))
            )
            self.check(
                lock["best_generic"] == generic,
                "TARGET_LOCK",
                "BEST_GENERIC differs from the source-only mean over K",
            )
        self.counts["target_choices_lock_present"] = True

    def unit_path(self, spec):
        return (
            self.args.public_root
            / "units"
            / spec["stage"]
            / spec["task"]["task_id"]
            / f"seed{spec['seed']}"
            / spec["method"]
            / f"K{spec['K']}_draw{spec['draw']['draw']}"
        )

    def verify_state(self, path, expected, receipt):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        identity = {key: value for key, value in expected.items() if key != "draw"}
        identity["support_ids"] = sorted(identity["support_ids"])
        self.check(
            saved["identity"] == identity
            and receipt["adaptation_id"] == payload_hash(identity),
            path,
            "Physical Support state identity or deduplication key differs",
        )
        state = saved["state"]
        method = expected["method"]
        if method == "ZERO":
            self.check(
                state is None, path, "ZERO must reuse f0 without an adaptation state"
            )
            return 0
        report = saved["details"]["parameter_report"]
        self.check(
            report["bridge_trainable"] == 0 and report["head_trainable"] == 0,
            path,
            "Support adaptation must freeze Bridge and Head",
        )
        if method in GENERIC:
            selected = state["selected_state"]
            tokens = (
                (".lora_b.weight",)
                if method == "LoRA_FA"
                else (".lora_a.weight", ".lora_b.weight")
            )
            parameters = sum(
                value.numel()
                for key, value in selected.items()
                if any(token in key for token in tokens)
            )
            self.check(
                parameters == report["adaptation_parameters"] == 9216,
                path,
                "Actual serialized LoRA adaptation budget differs from 9216",
            )
            self.check(
                state["adaptation_scale"] == expected["hyperparameters"]["eta"],
                path,
                "LoRA shrinkage differs from the source-selected eta",
            )
            source_key = (str(expected["fold_id"]), expected["seed"])
            if source_key not in self.reference_bh:
                source_path = (
                    reference_folder(
                        self.args.reference_root, expected["fold_id"], expected["seed"]
                    )
                    / "static/CHECKPOINT_SELECTED.pt"
                )
                if self.digest(source_path) is not None:
                    source = torch.load(
                        source_path, map_location="cpu", weights_only=False
                    )["model"]
                    self.reference_bh[source_key] = {
                        name: value
                        for name, value in source.items()
                        if name.startswith("head.")
                        or name.startswith("bridge.")
                        and not name.startswith("bridge.backbone.module.")
                    }
            if source_key in self.reference_bh:
                self.check(
                    all(
                        name in selected and torch.equal(value, selected[name])
                        for name, value in self.reference_bh[source_key].items()
                    ),
                    path,
                    "Serialized Bridge/Head tensors changed during Support adaptation",
                )
        else:
            parameters = int(np.asarray(state["a"]).size)
            self.check(
                np.isfinite(np.asarray(state["a"])).all(),
                path,
                "Adaptation coordinates are nonfinite",
            )
            self.check(
                parameters == report["adaptation_parameters"] and parameters > 0,
                path,
                "Actual coordinate dimension differs from the model budget",
            )
            if method == "CAL2":
                self.check(
                    parameters == 2, path, "CAL2 must have two trainable coordinates"
                )
        self.check(
            state["source_checkpoint_sha256"] == expected["source_checkpoint_sha256"],
            path,
            "Adaptation state belongs to another f0",
        )
        return parameters

    def verify_unit(self, spec, hyper, require_scored=True):
        folder = self.unit_path(spec)
        receipt = self.read(folder / "RECEIPT.json")
        if receipt is None:
            return None
        task, method, seed = spec["task"], spec["method"], spec["seed"]
        fold = task.get("fold_id", "final")
        reference, dictionary = (
            self.reference(fold, seed),
            self.dictionary(fold, seed, spec["stage"] == "D3"),
        )
        if reference is None or dictionary is None:
            return None
        expected = prediction_identity(
            spec["stage"],
            task,
            seed,
            method,
            spec["K"],
            spec["draw"]["draw"],
            spec["draw"]["support_ids"],
            hyper,
            dict(reference, dictionary_sha256=dictionary["sha256"]),
        )
        self.check(
            all(receipt.get(key) == value for key, value in expected.items()),
            folder,
            "Receipt differs from the registered key, source, Support, Query or selection",
        )
        self.check(
            receipt["status"] in ("PREDICTIONS_LOCKED", "SCORED")
            and receipt["final_target_query_labels_used_for_selection_or_adaptation"]
            == 0,
            folder,
            "Prediction lock or target selection information scope differs",
        )
        self.check(
            receipt["config_sha256"] == file_hash(self.args.config),
            folder,
            "Adaptation receipt configuration differs",
        )
        if require_scored and receipt["status"] != "SCORED":
            self.issue(
                "MISSING",
                folder / "CELL_METRICS.csv",
                "Registered Query scoring is pending",
            )
        self.check(
            receipt["is_duplicate"]
            == (spec["draw"].get("duplicate_of_draw") is not None),
            folder,
            "Support draw duplication flag differs",
        )
        prediction_path, state_path = (
            ROOT / receipt["prediction_path"],
            ROOT / receipt["state_path"],
        )
        if (
            self.hash_check(prediction_path, receipt["prediction_sha256"]) is None
            or self.hash_check(state_path, receipt["state_sha256"]) is None
        ):
            return None
        with np.load(prediction_path, allow_pickle=False) as archive:
            self.check(
                set(archive.files) == {"ids", "query", "prediction"},
                prediction_path,
                "Prediction archive must contain predictions and fixed axes only",
            )
            self.check(
                archive["ids"].astype(str).tolist() == task["query_ids"]
                and np.array_equal(archive["query"], GRID),
                prediction_path,
                "Prediction physical IDs or fixed cycle grid differs",
            )
            self.check(
                archive["prediction"].shape == (len(task["query_ids"]), 4900)
                and np.isfinite(archive["prediction"]).all(),
                prediction_path,
                "Prediction shape or numeric validity differs",
            )
        state_key = str(state_path.resolve())
        state_identity = {
            key: value for key, value in expected.items() if key != "draw"
        }
        state_identity["support_ids"] = sorted(state_identity["support_ids"])
        self.check(
            receipt["adaptation_id"] == payload_hash(state_identity),
            folder,
            "A receipt deduplication key differs from its actual physical Support state",
        )
        if state_key not in self.unit_summaries:
            parameters = self.verify_state(state_path, expected, receipt)
            self.unit_summaries[state_key] = {
                "parameters": parameters,
                "identity": state_identity,
            }
            self.parameter_records.append(
                {
                    "stage": spec["stage"],
                    "method": method,
                    "actual_trainable_parameters": parameters,
                }
            )
        self.check(
            self.unit_summaries[state_key]["identity"] == state_identity,
            state_path,
            "Distinct source, method or Support identities share an incompatible state file",
        )
        parameters = self.unit_summaries[state_key]["parameters"]
        if method not in ("ZERO", "CAL2", *GENERIC):
            expected_dimension = (
                dictionary["common_dimension"]
                if method.endswith("_COMMON")
                else dictionary["dimensions"][method]
            )
            self.check(
                parameters == expected_dimension,
                state_path,
                "Actual adaptation coordinates differ from their stored dictionary dimension",
            )
        if receipt["status"] == "SCORED":
            metric_path = folder / "CELL_METRICS.csv"
            if self.hash_check(metric_path, receipt["cell_metrics_sha256"]) is not None:
                metrics = pd.read_csv(metric_path)
                self.check(
                    set(metrics.cell_uid) == set(task["query_ids"])
                    and set(metrics.window) == WINDOWS
                    and len(metrics) == len(task["query_ids"]) * 5
                    and not metrics.duplicated(["cell_uid", "window"]).any(),
                    metric_path,
                    "Published metrics must cover every physical Query/window once",
                )
        return receipt

    def verify_inventory(self, phase, specifications):
        expected = {self.unit_path(spec).resolve() for spec in specifications}
        keys = []
        for path in (self.args.public_root / "units" / phase).glob("**/RECEIPT.json"):
            receipt = self.read(path)
            self.check(
                path.parent.resolve() in expected,
                path,
                "An unregistered primary or supplementary unit exists",
            )
            keys.append(
                tuple(
                    receipt[key]
                    for key in ("stage", "task_id", "seed", "method", "K", "draw")
                )
            )
        self.check(
            len(keys) == len(set(keys)),
            phase,
            "Delivered receipts contain duplicate semantic slot keys",
        )
        records_path = self.args.public_root / "ADAPTATION_RECORDS.jsonl"
        if self.digest(records_path) is not None:
            collected = []
            with records_path.open(encoding="utf-8") as handle:
                for line in handle:
                    receipt = json.loads(line)
                    if receipt["stage"] == phase:
                        collected.append(
                            tuple(
                                receipt[key]
                                for key in (
                                    "stage",
                                    "task_id",
                                    "seed",
                                    "method",
                                    "K",
                                    "draw",
                                )
                            )
                        )
            self.check(
                len(collected) == len(set(collected)) and set(collected) == set(keys),
                records_path,
                "Collected adaptation records omit, repeat or add unit keys",
            )

    def verify_source(self, spectrum=False):
        phase = "D3" if spectrum else "D2"
        seeds = (
            self.config["D3"]["training_seeds"]
            if spectrum
            else self.config["training_seeds"]
        )
        specs, complete, supplementary = [], [], []
        for task in self.registration["source_tasks"]["outer"]:
            fold = task["fold_id"]
            for seed in seeds:
                methods = self.methods(fold, seed, spectrum)
                primary = [
                    method for method in methods if not method.endswith("_COMMON")
                ]
                group = slot_specs(phase, [task], [seed], primary)
                specs.extend(group)
                supplementary.extend(
                    slot_specs(
                        phase,
                        [task],
                        [seed],
                        [method for method in methods if method.endswith("_COMMON")],
                    )[1:]
                )
                for spec in group:
                    selected = (
                        None
                        if spec["K"] == 0
                        else self.guard(
                            str(slot_key(spec)),
                            self.selection,
                            phase,
                            fold,
                            seed,
                            spec["method"],
                            spec["K"],
                        )
                    )
                    if spec["K"] and selected is None:
                        continue
                    receipt = self.guard(
                        str(slot_key(spec)),
                        self.verify_unit,
                        spec,
                        selected["hyperparameters"] if selected else {},
                    )
                    if receipt is not None:
                        complete.append(receipt)
        expected = matrix_counts(specs)
        if phase == "D2" and expected["nonzero_slots"]:
            self.check(
                expected["nonzero_slots"] == 945
                and expected["unique_nonzero_support_states"] == 861,
                phase,
                "D2 primary slot or distinct Support-state matrix differs",
            )
        if phase == "D3" and not self.na and expected["nonzero_slots"]:
            self.check(
                expected["nonzero_slots"] == 135,
                phase,
                "Estimable D3 must retain its 135 registered slots",
            )
        self.check(
            expected["slots"] == expected["unique_slot_keys"],
            phase,
            "Repeated source slot keys exist",
        )
        self.counts[phase] = {
            **expected,
            "delivered_slots": len(complete),
            "delivered_unique_nonzero_states": len(
                {row["adaptation_id"] for row in complete if row["K"] > 0}
            ),
            "common_supplementary_slots": len(supplementary),
        }
        if (
            phase == "D2"
            and expected["nonzero_slots"] == 945
            or phase == "D3"
            and all(
                (str(fold), seed, True) in self.dictionary_metadata
                for fold in range(3)
                for seed in seeds
            )
        ):
            self.verify_inventory(phase, [*specs, *supplementary])
        for spec in supplementary:
            selected = self.guard(
                str(slot_key(spec)),
                self.selection,
                phase,
                spec["task"]["fold_id"],
                spec["seed"],
                spec["method"],
                spec["K"],
            )
            if selected is not None:
                self.guard(
                    str(slot_key(spec)),
                    self.verify_unit,
                    spec,
                    selected["hyperparameters"],
                )

    def verify_target(self, require_scored):
        lock = self.lock()
        if lock is None:
            return
        methods = ["CAL2", "OUT", lock["best_generic"]]
        tasks = [
            task
            for task in self.registration["target_tasks"]
            if task["domain"] in self.config["D4"]["domains"]
        ]
        specs = slot_specs("D4", tasks, self.config["training_seeds"], methods)
        expected = matrix_counts(specs)
        self.check(
            expected
            == {
                "slots": 690,
                "unique_slot_keys": 690,
                "zero_references": 15,
                "nonzero_slots": 675,
                "unique_nonzero_support_states": 549,
            },
            "D4",
            "Primary target slot or distinct Support-state matrix differs",
        )
        self.verify_inventory("D4", specs)
        receipts = []
        for spec in specs:
            hyper = (
                {}
                if not spec["K"]
                else lock["selected_hyperparameters"][spec["method"]][str(spec["K"])]
            )
            receipt = self.guard(
                str(slot_key(spec)), self.verify_unit, spec, hyper, require_scored
            )
            if receipt is not None:
                receipts.append(receipt)
        self.counts["D4"] = {
            **expected,
            "delivered_slots": len(receipts),
            "scored_slots": sum(row["status"] == "SCORED" for row in receipts),
            "delivered_unique_nonzero_states": len(
                {row["adaptation_id"] for row in receipts if row["K"] > 0}
            ),
        }
        exposure_path = self.args.public_root / "TARGET_QUERY_EXPOSURE.json"
        exposure = (
            self.read(exposure_path)
            if require_scored or exposure_path.exists()
            else None
        )
        if exposure is not None:
            query_ids = {
                domain: sorted(
                    {
                        uid
                        for task in tasks
                        if task["domain"] == domain
                        for uid in task["query_ids"]
                    }
                )
                for domain in self.config["D4"]["domains"]
            }
            self.check(
                exposure["status"] == "FIXED_TARGET_QUERY_EXPOSED_BY_REGISTERED_SCORING"
                and exposure["query_ids"] == query_ids,
                exposure_path,
                "Target exposure identities or status differ from registered fixed Queries",
            )
            self.check(
                len(receipts) == 690
                and exposure["all_target_prediction_units_locked_before_scoring"]
                == 690,
                exposure_path,
                "Target exposure requires every prediction unit",
            )
            self.check(
                exposure["target_lock_sha256"]
                == file_hash(self.args.public_root / "TARGET_LOCK.json")
                and exposure["query_labels_used_for_adaptation_or_selection"] == 0
                and exposure["reuse_as_blind_confirmation"] is False,
                exposure_path,
                "Target lock and exposure information scope differ",
            )
            if len(receipts) == 690:
                prediction_hash = payload_hash(
                    [
                        {
                            "adaptation_id": row["adaptation_id"],
                            "prediction_sha256": row["prediction_sha256"],
                        }
                        for row in receipts
                    ]
                )
                self.check(
                    exposure["prediction_receipts_sha256"] == prediction_hash,
                    exposure_path,
                    "Target exposure refers to another prediction matrix",
                )

    def run(self, stages):
        actions = {
            "cv-references": lambda: self.verify_references(False),
            "final-references": lambda: self.verify_references(True),
            "basis": self.verify_basis,
            "inner": self.verify_inner,
            "D2": self.verify_source,
            "D3": lambda: self.verify_source(True),
            "target-lock": self.verify_target_lock,
            "target-predictions": lambda: self.verify_target(False),
            "target": lambda: self.verify_target(True),
        }
        for stage in stages:
            self.guard(stage, actions[stage])
        errors = sum(item["kind"] == "ERROR" for item in self.issues)
        missing = sum(item["kind"] == "MISSING" for item in self.issues)
        return {
            "status": "FAILED" if errors else "PARTIAL" if missing else "COMPLETE",
            "requested_stages": stages,
            "errors": errors,
            "pending_artifacts": missing,
            "counts": self.counts,
            "D3_not_estimable": self.na,
            "actual_parameter_records": dict(
                Counter(
                    f"{row['method']}:{row['actual_trainable_parameters']}"
                    for row in self.parameter_records
                )
            ),
            "issues": self.issues,
            "checked_distinct_file_hashes": len(self.hashes),
            "dataset_loads": 0,
            "Query_future_labels_read": 0,
            "model_initializations": 0,
            "scientific_performance_stopping_gates": 0,
            "verification_scope": "artifact identity, completeness, stored parameter budgets and fixed prediction axes; no prediction rescoring or model replay",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=(*STAGES, "references", "source", "all"),
        default=["all"],
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--private-root", type=Path, default=PRIVATE)
    parser.add_argument("--public-root", type=Path, default=PUBLIC)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.reference_root = args.reference_root or args.private_root / "references"
    stages = set(args.stage)
    if "all" in stages:
        stages.update(STAGES)
        stages.discard("target-predictions")
    if "references" in stages:
        stages.update(("cv-references", "final-references"))
    if "source" in stages:
        stages.update(("D2", "D3"))
    selected = [stage for stage in STAGES if stage in stages]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    registration = load_registration(ROOT / config["episodes_file"])
    report = DeliveryVerifier(args, config, registration).run(selected)
    write_json(args.output or args.public_root / "DELIVERY_VERIFICATION.json", report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("issues", "actual_parameter_records")
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    raise SystemExit({"COMPLETE": 0, "PARTIAL": 2, "FAILED": 1}[report["status"]])


if __name__ == "__main__":
    main()
