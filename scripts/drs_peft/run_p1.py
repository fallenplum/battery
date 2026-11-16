"""Run source-only P1 from the frozen P0 B0 selection, then aggregate seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch
from filelock import FileLock

from .data import ROOT, file_hash, write_json
from .gradient_audit import aggregate_audits, run_audit
from .train import (
    DEFAULT_CONFIG,
    bind,
    configure,
    load_selected,
    load_source,
    make_model,
    read_json,
)


def require_gate(path, stage):
    gate = read_json(path)
    accepted = gate.get("status") == "PASSED"
    if stage == "P0":
        accepted = (
            gate.get("status") in ("PASSED", "NEGATIVE")
            and gate.get("execution_complete") is True
        )
    if not accepted:
        raise RuntimeError(f"{stage} stage gate has not passed at {path}")
    return gate


def load_p0_checkpoint(output_root, bridge, seed, device="cpu"):
    folder = (
        Path(output_root)
        / "P0"
        / "formal"
        / bridge
        / "B0_pretrained_frozen"
        / f"seed{seed}"
    )
    receipt = read_json(folder / "SOURCE_DEVELOPMENT_COMPLETE.json")
    path = folder / "CHECKPOINT_SELECTED.pt"
    if receipt["selected_checkpoint_sha256"] != file_hash(path):
        raise RuntimeError("P0 selected checkpoint hash changed")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    identity = checkpoint["identity"]
    expected = dict(
        stage="P0", method="B0_pretrained_frozen", bridge=bridge, seed=int(seed)
    )
    if any(identity[key] != value for key, value in expected.items()):
        raise RuntimeError(
            "P1/P2 requires the matching formal P0 pretrained-frozen B0 checkpoint"
        )
    if "target_query_labels_used" in receipt:
        target_labels_used = receipt["target_query_labels_used"]
    elif "target_query_labels_read" in receipt:
        target_labels_used = receipt["target_query_labels_read"]
    else:
        raise RuntimeError(
            "P0 receipt lacks its target-Query information-budget declaration"
        )
    if target_labels_used != 0:
        raise RuntimeError(
            "The selected P0 source checkpoint accessed final target Query labels"
        )
    return checkpoint, path


def publish_audit(private, public):
    """Publish human-readable diagnostics and retain derived tensors in runtime."""
    private, public = Path(private), Path(public)
    public.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted(private.iterdir()):
        if (
            source.suffix.lower() not in (".json", ".csv", ".md", ".png", ".pdf")
            or source.name == "COMPLETE.json"
        ):
            continue
        destination = public / source.name
        if destination.exists() and file_hash(destination) != file_hash(source):
            raise RuntimeError("Published P1 diagnostic identity changed")
        shutil.copyfile(source, destination)
        copied.append(
            dict(
                path=destination.name,
                sha256=file_hash(destination),
                bytes=destination.stat().st_size,
            )
        )
    basis = private / "P1_BASES.pt"
    manifest = dict(
        status="PUBLISHED_SOURCE_DIAGNOSTIC",
        public_files=copied,
        private_audit_folder=str(private.resolve()),
        private_basis=dict(
            path=str(basis.resolve()),
            sha256=file_hash(basis),
            bytes=basis.stat().st_size,
        ),
        runtime_gradient_manifest="P1_CACHE_MANIFEST.json",
        private_tensors=[
            dict(
                path=str(path.resolve()),
                sha256=file_hash(path),
                bytes=path.stat().st_size,
            )
            for path in sorted(private.glob("*.pt"))
        ],
        target_Query_future_labels_accessed=False,
    )
    bind(public / "P1_PUBLIC_COMPLETE.json", manifest)
    return manifest


def audit_config(execution):
    p1 = execution["P1"]
    return dict(
        ranks=p1["ranks"],
        primary_rank=p1["primary_rank"],
        random_replicates=p1["random_basis_replicates"],
        basis_bootstrap_draws=p1["basis_bootstrap_refits"],
        bootstrap_draws=p1["bootstrap_draws"],
        absolute_control=p1["raw_and_nominal_units_control"],
        shuffle_control=True,
        candidate_family_diagnostic="qkvo_ffn"
        in p1["full_candidate_family_diagnostic"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "runtime/drs_peft/data_v1"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/drs_peft")
    parser.add_argument(
        "--private-root", type=Path, default=ROOT / "runtime/drs_peft/P1"
    )
    parser.add_argument(
        "--gradient-cache", type=Path, default=ROOT / "runtime/drs_peft/p1_gradients"
    )
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument("--p0-gate", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only and args.no_aggregate:
        parser.error("--aggregate-only conflicts with --no-aggregate")
    execution = read_json(args.config)
    selection = read_json(args.output_root / "P0/BRIDGE_SELECTION.json")
    bridge = selection["selected_bridge"]
    seeds = [args.seed] if args.seed is not None else execution["training_seeds"]
    if not set(seeds).issubset(execution["training_seeds"]):
        raise ValueError("P1 source seed was not registered")
    config = audit_config(execution)
    config["cache_root"] = str(args.gradient_cache)
    registration = dict(
        stage="P1",
        seeds=seeds,
        bridge=bridge,
        config=config,
        source_checkpoint_kind="formal B0 pretrained frozen, matching source seed",
        private_root=str(args.private_root),
        gradient_cache=str(args.gradient_cache),
        target_Query_future_labels_accessed=False,
        optimizer_updates=0,
    )
    if args.dry_run:
        print(registration, flush=True)
        return
    require_gate(args.p0_gate or args.output_root / "P0/P0_GATE.json", "P0")
    configure()
    if not args.aggregate_only:
        data, inputs, scaler, episodes = load_source(args.data_root)
        source_ids = episodes["source_cell_roles"]["source_train"]
        meta_ids = episodes["source_cell_roles"]["source_meta_validation"]
        config.update(source_ids=source_ids, validation_ids=meta_ids)
        for seed in seeds:
            checkpoint, path = load_p0_checkpoint(args.output_root, bridge, seed)
            model = make_model(
                args.backbone_dir,
                inputs["policy"].shape[-1],
                bridge,
                "B0_pretrained_frozen",
                seed,
            ).to(args.device)
            load_selected(model, checkpoint["model"])
            private = args.private_root / f"seed{seed}"
            run_audit(
                model,
                data,
                inputs,
                data.rows(source_ids),
                data.rows(meta_ids),
                private,
                seed,
                args.device,
                config,
            )
            publish_audit(private, args.output_root / "P1" / f"seed{seed}")
            del model, checkpoint
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    if args.no_aggregate:
        return
    lock = args.output_root / "P1/.locks/aggregate.execution.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock)):
        paths = [
            args.output_root / "P1" / f"seed{seed}"
            for seed in execution["training_seeds"]
            if (
                args.output_root / "P1" / f"seed{seed}" / "P1_PUBLIC_COMPLETE.json"
            ).exists()
        ]
        decision = aggregate_audits(
            paths,
            args.output_root / "P1",
            dict(
                required_seeds=execution["training_seeds"],
                primary_rank=config["primary_rank"],
                bootstrap_draws=config["bootstrap_draws"],
            ),
        )
    print(
        dict(stage="P1", status=decision["status"], seeds=decision["seeds"]), flush=True
    )


if __name__ == "__main__":
    main()
