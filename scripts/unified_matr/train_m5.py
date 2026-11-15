"""Single-arm M5 training wrapper. Reuse M4's frozen-data QA and six-split runner.

No changes to the archived M4 source/weights or the frozen MATR features.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.unified_matr import train_methodology_full as base

ARM = "M5_BOUNDED_TRANSITION"
MODEL_SOURCE = "models/unified_matr/m5_bounded.py"
TRAINER_SOURCE = "scripts/unified_matr/train_m5.py"


def build_model(inputs, settings):
    from models.unified_matr.m5_bounded import M5BoundedModel
    return M5BoundedModel(policy_dim=inputs["policy"].shape[-1],
                          global_dim=inputs["global_features"].shape[-1],
                          local_feature_dim=inputs["local_features"].shape[-1], seed=42,
                          observation_bound=settings["observation_residual_amplitude"],
                          soh_bounds=tuple(settings["observation_soh_bounds"]))


original_contract = base.model_contract
ORIGINAL_DEPENDENCIES = base.DEPENDENCIES
SOURCE_COMMIT = "85c6ff92301a2e14f9ae72299f672f34a26bebe3"


@contextmanager
def m5_runner_scope():
    """Restore the shared runner even when a delegated invocation fails."""
    names = ("ARM", "DEPENDENCIES", "make_model", "model_contract")
    saved = {name: getattr(base, name) for name in names}
    base.ARM = ARM
    base.DEPENDENCIES = ORIGINAL_DEPENDENCIES + (MODEL_SOURCE, TRAINER_SOURCE)
    base.make_model = build_model
    base.model_contract = m5_contract
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def m5_contract(data, data_root, config_path, methodology_path):
    from models.unified_matr.m5_bounded import model_registry
    with m5_runner_scope():
        out = original_contract(data, data_root, config_path, methodology_path)
    model = build_model(data[0], out["training"])
    out["study_id"] = "unified_matr_m5_bounded_seed42"
    out["config"]["study_id"] = "unified_matr_m5_bounded_seed42"
    out["model_implementation_registry"] = model_registry(model)
    out["computational_graph"]["I3"] = "bounded relative degradation intensity exp(-integral lambda), soft transition not knee"
    out["computational_graph"]["rate_unit"] = "reported absolute SOH per1000 cycles; internal relative hazard per1000 cycles"
    out["computational_graph"]["hazard_bounds"] = [0.0, 2.0]
    out["computational_graph"]["hazard_coefficient_bounds"] = [[-2.0, 2.0], [-1.5, 1.5], [-1.0, 1.0], [-2.0, 2.0]]
    out["computational_graph"]["soft_center_relative_thousand_cycles_bounds"] = [0.1, 3.6]
    out["computational_graph"]["soft_width_thousand_cycles_bounds"] = [0.05, 0.85]
    out["computational_graph"]["head_dimensions"] = [32, 32, 6]
    out["computational_graph"]["initial_relative_hazard_per_1000_cycles"] = 0.1
    out["computational_graph"]["soft_transition_supervision"] = "none; no certified physical knee labels"
    out["upstream_source_commit"] = SOURCE_COMMIT
    out["implementation_corrections"] = ["shared runner globals are transactionally restored"]
    out["training_budget"] = {"formal_optimizer_updates": 12450,
                              "engineering_optimizer_updates": 14,
                              "full_reproduction_optimizer_updates": 2100,
                              "total_optimizer_updates": 14564}
    out["historical_references"] = [
        {"arm": "M1", "root": "results/unified_matr_seed42/runs"},
        {"arm": "A1_M4_LEGACY_FLOW", "root": "results/unified_matr_ablations_seed42/runs"},
        {"arm": "M4_FULL_METHOD", "root": "results/unified_matr_methodology_full_seed42/runs"},
    ]
    return out


def main():
    with m5_runner_scope():
        base.main()


if __name__ == "__main__":
    main()
