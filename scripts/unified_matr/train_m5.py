"""Single-arm M5 training wrapper. Reuse M4's frozen-data QA and six-split runner.

No changes to the archived M4 source/weights or the frozen MATR features.
"""
from __future__ import annotations

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


def m5_contract(data, data_root, config_path, methodology_path):
    from models.unified_matr.m5_bounded import model_registry
    out = original_contract(data, data_root, config_path, methodology_path)
    model = build_model(data[0], out["training"])
    out["study_id"] = "unified_matr_m5_bounded_seed42"
    out["config"]["study_id"] = "unified_matr_m5_bounded_seed42"
    out["model_implementation_registry"] = model_registry(model)
    out["computational_graph"]["I3"] = "bounded relative degradation intensity exp(-integral lambda), soft transition not knee"
    out["computational_graph"]["rate_unit"] = "reported absolute SOH per1000 cycles; internal relative hazard per1000 cycles"
    out["computational_graph"]["hazard_bounds"] = [0.0, 2.0]
    return out


def main():
    base.ARM = ARM
    base.DEPENDENCIES = tuple(base.DEPENDENCIES) + (MODEL_SOURCE, TRAINER_SOURCE)
    base.make_model = build_model
    base.model_contract = m5_contract
    base.main()


if __name__ == "__main__":
    main()
