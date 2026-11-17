"""Independent C-matrix reconstruction and A/B/C audit routing regressions."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from models.drs_peft.abc_subspace import ARMS, SubspaceProbeLinear
from models.drs_peft.peft import SubspaceLinear, random_orthogonal_bases
from scripts.drs_peft import audit_abc
from scripts.drs_peft.e123_protocol import load_protocol
from scripts.drs_peft.run_abc_b import role_derangement

torch.set_num_threads(2)


def probe_layer(arm):
    torch.manual_seed(42)
    u, v = random_orthogonal_bases(384, 384, 40, seed=100)
    source = SubspaceLinear(
        nn.Linear(384, 384),
        u[:, :4],
        v[:, :4],
        core=torch.randn(4, 4) * 0.03,
    )
    initial_a = torch.full((384,), 0.01)
    initial_a[7] = -1
    bases = {
        "fixed_u": u[:, :25],
        "fixed_v": v[:, :25],
        "perp_u": u[:, 4:25],
        "perp_v": v[:, 4:33],
        "lora_a": initial_a,
    }
    layer = SubspaceProbeLinear(source, arm, bases).eval()
    with torch.no_grad():
        layer.delta_core.normal_(mean=0, std=0.02)
        if layer.core_perp is not None:
            layer.core_perp.normal_(mean=0, std=0.03)
        if layer.core_a_free is not None:
            layer.core_a_free.add_(0.025)
    return layer


def repeated_probe_state(layer):
    return {
        f"bridge.backbone.module.layer{index}.{name}": value.detach().cpu().clone()
        for index in range(12)
        for name, value in layer.state_dict().items()
    }


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("evaluation", ("trained_ball", "matched_norm"))
def test_numpy_c_norm_matches_actual_operator_for_all_five_arms(arm, evaluation):
    layer = probe_layer(arm)
    layer.evaluation_mode = evaluation
    state = repeated_probe_state(layer)
    independent = audit_abc.independent_subspace_norms(state, ["cell0", "cell1"])
    actual = layer.update_diagnostics(2)
    assert len(independent) == 24
    np.testing.assert_allclose(
        independent.independent_raw_increment_norm,
        actual["raw_increment_norm"][0].item(),
        rtol=3e-6,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        independent.radius, actual["radius"][0].item(), rtol=0, atol=0
    )
    if evaluation == "matched_norm":
        expected_norm = np.where(
            independent.independent_raw_increment_norm > 0, independent.radius, 0
        )
    else:
        expected_norm = np.minimum(
            independent.independent_raw_increment_norm, independent.radius
        )
    np.testing.assert_allclose(
        expected_norm, actual["increment_norm"][0].item(), rtol=3e-6, atol=3e-10
    )
    assert set(independent.layer) == {f"layer{index}" for index in range(12)}


def test_numpy_gauge_fixed_rank1_uses_the_registered_nonzero_anchor_slot():
    layer = probe_layer("matched_lora")
    assert layer.core_anchor.item() == 7
    state = repeated_probe_state(layer)
    independent = audit_abc.independent_subspace_norms(state, ["cell0"])
    raw_matrix = (
        layer.delta_core.detach().double()[:, None]
        * layer.lora_a().detach().double()[None]
    )
    np.testing.assert_allclose(
        independent.independent_raw_increment_norm,
        raw_matrix.norm().item(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_numpy_zero_increment_preserves_zero_raw_and_matched_support():
    layer = probe_layer("rank4_plus_small")
    with torch.no_grad():
        layer.delta_core.zero_()
        layer.core_perp.zero_()
    layer.evaluation_mode = "matched_norm"
    independent = audit_abc.independent_subspace_norms(
        repeated_probe_state(layer), ["cell0"]
    )
    assert independent.independent_raw_increment_norm.eq(0).all()
    assert layer.update_diagnostics(1)["matched_norm_unattainable"].all()
    assert layer.update_diagnostics(1)["increment_norm"].eq(0).all()


def shuffle_fixture():
    fold = {
        "fold_id": 1,
        "fit_ids": ["fit0", "fit1", "fit2"],
        "inner_validation_ids": ["inner0", "inner1"],
        "outer_test_ids": ["outer0", "outer1", "outer2"],
    }
    mapping = []
    for role, key in (
        ("fit", "fit_ids"),
        ("inner_validation", "inner_validation_ids"),
        ("outer_test", "outer_test_ids"),
    ):
        ids = fold[key]
        donors = role_derangement(ids, 42, fold["fold_id"], role)
        mapping.extend(
            {"role": role, "cell_uid": uid, "feature_donor_uid": ids[donors[index]]}
            for index, uid in enumerate(ids)
        )
    return fold, mapping


def test_independent_derangement_matches_frozen_role_seed_uid_definition():
    fold, mapping = shuffle_fixture()
    audit_abc.verify_derangement(mapping, fold, 42)
    audit_abc.verify_derangement(list(reversed(mapping)), fold, 42)


def test_independent_derangement_rejects_misaligned_donor():
    fold, mapping = shuffle_fixture()
    altered = copy.deepcopy(mapping)
    altered[0]["feature_donor_uid"] = altered[0]["cell_uid"]
    with pytest.raises(AssertionError):
        audit_abc.verify_derangement(altered, fold, 42)


def test_independent_derangement_rejects_duplicate_mapping_rows():
    fold, mapping = shuffle_fixture()
    with pytest.raises(AssertionError):
        audit_abc.verify_derangement(mapping + [dict(mapping[0])], fold, 42)


def test_independent_derangement_rejects_unknown_roles():
    fold, mapping = shuffle_fixture()
    with pytest.raises(AssertionError):
        audit_abc.verify_derangement(
            mapping
            + [
                {
                    "role": "final_target",
                    "cell_uid": "target0",
                    "feature_donor_uid": "target1",
                }
            ],
            fold,
            42,
        )


def test_registered_a_b_c_audit_has_135_distinct_models_and_99_new_trainings():
    config = audit_abc.read(audit_abc.ROOT / "configs/drs_peft/abc_v1.json")
    protocol = load_protocol()
    records = audit_abc.registered_units(config, protocol, ["A", "B", "C"])
    assert len(records) == 135
    assert sum(record["reused"] for record in records) == 36
    assert sum(not record["reused"] for record in records) == 99
    args = SimpleNamespace(private_root=Path("new"), e123_root=Path("legacy"))
    folders = [audit_abc.source_folder(args, record) for record in records]
    assert len(set(folders)) == 135


@pytest.mark.parametrize(
    "study,regime,reused",
    [
        ("A", "joint", False),
        ("B", "frozen", False),
        ("A", "frozen", True),
        ("C", None, False),
    ],
)
def test_source_audit_routes_match_latest_runner_schema(study, regime, reused):
    args = SimpleNamespace(private_root=Path("new"), e123_root=Path("legacy"))
    if study == "C":
        unit = {"fold_id": 0, "seed": 42, "arm": "matched_residual"}
        expected = Path("new/C/fold0/seed42/matched_residual")
    else:
        mode = "shuffled_dual" if study == "B" else "dual"
        unit = {
            "stage": "condition",
            "fold_id": 0,
            "seed": 42,
            "method": "grad_core",
            "radius": 1,
            "mode": mode,
            "regime": regime,
        }
        root = Path("legacy") if reused else Path("new") / study / regime
        expected = root / "fold0/seed42/condition/grad_core/radius1" / mode
    actual = audit_abc.source_folder(
        args, {"study": study, "unit": unit, "reused": reused}
    )
    assert actual == expected
