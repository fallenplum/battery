"""Per-cell-weighted local solves and actual nonlinear support fine-tuning."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from models.drs_peft.onesided import (
    CalibrationBatteryTSFM,
    DictionaryBatteryTSFM,
    LowRankBatteryTSFM,
    calibration_jacobian,
    dictionary_jacobian,
)
from scripts.drs_peft.train import selected_state


def observed_positions(observed, max_points=512):
    """Take deterministic equidistant entries from actual observed positions."""
    mask = np.asarray(observed, dtype=bool)
    if mask.ndim != 1 or int(max_points) < 1:
        raise ValueError("One physical cell and a positive point budget are required")
    available = np.flatnonzero(mask)
    if not len(available):
        raise ValueError("A Support cell has no actual future supervision")
    if len(available) <= int(max_points):
        return available
    selected = np.rint(np.linspace(0, len(available) - 1, int(max_points))).astype(
        np.int64
    )
    return available[selected]


def cache_local_design(model, early, query, backend="autograd"):
    """Build a label-free local design through the normal model entry point."""
    started = time.monotonic()
    if isinstance(model, DictionaryBatteryTSFM):
        prediction, design = dictionary_jacobian(model, early, query, backend=backend)
    elif isinstance(model, CalibrationBatteryTSFM):
        prediction, design = calibration_jacobian(model, early, query)
        backend = "analytic_CAL2"
    else:
        raise TypeError("Local design caching supports geometric arms and CAL2")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(design).all()):
        raise FloatingPointError("The local prediction or Jacobian is nonfinite")
    return {
        "f0": prediction.detach().cpu().numpy(),
        "design": design.detach().cpu().numpy(),
        "backend": backend,
        "seconds": time.monotonic() - started,
        "observations": len(prediction),
        "dimension": design.shape[1],
    }


def ridge_solve(designs, residuals, lambda_multiplier, eta=1.0, epsilon=1e-12):
    """Solve exact cell-mean Gram equations in FP64 with a scaled ridge prior."""
    if len(designs) != len(residuals) or not len(designs):
        raise ValueError("Every physical Support cell needs one design and residual")
    if not all(
        math.isfinite(float(value)) for value in (lambda_multiplier, eta, epsilon)
    ):
        raise ValueError("Local solve hyperparameters must be finite")
    if float(lambda_multiplier) <= 0 or float(epsilon) <= 0 or not 0 < float(eta) <= 1:
        raise ValueError("Ridge, numerical floor and damping must be positive")
    matrices = [
        torch.as_tensor(value, dtype=torch.float64, device="cpu").detach()
        for value in designs
    ]
    errors = [
        torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().reshape(-1)
        for value in residuals
    ]
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("Each physical cell needs a two-dimensional design")
    dimension = matrices[0].shape[1]
    if dimension < 1:
        raise ValueError("The actual local dimension must be positive")
    gram = torch.zeros(dimension, dimension, dtype=torch.float64)
    right_hand_side = torch.zeros(dimension, dtype=torch.float64)
    for matrix, error in zip(matrices, errors, strict=True):
        if (
            matrix.ndim != 2
            or matrix.shape != (len(error), dimension)
            or not len(error)
        ):
            raise ValueError("A local design must match that cell's observed residual")
        if not bool(torch.isfinite(matrix).all() and torch.isfinite(error).all()):
            raise ValueError("The local system contains a nonfinite value")
        gram += matrix.T @ matrix / len(error)
        right_hand_side += matrix.T @ error / len(error)
    gram /= len(matrices)
    right_hand_side /= len(matrices)
    gram = (gram + gram.T) / 2
    mean_diagonal = float(gram.diag().mean())
    ridge = float(lambda_multiplier) * max(mean_diagonal, float(epsilon))
    system = gram + ridge * torch.eye(dimension, dtype=torch.float64)
    factor, information = torch.linalg.cholesky_ex(system)
    if int(information) != 0:
        raise FloatingPointError(
            "The registered regularized local system is not positive definite"
        )
    solution = torch.cholesky_solve(right_hand_side[:, None], factor).squeeze(1)
    return {
        "a": (float(eta) * solution).numpy(),
        "undamped": solution.numpy(),
        "gram": gram.numpy(),
        "rhs": right_hand_side.numpy(),
        "gram_mean_diagonal": mean_diagonal,
        "lambda_actual": ridge,
        "lambda_multiplier": float(lambda_multiplier),
        "epsilon": float(epsilon),
        "eta": float(eta),
        "physical_support_cells": len(matrices),
        "observations_per_cell": [len(error) for error in errors],
        "dimension": dimension,
        "solver": "FP64_Cholesky",
    }


def _support_predictions(model, support):
    return [model(cell["early"], cell["query"])[0] for cell in support]


def train_lowrank_support(model, support, learning_rate, save_steps=(10, 30)):
    """Run one nonlinear trajectory fit and retain both registered step states."""
    if not isinstance(model, LowRankBatteryTSFM) or not len(support):
        raise ValueError(
            "An appended low-rank model and nonempty physical Support set are required"
        )
    steps = tuple(sorted({int(value) for value in save_steps}))
    if not steps or steps[0] < 1 or float(learning_rate) <= 0:
        raise ValueError("Positive learning rate and optimizer budgets are required")
    parameters = [value for value in model.parameters() if value.requires_grad]
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if any(".lora_a." not in name and ".lora_b." not in name for name in names):
        raise ValueError("Support fine-tuning attempted to update the source function")
    model.set_adaptation_scale(1.0)
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=0.0)
    history, snapshots = [], {}
    started = time.monotonic()
    for step in range(1, steps[-1] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for cell in support:
            if len(cell["early"]["x"]) != 1:
                raise ValueError("Each Support entry must identify one physical cell")
            prediction = model(cell["early"], cell["query"])[0]
            target = torch.as_tensor(
                cell["target"], device=prediction.device, dtype=prediction.dtype
            ).reshape(-1)
            if len(target) != len(prediction) or not bool(torch.isfinite(target).all()):
                raise ValueError(
                    "The sampled Support target does not match its physical cycle grid"
                )
            losses.append((prediction - target).square().mean())
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("The nonlinear Support objective is nonfinite")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "step": step,
                "cell_mean_MSE_before_update": float(loss.detach()),
                "gradient_norm_before_clip": float(norm),
            }
        )
        if step in steps:
            model.eval()
            with torch.no_grad():
                predictions = [
                    value.detach().cpu().numpy()
                    for value in _support_predictions(model, support)
                ]
            snapshots[step] = {
                "model": selected_state(model),
                "support_predictions": predictions,
                "optimizer_updates": step,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rate": float(learning_rate),
                "weight_decay": 0.0,
                "gradient_clip": 1.0,
                "physical_support_cells": len(support),
                "observations_per_cell": [len(value) for value in predictions],
            }
    return {
        "snapshots": snapshots,
        "history": history,
        "optimizer_updates": steps[-1],
        "physical_support_cells": len(support),
    }


def actual_local_predictions(model, early, query, solution, cached_design=None):
    """Evaluate the real nonlinear model and retain its local prediction gap."""
    if not isinstance(model, (DictionaryBatteryTSFM, CalibrationBatteryTSFM)):
        raise TypeError("Local injection requires a geometric arm or CAL2")
    coordinates = torch.as_tensor(
        solution, device=model.coordinates.device, dtype=model.coordinates.dtype
    )
    if coordinates.shape != model.coordinates.shape:
        raise ValueError("The solved coordinate dimension differs from the model")
    model.eval()
    with torch.no_grad():
        model.coordinates.copy_(coordinates)
        actual = model(early, query)[0].detach().cpu().numpy()
    if not np.isfinite(actual).all():
        raise FloatingPointError("The actual nonlinear adapted prediction is nonfinite")
    output = {"actual": actual}
    if cached_design is not None:
        linear = np.asarray(cached_design["f0"], dtype=np.float64) + np.asarray(
            cached_design["design"], dtype=np.float64
        ) @ np.asarray(solution, dtype=np.float64)
        if linear.shape != actual.shape:
            raise ValueError("The local cache corresponds to another prediction grid")
        gap = actual.astype(np.float64) - linear
        output.update(
            linear=linear,
            linearization_gap=gap,
            gap_MAE=float(np.abs(gap).mean()),
            gap_maximum_absolute=float(np.abs(gap).max()),
        )
    return output
