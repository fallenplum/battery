"""Record exact first-batch M4 forward/backward repeatability with zero updates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/unified_matr"))
import train_methodology_full as runner
import train_seed42 as old


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    runner.configure_execution()
    device = torch.device("cuda")
    data = old.load_frozen_data(ROOT / "data/unified_matr_seed42")
    inputs, ids, eligible, _, query, targets, observed, rate_targets, rate_mask, splits, _ = data
    config = json.loads((ROOT / "configs/unified_matr_seed42.json").read_text(encoding="utf-8"))
    roles = splits["splits"]["cell_fold_0"]
    train = np.asarray([ids.index(uid) for uid in roles["train"] if eligible[ids.index(uid)]])
    scaled, _ = old.fold_scaler(inputs, train)
    rows = np.random.default_rng(42).permutation(train)[:8]
    records, gradients = [], []
    for _ in range(args.repeats):
        old.seed_everything()
        model = runner.make_model(inputs, config["training"]).to(device)
        model.train()
        output = model(**old.to_batch(scaled, rows, device), query_cycles=torch.as_tensor(query, device=device), return_diagnostics=True)
        loss = runner.training_loss(output, rows, targets, observed, rate_targets, rate_mask, config["training"], device)
        loss.backward()
        gradients.append({name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                          for name, parameter in model.named_parameters()})
        records.append({"loss": float(loss.detach()), "prediction_sha256": hashlib.sha256(output["trajectory"].detach().cpu().numpy().tobytes()).hexdigest(),
                        "gradient_sha256": runner.gradient_receipt(model)["sha256"]})
        del output, model
    comparisons = []
    for number, actual in enumerate(gradients[1:], 1):
        differences = []
        for name, first in gradients[0].items():
            last = actual[name]
            if first is None or last is None:
                if first is not None or last is not None:
                    differences.append({"parameter": name, "missing_gradient_mismatch": True})
            elif not torch.equal(first, last):
                differences.append({"parameter": name, "different_elements": int(torch.count_nonzero(first != last)),
                                    "max_absolute_difference": float((first - last).abs().max()),
                                    "reference_max_absolute_value": float(first.abs().max())})
        comparisons.append({"repeat": number + 1, "exact_gradients": not differences, "differences": differences})
    result = {"tag": args.tag, "optimizer_updates": 0, "formal_updates": 0, "cell_uid": [ids[number] for number in rows],
              "records": records, "comparisons": comparisons,
              "all_forward_loss_and_gradients_exact": all(item == records[0] for item in records[1:]),
              "model_sources": [{"path": name, "sha256": digest(ROOT / name)} for name in
                                ("models/unified_matr/methodology_full.py", "models/unified_matr/methodology_operators.py")],
              "strict_determinism": torch.are_deterministic_algorithms_enabled(), "source_sha256": digest(Path(__file__))}
    target = ROOT / "results/unified_matr_methodology_full_seed42/engineering_diagnostics" / f"{args.tag}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError("Diagnostic receipt already exists")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tag": args.tag, "exact": result["all_forward_loss_and_gradients_exact"],
                      "different_parameter_tensors": [len(row["differences"]) for row in comparisons], "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
