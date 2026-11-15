"""Post-score protocol-support diagnostics from the already frozen predictions."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_experiment_b import metric_values
from run_d0 import write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main():
    support_rows, metric_rows, comparison_rows = [], [], []
    for family, directory in (("MATR", "experiment_b_matr"), ("Stanford_2", "experiment_b_stanford")):
        output = ROOT / "results" / directory
        split = {row["cell_id"]: row for row in read(output / "split.csv")}
        support = read(output / "target_cell_support.csv")
        predictions = read(output / "predictions.csv")
        for target in ("r200_500", "r500_1000"):
            available = {row["cell_id"] for row in support if row["target"] == target and row["observed_target"] == "True"}
            groups = defaultdict(Counter)
            for identifier in available:
                row = split[identifier]
                groups[row["protocol_group"]][row["role"]] += 1
            for group, counts in groups.items():
                support_rows.append({"family": family, "target": target, "protocol_group": group,
                                     "train_cells": counts["train"], "validation_cells": counts["val"], "test_cells": counts["test"],
                                     "test_group_seen_in_training": counts["train"] > 0,
                                     "analysis_stage": "postscore_descriptive"})
            tests = [row for row in predictions if row["target"] == target]
            for stratum in ("seen_training_protocol", "unseen_training_protocol"):
                selected = [row for row in tests if (groups[row["protocol_group"]]["train"] > 0) == (stratum == "seen_training_protocol")]
                if not selected:
                    continue
                by_model = defaultdict(dict)
                for row in selected:
                    by_model[row["model"]][row["cell_id"]] = row
                for model, cells in by_model.items():
                    values = list(cells.values())
                    metric_rows.append({"family": family, "target": target, "stratum": stratum,
                                        "model": model, **metric_values([float(r["truth"]) for r in values], [float(r["prediction"]) for r in values]),
                                        "analysis_stage": "postscore_descriptive"})
                baseline, candidate = by_model["context_capacity"], by_model["combined"]
                identifiers = sorted(baseline)
                if set(identifiers) != set(candidate):
                    raise RuntimeError("Frozen model support mismatch in protocol diagnostic")
                differences = np.array([abs(float(baseline[i]["prediction"]) - float(baseline[i]["truth"])) - abs(float(candidate[i]["prediction"]) - float(candidate[i]["truth"])) for i in identifiers])
                comparison_rows.append({"family": family, "target": target, "stratum": stratum,
                                        "test_cells": len(identifiers), "test_protocol_groups": len({baseline[i]["protocol_group"] for i in identifiers}),
                                        "MAE_improvement": float(differences.mean()), "improved_cells": int(np.sum(differences > 0)),
                                        "worsened_cells": int(np.sum(differences < 0)), "analysis_stage": "postscore_descriptive"})
    destination = ROOT / "results" / "extension_analysis"
    write_csv(destination / "protocol_support.csv", support_rows)
    write_csv(destination / "protocol_stratum_metrics.csv", metric_rows)
    write_csv(destination / "protocol_stratum_comparisons.csv", comparison_rows)
    write_json(destination / "DIAGNOSTIC_SCOPE.json", {"status": "COMPLETE_POSTSCORE_DESCRIPTIVE", "model_fits": 0,
                                                       "source": "frozen per-cell predictions and registered target-specific role masks",
                                                       "strata": "test protocol tuple previously observed in target-specific train cells or absent from those cells"})
    print(comparison_rows)


if __name__ == "__main__":
    main()
