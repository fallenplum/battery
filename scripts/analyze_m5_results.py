"""M5 bounded-head analysis with sealed historical M1, A1 and M4 references.

Generic arithmetic and completion/reproduction routines are adapted from the
sealed M4 analysis. Those historical sources remain byte-identical.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts.unified_matr import analyze_results as historical
from scripts.unified_matr import train_methodology_full as training
from scripts.unified_matr import train_m5
from scripts.analyze_methodology_ablations_seed42 import strict_nested_equal, strict_frame_equal
from scripts.analyze_methodology_full_seed42 import query_region_rows, summarize_query_regions

ARM = train_m5.ARM
SPLITS = training.SPLITS
REFERENCES = ("M1", "A1_M4_LEGACY_FLOW", "M4_FULL_METHOD")
read_json = historical.read_json
sha256 = training.sha256
write_json = training.write_json



def completed_units(output_root, data_root):
    """Authenticate the complete six-unit delivery before reading test tables."""
    run_root = output_root / "runs"
    matrix = read_json(run_root / "MATRIX_COMPLETE.json")
    contract = read_json(output_root / "MODEL_CONTRACT.json")
    registration = read_json(run_root / "RUN_REGISTRATION.json")
    if (matrix.get("status") != "COMPLETE" or matrix.get("completed_units") != 6
            or matrix.get("successful_updates") != 12450 or contract != registration
            or matrix["registration_sha256"] != sha256(run_root / "RUN_REGISTRATION.json")
            or matrix["model_contract_sha256"] != sha256(output_root / "MODEL_CONTRACT.json")
            or contract["data_freeze_sha256"] != sha256(data_root / "DATA_FREEZE.json")):
        raise RuntimeError("The complete registered six-unit delivery is required")
    for entry in contract["sources"]:
        if sha256(REPOSITORY / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Registered M5 source changed at " + entry["path"])
    for entry in contract["frozen_files"]:
        if sha256(data_root / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Frozen data changed at " + entry["path"])
    missing = [split for split in SPLITS if not (run_root / split / ARM / "COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError("Formal unit completion is pending for " + ", ".join(missing))
    units = []
    for split in SPLITS:
        root = run_root / split / ARM
        unit = read_json(root / "UNIT_CONTRACT.json")
        selection = read_json(root / "SELECTION_FROZEN.json")
        complete = read_json(root / "COMPLETE.json")
        if (unit["arm"] != ARM or unit["split_name"] != split
                or unit["registration_sha256"] != matrix["registration_sha256"]
                or complete["status"] != "EVALUATED"
                or complete["successful_updates"] != contract["expected_updates_by_split"][split]
                or selection["checkpoint_sha256"] != sha256(root / "SELECTED.pt")):
            raise RuntimeError("Unit identity, selection or budget differs at " + split)
        for entry in complete["files"]:
            if sha256(root / entry["path"]) != entry["sha256"]:
                raise RuntimeError("Completed unit artifact changed at " + split + "/" + entry["path"])
        units.append({"split": split, "arm": ARM, "root": root, "contract": unit, "selection": selection})
    return contract, units


def reproduction_comparison(output_root, contract):
    """Retain a failed complete repeat as a reported finding while analyzing."""
    import torch

    root = output_root / "training_reproduction"
    marker = read_json(root / "REPRODUCTION_COMPLETE.json")
    repeated = root / SPLITS[0] / ARM
    formal = output_root / "runs" / SPLITS[0] / ARM
    complete = read_json(repeated / "COMPLETE.json")
    if (marker.get("status") != "EVALUATED" or marker.get("additional_optimizer_updates") != 2100
            or marker["registration_sha256"] != sha256(root / "RUN_REGISTRATION.json")
            or read_json(root / "RUN_REGISTRATION.json") != contract
            or complete.get("successful_updates") != 2100):
        raise RuntimeError("The registered full fold0 reproduction is incomplete")
    for entry in complete["files"]:
        if sha256(repeated / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Full reproduction artifact changed at " + entry["path"])
    checks = []
    def record(name, equal):
        checks.append({"comparison": name, "exact_equal": bool(equal)})
    history_columns = ("epoch", "train_loss", "validation_mae", "selected", "successful_updates")
    histories = [[{key: row[key] for key in history_columns} for row in (
        json.loads(line) for line in (path / "training.jsonl").read_text(encoding="utf-8").splitlines())] for path in (formal, repeated)]
    if len(histories[1]) != 150:
        raise RuntimeError("The full duplicate must execute all150 epochs")
    record("all150_epoch_history_core", strict_nested_equal(histories[0], histories[1]))
    final = [torch.load(path / "TRAIN_STATE.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
    for key in ("model", "optimizer", "random_state", "selected_checkpoint", "next_epoch", "best_mae", "successful_updates", "contract"):
        record("final_" + key, strict_nested_equal(final[0][key], final[1][key]))
    selected = [torch.load(path / "SELECTED.pt", map_location="cpu", weights_only=False) for path in (formal, repeated)]
    record("selected_checkpoint_all_tensors_and_metadata", strict_nested_equal(selected[0], selected[1]))
    for name in ("UNIT_CONTRACT.json", "TRAIN_SCALER.json", "test_metrics.json"):
        record(name, strict_nested_equal(read_json(formal / name), read_json(repeated / name)))
    selections = [read_json(path / "SELECTION_FROZEN.json") for path in (formal, repeated)]
    record("selection_validation_and_epoch", strict_nested_equal(
        {k: v for k, v in selections[0].items() if k != "checkpoint_sha256"},
        {k: v for k, v in selections[1].items() if k != "checkpoint_sha256"}))
    for name in ("test_predictions.parquet", "test_cell_metrics.csv", "test_rate_predictions.csv", "test_eol_predictions.csv", "test_mechanism_summary.csv"):
        tables = [pd.read_parquet(path / name) if name.endswith(".parquet") else pd.read_csv(path / name) for path in (formal, repeated)]
        record(name, strict_frame_equal(tables[0], tables[1]))
    with np.load(formal / "test_mechanism_diagnostics.npz", allow_pickle=False) as first, np.load(repeated / "test_mechanism_diagnostics.npz", allow_pickle=False) as second:
        record("all_saved_mechanism_tensors", set(first.files) == set(second.files) and all(strict_nested_equal(first[key], second[key]) for key in first.files))
    return {"status": "PASS" if all(row["exact_equal"] for row in checks) else "FAIL",
            "split": SPLITS[0], "seed": 42, "epochs_per_fit": 150,
            "additional_optimizer_updates": 2100, "formal_optimizer_updates": 12450,
            "comparison_checks": checks, "repeat_complete_sha256": sha256(repeated / "COMPLETE.json"),
            "first_divergent_epoch": next((i + 1 for i, (a, b) in enumerate(zip(*histories)) if not strict_nested_equal(a, b)), None),
            "scope_limitation": "One full fold0 duplicate in one registered environment. Cross-seed variance is unmeasured."}


def fixed_comparisons(metrics):
    rows, pairs = [], []
    for scope in ("main_oof", "unseen_protocol_sensitivity"):
        scoped = metrics.loc[metrics.scope == scope]
        for metric in sorted(scoped.metric.unique()):
            wide = scoped.loc[scoped.metric == metric].pivot(index=["cell_uid", "protocol_uid"], columns="arm", values="value")
            for reference in REFERENCES:
                pair = wide[[reference, ARM]].rename(columns={reference: "reference_error", ARM: "candidate_error"}).reset_index()
                if not pair.reference_error.notna().equals(pair.candidate_error.notna()):
                    raise RuntimeError("Historical paired support differs")
                pair["improvement"] = pair.reference_error - pair.candidate_error
                pair = pair.assign(scope=scope, metric=metric, reference=reference, candidate=ARM)
                pairs.extend(pair.to_dict("records"))
                valid = pair.dropna(subset=["improvement"]).sort_values("cell_uid")
                protocols = valid.groupby("protocol_uid", sort=True)[["reference_error", "candidate_error", "improvement"]].mean()
                for estimator, sample in (("cell_mean", valid), ("protocol_macro", protocols)):
                    estimate = historical.paired_mean_bootstrap(sample.improvement.to_numpy(float))
                    first, last = sample.reference_error.mean(), sample.candidate_error.mean()
                    rows.append({"scope": scope, "metric": metric, "candidate": ARM, "reference": reference,
                                 "estimator": estimator, "reference_metric": first, "candidate_metric": last,
                                 **estimate,
                                 "relative_improvement_percent": (first - last) / first * 100 if first > 0 else None,
                                 "paired_cells": len(valid), "protocol_groups": len(protocols),
                                 "improved_cells": int(valid.improvement.gt(0).sum()),
                                 "worsened_cells": int(valid.improvement.lt(0).sum()),
                                 "tied_cells": int(valid.improvement.eq(0).sum()),
                                 "bootstrap_seed": 42, "bootstrap_replicates": 2000,
                                 "comparison_status": "historical_same_data_exposed_internal_single_seed"})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def eol_summary(eol, cells):
    truth = cells[["protocol_uid", "eol80_event_observed", "eol80_first_observed_crossing_cycle", "last_observed_valid_cycle"]].copy()
    truth["outcome_cycle"] = np.where(truth.eol80_event_observed, truth.eol80_first_observed_crossing_cycle, truth.last_observed_valid_cycle)
    merged = eol.merge(truth, left_on="cell_uid", right_index=True, validate="many_to_one")
    rows, contrasts = [], []
    for scope, frame in merged.groupby("scope", sort=True):
        risk = frame.loc[frame.outcome_cycle > 100]
        wide = risk.pivot(index="cell_uid", columns="arm", values="predicted_trend_eol80_cycle").sort_index()
        outcome = truth.loc[wide.index]
        times = outcome.outcome_cycle.to_numpy(float)
        events = outcome.eol80_event_observed.to_numpy(bool)
        predictions = {arm: wide[arm].to_numpy(float) for arm in (ARM,) + REFERENCES}
        scores = {}
        for arm, predicted in predictions.items():
            result = historical.harrell_concordance(times, events, predicted)
            scores[arm] = result["harrell_c"]
            arm_rows = frame.loc[frame.arm == arm]
            rows.append({"scope": scope, "arm": arm, **result,
                         "executed_test_cells": len(arm_rows), "at_risk100_cells": len(outcome),
                         "observed_events": int(events.sum()), "right_censored": int((~events).sum()),
                         "predicted_no_crossing_through5000": int(arm_rows.no_crossing_through_registered_horizon.sum())})
        bootstrap = historical.eol_bootstrap(times, events, predictions, outcome.protocol_uid.to_numpy(str))
        for unit, sampled in bootstrap.items():
            for reference in REFERENCES:
                delta = sampled[ARM] - sampled[reference]
                valid = delta[np.isfinite(delta)]
                low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
                contrasts.append({"scope": scope, "candidate": ARM, "reference": reference,
                                  "sampling_unit": unit, "reference_metric": scores[reference], "candidate_metric": scores[ARM],
                                  "improvement": scores[ARM] - scores[reference], "ci95_low": low, "ci95_high": high,
                                  "bootstrap_supported": len(valid), "bootstrap_requested": 2000,
                                  "bootstrap_seed": 42, "paired_cells": len(outcome)})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def historical_results(contract, cells, future, splits):
    results, identities = [], []
    for reference in contract['historical_references']:
        root = REPOSITORY / reference['root']
        registration = read_json(root / 'RUN_REGISTRATION.json')
        if registration['data_freeze_sha256'] != contract['data_freeze_sha256']:
            raise RuntimeError('Historical data freeze differs')
        for split in SPLITS:
            path = root / split / reference['arm']
            complete = read_json(path / 'COMPLETE.json')
            if complete['status'] != 'EVALUATED':
                raise RuntimeError('Historical fit is incomplete')
            for item in complete['files']:
                if sha256(path / item['path']) != item['sha256']:
                    raise RuntimeError('Historical result changed')
            unit = dict(arm=reference['arm'], split=split, root=path,
                        contract=read_json(path / 'UNIT_CONTRACT.json'),
                        selection=read_json(path / 'SELECTION_FROZEN.json'))
            if unit['selection']['checkpoint_sha256'] != sha256(path / 'SELECTED.pt'):
                raise RuntimeError('Historical checkpoint changed')
            results.append(historical.load_unit_results(unit, cells, future, splits, registration))
            identities.append(dict(arm=reference['arm'], split=split, root=path.relative_to(REPOSITORY).as_posix(),
                                   complete_sha256=sha256(path / 'COMPLETE.json')))
    return results, identities


def diagnostics(data_root, units, contract, device):
    training.configure_execution()
    data = training.load_frozen_data(data_root)
    inputs, ids, eligible, cells, _, _, observed, _, _, splits, _ = data
    lookup = {uid: i for i, uid in enumerate(ids)}
    queries = np.arange(100, 5001, dtype=np.float32)
    query_rows, soft_rows, monotonic_rows, mechanisms = [], [], [], []
    for unit in units:
        roles = splits['splits'][unit['split']]
        train = np.array([lookup[uid] for uid in roles['train']])
        test = np.array([lookup[uid] for uid in roles['test']])
        scaled, _ = training.fold_scaler(inputs, train)
        model = train_m5.build_model(inputs, contract['training']).to(device).eval()
        model.load_state_dict(torch.load(unit['root']/'SELECTED.pt', map_location=device, weights_only=False)['model'])
        scope = 'main_oof' if unit['split'].startswith('cell_fold_') else 'unseen_protocol_sensitivity'
        mechanisms.append(pd.read_csv(unit['root']/'test_mechanism_summary.csv').assign(split=unit['split'], arm=ARM))
        prediction = pd.read_parquet(unit['root']/'test_predictions.parquet')
        with torch.no_grad():
            for start in range(0, len(test), 8):
                part = test[start:start+8]
                dense = model(**training.to_batch(scaled, part, device), query_cycles=torch.as_tensor(queries, device=device), return_diagnostics=True)
                values = {key:dense[key].cpu().numpy() for key in ('raw_trend','observation_residual','rate_per_1000_cycles')}
                flow = dense['diagnostics']['flow']
                for local, number in enumerate(part):
                    uid = ids[number]
                    support = np.zeros(len(queries), bool)
                    support[1:1+observed.shape[1]] = observed[number]
                    current = query_region_rows(uid, scope, unit['split'], queries, support,
                                                int(cells.loc[uid,'last_observed_valid_cycle']),
                                                {key:value[local] for key,value in values.items()})
                    for row in current:
                        row['arm'] = ARM
                    query_rows.extend(current)
                    soft_rows.append(dict(scope=scope, split=unit['split'], arm=ARM, cell_uid=uid,
                                          **{key:float(flow[key][local].cpu()) for key in
                                             ('soft_transition_center_cycle','soft_transition_width_cycles','soft_transition_signed_amplitude')},
                                          relative_intensity_min_per1000=float(flow['relative_hazard_per_1000'][local].min().cpu()),
                                          relative_intensity_max_per1000=float(flow['relative_hazard_per_1000'][local].max().cpu()),
                                          certified_physical_knee=False))
                    frame = prediction[prediction.cell_uid==uid].sort_values('original_cycle_id')
                    pos = frame.original_cycle_id.to_numpy(int)-100
                    record = dict(scope=scope,split=unit['split'],arm=ARM,cell_uid=uid,
                                  observed_queries=len(frame),observed_adjacent_pairs=max(len(frame)-1,0))
                    for name, array in (('raw_trend',values['raw_trend'][local,pos]),
                                        ('trend',frame.trend.to_numpy()),('prediction',frame.prediction.to_numpy())):
                        delta = np.diff(array)
                        record[name+'_rise_count'] = int((delta>0).sum())
                        record[name+'_rise_over1e6_count'] = int((delta>1e-6).sum())
                        record[name+'_largest_positive_step'] = float(delta.max(initial=0))
                    monotonic_rows.append(record)
        del model, scaled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    query = pd.DataFrame(query_rows)
    return query, summarize_query_regions(query), pd.DataFrame(soft_rows), pd.DataFrame(monotonic_rows), pd.concat(mechanisms,ignore_index=True)


def analyze(data_root, output, device):
    contract, units = completed_units(output, data_root)
    method = contract['methodology_contract']
    if sha256(REPOSITORY/method['path']) != method['sha256']:
        raise RuntimeError('Registered M5 methodology changed')
    reproduction = reproduction_comparison(output, contract)
    cells, future, splits = historical.frozen_tables(data_root)
    current = [historical.load_unit_results(unit,cells,future,splits,contract) for unit in units]
    references, bound = historical_results(contract,cells,future,splits)
    results = current+references
    metrics = pd.DataFrame([row for result in results for row in result['metrics']])
    summary, protocols, folds = historical.metric_summaries(metrics)
    paired, pairs = fixed_comparisons(metrics)
    eol = pd.concat([item['eol'] for item in results],ignore_index=True)
    eol_stats, eol_pairs = eol_summary(eol,cells)
    query, query_stats, soft, monotonic, mechanism = diagnostics(data_root,units,contract,device)
    counterexample = metrics[(metrics.cell_uid=='MATR_b1c2') & (metrics.scope=='unseen_protocol_sensitivity')]
    tables = {'cell_metrics.csv':metrics,'metric_summary.csv':summary,'protocol_metrics.csv':protocols,
              'fold_metrics.csv':folds,'paired_comparisons.csv':paired,'paired_cells.csv':pairs,
              'eol_predictions.csv':eol,'eol_summary.csv':eol_stats,'eol_paired_comparisons.csv':eol_pairs,
              'training_units.csv':pd.DataFrame([item['training'] for item in results]),
              'query_range_cells.csv':query,'query_range_summary.csv':query_stats,
              'soft_transition_parameters.csv':soft,'observed_monotonicity.csv':monotonic,
              'mechanism_summary.csv':mechanism,'MATR_b1c2_counterexample.csv':counterexample}
    destination = output/'analysis'
    destination.mkdir(exist_ok=True)
    for name, table in tables.items():
        temporary=destination/(name+'.pending')
        table.to_csv(temporary,index=False)
        os.replace(temporary,destination/name)
    write_json(destination/'TRAINING_REPRODUCTION.json',reproduction)
    write_json(destination/'HISTORICAL_REFERENCES.json',{'units':bound})
    names=list(tables)+['TRAINING_REPRODUCTION.json','HISTORICAL_REFERENCES.json']
    write_json(destination/'ANALYSIS_COMPLETE.json',{
        'status':'ANALYZED' if reproduction['status']=='PASS' else 'ANALYZED_WITH_REPRODUCTION_FAILURE',
        'source_sha256':sha256(__file__),'model_contract_sha256':sha256(output/'MODEL_CONTRACT.json'),
        'formal_fits':6,'formal_optimizer_updates':12450,'repeat_optimizer_updates':2100,
        'training_seeds':[42],'historical_reference_fits_retrained':0,
        'soft_transition_is_certified_knee':False,
        'files':[{'path':name,'sha256':sha256(destination/name)} for name in names]})
    print(summary[summary.metric=='trajectory_mae'].to_string(index=False),flush=True)
    print(json.dumps({'status':'ANALYZED','full_training_reproduction':reproduction['status']}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    args=parser.parse_args()
    analyze(args.data.resolve(),args.output.resolve(),args.device)


if __name__=='__main__':
    main()
