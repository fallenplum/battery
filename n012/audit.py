"""N0 qualification, train-only moments and explicit observation coverage."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import load_bundle, TrainOnlyScaler, role_indices, file_hash


def audit_domain(repo_root, domain, output, bundle=None):
    from .run import bind_json, bundle_fingerprint, write_json
    repo_root, output = Path(repo_root).resolve(), Path(output).resolve()
    destination = output/domain
    destination.mkdir(parents=True, exist_ok=True)
    if domain == 'Stanford':
        supplied = Path('D:/develop/patchtst/BatteryLife/dataset/Stanford_2')
        files = sorted(supplied.glob('*.pkl'))
        report = dict(status='N0_BLOCKED', domain=domain, supplied_pkl_inventory=len(files),
                      reason='Available Nova_Regular records use BEEP-interpolated/reindexed supplier coordinates. No certified original-cycle1..100 tensor plus future SOH contract was located.',
                      checked_repository_paths=['data','report/STANFORD_EXTENSION_QUALIFICATION_ZH.md',
                                                'report/STANFORD_PREDICTION_FAILURE_DIAGNOSTIC_ZH.md'],
                      checked_local_paths=[str(supplied),str(repo_root.parent/'private/stanford_source_metadata'),
                                           str(repo_root.parent/'private/raw_sources')],
                      distinct_DATA_directory='Discovery Learning123-cell dataset, unrelated to Stanford Nova_Regular181',
                      training_fits=0,
                      evidence=[{'path':path,'sha256':file_hash(repo_root/path)} for path in
                                ('report/STANFORD_EXTENSION_QUALIFICATION_ZH.md','report/STANFORD_PREDICTION_FAILURE_DIAGNOSTIC_ZH.md')])
        bind_json(destination/'N0_READOUT.json',report)
        print(json.dumps(dict(phase='N0',domain=domain,status='N0_BLOCKED',supplied_cells=len(files))),flush=True)
        return report
    bundle = bundle or load_bundle(repo_root,domain)
    expected = {'MATR':169,'HUST':77}[domain]
    if len(bundle.ids)!=expected or not bundle.eligible.all():
        raise ValueError('Original qualified cohort changed')
    folds = {key:value for key,value in bundle.folds.items() if key.startswith('cell_fold_')}
    if len(folds)!=5:
        raise ValueError('Exactly five original outer folds required')
    test_ids = [uid for roles in folds.values() for uid in roles['test']]
    if sorted(test_ids)!=sorted(bundle.ids):
        raise ValueError('Each physical cell must appear in outer test exactly once')
    cell_rows=[]
    for i,uid in enumerate(bundle.ids):
        valid = bundle.mask[i].any(axis=-1)
        cycles = bundle.cycle[i,valid]
        future = bundle.query[bundle.observed[i]]
        cell_rows.append(dict(cell_uid=uid,input_eligible=bool(bundle.eligible[i]),
                              valid_prefix_curves=int(valid.sum()),valid_prefix_points=int(bundle.mask[i].sum()),
                              prefix_first_cycle=float(cycles.min()),prefix_last_cycle=float(cycles.max()),
                              future_observations=len(future),first_future_cycle=float(future.min()),last_future_cycle=float(future.max()),
                              internal_missing_future_cycles=int(future.max()-100-len(future)),
                              unavailable_through_domain_max=int(len(bundle.query)-len(future)),
                              policy_supported_coordinates=int(bundle.policy_mask[i].sum()),anchor_soh100=float(bundle.anchor[i]),
                              protocol_uid=str(bundle.cell_metadata.loc[uid,'protocol_uid']),
                              eol80_event_observed=bool(bundle.cell_metadata.loc[uid,'eol80_event_observed'])))
    cells=pd.DataFrame(cell_rows)
    scalers={}
    for fold in folds:
        rows=role_indices(bundle,fold)
        scaler=TrainOnlyScaler().fit(bundle,rows['train'])
        # A held-out target change cannot enter the normalizer's inputs.
        if not np.isfinite(np.r_[scaler.x_mean,scaler.x_std,scaler.p_mean,scaler.p_std]).all():
            raise ValueError('Nonfinite training-only moments')
        scalers[fold]=dict(training_ids=[bundle.ids[i] for i in rows['train']],statistics=scaler.to_dict())
    provenance=dict(bundle.provenance)
    provenance['files']=[dict(item,path=str(Path(item['path']).resolve().relative_to(repo_root)).replace('\\','/'))
                         for item in provenance['files']]
    report=dict(status='N0_PASS',domain=domain,qualified_cells=len(bundle.ids),
                physical_cycle_semantics='published original instrument cycle keys',prefix_cutoff=100,
                input_channels=['V','I','Q','Q_minus_prefix_reference_Q'],tensor_shape=list(bundle.x.shape),
                query_min=float(bundle.query.min()),query_max=float(bundle.query.max()),
                observed_future_points=int(bundle.observed.sum()),
                unavailable_future_grid_positions=int((~bundle.observed).sum()),
                internal_missing_future_cycles=int(cells.internal_missing_future_cycles.sum()),
                valid_prefix_curves=int(cells.valid_prefix_curves.sum()),
                unique_protocols=int(cells.protocol_uid.nunique()),
                observed_EOL80_events=int(cells.eol80_event_observed.sum()),
                EOL80_right_censored=int((~cells.eol80_event_observed).sum()),
                nominal_capacity_Ah=1.1,policy_dimensions=bundle.policy.shape[1],
                data_fingerprint=bundle_fingerprint(bundle),source=provenance,
                five_folds_cell_disjoint=True,OOF_each_cell_once=True,train_only_standardization=True,
                targets_used_as_predictor_inputs=False,seed=42)
    bind_json(destination/'N0_READOUT.json',report)
    bind_json(destination/'N0_FOLDS.json',dict(role_unit='physical-cell UID',roles=folds))
    bind_json(destination/'N0_TRAIN_SCALERS.json',scalers)
    cells.to_csv(destination/'N0_CELL_QUALIFICATION.csv',index=False)
    print(json.dumps(dict(phase='N0',domain=domain,status='PASS',cells=len(cells),observations=report['observed_future_points'])),flush=True)
    return report
