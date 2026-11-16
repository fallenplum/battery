"""Fixed, cell-weighted N012 analyses after both complete training matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import load_bundle, file_hash
from .model import ENCODERS, DECODERS
from .run import ROOT, write_json, verify_files, file_records
from scripts.unified_matr.analyze_results import paired_mean_bootstrap

WINDOWS = (('full',101,np.inf),('101_200',101,200),('201_500',201,500),
           ('501_1000',501,1000),('1001_plus',1001,np.inf))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def completed_units(output, domain):
    root=output/domain
    matrix=read_json(root/'MATRIX_COMPLETE.json')
    registration=read_json(root/'RUN_REGISTRATION.json')
    if matrix['status']!='COMPLETE' or matrix['units']!=45 or matrix['registration_sha256']!=file_hash(root/'RUN_REGISTRATION.json'):
        raise RuntimeError('The complete45-unit domain matrix is required')
    for row in registration['sources']:
        if file_hash(ROOT/row['path'])!=row['sha256']:
            raise RuntimeError('Registered source changed')
    verify_files(root,matrix)
    units={}
    for path in sorted(root.glob('cell_fold_*/*/COMPLETE.json')):
        unit=read_json(path)
        if unit['status']!='EVALUATED' or unit['unit_contract']['study_sha256']!=matrix['registration_sha256']:
            raise RuntimeError('Completed unit identity differs')
        verify_files(path.parent,unit)
        training=read_json(path.parent/'TRAINING_COMPLETE.json')
        verify_files(path.parent,training)
        if file_hash(path.parent/'TRAINING_COMPLETE.json')!=unit['training_complete_sha256']:
            raise RuntimeError('Training marker changed')
        selection=root/unit['selection_file'].split('/',1)[1]
        if file_hash(selection)!=unit['selection_sha256']:
            raise RuntimeError('Selection changed after test evaluation')
        key=(unit['fold'],unit['encoder'],unit['decoder'])
        units[key]=dict(root=path.parent,record=unit)
    if len(units)!=45:
        raise RuntimeError('Domain matrix unique unit count differs')
    return registration,units


def model_bindings(root,units):
    n1=read_json(root/'N1_SELECTION.json')
    n2=read_json(root/'N2_SELECTION.json')
    bindings=[]
    for fold in n1['selection_folds']:
        encoder=n1['encoder_per_outer_fold'][fold]
        for name in ENCODERS:
            bindings.append(dict(phase='N1',method=name,fold=fold,encoder=name,decoder='query_mlp'))
        bindings.append(dict(phase='N1',method='nested',fold=fold,encoder=encoder,decoder='query_mlp'))
        for decoder in DECODERS:
            bindings.append(dict(phase='N2',method=decoder,fold=fold,encoder=encoder,decoder=decoder))
        bindings.append(dict(phase='N2',method='nested',fold=fold,encoder=encoder,decoder=n2['selected_decoder_per_fold'][fold]))
    for row in bindings:
        row['unit_root']=units[(row['fold'],row['encoder'],row['decoder'])]['root']
    return bindings


def cell_rows(bundle, ids, query, prediction, target, observed, phase, method, fold, encoder, decoder):
    records=[]
    for i,uid in enumerate(ids):
        for window,lo,hi in WINDOWS:
            use=observed[i]&(query>=lo)&(query<=hi)
            error=prediction[i,use].astype(float)-target[i,use].astype(float)
            values=prediction[i,use].astype(float)
            records.append(dict(domain=bundle.domain,phase=phase,method=method,model=phase+':'+method,
                                fold=fold,encoder=encoder,decoder=decoder,cell_uid=uid,
                                protocol_uid=str(bundle.cell_metadata.loc[uid,'protocol_uid']),window=window,
                                n_obs=int(use.sum()),mae=float(np.abs(error).mean()) if len(error) else np.nan,
                                rmse=float(np.sqrt(np.square(error).mean())) if len(error) else np.nan,
                                bias=float(error.mean()) if len(error) else np.nan,
                                maximum_absolute_error=float(np.abs(error).max()) if len(error) else np.nan,
                                negative_predictions=int((values<0).sum()),predictions_above1p2=int((values>1.2).sum()),
                                observed_rises_over1e6=int((np.diff(values)>1e-6).sum())))
    return records


def load_role_metrics(bundle, bindings):
    records=[]
    lookup={uid:i for i,uid in enumerate(bundle.ids)}
    for role in bindings:
        with np.load(role['unit_root']/'test_predictions.npz',allow_pickle=False) as saved:
            ids=saved['ids'].astype(str).tolist()
            expected=bundle.folds[role['fold']]['test']
            indices=[lookup[uid] for uid in ids]
            if ids!=expected or not np.array_equal(saved['query'],bundle.query):
                raise RuntimeError('OOF input identity or query differs')
            if not np.array_equal(saved['observed'],bundle.observed[indices]) or not np.array_equal(saved['target'],bundle.targets[indices]):
                raise RuntimeError('Prediction targets differ from frozen source')
            records.extend(cell_rows(bundle,ids,saved['query'],saved['prediction'],saved['target'],saved['observed'],
                                     **{key:role[key] for key in ('phase','method','fold','encoder','decoder')}))
    return records


def historical_m1(bundle):
    root=ROOT/'results/unified_matr_seed42/runs'
    registration=read_json(root/'RUN_REGISTRATION.json')
    if registration['data_freeze_sha256']!=file_hash(ROOT/'data/unified_matr_seed42/DATA_FREEZE.json'):
        raise RuntimeError('Historical target source differs')
    lookup={uid:i for i,uid in enumerate(bundle.ids)}
    records=[]
    identities=[]
    for fold in (name for name in bundle.folds if name.startswith('cell_fold_')):
        path=root/fold/'M1'
        receipt=read_json(path/'COMPLETE.json')
        verify_files(path,receipt)
        table=pd.read_parquet(path/'test_predictions.parquet')
        ids=bundle.folds[fold]['test']
        indices=[lookup[uid] for uid in ids]
        prediction=np.zeros_like(bundle.targets[indices])
        seen=np.zeros_like(prediction,dtype=bool)
        local={uid:i for i,uid in enumerate(ids)}
        for row in table.itertuples(index=False):
            i,j=local[row.cell_uid],int(row.original_cycle_id)-101
            if seen[i,j] or np.float32(row.target)!=bundle.targets[indices[i],j]:
                raise RuntimeError('Historical trajectory target mismatch')
            seen[i,j]=True
            prediction[i,j]=row.prediction
        if not np.array_equal(seen,bundle.observed[indices]):
            raise RuntimeError('Historical observed support mismatch')
        records.extend(cell_rows(bundle,ids,bundle.query,prediction,bundle.targets[indices],seen,
                                 'HISTORICAL','M1',fold,'historical_M1','historical_M1'))
        identities.append(dict(fold=fold,complete_sha256=file_hash(path/'COMPLETE.json'),
                               predictions_sha256=file_hash(path/'test_predictions.parquet')))
    return records,identities


def summaries(frame):
    rows=[]
    for keys,group in frame.groupby(['domain','phase','method','model','window'],sort=True):
        valid=group[group.n_obs>0]
        protocols=valid.groupby('protocol_uid')[['mae','rmse']].mean()
        rows.append(dict(zip(['domain','phase','method','model','window'],keys),
                         execution_cells=len(group),supported_cells=len(valid),observations=int(group.n_obs.sum()),
                         supported_protocols=len(protocols),cell_mean_mae=valid.mae.mean(),cell_mean_rmse=valid.rmse.mean(),
                         protocol_macro_mae=protocols.mae.mean(),protocol_macro_rmse=protocols.rmse.mean(),
                         mean_signed_bias=valid.bias.mean()))
    return pd.DataFrame(rows)


def planned_contrasts(domain):
    pairs=[('N1:'+name,'N1:resmlp') for name in ENCODERS[1:]]
    pairs.extend(('N2:'+name,'N2:query_mlp') for name in DECODERS[1:])
    pairs.extend([('N2:physics_basis','N2:free_rate'),('N2:physics_basis','N2:monotone_spline')])
    if domain=='MATR':
        models=['N1:'+name for name in ENCODERS]+['N1:nested']+['N2:'+name for name in DECODERS]+['N2:nested']
        pairs.extend((model,'HISTORICAL:M1') for model in models)
    return pairs


def paired_statistics(frame):
    comparisons=[]
    cells=[]
    for domain,group in frame[frame.window=='full'].groupby('domain',sort=True):
        for metric in ('mae','rmse'):
            wide=group.pivot(index=['cell_uid','protocol_uid'],columns='model',values=metric)
            for candidate,reference in planned_contrasts(domain):
                pair=wide[[candidate,reference]].reset_index().rename(columns={candidate:'candidate_error',reference:'reference_error'})
                pair['improvement']=pair.reference_error-pair.candidate_error
                pair=pair.assign(domain=domain,metric=metric,candidate=candidate,reference=reference)
                cells.extend(pair.to_dict('records'))
                for unit,values in (('cell',pair),('protocol_group',pair.groupby('protocol_uid',sort=True)[['candidate_error','reference_error','improvement']].mean())):
                    comparisons.append(dict(domain=domain,metric=metric,candidate=candidate,reference=reference,
                                            sampling_unit=unit,paired_cells=len(pair),protocol_groups=pair.protocol_uid.nunique(),
                                            candidate_metric=values.candidate_error.mean(),reference_metric=values.reference_error.mean(),
                                            better_cells=int((pair.improvement>0).sum()),worse_cells=int((pair.improvement<0).sum()),
                                            **paired_mean_bootstrap(values.improvement.to_numpy()),bootstrap_seed=42,replicates=2000,
                                            scope='exploratory_conditional_on_fitted_predictions'))
    return pd.DataFrame(comparisons),pd.DataFrame(cells)


def analyze(output):
    all_rows=[]
    training=[]
    bindings_all=[]
    historical=[]
    for domain in ('MATR','HUST'):
        bundle=load_bundle(ROOT,domain)
        registration,units=completed_units(output,domain)
        if registration['tensor_fingerprint']!=read_json(output/domain/'N0_READOUT.json')['data_fingerprint']:
            raise RuntimeError('N0 source fingerprint differs')
        bindings=model_bindings(output/domain,units)
        all_rows.extend(load_role_metrics(bundle,bindings))
        bindings_all.extend(dict(row,domain=domain,unit_root=row['unit_root'].relative_to(ROOT).as_posix()) for row in bindings)
        for unit in units.values():
            record=unit['record']
            training.append(dict(domain=domain,fold=record['fold'],encoder=record['encoder'],decoder=record['decoder'],
                                 epochs=record['epochs'],updates=record['updates'],parameters=record['parameters'],
                                 validation_mae=record['validation_mae'],best_epoch=record['best_epoch'],
                                 elapsed_seconds=record['elapsed_seconds'],complete_sha256=file_hash(unit['root']/'COMPLETE.json')))
        if domain=='MATR':
            old,identities=historical_m1(bundle)
            all_rows.extend(old)
            historical.extend(identities)
    frame=pd.DataFrame(all_rows)
    summary=summaries(frame)
    protocols=frame[frame.n_obs>0].groupby(['domain','phase','method','window','protocol_uid']).agg(
        mae=('mae','mean'),rmse=('rmse','mean'),cells=('cell_uid','nunique'),observations=('n_obs','sum')).reset_index()
    contrasts,pairs=paired_statistics(frame)
    worst=frame[frame.n_obs>0].sort_values('mae',ascending=False).groupby(['domain','model','window'],sort=True).head(3)
    long=[]
    for domain in ('MATR','HUST'):
        for window in ('501_1000','1001_plus'):
            subset=frame[(frame.domain==domain)&(frame.phase=='N2')&(frame.window==window)]
            wide=subset.pivot(index=['cell_uid','protocol_uid'],columns='method',values='mae')
            for reference in ('free_rate','monotone_spline'):
                paired=wide[['physics_basis',reference]].rename(columns={'physics_basis':'candidate_error',reference:'reference_error'}).dropna().reset_index()
                paired['improvement']=paired.reference_error-paired.candidate_error
                long.extend(paired.assign(domain=domain,window=window,reference=reference,candidate='physics_basis').to_dict('records'))
    tables={'cell_metrics.csv':frame,'summary_metrics.csv':summary,'protocol_metrics.csv':protocols,
            'paired_comparisons.csv':contrasts,'paired_cells.csv':pairs,'training_units.csv':pd.DataFrame(training),
            'worst_cases.csv':worst,'long_horizon_paired_cells.csv':pd.DataFrame(long)}
    destination=output/'analysis'
    destination.mkdir(exist_ok=True)
    for name,table in tables.items():
        table.to_csv(destination/name,index=False)
    write_json(destination/'MODEL_ROLE_BINDINGS.json',dict(roles=bindings_all))
    write_json(destination/'HISTORICAL_M1_BINDINGS.json',dict(units=historical,scope='same_labels_and_folds_different_physics_inputs_and_auxiliary_loss'))
    names=list(tables)+['MODEL_ROLE_BINDINGS.json','HISTORICAL_M1_BINDINGS.json']
    write_json(destination/'ANALYSIS_COMPLETE.json',dict(status='ANALYZED',source_sha256=file_hash(__file__),
               evaluation_plan_sha256=file_hash(ROOT/'research/N012_EVALUATION_PLAN_ZH.md'),
               formal_unique_fits=90,formal_optimizer_updates=sum(row['updates'] for row in training),
               files=file_records(destination,names)))
    print(summary[summary.window=='full'].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results/N012_independent')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        analyze(args.out.resolve())


if __name__=='__main__':
    main()
