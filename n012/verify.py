"""Independent selected-weight replay, numerical head oracles and statistics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from threadpoolctl import threadpool_limits

from . import run
from .data import load_bundle, TrainOnlyScaler, role_indices, file_hash
from .model import TrajectoryNet, ENCODERS, DECODERS
from .engineering import numpy_mlp, free_rate_oracle
from scripts.verify_unified_seed42_results import Checks, compare_table, bootstrap_average


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(checks,name,a,b,atol=1e-10,rtol=1e-8):
    a,b=np.asarray(a),np.asarray(b)
    checks.require(name,a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True))


def numpy_head(head,name,state,anchor,query):
    u=(query-100)/1000
    if name=='free_rate':
        return free_rate_oracle(head,state,anchor,query)
    if name=='query_mlp':
        context=np.broadcast_to(state[:,None],(*u.shape,state.shape[-1]))
        values=numpy_mlp(head.net,np.concatenate((context,u[...,None]),axis=-1)).squeeze(-1)
        zero=numpy_mlp(head.net,np.column_stack((state,np.zeros(len(state))))).squeeze(-1)
        return anchor[:,None]+.2*(values-zero[:,None])
    if name=='legacy_flow':
        generated=numpy_mlp(head.gen,state)
        rate=np.logaddexp(0,generated[:,:16])+1e-4
        mean=np.tanh(generated[:,16:])
        initial=np.tanh(state@head.state.weight.detach().cpu().numpy().T+head.state.bias.detach().cpu().numpy())
        decay=np.exp(-rate[:,None]*u[...,None])
        latent=decay*initial[:,None]+(1-decay)*mean[:,None]
        return anchor[:,None]+.2*(numpy_mlp(head.decode,latent).squeeze(-1)-numpy_mlp(head.decode,initial).squeeze(-1)[:,None])
    if name=='monotone_spline':
        rates=np.logaddexp(0,numpy_mlp(head.net,state))*.15
        knots=np.linspace(0,5,11)
        durations=np.minimum(np.maximum(u[...,None]-knots[:-1],0),np.diff(knots))
        cumulative=(durations*rates[:,None]).sum(axis=-1)+np.maximum(u-5,0)*rates[:,-1,None]
        return anchor[:,None]*np.exp(-cumulative)
    parameters=numpy_mlp(head.net,state)
    a=np.logaddexp(0,parameters[:,0:1])*.05
    b=np.logaddexp(0,parameters[:,1:2])*.05
    c=np.logaddexp(0,parameters[:,2:3])*.06
    tau=4*expit(parameters[:,3:4])
    time_c=np.logaddexp(0,float(head.log_c.detach().cpu()))+.05
    smooth=np.logaddexp(0,float(head.log_w.detach().cpu()))+.03
    cumulative=a*np.log1p(u/time_c)+b*u+c*smooth*(np.logaddexp(0,(u-tau)/smooth)-np.logaddexp(0,-tau/smooth))
    return anchor[:,None]*np.exp(-cumulative)


def independent_scaler(bundle,train):
    supported=bundle.x[train][bundle.mask[train]].astype(float)
    mean=supported.mean(axis=0)
    std=np.maximum(np.sqrt(np.mean(np.square(supported-mean),axis=0)),1e-5)
    policy_mean=[]
    policy_std=[]
    for j in range(bundle.policy.shape[1]):
        values=bundle.policy[train,j][bundle.policy_mask[train,j]].astype(float)
        policy_mean.append(values.mean() if len(values) else 0.)
        policy_std.append(max(float(values.std()),1e-5) if len(values) else 1e-5)
    return dict(x_mean=mean,x_std=std,p_mean=policy_mean,p_std=policy_std)


def independent_cell_rows(bundle,ids,query,prediction,target,observed,labels,fold,encoder,decoder):
    records=[]
    for phase,method in labels:
        for i,uid in enumerate(ids):
            for window,lower,upper in (('full',101,np.inf),('101_200',101,200),('201_500',201,500),
                                       ('501_1000',501,1000),('1001_plus',1001,np.inf)):
                slots=np.flatnonzero(observed[i]&(query>=lower)&(query<=upper))
                values=prediction[i,slots].astype(float)
                delta=values-target[i,slots].astype(float)
                count=len(slots)
                records.append(dict(domain=bundle.domain,phase=phase,method=method,model=phase+':'+method,fold=fold,
                                    encoder=encoder,decoder=decoder,cell_uid=uid,protocol_uid=str(bundle.cell_metadata.loc[uid,'protocol_uid']),
                                    window=window,n_obs=count,mae=float(np.sum(np.abs(delta))/count) if count else np.nan,
                                    rmse=float(np.sqrt(np.dot(delta,delta)/count)) if count else np.nan,
                                    bias=float(delta.sum()/count) if count else np.nan,
                                    maximum_absolute_error=float(np.max(np.abs(delta))) if count else np.nan,
                                    negative_predictions=int(np.count_nonzero(values<0)),predictions_above1p2=int(np.count_nonzero(values>1.2)),
                                    observed_rises_over1e6=int(np.count_nonzero(np.diff(values)>1e-6))))
    return records


def labels_for(encoder,decoder,fold,n1,n2):
    labels=[]
    if decoder=='query_mlp':
        labels.append(('N1',encoder))
        if encoder==n1['encoder_per_outer_fold'][fold]:
            labels.append(('N1','nested'))
    if encoder==n1['encoder_per_outer_fold'][fold]:
        labels.append(('N2',decoder))
        if decoder==n2['selected_decoder_per_fold'][fold]:
            labels.append(('N2','nested'))
    return labels


def replay_unit(checks,path,bundle,registration,n1,n2,device):
    complete=read(path/'COMPLETE.json')
    contract=read(path/'UNIT_CONTRACT.json')
    fold,encoder,decoder=contract['fold'],contract['encoder'],contract['decoder']
    name=bundle.domain+'/'+fold+'/'+encoder+'/'+decoder
    roles=role_indices(bundle,fold)
    ids={role:[bundle.ids[i] for i in rows] for role,rows in roles.items()}
    checks.require(name+'/registered_roles_and_settings',contract['roles']==ids and contract['seed']==42
                   and contract['epochs']==150 and contract['batch_size']==8 and contract['width']==64
                   and contract['learning_rate']==.0003 and contract['weight_decay']==.01
                   and contract['loss']=='cell_weighted_masked_future_MAE'
                   and contract['study_sha256']==file_hash(path.parents[1]/'RUN_REGISTRATION.json'))
    for marker in ('TRAINING_COMPLETE.json','COMPLETE.json'):
        record=read(path/marker)
        checks.require(name+'/'+marker+'_all_hashes',all(file_hash(path/item['path'])==item['sha256'] for item in record['files']))
    checks.require(name+'/selection_before_test',complete['selection_frozen_before_test']
                   and complete['selection_sha256']==file_hash(path.parents[2]/complete['selection_file'])
                   and complete['training_complete_sha256']==file_hash(path/'TRAINING_COMPLETE.json'))
    statistics=independent_scaler(bundle,roles['train'])
    saved_scaler=read(path/'SCALER_TRAIN_ONLY.json')
    checks.require(name+'/scaler_training_ids',saved_scaler['training_ids']==ids['train'])
    for key in statistics:
        close(checks,name+'/independent_'+key,saved_scaler['statistics'][key],statistics[key],atol=1e-9,rtol=1e-8)
    scaler=TrainOnlyScaler().fit(bundle,roles['train'])
    x,p=scaler.apply(bundle)
    history=read(path/'HISTORY.json')
    steps=math.ceil(len(roles['train'])/8)
    checks.require(name+'/all150_epochs_and_update_counts',len(history)==150
                   and [row['epoch'] for row in history]==list(range(1,151))
                   and [row['updates'] for row in history]==[steps*i for i in range(1,151)]
                   and np.isfinite([[row['train_mae'],row['val_mae']] for row in history]).all())
    best=np.minimum.accumulate([row['val_mae'] for row in history])
    chosen=[i==0 or history[i]['val_mae']<best[i-1] for i in range(len(history))]
    best_row=min(history,key=lambda row:row['val_mae'])
    checks.require(name+'/strict_first_min_selection',chosen==[row['selected'] for row in history]
                   and complete['best_epoch']==best_row['epoch'] and complete['validation_mae']==best_row['val_mae'])
    final=torch.load(path/'TRAIN_STATE.pt',map_location='cpu',weights_only=False)
    checkpoint=torch.load(path/'BEST.pt',map_location='cpu',weights_only=False)
    checks.require(name+'/final_optimizer_and_resume_state',final['next_epoch']==150 and final['updates']==steps*150
                   and final['history']==history and final['contract']==contract
                   and all(int(value['step'])==steps*150 for value in final['optimizer']['state'].values())
                   and all(group['lr']==.0003 and group['weight_decay']==.01 and group['betas']==(.9,.999)
                           and group['eps']==1e-8 for group in final['optimizer']['param_groups']))
    checks.require(name+'/selected_weights_equal_recoverable_best',checkpoint['contract']==contract
                   and checkpoint['best_epoch']==best_row['epoch'] and checkpoint['validation_mae']==best_row['val_mae']
                   and all(torch.equal(value,final['best_checkpoint']['model'][key]) for key,value in checkpoint['model'].items())
                   and all(torch.isfinite(value).all() for value in checkpoint['model'].values()))
    del final
    model=TrajectoryNet(encoder,decoder,policy_dim=bundle.policy.shape[1],width=64).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    checks.require(name+'/actual_parameter_count',sum(parameter.numel() for parameter in model.parameters())==complete['parameters'])
    replays={}
    with torch.no_grad():
        for role in ('validation','test'):
            output=[]
            for start in range(0,len(roles[role]),8):
                batch=run.to_tensors(bundle,x,p,roles[role][start:start+8],device)
                output.append(model(batch['x'],batch['point_mask'],batch['cycle_ids'],batch['policy'],batch['policy_mask'],batch['anchor'],batch['query']).cpu().numpy())
            replays[role]=np.concatenate(output)
    validation_error=[]
    for i,number in enumerate(roles['validation']):
        support=bundle.observed[number]
        validation_error.append(np.abs(replays['validation'][i,support].astype(float)-bundle.targets[number,support]).mean())
    close(checks,name+'/normal_validation_replay',np.mean(validation_error),complete['validation_mae'],atol=1e-12,rtol=1e-10)
    with np.load(path/'test_predictions.npz',allow_pickle=False) as saved:
        checks.require(name+'/test_ids_targets_query_mask',saved['ids'].astype(str).tolist()==ids['test']
                       and np.array_equal(saved['query'],bundle.query) and np.array_equal(saved['observed'],bundle.observed[roles['test']])
                       and np.array_equal(saved['target'],bundle.targets[roles['test']]))
        maximum=float(np.max(np.abs(saved['prediction']-replays['test'])))
        checks.require(name+'/normal_test_replay_exact_bits',saved['prediction'].dtype==replays['test'].dtype
                       and saved['prediction'].shape==replays['test'].shape and saved['prediction'].tobytes()==replays['test'].tobytes(),
                       {'max_absolute_error':maximum})
    records=independent_cell_rows(bundle,ids['test'],bundle.query,replays['test'],bundle.targets[roles['test']],bundle.observed[roles['test']],
                                  labels_for(encoder,decoder,fold,n1,n2),fold,encoder,decoder)
    diagnostic=[]
    full_query=np.arange(100,5001,dtype=np.float32)
    sparse=np.array([100.,101.,207.,500.,1000.,2238.,5000.],np.float32)
    with torch.no_grad():
        for start in range(0,len(roles['test']),8):
            part=roles['test'][start:start+8]
            batch=run.to_tensors(bundle,x,p,part,device)
            args=[batch[key] for key in ('x','point_mask','cycle_ids','policy','policy_mask','anchor')]
            dense=model(*args,torch.as_tensor(full_query,device=device)).cpu().numpy()
            few=model(*args,torch.as_tensor(sparse,device=device)).cpu().numpy()
            close(checks,name+'/query_consistency_batch'+str(start),few,dense[:,sparse.astype(int)-100],atol=2e-6,rtol=2e-6)
            close(checks,name+'/anchor_batch'+str(start),dense[:,0],bundle.anchor[part],atol=2e-6,rtol=2e-6)
            state=model.encoder(*args[:5]).cpu().numpy().astype(float)
            coordinates=np.broadcast_to(full_query.astype(float),(len(part),len(full_query)))
            expected=numpy_head(model.decoder,decoder,state,bundle.anchor[part].astype(float),coordinates)
            close(checks,name+'/independent_head_math_batch'+str(start),dense,expected,atol=3e-6,rtol=3e-5)
            for local,number in enumerate(part):
                native=np.zeros(len(full_query),bool)
                native[1:1+len(bundle.query)]=bundle.observed[number]
                last=float(bundle.query[bundle.observed[number]].max())
                for region,mask in (('observed_future',native),('beyond_cell_followup',full_query>last),('all100_5000',np.ones(len(full_query),bool))):
                    values=dense[local,mask]
                    diagnostic.append(dict(domain=bundle.domain,fold=fold,encoder=encoder,decoder=decoder,cell_uid=bundle.ids[number],
                                           region=region,query_count=len(values),negative_count=int((values<0).sum()),
                                           above1p2_count=int((values>1.2).sum()),nonfinite_count=int((~np.isfinite(values)).sum()),
                                           rise_over1e6_count=int((np.diff(values)>1e-6).sum()),minimum=float(values.min()),maximum=float(values.max())))
            if decoder in ('free_rate','monotone_spline','physics_basis'):
                checks.require(name+'/monotone_structure_batch'+str(start),bool((np.diff(dense,axis=1)<=1e-6).all()))
            if decoder in ('monotone_spline','physics_basis'):
                checks.require(name+'/nonnegative_structure_batch'+str(start),bool((dense>=0).all()))
    del model
    return records,diagnostic,dict(domain=bundle.domain,fold=fold,encoder=encoder,decoder=decoder,
                                   selected_epoch=best_row['epoch'],updates=steps*150,parameters=complete['parameters'],replay_error=maximum)


def independent_summary(frame):
    result=[]
    for (domain,phase,method,model,window),group in frame.groupby(['domain','phase','method','model','window'],sort=True):
        supported=group[group.n_obs>0]
        protocol_values=[(part.mae.mean(),part.rmse.mean()) for _,part in supported.groupby('protocol_uid',sort=True)]
        result.append(dict(domain=domain,phase=phase,method=method,model=model,window=window,
                           execution_cells=len(group),supported_cells=len(supported),observations=int(group.n_obs.sum()),
                           supported_protocols=len(protocol_values),cell_mean_mae=supported.mae.mean(),cell_mean_rmse=supported.rmse.mean(),
                           protocol_macro_mae=np.mean([value[0] for value in protocol_values]) if protocol_values else np.nan,
                           protocol_macro_rmse=np.mean([value[1] for value in protocol_values]) if protocol_values else np.nan,
                           mean_signed_bias=supported.bias.mean()))
    return pd.DataFrame(result)


def verify_selections(checks,root,registration):
    n1,n2=read(root/'N1_SELECTION.json'),read(root/'N2_SELECTION.json')
    checks.require(root.name+'/N2_binds_frozen_N1',n2['N1_selection_sha256']==file_hash(root/'N1_SELECTION.json')
                   and n1['study_sha256']==n2['study_sha256']==file_hash(root/'RUN_REGISTRATION.json'))
    for fold in registration['folds']:
        encoder_values=[]
        for encoder in ENCODERS:
            path=root/fold/(encoder+'__query_mlp')
            trained=read(path/'TRAINING_COMPLETE.json')
            entry=n1['candidates'][fold][encoder]
            checks.require(root.name+'/'+fold+'/'+encoder+'/N1_validation_evidence',
                           entry['training_complete_sha256']==file_hash(path/'TRAINING_COMPLETE.json')
                           and entry['validation_mae']==trained['validation_mae'] and entry['best_epoch']==trained['best_epoch'])
            encoder_values.append(trained['validation_mae'])
        encoder=ENCODERS[int(np.argmin(encoder_values))]
        checks.require(root.name+'/'+fold+'/N1_own_fold_first_min',n1['encoder_per_outer_fold'][fold]==encoder)
        decoder_values=[]
        for decoder in DECODERS:
            path=root/fold/(encoder+'__'+decoder)
            trained=read(path/'TRAINING_COMPLETE.json')
            entry=n2['candidates'][fold][decoder]
            checks.require(root.name+'/'+fold+'/'+decoder+'/N2_validation_evidence',
                           entry['training_complete_sha256']==file_hash(path/'TRAINING_COMPLETE.json')
                           and entry['validation_mae']==trained['validation_mae'] and entry['best_epoch']==trained['best_epoch'])
            decoder_values.append(trained['validation_mae'])
        checks.require(root.name+'/'+fold+'/N2_own_fold_first_min',
                       n2['selected_encoder_per_fold'][fold]==encoder
                       and n2['selected_decoder_per_fold'][fold]==DECODERS[int(np.argmin(decoder_values))])
    return n1,n2


def historical_metrics(checks,bundle):
    records=[]
    for fold,roles in bundle.folds.items():
        if not fold.startswith('cell_fold_'):
            continue
        path=run.ROOT/'results/unified_matr_seed42/runs'/fold/'M1'
        complete=read(path/'COMPLETE.json')
        checks.require('historical_M1/'+fold+'/all_artifact_hashes',all(file_hash(path/item['path'])==item['sha256'] for item in complete['files']))
        saved=pd.read_parquet(path/'test_predictions.parquet')
        ids=roles['test']
        index=[bundle.ids.index(uid) for uid in ids]
        pred=np.zeros_like(bundle.targets[index])
        seen=np.zeros_like(bundle.observed[index])
        identity={uid:i for i,uid in enumerate(ids)}
        target_valid=True
        for row in saved.itertuples(index=False):
            i,j=identity[row.cell_uid],int(row.original_cycle_id)-101
            target_valid=target_valid and not seen[i,j] and np.float32(row.target)==bundle.targets[index[i],j]
            seen[i,j]=True
            pred[i,j]=row.prediction
        checks.require('historical_M1/'+fold+'/native_labels_and_support',target_valid and np.array_equal(seen,bundle.observed[index]))
        records.extend(independent_cell_rows(bundle,ids,bundle.query,pred,bundle.targets[index],seen,
                                             [('HISTORICAL','M1')],fold,'historical_M1','historical_M1'))
    return records


def independent_pairs(frame):
    comparisons=[]
    cells=[]
    for domain,group in frame[frame.window=='full'].groupby('domain',sort=True):
        contrasts=[('N1:'+encoder,'N1:resmlp') for encoder in ENCODERS[1:]]
        contrasts += [('N2:'+decoder,'N2:query_mlp') for decoder in DECODERS[1:]]
        contrasts += [('N2:physics_basis','N2:free_rate'),('N2:physics_basis','N2:monotone_spline')]
        if domain=='MATR':
            candidates=['N1:'+encoder for encoder in ENCODERS]+['N1:nested']+['N2:'+decoder for decoder in DECODERS]+['N2:nested']
            contrasts += [(candidate,'HISTORICAL:M1') for candidate in candidates]
        for metric in ('mae','rmse'):
            wide=group.pivot(index=['cell_uid','protocol_uid'],columns='model',values=metric)
            for candidate,reference in contrasts:
                paired=wide[[reference,candidate]].reset_index().rename(columns={reference:'reference_error',candidate:'candidate_error'})
                paired['improvement']=paired.reference_error-paired.candidate_error
                paired=paired.assign(domain=domain,metric=metric,candidate=candidate,reference=reference)
                cells.extend(paired.to_dict('records'))
                for unit,values in (('cell',paired),('protocol_group',paired.groupby('protocol_uid',sort=True)[['candidate_error','reference_error','improvement']].mean())):
                    comparisons.append(dict(domain=domain,metric=metric,candidate=candidate,reference=reference,sampling_unit=unit,
                                            paired_cells=len(paired),protocol_groups=paired.protocol_uid.nunique(),
                                            candidate_metric=values.candidate_error.mean(),reference_metric=values.reference_error.mean(),
                                            better_cells=int((paired.improvement>0).sum()),worse_cells=int((paired.improvement<0).sum()),
                                            **bootstrap_average(values.improvement.to_numpy()),bootstrap_seed=42,replicates=2000,
                                            scope='exploratory_conditional_on_fitted_predictions'))
    return pd.DataFrame(comparisons),pd.DataFrame(cells)


def verify(output,device,checks):
    run.configure()
    checks.require('both_domain_matrices_complete',all((output/domain/'MATRIX_COMPLETE.json').is_file() for domain in ('MATR','HUST')))
    global_contract=read(output/'EXECUTION_REGISTRATION.json')
    checks.require('global_registered_source_and_seed',global_contract['seed']==42 and global_contract['new_formal_fits']==90
                   and global_contract['sources']==run.source_hashes())
    all_rows=[]
    query_rows=[]
    replay_rows=[]
    for domain in ('MATR','HUST'):
        bundle=load_bundle(run.ROOT,domain)
        root=output/domain
        registration=read(root/'RUN_REGISTRATION.json')
        matrix=read(root/'MATRIX_COMPLETE.json')
        checks.require(domain+'/source_and_data_fingerprint',registration['sources']==run.source_hashes()
                       and registration['tensor_fingerprint']==run.bundle_fingerprint(bundle)
                       and registration['tensor_fingerprint']==global_contract['N0_fingerprints'][domain]
                       and registration['settings']==dict(seed=42,epochs=150,batch_size=8,lr=.0003,weight_decay=.01,width=64,device=device))
        paths=sorted(root.glob('cell_fold_*/*/COMPLETE.json'))
        checks.require(domain+'/45_unique_fits_budget',len(paths)==45 and matrix['units']==45
                       and matrix['successful_updates']==global_contract['formal_optimizer_updates'][domain])
        n1,n2=verify_selections(checks,root,registration)
        for marker in paths:
            print(json.dumps(dict(event='VERIFY_UNIT',domain=domain,unit=marker.parent.relative_to(root).as_posix())),flush=True)
            rows,diagnostic,replay=replay_unit(checks,marker.parent,bundle,registration,n1,n2,device)
            all_rows.extend(rows)
            query_rows.extend(diagnostic)
            replay_rows.append(replay)
        if domain=='MATR':
            all_rows.extend(historical_metrics(checks,bundle))
    frame=pd.DataFrame(all_rows)
    for (domain,model,window),group in frame.groupby(['domain','model','window']):
        expected=169 if domain=='MATR' else 77
        checks.require(domain+'/'+model+'/'+window+'/each_cell_once',len(group)==expected and group.cell_uid.nunique()==expected)
    summary=independent_summary(frame)
    comparisons,pairs=independent_pairs(frame)
    analysis=output/'analysis'
    marker=read(analysis/'ANALYSIS_COMPLETE.json')
    checks.require('analysis_all_hashes_and_source',marker['source_sha256']==file_hash(run.ROOT/'n012/analyze.py')
                   and marker['evaluation_plan_sha256']==file_hash(run.ROOT/'research/N012_EVALUATION_PLAN_ZH.md')
                   and all(file_hash(analysis/item['path'])==item['sha256'] for item in marker['files']))
    compare_table(checks,'independent_cell_metrics',pd.read_csv(analysis/'cell_metrics.csv'),frame,
                  ['domain','model','cell_uid','window'],atol=1e-10,rtol=1e-8)
    compare_table(checks,'independent_summaries',pd.read_csv(analysis/'summary_metrics.csv'),summary,
                  ['domain','model','window'],atol=1e-10,rtol=1e-8)
    compare_table(checks,'independent_paired_CIs',pd.read_csv(analysis/'paired_comparisons.csv'),comparisons,
                  ['domain','metric','candidate','reference','sampling_unit'],atol=1e-10,rtol=1e-8)
    compare_table(checks,'independent_paired_cells',pd.read_csv(analysis/'paired_cells.csv'),pairs,
                  ['domain','metric','candidate','reference','cell_uid'],atol=1e-10,rtol=1e-8)
    for domain in ('MATR','HUST'):
        for phase in ('N1','N2'):
            published=read(output/domain/(phase+'_RESULTS.json'))
            for method,scores in list(published['models'].items())+[('nested',published['nested'])]:
                row=summary[(summary.domain==domain)&(summary.phase==phase)&(summary.method==method)&(summary.window=='full')].iloc[0]
                close(checks,domain+'/'+phase+'/'+method+'/true_cell_weighted_output',
                      [scores['n_cells'],scores['n_obs'],scores['mae'],scores['rmse']],
                      [row.supported_cells,row.observations,row.cell_mean_mae,row.cell_mean_rmse],atol=1e-10)
        a=frame[(frame.domain==domain)&(frame.model=='N1:nested')].sort_values(['cell_uid','window'])
        b=frame[(frame.domain==domain)&(frame.model=='N2:query_mlp')].sort_values(['cell_uid','window'])
        close(checks,domain+'/N1_nested_equals_reused_N2_query',a[['mae','rmse','n_obs']],b[['mae','rmse','n_obs']],atol=0,rtol=0)
    checks.require('all90_replays_exact_141750_updates',len(replay_rows)==90 and all(row['replay_error']==0 for row in replay_rows)
                   and sum(row['updates'] for row in replay_rows)==141750)
    destination=output/'verification'
    destination.mkdir(exist_ok=True)
    pd.DataFrame(query_rows).to_csv(destination/'query_diagnostics.csv',index=False)
    pd.DataFrame(replay_rows).to_csv(destination/'unit_replays.csv',index=False)
    return dict(status='VERIFIED',seed=42,formal_fits=90,formal_optimizer_updates=141750,
                normal_selected_checkpoint_replays=90,max_absolute_prediction_replay_error=0.,
                source_checkpoint='f80ee8f65b751d798bf59c9d5b08dd97bf1a71ed',
                files=run.file_records(destination,['query_diagnostics.csv','unit_replays.csv']),
                analysis_complete_sha256=file_hash(analysis/'ANALYSIS_COMPLETE.json'),
                data_domains=['MATR','HUST'],Stanford='N0_BLOCKED',
                limitations=['Single seed42, historically exposed cells and unequal parameter counts.',
                             'One seed42 fit per registered encoder/decoder/fold unit. Resume determinism is tested by synthetic interruption regression.',
                             'Decoder mathematics independently verified given replayed encoder states. The same encoder architecture performs checkpoint replay.',
                             'No certified electrochemical mechanism or observed Stanford native-cycle training tensor.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=run.ROOT/'results/N012_independent')
    parser.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    args=parser.parse_args()
    output=args.out.resolve()
    checks=Checks()
    if (output/'RUN_VERIFICATION.json').exists():
        previous=(output/'RUN_VERIFICATION.json').read_bytes()
        attempts=output/'verification_attempts'
        attempts.mkdir(exist_ok=True)
        (attempts/('RUN_VERIFICATION_'+file_hash(output/'RUN_VERIFICATION.json')+'.json')).write_bytes(previous)
    try:
        with threadpool_limits(limits=1):
            result=verify(output,args.device,checks)
    except Exception:
        result=dict(status='FAIL',traceback=traceback.format_exc())
    result['checks']=checks.rows
    result['counts']=dict(Counter(row['status'] for row in checks.rows))
    result['verifier_source_sha256']=file_hash(__file__)
    run.write_json(output/'RUN_VERIFICATION.json',result)
    print(json.dumps(dict(status=result['status'],counts=result['counts'])),flush=True)
    if result['status']=='FAIL':
        print(result['traceback'],flush=True)
        raise SystemExit(1)


if __name__=='__main__':
    main()
