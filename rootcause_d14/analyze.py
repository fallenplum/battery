"""Frozen cell-weighted metrics, explicit contrasts and independent replay."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from n012.data import role_indices,file_hash
from n012.run import configure,write_json,file_records,verify_files,metrics
from n012.analyze import historical_m1
from .data import ROOT,OUT,SCOPE,ORACLE,settings,load_domain,baseline_path,scaled_inputs,selected_encoder
from .model import RootCauseNet
from .run import check_sources,predict

WINDOWS = [('all',101,np.inf),('101_200',101,200),('201_500',201,500),('501_1000',501,1000),('1001_plus',1001,np.inf)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def cell_scores(b,ids,query,pred,target,observed,arm,fold,scope=SCOPE,role='test',start=None):
    records = []
    for j,uid in enumerate(ids):
        meta = b.cell_metadata.loc[uid]
        crossing = meta['eol80_first_observed_crossing_cycle']
        event = bool(meta['eol80_event_observed'])
        windows = WINDOWS+([('pre_eol80',101,float(crossing)),('post_eol80',float(crossing)+1,np.inf)] if event else [])
        for window,lo,hi in windows:
            m = observed[j] & (query>=lo) & (query<=hi)
            e = pred[j,m].astype(float)-target[j,m].astype(float)
            vals = pred[j,m]
            records.append(dict(prediction_scope=scope,domain=b.domain,arm=arm,fold=fold,role=role,
                cell_uid=uid,protocol_uid=str(meta['protocol_uid']),window=window,n_obs=int(m.sum()),
                mae=float(np.abs(e).mean()) if len(e) else np.nan,
                rmse=float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,bias=float(e.mean()) if len(e) else np.nan,
                negative_predictions=int((vals<0).sum()),above1p2_predictions=int((vals>1.2).sum()),
                nonfinite_predictions=int((~np.isfinite(vals)).sum()),eol_event=event,
                observed_target_rises=int((np.diff(target[j,m])>1e-6).sum()),
                last_observed_cycle=float(query[observed[j]].max()),start=start))
    return records


def legal_cells():
    records,units = [],[]
    for domain in settings()['domains']:
        d = load_domain(domain)
        b = d.bundle
        for fold in settings()['folds']:
            roles = role_indices(b,fold)
            paths = {'D1-0':baseline_path(domain,fold)}
            paths |= {'N2-'+dec:baseline_path(domain,fold,dec) for dec in ('free_rate','monotone_spline','physics_basis')}
            paths |= {arm:OUT/domain/fold/arm for arm in settings()['arms']}
            for arm,path in paths.items():
                complete = read(path/'COMPLETE.json')
                verify_files(path,complete)
                with np.load(path/'test_predictions.npz',allow_pickle=False) as a:
                    ids = a['ids'].astype(str).tolist()
                    assert ids == [b.ids[i] for i in roles['test']]
                    np.testing.assert_array_equal(a['query'],b.query)
                    np.testing.assert_array_equal(a['observed'],b.observed[roles['test']])
                    np.testing.assert_array_equal(a['target'],b.targets[roles['test']])
                    assert np.isfinite(a['prediction']).all()
                    records += cell_scores(b,ids,a['query'],a['prediction'],a['target'],a['observed'],arm,fold)
                if arm in settings()['arms']:
                    tr = read(path/'TRAINING_COMPLETE.json')
                    h = read(path/'HISTORY.json')
                    ct = tr['unit_contract']
                    assert len(h)==150 and tr['updates']==ct['expected_updates']
                    assert int(np.argmin([r['val_mae'] for r in h]))+1 == tr['best_epoch']
                    units.append(dict(prediction_scope=SCOPE,domain=domain,fold=fold,arm=arm,
                        parameters=ct['parameters'],extra_params=ct['extra_params'],updates=tr['updates'],
                        best_epoch=tr['best_epoch'],validation_mae=tr['validation_mae'],
                        seconds=tr['elapsed_seconds'],checkpoint=(path/'BEST.pt').relative_to(ROOT).as_posix(),
                        checkpoint_sha256=file_hash(path/'BEST.pt'),resume_checkpoint=(path/'TRAIN_STATE.pt').relative_to(ROOT).as_posix(),
                        resume_sha256=file_hash(path/'TRAIN_STATE.pt')))
        if domain == 'MATR':
            old,identities = historical_m1(b)
            for r in old:
                r.update(prediction_scope=SCOPE,arm='HISTORICAL-M1',role='test')
                if r['window']=='full':
                    r['window']='all'
                r['above1p2_predictions'] = r['predictions_above1p2']
            records += old
            write_json(OUT/'HISTORICAL_M1_BINDINGS.json',identities)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'cell_metrics.csv',index=False)
    pd.DataFrame(units).to_csv(OUT/'CHECKPOINT_INDEX.csv',index=False)
    return frame,pd.DataFrame(units)


def summaries(frame):
    rows = []
    for (scope,domain,arm,role,window),g in frame.groupby(['prediction_scope','domain','arm','role','window']):
        valid = g[g.n_obs>0]
        proto = valid.groupby('protocol_uid').mae.mean()
        rows.append(dict(prediction_scope=scope,domain=domain,arm=arm,role=role,window=window,
            supported_cells=valid.cell_uid.nunique(),cell_fold_role_occurrences=len(valid),
            aggregation_unit='cell_fold_role',observations=int(valid.n_obs.sum()),cell_mean_mae=valid.mae.mean(),
            cell_mean_rmse=valid.rmse.mean(),bias=valid.bias.mean(),protocol_macro_mae=proto.mean(),
            protocols=len(proto),negative_predictions=int(valid.negative_predictions.sum()),
            above1p2_predictions=int(valid.above1p2_predictions.sum())))
    return pd.DataFrame(rows)


def boot(values,protocols):
    rng = np.random.default_rng(42)
    values = np.asarray(values,float)
    draws = values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
    groups = pd.DataFrame(dict(value=values,protocol=protocols)).groupby('protocol').value.mean().to_numpy()
    pdraws = groups[rng.integers(0,len(groups),(2000,len(groups)))].mean(1)
    return dict(cell_ci_low=float(np.quantile(draws,.025)),cell_ci_high=float(np.quantile(draws,.975)),
                protocol_macro_improvement=float(groups.mean()),protocol_ci_low=float(np.quantile(pdraws,.025)),
                protocol_ci_high=float(np.quantile(pdraws,.975)))


def comparisons(frame):
    output,paired,interaction,concentration = [],[],[],[]
    contrasts = [('D1-0',arm) for arm in settings()['arms']]
    contrasts += [('D1-0-capmatch','D1-GL'),('D1-0','D1-0-capmatch'),('D1-GL','D2-F1L1'),
                  ('D2-F0L1','D2-F1L1'),('D2-F1L1','D2-GL-rate-spline'),('D2-F1L1','D2-GL-rate-physics'),
                  ('D2-GL-rate-spline','D2-GL-rate-physics'),
                  ('N2-physics_basis','D2-GL-rate-physics'),('N2-monotone_spline','D2-GL-rate-spline'),
                  ('HISTORICAL-M1','D2-F1L1')]
    for domain in settings()['domains']:
        sub = frame[(frame.domain==domain)&(frame.role=='test')]
        for window,lo,hi in WINDOWS:
            g = sub[(sub.window==window)&(sub.n_obs>0)]
            for ref,cand in dict.fromkeys(contrasts):
                a,b = [g[g.arm==name].set_index('cell_uid').sort_index() for name in (ref,cand)]
                if not len(a) or not len(b):
                    continue
                assert a.index.equals(b.index)
                diff = a.mae.to_numpy()-b.mae.to_numpy()
                common = dict(prediction_scope=SCOPE,domain=domain,window=window,reference=ref,candidate=cand)
                output.append(dict(**common,n_cells=len(diff),mean_improvement=float(diff.mean()),
                    median_improvement=float(np.median(diff)),better=int((diff>0).sum()),worse=int((diff<0).sum()),
                    **boot(diff,a.protocol_uid.to_numpy())))
                for uid,v in zip(a.index,diff):
                    paired.append(dict(**common,cell_uid=uid,improvement=v,reference_mae=a.loc[uid,'mae'],candidate_mae=b.loc[uid,'mae']))
                positive = np.maximum(diff,0)
                concentration.append(dict(**common,n_cells=len(diff),positive_sum=float(positive.sum()),
                    top3_fraction_positive=float(np.sort(positive)[-3:].sum()/positive.sum()) if positive.sum() else np.nan,
                    largest_harm_cell=str(a.index[np.argmin(diff)]),largest_harm=float(diff.min())))
            arrays = {name:g[g.arm==name].set_index('cell_uid').sort_index() for name in ('D1-0','D1-GL','D2-F0L1','D2-F1L1')}
            f00,f10,f01,f11 = [arrays[name].mae.to_numpy() for name in arrays]
            for name,diff in [('feature_main',.5*((f00-f10)+(f01-f11))),('rate_main',.5*((f00-f01)+(f10-f11))),
                              ('interaction_gain',(f10-f11)-(f00-f01))]:
                interaction.append(dict(prediction_scope=SCOPE,domain=domain,window=window,effect=name,n_cells=len(diff),
                    mean_improvement=float(diff.mean()),median_improvement=float(np.median(diff)),
                    better=int((diff>0).sum()),worse=int((diff<0).sum()),**boot(diff,arrays['D1-0'].protocol_uid.to_numpy())))
    pd.DataFrame(output).to_csv(OUT/'paired_comparisons.csv',index=False)
    pairs = pd.DataFrame(paired)
    pairs.to_csv(OUT/'paired_cells.csv',index=False)
    pairs.sort_values('improvement').groupby(['domain','window','reference','candidate'],sort=False).head(5).to_csv(OUT/'worst_cases.csv',index=False)
    pd.DataFrame(interaction).to_csv(OUT/'factorial_effects.csv',index=False)
    pd.DataFrame(concentration).to_csv(OUT/'long_horizon_concentration.csv',index=False)


def oracle_cells(legal):
    records,initializations,bindings = [],[],[]
    for domain in settings()['domains']:
        d = load_domain(domain)
        b = d.bundle
        for stage in ('DEVELOPMENT','TEST'):
            bind = read(OUT/'ORACLE_DIAGNOSTIC'/domain/(stage+'_BINDINGS.json'))
            bindings.extend(bind)
            for r in bind:
                path = OUT/r['path']
                logs = read(path/'STARTS.json')
                best = logs['best_start']
                with np.load(path/'predictions.npz') as a:
                    q,y,pred = a['query'],a['target'],a['prediction'][best]
                records += cell_scores(b,[r['cell_uid']],q,pred[None],y[None],np.ones((1,len(q)),bool),
                                      r['family'],r['fold'],ORACLE,r['role'],best)
        paths = sorted((OUT/'ORACLE_DIAGNOSTIC'/domain).glob('**/STARTS.json'))
        for p in paths:
            logs = read(p)
            for r in logs['starts']:
                initializations.append(dict(prediction_scope=ORACLE,domain=domain,cell_uid=logs['cell_uid'],
                    family=logs['family'],fold_binding=logs['fold_binding'],**r))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'oracle_cell_metrics.csv',index=False)
    summaries(frame).to_csv(OUT/'oracle_summary_metrics.csv',index=False)
    fold_summaries=[]
    for fold,part in frame.groupby('fold'):
        table=summaries(part)
        table['fold']=fold
        fold_summaries.append(table)
    pd.concat(fold_summaries).to_csv(OUT/'oracle_fold_summary_metrics.csv',index=False)
    init = pd.DataFrame(initializations)
    for key in ('initial','raw','parameters'):
        init[key] = init[key].apply(json.dumps)
    init.to_csv(OUT/'oracle_initializations.csv',index=False)
    gaps = []
    for domain in settings()['domains']:
        for family in settings()['oracle']['families']:
            o = frame[(frame.domain==domain)&(frame.role=='test')&(frame.arm==family)&(frame.window=='all')].set_index('cell_uid')
            assert o.index.is_unique
            for arm in ('D1-0','N2-physics_basis','D2-GL-rate-physics'):
                g = legal[(legal.domain==domain)&(legal.arm==arm)&(legal.window=='all')].set_index('cell_uid').loc[o.index]
                for uid in o.index:
                    gaps.append(dict(prediction_scope=ORACLE,domain=domain,cell_uid=uid,oracle_family=family,legal_arm=arm,
                        oracle_mae=o.loc[uid,'mae'],legal_mae=g.loc[uid,'mae'],fit_gap=g.loc[uid,'mae']-o.loc[uid,'mae']))
    pd.DataFrame(gaps).to_csv(OUT/'oracle_fit_gaps.csv',index=False)
    quantiles = frame[(frame.role=='test')&(frame.n_obs>0)].groupby(['domain','arm','window']).mae.quantile([0,.25,.5,.75,.9,1]).reset_index()
    quantiles.rename(columns={'level_3':'quantile'},inplace=True)
    quantiles.insert(0,'prediction_scope',ORACLE)
    quantiles.to_csv(OUT/'oracle_quantiles.csv',index=False)
    return frame,init


def verify_replay(device='cuda'):
    results,query_rows = [],[]
    for domain in settings()['domains']:
        d = load_domain(domain)
        for fold in settings()['folds']:
            roles,x,p,phy,scaler = scaled_inputs(d,fold)
            for arm,spec in settings()['arms'].items():
                path = OUT/domain/fold/arm
                tr = read(path/'TRAINING_COMPLETE.json')
                verify_files(path,tr)
                assert read(path/'SCALER_TRAIN_ONLY.json')==scaler
                model = RootCauseNet(selected_encoder(domain,fold),spec['decoder'],d.bundle.policy.shape[1],spec['features']).to(device)
                cp = torch.load(path/'BEST.pt',map_location=device,weights_only=False)
                model.load_state_dict(cp['model'])
                pred = predict(model,d,x,p,phy,roles['test'],device)
                with np.load(path/'test_predictions.npz') as a:
                    np.testing.assert_array_equal(pred,a['prediction'])
                val = predict(model,d,x,p,phy,roles['validation'],device)
                val_mae = metrics(val,d.bundle.targets[roles['validation']],d.bundle.observed[roles['validation']],d.bundle.query)['mae']
                assert val_mae==cp['validation_mae']
                # Decoder-only numerical support beyond followup has no accuracy interpretation.
                from .run import batch
                inp = batch(d,x,p,phy,roles['test'],device)
                query = torch.arange(100,5001,device=device,dtype=torch.float32)
                inp['query'] = query
                model.eval()
                with torch.no_grad():
                    dense = model(**inp).cpu().numpy()
                assert np.isfinite(dense).all()
                for label,mask in [('dense_100_5000',np.ones_like(dense,bool)),
                    ('far_outside_observed',query.cpu().numpy()[None]>np.array([d.bundle.query[d.bundle.observed[i]].max() for i in roles['test']])[:,None])]:
                    v = dense[mask]
                    query_rows.append(dict(prediction_scope=SCOPE,domain=domain,fold=fold,arm=arm,query_regime=label,
                        numerical_diagnostic_only=True,n_values=len(v),minimum=float(v.min()),maximum=float(v.max()),
                        negative=int((v<0).sum()),above1p2=int((v>1.2).sum()),nonfinite=int((~np.isfinite(v)).sum())))
                state = torch.load(path/'TRAIN_STATE.pt',map_location='cpu',weights_only=False)
                assert state['next_epoch']==150 and state['updates']==tr['updates']
                steps = [int(v['step']) for v in state['optimizer']['state'].values()]
                assert all(n==tr['updates'] for n in steps)
                results.append(dict(prediction_scope=SCOPE,domain=domain,fold=fold,arm=arm,
                    replay_max_absolute_error=0.,validation_mae_replayed=val_mae,updates=tr['updates'],
                    optimizer_parameter_states=len(steps),hash_verified=True))
                del model,state
        print(json.dumps(dict(event='REPLAY',domain=domain,units=40)),flush=True)
    pd.DataFrame(results).to_csv(OUT/'replay_verification.csv',index=False)
    pd.DataFrame(query_rows).to_csv(OUT/'query_diagnostics.csv',index=False)
    write_json(OUT/'REPLAY_COMPLETE.json',dict(status='PASS',units=len(results),max_absolute_error=0.,
        updates=sum(r['updates'] for r in results),files=file_records(OUT,['replay_verification.csv','query_diagnostics.csv'])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['metrics','replay'],required=True)
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    configure()
    check_sources()
    if a.phase=='replay':
        verify_replay(a.device)
        return
    legal,units = legal_cells()
    summaries(legal).to_csv(OUT/'summary_metrics.csv',index=False)
    comparisons(legal)
    oracle,initializations = oracle_cells(legal)
    write_json(OUT/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',legal_neural_units=len(units),
        formal_updates=int(units.updates.sum()),unique_legal_evaluated_cells=246,
        oracle_individual_multistart_problems=int(initializations[['domain','cell_uid','family','fold_binding']].drop_duplicates().shape[0]),
        oracle_individual_optimizer_starts=len(initializations),oracle_shared_joint_problems=10,oracle_shared_joint_optimizer_starts=80,
        oracle_not_global_optimality_certified=True,seed=42,
        source_registration_sha256=file_hash(OUT/'SOURCE_REGISTRATION.json')))


if __name__=='__main__':
    main()
