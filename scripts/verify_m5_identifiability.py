"""Independent NumPy replay of train-only Ridge and future Hinge diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.unified_matr.train_seed42 import load_frozen_data, sha256, write_json
from scripts.verify_unified_seed42_results import Checks, compare_table, clean


def arrays_equal(checks, name, actual, expected, atol=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    checks.require(name, a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=1e-8, equal_nan=True))


def design_matrices(inputs, ids):
    prior = np.concatenate((np.where(inputs['policy_mask'], inputs['policy'], np.nan),
                            inputs['policy_mask'].astype(float), inputs['anchor_soh100'][:, None]), axis=1)
    extra, audit = [], []
    for i, uid in enumerate(ids):
        slots = []
        for t in range(inputs['cycle_ids'].shape[1]):
            if inputs['point_mask'][i, t].any():
                cycle = inputs['cycle_ids'][i, t]
                if not (np.isfinite(cycle) and 1 <= cycle <= 100 and cycle == int(cycle)):
                    raise ValueError('Nonphysical prefix coordinate')
                if inputs['global_mask'][i, t].any() or inputs['local_feature_mask'][i, t].any():
                    slots.append(t)
        value, mask = np.full(36, np.nan), np.zeros(36, bool)
        cycle = np.nan
        if slots:
            t = slots[-1]
            cycle = float(inputs['cycle_ids'][i, t])
            for j in range(20):
                mask[j] = inputs['global_mask'][i, t, j]
                if mask[j]:
                    value[j] = inputs['global_features'][i, t, j]
            for b in range(8):
                for j in range(2):
                    pos = 20 + 2 * b + j
                    mask[pos] = inputs['local_feature_mask'][i, t, b, j]
                    if mask[pos]:
                        value[pos] = inputs['local_features'][i, t, b, j]
        extra.append(np.concatenate((value, mask.astype(float))))
        audit.append(dict(cell_uid=uid, selected_physics_cycle=cycle, qualified_prefix_records=len(slots),
                          global_feature_support=int(mask[:20].sum()), local_mean_rms_support=int(mask[20:].sum()),
                          prefix_feature_cutoff=100))
    return {'C0': prior, 'C1': np.column_stack((prior, extra))}, pd.DataFrame(audit)


def regression_metrics(frame):
    error = frame.prediction.to_numpy() - frame.truth.to_numpy()
    baseline = np.abs(frame.training_mean_prediction.to_numpy() - frame.truth.to_numpy()).mean()
    rank = np.nan
    if len(frame) > 1 and frame.truth.nunique() > 1 and frame.prediction.nunique() > 1:
        rank = np.corrcoef(rankdata(frame.truth), rankdata(frame.prediction))[0, 1]
    return dict(unique_test_cells=frame.cell_uid.nunique(), observations=len(frame), mae=np.abs(error).mean(),
                signed_bias=error.mean(), training_mean_mae=baseline,
                skill_vs_training_mean=1-np.abs(error).mean()/baseline if baseline else np.nan, spearman=rank)


def profile_fit(x, y):
    """Independent least-squares profile using polynomial coordinate construction."""
    n = len(x)
    span = float(x[-1]-x[0]) if n else 0.
    row = dict(status='INSUFFICIENT_SUPPORT', raw_observations=n, raw_cycle_span=span,
               certified_physical_knee=False, conditional_linear_coefficients=3,
               free_hinge_parameter_count=4, quadratic_parameter_count=3)
    if n < 240 or span < 300:
        return row
    take = np.unique(np.rint(np.linspace(0, n-1, 500)).astype(int)) if n > 500 else np.arange(n)
    x, y = x[take], y[take]
    scale = max(x.std(), 1.)
    z = (x-x.mean())/scale
    quad = np.vander(z, 3, increasing=True)
    qsse = np.linalg.norm(y-quad@np.linalg.lstsq(quad, y, rcond=None)[0])**2
    grid = np.unique(x[np.linspace(int(.2*len(x)), int(.8*len(x)), 75).astype(int)])
    errors, slopes = [], []
    for k in grid:
        mat = np.stack((np.ones_like(x), z, np.clip(x-k, 0, None)/scale), axis=1)
        beta = np.linalg.lstsq(mat, y, rcond=None)[0]
        errors.append(np.sum((y-mat@beta)**2))
        slopes.append((beta[1]/scale, (beta[1]+beta[2])/scale))
    best = int(np.argmin(errors))
    compatible = grid[np.asarray(errors) <= 1.05*errors[best]+1e-12]
    score = len(x)*np.log(max(qsse, 1e-16)/max(errors[best], 1e-16))
    row.update(status='FITTED_UNVALIDATED_PROXY', fitted_observations=len(x), kappa_proxy=grid[best],
               knee_proxy_center_cycle=grid[best], profile_lower_cycle=compatible.min(), profile_upper_cycle=compatible.max(),
               profile_width_5pct_SSE=np.ptp(compatible), candidate_count=len(grid), candidate_grid_min=grid.min(),
               candidate_grid_max=grid.max(), median_candidate_spacing_cycles=np.median(np.diff(grid)),
               profile_reaches_left_grid=bool(compatible.min()==grid.min()), profile_reaches_right_grid=bool(compatible.max()==grid.max()),
               hinge_sse=errors[best], quadratic_sse=qsse, pre_SOHSlope=slopes[best][0], post_SOHSlope=slopes[best][1],
               conditional_three_coefficient_log_sse_ratio_n=score,
               bic_quad_minus_free_hinge=score-np.log(len(x)), bic_quad_minus_hinge=score-np.log(len(x)))
    return row


def verify(checks, data_root, output):
    complete = json.loads((output/'ANALYSIS_COMPLETE.json').read_text(encoding='utf-8'))
    contract = json.loads((output/'IDENTIFIABILITY_CONTRACT.json').read_text(encoding='utf-8'))
    checks.require('completion_and_source', complete['status']=='ANALYZED'
                   and complete['source_sha256']==sha256(ROOT/'scripts/unified_matr/analyze_m5_identifiability.py')
                   and complete['contract_sha256']==sha256(output/'IDENTIFIABILITY_CONTRACT.json'))
    for item in complete['files']:
        checks.require('artifact/'+item['path'], sha256(output/item['path'])==item['sha256'])
    for item in contract['frozen_files']:
        checks.require('data/'+item['path'], sha256(data_root/item['path'])==item['sha256'])
    inputs, ids, eligible, cells, _, targets, observed, rates, mask, splits, _ = load_frozen_data(data_root)
    checks.require('169_all_eligible', len(ids)==169 and eligible.all())
    designs, audit = design_matrices(inputs, ids)
    compare_table(checks, 'early_feature_cutoff', pd.read_csv(output/'early_feature_audit.csv'), audit, ['cell_uid'])
    checks.require('registered_dimensions', {k:v.shape[1] for k,v in designs.items()}=={'C0':31,'C1':103})
    shared = mask[:, 0]&mask[:, 1]
    tasks = {'level_mid_all':(rates[:,0].astype(float),mask[:,0]),
             'level_mid_shared':(rates[:,0].astype(float),shared),
             'level_late_shared':(rates[:,1].astype(float),shared),
             'change_late_minus_mid_shared':(rates[:,1].astype(float)-rates[:,0],shared)}
    checks.require('support151_and42', [int(v[1].sum()) for v in tasks.values()]==[151,42,42,42])
    lookup = {uid:i for i,uid in enumerate(ids)}
    models = json.loads((output/'SELECTED_RIDGE_MODELS.json').read_text(encoding='utf-8'))
    checks.require('40_selected_200_candidates', len(models['models'])==40 and models['candidate_fits']==200)
    records = []
    for model in models['models']:
        key = model['model_id']
        y, support = tasks[model['task']]
        roles = {role:[lookup[uid] for uid in splits['splits'][model['split']][role] if support[lookup[uid]]]
                 for role in ('train','validation','test')}
        checks.require(key+'/roles', model['roles']=={role:[ids[i] for i in positions] for role,positions in roles.items()}
                       and not model['refit_train_plus_validation'])
        x = designs[model['model']].copy()
        train = x[roles['train']]
        median = np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0. for column in train.T])
        x = np.where(np.isfinite(x), x, median)
        mean = x[roles['train']].mean(axis=0)
        scale = x[roles['train']].std(axis=0)
        scale[scale==0]=1
        x = (x-mean)/scale
        arrays_equal(checks,key+'/train_only_median',model['imputer_medians'],median)
        arrays_equal(checks,key+'/train_only_scaler_mean',model['scaler_mean'],mean)
        arrays_equal(checks,key+'/train_only_scaler_scale',model['scaler_scale'],scale)
        xt = x[roles['train']]
        xmean, ymean = xt.mean(axis=0), (1000*y[roles['train']]).mean()
        centered, response = xt-xmean, 1000*y[roles['train']]-ymean
        candidates=[]
        for alpha in (.01,.1,1.,10.,100.):
            coef = np.linalg.solve(centered.T@centered+alpha*np.eye(x.shape[1]),centered.T@response)
            intercept = ymean-xmean@coef
            error = np.abs(x[roles['validation']]@coef+intercept-1000*y[roles['validation']]).mean()
            candidates.append((error,alpha,coef,intercept))
        best=min(candidates,key=lambda item:item[0])
        checks.require(key+'/strict_val_first_min_alpha', best[1]==model['alpha'])
        arrays_equal(checks,key+'/all_val_trials',[item[0] for item in candidates],
                     [item['validation_mae_per1000'] for item in model['trials']])
        arrays_equal(checks,key+'/train_only_coefficients',model['ridge_coef_per1000'],best[2])
        arrays_equal(checks,key+'/train_only_intercept',model['ridge_intercept_per1000'],best[3])
        for i in roles['test']:
            records.append(dict(task=model['task'],model=model['model'],split=model['split'],cell_uid=ids[i],
                                protocol_uid=str(cells.loc[ids[i],'protocol_uid']),truth=y[i],
                                prediction=(x[i]@best[2]+best[3])/1000,training_mean_prediction=y[roles['train']].mean()))
    predicted=pd.DataFrame(records)
    saved=pd.read_csv(output/'rate_predictability_oof.csv')
    compare_table(checks,'independent_numpy_test_predictions',saved,predicted,['task','model','split','cell_uid'],atol=1e-12)
    summary,protocols=[],[]
    for (task,model),group in predicted.groupby(['task','model'],sort=True):
        sub=[]
        for protocol,part in group.groupby('protocol_uid',sort=True):
            row=dict(task=task,model=model,protocol_uid=protocol,**regression_metrics(part))
            sub.append(row)
            protocols.append(row)
        summary.append(dict(task=task,model=model,**regression_metrics(group),
                            protocol_macro_mae=np.mean([r['mae'] for r in sub]),supported_protocols=len(sub)))
    compare_table(checks,'independent_rate_metrics',pd.read_csv(output/'rate_predictability_summary.csv'),pd.DataFrame(summary),['task','model'])
    compare_table(checks,'independent_protocol_metrics',pd.read_csv(output/'rate_predictability_protocol_metrics.csv'),pd.DataFrame(protocols),['task','model','protocol_uid'])
    pair=saved[saved.model=='C0'].merge(saved[saved.model=='C1'],on=['task','split','cell_uid','protocol_uid'],suffixes=('_C0','_C1'),validate='one_to_one')
    pair['improvement_abs_error']=pair.abs_error_C0-pair.abs_error_C1
    pair['difference_signed_error']=pair.signed_error_C1-pair.signed_error_C0
    compare_table(checks,'paired_differences',pd.read_csv(output/'rate_predictability_paired_cells.csv'),
                  pair[['task','split','cell_uid','protocol_uid','improvement_abs_error','difference_signed_error']],['task','split','cell_uid'])
    from scripts.verify_unified_seed42_results import bootstrap_average
    for row in pd.read_csv(output/'rate_predictability_paired_summary.csv').itertuples(index=False):
        part=pair[pair.task==row.task]
        values=part.improvement_abs_error.to_numpy() if row.sampling_unit=='cell' else part.groupby('protocol_uid',sort=True).improvement_abs_error.mean().to_numpy()
        expected=bootstrap_average(values)
        arrays_equal(checks,'paired_bootstrap/'+row.task+'/'+row.sampling_unit,
                     [row.improvement,row.ci95_low,row.ci95_high],[expected[k] for k in ('improvement','ci95_low','ci95_high')])
    rows=[]
    for i,uid in enumerate(ids):
        x=np.flatnonzero(observed[i])+101
        y=targets[i,observed[i]].astype(float)
        for start in (101,201,301):
            use=x>=start
            rows.append(dict(cell_uid=uid,fit_start_cycle=start,**profile_fit(x[use],y[use])))
    compare_table(checks,'all507_independent_hinge_profiles',pd.read_csv(output/'knee_proxy_stability.csv'),pd.DataFrame(rows),['cell_uid','fit_start_cycle'],atol=1e-8)
    x=np.arange(101,2001,dtype=float)
    negatives=[]
    for name,y in {'smooth_quadratic':1.02-2e-5*(x-100)-6e-8*(x-100)**2,'smooth_exponential':1.02*np.exp(-.00015*(x-100))}.items():
        for start in (101,201,301):
            use=x>=start
            negatives.append(dict(synthetic_source=name,fit_start_cycle=start,synthetic_true_knee_event=False,**profile_fit(x[use],y[use])))
    compare_table(checks,'six_smooth_negative_controls',pd.read_csv(output/'knee_proxy_negative_controls.csv'),pd.DataFrame(negatives),['synthetic_source','fit_start_cycle'],atol=1e-8)
    mid,late=rates[shared,0].astype(float),rates[shared,1].astype(float)
    stats=contract['shared_label_statistics']
    arrays_equal(checks,'shared_label_statistics',[stats['positive_late_minus_mid'],stats['median_late_over_mid'],stats['pearson_mid_late']],
                 [(late>mid).sum(),np.median(late/mid),np.corrcoef(mid,late)[0,1]])
    return dict(selected_models=40,candidate_fits_replayed=200,OOF_rows=len(saved),hinge_qualification_rows=len(rows),
                frozen_files=len(contract['frozen_files']),max_feature_cycle=100,optimizer_updates=0,
                analysis_completion_sha256=sha256(output/'ANALYSIS_COMPLETE.json'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    checks=Checks()
    result={'status':'FAIL','verifier_source_sha256':sha256(__file__)}
    try:
        with threadpool_limits(limits=1):
            result.update(verify(checks,args.data,args.output))
        result['status']='VERIFIED'
    except Exception:
        result['traceback']=traceback.format_exc()
        raise
    finally:
        result['checks']=checks.rows
        result['passed_checks']=sum(row['status']=='PASS' for row in checks.rows)
        write_json(args.output/'RUN_VERIFICATION.json',clean(result))
    print(json.dumps({key:result[key] for key in ('status','passed_checks','selected_models','hinge_qualification_rows')}))


if __name__=='__main__':
    main()
