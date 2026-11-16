"""Independent metric/normalization/source closure for the registered experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from n012.data import role_indices,file_hash
from n012.verify import independent_scaler
from n012.model import PhysicsBasis,MonotoneSpline
from n012.run import write_json,configure
from .data import ROOT,OUT,SCOPE,FEATURES,settings,load_domain
from .oracle import evaluate,starts,physics_values
from .run import check_sources


def main():
    configure()
    check_sources()
    checks = []
    def check(label,condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    cells = pd.read_csv(OUT/'cell_metrics.csv')
    summary = pd.read_csv(OUT/'summary_metrics.csv')
    # Compare NumPy diagnostic functions to the original float64 torch heads.
    q = np.array([100.,101.,171.,200.,500.,1000.,2600.,6000.])
    formula_checks=[]
    for family,cls in [('physics_free',PhysicsBasis),('spline10',MonotoneSpline)]:
        for j,raw in enumerate(starts(family)):
            net = cls(1).double()
            dim = 4 if family=='physics_free' else 10
            net.net = torch.nn.Linear(1,dim).double()
            with torch.no_grad():
                net.net.weight.zero_()
                net.net.bias.copy_(torch.tensor(raw[:dim],dtype=torch.float64))
                if family=='physics_free':
                    net.log_c.copy_(torch.tensor(raw[4],dtype=torch.float64))
                    net.log_w.copy_(torch.tensor(raw[5],dtype=torch.float64))
                expected = net(torch.zeros(1,1,dtype=torch.float64),torch.tensor([.98],dtype=torch.float64),
                               torch.tensor(q[None],dtype=torch.float64)).numpy()[0]
            actual = evaluate(family,raw,(q-100)/1000,.98)[0]
            tolerance=2e-14
            if family=='physics_free':
                parameters=physics_values(raw)
                # Torch softplus uses x above 20; exact softplus differs by at most exp(-20).
                tolerance+=.98*parameters[2]*parameters[5]*np.exp(-20)
            error=float(np.max(np.abs(actual-expected)))
            check(f'original_head_formula_{family}_{j}',error<=tolerance)
            formula_checks.append(dict(prediction_scope='ORACLE_DIAGNOSTIC',family=family,start=j,
                maximum_absolute_error=error,analytic_absolute_bound=tolerance))
    write_json(OUT/'ORACLE_TORCH_NUMERIC_CHECK.json',dict(status='PASS',checks=formula_checks,
        bound_reason='Torch softplus threshold20 linear branch versus exact float64 logaddexp'))
    for domain in settings()['domains']:
        d = load_domain(domain)
        b = d.bundle
        ids_seen = {arm:[] for arm in settings()['arms']}
        for fold in settings()['folds']:
            roles = role_indices(b,fold)
            for arm in settings()['arms']:
                path = OUT/domain/fold/arm
                scaler = json.loads((path/'SCALER_TRAIN_ONLY.json').read_text())
                check(f'{domain}_{fold}_{arm}_scaler_UIDs',scaler['training_ids']==[b.ids[i] for i in roles['train']])
                actual = independent_scaler(b,roles['train'])
                for k,v in actual.items():
                    check(f'{domain}_{fold}_{arm}_raw_scaler_{k}',np.allclose(v,scaler['raw'][k],rtol=2e-10,atol=1e-11))
                for key,mask in FEATURES.values():
                    v = d.arrays[key][roles['train']].astype(float)
                    m = d.arrays[mask][roles['train']]
                    stored = scaler['physics'][key]
                    if key=='local_residual':
                        flat = v[m]
                        means,stds = flat.mean(),max(flat.std(),1e-5)
                    else:
                        flat,valid = v.reshape(-1,v.shape[-1]),m.reshape(-1,v.shape[-1])
                        values = [flat[valid[:,k],k] for k in range(flat.shape[-1])]
                        means = [a.mean() if len(a) else 0 for a in values]
                        stds = [max(a.std(),1e-5) if len(a) else 1e-5 for a in values]
                    check(f'{domain}_{fold}_{arm}_{key}_mean',np.allclose(means,stored['mean'],atol=1e-10,rtol=2e-8))
                    check(f'{domain}_{fold}_{arm}_{key}_sd',np.allclose(stds,stored['std'],atol=1e-10,rtol=2e-8))
                with np.load(path/'test_predictions.npz') as a:
                    ids = a['ids'].astype(str).tolist()
                    ids_seen[arm] += ids
                    for j,uid in enumerate(ids):
                        table = cells[(cells.domain==domain)&(cells.arm==arm)&(cells.cell_uid==uid)]
                        for r in table.itertuples():
                            if r.window in ('pre_eol80','post_eol80'):
                                continue
                            lo,hi = {'all':(101,np.inf),'101_200':(101,200),'201_500':(201,500),
                                     '501_1000':(501,1000),'1001_plus':(1001,np.inf)}[r.window]
                            m = a['observed'][j] & (a['query']>=lo)&(a['query']<=hi)
                            e = a['prediction'][j,m].astype(float)-a['target'][j,m].astype(float)
                            check(f'{domain}_{fold}_{arm}_{uid}_{r.window}_support',int(m.sum())==r.n_obs)
                            if len(e):
                                check(f'{domain}_{fold}_{arm}_{uid}_{r.window}_mae',np.isclose(np.abs(e).mean(),r.mae,atol=1e-14,rtol=1e-11))
                                check(f'{domain}_{fold}_{arm}_{uid}_{r.window}_rmse',np.isclose(np.sqrt(np.mean(e**2)),r.rmse,atol=1e-14,rtol=1e-11))
                h = json.loads((path/'HISTORY.json').read_text())
                contract = json.loads((path/'UNIT_CONTRACT.json').read_text())
                ntrain = len(roles['train'])
                check(f'{domain}_{fold}_{arm}_update_history',all(r['updates']==r['epoch']*int(np.ceil(ntrain/8)) for r in h))
                check(f'{domain}_{fold}_{arm}_epoch_budget',len(h)==150 and h[-1]['updates']==contract['expected_updates'])
            cap = json.loads((OUT/domain/fold/'D1-0-capmatch/UNIT_CONTRACT.json').read_text())['extra_params']
            full = json.loads((OUT/domain/fold/'D1-GL/UNIT_CONTRACT.json').read_text())['extra_params']
            check(f'{domain}_{fold}_capacity_match',abs(cap-full)/full<.01)
        for arm,ids in ids_seen.items():
            check(f'{domain}_{arm}_OOF_unique',len(ids)==len(b.ids) and set(ids)==set(b.ids))
        lookup = {uid:i for i,uid in enumerate(b.ids)}
        for path in sorted((OUT/'ORACLE_DIAGNOSTIC'/domain).glob('**/STARTS.json')):
            log = json.loads(path.read_text())
            i = lookup[log['cell_uid']]
            m = b.observed[i]
            label = path.parent.relative_to(OUT).as_posix()
            with np.load(path.parent/'predictions.npz') as a:
                check(label+'_query',np.array_equal(a['query'],b.query[m]))
                check(label+'_target',np.array_equal(a['target'],b.targets[i,m]))
                check(label+'_anchor',float(a['anchor'])==float(b.anchor[i]))
                check(label+'_eight_starts',len(log['starts'])==8 and a['prediction'].shape==(8,int(m.sum())))
                check(label+'_finite',np.isfinite(a['prediction']).all())
                for j,r in enumerate(log['starts']):
                    loss=float(np.abs(a['prediction'][j]-a['target']).mean())
                    check(label+f'_start_{j}_loss',abs(loss-r['mae'])<1e-13)
                    calculated=evaluate(log['family'],r['raw'],(a['query'].astype(float)-100)/1000,
                                        float(b.anchor[i]),log['shared_values'])[0]
                    check(label+f'_start_{j}_function',np.allclose(calculated,a['prediction'][j],atol=1e-13,rtol=1e-13))
            check(label+'_best',log['best_start']==min(log['starts'],key=lambda r:r['mae'])['start'])
        for fold in settings()['folds']:
            shared=json.loads((OUT/'ORACLE_DIAGNOSTIC'/domain/'shared'/fold/'COMPLETE.json').read_text())
            check(f'{domain}_{fold}_shared_train_only',shared['training_ids']==b.folds[fold]['train'])
        models = json.loads((OUT/'D4'/domain/'RATE_PROXY_MODELS.json').read_text())
        check(f'{domain}_rate_proxy_units',len(models)==80)
        for j,r in enumerate(models):
            rr = b.folds[r['fold']]
            check(f'{domain}_proxy_{j}_fold_seal',set(r['training_ids'])<=set(rr['train']) and set(r['validation_ids'])<=set(rr['validation']))
    for r in summary.itertuples():
        g = cells[(cells.domain==r.domain)&(cells.arm==r.arm)&(cells.window==r.window)&(cells.n_obs>0)]
        check(f'{r.domain}_{r.arm}_{r.window}_summary_support',len(g)==r.supported_cells and g.n_obs.sum()==r.observations)
        if len(g):
            check(f'{r.domain}_{r.arm}_{r.window}_summary_MAE',np.isclose(g.mae.mean(),r.cell_mean_mae,atol=1e-14))
            check(f'{r.domain}_{r.arm}_{r.window}_protocol_macro',np.isclose(g.groupby('protocol_uid').mae.mean().mean(),r.protocol_macro_mae,atol=1e-14))
    for path in sorted(OUT.glob('**/*.csv')):
        frame = pd.read_csv(path)
        check('table_scope_'+path.relative_to(OUT).as_posix(),'prediction_scope' in frame)
        check('scope_values_'+path.relative_to(OUT).as_posix(),set(frame.prediction_scope.dropna())<={SCOPE,'ORACLE_DIAGNOSTIC','FUTURE_PREFIX_DIAGNOSTIC'})
    write_json(OUT/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',checks=len(checks),passed=checks,
        source_registration_sha256=file_hash(OUT/'SOURCE_REGISTRATION.json')))
    print('INDEPENDENT_VERIFICATION_PASS',len(checks))


if __name__=='__main__':
    main()
