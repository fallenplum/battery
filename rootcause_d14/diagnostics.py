"""D4 parameter stability, profiles, and split-isolated future-rate probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from n012.data import role_indices,file_hash
from n012.run import configure,write_json,file_records
from .data import ROOT,OUT,ORACLE,SCOPE,load_domain,settings,masked_statistics,scaled_inputs,selected_encoder
from .oracle import evaluate,objective,physics_values,softplus
from .model import RootCauseNet
from .run import check_sources,predict


def parameter_diagnostics(domain):
    d = load_domain(domain)
    b = d.bundle
    root = OUT/'D4'/domain
    root.mkdir(parents=True,exist_ok=True)
    rows,svds,profiles,sensitivity = [],[],[],[]
    c = settings()['oracle']
    for fold in settings()['folds']:
        for i in role_indices(b,fold)['test']:
            for family in ('physics_free','physics_shared'):
                path = OUT/'ORACLE_DIAGNOSTIC'/domain/(fold if family=='physics_shared' else 'individual')/family/b.ids[i]
                log = json.loads((path/'STARTS.json').read_text(encoding='utf-8'))
                runs = log['starts']
                best = min(runs,key=lambda r:r['mae'])
                bestloss = best['mae']
                equivalent = [r for r in runs if r['mae']<=bestloss*1.05+1e-15 and r['mae']-bestloss<=.002+1e-15]
                values = np.array([r['parameters'] for r in equivalent])
                last = float(b.query[b.observed[i]].max())
                base = dict(prediction_scope=ORACLE,domain=domain,fold=fold,cell_uid=b.ids[i],family=family,
                            best_mae=bestloss,equivalent_starts=len(equivalent),last_cycle=last,
                            followup_group='1000_or_less' if last<=1000 else 'over_1000')
                for j,name in enumerate(('a','b','acceleration_c','tau','time_c','smooth')):
                    rows.append(dict(**base,parameter=name,minimum=float(values[:,j].min()),
                                     maximum=float(values[:,j].max()),range=float(np.ptp(values[:,j]))))
                for threshold in c['sensitivity_relative']:
                    eq = [r for r in runs if r['mae']<=bestloss*(1+threshold)+1e-15 and r['mae']-bestloss<=.002+1e-15]
                    sensitivity.append(dict(**base,relative_threshold=threshold,absolute_threshold=.002,
                        supported_starts=len(eq),tau_range=float(np.ptp([r['parameters'][3] for r in eq]))))
                q = np.asarray(b.query[b.observed[i]],float)
                raw = np.array(best['raw'])
                _,jac = evaluate(family,raw,(q-100)/1000,float(b.anchor[i]),log['shared_values'])
                chain = np.r_[expit(raw[:3])*np.array([.05,.05,.06]),4*expit(raw[3])*(1-expit(raw[3]))]
                if family == 'physics_free':
                    chain = np.r_[chain,expit(raw[4:])]
                # Convert optimizer derivatives to interpretable physical parameter coordinates.
                physical_jac = jac/np.maximum(chain[None],1e-30)
                scaled = physical_jac*np.array(c['svd_scaled_parameter_step'][:len(raw)])[None]
                s = np.linalg.svd(scaled,compute_uv=False)
                svds.append(dict(**base,singular_values=json.dumps(s.tolist()),max_sv=float(s[0]),min_sv=float(s[-1]),
                    condition=float(s[0]/max(s[-1],1e-300)),near_zero_directions=int((s<s[0]*c['jacobian_near_zero_relative']).sum()),
                    coordinate='physical_parameter_scaled',individual_parameters=len(raw),
                    shared_parameters=0 if family=='physics_free' else 2))
    # The profile subset is selected by UID and role before looking at losses.
    fold = 'cell_fold_0'
    role = role_indices(b,fold)
    profile_rows = [(r,i) for r in ('train','test') for i in sorted(role[r],key=lambda j:b.ids[j])[:3]]
    for role_name,i in profile_rows:
        path = OUT/'ORACLE_DIAGNOSTIC'/domain/'individual/physics_free'/b.ids[i]/'STARTS.json'
        logs = json.loads(path.read_text(encoding='utf-8'))
        best = min(logs['starts'],key=lambda r:r['mae'])
        init = np.array(best['raw'])
        m = b.observed[i]
        u,y,anchor = (b.query[m].astype(float)-100)/1000,b.targets[i,m].astype(float),float(b.anchor[i])
        fn = objective('physics_free',u,anchor,y)
        for kind,grid in [('tau',c['tau_profile_u']),('acceleration_c',c['amplitude_profile']),
                          ('rate200_500',c['rate_profile_scaled']),('rate500_1000',c['rate_profile_scaled'])]:
            for value in grid:
                if kind in ('tau','acceleration_c'):
                    idx = 3 if kind=='tau' else 2
                    fixed = np.log(value/(4-value)) if idx==3 else np.log(np.expm1(value/.06))
                    keep = [j for j in range(6) if j!=idx]
                    def reduced(v):
                        full = init.copy()
                        full[keep],full[idx] = v,fixed
                        loss,grad = fn(full)
                        return loss,grad[keep]
                    fit = minimize(reduced,init[keep],jac=True,method='L-BFGS-B',options=dict(maxiter=500,maxls=40,ftol=1e-12))
                    actual = value
                else:
                    left,right = (200,500) if kind=='rate200_500' else (500,1000)
                    uq = (np.array([left,right],float)-100)/1000
                    def equality(v):
                        p,j = evaluate('physics_free',v,uq,anchor)
                        return (p[0]-p[1])/(right-left)*1000-value
                    def equality_jac(v):
                        p,j = evaluate('physics_free',v,uq,anchor)
                        return (j[0]-j[1])/(right-left)*1000
                    with np.errstate(over='ignore',invalid='ignore'):
                        fit = minimize(fn,init,jac=True,method='SLSQP',constraints=[dict(type='eq',fun=equality,jac=equality_jac)],
                                       options=dict(maxiter=500,ftol=1e-10))
                    actual = float(equality(fit.x)+value)
                profiles.append(dict(prediction_scope=ORACLE,domain=domain,fold=fold,role=role_name,cell_uid=b.ids[i],
                    parameter=kind,fixed_value=value,achieved_value=actual,mae=float(fit.fun),best_unconstrained_mae=best['mae'],
                    converged=bool(fit.success),status=int(fit.status),iterations=int(fit.nit)))
    pd.DataFrame(rows).to_csv(root/'parameter_stability.csv',index=False)
    pd.DataFrame(svds).to_csv(root/'jacobian_svd.csv',index=False)
    pd.DataFrame(profiles).to_csv(root/'profiles.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(root/'threshold_sensitivity.csv',index=False)


def proxy_feature_groups(d):
    b = d.bundle
    base = np.concatenate((b.anchor[:,None],np.where(b.policy_mask,b.policy,0),b.policy_mask),1)
    m = b.mask[...,None]
    percycle = np.divide(np.where(m,b.x,0).sum(2),m.sum(2),out=np.zeros(b.x.shape[:2]+(4,),float),where=m.sum(2)>0)
    cm = np.broadcast_to(b.mask.any(2)[...,None],percycle.shape)
    raw = masked_statistics(percycle,cm)
    global_stats = masked_statistics(d.arrays['global_features'],d.arrays['global_mask'])
    local_stats = np.concatenate((masked_statistics(d.arrays['local_residual'],d.arrays['local_mask']),
                                  masked_statistics(d.arrays['local_features'],d.arrays['local_feature_mask'])),1)
    common = np.concatenate((base,raw),1)
    return {'anchor_policy':base,'raw_statistics':common,'plus_G':np.c_[common,global_stats],
            'plus_L':np.c_[common,local_stats],'plus_GL':np.c_[common,global_stats,local_stats]}


def fit_proxy(features,target,mask,roles,alphas):
    tr,va,te = [roles[r][mask[roles[r]]] for r in ('train','validation','test')]
    if not len(tr) or not len(va):
        raise ValueError('A frozen rate proxy has no train/validation support')
    scaler = StandardScaler().fit(features[tr])
    xtr,xva,xte = [scaler.transform(features[r]) for r in (tr,va,te)]
    fits = []
    for alpha in alphas:
        reg = Ridge(alpha=alpha,solver='svd').fit(xtr,target[tr]*1000)
        loss = mean_absolute_error(target[va]*1000,reg.predict(xva))
        fits.append((float(loss),alpha,reg))
    loss,alpha,reg = min(fits,key=lambda v:v[0])
    pred = reg.predict(xte)/1000
    return te,pred,dict(alpha=alpha,validation_mae_scaled=loss,training_rows=tr.tolist(),
        validation_rows=va.tolist(),candidate_validation_mae=[dict(alpha=a,mae=v) for v,a,r in fits],
        mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=reg.coef_.tolist(),intercept=float(reg.intercept_))


def proxies(domain,device='cuda'):
    d = load_domain(domain)
    b = d.bundle
    root = OUT/'D4'/domain
    root.mkdir(parents=True,exist_ok=True)
    base_groups = proxy_feature_groups(d)
    rows,contracts = [],[]
    for fold in settings()['folds']:
        roles,x,p,phy,_ = scaled_inputs(d,fold)
        groups = dict(base_groups)
        for arm in settings()['proxy']['z_arms']:
            spec = settings()['arms'][arm]
            model = RootCauseNet(selected_encoder(domain,fold),spec['decoder'],b.policy.shape[1],spec['features']).to(device)
            path = OUT/domain/fold/arm/'BEST.pt'
            model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model'])
            _,z = predict(model,d,x,p,phy,np.arange(len(b.ids)),device,return_z=True)
            groups['frozen_z_'+arm] = np.c_[base_groups['anchor_policy'],z]
            del model
        for group,features in groups.items():
            for window in range(2):
                te,pred,contract = fit_proxy(features,d.rates[:,window],d.rate_mask[:,window],roles,settings()['proxy']['alphas'])
                contracts.append(dict(prediction_scope=SCOPE,domain=domain,fold=fold,feature_set=group,window=window,
                    dimensions=features.shape[1],training_ids=[b.ids[i] for i in contract['training_rows']],
                    validation_ids=[b.ids[i] for i in contract['validation_rows']],**contract))
                for i,pr in zip(te,pred):
                    rows.append(dict(prediction_scope=SCOPE,diagnostic_target='future_rate',domain=domain,fold=fold,cell_uid=b.ids[i],
                        feature_set=group,window=['200_500','500_1000'][window],target=float(d.rates[i,window]),
                        prediction=float(pr),selected_alpha=contract['alpha']))
        print(json.dumps(dict(event='PROXY',domain=domain,fold=fold)),flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'rate_proxy_predictions.csv',index=False)
    summaries = []
    for (group,window),g in frame.groupby(['feature_set','window']):
        y,pred = g.target.to_numpy(),g.prediction.to_numpy()
        baseline = frame[(frame.feature_set=='anchor_policy')&(frame.window==window)].set_index('cell_uid')
        other = g.set_index('cell_uid').loc[baseline.index]
        increment = np.abs(baseline.prediction-baseline.target).to_numpy()-np.abs(other.prediction-other.target).to_numpy()
        summaries.append(dict(prediction_scope=SCOPE,domain=domain,feature_set=group,window=window,n_cells=len(g),
            mae=float(mean_absolute_error(y,pred)),rmse=float(mean_squared_error(y,pred)**.5),r2=float(r2_score(y,pred)),
            increment_over_anchor_policy=float(np.mean(increment))))
    pd.DataFrame(summaries).to_csv(root/'rate_proxy_summary.csv',index=False)
    write_json(root/'RATE_PROXY_MODELS.json',contracts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain',choices=['MATR','HUST'],required=True)
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    configure()
    check_sources()
    parameter_diagnostics(a.domain)
    proxies(a.domain,a.device)
    root = OUT/'D4'/a.domain
    write_json(root/'COMPLETE.json',dict(status='COMPLETE',prediction_scope=ORACLE,
        scopes=[ORACLE,SCOPE],source_sha256=file_hash(OUT/'SOURCE_REGISTRATION.json'),
        files=file_records(root,[p.name for p in root.iterdir() if p.suffix in ('.csv','.json') and p.name!='COMPLETE.json'])))


if __name__ == '__main__':
    main()
