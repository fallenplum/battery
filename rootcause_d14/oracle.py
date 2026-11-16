"""Future-label diagnostics. This module has no legal-training entry point."""
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from n012.data import role_indices,file_hash
from n012.run import write_json,file_records,verify_files
from .data import ROOT,OUT,ORACLE,load_domain,settings
from .run import check_sources


def softplus(x):
    return np.logaddexp(0,x)


def durations(u, knots):
    left,right = np.asarray(knots[:-1]),np.asarray(knots[1:])
    result = np.minimum(np.maximum(u[:,None]-left,0),right-left)
    result[:,-1] += np.maximum(u-knots[-1],0)
    return result


def physics_values(raw, shared=None):
    raw = np.asarray(raw)
    vals = [softplus(raw[0])*.05,softplus(raw[1])*.05,softplus(raw[2])*.06,4*expit(raw[3])]
    vals += [softplus(raw[4])+.05,softplus(raw[5])+.03] if shared is None else list(shared)
    return np.asarray(vals,float)


def evaluate(family,raw,u,anchor,shared=None):
    """Prediction and analytic Jacobian in optimizer coordinates, all float64."""
    raw,u = np.asarray(raw,float),np.asarray(u,float)
    if family.startswith('physics'):
        a,b,c,tau,tc,w = physics_values(raw,shared)
        v,v0 = (u-tau)/w,-tau/w
        diff = softplus(v)-softplus(v0)
        cumulative = a*np.log1p(u/tc)+b*u+c*w*diff
        pred = anchor*np.exp(-cumulative)
        derivatives = [np.log1p(u/tc),u,w*diff,c*(-expit(v)+expit(v0))]
        chain = [expit(raw[0])*.05,expit(raw[1])*.05,expit(raw[2])*.06,4*expit(raw[3])*(1-expit(raw[3]))]
        if shared is None:
            derivatives += [-a*u/(tc*(tc+u)),c*(diff-v*expit(v)+v0*expit(v0))]
            chain += [expit(raw[4]),expit(raw[5])]
        jac = -pred[:,None]*np.column_stack(derivatives)*np.array(chain)[None]
    elif family in ('spline10','positive_rate8'):
        key = 'spline_knots_u' if family == 'spline10' else 'positive_rate_knots_u'
        mat = durations(u,settings()['oracle'][key])
        rates = softplus(raw)*.15
        cum = mat@rates
        if family == 'spline10':
            pred = anchor*np.exp(-cum)
            jac = -pred[:,None]*mat*(expit(raw)*.15)[None]
        else:
            pred = anchor-cum
            jac = -mat*(expit(raw)*.15)[None]
    elif family == 'polynomial4':
        mat = np.column_stack([u**j for j in range(1,5)])
        pred,jac = anchor+mat@raw,mat
    elif family == 'linear1':
        pred = anchor-softplus(raw[0])*.15*u
        jac = (-expit(raw[0])*.15*u)[:,None]
    else:
        raise ValueError(family)
    return pred,jac


def starts(family,seed=42,shared=None):
    rng = np.random.default_rng(seed)
    dim = {'physics_free':6,'physics_shared':4,'spline10':10,'positive_rate8':8,'polynomial4':4,'linear1':1}[family]
    values = rng.normal(-1,1.3,(8,dim))
    values[0] = 0 if family.startswith('physics') else -.8
    if family.startswith('physics'):
        values[:,3] = np.array([-2.7,-1.95,-1.1,0,1.1,2,3,-.5])
        if dim == 6:
            values[:,4] = -.8+rng.normal(0,.8,8)
            values[:,5] = -1.5+rng.normal(0,.6,8)
            values[0,4:] = [-.8,-1.5]
    if family == 'polynomial4':
        values *= .02
        values[0] = 0
    return values


def objective(family,u,anchor,y,shared=None):
    def loss(raw):
        pred,jac = evaluate(family,raw,u,anchor,shared)
        error = pred-y
        return np.mean(np.abs(error)), np.mean(np.sign(error)[:,None]*jac,axis=0)
    return loss


def fit_cell(family,query,y,anchor,shared=None,seed=42):
    c = settings()['oracle']
    u = (query.astype(float)-100)/1000
    init = starts(family,seed,shared)
    results, predictions = [],[]
    fn = objective(family,u,anchor,y.astype(float),shared)
    for j,initial in enumerate(init):
        result = minimize(fn,initial,jac=True,method='L-BFGS-B',
                          options=dict(maxiter=c['maxiter'],ftol=c['ftol'],gtol=c['gtol'],maxls=40))
        pred,grad = evaluate(family,result.x,u,anchor,shared)
        assert np.isfinite(pred).all() and np.isfinite(result.x).all()
        mae = float(np.abs(pred-y).mean())
        params = physics_values(result.x,shared).tolist() if family.startswith('physics') else result.x.tolist()
        results.append(dict(start=j,initial=initial.tolist(),raw=result.x.tolist(),parameters=params,
            mae=mae,converged=bool(result.success),status=int(result.status),message=str(result.message),
            iterations=int(result.nit),evaluations=int(result.nfev),gradient_inf=float(np.max(np.abs(result.jac))),
            parameter_boundary='softplus-positive,tau(0,4),time_c>.05,smooth>.03' if family.startswith('physics')
                               else 'positive softplus rates' if family != 'polynomial4' else 'unconstrained finite degree 4'))
        predictions.append(pred)
    return results,np.stack(predictions)


def shared_fit(d,fold):
    """Joint per-cell a,b,c,tau and two train-only shared raw parameters."""
    dest = OUT/'ORACLE_DIAGNOSTIC'/d.bundle.domain/'shared'/fold
    marker = dest/'COMPLETE.json'
    if marker.exists():
        receipt = json.loads(marker.read_text(encoding='utf-8'))
        verify_files(dest,receipt)
        return receipt['shared_values']
    b = d.bundle
    rows = role_indices(b,fold)['train']
    cases = [(np.asarray((b.query[b.observed[i]]-100)/1000,float),float(b.anchor[i]),
              np.asarray(b.targets[i,b.observed[i]],float)) for i in rows]
    n = len(rows)
    def fn(raw):
        loss,g = 0.,np.zeros_like(raw)
        for j,(u,anchor,y) in enumerate(cases):
            cellraw = np.r_[raw[4*j:4*j+4],raw[-2:]]
            pred,jac = evaluate('physics_free',cellraw,u,anchor)
            error = pred-y
            loss += np.mean(np.abs(error))/n
            grad = np.mean(np.sign(error)[:,None]*jac,0)/n
            g[4*j:4*j+4] = grad[:4]
            g[-2:] += grad[4:]
        return loss,g
    logs = []
    for j in range(8):
        initial = np.r_[np.tile(starts('physics_free')[j,:4],n),starts('physics_free')[j,4:]]
        fit = minimize(fn,initial,jac=True,method='L-BFGS-B',options=dict(
            maxiter=settings()['oracle']['shared_train_maxiter'],ftol=1e-11,gtol=1e-7,maxls=30))
        logs.append(dict(prediction_scope=ORACLE,start=j,initial=initial.tolist(),raw=fit.x.tolist(),
            mae=float(fit.fun),converged=bool(fit.success),status=int(fit.status),iterations=int(fit.nit),message=str(fit.message)))
    best = min(logs,key=lambda r:r['mae'])
    values = [float(softplus(best['raw'][-2])+.05),float(softplus(best['raw'][-1])+.03)]
    write_json(dest/'JOINT_STARTS.json',dict(training_ids=[b.ids[i] for i in rows],starts=logs))
    write_json(marker,dict(prediction_scope=ORACLE,status='COMPLETE',shared_values=values,
        training_ids=[b.ids[i] for i in rows],optimizer_starts=8,
        files=file_records(dest,['JOINT_STARTS.json'])))
    print(json.dumps(dict(event='ORACLE_SHARED',domain=b.domain,fold=fold,shared=values)),flush=True)
    return values


def run(domain,phase):
    check_sources()
    d = load_domain(domain)
    b = d.bundle
    root = OUT/'ORACLE_DIAGNOSTIC'/domain
    root.mkdir(parents=True,exist_ok=True)
    for fold in settings()['folds']:
        shared = shared_fit(d,fold)
        roles = role_indices(b,fold)
        allowed = ['train','validation'] if phase == 'development' else ['test']
        for role in allowed:
            for i in roles[role]:
                for family in settings()['oracle']['families']:
                    binding = fold if family == 'physics_shared' else 'individual'
                    dest = root/binding/family/b.ids[i]
                    if (dest/'COMPLETE.json').exists():
                        verify_files(dest,json.loads((dest/'COMPLETE.json').read_text(encoding='utf-8')))
                        continue
                    m = b.observed[i]
                    q,y = b.query[m],b.targets[i,m]
                    logs,pred = fit_cell(family,q,y,float(b.anchor[i]),shared if family=='physics_shared' else None)
                    write_json(dest/'STARTS.json',dict(prediction_scope=ORACLE,domain=domain,cell_uid=b.ids[i],
                        family=family,fold_binding=binding,parameter_count=len(logs[0]['raw']),
                        extra_individual_shared_parameters=family=='physics_free',starts=logs,
                        best_start=min(logs,key=lambda r:r['mae'])['start'],first_access_role=role,first_access_fold=fold,
                        first_access_phase=phase,shared_values=shared if family=='physics_shared' else None))
                    np.savez_compressed(dest/'predictions.npz',query=q,target=y,anchor=np.array(b.anchor[i]),prediction=pred)
                    write_json(dest/'COMPLETE.json',dict(prediction_scope=ORACLE,status='COMPLETE',optimizer_starts=8,
                        files=file_records(dest,['STARTS.json','predictions.npz'])))
            print(json.dumps(dict(event='ORACLE_ROLE',domain=domain,fold=fold,role=role)),flush=True)
    bindings = [dict(prediction_scope=ORACLE,domain=domain,fold=fold,role=role,cell_uid=b.ids[i],family=family,
                     path=(root/(fold if family=='physics_shared' else 'individual')/family/b.ids[i]).relative_to(OUT).as_posix())
                for fold in settings()['folds'] for role,rows in role_indices(b,fold).items()
                if role in (['train','validation'] if phase=='development' else ['test'])
                for i in rows for family in settings()['oracle']['families']]
    write_json(root/(phase.upper()+'_BINDINGS.json'),bindings)
    write_json(root/(phase.upper()+'_COMPLETE.json'),dict(status='COMPLETE',prediction_scope=ORACLE,
        binding_count=len(bindings),source_registration_sha256=file_hash(OUT/'SOURCE_REGISTRATION.json'),
        files=file_records(root,[phase.upper()+'_BINDINGS.json'])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain',choices=['MATR','HUST'],required=True)
    p.add_argument('--phase',choices=['development','test'],required=True)
    a = p.parse_args()
    run(a.domain,a.phase)


if __name__ == '__main__':
    main()
