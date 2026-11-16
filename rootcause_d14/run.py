"""Legal K100 training only. Oracle outputs are never imported here."""
import argparse
import copy
import json
import math
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from n012.run import (configure,seed_all,random_state,restore_random,cell_mae,metrics,
                      write_json,save_torch,bind_json,file_records,verify_files,execution_lock,batches)
from n012.data import file_hash
from .data import ROOT,OUT,SCOPE,settings,load_domain,scaled_inputs,selected_encoder,CONFIG
from .model import RootCauseNet


def source_records():
    paths = [ROOT/'rootcause_d14'/name for name in ('__init__.py','data.py','model.py','run.py','oracle.py','diagnostics.py')]
    paths += [CONFIG,ROOT/'tests/test_d14.py',ROOT/'research/D1_D4_EXECUTION_CONTRACT_ZH.md']
    paths += [ROOT/'n012'/p for p in ('data.py','model.py','run.py')]
    return [dict(path=p.relative_to(ROOT).as_posix(),sha256=file_hash(p)) for p in paths]


def register_sources():
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    registration = dict(source_commit=commit,sources=source_records(),
                        config_sha256=file_hash(CONFIG),registration_sha256=file_hash(OUT/'RUN_REGISTRATION.json'))
    bind_json(OUT/'SOURCE_REGISTRATION.json',registration)
    return file_hash(OUT/'SOURCE_REGISTRATION.json')


def check_sources():
    r = json.loads((OUT/'SOURCE_REGISTRATION.json').read_text(encoding='utf-8'))
    assert r['sources'] == source_records(), 'Frozen executable source changed'
    assert r['config_sha256'] == file_hash(CONFIG)
    assert r['registration_sha256'] == file_hash(OUT/'RUN_REGISTRATION.json')
    return file_hash(OUT/'SOURCE_REGISTRATION.json')


def batch(d,x,p,physics,rows,device):
    b = d.bundle
    return dict(x=torch.as_tensor(x[rows],device=device),
        point_mask=torch.as_tensor(b.mask[rows],device=device),cycle_ids=torch.as_tensor(b.cycle[rows],device=device),
        policy=torch.as_tensor(p[rows],device=device),policy_mask=torch.as_tensor(b.policy_mask[rows],device=device),
        anchor=torch.as_tensor(b.anchor[rows],device=device),query=torch.as_tensor(b.query,device=device),
        physics={k:torch.as_tensor(v[rows],device=device) for k,v in physics.items()})


def rate_loss(pred, targets, supported, query):
    positions = {int(v):j for j,v in enumerate(query)}
    rates = torch.stack(((pred[:,positions[200]]-pred[:,positions[500]])/300,
                         (pred[:,positions[500]]-pred[:,positions[1000]])/500),1)
    n = supported.sum(1)
    error = torch.where(supported,(rates-targets).abs(),0.)
    values = error.sum(1)/n.clamp_min(1)
    return values[n>0].mean()*1000 if bool((n>0).any()) else pred.sum()*0


def losses(pred,d,rows):
    device = pred.device
    traj = cell_mae(pred,torch.as_tensor(d.bundle.targets[rows],device=device),
                    torch.as_tensor(d.bundle.observed[rows],device=device))
    aux = rate_loss(pred,torch.as_tensor(d.rates[rows],device=device),
                    torch.as_tensor(d.rate_mask[rows],device=device),d.bundle.query)
    return traj,aux


def predict(model,d,x,p,phy,rows,device,return_z=False):
    model.eval()
    pred,latents = [],[]
    with torch.no_grad():
        for start in range(0,len(rows),8):
            result = model(**batch(d,x,p,phy,rows[start:start+8],device),return_z=return_z)
            if return_z:
                result,z = result
                latents.append(z.cpu().numpy())
            pred.append(result.cpu().numpy())
    result = np.concatenate(pred)
    return (result,np.concatenate(latents)) if return_z else result


def gradient_probe(model,d,x,p,phy,rows,device,epoch):
    """Fixed TRAIN minibatch; eval mode and RNG state restored without updates."""
    rng = random_state()
    was_training = model.training
    model.eval()
    rows = rows[:8]
    inp = batch(d,x,p,phy,rows,device)
    pred,z = model(**inp,return_z=True)
    main,aux = losses(pred,d,rows)
    params = [v for v in model.parameters() if v.requires_grad]
    g1 = torch.autograd.grad(main,params,retain_graph=True,allow_unused=True)
    g2 = torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
    dot = sum((a*b).sum() for a,b in zip(g1,g2) if a is not None and b is not None)
    norm1 = sum((a*a).sum() for a in g1 if a is not None).sqrt()
    norm2 = sum((a*a).sum() for a in g2 if a is not None).sqrt()
    record = dict(epoch=epoch,gradient_cosine=float((dot/(norm1*norm2).clamp_min(1e-30)).detach().cpu()),
                  main_grad_norm=float(norm1.detach().cpu()),rate_grad_norm=float(norm2.detach().cpu()),
                  train_probe_ids=[d.bundle.ids[i] for i in rows])
    if model.base.decoder_name == 'physics_basis':
        head = model.base.decoder
        raw = head.net(z)
        # Retain a distinct raw path to measure dL/draw, using the actual head formula.
        a,b,c = F.softplus(raw[:,0:1])*.05,F.softplus(raw[:,1:2])*.05,F.softplus(raw[:,2:3])*.06
        tau = 4*torch.sigmoid(raw[:,3:4])
        tc,w = F.softplus(head.log_c)+.05,F.softplus(head.log_w)+.03
        u = (inp['query'][None]-100)/1000
        accel = c*w*(F.softplus((u-tau)/w)-F.softplus(-tau/w))
        total = a*torch.log1p(u/tc)+b*u+accel
        out = inp['anchor'][:,None]*torch.exp(-total)
        lt,lr = losses(out,d,rows)
        rawgrad = torch.autograd.grad(lt+.1*lr,raw)[0]
        arrays = dict(raw=raw.detach().cpu().numpy(),a=a.detach().cpu().numpy(),b=b.detach().cpu().numpy(),
                      c=c.detach().cpu().numpy(),tau=tau.detach().cpu().numpy(),raw_gradient=rawgrad.detach().cpu().numpy())
        record.update(time_c=float(tc.detach().cpu()),smooth=float(w.detach().cpu()),
                      quantiles={k:np.quantile(v,[0,.25,.5,.75,1],axis=0).tolist() for k,v in arrays.items()},
                      histograms={k:dict(counts=np.histogram(v,bins=10)[0].tolist(),edges=np.histogram(v,bins=10)[1].tolist())
                                  for k,v in arrays.items()},
                      values={k:v.tolist() for k,v in arrays.items()})
        last = np.array([d.bundle.query[d.bundle.observed[i]].max() for i in rows])
        absolute = 100+1000*tau.detach().cpu().numpy().ravel()
        record.update(last_observed_cycle=last.tolist(),absolute_tau=absolute.tolist(),
                      tau_within_followup=(absolute<=last).tolist(),acceleration_fraction={})
        ratio = (accel/total.clamp_min(1e-15)).detach().cpu().numpy()
        for lo,hi in ((101,200),(201,500),(501,1000),(1001,10000)):
            m = d.bundle.observed[rows] & (d.bundle.query>=lo) & (d.bundle.query<=hi)
            record['acceleration_fraction'][f'{lo}_{hi}'] = float(ratio[m].mean()) if m.any() else None
    model.train(was_training)
    restore_random(rng)
    return record


def train_unit(d,fold,arm,spec,output=OUT,epochs=150,device='cuda',stop_after=None,source_sha=None):
    c = settings()
    b = d.bundle
    dest = Path(output)/b.domain/fold/arm
    dest.mkdir(parents=True,exist_ok=True)
    roles,x,p,phy,scaler = scaled_inputs(d,fold)
    enc = selected_encoder(b.domain,fold)
    seed_all(42)
    model = RootCauseNet(enc,spec['decoder'],b.policy.shape[1],spec['features']).to(device)
    parameter_count = sum(v.numel() for v in model.parameters())
    extra_count = sum(v.numel() for v in model.branch.parameters()) if hasattr(model,'branch') else 0
    contract = dict(prediction_scope=SCOPE,domain=b.domain,fold=fold,arm=arm,encoder=enc,**spec,
        seed=42,epochs=epochs,batch_size=8,learning_rate=c['learning_rate'],weight_decay=.01,clip=1.,
        parameters=parameter_count,extra_params=extra_count,
        roles={r:[b.ids[i] for i in ix] for r,ix in roles.items()},source_sha256=source_sha,
        config_sha256=file_hash(CONFIG),data_audit_sha256=file_hash(OUT/'DATA_AUDIT.json'),
        expected_updates=epochs*math.ceil(len(roles['train'])/8))
    bind_json(dest/'UNIT_CONTRACT.json',contract)
    bind_json(dest/'SCALER_TRAIN_ONLY.json',scaler)
    if (dest/'TRAINING_COMPLETE.json').exists():
        receipt = json.loads((dest/'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
        verify_files(dest,receipt)
        return receipt
    optimizer = torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=.01)
    history, probes, best, selected, updates, start_epoch, elapsed = [],[],math.inf,None,0,0,0.
    state_path = dest/'TRAIN_STATE.pt'
    if state_path.exists():
        state = torch.load(state_path,map_location=device,weights_only=False)
        assert state['contract'] == contract
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        restore_random(state['random_state'])
        history,probes,best,selected,updates,start_epoch,elapsed = (
            state[k] for k in ('history','probes','best_mae','best_checkpoint','updates','next_epoch','elapsed_seconds'))
    started = time.monotonic()
    for epoch in range(start_epoch,epochs):
        model.train()
        main_total,aux_total,grad_norms, supported_cells = 0.,0.,[],0
        for rows in batches(roles['train'],8,42,epoch):
            optimizer.zero_grad(set_to_none=True)
            pred = model(**batch(d,x,p,phy,rows,device))
            main,aux = losses(pred,d,rows)
            loss = main+spec['rate_weight']*aux
            assert bool(torch.isfinite(loss)), 'Nonfinite training objective'
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            main_total += float(main.detach().cpu())*len(rows)
            aux_total += float(aux.detach().cpu())*len(rows)
            grad_norms.append(float(gn.detach().cpu()))
            supported_cells += int(d.rate_mask[rows].any(1).sum())
        valpred = predict(model,d,x,p,phy,roles['validation'],device)
        val = metrics(valpred,b.targets[roles['validation']],b.observed[roles['validation']],b.query)['mae']
        is_best = val < best
        if is_best:
            best = val
            selected = dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            contract=contract,best_epoch=epoch+1,validation_mae=best)
        probes.append(gradient_probe(model,d,x,p,phy,roles['train'],device,epoch+1))
        history.append(dict(prediction_scope=SCOPE,epoch=epoch+1,train_mae=main_total/len(roles['train']),
            train_rate_scaled=aux_total/len(roles['train']),weighted_rate_term=spec['rate_weight']*aux_total/len(roles['train']),
            val_mae=val,selected=is_best,updates=updates,learning_rate=optimizer.param_groups[0]['lr'],
            grad_norm_mean=float(np.mean(grad_norms)),grad_norm_max=max(grad_norms),
            rate_supported_cells=supported_cells,rate_supported_windows=int(d.rate_mask[roles['train']].sum())))
        state = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),random_state=random_state(),
            history=history,probes=probes,best_mae=best,best_checkpoint=selected,updates=updates,next_epoch=epoch+1,
            elapsed_seconds=elapsed+time.monotonic()-started,contract=contract)
        save_torch(state_path,state)
        if (epoch+1)%25 == 0 or epoch == start_epoch:
            print(json.dumps(dict(event='TRAIN',domain=b.domain,fold=fold,arm=arm,epoch=epoch+1,updates=updates)),flush=True)
        if stop_after is not None and epoch+1>=stop_after:
            return dict(status='INTERRUPTED_FOR_RECOVERY_TEST',updates=updates)
    # Always materialize authoritative final state, even if resume skips the epoch loop.
    write_json(dest/'HISTORY.json',history)
    write_json(dest/'GRADIENT_PROBES.json',probes)
    save_torch(dest/'BEST.pt',selected)
    assert len(history)==epochs and updates == contract['expected_updates']
    receipt = dict(prediction_scope=SCOPE,status='TRAINED_VALIDATION_ONLY',unit_contract=contract,
        best_epoch=selected['best_epoch'],validation_mae=best,updates=updates,
        elapsed_seconds=state['elapsed_seconds'],test_observations_consumed=0,
        files=file_records(dest,['UNIT_CONTRACT.json','SCALER_TRAIN_ONLY.json','HISTORY.json',
                                 'GRADIENT_PROBES.json','BEST.pt','TRAIN_STATE.pt']))
    write_json(dest/'TRAINING_COMPLETE.json',receipt)
    return receipt


def evaluate_unit(d,fold,arm,spec,device='cuda'):
    dest = OUT/d.bundle.domain/fold/arm
    tr = json.loads((dest/'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
    verify_files(dest,tr)
    if (dest/'COMPLETE.json').exists():
        complete = json.loads((dest/'COMPLETE.json').read_text(encoding='utf-8'))
        verify_files(dest,complete)
        return complete
    roles,x,p,phy,_ = scaled_inputs(d,fold)
    checkpoint = torch.load(dest/'BEST.pt',map_location=device,weights_only=False)
    model = RootCauseNet(selected_encoder(d.bundle.domain,fold),spec['decoder'],d.bundle.policy.shape[1],spec['features']).to(device)
    model.load_state_dict(checkpoint['model'])
    rows = roles['test']
    pred = predict(model,d,x,p,phy,rows,device)
    b = d.bundle
    np.savez_compressed(dest/'test_predictions.npz',ids=np.array([b.ids[i] for i in rows]),query=b.query,
                        prediction=pred,target=b.targets[rows],observed=b.observed[rows])
    complete = dict(prediction_scope=SCOPE,status='EVALUATED',training_sha256=file_hash(dest/'TRAINING_COMPLETE.json'),
        test=metrics(pred,b.targets[rows],b.observed[rows],b.query),
        files=file_records(dest,[v['path'] for v in tr['files']]+['TRAINING_COMPLETE.json','test_predictions.npz']))
    write_json(dest/'COMPLETE.json',complete)
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['audit','register','train','evaluate'],required=True)
    parser.add_argument('--domain',choices=['MATR','HUST'])
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    configure()
    if args.phase == 'audit':
        from .data import audit
        audit()
        return
    if args.phase == 'register':
        register_sources()
        return
    sha = check_sources()
    c = settings()
    names = [args.domain] if args.domain else c['domains']
    with execution_lock(OUT/'.legal.lock'):
        for domain in names:
            d = load_domain(domain)
            for fold in c['folds']:
                for arm,spec in c['arms'].items():
                    if args.phase == 'train':
                        train_unit(d,fold,arm,spec,device=args.device,source_sha=sha)
                    else:
                        evaluate_unit(d,fold,arm,spec,args.device)
            print(json.dumps(dict(event='DOMAIN_'+args.phase.upper()+'_COMPLETE',domain=domain)),flush=True)


if __name__ == '__main__':
    main()
