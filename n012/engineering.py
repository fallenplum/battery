"""Real-input checks before the fixed N012 training matrix."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.special import erf
import torch

from .data import load_bundle, role_indices, TrainOnlyScaler, file_hash
from .model import ENCODERS, DECODERS, TrajectoryNet
from . import run


def numpy_mlp(network, values):
    for layer in network:
        if isinstance(layer,torch.nn.Linear):
            values=values@layer.weight.detach().cpu().numpy().T+layer.bias.detach().cpu().numpy()
        elif isinstance(layer,torch.nn.GELU):
            values=.5*values*(1+erf(values/np.sqrt(2)))
        else:
            raise TypeError(type(layer).__name__)
    return values


def free_rate_oracle(head,state,anchor,query):
    grid=np.arange(100,max(101,int(np.ceil(query.max())))+1)
    u=(grid-100)/1000
    context=np.broadcast_to(state[:,None],(len(state),len(grid),state.shape[-1]))
    coordinates=np.broadcast_to(u[None,:,None],(len(state),len(grid),1))
    logits=numpy_mlp(head.net,np.concatenate((context,coordinates),axis=-1)).squeeze(-1)+float(head.bias.detach().cpu())
    rate=np.logaddexp(0,logits)
    integral=np.column_stack((np.zeros(len(state)),np.cumsum((rate[:,1:]+rate[:,:-1])/2000,axis=1)))
    left=np.minimum(np.floor(query-100).astype(int),len(grid)-2)
    fraction=query-100-left
    row=np.arange(len(state))[:,None]
    return anchor[:,None]-(integral[row,left]+fraction*(integral[row,left+1]-integral[row,left]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo-root',type=Path,default=run.ROOT)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    args=parser.parse_args()
    run.configure()
    checks=[]
    def require(name,passed,**details):
        checks.append(dict(name=name,status='PASS' if bool(passed) else 'FAIL',**details))
        if not passed:
            raise AssertionError(name)
    result=dict(status='FAIL',device=args.device,test_observations_consumed=0,formal_optimizer_updates=0)
    try:
        tree=ast.parse((run.ROOT/'n012/model.py').read_text(encoding='utf-8'))
        imported=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        require('independent_model_operators',not any('unified_matr' in name for name in imported))
        for domain in ('MATR','HUST'):
            bundle=load_bundle(args.repo_root,domain)
            roles=role_indices(bundle,'cell_fold_0')
            scaler=TrainOnlyScaler().fit(bundle,roles['train'])
            x,p=scaler.apply(bundle)
            rows=roles['train'][:8]
            batch=run.to_tensors(bundle,x,p,rows,args.device)
            combinations=[(enc,'query_mlp') for enc in ENCODERS]+[('cnn_tcn',dec) for dec in DECODERS[1:]]
            for encoder,decoder in combinations:
                repeats=[]
                for repeat in range(2):
                    run.seed_all(42)
                    model=TrajectoryNet(encoder,decoder,policy_dim=bundle.policy.shape[1]).to(args.device).train()
                    prediction=run.forward(model,batch)
                    loss=run.cell_mae(prediction,batch['y'],batch['mask'])
                    loss.backward()
                    require(domain+'/'+encoder+'/'+decoder+'/finite_all_gradients/'+str(repeat),
                            bool(torch.isfinite(loss)) and all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()))
                    digest=hashlib.sha256()
                    for name,parameter in model.named_parameters():
                        digest.update(name.encode())
                        digest.update(parameter.grad.detach().cpu().numpy().tobytes())
                    repeats.append(dict(loss=float(loss.detach().cpu()),gradient_sha256=digest.hexdigest(),
                                        prediction_sha256=hashlib.sha256(prediction.detach().cpu().numpy().tobytes()).hexdigest()))
                    del model,loss,prediction
                require(domain+'/'+encoder+'/'+decoder+'/fresh_seed42_exact',repeats[0]==repeats[1],repeats=repeats)
            query=torch.tensor([100.,101.,207.5,500.,1000.,5000.,10000.],device=args.device)
            for decoder in DECODERS:
                run.seed_all(42)
                model=TrajectoryNet('cnn_tcn',decoder,policy_dim=bundle.policy.shape[1]).to(args.device).eval()
                inputs={key:batch[key] for key in ('x','point_mask','cycle_ids','policy','policy_mask','anchor')}
                with torch.no_grad():
                    prediction=model(**inputs,query=query)
                    require(domain+'/'+decoder+'/far_queries_finite',bool(torch.isfinite(prediction).all()))
                    require(domain+'/'+decoder+'/anchor',torch.allclose(prediction[:,0],batch['anchor'],atol=2e-6,rtol=2e-6))
                    reordered=model(**inputs,query=query[[5,0,2,4,2]])
                    require(domain+'/'+decoder+'/query_order',torch.allclose(reordered,prediction[:,[5,0,2,4,2]],atol=2e-6,rtol=2e-6))
                    if decoder in ('free_rate','monotone_spline','physics_basis'):
                        require(domain+'/'+decoder+'/nonincreasing',bool((prediction[:,1:]<=prediction[:,:-1]+1e-6).all()))
                    if decoder in ('monotone_spline','physics_basis'):
                        require(domain+'/'+decoder+'/nonnegative',bool((prediction>=0).all()))
                    if decoder=='free_rate':
                        state=model.encoder(inputs['x'],inputs['point_mask'],inputs['cycle_ids'],inputs['policy'],inputs['policy_mask'])
                        q=np.broadcast_to(query.cpu().numpy().astype(float),(len(rows),len(query)))
                        expected=free_rate_oracle(model.decoder,state.cpu().numpy().astype(float),batch['anchor'].cpu().numpy(),q)
                        require(domain+'/independent_free_rate_integral',np.allclose(prediction.cpu().numpy(),expected,atol=2e-6,rtol=2e-5),
                                max_absolute_error=float(np.max(np.abs(prediction.cpu().numpy()-expected))))
                del model
            if domain=='MATR':
                smoke=run.train_unit(bundle,'cell_fold_0','cnn_tcn','query_mlp',args.out/'real_smoke',
                                     'engineering_only_no_test',epochs=2,device=args.device)
                require('real_two_epoch_smoke',smoke['updates']==28 and smoke['test_observations_consumed']==0)
                result['smoke_optimizer_updates']=smoke['updates']
                result['smoke_elapsed_seconds']=smoke['elapsed_seconds']
        result.update(status='PASS',source_sha256=file_hash(__file__),sources=run.source_hashes(),
                      actual_real_gradient_batches=36,smoke_epochs=2)
    except Exception:
        result['traceback']=traceback.format_exc()
        raise
    finally:
        result['checks']=checks
        result['passed_checks']=sum(row['status']=='PASS' for row in checks)
        run.write_json(args.out/('ENGINEERING_'+args.device.upper()+'.json'),result)
    print(json.dumps(dict(status=result['status'],checks=result['passed_checks'],smoke_updates=result['smoke_optimizer_updates'],
                          two_epoch_seconds=result['smoke_elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
