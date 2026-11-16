"""Source-only, resumable training and replay for fixed-head battery TSFM studies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch

from models.drs_peft.trajectory import BatteryTSFM
from models.drs_peft.peft import install_peft,trainable_parameter_report
from scripts.drs_peft.data import load_dataset,SourceOnlyScaler,file_hash,write_json
from scripts.drs_peft.episodes import load_registration

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'configs/drs_peft/execution_v1.json'
INPUT_KEYS=('x','point_mask','cycle_ids','policy','policy_mask','anchor')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def configure():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def save_torch(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    torch.save(value,pending)
    os.replace(pending,path)


def bind(path,value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:
            raise RuntimeError('Registered execution identity changed at '+str(path))
    else:
        write_json(path,value)


def source_snapshot(config_path=DEFAULT_CONFIG,method=None,extra_sources=()):
    names=['models/drs_peft/backbone.py','models/drs_peft/bridges.py','models/drs_peft/trajectory.py',
           'models/drs_peft/peft.py','scripts/drs_peft/train.py','scripts/drs_peft/data.py','scripts/drs_peft/episodes.py']
    if method=='B3_battery_direct':
        names.extend(['models/drs_peft/battery_baseline.py','models/unified_matr/model.py',
                      'models/unified_matr/legacy_operators.py'])
    names.extend(extra_sources)
    paths=[ROOT/name for name in names]
    paths.extend([Path(config_path),ROOT/'configs/drs_peft/episodes_v1.json'])
    return [dict(path=p.relative_to(ROOT).as_posix(),sha256=file_hash(p)) for p in paths]


def selected_state(model):
    """Save trainable/new state, while authenticating the frozen official backbone separately."""
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if not k.startswith('bridge.backbone.module.') or
            any(token in k for token in ('lora_a.','lora_b.','.core','.U','.V','singular','initial_','source_core','delta_core','target_active'))}


def load_selected(model,state,allow_adapter_initialization=False):
    expected=set(selected_state(model))
    missing=expected-set(state)
    allowed_missing={key for key in missing if key.startswith('bridge.backbone.module.')} if allow_adapter_initialization else set()
    if missing-allowed_missing or set(state)-expected:
        raise RuntimeError('Serialized selected state keys differ from the registered model')
    result=model.load_state_dict(state,strict=False)
    bad=[k for k in result.missing_keys if not k.startswith('bridge.backbone.module.')]
    if bad or result.unexpected_keys:
        raise RuntimeError('Incomplete reconstructed model state '+str((bad,result.unexpected_keys)))


def tensor_fingerprint(module):
    digest=hashlib.sha256()
    for key,tensor in sorted(module.state_dict().items()):
        digest.update(key.encode('utf-8'))
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_parameter_fingerprint(module):
    digest=hashlib.sha256()
    for name,parameter in sorted(module.named_parameters()):
        if parameter.requires_grad:
            continue
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(parameter.shape)).encode('ascii'))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_model(backbone_dir,policy_dim,bridge,method,seed,rank=4,bases=None,gradients=None,alpha=None,
               baseline_scaler=None):
    seed_all(seed)
    if method=='B3_battery_direct':
        from models.drs_peft.battery_baseline import B3M1SharedHead
        return B3M1SharedHead(policy_dim=policy_dim,seed=seed,input_scaler=baseline_scaler)
    model=BatteryTSFM(backbone_dir,policy_dim,bridge,pretrained=method!='B1_random_frozen')
    model.official_backbone_identity=model.bridge.backbone.audit()
    model.official_backbone_identity['actual_initial_tensor_sha256']=tensor_fingerprint(model.backbone)
    if method=='B2_standard_lora':
        install_peft(model.backbone,'lora',rank,alpha=rank if alpha is None else alpha,dropout=0.,seed=seed)
    elif method in ('lora','lora_xs','weight_svd_core','grad_core','random_core','lora_ga'):
        install_peft(model.backbone,method,rank,alpha=alpha,dropout=0.,seed=seed,bases=bases,gradients=gradients)
    elif method not in ('B0_pretrained_frozen','B1_random_frozen'):
        raise ValueError('Unregistered study method '+method)
    return model


def early_batch(inputs,rows,device,keys=INPUT_KEYS):
    return {key:torch.as_tensor(inputs[key][rows],device=device) for key in keys}


def cell_loss(prediction,target,observed):
    counts=observed.sum(1)
    if not bool((counts>0).all()):
        raise ValueError('A source/support cell has no actual future supervision')
    return (torch.where(observed,(prediction-target).abs(),0.).sum(1)/counts).mean()


def evaluate_arrays(prediction,target,observed,query,ids,protocols,coordinate):
    rows=[]
    windows=[('all',101,5000),('101_200',101,200),('201_500',201,500),('501_1000',501,1000),('1001_plus',1001,5000)]
    for i,uid in enumerate(ids):
        for label,left,right in windows:
            valid=observed[i]&(query>=left)&(query<=right)
            error=prediction[i,valid].astype(float)-target[i,valid].astype(float)
            rows.append(dict(prediction_scope='SOURCE_META_VALIDATION',cell_uid=uid,protocol_uid=str(protocols[i]),
                coordinate=coordinate,window=label,n_obs=int(valid.sum()),
                mae=float(np.abs(error).mean()) if len(error) else None,
                rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                bias=float(error.mean()) if len(error) else None))
    table=pd.DataFrame(rows)
    supported=table[(table.window=='all')&table.n_obs.gt(0)]
    return table,dict(cells=len(supported),observations=int(supported.n_obs.sum()),
        mae=float(supported.mae.mean()),rmse=float(supported.rmse.mean()),
        protocol_macro_mae=float(supported.groupby('protocol_uid').mae.mean().mean()))


def predict(model,inputs,rows,query,device,batch_size=8):
    model.eval()
    output=[]
    q=torch.as_tensor(query,device=device)
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            output.append(model(early_batch(inputs,rows[start:start+batch_size],device,getattr(model,'input_keys',INPUT_KEYS)),q).cpu().numpy())
    result=np.concatenate(output)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite fixed-grid prediction')
    return result


def coordinate_name(data):
    return 'supplied_cycle_number' if data.domain=='Stanford_2' else 'original_cycle_id'


def train_source_unit(data,inputs,scaler,episodes,config,config_path,backbone_dir,destination,
                      bridge,method,seed,epochs,device,rank=4,adapter_lr=None,bases=None,gradients=None,
                      stop_after=None,stage='P0',initial_state=None,initial_checkpoint_sha256=None,
                      model_factory=None,extra_sources=()):
    out=Path(destination)
    out.mkdir(parents=True,exist_ok=True)
    training=data.rows(episodes['source_cell_roles']['source_train'])
    validation=data.rows(episodes['source_cell_roles']['source_meta_validation'])
    assert not set(training)&set(validation)
    baseline_scaler=None
    if method=='B3_battery_direct':
        from models.drs_peft.battery_baseline import B3InputScaler
        baseline_scaler=B3InputScaler().fit(inputs,training,[data.ids[i] for i in training]).to_dict()
        bind(out/'BATTERY_BASELINE_SCALER.json',baseline_scaler)
    def construct():
        if model_factory is not None:
            return model_factory().to(device)
        return make_model(backbone_dir,inputs['policy'].shape[-1],bridge,method,seed,rank,bases,gradients,
                          baseline_scaler=baseline_scaler).to(device)
    model=construct()
    if initial_state is not None:
        load_selected(model,initial_state,allow_adapter_initialization=True)
    frozen_initial=frozen_parameter_fingerprint(model)
    trainable=trainable_parameter_report(model)
    parameter_report=model.parameter_report()
    parameter_report['peft_detail']=trainable
    identity=dict(stage=stage,method=method,bridge=bridge,seed=seed,epochs=epochs,batch_size=8,rank=rank,
        learning_rate=config['training']['learning_rate'],adapter_learning_rate=adapter_lr,
        weight_decay=.01,gradient_clip=1.,head=config['head'],loss=config['loss'],
        episode_sha256=episodes['episode_contract_sha256'],config_sha256=file_hash(config_path),
        data_cache_sha256=file_hash(Path(data.provenance['cache_root'])/'COMPLETE.json') if 'cache_root' in data.provenance else hashlib.sha256(json.dumps(data.provenance,sort_keys=True).encode()).hexdigest(),
        source_snapshot=source_snapshot(config_path,method,extra_sources),training_ids=[data.ids[i] for i in training],
        validation_ids=[data.ids[i] for i in validation],query_grid=[101,5000],
        expected_updates=epochs*math.ceil(len(training)/8),target_query_labels_used=0,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        source_commit=os.environ.get('BATTERY_SOURCE_COMMIT'),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    bind(out/'RUN_CONFIG.json',identity)
    bind(out/'SOURCE_SNAPSHOT.json',dict(files=identity['source_snapshot']))
    bind(out/'EPISODE_IDS.json',dict(source_train=identity['training_ids'],source_meta_validation=identity['validation_ids']))
    bind(out/'TRAINABLE_PARAMETER_REPORT.json',parameter_report)
    bind(out/'SOURCE_SCALER.json',scaler.to_dict())
    bind(out/'EXECUTION_ENVIRONMENT.json',dict(python=platform.python_version(),platform=platform.system(),
        torch=torch.__version__,cuda_runtime=torch.version.cuda,numpy=np.__version__,pandas=pd.__version__,
        device=device,gpu_name=torch.cuda.get_device_name() if device.startswith('cuda') else None,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),amp=False,tf32=False,cpu_threads=torch.get_num_threads()))
    if method!='B3_battery_direct':
        bind(out/'BACKBONE_IDENTITY.json',model.official_backbone_identity)
    else:
        bind(out/'BACKBONE_IDENTITY.json',dict(kind='M1_battery_encoder',pretrained=False,shared_head=True))
    bind(out/'DATA_FINGERPRINT.json',data.provenance)
    complete=out/'SOURCE_DEVELOPMENT_COMPLETE.json'
    if complete.exists():
        receipt=read_json(complete)
        for record in receipt['files']:
            assert file_hash(out/record['path'])==record['sha256']
        return receipt
    common,adapters=[],[]
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        (adapters if name.startswith('bridge.backbone.module.') else common).append(p)
    groups=[dict(params=common,lr=config['training']['learning_rate'])]
    if adapters:
        groups.append(dict(params=adapters,lr=adapter_lr or config['training']['learning_rate']))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    history,best,checkpoint,updates,start_epoch,elapsed=[],math.inf,None,0,0,0.
    state_path=out/'TRAIN_STATE.pt'
    if state_path.exists():
        state=torch.load(state_path,map_location=device,weights_only=False)
        if state['identity']!=identity:
            raise RuntimeError('Resume identity differs')
        load_selected(model,state['model'])
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['random'])
        history,best,checkpoint,updates,start_epoch,elapsed=[state[k] for k in (
            'history','best','selected','updates','next_epoch','elapsed')]
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started=time.monotonic()
    query=torch.as_tensor(data.query,device=device)
    for epoch in range(start_epoch,epochs):
        model.train()
        order=np.random.default_rng(seed+epoch*1009).permutation(training)
        total,gradient_norms=0.,[]
        for start in range(0,len(order),8):
            rows=order[start:start+8]
            # Supervision may use training masks. Inference always receives the fixed query grid.
            supervised=np.flatnonzero(data.observed[rows].any(0))
            q=query[torch.as_tensor(supervised,device=device)]
            optimizer.zero_grad(set_to_none=True)
            prediction=model(early_batch(inputs,rows,device,getattr(model,'input_keys',INPUT_KEYS)),q)
            loss=cell_loss(prediction,torch.as_tensor(data.targets[rows][:,supervised],device=device),
                           torch.as_tensor(data.observed[rows][:,supervised],device=device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            updates+=1
            total+=float(loss.detach().cpu())*len(rows)
            gradient_norms.append(float(norm.detach().cpu()))
        validation_prediction=predict(model,inputs,validation,data.query,device)
        _,score=evaluate_arrays(validation_prediction,data.targets[validation],data.observed[validation],data.query,
            [data.ids[i] for i in validation],data.protocol_ids[validation],coordinate_name(data))
        selected=score['mae']<best
        if selected:
            best=score['mae']
            checkpoint=dict(model=selected_state(model),identity=identity,best_epoch=epoch+1,validation=score,
                            backbone_id=read_json(out/'BACKBONE_IDENTITY.json'))
        history.append(dict(epoch=epoch+1,train_mae=total/len(training),validation_mae=score['mae'],
            updates=updates,selected=selected,learning_rates=[g['lr'] for g in optimizer.param_groups],
            gradient_norm_mean=float(np.mean(gradient_norms)),gradient_norm_max=max(gradient_norms)))
        state=dict(model=selected_state(model),optimizer=optimizer.state_dict(),random=rng_state(),history=history,
                   best=best,selected=checkpoint,updates=updates,next_epoch=epoch+1,
                   elapsed=elapsed+time.monotonic()-started,identity=identity)
        save_torch(state_path,state)
        if epoch==start_epoch or (epoch+1)%10==0:
            print(json.dumps(dict(event='SOURCE_TRAINING',stage=stage,method=method,bridge=bridge,seed=seed,
                                  epoch=epoch+1,updates=updates,validation_mae=score['mae'])),flush=True)
        if stop_after is not None and epoch+1>=stop_after:
            return dict(status='ENGINEERING_INTERRUPTED',updates=updates)
    write_json(out/'TRAIN_LOG.json',history)
    (out/'TRAIN_LOG.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
    save_torch(out/'CHECKPOINT_SELECTED.pt',checkpoint)
    write_json(out/'CHECKPOINT_SELECTION.json',dict(rule='first_strict_source_meta_validation_minimum',
               best_epoch=checkpoint['best_epoch'],validation=checkpoint['validation']))
    assert updates==identity['expected_updates'] and len(history)==epochs
    if frozen_parameter_fingerprint(model)!=frozen_initial:
        raise RuntimeError('Frozen model parameters changed during source training')
    training_memory=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()) if device.startswith('cuda') else dict(allocated=0,reserved=0)
    load_selected(model,checkpoint['model'])
    prediction=predict(model,inputs,validation,data.query,device)
    replay=predict(model,inputs,validation,data.query,device)
    assert np.array_equal(prediction,replay)
    table,score=evaluate_arrays(prediction,data.targets[validation],data.observed[validation],data.query,
        [data.ids[i] for i in validation],data.protocol_ids[validation],coordinate_name(data))
    assert score==checkpoint['validation']
    table.to_csv(out/'SOURCE_VALIDATION_CELL_METRICS.csv',index=False)
    np.savez_compressed(out/'SOURCE_VALIDATION_PREDICTIONS.npz',ids=np.asarray([data.ids[i] for i in validation]),
        query=data.query,prediction=prediction,target=data.targets[validation],observed=data.observed[validation])
    # Independent serialized reconstruction through the normal model entry point.
    serialized_checkpoint=torch.load(out/'CHECKPOINT_SELECTED.pt',map_location='cpu',weights_only=False)
    if serialized_checkpoint['identity']!=identity:
        raise RuntimeError('Serialized checkpoint execution identity differs')
    reconstructed=construct()
    if frozen_parameter_fingerprint(reconstructed)!=frozen_initial:
        raise RuntimeError('Reconstructed frozen parameters differ from training initialization')
    load_selected(reconstructed,serialized_checkpoint['model'])
    replay=predict(reconstructed,inputs,validation,data.query,device)
    assert np.array_equal(prediction,replay)
    memory=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()) if device.startswith('cuda') else dict(allocated=0,reserved=0)
    receipt=dict(status='SOURCE_DEVELOPMENT_COMPLETE',stage=stage,method=method,bridge=bridge,seed=seed,
        epochs=epochs,updates=updates,validation=score,best_epoch=checkpoint['best_epoch'],
        trainable_parameters=identity['trainable_parameters'],elapsed_seconds=state['elapsed'],
        peak_gpu_memory_bytes=training_memory,verification_peak_gpu_memory_bytes=memory,
        selected_checkpoint_sha256=file_hash(out/'CHECKPOINT_SELECTED.pt'),
        prediction_replay_max_absolute_error=0.,target_query_labels_used=0,
        frozen_parameter_sha256=frozen_initial,frozen_parameters_unchanged=True,
        final_target_test_status='SEALED_NOT_EVALUATED',
        files=[dict(path=p.name,sha256=file_hash(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='SOURCE_DEVELOPMENT_COMPLETE.json'])
    write_json(complete,receipt)
    del model,reconstructed,optimizer
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return receipt


def load_source(data_root):
    data=load_dataset('MATR',cache_root=data_root,build_if_missing=False)
    episodes=load_registration()
    rows=data.rows(episodes['source_cell_roles']['source_train'])
    scaler=SourceOnlyScaler().fit(data,rows)
    return data,scaler.transform(data),scaler,episodes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--data-root',type=Path,default=ROOT/'runtime/drs_peft/data_v1')
    parser.add_argument('--output-root',type=Path,default=ROOT/'results/drs_peft')
    parser.add_argument('--backbone-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--phase',choices=['pilot','formal','one'],default='pilot')
    parser.add_argument('--method',default='B0_pretrained_frozen')
    parser.add_argument('--bridge',default='native_multivariate')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    configure()
    config=read_json(args.config)
    data,inputs,scaler,episodes=load_source(args.data_root)
    if args.dry_run:
        print(json.dumps(dict(status='READY_NO_TRAINING',source_cells=len(episodes['source_cell_roles']['source_train']),
            validation_cells=len(episodes['source_cell_roles']['source_meta_validation']),source_snapshot=source_snapshot(args.config))))
        return
    if args.phase=='pilot':
        jobs=[(bridge,method,42,config['P0']['pilot_epochs']) for bridge in config['bridges']
              for method in config['P0']['methods'][:3]]
    elif args.phase=='formal':
        lock=read_json(args.output_root/'P0/BRIDGE_SELECTION.json')
        jobs=[(lock['selected_bridge'],method,seed,config['P0']['formal_epochs'])
              for seed in config['training_seeds'] for method in config['P0']['methods']]
    else:
        jobs=[(args.bridge,args.method,args.seed,args.epochs or config['P0']['pilot_epochs'])]
    receipts=[]
    for bridge,method,seed,epochs in jobs:
        dest=args.output_root/'P0'/args.phase/bridge/method/f'seed{seed}'
        receipts.append(train_source_unit(data,inputs,scaler,episodes,config,args.config,args.backbone_dir,dest,
                                         bridge,method,seed,epochs,args.device,rank=config['P0']['lora_rank']))
    if args.phase=='pilot':
        candidates={bridge:float(np.mean([r['validation']['mae'] for r in receipts
                    if r['bridge']==bridge and r['method'] in ('B0_pretrained_frozen','B2_standard_lora')])) for bridge in config['bridges']}
        best=min(config['bridges'],key=candidates.get)
        bind(args.output_root/'P0/BRIDGE_SELECTION.json',dict(selected_bridge=best,candidates=candidates,
             criterion=config['P0']['bridge_selection'],target_query_labels_used=0,
             pilot_receipts=[r['selected_checkpoint_sha256'] for r in receipts]))
    print(json.dumps(dict(event='P0_PHASE_COMPLETE',phase=args.phase,unique_units=len(receipts),
                         updates=sum(r['updates'] for r in receipts))),flush=True)


if __name__=='__main__':
    main()
