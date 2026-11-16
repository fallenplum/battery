import json
from pathlib import Path
import numpy as np
import pytest
import torch
from n012.data import synthetic_bundle,TrainOnlyScaler,role_indices,load_stanford
from n012.model import TrajectoryNet,ENCODERS,DECODERS
from n012.run import cell_mae,metrics,execute


def test_all_25_combinations_forward_backward():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    sample=synthetic_bundle(n=12)
    rows=np.array([0,1])
    x=torch.from_numpy(sample.x[rows]);mask=torch.from_numpy(sample.mask[rows])
    cyc=torch.from_numpy(sample.cycle[rows]);policy=torch.from_numpy(sample.policy[rows])
    pm=torch.from_numpy(sample.policy_mask[rows]);anchor=torch.from_numpy(sample.anchor[rows])
    query=torch.from_numpy(np.array([100,101,120,200,300],np.float32))
    for enc in ENCODERS:
        for dec in DECODERS:
            model=TrajectoryNet(enc,dec,policy_dim=15)
            model.train()
            pred=model(x,mask,cyc,policy,pm,anchor,query)
            assert pred.shape==(2,5),(enc,dec)
            assert torch.isfinite(pred).all(),(enc,dec)
            assert torch.allclose(pred[:,0],anchor,atol=2e-6),(enc,dec)
            loss=pred[:,1:].mean()
            loss.backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),(enc,dec)
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()),(enc,dec)


def test_physics_heads_are_monotonic_and_nonnegative():
    data=synthetic_bundle(10)
    x=torch.from_numpy(data.x[:2]);m=torch.from_numpy(data.mask[:2])
    cyc=torch.from_numpy(data.cycle[:2]);p=torch.from_numpy(data.policy[:2])
    pm=torch.from_numpy(data.policy_mask[:2]);a=torch.from_numpy(data.anchor[:2])
    query=torch.arange(100,5001,100).float()
    for head in ("monotone_spline","physics_basis"):
        model=TrajectoryNet("voltage_cnn",head,policy_dim=15).eval()
        with torch.no_grad():result=model(x,m,cyc,p,pm,a,query)
        assert (result>0).all()
        assert (result[:,1:]-result[:,:-1]<=1e-7).all()


def test_no_inference_effect_of_masked_viq():
    data=synthetic_bundle(10)
    data.mask[0,10:12,:]=False
    x=torch.from_numpy(data.x[:2]);m=torch.from_numpy(data.mask[:2]);cyc=torch.from_numpy(data.cycle[:2])
    p=torch.from_numpy(data.policy[:2]);pm=torch.from_numpy(data.policy_mask[:2]);a=torch.from_numpy(data.anchor[:2])
    query=torch.tensor([101.,150.,200.])
    model=TrajectoryNet("cnn_tcn","physics_basis",policy_dim=15).eval()
    with torch.no_grad():
        z=model(x,m,cyc,p,pm,a,query)
        xx=x.clone();xx[0,10:12,:,:]=10000.
        altered=model(xx,m,cyc,p,pm,a,query)
    assert torch.allclose(z,altered,rtol=0,atol=0)


def test_train_only_scaler_and_cell_split():
    data=synthetic_bundle(20)
    roles=role_indices(data,"cell_fold_0")
    assert not set(roles["train"]) & set(roles["validation"])
    assert not set(roles["train"]) & set(roles["test"])
    scaler=TrainOnlyScaler().fit(data,roles["train"])
    previous=scaler.to_dict()
    # Perturb a held-out test cell: scaling must remain exactly unchanged.
    data.x[roles["test"]]+=123456.
    scaler2=TrainOnlyScaler().fit(data,roles["train"])
    assert scaler2.to_dict()==previous


def test_synthetic_metrics_and_mask():
    p=np.array([[1.,3.,8.],[4.,4.,8.]])
    t=np.array([[1.,2.,3.],[4.,5.,6.]])
    m=np.array([[1,1,0],[1,0,1]],dtype=bool)
    result=metrics(p,t,m,np.array([101,201,501]))
    assert result["n_cells"]==2 and result["n_obs"]==4
    assert abs(result["mae"]-0.75)<1e-8  # 1st (0+1)/2, 2nd (0+2)/2


def test_stanford_fail_closed():
    with pytest.raises(RuntimeError,match="STANFORD_BLOCKED"):
        load_stanford("/mnt/data")


def test_nested_n1_n2_selection_on_single_fold(tmp_path):
    torch.set_num_threads(2)
    data=synthetic_bundle(15)
    execute(data,tmp_path,phase="all",fold_limit=1,seed=42,epochs=1,batch_size=4,device="cpu")
    n1=json.loads((tmp_path/"SYNTHETIC/N1_SELECTION.json").read_text())
    n2=json.loads((tmp_path/"SYNTHETIC/N2_SELECTION.json").read_text())
    assert n1["nested_selection"] and n2["nested_selection"]
    assert list(n1["encoder_per_outer_fold"])==["cell_fold_0"]
    assert n2["selected_decoder_per_fold"]["cell_fold_0"] in DECODERS
    assert n2["test_not_used_for_selection"] is True
    assert (tmp_path/"SYNTHETIC/cell_fold_0" ).exists()


def test_every_decoder_query_density_order_and_duplicates():
    from n012.model import HEADS
    torch.manual_seed(42)
    state=torch.randn(2,64,dtype=torch.float64)
    anchor=torch.ones(2,dtype=torch.float64)
    dense=torch.arange(100.,5001.,dtype=torch.float64)[None].expand(2,-1)
    sparse=torch.tensor([[100.,101.,207.5,500.,1000.,5000.]],dtype=torch.float64).expand(2,-1)
    for name in DECODERS:
        head=HEADS[name](64).double().eval()
        full=head(state,anchor,dense)
        few=head(state,anchor,sparse)
        integers=[0,1,400,900,4900]
        assert torch.allclose(few[:,[0,1,3,4,5]],full[:,integers],atol=2e-12,rtol=2e-12),name
        order=[5,2,0,3,1,2]
        assert torch.allclose(head(state,anchor,sparse[:,order]),few[:,order],atol=2e-12,rtol=2e-12),name
        assert torch.allclose(head(state,anchor,sparse[:,2:3]),few[:,2:3],atol=2e-12,rtol=2e-12),name


def test_actual_gap_skips_invalid_prefix_cycles():
    data=synthetic_bundle(10)
    mask=torch.from_numpy(data.mask[:2].copy())
    mask[:,10:12]=False
    model=TrajectoryNet('resmlp','query_mlp').eval()
    captured=[]
    handle=model.encoder.cycle_gap.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    model(torch.from_numpy(data.x[:2]),mask,torch.from_numpy(data.cycle[:2]),
          torch.from_numpy(data.policy[:2]),torch.from_numpy(data.policy_mask[:2]),
          torch.from_numpy(data.anchor[:2]),torch.tensor([101.,500.]))
    handle.remove()
    assert torch.equal(captured[0][:,12,1],torch.full((2,),.03))


def test_checkpoint_resume_has_exact_model_optimizer_rng(tmp_path):
    from unittest.mock import patch
    from n012 import run
    data=synthetic_bundle(15)
    settings=dict(seed=42,epochs=3,batch_size=4,device='cpu')
    full=tmp_path/'full'
    resumed=tmp_path/'resumed'
    run.train_unit(data,'cell_fold_0','cnn_transformer','query_mlp',full,'engineering',**settings)
    original=run.save_torch
    def interrupt(path,value):
        original(path,value)
        if path.name=='TRAIN_STATE.pt' and value['next_epoch']==1:
            raise RuntimeError('simulated interruption after committed epoch1')
    with patch.object(run,'save_torch',interrupt):
        with pytest.raises(RuntimeError,match='simulated interruption'):
            run.train_unit(data,'cell_fold_0','cnn_transformer','query_mlp',resumed,'engineering',**settings)
    run.train_unit(data,'cell_fold_0','cnn_transformer','query_mlp',resumed,'engineering',**settings)
    def exact(a,b):
        if type(a) is not type(b):
            return False
        if isinstance(a,torch.Tensor):
            return torch.equal(a,b)
        if isinstance(a,np.ndarray):
            return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        if isinstance(a,dict):
            return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):
            return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
        return a==b
    states=[torch.load(path/'SYNTHETIC/cell_fold_0/cnn_transformer__query_mlp/TRAIN_STATE.pt',weights_only=False) for path in (full,resumed)]
    for key in ('model','optimizer','random_state','best_checkpoint','history','updates','next_epoch','contract'):
        assert exact(states[0][key],states[1][key]),key


def test_nan_under_masks_and_no_hidden_target_input():
    data=synthetic_bundle(10)
    data.mask[0,5:7]=False
    data.policy_mask[0,2]=False
    x=torch.from_numpy(data.x[:2])
    point_mask=torch.from_numpy(data.mask[:2])
    policy=torch.from_numpy(data.policy[:2])
    policy_mask=torch.from_numpy(data.policy_mask[:2])
    cycles=torch.from_numpy(data.cycle[:2])
    anchor=torch.from_numpy(data.anchor[:2])
    for encoder in ENCODERS:
        model=TrajectoryNet(encoder,'query_mlp').eval()
        with torch.no_grad():
            expected=model(x,point_mask,cycles,policy,policy_mask,anchor,torch.tensor([101.,500.]))
            changed_x=x.clone()
            changed_x[~point_mask]=torch.nan
            changed_policy=policy.clone()
            changed_policy[~policy_mask]=torch.nan
            changed_cycle=cycles.clone()
            changed_cycle[~point_mask.any(-1)]=torch.nan
            actual=model(changed_x,point_mask,changed_cycle,changed_policy,policy_mask,anchor,torch.tensor([101.,500.]))
        assert torch.equal(actual,expected),encoder
