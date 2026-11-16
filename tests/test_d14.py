import copy
import unittest
import numpy as np
import torch
from n012.run import configure,seed_all
from rootcause_d14.data import PhysicsScaler
from rootcause_d14.model import PhysicsBranch,RootCauseNet
from rootcause_d14.run import rate_loss
from rootcause_d14.oracle import evaluate,starts,physics_values
from rootcause_d14.diagnostics import fit_proxy


def fake_arrays():
    rng = np.random.default_rng(7)
    arrays = {}
    for value,mask,shape in [('global_features','global_mask',(4,100,20)),
        ('local_residual','local_mask',(4,100,64)),('local_features','local_feature_mask',(4,100,8,26))]:
        arrays[value] = rng.normal(size=shape).astype('float32')
        arrays[mask] = rng.random(shape)>.2
    return arrays


class D14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_physics_masks(self):
        arrays = fake_arrays()
        a = {k:torch.tensor(v) for k,v in arrays.items()}
        b = copy.deepcopy(a)
        for k,m in [('global_features','global_mask'),('local_residual','local_mask'),('local_features','local_feature_mask')]:
            b[k][~b[m]] = float('nan')
        for kind in ('G','L','GL'):
            net = PhysicsBranch(kind).eval()
            torch.testing.assert_close(net(a),net(b),rtol=0,atol=0)
        for k in a:
            a[k] = torch.zeros_like(a[k])
        self.assertTrue(torch.isfinite(PhysicsBranch('GL')(a)).all())

    def test_scaler_train_only(self):
        arrays = fake_arrays()
        first = PhysicsScaler().fit(arrays,[0,1])
        for k in ('global_features','local_residual','local_features'):
            arrays[k][2:] = 1e20
        self.assertEqual(first.stats,PhysicsScaler().fit(arrays,[0,1]).stats)
        for k,m in [('global_features','global_mask'),('local_residual','local_mask'),('local_features','local_feature_mask')]:
            self.assertTrue((first.apply(arrays)[k][~arrays[m]]==0).all())

    def test_capacity_and_shared_initialization(self):
        seed_all(42)
        raw = RootCauseNet('voltage_cnn','query_mlp',15,'none')
        seed_all(42)
        full = RootCauseNet('voltage_cnn','query_mlp',15,'GL')
        seed_all(42)
        cap = RootCauseNet('voltage_cnn','query_mlp',15,'capacity')
        for k,v in raw.base.state_dict().items():
            torch.testing.assert_close(v,full.base.state_dict()[k],rtol=0,atol=0)
        n1,n2 = [sum(v.numel() for v in m.branch.parameters()) for m in (full,cap)]
        self.assertLess(abs(n1-n2)/n1,.01)

    def test_supported_cell_rate_weighting(self):
        q = np.array([200,500,1000])
        pred = torch.tensor([[1.,.7,.2],[1.,.4,.0],[1.,.5,.1]],requires_grad=True)
        target = torch.zeros(3,2)
        mask = torch.tensor([[True,False],[True,True],[False,False]])
        expected = (1.0+(2.0+.8)/2)/2
        self.assertAlmostEqual(float(rate_loss(pred,target,mask,q)),expected,places=6)
        rate_loss(pred,target,torch.zeros_like(mask),q).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_oracle_analytic_jacobians(self):
        u = np.array([.001,.13,.6,1.31,2.5])
        for family in ('physics_free','physics_shared','spline10','positive_rate8','polynomial4','linear1'):
            raw = starts(family)[1]
            shared = [.42,.23] if family=='physics_shared' else None
            p,jac = evaluate(family,raw,u,.98,shared)
            numerical = []
            for k in range(len(raw)):
                left,right = raw.copy(),raw.copy()
                left[k]-=1e-5
                right[k]+=1e-5
                numerical.append((evaluate(family,right,u,.98,shared)[0]-evaluate(family,left,u,.98,shared)[0])/2e-5)
            np.testing.assert_allclose(jac,np.column_stack(numerical),rtol=2e-5,atol=2e-8)

    def test_query_anchor_and_shapes(self):
        for family in ('physics_free','physics_shared','spline10','positive_rate8','linear1'):
            raw = starts(family)[0]
            shared = [.42,.23] if family=='physics_shared' else None
            p,_ = evaluate(family,raw,np.array([0,.1,.7,4.,6.]),1.,shared)
            self.assertAlmostEqual(p[0],1.)
            self.assertTrue((np.diff(p)<=1e-12).all())
            one,_ = evaluate(family,raw,np.array([.7]),1.,shared)
            self.assertEqual(p[2],one[0])

    def test_proxy_split_isolation(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(20,4))
        y = rng.normal(size=20)*.0001
        roles = dict(train=np.arange(10),validation=np.arange(10,15),test=np.arange(15,20))
        a = fit_proxy(x,y,np.ones(20,bool),roles,[.1,1.])
        y[15:] = 1e6
        b = fit_proxy(x,y,np.ones(20,bool),roles,[.1,1.])
        np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(a[2],b[2])

    def test_oracle_training_import_isolation(self):
        from pathlib import Path
        code = (Path(__file__).parents[1]/'rootcause_d14/run.py').read_text(encoding='utf-8')
        self.assertNotIn('from .oracle',code)
        self.assertNotIn("/'ORACLE_DIAGNOSTIC'",code)


if __name__ == '__main__':
    unittest.main()
