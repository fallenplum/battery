import numpy as np
from rootcause_d14.diagnostics import physical_jacobian
from rootcause_d14.oracle import physics_values,softplus


def test_physical_sensitivity_survives_saturated_raw_mapping():
    u = np.array([.01,.2,.8,2.5])
    physical = physics_values(np.array([0.,-1000.,.2,1000.,-.8,-1000.]))
    def pred(p):
        a,b,c,tau,tc,w = p
        h = a*np.log1p(u/tc)+b*u+c*w*(softplus((u-tau)/w)-softplus(-tau/w))
        return .98*np.exp(-h)
    actual = physical_jacobian(physical,u,.98,6)
    numeric = []
    for j in range(6):
        left,right = physical.copy(),physical.copy()
        left[j]-=1e-7
        right[j]+=1e-7
        numeric.append((pred(right)-pred(left))/2e-7)
    np.testing.assert_allclose(actual,np.column_stack(numeric),rtol=2e-5,atol=2e-8)
    assert np.linalg.norm(actual[:,1])>0
