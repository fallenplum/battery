from dataclasses import replace
import copy

import numpy as np
import pandas as pd
import pytest

from scripts.drs_peft.data import (load_dataset, SourceOnlyScaler, true_prefix_drift,
    extract_stanford_prefix, POLICY_COLUMNS, POLICY_PHYSICAL_SCALES, ROOT)
from scripts.drs_peft.episodes import load_registration, build_registration, support_draws


@pytest.fixture(scope='module')
def source():
    return load_dataset('MATR')


def test_episode_roles_and_outcome_independent_replay():
    registered=load_registration()
    assert build_registration()==registered
    roles=registered['source_group_roles']
    assert [len(roles[r]) for r in roles]==[47,15,15]
    sets=list(map(set,roles.values()))
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    cells=registered['source_cell_roles']
    assert [len(cells[r]) for r in cells]==[85,30,54]
    assert len(set.union(*map(set,cells.values())))==169
    assert registered['future_outcomes_read_for_split'] is False


def test_support_queries_and_draw_denominators():
    for task in load_registration()['target_tasks']:
        if task['status']!='READY':
            assert task['status']=='NOT_FEASIBLE_Q2' and task['available_cells']==1
            continue
        q=set(task['query_ids'])
        pool=set(task['support_pool_ids'])
        assert not q&pool
        for shot in task['shots']:
            if shot['status']=='NOT_FEASIBLE':
                assert len(pool)<shot['k']
                continue
            seen=set()
            for run in shot['draws']:
                support=run['support_ids']
                assert len(support)==len(set(support))==shot['k']
                assert set(support)<=pool and not set(support)&q
                assert run['query_ids']==task['query_ids']
                seen.add(run['support_set_sha256'])
            assert len(seen)==shot['unique_support_sets']
            if shot['k']==0:
                assert shot['unique_support_sets']==1
        if task['protocol'].startswith('A_'):
            assert next(s for s in task['shots'] if s['k']==10)['status']=='NOT_FEASIBLE'
        else:
            assert task['query_count']==(91 if task['domain']=='Stanford_2' else 39)


def test_duplicate_five_shot_support_is_not_five_independent_sets():
    episodes=support_draws(['s1','s2','s3','s4','s5'],['q1','q2'],seed=42,role='test')
    full=next(s for s in episodes if s['k']==5)
    assert full['unique_support_sets']==1
    assert [r['duplicate_of_draw'] for r in full['draws']]==[None,0,0,0,0]


def test_true_drift_uses_first_ten_supported_cycles():
    q=np.arange(1,101,dtype=float)[None,:,None]*np.ones((1,1,64))
    m=np.ones_like(q,bool)
    cycle=np.arange(1,101,dtype=float)[None]
    drift,mask=true_prefix_drift(q,m,cycle)
    np.testing.assert_allclose(drift[0,:,0],np.arange(1,101)-5.5)
    assert mask.all()
    changed=q.copy()
    changed[:,10:]*=10000
    after,_=true_prefix_drift(changed,m,cycle)
    np.testing.assert_array_equal(after[:,:10],drift[:,:10])
    sparse=m.copy()
    sparse[:,:6]=False
    _,support=true_prefix_drift(q,sparse,cycle)
    assert not support.any()


def test_canonical_units_and_voltages_are_not_z_scored(source):
    registration=load_registration()
    rows=source.rows(registration['source_cell_roles']['source_train'])
    scaler=SourceOnlyScaler().fit(source,rows)
    a=scaler.transform(source)
    np.testing.assert_array_equal(a['x'][...,0][source.point_mask],source.raw_viq[...,0][source.point_mask])
    np.testing.assert_allclose(a['x'][...,1],np.where(source.point_mask,
        source.raw_viq[...,1]/source.nominal_capacity_Ah[:,None,None],0),atol=1e-7)
    assert a['x'].shape==(169,100,64,4) and a['policy'].shape==(169,28)
    assert not scaler.to_dict()['x_is_z_scored']
    assert not (source.point_mask & ~source.drift_mask).any()
    assert source.policy_mask[:,POLICY_COLUMNS.index('first_switch_soc')].sum()==124


def test_normalization_ignores_future_labels_and_nontrain_cells(source):
    rows=source.rows(load_registration()['source_cell_roles']['source_train'])
    scaler=SourceOnlyScaler().fit(source,rows)
    altered=replace(source,targets=np.full_like(source.targets,np.nan),observed=~source.observed)
    assert SourceOnlyScaler().fit(altered,rows).to_dict()==scaler.to_dict()
    a,b=scaler.transform(source),scaler.transform(altered)
    for name in ('x','x_z','policy','policy_mask','global_features_z','local_features_z'):
        np.testing.assert_array_equal(a[name],b[name])
    raw=source.raw_viq.copy()
    policy=source.policy.copy()
    heldout=sorted(set(range(len(source.ids)))-set(rows.tolist()))
    raw[heldout]*=10000
    policy[heldout]*=10000
    changed=replace(source,raw_viq=raw,policy=policy)
    assert SourceOnlyScaler().fit(changed,rows).to_dict()==scaler.to_dict()


def test_unseen_policy_is_dropped_and_constant_nominal_has_physical_scale(source):
    rows=source.rows(load_registration()['source_cell_roles']['source_train'])
    scaler=SourceOnlyScaler().fit(source,rows)
    policy=source.policy.copy()
    mask=source.policy_mask.copy()
    unknown=POLICY_COLUMNS.index('formation_temperature_C')
    policy[:,unknown]=50
    mask[:,unknown]=True
    policy[:,0]=.24
    altered=replace(source,policy=policy,policy_mask=mask)
    result=scaler.transform(altered)
    assert not result['policy_mask'][:,unknown].any()
    assert (result['policy'][:,unknown]==0).all()
    assert np.max(np.abs(result['policy'][:,0]))<1
    assert POLICY_PHYSICAL_SCALES[0]==1


def test_stanford_prefix_is_invariant_to_all_suffix_contents():
    t=np.linspace(0,1,96)
    early=[]
    for cycle in range(1,101):
        early.append(dict(cycle_number=cycle,voltage_in_V=4.2-.9*t,current_in_A=np.full(96,-.18),
            discharge_capacity_in_Ah=.24*(1-cycle*.0001)*(t**1.2)))
    prior=pd.read_csv(ROOT/'results/extension_stanford/prior_context.csv').iloc[0].to_dict()
    prior['cell_id']='ENGINEERING_ONLY'
    base=dict(cell_id='ENGINEERING_ONLY',nominal_capacity_in_Ah=.24,cycle_data=early)
    original=extract_stanford_prefix(base,prior)
    contaminated=copy.copy(base)
    contaminated['cycle_data']=early+[dict(cycle_number=i,voltage_in_V=[float('nan')],current_in_A=[1e20],
        discharge_capacity_in_Ah=[-1e30]) for i in (5000,501,101)]
    altered=extract_stanford_prefix(contaminated,prior)
    for name in ('raw_viq','point_mask','cycle_ids','drift_q','drift_mask','anchor','policy','policy_mask'):
        np.testing.assert_array_equal(original[name],altered[name])
    for name in original['physics']:
        np.testing.assert_array_equal(original['physics'][name],altered['physics'][name])
