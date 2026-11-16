"""Qualified battery arrays and source-only, physical-unit-aware preprocessing."""
from __future__ import annotations

import argparse
from functools import lru_cache
import gc
from dataclasses import dataclass
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from n012.data import load_bundle as load_qualified_bundle
from scripts.unified_matr.physics_features import GLOBAL_COLUMNS, LOCAL_COLUMNS, PhysicsConfig, extract_prefix_features

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'runtime/drs_peft/data_v1'
DATASET_ROOT = Path(r'D:\develop\patchtst\BatteryLife\dataset')
QUERY = np.arange(101,5001,dtype=np.float32)
POLICY_COLUMNS = (
    'nominal_capacity_Ah','window_voltage_min_V','window_voltage_max_V','chamber_design_temperature_C',
    'charge_stage_count','charge_rate_1_C','charge_rate_2_C','charge_rate_3_C','charge_rate_4_C',
    'discharge_final_design_rate_C','charge_design_voltage_limit_V','discharge_design_voltage_limit_V',
    'first_switch_soc','fast_charge_end_soc','topoff_rate_C',
    'formation_temperature_C','formation_charge1_C','formation_transition_voltage_V','formation_charge2_C',
    'formation_verification_cycles','formation_post_rest_hours',
    'discharge_design_stage0_C','discharge_design_stage1_C','discharge_design_stage2_C','discharge_design_stage3_C',
    'observed_prefix_discharge_final_C','observed_prefix_charge_initial_C','discharge_design_observed_conflict')
POLICY_UNITS = ('Ah','V','V','degC','count','C-rate','C-rate','C-rate','C-rate','C-rate','V','V',
                'fraction','fraction','C-rate','degC','C-rate','V','C-rate','count','hours',
                'C-rate','C-rate','C-rate','C-rate','C-rate','C-rate','boolean')
POLICY_PHYSICAL_SCALES = np.asarray([1,1,1,10,1,1,1,1,1,1,1,1,1,1,1,10,1,1,1,1,72,1,1,1,1,1,1,1],float)
PHYSICS_PAIRS = {'global_features':'global_mask','local_residual':'local_mask','local_features':'local_feature_mask'}


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


@dataclass
class BatteryData:
    domain: str
    ids: list[str]
    raw_viq: np.ndarray
    point_mask: np.ndarray
    cycle_ids: np.ndarray
    drift_q: np.ndarray
    drift_mask: np.ndarray
    policy: np.ndarray
    policy_mask: np.ndarray
    anchor: np.ndarray
    query: np.ndarray
    targets: np.ndarray
    observed: np.ndarray
    nominal_capacity_Ah: np.ndarray
    voltage_grid_V: np.ndarray
    protocol_ids: np.ndarray
    physics: dict
    cell_metadata: pd.DataFrame
    provenance: dict

    @property
    def x(self):
        """Canonical units, never source-z-scored physical coordinates."""
        result=np.concatenate((self.raw_viq,self.drift_q[...,None]),axis=-1).astype(np.float32)
        result[...,1:]/=self.nominal_capacity_Ah[:,None,None,None]
        return np.where(self.feature_mask,result,0).astype(np.float32)

    @property
    def feature_mask(self):
        return np.concatenate((np.repeat(self.point_mask[...,None],3,axis=-1),self.drift_mask[...,None]),axis=-1)

    @property
    def mask(self):
        return self.point_mask

    @property
    def cycle(self):
        return self.cycle_ids

    @property
    def y(self):
        return self.targets

    def rows(self,ids):
        lookup={uid:i for i,uid in enumerate(self.ids)}
        return np.asarray([lookup[uid] for uid in ids],dtype=np.int64)


def true_prefix_drift(q,mask,cycles):
    q=np.asarray(q,float)
    mask=np.asarray(mask,bool)
    reference_mask=mask & (cycles[...,None]>=1) & (cycles[...,None]<=10)
    count=reference_mask.sum(axis=1)
    with np.errstate(invalid='ignore'):
        reference=np.nanmedian(np.where(reference_mask,q,np.nan),axis=1)
    supported=mask & (count[:,None]>=5) & np.isfinite(reference[:,None])
    drift=np.where(supported,q-reference[:,None],0).astype(np.float32)
    return drift,supported


def _common_policy(domain,ids,nominal,grid,original,original_mask,names):
    values=np.zeros((len(ids),len(POLICY_COLUMNS)),np.float32)
    supported=np.zeros_like(values,bool)
    index={name:j for j,name in enumerate(POLICY_COLUMNS)}
    old={name:j for j,name in enumerate(names)}
    def assign(name,value,mask=True):
        j=index[name]
        values[:,j]=np.where(mask,value,0)
        supported[:,j]=mask
    def transfer(name,source):
        j=old[source]
        assign(name,original[:,j],original_mask[:,j])
    assign('nominal_capacity_Ah',nominal)
    assign('window_voltage_min_V',float(grid[0]))
    assign('window_voltage_max_V',float(grid[-1]))
    if domain=='MATR':
        for name in ('charge_stage_count','fast_charge_end_soc','topoff_rate_C','chamber_design_temperature_C'):
            transfer(name,name)
        assign('first_switch_soc',original[:,old['first_switch_soc']],
               original_mask[:,old['first_switch_soc']] & (original[:,old['first_switch_soc_missing']]<.5))
        transfer('discharge_final_design_rate_C','discharge_rate_C')
        for stage in range(1,5):
            source=f'charge_rate_{stage}_C'
            j=old[source]
            available=original_mask[:,j] & (original[:,old[f'charge_rate_{stage}_missing']]<.5)
            assign(source,original[:,j],available)
    elif domain=='HUST':
        for name,source in {
            'charge_rate_1_C':'design_charge_initial_C','charge_rate_2_C':'design_charge_second_C',
            'charge_design_voltage_limit_V':'design_charge_voltage_limit_V',
            'discharge_final_design_rate_C':'design_discharge_stage3_C',
            'chamber_design_temperature_C':'design_chamber_setpoint_C',
            'observed_prefix_discharge_final_C':'observed_prefix_discharge_stage3_C',
            'observed_prefix_charge_initial_C':'observed_prefix_charge_initial_C',
            'discharge_design_observed_conflict':'discharge_final_design_observed_conflict',
            **{f'discharge_design_stage{k}_C':f'design_discharge_stage{k}_C' for k in range(4)}}.items():
            transfer(name,source)
        # Number of explicitly registered constant-current charge stages, excluding CV.
        assign('charge_stage_count',original_mask[:,old['design_charge_initial_C']].astype(float)+original_mask[:,old['design_charge_second_C']])
    elif domain=='Stanford_2':
        mapping={
            'charge_rate_1_C':'prior_charge_rate_C','discharge_final_design_rate_C':'prior_discharge_rate_C',
            'charge_design_voltage_limit_V':'prior_upper_voltage_V','discharge_design_voltage_limit_V':'prior_lower_voltage_V',
            'chamber_design_temperature_C':'prior_aging_chamber_setpoint_C',
            'formation_temperature_C':'prior_formation_temperature_C','formation_charge1_C':'prior_formation_charge1_C',
            'formation_transition_voltage_V':'prior_formation_transition_voltage_V','formation_charge2_C':'prior_formation_charge2_C',
            'formation_verification_cycles':'prior_verification_cycles','formation_post_rest_hours':'prior_postformation_rest_hours'}
        for name,source in mapping.items():
            transfer(name,source)
        assign('charge_stage_count',1.)
    if not np.isfinite(values[supported]).all():
        raise ValueError('Nonfinite known policy value')
    return values,supported


def _native_dataset(domain,root):
    bundle=load_qualified_bundle(root,domain)
    source=root/('data/unified_matr_seed42/early_viq_qv_64.npz' if domain=='MATR' else
                 'data/hust_raw_idea_v2/views/actual2C_loaded_response/early_inputs.npz')
    with np.load(source,allow_pickle=False) as archive:
        raw=archive['raw_viq'].copy()
        grid=archive['voltage_grid_V'].copy()
        physics={key:archive[key].copy() for pair in PHYSICS_PAIRS.items() for key in pair}
        if domain=='HUST':
            names=archive['policy_columns'].astype(str).tolist()
        else:
            names=json.loads((root/'data/unified_matr_seed42/dataset_manifest.json').read_text(encoding='utf-8'))['policy_columns']
    nominal=bundle.cell_metadata.loc[bundle.ids,'nominal_capacity_Ah'].to_numpy(np.float32)
    policy,policy_mask=_common_policy(domain,bundle.ids,nominal,grid,bundle.policy,bundle.policy_mask,names)
    drift,drift_mask=true_prefix_drift(raw[...,2],bundle.mask,bundle.cycle)
    targets=np.zeros((len(bundle.ids),len(QUERY)),np.float32)
    observed=np.zeros_like(targets,bool)
    if bundle.query.max()>QUERY.max():
        raise ValueError('Fixed global query grid omits observed native cycles; register a larger common grid')
    positions=bundle.query.astype(int)-101
    targets[:,positions]=bundle.targets
    observed[:,positions]=bundle.observed
    provenance=dict(domain=domain,coordinate='original_instrument_cycle',prefix_maximum=100,
        nominal_capacity_units='Ah',voltage_window_V=[float(grid[0]),float(grid[-1])],
        fourth_channel='actual Q_n(V) minus pointwise median Q(V) over supported cycles1..10',
        original_N012_fourth_channel_reused=False,source_files=bundle.provenance['files'],
        early_array_source=dict(path=source.relative_to(root).as_posix(),sha256=file_hash(source)),
        observed_future_points=int(observed.sum()),maximum_observed_cycle=int(bundle.query[bundle.observed.any(0)].max()),
        eligible_physical_cells=len(bundle.ids),right_censored_EOL80=domain=='HUST')
    return BatteryData(domain,bundle.ids,raw,bundle.mask,bundle.cycle,drift,drift_mask,policy,policy_mask,
        bundle.anchor,QUERY.copy(),targets,observed,nominal,grid,
        bundle.cell_metadata.loc[bundle.ids,'protocol_uid'].to_numpy(dtype=str),physics,bundle.cell_metadata.copy(),provenance)


def _validate(data):
    n=len(data.ids)
    if len(set(data.ids))!=n or data.raw_viq.shape!=(n,100,64,3):
        raise ValueError('Physical UID or prefix array shape invalid')
    if data.policy.shape!=(n,28) or data.targets.shape!=(n,len(QUERY)):
        raise ValueError('Known-policy schema or fixed future grid invalid')
    if not np.array_equal(data.query,QUERY) or not np.isfinite(data.targets[data.observed]).all():
        raise ValueError('Future target grid or supported target values invalid')
    if not np.isfinite(data.raw_viq[np.repeat(data.point_mask[...,None],3,axis=-1)]).all():
        raise ValueError('Nonfinite supported early electrical measurement')
    if not (data.nominal_capacity_Ah>0).all() or not np.isfinite(data.anchor).all():
        raise ValueError('Nominal capacity or early anchor unavailable')
    for key,mask in PHYSICS_PAIRS.items():
        if data.physics[key].shape!=data.physics[mask].shape or not np.isfinite(data.physics[key][data.physics[mask]]).all():
            raise ValueError('Invalid masked physical feature '+key)


def save_dataset(data,cache_root=CACHE):
    _validate(data)
    dest=Path(cache_root)/data.domain
    dest.mkdir(parents=True,exist_ok=True)
    values={name:getattr(data,name) for name in ('raw_viq','point_mask','cycle_ids','drift_q','drift_mask','policy','policy_mask',
        'anchor','query','targets','observed','nominal_capacity_Ah','voltage_grid_V','protocol_ids')}
    values.update(data.physics)
    values['ids']=np.asarray(data.ids)
    if any(value.dtype.kind=='O' for value in values.values()):
        raise ValueError('Object arrays are forbidden in the portable qualified cache')
    temporary=dest/'arrays.pending.npz'
    np.savez_compressed(temporary,**values)
    os.replace(temporary,dest/'arrays.npz')
    data.cell_metadata.to_csv(dest/'cell_metadata.csv',index=True)
    (dest/'adapter_source.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(status='QUALIFIED_FROZEN_DATA',domain=data.domain,physical_cells=len(data.ids),
        prefix_shape=list(data.raw_viq.shape),future_shape=list(data.targets.shape),
        policy_columns=list(POLICY_COLUMNS),policy_units=list(POLICY_UNITS),
        policy_physical_standard_deviation_floors=POLICY_PHYSICAL_SCALES.tolist(),
        canonical_x_units=['V','C-rate signed I/Ah','nominal SOH units Q/Ah','nominal SOH units true early drift/Ah'],
        source=data.provenance.get('source',data.provenance),adapter_sha256=file_hash(__file__),
        files=[dict(path=name,sha256=file_hash(dest/name)) for name in ('arrays.npz','cell_metadata.csv','adapter_source.py')])
    write_json(dest/'COMPLETE.json',manifest)
    return manifest


def load_dataset(domain,cache_root=CACHE,build_if_missing=True,root=ROOT,rebuild=False):
    domain='Stanford_2' if domain in ('Stanford','Stanford_2') else domain
    dest=Path(cache_root)/domain
    marker=dest/'COMPLETE.json'
    if rebuild or not marker.exists():
        if not build_if_missing:
            raise FileNotFoundError(marker)
        data=_native_dataset(domain,Path(root)) if domain in ('MATR','HUST') else build_stanford(Path(root))
        save_dataset(data,cache_root)
    manifest=json.loads(marker.read_text(encoding='utf-8'))
    for row in manifest['files']:
        if file_hash(dest/row['path'])!=row['sha256']:
            raise ValueError('Frozen DRS array cache changed: '+row['path'])
    with np.load(dest/'arrays.npz',allow_pickle=False) as archive:
        arrays={k:archive[k].copy() for k in archive.files}
    ids=arrays.pop('ids').astype(str).tolist()
    physics={k:arrays.pop(k) for pair in PHYSICS_PAIRS.items() for k in pair}
    meta=pd.read_csv(dest/'cell_metadata.csv',index_col=0)
    data=BatteryData(domain=domain,ids=ids,physics=physics,cell_metadata=meta,provenance=manifest,**arrays)
    _validate(data)
    return data


def _moments(values,mask,axes,floor):
    values=np.asarray(values,float)
    mask=np.asarray(mask,bool)
    count=mask.sum(axis=axes)
    mean=np.divide(np.where(mask,values,0).sum(axis=axes),count,out=np.zeros_like(count,dtype=float),where=count>0)
    variance=np.divide(np.where(mask,(values-mean)**2,0).sum(axis=axes),count,
        out=np.zeros_like(count,dtype=float),where=count>0)
    return dict(mean=np.asarray(mean).tolist(),std=np.maximum(np.sqrt(variance),floor).tolist(),
                supported=np.asarray(count).tolist(),physical_std_floor=np.asarray(floor).tolist())


def _physics_canonical(data,key):
    v=data.physics[key].astype(float).copy()
    nominal=data.nominal_capacity_Ah.astype(float)
    span=float(data.voltage_grid_V[-1]-data.voltage_grid_V[0])
    if key=='local_residual':
        return v/nominal[:,None,None]
    names=GLOBAL_COLUMNS if key=='global_features' else LOCAL_COLUMNS
    for j,name in enumerate(names):
        dimensionless=any(name.endswith(ending) for ending in ('signed_direction','direction_consistency','support_count','availability_fraction'))
        if dimensionless:
            continue
        if 'Ah2' in name:
            v[...,j]/=nominal.reshape((-1,)+(1,)*(v.ndim-2))**2
        elif 'Ah' in name:
            v[...,j]/=nominal.reshape((-1,)+(1,)*(v.ndim-2))
        elif name in ('absolute_residual_centroid_V','absolute_residual_peak_V'):
            v[...,j]=(v[...,j]-float(data.voltage_grid_V[0]))/span
        elif name.startswith('b_V'):
            v[...,j]/=span
    return v


class SourceOnlyScaler:
    """Fit only named source train cells. Canonical x remains in physical units."""
    def fit(self,data,rows):
        rows=np.asarray(rows,dtype=int)
        if len(rows)==0 or len(rows)!=len(set(rows.tolist())):
            raise ValueError('Source scaler requires unique, nonempty training cells')
        self.training_ids=[data.ids[i] for i in rows]
        self.stats={'x':_moments(data.x[rows],data.feature_mask[rows],(0,1,2),np.array([.1,1.,.05,.01])),
                    'policy':_moments(data.policy[rows],data.policy_mask[rows],(0,),POLICY_PHYSICAL_SCALES)}
        for key,mask in PHYSICS_PAIRS.items():
            v=_physics_canonical(data,key)
            axes=tuple(range(v.ndim-1)) if key!='local_residual' else (0,1,2)
            self.stats[key]=_moments(v[rows],data.physics[mask][rows],axes,1e-5)
        return self

    def transform(self,data):
        x=data.x
        out=dict(x=x,point_mask=data.point_mask,feature_mask=data.feature_mask,cycle_ids=data.cycle_ids,
                 anchor=data.anchor,nominal_capacity_Ah=data.nominal_capacity_Ah,
                 voltage_grid_V=np.broadcast_to(data.voltage_grid_V,(len(data.ids),64)),
                 raw_viq=data.raw_viq,drift_q=data.drift_q,drift_mask=data.drift_mask,
                 policy_raw=data.policy,policy_raw_mask=data.policy_mask)
        s=self.stats['x']
        out['x_z']=np.where(data.feature_mask,(x-np.asarray(s['mean']))/np.asarray(s['std']),0).astype(np.float32)
        s=self.stats['policy']
        m=data.policy_mask & (np.asarray(s['supported'])>0)[None]
        out['policy']=np.where(m,(data.policy-np.asarray(s['mean']))/np.asarray(s['std']),0).astype(np.float32)
        out['policy_mask']=m
        for key,mask in PHYSICS_PAIRS.items():
            out[key]=np.where(data.physics[mask],data.physics[key],0).astype(np.float32)
            out[mask]=data.physics[mask]
            s=self.stats[key]
            supported=np.asarray(s['supported'])>0
            m=data.physics[mask] & supported
            z=(_physics_canonical(data,key)-np.asarray(s['mean']))/np.asarray(s['std'])
            out[key+'_z']=np.where(m,z,0).astype(np.float32)
            out[key+'_z_mask']=m
        return out

    def to_dict(self):
        return dict(training_ids=self.training_ids,statistics=self.stats,
            canonical_x_units=['V','C-rate signed','nominal SOH','true prefix drift / nominal Ah'],
            x_is_z_scored=False,source_z_scored_x_key='x_z',policy_columns=list(POLICY_COLUMNS),
            source_all_missing_policy='drop across all domains: value0 and maskFalse',
            physics_raw_keys='retain original units; *_z is source-only standardized canonical feature geometry',
            physics_canonical_recipe='Ah quantities/nominal; Ah2 quantities/nominal2; b_V/window_span; centroid/peak=(V-window_min)/span')


@lru_cache(maxsize=1)
def _supplied_helpers():
    # The old standalone scanner has one sibling import. Restore the import path afterwards.
    scripts_path=str(ROOT/'scripts')
    inserted=scripts_path not in sys.path
    if inserted:
        sys.path.insert(0,scripts_path)
    try:
        from scripts.run_d0 import load_existing_pickle,capacity,current_reference,observed_curve,longest_run
    finally:
        if inserted:
            sys.path.remove(scripts_path)
    return load_existing_pickle,capacity,current_reference,observed_curve,longest_run


def extract_stanford_prefix(payload,prior):
    """Strict prefix-only function; all suffix contents and order are irrelevant."""
    _,capacity,current_reference,observed_curve,longest_run=_supplied_helpers()
    nominal=float(payload['nominal_capacity_in_Ah'])
    if nominal!=.24 or payload['cell_id']!=prior['cell_id']:
        raise ValueError('Stanford declared nominal capacity or physical metadata binding differs')
    early=sorted((c for c in payload['cycle_data'] if int(c['cycle_number'])<=100),key=lambda c:int(c['cycle_number']))
    if [int(c['cycle_number']) for c in early]!=list(range(1,101)):
        raise ValueError('Stanford supplied prefix must have each coordinate1..100 exactly once')
    grid=np.linspace(3.4,4.1,64)
    raw=np.zeros((100,64,3),np.float32)
    valid=np.zeros((100,64),bool)
    reference=current_reference(early,'discharge')
    setting=dict(direction='discharge',voltage_window=[3.4,4.1])
    reasons={}
    anchors=[]
    for c in early:
        number=int(c['cycle_number'])
        ratio=capacity(c,'discharge_capacity_in_Ah',nominal)
        if number>=96 and ratio is not None:
            anchors.append(ratio)
        curve,reason,_=observed_curve(c,setting,reference,nominal)
        reasons[reason]=reasons.get(reason,0)+1
        if curve is None:
            continue
        voltage=np.asarray(c['voltage_in_V'],float)
        current=np.asarray(c['current_in_A'],float)
        charge=np.asarray(c['discharge_capacity_in_Ah'],float)
        support=np.isfinite(voltage)&np.isfinite(current)&np.isfinite(charge)
        support &= np.abs(-current-reference)<=.05*reference
        indices=longest_run(support)
        order=np.argsort(voltage[indices],kind='stable')
        unique,position=np.unique(voltage[indices][order],return_index=True)
        selected=indices[order][position]
        if unique[0]>grid[0] or unique[-1]<grid[-1]:
            raise ValueError('Stanford prefix interpolation would extrapolate')
        raw[number-1,:,0]=grid
        raw[number-1,:,1]=np.interp(grid,unique,current[selected])
        raw[number-1,:,2]=np.interp(grid,unique,charge[selected])
        valid[number-1]=True
        reconstructed=(raw[number-1,:,2].astype(float)-raw[number-1,0,2])*-1
        if not np.allclose(reconstructed,curve,atol=3e-8,rtol=3e-6):
            raise ValueError('Observed-curve qualification and VIQ interpolation disagree')
    if not anchors or valid.any(1).sum()<80:
        raise ValueError('Stanford prefix anchor or minimum80 observed curves missing')
    cycles=np.arange(1,101,dtype=np.float32)
    drift,drift_mask=true_prefix_drift(raw[None,...,2],valid[None],cycles[None])
    q=np.where(valid,raw[...,2],np.nan)
    features=extract_prefix_features(q,valid.any(1),cycles,grid,PhysicsConfig(voltage_min=3.4,voltage_max=4.1))
    physics={}
    for value,mask in PHYSICS_PAIRS.items():
        physics[mask]=np.asarray(features[mask],bool)
        physics[value]=np.where(physics[mask],features[value],0).astype(np.float32)
    names=[name for name in prior if name.startswith('prior_')]
    original=np.array([[float(prior[name]) for name in names]],np.float32)
    policy,policy_mask=_common_policy('Stanford_2',[payload['cell_id']],np.array([nominal],np.float32),grid,
                                     original,np.isfinite(original),names)
    return dict(raw_viq=raw,point_mask=valid,cycle_ids=cycles,drift_q=drift[0],drift_mask=drift_mask[0],
        anchor=np.float32(np.median(anchors)),policy=policy[0],policy_mask=policy_mask[0],
        nominal_capacity_Ah=np.float32(nominal),physics=physics,
        diagnostics=dict(valid_curves=int(valid.any(1).sum()),curve_reasons=reasons,
            reference_current_A=float(reference),physics_reference_valid=bool(features['diagnostics']['reference_valid']),
            physical_cycle_axis_authenticated=False,prefix_coordinate='supplied_cycle_number'))


def build_stanford(root=ROOT,dataset_root=DATASET_ROOT):
    root,dataset_root=Path(root),Path(dataset_root)
    load_existing_pickle,capacity,_,_,_=_supplied_helpers()
    qualified=pd.read_csv(root/'results/d0/eligible_cells.csv')
    qualified=qualified[qualified.family.eq('Stanford_2')].set_index('cell_id')
    context=pd.read_csv(root/'results/extension_stanford/prior_context.csv').set_index('cell_id',drop=False)
    ids=sorted(qualified.index.tolist())
    if len(ids)!=181 or len(set(ids))!=181 or context.loc[ids,'source_device_id_sha256'].nunique()!=181:
        raise ValueError('Stanford physical identity chain must be181 suppliedUIDs and181 unique hardware hashes')
    if set(context.index)!=set(ids):
        raise ValueError('Formation-prior and supplied-cohort identities differ')
    prefix_arrays={name:[] for name in ('raw_viq','point_mask','cycle_ids','drift_q','drift_mask','policy','policy_mask','anchor','nominal_capacity_Ah')}
    physical={key:[] for pair in PHYSICS_PAIRS.items() for key in pair}
    targets=np.zeros((len(ids),len(QUERY)),np.float32)
    observed=np.zeros_like(targets,bool)
    metadata,source_records=[],[]
    maximum_cycle=0
    for row,uid in enumerate(ids):
        before=dataset_root/qualified.loc[uid,'provided_file']
        payload,digest=load_existing_pickle(before)
        if digest!=qualified.loc[uid,'provided_file_sha256']:
            raise ValueError('Supplied source bytes changed after exhaustive prior audit: '+uid)
        prior=context.loc[uid].to_dict()
        # Prefix extraction is completed before traversing future target values.
        prefix=extract_stanford_prefix(payload,prior)
        for name in prefix_arrays:
            prefix_arrays[name].append(prefix[name])
        for name in physical:
            physical[name].append(prefix['physics'][name])
        numbers=[int(c['cycle_number']) for c in payload['cycle_data']]
        if numbers!=sorted(set(numbers)):
            raise ValueError('Repeated or reordered supplied cycle identities in '+uid)
        maximum_cycle=max(maximum_cycle,max(numbers))
        if maximum_cycle>int(QUERY[-1]):
            raise ValueError('Supplied cohort exceeds fixed global5000 grid; register a larger common grid')
        event=None
        valid_future=0
        for cycle in payload['cycle_data']:
            number=int(cycle['cycle_number'])
            if number<=100:
                continue
            ratio=capacity(cycle,'discharge_capacity_in_Ah',float(payload['nominal_capacity_in_Ah']))
            if ratio is None:
                continue
            targets[row,number-101]=ratio
            observed[row,number-101]=True
            valid_future+=1
            if ratio<=.8 and event is None:
                event=number
        if valid_future==0:
            raise ValueError('Supplied cell has no observed valid future target: '+uid)
        original_meta={k:v for k,v in payload.items() if k!='cycle_data'}
        metadata.append(dict(cell_uid=uid,protocol_uid=prior['protocol_group'],
            source_device_id_sha256=prior['source_device_id_sha256'],
            nominal_capacity_Ah=float(payload['nominal_capacity_in_Ah']),anchor_soh100=float(prefix['anchor']),
            coordinate='supplied_cycle_number',physical_cycle_axis_authenticated=False,
            prefix_valid_curves=prefix['diagnostics']['valid_curves'],supplied_last_cycle=max(numbers),
            future_observed_capacity_count=valid_future,eol80_event_observed=event is not None,
            eol80_first_observed_crossing_cycle=event,right_censored=event is None,
            supplied_metadata_json=json.dumps(original_meta,ensure_ascii=False),
            prior_context_json=json.dumps(prior,ensure_ascii=False),
            prefix_qualification_json=json.dumps(prefix['diagnostics'],ensure_ascii=False)))
        source_records.append(dict(path=str(before),sha256=digest,bytes=before.stat().st_size,
            cell_uid=uid,source_device_id_sha256=prior['source_device_id_sha256']))
        if (row+1)%10==0 or row+1==len(ids):
            print(json.dumps(dict(event='STANFORD_SUPPLIED_BUILD',completed=row+1,total=len(ids))),flush=True)
        del payload,prefix
        gc.collect()
    values={name:np.asarray(parts) for name,parts in prefix_arrays.items()}
    features={name:np.stack(parts) for name,parts in physical.items()}
    meta=pd.DataFrame(metadata).set_index('cell_uid')
    provenance=dict(domain='Stanford_2',coordinate='supplied_cycle_number',prefix_maximum=100,
        physical_cycle_axis_authenticated=False,interpolated_supplier_payload=True,
        unknown_original_preprocessor_generation_binding=True,nominal_capacity_units='Ah',nominal_capacity_Ah=.24,
        chemistry='declared graphite / LiNi0.5Mn0.3Co0.2O2, pouch',voltage_window_V=[3.4,4.1],
        current_reference='prefix-only upper20percent median discharge current in cycles1..5',
        target_capacity='phase-confirmed maximum discharge Ah minus finite full-record baseline, divided by declared0.24Ah',
        anchor='median supported supplied cycles96..100 capacities / declared0.24Ah',
        future_policy='known design context retained; real future execution not independently authenticated',
        physics_scope='mathematical loaded-response decomposition of supplier-interpolated curves; electrode mechanisms unmeasured',
        fourth_channel='true Q_n(V) minus median supported Q(V) over supplied cycles1..10',
        source_files=source_records,observed_future_points=int(observed.sum()),maximum_observed_cycle=maximum_cycle,
        eligible_physical_cells=len(ids),original_metadata_fields=sorted(original_meta),
        metadata_and_identity_files=[dict(path=p,sha256=file_hash(root/p)) for p in
            ('results/d0/eligible_cells.csv','results/extension_stanford/prior_context.csv','results/extension_stanford/source_qualification.json')],
        recipe_files=[dict(path=p,sha256=file_hash(root/p)) for p in
            ('scripts/run_d0.py','scripts/physics_targets.py','scripts/unified_matr/physics_features.py')])
    return BatteryData(domain='Stanford_2',ids=ids,query=QUERY.copy(),targets=targets,observed=observed,
        voltage_grid_V=np.linspace(3.4,4.1,64),protocol_ids=meta.loc[ids,'protocol_uid'].to_numpy(dtype=str),
        physics=features,cell_metadata=meta,provenance=provenance,**values)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--domain',choices=['MATR','HUST','Stanford_2'],required=True)
    parser.add_argument('--cache-root',type=Path,default=CACHE)
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    data=load_dataset(args.domain,args.cache_root,rebuild=args.rebuild)
    print(json.dumps(dict(status='READY',domain=data.domain,cells=len(data.ids),input_shape=list(data.x.shape),
        policy_dim=len(POLICY_COLUMNS),future_observations=int(data.observed.sum()),
        coordinate=data.provenance['source']['coordinate'],cache=str(args.cache_root/data.domain))))


if __name__=='__main__':
    main()
