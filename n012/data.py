"""N0 read-only data adapters. All normalizers are fitted on TRAIN CELLS only.

MATR: use the repository's verified frozen loader (no duplicate science code).
HUST: use the independently qualified raw-D3 view and its exact future records.
Stanford: fail closed until a certified compatible tensor + cycle contract exists.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class BatteryBundle:
    domain: str
    ids: list[str]
    x: np.ndarray
    mask: np.ndarray
    cycle: np.ndarray
    policy: np.ndarray
    policy_mask: np.ndarray
    anchor: np.ndarray
    query: np.ndarray
    targets: np.ndarray
    observed: np.ndarray
    eligible: np.ndarray
    folds: dict
    provenance: dict
    cell_metadata: pd.DataFrame | None = None


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def checked_files(root, rows):
    records = []
    for row in rows:
        path = (Path(root) / row['path']).resolve()
        if file_hash(path) != row['sha256']:
            raise RuntimeError('Source identity changed at ' + str(path))
        records.append({'path': str(path), 'sha256': row['sha256'], 'bytes': path.stat().st_size})
    return records


def _assemble(archive, identifiers, eligible, future, folds, domain, provenance):
    # The physics-resolved local arrays remain untouched, reusable for later N3.
    # N1/N2 use identical four channels: V, I, absolute Q, early delta-Q.
    raw=np.asarray(archive["raw_viq"],dtype=np.float32)
    ref=np.asarray(archive["reference_viq"],dtype=np.float32)
    mask=np.asarray(archive["point_mask"],dtype=bool)
    cyc=np.asarray(archive["cycle_ids"],dtype=np.float32)
    policy=np.asarray(archive["policy"],dtype=np.float32)
    pm=np.asarray(archive["policy_mask"],dtype=bool)
    anchor=np.asarray(archive["anchor_soh100"],dtype=np.float32)
    if raw.shape != (len(identifiers),100,64,3) or ref.shape!=raw.shape or mask.shape!=raw.shape[:3]:
        raise ValueError("VIQ shape contradicts registered N0 input contract")
    if cyc.shape!=(len(identifiers),100):
        raise ValueError("cycle identity shape mismatch")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicated physical cell identity')
    if not np.isfinite(anchor[np.asarray(eligible,dtype=bool)]).all() or (anchor[np.asarray(eligible,dtype=bool)] < 0).any():
        raise ValueError("missing early SOH100 anchor")
    if not np.isfinite(policy[pm]).all():
        raise ValueError('Nonfinite available policy')
    valid=np.broadcast_to(mask[...,None],raw.shape)
    if not np.isfinite(raw[valid]).all() or not np.isfinite(ref[valid]).all():
        raise ValueError("nonfinite observed early values")
    x=np.concatenate((raw, (raw[...,2:3]-ref[...,2:3])),axis=-1)
    x=np.where(mask[...,None],x,0).astype(np.float32)
    supported_cycles = cyc[mask.any(axis=-1)]
    if not (np.isfinite(supported_cycles).all() and
            ((supported_cycles >= 1) & (supported_cycles <= 100) &
             (supported_cycles == np.floor(supported_cycles))).all()):
        raise ValueError('Early cycles must be finite original integers from1 to100')
    for cell in range(len(identifiers)):
        if (np.diff(cyc[cell, mask[cell].any(axis=-1)]) <= 0).any():
            raise ValueError('Original prefix cycle order is not strictly increasing')
    if not {"cell_uid","original_cycle_id","soh_nominal","capacity_valid"}.issubset(future.columns):
        raise ValueError("future table lacks required source semantics")
    future=future.loc[future.capacity_valid & (future.original_cycle_id>100)].copy()
    if future.empty:
        raise ValueError("no valid future observations")
    if not np.isfinite(future.soh_nominal.to_numpy()).all():
        raise ValueError("nonfinite observed future SOH")
    actual_cycles = future.original_cycle_id.to_numpy(float)
    if not np.isfinite(actual_cycles).all() or not (actual_cycles == np.floor(actual_cycles)).all():
        raise ValueError('Future physical cycle IDs must be finite integers')
    query=np.arange(101,int(future.original_cycle_id.max())+1,dtype=np.float32)
    y=np.zeros((len(identifiers),len(query)),dtype=np.float32)
    obs=np.zeros_like(y,dtype=bool)
    ix={uid:i for i,uid in enumerate(identifiers)}
    for row in future.itertuples(index=False):
        if row.cell_uid not in ix:
            raise ValueError(f"future cell {row.cell_uid} absent from prefix table")
        i=ix[row.cell_uid];j=int(row.original_cycle_id)-101
        if obs[i,j]:
            raise ValueError(f"duplicated actual future cycle: {row.cell_uid} {row.original_cycle_id}")
        y[i,j]=float(row.soh_nominal); obs[i,j]=True
    for fold,roles in folds.items():
        if not set(roles)=={"train","validation","test"}:
            raise ValueError(f"invalid split roles: {fold}")
        s=[set(roles[k]) for k in ("train","validation","test")]
        if any(len(roles[k]) != len(set(roles[k])) for k in roles):
            raise ValueError('Repeated UID within a split role')
        if any(s[a]&s[b] for a,b in ((0,1),(0,2),(1,2))):
            raise ValueError(f"physical cell overlap in {fold}")
        if set.union(*s) != set(identifiers):
            raise ValueError(f"split does not partition the complete physical cohort in {fold}")
    if not (obs[np.asarray(eligible, bool)].sum(axis=1) > 0).all():
        raise ValueError('Qualified cell has no observed future target')
    return BatteryBundle(domain,identifiers,x,mask,cyc,policy,pm,anchor,query,y,obs,
                         np.asarray(eligible,dtype=bool),folds,provenance)


def load_matr(repo_root):
    repo_root=Path(repo_root)
    from scripts.unified_matr.train_seed42 import load_frozen_data
    root=repo_root/"data/unified_matr_seed42"
    inputs, ids, eligible, cells, query, targets, observed, rate_targets, rate_mask, split, freeze=load_frozen_data(root)
    # Rebuild future from source only for the shared adapter. Verified legacy target arrays
    # are compared to our output as a mandatory exact guard.
    future=pd.read_parquet(root/"future_soh.parquet")
    bundle=_assemble(inputs,ids,eligible,future,split["splits"],"MATR",{
        "source":"original-axis 169-cell MATR", "data_freeze":"DATA_FREEZE_PASS",
        "historical_exposure":True,
        "freeze_file":str(root/"DATA_FREEZE.json")})
    if bundle.query.shape != query.shape or not np.array_equal(bundle.query,query):
        raise RuntimeError("query cycles disagree with certified legacy loader")
    if not np.array_equal(bundle.observed,observed) or not np.array_equal(bundle.targets[observed],targets[observed]):
        raise RuntimeError("future source/target mismatch with certified loader")
    bundle.provenance['files'] = checked_files(root, freeze['files'])
    bundle.provenance['files'].append({'path':str((root/'DATA_FREEZE.json').resolve()),
                                       'sha256':file_hash(root/'DATA_FREEZE.json')})
    bundle.cell_metadata = cells.copy()
    return bundle


def _hust_folds(ids,seed=42,k=5,val_fraction=.2):
    # Cell-disjoint fixed deterministic development folds; exact HUST protocol combinations
    # are singletons, so this is NOT evidence of within-protocol cell variance.
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(ids))
    bins=np.array_split(perm,k)
    folds={}
    for fold in range(k):
        test=bins[fold]
        available=np.asarray(sorted(set(range(len(ids)))-set(test)))
        vrng=np.random.default_rng(seed+1000+fold)
        vrng.shuffle(available)
        nv=max(1,round(len(available)*val_fraction))
        roles={"train":available[nv:],"validation":available[:nv],"test":test}
        folds[f"cell_fold_{fold}"]={role:[ids[j] for j in indices] for role,indices in roles.items()}
    return folds


def load_hust(repo_root):
    repo_root=Path(repo_root)
    root=repo_root/"data/hust_raw_idea_v2"
    qual=json.loads((root/"DATA_QUALIFICATION.json").read_text(encoding="utf-8"))
    if qual.get("status")!="DATA_QUALIFICATION_PASS":
        raise RuntimeError("HUST raw input not independently qualified")
    records = checked_files(repo_root, qual['records'])
    complete = json.loads((root/'DATASET_COMPLETE.json').read_text(encoding='utf-8'))
    records.extend(checked_files(root, complete['files']))
    records.append({'path':str((root/'DATA_QUALIFICATION.json').resolve()),
                    'sha256':file_hash(root/'DATA_QUALIFICATION.json')})
    with np.load(root/"views/actual2C_loaded_response/early_inputs.npz",allow_pickle=False) as archive:
        ids=archive["cell_uid"].astype(str).tolist()
        eligible=archive["input_eligible"].astype(bool)
        d={k:archive[k].copy() for k in ("raw_viq","reference_viq","point_mask","cycle_ids",
                                          "policy","policy_mask","anchor_soh100")}
    future=pd.read_parquet(root/"future_soh.parquet")
    bundle = _assemble(d,ids,eligible,future,_hust_folds(ids),"HUST",{
        "source":"qualified actual-2C D3 view, prefix raw-cycle identity",
        "data_freeze":"DATA_QUALIFICATION_PASS","historical_exposure":True,
        "same_protocol_replicates":False,
        "eol80":"all 77 events right-censored", 'files':records,
        'measurement_limitations':qual['measurement_limitations']})
    cells = pd.read_parquet(root/'cells.parquet').set_index('cell_uid').loc[ids]
    if not np.array_equal(bundle.anchor,cells.anchor_soh100.to_numpy(np.float32)):
        raise ValueError('HUST prefix anchor differs from qualified metadata')
    bundle.cell_metadata = cells
    return bundle


def load_stanford(repo_root):
    # The GitHub repository includes Stanford exploration results, not a certified
    # frozen 100x64 raw-axis trajectory dataset comparable to the existing N0 loaders.
    raise RuntimeError("N0_STANFORD_BLOCKED: Stanford_2 supplier cycle IDs are BEEP-reindexed; "
                       "a full original-cycle/measurement-compatible N0 input tensor "
                       "and explicitly qualified future SOH file do not exist in "
                       "the current battery repository. Never relabel supplier-cycle K100 "
                       "as original-cycle K100. Preserve the 181-cell inventory.")


def load_bundle(repo_root,domain):
    return {"MATR":load_matr,"HUST":load_hust,"Stanford":load_stanford}[domain](repo_root)


class TrainOnlyScaler:
    """Each channel fit on TRAIN physical cells and supported VIQ points only."""
    def fit(self,bundle,train_idx):
        x=bundle.x[train_idx].astype(np.float64)
        mask=np.broadcast_to(bundle.mask[train_idx][...,None],x.shape)
        count=mask.sum(axis=(0,1,2))
        self.x_mean=np.divide(np.where(mask,x,0).sum((0,1,2)),count,out=np.zeros(4),where=count>0)
        var=np.divide(np.where(mask,(x-self.x_mean)**2,0).sum((0,1,2)),count,
                      out=np.zeros(4),where=count>0)
        self.x_std=np.maximum(np.sqrt(var),1e-5)
        p=bundle.policy[train_idx].astype(np.float64);m=bundle.policy_mask[train_idx]
        count=m.sum(axis=0)
        self.p_mean=np.divide(np.where(m,p,0).sum(0),count,out=np.zeros(p.shape[-1]),where=count>0)
        var=np.divide(np.where(m,(p-self.p_mean)**2,0).sum(0),count,
                      out=np.zeros(p.shape[-1]),where=count>0)
        self.p_std=np.maximum(np.sqrt(var),1e-5)
        return self

    def apply(self,bundle):
        x=np.where(bundle.mask[...,None],(bundle.x-self.x_mean)/self.x_std,0).astype(np.float32)
        p=np.where(bundle.policy_mask,(bundle.policy-self.p_mean)/self.p_std,0).astype(np.float32)
        return x,p

    def to_dict(self):
        return {k:np.asarray(getattr(self,k)).tolist() for k in ("x_mean","x_std","p_mean","p_std")}


def role_indices(bundle,fold):
    idx={uid:i for i,uid in enumerate(bundle.ids)}
    return {role:np.asarray([idx[uid] for uid in bundle.folds[fold][role] if bundle.eligible[idx[uid]]],dtype=np.int64)
            for role in ("train","validation","test")}


def synthetic_bundle(n=24,seed=42):
    rng=np.random.default_rng(seed)
    ids=[f"SYN_{i:03d}" for i in range(n)]
    v=np.linspace(3.3,2.8,64,dtype=np.float32)
    cycles=np.arange(1,101,dtype=np.float32)
    x=np.zeros((n,100,64,4),np.float32)
    for i in range(n):
        factor=(i+1)/n
        x[i,:,:,0]=v[None,:]
        x[i,:,:,1]=-4.4
        x[i,:,:,2]=(3.3-v)[None,:]/0.5*(1-0.006*factor*cycles[:,None]/100)
        x[i,:,:,3]=-0.008*factor*cycles[:,None]/100
    future=np.arange(101,301,dtype=np.float32)
    y=np.stack([0.99*np.exp(-((0.006+(i/n)*0.015)*(future-100)/1000)) for i in range(n)]).astype(np.float32)
    folds=_hust_folds(ids,k=3)
    return BatteryBundle("SYNTHETIC",ids,x,np.ones((n,100,64),bool),np.tile(cycles,(n,1)),
                         rng.normal(size=(n,15)).astype(np.float32),np.ones((n,15),bool),
                         np.full(n,0.99,np.float32),future,y,np.ones_like(y,bool),
                         np.ones(n,bool),folds,{"source":"synthetic smoke only; never a scientific result"})
