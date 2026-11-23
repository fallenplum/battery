import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'runtime/drs_peft'; OUT=ROOT/'results/drs_peft/e1_domain_shift'; OUT.mkdir(parents=True,exist_ok=True)
paths={'MATR':R/'e1_domain_shift/MATR.npz','HUST':R/'e1_domain_shift/HUST.npz','CALCE':R/'blind_calce_v2/PAIRED_PREFIXES.npz','SNL':R/'blind_snl_v1/PAIRED_PREFIXES.npz','MICH':R/'blind_mich_v1/PAIRED_PREFIXES.npz','RWTH':R/'blind_rwth_v1/PAIRED_PREFIXES.npz'}
domains={}
for d,p in paths.items():
    if p.exists():
        with np.load(p,allow_pickle=False) as z: domains[d]={k:z[k] for k in z.files}
if 'MATR' not in domains: raise RuntimeError('MATR cache missing')

def flat(data):
    x=data['real_features'][...,:12].astype(float); m=data['feature_mask'][...,:12].astype(bool)
    return x.reshape(-1,12),m.reshape(-1,12)
def cells(data):
    x=data['real_features'][...,:12].astype(float); m=data['feature_mask'][...,:12].astype(bool); x=np.where(m,x,np.nan)
    return np.concatenate((np.nanmean(x,1),np.nanstd(x,1),np.nanmean(x[:,-20:],1)-np.nanmean(x[:,:20],1)),1)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def mmd_rbf(a,b):
    a=a[:500]; b=b[:500]; z=np.concatenate((a,b)); dist=((z[:,None]-z[None,:])**2).sum(-1); sigma=np.sqrt(np.median(dist[dist>0])+1e-12); k=np.exp(-dist/(2*sigma*sigma)); n=len(a); m=len(b); return float(k[:n,:n].mean()+k[n:,n:].mean()-2*k[:n,n:].mean())

ref,rm=flat(domains['MATR']); ref=ref[rm[:,0]]; rng=np.random.default_rng(42)
if len(ref)>5000: ref=ref[rng.choice(len(ref),5000,False)]
mu=ref.mean(0); cov=np.cov(ref,rowvar=False)+np.eye(12)*1e-6; inv=np.linalg.pinv(cov)
stats=[]; shifts=[]; corrs=[]; feats=[]; labs=[]
for d,data in domains.items():
    x,m=flat(data); safe=np.where(m,x,np.nan); q=np.nanquantile(safe,[.01,.25,.5,.75,.99],axis=0)
    for c in range(12):
        v=safe[:,c]; v=v[np.isfinite(v)]; rs=ref[:,c]
        if len(v)>5000: v=v[rng.choice(len(v),5000,False)]
        if len(rs)>5000: rs=rs[rng.choice(len(rs),5000,False)]
        k=ks_2samp(rs,v)
        stats.append(dict(domain=d,channel=c,count=len(v),mean=float(v.mean()),std=float(v.std()),q01=float(q[0,c]),q25=float(q[1,c]),q50=float(q[2,c]),q75=float(q[3,c]),q99=float(q[4,c])))
        shifts.append(dict(domain=d,channel=c,wasserstein=float(wasserstein_distance(rs,v)),ks=float(k.statistic),ks_p=float(k.pvalue)))
    arr=np.nan_to_num(safe,nan=np.nanmean(safe,0)); cm=np.corrcoef(arr,rowvar=False); corrs.append(dict(domain=d,correlation=json.dumps(cm.tolist())))
    dm=np.nanmean(safe,0)-mu; cc=np.cov(arr,rowvar=False); shifts.append(dict(domain=d,channel='ALL',mahalanobis=float(np.sqrt(max(dm@inv@dm,0))),coral=float(np.linalg.norm(cc-cov)/(np.linalg.norm(cov)+1e-12)),mmd_rbf=mmd_rbf(ref,arr)))
    feats.append(cells(data)); labs.extend([d]*len(feats[-1]))
X=np.concatenate(feats); y=np.asarray(labs); auc=[]
for d in sorted(set(y)):
    yy=(y==d).astype(int)
    if yy.sum()<2 or yy.sum()==len(yy): continue
    n=min(5,int(yy.sum()),int((1-yy).sum())); pred=np.zeros(len(yy))
    for tr,te in StratifiedKFold(n,shuffle=True,random_state=42).split(X,yy):
        model=LogisticRegression(max_iter=2000,class_weight='balanced'); model.fit(X[tr],yy[tr]); pred[te]=model.predict_proba(X[te])[:,1]
    auc.append(dict(domain=d,auc=float(roc_auc_score(yy,pred)),cells=int(yy.sum())))
pd.DataFrame(stats).to_csv(OUT/'BRIDGE_CHANNEL_STATS.csv',index=False); pd.DataFrame(shifts).to_csv(OUT/'BRIDGE_SHIFT_METRICS.csv',index=False); pd.DataFrame(corrs).to_csv(OUT/'BRIDGE_CHANNEL_CORRELATIONS.csv',index=False); pd.DataFrame(auc).to_csv(OUT/'BRIDGE_DOMAIN_CLASSIFIER_AUC.csv',index=False)
receipt={'status':'COMPLETE_E1_BRIDGE_DOMAIN_SHIFT_AUDIT','domains':sorted(domains),'future_labels_read':0,'metrics':['Wasserstein','KS','Mahalanobis','CORAL','domain_classifier_AUC'],'cache_shapes':{d:list(v['real_features'].shape) for d,v in domains.items()}}
(OUT/'BRIDGE_SHIFT_RECEIPT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(pd.DataFrame(auc).to_string(index=False))
