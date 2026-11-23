import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd,torch
from models.drs_peft.self_reference_model import SharedBatteryModel
from scripts.drs_peft.self_reference_training import load_common_state
from scripts.drs_peft.transfer_scope_reference import BACKBONE

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'runtime/drs_peft'; OUT=ROOT/'results/drs_peft/e1_domain_shift'; OUT.mkdir(parents=True,exist_ok=True)
paths={'MATR':R/'e1_domain_shift/MATR.npz','HUST':R/'e1_domain_shift/HUST.npz','CALCE':R/'blind_calce_v2/PAIRED_PREFIXES.npz','SNL':R/'blind_snl_v1/PAIRED_PREFIXES.npz','MICH':R/'blind_mich_v1/PAIRED_PREFIXES.npz','RWTH':R/'blind_rwth_v1/PAIRED_PREFIXES.npz'}
m=SharedBatteryModel(BACKBONE); load_common_state(m,torch.load(R/'self_reference_v1/COMMON_FOLD0.pt',map_location='cpu',weights_only=True)); m.eval(); rows=[]
with torch.no_grad():
 for d,p in paths.items():
  if not p.exists(): continue
  with np.load(p,allow_pickle=False) as z: f=torch.from_numpy(z['real_features'][...,:12]); mask=torch.from_numpy(z['feature_mask'][...,:12]); cyc=torch.from_numpy(z['cycles']); anchor=torch.from_numpy(z['real_anchor'])
  chunks=[]
  for s in range(0,len(f),16):
   x=f[s:s+16]; mm=mask[s:s+16]; cc=cyc[s:s+16]; aa=anchor[s:s+16]
   context=m.context(x,mm,cc,aa,channels=12); fused=m.fusion12(context)
   safe=torch.where(mm,x,torch.zeros_like(x)); mean,last=m.bridge.encode_response(safe,mm,cc,x,mm); response=torch.cat((mean,last),-1)
   chunks.append((response.cpu().numpy(),fused.cpu().numpy()))
  response=np.concatenate([x[0] for x in chunks]); fused=np.concatenate([x[1] for x in chunks]); rows.append({'domain':d,'cells':len(f),'response_mean_norm':float(np.linalg.norm(response.mean(0))),'response_std_mean':float(response.std(0).mean()),'fusion_mean_norm':float(np.linalg.norm(fused.mean(0))),'fusion_std_mean':float(fused.std(0).mean())})
frame=pd.DataFrame(rows); frame.to_csv(OUT/'RESPONSE_FUSION_SHIFT.csv',index=False,lineterminator='\n'); (OUT/'RESPONSE_FUSION_SHIFT_RECEIPT.json').write_text(json.dumps({'status':'COMPLETE_E1_RESPONSE_FUSION_SHIFT','domains':frame.domain.tolist(),'future_labels_read':0},indent=2)+'\n',encoding='utf-8'); print(frame.to_string(index=False))
