from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.drs_peft.self_reference_features import build_self_reference_prefixes

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'runtime/drs_peft/data_v1'
OUT = ROOT / 'runtime/drs_peft/e1_domain_shift'
OUT.mkdir(parents=True, exist_ok=True)

for domain in ('MATR', 'HUST'):
    with np.load(DATA / domain / 'arrays.npz', allow_pickle=False) as z:
        ids = z['ids'].astype(str).tolist()
    prefixes = build_self_reference_prefixes(domain, DATA, ids)
    np.savez_compressed(OUT / f'{domain}.npz', ids=np.asarray(prefixes.ids), real_features=prefixes.real_features, reference_features=prefixes.reference_features, feature_mask=prefixes.feature_mask, cycles=prefixes.cycles, real_anchor=prefixes.real_anchor, reference_anchor=prefixes.reference_anchor)
    (OUT / f'{domain}.json').write_text(__import__('json').dumps(prefixes.audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(domain, len(ids), prefixes.real_features.shape)
