"""Outcome-independent, protocol-disjoint DRS-PEFT episode registration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / 'configs/drs_peft/episodes_v1.json'
SHOTS = (0, 1, 3, 5, 10)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def key(seed, role, identifier):
    return hashlib.sha256(f'{seed}|{role}|{identifier}'.encode('utf-8')).hexdigest()


def ordered(values, seed, role):
    return sorted(values, key=lambda value: (key(seed, role, value), value))


def support_draws(pool, query, *, seed, role, draws=5):
    """Nested support orders. Duplicate support sets retain an explicit cache identity."""
    pool, query = list(pool), list(query)
    if len(pool) != len(set(pool)) or len(query) != len(set(query)) or set(pool) & set(query):
        raise ValueError('Support candidates and query must be unique, disjoint physical cells')
    result = []
    for shot in SHOTS:
        seen = {}
        if len(pool) < shot:
            result.append(dict(k=shot, status='NOT_FEASIBLE', requested_draws=draws,
                unique_support_sets=0, available_support_cells=len(pool), query_ids=query, draws=[]))
            continue
        runs = []
        for draw in range(draws):
            ids = sorted(ordered(pool, seed, f'{role}|draw{draw}')[:shot])
            fingerprint = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            duplicate = seen.get(fingerprint)
            if duplicate is None:
                seen[fingerprint] = draw
            runs.append(dict(draw=draw, support_ids=ids, support_set_sha256=fingerprint,
                             duplicate_of_draw=duplicate, query_ids=query))
        result.append(dict(k=shot, status='FEASIBLE', requested_draws=draws,
            unique_support_sets=len(seen), available_support_cells=len(pool),
            mathematically_available_support_sets=math.comb(len(pool), shot),query_ids=query,draws=runs))
    return result


def build_registration(root=ROOT, seed=42, draws=5):
    """Read identity/design columns only. No future outcomes are inspected."""
    root = Path(root)
    source_file = root / 'data/unified_matr_seed42/cells.parquet'
    cells = pd.read_parquet(source_file, columns=['cell_uid', 'protocol_uid'])
    if cells.cell_uid.duplicated().any() or len(cells) != 169:
        raise ValueError('MATR physical identity denominator differs from the frozen169 cells')
    groups = {str(name): sorted(group.cell_uid.tolist()) for name, group in cells.groupby('protocol_uid')}
    sizes = {name: len(ids) for name, ids in groups.items()}
    high_support = sorted(name for name in groups if sizes[name] >= 7)
    if len(groups) != 77 or len(high_support) != 4:
        raise ValueError('Expected exact protocol support changed; register a new cohort version')
    target = high_support + ordered(set(groups)-set(high_support), seed, 'target_group')[:15-len(high_support)]
    remaining = set(groups)-set(target)
    replicated = ordered([name for name in remaining if sizes[name] >= 2], seed, 'meta_replicated')[:3]
    meta = replicated + ordered(remaining-set(replicated), seed, 'meta_group')[:15-len(replicated)]
    train = sorted(remaining-set(meta))
    roles = dict(source_train=sorted(train),source_meta_validation=sorted(meta),target_test=sorted(target))
    if [len(roles[r]) for r in roles] != [47,15,15]:
        raise ValueError('Protocol role counts differ from the registered47/15/15 design')
    sets = [set(v) for v in roles.values()]
    if any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Exact protocol groups overlap across roles')
    source_roles = {role: sorted(uid for group in names for uid in groups[group]) for role,names in roles.items()}
    tasks = []
    for group in sorted(target):
        ids = ordered(groups[group], seed, 'A_query|'+group)
        if len(ids) < 2:
            tasks.append(dict(task_id='A_'+group,protocol='A_MATR_UNSEEN_EXACT_PROTOCOL',domain='MATR',
                protocol_uid=group,status='NOT_FEASIBLE_Q2',query_count=2,available_cells=len(ids),
                query_ids=[],support_pool_ids=[],shots=[]))
            continue
        query,pool = ids[:2],ids[2:]
        tasks.append(dict(task_id='A_'+group,protocol='A_MATR_UNSEEN_EXACT_PROTOCOL',domain='MATR',
            protocol_uid=group,status='READY',query_count=2,available_cells=len(ids),query_ids=query,
            support_pool_ids=pool,shots=support_draws(pool,query,seed=seed,role='A|'+group,draws=draws)))
    target_files = {
        'Stanford_2':root/'results/extension_stanford/prior_context.csv',
        'HUST':root/'data/hust_raw_idea_v2/cells.parquet',
    }
    target_identity_audit = {}
    for domain,path in target_files.items():
        if domain == 'Stanford_2':
            table = pd.read_csv(path,usecols=['cell_id','protocol_group','source_device_id_sha256'])
            ids = table.cell_id.tolist()
            if len(ids) != 181 or table.source_device_id_sha256.nunique() != 181:
                raise ValueError('Stanford supplied physical identity binding changed')
        else:
            table = pd.read_parquet(path,columns=['cell_uid','protocol_uid'])
            ids = table.cell_uid.tolist()
            if len(ids) != 77:
                raise ValueError('HUST physical cohort changed')
        if len(ids) != len(set(ids)):
            raise ValueError('Repeated target physical identifier')
        ids = ordered(ids,seed,domain+'|cross_domain_query')
        count = math.ceil(len(ids)/2)
        query,pool = ids[:count],ids[count:]
        tasks.append(dict(task_id=('B' if domain=='Stanford_2' else 'C')+'_'+domain,
            protocol='B_MATR_TO_STANFORD_SUPPLIED' if domain=='Stanford_2' else 'C_MATR_TO_HUST_NATIVE',
            domain=domain,status='READY',query_count=len(query),available_cells=len(ids),
            query_ids=query,support_pool_ids=pool,
            shots=support_draws(pool,query,seed=seed,role=domain+'|cross_domain',draws=draws)))
        target_identity_audit[domain] = dict(file=path.relative_to(root).as_posix(),sha256=digest(path),
            cells=len(ids),query=len(query),support_candidates=len(pool))
    feasibility = [dict(k=k,query_count=q,exact_protocol_groups=sum(n>=k+q for n in sizes.values()),
                        physical_cells=sum(n for n in sizes.values() if n>=k+q))
                   for q in (1,2) for k in SHOTS]
    result = dict(study_id='DRS_PEFT_20261117_v1',episode_seed=seed,model_training_seeds=[42,123,2026],
        pilot_training_seed=42,requested_support_draws=draws,shots=list(SHOTS),
        selection_uses='physical_UID_and_prefix_qualified_design_group_counts_only',
        future_outcomes_read_for_split=False,source_domain='MATR',source_group_roles=roles,
        source_cell_roles=source_roles,source_groups={g:groups[g] for g in sorted(groups)},
        source_protocol_size_distribution={str(n):sum(v==n for v in sizes.values()) for n in sorted(set(sizes.values()))},
        full_MATR_feasibility=feasibility,target_tasks=tasks,target_identity_audit=target_identity_audit,
        cohort_source_sha256=digest(source_file),
        fixed_prediction_grid=dict(minimum_cycle=101,maximum_cycle=5000,step=1,far_diagnostic=[5001,6000]),
        zero_shot='reuse identical source checkpoint; no target future outcomes and no optimization',
        duplicate_draw_policy='identical support sets share adapted checkpoint and are not independent support replicates',
        query_policy='fixed per task across shots, methods and support draws; Q2 protocol has no implicit Q1 fallback',
        preprocessing_policy='fit on source_train only; all-missing policy fields remain masked across target domains',
        historical_exposure='Retrospective cohorts previously used in battery research; fresh independent blind-test claims unavailable')
    encoded = json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    result['episode_contract_sha256'] = hashlib.sha256(encoded).hexdigest()
    return result


def load_registration(path=DEFAULT_PATH):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    stored = value.pop('episode_contract_sha256')
    actual = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
    value['episode_contract_sha256'] = stored
    if actual != stored:
        raise ValueError('Episode contract payload fingerprint changed')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=DEFAULT_PATH)
    args = parser.parse_args()
    value = build_registration(args.root)
    if args.output.exists():
        if load_registration(args.output) != value:
            raise RuntimeError('Existing episode registration differs; create a new version explicitly')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='EPISODES_FROZEN',source_cells={r:len(v) for r,v in value['source_cell_roles'].items()},
        source_groups={r:len(v) for r,v in value['source_group_roles'].items()},
        tasks=len(value['target_tasks']),fingerprint=value['episode_contract_sha256'])))


if __name__=='__main__':
    main()
