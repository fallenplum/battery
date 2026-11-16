"""Audit source-only P0 receipts, paired pretraining evidence and the P0 gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drs_peft.episodes import digest, load_registration

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/drs_peft/execution_v1.json'
METHODS = ('B0_pretrained_frozen', 'B1_random_frozen', 'B2_standard_lora', 'B3_battery_direct')
SUMMARY_COLUMNS = ('phase', 'bridge', 'method', 'seed', 'status', 'epochs', 'updates', 'best_epoch',
    'source_cells', 'meta_cells', 'observations', 'mae', 'rmse', 'protocol_macro_mae',
    'mae_101_200', 'mae_201_500', 'mae_501_1000', 'mae_1001_plus',
    'bridge_trainable', 'head_trainable', 'peft_trainable', 'trainable_parameters',
    'backbone_parameters', 'battery_encoder_parameters', 'state_fusion_parameters',
    'input_budget', 'elapsed_seconds', 'peak_allocated_MiB', 'peak_reserved_MiB',
    'checkpoint_sha256', 'receipt_sha256', 'predictions_sha256', 'independent_local_replay_status',
    'independent_local_replay_max_absolute_error', 'independent_local_replay_absolute_tolerance',
    'backbone_initial_tensor_sha256', 'authoritative_random_backbone_sha256')
BOOTSTRAP_COLUMNS = ('phase', 'bridge', 'comparison', 'difference_definition', 'resampling_unit',
    'seed_count', 'seeds', 'physical_cells', 'protocol_groups', 'bootstrap_draws',
    'mean_difference', 'ci_lower', 'ci_upper', 'confidence_level', 'seed_uncertainty_scope',
    'negative_difference_favors', 'source_scope')


@dataclass
class VerifiedUnit:
    folder: Path
    summary: dict
    ids: np.ndarray
    protocols: np.ndarray
    target: np.ndarray
    observed: np.ndarray
    prediction: np.ndarray
    cell_mae: np.ndarray


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    pending.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_no_target_use(record):
    keys = [name for name in ('target_query_labels_used', 'target_query_labels_read') if name in record]
    require(bool(keys) and all(record[name] == 0 for name in keys), 'Run records target Query use or lacks its audit field')


def cell_errors(prediction, target, observed):
    count = observed.sum(1)
    require(bool((count > 0).all()), 'A registered source-meta cell has no actual future observations')
    error = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0)
    return np.abs(error).sum(1) / count, np.sqrt((error**2).sum(1) / count)


def verify_unit(folder, phase, bridge, method, seed, config, registration, config_path):
    receipt_path = folder / 'SOURCE_DEVELOPMENT_COMPLETE.json'
    receipt = read_json(receipt_path)
    require(receipt['status'] == 'SOURCE_DEVELOPMENT_COMPLETE', 'Source receipt has an unexpected status')
    required_files = {'RUN_CONFIG.json', 'SOURCE_SNAPSHOT.json', 'DATA_FINGERPRINT.json',
        'EPISODE_IDS.json', 'BACKBONE_IDENTITY.json', 'TRAINABLE_PARAMETER_REPORT.json',
        'TRAIN_LOG.json', 'TRAIN_LOG.jsonl', 'CHECKPOINT_SELECTED.pt', 'CHECKPOINT_SELECTION.json',
        'SOURCE_SCALER.json', 'SOURCE_VALIDATION_CELL_METRICS.csv', 'SOURCE_VALIDATION_PREDICTIONS.npz'}
    require(required_files.issubset({record['path'] for record in receipt['files']}), 'Receipt omits required run artifacts')
    for record in receipt['files']:
        path = (folder / record['path']).resolve()
        require(path.parent == folder.resolve(), 'Receipt file escapes its run directory')
        require(path.is_file() and digest(path) == record['sha256'], 'Run artifact SHA differs: ' + path.name)
    identity = read_json(folder / 'RUN_CONFIG.json')
    epochs = config['P0'][phase + '_epochs']
    source_ids = registration['source_cell_roles']['source_train']
    meta_ids = registration['source_cell_roles']['source_meta_validation']
    expected_updates = epochs * ((len(source_ids) + 7) // 8)
    for name, expected in dict(stage='P0', method=method, bridge=bridge, seed=seed,
                               epochs=epochs, updates=expected_updates).items():
        require(receipt[name] == expected, f'Receipt {name} differs from the registered P0 unit')
    for name, expected in dict(stage='P0', method=method, bridge=bridge, seed=seed, epochs=epochs,
                               expected_updates=expected_updates, batch_size=8,
                               head=config['head'], loss=config['loss'], query_grid=[101, 5000],
                               config_sha256=digest(config_path),
                               episode_sha256=registration['episode_contract_sha256'],
                               training_ids=source_ids, validation_ids=meta_ids).items():
        require(identity[name] == expected, f'Run identity {name} differs from the registered P0 unit')
    require_no_target_use(identity)
    require_no_target_use(receipt)
    require(receipt['final_target_test_status'] == 'SEALED_NOT_EVALUATED', 'Final target was already evaluated')
    require(receipt['frozen_parameters_unchanged'] is True and receipt['prediction_replay_max_absolute_error'] == 0,
            'Run lacks frozen-parameter or exact reconstruction evidence')
    require(read_json(folder / 'EPISODE_IDS.json') == dict(source_train=source_ids, source_meta_validation=meta_ids),
            'Recorded source roles differ')
    scaler = read_json(folder / 'SOURCE_SCALER.json')
    require(scaler['training_ids'] == source_ids, 'Preprocessing was fitted outside source85')
    fingerprint = read_json(folder / 'DATA_FINGERPRINT.json')
    semantic_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    require(identity['data_cache_sha256'] == semantic_hash, 'Run data fingerprint differs from the recorded cache identity')
    split_lock = read_json(ROOT / 'research/drs_peft/DATA_SPLIT_LOCK.json')
    cache_manifest = ROOT / 'runtime/drs_peft/data_v1/MATR/COMPLETE.json'
    require(digest(cache_manifest) == split_lock['data_caches']['MATR']['manifest_sha256'], 'Current MATR cache manifest differs from split lock')
    require(read_json(cache_manifest) == fingerprint, 'Run data fingerprint differs from the frozen MATR data cache')
    snapshot = read_json(folder / 'SOURCE_SNAPSHOT.json')['files']
    require(snapshot == identity['source_snapshot'], 'Code snapshots differ within run identity')
    for record in snapshot:
        require(digest(ROOT / record['path']) == record['sha256'], 'Current frozen source differs: ' + record['path'])
    backbone = read_json(folder / 'BACKBONE_IDENTITY.json')
    random_backbone_sha256 = None
    if method == 'B3_battery_direct':
        require(backbone == dict(kind='M1_battery_encoder', pretrained=False, shared_head=True), 'B3 encoder identity differs')
        require(read_json(folder / 'BATTERY_BASELINE_SCALER.json')['training_ids'] == source_ids,
                'B3 feature scaler was fitted outside source85')
    else:
        official = read_json(ROOT / 'research/drs_peft/TSFM_BACKBONE_AUDIT.json')
        require(backbone['weight_sha256'] == official['weight_sha256'] and backbone['revision'] == official['revision'],
                'Official model revision or weight identity differs')
        require(backbone['official_pretrained_weights_loaded'] == (method != 'B1_random_frozen'),
                'Random/pretrained provenance differs')
        require(backbone['random_control'] == (method == 'B1_random_frozen'), 'Random backbone identity differs')
        if method == 'B1_random_frozen':
            random_folder = folder.parents[3] / 'random_backbones'
            manifest = read_json(random_folder / 'RANDOM_BACKBONE_MANIFEST.json')
            records = [record for record in manifest if record['seed'] == seed]
            require(len(records) == 1, 'Authoritative random backbone seed is missing or repeated')
            record = records[0]
            state_path = (random_folder / record['path']).resolve()
            require(state_path.parent == random_folder.resolve(), 'Random backbone manifest path escapes its directory')
            require(state_path.stat().st_size == record['bytes'] and digest(state_path) == record['sha256'],
                    'Authoritative random backbone file differs')
            require(record['tensor_sha256'] == backbone['actual_initial_tensor_sha256'],
                    'Authoritative random backbone tensor identity differs from training')
            random_backbone_sha256 = record['sha256']
    history = read_json(folder / 'TRAIN_LOG.json')
    require(len(history) == epochs, 'Training did not close the registered epoch budget')
    require([row['epoch'] for row in history] == list(range(1, epochs + 1)), 'Training epochs contain gaps or repeats')
    require([row['updates'] for row in history] == [11 * e for e in range(1, epochs + 1)], 'Optimization updates differ')
    logged = [json.loads(line) for line in (folder / 'TRAIN_LOG.jsonl').read_text(encoding='utf-8').splitlines() if line]
    require(logged == history, 'JSON and JSONL logs differ')
    selection = read_json(folder / 'CHECKPOINT_SELECTION.json')
    best_epoch = int(np.argmin([row['validation_mae'] for row in history])) + 1
    require(selection['best_epoch'] == receipt['best_epoch'] == best_epoch, 'Checkpoint is not the first strict source-meta minimum')
    require(receipt['selected_checkpoint_sha256'] == digest(folder / 'CHECKPOINT_SELECTED.pt'), 'Checkpoint SHA differs')
    with np.load(folder / 'SOURCE_VALIDATION_PREDICTIONS.npz', allow_pickle=False) as arrays:
        ids = arrays['ids'].astype(str)
        query = arrays['query'].copy()
        prediction, target, observed = (arrays[key].copy() for key in ('prediction', 'target', 'observed'))
    require(ids.tolist() == meta_ids, 'Source validation predictions contain a changed physical UID role/order')
    require(np.array_equal(query, np.arange(101, 5001)), 'Prediction query grid differs from fixed101..5000')
    require(prediction.shape == target.shape == observed.shape == (30, 4900), 'Source validation arrays have changed shapes')
    require(observed.dtype == np.bool_, 'Observed mask is not boolean')
    require(bool(np.isfinite(prediction).all() and np.isfinite(target[observed]).all()), 'Observed target or prediction is nonfinite')
    group_for = {uid: group for group, members in registration['source_groups'].items() for uid in members}
    protocols = np.asarray([group_for[uid] for uid in ids])
    mae, rmse = cell_errors(prediction, target, observed)
    proto_mae = np.mean([mae[protocols == group].mean() for group in np.unique(protocols)])
    recomputed = dict(cells=30, observations=int(observed.sum()), mae=float(mae.mean()),
                      rmse=float(rmse.mean()), protocol_macro_mae=float(proto_mae))
    for recorded in (receipt['validation'], selection['validation']):
        require(recorded.keys() == recomputed.keys(), 'Validation metric schema differs')
        for key, value in recomputed.items():
            require(np.isclose(recorded[key], value, atol=1e-12, rtol=1e-10), 'Independent metric mismatch: ' + key)
    csv = pd.read_csv(folder / 'SOURCE_VALIDATION_CELL_METRICS.csv')
    all_cells = csv.loc[csv.window.eq('all')].set_index('cell_uid').loc[ids]
    require(np.array_equal(all_cells.protocol_uid.to_numpy(str), protocols), 'CSV protocol binding differs')
    require(np.allclose(all_cells.mae, mae, atol=1e-12, rtol=1e-10), 'CSV cell metrics differ from NPZ replay')
    parameters = read_json(folder / 'TRAINABLE_PARAMETER_REPORT.json')
    peft = parameters['peft_detail']['adapter_trainable_parameters']
    trainable = parameters['trainable']
    require(trainable == parameters['bridge_trainable'] + parameters['head_trainable'] + peft,
            'Bridge/head/PEFT parameter decomposition does not close')
    require(trainable == receipt['trainable_parameters'] == identity['trainable_parameters'], 'Trainable parameter counts differ')
    summary = dict(phase=phase, bridge=bridge, method=method, seed=seed, status='VERIFIED_SOURCE_DEVELOPMENT',
        epochs=epochs, updates=expected_updates, best_epoch=best_epoch, source_cells=85,
        meta_cells=30, observations=recomputed['observations'], mae=recomputed['mae'], rmse=recomputed['rmse'],
        protocol_macro_mae=recomputed['protocol_macro_mae'], bridge_trainable=parameters['bridge_trainable'],
        head_trainable=parameters['head_trainable'], peft_trainable=peft, trainable_parameters=trainable,
        backbone_parameters=parameters['backbone_total'], battery_encoder_parameters=parameters.get('battery_encoder', 0),
        state_fusion_parameters=parameters.get('state_fusion', 0),
        input_budget=parameters.get('input_budget', 'canonical VIQ, true early drift, prefix masks, known policy, SOH100 anchor'),
        elapsed_seconds=receipt['elapsed_seconds'],
        peak_allocated_MiB=receipt['peak_gpu_memory_bytes']['allocated'] / (1024**2),
        peak_reserved_MiB=receipt['peak_gpu_memory_bytes']['reserved'] / (1024**2),
        checkpoint_sha256=receipt['selected_checkpoint_sha256'], receipt_sha256=digest(receipt_path),
        predictions_sha256=digest(folder / 'SOURCE_VALIDATION_PREDICTIONS.npz'),
        independent_local_replay_status='NOT_RECORDED', independent_local_replay_max_absolute_error=None,
        independent_local_replay_absolute_tolerance=None,
        backbone_initial_tensor_sha256=backbone.get('actual_initial_tensor_sha256'),
        authoritative_random_backbone_sha256=random_backbone_sha256)
    replay_path = folder / 'INDEPENDENT_LOCAL_REPLAY.json'
    if replay_path.is_file():
        replay = read_json(replay_path)
        require(replay['status'] == 'PASSED' and replay['checkpoint_sha256'] == summary['checkpoint_sha256'],
                'Independent local replay checkpoint/status differs')
        require_no_target_use(replay)
        require(replay['cells'] == 30 and replay['method'] == method and replay['bridge'] == bridge and replay['seed'] == seed,
                'Independent local replay covers a different registered unit')
        require(0 <= replay['maximum_absolute_error'] <= replay['absolute_tolerance'],
                'Independent local replay exceeds its recorded tolerance')
        if method == 'B1_random_frozen':
            require(replay.get('authoritative_random_backbone_loaded') is True,
                    'B1 local replay did not load the actual trained random backbone')
        summary.update(independent_local_replay_status=replay['status'],
            independent_local_replay_max_absolute_error=replay['maximum_absolute_error'],
            independent_local_replay_absolute_tolerance=replay['absolute_tolerance'])
    for name, left, right in [('101_200', 101, 200), ('201_500', 201, 500),
                             ('501_1000', 501, 1000), ('1001_plus', 1001, 5000)]:
        mask = observed & (query >= left) & (query <= right)
        valid = mask.any(1)
        errors = np.where(mask, np.abs(prediction.astype(float) - target.astype(float)), 0).sum(1)
        summary['mae_' + name] = float((errors[valid] / mask.sum(1)[valid]).mean()) if valid.any() else None
    return VerifiedUnit(folder, summary, ids, protocols, target, observed, prediction, mae)


def paired_bootstrap(units, draws, seed):
    """Crossed seed/cell bootstrap preserves physical identities across seeds."""
    rows = []
    comparisons = []
    for phase, bridge in sorted({(u.summary['phase'], u.summary['bridge']) for u in units}):
        subset = [u for u in units if u.summary['phase'] == phase and u.summary['bridge'] == bridge]
        paired = {method: {u.summary['seed']: u for u in subset if u.summary['method'] == method}
                  for method in METHODS[:2]}
        seeds = sorted(set(paired[METHODS[0]]) & set(paired[METHODS[1]]))
        if not seeds:
            continue
        reference = paired[METHODS[0]][seeds[0]]
        differences = []
        for model_seed in seeds:
            pretrained, random = (paired[method][model_seed] for method in METHODS[:2])
            for unit in (pretrained, random):
                require(np.array_equal(unit.ids, reference.ids), 'Paired bootstrap physical IDs differ')
                require(np.array_equal(unit.observed, reference.observed) and
                        np.array_equal(unit.target, reference.target, equal_nan=True), 'Paired methods have different labels or masks')
            require(pretrained.summary['bridge_trainable'] == random.summary['bridge_trainable'] and
                    pretrained.summary['head_trainable'] == random.summary['head_trainable'] and
                    pretrained.summary['peft_trainable'] == random.summary['peft_trainable'] == 0,
                    'Pretrained/random comparison has unmatched trainable budgets')
            differences.append(pretrained.cell_mae - random.cell_mae)
        difference = np.asarray(differences)
        groups, membership = np.unique(reference.protocols, return_inverse=True)
        role_seed = int.from_bytes(hashlib.sha256(f'{seed}|{phase}|{bridge}'.encode()).digest()[:8], 'little')
        generator = np.random.default_rng(role_seed)
        seed_samples = generator.integers(len(seeds), size=(draws, len(seeds)))
        seed_means = difference[seed_samples].mean(1)
        cell_samples = generator.integers(len(reference.ids), size=(draws, len(reference.ids)))
        cell_replicates = seed_means[np.arange(draws)[:, None], cell_samples].mean(1)
        group_counts = generator.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
        weights = group_counts[:, membership]
        protocol_replicates = (seed_means * weights).sum(1) / weights.sum(1)
        for unit, replicates in (('physical_cell', cell_replicates), ('protocol_cluster', protocol_replicates)):
            low, high = np.quantile(replicates, [.025, .975])
            rows.append(dict(phase=phase, bridge=bridge, comparison='B0_pretrained_frozen_minus_B1_random_frozen',
                difference_definition='MAE_pretrained_minus_MAE_random; paired physical-cell, seed-mean',
                resampling_unit=unit, seed_count=len(seeds), seeds='|'.join(map(str, seeds)),
                physical_cells=len(reference.ids), protocol_groups=len(groups), bootstrap_draws=draws,
                mean_difference=float(difference.mean()), ci_lower=float(low), ci_upper=float(high), confidence_level=.95,
                seed_uncertainty_scope='crossed_seed_and_shared_' + unit if len(seeds) > 1 else 'conditional_on_single_training_seed',
                negative_difference_favors='B0_pretrained_frozen', source_scope='SOURCE_META_VALIDATION'))
        comparisons.append(dict(phase=phase, bridge=bridge, seeds=seeds, ids=reference.ids,
            protocols=reference.protocols, pretrained=np.stack([paired[METHODS[0]][s].cell_mae for s in seeds]),
            random=np.stack([paired[METHODS[1]][s].cell_mae for s in seeds])))
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS), comparisons


def verify_bridge_selection(path, units, config):
    require(path.is_file(), 'Pilot Bridge selection has not been frozen')
    value = read_json(path)
    require(value['selected_bridge'] in config['bridges'], 'Unregistered Bridge was selected')
    require_no_target_use(value)
    pilots = [u for u in units if u.summary['phase'] == 'pilot']
    require(len(pilots) == 6, 'Both Bridge pilot matrices must be complete before Bridge selection')
    candidates = {bridge: float(np.mean([u.summary['mae'] for u in pilots
                    if u.summary['bridge'] == bridge and u.summary['method'] in (METHODS[0], METHODS[2])]))
                  for bridge in config['bridges']}
    selected = min(config['bridges'], key=candidates.get)
    require(value['selected_bridge'] == selected, 'Bridge selection differs from the preregistered source-only criterion')
    require(value['criterion'] == config['P0']['bridge_selection'], 'Bridge criterion differs from preregistration')
    require(set(value['pilot_receipts']) == {u.summary['checkpoint_sha256'] for u in pilots}, 'Bridge selection checkpoint identities differ')
    for bridge in candidates:
        require(np.isclose(value['candidates'][bridge], candidates[bridge], atol=1e-12, rtol=1e-10), 'Bridge candidate score differs')
    return value


def pretraining_figure(comparison, bootstrap, directory):
    """Python quantitative grid, matched inputs, per-cell paired values and CIs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'legend.frameon': False})
    pretrained, random = comparison['pretrained'].mean(0), comparison['random'].mean(0)
    source = pd.DataFrame(dict(cell_uid=comparison['ids'], protocol_uid=comparison['protocols'],
        phase=comparison['phase'], bridge=comparison['bridge'], seeds='|'.join(map(str, comparison['seeds'])),
        mae_pretrained=pretrained, mae_random=random, difference=pretrained-random))
    source.to_csv(directory / 'P0_PRETRAINING_FIGURE_SOURCE.csv', index=False, lineterminator='\n')
    selected = bootstrap.loc[bootstrap.phase.eq(comparison['phase']) & bootstrap.bridge.eq(comparison['bridge'])]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={'width_ratios': [1.05, 1.35]})
    for left, right in zip(random, pretrained):
        axes[0].plot([0, 1], [left, right], color='#C7CBCF', linewidth=.55, zorder=1)
    axes[0].scatter(np.zeros(len(random)), random, color='#8D969E', s=13, zorder=2)
    axes[0].scatter(np.ones(len(pretrained)), pretrained, color='#397FA6', s=13, zorder=2)
    axes[0].scatter([0, 1], [random.mean(), pretrained.mean()], marker='D', s=35, c='#26343F', zorder=3)
    axes[0].set(xticks=[0, 1], xticklabels=['Random frozen', 'Pretrained frozen'], ylabel='Source-meta cell MAE (SOH)')
    axes[0].margins(x=.22)
    labels = []
    for index, row in enumerate(selected.itertuples(index=False)):
        axes[1].plot([row.ci_lower, row.ci_upper], [index, index], color='#397FA6', linewidth=1.6)
        axes[1].scatter(row.mean_difference, index, color='#26343F', s=25, zorder=3)
        label = 'Physical cell' if row.resampling_unit == 'physical_cell' else 'Protocol cluster'
        labels.append(label + ' (fixed seed)' if len(comparison['seeds']) == 1 else 'Seed + ' + label.lower())
    axes[1].axvline(0, color='#8D969E', linestyle='--', linewidth=.8)
    axes[1].set(yticks=range(len(labels)), yticklabels=labels,
                xlabel='MAE difference (pretrained − random)', ylim=(-.6, len(labels)-.4))
    axes[1].invert_yaxis()
    axes[0].set_title('a  Paired physical cells', loc='left', fontsize=9)
    axes[1].set_title('b  Mean difference and 95% interval', loc='left', fontsize=9)
    seeds_label = 'seed' if len(comparison['seeds']) == 1 else 'seeds'
    fig.suptitle(f'{comparison["phase"].capitalize()} · {comparison["bridge"]} · '
                 f'{len(pretrained)} cells, {len(comparison["seeds"])} {seeds_label}', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=2.5)
    for suffix in ('png', 'pdf'):
        metadata = {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None
        fig.savefig(directory / ('P0_PRETRAINED_VS_RANDOM.' + suffix), dpi=600,
                    bbox_inches='tight', facecolor='white', metadata=metadata)
    plt.close(fig)


def make_report(summary, bootstrap, gate, failures, figure_exists):
    lines = ['# DRS-PEFT P0 源开发基线与预训练审计', '',
        f'本次状态为**{gate["status"]}**，真实执行闭环为`{gate["execution_complete"]}`。', '',
        f'有效pilot凭据{gate["pilot_receipts_closed"]}/6，有效formal凭据{gate["formal_receipts_closed"]}/12。'
        f'Bridge冻结状态为{gate["bridge_selected"]}，选择为`{gate.get("selected_bridge")}`。', '',
        '本报告重新校验每个有效run凭据所列文件SHA、完整训练步数、选权、源码身份、源归一化角色和验证预测NPZ。MAE/RMSE由逐物理电芯的实际观测点重算，再跨电芯等权。源验证标签用于Bridge和checkpoint选择，因此这些误差属于开发集证据。最终目标Query保持SEALED_NOT_EVALUATED。', '',
        '## 实际完成训练', '',
        '| 阶段 | Bridge | 方法 | seed | epochs/updates | MAE | RMSE | 工况宏MAE | Bridge/Head/PEFT参数 |',
        '|---|---|---|---:|---|---:|---:|---:|---|']
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.phase} | {row.bridge} | {row.method} | {row.seed} | {row.epochs}/{row.updates} | '
            f'{row.mae:.8f} | {row.rmse:.8f} | {row.protocol_macro_mae:.8f} | '
            f'{row.bridge_trainable}/{row.head_trainable}/{row.peft_trainable} |')
    formal = summary.loc[summary.phase.eq('formal')]
    if len(formal):
        lines += ['', '## 正式源验证汇总', '',
            '| 方法 | 实际seed数 | MAE均值 | RMSE均值 | MAE的seed范围 | 总可训练参数 |',
            '|---|---:|---:|---:|---|---:|']
        for method in METHODS:
            values = formal.loc[formal.method.eq(method)]
            if len(values):
                lines.append(f'| {method} | {len(values)} | {values.mae.mean():.8f} | {values.rmse.mean():.8f} | '
                    f'[{values.mae.min():.8f}, {values.mae.max():.8f}] | {int(values.trainable_parameters.iloc[0])} |')
        lines += ['',
            '均值覆盖实际已完成的训练seed，各seed共用30颗source-meta电芯、15个exact协议组及21827个真实未来观测点。当前比较限于登记的85颗源训练电芯和每单元120epochs/1320updates预算。', '']
    local = summary.loc[summary.independent_local_replay_status.eq('PASSED')]
    local_note = f'附加本机独立重放已有{len(local)}份PASSED记录。'
    if len(local):
        local_note += f'其中最大预测绝对差为{local.independent_local_replay_max_absolute_error.max():.10g}，各自容差保存在汇总CSV。'
    lines += ['',
        'pilot固定两条Bridge×B0/B1/B2×seed42，每单元60epochs、660updates。formal固定选定Bridge×B0/B1/B2/B3×3seeds，每单元120epochs、1320updates。source85/meta30与固定101–5000网格全程相同。四时距误差、可训练参数总量、backbone总量、峰值显存、耗时及所有checkpoint/receipt/NPZ SHA见P0_SUMMARY.csv。', '',
        f'本地已验收源训练optimizer updates合计{int(summary.updates.sum())}，其中pilot为{int(summary.loc[summary.phase.eq("pilot"), "updates"].sum())}、formal为{int(formal.updates.sum())}。工程smoke和恢复探针另行记账。', '',
        'B3使用现有M1电池编码器、额外显式前缀派生G20/local64/local8×26及mask、合法reference与已知policy，保留相同cycle-query Head。它引入额外的物理派生输入与编码结构，参数数目、信息预算和state-fusion数量单列，作为电池专用参照解释。', '',
        'B1跨OS按相同seed重新初始化时，已观察到57/94个张量存在约1.19e-7的浮点差异。服务器实际随机backbone的三seed完整状态保存在本地P0/random_backbones，RANDOM_BACKBONE_MANIFEST绑定文件SHA和原始张量SHA。独立本机重放严格载入该实际随机状态，随后核验初始tensor身份并重放预测。跨OS重放采用记录的容差，同机序列化重构逐位一致。复现B1须同时保留这些权重与manifest。', '',
        local_note, '',
        '## 预训练与随机主干配对证据', '',
        '差值定义为B0预训练冻结MAE减B1随机冻结MAE，负值表示预训练误差较低。两者在同Bridge中匹配可训练Bridge/Head与零PEFT预算。每个物理cell先保留各seed配对差，再先重抽训练seed，随后共享重抽物理UID或整工况组，保留同一物理cell跨seed的重复结构。protocol cluster对整组电芯加权重抽，主估计仍为cell-mean。固定2000次、95%百分位区间。', '',
        '| 阶段 | Bridge | 重抽样单位 | seeds | 电芯/工况 | 平均差 | 95%区间 |', '|---|---|---|---:|---|---:|---|']
    for row in bootstrap.itertuples(index=False):
        lines.append(f'| {row.phase} | {row.bridge} | {row.resampling_unit} | {row.seed_count} | '
            f'{row.physical_cells}/{row.protocol_groups} | {row.mean_difference:.8f} | [{row.ci_lower:.8f}, {row.ci_upper:.8f}] |')
    lines += ['',
        'pilot只有seed42，区间条件于该训练seed。正式三seed区间同时反映有限seed重抽与物理样本重抽，保留三seed覆盖的局限。开发集选权和Bridge选择已使用这些meta标签，区间描述当前开发证据。', '',
        f'预训练证据状态为`{gate["pretrained_gain_status"]}`。完整formal配对中两个敏感性区间上界均小于0时记录SUPPORTED，下界均大于0时记录NEGATIVE，其余情况记录INSUFFICIENT。该状态独立于执行闭环Gate，完整执行后继续P1诊断。', '']
    if figure_exists:
        evidence = gate['figure_evidence']
        lines += ['![主Bridge预训练与随机主干配对比较](figures/P0_PRETRAINED_VS_RANDOM.png)', '',
            f'当前图对应{evidence["phase"]}、{evidence["seed_count"]}个训练seed。图左连接相同物理电芯的seed平均MAE，菱形为电芯均值。图右为配对平均差及两种重抽样95%区间。单seed图的区间条件于该seed。PNG/PDF由Python生成，原始图数据与绘图脚本一同保存。', '']
    lines += ['## 运行验收与后续状态', '',
        'P0_GATE仅在6个pilot和12个formal凭据、Bridge冻结、源码/数据/预测校验及目标用途隔离全部通过后，将execution_complete设为true并记录PASSED。预训练收益不足仍按合同继续P1。PARTIAL/BLOCKED保留缺失或失败清单，等待补齐或修复。', '',
        'H1/H2/H3仍由P1/P3/P4分别检验。P0承担基线和预训练任务一致性审查。', '',
        '## 优势与劣势（风险）', '',
        '- 优势为官方/随机权重身份可追溯、同Bridge/Head对照、逐cell等权评分、真实SHA与独立指标重算闭环。',
        '- 劣势（风险）为meta30和有限训练seed、历史探索暴露、开发集选权和Bridge选择复用，以及B3显式物理派生输入的结构差异。最终跨域与few-shot效果仍待注册目标评估。', '',
        '## 缺失与失败', '']
    if failures:
        lines.extend('- ' + item['unit'] + '，' + item['kind'] + '，' + item.get('detail', '') for item in failures)
    else:
        lines.append('全部登记单元通过本次凭据与预测审计，逐电芯误差保留在各run的SOURCE_VALIDATION_CELL_METRICS.csv。')
    lines += ['', '## 置信度与复现', '',
        '文件身份、优化计数和误差重算的置信度取决于本报告实际通过的凭据。预训练判断只依据source-meta配对数据与区间，H1/H2/H3仍待后续Gate。', '',
        '```powershell', '& D:\\Anaconda\\envs\\tslib\\python.exe -m scripts.drs_peft.summarize_p0', '```', '',
        '拓展思考与可能关注的问题包括Bridge条件下的预训练差异、各工况开发误差及负迁移。固定目标Query保持后续独立评分角色。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/drs_peft')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'report/drs_peft')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    config = read_json(args.config)
    registration = load_registration()
    require(config['training_seeds'] == [42, 123, 2026], 'Formal seed contract changed')
    require(config['statistics']['bootstrap_draws'] == 2000, 'Paired bootstrap budget changed')
    base = args.output_root / 'P0'
    selection_path = base / 'BRIDGE_SELECTION.json'
    selection = read_json(selection_path) if selection_path.is_file() else None
    specs = [('pilot', bridge, method, 42) for bridge in config['bridges'] for method in METHODS[:3]]
    if selection is not None:
        specs.extend(('formal', selection['selected_bridge'], method, seed)
                     for seed in config['training_seeds'] for method in METHODS)
    if args.dry_run:
        print(json.dumps(dict(status='SUMMARY_READY',expected_pilot_units=6, expected_formal_units=12,
            existing_receipts=sum((base / phase / bridge / method / f'seed{seed}' / 'SOURCE_DEVELOPMENT_COMPLETE.json').is_file()
                                  for phase, bridge, method, seed in specs), bootstrap_draws=2000,
            target_query_labels_used=False, output=str(base)), ensure_ascii=False))
        return
    units, failures = [], []
    for phase, bridge, method, seed in specs:
        folder = base / phase / bridge / method / f'seed{seed}'
        key = f'{phase}/{bridge}/{method}/seed{seed}'
        if not (folder / 'SOURCE_DEVELOPMENT_COMPLETE.json').is_file():
            failures.append(dict(unit=key, kind='PENDING_RECEIPT', detail='Registered source training receipt unavailable'))
            continue
        try:
            unit = verify_unit(folder, phase, bridge, method, seed, config, registration, args.config)
            if units:
                require(np.array_equal(unit.ids, units[0].ids) and np.array_equal(unit.protocols, units[0].protocols),
                        'P0 methods or seeds have different source-meta physical cohorts')
                require(np.array_equal(unit.observed, units[0].observed) and
                        np.array_equal(unit.target, units[0].target, equal_nan=True),
                        'P0 methods or seeds have different source-meta labels or observation masks')
            units.append(unit)
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            failures.append(dict(unit=key, kind='FAILED_ARTIFACT_AUDIT', detail=str(error)))
    bridge_valid = False
    try:
        selection = verify_bridge_selection(selection_path, units, config)
        bridge_valid = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        kind = 'FAILED_ARTIFACT_AUDIT' if selection_path.is_file() and len(units) >= 6 else 'PENDING_BRIDGE_LOCK'
        failures.append(dict(unit='P0/BRIDGE_SELECTION', kind=kind, detail=str(error)))
    if not any(phase == 'formal' for phase, _, _, _ in specs):
        failures.append(dict(unit='formal_matrix', kind='PENDING_RECEIPT', detail='12 formal units await frozen Bridge'))
    bootstrap, comparisons = paired_bootstrap(units, 2000, registration['episode_seed'])
    summary = pd.DataFrame([unit.summary for unit in units], columns=SUMMARY_COLUMNS)
    pilot_count = sum(unit.summary['phase'] == 'pilot' for unit in units)
    formal_count = sum(unit.summary['phase'] == 'formal' for unit in units)
    complete = pilot_count == 6 and formal_count == 12 and bridge_valid and not failures
    formal_bootstrap = bootstrap.loc[bootstrap.phase.eq('formal') & bootstrap.seed_count.eq(3)]
    gain = len(formal_bootstrap) == 2 and bool(formal_bootstrap.ci_upper.lt(0).all())
    harm = len(formal_bootstrap) == 2 and bool(formal_bootstrap.ci_lower.gt(0).all())
    status = 'PASSED' if complete else (
        'BLOCKED' if any(f['kind'] == 'FAILED_ARTIFACT_AUDIT' for f in failures) else 'PARTIAL')
    gate = dict(stage='P0', status=status, execution_complete=complete,
        pilot_receipts_closed=pilot_count, formal_receipts_closed=formal_count,
        bridge_selected=bridge_valid, selected_bridge=selection['selected_bridge'] if selection else None,
        pretrained_gain_status=('SUPPORTED' if gain else 'NEGATIVE' if harm else 'INSUFFICIENT') if len(formal_bootstrap) == 2 else 'PENDING',
        gain_criterion='both physical-cell and protocol-cluster 95% source-meta paired intervals have upper bound below0',
        pretraining_gain_required_for_P1=False, target_query_labels_used=False,
        target_test_status='SEALED_NOT_EVALUATED', source_cells=85, source_meta_cells=30,
        formal_seeds=config['training_seeds'], bootstrap_draws=2000,
        summary_script_sha256=digest(Path(__file__)), config_sha256=digest(args.config),
        episode_contract_sha256=registration['episode_contract_sha256'],
        bridge_selection_sha256=digest(selection_path) if selection_path.is_file() else None,
        failures=failures,
        receipts=[dict(unit=str(unit.folder.relative_to(base)), sha256=unit.summary['receipt_sha256']) for unit in units])
    base.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(base / 'P0_SUMMARY.csv', index=False, lineterminator='\n')
    summary.to_csv(base / 'P0_METRICS.csv', index=False, lineterminator='\n')
    bootstrap.to_csv(base / 'P0_PAIRED_PRETRAINING_BOOTSTRAP.csv', index=False, lineterminator='\n')
    write_json(base / 'P0_FAILURES.json', failures)
    figure_exists = False
    candidates = [c for c in comparisons if c['bridge'] == gate['selected_bridge']
                  and (c['phase'] == 'pilot' or len(c['seeds']) == 3)]
    if bridge_valid and candidates and not args.no_figures:
        comparison = sorted(candidates, key=lambda c: c['phase'] == 'formal')[-1]
        figure_dir = args.report_dir / 'figures'
        figure_dir.mkdir(parents=True, exist_ok=True)
        pretraining_figure(comparison, bootstrap, figure_dir)
        figure_exists = True
        gate['figure_evidence'] = dict(phase=comparison['phase'], seed_count=len(comparison['seeds']),
            files=[dict(path=p.name, sha256=digest(p)) for p in (
                figure_dir / 'P0_PRETRAINING_FIGURE_SOURCE.csv', figure_dir / 'P0_PRETRAINED_VS_RANDOM.png',
                figure_dir / 'P0_PRETRAINED_VS_RANDOM.pdf')])
    (args.report_dir / 'P0_REPORT_ZH.md').write_text(make_report(summary, bootstrap, gate, failures, figure_exists), encoding='utf-8', newline='\n')
    gate['files'] = [dict(path=p.name, sha256=digest(p)) for p in (
        base / 'P0_SUMMARY.csv', base / 'P0_METRICS.csv', base / 'P0_PAIRED_PRETRAINING_BOOTSTRAP.csv', base / 'P0_FAILURES.json')]
    write_json(base / 'P0_GATE.json', gate)
    print(json.dumps(dict(status=status, execution_complete=complete, valid_pilot=pilot_count,
        valid_formal=formal_count, pretrained_gain_status=gate['pretrained_gain_status'], failures=len(failures)), ensure_ascii=False))


if __name__ == '__main__':
    main()
