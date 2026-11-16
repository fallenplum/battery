"""Read-only P1 result audit and descriptive source-error associations."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .episodes import digest, load_registration
from .summarize_p0 import read_json, require, write_json

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 2026)
FEATURES = ('gradient_norm', 'capture_grad_cov', 'capture_random_mean', 'capture_weight_svd',
            'capture_gap_vs_random', 'capture_gap_vs_weight_svd')
LAYERS = tuple(f'encoder.layers.{block}.self_attn.{projection}_proj'
               for block in range(6) for projection in ('q', 'v'))


def primary_capture(path):
    columns = ['seed', 'layer', 'protocol_uid', 'statistic', 'cell_uid', 'method',
               'rank', 'capture', 'gradient_norm', 'zero_gradient', 'evaluated_units']
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100000):
        parts.append(chunk.loc[chunk['rank'].eq(4)].copy())
    return pd.concat(parts, ignore_index=True)


def verify_seed(folder, seed, output_root, registration):
    manifest = read_json(folder / 'P1_PUBLIC_COMPLETE.json')
    require(manifest['status'] == 'PUBLISHED_SOURCE_DIAGNOSTIC', 'P1 public completion status differs')
    for record in manifest['public_files']:
        path = (folder / record['path']).resolve()
        require(path.parent == folder.resolve(), 'P1 public artifact escapes its seed directory')
        require(path.is_file() and path.stat().st_size == record['bytes'] and digest(path) == record['sha256'],
                'P1 public file fingerprint differs: ' + path.name)
    result = read_json(folder / 'P1_RESULT.json')
    provenance = read_json(folder / 'P1_REGISTRATION.json')
    require(result['status'] == 'COMPLETE_SOURCE_GRADIENT_DIAGNOSTIC' and result['seed'] == seed, 'P1 seed is incomplete or mismatched')
    require(result['source_model_before_sha256'] == result['source_model_after_sha256'] == provenance['model_state_sha256'],
            'P1 changed the source model or used a different registered state')
    require(result['optimizer_updates'] == 0 and result['target_query_labels_used'] == 0 and
            result['target_Query_future_labels_accessed'] is False, 'P1 violated zero-update source-only scope')
    require(provenance['target_Query_future_labels_accessed'] is False, 'P1 registration reports target Query access')
    require(provenance['implementation_sha256']['P1_gradient_audit'] == digest(ROOT / 'scripts/drs_peft/gradient_audit.py'),
            'P1 implementation differs from its frozen source')
    for role, key in [('source_train', 'source_ids'), ('source_meta_validation', 'meta_validation_ids')]:
        require(provenance[key] == registration['source_cell_roles'][role], 'P1 physical roles differ from frozen episodes')
    require(provenance['query_grid'] == list(range(101, 5001)), 'P1 query grid differs')
    expected = dict(source_cells=85, source_groups=47, meta_validation_cells=30, meta_validation_groups=15,
        primary_rank=4, target_matrices=12, physical_cell_gradient_evaluations=315,
        independent_endpoint_replay_physical_cells=10, independent_endpoint_replay_layer_gradients=120,
        total_gradient_forward_backward_evaluations=345)
    for key, value in expected.items():
        require(result[key] == value, 'P1 denominator or gradient count differs: ' + key)
    candidate = result['secondary_candidate_diagnostic']
    require(candidate['target_matrices'] == 42 and candidate['extra_gradient_forward_backward_evaluations'] == 20 and
            candidate['optimizer_updates'] == 0 and candidate['target_query_labels_used'] == 0 and
            candidate['main_QV_target_set_changed'] is False, 'Secondary candidate diagnostic changed scope')
    replay = read_json(folder / 'P1_GRADIENT_REPLAY.json')
    require(replay['status'] == 'PASSED' and replay['compared_layer_gradients'] == 120 and
            replay['replay_physical_cells'] == 10 and all(item['passed'] for item in replay['records']),
            'P1 gradient endpoint replay has not passed')
    p0 = output_root / 'P0/formal/native_multivariate/B0_pretrained_frozen' / f'seed{seed}'
    receipt = read_json(p0 / 'SOURCE_DEVELOPMENT_COMPLETE.json')
    require(receipt['selected_checkpoint_sha256'] == digest(p0 / 'CHECKPOINT_SELECTED.pt'), 'Matching P0 B0 checkpoint differs')
    identity = read_json(p0 / 'RUN_CONFIG.json')
    require(identity['seed'] == seed and identity['method'] == 'B0_pretrained_frozen' and
            identity['validation_ids'] == provenance['meta_validation_ids'], 'P1 is not paired with matching-seed P0 B0')
    gradients = pd.read_csv(folder / 'P1_CELL_GRADIENTS.csv')
    selected = gradients.loc[gradients.role.eq('meta_validation') & gradients.variant.eq('nominal')].copy()
    require(len(selected) == 360 and not selected.duplicated(['layer', 'cell_uid']).any(), 'Nominal meta gradient denominator differs')
    require(set(selected.layer) == set(LAYERS) and selected.cell_uid.nunique() == 30, 'P1 main QV/cell cohort differs')
    require(selected.connected.all(), 'A main meta gradient is disconnected')
    metrics = pd.read_csv(p0 / 'SOURCE_VALIDATION_CELL_METRICS.csv')
    baseline = metrics.loc[metrics.window.eq('all'), ['cell_uid', 'protocol_uid', 'mae', 'n_obs']]
    require(len(baseline) == 30 and not baseline.cell_uid.duplicated().any(), 'Matching P0 cell metrics differ')
    selected = selected.merge(baseline.rename(columns={'mae': 'base_validation_mae', 'n_obs': 'p0_observed_points'}),
                              on=['cell_uid', 'protocol_uid'], validate='many_to_one')
    require(len(selected) == 360 and bool((selected.observed_points == selected.p0_observed_points).all()),
            'P1 loss and P0 validation use different observed points or physical protocols')
    selected['p0_p1_loss_absolute_difference'] = (selected.loss - selected.base_validation_mae).abs()
    require(selected.p0_p1_loss_absolute_difference.max() <= 1e-5, 'P1 loss differs from matching P0 MAE by more than1e-5')
    capture = primary_capture(folder / 'P1_CAPTURE.csv')
    cells = capture.loc[capture.statistic.eq('individual_cell') &
                        capture.evaluated_units.eq('nominal_capacity_normalized')].copy()
    keys = ['seed', 'layer', 'cell_uid', 'protocol_uid']
    require(not cells.duplicated(keys + ['method']).any(), 'Repeated individual capture record')
    pivot = cells.pivot(index=keys, columns='method', values='capture').reset_index()
    random = [column for column in pivot if column.startswith('random_')]
    require(len(random) == 20 and len(pivot) == 360, 'Random capture replicates or main cell-layer denominator differs')
    pivot['capture_random_mean'] = pivot[random].mean(1, skipna=False)
    pivot['capture_grad_cov'] = pivot['grad_cov']
    pivot['capture_weight_svd'] = pivot['weight_svd']
    pivot['capture_gap_vs_random'] = pivot['grad_cov'] - pivot['capture_random_mean']
    pivot['capture_gap_vs_weight_svd'] = pivot['grad_cov'] - pivot['weight_svd']
    pivot['capture_gradient_mean_svd'] = pivot['gradient_mean_svd']
    pivot['capture_demeaned_cov'] = pivot['demeaned_cov']
    selected = selected.rename(columns={'source_seed': 'seed', 'loss': 'p1_loss'})
    extra = ['capture_gradient_mean_svd', 'capture_demeaned_cov']
    joined = selected.merge(pivot[keys + list(FEATURES[1:]) + extra], on=keys, validate='one_to_one')
    require(len(joined) == 360 and set(joined.seed) == {seed}, 'P1 error/capture merge changed its denominator')
    group = capture.loc[capture.statistic.eq('group_mean')].pivot(
        index=['seed', 'layer', 'protocol_uid'], columns='method', values='capture').reset_index()
    group['random_mean'] = group[random].mean(1, skipna=False)
    group['gap_vs_random'] = group['grad_cov'] - group['random_mean']
    group['gap_vs_weight_svd'] = group['grad_cov'] - group['weight_svd']
    group['gap_vs_shuffled'] = group['grad_cov'] - group['shuffled_cov']
    group['gap_vs_mean_svd'] = group['grad_cov'] - group['gradient_mean_svd']
    group['gap_vs_demeaned'] = group['grad_cov'] - group['demeaned_cov']
    require(len(group) == 180 and group.protocol_uid.nunique() == 15, 'Group capture support differs')
    return result, joined, group, dict(seed=seed, public_manifest_sha256=digest(folder / 'P1_PUBLIC_COMPLETE.json'),
        verified_public_files=len(manifest['public_files']), p0_checkpoint_sha256=receipt['selected_checkpoint_sha256'],
        p0_metrics_sha256=digest(p0 / 'SOURCE_VALIDATION_CELL_METRICS.csv'),
        loss_mae_max_difference=float(joined.p0_p1_loss_absolute_difference.max()),
        maximum_replay_gradient_absolute_error=max(item['max_absolute_error'] for item in replay['records']))


def associations(cells):
    records = []
    for (seed, layer), values in cells.groupby(['seed', 'layer'], sort=True):
        require(len(values) == 30 and values.cell_uid.nunique() == 30, 'Association requires independent physical N30')
        for feature in FEATURES:
            valid = np.isfinite(values[feature]) & np.isfinite(values.base_validation_mae)
            x = values.loc[valid, feature].to_numpy(float)
            y = values.loc[valid, 'base_validation_mae'].to_numpy(float)
            supported = len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
            for method in ('pearson', 'spearman'):
                left, right = x, y
                if supported and method == 'spearman':
                    left, right = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
                coefficient = float(np.corrcoef(left, right)[0, 1]) if supported else np.nan
                records.append(dict(seed=int(seed), layer=layer, rank=4, feature=feature,
                    response='matching_seed_P0_B0_source_meta_validation_cell_MAE', method=method,
                    physical_cells=30, finite_cells=len(x), exact_protocol_groups=values.protocol_uid.nunique(),
                    coefficient=coefficient, status='DESCRIPTIVE' if supported else 'UNDEFINED_INSUFFICIENT_VARIATION',
                    selection_or_gate_use=False, inference='no p-value, no causal or predictive-update claim'))
    return pd.DataFrame(records)


def plot_associations(table, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'legend.frameon': False})
    titles = {'gradient_norm': 'a  Gradient norm', 'capture_grad_cov': 'b  Gradient-subspace capture',
              'capture_gap_vs_random': 'c  Capture gap versus random',
              'capture_gap_vs_weight_svd': 'd  Capture gap versus weight SVD'}
    colors = {42: '#397FA6', 123: '#87929C', 2026: '#A280AF'}
    markers = {42: 'o', 123: 's', 2026: '^'}
    styles = {42: '-', 123: '--', 2026: ':'}
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.9), sharex=True, sharey=True)
    labels = [f'L{block} {projection.upper()}' for block in range(6) for projection in ('q', 'v')]
    for axis, (feature, title) in zip(axes.flat, titles.items()):
        for seed in sorted(table.seed.unique()):
            values = table.loc[table.feature.eq(feature) & table.method.eq('spearman') & table.seed.eq(seed)]
            values = values.set_index('layer').reindex(LAYERS)
            axis.plot(range(12), values.coefficient, marker=markers[int(seed)], linestyle=styles[int(seed)],
                      markersize=2.9, linewidth=1, color=colors[int(seed)], label=f'Seed {seed}')
        axis.axhline(0, color='#B8BEC3', linestyle='--', linewidth=.65)
        axis.set(title=title, ylim=(-1.04, 1.04), xticks=range(12), xticklabels=labels)
        axis.tick_params(axis='x', labelrotation=55)
    axes[0, 0].set_ylabel('Spearman correlation with MAE')
    axes[1, 0].set_ylabel('Spearman correlation with MAE')
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.002))
    fig.suptitle('Source-meta validation · 30 physical cells per seed/layer · descriptive associations', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95), w_pad=1.5, h_pad=1.6)
    for suffix in ('png', 'pdf'):
        metadata = {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None
        fig.savefig(directory / ('P1_ERROR_ASSOCIATIONS.' + suffix), dpi=600,
                    bbox_inches='tight', facecolor='white', metadata=metadata)
    plt.close(fig)


def make_report(gate, results, audits, correlation, controls, angles):
    lines = ['# P1 三种子源梯度子空间与验证误差审计', '',
        f'**P1预注册捕获分支为{gate["status"]}，H1状态为`{gate["H1_status"]}`。**', '',
        '本阶段使用各seed匹配的P0正式B0预训练冻结模型、已选native_multivariate Bridge和相同cycle-query Head。'
        'source85颗/47个exact组估计基方向，meta-validation30颗/15组只用于源开发留出诊断。'
        'P1预注册提交为04b6f79dda51e3472a40db33f1fc2fc01fea2b59。', '',
        '## 实际计算与身份闭环', '',
        '| seed | source/meta | 主矩阵 | 次矩阵 | 梯度前后向次数 | optimizer updates | 端点重放层梯度 | loss/MAE最大差 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for result, audit in zip(results, audits):
        lines.append(f'| {result["seed"]} | 85/30 | 12 Q/V | 42 Q/K/V/O/FFN | '
            f'{result["total_gradient_forward_backward_evaluations"]} | {result["optimizer_updates"]} | 120 | '
            f'{audit["loss_mae_max_difference"]:.3g} |')
    lines += ['',
        f'三个seed合计{sum(r["total_gradient_forward_backward_evaluations"] for r in results)}次真实梯度前后向、0次optimizer update。'
        '每seed315次为source/meta的nominal、绝对单位及source标签置换梯度，10次为独立首末电芯重放，20次为次要候选家族。'
        '每次前后向同时提取登记层的梯度。次42个矩阵包含主Q/V矩阵，家族覆盖分别记录。', '',
        '主集合为六层Q/V共12个384×384矩阵，rank4是预登记主值。r2/4/8/16/40/56/80的谱与投影保留在逐seed文件。'
        '次要Q/K/V/O/FFN共42个矩阵只用事先固定source12/meta8，记录mean-SVD谱和individual capture。'
        '次家族诊断保持描述性，原Q/V target set继续固定。', '',
        'source模型before/after指纹逐seed相同，requires_grad和train/eval标志由原审计恢复。公开完成清单全部文件的bytes/SHA重新核验。'
        '每seed10个首末端点的120个层梯度独立重放通过。P1 nominal meta loss逐层与同seed正式B0的逐cell验证MAE核对，绝对容差1e-5，观测点数与UID/protocol一致。', '',
        'P0在服务器完成后出现连接失败，P1切换本机RTX4060 Laptop、Torch2.4.1+cu121执行，结果及缓存保存在本地。'
        '源梯度缓存与U/V张量有独立manifest，公共诊断文件保存身份和统计。', '',
        '## 主rank4留出捕获与稳定性', '',
        '| 控制 | grad_cov−控制的平均捕获差 | 配对protocol 95%区间 | 有效工况组 | 三seed方向均正 |',
        '|---|---:|---|---:|---|']
    for comparison in gate['comparisons']:
        lines.append(f'| {comparison["reference"]} | {comparison["mean_capture_gap"]:.8f} | '
            f'[{comparison["paired_protocol_CI_low"]:.8f}, {comparison["paired_protocol_CI_high"]:.8f}] | '
            f'{comparison["valid_protocol_groups"]} | {comparison["all_registered_seed_directions_positive"]} |')
    lines += ['',
        '捕获为||UᵀGV||²_F/||G||²_F，U/V来自source梯度双侧二阶能量方向。random_mean平均20个随机正交基控制。'
        '先按层汇总同一留出工况的配对差，再配对三个seed中的同一exact protocol，以15个工况为单位重抽2000次。'
        '该区间条件于已拟合的三个source seed，seed方向与seed间波动在P1_GATE中单独保留。', '',
        f'基重合bootstrap均值为{gate["basis_stability_mean"]:.8f}，每seed做30次source工况/电芯重抽基拟合，左右基投影重合均纳入。', '',
        '| seed | 对random平均捕获差 | 对weight-SVD平均捕获差 | 基重合均值 |', '|---:|---:|---:|---:|']
    comparisons = {item['reference']: item for item in gate['comparisons']}
    for seed in SEEDS:
        lines.append(f'| {seed} | {comparisons["random_mean"]["seed_gaps"][str(seed)]:.8f} | '
            f'{comparisons["weight_svd"]["seed_gaps"][str(seed)]:.8f} | {gate["basis_stability_by_seed"][str(seed)]:.8f} |')
    lines += ['',
        '预注册捕获门槛要求相对random和weight-SVD平均差至少0.02、配对protocol区间下界大于0、三个seed方向均正，以及平均基重合至少0.5。'
        f'实际Gate状态为{gate["status"]}，完整数值与输入SHA绑定在P1_GATE.json。', '',
        '## mean-SVD与组内去均值控制', '',
        '现有CSV方法名为gradient_mean_svd，对47个源工况的平均梯度等权合并后做SVD。'
        'grad_cov对源工况等权、组内电芯等权的梯度二阶能量构造双侧基。'
        'demeaned_cov先从每颗源梯度减去其工况平均梯度，再以相同分层权重估计二阶能量。'
        '组内去均值有重复电芯支持的源工况为19/47，其余singleton保留零残差贡献。', '',
        '| seed | 方法 | meta工况均值梯度capture | meta单cell capture均值 | grad_cov−该方法（工况） |',
        '|---:|---|---:|---:|---:|']
    for row in controls.itertuples(index=False):
        lines.append(f'| {row.seed} | {row.method} | {row.protocol_mean_capture:.8f} | '
            f'{row.individual_cell_mean_capture:.8f} | {row.grad_cov_capture_gap:.8f} |')
    lines += ['',
        '各seed和逐层值完整保留在P1_MEAN_AND_DEMEANED_CONTROLS.csv与P1_LAYER_CAPTURE_SUMMARY.csv。'
        'mean-SVD对照在seed123的捕获差明显大于另两个seed，描述性汇总保留该不均匀性。'
        '以上复用既有梯度与控制，原预注册Gate继续保留。', '',
        'P1_BASIS_STABILITY的30次refit对应grad_cov。全局gradient_mean_svd和demeaned_cov的专属bootstrap稳定性尚有缺口。'
        '现有主角度矩阵比较各源工况mean-gradient SVD的左右基，补充概要如下，估计对象为跨源工况一致性。', '',
        '| seed | 基侧 | 工况对×层记录数 | 投影重合均值 | 平均主角度中位数（度） |',
        '|---:|---|---:|---:|---:|']
    for row in angles.itertuples(index=False):
        lines.append(f'| {row.seed} | {row.side} | {row.pair_layer_records} | {row.projection_overlap_mean:.6f} | '
            f'{row.mean_angle_degrees_median:.4f} |')
    lines += ['',
        '工况对共享源工况，跨层也共享输入，记录数保留组合覆盖含义。'
        'LoRA-SB已有mean-gradient导出的冻结基与可训练小核心。P1当前控制提供捕获差，mean-SVD基下的完整下游训练对照仍待补齐。'
        '后续相对random或LoRA-XS的P2改善，允许归因于实际被比较的方向与参数预算；相对LoRA-SB的优越性需要对应忠实下游对照。', '',
        '## 标签置换与容量单位边界', '',
        f'标签置换弱化观测为`{gate["permutation_capture_weakening_observed"]}`。真实标签梯度与shuffled_cov的差及区间见上表。'
        '当前置换控制缺少退化结局特异性证据。共享梯度捕获可能包含固定Head、Bridge、单cell梯度低秩及共同响应结构的贡献。'
        'H1通过范围为固定source checkpoint上的稳定共享梯度方向，退化特异性和电池特异机制继续保持未建立。', '',
        'source名义容量只有1.1Ah一个取值。absolute_units控制在相同checkpoint上把I/Q/drift恢复A/Ah尺度，再观察梯度；'
        '该控制描述输入单位扰动敏感性，跨容量/跨化学体系的不变性仍待独立数据验证。', '',
        '零更新P1未运行留出模型优化分支，捕获收益表示梯度投影能量差，预测更新收益留待P2实证。', '',
        '## 梯度、投影与验证误差的描述性关联', '',
        '每个seed/layer固定30颗独立meta电芯，将gradient_norm、grad_cov/random_mean/weight-SVD捕获及两个捕获差，'
        '分别与同seed正式B0的逐cell MAE计算Pearson和Spearman。原始合并表为P1_ERROR_GRADIENT_CAPTURE_CELL.csv，'
        '完整系数表为P1_ERROR_ASSOCIATIONS.csv。每层记录N30及15个工况支持，常量或缺少有限支持时保留未定义状态。'
        '关联计算属于补充描述分析，无p值、无新增Gate或选层用途。', '',
        '| seed | 量 | 层数 | Spearman中位数 | 12层范围 |', '|---:|---|---:|---:|---|']
    for (seed, feature), values in correlation.loc[correlation.method.eq('spearman')].groupby(['seed', 'feature']):
        finite = values.coefficient.dropna()
        if len(finite):
            lines.append(f'| {seed} | {feature} | {len(finite)} | {finite.median():.5f} | [{finite.min():.5f}, {finite.max():.5f}] |')
    lines += ['',
        '![P1逐层梯度与源验证误差的描述性关联](figures/P1_ERROR_ASSOCIATIONS.png)', '',
        '图展示全部12个主QV矩阵和三个source seed的Spearman系数。每点使用相同30颗物理电芯；'
        '跨seed共享电芯、同cell跨层及工况相关性保留在解释中。Spearman衡量秩关联，Pearson原值线性关联同时完整输出。'
        '这些关系用于描述当前checkpoint的误差结构，因果性、误差可预测性及真实PEFT更新改善仍需对应实验。', '',
        '## 优势与劣势（风险）', '',
        '- 优势为source/meta工况和物理UID互斥、三seed配对、梯度端点重放、source状态指纹及明确的random/W-SVD/label-shuffle控制。',
        '- 劣势（风险）为label-shuffle控制缺少弱化、source容量恒定、有限47源工况/15元验证工况，以及固定Head/Bridge可能产生共同梯度方向。'
        'source-meta已用于P0选择，队列也有历史探索暴露，当前证据属于retrospective源开发。', '',
        '## 目标隔离、后续阶段与复现', '',
        '全部最终target Query未来标签均未用于本阶段梯度、基拟合、关联或Gate计算。cohort缓存可整体载入内存，'
        '监督用途由注册UID角色限制。现有历史探索暴露已记录，评估解释保留该前史。', '',
        'P2按原登记矩阵比较标准LoRA、忠实LoRA-XS、weight-SVD core、gradient-core、random-core和LoRA-GA，'
        '执行同rank与近参数匹配对照。P1共享捕获通过仍需P2预测误差证明方向来源价值，H2/H3等待P3/P4。', '',
        '```powershell', '& D:\\Anaconda\\envs\\tslib\\python.exe -m scripts.drs_peft.summarize_p1', '```', '',
        '锁定gradient_audit、run_p1、train、peft和全部配置保持原字节，本CLI仅审读既有产物并生成补充统计与报告。'
        'Gate由原aggregate入口生成，本CLI保留其原始决定。', '',
        '## 置信度与拓展思考', '',
        '文件身份、分母、零更新、loss/MAE闭合及预注册捕获判定的置信度为高，均有实际记录与重放。'
        '退化特异性、跨容量不变性和下游少样本收益的证据仍不足。下一步由P2真实预测比较区分稳定共享方向与可利用适配方向。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/drs_peft')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'report/drs_peft')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    folder = args.output_root / 'P1'
    available = [seed for seed in SEEDS if (folder / f'seed{seed}/P1_PUBLIC_COMPLETE.json').is_file()]
    if args.dry_run or available != list(SEEDS) or not (folder / 'P1_GATE.json').is_file():
        print(dict(status='READY' if available == list(SEEDS) and (folder/'P1_GATE.json').is_file() else 'PENDING_COMPLETE_P1_GATE',
                   available_seeds=available, required_seeds=list(SEEDS)))
        return
    gate = read_json(folder / 'P1_GATE.json')
    require(gate['seeds'] == list(SEEDS) and gate['status'] in ('PASSED', 'NEGATIVE'), 'P1 Gate is not a completed three-seed decision')
    require(gate['optimizer_updates'] == 0 and gate['total_gradient_forward_backward_evaluations'] == 1035 and
            gate['target_Query_future_labels_accessed'] is False, 'P1 Gate changed its source-only budget')
    registration = load_registration()
    results, cells, groups, audits = [], [], [], []
    for seed in SEEDS:
        result, joined, group, audit = verify_seed(folder / f'seed{seed}', seed, args.output_root, registration)
        results.append(result)
        cells.append(joined)
        groups.append(group)
        audits.append(audit)
    cell_table = pd.concat(cells, ignore_index=True)
    group_table = pd.concat(groups, ignore_index=True)
    control_rows = []
    for seed in SEEDS:
        selected_groups = group_table.loc[group_table.seed.eq(seed)]
        selected_cells = cell_table.loc[cell_table.seed.eq(seed)]
        for method, column in [('grad_cov', 'capture_grad_cov'),
                               ('gradient_mean_svd', 'capture_gradient_mean_svd'),
                               ('demeaned_cov', 'capture_demeaned_cov')]:
            control_rows.append(dict(seed=seed, method=method, rank=4, meta_protocol_groups=15,
                meta_physical_cells=30, main_layers=12,
                protocol_mean_capture=float(selected_groups[method].mean()),
                individual_cell_mean_capture=float(selected_cells[column].mean()),
                grad_cov_capture_gap=float((selected_groups.grad_cov-selected_groups[method]).mean()),
                own_basis_bootstrap_stability_available=method == 'grad_cov'))
    controls = pd.DataFrame(control_rows)
    angle_rows = []
    for seed in SEEDS:
        values = pd.read_csv(folder/f'seed{seed}/P1_PRINCIPAL_ANGLES.csv')
        require(values['rank'].eq(4).all(), 'Source group principal-angle rank differs')
        for side, subset in values.groupby('side'):
            angle_rows.append(dict(seed=seed, side=side, rank=4, source_protocol_groups=47,
                pair_layer_records=len(subset), projection_overlap_mean=float(subset.projection_overlap.mean()),
                mean_angle_degrees_median=float(subset.mean_angle_degrees.median()),
                scope='source-group mean-gradient SVD pairwise consistency; dependent group pairs'))
    angles = pd.DataFrame(angle_rows)
    for reference, column in [('random_mean', 'gap_vs_random'), ('weight_svd', 'gap_vs_weight_svd'), ('shuffled_cov', 'gap_vs_shuffled')]:
        actual = group_table.groupby(['seed', 'protocol_uid'])[column].mean().groupby('protocol_uid').mean().mean()
        expected = next(item['mean_capture_gap'] for item in gate['comparisons'] if item['reference'] == reference)
        require(np.isclose(actual, expected, atol=1e-10, rtol=1e-9), 'Independent capture-gap aggregation differs from Gate')
    correlation = associations(cell_table)
    cell_table.to_csv(folder / 'P1_ERROR_GRADIENT_CAPTURE_CELL.csv', index=False, lineterminator='\n')
    correlation.to_csv(folder / 'P1_ERROR_ASSOCIATIONS.csv', index=False, lineterminator='\n')
    controls.to_csv(folder / 'P1_MEAN_AND_DEMEANED_CONTROLS.csv', index=False, lineterminator='\n')
    angles.to_csv(folder / 'P1_SOURCE_GROUP_MEAN_SVD_ANGLES.csv', index=False, lineterminator='\n')
    layer_summary = group_table.groupby(['seed', 'layer'])[
        ['grad_cov', 'random_mean', 'weight_svd', 'shuffled_cov', 'gradient_mean_svd', 'demeaned_cov',
         'gap_vs_random', 'gap_vs_weight_svd', 'gap_vs_shuffled', 'gap_vs_mean_svd', 'gap_vs_demeaned']].mean().reset_index()
    layer_summary.to_csv(folder / 'P1_LAYER_CAPTURE_SUMMARY.csv', index=False, lineterminator='\n')
    figures = args.report_dir / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    plot_associations(correlation, figures)
    report = args.report_dir / 'P1_GRADIENT_AUDIT_ZH.md'
    report.write_text(make_report(gate, results, audits, correlation, controls, angles), encoding='utf-8', newline='\n')
    write_json(folder / 'P1_SUMMARY_AUDIT.json', dict(status='PASSED_SUMMARY_VERIFICATION',
        source_gate_status=gate['status'], source_gate_sha256=digest(folder / 'P1_GATE.json'), seeds=list(SEEDS),
        gradient_forward_backward_evaluations=1035, optimizer_updates=0,
        target_Query_future_labels_used=False, physical_cells_per_seed_layer=30,
        association_rows=len(correlation), merged_cell_layer_rows=len(cell_table),
        descriptive_only=True, changes_gate_or_target_selection=False, audits=audits,
        summary_script_sha256=digest(Path(__file__)), helper_sha256=digest(ROOT/'scripts/drs_peft/summarize_p0.py'),
        outputs=[dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path)) for path in (
            folder/'P1_ERROR_GRADIENT_CAPTURE_CELL.csv', folder/'P1_ERROR_ASSOCIATIONS.csv',
            folder/'P1_LAYER_CAPTURE_SUMMARY.csv', folder/'P1_MEAN_AND_DEMEANED_CONTROLS.csv',
            folder/'P1_SOURCE_GROUP_MEAN_SVD_ANGLES.csv', figures/'P1_ERROR_ASSOCIATIONS.png',
            figures/'P1_ERROR_ASSOCIATIONS.pdf', report)]))
    print(dict(status='PASSED_SUMMARY_VERIFICATION', source_gate=gate['status'],
               seed_count=3, gradient_evaluations=1035, optimizer_updates=0,
               association_rows=len(correlation), merged_cell_layer_rows=len(cell_table)))


if __name__ == '__main__':
    main()
