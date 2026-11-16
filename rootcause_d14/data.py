"""Read-only adapters, exact endpoint targets, train-only feature statistics."""
from dataclasses import dataclass
from pathlib import Path
import json
from functools import lru_cache
import numpy as np
from n012.data import load_bundle, TrainOnlyScaler, role_indices, file_hash
from n012.run import ROOT, write_json

CONFIG = ROOT / 'configs/D1_D4_seed42.json'
OUT = ROOT / 'results/D1_D4'
REPORT = ROOT / 'report/D1_D4'
SCOPE = 'LEGAL_EARLY_FORECAST'
ORACLE = 'ORACLE_DIAGNOSTIC'
FEATURES = {'G': ('global_features', 'global_mask'),
            'R': ('local_residual', 'local_mask'),
            'L': ('local_features', 'local_feature_mask')}


@lru_cache(maxsize=1)
def settings():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


@dataclass
class Domain:
    bundle: object
    arrays: dict
    rates: np.ndarray
    rate_mask: np.ndarray
    source: Path


def load_domain(name, root=ROOT):
    b = load_bundle(root, name)
    source = root / ('data/unified_matr_seed42/early_viq_qv_64.npz' if name == 'MATR'
                     else 'data/hust_raw_idea_v2/views/actual2C_loaded_response/early_inputs.npz')
    with np.load(source, allow_pickle=False) as a:
        assert a['cell_uid'].astype(str).tolist() == b.ids
        arrays = {k: a[k].copy() for pair in FEATURES.values() for k in pair}
    for value, mask in FEATURES.values():
        v, m = arrays[value], arrays[mask]
        assert v.shape == m.shape and m.dtype == np.bool_
        assert np.isfinite(v[m]).all()
        assert not (m.reshape(len(b.ids),100,-1).any(-1) & (b.cycle > 100)).any()
    rates = np.zeros((len(b.ids),2), np.float32)
    rm = np.zeros_like(rates, bool)
    for j, (left,right) in enumerate(settings()['rate_endpoints']):
        li,ri = int(left-101),int(right-101)
        rm[:,j] = b.observed[:,li] & b.observed[:,ri]
        rates[:,j] = np.where(rm[:,j], (b.targets[:,li].astype(float)-b.targets[:,ri])/float(right-left), 0)
    if name == 'MATR':
        from scripts.unified_matr.train_seed42 import load_frozen_data
        frozen = load_frozen_data(root/'data/unified_matr_seed42')
        assert np.array_equal(rm, frozen[8])
        assert np.max(np.abs(rates[rm]-frozen[7][rm])) < settings()['endpoint_float32_tolerance']
        rates = frozen[7].copy()
    return Domain(b, arrays, rates, rm, source)


class PhysicsScaler:
    def fit(self, arrays, rows):
        self.stats = {}
        for key, mask in FEATURES.values():
            v = arrays[key][rows].astype(float)
            m = arrays[mask][rows]
            axes = tuple(range(v.ndim-1)) if key != 'local_residual' else (0,1,2)
            n = m.sum(axis=axes)
            mean = np.divide(np.where(m,v,0).sum(axis=axes),n,out=np.zeros_like(n,dtype=float),where=n>0)
            var = np.divide(np.where(m,(v-mean)**2,0).sum(axis=axes),n,out=np.zeros_like(n,dtype=float),where=n>0)
            self.stats[key] = dict(mean=np.asarray(mean).tolist(), std=np.maximum(np.sqrt(var),1e-5).tolist(),
                                   supported=np.asarray(n).tolist())
        return self

    def apply(self, arrays):
        out = {}
        for key,mask in FEATURES.values():
            s = self.stats[key]
            out[key] = np.where(arrays[mask], (arrays[key]-np.asarray(s['mean']))/np.asarray(s['std']),0).astype(np.float32)
            out[mask] = arrays[mask]
        return out


def masked_statistics(value, mask):
    """Fixed, low-dimensional proxy features with support fraction."""
    v = value.reshape(len(value),100,-1).astype(float)
    m = mask.reshape(v.shape)
    def moment(a,z):
        n = z.sum(1)
        return np.divide(np.where(z,a,0).sum(1),n,out=np.zeros_like(n,dtype=float),where=n>0)
    mean = moment(v,m)
    sd = np.sqrt(moment((v-mean[:,None])**2,m))
    diff = moment(v[:,80:],m[:,80:])-moment(v[:,:20],m[:,:20])
    return np.concatenate((mean,sd,diff,m.mean(1)),axis=1)


def scaled_inputs(d, fold):
    roles = role_indices(d.bundle,fold)
    raw = TrainOnlyScaler().fit(d.bundle,roles['train'])
    x,p = raw.apply(d.bundle)
    phy = PhysicsScaler().fit(d.arrays,roles['train'])
    return roles,x,p,phy.apply(d.arrays),dict(raw=raw.to_dict(),physics=phy.stats,
                                            training_ids=[d.bundle.ids[i] for i in roles['train']])


def selected_encoder(domain,fold):
    path = ROOT/f'results/N012_independent/{domain}/N1_SELECTION.json'
    return json.loads(path.read_text(encoding='utf-8'))['encoder_per_outer_fold'][fold]


def baseline_path(domain,fold,decoder='query_mlp'):
    return ROOT/f'results/N012_independent/{domain}/{fold}/{selected_encoder(domain,fold)}__{decoder}'


def audit():
    import subprocess
    from n012.run import bundle_fingerprint
    from n012.analyze import completed_units
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    config = settings()
    records, matrix, recovery = {}, [], []
    for name in config['domains']:
        d = load_domain(name)
        b = d.bundle
        _,old = completed_units(ROOT/'results/N012_independent',name)
        assert len(old) == 45
        with np.load(d.source) as a:
            m = a['point_mask']
            reconstructed = b.x[...,2]-b.x[...,3]
            reference_error = float(np.max(np.abs(reconstructed[m]-a['reference_viq'][...,2][m])))
            reference_nonq = [float(np.max(np.abs(a['reference_viq'][...,k][m]-a['raw_viq'][...,k][m]))) for k in (0,1)]
        records[name] = dict(prediction_scope=SCOPE,cells=len(b.ids),future_observations=int(b.observed.sum()),
            raw_shape=list(b.x.shape),policy_shape=list(b.policy.shape),query_shape=list(b.targets.shape),
            tensor_sha256=bundle_fingerprint(b),early_source_sha256=file_hash(d.source),
            rates_support=d.rate_mask.sum(0).tolist(),splits=b.folds,
            physical={k:dict(shape=list(d.arrays[k].shape),supported=int(d.arrays[m].sum()),
                        supported_cells=int(d.arrays[m].reshape(len(b.ids),-1).any(1).sum())) for k,m in FEATURES.values()},
            source_files=b.provenance['files'],old_units_verified=len(old))
        recovery.append(dict(domain=name,prediction_scope=SCOPE,reference_q_reconstruction_max_abs=reference_error,
                             reference_V_difference=reference_nonq[0],reference_I_difference=reference_nonq[1]))
        for fold in config['folds']:
            roles = role_indices(b,fold)
            for arm, spec in config['arms'].items():
                matrix.append(dict(prediction_scope=SCOPE,domain=name,fold=fold,arm=arm,encoder=selected_encoder(name,fold),
                    **spec,seed=42,epochs=150,training_cells=len(roles['train']),test_cells=len(roles['test']),
                    expected_updates=150*int(np.ceil(len(roles['train'])/8)),
                    comparator='D1-0 (reused N1 nested)',metric='cell_mean_MAE',minimum_effect=0.001,
                    scientific_role='retrospective_controlled_ablation',pass_rule='paired cell CI and effect size plus domain consistency',
                    redesign_rule='preserve negative result; no test-driven redesign',artifact=f'{name}/{fold}/{arm}'))
    import pandas as pd
    pd.DataFrame(matrix).to_csv(OUT/'04_EXPERIMENT_MATRIX.csv',index=False)
    pd.DataFrame(recovery).to_csv(OUT/'REFERENCE_RECOVERY_AUDIT.csv',index=False)
    write_json(OUT/'DATA_AUDIT.json',records)
    reg = dict(base_commit='16856167fbd49da6032073e9ee960f2e4bd52b4f',
        remote_base_verified_via='GitHub REST git/ref/heads/codex/n012-independent-seed42',
        configuration_sha256=file_hash(CONFIG),data_audit_sha256=file_hash(OUT/'DATA_AUDIT.json'),
        legal_units=80,expected_formal_updates=sum(r['expected_updates'] for r in matrix),
        legal_unique_cells=246,seed=42,scope=SCOPE,
        oracle_isolation='oracle fits are read only by diagnostics and reports; legal trainer imports no oracle module')
    write_json(OUT/'RUN_REGISTRATION.json',reg)
    text = '''# D1–D4 预注册与复用审计

本轮为已暴露队列上的单种子42回顾性研究。远端N012基线已由GitHub API核对为16856167fbd49da6032073e9ee960f2e4bd52b4f。

## 数据与比较单位

MATR169、HUST77均逐文件通过原冻结加载器验证，完整物理特征和五折UID见DATA_AUDIT.json。原N012两域90组训练逐文件认证，D1-0复用各折N1 validation选定的query_mlp预测。旧physics_basis、monotone_spline、free_rate与MATR历史M1保留为参照。Stanford继续N0_BLOCKED，未发现认证原生100圈新tensor。

原输入保留100×64×4及point/cycle/policy/mask/anchor。G为100×20，局部为100×64残差与100×8×26统计。特征先逐有效channel折内train标准化，以值和显式mask编码，周期/分段masked均值与last有效token汇聚，再残差加到原64维z。所有cell保留，采用union缺失策略。缺失统计不参与均值。capacity臂仅输入原z，参数数与GL支路相差小于1%，具体数记录每单元合同。

## 冻结矩阵

每域五折、每折8新增臂，共80合法训练单元，150epoch、batch8、AdamW lr0.0003、wd0.01、clip1。每折架构沿用N1本折validation选择。D1为G/L/GL/capmatch，D2为无特征+rate、GL+rate及预先登记的GL+rate两种形状decoder。完整清单见04_EXPERIMENT_MATRIX.csv。D1-0、F0L0复用旧query_mlp，F1L0复用D1-GL。融合固定residual-add。原encoder与decoder先按seed42初始化，额外支路用独立随机态初始化，保证各臂原主干初始参数相同。所有选权仅使用本折validation trajectory MAE的首个严格最优。

速率精确取200/500/1000真实端点，mask要求两个端点真实观测。MATR复用冻结标签并与float32端点差验证到5e-10。HUST从认证观测直接构造。辅助损失为0.1×1000×supported-cell平均的supported-window平均绝对误差。逐epoch记录损失尺度、梯度范数和固定训练probe的主辅梯度夹角。

## 诊断参数与隔离

D3使用float64和8初值，五函数族加physics共享/逐cell两模式。共享time_c与smooth仅由各折train未来曲线学习，逐cell额外自由度模式显式另列。首先在各折train/validation运行，正式结构和超参全部提交后才运行test oracle。诊断保存在ORACLE_DIAGNOSTIC目录，不作为神经网络输入、先验、尺度或初始化。

Physics边界沿用softplus正a/b/c、tau=4sigmoid、time_c>0.05、smooth>0.03。B保持10段指数累计函数。C使用8段非负SOH速率积分，D使用锚点固定四次多项式，E为单参数非负线性累计。共享学习8初值，个体所有初值与预测均保留。数值最优是搜索到的近似值，不能保证全局最优。

D4近等价集合同时满足MAE≤best×1.05及差≤0.002。额外1%与10%敏感性阈值预先冻结。所有OOF cell做缩放坐标Jacobian SVD，profile在fold0排序最前3个train和3个test UID固定进行，tau、加速幅度与两段有效速率网格见配置。rate proxy使用Ridge，alpha仅train/validation选，不合并trainval重拟合。冻结z轻头属于条件诊断。

## 指标与局限

主指标先逐cell观测future MAE/RMSE再cell等权，另报协议macro、四时距、偏差、EOL前后、回升、长寿命贡献、反例、参数量与数值越界。每张表均标prediction_scope。配对改善为reference−candidate，2000次cell及protocol bootstrap仅是单次拟合的条件区间。HUST协议单例，无法推断组内方差。本轮不升级论文方法，维持用户要求的单种子42。

优势为输入、辅助目标和函数族有固定受控对照。风险为历史测试暴露、单种子、参数量近似匹配与有限oracle优化，无法认证电化学机制。结论置信度按实际证据评级，不预定赢家。
'''
    (REPORT/'00_CONTRACT_AND_REUSE_AUDIT.md').write_text(text,encoding='utf-8')
    (REPORT/'UNMATCHED_INFORMATION.md').write_text('''# 信息预算审计

N012提供Q与ΔQ，逐有效点可由Q−ΔQ恢复参考Q，float32消减舍入误差见REFERENCE_RECOVERY_AUDIT.csv。原reference VIQ的V/I也已逐值比较。此代数可恢复性只适用于保留point mask的值。

实测MATR参考Q最大恢复误差为0，HUST为9.5367431640625e-7个存储容量单位。两域参考V与当前V相同。参考I与当前I存在差异，最大绝对差分别为0.3595829010和17.2298583984个存储电流单位。ΔQ只约束容量差，无法恢复独立变化的参考I，此项输入信息差异继续存在。

历史M1另外提供显式global/local分解、拟合权重、reference相关通路与结构性归纳偏置，并使用rate辅助、持续性惩罚及输出clamp。D1-GL补回六类物理值/mask，但其masked汇聚会压缩周期与分段细节。新增模型保留现有原始支路，未逐算子复刻旧网络。现有对照足以检查显式特征支路增量，对历史M1的信息与优化预算仍存在差异。

优势为参考Q的代数恢复与所有物理mask均可核查。风险为信息存在与网络能恢复之间存在表达和优化差异，不能将GL视为与历史M1完整等价。
''',encoding='utf-8')
    return reg
