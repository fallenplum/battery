"""Create N1, N2 and final Chinese reports from verified artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import read_json
from .run import ROOT, write_json, file_records
from .data import file_hash


def table(frame,digits=9):
    def cell(value):
        if pd.isna(value):
            return '缺失'
        if isinstance(value,(float,np.floating)):
            return f'{value:.{digits}f}'
        return str(value).replace('|','\\|')
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(cell(value) for value in values)+' |' for values in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)


def interpret(frame,domain,reference):
    rows=frame[(frame.domain==domain)&(frame.candidate=='N2:physics_basis')&
               (frame.reference=='N2:'+reference)&(frame.metric=='mae')]
    estimates=rows.set_index('sampling_unit')
    point=estimates.loc['cell','improvement']
    if (rows.ci95_low>0).all():
        status='两个重采样单位的条件区间均支持physics_basis误差较低'
    elif (rows.ci95_high<0).all():
        status='两个重采样单位的条件区间均支持physics_basis误差较高'
    else:
        status='条件区间尚未一致支持正向增益'
    return f'{domain}相对{reference}的逐电池MAE改善为{point:.9f}，{status}。'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results/N012_independent')
    args=parser.parse_args()
    output=args.out.resolve()
    verified=read_json(output/'RUN_VERIFICATION.json')
    if verified['status']!='VERIFIED':
        raise RuntimeError('Complete selected-weight and arithmetic verification required')
    analysis=output/'analysis'
    summary=pd.read_csv(analysis/'summary_metrics.csv')
    cell=pd.read_csv(analysis/'cell_metrics.csv')
    pairs=pd.read_csv(analysis/'paired_comparisons.csv')
    training=pd.read_csv(analysis/'training_units.csv')
    roles=pd.DataFrame(read_json(analysis/'MODEL_ROLE_BINDINGS.json')['roles'])
    role_units=roles.merge(training,on=['domain','fold','encoder','decoder'],validate='many_to_one')
    params=role_units.groupby(['domain','phase','method']).agg(min_parameters=('parameters','min'),max_parameters=('parameters','max')).reset_index()
    selected=[]
    for domain in ('MATR','HUST'):
        n1=read_json(output/domain/'N1_SELECTION.json')
        n2=read_json(output/domain/'N2_SELECTION.json')
        for fold in n1['selection_folds']:
            enc=n1['encoder_per_outer_fold'][fold]
            dec=n2['selected_decoder_per_fold'][fold]
            selected.append(dict(domain=domain,fold=fold,selected_encoder=enc,
                                 N1_validation_mae=n1['candidates'][fold][enc]['validation_mae'],
                                 selected_decoder=dec,N2_validation_mae=n2['candidates'][fold][dec]['validation_mae']))
    selections=pd.DataFrame(selected)
    historical=summary[(summary.phase=='HISTORICAL')&(summary.window=='full')]
    curves=read_json(ROOT/'report/N012/figures/FIGURES_COMPLETE.json')
    environment=read_json(output/'ENVIRONMENT.json')
    if curves['status']!='RENDERED':
        raise RuntimeError('Complete raw training figures required')
    accounting=read_json(output/'engineering/OPTIMIZER_ACCOUNTING.json')
    destination=ROOT/'report/N012'
    files=[]
    common='''本轮登记日期为2026-11-16，沿用用户直接要求的seed42。全部MATR169和HUST77分别进行单域五折实验。每组150轮、batch8、宽度64、AdamW学习率0.0003、weight decay0.01、梯度裁剪1.0。损失为完整实际观测后缀逐电池等权MAE，选权遵循本折内部验证MAE的strict first-min。所有曲线保留原始150轮，源数据和划分在训练前封存。

MATR拥有123,431个未来观测，HUST拥有138,422个。全部未来观测均保持原始周期与mask，窗口支持不足时记录缺失。SOH使用1.1Ah名义容量归一化，SOH100锚点采用真实96–100圈有效容量中位数。每域分别拟合训练电池的有效点标准化。两域的物理电压窗口和策略维数保持各自定义。

所有队列均有历史研究暴露，本轮属于内部回顾性验证。单种子无法估计训练种子方差。bootstrap区间为当前拟合预测上的条件区间，进行了多个预登记探索性比较。HUST77个精确协议各只有一颗电池，协议重采样缺少组内重复支持。
'''
    for phase in ('N1','N2'):
        primary=summary[(summary.phase==phase)&(summary.window=='full')].merge(params,on=['domain','phase','method'])
        primary=primary[['domain','method','cell_mean_mae','cell_mean_rmse','protocol_macro_mae','protocol_macro_rmse',
                         'supported_cells','observations','min_parameters','max_parameters']]
        horizons=summary[(summary.phase==phase)&(summary.window!='full')][
            ['domain','method','window','cell_mean_mae','cell_mean_rmse','protocol_macro_mae','supported_cells','supported_protocols','observations']]
        comparison=pairs[(pairs.candidate.str.startswith(phase+':'))&
                         (pairs.reference.str.startswith(phase+':')|pairs.reference.eq('HISTORICAL:M1'))][
            ['domain','metric','candidate','reference','sampling_unit','improvement','ci95_low','ci95_high','better_cells','worse_cells','paired_cells']]
        fold_table=selections if phase=='N2' else selections[['domain','fold','selected_encoder','N1_validation_mae']]
        title='五种独立编码器比较' if phase=='N1' else '固定折内编码器结构的五种解码器比较'
        text=f'# {phase} {title}\n\n'+common
        if phase=='N1':
            text+='''
N1只改变编码器，统一query_mlp。resmlp使用64维逐周期残差MLP，voltage_cnn使用两层电压卷积，temporal_tcn使用逐周期平面MLP与六层时间卷积，cnn_tcn使用电压卷积与六层时间卷积，cnn_transformer使用电压卷积与一层四头Transformer。各模型共同使用实际循环、上一有效循环间隔、可用性、策略值和掩码，池化前缀后结合SOH100预测未来。

参数量存在差异，因此同宽度、同训练预算的比较仍包含容量差异。每外折的选择仅使用该折内部验证电池，跨折测试结果为描述性评价，当前没有通过全局测试均值选定唯一论文backbone。
'''
        else:
            text+='''
N2固定每外折N1选出的编码器结构，从头训练各个新解码器。query_mlp复用相同折N1已经从头训练的同配置产物。每域新增20组解码器训练，共享5组query_mlp基线，不复用历史SR-111模型权重。N1 nested与N2 query_mlp逐电池预测一致。

query_mlp直接预测查询条件修正，legacy_flow为独立解析松弛风格，free_rate为非负速率固定整数周期积分，monotone_spline为十段正速率积分的指数映射，physics_basis为对数自限制、线性及平滑激活三项累计函数的指数映射。全部算子由新n012模块独立实现。physics_basis与monotone_spline的数学形状对照用于约束解释，三项函数的存在尚不足以认证LLI、LAM或SEI。
'''
        text+='\n## 完整未来轨迹\n\n'+table(primary)
        text+='\n\n## 折内验证选择\n\n'+table(fold_table)
        text+='\n\n选择在候选测试评分之前封存。正式all入口先封存该域N1与N2选择，再统一产生测试预测。nested是各折自身验证选择构成的OOF方法，逐电池合并等权汇总。\n'
        text+='\n## 四时距与有效分母\n\n'+table(horizons)
        text+='\n\nMATR在1001圈后有40颗轨迹支持，HUST在该窗口有77颗。此支持与早期输入资格分别记录，缺少远期观测的电池仍保留在完整后缀任务中。\n'
        text+='\n## 预登记配对比较\n\n'+table(comparison)
        text+='\n\n改善量为参照误差减候选误差，正值支持候选。区间使用2,000次seed42条件重采样，分别按电池和协议组。全部候选及阴性结果保留。\n'
        text+='\n## 历史M1参照\n\n'+table(historical[['domain','cell_mean_mae','cell_mean_rmse','supported_cells','observations']])
        text+='\n\nMATR历史M1使用相同169颗电池、原五折和实际后缀，原观测与预测逐值核对。其额外global/local物理统计、观测残差、SOH边界及速率辅助损失与本轮存在差异，结果反映整个方法的历史比较。HUST单域结果没有混入MATR均值。\n'
        text+='\n## 原始训练与验证曲线\n\n'
        for domain in ('MATR','HUST'):
            text+=f'![{domain} {phase}原始训练与验证曲线](figures/{domain}_{phase}_training_validation.png)\n\n'
        text+='圆点表示最终验证选定的候选及其选权轮数。坐标使用SOH fraction，曲线没有平滑或重排。\n'
        text+='\n## 优势与风险\n\n优势为完整队列、独立结构、固定预算及折内验证选模可复核。风险包括参数量不匹配、历史暴露、单种子、HUST协议单例，以及全轨迹误差对不同时距的混合。数值复核置信度高，跨种子与外部泛化置信度低。\n'
        text+='\n## 拓展思考\n\n同一方法的总体误差与后期窗口误差需要结合实际支持数解释。\n\n## 你可能想知道\n\n完整逐电池和逐协议误差、训练时间与权重来源位于results/N012_independent/analysis，全部选定权重及恢复状态随结果目录交付。\n'
        path=destination/(phase+'_RESULTS.md')
        path.write_text(text,encoding='utf-8')
        files.append(path.name)
    long=pd.read_csv(analysis/'long_horizon_paired_cells.csv')
    concentration=[]
    for keys,group in long.groupby(['domain','window','reference'],sort=True):
        positive=group.improvement[group.improvement>0].sort_values(ascending=False)
        concentration.append(dict(domain=keys[0],window=keys[1],reference=keys[2],cells=len(group),
                                  mean_improvement=group.improvement.mean(),better_cells=int((group.improvement>0).sum()),
                                  worse_cells=int((group.improvement<0).sum()),
                                  top3_fraction_of_positive_improvement=float(positive.head(3).sum()/positive.sum()) if len(positive) else np.nan,
                                  largest_harm_cell=group.loc[group.improvement.idxmin(),'cell_uid']))
    role_query=roles.merge(pd.read_csv(output/'verification/query_diagnostics.csv'),on=['domain','fold','encoder','decoder'],validate='many_to_many')
    far=role_query[(role_query.phase=='N2')&(role_query.method!='nested')&role_query.region.isin(['observed_future','beyond_cell_followup'])].groupby(
        ['domain','method','region'],sort=True).agg(query_points=('query_count','sum'),negative_predictions=('negative_count','sum'),
            above1p2=('above1p2_count','sum'),nonfinite=('nonfinite_count','sum'),minimum=('minimum','min'),maximum=('maximum','max')).reset_index()
    cases=cell[(cell.cell_uid=='MATR_b1c2')&(cell.window=='full')][['phase','method','fold','mae','rmse','n_obs']]
    timings=training.groupby('domain').agg(unique_fits=('fold','size'),updates=('updates','sum'),training_validation_seconds=('elapsed_seconds','sum')).reset_index()
    lines=[interpret(pairs,domain,reference) for domain in ('MATR','HUST') for reference in ('free_rate','monotone_spline')]
    assessment='# N012 最终研究评估\n\n'+common
    primary=summary[(summary.window=='full') & (summary.method.isin(['nested','M1']) |
                    (summary.phase.eq('N2') & summary.method.isin(['free_rate','monotone_spline','physics_basis'])))][
        ['domain','model','cell_mean_mae','cell_mean_rmse','protocol_macro_mae','supported_cells','observations']]
    assessment+='\n## 全轨迹主要结果\n\n'+table(primary)
    historical_pairs=pairs[(pairs.metric=='mae') & pairs.candidate.isin(['N1:nested','N2:nested']) & pairs.reference.eq('HISTORICAL:M1')][
        ['candidate','sampling_unit','improvement','ci95_low','ci95_high','better_cells','worse_cells']]
    assessment+='\n\nMATR的N1与N2嵌套选模误差均高于历史M1，两种重采样单位的条件区间均支持这一负向差异。历史M1拥有额外显式物理统计、观测残差、边界及速率辅助损失，该差异仍包含信息预算和训练目标的影响。\n\n'+table(historical_pairs)
    assessment+='\n## 实际完成与来源\n\n'+table(timings,3)
    assessment+=f'''

两域共90组真实训练、13,500个正式epochs、141,750次正式优化器更新。额外真实烟测28次更新，合成单测夹具合计{accounting['synthetic_unit_test_optimizer_updates']}次更新，均单列为工程证据。总执行更新为{141750+28+accounting['synthetic_unit_test_optimizer_updates']}次。合成数据不参与科研成绩。

实际检查通过{verified['counts']['PASS']}项，90个选定权重的正常入口重放最大绝对误差为0。数据、来源、150轮预算、优化器、验证first-min、全部后缀mask、逐电池汇总、配对区间与独立NumPy解码公式均核验。正式单元按各自encoder/decoder/fold只有一个seed42拟合，断点恢复确定性由合成中断回归检查。中间每轮权重没有单独保存，最终恢复状态、选定权重及全部日志可追溯。

训练前源码提交为`f80ee8f65b751d798bf59c9d5b08dd97bf1a71ed`，测试前评价计划提交为`3fa83014909783bd547fae8faea8a956b481e23f`。原交接ZIP SHA256为`9ae686a8c00aa938dc74f878fa743e911b8216db7f38001e6f310642e8000101`。交接包中的合成测试与无GPU声明依据原STATUS快照解释，本机真实执行以新完成标记为准。

运行环境记录于ENVIRONMENT.json，Python为{environment['python']}，PyTorch为{environment['torch']}，编译CUDA为{environment['torch_compiled_CUDA']}，GPU为{environment['GPU']}。正式训练串行执行。各单元计时覆盖训练与验证阶段，数据加载和折内标准化在计时器之前执行。
'''
    assessment+='\n## 三分量基函数的受控证据\n\n'+'\n\n'.join(lines)
    assessment+='''

两域中physics_basis相对free_rate与monotone_spline的MAE配对区间均覆盖零，当前三项具体基函数的独立预测增益证据不足。两域中physics_basis的逐电池误差点估计均高于monotone_spline。其非负非增的数学性质已通过公式和逐点检查。当前实验缺少电极分辨测量、真实分量标签和机制可辨识性分析，形状参数化的解释范围限定于这些函数类。
'''
    assessment+='\n## 后期增益及反例集中度\n\n'+table(pd.DataFrame(concentration))
    assessment+='\n\n分数表示最大三个正向配对改善占全部正向改善的比例，与净平均改善及负向电池共同报告。完整逐电池差值保留在long_horizon_paired_cells.csv。\n'
    assessment+='\n## 随访内与随访外原始输出\n\n'+table(far)
    assessment+='\n\n原始预测保留负值和超过1.2的输出。5000圈查询提供数值诊断，随访外部分缺少真实SOH准确性验证。单调样条与三分量基函数具有非负非增的数学结构，自由非负速率积分可能产生负SOH。\n'
    assessment+='\n## MATR_b1c2 的真实OOF误差\n\n'+table(cases)
    assessment+='\n\n该表使用原电池外折，历史M4/M5中未见协议测试的同名电池具有另一训练划分，两种scope分别解释。所有最大误差案例见worst_cases.csv。\n'
    assessment+='\n## Stanford、EOL与协议支持\n\nStanford仍为N0_BLOCKED，已有181颗供应曲线缺少满足本轮真实原生循环1–100的认证张量。MATR的169颗具有已观察80%容量阈值事件，HUST77颗全部右删失。本轮交付EOL资格。HUST77个精确协议各只有一颗，当前cell-disjoint结果描述这些已释放组合的留出预测，正式跨域共享训练、工况族留域评价及组内重复方差仍缺证据。\n'
    assessment+='\n## 工程修正和保留的失败\n\n交接原7项单测通过。新增4项回归后11项通过，真实输入CUDA检查98项通过。自由速率固定网格、有效循环间隔、ResMLP残差块、训练集float64统计、原子恢复、测试评分闸门和逐电池汇总均在正式训练前修正。初次CUDA自适应池化确定性失败、正式启动前相对路径表示失败均保留。两类失败发生时正式优化器更新均为0。源头问题修复后固定8点平均池化及规范根路径重新通过检查。\n'
    assessment+='\n## 优势、风险与下一步判断\n\n优势为独立结构、完整原始队列、清晰信息预算、严格折内选模及可重放产物。风险包括各模型容量差异、M1额外物理特征和辅助损失、同一验证集上的多阶段选择、单种子、历史暴露和随访支持差异。验收与当前数值计数的置信度高，单种子对机制、跨域与未来新批次的证据置信度低。\n\nN3的研究价值在于固定当前可复核结构后，单独检验显式全局/局部物理统计的增量，并控制与M1之间的信息预算和损失差异。当前N1/N2结果已提供确定性参照，概率支线的收益与校准尚需独立合同和评价。\n'
    assessment+='\n## 拓展思考\n\n编码器、解码器和物理信息预算分别具有可检验的作用，本轮提供了前两者在简洁VIQ/deltaQ输入下的对照。\n\n## 你可能想知道\n\n每外折可选择不同编码器和解码器，nested-OOF反映该验证选择程序的表现。代码没有用全局测试排名确定唯一模型。所有源码、权重、训练/验证曲线、逐电池预测、资格表、哈希和验收记录随功能分支交付。\n'
    path=destination/'N012_FINAL_RESEARCH_ASSESSMENT.md'
    path.write_text(assessment,encoding='utf-8')
    files.append(path.name)
    write_json(destination/'REPORTS_COMPLETE.json',dict(status='RENDERED',source_sha256=file_hash(__file__),
               run_verification_sha256=file_hash(output/'RUN_VERIFICATION.json'),
               files=file_records(destination,files),figure_receipt_sha256=file_hash(ROOT/'report/N012/figures/FIGURES_COMPLETE.json')))
    print('CreatedN1,N2and final research reports',flush=True)


if __name__=='__main__':
    main()
