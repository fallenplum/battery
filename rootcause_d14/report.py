"""Evidence-bounded Chinese reports and Python-only PNG/PDF figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from n012.run import write_json,file_records
from n012.report import table
from .data import OUT,REPORT,ROOT,SCOPE,ORACLE,settings


def md(frame, columns=None, scope=SCOPE):
    if columns:
        columns = ['prediction_scope']+columns if 'prediction_scope' in frame and 'prediction_scope' not in columns else columns
        frame = frame[columns].copy()
    else:
        frame = frame.copy()
    if 'prediction_scope' not in frame:
        frame.insert(0,'prediction_scope',scope)
    for name in frame.select_dtypes(include=['floating']).columns:
        values=frame[name]
        sensitive=(values.abs()>=1e6)|((values.abs()<1e-5)&values.ne(0))
        if sensitive.any():
            frame[name]=frame[name].astype(object)
            frame.loc[sensitive,name]=values[sensitive].map(lambda value:f'{value:.6g}')
    return table(frame,digits=6)


def common():
    return '''
本轮沿用种子42。MATR169、HUST77分别做原固定五折，原始K100与全部真实未来观测mask保持一致。训练使用150epoch、batch8、AdamW lr0.0003、wd0.01，选权为本折validation trajectory MAE首个严格最优。单位为SOH fraction。全部合法成绩按物理cell观测点先平均，再在cell间等权。当前队列存在历史研究暴露，结论属于内部回顾性研究。

所有配对区间均为当前单种子预测上的2000次条件bootstrap，无法估计训练种子变异。protocol区间对应协议macro estimand，cell区间对应cell等权estimand。HUST77协议各一颗电池，无法识别协议内方差。多重探索对比未进行家族误差率调整，不能据单一阳性比较升级论文结论。
'''


def reports():
    s = pd.read_csv(OUT/'summary_metrics.csv')
    p = pd.read_csv(OUT/'paired_comparisons.csv')
    fact = pd.read_csv(OUT/'factorial_effects.csv')
    o = pd.read_csv(OUT/'oracle_summary_metrics.csv')
    ini = pd.read_csv(OUT/'oracle_initializations.csv')
    units = pd.read_csv(OUT/'CHECKPOINT_INDEX.csv')
    counts = json.loads((OUT/'ANALYSIS_COMPLETE.json').read_text(encoding='utf-8'))
    cols = ['domain','arm','supported_cells','observations','cell_mean_mae','cell_mean_rmse','protocol_macro_mae','bias']
    full = s[s.window=='all']
    contrastcols = ['domain','window','reference','candidate','n_cells','mean_improvement','median_improvement','better','worse','cell_ci_low','cell_ci_high']
    d1arms = ['D1-0','D1-G','D1-L','D1-GL','D1-0-capmatch','HISTORICAL-M1']
    text = '# D1 显式早期物理特征消融\n'+common()+'\n## 全轨迹\n\n'+md(full[full.arm.isin(d1arms)],cols)
    text += '\n\n## 受控配对\n\n'+md(p[(p.window=='all')&p.candidate.isin(['D1-G','D1-L','D1-GL','D1-0-capmatch'])],contrastcols)
    text += '\n\n## 容量与时距\n\n'+md(units.groupby(['domain','arm'])[['parameters','extra_params']].mean().reset_index())
    text += '\n\n'+md(s[s.arm.isin(['D1-0','D1-GL'])&s.window.isin(['101_200','201_500','501_1000','1001_plus'])],['domain','arm','window','supported_cells','cell_mean_mae','bias'])
    text += '\n\n优势为同一原始支路、同一UID与相近参数容量对照。风险为额外分支改变优化路径，容量近似匹配仍无法消除函数形式差异。完整reference VIQ可由现有四通道近似重建，具体舍入误差与旧架构的表征差异见UNMATCHED_INFORMATION.md。额外分支使用固定派生初始化种子420014，原主干与decoder初值沿用seed42。逐cell反例与长寿命贡献保留在worst_cases.csv和long_horizon_concentration.csv。输入增量的方向只适用于当前折与单种子，置信度为中。\n'
    (REPORT/'D1_PHYSICS_INPUT_ABLATION.md').write_text(text,encoding='utf-8')
    text = '# D2 速率辅助目标正交对照\n'+common()
    text += '\nF0L0为D1-0，F1L0为D1-GL，F0L1为D2-F0L1，F1L1为D2-F1L1。rate loss为supported-cell平均的supported-window MAE×1000×0.1。MATR两个窗口支持151与42颗，HUST两个窗口均支持77颗，mask保持全部主队列。历史trainer源码使用相同逐cell再逐window规则，目标并未改成全局window均值。\n'
    text += '\n## 主表\n\n'+md(full[full.arm.isin(['D1-0','D1-GL','D2-F0L1','D2-F1L1','D2-GL-rate-spline','D2-GL-rate-physics'])],cols)
    text += '\n\n## 正交效应\n\n'+md(fact[fact.window=='all'])
    text += '\n\ninteraction_gain=(F1L0−F1L1)−(F0L0−F0L1)，正值表示特征存在时rate辅助的收益更强。feature_main与rate_main取另一个因素两个水平上的平均改善。feature因素同时增加支路参数，需结合D1-capmatch解释。\n'
    text += '\n## 501–1000圈偏差\n\n'+md(s[(s.window=='501_1000')&s.arm.isin(['D1-0','D1-GL','D2-F0L1','D2-F1L1','N2-physics_basis','D2-GL-rate-physics'])],['domain','arm','supported_cells','cell_mean_mae','bias'])
    history = []
    for r in units.itertuples():
        path = OUT/r.domain/r.fold/r.arm
        h = pd.DataFrame(json.loads((path/'HISTORY.json').read_text()))
        probe = json.loads((path/'GRADIENT_PROBES.json').read_text())
        history.append(dict(prediction_scope=SCOPE,domain=r.domain,fold=r.fold,arm=r.arm,
            mean_main_loss=h.train_mae.mean(),mean_weighted_rate=h.weighted_rate_term.mean(),
            mean_grad_norm=h.grad_norm_mean.mean(),mean_gradient_cosine=np.mean([v['gradient_cosine'] for v in probe]),
            conflicting_epochs=sum(v['gradient_cosine']<0 for v in probe),best_epoch=r.best_epoch))
    hf = pd.DataFrame(history)
    hf.to_csv(OUT/'loss_gradient_summary.csv',index=False)
    text += '\n\n## 损失与梯度\n\n'+md(hf.groupby(['domain','arm'])[['mean_main_loss','mean_weighted_rate','mean_gradient_cosine','conflicting_epochs']].mean().reset_index())
    text += '\n\n优势为固定lambda与正交复用，未按测试分数选择目标权重。风险为辅助标签支持集中在有长随访的电池，MATR第二段仅42颗，训练信号分布存在选择性。各epoch实际支持数及固定训练probe的梯度夹角完整保留，置信度为中。\n'
    (REPORT/'D2_RATE_OBJECTIVE_FACTORIAL.md').write_text(text,encoding='utf-8')
    oo = o[(o.role=='test')&(o.window=='all')]
    text = '# D3 未来标签函数表达诊断\n'+common()
    text += '\n所有本节函数拟合的prediction_scope均为ORACLE_DIAGNOSTIC。每颗cell允许读取其完整真实未来SOH，八初值float64 L-BFGS-B最小化绝对误差。每个family的全部初值、参数、状态及预测保存在独立诊断目录，未提供给合法训练器。\n'
    text += '\nphysics_shared的time_c与smooth由对应折train未来曲线联合学习，个体4参数在该cell全部future上拟合。physics_free另允许每cell自由time_c与smooth，共6参数。后者自由度更高。spline10沿用十段正rate指数累计，positive_rate8是八段正SOH损失rate积分，polynomial4是锚点固定四次非单调多项式，linear1为非负线性损失。\n'
    text += '\n## OOF身份下的诊断汇总\n\n'+md(oo,cols)
    text += '\n\n## 多初值优化状态\n\n'+md(ini.groupby(['domain','family']).agg(starts=('mae','size'),converged=('converged','sum'),median_iterations=('iterations','median'),median_gradient_inf=('gradient_inf','median')).reset_index(),scope=ORACLE)
    text += '\n\n数值搜索不保证全局最小。L-BFGS-B对非光滑MAE可能提前停止或达到预算，所有失败状态仍保留。softplus极小值在浮点中可能接近正边界，Jacobian病态与共享尺度漂移需结合D4解释。训练、验证、测试身份分别列在oracle_summary_metrics.csv，完整四时距、EOL前后与SOH回升在oracle_cell_metrics.csv。fit_gap包含函数自由度、共享尺度学习、参数映射与数值优化差异，无法解释为纯信息下界。优势为同锚点与同观测支持下的有限自由度对照。风险为未来标签访问和局部优化，函数能拟合的结论置信度为中，电化学机制识别证据仍不足。\n'
    (REPORT/'D3_ORACLE_EXPRESSIVITY.md').write_text(text,encoding='utf-8')
    stability,svd,proxy,profile,gradient = [],[],[],[],[]
    for domain in settings()['domains']:
        root = OUT/'D4'/domain
        stability.append(pd.read_csv(root/'parameter_stability.csv'))
        svd.append(pd.read_csv(root/'jacobian_svd.csv'))
        proxy.append(pd.read_csv(root/'rate_proxy_summary.csv'))
        profile.append(pd.read_csv(root/'profiles.csv'))
        for fold in settings()['folds']:
            records = json.loads((OUT/domain/fold/'D2-GL-rate-physics/GRADIENT_PROBES.json').read_text())
            for epoch in (1,12,150):
                r = records[epoch-1]
                gradient.append(dict(prediction_scope=SCOPE,domain=domain,fold=fold,epoch=epoch,time_c=r['time_c'],smooth=r['smooth'],
                    raw_tau_grad_norm=float(np.linalg.norm(np.array(r['values']['raw_gradient'])[:,3])),
                    raw_a_grad_norm=float(np.linalg.norm(np.array(r['values']['raw_gradient'])[:,0])),
                    raw_b_grad_norm=float(np.linalg.norm(np.array(r['values']['raw_gradient'])[:,1])),
                    raw_c_grad_norm=float(np.linalg.norm(np.array(r['values']['raw_gradient'])[:,2])),
                    tau_within_followup_fraction=float(np.mean(r['tau_within_followup'])),
                    median_tau_cycle=float(np.median(r['absolute_tau'])),acceleration_fraction_501_1000=r['acceleration_fraction']['501_1000']))
    st,sv,pr,prof,gr = map(pd.concat,(stability,svd,proxy,profile,[pd.DataFrame(gradient)]))
    gr.to_csv(OUT/'physics_gradient_summary.csv',index=False)
    text = '# D4 参数稳定性、速率可预测性与实际梯度\n'+common()
    text += '\n## 多初值稳定性\n\n近等价集合同时满足MAE≤best×1.05且差≤0.002，阈值敏感性1%与10%预先登记。\n\n'
    text += md(st.groupby(['domain','family','parameter']).agg(cells=('range','size'),median_range=('range','median'),max_range=('range','max'),median_equivalent_starts=('equivalent_starts','median')).reset_index(),scope=ORACLE)
    text += '\n\n## 缩放Jacobian\n\n'+md(sv.groupby(['domain','family','followup_group']).agg(cells=('condition','size'),median_condition=('condition','median'),median_near_zero=('near_zero_directions','median')).reset_index(),scope=ORACLE)
    text += '\n\nJacobian用实际可观测query，以物理参数预定尺度[0.05,0.05,0.06,1,0.5,0.25]计算SVD。它只刻画当前位置与支持区间的局部病态。shared模式的两个共享参数固定，free模式有6个个体方向。profile子集固定为每域fold0排序最前3个train与3个test UID，未按拟合效果筛选。有效rate网格单位为SOH fraction/1000 cycles，SLSQP未收敛或约束残差较大的点保留在profiles.csv。\n'
    text += '\n## 合法K100预测真实端点rate\n\n'+md(pr)
    text += '\n\n轻头只拟合本折有支持的train电池，StandardScaler也只fit该train支持集，alpha仅由本折validation选择，未合并trainval重拟合。每窗口按原UID的OOF test评分。冻结z来自相同折已冻结D1/D2网络，属于已有监督表征的条件诊断。共160个选定rate proxy模型，6候选alpha对应960次Ridge拟合，独立记录于RATE_PROXY_MODELS.json。MATR第二段n=42，R²对少量长寿命电池敏感。\n'
    text += '\n## 实际physics_basis训练probe\n\n'+md(gr)
    text += '\n\n每epoch固定本折前8个train UID进行无更新probe，记录raw与a/b/c/tau、共享尺度、分布、梯度与各时距加速贡献。probe前后恢复随机态与train/eval模式。完整150epoch可追溯。此次使用原始初始化，未运行可选tau初值控制。因此梯度弱只支持当前优化路径解释，无法确立初始化的因果作用。HUST全部右删失，无真实EOL预测评价。优势为拟合稳定性、真实rate代理和实际网络梯度分别可核查。风险为有限多初值、局部SVD与短随访，参数唯一性与真实电化学机制的置信度为低。\n'
    (REPORT/'D4_IDENTIFIABILITY_AND_GRADIENTS.md').write_text(text,encoding='utf-8')
    text = '# D1–D4 失效归因裁决\n'+common()+'\n## 合法完整轨迹结果\n\n'+md(full,cols)
    text += '\n\n## 四项假设\n\n'
    for domain in settings()['domains']:
        def pair(reference,candidate):
            return p[(p.domain==domain)&(p.window=='all')&(p.reference==reference)&(p.candidate==candidate)].iloc[0]
        h1,h2 = pair('D1-0','D1-GL'),pair('D1-GL','D2-F1L1')
        cap = pair('D1-0-capmatch','D1-GL')
        of = oo[(oo.domain==domain)&(oo.arm=='physics_free')].iloc[0]
        os = oo[(oo.domain==domain)&(oo.arm=='physics_shared')].iloc[0]
        cond = sv[(sv.domain==domain)&(sv.family=='physics_free')].condition.median()
        text += f'### {domain}\n\n'
        text += f'H1的GL配对MAE改善为{h1.mean_improvement:.6f}，cell条件区间[{h1.cell_ci_low:.6f}, {h1.cell_ci_high:.6f}]，相对容量对照改善为{cap.mean_improvement:.6f}。'
        text += ('当前配置支持显式特征增量。' if h1.cell_ci_low>0 and cap.mean_improvement>0 else '当前配置尚不足以确认显式特征的稳定独立收益。')
        text += '\n\n'
        text += f'H2的GL条件下rate辅助改善为{h2.mean_improvement:.6f}，cell条件区间[{h2.cell_ci_low:.6f}, {h2.cell_ci_high:.6f}]。'
        text += ('当前配置支持该辅助目标收益。' if h2.cell_ci_low>0 else '该目标的稳定收益证据不足，详见正交交互与窗口反例。')
        text += '\n\n'
        text += f'H3的physics_free oracle MAE为{of.cell_mean_mae:.6f}，train-only共享尺度oracle为{os.cell_mean_mae:.6f}。'
        text += ('预定0.01描述性阈值下，函数具备较低平均拟合误差，合法映射与优化仍需解释。' if of.cell_mean_mae<.01 else '该有限搜索仍有表达、测量形状或优化误差，不能单独归因于早期编码器。')
        text += '\n\n'
        text += f'H4的free模式Jacobian条件数中位数为{cond:.3g}。近等价多解、profile范围与真实rate代理预测提供有限参数病态证据，无法证明仅凭K100的全局不可辨识。'
        text += '\n\n'
    text += '## 验收计数\n\n'
    text += f"合法K100新增神经网络训练80组，正式SGD/AdamW更新{counts['formal_updates']:,}次，合法OOF评价246颗物理电池。D3个体多初值拟合问题{counts['oracle_individual_multistart_problems']}个，每个8初值，共{counts['oracle_individual_optimizer_starts']}次个体优化。另有10个train-only共享尺度联合问题，共80次联合优化。D4 profile另有312次有限约束优化。rate proxy另有160个选定轻头与960次候选Ridge拟合。额外真实恢复烟测28次更新，原N012测试夹具的30次更新单列，均未纳入126000正式更新。\n"
    text += '\n## 同输入、同辅助目标的解码头配对\n\n'
    text += md(p[(p.reference=='D2-GL-rate-spline')&(p.candidate=='D2-GL-rate-physics')&(p.window=='all')],contrastcols)
    text += '\n\n该对比属于原合同要求，初版汇总遗漏其单独行，现从同一批冻结预测补齐。全部既有对比的估计量保持原定义。两头的结果用于判断具体三分量参数化的独立贡献，具有相同输入与辅助loss。\n'
    text += '\n## 速率目标的时距权衡\n\n'
    text += md(s[s.arm.isin(['D1-GL','D2-F1L1'])&s.window.isin(['all','501_1000','1001_plus'])],
               ['domain','arm','window','supported_cells','cell_mean_mae','bias'])
    text += '\n\nMATR的rate辅助使完整轨迹主指标上升，同时降低501–1000圈MAE与正偏差。当前结果支持目标之间的权衡，无法将固定0.1权重解释为历史M1优势的充分原因。HUST主指标同样没有出现该辅助项的稳定增益。\n'
    text += '\n## 长随访贡献与反例\n\n'
    conc=pd.read_csv(OUT/'long_horizon_concentration.csv')
    text += md(conc[(conc.reference=='D1-0')&(conc.candidate=='D1-GL')&conc.window.isin(['all','1001_plus'])])
    text += '\n\n最大受损cell与top3正收益占比均保留，不能只依据总体均值解释所有电池。\n'
    text += '\n## 速率代理、梯度与数值限制\n\n'
    text += 'MATR的200–500圈rate加入G统计后R²约0.658，500–1000圈只有42颗支持，当前各固定代理的R²约0.062–0.190。HUST的500–1000圈rate在原始低维统计下R²约0.561。可预测性随域和窗口变化，现有代理结果无法证明K100的全局不可预测性。\n\n'
    text += 'MATR训练probe中tau梯度范数中位数从第1轮约0.000218升至第12轮约0.007678，后续存在训练/验证分化。当前证据支持阶段性的弱激活与优化问题，初始化的因果贡献仍需独立控制。局部Jacobian病态与少量近等价初值应结合短随访支持解释。\n\n'
    for domain in settings()['domains']:
        pp=pd.read_csv(OUT/'D4'/domain/'profiles.csv')
        text += f"{domain}共{len(pp)}次profile优化，{int(pp.converged.sum())}次报告收敛，{int((~pp.converged).sum())}次未收敛。未收敛点、约束残差及全部原始参数保留。\n\n"
    text += '保留局部分段物理表征与形状约束头的受控比较。当前三分量physics_basis的独立优势证据不足，固定query_mlp的2×2对照中rate=0.1未改善完整轨迹主指标。函数oracle的低误差支持继续检验参数映射和学习过程，不能直接认证真实电化学机制。\n'
    text += '\n## 结论范围与后续\n\n各域和时距的阴性结果与反例全部保留。本轮保持原问题、数据与80组预注册矩阵，不按测试结果扩模型。三分量形状先验仍可作为受控函数对照，具体LLI/LAM/SEI解释缺少电极分辨测量与独立分量真值。单种子与历史暴露限制外推，尚未满足论文候选升级要求。\n\n优势为同队列、正交目标、显式信息预算、未来oracle隔离与实际权重重放。风险为有限oracle搜索、轻头样本量、近似参数匹配与多重探索。运行完整性置信度为高，跨域机制与训练种子稳定性置信度为低。\n\n## 拓展思考\n\n函数拟合能力与前缀映射需要各自的证据。保留实际可预测rate、参数病态与梯度覆盖之间的差异，有助于约束下一轮最小修改。\n\n## 你可能想知道\n\n全部逐cell预测、负例、断点状态、参数profile与源码哈希均有索引。HUST的右删失终点未被当作EOL事件。\n'
    (REPORT/'FINAL_CAUSAL_ASSESSMENT.md').write_text(text,encoding='utf-8')


def figures():
    folder = REPORT/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    contract = '''# 图形证据合同

采用既有Python后端，定量证据面板，双栏180mm附近，PNG与含可编辑文字的PDF同步导出。所有数据由已冻结表生成。

1. 配对与时距图检查同cell输入/目标增量，明确域与N，避免按单个窗口选赢家。
2. Oracle图检查函数拟合与合法预测差距，明确未来标签诊断的访问权限差异。
3. 参数图检查近等价tau范围与profile曲线，保留无支持和不收敛状态，无法声称全局可辨识。
4. 训练曲线与偏差图检查验证选择、优化动态和中后期方向误差，避免把早停点误读为机制。

文字不覆盖曲线，所有误差单位SOH fraction。PDF由同一Matplotlib图导出，随后重渲染检查。
'''
    (REPORT/'FIGURE_CONTRACT.md').write_text(contract,encoding='utf-8')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    palette = ['#4C78A8','#EE9B62','#72AFA1','#B08DC3']
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=300)
        fig.savefig(folder/(name+'.pdf'))
        plt.close(fig)
    p = pd.read_csv(OUT/'paired_cells.csv')
    s = pd.read_csv(OUT/'summary_metrics.csv')
    o = pd.read_csv(OUT/'oracle_cell_metrics.csv')
    fig,axes = plt.subplots(2,2,figsize=(8,6),layout='constrained')
    for col,domain in enumerate(settings()['domains']):
        a = p[(p.domain==domain)&(p.window=='all')&(p.reference=='D1-0')&(p.candidate=='D2-F1L1')]
        axes[0,col].scatter(np.arange(len(a)),a.improvement,s=8,color=palette[0],alpha=.75)
        axes[0,col].axhline(0,color='.35',lw=.6)
        axes[0,col].set(title=f'{domain}, n={len(a)} cells\nD1-0 minus GL + rate',xlabel='OOF cell index',ylabel='MAE improvement, SOH fraction')
        for color,arm in zip(palette,['D1-0','D1-GL','D2-F0L1','D2-F1L1']):
            g=s[(s.domain==domain)&(s.arm==arm)].set_index('window').loc[[v[0] for v in __import__('rootcause_d14.analyze',fromlist=['WINDOWS']).WINDOWS][1:]]
            axes[1,col].plot(range(4),g.cell_mean_mae,'o-',ms=3,lw=1,color=color,label=arm)
        labels = [f'{name}\n(n={int(n)})' for name,n in zip(['101–200','201–500','501–1000','1001+'],g.supported_cells)]
        axes[1,col].set(xticks=range(4),xticklabels=labels,ylabel='MAE, SOH fraction',xlabel='Physical cycle')
    axes[1,1].legend(loc='upper left',fontsize=7)
    save(fig,'01_paired_and_horizons')
    fig,axes = plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,domain in zip(axes,settings()['domains']):
        legal = pd.read_csv(OUT/'cell_metrics.csv')
        fam = ['physics_shared','physics_free','spline10','positive_rate8','polynomial4','linear1']
        values = [legal[(legal.domain==domain)&(legal.arm=='D1-0')&(legal.window=='all')].mae.to_numpy()]
        values += [o[(o.domain==domain)&(o.role=='test')&(o.arm==f)&(o.window=='all')].mae.to_numpy() for f in fam]
        ax.boxplot(values,showfliers=True,widths=.6,flierprops=dict(marker='.',markersize=2,alpha=.45))
        ax.set(xticks=np.arange(1,8),xticklabels=['Legal\nK100','Shared\nphysics','Free\nphysics','Spline\n10','Rate\n8','Poly\n4','Linear\n1'],
               ylabel='Cell MAE, SOH fraction',title=f'{domain}, n={len(values[0])}\nGroups 2–7 use future-label oracle')
        ax.tick_params(axis='x',labelsize=7)
    save(fig,'02_oracle_expressivity')
    fig,axes = plt.subplots(2,2,figsize=(8,6.8),layout='constrained')
    for col,domain in enumerate(settings()['domains']):
        st = pd.read_csv(OUT/'D4'/domain/'parameter_stability.csv')
        g=st[(st.family=='physics_free')&(st.parameter=='tau')].sort_values('last_cycle')
        axes[0,col].vlines(np.arange(len(g)),g.minimum,g.maximum,color=palette[0],alpha=.65,lw=.8)
        axes[0,col].scatter(np.arange(len(g)),(g.minimum+g.maximum)/2,color=palette[0],s=2)
        axes[0,col].set(title=f'{domain}, n={len(g)} OOF cells',ylabel='Equivalent tau, 1000 cycles',xlabel='Cells sorted by follow-up')
        profile = pd.read_csv(OUT/'D4'/domain/'profiles.csv')
        for uid,g in profile[profile.parameter=='tau'].groupby('cell_uid'):
            axes[1,col].plot(g.fixed_value,g.mae,label=uid,lw=.8)
        axes[1,col].set(xlabel='Fixed tau, 1000 cycles',ylabel='Profile MAE, SOH fraction')
        axes[1,col].legend(fontsize=6,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.24))
    save(fig,'03_parameter_stability')
    fig,axes = plt.subplots(2,2,figsize=(8,6.8),layout='constrained')
    for col,domain in enumerate(settings()['domains']):
        for j,fold in enumerate(settings()['folds']):
            h = pd.DataFrame(json.loads((OUT/domain/fold/'D2-GL-rate-physics/HISTORY.json').read_text()))
            axes[0,col].plot(h.epoch,h.train_mae,color=plt.cm.viridis(j/5),lw=.7,alpha=.65)
            axes[0,col].plot(h.epoch,h.val_mae,color=plt.cm.viridis(j/5),lw=1,ls='--',label=fold.replace('cell_fold_','fold '))
        axes[0,col].set(title=f'{domain}, GL + rate + physics\nSolid train / dashed validation',xlabel='Epoch',ylabel='MAE, SOH fraction')
        axes[0,col].legend(fontsize=6,loc='upper center',ncol=5,bbox_to_anchor=(.5,-.24))
        arms=['D1-0','D1-GL','D2-F1L1','N2-physics_basis','D2-GL-rate-physics']
        g=s[(s.domain==domain)&(s.window=='501_1000')].set_index('arm').loc[arms]
        axes[1,col].bar(range(5),g.bias,color=palette[0])
        axes[1,col].axhline(0,color='.35',lw=.6)
        axes[1,col].set(xticks=range(5),xticklabels=['Raw','GL','GL +\nrate','N012\nphysics','GL rate\nphysics'],ylabel='501–1000 bias, SOH fraction')
    save(fig,'04_training_and_bias')
    write_json(folder/'FIGURES_COMPLETE.json',dict(status='GENERATED_PENDING_VISUAL_QA',backend='Python matplotlib',
        files=file_records(folder,[p.name for p in sorted(folder.iterdir()) if p.suffix in ('.png','.pdf')])))


def main():
    reports()
    figures()


if __name__=='__main__':
    main()
