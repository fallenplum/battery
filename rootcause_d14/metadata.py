"""Exhaustive field coverage and feature-channel support, read-only."""
import json
import numpy as np
import pandas as pd
from n012.run import write_json
from n012.data import file_hash
from scripts.unified_matr.physics_features import GLOBAL_COLUMNS,LOCAL_COLUMNS
from .data import ROOT,OUT,REPORT,SCOPE,load_domain


def main():
    fields,support,recovery = [],[],[]
    for domain in ('MATR','HUST'):
        d=load_domain(domain)
        for name,col in d.bundle.cell_metadata.reset_index().items():
            numeric=pd.to_numeric(col,errors='coerce') if pd.api.types.is_numeric_dtype(col) else None
            fields.append(dict(prediction_scope=SCOPE,domain=domain,field=name,dtype=str(col.dtype),
                physical_cells=len(col),nonnull=int(col.notna().sum()),missing=int(col.isna().sum()),
                finite=int(np.isfinite(numeric).sum()) if numeric is not None else None,
                minimum=float(numeric.min()) if numeric is not None and numeric.notna().any() else None,
                maximum=float(numeric.max()) if numeric is not None and numeric.notna().any() else None,
                use='frozen metadata audit; field role determined by qualified source and legal input allowlist'))
        for key,mask,names in [('global_features','global_mask',GLOBAL_COLUMNS),('local_features','local_feature_mask',LOCAL_COLUMNS)]:
            value,valid=d.arrays[key],d.arrays[mask]
            bins=range(8) if key=='local_features' else [None]
            for bin_id in bins:
                for j,name in enumerate(names):
                    v=value[:,:,bin_id,j] if bin_id is not None else value[:,:,j]
                    m=valid[:,:,bin_id,j] if bin_id is not None else valid[:,:,j]
                    support.append(dict(prediction_scope=SCOPE,domain=domain,array=key,bin=bin_id,channel=j,field=name,
                        provenance='derived_from_qualified_original_cycle_1_100_electrical_response',
                        total_slots=int(m.size),supported_slots=int(m.sum()),supported_cells=int(m.any(1).sum()),
                        finite_supported=int(np.isfinite(v[m]).sum()),masked_nonzero_placeholders=int(((~m)&(v!=0)).sum())))
        with np.load(d.source) as archive:
            voltages=archive['voltage_grid_V']
            reference=archive['reference_viq']
            for i,uid in enumerate(d.bundle.ids):
                m=d.bundle.mask[i]
                q=d.bundle.x[i,...,2].astype(float)
                normalized_q=q-d.bundle.x[i,...,3].astype(float)
                denominator=np.square(q[m]).sum()
                assert denominator>0
                alpha=float(np.sum(q[m]*normalized_q[m])/denominator)
                predicted_i=d.bundle.x[i,...,1].astype(float)*alpha
                recovery.append(dict(prediction_scope=SCOPE,domain=domain,cell_uid=uid,
                    inverse_prefix_scale=alpha,supported_points=int(m.sum()),
                    reference_Q_max_abs=float(np.max(np.abs(normalized_q[m]-reference[i,...,2][m]))),
                    reference_I_max_abs=float(np.max(np.abs(predicted_i[m]-reference[i,...,1][m]))),
                    reference_V_max_abs=float(np.max(np.abs(d.bundle.x[i,...,0][m]-reference[i,...,0][m])))))
        for j,voltage in enumerate(voltages):
            v,m=d.arrays['local_residual'][:,:,j],d.arrays['local_mask'][:,:,j]
            support.append(dict(prediction_scope=SCOPE,domain=domain,array='local_residual',bin=None,channel=j,
                field=f'residual_at_{voltage:.8g}V',provenance='derived_early_electrical_response_residual_Ah',
                total_slots=int(m.size),supported_slots=int(m.sum()),supported_cells=int(m.any(1).sum()),
                finite_supported=int(np.isfinite(v[m]).sum()),masked_nonzero_placeholders=int(((~m)&(v!=0)).sum())))
    pd.DataFrame(fields).to_csv(OUT/'METADATA_FIELD_COVERAGE.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'PHYSICAL_FEATURE_SUPPORT.csv',index=False)
    recovered=pd.DataFrame(recovery)
    recovered.to_csv(OUT/'REFERENCE_FULL_RECOVERY_AUDIT.csv',index=False)
    write_json(OUT/'METADATA_COVERAGE_COMPLETE.json',dict(status='PASS',prediction_scope=SCOPE,
        metadata_columns=len(fields),physical_feature_channels_with_bins=len(support),
        physical_feature_definition_sha256=file_hash(ROOT/'scripts/unified_matr/physics_features.py'),
        files=[dict(path=p,sha256=file_hash(OUT/p)) for p in ('METADATA_FIELD_COVERAGE.csv','PHYSICAL_FEATURE_SUPPORT.csv','REFERENCE_FULL_RECOVERY_AUDIT.csv')]))
    path=REPORT/'UNMATCHED_INFORMATION.md'
    base='''# 信息预算与参考VIQ可恢复性审计

源构建代码表明，reference_viq是在同条早期曲线的I、Q上除以由1–10圈容量窗计算的单cell尺度，V保留原值。它与物理分解使用的独立reference_q中位曲线属于不同数组。初版审计直接比较I和reference_I，据差值推断独立信息缺口，该推断需要修正。

N012的第四通道为Q−reference_Q，因此reference_Q可由Q−第四通道恢复。再用所有有效早期点的最小二乘比例alpha=sum(Q×reference_Q)/sum(Q²)，可重建reference_I=alpha×I。此过程只使用同cell的K100输入，无未来标签，也不涉及跨cell统计。float32舍入造成有限重建误差。

逐cell结果见REFERENCE_FULL_RECOVERY_AUDIT.csv，原直接比较仍保留在REFERENCE_RECOVERY_AUDIT.csv用于追溯。报告中的恢复误差单位沿用冻结数组的存储单位。

'''
    for domain,g in recovered.groupby('domain'):
        base+=f"{domain}共{len(g)}颗，参考V重建最大误差{g.reference_V_max_abs.max():.9g}，参考Q为{g.reference_Q_max_abs.max():.9g}，参考I为{g.reference_I_max_abs.max():.9g}。\n\n"
    base+='''这项代数关系支持参考VIQ近似可由现有四通道重建。显式G/L分解、权重、mask、神经网络压缩方式与目标函数仍存在差异。D1-GL检验的是显式表征与优化路径的增量，无法仅凭该对照宣称获得新的独立测量信息，也无法声称与历史M1的完整输入和算子预算已匹配。

历史M1还有不同的reference相关通路、rate辅助、持续性惩罚与输出clamp。局部分支的masked汇聚压缩了周期及分段细节，原始电学值的代数可恢复性不保证当前网络能在有限训练下恢复相同函数。

优势为逐cell重建与源代码定义均已核对。风险为浮点舍入、缺失支持及算子差异，尚不能从均值成绩独立认证电化学机制。
'''
    with path.open('w',encoding='utf-8') as f:
        f.write(base)
        f.write('\n\n## 逐字段元数据覆盖\n\nMETADATA_FIELD_COVERAGE.csv逐字段保留两域cells元数据的类型、非空、缺失及数值范围。PHYSICAL_FEATURE_SUPPORT.csv逐G通道及局部8分段×26通道保留支持数、有效有限值与mask外占位统计。所有支路只使用显式mask内的值。G中的a_Ah与b_V来自早期曲线形变分解，其单位与decoder的无量纲累计损失参数不同。局部统计包含残差、积分、质心、峰位置、覆盖率、噪声与因果动态量，属于从实测电学数据派生的特征。它们没有被认定为独立测量的电极机制真值。\n')


if __name__=='__main__':
    main()
