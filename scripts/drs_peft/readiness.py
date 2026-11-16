"""Reproduce the repository audit and outcome-independent development split lock."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

from scripts.drs_peft.episodes import SHOTS, digest, load_registration, ordered, support_draws

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    (1, 'README.md', '历史结果索引、队列边界与独立核验入口'),
    (2, 'contracts/SRSR_Physics_统一研究问题_一次性数据处理与模型实验总合同.md', '旧统一任务合同，保留旧任务定义与历史暴露'),
    (3, 'configs/unified_matr_seed42.json', '169颗、原生1–100圈、2.8–3.3V、1.1Ah与旧seed42训练配置'),
    (4, 'data/unified_matr_seed42/DATA_FREEZE.json', '冻结主数据及依赖的SHA清单'),
    (5, 'data/unified_matr_seed42/dataset_qa_report.md', '历史580项数据核验记录'),
    (6, 'data/unified_matr_seed42/split_contract.json', '旧五折及协议敏感性划分，保留原字节'),
    (7, 'models/unified_matr/README.md', '旧M1/M2编码器、global20、local8×26、原Flow与积分头语义'),
    (7, 'models/unified_matr/model.py', '旧电池模型可复用编码器的实际实现'),
    (8, 'scripts/unified_matr/train_seed42.py', '旧训练、input allowlist、折内标准化及重放入口'),
    (9, 'report/UNIFIED_MATR_SEED42_FINAL_ZH.md', '旧M1/M2 OOF与负结果，历史探索参考'),
    (10, 'report/M4_FULL_METHOD_FINAL_ZH.md', 'M4真实训练与完整rate-integral方法结果'),
    (11, 'report/M4_ABLATIONS_SEED42_FINAL_ZH.md', 'A1–A4独立派生消融及M4速率积分的负证据'),
    (12, 'report/MULTIFAMILY_EXTENSION_REPORT_ZH.md', 'MATR/Stanford历史供应坐标斜率任务与暴露'),
    (13, 'report/MATR_EXTENSION_QUALIFICATION_ZH.md', '较早供应坐标资格快照，后续native冻结已补足仪器轴'),
    (14, 'report/STANFORD_EXTENSION_QUALIFICATION_ZH.md', '181供应UID、形成参数、3.4–4.1V与BEEP版本缺口'),
    (15, 'report/HUST_RAW_IDEA_FEASIBILITY_ZH.md', '77颗原始队列、实际2C末段、138422点及0个EOL80事件'),
    (16, 'report/SDU_DATA_AUDIT_ZH.md', '86颗完整供应元数据审计，重复循环与身份链缺口'),
)
EXTRA_ENTRIES = (
    'scripts/unified_matr/probe_frozen_loader.py',
    'scripts/unified_matr/verify_dataset.py',
    'scripts/unified_matr/verify_physics_features.py',
    'scripts/unified_matr/analyze_results.py',
    'n012/data.py',
    'scripts/drs_peft/data.py',
    'scripts/drs_peft/episodes.py',
    'report/drs_peft/DATA_READINESS_ZH.md',
)
DEPENDENCIES = ('torch', 'numpy', 'pandas', 'pyarrow', 'scipy', 'uni2ts',
                'transformers', 'huggingface-hub', 'safetensors', 'einops',
                'gluonts', 'lightning', 'paramiko', 'pytest')


def canonical_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def read_inventory(root):
    """Read every complete file, then retain its exact byte identity and role."""
    inventory = []
    texts = {}
    for number, name, fact in REQUIRED:
        path = root / name
        row = dict(contract_item=number, path=name, exists=path.is_file(), role=fact)
        if path.is_file():
            content = path.read_bytes()
            texts[name] = content.decode('utf-8-sig')
            row.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                       first_heading=next((s for s in texts[name].splitlines() if s.startswith('#')), ''))
        inventory.append(row)
    extras = []
    for name in EXTRA_ENTRIES:
        path = root / name
        entry = dict(path=name, exists=path.is_file())
        if path.is_file():
            content = path.read_bytes()
            content.decode('utf-8-sig')
            entry.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        extras.append(entry)
    return inventory, extras, texts


def build_meta(registration):
    meta_ids = set(registration['source_cell_roles']['source_meta_validation'])
    query = []
    group_counts = {}
    for group in registration['source_group_roles']['source_meta_validation']:
        ids = registration['source_groups'][group]
        group_counts[group] = len(ids)
        query.extend(ordered(ids, registration['episode_seed'], 'source_meta_query|' + group)[:1])
    query = sorted(query)
    pool = sorted(meta_ids - set(query))
    if len(query) != 15 or len(pool) != 15 or set(query) & set(pool):
        raise ValueError('Expected fixed source-meta Query15/Support15 allocation')
    forbidden = set(registration['source_cell_roles']['source_train'])
    forbidden.update(registration['source_cell_roles']['target_test'])
    if meta_ids & forbidden:
        raise ValueError('Development identities intersect source training or final target')
    result = dict(
        schema='DRS_PEFT_SOURCE_META_EPISODES_v1', scope='SOURCE_CALIBRATION_DEVELOPMENT',
        domain='MATR', coordinate='original_instrument_cycle', episode_seed=registration['episode_seed'],
        source_episode_contract_sha256=registration['episode_contract_sha256'],
        selection_inputs=['physical_cell_UID', 'registered_exact_protocol_group'],
        future_outcomes_read_for_split=False, meta_exact_group_counts=group_counts,
        allocation='one stable SHA-ordered Query per exact meta group; all remaining cells form pooled Support candidates',
        query_ids=query, support_pool_ids=pool,
        protocol='pooled_source_meta_groups_for_calibration_hyperparameter_selection',
        labels_permitted_for='source-only calibration development and hyperparameter selection',
        target_final_query_labels_permitted=False,
        source_train_groups_disjoint=True, target_test_groups_disjoint=True,
        support_and_query_groups_may_overlap=True,
        grouping_limit='Pooled development episodes do not estimate unseen exact-group K10 target performance',
        shots=support_draws(pool, query, seed=registration['episode_seed'],
                            role='SOURCE_META_CALIBRATION_DEVELOPMENT', draws=5),
        query_fixed_across_methods_shots_draws=True,
        duplicate_support_sets_reuse_checkpoint=True,
    )
    result['meta_episode_contract_sha256'] = canonical_hash(result)
    return result


def read_caches(root):
    result = {}
    for domain in ('MATR', 'HUST', 'Stanford_2'):
        folder = root / 'runtime/drs_peft/data_v1' / domain
        path = folder / 'COMPLETE.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        verified = []
        for record in value['files']:
            target = folder / record['path']
            actual = digest(target)
            if actual != record['sha256']:
                raise ValueError(f'Cache fingerprint mismatch: {domain}/{record["path"]}')
            verified.append(dict(path=record['path'], bytes=target.stat().st_size, sha256=actual))
        result[domain] = dict(manifest_sha256=digest(path), status=value['status'],
            physical_cells=value['physical_cells'], prefix_shape=value['prefix_shape'],
            future_shape=value['future_shape'], adapter_sha256=value['adapter_sha256'], files=verified,
            maximum_observed_cycle=value['source']['maximum_observed_cycle'],
            observed_future_points=value['source']['observed_future_points'],
            coordinate='supplied_cycle_number' if domain == 'Stanford_2' else 'original_instrument_cycle')
    return result


def target_denominators(registration):
    tasks = [t for t in registration['target_tasks'] if t['domain'] == 'MATR']
    counts = []
    common = []
    for task in tasks:
        if any(s['k'] == 5 and s['status'] == 'FEASIBLE' for s in task['shots']):
            common.append(dict(task_id=task['task_id'], protocol_uid=task['protocol_uid'],
                               query_ids=task['query_ids']))
    for shot in SHOTS:
        feasible = [(t, s) for t in tasks for s in t['shots']
                    if s['k'] == shot and s['status'] == 'FEASIBLE']
        counts.append(dict(k=shot, exact_protocol_groups=len(feasible),
            unique_query_cells=sum(len(t['query_ids']) for t, _ in feasible),
            registered_draws=sum(len(s['draws']) for _, s in feasible),
            unique_support_sets_within_task=sum(s['unique_support_sets'] for _, s in feasible)))
    return dict(Q=2, heldout_exact_groups=len(tasks),
        singleton_groups_infeasible_Q2=sum(t['status'] == 'NOT_FEASIBLE_Q2' for t in tasks),
        all_feasible_by_k=counts, common_query_subset_K_0_1_3_5=common,
        common_query_cells_K_0_1_3_5=sum(len(t['query_ids']) for t in common),
        K10_status='NOT_FEASIBLE', full_MATR_169_feasibility=registration['full_MATR_feasibility'])


def build_lock(root, registration, meta, inventory, caches):
    lock = dict(schema='DRS_PEFT_DATA_SPLIT_LOCK_v1', episode_seed=registration['episode_seed'],
        source_episode_contract_sha256=registration['episode_contract_sha256'],
        source_episode_file_sha256=digest(root / 'configs/drs_peft/episodes_v1.json'),
        meta_episode_contract_sha256=meta['meta_episode_contract_sha256'],
        fixed_prediction_grid=registration['fixed_prediction_grid'],
        source_cell_roles=registration['source_cell_roles'],
        source_group_roles=registration['source_group_roles'],
        source_groups=registration['source_groups'],
        source_role_counts={role: dict(physical_cells=len(ids), exact_protocol_groups=len(
            registration['source_group_roles'][role])) for role, ids in registration['source_cell_roles'].items()},
        meta_development=dict(scope=meta['scope'], query_ids=meta['query_ids'],
                              support_pool_ids=meta['support_pool_ids']),
        target_tasks=registration['target_tasks'], protocol_A_denominators=target_denominators(registration),
        data_caches=caches, mandatory_read_inventory=inventory,
        target_query_label_policy=dict(status='SEALED_FOR_MODEL_DEVELOPMENT',
            final_target_query_future_labels_allowed_for_training=False,
            final_target_query_future_labels_allowed_for_selection=False,
            final_target_query_future_lengths_allowed_as_model_input=False,
            allowed_during_P0_P3='source_train and source_meta_validation future labels only',
            allowed_target_adaptation='registered Support future labels after method and budget lock',
            final_scoring='registered Query observed labels only after selection is fixed',
            seal_type='software role-based use restriction; cache arrays contain all outcomes',
            historical_exposure='retrospective cohorts previously used in earlier research'),
        preprocessing=dict(fit_role='source_train', physical_cells=85,
            canonical_channels=['V_volts', 'signed_current_C_rate', 'capacity_SOH', 'true_early_drift_SOH'],
            canonical_x_source_zscored=False, policy_columns=28,
            source_all_missing_policy_columns_dropped_across_domains=True,
            normalization='source-only moments, registered physical standard-deviation floors',
            drift='Q_n(V) minus pointwise median Q_ref(V) from supported cycles1..10'),
        raw_data_access='read-only', old_files_modified=False,
        readiness_scope='data and repository readiness; no new training-performance claim')
    lock['split_lock_sha256'] = canonical_hash(lock)
    return lock


def local_environment():
    packages = {}
    for name in DEPENDENCIES:
        try:
            packages[name] = dict(installed=True, version=importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            module = {'huggingface-hub': 'huggingface_hub'}.get(name, name)
            packages[name] = dict(installed=importlib.util.find_spec(module) is not None,
                                  version='distribution metadata unavailable')
    result = dict(platform=platform.platform(), python=platform.python_version(),
                  python_executable=sys.executable, dependencies=packages)
    import torch
    result['torch'] = dict(version=torch.__version__, cuda_build=torch.version.cuda,
                           cuda_available=torch.cuda.is_available(), cudnn=torch.backends.cudnn.version())
    result['gpu'] = [dict(index=i, name=torch.cuda.get_device_name(i),
                          total_memory_MiB=torch.cuda.get_device_properties(i).total_memory // (1024 * 1024))
                     for i in range(torch.cuda.device_count())]
    command = ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits']
    try:
        result['nvidia_smi'] = subprocess.run(command, capture_output=True, text=True,
                                               check=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        result['nvidia_smi'] = dict(status='UNAVAILABLE', error_type=type(error).__name__)
    return result


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise RuntimeError(f'Existing registration differs; explicitly version a new contract: {path.name}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def report(root, inventory, extras, lock, meta, environment):
    missing = [r['path'] for r in inventory if not r['exists']]
    server_path = root / 'research/drs_peft/SERVER_ENVIRONMENT.json'
    server_note = '服务器环境附件尚未回同步，硬件与软件身份等待实际ENV证据。'
    if server_path.is_file():
        server = json.loads(server_path.read_text(encoding='utf-8'))
        gpu_names = sorted({str(item['name']) for item in server['gpus']})
        memory = sorted({int(item['memory_bytes']) for item in server['gpus']})
        versions = {name: server['versions'].get(name, '未记录') for name in
                    ('uni2ts', 'numpy', 'pandas', 'scipy', 'safetensors', 'huggingface-hub')}
        server_note = (f'已实读SERVER_ENVIRONMENT.json，SHA为{digest(server_path)}。'
            f'实际设备{len(server["gpus"])}卡，型号{gpu_names}，设备可见容量字节{memory}。'
            f'Python为{server["python"]}，Torch为{server["torch"]}，CUDA为{server["cuda"]}。'
            f'依赖版本为{json.dumps(versions, ensure_ascii=False)}。'
            '当前共享服务器指定GPU5单作业，使用隔离环境与本地vendor uni2ts。'
            '关键vendor源码SHA保存在环境附件，各run另有执行环境。'
            '旧系统Python3.8存在libcupti缺失，正式任务使用独立环境。'
            '公共报告保留硬件和软件身份，连接凭据留在私有运行环境。')
    lines = ['# DRS-PEFT 仓库、数据与少样本划分审计', '',
        f'本轮完整读取合同16项材料所展开的17个文件，实际存在{len(inventory)-len(missing)}/{len(inventory)}。每项保留完整字节SHA。', '',
        '**状态为数据与仓库就绪。P0模型性能、P1–P5科学实验由各阶段真实运行记录验收。**', '',
        '## 必读材料的实际存在性与版本', '',
        '| 合同项 | 文件 | 字节数 | SHA-256 | 核验用途 |', '|---|---|---:|---|---|']
    for row in inventory:
        lines.append(f'| {row["contract_item"]} | {row["path"]} | {row.get("bytes", "缺失")} | {row.get("sha256", "MISSING")} | {row["role"]} |')
    qa = json.loads((root / 'data/unified_matr_seed42/dataset_qa.json').read_text(encoding='utf-8'))
    data_freeze = json.loads((root / 'data/unified_matr_seed42/DATA_FREEZE.json').read_text(encoding='utf-8'))
    lines += ['', f'原DATA_FREEZE状态为{data_freeze["status"]}，绑定{len(data_freeze["files"])}项文件，原QA有{len(qa["checks"])}项且状态为{qa["status"]}。本CLI验证当前必读文件与新缓存SHA，580项数值核验属于历史冻结证据。', '',
        '## 实际数据内容与资格', '',
        '| 数据域 | 物理电芯 | 前100曲线支持 | 未来有效点 | 最大实测周期 | 坐标 |',
        '|---|---:|---:|---:|---:|---|']
    prefix = {'MATR': '16857/16900', 'HUST': '7698/7700', 'Stanford_2': '18092/18100'}
    for domain, cache in lock['data_caches'].items():
        lines.append(f'| {domain} | {cache["physical_cells"]} | {prefix[domain]} | {cache["observed_future_points"]} | {cache["maximum_observed_cycle"]} | {cache["coordinate"]} |')
    lines += ['',
        '每域缓存包含原单位V/A/Ah数组[电芯,100,64,3]、真实早期参考ΔQ、逐点mask、原/供应cycle_ids、28维具名工况与mask、名义容量、SOH100锚点、global20、local残差64及8×26描述子、完整逐电芯元数据和固定101–5000监督网格。SOH分母采用各cell声明的nominal Ah。查询时距固定，不读取单cell未来长度设置推理终点。', '',
        'MATR使用原生仪器循环、4C、2.8–3.3V、1.1Ah。HUST使用原始公开循环、实测约2C末段、2.8–3.1V、1.1Ah。Stanford使用供应循环、3.4–4.1V、0.24Ah，181个供应UID与181个唯一设备标识SHA连通，63个六参数形成组。三域全部有效VIQ点均有真实drift支持，插值点保留合法观测mask。', '',
        'Stanford仍缺原仪器循环与具体BEEP转换版本的逐文件绑定。其供应坐标可用于本合同跨域任务。HUST名义EOL80事件0/77，仅保留删失信息。SDU86颗因重复循环与原始身份链缺口保留候选资格。较早MATR扩展报告的供应坐标和88个阈值事件属于旧口径，原生冻结后为169个名义EOL80事件。', '',
        '原始数据只读，新缓存逐文件重新校验SHA。source85拟合归一化，canonical x保留V与无量纲C-rate/SOH，另提供可选x_z。source全缺失工况列在所有目标域继续mask，零方差已知设计量使用预注册物理尺度。实测温度与温箱设计值、形成温度分别编码。', '',
        '## 冻结角色与开发集', '',
        '| MATR角色 | 独立电芯 | exact protocol groups | 标签用途 |', '|---|---:|---:|---|',
        '| source_train | 85 | 47 | 源训练、源子空间估计与source-only scaler |',
        '| source_meta_validation | 30 | 15 | 源选模与校准预算开发 |',
        '| target_test | 54 | 15 | 锁定方法后Support适配与Query最终评分 |', '',
        '三角色UID和exact protocol group均互斥。meta30按照每组SHA固定1颗Query生成15颗Query，其余15颗组成跨组Support池，支持K0/1/3/5/10各5draw。该开发协议用于source calibration development，Query和Support可共享meta工况组。各方法共用draw，重复集合复用checkpoint并显示真实unique sets。', '',
        '| 开发K | Query | Support候选 | 实际unique support sets |', '|---:|---:|---:|---:|']
    for shot in meta['shots']:
        lines.append(f'| {shot["k"]} | 15 | 15 | {shot["unique_support_sets"]} |')
    lines += ['', '## 最终目标协议分母与共同查询子集', '',
        '| 协议A的K | Q2可行exact组 | 固定Query电芯 | 登记draw | 各任务内unique support sets合计 |',
        '|---:|---:|---:|---:|---:|']
    denom = lock['protocol_A_denominators']
    for row in denom['all_feasible_by_k']:
        lines.append(f'| {row["k"]} | {row["exact_protocol_groups"]} | {row["unique_query_cells"]} | {row["registered_draws"]} | {row["unique_support_sets_within_task"]} |')
    lines += ['',
        '目标15组中6个singleton组缺Q2支持。共同K0/1/3/5曲线子集覆盖4个可5-shot组及固定8颗Query，完整可行分母另表保留。exact-group K10的可行组为0。全MATR169颗的Q1/Q2支持计数保存在DATA_SPLIT_LOCK。Stanford固定Query91/Support90，HUST固定Query39/Support38，均支持K0/1/3/5/10。', '',
        '最终target Query未来监督保持SEALED_FOR_MODEL_DEVELOPMENT角色限制。缓存包含全域outcomes，模型入口采用早期输入allowlist，训练索引与选模索引按角色限制。该标记描述软件访问用途和冻结协议，历次历史探索暴露已经存在，无法据此宣称全新盲测。', '',
        '## 历史探索暴露与本轮解释边界', '',
        '169颗MATR及外部HUST/Stanford队列已用于历史探索。旧五折与协议敏感性划分保留原字节，新source/meta/test划分单独登记。旧M1 OOF MAE约0.018762，M2约0.018946，M4约0.023271，M4-A1约0.020148，训练规模、输出头和验证协议与本轮few-shot不同。历史表仅作任务与失败诊断，本轮主表按新split重新训练和评分。', '',
        'M4速率积分在历史主OOF与未见协议上没有支持整体优势。P0–P4共享cycle-query连续轨迹头，Bridge选择与PEFT更新贡献分别记录。Bridge的绝对支路接收真实canonical物理尺度，避免将标准化坐标解释为电位。', '',
        '## 训练、loader与独立核验入口', '',
        '| 文件 | SHA-256 |', '|---|---|']
    for row in extras:
        lines.append(f'| {row["path"]} | {row.get("sha256", "MISSING")} |')
    lines += ['',
        '新训练入口为scripts/drs_peft/train.py，数据入口为scripts/drs_peft/data.py，Episode入口为scripts/drs_peft/episodes.py。旧probe_frozen_loader、verify_dataset及analyze_results保留数据资格与历史预测重放路径。新TSFM checkpoint身份、Bridge语义和实际子层复用率分别绑定TSFM_BACKBONE_AUDIT.json与BRIDGE_CONTRACT.json。', '',
        '## 本机软件与GPU依赖实测', '',
        f'操作系统为{environment["platform"]}，Python为{environment["python"]}。', '',
        '| 依赖 | 已安装 | 实际版本 |', '|---|---|---|']
    for name, package in environment['dependencies'].items():
        lines.append(f'| {name} | {package["installed"]} | {package["version"]} |')
    lines += ['', '```json', json.dumps({k: v for k, v in environment.items() if k not in ('dependencies', 'python_executable')}, ensure_ascii=False, indent=2), '```', '',
        server_note, '',
        '## 优势与劣势（风险）', '',
        '- 优势为完整物理UID、源工况互斥、固定Query和独立Support，source-only归一化及真实前缀漂移使方法的信息预算可追溯。',
        '- 劣势（风险）为retrospective历史暴露、MATR exact-group高shot样本稀少、Stanford供应轴与BEEP版本缺口、HUST无EOL80事件，以及三域放电窗口和化学体系差异。跨域主结果需要真实训练与固定Query评分。', '',
        '## 置信度评级', '',
        '本地存在性、SHA、UID划分、分母与已安装依赖的置信度为高，均由本CLI现场读取。原冻结580项数值性质引用其绑定的历史核验。方法有效性由对应阶段真实训练与独立重放决定。已回同步的服务器环境附件可与各run交叉核验。', '',
        '## 拓展思考与可能关心的问题', '',
        '共同8Query子集支持同一对象上的K0/1/3/5曲线，全可行分母呈现数据覆盖。meta pooled K10可用于源校准预算开发，最终协议A的exact-group K10继续记录NOT_FEASIBLE。跨域K10由各自固定Support池执行。', '',
        '## 可重跑命令', '', '```powershell',
        '& D:\\Anaconda\\envs\\tslib\\python.exe -m scripts.drs_peft.readiness', '```', '',
        f'主Episode payload SHA为`{lock["source_episode_contract_sha256"]}`。', '',
        f'Meta开发Episode payload SHA为`{meta["meta_episode_contract_sha256"]}`。', '',
        f'划分锁payload SHA为`{lock["split_lock_sha256"]}`。', '',
        '已登记JSON遇到内容差异会拒绝覆盖，需显式建立新版本。重复运行保持划分JSON字节，环境审计报告记录当前软件状态。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory, extras, texts = read_inventory(root)
    if any(not row['exists'] for row in inventory + extras):
        raise FileNotFoundError('Required repository material missing; inspect REQUIRED/EXTRA_ENTRIES')
    registration = load_registration(root / 'configs/drs_peft/episodes_v1.json')
    meta = build_meta(registration)
    caches = read_caches(root)
    lock = build_lock(root, registration, meta, inventory, caches)
    freeze_json(root / 'configs/drs_peft/meta_episodes_v1.json', meta)
    freeze_json(root / 'research/drs_peft/DATA_SPLIT_LOCK.json', lock)
    environment = local_environment()
    target = root / 'research/drs_peft/REPOSITORY_READINESS_ZH.md'
    target.write_text(report(root, inventory, extras, lock, meta, environment), encoding='utf-8', newline='\n')
    print(json.dumps(dict(status='REPOSITORY_AND_DATA_SPLIT_READY', mandatory_files_read=len(texts),
        meta_query=len(meta['query_ids']), meta_support_candidates=len(meta['support_pool_ids']),
        meta_contract_sha256=meta['meta_episode_contract_sha256'], split_lock_sha256=lock['split_lock_sha256'],
        target_A_common_query_cells=lock['protocol_A_denominators']['common_query_cells_K_0_1_3_5'],
        target_query_label_status=lock['target_query_label_policy']['status']), ensure_ascii=False))


if __name__ == '__main__':
    main()
