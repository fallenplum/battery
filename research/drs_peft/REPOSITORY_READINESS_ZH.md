# DRS-PEFT 仓库、数据与少样本划分审计

本轮完整读取合同16项材料所展开的17个文件，实际存在17/17。每项保留完整字节SHA。

**状态为数据与仓库就绪。P0模型性能、P1–P5科学实验由各阶段真实运行记录验收。**

## 必读材料的实际存在性与版本

| 合同项 | 文件 | 字节数 | SHA-256 | 核验用途 |
|---|---|---:|---|---|
| 1 | README.md | 14752 | 9fa6ee24c89e87612ca63a36beeb850c84ea5127361dab09e0b33fcf16ca8d16 | 历史结果索引、队列边界与独立核验入口 |
| 2 | contracts/SRSR_Physics_统一研究问题_一次性数据处理与模型实验总合同.md | 15796 | c59764b219f629eb5767706809a5393536fe977fa6092ae127156340c62a50d7 | 旧统一任务合同，保留旧任务定义与历史暴露 |
| 3 | configs/unified_matr_seed42.json | 3934 | 25a488e1e27e6dffe5246ad8a9433810a9906aa7eb29d1288f4d21294dc6e72d | 169颗、原生1–100圈、2.8–3.3V、1.1Ah与旧seed42训练配置 |
| 4 | data/unified_matr_seed42/DATA_FREEZE.json | 85717 | 276016ea2f72c2d9c366c44b00906c80cf8234f741b48b60a6a2562c57fff4fb | 冻结主数据及依赖的SHA清单 |
| 5 | data/unified_matr_seed42/dataset_qa_report.md | 58010 | 4ba81bd4360a52ac907ecec15ecbd4b654dc3adad704886a247da2b5fed20538 | 历史580项数据核验记录 |
| 6 | data/unified_matr_seed42/split_contract.json | 36501 | 292c3f5e2615b700f3604b730c44963a0a1760aa9a4f29aa56ec6f16a2492b9b | 旧五折及协议敏感性划分，保留原字节 |
| 7 | models/unified_matr/README.md | 10844 | b6ca3055287549403f414c39b0fdde44a960330b8f1e11250f38c83d584a9f5f | 旧M1/M2编码器、global20、local8×26、原Flow与积分头语义 |
| 7 | models/unified_matr/model.py | 20487 | 485bfd06f41da5137f85645577b443ebf7a9b72687d4b5dc84928020a4c96252 | 旧电池模型可复用编码器的实际实现 |
| 8 | scripts/unified_matr/train_seed42.py | 30240 | b5aa9251414191bb74c818064cef5bb6fe2fc1f01971bdd6f3027a757f174608 | 旧训练、input allowlist、折内标准化及重放入口 |
| 9 | report/UNIFIED_MATR_SEED42_FINAL_ZH.md | 12368 | 1ae460cee979d4653ef20f30f51933ccfa1ba6c650d8596ed633a45f010bf1f4 | 旧M1/M2 OOF与负结果，历史探索参考 |
| 10 | report/M4_FULL_METHOD_FINAL_ZH.md | 6928 | f6684cbc2f0c2972fcf49a24c5a708d27dffd236c8ed7dac064e96c4082393b7 | M4真实训练与完整rate-integral方法结果 |
| 11 | report/M4_ABLATIONS_SEED42_FINAL_ZH.md | 9485 | 62ecbba553d15c88346c67f083e74d14f4619812218023b0519a5d396e8e758b | A1–A4独立派生消融及M4速率积分的负证据 |
| 12 | report/MULTIFAMILY_EXTENSION_REPORT_ZH.md | 10974 | e82e293f284691ff8865c8787cf9af0698c266f24e27b9820ce7bc182503f744 | MATR/Stanford历史供应坐标斜率任务与暴露 |
| 13 | report/MATR_EXTENSION_QUALIFICATION_ZH.md | 4714 | 8d2aca3b75a3fa3f2f989778ff52cb89e55c319858987e89dae241e700b6a569 | 较早供应坐标资格快照，后续native冻结已补足仪器轴 |
| 14 | report/STANFORD_EXTENSION_QUALIFICATION_ZH.md | 3835 | 09f90c4465d1655a38e1002e5502d21d209d52e37cbc780f0736370b4c0e0079 | 181供应UID、形成参数、3.4–4.1V与BEEP版本缺口 |
| 15 | report/HUST_RAW_IDEA_FEASIBILITY_ZH.md | 10290 | 79b6ef33c835cc8b8a9343fe39251ece2ffc6b0bd886373f2ebb8ce1182b12d6 | 77颗原始队列、实际2C末段、138422点及0个EOL80事件 |
| 16 | report/SDU_DATA_AUDIT_ZH.md | 8378 | 4ac209f99546a11ef0a22ff69fa1c61fdd6bbbb18d4ffa94e80705e87c6f5cf2 | 86颗完整供应元数据审计，重复循环与身份链缺口 |

原DATA_FREEZE状态为DATA_FREEZE_PASS，绑定44项文件，原QA有580项且状态为PASS。本CLI验证当前必读文件与新缓存SHA，580项数值核验属于历史冻结证据。

## 实际数据内容与资格

| 数据域 | 物理电芯 | 前100曲线支持 | 未来有效点 | 最大实测周期 | 坐标 |
|---|---:|---:|---:|---:|---|
| MATR | 169 | 16857/16900 | 123431 | 2238 | original_instrument_cycle |
| HUST | 77 | 7698/7700 | 138422 | 2689 | original_instrument_cycle |
| Stanford_2 | 181 | 18092/18100 | 170397 | 1624 | supplied_cycle_number |

每域缓存包含原单位V/A/Ah数组[电芯,100,64,3]、真实早期参考ΔQ、逐点mask、原/供应cycle_ids、28维具名工况与mask、名义容量、SOH100锚点、global20、local残差64及8×26描述子、完整逐电芯元数据和固定101–5000监督网格。SOH分母采用各cell声明的nominal Ah。查询时距固定，不读取单cell未来长度设置推理终点。

MATR使用原生仪器循环、4C、2.8–3.3V、1.1Ah。HUST使用原始公开循环、实测约2C末段、2.8–3.1V、1.1Ah。Stanford使用供应循环、3.4–4.1V、0.24Ah，181个供应UID与181个唯一设备标识SHA连通，63个六参数形成组。三域全部有效VIQ点均有真实drift支持，插值点保留合法观测mask。

Stanford仍缺原仪器循环与具体BEEP转换版本的逐文件绑定。其供应坐标可用于本合同跨域任务。HUST名义EOL80事件0/77，仅保留删失信息。SDU86颗因重复循环与原始身份链缺口保留候选资格。较早MATR扩展报告的供应坐标和88个阈值事件属于旧口径，原生冻结后为169个名义EOL80事件。

原始数据只读，新缓存逐文件重新校验SHA。source85拟合归一化，canonical x保留V与无量纲C-rate/SOH，另提供可选x_z。source全缺失工况列在所有目标域继续mask，零方差已知设计量使用预注册物理尺度。实测温度与温箱设计值、形成温度分别编码。

## 冻结角色与开发集

| MATR角色 | 独立电芯 | exact protocol groups | 标签用途 |
|---|---:|---:|---|
| source_train | 85 | 47 | 源训练、源子空间估计与source-only scaler |
| source_meta_validation | 30 | 15 | 源选模与校准预算开发 |
| target_test | 54 | 15 | 锁定方法后Support适配与Query最终评分 |

三角色UID和exact protocol group均互斥。meta30按照每组SHA固定1颗Query生成15颗Query，其余15颗组成跨组Support池，支持K0/1/3/5/10各5draw。该开发协议用于source calibration development，Query和Support可共享meta工况组。各方法共用draw，重复集合复用checkpoint并显示真实unique sets。

| 开发K | Query | Support候选 | 实际unique support sets |
|---:|---:|---:|---:|
| 0 | 15 | 15 | 1 |
| 1 | 15 | 15 | 4 |
| 3 | 15 | 15 | 5 |
| 5 | 15 | 15 | 5 |
| 10 | 15 | 15 | 5 |

## 最终目标协议分母与共同查询子集

| 协议A的K | Q2可行exact组 | 固定Query电芯 | 登记draw | 各任务内unique support sets合计 |
|---:|---:|---:|---:|---:|
| 0 | 9 | 18 | 45 | 9 |
| 1 | 6 | 12 | 30 | 21 |
| 3 | 6 | 12 | 30 | 20 |
| 5 | 4 | 8 | 20 | 13 |
| 10 | 0 | 0 | 0 | 0 |

目标15组中6个singleton组缺Q2支持。共同K0/1/3/5曲线子集覆盖4个可5-shot组及固定8颗Query，完整可行分母另表保留。exact-group K10的可行组为0。全MATR169颗的Q1/Q2支持计数保存在DATA_SPLIT_LOCK。Stanford固定Query91/Support90，HUST固定Query39/Support38，均支持K0/1/3/5/10。

最终target Query未来监督保持SEALED_FOR_MODEL_DEVELOPMENT角色限制。缓存包含全域outcomes，模型入口采用早期输入allowlist，训练索引与选模索引按角色限制。该标记描述软件访问用途和冻结协议，历次历史探索暴露已经存在，无法据此宣称全新盲测。

## 历史探索暴露与本轮解释边界

169颗MATR及外部HUST/Stanford队列已用于历史探索。旧五折与协议敏感性划分保留原字节，新source/meta/test划分单独登记。旧M1 OOF MAE约0.018762，M2约0.018946，M4约0.023271，M4-A1约0.020148，训练规模、输出头和验证协议与本轮few-shot不同。历史表仅作任务与失败诊断，本轮主表按新split重新训练和评分。

M4速率积分在历史主OOF与未见协议上没有支持整体优势。P0–P4共享cycle-query连续轨迹头，Bridge选择与PEFT更新贡献分别记录。Bridge的绝对支路接收真实canonical物理尺度，避免将标准化坐标解释为电位。

## 训练、loader与独立核验入口

| 文件 | SHA-256 |
|---|---|
| scripts/unified_matr/probe_frozen_loader.py | b6d418d188c48169576047bf9c3def4bbc27ea9ffbff6a81cbe07deaef980027 |
| scripts/unified_matr/verify_dataset.py | df34c6ed383eb9f54b76d6c7102ff1003aaa93d62ac1d936ca1c8a9732a29efc |
| scripts/unified_matr/verify_physics_features.py | 9ca697a5585f274ce5977f6e6d766dac5f0fd027064b5b5fd4d1bf4db0cf8712 |
| scripts/unified_matr/analyze_results.py | 328dd6817d274613c4ae721bb610bdcefdef9e1c368759bada4e17102fcec7c4 |
| n012/data.py | 452239053e363957e697585964a0d0e36002ff5ff9889c1d57f9cf958c5bb728 |
| scripts/drs_peft/data.py | 4dc110c68ed99a28ef61dfeb0bad50951d4855a3fb17b29dabc86d4b9ed5f585 |
| scripts/drs_peft/episodes.py | d8540a62a8aa5d932646315a863cff03f3e3d56e563a0e0caf4e96c006383a94 |
| report/drs_peft/DATA_READINESS_ZH.md | 1013ffbf4bea9f6fc899159a834219dad27d2b8cffe5e47c39345cbf6a88b263 |

新训练入口为scripts/drs_peft/train.py，数据入口为scripts/drs_peft/data.py，Episode入口为scripts/drs_peft/episodes.py。旧probe_frozen_loader、verify_dataset及analyze_results保留数据资格与历史预测重放路径。新TSFM checkpoint身份、Bridge语义和实际子层复用率分别绑定TSFM_BACKBONE_AUDIT.json与BRIDGE_CONTRACT.json。

## 本机软件与GPU依赖实测

操作系统为Windows-10-10.0.26200-SP0，Python为3.11.14。

| 依赖 | 已安装 | 实际版本 |
|---|---|---|
| torch | True | 2.4.1+cu121 |
| numpy | True | 1.26.4 |
| pandas | True | 2.1.4 |
| pyarrow | True | 23.0.0 |
| scipy | True | 1.11.4 |
| uni2ts | True | 2.0.0 |
| transformers | True | 5.12.1 |
| huggingface-hub | True | 1.20.1 |
| safetensors | True | 0.8.0 |
| einops | True | 0.7.0 |
| gluonts | True | 0.14.4 |
| lightning | True | 2.6.5 |
| paramiko | True | 3.5.1 |
| pytest | True | 8.4.2 |

```json
{
  "platform": "Windows-10-10.0.26200-SP0",
  "python": "3.11.14",
  "torch": {
    "version": "2.4.1+cu121",
    "cuda_build": "12.1",
    "cuda_available": true,
    "cudnn": 90100
  },
  "gpu": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
      "total_memory_MiB": 8187
    }
  ],
  "nvidia_smi": "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 560.94"
}
```

服务器同期硬件核验由主执行代理提供，当前记录为8×RTX A6000、每卡49140MiB，准备以GPU5隔离单作业执行。新隔离环境计划复用只读Torch2.4.1+cu121。旧系统Python3.8存在libcupti缺失。服务器软件与实际运行细节以主执行代理后续ENV.json为准，本报告未复制服务器连接凭据。

## 优势与劣势（风险）

- 优势为完整物理UID、源工况互斥、固定Query和独立Support，source-only归一化及真实前缀漂移使方法的信息预算可追溯。
- 劣势（风险）为retrospective历史暴露、MATR exact-group高shot样本稀少、Stanford供应轴与BEEP版本缺口、HUST无EOL80事件，以及三域放电窗口和化学体系差异。跨域主结果需要真实训练与固定Query评分。

## 置信度评级

本地存在性、SHA、UID划分、分母与已安装依赖的置信度为高，均由本CLI现场读取。原冻结580项数值性质引用其绑定的历史核验。方法有效性尚待真实训练和独立重放。服务器硬件依据主执行代理同期实测记录，环境附件补齐后可独立复核。

## 拓展思考与可能关心的问题

共同8Query子集支持同一对象上的K0/1/3/5曲线，全可行分母呈现数据覆盖。meta pooled K10可用于源校准预算开发，最终协议A的exact-group K10继续记录NOT_FEASIBLE。跨域K10由各自固定Support池执行。

## 可重跑命令

```powershell
& D:\Anaconda\envs\tslib\python.exe -m scripts.drs_peft.readiness
```

主Episode payload SHA为`1bc2f831eebc70998c73c6224773e590754c23e8abc395b0f3dbe597b4ff80c6`。

Meta开发Episode payload SHA为`a3ddb86c9130826211bb6ee1999ad825720f3764acb539f0840dbe24e6e43027`。

划分锁payload SHA为`1179b9b9860d785fdc676f3ea8c89082d23ca29ba8238363fcf3c42bb35b9f0c`。

已登记JSON遇到内容差异会拒绝覆盖，需显式建立新版本。重复运行保持划分JSON字节，环境审计报告记录当前软件状态。
