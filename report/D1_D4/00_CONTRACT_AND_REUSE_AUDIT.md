# D1–D4 预注册与复用审计

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
