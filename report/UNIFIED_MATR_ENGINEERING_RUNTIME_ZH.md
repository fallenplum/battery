# MATR seed42 模型运行配置工程测量

## 测量范围

人工张量使用 P15/G20/L26、batch8、cycle101–2500 共2400个query，FP32、strictdet、CUBLAS4096:8、Torch intra-op4、正式入口默认inter-op24。每组每backend进行2次warmup与3次forward/backward计时。模型种子42，人工输入fixture种子422026。模型更新次数0，未来标签读取数0。

## 同进程耗时与内存

| 模型 | loop中位毫秒 | vectorized中位毫秒 | vectorized峰值allocated MiB | 峰值reserved MiB |
|---|---:|---:|---:|---:|
| M0 | 47.43 | 18.98 | 151.78 | 168.00 |
| M1 | 32.34 | 26.59 | 158.93 | 174.00 |
| M2 | 162.33 | 28.74 | 200.50 | 216.00 |
| M3 | 146.62 | 26.91 | 199.90 | 216.00 |
| M3-NL | 133.28 | 28.00 | 186.13 | 212.00 |
| M3-NP | 24.01 | 32.05 | 158.33 | 174.00 |
| M3-NR | 148.60 | 25.36 | 200.50 | 216.00 |

M2、M3、M3-NL、M3-NR 的同进程时间比为4.76–5.86。M0、M1、M3-NP控制路径的短测波动记录了启动、调度与当前环境差异。每项参数增量均为0。

## 同步位置与瓶颈

轻量计时下，M3 SoftPersistence forward的host时间从43.82ms降至1.41ms。它采用64个learned retain的dense causal EMA展开，保留首个有效观测、缺失内部hold、gapclamp1与原soft penalty。独立loop oracle覆盖28组CPU/CUDA、FP64/FP32 forward与value/cycle/retain梯度，含allmissing、首尾缺失、非均匀/倒序坐标及retain0/1。

最终vectorized五个CPU scan bridge的host中位时间分别为 elapsed0.092ms、local SSD0.186ms、history0 0.105ms、history1 0.099ms、rate integral0.159ms。逐pass合计中位0.781ms，forward单向payload233600bytes。这些桥接时间包含前置GPU工作的同步等待。算子profiler另行执行，轻量位置计时与七组批次中位数分别记录。

## 当前资源环境

匹配测量期间，数据构建进程PID69028持续运行，CPU累计时间122.11→130.33秒，RSS343.5–388.8MiB。主机可用RAM9.42–10.01GiB，profile进程采样RSS峰值1057.30MiB。GPU总量8188MiB，系统占用采样2956–3232MiB，GPU利用率采样11%–39%。进程、遥测和算子trace均保留原始回执。

## 87150updates时间投影

按七组相同12450updates加权，vectorized纯forward/backward投影2323.61秒，约0.645小时。首轮与配对loop投影为1.98–2.40小时。完整训练另有优化器、梯度裁剪、输入传输、验证与checkpoint成本。3次短测及实时资源竞争限制端到端外推精度，置信度中。

## 优势与风险

参数预算和数学oracle共同约束了优化范围。GPU allocated峰值约200.50MiB，相对loop约增加40.64MiB。向量化使用与前缀长度平方相关的中间量，本任务前缀固定100。数学等价核验置信度高，当前geometry内存测量置信度高。

## 拓展思考

正式运行已有逐epoch累计时间记录，可据它核对完整训练的实际进度。

## 你可能想知道

本记录度量当前硬件上的工程开销，研究效应由后续冻结数据上的完整模型实验提供。
