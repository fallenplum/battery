# P1 三种子门槛审计

源seed为[42, 123, 2026]，门槛状态为**PASSED**。主rank4的捕获差按同一exact protocol跨seed配对，再以protocol为单位进行2000次bootstrap。seed方向、seed间波动和每seed基稳定性另行保留。该区间以本轮已拟合的source seed为条件。

平均bootstrap基重合为0.937482619089002。label-shuffle控制弱化观测为False。当前证据限于固定source checkpoint的共享梯度方向，电池特异机制仍未建立。单cell低秩与固定head/bridge共同方向会产生相近谱形。

P2同rank主比较及近实际参数匹配对照均允许继续。额外rank2/8曲线及P3/P4扩展放行状态为True。零更新P1尚未评价真实留出更新收益分支，阴性记录保留该证据边界。最终目标Query标签参与梯度与选模计算为0。
