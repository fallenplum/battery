# V2 实验的文献依据

- Wortsman 等在 CVPR 2022 正文第 3 节给出 WiSE-FT，将零样本权重与微调权重线性插值。E0 对整个额外 LoRA 权重增量使用一个系数，缩放由源内数据选择。原论文的实证范围为视觉识别。 [论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf)
- Harrison、Sharma 和 Pavone 在 ALPaCA 第 III–IV 节从源任务学习有限维表示和权重先验，在线阶段执行贝叶斯线性回归。V2沿用冻结低维表示和源数据定尺度的思想，求解目标为正则化点估计。 [论文](https://arxiv.org/html/1807.08912)
- Patacchiola 等在 NeurIPS 2020 第 3 节及算法 1 通过共享深核从源任务迁移，并在给定少量 Support 后预测 Query。V2采用冻结表示上的固定高斯核，并在相同 Support 预算下比较全局和局部作用范围。 [论文](https://proceedings.neurips.cc/paper/2020/file/b9cfe8b6042cf759dc4c0cccb27a6737-Paper.pdf)
- Zhu、Cui 和 Zhang 在 NeurIPS 2024 第 3.2 节用测试样本到零样本失败集合的距离确定模型融合权重。样本条件化的更新强度已有先例。V2的可检验问题是电池早期响应相似性是否改善跨电芯的未来轨迹校准。 [论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/8cc7e1509fbfee9cabaacd3ab0bfe2b1-Paper-Conference.pdf)
