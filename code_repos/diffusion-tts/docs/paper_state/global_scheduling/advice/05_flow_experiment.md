# 修改方案: Flow-based Model实验

## 问题描述

当前实验缺少flow-based model:
- 只有Stable Diffusion (SDE-based)
- 只有EDM
- 没有纯ODE/flow model

## 解决方案

### 1. 选择合适的Flow Model

推荐选项（按实验便利性排序）:

| 模型 | 类型 | 优点 | 缺点 |
|------|------|------|------|
| Rectified Flow (small) | Flow | 轻量级 | 可能没有预训练权重 |
| Consistency Models | Flow-like | 有预训练 | 不是纯flow |
| Flow Matching (CIFAR) | Flow | 经典 | 分辨率低 |
| mini-SD with DDIM | ODE | 已有代码 | 技术上是ODE不是flow |

**导师建议**: 不需要大模型，小模型能出结果即可

### 2. 实验设计

```latex
\subsection{Flow-based Model: Applicability to Deterministic Samplers}
\label{sec:exp-flow}

To demonstrate that our scheduling framework applies beyond stochastic
samplers, we evaluate on [MODEL NAME], a flow-based model that uses
deterministic ODE transitions. Following prior work~\citep{...}, we
introduce stochasticity by injecting small noise at selected timesteps,
enabling exploration of different generation paths.

\begin{table}[t]
\centering
\caption{Flow-based model results. Even with originally deterministic
sampling, our global scheduling improves quality by strategically
allocating search compute.}
\label{tab:flow_results}
\begin{tabular}{c|c|c}
\toprule
\textbf{NFE} & \textbf{Naive} & \textbf{Online (Ours)} \\
\midrule
[待填充] & [待填充] & [待填充] \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:flow_results} shows that...
```

### 3. 实现要点

1. **Noise注入策略**:
   - 在高价值timestep注入小noise
   - 或使用stochastic版本的sampler

2. **验证指标**:
   - 与SD/EDM保持一致（Brightness, Compressibility）
   - 或使用该模型常用的指标

3. **NFE设置**:
   - 与模型默认步数相关
   - 选择3个不同budget级别

### 4. 呼应Introduction

确保Introduction中有如下呼应:

> "For deterministic samplers, this is achieved by intentionally
> injecting noise at selected timesteps..."

实验结果验证这一claim。

### 5. 验证清单

- [ ] 选定了具体的flow model
- [ ] 实验代码能运行
- [ ] 结果填入table
- [ ] Introduction中有对应说明
- [ ] 讨论了noise注入策略
