# 修改方案: Flow Model / ODE 兼容性

## 问题描述

很多读者(约70%)认为SOTA是flow model/ODE:
- 中间步骤是deterministic的
- noise只在最开始(xT)
- 后面每一步都是确定性的

这会让读者觉得"这个方法不是为我设计的"。

## 解决方案

### 1. Introduction修改

在Introduction的第2-3段添加以下内容:

```latex
% 在描述完inference-time scaling后添加

Importantly, our framework applies broadly to both stochastic and deterministic
samplers. While some diffusion models---particularly flow-based methods and
ODE solvers---use deterministic transitions after the initial noise,
inference-time search \emph{requires} stochasticity to explore different
generation paths. For deterministic samplers, this is achieved by intentionally
injecting noise at selected timesteps, enabling the search to explore diverse
trajectories that would otherwise be inaccessible. This deliberate introduction
of stochasticity for exploration has been adopted in prior work on
reinforcement learning for diffusion models~\citep{black2023training,fan2024reinforcement}
and is conceptually similar to exploration bonuses in bandit algorithms.
```

### 2. 关键论点

1. **Noise是为了Exploration**: 如果不加noise，本质上没得探索
2. **适用于ODE sampler**: 通过主动引入noise实现
3. **已有先例**: 引用RL for diffusion的相关工作

### 3. 需要添加的引用

```bibtex
@article{black2023training,
  title={Training diffusion models with reinforcement learning},
  author={Black, Kevin and others},
  journal={arXiv preprint arXiv:2305.13301},
  year={2023}
}

@article{fan2024reinforcement,
  title={Reinforcement learning for fine-tuning text-to-image diffusion models},
  author={Fan, Ying and others},
  journal={NeurIPS},
  year={2024}
}
```

### 4. 验证清单

- [ ] Introduction前3段内是否提到flow model兼容性
- [ ] 是否解释了noise的exploration作用
- [ ] 是否引用了相关工作
- [ ] 读者能否理解方法对ODE也适用
