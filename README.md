# Global Search 与 Local Search 实验项目

## 📖 项目概述

本项目验证在扩散模型推理中，**compute-aware的全局搜索调度策略**相比纯增加采样步数或简单的search方法，能够在相同NFE预算下获得更好的性能。

### 核心思想

- **Local Search**: 通过verifier和算法从xt到xt-1
- **Global Search**: 决定在不同步数使用什么search策略，如何分配算力
- **Action**: 不是噪声本身，而是search策略的选择（search_mode, budget, primitive_type）

## 📁 文档结构

- **[PIPELINE_GUIDE.md](./PIPELINE_GUIDE.md)** - Pipeline使用指南（**重要**）
- **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)** - 实现总结
- **[experiment_plan.md](./experiment_plan.md)** - 完整的8周实验计划（详细版）
- **[experiment_checklist.md](./experiment_checklist.md)** - 可执行的实验清单和进度追踪
- **[quick_start.md](./quick_start.md)** - 快速开始指南
- **[PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)** - 项目结构说明

## 🎯 实验目标

### 验证假设
1. 纯增加采样步数（50→100→200）的提升有限
2. 将部分NFE用于search/local search可以明显提升FID/IS
3. Compute-aware调度策略能够更高效地分配NFE

### 实验设计
- **数据集**: CIFAR-10 → ImageNet-64
- **Baseline方法**: Pure Sampling, Random Search, NLG, ZO-N
- **目标方法**: Compute-aware调度策略（初期用heuristic，未来可扩展为RL）

## 🗺️ 实验路线图

```
Phase 0: 环境搭建 (第1-2天)
    ↓
Phase 1: CIFAR-10基础实验 (第1-2周)
    ├── Pure Sampling Baseline
    ├── Random Search
    ├── NLG
    └── ZO-N
    ↓
Phase 2: ImageNet-64迁移 (第3-4周)
    ├── 框架迁移
    ├── 复现Random/ZO
    └── NLG+Random组合
    ↓
Phase 3: Compute-Aware策略 (第5-6周)
    ├── 方法实现（Heuristic策略）
    ├── 与Baseline对比
    └── Ablation Study
    ↓
Phase 4: 结果整理 (第7-8周)
    ├── 结果可视化
    ├── 实验报告
    └── 代码整理
```

## 🚀 快速开始

### 环境设置

1. **克隆项目**
```bash
git clone https://github.com/pkucaoyuan/Global_searching.git
cd Global_searching
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备外部代码库**
本项目需要克隆以下外部代码库到 `code_repos/` 目录：
```bash
mkdir -p code_repos
cd code_repos

# Diffusion-TTS (用于EDM模型和搜索方法)
git clone https://github.com/rvignav/diffusion-tts.git

# 其他可选代码库（根据需要）
# git clone https://github.com/sayakpaul/tt-scale-flux.git
# git clone https://github.com/XiangchengZhang/Diffusion-inference-scaling.git
```

### 运行第一个实验

参考 [实验步骤指南](./docs/EXPERIMENT_STEPS.md) 或查看 `quick_start.md`

**ImageNet-64 Diffusion-TTS 实验**:
```bash
python scripts/run_diffusion_tts_experiment.py \
    --config configs/imagenet64_diffusion_tts.yaml
```

## 📊 预期结果

### Baseline结果（第2周结束）
- CIFAR-10上pure sampling的scaling曲线（验证变平现象）
- Random Search和NLG在相同NFE下的性能提升
- 结论：search方法确实比纯加步数更有效

### 最终结果（第6周结束）
- Compute-aware策略在CIFAR-10和ImageNet-64上的性能
- 与各种baseline的对比分析
- Ablation study结果

## 🛠️ 关键技术点

### MDP建模
- **State**: `(xt, t, prompt, history/score)`
- **Action**: `(search_mode, budget, primitive_type)`
- **Reward**: `Δt = verifier(xt-1) - verifier(xt) - λ·computation`

### Search Primitives
- Random Search: 多噪声并行采样
- ZO-N: Pivot-based迭代搜索
- NLG: 噪声级别引导
- BFS/DFS: 经典搜索方法

### Compute-Aware策略（初期Heuristic）
1. **固定分配**: 根据step重要性固定分配budget
2. **自适应阈值**: 根据verifier改善情况动态调整
3. **多阶段策略**: heavy→light→no search

## 📚 相关代码库

### 核心框架
- `sayakpaul/tt-scale-flux` - Random Search + ZO + verifier接口
- `XiangchengZhang/Diffusion-inference-scaling` - ImageNet实验脚本

### Local Search方法
- `harveymannering/NoiseLevelGuidance` - NLG实现
- `rvignav/diffusion-tts` - Noise Trajectory Search

### 参考实现
- `zacharyhorvitz/Fk-Diffusion-Steering` - 粒子系统设计
- `masa-ue/SVDD` - value-based方法

## ✅ 进度追踪

使用`experiment_checklist.md`追踪每日进度，包含：
- 任务完成情况
- 关键指标追踪表格
- 问题排查指南

## 📝 实验记录

建议为每个实验创建单独的记录文件，包含：
- 实验配置
- 运行命令
- 结果数据
- 观察和分析

## 🔍 关键检查点

- **第1周结束**: CIFAR-10 baseline完成
- **第2周结束**: 验证search方法有效性
- **第4周结束**: ImageNet迁移完成
- **第6周结束**: Compute-aware策略有初步结果

## 💡 注意事项

1. **NFE对齐**: 所有对比实验必须确保NFE预算相同
2. **复现性**: 记录随机种子和所有配置参数
3. **计算资源**: ImageNet实验需要大量GPU，合理规划
4. **代码整理**: 保持代码结构清晰，方便后续扩展

---

## ✨ 已实现的功能

✅ **完整的Pipeline架构**
- Local Search层：NoSearch, RandomSearch, LocalNoiseSearch, ZeroOrderSearch
- Global Search层：FixedBudgetPolicy, AdaptiveThresholdPolicy, MultiStagePolicy
- 统一的接口和NFE计数

✅ **两层级Search架构**
- Local Search：从xt到xt-1的搜索方法
- Global Search：高层调度策略，决定何时使用哪种search、分配多少算力

✅ **MDP建模框架**
- State: (xt, t, prompt, history/score)
- Action: (search_mode, budget, primitive_type)
- Reward: Δt - λ·computation

---

**开始使用**: 查看 `PIPELINE_GUIDE.md`  
**了解实现**: 查看 `IMPLEMENTATION_SUMMARY.md`  
**详细计划**: 查看 `experiment_plan.md`  
**进度追踪**: 使用 `experiment_checklist.md`

