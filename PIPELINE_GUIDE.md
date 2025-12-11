# Pipeline使用指南

## 📋 概述

本项目实现了**两层级的Search架构**：

1. **Local Search（局部搜索）**：在单个时间步从xt到xt-1的搜索方法
2. **Global Search（全局搜索）**：高层调度策略，决定在不同步数使用什么search策略、如何分配算力

## 🏗️ 架构设计

### Local Search层

Local Search是在单个时间步上执行的搜索方法，包括：

- **NoSearch**: 标准采样，不进行搜索
- **RandomSearch**: 采样多个完整轨迹，选择verifier score最高的
- **LocalNoiseSearch**: 在每个时间步采样多个候选xt-1，选择最优的
- **ZeroOrderSearch**: Pivot-based迭代搜索，在初始噪声邻域搜索

### Global Search层

Global Search是高层调度框架，基于MDP建模：

- **State**: `(xt, t, prompt, history/score)`
- **Action**: `(search_mode, budget, primitive_type)`
- **Reward**: `Δt = verifier(xt-1) - verifier(xt) - λ·computation`

**Global Search策略**：

1. **FixedBudgetPolicy**: 固定分配策略
   - 根据step重要性固定分配budget
   - 例如：前1/3步60% budget，后2/3步40%

2. **AdaptiveThresholdPolicy**: 自适应阈值策略
   - 根据verifier改善情况动态调整
   - 如果改善不足，增加search budget

3. **MultiStagePolicy**: 多阶段策略
   - 前期：heavy search
   - 中期：light search
   - 后期：no search

## 🚀 使用方法

### 方法1：直接使用Local Search

```python
from src.pipeline.sampling_pipeline import create_pipeline
from src.models.base_model import BaseDiffusionModel  # 需要实现具体模型
from src.verifiers.base_verifier import BaseVerifier  # 需要实现具体verifier

# 创建pipeline（不使用Global Search）
pipeline = create_pipeline(
    model=model,
    verifier=verifier,
    method="random",  # 或 "no_search", "local", "zo"
)

# 采样
samples, info = pipeline.sample(
    method="random",
    batch_size=32,
    num_steps=50,
    num_candidates=4,  # Random Search的候选数
)
```

### 方法2：使用Global Search

```python
from src.pipeline.sampling_pipeline import create_pipeline
from src.search.global_search import FixedBudgetPolicy

# 创建pipeline（使用Global Search）
pipeline = create_pipeline(
    model=model,
    verifier=verifier,
    method="global",
    global_policy_type="fixed",
    total_nfe_budget=200,
    early_ratio=0.6,
    early_steps_ratio=0.33,
)

# 采样
samples, info = pipeline.sample(
    method="global",
    batch_size=32,
    num_steps=50,
)
```

### 方法3：自定义Global Search策略

```python
from src.search.global_search import (
    GlobalSearch,
    AdaptiveThresholdPolicy,
)
from src.pipeline.sampling_pipeline import SamplingPipeline

# 创建自定义策略
policy = AdaptiveThresholdPolicy(
    total_nfe_budget=200,
    threshold=0.1,
    base_budget=10,
    max_budget=50,
)

# 创建Global Search
global_search = GlobalSearch(
    model=model,
    verifier=verifier,
    policy=policy,
)

# 采样
samples, info = global_search.sample(
    batch_size=32,
    num_steps=50,
)
```

## 📝 代码示例

### 完整示例：运行不同方法的对比实验

```python
import torch
from src.pipeline.sampling_pipeline import create_pipeline
from src.utils.nfe_counter import NFECounter

# 假设model和verifier已加载
# model = load_your_model()
# verifier = load_your_verifier()

methods = ["no_search", "random", "local", "zo", "global"]
results = {}

for method in methods:
    print(f"\n运行方法: {method}")
    
    # 创建pipeline
    pipeline = create_pipeline(
        model=model,
        verifier=verifier,
        method=method if method != "global" else "global",
        global_policy_type="fixed" if method == "global" else None,
        total_nfe_budget=200,
    )
    
    # 采样
    nfe_counter = NFECounter()
    samples, info = pipeline.sample(
        method=method,
        batch_size=32,
        num_steps=50,
        nfe_counter=nfe_counter,
    )
    
    results[method] = {
        "samples": samples,
        "nfe": info.get("nfe", nfe_counter.total_nfe),
        "verifier_score": info.get("final_score", 0.0),
    }
    
    print(f"  NFE: {results[method]['nfe']}")
    print(f"  Verifier Score: {results[method]['verifier_score']:.4f}")
```

## 🔧 配置示例

### Local Search配置

```python
# Random Search
pipeline = create_pipeline(model, verifier, method="random")
samples, info = pipeline.sample(
    method="random",
    num_candidates=8,  # 候选轨迹数
)

# Zero-Order Search
pipeline = create_pipeline(model, verifier, method="zo")
samples, info = pipeline.sample(
    method="zo",
    num_iterations=4,   # 迭代次数
    num_neighbors=8,    # 每次迭代的邻居数
    noise_scale=0.1,    # 噪声缩放
)
```

### Global Search配置

```python
# 固定分配策略
pipeline = create_pipeline(
    model, verifier,
    method="global",
    global_policy_type="fixed",
    total_nfe_budget=200,
    early_ratio=0.6,           # 前期budget比例
    early_steps_ratio=0.33,    # 前期步数比例
    search_mode_early="heavy_local",
    search_mode_late="light_local",
)

# 自适应阈值策略
pipeline = create_pipeline(
    model, verifier,
    method="global",
    global_policy_type="adaptive",
    total_nfe_budget=200,
    threshold=0.0,      # 改善阈值
    base_budget=10,     # 基础budget
    max_budget=50,      # 最大budget
)

# 多阶段策略
pipeline = create_pipeline(
    model, verifier,
    method="global",
    global_policy_type="multi_stage",
    total_nfe_budget=200,
    early_ratio=0.5,    # 前期budget
    mid_ratio=0.3,      # 中期budget
    late_ratio=0.2,     # 后期budget
)
```

## 📊 结果信息

每次采样返回的`info`字典包含：

### Local Search结果

```python
{
    "method": "random_search",
    "num_candidates": 4,
    "nfe": 200,
    "verifier_scores": [0.5, 0.6, 0.55, 0.65],
    "best_idx": 3,
    "best_score": 0.65,
}
```

### Global Search结果

```python
{
    "method": "global_search",
    "policy": "FixedBudgetPolicy",
    "nfe": 180,
    "actions": [
        "Action(mode=heavy_local, budget=12, primitive=random)",
        "Action(mode=light_local, budget=8, primitive=random)",
        ...
    ],
    "rewards": [0.05, 0.03, -0.01, ...],
    "final_score": 0.72,
}
```

## 🔍 扩展指南

### 添加新的Local Search方法

1. 在`src/search/local_search.py`中创建新类
2. 继承`BaseSearch`类
3. 实现`search()`方法

```python
class YourLocalSearch(BaseSearch):
    def search(self, initial_noise, batch_size, num_steps, nfe_counter, **kwargs):
        # 实现你的搜索逻辑
        ...
        return samples, info
```

### 添加新的Global Search策略

1. 在`src/search/global_search.py`中创建新类
2. 继承`GlobalSearchPolicy`类
3. 实现`decide_action()`方法

```python
class YourPolicy(GlobalSearchPolicy):
    def decide_action(self, state, num_steps):
        # 实现你的策略逻辑
        ...
        return Action(...)
```

## ⚠️ 注意事项

1. **NFE对齐**：所有方法都应使用`NFECounter`确保公平对比
2. **Verifier评估**：某些Local Search方法需要在中间步骤评估，可能影响性能
3. **内存管理**：Random Search和ZO会采样多个轨迹，注意内存使用
4. **策略参数**：Global Search的策略参数需要根据具体任务调优

## 🎯 下一步

1. 实现具体的模型加载器（DDPM/EDM等）
2. 实现Verifier（分类器）
3. 运行baseline实验对比不同方法
4. 根据结果调优Global Search策略参数

参考 `scripts/run_pipeline.py` 查看完整的实验脚本示例。


