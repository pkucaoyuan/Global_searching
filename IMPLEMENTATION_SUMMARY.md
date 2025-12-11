# Pipeline实现总结

## ✅ 已完成的功能

### 1. Local Search实现 ✓

**文件**: `src/search/local_search.py`

实现了4种Local Search方法：

- **NoSearch**: 标准采样baseline
- **RandomSearch**: 采样多个完整轨迹，选择verifier score最高的
- **LocalNoiseSearch**: 在每个时间步采样多个候选xt-1（单步搜索）
- **ZeroOrderSearch**: Pivot-based迭代搜索，在初始噪声邻域搜索

**特点**：
- 所有方法继承`BaseSearch`统一接口
- 统一的NFE计数
- 返回样本和详细信息字典

### 2. Global Search框架实现 ✓

**文件**: `src/search/global_search.py`

实现了完整的Global Search框架：

**核心类**：
- `State`: MDP状态（xt, t, prompt, history/score）
- `Action`: 动作（search_mode, budget, primitive_type）
- `GlobalSearchPolicy`: 策略基类
- `GlobalSearch`: 主执行类

**策略实现**：
1. **FixedBudgetPolicy**: 固定分配策略
   - 根据步数重要性固定分配budget
   - 可配置前期/后期比例

2. **AdaptiveThresholdPolicy**: 自适应阈值策略
   - 根据verifier改善情况动态调整
   - 改善不足时增加budget

3. **MultiStagePolicy**: 多阶段策略
   - 前期heavy search
   - 中期light search
   - 后期no search

**特点**：
- 基于MDP建模
- Reward = Δt - λ·computation
- 灵活的策略接口，易于扩展

### 3. Pipeline整合 ✓

**文件**: `src/pipeline/sampling_pipeline.py`

实现了完整的采样Pipeline：

- `SamplingPipeline`: 主Pipeline类
  - 支持Local Search模式
  - 支持Global Search模式
  - 统一的接口

- `create_pipeline()`: 工厂函数
  - 便捷创建Pipeline
  - 支持配置不同策略

### 4. 实验脚本 ✓

**文件**: `scripts/run_pipeline.py`

提供了完整的实验脚本框架：
- 支持不同方法切换
- 批量生成样本
- 结果保存和评估

## 📁 文件结构

```
src/
├── search/
│   ├── base_search.py          # Search基类
│   ├── local_search.py          # Local Search实现 ✓
│   └── global_search.py         # Global Search框架 ✓
│
├── pipeline/
│   ├── __init__.py
│   └── sampling_pipeline.py     # Pipeline整合 ✓
│
├── models/
│   └── base_model.py            # 模型接口（需实现具体模型）
│
├── verifiers/
│   └── base_verifier.py         # Verifier接口（需实现具体verifier）
│
└── utils/
    ├── nfe_counter.py           # NFE计数工具 ✓
    └── config.py                # 配置管理 ✓
```

## 🔄 Pipeline工作流程

### Local Search流程

```
初始化 → 采样初始噪声 → 逐步采样 → 返回结果
```

对于Random/ZO等方法：
```
初始化 → 采样多个候选 → 并行采样 → Verifier评估 → 选择最优 → 返回结果
```

### Global Search流程

```
初始化State → 
循环每个时间步:
  1. Global Policy决定Action (search_mode, budget, primitive)
  2. 根据Action选择Local Search方法
  3. 执行Local Search（单步或完整流程）
  4. 计算Reward = Δt - λ·computation
  5. 更新State (xt, history_scores)
→ 返回最终样本和完整信息
```

## 🎯 核心设计理念

### 两层架构

1. **Local Search（低层）**：
   - 负责单个时间步或完整轨迹的搜索
   - 输入：xt, t, verifier
   - 输出：xt-1 或完整样本

2. **Global Search（高层）**：
   - 负责整个轨迹的调度
   - 输入：State (xt, t, history)
   - 输出：Action (search_mode, budget, primitive)
   - 控制：何时使用哪种Local Search，分配多少算力

### MDP建模

- **State**: `(xt, t, prompt, history/score)`
- **Action**: `(search_mode, budget, primitive_type)`
  - search_mode: none, light_local, heavy_local, global_resample
  - budget: 分配的NFE预算
  - primitive_type: random, zo, local, etc.
- **Reward**: `Δt = verifier(xt-1) - verifier(xt) - λ·computation`

## 📊 使用示例

### 快速开始

```python
from src.pipeline.sampling_pipeline import create_pipeline

# 1. Local Search示例
pipeline = create_pipeline(model, verifier, method="random")
samples, info = pipeline.sample(method="random", batch_size=32, num_steps=50)

# 2. Global Search示例
pipeline = create_pipeline(
    model, verifier,
    method="global",
    global_policy_type="fixed",
    total_nfe_budget=200,
)
samples, info = pipeline.sample(method="global", batch_size=32, num_steps=50)
```

## 🔧 需要完善的部分

### 1. 模型实现
- [ ] 实现具体的DDPM模型加载器
- [ ] 实现EDM模型加载器
- [ ] 适配不同模型的采样接口

### 2. Verifier实现
- [ ] 完善ClassifierVerifier的实现
- [ ] 实现CIFAR-10分类器加载
- [ ] 实现ImageNet分类器加载

### 3. Local Search优化
- [ ] 优化LocalNoiseSearch的单步搜索效率
- [ ] 实现NLG（Noise-Level Guidance）
- [ ] 实现更高效的搜索算法

### 4. Global Search优化
- [ ] 优化Global Search的单步Local Search调用
- [ ] 实现更精确的reward计算
- [ ] 实现RL-based策略（未来扩展）

### 5. 评估工具
- [ ] 完善FID/IS计算
- [ ] 实现批量评估
- [ ] 实现结果可视化

## 📝 下一步行动

1. **实现模型加载**（最优先）
   - 选择一种模型（如DDPM）
   - 实现模型加载和采样接口
   - 测试基本采样功能

2. **实现Verifier**
   - 实现分类器加载
   - 测试verifier评分功能

3. **运行第一个实验**
   - 使用NoSearch作为baseline
   - 对比Random Search
   - 验证pipeline正确性

4. **逐步完善**
   - 实现更多Local Search方法
   - 调优Global Search策略
   - 运行完整对比实验

## 🎉 成就

✅ 完整的两层级Search架构  
✅ 统一的接口设计  
✅ 灵活的扩展能力  
✅ 清晰的代码结构  
✅ 详细的文档说明  

整个Pipeline框架已经搭建完成，可以开始实现具体的模型和verifier了！


