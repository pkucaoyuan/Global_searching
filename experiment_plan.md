# Global Search 与 Local Search 实验计划

## 项目目标

验证在相同total NFE/GFLOPs下：
- 纯增加采样步数的提升有限
- 将部分NFE用于search/local search on noise/trajectory可以获得更优的FID/IS
- 验证compute-aware调度策略的有效性

---

## Phase 0: 环境搭建与代码库准备（第1-2天）

### 0.1 代码库获取与整理

**目标代码库：**

1. **基础框架类**
   - `sayakpaul/tt-scale-flux` - Random Search + ZO + verifier接口参考
   - `XiangchengZhang/Diffusion-inference-scaling` - ImageNet实验脚本（BFS/DFS/MCMC）

2. **Local Search方法**
   - `harveymannering/NoiseLevelGuidance` - NLG实现
   - `rvignav/diffusion-tts` - Noise Trajectory Search (ε-greedy)

3. **参考实现**
   - `zacharyhorvitz/Fk-Diffusion-Steering` - 粒子系统设计参考
   - `masa-ue/SVDD` - value-based方法参考

**任务清单：**
- [ ] Clone所有相关代码库
- [ ] 分析代码结构，提取关键模块（verifier、search primitives、evaluation）
- [ ] 整理统一的接口规范（方便后续替换不同方法）

### 0.2 模型与数据集准备

**CIFAR-10:**
- [ ] 选择预训练模型（OpenAI DDPM / EDM / score-sde任选一）
- [ ] 下载模型checkpoint
- [ ] 准备评估脚本（FID_50k + IS）

**ImageNet-64/256:**
- [ ] 准备ImageNet-64数据集
- [ ] 获取预训练模型（SiT-B/L/XL或FLUX相关）
- [ ] 配置evaluation pipeline

### 0.3 基础Infrastructure

- [ ] 设置统一的实验配置管理（config文件）
- [ ] 建立实验日志和结果存储系统
- [ ] 编写NFE计数工具（确保对比公平）

---

## Phase 1: CIFAR-10基础实验（第1-2周）

### 1.1 Pure Sampling Baseline（3-4天）

**目标：** 建立baseline曲线，验证"单纯加步数提升有限"

**实验设计：**

```
固定初始噪声N=1，只改变采样步数：
- CIFAR-10: [25, 50, 100, 200] 步
- 评估指标：FID_50k + IS
```

**任务清单：**
- [ ] 实现标准DDPM/EDM采样流程
- [ ] 实现FID和IS评估脚本
- [ ] 运行不同步数实验，绘制scaling曲线
- [ ] **预期结果：** 曲线会在某个点后变平，作为后续对比基准

**输出：**
- Baseline FID/IS vs NFE曲线图
- 实验日志和详细数据

### 1.2 Random Search Baseline（2-3天）

**目标：** 实现最简单的全局搜索方法

**实验设计：**

```
固定每条轨迹S=50步
采样N个初始噪声，独立采样
最终选择verifier score最高的轨迹
NFE ≈ N × S

Verifier选择：
- CIFAR-10: 预训练classifier的log p(class|x)作为reward
```

**任务清单：**
- [ ] 实现verifier（classifier-based）
- [ ] 实现Random Search：多噪声并行采样 + 选择最优
- [ ] 实验N=[4, 8, 16]的Random Search
- [ ] 与pure sampling baseline对比（NFE对齐）

**输出：**
- Random Search vs Pure Sampling对比图
- NFE效率分析（FID/IS per NFE）

### 1.3 NLG (Noise-Level Guidance) Baseline（2-3天）

**目标：** 实现轻量级local search baseline

**实验设计：**

```
在sampling开始前：
- 在t=T处对初始噪声进行K步NLG refine
- 然后照常进行S步采样
- NFE ≈ K (preprocessing) + S (sampling)

实验K=[5, 10, 20]
```

**任务清单：**
- [ ] 从官方repo获取NLG代码
- [ ] 适配到CIFAR-10模型
- [ ] 实现NLG + standard sampling pipeline
- [ ] 运行不同K值的实验

**输出：**
- NLG vs Pure Sampling对比
- 不同K值的trade-off分析

### 1.4 Zero-Order Search (ZO-N) Baseline（3-4天）

**目标：** 实现pivot-based迭代搜索

**实验设计：**

```
从一个pivot噪声出发
在邻域采样多个噪声
选择verifier score最好的作为新pivot
迭代N次

实验N=[2, 4, 8]
```

**任务清单：**
- [ ] 参考Ma论文实现ZO算法
- [ ] 适配到CIFAR-10
- [ ] 注意：ZO会消耗较多额外NFEs（每次pivot需要完整轨迹）
- [ ] 与Random Search和Pure Sampling对比

**输出：**
- ZO-N vs 其他方法对比图
- NFE效率分析

### 1.5 Phase 1总结与对比（2天）

**任务清单：**
- [ ] 整理所有baseline结果
- [ ] 绘制统一的对比图（FID/IS vs NFE）
- [ ] 分析不同方法的NFE效率
- [ ] 验证假设：search方法确实比纯加步数更有效

**输出：**
- CIFAR-10完整baseline报告
- 决定哪些方法值得迁移到ImageNet

---

## Phase 2: ImageNet-64迁移与扩展（第3-4周）

### 2.1 框架迁移（3-4天）

**任务清单：**
- [ ] 将CIFAR-10实验框架迁移到ImageNet-64
- [ ] 使用`Diffusion-inference-scaling/imagenet`或`tt-scale-flux`作为起点
- [ ] 确保verifier、evaluation、NFE计数等工具正常工作
- [ ] 复现ImageNet-64的pure sampling baseline

### 2.2 复现现有方法（5-6天）

**目标：** 在ImageNet上复现Random和ZO方法，验证infra正确性

**任务清单：**
- [ ] 在ImageNet-64上复现Random Search
- [ ] 在ImageNet-64上复现ZO方法
- [ ] 对比结果与论文/官方实现是否一致
- [ ] 如有差异，调试并解决

### 2.3 NLG + Random组合实验（2-3天）

**实验设计：**

```
先用NLG refine初始噪声（K步）
然后进行Random Search（N个噪声，S步采样）
NFE ≈ K + N × S
```

**任务清单：**
- [ ] 实现NLG + Random组合
- [ ] 实验不同K和N的组合
- [ ] 分析是否比单独使用NLG或Random更好

**输出：**
- 组合方法实验结果
- 方法组合效果分析

---

## Phase 3: Compute-Aware调度策略实现（第5-6周）

### 3.1 方法设计与接口定义（2-3天）

**核心思想：**
- **Action不是噪声，而是search策略选择**
- `action_t = (search_mode, budget, primitive_type)`
  - `search_mode ∈ {none, light_local, heavy_local, global_resample}`
  - `budget`: 在该时刻分配的额外NFE
  - `primitive_type`: 使用的search算法（Random, ZO, BFS, etc.）

**任务清单：**
- [ ] 定义清晰的MDP结构（基于手写笔记）
  - State: `(xt, t, prompt, history/score)`
  - Action: `search_mode + budget allocation`
  - Reward: `Δt = verifier(xt-1) - verifier(xt) - λ·computation`
- [ ] 设计统一接口，可以插入不同的调度策略
- [ ] 实现状态提取和奖励计算函数

### 3.2 简单Heuristic实现（初期不需要RL）（4-5天）

**策略1: 固定分配策略**
```
根据不同step的重要性固定分配budget
- T2I任务：前1/3步分配更多NFE（如60%）
- 后2/3步分配较少NFE（如40%）
```

**策略2: 自适应阈值策略**
```
根据verifier(xt)的改善情况动态调整：
- 如果Δt < threshold: 增加search budget（当前状态不好）
- 如果Δt > threshold: 减少search budget（已经改善）
```

**策略3: 多阶段策略**
```
- 前期（t接近T）: 使用heavy search（ZO, 更多samples）
- 中期（t中等）: 使用light search（Random, 较少samples）
- 后期（t接近0）: 使用no search（直接采样）
```

**任务清单：**
- [ ] 实现上述3个简单策略
- [ ] 在CIFAR-10上测试每个策略
- [ ] 对比不同策略的效果
- [ ] 分析什么情况下哪种策略更好

### 3.3 与Baseline对比实验（3-4天）

**对比维度：**
- 相同NFE budget下，compute-aware策略 vs Random/ZO/NLG
- 不同NFE regime下的表现（low/medium/high compute）
- 不同数据集上的泛化性（CIFAR-10 vs ImageNet-64）

**任务清单：**
- [ ] 设计NFE对齐的实验（确保公平对比）
- [ ] 运行完整对比实验
- [ ] 绘制详细的对比图表
- [ ] 统计分析显著性差异

### 3.4 Ablation Study（2-3天）

**研究问题：**
- 不同budget分配比例的影响
- 不同search primitive选择的影响
- Verifier的选择对策略效果的影响
- λ参数（computation cost权重）的敏感性

**任务清单：**
- [ ] 设计ablation实验
- [ ] 运行并分析结果
- [ ] 总结关键设计选择的重要性

---

## Phase 4: 结果整理与论文准备（第7-8周）

### 4.1 结果可视化（2-3天）

- [ ] 绘制所有对比曲线（FID/IS vs NFE）
- [ ] 制作方法对比表格
- [ ] 准备可视化图表（trajectory示例、budget分配可视化等）

### 4.2 实验报告撰写（3-4天）

**报告结构：**
1. Introduction: 问题定义、动机
2. Background: Local search vs Global search框架
3. Method: Compute-aware调度策略
4. Experiments:
   - CIFAR-10结果
   - ImageNet-64结果
   - Ablation studies
5. Discussion: 何时有效、局限性、未来工作

### 4.3 代码整理（1-2天）

- [ ] 整理实验代码，添加注释
- [ ] 创建README和使用说明
- [ ] 确保代码可复现

---

## 关键检查点

### 第1周结束
- ✅ CIFAR-10 pure sampling baseline完成
- ✅ Random Search和NLG至少有一个跑通

### 第2周结束
- ✅ CIFAR-10所有baseline完成
- ✅ 验证search方法确实比纯加步数更有效

### 第4周结束
- ✅ ImageNet-64迁移完成
- ✅ 至少复现了Random和ZO方法

### 第6周结束
- ✅ Compute-aware策略实现完成
- ✅ 初步对比实验有积极结果

---

## 风险与应对

1. **代码库兼容性问题**
   - 风险：不同代码库依赖冲突
   - 应对：使用虚拟环境隔离，必要时重写关键模块

2. **计算资源限制**
   - 风险：ImageNet实验需要大量GPU时间
   - 应对：先在CIFAR-10验证思路，ImageNet只做关键实验

3. **方法效果不明显**
   - 风险：compute-aware策略没有明显优势
   - 应对：深入分析失败原因，调整策略设计，或转向其他方向

4. **Baseline复现困难**
   - 风险：无法复现论文结果
   - 应对：联系作者，检查实现细节，必要时简化baseline

---

## 实验记录建议

建议为每个实验创建单独的记录文件，包含：
- 实验配置（模型、数据集、参数）
- 运行命令
- 结果数据（FID/IS/NFE）
- 观察和分析
- 下一步计划

这样可以方便追踪实验历史和调试问题。


