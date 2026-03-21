# Claude Code 项目规范 - Vidur智能调度器项目

## 🎯 项目定位

**项目名称**: Vidur智能调度器 - 基于强化学习的 LLM 推理请求调度系统

**研究领域**:
- LLM 推理优化 (Continuous Batching, PagedAttention)
- 强化学习 (PPO with RUDDER Credit Assignment)
- 调度算法 (Global Scheduler + Replica Scheduler)

**核心目标**:
1. ✅ 超越 SARATHI (FIFO) 和 SJF 基准调度器 - **已实现 (Enhancement 18)**
2. 实现 RUDDER 信用分配解决延迟奖励问题
3. 支持 MMPP 突发流量模式训练

**🏆 最新成果 (2026-02-01) - 统计验证通过**:
- Thompson Sampling 在 Poisson QPS 9-10 下同时击败 SARATHI 和 SJF
- Mean latency: -2.0% to -2.4% vs SARATHI, -2.0% to -2.4% vs SJF
- P99 latency: -1.5% to -3.6% vs SARATHI, -1.2% to -3.1% vs SJF
- **多种子验证**: 3个种子全部通过，100%胜率
- **多Arrival Pattern验证**: 6种模式全部通过，100%胜率 (Poisson + Gamma CV 0.5-3.0)
- 详见: `docs/progress/2026_02_01_enhancement18_thompson_sampling.md`

**🔬 PAN Scheduler Aging 策略研究 (2026-02-02 更新)**:
- **50K 规模验证**: SJF P99 惩罚从 +11.5% 降至 +5%，规模效应显著
- **10K 结论**: ASJF(1000) 最优 (Mean -6%, P99 +2.8%)
- **50K 结论**: **SJF 成为最优** (Mean -9.6%, P99 +5%)
- **Max 延迟保护**: ASJF(1000) 仍优于 SJF (33s vs 47s)
- 详见: `docs/progress/2026-02-02_pan_asjf_50k_validation.md`

**🔬 Token 分布对调度策略影响研究 (2026-02-02)**:
- **核心发现**: ZIPF 分布是 SJF 最差情况，Bimodal 分布反而帮助 SJF
- **Bimodal (少中间)**: SJF P99 +1.4% ~ +5.0% (好)
- **MediumHeavy (多中间)**: SJF P99 +9.1% ~ +9.5% (差)
- **ZIPF (连续分布)**: SJF P99 +11.5% (最差)
- **ASJF 稳定性**: 所有分布下 P99 < +3%，甚至 -0.24%
- 详见: `docs/progress/2026-02-02_pan_bimodal_distribution_analysis.md`

**📊 4-Replica Scheduler Tradeoff Analysis (2026-02-02)**:
- **配置**: 4 Replica + SQF Global + QPS 20-40
- **SJF 饥饿问题**: 高负载(QPS>30) P99 退化 +100~200%，Max 延迟达 170s
- **NAILONG 优势**: 效率-公平性比值是 SJF 的 2-5 倍
- **结论**: 生产环境推荐 NAILONG Thompson Sampling
- 详见: `docs/progress/2026-02-02_4replica_tradeoff_analysis.md`
- 调度器对比文档: `docs/modules/schedulers/replica/scheduler_comparison_sarathi_sjf_nailong.md`

**🔬 Social Cost Global Scheduler (2026-02-11)**:
- **双预测器路由**: `Score_r = SelfPred(r) + beta * ExtPred(r)`，选择 argmin
- **统计验证通过 (20K requests, 5 seeds, p < 0.05)**: QPS 5-6 全部显著
- **最优调参**: QPS=5: P99 -1.31%, QPS=6: P99 -0.43% (Safety clamp mlr=1.5)
- 详见: `docs/progress/2026_02_11_social_cost_scheduler.md`

**🏆 SC + NAILONG 组合验证通过 (多规模+多副本, 2026-02-13)**:
- **组合效果**: 全局路由(SC) + 自适应批处理(NAILONG TS) 复合增益
- **4R (50K)**: Mean -3.56% (p<0.0001), P99 -2.91% (p<0.0001)
- **8R (20K)**: Mean -3.19% (p<0.0001), **P99 -3.67% (p<0.0001)**
- **2R (20K)**: Mean -2.60%, P99 -2.24% (combo有效，但SC增量不显著)
- **SC增量随副本数增强**: 2R无效 → 4R P99 -0.43%(p=0.025) → **8R P99 -1.33%(p=0.0008)**
- **NAILONG 尺度不变**: ~2.4% Mean/P99 无论副本数 (2R/4R/8R)
- **SC最低有效规模**: 需要4+副本，2副本时SQF已接近最优
- **验证规模**: 50K(4R) + 20K(2R/8R), 5 seeds, combo 100% 胜率
- 详见: `docs/progress/2026_02_12_sc_nailong_combination.md`

**项目计划文档**:
| 文档 | 内容 |
|------|------|
| `docs/plan/beating_sqf_updated_plan.md` | 超越SQF实施计划（最新） |
| `docs/guides/` | 用户指南和操作说明 |
| `docs/modules/` | 模块接口文档 |
| `docs/modules/evolution/` | 版本演进历史 |
| `docs/experiments/` | 实验记录和分析 |
| `docs/progress/` | 进度报告和总结 |
| `docs/daily_logs/YYYY-MM/` | 每日工作日志 |
| `docs/research/` | 研究理论文档 |
| `docs/research/problems/` | **问题分析与Deep Research** |

**当前技术架构**:
```
vidur/
├── scheduler/
│   ├── global_scheduler/
│   │   ├── ppo_scheduler_modular.py    # PPO 调度器主逻辑
│   │   ├── shortest_queue_global_scheduler.py
│   │   ├── sqf_lrtf_global_scheduler.py
│   │   ├── meta_global_scheduler.py    # Meta 调度器
│   │   ├── ltr_global_scheduler.py     # LTR 调度器
│   │   ├── thompson_sampling_global_scheduler.py  # Thompson Sampling 调度器
│   │   ├── learned_speed_sqf_global_scheduler.py  # Learned Speed-Aware SQF
│   │   ├── social_cost_global_scheduler.py    # 🆕 Social Cost (Self+Ext预测器)
│   │   └── forced_action_global_scheduler.py  # 🆕 反事实数据收集
│   └── replica_scheduler/
│       ├── sarathi_scheduler.py        # Chunked Prefill 调度 (FIFO baseline)
│       ├── heuristic_sarathi_scheduler.py  # SJF baseline
│       ├── pan_scheduler.py            # PAN Scheduler (FIFO/SJF/ASJF/WSJF/LJF)
│       ├── nailong_scheduler.py        # 🆕 NAILONG + Thompson Sampling (Enhancement 18)
│       └── meta_replica_scheduler.py   # Meta Replica 调度器
├── config/config.py                    # 配置参数定义
└── metrics/metrics_store.py            # 指标收集

src/core/
├── algorithms/ppo_trainer.py           # PPO 训练器
├── credit_assignment/rudder.py         # RUDDER 信用分配
├── models/
│   ├── shared_actor_critic.py          # Actor-Critic 网络
│   ├── request_ltr_model.py            # LTR 排序模型 (Enhancement 17)
│   ├── social_cost_model.py            # 🆕 双 MLP 预测器 (Self+Ext)
│   └── enriched_minimal_state_builder.py  # 13维特征提取 (含token分布)
├── rewards/vidur_enhanced_reward_calculator.py
└── utils/infrastructure/memory_manager.py
```

**🎯 当前最佳配置 (Enhancement 18 - Thompson Sampling)**:
```yaml
# Thompson Sampling 在 Poisson/Gamma 下击败 SARATHI 和 SJF
replica_scheduler_config_type: nailong
nailong_scheduler_config_learning_mode: THOMPSON_SAMPLING
nailong_scheduler_config_training_mode: true
nailong_scheduler_config_reward_mode: CONTRASTIVE
# 推荐 QPS 范围: 9-10 (100% double win rate)
# 验证通过的 Arrival Patterns: Poisson, Gamma (CV 0.5-3.0)
# 结果: Mean -2.0%, P99 -2.5% vs both baselines (average across patterns)
```

**🎯 Social Cost 最佳配置 (2026-02-11)**:
```yaml
# Social Cost 全局调度器 - P99 优化, 统计验证通过
global_scheduler_config_type: social_cost
social_cost_global_scheduler_config_self_checkpoint: outputs/checkpoints/social_cost/self_predictor.pt
social_cost_global_scheduler_config_externality_checkpoint: outputs/checkpoints/social_cost/externality_predictor.pt
social_cost_global_scheduler_config_warmup_steps: 100
# QPS=5 最优: β=5, temp=0.1 → P99 -1.31% (p=0.002), 5/5 wins
social_cost_global_scheduler_config_beta: 5.0
social_cost_global_scheduler_config_score_temperature: 0.1
# QPS=6 需要安全阀: β=3, mlr=1.5 → P99 -0.43%, 消除灾难性路由
# social_cost_global_scheduler_config_beta: 3.0
# social_cost_global_scheduler_config_max_load_ratio: 1.5
```

**🎯 SC+NAILONG 组合最佳配置 (多副本验证, 2026-02-13)**:
```yaml
# SC+NAILONG 组合 (per-replica ~4.5 QPS)
# 4R(50K): Mean -3.56%, P99 -2.91% | 8R(20K): Mean -3.19%, P99 -3.67%
# SC 增量随副本数增强: 4R P99 -0.43% → 8R P99 -1.33% (p=0.0008)
# 推荐: 4+ 副本 (2副本时SC无增量)
global_scheduler_config_type: social_cost
social_cost_global_scheduler_config_beta: 3.0
social_cost_global_scheduler_config_max_load_ratio: 1.5
social_cost_global_scheduler_config_warmup_steps: 100
replica_scheduler_config_type: nailong
nailong_scheduler_config_learning_mode: THOMPSON_SAMPLING
nailong_scheduler_config_training_mode: true
nailong_scheduler_config_reward_mode: CONTRASTIVE
nailong_scheduler_config_warmup_requests: 200
nailong_scheduler_config_chunk_size: 512
# NAILONG 尺度不变 (~2.4% 无论副本数), SC需4+副本才有增量
```

---

## 📁 目录结构规范

**标准目录结构**:
```
Vidur/
├── src/                  # 核心代码（模块化设计）
├── vidur/                # Vidur仿真器代码
├── configs/              # 配置文件
├── scripts/              # 运行脚本（按工作流组织）
│   ├── training/         # 训练脚本（ppo/, bc/, pretrain/）
│   ├── evaluation/       # 模型对比脚本
│   ├── monitoring/       # 训练监控脚本
│   ├── data_collection/  # 数据收集脚本
│   ├── experiments/      # 快速测试脚本
│   └── utils/            # 验证工具脚本
├── outputs/              # 实验输出（git忽略）
│   ├── experiments/      # 结构化实验（按日期分组）
│   ├── runs/             # TensorBoard日志
│   ├── checkpoints/      # 模型检查点
│   │   └── archive/      # 备份归档
│   └── normalizer_stats/ # Normalizer统计
├── docs/                 # 文档（3-4级分类，161文件）
│   ├── README.md               # 导航索引
│   ├── PROJECT_TODO.md         # 项目TODO
│   ├── plan/             # 项目计划和路线图（9文件）
│   │   └── replica_ppo_training/  # 子项目计划
│   ├── guides/           # 用户指南（29文件，6类别）
│   │   ├── getting_started/    # 入门配置
│   │   ├── training/           # 训练工作流
│   │   ├── rewards/            # 奖励设计
│   │   ├── schedulers/         # 调度器配置
│   │   ├── optimization/       # 性能优化
│   │   └── infrastructure/     # 基础设施
│   ├── modules/          # 模块文档（57文件，6类别）
│   │   ├── schedulers/         # 调度器模块
│   │   │   ├── global/         # 全局调度器
│   │   │   └── replica/        # 副本调度器
│   │   ├── training/           # 训练组件
│   │   ├── infrastructure/     # 基础设施
│   │   ├── data/               # 数据生成
│   │   ├── specialized/        # 专用组件
│   │   └── evolution/          # 演进历史
│   ├── research/         # 研究文档（18文件，5类别）
│   │   ├── theory/             # 理论基础
│   │   ├── methods/            # 方法论
│   │   ├── debugging/          # 调试指南
│   │   ├── papers/             # 学术论文（PDF）
│   │   └── problems/           # 问题分析（Deep Research）
│   ├── experiments/      # 实验记录（5文件）
│   ├── progress/         # 进度报告（10文件）
│   ├── deprecated/       # 废弃功能（16文件）
│   ├── optimization/     # 优化分析（8文件）
│   ├── daily_logs/       # 工作日志（YYYY-MM/）
│   └── templates/        # 文档模板（3文件）
├── tests/                # 测试脚本
├── notebooks/            # Jupyter分析
└── tmp/                  # 临时文件（git忽略）
```

**目录功能说明**:
- `src/`: 核心RL训练代码，模块化设计
- `vidur/`: Vidur仿真器源码（调度器、配置、指标）
- `scripts/`: 按工作流分类的脚本（见下方"脚本目录索引"）
- `outputs/experiments/`: 结构化实验输出，包含完整元数据
- `outputs/checkpoints/archive/`: 统一管理的备份检查点
- `docs/`: 按用途分类的文档（见docs/README.md）

**临时文件管理**:
- 临时测试文件必须在 `tmp/` 目录
- 文件生命周期 ≤ 1天，使用后立即删除
- 正式代码不得import tmp/中的文件
- tmp/目录已在.gitignore中

---

## 📁 文档目录结构

### 三层文档架构

**Level 1 - 主分类** (10个顶级目录):
```
docs/
├── plan/           # 项目级规划
├── guides/         # 操作指南（面向用户）
├── modules/        # 接口文档（面向开发者）
├── research/       # 理论研究（面向研究者）
├── experiments/    # 实验记录
├── progress/       # 进度总结
├── deprecated/     # 历史归档
├── optimization/   # 优化分析
├── daily_logs/     # 工作日志
└── templates/      # 文档模板
```

**Level 2 - 子分类**:
- **guides/**: 6个子分类 (getting_started, training, rewards, schedulers, optimization, infrastructure)
- **modules/**: 6个子分类 (schedulers, training, infrastructure, data, specialized, evolution)
- **research/**: 4个子分类 (theory, methods, debugging, papers)

**Level 3 - 细分** (仅schedulers):
- `modules/schedulers/global/` - 全局调度器
- `modules/schedulers/replica/` - 副本调度器

### 快速导航

#### 我想开始使用系统
→ `docs/guides/getting_started/developer_onboarding.md`

#### 我想训练模型
→ `docs/guides/training/ppo_warm_start_guide.md`

#### 我想调整奖励函数
→ `docs/guides/rewards/reward_config_guide.md`

#### 我想理解调度器
→ `docs/modules/schedulers/global/base_global_scheduler.md`

#### 我想调试问题
→ `docs/research/debugging/rl_training_debugging_guide.md`

### 查找文档命令

```bash
# 按主题查找
find docs/guides -name "*reward*"          # 奖励相关
find docs/modules/training -name "*ppo*"   # PPO模块

# 按类型浏览
ls docs/guides/training/                   # 训练指南
ls docs/modules/schedulers/global/         # 全局调度器

# 搜索内容
grep -r "RUDDER" docs/research/            # 查找RUDDER
```

### 文档命名规范

**禁止**:
- ❌ 中文文件名 (使用英文)
- ❌ enhanced/improved前缀
- ❌ v2/new后缀

**推荐**:
- ✅ `ppo_warm_start_guide.md` - 清晰功能描述
- ✅ `scheduler_comparison_guide.md` - 用途明确

---

## 🚨 核心编程规范

### 八荣八耻编程原则

1. **以暗猜接口为耻，以认真查阅为荣** - 禁止臆测API行为，必须查阅文档确认
2. **以模糊执行为耻，以寻求确认为荣** - 不确定的实现必须先向用户确认
3. **以默认忽略为耻，以主动报告为荣** - 遇到异常、错误必须主动报告
4. **以隐式假设为耻，以显式验证为荣** - 所有假设必须通过代码验证
5. **以随意修改为耻，以谨慎调试为荣** - 修改前必须理解原理
6. **以表面应付为耻，以深入理解为荣** - 解决问题必须找到根本原因
7. **以复制粘贴为耻，以原创思考为荣** - 理解每行代码含义
8. **以孤立开发为耻，以协同沟通为荣** - 主动汇报进度和问题

### 文件命名规范

**禁用前缀后缀列表**：
- ❌ `enhanced_*` / `*_enhanced`
- ❌ `integrated_*` / `*_integrated`
- ❌ `cleaned_*` / `*_clean`
- ❌ `improved_*` / `*_improved`
- ❌ `*_v2` / `*_new` / `*_old` / `*_temp`
- ❌ **中文字符** - 所有文件名必须使用英文

**正确命名原则**：
- ✅ **功能导向**: `reward_calculator.py`, `ppo_trainer.py`
- ✅ **模块化**: `scheduler/`, `metrics/`, `config/`
- ✅ **简洁明确**: 使用下划线分隔，全小写，仅英文字符
- ✅ **描述性英文**: `rl_training_debugging_guide.md` 而非 `强化学习训练崩溃调试指南.md`

**中文文件名转换示例**:
| 禁止 ❌ | 正确 ✅ |
|---------|---------|
| `PPO推理模式贪婪策略说明.md` | `ppo_inference_greedy_policy.md` |
| `强化学习收敛问题与差分奖励.md` | `rl_convergence_differential_reward.md` |
| `在线LOR专家指导说明.md` | `online_expert_guidance.md` |

### 错误处理规范

```python
# ❌ 禁止fallback模式
try:
    result = complex_operation()
except Exception:
    result = fallback_operation()  # 禁止！

# ✅ 让错误自然抛出
result = complex_operation()  # 便于从本质上解决问题
```

**核心要求**：
- 🔥 **禁止静默捕获异常** - 让错误traceback显示
- 🔥 **禁止fallback方案** - 缺少属性直接报错

### 脚本组织规范

**核心原则**:
- **按工作流分类**: 脚本按用途放入对应子目录
- **简单脚本**: 直接在scripts/子目录中实现，最多50行
- **复杂逻辑**: 必须分离到src/模块中，脚本仅做调用
- **禁止内嵌**: 严禁在脚本中写大段Python代码或函数
- **禁止重复**: 相同逻辑不得在多个脚本中重复实现

**目录结构** (详见scripts/README.md):
```
scripts/
├── training/         # 训练工作流
│   ├── ppo/         # PPO训练（25个脚本）
│   ├── bc/          # 行为克隆（5个脚本）
│   ├── pretrain/    # 预训练（2个脚本）
│   └── social_cost/ # 🆕 Social Cost预测器训练（3个脚本）
├── evaluation/      # 模型对比（20个脚本）
├── monitoring/      # 训练监控（2个脚本）
├── data_collection/ # 数据收集（7个脚本）
├── experiments/     # 快速测试（7个脚本）
└── utils/           # 验证工具（3个脚本）
```

**常用脚本快速索引**:
| 工作流 | 脚本路径 |
|--------|----------|
| PPO训练 | `scripts/training/ppo/train_ppo_shared.sh` |
| RUDDER训练 | `scripts/training/ppo/train_ppo_rudder_mmpp.sh` |
| BC训练 | `scripts/training/bc/train_bc_aac.sh` |
| 模型对比 | `scripts/evaluation/compare_with_latest_checkpoint.sh` |
| 训练监控 | `scripts/monitoring/monitor_training.sh` |
| **SC D1数据** | `scripts/data_collection/collect_self_latency_data.py` |
| **SC D2数据** | `scripts/data_collection/collect_externality_data.py` |
| **SC训练** | `scripts/training/social_cost/train_self_predictor.py` |
| **SC Beta搜索** | `scripts/evaluation/sweep_beta.py` |
| **SC最终评估** | `scripts/evaluation/final_social_cost_eval.py` |
| **SC优势确认** | `scripts/evaluation/confirm_advantage_zone.py` |
| **SC Tradeoff** | `scripts/evaluation/explore_tradeoff.py` |
| **SC+NAILONG探索** | `scripts/evaluation/compare_social_cost_nailong.py` |
| **SC+NAILONG验证** | `scripts/evaluation/validate_social_cost_nailong.py` |

---

## 🎮 强化学习代码规范

### PPO训练规范

**训练流程**:
```python
# Rollout收集: 每个episode完整收集后才更新
for step in range(rollout_steps):
    action, log_prob, value = policy(state)
    next_state, reward = env.step(action)
    buffer.add(state, action, reward, log_prob, value)

# PPO更新
advantages = compute_gae(rewards, values, gamma=0.99, lambda_=0.95)
for epoch in range(ppo_epochs):
    update_policy(buffer, clip_epsilon=0.2)
```

**关键超参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE lambda |
| clip_epsilon | 0.2 | PPO clip范围 |
| ppo_epochs | 4 | 每次更新的epoch数 |
| entropy_coef | 0.01 | 熵正则化系数 |

### RUDDER信用分配规范

**核心组件**:
- `EpisodeBuffer`: 存储完整episode用于LSTM训练
- `ReturnPredictor`: LSTM预测累积回报
- `redistribute_rewards()`: 奖励再分配

**配置参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| rudder_hidden_dim | 128 | LSTM隐藏层维度 |
| rudder_buffer_size | 1000 | Episode buffer最大容量 |
| rudder_update_interval | 100 | LSTM训练间隔（steps） |
| rudder_weight | 0.7 | 再分配奖励混合权重 |

**RUDDER理论** (NeurIPS 2019):
```
redistributed_reward(t) = G_hat(t) - G_hat(t-1)
final_reward = (1 - weight) * original + weight * redistributed
```

### 奖励设计规范

**奖励组件**:
```python
# Latency reward: 越低越好
latency_reward = -delta_latency / latency_scale

# Throughput reward: 越高越好
throughput_reward = delta_throughput / throughput_scale

# 总奖励
reward = latency_weight * latency_reward + throughput_weight * throughput_reward
```

**归一化**:
- 使用RunningNormalizer进行在线归一化
- Welford算法计算running mean/std
- clip_range默认±10

---

## 📋 模块文档管理规范

### 核心原则
- **变更可追溯**: 每个模块维护完整演进历史
- **废弃功能保留**: deprecated功能统一管理
- **版本记录标准化**: 使用统一格式

### 文档结构 (3-4级分类，161文件)

```
docs/
├── README.md                   # 导航索引
├── PROJECT_TODO.md             # 项目TODO
├── plan/                       # 项目计划和路线图 (9文件)
│   └── replica_ppo_training/   # 子项目计划
├── guides/                     # 用户指南 (29文件, 6类别)
│   ├── getting_started/        # 入门配置
│   ├── training/               # 训练工作流
│   ├── rewards/                # 奖励设计
│   ├── schedulers/             # 调度器配置
│   ├── optimization/           # 性能优化
│   └── infrastructure/         # 基础设施
├── modules/                    # 模块文档 (57文件, 6类别)
│   ├── schedulers/
│   │   ├── global/             # 全局调度器
│   │   └── replica/            # 副本调度器
│   ├── training/               # 训练组件
│   ├── infrastructure/         # 基础设施
│   ├── data/                   # 数据生成
│   ├── specialized/            # 专用组件
│   └── evolution/              # 演进历史
├── research/                   # 研究文档 (18文件, 5类别)
│   ├── theory/                 # 理论基础
│   ├── methods/                # 方法论
│   ├── debugging/              # 调试指南
│   ├── papers/                 # 学术论文 (PDF)
│   └── problems/               # 问题分析 (Deep Research)
├── experiments/                # 实验记录 (5文件)
├── progress/                   # 进度报告 (10文件)
├── deprecated/                 # 废弃功能 (16文件)
├── optimization/               # 优化分析 (8文件)
├── daily_logs/                 # 工作日志 (YYYY-MM/)
└── templates/                  # 文档模板 (3文件)
```

**文档导航**（详见 [docs/README.md](../docs/README.md)）:
- 快速入门 → `guides/getting_started/`
- 训练模型 → `guides/training/`
- 理解系统 → `plan/`, `research/theory/`, `modules/`
- 调试问题 → `research/debugging/rl_training_debugging_guide.md`
- 实现功能 → `modules/training/`, `modules/schedulers/`

### Research Problems 文档规范

**位置**: `docs/research/problems/NNN_problem_name.md`

**命名**: 三位数字编号 + 问题描述 (如 `001_ppo_training_pipeline_analysis.md`)

**内容结构**:
1. 系统概述 - 组件架构图
2. 组件详解 - 每个模块的输入/输出/配置
3. 数据流 - 训练过程的完整流程
4. 实验结果 - 关键指标和观察
5. 问题分析 - 根因假设
6. 待研究问题 - 供 Deep Research 的具体问题
7. 相关文件索引 - 代码和文档链接

**用途**:
- 记录训练过程中遇到的问题
- 为 Deep Research 提供上下文
- 追踪问题解决进度

### Python模块头部规范

**根据模块类型引用正确路径**:

```python
"""
[模块简短描述]

Documentation:
    Interface: docs/modules/[category]/[module_name].md
    Evolution: docs/modules/evolution/[module_name]_evolution.md
    User Guide: docs/guides/[category]/[related_guide].md

Key Features:
    - Feature 1: Brief description
    - Feature 2: Brief description

Example:
    >>> from module import MainClass
    >>> obj = MainClass()
"""
```

**路径示例**:
- Scheduler模块: `docs/modules/schedulers/global/ppo_scheduler_modular.md`
- Training模块: `docs/modules/training/ppo_trainer.md`
- 相关指南: `docs/guides/training/ppo_warm_start_guide.md`

详细模板见: `docs/templates/module_doc_template.md`

---

## 🧪 测试与CI/CD规范

**核心原则**: 代码质量保证，CI/CD自动化

**关键内容**:
- 测试框架: pytest, 16个现有测试文件
- CI/CD流程: GitHub Actions, make lint/format
- 测试编写: AAA模式, 覆盖率要求 ≥ 60%
- 提交前检查: `make format && make lint && pytest tests/`

**快速参考**:
```bash
# 格式化代码
make format

# 运行测试
pytest tests/

# CI/CD检查
make lint
```

**详见**: [.claude/sections/testing_cicd.md](.claude/sections/testing_cicd.md)

---

## 🐛 调试与错误处理规范

**核心原则**: 系统性诊断，让错误自然暴露

**关键内容**:
- 调试流程: 6步系统性工作流 (症状 → metrics → TensorBoard → logs → pdb → profiler)
- 症状诊断: 熵爆炸, KL爆炸, 价值损失停滞, 解释方差为0等
- 指标健康范围: EV > 0.3, KL < 0.05, clipfrac < 0.30, entropy > 0.1
- 错误处理: 禁止静默捕获, 禁止fallback

**快速诊断**:
| 症状 | 检查指标 | 常见原因 |
|------|----------|----------|
| 熵爆炸 | `explained_var` | 奖励信号弱 |
| KL爆炸 | `approx_kl` | 学习率过大 |
| 价值停滞 | `value_loss` | 状态表示问题 |

**详见**: [.claude/sections/debugging.md](.claude/sections/debugging.md)

---

## 🚀 部署与生产规范

**核心原则**: 安全部署，快速回滚

**关键内容**:
- 生产就绪清单: 7项必检 (测试, lint, checkpoint验证, baseline对比等)
- 环境分离: dev/staging/prod 三层隔离
- Checkpoint管理: 版本控制, 验证流程, 回滚策略
- SLA定义: p99延迟 < 50ms, 可用性 99.9%, 性能超baseline 10%

**快速回滚**:
```bash
pkill -f train_ppo
cp outputs/checkpoints/archive/YYYY-MM-DD/*.pt outputs/checkpoints/latest.pt
./scripts/training/ppo/train_ppo_shared.sh --resume
```

**详见**: [.claude/sections/deployment.md](.claude/sections/deployment.md)

---

## ⚡ 性能优化规范

**核心原则**: Profiling驱动优化，避免过早优化

**关键内容**:
- Profiling工作流: cProfile → PyTorch Profiler → TensorBoard可视化
- 关键参数: batch_size, num_rollout_steps, use_fused_adam
- 内存优化: detach hidden states, 清理buffer, 减少TensorBoard记录
- 分布式训练: DistributedDataParallel推荐

**优化优先级**:
```
1. 算法优化 (最大收益) → 奖励设计, 状态表示
2. 超参数优化 (中等) → batch_size, lr
3. 代码优化 (有限) → Profiling, kernel优化
4. 硬件升级 (最贵) → 更好的GPU
```

**详见**: [.claude/sections/performance.md](.claude/sections/performance.md)

---

## 🤝 贡献指南

**核心原则**: 规范流程，高质量代码

**关键内容**:
- PR流程: Fork → 分支 → 实现 → 测试 → 提交 → 审核
- 分支命名: `feat/`, `fix/`, `docs/`, `exp/`, `refactor/`
- Commit格式: `type(scope): subject` (conventional commits)
- PR前缀: `[Bugfix]`, `[Feat]`, `[Doc]`, `[Core]` 等

**提交前检查**:
- [ ] `make format && make lint`
- [ ] 测试通过
- [ ] 文档更新

**详见**: [.claude/sections/contribution.md](.claude/sections/contribution.md)

---

## ⚙️ 配置管理规范

**核心原则**: 版本控制，环境隔离

**关键内容**:
- 配置结构: `configs/base/`, `configs/experiments/`, `configs/production/`
- 多环境策略: dev/staging/prod 配置分离
- 配置验证: 运行时检查参数合理性
- 版本追溯: 自动保存 config.yaml + git_hash

**参数文档化**:
```python
@dataclass
class PPOConfig:
    gamma: float = 0.99
    """Discount factor. Range: (0,1], Impact: 长期回报权重"""
```

**详见**: [.claude/sections/configuration.md](.claude/sections/configuration.md)

---

## 🧪 实验可复现规范

### 训练脚本索引

**脚本已按工作流重组**（详见scripts/README.md）:

| 类别 | 脚本路径 | 用途 |
|------|----------|------|
| PPO训练 | `scripts/training/ppo/train_ppo_*.sh` | PPO训练（25种配置） |
| BC训练 | `scripts/training/bc/train_bc_*.sh` | 行为克隆训练（5种） |
| 预训练 | `scripts/training/pretrain/standalone_pretrain.sh` | 独立预训练 |
| **LTR训练** | `scripts/training/ltr/train_request_ltr.py` | 🆕 Learning-to-Rank 离线训练 |
| 模型对比 | `scripts/evaluation/compare_*.sh` | 调度器性能对比（17种） |
| 监控 | `scripts/monitoring/monitor_training.sh` | 实时监控训练 |

**使用示例**:
```bash
# PPO训练
./scripts/training/ppo/train_ppo_shared.sh

# RUDDER训练
./scripts/training/ppo/train_ppo_rudder_mmpp.sh

# 与baseline对比
./scripts/evaluation/compare_with_latest_checkpoint.sh

# 监控训练
./scripts/monitoring/monitor_training.sh

# 查看TensorBoard
tensorboard --logdir outputs/runs/ppo_training/
```

### Outputs目录结构

**新组织方式**（详见outputs/experiments/README.md）:

```
outputs/
├── experiments/              # 结构化实验输出（新）
│   ├── 2025-12/             # 按月分组
│   ├── 2026-01/
│   │   ├── {exp_name}/      # 实验目录
│   │   │   ├── config.yaml  # 完整配置
│   │   │   ├── git_hash.txt # 代码版本
│   │   │   ├── metrics.db   # SQLite数据库
│   │   │   ├── checkpoints/ # 模型检查点
│   │   │   └── logs/        # 训练日志
├── runs/                     # TensorBoard日志
│   ├── archive/             # 旧运行归档
│   └── ppo_training/        # 保留最近20次
├── checkpoints/             # 模型检查点
│   ├── checkpoint_step_*.pt
│   ├── latest.pt
│   ├── archive/             # 备份归档（新）
│   │   ├── backup_undated/
│   │   ├── 2025-12-11/
│   │   └── 2025-12-12/
│   └── ppo_*/               # 分实验保存
└── normalizer_stats/         # Normalizer统计
```

### 实验元数据要求

**关键变更**: 所有新实验必须包含完整元数据

**必需文件**:
1. **config.yaml** - 完整实验配置
   ```yaml
   experiment_name: ppo_shared_20260117
   created: 2026-01-17T12:00:00+00:00
   git_hash: 94f8171b22d7f43a
   git_branch: toymodel_PPO
   hyperparameters:
     gamma: 0.99
     clip_epsilon: 0.2
     # ...
   ```

2. **git_hash.txt** - 代码版本追踪
   ```bash
   git rev-parse HEAD > "${OUTPUT_DIR}/git_hash.txt"
   ```

3. **metrics.db** - SQLite指标数据库
   - 训练指标（每步）
   - 趋势分析
   - 健康评分

**命名规范**:
- 实验目录: `{scope}_{samples}samples` 或描述性名称
- 避免时间戳后缀（使用日期目录分组）
- 避免版本号（使用git_hash代替）

**训练脚本模板**（新实验必须生成元数据）:
```bash
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ppo_shared_$(date +%Y%m%d)}"
OUTPUT_DIR="outputs/experiments/$(date +%Y-%m)/${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"/{checkpoints,logs}

# 保存git hash
git rev-parse HEAD > "${OUTPUT_DIR}/git_hash.txt"

# 保存配置
cat > "${OUTPUT_DIR}/config.yaml" << EOF
experiment_name: ${EXPERIMENT_NAME}
created: $(date -Iseconds)
git_hash: $(git rev-parse HEAD)
git_branch: $(git branch --show-current)
# ... 超参数 ...
EOF
```

### 训练监控规范

**核心原则**: 监控训练时使用SQLite metrics数据库，而非tail log文件

**MetricsDB位置**: `outputs/runs/ppo_training/YYYYMMDD_HHMMSS/metrics.db`

**监控方式**:
```bash
# ✅ 推荐: 查询SQLite数据库
sqlite3 outputs/runs/ppo_training/*/metrics.db "SELECT step, ev, kl, clip_frac FROM ppo_metrics ORDER BY step DESC LIMIT 20"

# ✅ 查看训练趋势
sqlite3 outputs/runs/ppo_training/*/metrics.db "SELECT step, avg(ev) as avg_ev, avg(kl) as avg_kl FROM ppo_metrics GROUP BY step/1000 ORDER BY step"

# ✅ 检查关键指标
sqlite3 outputs/runs/ppo_training/*/metrics.db "SELECT * FROM training_summary ORDER BY timestamp DESC LIMIT 5"

# ❌ 避免: tail log文件 (信息分散，难以分析)
# tail -f outputs/runs/ppo_training/*/ppo_training.log
```

**关键监控指标**:
| 指标 | 健康范围 | 说明 |
|------|----------|------|
| `ev` (Explained Variance) | > 0.3 | Critic预测质量 |
| `kl` (KL Divergence) | < 0.05 | 策略更新幅度 |
| `clip_frac` | < 0.30 | PPO clip触发比例 |
| `policy_loss` | 稳定下降 | 策略优化进度 |
| `value_loss` | 稳定下降 | 价值函数拟合 |
| `entropy` | > 0.1 | 探索程度 |

**数据库Schema** (MetricsDB):
- `metrics`: 每次PPO更新的详细指标 (step, pi_loss, vf_loss, entropy, approx_kl, clipfrac, explained_var, reward_mean等)
- `trend_summary`: 定期趋势分析 (ev_trend, loss_trend, health_score, warnings等)

**示例查询**:
```sql
-- 查看最近20次更新的关键指标
SELECT step, explained_var, approx_kl, clipfrac, reward_mean FROM metrics ORDER BY step DESC LIMIT 20;

-- 查看健康评分和警告
SELECT step_end, health_score, ev_mean, kl_mean, warnings FROM trend_summary ORDER BY step_end DESC LIMIT 5;
```

### 性能对比测试规范

**核心原则**: 长时间测试必须启用 Memory Manager 和 Cleanup

**强制要求**:
- 🔥 **所有 ≥5000 requests 的测试必须启用 memory cleanup**
- 🔥 **生产性能评估必须使用 20000+ requests**
- 🔥 **每次测试后强制 gc.collect()**

**Memory Manager 配置模板**:
```python
from src.core.utils.infrastructure.memory_manager import create_memory_manager

memory_config = {
    'memory_management': {
        'threshold_gb': 3.0,
        'check_interval': 5000,  # Check every 5000 requests
        'enable_auto_cleanup': True,
    },
    'enable_partial_cleanup': True,
    'partial_cleanup_ratio': 0.8,  # Keep 20% recent data
    'skip_critical_metrics_cleanup': False,  # Clean all for comparison
}
memory_manager = create_memory_manager(memory_config)
```

**测试规模要求**:
| 测试类型 | 最小 Requests | Memory Cleanup | 说明 |
|----------|--------------|----------------|------|
| 快速验证 | 1000 | 可选 | 功能验证，不保证性能准确 |
| 性能对比 | 5000 | **必须** | 初步性能对比 |
| 生产评估 | 20000 | **必须** | 最终性能结论 |
| 极限测试 | 50000+ | **必须** | 稳定性和内存泄漏测试 |

**禁止**:
- ❌ 使用 1000 requests 结果作为最终性能结论
- ❌ 长时间测试不启用 memory cleanup
- ❌ 忽略内存增长监控

---

## 🔗 关键引用

### 核心参考

| 文献 | 出处 | 关联 |
|------|------|------|
| Vidur | OSDI 2024 | LLM推理模拟器基础 |
| Sarathi-Serve | arXiv 2024 | Chunked Prefill调度 |
| RUDDER | NeurIPS 2019 | 延迟奖励信用分配 |
| PPO | arXiv 2017 | 策略梯度算法 |
| GAE | ICLR 2016 | 广义优势估计 |

### 调度算法参考

| 算法 | 说明 |
|------|------|
| SQF (Shortest Queue First) | 选择队列最短的replica |
| SJF (Shortest Job First) | 优先处理总token数少的请求 |
| SRTF (Shortest Remaining Tokens First) | 优先处理剩余token少的请求 |
| FCFS (First Come First Serve) | 先到先服务，公平但非最优 |
| Round Robin | 轮询分配 |

---

## 📝 TODO管理

### 项目级TODO
位置: `docs/PROJECT_TODO.md`

### 优先级定义
- **P0**: 阻塞开发/严重bug，立即处理
- **P1**: 新功能/重构，本周完成
- **P2**: 代码清理/文档补充，本月完成

### 使用规范
- 会话开始前查看 P0/P1 任务
- 会话中使用 TodoWrite 工具管理当前任务
- 会话结束后更新 PROJECT_TODO.md

---

## 📅 每日工作日志

**位置**: `docs/daily_logs/YYYY-MM/YYYY-MM-DD.md`

**内容规范**:
- 今日完成（功能/任务）
- 新增/修改的文件路径
- 验证结果
- 待办事项
- 相关文档链接

**查看历史**:
```bash
# 搜索某个功能是何时实现的
grep -r "RUDDER" docs/daily_logs/
```

---

*最后更新: 2026-02-13*
*项目版本: v2.4.3 - SC+NAILONG Multi-Replica Validated*

**v2.4.3 变更** (2026-02-13):
- ✅ **多副本验证**: 2R(QPS=9) + 8R(QPS=36) 20K requests, 5 seeds
- ✅ **8R combo最强P99**: -3.67% (p<0.0001), 超越4R的-2.91%
- ✅ **SC增量随副本数增强**: 2R无效 → 4R P99 -0.43% → 8R P99 -1.33% (p=0.0008)
- ✅ **NAILONG尺度不变**: ~2.4% Mean/P99 无论副本数
- ✅ **SC最低有效规模**: 需要4+副本 (2副本时SQF已近最优)
- ✅ `validate_social_cost_nailong.py`: 支持 `NUM_REPLICAS`/`QPS` 环境变量

**v2.4.2 变更** (2026-02-13):
- ✅ **50K 极限规模验证**: SC+NAILONG Mean -3.56%, P99 -2.91% (p<0.0001, 5/5 wins)
- ✅ SC P99 增量恢复显著: -0.43% (p=0.025), 20K p=0.11 确认为采样波动
- ✅ SC Mean 增量单调递增: -0.88%(10K) → -1.05%(20K) → -1.10%(50K)
- ✅ NAILONG重训练SC实验: 探索完毕，保留原SARATHI训练checkpoint
- ✅ `sweep_beta.py` 增强: 支持 `--temp` 和 `--mlr` 参数

**v2.4.0 变更** (2026-02-12):
- ✅ SC+NAILONG 组合验证通过: Mean -3.39%, P99 -2.83% vs baseline (20K, p<0.0001, 5/5 wins)
- ✅ SC Mean增量稳健: -1.05% (p=0.0009)
- ✅ SC P99增量有限: -0.36% (p=0.11, 不显著)
- ✅ Phase 1 探索 (4 QPS × 2 seeds): 确定 QPS=18 为最优复合点
- ✅ Phase 3 验证 (10K + 20K requests, 5 seeds)
- ✅ 新增 `scripts/evaluation/compare_social_cost_nailong.py` (Phase 1 探索脚本)
- ✅ 新增 `scripts/evaluation/validate_social_cost_nailong.py` (Phase 3 验证脚本)
- ✅ 进度报告: `docs/progress/2026_02_12_sc_nailong_combination.md`

**v2.3.0 变更** (2026-02-11):
- ✅ 新增 Social Cost Global Scheduler（双 MLP 预测器路由）
- ✅ 新增 `vidur/scheduler/global_scheduler/social_cost_global_scheduler.py` (369行)
- ✅ 新增 `vidur/scheduler/global_scheduler/forced_action_global_scheduler.py` (140行)
- ✅ 新增 `src/core/models/social_cost_model.py` (282行)
- ✅ EnrichedMinimalStateBuilder 扩展至 13 维（+5 token 分布特征）
- ✅ 完整 Pipeline: 数据收集 → 训练 → Beta 调优 → Tradeoff 探索 → 统计验证
- ✅ **统计验证通过 (20K requests, 5 seeds)**: P99 胜率 15/15, 所有 p < 0.05
- ✅ Safety clamp (mlr=1.5) 消除 QPS=6 灾难性路由
- ✅ Score temperature (0.1) softmin 概率选择优于 argmin
- ✅ 评估脚本: `confirm_advantage_zone.py`, `explore_tradeoff.py`, `sweep_beta.py`
- ✅ 进度报告: `docs/progress/2026_02_11_social_cost_scheduler.md`

**v2.2.0 变更** (2026-02-01):
- ✅ 新增 Learned Speed-Aware SQF 全局调度器
- ✅ 新增 `vidur/scheduler/global_scheduler/learned_speed_sqf_global_scheduler.py`
- ✅ 在线学习副本速度因子，使用加权 SQF 路由
- ✅ 在异构副本环境下性能提升 17.6% (vs Plain SQF)
- ✅ 进度报告: `docs/progress/2026_02_01_learned_speed_sqf_scheduler.md`

**v2.1.0 变更** (2026-02-01):
- ✅ Enhancement 17: Learning-to-Rank 请求调度实现
- ✅ Enhancement 18: Thompson Sampling 击败 SARATHI + SJF
- ✅ 新增 `src/core/models/request_ltr_model.py` - LTR 排序模型
- ✅ 新增 `scripts/training/ltr/train_request_ltr.py` - 离线训练脚本
- ✅ NAILONG 调度器集成 LTR 数据收集和推理模式
- ✅ 完成与 SARATHI 基线对比测试 (LTR 无法超越)
- ✅ 进度报告: `docs/progress/2026_02_01_enhancement17_ltr_scheduling.md`

**v2.0.0 变更** (2026-01-18):
- ✅ 删除57个重复文件（218 → 161文件）
- ✅ 合并 `plans/` → `plan/` 目录
- ✅ 清理 `modules/` 扁平结构（保留层级结构）
- ✅ 修复PDF命名（删除空格，统一小写）
- ✅ 重命名中文文件名为英文

**v1.9.0 变更** (2026-01-17):
- ✅ 测试与CI/CD规范
- ✅ 调试与错误处理规范
- ✅ 部署与生产规范
- ✅ 性能优化规范
- ✅ 贡献指南
- ✅ 配置管理规范
