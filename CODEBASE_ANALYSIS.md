# 代码库分析与组件汇总

## 📁 代码库结构概览

### 1. tt-scale-flux (sayakpaul/tt-scale-flux)

**主要功能：** Random Search + Zero-Order Search + 多种Verifier

**核心目录结构：**
```
tt-scale-flux/
├── main.py                 # 主实验脚本
├── utils.py                # 工具函数（包含Random Search和ZO Search实现）
├── verifiers/              # Verifier实现
│   ├── base_verifier.py
│   ├── gemini_verifier.py
│   ├── qwen_verifier.py
│   ├── openai_verifier.py
│   └── laion_aesthetics.py
└── configs/                # 配置文件
```

### 2. Diffusion-inference-scaling (XiangchengZhang/Diffusion-inference-scaling)

**主要功能：** BFS/DFS/MCMC等经典搜索方法 + 完整的pipeline

**核心目录结构：**
```
Diffusion-inference-scaling/
├── imagenet/
│   ├── methods/            # 各种Local Search方法
│   │   ├── base.py         # BaseGuidance基类
│   │   ├── bfs.py          # BFS搜索
│   │   ├── cg.py           # 其他方法
│   │   ├── dps.py
│   │   ├── freedom.py
│   │   ├── lgd.py
│   │   ├── mpgd.py
│   │   ├── tfg.py
│   │   └── ugd.py
│   ├── pipeline.py         # Pipeline实现
│   ├── searching.py        # Beam Search实现
│   ├── evaluations/        # 评估工具（FID/IS等）
│   └── tasks/              # 任务相关代码
├── text_to_image/          # Text-to-Image相关
└── locomotion/             # 运动规划相关
```

---

## 🔍 Local Search算法汇总

### 来自 tt-scale-flux

#### 1. **Random Search** ✅
- **位置**: `utils.py` + `main.py`
- **实现方式**: 
  - 每个search round采样 `2^search_round` 个初始噪声
  - 并行采样完整轨迹
  - 使用verifier评估所有候选
  - 选择verifier score最高的
- **关键函数**:
  - `get_noises()`: 生成多个初始噪声
  - `sample()`: 批量采样和评估
- **特点**: 简单直接，适合作为baseline

#### 2. **Zero-Order Search (ZO-N)** ✅
- **位置**: `utils.py` + `main.py`
- **实现方式**:
  - 从pivot噪声开始
  - 使用`generate_neighbors()`在pivot邻域生成多个噪声
  - 评估所有邻居，选择最优的作为新pivot
  - 迭代进行
- **关键函数**:
  - `generate_neighbors()`: 在单位球面上生成正交邻居（threshold=0.95）
  - 支持多轮迭代，每轮根据改善情况决定是否继续
- **特点**: Pivot-based迭代搜索，比Random Search更高效

---

### 来自 Diffusion-inference-scaling

#### 3. **BFS (Breadth-First Search)** ✅
- **位置**: `methods/bfs.py`
- **实现方式**:
  - 在每个时间步维护多个候选粒子
  - 使用Monte Carlo估计guidance梯度
  - 支持两种模式：
    - `bfs-resample`: Resampling模式
    - `bfs-prune`: Pruning模式
- **关键特性**:
  - 在特定步骤进行resampling/pruning
  - 使用温度参数控制resampling概率
  - 支持多粒子系统
- **特点**: 真正的树搜索，适合需要多候选的场景

#### 4. **其他方法** (可参考)
- **CG (Classical Guidance)**: `methods/cg.py`
- **DPS (Diffusion Posterior Sampling)**: `methods/dps.py`
- **LGD (Local Gradient Descent)**: `methods/lgd.py`
- **MPGD (Multi-Path Gradient Descent)**: `methods/mpgd.py`
- **TFG**: `methods/tfg.py`
- **UGD**: `methods/ugd.py`

**注意**: 这些方法主要是guidance-based的方法，可以在单步上应用，可以作为Local Search的primitive。

---

## ✅ Verifier汇总

### 来自 tt-scale-flux

#### 1. **GeminiVerifier** ✅
- **位置**: `verifiers/gemini_verifier.py`
- **特点**:
  - 使用Gemini 2.0 Flash模型
  - 支持多种metrics:
    - `accuracy_to_prompt`
    - `creativity_and_originality`
    - `visual_quality_and_realism`
    - `consistency_and_cohesion`
    - `emotional_or_thematic_resonance`
    - `overall_score`
  - 结构化输出（JSON格式）
  - 并行处理多个输入

#### 2. **QwenVerifier** ✅
- **位置**: `verifiers/qwen_verifier.py`
- **特点**:
  - 使用Qwen2.5 VL模型
  - 支持结构化输出（使用outlines和pydantic）
  - 类似的metrics支持

#### 3. **OpenAIVerifier** ✅
- **位置**: `verifiers/openai_verifier.py`
- **特点**:
  - 使用OpenAI的视觉模型
  - 结构化输出

#### 4. **LAIONAestheticVerifier** ✅
- **位置**: `verifiers/laion_aesthetics.py`
- **特点**:
  - 使用LAION的aesthetic predictor
  - 专门用于评估美学质量

**统一接口**:
```python
class BaseVerifier:
    def prepare_inputs(images, prompts):  # 准备输入
    def score(inputs):                    # 计算分数
```

---

### 来自 Diffusion-inference-scaling

#### 5. **Classifier-based Verifier** ✅
- **位置**: `tasks/` 目录下各种任务的guider
- **实现方式**:
  - 使用预训练分类器（如ImageNet classifier）
  - 通过`BaseGuider`类封装
  - 提供`get_guidance()`方法返回log概率或梯度
- **特点**: 更适合图像分类任务，可以直接使用分类器的log概率

---

## 🔧 关键代码组件提取建议

### Local Search算法提取

#### 从 tt-scale-flux 提取：

1. **Random Search核心逻辑**:
   - `utils.py::get_noises()` - 噪声生成
   - `main.py::sample()` - 采样和评估流程
   - 噪声池大小：`2^search_round`

2. **Zero-Order Search核心逻辑**:
   - `utils.py::generate_neighbors()` - 邻居生成算法
   - `main.py`中的ZO迭代逻辑（lines 290-301）
   - threshold参数控制邻居距离

#### 从 Diffusion-inference-scaling 提取：

1. **BFS核心逻辑**:
   - `methods/bfs.py::BFSGuidance` - BFS实现
   - `guide_step()` - 单步guidance计算
   - resampling/pruning逻辑（lines 151-166）
   - Monte Carlo估计（`tilde_get_guidance()`）

2. **BaseGuidance框架**:
   - `methods/base.py::BaseGuidance` - 统一基类
   - `_predict_x0()`, `_predict_x_prev_from_zero()` - DDIM核心步骤

---

### Verifier提取

#### 从 tt-scale-flux 提取：

1. **BaseVerifier接口**:
   - `verifiers/base_verifier.py` - 统一的Verifier接口
   - `prepare_inputs()` + `score()` 方法

2. **具体Verifier实现**:
   - 可以直接复用Gemini/Qwen/OpenAI Verifier
   - 或者提取接口设计思路，实现自己的分类器Verifier

#### 从 Diffusion-inference-scaling 提取：

1. **Classifier Guider**:
   - `tasks/base.py::BaseGuider` - 分类器guider基类
   - 可以直接用于图像分类任务作为verifier

---

## 📋 适配到我们的Pipeline的建议

### Local Search方法适配

1. **Random Search** → 已实现，可直接参考tt-scale-flux的实现完善
2. **Zero-Order Search** → 已实现框架，需要完善`generate_neighbors()`的实现
3. **BFS** → 需要适配：
   - 提取BFSGuidance的核心逻辑
   - 适配到我们的BaseSearch接口
   - 注意：BFS是在每个时间步进行，需要支持单步操作

### Verifier适配

1. **Classifier Verifier** → 已实现框架，需要：
   - 完善模型加载逻辑
   - 实现具体的score计算

2. **高级Verifier** (Gemini/Qwen等):
   - 可以保留原实现，通过wrapper适配到我们的BaseVerifier接口
   - 或者提取接口设计，实现简化版本

---

## 🎯 下一步行动计划

### 第一步：提取Random Search
- [ ] 从tt-scale-flux提取`get_noises()`逻辑
- [ ] 完善我们的`RandomSearch`类实现

### 第二步：完善Zero-Order Search
- [ ] 从tt-scale-flux提取`generate_neighbors()`实现
- [ ] 完善我们的`ZeroOrderSearch`类

### 第三步：提取BFS方法
- [ ] 分析BFSGuidance的实现
- [ ] 创建一个BFS Local Search适配器

### 第四步：完善Verifier
- [ ] 从tt-scale-flux提取BaseVerifier接口设计
- [ ] 实现ClassifierVerifier的具体逻辑

---

## 📊 代码结构对比

| 组件 | tt-scale-flux | Diffusion-inference-scaling | 我们的实现 |
|------|--------------|----------------------------|----------|
| **Random Search** | ✅ 完整实现 | ❌ | ✅ 框架已实现 |
| **Zero-Order** | ✅ 完整实现 | ❌ | ✅ 框架已实现 |
| **BFS** | ❌ | ✅ 完整实现 | ❌ 需要添加 |
| **Verifier接口** | ✅ 清晰 | ⚠️ 分散 | ✅ 框架已实现 |
| **高级Verifier** | ✅ Gemini/Qwen/OpenAI | ⚠️ 主要是分类器 | ❌ 需要实现 |

---

## 💡 关键发现

1. **tt-scale-flux的优势**:
   - 代码结构清晰，接口设计好
   - Verifier实现完整，支持多种API
   - Random和ZO Search实现简洁易懂

2. **Diffusion-inference-scaling的优势**:
   - BFS/DFS等经典搜索方法实现完整
   - 单步guidance计算详细（适合Local Search）
   - Pipeline结构完整，适合参考

3. **我们的Pipeline优势**:
   - 两层级架构清晰（Local + Global）
   - 统一的接口设计
   - 易于扩展和组合

4. **整合策略**:
   - 直接提取算法核心逻辑
   - 适配到我们的统一接口
   - 保持代码简洁和模块化


