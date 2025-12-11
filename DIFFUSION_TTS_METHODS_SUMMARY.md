# Diffusion-TTS 方法整合总结

## ✅ 已完成

### 1. 方法实现

已从 `rvignav/diffusion-tts` 提取并实现三个方法：

#### **BestOfNSearch** (Best-of-N / Rejection Sampling)
- **位置**: `src/search/diffusion_tts_search.py`
- **算法**: 采样N个完整轨迹，选择verifier score最高的
- **NFE**: N × num_steps
- **参数**: `n_candidates=4`

#### **ZeroOrderSearchTTS** (Zero-Order Search)
- **位置**: `src/search/diffusion_tts_search.py`
- **算法**: 在每个时间步进行K轮局部搜索，每轮生成N个候选噪声
- **NFE**: num_steps × (K × N × 2 + 1)
- **参数**: `n_candidates=4`, `search_steps=20`, `lambda_param=0.15`

#### **EpsilonGreedySearch** (ε-greedy Search)
- **位置**: `src/search/diffusion_tts_search.py`
- **算法**: 类似Zero-Order，但以概率ε使用新鲜高斯样本（全局探索）
- **NFE**: 同Zero-Order
- **参数**: `n_candidates=4`, `search_steps=20`, `lambda_param=0.15`, `epsilon=0.4`

### 2. 接口适配

- ✅ 所有方法继承 `BaseSearch`
- ✅ 实现 `search()` 方法
- ✅ 支持NFE计数
- ✅ 返回统一的info字典

### 3. 实验配置

- ✅ 创建 `configs/imagenet64_diffusion_tts.yaml`
- ✅ 配置与论文一致（18步，参数相同）

## 📋 待完成

### 1. EDM模型Wrapper

需要创建 `src/models/edm_model.py`:
- 适配EDM模型到 `BaseDiffusionModel`
- 实现 `denoise_step()` 和 `sample_noise()`
- 支持class-conditional生成

### 2. Scorer Verifier

需要创建 `src/verifiers/scorer_verifier.py`:
- 适配scorer到 `BaseVerifier`
- 支持三种scorer: Brightness, Compressibility, ImageNet
- 从 `code_repos/diffusion-tts/edm/scorers.py` 提取

### 3. 实验脚本

需要创建 `scripts/run_diffusion_tts_experiment.py`:
- 加载EDM模型
- 运行三个方法
- 使用三种scorer评估
- 保存结果

## 🔧 使用方法

### 在Pipeline中使用

```python
from src.search.diffusion_tts_search import BestOfNSearch, ZeroOrderSearchTTS, EpsilonGreedySearch
from src.pipeline.sampling_pipeline import SamplingPipeline

# 创建search方法
search_method = EpsilonGreedySearch(
    model=model,
    verifier=verifier,
    n_candidates=4,
    search_steps=20,
    lambda_param=0.15,
    epsilon=0.4
)

# 在pipeline中使用
pipeline = SamplingPipeline(model, verifier)
samples, info = pipeline.sample(
    method="local_search",
    batch_size=36,
    num_steps=18,
    local_search_type="epsilon_greedy",
    n_candidates=4,
    search_steps=20,
    lambda_param=0.15,
    epsilon=0.4
)
```

## 📊 论文结果对比

根据论文Table 1，在ImageNet-64上的结果：

| Method | Brightness | Compressibility | Classifier | NFEs |
|--------|-----------|----------------|------------|------|
| Naive Sampling | 0.4965±0.01 | 0.3563±0.07 | 0.3778±0.04 | 18 |
| Best of 4 | 0.5767±0.01 | 0.4220±0.02 | 0.5461±0.00 | 72 |
| Zero-Order (N=4, K=20) | 0.6083±0.01 | 0.3751±0.02 | 0.6261±0.04 | 1440 |
| ε-greedy (N=4, K=20) | **0.9813±0.01** | **0.7208±0.03** | **0.9885±0.04** | 1440 |

## 🎯 下一步

1. **实现EDM模型wrapper** - 适配EDM到我们的接口
2. **实现Scorer verifier** - 适配scorer到我们的接口
3. **创建实验脚本** - 运行完整实验
4. **验证结果** - 对比论文结果


