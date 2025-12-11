# Diffusion-TTS 整合完成总结

## ✅ 已完成的工作

### 1. Scorer Verifier ✅

**文件**: `src/verifiers/scorer_verifier.py`

实现了三种scorer的适配：
- **BrightnessScorer**: 计算感知亮度 (0.2126*R + 0.7152*G + 0.0722*B)
- **CompressibilityScorer**: 基于JPEG压缩大小计算可压缩性
- **ImageNetScorer**: 使用ImageNet分类器计算目标类别概率

**关键特性**:
- 继承 `BaseVerifier` 接口
- 支持NFE计数
- 自动下载ImageNet分类器（如果需要）
- 处理不同的图像格式（uint8/float32）

### 2. EDM模型Wrapper ✅

**文件**: `src/models/edm_model.py`

实现了EDM模型的完整包装：
- 支持class-conditional ImageNet-64x64生成
- 实现EDM采样器（Heun方法，二阶校正）
- 支持自定义采样参数（sigma_min, sigma_max, rho, S_churn等）
- 继承 `BaseDiffusionModel` 接口

**关键特性**:
- 自动从URL加载模型checkpoint
- 支持时间步离散化
- 实现完整的EDM采样流程
- 支持NFE计数

### 3. Search方法 ✅

**文件**: `src/search/diffusion_tts_search.py`

实现了三个search方法：
- **BestOfNSearch**: Best-of-N (Rejection Sampling)
- **ZeroOrderSearchTTS**: Zero-Order Search
- **EpsilonGreedySearch**: ε-greedy Search

所有方法都已适配到 `BaseSearch` 接口。

### 4. Pipeline集成 ✅

**更新**: `src/pipeline/sampling_pipeline.py`

添加了对新方法的支持：
- `best_of_n`
- `zero_order_tts`
- `epsilon_greedy`

### 5. 实验脚本 ✅

**文件**: `scripts/run_diffusion_tts_experiment.py`

创建了完整的实验脚本，支持：
- 加载EDM模型
- 创建Scorer Verifier
- 运行三种search方法
- 计算评估指标
- 保存结果

### 6. 配置文件 ✅

**文件**: `configs/imagenet64_diffusion_tts.yaml`

创建了与论文一致的实验配置。

## 📋 使用方法

### 基本使用

```python
from src.models.edm_model import EDMModel
from src.verifiers.scorer_verifier import ScorerVerifier
from src.search.diffusion_tts_search import EpsilonGreedySearch

# 创建模型
model = EDMModel(
    model_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
    device="cuda",
    image_size=64
)

# 创建verifier
verifier = ScorerVerifier(
    scorer_type="imagenet",  # 或 "brightness", "compressibility"
    device="cuda",
    image_size=64
)

# 创建search方法
search = EpsilonGreedySearch(
    model=model,
    verifier=verifier,
    n_candidates=4,
    search_steps=20,
    lambda_param=0.15,
    epsilon=0.4
)

# 运行搜索
class_labels = torch.eye(1000)[torch.randint(1000, size=(36,))].to("cuda")
samples, info = search.search(
    batch_size=36,
    num_steps=18,
    class_labels=class_labels
)
```

### 使用实验脚本

```bash
python scripts/run_diffusion_tts_experiment.py \
    --config configs/imagenet64_diffusion_tts.yaml
```

## 🔧 配置说明

### Scorer类型

在配置文件中设置 `verifier.scorer_type`:
- `brightness`: 亮度scorer
- `compressibility`: 可压缩性scorer
- `imagenet`: ImageNet分类器scorer（需要class_labels）

### Search方法参数

在 `pipeline.local_search` 中配置：

**Best-of-N**:
```yaml
best_of_n:
  n_candidates: 4
```

**Zero-Order**:
```yaml
zero_order_tts:
  n_candidates: 4
  search_steps: 20
  lambda_param: 0.15
```

**ε-greedy**:
```yaml
epsilon_greedy:
  n_candidates: 4
  search_steps: 20
  lambda_param: 0.15
  epsilon: 0.4
```

## 📊 预期结果

根据论文Table 1，在ImageNet-64上的结果：

| Method | Brightness | Compressibility | Classifier | NFEs |
|--------|-----------|----------------|------------|------|
| Naive | 0.4965±0.01 | 0.3563±0.07 | 0.3778±0.04 | 18 |
| Best of 4 | 0.5767±0.01 | 0.4220±0.02 | 0.5461±0.00 | 72 |
| Zero-Order | 0.6083±0.01 | 0.3751±0.02 | 0.6261±0.04 | 1440 |
| ε-greedy | **0.9813±0.01** | **0.7208±0.03** | **0.9885±0.04** | 1440 |

## ⚠️ 注意事项

1. **依赖**: 需要 `code_repos/diffusion-tts/edm/` 目录存在
   - 用于导入 `dnnlib` 和 `unet.EncoderUNetModel`

2. **模型下载**: EDM模型和ImageNet分类器会自动下载
   - EDM模型: ~500MB
   - ImageNet分类器: ~50MB

3. **内存要求**: 
   - ε-greedy和Zero-Order需要较多内存（K*N个候选）
   - 建议使用A100或类似GPU

4. **Class Labels**: ImageNet scorer需要class_labels
   - 格式: [B, num_classes] (one-hot) 或 [B] (class indices)

## 🎯 下一步

1. **运行实验**: 使用实验脚本运行完整实验
2. **验证结果**: 对比论文中的结果
3. **扩展**: 可以添加其他scorer或search方法


