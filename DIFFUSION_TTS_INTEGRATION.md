# Diffusion-TTS 方法整合计划

## 目标

从 `rvignav/diffusion-tts` 提取并整合三个方法到我们的pipeline：
1. **Best-of-N (Rejection Sampling)**
2. **Zero-Order Search**
3. **ε-greedy Search**

复现论文实验 5.1: Class-Conditional Image Generation on ImageNet-64

## 实验设定（与论文一致）

- **模型**: EDM ImageNet-64x64 (class-conditional)
- **采样步数**: 18 steps
- **Verifier/Scorer**:
  - BrightnessScorer
  - CompressibilityScorer  
  - ImageNetScorer (classifier probability)
- **参数**:
  - Best-of-N: N=4
  - Zero-Order: N=4, K=20, λ=0.15
  - ε-greedy: N=4, K=20, λ=0.15, ε=0.4

## 实现计划

### 1. 提取核心算法

从 `code_repos/diffusion-tts/edm/main.py` 提取：
- `REJECTION_SAMPLING` (lines 101-137)
- `ZERO_ORDER` / `EPS_GREEDY` (lines 714-860)

### 2. 创建适配器

- **EDM Model Wrapper**: 适配EDM模型到 `BaseDiffusionModel`
- **Scorer Verifier**: 适配scorer到 `BaseVerifier`
- **Search Methods**: 实现三个方法适配 `BaseSearch` 接口

### 3. 实验配置

创建 `configs/imagenet64_diffusion_tts.yaml` 配置文件

## 关键差异

diffusion-tts的方法是在**每个时间步**内进行搜索，而我们的pipeline支持：
- **全局搜索**: 完整轨迹搜索（如Best-of-N）
- **局部搜索**: 每个时间步的搜索（如Zero-Order, ε-greedy）

需要同时支持两种模式。


