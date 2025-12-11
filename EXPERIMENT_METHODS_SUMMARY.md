# ImageNet/CIFAR-10 可用Local Search方法汇总

## 📊 数据集支持情况

### ✅ 明确支持 ImageNet/CIFAR-10 的代码库

#### 1. **Diffusion-inference-scaling** (Zhang et al., 2025)
- **论文**: "Inference-time Scaling of Diffusion Models through Classical Search"
- **arXiv**: 2505.23614
- **支持数据集**: 
  - ✅ **CIFAR-10** (32x32, class-conditional)
  - ✅ **ImageNet** (256x256, class-conditional)
- **实验脚本位置**:
  - CIFAR-10: `imagenet/scripts/cifar10_label.sh`
  - ImageNet: `imagenet/scripts/imagenet_label.sh`
  - BFS搜索: `imagenet/scripts/search/bfs_*.sh`

#### 2. **tt-scale-flux** (Ma et al., 2025)
- **论文**: "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps"
- **arXiv**: 2501.09732
- **支持数据集**: 
  - ⚠️ 主要针对 **Text-to-Image** (FLUX, SDXL等)
  - ❌ 不直接支持 CIFAR-10/ImageNet class-conditional
  - ✅ 但算法（Random Search, ZO）可以适配

---

## 🔍 可用的Local Search方法

### 方法1: **Random Search** ✅

**来源**: tt-scale-flux (Ma et al.)

**实现位置**: 
- `code_repos/tt-scale-flux/main.py::sample()`
- `code_repos/tt-scale-flux/utils.py::get_noises()`

**算法描述**:
- 每个search round采样 `2^round` 个初始噪声
- 并行采样完整轨迹
- 使用verifier评估所有候选
- 选择verifier score最高的

**论文结果** (Ma et al.):
- 主要在 **Text-to-Image** 任务上验证
- 使用Gemini/Qwen作为verifier
- 结果显示增加search rounds可以提升生成质量

**适配到CIFAR-10/ImageNet**:
- ✅ 可以直接适配
- 需要替换verifier为分类器（CIFAR-10/ImageNet classifier）
- 我们的pipeline中已有实现框架

**关键参数**:
```python
search_rounds = 4  # 搜索轮数
num_noises_per_round = 2^round  # 每轮噪声数
```

---

### 方法2: **Zero-Order Search (ZO-N)** ✅

**来源**: tt-scale-flux (Ma et al.)

**实现位置**:
- `code_repos/tt-scale-flux/utils.py::generate_neighbors()`
- `code_repos/tt-scale-flux/main.py` (lines 290-301)

**算法描述**:
- 从pivot噪声开始
- 在单位球面上生成正交邻居（threshold=0.95）
- 评估所有邻居，选择最优的作为新pivot
- 迭代进行

**论文结果** (Ma et al.):
- 在Text-to-Image任务上验证
- 比Random Search更高效（更少的NFE）
- 支持多轮迭代

**适配到CIFAR-10/ImageNet**:
- ✅ 可以直接适配
- 核心算法 `generate_neighbors()` 已实现
- 我们的pipeline中已有框架

**关键参数**:
```python
threshold = 0.95  # 邻居距离阈值
num_neighbors = 4  # 每轮邻居数
search_rounds = 4  # 迭代轮数
```

---

### 方法3: **BFS (Breadth-First Search)** ✅

**来源**: Diffusion-inference-scaling (Zhang et al.)

**实现位置**:
- `code_repos/Diffusion-inference-scaling/imagenet/methods/bfs.py::BFSGuidance`

**算法描述**:
- 在每个时间步维护多个候选粒子
- 使用Monte Carlo估计guidance梯度
- 支持两种模式：
  - `bfs-resample`: Resampling模式（按概率重采样）
  - `bfs-prune`: Pruning模式（剪枝低分粒子）
- 在特定步骤进行resampling/pruning

**论文结果** (Zhang et al.):
- ✅ **ImageNet-256**: 有完整实验结果
- ✅ **CIFAR-10**: 有实验脚本和配置
- 论文中展示了BFS相比baseline的FID/IS提升
- 支持多粒子系统（`per_sample_batch_size`）

**实验配置示例** (ImageNet):
```bash
dataset="imagenet"
model_name_or_path='models/openai_imagenet.pt'
guidance_name='bfs-resample'  # 或 'bfs-prune'
per_sample_batch_size=12  # 粒子数
rho=0.2
mu=0.4
sigma=0.1
start=25  # resampling起始步
step_size=25  # resampling间隔
temp=1.0  # 温度参数
```

**实验配置示例** (CIFAR-10):
```bash
dataset="cifar10"
model_name_or_path='openai_cifar10.pt'
image_size=32
guidance_name='tfg'  # 或其他guidance方法
per_sample_batch_size=128
```

**关键参数**:
```python
per_sample_batch_size = 12  # 粒子数
rho = 0.2  # x_t guidance强度
mu = 0.4  # x_0 guidance强度
sigma = 0.1  # Monte Carlo噪声标准差
start = 25  # resampling起始步
step_size = 25  # resampling间隔
temp = 1.0  # 温度参数（控制resampling概率）
```

---

### 方法4: **TFG (Training-Free Guidance)** ✅

**来源**: Diffusion-inference-scaling (基于Training-Free-Guidance框架)

**实现位置**:
- `code_repos/Diffusion-inference-scaling/imagenet/methods/tfg.py`

**算法描述**:
- 基于分类器的guidance方法
- 在每个时间步计算guidance梯度
- 可以与其他方法组合

**论文结果**:
- 在CIFAR-10和ImageNet上都有实验
- 作为baseline方法

**实验配置** (CIFAR-10):
```bash
dataset="cifar10"
guidance_name='tfg'
guide_network='resnet_cifar10.pt'
rho=1
mu=0.25
sigma=0.001
```

---

### 方法5: **其他Guidance方法** (可参考)

**来源**: Diffusion-inference-scaling

**可用方法**:
- `cg.py` - Classical Guidance
- `dps.py` - Diffusion Posterior Sampling
- `lgd.py` - Local Gradient Descent
- `mpgd.py` - Multi-Path Gradient Descent
- `ugd.py` - Unconditional Guidance

**注意**: 这些主要是单步guidance方法，可以作为Local Search的primitive使用。

---

## 📈 论文实验结果总结

### Zhang et al. (Diffusion-inference-scaling)

**论文**: "Inference-time Scaling of Diffusion Models through Classical Search"

**实验结果**:
- ✅ **ImageNet-256**: 
  - 使用BFS (bfs-resample/bfs-prune)
  - 展示了FID/IS的提升
  - 支持多粒子系统
- ✅ **CIFAR-10**:
  - 有完整的实验脚本
  - 使用分类器作为verifier/guider
  - 支持多种guidance方法

**关键发现**:
- BFS在ImageNet上有效
- 多粒子系统可以提升性能
- Resampling和Pruning两种策略各有优势

---

### Ma et al. (tt-scale-flux)

**论文**: "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps"

**实验结果**:
- ✅ **Text-to-Image** (FLUX, SDXL):
  - Random Search和ZO Search都有效
  - 使用高级verifier (Gemini/Qwen)
  - 展示了多轮搜索的scaling效果
- ⚠️ **CIFAR-10/ImageNet**:
  - 论文中**没有**直接报告CIFAR-10/ImageNet的结果
  - 但算法可以适配

**关键发现**:
- Random Search简单有效
- ZO Search比Random更高效
- Verifier的选择很重要

---

## 🎯 推荐实验方案

### Phase 1: CIFAR-10 实验

#### 方法优先级排序:

1. **Random Search** ⭐⭐⭐⭐⭐
   - ✅ 实现简单
   - ✅ 已有框架
   - ✅ 作为baseline必须
   - **来源**: tt-scale-flux

2. **BFS (bfs-resample/bfs-prune)** ⭐⭐⭐⭐⭐
   - ✅ 有论文结果
   - ✅ 代码完整
   - ✅ 支持CIFAR-10
   - **来源**: Diffusion-inference-scaling

3. **Zero-Order Search** ⭐⭐⭐⭐
   - ✅ 算法清晰
   - ✅ 已有框架
   - ⚠️ 需要完善实现
   - **来源**: tt-scale-flux

4. **TFG** ⭐⭐⭐
   - ✅ 作为baseline
   - ✅ 代码完整
   - **来源**: Diffusion-inference-scaling

---

### Phase 2: ImageNet 实验

#### 方法优先级排序:

1. **BFS** ⭐⭐⭐⭐⭐
   - ✅ 论文中有ImageNet结果
   - ✅ 代码完整
   - ✅ 有实验脚本
   - **来源**: Diffusion-inference-scaling

2. **Random Search** ⭐⭐⭐⭐
   - ✅ 适配到ImageNet
   - ✅ 作为baseline
   - **来源**: tt-scale-flux (适配)

3. **Zero-Order Search** ⭐⭐⭐
   - ✅ 适配到ImageNet
   - ⚠️ 需要验证效果
   - **来源**: tt-scale-flux (适配)

---

## 📋 实验配置建议

### CIFAR-10 配置

```yaml
dataset: cifar10
image_size: 32
model_name_or_path: 'openai_cifar10.pt'  # 或 'google/ddpm-cifar10-32'
guide_network: 'resnet_cifar10.pt'  # 或 'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10'
inference_steps: 50
num_samples: 50000  # 用于FID评估
```

### ImageNet 配置

```yaml
dataset: imagenet
image_size: 256
model_name_or_path: 'models/openai_imagenet.pt'
guide_network: 'google/vit-base-patch16-224'
inference_steps: 100
num_samples: 50000  # 用于FID评估
```

---

## 🔧 实现优先级

### 高优先级 (必须实现)

1. **Random Search** - 基础baseline
2. **BFS (bfs-resample)** - 有论文结果，代码完整
3. **Classifier Verifier** - 用于CIFAR-10/ImageNet

### 中优先级 (建议实现)

4. **Zero-Order Search** - 算法清晰，需要完善
5. **BFS (bfs-prune)** - 作为BFS的变体

### 低优先级 (可选)

6. **TFG** - 作为baseline参考
7. **其他Guidance方法** - 可作为primitive

---

## 📚 参考文献

1. **Ma et al. (2025)**: "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps"
   - arXiv: 2501.09732
   - 代码: `sayakpaul/tt-scale-flux`
   - 主要贡献: Random Search, ZO Search, 高级Verifier

2. **Zhang et al. (2025)**: "Inference-time Scaling of Diffusion Models through Classical Search"
   - arXiv: 2505.23614
   - 代码: `XiangchengZhang/Diffusion-inference-scaling`
   - 主要贡献: BFS/DFS, ImageNet/CIFAR-10实验

---

## ✅ 下一步行动

1. **提取BFS实现** - 从Diffusion-inference-scaling提取
2. **完善Random Search** - 适配到CIFAR-10/ImageNet
3. **实现Classifier Verifier** - 用于CIFAR-10/ImageNet评估
4. **完善Zero-Order Search** - 提取`generate_neighbors()`实现
5. **创建实验脚本** - 基于现有脚本创建统一配置


