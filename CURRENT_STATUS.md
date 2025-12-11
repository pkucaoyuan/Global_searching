# 当前项目状态

## ✅ 已完成的工作

### Phase 0: 项目结构搭建

1. **项目目录结构** ✓
   - 创建了清晰的模块化目录结构
   - 分离了模型、verifier、search、evaluation等模块

2. **基础框架代码** ✓
   - `BaseDiffusionModel`: 扩散模型基类接口
   - `BaseVerifier`: Verifier基类接口
   - `BaseSearch`: Search方法基类接口
   - `NFECounter`: 统一的NFE计数工具
   - `Config`: 配置管理类

3. **实验脚本框架** ✓
   - `run_baseline.py`: 基线实验脚本模板
   - `download_models.sh`: 模型下载脚本
   - `download_classifiers.py`: 分类器下载脚本

4. **配置文件** ✓
   - `cifar10_baseline.yaml`: CIFAR-10实验配置模板

5. **文档** ✓
   - README.md: 项目总览
   - experiment_plan.md: 详细实验计划
   - experiment_checklist.md: 实验检查清单
   - quick_start.md: 快速开始指南
   - PROJECT_STRUCTURE.md: 项目结构说明
   - download_models_detailed.md: 模型下载详细指南

---

## 🚧 下一步工作（Phase 1）

### 1.1 实现具体的模型加载

**任务**:
- [ ] 实现DDPM模型加载器 (`src/models/ddpm_model.py`)
- [ ] 实现EDM模型加载器 (可选)
- [ ] 创建模型工厂函数，根据配置自动选择模型

**需要参考的代码库**:
- OpenAI improved-diffusion
- NVIDIA EDM
- score-sde

### 1.2 实现评估指标

**任务**:
- [ ] 完善FID计算 (`src/evaluation/metrics.py`)
- [ ] 完善IS计算
- [ ] 实现图像保存和加载工具
- [ ] 实现FID统计文件的生成

**依赖**:
- pytorch-fid库
- InceptionV3模型

### 1.3 实现Pure Sampling Baseline

**任务**:
- [ ] 完善`run_baseline.py`脚本
- [ ] 实现完整的采样流程
- [ ] 实现批量采样和评估
- [ ] 添加结果可视化和保存

### 1.4 实现Verifier

**任务**:
- [ ] 完善`ClassifierVerifier`的实现
- [ ] 实现CIFAR-10分类器加载
- [ ] 测试verifier的score计算

---

## 📝 代码实现指南

### 实现DDPM模型

```python
# src/models/ddpm_model.py
from .base_model import BaseDiffusionModel
import torch
import torch.nn as nn

class DDPMModel(BaseDiffusionModel):
    def load_model(self, path: str):
        # 加载checkpoint
        # 初始化UNet等组件
        pass
    
    def denoise_step(self, x_t, t):
        # 实现DDPM的去噪步骤
        pass
    
    # ... 其他方法
```

### 实现Random Search

```python
# src/search/random_search.py
from .base_search import BaseSearch
import torch

class RandomSearch(BaseSearch):
    def search(self, initial_noise=None, batch_size=1, num_steps=50, 
               nfe_counter=None, num_candidates=4, **kwargs):
        # 采样多个初始噪声
        # 并行采样多条轨迹
        # 使用verifier选择最优
        pass
```

---

## 🔍 需要从其他代码库提取的内容

### 从 improved-diffusion
- DDPM模型架构定义
- Checkpoint加载逻辑
- 采样流程实现

### 从 EDM
- EDM模型架构
- 采样schedule

### 从 score-sde
- Score-based模型实现
- SDE采样器

### 从 Diffusion-inference-scaling
- ImageNet模型加载
- BFS/DFS/MCMC实现
- 实验脚本结构

### 从 NoiseLevelGuidance
- NLG算法实现
- 噪声refinement逻辑

---

## ⚠️ 注意事项

1. **不要直接复制代码**: 提取关键逻辑，适配到我们的接口
2. **保持接口统一**: 所有模型/方法都使用相同的基类接口
3. **NFE计数**: 确保所有操作都正确计数NFE
4. **配置驱动**: 使用配置文件管理所有参数
5. **可复现性**: 设置随机种子，保存实验配置

---

## 📦 文件组织建议

在实现具体功能时，建议按以下顺序：

1. **先实现模型加载** (最重要，其他都依赖它)
2. **实现评估工具** (用于验证结果)
3. **实现Pure Sampling** (最简单的baseline)
4. **实现Verifier** (Search方法需要)
5. **实现Search方法** (Random, ZO, NLG等)

---

## 🎯 第一个可运行的实验目标

目标：在CIFAR-10上运行pure sampling baseline，得到：
- 不同步数（25, 50, 100, 200）的FID/IS曲线
- 验证"纯加步数提升有限"的假设
- 建立后续实验的对比基准

---

## 📞 获取帮助

如果遇到问题：
1. 查看对应代码库的README和文档
2. 查看代码注释和示例
3. 参考`experiment_plan.md`中的实验设计
4. 检查配置文件和代码接口是否匹配


