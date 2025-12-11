# 快速开始指南

## 🚀 立即开始的3步

### Step 1: 代码库准备（今天完成）

```bash
# 创建项目目录结构
mkdir -p code_repos
mkdir -p models
mkdir -p results
mkdir -p configs

# Clone核心代码库
cd code_repos
git clone https://github.com/sayakpaul/tt-scale-flux.git
git clone https://github.com/XiangchengZhang/Diffusion-inference-scaling.git
git clone https://github.com/harveymannering/NoiseLevelGuidance.git
git clone https://github.com/rvignav/diffusion-tts.git
```

### Step 2: 第一个实验（明天开始）

**目标：** 在CIFAR-10上运行pure sampling baseline

**最小可行代码结构：**
```
your_project/
├── src/
│   ├── sampling.py          # 标准采样流程
│   ├── evaluation.py        # FID/IS计算
│   └── nfe_counter.py       # NFE计数工具
├── configs/
│   └── cifar10_config.yaml  # 实验配置
└── scripts/
    └── run_baseline.py      # 主实验脚本
```

**第一个脚本示例（run_baseline.py骨架）：**
```python
import torch
from src.sampling import standard_sampling
from src.evaluation import compute_fid_is
from src.nfe_counter import NFECounter

def run_pure_sampling_baseline(model, steps_list=[25, 50, 100, 200]):
    results = []
    for steps in steps_list:
        counter = NFECounter()
        samples = standard_sampling(model, steps=steps, nfe_counter=counter)
        fid, is_score = compute_fid_is(samples)
        results.append({
            'steps': steps,
            'nfe': counter.total_nfe,
            'fid': fid,
            'is': is_score
        })
        print(f"Steps: {steps}, NFE: {counter.total_nfe}, FID: {fid:.4f}, IS: {is_score:.4f}")
    return results
```

### Step 3: 验证假设（第1周内）

运行完pure sampling baseline后，检查：
- [ ] FID/IS曲线是否在某个点后变平？
- [ ] 如果是，说明假设成立，可以继续做search方法
- [ ] 如果不是，需要重新检查模型或评估方法

---

## 📋 第一周具体任务清单

### Day 1-2: 环境搭建
- [ ] 安装PyTorch和相关依赖
- [ ] 下载CIFAR-10预训练模型
- [ ] 测试标准采样流程能否跑通
- [ ] 实现基础的FID计算（可以使用`pytorch-fid`库）

### Day 3-4: Pure Sampling Baseline
- [ ] 实现多步数采样脚本
- [ ] 运行[25, 50, 100, 200]步实验
- [ ] 收集FID/IS数据
- [ ] 绘制baseline曲线

### Day 5-7: 第一个Search方法
选择最简单的开始：
- [ ] 实现Random Search（比NLG更直观）
- [ ] 实现简单的verifier（classifier logit）
- [ ] 运行N=4的Random Search实验
- [ ] 对比结果

---

## 🎯 成功标准（第1周结束）

你应该能够回答：
1. ✅ Pure sampling在CIFAR-10上的scaling曲线是什么样的？
2. ✅ Random Search在相同NFE下是否比pure sampling更好？
3. ✅ 如果更好，提升了多少？（FID改善多少？）

---

## 💡 遇到问题？

### 问题1: 找不到合适的预训练模型
**解决方案：**
- CIFAR-10: 使用`score-sde`官方repo的checkpoint
- 或者：用`denoising-diffusion-pytorch`快速训练一个小模型做验证

### 问题2: FID计算太慢
**解决方案：**
- 先用少量样本（比如1k）快速验证pipeline
- 最终评估再用50k样本

### 问题3: 代码库依赖冲突
**解决方案：**
- 不要直接运行代码库的代码，而是提取关键函数
- 创建自己的统一接口，调用不同代码库的功能模块

---

## 🔗 有用的资源

### 预训练模型
- CIFAR-10 DDPM: [OpenAI官方](https://github.com/openai/improved-diffusion)
- EDM模型: [EDM官方](https://github.com/NVlabs/edm)

### 评估工具
- FID: `pip install pytorch-fid`
- IS: 可以使用`pytorch-fid`或其他实现

### 参考实现
- 最简单的采样实现：[denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

---

## 📞 下一步

完成第一周任务后，继续：
1. 查看`experiment_plan.md`了解完整规划
2. 使用`experiment_checklist.md`追踪进度
3. 开始Phase 1的其他方法（NLG, ZO-N）


