# 模型下载详细指南

## 📥 CIFAR-10 模型下载

### 选项1: OpenAI DDPM

```bash
# 从官方仓库下载
git clone https://github.com/openai/improved-diffusion.git
cd improved-diffusion
# 下载CIFAR-10模型checkpoint
# 通常需要从Google Drive或指定URL下载
# 放置到: models/cifar10/ddpm_cifar10.pt
```

**官方仓库**: https://github.com/openai/improved-diffusion

### 选项2: EDM (NVIDIA)

```bash
# 从NVIDIA EDM仓库下载
git clone https://github.com/NVlabs/edm.git
cd edm
# 下载CIFAR-10预训练模型
# 放置到: models/cifar10/edm_cifar10.pkl
```

**官方仓库**: https://github.com/NVlabs/edm  
**模型URL**: 通常从项目README或模型zoo获取

### 选项3: score-sde

```bash
# 从score-sde仓库下载
git clone https://github.com/yang-song/score_sde.git
cd score_sde
# 下载CIFAR-10模型
# 放置到: models/cifar10/score_sde_cifar10.ckpt
```

**官方仓库**: https://github.com/yang-song/score_sde

### 选项4: 使用Hugging Face (如果可用)

```bash
# 如果模型已上传到Hugging Face
huggingface-cli download <model_id> --local-dir models/cifar10/
```

---

## 📥 ImageNet-64/256 模型下载

### SiT (Scalable Interpolant Transformer)

```bash
# 从Diffusion Inference Scaling仓库
git clone https://github.com/XiangchengZhang/Diffusion-inference-scaling.git
cd Diffusion-inference-scaling

# 查看imagenet/目录中的脚本
# 通常包含模型下载链接
# 下载SiT-B/L/XL模型
# 放置到: models/imagenet64/sit_b.pt (或 sit_l.pt, sit_xl.pt)
```

**官方仓库**: https://github.com/XiangchengZhang/Diffusion-inference-scaling

### FLUX (如果使用)

```bash
# 使用Hugging Face Hub
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir models/imagenet64/flux/
```

**模型页面**: https://huggingface.co/black-forest-labs/FLUX.1-dev

---

## 📥 分类器模型下载

### CIFAR-10 分类器

**选项1: 使用torchvision预训练模型（需要fine-tune）**
```python
# 使用 download_classifiers.py
python scripts/download_classifiers.py cifar10
```

**选项2: 使用专门训练的CIFAR-10分类器**
```bash
# 从PyTorch CIFAR10项目下载
git clone https://github.com/huyvnphan/PyTorch_CIFAR10.git
# 下载训练好的分类器
# 放置到: models/cifar10/cifar10_classifier.pth
```

### ImageNet 分类器 (InceptionV3)

```python
# 使用 download_classifiers.py
python scripts/download_classifiers.py imagenet
```

这会自动下载torchvision的预训练InceptionV3模型。

---

## 📊 FID统计文件

### CIFAR-10 FID统计

需要预先计算CIFAR-10数据集的FID统计：

```python
# 使用pytorch-fid计算
from pytorch_fid import fid_score

# 计算参考数据集的统计
fid_score.calculate_fid_given_paths(
    ['path/to/cifar10/train/images'],
    batch_size=50,
    device='cuda',
    dims=2048,
    save_stats='data/cifar10/fid_stats.npz'
)
```

或者从已有项目下载预计算的统计文件。

---

## 🔗 有用的链接

1. **OpenAI Improved Diffusion**: https://github.com/openai/improved-diffusion
2. **NVIDIA EDM**: https://github.com/NVlabs/edm
3. **Score SDE**: https://github.com/yang-song/score_sde
4. **Diffusion Inference Scaling**: https://github.com/XiangchengZhang/Diffusion-inference-scaling
5. **tt-scale-flux**: https://github.com/sayakpaul/tt-scale-flux

---

## ⚠️ 注意事项

1. **文件大小**: 模型文件通常很大（几百MB到几GB），确保有足够存储空间
2. **下载速度**: 某些模型可能需要从Google Drive下载，速度较慢
3. **格式**: 不同项目的checkpoint格式可能不同，需要适配加载代码
4. **许可证**: 注意检查模型的使用许可证
5. **版本兼容**: 确保模型与你的PyTorch版本兼容

---

## ✅ 验证下载

下载后，可以运行简单的验证脚本：

```python
import torch

# 验证模型文件
model_path = "models/cifar10/ddpm_cifar10.pt"
checkpoint = torch.load(model_path, map_location='cpu')
print(f"模型键: {checkpoint.keys()}")
print(f"模型大小: {sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))}")
```


