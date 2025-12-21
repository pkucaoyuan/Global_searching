# 项目设置指南

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/pkucaoyuan/Global_searching.git
cd Global_searching
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 准备外部代码库

本项目需要克隆外部代码库以获取EDM模型和相关组件：

```bash
# 创建目录
mkdir -p code_repos
cd code_repos

# 克隆 diffusion-tts（必需，用于EDM模型）
git clone https://github.com/rvignav/diffusion-tts.git
```

**重要**: `code_repos/` 目录不会被提交到Git仓库（已在`.gitignore`中排除），需要手动克隆。

### 4. 运行实验

参考实验步骤说明，或直接运行：

```bash
python scripts/run_diffusion_tts_experiment.py \
    --config configs/imagenet64_diffusion_tts.yaml
```

## 项目结构

```
Global_searching/
├── src/                    # 核心代码
│   ├── models/            # 模型包装器
│   ├── verifiers/         # Verifier实现
│   ├── search/            # Local和Global Search方法
│   ├── pipeline/          # 采样流程
│   └── evaluation/        # 评估指标
├── configs/               # 配置文件
├── scripts/               # 实验脚本
├── code_repos/            # 外部代码库（需手动克隆）
├── requirements.txt       # Python依赖
└── README.md             # 项目说明
```

## 外部依赖说明

### Diffusion-TTS (必需)
- **用途**: EDM模型wrapper、Best-of-N/Zero-Order/ε-greedy搜索方法
- **位置**: `code_repos/diffusion-tts/`
- **克隆命令**: `git clone https://github.com/rvignav/diffusion-tts.git code_repos/diffusion-tts`

### 其他可选代码库
- `tt-scale-flux`: Random Search和Zero-Order Search的参考实现
- `Diffusion-inference-scaling`: ImageNet实验脚本和BFS/DFS实现

## 注意事项

1. **模型文件**: 模型checkpoint会自动下载，或使用`scripts/download_models.sh`
2. **GPU要求**: ImageNet-64实验建议使用A100或类似GPU
3. **内存**: ε-greedy和Zero-Order方法需要较多内存
4. **路径**: 确保`code_repos/diffusion-tts/edm/`目录存在，代码会动态导入其中的模块

## 常见问题

**Q: 找不到`dnnlib`模块？**  
A: 确保已克隆`diffusion-tts`到`code_repos/diffusion-tts/`

**Q: 模型下载失败？**  
A: 可以手动下载模型文件到`data/checkpoints/`目录

**Q: 内存不足？**  
A: 减小配置文件中的`batch_size`参数




