#!/bin/bash
# 模型下载脚本
# 在远程服务器上运行此脚本以下载所需的预训练模型

set -e

# 创建模型存储目录
MODELS_DIR="models"
mkdir -p "${MODELS_DIR}/cifar10"
mkdir -p "${MODELS_DIR}/imagenet64"

echo "开始下载模型..."

# ============================================
# CIFAR-10 模型下载
# ============================================
echo "================================"
echo "下载 CIFAR-10 模型"
echo "================================"

# 选项1: OpenAI DDPM CIFAR-10
# 如果需要，可以手动下载：
# wget -O "${MODELS_DIR}/cifar10/ddpm.pt" <URL>

# 选项2: EDM CIFAR-10
# 从 https://github.com/NVlabs/edm 下载
echo "EDM CIFAR-10 模型请从以下位置下载："
echo "  https://github.com/NVlabs/edm"
echo "  下载后放置到: ${MODELS_DIR}/cifar10/edm_cifar10.pt"

# 选项3: score-sde CIFAR-10
echo "score-sde CIFAR-10 模型请从以下位置下载："
echo "  https://github.com/yang-song/score_sde"
echo "  下载后放置到: ${MODELS_DIR}/cifar10/score_sde_cifar10.ckpt"

# ============================================
# ImageNet-64 模型下载
# ============================================
echo ""
echo "================================"
echo "下载 ImageNet-64 模型"
echo "================================"

# SiT (Scalable Interpolant Transformer)
echo "SiT ImageNet-64 模型请从以下位置下载："
echo "  https://github.com/saic-fi/tt-scale"
echo "  或参考: https://github.com/XiangchengZhang/Diffusion-inference-scaling"
echo "  下载后放置到: ${MODELS_DIR}/imagenet64/sit_b.pt (或 sit_l.pt, sit_xl.pt)"

# FLUX 模型（如果使用）
echo ""
echo "FLUX 模型（如果使用）请从："
echo "  https://huggingface.co/black-forest-labs/FLUX.1-dev"
echo "  下载后放置到: ${MODELS_DIR}/imagenet64/flux/"

# ============================================
# 预训练分类器（用于Verifier）
# ============================================
echo ""
echo "================================"
echo "下载预训练分类器（Verifier）"
echo "================================"

# CIFAR-10 分类器
echo "CIFAR-10 分类器："
echo "  可以使用 torchvision.models 中的预训练模型"
echo "  python scripts/download_classifiers.py"

# ImageNet 分类器（InceptionV3）
echo "ImageNet InceptionV3 分类器："
echo "  可以使用 torchvision.models.inception_v3"
echo "  或下载预训练的: ${MODELS_DIR}/imagenet64/inception_v3.pt"

echo ""
echo "================================"
echo "模型下载指南完成"
echo "================================"
echo ""
echo "注意："
echo "1. 某些模型可能需要从Hugging Face下载，请使用 huggingface-cli"
echo "2. 某些模型文件较大，请确保有足够的存储空间"
echo "3. 下载后请验证文件完整性"


