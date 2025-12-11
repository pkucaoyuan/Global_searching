#!/usr/bin/env python3
"""
下载预训练分类器（用于Verifier）
这些模型较小，可以直接从torchvision下载
"""

import torch
import torchvision.models as models
from pathlib import Path
import sys

def download_cifar10_classifier():
    """下载CIFAR-10分类器"""
    print("下载CIFAR-10分类器...")
    models_dir = Path("models/cifar10")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用ResNet作为CIFAR-10分类器
    # 注意：torchvision的ResNet是为ImageNet训练的，需要fine-tune到CIFAR-10
    # 这里提供一个基础版本，实际使用时可能需要fine-tune
    model = models.resnet18(pretrained=True)
    torch.save(model.state_dict(), models_dir / "classifier_resnet18.pth")
    print(f"✓ CIFAR-10分类器已保存到: {models_dir / 'classifier_resnet18.pth'}")
    
    # 更好的选择：使用专门为CIFAR-10训练的模型
    print("\n提示：建议使用专门为CIFAR-10训练的模型，例如：")
    print("  - https://github.com/huyvnphan/PyTorch_CIFAR10")
    print("  下载后放置到: models/cifar10/cifar10_classifier.pth")

def download_imagenet_classifier():
    """下载ImageNet分类器（InceptionV3）"""
    print("\n下载ImageNet分类器（InceptionV3）...")
    models_dir = Path("models/imagenet64")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # InceptionV3用于FID/IS计算
    model = models.inception_v3(pretrained=True, transform_input=False)
    torch.save(model.state_dict(), models_dir / "inception_v3.pth")
    print(f"✓ InceptionV3已保存到: {models_dir / 'inception_v3.pth'}")

if __name__ == "__main__":
    print("=" * 50)
    print("下载预训练分类器")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "cifar10":
            download_cifar10_classifier()
        elif sys.argv[1] == "imagenet":
            download_imagenet_classifier()
        else:
            print("用法: python download_classifiers.py [cifar10|imagenet|all]")
    else:
        # 默认下载全部
        download_cifar10_classifier()
        download_imagenet_classifier()
    
    print("\n✓ 完成！")


