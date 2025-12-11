# src/verifiers/scorer_verifier.py

"""
Scorer Verifier - 适配 diffusion-tts 的 scorer 到我们的 BaseVerifier 接口

支持三种 scorer:
- BrightnessScorer: 计算图像亮度
- CompressibilityScorer: 计算图像可压缩性
- ImageNetScorer: 使用 ImageNet 分类器计算类别概率
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import urllib.request
import io
from typing import Optional, Union
import sys
from pathlib import Path

from .base_verifier import BaseVerifier
from ..utils.nfe_counter import NFECounter


class ScorerVerifier(BaseVerifier):
    """
    适配 diffusion-tts 的 scorer 到 BaseVerifier 接口
    """
    def __init__(
        self,
        scorer_type: str = "brightness",  # "brightness", "compressibility", "imagenet"
        model_path: Optional[str] = None,
        device: str = "cuda",
        nfe_counter: Optional[NFECounter] = None,
        # CompressibilityScorer 参数
        quality: int = 80,
        min_size: int = 0,
        max_size: int = 3000,
        # ImageNetScorer 参数
        image_size: int = 64,
        class_labels: Optional[torch.Tensor] = None,  # For ImageNet scorer
    ):
        """
        初始化 Scorer Verifier
        
        Args:
            scorer_type: scorer类型 ("brightness", "compressibility", "imagenet")
            model_path: 模型路径（对于ImageNet scorer，会自动下载）
            device: 设备
            nfe_counter: NFE计数器
            quality: JPEG质量（用于CompressibilityScorer）
            min_size, max_size: 压缩大小范围（用于CompressibilityScorer）
            image_size: 图像尺寸（用于ImageNetScorer）
            class_labels: 类别标签（用于ImageNetScorer，one-hot或class indices）
        """
        super().__init__(model_path, device)
        self.scorer_type = scorer_type
        self.quality = quality
        self.min_size = min_size
        self.max_size = max_size
        self.image_size = image_size
        self.class_labels = class_labels
        self.set_nfe_counter(nfe_counter)
        
        # 初始化对应的scorer
        if scorer_type == "brightness":
            self.scorer = BrightnessScorer()
        elif scorer_type == "compressibility":
            self.scorer = CompressibilityScorer(quality=quality, min_size=min_size, max_size=max_size)
        elif scorer_type == "imagenet":
            self.scorer = ImageNetScorer(image_size=image_size)
            if model_path:
                self.scorer.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise ValueError(f"Unknown scorer_type: {scorer_type}. Must be one of: brightness, compressibility, imagenet")
        
        # 移动到设备
        if hasattr(self.scorer, 'model'):
            self.scorer.model = self.scorer.model.to(device)
            self.scorer.model.eval()
    
    def load_model(self, path: str):
        """加载模型（主要用于ImageNet scorer）"""
        if self.scorer_type == "imagenet" and hasattr(self.scorer, 'model'):
            state_dict = torch.load(path, map_location=self.device)
            self.scorer.model.load_state_dict(state_dict)
            self.scorer.model.eval()
        else:
            print(f"Warning: load_model not needed for scorer_type={self.scorer_type}")
    
    def score(self, images: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算样本分数
        
        Args:
            images: 图像张量 [B, C, H, W]，值域应为[0, 1]或[0, 255]（uint8）
            class_labels: 类别标签（用于ImageNet scorer），如果为None则使用self.class_labels
        
        Returns:
            分数 [B]，分数越高表示质量越好
        """
        # 使用提供的class_labels或默认的
        labels_to_use = class_labels if class_labels is not None else self.class_labels
        
        # 转换图像格式
        if images.dtype == torch.uint8:
            # 如果是uint8
            if self.scorer_type == "compressibility":
                images_processed = images  # CompressibilityScorer需要uint8
            else:
                images_processed = images.float() / 255.0  # Brightness和ImageNet需要float32 [0, 1]
        else:
            # 如果是float，假设在[0, 1]范围
            if self.scorer_type == "compressibility":
                # CompressibilityScorer需要uint8，确保值域在[0, 1]然后转换
                images_processed = (images.clamp(0, 1) * 255).to(torch.uint8)
            else:
                images_processed = images.clamp(0, 1)  # Brightness和ImageNet需要[0, 1]
        
        # 创建timesteps（某些scorer需要，但这里我们传0表示完全去噪的图像）
        timesteps = torch.zeros(images_processed.shape[0], device=self.device)
        
        # 调用scorer
        with torch.no_grad():
            # 计数NFE（verifier调用也算NFE）
            if self.nfe_counter is not None:
                self.nfe_counter.add(1)
            if self.scorer_type == "imagenet":
                # ImageNetScorer需要class_labels
                if labels_to_use is None:
                    raise ValueError("class_labels must be provided for ImageNetScorer")
                scores = self.scorer(images_processed, labels_to_use, timesteps)
            else:
                # BrightnessScorer和CompressibilityScorer不需要class_labels
                scores = self.scorer(images_processed, None, timesteps)
        
        return scores.to(self.device)


class BrightnessScorer(nn.Module):
    """亮度Scorer - 计算感知亮度"""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.eval()
    
    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts, timesteps: torch.Tensor) -> torch.Tensor:
        """
        计算亮度分数
        
        Args:
            images: [B, C, H, W]，值域[0, 1]
            prompts: 未使用（保持接口一致）
            timesteps: 未使用（保持接口一致）
        
        Returns:
            亮度分数 [B]，值域[0, 1]
        """
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        # 确保值域在[0, 1]并处理NaN/Inf
        images = images.clamp(0, 1)
        if torch.isnan(images).any() or torch.isinf(images).any():
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 应用感知亮度公式: 0.2126*R + 0.7152*G + 0.0722*B
        if images.size(1) == 3:  # RGB
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
            luminance = (images * weights).sum(dim=1).mean(dim=(1, 2))
        else:
            # 如果不是RGB，使用平均值
            luminance = images.mean(dim=(1, 2, 3))
        
        # 确保值域在[0, 1]并处理NaN/Inf
        luminance = torch.clamp(luminance, 0.0, 1.0)
        luminance = torch.nan_to_num(luminance, nan=0.0, posinf=1.0, neginf=0.0)
        return luminance


class CompressibilityScorer(nn.Module):
    """可压缩性Scorer - 基于JPEG压缩大小"""
    def __init__(self, quality=80, min_size=0, max_size=3000, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.quality = quality
        self.min_size = min_size
        self.max_size = max_size
        self.eval()
    
    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts, timesteps: torch.Tensor) -> torch.Tensor:
        """
        计算可压缩性分数
        
        Args:
            images: [B, C, H, W]，值域[0, 255]（uint8）或[0, 1]（float）
            prompts: 未使用
            timesteps: 未使用
        
        Returns:
            可压缩性分数 [B]，值域[0, 1]，越高表示越可压缩
        """
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:  # batch
                scores = torch.tensor([
                    self._calculate_score(img.cpu().numpy()) 
                    for img in images
                ], device=images.device)
            else:  # single image
                scores = torch.tensor([self._calculate_score(images.cpu().numpy())], device=images.device)
        else:
            raise TypeError(f"Expected torch.Tensor, got {type(images)}")
        
        return scores
    
    def _calculate_score(self, image: np.ndarray) -> float:
        """计算单个图像的可压缩性分数"""
        # 处理不同的图像格式
        if image.ndim == 3:
            if image.shape[0] == 1 or image.shape[0] == 3:  # CHW
                image = np.transpose(image, (1, 2, 0))  # HWC
            if image.shape[2] == 1:  # 灰度
                image = image.squeeze(2)
        
        # 确保是uint8格式
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 确保图像值域正确
        image = np.clip(image, 0, 255)
        
        # JPEG压缩
        try:
            buffer = io.BytesIO()
            img = Image.fromarray(image)
            img.save(buffer, format="JPEG", quality=self.quality)
            compressed_size = len(buffer.getvalue())
            
            # 调试：检查压缩大小
            if compressed_size == 0:
                print(f"Warning: Compressed size is 0 for image with stats: min={image.min()}, max={image.max()}, dtype={image.dtype}, shape={image.shape}")
                return 0.0
        except Exception as e:
            print(f"Warning: Error compressing image: {e}, image stats: min={image.min()}, max={image.max()}, dtype={image.dtype}, shape={image.shape}")
            return 0.0
        
        # 归一化到[0, 1]，越小（越可压缩）分数越高
        if self.max_size > self.min_size:
            normalized_score = 1.0 - min(1.0, max(0.0, 
                (compressed_size - self.min_size) / (self.max_size - self.min_size)
            ))
        else:
            normalized_score = 0.0
        
        # 调试：检查分数
        if normalized_score == 0.0 and compressed_size > self.max_size:
            print(f"Debug: Compressibility score is 0.0 for compressed_size={compressed_size} (max_size={self.max_size})")
        
        return float(normalized_score)


class ImageNetScorer(nn.Module):
    """ImageNet分类器Scorer - 计算目标类别的概率"""
    def __init__(self, image_size=64, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.image_size = image_size
        self.eval()
        
        # 下载并加载模型
        url = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt"
        cache_dir = os.path.expanduser("~/.cache/imagenet_classifier")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "64x64_classifier.pt")
        
        if not os.path.exists(model_path):
            print(f"Downloading ImageNet classifier from {url}...")
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to {model_path}")
        
        # 创建模型
        self.model = self._create_classifier(
            image_size=image_size,
            classifier_use_fp16=False,
            classifier_width=128,
            classifier_depth=4,
            classifier_attention_resolutions="32,16,8",
            classifier_use_scale_shift_norm=True,
            classifier_resblock_updown=True,
            classifier_pool="attention"
        )
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def _create_classifier(
        self,
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    ):
        """创建分类器模型"""
        # 需要导入 unet 模块
        # 由于 unet 在 diffusion-tts 中，我们需要复制相关代码或导入
        # 这里先创建一个占位符，实际使用时需要从 diffusion-tts 导入
        
        # 尝试从 diffusion-tts 导入
        edm_dir = Path(__file__).parent.parent.parent / "code_repos" / "diffusion-tts" / "edm"
        if edm_dir.exists():
            sys.path.insert(0, str(edm_dir))
            try:
                from unet import EncoderUNetModel  # type: ignore
            except ImportError:
                raise ImportError(
                    f"Could not import EncoderUNetModel from {edm_dir}/unet.py. "
                    "Please ensure diffusion-tts is cloned in code_repos/"
                )
        else:
            raise FileNotFoundError(
                f"EDM directory not found at {edm_dir}. "
                "Please clone diffusion-tts repository to code_repos/"
            )
        
        # 确定channel_mult
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"Unsupported image size: {image_size}")
        
        # 确定attention resolutions
        attention_ds = []
        for res in classifier_attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        
        return EncoderUNetModel(
            image_size=image_size,
            in_channels=3,
            model_channels=classifier_width,
            out_channels=1000,
            num_res_blocks=classifier_depth,
            attention_resolutions=tuple(attention_ds),
            channel_mult=channel_mult,
            use_fp16=classifier_use_fp16,
            num_head_channels=64,
            use_scale_shift_norm=classifier_use_scale_shift_norm,
            resblock_updown=classifier_resblock_updown,
            pool=classifier_pool,
        )
    
    @torch.no_grad()
    def __call__(self, images: torch.Tensor, class_labels: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        计算ImageNet分类器分数
        
        Args:
            images: [B, C, H, W]，值域[0, 1]
            class_labels: [B, num_classes] (one-hot) 或 [B] (class indices)
            timesteps: [B]，用于分类器（通常传0表示完全去噪）
        
        Returns:
            目标类别的概率 [B]
        """
        device = next(self.parameters()).device
        
        # 处理图像
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        processed_images = images.to(device).clamp(0, 1)
        # 处理NaN/Inf
        if torch.isnan(processed_images).any() or torch.isinf(processed_images).any():
            processed_images = torch.nan_to_num(processed_images, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 处理timesteps
        timesteps = timesteps.to(device)
        
        # 模型预测
        logits = self.model(processed_images, timesteps)
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # 获取目标类别
        if class_labels.dim() > 1:  # one-hot
            # 检查是否有非零元素
            nonzero_mask = class_labels.sum(dim=1) > 0
            if not nonzero_mask.all():
                print(f"Warning: Found {(~nonzero_mask).sum().item()}/{len(nonzero_mask)} class_labels with all zeros in ImageNetScorer")
            target_classes = torch.argmax(class_labels, dim=1)
        else:  # class indices
            target_classes = class_labels
        
        # 确保target_classes在有效范围内
        target_classes = target_classes.to(device).long()
        target_classes = torch.clamp(target_classes, 0, probs.size(1) - 1)
        
        # 提取目标类别的概率
        batch_indices = torch.arange(probs.size(0), device=device)
        scores = probs[batch_indices, target_classes]
        
        # 处理NaN/Inf
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 如果class_labels全为0，分数可能不正确，返回0
        if class_labels.dim() > 1:
            zero_mask = class_labels.sum(dim=1) == 0
            if zero_mask.any():
                scores[zero_mask] = 0.0
        
        return scores

