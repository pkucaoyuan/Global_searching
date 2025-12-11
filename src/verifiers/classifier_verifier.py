"""
基于分类器的Verifier
使用预训练分类器的log概率作为reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_verifier import BaseVerifier


class ClassifierVerifier(BaseVerifier):
    """
    分类器Verifier
    
    对于CIFAR-10：使用分类器的log p(class|x)作为reward
    对于ImageNet：类似，可以使用class logit或entropy
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 10,
        device: str = "cuda",
        use_log_prob: bool = True,
    ):
        """
        初始化分类器Verifier
        
        Args:
            model_path: 分类器模型路径
            num_classes: 类别数
            device: 设备
            use_log_prob: 是否使用log概率（True）还是logit（False）
        """
        self.num_classes = num_classes
        self.use_log_prob = use_log_prob
        super().__init__(model_path, device)
    
    def load_model(self, path: str):
        """加载分类器模型"""
        # 这里需要根据实际使用的分类器架构来加载
        # 示例：如果是ResNet
        # from torchvision.models import resnet18
        # self.model = resnet18(num_classes=self.num_classes)
        # self.model.load_state_dict(torch.load(path))
        
        # 暂时使用占位符，实际使用时需要根据具体模型实现
        if path:
            print(f"加载分类器模型: {path}")
            # TODO: 实现实际的模型加载逻辑
            # self.model = ...
            # self.model.to(self.device)
            # self.model.eval()
        else:
            print("警告: 未提供分类器模型路径，使用随机分数")
            self.model = None
    
    def score(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算分类器分数
        
        Args:
            images: [B, C, H, W]，值域应为[0, 1]
        
        Returns:
            scores: [B]，分数越高越好
        """
        if self.model is None:
            # 返回随机分数作为占位符
            return torch.rand(images.size(0), device=images.device)
        
        # 确保图像值域正确
        if images.max() > 1.0:
            images = images / 255.0
        
        # 前向传播
        with torch.no_grad():
            logits = self.model(images)
            
            if self.use_log_prob:
                # 使用最大类别的log概率
                probs = F.softmax(logits, dim=1)
                scores = torch.log(probs.max(dim=1)[0] + 1e-8)
            else:
                # 使用最大logit
                scores = logits.max(dim=1)[0]
        
        return scores


