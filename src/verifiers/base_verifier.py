"""
Verifier基类
用于评估生成样本的质量
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from ..utils.nfe_counter import NFECounterMixin


class BaseVerifier(NFECounterMixin, ABC):
    """
    Verifier基类
    
    Verifier用于评估生成样本的质量，可以作为reward信号
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        初始化Verifier
        
        Args:
            model_path: 模型路径
            device: 设备
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    @abstractmethod
    def load_model(self, path: str):
        """加载模型"""
        pass
    
    @abstractmethod
    def score(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算样本分数
        
        Args:
            images: 图像张量 [B, C, H, W]，值域应为[0, 1]
        
        Returns:
            分数 [B]，分数越高表示质量越好
        """
        pass
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """支持直接调用"""
        return self.score(images)


