"""
基础模型接口
所有扩散模型都应实现此接口
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
from ..utils.nfe_counter import NFECounter, NFECounterMixin


class BaseDiffusionModel(NFECounterMixin, ABC):
    """
    扩散模型基类
    
    所有扩散模型都应继承此类并实现抽象方法
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        初始化模型
        
        Args:
            model_path: 模型checkpoint路径
            device: 设备（cuda/cpu）
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    @abstractmethod
    def load_model(self, path: str):
        """加载模型checkpoint"""
        pass
    
    @abstractmethod
    def denoise_step(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        执行一步去噪
        
        Args:
            x_t: 当前时刻的噪声图像 [B, C, H, W]
            t: 时间步
        
        Returns:
            x_{t-1}: 去噪后的图像
        """
        pass
    
    @abstractmethod
    def sample_noise(self, batch_size: int, image_size: int) -> torch.Tensor:
        """
        采样初始噪声
        
        Args:
            batch_size: batch大小
            image_size: 图像尺寸
        
        Returns:
            初始噪声 [B, C, H, W]
        """
        pass
    
    def sample(
        self,
        batch_size: int,
        num_steps: int,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        标准采样流程
        
        Args:
            batch_size: batch大小
            num_steps: 采样步数
            nfe_counter: NFE计数器
            **kwargs: 其他参数
        
        Returns:
            生成的样本 [B, C, H, W]
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        self.set_nfe_counter(nfe_counter)
        
        # 采样初始噪声
        x = self.sample_noise(batch_size, self.image_size)
        
        # 逐步去噪
        for t in range(num_steps - 1, -1, -1):
            with nfe_counter.count():
                x = self.denoise_step(x, t)
        
        return x
    
    @property
    @abstractmethod
    def image_size(self) -> int:
        """返回模型训练的图像尺寸"""
        pass
    
    @property
    @abstractmethod
    def num_channels(self) -> int:
        """返回图像通道数"""
        pass


