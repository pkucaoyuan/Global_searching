"""
Search方法基类
定义统一的search接口
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import torch
from ..models.base_model import BaseDiffusionModel
from ..verifiers.base_verifier import BaseVerifier
from ..utils.nfe_counter import NFECounter


class BaseSearch(ABC):
    """
    Search方法基类
    
    所有search方法（Random, ZO, NLG等）都应继承此类
    """
    
    def __init__(
        self,
        model: BaseDiffusionModel,
        verifier: BaseVerifier,
        nfe_budget: int = 0,
    ):
        """
        初始化Search方法
        
        Args:
            model: 扩散模型
            verifier: Verifier
            nfe_budget: 可用于search的NFE预算
        """
        self.model = model
        self.verifier = verifier
        self.nfe_budget = nfe_budget
    
    @abstractmethod
    def search(
        self,
        initial_noise: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_steps: int = 50,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        执行search并生成样本
        
        Args:
            initial_noise: 初始噪声（可选，如果None则随机采样）
            batch_size: batch大小
            num_steps: 采样步数
            nfe_counter: NFE计数器
            **kwargs: 其他参数
        
        Returns:
            samples: 生成的样本 [B, C, H, W]
            info: 包含search信息的字典（如verifier scores, search轨迹等）
        """
        pass
    
    def __call__(self, *args, **kwargs):
        """支持直接调用"""
        return self.search(*args, **kwargs)


