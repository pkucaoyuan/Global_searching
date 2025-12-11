# src/models/edm_model.py

"""
EDM (Elucidating the Design Space of Diffusion Models) Model Wrapper

适配 EDM 模型到 BaseDiffusionModel 接口
从 diffusion-tts 的 EDM 实现中提取采样逻辑
"""

import torch
import pickle
import numpy as np
from typing import Optional, Dict, Any
import sys
from pathlib import Path

from .base_model import BaseDiffusionModel
from ..utils.nfe_counter import NFECounter


class EDMModel(BaseDiffusionModel):
    """
    EDM 模型包装器
    
    支持 class-conditional ImageNet-64x64 生成
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        image_size: int = 64,
        num_channels: int = 3,
        num_classes: int = 1000,
        # EDM采样参数
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        S_churn: float = 40,
        S_min: float = 0.05,
        S_max: float = 50,
        S_noise: float = 1.003,
    ):
        """
        初始化EDM模型
        
        Args:
            model_path: 模型checkpoint路径（.pkl文件或URL）
            device: 设备
            image_size: 图像尺寸
            num_channels: 通道数
            num_classes: 类别数（用于class-conditional）
            sigma_min, sigma_max, rho: EDM时间步参数
            S_churn, S_min, S_max, S_noise: EDM采样器参数
        """
        self._image_size = image_size
        self._num_channels = num_channels
        self._num_classes = num_classes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        
        # 导入dnnlib（从diffusion-tts）
        self._setup_dnnlib()
        
        super().__init__(model_path, device)
    
    def _setup_dnnlib(self):
        """设置dnnlib工具"""
        edm_dir = Path(__file__).parent.parent.parent / "code_repos" / "diffusion-tts" / "edm"
        if edm_dir.exists():
            sys.path.insert(0, str(edm_dir))
            try:
                import dnnlib
                import dnnlib.util
                self.dnnlib = dnnlib
                self.dnnlib_util = dnnlib.util
            except ImportError:
                raise ImportError(
                    f"Could not import dnnlib from {edm_dir}. "
                    "Please ensure diffusion-tts is cloned in code_repos/"
                )
        else:
            raise FileNotFoundError(
                f"EDM directory not found at {edm_dir}. "
                "Please clone diffusion-tts repository to code_repos/"
            )
    
    def load_model(self, path: str):
        """加载EDM模型checkpoint"""
        print(f'Loading EDM network from "{path}"...')
        with self.dnnlib_util.open_url(path) as f:
            checkpoint = pickle.load(f)
            self.model = checkpoint['ema'].to(self.device)
            self.model.eval()
        print("EDM model loaded successfully.")
    
    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        执行一步去噪（EDM采样器）
        
        Args:
            x_t: 当前时刻的噪声图像 [B, C, H, W]
            t: 时间步索引（0到num_steps-1，从大到小）
            class_labels: 类别标签 [B, num_classes] (one-hot) 或 [B] (class indices)
            **kwargs: 其他参数，可能包含：
                - num_steps: 总步数
                - t_steps: 时间步张量（如果已计算）
        
        Returns:
            x_{t-1}: 去噪后的图像
        """
        # 获取时间步信息
        num_steps = kwargs.get("num_steps", 18)
        t_steps = kwargs.get("t_steps", None)
        
        if t_steps is None:
            # 计算时间步
            step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
            t_steps = (
                self.sigma_max ** (1 / self.rho) + 
                step_indices / (num_steps - 1) * 
                (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
            ) ** self.rho
            t_steps = torch.cat([self.model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        # 获取当前和下一个时间步
        t_cur = t_steps[t]
        t_next = t_steps[t + 1] if t < len(t_steps) - 1 else torch.zeros_like(t_cur)
        
        # EDM采样步骤
        gamma = min(self.S_churn / num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
        t_hat = self.model.round_sigma(t_cur + gamma * t_cur)
        
        # 添加噪声（如果需要）
        if gamma > 0:
            eps_i = torch.randn_like(x_t)
            x_hat = x_t + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * eps_i
        else:
            x_hat = x_t
        
        # 去噪
        with self.nfe_counter.count():
            denoised = self.model(x_hat, t_hat, class_labels).to(torch.float64)
        
        # 检查NaN
        if torch.isnan(denoised).any() or torch.isinf(denoised).any():
            print(f"Warning: NaN/Inf in denoised at step {t}")
            denoised = torch.nan_to_num(denoised, nan=0.0, posinf=1.0, neginf=-1.0)
        
        d_cur = (x_hat - denoised) / (t_hat + 1e-8)  # 避免除零
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # 检查NaN
        if torch.isnan(x_next).any() or torch.isinf(x_next).any():
            print(f"Warning: NaN/Inf in x_next after first step at t={t}")
            x_next = torch.nan_to_num(x_next, nan=x_hat, posinf=x_hat, neginf=x_hat)
        
        # 二阶校正（Heun方法）
        if t < len(t_steps) - 1:
            with self.nfe_counter.count():
                denoised = self.model(x_next, t_next, class_labels).to(torch.float64)
            
            # 检查NaN
            if torch.isnan(denoised).any() or torch.isinf(denoised).any():
                print(f"Warning: NaN/Inf in denoised (second step) at step {t}")
                denoised = torch.nan_to_num(denoised, nan=0.0, posinf=1.0, neginf=-1.0)
            
            d_prime = (x_next - denoised) / (t_next + 1e-8)  # 避免除零
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
            # 最后检查NaN
            if torch.isnan(x_next).any() or torch.isinf(x_next).any():
                print(f"Warning: NaN/Inf in final x_next at t={t}")
                x_next = torch.nan_to_num(x_next, nan=x_hat, posinf=x_hat, neginf=x_hat)
        
        return x_next.to(x_t.dtype)
    
    def sample_noise(self, batch_size: int, image_size: int) -> torch.Tensor:
        """
        采样初始噪声
        
        Args:
            batch_size: batch大小
            image_size: 图像尺寸
        
        Returns:
            初始噪声 [B, C, H, W]
        """
        # EDM使用sigma_max作为初始噪声尺度
        noise = torch.randn(
            batch_size,
            self.num_channels,
            image_size,
            image_size,
            device=self.device
        )
        return noise * self.sigma_max
    
    def sample(
        self,
        batch_size: int,
        num_steps: int,
        nfe_counter: Optional[NFECounter] = None,
        initial_noise: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        EDM采样流程
        
        Args:
            batch_size: batch大小
            num_steps: 采样步数
            nfe_counter: NFE计数器
            initial_noise: 初始噪声（可选）
            class_labels: 类别标签（可选，用于class-conditional）
            **kwargs: 其他参数
        
        Returns:
            生成的样本 [B, C, H, W]
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        self.set_nfe_counter(nfe_counter)
        
        # 计算时间步
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            self.sigma_max ** (1 / self.rho) + 
            step_indices / (num_steps - 1) * 
            (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        # 初始化
        # 根据原始实现：x_next = latents.to(torch.float64) * t_steps[0]
        # latents是标准正态噪声 torch.randn([batch_size, 3, 64, 64])
        if initial_noise is None:
            x = self.sample_noise(batch_size, self.image_size).to(torch.float64)
        else:
            # 原始实现中，latents是标准正态噪声，需要乘以t_steps[0]
            # 但如果initial_noise来自sample_noise（已经乘以sigma_max），则需要除以sigma_max再乘以t_steps[0]
            # 为简单起见，假设initial_noise是标准正态噪声（来自repeat_interleave等）
            x = initial_noise.to(self.device).to(torch.float64) * t_steps[0]
        
        # 逐步去噪
        from tqdm import tqdm
        for i in tqdm(range(num_steps), desc="EDM Sampling"):
            t = num_steps - 1 - i  # 从大到小
            with nfe_counter.count():
                x = self.denoise_step(
                    x, t,
                    class_labels=class_labels,
                    num_steps=num_steps,
                    t_steps=t_steps,
                    **kwargs
                )
        
        return x
    
    @property
    def image_size(self) -> int:
        return self._image_size
    
    @property
    def num_channels(self) -> int:
        return self._num_channels
    
    @property
    def num_classes(self) -> int:
        return self._num_classes

