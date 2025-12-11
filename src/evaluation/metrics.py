"""
评估指标计算
包括FID和IS
"""

import torch
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


def compute_fid(
    generated_images: np.ndarray,
    reference_stats_path: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """
    计算FID分数
    
    Args:
        generated_images: 生成的图像，numpy array，shape [N, H, W, C]，值域[0, 255]
        reference_stats_path: 参考数据集的FID统计文件路径（.npz格式）
        device: 计算设备
    
    Returns:
        FID分数（越低越好）
    """
    try:
        from pytorch_fid import fid_score
        
        # 如果提供了参考统计，使用它
        if reference_stats_path:
            # 将生成图像保存到临时目录
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # 保存生成的图像
                gen_dir = Path(tmpdir) / "generated"
                gen_dir.mkdir()
                # TODO: 保存图像到文件
                
                # 计算FID
                fid_value = fid_score.calculate_fid_given_paths(
                    [str(gen_dir), reference_stats_path],
                    batch_size=50,
                    device=device,
                    dims=2048,
                )
        else:
            # 需要提供参考数据集目录
            raise ValueError("需要提供reference_stats_path或参考数据集目录")
        
        return fid_value
    
    except ImportError:
        print("警告: pytorch-fid未安装，返回占位符FID值")
        print("安装: pip install pytorch-fid")
        return 0.0


def compute_is(
    images: np.ndarray,
    device: str = "cuda",
    batch_size: int = 50,
) -> Tuple[float, float]:
    """
    计算IS（Inception Score）
    
    Args:
        images: 图像numpy array，shape [N, H, W, C]，值域[0, 255]
        device: 计算设备
        batch_size: batch大小
    
    Returns:
        (IS均值, IS标准差)
    """
    try:
        import torch
        import torch.nn.functional as F
        from torchvision.models import inception_v3
        
        # 加载InceptionV3
        model = inception_v3(pretrained=True, transform_input=False)
        model.to(device)
        model.eval()
        
        # 预处理图像
        # TODO: 实现图像预处理和batch处理
        
        # 计算IS
        # IS = exp(E[KL(p(y|x) || p(y))])
        
        print("警告: IS计算未完整实现，返回占位符值")
        return (0.0, 0.0)
    
    except Exception as e:
        print(f"IS计算错误: {e}")
        return (0.0, 0.0)


def compute_fid_is(
    generated_images: np.ndarray,
    reference_stats_path: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[float, Tuple[float, float]]:
    """
    同时计算FID和IS
    
    Returns:
        (FID, (IS_mean, IS_std))
    """
    fid = compute_fid(generated_images, reference_stats_path, device)
    is_mean, is_std = compute_is(generated_images, device)
    
    return fid, (is_mean, is_std)


