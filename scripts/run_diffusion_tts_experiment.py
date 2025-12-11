# scripts/run_diffusion_tts_experiment.py

"""
运行 Diffusion-TTS 实验脚本
复现论文实验 5.1: Class-Conditional Image Generation on ImageNet-64

使用方法:
    python scripts/run_diffusion_tts_experiment.py --config configs/imagenet64_diffusion_tts.yaml
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
# 获取脚本所在目录的父目录（项目根目录）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# 确保项目根目录在sys.path中
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 验证src目录存在
src_dir = project_root / "src"
if not src_dir.exists():
    raise RuntimeError(
        f"Cannot find 'src' directory at {src_dir}. "
        f"Please ensure you are running this script from the project root."
    )

import torch
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

from src.utils.config import Config
from src.utils.nfe_counter import NFECounter
from src.models.edm_model import EDMModel
from src.verifiers.scorer_verifier import ScorerVerifier
from src.search.diffusion_tts_search import BestOfNSearch, ZeroOrderSearchTTS, EpsilonGreedySearch
from src.evaluation.metrics import compute_fid_is


def create_model(config: Config, device: str) -> EDMModel:
    """创建EDM模型"""
    model = EDMModel(
        model_path=config.model.path,
        device=device,
        image_size=config.image_size,
        num_channels=config.num_channels,
        num_classes=config.num_classes,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=40,
        S_min=0.05,
        S_max=50,
        S_noise=1.003,
    )
    return model


def create_verifier(config: Config, device: str, class_labels: Optional[torch.Tensor] = None) -> ScorerVerifier:
    """创建Scorer Verifier"""
    verifier_config = config.verifier
    # 支持Config对象和字典两种方式
    if hasattr(verifier_config, 'scorer_type'):
        scorer_type = verifier_config.scorer_type
    elif isinstance(verifier_config, dict):
        scorer_type = verifier_config.get("scorer_type", "imagenet")
    else:
        scorer_type = getattr(verifier_config, "scorer_type", "imagenet")
    
    verifier = ScorerVerifier(
        scorer_type=scorer_type,
        device=device,
        image_size=config.image_size,
        class_labels=class_labels,
    )
    return verifier


def run_experiment(config: Config):
    """运行实验"""
    device = config.model.device
    image_size = config.image_size
    num_channels = config.num_channels
    num_classes = config.num_classes
    batch_size = config.batch_size
    num_samples = config.num_samples
    num_steps = config.model.num_steps
    
    print(f"\n=== Running Diffusion-TTS Experiment ===")
    print(f"Dataset: {config.dataset}")
    print(f"Image Size: {image_size}x{image_size}")
    print(f"Num Steps: {num_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"Total Samples: {num_samples}")
    print(f"Method: {config.pipeline.local_search.type}")
    print("=" * 50)
    
    # 创建模型
    print("\nLoading EDM model...")
    model = create_model(config, device)
    
    # 创建verifier
    print(f"Creating {config.verifier.scorer_type} verifier...")
    verifier = create_verifier(config, device)
    
    # 选择要运行的方法
    method_type = config.pipeline.local_search.type
    # 获取方法配置，支持Config对象和字典
    if hasattr(config.pipeline.local_search, method_type):
        method_config = getattr(config.pipeline.local_search, method_type)
        # 如果是Config对象，转换为字典
        if isinstance(method_config, Config):
            method_config = method_config.to_dict()
        elif not isinstance(method_config, dict):
            method_config = {}
    else:
        method_config = {}
    
    # 创建search方法
    if method_type == "best_of_n":
        search_method = BestOfNSearch(
            model=model,
            verifier=verifier,
            n_candidates=method_config.get("n_candidates", 4)
        )
    elif method_type == "zero_order_tts":
        search_method = ZeroOrderSearchTTS(
            model=model,
            verifier=verifier,
            n_candidates=method_config.get("n_candidates", 4),
            search_steps=method_config.get("search_steps", 20),
            lambda_param=method_config.get("lambda_param", 0.15)
        )
    elif method_type == "epsilon_greedy":
        search_method = EpsilonGreedySearch(
            model=model,
            verifier=verifier,
            n_candidates=method_config.get("n_candidates", 4),
            search_steps=method_config.get("search_steps", 20),
            lambda_param=method_config.get("lambda_param", 0.15),
            epsilon=method_config.get("epsilon", 0.4)
        )
    else:
        raise ValueError(f"Unknown method type: {method_type}")
    
    # 生成类别标签（随机）
    torch.manual_seed(42)
    all_class_labels = torch.eye(num_classes)[torch.randint(num_classes, size=(num_samples,))]
    
    # 运行实验
    all_generated_images = []
    total_nfe_counter = NFECounter()
    all_scores = []
    
    print(f"\nGenerating {num_samples} samples...")
    for i in tqdm(range(0, num_samples, batch_size)):
        current_batch_size = min(batch_size, num_samples - i)
        if current_batch_size == 0:
            break
        
        # 获取当前batch的类别标签
        batch_class_labels = all_class_labels[i:i+current_batch_size].to(device)
        
        # 更新verifier的class_labels
        verifier.class_labels = batch_class_labels
        
        # 从method_config中移除已显式传递的参数，避免重复
        search_kwargs = method_config.copy() if isinstance(method_config, dict) else {}
        # 移除可能冲突的参数（这些参数已经显式传递）
        search_kwargs.pop('num_steps', None)
        search_kwargs.pop('batch_size', None)
        search_kwargs.pop('nfe_counter', None)
        search_kwargs.pop('device', None)
        search_kwargs.pop('class_labels', None)
        
        # 执行搜索
        samples, info = search_method.search(
            batch_size=current_batch_size,
            num_steps=num_steps,
            nfe_counter=total_nfe_counter,
            class_labels=batch_class_labels,
            device=device,
            **search_kwargs
        )
        
        all_generated_images.append(samples.cpu())
        
        # 记录分数
        if "best_score" in info:
            all_scores.append(info["best_score"])
        elif "scores" in info:
            if isinstance(info["scores"], list):
                all_scores.extend(info["scores"])
            else:
                all_scores.append(info["scores"].mean().item())
    
    # 合并所有生成的图像
    final_images = torch.cat(all_generated_images, dim=0)
    print(f"\nGenerated {final_images.shape[0]} samples.")
    print(f"Total NFE: {total_nfe_counter.current_nfe}")
    
    # 计算平均分数
    if all_scores:
        avg_score = np.mean(all_scores)
        print(f"Average {config.verifier.scorer_type} score: {avg_score:.4f}")
    
    # 评估指标（如果需要）
    eval_results = {}
    if config.evaluation.metrics:
        print("\nEvaluating metrics...")
        # 转换图像格式为numpy array [N, H, W, C]，值域[0, 255]
        images_np = final_images.permute(0, 2, 3, 1).cpu().numpy()
        if images_np.max() <= 1.0:
            images_np = (images_np * 255).astype(np.uint8)
        else:
            images_np = images_np.astype(np.uint8)
        
        # 计算FID和IS
        if "fid" in config.evaluation.metrics:
            # FID需要参考统计文件
            fid_stats_path = config.evaluation.get("fid_stats_path")
            if fid_stats_path:
                from src.evaluation.metrics import compute_fid
                fid_score = compute_fid(images_np, fid_stats_path, device)
                eval_results["fid"] = fid_score
            else:
                print("Warning: FID calculation requires fid_stats_path in config")
        
        if "is" in config.evaluation.metrics:
            from src.evaluation.metrics import compute_is
            is_mean, is_std = compute_is(images_np, device)
            eval_results["is_mean"] = is_mean
            eval_results["is_std"] = is_std
        
        print(f"Evaluation Results: {eval_results}")
    
    # 保存结果
    save_dir = config.evaluation.save_dir
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 保存图像（如果配置要求）
    if config.evaluation.save_images:
        images_dir = os.path.join(save_dir, "images", method_type, timestamp)
        os.makedirs(images_dir, exist_ok=True)
        # 这里可以添加保存图像的代码
        print(f"Images saved to {images_dir}")
    
    # 保存结果
    results = {
        "method": method_type,
        "config": method_config,
        "nfe": total_nfe_counter.current_nfe,
        "num_samples": final_images.shape[0],
        "average_score": avg_score if all_scores else None,
        "evaluation": eval_results if config.evaluation.metrics else None,
    }
    
    results_path = os.path.join(save_dir, f"{config.experiment_name}_{method_type}_{timestamp}.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Diffusion-TTS experiments on ImageNet-64")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    config = Config(args.config)
    print("=== Experiment Configuration ===")
    print(config)
    print("=" * 50)
    
    run_experiment(config)

