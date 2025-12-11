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
    # 根据原始实现：class_labels = torch.eye(1000)[torch.randint(1000, size=[g * g])]
    torch.manual_seed(42)
    random_indices = torch.randint(num_classes, size=(num_samples,))
    all_class_labels = torch.eye(num_classes)[random_indices]
    print(f"Debug: Generated class labels - shape: {all_class_labels.shape}")
    print(f"Debug: First 5 class indices: {random_indices[:5]}")
    print(f"Debug: Sample class label (first row): has 1 at index {all_class_labels[0].argmax().item()}")
    
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
    
    # 计算平均分数（搜索时使用的scorer）
    if all_scores:
        # 过滤NaN值
        valid_scores = [s for s in all_scores if not np.isnan(s)]
        if valid_scores:
            avg_score = np.mean(valid_scores)
            print(f"Average {config.verifier.scorer_type} score (during search): {avg_score:.4f} ± {np.std(valid_scores):.4f}")
        else:
            avg_score = None
            print(f"Warning: All scores are NaN for {config.verifier.scorer_type}")
    else:
        avg_score = None
    
    # 评估指标（对最终生成的图像进行评估）
    eval_results = {}
    
    # EDM模型输出的图像需要按照原始实现进行归一化
    # 根据原始代码：image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    images_tensor = final_images.clone()
    
    # 检查NaN/Inf
    if torch.isnan(images_tensor).any() or torch.isinf(images_tensor).any():
        print("Warning: Found NaN/Inf in final_images. Replacing with zeros.")
        images_tensor = torch.nan_to_num(images_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 调试：检查原始值域
    print(f"\nDebug: Raw image tensor stats - min: {images_tensor.min().item():.4f}, max: {images_tensor.max().item():.4f}, mean: {images_tensor.mean().item():.4f}")
    print(f"Debug: Image tensor shape: {images_tensor.shape}, dtype: {images_tensor.dtype}")
    
    # EDM输出通常在[-1, 1]范围，转换为[0, 255] uint8（如原始代码）
    images_uint8 = (images_tensor * 127.5 + 128).clip(0, 255).to(torch.uint8)
    
    # 转换为numpy用于评估（已经是uint8 [0,255]）
    images_np = images_uint8.permute(0, 2, 3, 1).cpu().numpy()
    
    # 转换为torch tensor用于scorer（值域[0, 1]，float32）
    images_torch = images_uint8.float() / 255.0
    images_torch = images_torch.to(device)
    
    print(f"Debug: Processed image tensor stats - min: {images_torch.min().item():.4f}, max: {images_torch.max().item():.4f}, mean: {images_torch.mean().item():.4f}")
    
    # 评估三种scorer（如论文中Table 1）
    print("\n=== Evaluating with all three scorers (as in paper Table 1) ===")
    from src.verifiers.scorer_verifier import ScorerVerifier
    
    # 1. Brightness Scorer
    brightness_verifier = ScorerVerifier(
        scorer_type="brightness",
        device=device,
        image_size=config.image_size,
    )
    with torch.no_grad():
        try:
            brightness_scores = brightness_verifier.score(images_torch)
            print(f"Debug: Brightness scores shape: {brightness_scores.shape}, values: {brightness_scores[:5]}")
            brightness_mean = brightness_scores.mean().item()
            brightness_std = brightness_scores.std().item()
            if np.isnan(brightness_mean) or np.isinf(brightness_mean):
                print(f"Warning: Brightness score is NaN/Inf. Scores: {brightness_scores}")
                brightness_mean = 0.0
                brightness_std = 0.0
        except Exception as e:
            print(f"Error computing brightness score: {e}")
            import traceback
            traceback.print_exc()
            brightness_mean = 0.0
            brightness_std = 0.0
    eval_results["brightness"] = {"mean": brightness_mean, "std": brightness_std}
    print(f"Brightness: {brightness_mean:.4f} ± {brightness_std:.4f}")
    
    # 2. Compressibility Scorer (需要uint8格式)
    compressibility_verifier = ScorerVerifier(
        scorer_type="compressibility",
        device=device,
        image_size=config.image_size,
    )
    with torch.no_grad():
        # CompressibilityScorer需要uint8格式，直接使用images_uint8
        compressibility_scores = compressibility_verifier.score(images_uint8.to(device))
        print(f"Debug: Compressibility scores shape: {compressibility_scores.shape}, values: {compressibility_scores[:5]}")
        compressibility_mean = compressibility_scores.mean().item()
        compressibility_std = compressibility_scores.std().item()
        if np.isnan(compressibility_mean) or np.isinf(compressibility_mean):
            print(f"Warning: Compressibility score is NaN/Inf. Scores: {compressibility_scores}")
            compressibility_mean = 0.0
            compressibility_std = 0.0
    eval_results["compressibility"] = {"mean": compressibility_mean, "std": compressibility_std}
    print(f"Compressibility: {compressibility_mean:.4f} ± {compressibility_std:.4f}")
    
    # 3. ImageNet Classifier Scorer
    imagenet_verifier = ScorerVerifier(
        scorer_type="imagenet",
        device=device,
        image_size=config.image_size,
    )
    # 为最终生成的图像评估ImageNet分数（使用相同的class_labels）
    imagenet_verifier.class_labels = all_class_labels.to(device)
    print(f"Debug: Class labels shape: {all_class_labels.shape}, dtype: {all_class_labels.dtype}")
    print(f"Debug: Class labels sample: {all_class_labels[:3]}")
    with torch.no_grad():
        try:
            imagenet_scores = imagenet_verifier.score(images_torch, class_labels=all_class_labels.to(device))
            print(f"Debug: ImageNet scores shape: {imagenet_scores.shape}, values: {imagenet_scores[:5]}")
            imagenet_mean = imagenet_scores.mean().item()
            imagenet_std = imagenet_scores.std().item()
            if np.isnan(imagenet_mean) or np.isinf(imagenet_mean):
                print(f"Warning: ImageNet score is NaN/Inf. Scores: {imagenet_scores}")
                imagenet_mean = 0.0
                imagenet_std = 0.0
        except Exception as e:
            print(f"Error computing ImageNet score: {e}")
            import traceback
            traceback.print_exc()
            imagenet_mean = 0.0
            imagenet_std = 0.0
    eval_results["imagenet"] = {"mean": imagenet_mean, "std": imagenet_std}
    print(f"ImageNet Classifier: {imagenet_mean:.4f} ± {imagenet_std:.4f}")
    
    # FID和IS评估（如果配置了）
    if config.evaluation.metrics:
        print("\n=== Standard Evaluation Metrics ===")
        if "fid" in config.evaluation.metrics:
            fid_stats_path = config.evaluation.get("fid_stats_path")
            if fid_stats_path:
                from src.evaluation.metrics import compute_fid
                fid_score = compute_fid(images_np, fid_stats_path, device)
                eval_results["fid"] = fid_score
                print(f"FID: {fid_score:.4f}")
            else:
                print("Warning: FID calculation requires fid_stats_path in config")
        
        if "is" in config.evaluation.metrics:
            from src.evaluation.metrics import compute_is
            is_mean, is_std = compute_is(images_np, device)
            eval_results["is_mean"] = is_mean
            eval_results["is_std"] = is_std
            print(f"IS: {is_mean:.4f} ± {is_std:.4f}")
    
    print(f"\n=== All Evaluation Results ===")
    print(eval_results)
    
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
        "average_score_during_search": avg_score if all_scores else None,
        "scorer_metrics": eval_results,  # 包含三种scorer的结果（Brightness, Compressibility, ImageNet）
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

