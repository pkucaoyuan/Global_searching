#!/usr/bin/env python3
"""
运行Pure Sampling Baseline实验
这是第一个实验脚本，用于验证基础流程
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.nfe_counter import NFECounter


def run_pure_sampling_experiment(config_path: str, steps_list: list = None):
    """
    运行pure sampling baseline实验
    
    Args:
        config_path: 配置文件路径
        steps_list: 要测试的步数列表，如 [25, 50, 100, 200]
    """
    # 加载配置
    config = Config(config_path)
    
    if steps_list is None:
        steps_list = [25, 50, 100, 200]
    
    print("=" * 60)
    print("Pure Sampling Baseline 实验")
    print("=" * 60)
    print(f"数据集: {config['dataset']}")
    print(f"测试步数: {steps_list}")
    print(f"每个配置采样样本数: {config.get('num_samples', 50000)}")
    print()
    
    results = []
    
    for num_steps in steps_list:
        print(f"\n正在测试 {num_steps} 步采样...")
        
        # 更新配置
        config['num_steps'] = num_steps
        
        # 初始化NFE计数器
        nfe_counter = NFECounter()
        
        # TODO: 加载模型
        # model = load_model(config)
        
        # TODO: 执行采样
        # 这里需要实现实际的采样逻辑
        # 示例结构：
        # samples = []
        # num_batches = config['num_samples'] // config['batch_size']
        # for batch_idx in tqdm(range(num_batches)):
        #     batch_samples = model.sample(
        #         batch_size=config['batch_size'],
        #         num_steps=num_steps,
        #         nfe_counter=nfe_counter
        #     )
        #     samples.append(batch_samples)
        # samples = torch.cat(samples, dim=0)
        
        # TODO: 计算评估指标
        # fid, (is_mean, is_std) = compute_fid_is(samples, ...)
        
        # 占位符结果（实际运行时需要替换）
        total_nfe = nfe_counter.total_nfe
        fid = 0.0  # TODO: 实际计算
        is_mean = 0.0  # TODO: 实际计算
        
        result = {
            'num_steps': num_steps,
            'total_nfe': total_nfe,
            'fid': fid,
            'is_mean': is_mean,
        }
        results.append(result)
        
        print(f"  步数: {num_steps}")
        print(f"  NFE: {total_nfe}")
        print(f"  FID: {fid:.4f}")
        print(f"  IS: {is_mean:.4f} ± {0.0:.4f}")
    
    # 保存结果
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "baseline_results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("实验完成！")
    print(f"结果已保存到: {results_file}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行Pure Sampling Baseline实验")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cifar10_baseline.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[25, 50, 100, 200],
        help="要测试的步数列表"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    run_pure_sampling_experiment(args.config, args.steps)


