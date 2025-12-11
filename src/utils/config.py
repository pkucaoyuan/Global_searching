"""
配置管理模块
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from omegaconf import OmegaConf


class Config:
    """实验配置类"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径（YAML格式）
            **kwargs: 额外的配置参数
        """
        self._config = {}
        
        # 加载默认配置
        self._load_defaults()
        
        # 从文件加载
        if config_path:
            self.load_from_file(config_path)
        
        # 从kwargs更新
        if kwargs:
            self.update(kwargs)
    
    def _load_defaults(self):
        """加载默认配置"""
        self._config = {
            # 数据集
            "dataset": "cifar10",
            "image_size": 32,
            "num_classes": 10,
            
            # 模型
            "model_type": "ddpm",
            "model_path": None,
            
            # 采样
            "num_steps": 50,
            "num_samples": 50000,  # 用于评估的样本数
            "batch_size": 64,
            
            # Verifier
            "verifier_type": "classifier",
            "verifier_path": None,
            
            # Search方法
            "search_method": "none",  # none, random, zo, nlg
            "search_budget": 0,  # 额外的NFE预算用于search
            
            # 评估
            "eval_fid": True,
            "eval_is": True,
            "fid_stats_path": None,  # FID统计文件路径
            
            # 实验
            "seed": 42,
            "device": "cuda",
            "output_dir": "results",
        }
    
    def load_from_file(self, path: str):
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
        if file_config:
            self._config.update(file_config)
    
    def save_to_file(self, path: str):
        """保存配置到YAML文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
    
    def update(self, d: Dict[str, Any]):
        """更新配置"""
        self._config.update(d)
    
    def get(self, key: str, default: Any = None):
        """获取配置值"""
        return self._config.get(key, default)
    
    def __getitem__(self, key: str):
        """支持[]访问"""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        """支持[]设置"""
        self._config[key] = value
    
    def __contains__(self, key: str):
        """支持in操作"""
        return key in self._config
    
    def __repr__(self):
        return f"Config({self._config})"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()


