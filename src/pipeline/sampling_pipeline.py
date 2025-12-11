"""
完整的采样Pipeline
整合Local Search和Global Search
"""

from typing import Optional, Dict, Any, Tuple
import torch
from ..models.base_model import BaseDiffusionModel
from ..verifiers.base_verifier import BaseVerifier
from ..utils.nfe_counter import NFECounter
from ..search.local_search import NoSearch, RandomSearch, LocalNoiseSearch, ZeroOrderSearch
from ..search.diffusion_tts_search import BestOfNSearch, ZeroOrderSearchTTS, EpsilonGreedySearch
from ..search.global_search import (
    GlobalSearch,
    GlobalSearchPolicy,
    FixedBudgetPolicy,
    AdaptiveThresholdPolicy,
    MultiStagePolicy,
    SearchMode,
)


class SamplingPipeline:
    """
    完整的采样Pipeline
    
    支持两种模式：
    1. 直接使用Local Search（baseline方法）
    2. 使用Global Search控制Local Search（目标方法）
    """
    
    def __init__(
        self,
        model: BaseDiffusionModel,
        verifier: BaseVerifier,
        use_global_search: bool = False,
        global_policy: Optional[GlobalSearchPolicy] = None,
    ):
        """
        Args:
            model: 扩散模型
            verifier: Verifier
            use_global_search: 是否使用Global Search
            global_policy: Global Search策略（如果use_global_search=True）
        """
        self.model = model
        self.verifier = verifier
        self.use_global_search = use_global_search
        
        if use_global_search:
            if global_policy is None:
                # 默认使用固定分配策略
                global_policy = FixedBudgetPolicy(total_nfe_budget=200)
            self.global_search = GlobalSearch(model, verifier, global_policy)
        else:
            self.global_search = None
    
    def sample(
        self,
        method: str = "no_search",
        batch_size: int = 1,
        num_steps: int = 50,
        initial_noise: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        执行采样
        
        Args:
            method: 采样方法
                - "no_search": 标准采样
                - "random": Random Search
                - "local": Local Noise Search
                - "zo": Zero-Order Search
                - "best_of_n": Best-of-N (Rejection Sampling)
                - "zero_order_tts": Zero-Order Search (from diffusion-tts)
                - "epsilon_greedy": ε-greedy Search (from diffusion-tts)
                - "global": Global Search（如果use_global_search=True）
            batch_size: batch大小
            num_steps: 采样步数
            initial_noise: 初始噪声
            prompt: 提示词
            nfe_counter: NFE计数器
            **kwargs: 其他参数
        
        Returns:
            samples: 生成的样本
            info: 采样信息
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        if self.use_global_search and method == "global":
            # 使用Global Search
            return self.global_search.sample(
                initial_noise=initial_noise,
                batch_size=batch_size,
                num_steps=num_steps,
                prompt=prompt,
                nfe_counter=nfe_counter,
            )
        else:
            # 使用Local Search方法
            if method == "no_search":
                local_search = NoSearch(self.model, self.verifier)
            elif method == "random":
                num_candidates = kwargs.get("num_candidates", 4)
                local_search = RandomSearch(self.model, self.verifier, num_candidates=num_candidates)
            elif method == "local":
                num_candidates = kwargs.get("num_candidates_per_step", 4)
                local_search = LocalNoiseSearch(self.model, self.verifier, num_candidates_per_step=num_candidates)
            elif method == "zo":
                num_iterations = kwargs.get("num_iterations", 4)
                num_neighbors = kwargs.get("num_neighbors", 8)
                local_search = ZeroOrderSearch(
                    self.model, self.verifier,
                    num_iterations=num_iterations,
                    num_neighbors=num_neighbors,
                )
            elif method == "best_of_n":
                n_candidates = kwargs.get("n_candidates", 4)
                local_search = BestOfNSearch(
                    self.model, self.verifier,
                    n_candidates=n_candidates
                )
            elif method == "zero_order_tts":
                n_candidates = kwargs.get("n_candidates", 4)
                search_steps = kwargs.get("search_steps", 20)
                lambda_param = kwargs.get("lambda_param", 0.15)
                local_search = ZeroOrderSearchTTS(
                    self.model, self.verifier,
                    n_candidates=n_candidates,
                    search_steps=search_steps,
                    lambda_param=lambda_param
                )
            elif method == "epsilon_greedy":
                n_candidates = kwargs.get("n_candidates", 4)
                search_steps = kwargs.get("search_steps", 20)
                lambda_param = kwargs.get("lambda_param", 0.15)
                epsilon = kwargs.get("epsilon", 0.4)
                local_search = EpsilonGreedySearch(
                    self.model, self.verifier,
                    n_candidates=n_candidates,
                    search_steps=search_steps,
                    lambda_param=lambda_param,
                    epsilon=epsilon
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return local_search.search(
                initial_noise=initial_noise,
                batch_size=batch_size,
                num_steps=num_steps,
                nfe_counter=nfe_counter,
                **kwargs
            )


def create_pipeline(
    model: BaseDiffusionModel,
    verifier: BaseVerifier,
    method: str = "no_search",
    global_policy_type: str = "fixed",
    total_nfe_budget: int = 200,
    **policy_kwargs
) -> SamplingPipeline:
    """
    创建Pipeline的工厂函数
    
    Args:
        model: 扩散模型
        verifier: Verifier
        method: 采样方法
        global_policy_type: Global Search策略类型（fixed/adaptive/multi_stage）
        total_nfe_budget: 总NFE预算
        **policy_kwargs: 策略参数
    
    Returns:
        SamplingPipeline实例
    """
    use_global = (method == "global")
    global_policy = None
    
    if use_global:
        if global_policy_type == "fixed":
            global_policy = FixedBudgetPolicy(total_nfe_budget, **policy_kwargs)
        elif global_policy_type == "adaptive":
            global_policy = AdaptiveThresholdPolicy(total_nfe_budget, **policy_kwargs)
        elif global_policy_type == "multi_stage":
            global_policy = MultiStagePolicy(total_nfe_budget, **policy_kwargs)
        else:
            raise ValueError(f"Unknown policy type: {global_policy_type}")
    
    return SamplingPipeline(
        model=model,
        verifier=verifier,
        use_global_search=use_global,
        global_policy=global_policy,
    )

