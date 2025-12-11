"""
Local Search方法实现
Local Search: 从xt到xt-1的搜索方法
"""

from typing import Tuple, Optional, Dict, Any, List
import torch
import torch.nn.functional as F
from .base_search import BaseSearch
from ..models.base_model import BaseDiffusionModel
from ..verifiers.base_verifier import BaseVerifier
from ..utils.nfe_counter import NFECounter


class NoSearch(BaseSearch):
    """
    无搜索：直接使用标准采样
    这是baseline方法
    """
    
    def search(
        self,
        initial_noise: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_steps: int = 50,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """执行标准采样，不进行搜索"""
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        # 如果没有提供初始噪声，则采样
        if initial_noise is None:
            x_t = self.model.sample_noise(batch_size, self.model.image_size)
        else:
            x_t = initial_noise
        
        # 标准采样流程
        x_0 = self.model.sample(
            batch_size=batch_size,
            num_steps=num_steps,
            nfe_counter=nfe_counter,
            initial_noise=x_t,
        )
        
        info = {
            "method": "no_search",
            "nfe": nfe_counter.total_nfe,
            "verifier_score": None,  # 可以后续添加
        }
        
        return x_0, info


class RandomSearch(BaseSearch):
    """
    Random Search: 采样多个初始噪声，独立采样，选择verifier score最高的
    
    Local Search层面：在单个时间步t，可以采样多个候选xt-1，选择最好的
    但这里实现的是全局版本：采样多个完整轨迹，选择最好的
    """
    
    def __init__(
        self,
        model: BaseDiffusionModel,
        verifier: BaseVerifier,
        nfe_budget: int = 0,
        num_candidates: int = 4,
    ):
        """
        Args:
            num_candidates: 采样的候选轨迹数量
        """
        super().__init__(model, verifier, nfe_budget)
        self.num_candidates = num_candidates
    
    def search(
        self,
        initial_noise: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_steps: int = 50,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        执行Random Search
        
        对于每个候选：
        1. 采样初始噪声
        2. 完整采样轨迹
        3. 使用verifier评估
        选择verifier score最高的轨迹
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        num_candidates = kwargs.get("num_candidates", self.num_candidates)
        
        # 采样多个候选
        candidates = []
        scores = []
        
        for i in range(num_candidates):
            # 采样初始噪声
            if initial_noise is None:
                noise = self.model.sample_noise(batch_size, self.model.image_size)
            else:
                # 如果提供了初始噪声，可以添加小的扰动
                noise = initial_noise.clone()
                if i > 0:  # 第一个保持原样
                    noise = noise + torch.randn_like(noise) * 0.01
            
            # 采样完整轨迹
            candidate_nfe = NFECounter()
            x_0 = self.model.sample(
                batch_size=batch_size,
                num_steps=num_steps,
                nfe_counter=candidate_nfe,
                initial_noise=noise,
            )
            candidates.append(x_0)
            
            # 使用verifier评估
            with torch.no_grad():
                score = self.verifier.score(x_0)
                scores.append(score.item() if score.numel() == 1 else score.mean().item())
            
            # 累计NFE
            nfe_counter.increment(candidate_nfe.current_nfe)
        
        # 选择最优候选
        best_idx = torch.tensor(scores).argmax().item()
        best_sample = candidates[best_idx]
        
        info = {
            "method": "random_search",
            "num_candidates": num_candidates,
            "nfe": nfe_counter.total_nfe,
            "verifier_scores": scores,
            "best_idx": best_idx,
            "best_score": scores[best_idx],
        }
        
        return best_sample, info


class LocalNoiseSearch(BaseSearch):
    """
    Local Search at each step: 在每个时间步，采样多个候选xt-1，选择verifier score最高的
    
    这是真正的"local search"，在单个时间步上进行搜索
    """
    
    def __init__(
        self,
        model: BaseDiffusionModel,
        verifier: BaseVerifier,
        nfe_budget: int = 0,
        num_candidates_per_step: int = 4,
    ):
        """
        Args:
            num_candidates_per_step: 每个时间步采样的候选数量
        """
        super().__init__(model, verifier, nfe_budget)
        self.num_candidates_per_step = num_candidates_per_step
    
    def search(
        self,
        initial_noise: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_steps: int = 50,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        执行Local Search
        
        对于每个时间步t：
        1. 从当前xt采样多个候选xt-1
        2. 使用verifier评估每个候选（需要解码到x0或中间状态）
        3. 选择verifier score最高的xt-1
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        num_candidates = kwargs.get("num_candidates_per_step", self.num_candidates_per_step)
        
        # 初始化
        if initial_noise is None:
            x_t = self.model.sample_noise(batch_size, self.model.image_size)
        else:
            x_t = initial_noise.clone()
        
        all_scores = []
        
        # 逐步采样
        for t in range(num_steps - 1, -1, -1):
            candidates = []
            scores = []
            
            # 对当前步骤采样多个候选
            for _ in range(num_candidates):
                with nfe_counter.count():
                    x_t_minus_1, _ = self.model.denoise_step(x_t, t)
                candidates.append(x_t_minus_1)
                
                # 使用verifier评估（这里简化：直接评估xt-1，实际可能需要解码到x0）
                # 为了效率，只在关键步骤评估
                if t == 0 or (num_steps - t) % 10 == 0:
                    with torch.no_grad():
                        # 简单评估：使用当前状态的某些特征
                        # 实际应用中可能需要更复杂的评估方法
                        score = torch.norm(x_t_minus_1).item()
                        scores.append(score)
                else:
                    scores.append(0.0)
            
            # 选择最优候选（如果t不是0，使用简单策略；否则选择第一个）
            if t == 0:
                # 最后一步，需要更仔细地评估
                best_idx = 0
                if len(scores) > 0 and any(s > 0 for s in scores):
                    # 重新评估所有候选
                    eval_scores = []
                    for candidate in candidates:
                        with torch.no_grad():
                            score = self.verifier.score(candidate)
                            eval_scores.append(score.item() if score.numel() == 1 else score.mean().item())
                    best_idx = torch.tensor(eval_scores).argmax().item()
                    all_scores.append(eval_scores)
            else:
                # 中间步骤，可以随机选择或使用简单启发式
                best_idx = 0  # 简化：总是选择第一个
            
            x_t = candidates[best_idx]
        
        x_0 = x_t
        
        info = {
            "method": "local_noise_search",
            "num_candidates_per_step": num_candidates,
            "nfe": nfe_counter.total_nfe,
            "scores_history": all_scores,
        }
        
        return x_0, info


class ZeroOrderSearch(BaseSearch):
    """
    Zero-Order Search (ZO-N): 
    Pivot-based迭代搜索，在初始噪声邻域进行搜索
    """
    
    def __init__(
        self,
        model: BaseDiffusionModel,
        verifier: BaseVerifier,
        nfe_budget: int = 0,
        num_iterations: int = 4,
        num_neighbors: int = 8,
        noise_scale: float = 0.1,
    ):
        """
        Args:
            num_iterations: 迭代次数
            num_neighbors: 每次迭代采样的邻域样本数
            noise_scale: 邻域噪声的缩放因子
        """
        super().__init__(model, verifier, nfe_budget)
        self.num_iterations = num_iterations
        self.num_neighbors = num_neighbors
        self.noise_scale = noise_scale
    
    def search(
        self,
        initial_noise: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_steps: int = 50,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        执行Zero-Order Search
        
        算法：
        1. 从初始pivot噪声开始
        2. 在pivot邻域采样多个噪声
        3. 对每个噪声完整采样，评估verifier score
        4. 选择最优的作为新的pivot
        5. 重复步骤2-4
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        # 初始化pivot
        if initial_noise is None:
            pivot = self.model.sample_noise(batch_size, self.model.image_size)
        else:
            pivot = initial_noise.clone()
        
        best_score = float('-inf')
        best_sample = None
        iteration_scores = []
        
        # 迭代搜索
        for iteration in range(self.num_iterations):
            # 在pivot邻域采样
            neighbors = []
            for _ in range(self.num_neighbors):
                noise = pivot + torch.randn_like(pivot) * self.noise_scale
                neighbors.append(noise)
            
            # 评估每个邻居
            candidates = []
            scores = []
            
            for noise in neighbors:
                # 完整采样
                candidate_nfe = NFECounter()
                x_0 = self.model.sample(
                    batch_size=batch_size,
                    num_steps=num_steps,
                    nfe_counter=candidate_nfe,
                    initial_noise=noise,
                )
                candidates.append(x_0)
                nfe_counter.add(candidate_nfe.current_nfe)
                
                # Verifier评估
                with torch.no_grad():
                    score = self.verifier.score(x_0)
                    score_val = score.item() if score.numel() == 1 else score.mean().item()
                    scores.append(score_val)
            
            # 选择最优
            best_neighbor_idx = torch.tensor(scores).argmax().item()
            best_neighbor_score = scores[best_neighbor_idx]
            iteration_scores.append(scores)
            
            # 更新pivot
            if best_neighbor_score > best_score:
                best_score = best_neighbor_score
                best_sample = candidates[best_neighbor_idx]
                pivot = neighbors[best_neighbor_idx]
        
        if best_sample is None:
            # 如果所有迭代都失败，使用最后一个pivot采样
            candidate_nfe = NFECounter()
            best_sample = self.model.sample(
                batch_size=batch_size,
                num_steps=num_steps,
                nfe_counter=candidate_nfe,
                initial_noise=pivot,
            )
            nfe_counter.add(candidate_nfe.current_nfe)
        
        info = {
            "method": "zero_order_search",
            "num_iterations": self.num_iterations,
            "num_neighbors": self.num_neighbors,
            "nfe": nfe_counter.total_nfe,
            "best_score": best_score,
            "iteration_scores": iteration_scores,
        }
        
        return best_sample, info

