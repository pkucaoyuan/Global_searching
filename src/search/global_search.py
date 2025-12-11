"""
Global Search框架
Global Search: 决定在不同步数使用什么search策略，如何分配算力

基于MDP建模：
- State: (xt, t, prompt, history/score)
- Action: (search_mode, budget, primitive_type)
- Reward: Δt = verifier(xt-1) - verifier(xt) - λ·computation
"""

from typing import Tuple, Optional, Dict, Any, List, Callable
from enum import Enum
import torch
from ..models.base_model import BaseDiffusionModel
from ..verifiers.base_verifier import BaseVerifier
from ..utils.nfe_counter import NFECounter
from .local_search import NoSearch, RandomSearch, LocalNoiseSearch, ZeroOrderSearch


class SearchMode(Enum):
    """Search模式枚举"""
    NO_SEARCH = "no_search"
    LIGHT_LOCAL = "light_local"
    HEAVY_LOCAL = "heavy_local"
    GLOBAL_RESAMPLE = "global_resample"


class Action:
    """Global Search的Action"""
    
    def __init__(
        self,
        search_mode: SearchMode,
        budget: int,
        primitive_type: str = "random",
        **kwargs
    ):
        """
        Args:
            search_mode: 搜索模式
            budget: 分配的NFE预算
            primitive_type: 使用的底层search primitive（random, zo, nlg等）
            **kwargs: 其他参数
        """
        self.search_mode = search_mode
        self.budget = budget
        self.primitive_type = primitive_type
        self.params = kwargs
    
    def __repr__(self):
        return f"Action(mode={self.search_mode.value}, budget={self.budget}, primitive={self.primitive_type})"


class State:
    """Global Search的State"""
    
    def __init__(
        self,
        x_t: torch.Tensor,
        t: int,
        prompt: Optional[str] = None,
        history_scores: Optional[List[float]] = None,
        previous_action: Optional[Action] = None,
    ):
        """
        Args:
            x_t: 当前状态（噪声图像）
            t: 当前时间步
            prompt: 提示词（如果有）
            history_scores: 历史verifier分数
            previous_action: 上一个action
        """
        self.x_t = x_t
        self.t = t
        self.prompt = prompt
        self.history_scores = history_scores or []
        self.previous_action = previous_action
    
    def get_score(self) -> float:
        """获取最近的verifier分数"""
        return self.history_scores[-1] if self.history_scores else 0.0
    
    def __repr__(self):
        return f"State(t={self.t}, score={self.get_score()})"


class GlobalSearchPolicy:
    """
    Global Search策略基类
    
    决定在每个时间步选择什么action（search_mode, budget, primitive）
    """
    
    def __init__(self, total_nfe_budget: int):
        """
        Args:
            total_nfe_budget: 总的NFE预算
        """
        self.total_nfe_budget = total_nfe_budget
        self.used_budget = 0
    
    def decide_action(
        self,
        state: State,
        num_steps: int,
    ) -> Action:
        """
        根据当前state决定action
        
        Args:
            state: 当前状态
            num_steps: 总采样步数
        
        Returns:
            action: 要执行的action
        """
        raise NotImplementedError
    
    def reset(self):
        """重置策略状态"""
        self.used_budget = 0


class FixedBudgetPolicy(GlobalSearchPolicy):
    """
    固定分配策略：根据不同step的重要性固定分配budget
    
    例如：T2I任务，前1/3步分配60% budget，后2/3步分配40%
    """
    
    def __init__(
        self,
        total_nfe_budget: int,
        early_ratio: float = 0.6,
        early_steps_ratio: float = 0.33,
        search_mode_early: SearchMode = SearchMode.HEAVY_LOCAL,
        search_mode_late: SearchMode = SearchMode.LIGHT_LOCAL,
    ):
        """
        Args:
            early_ratio: 前期分配的budget比例
            early_steps_ratio: 前期的步数比例
            search_mode_early: 前期使用的search模式
            search_mode_late: 后期使用的search模式
        """
        super().__init__(total_nfe_budget)
        self.early_ratio = early_ratio
        self.early_steps_ratio = early_steps_ratio
        self.search_mode_early = search_mode_early
        self.search_mode_late = search_mode_late
        
        # 计算每步的budget分配
        self.budget_per_step_early = None
        self.budget_per_step_late = None
    
    def decide_action(
        self,
        state: State,
        num_steps: int,
    ) -> Action:
        """根据步数决定action"""
        # 计算分界点
        early_steps = int(num_steps * self.early_steps_ratio)
        early_budget = int(self.total_nfe_budget * self.early_ratio)
        late_budget = self.total_nfe_budget - early_budget
        
        # 计算每步budget
        if state.t >= num_steps - early_steps:
            # 前期
            remaining_steps = state.t - (num_steps - early_steps) + 1
            budget = early_budget // early_steps if early_steps > 0 else 0
            search_mode = self.search_mode_early
        else:
            # 后期
            budget = late_budget // (num_steps - early_steps) if (num_steps - early_steps) > 0 else 0
            search_mode = self.search_mode_late
        
        # 确保不超过剩余budget
        remaining_budget = self.total_nfe_budget - self.used_budget
        budget = min(budget, remaining_budget)
        
        return Action(
            search_mode=search_mode,
            budget=budget,
            primitive_type="random",
        )


class AdaptiveThresholdPolicy(GlobalSearchPolicy):
    """
    自适应阈值策略：根据verifier改善情况动态调整
    
    如果Δt < threshold: 增加search budget（当前状态不好）
    如果Δt > threshold: 减少search budget（已经改善）
    """
    
    def __init__(
        self,
        total_nfe_budget: int,
        threshold: float = 0.0,
        base_budget: int = 10,
        max_budget: int = 50,
        min_budget: int = 0,
    ):
        """
        Args:
            threshold: 改善阈值
            base_budget: 基础budget
            max_budget: 最大budget
            min_budget: 最小budget
        """
        super().__init__(total_nfe_budget)
        self.threshold = threshold
        self.base_budget = base_budget
        self.max_budget = max_budget
        self.min_budget = min_budget
    
    def decide_action(
        self,
        state: State,
        num_steps: int,
    ) -> Action:
        """根据历史分数变化决定action"""
        # 计算改善量
        if len(state.history_scores) >= 2:
            delta = state.history_scores[-1] - state.history_scores[-2]
        else:
            delta = 0.0
        
        # 根据改善量调整budget
        if delta < self.threshold:
            # 改善不足，增加budget
            budget = min(self.base_budget * 2, self.max_budget)
            search_mode = SearchMode.HEAVY_LOCAL
        else:
            # 改善良好，减少budget
            budget = max(self.base_budget // 2, self.min_budget)
            search_mode = SearchMode.LIGHT_LOCAL
        
        # 确保不超过剩余budget
        remaining_budget = self.total_nfe_budget - self.used_budget
        budget = min(budget, remaining_budget)
        
        return Action(
            search_mode=search_mode,
            budget=budget,
            primitive_type="random",
        )


class MultiStagePolicy(GlobalSearchPolicy):
    """
    多阶段策略：
    - 前期（t接近T）: 使用heavy search
    - 中期（t中等）: 使用light search
    - 后期（t接近0）: 使用no search
    """
    
    def __init__(
        self,
        total_nfe_budget: int,
        early_ratio: float = 0.5,
        mid_ratio: float = 0.3,
        late_ratio: float = 0.2,
    ):
        """
        Args:
            early_ratio: 前期budget比例
            mid_ratio: 中期budget比例
            late_ratio: 后期budget比例（通常用于no search）
        """
        super().__init__(total_nfe_budget)
        self.early_ratio = early_ratio
        self.mid_ratio = mid_ratio
        self.late_ratio = late_ratio
    
    def decide_action(
        self,
        state: State,
        num_steps: int,
    ) -> Action:
        """根据阶段决定action"""
        # 划分三个阶段
        t_normalized = state.t / num_steps
        
        if t_normalized > 0.66:  # 前期
            budget_ratio = self.early_ratio
            search_mode = SearchMode.HEAVY_LOCAL
            primitive = "zo"
        elif t_normalized > 0.33:  # 中期
            budget_ratio = self.mid_ratio
            search_mode = SearchMode.LIGHT_LOCAL
            primitive = "random"
        else:  # 后期
            budget_ratio = self.late_ratio
            search_mode = SearchMode.NO_SEARCH
            primitive = "none"
        
        # 计算budget（简化：平均分配到该阶段的每一步）
        budget = int(self.total_nfe_budget * budget_ratio / (num_steps / 3))
        
        # 确保不超过剩余budget
        remaining_budget = self.total_nfe_budget - self.used_budget
        budget = min(budget, remaining_budget)
        
        return Action(
            search_mode=search_mode,
            budget=budget,
            primitive_type=primitive,
        )


class GlobalSearch:
    """
    Global Search主类
    
    整合Local Search和Global Search策略，实现完整的采样流程
    """
    
    def __init__(
        self,
        model: BaseDiffusionModel,
        verifier: BaseVerifier,
        policy: GlobalSearchPolicy,
    ):
        """
        Args:
            model: 扩散模型
            verifier: Verifier
            policy: Global Search策略
        """
        self.model = model
        self.verifier = verifier
        self.policy = policy
        
        # 创建Local Search方法池
        self.local_search_pool = {
            "no_search": NoSearch(model, verifier),
            "random": RandomSearch(model, verifier),
            "local": LocalNoiseSearch(model, verifier),
            "zo": ZeroOrderSearch(model, verifier),
        }
    
    def _action_to_local_search(
        self,
        action: Action,
        nfe_counter: NFECounter,
    ) -> BaseSearch:
        """根据action创建对应的Local Search方法"""
        # 根据search_mode和primitive_type选择合适的search方法
        if action.search_mode == SearchMode.NO_SEARCH:
            return self.local_search_pool["no_search"]
        elif action.search_mode == SearchMode.LIGHT_LOCAL:
            if action.primitive_type == "random":
                search = RandomSearch(self.model, self.verifier, nfe_budget=action.budget)
                search.num_candidates = 2  # Light search使用较少候选
                return search
            elif action.primitive_type == "local":
                search = LocalNoiseSearch(self.model, self.verifier, nfe_budget=action.budget)
                search.num_candidates_per_step = 2
                return search
        elif action.search_mode == SearchMode.HEAVY_LOCAL:
            if action.primitive_type == "zo":
                search = ZeroOrderSearch(self.model, self.verifier, nfe_budget=action.budget)
                search.num_iterations = 4
                search.num_neighbors = 8
                return search
            elif action.primitive_type == "random":
                search = RandomSearch(self.model, self.verifier, nfe_budget=action.budget)
                search.num_candidates = 8  # Heavy search使用更多候选
                return search
        
        # 默认返回no_search
        return self.local_search_pool["no_search"]
    
    def sample(
        self,
        initial_noise: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_steps: int = 50,
        prompt: Optional[str] = None,
        nfe_counter: Optional[NFECounter] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        执行Global Search采样
        
        Args:
            initial_noise: 初始噪声
            batch_size: batch大小
            num_steps: 采样步数
            prompt: 提示词
            nfe_counter: NFE计数器
        
        Returns:
            samples: 生成的样本
            info: 包含完整信息的字典
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        self.policy.reset()
        
        # 初始化状态
        if initial_noise is None:
            x_t = self.model.sample_noise(batch_size, self.model.image_size)
        else:
            x_t = initial_noise.clone()
        
        state = State(
            x_t=x_t,
            t=num_steps - 1,
            prompt=prompt,
            history_scores=[],
        )
        
        actions_taken = []
        rewards = []
        
        # 逐步采样
        for step in range(num_steps - 1, -1, -1):
            # 更新state的t
            state.t = step
            state.x_t = x_t
            
            # Global Search决策：选择action
            action = self.policy.decide_action(state, num_steps)
            actions_taken.append(action)
            
            # 执行Local Search
            local_search = self._action_to_local_search(action, nfe_counter)
            
            # 如果action是NO_SEARCH，直接去噪一步
            if action.search_mode == SearchMode.NO_SEARCH:
                with nfe_counter.count():
                    x_t_minus_1, _ = self.model.denoise_step(x_t, step)
            else:
                # 使用Local Search方法
                # 注意：这里需要适配，因为Local Search通常是完整的采样流程
                # 在实际实现中，可能需要修改Local Search以支持单步操作
                # 简化版本：如果budget足够，执行search
                if action.budget > 0:
                    # 执行search（这里简化处理）
                    # 实际实现中，Local Search应该支持单步操作
                    with nfe_counter.count():
                        x_t_minus_1, _ = self.model.denoise_step(x_t, step)
                    # TODO: 实现真正的单步Local Search
                else:
                    with nfe_counter.count():
                        x_t_minus_1, _ = self.model.denoise_step(x_t, step)
            
            # 评估reward
            with torch.no_grad():
                score_t_minus_1 = self.verifier.score(x_t_minus_1)
                score_val_t_minus_1 = score_t_minus_1.item() if score_t_minus_1.numel() == 1 else score_t_minus_1.mean().item()
                
                if state.history_scores:
                    score_t = state.history_scores[-1]
                    delta = score_val_t_minus_1 - score_t
                else:
                    delta = score_val_t_minus_1
                
                # Reward = Δt - λ·computation
                computation_cost = action.budget * 0.01  # λ=0.01，可配置
                reward = delta - computation_cost
                rewards.append(reward)
            
            # 更新state
            state.history_scores.append(score_val_t_minus_1)
            state.previous_action = action
            x_t = x_t_minus_1
            
            # 更新已用budget
            self.policy.used_budget += nfe_counter.total_nfe - sum(r.get('nfe', 0) for r in rewards)
        
        x_0 = x_t
        
        info = {
            "method": "global_search",
            "policy": type(self.policy).__name__,
            "nfe": nfe_counter.total_nfe,
            "actions": [str(a) for a in actions_taken],
            "rewards": rewards,
            "final_score": state.history_scores[-1] if state.history_scores else 0.0,
        }
        
        return x_0, info


