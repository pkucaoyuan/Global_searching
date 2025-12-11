# src/search/diffusion_tts_search.py

"""
Local Search methods extracted from diffusion-tts (Ramesh et al., 2025):
- Best-of-N (Rejection Sampling)
- Zero-Order Search
- ε-greedy Search

Paper: "Test-Time Scaling of Diffusion Models via Noise Trajectory Search"
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
from ..models.base_model import BaseDiffusionModel
from ..verifiers.base_verifier import BaseVerifier
from ..utils.nfe_counter import NFECounter
from .base_search import BaseSearch
from tqdm import tqdm


class BestOfNSearch(BaseSearch):
    """
    Best-of-N (Rejection Sampling) from diffusion-tts.
    
    Algorithm:
    1. Sample N complete trajectories from root to leaf
    2. Score all final images with verifier
    3. Select the trajectory with highest score
    
    NFE: N * num_steps (one full trajectory per candidate)
    """
    def __init__(
        self,
        model: BaseDiffusionModel,
        verifier: BaseVerifier,
        nfe_budget: int = 0,
        n_candidates: int = 4
    ):
        super().__init__(model, verifier, nfe_budget)
        self.n_candidates = n_candidates

    def search(
        self,
        initial_noise: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_steps: int = 50,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Best-of-N (Rejection Sampling): Sample N complete trajectories and select best.
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        n_candidates = kwargs.get("n_candidates", self.n_candidates)
        device = kwargs.get("device", "cuda")
        
        # 根据原始实现，Best-of-N应该：
        # 1. 使用相同的初始噪声，扩展到N倍batch
        # 2. 在每个时间步，使用不同的随机噪声
        # 3. 最后评分并选择最好的
        
        class_labels = kwargs.get("class_labels", None)
        device = kwargs.get("device", "cuda")
        
        # 根据原始实现，Best-of-N使用相同的初始噪声，扩展到N倍
        # 原始：x_next_expanded = x_next.repeat_interleave(N, dim=0)
        # 其中x_next是 latents * t_steps[0]，latents是标准正态噪声
        
        # 准备初始噪声（如果提供，否则使用模型生成）
        # 注意：在EDMModel.sample()中会处理噪声的缩放（乘以t_steps[0]）
        if initial_noise is None:
            # 生成标准正态噪声（原始实现中的latents）
            base_noise = torch.randn(
                batch_size,
                self.model.num_channels,
                self.model.image_size,
                self.model.image_size,
                device=device,
                dtype=torch.float64
            )
        else:
            base_noise = initial_noise.to(device).to(torch.float64)
        
        # 扩展到N倍：每个batch样本重复N次
        # [batch_size, C, H, W] -> [batch_size * N, C, H, W]
        # 注意：在EDMModel.sample()中会乘以t_steps[0]
        x_expanded = base_noise.repeat_interleave(n_candidates, dim=0)
        
        # 扩展class_labels（如果有）
        class_labels_expanded = None
        if class_labels is not None:
            class_labels_expanded = class_labels.repeat_interleave(n_candidates, dim=0).to(device)
        
        # 运行完整的采样流程（batch_size * N个样本）
        candidate_nfe = NFECounter()
        x_final_expanded = self.model.sample(
            batch_size=batch_size * n_candidates,
            num_steps=num_steps,
            nfe_counter=candidate_nfe,
            initial_noise=x_expanded,
            class_labels=class_labels_expanded,
        )
        
        # 转换为uint8用于评分（按照原始实现）
        # 原始：image_for_scoring = (x_next_expanded * 127.5 + 128).clip(0, 255).to(torch.uint8)
        images_for_scoring = (x_final_expanded * 127.5 + 128).clip(0, 255).to(torch.uint8)
        
        # 评分
        with torch.no_grad():
            scores_tensor = self.verifier.score(images_for_scoring, class_labels=class_labels_expanded)
            if scores_tensor.dim() == 0:
                scores_tensor = scores_tensor.unsqueeze(0)
        
        # 重塑为 [batch_size, N]
        scores = scores_tensor.view(batch_size, n_candidates)
        
        # 选择最好的候选
        best_indices = scores.argmax(dim=1)  # [batch_size]
        
        # 重塑x_final_expanded为 [batch_size, N, C, H, W]
        x_final_reshaped = x_final_expanded.view(batch_size, n_candidates, *x_final_expanded.shape[1:])
        
        # 选择最好的候选（按照原始实现）
        best_sample = torch.stack([
            x_final_reshaped[i, idx] 
            for i, idx in enumerate(best_indices)
        ])  # [batch_size, C, H, W]
        
        # 记录所有分数（用于返回信息）
        all_scores_list = scores.cpu().tolist()
        
        # Accumulate NFE
        nfe_counter.increment(candidate_nfe.current_nfe)
        
        # 获取每个样本的最佳分数
        best_scores_per_sample = scores[torch.arange(batch_size, device=device), best_indices].cpu().tolist()
        
        return best_sample, {
            "method": "best_of_n",
            "n_candidates": n_candidates,
            "nfe": nfe_counter.current_nfe,
            "scores": all_scores_list,  # [batch_size, n_candidates]
            "best_indices": best_indices.cpu().tolist(),  # [batch_size]
            "best_scores": best_scores_per_sample,  # [batch_size]
            "best_score": best_scores_per_sample[0] if batch_size == 1 else np.mean(best_scores_per_sample),
        }


class ZeroOrderSearchTTS(BaseSearch):
    """
    Zero-Order Search from diffusion-tts.
    
    Algorithm:
    - At each timestep, start with a random pivot noise
    - For K iterations:
      - Generate N candidate noises by perturbing pivot (within λ-radius ball)
      - Denoise all candidates and score them (using x_0 estimate)
      - Select best candidate as new pivot
    - Use final pivot for denoising step
    
    NFE: num_steps * (K * N * 2 + 1) (per timestep: K iterations * N candidates * (denoise + verify) + final denoise)
    """
    def __init__(
        self,
        model: BaseDiffusionModel,
        verifier: BaseVerifier,
        nfe_budget: int = 0,
        n_candidates: int = 4,
        search_steps: int = 20,
        lambda_param: float = 0.15,
    ):
        super().__init__(model, verifier, nfe_budget)
        self.n_candidates = n_candidates
        self.search_steps = search_steps
        self.lambda_param = lambda_param
        # Scale lambda by image dimensions (as in paper)
        # For 64x64 RGB: sqrt(3 * 64 * 64) = sqrt(12288) ≈ 110.85
        # We'll compute this dynamically based on actual image size
        self.lambda_param_base = lambda_param

    def search(
        self,
        initial_noise: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_steps: int = 50,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Zero-Order Search: At each timestep, perform K iterations of local search.
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        device = kwargs.get("device", "cuda")
        n_candidates = kwargs.get("n_candidates", self.n_candidates)
        K = kwargs.get("search_steps", self.search_steps)
        lambda_param = kwargs.get("lambda_param", self.lambda_param_base)
        
        # Compute lambda_scaled based on actual image size
        image_size = self.model.image_size
        num_channels = self.model.num_channels
        lambda_scaled = lambda_param * np.sqrt(num_channels * image_size * image_size)
        
        # Initialize
        if initial_noise is None:
            x = self.model.sample_noise(batch_size, image_size)
        else:
            x = initial_noise.to(device)
        
        history_scores_all_steps = []
        
        # Main sampling loop
        for step_idx in tqdm(range(num_steps), desc="Zero-Order Search"):
            t = num_steps - 1 - step_idx  # Time step (counting down)
            x_cur = x
            
            # Initialize pivot noise for this timestep
            pivot_noise = torch.randn_like(x_cur)
            history_scores_this_step = []
            
            # K iterations of local search at this timestep
            for k in range(K):
                # Generate N candidate noises by perturbing pivot
                candidate_noises = []
                for n in range(n_candidates):
                    # Generate random unit direction
                    random_direction = torch.randn_like(pivot_noise)
                    dims = tuple(range(1, random_direction.dim()))
                    random_direction = random_direction / torch.norm(
                        random_direction, p=2, dim=dims, keepdim=True
                    ).clamp_min(1e-12)
                    
                    # Scale by random factor between 0 and lambda
                    scale = torch.rand(
                        (random_direction.shape[0],) + (1,) * (random_direction.dim() - 1),
                        device=device
                    ) * lambda_scaled
                    
                    candidate_noise = pivot_noise + scale * random_direction
                    candidate_noises.append(candidate_noise)
                
                # Denoise all candidates
                all_noises = torch.cat(candidate_noises, dim=0)
                x_cur_expanded = x_cur.repeat(n_candidates, 1, 1, 1)
                
                # 需要计算时间步信息
                num_steps = kwargs.get("num_steps", 50)
                step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
                t_steps = (
                    self.model.sigma_max ** (1 / self.model.rho) + 
                    step_indices / (num_steps - 1) * 
                    (self.model.sigma_min ** (1 / self.model.rho) - self.model.sigma_max ** (1 / self.model.rho))
                ) ** self.model.rho
                t_steps = torch.cat([self.model.model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
                
                t_cur = t_steps[t] if t < len(t_steps) else torch.zeros_like(t_steps[0])
                t_next = t_steps[t + 1] if t < len(t_steps) - 1 else torch.zeros_like(t_cur)
                
                # 根据原始实现，需要在x_cur_expanded上添加噪声，然后去噪
                # 原始：step函数内部处理噪声添加
                # 我们需要直接调用denoise_step，但需要确保参数正确
                
                class_labels = kwargs.get("class_labels", None)
                class_labels_expanded = None
                if class_labels is not None:
                    class_labels_expanded = class_labels.repeat(n_candidates, 1)
                
                # 添加噪声到x_cur_expanded（模拟原始实现的step函数）
                # 原始实现中，step函数会添加噪声：x_hat = x_cur + noise
                # 这里我们直接在x_cur_expanded上添加候选噪声
                gamma = min(40.0 / num_steps, np.sqrt(2) - 1) if 0.05 <= t_cur <= 50.0 else 0
                t_hat = self.model.model.round_sigma(t_cur + gamma * t_cur) if gamma > 0 else t_cur
                
                # 对于每个候选噪声，添加到x_cur_expanded
                # 原始实现：x_hat = x_cur + (t_hat^2 - t_cur^2)^0.5 * S_noise * eps_i
                S_noise = 1.003
                if gamma > 0:
                    x_hat_expanded = x_cur_expanded + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * all_noises
                else:
                    x_hat_expanded = x_cur_expanded
                
                # 去噪（获取x_0估计）
                # 根据原始实现，我们需要调用模型获取denoised（x_0估计）
                with nfe_counter.count(n_candidates):
                    denoised_candidates = self.model.model(x_hat_expanded, t_hat, class_labels_expanded).to(torch.float64)
                
                # 评分使用x_0估计（按照原始实现）
                # 原始：x_for_scoring = x0_candidates.reshape(-1, *x0_candidates.shape[2:])
                # 原始：x_for_scoring = (x_for_scoring * 127.5 + 128).clip(0, 255).to(torch.uint8)
                x0_for_scoring = (denoised_candidates * 127.5 + 128).clip(0, 255).to(torch.uint8)
                
                # 评分
                timesteps = torch.zeros(x0_for_scoring.shape[0], device=device)
                with torch.no_grad():
                    scores = self.verifier.score(x0_for_scoring, class_labels=class_labels_expanded)
                
                scores = scores.view(n_candidates, batch_size)
                history_scores_this_step.append(scores.cpu().numpy())
                
                # Find best noise for each batch element
                best_indices = scores.argmax(dim=0)
                all_noises_reshaped = all_noises.view(
                    n_candidates, batch_size, *x_cur.shape[1:]
                )
                
                # Update pivot to best noise
                pivot_noise = torch.stack([
                    all_noises_reshaped[best_idx, batch_idx]
                    for batch_idx, best_idx in enumerate(best_indices)
                ])
            
            # Use final pivot for actual denoising step
            with nfe_counter.count():
                x = self.model.denoise_step(x_cur, t)
            
            history_scores_all_steps.append(history_scores_this_step)
        
        return x, {
            "method": "zero_order_tts",
            "n_candidates": n_candidates,
            "K": K,
            "lambda_param": lambda_param,
            "nfe": nfe_counter.current_nfe,
            "history_scores": history_scores_all_steps,
        }


class EpsilonGreedySearch(BaseSearch):
    """
    ε-greedy Search from diffusion-tts.
    
    Algorithm:
    - Similar to Zero-Order, but with probability ε, use a fresh Gaussian sample
    - With probability (1-ε), use perturbed pivot (local search)
    - This allows global exploration while maintaining local exploitation
    
    NFE: Same as Zero-Order (num_steps * (K * N * 2 + 1))
    """
    def __init__(
        self,
        model: BaseDiffusionModel,
        verifier: BaseVerifier,
        nfe_budget: int = 0,
        n_candidates: int = 4,
        search_steps: int = 20,
        lambda_param: float = 0.15,
        epsilon: float = 0.4,
    ):
        super().__init__(model, verifier, nfe_budget)
        self.n_candidates = n_candidates
        self.search_steps = search_steps
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.lambda_param_base = lambda_param

    def search(
        self,
        initial_noise: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_steps: int = 50,
        nfe_counter: Optional[NFECounter] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        ε-greedy Search: Similar to Zero-Order but with ε probability of using fresh Gaussian.
        """
        if nfe_counter is None:
            nfe_counter = NFECounter()
        
        device = kwargs.get("device", "cuda")
        n_candidates = kwargs.get("n_candidates", self.n_candidates)
        K = kwargs.get("search_steps", self.search_steps)
        lambda_param = kwargs.get("lambda_param", self.lambda_param_base)
        epsilon = kwargs.get("epsilon", self.epsilon)
        
        # Compute lambda_scaled
        image_size = self.model.image_size
        num_channels = self.model.num_channels
        lambda_scaled = lambda_param * np.sqrt(num_channels * image_size * image_size)
        
        # Initialize
        if initial_noise is None:
            x = self.model.sample_noise(batch_size, image_size)
        else:
            x = initial_noise.to(device)
        
        history_scores_all_steps = []
        random_choices_all_steps = []
        
        # Main sampling loop
        for step_idx in tqdm(range(num_steps), desc="Zero-Order Search"):
            t = num_steps - 1 - step_idx
            x_cur = x
            
            # Initialize pivot noise for this timestep
            pivot_noise = torch.randn_like(x_cur)
            history_scores_this_step = []
            random_choices_this_step = []
            
            # K iterations of local search at this timestep
            for k in range(K):
                # Generate N candidate noises
                candidate_noises = []
                for n in range(n_candidates):
                    # With probability ε, use fresh Gaussian
                    # With probability (1-ε), use perturbed pivot
                    if torch.rand(1, device=device).item() < epsilon:
                        # Fresh Gaussian sample
                        candidate_noise = torch.randn_like(x_cur)
                        random_choices_this_step.append((k, n, True))
                    else:
                        # Perturbed pivot (local search)
                        random_direction = torch.randn_like(pivot_noise)
                        dims = tuple(range(1, random_direction.dim()))
                        random_direction = random_direction / torch.norm(
                            random_direction, p=2, dim=dims, keepdim=True
                        ).clamp_min(1e-12)
                        
                        scale = torch.rand(
                            (random_direction.shape[0],) + (1,) * (random_direction.dim() - 1),
                            device=device
                        ) * lambda_scaled
                        
                        candidate_noise = pivot_noise + scale * random_direction
                        random_choices_this_step.append((k, n, False))
                    
                    candidate_noises.append(candidate_noise)
                
                # Denoise all candidates（与Zero-Order Search相同逻辑）
                all_noises = torch.cat(candidate_noises, dim=0)  # [N*batch_size, C, H, W]
                x_cur_expanded = x_cur.repeat(n_candidates, 1, 1, 1)  # [N*batch_size, C, H, W]
                
                # 计算时间步信息
                num_steps = kwargs.get("num_steps", 50)
                step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
                t_steps = (
                    self.model.sigma_max ** (1 / self.model.rho) + 
                    step_indices / (num_steps - 1) * 
                    (self.model.sigma_min ** (1 / self.model.rho) - self.model.sigma_max ** (1 / self.model.rho))
                ) ** self.model.rho
                t_steps = torch.cat([self.model.model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
                
                class_labels = kwargs.get("class_labels", None)
                class_labels_expanded = None
                if class_labels is not None:
                    class_labels_expanded = class_labels.repeat(n_candidates, 1)
                
                # 调用denoise_step，传入噪声
                x_candidates, x0_candidates = self.model.denoise_step(
                    x_cur_expanded, t,
                    class_labels=class_labels_expanded,
                    num_steps=num_steps,
                    t_steps=t_steps,
                    noise=all_noises,
                    **kwargs
                )
                
                # 重塑为 [N, batch_size, C, H, W]
                total_images = x_candidates.shape[0]
                channels, height, width = x_cur.shape[1:]
                effective_batch_size = total_images // n_candidates
                x0_candidates_reshaped = x0_candidates.reshape(n_candidates, effective_batch_size, channels, height, width)
                
                # 评分使用x_0估计
                x0_for_scoring = x0_candidates_reshaped.reshape(-1, *x0_candidates_reshaped.shape[2:])
                x0_for_scoring_uint8 = (x0_for_scoring * 127.5 + 128).clip(0, 255).to(torch.uint8)
                
                # 评分
                timesteps = torch.zeros(x0_for_scoring_uint8.shape[0], device=device)
                with torch.no_grad():
                    scores = self.verifier.score(x0_for_scoring_uint8, class_labels=class_labels_expanded)
                
                scores = scores.view(n_candidates, batch_size)
                history_scores_this_step.append(scores.cpu().numpy())
                
                # Find best noise for each batch element
                best_indices = scores.argmax(dim=0)  # [batch_size]
                all_noises_reshaped = all_noises.view(n_candidates, batch_size, *all_noises.shape[1:])
                
                # Update pivot to best noise
                pivot_noise = torch.stack([
                    all_noises_reshaped[best_idx, batch_idx]
                    for batch_idx, best_idx in enumerate(best_indices)
                ])
            
            # Use final pivot for actual denoising step
            num_steps = kwargs.get("num_steps", 50)
            step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
            t_steps = (
                self.model.sigma_max ** (1 / self.model.rho) + 
                step_indices / (num_steps - 1) * 
                (self.model.sigma_min ** (1 / self.model.rho) - self.model.sigma_max ** (1 / self.model.rho))
            ) ** self.model.rho
            t_steps = torch.cat([self.model.model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
            
            x, _ = self.model.denoise_step(
                x_cur, t,
                class_labels=kwargs.get("class_labels", None),
                num_steps=num_steps,
                t_steps=t_steps,
                noise=pivot_noise,
                **kwargs
            )
            
            history_scores_all_steps.append(history_scores_this_step)
            random_choices_all_steps.append(random_choices_this_step)
        
        return x, {
            "method": "epsilon_greedy",
            "n_candidates": n_candidates,
            "K": K,
            "lambda_param": lambda_param,
            "epsilon": epsilon,
            "nfe": nfe_counter.current_nfe,
            "history_scores": history_scores_all_steps,
            "random_choices": random_choices_all_steps,
        }

