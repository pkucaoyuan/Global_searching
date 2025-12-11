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
        
        # Sample N complete trajectories
        candidates = []
        scores = []
        
        for i in range(n_candidates):
            # Sample initial noise (or use provided)
            if initial_noise is None:
                noise = self.model.sample_noise(batch_size, self.model.image_size)
            else:
                noise = initial_noise.clone()
                if i > 0:  # Add small perturbation for other candidates
                    noise = noise + torch.randn_like(noise) * 0.01
            
            # Sample complete trajectory
            candidate_nfe = NFECounter()
            x_0 = self.model.sample(
                batch_size=batch_size,
                num_steps=num_steps,
                nfe_counter=candidate_nfe,
                initial_noise=noise,
            )
            candidates.append(x_0)
            
            # Score final image
            with torch.no_grad():
                # 获取class_labels（如果提供）
                class_labels = kwargs.get("class_labels", None)
                score = self.verifier.score(x_0, class_labels=class_labels)
                if score.numel() == 1:
                    score_val = score.item()
                else:
                    score_val = score.mean().item()
                # 检查NaN
                if np.isnan(score_val) or np.isinf(score_val):
                    print(f"Warning: Invalid score value {score_val} for candidate {i}")
                    score_val = 0.0
                scores.append(score_val)
            
            # Accumulate NFE
            nfe_counter.increment(candidate_nfe.current_nfe)
        
        # Select best candidate
        best_idx = torch.tensor(scores).argmax().item()
        best_sample = candidates[best_idx]
        
        return best_sample, {
            "method": "best_of_n",
            "n_candidates": n_candidates,
            "nfe": nfe_counter.current_nfe,
            "scores": scores,
            "best_idx": best_idx,
            "best_score": scores[best_idx],
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
                
                with nfe_counter.count(n_candidates):
                    x_prev_candidates = self.model.denoise_step(x_cur_expanded, t)
                
                # Score all candidates (using x_0 estimate if available)
                with nfe_counter.count(n_candidates):
                    scores = self.verifier.score(x_prev_candidates)
                
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
                
                # Denoise all candidates
                all_noises = torch.cat(candidate_noises, dim=0)
                x_cur_expanded = x_cur.repeat(n_candidates, 1, 1, 1)
                
                with nfe_counter.count(n_candidates):
                    x_prev_candidates = self.model.denoise_step(x_cur_expanded, t)
                
                # Score all candidates
                with nfe_counter.count(n_candidates):
                    scores = self.verifier.score(x_prev_candidates)
                
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

