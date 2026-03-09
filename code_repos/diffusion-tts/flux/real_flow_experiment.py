"""
Real Flow Model Experiment with Global Scheduling

Uses PixArt-Sigma (a real rectified flow model) to demonstrate:
1. ODE-to-SDE conversion for noise trajectory search
2. Our global scheduling (offline + online) vs naive uniform allocation
3. Actual GPU experiments with real model inference

Usage:
    python flux/real_flow_experiment.py --run_comparison --scorer brightness --nfe 200
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
from tqdm import tqdm
import time

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'flux'))

from diffusers import PixArtSigmaPipeline, DPMSolverMultistepScheduler


@dataclass
class FlowSearchConfig:
    """Configuration for flow model search."""
    num_steps: int = 20
    noise_scale: float = 0.3  # Alpha for stochastic injection
    total_nfe: int = 200
    guidance_scale: float = 4.5  # Default CFG for PixArt


class FlowNoiseSearch:
    """
    Noise trajectory search for flow models with SDE sampling.

    Converts the deterministic ODE: dx = v(x,t) dt
    To SDE: dx = [v(x,t) + (g²/2)*score(x,t)] dt + g(t) dW

    Based on Dockhorn et al. and Kim et al.
    """

    def __init__(
        self,
        pipe: PixArtSigmaPipeline,
        config: FlowSearchConfig,
        scorer: Callable,
        device: str = "cuda",
    ):
        self.pipe = pipe
        self.config = config
        self.scorer = scorer
        self.device = device

        # Get transformer and VAE
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

    def get_noise_coefficient(self, t: float) -> float:
        """
        Compute diffusion coefficient g(t) = alpha * sqrt(t).
        NonSingular schedule: small noise near t=0 (data), larger near t=1 (noise).
        """
        return self.config.noise_scale * np.sqrt(max(t, 1e-6))

    @torch.no_grad()
    def search_step(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        added_cond_kwargs: dict,
        timestep: torch.Tensor,
        t_value: float,
        K: int,
        guidance_scale: float = 4.5,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Search over K noise trajectories at one timestep.

        Args:
            latents: Current latent state [B, C, H, W]
            prompt_embeds: Text embeddings
            prompt_attention_mask: Attention mask
            added_cond_kwargs: Additional conditioning
            timestep: Current timestep tensor
            t_value: Current timestep value (normalized 0-1)
            K: Number of noise samples to try
            guidance_scale: CFG scale

        Returns:
            best_latents: Best latent found
            best_score: Score of best candidate
            variance: Score variance across candidates
        """
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
        timestep_expanded = timestep.expand(latent_model_input.shape[0])

        # Predict noise
        noise_pred = self.transformer(
            latent_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timestep_expanded,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # Apply CFG
        if guidance_scale > 1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute diffusion coefficient for SDE
        g_t = self.get_noise_coefficient(t_value)

        # Compute dt from scheduler
        step_idx_list = (self.scheduler.timesteps == timestep).nonzero()
        if len(step_idx_list) == 0:
            dt = -50.0  # Default
        else:
            step_idx = step_idx_list[0].item()
            if step_idx < len(self.scheduler.timesteps) - 1:
                t_curr = self.scheduler.timesteps[step_idx].item()
                t_next = self.scheduler.timesteps[step_idx + 1].item()
                dt = t_next - t_curr
            else:
                dt = -timestep.item() / 2

        # Normalize dt to [0,1] range (scheduler uses 0-1000)
        dt_normalized = dt / 1000.0

        best_latents = None
        best_score = float('-inf')
        all_scores = []

        for k in range(K):
            # Sample noise for SDE injection
            noise = torch.randn_like(latents)

            # SDE step: deterministic + stochastic
            # x_{t-dt} = x_t + dt*noise_pred + sqrt(|dt|)*g(t)*noise
            candidate_latents = (
                latents
                + dt_normalized * noise_pred
                + np.sqrt(abs(dt_normalized)) * g_t * noise
            )

            # Decode to image for scoring
            candidate_image = self.decode_latents(candidate_latents)

            # Score the candidate
            score = self.scorer(candidate_image).item()
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_latents = candidate_latents.clone()

        if best_latents is None:
            # Fallback to deterministic step
            best_latents = latents + dt_normalized * noise_pred
            best_score = 0.0

        variance = np.var(all_scores) if len(all_scores) > 1 else 0.0
        return best_latents, best_score, variance

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to image using VAE."""
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image


class NaiveScheduler:
    """Uniform K allocation across all timesteps."""

    def __init__(self, total_nfe: int, num_steps: int):
        self.K_per_step = max(1, total_nfe // num_steps)
        self.num_steps = num_steps
        self.remaining = total_nfe

    def get_K(self, step_idx: int) -> int:
        K = min(self.K_per_step, self.remaining)
        return K

    def use_budget(self, amount: int):
        self.remaining -= amount


class OnlineScheduler:
    """
    Our offline + online scheduling algorithm.
    - Offline: Non-uniform K based on timestep importance (middle > ends)
    - Online: Tracks statistics for early stopping
    """

    def __init__(
        self,
        total_nfe: int,
        num_steps: int,
        high_region: Tuple[int, int] = (4, 14),  # Middle region
        high_K_ratio: float = 1.5,
    ):
        self.total_nfe = total_nfe
        self.num_steps = num_steps
        self.high_region = high_region

        # Compute offline allocation
        self.K_allocation = self._compute_offline_allocation(high_K_ratio)
        self.remaining = total_nfe

        # Online statistics
        self.historical_gains = []
        self.historical_variances = []

    def _compute_offline_allocation(self, high_K_ratio: float) -> List[int]:
        """Compute non-uniform K allocation based on timestep importance."""
        K_base = self.total_nfe / self.num_steps

        allocation = []
        for i in range(self.num_steps):
            if self.high_region[0] <= i <= self.high_region[1]:
                K = int(K_base * high_K_ratio)
            else:
                K = int(K_base / high_K_ratio)
            allocation.append(max(1, K))

        # Normalize to match budget
        total = sum(allocation)
        scale = self.total_nfe / total
        allocation = [max(1, int(k * scale)) for k in allocation]

        # Adjust remainder
        diff = self.total_nfe - sum(allocation)
        for i in range(abs(diff)):
            idx = (self.high_region[0] + i) % self.num_steps
            allocation[idx] += 1 if diff > 0 else -1
            allocation[idx] = max(1, allocation[idx])

        return allocation

    def get_K(self, step_idx: int) -> int:
        K = min(self.K_allocation[step_idx], self.remaining)
        return K

    def use_budget(self, amount: int):
        self.remaining -= amount

    def update_history(self, gain: float, variance: float):
        self.historical_gains.append(gain)
        self.historical_variances.append(variance)


class BrightnessScorer:
    """Score images based on perceived luminance."""

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            # Apply luminance formula: 0.2126*R + 0.7152*G + 0.0722*B
            if images.size(1) == 3:
                weights = torch.tensor([0.2126, 0.7152, 0.0722], device=images.device).view(1, 3, 1, 1)
                luminance = (images * weights).sum(dim=1).mean(dim=(1, 2))
            else:
                luminance = images.mean(dim=(1, 2, 3))
            return luminance.mean()
        return images.mean()


def run_flow_search(
    pipe: PixArtSigmaPipeline,
    config: FlowSearchConfig,
    scheduler_type: str,
    scorer: Callable,
    prompt: str,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[float, int, float]:
    """
    Run flow noise search with specified scheduler.

    Returns:
        final_score: Score of final generated image
        total_nfe_used: Total NFE actually used
        time_elapsed: Time taken in seconds
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = pipe.device

    # Initialize search
    search = FlowNoiseSearch(pipe, config, scorer, device)

    # Initialize scheduler
    if scheduler_type == "naive":
        budget_scheduler = NaiveScheduler(config.total_nfe, config.num_steps)
    else:
        budget_scheduler = OnlineScheduler(
            config.total_nfe,
            config.num_steps,
            high_region=(4, 14),
            high_K_ratio=1.5,
        )

    # Encode prompt
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
        pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            do_classifier_free_guidance=config.guidance_scale > 1,
            num_images_per_prompt=1,
            device=device,
        )
    )

    # Concatenate for CFG
    if config.guidance_scale > 1:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])

    # Get latent dimensions
    height, width = 512, 512
    latent_channels = pipe.transformer.config.in_channels

    # Initialize latents
    latents = torch.randn(
        (1, latent_channels, height // 8, width // 8),
        device=device,
        dtype=pipe.transformer.dtype,
    )

    # Prepare added conditions (resolution, aspect ratio)
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

    # Set up timesteps
    pipe.scheduler.set_timesteps(config.num_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    total_nfe_used = 0
    prev_score = float('-inf')
    start_time = time.time()

    for step_idx, t in enumerate(tqdm(timesteps, desc=f"{scheduler_type}", disable=not verbose)):
        t_value = t.item() / 1000.0  # Normalize to [0, 1]

        K = budget_scheduler.get_K(step_idx)
        if K <= 0:
            continue

        # Run search at this step
        latents, best_score, variance = search.search_step(
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            timestep=t,
            t_value=t_value,
            K=K,
            guidance_scale=config.guidance_scale,
        )

        total_nfe_used += K
        budget_scheduler.use_budget(K)

        if scheduler_type == "online":
            gain = best_score - prev_score if prev_score != float('-inf') else best_score
            budget_scheduler.update_history(gain, variance)

        prev_score = best_score

    # Decode final image
    final_image = search.decode_latents(latents)
    final_score = scorer(final_image).item()

    elapsed = time.time() - start_time

    return final_score, total_nfe_used, elapsed


def run_comparison(
    model_id: str,
    scorer_name: str,
    nfe_budgets: List[int],
    n_seeds: int = 3,
    prompt: str = "a bright sunny landscape with mountains",
):
    """Run comparison between naive and our method on real flow model."""

    print(f"\n{'='*60}")
    print(f"Flow Model Real Experiment: {scorer_name}")
    print(f"Model: {model_id}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading PixArt-Sigma model...")
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()  # Save GPU memory
    print("Model loaded.\n")

    # Initialize scorer
    scorer = BrightnessScorer()

    results = {"naive": {}, "online": {}}

    for nfe in nfe_budgets:
        print(f"\n--- NFE Budget: {nfe} ---")

        config = FlowSearchConfig(
            num_steps=20,
            noise_scale=0.3,
            total_nfe=nfe,
            guidance_scale=4.5,
        )

        for method in ["naive", "online"]:
            scores = []
            times = []

            for seed in range(n_seeds):
                score, nfe_used, elapsed = run_flow_search(
                    pipe, config, method, scorer,
                    prompt=prompt,
                    seed=seed,
                    verbose=False,
                )
                scores.append(score)
                times.append(elapsed)
                print(f"    {method} seed={seed}: score={score:.4f}")

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            mean_time = np.mean(times)
            results[method][nfe] = (mean_score, std_score, mean_time)

            print(f"  {method:8s}: {mean_score:.4f} ± {std_score:.4f} ({mean_time:.1f}s)")

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary Table (for paper)")
    print(f"{'='*60}")
    print(f"{'NFE':>6s} | {'Naive':>20s} | {'Online (Ours)':>20s} | {'Δ':>8s}")
    print("-" * 60)

    for nfe in nfe_budgets:
        naive_mean, naive_std, _ = results["naive"][nfe]
        online_mean, online_std, _ = results["online"][nfe]
        delta = online_mean - naive_mean
        print(f"{nfe:6d} | {naive_mean:.4f} ± {naive_std:.4f} | "
              f"{online_mean:.4f} ± {online_std:.4f} | {delta:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Real Flow Model Experiment')
    parser.add_argument('--model_id', type=str, default='PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers',
                        help='Flow model to use')
    parser.add_argument('--method', choices=['naive', 'online'], default='naive',
                        help='Scheduling method')
    parser.add_argument('--scorer', choices=['brightness'], default='brightness',
                        help='Scorer to use')
    parser.add_argument('--nfe', type=int, default=200, help='Total NFE budget')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--prompt', type=str, default='a bright sunny landscape with mountains',
                        help='Text prompt')
    parser.add_argument('--run_comparison', action='store_true',
                        help='Run full comparison experiment')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.run_comparison:
        run_comparison(
            model_id=args.model_id,
            scorer_name=args.scorer,
            nfe_budgets=[100, 200],
            n_seeds=3,
            prompt=args.prompt,
        )
    else:
        # Single run
        print(f"Loading model {args.model_id}...")
        pipe = PixArtSigmaPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")

        scorer = BrightnessScorer()
        config = FlowSearchConfig(
            num_steps=20,
            noise_scale=0.3,
            total_nfe=args.nfe,
        )

        print(f"Running {args.method} scheduler with NFE={args.nfe}")
        score, nfe_used, elapsed = run_flow_search(
            pipe, config, args.method, scorer,
            prompt=args.prompt,
            seed=args.seed,
            verbose=args.verbose,
        )
        print(f"\nFinal score: {score:.4f}, NFE used: {nfe_used}, Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
