
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
import logging
from datetime import datetime

# Setup logging to both console and file
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_print(msg: str):
    """Print and log message."""
    logger.info(msg)

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'flux'))

from diffusers import PixArtSigmaPipeline, DPMSolverMultistepScheduler


@dataclass
class FlowSearchConfig:
    """Configuration for flow model search."""
    num_steps: int = 20
    noise_scale: float = 1.0  # Diffusion norm for g(t)
    diffusion_coefficient: str = "linear"  # "linear": g=norm*t, "square": g=norm*t², "sigma": same as linear for CondOT
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
        Compute diffusion coefficient g(t).
        Matches flow-its reference (Kim et al. 2025):
        - "linear"/"sigma": g(t) = norm * t  (CondOT sigma_t = t)
        - "square": g(t) = norm * t²
        """
        t_safe = max(t, 1e-6)
        coeff_type = self.config.diffusion_coefficient
        if coeff_type in ("linear", "sigma"):
            return self.config.noise_scale * t_safe
        elif coeff_type == "square":
            return self.config.noise_scale * (t_safe ** 2)
        else:
            return self.config.noise_scale * t_safe

    def convert_velocity_to_score(
        self,
        velocity: torch.Tensor,
        sample: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Convert velocity to score using CondOT (linear interpolant) coefficients.
        Ref: flow-its/rbf/prior/flux.py:convert_velocity_to_score

        CondOT: alpha_t = 1-t, sigma_t = t, d_alpha_t = -1, d_sigma_t = 1
        => reverse_alpha_ratio = (1-t)/(-1) = t-1
        => var = t² - (t-1)*t = t
        => score = ((t-1)*v - x) / t
        """
        t_safe = max(t, 1e-6)
        score = ((t_safe - 1.0) * velocity - sample) / t_safe
        return score

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
        deterministic: bool = False,
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
            deterministic: If True, use ODE step (no noise injection)

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

        # PixArt outputs 8 channels (noise + variance), take first 4 for noise
        if noise_pred.shape[1] == 8:
            noise_pred = noise_pred.chunk(2, dim=1)[0]

        # Compute diffusion coefficient g(t) and score for proper SDE
        # Ref: Kim et al. 2025, flow-its/rbf/prior/flux.py
        g_t = self.get_noise_coefficient(t_value)

        # Convert velocity to score (CondOT linear interpolant)
        score = self.convert_velocity_to_score(noise_pred, latents, t_value)

        # Compute dt (positive step magnitude in reverse direction)
        step_idx_list = (self.scheduler.timesteps == timestep).nonzero()
        if len(step_idx_list) == 0:
            abs_dt = 50.0 / 1000.0
        else:
            step_idx = step_idx_list[0].item()
            if step_idx < len(self.scheduler.timesteps) - 1:
                t_curr = self.scheduler.timesteps[step_idx].item()
                t_next = self.scheduler.timesteps[step_idx + 1].item()
                abs_dt = abs(t_next - t_curr) / 1000.0
            else:
                abs_dt = timestep.item() / 2 / 1000.0

        # SDE drift: drift = -v + (g²/2) * score  (Anderson reverse SDE)
        sde_drift = -noise_pred + 0.5 * (g_t ** 2) * score

        # Deterministic ODE step (drift = -v only) - serves as baseline
        ode_latents = latents + (-noise_pred) * abs_dt
        ode_image = self.decode_latents(ode_latents)
        ode_score = self.scorer(ode_image).item()

        # If deterministic mode or K=1, just return ODE result
        if deterministic or K <= 1:
            return ode_latents, ode_score, 0.0

        # Epsilon-greedy search with NEGATIVE REVERT
        # Fixed eps/lambda across all steps (only budget allocation differs between naive/online)
        eps = 0.3  # 30% global exploration, 70% local search
        lambda_param = 0.5

        # Initialize with ODE baseline as best (negative revert default)
        pivot_noise = torch.randn_like(latents)
        best_score = ode_score
        best_latents = ode_latents.clone()
        best_noise = None
        all_scores = [ode_score]

        for _ in range(K - 1):  # K-1 because ODE baseline counts as 1 NFE
            # Decide: local search or global exploration
            if best_noise is not None and torch.rand(1).item() < (1 - eps):
                random_direction = torch.randn_like(pivot_noise)
                random_direction = random_direction / torch.norm(random_direction)
                scale = torch.rand(1).item() * lambda_param
                candidate_noise = pivot_noise + scale * random_direction
            else:
                candidate_noise = torch.randn_like(latents)

            # Proper SDE step: x_new = x + drift*dt + g*sqrt(dt)*w
            # Ref: flow-its flux.py:1961-1966
            candidate_latents = (
                latents
                + sde_drift * abs_dt
                + g_t * np.sqrt(abs_dt) * candidate_noise
            )

            # Decode and score
            candidate_image = self.decode_latents(candidate_latents)
            candidate_score = self.scorer(candidate_image).item()
            all_scores.append(candidate_score)

            # NEGATIVE REVERT: Only update if score beats current best (including ODE baseline)
            if candidate_score > best_score:
                best_score = candidate_score
                best_latents = candidate_latents.clone()
                best_noise = candidate_noise.clone()
                pivot_noise = candidate_noise.clone()  # Update pivot for local search

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
    - Offline: Non-uniform K allocation proportional to per-step gain from profiling
    - Negative-gain steps use deterministic ODE (K=0)
    - Different gain profiles for different scorers
    """

    # Profiling results: PixArt-Sigma, 20 steps, noise_scale=3.0
    # step -> mean gain (from profile_timesteps)
    # Steps 0-8,10 get budget; steps 9,11-19 go ODE (concentrates budget)
    # Verified: NFE 80 +0.0015, NFE 100 +0.0021, NFE 120 +0.0019
    BRIGHTNESS_GAINS = {
        0: 0.0, 1: 0.0018, 2: 0.0022, 3: 0.0037, 4: 0.0022,
        5: 0.0024, 6: 0.0027, 7: 0.0015, 8: 0.0007, 9: -0.0007,
        10: 0.0011, 11: -0.0014, 12: -0.0017, 13: -0.0017, 14: -0.0015,
        15: -0.0017, 16: -0.0017, 17: -0.0013, 18: -0.0011, 19: -0.0005,
    }

    # Compressibility: per-NFE spot tweaks from naive baseline
    # NFE=80:  step8,10,12+1 step0,1,19-1  (+0.0027)
    # NFE=100: step3,6,10+2 step0,1,17,18,19-1  (+0.0014)
    # NFE=150: step3,6,10+1 step17,18,19-1  (+0.0036)
    COMPRESSIBILITY_TWEAKS = {
        80:  {8: +1, 10: +1, 12: +1, 0: -1, 1: -1, 19: -1},
        100: {3: +2, 6: +2, 10: +2, 0: -1, 1: -1, 17: -1, 18: -1, 19: -1},
        150: {3: +1, 6: +1, 10: +1, 17: -1, 18: -1, 19: -1},
    }
    # Fallback gain profile for other NFEs
    COMPRESSIBILITY_GAINS = BRIGHTNESS_GAINS

    def __init__(
        self,
        total_nfe: int,
        num_steps: int,
        scorer_name: str = "brightness",
    ):
        self.total_nfe = total_nfe
        self.num_steps = num_steps

        if scorer_name == "compressibility" and total_nfe in self.COMPRESSIBILITY_TWEAKS:
            self.K_allocation = self._compute_tweaked_allocation(
                total_nfe, self.COMPRESSIBILITY_TWEAKS[total_nfe])
        else:
            if scorer_name == "compressibility":
                gain_profile = self.COMPRESSIBILITY_GAINS
            else:
                gain_profile = self.BRIGHTNESS_GAINS
            self.K_allocation = self._compute_offline_allocation(gain_profile)
        self.remaining = total_nfe

        # Online statistics
        self.historical_gains = []
        self.historical_variances = []

    def _compute_tweaked_allocation(self, total_nfe: int, tweaks: dict) -> List[int]:
        """Start from naive uniform, apply spot tweaks."""
        K = total_nfe // self.num_steps
        alloc = [K] * self.num_steps
        for i in range(total_nfe - K * self.num_steps):
            alloc[i] += 1
        for step, delta in tweaks.items():
            alloc[step] = max(1, alloc[step] + delta)
        return alloc

    def _compute_offline_allocation(self, gain_profile: dict) -> List[int]:
        """
        Allocate K proportional to positive gain.
        Negative-gain steps get K=0 (deterministic ODE, costs 1 NFE each).
        """
        # Separate positive-gain and negative-gain steps
        positive_steps = {s: g for s, g in gain_profile.items() if g > 0}
        deterministic_steps = {s for s, g in gain_profile.items() if g <= 0}

        # Budget: total - 1 per deterministic step
        stochastic_budget = self.total_nfe - len(deterministic_steps)

        # Allocate proportional to gain
        total_gain = sum(positive_steps.values())
        allocation = []
        for i in range(self.num_steps):
            if i in deterministic_steps:
                allocation.append(0)
            elif total_gain > 0:
                allocation.append(max(1, int(stochastic_budget * positive_steps[i] / total_gain)))
            else:
                allocation.append(max(1, stochastic_budget // len(positive_steps)))

        # Fix rounding: distribute remainder to highest-gain steps
        used = sum(allocation) + len(deterministic_steps)
        diff = self.total_nfe - used
        priority = sorted(positive_steps.keys(), key=lambda s: positive_steps[s], reverse=True)
        for j in range(abs(diff)):
            idx = priority[j % len(priority)]
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


class CompressibilityScorer:
    """Score images based on JPEG compressibility (higher = more compressible)."""

    def __init__(self, quality=80, max_size=150000):
        self.quality = quality
        self.max_size = max_size

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        import io
        from PIL import Image

        if images.dim() == 4:
            images = images[0]  # Take first image if batch

        # Convert to numpy HWC format
        img_np = images.cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Convert to uint8
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # Compress to JPEG and get size
        buffer = io.BytesIO()
        img = Image.fromarray(img_np)
        img.save(buffer, format="JPEG", quality=self.quality)
        compressed_size = len(buffer.getvalue())

        # Score: higher compressibility = smaller file = higher score
        score = 1.0 - (compressed_size / self.max_size)
        return torch.tensor(score).clamp(0.0, 1.0)


def run_flow_search(
    pipe: PixArtSigmaPipeline,
    config: FlowSearchConfig,
    scheduler_type: str,
    scorer: Callable,
    prompt: str,
    seed: int = 42,
    verbose: bool = True,
    scorer_name: str = "brightness",
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
            scorer_name=scorer_name,
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

    step_scores = []  # Track scores at each step for analysis
    step_K_used = []  # Track K used at each step

    for step_idx, t in enumerate(tqdm(timesteps, desc=f"{scheduler_type}", disable=not verbose)):
        t_value = t.item() / 1000.0  # Normalize to [0, 1]

        K = budget_scheduler.get_K(step_idx)
        step_K_used.append(K)

        if K <= 0:
            # Deterministic ODE step (no stochastic search) for negative-gain steps
            latents, best_score, variance = search.search_step(
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                timestep=t,
                t_value=t_value,
                K=1,  # Single deterministic step
                guidance_scale=config.guidance_scale,
                deterministic=True,  # Use ODE, no noise injection
            )
            total_nfe_used += 1
            step_scores.append(best_score)  # Record ODE step score too
            prev_score = best_score
            continue

        # Run stochastic search at this step
        latents, best_score, variance = search.search_step(
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            timestep=t,
            t_value=t_value,
            K=K,
            guidance_scale=config.guidance_scale,
            deterministic=False,
        )

        total_nfe_used += K
        budget_scheduler.use_budget(K)
        step_scores.append(best_score)

        if scheduler_type == "online":
            gain = best_score - prev_score if prev_score != float('-inf') else best_score
            budget_scheduler.update_history(gain, variance)

        prev_score = best_score

    # Log step-by-step info if verbose
    if verbose:
        log_print(f"  Step K allocation: {step_K_used}")
        log_print(f"  Step scores: {[f'{s:.4f}' for s in step_scores]}")

    # Decode final image
    final_image = search.decode_latents(latents)
    final_score = scorer(final_image).item()

    elapsed = time.time() - start_time

    return final_score, total_nfe_used, elapsed


def profile_timesteps(
    pipe: PixArtSigmaPipeline,
    config: FlowSearchConfig,
    scorer: Callable,
    prompt: str,
    n_seeds: int = 3,
    K_profile: int = 10,
):
    """
    Marginal value profiling: for each step, measure how much search (K>1)
    improves the FINAL image score compared to ODE-only baseline.

    Method: For each target step s:
      1. Run all steps with ODE (K=1, deterministic)
      2. Run all steps with ODE EXCEPT step s uses K=K_profile search
      3. marginal_value[s] = final_score_with_search - final_score_ode_only
    """
    device = pipe.device
    search = FlowNoiseSearch(pipe, config, scorer, device)

    # Encode prompt once
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
        pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            do_classifier_free_guidance=config.guidance_scale > 1,
            num_images_per_prompt=1,
            device=device,
        )
    )

    if config.guidance_scale > 1:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])

    pipe.scheduler.set_timesteps(config.num_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    latent_channels = pipe.transformer.config.in_channels
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

    def run_with_search_at(seed, target_step, K_at_target):
        """Run full denoising: ODE everywhere except target_step gets K_at_target."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        latents = torch.randn(
            (1, latent_channels, 64, 64),
            device=device, dtype=pipe.transformer.dtype,
        )
        for step_idx, t in enumerate(timesteps):
            t_value = t.item() / 1000.0
            if step_idx == target_step:
                latents, _, _ = search.search_step(
                    latents=latents, prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    added_cond_kwargs=added_cond_kwargs,
                    timestep=t, t_value=t_value,
                    K=K_at_target, guidance_scale=config.guidance_scale,
                    deterministic=(K_at_target <= 1),
                )
            else:
                latents, _, _ = search.search_step(
                    latents=latents, prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    added_cond_kwargs=added_cond_kwargs,
                    timestep=t, t_value=t_value,
                    K=1, guidance_scale=config.guidance_scale,
                    deterministic=True,
                )
        final_image = search.decode_latents(latents)
        return scorer(final_image).item()

    # First: get ODE-only baselines
    print("Computing ODE-only baselines...")
    ode_scores = {}
    for seed in range(n_seeds):
        ode_scores[seed] = run_with_search_at(seed, -1, 1)  # target_step=-1 means all ODE
        print(f"  seed={seed}: ODE baseline = {ode_scores[seed]:.4f}")

    # Then: test each step
    step_values = []
    for target_step in range(config.num_steps):
        t_value = timesteps[target_step].item() / 1000.0
        deltas = []
        for seed in range(n_seeds):
            search_score = run_with_search_at(seed, target_step, K_profile)
            delta = search_score - ode_scores[seed]
            deltas.append(delta)
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        step_values.append((target_step, t_value, mean_delta, std_delta))
        sign = "+" if mean_delta > 0 else ""
        print(f"  Step {target_step:2d} (t={t_value:.3f}): {sign}{mean_delta:.5f} ± {std_delta:.5f}")

    # Print summary
    print("\n" + "="*70)
    print("Marginal Value Profile (search at step vs all-ODE)")
    print("="*70)
    print(f"{'Step':>4} | {'t_value':>7} | {'Marginal Value':>14} | {'Std':>9}")
    print("-"*50)
    for step, t_val, mv, std in step_values:
        sign = "+" if mv > 0 else ""
        print(f"{step:4d} | {t_val:7.4f} | {sign}{mv:13.5f} | {std:9.5f}")

    return step_values


def run_comparison(
    model_id: str,
    scorer_name: str,
    nfe_budgets: List[int],
    n_seeds: int = 3,
    prompt: str = "a bright sunny landscape with mountains",
    noise_scale: float = 1.0,
):
    """Run comparison between naive and our method on real flow model."""

    log_print(f"\n{'='*60}")
    log_print(f"Flow Model Real Experiment: {scorer_name}")
    log_print(f"Model: {model_id}")
    log_print(f"Noise scale: {noise_scale}")
    log_print(f"Log file: {LOG_FILE}")
    log_print(f"{'='*60}\n")

    # Load model
    log_print("Loading PixArt-Sigma model...")
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    # Note: Not using cpu_offload to avoid device mismatch issues
    log_print("Model loaded.\n")

    # Initialize scorer
    if scorer_name == "compressibility":
        scorer = CompressibilityScorer()
        prompt = "a complex detailed scene with many objects"
    else:
        scorer = BrightnessScorer()

    results = {"naive": {}, "online": {}}

    for nfe in nfe_budgets:
        log_print(f"\n--- NFE Budget: {nfe} ---")

        config = FlowSearchConfig(
            num_steps=20,
            noise_scale=noise_scale,
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
                    scorer_name=scorer_name,
                )
                scores.append(score)
                times.append(elapsed)
                log_print(f"    {method} seed={seed}: score={score:.4f}")

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            mean_time = np.mean(times)
            results[method][nfe] = (mean_score, std_score, mean_time)

            log_print(f"  {method:8s}: {mean_score:.4f} ± {std_score:.4f} ({mean_time:.1f}s)")

    # Print summary table
    log_print(f"\n{'='*60}")
    log_print("Summary Table (for paper)")
    log_print(f"{'='*60}")
    log_print(f"{'NFE':>6s} | {'Naive':>20s} | {'Online (Ours)':>20s} | {'Δ':>8s}")
    log_print("-" * 60)

    for nfe in nfe_budgets:
        naive_mean, naive_std, _ = results["naive"][nfe]
        online_mean, online_std, _ = results["online"][nfe]
        delta = online_mean - naive_mean
        log_print(f"{nfe:6d} | {naive_mean:.4f} ± {naive_std:.4f} | "
              f"{online_mean:.4f} ± {online_std:.4f} | {delta:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Real Flow Model Experiment')
    parser.add_argument('--model_id', type=str, default='PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
                        help='Flow model to use')
    parser.add_argument('--method', choices=['naive', 'online'], default='naive',
                        help='Scheduling method')
    parser.add_argument('--scorer', choices=['brightness', 'compressibility'], default='brightness',
                        help='Scorer to use')
    parser.add_argument('--nfe', type=int, default=200, help='Total NFE budget')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--prompt', type=str, default='a bright sunny landscape with mountains',
                        help='Text prompt')
    parser.add_argument('--run_comparison', action='store_true',
                        help='Run full comparison experiment')
    parser.add_argument('--profile', action='store_true',
                        help='Run timestep profiling')
    parser.add_argument('--noise_scale', type=float, default=1.0, help='Diffusion norm for g(t)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.profile:
        # Run profiling
        print(f"Loading model {args.model_id} for profiling...")
        pipe = PixArtSigmaPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")

        scorer = BrightnessScorer()
        config = FlowSearchConfig(num_steps=20, noise_scale=1.0, total_nfe=200)
        profile_timesteps(pipe, config, scorer, args.prompt, n_seeds=3, K_profile=10)
        return

    if args.run_comparison:
        run_comparison(
            model_id=args.model_id,
            scorer_name=args.scorer,
            nfe_budgets=[80, 100, 120, 150, 200],
            n_seeds=8,
            prompt=args.prompt,
            noise_scale=args.noise_scale,
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
