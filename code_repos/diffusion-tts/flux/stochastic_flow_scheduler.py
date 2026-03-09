"""
Stochastic Flow Scheduler: Convert deterministic flow ODE to SDE for noise trajectory search.

Based on:
- Dockhorn et al. "Stochastic Sampling from Deterministic Flow Models" (arXiv:2410.02217)
- Kim et al. "Inference-Time Scaling for Flow Models" (arXiv:2503.19385)

Key conversion:
    ODE: dx = v(x,t) dt
    SDE: dx = [v(x,t) + (g^2/2) * score(x,t)] dt + g(t) dW

where score is computed from velocity for linear interpolant:
    score(x,t) = nabla_x ln p_t(x) = (-(1-t)*v(x,t) - x) / t
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import numpy as np


class StochasticFlowScheduler:
    """
    Stochastic scheduler for flow models.
    Converts deterministic ODE sampling to SDE for exploration.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 20,
        noise_scale: float = 0.5,  # alpha parameter for g(t) = alpha * sqrt(t)
        noise_schedule: str = "nonsingular",  # "nonsingular", "constant", "zeroends"
        shift: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.noise_scale = noise_scale
        self.noise_schedule = noise_schedule
        self.shift = shift

        # Initialize timesteps
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu"):
        """Set the timestep schedule."""
        self.num_inference_steps = num_inference_steps

        # Linear schedule from 1.0 to 0.0 (flow convention: t=1 is noise, t=0 is data)
        timesteps = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float32)

        # Apply shift if specified
        if self.shift != 1.0:
            timesteps = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)

        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.sigmas = self.timesteps.clone()

    def get_noise_coefficient(self, t: float) -> float:
        """
        Compute diffusion coefficient g(t) based on the chosen schedule.

        Args:
            t: Current timestep (in [0, 1], where t=1 is noise, t=0 is data)

        Returns:
            g(t): Diffusion coefficient
        """
        alpha = self.noise_scale

        if self.noise_schedule == "nonsingular":
            # g(t) = alpha * sqrt(t)
            # NonSingular: small noise near t=0 (data), larger near t=1 (noise)
            return alpha * np.sqrt(max(t, 1e-6))
        elif self.noise_schedule == "constant":
            # g(t) = alpha
            return alpha
        elif self.noise_schedule == "zeroends":
            # g(t) = alpha * sqrt(t * (1-t))
            # Zero at both ends
            return alpha * np.sqrt(max(t * (1 - t), 1e-6))
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def compute_score_from_velocity(
        self,
        velocity: torch.Tensor,
        sample: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Compute score function nabla_x ln p_t(x) from velocity field.

        For linear interpolant (alpha_t = 1-t, sigma_t = t) with Gaussian prior:
            score(x, t) = (-(1-t) * v(x,t) - x) / t

        Args:
            velocity: Model output v(x, t)
            sample: Current sample x_t
            t: Current timestep

        Returns:
            Score: nabla_x ln p_t(x)
        """
        # Avoid division by zero near t=0
        t_safe = max(t, 1e-4)

        # Linear interpolant: alpha_t = 1-t, sigma_t = t
        # score = (-(1-t) * v - x) / t
        alpha_t = 1.0 - t_safe
        score = (-(alpha_t) * velocity - sample) / t_safe

        return score

    def stochastic_step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_noise: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform one SDE step instead of ODE step.

        ODE: x_{t-dt} = x_t + dt * v(x_t, t)
        SDE: x_{t-dt} = x_t + dt * [v(x_t, t) + (g^2/2) * score(x_t, t)] + sqrt(dt) * g(t) * noise

        Args:
            model_output: Velocity v(x_t, t) from the flow model
            timestep: Current timestep t
            sample: Current sample x_t
            generator: Random number generator for reproducibility
            return_noise: Whether to return the noise used

        Returns:
            prev_sample: x_{t-dt} after one SDE step
            (optional) noise: The noise tensor used
        """
        # Upcast for precision
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        # Get current and next timestep
        step_idx = (self.timesteps == timestep).nonzero()
        if len(step_idx) == 0:
            step_idx = 0
        else:
            step_idx = step_idx[0].item()

        t = self.timesteps[step_idx].item()
        t_next = self.timesteps[min(step_idx + 1, len(self.timesteps) - 1)].item()
        dt = t_next - t  # Negative since we go from t=1 to t=0

        # Compute diffusion coefficient
        g_t = self.get_noise_coefficient(t)

        # Compute score from velocity
        score = self.compute_score_from_velocity(model_output, sample, t)

        # Corrected drift: f(x,t) = v(x,t) + (g^2 / 2) * score(x,t)
        drift_correction = (g_t ** 2 / 2) * score
        corrected_drift = model_output + drift_correction

        # ODE part: x + dt * f(x,t)
        prev_sample = sample + dt * corrected_drift

        # SDE part: add noise term sqrt(|dt|) * g(t) * noise
        if g_t > 0 and abs(dt) > 1e-8:
            noise = torch.randn(
                sample.shape,
                dtype=sample.dtype,
                device=sample.device,
                generator=generator,
            )
            noise_term = np.sqrt(abs(dt)) * g_t * noise
            prev_sample = prev_sample + noise_term

            if return_noise:
                return prev_sample, noise
        else:
            if return_noise:
                return prev_sample, torch.zeros_like(sample)

        return prev_sample

    def deterministic_step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform standard deterministic ODE step (for baseline comparison).

        ODE: x_{t-dt} = x_t + dt * v(x_t, t)
        """
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        step_idx = (self.timesteps == timestep).nonzero()
        if len(step_idx) == 0:
            step_idx = 0
        else:
            step_idx = step_idx[0].item()

        t = self.timesteps[step_idx].item()
        t_next = self.timesteps[min(step_idx + 1, len(self.timesteps) - 1)].item()
        dt = t_next - t

        prev_sample = sample + dt * model_output

        return prev_sample


def test_scheduler():
    """Test the stochastic flow scheduler."""
    scheduler = StochasticFlowScheduler(
        num_inference_steps=10,
        noise_scale=0.3,
        noise_schedule="nonsingular",
    )

    # Simulate a denoising process
    x = torch.randn(1, 4, 64, 64)  # Latent
    print(f"Initial x norm: {x.norm().item():.4f}")

    for i, t in enumerate(scheduler.timesteps[:-1]):
        # Fake velocity (in practice, this comes from the model)
        v = torch.randn_like(x) * 0.1

        # Stochastic step
        x_stoch = scheduler.stochastic_step(v, t, x)

        # Deterministic step for comparison
        x_det = scheduler.deterministic_step(v, t, x)

        print(f"Step {i}: t={t.item():.3f}, "
              f"stoch_norm={x_stoch.norm().item():.4f}, "
              f"det_norm={x_det.norm().item():.4f}, "
              f"diff={torch.norm(x_stoch - x_det).item():.4f}")

        x = x_stoch


if __name__ == "__main__":
    test_scheduler()
