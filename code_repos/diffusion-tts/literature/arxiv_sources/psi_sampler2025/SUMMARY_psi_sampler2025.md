# Summary: Psi-Sampler - Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment

**arXiv:** 2506.01320
**Authors:** Taehoon Yoon, Yunhong Min, Kyeongmin Yeo, Minhyuk Sung (KAIST)
**Venue:** NeurIPS 2025
**Project:** https://psi-sampler.github.io/

## Core Innovation

The key insight is that **initial particle placement matters critically for SMC-based reward alignment**. Rather than sampling initial particles from the Gaussian prior, Psi-Sampler samples from the **reward-informed posterior**:

```
p_1^*(x_1) = (1/Z_1) * p_1(x_1) * exp(r(x_0|1) / alpha)
```

where x_0|1 is Tweedie's estimate of the clean sample from x_1.

## Motivation: Why Initial Particles Matter

1. **Vanishing diffusion coefficient**: As t -> 0, g(t)^2 -> 0, weakening the reward signal's influence on the SMC proposal
2. **Multi-modal rewards**: Initial position is critical for non-convex, multi-modal reward landscapes
3. **Early exploration is easier**: At t=1, the posterior is more diffuse and better connected across modes
4. **Distilled models**: Modern distilled flow models (FLUX, etc.) have straighter trajectories, making Tweedie estimates at t=1 more accurate

## Method: pCNL Algorithm

### Challenge
Standard MCMC (MALA, ULA) fails in extremely high dimensions (e.g., 65,536 for FLUX):
- MALA acceptance probability degrades as d -> infinity
- Step size must shrink as O(d^(-1/3)) for reasonable acceptance

### Solution: Preconditioned Crank-Nicolson Langevin (pCNL)

Semi-implicit Euler discretization:
```
x' = rho*x + sqrt(1-rho^2) * (z + (sqrt(eps)/2) * nabla r(x_0|1)/alpha)
```
where rho = (1-eps/4)/(1+eps/4) and z ~ N(0,I).

**Key property**: pCNL maintains non-zero acceptance probability even in infinite dimensions because:
1. Prior-preserving proposal (Gaussian reference measure is invariant)
2. Semi-implicit discretization avoids the d-dependent step size scaling

### Full Pipeline (Psi-Sampler)
1. **pCNL phase**: Run MCMC on the posterior at t=1 to generate K initial particles
2. **SMC phase**: Standard SMC-based denoising from t=1 to t=0

## Connection to Flow/ODE vs SDE

The paper works within the **stochastic optimal control (SOC) framework**:

1. **Optimal initial distribution**: p_1^*(x_1) (Eq. 6 in paper)
2. **Optimal transition kernel**:
   ```
   p^*_theta(x_{t-dt}|x_t) = [exp(r(x_0|t-dt)/alpha) / exp(r(x_0|t)/alpha)] * p_theta(x_{t-dt}|x_t)
   ```
3. **SMC proposal** (with guidance):
   ```
   q(x_{t-dt}|x_t) = N(x_t - f(x_t,t)dt + g^2(t)*nabla r(x_0|t)/alpha * dt, g(t)^2*dt*I)
   ```

Both flow-based and diffusion models are treated as **score-based generative models** - flow models can be extended to SDE formulation with same marginals.

## Verifier/Reward Usage

The reward function appears in:
1. **Initial posterior sampling**: exp(r(x_0|1)/alpha) weights the prior
2. **SMC weights**: w_t = exp(r(x_0|t)/alpha) terms
3. **Guided proposals**: nabla r(x_0|t) gradient steers particles

This is a **soft optimal policy** approach where:
- The reward acts as a "twisting potential"
- Value functions are approximated via Tweedie's formula: v(x_t) approx r(x_0|t)

## Connection to Our Global Scheduling Work

### Direct Parallels
1. **Front-loaded computation**: Psi-Sampler invests MCMC budget at t=1 rather than spreading it uniformly - analogous to our insight that some regions need more K than others
2. **Particle quality vs quantity**: Better initial particles > more particles with random init
3. **Two-phase approach**: MCMC (exploration) + SMC (refinement) mirrors our high-K (exploration) + low-K (refinement) regions

### Key Insights for Our Work
1. **Initial state quality propagates**: Poor initial particles cannot be fully corrected by later SMC steps
2. **Dimension-robust methods exist**: pCNL works in 65K+ dimensions - relevant if we scale to larger models
3. **Reward-informed initialization**: Instead of random seeds, use verifier-informed selection

### Numerical Insight
From toy experiments:
- Prior-only SMC (100 NFE) < MALA init + SMC (50+50 NFE) < pCNL init + SMC (50+50 NFE)
- The initialization method matters more than total compute when compute is fixed

## Experimental Tasks

1. **Layout-to-image**: Place objects in designated bounding boxes
2. **Quantity-aware**: Generate specific number of objects
3. **Aesthetic preference**: Optimize LAION aesthetic score

All use FLUX as base model (65,536-dimensional latent space).

## Mathematical Framework

Grounded in:
1. **Stochastic Optimal Control (SOC)**: Continuous-time formulation of reward alignment
2. **Feynman-Kac formula**: For deriving optimal initial distribution
3. **Tweedie's formula**: x_0|t = E[x_0 | x_t] for reward estimation from intermediate states

## Practical Takeaways

1. **Burn-in and thinning**: Standard MCMC practices apply for initial particle sampling
2. **Fixed step size**: Simplicity over adaptive (works well with pCNL)
3. **Budget allocation**: Splitting compute between MCMC init and SMC can be more effective than all-SMC
