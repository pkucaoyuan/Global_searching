# Summary: Stochastic Sampling from Deterministic Flow Models

**arXiv:** 2410.02217
**Authors:** Saurabh Singh, Ian Fischer (Google DeepMind)
**Venue:** ICLR 2025

## Core Contribution

This paper provides a **general method to convert deterministic flow ODEs into families of SDEs** that preserve the same marginal distributions. This enables stochastic sampling from pretrained deterministic flow models (like Rectified Flow) without retraining.

## Key Theorems

### Theorem 1 (Main Result)
Given an SDE with drift f and diffusion G, an infinite family of SDEs with the same marginal distributions can be constructed:

```
dx = f_bar(x,t)dt + G_bar(x,t)dW_t
```

where:
- `f_bar = f - (1/2)[nabla . ((1-gamma_t)GG^T - G_tilde G_tilde^T) + ((1-gamma_t)GG^T - G_tilde G_tilde^T) . nabla ln p_t]`
- `G_bar = [gamma_t GG^T + G_tilde G_tilde^T]^(1/2)`

**Key insight:** gamma_t and G_tilde are arbitrary functions that can be chosen at sampling time.

### Corollary 2 (For Deterministic Flows)
For ODE dx = v(x,t)dt, the stochastic version is:

```
dx = [v(x,t) + (g_tilde^2(t)/2) nabla_x ln p_t(x)]dt + g_tilde(t)dW_t
```

## Flow/ODE vs SDE Relationship

1. **Deterministic flows are special cases** where diffusion coefficient is zero
2. **Score function derivable from velocity field** for Gaussian flows:
   ```
   nabla_x ln p_t(x) = (-(1-t)v(x,t) + mu_1 - x) / (t * sigma_1^2)
   ```
3. **Any trained deterministic flow can be converted** to a family of SDEs without retraining

## Practical SDE Families (Table 1)

| Name | g_tilde(t) | Properties |
|------|------------|------------|
| Deterministic | 0 | Base flow model |
| Constant | alpha | Constant g, singular f |
| Singular | alpha*sqrt(t/(1-t)) | Singular g,f |
| **NonSingular** | alpha*sqrt(t) | Non-singular g,f |
| ZeroEnds | alpha*sqrt(t(1-t)) | g(0)=g(1)=0 |

**NonSingular and ZeroEnds perform best** - they have small diffusion near t=0 where noise is most harmful.

## Connection to Global Scheduling

### Direct Relevance
1. **Sampling flexibility without retraining**: Our global K-schedule can be viewed as analogous to choosing different stochastic samplers at test time
2. **Time-dependent control**: alpha parameter controls stochasticity magnitude - similar to how K controls computational allocation
3. **Bias-variance tradeoff**: Stochastic samplers trade deterministic bias for variance - analogous to our exploration/exploitation tradeoff

### Key Insights for Our Work
1. **Stochasticity helps at coarse discretization**: With fewer steps, stochastic samplers significantly outperform deterministic ones
2. **Non-singular schedules are crucial**: Schedules that have small noise near t=0 perform best - this aligns with our observation that low-noise regions need more careful handling
3. **No retraining needed**: Test-time flexibility is valuable - supports our approach of scheduling K without model changes

## Verifier/Reward Usage

- No explicit reward/verifier - paper focuses on matching marginal distributions
- However, the stochasticity parameter alpha provides controllable diversity
- Compatible with classifier-free guidance (CFG): v_cfg = (1+lambda)v(x,t,c) - lambda*v(x,t,null)

## Experimental Results

- **ImageNet generation**: NonSingular achieves FID 2.95 (vs 3.07 deterministic) at 64x64
- **Toy Gaussian**: Deterministic samplers systematically underestimate variance; stochastic samplers correct this
- **Discretization robustness**: Stochastic samplers degrade gracefully with fewer steps

## Mathematical Framework

The paper is grounded in:
1. Fokker-Planck-Kolmogorov equations
2. Time reversal of SDEs (Anderson 1982)
3. Score matching and denoising

The core insight is that the FPK equation constraints marginals but not the specific dynamics - infinitely many SDEs can have the same marginal evolution.
