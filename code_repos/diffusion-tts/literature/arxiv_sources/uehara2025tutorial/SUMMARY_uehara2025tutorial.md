# Summary: Inference-Time Alignment in Diffusion Models with Reward-Guided Generation

**arXiv:** 2501.09685
**Authors:** Masatoshi Uehara et al. (Genentech, Princeton, MIT, UC Berkeley)
**Type:** Tutorial/Review

## Core Framework: Unified View of Inference-Time Techniques

The tutorial establishes that **all inference-time alignment methods approximate the same target**:

```
p^(alpha)(x) = p^pre(x) * exp(r(x)/alpha) / Z
```

This is achieved via **soft optimal policies** at each denoising step:

```
p^*_{t-1}(.|x_t) = p^pre_{t-1}(.|x_t) * exp(v_{t-1}(.)/alpha) / Z_t
```

where **soft value functions** serve as look-ahead functions predicting future rewards:

```
v_{t-1}(x) = alpha * log E[exp(r(x_0)/alpha) | x_{t-1} = x]
```

## Methods Classification

### 1. Derivative-Free Methods (Most relevant for our work)

#### SMC-Based Guidance (Algorithm 1)
- Maintains N particles with importance weights
- Weight update: `w^[i] = p^pre * exp(v/alpha) / (q * exp(v_prev/alpha)) * w^[i]`
- Resampling when effective sample size drops
- **Pro**: Global interaction eliminates inferior particles
- **Con**: Mode collapse risk for small alpha

#### Value-Based Importance Sampling / SVDD (Algorithm 2)
- For each particle, generate M proposals, select best via importance sampling
- **Key difference from SMC**: No global interaction between batch samples
- Weight: `w^[i,j] = exp(v(x^[i,j])/alpha) * p^pre/q`
- **Pro**: Better for reward maximization (small alpha)
- **Con**: No cross-particle filtering

#### Beam Search with Value Functions (Algorithm 3)
- Special case of SVDD with alpha -> 0
- Simply select argmax_j v(x^[i,j])
- Greedy but value functions provide lookahead

### 2. Derivative-Based Methods (Classifier Guidance)
- Add gradient term to drift: `dx = [f + g^2 * nabla_x log p(y|x)]dt`
- Requires differentiable reward models
- Less applicable for black-box rewards (physics simulations, etc.)

## Value Function Approximation Methods

### 1. Posterior Mean Approximation (Simple)
```
v(x_t) approx r(x_0_hat(x_t))
```
where x_0_hat is Tweedie's estimate. Used in DPS, Universal Guidance.

### 2. Monte Carlo Regression
Train value network by regression on rolled-out trajectories.

### 3. Soft Q-Learning (Fitted Q-Iteration)
Use soft Bellman equations recursively:
```
exp(v_t/alpha) = E[exp(v_{t-1}/alpha) | x_t]
```

## Connection to Our Global Scheduling Work

### Direct Parallels
1. **Value functions = our verifier**: Both predict future quality from intermediate states
2. **Particle selection = our K-schedule**: Both allocate computation based on state quality
3. **SMC resampling = our branching**: Both prune low-quality trajectories

### Key Insights
1. **Scaling inference-time compute works**: Figure 1.12 shows reward improves with computational budget (beam width)
2. **Soft optimal policy is the target**: Validates our approach of using verifier-weighted sampling
3. **Derivative-free methods are practical**: Important for black-box rewards like physics simulations

### Nested-SMC (Algorithm 4)
Combines local sampling (SVDD-style) with global resampling (SMC-style):
1. Local sampling: Generate M proposals per particle, importance sample
2. Global resampling: Resample across all N particles based on normalizing constants

This is closest to our hierarchical scheduling approach.

## Verifier/Reward Design

Three scenarios:
1. **Conditioning**: Reward = log p(y|x), classifier
2. **Inverse Problems**: Reward = -||y - A(x)||^2, known observation model
3. **Alignment**: Reward = general regressor (binding affinity, stability, etc.)

## SDE vs ODE Perspective

The tutorial frames diffusion models as:
1. **Forward SDE**: dx = f(x,t)dt + g(t)dW (noising)
2. **Reverse SDE**: dx = [f - g^2 nabla log p_t]dt + gdW (denoising with stochasticity)
3. **Probability Flow ODE**: dx = [f - (g^2/2) nabla log p_t]dt (deterministic)

All three have the same marginals p_t(x) - connects to Singh & Fischer (2024).

## Practical Recommendations

1. **For alignment (reward maximization)**: Use SVDD/beam search (avoid SMC mode collapse)
2. **For conditioning**: SMC may be preferred (global filtering helps)
3. **Value function quality is critical**: Better approximation = better results
4. **Computational scaling**: Larger beam width / more particles consistently helps

## Code Reference
https://github.com/masa-ue/AlignInversePro
