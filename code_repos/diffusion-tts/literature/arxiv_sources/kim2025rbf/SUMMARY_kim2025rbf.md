# Summary: Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing

**Paper:** arXiv:2503.19385 (NeurIPS 2025 submission)
**Authors:** Jaihoon Kim*, Taehoon Yoon*, Jisung Hwang*, Minhyuk Sung (KAIST)
**Project Page:** flow-inference-time-scaling.github.io

---

## 1. Setting & Model

### Problem Formulation
The paper addresses **inference-time reward alignment** for flow models. Given:
- A pretrained flow model mapping source distribution p_1 (Gaussian) to data distribution p_0
- A reward function r(x) measuring text alignment or user preference

The objective is to find samples from:
```
p*_0 = argmax_q E_{x_0 ~ q}[r(x_0)] - beta * D_KL[q || p_0]
```

This yields the target distribution:
```
p*_0(x_0) = (1/Z) * p_0(x_0) * exp(r(x_0)/beta)
```

### Core Challenge
**Flow models are deterministic (ODE-based)**, which prevents direct application of particle sampling methods that have been successful for diffusion models. The stochasticity in diffusion models enables exploring high-reward regions through particle branching/termination.

### Three Key Technical Contributions

#### 1. Inference-Time SDE Conversion
Transform the deterministic flow ODE into a stochastic SDE:

**Flow ODE:** `dx_t = u_t(x_t) dt`

**Converted SDE:** `dx_t = f_t(x_t) dt + g_t dw`

where: `f_t(x_t) = u_t(x_t) - (g_t^2 / 2) * grad(log p_t(x_t))`

The score function can be computed from the velocity:
```
grad(log p_t(x_t)) = (1/sigma_t) * (alpha_t * u_t - dot(alpha_t) * x_t) / (dot(alpha_t)*sigma_t - alpha_t*dot(sigma_t))
```

This introduces stochasticity via a freely-chosen diffusion coefficient g_t (they use g_t = 3*t^2).

#### 2. Interpolant Conversion (Linear to VP)
Flow models use **linear interpolant**: alpha_t = 1-t, sigma_t = t

Diffusion models use **Variance Preserving (VP) interpolant**:
- alpha_t = exp(-0.5 * integral(beta_s ds))
- sigma_t = sqrt(1 - exp(-integral(beta_s ds)))

**Key insight:** VP-SDE provides greater sample diversity than Linear-SDE because:
1. VP has lower log-SNR at each timestep (noisier latents)
2. Timestep conversion + diffusion coefficient scaling work synergistically
3. Noisier latents produce more diverse samples (SDEdit principle)

The velocity is transformed via scale-time transformation:
```
bar{x}_s = c_s * x_{t_s}
t_s = rho^{-1}(bar{rho}(s))
c_s = bar{sigma}_s / sigma_{t_s}
```

#### 3. Stochastic Interpolant Framework
Unified formulation: `x_t = alpha_t * x_0 + sigma_t * x_1`

The proposal distribution for particle sampling becomes:
```
p_theta(x_{t-dt} | x_t) = N(x_t - f_t(x_t)*dt, g_t^2*dt*I)
```

---

## 2. Rollover Budget Forcing (RBF) - The Core Algorithm

### Motivation
Previous particle sampling methods (SVDD, CoDe, SMC) allocate NFEs **uniformly** across timesteps. However, the authors' analysis shows:
- The NFEs required to find a higher-reward particle varies significantly across timesteps
- Uniform allocation leads to wasted compute (under-utilization at some steps, insufficient at others)

### RBF Algorithm
```
Input: M denoising steps, timesteps {S^(i)}, NFE quotas {Q^(i)}
Output: Aligned sample x_0

Initialize x_1 ~ N(0, I), r* = r(x_{0|1})

For i in {1, ..., M}:
    s = S^(i), delta_s = S^(i) - S^(i+1), q = Q^(i)

    For j in {1, ..., q}:
        # Sample particle via stochastic denoising
        x_{s-ds}^(j) = stoch_denoise(x_s, s, ds)

        If r* < r(x_{0|s-ds}^(j)):  # Found better sample!
            Q^(i+1) += Q^(i) - j     # ROLLOVER remaining budget
            r* = r(x_{0|s-ds}^(j))
            x_{s-ds} = x_{s-ds}^(j)
            break                    # Early stopping!

        If j == q:  # Exhausted quota without improvement
            k* = argmax_k r(x_{0|s-ds}^(k))
            x_{s-ds} = x_{s-ds}^(k*)  # Take best from current set
```

### Key Mechanism
1. **Early Stopping:** If a better particle is found, immediately proceed to next timestep
2. **Budget Rollover:** Unused NFEs are transferred to subsequent timesteps
3. **Fallback:** If quota exhausted without improvement, select best particle (like SVDD)

---

## 3. Main Results

### Experimental Setup
- **Base Model:** FLUX (flow-based text-to-image)
- **Total Budget:** 500 NFEs, 10 denoising steps (50 NFEs/step)
- **Tasks:**
  - Compositional text-to-image (GenAI-Bench, 121 prompts)
  - Quantity-aware image generation (T2I-CompBench++, 100 prompts)
  - Aesthetic image generation (DDPO prompts, 45 prompts)

### Key Findings

| Method | Process | Performance |
|--------|---------|-------------|
| BoN | Linear-ODE | Baseline |
| SoP | Linear-ODE | Slightly better than BoN |
| SVDD | Linear-SDE | Better than Linear-ODE methods |
| SVDD | VP-SDE | Even better (expanded search space) |
| **RBF** | VP-SDE | **Best performance** |

**Quantitative Results (Compositional T2I):**
- VP-SDE + RBF achieves ~0.925 VQAScore vs. ~0.788 baseline FLUX
- 4-6x improvement in counting accuracy for quantity-aware generation
- RBF outperforms SVDD, SMC, CoDe across all metrics

**Scaling Behavior:**
- BoN plateaus after ~300 NFEs
- RBF continues to improve even at 1000 NFEs
- RBF at 500 NFEs beats BoN at any budget

---

## 4. Connection to Our Paper

### HIGH SIMILARITY: RBF vs. Our Online Scheduler with Early Stopping

| Aspect | Kim et al. (RBF) | Our Approach |
|--------|------------------|--------------|
| **Budget Allocation** | Non-uniform across timesteps | Non-uniform across timesteps |
| **Mechanism** | Early stopping + budget rollover | Early stopping + budget reallocation |
| **Decision Criterion** | r(x_{0|t}^new) > r* (best so far) | Improvement over threshold |
| **Scope** | Single-sample trajectory | Single-sample trajectory |
| **Profiling** | None (purely reactive/online) | Offline profiling + online adaptation |
| **Search Type** | Local (particles from current state) | Local operators (K iterations) |

### Key Similarities
1. **Non-uniform NFE allocation:** Both recognize that uniform compute allocation is suboptimal
2. **Early stopping:** Both terminate search early when sufficient improvement found
3. **Budget transfer:** Both move unused compute to subsequent steps
4. **Online decision making:** Both make real-time decisions during generation

### Key Differences

| Aspect | RBF | Our Approach |
|--------|-----|--------------|
| **Profiling** | None | Offline profiling to learn step difficulty |
| **Initial Budget** | Uniform Q per step | Profile-based allocation |
| **Threshold** | Global best r* | Step-specific thresholds |
| **Architecture** | Single-level | Two-level (local + global) |
| **Applicability** | Flow models only | EDM/general diffusion |
| **Search Space** | Stochastic particles | Multi-run + stochastic operators |

### What RBF Lacks (Our Advantages)
1. **No offline profiling:** RBF starts with uniform allocation, relying entirely on online adaptation
2. **No step-specific intelligence:** All steps treated identically initially
3. **Single-sample focus:** Batch-size N=2 limits exploration
4. **Flow-specific:** Requires ODE-to-SDE conversion for flow models

### What RBF Has (Their Advantages)
1. **Simpler:** No profiling phase required
2. **Interpolant conversion:** VP-SDE expansion of search space
3. **Theoretical grounding:** Clean optimal policy derivation
4. **Flow model focus:** State-of-the-art FLUX integration

---

## 5. Key Quotes

### On Motivation for RBF
> "Previous particle sampling methods for diffusion models employ a fixed number of particles across all denoising steps. However, our analysis shows that this uniform allocation may lead to inefficiency, where the NFEs required at each denoising step to obtain a sample x_{t-dt} with a higher reward than the current sample x_t significantly varies across different runs."

### On the Rollover Mechanism
> "Given a total NFEs budget, the NFEs quota Q is allocated uniformly across timesteps. Then at each timestep, if a particle x_{t-dt} yields a higher reward than the current sample x_t within the quota, we immediately proceed to the next timestep from the newly identified high-reward sample, rolling over the remaining NFEs to the next step."

### On Sample Diversity
> "A key factor in seeking high-reward samples using particle sampling is defining the proposal distribution to sufficiently cover the distribution of high-reward samples. Consider a scenario where high-reward samples reside in a low density region of the original data distribution."

### On VP Interpolant
> "To further expand the exploration space, we consider not only stochasticity but also the choice of the interpolant. While typical flow models use a linear interpolant, diffusion models commonly adopt a Variance-Preserving (VP) interpolant."

---

## 6. Critical Analysis

### Strengths
1. **Clean formulation:** Unified stochastic interpolant framework is elegant
2. **Principled motivation:** Optimal policy derivation from RL perspective
3. **Practical:** Works with off-the-shelf FLUX model
4. **Comprehensive:** SDE conversion + interpolant conversion + RBF all contribute

### Weaknesses
1. **RBF is reactive, not predictive:** No learning from past runs
2. **Uniform initial allocation:** Misses opportunity from offline profiling
3. **Limited batch size:** N=2 constrains exploration
4. **Flow-specific focus:** Extra complexity for diffusion models

### Implications for Our Work
1. **Validate our approach:** RBF independently discovers similar principles
2. **Differentiation:** Our offline profiling provides head start
3. **Potential synthesis:** Combine our profiling with their VP-SDE conversion
4. **Citation necessary:** Concurrent/related work, must cite

---

## 7. BibTeX Entry

```bibtex
@article{kim2025inference,
  title={Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing},
  author={Kim, Jaihoon and Yoon, Taehoon and Hwang, Jisung and Sung, Minhyuk},
  journal={arXiv preprint arXiv:2503.19385},
  year={2025},
  note={NeurIPS 2025 submission}
}
```

---

## 8. Summary Table

| Component | Technical Detail |
|-----------|------------------|
| **SDE Conversion** | g_t = 3*t^2, score from velocity |
| **Interpolant** | Linear -> VP via scale-time transform |
| **RBF Trigger** | r(new) > r* (global best) |
| **RBF Action** | Q^(i+1) += Q^(i) - j (rollover) |
| **Fallback** | argmax over current particles |
| **Base Model** | FLUX (10 steps, 50 NFE/step) |
| **Total Budget** | 500 NFEs |

---

## 9. Relation to Our Two-Level Framework

Our framework structure vs. RBF:

```
OUR APPROACH:
+------------------+
|  Global Scheduler |  <-- Offline profile + online adaptation
+------------------+
        |
        v
+------------------+
|  Local Operators  |  <-- K-iteration search per step
+------------------+

RBF:
+------------------+
|    RBF (Online)   |  <-- Reactive budget rollover only
+------------------+
        |
        v
+------------------+
| Particle Sampling |  <-- Stochastic denoising
+------------------+
```

**RBF is essentially an online-only scheduler without the offline profiling component.** This makes our approach complementary - we could potentially use RBF's rollover mechanism as part of our online adaptation layer while maintaining our profiling-based initial allocation.

---

*Summary prepared for literature review - Global Scheduling of Noise Trajectory Search in Diffusion Models*
