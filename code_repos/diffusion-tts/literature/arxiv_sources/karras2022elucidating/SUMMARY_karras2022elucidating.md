# SUMMARY: Elucidating the Design Space of Diffusion-Based Generative Models (EDM)

**Paper**: Karras, Aittala, Aila, Laine. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.
**arXiv**: 2206.00364
**Key**: karras2022elucidating

---

## Setting & Model

### Focus
Systematically analyzing the design space of diffusion models by separating independent components: sampling, training, and preconditioning. The paper shows that sampler improvements alone can dramatically improve pre-trained models, and provides principled derivations for all design choices.

### Key Assumptions
1. **Modular design**: Sampling, training, and network preconditioning are largely independent and can be optimized separately.
2. **Denoiser formulation**: The core object is a denoiser $D(\mathbf{x}; \sigma)$ that minimizes $L_2$ denoising error.
3. **Score-denoiser equivalence**: $\nabla_\mathbf{x} \log p(\mathbf{x}; \sigma) = (D(\mathbf{x}; \sigma) - \mathbf{x}) / \sigma^2$

### Notation
| Symbol | Meaning |
|--------|---------|
| $\sigma(t)$ | Noise level schedule |
| $s(t)$ | Signal scaling schedule |
| $D_\theta(\mathbf{x}; \sigma)$ | Learned denoiser |
| $F_\theta$ | Raw neural network before preconditioning |
| $c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}, c_{\text{noise}}$ | Preconditioning functions |
| $\rho$ | Discretization schedule parameter |
| $\gamma_i$ | Stochastic "churn" factor |
| NFE | Number of function evaluations |

---

## Main Results

### Theorem 1: Unified ODE Formulation
The probability flow ODE can be written as:
$$d\mathbf{x} = -\dot{\sigma}(t) \sigma(t) \nabla_\mathbf{x} \log p(\mathbf{x}; \sigma(t)) \, dt$$

With scaling:
$$d\mathbf{x} = \left[\frac{\dot{s}(t)}{s(t)}\mathbf{x} - s(t)^2 \dot{\sigma}(t) \sigma(t) \nabla_\mathbf{x} \log p\left(\frac{\mathbf{x}}{s(t)}; \sigma(t)\right)\right] dt$$

**Key insight**: Setting $\sigma(t) = t$ and $s(t) = 1$ minimizes trajectory curvature.

### Theorem 2: Optimal Time Discretization
For step sizes that approximately equalize truncation error:
$$\sigma_{i<N} = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho})\right)^\rho$$

Empirically, $\rho \in [5, 10]$ works best (paper uses $\rho = 7$), suggesting errors near $\sigma_{\min}$ have large impact.

### Theorem 3: Stochastic Sampling with Langevin Churn
The general SDE for diffusion models:
$$d\mathbf{x}_{\pm} = -\dot{\sigma}(t)\sigma(t)\nabla_\mathbf{x}\log p(\mathbf{x};\sigma(t))dt \pm \beta(t)\sigma(t)^2\nabla_\mathbf{x}\log p(\mathbf{x};\sigma(t))dt + \sqrt{2\beta(t)}\sigma(t)d\mathbf{w}$$

The Langevin term corrects errors but introduces its own discretization error.

### Theorem 4: Optimal Preconditioning
For unit-variance inputs and targets with minimal error amplification:
- $c_{\text{in}}(\sigma) = 1/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$
- $c_{\text{skip}}(\sigma) = \sigma_{\text{data}}^2/(\sigma^2 + \sigma_{\text{data}}^2)$
- $c_{\text{out}}(\sigma) = \sigma \cdot \sigma_{\text{data}}/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$

### Key Empirical Results
- CIFAR-10 conditional: **FID 1.79** (SOTA)
- CIFAR-10 unconditional: **FID 1.97** (SOTA)
- ImageNet-64: **FID 1.36** (SOTA, trained from scratch)
- Pre-trained ImageNet-64 (ADM): FID improved from 2.07 to **1.55** via sampler alone
- Only **35 NFE** needed for high-quality samples

---

## Proof Techniques

1. **ODE reformulation**: Start from Song et al.'s SDE, substitute $f(t)$ and $g(t)$ in terms of $\sigma(t)$ and $s(t)$, yielding a formulation where the schedule functions are first-class citizens.

2. **Denoising-score equivalence**: Derive that the optimal $L_2$ denoiser gives the score function, establishing the connection rigorously for finite datasets.

3. **Truncation error analysis**: Analyze local ODE solver error as a function of step size and trajectory curvature to derive optimal discretization.

4. **Fokker-Planck derivation**: Derive the SDE family from the heat equation PDE that generates the correct marginal distributions.

5. **Preconditioning optimization**: Set up variance normalization constraints and solve for $c_{\text{skip}}$ that minimizes output scaling $c_{\text{out}}$.

---

## Connection to Our Paper

### What We Borrow
1. **Modular design philosophy**: Our framework similarly separates local search operators from global scheduling.
2. **Non-uniform timestep importance**: EDM shows $\rho = 7$ works better than uniform steps, supporting our hypothesis that different timesteps need different compute.
3. **Stochasticity analysis**: Their analysis of when stochasticity helps/hurts informs our search operator design.
4. **NFE as compute metric**: We use similar NFE-based compute budgeting.
5. **Pre-trained model improvement**: EDM shows samplers can improve pre-trained models; our approach extends this via search.

### How We Differ
1. **Fixed discretization vs. adaptive search**: EDM uses fixed (optimized) discretization; we search over multiple trajectories.
2. **Analytical vs. learned allocation**: EDM derives $\rho$ analytically; we learn allocation via offline profiling.
3. **No online adaptation**: EDM's schedule is fixed; our online controller adapts based on per-sample difficulty.
4. **Single trajectory vs. multi-trajectory**: EDM samples one trajectory; we search over multiple candidates.
5. **Early stopping**: EDM doesn't consider early termination; our scheduler can stop search when marginal returns diminish.

### Critical Insight for Our Work
EDM's finding that $\rho \in [5, 10]$ (concentrating steps near $\sigma_{\min}$) outperforms $\rho = 3$ (uniform truncation error) suggests:
> "Errors near $\sigma_{\min}$ have a large impact."

This directly motivates our non-uniform compute allocation, where we may want more search budget at low noise levels.

---

## Key Quotes

> "We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted."

> "The goal is to obtain better insights into how these components are linked together and what degrees of freedom are available in the design of the overall system."

> "Our hypothesis is that the choices related to the sampling process are largely independent of the other components... from the viewpoint of the sampler, $D_\theta$ is simply a black box."

> "Setting $\rho = 3$ nearly equalizes the truncation error at each step, but $\rho$ in range of 5 to 10 performs much better for sampling images. This suggests that errors near $\sigma_{\min}$ have a large impact."

> "The relevance of stochastic sampling appears to diminish as the model itself improves."

> "Through sampler improvements alone, we are able to bring the ImageNet-64 model that originally achieved FID 2.07 to 1.55."

---

## BibTeX

```bibtex
@inproceedings{karras2022elucidating,
  title={Elucidating the Design Space of Diffusion-Based Generative Models},
  author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={26565--26577},
  year={2022}
}
```
