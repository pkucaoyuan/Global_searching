# SUMMARY: Score-Based Generative Modeling through Stochastic Differential Equations

**Paper**: Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
**arXiv**: 2011.13456
**Key**: song2021scorebased

---

## Setting & Model

### Focus
Unifying score-based generative models (SMLD) and denoising diffusion probabilistic models (DDPM) through the lens of stochastic differential equations. The paper shows both are discretizations of continuous-time SDEs and proposes improved sampling via predictor-corrector methods and probability flow ODEs.

### Key Assumptions
1. **Continuous diffusion**: Data corruption is modeled as a continuous-time SDE rather than discrete steps.
2. **Known score function**: The time-dependent score $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ can be estimated via neural networks.
3. **Reversible diffusion**: The reverse-time SDE can be derived from the forward SDE given the score function (Anderson's theorem).

### Notation
| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}(t)$ | Sample at continuous time $t \in [0, T]$ |
| $p_t(\mathbf{x})$ | Marginal distribution at time $t$ |
| $p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))$ | Transition kernel |
| $\mathbf{f}(\mathbf{x}, t)$ | Drift coefficient of SDE |
| $g(t)$ | Diffusion coefficient |
| $\mathbf{s}_\theta(\mathbf{x}, t)$ | Time-dependent score network |
| $\mathbf{w}$ | Standard Wiener process |
| $\bar{\mathbf{w}}$ | Reverse-time Wiener process |

---

## Main Results

### Theorem 1: Reverse-Time SDE (Anderson 1982)
The reverse of the forward SDE
$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}$$
is given by the reverse-time SDE:
$$d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}}$$

This is the foundational result enabling score-based sampling.

### Theorem 2: VE and VP SDEs
**Variance Exploding (VE) SDE** (generalizes SMLD/NCSN):
$$d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}} d\mathbf{w}$$

**Variance Preserving (VP) SDE** (generalizes DDPM):
$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x} \, dt + \sqrt{\beta(t)} d\mathbf{w}$$

SMLD and DDPM training objectives are discretizations of the continuous score matching objective.

### Theorem 3: Probability Flow ODE
There exists a deterministic ODE with the same marginal distributions as the SDE:
$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})\right] dt$$

This enables:
- Exact likelihood computation via instantaneous change of variables
- Deterministic sampling with adaptive ODE solvers
- Uniquely identifiable latent encodings

### Theorem 4: Continuous Score Matching Objective
$$\theta^* = \arg\min_\theta \mathbb{E}_t \left\{ \lambda(t) \mathbb{E}_{\mathbf{x}(0)} \mathbb{E}_{\mathbf{x}(t)|\mathbf{x}(0)} \left[ \|\mathbf{s}_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))\|^2 \right] \right\}$$

### Key Empirical Results
- CIFAR-10: **FID 2.20**, IS 9.89 (unconditional SOTA)
- NLL 2.99 bits/dim (SOTA on uniformly dequantized CIFAR-10)
- First high-fidelity 1024x1024 generation from score-based models

---

## Proof Techniques

1. **SDE derivation from discrete Markov chains**: Show that SMLD/DDPM perturbation kernels converge to VE/VP SDEs as $N \to \infty$ via Taylor expansion.

2. **Anderson's reversal theorem**: Apply time-reversal formula to recover the reverse-time SDE.

3. **Fokker-Planck correspondence**: The probability flow ODE is derived by finding an ODE whose density evolution matches the SDE's Fokker-Planck equation.

4. **Predictor-Corrector analysis**: Error from discretization can be corrected using score-based MCMC (e.g., Langevin dynamics) at each step.

---

## Connection to Our Paper

### What We Borrow
1. **Continuous-time perspective**: We conceptualize the noise trajectory as a continuous path, even if discretized for computation.
2. **Predictor-Corrector framework**: Our local search operators are analogous to "correctors" that refine the sample at each timestep.
3. **Flexibility in discretization**: The paper shows sampling discretization is independent of training, supporting our variable compute allocation.
4. **Error accumulation insight**: "The estimation errors of the score-based model at multiple time steps can compound" - this motivates our early stopping.

### How We Differ
1. **Fixed corrector steps vs. adaptive search**: PC uses fixed corrector iterations; we adaptively allocate search budget based on profiled difficulty.
2. **No global scheduling**: PC doesn't consider cross-timestep compute allocation; our scheduler optimizes global budget distribution.
3. **Online control**: We use online statistics to decide when to stop; PC uses predetermined iteration counts.
4. **Compute-quality tradeoff**: PC's tradeoff is implicit in step count; we explicitly model and optimize this tradeoff.
5. **Inference-time scaling**: PC was not designed for scaling compute at inference; our framework explicitly supports this.

### Relevant Insight for Our Work
> "The error of sampling with the reverse-time SDE comes from two sources: discretization of the SDE, and estimation of the true scores... The estimation error can be reduced by incorporating the corrector step."

This supports our hypothesis that additional compute (via search) at certain timesteps can reduce overall error.

---

## Key Quotes

> "Creating noise from data is easy; creating data from noise is generative modeling."

> "We propose a predictor-corrector framework to correct errors in the evolution of the discretized reverse-time SDE."

> "The estimation errors of the score-based model at multiple time steps can compound. This estimation error can be reduced by incorporating the corrector step."

> "It is typically better than doubling the number of predictor steps without adding a corrector."

> "Our encoding is uniquely identifiable, meaning that with sufficient training data, model capacity, and optimization accuracy, the encoding for an input is uniquely determined by the data distribution."

---

## BibTeX

```bibtex
@inproceedings{song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Song, Yang and Sohl-Dickstein, Jascha and Kingma, Diederik P and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
