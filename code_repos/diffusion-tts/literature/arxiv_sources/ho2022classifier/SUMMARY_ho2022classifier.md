# Summary: Classifier-Free Diffusion Guidance

**Paper**: Ho & Salimans (2022) - "Classifier-Free Diffusion Guidance"
**arXiv**: 2207.12598
**Venue**: NeurIPS 2021 Workshop (short version); full paper 2022

---

## Setting & Model

### Focus
Classifier-free guidance achieves the sample quality benefits of classifier guidance without requiring a separate trained classifier. Instead, it jointly trains conditional and unconditional diffusion models and combines their score estimates during sampling.

### Key Assumptions
1. **Joint Training Feasibility**: A single network can learn both conditional $p(x|c)$ and unconditional $p(x)$ distributions by randomly dropping conditioning during training
2. **Implicit Classifier**: The difference between conditional and unconditional scores approximates a classifier gradient
3. **Linear Combination Suffices**: A simple weighted combination of conditional and unconditional scores produces the guidance effect

### Notation
- $\mathbf{z}_\lambda$: Noised sample at log-SNR level $\lambda$
- $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c})$: Conditional score estimate
- $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda)$: Unconditional score estimate (with $\mathbf{c} = \varnothing$)
- $w$: Guidance strength parameter
- $p_{\text{uncond}}$: Probability of dropping conditioning during training

---

## Main Results

### Key Finding 1: Classifier-Free Guidance Formula
The guided score estimate is a linear combination:
$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda, \mathbf{c}) = (1+w)\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c}) - w\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda)$$

This can be rewritten as:
$$\tilde{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c}) + w \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda))$$

The term $(\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda))$ approximates the gradient of an implicit classifier.

### Key Finding 2: Training Algorithm
```
Repeat:
  Sample (x, c) from dataset
  c <- null with probability p_uncond  # Randomly drop conditioning
  Sample lambda from p(lambda)
  Sample epsilon ~ N(0, I)
  z_lambda = alpha_lambda * x + sigma_lambda * epsilon
  Take gradient step on ||epsilon_theta(z_lambda, c) - epsilon||^2
Until converged
```

### Key Finding 3: FID/IS Trade-offs Match Classifier Guidance
| ImageNet 128x128 | w=0.0 | w=0.3 | w=1.0 | w=4.0 |
|------------------|-------|-------|-------|-------|
| FID | 7.27 | **2.43** | 7.86 | 21.53 |
| IS | 82.45 | 158.47 | 297.98 | **421.03** |

Best FID achieved at moderate guidance ($w \approx 0.3$); best IS at strong guidance ($w \geq 4$).

### Key Finding 4: State-of-the-Art Results
- ImageNet 128x128: FID 2.43 (outperforms ADM-G's 2.97)
- This is achieved with **no separate classifier training**
- Matches or exceeds BigGAN-deep on both FID and IS simultaneously at $w=4.0$

### Key Finding 5: Unconditional Training Probability
- $p_{\text{uncond}} \in \{0.1, 0.2\}$ work equally well
- $p_{\text{uncond}} = 0.5$ consistently underperforms
- Only a small fraction of model capacity needs to be dedicated to unconditional generation

---

## Proof Techniques / Methods

### Implicit Classifier Interpretation
If exact scores $\boldsymbol{\epsilon}^*(\mathbf{z}_\lambda, \mathbf{c})$ and $\boldsymbol{\epsilon}^*(\mathbf{z}_\lambda)$ were available, the implicit classifier gradient would be:
$$\nabla_{\mathbf{z}_\lambda} \log p^i(\mathbf{c}|\mathbf{z}_\lambda) = -\frac{1}{\sigma_\lambda}[\boldsymbol{\epsilon}^*(\mathbf{z}_\lambda, \mathbf{c}) - \boldsymbol{\epsilon}^*(\mathbf{z}_\lambda)]$$

Classifier guidance with this implicit classifier gives the same formula as classifier-free guidance.

### Key Insight: Negative Score Direction
Classifier-free guidance decreases the unconditional likelihood while increasing the conditional likelihood. The **negative** unconditional score term $-w\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda)$ pushes samples away from the unconditional mode.

### Non-Conservative Vector Fields
Unlike true classifier gradients, the difference $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda)$ is not necessarily a gradient of any scalar function, since neural networks are unconstrained. This means classifier-free guidance is fundamentally different from classifier guidance, despite similar effects.

---

## Connection to Our Paper

### What We Borrow
1. **Guidance Framework**: Classifier-free guidance is our primary conditioning mechanism for text-to-image experiments
2. **Quality-Diversity Trade-off**: The $w$ parameter provides a direct lever for trading off sample quality vs. diversity
3. **Efficient Architecture**: Single network for both conditional and unconditional removes classifier overhead

### How We Differ
1. **Guidance is Local, Scheduling is Global**: Guidance strength $w$ is typically fixed across all timesteps; our scheduler could adapt guidance per timestep
2. **Single Parameter vs. Per-Timestep Control**: They use one $w$ for all steps; we can allocate different compute to different timesteps
3. **Training Consideration vs. Inference Focus**: They modify training; we focus on inference-time compute allocation
4. **Complementary Approaches**: Classifier-free guidance can be combined with our global scheduling

### Relevance to Our Framework
- **Guidance as Local Operator**: Classifier-free guidance can be viewed as a local search operator that improves each step
- **Double Forward Pass**: Each guidance step requires two network evaluations (conditional + unconditional), doubling per-step compute
- **Variable Guidance Strength**: Our framework could vary $w$ across timesteps as part of global scheduling
- **Compute Trade-offs**: At $T=256$ steps with guidance, effective compute is $2 \times 256 = 512$ forward passes; our scheduler can optimize this allocation

### Insights for Global Scheduling
1. **Non-Uniform Guidance**: Different timesteps may benefit from different guidance strengths
2. **Early vs. Late Steps**: High-noise steps establish structure; low-noise steps refine details - guidance may matter differently
3. **Compute Budget**: With fixed total forward passes, trading guidance strength across timesteps is a form of compute allocation
4. **Implicit Classifier Quality**: The implicit classifier $p^i(c|z_\lambda)$ quality varies with noise level, suggesting timestep-dependent optimal $w$

---

## Key Quotes

> "Classifier guidance combines the score estimate of a diffusion model with the gradient of an image classifier and thereby requires training an image classifier separate from the diffusion model."

> "In what we call classifier-free guidance, we jointly train a conditional and an unconditional diffusion model, and we combine the resulting conditional and unconditional score estimates."

> "We show that guidance can be indeed performed by a pure generative model without such a classifier."

> "Classifier-free guidance accomplishes this by decreasing the unconditional likelihood with a negative score term, which to our knowledge has not yet been explored."

> "We have demonstrated that pure generative diffusion models are capable of maximizing classifier-based sample quality metrics while entirely avoiding classifier gradients."

---

## BibTeX

```bibtex
@article{ho2022classifier,
  title={Classifier-Free Diffusion Guidance},
  author={Ho, Jonathan and Salimans, Tim},
  journal={arXiv preprint arXiv:2207.12598},
  year={2022}
}
```

---

## Additional Notes

### Practical Simplicity
The method is remarkably simple:
- **Training**: One line change (randomly drop conditioning)
- **Sampling**: One line change (linear combination of scores)

### Sampling Cost
Each sampling step requires **two** forward passes:
1. $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda, \mathbf{c})$ - conditional
2. $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda)$ - unconditional

This doubles the compute compared to unguided sampling, making efficient scheduling even more important.

### Connection to Temperature Sampling
Unlike low-temperature sampling (which produces blurry images in diffusion models), classifier-free guidance successfully trades off diversity for fidelity without quality degradation.

### Relevance to EDM
EDM-style noise scheduling can be combined with classifier-free guidance. Our framework evaluates both combinations, using classifier-free guidance as the conditioning mechanism and EDM/Stable Diffusion as the diffusion formulation.

---

## Algorithms

### Algorithm 1: Training with Classifier-Free Guidance
```
Input: p_uncond (probability of unconditional training)
Repeat:
  (x, c) ~ p(x, c)  # Sample data with conditioning
  c <- null with probability p_uncond  # Randomly discard conditioning
  lambda ~ p(lambda)  # Sample log SNR
  epsilon ~ N(0, I)
  z_lambda = alpha_lambda * x + sigma_lambda * epsilon
  Take gradient step on ||epsilon_theta(z_lambda, c) - epsilon||^2
Until converged
```

### Algorithm 2: Sampling with Classifier-Free Guidance
```
Input: w (guidance strength), c (conditioning), lambda_1...lambda_T (log SNR sequence)
z_1 ~ N(0, I)
For t = 1 to T:
  # Classifier-free guided score
  epsilon_tilde = (1 + w) * epsilon_theta(z_t, c) - w * epsilon_theta(z_t)

  # Sampling step (ancestral or DDIM)
  x_tilde = (z_t - sigma * epsilon_tilde) / alpha
  z_{t+1} ~ N(mu(z_t, x_tilde), sigma^2)  # if t < T, else z_{t+1} = x_tilde
Return z_{T+1}
```
