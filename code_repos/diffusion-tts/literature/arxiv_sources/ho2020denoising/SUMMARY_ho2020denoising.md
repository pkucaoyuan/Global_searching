# SUMMARY: Denoising Diffusion Probabilistic Models (DDPM)

**Paper**: Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
**arXiv**: 2006.11239
**Key**: ho2020denoising

---

## Setting & Model

### Focus
Diffusion probabilistic models that generate high-quality images by learning to reverse a gradual noising process. The paper establishes that diffusion models can achieve sample quality competitive with GANs.

### Key Assumptions
1. **Fixed forward process**: The diffusion (forward) process is a fixed Markov chain that gradually adds Gaussian noise according to a variance schedule $\beta_1, \ldots, \beta_T$.
2. **Gaussian transitions**: Both forward and reverse processes use Gaussian conditionals, justified when $\beta_t$ are small.
3. **Discrete timesteps**: Uses $T=1000$ discrete noise levels with linear schedule $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.

### Notation
| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}_0$ | Clean data sample |
| $\mathbf{x}_t$ | Noisy sample at timestep $t$ |
| $\beta_t$ | Forward process variance at step $t$ |
| $\alpha_t = 1 - \beta_t$ | Complementary noise schedule |
| $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | Cumulative product |
| $\epsilon_\theta(\mathbf{x}_t, t)$ | Neural network predicting noise |
| $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ | Forward transition kernel |
| $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ | Learned reverse transition |

---

## Main Results

### Theorem 1: Closed-Form Marginal
The forward process admits sampling $\mathbf{x}_t$ at arbitrary timestep $t$ in closed form:
$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

This enables efficient training by directly sampling any noise level without sequential computation.

### Theorem 2: Variational Bound Decomposition
The variational bound can be rewritten as:
$$L = \mathbb{E}_q \left[ D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T)) + \sum_{t>1} D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]$$

All KL divergences are between Gaussians and can be computed in closed form.

### Theorem 3: Equivalence to Denoising Score Matching
The $\epsilon$-prediction parameterization yields a training objective equivalent to denoising score matching:
$$L_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \|^2 \right]$$

The sampling procedure resembles Langevin dynamics with learned score.

### Key Empirical Result
- CIFAR-10 unconditional: **FID 3.17**, IS 9.46 (SOTA at the time)
- Demonstrates rate-distortion behavior: majority of codelength describes imperceptible details

---

## Proof Techniques

1. **Reparameterization**: Express $\mathbf{x}_t$ as $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ to enable direct noise-level sampling.

2. **Posterior derivation**: Compute $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ using Bayes' rule on Gaussian distributions:
$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t$$

3. **Loss reweighting**: The simplified objective down-weights small $t$ terms, focusing training on harder denoising tasks at larger noise levels.

---

## Connection to Our Paper

### What We Borrow
1. **Discrete timestep framework**: Our local search operates at discrete timesteps similar to DDPM's $T$ steps.
2. **$\epsilon$-prediction formulation**: The noise prediction parameterization is standard in our work.
3. **Progressive generation insight**: DDPM's observation that large-scale features appear first, details last, motivates our non-uniform compute allocation.
4. **Rate-distortion analysis**: Their analysis showing most bits encode imperceptible details supports our hypothesis that different timesteps require different compute.

### How We Differ
1. **Fixed vs. adaptive compute**: DDPM uses uniform compute per step; we dynamically allocate based on timestep difficulty.
2. **No search at inference**: DDPM samples deterministically from learned distributions; we search over noise trajectories.
3. **Offline profiling**: DDPM doesn't profile per-timestep difficulty; our global scheduler uses profiled statistics.
4. **Early stopping**: DDPM runs all $T$ steps; our online controller can terminate search early.
5. **Compute scaling**: DDPM's quality is fixed at inference; we trade compute for quality via inference-time scaling.

---

## Key Quotes

> "We present high quality image synthesis results using diffusion probabilistic models... our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding."

> "The complete sampling procedure... resembles Langevin dynamics with $\epsilon_\theta$ as a learned gradient of the data density."

> "More than half of the lossless codelength describes imperceptible distortions."

> "Large scale image features appear first and details appear last."

> "We can therefore interpret the Gaussian diffusion model as a kind of autoregressive model with a generalized bit ordering that cannot be expressed by reordering data coordinates."

---

## BibTeX

```bibtex
@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```
