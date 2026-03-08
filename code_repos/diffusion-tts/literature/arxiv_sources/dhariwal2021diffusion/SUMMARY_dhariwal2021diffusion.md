# Summary: Diffusion Models Beat GANs on Image Synthesis

**Paper**: Dhariwal & Nichol (2021) - "Diffusion Models Beat GANs on Image Synthesis"
**arXiv**: 2105.05233
**Venue**: NeurIPS 2021

---

## Setting & Model

### Focus
This paper demonstrates that diffusion models can surpass GANs in image sample quality through (1) architectural improvements to the UNet backbone and (2) classifier guidance for trading off diversity for fidelity during sampling.

### Key Assumptions
1. **Architecture Matters**: Careful UNet architecture design substantially impacts sample quality
2. **Conditional Guidance**: A classifier trained on noisy images can guide the diffusion process toward high-quality, class-consistent samples
3. **Diversity-Fidelity Trade-off**: Like GAN truncation, diffusion models benefit from mechanisms to trade off sample diversity for individual sample quality

### Notation
- $x_t$: Noised sample at timestep $t$
- $\epsilon_\theta(x_t, t)$: Denoising network predicting noise
- $p_\phi(y|x_t)$: Classifier predicting label $y$ from noisy sample
- $\mu_\theta(x_t, t), \Sigma_\theta(x_t, t)$: Predicted mean and variance of reverse process
- $s$: Classifier gradient scale factor

---

## Main Results

### Key Finding 1: Architecture Improvements (ADM)
The "Ablated Diffusion Model" (ADM) architecture improves upon DDPM through:
- Multi-resolution attention at 32x32, 16x16, and 8x8 (not just 16x16)
- 64 channels per attention head (following Transformer conventions)
- BigGAN residual blocks for up/downsampling
- Adaptive Group Normalization (AdaGN) for timestep/class embedding injection

These changes yield FID improvements of ~3 points on ImageNet 128x128.

### Key Finding 2: Classifier Guidance (ADM-G)
The conditional sampling distribution becomes:
$$p_{\theta,\phi}(x_t|x_{t+1}, y) = Z \cdot p_\theta(x_t|x_{t+1}) \cdot p_\phi(y|x_t)$$

Under Gaussian approximation, this shifts the mean by classifier gradients:
$$\mu_{\text{guided}} = \mu + s \cdot \Sigma \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

where $s$ is the gradient scale controlling the diversity-fidelity trade-off.

### Key Finding 3: DDIM Guidance Formulation
For deterministic DDIM sampling, guidance modifies the noise prediction:
$$\hat{\epsilon}(x_t) = \epsilon_\theta(x_t) - \sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

This allows classifier guidance with fewer sampling steps.

### Key Finding 4: State-of-the-Art Results
| Dataset | Model | FID |
|---------|-------|-----|
| ImageNet 128x128 | ADM-G | 2.97 |
| ImageNet 256x256 | ADM-G | 4.59 |
| ImageNet 512x512 | ADM-G | 7.72 |
| LSUN Bedrooms | ADM | 1.90 |

These results surpass BigGAN-deep while maintaining higher recall (distribution coverage).

### Key Finding 5: Gradient Scale Trade-offs
- Low scale ($s \approx 1$): High diversity, lower fidelity
- High scale ($s \approx 10$): Low diversity, high fidelity
- Optimal FID achieved at intermediate scales ($s \approx 1.0$ for 256x256)
- Precision increases monotonically with $s$; recall decreases monotonically

---

## Proof Techniques / Methods

### Conditional Reverse Process Derivation
Starting from Bayes' rule for the conditional noising process:
$$\hat{q}(x_t|x_{t+1}, y) = \frac{q(x_t|x_{t+1}) \cdot \hat{q}(y|x_t)}{\hat{q}(y|x_{t+1})}$$

Key insight: $\hat{q}(y|x_t)$ depends only on $x_t$, not on noisier versions $x_{t+1}$.

### Taylor Expansion Approximation
Approximate $\log p_\phi(y|x_t)$ around $x_t = \mu$:
$$\log p_\phi(y|x_t) \approx (x_t - \mu) \cdot g + C$$
where $g = \nabla_{x_t} \log p_\phi(y|x_t)|_{x_t=\mu}$

This yields a Gaussian with shifted mean: $\mathcal{N}(\mu + \Sigma g, \Sigma)$

### Scaling Interpretation
Scaling gradients by $s$ corresponds to sampling from:
$$\tilde{p}(x_t|y) \propto p(x_t|y) \cdot p_\phi(y|x_t)^{s-1}$$

Larger $s$ sharpens the classifier distribution, focusing on modes.

---

## Connection to Our Paper

### What We Borrow
1. **ADM Architecture**: We use the ADM architecture as our diffusion backbone
2. **Guidance Framework**: Classifier guidance provides a baseline for quality-diversity trade-offs
3. **Evaluation Metrics**: FID, IS, Precision/Recall framework for evaluating our methods
4. **Sampling Step Analysis**: Their finding that 25-250 steps suffice informs our compute allocation

### How We Differ
1. **Compute Allocation Focus**: They use fixed compute per step; we allocate compute non-uniformly across timesteps
2. **Single-Trajectory vs. Multi-Trajectory**: Classifier guidance improves a single trajectory; we search over multiple trajectories
3. **Training-Free vs. Trained Classifier**: Classifier guidance requires training a noisy classifier; our approach is training-free
4. **Global vs. Local Optimization**: Classifier guidance is a local (per-step) modification; we optimize globally across the trajectory

### Relevance to Our Framework
- **Gradient Scale as Compute Proxy**: Higher gradient scale requires more careful sampling; analogous to allocating more compute to important timesteps
- **DDIM Compatibility**: Their DDIM guidance formula shows how to incorporate guidance with fewer steps, relevant to our efficient scheduling
- **Architecture Baseline**: ADM provides our experimental platform for evaluating global scheduling strategies

### Insights for Global Scheduling
- Their observation that "scaling up classifier gradients" improves quality suggests that some timesteps benefit from more "effort"
- The need for 10x gradient scaling at unconditional vs. conditional models suggests different timesteps may have different guidance requirements
- Their sampling schedule sweeps (90,60,60,20,20 for LSUN) show non-uniform step allocation already improves FID

---

## Key Quotes

> "We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models."

> "We achieve this on unconditional image synthesis by finding a better architecture through a series of ablations. For conditional image synthesis, we further improve sample quality with classifier guidance: a simple, compute-efficient method for trading off diversity for fidelity using gradients from a classifier."

> "Using a larger gradient scale focuses more on the modes of the classifier, which is potentially desirable for producing higher fidelity (but less diverse) samples."

> "Classifier guidance is strictly better than BigGAN-deep when trading off FID for Inception Score."

> "While the samples are of similar perceptual quality, the diffusion model contains more modes than the GAN."

---

## BibTeX

```bibtex
@inproceedings{dhariwal2021diffusion,
  title={Diffusion Models Beat {GAN}s on Image Synthesis},
  author={Dhariwal, Prafulla and Nichol, Alexander},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  pages={8780--8794},
  year={2021}
}
```

---

## Additional Notes

### Relevance to Noise Trajectory Search
The paper's classifier guidance can be viewed as a form of "local search" that improves each step independently. Our global scheduler complements this by determining where to apply such local improvements most effectively.

### Compute Analysis
Their compute comparisons (Table 8) show:
- BigGAN-deep: 128-256 V100-days
- ADM-G comparable quality: 63-362 V100-days (with early stopping)

This suggests significant compute savings are possible with smarter training/inference strategies.

### Sampling Schedule Insights
Their LSUN schedule sweep (Table 10) found optimal 250-step schedule (90,60,60,20,20) allocates:
- 36% of steps to earliest noise levels ($t \in [0, 199]$)
- Only 8% to final noise levels ($t \in [800, 999]$)

This non-uniform allocation is a precursor to our systematic global scheduling approach.
