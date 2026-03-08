# Summary: High-Resolution Image Synthesis with Latent Diffusion Models

**Paper**: Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)
**arXiv**: 2112.10752
**Venue**: CVPR 2022

---

## Setting & Model

### Focus
Latent Diffusion Models (LDMs) address the computational inefficiency of pixel-space diffusion models by operating in a compressed latent space learned by an autoencoder. This enables high-resolution image synthesis with substantially reduced training and inference costs.

### Key Assumptions
1. **Perceptual vs. Semantic Compression**: Image formation can be decomposed into two stages: perceptual compression (removing imperceptible high-frequency details) and semantic compression (learning the conceptual composition).
2. **Latent Space Sufficiency**: A pretrained autoencoder can provide a perceptually equivalent but lower-dimensional space suitable for diffusion modeling.
3. **Spatial Structure Preservation**: The latent space preserves 2D spatial structure, enabling efficient convolutional processing via UNet architectures.

### Notation
- $x \in \mathbb{R}^{H \times W \times 3}$: Input image in RGB space
- $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$: Latent representation
- $\tilde{x} = \mathcal{D}(z)$: Reconstructed image
- $f = H/h = W/w$: Downsampling factor (typically $f \in \{4, 8, 16\}$)
- $\epsilon_\theta(z_t, t)$: Denoising network predicting noise in latent space
- $\tau_\theta(y)$: Conditioning encoder mapping conditioning $y$ to intermediate representation

---

## Main Results

### Key Finding 1: Latent Space Diffusion Training Objective
The LDM training objective operates in the compressed latent space:

$$L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(z_t, t)\|_2^2\right]$$

This enables training on high-resolution images with substantially reduced computational cost compared to pixel-space diffusion.

### Key Finding 2: Optimal Compression Rate
LDM-4 and LDM-8 (downsampling factors of 4 and 8) achieve the best balance between:
- Computational efficiency
- Perceptual fidelity in reconstructions
- Sample quality metrics (FID)

Too little compression (LDM-1, LDM-2) results in slow training; too much compression (LDM-32) causes information loss.

### Key Finding 3: Cross-Attention Conditioning
General-purpose conditioning via cross-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

where:
- $Q = W_Q^{(i)} \cdot \varphi_i(z_t)$ (queries from UNet features)
- $K = W_K^{(i)} \cdot \tau_\theta(y)$, $V = W_V^{(i)} \cdot \tau_\theta(y)$ (keys/values from conditioning)

This enables flexible conditioning on text, layouts, semantic maps, etc.

### Key Finding 4: State-of-the-Art Results
- CelebA-HQ 256x256: FID 5.11 (new SOTA for likelihood-based models)
- ImageNet 256x256 (class-conditional): FID 3.60 with classifier-free guidance
- Competitive text-to-image results on MS-COCO with 1.45B parameter model

---

## Proof Techniques / Methods

### Two-Stage Training
1. **Stage 1**: Train autoencoder with perceptual loss and adversarial objective:
   $$L_{AE} = \min_{\mathcal{E},\mathcal{D}} \max_\psi \left(L_{rec}(x, \mathcal{D}(\mathcal{E}(x))) - L_{adv}(\mathcal{D}(\mathcal{E}(x))) + \log D_\psi(x) + L_{reg}\right)$$

2. **Stage 2**: Train diffusion model in frozen latent space using the standard denoising objective.

### Latent Space Regularization
Two variants explored:
- **KL-regularization**: Soft KL penalty towards standard normal (VAE-like)
- **VQ-regularization**: Vector quantization layer (VQGAN-like)

### Convolutional Sampling
For dense conditioning tasks (super-resolution, inpainting, semantic synthesis), LDMs can be applied convolutionally to generate images up to megapixel resolution.

---

## Connection to Our Paper

### What We Borrow
1. **Evaluation Platform**: Stable Diffusion (LDM-4/LDM-8) serves as our primary evaluation testbed for noise trajectory search
2. **Latent Space Structure**: Our local search operators work in the same latent space where LDMs operate
3. **Computational Efficiency Motivation**: The LDM approach of operating in compressed space aligns with our goal of efficient inference-time scaling

### How We Differ
1. **Focus on Inference vs. Training**: LDMs focus on training efficiency; we focus on inference-time compute allocation
2. **Fixed Architecture vs. Dynamic Scheduling**: LDMs use fixed sampling schedules; we propose adaptive global scheduling across timesteps
3. **Single Trajectory vs. Trajectory Search**: LDMs sample a single noise trajectory; we explore multiple candidate trajectories with compute-aware branching
4. **Complementary Approaches**: Our global scheduling framework can be applied on top of LDMs to improve sample quality through intelligent inference-time compute allocation

### Relevance to Our Framework
- The hierarchical latent structure (high/low resolution paths in UNet) suggests non-uniform importance of timesteps
- The observation that perceptual compression happens early while semantic compression happens later motivates our timestep-aware compute allocation
- LDM's efficiency gains at training time parallel our efficiency goals at inference time

---

## Key Quotes

> "By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models achieve state-of-the-art synthesis results on image data and beyond."

> "We observe that although diffusion models allow to ignore perceptually irrelevant details by undersampling the corresponding loss terms, they still require costly function evaluations in pixel space."

> "Our latent diffusion models (LDMs) achieve new state-of-the-art scores for image inpainting and class-conditional image synthesis and highly competitive performance on various tasks... while significantly reducing computational requirements compared to pixel-based DMs."

> "LDM-4 and -8 offer the best conditions for achieving high-quality synthesis results."

> "Classifier-free diffusion guidance greatly boosts sample quality."

---

## BibTeX

```bibtex
@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10684--10695},
  year={2022}
}
```

---

## Additional Notes

### Relevance to Global Scheduling
The paper's finding that different compression levels (f=4 vs f=8 vs f=16) trade off efficiency and quality suggests that similar trade-offs exist across the temporal dimension of diffusion sampling. Our global scheduler can be viewed as dynamically choosing the "compression level" of compute at each timestep.

### Connection to EDM
LDMs with their learned latent space complement EDM's focus on noise scheduling. Our framework evaluates on both, leveraging LDM's latent structure and EDM's principled noise level formulation.
