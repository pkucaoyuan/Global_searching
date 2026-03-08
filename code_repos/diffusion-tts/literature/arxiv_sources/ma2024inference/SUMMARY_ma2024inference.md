# Summary: Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps

**Citation Key:** ma2024inference
**arXiv:** 2501.09732
**Authors:** Ma et al. (Google DeepMind)

---

## Setting & Model

### Focus
This paper studies inference-time scaling for diffusion models through **noise trajectory search**. The key insight is that performance gains from increasing denoising steps plateau quickly, but additional compute can be invested into searching for better noises during sampling.

### Key Assumptions
1. Some noises are "better" than others - they lead to higher quality generations
2. There is a deterministic mapping from initial/intermediate noises to final samples
3. Verifiers (reward models) can provide meaningful feedback on sample quality
4. The search space over noises is high-dimensional but structured

### Notation
- $D_\theta$: Pre-trained diffusion model
- $\mathcal{V}: \mathbb{R}^{H \times W \times C} \times \mathbb{R}^d \to \mathbb{R}$: Verifier function mapping samples and conditions to scores
- NFE: Number of function evaluations (compute budget)
- $\sigma$: Noise level in the diffusion process
- $N$: Number of candidates in search
- $\lambda$: Neighborhood radius for zero-order search

---

## Main Results

### 1. Search Framework Design Axes
The paper identifies two primary design axes:
- **Verifiers**: Oracle (FID/IS), Supervised (CLIP/DINO classifiers), Self-supervised (feature similarity)
- **Algorithms**: Random Search (best-of-N), Zero-Order Search (iterative local refinement), Search over Paths (trajectory-level search)

### 2. Search Algorithms

**Random Search:**
- Sample N Gaussian noises, generate samples, select best according to verifier
- Simple but effective baseline
- Can lead to "verifier hacking" (overfitting to verifier bias)

**Zero-Order Search (ZO-N):**
1. Start with random Gaussian noise as pivot
2. Find N candidates in neighborhood $S_n^\lambda = \{y: d(y,n) = \lambda\}$
3. Run ODE solver, evaluate with verifier
4. Update pivot to best candidate, repeat
- Maintains locality, reducing verifier hacking

**Search over Paths (Paths-N):**
1. Sample N initial noises, run ODE to noise level $\sigma$
2. For each noisy sample, add M noises and simulate forward noising
3. Run ODE back, evaluate with verifier, keep top N
4. Repeat until $\sigma = 0$
- Searches along diffusion sampling trajectories
- Requires $\Delta b > \Delta f$ for termination

### 3. Scaling Behavior
- Substantial improvements beyond purely scaling NFEs with denoising steps
- Performance continues to improve with search budget allocation
- Small models with search can match or outperform large models without search

### 4. Verifier-Task Alignment
- No single verifier-algorithm combination is universally optimal
- Each task requires specific search setup for best performance
- Verifier Ensemble (averaging rankings) provides good generalizability
- "Verifier hacking" occurs when search exploits verifier biases

---

## Proof Techniques / Methods

### Experimental Methodology
1. **ImageNet Class-Conditional Generation**: SiT-XL model, Heun sampler, 250 NFE base
2. **Text-to-Image**: FLUX.1-dev, PixArt-$\Sigma$ on DrawBench and T2I-CompBench
3. **Metrics**: FID, IS, CLIPScore, Aesthetic Score, ImageReward, LLM Grader

### Key Experimental Findings
- Random search with oracle verifiers scales substantially
- Zero-order search with N=4 effectively estimates local optimum
- Search over Paths: small N efficient for low budget, large N better at scale
- NFEs/iter has compute-optimal regions (50 for ImageNet, 30 for text-to-image)

---

## Connection to Our Paper

### Direct Relevance: HIGH

This paper is **highly relevant** to our work on "Global Scheduling of Noise Trajectory Search in Diffusion Models":

1. **Local Search Operators**: Ma et al.'s three search algorithms (Random, Zero-Order, Paths) correspond to different **local search operators** in our framework. Their Zero-Order Search and Search over Paths both implement local refinement strategies.

2. **Global Scheduling Missing**: The paper acknowledges different compute-optimal configurations but does **not** provide a principled global scheduler. They note:
   - "NFEs/iter can reveal distinct compute-optimal regions"
   - "Different values of N lead to different scaling behavior, with small N being compute efficient in small generation budget, and large N having advantage when scaling up"

   This is exactly the gap our global scheduler addresses - how to dynamically allocate compute across timesteps.

3. **Timestep-Dependent Difficulty**: Their Search over Paths algorithm implicitly acknowledges timestep-dependent structure, but treats all timesteps uniformly. Our paper's key insight is that **difficulty varies across the noise trajectory**, requiring adaptive allocation.

4. **Verifier Hacking**: Their observation of "verifier hacking" motivates our approach - rather than relying on a single verifier, our global scheduler can balance local improvements while respecting trajectory-level constraints.

5. **Axes of Compute Investment**: Section 6 discusses three axes:
   - Number of search iterations
   - Compute per search iteration
   - Compute for final generation

   Our global scheduler provides a principled framework for optimizing across these axes simultaneously.

### Key Differences
- **Ma et al.**: Focus on local search algorithms, uniform compute allocation across steps
- **Our Paper**: Global scheduler that allocates compute non-uniformly based on timestep difficulty

### Opportunities for Synthesis
Our global scheduler could be combined with their search algorithms to create a **hierarchical system**: global scheduler determines budget per timestep, local operators (e.g., Zero-Order Search) perform the actual optimization.

---

## Key Quotes

> "Rather than solely allocating NFEs for denoising steps, which often leads to a quick performance plateau, this work investigates methods to effectively utilize compute during inference through search."

> "For Zero-Order Search, we note that the effectiveness of increasing N is marginal, and N=4 seems to already be a good estimation of the local optimum."

> "Different values of N lead to different scaling behavior, with small N being compute efficient in small generation budget, and large N having advantage when scaling up compute more."

> "Due to the locality nature of the two algorithms, both of them manage to alleviate the diversity issue of FID to some extent while maintaining a scaling Inception Score."

> "Smaller NFEs/iter during search enables efficient convergence, though with a lower final performance. Conversely, larger NFEs/iter result in slower convergence but yield improved performance."

> "The effectiveness of a verifier depends on how well its criteria align with the specific requirements of the task, with certain verifiers being better suited for particular tasks than others."

---

## BibTeX

```bibtex
@article{ma2024inference,
  title={Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps},
  author={Ma, Nanye and Goldstein, Mark and Albergo, Michael S. and Boffi, Nicholas M. and Vanden-Eijnden, Eric and Xie, Saining},
  journal={arXiv preprint arXiv:2501.09732},
  year={2025}
}
```
