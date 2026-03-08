# Summary: Test-Time Scaling of Diffusion Models via Noise Trajectory Search

**Paper:** arXiv:2506.03164
**Authors:** Vignav Ramesh (Harvard University), Morteza Mardani (NVIDIA)
**Venue:** NeurIPS 2025 submission
**Last Updated:** September 2025

---

## 1. Setting & Model

### Framework
The paper addresses **test-time scaling** for diffusion models through **noise trajectory optimization** in SDE-based samplers. The key insight is that simply increasing denoising steps yields diminishing returns, so they propose optimizing the *noise trajectory* Z = [z_T, z_{T-1}, ..., z_1] instead.

### Formulation
1. **Diffusion as MDP:** They cast the T-step reverse diffusion process as a Markov Decision Process:
   - States: s_t = (c, x_t, t) where c is context (class label or text prompt)
   - Actions: a_t = z_t (the injected noise vector)
   - Transitions: Deterministic given the sampler f
   - Reward: Terminal reward r(x_0, c) at final step only

2. **Relaxation to Contextual Bandits:** To make the MDP tractable, they relax it to a sequence of **T independent contextual bandit problems**:
   - At each timestep t, treat it as an independent optimization
   - Use Tweedie's formula to predict x_0 from intermediate states
   - Ignore inter-step dependencies (the "greedy" approximation)

### Key Assumptions
- SDE-based diffusion (not ODE) - this creates the noise trajectory search problem
- Access to black-box (non-differentiable) reward functions
- Tweedie's formula provides reasonable x_0 estimates at each timestep
- Noise injections at each timestep should be independent (Gaussian)

---

## 2. Main Results

### Proposed Algorithm: Epsilon-Greedy Noise Search
At each timestep t:
1. Sample pivot p ~ N(0, sigma_t^2 * I)
2. For K iterations:
   - With probability epsilon: sample z_t^(i) ~ N(0, sigma_t^2 * I) [global exploration]
   - Otherwise: sample z_t^(i) from neighborhood of p [local exploitation]
   - Set p = best candidate based on Tweedie-predicted reward
3. Use final p as the noise for this timestep

**Key Hyperparameters:**
- epsilon = 0.4 (global exploration probability)
- lambda = 0.15 (step size scaling factor)
- N = 4 (branching factor / noise candidates per iteration)
- K = 20 (local search iterations)

### Performance Claims
- **Up to 164% improvement** over naive sampling on various rewards
- **Outperforms MCTS** while using only O(NKT) NFEs vs O((N+S)T^2) for MCTS
- Works across EDM (class-conditional) and Stable Diffusion (text-to-image)
- Tested rewards: brightness, compressibility, classifier probability, CLIP score

**Quantitative Results (EDM, classifier reward):**
| Method | Reward | NFEs |
|--------|--------|------|
| Naive | 0.378 | 18 |
| Best-of-4 | 0.546 | 72 |
| Beam Search | 0.554 | 144 |
| MCTS (N=4, S=8) | 0.967 | 3888 |
| Zero-Order Search | 0.626 | 1440 |
| **epsilon-greedy** | **0.989** | 1440 |

### Theoretical Result (Regret Bound)
For epsilon-greedy with M = epsilon*N*K global samples per step:
```
E[regret per step] <= delta + C_d * (epsilon*N*K)^(-1/d)
```
where delta is Tweedie approximation error and d is dimensionality.

---

## 3. Timestep Allocation Strategy

### Key Finding: Adaptive Local vs Global Search
The paper discovers that **different timesteps require different search strategies**:

1. **Early timesteps (high noise, large sigma_t):**
   - Reward landscape is nearly flat
   - Global exploration is preferred (mostly random Normal draws)
   - k_bar ~ 10 (random draw selected at all iterations)

2. **Intermediate timesteps (medium noise):**
   - Reward landscape is highly sensitive (demixing occurs)
   - Local exploitation is crucial
   - k_bar << 10 (random draws only in early iterations, then hill-climbing)
   - Lipschitz constant peaks here

3. **Late timesteps (low noise, small sigma_t):**
   - Image is already well-refined
   - Global exploration resumes
   - Limited benefit from local search

### Evidence: Lipschitz Constant Analysis
They estimate the "sensitivity" of the reward landscape via:
```
||grad_{x_t} r(x_hat_0^(t), c)||_2
```
This peaks at intermediate timesteps, confirming that local search matters most there.

### CRITICAL: Uniform K Across Timesteps
**The paper uses uniform K=20 at all timesteps by default.** They mention varying K_t as future work:
> "Letting K_t be the number of local search iterations at timestep t, we can significantly bring down E[K_t] by noting that K_t only need be high at intermediate denoising steps where de-mixing occurs, and can be low at extreme timesteps."

They show that with adaptive K_t (K=20 only for 0.01 <= sigma_t <= 1, K=1 otherwise):
- Performance within +/- 0.04 of original
- NFEs cut by more than half

---

## 4. Connection to Our Paper

### Overlap
| Aspect | Ramesh & Mardani (This Paper) | Our Paper |
|--------|-------------------------------|-----------|
| **Problem** | Noise trajectory optimization | Same |
| **Framework** | MDP relaxed to bandits | Two-level: local operators + global scheduler |
| **Local Search** | epsilon-greedy at each step | Multiple operators (epsilon-greedy is one option) |
| **Timestep Handling** | Implicitly adaptive via epsilon | Explicit global scheduler |
| **Profiling** | Lipschitz analysis (one-time) | Offline profiling + learned models |
| **Compute Allocation** | Uniform K (mostly) | Non-uniform, budget-aware |
| **Early Stopping** | None | Online control with early stopping |

### Key Differences

1. **No Global Scheduler:**
   - They treat each timestep independently (contextual bandit approximation)
   - The epsilon parameter provides *implicit* adaptation (random draws selected more at extreme timesteps)
   - But **K is fixed** - no budget reallocation across timesteps

2. **No Offline Profiling:**
   - Their Lipschitz analysis is post-hoc explanation, not used for scheduling
   - We explicitly profile and learn when search is most effective

3. **No Online Control / Early Stopping:**
   - They run all K iterations at every timestep
   - No mechanism to stop early when good solution found
   - No remaining budget redistribution

4. **Single Local Operator:**
   - They only consider epsilon-greedy (with ablation to zero-order)
   - We support multiple local search operators with adaptive selection

### What We Add

1. **Principled Global Scheduling:** Rather than implicit adaptation via epsilon, we explicitly model and optimize compute allocation across timesteps

2. **Budget-Aware Optimization:** Given total budget B, how to optimally distribute K_t across timesteps

3. **Online Adaptation:** Early stopping when marginal improvement drops, with budget redistribution

4. **Operator Selection:** Not just epsilon-greedy, but portfolio of local operators with learned selection

5. **Offline Profiling:** Systematic characterization of which timesteps benefit from search (beyond one-time Lipschitz plot)

---

## 5. Key Quotes

### On the core idea:
> "We address this by first casting diffusion as a Markov Decision Process (MDP) with a terminal reward, showing tree-search methods such as Monte Carlo tree search (MCTS) to be meaningful but impractical. To balance performance and efficiency, we then resort to a relaxation of MDP, where we view denoising as a sequence of independent contextual bandits."

### On explore vs exploit:
> "This allows us to introduce an epsilon-greedy search algorithm that globally explores at extreme timesteps and locally exploits during the intermediate steps where de-mixing occurs."

### On why epsilon-greedy works:
> "During vanilla zero-order search, at each local search iteration of each timestep we are restricted to the lambda*sqrt(2d)-radius ball around the original pivot... But with epsilon-greedy, we can explore the entire Normal with epsilon probability; a few such 'random jumps' can move us into Z [the optimal region], at which point we see large returns from hill-climbing."

### On timestep-varying search:
> "To our knowledge, this is the first work to demonstrate the importance of adapting the search strategy -- local vs. global -- based on the diffusion timestep."

### On adaptive K (future work hint):
> "Letting K_t be the number of local search iterations at timestep t, we can significantly bring down E[K_t] by noting that K_t only need be high at intermediate denoising steps where de-mixing occurs, and can be low at extreme (beginning and end) timesteps."

### Limitation they acknowledge:
> "For N noise candidates and K local search iterations per timestep, the epsilon-greedy approach requires NK times the NFEs compared to vanilla sampling... This additional computational cost could limit practical deployment scenarios."

---

## 6. Relevance Assessment

**Degree of Overlap: MODERATE-HIGH**

This paper establishes the *problem formulation* (noise trajectory search) and demonstrates that *timestep-adaptive search matters*. However, their solution is:
- **Implicit adaptation** (epsilon parameter causes different behavior at different timesteps)
- **Fixed compute per timestep** (K=20 everywhere by default)
- **No principled scheduling** (just tune epsilon globally)

**Our contribution is orthogonal and complementary:**
- They answer: "What local search algorithm works well?"
- We answer: "How to globally allocate search budget across timesteps?"

**Critical differentiation needed in our paper:**
1. Cite their epsilon-greedy as a strong local operator
2. Show that even their best local operator benefits from our global scheduling
3. Demonstrate gains from non-uniform K_t allocation
4. Show online early stopping provides additional efficiency

---

## 7. BibTeX Entry

```bibtex
@article{ramesh2025nts,
  title={Test-Time Scaling of Diffusion Models via Noise Trajectory Search},
  author={Ramesh, Vignav and Mardani, Morteza},
  journal={arXiv preprint arXiv:2506.03164},
  year={2025},
  note={NeurIPS 2025 submission}
}
```

Alternative format:
```bibtex
@misc{ramesh2025testtimescalingdiffusion,
  title={Test-Time Scaling of Diffusion Models via Noise Trajectory Search},
  author={Vignav Ramesh and Morteza Mardani},
  year={2025},
  eprint={2506.03164},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## 8. Files in This Directory

- `neurips_2025.tex` - Main paper source
- `neurips_2025.bbl` - Bibliography
- `references.bib` - BibTeX references
- `neurips_2025.sty` - Style file
- `*.png` - Figures (tree.png, scaling1.png, scaling2.png, lipschitz_constant_plot.png, etc.)
- `SUMMARY_ramesh2025nts.md` - This summary

---

## 9. Action Items for Our Paper

1. **Must cite** this paper as concurrent/related work
2. **Use their epsilon-greedy** as one of our local operators (or as baseline)
3. **Contrast our approach**: They find epsilon-greedy implicitly adapts; we make this explicit and principled
4. **Build on their insight**: Their Lipschitz analysis validates that timestep-varying K matters
5. **Show complementarity**: Our global scheduler improves upon their fixed-K approach
6. **Acknowledge overlap**: Both papers identify that intermediate timesteps need more search
