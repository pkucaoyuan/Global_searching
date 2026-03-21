# Prompt: Extend Theory to General Local Search Operators

## Context

**Paper**: "Leveraging Inference-Time Compute for Diffusion Models via Global Scheduling of Denoising Trajectories"

We study how to allocate a computational budget $B$ (in function evaluations) across $T$ denoising timesteps in a diffusion model. At each timestep $t$, a **local search operator** $\mathcal{L}_t$ spends $K_t$ NFE to find a good noise $\varepsilon_t$. A **global scheduler** decides $\{K_t\}$ under $\sum_t K_t = B$.

The paper has two scheduling results:
1. **Offline (Prop offline)**: Under prompt-independent $\sigma_t$, optimal allocation follows a water-filling structure ($K_t^*$ increasing in $\sigma_t$).
2. **Online (Prop online)**: Under prompt-dependent $\sigma_t(c, x_t)$, any fixed allocation incurs a Jensen gap; online adaptation via dual-threshold early stopping can partially recover this gap.

Both results currently assume **pure random search** (draw $K$ i.i.d. candidates, keep the best), giving $G_t(K) = \sigma_t a_K$ where $a_K = \mathbb{E}[\max(Z_1,\ldots,Z_K)]$.

---

## Current Theory

### SDE Setup

Reverse-time SDE in Euler form:
$$X_{t-h} = X_t + b_t(X_t, c)\,h + g_t\sqrt{h}\,\xi, \qquad \xi \sim \mathcal{N}(0, I)$$

Verifier score function: $\Psi_t(x; c) := v(D_\theta(x, t{-}h); c)$

Deterministic Euler point: $\bar{X}_{t-h} := X_t + b_t(X_t, c)\,h$

**Assumption (Verifier Smoothness)**: $\Psi_t(\cdot; c) \in C^2(\mathbb{R}^d)$, $\|\nabla^2\Psi_t\|_{op} \le L_t$.

### Location-Scale Theorem (current, random search only)

Define $\mu_t := \Psi_t(\bar{X}_{t-h}; c)$ and $\sigma_t := g_t\sqrt{h}\|\nabla\Psi_t(\bar{X}_{t-h}; c)\|$. Then:
- $S_t = \mu_t + \sigma_t Z_t + R_t$, $Z_t \sim \mathcal{N}(0,1)$, $|R_t| = O(g_t^2 h)$
- For $K$ i.i.d. candidates: $G_t(K) = \sigma_t a_K + O_K(g_t^2 h)$

### Current Offline Result (Prop offline)

For prompt-independent $\sigma_t$:
$$\max_{\{K_t\}: \sum K_t = B} \sum_{t=1}^T \sigma_t\,a_{K_t}$$
has water-filling solution: $K_t^*$ increasing in $\sigma_t$, non-uniform strictly beats uniform.

### Current Online Result (Prop online)

For prompt-dependent $\sigma_t = \sigma_t(c, x_t)$, define oracle:
$$V^*(\boldsymbol{\sigma}) = \max_{\{K_t\}: \sum K_t = B} \sum_t \sigma_t\,a_{K_t}$$
Then:
- (i) $V^*$ is convex in $\boldsymbol{\sigma}$. Jensen gap: $\mathbb{E}[V^*(\boldsymbol{\sigma})] \ge V^*(\bar{\boldsymbol{\sigma}})$, equality iff $\boldsymbol{\sigma}$ constant across instances.
- (ii) Marginal gain factorizes: $G_t'(K) = \sigma_t \cdot a_K'$ (sensitivity $\times$ exhaustion). Dual-threshold stopping: stop when BOTH $\sigma_t$ small AND $a_K'$ small.

---

## The Problem: Extension to General Local Search

### What We Want

Extend **both** the offline (water-filling) and online (Jensen gap + dual-threshold) results to a class of local search operators beyond pure random search.

### Practical Operators We Use

These are the actual operators implemented in our codebase. **Assumptions must be verifiable for each of these**:

**Operator 1: Random search** (current theory covers this)
- Draw $K$ i.i.d. $\xi_k \sim \mathcal{N}(0, I)$, evaluate $\Psi_t(\bar{X} + g_t\sqrt{h}\,\xi_k)$, keep the best.
- One NFE per candidate. $K$ candidates = $K$ NFE.

**Operator 2: $\epsilon$-greedy search** (used in our SD and EDM experiments)
- Maintain a pivot noise $\xi^*$ (current best).
- Each iteration: with probability $\epsilon$, draw $\xi_{new} \sim \mathcal{N}(0,I)$ (explore); with probability $1-\epsilon$, set $\xi_{new} = \xi^* + \eta\,\zeta$ where $\zeta \sim \mathcal{N}(0,I)$ (exploit by perturbing current best).
- Evaluate $\Psi_t(\bar{X} + g_t\sqrt{h}\,\xi_{new})$. If better than $\Psi_t(\bar{X} + g_t\sqrt{h}\,\xi^*)$, update $\xi^* \leftarrow \xi_{new}$.
- One NFE per iteration. $K$ iterations = $K$ NFE.

**Operator 3: Zero-order optimization** (gradient-free hill-climbing)
- Each iteration: estimate gradient via finite differences in 1 or 2 random directions.
- Update: $\xi^{(j+1)} = \xi^{(j)} + \eta \hat{g}$ where $\hat{g}$ is the estimated gradient direction.
- 1-2 NFE per iteration (depending on one-sided or two-sided difference).

**Operator 4: Langevin MCMC in noise space**
- $\xi^{(j+1)} = \xi^{(j)} + \eta \nabla_\xi \Psi_t(\bar{X} + g_t\sqrt{h}\xi^{(j)}) + \sqrt{2\eta}\,\zeta$
- Targets the distribution $\propto \exp(\Psi_t / \tau)$ for temperature $\tau$.

---

## Required Derivation

### Part 1: Abstract Local Search Framework

Define a general local search operator $\mathcal{L}$ as producing iterates $\xi^{(0)}, \xi^{(1)}, \ldots, \xi^{(K)}$ in $\mathbb{R}^d$, with gain:
$$G_t^{\mathcal{L}}(K) := \mathbb{E}\Bigl[\max_{0 \le j \le K} \Psi_t(\bar{X}_{t-h} + g_t\sqrt{h}\,\xi^{(j)}; c) \,\Big|\, X_t, c\Bigr] - \mu_t$$

Propose **assumptions on $\mathcal{L}$** that are:
- **Minimal**: As few and as weak as possible
- **Grounded**: Each assumption should correspond to a natural property of the actual operators listed above (not abstract for its own sake)
- **Sufficient**: Together they imply the results below

Suggested starting point (modify as needed):

- **A1 (Monotone improvement)**: $G_t^{\mathcal{L}}(K)$ is non-decreasing in $K$
  - *Grounded*: All four operators keep the best-so-far, so the running maximum can only improve. This is a property of the "keep best" wrapper, not the search itself.

- **A2 (Diminishing returns)**: $G_t^{\mathcal{L}}(K)$ is concave in $K$
  - *Grounded*: For random search, this follows from order-statistic theory ($a_K$ is concave). For $\epsilon$-greedy, the explore component contributes $a_{\lceil\epsilon K\rceil}$ (concave) while the exploit component converges (diminishing marginal gain). For zero-order, gradient ascent on a smooth function with bounded gradient has diminishing gains. For Langevin, mixing time bounds imply diminishing returns.
  - *If full concavity is too strong*: Weaken to $G_t^{\mathcal{L}}(K+1) - G_t^{\mathcal{L}}(K)$ is non-increasing (discrete concavity), or even just sub-additivity.

- **A3 (Sensitivity scaling)**: In the linearized regime ($g_t\sqrt{h} \to 0$),
  $$G_t^{\mathcal{L}}(K) = \sigma_t \cdot \phi_K^{\mathcal{L}} + O(g_t^2 h)$$
  where $\phi_K^{\mathcal{L}} \ge 0$ depends on the operator $\mathcal{L}$ and budget $K$, but **not on the timestep $t$**.
  - *Grounded*: In the linearized regime, $\Psi_t(\bar{X} + g_t\sqrt{h}\,\xi) \approx \mu_t + \sigma_t\langle\nabla\Psi_t/\|\nabla\Psi_t\|, \xi\rangle$. Any local search on $\xi$ is effectively optimizing a linear function $\langle u, \xi\rangle$ (where $u = \nabla\Psi_t/\|\nabla\Psi_t\|$ is a fixed unit vector), scaled by $\sigma_t$. The gain from optimizing a linear function depends on the search method but not on the direction $u$ or the scale $\sigma_t$ beyond a multiplicative factor.

### Part 2: Verify Assumptions for Each Operator

For each of the four operators, **prove or argue carefully** that A1-A3 hold. For A3, derive the operator-specific $\phi_K^{\mathcal{L}}$:

| Operator | $\phi_K$ | How derived |
|----------|----------|-------------|
| Random search | $a_K = \mathbb{E}[\max(Z_1,\ldots,Z_K)]$ | Order statistics (already proved) |
| $\epsilon$-greedy | $\phi_K^{(\epsilon)} = ?$ | Mixture: $\epsilon$-fraction random + $(1-\epsilon)$-fraction local hill-climb |
| Zero-order | $\phi_K^{(ZO)} = ?$ | Gradient ascent rate on linear function + Gaussian concentration of $\|\xi\|$ |
| Langevin | $\phi_K^{(L)} = ?$ | Mixing time to stationary distribution of $\propto \exp(\langle u, \xi\rangle / \tau)$ |

### Part 3: General Offline Result (Water-Filling)

Under A1-A3, the allocation problem becomes:
$$\max_{\{K_t\}: \sum K_t = B} \sum_{t=1}^T \sigma_t \cdot \phi_{K_t}^{\mathcal{L}}$$

**Prove**: If $\phi_K$ is concave and increasing with $\phi_1 = 0$, then:
- (i) Optimal $K_t^*$ is increasing in $\sigma_t$ (water-filling with operator-specific returns)
- (ii) Non-uniform strictly dominates uniform when $\{\sigma_t\}$ are heterogeneous
- (iii) The gain from non-uniform allocation over uniform grows with the dispersion of $\{\sigma_t\}$

Note: This should be a straightforward generalization since the proof of the current Prop (offline) only uses concavity of $a_K$, not the specific form $a_K = \mathbb{E}[\max Z_k]$. So replacing $a_K$ with a general concave $\phi_K$ should work directly.

### Part 4: General Online Result (Jensen Gap + Dual Threshold)

This is the part that needs new work. Under A1-A3 with prompt-dependent $\sigma_t(c, x_t)$:

**4a. Jensen gap still holds?**

Define the generalized oracle:
$$V^*_{\mathcal{L}}(\boldsymbol{\sigma}) = \max_{\{K_t\}: \sum K_t = B} \sum_t \sigma_t \cdot \phi_{K_t}^{\mathcal{L}}$$

- Is $V^*_{\mathcal{L}}$ still convex in $\boldsymbol{\sigma}$?
  - Yes if for each fixed $\{K_t\}$, the map $\boldsymbol{\sigma} \mapsto \sum_t \sigma_t \phi_{K_t}$ is linear (it is), and $V^*$ is the pointwise max of linear functions (convex). This argument does NOT depend on the specific form of $\phi_K$.
  - **So the Jensen gap $\mathbb{E}[V^*_{\mathcal{L}}(\boldsymbol{\sigma})] \ge V^*_{\mathcal{L}}(\bar{\boldsymbol{\sigma}})$ should hold for any $\phi_K$.**

**4b. Marginal gain factorization still holds?**

With general $\phi_K$:
$$G_t'(K) = \sigma_t \cdot (\phi_K^{\mathcal{L}})' $$

This is still a product of timestep-specific sensitivity $\sigma_t$ and operator-specific exhaustion $(\phi_K^{\mathcal{L}})'$.

- The dual-threshold argument should carry over: stop when BOTH $\sigma_t$ is small (sensitivity low) AND $(\phi_K^{\mathcal{L}})'$ is small (search exhausted).
- The running mean $\bar{g}$ still estimates the shadow price $\lambda^*$.
- The conjunction of both thresholds still prevents premature stopping.

**Prove or argue**: The online controller design from the current paper (Algorithm 1, the gain + variance dual threshold) remains valid under A1-A3, regardless of which $\mathcal{L}$ is used.

**4c. Does the choice of $\mathcal{L}$ affect the Jensen gap magnitude?**

Interesting question: Different operators have different $\phi_K$ functions. Does the Jensen gap $\mathbb{E}[V^*] - V^*(\bar{\boldsymbol{\sigma}})$ depend on the curvature of $\phi_K$? Specifically:
- Faster-growing $\phi_K$ (e.g., zero-order with linear initial growth) → larger Jensen gap?
- Slower-growing $\phi_K$ (e.g., random search with $\sqrt{\log K}$ growth) → smaller Jensen gap?

If so, this would mean that **online adaptation is more valuable for more efficient local search operators**, which is a practically interesting insight.

### Part 5: Operator Comparison Under Fixed Budget

Given the general framework, analyze:

- **Under what conditions does zero-order beat random search?**
  In the linearized regime, random search achieves $G_t(K) \approx \sigma_t\sqrt{2\log K}$ while zero-order might achieve $G_t(K) \approx \sigma_t \cdot \min(cK, \sqrt{d})$ (linear growth capped by the sphere constraint $\|\xi\| \approx \sqrt{d}$).
  - For small $K$: zero-order is better ($cK > \sqrt{2\log K}$ when $K$ is small)
  - For large $K$: random search catches up (if $K$ is large enough for $\sqrt{2\log K}$ to matter)
  - Crossover point: $K^* \approx ?$

- **Does the optimal operator choice depend on the timestep?**
  If $\sigma_t$ is large, is one operator better? If $\sigma_t$ is small? This could motivate **operator selection** as part of the global scheduling (mentioned in our MDP appendix as future work).

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $X_t$ | Latent state at timestep $t$ |
| $b_t(X_t, c)$ | Drift of reverse SDE (from learned score network) |
| $g_t$ | Diffusion coefficient at timestep $t$ |
| $h$ | Euler step size |
| $\xi$ | Noise variable, $\xi \sim \mathcal{N}(0, I_d)$ |
| $\Psi_t(x; c)$ | Verifier score function: $v(D_\theta(x, t-h); c)$ |
| $\bar{X}_{t-h}$ | Deterministic Euler point: $X_t + b_t h$ |
| $\mu_t$ | $\Psi_t(\bar{X}_{t-h}; c)$ (deterministic baseline score) |
| $\sigma_t$ | $g_t\sqrt{h}\|\nabla\Psi_t(\bar{X}_{t-h}; c)\|$ (timestep sensitivity) |
| $S_t$ | Score from one noise candidate |
| $G_t^{\mathcal{L}}(K)$ | Expected gain from $K$ iterations of local search $\mathcal{L}$ |
| $\phi_K^{\mathcal{L}}$ | Operator-specific gain sequence: $G_t^{\mathcal{L}}(K) = \sigma_t \phi_K^{\mathcal{L}} + O(\cdot)$ |
| $a_K$ | Random-search gain: $\mathbb{E}[\max(Z_1,\ldots,Z_K)]$ |
| $K_t$ | Budget allocated to timestep $t$ |
| $B$ | Total NFE budget |
| $V^*_{\mathcal{L}}(\boldsymbol{\sigma})$ | Oracle value under operator $\mathcal{L}$ and sensitivity profile $\boldsymbol{\sigma}$ |
| $\lambda^*$ | Shadow price (Lagrange multiplier) of the budget constraint |
| $L_t$ | Hessian bound: $\|\nabla^2\Psi_t\|_{op} \le L_t$ |
| $d$ | Dimension of latent space |

## Hints

### Hint 1: Linearization is the key
In the regime $g_t\sqrt{h} \to 0$, the score is approximately linear in $\xi$:
$$\Psi_t(\bar{X} + g_t\sqrt{h}\,\xi) \approx \mu_t + \sigma_t \frac{\langle \nabla\Psi_t, \xi\rangle}{\|\nabla\Psi_t\|}$$
Any local search on $\xi$ reduces to optimizing the linear function $f(\xi) = \langle u, \xi\rangle$ where $u$ is a fixed unit vector, scaled by $\sigma_t$. This is why $\sigma_t$ is the universal scaling factor.

### Hint 2: The sphere constraint
Since $\xi \sim \mathcal{N}(0, I_d)$, we have $\|\xi\|^2 \approx d$ by concentration. So effectively any search is constrained to a sphere of radius $\approx \sqrt{d}$. The maximum of $\langle u, \xi\rangle$ on this sphere is $\sqrt{d}$, giving an upper bound $G_t^{\mathcal{L}}(K) \le \sigma_t\sqrt{d}$ for all $K$ and $\mathcal{L}$.

### Hint 3: Jensen gap argument is purely structural
The Jensen gap $\mathbb{E}[V^*(\boldsymbol{\sigma})] \ge V^*(\bar{\boldsymbol{\sigma}})$ follows from $V^*$ being a pointwise max of linear functions. This argument uses ONLY:
- Linearity of $\sum_t \sigma_t \phi_{K_t}$ in $\boldsymbol{\sigma}$ (for fixed $\{K_t\}$)
- $V^* = \max$ over a finite set of allocations

It does NOT use the specific form of $\phi_K$. So this result should hold for any operator.

### Hint 4: Dual-threshold stopping logic
The current dual-threshold ($\sigma_t$ small AND $(\phi_K)' $ small) prevents two types of premature stopping:
- Low exhaustion, high sensitivity → search is still productive
- Low sensitivity, high exhaustion → timestep is inherently unimportant

This logic applies to any $\phi_K$ as long as the marginal gain factorizes as $\sigma_t \cdot (\phi_K)'$. Under A3, this factorization holds for all operators.

### Hint 5: $\epsilon$-greedy as a mixture
$\epsilon$-greedy can be decomposed: in expectation, $\epsilon K$ iterations are fresh random draws and $(1-\epsilon)K$ are local perturbations. The fresh draws contribute $\approx a_{\epsilon K}$ (random search on a subset). The local perturbations contribute additional gain from hill-climbing. So:
$$\phi_K^{(\epsilon)} \ge a_{\lceil\epsilon K\rceil}$$
with equality when the exploitation steps contribute nothing. The question is how much the exploitation adds.

---

## Deliverables

1. **Assumption set (A1-A3)**: Grounded in the actual operators, not abstract
2. **Verification**: Prove/argue each of {random search, $\epsilon$-greedy, zero-order, Langevin} satisfies A1-A3
3. **General Offline Theorem**: Water-filling under general $\phi_K$ (should be a straightforward extension)
4. **General Online Theorem**: Jensen gap + dual-threshold validity under general $\phi_K$ (needs careful argument, especially 4c)
5. **Operator-specific $\phi_K$**: Characterization (exact or asymptotic) for each operator
6. **Operator comparison**: When does zero-order beat random search? Crossover in $K$?
7. **LaTeX code**: Ready to insert in §4.3-§4.4

## Verification Checklist

- [ ] Random search recovered as special case ($\phi_K = a_K$)
- [ ] $\phi_K$ is concave (or sub-additive) for all proposed operators
- [ ] $\sigma_t$ is the universal scaling factor for all operators
- [ ] Water-filling result (Prop offline) holds for general $\phi_K$
- [ ] **Jensen gap result (Prop online, part i) holds for general $\phi_K$**
- [ ] **Marginal factorization (Prop online, part ii) holds: $G_t' = \sigma_t \cdot (\phi_K)'$**
- [ ] **Dual-threshold stopping logic still valid for general $\phi_K$**
- [ ] Remainder terms controlled
- [ ] No hidden dependence on $t$ in $\phi_K$

## Output Format

```latex
% Insert in §4.3 after Corollary (SDE-based approximate allocation)

\begin{assumption}[Local search regularity]
\label{asm:local-search}
A local search operator $\mathcal{L}$ produces iterates
$\xi^{(0)}, \ldots, \xi^{(K)}$ in $\mathbb{R}^d$ with gain
$G_t^{\mathcal{L}}(K) := \mathbb{E}[\max_{j} \Psi_t(\bar{X} + g_t\sqrt{h}\,\xi^{(j)}) \mid X_t, c] - \mu_t$.
We assume:
(A1) ...
(A2) ...
(A3) ...
\end{assumption}

\begin{theorem}[General gain factorization]
\label{thm:general-gain}
Under Assumptions~\ref{asm:smooth} and~\ref{asm:local-search}, ...
\end{theorem}

\begin{corollary}[Offline water-filling for general operators]
...
\end{corollary}

\begin{corollary}[Online Jensen gap for general operators]
...
\end{corollary}

\begin{remark}[Dual-threshold stopping under general operators]
...
\end{remark}

\begin{example}[Random search]  $\phi_K = a_K$. ...  \end{example}
\begin{example}[$\epsilon$-greedy]  $\phi_K^{(\epsilon)} = ...$  \end{example}
\begin{example}[Zero-order optimization]  $\phi_K^{(ZO)} = ...$  \end{example}
```
