# Aolaritei, Van Parys, Lam, Jordan (2025) -- "Stochastic Optimization with Optimal Importance Sampling"

**Authors:** Liviu Aolaritei (UC Berkeley), Bart P.G. Van Parys (CWI Amsterdam), Henry Lam (Columbia), Michael I. Jordan (UC Berkeley / Inria)

---

## 1. Setting & Model

The paper studies **convex stochastic optimization with linear constraints**:

$$\min_\theta \; f(\theta) := \mathbb{E}_{X \sim \mathbb{P}}[F(\theta, X)] \quad \text{s.t.} \quad \theta \in \Theta := \{\theta \in \mathbb{R}^s : A\theta \leq b\}$$

where the expectation cannot be evaluated in closed form and is accessed via stochastic gradients $G(\theta, X) = \nabla_\theta F(\theta, X)$.

**Core challenge -- Circularity:** The optimal importance sampling (IS) distribution depends on the unknown optimal solution $\theta^\star$, but finding $\theta^\star$ efficiently requires a good IS distribution. This creates a **circular dependency** between decision optimization and IS calibration.

**IS framework.** A parametrized family of IS distributions $\{\mathbb{P}_\mu\}_{\mu \in \mathcal{M}}$ is considered, with $\mathcal{M} = \{\mu \in \mathbb{R}^m : C\mu \leq d\}$. IS-reweighted stochastic gradients are $G_\mu(\theta, x) := \ell(x, \mu) G(\theta, x)$ where $\ell(x, \mu) = d\mathbb{P}/d\mathbb{P}_\mu(x)$ is the likelihood ratio. The optimal IS parameter $\mu^\star$ minimizes:

$$\mu^\star = \arg\min_{\mu \in \mathcal{M}} \; \text{Tr}\left(\text{Var}_{X^{(\mu)} \sim \mathbb{P}_\mu}\left[\text{P}_{A^\star_a} G_\mu(\theta^\star, X^{(\mu)})\right]\right)$$

where $\text{P}_{A^\star_a}$ is the projector onto the null space of the active constraints at $\theta^\star$.

Three IS families are studied in detail:
- **Exponential tilting:** $\ell_{\text{ET}}(x,\mu) = \exp(-\mu^\top x + \phi(\mu))$ where $\phi$ is the cumulant generating function.
- **Mean translation:** $\ell_{\text{MT}}(x, \mu) = \exp(-(\Delta(x) - \Delta(x-\mu)))$ for log-concave base density with $\Delta(x) = -\log p(x)$.
- **Mixture models:** $\ell_{\text{MM}}(x, \mu) = (\sum_i \mu_i \ell_i(x)^{-1})^{-1}$, mixing over $I$ base importance samplers.

All three satisfy the key structural requirement: $\mu \mapsto \ell(x, \mu)$ is **log-convex** (Assumption 3.1(i)), which ensures convexity of the IS calibration subproblem.

**Proposed algorithm.** A **single-loop joint Nesterov Dual Averaging (NDA)** iteration that simultaneously updates $(\theta, \mu)$:

$$\begin{bmatrix}\theta_{n+1} \\ \mu_{n+1}\end{bmatrix} = \arg\min_{(\theta, \mu) \in \Theta \times \mathcal{M}} \left\{ \left\langle \sum_{k=0}^n \alpha_{k+1} \begin{bmatrix} G_{\mu_k}(\theta_k, X_{k+1}^{(\mu_k)}) \\ H(\theta_k, \mu_k, X_{k+1}) \end{bmatrix}, \begin{bmatrix}\theta \\ \mu\end{bmatrix} \right\rangle + \frac{1}{2}\left\|\begin{bmatrix}\theta - \theta_0 \\ \mu - \mu_0\end{bmatrix}\right\|^2 \right\}$$

with step sizes $\alpha_n = \alpha / n^\gamma$, $\gamma \in (1/2, 1)$, averaged iterates $\bar{\theta}_n = n^{-1}\sum_{i=0}^{n-1}\theta_i$, and where $H(\theta, \mu, X) = \|\text{P}_{A_a^\theta} G(\theta, X)\|^2 \nabla_\mu \ell(X, \mu)$ is the stochastic gradient for the IS calibration subproblem. Crucially, $G_k$ is sampled from $\mathbb{P}_{\mu_k}$ (IS distribution) while $H_k$ is sampled from $\mathbb{P}$ (nominal distribution), making the two gradient streams **independent** conditionally.

---

## 2. Main Results (key theorems with LaTeX statements)

### Lemma 2.2 (Projected Gradient CLT -- Baseline without IS)

For any iterate sequence $\bar{\theta}_n$ achieving the Duchi (2021) optimal CLT, the projected gradient satisfies:

$$\sqrt{n} \; \text{P}_{A^\star_a} \nabla f(\bar{\theta}_n) \overset{d}{\to} \mathcal{N}\left(0, \text{Var}_{X \sim \mathbb{P}}\left[\text{P}_{A^\star_a} G(\theta^\star, X)\right]\right)$$

This establishes the variance **without IS** as the baseline. The paper's goal is to replace $\text{Var}[\cdot]$ under $\mathbb{P}$ with the smaller $\text{Var}[\cdot]$ under $\mathbb{P}_{\mu^\star}$.

### Theorem 4.1 (Almost Sure Convergence)

Under Assumptions 2.1 (convexity, differentiability, unique minimizer, bounded $\Theta$), 3.2 (IS regularity), and 4.1 (gradient regularity, strong convexity-like growth):

$$\begin{bmatrix}\theta_n \\ \mu_n\end{bmatrix} \overset{\text{a.s.}}{\to} \begin{bmatrix}\theta^\star \\ \mu^\star\end{bmatrix}$$

This holds **despite the joint problem over $(\theta, \mu)$ lacking convexity** and **without time-scale separation or nested optimization**.

### Proposition 4.3 (Finite-Time Active Constraint Identification)

There exists a random finite $N$ such that for all $n \geq N$:
- $A_a^\star \theta_n = b_a^\star$, $A_i^\star \theta_n < b_i^\star$ (decision constraints correctly identified)
- $C_a^\star \mu_n = d_a^\star$, $C_i^\star \mu_n < d_i^\star$ (IS constraints correctly identified)

This finite-time identification is a unique property of NDA (not shared by projected SGD) and is **essential** for the asymptotic normality proof.

### Theorem 4.4 (Asymptotic Optimality I -- Joint CLT)

$$\sqrt{n}\begin{bmatrix}\bar{\theta}_n - \theta^\star \\ \bar{\mu}_n - \mu^\star\end{bmatrix} \overset{d}{\to} \mathcal{N}\left(\begin{bmatrix}0 \\ 0\end{bmatrix}, \begin{bmatrix}\Sigma_G^\star & 0 \\ 0 & \Sigma_H^\star\end{bmatrix}\right)$$

where:
- $\Sigma_G^\star = Q^\dagger \text{Var}_{X^{(\mu^\star)} \sim \mathbb{P}_{\mu^\star}}[G_{\mu^\star}(\theta^\star, X^{(\mu^\star)})] Q^\dagger$, with $Q = \text{P}_{A^\star_a} \nabla^2 f(\theta^\star) \text{P}_{A^\star_a}$
- $\Sigma_H^\star = R^\dagger \text{Var}_{X \sim \mathbb{P}}[H(\theta^\star, \mu^\star, X)] R^\dagger$, with $R = \text{P}_{C^\star_a} \nabla^2_\mu v(\theta^\star, \mu^\star) \text{P}_{C^\star_a}$

The block-diagonal structure arises from independence of the two sampling streams.

### Corollary 4.5 (Asymptotic Optimality II -- Projected Gradient CLT)

$$\sqrt{n} \; \text{P}_{A^\star_a} \nabla f(\bar{\theta}_n) \overset{d}{\to} \mathcal{N}\left(0, \text{Var}_{X^{(\mu^\star)} \sim \mathbb{P}_{\mu^\star}}\left[\text{P}_{A^\star_a} G_{\mu^\star}(\theta^\star, X^{(\mu^\star)})\right]\right)$$

where $\mu^\star = \arg\min_{\mu \in \mathcal{M}} \text{Tr}(\text{Var}_{\mathbb{P}_\mu}[\text{P}_{A^\star_a} G_\mu(\theta^\star, X^{(\mu)})])$.

This **matches the performance of an oracle** that knows $\mu^\star$ a priori, resolving the circularity challenge in the asymptotic regime.

### Motivating Example: Normal Quantile Estimation (Example 3.3)

For estimating the $\alpha$-quantile of $\mathcal{N}(0,1)$ with $\alpha \ll 1$:
- Without IS: asymptotic variance $\sigma^2 \geq \sqrt{2\pi}\exp(\theta^{\star 2}/2) / (2(\theta^\star + 1/\theta^\star))$ -- **exponentially large** as $\alpha \to 0$.
- With exponential tilting IS at $\mu = \theta^\star$: $\sigma^2(\theta^\star) \leq 1/2$ -- **bounded independent of $\alpha$**.

The numerical experiments on this problem ($\alpha = 0.9999$, $\theta^\star \approx 3.72$) show the proposed method achieves **three orders of magnitude** variance reduction over baselines.

---

## 3. Proof Techniques

1. **Robbins-Siegmund supermartingale framework (Theorem 4.1).** Almost sure convergence uses the Robbins-Siegmund theorem. A Lyapunov quantity $R_{n+1}$ is constructed from NDA optimality conditions:

$$R_{n+1} = \left\langle \sum_{k=0}^n \alpha_{k+1} \begin{bmatrix}G_k \\ H_k\end{bmatrix} + \begin{bmatrix}\theta_{n+1} \\ \mu_{n+1}\end{bmatrix}, \begin{bmatrix}\theta^\star - \theta_{n+1} \\ \mu^\star - \mu_{n+1}\end{bmatrix}\right\rangle + \frac{1}{2}\left\|\begin{bmatrix}\theta_{n+1} - \theta^\star \\ \mu_{n+1} - \mu^\star\end{bmatrix}\right\|^2$$

   The **key technical novelty** is handling the cross-coupling term $\langle \nabla_\mu v(\theta_n, \mu_n) - \nabla_\mu v(\theta^\star, \mu_n), \mu_n - \mu^\star \rangle$, which is **not guaranteed nonnegative** when $\theta_n \neq \theta^\star$. Standard SA proofs require nonnegativity here. The paper resolves this by (a) decomposing into a nonnegative part $\langle \nabla_\mu v(\theta^\star, \mu_n), \mu_n - \mu^\star \rangle \geq 0$ and (b) showing the perturbation $\|\nabla_\mu v(\theta_n, \mu_n) - \nabla_\mu v(\theta^\star, \mu_n)\| \leq c_3\|\theta_n - \theta^\star\|^2$ is summable after finite-time constraint identification. This avoids time-scale separation entirely.

2. **Finite-time active constraint identification via NDA geometry (Proposition 4.3).** Leveraging Duchi (2021, Lemma 4.2), the NDA structure with cumulative gradient sums forces iterates onto the active constraint manifold permanently once close enough, provided KKT multipliers are strictly positive (strict complementarity). This is **not possible** with projected SGD, which bounces off constraints infinitely often.

3. **Martingale CLT for coupled iterates (Theorem 4.4).** After constraint identification at random time $N$, the iteration on the active manifold reduces to a Polyak-Ruppert-type recursion:

$$\Delta_{n+1} = \Delta_n - \alpha_{n+1} P H P \Delta_n - \alpha_{n+1} P(\xi_n + \zeta_n) + \epsilon_n$$

   where $\xi_n$ is the martingale noise, $\zeta_n$ is the second-order remainder (controlled by $\|\Delta_n\|^2$), and $\epsilon_n = 0$ for $n \geq N$. The martingale CLT (conditional Lindeberg condition) is then applied. Block-diagonality of the asymptotic covariance follows from the **independence** of the two sampling streams ($X^{(\mu_k)} \sim \mathbb{P}_{\mu_k}$ and $X \sim \mathbb{P}$).

4. **Log-convexity ensures IS subproblem convexity (Lemma 3.4).** The IS variance objective $v(\theta, \mu) = \mathbb{E}_{\mathbb{P}}[\|P_{A_a^\theta} G(\theta, X)\|^2 \ell(X, \mu)]$ is convex in $\mu$ because log-convexity implies convexity of $\ell(x, \mu)$ (via Young's inequality), and integration preserves convexity. This is what makes single-loop updates viable.

---

## 4. Connection to Our Paper

Our paper "Where to Search: GAINS" allocates computational budget across diffusion timesteps -- a resource allocation problem under budget constraints. Lam et al.'s work on optimal importance sampling connects to this via the broader theme of optimally allocating simulation resources.

### What We Can Borrow (OR framing, resource allocation theory)

- **Variance-as-allocation-objective framing.** Lam et al. formalize IS as minimizing $\text{Tr}(\text{Var}[\cdot])$ subject to constraints on the sampling parameter -- directly analogous to GAINS minimizing search error by allocating NFE budget across timesteps. Their trace-of-variance criterion provides a clean objective template. In both papers, higher variance at a particular "location" (timestep / sampling direction) calls for more resources there.

- **Circularity resolution as a design principle.** The circularity between $\theta^\star$ and $\mu^\star$ parallels our setting: optimal per-timestep budget depends on score variance, which can only be estimated with sufficient budget. Lam et al.'s resolution -- a single-loop joint update that achieves oracle-level performance without knowing the answer in advance -- validates the principle that adaptive algorithms can resolve such chicken-and-egg problems without penalty (at least asymptotically).

- **Constrained optimization with active set structure.** Their treatment of linear constraints $A\theta \leq b$, with finite-time identification of which constraints bind, maps directly to budget allocation where some timesteps may receive zero allocation (budget constraint active at boundary). The NDA framework's ability to lock onto the correct active set could inform allocation schemes that must decide which timesteps to "turn off."

- **Asymptotic optimality as a benchmark.** Their "price of adaptivity is zero" result (adaptive scheme matches oracle) provides a template for proving analogous results in our context -- e.g., that online budget adaptation matches the best fixed allocation in hindsight.

### How We Differ

- **Discrete vs. continuous allocation.** GAINS allocates integer NFE counts across a finite set of timesteps (combinatorial), while Lam et al. work with continuous IS parameters from exponential families. Our problem lacks the smoothness and convexity that enables their NDA-based analysis.

- **Non-asymptotic vs. asymptotic regime.** GAINS operates with finite, often small computational budgets where asymptotic guarantees may not apply. Lam et al.'s results are fundamentally asymptotic ($n \to \infty$); their numerical illustrations require $10^5$ iterations before the variance advantage manifests. We need finite-sample allocation rules.

- **Sequential dependence vs. i.i.d. sampling.** Diffusion timesteps have sequential dependence: the noise choice at step $t$ affects the input to step $t+1$. Lam et al. assume i.i.d. samples at each iteration. This sequential structure fundamentally changes the allocation problem.

- **Non-convex objectives.** Lam et al. require convexity of $f(\theta)$ (Assumption 2.1(i)), while diffusion-based search objectives are non-convex. The global convergence guarantees do not transfer.

- **Quality metric.** Lam et al. minimize asymptotic variance of a gradient estimator (statistical efficiency). We maximize reward/quality of generated samples (optimization quality). These are fundamentally different objectives, though both involve "allocating effort where it matters most."

---

## 5. Key Quotes (2-3)

> "The decision variable and the importance sampling distribution are mutually dependent, creating a circular optimization structure. This interdependence complicates both convergence analysis and variance control." (Abstract)

> "Without a good decision, we cannot identify an effective IS distribution; and without an effective IS distribution, sampling is too inefficient to reliably identify a good decision." (Section 1, Introduction)

> "The method is globally convergent and achieves the minimal asymptotic variance among stochastic gradient schemes, which moreover matches the performance of an oracle sampler adapted to the optimal solution and thus effectively resolves the circular optimization challenge." (Abstract)

---

## 6. BibTeX

```bibtex
@article{aolaritei2025stochastic,
  title={Stochastic Optimization with Optimal Importance Sampling},
  author={Aolaritei, Liviu and Van Parys, Bart P. G. and Lam, Henry and Jordan, Michael I.},
  journal={arXiv preprint arXiv:2504.03560},
  year={2025}
}
```
