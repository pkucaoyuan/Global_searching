# Jia, Zhou (2024) --- "Reinforcement Learning for Jump-Diffusions, with Financial Applications"

**ArXiv:** 2405.16449
**Authors:** Xuefeng Gao, Lingfei Li, Xun Yu Zhou
**Affiliations:** CUHK (Systems Engineering and Engineering Management), Columbia University (IEOR and Data Science Institute)
**Keywords:** Reinforcement learning, continuous time, jump-diffusions, exploratory formulation, well-posedness, Hamiltonian, martingale, q-learning

---

## 1. Setting & Model

The paper studies continuous-time reinforcement learning (RL) for stochastic control problems where system dynamics are governed by **jump-diffusion processes** (Levy SDEs). This extends the continuous-time RL framework of Wang & Zhou (2020) and the q-learning trilogy (Jia & Zhou 2022a,b,c), which only considered pure diffusion processes.

### Controlled State Dynamics

The state evolves according to a Levy SDE:

$$dX_s^a = b(s, X_{s-}^a, a_s)\,ds + \sigma(s, X_{s-}^a, a_s)\,dW_s + \int_{\mathbb{R}^\ell} \gamma(s, X_{s-}^a, a_s, z)\,\widetilde{N}(ds, dz),\quad s \in [0, T],$$

where $W$ is a standard Brownian motion in $\mathbb{R}^m$, $\widetilde{N}(ds, dz)$ is a compensated Poisson random measure associated with $\ell$ independent 1D Levy processes with Levy measure $\nu$, and the control $a_s \in \mathcal{A} \subseteq \mathbb{R}^n$ affects all three components: drift $b$, diffusion $\sigma$, and jump coefficient $\gamma$. The objective is to maximize:

$$\mathbb{E}\left[\int_t^T e^{-\beta(s-t)} r(s, X_s^a, a_s)\,ds + e^{-\beta(T-t)} h(X_T^a) \;\Big|\; X_t^a = x\right]$$

### The Exploratory Challenge with Jumps

In RL, agents explore by randomizing actions via stochastic policies $\boldsymbol{\pi}(\cdot|t,x)$. The theoretical analysis requires formulating an "exploratory SDE" representing the averaged dynamics over randomized actions. For pure diffusions, this is derived by a law-of-large-numbers argument on the first two moments. **This fails for jump-diffusions because they are not uniquely determined by their first two moments.**

The paper overcomes this by analyzing the **infinitesimal generator** of the grid sample state process (where actions are sampled on a discrete time grid and held piecewise constant). The key result is that the exploratory generator equals the policy-weighted average of classical generators:

$$\mathcal{L}^{\boldsymbol{\pi}} f(t,x) = \int_{\mathcal{A}} \mathcal{L}^a f(t,x)\,\boldsymbol{\pi}(a|t,x)\,da$$

This motivates the **exploratory Levy SDE** formulated on an extended probability space with Poisson random measure $N'(dt, dz, du)$ on $[0,T] \times \mathbb{R} \times [0,1]^n$:

$$d\tilde{X}_s^{\boldsymbol{\pi}} = \tilde{b}(s, \tilde{X}_{s-}^{\boldsymbol{\pi}}, \boldsymbol{\pi})\,ds + \tilde{\sigma}(s, \tilde{X}_{s-}^{\boldsymbol{\pi}}, \boldsymbol{\pi})\,dW_s + \int_{\mathbb{R} \times [0,1]^n} \gamma(s, \tilde{X}_{s-}^{\boldsymbol{\pi}}, G^{\boldsymbol{\pi}}(s, \tilde{X}_{s-}^{\boldsymbol{\pi}}, u), z)\,\widetilde{N}'(ds, dz, du)$$

where $\tilde{b}$ and $\tilde{\sigma}$ are policy-averaged drift and diffusion coefficients, and crucially, the jump component **cannot** be simplified to a policy average of $\gamma$ -- this is a main distinctive feature of the jump-diffusion setting.

### Entropy-Regularized Objective

Following the exploratory formulation, the value function under a stochastic policy is:

$$J(t, x; \boldsymbol{\pi}) = \mathbb{E}_{t,x}\left[\int_t^T e^{-\beta(s-t)} \int_{\mathcal{A}} \big(r(s, \tilde{X}_s^{\boldsymbol{\pi}}, a) - \theta \log \boldsymbol{\pi}(a|s, \tilde{X}_{s-}^{\boldsymbol{\pi}})\big) \boldsymbol{\pi}(a|s, \tilde{X}_{s-}^{\boldsymbol{\pi}})\,da\,ds + e^{-\beta(T-t)} h(\tilde{X}_T^{\boldsymbol{\pi}})\right]$$

where $\theta > 0$ is the temperature parameter controlling exploration intensity.

### Applications

The paper studies two financial applications:

1. **Mean-variance portfolio selection** with stock price modeled as a jump-diffusion. This is an LQ problem where the optimal stochastic policy is Gaussian and the value function is quadratic, **regardless** of whether jumps are present -- both RL algorithms and parameterizations are jump-invariant due to the LQ structure.

2. **Mean-variance hedging of European options.** This is a non-LQ problem (random target, higher-dimensional state) where the paper derives analytical representations for the optimal Gaussian policy and value function, and develops an actor-critic algorithm using Gaussian process regression for the critic. Empirical results on S&P 500 data show that the RL policy significantly outperforms MLE-based plug-in methods in terms of mean squared hedging error.

---

## 2. Main Results (key theorems in LaTeX)

### Well-posedness of the Exploratory Levy SDE (Proposition 2)

Under Assumptions 1-2 (local Lipschitz continuity and linear growth of $b, \sigma, \gamma$, plus Lipschitz continuity of $\gamma$ in $a$ and of $G^{\boldsymbol{\pi}}$ in $x$), for any admissible policy $\boldsymbol{\pi}$:

$$\mathbb{E}_{t,x}\left[\sup_{t \le s \le T} |\tilde{X}_s^{\boldsymbol{\pi}}|^p\right] \le C_p(1 + |x|^p), \quad \forall p \ge 2$$

### Optimal Stochastic Policy is Gibbs (Lemma 4)

The optimal stochastic policy is the Gibbs/Boltzmann distribution:

$$\boldsymbol{\pi}^*(a|t,x) \propto \exp\left(\frac{1}{\theta} H(t,x,a, \partial_x J^*, \partial_x^2 J^*, J^*)\right)$$

where the Hamiltonian includes the jump integral:

$$H = r(t,x,a) + b \circ \partial_x V + \tfrac{1}{2}\sigma^2 \circ \partial_x^2 V + \sum_{k=1}^\ell \int_{\mathbb{R}} \big(V(t, x+\gamma_k) - V - \gamma_k \circ \partial_x V\big)\,\nu_k(dz)$$

### Exploratory HJB (PIDE)

$$\partial_t J^*(t,x) + \theta \log\left(\int_{\mathcal{A}} \exp\left(\frac{1}{\theta} H(t,x,a,\partial_x J^*, \partial_x^2 J^*, J^*)\right) da\right) - \beta J^*(t,x) = 0$$

### Policy Improvement (Theorem 3)

For any $\boldsymbol{\pi} \in \boldsymbol{\Pi}$, define $\boldsymbol{\pi}'(\cdot | t, x) \propto \exp\big(\frac{1}{\theta} q(t,x, \cdot; \boldsymbol{\pi})\big)$. If $\boldsymbol{\pi}' \in \boldsymbol{\Pi}$, then $J(t,x; \boldsymbol{\pi}') \geq J(t,x; \boldsymbol{\pi})$. If the Gibbs map has a fixed point $\boldsymbol{\pi}^*$, then $\boldsymbol{\pi}^*$ is optimal.

### Martingale Characterization of the q-Function (Theorem 4)

$\hat{q}(t,x,a) = q(t,x,a; \boldsymbol{\pi})$ for all $(t,x,a)$ **if and only if** for any grid $\mathbb{S}$ of $[t,T]$, the process

$$e^{-\beta s} J(s, X_s^{\boldsymbol{\pi}, \mathbb{S}}; \boldsymbol{\pi}) + \int_t^s e^{-\beta \tau} \big(r(\tau, X_\tau^{\boldsymbol{\pi}, \mathbb{S}}, a_\tau^{\boldsymbol{\pi}, \mathbb{S}}) - \hat{q}(\tau, X_\tau^{\boldsymbol{\pi}, \mathbb{S}}, a_\tau^{\boldsymbol{\pi}, \mathbb{S}})\big)\,d\tau$$

is a $\mathcal{G}^{\mathbb{S}}$-martingale. This characterization is **identical in form** to the pure diffusion case -- the Hamiltonian absorbs the difference.

### Core Practical Result: Algorithmic Invariance

Using Ito's formula for jump-diffusions, the q-function values along data trajectories can be computed via temporal difference of the value function. Because Ito's formula absorbs jump terms into $dJ$, the resulting RL algorithms (both offline-episodic and online-incremental q-learning, Algorithms 1-2) are **identical** to those developed for pure diffusions. One does not need to know a priori whether data come from a diffusion or a jump-diffusion.

### Convergence of Grid Sample Value Functions (Theorem 7)

Under additional smoothness assumptions (Assumption 3), for any admissible policy $\boldsymbol{\pi}$:

$$|J^{\boldsymbol{\pi}, \mathbb{S}}(0, x) - J^{\boldsymbol{\pi}}(0, x)| \leq C|\mathbb{S}|$$

This extends Jia (2025) from diffusions to jump-diffusions, establishing first-order convergence in mesh size.

### Jumps Can Affect Parameterization (Section 5.4)

For the MV portfolio selection problem, the LQ structure ensures that both algorithms and parameterizations are jump-invariant. But this is an exception: the paper constructs a counterexample with $\gamma(a,z) = a^2$ where the optimal stochastic policy **either does not exist or is non-Gaussian** when jumps are present, whereas it is Gaussian without jumps. This demonstrates that while algorithms are universal, parameterizations must generally respond to jumps.

---

## 3. Proof Techniques

### Infinitesimal Generator Approach

Rather than the moment-based LLN argument used for pure diffusions (which fails because jump-diffusions are not determined by first two moments), the paper directly computes the infinitesimal behavior:

$$\lim_{s \to 0} \frac{\mathbb{E}_{t,x}[f(t+s, X_{t+s}^{\boldsymbol{\pi}, \mathbb{S}})] - f(t,x)}{s} = \int_{\mathcal{A}} \mathcal{L}^a f(t,x)\,\boldsymbol{\pi}(a|t,x)\,da$$

The jump integral part requires Fubini's theorem justified by the bound $|\int_{\mathbb{R}}(f(t,x+\gamma_k)-f-\gamma_k \circ \partial_x f)\nu_k(dz)| \le C(1+|x|^2)$, which is independent of $a$.

### Extended Poisson Random Measure

To represent exploratory jump dynamics, the authors introduce a Poisson random measure $N'_k(dt, dz_k, du)$ on $[0,T] \times \mathbb{R} \times [0,1]^n$ with intensity $\nu_k(dz)\,du$. This encodes both jump randomness ($z$) and action randomness ($u$) in a single measure. The approach is inspired by Kushner (2000) on relaxed control for jump-diffusions, but differs in considering stochastic **feedback** policies, which creates additional technical issues for well-posedness.

### Well-posedness via Localization

Existence and uniqueness of the exploratory SDE are established by verifying local Lipschitz continuity and linear growth of the exploratory coefficients (Lemmas 2-4), then applying classical results (Kunita 2004, Rong 2006). A key technical contribution is Assumption 2, requiring $G^{\boldsymbol{\pi}}(t,x,u)$ to be locally Lipschitz in $x$ in an $L^p$-integrated sense. For Gaussian policies $\boldsymbol{\pi} \sim \mathcal{N}(\mu(t,x), A(t,x)A(t,x)^\top)$, this reduces to local Lipschitz continuity of $\mu$ and $A$.

### Martingale Arguments for q-Function Characterization

The proofs apply Ito's formula for jump-diffusions to the value function over grid sample state processes. The main subtlety is verifying that stochastic integrals w.r.t. both $dW$ and $\widetilde{N}(dt,dz)$ are true martingales (not just local martingales). For infinite jump activity ($\int_{\mathbb{R}}\nu_k(dz) = \infty$), the integral is split into small jumps ($|z| \le 1$, handled via mean-value theorem and bounded $\gamma$) and large jumps ($|z| > 1$, finite measure).

The converse direction (martingale implies $\hat{q} = q$) uses a continuity-based contradiction argument: if $f(t^*,x^*,a^*) := q - \hat{q} \ne 0$ at some point, the continuous-path integral $\int f\,d\tau$ cannot be a.s. zero due to the full-support property of admissible policies.

### Convergence Rate via Telescoping

Theorem 7 decomposes $J^{\boldsymbol{\pi},\mathbb{S}}(0,x) - J^{\boldsymbol{\pi}}(0,x)$ over grid intervals. Within each interval, the error is $\mathbb{E}[e_j] = \mathbb{E}[\int_{t_{j-1}}^{t_j}(\mathcal{L}^{a_j}\phi(\tau,X_\tau) - \mathcal{L}^{a_j}\phi(t_{j-1},X_{t_{j-1}}))\,d\tau]$, which vanishes at first order because $\mathbb{E}[\mathcal{L}^{a_j}\phi(t_{j-1},X_{t_{j-1}})] = \mathcal{L}^{\boldsymbol{\pi}}\phi(t_{j-1},X_{t_{j-1}}) = 0$ by the PIDE. The jump-related error terms involving the integral operator $I_k^\phi$ are controlled via Lemma 8 on differentiability, yielding $|\mathbb{E}[e_j]| \le C(t_j - t_{j-1})^2$ that sums to $O(|\mathbb{S}|)$.

---

## 4. Connection to Our Paper

GAINS formulates noise trajectory search as an MDP (Appendix A) with budget-constrained sequential decisions. Zhou's continuous-time RL connects to our theoretical foundations.

### What We Can Borrow

- **Entropy-regularized exploration framework.** Zhou et al.'s entropy-regularized objective provides a principled continuous-time analog of our discrete search MDP. The temperature $\theta$ that controls exploration intensity directly parallels our budget-aware exploration: as remaining budget shrinks, one should reduce exploration. In Zhou's MV portfolio selection, the optimal policy variance decreases as $t \to T$, which is analogous to our strategy of shifting from Search to Verify as budget depletes. The Gibbs policy $\boldsymbol{\pi}^*(a|t,x) \propto \exp(q^*/\theta)$ provides theoretical justification for temperature-annealed sampling of noise candidates.

- **Martingale characterization for TD learning.** The key insight that temporal difference algorithms are invariant to jump presence (Theorems 4-6) suggests that any value-function learning algorithm applied to our search MDP would remain valid even if the score landscape has discontinuous structure across timesteps (analogous to Levy jumps). The martingale condition could inform online value function learning in our noise trajectory search.

- **Grid sample convergence (Theorem 7).** The $O(|\mathbb{S}|)$ convergence rate of discretized value functions to the continuous limit directly applies: our MDP operates on a discrete diffusion timestep grid, and this result bounds the approximation error from time discretization. This justifies working at coarse timestep granularity with controlled error.

- **Algorithmic robustness to model misspecification.** The paper's central finding -- RL algorithms are invariant to whether jumps are present -- provides a powerful robustness guarantee. For GAINS, this suggests that the same allocation/search algorithm could work regardless of the score distribution family (smooth Gaussian vs. heavy-tailed), without needing to characterize it in advance.

### How We Differ

- **Discrete vs. continuous action/state.** GAINS has a discrete action space (Search/Verify/Move among $K$ candidate positions) on a finite timestep grid. Zhou et al. work in continuous time with continuous $\mathcal{A} \subseteq \mathbb{R}^n$ where optimal policies are smooth densities (Gaussian). Our combinatorial structure (candidate sets, integer budgets) is absent from their framework.

- **Budget constraint vs. time-discounting.** GAINS has an explicit budget constraint $\sum_t K_t \le K$ that creates a Lagrangian water-filling structure (Proposition 1: equalize marginal gains $G_t'(K_t^*) = \lambda^*$). Zhou uses standard finite-horizon discounting $e^{-\beta(s-t)}$ without resource constraints. The water-filling / equalize-marginal-gains structure in GAINS does not arise in Zhou's formulation.

- **State dynamics and control.** In Zhou, the agent's actions directly affect future states through drift/diffusion/jump coefficients of the wealth/price process. In GAINS, the denoising trajectory is determined by the diffusion model, and the agent's "control" (which noise candidate to evaluate) only affects which trajectory branch is explored, not the underlying dynamics.

- **Structure exploitation vs. model-free.** Zhou emphasizes model-free algorithms that work without knowing the data-generating process. GAINS explicitly exploits the known structure of the diffusion model (score function, denoising schedule) and per-timestep variance profiles from profiling data. Zhou's MV hedging application uses problem-specific parametrizations (quadratic value function, Gaussian policy), which is more analogous to our approach.

- **Financial vs. generative models.** Zhou's applications (portfolio selection, option hedging) involve real-valued wealth processes where jumps model sudden market movements (earning reports, crashes). GAINS operates in the high-dimensional latent space of diffusion models where the relevant stochasticity is the score function evaluation noise and verifier reward signal.

---

## 5. Key Quotes

> "We can simply use the same policy evaluation and q-learning algorithms... originally developed for controlled diffusions, without needing to check a priori whether the underlying data come from a pure diffusion or a jump-diffusion." (Abstract)

> "It is tempting to think the exploratory jump coefficient $\tilde{\gamma}$ is similarly the average of $\gamma$ with respect to $\boldsymbol{\pi}$; but unfortunately it is generally not true. This in turn is one of the main distinctive features in studying RL for jump-diffusions." (Section 2.3)

> "Most important of all, in the resulting RL algorithms, the Hamiltonian (or equivalently the q-function) can be computed using temporal difference of the value function by virtue of the Ito lemma; as a result the algorithms are completely identical no matter whether or not there are jumps." (Section 1, Introduction)

> "A key insight from this research is that temporal-difference algorithms designed for diffusions can work seamlessly for jump-diffusions. However, unless using general neural networks, policy parameterization does need to respond to the presence of jumps if one is to take advantage of any special structure of an underlying problem." (Section 7, Conclusions)

> "Even though we can apply the same RL algorithms irrespective of the presence of jumps, the parametrization of the policy and value function may still depend on it, if we try to exploit certain special structure of the problem instead of using general neural networks for parameterization." (Section 1, Introduction)

---

## 6. BibTeX

```bibtex
@article{gao2024reinforcement,
  title={Reinforcement Learning for Jump-Diffusions, with Financial Applications},
  author={Gao, Xuefeng and Li, Lingfei and Zhou, Xun Yu},
  journal={arXiv preprint arXiv:2405.16449},
  year={2024},
  note={Revised August 2025}
}
```

### Key References from the Paper

```bibtex
@article{wang2020reinforcement,
  title={Reinforcement Learning in Continuous Time and Space: A Stochastic Control Approach},
  author={Wang, Haoran and Zariphopoulou, Thaleia and Zhou, Xun Yu},
  journal={Journal of Machine Learning Research},
  volume={21},
  number={198},
  pages={1--34},
  year={2020}
}

@article{jia2022q,
  title={q-Learning in Continuous Time},
  author={Jia, Yanwei and Zhou, Xun Yu},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={161},
  pages={1--61},
  year={2023}
}

@article{jia2022policy,
  title={Policy Evaluation and Temporal-Difference Learning in Continuous Time and Space: A Martingale Approach},
  author={Jia, Yanwei and Zhou, Xun Yu},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={154},
  pages={1--55},
  year={2022}
}

@article{jia2025accuracy,
  title={On the Accuracy of Discrete Approximations to Continuous-Time Reinforcement Learning},
  author={Jia, Yanwei and Zhou, Xun Yu},
  journal={arXiv preprint},
  year={2025}
}

@article{wang2020continuous,
  title={Continuous-Time Mean-Variance Portfolio Selection: A Reinforcement Learning Framework},
  author={Wang, Haoran and Zhou, Xun Yu},
  journal={Mathematical Finance},
  volume={30},
  number={4},
  pages={1273--1308},
  year={2020}
}

@book{tankov2004financial,
  title={Financial Modelling with Jump Processes},
  author={Tankov, Peter},
  year={2004},
  publisher={CRC Press}
}
```
