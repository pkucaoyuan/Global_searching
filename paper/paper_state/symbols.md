# Symbol Registry: Where to Search (GAINS)

**Last Updated**: 2026-03-23

---

## Core Framework

| Symbol | Meaning | Introduced | Context |
|--------|---------|------------|---------|
| $T$ | Number of denoising timesteps | Sec 3.1 | Trajectory length |
| $t$ | Current timestep index | Sec 3.1 | $t \in \{1, \ldots, T\}$ |
| $x_t$ | Latent at timestep $t$ | Eq 1 | State variable |
| $x_0$ | Final generated sample | Sec 3.1 | Output |
| $\varepsilon_t$ | Noise variable at step $t$ | Eq 1 | Source of stochasticity |
| $c$ | Conditioning signal | Eq 1 | Prompt / class label |
| $F_\theta$ | Base sampler transition | Eq 1 | $x_{t-1} = F_\theta(x_t, t, c, \varepsilon_t)$ |
| $D_\theta$ | Denoising prediction (Tweedie) | Eq 3 | $\hat{x}_0 = D_\theta(x_{t-1}, t-1, c)$ |
| $v(x_0, c)$ | Verifier score function | Sec 3.1 | Quality metric |
| $B$ | Total NFE budget | Sec 3.1 | Constraint |
| $K_t$ | Per-step iteration budget | Sec 3.2 | From offline profiling |
| $\hat{K}_t$ | Realized per-step iterations | Alg 1 | After online adjustment |

## Local Search

| Symbol | Meaning | Introduced | Context |
|--------|---------|------------|---------|
| $\mathcal{L}_t$ | Local search operator at step $t$ | Sec 3.1 | Black-box abstraction |
| $s^{(k)}$ | Verifier score of the $k$-th candidate | Eq 4 | $v(\hat{x}_0^{(k)}, c)$ |
| $K$ | Number of candidates per step | Sec 3.1 | Per-step budget |

## Theoretical Analysis

| Symbol | Meaning | Introduced | Context |
|--------|---------|------------|---------|
| $S_t$ | Score random variable at step $t$ | Eq 5 | $S_t = v(D_\theta(F_\theta(x_t, t, c, \varepsilon_t)))$ |
| $\mathcal{D}_t(c, x_t)$ | Distribution of $S_t$ | Sec 3.3 | Per-step score distribution |
| $\mu_t$ | $\Psi_t(\bar{X}_{t-h};c)$ | Sec 4.3 | Baseline score at the deterministic Euler point |
| $\sigma_t$ | $g_t\sqrt{h}\,\|\nabla\Psi_t(\bar{X}_{t-h};c)\|$ | Sec 4.3 | Per-step sensitivity parameter |
| $M_t^{(K)}$ | Max order statistic | Sec 3.3 | $\max_{k \le K} S_t^{(k)}$ |
| $G_t(K)$ | Expected gain from $K$ candidates | Eq 6 | $\mathbb{E}[M_t^{(K)}] - \mu_t$ |
| $a_K$ | Universal gain sequence | Eq 7 | $\mathbb{E}[\max(Z_1, \ldots, Z_K)]$ |
| $\Delta a_K$ | Discrete exhaustion sequence | Sec 4.4 | $a_{K+1} - a_K$ |
| $Z$ | Standardized score variable | Eq 7 | Location-scale family |
| $\lambda^*$ | Lagrange multiplier (shadow price) | Prop 1 | Budget constraint dual |
| $\boldsymbol{\sigma}$ | Sensitivity profile | Prop 2 | $(\sigma_1, \ldots, \sigma_T)$ |
| $V^*(\boldsymbol{\sigma})$ | Oracle allocation value | Prop 2 | Convex in $\boldsymbol{\sigma}$ |

## Online Controller

| Symbol | Meaning | Introduced | Context |
|--------|---------|------------|---------|
| $g_t^{(j)}$ | Incremental gain at iteration $j$ | Sec 3.2 | $s_t^{(j)} - s_t^{(j-1)}$ |
| $\text{Var}_t^{(j)}$ | Empirical variance of seen candidate scores | Sec 3.2 | Proxy for low sensitivity, not the same object as $\sigma_t$ |
| $\tilde{g}_t^{(j)}$ | Windowed mean gain | Alg 1 | Mean of recent $\mathcal{H}$ |
| $\mathcal{H}_g$ | Historical mean gains | Alg 1 | Across completed steps |
| $\mathcal{H}_\sigma$ | Historical empirical variances | Alg 1 | Across completed steps |
| $\beta_g$ | Gain threshold coefficient | Alg 1 | Hyperparameter |
| $\beta_\sigma$ | Variance threshold coefficient | Alg 1 | Hyperparameter |
| $\delta$ | Slack window | Alg 1 | Over/under-spend margin |
| $W_g$ | Gain history window size | Alg 1 | Recent gains to average |
| $R$ | Remaining budget | Alg 1 | $R \leftarrow B$ initially |

## General Local Search (Sec 4.5)

| Symbol | Meaning | Introduced | Context |
|--------|---------|------------|---------|
| $G_t^{\mathcal{L}}(K)$ | Expected gain from $K$ iterations of operator $\mathcal{L}$ | Eq (4.5) | Generalizes $G_t(K)$ |
| $\phi_K^{\mathcal{L}}$ | Operator-specific gain sequence | Eq (4.5) | $G_t = \sigma_t \phi_K + O_{K,d,L_t}(g_t^2 h)$ |
| $\phi_K^{\mathrm{RS}}$ | Random search gain: $a_K$ | Ex 4.7 | Recovers original |
| $\phi_K^{(\epsilon)}$ | $\epsilon$-greedy gain: $\ge a_{\lceil\epsilon K\rceil}$ | Ex 4.8 | Lower bound |
| $c_d(\lambda)$ | Uniform-ball local-perturbation drift constant | Eq (4.5) | $\frac{\lambda\sqrt d}{4\sqrt{\pi}}\frac{\Gamma(d/2)}{\Gamma((d+1)/2)}$ |
| $\phi_K^{(\mathrm{LP})}$ | Local perturbation gain: $c_d(\lambda)K$ | Ex 4.9 | Exact in the linearized regime |
| $V_{\mathcal{L}}^*(\boldsymbol{\sigma})$ | Generalized oracle value | Cor (4.5) | Extends $V^*$ |
| $K^*$ | RS/LP crossover scale | Prop (4.5) | $\tilde{\Theta}(c_d(\lambda)^{-1})=\tilde{\Theta}(\lambda^{-1})$ |

## MDP Formulation (Appendix A)

| Symbol | Meaning | Introduced | Context |
|--------|---------|------------|---------|
| $s = (c, t, x_t, \mathcal{H}_t, R)$ | MDP state | App A | Full state representation |
| $\mathcal{A}(s)$ | Action space | App A | Search / Verify / Move |
| $\pi$ | Policy | App A | $\pi: \mathcal{S} \to \mathcal{A}$ |
