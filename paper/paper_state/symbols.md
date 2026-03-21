# Symbol Registry: Where to Search (GAINS)

**Last Updated**: 2026-03-16

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
| $s^{(k)}$ | Verifier score of $k$-th candidate | Eq 4 | $v(\hat{x}_0^{(k)}, c)$ |
| $K$ | Number of candidates per step | Sec 3.1 | Per-step budget |

## Theoretical Analysis

| Symbol | Meaning | Introduced | Context |
|--------|---------|------------|---------|
| $S_t$ | Score random variable at step $t$ | Eq 5 | $S_t = v(D_\theta(F_\theta(x_t, t, c, \varepsilon_t)))$ |
| $\mathcal{D}_t(c, x_t)$ | Distribution of $S_t$ | Sec 3.3 | Per-step score distribution |
| $\mu_t$ | $\mathbb{E}[S_t]$ | Sec 3.3 | Mean score |
| $\sigma_t^2$ | $\text{Var}(S_t)$ | Sec 3.3 | Score variance (sensitivity) |
| $M_t^{(K)}$ | Max order statistic | Sec 3.3 | $\max_{k \leq K} S_t^{(k)}$ |
| $G_t(K)$ | Expected gain from $K$ candidates | Eq 6 | $\mathbb{E}[M_t^{(K)}] - \mu_t$ |
| $a_K$ | Universal gain sequence | Eq 7 | $\mathbb{E}[\max(Z_1, \ldots, Z_K)]$ |
| $Z$ | Standardized score variable | Eq 7 | Location-scale family |
| $\lambda^*$ | Lagrange multiplier (shadow price) | Prop 1 | Budget constraint dual |
| $\boldsymbol{\sigma}$ | Variance profile | Prop 2 | $(\sigma_1, \ldots, \sigma_T)$ |
| $V^*(\boldsymbol{\sigma})$ | Oracle allocation value | Prop 2 | Convex in $\boldsymbol{\sigma}$ |

## Online Controller

| Symbol | Meaning | Introduced | Context |
|--------|---------|------------|---------|
| $g_t^{(j)}$ | Incremental gain at iteration $j$ | Sec 3.2 | $s_t^{(j)} - s_t^{(j-1)}$ |
| $\text{Var}_t^{(j)}$ | Per-iteration candidate variance | Sec 3.2 | Sensitivity signal |
| $\tilde{g}_t^{(j)}$ | Windowed mean gain | Alg 1 | Mean of recent $\mathcal{H}$ |
| $\mathcal{H}_g$ | Historical mean gains | Alg 1 | Across completed steps |
| $\mathcal{H}_\sigma$ | Historical variances | Alg 1 | Across completed steps |
| $\beta_g$ | Gain threshold coefficient | Alg 1 | Hyperparameter |
| $\beta_\sigma$ | Variance threshold coefficient | Alg 1 | Hyperparameter |
| $\delta$ | Slack window | Alg 1 | Over/under-spend margin |
| $W_g$ | Gain history window size | Alg 1 | Recent gains to average |
| $R$ | Remaining budget | Alg 1 | $R \leftarrow B$ initially |

## MDP Formulation (Appendix A)

| Symbol | Meaning | Introduced | Context |
|--------|---------|------------|---------|
| $s = (c, t, x_t, \mathcal{H}_t, R)$ | MDP state | App A | Full state representation |
| $\mathcal{A}(s)$ | Action space | App A | Search ∪ Verify ∪ Move |
| $\pi$ | Policy | App A | $\pi: \mathcal{S} \to \mathcal{A}$ |
