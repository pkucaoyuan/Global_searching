# Tang, Zhao (2024) -- "Score-based Diffusion Models via Stochastic Differential Equations -- A Technical Tutorial"

**Authors**: Wenpin Tang, Hanyang Zhao (Columbia IEOR)
**Venue**: *Statistics Surveys*, Vol. 19, pp. 28--64, 2025
**arXiv**: 2402.07487

---

## 1. Setting & Model

### What does this tutorial cover?

A rigorous mathematical tutorial on score-based diffusion models formulated through stochastic differential equations (SDEs). The paper provides a unified continuous-time treatment of diffusion generative models, organized around the two foundational pillars: **sampling** (reverse-time SDE/ODE) and **score matching** (learning the Stein score function $\nabla \log p(t, x)$). Written from an applied probability and statistics perspective at Columbia IEOR, with short self-contained proofs for all major results. Assumes familiarity with basic stochastic calculus.

**Paper structure**: (1) Time reversal formula; (2) Forward process examples (OU, VE, VP, sub-VP, CDPM); (3) Score matching techniques (implicit, sliced, denoising); (4) Probability flow ODE and consistency models; (5) Convergence analysis (TV and $W_2$); (6) Further score matching results including RL-based fine-tuning.

### General Forward SDE

Data is progressively corrupted by an Ito SDE:
$$dX_t = f(t, X_t)\,dt + g(t, X_t)\,dB_t, \quad X_0 \sim p_{\text{data}}(\cdot)$$

The **time reversal formula** (Anderson 1982, Theorem 2.1 in the paper) gives the backward process:
$$dY_t = \bigl[-f(T-t, Y_t) + a(T-t, Y_t)\nabla\log p(T-t, Y_t) + \nabla \cdot a(T-t, Y_t)\bigr]\,dt + g(T-t, Y_t)\,dB_t$$
where $a(t,x) = g(t,x)g(t,x)^\top$. When $g(t,x) = g(t)I$ (the practical case), this simplifies to a scalar diffusion coefficient with the score $\nabla \log p(t,x)$ as the only unknown quantity.

### Key Formulations

**Variance Exploding (VE) SDE** -- continuum limit of SMLD (Song et al. 2019):
$$f(t,x) = 0, \quad g(t) = \sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{t/T}\sqrt{\frac{2}{T}\log\frac{\sigma_{\max}}{\sigma_{\min}}}$$
Transition kernel: $p(t, \cdot; x) = \mathcal{N}\bigl(x,\; \sigma_{\min}^2((\sigma_{\max}/\sigma_{\min})^{2t/T} - 1)I\bigr)$.
Noise prior: $p_{\text{noise}} = \mathcal{N}(0, (\sigma_{\max}^2 - \sigma_{\min}^2)I)$.
Name origin: $\text{Var}(X_0) \ll \text{Var}(X_T)$ since $\sigma_{\min} \ll \sigma_{\max}$.

**Variance Preserving (VP) SDE** -- continuum limit of DDPM (Ho et al. 2020):
$$f(t,x) = -\tfrac{1}{2}\beta(t)x, \quad g(t) = \sqrt{\beta(t)}, \quad \beta(t) = \beta_{\min} + \tfrac{t}{T}(\beta_{\max} - \beta_{\min})$$
Transition kernel: $p(t, \cdot; x) = \mathcal{N}\bigl(e^{-\frac{1}{2}\int_0^t \beta(s)ds}\,x,\; (1 - e^{-\int_0^t \beta(s)ds})I\bigr)$.
Noise prior: $p_{\text{noise}} = \mathcal{N}(0, I)$.
Name origin: variance stays bounded $\approx 1$. Used in Stable Diffusion.

**Sub-Variance Preserving (sub-VP) SDE** (Song et al. 2020):
$$f(t,x) = -\tfrac{1}{2}\beta(t)x, \quad g(t) = \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)ds})}$$
Transition kernel: $p(t, \cdot; x) = \mathcal{N}\bigl(e^{-\frac{1}{2}\int_0^t \beta(s)ds}\,x,\; (1 - e^{-\int_0^t \beta(s)ds})^2 I\bigr)$.
Key property: $\text{Var}_{\text{sub-VP}}(X_t) \le \text{Var}_{\text{VP}}(X_t)$, hence the name.

The paper also discusses the **Contractive VP (CVP) SDE** which forces contraction on the backward process (sign flip: $f(t,x) = +\tfrac{1}{2}\beta(t)x$) to reduce sensitivity to score matching errors at the cost of noise approximation bias.

---

## 2. Main Results

### Score Matching

**Theorem 3.1 (Implicit Score Matching, Hyvarinen 2005).** The explicit score matching objective $\mathcal{J}_{\text{ESM}}(\theta) = \mathbb{E}_{p(t,\cdot)}|s_\theta(t,X) - \nabla\log p(t,X)|^2$ is equivalent (up to a $\theta$-independent constant) to the computable implicit objective:
$$\mathcal{J}_{\text{ISM}}(\theta) = \mathbb{E}_{p(t,\cdot)}\bigl[|s_\theta(t,X)|^2 + 2\nabla \cdot s_\theta(t,X)\bigr]$$

**Theorem 3.3 (Denoising Score Matching, Vincent 2011).** With Gaussian perturbation kernels $(X_t | X_0) \sim \mathcal{N}(\mu_t(X_0), \sigma_t^2 I)$, the DSM objective with optimal weighting $\lambda(t) = \sigma_t^2$ becomes:
$$\tilde{\mathcal{J}}_{\text{DSM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,T)}\mathbb{E}_{X_0, \varepsilon}\bigl[|\sigma_t s_\theta(t, \mu_t(X_0) + \sigma_t\varepsilon) + \varepsilon|^2\bigr]$$

### Convergence of Stochastic Samplers

**Blackbox assumption (Assumption 5.1).** $\mathbb{E}_{p(t,\cdot)}|s_\theta(t,X) - \nabla\log p(t,X)|^2 \le \varepsilon^2$ for all $t \in [0,T]$.

**Theorem 5.1 (Total Variation Bound, Chen et al. 2023).** Under the blackbox score assumption with $g(t)$ bounded away from zero:
$$d_{TV}(\mathcal{L}(Y_T), p_{\text{data}}) \le \underbrace{d_{TV}(p(T,\cdot), p_{\text{noise}})}_{\text{noise approx. error}} + \underbrace{\varepsilon\sqrt{T/2}}_{\text{score matching error}}$$
This is a **polynomial** bound in $T$, improving on the earlier exponential bound $D\varepsilon\exp(D'T)$ of De Bortoli et al. (2021).

**Corollary 5.2 (TV for VP).** For the VP backward process with $\mathbb{E}_{p_{\text{data}}}|x|^2 < \infty$:
$$d_{TV}(\mathcal{L}(Y_T), p_{\text{data}}) \le e^{-\frac{1}{2}\int_0^T \beta(s)ds}\sqrt{\mathbb{E}_{p_{\text{data}}}|x|^2/2} + \varepsilon\sqrt{T/2}$$

**Theorem 5.4 (Wasserstein-2 Bound, Tang & Zhao 2024).** Under the blackbox assumption plus Lipschitz score matching ($|s_\theta(t,x) - s_\theta(t,x')| \le L|x - x'|$) and one-sided growth of $f$:
$$W_2(p_{\text{data}}, \mathcal{L}(Y_T)) \le \sqrt{W_2^2(p(T,\cdot), p_{\text{noise}})e^{u(T)} + \frac{\varepsilon^2}{2h}\int_0^T g^2(t)e^{u(T)-u(T-t)}dt}$$
where $u(t) = \int_{T-t}^T(-2r_f(s) + (2L+2h)g^2(s))ds$ and $h > 0$ is a tunable hyperparameter.

**Theorem 5.5 (Wasserstein for VP under log-concavity).** If $p_{\text{data}}$ is $\kappa$-strongly log-concave with $\kappa > 1/2$, the score matching error term does **not** grow with $T$:
$$W_2^2(p_{\text{data}}, \mathcal{L}(Y_T)) \le e^{-\{\beta_{\min} + \beta_{\max}(2\min(\kappa,1) - 1 - 2h)\}T}(\mathbb{E}|x|^2 + o(1)d) + \frac{\varepsilon^2\beta_{\max}^2}{2h(2\min(\kappa,1) - 1 - 2h)}$$

### Discretization Error Summary (Table 2)

| Model | Metric | Sampler | Discretization Error |
|-------|--------|---------|---------------------|
| OU | TV | Euler | $\sqrt{Td}\sqrt{\delta}$ |
| OU | TV | PC | $\mathcal{O}_T(1)\sqrt{d}\sqrt{\delta}$ |
| OU | TV | EI | $\sqrt{Td^7}\sqrt{\delta^3}$ |
| VE | $W_2$ | Euler | $e^{\mathcal{O}(T)}\sqrt{d}\,\delta$ |
| VP | $W_2$ | Euler | $e^{\mathcal{O}(T^3)}\sqrt{d}\,\delta$ |
| CDPM | $W_2$ | Euler | $\mathcal{O}_T(1)\sqrt{d}\sqrt{\delta}$ |

---

## 3. Proof Techniques

1. **Fokker-Planck matching for time reversal.** The reverse-time SDE coefficients are derived by matching the Fokker-Planck equation of the time-reversed density $\bar{p}(t,x) = p(T-t,x)$ with that of a candidate SDE, yielding Anderson's formula via algebraic identification.

2. **Girsanov theorem for KL on path space.** The key identity $\text{KL}(Q''_T, Q'_T) = \mathbb{E}_{Q''_T}\int_0^T |s_\theta(T-t,Y_t) - \nabla\log p(T-t,Y_t)|^2\,dt$ converts the score estimation error into a KL divergence between path measures, which is then reduced to TV via Pinsker's inequality.

3. **Integration by parts for implicit score matching.** The proof of ISM = ESM + const uses the divergence theorem to eliminate the unknown score $\nabla\log p$ from the gradient $\nabla_\theta \mathcal{J}$.

4. **Synchronous coupling + Gronwall for Wasserstein bounds.** Two reverse SDEs (one with true score, one with estimated score) are driven by the **same** Brownian motion. Ito's formula gives a differential inequality for $\mathbb{E}|U_t - V_t|^2$, and Gronwall's lemma bounds the accumulated error.

5. **Convexity of KL + log-concavity preservation.** The VP convergence corollaries exploit that convolution preserves strong log-concavity, yielding $p(T-t, \cdot)$ is log-concave with a known constant that improves the Gronwall bound.

6. **Wasserstein coupling for consistency model analysis.** The gap between consistency distillation (CD) and consistency training (CT) is quantified by computing $W_2^2$ between the joint distributions $(Y_+^{CD}, Y_-^{CD})$ and $(Y_+^{CT}, Y_-^{CT})$, showing it is of order $\sqrt{(1 - \sqrt{2}/2)d}$.

---

## 4. Connection to Our Paper

Our paper "Where to Search: GAINS" uses diffusion models as the base generative model and studies inference-time compute allocation. This tutorial provides rigorous SDE foundations relevant to our OR audience.

### What We Can Borrow (theoretical framework, notation)

1. **SDE formulation and notation.** The forward/backward SDE setup ($f$, $g$, $p(t,\cdot)$, $s_\theta$) is the standard language for diffusion models. We can cite this tutorial as the canonical reference for the continuous-time framework that underpins our discrete-step abstraction, especially when writing for an OR audience that is comfortable with SDEs.

2. **Three-way error decomposition.** The decomposition into noise approximation error + score matching error + discretization error (Section 5/6) directly motivates our insight that compute allocation across timesteps is non-trivial. Different timesteps contribute differently to the total error, and GAINS exploits this heterogeneity.

3. **VE/VP taxonomy.** The unified treatment of VE, VP, and sub-VP as special cases of the same SDE framework supports our claim that the two-level scheduling framework is model-agnostic: GAINS experiments span VP-based (Stable Diffusion) and VE-based (EDM) models.

4. **Polynomial score error growth.** The breakthrough result that score error accumulation is $O(\varepsilon\sqrt{T})$ rather than $O(\varepsilon e^{T})$ means that later denoising stages still have non-negligible marginal benefit from noise refinement -- justifying non-trivial budget allocation at all stages, not just early ones.

5. **Probability flow ODE.** The equivalence between SDE and ODE samplers (Theorem in Section 5) is relevant because GAINS can operate on both stochastic and deterministic sampling pipelines.

### How We Differ

1. **Fixed score, optimized noise.** This tutorial focuses on learning $\nabla\log p_t$ (score matching); GAINS takes the learned score as fixed and optimizes over the noise variables $\varepsilon_t$ injected at each denoising step.

2. **Compute allocation as an OR problem.** The tutorial treats $T$ (total time) and score network capacity as the main design levers. GAINS introduces a new decision dimension: per-timestep search budget allocation under a global NFE constraint, which is a resource allocation / knapsack-type optimization problem outside the scope of SDE convergence theory.

3. **Instance-adaptive online control.** The tutorial's convergence bounds are worst-case over the trajectory. GAINS shows that instance-adaptive online policies can recover the Jensen gap between optimal fixed allocation and instance-specific optima.

4. **Verifier-guided search.** The tutorial's RL section (Section 5.3) treats score fine-tuning via policy gradient; GAINS uses a separate verifier (reward model) at inference time to guide noise selection without modifying the score network.

---

## 5. Key Quotes

> "The view is that the SDEs unveil structural properties of the models, whereas the discrete counterparts give practical implementation." (Section 1)

> "The disadvantage of this result is that the score matching error grows exponentially in time $T$. A recent breakthrough improved this bound to be polynomial in $T$ (and $d$)." (Section 5.1, discussing Chen et al. 2023)

> "Among these examples, VE and VP SDEs are the most widely used models (e.g., Stable Diffusion uses VP, and consistency models rely on VE)." (Section 2)

---

## 6. BibTeX

```bibtex
@article{tang2024sde_tutorial,
  title={Score-based Diffusion Models via Stochastic Differential Equations -- a Technical Tutorial},
  author={Tang, Wenpin and Zhao, Hanyang},
  journal={Statistics Surveys},
  volume={19},
  pages={28--64},
  year={2025},
  publisher={Institute of Mathematical Statistics},
  doi={10.1214/25-SS152},
  note={arXiv preprint arXiv:2402.07487}
}
```
