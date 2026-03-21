# Tang, Zhao (2024) -- "Contractive Diffusion Probabilistic Models"

**Paper:** arXiv:2401.13115
**Authors:** Wenpin Tang, Hanyang Zhao
**Affiliation:** Department of Industrial Engineering and Operations Research, Columbia University
**Keywords:** Contraction, diffusion probabilistic models, discretization, generative models, image synthesis, sampling, score matching, stochastic differential equations

---

## 1. Setting & Model

### Problem Formulation

The paper addresses a fundamental design question for score-based diffusion probabilistic models (DPMs):

> How to design the drift and diffusion coefficients $(b(\cdot,\cdot), \sigma(\cdot))$ of the forward SDE for better generation quality?

Standard DPMs rely on a forward process governed by:

$$dX_t = b(t, X_t)\,dt + \sigma(t)\,dB_t, \quad X_0 \sim p_{\text{data}}$$

and a backward (generative) process obtained via time reversal:

$$d\bar{X}_t = \left(-b(T-t, \bar{X}_t) + \sigma^2(T-t)\,\nabla \log p(T-t, \bar{X}_t)\right)dt + \sigma(T-t)\,d\bar{B}_t$$

In practice, the true score $\nabla \log p(t,x)$ is replaced by a learned approximation $s_\theta(t,x)$ via score matching (denoising score matching / DSM). The paper observes that score matching errors are unavoidable and potentially large, and proposes a structural criterion -- **contractiveness of the backward process** -- to make DPMs robust to these errors.

### Key Insight: Contractive Backward Processes

Existing DPMs (OU, VP, VE, subVP) have **contractive forward processes** but not necessarily contractive backward processes. Score matching errors introduced at each backward step can accumulate and amplify over time. The paper's central proposal is:

> The backward process of the DPM shall be **contractive**.

A backward process is contractive when a Lyapunov-type condition ensures that perturbations (from score matching errors and discretization) shrink rather than grow during backward sampling. Formally, the practical contractiveness condition requires:

$$r_b(t) > 0 \quad \text{for } t \in [\epsilon, T]$$

where $r_b(t)$ quantifies the one-sided Lipschitz constant of the drift: $(x - x') \cdot (b(t,x) - b(t,x')) \ge r_b(t)|x-x'|^2$. For linear SDEs with $b(t,x) = b(t) \cdot x$, this reduces to $b(t) > 0$ (repulsive drift in the forward process, contractive drift in the backward process).

### Contractive DPM Examples

The paper proposes three concrete CDPM instances by flipping the sign of the drift coefficient compared to standard DPMs:

| Model | $b(t)$ | $\sigma(t)$ | Prior $p_\infty$ |
|-------|--------|-------------|-------------------|
| COU | $+\theta$ | $\sigma$ | $\mathcal{N}(0, \frac{\sigma^2}{2\theta}(e^{2\theta T}-1)I)$ |
| CVP | $+\frac{1}{2}\beta(t)$ | $\sqrt{\beta(t)}$ | $\mathcal{N}(0, (e^{\frac{T}{2}(\beta_{\max}+\beta_{\min})}-1)I)$ |
| CsubVP | $+\frac{1}{2}\beta(t)$ | $\sqrt{\beta(t)(e^{2\int_0^t\beta(s)ds}-1)}$ | $\mathcal{N}(0, (e^{\frac{T}{2}(\beta_{\max}+\beta_{\min})}-1)^2I)$ |

Note the positive sign on $b(t)$, contrasting with the negative sign in VP ($-\frac{1}{2}\beta(t)$) and OU ($-\theta$). This means the forward process is **variance-exploding** (the signal grows rather than decays), producing a wider prior distribution, but the backward process inherits contractive dynamics.

### Standard DPMs for Comparison

| Model | $b(t)$ | $\sigma(t)$ | Prior $p_\infty$ |
|-------|--------|-------------|-------------------|
| OU | $-\theta$ | $\sigma$ | $\mathcal{N}(0, I)$ |
| VP | $-\frac{1}{2}\beta(t)$ | $\sqrt{\beta(t)}$ | $\mathcal{N}(0, I)$ |
| subVP | $-\frac{1}{2}\beta(t)$ | $\sqrt{\beta(t)(1-e^{-2\int_0^t\beta(s)ds})}$ | $\mathcal{N}(0, I)$ |
| VE | $0$ | $\sigma_{\min}(\sigma_{\max}/\sigma_{\min})^{t/T}\sqrt{(2/T)\log(\sigma_{\max}/\sigma_{\min})}$ | $\mathcal{N}(0, \sigma_{\max}^2 I)$ |

### Connection to VE and Pretrained Models

The paper shows that VE SDEs exhibit a **hidden contractive property** at earlier denoising steps (when the distribution is far from the data), but lose it near the end of the backward process. CDPM enforces contraction throughout. Crucially, CDPM can be derived from VE via a time/space change of variables, so **pretrained VE score networks can be reused without retraining**:

$$\nabla \log p_{\text{CsubVP}}(t,x) \approx s_{\text{pre}}(\tau(t), x/f(t))$$

where $f(t) = e^{\frac{t^2}{4T}(\beta_{\max}-\beta_{\min}) + \frac{t}{2}\beta_{\min}}$ and $\tau(t)$ is an explicit time reparameterization. The perturbation kernel of CDPMs is *increasing* in time, as opposed to the constant (VE) or decreasing (VP, subVP) kernels of standard models.

---

## 2. Main Results (all theorems in LaTeX)

### Assumptions

**Assumption 1** (Sensitivity conditions):
1. There exists $r_b: [0,T] \to \mathbb{R}$ such that $(x-x') \cdot (b(t,x)-b(t,x')) \ge r_b(t)|x-x'|^2$ for all $t, x, x'$.
2. There exists $L > 0$ such that $|\nabla\log p(t,x) - \nabla\log p(t,x')| \le L|x-x'|$ for all $t, x, x'$.
3. There exists $\varepsilon > 0$ such that $\mathbb{E}|s_\theta(t,\bar{X}_{T-t}) - \nabla\log p(t,\bar{X}_{T-t})|^2 \le \varepsilon^2$ for all $t$.

**Assumption 2** (Regularity for discretization):
1. $|\sigma(t)-\sigma(t')| \le L_\sigma|t-t'|$ (Lipschitz diffusion coefficient)
2. $\sigma(t) \le R_\sigma$ (bounded diffusion coefficient)
3. $|b(t,x)-b(t',x')| \le L_b(|t-t'|+|x-x'|)$ (Lipschitz drift)
4. $|s_\theta(t,x)-s_\theta(t',x')| \le L_s(|t-t'|+|x-x'|)$ (Lipschitz score matching)
5. $|s_\theta(T,x)| \le R_s(1+|x|)$ (linear growth of score matching)

**Assumption 3** (Contraction):
$$\beta := \inf_{0\le t\le T}(r_b(t) - L_s\sigma^2(t)) > 0$$

or equivalently: $\int_{T-t}^T (r_b(s) - L_s\sigma^2(s))\,ds \ge \beta t$ for all $t$.

### Theorem 1 (Continuous-time sampling error -- Wasserstein-2 bound)

Let Assumption 1 hold and $h > 0$. Define $\eta := W_2(p(T,\cdot), p_{\text{noise}}(\cdot))$ and

$$u(t) := \int_{T-t}^T \left(-2r_b(s) + (2L+2h)\sigma^2(s)\right)ds$$

Then:

$$W_2(p_{\text{data}}(\cdot), \bar{X}_T) \le \sqrt{\eta^2 e^{u(T)} + \frac{\varepsilon^2}{2h}\int_0^T \sigma^2(t) e^{u(T)-u(T-t)}\,dt}$$

**Key implication for CDPMs:** When $r_b(t) > 0$ is sufficiently large, $u(T) < 0$, so the initialization error $\eta$ is exponentially damped and the score matching error $\varepsilon$ does not grow with $T$. The bound simplifies to:

$$\underbrace{(\text{noise inaccuracy})\cdot e^{-T}}_{\text{initialization error}} + \underbrace{(\text{score mismatch})\cdot(1-e^{-T})}_{\text{score error}} + \underbrace{\text{Poly}(\text{step size})}_{\text{discretization error}}$$

This contrasts with standard DPMs where both score error and discretization error grow as $\text{Poly}(T)$.

### Theorem 2 (CVP-specific Wasserstein-2 bound under log-concavity)

Let $\bar{X}$ follow the CVP backward process. Assume $\log p_{\text{data}}(\cdot)$ is $\kappa$-strongly log-concave and $\mathbb{E}_{p_{\text{data}}}|x|^2 < \infty$. For $h < \min(\frac{1}{2}, \frac{1}{\beta_{\max}T}\frac{\kappa}{1+\kappa})$:

$$W_2^2(p_{\text{data}}(\cdot), \bar{X}_T) \le e^{-2\left(\frac{\kappa}{1+\kappa} - \beta_{\max}hT + \mathcal{O}(e^{-\beta_{\min}T})\right)}\mathbb{E}_{p_{\text{data}}}|x|^2 + \frac{\varepsilon^2}{2h(1-2h)}$$

**Interpretation:** The first term decays exponentially (initialization error vanishes for large $T$), while the second term is controlled by the score matching error $\varepsilon$ and is independent of $T$.

### Theorem 3 (Discretization error -- independent of $T$)

Let Assumptions 1, 2, and 3 hold. Then there exists $C > 0$ independent of $\delta$ and $T$ such that for $\delta > 0$ sufficiently small:

$$\left(\mathbb{E}|\bar{X}_T - \hat{X}_N|^2\right)^{1/2} \le C\sqrt{\delta}$$

**Interpretation:** The discretization error for CDPMs is $O(\sqrt{\delta})$ with a constant **independent of $T$**. Classical SDE discretization theory gives $C(T)$ exponential in $T$; the contraction property eliminates this dependence.

### Theorem 4 (VE-to-CDPM transformation)

Assume $\sigma_{\max}^2 - \sigma_{\min}^2 > g^2(T)$. Then for $t \in [0,T]$:

$$p_{\text{CsubVP}}(t,x) = f(t)^{-d}\,p_{\text{VE}}(\tau(t), x/f(t))$$

where $\tau(t) = \frac{T}{2}\frac{\log(1+g^2(t)/\sigma_{\min}^2)}{\log(\sigma_{\max}/\sigma_{\min})}$ and $g(t) = f(t) - f^{-1}(t)$.

This enables direct reuse of pretrained VE score networks for CDPM sampling: $\nabla \log p_{\text{CsubVP}}(t,x) \approx s_{\text{pre}}(\tau(t), x/f(t))$.

### Contraction Lemma

Under the contractive condition (Assumption 3), for two coupled backward trajectories starting from $x$ and $y$ (driven by the same Brownian motion):

$$\left(\mathbb{E}|\bar{X}_t^x - \bar{X}_t^y|^2\right)^{1/2} \le \left(\mathbb{E}|x-y|^2\right)^{1/2}\exp(-2\beta t)$$

This exponential contraction is the key property that prevents error accumulation.

---

## 3. Proof Techniques

### Synchronous Coupling for Sampling Error (Theorem 1)

The proof constructs a coupling between two backward SDEs driven by the same Brownian motion: $Y_t$ uses the true score $\nabla\log p$ and $Z_t$ uses the learned score $s_\theta$. By Ito's formula, the evolution of $\mathbb{E}|Y_t - Z_t|^2$ satisfies the differential inequality:

$$\frac{d}{dt}\mathbb{E}|Y_t - Z_t|^2 \le \left(-2r_b(T-t) + (2h+2L)\sigma^2(T-t)\right)\mathbb{E}|Y_t - Z_t|^2 + \frac{\varepsilon^2}{2h}\sigma^2(T-t)$$

Gronwall's inequality yields the bound. The contractiveness condition ($r_b$ sufficiently positive) ensures the coefficient on $\mathbb{E}|Y_t-Z_t|^2$ is negative, so perturbations decay rather than grow.

### Local-to-Global Discretization Analysis (Theorem 3)

The proof proceeds in four steps:

1. **Decompose** the global error $e_{k+1}^2 = \mathbb{E}|\bar{X}_{k+1} - \hat{X}_{k+1}|^2$ into a contraction term (a), a local discretization error (b), and a cross-product (c).

2. **Contraction** (term a): The Contraction Lemma gives $\mathbb{E}|\bar{X}_{t_{k+1}}^{t_k,\bar{X}_k} - \bar{X}_{t_{k+1}}^{t_k,\hat{X}_k}|^2 \le e_k^2\exp(-2\beta\delta)$.

3. **Local error** (term b): Standard Euler-Maruyama analysis gives $O(\delta^3)$ per step, using classical SDE moment bounds.

4. **Recursion**: The combined recursion $e_{k+1}^2 \le e_k^2(1 - \frac{1}{4}\beta\delta) + D\delta^2$ telescopes to give $e_N = O(\sqrt{\delta})$ independently of $N = T/\delta$.

### Log-Concavity Exploitation (Theorem 2)

For CVP under $\kappa$-strongly log-concave data, a result from Gao-Niles-Weed-Zhao (2023) shows that $\nabla\log p(T-t, \cdot)$ has a **negative** Lipschitz-type bound (strong concavity), which strengthens the contraction beyond what the generic Assumption 1 provides. This replaces $+L\sigma^2(s)$ with a negative term in $u(t)$, yielding the sharper CVP-specific bound.

### Perturbation Kernel Transformation (Theorem 4)

The connection to VE exploits the EDM framework of Karras et al. (2022): SDEs with linear drift $b(t,x) = b(t) \cdot x$ have perturbation kernels $X_t | X_0 \sim \mathcal{N}(f(t)X_0, f(t)^2 g(t)^2 I)$. Matching the conditional variance structure between CsubVP and VE gives the time reparameterization $\tau(t)$, and the spatial rescaling $x/f(t)$ accounts for the different scaling factors.

---

## 4. Connection to Our Paper

GAINS studies non-uniform compute allocation across diffusion timesteps. The contractive property explains why backward sampling quality varies per timestep -- central to our motivation.

### What We Can Borrow

1. **Per-timestep error sensitivity varies with contraction strength.** Theorem 1 shows that the score matching error at backward time $t$ is amplified by $e^{u(T)-u(T-t)}$, where the integrand $-2r_b(s) + (2L+2h)\sigma^2(s)$ gives a per-timestep error amplification rate. This directly supports our argument that some timesteps are more "error-sensitive" than others and thus deserve more search compute. GAINS can cite this as the theoretical mechanism underlying the heterogeneity in per-timestep sensitivity.

2. **VE's hidden contraction explains the "easy early, hard late" pattern.** Section 4.1 shows that VE is contractive at earlier denoising steps (when the distribution is still Gaussian-like and the score is nearly affine) but loses contractiveness near the data distribution. This aligns with the empirical observation that the final denoising steps are hardest and most sensitive to errors -- exactly where GAINS should allocate more compute.

3. **The initialization-vs-score-error tradeoff.** CDPMs trade a larger initialization error (from a wider prior) for better error propagation control. This tradeoff structure is analogous to the budget allocation tradeoff in GAINS: spending more compute on early steps reduces initialization error, while spending more on late steps reduces score-error amplification. Theorem 2 makes this tradeoff quantitative.

4. **$T$-independent discretization error.** Theorem 3 shows that under contraction, discretization error is $O(\sqrt{\delta})$ with a constant independent of $T$. This means the number of function evaluations needed for a given discretization accuracy does not grow with the diffusion horizon -- relevant when GAINS considers how to distribute a fixed NFE budget.

### How We Differ

1. **GAINS operates at the search/sampling level, not the SDE design level.** Tang and Zhao propose new SDE coefficients (CDPM) that change the forward/backward process dynamics. GAINS takes the SDE as given (e.g., standard VP or VE) and optimizes the **allocation of parallel search candidates** (noise samples evaluated by a verifier) across timesteps. The two approaches are complementary: CDPMs improve the underlying dynamics, while GAINS improves the search efficiency on top of any given dynamics.

2. **Compute budget vs. error propagation.** CDPM focuses on controlling how errors propagate through the backward process (a continuous-time, single-trajectory analysis). GAINS focuses on how a finite compute budget should be distributed to maximize the expected best-of-K outcome at each step (a discrete, multi-sample optimization). The variance that drives GAINS's allocation is the score distribution variance across candidate noises, not the score matching error studied here.

3. **Verifier-guided vs. verifier-free.** CDPM is a purely generative approach with no external verifier or reward signal. GAINS relies on a verifier (reward model) to evaluate candidate samples and select the best one at each step. The contractive property in CDPM structurally reduces the need for error correction, while GAINS uses active search to correct errors at inference time.

4. **Theoretical regime.** CDPM's theory assumes bounded $L^2$ score matching error uniformly across time and a Lipschitz score. GAINS's theory works with per-timestep score distributions and their variances, which are empirically measured and can vary dramatically across timesteps. The theoretical frameworks operate at different levels of abstraction.

---

## 5. Key Quotes

> "Our key insight is that the contraction property can provably narrow score matching errors and discretization errors, thus our proposed CDPMs are robust to both sources of error." (Abstract)

> "The idea is simply to make $u(t)$ be negative, that is to set $r_b(t) > 0$ sufficiently large, in order to prevent the score matching error from propagating in backward sampling." (Section 3.1)

> "CDPMs are inherently different from existing DPMs in the sense that these DPMs often have contractive forward processes, while our proposal requires contractive backward processes." (Section 3.1)

> "VE may lose this contractive property when the distribution is close to the target data distribution. The score matching error and discretization error near $t \approx T - \epsilon$ in the backward process indeed plays a large impact." (Section 4.1)

> "For practical use, we show that CDPM can leverage weights of pretrained DPMs by a simple transformation, and does not need retraining." (Abstract)

> "The main takeaway is that the contraction of the backward process limits score matching errors from propagating, and controls discretization error as well." (Section 5, Conclusion)

---

## 6. BibTeX

```bibtex
@article{tang2024contractive,
  title={Contractive Diffusion Probabilistic Models},
  author={Tang, Wenpin and Zhao, Hanyang},
  journal={arXiv preprint arXiv:2401.13115},
  year={2024}
}
```

### Related References

```bibtex
@inproceedings{song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Song, Yang and Sohl-Dickstein, Jascha and Kingma, Diederik P. and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{karras2022elucidating,
  title={Elucidating the Design Space of Diffusion-Based Generative Models},
  author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  booktitle={NeurIPS},
  year={2022}
}

@article{chen2023improved,
  title={Improved Analysis of Score-Based Generative Modeling: User-Friendly Bounds under Minimal Smoothness Assumptions},
  author={Chen, Hongrui and Lee, Holden and Lu, Jianfeng},
  journal={arXiv preprint arXiv:2211.01916},
  year={2023}
}

@article{gao2023wasserstein,
  title={Wasserstein Convergence Guarantees for a General Class of Score-Based Generative Models},
  author={Gao, Xuefeng and Niles-Weed, Jonathan and Zhao, Hanyang},
  journal={arXiv preprint},
  year={2023}
}
```

### Experimental Results Summary

| Dataset | Model | Metric | Score |
|---------|-------|--------|-------|
| Swiss Roll | COU | $W_2$ | **0.10** (vs OU 0.29, VE 0.18) |
| Swiss Roll | CsubVP | $W_2$ | **0.14** (vs subVP 0.34) |
| MNIST | CsubVP | FID | **0.03** (vs VE 0.20, VP 0.79) |
| CIFAR-10 (NCSN++) | CsubVP | FID / IS | **2.47 / 10.18** (vs VE 2.50/9.68, VP 2.55/9.58) |
| CIFAR-10 (EDM, VP cond) | EDM+contraction | FID | **1.83** (vs EDM 1.85) |
| AFHQv2 64x64 (VE uncond) | EDM+contraction | FID | **2.20** (vs EDM 2.24) |
