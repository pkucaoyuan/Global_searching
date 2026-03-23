# Prompt for Gemini: Comprehensive Theory Audit — GAINS Paper

## Context

**Paper**: Leveraging Inference-Time Compute for Diffusion Models via Global Scheduling of Denoising Trajectories (GAINS)
**Sections under review**: §4 (algorithm.tex), §4.5 (general_local_search.tex), Appendix B (appendix_proofs.tex)
**Theory extract**: `paper/theory_extract.tex` (standalone compilable PDF, 10 pages, 24 blocks, 0 errors)

One-sentence summary: GAINS performs per-timestep noise search in diffusion model reverse sampling, scheduling a fixed NFE budget across timesteps using a gain factorization $G_t(K) = \sigma_t \phi_K^{\mathcal{L}} + O_K(g_t^2 h)$ derived from second-order Taylor expansion of the verifier score.

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $X_{t-h} = X_t + b_t h + g_t\sqrt{h}\,\xi$ | Reverse SDE Euler step; $\xi \sim \mathcal{N}(0,I)$ is the noise searched |
| $\Psi_t(x;c) = v(D_\theta(x,t{-}h);c)$ | Verifier score of the denoised prediction |
| $\bar{X}_{t-h} = X_t + b_t h$ | Deterministic Euler point (no noise) |
| $\mu_t = \Psi_t(\bar{X}_{t-h};c)$ | Baseline score at deterministic point |
| $\sigma_t = g_t\sqrt{h}\,\|\nabla\Psi_t(\bar{X}_{t-h};c)\|$ | Per-step sensitivity (location-scale parameter) |
| $G_t(K)$ | Expected gain from $K$ i.i.d. noise draws (random search) |
| $G_t^{\mathcal{L}}(K)$ | Expected gain from $K$ iterations of operator $\mathcal{L}$ |
| $\phi_K^{\mathcal{L}}$ | Operator-specific gain sequence; $\phi_0 = 0$, increasing, non-accelerating |
| $a_K = \mathbb{E}[\max(Z_1,\dots,Z_K)]$, $Z_k \sim \mathcal{N}(0,1)$ | Expected maximum of $K$ standard normals; $a_K \sim \sqrt{2\log K}$ |
| $(A1)$–$(A4)$ | Strict monotone, non-accelerating marginals, sensitivity scaling, rotational equivariance |
| $B$ | Total NFE budget; $\{K_t\}$ allocation with $\sum_t K_t = B$ |
| $V^*(\boldsymbol{\sigma})$ | Oracle value function (max gain under known $\boldsymbol{\sigma}$) |
| $\lambda^*$ | Shadow price of budget constraint (KKT multiplier) |
| $\eta$ | Perturbation scale in ε-greedy / local perturbation |
| $\epsilon$ | Exploration probability in ε-greedy search |

---

## Complete Theory Structure (14 results, 6 proofs)

### Chain 1: SDE foundations (algorithm.tex, §4.3)

| ID | Type | Label | Statement | Proof loc |
|----|------|-------|-----------|-----------|
| 1 | Assumption | asm:smooth | $\Psi_t \in C^2$, Hessian $\le L_t$ near the Euler segment | — |
| 2 | Proposition | prop:taylor | Score Taylor: $S_t = \mu_t + \sigma_t Z_t + R_t$, $\|R_t\| \le \frac{1}{2}L_t g_t^2 h \|\xi\|^2$ | Inline §4.3 |
| 3 | Theorem | thm:loc-scale | Location-scale: $\text{Var}(S_t) = \sigma_t^2 + O(g_t^3 h^{3/2})$; $G_t(K) = \sigma_t a_K + O_K(g_t^2 h)$ | Inline §4.3 |
| 4 | Corollary | cor:sde-alloc | Offline allocation reduces to $\max \sum_t \sigma_t a_{K_t}$ with SDE-derived $\sigma_t$ | Inline §4.3 |
| 5 | Remark | — | $\sigma_t = g_t\sqrt{h}\|\nabla\Psi_t\|$ governs sensitivity; explains why early/middle steps dominate | — |

### Chain 2: Scheduling optimality (algorithm.tex, §4.4)

| ID | Type | Label | Statement | Proof loc |
|----|------|-------|-----------|-----------|
| 6 | Proposition | prop:offline | Water-filling: optimal $K_t^*$ monotone in $\sigma_t$; non-uniform strictly dominates uniform | Appendix B |
| 7 | Remark | — | Justifies offline profiling: under independence, single schedule is simultaneously optimal | — |
| 8 | Proposition | prop:online | $V^*$ is convex; Jensen gap motivates online adaptation; dual-threshold stopping derived | Appendix B |
| 9 | Remark | — | Explains two-factor marginal $G_t'(K) = \sigma_t \cdot a_K'$; stopping diagram | — |

### Chain 3: General operators (general_local_search.tex, §4.5)

| ID | Type | Label | Statement | Proof loc |
|----|------|-------|-----------|-----------|
| 10 | Assumption | asm:local-search | Four axioms (A1)–(A4) for any operator $\mathcal{L}$ | — |
| 11 | Remark | rmk:grounding | Grounds (A1)–(A4) in order statistics / convergence / MCMC theory | — |
| 12 | Theorem | thm:general-gain | General factorization: $G_t^{\mathcal{L}}(K) = \sigma_t \phi_K^{\mathcal{L}} + O_K(g_t^2 h)$ | Inline §4.5 |
| 13 | Corollary | cor:offline-general | Water-filling for general operators; degenerates to budget concentration when $\phi_K$ is linear | Inline §4.5 |
| 14 | Corollary | cor:online-general | Convexity of $V_{\mathcal{L}}^*$; Jensen gap; dual-threshold valid for any $(A1)$–$(A4)$ operator | Inline §4.5 |
| 15 | Remark | rmk:jensen-curvature | Faster-growing $\phi_K$ enlarges Jensen gap; online adaptation more valuable for efficient operators | — |
| 16 | Example | ex:random-search | $\phi_K^{RS} = a_K \sim \sqrt{2\log K}$; (A1)–(A4) verified | — |
| 17 | Example | ex:eps-greedy | $\phi_K^{(\epsilon)} \ge a_{\lceil\epsilon K\rceil}$; detailed (A1)–(A4) verification | — |
| 18 | Example | ex:local-perturbation | $\phi_K^{(LP)} = \eta K/\sqrt{2\pi}$ (linear); (A1)–(A4) verified | — |
| 19 | Table | tab:phi-summary | Growth rate summary: RS $O(\sqrt{\log K})$, ε-greedy $\Omega(\sqrt{\log K})$, LP $O(K)$ | — |
| 20 | Proposition | prop:crossover | Crossover $K^* = \tilde{\Theta}(\sqrt{2\pi}/\eta)$: LP dominates for $K \ll K^*$, remainder-bounded for $K \gg K^*$ | Inline §4.5 |
| 21 | Remark | rmk:operator-selection | Operator selection as richer MDP action; MDP framework already accommodates it | — |

---

## Task 1: Verify Logical Completeness

Please verify each of the following logical claims. For each, indicate: ✅ correct | ⚠️ gap or imprecision | ❌ error.

### 1a. prop:taylor (Score Taylor expansion)

The claim is: $|R_t| \le \frac{1}{2} L_t g_t^2 h \|\xi\|^2$ almost surely, where $R_t = \frac{1}{2}(g_t\sqrt{h}\,\xi)^\top \nabla^2\Psi_t(\widetilde{X};c)(g_t\sqrt{h}\,\xi)$ for some $\widetilde{X}$ on the segment $[\bar{X}_{t-h}, X_{t-h}]$.

**Check**: Is the Cauchy–Schwarz bound $|x^\top A x| \le \|A\|_{\text{op}} \|x\|^2$ applied correctly? Does the "almost surely" qualifier apply given the assumption on a neighborhood of the segment?

### 1b. thm:loc-scale(ii) (Variance approximation)

The claim is: $\text{Var}(S_t | X_t, c) = \sigma_t^2 + O(g_t^3 h^{3/2})$.

The proof bounds $\text{Cov}(\sigma_t Z_t, R_t) = O(g_t^3 h^{3/2})$ by noting $\sigma_t Z_t = O(g_t\sqrt{h})$ and $R_t = O(g_t^2 h)$, so their covariance is $O(g_t^3 h^{3/2})$.

**Check**: Is this covariance bound rigorous? Specifically, does $\text{Cov}(X, Y) = O(\text{std}(X) \cdot \text{std}(Y))$ hold here? Note $\text{std}(\sigma_t Z_t) = \sigma_t = g_t\sqrt{h}\|\nabla\Psi_t\|$ and $\text{std}(R_t) \le \frac{1}{2}L_t g_t^2 h \sqrt{\mathbb{E}[\|\xi\|^4]} = O(g_t^2 h d)$. So $\text{Cov} \le \text{std}(\sigma_t Z_t)\cdot\text{std}(R_t) = O(g_t^3 h^{3/2} d)$. Does this match the claimed $O(g_t^3 h^{3/2})$?

### 1c. thm:loc-scale(iii) (Gain factorization)

The proof uses:
$$|\max_j(a_j+b_j) - \max_j a_j| \le \max_j |b_j|$$

**Check**: Is this inequality correct? (Counterexample candidate: $a_1=0, b_1=1, a_2=2, b_2=-2$: LHS = $|2-2|=0$, RHS = $\max(1,2)=2$. ✅) Does the union bound $\max_j |b_j| \le \sum_j |b_j|$ used afterward only waste a factor of $K$? Yes — is the $O_K(g_t^2 h)$ remainder therefore $K \cdot \frac{1}{2}L_t g_t^2 h d$, so the constant grows as $O(Kd)$? Confirm this is correctly tracked.

### 1d. thm:general-gain (General gain factorization)

The proof applies rotational equivariance (A4) to conclude that $\mathbb{E}[\max_j \langle u_t, \xi^{(j)}\rangle]$ does not depend on the direction $u_t$.

**Check**: The argument is that $\{Q\xi^{(j)}\}$ has the same distribution as $\{\xi^{(j)}\}$ started from $Q\xi^{(0)}$ — but the initial condition $\xi^{(0)} \sim \mathcal{N}(0,I)$ is itself rotationally invariant, so $Q\xi^{(0)} \stackrel{d}{=} \xi^{(0)}$. Does this complete the argument that $\phi_K^{\mathcal{L}} = \mathbb{E}[\max_j \langle u_t, \xi^{(j)}\rangle]$ is $u_t$-independent?

### 1e. cor:offline-general (Water-filling for general operators)

Part (ii) states: "When $\phi_K$ is strictly concave, $K_t^*$ is strictly increasing (water-filling); when $\phi_K$ is linear, the solution concentrates all budget on $\arg\max_t \sigma_t$."

**Check**: The original prop:offline (and its appendix proof) explicitly requires $a_K$ to be **strictly concave** to derive strict water-filling monotonicity via KKT. The assumption (A2) only requires **non-accelerating marginals** (marginals bounded above by a non-increasing sequence) — which allows both strict concavity AND linearity. So prop:offline's proof (which uses strict concavity) cannot be directly cited for the full generality of (A2). Is this distinction made clear? Specifically: does cor:offline-general correctly restrict the strict water-filling claim to the strict concavity sub-case?

### 1f. prop:online Part (ii) (Dual-threshold stopping)

The proof establishes that $\bar{g}_s \ge G_s'(\hat{K}_s)$ via the chord $\ge$ tangent inequality for concave functions. This uses:
$$\bar{g}_s = \frac{G_s(\hat{K}_s) - G_s(1)}{\hat{K}_s - 1} \ge G_s'(\hat{K}_s)$$

**Check**: This requires $G_s(1) = 0$ (zero gain from a single candidate). Is this consistent with the definition $G_t(K) = \mathbb{E}[\max_{1\le k\le K} S_t^{(k)} | X_t, c] - \mu_t$? For $K=1$: $G_t(1) = \mathbb{E}[S_t^{(1)} | X_t, c] - \mu_t = \mathbb{E}[\mu_t + \sigma_t Z + R] - \mu_t = O(g_t^2 h d)$. So $G_t(1) = O(g_t^2 h)$, not exactly zero. Does this approximation affect the validity of the chord bound? (Yes: the bound becomes $\bar{g}_s \ge G_s'(\hat{K}_s) - O(g_t^2 h / (\hat{K}_s - 1))$, which is small. Confirm this is negligible in the regime of interest.)

### 1g. prop:crossover Part (ii)

The current version states: "above $K^*$, the quadratic remainder $R_t(\xi) = O(g_t^2 h \|\xi^{(K)}\|^2)$ increasingly penalizes local perturbation's drifted iterates."

**Check**: Is this claim rigorous? Does the paper define "drifted iterates" precisely? For local perturbation, the incumbent $\xi^*$ drifts away from the origin as $K \to \infty$: $\|\xi^{(K)}\|^2 \approx K \cdot (\eta/\sqrt{2\pi})^2 \cdot \pi/2$ (random walk with positive drift). Does the paper quantify this drift? Is part (ii) of prop:crossover a formal proposition or informal description? If informal, should it be labeled a "Remark" rather than a numbered result in a proposition?

---

## Task 2: Check Theory–Code Alignment

The GAINS codebase has two key operator implementations. Compare each with the paper's theoretical description.

### 2a. ε-greedy operator

**Paper** (ex:eps-greedy, general_local_search.tex lines 341–427):
- With prob $\epsilon$: draw $\xi_{\text{new}} \sim \mathcal{N}(0,I)$ (exploration)
- With prob $(1-\epsilon)$: set $\xi_{\text{new}} = \xi^* + \eta\zeta$, $\zeta \sim \mathcal{N}(0,I)$ (exploitation)
- Expected exploitation improvement: $\eta \mathbb{E}[\max(0,Z)] = \eta/\sqrt{2\pi}$ per step

**Code** (`src/search/diffusion_tts_search.py`, `EpsilonGreedySearch`, lines 419–443):
```python
if torch.rand(1) < epsilon:
    candidate_noise = torch.randn_like(x_cur)        # ε: fresh N(0,I)
else:
    random_direction = normalize(torch.randn_like(pivot_noise))   # unit vector
    scale = torch.rand(...) * lambda_scaled           # Uniform(0, λ)
    candidate_noise = pivot_noise + scale * random_direction      # uniform ball
```

**The discrepancy**: The paper assumes exploitation perturbation $\eta\zeta$, $\zeta \sim \mathcal{N}(0,I)$, giving per-step improvement $\eta/\sqrt{2\pi}$. The code uses a **uniform ball** perturbation: scale $\sim U(0, \lambda)$, direction $\sim$ random unit vector. The projected perturbation distribution differs.

**Please compute**: For the uniform ball model, what is the expected improvement per exploitation step in the linearized regime?
- The perturbation in the projected coordinate $u$ is $\text{scale} \cdot \langle u, \text{direction}\rangle = \text{scale} \cdot \cos\theta$, where $\cos\theta$ is the cosine of a random angle in $d$ dimensions.
- For large $d$: $\cos\theta \approx 0$ (concentration of measure). So in high dimensions, the exploitation step gives essentially zero gain?
- Alternatively, is the projection $\langle \text{direction}, u\rangle \sim N(0, 1/d)$? Then the expected improvement is $\sim \frac{\lambda}{2} \cdot \frac{1}{\sqrt{d}} \cdot \frac{1}{\sqrt{2\pi}}$ (much smaller than $\eta/\sqrt{2\pi}$ in high $d$).

**Question**: Should the paper's exploitation model be updated to use uniform ball perturbation? Or should the code be updated to use Gaussian perturbation? Which better reflects the experiments? What is the correct $\phi_K^{(\epsilon)}$ formula for the uniform ball model?

### 2b. Local perturbation operator (ε-greedy with ε=0)

**Paper** (ex:local-perturbation, lines 429–485):
- Each iteration: $\xi_{\text{new}} = \xi^* + \eta\zeta$, $\zeta \sim \mathcal{N}(0,I)$
- Projected improvement: $\eta\langle u,\zeta\rangle \sim \mathcal{N}(0,\eta^2)$
- Accept when positive: expected improvement $\eta\mathbb{E}[\max(0,Z)] = \eta/\sqrt{2\pi}$
- Hence: $\phi_K^{(LP)} = \eta K/\sqrt{2\pi}$ (linear)

**Code** (`ZeroOrderSearchTTS`, lines 226–244):
```python
random_direction = normalize(torch.randn_like(pivot_noise))
scale = torch.rand(...) * lambda_scaled   # Uniform(0, λ)
candidate_noise = pivot_noise + scale * random_direction
```
Note: in `EpsilonGreedySearch` with `epsilon=0`, the same uniform ball perturbation is used.

**Same discrepancy as 2a**: The paper uses Gaussian $\eta\zeta$ but the code uses uniform ball.

**Question**: Does the high-dimensional uniform ball model still give a linear gain sequence $\phi_K^{(LP)} \propto K$? (Yes — linearity follows from the fact that each iteration is identically distributed given accept-reject, regardless of the marginal distribution.) But the **proportionality constant** is different from $\eta/\sqrt{2\pi}$.

**Compute**: For the uniform ball model in $d$ dimensions, what is the constant $c_d$ such that $\phi_K^{(LP)} = c_d \cdot K$? Is $c_d \to 0$ as $d \to \infty$?

### 2c. GAINS implementation

**Paper** (algorithm.tex, §4.1–4.2):
- Offline: coarse per-step budget $\{K_t\}$ from profiling
- Online: dual-threshold stopping based on $\widetilde{g}^{(j)} < \beta_g \bar{g}$ AND $\sigma^{(j)} < \beta_\sigma \bar{\sigma}$

**Code** (`src/pipeline/sampling_pipeline.py`, or local_search.py): Please check whether the online stopping condition in the algorithm (revert-on-negative, dual-threshold check in Algorithm 1) is implemented as described. In particular:
- Is `g_th = β_g * Mean(H_g)` computed as the mean of previous-step gains, or something else?
- Is the `revert-on-negative` rule exactly as in Algorithm 1 line 9–10?

---

## Task 3: Identify Strengthening Opportunities

Based on the current theory, evaluate whether the following strengthening directions are achievable:

### 3a. Non-asymptotic bounds

The current theory works in the "linearized regime" ($g_t\sqrt{h} \to 0$). Can any of the results be made non-asymptotic?

Specifically: the remainder in thm:loc-scale(iii) is $O_K(g_t^2 h)$ where the constant is $K \cdot \frac{1}{2} L_t d$. For the theory to be useful at practical $g_t, h$, this error must be small relative to the signal $\sigma_t a_K$.

**Question**: Can you state a sufficient condition of the form "$g_t\sqrt{h} \le \delta(\epsilon, K, d, L_t)$" under which the approximation error is at most $\epsilon \cdot \sigma_t a_K$? This would give a non-asymptotic regime validity certificate.

### 3b. Tightening the ε-greedy lower bound

The current paper only proves $\phi_K^{(\epsilon)} \ge a_{\lceil\epsilon K\rceil}$ (a lower bound). Can an upper bound or exact formula be derived?

In the linearized regime: the exploration component gives gain $a_{\lceil\epsilon K\rceil}$ and the exploitation component gives additional gain beyond this. Can the exploitation contribution be bounded above?

**Hint**: For $\epsilon$ close to 1, the operator is close to random search: $\phi_K^{(\epsilon)} \approx a_K$. For small $\epsilon$, exploitation dominates: $\phi_K^{(\epsilon)} \approx \phi_K^{(LP)} = \eta K/\sqrt{2\pi}$. Is there an interpolation formula?

### 3c. Stronger crossover result

Prop:crossover Part (ii) currently only provides an informal argument about remainder-based bounding. Can a quantitative version be proved?

Specifically: In the full (non-linearized) problem, the incumbent drift after $K$ steps is $\|\xi^{(K)}\|^2 \approx K (\eta/\sqrt{2\pi})^2 \cdot C$ for some constant $C$. The remainder penalty is $\frac{1}{2}L_t g_t^2 h \|\xi^{(K)}\|^2 = O(K g_t^2 h)$. For this to overwhelm the linear gain $\eta K/\sqrt{2\pi} \sigma_t$, we need $g_t^2 h \gg \sigma_t / L_t$. Can a precise crossing point $K^{**}$ be derived for the full problem?

### 3d. Submodularity / matroid structure

The offline allocation problem~(6) is a separable resource allocation problem. It has additional structure:
- Each $\phi_K^{\mathcal{L}}$ is non-accelerating (concave-like).
- The constraint $\sum_t K_t = B$ is a budget constraint.

Is there a submodularity interpretation? Could a greedy algorithm (assign one unit at a time to the timestep with highest marginal gain) achieve the optimum? (Yes for concave separable maximization.) Can this be noted in the paper as an efficient algorithm?

### 3e. Regret bound for online adaptation

The Jensen gap (eq:jensen-general) is $\mathbb{E}[V^*(\boldsymbol{\sigma})] - V^*(\bar{\boldsymbol{\sigma}}) \ge 0$. Can this gap be bounded above?

For random search: $V^*(\boldsymbol{\sigma}) = \max_K \sum_t \sigma_t a_{K_t}$ where $a_{K_t}$ grows as $\sqrt{2\log K_t}$. The gap depends on the variance of $\boldsymbol{\sigma}$. Can a bound of the form:
$$\text{Gap} \le \frac{1}{2} \text{Var}(\boldsymbol{\sigma}) \cdot \max_t \phi''_{K_t^*}$$
be derived from the convexity of $V^*$? (Here $\phi''$ is the second derivative w.r.t. $\sigma$, capturing curvature.)

---

## Task 4: After Identifying Issues, Produce Fixes

For each issue found in Tasks 1–3, if a fix is needed, provide:
1. The corrected mathematical statement or proof step
2. The specific location in the tex files (file, approximate lines)
3. The corrected LaTeX code

### Priority ranking:
- **Critical** (must fix before submission): logical errors, wrong formulas
- **Important** (should fix): imprecisions that affect claims
- **Minor** (optional): clarifications, additional remarks

---

## Task 5: Sync Instructions

After identifying and implementing all fixes:

### Step 5a: Update main paper

For each fix, apply it to the corresponding source file:
- `paper/sections/algorithm.tex` — for changes to §4.3–§4.4
- `paper/sections/general_local_search.tex` — for changes to §4.5
- `paper/sections/appendix_proofs.tex` — for changes to Appendix B proofs

### Step 5b: Regenerate theory extract

```bash
cd /home/CPU/Global_searching/paper
python extract_theory.py --compile
```

Verify:
- [ ] All 24 theory blocks still extracted (or updated count)
- [ ] PDF compiles without errors
- [ ] Cross-reference stubs count unchanged (or documented)
- [ ] theory_extract.pdf page count noted

### Step 5c: Verify consistency

```bash
cd /home/CPU/Global_searching/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Confirm: main.pdf compiles without errors.

---

## Deliverables

1. **Logic audit table**: Each of the 14 results with status (✅ / ⚠️ / ❌) and brief comment
2. **Code-theory alignment report**: For each operator, state whether paper matches code; propose fix direction (update paper vs. update code)
3. **Strengthening assessment**: For each of 3a–3e, state: achievable / requires new ideas / out of scope; provide sketch if achievable
4. **Fix list**: Ordered by priority (Critical/Important/Minor), with exact LaTeX changes
5. **Updated source files**: Ready-to-compile .tex with all fixes applied

## Verification Checklist

Before submitting deliverables:
- [ ] Each logical gap identified has a concrete fix (not just "this needs more work")
- [ ] The crossover $K^* = \tilde\Theta(\sqrt{2\pi}/\eta)$ derivation verified numerically for $\eta = 0.1, 0.5, 1.0$
- [ ] The Gaussian vs. uniform ball discrepancy (Task 2a–2b) has a recommendation
- [ ] After fixes, extract_theory.py runs without errors and theory_extract.pdf compiles
- [ ] After fixes, main.tex compiles without errors

---

## Known Issues from Prior Verification Rounds

The following issues were identified and fixed in earlier rounds (do NOT reintroduce):
1. ε-greedy (A2): original flawed argument (exploitation marginal is constant, not decreasing) → fixed with coupling decomposition: $\Delta_K = \epsilon \cdot (\text{non-increasing}) + (1-\epsilon) \cdot \eta/\sqrt{2\pi}$
2. Local perturbation (A2): original claimed strict concavity → weakened (A2) to "non-accelerating marginals"
3. Crossover Part (ii): original false "$\sqrt{d}$ saturation" claim → replaced with remainder-based bounding
4. Zero-order gradient estimation → removed; replaced with local perturbation search
5. Langevin MCMC example → removed (improper target in linearized regime)
6. `\srcfile{}` with underscores → fixed with `\detokenize{}` in extract_theory.py

The **primary open question** from prior rounds is #7 (not yet fixed):
> The paper's operator model uses Gaussian perturbation $\eta\zeta$, but the actual codebase uses uniform ball perturbation. This affects the $\eta/\sqrt{2\pi}$ marginal gain formula and the crossover constant.

---

## Output Format

```latex
% Fix for [issue description]
% File: paper/sections/[filename].tex
% Lines: approximately [N]–[M]
% Replace:
\begin{[env]}
[old content]
\end{[env]}
% With:
\begin{[env]}
[corrected content]
\end{[env]}
```

For operator-model alignment fix, also provide:
```python
# Updated theoretical model in general_local_search.tex
# The correct formula for uniform-ball perturbation gain
```
