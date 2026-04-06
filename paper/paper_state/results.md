# Results Registry: Where to Search (GAINS)

**Last Updated**: 2026-03-21

---

## Theoretical Results

### Assumption 1 (Verifier Smoothness) — `asm:smooth`
- **Location**: §4.3 (algorithm.tex)
- **Statement**: $\Psi_t(\cdot; c) \in C^2(\mathbb{R}^d)$ with $\|\nabla^2 \Psi_t(x;c)\|_{op} \le L_t$ in a neighborhood of $[\bar{X}_{t-h}, X_{t-h}]$
- **Used by**: Prop (Taylor), Theorem (Loc-Scale), Corollary (SDE alloc)

### Proposition (Score Taylor Expansion) — `prop:taylor`
- **Location**: §4.3 (algorithm.tex), proof inline
- **Statement**: $S_t = \Psi_t(\bar{X}_{t-h}; c) + g_t\sqrt{h}\langle\nabla\Psi_t(\bar{X}_{t-h}; c), \xi\rangle + R_t$ with $|R_t| \le \frac{1}{2}L_t g_t^2 h \|\xi\|^2$
- **Verified**: ✅ (2026-03-20)

### Theorem (Location-Scale Structure) — `thm:loc-scale` ⭐ Main Result
- **Location**: §4.3 (algorithm.tex), proof inline
- **Statement**: With $\mu_t := \Psi_t(\bar{X}; c)$, $\sigma_t := g_t\sqrt{h}\|\nabla\Psi_t(\bar{X}; c)\|$:
  - (i) $S_t = \mu_t + \sigma_t Z_t + R_t$, $Z_t \sim \mathcal{N}(0,1)$
  - (ii) $\text{Var}(S_t) = \sigma_t^2 + O(g_t^3 h^{3/2})$
  - (iii) $G_t(K) = \sigma_t a_K + O_K(g_t^2 h)$
- **Significance**: Derives location-scale from SDE, gives explicit $\sigma_t$ formula
- **Verified**: ✅ (2026-03-20)

### Corollary (SDE-based Approximate Allocation) — `cor:sde-alloc`
- **Location**: §4.3 (algorithm.tex)
- **Statement**: $\max_{\{K_t\}: \sum K_t=B} \sum_t g_t\sqrt{h}\|\nabla\Psi_t(\bar{X}; c)\| a_{K_t}$
- **Verified**: ✅ (2026-03-20, trivial)

### Proposition (Offline Optimality) — `prop:offline`
- **Location**: §4.4 (algorithm.tex), proof in App B
- **Statement**: Under prompt-independent $\sigma_t$:
  - (i) Optimal allocation universal across prompts
  - (ii) Water-filling: $K_t^*$ increasing in $\sigma_t$
  - (iii) Non-uniform strictly beats uniform when $\{\sigma_t\}$ heterogeneous
- **Verified**: ✅ (2026-03-20)

### Proposition (Online Adaptation) — `prop:online`
- **Location**: §4.4 (algorithm.tex), proof in App B
- **Statement**: Under prompt-dependent $\sigma_t(c, x_t)$:
  - (i) $V^*$ convex; Jensen gap $\ge 0$ for any fixed allocation
  - (ii) Marginal gain = $\sigma_t \cdot a_K'$; dual-threshold justified
- **Verified**: ✅ (2026-03-20)

### Assumption 2 (Local Search Regularity) — `asm:local-search`
- **Location**: §4.5 (general_local_search.tex)
- **Statement**: (A1) strict monotone improvement, (A2) non-accelerating marginals (marginals bounded above by non-increasing sequence), (A3) sensitivity scaling $G_t^{\mathcal{L}} = \sigma_t \phi_K^{\mathcal{L}} + O_{K,d,L_t}(g_t^2 h)$, (A4) rotational equivariance
- **Used by**: Theorem (General Gain), Cor (Offline General), Cor (Online General)

### Theorem (General Gain Factorization) — `thm:general-gain`
- **Location**: §4.5 (general_local_search.tex), proof inline
- **Statement**: Under asm:smooth + asm:local-search: $G_t^{\mathcal{L}}(K) = \sigma_t \phi_K^{\mathcal{L}} + O_{K,d,L_t}(g_t^2 h)$, with $\phi_K$ strictly increasing, non-accelerating marginals, $\phi_0=0$
- **Depends on**: asm:smooth, asm:local-search, prop:taylor
- **Verified**: ✅ (2026-03-23, uniform-ball alignment) — (A4) argument fully explicit; remainder $O_{K,d,L_t}$; operators updated to uniform-ball model

### Corollary (Offline Water-Filling, General) — `cor:offline-general`
- **Location**: §4.5 (general_local_search.tex), proof inline
- **Statement**: prop:offline holds with $\phi_K$ replacing $a_K$: water-filling (non-decreasing $K_t^*$ in $\sigma_t$; strictly increasing when $\phi_K$ strictly concave; bang-bang when linear)
- **Depends on**: thm:general-gain, prop:offline
- **Verified**: ✅ (2026-03-21)

### Corollary (Online Jensen Gap, General) — `cor:online-general`
- **Location**: §4.5 (general_local_search.tex), proof inline
- **Statement**: (i) $V^*_{\mathcal{L}}$ convex, Jensen gap holds; (ii) marginal = $\sigma_t (\phi_K)'$, dual-threshold valid
- **Depends on**: thm:general-gain, prop:online
- **Verified**: ✅ (2026-03-21)

### Proposition (Crossover) — `prop:crossover`
- **Location**: §4.5 (general_local_search.tex), proof inline
- **Statement**: ZO (LP) vs RS crossover at $K^* = \tilde{\Theta}(c_d(\lambda)^{-1}) = \tilde{\Theta}(\lambda^{-1})$ (for fixed $d$); ZO dominates for small $K$
- **Depends on**: ex:random-search, ex:local-perturbation
- **Verified**: ✅ (2026-03-23, re-verified) — Updated to uniform-ball model: $c_d(\lambda)=\frac{\lambda\sqrt{d}}{4\sqrt{\pi}}\frac{\Gamma(d/2)}{\Gamma((d+1)/2)}$; old $\tilde\Theta(d/\eta)$ formula removed; `rmk:crossover-full` added for full-problem caveat

---

## Algorithms

### Algorithm 1: GAINS (Global Adaptive Inference-time Noise Scheduling)
- **Location**: §4.2 (algorithm.tex)
- **Input**: Timesteps, budget $B$, offline $\{K_t\}$, slack $\delta$, window $W_g$, thresholds $(\beta_g, \beta_\sigma)$
- **Features**: Offline coarse + online early stopping + revert-on-negative
- **Budget guarantee**: Exact NFE = $B$ via redistribution

---

## Experimental Results

### Table 1: SD NFE Scaling
| NFE | Naive | GAINS | Savings |
|-----|-------|-------|---------|
| 400 Brightness | 0.7025 | 0.7248 | GAINS@400 > Naive@500 → 20% fewer |
| 800 Compressibility | 0.8892 | 0.9008 | GAINS@400 > Naive@800 → 50% fewer |

### Table 2: EDM NFE Scaling
| NFE | Naive | GAINS | Savings |
|-----|-------|-------|---------|
| 144 Brightness | 0.9507 | 0.9887 | GAINS@144 > Naive@288 → 50% fewer |
| 180 Compressibility | 0.6774 | 0.7003 | GAINS@180 > Naive@288 → 37.5% fewer |

### Table 3: Ablation (SD, NFE=400)
- Offline Only > Naive (coarse budgeting helps)
- GAINS > Offline Only (online control adds value)

### Table 4: Larger Prompt Set (20 prompts × 10 repeats)
- Robust improvement: Brightness +0.0264, Compressibility +0.0138

### Table 5: Local Search Operator Compatibility
- GAINS improves both Zero-order and Random local search
- Confirms modularity of two-level design

### Table 6: Flow-based Models
- GAINS@80 > Naive@100 (Brightness) → 20% fewer
- GAINS@100 > Naive@150 (Compressibility) → 33% fewer
- Margins smaller (flow models less noise-sensitive) but consistent
