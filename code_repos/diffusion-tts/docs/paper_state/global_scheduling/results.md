# Results Registry: Global Scheduling of Noise Trajectory Search

**Last Updated**: 2026-03-08
**Total Results**: 0 theorems, 1 algorithm, 6 tables

---

## Main Algorithm

| ID | Label | Section | Description | Status |
|----|-------|---------|-------------|--------|
| Alg 1 | `alg:offline_online_budget` | 3.2 | Offline-to-Online Budget Scheduling (Windowed Early Stop) | ✅ Complete |

---

## Key Equations

| Eq # | Content | Section | Referenced In |
|------|---------|---------|---------------|
| (1) | x_{t-1} = F_θ(x_t, t, c, ε_t) | 3.1 | Throughout |
| - | L_t operator definition | 3.1 | 3.1, 3.2 |
| - | G scheduler definition | 3.1 | 3.1, 3.2 |
| - | g_t^{(j)} = s_t^{(j)} - s_t^{(j-1)} | 3.2 | Alg 1 |
| - | Var_t^{(j)} definition | 3.2 | Alg 1 |

---

## Experimental Results (Tables)

| Table | Label | Section | Content | Key Finding |
|-------|-------|---------|---------|-------------|
| 1 | `tab:sd_budget` | 4.2 | SD results across NFE budgets | 20-50% NFE savings |
| 2 | `tab:edm_budget` | 4.3 | EDM results across NFE budgets | 37.5-50% NFE savings |
| 3 | `tab:sd_400_ablation` | 4.4 | Offline vs Offline+Online ablation | Online control adds +0.009 brightness |
| 4 | `tab:larger_prompt_exp` | 4.5 | Larger prompt set (20 prompts × 10 repeats) | Consistent improvement |
| 5 | `tab:local_search_combo` | 4.6 | Different local search operators | Scheduler is modular |
| 6 | `tab:flow_results` | 4.7 | Flow-based model (DDIM ODE) | Works with deterministic samplers |

---

## Result Dependencies

```
Two-Level Framework (sec:method-global)
    ↓
Local Operator L_t (abstraction)
    ↓
Global Scheduler G (formulation)
    ↓
Offline Profiling (coarse allocation)
    ↓
Online Control (fine-grained adaptation)
    ↓
Algorithm 1 (complete procedure)
```

---

## Validation Status

| Result | Description | Experimentally Validated | Key Metric |
|--------|-------------|--------------------------|------------|
| SD Scaling | Global scheduling improves SD | ✅ Table 1 | +0.032 brightness @ 800 NFE |
| EDM Scaling | Global scheduling improves EDM | ✅ Table 2 | +0.020 brightness @ 288 NFE |
| NFE Savings | Achieve same quality with fewer NFE | ✅ Tables 1-2 | 20-50% reduction |
| Ablation | Online control adds value over offline-only | ✅ Table 3 | +0.009/+0.007 |
| Robustness | Works across prompts | ✅ Table 4 | Consistent gains |
| Modularity | Works with different local operators | ✅ Table 5 | Both zero-order and random |

---

## Key Claims (from Abstract/Intro)

| Claim | Evidence | Section |
|-------|----------|---------|
| Two-level framework unifies existing methods | Conceptual analysis | 3.1 |
| Offline profiling captures step importance | SD vs EDM patterns differ | 3.2 |
| Online control adapts per-instance | Ablation shows improvement | 4.4 |
| 20-50% NFE savings at matched quality | Tables 1-2 comparisons | 4.2-4.3 |
