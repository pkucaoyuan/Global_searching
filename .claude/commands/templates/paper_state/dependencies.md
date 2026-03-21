# Dependency Registry: [Paper Title]

**Last Updated**: [date]
**Purpose**: Track what depends on what - so changes propagate correctly

---

## Assumption → Result Dependencies

**CRITICAL**: Before removing/changing an assumption, check what breaks.

### Assumption 3.1: Bounded Outcomes (Y, F ∈ [0,1])

**Used By**:
| Result | Section | How Used |
|--------|---------|----------|
| Thm 4.1 (CS validity) | analysis | Bounded martingale |
| Thm 5.1 (δ-correct) | algorithm | Concentration bound |
| Thm 5.2 (Neyman) | algorithm | Variance bound |
| Thm 7.1 (Lower bound) | theory_lb | Information bound |

**If removed**: ALL main theorems fail. Cannot remove.

---

### Assumption 3.2: MAR (Missing at Random)

**Used By**:
| Result | Section | How Used |
|--------|---------|----------|
| Lem 4.1 (IPW unbiased) | method | E[IPW] = E[Y] |
| Thm 5.1 (δ-correct) | algorithm | Unbiased estimation |

**If removed**: Need MNAR correction. Major revision.

---

### Assumption 3.5: LAN (Local Asymptotic Normality)

**Used By**:
| Result | Section | How Used |
|--------|---------|----------|
| Thm 7.1 (Lower bound) | theory_lb | Information lower bound |
| Prop 7.2 (Decomposition) | theory_lb | Fisher information split |

**If removed**: Lower bound section needs rewrite. Can weaken to "regularity".

---

## Result → Result Dependencies

### Dependency Graph

```
Assumption 3.1 (Bounded)
    │
    ├──→ Lemma 4.1 (IPW unbiased)
    │        │
    │        ├──→ Theorem 5.1 (δ-correct)
    │        │        │
    │        │        └──→ Theorem 6.1 (Cost bound)
    │        │
    │        └──→ Theorem 5.2 (Neyman optimal)
    │                 │
    │                 └──→ Theorem 7.1 (Lower bound)
    │
    └──→ Lemma 4.2 (CS validity)
             │
             └──→ Theorem 5.1 (δ-correct)

Assumption 3.5 (LAN)
    │
    └──→ Theorem 7.1 (Lower bound)
             │
             └──→ Proposition 7.2 (Decomposition)
                      │
                      └──→ Theorem 7.3 (Optimality)
```

### Impact Analysis

| If You Change... | Then Check... |
|------------------|---------------|
| Lemma 4.1 | Thm 5.1, Thm 5.2 |
| Lemma 4.2 | Thm 5.1 |
| Thm 5.1 | Thm 6.1 |
| Thm 5.2 | Thm 7.1 |
| Thm 7.1 | Prop 7.2, Thm 7.3 |

---

## Claim → Evidence Dependencies

**Every claim needs supporting evidence.**

### Main Claims

| Claim | Evidence Type | Evidence Location | Table/Figure |
|-------|--------------|-------------------|--------------|
| "90% cost savings" | Experiment | exp_tickets.py | Table 2, Row 4 |
| "100% accuracy" | Experiment | exp_tickets.py | Table 2 |
| "48% cost reduction" | Experiment | exp3_neyman.py | Table 1 |
| "98.8% coverage" | Experiment | exp6_cs.py | Table 1 |
| "Judge-only fails" | Experiment | exp1_failure.py | Figure 2 |
| "O(1/Δ²) scaling" | Theory + Exp | Thm 7.1 + exp4 | Figure 4 |

### Evidence Chain

```
Claim: "Our method achieves 90% cost savings with no accuracy loss"
   │
   ├── Evidence 1: Table 2, Row "fixed_5%"
   │      └── Accuracy: 100%, Audit Rate: 5.4%
   │
   ├── Evidence 2: Comparison baseline
   │      └── fixed_50%: 100% accuracy, 51.2% audit rate
   │
   └── Calculation: (51.2 - 5.4) / 51.2 = 89.5% ≈ 90%
```

---

## Symbol → Location Dependencies

**Where each symbol is defined and used.**

### Symbol: π (audit probability)

| Type | Location | Line |
|------|----------|------|
| **Definition** | model.tex | L20 |
| Usage | method.tex | L15, L30, L45 |
| Usage | algorithm.tex | L25, L80, L120 |
| Usage | analysis.tex | L10, L55 |
| Usage | theory_lb.tex | L30, L70, L95 |

**If definition changes**: Check ALL 12 usage locations

---

### Symbol: k̂(t) (estimated best arm)

| Type | Location | Line |
|------|----------|------|
| **Definition** | algorithm.tex | L50 |
| Usage | algorithm.tex | L55, L60, L75 |
| Usage | analysis.tex | L40 |

**Conflict check**: Does NOT conflict with b_k (bias) ✅

---

## Example → Result Pairing

| Example | Illustrates | Distance | Status |
|---------|-------------|----------|--------|
| Ex 3.1 (Service) | Def 3.1 (Arms) | Same section | ✅ |
| Ex 5.1 (Gaussian) | Thm 5.2 (Neyman) | Same section | ✅ |
| Ex 7.1 (Beta) | Thm 7.1 (LAN) | Same section | ✅ |
| EC.1 (Call center) | Problem setup | Should be in Intro | ⚠️ Move |

---

## Update Protocol

Before ANY change:
1. Check this dependency registry
2. Identify all downstream effects
3. Update ALL affected locations
4. Verify no broken references
5. Update this registry
