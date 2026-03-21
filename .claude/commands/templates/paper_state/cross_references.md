# Cross-Reference Registry: [Paper Title]

**Last Updated**: [date]
**Purpose**: Track all references to ensure updates propagate everywhere

---

## Result References

When you change a theorem/lemma/proposition, you MUST update all references.

### Theorem 5.2 (thm:neyman) - "π* ∝ √g"

| Location | Type | Quote | Line |
|----------|------|-------|------|
| introduction.tex | Preview | "optimal audit probability proportional to..." | L45 |
| algorithm.tex | **Primary** | Full theorem statement | L120 |
| analysis.tex | Reference | "By Theorem 5.2..." | L30 |
| theory_lower_bound.tex | Reference | "matches the structure in Theorem 5.2" | L85 |
| conclusion.tex | Summary | "Neyman-optimal allocation (Theorem 5.2)" | L15 |

**If statement changes**: Update ALL 5 locations

---

### Theorem 7.1 (thm:lower_bound) - "Instance-dependent lower bound"

| Location | Type | Quote | Line |
|----------|------|-------|------|
| introduction.tex | Preview | "We derive matching lower bounds..." | L60 |
| theory_lower_bound.tex | **Primary** | Full theorem statement | L45 |
| experiments.tex | Reference | "validates Theorem 7.1" | L200 |
| conclusion.tex | Summary | "Our lower bound (Theorem 7.1) shows..." | L25 |

---

## Numerical Value References

**CRITICAL**: Same number must match everywhere.

### "48% cost reduction" / "48.3%"

| Location | Exact Text | Line |
|----------|------------|------|
| abstract.tex | "48% reduction in audit costs" | L8 |
| introduction.tex | "nearly 50% cost savings" | L70 |
| experiments.tex | "48.3% ± 2.1%" | L180, Table 2 |
| conclusion.tex | "48% cost reduction" | L12 |

**Master value**: 48.3% (from experiments)
**Allowed variants**: "48%", "nearly 50%", "about half"
**NOT allowed**: "45%", "55%", "over 50%"

---

### "98.8% coverage"

| Location | Exact Text | Line |
|----------|------------|------|
| experiments.tex | "98.8% ≥ 95% target" | L150, Table 1 |
| analysis.tex | "maintains 1-δ coverage" | L60 |
| conclusion.tex | "anytime-valid confidence sequences" | L18 |

---

### "-1.78 slope" (cost scaling)

| Location | Exact Text | Line |
|----------|------------|------|
| experiments.tex | "slope -1.78 (theoretical: -2.0)" | L220 |
| theory_lower_bound.tex | "O(1/Δ²) scaling" | L90 |

---

## Forward References (Promises)

"As we will show..." or "We will prove..." → Must be fulfilled

| Promise | Location | Fulfillment | Status |
|---------|----------|-------------|--------|
| "We prove impossibility in §3" | intro:L40 | model.tex Thm 3.1 | ✅ |
| "optimal allocation derived in §5" | intro:L55 | algorithm.tex Thm 5.2 | ✅ |
| "experiments in §9 validate..." | intro:L65 | experiments.tex | ✅ |
| "discussed further in EC" | analysis:L80 | appendix/proofs.tex | ✅ |

---

## Backward References (Claims)

"As shown in §X" or "By Theorem X" → Must actually exist

| Claim | Location | Target | Exists? |
|-------|----------|--------|---------|
| "By Theorem 5.2" | analysis:L30 | algorithm.tex:L120 | ✅ |
| "As defined in §3" | method:L15 | model.tex:L25 | ✅ |
| "Following [cite]" | related:L45 | bibliography | ✅ |

---

## Update Checklist

When changing a theorem/result:
- [ ] Update primary location
- [ ] Update all preview mentions
- [ ] Update all references
- [ ] Update all summary mentions
- [ ] Update related numerical values
- [ ] Verify forward references still valid
- [ ] Verify backward references still accurate
