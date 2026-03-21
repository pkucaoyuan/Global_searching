# Abbreviation & Acronym Registry: [Paper Title]

**Last Updated**: [date]
**Purpose**: Ensure all abbreviations are defined before use, and used consistently

---

## Abbreviations

| Abbrev | Full Form | First Defined | Re-defined In | Status |
|--------|-----------|---------------|---------------|--------|
| BAI | Best Arm Identification | intro:L15 | - | ✅ |
| MAB | Multi-Armed Bandit | intro:L12 | - | ✅ |
| LLM | Large Language Model | abstract:L3 | - | ✅ |
| IPW | Inverse Propensity Weighting | method:L20 | - | ✅ |
| DR | Doubly Robust | method:L45 | - | ✅ |
| CS | Confidence Sequence | method:L30 | - | ✅ |
| MAR | Missing At Random | model:L40 | - | ✅ |
| LAN | Local Asymptotic Normality | theory_lb:L25 | - | ✅ |
| EC | Electronic Companion | intro:L80 | - | ✅ |

---

## Rule: First Use Must Define

**Pattern**: "Best Arm Identification (BAI)" on first use, then "BAI" thereafter.

### Violations to Fix

| Abbrev | First Occurrence | Defined? | Action |
|--------|------------------|----------|--------|
| LUCB | algorithm:L5 | ❌ | Add "(Lower/Upper Confidence Bound)" |
| PP | algorithm:L50 | ❌ | Add "(Price of Precision)" |

---

## Section-by-Section First Use

### Abstract
- LLM ✅ (defined)
- BAI ❌ (not defined - too short for abstract, spell out)

### Introduction
- LLM ✅ (re-use from abstract)
- BAI ✅ (defined L15)
- MAB ✅ (defined L12)

### Model
- MAR ✅ (defined L40)

### Method
- IPW ✅ (defined L20)
- CS ✅ (defined L30)
- DR ✅ (defined L45)

### Algorithm
- LUCB ❌ (need definition)
- PP ❌ (need definition)

### Theory Lower Bound
- LAN ✅ (defined L25)

---

## Consistency Check

Same abbreviation must mean same thing throughout.

| Abbrev | Meaning 1 | Meaning 2 | Conflict? |
|--------|-----------|-----------|-----------|
| CS | Confidence Sequence | - | ✅ No conflict |
| DR | Doubly Robust | - | ✅ No conflict |
| PP | Price of Precision | - | ✅ No conflict |

---

## Mathematical Notation Shortcuts

| Notation | Meaning | Defined |
|----------|---------|---------|
| i.i.d. | independent and identically distributed | Assumed known |
| a.s. | almost surely | Assumed known |
| w.p. | with probability | Assumed known |
| s.t. | such that / subject to | Assumed known |
| w.r.t. | with respect to | Assumed known |
| iff | if and only if | Assumed known |

---

## Journal-Specific Requirements

### Management Science
- Spell out on first use, even common ones
- Avoid excessive abbreviations (reader-unfriendly)
- EC for Electronic Companion (standard)

### NeurIPS/ICML
- More abbreviations acceptable
- Define technical ones (IPW, DR, CS)
- Standard ML terms (i.i.d., MLP, SGD) often assumed

### Operations Research
- Spell out for clarity
- OR-specific terms defined (LP, MIP, TSP)

---

## Checklist Before Submission

- [ ] Every abbreviation defined on first use
- [ ] No abbreviation used before definition
- [ ] Abstract: minimal abbreviations (or spelled out)
- [ ] Conclusion: can use abbreviations (reader knows by now)
- [ ] No conflicting meanings
- [ ] Journal-appropriate density
