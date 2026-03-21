# Paper Overview - Living Document for Paper State

You are a paper documentation agent. Your task is to create and maintain a structured overview document that captures the **current state** of the paper.

## Why This Matters

Without this document:
```
polish-paper → "Looks clean" → Reviewer: "14 comments"
            → No record of what each section does
            → No record of symbol definitions
            → No record of section relationships
            → Blind polishing in circles
```

With this document:
```
paper-overview → Creates docs/paper_state/[paper_name].md
              → Records: WHAT each section does
              → Records: HOW it does it (key concepts, symbols, results)
              → Records: CONNECTIONS between sections
              → Any change → Update document → Verify consistency
```

## ⚠️ Protocol Reference

This command creates/updates `overview.md`. After any state doc modification, you MUST update `changelog.md` with a dated entry. See `.claude/commands/_shared/unified_protocol.md` Step 4 (Universal Post-Edit Rule).

## Arguments

- `$ARGUMENTS` - Action: `create`, `update`, `verify`, or path to paper directory

## The Paper State Document

Location: `docs/paper_state/[paper_name]_overview.md`

This document is the **single source of truth** for the paper's structure.

## Document Structure

```markdown
# Paper Overview: [Title]

**Last Updated**: [date]
**Target Journal**: [journal]
**Status**: Draft / Under Review / Revision

---

## 1. One-Sentence Summary

[What is this paper about in ONE sentence?]

---

## 2. Section Map

### Section 1: Introduction
**What**: [What does this section accomplish?]
**Key Claims**:
- Claim 1: [quote or paraphrase]
- Claim 2: ...

**Key Concepts Introduced**:
| Concept | Definition | Symbol |
|---------|------------|--------|
| Arms | Service configurations | k ∈ [K] |
| Judge | Automated evaluator | F |
| Audit | Human review | Y |

**Promises Made** (to be fulfilled later):
- [ ] "We prove impossibility" → Section 3
- [ ] "We develop algorithm" → Section 5

---

### Section 2: Related Work
**What**: [Purpose]
**Positioning**: How does this paper differ from prior work?
**Key Distinctions**:
| Prior Work | Their Focus | Our Difference |
|------------|-------------|----------------|
| [cite] | X | We do Y |

---

### Section 3: Model
**What**: [Purpose]
**Key Definitions**:
| Term | Symbol | Definition | First Defined |
|------|--------|------------|---------------|
| Bias | b_k(x) | E[F|k,x] - E[Y|k,x] | Line 8 |
| ... | ... | ... | ... |

**Assumptions**:
| Assumption | Label | Used In |
|------------|-------|---------|
| Bounded outcomes | ass:bounded | Thm 5.1, 6.1 |
| MAR | ass:mar | Thm 5.1 |

**Results in This Section**:
| Result | Label | Core Claim |
|--------|-------|------------|
| Theorem 3.1 | thm:mf-failure | Judge-only fails |

---

### Section 4: Estimation
**What**: [Purpose]
**Key Equations**:
| Equation | Number | Purpose |
|----------|--------|---------|
| IPW estimator | (5) | Debias selective audits |

**Builds On**: Section 3 (uses bias definition)
**Used By**: Section 5 (algorithm uses estimator)

---

### Section 5: Algorithm
**What**: [Purpose]
**Algorithm Name**: PP-LUCB
**Key Components**:
1. Outer loop: [what it does]
2. Inner loop: [what it does]

**Key Results**:
| Result | Label | Core Claim | Proof Location |
|--------|-------|------------|----------------|
| Theorem 5.1 | thm:delta-correct | δ-correctness | Appendix B |
| Theorem 5.2 | thm:neyman | π* ∝ √g | Appendix C |

**Symbol Alert**:
- ⚠️ b(t) used here for "best arm" - conflicts with b_k (bias) in Section 3?

---

### Section 6: Analysis
**What**: [Purpose]
**Key Results**:
| Result | Label | Core Claim |
|--------|-------|------------|

**Redundancy Check**:
- [ ] Any result here also stated in Section 5?
- [ ] Any result here also stated in Section 7?

---

### Section 7: Lower Bounds
**What**: [Purpose]
**Key Results**:
| Result | Label | Core Claim |
|--------|-------|------------|

**Redundancy Check**:
- Does π* ∝ √g appear here AND in Section 5?
  - Section 5: [quote]
  - Section 7: [quote]
  - Action: [consolidate / cross-reference / keep both because...]

---

### Section 8: Delays
**What**: [Purpose]
**Key Results**: ...

---

### Section 9: Experiments
**What**: [Purpose]
**Experiments**:
| Experiment | Arms Meaning | Business Context |
|------------|--------------|------------------|
| MT-Bench | LLM models only | ⚠️ Too narrow? |
| Support Tickets | LLM models | ⚠️ Missing operational layer? |

**Framing Check**:
- [ ] Arms defined broadly in experiments?
- [ ] Business context clear?
- [ ] Results have managerial interpretation?

---

### Section 10: Discussion/Conclusion
**What**: [Purpose]
**Insights Provided**:
1. [Insight 1] - Actionable? [Y/N]
2. [Insight 2] - Actionable? [Y/N]

**MS Style Check**:
- [ ] Addresses "managers/practitioners"?
- [ ] Provides decision guidance?

---

## 3. Symbol Registry

| Symbol | Meaning | First Defined | Sections Used |
|--------|---------|---------------|---------------|
| K | Number of arms | Sec 3, L5 | 3,4,5,6,7,8,9 |
| k | Arm index | Sec 3, L5 | All |
| b_k(x) | Bias function | Sec 3, L8 | 3,7 |
| b(t) | Best arm at t | Sec 5, L12 | 5 only |
| F | Judge score | Sec 3, L10 | All |
| Y | Human label | Sec 3, L11 | All |
| π | Audit probability | Sec 3, L15 | 3,4,5,6,7 |
| g(x,f) | Residual variance | Sec 5, L30 | 5,7 |

**Conflicts Detected**:
- ⚠️ b_k vs b(t): Different meanings, same letter

---

## 4. Result Registry

| Result | Label | Section | Core Claim | Also Stated In |
|--------|-------|---------|------------|----------------|
| Thm 3.1 | thm:mf-failure | 3 | Judge-only fails | - |
| Thm 5.1 | thm:delta-correct | 5 | δ-correctness | - |
| Thm 5.2 | thm:neyman | 5 | π* ∝ √g | Sec 7.2, Prop 7.3 ⚠️ |
| ... | ... | ... | ... | ... |

**Redundancy Issues**:
- ⚠️ π* ∝ √g stated in: Thm 5.2, Sec 7.2 para, Prop 7.3(c)

---

## 5. Section Dependencies

```
Introduction
    ↓ (promises)
Model ←──────────────────┐
    ↓ (defines symbols)   │
Estimation                │
    ↓ (provides estimator)│
Algorithm                 │
    ↓ (uses estimator)    │
Analysis ────────────────→│ (both discuss cost)
    ↓                     │
Lower Bounds ─────────────┘
    ↓
Delays
    ↓
Experiments
    ↓
Conclusion (summarizes all)
```

---

## 6. Consistency Checklist

### Symbols
- [ ] No symbol conflicts (same symbol, different meanings)
- [ ] All symbols defined before use
- [ ] Notation consistent across sections

### Results
- [ ] Each major result stated in ONE primary location
- [ ] Other mentions are references, not restatements
- [ ] Examples near their general theorems

### Framing
- [ ] Arms defined broadly (operations + technology + parameters)
- [ ] Experiments have business context
- [ ] Discussion provides managerial insights

### Claims
- [ ] Introduction claims match conclusion
- [ ] Numerical values consistent across sections

---

## 7. Action Items

### Must Fix
1. [ ] [Issue from checklist]
2. [ ] ...

### Should Fix
1. [ ] ...

### Verify After Changes
- [ ] Update this document after any structural change
- [ ] Re-verify consistency checklist
```

## Workflow

### `create` - Initial Documentation

1. Read all .tex files
2. For each section:
   - Extract WHAT it does
   - Extract KEY concepts/symbols/results
   - Note PROMISES made
3. Build symbol registry
4. Build result registry
5. Draw section dependency graph
6. Run consistency checklist
7. Generate action items

### `update` - After Changes

1. Read the existing overview document
2. Read changed .tex files
3. Update affected sections in overview
4. Re-run consistency checklist
5. Update action items

### `verify` - Consistency Check

1. Read overview document
2. Read all .tex files
3. Verify:
   - Symbol registry matches actual usage
   - Result registry matches actual theorems
   - No new redundancies introduced
   - Framing still consistent
4. Report discrepancies

## Output

Creates/updates: `docs/paper_state/[paper_name]_overview.md`

## Integration with Other Skills

```
/paper-overview create    → Creates initial state document
     ↓
/review-paper-full MS     → Uses state document for context
     ↓
[Make changes to paper]
     ↓
/paper-overview update    → Updates state document
     ↓
/paper-overview verify    → Checks consistency
```

## Begin

Based on `$ARGUMENTS`:

- `create`: Build full overview from scratch
- `update`: Update existing overview with recent changes
- `verify`: Check current paper against overview
- `[path]`: Create overview for paper at specified path
