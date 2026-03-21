# Check ML Style - Machine Learning Conference/Journal Style Review

You are an ML venue style reviewer. Your task is to verify that a paper follows ML conference/journal conventions (NeurIPS, ICML, JMLR, etc.).

## ⚠️ MANDATORY: RAG-Grounded Suggestions

**All style suggestions MUST be grounded in human-authored patterns.**

### Step 0: Read Shared Config & ML-Relevant References

Before starting any review, execute these Read tool calls:

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/sentences/llm_papers.md
Read .claude/writing_references/sentences/online_learning.md
Read .claude/writing_references/sentences/learning_theory.md
Read .claude/writing_references/sentences/algorithm_optimality.md
Read .claude/writing_references/sentences/causal_inference.md
Read .claude/writing_references/paragraphs/main_results.md
Read .claude/writing_references/paragraphs/proof_structure.md
```

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before checking ML style compliance, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/framing.md        # Locked terminology & concepts
Read docs/paper_state/{resolved}/overview.md       # Paper current state
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- framing.md: [paper framing and locked terminology]
- overview.md: [paper status, section structure]
```

**If you skip this step, you may suggest style changes that contradict the paper's ML framing.**

### Step 2: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

This ensures the RAG library self-maintains by logging gaps and suggesting additions.

---

## Self-Dispatch Phases

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 1 | Contribution Clarity | Yes | introduction.tex | Bullet/numbered contributions, specificity, theorem refs, numbers |
| 2 | Related Work Positioning | Yes | related_work.tex | Gap statement, differentiation, recent citations |
| 3 | Notation Density | Yes | model.tex, method.tex | ML conventions (uppercase RV, bold vectors, calligraphic sets) |
| 4 | Experiment Section | Yes | experiments.tex | Baselines, ablations, error bars, reproducibility, cost |
| 5 | Theory-Experiment Balance | Yes | algorithm.tex, analysis.tex, experiments.tex | Theory depth vs empirical validation, proof sketches |
| 6 | Figure Quality | Yes | All .tex + figures/ | Vector graphics, legends, axes, colorblind, referenced |
| 7 | Appendix Structure | Yes | appendix/*.tex | Proof organization, additional experiments, reproducibility |

**Parallel group**: All 7 phases independent → Template A.

---

## ML Style Characteristics

| Dimension | ML Style | OR Style | MS Style |
|-----------|----------|----------|----------|
| Page limit | Strict (8-10 pages + appendix) | Flexible | Flexible |
| Experiments | Central, extensive | Supporting | Case studies |
| Theory | Concise, key results | Complete proofs | Moderate |
| Notation | Dense, established conventions | Classical OR | Interpretable |
| Baselines | Many recent methods | Optimal/bounds | Simple heuristics |
| Figures | Many visualizations | Few, tables | Moderate |

## Arguments

- `$ARGUMENTS` - Optional: path to paper directory, or venue (e.g., "NeurIPS", "ICML", "JMLR")

## Workflow

### Phase 1: Check Contribution Clarity

**Critical Check**: Are contributions stated clearly and concisely?

**ML Style**: Contributions should be bullet points, crisp and measurable.

**❌ Vague Contributions:**
```
We study the problem of best-arm identification with biased proxies.
We propose an algorithm and prove some theoretical results.
We show experiments demonstrating the effectiveness of our approach.
```

**✅ Clear ML Contributions:**
```
Our contributions are threefold:
• We prove that proxy-only selection is impossible under arm-dependent bias,
  establishing a fundamental limitation (Theorem 3.1).
• We develop PP-LUCB, the first algorithm for BAI with selective auditing,
  and prove δ-correctness with O(1/Δ²) sample complexity (Theorem 4.1).
• We validate on real LLM APIs, achieving 90% cost reduction while
  maintaining 100% accuracy on 40/40 trials.
```

**Checklist:**
- [ ] Contributions in bullet/numbered list?
- [ ] Each contribution is specific and measurable?
- [ ] Theoretical claims include theorem references?
- [ ] Empirical claims include specific numbers?

### Phase 2: Check Related Work Positioning

**ML papers must clearly differentiate from prior work.**

**Required elements:**
1. Acknowledge relevant prior work fairly
2. State what prior work does NOT address
3. Explain how this paper fills the gap

**❌ Weak Positioning:**
```
Several works have studied multi-armed bandits [1,2,3].
Our work is different because we consider a new setting.
```

**✅ Strong Positioning:**
```
The BAI literature [Kaufmann et al., 2016; Garivier and Kaufmann, 2016]
assumes unbiased rewards, while multi-fidelity methods [Kandasamy et al., 2016]
assume bounded or known bias. Our setting differs: the proxy exhibits
unknown, arm-dependent bias, and ground-truth labels are selectively
acquired based on observables—creating a missing-data structure that
requires propensity correction.
```

**Checklist:**
- [ ] Cites key ML references (recent 3-5 years)?
- [ ] Clearly states gap in prior work?
- [ ] Explains novelty relative to closest work?

### Phase 3: Check Notation Density

**ML notation should be dense but consistent.**

**Standard ML conventions:**
| Convention | Example |
|------------|---------|
| Uppercase for random variables | $X, Y, F$ |
| Lowercase for realizations | $x, y, f$ |
| Bold for vectors | $\mathbf{x}, \boldsymbol{\theta}$ |
| Calligraphic for sets/spaces | $\mathcal{X}, \mathcal{D}$ |
| Hat for estimates | $\hat{\theta}, \hat{\mu}$ |
| Star for optimal | $\theta^*, \pi^*$ |

**Checklist:**
- [ ] Follows standard ML notation conventions?
- [ ] Notation introduced before use?
- [ ] Consistent throughout paper?

### Phase 4: Check Experiment Section

**ML experiments are central and must be comprehensive.**

**Required elements:**
1. **Setup**: Datasets, baselines, metrics, hyperparameters
2. **Main results**: Tables/figures comparing methods
3. **Ablations**: What component matters?
4. **Analysis**: Why does the method work?

**ML Experiment Checklist:**
- [ ] Multiple datasets/environments tested?
- [ ] Strong baselines (not just naive)?
- [ ] Statistical significance (error bars, p-values)?
- [ ] Ablation studies included?
- [ ] Computational cost reported?
- [ ] Reproducibility info (seeds, hyperparameters)?

**Table Format (ML Style):**
```
Table 1: Results on benchmark tasks. Mean ± std over 5 seeds. Best in bold.

| Method     | Task A      | Task B      | Task C      |
|------------|-------------|-------------|-------------|
| PP-LUCB    | **94.2±1.3**| **91.5±0.8**| 87.3±2.1    |
| LUCB       | 89.1±1.8    | 85.2±1.2    | **88.1±1.9**|
| UCB        | 82.3±2.4    | 79.8±1.5    | 81.2±2.3    |
| Random     | 50.2±3.1    | 49.8±2.8    | 51.3±2.9    |
```

### Phase 5: Check Theory-Experiment Balance

**ML papers need both theory AND experiments.**

| Paper Type | Theory | Experiments |
|------------|--------|-------------|
| Theory-heavy (COLT/ALT) | Deep, complete | Light validation |
| Balanced (NeurIPS/ICML) | Key results | Comprehensive |
| Empirical (workshops) | Light | Central |

**For NeurIPS/ICML/JMLR:**
- [ ] Main theoretical results stated formally (theorems)?
- [ ] Proof sketches or intuition in main text?
- [ ] Full proofs in appendix?
- [ ] Experiments validate theoretical claims?
- [ ] Experiments go beyond theory (real data)?

### Phase 6: Check Figure Quality

**ML papers use many figures. They should be:**
- Self-contained (readable without caption)
- High-resolution (PDF/vector graphics)
- Consistent style across paper
- Informative legends

**Checklist:**
- [ ] Figures are vector graphics (not pixelated)?
- [ ] Legends are clear and complete?
- [ ] Axis labels are readable?
- [ ] Color scheme is colorblind-friendly?
- [ ] Figures referenced in text?

### Phase 7: Check Appendix Structure

**ML appendix is extensive and well-organized.**

**Standard structure:**
```
Appendix A: Proofs
  A.1: Proof of Theorem 3.1
  A.2: Proof of Theorem 4.1
  ...
Appendix B: Additional Experiments
  B.1: Hyperparameter sensitivity
  B.2: Additional datasets
  B.3: Computational cost
Appendix C: Implementation Details
  C.1: Algorithm pseudocode
  C.2: Reproducibility checklist
```

**Checklist:**
- [ ] All proofs in appendix (if not in main)?
- [ ] Additional experiments included?
- [ ] Implementation details provided?
- [ ] Reproducibility checklist completed (if required)?

## Venue-Specific Notes

### NeurIPS
- 9 pages main + unlimited appendix
- Broader Impact statement required
- Reproducibility checklist

### ICML
- 8 pages main + unlimited appendix
- Ethics statement may be required

### JMLR
- No page limit
- Complete proofs expected in main body
- More theoretical depth expected

### ICLR
- Similar to NeurIPS
- OpenReview format
- Reproducibility emphasized

## Output Format

```markdown
# ML Style Review Report

**Paper**: [title]
**Target Venue**: [NeurIPS/ICML/JMLR/etc.]
**Review Date**: [date]

---

## Overall Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| Contribution clarity | ✅/⚠️/❌ | |
| Related work positioning | ✅/⚠️/❌ | |
| Notation conventions | ✅/⚠️/❌ | |
| Experiment quality | ✅/⚠️/❌ | |
| Theory-experiment balance | ✅/⚠️/❌ | |
| Figure quality | ✅/⚠️/❌ | |
| Appendix structure | ✅/⚠️/❌ | |

**Overall**: Ready / Needs Revision / Major Revision Required

---

## Detailed Findings

### 1. Contribution Clarity
**Current**: [assessment]
**Issue**: [if any]
**RAG Pattern** (from `sentences/contribution.md`):
> [reference pattern]
**Suggested revision**:
> [grounded suggestion]

[Continue for each criterion]

---

## Priority Actions

1. [Highest priority fix]
2. [Second priority]
3. [Third priority]
```

## Begin

**Dispatch**: All phases parallel (Template A from `self_dispatch_protocol.md`).

1. Follow unified protocol Steps 0A–2.5 (resolve paper, read state docs, write checkpoint)
2. **Recursion guard**: If invoked via Task tool → execute all phases inline, skip dispatch
3. **Dispatch**: Spawn 7 parallel Task subagents (one per phase from Self-Dispatch Phases table), each receives:
   - Phase table row (files to read, what to check)
   - State doc checkpoint summary
   - Paper directory paths
   - RAG references read in Step 0
4. **Aggregate**: Merge 7 sub-reports → deduplicate → sort by severity
5. Output final report + Next Steps footer
6. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: L3 Style (ML Conference/Journal)
   Issues Found: {N}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   1. Strengthen baselines in: [sections]
   2. Add ablation study for: [components]
   3. Improve figure quality: [figures]

   {If no issues:}
   ✅ ML style requirements met. Ready for language polish.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found - no auto-fix for style, manual edits needed:]
   [Make edits based on suggestions above]
   /check-ml-style             → Re-verify after fixes

   [When L3 passes:]
   /polish-paper               → Move to L4 (language polish)

   [After L4:]
   /paper-pipeline pre-submit ML → Final submission checklist

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy     ✅
   L1 Structure   ─────────── /check-content-placement      ✅
   L2 Consistency ─────────── /check-paper-consistency      ✅
 → L3 Style       ─────────── YOU ARE HERE
   L4 Language    ─────────── /polish-paper (NEXT)

💡 TIP: Use /paper-pipeline status to see overall progress
```
