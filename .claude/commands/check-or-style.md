# Check OR Style - Operations Research Journal Style Review

You are an Operations Research journal style reviewer. Your task is to verify that a paper follows OR journal conventions.

## ⚠️ MANDATORY: RAG-Grounded Suggestions

**All style suggestions MUST be grounded in human-authored patterns.**

### Step 0: Read Shared Config & OR-Relevant References

Before starting any review, execute these Read tool calls:

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/sentences/or_applications.md
Read .claude/writing_references/sentences/queueing_theory.md
Read .claude/writing_references/sentences/inventory_management.md
Read .claude/writing_references/sentences/stochastic_programming.md
Read .claude/writing_references/sentences/robust_optimization.md
Read .claude/writing_references/sentences/network_optimization.md
Read .claude/writing_references/paragraphs/main_results.md
```

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before checking OR style compliance, you MUST:
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

**If you skip this step, you may suggest style changes that contradict the paper's OR framing.**

### Step 2: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

This ensures the RAG library self-maintains by logging gaps and suggesting additions.

---

## OR vs MS vs ML Style Differences

| Dimension | OR Style | MS Style | ML Style |
|-----------|----------|----------|----------|
| Focus | Algorithmic/Methodological | Managerial insights | Empirical performance |
| Theory depth | Deep, complete proofs | Moderate, some in appendix | Light, focus on experiments |
| Applications | Canonical OR problems | Business operations | Benchmarks |
| Notation | Classical OR (e.g., $x_{ij}$) | Interpretable | Dense, compact |
| Results | Optimality, complexity | Prescriptions | Accuracy numbers |

## Arguments

- `$ARGUMENTS` - Optional: path to paper directory

## Workflow

### Phase 1: Check Methodological Contribution

**Critical Check**: Does the paper make a clear methodological contribution?

**❌ Weak OR Contribution:**
```
We apply standard techniques to a new problem domain.
Our experiments show good performance on benchmark datasets.
```

**✅ Strong OR Contribution:**
```
We establish the computational complexity of the joint sampling-auditing
problem (Theorem 3.1). We develop a polynomial-time approximation algorithm
achieving a (1-1/e) guarantee (Theorem 4.2). Our analysis reveals a novel
decomposition structure that separates the inner auditing problem from
the outer sampling allocation (Proposition 4.1).
```

**Checklist:**
- [ ] Clear complexity/hardness results?
- [ ] Novel algorithmic contribution (not just application)?
- [ ] Theoretical guarantees (approximation ratio, regret bounds, sample complexity)?
- [ ] Structural insights about the problem?

### Phase 2: Check Problem Formulation Rigor

**Critical Check**: Is the problem formulation mathematically rigorous?

**OR papers require:**
1. Formal problem definition (optimization or decision problem)
2. Complete specification of constraints
3. Clear objective function
4. Discussion of problem structure (convexity, submodularity, etc.)

**Checklist:**
- [ ] Problem stated as formal optimization/decision problem?
- [ ] All constraints explicitly listed?
- [ ] Objective function clearly defined?
- [ ] Problem structure analyzed (NP-hardness, special structure)?

### Phase 3: Check Proof Completeness

**Critical Check**: Are proofs complete and rigorous?

**OR Style Requirements:**
- Main results should have complete proofs (can be in appendix)
- Proof sketches in main body should be substantive
- All lemmas and intermediate results should be proven

**❌ ML-Style Proof Sketch:**
```
Proof Sketch: The result follows from standard concentration arguments.
See appendix for details.
```

**✅ OR-Style Proof Sketch:**
```
Proof Sketch: We proceed in three steps. First, we establish a martingale
structure for the IPW residuals (Lemma B.1). Second, we apply a time-uniform
concentration bound (Proposition B.2) to obtain confidence sequences.
Third, we use a change-of-measure argument to lower bound the KL divergence
(Lemma B.3). The full proof appears in Appendix B.
```

**Checklist:**
- [ ] All theorems have proofs (main text or appendix)?
- [ ] Proof sketches outline the key steps?
- [ ] Key lemmas are stated and proven?
- [ ] Proof techniques clearly identified?

### Phase 4: Check OR Terminology

**Use standard OR terminology:**

| Avoid | Use Instead |
|-------|-------------|
| "model" (ML sense) | "algorithm", "policy", "procedure" |
| "training" | "optimization", "learning" |
| "features" | "covariates", "attributes", "parameters" |
| "loss" | "cost", "objective", "regret" |
| "accuracy" | "optimality gap", "approximation ratio" |

**Checklist:**
- [ ] Uses "algorithm" not "model" for procedures?
- [ ] Uses "cost" or "regret" not "loss"?
- [ ] Uses "optimality" language for performance?

### Phase 5: Check Literature Positioning

**OR papers should cite:**
1. Classical OR foundations (if applicable)
2. Related optimization/algorithm literature
3. Application domain literature

**Checklist:**
- [ ] Cites foundational OR work (when relevant)?
- [ ] Positions contribution within OR literature?
- [ ] Discusses relationship to operations/optimization community?

### Phase 6: Check Computational Experiments

**OR Experiment Style:**
- Compare against optimal solutions (when tractable)
- Report optimality gaps, not just raw metrics
- Test on structured instances (not just random)
- Include sensitivity analysis

**❌ ML-Style Experiments:**
```
Table 1: Accuracy on benchmark datasets
| Model | Dataset A | Dataset B |
| Ours  | 94.2%     | 91.3%     |
| Baseline | 89.1%  | 87.2%     |
```

**✅ OR-Style Experiments:**
```
Table 1: Performance relative to optimal solution
| Algorithm | Opt Gap (%) | Runtime (s) | #Iterations |
| PP-LUCB   | 2.3 ± 0.4   | 12.5        | 847         |
| Track-and-Stop | 3.1 ± 0.6 | 8.2       | 623         |
| Uniform   | 8.7 ± 1.2   | 15.3        | 1204        |
| Lower Bound | 0 (oracle) | —          | —           |
```

**Checklist:**
- [ ] Reports optimality gaps or approximation ratios?
- [ ] Compares against theoretical bounds?
- [ ] Includes runtime/complexity analysis?
- [ ] Tests on structured problem instances?

## Output Format

```markdown
# OR Style Review Report

**Paper**: [title]
**Target Journal**: Operations Research
**Review Date**: [date]

---

## Overall Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| Methodological contribution | ✅/⚠️/❌ | |
| Problem formulation rigor | ✅/⚠️/❌ | |
| Proof completeness | ✅/⚠️/❌ | |
| OR terminology | ✅/⚠️/❌ | |
| Literature positioning | ✅/⚠️/❌ | |
| Computational experiments | ✅/⚠️/❌ | |

**Overall**: Ready / Needs Revision / Major Revision Required

---

## Detailed Findings

### 1. Methodological Contribution
**Current**: [assessment]
**Issue**: [if any]
**RAG Pattern** (from `sentences/algorithm_optimality.md`):
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

## Self-Dispatch Phases

**This skill has 6 independent check phases. Follow `.claude/commands/_shared/self_dispatch_protocol.md`.**

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 1 | Methodological contribution | Yes | `introduction.tex`, `algorithm.tex` | Clear complexity/hardness results; novel algorithmic contribution; theoretical guarantees |
| 2 | Problem formulation rigor | Yes | `model.tex`, `method.tex` | Formal problem definition; complete constraints; clear objective; structure analysis |
| 3 | Proof completeness | Yes | All `sections/*.tex`, `appendix/*.tex` | All theorems have proofs; proof sketches substantive; key lemmas stated |
| 4 | OR terminology | Yes | All `sections/*.tex` | Uses "algorithm" not "model"; "cost"/"regret" not "loss"; "optimality" language |
| 5 | Literature positioning | Yes | `related_work.tex`, `introduction.tex` | Cites foundational OR work; positions within OR community |
| 6 | Computational experiments | Yes | `experiments.tex` | Compares against optimal; reports optimality gaps; includes sensitivity analysis |

**Parallel group**: All 6 phases can run in parallel (each checks different aspect).
**Aggregation**: Merge 6 sub-reports into single OR style report with overall assessment table.

---

## Begin

**Dispatch**: All phases parallel — **Template A** from `self_dispatch_protocol.md`.

1. Follow unified protocol Steps 0A–2.5 (include RAG references)
2. Recursion guard → if subagent, execute inline
3. Dispatch 6 parallel Task subagents (Phases 1-6)
4. Aggregate → deduplicate → sort by severity
5. Every suggestion must cite its source RAG pattern
6. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: L3 Style (Operations Research Journal)
   Issues Found: {N}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   1. Strengthen optimality claims in: [sections]
   2. Add computational complexity analysis
   3. Improve proof rigor in: [proofs]

   {If no issues:}
   ✅ OR style requirements met. Ready for language polish.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found - no auto-fix for style, manual edits needed:]
   [Make edits based on suggestions above]
   /check-or-style             → Re-verify after fixes

   [When L3 passes:]
   /polish-paper               → Move to L4 (language polish)

   [After L4:]
   /paper-pipeline pre-submit OR → Final submission checklist

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy     ✅
   L1 Structure   ─────────── /check-content-placement      ✅
   L2 Consistency ─────────── /check-paper-consistency      ✅
 → L3 Style       ─────────── YOU ARE HERE
   L4 Language    ─────────── /polish-paper (NEXT)

💡 TIP: Use /paper-pipeline status to see overall progress
```
