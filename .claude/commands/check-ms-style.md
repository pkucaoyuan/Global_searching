# Check MS Style - Management Science Journal Style Review

You are a Management Science style reviewer. Your task is to verify that a paper follows MS journal conventions, not ML/CS conference style.

## ⚠️ MANDATORY: RAG-Grounded Suggestions

**All style suggestions MUST be grounded in human-authored patterns from the writing reference library.**

### Step 0: Read Shared Config & MS-Relevant References

Before starting any review, execute these Read tool calls:

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/sentences/or_applications.md
Read .claude/writing_references/sentences/revenue_management.md
Read .claude/writing_references/sentences/dynamic_pricing.md
Read .claude/writing_references/sentences/choice_models.md
Read .claude/writing_references/paragraphs/main_results.md
Read .claude/writing_references/sentences/contribution.md
```

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before checking MS style compliance, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/framing.md        # Locked terminology & concepts
Read docs/paper_state/{resolved}/overview.md       # Paper current state
Read docs/paper_state/{resolved}/insights.md       # Key managerial insights
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- framing.md: [paper framing, e.g., "service system design" not "ML pipeline"]
- overview.md: [paper status, section structure]
- insights.md: [managerial insights for MS-style emphasis]
```

**If you skip this step, you may suggest style changes that contradict the paper's MS framing.**

These files contain patterns from Management Science, Operations Research, and related venues. Use them when suggesting rewrites.

### Step 2: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

This ensures the RAG library self-maintains by logging gaps and suggesting additions.

---

## Why This Matters

MS (Management Science) papers have fundamentally different style from ML papers:

| Dimension | ML Paper Style | MS Paper Style |
|-----------|----------------|----------------|
| Problem framing | Technical problem | Business/operational decision |
| "Arms" meaning | Algorithm variants | **Service design options** (broad) |
| Results | Algorithm performance | **Managerial insights** |
| Discussion | Limitations & future work | **Prescriptions for practitioners** |
| Experiments | Benchmark comparisons | **Operational scenarios** |
| Notation | Dense, compact | Clear, with interpretation |

## Arguments

- `$ARGUMENTS` - Optional: path to paper directory

## Workflow

### Phase 1: Read Key Sections

Read these sections to assess MS style compliance:
1. `introduction.tex` - Framing and arm definitions
2. `model.tex` or `setup.tex` - Problem formalization
3. `experiments.tex` - Case studies
4. `conclusion.tex` or `discussion.tex` - Insights

### Phase 2: Check Arm Definition Breadth

**Critical Check**: Are "arms" defined broadly as service design options?

**❌ Too Narrow (ML style):**
```
Arms are K different LLM models: GPT-4, Claude, Llama...
```

**✅ Broad (MS style):**
```
Arms represent deployable service configurations. Each arm encodes:
1. Operational design: workflow structure, routing rules, priority schemes
2. Technology choices: LLM model, reasoning depth, tool availability
3. Parameter settings: response length, confidence thresholds
```

**Checklist:**
- [ ] Arms defined as "service configurations" or "design options", not just "models"
- [ ] At least 2 of 3 layers mentioned: (1) operations, (2) technology, (3) parameters
- [ ] Concrete examples given that span multiple layers
- [ ] Introduction connects arms to operational decisions

**If Failed:** Suggest expanding arm definition with operational framing.

### Phase 3: Check Managerial Insights

**Critical Check**: Does the paper provide actionable insights for practitioners?

**❌ ML Style Discussion:**
```
Our algorithm achieves 90% accuracy with 5% audit rate.
The confidence sequences maintain 98% coverage.
Future work includes extending to contextual bandits.
```

**✅ MS Style Discussion:**
```
Our results yield concrete prescriptions for service managers:

1. When to trust the proxy: If judge reliability varies across segments,
   concentrate audits where the proxy is least reliable.

2. Audit budget allocation: The Neyman rule π* ∝ √g suggests auditing
   high-variance cases, not borderline cases.

3. Cost-quality trade-off: Managers can achieve 90% cost savings with
   minimal accuracy loss when quality gaps exceed Δ > 0.05.
```

**Checklist:**
- [ ] Discussion section addresses "managers" or "practitioners" explicitly
- [ ] At least 3 actionable prescriptions (not just performance numbers)
- [ ] Insights are phrased as decisions, not algorithm properties
- [ ] Trade-offs are quantified with practical thresholds

**If Failed:** Suggest rewriting discussion with managerial framing.

### Phase 4: Check Experiment Framing

**Critical Check**: Are experiments framed as operational scenarios?

**❌ ML Style:**
```
We evaluate on MT-Bench, a standard LLM benchmark.
Arms are 6 models: gpt-4, claude-v1, ...
```

**✅ MS Style:**
```
We validate our framework through two service system scenarios:

Scenario 1: Call Center Quality Monitoring
- Arms represent different chatbot configurations for customer support
- Each configuration varies in: response style, escalation rules, LLM backend
- Ground truth: supervisor quality scores

Scenario 2: Content Moderation Pipeline
- Arms represent moderation policy variants
- Configurations differ in: strictness thresholds, appeal routing, model choice
```

**Checklist:**
- [ ] Experiments introduced as "service scenarios" or "operational settings"
- [ ] Each experiment has a clear business context
- [ ] Arms are described with operational meaning, not just model names
- [ ] Results discussed in terms of operational impact

### Phase 5: Check Notation Interpretation

**Critical Check**: Is mathematical notation accompanied by operational interpretation?

**❌ Dense ML Style:**
```
Let π_t = P(A_t=1|F_t,X_t) with π_t ≥ π_min.
```

**✅ MS Style with Interpretation:**
```
Let π_t denote the audit probability—the likelihood that a supervisor
reviews a given interaction. The constraint π_t ≥ π_min ensures every
segment has positive audit coverage, preventing blind spots in quality
monitoring.
```

**Checklist:**
- [ ] Key symbols have plain-English interpretations
- [ ] Operational meaning explained for: π (audit), F (judge), Y (ground truth)
- [ ] Constraints interpreted in business terms

### Phase 6: Check Language Register (ML → MS Transformation)

**Critical Check**: Does the paper use MS-register language throughout, or does it still read like an ML/CS paper?

This phase scans all sections for ML-register phrases and flags them with MS-register alternatives. This is distinct from structural checks (Phases 1-5) — it operates at the **sentence level**.

**ML → MS Phrase Mapping (General Rules):**

| ML Register | MS Register | Context |
|-------------|-------------|---------|
| "We propose an algorithm" | "We develop a decision framework" | Contribution claims |
| "We propose" | "We develop" / "We design" / "We introduce" | General |
| "Our method achieves" | "Our approach yields" / "Our framework delivers" | Results |
| "state-of-the-art" | "best-in-class" / "among compared approaches" | Comparisons |
| "benchmark" | "case study" / "operational scenario" | Experiments |
| "training data" | "historical observations" / "operational data" | Data |
| "model performance" | "system effectiveness" / "operational outcomes" | Metrics |
| "hyperparameter" | "design parameter" / "operational parameter" | Settings |
| "ablation study" | "sensitivity analysis" / "component analysis" | Analysis |
| "baseline" | "benchmark policy" / "status quo" | Comparisons |
| "deploy the model" | "implement the framework" / "operationalize" | Practice |
| "training" / "fine-tuning" | "calibration" / "configuration" | Setup |
| "features" | "covariates" / "observable characteristics" | Variables |
| "ground truth labels" | "expert assessments" / "verified outcomes" | Evaluation |
| "batch size" | "sample allocation" | Design |
| "epochs" | "iterations" / "cycles" | Process |
| "loss function" | "objective function" / "cost criterion" | Optimization |
| "neural network" | "predictive model" (if not the focus) | Architecture |
| "We show that" | "We establish that" / "We demonstrate that" | Claims |
| "empirically" | "through operational case studies" | Evidence |

**Sentence-Level Patterns to Flag:**

```bash
# ML-register detection patterns
grep -n "we propose\|we develop.*algorithm\|state-of-the-art" sections/*.tex
grep -n "ablation\|hyperparameter\|training data\|ground truth label" sections/*.tex
grep -n "benchmark.*dataset\|deploy.*model\|fine-tun" sections/*.tex
grep -n "neural network\|batch size\|epoch\|loss function" sections/*.tex
```

**How to Check:**

1. Read each section and count ML-register phrases
2. For each occurrence, suggest MS-register alternative
3. Distinguish between:
   - **Must fix**: ML jargon that has no place in MS (e.g., "ablation study" → "sensitivity analysis")
   - **Context-dependent**: Terms that are OK in technical sections but not in intro/conclusion (e.g., "algorithm" is fine in §5, but intro should say "framework" or "approach")
   - **Keep as-is**: ML terms that are the actual subject of the paper (e.g., "LLM" is fine since LLMs are the topic)

**Checklist:**
- [ ] Introduction uses MS-register (service design, operational decisions) not ML-register?
- [ ] Conclusion/Discussion addresses practitioners, not ML researchers?
- [ ] Experiments described as scenarios/case studies, not benchmarks?
- [ ] Technical sections use operational interpretations alongside formal notation?
- [ ] No gratuitous ML jargon in non-technical sections?

**Output for this phase:**
```
## Language Register Analysis

| Section | ML-Register Count | Examples | Severity |
|---------|-------------------|----------|----------|
| Introduction | 2 | "we propose an algorithm" (L15) | ⚠️ Fix |
| Model | 0 | — | ✅ OK |
| Experiments | 5 | "benchmark" (L3), "ablation" (L45) | ⚠️ Fix |
| Conclusion | 3 | "state-of-the-art" (L8) | ⚠️ Fix |

Total ML-register phrases: 10
Recommended: Rewrite using MS-register alternatives from mapping table above.
```

---

### Phase 7: Check Problem-Setting Novelty

**Critical Check**: Does the paper frame the problem setting itself as a contribution, not just the solution?

**❌ Solution-Only Framing:**
```
We solve the multi-armed bandit problem in the context of LLM services.
```

**✅ Problem-Setting Novelty:**
```
We identify a new operational challenge — auditing LLM-generated outputs
under budget constraints — and formalize it as a structured bandit problem
with proxy feedback. This formalization itself reveals key insights:
the audit allocation depends on judge reliability, not case difficulty.
```

**Checklist:**
- [ ] Introduction frames the *problem setting* (not just the method) as novel
- [ ] Formalization yields insights beyond the technical solution
- [ ] Paper argues why existing formulations are insufficient
- [ ] At least one "modeling insight" distinct from "algorithmic insight"

**If Failed:** Suggest reframing introduction to emphasize problem-setting contribution.

### Phase 8: Check for Conference Extension Language

**Critical Check**: Does the paper read as standalone, not an extension?

**❌ Extension Language:**
```
This paper extends our conference version [cite] with...
The conference paper established X; here we add Y.
Building on our prior work...
```

**✅ Standalone Language:**
```
A preliminary version of this work appeared in [conference].
The present paper provides complete proofs, additional case studies,
and an extended theoretical analysis.
```

**Checklist:**
- [ ] No "extends", "builds on", "adds to" language
- [ ] Conference version mentioned once, briefly
- [ ] Paper reads as self-contained contribution

## Output Format

```markdown
# MS Style Review Report

**Paper**: [title]
**Target Journal**: Management Science
**Review Date**: [date]

---

## Overall Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| Arm definition breadth | ✅/⚠️/❌ | |
| Managerial insights | ✅/⚠️/❌ | |
| Experiment framing | ✅/⚠️/❌ | |
| Notation interpretation | ✅/⚠️/❌ | |
| Language register (ML→MS) | ✅/⚠️/❌ | |
| Problem-setting novelty | ✅/⚠️/❌ | |
| Standalone language | ✅/⚠️/❌ | |

**Overall**: Ready / Needs Revision / Major Revision Required

---

## Detailed Findings

### 1. Arm Definition
**Current**: [quote current definition]
**Issue**: [if any]
**Suggested revision**:
```
[concrete suggestion]
```

### 2. Managerial Insights
**Current insights count**: X
**Missing**: [what's needed]
**Suggested additions**:
- Insight 1: ...
- Insight 2: ...

### 3. Experiment Framing
[Similar structure]

### 4. Notation Interpretation
[Similar structure]

### 5. Language Register
**ML-register phrases found**: X
**Locations**: [list]
**Suggested MS-register rewrites**: [table]

### 6. Extension Language
[Similar structure]

---

## Priority Actions

1. [Highest priority fix with specific location]
2. [Second priority]
3. [Third priority]

---

## Suggested Text Revisions

**⚠️ Every suggestion MUST cite a RAG pattern from the references you read.**

### Arm Definition (Introduction)
**Location**: Section 1, paragraph X
**Current**:
> [exact quote]

**RAG Pattern** (from `sentences/or_applications.md`):
> [cite the reference pattern]

**Suggested** (adapted from pattern):
> [revised text grounded in the pattern]

[Continue for each major revision needed]
```

## Self-Dispatch Phases

**This skill has 8 independent check phases. Follow `.claude/commands/_shared/self_dispatch_protocol.md`.**

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 1 | Arm definition breadth | Yes | `introduction.tex`, `model.tex` | Arms defined as service configurations, not just "models"; multi-layer coverage |
| 2 | Managerial insights | Yes | `conclusion.tex`, `experiments.tex` | Actionable prescriptions for practitioners; decisions, not algorithm properties |
| 3 | Experiment framing | Yes | `experiments.tex`, `introduction.tex` | Experiments framed as operational scenarios with business context |
| 4 | Notation interpretation | Yes | `model.tex`, `method.tex`, `algorithm.tex` | Key symbols have plain-English operational interpretations |
| 5 | Language register | Yes | All `sections/*.tex` | ML-register phrases replaced with MS-register; sentence-level venue transformation |
| 6 | Problem-setting novelty | Yes | `introduction.tex`, `model.tex` | Problem formalization as contribution; modeling insight distinct from algorithmic |
| 7 | Conference extension language | Yes | All `sections/*.tex` | Paper reads as standalone; no "extends", "builds on" language |

**Parallel group**: All 7 phases can run in parallel (each checks different aspect).
**Aggregation**: Merge 7 sub-reports into single MS style report with overall assessment table.

---

## Begin

**Dispatch**: All phases parallel — **Template A** from `self_dispatch_protocol.md`.

1. Follow unified protocol Steps 0A–2.5 (include RAG references)
2. Recursion guard → if subagent, execute inline
3. Dispatch 7 parallel Task subagents (Phases 1-7)
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

📊 This Check: L3 Style (Management Science)
   Issues Found: {N}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   1. Add managerial insights to: [sections]
   2. Strengthen prescriptions in: [sections]
   3. Broaden arm definition in: [locations]
   4. Re-run this check to verify

   {If no issues:}
   ✅ MS style requirements met. Ready for language polish.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found - fix first, then:]
   /check-ms-style             → Re-verify after fixes

   [When L3 passes:]
   /polish-paper               → Move to L4 (language polish)

   [After L4:]
   /paper-pipeline pre-submit MS → Final submission checklist

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy     ✅
   L1 Structure   ─────────── /check-content-placement      ✅
   L2 Consistency ─────────── /check-paper-consistency      ✅
 → L3 Style       ─────────── YOU ARE HERE
   L4 Language    ─────────── /polish-paper (NEXT)

💡 TIP: Use /paper-pipeline status to see overall progress
```
