# MS Style Reference — Phrase Mappings, Examples & Output Templates

## ML → MS Language Register Mapping

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

### Detection Patterns

```bash
grep -n "we propose\|we develop.*algorithm\|state-of-the-art" sections/*.tex
grep -n "ablation\|hyperparameter\|training data\|ground truth label" sections/*.tex
grep -n "benchmark.*dataset\|deploy.*model\|fine-tun" sections/*.tex
grep -n "neural network\|batch size\|epoch\|loss function" sections/*.tex
```

### Classification

- **Must fix**: ML jargon with no place in MS (e.g., "ablation study" → "sensitivity analysis")
- **Context-dependent**: OK in technical sections but not in intro/conclusion (e.g., "algorithm" fine in §5, but intro should say "framework")
- **Keep as-is**: ML terms that are the actual subject (e.g., "LLM" is fine since LLMs are the topic)

---

## Problem-Setting Novelty Examples

### Weak Problem-Setting Positioning

```
We study best-arm identification with biased LLM judges.
```
(Reader thinks: "So it's BAI with a twist. Incremental.")

### Strong Problem-Setting Positioning

```
Classical service system optimization assumes that performance samples
are scalar observations from simulations or operational data. In many
modern service systems, however, the primary evidence of system
performance takes the form of unstructured text—customer service
transcripts, compliance review reports, medical encounter notes. These
text records cannot be directly compared or aggregated using standard
statistical methods. We study how to design service systems when
performance evidence is textual, using LLM judges to convert text into
proxy scores that enable optimization.
```
(Reader thinks: "This is a genuinely new problem setting with broad applicability.")

### Common Failures in MS Submissions

- Problem novelty stated only in introduction, never echoed in model/experiments/conclusion
- Problem novelty conflated with method novelty ("our contribution is the algorithm" — but what makes the PROBLEM hard?)
- Experiments use standard benchmarks that don't showcase the new problem setting
- Related work positions against methodological competitors but not against alternative problem formulations

---

## Output Format Template

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
| **Problem-setting novelty** | ✅/⚠️/❌ | |
| Standalone language | ✅/⚠️/❌ | |

**Overall**: Ready / Needs Revision / Major Revision Required

---

## Detailed Findings

### 1. Arm Definition
**Current**: [quote current definition]
**Issue**: [if any]
**Suggested revision**:
> [concrete suggestion]

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

### 6. Problem-Setting Novelty
[Score table from Phase 7]

### 7. Extension Language
[Similar structure]

---

## Priority Actions

1. [Highest priority fix with specific location]
2. [Second priority]
3. [Third priority]

---

## Suggested Text Revisions

**⚠️ Every suggestion MUST cite a RAG pattern from the references you read.**

### [Section Name]
**Location**: Section X, paragraph Y
**Current**:
> [exact quote]

**RAG Pattern** (from `sentences/[file].md`):
> [cite the reference pattern]

**Suggested** (adapted from pattern):
> [revised text grounded in the pattern]
```

---

## Language Register Analysis Output

```
## Language Register Analysis

| Section | ML-Register Count | Examples | Severity |
|---------|-------------------|----------|----------|
| Introduction | 2 | "we propose an algorithm" (L15) | ⚠️ Fix |
| Model | 0 | — | ✅ OK |
| Experiments | 5 | "benchmark" (L3), "ablation" (L45) | ⚠️ Fix |
| Conclusion | 3 | "state-of-the-art" (L8) | ⚠️ Fix |

Total ML-register phrases: 10
Recommended: Rewrite using MS-register alternatives from mapping table.
```

---

## Next Steps Footer

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
