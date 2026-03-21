# Check MSOM Style - M&SOM Journal Style Review

You are an M&SOM (Manufacturing & Service Operations Management) style reviewer. Your task is to verify that a paper follows MSOM journal conventions, which differ from both MS (Management Science) and ML conference styles.

## ⚠️ MANDATORY: RAG-Grounded Suggestions

**All style suggestions MUST be grounded in human-authored patterns from the writing reference library.**

### Step 0: Read Shared Config & MSOM-Relevant References

Before starting any review, execute these Read tool calls:

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/sentences/or_applications.md
Read .claude/writing_references/sentences/inventory_management.md
Read .claude/writing_references/sentences/contribution.md
Read .claude/writing_references/paragraphs/main_results.md
```

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before checking MSOM style compliance, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/framing.md        # Locked terminology & concepts
Read docs/paper_state/{resolved}/overview.md       # Paper current state
Read docs/paper_state/{resolved}/insights.md       # Key operational insights
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- framing.md: [paper framing, e.g., "OR model debugging workflow" not "ML pipeline"]
- overview.md: [paper status, section structure]
- insights.md: [operational insights for MSOM-style emphasis]
```

**If you skip this step, you may suggest style changes that contradict the paper's MSOM framing.**

### Step 2: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

---

## Why MSOM Differs from MS and ML

MSOM occupies a distinct position: more applied than OR, more operations-focused than MS, and more practice-oriented than ML.

| Dimension | MSOM Style | MS Style | ML Style | OR Style |
|-----------|-----------|----------|----------|----------|
| Core focus | **Operational practice** | Managerial insights | Empirical performance | Algorithmic/methodological |
| Paper goal | Improve how operations work | Inform decisions | Advance methods | Prove optimality |
| Application depth | **Deep, realistic** | Moderate | Benchmark-oriented | Canonical problems |
| Industry relevance | **Explicit and central** | Present but secondary | Optional | Optional |
| Data/experiments | Real or realistic operational data | Case studies | Benchmarks | Structured instances |
| Managerial insights | Operational prescriptions | Strategic prescriptions | Light discussion | Complexity results |
| Page limit | **32 pages** (all-inclusive) | Flexible | 8-10 + appendix | Flexible |
| Footnotes | **Prohibited** | Allowed | Allowed | Allowed |
| Online supplement | **Max 16 pages** | Unlimited e-companion | Unlimited appendix | Unlimited appendix |
| Review process | Double-anonymous | Double-anonymous | Double-anonymous | Double-anonymous |

## MSOM Journal Requirements (Hard Constraints)

| Requirement | Specification |
|-------------|--------------|
| Page limit | **32 pages** including references, tables, figures, appendices |
| Online supplement | Max **16 pages** (separate file) |
| Spacing | Double-spaced, max 33 lines per page |
| Footnotes | **Not allowed** — incorporate into main text in parentheses |
| Citations | Author-year style: (Norman 1977) or Norman (1977) |
| Review type | Double-anonymous |
| Template | MSOM LaTeX/Word template from INFORMS |

## Arguments

- `$ARGUMENTS` - Optional: path to paper directory

## Workflow

### Phase 1: Check Hard Format Constraints

**Critical Check**: Does the paper satisfy MSOM's strict formatting rules?

**Checklist:**
- [ ] Total pages ≤ 32 (including everything)?
- [ ] No footnotes anywhere in the manuscript? (grep for `\footnote`)
- [ ] Double-spaced throughout (including abstract and references)?
- [ ] Citations use author-year format (not numbered)?
- [ ] Online supplement (if any) ≤ 16 pages?
- [ ] Manuscript is anonymous (no author names, affiliations, acknowledgments)?

**If Failed:** These are non-negotiable — flag as 🔴 BLOCKING.

### Phase 2: Check Operational Relevance

**Critical Check**: Is the paper framed around improving operational practice?

MSOM papers must demonstrate clear relevance to how operations are managed in practice. The central question should be: "How does this help someone running operations?"

**❌ Too Academic/Technical:**
```
We develop a benchmark for evaluating LLM debugging capabilities
on optimization models, achieving 95.3% recovery rate.
```

**✅ Operations-Focused:**
```
Operations teams routinely build and maintain optimization models for
planning, scheduling, and resource allocation. When these models
contain errors—yielding infeasible or suboptimal solutions—the
debugging process is manual, time-consuming, and requires specialized
expertise. We develop a systematic framework for automating this
debugging workflow, reducing resolution time from hours to minutes
while maintaining solution quality.
```

**Checklist:**
- [ ] Introduction identifies a specific operational pain point?
- [ ] Problem is motivated by how operations teams actually work?
- [ ] The paper explains who benefits (operations analysts, planners, managers)?
- [ ] Application context appears early (not buried in experiments)?
- [ ] Experiments use realistic operational settings, not just synthetic benchmarks?

**If Failed:** Suggest strengthening the operational motivation in the introduction.

### Phase 3: Check Practical Impact Articulation

**Critical Check**: Does the paper clearly articulate practical impact beyond methodology?

MSOM reviewers ask: "So what? How would a practitioner use this?"

**❌ Method-Focused Impact:**
```
Our trained model achieves 97.3% recovery rate, surpassing
all 22 API baselines by 11 percentage points.
```

**✅ Practice-Focused Impact:**
```
Our framework enables operations teams to resolve optimization model
errors in an average of 1.82 diagnostic steps—compared to the 3-5 steps
required by general-purpose AI assistants. For a planning team maintaining
dozens of optimization models, this translates to substantial reductions
in debugging time and analyst workload. The framework requires no
proprietary API access: a single 8-billion-parameter model, trainable
on standard hardware, matches or exceeds commercial alternatives.
```

**Checklist:**
- [ ] Results are interpreted in operational terms (time saved, cost reduced, quality improved)?
- [ ] At least 3 practical implications are stated explicitly?
- [ ] Implications address operational decisions, not just model performance?
- [ ] Resource requirements are discussed (hardware, data, expertise needed)?
- [ ] Comparison to current practice (status quo) is provided?

### Phase 4: Check MSOM Experiment Framing

**Critical Check**: Are experiments framed as operational evaluations, not just benchmarks?

**❌ ML Benchmark Style:**
```
We evaluate on OR-Debug-Bench, a dataset of 450 optimization problems
with 9 error types. Table 1 shows recovery rates across 25 models.
```

**✅ MSOM Operational Evaluation Style:**
```
We evaluate our framework across a representative set of operational
optimization scenarios spanning production planning, resource allocation,
network design, and scheduling. The test suite comprises 450 problems
derived from real-world model structures, categorized by the type of
modeling error that operations teams commonly encounter (constraint
specification errors, bound misconfigurations, objective function
mistakes, etc.).
```

**Checklist:**
- [ ] Experiments introduced with operational context (what domain, what decisions)?
- [ ] Problem instances described by operational relevance, not just technical properties?
- [ ] Error types connected to common practitioner mistakes?
- [ ] Results discuss operational efficiency (steps, time, resources), not just accuracy?
- [ ] At least one experiment or analysis addresses deployment considerations?

### Phase 5: Check Language Register (ML/CS → MSOM Transformation)

**Critical Check**: Does the paper use MSOM-register language, not ML/CS jargon?

**ML/CS → MSOM Phrase Mapping:**

| ML/CS Register | MSOM Register | Context |
|----------------|---------------|---------|
| "We propose a model" | "We develop a framework" / "We design a system" | Contribution |
| "benchmark" | "test suite" / "evaluation scenarios" | Experiments |
| "state-of-the-art" | "best available" / "among current approaches" | Comparisons |
| "training data" | "expert demonstration data" / "historical debugging traces" | Data |
| "fine-tuning" | "domain adaptation" / "specialization" | Training |
| "ablation study" | "component analysis" / "sensitivity analysis" | Analysis |
| "baseline models" | "benchmark approaches" / "alternative methods" | Comparisons |
| "inference" | "model deployment" / "real-time operation" | Usage |
| "ground truth" | "verified solutions" / "expert assessments" | Validation |
| "hyperparameter" | "configuration parameter" / "design choice" | Settings |
| "epoch" | "training iteration" / "learning cycle" | Training |
| "loss function" | "objective criterion" / "optimization metric" | Training |
| "SOTA" | Spell out or avoid | Anywhere |
| "LLM agent" | "AI-assisted debugging system" / "automated diagnostic tool" | Framing |
| "reward function" | "performance criterion" / "quality measure" | RL context |
| "deployment" | "implementation" / "operationalization" | Practice |
| "pipeline" | "workflow" / "process" / "methodology" | Architecture |

**Sentence-Level Patterns to Flag:**

```bash
# ML-register detection patterns
grep -n "state-of-the-art\|SOTA\|ablation" sections/*.tex
grep -n "ground truth\|hyperparameter\|epoch" sections/*.tex
grep -n "fine-tun\|inference\|loss function" sections/*.tex
grep -n "\\\\footnote{" sections/*.tex  # Footnotes are prohibited
```

**Checklist:**
- [ ] Introduction uses operational language (workflows, processes, teams)?
- [ ] Conclusion addresses operations practitioners, not ML researchers?
- [ ] Technical terms are explained in operational context?
- [ ] No footnotes in the manuscript (MSOM prohibition)?
- [ ] No gratuitous ML jargon in non-technical sections?

### Phase 6: Check Contribution Positioning for MSOM

**Critical Check**: Are contributions framed as operational advances, not just technical ones?

MSOM values three types of contributions:
1. **Operational insight**: New understanding of how operations work or should work
2. **Practical methodology**: Tools/frameworks that practitioners can adopt
3. **Empirical evidence**: Data-driven findings about operational phenomena

**❌ Technical Contribution (ML-style):**
```
Our contributions are: (1) a new benchmark dataset, (2) a training
pipeline using SFT and GRPO, (3) showing small models outperform
large ones.
```

**✅ Operational Contribution (MSOM-style):**
```
Our contributions are threefold. First, we formalize the optimization
model debugging process as a sequential decision problem, providing a
structured framework that captures how operations analysts diagnose and
repair modeling errors. Second, we develop a domain-specialized AI
assistant that automates this process, demonstrating that targeted
training on expert debugging traces yields tools substantially more
effective than general-purpose alternatives. Third, we provide empirical
evidence that small, locally deployable models can match or exceed
commercial AI services for specialized operational tasks—an important
finding for organizations with data privacy constraints or limited
API budgets.
```

**Checklist:**
- [ ] Each contribution addresses an operational need, not just a technical gap?
- [ ] Contributions explain the "so what" for practitioners?
- [ ] At least one contribution has direct practical applicability?
- [ ] Contributions avoid pure ML jargon (no "we train", "we benchmark")?

### Phase 7: Check Related Work Coverage

**Critical Check**: Does the related work cover the operations management literature, not just ML/AI?

MSOM reviewers expect engagement with the OM literature relevant to the problem domain, even for AI-focused papers.

**Required literature coverage for an AI/ML paper in MSOM:**
1. **Operations management literature**: The application domain (e.g., model building, decision support, optimization in practice)
2. **AI/ML methodology literature**: The technical approach (acceptable to be more concise)
3. **AI in operations literature**: Prior work on AI/ML applied to operations

**Checklist:**
- [ ] Related work cites OM/OR journals (MSOM, MS, OR, POM), not just ML venues?
- [ ] Application domain literature is covered (optimization modeling practice, decision support)?
- [ ] AI-in-operations literature is cited (prior MSOM/MS papers on AI/ML in operations)?
- [ ] Clear positioning: what does this paper do that prior OM work does not?

### Phase 8: Check Page Budget Allocation

**Critical Check**: Is the 32-page budget well-allocated for MSOM readership?

MSOM's hard page limit requires careful allocation. Reviewers notice when the balance is wrong.

**Recommended allocation for an AI-in-operations paper:**

| Section | Pages | Guidance |
|---------|-------|----------|
| Introduction | 3-4 | Operational motivation, contributions, paper outline |
| Related Work | 2-3 | OM + AI/ML + AI-in-operations |
| Problem Formulation | 3-4 | Formal setup with operational interpretation |
| Methodology | 5-7 | Core technical content |
| Experiments | 6-8 | Operational evaluation, comparisons, analysis |
| Discussion/Conclusion | 2-3 | Practical implications, limitations, future work |
| References | 2-3 | — |
| Tables & Figures | 3-5 | — |

**Checklist:**
- [ ] Introduction is ≤ 4 pages (not too long for MSOM)?
- [ ] Experiments get adequate space (≥ 6 pages for empirical MSOM papers)?
- [ ] Discussion explicitly addresses practical implications?
- [ ] Total ≤ 32 pages?
- [ ] Dense technical material (proofs, derivations) moved to online supplement?
- [ ] Online supplement ≤ 16 pages?

---

## Output Format

```markdown
# MSOM Style Review Report

**Paper**: [title]
**Target Journal**: Manufacturing & Service Operations Management
**Track**: AI in Operations (if applicable)
**Review Date**: [date]

---

## Hard Constraint Check

| Constraint | Status | Notes |
|------------|--------|-------|
| Page limit (≤32) | ✅/❌ | Current: X pages |
| No footnotes | ✅/❌ | Found: X footnotes |
| Double spacing | ✅/❌ | |
| Author-year citations | ✅/❌ | |
| Online supplement (≤16) | ✅/❌/N/A | |
| Anonymous | ✅/❌ | |

## Style Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| Operational relevance | ✅/⚠️/❌ | |
| Practical impact | ✅/⚠️/❌ | |
| Experiment framing | ✅/⚠️/❌ | |
| Language register (ML→MSOM) | ✅/⚠️/❌ | |
| Contribution positioning | ✅/⚠️/❌ | |
| Related work coverage | ✅/⚠️/❌ | |
| Page budget allocation | ✅/⚠️/❌ | |

**Overall**: Ready / Needs Revision / Major Revision Required

---

## Detailed Findings

### 1. Hard Constraints
[Any blocking issues]

### 2. Operational Relevance
**Current**: [quote current framing]
**Issue**: [if any]
**Suggested revision**:
```
[concrete suggestion]
```

### 3. Practical Impact
**Current impact statements**: X
**Missing**: [what's needed]
**Suggested additions**:
- Implication 1: ...
- Implication 2: ...

### 4. Experiment Framing
[Assessment]

### 5. Language Register
**ML-register phrases found**: X
**Locations**: [list]
**Suggested MSOM-register rewrites**: [table]

### 6. Contribution Positioning
[Assessment]

### 7. Related Work
[Assessment]

### 8. Page Budget
**Current allocation**: [breakdown]
**Issues**: [over/under-allocated sections]

---

## Priority Actions

1. [Highest priority fix with specific location]
2. [Second priority]
3. [Third priority]

---

## Suggested Text Revisions

**⚠️ Every suggestion MUST cite a RAG pattern from the references you read.**

[Continue for each major revision needed]
```

## Self-Dispatch Phases

**This skill has 8 independent check phases. Follow `.claude/commands/_shared/self_dispatch_protocol.md`.**

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 1 | Hard format constraints | Yes | All `sections/*.tex` | Page count, footnotes, spacing, citations, anonymity |
| 2 | Operational relevance | Yes | `introduction.tex`, `setup.tex` | Pain point identification, operational motivation, who benefits |
| 3 | Practical impact | Yes | `experiments.tex`, `conclusion.tex` | Operational implications, resource discussion, status quo comparison |
| 4 | Experiment framing | Yes | `experiments.tex`, `introduction.tex` | Operational context, domain relevance, deployment considerations |
| 5 | Language register | Yes | All `sections/*.tex` | ML-register phrases, footnote prohibition, MSOM terminology |
| 6 | Contribution positioning | Yes | `introduction.tex` | Operational need, "so what" for practitioners, practical applicability |
| 7 | Related work coverage | Yes | `related_work.tex` | OM literature, AI-in-operations, positioning against OM work |
| 8 | Page budget allocation | Yes | All `sections/*.tex`, appendix | Page counts per section, balance assessment, supplement size |

**Parallel group**: All 8 phases can run in parallel (each checks different aspect).
**Aggregation**: Merge 8 sub-reports into single MSOM style report with overall assessment table.

---

## Begin

**Dispatch**: All phases parallel — **Template A** from `self_dispatch_protocol.md`.

1. Follow unified protocol Steps 0A–2.5 (include RAG references)
2. Recursion guard → if subagent, execute inline
3. Dispatch 8 parallel Task subagents (Phases 1-8)
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

📊 This Check: L3 Style (MSOM - Manufacturing & Service Operations Management)
   Issues Found: {N}
   Hard Constraint Violations: {M}

🔴 IMMEDIATE ACTIONS:
   {If hard constraint violations:}
   1. ❌ BLOCKING: Remove all footnotes (MSOM prohibition)
   2. ❌ BLOCKING: Reduce page count to ≤32
   [etc.]

   {If style issues found:}
   1. Strengthen operational framing in: [sections]
   2. Add practical implications to: [sections]
   3. Replace ML jargon in: [locations]
   4. Expand OM literature coverage in related work

   {If no issues:}
   ✅ MSOM style requirements met. Ready for language polish.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found - fix first, then:]
   /check-msom-style             → Re-verify after fixes

   [When L3 passes:]
   /polish-paper               → Move to L4 (language polish)

   [After L4:]
   /paper-pipeline pre-submit MSOM → Final submission checklist

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy     ✅
   L1 Structure   ─────────── /check-content-placement      ✅
   L2 Consistency ─────────── /check-paper-consistency      ✅
 → L3 Style       ─────────── YOU ARE HERE
   L4 Language    ─────────── /polish-paper (NEXT)

💡 TIP: Use /paper-pipeline status to see overall progress
```
