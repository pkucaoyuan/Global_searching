# Init Paper State - Comprehensive Documentation Ecosystem

You are a paper state initialization agent. Your task is to create a **structured documentation ecosystem** for managing all aspects of an academic paper.

## Why This Matters

Without structured documentation:
```
Long context → Information lost → Inconsistent changes
Polish round 3 → Forgot what was decided in round 1
New session → Must re-read entire paper → Miss subtle issues
14 reviewer comments → Same issues keep recurring
```

With structured documentation:
```
docs/paper_state/[paper]/
├── overview.md           → Current state at a glance
├── symbols.md            → All notation in one place
├── results.md            → All theorems/lemmas with status
├── framing.md            → Locked concepts and terminology
├── changelog.md          → Track ALL modifications
├── cross_references.md   → Reference tracking
├── dependencies.md       → Assumption→Theorem chains
├── abbreviations.md      → Acronym registry
├── figures_tables.md     → All figures and tables
├── insights.md           → Key takeaways (for MS: managerial)
├── review_responses.md   → Reviewer comments and responses
└── consistency_log.md    → Check command history

Any change → Update relevant doc → Nothing gets lost
New session → Read state docs → Full context in minutes
```

## ⚠️ Protocol Reference

This command creates the state doc ecosystem defined in `unified_protocol.md`. After initialization, update `changelog.md` with a dated entry. See `.claude/commands/_shared/unified_protocol.md` for the canonical 12-file list and post-edit rules.

## Arguments

- `$ARGUMENTS` - Paper name or path (e.g., `ms_judge_paper`, `paper/journal/`)

## Directory Structure Created

```
docs/paper_state/[paper_name]/
├── overview.md           # Paper summary and current status
├── symbols.md            # Symbol/notation registry
├── results.md            # Theorems, lemmas, propositions registry
├── framing.md            # Locked concepts (from /define-paper-framing)
├── changelog.md          # Modification history
├── cross_references.md   # Track all refs to results/numbers
├── dependencies.md       # Assumption→Result→Result chains
├── abbreviations.md      # Acronym registry
├── figures_tables.md     # Combined figure & table registry
├── insights.md           # Key insights and takeaways
├── review_responses.md   # Reviewer comments tracking
└── consistency_log.md    # Consistency check history
```

**Templates**: All 12 templates available in `.claude/commands/templates/paper_state/`

### Why These Extra Files Matter

| File | Prevents | Example Issue |
|------|----------|---------------|
| `cross_references.md` | Changing theorem without updating references | "By Thm 5.2" but statement changed |
| `dependencies.md` | Removing assumption that breaks theorems | Delete A.2 → Thm 7.1 invalid |
| `abbreviations.md` | Using acronym before defining it | "LUCB algorithm" (undefined) |

## Document Templates

### 1. overview.md

```markdown
# Paper Overview: [Title]

**Created**: [date]
**Last Updated**: [date]
**Target Venue**: [MS/OR/ML]
**Status**: Draft / Polishing / Submitted / Revision

---

## One-Sentence Summary

[What is this paper about?]

---

## Current State

| Aspect | Status | Notes |
|--------|--------|-------|
| Content (L0) | ✅/⚠️/❌ | [notes] |
| Structure (L1) | ✅/⚠️/❌ | [notes] |
| Consistency (L2) | ✅/⚠️/❌ | [notes] |
| Style (L3) | ✅/⚠️/❌ | [notes] |
| Language (L4) | ✅/⚠️/❌ | [notes] |

---

## Section Map

| Section | File | Purpose | Key Results |
|---------|------|---------|-------------|
| 1. Introduction | introduction.tex | Motivation, contributions | - |
| 2. Related Work | related_work.tex | Positioning | - |
| 3. Model | model.tex | Problem setup | Thm 3.1 |
| ... | ... | ... | ... |

---

## Quick Links

- [Symbols](./symbols.md) - All notation
- [Results](./results.md) - All theorems
- [Insights](./insights.md) - Key takeaways
- [Figures & Tables](./figures_tables.md) - All figures and tables
- [Changelog](./changelog.md) - Modification history

---

## Active Issues

| Priority | Issue | Location | Status |
|----------|-------|----------|--------|
| P0 | [Critical issue] | sec:X | Open |
| P1 | [Important issue] | sec:Y | In Progress |
| P2 | [Minor issue] | sec:Z | Open |

---

## Next Actions

1. [ ] [Action 1]
2. [ ] [Action 2]
```

### 2. symbols.md

```markdown
# Symbol Registry: [Paper Title]

**Last Updated**: [date]
**Total Symbols**: [count]

---

## Core Symbols

| Symbol | Meaning | Type | First Defined | Sections Used |
|--------|---------|------|---------------|---------------|
| K | Number of arms | Scalar | model:L5 | All |
| k | Arm index | Index | model:L5 | All |
| t | Time step | Index | model:L10 | All |
| F_t | Judge score at t | RV | model:L12 | 3,4,5,6,7 |
| Y_t | Human label at t | RV | model:L13 | 3,4,5,6,7 |
| π_t | Audit probability | Prob | model:L20 | 3,4,5,6,7 |

---

## Derived Quantities

| Symbol | Meaning | Formula | First Defined | Used In |
|--------|---------|---------|---------------|---------|
| R_t | Residual | Y_t - F_t | method:L5 | 4,5,6 |
| b_k(x) | Bias function | E[F|k,x] - E[Y|k,x] | model:L30 | 3,7 |
| g_k(x) | Residual variance | Var(R|k,x) | method:L15 | 5,7 |

---

## Algorithm-Specific Symbols

| Symbol | Meaning | Algorithm | Notes |
|--------|---------|-----------|-------|
| U_k(t) | Upper bound | LUCB | - |
| L_k(t) | Lower bound | LUCB | - |
| k̂(t) | Estimated best | LUCB | Use instead of b(t) |

---

## Potential Conflicts

| Symbol | Meaning 1 | Location 1 | Meaning 2 | Location 2 | Resolution |
|--------|-----------|------------|-----------|------------|------------|
| b | bias function b_k | model | best arm b(t) | algorithm | Use k̂(t) for best arm |

---

## Conventions

- **Subscripts**: k for arm, t for time, i for observation
- **Superscripts**: (F) for judge-only, (R) for residual
- **Hats**: Estimated quantities (μ̂, π̂)
- **Stars**: Optimal quantities (π*, k*)

---

## LaTeX Macros

```latex
\newcommand{\arms}{K}
\newcommand{\arm}{k}
\newcommand{\judgescore}{F}
\newcommand{\humanlabel}{Y}
\newcommand{\residual}{R}
\newcommand{\auditprob}{\pi}
```
```

### 3. results.md

```markdown
# Results Registry: [Paper Title]

**Last Updated**: [date]
**Total Results**: [count] (X theorems, Y lemmas, Z propositions)

---

## Main Theorems

| ID | Label | Section | Statement (1-line) | Proof Location | Status |
|----|-------|---------|-------------------|----------------|--------|
| Thm 1 | `thm:failure` | 3 | Judge-only selection fails | App A | ✅ Verified |
| Thm 2 | `thm:correctness` | 5 | Algorithm is δ-correct | App B | ✅ Verified |
| Thm 3 | `thm:neyman` | 5 | Optimal π* ∝ √g | App C | ✅ Verified |
| Thm 4 | `thm:cost` | 6 | Cost scales as O(1/Δ²) | App D | ✅ Verified |
| Thm 5 | `thm:lower` | 7 | Instance-dependent lower bound | App E | ✅ Verified |

---

## Propositions & Lemmas

| ID | Label | Section | Statement (1-line) | Used By |
|----|-------|---------|-------------------|---------|
| Lem 1 | `lem:cs_valid` | 4 | CS maintains coverage | Thm 2 |
| Prop 1 | `prop:decomp` | 7 | Cost decomposes | Thm 5 |

---

## Key Equations

| Eq # | Content | Section | Referenced In |
|------|---------|---------|---------------|
| (1) | IPW estimator | 4 | Sec 5, 6, 7 |
| (2) | Neyman allocation | 5 | Sec 7 |
| (3) | Cost bound | 6 | Conclusion |

---

## Result Dependencies

```
Thm 1 (Failure)
    ↓
Thm 2 (Correctness) ←── Lem 1 (CS Valid)
    ↓
Thm 3 (Neyman) ←── Prop 1 (Decomp)
    ↓
Thm 4 (Cost) ←── Thm 3
    ↓
Thm 5 (Lower Bound) ←── Thm 3, Prop 1
```

---

## Redundancy Check

| Result | Primary Location | Also Mentioned In | Action |
|--------|------------------|-------------------|--------|
| π* ∝ √g | Thm 5.2 | Sec 7.2, Prop 7.3 | ⚠️ Consolidate |

---

## Validation Status

| Result | Theory Proven | Experimentally Validated | Key Metric |
|--------|---------------|--------------------------|------------|
| Thm 1 | ✅ | ✅ exp1 | 0% accuracy |
| Thm 2 | ✅ | ✅ exp6 | 98.8% coverage |
| Thm 3 | ✅ | ✅ exp3 | 48% cost reduction |
| Thm 4 | ✅ | ✅ exp4 | slope -1.78 |
| Thm 5 | ✅ | ✅ exp4 | matches bound |
```

### 4. insights.md

```markdown
# Insights Registry: [Paper Title]

**Last Updated**: [date]
**Target Venue**: [MS/OR/ML]

---

## Main Contributions (Abstract-Level)

1. **C1**: [First contribution in one sentence]
2. **C2**: [Second contribution]
3. **C3**: [Third contribution]

---

## Managerial Insights (for MS)

### Insight 1: [Title]
**What**: [The insight]
**Why it matters**: [Business relevance]
**Actionable prescription**: [What should managers do?]
**Supporting evidence**: [Thm X, Exp Y]

### Insight 2: [Title]
...

---

## Technical Insights (for OR/ML)

### Insight 1: [Title]
**What**: [The insight]
**Why non-trivial**: [Technical significance]
**Supporting evidence**: [Thm X]

---

## Comparative Insights

| Our Approach | Prior Work | Improvement | Evidence |
|--------------|------------|-------------|----------|
| Adaptive auditing | Fixed rate | 48% cost savings | Thm 3, Exp 3 |
| IPW estimation | Naive mean | Unbiased | Lem 1 |

---

## Limitations & Future Work

| Limitation | Section Mentioned | Potential Extension |
|------------|-------------------|---------------------|
| Assumes MAR | Conclusion | Relaxation to MNAR |
| Single judge | Conclusion | Multi-judge extension |

---

## Key Takeaways for Different Audiences

### For Practitioners
1. [Actionable takeaway 1]
2. [Actionable takeaway 2]

### For Researchers
1. [Technical contribution 1]
2. [Open problem identified]

### For Reviewers
1. [Novelty claim 1]
2. [Significance claim 2]
```

### 5. figures_tables.md

> **Template**: `.claude/commands/templates/paper_state/figures_tables.md`
>
> Use the standalone template file. It includes figure registry, table registry,
> generation standards (rcParams, color palette, legend placement), LaTeX standards,
> quality audit trail, and cross-reference check sections.

### 6. changelog.md

```markdown
# Changelog: [Paper Title]

Track all modifications to maintain consistency and enable rollback.

---

## Format

```
## [Date] - [Session/Round ID]

### Changed
- [File]: [What changed] (reason)

### Added
- [File]: [What added]

### Removed
- [File]: [What removed]

### Decisions Made
- [Decision]: [Rationale]
```

---

## [2026-02-03] - Polish Round 4

### Changed
- `introduction.tex`: Rewrote opening paragraph (RAG pattern from OR_applications)
- `model.tex`: Moved Assumption A.4 to Section 7 (just-in-time placement)
- `experiments.tex`: Added managerial interpretation to Table 1

### Decisions Made
- Use "audit" consistently, not "review" or "label"
- Keep π* ∝ √g in Section 5 only, reference elsewhere

### Consistency Verified
- [x] symbols.md updated
- [x] results.md unchanged
- [x] insights.md updated

---

## [2026-02-02] - MS Style Revision

### Changed
- `introduction.tex`: Service system framing
- `conclusion.tex`: Removed "extension" language

### Added
- `related_work.tex`: Service system design literature

### Decisions Made
- Arms = service configurations (not just LLM models)

---

## [Initial] - Paper Creation

### Created
- All section files
- Initial figures and tables
```

### 7. review_responses.md

```markdown
# Review Responses: [Paper Title]

Track reviewer comments and responses systematically.

---

## Round 1 - [Date]

### Reviewer 1

| # | Comment | Category | Response | Status | Location |
|---|---------|----------|----------|--------|----------|
| R1.1 | "Arms definition too narrow" | L3-Style | Broadened to service configurations | ✅ Done | intro:L20 |
| R1.2 | "Missing managerial insights" | L3-Style | Added Discussion section | ✅ Done | discussion.tex |

### Reviewer 2

| # | Comment | Category | Response | Status | Location |
|---|---------|----------|----------|--------|----------|
| R2.1 | "Symbol b_k conflicts with b(t)" | L2-Consistency | Changed b(t) to k̂(t) | ✅ Done | algorithm.tex |

### Reviewer 3
...

---

## Round 2 - [Date]

### New Comments

| # | Comment | Category | Response | Status |
|---|---------|----------|----------|--------|
| ... | ... | ... | ... | ... |

---

## Comment Category Summary

| Category | Count | Status |
|----------|-------|--------|
| L0-Content | 2 | 2/2 resolved |
| L1-Structure | 3 | 3/3 resolved |
| L2-Consistency | 4 | 3/4 resolved |
| L3-Style | 5 | 5/5 resolved |
| L4-Language | 0 | - |

---

## Recurring Issues

| Issue | Occurrences | Root Cause | Prevention |
|-------|-------------|------------|------------|
| Symbol conflicts | 3 | No symbol registry | Use symbols.md |
| Inconsistent terms | 2 | No framing doc | Use framing.md |
```

## Workflow

### Step 1: Create Directory Structure

```bash
mkdir -p docs/paper_state/[paper_name]
```

### Step 2: Initialize Each Document

Use templates from `.claude/commands/templates/paper_state/` for initial content.

1. **overview.md**: Read all .tex files, extract section map
2. **symbols.md**: Grep all math symbols, build registry
3. **results.md**: Extract all theorem/lemma environments
4. **framing.md**: Create if doesn't exist (or link to existing)
5. **changelog.md**: Initialize with creation entry
6. **cross_references.md**: Extract all `\ref{}`, `\eqref{}` references
7. **dependencies.md**: Map assumption→theorem→proposition chains
8. **abbreviations.md**: Extract all acronyms and abbreviations
9. **figures_tables.md**: Find all `\includegraphics` and `\begin{table}`, extract captions
10. **insights.md**: Extract from abstract, intro, conclusion
11. **review_responses.md**: Initialize empty template
12. **consistency_log.md**: Initialize empty template

### Step 3: Cross-Reference

- Link results.md entries to symbols.md
- Link figures_tables.md to results they illustrate
- Link insights.md to supporting results

## Integration with Other Skills

```
/init-paper-state ms_judge    → Creates full documentation ecosystem
        ↓
/define-paper-framing         → Creates framing.md (if not exists)
        ↓
[Write paper sections]        → Consult state docs
        ↓
/review-paper-full MS         → Uses state docs for context
        ↓
/update-paper-state           → Updates all registries
        ↓
[Receive reviews]
        ↓
/track-review-comments        → Updates review_responses.md
```

## Begin

1. Parse `$ARGUMENTS` for paper name/path
2. Create directory structure
3. Read all .tex files
4. Initialize each document with extracted content
5. Report created files and next steps
6. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 Initialization Complete
   Paper: {paper_name}
   Files Created: {N}
   Location: docs/paper_state/{paper_name}/

✅ DOCUMENTATION CREATED:
   ├── overview.md      → Paper state summary
   ├── symbols.md       → Notation registry
   ├── results.md       → Theorem/lemma tracking
   ├── framing.md       → Locked terminology
   ├── changelog.md     → Modification log
   └── [others...]

🛠️ RECOMMENDED COMMANDS (in order):

   1. /define-paper-framing     → Lock terminology before writing
   2. [Write/edit paper sections]
   3. /update-paper-state       → Sync docs after changes
   4. /review-paper-full MS     → Comprehensive review

📋 WORKFLOW REMINDER:
   /session-start {paper_name}  → Start future sessions
   /update-paper-state          → Sync after any changes
   /paper-pipeline status       → Check overall progress

💡 TIP: Always update state docs after significant changes
```
