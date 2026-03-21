# Review Paper Full - Comprehensive Multi-Level Paper Review

You are a comprehensive paper reviewer. Your task is to perform a full review covering ALL levels of paper quality, not just language.

## ⚠️ MANDATORY: RAG-Grounded Review

**All suggestions MUST be grounded in human-authored patterns from the writing reference library.**

See: `.claude/commands/_shared/rag_config.md` for the full RAG protocol.

### Step 0: Read Shared Config & RAG References

Before starting any review, execute these Read tool calls:

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/phrases/transitions.md
Read .claude/writing_references/guides/academic_writing_principles.md
```

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before starting the review, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL state files for comprehensive context:

```
Read docs/paper_state/{resolved}/overview.md       # Current status
Read docs/paper_state/{resolved}/framing.md        # Locked terminology
Read docs/paper_state/{resolved}/symbols.md        # Symbol registry
Read docs/paper_state/{resolved}/results.md        # Theorem registry
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- overview.md: [paper status, page count, section count]
- framing.md: [locked terms and definitions]
- symbols.md: [key symbol definitions and potential conflicts]
- results.md: [theorem registry with labels and locations]
```

**If you skip this step, your review will miss known constraints and flag already-resolved issues.**

Then, based on paper topic, read topic-specific references:
- **BAI/Bandits**: `sentences/algorithm_optimality.md`, `sentences/online_learning.md`
- **OR/Service Systems**: `sentences/or_applications.md`, `sentences/revenue_management.md`
- **LLM/AI**: `sentences/llm_papers.md`

---

## The Problem This Solves

Previous approach (WRONG):
```
polish-paper → Check AI words, passive voice, transitions
              → "Converged" after 4 rounds
              → Reviewer still has 14 comments
```

This approach (RIGHT):
```
review-paper-full → Level 0: Content & Concepts
                  → Level 1: Structure & Redundancy
                  → Level 2: Consistency (symbols, terms)
                  → Level 3: Journal Style (MS vs ML)
                  → Level 4: Language & Polish
                  → Real convergence when ALL levels pass
```

## Arguments

- `$ARGUMENTS` - Required: target venue. Supported values:
  - `MS` - Management Science (managerial insights, service framing)
  - `OR` - Operations Research (methodological contribution, proofs)
  - `ML` or `NeurIPS` or `ICML` - ML venues (experiments, baselines, ablations)
  - `JMLR` - Journal of ML Research (theory + experiments balance)

## The 5-Level Review Framework

```
┌─────────────────────────────────────────────────────────┐
│ Level 0: CONTENT (Most Important)                       │
│ - Are concepts defined broadly enough?                  │
│ - Are results non-redundant?                            │
│ - Do experiments match the framing?                     │
├─────────────────────────────────────────────────────────┤
│ Level 1: STRUCTURE                                      │
│ - Are sections logically organized?                     │
│ - Is there content redundancy across sections?          │
│ - Do examples appear near their related theorems?       │
├─────────────────────────────────────────────────────────┤
│ Level 2: CONSISTENCY                                    │
│ - Are symbols used consistently?                        │
│ - Are terms defined once and used uniformly?            │
│ - Does algorithm match model setup?                     │
├─────────────────────────────────────────────────────────┤
│ Level 3: JOURNAL STYLE                                  │
│ - Does framing match target journal?                    │
│ - Are there managerial insights (for MS)?               │
│ - Is notation interpreted operationally?                │
├─────────────────────────────────────────────────────────┤
│ Level 4: LANGUAGE (Least Important - do last)           │
│ - AI word removal                                       │
│ - Passive voice                                         │
│ - Transitions                                           │
└─────────────────────────────────────────────────────────┘
```

## Workflow

### Step 0: Setup

1. Identify paper directory
2. Note target journal from `$ARGUMENTS`
3. List all .tex files to review

### Step 1: Level 0 - Content Review

**Run these checks:**

#### 1.1 Concept Definition Breadth

For the core concepts, check:

| Concept | Question | Good | Bad |
|---------|----------|------|-----|
| Arms | Defined as broad design options? | "service configurations varying in operations, technology, parameters" | "K different LLM models" |
| Audit | Operational interpretation given? | "supervisor review of interaction quality" | "binary indicator A_t" |
| Cost | Business meaning clear? | "human expert time at $0.20/review" | "c_Y per observation" |

**RAG Reference**: When suggesting broader definitions, check `sentences/or_applications.md` and `sentences/problem_setup.md` for how top papers frame operational concepts.

**Output**: List of concepts needing broader definition

#### 1.2 Result Uniqueness

For each theorem/proposition:
- Is this result stated elsewhere?
- Could it be merged with another result?
- Is the proof sketch redundant with another section?

**Output**: List of redundant results

#### 1.3 Experiment-Framing Alignment

Do the experiments demonstrate what the framing promises?

| Framing Promise | Experiment Delivers? |
|-----------------|---------------------|
| "service configurations" | Just LLM model comparison? ⚠️ |
| "operational decisions" | Algorithm benchmarks only? ⚠️ |
| "managerial insights" | Only accuracy numbers? ⚠️ |

**Output**: Framing-experiment mismatches

### Step 2: Level 1 - Structure Review

**Run these checks:**

#### 2.1 Section Organization

For each section:
```
| Section | Unique Contribution | Could Merge With | Could Move to Appendix |
```

#### 2.2 Example-Theorem Pairing

For each example:
- Is the related theorem in the same section?
- Does example come after the general result?

#### 2.3 Preview-Detail-Summary Pattern

For each major result:
- Brief preview in intro?
- Full statement in ONE section?
- Brief summary in conclusion?
- NO detailed restatement elsewhere?

**RAG Reference**: Use `paragraphs/paper_roadmap.md` for section organization patterns, `paragraphs/main_results.md` for how to present results in ONE location.

**Output**: Structure issues with specific reorganization suggestions

### Step 3: Level 2 - Consistency Review

**Run these checks:**

#### 3.1 Symbol Conflicts

Build symbol registry, find conflicts:
```
| Symbol | Meaning 1 | Location 1 | Meaning 2 | Location 2 |
```

#### 3.2 Term Consistency

Check key terms are used uniformly:
- "arm" vs "alternative" vs "configuration"
- "audit" vs "review" vs "label"
- "judge" vs "proxy" vs "automated evaluator"

#### 3.3 Model-Algorithm Match

Verify algorithm implements what model describes:
- Same observation order?
- Same notation?
- Same constraints?

**RAG Reference**: Use `sentences/problem_setup.md` for standard notation patterns in top papers.

**Output**: Consistency issues with specific fixes

### Step 4: Level 3 - Venue-Specific Style Review

**Invoke the appropriate style skill based on `$ARGUMENTS`:**

| Argument | Skill to Invoke | RAG References |
|----------|-----------------|----------------|
| `MS` | `/check-ms-style` | `or_applications.md`, `revenue_management.md`, `dynamic_pricing.md` |
| `OR` | `/check-or-style` | `or_applications.md`, `queueing_theory.md`, `robust_optimization.md` |
| `ML`/`NeurIPS`/`ICML` | `/check-ml-style` | `llm_papers.md`, `online_learning.md`, `learning_theory.md` |
| `JMLR` | `/check-ml-style` (with theory emphasis) | `algorithm_optimality.md`, `proof_structure.md` |

**For Management Science ($ARGUMENTS = "MS"):**
- Run `/check-ms-style` - checks managerial insights, service framing, prescriptions
- Key: Arms as service design options, actionable prescriptions, business context

**For Operations Research ($ARGUMENTS = "OR"):**
- Run `/check-or-style` - checks methodological contribution, proof rigor
- Key: Complexity results, optimality guarantees, complete proofs

**For ML Venues ($ARGUMENTS = "ML"/"NeurIPS"/"ICML"):**
- Run `/check-ml-style` - checks experiments, baselines, ablations
- Key: Comprehensive experiments, strong baselines, reproducibility

**Output**: Style issues with rewrite suggestions

### Step 5: Level 4 - Language Review

**Only after Levels 0-3 pass:**

- AI word detection
- Passive voice
- Transition variety
- Subject-verb proximity

**Output**: Language issues (lowest priority)

## Convergence Criteria

**Real convergence requires ALL levels to pass:**

```
Level 0 (Content):     □ Pass  □ Fail
Level 1 (Structure):   □ Pass  □ Fail
Level 2 (Consistency): □ Pass  □ Fail
Level 3 (Style):       □ Pass  □ Fail
Level 4 (Language):    □ Pass  □ Fail

Overall: PASS only if all levels pass
```

**NOT convergent if:**
- Level 4 passes but Level 0 has issues (common mistake!)
- "Modifications < 3" but Level 1 has structural problems
- Language is clean but concepts are too narrow

## Output Format

```markdown
# Full Paper Review Report

**Paper**: [title]
**Target Journal**: [journal]
**Date**: [date]

---

## Executive Summary

| Level | Status | Critical Issues |
|-------|--------|-----------------|
| 0: Content | ✅/⚠️/❌ | [count] |
| 1: Structure | ✅/⚠️/❌ | [count] |
| 2: Consistency | ✅/⚠️/❌ | [count] |
| 3: Style | ✅/⚠️/❌ | [count] |
| 4: Language | ✅/⚠️/❌ | [count] |

**Overall**: Ready / Needs Revision / Major Revision

---

## Level 0: Content Issues

### Concept Definition
[Issues found]

### Result Redundancy
[Issues found]

### Experiment-Framing Alignment
[Issues found]

---

## Level 1: Structure Issues

### Section Organization
[Issues found]

### Example-Theorem Pairing
[Issues found]

---

## Level 2: Consistency Issues

### Symbol Conflicts
[Issues found]

### Term Consistency
[Issues found]

---

## Level 3: Style Issues (for [journal])

### [Journal-specific checks]
[Issues found]

---

## Level 4: Language Issues

[Only if Levels 0-3 pass]

---

## Priority Actions

### Must Fix Before Submission
1. [Critical Level 0 issue]
2. [Critical Level 1 issue]
...

### Should Fix
1. [Moderate issue]
...

### Nice to Fix
1. [Minor issue]
...

---

## Detailed Revision Suggestions

**⚠️ All revision suggestions MUST cite a RAG pattern.**

### [Issue 1]: [Title]
**Level**: [0-4]
**Location**: [file:line]
**Current**:
> [quote]

**Problem**: [explanation]

**RAG Pattern**: [cite source, e.g., "From sentences/contribution.md - Kaufmann2016"]
> [the reference pattern]

**Suggested revision** (adapted from pattern):
> [rewrite grounded in the pattern]

[Continue for each issue]
```

## Begin

1. Note target journal from arguments
2. Read all .tex files
3. Execute Level 0 checks (content)
4. Execute Level 1 checks (structure)
5. Execute Level 2 checks (consistency)
6. Execute Level 3 checks (journal style)
7. Only if 0-3 pass, execute Level 4 (language)
8. Generate comprehensive report
9. Determine true convergence status
