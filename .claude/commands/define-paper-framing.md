# Define Paper Framing - Pre-Writing Concept Definition

You are a paper framing agent. Your task is to define core concepts, tone, and terminology **BEFORE** writing begins, preventing inconsistency issues later.

## Why This Prevents Reviewer Comments

**Without framing document:**
```
Write introduction → Use "arms" = LLM models
Write experiments → Use "arms" = service configurations
Reviewer: "Arms definition inconsistent, too narrow in intro"
```

**With framing document:**
```
/define-paper-framing → Lock "arms" = service configurations (3 layers)
Write introduction → Check framing doc → Use locked definition
Write experiments → Check framing doc → Use same definition
Reviewer: No comment on arms definition ✓
```

## ⚠️ Protocol Reference

This command modifies `framing.md` (and may seed `symbols.md`, `abbreviations.md`). After any state doc modification, you MUST update `changelog.md` with a dated entry. See `.claude/commands/_shared/unified_protocol.md` Step 4 (Universal Post-Edit Rule).

## Arguments

- `$ARGUMENTS` - Optional: target venue (MS, OR, ML) to load venue-specific templates

## Setup: Resolve Paper Name

**⚠️ MANDATORY.** Before creating any framing document:
1. Resolve `[paper_name]` → run `ls docs/paper_state/` to find the actual directory name
2. If existing state docs exist, read them FIRST to avoid contradictions:

```
Read docs/paper_state/{resolved}/framing.md       # May already exist
Read docs/paper_state/{resolved}/overview.md      # Current paper status
```

## Output Location

Creates/updates: `docs/paper_state/{resolved}/framing.md`

This file is the **single source of truth** for:
1. Core concept definitions
2. Terminology choices
3. Target audience and tone
4. Key claims and their evidence

## Workflow

### Step 1: Define Core Concepts

For each core concept in the paper, lock down:

```markdown
## Core Concept: [Name]

### Definition (LOCKED)
> [Exact wording to use throughout the paper]

### Scope
- Includes: [what this concept covers]
- Excludes: [what this concept does NOT cover]

### Examples (use these consistently)
1. [Example 1]
2. [Example 2]

### Synonyms (AVOID these to prevent confusion)
- ❌ Do NOT use: [alternative term 1]
- ❌ Do NOT use: [alternative term 2]

### First Introduced
- Section: [where first defined]
- Line: [approximate line number]
```

**Example for "Arms":**
```markdown
## Core Concept: Arms

### Definition (LOCKED)
> Arms represent deployable service configurations. Each arm encodes:
> (1) operational design (workflow, routing, priority),
> (2) technology choices (LLM model, reasoning depth, tools), and
> (3) parameter settings (thresholds, response length).

### Scope
- Includes: Any combination of operations + technology + parameters
- Excludes: Individual parameter values (those are within-arm variations)

### Examples
1. Customer support chatbot: routing rules × LLM model × escalation threshold
2. Content moderation: strictness level × classifier × appeal routing

### Synonyms (AVOID)
- ❌ "models" (too narrow, implies only technology layer)
- ❌ "alternatives" (too generic)
- ✅ Use: "configurations", "service designs", "design options"

### First Introduced
- Section: Introduction, paragraph 3
```

### Step 2: Define Terminology Mappings

Create a locked terminology table:

```markdown
## Terminology Mappings

| Concept | LOCKED Term | Avoid | Reason |
|---------|-------------|-------|--------|
| Biased automated score | "judge score" | "proxy", "prediction" | MS readability |
| Ground truth label | "human audit" | "label", "annotation" | Service framing |
| Selection probability | "audit probability" | "propensity", "p(audit)" | Operational meaning |
| Best alternative | "best arm" | "optimal arm", "winner" | Bandit convention |
```

### Step 3: Define Tone and Audience

```markdown
## Target Audience

**Primary**: Operations managers designing service evaluation systems
**Secondary**: ML researchers interested in bandit algorithms

## Tone Guidelines

| Aspect | Do | Don't |
|--------|------|-------|
| Results | "yields prescriptions for managers" | "achieves SOTA performance" |
| Method | "enables practitioners to..." | "we propose a novel algorithm" |
| Limitations | "when quality gaps are small, additional auditing has diminishing returns" | "even aggressive auditing cannot fully correct" |
```

### Step 4: Define Key Claims

Lock down the main claims and their evidence:

```markdown
## Key Claims (each stated ONCE, referenced elsewhere)

### Claim 1: Proxy-only selection fails
- **Statement location**: Section 3, Theorem 3.1
- **Evidence**: Impossibility proof
- **How to reference elsewhere**: "By Theorem 3.1, proxy-only selection fails when..."

### Claim 2: Neyman allocation is optimal
- **Statement location**: Section 5, Theorem 5.2
- **Evidence**: Variance minimization proof
- **How to reference elsewhere**: "The optimal audit policy (Theorem 5.2) satisfies π* ∝ √g"
- **⚠️ Do NOT restate in**: Section 7 (only reference)
```

### Step 5: Define Symbol Registry (Pre-allocated)

```markdown
## Symbol Registry

| Symbol | Meaning | Reserved For | Conflicts To Avoid |
|--------|---------|--------------|-------------------|
| k | arm index | all sections | don't use for iteration |
| K | number of arms | all sections | - |
| t | time step | all sections | don't use for threshold |
| b_k(x) | bias function | Section 3 | don't use b(t) for best arm |
| k̂(t) | estimated best arm | Section 5 | use this instead of b(t) |
| π | audit probability | all sections | - |
| π* | optimal audit | Section 5, 7 | - |
```

## Output Format

```markdown
# Paper Framing Document: [Title]

**Created**: [date]
**Target Venue**: [MS/OR/ML]
**Last Updated**: [date]

---

## 1. Core Concepts

[For each concept: definition, scope, examples, synonyms to avoid]

---

## 2. Terminology Mappings

[Locked term choices]

---

## 3. Tone and Audience

[Target audience, tone guidelines]

---

## 4. Key Claims Registry

[Each claim: where stated, how to reference]

---

## 5. Symbol Registry

[Pre-allocated symbols to prevent conflicts]

---

## 6. Section Dependency Graph

```
Introduction
    ↓ (defines arms, problem)
Model (Section 3)
    ↓ (defines symbols: F, Y, π)
    ↓ (states Theorem 3.1 - failure modes)
Estimation (Section 4)
    ↓ (builds on Section 3 symbols)
Algorithm (Section 5)
    ↓ (states Theorem 5.2 - Neyman, referenced later)
Analysis (Section 6)
    ↓ (uses Theorem 5.2 by reference only)
Lower Bounds (Section 7)
    ↓ (references Theorem 5.2, does NOT restate)
...
```

---

## 7. Consistency Checklist (Use Before Each Section)

Before writing/editing any section:
- [ ] Check core concept definitions match framing doc
- [ ] Check terminology matches locked terms
- [ ] Check new symbols don't conflict with registry
- [ ] Check new results aren't restating existing claims
- [ ] Check tone matches target audience
```

## Integration with Other Skills

```
/define-paper-framing    → Creates framing doc (BEFORE writing)
       ↓
[Write paper sections]   → Consult framing doc while writing
       ↓
/paper-overview create   → Verify paper matches framing doc
       ↓
/check-paper-consistency → Detect any drift from framing
```

## Begin

1. Identify paper title and target venue
2. Extract core concepts from existing draft (or define new)
3. Lock terminology, tone, claims, symbols
4. Create section dependency graph
5. Save to `docs/paper_state/[paper_name]_framing.md`
6. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 Framing Complete
   Paper: {paper_name}
   Venue: {target_venue}
   Saved: docs/paper_state/{paper_name}/framing.md

✅ LOCKED ELEMENTS:
   - Core Concepts: {N} defined
   - Terminology: {M} terms locked
   - Key Claims: {K} claims specified
   - Symbols: {S} notations defined

🛠️ RECOMMENDED COMMANDS (in order):

   1. [Write paper sections]  → Consult framing.md while writing
   2. /check-term-consistency → Verify terms match framing
   3. /check-paper-consistency → Verify symbols match framing
   4. /update-paper-state     → Sync after significant changes

📋 WORKFLOW REMINDER:
   - ALWAYS check framing.md before introducing new terms
   - Run /check-term-consistency if unsure about terminology
   - Update framing.md if scope changes (rare!)

💡 TIP: Framing should rarely change. If it does, re-run all checks.
```
