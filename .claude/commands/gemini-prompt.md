# Gemini Prompt Generator

Generate structured prompts for Gemini to fix, verify, or extend mathematical proofs in academic papers.

## Usage

```
/gemini-prompt [type] [target] [--file <path>]
```

**Types:**
- `proof-fix` — Fix a gap or error in a proof
- `proof-verify` — Verify a proof's correctness
- `proof-extend` — Extend a proof to cover additional cases
- `derivation` — Derive a new equation or result
- `simplify` — Simplify a complex proof
- `alternative` — Find alternative proof approach

**Target:** The theorem/lemma/proposition/section name (e.g., `theorem2`, `lemma3.1`, `proposition-A2`)

**Options:**
- `--file <path>` — Specify the source file containing the proof (default: auto-detect)

---

## Instructions for Claude

When the user invokes `/gemini-prompt`, follow these steps:

### Step 1: Gather Context

1. **Identify the target**: Which theorem/lemma/proof needs work?
2. **Read the source**: Find and read the relevant proof in the paper
3. **Understand the gap**: What specifically is wrong or missing?
4. **Check literature**: Are there relevant technique files in the repo?

### Step 2: Generate Structured Prompt

Use the following template:

```markdown
# Prompt for Gemini: [Action] [Target]

## Context

**Paper**: [Paper title or working name]
**Section**: [Section number/name containing the proof]
**File**: [Path to the file, e.g., `paper/appendix.tex`]
**Lines**: [Approximate line numbers, e.g., lines 209-270]

[1-2 sentence description of what this proof is about]

## Current State

[Quote the relevant portion of the current proof, or summarize it]

```latex
[Include key equations if helpful]
```

## The Problem

### What's Wrong/Missing

[Specific technical description of the gap or error]

### Why It Matters

[Why this affects the paper's correctness or rigor]

## Required Fix

### Step A: [First task]

[Mathematical details, equations, what needs to be derived]

### Step B: [Second task]

[Next step in the fix]

### Step C: [Verification]

[How to verify the fix is correct]

## Hints

### Hint 1: [Approach suggestion]
[Mathematical hint without giving away the answer]

### Hint 2: [Relevant technique]
[Reference to literature or standard technique]

### Hint 3: [Simplification]
[Suggest simplified case to try first if helpful]

## Deliverables

1. **[Specific output 1]**: [Description]
2. **[Specific output 2]**: [Description]
3. **LaTeX code**: Ready to insert at [location]

## Verification Checklist

Before submitting, verify:
- [ ] [Mathematical check 1]
- [ ] [Mathematical check 2]
- [ ] [Boundary/edge case check]
- [ ] Paper compiles without errors

## Output Format

```latex
% Insert at [location] in [file]:

[Expected LaTeX structure]
```
```

### Step 3: Save the Prompt

Save to: `paper/prompts/[target]_[type]_prompt.md`

Or if the user specifies a different location, use that.

---

## Prompt Quality Principles

### 1. Mathematical Precision

- ✅ Include exact equations with proper notation
- ✅ Specify all boundary conditions and constraints
- ✅ Reference specific line numbers when possible
- ❌ Don't be vague about what's wrong
- ❌ Don't skip mathematical details

### 2. Actionable Structure

- ✅ Break into clear, sequential steps
- ✅ Each step should be independently verifiable
- ✅ Include hints that guide without solving
- ❌ Don't make it one monolithic task
- ❌ Don't give the answer in the hints

### 3. Verification Focus

- ✅ Include explicit verification checklist
- ✅ Specify what "correct" looks like
- ✅ Ask for boundary condition checks
- ❌ Don't trust output without verification criteria

### 4. Context Completeness

- ✅ Reference relevant literature/techniques
- ✅ Link to notation definitions if non-standard
- ✅ Mention related results in the paper
- ❌ Don't assume Gemini knows your notation

### 5. Minimal Disruption

- ✅ Specify exact insertion point
- ✅ Ask for minimal changes
- ✅ Preserve what already works
- ❌ Don't request unnecessary rewrites

---

## Type-Specific Guidelines

### `proof-fix`

Focus on:
- Identifying the exact logical gap
- Providing the mathematical context needed to fill it
- Suggesting which technique might work

Example gaps:
- "Claimed X but didn't prove it"
- "Assumed Y without justification"
- "Missing case analysis for Z"

### `proof-verify`

Focus on:
- Checking each logical step
- Verifying calculations
- Testing boundary cases

Ask Gemini to:
- Re-derive key equations independently
- Check signs, indices, boundary conditions
- Identify any hidden assumptions

### `proof-extend`

Focus on:
- What new cases need coverage
- How the current proof structure can accommodate them
- What modifications are needed

### `derivation`

Focus on:
- Clear statement of what to derive
- All given information and constraints
- Expected form of the result (if known)

### `simplify`

Focus on:
- Which parts are overly complex
- What the key insight should be
- Acceptable trade-offs (generality vs. clarity)

### `alternative`

Focus on:
- Why current approach is problematic
- What properties an alternative should have
- Any constraints on the new approach

---

## Notation Handling

When the paper uses non-standard notation, include a notation table:

```markdown
## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $X$ | [Description] |
| $Y$ | [Description] |
| $\alpha$ | [Description] |
```

---

## Literature Integration

If the repo contains literature summaries, reference them:

```markdown
## Relevant Literature

- **[Author et al. Year]**: [Technique that applies]
  - See: `paper/literature/[file].md`, Section [X]
- **[Author et al. Year]**: [Another relevant technique]
```

---

## Iterative Workflow

```
┌─────────────────────────────────────────────────────────┐
│  1. Identify Gap                                        │
│     └─> Read proof, find specific issue                 │
├─────────────────────────────────────────────────────────┤
│  2. Generate Prompt                                     │
│     └─> /gemini-prompt proof-fix [target]               │
├─────────────────────────────────────────────────────────┤
│  3. Send to Gemini                                      │
│     └─> Copy prompt, get response                       │
├─────────────────────────────────────────────────────────┤
│  4. Verify Result                                       │
│     └─> Check math independently                        │
│     └─> If issues: go to step 5                         │
│     └─> If correct: done                                │
├─────────────────────────────────────────────────────────┤
│  5. Generate Follow-up                                  │
│     └─> /gemini-prompt proof-fix [target]-v2            │
│     └─> Go to step 3                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Examples

### Example 1: Fixing a Proof Gap

**User**: `/gemini-prompt proof-fix theorem2`

**Claude Actions**:
1. Read theorem 2's proof in the paper
2. Identify the specific gap (e.g., "monotonicity claimed but not proved")
3. Generate structured prompt with:
   - Current proof state
   - Exact mathematical gap
   - Hints for fixing
   - Verification checklist

### Example 2: Verifying a Complex Derivation

**User**: `/gemini-prompt proof-verify lemma-A3`

**Claude Actions**:
1. Read lemma A3's proof
2. Generate prompt asking Gemini to:
   - Re-derive each step
   - Check all calculations
   - Verify boundary conditions
   - List any hidden assumptions

### Example 3: Extending to New Cases

**User**: `/gemini-prompt proof-extend proposition1 --file paper/theory.tex`

**Claude Actions**:
1. Read proposition 1 in theory.tex
2. Understand current scope
3. Generate prompt for extending to requested new cases

---

## Common Patterns

### Pattern: Missing Inequality Proof

```markdown
## The Problem
The proof claims [inequality] but only provides intuition, not a rigorous proof.

## Required Fix
### Step A: Setup
Write down the expressions for both sides of the inequality.

### Step B: Analysis
[Suggest technique: direct comparison, auxiliary function, etc.]

### Step C: Verify
Check that the inequality is strict/non-strict as claimed.
```

### Pattern: Unverified Boundary Condition

```markdown
## The Problem
The solution satisfies the ODE but boundary condition at [point] is not verified.

## Required Fix
### Step A: State the boundary condition
[Exact mathematical statement]

### Step B: Substitute and verify
Plug the solution into the boundary condition.

### Step C: Handle edge cases
Check behavior as [parameter] → [limit].
```

### Pattern: Missing Case in Case Analysis

```markdown
## The Problem
The proof handles cases [A, B, C] but case [D] is not addressed.

## Required Fix
### Step A: Characterize case D
When does case D occur? What are its properties?

### Step B: Analyze case D
Apply the same proof technique or explain why it differs.

### Step C: Verify completeness
Confirm that A, B, C, D cover all possibilities.
```

---

## Anti-Patterns to Avoid

❌ **Too Vague**
> "The proof has some issues, please fix them."

✅ **Specific**
> "Line 245 claims $p_1(t) \le p_1(T)$ but this requires $(1-p)p_2 < p_1$ which is not proved."

❌ **Solving in the Prompt**
> "You should use integration by parts to get [exact answer]."

✅ **Guiding**
> "Hint: Consider using integration by parts on the second term."

❌ **No Verification**
> "Please fix the proof."

✅ **With Verification**
> "After fixing, verify: (1) ODE is satisfied, (2) boundary conditions hold, (3) inequality is strict."
