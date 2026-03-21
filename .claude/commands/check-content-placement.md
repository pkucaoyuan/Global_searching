# Check Content Placement - Verify Assumptions, Examples, Proofs in Correct Locations

You are a content placement checker. Your task is to verify that technical content (assumptions, examples, proofs, definitions) appears in the optimal location according to academic writing conventions.

## ⚠️ MANDATORY: Unified Protocol

### Step 0: Read Shared Config
```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
```

### Step 1: Read RAG Files
```
Read .claude/writing_references/paragraphs/proof_structure.md
Read .claude/writing_references/sentences/problem_setup.md
```

### Step 2: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before checking content placement, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/dependencies.md   # Assumption → Result chains
Read docs/paper_state/{resolved}/results.md        # Where theorems are defined
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- dependencies.md: [number of dependency chains, key assumption→theorem links]
- results.md: [theorem locations and their current section placements]
```

**If you skip this step, you will suggest moving content that has deliberate placement dependencies.**

### Step 3: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

This ensures the RAG library self-maintains by logging gaps and suggesting additions.

---

## Why This Prevents Reviewer Comments

**Comment 4**: "Assumptions 3.4, 3.5 过早引入"
**Comment 7**: "Proof sketches 移到附录"
**Comment 11**: "EC.2 motivating examples → Introduction"

These problems arise when:
- Assumptions introduced before they're needed
- Proof sketches in main text instead of appendix
- Examples far from their related theorems
- Motivating examples in appendix instead of introduction

## Arguments

- `$ARGUMENTS` - Optional: path to paper directory

## Placement Rules

### Rule 1: Assumptions - "Just-In-Time" Principle

**Detection threshold**: Flag any assumption defined **2+ sections before** its first use (excluding universal assumptions like boundedness that are used throughout).

**❌ Bad**: All assumptions in Section 3 (model section)
**✅ Good**: Each assumption introduced just before first use

```markdown
## Assumption Placement Analysis

| Assumption | Currently | First Used | Recommendation |
|------------|-----------|------------|----------------|
| A1 (Bounded) | model.tex:L10 | model.tex:L15 | ✅ OK |
| A2 (MAR) | model.tex:L20 | method.tex:L5 | ✅ OK (core to method) |
| A3 (Positivity) | model.tex:L30 | method.tex:L8 | ✅ OK (core to method) |
| A4 (Smooth g) | model.tex:L45 | theory_lb.tex:L20 | ⚠️ Move to Section 7 |
| A5 (LAN) | model.tex:L60 | theory_lb.tex:L40 | ⚠️ Move to Section 7 |
```

**Exception**: Assumptions needed throughout (like boundedness) should be early.

### Rule 2: Examples - "Near Related Theorem" Principle

**Detection threshold**: Flag any example whose related theorem/proposition is **in a different section** (>0 section boundary away). Examples should be in the same section as their theorem, ideally immediately after.

**❌ Bad**: Example in Section 6, related Theorem in Section 7
**✅ Good**: Example immediately follows its Theorem (as subsection or remark)

```markdown
## Example-Theorem Pairing

| Example | Related Result | Distance | Recommendation |
|---------|----------------|----------|----------------|
| Ex 6.1 (Gaussian) | Thm 7.1 (Lower Bound) | 1 section | ⚠️ Move to Section 7.1.1 |
| Ex 5.1 (Service) | Def 5.1 (Arms) | Same section | ✅ OK |
```

**Pattern for examples:**
```latex
\begin{theorem}[General Result]
...
\end{theorem}

\begin{example}[Illustrative Special Case]
% Immediately after theorem
...
\end{example}
```

### Rule 3: Proofs - "Main vs Appendix" Principle

**Detection threshold**: Flag any proof or proof sketch in the main text that exceeds **10 lines** (for conference papers) or **15 lines** (for journal papers). Proof sketches should convey key insight in 3-8 lines; full proofs belong in the appendix.

| Proof Type | Location | Rationale |
|------------|----------|-----------|
| 1-2 line proofs | Inline | Reader can verify immediately |
| Key insight proofs | Main text (proof sketch) + Appendix (full) | Intuition in main, rigor in appendix |
| Technical lemmas | Appendix only | Doesn't distract from main flow |
| Straightforward proofs | Appendix only | Standard techniques |

**❌ Bad (main text):**
```latex
\begin{proof}
We proceed by induction. For the base case... [2 pages]
\end{proof}
```

**✅ Good (main text):**
```latex
\begin{proofsketch}
The key insight is the martingale structure of IPW residuals (Lemma B.1),
which enables time-uniform concentration. Full proof in Appendix B.
\end{proofsketch}
```

### Rule 4: Motivating Examples - "Introduction" Principle

**Detection threshold**: Flag any motivating example (one that illustrates the problem being solved, not a technical result) that is placed **in the appendix** or **after the method section**. Motivating examples must appear in or near the introduction.

**❌ Bad**: Motivating examples in Appendix
**✅ Good**: Motivating examples in Introduction (after problem statement)

```markdown
## Motivating Example Placement

| Example | Purpose | Currently | Recommendation |
|---------|---------|-----------|----------------|
| Call center QA | Motivate problem | Appendix EC.2 | ⚠️ Move to Intro §1.1 |
| Content moderation | Show breadth | Appendix EC.2 | ⚠️ Move to Intro §1.1 |
| Technical example | Illustrate theorem | Section 7 | ✅ OK (not motivating) |
```

### Rule 5: Definitions - "Before First Use" Principle

**❌ Bad**: Define symbol after using it
**✅ Good**: Define symbol in the paragraph where first used

```markdown
## Definition Placement

| Symbol/Concept | Defined | First Used | Gap |
|----------------|---------|------------|-----|
| θ_k | model.tex:L12 | model.tex:L14 | 2 lines ✅ |
| g(x,f) | theory.tex:L30 | algorithm.tex:L15 | Wrong file ⚠️ |
```

### Rule 6: Related Work - "After Introduction" or "Before Conclusion"

| Venue | Convention |
|-------|------------|
| MS/OR | After Introduction (show positioning early) |
| NeurIPS/ICML | Before Conclusion (focus on contribution first) |
| JMLR | Either (author preference) |

## Workflow

### Phase 1: Extract All Technical Elements

Build a registry:

```bash
grep -n "\\begin{assumption}\|\\begin{example}\|\\begin{proof}\|\\begin{definition}" sections/*.tex
```

### Phase 2: Build Dependency Graph

For each element, identify:
- Where it's defined
- Where it's first used
- Where it's referenced

### Phase 3: Check Placement Rules

Apply each rule and flag violations.

### Phase 4: Generate Relocation Suggestions

For each violation, suggest:
- What to move
- Where to move it
- How to update references

## Output Format

```markdown
# Content Placement Report

**Paper**: [title]
**Date**: [date]

---

## 1. Assumptions

| Assumption | Current Location | First Used | Recommendation |
|------------|------------------|------------|----------------|
| A3.4 (Smooth g) | model.tex:L45 | theory_lb:L20 | Move to Section 7 |
| A3.5 (LAN) | model.tex:L60 | theory_lb:L40 | Move to Section 7 |

**Suggested edit for A3.4:**
```latex
% DELETE from model.tex:L45-55

% ADD to theory_lower_bound.tex before Theorem 7.1:
To state the lower bound, we introduce additional regularity:
\begin{assumption}[Smooth residual second moment]
...
\end{assumption}
```

---

## 2. Examples

| Example | Related Result | Current | Recommendation |
|---------|----------------|---------|----------------|
| Ex 6.1 | Thm 7.1 | analysis.tex | Move to theory_lb.tex §7.1.1 |
| EC.2 (Motivating) | Problem statement | appendix | Move to intro.tex §1.1 |

---

## 3. Proofs

| Proof | Current | Recommendation |
|-------|---------|----------------|
| Thm 7.1 proof sketch | theory_lb.tex (15 lines) | Keep sketch, move details to Appendix |
| Thm 8.1 proof sketch | delays.tex (20 lines) | Shorten to 5 lines, full proof in Appendix |
| Lemma B.1 | appendix | ✅ OK (technical lemma) |

---

## 4. Definitions

| Symbol | Defined | First Used | Issue |
|--------|---------|------------|-------|
| g(x,f) | theory.tex:L30 | alg.tex:L15 | Define earlier |

---

## 5. Summary

| Category | Violations | Severity |
|----------|------------|----------|
| Assumption placement | 2 | 🟡 Medium |
| Example placement | 2 | 🟡 Medium |
| Proof length in main | 2 | 🟡 Medium |
| Definition order | 1 | 🟢 Minor |

---

## 6. Priority Relocations

1. Move A3.4, A3.5 to Section 7 (before lower bound theorem)
2. Move Example 6.1 to Section 7.1.1 (after Theorem 7.1)
3. Move EC.2 motivating examples to Introduction §1.1
4. Shorten proof sketches in Sections 7-8
```

## Begin

1. Extract all assumptions, examples, proofs, definitions
2. Build dependency graph (where defined → where used)
3. Apply placement rules
4. Generate relocation suggestions with concrete edits
5. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: L1 Structure (Placement)
   Issues Found: {N}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   /fix-issues placement       → Auto-move content (with confirmation)
   /fix-issues placement --dry-run → Preview moves first

   {If no issues:}
   ✅ All content properly placed. Proceed to next check.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found:]
   /fix-issues placement       → Auto-move content
   /check-content-placement    → Verify fixes applied

   [Same level (L1) - complete these:]
   /check-paper-flow           → Check section transitions

   [When L1 passes:]
   /check-paper-consistency    → Move to L2 (consistency)

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy     ✅
 → L1 Structure   ─────────── YOU ARE HERE
   L2 Consistency ─────────── /check-paper-consistency
   L3 Style       ─────────── /check-ms-style
   L4 Language    ─────────── /polish-paper

💡 TIP: Use /paper-pipeline status to see overall progress
```
