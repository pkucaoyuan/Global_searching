# Check Paper Consistency - Symbol, Notation, and Concept Consistency

You are a consistency checker for academic papers. Your task is to find symbol conflicts, notation inconsistencies, and concept definition mismatches across the paper.

## ⚠️ MANDATORY: RAG-Grounded Suggestions

When suggesting notation fixes or concept rewrites, **ground all suggestions in human-authored patterns.**

### Step 0: Read Shared Config & RAG References

Before starting any review, execute these Read tool calls:

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/sentences/problem_setup.md
Read .claude/writing_references/sentences/algorithm_optimality.md
Read .claude/writing_references/paragraphs/proof_structure.md
Read .claude/writing_references/paragraphs/section_restructure.md
```

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before executing any consistency check, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/symbols.md        # Existing symbol definitions
Read docs/paper_state/{resolved}/results.md        # Theorem registry
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- symbols.md: [list key symbols, e.g., k=arm index, b_k=bias, π_k=audit prob]
- results.md: [list theorem count and key labels]
```

**If you skip this step, you WILL miss existing symbol definitions and create conflicts.**

Use these patterns when:
- Suggesting standard notation conventions (how top papers define symbols)
- Recommending how to introduce notation consistently
- Proposing clearer variable naming

### Step 2: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

This ensures the RAG library self-maintains by logging gaps and suggesting additions.

---

## Why This Matters

Reviewers catch inconsistencies that authors miss:
- Same symbol used for different concepts (b_k = bias vs b(t) = best arm)
- Same concept explained differently in different sections
- Theorem/result stated in one section, restated slightly differently elsewhere
- Algorithm description inconsistent with model setup

## Arguments

- `$ARGUMENTS` - Optional: path to paper directory

## Workflow

### Phase 1: Build Symbol Registry

Read all .tex files and extract every symbol definition:

```
For each file:
  Find patterns:
    - "let $X$" or "Let $X$"
    - "define $X$" or "Define $X$"
    - "$X \coloneqq$" or "$X :=$" or "$X \triangleq$"
    - "$X$ denotes" or "$X$ represents"
    - "where $X$ is"
```

Build a registry:
```
| Symbol | Meaning | Location | Context |
|--------|---------|----------|---------|
| b_k    | bias    | model.tex:L8 | "b_k(x) is an unknown bias" |
| b(t)   | best arm | algorithm.tex:L5 | "b(t) ∈ argmax..." |
| ...    | ...     | ...      | ... |
```

### Phase 2: Detect Symbol Conflicts

**Check 1: Same symbol, different meanings**

```python
# Pseudocode
for symbol in registry:
    meanings = unique(registry[symbol].meaning)
    if len(meanings) > 1:
        CONFLICT: symbol has multiple meanings
```

**Common conflicts to watch:**
| Symbol | Conflict Type |
|--------|--------------|
| b | bias vs best arm |
| k | arm index vs iteration |
| t | time vs threshold |
| n | sample size vs dimension |
| X | context vs random variable |
| F | judge score vs CDF vs function |

**Check 2: Similar symbols, easy confusion**
- $k$ vs $K$ (arm index vs number of arms)
- $t$ vs $T$ (time vs horizon)
- $\pi$ vs $\pi_t$ vs $\pi^*$ (policy variants)

### Phase 3: Check Concept Consistency

**Check 1: Same concept, different explanations**

Find repeated explanations of the same concept:
```
grep -n "Neyman" *.tex
grep -n "audit.*probability" *.tex
grep -n "π.*sqrt" *.tex
```

For each concept mentioned multiple times:
- Are the explanations consistent?
- Is one more detailed than another?
- Should duplicates be consolidated?

**Check 2: Theorem/Result Redundancy**

Find theorems/propositions about the same topic:
```
| Result | Location | Core Claim |
|--------|----------|------------|
| Theorem 5.2 | algorithm.tex | π* ∝ √g minimizes variance |
| Prop 7.1 | theory.tex | optimal audit is π* ∝ √g |
| Section 7.3 | theory.tex | Neyman allocation gives π* ∝ √g |
```

If multiple results say the same thing:
- Consolidate into one location
- Reference from other locations

### Phase 4: Check Model-Algorithm Consistency

**Check 1: Setup matches algorithm**

Compare model setup with algorithm description:

| Model Says | Algorithm Does | Consistent? |
|------------|----------------|-------------|
| "Observe context X_t, select arm k_t" | Line 8: "Draw X ~ D; observe F" | ⚠️ Order unclear |
| "Audit decision A_t" | Line 12: "Draw A ~ Bern(π)" | ✅ |

**Check 2: Notation in proofs matches definitions**

Verify proof notation matches earlier definitions:
- Does proof use same symbols as theorem statement?
- Are subscripts/superscripts consistent?

### Phase 5: Prose-Algorithm Alignment

**Check 1: Step ordering**

Compare the prose model description (typically in model/method section) with the algorithm pseudocode:

```
| Step | Prose Says | Algorithm Does | Consistent? |
|------|-----------|----------------|-------------|
| 1 | "First observe context" | Line 3: "Select arm" | ⚠️ Order mismatch |
| 2 | "Then select action" | Line 5: "Observe context" | ⚠️ Reversed |
```

The prose model should describe steps in the same order as the algorithm executes them. Flag any ordering discrepancy.

**Check 2: Variable definitions match**

For each variable in the algorithm pseudocode, verify it matches the prose definition:

```
| Algorithm Variable | Prose Definition | Consistent? |
|-------------------|------------------|-------------|
| "Pull arm k_t" | "Select arm k_t ∈ [K]" | ✅ |
| "Draw A_t ~ Bern(π_t)" | "Decide whether to audit" | ⚠️ Prose doesn't mention Bernoulli |
```

**Check 3: Per-iteration semantics**

If the algorithm has a "for each round t" loop, verify:
- Does each iteration pull one unit, multiple units, or a batch?
- Does the prose description match?
- Are index semantics clear (does t index rounds, samples, or something else)?

### Phase 6: Definition-Instantiation Alignment

**Check 1: Breadth consistency**

When a concept is defined broadly (e.g., "arms represent service configurations"), verify that:
- Experiments instantiate examples matching the full breadth of the definition
- The conclusion doesn't narrow the definition implicitly

```
| Concept | Definition Scope | Instantiation Scope | Consistent? |
|---------|-----------------|---------------------|-------------|
| "Arms" | "Service configurations" | Experiment uses only model variants | ⚠️ Narrower |
| "Feedback" | "Noisy proxy signal" | Experiments use LLM scores | ✅ |
```

**Check 2: Claim-evidence alignment**

For each claim in the introduction/abstract:
- Is there a theorem, proposition, or experiment that supports it?
- Does the evidence match the scope of the claim?

```
| Claim | Scope | Evidence | Scope Match? |
|-------|-------|----------|-------------|
| "Works for any service system" | General | 2 specific experiments | ⚠️ Evidence narrower |
| "Achieves optimal rate" | Formal | Theorem + simulation | ✅ |
```

### Phase 7: Formula Consistency (Main Text ↔ Appendix)

**Why this matters**: When a result appears in both the main text (informal or example) and the appendix (formal derivation), the mathematical forms can silently diverge. This was a real issue in Section 6 restructure: the main text used an additive form `c_F σ²/κ + c_Y σ²/κ` while the appendix derivation yielded a squared-sum form `(√c_F · σ/√κ + √c_Y · σ/√κ)²`.

**Detection procedure**:

1. **Identify paired results**: Find theorems/propositions/examples that appear in both main text (`sections/*.tex`) and appendix (`appendix/*.tex`):
   ```
   For each \label{thm:*}, \label{prop:*}, \label{ex:*} in sections/:
     Search appendix/ for matching references or proof environments
   ```

2. **Extract mathematical expressions**: For each pair, extract the key mathematical formula from:
   - Main text statement (theorem body, example formula)
   - Appendix derivation (final result of proof, derived expression)

3. **Compare functional forms**:
   | Check | What to Compare |
   |-------|----------------|
   | Additive vs multiplicative | `a + b` vs `(√a + √b)²` |
   | Variance vs std dev | `σ²` vs `σ` in same position |
   | With/without correction | `σ²` vs `σ²/κ` |
   | Subscript consistency | `σ_F` vs `σ_{F,k}` vs `σ` |
   | Summation structure | `Σ_k f(k)` vs single-arm form |

4. **Flag mismatches**:
   ```
   ⚠️ FORMULA MISMATCH
   Location: theory_lower_bound.tex:L52 vs proofs_cost_bounds.tex:L78
   Main text: c_F σ²_F/κ_F + c_Y σ²_R/κ_R  (additive)
   Appendix:  (√(c_F) σ_F/√κ_F + √(c_Y) σ_R/√κ_R)²  (squared-sum)
   Severity: 🔴 Critical — different functional forms
   ```

**Common divergence patterns**:
- Result promoted from appendix to main text but simplified differently
- Example uses special case while appendix has general form (mismatch in reduction)
- Notation changed in main text but not propagated to appendix proof
- Truncation/approximation applied inconsistently

### Phase 8: Check Cross-Reference Accuracy

**Check 1: Forward/backward references**
```
grep -n "\\ref{" *.tex
```

For each reference:
- Does the referenced item exist?
- Does the reference make sense in context?

**Check 2: Equation numbering**
- Are important equations numbered?
- Are numbered equations actually referenced?

## Output Format

```markdown
# Consistency Check Report

**Paper**: [title]
**Files Checked**: [count]
**Date**: [date]

---

## 1. Symbol Conflicts

### Critical Conflicts (Must Fix)

| Symbol | Meaning 1 | Location 1 | Meaning 2 | Location 2 |
|--------|-----------|------------|-----------|------------|
| b | bias b_k(x) | model.tex:L8 | best arm b(t) | algorithm.tex:L5 |

**Suggested Fix**:
- Rename b(t) to ĥat{k}(t) in algorithm.tex
- Or use different symbol for bias (e.g., β_k)

### Minor Conflicts (Consider Fixing)
[Similar table]

---

## 2. Concept Redundancy

### π ∝ √g (Neyman Allocation)

**Mentioned in**:
1. Section 5.2, Theorem 5.2: "optimal audit π* ∝ √g"
2. Section 7.2, paragraph 3: "Neyman-shaped auditing π* ∝ √g"
3. Section 7.3, Proposition 7.3: "square-root scaling π* ∝ √g"

**Issue**: Same result stated 3 times
**Suggestion**:
- State once in Section 5 as main result
- Reference from Sections 7.2, 7.3 instead of restating

### [Other redundant concepts]

---

## 3. Model-Algorithm Inconsistencies

| Aspect | Model (Section 3) | Algorithm (Section 5) | Issue |
|--------|-------------------|----------------------|-------|
| Context observation | "Observe X_t, select k_t" | "for k ∈ {b(t),c(t)}: Draw X" | Order/timing unclear |

**Suggested Fix**: [specific fix]

---

## 4. Reference Issues

### Missing References
- Equation (15) defined but never referenced
- Assumption 3.4 referenced in Section 7 but defined in Section 3

### Broken References
- \ref{thm:foo} in line 45 - label not found

---

## 5. Summary

| Category | Count | Severity |
|----------|-------|----------|
| Symbol conflicts | X | 🔴 Critical |
| Concept redundancy | Y | 🟡 Moderate |
| Model-algorithm mismatch | Z | 🟡 Moderate |
| Reference issues | W | 🟢 Minor |

### Priority Fixes
1. [Most critical]
2. [Second]
3. [Third]
```

## Quick Checks (grep patterns)

Run these to quickly find potential issues:

```bash
# Symbol definitions
grep -n "\\\\coloneqq\|:=\|\\\\triangleq" sections/*.tex

# Potential b conflicts
grep -n "b_k\|b(t)\|b_t\|\\\\hat{b}" sections/*.tex

# Neyman/sqrt g mentions
grep -n "sqrt.*g\|\\\\propto.*sqrt\|Neyman" sections/*.tex

# Assumption references
grep -n "\\\\ref{ass:" sections/*.tex
grep -n "\\\\label{ass:" sections/*.tex
```

## Self-Dispatch Phases

**This skill has 1 setup phase + 6 parallel check phases. Follow `.claude/commands/_shared/self_dispatch_protocol.md`.**

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 0 | Build symbol registry | No (setup) | All `sections/*.tex`, `appendix/*.tex` | Extract every symbol definition (`let`, `define`, `:=`, `denotes`) into registry |
| 1 | Symbol conflicts | Yes (after 0) | (uses registry) | Same symbol with different meanings; similar symbols causing confusion |
| 2 | Concept consistency | Yes (after 0) | All `sections/*.tex` | Same concept explained differently across sections; result redundancy |
| 3 | Model-algorithm consistency | Yes (after 0) | `model.tex`, `algorithm.tex`, `method.tex` | Setup matches algorithm; notation in proofs matches definitions |
| 4 | Cross-reference accuracy | Yes (after 0) | All `sections/*.tex` | Forward/backward references valid; equation numbering consistent |
| 5 | Prose-algorithm alignment | Yes (after 0) | `model.tex`, `method.tex`, `algorithm.tex` | Prose step ordering matches pseudocode; variable definitions match; per-iteration semantics clear |
| 6 | Definition-instantiation alignment | Yes (after 0) | `introduction.tex`, `model.tex`, `experiments.tex`, `conclusion.tex` | Concepts defined broadly are instantiated with matching breadth; claims have matching-scope evidence |
| 7 | Formula consistency (main↔appendix) | Yes (after 0) | All `sections/*.tex`, `appendix/*.tex` | Same result in main text and appendix uses same mathematical form; no additive-vs-multiplicative, variance-vs-stddev, or notation divergence |

**Sequential**: Phase 0 (setup) must complete first — produces the symbol registry.
**Parallel group**: Phases 1-7 can run in parallel (all consume Phase 0 output).
**Aggregation**: Merge 7 sub-reports, sort by severity (formula mismatch > symbol conflicts > prose-alg > definition-instantiation > concept > model-alg > refs).

---

## Begin

**Dispatch**: Setup → parallel — **Template B** from `self_dispatch_protocol.md`.
**Setup output**: Symbol registry (all symbol definitions with meanings, locations, contexts).

1. Follow unified protocol Steps 0A–2.5 (include RAG references)
2. Execute Phase 0 inline (build symbol registry)
3. Recursion guard → if subagent, execute remaining phases inline
4. Dispatch 7 parallel Task subagents (Phases 1-7), each receives symbol registry
5. Aggregate → deduplicate → sort by severity
6. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: L2 Consistency (Symbol/Notation)
   Issues Found: {N}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   /fix-issues symbols         → Auto-fix symbol conflicts
   /fix-issues symbols --dry-run → Preview changes first

   {If no issues:}
   ✅ No symbol conflicts found. Proceed to next check.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found:]
   /fix-issues symbols         → Auto-fix conflicts
   /check-paper-consistency    → Verify fixes applied

   [Same level (L2) - complete these:]
   /check-term-consistency     → Check "audit" vs "review" etc.
   /check-cross-references     → Verify all refs valid

   [When ALL L2 checks pass:]
   /check-ms-style             → Move to L3 (venue style)
   /check-or-style             → (alternative for OR journals)

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy
   L1 Structure   ─────────── /check-content-placement, /check-paper-flow
 → L2 Consistency ─────────── YOU ARE HERE
   L3 Style       ─────────── /check-ms-style
   L4 Language    ─────────── /polish-paper (only after L0-L3 pass)

💡 TIP: Use /paper-pipeline status to see overall progress
```
