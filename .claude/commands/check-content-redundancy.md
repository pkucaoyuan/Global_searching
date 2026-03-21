# Check Content Redundancy - Detect Repeated Results and Definitions

You are a redundancy detector for academic papers. Your task is to find where the same result, definition, or explanation appears multiple times across sections.

## ⚠️ MANDATORY: RAG-Grounded Consolidation Suggestions

When suggesting how to consolidate redundant content, **ground all suggestions in human-authored patterns.**

### Step 0: Read Shared Config & RAG References

Before starting any review, execute these Read tool calls:

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/paragraphs/main_results.md
Read .claude/writing_references/paragraphs/paper_roadmap.md
Read .claude/writing_references/sentences/contribution.md
Read .claude/writing_references/phrases/transitions.md
```

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before checking for redundancy, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/results.md        # Track all theorems/lemmas
Read docs/paper_state/{resolved}/overview.md       # Paper structure
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- results.md: [total theorems/propositions, which are PRIMARY vs cross-referenced]
- overview.md: [section structure, page counts]
```

**If you skip this step, you may flag intentional cross-references as redundancy.**

Use these patterns when:
- Suggesting how to present a result in ONE primary location
- Recommending how to cross-reference from other sections
- Proposing how to write brief previews/summaries without redundancy

### Step 2: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

This ensures the RAG library self-maintains by logging gaps and suggesting additions.

---

## Why This Matters

Reviewer complaint: "Sections 5, 6, 7 都在说同一件事"

This happens when:
1. Same theorem/result stated in multiple sections
2. Same formula derived multiple times with slight variations
3. Same intuition explained repeatedly
4. Preview in one section, full version in another, then summary in third

## Arguments

- `$ARGUMENTS` - Optional: path to paper directory

## Workflow

### Phase 1: Extract All Results (with Formula Fingerprinting)

Build a registry of all formal results, **including the display math formulas** inside each environment:

```
| Type | Label | Core Claim | Display Math | Location |
|------|-------|------------|-------------|----------|
| Theorem | thm:main | optimal rate | C^* = \sum_k \sigma_k^2 / \Delta_k^2 | algorithm.tex:L28 |
| Proposition | prop:decomp | inner/outer separation | T = T_{inner} + T_{outer} | theory.tex:L46 |
| Example | ex:special | special case bound | C \geq n \log(1/\delta) | analysis.tex:L22 |
| Remark | rem:connection | matches classical result | \pi^* \propto \sqrt{g} | theory.tex:L58 |
```

**Formula extraction**: For each `\begin{theorem/proposition/lemma/corollary/example/remark}` environment, extract all `\[...\]`, `$$...$$`, `equation`, `align`, and `gather` display math blocks inside it. Normalize by removing whitespace and standardizing macro names (e.g., `\frac` vs `\tfrac`, `\sum` with/without limits).

### Phase 2: Find Semantic Duplicates

**Step 1: Group by topic**

Cluster results that discuss the same topic:

```
Topic: "Optimal audit probability"
- Theorem 5.2: π* ∝ √g minimizes variance
- Section 7.2, paragraph: "Neyman-shaped π* ∝ √g emerges from lower bound"
- Proposition 7.3(c): "square-root scaling π* ∝ √g"
- Remark 7.1: "practical algorithm uses π ∝ √ĝ"
```

**Step 2: Formula fingerprint matching**

Compare normalized display math across all formal environments. Flag when:
- **Identical formula** appears in two different numbered environments (theorem, proposition, lemma, etc.)
- **Near-identical formula** (same structure, different variable names) appears in different environments

```
| Formula | Environment 1 | Environment 2 | Match Type |
|---------|---------------|---------------|------------|
| C = \sum \sigma_k^2/\Delta_k^2 | Theorem 5.2 | Proposition 7.3 | IDENTICAL → redundant |
| \pi^* \propto \sqrt{g} | Theorem 5.2 | Remark 7.1 | IDENTICAL → check if intentional |
| C \geq n f(\delta) | Theorem 7.1 | Example 6.1 | STRUCTURAL → related but distinct |
```

**Step 3: Assess redundancy level**

| Level | Description | Action |
|-------|-------------|--------|
| **Full duplicate** | Same result, same proof idea, same/identical formula | Remove one |
| **Formula duplicate** | Same display math in different theorem environments | Consolidate: keep one as primary, reference from other |
| **Partial overlap** | Same result, different angle | Consolidate or cross-reference |
| **Intentional recap** | Summary of earlier result | OK if brief, labeled as such |
| **No redundancy** | Related but distinct | Keep both |

### Phase 3: Check Result-Example Pairing

**Issue**: Example and its generalization in different sections

**Bad structure:**
```
Section 6: Example 6.1 (Gaussian lower bound)
Section 7: Theorem 7.1 (General lower bound)
```
→ Reader sees special case before general case, then can't connect them

**Good structure:**
```
Section 7.1: Theorem 7.1 (General lower bound)
Section 7.1.1: Example (Gaussian illustration)
```
→ General result first, then concrete illustration

**Check**: For each Example, is the related Theorem/Proposition nearby?

### Phase 4: Check Preview-Detail-Summary Pattern

Papers often have:
1. **Preview** in introduction: "We show that π* ∝ √g"
2. **Full result** in technical section: Theorem with proof
3. **Summary** in conclusion: "Our key finding is π* ∝ √g"

This is OK. But problems arise when:
- Preview is too detailed (spoils the technical section)
- Multiple "full results" in different sections
- Technical detail repeated in conclusion

**Check each major result:**
- [ ] Mentioned in intro? (Should be brief preview)
- [ ] Full statement in ONE technical section?
- [ ] Mentioned in conclusion? (Should be brief summary)
- [ ] Mentioned elsewhere? (⚠️ Potential redundancy)

### Phase 5: Detect Explanation Redundancy

**Pattern**: Same intuition explained multiple times

Search for repeated explanations:
```bash
# Find all explanations of a concept
grep -n "intuition\|intuitively\|the idea is\|in other words" sections/*.tex

# Find repeated "this means" explanations
grep -n "this means\|this implies\|in practice" sections/*.tex
```

**Example of redundant explanations:**
```
Section 5: "Intuitively, we audit more where the proxy is unreliable..."
Section 7: "The intuition is to concentrate audits where judge quality is poor..."
Section 8: "In other words, audit probability should be higher in uncertain regions..."
```
→ Same intuition, three times. Keep one, reference from others.

### Phase 6: Check Section Necessity

For each section, ask:
1. What unique content does this section provide?
2. Could this section be merged with another?
3. Could this section move to appendix?

**Red flags:**
- Section that only restates earlier results with different notation
- Section that only provides "another perspective" on same result
- Very short section (< 1 page) that could be a subsection

## Output Format

```markdown
# Content Redundancy Report

**Paper**: [title]
**Date**: [date]

---

## 1. Result Redundancy

### High Redundancy (Same result, multiple locations)

**Result**: π* ∝ √g (Neyman allocation)

| Location | Form | Redundancy Level |
|----------|------|------------------|
| Theorem 5.2 | Full theorem | PRIMARY |
| Section 7.2 para 3 | Prose restatement | REDUNDANT |
| Prop 7.3(c) | Numbered result | REDUNDANT |
| Remark 7.1 | Connection to algorithm | OK (different purpose) |

**Recommendation**:
- Keep Theorem 5.2 as the primary statement
- In Section 7.2, write: "By Theorem 5.2, the optimal audit policy satisfies π* ∝ √g"
- Remove Prop 7.3(c) or merge into Theorem 5.2

### Medium Redundancy
[Similar analysis]

---

## 2. Example-Result Misplacement

| Example | Related Result | Issue | Fix |
|---------|----------------|-------|-----|
| Example 6.1 (Gaussian) | Theorem 7.1 (General LB) | Different sections | Move Example to Section 7.1.1 |

---

## 3. Preview-Detail-Summary Analysis

| Result | Intro | Technical | Conclusion | Issue |
|--------|-------|-----------|------------|-------|
| π* ∝ √g | ✓ Brief | ✓ Thm 5.2 | ✓ Brief | ✅ OK |
| Cost bound | ✓ Brief | ✓ Thm 6.1 + Ex 6.1 | ✓ | ⚠️ Ex 6.1 should be with Thm 7.1 |
| Delayed CS | ✓ | ✓ Thm 8.1 | ✓ | ✅ OK |

---

## 4. Explanation Redundancy

### "Audit where proxy is unreliable"
- Section 5, L34: "audit more where the proxy is least reliable"
- Section 7.2, L12: "concentrate audits where judge quality is poor"
- Section 10, L8: "focus reviews on uncertain regions"

**Recommendation**: Keep first occurrence, remove/shorten others

---

## 5. Section Consolidation Suggestions

| Section | Current Role | Suggestion |
|---------|--------------|------------|
| Section 6 | Cost bounds + Gaussian example | Move Gaussian example to Section 7; keep only Theorem 6.1 |
| Section 7.3 | Comparative statics | Could merge with Section 7.2 |

---

## 6. Summary

| Redundancy Type | Count | Severity |
|-----------------|-------|----------|
| Result redundancy | X | 🔴 High |
| Example misplacement | Y | 🟡 Medium |
| Explanation redundancy | Z | 🟡 Medium |
| Section consolidation needed | W | 🟡 Medium |

### Priority Actions
1. Consolidate π* ∝ √g to one location
2. Move Example 6.1 next to Theorem 7.1
3. Remove repeated intuition explanations
```

## Quick Detection Commands

```bash
# Find all theorems/propositions
grep -n "\\\\begin{theorem}\|\\\\begin{proposition}\|\\\\begin{lemma}" sections/*.tex

# Find all examples
grep -n "\\\\begin{example}" sections/*.tex

# Find sqrt g mentions (potential Neyman redundancy)
grep -n "sqrt.*g\|\\\\propto.*g\|proportional.*g" sections/*.tex

# Find "intuition" explanations
grep -n "intuitiv\|the idea\|in other words\|this means" sections/*.tex

# Find cost/lower bound mentions
grep -n "lower bound\|cost.*bound\|1/Delta" sections/*.tex
```

## Self-Dispatch Phases

**This skill has 1 setup phase + 5 parallel analysis phases. Follow `.claude/commands/_shared/self_dispatch_protocol.md`.**

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 0 | Extract all results | No (setup) | All `sections/*.tex` | Build registry of all theorems, propositions, examples, remarks with labels and locations |
| 1 | Semantic duplicates | Yes (after 0) | All `sections/*.tex` | Group results by topic; find same result stated multiple times |
| 2 | Result-example pairing | Yes (after 0) | All `sections/*.tex` | Each example is near its related theorem/proposition |
| 3 | Preview-detail-summary | Yes (after 0) | intro, technical sections, conclusion | Each major result: brief intro preview, ONE full statement, brief conclusion summary |
| 4 | Explanation redundancy | Yes (after 0) | All `sections/*.tex` | Same intuition explained multiple times with different wording |
| 5 | Section necessity | Yes (after 0) | All `sections/*.tex` | Each section provides unique content; no section is just a restatement |

**Sequential**: Phase 0 (setup) must complete first — produces the result registry.
**Parallel group**: Phases 1-5 can run in parallel (all consume Phase 0 output).
**Aggregation**: Merge 5 sub-reports into single redundancy report, sort by severity.

---

## Begin

**Dispatch**: Setup → parallel — **Template B** from `self_dispatch_protocol.md`.
**Setup output**: Result registry (all theorems, propositions, examples with labels and locations).

1. Follow unified protocol Steps 0A–2.5 (include RAG references)
2. Execute Phase 0 inline (build result registry)
3. Recursion guard → if subagent, execute remaining phases inline
4. Dispatch 5 parallel Task subagents (Phases 1-5), each receives result registry
5. Aggregate → deduplicate → sort by severity
6. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: L0 Content (Redundancy)
   Issues Found: {N}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   1. Consolidate [result] to ONE location: [section]
   2. Replace duplicate at [location] with reference
   3. Re-run this check to verify

   {If no issues:}
   ✅ No content redundancy. Proceed to next check.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found - fix first, then:]
   /check-content-redundancy   → Re-verify after fixes

   [Same level (L0) - complete these:]
   /check-content-placement    → Check example/proof placement

   [When L0 passes:]
   /check-paper-flow           → Move to L1 (structure)

📋 REVIEW LEVELS REMINDER:
 → L0 Content     ─────────── YOU ARE HERE
   L1 Structure   ─────────── /check-paper-flow
   L2 Consistency ─────────── /check-paper-consistency
   L3 Style       ─────────── /check-ms-style
   L4 Language    ─────────── /polish-paper

💡 TIP: Use /paper-pipeline status to see overall progress
```
