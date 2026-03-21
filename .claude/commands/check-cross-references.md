# Check Cross-References - Verify All References Are Valid and Consistent

You are a cross-reference verification agent. Your task is to ensure all forward/backward references, numerical values, and result citations are consistent throughout the paper.

## ⚠️ MANDATORY: Unified Protocol

### Step 0: Read Shared Config

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
```

### Step 1: Read Paper State Files (CRITICAL for this command)

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** This command depends ENTIRELY on paper state docs for accuracy. You MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/cross_references.md  # Existing reference map
Read docs/paper_state/{resolved}/results.md           # Theorem registry
Read docs/paper_state/{resolved}/symbols.md           # Symbol definitions
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- cross_references.md: [number of tracked references, key cross-ref targets]
- results.md: [theorem/proposition/lemma count and labels]
- symbols.md: [key symbol definitions]
```

**If you skip this step, your cross-reference check will miss renamed theorems and changed numbering.**

**Note**: This command is primarily mechanical (checking refs), so RAG is optional. But paper_state is CRITICAL.

### Step 2: Apply RAG Miss Detection (if using RAG)

If you use RAG for suggestion generation and any search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

---

## Why This Matters

**Problem**: Long papers have 50+ cross-references. LLM context can't track all of them.

```
Session 1: Change Theorem 5.2 statement
Session 2: Polish Section 7 (doesn't remember Thm 5.2 changed)
Session 3: Polish Conclusion (references old Thm 5.2 statement)
→ Reviewer: "Theorem 5.2 statement differs in Section 7 and Conclusion"
```

## Arguments

- `$ARGUMENTS` - Optional: paper path or specific check type:
  - `[path]` - Check paper at path
  - `results` - Check only result references
  - `numbers` - Check only numerical values
  - `forward` - Check only forward references
  - `backward` - Check only backward references

## Checks Performed

### 1. Result Reference Consistency

For each theorem/lemma/proposition:

```bash
# Find primary definition
grep -n "\\\\begin{theorem}.*thm:neyman" sections/*.tex

# Find all references
grep -n "Theorem 5.2\\|Thm 5.2\\|thm:neyman\\|\\\\ref{thm:neyman}" sections/*.tex
```

**Verification**:
| Location | Type | Quote | Matches Primary? |
|----------|------|-------|------------------|
| algorithm.tex:L120 | Primary | "π* ∝ √g" | - |
| analysis.tex:L30 | Reference | "By Theorem 5.2..." | ✅ |
| conclusion.tex:L15 | Summary | "Neyman-optimal allocation" | ✅ |

### 2. Numerical Value Consistency

For each key number (cost savings, accuracy, etc.):

```bash
# Find all occurrences of a number
grep -n "48%\\|48.3%\\|48 percent\\|forty-eight" sections/*.tex
```

**Verification**:
| Value | Location 1 | Location 2 | Location 3 | Match? |
|-------|------------|------------|------------|--------|
| 48% | abstract:L8 | experiments:L180 | conclusion:L12 | ✅ |
| 98.8% | experiments:L150 | conclusion:L18 | - | ✅ |

### 3. Forward Reference Validation

Find all "we will show" / "as we prove" promises:

```bash
grep -n "we will\\|we shall\\|we prove\\|later\\|Section [0-9]" sections/*.tex | grep -i "show\\|prove\\|demonstrate\\|discuss"
```

**Verification**:
| Promise | Location | Target | Fulfilled? |
|---------|----------|--------|------------|
| "We prove in §3" | intro:L40 | model.tex | ✅ Thm 3.1 exists |
| "discussed in §7" | method:L60 | theory_lb.tex | ✅ Content exists |
| "shown in EC.3" | analysis:L80 | appendix | ❌ EC.3 missing! |

### 4. Backward Reference Validation

Find all "as shown" / "by Theorem" claims:

```bash
grep -n "as shown\\|By Theorem\\|By Lemma\\|in Section\\|\\\\ref{" sections/*.tex
```

**Verification**:
| Claim | Location | Target | Valid? |
|-------|----------|--------|--------|
| "By Theorem 5.2" | analysis:L30 | algorithm.tex:L120 | ✅ |
| "as in Section 3" | method:L15 | model.tex | ✅ |
| "By Lemma B.2" | analysis:L50 | appendix | ❌ B.2 doesn't exist! |

### 5. Label Existence Check

```bash
# Find all \ref{} calls
grep -oP '\\ref\{[^}]+\}' sections/*.tex | sort | uniq

# Find all \label{} definitions
grep -oP '\\label\{[^}]+\}' sections/*.tex appendix/*.tex | sort | uniq

# Find orphan refs (ref without label)
comm -23 refs.txt labels.txt
```

### 6. Citation Consistency

Same paper should be cited consistently:

```bash
# Find all citation keys
grep -oP '\\cite[tp]?\{[^}]+\}' sections/*.tex | sort | uniq -c
```

**Issues to detect**:
- Same paper cited with different keys: `\cite{smith2023}` vs `\cite{smith2023a}`
- Inconsistent citation style: "Smith et al. (2023)" vs "[Smith23]"

### 7. Text-Based Appendix References

Many papers reference appendix sections using prose text rather than `\ref{}`. These text-based references are invisible to LaTeX and break silently when appendix sections are renumbered or removed.

**Detection patterns:**
```bash
# Find text-based appendix references (common patterns)
grep -n "Appendix [A-Z]\|Appendix EC\.\|in EC\.\|see EC\.\|See EC\.\|in Appendix\|see Appendix\|See Appendix" sections/*.tex
grep -n "Online Appendix\|Electronic Companion\|Supplement [A-Z]" sections/*.tex
```

**For each text-based reference found:**
1. Extract the appendix section identifier (e.g., "EC.3", "Appendix B.2", "Appendix A")
2. Check that a corresponding `\label{}` or `\section{}` exists in the appendix files
3. Verify the content described in the reference actually exists at that location

**Verification table:**
| Text Reference | Location | Target | Exists? |
|---------------|----------|--------|---------|
| "See Appendix EC.3" | analysis.tex:L80 | appendix/ | ❌ No EC.3 section |
| "details in Appendix B" | method.tex:L45 | appendix/proofs.tex | ✅ |
| "as shown in EC.2.1" | theory.tex:L30 | appendix/examples.tex | ⚠️ Section renumbered |

**Common failure modes:**
- Appendix section deleted but text references remain
- Appendix sections renumbered after reorganization
- EC (Electronic Companion) section references that don't match actual EC structure
- "See Appendix X for details" where Appendix X discusses something different

### 8. Content Move Integrity

**Why this matters**: When restructuring sections (e.g., promoting a theorem from appendix to main text, moving a proposition to conclusion, flattening subsections), references to moved content often break silently. During the Section 6 restructure, `prop:decomposition` was removed from the main text but was still referenced in `experiments.tex`. This phase detects such orphaned references.

**Detection procedure**:

1. **Build content location map**: For each `\label{}` in the paper, record its current location:
   ```
   | Label | Current File | Current Section |
   |-------|-------------|----------------|
   | thm:cost_bound | theory_lower_bound.tex | Section 6 |
   | prop:decomposition | proofs_lower_bound.tex | Appendix |
   ```

2. **Cross-check with state docs**: Compare current label locations against `cross_references.md` and `results.md`:
   ```
   State doc says thm:cost_bound is in: appendix/proofs_cost_bounds.tex
   Actual location: sections/theory_lower_bound.tex
   → MOVED (state doc stale)
   ```

3. **Detect orphaned references**: For each `\ref{}` or `\Cref{}`:
   - If the referenced label exists but was recently moved (state doc mismatch), check if the reference context still makes sense
   - If the referenced label was part of a `\Cref{A,B,C}` group, check if all items in the group still exist in the same scope

4. **Detect dangling prose references**: Search for text-based references to moved content:
   ```bash
   # Find prose references to specific results
   grep -n "Section 6.2\|Section 6.3\|Section 6.4" sections/*.tex
   grep -n "Proposition.*decomposition\|Proposition.*comparative" sections/*.tex
   ```

5. **Flag issues**:
   ```
   ⚠️ ORPHANED REFERENCE
   Location: experiments.tex:L63
   Reference: \Cref{prop:decomposition,thm:neyman}
   Issue: prop:decomposition moved from Section 6 to Appendix — still appropriate in this Cref group?

   ⚠️ STALE LOCATION IN STATE DOC
   Label: thm:cost_bound
   State doc: appendix/proofs_cost_bounds.tex
   Actual: sections/theory_lower_bound.tex
   Action: Update cross_references.md and results.md
   ```

**Common failure modes after restructuring**:
- Result moved to appendix but main text still references it as "in Section X"
- Subsection removed but cross-references to "Section X.Y" remain in prose
- Theorem promoted to main text but appendix proof still says "we now prove Theorem X" with old numbering
- Multi-result `\Cref{}` groups that mix main text and appendix labels after a move

---

## Output Format

```markdown
# Cross-Reference Verification Report

**Paper**: [name]
**Date**: [date]
**Total References Checked**: [count]

---

## Summary

| Check Type | Total | Valid | Invalid |
|------------|-------|-------|---------|
| Result references | 25 | 24 | 1 |
| Numerical values | 12 | 12 | 0 |
| Forward references | 8 | 7 | 1 |
| Backward references | 15 | 14 | 1 |
| Labels | 45 | 45 | 0 |
| Citations | 30 | 30 | 0 |

**Overall Status**: ⚠️ 3 issues found

---

## Issues Found

### Issue 1: Broken Forward Reference
**Location**: analysis.tex:L80
**Promise**: "shown in EC.3"
**Target**: EC.3 does not exist
**Action**: Create EC.3 or change reference to EC.2

### Issue 2: Outdated Result Reference
**Location**: conclusion.tex:L15
**Reference**: "Theorem 5.2 shows π* ∝ √g"
**Current Thm 5.2**: "Optimal allocation satisfies π* = √(c_F σ²_F / c_Y σ²_R)"
**Action**: Update conclusion to match current statement

### Issue 3: Missing Label
**Location**: algorithm.tex:L95
**Ref**: \ref{eq:neyman}
**Status**: No \label{eq:neyman} found
**Action**: Add label or fix ref

---

## Numerical Value Tracker

| Value | Master Location | All Occurrences | Status |
|-------|-----------------|-----------------|--------|
| 48.3% | experiments:Table2 | abstract, intro, exp, conc | ✅ All match |
| 98.8% | experiments:Table1 | exp, conclusion | ✅ All match |
| -1.78 | experiments:L220 | theory_lb:L90 | ⚠️ theory says "≈-2" |

---

## Result Reference Map

### Theorem 5.2 (thm:neyman)
**Primary**: algorithm.tex:L120
**Statement**: "π* ∝ √(g_k/c_Y)"

| Ref Location | Type | Status |
|--------------|------|--------|
| intro:L45 | Preview | ✅ Consistent |
| analysis:L30 | Reference | ✅ Consistent |
| theory_lb:L85 | Reference | ✅ Consistent |
| conclusion:L15 | Summary | ⚠️ Outdated |

---

## Recommendations

1. **High Priority**: Fix broken EC.3 reference
2. **High Priority**: Update conclusion Thm 5.2 description
3. **Medium Priority**: Harmonize theory vs experiment on slope value

---

## Update cross_references.md

After fixing issues, update `docs/paper_state/[paper]/cross_references.md` with current state.
```

## Integration

```
/check-cross-references           → Find all issues
        ↓
[Fix issues in paper]
        ↓
/update-paper-state               → Sync docs
        ↓
/check-cross-references           → Verify fixes
```

## Self-Dispatch Phases

**This skill has 7 independent phases. Follow `.claude/commands/_shared/self_dispatch_protocol.md`.**

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 1 | Result references | Yes | All `sections/*.tex` | Theorem/lemma/proposition references match primary definitions |
| 2 | Numerical values | Yes | abstract, intro, experiments, conclusion .tex | Key numbers (percentages, counts) are consistent across all mentions |
| 3 | Forward references | Yes | All `sections/*.tex` | Every "we will show" / "Section X" promise is fulfilled |
| 4 | Backward references | Yes | All `sections/*.tex` | Every "as shown" / "By Theorem" claim points to valid target |
| 5 | Label existence | Yes | All `sections/*.tex`, `appendix/*.tex` | Every `\ref{}` has a matching `\label{}` |
| 6 | Citation consistency | Yes | All `sections/*.tex`, `.bib` files | Same paper cited with same key; citation style uniform |
| 7 | Text-based appendix refs | Yes | All `sections/*.tex`, `appendix/*.tex` | Prose references to appendix ("See EC.3", "Appendix B.2") point to existing sections |
| 8 | Content move integrity | Yes | All `sections/*.tex`, `appendix/*.tex`, `cross_references.md`, `results.md` | Labels match state doc locations; no orphaned refs after content redistribution; multi-label Cref groups still coherent |

**Parallel group**: All 8 phases can run in parallel (no data dependency).
**Aggregation**: Merge 8 sub-reports into single cross-reference report, dedup any overlapping findings.

---

## Begin

**Dispatch**: All phases parallel — **Template A** from `self_dispatch_protocol.md`.

1. Follow unified protocol Steps 0A–2.5
2. Recursion guard → if subagent, execute inline
3. Dispatch 8 parallel Task subagents (Phases 1-8)
4. Aggregate → deduplicate → sort by severity
5. Output report + fix instructions
6. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: L2 Consistency (Cross-References)
   Issues Found: {N}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   /fix-issues refs            → Auto-fix broken references
   /fix-issues numbers         → Harmonize numerical values
   /fix-issues refs --dry-run  → Preview changes first

   {If no issues:}
   ✅ All cross-references valid. Proceed to next check.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found:]
   /fix-issues refs            → Auto-fix references
   /fix-issues numbers         → Fix number inconsistencies
   /check-cross-references     → Verify fixes applied
   /update-paper-state [name]  → Sync cross_references.md

   [Same level (L2) - complete these:]
   /check-paper-consistency    → Check symbol conflicts
   /check-term-consistency     → Check terminology

   [When ALL L2 checks pass:]
   /check-ms-style             → Move to L3 (venue style)

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy
   L1 Structure   ─────────── /check-content-placement, /check-paper-flow
 → L2 Consistency ─────────── YOU ARE HERE
   L3 Style       ─────────── /check-ms-style
   L4 Language    ─────────── /polish-paper (only after L0-L3 pass)

💡 TIP: Use /paper-pipeline status to see overall progress
```
