# Regression Guard Protocol

**Any skill that EDITS paper .tex files MUST use this protocol to prevent reintroducing previously fixed issues.**

---

## Problem

Without regression guards, a common failure loop occurs:

```
Session 1: /check-paper-consistency → finds symbol conflict → /fix-issues symbols → fixed
Session 2: /polish-paper → rewrites sentence → accidentally reintroduces old term
Session 3: /refine-theory → changes notation → breaks cross-reference fixed in Session 1
```

The root cause: **editing commands don't know what was previously fixed.**

---

## Protocol: Three Phases

### Phase 1: Pre-Edit — Load Constraint Set

Before making ANY edit to .tex files, read the consistency log:

```
Read docs/paper_state/{resolved}/consistency_log.md
```

Extract all issues with status **Resolved** into a constraint set:

```
## Active Constraints (from resolved issues)

| # | Constraint | Type | Don't Do This |
|---|-----------|------|---------------|
| 1 | Symbol: use β_k not b_k for bias | symbol | Don't write b_k when meaning bias |
| 2 | Term: use "audit" not "review" | term | Don't write "review" for audit |
| 3 | Ref: EC.2 not EC.3 | reference | Don't reference EC.3 (doesn't exist) |
| 4 | Placement: A.5 in §7 not §3 | placement | Don't move A.5 back to §3 |
```

**Output a brief checkpoint:**
```
Regression guard loaded: {N} active constraints from consistency_log.md
```

### Phase 2: During Edit — Constraint-Aware Editing

When rewriting any sentence, paragraph, or section:

1. **Check each edit against the constraint set** before applying
2. If an edit would violate a constraint:
   - Do NOT apply the edit as-is
   - Rewrite to satisfy both the editing goal AND the constraint
3. **Locked terms from framing.md take absolute priority** — never change locked terminology even if a writing pattern suggests different wording

**Constraint priority** (highest first):
1. `framing.md` locked terms — never override
2. `consistency_log.md` resolved issues — don't reintroduce
3. `symbols.md` registered symbols — don't conflict
4. Current edit goal — achieve within above constraints

### Phase 3: Post-Edit — Verify No Regressions

After completing ALL edits, run a targeted regression check:

**Step 1: Spot-check resolved issues**

For each resolved issue in the constraint set, verify the fix is still in place:

```
For each constraint:
  1. Grep the edited files for the forbidden pattern
  2. If found → REGRESSION DETECTED
  3. If not found → constraint still satisfied
```

**Step 2: Run targeted re-checks based on edit type**

| What Was Edited | Re-Check |
|-----------------|----------|
| Notation/symbols | Grep for old symbol in edited files |
| Terminology | Grep for non-canonical term in edited files |
| Sentence rewrites | Verify no AI words reintroduced, no locked terms changed |
| Proof text | Verify cross-references still valid |
| Content moved | Verify forward/backward references still valid |

**Step 3: Handle regressions**

If any regression is detected:
1. **Fix immediately** — don't report and move on
2. **Log the regression** in the output:
   ```
   ⚠️ REGRESSION DETECTED AND FIXED:
   - Constraint #2: "audit" not "review"
   - File: experiments.tex:L67
   - Edit had written "review process" → corrected to "audit process"
   ```
3. **Re-verify** after fixing

**Step 4: Update consistency_log.md**

If any new issues were found during post-edit verification, add them to consistency_log.md.

---

## Iterative Check-Fix Loop

**For commands that modify multiple files or run multiple rounds**, use this convergence loop:

```
ROUND 1:
  1. Pre-edit: load constraints
  2. Apply edits (respecting constraints)
  3. Post-edit: verify no regressions
  4. Count: new_issues + regressions = R

IF R > 0:
  ROUND 2:
    1. Fix regressions and new issues
    2. Post-edit: verify again
    3. Count: R2

  IF R2 > 0:
    ROUND 3: (same pattern)
    ...

  IF R == 0:
    CONVERGED — no regressions, no new issues from edits

MAX ROUNDS: 3 (if not converged after 3 rounds, report remaining issues)
```

**Convergence means**: the edit was applied AND no previously-fixed issues were broken.

---

## Integration Points

### For `/fix-issues`

Add after Step 3 (Apply Fixes):
```
### Step 3.5: Regression Guard (Post-Fix Verification)
Follow Phase 3 of `.claude/commands/_shared/regression_guard.md`:
- Verify that fixing issue A didn't reintroduce issue B
- Re-check all resolved constraints from consistency_log.md
- If regressions found → fix immediately before proceeding to Step 4
```

### For `/polish-paper`

Add to each round:
```
### Regression Guard (Per-Round)
At the START of each polish round:
- Phase 1: Load constraint set from consistency_log.md
- During polishing: check each rewrite against constraints
At the END of each polish round:
- Phase 3: Verify no resolved issues were reintroduced
- Add REGRESSION count to modification tracking
```

### For `/refine-theory`

Add to Step 5 (iteration loop):
```
### Regression Guard (Per-Iteration)
At the START of each iteration:
- Phase 1: Load constraints (especially symbols.md conflicts, dependency chains)
During refinement:
- Phase 2: Don't change locked notation, don't break dependency chains
At the END of each iteration:
- Phase 3: Verify cross-references still valid, symbols still consistent
```

---

## Quick Reference: Grep Patterns for Common Regressions

```bash
# Symbol regressions (example: b_k conflict was fixed to β_k)
grep -n "FORBIDDEN_SYMBOL" sections/*.tex

# Term regressions (example: "review" was standardized to "audit")
grep -n "FORBIDDEN_TERM" sections/*.tex

# Broken appendix references
grep -n "EC\.\|Appendix" sections/*.tex

# AI words reintroduced
grep -n "significantly\|novel\|comprehensive\|leverages\|utilizes" sections/*.tex

# Locked framing terms violated
grep -n "REPLACED_TERM" sections/*.tex
```

---

## Output Format

Every editing command that uses this protocol should include a regression summary in its output:

```
═══════════════════════════════════════════════════════════════════
                    REGRESSION GUARD SUMMARY
═══════════════════════════════════════════════════════════════════

Constraints loaded: {N} (from consistency_log.md)
Constraints checked: {N}
Regressions detected: {R}
Regressions fixed: {R}

{If R > 0:}
  Fixed regressions:
  - [constraint description]: [file:line] → [fix applied]

{If R == 0:}
  ✅ No regressions. All previously fixed issues remain fixed.
```
