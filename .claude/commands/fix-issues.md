# Fix Issues - Automated Paper Issue Resolution

You are a paper fix agent. Your task is to automatically apply fixes for issues found by check commands.

## ⚠️ MANDATORY: Unified Protocol

**Before ANY fix, you MUST follow the unified protocol:**

### Step 0: Read Shared Config
```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
```

### Step 1: Read RAG Files (for rewrite suggestions)
```
# For symbols:
Read .claude/writing_references/sentences/problem_setup.md

# For terms:
Read .claude/writing_references/phrases/transitions.md
Read .claude/writing_references/sentences/motivation.md

# For placement:
Read .claude/writing_references/paragraphs/proof_structure.md
```

### Step 2: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before applying ANY fix, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL relevant state files using the resolved path:

```
# ALWAYS read these first to avoid creating new conflicts:
Read docs/paper_state/{resolved}/symbols.md      # For symbols fix
Read docs/paper_state/{resolved}/framing.md      # For terms fix
Read docs/paper_state/{resolved}/cross_references.md  # For refs/numbers fix
Read docs/paper_state/{resolved}/dependencies.md # For placement fix
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- symbols.md: [key symbols that must not conflict]
- framing.md: [locked terms that must not be changed]
- cross_references.md: [existing reference targets]
- dependencies.md: [assumption→theorem chains]
```

**If you skip this step, you WILL introduce new conflicts while fixing old ones.**

### Step 2.5: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

This ensures the RAG library self-maintains by logging gaps and suggesting additions.

### Step 3: Apply Fixes (with RAG-grounded rewrites)

### Step 3.5: Regression Guard (Post-Fix Verification)

**⚠️ MANDATORY — Follow `.claude/commands/_shared/regression_guard.md` Phase 3.**

After applying fixes, verify that fixing issue A didn't reintroduce issue B:

1. **Read constraint set**: `Read docs/paper_state/{resolved}/consistency_log.md`
2. **Spot-check**: For each resolved issue, grep edited files for the forbidden pattern
3. **If regression found**: Fix immediately before proceeding to Step 4
4. **Iterate**: If fixing the regression caused another issue, repeat (max 3 rounds)
5. **Report**: Include regression guard summary in output

```
Example:
  Fixed: symbol conflict b_k → β_k
  Regression check: grep "b_k" in edited files → found 0 old occurrences ✅
  But: renaming also changed "b_k" in a citation key → fix citation key
  Re-check: 0 regressions ✅ → proceed to Step 4
```

### Step 4: Update State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** After applying fixes to .tex files, you MUST update the state docs to keep them in sync. Use the resolved paper name from Step 2:

```
# After fixing, sync the state docs:
Edit docs/paper_state/{resolved}/symbols.md    # Record new/changed symbols
Edit docs/paper_state/{resolved}/framing.md    # Record new/changed terms
Edit docs/paper_state/{resolved}/changelog.md  # Log ALL changes (date + description)
```

**Update rules:**
- **symbols fix** → update `symbols.md` + `changelog.md`
- **terms fix** → update `framing.md` + `changelog.md`
- **refs fix** → update `cross_references.md` + `changelog.md`
- **placement fix** → update `dependencies.md` + `changelog.md`
- **figures fix** → update `figures_tables.md` + `changelog.md`

**If you skip this step, the state docs become stale and future commands will make conflicting decisions.**

---

## Why This Matters

Current workflow (tedious):
```
/check-paper-consistency → "Found 3 symbol conflicts"
[User manually edits each file]
/check-paper-consistency → Verify
```

New workflow (automated):
```
/check-paper-consistency → "Found 3 symbol conflicts"
/fix-issues symbols      → Automatically fix all 3
/check-paper-consistency → Verify (should pass)
```

## Arguments

- `$ARGUMENTS` - Fix type and options:
  - `symbols` - Fix symbol conflicts (rename/standardize)
  - `terms` - Fix terminology inconsistencies (search & replace)
  - `refs` - Fix broken references
  - `numbers` - Harmonize numerical values
  - `placement` - Move content to correct locations
  - `all` - Apply all safe fixes
  - `--dry-run` - Show what would be fixed without applying

## Fix Types

### 1. `symbols` - Fix Symbol Conflicts

**Input**: Symbol conflicts from `/check-paper-consistency`

**Process**:
1. Read `docs/paper_state/[paper]/consistency_log.md` for conflicts
2. For each conflict, determine canonical symbol
3. Generate rename operations
4. Apply with Edit tool

**Example**:
```
Conflict: b_k (bias) vs b(t) (best arm)

Decision: Keep b_k for bias, rename b(t) to k̂(t)

Actions:
1. In algorithm.tex: Replace "b(t)" → "\\hat{k}(t)" (15 occurrences)
2. In analysis.tex: Replace "b(t)" → "\\hat{k}(t)" (8 occurrences)
3. Update symbols.md with new canonical symbol
```

**Output**:
```
═══════════════════════════════════════════════════════════════════
                      SYMBOL FIXES APPLIED
═══════════════════════════════════════════════════════════════════

✅ Fixed 1 symbol conflict:

   b(t) → k̂(t) (best arm at time t)
   Files modified: algorithm.tex, analysis.tex
   Occurrences: 23

📋 NEXT STEPS:
   /check-paper-consistency    → Verify no more conflicts
   /fix-issues terms           → Fix terminology (if needed)
```

---

### 2. `terms` - Fix Terminology Inconsistencies

**Input**: Term inconsistencies from `/check-term-consistency`

**Process**:
1. Read framing.md for canonical terms
2. For each inconsistent term, generate replacements
3. Apply search & replace

**Example**:
```
Inconsistency: "audit" (45x) vs "review" (8x)
Canonical: "audit" (per framing.md)

Actions:
1. In intro.tex L23: "human review" → "human audit"
2. In experiments.tex L67: "review process" → "audit process"
3. In experiments.tex L89: "reviewed" → "audited"
```

**Smart Replacement Rules**:
- "review" → "audit" (noun)
- "reviewed" → "audited" (past tense)
- "reviewing" → "auditing" (gerund)
- "reviewer" → "auditor" (agent) [only if appropriate]
- Preserve capitalization
- Don't replace in citations or proper nouns

**Output**:
```
═══════════════════════════════════════════════════════════════════
                    TERMINOLOGY FIXES APPLIED
═══════════════════════════════════════════════════════════════════

✅ Fixed 1 terminology inconsistency:

   "review" → "audit" (8 occurrences)
   Files modified: intro.tex, experiments.tex

   Specific changes:
   - intro.tex:23: "human review" → "human audit"
   - experiments.tex:67: "review process" → "audit process"
   - experiments.tex:89: "reviewed" → "audited"

📋 NEXT STEPS:
   /check-term-consistency     → Verify all terms consistent
   /update-paper-state [name]  → Sync framing.md
```

---

### 3. `refs` - Fix Broken References

**Input**: Broken refs from `/check-cross-references`

**Process**:
1. For missing labels: Add \label{} at definition
2. For broken refs: Fix \ref{} to correct label
3. For orphan labels: Remove or add reference

**Example**:
```
Issue 1: \ref{thm:foo} but no \label{thm:foo}
Action: Find theorem about "foo", add \label{thm:foo}

Issue 2: \label{eq:old} but never referenced
Action: Remove label (or ask user)

Issue 3: "as shown in EC.3" but EC.3 doesn't exist
Action: Flag for user (can't auto-fix content gaps)
```

**Output**:
```
═══════════════════════════════════════════════════════════════════
                    REFERENCE FIXES APPLIED
═══════════════════════════════════════════════════════════════════

✅ Fixed 2 reference issues:

   1. Added \label{thm:neyman} to algorithm.tex:L120
   2. Fixed \ref{eq:ipw} → \ref{eq:ipw_estimator} in analysis.tex:L45

⚠️ Cannot auto-fix (requires content):
   - "as shown in EC.3" but EC.3 doesn't exist
     → Create EC.3 or change reference

📋 NEXT STEPS:
   /check-cross-references     → Verify refs valid
   [Manual fix for EC.3 issue]
```

---

### 4. `numbers` - Harmonize Numerical Values

**Input**: Number inconsistencies from `/check-cross-references`

**Process**:
1. Find master value (usually in experiments/tables)
2. Update all other occurrences to match

**Example**:
```
Value: Cost savings percentage
Master: 48.3% (experiments.tex, Table 2)
Inconsistent: "48%" (abstract), "about 50%" (intro)

Actions:
1. abstract.tex: "48%" → "48.3%" [exact match]
2. intro.tex: "about 50%" → "nearly 50%" [keep approximate, adjust]
   OR: "about 50%" → "48.3%" [exact match]
```

**User Choice**:
```
Found inconsistent number: 48% vs 48.3%

Options:
[1] Use exact value everywhere: 48.3%
[2] Keep approximations but make consistent: 48.3% (exact), ~48% (approx)
[3] Show me each occurrence to decide

Your choice: _
```

---

### 5. `placement` - Move Content to Correct Locations

**Input**: Placement issues from `/check-content-placement`

**Process**:
1. Extract content to move (assumption, example, proof)
2. Remove from current location
3. Insert at correct location
4. Update any references

**Example**:
```
Issue: Assumption 3.5 (LAN) used only in Section 7
Action: Move from model.tex to theory_lower_bound.tex

Steps:
1. Cut Assumption 3.5 block from model.tex
2. Paste before Theorem 7.1 in theory_lower_bound.tex
3. Renumber as Assumption 7.1
4. Update all \ref{ass:lan} calls
```

**Safety**: Always show preview before moving content.

---

### 6. `all` - Apply All Safe Fixes

Runs all fix types in order, skipping anything that needs user input.

```
═══════════════════════════════════════════════════════════════════
                      APPLYING ALL SAFE FIXES
═══════════════════════════════════════════════════════════════════

[1/5] Symbols...
      ✅ Fixed 2 conflicts

[2/5] Terms...
      ✅ Fixed 3 inconsistencies

[3/5] References...
      ✅ Fixed 1 broken ref
      ⚠️ Skipped 1 (needs content)

[4/5] Numbers...
      ⏭️ Skipped (needs user choice)

[5/5] Placement...
      ⏭️ Skipped (needs confirmation)

═══════════════════════════════════════════════════════════════════
                         SUMMARY
═══════════════════════════════════════════════════════════════════

✅ Auto-fixed: 6 issues
⚠️ Needs manual: 2 issues

📋 NEXT STEPS:
   /fix-issues numbers         → Choose number format
   /fix-issues placement       → Confirm content moves
   /paper-pipeline quick       → Verify all fixes
```

---

## Dry Run Mode

Add `--dry-run` to see what would be changed:

```
/fix-issues terms --dry-run
```

Output:
```
═══════════════════════════════════════════════════════════════════
                    DRY RUN - NO CHANGES MADE
═══════════════════════════════════════════════════════════════════

Would fix 8 terminology issues:

1. intro.tex:23
   - "human review" → "human audit"

2. experiments.tex:67
   - "review process" → "audit process"

3. experiments.tex:89
   - "reviewed" → "audited"

[... more ...]

To apply these changes:
   /fix-issues terms           → Apply all
   /fix-issues terms --interactive → Review each change
```

---

## Safety Rules

1. **Always backup**: Create backup before multi-file edits
2. **Preserve meaning**: Never change content meaning
3. **Ask when uncertain**: If fix is ambiguous, ask user
4. **Show diff**: For complex changes, show before/after
5. **One type at a time**: Don't mix fix types without confirmation

---

## Integration with Check Commands

```
/check-paper-consistency
    │
    ├── Issues found? ──────────────────────┐
    │                                       │
    ▼                                       ▼
/fix-issues symbols                 [No issues]
    │                                       │
    ▼                                       │
/check-paper-consistency ◄──────────────────┘
    │
    ▼
[Repeat for other issue types]
```

---

## Begin

1. Parse `$ARGUMENTS` for fix type
2. Read relevant state docs and check logs
3. Generate fix plan
4. If `--dry-run`: Show plan only
5. Else: Apply fixes with Edit tool
6. Update state docs
7. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 Fixes Applied: {N} issues fixed
   Type: {symbols/terms/refs/numbers/placement}

🔴 IMMEDIATE ACTIONS:
   1. Review changes made above
   2. Re-run the corresponding check command to verify

🛠️ RECOMMENDED COMMANDS (in order):

   [After /fix-issues symbols:]
   /check-paper-consistency    → Verify no more symbol conflicts

   [After /fix-issues terms:]
   /check-term-consistency     → Verify terminology consistent

   [After /fix-issues refs or numbers:]
   /check-cross-references     → Verify all refs valid

   [After /fix-issues placement:]
   /check-content-placement    → Verify content properly placed

   [After all L2 checks pass:]
   /check-ms-style             → Move to L3 (venue style)

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy
   L1 Structure   ─────────── /check-content-placement
 → L2 Consistency ─────────── /fix-issues works here
   L3 Style       ─────────── /check-ms-style
   L4 Language    ─────────── /polish-paper

💡 TIP: Use /paper-pipeline status to see overall progress
```
