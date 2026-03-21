# Fix Issues - Example Outputs & Templates

## Symbol Fix Example

```
Conflict: b_k (bias) vs b(t) (best arm)

Decision: Keep b_k for bias, rename b(t) to k̂(t)

Actions:
1. In algorithm.tex: Replace "b(t)" → "\\hat{k}(t)" (15 occurrences)
2. In analysis.tex: Replace "b(t)" → "\\hat{k}(t)" (8 occurrences)
3. Update symbols.md with new canonical symbol
```

**Output:**
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

## Terms Fix Example

```
Inconsistency: "audit" (45x) vs "review" (8x)
Canonical: "audit" (per framing.md)

Actions:
1. In intro.tex L23: "human review" → "human audit"
2. In experiments.tex L67: "review process" → "audit process"
3. In experiments.tex L89: "reviewed" → "audited"
```

**Smart Replacement Rules:**
- "review" → "audit" (noun)
- "reviewed" → "audited" (past tense)
- "reviewing" → "auditing" (gerund)
- "reviewer" → "auditor" (agent) [only if appropriate]
- Preserve capitalization
- Don't replace in citations or proper nouns

**Output:**
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

## Refs Fix Example

```
Issue 1: \ref{thm:foo} but no \label{thm:foo}
Action: Find theorem about "foo", add \label{thm:foo}

Issue 2: \label{eq:old} but never referenced
Action: Remove label (or ask user)

Issue 3: "as shown in EC.3" but EC.3 doesn't exist
Action: Flag for user (can't auto-fix content gaps)
```

**Output:**
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

## Numbers Fix — User Choice

```
Found inconsistent number: 48% vs 48.3%

Options:
[1] Use exact value everywhere: 48.3%
[2] Keep approximations but make consistent: 48.3% (exact), ~48% (approx)
[3] Show me each occurrence to decide

Your choice: _
```

---

## Placement Fix Example

```
Issue: Assumption 3.5 (LAN) used only in Section 7
Action: Move from model.tex to theory_lower_bound.tex

Steps:
1. Cut Assumption 3.5 block from model.tex
2. Paste before Theorem 7.1 in theory_lower_bound.tex
3. Renumber as Assumption 7.1
4. Update all \ref{ass:lan} calls
```

---

## "All" Fix Output

```
═══════════════════════════════════════════════════════════════════
                     APPLYING ALL SAFE FIXES
═══════════════════════════════════════════════════════════════════

[1/6] Symbols...
      ✅ Fixed 2 conflicts

[2/6] Terms...
      ✅ Fixed 3 inconsistencies

[3/6] References...
      ✅ Fixed 1 broken ref
      ⚠️ Skipped 1 (needs content)

[4/6] Numbers...
      ⏭️ Skipped (needs user choice)

[5/6] Placement...
      ⏭️ Skipped (needs confirmation)

[6/6] Figures...
      ✅ Fixed 2 rendering issues

═══════════════════════════════════════════════════════════════════
                        SUMMARY
═══════════════════════════════════════════════════════════════════

✅ Auto-fixed: 6 issues
⚠️ Needs manual: 2 issues

📋 NEXT STEPS:
   /fix-issues numbers         → Choose number format
   /fix-issues placement       → Confirm content moves
   /fix-figures all            → Fix figure rendering issues
   /paper-pipeline quick       → Verify all fixes
```

---

## Dry Run Output

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

## Next Steps Footer Template

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

   [After /fix-issues figures:]
   /check-figures-tables       → Verify figure rendering

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
