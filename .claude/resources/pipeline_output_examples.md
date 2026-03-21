# Pipeline Output Examples

Reference output formats for each pipeline stage. The orchestrator should adapt these formats to the actual results.

## Pipeline 1: `init` Output

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1/3: Create Documentation Ecosystem                       │
│ ████████████████████░░░░░░░░░░░░░░░░░░░░ 33%                   │
├─────────────────────────────────────────────────────────────────┤
│ Running: /init-paper-state [name]                               │
│                                                                 │
│ ✅ Created: docs/paper_state/[name]/overview.md                 │
│ ✅ Created: docs/paper_state/[name]/symbols.md                  │
│ ✅ Created: docs/paper_state/[name]/results.md                  │
│ ✅ Created: docs/paper_state/[name]/cross_references.md         │
│ ✅ Created: docs/paper_state/[name]/dependencies.md             │
│ ... (12 more files)                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2/3: Define Paper Framing                                 │
│ ████████████████████████████████████░░░░ 67%                   │
├─────────────────────────────────────────────────────────────────┤
│ Running: /define-paper-framing [venue]                          │
│                                                                 │
│ ✅ Extracted core concepts from paper                           │
│ ✅ Created terminology mappings                                 │
│ ✅ Built symbol registry                                        │
│ ✅ Saved to: docs/paper_state/[name]/framing.md                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3/3: Initial Consistency Scan                             │
│ ████████████████████████████████████████ 100%                  │
├─────────────────────────────────────────────────────────────────┤
│ Running: /check-paper-consistency                               │
│                                                                 │
│ Found: 3 potential symbol conflicts                             │
│ Found: 2 undefined abbreviations                                │
│ Saved to: docs/paper_state/[name]/consistency_log.md            │
└─────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
                       INITIALIZATION COMPLETE
═══════════════════════════════════════════════════════════════════

📊 Summary:
   • 15 documentation files created
   • 3 issues found in initial scan

📋 Next Steps:
   1. Review docs/paper_state/[name]/framing.md - verify terminology
   2. Fix 3 symbol conflicts identified
   3. Run: /paper-pipeline review [venue] - for full review

🛠️ Available Commands:
   /paper-pipeline review MS    → Full 5-level review
   /paper-pipeline quick        → Fast consistency check
   /update-paper-state [name]   → Sync docs after changes
```

---

## Pipeline 2: `review` Output

```
═══════════════════════════════════════════════════════════════════
                   FULL PAPER REVIEW PIPELINE
                   Target Venue: Management Science
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 0: CONTENT (Most Critical)                                │
│ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 20%                   │
├─────────────────────────────────────────────────────────────────┤
│ [1/3] Checking concept definition breadth...                    │
│       → Arms definition: ⚠️ Too narrow (LLM models only)        │
│                                                                 │
│ [2/3] Checking result redundancy...                             │
│       → π*∝√g stated in: Thm 5.2, Sec 7.2, Prop 7.3 ⚠️          │
│                                                                 │
│ [3/3] Checking experiment-framing alignment...                  │
│       → Experiments match framing ✅                            │
│                                                                 │
│ LEVEL 0 RESULT: ⚠️ 2 issues found                               │
└─────────────────────────────────────────────────────────────────┘

⏸️ CHECKPOINT: Level 0 has issues. Options:
   [1] Fix now → Edit paper, then run: /paper-pipeline review MS --resume L1
   [2] Continue → See all issues first, fix later
   [3] Details → Show detailed issue report

User input: 2 (continue)

┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 1: STRUCTURE                                              │
│ ████████████████░░░░░░░░░░░░░░░░░░░░░░░░ 40%                   │
├─────────────────────────────────────────────────────────────────┤
│ Running: /check-content-redundancy                              │
│       → Section organization ✅                                 │
│       → Example placement ⚠️ Ex 6.1 far from Thm 7.1            │
│                                                                 │
│ Running: /check-content-placement                               │
│       → Assumptions: A.4, A.5 introduced too early ⚠️           │
│       → Proofs: Sketch in Sec 7 too long ⚠️                     │
│                                                                 │
│ LEVEL 1 RESULT: ⚠️ 3 issues found                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 2: CONSISTENCY                                            │
│ ████████████████████████░░░░░░░░░░░░░░░░ 60%                   │
├─────────────────────────────────────────────────────────────────┤
│ Running: /check-paper-consistency                               │
│       → Symbol conflicts: b_k vs b(t) ⚠️                        │
│       → Term uniformity ✅                                      │
│                                                                 │
│ Running: /check-term-consistency                                │
│       → "audit" vs "review" inconsistency ⚠️                    │
│                                                                 │
│ Running: /check-cross-references                                │
│       → All refs valid ✅                                       │
│       → Numbers match ✅                                        │
│                                                                 │
│ LEVEL 2 RESULT: ⚠️ 2 issues found                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 3: VENUE STYLE (Management Science)                       │
│ ████████████████████████████████░░░░░░░░ 80%                   │
├─────────────────────────────────────────────────────────────────┤
│ Running: /check-ms-style                                        │
│       → Managerial insights: ⚠️ Missing in conclusion           │
│       → Service framing: ✅                                     │
│       → Prescriptions: ⚠️ Not actionable enough                 │
│                                                                 │
│ LEVEL 3 RESULT: ⚠️ 2 issues found                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 4: LANGUAGE (Only if L0-L3 pass)                          │
│ ████████████████████████████████████████ 100%                  │
├─────────────────────────────────────────────────────────────────┤
│ ⏭️ SKIPPED: Fix Level 0-3 issues first                          │
│                                                                 │
│ Language polish is premature when content/structure has issues. │
│ Run /polish-paper AFTER fixing higher-level problems.           │
└─────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
                       REVIEW COMPLETE
═══════════════════════════════════════════════════════════════════

📊 Summary by Level:
   Level 0 (Content):     ⚠️ 2 issues
   Level 1 (Structure):   ⚠️ 3 issues
   Level 2 (Consistency): ⚠️ 2 issues
   Level 3 (Style):       ⚠️ 2 issues
   Level 4 (Language):    ⏭️ Skipped
   ─────────────────────────────────
   TOTAL:                 9 issues

📋 Priority Fix Order:
   1. [L0] Broaden arms definition (intro, model)
   2. [L0] Consolidate π*∝√g to one location
   3. [L2] Resolve b_k vs b(t) symbol conflict
   4. [L2] Use "audit" consistently, not "review"
   5. [L1] Move A.4, A.5 to Section 7 (just-in-time)
   6. [L1] Move Ex 6.1 near Thm 7.1
   7. [L3] Add managerial insights to conclusion
   8. [L3] Make prescriptions more actionable
   9. [L1] Shorten proof sketch in Sec 7

📁 Full report saved to:
   docs/paper_state/[name]/review_report_[date].md

🛠️ Next Steps:
   /paper-pipeline status              → See current state
   [Fix issues in paper]
   /update-paper-state [name]          → Sync docs
   /paper-pipeline review MS --resume  → Re-run from checkpoint
   /paper-pipeline quick               → Fast re-check
```

---

## Pipeline 3: `quick` Output

```
═══════════════════════════════════════════════════════════════════
                     QUICK CONSISTENCY CHECK
═══════════════════════════════════════════════════════════════════

Running 6 checks...

[1/6] Symbol consistency    ████████████████████ ✅ No conflicts
[2/6] Term consistency      ████████████████████ ✅ All consistent
[3/6] Cross-references      ████████████████████ ⚠️ 1 broken ref
[4/6] Numerical values      ████████████████████ ✅ All match
[5/6] Formula consistency   ████████████████████ ⚠️ 1 main↔appendix mismatch
[6/6] State doc freshness   ████████████████████ ⚠️ 2 stale docs

═══════════════════════════════════════════════════════════════════
                       QUICK CHECK COMPLETE
═══════════════════════════════════════════════════════════════════

📊 Result: ⚠️ 3 issues found

⚠️ Issue 1: Broken forward reference
   Location: analysis.tex:L80
   Text: "as shown in EC.3"
   Problem: EC.3 does not exist

⚠️ Issue 2: State doc staleness
   figures_tables.md: Missing tab:queue-config (exists in service_system_details.tex:311)
   cross_references.md: Missing tab:queue-config
   Action: Auto-fix by adding missing entries

🛠️ Next Steps:
   [Fix the broken reference]
   [State docs auto-updated with missing entries]
   /paper-pipeline quick     → Re-run quick check
   /paper-pipeline review MS → Full review when ready
```

---

## Pipeline 4: `pre-submit` Output

```
═══════════════════════════════════════════════════════════════════
                   PRE-SUBMISSION CHECKLIST
                   Target: Management Science
═══════════════════════════════════════════════════════════════════

DOCUMENTATION
[✅] Paper state docs up to date
[✅] Changelog has recent entries
[✅] Framing doc matches paper

CONTENT
[✅] All theorems validated experimentally
[✅] No redundant result statements
[⚠️] Abstract mentions "48%" but experiments say "48.3%"

CONSISTENCY
[✅] No symbol conflicts
[✅] Terms used uniformly
[✅] All cross-references valid

STYLE (MS-specific)
[✅] Managerial insights present
[✅] Service system framing clear
[⚠️] Conclusion could be more prescriptive

LANGUAGE
[✅] No AI-flagged words
[✅] Transitions varied
[✅] Subject-verb proximity good

FORMATTING
[✅] INFORMS4 template
[✅] Page limit OK (61 pages with EC)
[⚠️] Figure 3 legend inside plot area

═══════════════════════════════════════════════════════════════════
                    SUBMISSION READINESS: 85%
═══════════════════════════════════════════════════════════════════

🔴 Must Fix (3):
   1. Harmonize abstract "48%" with experiments "48.3%"
   2. Make conclusion more prescriptive
   3. Move Figure 3 legend outside plot

🟡 Recommended (0):
   None

🟢 Ready: All other checks pass

🛠️ After fixes:
   /paper-pipeline pre-submit MS   → Re-run checklist
   [Submit when 100%]
```

---

## Pipeline 5: `revision` Output

```
═══════════════════════════════════════════════════════════════════
                     REVISION WORKFLOW
═══════════════════════════════════════════════════════════════════

STAGE 1: Load Reviewer Comments
──────────────────────────────────────────────────────────────────
/track-review-comments status

📊 Current Status:
   Total comments: 14
   Resolved: 0 (0%)
   In progress: 0
   Open: 14

STAGE 2: Categorize Comments
──────────────────────────────────────────────────────────────────
Auto-categorizing by review level...

| Level | Comments | IDs |
|-------|----------|-----|
| L0 Content | 3 | R1.1, R2.3, R3.1 |
| L1 Structure | 2 | R1.4, R2.1 |
| L2 Consistency | 4 | R1.2, R2.2, R2.4, R3.2 |
| L3 Style | 4 | R1.3, R2.5, R3.3, R3.4 |
| L4 Language | 1 | R1.5 |

STAGE 3: Suggested Fix Order
──────────────────────────────────────────────────────────────────
Fix Level 0 first (affects most downstream):
   1. R1.1 - Arms definition
   2. R2.3 - Missing ablation
   3. R3.1 - Related work gap

Then Level 2 (prevents cascading issues):
   4. R1.2 - Symbol conflict b_k vs b(t)
   ...

🛠️ Commands for this session:
   /track-review-comments respond R1.1 "[response]"
   /track-review-comments resolve R1.1
   /update-paper-state [name]
   /track-review-comments status   → Check progress
```

---

## Pipeline 7: `restructure` Output

```
═══════════════════════════════════════════════════════════════════
                  SECTION RESTRUCTURE WORKFLOW
                  Target: Section 6
═══════════════════════════════════════════════════════════════════

STAGE 1: Pre-Restructure Snapshot
──────────────────────────────────────────────────────────────────
[1/4] Catalog current structure...
      → Section 6: 4 subsections, 114 lines
      → Labels: thm:cost_bound, prop:decomposition, thm:optimality, ...
      → Cross-refs TO section 6: 8 (from intro, algorithm, experiments, conclusion)
      → Cross-refs FROM section 6: 3 (to appendix proofs)

[2/4] Build content map...
      → 3 theorems, 2 propositions, 1 example, 1 algorithm
      → Each with: label, current file, all references

[3/4] Record scope constraint...
      → Protected sections: 1-5 (do NOT modify)
      → Editable sections: 6+ and appendix

[4/4] Snapshot state docs...
      → cross_references.md: backed up
      → results.md: backed up

STAGE 2: Execute Restructure (User-Guided)
──────────────────────────────────────────────────────────────────
For each content move, track:

| Content | From | To | Action |
|---------|------|----|--------|
| thm:cost_bound | appendix | Section 6 | PROMOTE |
| prop:comparative_statics | Section 6.3 | conclusion | MOVE |
| thm:optimality | Section 6.4 | appendix | DEMOTE |
| prop:decomposition | Section 6.2 | (keep in appendix) | REMOVE from main |

STAGE 3: Post-Restructure Verification
──────────────────────────────────────────────────────────────────
[1/5] Content move integrity...
      → Check all references to moved content ✅
      → Check orphaned Cref groups ⚠️ 1 found

[2/5] Formula consistency...
      → Compare main text examples with appendix derivations ✅
      → Check notation propagation ✅

[3/5] Cross-reference update...
      → Update cross_references.md ✅
      → Update results.md ✅

[4/5] Scope constraint verification...
      → Verify no edits to protected sections ✅

[5/5] Compile and check...
      → pdflatex: 0 errors ✅
      → 0 undefined references ✅
      → 0 multiply-defined labels ✅

═══════════════════════════════════════════════════════════════════
                   RESTRUCTURE COMPLETE
═══════════════════════════════════════════════════════════════════

📊 Summary:
   • 4 content moves executed
   • 1 orphaned reference fixed
   • 0 formula mismatches
   • Protected sections: untouched ✅

🛠️ Next Steps:
   /paper-pipeline quick         → Verify consistency
   /polish-paper [section]       → Polish restructured section
   /update-paper-state [name]    → Sync all state docs
```

---

## Pipeline 8: `status` Output

```
═══════════════════════════════════════════════════════════════════
                       PAPER STATUS
═══════════════════════════════════════════════════════════════════

📄 Paper: ms_judge_paper
📍 Venue: Management Science
📅 Last Updated: 2026-02-03 15:30

REVIEW STATUS
─────────────────────────────────────────
Last full review: 2026-02-03 10:00
Issues then: 9
Issues now: 3 (after fixes)

Level 0 (Content):     ✅ Passed
Level 1 (Structure):   ✅ Passed
Level 2 (Consistency): ⚠️ 1 issue remaining
Level 3 (Style):       ⚠️ 2 issues remaining
Level 4 (Language):    ⏸️ Not started

RECENT CHANGES (from changelog.md)
─────────────────────────────────────────
• 2026-02-03 15:30: Fixed arms definition
• 2026-02-03 14:00: Consolidated π*∝√g
• 2026-02-03 12:00: Resolved symbol conflict

DOCUMENTATION STATUS
─────────────────────────────────────────
symbols.md:         ✅ Up to date
results.md:         ✅ Up to date
cross_references.md: ⚠️ Needs sync
changelog.md:       ✅ Up to date

═══════════════════════════════════════════════════════════════════
                   RECOMMENDED NEXT ACTION
═══════════════════════════════════════════════════════════════════

Based on current state, you should:

   1. /update-paper-state ms_judge   → Sync cross_references.md
   2. Fix remaining L2 issue (term consistency)
   3. Fix remaining L3 issues (MS style)
   4. /paper-pipeline quick          → Verify fixes
   5. /polish-paper                  → Language polish (when L0-L3 pass)

🛠️ Quick Commands:
   /paper-pipeline review MS    → Full re-review
   /paper-pipeline quick        → Fast consistency check
   /paper-pipeline pre-submit MS → Submission checklist
```
