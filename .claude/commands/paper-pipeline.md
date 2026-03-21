# Paper Pipeline - Automated Multi-Stage Review Orchestration

You are a paper review orchestrator. Your task is to run the complete review pipeline automatically, showing progress and next steps at each stage.

## Why This Matters

Manual workflow:
```
/check-paper-consistency → "Done, what next?"
/check-content-redundancy → "Done, what next?"
... (user must remember all 10+ skills)
```

Automated pipeline:
```
/paper-pipeline review MS → Runs ALL checks automatically
                         → Shows progress bar
                         → Suggests next action at each stage
```

## ⚠️ MANDATORY FIRST STEP: Resolve Paper Name

**STOP.** Before running ANY pipeline stage, you MUST:
1. Run `ls docs/paper_state/` to find the actual paper directory name
2. Set `PAPER_STATE_DIR` for all subsequent stages (e.g., `docs/paper_state/ms_journal`)
3. Read the paper overview to understand current state:
```
Read docs/paper_state/{resolved}/overview.md
Read docs/paper_state/{resolved}/changelog.md
```

**All sub-commands invoked by the pipeline MUST use the resolved paper name, not `[paper]`.**

## Arguments

- `$ARGUMENTS` - Pipeline type and options:
  - `init [name] [venue]` - Initialize new paper (creates state docs)
  - `review [venue]` - Full 5-level review pipeline
  - `quick` - Fast consistency checks only
  - `pre-submit [venue]` - Final submission checklist
  - `revision` - Post-review revision workflow
  - `restructure [section]` - Section reorganization checklist
  - `status` - Show current pipeline state and next steps
  - `lit [keys...]` - Download arXiv sources & generate structured summaries
  - `lit scan` - Audit references.bib vs existing summaries
  - `lit --missing` - Process all bib entries that lack summaries

### Global Options

- `--scope "section N+"` — Restrict all checks to Section N and later. Protects earlier sections from accidental modification.
- `--scope "file1.tex,file2.tex"` — Restrict checks to specific files only.
- `--dry-run` — Preview changes without applying (for fix/restructure pipelines).

## Pipelines

### Pipeline 1: `init [name] [venue]`

**Purpose**: Set up new paper for structured workflow

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

### Pipeline 2: `review [venue]`

**Purpose**: Complete 5-level review with automatic progression

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

### Pipeline 3: `quick`

**Purpose**: Fast consistency verification (use after small changes)

**Implementation**: Run 5 checks (4 content checks in parallel, then 1 state doc freshness check).

**Check 5 (State Doc Freshness)** is critical: it detects when .tex files were edited but state docs were NOT updated. This catches the common failure mode where ad-hoc edits bypass the formal command pipeline.

**State Doc Freshness Check procedure**:
1. Grep all `\label{tab:*}` and `\label{fig:*}` in `paper/journal/sections/**/*.tex`
2. Compare against entries in `figures_tables.md` and `cross_references.md`
3. Any label in .tex but NOT in state docs → STALE
4. Check `changelog.md` last entry date vs today → if edits were made today but changelog has no today entry → STALE

```
═══════════════════════════════════════════════════════════════════
                      QUICK CONSISTENCY CHECK
═══════════════════════════════════════════════════════════════════

Running 5 checks...

[1/5] Symbol consistency    ████████████████████ ✅ No conflicts
[2/5] Term consistency      ████████████████████ ✅ All consistent
[3/5] Cross-references      ████████████████████ ⚠️ 1 broken ref
[4/5] Numerical values      ████████████████████ ✅ All match
[5/5] State doc freshness   ████████████████████ ⚠️ 2 stale docs

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

**When staleness is detected**: The pipeline MUST auto-fix by updating the stale state docs immediately, then report what was fixed. Do NOT just report — fix it.

---

### Pipeline 4: `pre-submit [venue]`

**Purpose**: Final checklist before submission

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

### Pipeline 5: `revision`

**Purpose**: Post-review revision workflow

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

### Pipeline 6: `restructure [section]`

**Purpose**: Guided workflow for section reorganization (promote/demote/move content between main text and e-companion)

```
═══════════════════════════════════════════════════════════════════
                    SECTION RESTRUCTURE WORKFLOW
                    Target: §6 Stochastic Boundaries
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1/3: Pre-Restructure Snapshot                             │
│ ████████████████████░░░░░░░░░░░░░░░░░░░░ 33%                   │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Cataloged section structure (4 theorems, 2 corollaries)      │
│ ✅ Built content map: labels → locations                        │
│ ✅ Recorded scope constraint: §6 only                           │
│ ✅ Backed up state docs (cross_references.md, results.md)       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2/3: Execute Restructure                                  │
│ ████████████████████████████░░░░░░░░░░░░ 67%                   │
├─────────────────────────────────────────────────────────────────┤
│ Tracking moves:                                                 │
│   DEMOTE  Cor 6.3 → EC.4.1        (cross-refs updated: 3)      │
│   DEMOTE  Rem 6.4 → EC.4.2        (cross-refs updated: 1)      │
│   PROMOTE EC.3.5 → Thm 6.2        (cross-refs updated: 5)      │
│   MOVE    §6.5 content → §6.3     (renumbered)                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3/3: Post-Restructure Verification                        │
│ ████████████████████████████████████████ 100%                  │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Content move integrity: all text accounted for               │
│ ✅ Formula consistency: no broken equations                     │
│ ✅ Cross-references: all \ref{} updated                         │
│ ✅ Scope constraint: only §6 and EC.4 modified                  │
│ ✅ LaTeX compile: 0 errors, 0 undefined refs                    │
└─────────────────────────────────────────────────────────────────┘
```

**Stage Details**:
1. **Pre-restructure snapshot**: Catalog structure, build label→location map, record scope constraint, backup state docs
2. **Execute restructure**: Track each move (PROMOTE/DEMOTE/MOVE/REMOVE) with source, destination, affected cross-references. Update all `\ref{}`, `\label{}`, proof pointers.
3. **Post-restructure verification**: Content move integrity, formula consistency, cross-reference update, scope constraint verification, compile check

Auto-runs after completion: label→location map refresh, reference graph update, state doc sync via `/update-paper-state`.

---

### Pipeline 7: `status`

**Purpose**: Show current state and suggest next action

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

---

### Pipeline 8: `lit [keys...]`

**Purpose**: Download arXiv paper sources and generate structured summaries for cited literature

```
═══════════════════════════════════════════════════════════════════
                    LITERATURE ACQUISITION PIPELINE
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1/3: Coverage Audit                                       │
│ ████████████████████░░░░░░░░░░░░░░░░░░░░ 33%                   │
├─────────────────────────────────────────────────────────────────┤
│ references.bib: 52 entries                                      │
│ With summaries: 8                                               │
│ Missing: 44                                                     │
│                                                                 │
│ Targets (from args or --missing):                               │
│   wierman2012, bansal2007, halfin1981, yds1995 ...              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2/3: Download & Summarize (Parallel)                      │
│ ████████████████████████████░░░░░░░░░░░░ 67%                   │
├─────────────────────────────────────────────────────────────────┤
│ [Batch 1] wierman2012, bansal2007        🔄 Running...          │
│ [Batch 2] halfin1981, yds1995            🔄 Running...          │
│ [Batch 3] lin2013, gandhi2009            🔄 Running...          │
│                                                                 │
│ ✅ wierman2012 — Downloaded + SUMMARY written                   │
│ ✅ bansal2007  — Downloaded + SUMMARY written                   │
│ ⚠️ halfin1981  — No arXiv source, web-based summary             │
│ ...                                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3/3: Update Index                                         │
│ ████████████████████████████████████████ 100%                  │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Updated literature/README.md with 6 new entries              │
└─────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
                    LITERATURE ACQUISITION COMPLETE
═══════════════════════════════════════════════════════════════════

📊 Summary:
   Downloaded: 5 arXiv sources
   Web-based: 1 (no arXiv source)
   Summaries written: 6
   Failed: 0

📋 Next Steps:
   /paper-pipeline lit scan        → Check remaining coverage gaps
   /paper-pipeline review OR       → Full review with literature grounded
   /proofread-references           → Verify bib entries match sources
```

**Implementation**: Dispatch to `/lit-acquire` skill.

**Search Backend Priority** (when looking up a paper):
1. **arXiv API** (existing) — primary source for CS/OR papers
2. **Semantic Scholar MCP** (new) — fallback when arXiv search misses, also provides:
   - Citation count and influence metrics
   - BibTeX export
   - Related papers list
   - Abstract even for non-arXiv papers (e.g., journal-only publications)

When Semantic Scholar MCP is available (see `.claude/commands/_shared/mcp_writing_tools.md`):
- Use `semantic-scholar.search_papers` for keyword searches
- Use `semantic-scholar.get_paper` for DOI/arXiv ID lookups
- Graceful degradation: if MCP unavailable, use arXiv API only

**Dispatch Plan**:
- `lit scan` → Skill: lit-acquire, args: "scan"
- `lit [keys...]` → Skill: lit-acquire, args: "batch [keys...]"
- `lit --missing` → Skill: lit-acquire, args: "batch --from-bib --missing"

---

## Implementation: Task-Based Subagent Dispatch

**The pipeline orchestrator MUST use the Task tool to dispatch checks as independent subagents.** This is critical for quality — each check gets its own clean context, and independent checks run in parallel.

### Dispatch Protocol

Follow `.claude/commands/_shared/orchestrator_protocol.md` for the Dispatch Mode Selection rules.

**Before dispatching any Task**, resolve the paper name and state directory:
```
PAPER_DIR = paper/journal/
PAPER_STATE_DIR = docs/paper_state/ms_journal/   (resolved via ls docs/paper_state/)
```

### Task Prompt Template

Every Task dispatch MUST include these four elements:

```
Task(
  subagent_type: "general-purpose",
  description: "[short description]",
  prompt: """
    You are executing the [skill-name] skill.

    **Instructions**: Read .claude/commands/[skill-name].md and follow it completely.
    **Protocol**: Read .claude/commands/_shared/unified_protocol.md (mandatory read protocol).
    **Paper directory**: [PAPER_DIR]
    **Paper state directory**: [PAPER_STATE_DIR]

    Execute the skill and return a structured report with:
    - Issues found (location, severity, description)
    - Suggested fixes
    - Summary count
  """
)
```

### Pipeline 2 (`review`) — Dispatch Plan

**Level 0 — Content**: Sequential (depends on each other)
```
Task: "Execute check-content-redundancy for [PAPER_DIR]"  → wait for result
Task: "Execute check-paper-flow for [PAPER_DIR]"          → wait for result
```

**Level 1 — Structure**: Parallel (independent)
```
Task: "Execute check-content-redundancy for [PAPER_DIR]"  ┐
Task: "Execute check-content-placement for [PAPER_DIR]"   ┘ parallel
```

**Level 2 — Consistency**: Parallel (independent)
```
Task: "Execute check-paper-consistency for [PAPER_DIR]"   ┐
Task: "Execute check-term-consistency for [PAPER_DIR]"    │ parallel
Task: "Execute check-cross-references for [PAPER_DIR]"    ┘
```

**Level 3 — Venue Style**: Single Task
```
Task: "Execute check-{venue}-style for [PAPER_DIR]"
  MS   → check-ms-style
  MSOM → check-msom-style
  OR   → check-or-style
  ML   → check-ml-style
```

**Level 4 — Language**: Single Task (only if L0-L3 pass)
```
Task: "Execute polish-paper for [PAPER_DIR]"
```

**After all levels**: Aggregate results from all Task outputs, generate summary.

### Pipeline 3 (`quick`) — Dispatch Plan

All 4 content checks in parallel, then 1 sequential freshness check:

```
// Parallel batch (single message with 4 Task calls):
Task: "Execute check-paper-consistency for [PAPER_DIR]. Focus on symbols only. Return concise report."
Task: "Execute check-term-consistency for [PAPER_DIR]. Return concise report."
Task: "Execute check-cross-references for [PAPER_DIR]. Return concise report."
Task: "Check numerical consistency in [PAPER_DIR]. Compare numbers across abstract, intro, experiments, conclusion."

// After parallel batch completes:
// State Doc Freshness Check (run directly, no subagent needed — simple grep + compare)
```

### Pipeline 4 (`pre-submit`) — Dispatch Plan

Run all checks in parallel, then aggregate into checklist:

```
// Parallel batch:
Task: "Execute check-paper-consistency for [PAPER_DIR]"
Task: "Execute check-term-consistency for [PAPER_DIR]"
Task: "Execute check-cross-references for [PAPER_DIR]"
Task: "Execute check-ms-style for [PAPER_DIR]"           (or venue-specific)
Task: "Execute check-figures-tables for [PAPER_DIR]"

// After all complete: Aggregate into submission checklist format
```

### Lightweight Operations (Use Skill tool directly)

These don't need Task subagents:
- `status` — just reads state docs
- `revision` — interactive, needs user context
- `init` — sequential, needs conversation flow

## Begin

Parse `$ARGUMENTS` and execute appropriate pipeline. If `--scope` is present, pass the scope constraint to all sub-commands.

- `init` → Skill: init-paper-state → Skill: define-paper-framing → Task: check-paper-consistency
- `review` → Task-based 5-level review (see dispatch plan above). **Venue detection**: `MS`→check-ms-style, `MSOM`/`M&SOM`→check-msom-style, `OR`→check-or-style, `ML`/`NeurIPS`/`ICML`→check-ml-style
- `quick` → Task-based parallel checks (see dispatch plan above)
- `pre-submit` → Task-based parallel checks → aggregate into checklist
- `revision` → Skill: track-review-comments (interactive)
- `restructure` → 3-stage guided workflow (snapshot → execute → verify). Scope constraint mandatory.
- `status` → Direct: read overview.md + changelog.md
- `lit` → Skill: lit-acquire (download sources + write summaries)
