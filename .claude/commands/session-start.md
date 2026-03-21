# Session Start - Context Recovery for New Sessions

You are a session initialization agent. Your task is to quickly restore context at the start of a new editing session by reading state documentation.

## Why This Matters

Without session start:
```
New session → "Where was I?"
           → Re-read 60 page paper
           → Miss what changed last time
           → Make inconsistent edits
           → 30 min wasted recovering context
```

With session start:
```
/session-start ms_judge → Read state docs (2 min)
                       → Show last changes
                       → Show pending issues
                       → Show recommended actions
                       → Ready to work immediately
```

## ⚠️ Protocol Reference

This command reads state docs following `.claude/commands/_shared/unified_protocol.md` Steps 0A–2.5 (resolve paper name, read state files, write checkpoint summary). It is read-only and does not trigger the post-edit rule.

## Arguments

- `$ARGUMENTS` - Paper name (e.g., `ms_judge`, `or_debug_bench`)

## Workflow

### Step 1: Load State Documentation

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** First resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name (e.g., `ms_journal`).

Then read key state files (in parallel for speed):

```
docs/paper_state/{resolved}/
├── overview.md          → Current status
├── changelog.md         → Recent changes
├── consistency_log.md   → Recent check results
└── review_responses.md  → Open reviewer comments
```

Write a **verification checkpoint** after reading:
```
State doc context loaded:
- overview.md: [paper status summary]
- changelog.md: [date of last change, what changed]
- consistency_log.md: [last check date, open issues]
```

### Step 2: Generate Context Summary

```
═══════════════════════════════════════════════════════════════════
                    SESSION START: ms_judge
                    2026-02-03 16:00
═══════════════════════════════════════════════════════════════════

📄 PAPER OVERVIEW
─────────────────────────────────────────
Title: Designing Service Systems with LLM Judges
Venue: Management Science
Status: Polishing (Round 3)
Pages: 61 (main + EC)

📅 LAST SESSION (2026-02-03 14:00)
─────────────────────────────────────────
Changes made:
• Fixed arms definition in introduction
• Consolidated π*∝√g to Theorem 5.2 only
• Resolved b_k vs b(t) symbol conflict

Files modified:
• sections/introduction.tex (+15 lines)
• sections/algorithm.tex (-5 lines, symbol fix)
• sections/theory_lower_bound.tex (-20 lines, removed redundancy)

📊 CURRENT REVIEW STATUS
─────────────────────────────────────────
Level 0 (Content):     ✅ Passed
Level 1 (Structure):   ✅ Passed
Level 2 (Consistency): ⚠️ 1 issue (term: "audit" vs "review")
Level 3 (Style):       ⚠️ 2 issues (managerial insights)
Level 4 (Language):    ⏸️ Waiting for L0-L3

📝 OPEN ISSUES (Priority Order)
─────────────────────────────────────────
1. [L2] Term inconsistency: "audit" (45x) vs "review" (8x)
   → Locations: intro:L23, experiments:L67, L89
   → Action: Replace "review" with "audit"

2. [L3] Missing managerial insights in conclusion
   → Location: conclusion.tex
   → Action: Add 2-3 actionable prescriptions

3. [L3] Prescriptions not specific enough
   → Location: discussion.tex
   → Action: Add numerical guidance

📋 REVIEWER COMMENTS (if in revision)
─────────────────────────────────────────
Total: 14 | Resolved: 10 | Open: 4

Open comments:
• R1.4: "Add ablation study" [L0] - In progress
• R2.5: "Figure legend placement" [L4] - Open
• R3.1: "Related work on service systems" [L0] - Open
• R3.4: "More managerial implications" [L3] - Open

═══════════════════════════════════════════════════════════════════
                    RECOMMENDED ACTIONS
═══════════════════════════════════════════════════════════════════

Based on current state, suggested work order:

1. 🔴 HIGH: Fix term inconsistency ("review" → "audit")
   Command: Search and replace in intro, experiments

2. 🟡 MEDIUM: Add managerial insights to conclusion
   Reference: .claude/writing_references/sentences/or_applications.md

3. 🟡 MEDIUM: Complete R1.4 ablation study
   File: experiments.tex

4. 🟢 LOW: Fix figure legend (after content stable)

═══════════════════════════════════════════════════════════════════
                    QUICK REFERENCE
═══════════════════════════════════════════════════════════════════

📁 Key Files:
   Paper:  paper/journal/main_ms.tex
   State:  docs/paper_state/ms_judge/
   Refs:   .claude/writing_references/

🛠️ Useful Commands:
   /paper-pipeline quick        → Fast consistency check
   /paper-pipeline review MS    → Full 5-level review
   /update-paper-state ms_judge → Sync docs after changes
   /check-term-consistency      → Fix "audit/review" issue
   /track-review-comments status → Check comment progress

📖 RAG References for Current Issues:
   Term consistency: writing_references/phrases/transitions.md
   Managerial insights: writing_references/sentences/or_applications.md
   Prescriptions: writing_references/paragraphs/main_results.md

═══════════════════════════════════════════════════════════════════
                    READY TO WORK
═══════════════════════════════════════════════════════════════════
Context loaded. You have full awareness of:
✅ Paper current state
✅ Last session changes
✅ Open issues and priority
✅ Reviewer comment status
✅ Recommended next actions
```

### Step 3: Load Relevant Context Files

After showing summary, automatically read into context:
- `framing.md` - Terminology rules
- `symbols.md` - Notation reference
- `cross_references.md` - What refs to update if changing results

This ensures the LLM has the key constraints loaded.

## Output Checklist

Session start output MUST include:

- [ ] Paper title and venue
- [ ] Current status (draft/polishing/revision)
- [ ] Last session summary (what changed)
- [ ] Open issues by priority
- [ ] Reviewer comment status (if applicable)
- [ ] Recommended next actions
- [ ] Quick command reference
- [ ] Relevant RAG references for current issues

## Integration

```
[Start new session]
        ↓
/session-start ms_judge     → Load context (2 min)
        ↓
[Work on recommended actions]
        ↓
/update-paper-state ms_judge → Sync before ending
        ↓
[End session - state persisted]
        ↓
[Next session]
        ↓
/session-start ms_judge     → Instant context recovery
```

## Begin

1. Parse `$ARGUMENTS` for paper name
2. Read state documentation files
3. Generate comprehensive context summary
4. Show recommended actions
5. Load key constraint files (framing, symbols)
6. Report ready status
7. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 Session Context Loaded
   Paper: {paper_name}
   Last Modified: {date}
   Pending Issues: {N}

📋 CONTEXT SUMMARY:
   - Last changes: [from changelog.md]
   - Open issues: [from review_responses.md or overview.md]
   - Key constraints: [from framing.md]

🔴 RECOMMENDED ACTIONS:
   {Based on paper state:}
   1. [Most urgent issue or next step]
   2. [Second priority]
   3. [Third priority]

🛠️ AVAILABLE COMMANDS:

   [For writing/editing:]
   /check-paper-consistency    → Verify symbols/notation
   /check-paper-flow           → Verify coherence

   [For review:]
   /paper-pipeline quick       → Fast consistency check
   /review-paper-full MS       → Comprehensive review

   [Before ending session:]
   /update-paper-state         → Sync all changes

💡 TIP: Run /update-paper-state before ending this session
```
