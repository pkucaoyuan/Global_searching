# Track Review Comments - Systematic Reviewer Response Management

You are a review tracking agent. Your task is to systematically record, categorize, and track responses to reviewer comments.

## Why This Matters

Without tracking:
```
Reviewer: "14 comments"
→ Fix 10, forget 4
→ Resubmit
→ Reviewer: "4 issues still unaddressed"
→ Desk reject
```

With tracking:
```
/track-review-comments add [file]    → All 14 comments recorded
/track-review-comments status        → 10/14 resolved, 4 open
→ Can't miss any comment
```

## ⚠️ Protocol Reference

This command modifies `review_responses.md`. After any state doc modification, you MUST update `changelog.md` with a dated entry. See `.claude/commands/_shared/unified_protocol.md` Step 4 (Universal Post-Edit Rule).

## Arguments

- `$ARGUMENTS` - Action and parameters:
  - `add [file]` - Parse and add comments from file/text
  - `status` - Show current status of all comments
  - `resolve [id]` - Mark comment as resolved
  - `respond [id] "[response]"` - Add response to comment
  - `categorize` - Auto-categorize all comments by review level
  - `export` - Generate response letter

## Document Location

**⚠️ MANDATORY.** Resolve `[paper_name]` first → run `ls docs/paper_state/` to find the actual directory name (e.g., `ms_journal`).

`docs/paper_state/{resolved}/review_responses.md`

Before adding or updating review comments, read existing state docs:
```
Read docs/paper_state/{resolved}/review_responses.md  # Existing comments (if any)
Read docs/paper_state/{resolved}/changelog.md         # Recent changes
Read docs/paper_state/{resolved}/overview.md          # Paper status
```

## Comment Structure

Each comment is tracked with:

```markdown
### [R1.3] Reviewer 1, Comment 3

**Original Comment**:
> "The arms definition in the introduction (Section 1) is too narrow..."

**Category**: L3-Style (MS framing)
**Severity**: Major / Minor / Editorial
**Status**: Open / In Progress / Resolved / Won't Fix

**Analysis**:
- Root cause: [Why did this happen?]
- Skill to prevent: `/check-ms-style` or `/define-paper-framing`

**Response**:
> We have broadened the arms definition to include...

**Changes Made**:
| File | Line | Change |
|------|------|--------|
| introduction.tex | L20-25 | Rewrote arms definition |
| model.tex | L15 | Added "service configurations" |

**Verification**:
- [ ] Change made
- [ ] Documented in changelog.md
- [ ] No new inconsistencies introduced
```

## Workflow

### Step 1: Add Comments (`add`)

Parse reviewer comments from:
- PDF review file
- Email text
- Pasted text

```
/track-review-comments add "
Reviewer 1:
1. The arms definition is too narrow.
2. Missing managerial insights in discussion.

Reviewer 2:
1. Symbol b_k conflicts with b(t).
"
```

Output:
```markdown
## Comments Added

| ID | Reviewer | Summary | Category | Severity |
|----|----------|---------|----------|----------|
| R1.1 | 1 | Arms definition narrow | L3-Style | Major |
| R1.2 | 1 | Missing insights | L3-Style | Major |
| R2.1 | 2 | Symbol conflict | L2-Consistency | Minor |

Total: 3 comments added
```

### Step 2: Categorize (`categorize`)

Auto-assign categories based on content:

| Pattern | Category | Skill |
|---------|----------|-------|
| "definition", "framing", "narrow", "broad" | L0-Content | `/check-ms-style` |
| "structure", "section", "move", "reorganize" | L1-Structure | `/check-content-placement` |
| "symbol", "notation", "conflict", "inconsistent" | L2-Consistency | `/check-paper-consistency` |
| "managerial", "insights", "practitioners" | L3-Style | `/check-ms-style` |
| "writing", "clarity", "passive", "wordy" | L4-Language | `/polish-paper` |

### Step 3: Track Progress (`status`)

```
/track-review-comments status
```

Output:
```markdown
# Review Response Status

**Paper**: [name]
**Review Round**: 1
**Date Received**: [date]
**Deadline**: [date]

---

## Overall Progress

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Resolved | 10 | 71% |
| 🚧 In Progress | 2 | 14% |
| ❌ Open | 2 | 14% |
| ⏸️ Won't Fix | 0 | 0% |

**Progress Bar**: ████████████░░ 71%

---

## By Category

| Category | Total | Resolved | Open |
|----------|-------|----------|------|
| L0-Content | 2 | 2 | 0 |
| L1-Structure | 3 | 3 | 0 |
| L2-Consistency | 4 | 2 | 2 |
| L3-Style | 3 | 2 | 1 |
| L4-Language | 2 | 1 | 1 |

---

## By Reviewer

| Reviewer | Total | Resolved | Open |
|----------|-------|----------|------|
| Reviewer 1 | 6 | 5 | 1 |
| Reviewer 2 | 5 | 3 | 2 |
| Reviewer 3 | 3 | 2 | 1 |

---

## Open Comments (Need Attention)

| ID | Summary | Category | Assigned |
|----|---------|----------|----------|
| R1.4 | Missing ablation | L0-Content | - |
| R2.2 | Figure legend | L4-Language | - |
| R2.5 | Proof sketch too long | L1-Structure | - |
| R3.1 | Related work gap | L0-Content | - |

---

## Recently Resolved

| ID | Summary | Resolved Date |
|----|---------|---------------|
| R1.1 | Arms definition | 2026-02-03 |
| R1.2 | Managerial insights | 2026-02-03 |
```

### Step 4: Respond (`respond`)

```
/track-review-comments respond R1.1 "We have broadened the arms definition to include operational design, technology choices, and parameter settings, following the service system design literature."
```

Updates the comment entry with response text.

### Step 5: Resolve (`resolve`)

```
/track-review-comments resolve R1.1
```

Marks comment as resolved, requires:
- Response text filled in
- Changes made documented
- Verification checklist complete

### Step 6: Export (`export`)

Generate response letter for resubmission:

```markdown
# Response to Reviewers

**Paper**: [Title]
**Manuscript #**: [number]
**Date**: [date]

---

Dear Editor and Reviewers,

We thank the reviewers for their constructive feedback. We have carefully addressed all comments as detailed below.

---

## Response to Reviewer 1

### Comment R1.1
> "The arms definition in the introduction is too narrow..."

**Response**: We have broadened the arms definition to include operational design, technology choices, and parameter settings. Specifically:
- Introduction (page 2, lines 20-25): Rewrote arms definition
- Model section (page 5, line 15): Added "service configurations"

### Comment R1.2
> "Missing managerial insights..."

**Response**: We have added a dedicated Discussion section (Section 10) that provides actionable insights for practitioners...

---

## Response to Reviewer 2

...

---

## Summary of Major Changes

1. Broadened arms definition throughout (R1.1)
2. Added Discussion section with managerial insights (R1.2, R3.2)
3. Resolved symbol conflicts: changed b(t) to k̂(t) (R2.1)
4. ...

---

Sincerely,
The Authors
```

## Recurring Issue Detection

After resolving all comments, analyze for patterns:

```markdown
## Recurring Issues Analysis

| Issue Pattern | Count | Root Cause | Prevention |
|---------------|-------|------------|------------|
| Symbol conflicts | 3 | No symbol registry | `/init-paper-state` → symbols.md |
| Inconsistent terms | 2 | No framing doc | `/define-paper-framing` |
| Content redundancy | 2 | No result registry | `/init-paper-state` → results.md |
| MS style issues | 4 | Not using style checker | `/check-ms-style` |

## Recommendations for Future Papers

1. **Before writing**: `/init-paper-state` + `/define-paper-framing`
2. **During writing**: Consult state docs frequently
3. **Before submission**: `/review-paper-full [venue]`
```

## Integration

```
[Receive reviews]
        ↓
/track-review-comments add [file]     → Parse and record
        ↓
/track-review-comments categorize     → Auto-categorize
        ↓
[Work on responses]
        ↓
/track-review-comments respond [id]   → Add response
        ↓
/track-review-comments resolve [id]   → Mark done
        ↓
/track-review-comments status         → Check progress
        ↓
/track-review-comments export         → Generate letter
```

## Begin

Based on `$ARGUMENTS`:

1. `add [file/text]`: Parse comments, create entries
2. `status`: Show current progress
3. `respond [id] "[text]"`: Add response to comment
4. `resolve [id]`: Mark as resolved
5. `categorize`: Auto-assign categories
6. `export`: Generate response letter
7. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 Review Comment Status
   Total Comments: {N}
   Resolved: {X}/{N} ({percent}%)
   Pending: {Y}

🔴 IMMEDIATE ACTIONS:
   {List pending high-priority comments}
   1. [Comment ID]: [Brief description]
   2. [Comment ID]: [Brief description]

🛠️ RECOMMENDED COMMANDS:

   [To address comments:]
   /track-review-comments respond [id] "[response]"
   /track-review-comments resolve [id]

   [After making changes:]
   /update-paper-state       → Sync documentation
   /paper-pipeline quick     → Verify consistency

   [When all resolved:]
   /track-review-comments export → Generate response letter
   /paper-pipeline pre-submit MS → Final check

📋 COMMENT CATEGORIES:
   🔴 Critical: {count}  → Address first
   🟡 Important: {count} → Address next
   🟢 Minor: {count}     → Address last

💡 TIP: Run /track-review-comments status regularly to track progress
```
