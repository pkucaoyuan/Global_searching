# Update Paper State - Synchronize Documentation After Changes

You are a paper state maintenance agent. Your task is to update the documentation ecosystem after paper modifications.

## ⚠️ MANDATORY: Unified Protocol

### Step 0: Read Shared Config

```
Read .claude/commands/_shared/unified_protocol.md
```

### Step 1: Read Current Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before updating ANY state docs, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name (e.g., `ms_journal`)
2. Read ALL current state files to understand the baseline:

```
Read docs/paper_state/{resolved}/overview.md       # Current status
Read docs/paper_state/{resolved}/symbols.md        # Existing symbols
Read docs/paper_state/{resolved}/results.md        # Existing theorems
Read docs/paper_state/{resolved}/changelog.md      # Recent changes
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- overview.md: [paper status, page count]
- symbols.md: [number of defined symbols]
- results.md: [number of theorems/propositions]
- changelog.md: [date of last entry]
```

**If you skip this step, you may overwrite existing state doc content with incomplete information.**

---

## Why This Matters

After making changes:
```
Edit introduction.tex → symbols.md out of sync
Add new theorem → results.md missing entry
Change figure → figures_tables.md not updated
→ Next session: Stale documentation causes errors
```

With regular updates:
```
/update-paper-state → All docs synchronized
→ Next session: Documentation is source of truth
```

## Arguments

- `$ARGUMENTS` - Paper name or specific doc to update:
  - `[paper_name]` - Update all docs for paper
  - `[paper_name] symbols` - Update only symbols.md
  - `[paper_name] results` - Update only results.md
  - `[paper_name] figures_tables` - Update only figures_tables.md
  - `[paper_name] changelog "[message]"` - Add changelog entry

## Update Procedures

### Full Update (default)

When no specific doc is specified, update ALL documents:

1. **symbols.md**
   - Grep all `$...$` and `\[...\]` in .tex files
   - Compare with existing registry
   - Flag new symbols, removed symbols, changed definitions

2. **results.md**
   - Find all `\begin{theorem}`, `\begin{lemma}`, `\begin{proposition}`
   - Compare with existing registry
   - Flag new results, removed results, changed statements

3. **figures_tables.md**
   - Find all `\includegraphics`, `\begin{figure}`, `\begin{table}`, `\begin{tabular}`
   - Compare with existing registry
   - Flag new/removed figures and tables, changed captions

4. **cross_references.md**
   - Scan all `\ref{}`, `\eqref{}`, `\label{}` in .tex files
   - Flag broken references, orphaned labels

5. **dependencies.md**
   - Update assumption→theorem chains if results changed

6. **abbreviations.md**
   - Scan for new acronyms

7. **insights.md**
   - Update if abstract/conclusion changed

8. **review_responses.md**
   - Update status of open comments if addressed

9. **consistency_log.md**
   - Add entry if consistency checks were run

10. **overview.md**
   - Update section map if structure changed
   - Update status indicators
   - Update active issues

11. **changelog.md**
    - Auto-generate entry from git diff
    - Add timestamp and session ID

### Specific Updates

#### `update [paper] symbols`

```bash
# Extract all symbols from .tex files
grep -oP '\$[^$]+\$' sections/*.tex | sort | uniq

# Compare with symbols.md
# Report: NEW, REMOVED, UNCHANGED
```

#### `update [paper] results`

```bash
# Extract all results
grep -n '\\begin{theorem}\|\\begin{lemma}\|\\begin{proposition}' sections/*.tex

# Compare with results.md
# Report: NEW, REMOVED, CHANGED
```

#### `update [paper] changelog "[message]"`

Add a new changelog entry:

```markdown
## [YYYY-MM-DD] - [Auto-generated Session ID]

### Changed
- [From git diff: list modified files with brief description]

### Notes
- [User message]

### Verification
- [ ] symbols.md synced
- [ ] results.md synced
- [ ] figures_tables.md synced
```

## Diff Detection

### Symbol Diff

```markdown
## Symbol Changes Detected

### New Symbols (not in registry)
| Symbol | First Seen | Context |
|--------|------------|---------|
| τ | delays.tex:L15 | "delay parameter τ" |

### Removed Symbols (in registry but not in paper)
| Symbol | Was In | Action Needed |
|--------|--------|---------------|
| ε | model.tex | Remove from registry |

### Changed Definitions
| Symbol | Old Definition | New Definition | Location |
|--------|----------------|----------------|----------|
| π | audit prob | adaptive audit prob | model.tex:L20 |
```

### Result Diff

```markdown
## Result Changes Detected

### New Results
| Result | Label | Section | Statement Preview |
|--------|-------|---------|-------------------|
| Theorem 8.1 | thm:delay | delays | "Under delayed feedback..." |

### Changed Statements
| Result | Label | Change |
|--------|-------|--------|
| Thm 5.2 | thm:neyman | Added "under mild conditions" |

### Removed Results
| Result | Was In | Action |
|--------|--------|--------|
| Prop 6.3 | analysis | Merged into Thm 6.1 |
```

## Conflict Resolution

When changes conflict with documentation:

```markdown
## Conflicts Detected

### Symbol Conflict
- **Symbol**: b
- **In paper**: b(t) for best arm (algorithm.tex:L30)
- **In registry**: b_k(x) for bias (symbols.md:L15)
- **Resolution needed**: Choose one notation

### Result Conflict
- **Label**: thm:cost
- **In paper**: "Cost scales as O(K/Δ²)"
- **In registry**: "Cost scales as O(log K/Δ²)"
- **Resolution needed**: Verify correct statement
```

## Automation

### Git Hook Integration

```bash
# .git/hooks/post-commit
# Reminder to update paper state
echo "Reminder: Run /update-paper-state [paper_name] to sync documentation"
```

### Session Start

At start of any paper editing session:
1. Read overview.md for current state
2. Check changelog.md for recent changes
3. Verify symbols.md against current usage

## Output Format

```markdown
# Paper State Update Report

**Paper**: [name]
**Date**: [date]
**Docs Updated**: [count]

---

## Changes Summary

| Document | Status | Changes |
|----------|--------|---------|
| symbols.md | ✅ Updated | +2 new, -1 removed |
| results.md | ✅ Updated | +1 new theorem |
| figures_tables.md | ⚠️ Needs review | 1 caption mismatch |
| overview.md | ✅ Updated | Status updated |
| changelog.md | ✅ Entry added | - |

---

## Actions Required

1. [ ] Review figure 3 caption mismatch
2. [ ] Verify new symbol τ definition

---

## Verification Checklist

- [x] All symbols in paper are in registry
- [x] All results in paper are in registry
- [ ] All figures/tables in paper are in registry (1 issue)
- [x] Changelog entry added
```

## Integration

```
[Make paper edits]
        ↓
/update-paper-state [paper]   → Sync all docs
        ↓
[Review update report]
        ↓
[Resolve any conflicts]
        ↓
[Continue editing with accurate docs]
```

## Begin

1. Parse `$ARGUMENTS` for paper name and update scope
2. Read current state docs
3. Read current .tex files
4. Compute diffs
5. Update affected documents
6. Generate update report
7. List any actions required
8. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 State Update Complete
   Paper: {paper_name}
   Files Updated: {N}
   Conflicts Found: {M}

✅ DOCUMENTS SYNCED:
   ├── symbols.md      → {X} symbols ({+A new, -B removed})
   ├── results.md      → {Y} theorems
   ├── changelog.md    → {Z} entries added
   └── [others...]

{If conflicts found:}
🔴 CONFLICTS REQUIRING RESOLUTION:
   1. [File]: [Conflict description]
   2. [File]: [Conflict description]

{If no conflicts:}
✅ All documents in sync. No conflicts.

🛠️ RECOMMENDED COMMANDS:

   [If conflicts:]
   [Manually resolve conflicts in listed files]
   /update-paper-state       → Re-sync after resolution

   [When synced:]
   /paper-pipeline quick     → Verify consistency
   /session-start {paper}    → Next session context

💡 TIP: Run this command before ending any editing session
```
