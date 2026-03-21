# Progress Report Templates

## Progress Report Template

**File naming**: `docs/progress/YYYY_MM_DD_{topic}.md`

```markdown
# [Feature/Topic Name] - YYYY-MM-DD

## Status
✅ Completed / 🚧 In Progress / ❌ Blocked

## Summary
[1-2 sentences summarizing the main outcome]

## Completed Work

### Main Features
- **Feature 1**: [description]
- **Feature 2**: [description]

### Improvements
- [Improvement 1]
- [Improvement 2]

### Bug Fixes
- [Fix 1]
- [Fix 2]

## Code Changes

| Type | Count |
|------|-------|
| New files | X |
| Modified files | Y |
| Deleted files | Z |

### Key File Changes

| File | Change Type | Description |
|------|------------|-------------|
| `path/to/file.py` | New/Modified/Deleted | [description] |

## Related Commits

```
[commit hash] [commit message]
```

## Issues & Solutions

### Issue 1: [Title]
**Symptom**: [description]
**Cause**: [analysis]
**Solution**: [fix]

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual verification

## Next Steps

- [ ] [TODO 1]
- [ ] [TODO 2]
- [ ] [TODO 3]

## Related Docs

- [Related guide](../guides/xxx.md)

---

**Author**: [auto-generated]
**Review**: Pending
```

---

## Weekly/Monthly Summary Template

```markdown
# Progress Summary - YYYY-MM-DD to YYYY-MM-DD

## Overview

| Metric | Value |
|--------|-------|
| Features completed | X |
| Bugs fixed | Y |
| Lines added | +Z |
| Commits | N |

## Main Achievements

### 1. [Feature/Module 1]
- Progress report: [link]
- Status: ✅/🚧
- Summary: ...

### 2. [Feature/Module 2]
- Progress report: [link]
- Status: ✅/🚧
- Summary: ...

## Key Milestones

- [x] Milestone 1
- [x] Milestone 2
- [ ] Milestone 3 (in progress)

## Issues & Risks

| Issue | Impact | Status | Owner |
|-------|--------|--------|-------|
| [Issue 1] | Medium | Resolved | - |
| [Issue 2] | High | Tracking | - |

## Next Week/Month Plan

- [ ] Plan 1
- [ ] Plan 2
- [ ] Plan 3

## Related Reports

- [Report 1](./YYYY_MM_DD_xxx.md)
- [Report 2](./YYYY_MM_DD_xxx.md)
```

---

## CLAUDE.md Update Patterns

After generating a progress report, **auto-update CLAUDE.md** sections:

### A. `src/` changes → Update Directory Structure + Status

```markdown
## Directory Structure
src/
├── new_module/              # 🆕 New module
│   └── new_file.py          # Description

## Status - Current Phase
*   **New Components**:
    *   `src/module/new_file.py`: [description]
*   **Updated Files**:
    *   `src/module/existing.py`: [change] (+X lines)
```

### B. `scripts/` changes → Update Scripts section

```markdown
*   **Scripts**:
    *   `scripts/experiments/xxx.py`: [purpose]
    *   `scripts/analysis/yyy.py`: [purpose]
```

### C. `docs/` changes → Update Documentation section

```markdown
*   `docs/progress/YYYY_MM_DD_xxx.md` - [topic]
*   **Validation Reports**:
    *   `docs/experiments/new_report.md` (details)
```

### D. `experiments/` changes → Update Experiments section

```markdown
*   **Experiments**:
    *   `experiments/new_exp.py`: [description]
    *   `experiments/theory_validation/exp_xxx.py`: [theorem validation]
```

### E. `paper/` changes → Update Paper section

```markdown
#### Paper Structure
paper/journal/main.tex
├── sections/new_section.tex   ← 🆕 New section
```

### F. `tests/` changes → Update Tests section

```markdown
*   **Tests**: X unit tests (all passing)
    *   `tests/module/test_xxx.py`: [description]
```

### G. `.claude/` changes → Update Standards References

```markdown
## Standards References
| [`new_guide.md`](.claude/standards/new_guide.md) | Description |
```

### Complete Update Example

**git diff output:**
```
A  src/bai_judge/audit/new_allocator.py
M  src/bai_judge/algorithms/lucb_joint.py
A  scripts/experiments/run_new_exp.py
A  docs/progress/2026_01_31_feature_update.md
```

**Auto Edit operations:**
1. Update Directory Structure (if new directory)
2. Update current Phase Status (new/modified files, scripts, experiments)
3. Update Paper section (if paper/ changed)

### Verification Checklist

After update, confirm:
- [ ] Directory Structure updated (new dirs/files)
- [ ] Status section's current Phase updated
- [ ] New code files recorded
- [ ] New scripts recorded
- [ ] New docs recorded
- [ ] New experiments recorded
- [ ] Progress report link added
- [ ] Dates correct

---

## Sync Command Output Formats

**Code module:**
```markdown
#### Code Changes (YYYY-MM-DD)
*   **New**: `src/module/new_feature.py`: [description]
*   **Modified**: `src/module/existing.py`: [change] (+X/-Y lines)
*   **Deleted**: `src/module/deprecated.py`
```

**Scripts module:**
```markdown
#### Scripts (YYYY-MM-DD)
| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/new_script.py` | [description] | 🆕 |
| `scripts/updated.sh` | [description] | ✏️ Updated |
```

**Documents module:**
```markdown
#### Documentation (YYYY-MM-DD)
*   **Progress**: `docs/progress/YYYY_MM_DD_xxx.md`
*   **Technical**: `docs/experiments/xxx.md`
*   **Guides**: `docs/guides/xxx.md`
```

---

## Status Markers

| Marker | Meaning | Use |
|--------|---------|-----|
| ✅ Completed | Feature done, tests pass | Ready to publish/deploy |
| 🚧 In Progress | Under development | Still being worked on |
| ❌ Blocked | Blocked, needs help | Waiting on dependency/decision |
| ⏸️ Paused | Temporarily shelved | Priority changed |

## Content Requirements

**Required**: Status marker, summary (1-2 sentences), completed items, code change stats, next steps

**Recommended**: Key file changes table, related commits, issues & solutions, test status

**Optional**: Architecture diagrams, performance metrics, related doc links
