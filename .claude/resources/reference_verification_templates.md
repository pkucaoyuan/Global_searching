# Reference Verification Templates

## Verification Report Template

Save to `docs/progress/paper_writing/reference_verification_report.md`:

```markdown
# Reference Verification Report

**Date**: [YYYY-MM-DD]
**Total References**: [count] (in references.bib)
**Verified**: [X]/[Y]
**Unverifiable**: [Z]

---

## Verification Summary

| Status | Count | Description |
|--------|-------|-------------|
| **Verified** | X | arXiv/DOI confirmed, metadata correct |
| **Fixed** | Y | Errors found and corrected |
| **Flagged** | Z | Cannot verify (list keys) |
| **Note** | W | Special cases |

---

## Issues Found and Corrections Made

### [N]. [Issue Type] - `bibtex_key`

**Status**: FIXED / FLAGGED / CANNOT VERIFY

| Field | Old (Incorrect) | New (Correct) |
|-------|-----------------|---------------|
| [field] | [old value] | [new value] |

**Evidence**: [How verified - arXiv link, DOI, search results]

---

## Verified References (All Correct)

### [Category Name] ([count] refs)

| Key | arXiv/Venue | Status |
|-----|-------------|--------|
| key1 | arXiv:XXXX.XXXXX | Verified |
| key2 | DOI:10.XXXX/XXXX | Verified |

---

## Verification Methods Used

1. **arXiv API**: `curl https://arxiv.org/abs/{ID}` - Verified X papers
2. **DOI Resolution**: `curl -I https://doi.org/{DOI}` - Verified Y DOIs
3. **OpenAlex API**: Title/author fuzzy matching
4. **Web Search**: Conference proceedings

---

## Citation Description Issues

### Accurate Descriptions ([count])
All descriptions match source abstracts.

### Needs Revision ([count])

| Key | Issue | Suggested Fix |
|-----|-------|---------------|
| `key1` | Mischaracterizes scope | [new description] |

### Inaccurate ([count])

| Key | Your Description | Actual Focus | Status |
|-----|------------------|--------------|--------|
| `key2` | "focuses on X" | "focuses on Y" | FIXED / FLAGGED |

---

## Recommendations

1. [Action items for unverifiable references]
2. [Action items for description corrections]
3. [Monitoring notes for special cases]

---

*Report generated: [YYYY-MM-DD]*
```

---

## Corrections Log Template

Append to `docs/progress/paper_writing/corrections_log.md`:

```markdown
## [YYYY-MM-DD] - Reference Verification Corrections

**Files Modified**:
- `paper/references.bib`

**Changes**:
| BibTeX Key | Field | Old | New | Reason |
|------------|-------|-----|-----|--------|
| key1 | year | 2023 | 2024 | ICLR 2024 proceedings |
| key2 | title | ... | ... | Official title from publisher |

**Verification Method**: OpenAlex API + manual confirmation
```

---

## Citation Description Output Format

```markdown
## Citation Description Issues

### ⚠️ NEEDS_REVISION - `bibtex_key`

**Your description**:
> "introduced tool-use evaluation but focused on general optimization tools"

**Actual abstract**:
> "evaluates LLM workflow execution under simulated realistic conditions..."

**Issue**: Mischaracterizes the paper's focus.

**Suggested fix**:
> "evaluates LLM workflow execution under probabilistic tool failures..."
```

---

## Common Mischaracterization Patterns

```latex
% Pattern 1: Scope Mischaracterization
% ❌ "X focuses on image classification"
% ✅ "X focuses on object detection with classification as auxiliary"

% Pattern 2: Contribution Inflation
% ❌ "X achieved state-of-the-art results"
% ✅ "X achieved competitive results on subset Y"

% Pattern 3: Method Confusion
% ❌ "X uses reinforcement learning"
% ✅ "X uses supervised learning with RL-inspired objective"

% Pattern 4: Outdated Description (preprint → camera-ready)
% ❌ Description based on v1 arXiv
% ✅ Updated to match accepted version
```

---

## Summary Report Template

```markdown
# Reference Verification Complete

## Metadata Verification

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Verified | X | X% |
| 🔧 Fixed | Y | Y% |
| ⚠️ Needs Review | Z | Z% |
| ❌ Not Found | W | W% |
| 📚 Manual Check | V | V% |
| 🚨 Retracted | R | R% |

## Description Verification (Summary Check)

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Accurate | A | A% |
| ⚠️ Needs Revision | B | B% |
| ❌ Inaccurate | C | C% |
| 🔍 Cannot Verify | D | D% |

## Files Modified

- [ ] `paper/references.bib` - X corrections applied
- [ ] `paper/sections/*.tex` - Y description fixes applied
- [ ] `docs/.../reference_verification_report.md` - Updated
- [ ] `docs/.../corrections_log.md` - Logged changes

## Critical Actions Required

1. 🚨 **URGENT**: [Retracted papers to remove]
2. ❌ **HIGH**: [Unverifiable references]
3. ❌ **HIGH**: [Inaccurate descriptions to fix]
4. ⚠️ **MEDIUM**: [Items needing manual review]

## Pre-Submission Checklist

- [ ] All references VERIFIED or MANUAL_CHECK
- [ ] No RETRACTED papers
- [ ] DOI coverage > 80%
- [ ] All required fields present
- [ ] All citation descriptions accurate
- [ ] Documentation updated

**Status**: READY FOR SUBMISSION / NEEDS FIXES
```

---

## Next Steps Footer

```
═══════════════════════════════════════════════════════════════════
                        NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: Reference Verification
   Issues Found: {N}
   Verified: {X}/{Y}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   1. Fix BibTeX entries flagged above
   2. Correct citation descriptions marked INACCURATE
   3. Re-run /proofread-references to verify

   {If all verified:}
   ✅ References verified. Ready for submission checks.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found:]
   /proofread-references       → Re-verify after fixes

   [When references verified:]
   /check-paper-consistency    → Verify symbols/notation
   /polish-paper               → Final language polish

   [Before submission:]
   /paper-pipeline pre-submit MS → Final checklist

💡 TIP: Update docs/paper_state/[paper]/cross_references.md with verified refs
```
