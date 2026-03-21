# Proofread References - Academic Citation Verification

You are a reference verification agent. Your task is to verify the authenticity and correctness of academic citations in BibTeX files, and maintain verification documentation.

## Arguments

- `$ARGUMENTS` - Optional: path to BibTeX file (default: auto-detect `paper/references.bib` or `main.bib`)

## ⚠️ MANDATORY: Follow Unified Protocol

**STOP.** Before verifying references, follow `.claude/commands/_shared/unified_protocol.md` Steps 0A–2.5 (resolve paper name, read state files, write checkpoint). If this command results in any `.tex` edits, you MUST also follow Step 4 (Universal Post-Edit Rule).

### Read Paper State Files FIRST

Before verifying references, you MUST load the paper's state docs to understand which citations are critical and which theorems depend on specific references.

1. Resolve paper name → run `ls docs/paper_state/` to find the actual directory
2. Read required state files:

```
Read docs/paper_state/{resolved}/cross_references.md  # Which references are cited where
Read docs/paper_state/{resolved}/results.md           # Which theorems cite which papers
Read docs/paper_state/{resolved}/dependencies.md      # Reference dependency chains
```

3. Write a **verification checkpoint**:
```
State doc context loaded:
- cross_references.md: [number of unique citations, key references]
- results.md: [theorems that depend on specific cited results]
- dependencies.md: [critical reference chains]
```

**If you skip this step, you may miss critical citations that theorems depend on.**

## Setup

**Step 0: Locate Files**

1. **BibTeX File**:
   - If `$ARGUMENTS` is provided, use that as the BibTeX file path
   - Otherwise, search for: `paper/references.bib` → `references.bib` → `main.bib` → `*.bib`

2. **Verification Guide** (must read before starting):
   - `.claude/reference_verification_guide.md` - Complete verification methodology

3. **Existing Report** (read if exists):
   - `docs/progress/paper_writing/reference_verification_report.md` - Previous verification results

Report discovered files before proceeding.

**Step 1: Check Dependencies**

```bash
pip install pyalex bibtexparser thefuzz requests pandas tabulate --quiet
```

## Workflow

### PHASE 1: Parse and Categorize Entries

**Step 1.1: Parse BibTeX File**

Read and parse the BibTeX file. Count entries by type:
- `@article` (journal papers)
- `@inproceedings` (conference papers)
- `@misc` (preprints, arXiv)
- `@book` (textbooks)
- `@techreport` (technical reports)

**Step 1.2: Compare with Previous Report**

If `reference_verification_report.md` exists:
1. Identify NEW entries (not in previous report)
2. Identify REMOVED entries (in report but not in BibTeX)
3. Focus verification on NEW entries
4. Preserve previous VERIFIED status for unchanged entries

**Step 1.3: Prioritize Entries**

| Priority | Criteria | Action |
|----------|----------|--------|
| HIGH | Core contributions, theory-dependent citations | Must verify 100% |
| MEDIUM | Method comparisons, experimental baselines | Should verify |
| LOW | Textbooks, surveys, gray literature | Manual check only |

### PHASE 2: Automated Verification via OpenAlex

**Step 2.1: Configure API**

```python
from pyalex import Works, config
config.email = "your@email.com"  # Required for rate limits
```

**Step 2.2: Verify Each Entry**

For each entry, try verification in this order:
1. **DOI lookup** (most reliable) - if `doi` field exists
2. **arXiv ID lookup** - if `eprint` field exists or arXiv pattern in journal
3. **Title search** - fuzzy match against OpenAlex database

**Step 2.3: Match Quality Assessment**

| Score | Status | Meaning |
|-------|--------|---------|
| ≥95% | VERIFIED | Exact match found |
| 85-94% | VERIFIED | High confidence match |
| 70-84% | NEEDS_REVIEW | Partial match, metadata differences |
| <70% | SUSPECT | Low confidence, verify manually |
| 0% | NOT_FOUND | No match in database |

### PHASE 3: Format Validation

**Step 3.1: Required Fields Check**

| Entry Type | Required Fields |
|------------|-----------------|
| @article | title, author, journal, year, volume |
| @inproceedings | title, author, booktitle, year |
| @misc (arXiv) | title, author, year, eprint, archivePrefix, primaryClass |
| @book | title, author, publisher, year |
| @techreport | title, author, institution, year |

**Step 3.2: Special Character Escaping**

Check for unescaped LaTeX special characters:
- `&` → `\&`
- `%` → `\%`
- `_` → `\_`
- `$` → `\$`

**Step 3.3: Entry Type Correctness**

Common mistakes to flag:
- NeurIPS/ICML/ICLR papers as `@article` (should be `@inproceedings`)
- arXiv preprints as `@article` (should be `@misc`)
- JMLR papers as `@inproceedings` (should be `@article`)

### PHASE 4: Retraction Check

For all VERIFIED entries with DOI, check Crossref for retraction status:

```python
def check_retraction(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.ok:
        data = response.json()["message"]
        if "update-to" in data:
            for update in data["update-to"]:
                if update.get("type") == "retraction":
                    return True, update.get("DOI")
    return False, None
```

### PHASE 5: Citation Description Verification (Summary Check)

**Purpose**: Verify that your descriptions of cited papers accurately represent their actual contributions.

**Step 5.1: Extract Citation Contexts**

For each `\citep{key}` or `\citet{key}` in `.tex` files, extract:
- The surrounding sentence(s) describing the cited work
- The specific claim being made about the paper

Example extraction:
```
Key: pilotbench2026
Context: "PILOT-Bench~\citep{pilotbench2026} introduced tool-use evaluation but focused on general optimization tools"
Claim: "introduced tool-use evaluation", "focused on general optimization tools"
```

**Step 5.2: Fetch Source Abstracts**

For each citation, retrieve the abstract from:
1. **OpenAlex** - `work['abstract_inverted_index']` or `work['abstract']`
2. **arXiv API** - `http://export.arxiv.org/api/query?id_list={arxiv_id}`
3. **Semantic Scholar** - `https://api.semanticscholar.org/graph/v1/paper/{id}?fields=abstract,tldr`
4. **OpenReview** - For ICLR/NeurIPS submissions (web fetch)

**Step 5.3: Compare Description vs Source**

For each citation, verify:

| Check | Question | Pass | Fail |
|-------|----------|------|------|
| Topic | Does description match paper's actual focus? | ✅ | ❌ Mischaracterization |
| Contribution | Are contributions accurately stated? | ✅ | ❌ Over/understated |
| Methodology | Are methods correctly attributed? | ✅ | ❌ Wrong method |
| No Fabrication | Are all claims traceable to source? | ✅ | ❌ Invented claims |

**Step 5.4: Classification**

| Status | Meaning | Action |
|--------|---------|--------|
| ACCURATE | Description matches source | None |
| NEEDS_REVISION | Minor inaccuracies | Suggest fix |
| INACCURATE | Major misrepresentation | Flag for correction |
| CANNOT_VERIFY | Abstract unavailable | Manual check |

**Step 5.5: Common Mischaracterization Patterns**

Watch for these common errors:

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

**Step 5.6: Output Description Issues**

Add to verification report:

```markdown
## Citation Description Issues

### ⚠️ NEEDS_REVISION - `pilotbench2026`

**Your description**:
> "introduced tool-use evaluation but focused on general optimization tools"

**Actual abstract**:
> "evaluates LLM workflow execution under simulated realistic conditions with probabilistic tool failures and variable instruction quality"

**Issue**: Mischaracterizes the paper's focus. PILOT-Bench is about probabilistic tool failures in workflow execution, not "general optimization tools".

**Suggested fix**:
> "evaluates LLM workflow execution under probabilistic tool failures and varying instruction quality"

---
```

### PHASE 6: Apply Fixes

For any issues found, **directly edit the files**:

**BibTeX Fixes** (`references.bib`):
1. **Wrong entry type**: Change `@article` to `@inproceedings` etc.
2. **Missing fields**: Add required fields (pages, volume, etc.)
3. **Year errors**: Correct publication year
4. **Title errors**: Fix typos or update to official title
5. **Special characters**: Escape LaTeX characters

**LaTeX Fixes** (`.tex` files):
6. **Inaccurate descriptions**: Revise citation context to match actual contribution
7. **Scope corrections**: Fix mischaracterizations of paper focus
8. **Outdated claims**: Update based on camera-ready version

**Record all changes** for the documentation update.

### PHASE 7: Update Documentation

**Step 7.1: Update Verification Report**

Update `docs/progress/paper_writing/reference_verification_report.md` with:

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

**Step 7.2: Log Corrections**

If any BibTeX corrections were made, append to `docs/progress/paper_writing/corrections_log.md`:

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

### PHASE 8: Generate Summary Report

Output a summary to the user:

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
- [ ] `docs/progress/paper_writing/reference_verification_report.md` - Updated
- [ ] `docs/progress/paper_writing/corrections_log.md` - Logged changes

## Critical Actions Required

1. 🚨 **URGENT**: [List retracted papers to remove]
2. ❌ **HIGH**: [List unverifiable references]
3. ❌ **HIGH**: [List inaccurate descriptions to fix]
4. ⚠️ **MEDIUM**: [List items needing manual review]

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

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

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

---

## Constraints

- **API Rate Limits**: Add 0.1s delay between requests
- **Email Required**: OpenAlex requires email for authenticated requests
- **Fallback**: If API fails, mark as NEEDS_REVIEW (not NOT_FOUND)
- **Books/Tech Reports**: Skip API verification, mark as MANUAL_CHECK
- **Timeout**: 30 seconds per API request max
- **Preserve Evidence**: Always document how each reference was verified

## Quick Fixes Reference

| Issue | Fix |
|-------|-----|
| Wrong entry type for NeurIPS | Change `@article` to `@inproceedings`, `journal` to `booktitle` |
| arXiv as article | Change to `@misc`, add `eprint`, `archivePrefix`, `primaryClass` |
| Missing pages | Look up on publisher website or OpenAlex result |
| Author format inconsistent | Standardize to `Last, First and Last2, First2` |
| Title capitalization | Use `{Title Case}` or `{{Acronym}}` to preserve |

## Verification Sources

| Source | Best For | Limitations |
|--------|----------|-------------|
| OpenAlex | Most papers (2.5B+ records) | 1-2 week indexing delay |
| Crossref | DOI validation, retraction check | Limited arXiv coverage |
| arXiv API | Preprint verification | arXiv papers only |
| Google Scholar | Manual fallback | No API, requires browser |

## Documentation Paths

| Document | Path | Purpose |
|----------|------|---------|
| Verification Guide | `.claude/reference_verification_guide.md` | Methodology reference |
| Verification Report | `docs/progress/paper_writing/reference_verification_report.md` | Main verification results |
| Corrections Log | `docs/progress/paper_writing/corrections_log.md` | Change history |
| Proofread README | `docs/progress/paper_writing/proofread/README.md` | Overall progress tracking |

## Begin

1. Read `.claude/reference_verification_guide.md` for methodology
2. Locate BibTeX file (use `$ARGUMENTS` if provided)
3. Read existing `reference_verification_report.md` if available
4. Parse BibTeX and identify new/changed entries
5. Run automated metadata verification (PHASE 2-4)
6. Extract citation contexts from `.tex` files
7. Fetch source abstracts and verify descriptions (PHASE 5)
8. Apply fixes to BibTeX and LaTeX files (PHASE 6)
9. **Update `reference_verification_report.md`** with complete results
10. **Append to `corrections_log.md`** if changes were made
11. Generate summary report for user
