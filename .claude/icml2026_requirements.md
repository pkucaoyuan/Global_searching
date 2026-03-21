# ICML 2026 Paper Submission Requirements

> **Source**: [ICML 2026 Author Instructions](https://icml.cc/Conferences/2026/AuthorInstructions) | [Call for Papers](https://icml.cc/Conferences/2026/CallForPapers)

## Conference Information

- **Location**: Seoul, South Korea
- **Dates**: July 7-12, 2026
- **Format**: In-person only (no virtual/hybrid support)

---

## Critical Deadlines

| Milestone | Deadline | Notes |
|-----------|----------|-------|
| Submission site opens | January 8, 2026 | Create OpenReview account |
| **Abstract deadline** | **January 23, 2026 AoE** | January 24, 12:00 UTC |
| **Full paper deadline** | **January 28, 2026 AoE** | January 29, 12:00 UTC |

> **Warning**: "Abstract and paper submission deadlines are strict. In no circumstances will extensions be given."

---

## Page Limits (Strict)

| Section | Limit | Notes |
|---------|-------|-------|
| Main body | **8 pages max** | Exceeding = automatic rejection |
| References | Unlimited | After main body |
| Impact statement | Unlimited | Separate section |
| Appendix | Unlimited | After references |
| Camera-ready | 9 pages | +1 page for accepted papers |

**File Size Limits**:
- Submission PDF: **50MB max**
- Camera-ready PDF: **20MB max**

---

## Formatting Requirements

### Mandatory

- Use official ICML 2026 LaTeX style files (`icml2026.zip`)
- **Single PDF file** containing all content (no separate supplementary files)
- LaTeX only ("There is no support for any typesetting software other than LaTeX")
- Follow `example_paper.pdf` format exactly

### Automatic Rejection Triggers

- Main body > 8 pages
- Non-anonymized submission
- Incorrect formatting/style
- LLM listed as author
- Prompt injection in submission
- Dual/concurrent submissions with substantial similarity
- Author list changes after abstract deadline (without exceptional approval)
- Failure to meet reciprocal reviewer requirement

---

## Double-Blind Anonymization

### Must Remove

- Author names and affiliations
- Acknowledgements
- Grant numbers
- Links to public code repositories
- Personal website URLs
- Any identifying information

### Self-Citation Rules

- Refer to own prior work in **third person**
- Example: "Smith et al. (2024) showed..." not "We previously showed..."
- Cite overlapping prior work while preserving anonymity
- Explain differences from prior work

### Preprint Policy

- Authors **may** post preprint versions and give talks
- Must **not** advertise work "as an ICML submission at any time during the review period"

### Code Anonymization

- Remove author names from code
- Remove licenses with author information
- Use anonymous GitHub repository on **frozen branch**
- Branch must not be modified after submission deadline

---

## Supplementary Materials

> **Important**: "There will be no separate deadline for the submission of supplementary material."

All supplementary content must be included **within your submission PDF file**:

| Type | Where to Include |
|------|------------------|
| Proofs/Derivations | Appendix (after references) |
| Additional experiments | Appendix |
| Code | Appendix or anonymous GitHub link |
| Detailed algorithms | Appendix |

---

## LLM/AI Usage Policy

| Allowed | Prohibited |
|---------|------------|
| LLM assistance in writing | LLM as author |
| LLM for code/research | Prompt injection |
| AI-generated figures (disclosed) | Undisclosed AI content |

**Key Rule**: Authors bear full responsibility for all content, including AI-generated material.

---

## Reciprocal Reviewing

| Author Submissions | Requirement |
|--------------------|-------------|
| 1-3 papers | No reviewing required |
| 4+ papers | Must serve as reviewer |
| Reviewer role | Can cover up to 2 submissions |

**Penalty**: Failure to meet requirements may result in desk rejection.

---

## Camera-Ready Requirements

For accepted papers:

1. **Extra Page**: 9 pages allowed for main body (+1 page)
2. **Size Limit**: 20MB max (reduced from 50MB)
3. **De-anonymization**: Add author info, acknowledgements
4. **Lay Summary**: Short summary for OpenReview (required)
5. **Content Changes**: "Authors may choose to (but are not required to) make changes based on reviewer feedback, provided core content remains unchanged"

---

## Style File Checklist

```latex
% For submission (anonymous)
\usepackage{icml2026}

% For camera-ready (accepted)
\usepackage[accepted]{icml2026}

% Required packages
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
```

---

## Quick Checklist Before Submission

### Content
- [ ] Main body ≤ 8 pages
- [ ] **Impact Statement included** (before References, does not count toward page limit)
- [ ] PDF size ≤ 50MB
- [ ] Single PDF file (no separate supplementary)
- [ ] References after main body
- [ ] Appendix after references

### Anonymization
- [ ] All author info removed
- [ ] Self-citations in third person
- [ ] No acknowledgements/grants
- [ ] No public code links (or use anonymous frozen branch)

### Formatting
- [ ] Uses official ICML 2026 style (`icml2026.zip`)
- [ ] LaTeX compiled without errors
- [ ] Follows `example_paper.pdf` format

### Pre-submission
- [ ] Abstract submitted by January 23, 2026 AoE
- [ ] Full paper submitted by January 28, 2026 AoE
- [ ] OpenReview account created

---

## Key Links

- Author Instructions: [icml.cc/Conferences/2026/AuthorInstructions](https://icml.cc/Conferences/2026/AuthorInstructions)
- Call for Papers: [icml.cc/Conferences/2026/CallForPapers](https://icml.cc/Conferences/2026/CallForPapers)
- Peer Review FAQ: [icml.cc/Conferences/2026/PeerReviewFAQ](https://icml.cc/Conferences/2026/PeerReviewFAQ)
- Style Files: [icml.cc/Conferences/2026/StyleFiles](https://icml.cc/Conferences/2026/StyleFiles)

---

*Last Updated: 2026-01-27 (from official ICML website)*
