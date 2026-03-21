# Literature Acquire - Download ArXiv Sources & Generate Structured Summaries

You are a literature acquisition agent that **directly executes** the full pipeline: identify missing papers, download arXiv source files, read them, and write structured SUMMARY_*.md files following the project's established template.

## Arguments

- `$ARGUMENTS` — Action and targets. Examples:
  - `scan` — Audit references.bib vs existing summaries, show coverage gaps
  - `download 2010.11629` — Download single arXiv source
  - `summarize bamas2020` — Generate summary for an already-downloaded paper
  - `batch yds1995 bansal2007 wierman2012` — Download + summarize multiple papers
  - `batch --from-bib --missing` — Process all bib entries that lack summaries
  - `status` — Show coverage statistics

## Core Principle: Direct Execution with Parallel Dispatch

**CRITICAL**: This skill EXECUTES downloads and writes summaries directly.

- For **1-2 papers**: Execute sequentially in the main context
- For **3+ papers**: Dispatch parallel Task subagents (one per paper or per batch of 2-3)
- Each subagent downloads source, reads it, and writes the summary file

## Directory Convention

```
literature/arxiv_sources/
├── {author_or_key}{year}/          # e.g., bamas2020/, wierman2012/
│   ├── *.tex                       # Downloaded LaTeX source files
│   ├── *.bbl, *.bib               # Bibliography files (if present)
│   ├── figures/                    # Figures (if present)
│   └── SUMMARY_{key}.md           # Structured summary (our output)
```

**Directory naming**: Use the BibTeX key as directory name. If key has underscores (e.g., `wei2020`), use as-is.

## Protocol

### Phase 1: Parse Arguments & Identify Targets

1. Parse `$ARGUMENTS` for action type and paper identifiers
2. Read `paper/energy_optimal_af/bib/references.bib` to get full paper metadata
3. Read `literature/README.md` for existing coverage
4. List `literature/arxiv_sources/*/SUMMARY_*.md` to find what already exists

### Phase 2: For Each Target Paper

#### Step 2a: Download ArXiv Source (if not already present)

For papers with arXiv IDs (extractable from bib entry's `eprint` or `url` field):

```bash
# Create directory
mkdir -p literature/arxiv_sources/{key}/

# Download source tarball
wget -q "https://arxiv.org/e-print/{arxiv_id}" -O /tmp/{key}_source.tar.gz

# Extract
cd literature/arxiv_sources/{key}/
tar xzf /tmp/{key}_source.tar.gz 2>/dev/null || \
  # Some are single .tex files, not tarballs
  cp /tmp/{key}_source.tar.gz ./{key}.tex

# Clean up
rm /tmp/{key}_source.tar.gz
```

**If no arXiv ID exists** (e.g., textbooks like `boyd2004`, `kleinrock1976`):
1. Use WebSearch + WebFetch to find the paper abstract/HTML version
2. Extract key information from the web page
3. Create the summary from web-sourced content
4. Mark in the summary: `**Source:** Web (no arXiv source available)`

#### Step 2b: Read & Analyze the Paper Source

1. **Find main .tex file**: Look for `\documentclass` or the largest .tex file
2. **Read key sections** (prioritize in order):
   - Abstract
   - Introduction / Problem statement
   - Model / Setup / Assumptions
   - Main results / Theorems / Propositions (with full statements)
   - Proof sketches (for key results)
   - Conclusion
3. **Extract**:
   - All theorem/lemma/proposition/corollary statements (verbatim LaTeX)
   - Key assumptions (numbered or named)
   - Proof techniques used
   - Concrete numerical results or bounds

#### Step 2c: Write SUMMARY_{key}.md

Follow **exactly** this template (matching existing summaries in the project):

```markdown
# {Authors} ({Year}) — "{Title}"

**Authors:** {Full author list}
**ArXiv:** {arXiv ID or "N/A"}
**BibTeX key:** `{key}`
**Source files:** `literature/arxiv_sources/{dirname}/`
**Venue:** {Journal/Conference, Year}

---

## 1. Setting & Model

### 1.1 Focus

{One paragraph: what problem does this paper study? What is the input/output?}

### 1.2 Key Assumptions

{Numbered list of formal assumptions. Use LaTeX math notation.}

### 1.3 Notation

{Table of key notation if the paper defines substantial notation}

| Symbol | Meaning |
|--------|---------|
| ... | ... |

---

## 2. Main Results

### Theorem/Result 1: {Name or number}

**Statement:** {Verbatim or faithful LaTeX rendering of the theorem}

**Significance:** {Why this matters, what it implies}

### Theorem/Result 2: {Name or number}

{Same structure. Include ALL main theorems, not just one.}

### Key Bounds / Formulas

{Any important formulas, competitive ratios, approximation guarantees}

---

## 3. Proof Techniques

### 3.1 {Technique Name} (for Result X)

{2-3 sentences on how the proof works. Mention specific tools: potential functions, LP relaxation, primal-dual, amortized analysis, coupling, etc.}

### 3.2 {Technique Name} (for Result Y)

{Same structure}

---

## 4. Connection to Our Paper

### 4.1 {Specific connection point}

{How does this result/technique relate to our energy-optimal A/F framework?}

### 4.2 {Another connection point}

{Another dimension of connection}

### 4.3 What We Borrow

- {Concrete item 1: technique, bound, model structure}
- {Concrete item 2}

### 4.4 What We Add / How We Differ

- {Key difference 1: our model has X which their model lacks}
- {Key difference 2: our setting introduces Y}

---

## 5. Key Quotes

> "{Important quote from the paper}" (Section X, p. Y)

> "{Another key quote}" (Section Z)

---

## 6. BibTeX

```bibtex
{The BibTeX entry for this paper}
```
```

**Template Rules**:
- Section 1-3: **Objective description** of the paper's content
- Section 4: **Subjective analysis** connecting to our energy-optimal A/F framework
- Section 5: Direct quotes that are particularly relevant (2-4 quotes)
- Section 6: BibTeX entry for easy citation
- Use `$$...$$` for display math, `$...$` for inline math
- Include ALL main theorems, not just summaries
- For proof techniques, name the classical technique (e.g., "Yao's minimax principle", "potential function method") so we can cross-reference

### Phase 3: Update Literature Index

After all summaries are written:

1. **Update `literature/README.md`**: Add new entries to the ArXiv Source Files table
2. **Report** to user:
   - Papers downloaded: N
   - Summaries written: N
   - Papers that failed (no arXiv source, download error): list

## Dispatch Protocol (for batch mode)

When processing 3+ papers, use parallel Task subagents:

```
// Group papers into batches of 2-3
// Launch one Task per batch, all in a single message

Task(
  subagent_type: "general-purpose",
  description: "Summarize {key1} and {key2}",
  prompt: """
    You are a literature analysis agent. For each paper below:

    1. Download the arXiv source:
       - mkdir -p literature/arxiv_sources/{key}/
       - wget "https://arxiv.org/e-print/{arxiv_id}" and extract
    2. Read the main .tex file(s)
    3. Write SUMMARY_{key}.md following the template below

    Papers to process:
    - {key1}: arXiv:{id1}, "{title1}"
    - {key2}: arXiv:{id2}, "{title2}"

    **Our paper context**: We study energy-optimal A/F ratios in disaggregated LLM serving.
    Key concepts: r-A-1F topology, TPW (tokens per watt), idle power gap, speed scaling.
    Read paper/energy_optimal_af/or_draft/sections/introduction.tex for context if needed.

    Template for SUMMARY_{key}.md:
    [... full template from Phase 2c ...]

    Return: list of files written and any errors encountered.
  """
)
```

## Modes

### `scan` — Coverage Audit

```
═══════════════════════════════════════════════════════════════════
                    LITERATURE COVERAGE AUDIT
═══════════════════════════════════════════════════════════════════

references.bib entries: 52
With arXiv source downloaded: 13
With SUMMARY_*.md written: 8
Missing summaries: 44

BY CATEGORY:
─────────────────────────────────────────
Speed Scaling (7 entries):
  ✅ bamas2020       — Has source + summary
  ❌ yds1995         — No source, no summary
  ❌ bansal2007      — No source, no summary
  ❌ wierman2012     — No source, no summary
  ...

Queueing Theory (7 entries):
  ❌ halfin1981      — No source, no summary
  ...

PRIORITY TARGETS (most cited in our paper):
  1. wierman2012     — Referenced 8× in proofs
  2. bansal2007      — Referenced 6× in proofs
  3. halfin1981      — Referenced 5× in Section 6
  ...
```

### `batch --from-bib --missing` — Process All Missing

1. Run `scan` internally to identify all missing papers
2. Filter to papers with arXiv IDs (can be downloaded)
3. Group into batches of 2-3
4. Launch parallel Task subagents for each batch
5. After all complete, update README.md
6. Report results

### `status` — Coverage Statistics

Read existing summaries and show:
- Total bib entries vs summaries
- Per-category breakdown
- Quality check (do summaries have all 6 sections?)

## Quality Checks

After writing a summary, verify:
- [ ] All 6 sections present
- [ ] At least 2 main results with formal statements
- [ ] At least 1 proof technique described
- [ ] Section 4 (Connection) has both "What We Borrow" and "What We Add"
- [ ] BibTeX entry included
- [ ] Math notation renders correctly (LaTeX syntax)

## Error Handling

- **arXiv rate limit**: Wait 3 seconds between downloads. If 429 error, wait 30 seconds and retry once.
- **No arXiv source**: Some papers have source disabled. Fall back to WebFetch on the abstract page.
- **Textbook/non-arXiv**: Use WebSearch to find key content. Mark summary as `**Source:** Web-based (no LaTeX source)`.
- **Corrupted tarball**: Try treating the download as a single .tex file (some arXiv sources are not tarballs).

## Integration Points

### With `/deepresearch`
- `/deepresearch` finds papers via web search → outputs recommended papers
- `/lit-acquire` downloads and summarizes those specific papers
- Workflow: `/deepresearch [topic]` → identify papers → `/lit-acquire batch [keys]`

### With `/paper-pipeline`
- `/paper-pipeline lit [keys]` routes to this skill
- Pre-review stage: ensure all cited papers have summaries before writing

### With `/proofread-references`
- After acquiring literature, run `/proofread-references` to verify bib entries match source

## Begin

When invoked, immediately:
1. Parse `$ARGUMENTS` for action type
2. Read `paper/energy_optimal_af/bib/references.bib` for paper metadata
3. Execute the appropriate mode (scan/download/summarize/batch/status)
4. Write files and update index
5. Report results to user
