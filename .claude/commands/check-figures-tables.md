# Check Figures & Tables - Visual Element Quality Audit

You are a figure and table quality auditor for academic papers. Your task is to systematically check all visual elements for publication readiness.

## Arguments

- `$ARGUMENTS` - Optional: scope (`figures`, `tables`, `all`) or specific label (e.g., `fig:coverage`). Default: `all`

## ⚠️ MANDATORY: Follow Unified Protocol

**STOP. Before ANY action, execute Steps 0A-2.5 from:**
```
Read .claude/commands/_shared/unified_protocol.md
```

**State files to read**: `figures_tables.md`, `cross_references.md`, `framing.md`
**State files to update**: `figures_tables.md`, `consistency_log.md`, `changelog.md`

---

## Why This Matters

Reviewer complaints:
- "Figure 3 legend obscures data points"
- "Table 2 numbers don't match Section 5"
- "Figures use inconsistent fonts/colors"
- "Can't read axis labels at printed size"

These are preventable with systematic checking.

---

## Phase 1: Discovery — Build Figure/Table Registry

### 1.1 Find All Visual Elements in LaTeX

Search all `.tex` files for figures/tables. Figures may be inline in section files or modular in `figures/*.tex` (loaded via `\input{figures/...}`):

```bash
# Figures (search both sections/ and figures/ directories)
grep -rn "\\begin{figure}" sections/*.tex figures/*.tex 2>/dev/null
grep -rn "\\includegraphics" sections/*.tex figures/*.tex 2>/dev/null
grep -rn "\\label{fig:" sections/*.tex figures/*.tex 2>/dev/null

# Tables
grep -rn "\\begin{table}" sections/*.tex figures/*.tex 2>/dev/null
grep -rn "\\label{tab:" sections/*.tex figures/*.tex 2>/dev/null

# Tikz/pgfplots (inline or modular)
grep -rn "\\begin{tikzpicture}" sections/*.tex figures/*.tex 2>/dev/null
grep -rn "\\begin{pgfplot}" sections/*.tex figures/*.tex 2>/dev/null

# Modular figure \input references (figures loaded from separate files)
grep -rn "\\\\input{figures/" sections/*.tex 2>/dev/null
```

### 1.2 Find All Source Files

```bash
# List all figure image files (not .tex)
ls figures/*.pdf figures/*.png figures/*.eps 2>/dev/null | grep -v '\.tex$'

# List modular TikZ figure files
ls figures/fig_*.tex 2>/dev/null

# Find generation scripts
grep -rn "savefig\|plt.save\|fig.save" scripts/ experiments/ paper/
```

### 1.3 Build Registry

For each visual element, record:

```markdown
| ID | Label | Type | Source File | Generation Script | Section | First Ref |
|----|-------|------|-------------|-------------------|---------|-----------|
| F1 | fig:coverage | Figure | fig1_coverage.pdf | generate_theory_figures.py | experiments | L31 |
| F2 | fig:dynamics | TikZ | figures/fig_dynamics.tex | (modular TikZ) | model | L59 |
| T1 | tab:allocator-convergence | Table | (inline) | - | experiments | L150 |
```

---

## Phase 2: LaTeX Quality — Structure & References

### 2.1 Figure Environment Checklist

For each `\begin{figure}`:

| Check | Standard | Example |
|-------|----------|---------|
| Placement specifier | `[t]` or `[tb]` (never `[h]` alone) | `\begin{figure}[t]` |
| `\centering` present | Required | `\centering` |
| `\caption{}` present | Required, below figure | `\caption{...}` |
| `\label{}` present | Required, after `\caption` | `\label{fig:name}` |
| Width specification | `width=0.75\columnwidth` to `\columnwidth` | `width=0.8\linewidth` |
| Format | PDF preferred (vector) | `figures/name.pdf` |

### 2.2 Table Environment Checklist

For each `\begin{table}`:

| Check | Standard | Example |
|-------|----------|---------|
| Placement specifier | `[t]` or `[tb]` | `\begin{table}[t]` |
| `\centering` present | Required | `\centering` |
| `\caption{}` present | Required, **above** table | `\caption{...}` |
| `\label{}` present | Required, after `\caption` | `\label{tab:name}` |
| Booktabs rules | `\toprule`, `\midrule`, `\bottomrule` | No `\hline` |
| Font size | `\small` or `\footnotesize` for dense tables | `{\small ...}` |

### 2.3 Cross-Reference Validity

| Check | How |
|-------|-----|
| Every figure has `\label{}` | grep for figures without labels |
| Every `\label{fig:*}` is referenced | grep for orphan labels |
| Every `\ref{fig:*}` has matching label | grep for broken refs |
| First reference is BEFORE figure placement | check line numbers |
| Same checks for `tab:*` | parallel scan |

### 2.4 Caption Quality

For each caption, check:

| Criterion | Good | Bad |
|-----------|------|-----|
| Self-contained | "Coverage rate of CS at $\delta=0.05$. All three CS methods exceed the 95% target." | "Results." |
| States the takeaway | "Neyman allocation achieves 48% cost reduction over uniform." | "Cost comparison." |
| References specific elements | "Blue: Neyman; Red: Uniform" | No color mapping |
| No orphan references | All `\Cref{}` in caption resolve | Dangling refs |
| Consistent style | Same structure across all captions | Mixed styles |

**Caption Pattern** (from top venues):
```
[What is shown]. [Key takeaway]. [Method details if needed].
```

Example:
```latex
\caption{Anytime-valid coverage at $\delta = 0.05$.
All three confidence sequence methods maintain coverage above
the 95\% target (dashed line) across 200 trials, confirming
\Cref{thm:anytime-ci}.}
```

---

## Phase 3: Source File Quality — Format & Resolution

### 3.1 File Format Check

| Format | Status | When to Use |
|--------|--------|-------------|
| PDF | Preferred | All line plots, diagrams, charts |
| EPS | Acceptable | Legacy workflows |
| PNG | Warning | Only for screenshots/photos (≥300 DPI) |
| JPG | Reject | Never for scientific figures |
| SVG | Convert | Convert to PDF before inclusion |

**Check**: `ls figures/*.png figures/*.jpg 2>/dev/null` — flag any raster files.

### 3.2 Resolution Check (Raster Files Only)

For any PNG/JPG files:
```bash
identify -verbose figures/*.png 2>/dev/null | grep -E "Resolution|Geometry"
```

Minimum: 300 DPI at printed size. At `0.75\columnwidth` (~3.5 inches), minimum pixel width = 1050px.

### 3.3 Font Consistency Check

If Python generation scripts exist, verify rcParams:

| Parameter | Required Value | Why |
|-----------|---------------|-----|
| `font.family` | `"serif"` | Match LaTeX document |
| `font.serif` | `["CMU Serif", "Computer Modern Roman", ...]` | LaTeX compatibility |
| `mathtext.fontset` | `"cm"` | Match document math |
| `font.size` | 10-11 | Readable at column width |
| `axes.labelsize` | 10-12 | Clear axis labels |
| `legend.fontsize` | 8-10 | Readable but not dominant |
| `figure.dpi` / `savefig.dpi` | 300 | Journal quality |
| `savefig.bbox` | `"tight"` | No whitespace waste |

**Check**: Search generation scripts for `rcParams`, `plt.rc`, `mpl.rc`.

---

## Phase 4: Visual Design — Readability & Consistency

### 4.1 Legend Placement

| Placement | When | How |
|-----------|------|-----|
| Outside below | Default for multi-item | `bbox_to_anchor=(0.5, -0.15), loc='upper center'` |
| Outside right | Vertical legend | `bbox_to_anchor=(1.02, 1.0), loc='upper left'` |
| Inside (top-right) | Only if no data overlap | `loc='upper right'` |
| Inside (other) | Avoid | Data obscuring risk |

**Check**: For each figure, verify legend does not overlap data points.

### 4.2 Color Palette Consistency

All figures should use the same semantic color mapping:

| Semantic Role | Color | Hex | Usage |
|---------------|-------|-----|-------|
| Best/Success | Green | `#2ecc71` | Neyman allocator, correct results |
| Baseline | Blue | `#3498db` | Uniform, fixed allocators |
| Failure/Warning | Red | `#e74c3c` | Failure modes, elevated costs |
| Neutral | Gray | `#95a5a6` | Background, reference lines |
| Alternative | Purple | `#9b59b6` | Secondary methods |
| Accent | Orange | `#f39c12` | Highlights |
| Theory/Bound | Dark gray | `#34495e` | Theoretical lines |

**Check**: Across all generation scripts, verify same colors map to same concepts.

### 4.3 Axis Labels & Ticks

| Check | Standard |
|-------|----------|
| Axis labels present | Both X and Y labeled |
| Units specified | "$\Delta$ (quality gap)" not just "$\Delta$" |
| Tick labels readable | Font size ≥ 8pt at printed column width |
| Grid lines | Light grid (`alpha=0.2`) or none |
| Log scale labeled | If log-log, both axes clearly marked |

### 4.4 Annotation Checks

| Check | Standard |
|-------|----------|
| Text doesn't overlap data | All annotations clear of plot elements |
| Arrow annotations point correctly | Arrows connect text to data |
| Reference lines labeled | Dashed lines have legend entry or annotation |
| Consistent annotation style | Same font, box style across figures |

---

## Phase 5: Table Data — Accuracy & Formatting

### 5.1 Number Verification

For each number in a table:
1. Find the source (experiment script, results file)
2. Verify the number matches
3. Check precision is appropriate (e.g., 98.8% not 98.7532%)

### 5.2 Table Design Rules

| Rule | Standard | Bad Example |
|------|----------|-------------|
| No vertical rules | Use booktabs | `|c|c|c|` |
| Minimal horizontal rules | `\toprule`, `\midrule`, `\bottomrule` only | `\hline` everywhere |
| Column alignment | Numbers right-aligned or decimal-aligned | Left-aligned numbers |
| Bold for headers only | `\textbf{Header}` | Bold throughout |
| Units in header | "Cost (\$)" not "Cost" with unit in body | Units scattered |

### 5.3 Cross-Table Consistency

If the same metric appears in multiple tables or in text:
- Values must match exactly
- Precision must be consistent (all 1 decimal or all 2)

---

## Phase 6: Accessibility

### 6.1 Colorblind Safety

| Check | Action |
|-------|--------|
| Red-green combination | Add pattern/marker differentiation |
| Information encoded only by color | Add labels or markers |
| Grayscale readable | Test: does figure work in B&W? |

### 6.2 Print Readability

| Check | Standard |
|-------|----------|
| Line thickness | ≥ 1.5pt for main lines |
| Marker size | ≥ 6pt |
| Font size at print | ≥ 8pt when scaled to column width |
| Contrast | Dark lines on light background |

---

## Output Format

```markdown
# Figure & Table Quality Report

**Paper**: [title]
**Date**: [date]
**Scope**: [all / figures / tables / specific label]

---

## Registry

### Figures ([count])

| # | Label | File | Script | Section | Format | Status |
|---|-------|------|--------|---------|--------|--------|
| F1 | fig:coverage | fig1_coverage.pdf | generate_theory_figures.py | experiments | PDF | OK |
| F2 | fig:ablations | fig5_ablations.png | - | experiments | PNG | WARNING |

### Tables ([count])

| # | Label | Section | Rows | Status |
|---|-------|---------|------|--------|
| T1 | tab:allocator-convergence | experiments | 3 | OK |

---

## Issues Found

### Critical (Must Fix)

| # | Type | Location | Issue | Fix |
|---|------|----------|-------|-----|
| 1 | Raster format | fig5_ablations.png | PNG not PDF | Regenerate as PDF |
| 2 | Missing label | experiments.tex:L180 | Figure without \label{} | Add label |

### Warnings (Should Fix)

| # | Type | Location | Issue | Fix |
|---|------|----------|-------|-----|
| 3 | Legend overlap | fig:cost-scaling | Legend covers data | Use bbox_to_anchor |
| 4 | Inconsistent color | fig:delayed-coverage | Green = baseline (should be blue) | Swap colors |

### Info (Optional)

| # | Type | Location | Issue |
|---|------|----------|-------|
| 5 | Caption style | fig:coverage | Could be more self-contained |

---

## Consistency Matrix

### Color Usage Across Figures

| Figure | Green | Blue | Red | Consistent? |
|--------|-------|------|-----|-------------|
| fig:coverage | Betting CS | EB CS | Hoeffding | OK |
| fig:cost-scaling | Neyman | Uniform | Oracle | OK |

### Font Check

| Script | font.family | font.size | mathtext | DPI | Status |
|--------|------------|-----------|----------|-----|--------|
| generate_theory_figures.py | serif | 10 | cm | 300 | OK |
| generate_service_figures.py | serif | 10 | cm | 300 | OK |

---

## Summary

| Category | Pass | Warn | Fail |
|----------|------|------|------|
| LaTeX structure | X | Y | Z |
| File format | X | Y | Z |
| Visual design | X | Y | Z |
| Table accuracy | X | Y | Z |
| Accessibility | X | Y | Z |
| **Total** | **X** | **Y** | **Z** |

### Priority Actions
1. [Most critical fix]
2. [Second priority]
3. ...
```

---

## Quick Detection Commands

```bash
# Find all figures (inline + modular)
grep -rn "\\\\begin{figure}" sections/*.tex figures/*.tex 2>/dev/null

# Find modular figure \input references
grep -rn "\\\\input{figures/" sections/*.tex 2>/dev/null

# Find figures without labels
grep -A5 "\\\\begin{figure}" sections/*.tex figures/*.tex 2>/dev/null | grep -v "label{fig"

# Find orphan figure labels (defined but never referenced)
LABELS=$(grep -rho "\\\\label{fig:[^}]*}" sections/*.tex figures/*.tex 2>/dev/null | sed 's/\\label{//;s/}//')
for L in $LABELS; do
  COUNT=$(grep -rc "ref{$L}" sections/*.tex figures/*.tex 2>/dev/null | awk -F: '{s+=$2}END{print s}')
  [ "$COUNT" -eq 0 ] && echo "ORPHAN: $L"
done

# Find raster figures
ls figures/*.png figures/*.jpg 2>/dev/null

# Check Python script font settings
grep -n "rcParams\|font.family\|font.serif\|mathtext\|savefig.dpi" paper/journal/scripts/*.py

# Check legend placement
grep -n "bbox_to_anchor\|loc=.*legend\|ax.legend" paper/journal/scripts/*.py
```

## Self-Dispatch Phases

**This skill has 1 setup phase + 5 parallel audit phases. Follow `.claude/commands/_shared/self_dispatch_protocol.md`.**

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 0 | Discovery (registry) | No (setup) | All `sections/*.tex`, `figures/*.tex`, generation scripts | Build figure/table registry with labels, files, scripts, sections |
| 1 | LaTeX quality | Yes (after 0) | All `sections/*.tex`, `figures/*.tex` | Placement specifiers, centering, captions, labels, cross-refs, caption quality |
| 2 | Source file quality | Yes (after 0) | `figures/` (images + TikZ .tex), generation scripts | File format (PDF preferred), resolution, font settings in scripts |
| 3 | Visual design | Yes (after 0) | `figures/` (images + TikZ .tex), generation scripts | Legend placement, color palette consistency, axis labels, annotations |
| 4 | Table data accuracy | Yes (after 0) | All `sections/*.tex`, `figures/*.tex`, experiment results | Numbers match source; precision consistent; cross-table values match |
| 5 | Accessibility | Yes (after 0) | `figures/` (images + TikZ .tex), generation scripts | Colorblind safety, print readability, line thickness, font size |

**Sequential**: Phase 0 (discovery) must complete first — produces the figure/table registry.
**Parallel group**: Phases 1-5 can run in parallel (all consume Phase 0 registry).
**Aggregation**: Merge 5 sub-reports into single quality report; build consistency matrix.

---

## Begin

**Dispatch**: Setup → parallel — **Template B** from `self_dispatch_protocol.md`.
**Setup output**: Figure/table registry (labels, source files, generation scripts, sections).

1. Follow unified protocol Steps 0A–2.5
2. Execute Phase 0 inline (discover all figures/tables, build registry)
3. Recursion guard → if subagent, execute remaining phases inline
4. Dispatch 5 parallel Task subagents (Phases 1-5), each receives registry
5. Aggregate → deduplicate → sort by severity
6. Update `figures_tables.md` state doc
