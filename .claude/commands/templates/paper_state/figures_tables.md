# Figures & Tables Registry: [Paper Name]

**Paper**: [Title]

**Last Updated**: [date]

**Last Audit**: [date] ([audit type])

---

## Figure Registry

| # | Label | Source File | Generation Script | Section | Format | Caption Summary |
|---|-------|-------------|-------------------|---------|--------|-----------------|
| F1 | `fig:example` | example.pdf | generate_figures.py | experiments | PDF | Example result |

---

## Table Registry

| # | Label | Section | Data Source | Rows | Caption Summary |
|---|-------|---------|-------------|------|-----------------|
| T1 | `tab:example` | experiments | exp_results.db | 3 | Comparison |

---

## Generation Standards

### Python rcParams

```python
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
```

### Color Palette

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Best/Success | Green | `#2ecc71` | Neyman, correct |
| Baseline | Blue | `#3498db` | Uniform, fixed |
| Failure | Red | `#e74c3c` | Errors, warnings |
| Neutral | Gray | `#95a5a6` | Reference |
| Alternative | Purple | `#9b59b6` | Secondary |
| Accent | Orange | `#f39c12` | Highlights |
| Theory | Dark gray | `#34495e` | Bounds |

### Legend Placement

- **Default**: Outside below — `bbox_to_anchor=(0.5, -0.15), loc='upper center'`
- **Vertical**: Outside right — `bbox_to_anchor=(1.02, 1.0), loc='upper left'`
- **Inside**: Only if no data overlap

### LaTeX Standards

- Figures: `\begin{figure}[t]`, `\centering`, caption below, `\label{fig:*}`
- Tables: `\begin{table}[t]`, `\centering`, caption above, `\label{tab:*}`, booktabs rules
- Width: `0.75\columnwidth` (single) or `\columnwidth` (full)
- Format: PDF (vector), PNG only for photos at ≥300 DPI

---

## Quality Audit Trail

| Date | Scope | Issues Found | Issues Fixed | Notes |
|------|-------|-------------|-------------|-------|
| [date] | [all/figures/tables] | [count] | [count] | [brief] |

---

## Known Issues

None.

---

## Cross-Reference Check

| Figure/Table | Defined At | First Referenced At | Orphan? |
|--------------|-----------|--------------------|---------|
| `fig:example` | experiments.tex:L33 | experiments.tex:L28 | No |
