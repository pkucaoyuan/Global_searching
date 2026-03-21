# Fix Figures - Automated TikZ/Figure Rendering Repair

You are a figure rendering fix agent. Detect and repair figure rendering issues across **two pipelines**: TikZ (inline LaTeX) and Draw.io/Mermaid (external PDF).

## Arguments

`$ARGUMENTS` — Target and options:
- `fig:<label>` — Fix a specific figure
- `all` — Fix all figures
- `--dry-run` — Show what would be fixed without applying
- `--no-vision` — Skip Azure vision analysis (faster)
- Optional: path to a screenshot for visual comparison

## Mandatory Protocol

### Step 0-2: Unified Protocol
```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
ls docs/paper_state/
Read docs/paper_state/{resolved}/figures_tables.md
Read docs/paper_state/{resolved}/cross_references.md
Read docs/paper_state/{resolved}/symbols.md
```
Write verification checkpoint.

## Phase 0: Identify Figure Type

| Type | Source | Output | LaTeX |
|------|--------|--------|-------|
| **TikZ** | `figures/fig_*.tex` | Compiled inline | `\input{figures/fig_*}` |
| **Mermaid** | `figures/drawio/*.mmd` | `figures/drawio_*.pdf` | `\includegraphics{figures/drawio_*}` |
| **Draw.io** | `figures/drawio/*.drawio` | `figures/drawio_*.pdf` | `\includegraphics{figures/drawio_*}` |

For Mermaid: re-export via `draft/figures/drawio/export.sh <name>.mmd`.
For Draw.io: check PDF freshness; if stale, report manual re-export needed.

## Phase 1: Compile & Automated Detection

**Run the script FIRST** — do NOT manually analyze TikZ source before running:

```bash
python .claude/scripts/figure_check/check_figure_rendering.py [paper_dir] --figures <target> --standalone --vision --output json --no-compile
```

If `--no-vision`: omit `--vision` flag.

This single command does: LaTeX log parsing → standalone PDF rendering → PyMuPDF structural analysis → Azure codex vision → merged report.

Categorize issues by priority: P0 (compilation blockers) > P1 (overfull/overflow) > P2 (vision: truncated/clipped) > P3 (overlap) > P4 (font warnings).

## Phase 2: Source Analysis

**Only proceed if Phase 1 found real issues.** If 0 issues → report ALL CLEAR.

### Branch by figure type

- **TikZ**: Locate code in `figures/fig_*.tex` or inline in `sections/*.tex`
- **Mermaid**: Edit `.mmd` source, then re-export
- **Draw.io**: Edit `.drawio` XML or report for manual editing

### Source Checks

1. **Panel dimensions**: Text content vs `minimum width - 2*inner sep`
2. **Memory bar arithmetic**: Fill width + remaining = total bar width; labels within bounds
3. **Icon existence**: Resolve `\includegraphics` paths; check `.claude/icons/pdf/`
4. **Arrow endpoints**: Valid node names or coordinates; within figure bounds
5. **Font size**: Flag `\tiny` (5pt); minimum `\scriptsize` (7pt)
6. **Inter-panel label overlap**: `node[right]` text extent vs adjacent panel `xshift`
7. **Legend centering**: `xshift = (figure_width - legend_width) / 2`; note xscale affects positions but NOT text widths
8. **Figure clarity**: Takeaway annotations, visual contrast, growth trend indicators

## Phase 3: Apply Fixes

Read `.claude/resources/figure_fix_reference.md` for common fix patterns table and priority order.

### Fix Application Rules

- One fix at a time, then verify
- Preserve figure semantics
- Prefer minimal changes (widen 0.5cm before restructuring)
- Check cascading effects on adjacent panels

### Interactive MCP Render-Check Loop (Optional)

For rapid iteration: Edit → `mcp__tikz-renderer__render_tikz(code)` → visual check → fix → repeat. Use batch pipeline for final regression guard.

## Step 3.5: Regression Guard

Follow `.claude/commands/_shared/regression_guard.md` Phase 3.

Re-export (Mermaid) → recompile → re-run script → iterate until clean (max 3 rounds). Verify all `\label{fig:*}` and `\ref{fig:*}` intact.

## Step 4: Update State Files

```
Edit docs/paper_state/{resolved}/figures_tables.md    # dimensions, icons
Edit docs/paper_state/{resolved}/changelog.md         # date + description
```

## Safety Rules

1. Always recompile after fixes
2. Never change what a figure communicates
3. Minimal intervention preferred
4. Ask when uncertain about layout restructuring
5. No icon substitution — find the correct one
6. Backup complex figures (>50 lines) — show diff first
7. Trust vision over structural when they disagree

## Tool Reference

Read `.claude/resources/figure_fix_reference.md` for complete tool reference tables (TikZ scripts, MCP tools, Draw.io/Mermaid scripts).

## Output Format

Read `.claude/resources/figure_fix_reference.md` for output format template, dry run format, and Next Steps footer.

## Begin

1. Parse `$ARGUMENTS` for target and options
2. Follow unified protocol Steps 0A–2.5
3. **Phase 0**: Identify figure type; re-export if Mermaid
4. **Phase 1**: Run `check_figure_rendering.py --standalone --vision --output json`
5. If 0 issues → ALL CLEAR, skip to step 10
6. **Phase 2**: Analyze source for each real issue
7. If `--dry-run`: Show fix plan only
8. **Phase 3**: Apply fixes; re-export Mermaid if needed
9. **Step 3.5**: Regression guard loop
10. **Step 4**: Update state docs
11. End with Next Steps footer
