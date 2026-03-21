# Figure Fix Reference — Patterns, Tools & Output Templates

## Common Fix Patterns

| Issue | Detection | Fix Strategy |
|-------|-----------|--------------|
| **Text overflow** | Text wider than `minimum width - 2*inner sep` | Option A: Widen panel (+1-2cm). Option B: Add `text width=Xcm` for wrapping. Option C: Shrink font one level. Option D: Abbreviate text. |
| **Memory bar label outside panel** | Label anchor position > panel width | Move label outside bar (`anchor=west` after bar end) with 0.15cm gap |
| **Missing icon** | `\includegraphics` file not found | Copy from `.claude/icons/pdf/`. If only SVG, run `.claude/icons/convert.sh` first. |
| **Arrow misaligned** | Endpoint uses wrong panel dimension variable | Recalculate using correct `\pw`, `\ph`, `\gap` values |
| **Font too small** | `\tiny` (5pt) in TikZ nodes | Replace `\tiny` with `\scriptsize` (7pt minimum) |
| **Overfull hbox in node** | pdflatex log warning | Add `text width` constraint or widen containing node |
| **Text too close to border** | Text within 2pt of stroked edge | Increase `yshift` or reduce `below` for padding. Stroke ~0.8pt wide. |
| **Legend/annotation truncated** | Vision detected clipped text | Widen figure bounding box, increase panel dimensions, or reposition legend |
| **Axis label overlaps adjacent panel** | `node[right]` extends into next `xshift` scope | Remove `node[right]`, use centered `\node` below axis. Or combine with annotation. |
| **Legend not centered** | Legend `xshift` off-center | Recalculate: `xshift = (figure_width - legend_width) / 2`. Note: in scaled figures, positions scale but text widths do NOT. |
| **Figure too vague** | No takeaway annotation | Add brief "so what?" annotations. Use green/red for good/bad. Add trend indicators. |
| **Mermaid text overflow** | Node text too long in `.mmd` | Split with `<br/>`, shorten labels, or widen with `style` |
| **Draw.io icon missing** | `shape=image` with bad base64 | Regenerate with `draft/figures/drawio/icon_b64.sh <name>` |
| **Draw.io stale PDF** | Source newer than exported PDF | Report to user: re-export needed (manual for `.drawio`, auto for `.mmd`) |

### Fix Priority Order

1. **Missing icons / stale PDFs** (compilation errors)
2. **TikZ syntax errors** (compilation errors)
3. **Mermaid/Draw.io re-export** (source changed but PDF not updated)
4. **Text overflow / truncation** (rendering issues)
5. **Font size violations** (readability)
6. **Arrow/alignment issues** (aesthetics)

---

## Tool Reference

### TikZ Checking Scripts (`.claude/scripts/figure_check/`)

| Tool | Command | Purpose |
|------|---------|---------|
| **Full check** | `python .../check_figure_rendering.py <dir> --standalone --vision --output json` | Structural + codex vision (recommended) |
| **Structural only** | `python .../check_figure_rendering.py <dir> --standalone --no-compile --output text` | PyMuPDF overlap/overflow/font |
| **Overlap visualizer** | `python .../visualize_overlaps.py <dir> --figures <labels> --output-dir /tmp/figure_check` | Annotated PNGs with colored boxes |
| **Single figure** | Add `--figures fig:dynamics` | Focus on one figure |
| **JSON output** | Add `--output json` | Machine-readable with vision raw results |

**Config**: `.claude/scripts/figure_check/config.py` (endpoint, model, API version)
**API key**: `.claude/scripts/figure_check/.env` (gitignored, auto-loaded)

### MCP Interactive Rendering Tools

| Tool | MCP Call | Purpose |
|------|----------|---------|
| **TikZ render** | `mcp__tikz-renderer__render_tikz(tikz_code)` | Render TikZ → PNG in-conversation |
| **Mermaid render** | `mcp__mcp-mermaid__generate_mermaid_diagram(...)` | Render Mermaid → base64/SVG/file |
| **UML multi-format** | `mcp__uml-mcp__generate_uml(...)` | PlantUML/D2/Graphviz via Kroki |
| **Draw.io viewer** | `mcp__drawio__open_drawio_xml(content)` | Open Draw.io XML for viewing |
| **Codex assess** | `python .../codex_assess.py <img>` | Send PNG to Codex for assessment |

**When to use**: MCP tools for interactive fix-render-check cycles; batch pipeline for final regression guard.

### Draw.io / Mermaid Scripts (`draft/figures/drawio/`)

| Tool | Command | Purpose |
|------|---------|---------|
| **Export all** | `draft/figures/drawio/export.sh` | Export all `.mmd`/`.drawio` → PDF |
| **Export one** | `draft/figures/drawio/export.sh <file>` | Export specific source |
| **Icon lookup** | `draft/figures/drawio/icon_b64.sh <name>` | Get base64 data URI for XML embedding |
| **Icon list** | `draft/figures/drawio/icon_b64.sh --list` | List all 49 available icons |

**Puppeteer config**: `draft/figures/drawio/puppeteer.json` (headless VM: `--no-sandbox`)
**Output naming**: `draft/figures/drawio_<name>.pdf` (prefix `drawio_` distinguishes from TikZ)

---

## Dry Run Output Example

```
═══════════════════════════════════════════════════════════════════
                   DRY RUN - NO CHANGES MADE
═══════════════════════════════════════════════════════════════════

Automated Detection Results:
  Structural (PyMuPDF): 2 warnings
  Vision (gpt-5.1-codex-mini): 1 warning

Would fix 3 issues in fig:dynamics:

1. figures/fig_dynamics.tex:L22
   Source: structural (text_overflow, 4.2pt)
   Issue: Text overflow in Panel 3 (Admit step)
   Fix: Widen panel minimum width from 3.5cm to 4.0cm

2. figures/fig_dynamics.tex:L58
   Source: structural (font_size)
   Issue: \tiny font in state vector labels
   Fix: Replace \tiny with \scriptsize

3. figures/fig_dynamics.tex:L95
   Source: vision (text_cutoff)
   Issue: Legend text truncated at panel border
   Fix: Increase figure y-extent by 0.3cm

To apply these changes:
   /fix-figures fig:dynamics           → Apply all
```

---

## Output Format Template

```
═══════════════════════════════════════════════════════════════════
                   FIGURE FIXES APPLIED
═══════════════════════════════════════════════════════════════════

Target: {fig:label / all}
Fixes Applied: {N} issues fixed

Phase 1 (Automated Detection):
   Structural (PyMuPDF):
   • Text overlap: {count}
   • Text overflow: {count}
   • Border overlap: {count}
   • Font size: {count}
   • Element overlap: {count}
   Vision (gpt-5.1-codex-mini):
   • Truncated text: {count}
   • Misalignment: {count}
   • Other: {count}
   Log parsing:
   • Overfull hbox: {count}
   • Missing files: {count}

Phase 2 (Source Analysis):
   • Panel overflow: {count}
   • Memory bar issues: {count}
   • Font violations: {count}
   • Arrow issues: {count}

Fixes Applied:
   1. {file}:L{line}: {description} (source: {structural/vision/log})
   2. {file}:L{line}: {description} (source: {structural/vision/log})
   ...

Regression Guard:
   • Recompiled: ✅ / ❌
   • Re-check (structural): {count} issues
   • Re-check (vision): {count} issues
   • Labels intact: ✅ / ❌
   • Iterations: {N}/3

═══════════════════════════════════════════════════════════════════
                        NEXT STEPS
═══════════════════════════════════════════════════════════════════

Fixes Applied: {N} issues fixed
   Target: {fig:label / all figures}

IMMEDIATE ACTIONS:
   1. Visually inspect the rendered PDF to confirm fixes
   2. Re-run /check-figures-tables to verify no remaining issues

RECOMMENDED COMMANDS (in order):

   /check-figures-tables              → Verify all figures pass
   /paper-pipeline quick              → Fast consistency check
   /fix-issues all                    → Fix any other issue types

REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy
   L1 Structure   ─────────── /check-content-placement
 → L1 Figures     ─────────── /fix-figures works here
   L2 Consistency ─────────── /fix-issues symbols/terms/refs
   L3 Style       ─────────── /check-ms-style
   L4 Language    ─────────── /polish-paper

TIP: Use /paper-pipeline status to see overall progress
```
