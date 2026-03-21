# TikZ Drawing Best Practices for Academic Papers

This guide establishes standards for creating high-quality, professional, and error-free TikZ illustrations in academic papers.

## 🎯 Core Philosophy: "Spacing, Alignment, Layering"

Good academic figures should be legible, unambiguous, and aesthetically pleasing. The most common issues are text overlapping with lines, crowded elements, and cluttered backgrounds.

### 1. The Principle of Avoidance (避让原则)

*   **Rule**: Text labels must NEVER overlap with the lines, points, or regions they describe.
*   **Technique**: Do not rely on default TikZ placement. Manually push text away into whitespace.
    *   Use `above`, `below`, `left`, `right` combined with explicit `xshift` and `yshift`.
    *   Use `pos` to slide text along a path to a clearer spot.
    *   Use `pin` or `label` with explicit angles and distances for points.

### 2. The Principle of Breathing Room (留白原则)

*   **Rule**: Elements should not feel cramped. Increase spacing to improve readability.
*   **Technique**:
    *   Add `inner sep=2pt` or more to text nodes to give them a protective buffer.
    *   Increase `node distance` when using the `positioning` library.
    *   Extend axes slightly beyond the data range to avoid edge-crowding.

### 3. The Principle of Layering & Backgrounds (层次原则)

*   **Rule**: Background elements should be subtle; foreground information must pop.
*   **Technique**:
    *   Avoid dense patterns (like crosshatch) for regions; use semi-transparent fills instead (`fill=red, opacity=0.1`).
    *   If text *must* cross a line, give the text a background that matches the page color (usually white) with high opacity (`fill=white, fill opacity=0.9`) to "break" the line underneath.

---

## 🛠️ Technical Implementation Standards

### 1. Robust Syntax

*   **Commands**: Always use single backslashes for LaTeX commands (e.g., `$\alpha$`, not `$\alpha$`).
*   **Newlines**: Always use double backslashes for line breaks inside nodes (e.g., `Line 1\\Line 2`).
*   **Arrows**: Use the syntax `-Stealth` (requires `arrows.meta` library) instead of `->, Stealth`.
*   **Anchors**: Use compass directions (`north`, `south east`), never `top` or `bottom`.

### 2. Positioning & Coordinates

*   **Relative Positioning**: Prefer relative positioning (`right=of nodeA`) over absolute coordinates (`at (5,2)`) for flowcharts and diagrams. It makes refactoring easier.
*   **Shifted Labels**:
    *   ❌ `node[above] {Text}` (Often too close)
    *   ✅ `node[above, yshift=3pt] {Text}` (Better)
*   **Sloped Text**: For labels on lines, use `sloped` to align text rotation with the line, and `above` to lift it off the line.
    *   `\draw (A) -- (B) node[midway, sloped, above] {Label};

### 3. Node Styles

*   **Inline Styles**: For simple figures, define styles inline or locally within the `tikzpicture` environment to avoid global namespace pollution.
*   **Text Nodes**:
    *   For multi-line text, specify `align=center` (or left/right) or `text width=...`.
    *   Example: `\node[align=center, font=\small] {Line 1\\Line 2};

### 4. Avoiding "Standalone" Issues

*   **Direct Embedding**: For maximum stability in complex projects, insert TikZ code directly into the main `.tex` files (or `\input` files that contain *only* the `tikzpicture` environment, with no preamble).
*   **No Preambles**: `\input` files must NOT contain `\documentclass`, `\usepackage`, or `\begin{document}`.

---

## 🔍 Pre-Flight Checklist for TikZ Figures

Before finalizing a figure, verify:

1.  [ ] **No Overlaps**: Do any labels cross lines or other text?
2.  [ ] **Legibility**: Is the font size appropriate (usually `\small` or `\footnotesize`)?
3.  [ ] **Syntax**: Are arrows defined as `-Stealth`? Are newlines `\\`?
4.  [ ] **Aesthetics**: Are regions transparent (`opacity < 0.2`)? Is there enough whitespace?
5.  [ ] **Compilation**: Does it compile without errors in the main document context?

---

## 💡 Examples

### Example 1: Labeling a Line without Overlap

```latex
% Bad
\draw (0,0) -- (4,4) node[midway] {Label};

% Good
\draw (0,0) -- (4,4) node[midway, sloped, above, yshift=2pt] {Label};
```

### Example 2: Highlighting a Region

```latex
% Bad (Dense pattern)
\fill[pattern=crosshatch] (0,0) rectangle (2,2);

% Good (Transparent fill)
\fill[blue, opacity=0.1] (0,0) rectangle (2,2);
```

### Example 3: Text Box with Arrows

```latex
% Bad (Hardcoded, rigid)
\node (A) at (0,0) {Start};
\node (B) at (2,0) {End};
\draw[->] (A) -- (B);

% Good (Flexible, robust)
\node (A) {Start};
\node[right=1.5cm of A] (B) {End};
\draw[-Stealth, thick] (A) -- (B);
```
