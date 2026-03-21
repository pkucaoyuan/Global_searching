# LaTeX Writing Standards and Best Practices

## 📖 Overview

**Purpose**: This guide provides LaTeX code standards, best practices, and debugging guidance for academic paper writing in operations research and machine learning venues.

**Core Philosophy**: Write clean, maintainable LaTeX code that compiles correctly and produces professional-quality output.

**When to Consult This Guide**:
- When writing or modifying LaTeX code
- When encountering LaTeX compilation errors
- When needing to insert complex formulas, figures, or tables
- When uncertain about LaTeX syntax or best practices
- When setting up cross-references or citations

---

## ✍️ Content Modification Principles

### 1. Understand Before Modifying

Before making ANY LaTeX modification:

- **Read complete relevant sections** - Not just the target line, but surrounding context
- **Understand mathematical symbol definitions** - Check `command.tex` for custom macros
- **Understand the role in overall argument** - How does this part contribute to the paper?
- **Verify dependencies** - Check for cross-references (`\ref{}`, `\eqref{}`, `\cite{}`)

### 2. Maintain Consistency

**Symbol Consistency**:
- Same concept → Same symbol throughout (e.g., price always `$p$`, demand always `$D$`)
- Check existing usage before introducing new symbols
- Document symbol definitions in comments when non-standard

**Terminology Consistency**:
- Key terms must remain uniform (e.g., "misspecification" vs "model misspecification")
- Check paper-wide usage with search before modifying terms
- Maintain consistency with established OR/ML literature

**Format Consistency**:
- Theorem/lemma/proposition numbering and referencing style uniform
- Figure/table caption formats consistent
- Equation display formats (inline vs display, alignment) consistent

### 3. Mathematical Formula Standards

**When modifying formulas**:
- ✅ Understand derivation logic completely
- ✅ Verify all symbols are defined
- ✅ Check bracket/parenthesis matching
- ✅ Maintain formula numbering sequence
- ✅ Update all references if formula numbers change
- ⚠️ **Level 3 modification** - requires user confirmation for substantive changes

**Formula presentation**:
- Use inline math `$...$` for simple expressions in text
- Use display math `\[...\]` or `equation` for important standalone formulas
- Use `align` environment for multi-line derivations with alignment

### 4. Citation Standards

**Citation commands**:
- `\citep{}` for parenthetical citations: (Smith 2025)
- `\citet{}` for textual citations: Smith (2025)
- Ensure citation key exists in `.bib` file before use
- Multiple citations: `\citep{author2020, author2021}` (sorted by year)

**Citation management**:
- ✅ All cited works must have complete `.bib` entries
- ✅ Remove unused citations (checked via compilation warnings)
- ✅ Maintain consistent citation key format: `FirstAuthorLastNameYear`

---

## 🎨 LaTeX Code Style

### 1. Indentation and Formatting

**Environment indentation**:
```latex
\begin{theorem}\label{thm:main}
  Statement of the theorem with proper indentation.
\end{theorem}

\begin{proof}
  Proof content indented consistently.
  \begin{align}
    x &= y + z \\
    &= w
  \end{align}
  Continued proof text.
\end{proof}
```

**Line breaks**:
- Leave blank line between paragraphs
- No blank line within a paragraph (LaTeX treats blank line as paragraph break)
- Break long lines for readability (no strict column limit, but keep reasonable)

### 2. Comment Standards

**When to comment**:
- Complex formula transformations
- Non-obvious LaTeX tricks or workarounds
- Temporary deletions (use `%` to comment out, not delete)
- Section boundaries in long files

**Example**:
```latex
% The following uses a custom alignment to handle the long equation
\begin{align}
  % First term: revenue from sales
  R(p) &= p \cdot D(p) \\
  % Second term: cost from inventory
       &\quad - c \cdot I(p)  % c is unit cost
\end{align}
```

### 3. Macro and Command Definitions

**Define in `command.tex`**:
```latex
% Mathematical sets
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}

% Operators
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}

% Common notation
\newcommand{\E}{\mathbb{E}}  % Expectation
\newcommand{\Prob}{\mathbb{P}}  % Probability
```

**Best practices**:
- ✅ Use semantic names: `\demand` not `\d`
- ✅ Group related definitions with comments
- ✅ Avoid redefining standard LaTeX commands
- ✅ Document complex macro definitions

---

## 📖 LaTeX Best Practices

### Mathematical Environments

**Inline math**: `$...$` or `\(...\)`
```latex
The optimal price $p^*$ maximizes revenue.
```

**Display math (no numbering)**: `\[...\]`
```latex
\[
  p^* = \argmax_{p \in [0,1]} R(p)
\]
```

**Numbered equations**: `equation` environment
```latex
\begin{equation}\label{eq:revenue}
  R(p) = p \cdot D(p)
\end{equation}
```

**Multi-line alignment**: `align` environment
```latex
\begin{align}
  R(p) &= p \cdot D(p) \label{eq:revenue-def} \\
       &= p \cdot (\alpha - \beta p) \label{eq:revenue-linear} \\
       &= \alpha p - \beta p^2 \label{eq:revenue-expanded}
\end{align}
```

**When to use each**:
- Inline: Simple expressions, parameter mentions
- Display unnumbered: Intermediate steps, definitions not referenced later
- Numbered equation: Key results, definitions referenced elsewhere
- Align: Multi-step derivations, system of equations

### Theorem Environments

**Standard theorem-like environments**:
```latex
\begin{theorem}\label{thm:main}
  Under assumptions A1-A3, the optimal policy achieves $O(\log T)$ regret.
\end{theorem}

\begin{proof}
  We proceed in three steps...
\end{proof}

\begin{lemma}\label{lem:helper}
  Technical result needed for main theorem.
\end{lemma}

\begin{proposition}\label{prop:characterization}
  The optimal solution has the following structure...
\end{proposition}

\begin{corollary}\label{cor:special-case}
  In the special case where $\alpha = 1$, we obtain...
\end{corollary}
```

**Best practices**:
- Always use `\label{}` for theorems/lemmas/propositions you reference
- Place `\label{}` immediately after `\begin{theorem}` on same line
- Use descriptive label names: `thm:main-result` not `thm:1`
- End proofs with `\end{proof}` (produces automatic QED symbol)

### Figures and Tables

**Figure template**:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.8\linewidth]{figs/figure1.pdf}
  \caption{Caption describing the figure content. Explain what readers should observe.}
  \label{fig:results}
\end{figure}
```

**Table template**:
```latex
\begin{table}[ht]
  \centering
  \caption{Caption for the table.}
  \label{tab:comparison}
  \begin{tabular}{lcc}
    \toprule
    Method & Regret & Runtime \\
    \midrule
    Algorithm 1 & $O(\log T)$ & $O(T)$ \\
    Algorithm 2 & $O(\sqrt{T})$ & $O(1)$ \\
    \bottomrule
  \end{tabular}
\end{table}
```

**Best practices**:
- Use `[ht]` placement: "here" or "top" (avoid `[h!]` which can cause issues)
- Always include `\caption{}` and `\label{}`
- For tables, caption goes BEFORE tabular; for figures, AFTER includegraphics
- Use `\centering` not `\begin{center}...\end{center}` (avoids extra spacing)
- Prefer vector graphics (PDF) over raster (PNG) for figures

### Symbol and Macro Definitions

**Standard mathematical sets**:
```latex
\newcommand{\R}{\mathbb{R}}  % Real numbers
\newcommand{\N}{\mathbb{N}}  % Natural numbers
\newcommand{\Z}{\mathbb{Z}}  % Integers
```

**Common operators**:
```latex
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator{\Tr}{Tr}  % Trace
```

**Problem-specific notation**:
```latex
\newcommand{\price}{p}  % Price
\newcommand{\demand}{D}  % Demand function
\newcommand{\revenue}{R}  % Revenue function
```

**Formatting shortcuts**:
```latex
\newcommand{\norm}[1]{\lVert #1 \rVert}  % Norm: \norm{x}
\newcommand{\abs}[1]{\lvert #1 \rvert}  % Absolute value: \abs{x}
\newcommand{\inner}[2]{\langle #1, #2 \rangle}  % Inner product
```

---

## 🔗 Cross-Reference Best Practices

### Label Naming Conventions

**Use prefixes to identify reference type**:
- `sec:` for sections: `\label{sec:intro}`
- `eq:` for equations: `\label{eq:bellman}`
- `thm:` for theorems: `\label{thm:main-result}`
- `lem:` for lemmas: `\label{lem:concentration}`
- `prop:` for propositions: `\label{prop:optimality}`
- `fig:` for figures: `\label{fig:convergence}`
- `tab:` for tables: `\label{tab:results}`

**Use descriptive names**:
- ✅ `\label{eq:revenue-formula}` - Clear what it refers to
- ❌ `\label{eq:1}` - No semantic meaning
- ✅ `\label{thm:asymptotic-optimality}` - Describes result
- ❌ `\label{thm:main}` - Too generic if multiple main theorems

### Referencing Commands

**Sections**:
```latex
\section{Introduction}\label{sec:intro}
...
As discussed in Section~\ref{sec:intro}...
```

**Equations**:
```latex
\begin{equation}\label{eq:bellman}
  V(s) = \max_a \{r(s,a) + \gamma V(s')\}
\end{equation}
...
By equation~\eqref{eq:bellman}, we have...
```
Note: Use `\eqref{}` for equations (adds parentheses automatically)

**Theorems/Lemmas**:
```latex
\begin{theorem}\label{thm:regret-bound}
  The regret is $O(\log T)$.
\end{theorem}
...
Theorem~\ref{thm:regret-bound} establishes...
```

**Figures/Tables**:
```latex
\begin{figure}...\label{fig:results}\end{figure}
...
Figure~\ref{fig:results} shows...
```

### Maintaining Cross-Reference Consistency

**Before modifying**:
- ✅ Search for all uses of a label before deleting it
- ✅ Update all references if renaming a label
- ✅ Check compilation warnings for undefined references

**After modifying**:
- ✅ Compile twice (first pass updates labels, second pass resolves refs)
- ✅ Check for "Reference `xxx' undefined" warnings
- ✅ Verify numbering is correct in final PDF

---

## 🔍 Common LaTeX Tasks

### 1. Extending Section Content

**Task**: Add new content to existing section

**Checklist**:
- [ ] Read entire section first to understand flow
- [ ] Identify where new content fits logically
- [ ] Check if new math symbols conflict with existing definitions
- [ ] Add appropriate cross-references to related sections
- [ ] Maintain consistent notation and terminology

**Example**:
```latex
% Existing section
\section{Problem Formulation}\label{sec:formulation}
We consider a dynamic pricing problem...

% Adding new content
% First, understand: What's already defined? What notation exists?
% Then add content that flows naturally:

\paragraph{Resource Constraints}
Unlike the classical model, we incorporate resource constraints...
```

### 2. Optimizing Proof Structure

**Task**: Make proof clearer or more organized

**Strategies**:
- Use paragraph headings for proof steps:
  ```latex
  \begin{proof}
    \paragraph{Step 1: Upper bound}
    We first establish...

    \paragraph{Step 2: Lower bound}
    Next, we show...

    \paragraph{Step 3: Combining bounds}
    Putting these together...
  \end{proof}
  ```

- Number key equations in proofs for easy reference
- Move technical details to appendix, keep main proof high-level
- Add intuitive explanation before formal argument

### 3. Improving Mathematical Derivations

**Task**: Fill gaps in mathematical reasoning

**Best practices**:
- Explain non-trivial steps with "by [lemma/theorem]" or "since"
- Add intermediate equations for complex derivations
- Use `align` environment to show step-by-step transformations
- Define all notation before use

**Example**:
```latex
% Before (too sparse)
\begin{align}
  R(p) &= p \cdot D(p) \\
       &= \alpha p - \beta p^2
\end{align}

% After (shows reasoning)
\begin{align}
  R(p) &= p \cdot D(p) \\
       &= p \cdot (\alpha - \beta p)  && \text{(linear demand)} \\
       &= \alpha p - \beta p^2  && \text{(expanding)}
\end{align}
```

### 4. Compilation Error Fixing

See **LaTeX Code Debugging** section below.

### 5. Format Optimization

**Common tasks**:
- Adjust figure/table placement: Try different position specifiers `[ht]`, `[tb]`, `[p]`
- Fix overfull/underfull hbox warnings: Reword sentences, use `\linebreak` sparingly
- Control spacing: Use `\vspace{}`, `\hspace{}` minimally (prefer LaTeX defaults)
- Balance columns (for two-column format): Use `\balance` command if available

---

## 🛠️ LaTeX Code Debugging

### Common Compilation Errors

#### 1. Missing `$` (Math Mode Error)

**Error message**:
```
! Missing $ inserted.
```

**Cause**: Mathematical symbol used outside math mode

**Fix**:
```latex
% Wrong
The price p maximizes revenue.

% Correct
The price $p$ maximizes revenue.
```

#### 2. Undefined Control Sequence

**Error message**:
```
! Undefined control sequence.
l.42 \demand
```

**Cause**: Command not defined or package not loaded

**Fix**:
- Check if command defined in `command.tex`
- Verify required package is loaded in preamble
- Check for typos in command name

#### 3. Missing `}` or `\end{}`

**Error message**:
```
! File ended while scanning use of \@writefile.
```
or
```
! LaTeX Error: \begin{document} ended by \end{align}.
```

**Cause**: Unmatched braces or environments

**Fix**:
- Use editor's bracket matching feature
- Check each `\begin{}` has matching `\end{}`
- Verify all `{` have matching `}`

#### 4. Reference Undefined

**Warning message**:
```
LaTeX Warning: Reference `eq:main' on page 5 undefined.
```

**Cause**: Label doesn't exist or compilation needs second pass

**Fix**:
- Compile twice (first pass records labels, second resolves refs)
- Check spelling of label names
- Verify `\label{}` exists for referenced item

#### 5. Citation Undefined

**Warning message**:
```
LaTeX Warning: Citation `Smith2025' on page 3 undefined.
```

**Cause**: BibTeX entry missing or BibTeX not run

**Fix**:
- Check citation key exists in `.bib` file
- Run compilation sequence: LaTeX → BibTeX → LaTeX → LaTeX
- Verify citation key spelling matches `.bib` entry

### Error Location Methods

**Reading error messages**:
```
! LaTeX Error: ...
l.142 \end{align}
```
The `l.142` indicates error is at/near line 142.

**Strategies**:
- **Binary search**: Comment out half the document, recompile, narrow down
- **Check recent changes**: Error often near recent modifications
- **Look backward**: Error may be before the reported line (e.g., missing `\begin{}`)

### Fix Verification

After fixing errors:
- [ ] **Full clean compilation**: Delete `.aux`, `.bbl`, `.blg` files, recompile from scratch
- [ ] **Check PDF output**: Verify formulas, numbering, references display correctly
- [ ] **Review warnings**: Address remaining warnings (undefined refs, overfull boxes, etc.)

---

## 📚 LaTeX Resources

### Official Documentation

- **CTAN (Comprehensive TeX Archive Network)**: [https://ctan.org/](https://ctan.org/)
  - Package documentation
  - Symbol lists
  - LaTeX guides

- **LaTeX Symbol List**: [https://ctan.org/pkg/comprehensive](https://ctan.org/pkg/comprehensive)
  - Comprehensive symbol reference with commands

- **Short Math Guide for LaTeX**: [https://ctan.org/pkg/short-math-guide](https://ctan.org/pkg/short-math-guide)
  - Mathematical typesetting reference

### Useful Tools

- **Detexify**: [http://detexify.kirelabs.org/classify.html](http://detexify.kirelabs.org/classify.html)
  - Draw symbol to find LaTeX command

- **Overleaf Documentation**: [https://www.overleaf.com/learn](https://www.overleaf.com/learn)
  - Tutorials and examples

### Quick Reference

**Essential packages for OR papers**:
```latex
\usepackage{amsmath, amssymb, amsthm}  % Math support
\usepackage{algorithm, algorithmic}    % Algorithms
\usepackage{graphicx}                  % Figures
\usepackage{booktabs}                  % Professional tables
\usepackage{natbib}                    % Citations
\usepackage{hyperref}                  % Hyperlinks (load last)
```

---

## ✅ LaTeX Quality Checklist

### Before Submission

- [ ] **Compilation clean**: No errors, all warnings addressed
- [ ] **Cross-references work**: All `\ref{}`, `\eqref{}` resolve correctly
- [ ] **Citations complete**: All cited works in `.bib` file with complete info
- [ ] **Numbering consistent**: Theorems, equations, figures numbered correctly
- [ ] **Notation consistent**: Same symbols used throughout for same concepts
- [ ] **Figures display correctly**: All figures visible, correct size, good quality
- [ ] **Tables formatted professionally**: Use `booktabs` package for clean tables
- [ ] **No LaTeX artifacts in PDF**: No "??" for undefined refs, no "[]" for missing citations

### Code Quality

- [ ] **Indentation consistent**: Nested environments properly indented
- [ ] **Comments where needed**: Complex formulas/tricks explained
- [ ] **No redundant code**: Remove commented-out old versions before submission
- [ ] **Macros defined properly**: All custom commands in `command.tex`
- [ ] **Line breaks reasonable**: Code readable, not cramped or excessive wrapping

---

## 📝 Final Tips

### When Writing LaTeX

1. **Compile frequently**: Catch errors early, don't accumulate many errors
2. **Use version control**: Git commits before major changes
3. **Descriptive labels**: Future you will thank present you
4. **Comments for complex code**: Explain non-obvious LaTeX tricks
5. **Test on clean compilation**: Delete auxiliary files, recompile from scratch

### When Debugging

1. **Read error messages carefully**: Line numbers and context are clues
2. **Binary search for errors**: Comment out sections to isolate problem
3. **Check recent changes first**: Error often in recent modifications
4. **Compile multiple times**: Some warnings resolve after second compilation
5. **Ask for help**: If stuck, show minimal example that reproduces error

---

**Document Version**: v2.0 (Modular architecture)
**Created**: 2025-11-30
**Updated**: 2026-01-12
**Maintenance Status**: Active maintenance
**Applicable Projects**: Academic paper writing (LaTeX format)
