# Symbol Registry: [Paper Title]

**Last Updated**: [date]
**Total Symbols**: [count]

---

## Core Symbols

| Symbol | Meaning | Type | First Defined | Sections Used |
|--------|---------|------|---------------|---------------|
| K | Number of arms | Scalar | model:L5 | All |
| k | Arm index | Index | model:L5 | All |
| t | Time step | Index | model:L10 | All |

---

## Derived Quantities

| Symbol | Meaning | Formula | First Defined | Used In |
|--------|---------|---------|---------------|---------|
| R_t | Residual | Y_t - F_t | method:L5 | 4,5,6 |

---

## Algorithm-Specific Symbols

| Symbol | Meaning | Algorithm | Notes |
|--------|---------|-----------|-------|
| U_k(t) | Upper bound | LUCB | - |
| L_k(t) | Lower bound | LUCB | - |

---

## Potential Conflicts

| Symbol | Meaning 1 | Location 1 | Meaning 2 | Location 2 | Resolution |
|--------|-----------|------------|-----------|------------|------------|

---

## Conventions

- **Subscripts**: k for arm, t for time, i for observation
- **Superscripts**: (F) for judge-only, (R) for residual
- **Hats**: Estimated quantities
- **Stars**: Optimal quantities

---

## LaTeX Macros

```latex
\newcommand{\arms}{K}
\newcommand{\arm}{k}
```
