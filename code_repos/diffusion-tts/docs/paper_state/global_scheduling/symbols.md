# Symbol Registry: Global Scheduling of Noise Trajectory Search

**Last Updated**: 2026-03-08
**Total Symbols**: 25+

---

## Core Symbols

| Symbol | Meaning | Type | First Defined | Sections Used |
|--------|---------|------|---------------|---------------|
| T | Total number of timesteps | Scalar | sec:method | 3, 4 |
| t | Timestep index | Index | sec:method | All |
| x_t | Latent at timestep t | Vector | sec:method | 3 |
| x_0 | Final sample | Vector | sec:method | 3 |
| c | Conditioning signal (class/prompt) | Vector | sec:method | 3 |
| F_θ | Base diffusion sampler | Function | sec:method | 3 |
| ε_t | Injected noise at timestep t | Vector | sec:method | 3 |
| v | Verifier function | Function | sec:method | 3 |
| B | Total NFE budget | Scalar | sec:method | 3, 4 |
| D_θ | Denoising network | Function | sec:method-global | 3 |
| x̂_0 | Predicted clean image | Vector | sec:method-global | 3 |
| α_t | Scheduler coefficient (signal) | Scalar | sec:method-global | 3 |
| σ_t | Scheduler coefficient (noise) | Scalar | sec:method-global | 3 |
| K | Number of candidate noises | Scalar | sec:method-global | 3 |

---

## Operator Symbols

| Symbol | Meaning | Type | First Defined | Sections Used |
|--------|---------|------|---------------|---------------|
| L_t | Local search operator at step t | Function | sec:method-global | 3 |
| G | Global scheduler | Function | sec:method-global | 3 |

---

## Algorithm-Specific Symbols

| Symbol | Meaning | Algorithm | Notes |
|--------|---------|-----------|-------|
| K_t | Offline budget allocation for step t | Alg 1 | Per-step budget |
| K̂_t | Realized iterations at step t | Alg 1 | Actual spent |
| s | Slack parameter | Alg 1 | Flexibility window |
| W_g | Gain window size | Alg 1 | For moving average |
| β_g | Gain threshold coefficient | Alg 1 | Early stop param |
| β_v | Variance threshold coefficient | Alg 1 | Early stop param |
| R | Remaining budget | Alg 1 | Dynamic tracking |

---

## Score & Statistics Symbols

| Symbol | Meaning | Formula | First Defined | Used In |
|--------|---------|---------|---------------|---------|
| s_t^{(j)} | Best score at step t after j iterations | - | sec:method-ours | 3 |
| g_t^{(j)} | Incremental gain | s_t^{(j)} - s_t^{(j-1)} | sec:method-ours | 3 |
| Var_t^{(j)} | Per-iteration variance | Var_i(s_{t,cand}^{(j,i)}) | sec:method-ours | 3 |
| g̃^{(j)} | Windowed mean gain | Mean(H) | Alg 1 | 3 |

---

## Sets & Collections

| Symbol | Meaning | Type | First Defined | Used In |
|--------|---------|------|---------------|---------|
| G | Historical gains | Set | Alg 1 | 3 |
| V | Historical variances | Set | Alg 1 | 3 |
| H | Recent gains (window) | Set | Alg 1 | 3 |

---

## Potential Conflicts

| Symbol | Meaning 1 | Location 1 | Meaning 2 | Location 2 | Resolution |
|--------|-----------|------------|-----------|------------|------------|
| G | Global scheduler | sec:method | Historical gains set | Alg 1 | Use calligraphic G for scheduler |

---

## Conventions

- **Subscripts**: t for timestep, i for candidate index, j for iteration
- **Superscripts**: (j) for iteration count
- **Hats**: Realized/estimated quantities (K̂_t)
- **Tildes**: Averaged quantities (g̃)
- **Calligraphic**: Operators and sets (L, G, H, V)

---

## LaTeX Macros

```latex
\newcommand{\timesteps}{T}
\newcommand{\timestep}{t}
\newcommand{\latent}{x}
\newcommand{\condition}{c}
\newcommand{\sampler}{F_\theta}
\newcommand{\noise}{\varepsilon}
\newcommand{\verifier}{v}
\newcommand{\budget}{B}
\newcommand{\localop}{\mathcal{L}}
\newcommand{\globalop}{\mathcal{G}}
```
