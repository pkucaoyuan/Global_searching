# Changelog: Where to Search (GAINS)

**Last Updated**: 2026-03-21

---

## 2026-03-21 — §4.5 General Local Search Theory + Verify-Proof + Refine-Theory + Polish

- **polish-paper** (2 rounds, 7 modifications, converged):
  - VERB_STRENGTHEN (2): "develops"→"extends", "enables"→"permits"
  - COMPRESS (3): intro (-1 sentence), Jensen remark (-2 lines), crossover remark (-1 line)
  - FLOW (2): causal restructuring in scheduling intro; parallel structure in crossover remark
  - No AI words, no zombie nouns, no passive voice, no S-V gap issues detected
  - Compilation: 32 pages, 0 errors, 0 warnings

- **refine-theory** (2 iterations, 11 issues fixed):
  - Fixed "three assumptions" → "four assumptions" count
  - Fixed intro "satisfying (A3)" → "satisfying Assumption 4.X"
  - Replaced informal "(const depending on η,τ)" with proper prose
  - Improved ε-greedy (A2) concavity argument (both marginals non-increasing)
  - Clarified Jensen gap curvature mechanism (sharper allocation switching)
  - Added measure-zero justification for ZO (A1) strict monotonicity
  - Qualified crossover $o(1)$ with "$d/\eta\to\infty$" limit
  - Weakened cor:offline-general Part (ii)/(iii) to match concavity level
  - Added $K_t \ge 1$ constraint to eq:alloc-general and eq:oracle-general
  - Fixed "(A1)--(A3)" → "(A1)--(A4)" in body text and table caption
  - Compilation: 32 pages, 0 errors, 0 warnings

- **New section**: §4.5 Extension to General Local Search Operators (general_local_search.tex)
  - Assumption (A1–A4): strict monotonicity, concavity, sensitivity scaling, rotational equivariance
  - Theorem (General Gain Factorization): $G_t^{\mathcal{L}}(K) = \sigma_t \phi_K + O_K(g_t^2 h)$
  - Corollary (Offline Water-Filling, General): extends prop:offline to general operators
  - Corollary (Online Jensen Gap, General): extends prop:online to general operators
  - Proposition (Crossover): ZO vs RS crossover at $K^* = \tilde{\Theta}(d/\eta)$
  - 4 Examples: Random search, ε-greedy, Zero-order, Langevin MCMC
  - Summary table (tab:phi-summary)
- **verify-proof Round 3** (3 final issues fixed):
  - Weakened (A2) from strict concavity to concavity (ZO has linear $\phi_K = \eta K/d$)
  - Fixed table caption: (A1)--(A3) → (A1)--(A4)
  - Fixed prop:crossover Part(ii): removed false $\sqrt{d}$ saturation claim
  - Added degenerate LP note to cor:offline-general for linear $\phi_K$
- **verify-proof Rounds 1-2** (10 issues found, all fixed):
  - Added (A4) rotational equivariance (was hidden assumption in proof)
  - (A1) strengthened to strict monotonicity (needed by prop:offline)
  - Fixed remainder bound: $O(g_t^2 h)$ → $O_K(g_t^2 h)$ throughout
  - Fixed ZO example: removed incorrect $\sqrt{d}$ cap, derived exact $\phi_K = \eta K/d$
  - Fixed Langevin (A3) verification: clarified steady-state mean argument
  - Removed all `\approx` from mathematical statements (4 instances)
- **Compilation**: 31 pages, 0 errors, 0 warnings

## 2026-03-17 — OR Style Polish + Consistency Fix

- **Polish Round 1**: 131 modifications across 9 files
  - VERB_STRENGTHEN (15), COMPRESS (24), META_REMOVE (6), NAMED_ATTRIBUTION (5)
  - NO_POINTER_CHAIN (5), PROOF_SIGNPOST (6), PROOF_FLOW (2), MANAGERIAL_POINT (1)
- **Consistency fixes** (3 critical):
  - `\cref{sec:method-global}` → `\cref{sec:framework}` (broken ref → "??")
  - Jensen gap "recovers" → "can partially recover" (overstatement)
  - $\sigma_t$ footnote disambiguation (SGM noise vs score variance)
- **OR terminology**: "baseline" → "benchmark method" / "uniform allocation" (15 occurrences)
- **Compilation**: 24 pages, 0 errors, 0 warnings

## 2026-03-17 — Literature Acquisition (5 papers)

- **Downloaded + summarized** (arXiv source + SUMMARY_*.md):
  - Gao, Zha, Zhou (2024) — Reward-Directed Diffusion via q-Learning
  - Tang, Zhao (2024/2025) — Score-based Diffusion via SDEs (Tutorial)
  - Tang, Zhao (2024) — Contractive DPMs
  - Jia, Zhou (2024) — RL for Jump-Diffusions
  - Aolaritei, Van Parys, Lam, Jordan (2025) — Optimal Importance Sampling
- **9 BibTeX entries** added to main.bib
- **"Connections to OR" paragraph** added in §1.1

## 2026-03-16 — Paper State Initialization + OR Restructure

- **Created** paper_state/ with 10 state docs
- **Restructured** from ML conference (5 sections) to OR journal (6 sections + 2 appendices)
- **New sections**: abstract.tex, preliminaries.tex (3 Definitions), framework.tex
- **Expanded**: introduction (OR motivation + §1.1 Related Lit), conclusion (operational guidelines + limitations + future)
- **OR style guide** extracted from OPRE reference paper (102-page PDF)
- **Removed**: related_work.tex, methodology.tex, method_*.tex (content preserved in new structure)
