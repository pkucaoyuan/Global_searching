# Changelog: Where to Search (GAINS)

**Last Updated**: 2026-03-17

---

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
