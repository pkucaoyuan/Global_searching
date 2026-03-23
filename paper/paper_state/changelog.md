# Changelog: Where to Search (GAINS)

**Last Updated**: 2026-03-23

---

## 2026-03-23 - Theory Alignment Pass (uniform-ball local perturbation + proof tightening)

- Replaced the Gaussian exploitation model in Sec 4.5 with the code-aligned
  uniform-ball perturbation model
  $\xi_{\mathrm{new}}=\xi^*+RU$,
  $R\sim\mathrm{Unif}[0,\lambda\sqrt d]$,
  $U\sim\mathrm{Unif}(\mathbb{S}^{d-1})$.
- Introduced the exact linearized local-perturbation constant
  $c_d(\lambda)=\frac{\lambda\sqrt d}{4\sqrt{\pi}}
  \frac{\Gamma(d/2)}{\Gamma((d+1)/2)}$
  and replaced the old $\eta/\sqrt{2\pi}$ formulas throughout theory.
- Updated the Sec 4.5 operator table and crossover proposition so the
  RS/LP comparison is stated only for the linearized regime, with the
  large-budget full-problem discussion downgraded to a remark.
- Tightened the location-scale proofs in Sec 4.3:
  covariance now uses an explicit Cauchy--Schwarz bound and Gaussian
  fourth moments, and gain remainders are written as
  $O_{K,d,L_t}(g_t^2 h)$.
- Tightened the online theory in Sec 4.4 and Appendix B:
  Jensen equality now requires a common optimal allocation on the support
  of $\boldsymbol{\sigma}$, and the stopping proof now uses discrete
  marginals / average marginals instead of the old $G(1)=0$ shortcut.
- Clarified that the controller's variance statistic is an empirical
  proxy for low sensitivity, not an exact test of the theoretical
  $\sigma_t$ parameter.

## 2026-03-21 - General Local Search Theory Expansion

- Added Sec 4.5 to extend the scheduling theory from pure random search
  to a broader class of local search operators.
- Introduced the abstract assumptions (A1)--(A4), the general gain
  factorization theorem, and the offline / online corollaries.
- Performed multiple proof-tightening and paper-polish rounds; see git
  history for the detailed intermediate log.

## 2026-03-17 - OR Style Polish + Consistency Fix

- Polished the paper toward an OR-journal style and fixed several
  cross-reference and terminology inconsistencies.

## 2026-03-16 - Paper State Initialization + OR Restructure

- Created `paper_state/` and restructured the paper into the current OR
  framing with updated section layout.
