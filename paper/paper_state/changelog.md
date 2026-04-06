# Changelog: Where to Search (GAINS)

**Last Updated**: 2026-03-23

---

## 2026-03-24 - verify-proof: Post-Revision Full Audit

- Ran `/verify-proof` across all 14 registered theory results after writing revision.
- **Status**: All 14 results ✅ mathematically correct. No errors found.
- **Detailed step-by-step verification** of:
  - `prop:taylor`: Taylor expansion, remainder bound, conditional expectation — all correct
  - `thm:loc-scale` (i/ii/iii): Projection decomposition, Cauchy-Schwarz variance bound,
    elementary max inequality, $\mathbb{E}[\|\xi\|^4]=d(d+2)$ verified
  - `prop:offline` (i/ii/iii): Lagrangian/KKT, water-filling monotonicity, perturbation argument
  - `prop:online` (i/ii): Convexity of $V^*$, Jensen inequality, chord bound, discrete KKT
  - `thm:general-gain`: (A1)-(A4) usage, rotational equivariance argument with explicit $Q$ construction
  - `cor:offline-general`, `cor:online-general`: Curvature dichotomy, linear vs concave cases
  - `prop:crossover`: $c_d(\lambda)$ formula verified analytically and numerically
  - All 3 examples (RS, $\epsilon$-greedy, LP): (A1)-(A4) verification, gain bounds
- **3 minor issues noted** (none are errors):
  1. `thm:loc-scale(ii)`: $O_d$ subscript could be $O_{d,L_t}$ for consistency (known)
  2. `prop:offline(i)` main text: "$\sigma_t = \sigma_t$" is tautological phrasing
  3. `results.md` uses stale label `ex:zero-order`; actual label is `ex:local-perturbation`
- **$c_d(\lambda)$ independently re-confirmed** via Beta function identity
- **2 `\approx` found** in `ex:eps-greedy` informal text (acceptable — not in formal claims)

## 2026-03-23 (evening) - verify-proof: Full Theory Audit

- Ran `/verify-proof` across all 14 registered theory results.
- **Status**: All 14 results ✅ logically correct; $c_d(\lambda)$ formula independently
  confirmed via Beta function calculation.
- **3 stale state-doc entries fixed**:
  - `asm:local-search` (A2): "concavity" → "non-accelerating marginals"
  - `thm:general-gain`: $O_K$ → $O_{K,d,L_t}$; verified date updated
  - `prop:crossover`: crossover formula updated from old $\tilde\Theta(d/\eta)$
    to $\tilde\Theta(c_d(\lambda)^{-1})=\tilde\Theta(\lambda^{-1})$ (for fixed $d$)
- **2 minor notation issues noted** (not blocking):
  - `thm:loc-scale(ii)` variance uses $O(g_t^3 h^{3/2})$ without $d$-tracking
    (inconsistent with $O_{K,d,L_t}$ used in general results)
  - `prop:offline Part(ii)` statement uses continuous $a_K'$ while appendix
    proof now uses discrete $\Delta a_K$ (style inconsistency, both valid)

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
