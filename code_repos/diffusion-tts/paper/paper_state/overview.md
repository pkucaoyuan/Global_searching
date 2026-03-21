# Paper Overview: Where to Search — Global Scheduling of Noise Trajectory Search in Diffusion Models

**Created**: 2026-03-16
**Last Updated**: 2026-03-16
**Target Venue**: Operations Research (OPRE)
**Status**: Draft (converting from ML conference style to OR journal style)

---

## One-Sentence Summary

We formalize inference-time noise refinement in diffusion models as a two-level optimization problem — local noise search and global timestep scheduling — and propose GAINS, a budget-aware algorithm combining offline profiling with online control that achieves the same sample quality with 20–50% fewer function evaluations.

---

## Current State

| Aspect | Status | Notes |
|--------|--------|-------|
| Content (L0) | ✅ | Core framework + 2 propositions + 6 experiment tables + MDP appendix |
| Structure (L1) | ⚠️ | ML conference format; needs restructuring for OR journal |
| Consistency (L2) | ⚠️ | Not yet checked |
| Style (L3) | ❌ | Currently ML style; needs OR journal conversion |
| Language (L4) | ❌ | Needs OR-style polish |

---

## Section Map

| Section | File | Purpose | Key Results |
|---------|------|---------|-------------|
| Abstract | main.tex | Summary | - |
| 1. Introduction | introduction.tex | Motivation, two-level view, contributions | 3 contributions |
| 2. Related Work | related_work.tex | 3 paragraphs: inference scaling, noise search, adaptive compute | 20+ citations |
| 3. Methodology | methodology.tex → method_framework.tex + method_algorithm.tex + method_theory.tex | - | - |
| 3.1 Two-Level Framework | method_framework.tex | Local operator + global scheduler formalization | Eq 1-2, Fig 1 |
| 3.2 GAINS Algorithm | method_algorithm.tex | Offline + online scheduling | Alg 1 |
| 3.3 Theoretical Motivation | method_theory.tex | Gain factorization, water-filling, Jensen gap | Prop 1, Prop 2 |
| 4. Experiments | experiments.tex | SD + EDM + Flow + ablation | Tables 1-6, 6 subsections |
| 5. Conclusion | conclusion.tex | Summary (3 lines) | - |
| App A. MDP Formulation | appendix_mdp.tex | General MDP with action taxonomy | Table 7 (10 methods classified) |
| App B. Proofs | appendix_proofs.tex | Proofs of Props 1-2 | - |

---

## Key Contributions

1. **Two-level framework**: Separates local noise refinement from global timestep scheduling; unifies existing methods as special cases
2. **GAINS algorithm**: Offline profiling (learn where search matters) + online control (gain/variance early stopping with strict NFE budget)
3. **Empirical validation**: SD, EDM, flow models — consistent gains, 20-50% NFE savings at matched quality

---

## OR Conversion Requirements

| Issue | Current (ML) | Target (OR) | Priority |
|-------|-------------|-------------|----------|
| Venue framing | "image generation" | OR/OM applications (simulation, stochastic optimization) | P0 |
| Motivation | Computer vision focus | Operations research practitioners need simulators | P0 |
| Theory depth | 2 propositions (supporting) | Need stronger theoretical contribution | P0 |
| Experiment scope | Brightness/Compressibility verifiers | OR-relevant metrics (simulation accuracy, cost) | P1 |
| Writing style | Informal ML, short paper | Formal OR, long-form with detailed proofs | P1 |
| Related work | ML inference scaling focus | Connect to OR simulation, stochastic optimization | P1 |
| Conclusion | 3 lines | Need managerial insights, limitations, future work | P2 |

---

## Quick Links

- [Symbols](./symbols.md) - All notation
- [Results](./results.md) - All formal results
- [Framing](./framing.md) - Locked terminology
- [Figures & Tables](./figures_tables.md) - Visual elements
- [Changelog](./changelog.md) - Modification history
- [Review Responses](./review_responses.md) - Reviewer comments
- [Consistency Log](./consistency_log.md) - Check history
- [OR Style Guide](./or_style_guide.md) - OR writing conventions (from OPRE reference)

---

## Active Issues

| Priority | Issue | Location | Status |
|----------|-------|----------|--------|
| P0 | Reframe for OR audience (motivation, applications) | sec:intro | Open |
| P0 | Strengthen theoretical contribution for OR journal | sec:theory | Open |
| P1 | Convert to OPRE LaTeX template | main.tex | Open |
| P1 | Add OR-relevant experiments (simulation, scheduling) | sec:experiments | Open |
| P1 | Expand conclusion with managerial insights | sec:conclusion | Open |
| P2 | Add numbered equations per OR convention | Throughout | Open |
| P2 | Expand proofs from appendix sketch to full detail | appendix_proofs.tex | Open |

---

## Source Material

- **LaTeX source**: `code_repos/diffusion-tts/paper/`
- **OR style reference**: `OPRE-2024-11-1450.R1_Proof_hi (1).pdf` (102 pages, accepted at OPRE)
- **Figures**: `figures/fig_architecture.pdf`, `figures/fig_stopping_decision.pdf`
