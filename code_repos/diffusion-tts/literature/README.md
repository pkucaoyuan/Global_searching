# Literature Collection

Literature sources and summaries for "Where to Search: Global Scheduling of Noise Trajectory Search in Diffusion Models"

## ArXiv Source Files

| Key | arXiv | Title | Summary |
|-----|-------|-------|---------|
| `ho2020denoising` | 2006.11239 | Denoising Diffusion Probabilistic Models (DDPM) | [SUMMARY](arxiv_sources/ho2020denoising/SUMMARY_ho2020denoising.md) |
| `song2021scorebased` | 2011.13456 | Score-Based Generative Modeling through SDEs | [SUMMARY](arxiv_sources/song2021scorebased/SUMMARY_song2021scorebased.md) |
| `karras2022elucidating` | 2206.00364 | Elucidating the Design Space (EDM) | [SUMMARY](arxiv_sources/karras2022elucidating/SUMMARY_karras2022elucidating.md) |
| `rombach2022high` | 2112.10752 | Latent Diffusion Models (Stable Diffusion) | [SUMMARY](arxiv_sources/rombach2022high/SUMMARY_rombach2022high.md) |
| `dhariwal2021diffusion` | 2105.05233 | Diffusion Models Beat GANs | [SUMMARY](arxiv_sources/dhariwal2021diffusion/SUMMARY_dhariwal2021diffusion.md) |
| `ho2022classifier` | 2207.12598 | Classifier-Free Diffusion Guidance | [SUMMARY](arxiv_sources/ho2022classifier/SUMMARY_ho2022classifier.md) |
| `ma2024inference` | 2501.09732 | Inference-Time Scaling beyond Denoising Steps | [SUMMARY](arxiv_sources/ma2024inference/SUMMARY_ma2024inference.md) |
| `graves2016adaptive` | 1603.08983 | Adaptive Computation Time for RNNs | [SUMMARY](arxiv_sources/graves2016adaptive/SUMMARY_graves2016adaptive.md) |
| `ramesh2025nts` | 2506.03164 | Test-Time Scaling via Noise Trajectory Search | [SUMMARY](arxiv_sources/ramesh2025nts/SUMMARY_ramesh2025nts.md) |
| `kim2025rbf` | 2503.19385 | Flow Models + Rollover Budget Forcing | [SUMMARY](arxiv_sources/kim2025rbf/SUMMARY_kim2025rbf.md) |
| `dockhorn2024stochastic` | 2410.02217 | Stochastic Sampling from Deterministic Flow Models | [SUMMARY](arxiv_sources/dockhorn2024stochastic/SUMMARY_dockhorn2024stochastic.md) |
| `uehara2025tutorial` | 2501.09685 | Inference-Time Alignment Tutorial | [SUMMARY](arxiv_sources/uehara2025tutorial/SUMMARY_uehara2025tutorial.md) |
| `psi_sampler2025` | 2506.01320 | Ψ-Sampler: SMC-Based Reward Alignment | [SUMMARY](arxiv_sources/psi_sampler2025/SUMMARY_psi_sampler2025.md) |

## Categories

### Foundational Diffusion Models
- `ho2020denoising` - DDPM, discrete timestep formulation
- `song2021scorebased` - Score SDE, continuous-time view, predictor-corrector
- `karras2022elucidating` - EDM, modular design, timestep importance analysis

### Text-to-Image and Guidance
- `rombach2022high` - Latent Diffusion / Stable Diffusion
- `dhariwal2021diffusion` - ADM, classifier guidance
- `ho2022classifier` - Classifier-free guidance

### Inference-Time Scaling (Most Related)
- `ma2024inference` - Search framework with verifiers + algorithms
- `ramesh2025nts` - **Concurrent work**: epsilon-greedy noise trajectory search, uniform K
- `kim2025rbf` - **Concurrent work**: Rollover Budget Forcing, online-only adaptation
- `uehara2025tutorial` - Tutorial on reward-guided generation and verifiers
- `psi_sampler2025` - SMC-based particle sampling for reward alignment

### ODE-to-SDE Conversion (Flow Model Compatibility)
- `dockhorn2024stochastic` - Stochastic sampling from deterministic flows
- `kim2025rbf` - SDE conversion + VP interpolant for flow models

### Adaptive Computation
- `graves2016adaptive` - ACT for RNNs, per-step pondering

## Key Comparisons with Our Work

### vs. Ramesh & Mardani (2506.03164)
| Aspect | Their Work | Ours |
|--------|------------|------|
| Timestep handling | Implicit via epsilon | Explicit global scheduler |
| Compute allocation | Uniform K | Non-uniform, profile-based |
| Profiling | None | Offline profiling |
| Early stopping | None | Online control |

### vs. Kim et al. (2503.19385)
| Aspect | Their Work (RBF) | Ours |
|--------|------------------|------|
| Initial budget | Uniform | Profile-based |
| Adaptation | Purely reactive/online | Offline + online |
| Threshold | Global best r* | Step-specific thresholds |
| Architecture | Single-level | Two-level (local + global) |

## Coverage Status

- **Downloaded:** 10/10 key papers
- **Summarized:** 10/10 papers
- **Integrated into paper:** References added to main.bib, related work updated

---

*Last updated: 2026-03-08*
