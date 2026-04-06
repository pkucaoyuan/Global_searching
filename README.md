# Global Scheduling of Denoising Trajectories

**Leveraging Inference-Time Compute for Diffusion Models via Global Scheduling of Denoising Trajectories**

Diffusion models generate samples by traversing a denoising trajectory — a sequence of noise-reduction steps that progressively transforms random noise into data. Standard approaches allocate compute uniformly across steps, but this is wasteful: early steps that determine global structure benefit more from additional compute than late steps that refine local details.

We formalize this as a **budget allocation problem**: given *B* total function evaluations across *T* denoising steps, find the allocation that maximizes sample quality. We prove that the optimal allocation follows a **water-filling rule** that concentrates compute on steps where additional refinement yields the largest quality gains. On Stable Diffusion, EDM, and flow-based models, our method achieves **20-50% budget reduction** while maintaining the same quality as uniform-allocation benchmarks.

## Repository Structure

```
Global_searching/
├── README.md
├── code_repos/
│   └── diffusion-tts/              # Main codebase (forked from Ramesh & Mardani, 2025)
│       ├── main.py                 # Unified entry point for all backends
│       ├── edm/                    # EDM backbone (class-conditional generation)
│       │   ├── main.py             # EDM sampling & search
│       │   ├���─ generate.py         # Image generation
│       │   ├── scorers.py          # Quality scoring (ImageNet classifier, etc.)
│       │   ├── dnnlib/             # Deep learning utilities
│       │   └── torch_utils/        # PyTorch utilities
│       ├── sd/                     # Stable Diffusion backbone (text-to-image)
│       │   ├── main.py             # SD sampling & search
│       │   ├── scorers.py          # Quality scoring (CLIP, brightness, etc.)
│       │   └── diffusers/          # Diffusers integration
│       ├── flux/                   # Flow model experiments
│       │   ├── flow_experiment.py  # Flow-based search experiments
│       │   ├── real_flow_experiment.py
│       │   ├── stochastic_flow_scheduler.py
│       │   └── scorers.py
│       ├── paper/                  # Original paper source
│       └── docs/                   # Documentation
└── paper/                          # Our paper source (LaTeX)
    ├── main.tex                    # Main document
    ├── main.bib                    # Bibliography
    ├── sections/                   # Paper sections
    │   ├── introduction.tex
    │   ├── preliminaries.tex
    │   ├── framework.tex           # Two-level framework
    │   ├── algorithm.tex           # GAINS algorithm + theory
    ��   ├── general_local_search.tex
    │   ├── experiments.tex
    │   ├── conclusion.tex
    │   ├── appendix_mdp.tex        # MDP formulation
    │   └── appendix_proofs.tex     # Proof details
    ├── figures/
    └── literature/
```

## Key Ideas

**Two-Level Framework.** We decompose noise trajectory search into:
1. **Local search** (within-step): at each denoising step, draw multiple noise candidates and select the best via a verifier score.
2. **Global scheduler** (across-step): allocate the total NFE budget across steps to maximize end-to-end sample quality.

**GAINS Algorithm.** Our Global Allocation for Inference-time Noise Search (GAINS) solves the budget allocation via a water-filling rule derived from the marginal gain structure of local search.

**Theoretical Guarantees.** We prove optimality of the water-filling allocation under a concave marginal gain model, and extend the theory to general local search operators (uniform-ball search, Gaussian perturbation, zeroth-order optimization).

## Core Implementation

The global search logic is implemented inside `code_repos/diffusion-tts/edm/main.py` (~1500 lines), which contains all sampling methods and the budget scheduling mechanism:

| Code Location | Description |
|---|---|
| `edm/main.py` L26-34 | `SamplingMethod` enum: `NAIVE`, `ZERO_ORDER`, `EPS_GREEDY`, `EPS_GREEDY_1`, `EPS_GREEDY_ONLINE`, `MCTS`, `BEAM_SEARCH` |
| `edm/main.py` L36-61 | `SamplingParams`: search hyperparameters including budget scheduler params (`extra_budget`, `tau0`, `alpha`, `probe_M`) |
| `edm/main.py` L733-975 | **`EPS_GREEDY` / `EPS_GREEDY_1`**: per-step local search with fixed K allocation (uniform budget) |
| `edm/main.py` L1163-1417 | **`EPS_GREEDY_ONLINE`**: global scheduling — dynamic budget allocation with probe-based gain estimation, early stopping, and threshold decay |
| `edm/scorers.py` | Verifier implementations (ImageNet classifier, compressibility) |
| `sd/main.py` | Stable Diffusion backend (dispatches to same search methods) |
| `sd/scorers.py` | SD verifiers (CLIP, brightness, compressibility) |
| `flux/flow_experiment.py` | Flow model extension (`FlowNoiseSearch` class) |

**Key mechanism** (`EPS_GREEDY_ONLINE`): at each denoising step, the scheduler decides how many local search iterations to run based on remaining budget and estimated marginal gain. It uses `probe_M` candidate noises to estimate per-step gain, compares against a decaying threshold `tau(t)`, and stops early when gains fall below the threshold — implementing the water-filling allocation from the paper.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/pkucaoyuan/Global_searching.git
cd Global_searching

# Set up environment
conda env create -f code_repos/diffusion-tts/environment.yml -n diffusion-tts
conda activate diffusion-tts

# Run experiment (EDM + epsilon-greedy search)
cd code_repos/diffusion-tts
python main.py --backend edm --scorer imagenet --method eps_greedy

# Run experiment (Stable Diffusion + zero-order search)
python main.py --backend sd --scorer clip --method zero_order --prompt "A beautiful landscape"
```

## Reference

This project builds on the Noise Trajectory Search framework by [Ramesh & Mardani (2025)](https://arxiv.org/abs/2506.03164). The `code_repos/diffusion-tts/` directory contains their reference implementation with our modifications.
