# Global Scheduling of Denoising Trajectories

**Leveraging Inference-Time Compute for Diffusion Models via Global Scheduling of Denoising Trajectories**

Diffusion models generate samples by traversing a denoising trajectory — a sequence of noise-reduction steps that progressively transforms random noise into data. Standard approaches allocate compute uniformly across steps, but this is wasteful: early steps that determine global structure benefit more from additional compute than late steps that refine local details.

We formalize this as a **budget allocation problem**: given *B* total function evaluations across *T* denoising steps, find the allocation that maximizes sample quality. We prove that the optimal allocation follows a **water-filling rule** that concentrates compute on steps where additional refinement yields the largest quality gains. On Stable Diffusion, EDM, and flow-based models, our method achieves **20-50% budget reduction** while maintaining the same quality as uniform-allocation benchmarks.

## Repository Structure

```
Global_searching/
├── README.md
├── code_repos/
│   └── diffusion-tts/          # Reference implementation (Ramesh & Mardani, 2025)
│       ├── edm/                # EDM backbone (class-conditional generation)
│       ├── sd/                 # Stable Diffusion backbone (text-to-image)
│       ├── flux/               # Flow model experiments
│       ├── paper/              # Original paper source
│       └── main.py             # Unified entry point
├── src/                        # Our implementation
│   ├── search/                 # Search algorithms
│   │   ├── global_search.py    # Global scheduler (budget allocation across steps)
│   │   ├── local_search.py     # Local search (within-step noise optimization)
│   │   ├── diffusion_tts_search.py  # Integration with diffusion-tts backends
│   │   └── base_search.py      # Abstract search interface
│   ├── models/                 # Model wrappers
│   │   ├── edm_model.py        # EDM model interface
│   │   └── base_model.py       # Abstract model interface
│   ├── verifiers/              # Quality scoring (verifiers)
│   │   ├── scorer_verifier.py  # Score-based verifier (CLIP, ImageReward, etc.)
│   │   ├── classifier_verifier.py  # Classifier-based verifier
│   │   └── base_verifier.py    # Abstract verifier interface
│   ├── evaluation/             # Evaluation metrics (FID, IS)
│   │   └── metrics.py
│   ├── pipeline/               # End-to-end sampling pipeline
│   │   └── sampling_pipeline.py
│   └── utils/                  # Configuration and utilities
│       ├── config.py
│       └── nfe_counter.py      # NFE budget tracking
├── scripts/                    # Experiment scripts
│   ├── run_diffusion_tts_experiment.py  # Main experiment runner
│   ├── run_pipeline.py         # Pipeline runner
│   ├── run_baseline.py         # Baseline comparison
│   ├── download_models.sh      # Model download helper
│   └── download_classifiers.py # Classifier download helper
├── configs/                    # Experiment configurations (YAML)
│   ├── cifar10_baseline.yaml
│   └── imagenet64_diffusion_tts.yaml
└── paper/                      # Paper source (LaTeX)
    ├── main.tex                # Main document
    ├── main.bib                # Bibliography
    ├── sections/               # Paper sections
    │   ├── introduction.tex
    │   ├── preliminaries.tex
    │   ├── framework.tex       # Two-level framework
    │   ├── algorithm.tex       # GAINS algorithm + theory
    │   ├── general_local_search.tex  # General local search extension
    │   ├── experiments.tex
    │   ├── conclusion.tex
    │   ├── appendix_mdp.tex    # MDP formulation
    │   └── appendix_proofs.tex # Proof details
    ├── figures/
    └── literature/
```

## Key Ideas

**Two-Level Framework.** We decompose noise trajectory search into:
1. **Local search** (within-step): at each denoising step, draw multiple noise candidates and select the best via a verifier score.
2. **Global scheduler** (across-step): allocate the total NFE budget across steps to maximize end-to-end sample quality.

**GAINS Algorithm.** Our Global Allocation for Inference-time Noise Search (GAINS) solves the budget allocation via a water-filling rule derived from the marginal gain structure of local search.

**Theoretical Guarantees.** We prove optimality of the water-filling allocation under a concave marginal gain model, and extend the theory to general local search operators (uniform-ball search, Gaussian perturbation, zeroth-order optimization).

## Quick Start

```bash
# Clone with submodule
git clone https://github.com/pkucaoyuan/Global_searching.git
cd Global_searching

# Set up environment
conda env create -f code_repos/diffusion-tts/environment.yml -n diffusion-tts
conda activate diffusion-tts

# Run experiment (EDM + global search)
python scripts/run_diffusion_tts_experiment.py \
    --backend edm \
    --scorer imagenet \
    --method eps_greedy \
    --search_budget 100
```

## Reference

This project builds on the Noise Trajectory Search framework by [Ramesh & Mardani (2025)](https://arxiv.org/abs/2506.03164). The `code_repos/diffusion-tts/` directory contains their reference implementation.
