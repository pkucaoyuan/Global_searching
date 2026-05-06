# GAINS: Global Adaptive Inference-time Noise Scheduling

PyTorch implementation accompanying the NeurIPS 2026 submission *"Global Adaptive Inference-time Noise Scheduling for Diffusion Models"*. This repository contains code, scripts, and experiment configurations for reproducing the EDM, Stable Diffusion, and flow-based (PixArt-Sigma) results in the paper.

The implementation builds on the noise-trajectory-search (NTS) baseline of Ramesh & Mardani (2025) and extends it with: (i) an offline sensitivity-profiling stage that yields a non-uniform per-timestep allocation $\{K_t\}$, and (ii) an online controller that performs windowed early stopping driven by recent score gain and within-step candidate variance, redistributing saved NFE under a strict total-budget constraint.

## Repository layout

```
.
├── main.py                       # Top-level dispatcher (EDM / SD backends)
├── prompts.csv                   # Prompt set for SD experiments
├── eps1_gain_probe.py            # Offline sensitivity profiling tool (per-step gain)
├── eps1_gain_plot.py             # Plotting utilities for offline profiles
├── SEARCH_METHODS_ANALYSIS.md    # Detailed walkthrough of each search method
├── edm/                          # EDM backend (class-conditional ImageNet)
│   ├── main.py                   # EDM sampling loop with all search methods
│   ├── scorers.py                # Brightness / Compressibility / ImageNet
│   └── ...
├── sd/                           # Stable Diffusion backend (text-to-image)
│   ├── main.py
│   ├── scorers.py                # Brightness / Compressibility / CLIP
│   └── diffusers/                # Vendored diffusers fork
├── flux/                         # Flow-based experiments (PixArt-Sigma)
│   ├── stochastic_flow_scheduler.py   # ODE-to-SDE conversion
│   ├── real_flow_experiment.py        # Naive vs GAINS scheduling on flows
│   └── EXPERIMENT_REPORT.md
├── literature/                   # Reference papers in arxiv source form
├── environment.yml
└── LICENSE.md
```

## Requirements

* Linux with one high-end NVIDIA GPU (we used A100 80GB).
* Python 3.10, PyTorch ≥ 2.1, CUDA 12.1.
* Install via Miniconda:
  ```bash
  conda env create -f environment.yml -n diffusion-tts
  conda activate diffusion-tts
  ```

## Sampling methods

The `--method` flag selects the search strategy. All methods consume a fixed total NFE budget; methods that perform per-step search use an inner `K_t` candidate count.

| Method | Description | Source |
|--------|-------------|--------|
| `naive` | Baseline sampler, no search | — |
| `rejection` | Best-of-$N$ at every timestep | — |
| `beam` | Beam search over noise candidates | — |
| `mcts` | Monte Carlo Tree Search | — |
| `zero_order` | Zero-order local perturbation | Tang et al. 2024 |
| `eps_greedy` | $\epsilon$-greedy noise search, uniform $K_t$ | Ramesh & Mardani 2025 |
| `epsilon_1` | **GAINS offline**: high/low-value region split with $K_1, K_2$ | This work |
| `epsilon_online` | **GAINS online**: offline allocation + windowed early stopping with budget redistribution | This work |

## Quick start

Generate a single image with a chosen backend, scorer, and search method:

```bash
# Stable Diffusion + brightness reward + uniform baseline
python main.py --backend sd --scorer brightness --method naive \
    --prompt "A beautiful landscape"

# EDM + ImageNet classifier reward + zero-order local search
python main.py --backend edm --scorer imagenet --method zero_order

# GAINS offline (epsilon_1) on SD with compressibility reward
python main.py --backend sd --scorer compressibility --method epsilon_1 \
    --K1 25 --K2 15 --prompt_csv prompts.csv

# GAINS online (epsilon_online) with strict total budget
python main.py --backend edm --scorer compressibility --method epsilon_online \
  --num_steps 18 --K1 11 --eps 0.4 --lambda_ 0.15 --total_budget 144 --n_runs 20 \
  --N 4 --high_slack 2  --thresh_gain_coef 0.3 --revert_on_negative \
  --thresh_var_coef 0.7 \
```

### Common flags

| Flag | Meaning |
|------|---------|
| `--backend`              | `sd` or `edm` |
| `--scorer`               | `brightness`, `compressibility`, `clip` (SD), `imagenet` (EDM) |
| `--method`               | See table above |
| `--prompt` / `--prompt_csv` | Single prompt string or CSV with prompts (SD only) |
| `--num_steps`            | Number of denoising steps (SD only; default 50) |
| `--seed`, `--n_runs`, `--repeat_per_prompt` | Seeding and averaging |
| `--K`, `--N`, `--lambda_`, `--eps` | Local-operator hyperparameters |
| `--K1`, `--K2`           | Offline split |
| `--total_budget`         | Strict NFE budget (`epsilon_online`) |
| `--high_slack`           | Watch-region slack for early stopping (`epsilon_online`) |
| `--thresh_gain_coef`, `--thresh_var_coef` | Online thresholds $\beta_g$, $\beta_\sigma$ |
| `--revert_on_negative`   | Keep previous pivot when an iteration's gain is negative |
| `--log_gain`             | Dump per-timestep gains for analysis |

Run `python main.py --help` for the full list.

## License

Released under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citation

This is anonymous code accompanying a NeurIPS 2026 submission. A citation block will be added on acceptance.

For the underlying noise-trajectory-search baseline that this work extends:

```
@misc{ramesh2025testtimescalingdiffusionmodels,
  title  = {Test-Time Scaling of Diffusion Models via Noise Trajectory Search},
  author = {Vignav Ramesh and Morteza Mardani},
  year   = {2025},
  eprint = {2506.03164},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2506.03164}
}
```
