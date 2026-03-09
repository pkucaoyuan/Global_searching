# Dependencies: Global Scheduling of Noise Trajectory Search

**Last Updated**: 2026-03-08

---

## Conceptual Dependencies

```
Diffusion Model Basics
    ↓
Inference-Time Scaling (motivation)
    ↓
Noise Trajectory Search (problem)
    ↓
Two-Level Framework
    ├── Local Operator L_t (abstraction)
    └── Global Scheduler G (formulation)
            ↓
    ┌───────┴───────┐
    │               │
Offline         Online
Profiling       Control
    │               │
    └───────┬───────┘
            ↓
    Algorithm 1
            ↓
    Experiments
```

---

## Methodological Dependencies

| Component | Depends On | Used By |
|-----------|------------|---------|
| Local operator L_t | Base sampler F_θ, Verifier v | Global scheduler G |
| Global scheduler G | Local operators {L_t}, Budget B | Algorithm 1 |
| Offline allocation {K_t} | Profiling data | Online controller |
| Online controller | {K_t}, gain/variance signals | Algorithm 1 |
| Early stopping | Historical gains G, variances V | Algorithm 1 |

---

## Experimental Dependencies

| Experiment | Depends On | Validates |
|------------|------------|-----------|
| SD Scaling (4.2) | SD model, Brightness/Compressibility verifiers | Global scheduling improves SD |
| EDM Scaling (4.3) | EDM model, same verifiers | Global scheduling improves EDM |
| Ablation (4.4) | SD model, NFE=400 | Online control adds value |
| Larger Prompts (4.5) | 20 prompts × 10 repeats | Robustness across prompts |
| Local Operators (4.6) | Zero-order + Random operators | Scheduler is modular |

---

## Symbol Dependencies

| Symbol | Depends On | Required For |
|--------|------------|--------------|
| x_{t-1} | x_t, ε_t, F_θ | Trajectory generation |
| s_t^{(j)} | v, x_0 prediction | Gain computation |
| g_t^{(j)} | s_t^{(j)}, s_t^{(j-1)} | Early stopping decision |
| Var_t^{(j)} | {s_{t,cand}^{(j,i)}} | Early stopping decision |
| K̂_t | K_t, online decisions | Budget tracking |

---

## Assumption Chain

1. **A1**: Pretrained diffusion model F_θ available
   - Required by: All experiments

2. **A2**: Verifier v provides meaningful quality signal
   - Required by: Local search, offline profiling

3. **A3**: Timestep sensitivity varies across trajectory
   - Required by: Global scheduling motivation
   - Validated by: Offline profiling shows SD ≠ EDM patterns

4. **A4**: Fixed total NFE budget B
   - Required by: Problem formulation
   - Enforced by: Algorithm 1

---

## If-Then Impact Analysis

| If Changed | Then Impact |
|------------|-------------|
| Different F_θ (model) | Need new offline profiling |
| Different v (verifier) | Need new offline profiling |
| Different B (budget) | Scale K_t proportionally |
| Different local operator | Algorithm 1 still works (modular) |
