# Flow Model Experiment Implementation Report

## Overview

This report summarizes the implementation of noise trajectory search for flow-based models (PixArt-Sigma), extending our global scheduling method from diffusion models to rectified flow models.

## Implementation Status

### Completed

1. **Stochastic Flow Scheduler** (`stochastic_flow_scheduler.py`)
   - ODE-to-SDE conversion for flow models
   - NonSingular noise schedule: `g(t) = α * √t`
   - Score computation from velocity: `score(x,t) = (-(1-t)*v - x) / t`
   - Based on Dockhorn et al. (arXiv:2410.02217) and Kim et al. (arXiv:2503.19385)

2. **Real Flow Experiment** (`real_flow_experiment.py`)
   - Full experiment framework for PixArt-Sigma model
   - Two scheduling strategies:
     - **Naive**: Uniform K allocation across all timesteps
     - **Online (Ours)**: Non-uniform allocation (more K at middle timesteps 4-14)
   - Brightness scorer for evaluation
   - Comparison framework with multiple seeds

3. **Key Algorithm Components**
   ```python
   # SDE step for flow models
   x_{t-dt} = x_t + dt * v(x_t, t) + sqrt(|dt|) * g(t) * noise

   # Noise coefficient (NonSingular schedule)
   g(t) = noise_scale * sqrt(t)

   # K allocation (Online scheduler)
   - Steps 0-3:   K = K_base / 1.5  (low importance)
   - Steps 4-14:  K = K_base * 1.5  (high importance)
   - Steps 15-19: K = K_base / 1.5  (low importance)
   ```

### Blocked

**Model Download Issue**
- Model: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (~10GB)
- Status: Partially downloaded (1.2GB / 10GB)
- Missing: `text_encoder/model-00002-of-00002.safetensors` (~5GB)
- Cause: Insufficient disk space (need ~10GB, only 1.2GB available)

## Theoretical Foundation

### ODE-to-SDE Conversion for Flow Models

Flow models use deterministic ODE sampling:
```
dx = v(x, t) dt
```

We convert to SDE for stochastic search:
```
dx = [v(x,t) + (g²/2) * score(x,t)] dt + g(t) dW
```

Where score is derived from velocity using the linear interpolant relationship:
```
x_t = (1-t) * x_0 + t * ε
v = dx/dt = ε - x_0
score(x,t) = (-(1-t) * v - x) / t
```

### Why This Works for Search

1. **Stochastic injection** creates multiple candidate trajectories
2. **Middle timesteps** (t ≈ 0.3-0.7) form semantic content → higher search value
3. **Our scheduler** allocates more budget to high-value regions
4. **Online control** enables early stopping when gains plateau

## Files

| File | Description | Status |
|------|-------------|--------|
| `flux/stochastic_flow_scheduler.py` | ODE-to-SDE scheduler | Complete |
| `flux/real_flow_experiment.py` | Main experiment script | Complete |
| `flux/flow_experiment.py` | Mock experiment (for testing) | Complete |

## Next Steps

1. **Resolve disk space** - Need ~10GB free space for model download
2. **Complete model download** - Download remaining text_encoder weights
3. **Run experiments** - Execute comparison with NFE budgets [100, 200, 400]
4. **Update paper** - Fill in Table 6 (Flow Model Results) with real data

## Command to Run (Once Model Downloaded)

```bash
CUDA_VISIBLE_DEVICES=0 /home/lg/.conda/envs/qzh/bin/python \
    flux/real_flow_experiment.py \
    --model_id PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
    --run_comparison \
    --scorer brightness
```

## References

- Dockhorn et al. "Stochastic Interpolants with Data-Dependent Couplings" (arXiv:2410.02217)
- Kim et al. "Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing" (arXiv:2503.19385)
- PixArt-Sigma: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS

---
*Generated: 2024-03-09*
