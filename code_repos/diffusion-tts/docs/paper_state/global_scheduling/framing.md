# Framing Document: Global Scheduling of Noise Trajectory Search

**Last Updated**: 2026-03-08
**Status**: Initial Draft

---

## Locked Terminology

| Term | Definition | Do NOT Use |
|------|------------|------------|
| Global scheduling | Allocating search compute across timesteps | Time allocation, step allocation |
| Local search operator | Search procedure at a single timestep | Per-step search, local optimizer |
| NFE | Number of Function Evaluations | Compute budget, iterations |
| Verifier | Function that scores sample quality | Reward, scorer, critic |
| Offline profiling | Pre-computed timestep sensitivity analysis | Pre-training, calibration |
| Online control | Instance-specific adaptive budget adjustment | Runtime adaptation |

---

## Key Concepts

### Two-Level Framework
- **Local level**: How to search at a given timestep (L_t operator)
- **Global level**: Where to allocate compute across timesteps (G scheduler)
- **Key insight**: These are separable concerns

### Timestep Sensitivity
- Different timesteps have different sensitivity to noise perturbations
- This varies by model (SD: early steps important; EDM: middle steps important)
- Uniform allocation wastes compute on low-sensitivity steps

### Offline-to-Online Scheduling
- **Offline**: Profile timestep importance on small sample set
- **Online**: Adapt based on gain + variance signals per instance
- **Budget enforcement**: Exact NFE budget guaranteed

---

## Positioning Statements

### What This Paper IS About
- Global scheduling of inference-time search in diffusion models
- Step-aware compute allocation under fixed NFE budget
- Combining offline profiling with online adaptation

### What This Paper is NOT About
- New local search operators (we use existing ones)
- Training-time improvements
- New sampler architectures
- Guidance scaling or classifier-free guidance

---

## Contribution Claims

1. **Framework**: First to explicitly formalize global scheduling for diffusion noise search
2. **Algorithm**: Novel offline-to-online scheduling with windowed early stopping
3. **Empirical**: Consistent 20-50% NFE savings across SD and EDM

---

## Target Audience

- Primary: ML researchers working on diffusion models
- Secondary: Practitioners deploying diffusion models with compute constraints
- Tertiary: Researchers studying test-time scaling in generative models
