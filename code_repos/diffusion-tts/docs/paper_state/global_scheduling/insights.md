# Insights Registry: Global Scheduling of Noise Trajectory Search

**Last Updated**: 2026-03-08
**Target Venue**: ML (NeurIPS/ICML/ICLR)

---

## Main Contributions (Abstract-Level)

1. **C1**: A unified two-level view of noise trajectory search that separates local noise refinement from global timestep scheduling
2. **C2**: A novel global scheduling algorithm that integrates offline timestep profiling with online, feedback-driven control
3. **C3**: Extensive experiments showing consistent gains (20-50% NFE savings) over existing inference-time methods

---

## Technical Insights (for ML)

### Insight 1: Timestep Sensitivity Varies Dramatically
**What**: Different timesteps exhibit dramatically different sensitivities to noise perturbations
**Why non-trivial**: Prior work treats denoising trajectory as homogeneous
**Supporting evidence**: Offline profiling shows SD gains concentrate on early steps, EDM on middle steps

### Insight 2: Two-Level Decomposition is Natural
**What**: Inference-time search naturally decomposes into local (how to search) and global (where to search) decisions
**Why non-trivial**: Unifies disparate methods (Best-of-N, tree search, greedy) under common framework
**Supporting evidence**: Sec 3.1 shows existing methods as special cases

### Insight 3: Online Adaptation Adds Value Over Static Allocation
**What**: Instance-specific feedback (gain + variance) enables better budget utilization
**Why non-trivial**: Offline profiling alone captures only population-level patterns
**Supporting evidence**: Ablation Table 3 shows +0.009 brightness from online control

---

## Comparative Insights

| Our Approach | Prior Work | Improvement | Evidence |
|--------------|------------|-------------|----------|
| Step-aware scheduling | Uniform allocation | 20-50% NFE savings | Tables 1-2 |
| Offline+Online | Offline only | +0.009 brightness | Table 3 |
| Global budget reasoning | Step-agnostic heuristics | Consistent across models | SD + EDM results |

---

## Limitations & Future Work

| Limitation | Section Mentioned | Potential Extension |
|------------|-------------------|---------------------|
| Requires offline profiling per model | Implicit | Transfer learning of schedules |
| Single-image verifiers only | Experiments | Extend to perceptual/learned verifiers |
| Greedy timestep traversal | Methodology | Non-sequential scheduling |

---

## Key Takeaways for Different Audiences

### For Practitioners
1. Don't allocate search compute uniformly - profile your model first
2. Early stopping based on gain+variance signals saves compute without quality loss
3. The scheduler is modular - works with any local search operator

### For Researchers
1. Two-level framework provides common language for inference-time methods
2. Offline profiling reveals model-specific patterns (SD vs EDM differ)
3. Open problem: how to transfer schedules across models/datasets

### For Reviewers
1. First work to explicitly study global scheduling of diffusion noise search
2. Consistent 20-50% compute savings across two major diffusion architectures
3. Ablation cleanly separates offline vs online contributions
