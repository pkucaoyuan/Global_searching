# Figures & Tables Registry: Global Scheduling of Noise Trajectory Search

**Last Updated**: 2026-03-08
**Total Figures**: 0
**Total Tables**: 6

---

## Table Registry

| ID | Label | Section | Caption | Data Source |
|----|-------|---------|---------|-------------|
| 1 | `tab:sd_budget` | 4.2 | SD results across different NFE budgets | Experiments |
| 2 | `tab:edm_budget` | 4.3 | EDM results across different NFE budgets | Experiments |
| 3 | `tab:sd_400_ablation` | 4.4 | SD results under fixed NFE=400 (ablation) | Experiments |
| 4 | `tab:larger_prompt_exp` | 4.5 | Larger prompt set evaluation (20×10) | Experiments |
| 5 | `tab:local_search_combo` | 4.6 | Global scheduling with different local operators | Experiments |
| 6 | `tab:flow_results` | 4.7 | Flow-based model results | **TODO: Experiments needed** |

---

## Table Details

### Table 1: SD Results (tab:sd_budget)
- **Columns**: NFE, Naive, Online (Ours)
- **Rows**: Brightness (400/500/800), Compressibility (400/500/800)
- **Key finding**: 20-50% NFE savings at matched quality

### Table 2: EDM Results (tab:edm_budget)
- **Columns**: NFE, Naive, Online (Ours)
- **Rows**: Brightness (144/180/288), Compressibility (144/180/288)
- **Key finding**: 37.5-50% NFE savings at matched quality

### Table 3: Ablation (tab:sd_400_ablation)
- **Columns**: Metric, Naive, Online (Ours), Offline Only
- **Rows**: Brightness, Compressibility
- **Key finding**: Online control adds +0.009 brightness over offline-only

### Table 4: Larger Prompts (tab:larger_prompt_exp)
- **Columns**: Method, Brightness, Compressibility
- **Rows**: Baseline, Ours
- **Key finding**: Consistent improvement across 20 prompts × 10 repeats

### Table 5: Local Operators (tab:local_search_combo)
- **Columns**: Metric, Naive (Zero-order/Random), Ours (Zero-order/Random)
- **Rows**: Brightness (B), Compressibility (C)
- **Key finding**: Scheduler improves both zero-order and random operators

### Table 6: Flow-based Model (tab:flow_results) [TODO]
- **Columns**: NFE, Naive, Online (Ours)
- **Rows**: Brightness, Compressibility
- **Key finding**: Validates applicability to deterministic samplers with injected noise
- **Status**: Placeholder - awaiting experimental results

---

## Figure Registry

| ID | Label | Section | Caption | Generation Script |
|----|-------|---------|---------|-------------------|
| - | - | - | (No figures currently) | - |

---

## Planned Figures (TODO)

| ID | Purpose | Section | Status |
|----|---------|---------|--------|
| Fig 1 | Conceptual diagram of two-level framework | 3.1 | TODO |
| Fig 2 | Offline profiling results (SD vs EDM sensitivity) | 3.2 | TODO |
| Fig 3 | NFE vs Quality scaling curves | 4.2-4.3 | TODO |

---

## Generation Standards

### Matplotlib rcParams
```python
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.5),  # Single column
    'figure.dpi': 300,
})
```

### Color Palette
- Ours: #2E86AB (blue)
- Baseline: #A23B72 (magenta)
- Offline Only: #F18F01 (orange)

---

## Cross-Reference Check

| Table/Figure | Referenced In | Matches Caption |
|--------------|---------------|-----------------|
| Table 1 | Sec 4.2 | ✅ |
| Table 2 | Sec 4.3 | ✅ |
| Table 3 | Sec 4.4 | ✅ |
| Table 4 | Sec 4.5 | ✅ |
| Table 5 | Sec 4.6 | ✅ |
