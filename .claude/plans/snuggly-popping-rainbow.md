# Plan: Draft Numerical Experiments Section for NeurIPS Paper

## Configuration
- **Language**: English (NeurIPS standard)
- **Figure Format**: PDF (vector, LaTeX-ready)
- **Scope**: Full (all 5 subsections)

## Objective
Draft the complete numerical experiments section based on evaluation results from `outputs/results.db`, including:
- Multiple focused sections (by metrics, experiments, ablations)
- Clear experimental settings
- Publication-quality figures with CMU Serif font

---

## Section Structure

### 5.1 Experimental Setup
- Training configuration (Qwen3-8B, LoRA, hardware)
- Benchmark description (OR-Debug-Bench Expert, 200 problems)
- Evaluation metrics definitions (RR@k, DA, OP, Avg Steps)

### 5.2 Main Results: Model Comparison (7 LLMs)
- Table 1: Production LLM comparison on OR-Debug-Bench
- Models: o1, o4-mini, gpt-5-mini, Kimi-K2-Thinking, Llama-3.3-70B, etc.
- **Figure 1**: RR@k curves for all models (line chart)

### 5.3 Ablation Study: RL Training Methods (Exp1-4)
- Table 2: Method ablation (SFT, DAPO, Curriculum, PRM)
- Key findings: Curriculum+DAPO best RR, PRM improves DA
- **Figure 2**: Training efficiency vs performance (dual-axis)
- **Figure 3**: DA vs RR@5 trade-off (scatter plot)

### 5.4 RAG Enhancement Results
- Table 3: RAG strategies comparison (quick_fix, reasoning, by_type)
- Table 4: K-value ablation (k=1,3,5,7)
- **Figure 4**: RR improvement with RAG (bar chart)
- **Figure 5**: DA metric variants comparison

### 5.5 Negative Result: Ceiling Effect (Exp5)
- Table 5: SFT vs DAPO on harder benchmark
- Discussion of when RL fails to improve over SFT

### 5.6 Case Studies
- 2-3 qualitative examples showing model behavior

---

## Figure Generation Plan

### Figure Configuration (CMU Serif)
```python
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})
```

### Figures to Generate

| Figure | Type | Data Source | Key Design |
|--------|------|-------------|------------|
| Fig1 | Line chart | model_summaries | RR@k (k=1,5,10,15,20) for 7 models |
| Fig2 | Dual-axis bar | Exp1-4 results | Training time vs ΔRR@5 |
| Fig3 | Scatter plot | Exp1-4 results | DA (y) vs RR@5 (x), Pareto frontier |
| Fig4 | Grouped bar | RAG results | Baseline vs RAG RR@5 |
| Fig5 | Grouped bar | RAG DA metrics | 6 DA variants comparison |

---

## Data Sources

### SQLite Database (`outputs/results.db`)
- `model_summaries`: 7 models with all metrics
- `evaluation_results`: 1,600 per-problem records

### Documentation Files
- `docs/progress/EXP1-4_PAPER_RESULTS.md`: Exp1-4 numeric results
- `docs/progress/EXP1-5_SUMMARY.md`: Full experiment landscape
- `docs/progress/PAPER_RESULTS_STRUCTURE.md`: Figure templates

### RAG Results
- K-value ablation: k=1 (86.5%), k=3 (94.5%), k=5 (96.5%), k=7 (97.0%)
- DA metrics: IIS_Recall, Any_Key_Fix, Success_And_Key

---

## Implementation Steps

### Step 1: Create Visualization Script
**File**: `scripts/visualization/plot_paper_figures.py`
- Configure CMU Serif font
- Query database for metrics
- Generate all 5 figures
- Save to `outputs/paper_figures/`

### Step 2: Draft Section 5.1 (Setup)
**File**: `docs/paper/numerical_experiments.md`
- Hardware, training config table
- Benchmark statistics
- Metric definitions

### Step 3: Draft Section 5.2 (Model Comparison)
- Extract data from `model_summaries`
- Create Table 1 with 7 models × 6 metrics
- Generate Figure 1

### Step 4: Draft Section 5.3 (Ablation)
- Use Exp1-4 results from docs
- Create Table 2 with method comparison
- Generate Figures 2 & 3

### Step 5: Draft Section 5.4 (RAG)
- Use RAG experiment results
- Create Tables 3 & 4
- Generate Figures 4 & 5

### Step 6: Draft Section 5.5 (Negative Result)
- Document Exp5 ceiling effect
- Create Table 5

### Step 7: Draft Section 5.6 (Case Studies)
- Select 2-3 representative examples from database

---

## Output Files

```
outputs/paper_figures/
├── fig1_rr_at_k_models.pdf          # RR@k curves for 7 models
├── fig2_training_efficiency.pdf      # Training time vs performance
├── fig3_da_rr_tradeoff.pdf          # DA-RR Pareto frontier
├── fig4_rag_improvement.pdf          # RAG vs baseline comparison
└── fig5_da_metrics_comparison.pdf    # DA metric variants

docs/paper/
└── numerical_experiments.md          # Full draft in English markdown

scripts/visualization/
└── plot_paper_figures.py             # Figure generation script
```

## Bash Commands for Verification

```bash
# Generate all figures
python scripts/visualization/plot_paper_figures.py

# Check figure files
ls -lh outputs/paper_figures/*.pdf

# Verify CMU Serif font (requires fc-list)
fc-list | grep -i "CMU Serif"
```

---

## Verification

1. **Figure Quality**: Check 300 DPI, CMU Serif font rendering
2. **Data Accuracy**: Cross-check numbers with database queries
3. **Legend Placement**: Ensure no overlaps (use `bbox_to_anchor` if needed)
4. **Color Accessibility**: Use colorblind-friendly palette
5. **English Text**: All labels, titles, legends in English

---

## Critical Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `scripts/visualization/plot_paper_figures.py` | Create | Generate all figures |
| `docs/paper/numerical_experiments.md` | Create | Draft paper section |
| `outputs/paper_figures/` | Create dir | Store figures |

---

## Key Numeric Results Summary

### Model Comparison (Table 1)
| Model | RR@5 | RR | DA | Avg Steps | OP |
|-------|------|-----|-----|-----------|-----|
| o1 | 76.0% | 94.0% | 41.67% | 3.86 | 90.59% |
| o4-mini | 77.5% | 97.5% | 36.01% | 4.49 | 67.02% |
| Kimi-K2-Thinking | 61.5% | 99.5% | 14.97% | 5.45 | 58.57% |
| gpt-5-mini | 57.5% | 99.0% | 19.31% | 5.58 | 58.66% |
| Llama-3.3-70B | 45.5% | 95.5% | 37.74% | 6.51 | 81.83% |
| gpt-4.1-mini | 19.5% | 55.5% | 6.94% | 10.30 | 56.18% |
| gpt-5-nano | 10.5% | 28.5% | 6.94% | 12.86 | 76.90% |

### RL Ablation (Table 2)
| Method | RR@5 | DA | Avg Steps | Training |
|--------|------|-----|-----------|----------|
| SFT Baseline | 91.5% | 68.0% | 2.49 | - |
| SFT+DAPO (Exp1) | 92.0% | 66.0% | 2.33 | ~2h |
| Curriculum+DAPO (Exp2) | **95.0%** | 68.0% | **2.25** | ~1.5h |
| Curriculum+DAPO+PRM (Exp4) | 92.0% | **72.7%** | 2.42 | ~2h |

### RAG k-Ablation (Table 4)
| k | RR | Avg Steps | DA_any_key |
|---|-----|-----------|------------|
| 1 | 86.5% | 2.19 | 80.0% |
| 3 | 94.5% | 1.51 | 82.0% |
| **5** | **96.5%** | 1.62 | **85.0%** |
| 7 | 97.0% | 1.53 | 85.0% |
