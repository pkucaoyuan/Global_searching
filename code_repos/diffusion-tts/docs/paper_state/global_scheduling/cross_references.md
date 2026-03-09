# Cross-References: Global Scheduling of Noise Trajectory Search

**Last Updated**: 2026-03-08

---

## Section References

| Label | Type | Section | Content |
|-------|------|---------|---------|
| sec:intro | Section | 1 | Introduction |
| sec:related | Section | 2 | Related Work |
| sec:method | Section | 3 | Methodology |
| sec:method-global | Subsection | 3.1 | Global Modeling of Noise Trajectory Search |
| sec:method-ours | Subsection | 3.2 | Proposed Global Scheduling Algorithm |
| sec:experiments | Section | 4 | Experiments |
| sec:exp-setting | Subsection | 4.1 | Experimental Setting |
| sec:exp-sd-scaling | Subsection | 4.2 | Stable Diffusion: Scaling with NFE |
| sec:exp-edm-scaling | Subsection | 4.3 | EDM: Scaling with NFE |
| sec:exp-ablation-offline-online | Subsection | 4.4 | Ablation: Offline vs Offline+Online |
| sec:exp-larger-prompts | Subsection | 4.5 | Larger Prompt Set Evaluation |
| sec:exp-local-operator | Subsection | 4.6 | Compatibility with Different Local Operators |
| sec:conclusion | Section | 5 | Conclusion |

---

## Table References

| Label | Type | Section | Caption |
|-------|------|---------|---------|
| tab:sd_budget | Table | 4.2 | SD results across different NFE budgets |
| tab:edm_budget | Table | 4.3 | EDM results across different NFE budgets |
| tab:sd_400_ablation | Table | 4.4 | SD results under fixed NFE=400 |
| tab:larger_prompt_exp | Table | 4.5 | Larger prompt set evaluation |
| tab:local_search_combo | Table | 4.6 | Global scheduling with different local operators |

---

## Algorithm References

| Label | Type | Section | Caption |
|-------|------|---------|---------|
| alg:offline_online_budget | Algorithm | 3.2 | Offline-to-Online Budget Scheduling (Windowed Early Stop) |

---

## Internal Cross-References (in text)

| From Section | Reference | To Section | Status |
|--------------|-----------|------------|--------|
| 3.2 | \cref{sec:experiments} | 4 | ✅ |
| 4.2 | Table~\ref{tab:sd_budget} | 4.2 | ✅ |
| 4.3 | Table~\ref{tab:edm_budget} | 4.3 | ✅ |
| 4.4 | Table~\ref{tab:sd_400_ablation} | 4.4 | ✅ |
| 4.5 | Table~\ref{tab:larger_prompt_exp} | 4.5 | ✅ |
| 4.6 | Table~\ref{tab:local_search_combo} | 4.6 | ✅ |

---

## Citation References (TODO)

| Citation Key | First Used | Sections Used | Status |
|--------------|------------|---------------|--------|
| (TBD) | - | - | Need to add bibliography |

---

## Verification Checklist

- [x] All \ref{} targets exist
- [x] All \cref{} targets exist
- [x] All table labels match captions
- [ ] All citations exist in .bib file (TODO)
