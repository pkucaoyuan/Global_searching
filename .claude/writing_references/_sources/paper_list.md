# Target Papers for Writing Reference Extraction

Papers with excellent academic writing in OR/OM/Bandits/ML.

## Extraction Summary

| Category | Entries | Files |
|----------|---------|-------|
| **Phrases** | 14 | 2 |
| **Sentences** | 287 | 29 |
| **Paragraphs** | 40 | 5 |
| **Total** | **341** | **36** |

## Priority 1: Operations Research / Operations Management

| Paper | Authors | Source | Status |
|-------|---------|--------|--------|
| Dynamic Pricing with Demand Learning | Keskin & Zeevi, Javanmard | MS/arXiv | ✅ Done |
| Logarithmic Regret for Online Control | Agarwal et al. | NeurIPS 2019 | ✅ Done |
| Online Learning for Online Control | Cohen et al. | NeurIPS 2018 | ✅ Done |
| Bandits with Knapsacks | Badanidiyuru et al. | JACM 2018 | ✅ Done |
| Optimal Dynamic Pricing of Inventories | Gallego & Van Ryzin | MS 1994/97 | ✅ Done |
| Robust Convex Optimization | Ben-Tal & Nemirovski | MOR 1998 | ✅ Done |
| Online Stochastic Matching: Beating 1-1/e | Feldman et al. | FOCS 2009 | ⏳ |
| Revenue Management with Choice Model | Talluri & van Ryzin | MS 2004 | ✅ Done |

## Priority 2: Queueing / Service Operations

| Paper | Authors | Source | Status |
|-------|---------|--------|--------|
| Scheduling Policies and Workload in a Dynamic Network | Harrison | QS 1988 | ⏳ |
| Heavy-Traffic Analysis of a System with Parallel Servers | Harrison & Reiman | JAS 1981 | ⏳ |
| Stein's Method for Steady-State Diffusion Approximations | Braverman et al. | AoP 2017 | ⏳ |

## Priority 3: BAI/Bandits

| Paper | Authors | Source | Status |
|-------|---------|--------|--------|
| On the Complexity of Best-Arm Identification | Kaufmann, Cappé, Garivier | JMLR 2016 | ✅ Done |
| Time-uniform confidence sequences | Howard et al. | AoS 2021 | ✅ Done |
| Optimal Best Arm Identification with Fixed Confidence | Garivier, Kaufmann | ALT 2016 | ✅ Done |

## Priority 4: ML Theory / Econometrics

| Paper | Authors | Source | Status |
|-------|---------|--------|--------|
| A Modern Introduction to Online Learning | Orabona | arXiv 2019 | ✅ Done |
| Estimation and Inference of Heterogeneous Treatment Effects | Wager & Athey | JASA 2018 | ✅ Done |
| Bandit Algorithms (Book) | Lattimore, Szepesvári | banditalgs.com | ⏳ |

## Priority 5: Large Language Models

| Paper | Authors | Source | Status |
|-------|---------|--------|--------|
| Attention Is All You Need | Vaswani et al. | NeurIPS 2017 | ✅ Done |
| BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al. | NAACL 2019 | ✅ Done |
| Language Models are Few-Shot Learners (GPT-3) | Brown et al. | NeurIPS 2020 | ✅ Done |
| Training language models to follow instructions (InstructGPT) | Ouyang et al. | NeurIPS 2022 | ✅ Done |
| Chain-of-Thought Prompting Elicits Reasoning | Wei et al. | NeurIPS 2022 | ✅ Done |

## Extraction Progress

### Completed Papers (31)
1. ✅ **Kaufmann et al. (JMLR 2016)** - BAI complexity → `problem_setup.md`, `related_work.md`
2. ✅ **Howard et al. (AoS 2021)** - Confidence sequences → `motivation.md`, `citation_patterns.md`, `properties_list.md`
3. ✅ **Badanidiyuru et al. (JACM 2018)** - Bandits with Knapsacks → `or_applications.md`, `main_results.md`
4. ✅ **Orabona (arXiv 2019)** - Online learning tutorial → `tutorial_style.md`
5. ✅ **Wager & Athey (JASA 2018)** - Causal forests → `causal_inference.md`
6. ✅ **Agarwal et al. (NeurIPS 2019)** - Online control → `control_theory.md`
7. ✅ **Garivier & Kaufmann (ALT 2016)** - Optimal BAI → `algorithm_optimality.md`, `proof_structure.md`
8. ✅ **Cohen et al. (NeurIPS 2018)** - Online LQ control → `online_learning.md`
9. ✅ **Keskin & Zeevi, Javanmard et al.** - Dynamic pricing → `dynamic_pricing.md`
10. ✅ **Gallego & Van Ryzin (MS 1994/97)** - Revenue management → `revenue_management.md`
11. ✅ **Ben-Tal & Nemirovski (MOR 1998)** - Robust optimization → `robust_optimization.md`
12. ✅ **Karp, Vazirani, Vazirani (STOC 1990)** - Online matching → `online_matching.md`
13. ✅ **Halfin & Whitt (OR 1981)** - Heavy traffic queueing → `queueing_theory.md`
14. ✅ **Arrow et al. / Newsvendor literature** - Inventory management → `inventory_management.md`
15. ✅ **Birge & Louveaux (Springer)** - Stochastic programming → `stochastic_programming.md`
16. ✅ **Prophet/Secretary literature** - Optimal stopping → `optimal_stopping.md`
17. ✅ **Approximation algorithms literature** - PTAS/FPTAS → `approximation_algorithms.md`
18. ✅ **Myerson (1981)** - Mechanism design → `mechanism_design.md`
19. ✅ **Puterman (1994)** - MDP → `markov_decision_process.md`
20. ✅ **Ford & Fulkerson (1956)** - Network flow → `network_optimization.md`
21. ✅ **Nash (1950)** - Game theory → `game_theory.md`
22. ✅ **Lee et al. (MS 1997)** - Supply chain → `supply_chain.md`
23. ✅ **Vapnik / PAC literature** - Learning theory → `learning_theory.md`
24. ✅ **Generic patterns** → `introduction.md`, `contribution.md`, `transitions.md`, `hedging.md`
25. ✅ **Vaswani et al. (NeurIPS 2017)** - Transformer → `llm_papers.md`
26. ✅ **Devlin et al. (NAACL 2019)** - BERT → `llm_papers.md`
27. ✅ **Brown et al. (NeurIPS 2020)** - GPT-3 → `llm_papers.md`
28. ✅ **Ouyang et al. (NeurIPS 2022)** - InstructGPT/RLHF → `llm_papers.md`
29. ✅ **Wei et al. (NeurIPS 2022)** - Chain-of-Thought → `llm_papers.md`
30. ✅ **Bertsimas & Sim (OR 2004)** - Price of Robustness → `price_of_robustness.md`
31. ✅ **Talluri & van Ryzin (MS 2004)** - Choice Models → `choice_models.md`

### Pending Papers
- [ ] Feldman et al. (FOCS 2009) - Online matching
- [ ] Harrison papers - Queueing theory
- [ ] Dean et al. - Sample complexity of LQR
- [ ] Lattimore & Szepesvári - Bandit Algorithms Book

## Downloaded Sources

```
_sources/
├── kaufmann2016/      # BAI complexity
├── howard2021/        # Confidence sequences
├── bwk2013/           # Bandits with Knapsacks
├── orabona2019/       # Online learning
├── garivier2016/      # Optimal BAI
├── agarwal2019/       # Online control
└── cohen2018/         # Online LQ control
```

## Notes

- Focus on Introduction and Contribution sections first
- Extract transition phrases, hedging, and sentence structures
- Note the source for each extracted reference
- Papers without arXiv LaTeX sources are harder to process (PDFs only)
