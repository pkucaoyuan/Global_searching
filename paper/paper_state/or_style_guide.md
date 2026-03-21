# OR Style Guide — Extracted from OPRE-2024-11-1450.R1

**Source**: "Learning to Simulate from Heavy-tailed Distribution via Diffusion Model" (Operations Research, accepted)
**Purpose**: Guide conversion of GAINS paper from ML conference style to OR journal style

---

## 1. Overall Structure Differences (ML vs OR)

| Aspect | ML Conference (Current) | OR Journal (Target) |
|--------|------------------------|---------------------|
| **Length** | 8-10 pages + appendix | 25-40 pages (main body), no strict limit |
| **Abstract** | ~150 words, 1 paragraph | ~150 words, single paragraph, keywords below |
| **Introduction** | Brief, 1-2 pages | 2-4 pages, detailed motivation + full contribution list |
| **Related Work** | Separate section after intro | 1.1 subsection within Introduction |
| **Theory** | Supporting role, props/lemmas | Central contribution, full theorems with proof sketches |
| **Proofs** | Appendix only | Important proofs inline or immediately after theorems; lengthy proofs in appendix |
| **Experiments** | Central, many tables | Supporting theory; fewer but deeper experiments |
| **Conclusion** | 2-3 lines | Full section with summary, limitations, managerial implications, future work |
| **Appendix** | Supplementary | Integral part of paper (proofs, additional examples, experiment details) |

---

## 2. Language & Register

### 2.1 Formality Level
OR journals use **formal academic English** — more formal than ML conferences:

| ML Style (Current) | OR Style (Target) |
|--------------------|-------------------|
| "shows that X is better" | "demonstrates that X yields improvement" |
| "we find that" | "we observe that" or "our analysis reveals that" |
| "X works well" | "X performs effectively" or "X achieves favorable performance" |
| "gets 20% fewer NFEs" | "achieves a 20% reduction in function evaluations" |
| "key limitation" | "a fundamental limitation" |
| "This overlooks a fundamental empirical fact" | "This approach neglects a structural property" |

### 2.2 Voice & Person
- **First person plural** ("we") is standard and natural in OR
- **Active voice** preferred but passive acceptable for variety
- Reference example: "We provide the specific formulation..." / "In this work, we theoretically show that..."

### 2.3 Sentence Structure
OR papers favor:
- **Longer, flowing sentences** with subordinate clauses (vs ML's short punchy sentences)
- **Explicit logical connectors**: "Consequently," "Moreover," "To this end," "In particular,"
- **Formal transition**: "We remark that..." "It is worth noting that..."
- **Numbered contributions** with full sentences, not fragments

### 2.4 Word Choice Patterns from OPRE Reference

**Mathematical framing**:
- "We deliver the following analysis and results in this work."
- "One premise in our considered situations is that..."
- "The task considered in our work is to..."
- "We provide a set of theoretical results to analyze..."

**Connecting theory to practice**:
- "To make this concept tangible, we have restructured..."
- "This theoretical guideline provides a principled range for..."
- "The key insight is that..."
- "In practice, we find that this theoretical starting point..."

**Discussing limitations of existing work**:
- "existing diffusion models encounter challenges in addressing..."
- "the formulation exclusively relies on specific properties of Gaussian"
- "Gaussian perturbation does not create enough connection between data at the tail and the center"

---

## 3. Section-by-Section OR Conventions

### 3.1 Introduction Pattern
The OPRE reference intro follows this structure:
1. **Application context** (OR/OM): "Modeling and simulating from a multi-dimensional distribution have witnessed a range of applications in several domains of operations research and management, including service systems, financial platforms, logistics and supply chain systems"
2. **Problem statement**: Clear, formal description of the task
3. **Why it matters for OR/OM**: Connect to real operational decisions
4. **Gap in existing approaches**: What current methods miss
5. **Numbered contributions**: 4 contributions, each a full paragraph

### 3.2 Background / Problem Setting
- Provide **complete formal definitions** (Definition 1, 2, 3...)
- Include **examples** (Example 1: Location-scale t distribution)
- Use **subsection numbering** (2.1, 2.1.1, 2.2)
- Define ALL notation before use

### 3.3 Theory Sections
- **Theorem/Proposition/Lemma hierarchy** with clear numbering
- **Remarks** after major results to explain intuition
- **Proof sketches** inline for key results; full proofs in appendix
- **Numerical examples** to illustrate theoretical findings (e.g., Figure 1 in reference)

### 3.4 Experiments
- **Detailed experimental setup** with explicit parameter values
- **Multiple datasets** spanning synthetic and real-world
- **Evaluation metrics** formally defined with equations
- **Ablation studies** clearly motivated
- **Connect back to theory**: "The results are consistent with the theoretical result in Proposition 1"

### 3.5 Conclusion
Must include:
- Summary of contributions
- Managerial/practical implications
- Limitations
- Future directions (specific, not vague)

---

## 4. Formatting Conventions

### 4.1 Mathematics
- **All equations numbered** (not just referenced ones)
- **Definitions, Theorems, Propositions** use `\newtheorem` with section numbering
- **Remarks** after theorems for interpretation
- **Notation table** or notation section early in paper
- Display math for all important equations

### 4.2 Citations
- **Parenthetical**: `(Author et al. YEAR)` for general references
- **Textual**: `Author et al. (YEAR)` when author is subject of sentence
- **"see"** prefix: "see Glynn and Whitt (1994)" for supporting references
- OR papers cite heavily from OR/OM literature

### 4.3 Tables & Figures
- Tables use `\toprule`, `\midrule`, `\bottomrule` (booktabs)
- Caption is **descriptive** (full sentence explaining what table shows)
- Bold for best results
- Method names in SMALL CAPS

### 4.4 Footnotes
- Used more freely than ML papers for technical clarifications
- Example: "Our notations follow from Ho et al. (2020)..."

---

## 5. Key Vocabulary Mapping (ML → OR)

| ML Term | OR Equivalent |
|---------|---------------|
| inference-time scaling | test-time computation allocation |
| sample quality | simulation fidelity / output quality |
| function evaluations (NFE) | computational budget / oracle queries |
| verifier | quality metric / reward function / performance measure |
| prompt | input condition / problem instance |
| noise search | perturbation optimization |
| diffusion model | score-based generative model / simulation framework |
| DDPM/SDE | stochastic process / forward-backward Markov chain |
| latent trajectory | state trajectory / process evolution |
| score function | (Stein) score function (keep, but define explicitly) |

---

## 6. OR-Specific Content to Add

### 6.1 Motivation Anchors
- Simulation optimization in supply chains
- Monte Carlo simulation for risk assessment
- Stochastic process generation for queueing/inventory
- Digital twin simulation

### 6.2 Literature to Cite
- OR simulation: Law (2015), Glasserman (2003), Nelson (2013)
- Stochastic optimization: Shapiro et al. (2021)
- Simulation metamodeling: Barton and Meckesheimer (2006)
- Score-based simulation: Cao et al. (2024) — already cited in reference paper

### 6.3 Managerial Insights
- Budget allocation principle: non-uniform compute allocation outperforms uniform
- Diminishing returns characterization: when to stop investing compute
- Practical guideline: offline profiling cost is amortized across instances

---

*Extracted from OPRE-2024-11-1450.R1 on 2026-03-16*
