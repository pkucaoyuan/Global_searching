# Theory References: Scheduling Theory - Deep Foundations

You are a scheduling theory specialist providing **deep theoretical foundations** for LLM resource allocation. Your role is to connect classical scheduling theory to modern LLM serving challenges.

## Arguments

`$ARGUMENTS` — Optional topic focus. Examples:
- (empty) — Comprehensive scheduling theory overview
- `parallel` — Parallel machine scheduling (P||, Q||, R||)
- `online` — Online scheduling and competitive analysis
- `approximation` — Approximation algorithms and LP relaxation
- `flow-time` — Flow time minimization (SRPT, preemption)
- `batch` — Batch scheduling with setup costs
- `learning-augmented` — Scheduling with ML predictions

## Protocol

### Phase 1: Load Foundation References

Read the comprehensive reference document:
```
Read paper/energy_optimal_af/theory_references.md (Section 14.2)
```

### Phase 2: Execute Deep Analysis

Launch a Task agent using the scheduling theory agent configuration:

```
Task("Scheduling theory deep analysis: [TOPIC]", "general-purpose",
  "You are a scheduling theory specialist.

   **Read your agent configuration first:**
   Read .claude/agents/theory-scheduling-agent.md

   **Research Topic:** [TOPIC]

   **Execution:**
   1. Search for classical papers using the search strategy in your agent config
      - Graham (1966, 1969), LST (1990), Kalyanasundaram-Pruhs (2000)
   2. For each major result:
      - State theorem with precise competitive/approximation ratio
      - Identify proof techniques (LP, primal-dual, potential function)
      - Assess applicability to LLM serving
   3. Build connection map:
      - Classical result → LLM application
      - Required adaptations
      - Open theoretical gaps (especially 2 vs 3/2 for R||C_max)

   **Return:** Structured analysis following your agent's output format.")
```

## Core Reference Library

### 1. Graham's Scheduling Foundations

| Result | Statement | Bound | LLM Connection |
|--------|-----------|-------|----------------|
| **List Scheduling** | Assign arriving job to least loaded machine | $2 - 1/m$ competitive | Online batch dispatch |
| **LPT** | Sort by decreasing size, then list | $4/3 - 1/(3m)$ approx | Offline optimization |
| **Anomalies** | Adding resources can hurt | Worst case analysis | Capacity planning pitfalls |

### 2. Parallel Machine Hierarchy

| Problem | Notation | Best Known | Lower Bound | Status |
|---------|----------|------------|-------------|--------|
| Identical | $P \| C_{\max}$ | PTAS | - | Closed |
| Uniform | $Q \| C_{\max}$ | PTAS | - | Closed |
| **Unrelated** | $R \| C_{\max}$ | **2-approx** | **3/2** | **35-year gap!** |

**LST 1990 Key Result**:
For unrelated parallel machines:
- LP relaxation gives lower bound
- Rounding achieves makespan ≤ T* + p_max ≤ 2·OPT
- No algorithm can achieve < 3/2 unless P=NP

**LLM Application**: A and F are "unrelated machines":
- Processing time depends on machine type AND job characteristics
- A: memory-bound, grows with KV cache
- F: compute-bound, stable per-token cost

### 3. Online Scheduling

| Algorithm | Setting | Competitive Ratio | Technique |
|-----------|---------|-------------------|-----------|
| List Scheduling | Makespan | $2 - 1/m$ | Worst-case |
| Randomized | Makespan | 1.916 | Balls-into-bins |
| SRPT | Flow time | **Optimal** (preemptive) | Greedy |
| Round Robin | Fairness | O(1) stretch | Time-slicing |

**Competitive Analysis Framework** (Sleator-Tarjan 1985):
$$\text{ALG}(\sigma) \leq c \cdot \text{OPT}(\sigma) + b$$
Algorithm is $c$-competitive if this holds for all inputs $\sigma$.

### 4. Resource Augmentation

**Kalyanasundaram-Pruhs Theorem (2000)**:

> With $(1+\epsilon)$-speed augmentation, non-clairvoyant algorithms can achieve O(1)-competitive ratio for flow time.

**Formal Statement**:
If ALG runs on machines with speed $1+\epsilon$ while OPT runs on speed-1 machines:
$$\text{ALG}_{1+\epsilon}(\sigma) \leq O(1) \cdot \text{OPT}_1(\sigma)$$

**LLM Application**: Over-provisioning by 10-20% yields near-optimal performance guarantee. This justifies practical capacity planning with slack.

### 5. Flow Time Minimization

| Result | Author(s) | Setting | Guarantee |
|--------|-----------|---------|-----------|
| **SRPT optimal** | Schrage (1968) | Single machine, preemptive | Minimizes $\sum C_j$ |
| **Weighted flow** | Chekuri et al. (2001) | Online, weighted | $O(\log^2 P)$ competitive |
| **Heavy-tailed** | Nair et al. | Power-law sizes | SRPT still good |

**SRPT for LLM**: Process shortest remaining request first.
- Challenge: Don't know output length in advance
- Solution: Learning-augmented prediction

### 6. Learning-Augmented Scheduling

**Framework** (Purohit-Svitkina-Kumar 2018):

| Property | Definition |
|----------|------------|
| **Consistency** | Competitive ratio when predictions are correct |
| **Robustness** | Competitive ratio when predictions are adversarial |
| **Smoothness** | Performance degrades gracefully with prediction error |

**Key Trade-off Theorem** (Wei-Zhang 2020):
For any $(1+\lambda)$-consistent deterministic algorithm:
$$\text{Robustness} \geq \frac{(1+\lambda)^2}{2\lambda}$$

This is **tight** — there exist algorithms achieving this frontier.

**LLM Application**: Predict output length $\hat{o}_i$
- If accurate: near-optimal admission/scheduling
- If wrong: graceful degradation to competitive baseline
- Define error: $\eta = \sum_i |\hat{o}_i - o_i| / \sum_i o_i$

### 7. Batch Scheduling

| Result | Author(s) | Key Finding |
|--------|-----------|-------------|
| **Optimal batch policies** | Potts-Kovalyov (2000) | Threshold-based |
| **Setup time batching** | Allahverdi et al. (2008) | Sequence-dependent setup |
| **Online batching** | Various | Competitive bounds |

**Threshold Policies**: Serve when queue reaches threshold $M$
- Optimal for many objectives
- WAIT algorithm is threshold-based

## Proof Technique Toolkit

### LP Relaxation & Rounding

**Configuration LP** (LST 1990):
```
min T
s.t. Σ_i x_{ij} = 1                    ∀j  (job assignment)
     Σ_j p_{ij} · x_{ij} ≤ T           ∀i  (machine load)
     x_{ij} ∈ [0,1]
```

**Rounding Guarantee**: Integral solution has makespan ≤ LP* + max processing time ≤ 2·OPT.

### Potential Function Method

For online competitive analysis:
1. Define potential Φ(state)
2. Show: Cost_ALG + ΔΦ ≤ c · Cost_OPT for each step
3. Telescope: Total_ALG ≤ c · Total_OPT + Φ(initial) - Φ(final)

### Primal-Dual Schema

1. Write LP and dual
2. Build primal solution incrementally
3. Update dual variables to maintain feasibility
4. Analyze complementary slackness

## LLM-Specific Gaps Analysis

| Classical Assumption | LLM Reality | Required Extension |
|---------------------|-------------|-------------------|
| Known processing times | Unknown output length | Learning-augmented |
| Independent jobs | KV cache shared/conflicting | Memory constraints |
| Single objective | Throughput + latency + energy | Multi-objective |
| Homogeneous machines | A ≠ F characteristics | Unrelated + coupling |
| Immediate availability | GPU warmup, model load | Setup time |

## Topic-Specific Deep Dives

### Approximation Algorithms: The 2 vs 3/2 Gap

**Why it matters**: LST 1990 is 35 years old. The 2-approximation vs 3/2 lower bound gap is one of the most famous open problems.

**Potential approaches for LLM**:
1. Exploit structure: A/F jobs have special structure
2. Resource augmentation: Allow slight over-provisioning
3. Stochastic input: Random arrival order helps

### Online Scheduling: Semi-Online Models

**k-Lookahead** (2023):
- See next k arrivals
- 2 machines: achieves 4/3 (optimal for k≥2)
- 3 machines: achieves 16/11 ≈ 1.45

**Connection to Balance-Future**: BF uses rolling horizon optimization ≈ k-lookahead.

### Learning-Augmented: Specific Results

**Bamas et al. (NeurIPS 2023)** for energy-efficient scheduling:

| Objective | Consistency | Robustness |
|-----------|-------------|------------|
| Flow time | $(1 + (2\lambda)^{1/\alpha})^\alpha$ | O(1) |
| Weighted flow | $(1 + ((\alpha/\log\alpha)^2\lambda)^{1/\alpha})^\alpha$ | O(1) |
| Deadlines | $1 + \lambda$ | $O(4^{\alpha^2}/\lambda^{\alpha-1})$ |

## Output Format

```markdown
# 深度调度理论分析: [Topic]

## 1. 核心定理精确陈述

### 定理 [Name] ([Author Year])

**问题**: [Formal problem notation, e.g., R||C_max]

**陈述**: [Approximation ratio / competitive ratio]

**证明技术**: [LP, primal-dual, potential function]

**LLM 适用性**: ✅/⚠️/❌ + [解释]

## 2. 证明工具

| 工具 | 经典用途 | LLM 应用 |
|------|---------|---------|
| LP 松弛 | 下界 + 舍入 | A/F 分配 |
| 势函数 | 在线分析 | 竞争比证明 |

## 3. 与 LLM 的桥梁

| 经典结果 | LLM 问题 | 适配方法 |
|---------|---------|---------|
| LST 2-approx | A/F 异构 | 加约束后重新分析 |
| SRPT | 短请求优先 | 需预测长度 |

## 4. 开放研究问题

1. 能耗约束 R||C_max 的近似比?
2. 两阶段作业 (prefill→decode) 的调度?
```

## Begin

Parse `$ARGUMENTS` for topic focus, then:
1. Load relevant sections from theory_references.md
2. Launch Task agent for deep analysis
3. Generate structured output with precise bounds
4. Identify LLM-specific adaptations needed
5. Highlight famous open problems (like 2 vs 3/2)
