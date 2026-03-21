# Theory References: Queueing Theory - Deep Foundations

You are a queueing theory specialist providing **deep theoretical foundations** for LLM serving system analysis. Your role is to connect classical queueing theory to modern LLM challenges.

## Arguments

`$ARGUMENTS` — Optional topic focus. Examples:
- (empty) — Comprehensive queueing theory overview
- `heavy-traffic` — Heavy traffic limits and diffusion approximations
- `processor-sharing` — PS queues and continuous batching
- `fork-join` — Synchronization and tensor parallelism
- `vacation` — Setup time and GPU sleep cycles
- `state-dependent` — State-dependent service (KV cache growth)
- `resource-pooling` — Multi-server pooling (A/F disaggregation)

## Protocol

### Phase 1: Load Foundation References

Read the comprehensive reference document:
```
Read paper/energy_optimal_af/theory_references.md (Section 14.1)
```

### Phase 2: Execute Deep Analysis

Launch a Task agent using the queueing theory agent configuration:

```
Task("Queueing theory deep analysis: [TOPIC]", "general-purpose",
  "You are a queueing theory specialist.

   **Read your agent configuration first:**
   Read .claude/agents/theory-queueing-agent.md

   **Research Topic:** [TOPIC]

   **Execution:**
   1. Search for classical papers using the search strategy in your agent config
   2. For each major result:
      - State theorem precisely with conditions
      - Identify proof techniques used
      - Assess applicability to LLM serving
      - Identify where LLM breaks assumptions
   3. Build connection map:
      - Classical result → LLM application
      - Required adaptations
      - Open theoretical gaps

   **Return:** Structured analysis following your agent's output format.")
```

### Phase 3: Generate Analysis

## Core Reference Library

### 1. Foundational Laws

| Result | Author(s) | Statement | LLM Connection |
|--------|-----------|-----------|----------------|
| **Little's Law** | Little (1961) | $L = \lambda W$ | Throughput-latency tradeoff foundation |
| **PASTA** | Wolff (1982) | Poisson arrivals see time averages | Arrival process analysis |
| **Burke's Theorem** | Burke (1956) | Output of M/M/1 is Poisson | Cascaded stage analysis |

### 2. Product Form Networks

| Result | Author(s) | Key Condition | LLM Connection |
|--------|-----------|---------------|----------------|
| **Jackson Networks** | Jackson (1957, 1963) | Exp service, routing matrix | r-A-1F as open network |
| **BCMP Theorem** | BCMP (1975) | 4 service disciplines (FCFS/LCFS-PR/PS/IS) | Multi-class requests |
| **Kelly Networks** | Kelly (1979) | General routing, symmetric queues | Batch routing policies |

**Key Insight**: Product form requires **local balance** (detailed balance). LLM's KV cache growth **breaks local balance** because service time depends on job history, not just current queue state.

### 3. Heavy Traffic Theory

| Result | Author(s) | Statement | LLM Connection |
|--------|-----------|-----------|----------------|
| **Kingman's Bound** | Kingman (1961) | $\mathbb{E}[W] \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2\mu}$ | High-load delay analysis |
| **Diffusion Limit** | Reiman (1984) | GI/G/1 → Reflected BM | Scaling to continuous model |
| **State Space Collapse** | Harrison (1998) | Multi-dim → 1-dim | Multi-GPU simplification |
| **Halfin-Whitt (QED)** | Halfin-Whitt (1981) | $\rho = 1 - \beta/\sqrt{n}$ → QED | GPU cluster scaling |

**Application**: Halfin-Whitt regime gives **optimal capacity scaling**:
- $n$ GPUs with $\rho = 1 - \beta/\sqrt{n}$
- Achieves both high utilization AND short delays
- $\sqrt{n}$ spare capacity rule

### 4. Processor Sharing

| Result | Author(s) | Statement | LLM Connection |
|--------|-----------|-----------|----------------|
| **PS = M/M/1 sojourn** | Kleinrock (1976) | E[T] = E[S]/(1-ρ) | Continuous batching delay |
| **Insensitivity** | Schassberger (1977) | Stationary depends only on mean | Model simplification |
| **DPS** | Fayolle et al. (1980) | Weighted PS analysis | Priority batching |
| **Sojourn Distribution** | Yashkov (1983) | Complete distribution of T | Tail latency |

**Key Insight**: PS insensitivity means stationary distribution depends **only on mean service time**, not higher moments. But LLM's KV cache growth **breaks insensitivity**.

### 5. Fork-Join Queues

| Result | Author(s) | Statement | LLM Connection |
|--------|-----------|-----------|----------------|
| **Response Time** | Nelson-Tantawi (1988) | $T = \max_i X_i$ dominates | Tensor parallel sync |
| **Bounds** | Varki et al. (2008) | Tight upper/lower bounds | Performance prediction |
| **Asymptotic** | Ko-Serfozo (2004) | Large-scale approximations | Many-GPU analysis |

**Application**: In tensor parallelism with $k$ GPUs:
$$T_{\text{layer}} = \max_{i=1}^k T_i + T_{\text{allreduce}}$$
Response time dominated by slowest shard.

### 6. Vacation & Setup

| Result | Author(s) | Statement | LLM Connection |
|--------|-----------|-----------|----------------|
| **Decomposition** | Doshi (1986) | $W = W_0 + W_V$ | GPU sleep overhead |
| **Polling Systems** | Levy-Sidi (1990) | Gated vs exhaustive | Multi-GPU polling |
| **Setup Time** | Welch (1964) | M/G/1 with setup | Model loading delay |

**Application**: GPU sleep state analysis:
- $W_0$ = queueing delay if always active
- $W_V$ = additional delay from vacation/setup
- Trade-off: energy savings vs latency penalty

### 7. State-Dependent Service

| Result | Author(s) | Statement | LLM Connection |
|--------|-----------|-----------|----------------|
| **M_n/G_n/1** | Abouee-Mehrizi (2016) | Birth-death equivalence | KV cache growth |
| **Optimal Policy** | OR Journal | Monotonic μ*(n) | Speed scaling |

**Key Gap**: Classical state-dependence is on **queue state** (total jobs). LLM has **per-job state** (each job's KV cache grows). This is a **fundamental modeling gap**.

## Topic-Specific Deep Dives

### Heavy Traffic: Proof Techniques

1. **Skorokhod Reflection**
   - Map RBM to queue process
   - Handle boundary behavior

2. **Weak Convergence**
   - Prohorov's theorem
   - Continuous mapping theorem

3. **Large Deviations**
   - Cramér's theorem
   - Rate function for rare events

### Processor Sharing: Key Equations

**M/G/1-PS Mean Response Time**:
$$\mathbb{E}[T] = \frac{\mathbb{E}[S]}{1 - \rho}$$

**Conditional Response Time** (given job size $x$):
$$\mathbb{E}[T | S=x] = \frac{x}{1 - \rho}$$

**Interpretation**: Each job "sees" service slowed by factor $(1-\rho)^{-1}$.

### Fork-Join: Bounds

**Upper Bound** (independent service):
$$\mathbb{E}[T_{FJ}] \leq \mathbb{E}[T_{M/M/1}] + \frac{H_k - 1}{\mu}$$

where $H_k = \sum_{i=1}^k 1/i$ is harmonic number.

**Implication**: Fork-join overhead grows as $O(\log k)$ with $k$ parallel shards.

## LLM-Specific Gaps Analysis

| Classical Assumption | LLM Reality | Required Extension |
|---------------------|-------------|-------------------|
| Queue-level state dependence | Per-job state (KV cache) | New model class |
| Memoryless service | KV cache accumulates | Markov-modulated service |
| Independent servers | Synchronization barriers | Fork-join with coupling |
| Product form networks | A/F coupling breaks BCMP | Non-product analysis |
| Insensitivity | Heavy-tailed tokens | Sensitivity analysis |

## Output Format

```markdown
# 深度排队论分析: [Topic]

## 1. 核心定理精确陈述

### 定理 [Name] ([Author Year])

**陈述**: [Mathematical statement]

**条件**: [Required assumptions]

**证明技术**: [Proof method]

**LLM 适用性**: ✅/⚠️/❌ + [解释]

## 2. 证明工具

| 工具 | 经典用途 | LLM 应用 |
|------|---------|---------|
| [工具] | [用途] | [应用] |

## 3. 与 LLM 的桥梁

| 经典结果 | LLM 问题 | 适配方法 |
|---------|---------|---------|
| [结果] | [问题] | [方法] |

## 4. 开放研究问题

1. [问题]: [所需工具]
```

## Begin

Parse `$ARGUMENTS` for topic focus, then:
1. Load relevant sections from theory_references.md
2. Launch Task agent for deep analysis
3. Generate structured output with precise theorem statements
4. Identify LLM-specific adaptations needed
5. Highlight open research opportunities
