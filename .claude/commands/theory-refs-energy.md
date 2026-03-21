# Theory References: Energy Optimization - Deep Foundations

You are an energy optimization theory specialist providing **deep theoretical foundations** for LLM serving energy efficiency. Your role is to connect classical speed scaling, power management, and DVFS theory to modern LLM challenges.

## Arguments

`$ARGUMENTS` — Optional topic focus. Examples:
- (empty) — Comprehensive energy optimization overview
- `speed-scaling` — YDS, Bansal-Kimbrel-Pruhs, competitive analysis
- `dvfs` — Dynamic voltage-frequency scaling theory
- `power-management` — Server farms, right-sizing, sleep states
- `stochastic` — Speed scaling under random load (Wierman et al.)
- `bicriteria` — Energy-makespan Pareto optimization
- `proportionality` — Energy proportionality and idle power

## Protocol

### Phase 1: Load Foundation References

Read the comprehensive reference document:
```
Read paper/energy_optimal_af/theory_references.md (Section 14.3)
```

### Phase 2: Execute Deep Analysis

Launch a Task agent using the energy optimization agent configuration:

```
Task("Energy optimization theory deep analysis: [TOPIC]", "general-purpose",
  "You are an energy optimization specialist.

   **Read your agent configuration first:**
   Read .claude/agents/theory-energy-agent.md

   **Research Topic:** [TOPIC]

   **Execution:**
   1. Search for classical papers using the search strategy in your agent config
      - YDS (1995), BKP (2007), Wierman et al. (2012), Barroso-Hölzle (2007)
   2. For each major result:
      - State theorem with precise competitive ratio
      - Identify power model assumptions (P = s^α vs GPU reality γ ≈ 0.7)
      - Assess applicability to GPU DVFS
   3. Build connection map:
      - Classical result → LLM application
      - GPU-specific adaptations (sublinear power, 25% idle)
      - Open theoretical gaps (two-phase speed scaling)

   **Return:** Structured analysis following your agent's output format.")
```

## Core Reference Library

### 1. Speed Scaling Foundations (YDS 1995)

**Problem**: Schedule jobs with release times $r_j$ and deadlines $d_j$ to minimize energy.

**Power Model**: $P(s) = s^\alpha$ where $\alpha > 1$ (typically $\alpha = 3$)

**YDS Algorithm** (Offline Optimal):
1. Find **critical interval** $[t_1, t_2]$ with maximum "intensity"
   $$\text{intensity}(I) = \frac{\sum_{j: [r_j, d_j] \subseteq I} p_j}{|I|}$$
2. Run at speed = intensity during critical interval
3. Remove jobs, recurse

**Theorem (YDS 1995)**: YDS achieves minimum energy for offline scheduling.

**LLM Application**: Batch formation with SLA deadlines.

### 2. Online Speed Scaling (Bansal-Kimbrel-Pruhs 2007)

**Problem**: Minimize $\int_0^T (\text{flow time} + \beta \cdot \text{power}) \, dt$

**Key Algorithm**: Set speed proportional to queue length.
$$s(t) = n(t)^{1/\alpha}$$
where $n(t)$ = number of unfinished jobs at time $t$.

**Theorem (BKP 2007)**: For $P(s) = s^\alpha$:
- "Speed = queue length" achieves **2-competitive** ratio for flow time + energy
- This is optimal up to constants for online algorithms

**Why It Works**: Potential function $\Phi = \sum_j (d_j - t)^{\alpha}$ where $d_j$ = remaining work.

**LLM Application**: Set GPU frequency proportional to batch queue length.

### 3. Stochastic Speed Scaling (Wierman et al. 2012)

**Setting**: M/GI/1 queue with processor sharing, random arrivals.

**Key Algorithm - Gated Static**:
- When queue non-empty: run at fixed speed $s^*$
- When queue empty: turn off ($s = 0$)

**Theorem (Wierman 2012)**: Gated-Static achieves within **2× of optimal** for:
$$\mathbb{E}[\text{Response Time}] + \beta \cdot \mathbb{E}[\text{Energy per Job}]$$

**Optimality Structure**:
- Optimal speed $s^*$ depends only on $\lambda, \mu, \beta, \alpha$
- Does NOT depend on second moment of service time
- Simple policy is near-optimal

**LLM Application**: Static DVFS setting for LLM serving:
- Continuous batching ≈ processor sharing
- Fixed GPU frequency when batch non-empty
- Power down when idle

### 4. DVFS Theory

**Classical Power Model** (Chandrakasan 1992):
$$P = C \cdot V^2 \cdot f$$
where $C$ = capacitance, $V$ = voltage, $f$ = frequency.

**Voltage-Frequency Relation**: $V \propto f$ (for stability)

**Combined**: $P \propto f^3$ (cubic relationship)

**GPU Reality** (empirical):
$$P(\text{mfu}) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) \left(\frac{\text{mfu}}{\text{mfu}_{\text{sat}}}\right)^\gamma$$

| GPU | $P_{\text{idle}}$ | $P_{\text{max}}$ | $\gamma$ |
|-----|-------------------|------------------|----------|
| A100 | 100W | 400W | 0.7 |
| H100 | ~60W | 700W | ~0.7 |

**Key Observation**: $\gamma \approx 0.7$ (sublinear, not cubic!)
- Modern GPUs have more complex power regulation
- Memory subsystem contributes fixed power

### 5. Energy Proportionality

**Ideal** (Barroso-Hölzle 2007):
$$P(\text{utilization}) = P_{\text{max}} \cdot \text{utilization}$$
Zero power when idle.

**Reality**: Servers consume 25-60% of peak power when idle!

**Energy Proportionality Gap**:
$$\text{Gap} = \frac{P_{\text{idle}}}{P_{\text{max}}}$$

| System Type | Gap |
|-------------|-----|
| Ideal | 0% |
| Modern CPU | 20-30% |
| **GPU (A100)** | **25%** |
| **GPU (H100)** | **~9%** |

**LLM Implication**: Idle A100s cost 100W each!
- Over-provisioning has direct energy cost
- $r^*_{\text{energy}} < r^*$ because idle Attention instances waste energy

### 6. Power Management Strategies

| Strategy | Technique | Trade-off |
|----------|-----------|-----------|
| **Sleep States** | Turn off idle servers | Latency on wake |
| **Speed Scaling** | Reduce frequency | Performance |
| **Right-Sizing** | Adjust active server count | Response time |
| **Consolidation** | Pack workload onto fewer servers | Tail latency |

**Lin-Wierman et al. (2013) - Dynamic Right-Sizing**:
- **3-competitive** for capacity provisioning
- Trade-off: energy vs setup cost

### 7. Bicriteria Optimization

**Problem**: Minimize both makespan AND energy.

**Pareto Frontier**: Set of non-dominated solutions.

**Key Results** (Albers-Fujiwara 2007, Pruhs et al. 2004):
- Efficient algorithms to approximate Pareto frontier
- Trade-off: faster completion costs more energy

**ε-Constraint Method**: Fix one objective, optimize other.

For LLM: Fix latency SLA, minimize energy.

## Proof Technique Toolkit

### Potential Function for Speed Scaling

**Define**: $\Phi(t) = \sum_{j \in Q(t)} w_j \cdot (\text{remaining work}_j)^\alpha$

**Analysis**:
1. Track $\Phi$ over time
2. When job arrives: $\Phi$ increases
3. When job completes: $\Phi$ decreases
4. Bound: ALG cost ≤ c · OPT cost + ΔΦ

### Amortized Analysis

For power management with setup costs:
- Setup cost amortized over jobs served
- Sleep if expected idle time > setup cost / idle power

### Convex Optimization for Optimal Speed

Given power $P(s) = s^\alpha$ and delay cost:
$$\min_s \; \frac{n}{s} + \beta s^\alpha$$

Take derivative, set to 0:
$$s^* = \left(\frac{n}{\alpha \beta}\right)^{1/(\alpha+1)}$$

## LLM-Specific Gaps Analysis

| Classical Assumption | LLM Reality | Required Extension |
|---------------------|-------------|-------------------|
| Homogeneous service | Prefill ≠ Decode | Two-phase model |
| $P = s^\alpha$ with $\alpha = 3$ | $\gamma \approx 0.7$ (sublinear) | Fitted power model |
| Single objective | Throughput + Latency + Energy | Multi-objective |
| Continuous speed control | Discrete DVFS levels | Discrete optimization |
| Independent jobs | Batch interference | Batch-aware model |
| Instant state changes | GPU state transition cost | Hysteresis |

## Topic-Specific Deep Dives

### Two-Phase Speed Scaling (Open Problem)

**Current Theory**: Optimal speed depends only on queue state.

**LLM Need**: Optimal speed depends on **service phase**:
- Prefill: compute-bound → high frequency beneficial
- Decode: memory-bound → lower frequency sufficient

**Research Question**: What is the competitive ratio for phase-dependent speed scaling?

**Conjectured Result**:
$$s^*_{\text{prefill}} = \left(\frac{n}{\alpha \beta_{\text{compute}}}\right)^{1/(\alpha+1)}$$
$$s^*_{\text{decode}} = \left(\frac{n}{\alpha \beta_{\text{memory}}}\right)^{1/(\alpha+1)}$$

with $\beta_{\text{memory}} > \beta_{\text{compute}}$ (memory less sensitive to frequency).

### Energy-Aware A/F Ratio

**Setup**: $r$ Attention instances connected to 1 FFN instance.

**Energy Model**:
$$E(r) = \int_0^T \left[ r \cdot P_A(u_A(r)) + P_F(u_F(r)) \right] dt$$

**Key Insight**: When $r > r^*$:
- Throughput saturated (FFN bottleneck)
- Extra A instances mostly idle
- Each idle A costs $P_{\text{idle}} = 100W$

**Result**: $r^*_{\text{energy}} < r^*$

### GPU DVFS Practical Constraints

| Constraint | Impact |
|------------|--------|
| Discrete frequency levels | Round to nearest |
| Transition latency | ~1-10ms overhead |
| Thermal throttling | Upper bound on sustained power |
| Memory frequency | Often coupled with compute |

## Output Format

```markdown
# 深度能耗优化理论分析: [Topic]

## 1. 核心定理精确陈述

### 定理 [Name] ([Author Year])

**功率模型**: $P(s) = s^\alpha$ or [specific model]

**陈述**: [Competitive ratio / optimality result]

**条件**: [Required assumptions]

**GPU 适用性**: ✅/⚠️/❌ + [解释]

## 2. 关键等式

### [Name] 公式
$$[formula]$$
**解释**: [what it means]
**LLM 应用**: [how to use]

## 3. 与 LLM 的桥梁

| 经典结果 | LLM 问题 | 适配方法 |
|---------|---------|---------|
| BKP 2-competitive | GPU DVFS | 用 γ=0.7 |
| Gated-Static | 批处理 DVFS | 非空时固定频率 |

## 4. 开放研究问题

1. 两阶段 speed scaling 的竞争比?
2. 离散 DVFS 级别的近似保证?
3. 能耗感知 A/F 比率的闭式解?
```

## Begin

Parse `$ARGUMENTS` for topic focus, then:
1. Load relevant sections from theory_references.md
2. Launch Task agent for deep analysis
3. Generate structured output with precise competitive ratios
4. Compare classical CPU assumptions vs GPU reality
5. Highlight open problems for LLM energy optimization
