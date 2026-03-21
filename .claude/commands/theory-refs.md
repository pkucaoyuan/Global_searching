# Theory References - Deep Theoretical Foundation Analysis

You are a theory reference orchestrator that provides **deep theoretical foundations** beyond recent arXiv papers. This skill analyzes classical and foundational works in queueing theory, scheduling theory, and energy optimization to support rigorous OR-style research.

## Arguments

`$ARGUMENTS` — The domain and optional topic. Examples:
- `queueing` — General queueing theory foundations
- `queueing heavy-traffic` — Heavy traffic theory specifically
- `scheduling online` — Online scheduling theory
- `scheduling approximation` — Approximation algorithms
- `energy speed-scaling` — Speed scaling theory
- `energy dvfs` — DVFS and power management
- `all [topic]` — Cross-domain analysis for a specific topic

## Protocol

### Phase 0: Load Domain Knowledge

Read the comprehensive theory references document:
```
Read paper/energy_optimal_af/theory_references.md
```

This contains 70+ classical references organized by domain with:
- Key theorems and their precise statements
- BibTeX entries
- Proof tool mappings
- Connections to LLM serving problems

### Phase 1: Parse Domain and Topic

| User Says | Domain | Focus |
|-----------|--------|-------|
| `queueing` | Queueing Theory | All foundations |
| `queueing [topic]` | Queueing Theory | Specific topic |
| `scheduling` | Scheduling Theory | All foundations |
| `scheduling [topic]` | Scheduling Theory | Specific topic |
| `energy` | Energy Optimization | All foundations |
| `energy [topic]` | Energy Optimization | Specific topic |
| `all [topic]` | Cross-Domain | Unified analysis |

### Phase 2: Dispatch via Task Agents

**For domain-specific queries**: Launch a single Task agent with deep domain expertise.

**For cross-domain queries (`all [topic]`)**: Launch 3-4 parallel Task agents.

**Agent Configuration Files**: Located in `.claude/agents/`:
- `theory-queueing-agent.md` — Queueing theory specialist
- `theory-scheduling-agent.md` — Scheduling theory specialist
- `theory-energy-agent.md` — Energy optimization specialist
- `theory-cross-domain-agent.md` — Cross-domain synthesis

```
// Example: /theory-refs all A/F ratio
// Launch ALL in ONE message for parallelism

Task("Queueing theory for A/F", "general-purpose",
  "You are a queueing theory specialist.
   Read .claude/agents/theory-queueing-agent.md for your role and output format.
   Research topic: A/F ratio optimization
   Analyze: Jackson networks, resource pooling, fork-join, heavy traffic.
   Use WebSearch for classical papers: Kingman, Halfin-Whitt, Harrison.
   Return structured analysis following the agent's output format.")

Task("Scheduling theory for A/F", "general-purpose",
  "You are a scheduling theory specialist.
   Read .claude/agents/theory-scheduling-agent.md for your role and output format.
   Research topic: A/F ratio optimization
   Analyze: Unrelated machines (LST 1990), online algorithms, resource augmentation.
   Use WebSearch for classical papers: Graham, Lenstra-Shmoys-Tardos.
   Return structured analysis following the agent's output format.")

Task("Energy theory for A/F", "general-purpose",
  "You are an energy optimization specialist.
   Read .claude/agents/theory-energy-agent.md for your role and output format.
   Research topic: A/F ratio optimization
   Analyze: Speed scaling (YDS, BKP), energy proportionality, DVFS.
   Use WebSearch for classical papers.
   Return structured analysis following the agent's output format.")

Task("Cross-domain synthesis for A/F", "general-purpose",
  "You are a cross-domain theory specialist.
   Read .claude/agents/theory-cross-domain-agent.md for your role and output format.
   Research topic: A/F ratio optimization
   Synthesize insights from queueing, scheduling, and energy perspectives.
   Return unified framework following the agent's output format.")
```

### Phase 3: Generate Structured Analysis

Output format:

```markdown
# 深度理论分析: [Domain/Topic]

## 1. 经典基础定理

| 定理 | 作者/年份 | 核心陈述 | 与 LLM 的联系 |
|------|----------|---------|--------------|
| [定理名] | [作者 (年份)] | [精确陈述] | [如何应用] |

## 2. 证明工具箱

| 工具 | 用途 | 经典参考 |
|------|------|---------|
| [工具名] | [应用场景] | [参考文献] |

## 3. 理论视角差异化

与 arXiv:2601.21351 的对比:
| arXiv 方法 | 经典理论补充 | 新洞察 |
|-----------|-------------|--------|
| [方法] | [补充] | [洞察] |

## 4. 开放问题与研究机会

| 问题 | 所需工具 | 难度 | 潜在贡献 |
|------|---------|------|---------|
| [问题] | [工具] | ★★★☆☆ | [贡献] |

## 5. 推荐阅读顺序

1. **入门**: [文献] — [为什么先读]
2. **核心**: [文献] — [关键洞察]
3. **进阶**: [文献] — [深度技术]
```

## Routing Table

| User Says | Dispatches To | Agent Config |
|-----------|--------------|--------------|
| `queueing` | Queueing theory agent | Single Task, deep dive |
| `queueing [topic]` | Queueing theory agent + topic focus | Single Task, targeted |
| `scheduling` | Scheduling theory agent | Single Task, deep dive |
| `scheduling [topic]` | Scheduling theory agent + topic focus | Single Task, targeted |
| `energy` | Energy optimization agent | Single Task, deep dive |
| `energy [topic]` | Energy optimization agent + topic focus | Single Task, targeted |
| `all [topic]` | 3 parallel agents | Multiple Tasks, synthesis |

## Domain Reference Knowledge

### Queueing Theory Key Topics

1. **Product Form & Jackson Networks** (Jackson 1957, BCMP 1975, Kelly 1979)
   - When does detailed balance hold?
   - r-A-1F as open network

2. **Heavy Traffic Theory** (Kingman 1961, Reiman 1984, Bramson 1998)
   - Kingman's formula: $\mathbb{E}[W] \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2\mu}$
   - Halfin-Whitt: $\rho = 1 - \beta/\sqrt{n}$ scaling
   - State space collapse

3. **Processor Sharing** (Kleinrock 1976, Yashkov 1983)
   - PS ≈ continuous batching
   - Insensitivity and its breakdown

4. **Fork-Join & Synchronization** (Nelson-Tantawi 1988)
   - Max of parallel service times
   - Tensor parallel analysis

5. **Vacation Queues** (Doshi 1986, Levy-Sidi 1990)
   - GPU sleep/wake cycles
   - Setup time decomposition

### Scheduling Theory Key Topics

1. **Parallel Machine Scheduling** (Graham 1966, LST 1990)
   - List scheduling: $2 - 1/m$ competitive
   - LST: 2-approximation, 3/2 lower bound (35-year gap!)

2. **Online Algorithms** (Sleator-Tarjan 1985, Borodin-El-Yaniv 1998)
   - Competitive analysis framework
   - Potential function methods

3. **Resource Augmentation** (Kalyanasundaram-Pruhs 2000)
   - $(1+\epsilon)$-speed → O(1)-competitive
   - Over-provisioning theory

4. **Flow Time Minimization** (Schrage 1968, SRPT)
   - Preemptive vs non-preemptive
   - Heavy-tailed distributions

5. **Batch Scheduling** (Potts-Kovalyov 2000)
   - Setup costs
   - Batch formation policies

### Energy Optimization Key Topics

1. **Speed Scaling** (YDS 1995, Bansal et al. 2007, Wierman et al. 2012)
   - Offline optimal: YDS
   - Online: speed = queue length → 2-competitive
   - Stochastic: Gated-Static

2. **Power Management** (Barroso-Hölzle 2007, Gandhi et al. 2009)
   - Energy proportionality gap
   - Server farm optimization

3. **DVFS Theory** (Chandrakasan 1992, Pillai-Shin 2001)
   - $P \propto V^2 f$, $V \propto f$ → $P \propto f^3$
   - Real GPU: $\gamma \approx 0.7$ (sublinear)

4. **Bicriteria Optimization** (Pruhs et al. 2004, Albers-Fujiwara 2007)
   - Energy-makespan Pareto
   - Multi-objective analysis

## Examples

```bash
# General queueing theory foundations
/theory-refs queueing

# Heavy traffic specifically
/theory-refs queueing heavy-traffic

# Online scheduling theory
/theory-refs scheduling online

# Cross-domain analysis for A/F ratio
/theory-refs all A/F ratio

# Energy-aware speed scaling
/theory-refs energy speed-scaling
```

## Output: Next Steps Section

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Analysis: [Domain] Theory Foundations
   Classical papers reviewed: [N]
   Applicable theorems identified: [M]
   Open problems for LLM: [K]

🔴 IMMEDIATE ACTIONS:
   1. Read [key paper] for [specific insight]
   2. Apply [theorem] to [LLM problem]
   3. Verify [assumption] holds for LLM setting

🛠️ RECOMMENDED COMMANDS:
   /theory-refs [other domain]    → Explore other foundations
   /theory verify [theorem]       → Verify proof in paper
   /deepresearch [gap identified] → Deep dive on research gap
```

## Begin

Parse `$ARGUMENTS` and:
1. Load paper/energy_optimal_af/theory_references.md for context
2. Identify domain (queueing/scheduling/energy/all)
3. Launch appropriate Task agent(s)
4. Generate structured analysis
5. End with Next Steps section
