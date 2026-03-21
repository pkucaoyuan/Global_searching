# Theory Research Agents

This directory contains specialized agent configurations for deep theoretical research on LLM serving systems.

## Dispatch Protocol

All agents follow the **orchestrator_protocol.md** and **self_dispatch_protocol.md** patterns.

### Key Principles

1. **Task tool dispatch**: Agents are launched via Task tool with `subagent_type: "general-purpose"`
2. **Recursion guard**: Maximum dispatch depth = 2 (prevents infinite spawning)
3. **Self-dispatch phases**: Agents can spawn parallel subagents for independent phases
4. **Structured output**: Each agent has a defined output format

---

## Agent Registry

| Agent | File | Expertise | Dispatch Pattern |
|-------|------|-----------|------------------|
| **Queueing Theory** | `theory-queueing-agent.md` | Heavy traffic, PS, fork-join, vacation | Template A (all parallel) |
| **Scheduling Theory** | `theory-scheduling-agent.md` | Parallel machines, online algorithms, LP | Template A (all parallel) |
| **Energy Optimization** | `theory-energy-agent.md` | Speed scaling, DVFS, power management | Template A (all parallel) |
| **Cross-Domain** | `theory-cross-domain-agent.md` | Multi-domain synthesis | Template B (setup + parallel) |

---

## How to Dispatch Agents

### Via Skill Commands (Recommended)

The skills in `.claude/commands/` handle dispatch automatically:

```bash
# Single-domain analysis
/theory refs queueing heavy-traffic
/theory refs scheduling approximation
/theory refs energy speed-scaling

# Cross-domain synthesis (spawns 4 parallel agents)
/theory refs deep all A/F ratio
```

### Via Direct Task Tool

For programmatic dispatch, follow this pattern:

```python
Task(
  subagent_type: "general-purpose",
  description: "Queueing theory analysis for heavy-traffic",
  prompt: """
    You are executing the theory-queueing-agent.

    Instructions: Read and follow .claude/agents/theory-queueing-agent.md
    Protocol: Read .claude/commands/_shared/orchestrator_protocol.md for dispatch rules

    Research Topic: heavy-traffic

    Context:
    - Reference document: paper/energy_optimal_af/theory_references.md (Section 14.1)

    Your task:
    1. Use WebSearch to find classical queueing theory papers
    2. State theorems precisely with conditions
    3. Assess LLM applicability
    4. Return structured analysis following the output format

    Return a structured report.
  """
)
```

**Key elements in every Task prompt**:
1. Which agent .md file to read for instructions
2. Reference to orchestrator_protocol.md
3. Research topic
4. Context (reference documents)
5. Expected output format

### Parallel Multi-Domain Analysis

For cross-domain analysis, launch multiple agents in **ONE message**:

```python
// All three run simultaneously in parallel:
Task("Queueing for A/F", "general-purpose", "Read .claude/agents/theory-queueing-agent.md...")
Task("Scheduling for A/F", "general-purpose", "Read .claude/agents/theory-scheduling-agent.md...")
Task("Energy for A/F", "general-purpose", "Read .claude/agents/theory-energy-agent.md...")
Task("Cross-domain synthesis", "general-purpose", "Read .claude/agents/theory-cross-domain-agent.md...")
```

---

## Self-Dispatch Protocol

Agents with 3+ independent phases use `self_dispatch_protocol.md`:

### Template A: All Phases Parallel

Used by: queueing, scheduling, energy agents

```
1. Load reference document
2. Parse topic → identify relevant phases
3. Recursion guard: If already subagent → execute inline
4. Dispatch: Spawn N parallel Task subagents (one per phase)
5. Aggregate: Merge sub-reports
6. Return structured output
```

### Template B: Setup + Parallel

Used by: cross-domain agent

```
1. Load reference document (Phase 0)
2. Identify cross-domain connections
3. Recursion guard: If already subagent → execute inline
4. Dispatch: Spawn 3 parallel domain agents
5. Synthesis: Aggregate + find connections/conflicts
6. Return unified framework
```

---

## Recursion Guard

**CRITICAL**: To prevent infinite spawning:

- **Detection**: If invoked via Task tool prompt → you are a subagent
- **Rule**: Maximum dispatch depth = 2
- **Action**: If already a subagent → execute directly, do NOT spawn further subagents

```
Level 0: User or Orchestrator (/theory refs ...)
Level 1: Skill spawns Task subagent (theory-queueing-agent)  ← OK to spawn
Level 2: Subagent executes directly                           ← STOP, no more spawning
```

---

## Agent Capabilities

### Queueing Theory Agent
- **Self-Dispatch Phases**: Heavy Traffic, Product Form, Processor Sharing, Fork-Join, Vacation
- **Search Strategy**: Kingman, Halfin-Whitt, Jackson, BCMP, Kleinrock, Doshi
- **LLM Mappings**: Continuous batching ↔ PS, tensor parallel ↔ fork-join

### Scheduling Theory Agent
- **Self-Dispatch Phases**: Parallel Machines, Online Algorithms, Approximation, Flow Time, Learning-Augmented, Resource Augmentation
- **Search Strategy**: Graham, LST, Sleator-Tarjan, Kalyanasundaram-Pruhs, Purohit
- **LLM Mappings**: A/F ↔ unrelated machines, admission ↔ online scheduling

### Energy Optimization Agent
- **Self-Dispatch Phases**: Speed Scaling, Stochastic, DVFS Theory, Power Management, Bicriteria
- **Search Strategy**: YDS, BKP, Wierman, Barroso-Hölzle, Chandrakasan
- **LLM Mappings**: GPU DVFS ↔ speed scaling, batch control ↔ power management

### Cross-Domain Agent
- **Self-Dispatch Phases**: Setup → (Queueing, Scheduling, Energy in parallel) → Synthesis
- **Synthesis Methods**: Connection identification, conflict resolution, unified framework
- **Output**: Multi-perspective analysis with trade-offs

---

## Knowledge Sources

All agents reference:
1. `paper/energy_optimal_af/theory_references.md` — Comprehensive 70+ reference document
2. `.claude/commands/_shared/orchestrator_protocol.md` — Dispatch rules
3. `.claude/commands/_shared/self_dispatch_protocol.md` — Phase management
4. WebSearch for classical and recent papers

---

## Output Format

All agents produce structured output with:
- **Core Theorems**: Precise statements with conditions
- **Proof Techniques**: Methods used in proofs
- **LLM Applicability**: Where assumptions hold/break (✅/⚠️/❌)
- **Gap Analysis**: Required extensions for LLM
- **Open Problems**: Research opportunities

---

## Integration with Paper Writing

These agents support the paper writing workflow:
1. Use `/theory refs [domain]` to gather theoretical foundations
2. Results inform `paper/energy_optimal_af/main.tex`
3. References added to `theory_references.md`
4. Proofs verified via `/theory verify [theorem]`

---

## Adding New Agents

To add a new specialized agent:

1. Create `theory-[domain]-agent.md` in this directory with:
   - **Dispatch Protocol** section with Task prompt pattern
   - **Self-Dispatch Phases** table
   - **Execution Protocol** with recursion guard
   - **Search Strategy** with WebSearch queries
   - **Output Format** template

2. Add routing in `.claude/commands/theory.md`

3. Optionally create dedicated skill in `.claude/commands/theory-refs-[domain].md`
