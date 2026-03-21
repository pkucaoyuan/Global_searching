# Self-Dispatch Protocol for Compound Skills

**Any skill with 3+ independent phases SHOULD use this protocol to spawn subagents.**

---

## When to Self-Dispatch

A skill should spawn Task subagents when:
1. It has **3+ phases** that are **independent** (no data dependency between them)
2. Each phase reads multiple files and produces a sub-report
3. Running all phases inline would bloat the current context

A skill should NOT self-dispatch when:
- It has <3 phases
- Phases depend on each other (sequential pipeline)
- It's interactive (needs user feedback mid-execution)
- It's already running as a subagent (avoid recursive spawning)

---

## Recursion Guard

**CRITICAL**: Before spawning subagents, check if you are ALREADY a subagent.

Detection: If the skill was invoked via a Task tool prompt (not via Skill tool or user `/command`), you are a subagent. In that case, **execute directly — do NOT spawn further subagents.**

Rule: **Maximum dispatch depth = 2**
```
Level 0: User or Orchestrator
Level 1: Skill spawns Task subagents for phases     ← OK
Level 2: Subagent executes phase directly            ← STOP, no more spawning
```

---

## Execution Pattern

### Step 1: Shared Setup (in current context)

Before spawning subagents, do the shared work that all phases need:

```
1. Resolve paper name (Step 0A from unified_protocol.md)
2. Read state docs (Step 2 from unified_protocol.md)
3. Write state doc checkpoint summary (Step 2.5)
```

This avoids every subagent redundantly reading the same state docs.

### Step 2: Spawn Subagents for Independent Phases

Use Task tool with `general-purpose` subagent. Send **all independent phases in a single message** for parallel execution.

**Task prompt template for each phase:**

```
You are executing Phase N of [skill-name]: [phase description].

Context (from parent):
- Paper directory: [PAPER_DIR]
- Paper state directory: [PAPER_STATE_DIR]
- State doc summary: [paste checkpoint summary from Step 2.5]

Your task:
- Read the following .tex files: [specific files for this phase]
- Check for: [specific checks for this phase]
- Return a structured sub-report:
  - Issues found (location, severity, description)
  - Count
```

**Key**: Include the state doc checkpoint summary in the prompt so subagents don't need to re-read state docs.

### Step 3: Aggregate Results

After all subagents return:
1. Collect sub-reports from each phase
2. Merge into a single report (dedup if needed)
3. Sort by severity
4. Add summary counts
5. Output the final report

### Step 4: Cascade (Optional)

If issues were found and the skill knows which downstream skill can fix them:

```
Found 3 symbol conflicts → Suggest: /fix-issues symbols
Found 2 broken refs → Suggest: /fix-issues refs
```

Do NOT auto-invoke fix skills unless the user asked for it. Just suggest.

---

## Phase Declaration Format

Each compound skill declares its phases using this table:

```markdown
## Self-Dispatch Phases

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 1 | [name] | Yes/No | [files] | [checks] |
| 2 | [name] | Yes/No | [files] | [checks] |
| ...

**Parallel group**: Phases [1,2,3] can run in parallel.
**Sequential**: Phase 4 depends on Phase 1-3 results.
```

---

## Begin Section Templates

Skills with Self-Dispatch Phases tables MUST use one of these two templates for their `## Begin` section, instead of writing custom dispatch logic.

### Template A: All Phases Parallel (no setup phase)

Use when ALL phases are independent (e.g., check-cross-references, check-ms-style, check-or-style).

```
## Begin

**Dispatch**: All phases parallel (Template A from `self_dispatch_protocol.md`).

1. Follow unified protocol Steps 0A–2.5 (resolve paper, read state docs, write checkpoint)
2. **Recursion guard**: If invoked via Task tool → execute all phases inline, skip dispatch
3. **Dispatch**: Spawn [N] parallel Task subagents (one per phase), each receives:
   - Phase table row (files to read, what to check)
   - State doc checkpoint summary
   - Paper directory paths
4. **Aggregate**: Merge [N] sub-reports → deduplicate → sort by severity
5. Output final report + Next Steps footer
```

### Template B: Setup Phase + Parallel Phases

Use when Phase 0 must complete before dispatching Phases 1–N (e.g., check-paper-consistency, check-term-consistency, check-content-redundancy, check-figures-tables).

```
## Begin

**Dispatch**: Setup (Phase 0) → parallel (Phases 1–N) (Template B from `self_dispatch_protocol.md`).

1. Follow unified protocol Steps 0A–2.5 (resolve paper, read state docs, write checkpoint)
2. **Execute Phase 0** inline — produces [registry/map/index] needed by all subsequent phases
3. **Recursion guard**: If invoked via Task tool → execute all remaining phases inline, skip dispatch
4. **Dispatch**: Spawn [N] parallel Task subagents (one per phase), each receives:
   - Phase table row (files to read, what to check)
   - Phase 0 output (registry/map)
   - State doc checkpoint summary
5. **Aggregate**: Merge [N] sub-reports → deduplicate → sort by severity
6. Output final report + Next Steps footer
```

### Which Template to Use

| Skill | Template | Setup Phase | Parallel Phases |
|-------|----------|-------------|-----------------|
| check-cross-references | A | — | 1-7 (all) |
| check-ml-style | A | — | 1-7 (all) |
| check-ms-style | A | — | 1-6 (all) |
| check-or-style | A | — | 1-6 (all) |
| check-content-redundancy | B | Phase 0: result registry | 1-5 |
| check-paper-consistency | B | Phase 0: symbol registry | 1-6 |
| check-term-consistency | B | Phase 0: term usage map | 1-5 |
| check-figures-tables | B | Phase 0: figure/table discovery | 1-5 |

---

## Example: check-cross-references (Template A)

```
Step 1 (shared): Read cross_references.md, results.md → checkpoint summary

Step 2 (parallel subagents):
  Task 1: "Check result references in all .tex files. Verify theorem/proposition numbers match results.md."
  Task 2: "Check numerical values across abstract, intro, experiments, conclusion."
  Task 3: "Check forward references (all \ref{} point to existing \label{})."
  Task 4: "Check citation consistency (all \cite{} have bib entries)."

Step 3: Aggregate 4 sub-reports → final cross-reference report

Step 4: Cascade → "3 issues found. Suggest: /fix-issues refs"
```

## Example: check-paper-consistency (Template B)

```
Step 1 (shared): Read symbols.md, results.md → checkpoint summary

Step 2 (Phase 0 inline): Build symbol registry from all .tex files → registry output

Step 3 (parallel subagents, each receives registry):
  Task 1: "Detect symbol conflicts — same symbol with different meanings."
  Task 2: "Check concept consistency — same concept explained differently."
  Task 3: "Check model-algorithm consistency — setup matches algorithm."
  Task 4: "Check cross-reference accuracy — forward/backward refs valid."

Step 4: Aggregate 4 sub-reports → final consistency report

Step 5: Cascade → "2 symbol conflicts found. Suggest: /fix-issues symbols"
```

---

## Cost-Benefit Rule

Self-dispatch has overhead (Task tool calls, prompt duplication). Only use it when:

- **Benefit**: Each phase reads 3+ files AND the skill has 3+ independent phases
- **Cost**: Each Task subagent consumes ~2000 tokens of prompt overhead

If the entire skill would take <5000 tokens to execute inline, skip self-dispatch and run directly.
