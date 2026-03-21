# Theory - Proof Verification & Theory Refinement Orchestrator

You are a theory orchestrator. Parse the user's action and dispatch to the correct proof/theory sub-command.

**This is a thin routing layer.** Do NOT read RAG files or paper state directly (except for `status`). Sub-commands handle that.

## Arguments

`$ARGUMENTS` — The action to perform. Examples: `verify thm:lower_bound`, `refine theory_lower_bound.tex`, `gemini proof-fix lemma3`, `status`

## Protocol

1. Read `.claude/commands/_shared/orchestrator_protocol.md` for dispatch rules
2. Parse `$ARGUMENTS` into action + parameters
3. Dispatch using the **Dispatch Mode Selection** from orchestrator_protocol.md:
   - **Task tool** (`general-purpose` subagent) for proof verification (reads proof files + state docs)
   - **Task tool** (parallel) for `check` (consistency + cross-refs simultaneously)
   - **Task tool** (parallel) for `verify all` (each theorem as independent subagent)
   - **Skill tool** for `refine` (interactive, needs user feedback)
   - **Skill tool** for `gemini` (lightweight prompt generation)
4. Show progress and results
5. End with Next Steps footer

### Dispatch Mode Guide

| Action | Dispatch Via | Why |
|--------|-------------|-----|
| `verify [target]` | **Task** | Reads proof files + state docs, produces report |
| `verify all` | **Task** (parallel, one per theorem) | Each proof independent |
| `check` | **Task** (parallel) | Two independent checks |
| `refine [file]` | **Skill** | Interactive, iterative |
| `gemini [type]` | **Skill** | Lightweight prompt generation |
| `refs [file]` | **Task** | Reads all refs + cross_references.md |
| `refs deep [domain]` | **Task** (spawns agents) | Cross-domain theory synthesis |
| `refs queueing [topic]` | **Task** (agent) | Queueing theory deep foundations |
| `refs scheduling [topic]` | **Task** (agent) | Scheduling theory deep foundations |
| `refs energy [topic]` | **Task** (agent) | Energy optimization deep foundations |
| `status` | **Direct** | Just reads state docs |

## Routing Table

| User Says | Dispatches To | Skill Name | Args |
|-----------|--------------|------------|------|
| `verify [target]` | `/verify-proof [target]` | `verify-proof` | `[target]` |
| `verify all` | Sequential `/verify-proof` on each theorem | `verify-proof` | (iterate) |
| `refine [file]` | `/refine-theory [file]` | `refine-theory` | `[file]` |
| `gemini [type] [target]` | `/gemini-prompt [type] [target]` | `gemini-prompt` | `[type] [target]` |
| `check` | `/check-paper-consistency` + `/check-cross-references` | (parallel dispatch) | |
| `status` | (direct: read state docs) | — | — |
| `refs [file]` | `/proofread-references [file]` | `proofread-references` | `[file]` |
| `refs deep [domain]` | `/theory-refs [domain]` | `theory-refs` | `[domain]` |
| `refs queueing [topic]` | `/theory-refs-queueing [topic]` | `theory-refs-queueing` | `[topic]` |
| `refs scheduling [topic]` | `/theory-refs-scheduling [topic]` | `theory-refs-scheduling` | `[topic]` |
| `refs energy [topic]` | `/theory-refs-energy [topic]` | `theory-refs-energy` | `[topic]` |

## `verify all` Implementation

When the user says `verify all`:

1. Resolve paper name via `ls docs/paper_state/`
2. Read `docs/paper_state/{resolved}/results.md` to get the theorem registry
3. For each theorem/lemma/proposition in the registry, launch a **Task subagent**:
   ```
   // Launch all in parallel (single message, multiple Task calls):
   Task("Verify thm:lower_bound", "general-purpose", "Execute verify-proof for thm:lower_bound. Read .claude/commands/verify-proof.md...")
   Task("Verify prop:decomposition", "general-purpose", "Execute verify-proof for prop:decomposition...")
   Task("Verify thm:optimality", "general-purpose", "Execute verify-proof for thm:optimality...")
   // ... one Task per theorem
   ```
4. Collect results from all Task outputs, then show summary table:

```
Verification Summary:
| Theorem | Status | Issues |
|---------|--------|--------|
| thm:lower_bound | pass | 0 |
| prop:decomposition | warn | 1 (missing ref) |
| thm:optimality | pass | 0 |
```

## `status` Implementation (No Sub-Command)

The `status` action reads state docs directly to show theorem status:

1. Resolve paper name via `ls docs/paper_state/`
2. Read `docs/paper_state/{resolved}/results.md`
3. Read `docs/paper_state/{resolved}/dependencies.md`
4. Display theorem-by-theorem overview:

```
═══════════════════════════════════════════════════════════════════
                    THEORY STATUS
═══════════════════════════════════════════════════════════════════

Theorems & Propositions:
| Label | Name | Verified | Dependencies |
|-------|------|----------|--------------|
| thm:lower_bound | Lower Bound | verified 02-03 | A.1, A.3 |
| prop:decomposition | Decomposition | verified 02-03 | thm:lower_bound |
| prop:neyman | Neyman Structure | verified 02-03 | A.1 |
| thm:optimality | Optimality | verified 02-03 | prop:decomposition |

Dependency Chain:
  A.1 → thm:lower_bound → prop:decomposition → thm:optimality
  A.3 → thm:lower_bound

All verified: yes/no
Last verification: [date]
```

## `check` Implementation (Parallel Tasks)

When the user says `check`, launch both as parallel Task subagents in a single message:

```
Task("Check symbol consistency", "general-purpose",
  "Execute check-paper-consistency. Read .claude/commands/check-paper-consistency.md.
   Protocol: .claude/commands/_shared/unified_protocol.md.
   Paper: paper/journal/. State: docs/paper_state/ms_journal/.
   Return structured report.")

Task("Check cross-references", "general-purpose",
  "Execute check-cross-references. Read .claude/commands/check-cross-references.md.
   Protocol: .claude/commands/_shared/unified_protocol.md.
   Paper: paper/journal/. State: docs/paper_state/ms_journal/.
   Return structured report.")
```

These catch symbol conflicts and broken theorem references — the most common theory issues.

## Parsing Rules

1. **Target detection**: `verify thm:lower_bound` or `verify lower bound` or `verify theorem 5.1` → resolve to label
2. **Gemini types**: `proof-fix`, `proof-verify`, `proof-extend`, `derivation`, `simplify`, `alternative`
3. **File detection**: If target ends in `.tex`, treat as file path; otherwise treat as theorem label
4. **Default**: If no action given, show available actions

## Examples

```
/theory verify thm:lower_bound    → Skill: verify-proof, args: "thm:lower_bound"
/theory verify all                → Sequential verify-proof for each registered theorem
/theory refine theory_lower_bound.tex → Skill: refine-theory, args: "theory_lower_bound.tex"
/theory gemini proof-fix lemma3   → Skill: gemini-prompt, args: "proof-fix lemma3"
/theory check                     → Parallel: check-paper-consistency + check-cross-references
/theory status                    → Direct: read results.md + dependencies.md
/theory refs                      → Skill: proofread-references

# Deep Theory Reference Commands (NEW)
/theory refs deep all A/F ratio   → Cross-domain theory analysis for A/F optimization
/theory refs queueing             → Deep queueing theory foundations
/theory refs queueing heavy-traffic → Heavy traffic theory specifically
/theory refs scheduling online    → Online scheduling competitive analysis
/theory refs scheduling approximation → Approximation algorithms (LST, etc.)
/theory refs energy speed-scaling → Speed scaling theory (YDS, BKP)
/theory refs energy dvfs          → DVFS and power management theory
```

## No-Match Behavior

If the action doesn't match any route:

```
Unknown action: "[action]"

Available actions:
  verify [target]          - Verify a specific proof (or "all")
  refine [file]            - Iterative theory refinement
  gemini [type] [target]   - Generate Gemini prompt for proof work
  check                    - Symbol consistency + cross-references (parallel)
  status                   - Theorem registry overview
  refs [file]              - Proofread references
  refs deep [domain]       - Cross-domain theory analysis (queueing+scheduling+energy)
  refs queueing [topic]    - Deep queueing theory foundations
  refs scheduling [topic]  - Deep scheduling theory foundations
  refs energy [topic]      - Deep energy optimization foundations
```

## Begin

Parse `$ARGUMENTS` and dispatch following the Dispatch Mode Guide above. For proof verification, use Task tool with `general-purpose` subagent. For interactive operations (refine, gemini), use Skill tool. Show which sub-command and dispatch mode you're using.
