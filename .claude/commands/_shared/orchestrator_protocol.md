# Orchestrator Dispatch Protocol

**Shared protocol for all orchestrator commands (`/paper`, `/theory`, `/project`, `/state`, `/do`).**

Orchestrators are thin routing layers. They parse user intent, dispatch to the correct sub-command via the Skill tool, and display progress. They do NOT read RAG files or paper state directly — sub-commands handle that via `unified_protocol.md`.

---

## Paper Name Resolution

Reuse Step 0A from `unified_protocol.md`:

1. Run `ls docs/paper_state/` to find the actual paper directory
2. If only one directory exists, use it
3. If multiple exist, match to the user's argument or ask
4. Pass the resolved name to all sub-command invocations

---

## Dispatch Mechanism

### Dispatch Mode Selection

Choose the right tool based on the operation's weight:

| Scenario | Tool | Why |
|----------|------|-----|
| Simple routing (1 skill, lightweight) | **Skill tool** | Low overhead, context shared |
| Heavy analysis (reads >3 files, produces report) | **Task tool** (`general-purpose`) | Clean context, no pollution |
| Multiple independent checks | **Task tool** (parallel) | Each check gets own context, runs simultaneously |
| Sequential with dependencies | **Task tool** (sequential) | Clean context per step |
| Status / read-only | **Direct** (Read tool) | No dispatch needed |

**Rule of thumb**: If a skill will read multiple .tex files and produce an analysis report, use Task tool. If it's a simple one-step action, use Skill tool.

### Task Tool Dispatch Pattern

When dispatching via Task tool, the subagent needs enough context to execute independently:

```
Task(
  subagent_type: "general-purpose",
  description: "Check paper consistency",       // short description
  prompt: """
    You are executing the check-paper-consistency skill.

    Instructions: Read and follow .claude/commands/check-paper-consistency.md
    Protocol: Read and follow .claude/commands/_shared/unified_protocol.md
    Paper directory: paper/journal/
    Paper state: docs/paper_state/ms_journal/

    Return a structured report of all findings.
  """
)
```

**Key elements in every Task prompt**:
1. Which skill .md file to read for instructions
2. Reference to unified_protocol.md (mandatory read protocol)
3. Resolved paper directory and paper state directory
4. What output format to return

### Parallel Task Dispatch

For multiple independent checks, launch them in a **single message** with multiple Task tool calls:

```
// All three run simultaneously in parallel:
Task("Check symbols", "general-purpose", "Execute check-paper-consistency for paper/journal/...")
Task("Check terms", "general-purpose", "Execute check-term-consistency for paper/journal/...")
Task("Check refs", "general-purpose", "Execute check-cross-references for paper/journal/...")
```

Each subagent:
- Reads its own skill .md file
- Follows unified_protocol.md independently
- Returns a focused report
- Has its own clean context (no cross-contamination)

### Skill Tool Dispatch (Lightweight)

Use Skill tool only for simple routing where context sharing is beneficial:

```
Skill: fix-issues, args: "symbols"     // needs current conversation context
Skill: polish-paper                     // interactive, user may intervene
```

### When to Prefer Task over Skill

| Indicator | Use Task |
|-----------|----------|
| Skill reads >3 .tex files | Yes — prevents context bloat |
| Multiple independent skills at one level | Yes — parallel execution |
| Skill produces a long report | Yes — keeps orchestrator context clean |
| Skill needs user interaction mid-execution | No — use Skill |
| Skill is a simple fix/edit | No — use Skill |

---

## Progress Display Format

Use box-drawing characters for visual progress:

```
┌─────────────────────────────────────────────────────────────────┐
│ ACTION 1/N: [Description]                                       │
├─────────────────────────────────────────────────────────────────┤
│ Dispatching: /sub-command [args]                                │
│ Status: Running...                                              │
└─────────────────────────────────────────────────────────────────┘
```

After completion:
```
┌─────────────────────────────────────────────────────────────────┐
│ ACTION 1/N: [Description]                                  Done │
├─────────────────────────────────────────────────────────────────┤
│ Result: [summary of sub-command output]                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

If a sub-command doesn't exist in the current project:
```
Not available: /sub-command is not installed in this project.
Skipping and continuing with next action.
```

If a sub-command fails:
```
Failed: /sub-command encountered an error.
Error: [brief description]
Continuing with next action.
```

---

## Next Steps Footer

Every orchestrator output MUST end with:

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

Completed: [N/M] actions
Issues found: [count]

Recommended:
  1. [Most urgent follow-up]
  2. [Second priority]

Quick commands:
  /paper [action]    - Paper review & checks
  /theory [action]   - Proof verification
  /project [action]  - Session & project ops
  /state [action]    - Paper state management
  /do [anything]     - When in doubt
```

---

## RAG Integration

Orchestrators do **NOT** read RAG files. Each sub-command (whether invoked via Skill or Task) follows `unified_protocol.md` which includes mandatory RAG reading. When dispatching via Task tool, the subagent reads the skill .md file which specifies which RAG files to load. This avoids redundant token usage in the orchestrator's context.

---

## Backward Compatibility

All 28+ existing commands remain directly callable. Orchestrators are pure additions:
- `/paper review MS` dispatches to `/paper-pipeline review MS`
- `/paper-pipeline review MS` still works directly
- Users can mix orchestrator and direct commands freely
