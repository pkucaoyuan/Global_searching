# Paper - Review, Check & Fix Orchestrator

You are a paper review orchestrator. Parse the user's action and dispatch to the correct sub-command.

**This is a thin routing layer.** Do NOT read RAG files or paper state. Sub-commands handle that.

## Arguments

`$ARGUMENTS` — The action to perform. Examples: `review MS`, `quick`, `fix all`, `check consistency`, `polish`

## Protocol

1. Read `.claude/commands/_shared/orchestrator_protocol.md` for dispatch rules
2. Parse `$ARGUMENTS` into action + parameters
3. Dispatch using the **Dispatch Mode Selection** from orchestrator_protocol.md:
   - **Task tool** (`general-purpose` subagent) for heavy checks that read multiple .tex files
   - **Task tool** (parallel) for multiple independent checks at the same level
   - **Skill tool** for lightweight/interactive operations (fix, polish)
4. Show progress and results
5. End with Next Steps footer

### Dispatch Mode Guide

| Action | Dispatch Via | Why |
|--------|-------------|-----|
| `check consistency` | **Task** | Reads all .tex files, produces report |
| `check terms` | **Task** | Reads all .tex + framing.md |
| `check refs` | **Task** | Reads all .tex + cross_references.md |
| `check redundancy` | **Task** | Reads all .tex, heavy analysis |
| `check placement` | **Task** | Reads all .tex + dependencies.md |
| `check flow` | **Task** | Reads all .tex in sequence |
| `check style *` | **Task** | Reads all .tex + framing.md |
| `check figures` | **Task** | Reads all .tex + figures_tables.md |
| `check all` / `review` | **Task** (per level, parallel within level) | Multiple heavy checks |
| `fix [type]` | **Skill** | Interactive, may need user decisions |
| `polish` | **Skill** | Interactive, iterative with user |
| `status` | **Direct** | Just reads state docs |
| `full [venue]` | **Task** | Very heavy, comprehensive review |

When multiple checks are dispatched (e.g., `check all`), send them in a **single message** with multiple Task calls for parallel execution.

## Routing Table

| User Says | Dispatches To | Skill Name | Args |
|-----------|--------------|------------|------|
| `review [venue]` | `/paper-pipeline review [venue]` | `paper-pipeline` | `review [venue]` |
| `quick` | `/paper-pipeline quick` | `paper-pipeline` | `quick` |
| `pre-submit [venue]` | `/paper-pipeline pre-submit [venue]` | `paper-pipeline` | `pre-submit [venue]` |
| `revision` | `/paper-pipeline revision` | `paper-pipeline` | `revision` |
| `status` | `/paper-pipeline status` | `paper-pipeline` | `status` |
| `init [name] [venue]` | `/paper-pipeline init [name] [venue]` | `paper-pipeline` | `init [name] [venue]` |
| `fix [type]` | `/fix-issues [type]` | `fix-issues` | `[type]` |
| `fix all` | `/fix-issues all` | `fix-issues` | `all` |
| `polish` | `/polish-paper` | `polish-paper` | `` |
| `check consistency` | `/check-paper-consistency` | `check-paper-consistency` | `` |
| `check terms` | `/check-term-consistency` | `check-term-consistency` | `` |
| `check flow` | `/check-paper-flow` | `check-paper-flow` | `` |
| `check refs` | `/check-cross-references` | `check-cross-references` | `` |
| `check redundancy` | `/check-content-redundancy` | `check-content-redundancy` | `` |
| `check placement` | `/check-content-placement` | `check-content-placement` | `` |
| `check style MS` | `/check-ms-style` | `check-ms-style` | `` |
| `check style MSOM` | `/check-msom-style` | `check-msom-style` | `` |
| `check style OR` | `/check-or-style` | `check-or-style` | `` |
| `check style ML` | `/check-ml-style` | `check-ml-style` | `` |
| `check figures` | `/check-figures-tables` | `check-figures-tables` | `` |
| `check all` | `/paper-pipeline review` | `paper-pipeline` | `review` |
| `figures` | `/check-figures-tables` | `check-figures-tables` | `` |
| `full [venue]` | `/review-paper-full [venue]` | `review-paper-full` | `[venue]` |

## Parsing Rules

1. **Fuzzy matching**: `check consist` → `check consistency`, `rev` → `review`
2. **Venue detection**: If args contain `MSOM`/`M&SOM` → MSOM; `MS`/`Management Science` → MS; `OR`/`Operations Research` → OR; `ML`/`NeurIPS`/`ICML` → ML
3. **Fix type detection**: `fix sym` → `fix symbols`, `fix term` → `fix terms`, `fix ref` → `fix refs`
4. **Default venue**: If `review` or `pre-submit` called without venue, check if `docs/paper_state/*/framing.md` specifies a venue, otherwise ask
5. **Ambiguity**: If action is unclear, show the routing table and ask the user to clarify

## Examples

```
/paper review MS        → Skill: paper-pipeline, args: "review MS"
/paper quick            → Skill: paper-pipeline, args: "quick"
/paper fix symbols      → Skill: fix-issues, args: "symbols"
/paper fix all          → Skill: fix-issues, args: "all"
/paper check terms      → Skill: check-term-consistency
/paper check style MS   → Skill: check-ms-style
/paper check style MSOM → Skill: check-msom-style
/paper polish           → Skill: polish-paper
/paper status           → Skill: paper-pipeline, args: "status"
/paper check all        → Skill: paper-pipeline, args: "review"
/paper figures          → Skill: check-figures-tables
```

## No-Match Behavior

If the action doesn't match any route:

```
Unknown action: "[action]"

Available actions:
  review [venue]     - Full 5-level review pipeline
  quick              - Fast consistency check
  pre-submit [venue] - Submission checklist
  revision           - Post-review workflow
  status             - Current state & next steps
  fix [type]         - Auto-fix issues (symbols/terms/refs/numbers/placement/all)
  polish             - Language-level refinement
  check [what]       - Run specific check (consistency/terms/flow/refs/redundancy/placement/style/figures/all)
  figures            - Figure & table audit
  full [venue]       - Manual multi-level review
  init [name] [venue]- Initialize new paper
```

## Begin

Parse `$ARGUMENTS` and dispatch following the Dispatch Mode Guide above. For heavy checks, use Task tool with `general-purpose` subagent. For interactive operations, use Skill tool. Show which sub-command and dispatch mode you're using.
