# Project - Session & Project Operations Orchestrator

You are a project operations orchestrator. Parse the user's action and dispatch to the correct session/project sub-command.

**This is a thin routing layer.** Do NOT read RAG files or paper state directly (except for `status`). Sub-commands handle that.

## Arguments

`$ARGUMENTS` — The action to perform. Examples: `start ms_journal`, `organize`, `progress update`, `rag stats`, `end`

## Protocol

1. Read `.claude/commands/_shared/orchestrator_protocol.md` for dispatch rules
2. Parse `$ARGUMENTS` into action + parameters
3. Dispatch using the **Dispatch Mode Selection** from orchestrator_protocol.md:
   - **Task tool** (`general-purpose` subagent) for heavy operations (research, rag review, organize)
   - **Skill tool** for interactive/lightweight operations (start, progress, icons)
   - **Sequential Tasks** for `end` workflow
4. Show progress and results
5. End with Next Steps footer

### Dispatch Mode Guide

| Action | Dispatch Via | Why |
|--------|-------------|-----|
| `start [paper]` | **Skill** | Interactive, needs conversation context |
| `organize [sub]` | **Task** | Heavy codebase analysis |
| `progress [sub]` | **Skill** | Interactive, needs user confirmation |
| `rag [sub]` | **Task** (for review/import), **Skill** (for stats/search) | Review is heavy analysis |
| `research [topic]` | **Task** | Web search + synthesis, independent |
| `icons [query]` | **Skill** | Lightweight lookup |
| `end [paper]` | **Skill** (sequential) | Needs conversation context for state sync |
| `status` | **Direct** | Just reads state docs |
| `broadcast [sub]` | **Skill** | Runs deploy script, lightweight |

## Routing Table

| User Says | Dispatches To | Skill Name | Args |
|-----------|--------------|------------|------|
| `start [paper]` | `/session-start [paper]` | `session-start` | `[paper]` |
| `organize [sub]` | `/organize [sub]` | `organize` | `[sub]` |
| `progress [sub]` | `/update-progress [sub]` | `update-progress` | `[sub]` |
| `rag [sub]` | `/rag-maintain [sub]` | `rag-maintain` | `[sub]` |
| `status` | (direct: aggregate status) | — | — |
| `end [paper]` | Sequential: update-paper-state → update-progress | (see below) | |
| `icons [query]` | `/find-icon [query]` | `find-icon` | `[query]` |
| `research [topic]` | `/deepresearch [topic]` | `deepresearch` | `[topic]` |
| `broadcast [sub]` | `/broadcast [sub]` | `broadcast` | `[sub]` |

## `end` Implementation (Sequential Workflow)

The `end` action provides a clean session close:

1. Resolve paper name via `ls docs/paper_state/`
2. Invoke `Skill: update-paper-state, args: "[resolved_name]"` — sync state docs
3. Invoke `Skill: update-progress, args: "update"` — update progress docs
4. Show session summary:

```
═══════════════════════════════════════════════════════════════════
                    SESSION END: [paper_name]
═══════════════════════════════════════════════════════════════════

State docs synced: yes
Progress updated: yes
Last changes recorded in: changelog.md

Next session: /project start [paper_name]
```

## `status` Implementation (No Sub-Command)

The `status` action aggregates project health:

1. Resolve paper name via `ls docs/paper_state/`
2. Read `docs/paper_state/{resolved}/overview.md` — paper status
3. Read `docs/paper_state/{resolved}/changelog.md` — recent changes
4. Display aggregate:

```
═══════════════════════════════════════════════════════════════════
                    PROJECT STATUS
═══════════════════════════════════════════════════════════════════

Paper: [name]
Venue: [venue]
Last modified: [date from changelog]

Recent changes:
  - [from changelog.md, last 3 entries]

Session context: [loaded/not loaded]

Available workflows:
  /project start    - Load session context
  /project end      - Sync & close session
  /project organize - Repository organization
  /project rag stats - RAG library health
```

## Parsing Rules

1. **Start alias**: `begin`, `init session`, `load` → `start`
2. **End alias**: `close`, `finish`, `save` → `end`
3. **RAG subcommands**: `rag add`, `rag search`, `rag review`, `rag stats`, `rag import`
4. **Organize subcommands**: `organize structure`, `organize docs`, `organize clean`
5. **Progress subcommands**: `progress update`, `progress status`, `progress view`
6. **Broadcast alias**: `deploy`, `sync` → `broadcast`
7. **Default**: If no action given, show available actions

## Examples

```
/project start ms_journal     → Skill: session-start, args: "ms_journal"
/project end                  → Sequential: update-paper-state → update-progress
/project organize             → Skill: organize
/project progress update      → Skill: update-progress, args: "update"
/project rag stats            → Skill: rag-maintain, args: "stats"
/project rag review           → Skill: rag-maintain, args: "review"
/project status               → Direct: read overview.md + changelog.md
/project icons queue          → Skill: find-icon, args: "queue"
/project research bandits     → Skill: deepresearch, args: "bandits"
/project broadcast            → Skill: broadcast, args: "all"
/project broadcast shared     → Skill: broadcast, args: "shared"
/project broadcast status     → Skill: broadcast, args: "status"
```

## No-Match Behavior

If the action doesn't match any route:

```
Unknown action: "[action]"

Available actions:
  start [paper]      - Load session context
  end [paper]        - Sync state & close session
  organize [sub]     - Repository structure
  progress [sub]     - Progress documentation
  rag [sub]          - RAG library maintenance (add/search/review/stats/import)
  status             - Aggregate project health
  icons [query]      - Search icon library
  research [topic]   - Deep research
  broadcast [sub]    - Deploy .claude/ to all projects
```

## Begin

Parse `$ARGUMENTS` and dispatch following the Dispatch Mode Guide above. For heavy operations (research, organize, rag review), use Task tool with `general-purpose` subagent. For interactive operations, use Skill tool. Show which sub-command and dispatch mode you're using.
