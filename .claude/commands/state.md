# State - Paper State Lifecycle Orchestrator

You are a paper state orchestrator. Parse the user's action and dispatch to the correct state management sub-command.

**This is a thin routing layer.** Do NOT read RAG files or paper state directly (except for `status`). Sub-commands handle that.

## Arguments

`$ARGUMENTS` — The action to perform. Examples: `init my_paper MS`, `update`, `framing MS`, `overview`, `comments status`

## Protocol

1. Read `.claude/commands/_shared/orchestrator_protocol.md` for dispatch rules
2. Parse `$ARGUMENTS` into action + parameters
3. Dispatch using the **Dispatch Mode Selection** from orchestrator_protocol.md:
   - **Task tool** (`general-purpose` subagent) for heavy analysis (refs, update with full scan)
   - **Skill tool** for interactive/sequential operations (init, framing, comments)
   - **Direct** for status (just reads files)
4. Show progress and results
5. End with Next Steps footer

### Dispatch Mode Guide

| Action | Dispatch Via | Why |
|--------|-------------|-----|
| `init [name] [venue]` | **Skill** (sequential) | Interactive setup, needs conversation flow |
| `update [name]` | **Task** | Scans all .tex files, heavy analysis |
| `framing [venue]` | **Skill** | Interactive, user may adjust terms |
| `overview` | **Skill** | Lightweight document generation |
| `comments [sub]` | **Skill** | Interactive comment tracking |
| `refs [file]` | **Task** | Reads all references, heavy analysis |
| `status` | **Direct** | Just reads state docs |

## Routing Table

| User Says | Dispatches To | Skill Name | Args |
|-----------|--------------|------------|------|
| `init [name] [venue]` | `/init-paper-state` → `/define-paper-framing` (sequential) | (see below) | |
| `update [name]` | `/update-paper-state [name]` | `update-paper-state` | `[name]` |
| `framing [venue]` | `/define-paper-framing [venue]` | `define-paper-framing` | `[venue]` |
| `overview` | `/paper-overview` | `paper-overview` | `` |
| `comments [sub]` | `/track-review-comments [sub]` | `track-review-comments` | `[sub]` |
| `refs [file]` | `/proofread-references [file]` | `proofread-references` | `[file]` |
| `status` | (direct: read state docs) | — | — |

## `init` Implementation (Sequential Workflow)

The `init` action creates the full documentation ecosystem:

1. Parse name and venue from arguments
2. Invoke `Skill: init-paper-state, args: "[name]"` — create 12 state doc files
3. Invoke `Skill: define-paper-framing, args: "[venue]"` — lock terminology and framing
4. Show initialization summary:

```
═══════════════════════════════════════════════════════════════════
                    PAPER STATE INITIALIZED
═══════════════════════════════════════════════════════════════════

Paper: [name]
Venue: [venue]
State dir: docs/paper_state/[name]/

Created files (12):
  overview.md, symbols.md, results.md, framing.md,
  changelog.md, cross_references.md, dependencies.md,
  abbreviations.md, figures_tables.md, insights.md,
  review_responses.md, consistency_log.md

Framing defined: yes
  - Core concepts locked
  - Terminology mapped
  - Symbol registry seeded

Next: Start writing, then /state update to sync
```

## `status` Implementation (No Sub-Command)

The `status` action shows paper state health:

1. Resolve paper name via `ls docs/paper_state/`
2. Read `docs/paper_state/{resolved}/overview.md`
3. Read `docs/paper_state/{resolved}/changelog.md`
4. Read `docs/paper_state/{resolved}/review_responses.md`
5. Display:

```
═══════════════════════════════════════════════════════════════════
                    PAPER STATE STATUS
═══════════════════════════════════════════════════════════════════

Paper: [name]
State dir: docs/paper_state/[resolved]/

Documentation files:
  overview.md           [last modified date]
  symbols.md            [last modified date]
  results.md            [last modified date]
  framing.md            [last modified date]
  changelog.md          [last modified date]
  cross_references.md   [last modified date]
  ...

Last changelog entry: [date] [description]

Reviewer comments:
  Total: [N] | Resolved: [M] | Open: [N-M]

Staleness check:
  [List any state docs not updated recently]
```

## `comments` Subcommand Routing

The `comments` action supports all track-review-comments subcommands:

| User Says | Args Passed |
|-----------|-------------|
| `comments add [file]` | `add [file]` |
| `comments status` | `status` |
| `comments respond [id] [text]` | `respond [id] [text]` |
| `comments resolve [id]` | `resolve [id]` |
| `comments export` | `export` |

## Parsing Rules

1. **Init requires name**: `init` without a name → ask user for paper name
2. **Venue aliases**: `MS`/`Management Science` → MS; `OR`/`Operations Research` → OR; `ML`/`NeurIPS`/`ICML` → ML
3. **Update default**: `update` without name → resolve via `ls docs/paper_state/`
4. **Comments default**: `comments` alone → `comments status`
5. **Default**: If no action given, show available actions

## Examples

```
/state init my_paper MS        → Sequential: init-paper-state + define-paper-framing
/state update                  → Skill: update-paper-state (auto-resolve name)
/state framing MS              → Skill: define-paper-framing, args: "MS"
/state overview                → Skill: paper-overview
/state comments status         → Skill: track-review-comments, args: "status"
/state comments add review.pdf → Skill: track-review-comments, args: "add review.pdf"
/state refs                    → Skill: proofread-references
/state status                  → Direct: read overview + changelog + review_responses
```

## No-Match Behavior

If the action doesn't match any route:

```
Unknown action: "[action]"

Available actions:
  init [name] [venue]  - Create documentation ecosystem + define framing
  update [name]        - Sync state docs after paper changes
  framing [venue]      - Define/update paper framing and terminology
  overview             - Generate paper overview document
  comments [sub]       - Track reviewer comments (add/status/respond/resolve/export)
  refs [file]          - Proofread references and citations
  status               - Paper state health check
```

## Begin

Parse `$ARGUMENTS` and dispatch following the Dispatch Mode Guide above. For heavy operations (update, refs), use Task tool with `general-purpose` subagent. For interactive operations, use Skill tool. Show which sub-command and dispatch mode you're using.
