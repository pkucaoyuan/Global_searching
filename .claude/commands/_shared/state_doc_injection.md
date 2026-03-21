# State Doc Injection Protocol

**Purpose**: Ensure subagents receive critical paper state DIRECTLY in their prompts, not just file paths.

---

## Why Injection Is Required

**Problem**: Current pattern tells subagents "read framing.md" but:
1. Subagents may not actually read the file
2. Subagents may read stale cached versions
3. Reading adds latency and token overhead per subagent
4. No enforcement that subagent understood the content

**Solution**: Orchestrators INJECT critical state doc contents directly into Task prompts.

---

## Injection Categories

### Category A: Always Inject (Every Subagent)

These contents MUST be included in EVERY Task subagent prompt:

| State Doc | Content to Extract | Format |
|-----------|-------------------|--------|
| `framing.md` | "Preferred Phrasing" table (lines 149+) | Markdown table |
| `symbols.md` | Core symbol definitions (first 30 lines) | Markdown list |
| `changelog.md` | Last 5 entries | Markdown list |

### Category B: Skill-Specific Injection

| Skill Type | Additional State Docs |
|------------|----------------------|
| check-*-consistency | Full `framing.md` "Key Terminology" section |
| check-cross-references | Full `cross_references.md` |
| check-figures-tables | Full `figures_tables.md` |
| verify-proof | `results.md`, `dependencies.md` |

---

## Injection Format

When dispatching a Task, include an `INJECTED_STATE` block:

```
Task(
  subagent_type: "general-purpose",
  description: "Check term consistency",
  prompt: """
    You are executing the check-term-consistency skill.

    Instructions: Read .claude/commands/check-term-consistency.md
    Protocol: Read .claude/commands/_shared/unified_protocol.md
    Paper directory: paper/journal/
    Paper state: docs/paper_state/ms_journal/

    ═══════════════════════════════════════════════════════════════════
    INJECTED STATE (from paper state docs - use this, don't re-read)
    ═══════════════════════════════════════════════════════════════════

    ## Locked Terminology (from framing.md)
    | Use This | Not This |
    |----------|----------|
    | proxy score | judge score, LLM score |
    | samples | data (except compounds) |
    | human audit | human review |
    | ...

    ## Core Symbols (from symbols.md)
    - F: proxy score ∈ [0,1]
    - Y: human label ∈ [0,1]
    - π: audit probability
    - ...

    ## Recent Changes (from changelog.md)
    - 2026-02-08: Fixed "judge score" → "proxy score" (Comment #2)
    - ...

    ═══════════════════════════════════════════════════════════════════

    Execute the skill. Flag ANY term that violates the Locked Terminology table.
    Return structured report.
  """
)
```

---

## Extraction Commands

Before dispatching subagents, the orchestrator MUST run these to extract injectable state:

```bash
# Extract Preferred Phrasing table from framing.md
grep -A 20 "### Preferred Phrasing" docs/paper_state/{paper}/framing.md

# Extract core symbols
head -40 docs/paper_state/{paper}/symbols.md

# Extract recent changelog
tail -20 docs/paper_state/{paper}/changelog.md
```

---

## Enforcement Checkpoint

The orchestrator MUST verify state doc extraction succeeded before dispatching:

```
Injected state verification:
- framing.md "Preferred Phrasing": [N] locked terms extracted
- symbols.md: [N] symbols extracted
- changelog.md: [N] recent entries extracted

If any extraction returns empty, STOP and report staleness.
```

---

## Anti-Patterns

### Wrong (current pattern):
```
Task("Check terms", "general-purpose", """
  Read check-term-consistency.md
  Paper state: docs/paper_state/ms_journal/
  Return report.
""")
```
→ Subagent may not read state docs, or may read them incorrectly.

### Correct (with injection):
```
Task("Check terms", "general-purpose", """
  Read check-term-consistency.md

  INJECTED STATE:
  Locked terms: proxy score (not judge score), samples (not data)...

  Flag ANY violation of injected locked terms.
  Return report.
""")
```
→ State is in the prompt; subagent cannot ignore it.

---

## Integration with Other Protocols

- `unified_protocol.md`: Subagents still follow Steps 0-4, but skip re-reading files that were injected
- `orchestrator_protocol.md`: Add injection step before Task dispatch
- `check-term-consistency.md`: Check against INJECTED locked terms, not re-reading framing.md
