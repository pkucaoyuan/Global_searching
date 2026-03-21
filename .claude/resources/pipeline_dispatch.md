# Pipeline Dispatch Plans & Task Prompt Template

## Task Prompt Template

Every Task dispatch MUST include these elements, including **INJECTED STATE**:

**Mandatory: State Doc Injection**

Before dispatching, the orchestrator MUST:
1. Read `.claude/commands/_shared/state_doc_injection.md`
2. Extract locked terms from `framing.md` "Preferred Phrasing" table
3. Extract core symbols from `symbols.md`
4. Extract recent changes from `changelog.md`
5. Include these in the Task prompt's `INJECTED STATE` block

```
Task(
  subagent_type: "general-purpose",
  description: "[short description]",
  prompt: """
    You are executing the [skill-name] skill.

    **Instructions**: Read .claude/commands/[skill-name].md and follow it completely.
    **Protocol**: Read .claude/commands/_shared/unified_protocol.md (mandatory read protocol).
    **Paper directory**: [PAPER_DIR]
    **Paper state directory**: [PAPER_STATE_DIR]

    ═══════════════════════════════════════════════════════════════════
    INJECTED STATE (authoritative - do not re-read these docs)
    ═══════════════════════════════════════════════════════════════════

    ## Locked Terminology (from framing.md)
    | Use This | Not This |
    |----------|----------|
    | proxy score | judge score, LLM score |
    | samples | data (except compounds) |
    | human audit | human review |
    | verified | ground-truth (except compounds) |
    | instance | context (except established terms) |

    ## Core Symbols (from symbols.md)
    - F: proxy score ∈ [0,1]
    - Y: human label ∈ [0,1]
    - π: audit probability
    - θ_k: true mean for arm k

    ## Recent Changes (from changelog.md)
    [Last 3-5 changelog entries]

    ═══════════════════════════════════════════════════════════════════

    Execute the skill. Flag ANY term/symbol that violates the injected state.
    Return a structured report with:
    - Issues found (location, severity, description)
    - Suggested fixes
    - Summary count
  """
)
```

---

## Review Pipeline (`review`) — Dispatch Plan

**Level 0 — Content**: Sequential (depends on each other)
```
Task: "Execute check-content-redundancy for [PAPER_DIR]"  → wait for result
Task: "Execute check-paper-flow for [PAPER_DIR]"          → wait for result
```

**Level 1 — Structure**: Parallel (independent)
```
Task: "Execute check-content-redundancy for [PAPER_DIR]"  ┐
Task: "Execute check-content-placement for [PAPER_DIR]"   │ parallel
Task: "Execute check-figures-tables for [PAPER_DIR]"      ┘
```
Note: If `check-figures-tables` finds rendering issues, suggest `/fix-figures` to resolve them.

**Level 2 — Consistency**: Parallel (independent)
```
Task: "Execute check-paper-consistency for [PAPER_DIR]"   ┐
Task: "Execute check-term-consistency for [PAPER_DIR]"    │ parallel
Task: "Execute check-cross-references for [PAPER_DIR]"    ┘
```

**Level 3 — Venue Style**: Single Task
```
Task: "Execute check-ms-style for [PAPER_DIR]"   (or check-msom-style / check-or-style / check-ml-style)
```

**Level 4 — Language**: Single Task (only if L0-L3 pass)
```
Task: "Execute polish-paper for [PAPER_DIR]"
```

**After all levels**: Aggregate results from all Task outputs, generate summary.

---

## Quick Pipeline (`quick`) — Dispatch Plan

All 5 content checks in parallel, then 1 sequential freshness check:

```
// Parallel batch (single message with 5 Task calls):
Task: "Execute check-paper-consistency for [PAPER_DIR]. Focus on symbols only. Return concise report."
Task: "Execute check-term-consistency for [PAPER_DIR]. Return concise report."
Task: "Execute check-cross-references for [PAPER_DIR]. Return concise report."
Task: "Check numerical consistency in [PAPER_DIR]. Compare numbers across abstract, intro, experiments, conclusion."
Task: "Check formula consistency in [PAPER_DIR]. Compare mathematical forms of results that appear in both main text (sections/) and appendix (appendix/). Flag any additive-vs-multiplicative, variance-vs-stddev, or notation divergences."
Task: "Check figure rendering in [PAPER_DIR]. Run: python .claude/scripts/figure_check/check_figure_rendering.py [PAPER_DIR] --no-compile --output text. If the tool is unavailable, fall back to pdflatex log parsing for overfull/underfull hbox warnings in figure environments and missing icon/image files. Return concise report. If issues found, recommend /fix-figures."

// After parallel batch completes:
// State Doc Freshness Check (run directly, no subagent needed — simple grep + compare)
```

---

## Pre-Submit Pipeline (`pre-submit`) — Dispatch Plan

Run all checks in parallel, then aggregate into checklist:

```
// Parallel batch:
Task: "Execute check-paper-consistency for [PAPER_DIR]"
Task: "Execute check-term-consistency for [PAPER_DIR]"
Task: "Execute check-cross-references for [PAPER_DIR]"
Task: "Execute check-ms-style for [PAPER_DIR]"           (or check-msom-style / venue-specific)
Task: "Execute check-figures-tables for [PAPER_DIR]"

// After all complete: Aggregate into submission checklist format
// If check-figures-tables finds rendering issues, run /fix-figures to resolve them before resubmitting
```

---

## Restructure Writing Patterns

When rewriting a restructured section, apply these patterns from top-venue papers (also read `paragraphs/section_restructure.md`):

| Writing Task | Pattern to Use | Example |
|-------------|---------------|---------|
| **Flat section opening** | Link backward to previous section, preview results, state punchline | "Having established X, we now analyze Y. We first derive... then establish... Comparing the two shows..." |
| **Upper→Lower bound bridge** | "Natural question" framing | "A natural question is whether this rate can be improved." |
| **Rate optimality paragraph** | State both bounds, compare rates, locate gap | "Together, Thm A and Thm B show that [alg] is rate-optimal: both scale as log(1/δ)/Δ²" |
| **Example bridge** | "Exposes a trade-off... makes it concrete" | "This decomposition exposes a fundamental trade-off... The following example makes this concrete." |
| **Demoting to appendix** | Remark acknowledging result + practical limitation | "Appendix X describes [variant] that achieves [optimal]. However, [practical issue], making [original] preferred." |
| **Promoting from appendix** | Clean statement + proof pointer | "The following result, proved in Appendix X, [bounds/characterizes]..." |
| **Moving to Discussion** | Reframe as operational insight | "Beyond the rate analysis, the cost structure reveals [insight]. Proposition X formalizes this." |
| **Truncated distribution** | Justify bounded support + note correction factor | "To satisfy Assumption X, we consider TN(μ,σ²,[0,1]). When well-centered, κ≈1, recovering the standard case." |
