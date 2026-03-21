# Unified Protocol for All Paper Review Commands

**EVERY paper review command MUST follow this protocol.**

---

## ⚠️ UNIVERSAL POST-EDIT RULE (Applies to ALL .tex modifications)

**This rule applies ALWAYS — whether you are running a formal command (`/fix-issues`, `/polish-paper`) or doing ad-hoc edits in response to user questions.**

### Rule: After editing ANY .tex file, you MUST immediately update state docs

**Trigger**: Any `Edit` or `Write` to a `.tex` file under `paper/`

**Required updates** (check each, skip if not applicable):

| What Changed | State Doc to Update | What to Add |
|-------------|--------------------|----|
| Added/removed `\label{fig:*}` or `\label{tab:*}` | `figures_tables.md` | Add/remove entry in registry |
| Added/removed `\label{*}` (any type) | `cross_references.md` | Add/remove from label list |
| Changed any `\begin{theorem}` / result statement | `results.md` | Update theorem registry |
| Added/changed symbols | `symbols.md` | Update symbol registry |
| Changed terminology | `framing.md` | Update term registry |
| ANY modification at all | `changelog.md` | Add dated entry describing changes |

**Minimum update**: `changelog.md` is ALWAYS updated after any .tex edit. The others are conditional.

**Why this exists**: Without this rule, ad-hoc edits (responding to user questions like "是否写清楚了") create state doc staleness that isn't caught until someone explicitly audits. This rule prevents that drift.

**Anti-pattern (what went wrong before)**:
```
User: "实验setting写清楚吗？"
→ Agent edits experiments.tex (adds δ, clarification sentence)
→ Agent edits service_system_details.tex (adds tab:queue-config)
→ Agent does NOT update figures_tables.md, cross_references.md, changelog.md
→ State docs are now stale
→ User must manually ask "paper states有正确使用吗" to discover this
```

**Correct pattern**:
```
User: "实验setting写清楚吗？"
→ Agent edits experiments.tex (adds δ, clarification sentence)
→ Agent edits service_system_details.tex (adds tab:queue-config)
→ Agent IMMEDIATELY updates:
   - figures_tables.md: add tab:queue-config entry
   - cross_references.md: add tab:queue-config to table labels
   - changelog.md: add entry for today's changes
→ State docs remain synchronized
```

---

## Two Mandatory Data Sources

### 1. RAG: Writing Reference Library
**Location**: `.claude/writing_references/`
**Purpose**: Ground all suggestions in human-authored patterns from top venues

```
.claude/writing_references/
├── sentences/           # Sentence-level patterns
│   ├── introduction.md  # "We study...", "This paper..."
│   ├── contribution.md  # "Our main contribution is..."
│   ├── motivation.md    # "Despite progress...", "However..."
│   ├── or_applications.md # OR/OM specific
│   └── ...
├── phrases/             # Phrase-level patterns
│   └── transitions.md   # "Furthermore...", "In contrast..."
└── paragraphs/          # Paragraph-level patterns
    ├── main_results.md  # Result announcement patterns
    └── proof_structure.md # Proof organization
```

**When to use RAG**:
- Suggesting rewrites (polish-paper, fix-issues)
- Checking style compliance (check-ms-style, check-or-style)
- Identifying "AI smell" patterns (check-term-consistency)

### 2. Paper State: Documentation Library
**Location**: `docs/paper_state/[paper_name]/`
**Purpose**: Track paper-specific decisions to ensure consistency

```
docs/paper_state/[paper_name]/
├── overview.md          # Current status
├── symbols.md           # Symbol definitions (b_k = bias, NOT best arm)
├── results.md           # All theorems (avoid redundancy)
├── framing.md           # Locked terminology (use "audit", not "review")
├── changelog.md         # Modification history
├── cross_references.md  # All refs to results/numbers
├── dependencies.md      # Assumption → Theorem chains
├── abbreviations.md     # Acronym registry
├── figures_tables.md    # Figure/table registry, generation standards, quality audit
├── insights.md          # Key takeaways (managerial for MS)
├── review_responses.md  # Reviewer comments tracking
└── consistency_log.md   # Check command history
```

**Templates**: All 12 templates available in `.claude/commands/templates/paper_state/`

**When to use paper_state**:
- Checking consistency (check-paper-consistency → symbols.md)
- Checking terminology (check-term-consistency → framing.md)
- Fixing issues (fix-issues → read relevant .md before fixing)
- Any command that modifies the paper

---

## Command-Specific Requirements

> **Note**: In all tables below, state file paths use the resolved `PAPER_STATE_DIR` from Step 0A.
> Example: `symbols.md` means `docs/paper_state/{resolved_name}/symbols.md`.

### Check Commands

| Command | RAG Files to Read | State Files to Read |
|---------|-------------------|---------------------|
| check-paper-consistency | sentences/problem_setup.md | **symbols.md**, results.md |
| check-term-consistency | phrases/transitions.md | **framing.md**, abbreviations.md |
| check-cross-references | - | **cross_references.md**, results.md |
| check-content-placement | paragraphs/proof_structure.md | **dependencies.md**, results.md |
| check-content-redundancy | sentences/contribution.md | **results.md** |
| check-paper-flow | phrases/transitions.md | **overview.md**, framing.md |
| check-ms-style | sentences/or_applications.md, sentences/contribution.md | **framing.md** |
| check-or-style | sentences/algorithm_optimality.md | **framing.md** |
| check-ml-style | sentences/contribution.md | **framing.md** |
| check-figures-tables | - | **figures_tables.md**, cross_references.md, framing.md |
| polish-paper | ALL sentence files, phrases/transitions.md | **overview.md**, framing.md |

### Fix Commands

| Command | RAG Files to Read | State Files to Read | State Files to Update |
|---------|-------------------|---------------------|----------------------|
| fix-issues symbols | sentences/problem_setup.md | **symbols.md** | symbols.md |
| fix-issues terms | phrases/transitions.md | **framing.md** | framing.md |
| fix-issues refs | - | **cross_references.md** | cross_references.md |
| fix-issues numbers | - | **cross_references.md** | cross_references.md |
| fix-issues placement | paragraphs/proof_structure.md | **dependencies.md** | dependencies.md |

### Theory Commands

| Command | RAG Files to Read | State Files to Read | State Files to Update |
|---------|-------------------|---------------------|----------------------|
| verify-proof | paragraphs/proof_structure.md, sentences/algorithm_optimality.md | **symbols.md**, **results.md**, **dependencies.md**, cross_references.md | results.md (verification status), changelog.md |
| refine-theory | paragraphs/proof_structure.md, sentences/algorithm_optimality.md, sentences/problem_setup.md, phrases/hedging.md | **symbols.md**, **results.md**, **dependencies.md**, framing.md, cross_references.md, changelog.md | results.md, symbols.md, dependencies.md, changelog.md |
| proofread-references | - | **cross_references.md**, results.md, dependencies.md | cross_references.md, changelog.md |

---

## Mandatory Read Protocol

**At the START of every command execution:**

### Step 0A: Resolve Paper Name

**⚠️ STOP. Before ANY other action, resolve the `[paper]` placeholder.**

The `[paper]` placeholder in all commands refers to an actual directory under `docs/paper_state/`. You MUST resolve it:

```bash
ls docs/paper_state/
```

**Resolution rules:**
1. If `$ARGUMENTS` contains a path like `paper/journal/`, map it: `journal` → check if `docs/paper_state/ms_journal/` exists
2. If only one directory exists under `docs/paper_state/`, use that (e.g., `ms_journal`)
3. If multiple directories exist, pick the one matching the paper being reviewed
4. **NEVER leave `[paper]` unresolved.** If you cannot determine the paper name, ask the user.

**After resolution, set `PAPER_STATE_DIR` for all subsequent steps:**
```
PAPER_STATE_DIR = docs/paper_state/ms_journal   # (example)
```

### Step 0B: Read Shared Config
```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
```

### Step 1: Read Relevant RAG Files
```
# Example for check-ms-style:
Read .claude/writing_references/sentences/or_applications.md
Read .claude/writing_references/sentences/contribution.md
Read .claude/writing_references/paragraphs/main_results.md
```

### Step 2: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP. Before executing the command logic, you MUST read the paper state files listed for this command in the table above.** These files contain paper-specific decisions (symbol definitions, locked terminology, theorem registry) that prevent you from introducing conflicts.

```
# Example for check-ms-style (use resolved PAPER_STATE_DIR):
Read docs/paper_state/ms_journal/framing.md
Read docs/paper_state/ms_journal/overview.md
```

**If you skip this step, you WILL introduce inconsistencies** (e.g., renaming a symbol that is already defined for a different purpose, or using terminology that contradicts the locked framing).

### Step 2.5: Verification Checkpoint

**Before proceeding to Step 3, you MUST confirm what you learned from the state docs.** Write a brief summary:

```
State doc context loaded:
- symbols.md: [key symbols and their meanings]
- framing.md: [locked terms and their definitions]
- results.md: [theorem registry summary]
- (other files as applicable)
```

**This checkpoint is NOT optional.** It forces you to actually process the state doc contents rather than just reading and ignoring them.

### Step 3: Execute Command Logic

### Step 4: Update State Files (for ANY .tex modification)

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**This step implements the Universal Post-Edit Rule defined at the top of this document.** It applies to ALL .tex edits — whether through a formal command or ad-hoc editing.

**STOP. After applying any changes to .tex files, you MUST update the relevant state docs.** Every modification must be recorded so that future sessions maintain consistency.

```
# Example for fix-issues terms (use resolved PAPER_STATE_DIR):
Edit docs/paper_state/ms_journal/framing.md      # Record new/changed terms
Edit docs/paper_state/ms_journal/changelog.md     # Log what changed and why
```

**Update rules:**
1. **figures_tables.md**: If you added, renamed, or removed any `\label{fig:*}` or `\label{tab:*}`
2. **cross_references.md**: If you added, renamed, or removed any `\label{*}` (any type)
3. **symbols.md**: If you added, renamed, or removed any symbol
4. **framing.md**: If you changed any terminology or framing
5. **results.md**: If you modified any theorem statement or numbering
6. **changelog.md**: ALWAYS update for any modification (date + description)

**Common failure mode**: Editing appendix tables (e.g., adding `tab:queue-config`) without updating `figures_tables.md` and `cross_references.md`. The Universal Post-Edit Rule at the top of this document exists specifically to prevent this.

**If you skip this step, the state docs become stale** and future commands will make conflicting decisions.

---

## Why Both Are Required

### Without RAG:
```
Suggestion: "Change 'We present a novel approach' to 'This paper proposes a new method'"
→ Generic AI rewrite
→ Doesn't match venue style
→ Reviewer: "Sounds machine-generated"
```

### With RAG:
```
RAG Pattern (from JMLR): "We develop a framework for..."
Suggestion: "Change 'We present a novel approach' to 'We develop a framework for...'"
→ Matches top-venue style
→ Reviewer: "Well-written"
```

### Without State Docs:
```
fix-issues symbols: Rename b(t) to k(t)
→ But symbols.md says k is already used for arm index!
→ Creates new conflict
```

### With State Docs:
```
Read symbols.md: k = arm index, b_k = bias
fix-issues symbols: Rename b(t) to ĥat{k}(t) (avoids conflict)
→ Update symbols.md with new symbol
→ Consistent throughout
```

---

## Self-Dispatch for Compound Skills

**Compound skills** (with 3+ independent phases) SHOULD spawn Task subagents for parallel execution. Read `.claude/commands/_shared/self_dispatch_protocol.md` for the full protocol.

**Quick rule**: After completing Steps 0-2 (setup), check if this skill has a "Self-Dispatch Phases" table. If yes, follow the self-dispatch protocol. If no, execute inline.

---

## Verification Checklist (MANDATORY)

Before completing any command OR any ad-hoc .tex edit, verify ALL of the following:

- [ ] **Paper name resolved** — `[paper]` replaced with actual directory (e.g., `ms_journal`)
- [ ] **RAG files read** — Relevant writing references loaded (if command suggests rewrites)
- [ ] **State files read** — Relevant paper state docs loaded AND checkpoint summary written
- [ ] **State files updated** — If .tex files were modified, ALL relevant state docs updated per Universal Post-Edit Rule
- [ ] **Labels synced** — Any new `\label{fig:*}` or `\label{tab:*}` added to `figures_tables.md` AND `cross_references.md`
- [ ] **Changelog entry added** — If any paper modification was made
- [ ] **RAG source cited** — Each suggestion grounded in a specific reference pattern
- [ ] **No conflicts introduced** — Checked state docs before applying any symbol/term changes

**FAILURE MODE**: If any checkbox above is unchecked, the command execution is INCOMPLETE. Go back and complete the missing steps.

**CRITICAL**: The "Labels synced" checkbox is the most commonly missed. Every time you add a `\label{}` to a .tex file, you MUST add it to `figures_tables.md` (for fig/tab labels) and `cross_references.md` (for all label types).
