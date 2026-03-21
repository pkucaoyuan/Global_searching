# Check Term Consistency - Detect Inconsistent Terminology

You are a terminology consistency checker. Your task is to find where the same concept is described using different words, creating confusion and a "machine-generated" feel.

## ⚠️ MANDATORY: Unified Protocol

### Step 0: Read Shared Config & RAG References

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/phrases/transitions.md
Read .claude/writing_references/sentences/motivation.md
```

### Step 0b: Load Project-Specific Concept Families (if any)

If `.claude/commands/_local/concept_families.md` exists, read it. Merge any additional
concept families into the "Common Term Families to Check" table below. The _local file
uses the same table format (`| Concept | Variants to Scan |`).

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before executing any terminology check, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/framing.md        # Locked terminology
Read docs/paper_state/{resolved}/abbreviations.md  # Acronym registry
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- framing.md: [list locked terms, e.g., "audit" not "review", "arm" not "option"]
- abbreviations.md: [list registered acronyms, e.g., BAI, CS, IPW, LLM]
```

**If you skip this step, you WILL flag terms that are intentionally locked, wasting review effort.**

### Step 2: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

This ensures the RAG library self-maintains by logging gaps and suggesting additions.

---

## Why This Prevents Reviewer Comments

**Comment 13**: "机器生成味道较重... 注意前后用词一致性"

This happens when:
- Same concept called "audit" in one place, "review" in another
- Same action described as "observe" vs "acquire" vs "collect"
- Same entity called "proxy" vs "judge" vs "automated evaluator"

## Arguments

- `$ARGUMENTS` - Optional: path to paper directory or path to framing document

## Workflow

### Phase 1: Build Term Usage Map

For each key concept, find all terms used to describe it:

```bash
# Example: Find all terms for "getting human feedback"
grep -n "audit\|review\|label\|annotation\|feedback\|assess" sections/*.tex
```

Group by concept:

```markdown
## Concept: Human Feedback Acquisition

| Term | Count | Locations |
|------|-------|-----------|
| "audit" | 45 | intro:3, model:12, algorithm:15, experiments:10, conclusion:5 |
| "review" | 8 | intro:2, experiments:6 |
| "label" | 12 | model:5, experiments:7 |
| "annotation" | 2 | related:2 |

**Issue**: 4 different terms for same concept
**Recommendation**: Use "audit" consistently (most frequent, matches service framing)
```

### Phase 2: Check Against Framing Document

If `docs/paper_state/[paper]_framing.md` exists:

```
For each locked term in framing doc:
    Find all usages in paper
    Flag any usage of "avoid" synonyms
```

### Phase 3: Detect Semantic Drift

**Pattern**: Same concept described differently in different sections

```markdown
## Concept: Why We Audit

### Introduction (paragraph 5):
> "to correct for the proxy's systematic bias"

### Section 5 (paragraph 2):
> "to debias the selective labeling process"

### Section 7 (paragraph 1):
> "to obtain unbiased estimates of the residual"

**Issue**: Same reason, three phrasings
**Recommendation**: Pick one and use throughout:
> "to correct for the proxy's bias through propensity weighting"
```

### Phase 4: Check Verb Consistency

**Common inconsistencies:**

| Concept | Inconsistent Verbs | Pick One |
|---------|-------------------|----------|
| Getting proxy score | "observe F" / "receive F" / "obtain F" | "observe F" |
| Selecting arm | "pull arm" / "select arm" / "choose arm" | "select arm" (MS) or "pull arm" (ML) |
| Making audit decision | "decide to audit" / "trigger audit" / "request review" | "request audit" |

### Phase 5: Check Adjective Consistency

**Pattern**: Same property described with different adjectives

```markdown
## Property: Judge's unreliability

| Phrasing | Location |
|----------|----------|
| "biased proxy" | intro:L12 |
| "unreliable judge" | model:L34 |
| "imperfect evaluator" | experiments:L56 |

**Recommendation**: Use "biased" consistently (matches formal definition b_k(x))
```

### Phase 6: Detect "Furthermore" Chains

**Machine-generated smell**: Repetitive transitions

```bash
grep -n "Furthermore\|Moreover\|Additionally\|In addition" sections/*.tex
```

If 3+ consecutive paragraphs use the same transition:
→ Flag as machine-generated smell
→ Suggest varying transitions from `phrases/transitions.md`

### Phase 6: Cross-Field Term Collision ⭐

**This phase checks whether any term used in the paper has a conflicting, well-established meaning in an adjacent field that target readers are likely to know.** Internal consistency alone is insufficient — a term can be used consistently throughout the paper while still confusing readers who associate that term with a different concept from their own field.

**Why this matters:** Academic papers are read by cross-disciplinary audiences. A term that is perfectly clear to the authors may trigger a different mental model in readers from adjacent fields. Example: "context" in a bandit paper may be interpreted as the contextual covariate in contextual bandits, even if the authors intend it as "evaluation instance." This causes friction, misunderstanding, and reviewer objections.

**Step 1: Identify the paper's field neighborhood**

Based on the paper's topic and target venue, determine which adjacent fields readers are likely familiar with:

| Paper Topic | Adjacent Fields to Check |
|-------------|------------------------|
| Bandits / BAI | Contextual bandits, Reinforcement learning, Causal inference, Clinical trials |
| Service systems | Queueing theory, Simulation optimization, Revenue management |
| LLM evaluation | NLP, Information retrieval, Human-computer interaction |
| Optimization | Mathematical programming, Stochastic optimization, Machine learning |
| Causal inference | Statistics, Econometrics, Epidemiology |

**Step 2: Check key terms against adjacent-field meanings**

For each key term in the paper, ask: "Does this term have a different, well-known meaning in an adjacent field?"

**Known collision-prone terms (non-exhaustive registry):**

| Term in Paper | Adjacent Field | Conflicting Meaning | Risk Level |
|--------------|----------------|---------------------|------------|
| "context" | Contextual bandits | Covariate that informs arm selection (X in contextual MAB) | 🔴 High |
| "reward" | RL / Economics | Cumulative return / Monetary compensation | 🟡 Medium |
| "arm" | Manufacturing / Medicine | Physical component / Body part | 🟢 Low (standard in bandits) |
| "policy" | RL / Public policy | π(s) mapping states to actions / Government regulation | 🟡 Medium |
| "regret" | Decision theory / Psychology | Minimax regret / Emotional state | 🟢 Low |
| "exploration" | RL | Explore-exploit trade-off (different from pure exploration in BAI) | 🟡 Medium |
| "agent" | Multi-agent systems | Autonomous entity (vs. human decision-maker) | 🟡 Medium |
| "treatment" | Causal inference / Medicine | Intervention in RCT / Medical procedure | 🟡 Medium |
| "sample" | Statistics / Signal processing | Random variable realization / Discrete signal value | 🟢 Low |
| "type" | Mechanism design / Programming | Private information / Data type | 🟡 Medium |
| "state" | RL / Markov chains | Environment state s ∈ S | 🟡 Medium |
| "feature" | ML | Input variable to a model | 🟡 Medium |
| "label" | ML / Classification | Ground truth class assignment | 🟡 Medium |
| "model" | Statistics / ML | Statistical model / Neural network | 🟡 Medium |
| "instance" | Optimization / ML | Problem instance / Data point | 🟢 Low |
| "oracle" | Complexity theory | Computational oracle with exact answers | 🟡 Medium |

**Step 3: Flag and suggest alternatives**

For each collision detected:
1. Quote the term and its usage in the paper
2. Identify the adjacent field and conflicting meaning
3. Assess risk: How likely are target-venue readers to misinterpret?
4. Suggest alternatives if risk is medium or high

**Output format for this phase:**
```markdown
### Cross-Field Term Collisions

| Term | Paper's Meaning | Adjacent Field | Their Meaning | Risk | Suggestion |
|------|----------------|----------------|---------------|------|------------|
| "context" | Evaluation instance X_t | Contextual bandits | Covariate for arm selection | 🔴 | Rename to "instance" |
| "reward" | True quality score Y | RL | Cumulative discounted return | 🟡 | Keep (standard in bandits) |
```

**Note:** This check is complementary to Phase 1 (internal term consistency). Phase 1 ensures the paper doesn't use two words for one concept. Phase 6 ensures the paper doesn't use one word that means two things to different readers.

---

## Locked Term Enforcement (Project-Specific)

**Purpose**: Enforce terminology locked in `framing.md` and `_local/concept_families.md`. This is NOT a generic synonym check — it's enforcement of paper-specific decisions.

**⚠️ MANDATORY: Read _local/concept_families.md FIRST**

Before running any grep, read `.claude/commands/_local/concept_families.md` to get:
1. The canonical terms for this paper
2. The variants that are FORBIDDEN
3. Compound exceptions that are ALLOWED

**Procedure**:
1. Read `_local/concept_families.md` → extract "Locked Concept Families" table
2. Read `framing.md` → verify alignment with "Preferred Phrasing" table
3. For EACH locked concept family:
   a. Grep ALL .tex files for the FORBIDDEN variants (not the canonical)
   b. If ANY forbidden variant has count > 0 → **FLAG as violation**
   c. Report: exact locations, suggested replacement (canonical term)

**Example: Proxy/judge output**
```bash
# Canonical: "proxy score"
# Forbidden: "judge score", "LLM score"
grep -nwi "judge score" paper/journal/sections/*.tex   # Should be 0
grep -nwi "LLM score" paper/journal/sections/*.tex     # Should be 0
```

If `grep -c "judge score"` returns non-zero → VIOLATION.

---

## Generic Term Families (Fallback)

**Purpose**: For projects WITHOUT `_local/concept_families.md`, use these generic families as a fallback. If `_local/concept_families.md` exists, it OVERRIDES this table.

| Concept | Variants to Scan (grep patterns) |
|---------|----------------------------------|
| Human feedback | audit, review, label, annotation, ground truth |
| Proxy/judge output | judge score, proxy score, LLM score, automated score, prediction score |
| Observation unit | sample, data, observation, measurement, data point |
| Selection action | pull, select, choose, evaluate |
| Best alternative | best arm, optimal arm, winner, top configuration |
| Confidence/uncertainty | uncertainty, precision, reliability, confidence |
| Algorithm termination | terminates, stops, converges, completes |
| Verified outcome | ground truth, verified label, human label, gold standard, true label |
| Benchmark/baseline | baseline, benchmark, reference |

**Exclusion rules** (do not flag these as inconsistencies):
- Compound terms are distinct from standalone: "data-driven" ≠ "data", "missing data mechanism" ≠ "data"
- Entity names vs concept names: "LLM judge" (entity) ≠ "judge score" (concept)
- Verb vs noun usage: "The judge scores quality" (verb) ≠ "judge score" (noun compound)
- Section headings quoting other work or describing entities are excluded

## Output Format

```markdown
# Term Consistency Report

**Paper**: [title]
**Date**: [date]
**Framing Doc**: [path if exists]

---

## 1. Term Inconsistencies

### High Priority (Same concept, multiple terms)

| Concept | Terms Found | Recommended | Locations to Fix |
|---------|-------------|-------------|------------------|
| Human feedback | audit (45), review (8), label (12) | audit | review: intro:L23, L45; experiments:L67... |
| ... | ... | ... | ... |

### Medium Priority (Semantic drift)

| Concept | Phrasings | Recommended |
|---------|-----------|-------------|
| Why audit | 3 different | "to correct for proxy bias" |

---

## 2. Verb Inconsistencies

| Action | Verbs Found | Recommended |
|--------|-------------|-------------|
| Get proxy | observe (30), receive (5), obtain (3) | observe |

---

## 3. Machine-Generated Smells

### Repetitive Transitions
- intro.tex: "Furthermore" used 4 times in L20-35
- experiments.tex: "Moreover" used 3 times in L50-65

**Fix**: Vary transitions using `phrases/transitions.md`

---

## 4. Summary

| Category | Issues | Severity |
|----------|--------|----------|
| Term inconsistency | X | 🔴 High |
| Semantic drift | Y | 🟡 Medium |
| Verb inconsistency | Z | 🟡 Medium |
| Repetitive transitions | W | 🟢 Minor (but flags AI) |

---

## 5. Recommended Terminology (Lock These)

**Source**: Pull canonical terms from `framing.md` "Preferred Phrasing" table. Do NOT invent recommendations — only echo what the framing doc locks.

| Concept | Canonical (from framing.md) | Avoid |
|---------|----------------------------|-------|
| Human feedback | [read from framing.md] | [all other variants found] |
| Proxy/judge output | [read from framing.md] | [all other variants found] |
| Observation unit | [read from framing.md] | [all other variants found] |
| ... | ... | ... |
```

## Integration

```
/define-paper-framing     → Lock terms BEFORE writing
        ↓
[Write paper]             → Consult locked terms
        ↓
/check-term-consistency   → Verify no drift occurred
        ↓
/polish-paper            → Fix remaining language issues
```

## Self-Dispatch Phases

**This skill has 1 setup phase + 5 parallel check phases. Follow `.claude/commands/_shared/self_dispatch_protocol.md`.**

| # | Phase | Independent? | Files to Read | What to Check |
|---|-------|-------------|---------------|---------------|
| 0 | Build term usage map | No (setup) | All `sections/*.tex` | For each key concept, find all terms used; build concept→terms mapping |
| 1 | Framing compliance | Yes (after 0) | (uses map + `framing.md`) | Locked terms from framing doc used consistently; "avoid" synonyms flagged |
| 2 | Semantic drift | Yes (after 0) | All `sections/*.tex` | Same concept described differently across sections ("to correct bias" vs "to debias") |
| 3 | Verb consistency | Yes (after 0) | All `sections/*.tex` | Same action uses same verb ("observe F" not "receive F"/"obtain F") |
| 4 | Adjective consistency | Yes (after 0) | All `sections/*.tex` | Same property uses same adjective ("biased" not "unreliable"/"imperfect") |
| 5 | Machine-generated smells | Yes (after 0) | All `sections/*.tex` | "Furthermore" chains; repetitive transitions; AI word patterns |
| 6 | Cross-field term collision | Yes (after 0) | All `sections/*.tex` | Key terms that collide with established terminology in adjacent fields (e.g., "context" vs contextual bandits) |

**Sequential**: Phase 0 (setup) must complete first — produces the term usage map.
**Parallel group**: Phases 1-6 can run in parallel (all consume Phase 0 output).
**Aggregation**: Merge 6 sub-reports into single term consistency report; build recommended terminology table.

---

## Begin

**Dispatch**: Setup → parallel — **Template B** from `self_dispatch_protocol.md`.
**Setup output**: Term usage map (concept → terms used, with counts and locations).

1. Follow unified protocol Steps 0A–2.5 (include RAG references)
2. Execute Phase 0 inline (build term usage map)
3. Recursion guard → if subagent, execute remaining phases inline
4. Dispatch 6 parallel Task subagents (Phases 1-6), each receives term usage map
5. Aggregate → deduplicate → sort by severity
6. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: L2 Consistency (Terminology)
   Issues Found: {N}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   /fix-issues terms           → Auto-fix terminology
   /fix-issues terms --dry-run → Preview changes first

   {If no issues:}
   ✅ Terminology is consistent. Proceed to next check.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found:]
   /fix-issues terms           → Auto-fix inconsistencies
   /check-term-consistency     → Verify fixes applied
   /update-paper-state [name]  → Sync terminology to framing.md

   [Same level (L2) - complete these:]
   /check-paper-consistency    → Check symbol conflicts
   /check-cross-references     → Check all refs valid

   [When ALL L2 checks pass:]
   /check-ms-style             → Move to L3 (venue style)

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy
   L1 Structure   ─────────── /check-content-placement, /check-paper-flow
 → L2 Consistency ─────────── YOU ARE HERE
   L3 Style       ─────────── /check-ms-style
   L4 Language    ─────────── /polish-paper (only after L0-L3 pass)

💡 TIP: Use /paper-pipeline status to see overall progress
```
