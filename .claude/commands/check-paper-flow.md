# Check Paper Flow - Coherence and Consistency Review

You are a paper flow analysis agent. Your task is to systematically review academic papers for logical flow, contradictions, and redundancies.

## ⚠️ MANDATORY: RAG-Grounded Suggestions

**All flow improvement suggestions MUST be grounded in human-authored patterns.**

### Step 0: Read Shared Config & Flow-Relevant References

Before starting any review, execute these Read tool calls:

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/phrases/transitions.md
Read .claude/writing_references/paragraphs/paper_roadmap.md
Read .claude/writing_references/paragraphs/main_results.md
Read .claude/writing_references/guides/academic_writing_principles.md
```

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before checking paper flow, you MUST:
1. Resolve `[paper]` → run `ls docs/paper_state/` to find the actual directory name
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/overview.md       # Paper structure
Read docs/paper_state/{resolved}/framing.md        # Locked concepts
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- overview.md: [section order, paper structure, page count]
- framing.md: [locked concepts and terminology decisions]
```

**If you skip this step, you may suggest flow changes that contradict the paper's deliberate framing.**

Use these patterns when:
- Suggesting transitions between sections
- Recommending how to connect paragraphs
- Proposing preview-detail-summary structures

### Step 2: Apply RAG Miss Detection

If any RAG search fails to find a good match (similarity < 0.7), follow the protocol in:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

This ensures the RAG library self-maintains by logging gaps and suggesting additions.

---

## Related Skills

This skill focuses on **flow and transitions**. For deeper checks, also run:
- `/check-paper-consistency` - Symbol conflicts, notation uniformity
- `/check-content-redundancy` - Result/definition redundancy across sections
- `/check-ms-style` - Journal-specific style (for MS submissions)
- `/review-paper-full MS` - Comprehensive multi-level review

## Arguments

- `$ARGUMENTS` - Optional: specific section to check (e.g., `introduction`, `training`). If omitted, checks entire paper.

## Setup

**Step 0: Locate Paper Files**

1. Find paper root directory (typically `paper/`)
2. Discover all main section files in order:
   ```
   paper/sections/abstract.tex
   paper/sections/introduction.tex
   paper/sections/setup.tex
   paper/sections/construction.tex
   paper/sections/training.tex
   paper/sections/experiments.tex
   paper/sections/related.tex
   paper/sections/conclusion.tex
   ```
3. If `$ARGUMENTS` specifies a section, focus on that section and its neighbors

## Workflow

### Phase 1: Sequential Reading

Read all sections **in order**, building a mental model of:
- Key claims and contributions
- Numerical values (dataset sizes, model counts, metrics)
- Terminology definitions
- Method descriptions

### Phase 2: Flow Analysis

For each section transition, evaluate:

| Transition | Check | Good Example | Bad Example |
|------------|-------|--------------|-------------|
| Abstract → Intro | Does intro expand on abstract? | "Section 1 elaborates..." | Repeats abstract verbatim |
| Intro → Setup | Clear motivation for benchmarks? | "To address this, we design..." | Jumps to technical details |
| Setup → Construction | Logical build-up? | "Having defined metrics, we now describe..." | No connection |
| Construction → Training | Method connection? | "These benchmarks enable training via..." | Abrupt topic change |
| Training → Experiments | Evaluation setup clear? | "We evaluate the trained models..." | Missing bridge |
| Experiments → Related | Positioning clear? | "Our findings extend prior work..." | Disconnected |
| Related → Conclusion | Synthesis? | "In summary, we contributed..." | Just repeats |

**Within-Section Flow:**
- Topic sentences present at paragraph starts?
- Old→New information order followed?
- Logical connectors appropriate (not "Furthermore" chains)?

### Phase 3: Contradiction Detection

Compare these critical elements across sections:

**Numerical Consistency:**
| Element | Check Locations |
|---------|-----------------|
| Dataset size | abstract, intro contributions, setup, experiments |
| Model count | abstract, experiments, appendix |
| Performance metrics | training results, experiments tables, conclusion |
| Training hyperparameters | training section, appendix |

**Claim Consistency:**
| Claim Type | Check |
|------------|-------|
| Contributions | intro list vs conclusion summary |
| Method novelty | intro vs related work positioning |
| Results interpretation | experiments vs conclusion |

**Thesis Consistency:**

Extract the paper's core thesis from the abstract/introduction (the main argument the paper makes). Then scan every section for claims that undermine or contradict it.

| Check | How |
|-------|-----|
| Extract thesis | Identify the 1-2 sentence core argument from abstract + intro |
| Scan for contradictions | For each section, check if any claim suggests the opposite of the thesis |
| Check messaging | Ensure no section sends a message that undermines the paper's story |

**Common thesis contradictions:**
- Abstract says "method X solves problem Y" but a section says "problem Y remains challenging even with X"
- Introduction claims general applicability but experiments section says "limited to setting Z"
- Paper argues for approach A's superiority but related work section describes approach B as equally effective without clear differentiation

**Diagnostic**: For each section, ask: "Does this section reinforce or weaken the paper's main argument?" Any section that weakens it (even unintentionally) is a thesis consistency issue.

**Terminology Consistency:**
- Same metric names throughout (RR@5, DA, etc.)
- Same model names (Qwen3-GRPO vs Qwen-GRPO)
- Same benchmark names (OR-Debug vs OR-Debug-Bench)

### Phase 4: Redundancy Detection

Check for:

**Definition Redundancy:**
- IIS explained multiple times?
- Newsvendor formula repeated?
- Benchmark descriptions duplicated?

**Claim Redundancy:**
- Same contribution stated in intro and conclusion with identical wording?
- Performance numbers cited multiple times unnecessarily?

**Content Overlap:**
- Related work content also in introduction?
- Method details in both main paper and appendix without clear distinction?

### Phase 5: Novelty Threading ⭐

**This phase checks whether the paper's core novelty — the key reason this work is interesting — is clearly articulated and threaded through all major sections.** This is distinct from thesis consistency (Phase 3), which checks for contradictions. Novelty threading checks for *presence* and *variation*.

**Why this matters:** A paper can be internally consistent (no contradictions) while still burying its strongest selling point. Reviewers skim sections; if the novelty is stated only in the introduction, later sections feel disconnected from the paper's raison d'être.

**Step 1: Extract Core Novelty**

Distinguish three layers of novelty:
1. **Problem-setting novelty**: What is new about the PROBLEM being solved (not the method)? What assumption of prior work does this paper relax, or what new domain does it address?
2. **Methodological novelty**: What is new about the APPROACH or TECHNIQUE?
3. **Empirical novelty**: What new findings or insights emerge from the experiments?

For each, write a one-sentence statement. Example:
- Problem-setting: "Service performance evidence is textual, not scalar — existing optimization methods cannot process it."
- Methodological: "We combine IPW with confidence sequences for anytime-valid inference under selective auditing."
- Empirical: "90% audit cost savings with perfect accuracy on real service data."

**Step 2: Check Threading Across Sections**

For each novelty statement, check whether it is **present** (mentioned or alluded to) in each major section:

| Section | Problem-Setting | Methodological | Empirical |
|---------|----------------|----------------|-----------|
| Abstract | ✅/❌ | ✅/❌ | ✅/❌ |
| Introduction | ✅/❌ | ✅/❌ | ✅/❌ |
| Related Work | ✅/❌ | ✅/❌ | — |
| Model/Setup | ✅/❌ | — | — |
| Method | — | ✅/❌ | — |
| Algorithm | — | ✅/❌ | — |
| Experiments | ✅/❌ | — | ✅/❌ |
| Discussion | ✅/❌ | — | ✅/❌ |
| Conclusion | ✅/❌ | ✅/❌ | ✅/❌ |

**Rules:**
- **Problem-setting novelty** should appear in ≥5 sections (abstract, intro, model, experiments, conclusion). If it appears only in the introduction, flag as 🔴 **CRITICAL**.
- **Methodological novelty** should appear in ≥3 sections (abstract, intro, conclusion).
- **Empirical novelty** should appear in ≥3 sections (abstract, experiments, conclusion).

**Step 3: Check Variation**

When the same novelty appears in multiple sections, check that it is NOT copy-pasted or near-identical phrasing. Each occurrence should:
- Have a **different emphasis** appropriate to its section (e.g., motivational in intro, technical in model, practical in discussion)
- **Connect to local context** (e.g., in experiments: "These text-based performance records exemplify...")
- Use **different wording** while conveying the same core idea

**Common failures:**
- Novelty stated once in intro paragraph 2, then never again → reviewer doesn't notice it
- Novelty repeated verbatim in 4 places → feels mechanical/machine-generated
- Problem-setting novelty absent from model/setup → the formal model looks identical to classical work, novelty seems cosmetic
- Problem-setting novelty absent from experiments → case studies don't highlight what makes the data special

**Diagnostic question for each section**: "If a reviewer reads ONLY this section, would they understand what makes this paper's problem setting different from prior work?"

## Output Format

Generate a structured report:

```markdown
# Paper Flow Analysis Report

**Paper**: [title from main.tex]
**Sections Analyzed**: [count]
**Analysis Date**: [date]

---

## 1. Flow Issues

### Section Transitions
| From → To | Rating | Issue | Suggestion |
|-----------|--------|-------|------------|
| intro → setup | ⚠️ | Abrupt | Add "To evaluate these capabilities, we design..." |

### Within-Section Flow
| Section | Line | Issue |
|---------|------|-------|
| training | L40-45 | "Furthermore" chain (3x) |

---

## 2. Contradictions Found

### Numerical Discrepancies
| Value | Location 1 | Location 2 | Discrepancy |
|-------|------------|------------|-------------|
| Dataset size | abstract: "5,000" | setup: "4,800" | 200 difference |

### Claim Inconsistencies
| Claim | Location 1 | Location 2 | Issue |
|-------|------------|------------|-------|
| Best model | experiments: "GRPO" | conclusion: "Curriculum" | Different claims |

---

## 3. Redundancies Found

### Repeated Definitions
| Concept | Location 1 | Location 2 | Suggestion |
|---------|------------|------------|------------|
| IIS definition | intro:L7 | setup:L15 | Keep in setup only |

### Duplicate Claims
| Claim | Locations | Suggestion |
|-------|-----------|------------|
| "48% bias reduction" | intro, training, experiments, conclusion | Reduce to 2 mentions |

---

## 4. Novelty Threading

### Core Novelty Statements
| Layer | Statement |
|-------|-----------|
| Problem-setting | [1-sentence] |
| Methodological | [1-sentence] |
| Empirical | [1-sentence] |

### Threading Matrix
| Section | Problem-Setting | Methodological | Empirical |
|---------|:-:|:-:|:-:|
| Abstract | ✅/❌ | ✅/❌ | ✅/❌ |
| Introduction | ✅/❌ | ✅/❌ | ✅/❌ |
| Model/Setup | ✅/❌ | — | — |
| Experiments | ✅/❌ | — | ✅/❌ |
| Discussion | ✅/❌ | — | ✅/❌ |
| Conclusion | ✅/❌ | ✅/❌ | ✅/❌ |

### Gaps Found
| Novelty Layer | Missing From | Severity | Suggestion |
|---------------|-------------|----------|------------|
| Problem-setting | model.tex | 🔴 Critical | Add to X_t definition |

### Variation Quality
| Section Pair | Same Wording? | Different Emphasis? |
|-------------|:---:|:---:|
| intro vs conclusion | Yes/No | Yes/No |

---

## 5. Summary

| Category | Count | Severity |
|----------|-------|----------|
| Flow issues | X | ⚠️ Moderate |
| Contradictions | Y | 🔴 Critical if Y>0 |
| Redundancies | Z | 🟡 Minor |
| Novelty threading gaps | W | 🔴 Critical if W>2 |

### Recommended Actions
1. [Highest priority fix]
2. [Second priority]
3. ...

---

## 5. Section-by-Section Notes

### Abstract
- [Specific observations]

### Introduction
- [Specific observations]

[...continue for each section...]
```

## Constraints

- **Be specific**: Always cite line numbers or exact text
- **Prioritize**: Mark critical vs minor issues
- **Be constructive**: Provide concrete fix suggestions
- **Respect intent**: Don't suggest changes that alter meaning
- **Check both directions**: Forward references (will discuss) and backward references (as shown)

## Begin

1. Read all section files in order
2. Build consistency tracking (numbers, claims, terms)
3. Analyze flow at each transition
4. Detect contradictions and redundancies
5. **Extract core novelty (3 layers) and check threading across all sections**
6. Generate comprehensive report with actionable suggestions
7. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: L1 Structure (Flow/Coherence)
   Issues Found: {N}

🔴 IMMEDIATE ACTIONS:
   {If issues found:}
   1. Fix transition at: [section boundary]
   2. Add forward reference at: [location]
   3. Resolve contradiction: [section A] vs [section B]
   4. Re-run this check to verify

   {If no issues:}
   ✅ Paper flow is coherent. Proceed to next check.

🛠️ RECOMMENDED COMMANDS (in order):

   [If issues found - fix first, then:]
   /check-paper-flow           → Re-verify after fixes

   [Same level (L1) - complete these:]
   /check-content-placement    → Check example/proof placement

   [When L1 passes:]
   /check-paper-consistency    → Move to L2 (consistency)

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy     ✅
 → L1 Structure   ─────────── YOU ARE HERE
   L2 Consistency ─────────── /check-paper-consistency
   L3 Style       ─────────── /check-ms-style
   L4 Language    ─────────── /polish-paper

💡 TIP: Use /paper-pipeline status to see overall progress
```
