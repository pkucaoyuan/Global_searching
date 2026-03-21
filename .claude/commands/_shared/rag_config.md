# Shared RAG Configuration for Paper Review Skills

**IMPORTANT**: This file defines the RAG usage protocol for ALL paper review skills. Every skill that reviews or modifies paper content MUST use these references.

## Core Principle

> "Every rewrite should be grounded in human-authored patterns from top venues, not generic AI paraphrasing."

## Reference Library Location

All references are in `.claude/writing_references/`:
- **341 entries** from JMLR, NeurIPS, Management Science, Operations Research, etc.
- **36 files** covering phrases, sentences, and paragraphs

## Section → Reference Mapping

When reviewing/rewriting any section, read the corresponding reference files FIRST:

| Section Type | Reference Files to Read |
|--------------|------------------------|
| **Introduction** | `sentences/introduction.md`, `sentences/contribution.md`, `sentences/motivation.md` |
| **Related Work** | `sentences/related_work.md`, `sentences/citation_patterns.md`, `phrases/transitions.md` |
| **Model/Setup** | `sentences/problem_setup.md`, `sentences/or_applications.md` |
| **Algorithm** | `sentences/algorithm_optimality.md`, `paragraphs/proof_structure.md` |
| **Theory/Analysis** | `phrases/hedging.md`, `paragraphs/proof_structure.md` |
| **Experiments** | `sentences/dynamic_pricing.md`, `paragraphs/main_results.md` |
| **Discussion** | `sentences/contribution.md`, `paragraphs/main_results.md` |
| **Conclusion** | `sentences/contribution.md` |

## Topic-Specific References

When the paper topic matches, also read:

| Paper Topic | Additional References |
|-------------|----------------------|
| LLM/AI | `sentences/llm_papers.md` |
| Bandits/BAI | `sentences/algorithm_optimality.md`, `sentences/online_learning.md` |
| Operations Research | `sentences/or_applications.md`, `sentences/revenue_management.md` |
| Pricing | `sentences/dynamic_pricing.md`, `sentences/choice_models.md` |
| Optimization | `sentences/robust_optimization.md`, `sentences/approximation_algorithms.md` |
| Causal Inference | `sentences/causal_inference.md` |
| Queueing | `sentences/queueing_theory.md` |
| Inventory | `sentences/inventory_management.md` |
| MDP/RL | `sentences/markov_decision_process.md` |

## Universal References (Always Useful)

These files are useful regardless of section:

| File | Use Case |
|------|----------|
| `phrases/transitions.md` | Section/paragraph transitions |
| `phrases/hedging.md` | Cautious academic language |
| `paragraphs/paper_roadmap.md` | Organizing paper structure |
| `guides/academic_writing_principles.md` | Core writing principles |

## MS (Management Science) Specific References

For MS journal submissions, prioritize:

| File | Why |
|------|-----|
| `sentences/or_applications.md` | OR framing patterns |
| `sentences/revenue_management.md` | MS-style business context |
| `sentences/dynamic_pricing.md` | Managerial insights language |
| `sentences/choice_models.md` | Decision-making framing |
| `paragraphs/main_results.md` | Results presentation |

## How to Use References in Any Skill

### Step 1: Read Before Reviewing
```
Before reviewing Section X:
1. Identify section type from mapping above
2. Read the corresponding reference files using Read tool
3. Note patterns that match the paper's content
```

### Step 2: Match Patterns When Suggesting Changes
```
When suggesting a rewrite:
1. Find a reference pattern that matches the intent
2. Adapt the pattern to the specific content
3. Cite the pattern in your suggestion (e.g., "Pattern from Kaufmann2016")
```

### Step 3: Verify Rewrites Against References
```
After proposing a change:
1. Does it follow the sentence structure from references?
2. Does it use transitions from the phrase library?
3. Does it match the hedging level appropriate for the claim?
```

## Anti-Patterns (What NOT to Do)

❌ **Generic AI paraphrasing** without consulting references
❌ **Inventing transitions** when `phrases/transitions.md` has proven patterns
❌ **Stating contributions** without checking `sentences/contribution.md`
❌ **Writing proofs** without checking `paragraphs/proof_structure.md`

## Integration Protocol

Every paper review skill should:

1. **At startup**: Read this config + relevant reference files
2. **During review**: Match issues to reference patterns
3. **When suggesting fixes**: Ground suggestions in specific patterns
4. **In output**: Note which patterns were applied

## Quick Reference: File Paths

```
.claude/writing_references/
├── phrases/transitions.md          # 7 entries
├── phrases/hedging.md              # 7 entries
├── sentences/introduction.md       # 6 entries
├── sentences/contribution.md       # 6 entries
├── sentences/motivation.md         # 8 entries
├── sentences/problem_setup.md      # 9 entries
├── sentences/related_work.md       # 8 entries
├── sentences/citation_patterns.md  # 10 entries
├── sentences/or_applications.md    # 9 entries
├── sentences/algorithm_optimality.md # 12 entries
├── sentences/llm_papers.md         # 13 entries
├── paragraphs/main_results.md      # 10 entries
├── paragraphs/proof_structure.md   # 11 entries
├── paragraphs/paper_roadmap.md     # 5 entries
├── guides/academic_writing_principles.md
└── _rag_log.md                     # 🆕 Track misses
```

---

## 🆕 RAG Self-Maintenance Protocol

### Detecting RAG Misses

When searching for patterns, if no good match is found (similarity < 0.7):

1. **Log the miss** to `_rag_log.md`:
   ```
   | 2026-02-04 | 14:30 | "delayed feedback bandit" | algorithm_optimality | 0.45 | ❌ |
   ```

2. **Warn the user**:
   ```
   ⚠️ RAG_MISS: No good pattern for "delayed feedback bandit"
   → Best match: 0.45 similarity (threshold: 0.7)
   → Consider: /rag-maintain add sentences/algorithm_optimality "[pattern]" source
   ```

3. **Prompt for addition** (if user provides good example):
   ```
   💡 Good example detected! Add to RAG library?
   /rag-maintain add sentences/algorithm_optimality "..." user_provided
   ```

### Auto-Add Protocol

When user provides a better example during review:

1. **Detect quality**: Well-structured sentence matching academic style
2. **Suggest category**: Based on section type and content
3. **Prompt for approval**:
   ```
   📝 Would you like to add this pattern to RAG?
   Pattern: "We study delayed feedback in bandit optimization..."
   Category: sentences/algorithm_optimality
   [Y] Yes, add  [N] No, skip  [E] Edit first
   ```
4. **Add if approved**: Update the reference file + _rag_log.md

### Periodic Maintenance

Run `/rag-maintain review` periodically to:
- Review accumulated misses
- Identify coverage gaps
- Suggest patterns to add

### Commands

| Command | Purpose |
|---------|---------|
| `/rag-maintain add [cat] "[pattern]" [src]` | Add new pattern |
| `/rag-maintain search [query]` | Search with quality metrics |
| `/rag-maintain review` | Review misses, suggest additions |
| `/rag-maintain stats` | Show library coverage |
| `/rag-maintain import [file]` | Extract patterns from paper |
