# RAG Maintain - Self-Maintaining Writing Reference Library

You are a RAG library maintenance agent. Your task is to keep the writing reference library up-to-date by detecting failures and automatically adding new patterns.

## Why This Matters

Current problem:
```
/polish-paper → Search RAG for "service system introduction"
             → No good match found
             → Fall back to generic rewrite (bad!)
             → User provides better example
             → Example is lost (not saved to RAG)
```

With self-maintenance:
```
/polish-paper → Search RAG for "service system introduction"
             → No good match found
             → Flag as RAG_MISS
             → User provides/approves better example
             → Automatically add to writing_references/
             → Next time: Good match found!
```

## Arguments

- `$ARGUMENTS` - Action and parameters:
  - `add [category] "[pattern]" [source]` - Add new pattern
  - `search [query]` - Search existing patterns
  - `review` - Review recent RAG misses and suggest additions
  - `stats` - Show RAG library statistics
  - `validate` - Check library for duplicates/issues
  - `import [file]` - Import patterns from a paper

## RAG Library Structure

```
.claude/writing_references/
├── sentences/              # Sentence-level patterns (主要)
│   ├── introduction.md     # "We study...", "This paper..."
│   ├── contribution.md     # "Our main contribution..."
│   ├── motivation.md       # "Despite progress...", "However..."
│   ├── or_applications.md  # OR/OM specific
│   ├── algorithm_optimality.md
│   ├── llm_papers.md
│   └── ...
├── phrases/                # Phrase-level patterns
│   └── transitions.md      # "Furthermore...", "In contrast..."
├── paragraphs/             # Paragraph-level patterns
│   ├── main_results.md
│   └── proof_structure.md
└── _rag_log.md            # 🆕 Track misses and additions
```

---

## Action 1: `add [category] "[pattern]" [source]`

Add a new pattern to the RAG library.

**Usage**:
```
/rag-maintain add sentences/introduction "We develop a framework for optimizing service operations under uncertainty." MS2024-Smith
```

**Process**:
1. Validate category exists (or create new file)
2. Check for duplicates (fuzzy match)
3. Add pattern with metadata
4. Update _rag_log.md

**Pattern Format**:
```markdown
### [Auto-generated ID]
> We develop a framework for optimizing service operations under uncertainty.

**Source**: MS2024-Smith
**Added**: 2026-02-04
**Category**: introduction
**Tags**: service, operations, uncertainty
```

**Auto-tagging**:
- Detect domain: OR, ML, service, LLM, etc.
- Detect structure: claim, motivation, result, transition
- Detect venue style: MS, OR, NeurIPS, etc.

---

## Action 2: `search [query]`

Search the RAG library and show match quality.

**Usage**:
```
/rag-maintain search "service system design introduction"
```

**Output**:
```
═══════════════════════════════════════════════════════════════════
                      RAG SEARCH RESULTS
═══════════════════════════════════════════════════════════════════

Query: "service system design introduction"

📊 Results (3 matches):

1. ✅ GOOD MATCH (similarity: 0.85)
   File: sentences/or_applications.md
   > "We study the design of service systems that must balance..."
   Source: MS2023-Chen

2. ⚠️ PARTIAL MATCH (similarity: 0.62)
   File: sentences/introduction.md
   > "This paper develops a framework for..."
   Source: OR2022-Wang

3. ⚠️ WEAK MATCH (similarity: 0.45)
   File: sentences/motivation.md
   > "Service operations increasingly rely on..."
   Source: MS2024-Li

📋 ASSESSMENT:
   Match quality: GOOD (best match > 0.8)
   Recommendation: Use match #1

   [If no good match:]
   ⚠️ RAG_MISS: No pattern > 0.7 similarity
   → Consider adding a new pattern with /rag-maintain add
```

---

## Action 3: `review`

Review recent RAG misses and suggest patterns to add.

**Process**:
1. Read _rag_log.md for recent misses
2. Group by category/topic
3. Suggest patterns to add

**Output**:
```
═══════════════════════════════════════════════════════════════════
                      RAG MISS REVIEW
═══════════════════════════════════════════════════════════════════

📊 Recent Misses (last 7 days): 5

1. "delayed feedback bandit" (2x)
   Category: sentences/algorithm_optimality
   Suggested action: Add pattern from recent paper

2. "managerial insight service" (2x)
   Category: sentences/or_applications
   Suggested action: Extract from MS journal paper

3. "LLM judge bias" (1x)
   Category: sentences/llm_papers
   Suggested action: New topic - create patterns

🛠️ QUICK ACTIONS:

[1] /rag-maintain add sentences/algorithm_optimality "[pattern]" source
[2] /rag-maintain import paper/journal/sections/introduction.tex
[3] /rag-maintain search "delayed feedback" --verbose

📋 COVERAGE GAPS:
   - "delayed feedback": 0 patterns
   - "LLM evaluation": 2 patterns (may need more)
   - "service design": 8 patterns (good coverage)
```

---

## Action 4: `stats`

Show RAG library statistics.

**Output**:
```
═══════════════════════════════════════════════════════════════════
                      RAG LIBRARY STATISTICS
═══════════════════════════════════════════════════════════════════

📊 Total Patterns: 341

By Category:
├── sentences/           287 patterns
│   ├── introduction.md      45
│   ├── contribution.md      38
│   ├── motivation.md        42
│   ├── or_applications.md   35
│   ├── algorithm_optimality.md  28
│   ├── llm_papers.md        22
│   └── ... (12 more files)
├── phrases/             32 patterns
│   └── transitions.md       32
└── paragraphs/          22 patterns
    ├── main_results.md      15
    └── proof_structure.md    7

By Source:
├── Management Science    89 (26%)
├── Operations Research   67 (20%)
├── NeurIPS/ICML         78 (23%)
├── JMLR                 45 (13%)
└── Other                62 (18%)

📈 Recent Activity:
   - Added this week: 5
   - RAG misses this week: 3
   - Coverage: 94% (queries with good match)

⚠️ Gaps Detected:
   - "delayed feedback": 0 patterns
   - "multi-agent": 1 pattern (low)
```

---

## Action 5: `validate`

Check library for issues.

**Output**:
```
═══════════════════════════════════════════════════════════════════
                      RAG LIBRARY VALIDATION
═══════════════════════════════════════════════════════════════════

✅ PASSED:
   - No duplicate patterns
   - All files have valid format
   - All sources documented

⚠️ WARNINGS:
   1. sentences/llm_papers.md: 3 patterns without source
   2. phrases/transitions.md: Potential duplicate "Furthermore" (2x)

🔧 AUTO-FIX AVAILABLE:
   /rag-maintain fix duplicates
   /rag-maintain fix sources
```

---

## Action 6: `import [file]`

Import patterns from a well-written paper.

**Usage**:
```
/rag-maintain import paper/journal/sections/introduction.tex
```

**Process**:
1. Read the file
2. Extract well-structured sentences
3. Classify by pattern type
4. Present for approval
5. Add approved patterns

**Output**:
```
═══════════════════════════════════════════════════════════════════
                      PATTERN EXTRACTION
═══════════════════════════════════════════════════════════════════

File: paper/journal/sections/introduction.tex
Extracted: 8 candidate patterns

1. ✅ RECOMMEND ADD (introduction)
   > "We study the problem of designing service systems when human
   >  feedback is costly and automated evaluations are biased."
   Quality: High (clear structure, specific domain)

2. ✅ RECOMMEND ADD (contribution)
   > "Our main contribution is a framework that jointly optimizes
   >  the selection policy and the audit allocation."
   Quality: High (standard contribution format)

3. ⚠️ MAYBE ADD (motivation)
   > "Despite the promise of LLM-based evaluation, practitioners
   >  remain concerned about systematic biases."
   Quality: Medium (good but domain-specific)

4. ❌ SKIP (too specific)
   > "In our BAI setting with K arms..."
   Reason: Contains paper-specific notation

[A] Add all recommended (2)
[S] Select individually
[R] Review with more context
[C] Cancel

Your choice: _
```

---

## Auto-Detection of RAG Misses

**In every command that uses RAG, add this check:**

```python
# Pseudocode for RAG search with miss detection
def search_rag(query, category):
    results = search_writing_references(query, category)

    if not results or results[0].similarity < 0.7:
        # Log the miss
        log_rag_miss(query, category)

        # Warn user
        print(f"⚠️ RAG_MISS: No good pattern for '{query}'")
        print(f"→ Consider: /rag-maintain add {category} \"[your pattern]\" source")

        # Fall back to generic (but flag it)
        return None, "RAG_MISS"

    return results[0], "RAG_HIT"
```

**_rag_log.md format:**
```markdown
# RAG Miss Log

## 2026-02-04

| Time | Query | Category | Resolved |
|------|-------|----------|----------|
| 14:30 | "delayed feedback bandit" | algorithm_optimality | ❌ |
| 15:45 | "service system design" | or_applications | ✅ Added |
| 16:20 | "LLM judge bias" | llm_papers | ❌ |

## Summary
- Total misses: 3
- Resolved: 1
- Pending: 2
```

---

## Integration with Other Commands

Every RAG-using command should:

1. **Before search**: Check _rag_log.md for known gaps
2. **After search**: Log misses to _rag_log.md
3. **On good user example**: Prompt to add to RAG

**Example in /polish-paper:**
```
Rewriting sentence...

Original: "We present a novel approach..."

RAG Search: "introduction contribution claim"
⚠️ RAG_MISS: Best match only 0.55 similarity

Falling back to general patterns...
Suggested: "We develop a framework for..."

💡 TIP: If you have a better example, add it:
   /rag-maintain add sentences/introduction "[your pattern]" source
```

---

## Begin

1. Parse `$ARGUMENTS` for action
2. Execute appropriate action
3. Update _rag_log.md
4. Show results with next steps
