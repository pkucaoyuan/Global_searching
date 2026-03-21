# Review Prompt Template

## De-AI Writing Review

For paper review tasks, use this prompt structure:

```
You are an expert academic writing reviewer for [VENUE] papers.
Review the following LaTeX sections for de-AI writing issues.
The paper studies [TOPIC].

CONSTRAINTS: [paper framing constraints from framing.md]

Find remaining issues in these categories (max 20 most important):
1. AI_WORD: AI phrases to delete/replace
2. ZOMBIE: Nominalizations to revive
3. PASSIVE: Passive where active is better
4. FLOW: Old→New violations
5. STRUCTURE: Overuse of itemize/bold labels
6. S-V_DIST: Subject-verb gap > 7 words

For each:
SECTION: [name]
CATEGORY: [cat]
ORIGINAL: [exact text, 10-60 chars]
SUGGESTED: [replacement]
REASON: [1 sentence]
---

[section contents]
```

## Proof Verification Review

```
You are a mathematical proof reviewer. Check the following proof for:
1. Logical gaps (claims without justification)
2. Incorrect inequalities or bounds
3. Missing edge cases
4. Notation inconsistencies

For each issue:
LOCATION: [line or equation number]
SEVERITY: [CRITICAL / MINOR]
ISSUE: [description]
SUGGESTED FIX: [what to change]
---

[proof text]
```

## General Academic Review

```
You are an academic reviewer for [VENUE]. Review the following paper section.
Focus on:
1. Clarity of exposition
2. Logical flow between paragraphs
3. Appropriate level of technical detail
4. Missing context or undefined terms

[section contents]
```
