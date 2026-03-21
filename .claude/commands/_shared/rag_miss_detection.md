# RAG Miss Detection Protocol (Shared)

**Include this section in ALL commands that use RAG for suggestions.**

---

## RAG Miss Detection

When searching the writing reference library for patterns:

### 1. Evaluate Match Quality

```
Search result similarity:
- ≥ 0.8: ✅ GOOD MATCH - Use pattern directly
- 0.7-0.8: ⚠️ PARTIAL MATCH - Adapt with caution
- < 0.7: ❌ RAG_MISS - Log and warn
```

### 2. On RAG_MISS (similarity < 0.7)

**Step A**: Log the miss
```
Append to .claude/writing_references/_rag_log.md:
| [date] | [time] | "[query]" | [category] | [best_match] | [similarity] | ❌ |
```

**Step B**: Warn in output
```
⚠️ RAG_MISS: No good pattern for "[query]"
   Best match: "[pattern snippet]" (similarity: 0.XX)
   Threshold: 0.70

   Falling back to general suggestion...

   💡 TIP: If you have a better example, add it:
   /rag-maintain add [category] "[your pattern]" [source]
```

**Step C**: Prompt for addition (if user provides good example)
```
📝 This looks like a good pattern! Add to RAG library?

Pattern: "[user's example]"
Category: [suggested category]

→ /rag-maintain add [category] "[pattern]" user_provided
```

### 3. Output Format

Every RAG-using command should include in its output:

```
═══════════════════════════════════════════════════════════════════
                      RAG SEARCH SUMMARY
═══════════════════════════════════════════════════════════════════

Patterns searched: {N}
├── ✅ Good matches: {X}
├── ⚠️ Partial matches: {Y}
└── ❌ Misses: {Z}

{If any misses:}
RAG gaps detected. Run /rag-maintain review to see suggestions.
```

---

## Integration Checklist

For each RAG-using command, ensure:

- [ ] Reads `_shared/rag_config.md` (has miss protocol)
- [ ] Logs misses to `_rag_log.md`
- [ ] Shows warning on miss
- [ ] Suggests `/rag-maintain add` when appropriate
- [ ] Includes RAG summary in output
