# Polish Report Template

## Final Report Format

When converged, output:

```markdown
# Polish Report

## Summary
- **Paper root**: [discovered path]
- **Files processed**: [count]
- **Rounds completed**: N
- **Total modifications**: M
- **Convergence**: Yes (last round: X modifications < 3)

## Modifications by Category
| Category | Count | Examples |
|----------|-------|----------|
| AI_WORD | X | "significantly" → "+9.1%" |
| PASSIVE | Y | "is shown" → "we show" |
| ZOMBIE | Z | "implementation" → "implementing" |
| FLOW | W | reordered old→new |
| STRUCTURE | V | varied transitions |
| DOMAIN | U | domain-specific improvement |

## Files Modified
- [file1.tex]: N changes
- [file2.tex]: M changes
- ...

## Remaining Issues (Manual Review Suggested)
- [List any patterns that need human judgment]
```

## Modification Categories

Track modifications by category:

**Advisor Rule Categories (L1 — highest priority):**
- `VERB_STRENGTHEN`: Weak/vague verb → precise, assertive verb
- `COMPRESS`: Sentence compression (remove filler, tighten phrasing)
- `META_REMOVE`: Meta-discourse removal ("This section discusses...")
- `HEDGE_CALIBRATE`: Hedging calibration (over-hedged or under-hedged)
- `PRECISION`: Precision increase (vague claim → specific number/reference)
- `PARA_MERGE`: Paragraph merging (fragmented short paragraphs combined)
- `PREVIEW_REMOVE`: Result preview removal (don't telegraph results before proving)
- `DERIVATION`: Physical derivation improvement (notation, step clarity)
- `FORMAT`: Formatting improvement (multi-line equations, display math)
- `CAPTION`: Caption consequence (add "what to learn" to figure/table captions)

**General Categories (L0, L2-L7):**
- `AI_WORD`: Forbidden word replaced/deleted
- `PASSIVE`: Passive→active voice
- `ZOMBIE`: Nominalization revived
- `FLOW`: Logic flow improved (old→new, topic/stress positions)
- `STRUCTURE`: Sentence/transition restructured
- `DOMAIN`: Domain-specific improvement (notation, terminology, conventions)
- `REGRESSION`: Previously fixed issue detected and re-fixed (from regression guard)

## Next Steps Footer

Every output MUST end with:

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: L4 Language (Polish)
   Issues Fixed: {N}
   Remaining: {M}

🔴 IMMEDIATE ACTIONS:
   {If issues remain:}
   1. Review suggested edits above
   2. Accept/reject each change
   3. Re-run /polish-paper for another pass

   {If converged:}
   ✅ Language polish complete. Ready for final checks.

🛠️ RECOMMENDED COMMANDS (in order):

   [If not converged:]
   /polish-paper               → Another polish iteration

   [When L4 converges:]
   /paper-pipeline quick       → Fast consistency re-check
   /paper-pipeline pre-submit MS → Final submission checklist

📋 REVIEW LEVELS REMINDER:
   L0 Content     ─────────── /check-content-redundancy     ✅
   L1 Structure   ─────────── /check-content-placement      ✅
   L2 Consistency ─────────── /check-paper-consistency      ✅
   L3 Style       ─────────── /check-ms-style               ✅
 → L4 Language    ─────────── YOU ARE HERE (FINAL LEVEL)

🎉 After all checks pass:
   /paper-pipeline pre-submit MS → 100% = Ready to submit!
```
