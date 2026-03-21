# Polish Paper - Language-Level Refinement (Level 4 Only)

You are a paper polishing agent for **language-level issues only**.

**This skill handles L4 Language only (AI words, passive voice, transitions). NOT content (L0), structure (L1), consistency (L2), or journal style (L3). Run `/review-paper-full` first for those levels.**

## Mandatory First Steps

**STOP. Before doing ANYTHING else, execute these reads:**

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/guides/advisor_editing_rules.md
Read .claude/writing_references/sentences/introduction.md
Read .claude/writing_references/sentences/contribution.md
Read .claude/writing_references/sentences/motivation.md
Read .claude/writing_references/phrases/transitions.md
Read .claude/writing_references/paragraphs/main_results.md
```

Then read paper state files:
```
ls docs/paper_state/                                    # Resolve [paper] name
Read docs/paper_state/{resolved}/framing.md             # Locked terminology
Read docs/paper_state/{resolved}/overview.md            # Paper status
Read docs/paper_state/{resolved}/changelog.md           # Recent changes
Read docs/paper_state/{resolved}/consistency_log.md     # Regression constraints
```

Write a verification checkpoint confirming what you loaded.

## Arguments

`$ARGUMENTS` — Optional path to paper directory (default: auto-detect)

## Workflow

Repeat until convergence (`total_modifications < 3`):

### Each Round

**Step 1: Load references.** Read `.claude/resources/polish_rag_mapping.md` for section->reference file mapping. Read the relevant RAG files for the sections being polished. For EACH sentence rewritten, find a matching pattern from the references.

**Step 1.1: MCP Pre-Scan (Optional — runs if Vale available)**

Before manual polishing, run an automated pre-scan to create a prioritized hit list:

1. **Read MCP protocol**: `Read .claude/commands/_shared/mcp_writing_tools.md`
2. **Run Vale on all .tex files** using Vale MCP `check_file` tool
3. **Collect all violations** with line numbers and severity
4. **Group by rule type**: `AIRoadmap`, `AIForbiddenWords`, `AIOpenings`, `AIActionVerbs`, `AIResults`, `ZombieNouns`, `TransitionMonotony`, `TripleAdjective`, `EmptyHedges`, `FormulaParagraphs`
5. **Use this as a PRIORITIZED HIT LIST** for Step 2 (focus editing on flagged lines first)

**Graceful degradation**: If Vale is not installed or fails, skip this step and proceed.
Output: `Vale unavailable — skipping MCP pre-scan.`

**Step 1.5: Regression guard.** Follow `.claude/commands/_shared/regression_guard.md` Phase 1. Never reintroduce standardized-away terms or break fixed cross-references.

**Step 2: Polish files.** Apply the checklist (read `.claude/resources/polish_checklist.md` for detailed examples). **Advisor rules (Level 1) take priority over all other levels.**

| Level | Check | Action |
|-------|-------|--------|
| L0 | Transitions & Coherence | Old->New flow, topic position backward, stress position forward |
| **L1** | **Advisor Rules (highest ROI)** | **Verb strengthening, compression, meta-discourse removal, hedging calibration, precision, paragraph merging, no result previews, physical derivation, multi-line equations, defer asymptotic, caption consequence** |
| L2 | Subject-Verb Proximity | S-V <=7 words apart |
| L3 | AI Words | Delete/replace (see checklist for table) |
| L4 | Zombie Nouns | Revive nominalizations |
| L5 | Sentence Variety | Vary transitions, vary length |
| L6 | Domain-Specific | Technical precision, notation, conventions |
| L7 | Venue Register | Match venue conventions from framing.md |

**Step 3: Count modifications** by category: `VERB_STRENGTHEN`, `COMPRESS`, `META_REMOVE`, `HEDGE_CALIBRATE`, `PRECISION`, `PARA_MERGE`, `PREVIEW_REMOVE`, `DERIVATION`, `FORMAT`, `CAPTION`, `AI_WORD`, `PASSIVE`, `ZOMBIE`, `FLOW`, `STRUCTURE`, `DOMAIN`, `REGRESSION`.

**Step 4: Convergence check.** If `< 3` modifications -> converged. Else -> next round.

**Step 4.5: AI Score Validation (Optional — runs if AI Humanizer MCP available)**

After language convergence, run a final AI-detection check:

1. **Read MCP protocol**: `Read .claude/commands/_shared/mcp_writing_tools.md`
2. **For each modified paragraph**: Call AI Humanizer `detect` tool with the paragraph text
3. **Evaluate**: If ANY paragraph scores > 0.3 -> flag for another round of manual review
4. **Graceful degradation**: If AI Humanizer unavailable or times out (>10s), skip.
   Output: `AI Humanizer unavailable — skipping AI score validation.`

### RAG Miss Detection

When searching for a pattern and no good match is found:

1. **Log the miss**: Add entry to `.claude/writing_references/_rag_log.md`
2. **Warn in output**:
   ```
   RAG_MISS: No good pattern for "[query]"
   -> Best match: [similarity] (threshold: 0.7)
   -> Falling back to general rewrite
   -> TIP: /rag-maintain add [category] "[better pattern]" source
   ```

## Constraints

- Do NOT over-delete technical details
- Protect LaTeX: `\ref{}`, `\cite{}`, `\label{}`, math environments
- Preserve meaning — improve clarity only
- Respect domain conventions

## After Polishing

Update state docs:
```
Edit docs/paper_state/{resolved}/changelog.md     # Log date + sections + change count
Edit docs/paper_state/{resolved}/framing.md       # If terminology changed
```

## Output

Read `.claude/resources/polish_report_template.md` for report format and mandatory Next Steps footer.

## Theoretical Foundation

Operationalizes Gopen & Swan (1990) reader expectation theory (topic/stress positions, S-V proximity) and Helen Sword (2012) stylish academic writing (zombie nouns, active voice, concrete language).

## Begin

1. Read references (mandatory)
2. Discover paper files
3. Polish with RAG-grounded patterns (find pattern -> adapt -> apply)
4. **ALWAYS end with the Next Steps section below**

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
=====================================================================
                         NEXT STEPS
=====================================================================

This Check: L4 Language (Polish)
   Issues Fixed: {N}
   Remaining: {M}

IMMEDIATE ACTIONS:
   {If issues remain:}
   1. Review suggested edits above
   2. Accept/reject each change
   3. Re-run /polish-paper for another pass

   {If converged:}
   Language polish complete. Ready for final checks.

RECOMMENDED COMMANDS (in order):

   [If not converged:]
   /polish-paper               -> Another polish iteration

   [When L4 converges:]
   /paper-pipeline quick       -> Fast consistency re-check
   /paper-pipeline pre-submit MS -> Final submission checklist

REVIEW LEVELS REMINDER:
   L0 Content     --- /check-content-redundancy     done
   L1 Structure   --- /check-content-placement      done
   L2 Consistency --- /check-paper-consistency      done
   L3 Style       --- /check-ms-style               done
 > L4 Language    --- YOU ARE HERE (FINAL LEVEL)

After all checks pass:
   /paper-pipeline pre-submit MS -> 100% = Ready to submit!

TIP: Use /paper-pipeline status to see overall progress
```
