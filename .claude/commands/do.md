# Do - Natural Language Meta-Router

You are a meta-router. Classify the user's natural language intent and dispatch to the correct orchestrator.

**This is the "when in doubt" command.** Users don't need to remember specific command names — just describe what they want.

## Arguments

`$ARGUMENTS` — Free-form natural language describing what the user wants. Examples: `review my paper for MS`, `check if proofs are correct`, `start a new session`, `initialize paper state`

## Protocol

1. Read `.claude/commands/_shared/orchestrator_protocol.md` for dispatch rules
2. Classify intent using the keyword table below
3. Dispatch to the correct orchestrator via Skill tool
4. Pass the parsed action as arguments

## Intent Classification

| Keywords in User Input | Routes To | Skill | Parsed Args |
|----------------------|-----------|-------|-------------|
| check, review, fix, polish, submit, pipeline, consistency, redundancy, flow, style, figures | `/paper` | `paper` | (extract action) |
| proof, theorem, verify, derive, gemini, lemma, proposition, theory, refine | `/theory` | `theory` | (extract action) |
| session, start, organize, progress, rag, end, begin, close, icons, research, broadcast, deploy, sync | `/project` | `project` | (extract action) |
| init state, framing, overview, comments, reviewer, refs, citations, state | `/state` | `state` | (extract action) |

## Classification Rules

1. **Scan for keywords** in `$ARGUMENTS` (case-insensitive)
2. **Prioritize specificity**: "verify proof of theorem 5" → `/theory verify thm:5` (not `/paper check`)
3. **Multiple matches**: If keywords match multiple orchestrators, pick the most specific:
   - "check proof consistency" → `/theory check` (proof is more specific)
   - "check paper consistency" → `/paper check consistency`
4. **Venue extraction**: Look for MS, OR, ML, Management Science, Operations Research, NeurIPS, ICML, JMLR
5. **No match**: Show the cheat sheet and ask user to clarify

## Parsing Examples

| User Says | Classification | Dispatches To |
|-----------|---------------|---------------|
| `review my paper for MS` | paper + review + MS | `Skill: paper, args: "review MS"` |
| `check if proofs are correct` | theory + verify | `Skill: theory, args: "verify all"` |
| `start a new session` | project + start | `Skill: project, args: "start"` |
| `fix all the symbol issues` | paper + fix + symbols | `Skill: paper, args: "fix symbols"` |
| `initialize paper state for my_paper` | state + init | `Skill: state, args: "init my_paper"` |
| `what's the status of reviewer comments` | state + comments | `Skill: state, args: "comments status"` |
| `run the full pipeline` | paper + pipeline | `Skill: paper, args: "review"` |
| `check figures and tables` | paper + figures | `Skill: paper, args: "figures"` |
| `generate gemini prompt for proof-fix` | theory + gemini | `Skill: theory, args: "gemini proof-fix"` |
| `how's my RAG library doing` | project + rag | `Skill: project, args: "rag stats"` |
| `end my session and save` | project + end | `Skill: project, args: "end"` |
| `deploy commands to all projects` | project + deploy | `Skill: project, args: "broadcast commands"` |
| `sync shared files` | project + sync | `Skill: project, args: "broadcast shared"` |
| `broadcast standards` | project + broadcast | `Skill: project, args: "broadcast standards"` |
| `is my paper ready to submit` | paper + pre-submit | `Skill: paper, args: "pre-submit"` |
| `track the new reviewer comments` | state + comments | `Skill: state, args: "comments add"` |
| `refine the theory section` | theory + refine | `Skill: theory, args: "refine"` |
| `verify theorem lower bound` | theory + verify | `Skill: theory, args: "verify thm:lower_bound"` |
| `quick check` | paper + quick | `Skill: paper, args: "quick"` |

## No-Match Fallback

If no keywords match, show the cheat sheet:

```
I couldn't determine what you'd like to do from: "[input]"

Here's what's available:

  /paper review MS     - Full paper review
  /paper quick         - Fast consistency check
  /paper fix all       - Auto-fix issues
  /paper check [what]  - Specific checks

  /theory verify       - Check proofs
  /theory status       - Theorem overview
  /theory refine       - Refine theory sections

  /project start       - New session
  /project end         - Close session
  /project rag stats   - RAG library health

  /state init [name]   - Initialize paper docs
  /state comments      - Track reviewer comments
  /state status        - Paper state health

Or describe what you want in different words and I'll try again.
```

## Begin

Classify `$ARGUMENTS` and dispatch to the correct orchestrator via Skill tool. Show your classification reasoning briefly.
