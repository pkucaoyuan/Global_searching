# MCP Writing Tools Protocol

This document describes the MCP-based writing tools available for paper review workflows.
All MCP integrations are **optional** — skills must degrade gracefully when servers are unavailable.

## Available MCP Servers

| Server | Tool | Purpose | Fallback |
|--------|------|---------|----------|
| **vale** | `check_file` | Rule-based prose linting via AcademicDeAI rules | Use manual grep patterns from de_ai_writing_guide.md |
| **ai-humanizer** | `detect`, `humanize` | AI fingerprint detection (0-1 score per paragraph) | Skip; rely on manual checklist |
| **semantic-scholar** | `search_papers`, `get_paper` | Academic paper search + metadata | Use arXiv API directly |

## Vale MCP Integration

### What It Provides
Deterministic pattern matching against 10 custom rule files in `.vale/styles/AcademicDeAI/`:

| Rule | Detects |
|------|---------|
| `AIRoadmap.yml` | "We first X, then Y", procedural signposting |
| `AIForbiddenWords.yml` | significantly, novel, comprehensive, leverages... |
| `AIOpenings.yml` | "In recent years", "plays a crucial role"... |
| `AIActionVerbs.yml` | "This paper delves into", "aims to address"... |
| `AIResults.yml` | "The results demonstrate that"... |
| `ZombieNouns.yml` | "implementation of", "conduct an analysis"... |
| `TransitionMonotony.yml` | Furthermore, Moreover, Additionally chains |
| `TripleAdjective.yml` | "comprehensive, robust, and scalable" |
| `EmptyHedges.yml` | "It is worth noting that", "In order to"... |
| `FormulaParagraphs.yml` | Formulaic `\paragraph{Interpretation.}` headers repeated after every result |

### How to Call (from skills)

**Via CLI** (preferred — works without MCP server running):
```bash
vale paper/energy_optimal_af/or_draft/sections/*.tex
```

**Via MCP** (when server is available):
```
Tool: vale.check_file
Args: { "path": "paper/energy_optimal_af/or_draft/sections/robustness.tex" }
```

### Output Format
Vale returns violations with:
- File path and line number
- Rule name (e.g., `AcademicDeAI.AIRoadmap`)
- Severity (warning/suggestion)
- Message with replacement guidance

## AI Humanizer MCP Integration

### What It Provides
Holistic AI-detection scoring: given a paragraph of text, returns a score from 0 (clearly human) to 1 (clearly AI-generated).

### How to Call
```
Tool: ai-humanizer.detect
Args: { "text": "paragraph text here" }
```

### Threshold
- Score > 0.3 → flag paragraph for manual review
- Score > 0.5 → strongly recommend rewriting

### Note
This is a cloud-based service. May have latency. Skip if response takes > 10 seconds.

## Semantic Scholar MCP Integration

### What It Provides
- Paper search by keyword or title
- Paper metadata (abstract, citation count, BibTeX, related papers)
- Extends the `/lit-acquire` pipeline

### How to Call
```
Tool: semantic-scholar.search_papers
Args: { "query": "speed scaling energy optimization", "limit": 5 }
```

## Graceful Degradation Protocol

When any MCP server is unavailable, the calling skill MUST:

1. **Detect unavailability**: If tool call fails or times out (>10s)
2. **Log**: Note in output that MCP tool was unavailable
3. **Fall back**: Use the manual alternative (grep patterns, manual checklist, arXiv API)
4. **Continue**: Do NOT abort the skill workflow

Example output when Vale MCP is unavailable:
```
⚠️ Vale MCP unavailable — falling back to manual AI pattern grep.
   Run `vale paper/energy_optimal_af/or_draft/sections/*.tex` manually to get detailed results.
```

## Configuration

- **Vale config**: `.vale.ini` (project root)
- **Vale rules**: `.vale/styles/AcademicDeAI/*.yml`
- **MCP registration**: `.mcp.json` (project root)
- **Server binaries**: `~/.claude/mcp-servers/{vale-mcp,ai-humanizer,semantic-scholar}/`
