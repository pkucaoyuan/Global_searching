# Paper Review Skills - Architecture

## Quick Start: 5 Orchestrators + 1 Meta-Router

Instead of remembering 30+ individual commands, use these 5 orchestrators:

```
/paper [action]    - Review, check, fix, polish (wraps 15+ check/fix commands)
/theory [action]   - Verify proofs, refine theory, generate Gemini prompts
/project [action]  - Session start/end, organize, progress, RAG maintenance
/state [action]    - Paper state lifecycle: init, framing, overview, comments
/do [anything]     - Natural language router (when you can't remember which)
```

### Cheat Sheet (Post-it friendly)

```
/paper review MS    ← Full review (MS)
/paper review MSOM  ← Full review (MSOM)
/paper quick        ← Fast check
/paper fix all      ← Auto-fix
/theory verify      ← Check proofs
/theory status      ← Theorem overview
/project start      ← New session
/project end        ← Close session
/state comments     ← Track reviews
/do [anything]      ← When in doubt
```

All 28+ underlying commands remain directly callable (e.g., `/check-paper-consistency` still works).

---

## Core Principle: RAG-Grounded Suggestions

**All paper review skills use RAG (Retrieval-Augmented Generation) with the writing reference library.**

Every suggestion MUST be grounded in human-authored patterns from:
- `.claude/writing_references/` - 341 entries from top venues (JMLR, NeurIPS, MS, OR)
- See `.claude/commands/_shared/rag_config.md` for the full protocol

This ensures suggestions match the style of top papers, not generic AI paraphrasing.

---

## The Problem

Previous approach gave false confidence:
```
/polish-paper → 4 rounds → "Converged"
             → Reviewer still has 14 comments
```

Why? Because `/polish-paper` only checked **language** (AI words, passive voice), not:
- Content (concept breadth, result redundancy)
- Structure (section organization)
- Consistency (symbol conflicts)
- Journal style (MS vs ML framing)

## The 6-Level Review Framework

```
┌─────────────────────────────────────────────────────────────┐
│ Level 0: CONTENT                    /review-paper-full      │
│ - Concept definition breadth        /check-content-redundancy│
│ - Result redundancy                                         │
│ - Experiment-framing alignment                              │
├─────────────────────────────────────────────────────────────┤
│ Level 1: STRUCTURE                  /check-content-redundancy│
│ - Section organization              /check-paper-flow        │
│ - Example-theorem pairing                                   │
├─────────────────────────────────────────────────────────────┤
│ Level 2: CONSISTENCY                /check-paper-consistency │
│ - Symbol conflicts (b_k vs b(t))                            │
│ - Term uniformity                                           │
│ - Model-algorithm match                                     │
├─────────────────────────────────────────────────────────────┤
│ Level 3: JOURNAL STYLE              /check-ms-style          │
│ - MS vs ML framing                                          │
│ - Managerial insights                                       │
│ - Arm definition breadth                                    │
├─────────────────────────────────────────────────────────────┤
│ Level 4: FIGURES & TABLES           /check-figures-tables    │
│ - Vector format, DPI, font (CMU Serif)                      │
│ - Legend placement (outside, no overlap)                     │
│ - Caption quality (self-contained)                          │
│ - Color consistency, table booktabs                         │
├─────────────────────────────────────────────────────────────┤
│ Level 5: LANGUAGE                   /polish-paper            │
│ - AI word removal                                           │
│ - Passive voice                                             │
│ - Transitions                                               │
└─────────────────────────────────────────────────────────────┘
```

## Skill Descriptions

### 🚀 Automation Skills (Start Here)
| Skill | Purpose |
|-------|---------|
| `/paper-pipeline init [name] MS` | **New paper?** Creates all docs + initial scan |
| `/paper-pipeline review MS` | **Full review.** Runs all 5 levels automatically |
| `/paper-pipeline quick` | **After changes.** Fast consistency check |
| `/paper-pipeline pre-submit MS` | **Before submit.** Final checklist |
| `/paper-pipeline revision` | **Got reviews?** Structured revision workflow |
| `/paper-pipeline status` | **Where am I?** Current state + next steps |
| `/session-start [name]` | **New session?** Instant context recovery |

### 🔧 Fix Skills (After Check Commands)
| Skill | Purpose |
|-------|---------|
| `/fix-issues symbols` | Auto-fix symbol conflicts |
| `/fix-issues terms` | Auto-fix terminology inconsistencies |
| `/fix-issues refs` | Auto-fix broken references |
| `/fix-issues numbers` | Harmonize numerical values |
| `/fix-issues placement` | Move content to correct locations |
| `/fix-issues all` | Apply all safe fixes |
| `/fix-issues [type] --dry-run` | Preview changes without applying |

### 📚 RAG Library Maintenance (Self-Maintaining)
| Skill | Purpose |
|-------|---------|
| `/rag-maintain add [cat] "[pattern]" [src]` | Add new writing pattern |
| `/rag-maintain search [query]` | Search patterns with quality metrics |
| `/rag-maintain review` | Review recent misses, suggest additions |
| `/rag-maintain stats` | Show library coverage statistics |
| `/rag-maintain import [file]` | Extract patterns from well-written paper |

### Master Review Skill
| Skill | Purpose |
|-------|---------|
| `/review-paper-full MS` | Runs all 5 levels (manual version) |

### Individual Level Skills
| Skill | Level | Checks |
|-------|-------|--------|
| `/check-content-redundancy` | 0-1 | Result redundancy, example placement |
| `/check-paper-consistency` | 2 | Symbol conflicts, term uniformity |
| `/check-paper-flow` | 1-2 | Section transitions, claim consistency |
| `/check-figures-tables` | 4 | Figure format, legends, colors, table booktabs, captions |
| `/polish-paper` | 5 | Language only (AI words, voice, transitions) |

### Venue-Specific Style Skills
| Skill | Target Venue | Focus |
|-------|--------------|-------|
| `/check-ms-style` | Management Science | Managerial insights, service framing, prescriptions |
| `/check-msom-style` | M&SOM | Operational relevance, practical impact, 32-page limit, no footnotes |
| `/check-or-style` | Operations Research | Methodological contribution, proof rigor, optimality |
| `/check-ml-style` | NeurIPS/ICML/JMLR | Experiments, baselines, ablations, figures |

### Pre-Writing Skills (Use BEFORE writing)
| Skill | Purpose |
|-------|---------|
| `/define-paper-framing` | Lock core concepts, terminology, symbols BEFORE writing |
| `/init-paper-state [name]` | **Create full documentation ecosystem** for paper |

### Paper State Management (Prevent Information Loss)
| Skill | Purpose |
|-------|---------|
| `/init-paper-state [name]` | Create structured docs: symbols, results, figures, tables, changelog |
| `/update-paper-state [name]` | Sync all documentation after paper changes |
| `/track-review-comments add` | Parse and track reviewer comments systematically |
| `/track-review-comments status` | Show progress on addressing all comments |
| `/paper-overview create` | Build living documentation for paper state |

### Additional Consistency Skills
| Skill | Purpose |
|-------|---------|
| `/check-term-consistency` | Detect same concept with different words ("audit" vs "review") |
| `/check-content-placement` | Verify assumptions, examples, proofs in correct locations |

### Other Skills
| Skill | Purpose |
|-------|---------|
| `/verify-proof` | Math proof verification |
| `/proofread-references` | Citation checking |
| `/refine-theory` | Theory section refinement |

## Recommended Workflow

### For Journal Submission (e.g., Management Science)

```bash
# Step 1: Comprehensive review (runs all levels)
/review-paper-full MS

# Step 2: Fix issues found, then re-run specific checks
/check-ms-style           # If style issues remain
/check-paper-consistency  # If symbol conflicts remain
/check-content-redundancy # If redundancy issues remain

# Step 3: Only after levels 0-3 pass, do language polish
/polish-paper
```

### Quick Checks

```bash
# Check for specific issues:
/check-paper-consistency  # Symbol conflicts like b_k vs b(t)
/check-content-redundancy # π∝√g stated in 3 sections
/check-ms-style          # Arms too narrowly defined
```

## Key Insight

**"Converged" in `/polish-paper` only means language is clean.**

True submission readiness requires ALL levels to pass:
- Level 0: Content ✅
- Level 1: Structure ✅
- Level 2: Consistency ✅
- Level 3: Style ✅
- Level 4: Figures & Tables ✅
- Level 5: Language ✅

## Mapping Reviewer Comments to Skills

| Comment Type | Skill |
|--------------|-------|
| "Arms definition too narrow" | `/check-ms-style` |
| "Sections 5,6,7 repeat same result" | `/check-content-redundancy` |
| "Symbol b_k conflicts with b(t)" | `/check-paper-consistency` |
| "Too ML style, not MS style" | `/check-ms-style` |
| "Missing managerial insights" | `/check-ms-style` |
| "Not relevant to operations practice" | `/check-msom-style` |
| "Too much ML jargon for MSOM" | `/check-msom-style` |
| "Exceeds page limit" | `/check-msom-style` |
| "Paper has footnotes" | `/check-msom-style` |
| "Example should be near theorem" | `/check-content-redundancy` |
| "Figure legend obscures data" | `/check-figures-tables` |
| "Inconsistent figure styles" | `/check-figures-tables` |
| "Can't read axis labels" | `/check-figures-tables` |
| "Table should use booktabs" | `/check-figures-tables` |
| "Machine-generated feel" | `/polish-paper` (but fix levels 0-4 first!) |

---

## Paper State Documentation Ecosystem

**Problem**: Long context causes information loss → inconsistent changes → reviewer comments

**Solution**: Structured documentation that persists across sessions

### Directory Structure (12 Canonical Files)
```
docs/paper_state/[paper_name]/
├── overview.md           # Current state at a glance
├── symbols.md            # All notation (prevents b_k vs b(t) conflicts)
├── results.md            # All theorems (prevents redundant statements)
├── framing.md            # Locked concepts and terminology
├── changelog.md          # Track ALL modifications
├── cross_references.md   # All refs to results/numbers
├── dependencies.md       # Assumption → Theorem chains
├── abbreviations.md      # Acronym registry
├── figures_tables.md     # Figure/table registry, generation standards, quality audit
├── insights.md           # Key takeaways (for MS: managerial insights)
├── review_responses.md   # Reviewer comments and responses
└── consistency_log.md    # Check command history
```

**Templates**: All 12 templates in `.claude/commands/templates/paper_state/`

### Full Paper Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: SETUP (Before Writing)                                 │
│                                                                 │
│   /init-paper-state ms_judge     → Create documentation        │
│   /define-paper-framing MS       → Lock terminology            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 2: WRITING (Consult Docs)                                 │
│                                                                 │
│   [Write sections]               → Check symbols.md for notation│
│   [Add theorem]                  → Add to results.md           │
│   /update-paper-state            → Sync after changes          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 3: REVIEW (Before Submission)                             │
│                                                                 │
│   /review-paper-full MS          → Comprehensive 5-level review │
│   /check-paper-consistency       → Verify symbols.md matches    │
│   /check-term-consistency        → Verify framing.md matches    │
│   /polish-paper                  → Final language polish        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 4: REVISION (After Reviews)                               │
│                                                                 │
│   /track-review-comments add     → Parse all reviewer comments  │
│   /track-review-comments status  → Track progress (10/14 done)  │
│   [Make changes]                 → Update paper                 │
│   /update-paper-state            → Sync documentation           │
│   /track-review-comments export  → Generate response letter     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Benefits

| Without State Docs | With State Docs |
|--------------------|-----------------|
| "What symbol did I use for bias?" | Check `symbols.md` |
| "Did I already state this theorem?" | Check `results.md` |
| "What figure shows this result?" | Check `figures_tables.md` |
| "What changes did I make last session?" | Check `changelog.md` |
| "How many reviewer comments left?" | `/track-review-comments status` |
| Context lost between sessions | Documentation persists forever |

---

## Quick Start Guides

### New Paper
```bash
/paper-pipeline init my_paper MS    # Create docs + initial scan
# Write paper sections
/paper-pipeline review MS           # Full review when ready
```

### Continuing Work
```bash
/session-start my_paper             # Load context (2 min)
# Make changes
/paper-pipeline quick               # Verify consistency
/update-paper-state my_paper        # Sync docs before ending
```

### Got Reviewer Comments
```bash
/track-review-comments add [file]   # Parse all comments
/paper-pipeline revision            # Structured workflow
# Fix issues
/track-review-comments status       # Track progress
/track-review-comments export       # Generate response letter
```

### Before Submission
```bash
/paper-pipeline pre-submit MS       # Final checklist
# Fix any issues
/paper-pipeline pre-submit MS       # Verify 100%
# Submit!
```
