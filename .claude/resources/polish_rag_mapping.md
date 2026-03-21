# Polish RAG Mapping - Section → Reference Files

## Mandatory: Advisor Editing Rules

**Always load first** (applies to ALL sections):
- `guides/advisor_editing_rules.md` — Prof JD's transformation patterns (verb strengthening, compression, meta-discourse removal, hedging calibration, precision increase, paragraph merging)

## Section-Specific References

Before polishing each section, read the corresponding reference files from `.claude/writing_references/`:

| Section Being Polished | Files to Read |
|------------------------|---------------|
| **Introduction** | `sentences/introduction.md`, `sentences/contribution.md`, `sentences/motivation.md` |
| **Related Work** | `sentences/related_work.md`, `sentences/citation_patterns.md`, `phrases/transitions.md` |
| **Model/Problem Setup** | `sentences/problem_setup.md`, `sentences/or_applications.md` |
| **Algorithm/Method** | `sentences/algorithm_optimality.md`, `paragraphs/proof_structure.md` |
| **Theory/Analysis** | `phrases/hedging.md`, `paragraphs/proof_structure.md`, `paragraphs/section_restructure.md`, `sentences/control_theory.md` |
| **Experiments** | `sentences/dynamic_pricing.md`, `paragraphs/main_results.md` |
| **Conclusion** | `sentences/contribution.md`, `paragraphs/section_restructure.md` |

## Topic-Specific References

| Paper Topic | Additional Files |
|-------------|-----------------|
| LLM/AI | `sentences/llm_papers.md` |
| Bandits/BAI | `sentences/algorithm_optimality.md`, `sentences/online_learning.md` |
| Operations Research | `sentences/or_applications.md`, `sentences/revenue_management.md` |
| Pricing | `sentences/dynamic_pricing.md`, `sentences/choice_models.md` |
| Optimization | `sentences/robust_optimization.md`, `sentences/approximation_algorithms.md` |

## How to Use References

1. Read the relevant files: `Read .claude/writing_references/sentences/[file].md`
2. Each file contains patterns: `## [Category] Title` → `**Context**: When to use` → `> Template`
3. Match the style and structure when rewriting sentences
4. Prefer patterns from top venues (JMLR, NeurIPS, Management Science)

## RAG Miss Detection

When no good match found:
1. Log miss to `.claude/writing_references/_rag_log.md`
2. Warn: `RAG_MISS: No good pattern for "[query]"` → fallback to general rewrite
3. Suggest: `/rag-maintain add sentences/[category] "[pattern]" source`
