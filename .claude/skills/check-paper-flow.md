# Check Paper Flow

Comprehensive review of academic paper for logical flow, contradictions, and redundancies.

## Description

This skill performs a thorough review of the paper to identify:
- **Flow issues**: Unnatural transitions between sections/paragraphs
- **Contradictions**: Inconsistent claims, numbers, or statements
- **Redundancies**: Repeated content, overlapping explanations, or duplicate claims

## Usage

```
/check-paper-flow [section]
```

## Parameters

- `section` (optional): Specific section to check (e.g., `introduction`, `training`). If omitted, checks entire paper.

## Procedure

### Step 1: Read All Main Sections

Read all `.tex` files in `paper/sections/` in order:
1. `abstract.tex`
2. `introduction.tex`
3. `setup.tex`
4. `construction.tex`
5. `training.tex`
6. `experiments.tex`
7. `related.tex`
8. `conclusion.tex`

### Step 2: Analyze Flow and Transitions

For each section boundary, check:
- Does the ending of section N naturally lead to section N+1?
- Are there clear topic sentences at paragraph beginnings?
- Is the logical progression clear (problem → method → results)?
- Are forward/backward references consistent?

### Step 3: Check for Contradictions

Compare claims across sections:
- **Numbers**: Dataset sizes, model counts, performance metrics
- **Methods**: Training procedures, reward weights, hyperparameters
- **Claims**: Contributions, findings, comparisons
- **Terminology**: Consistent use of terms (e.g., RR@5 vs RR@k)

Common contradiction patterns:
- Abstract claims vs. Experiments results
- Introduction contributions vs. Conclusion summary
- Methods description vs. Appendix details
- Table values vs. in-text citations

### Step 4: Identify Redundancies

Check for:
- **Repeated definitions**: Same concept explained multiple times
- **Duplicate claims**: Same contribution stated in multiple sections
- **Overlapping content**: Similar paragraphs in different sections
- **Redundant citations**: Same work cited with different context unnecessarily

### Step 5: Generate Report

Output format:

```markdown
## Paper Flow Analysis Report

### 1. Flow Issues
| Location | Issue | Suggestion |
|----------|-------|------------|
| intro→setup | Abrupt transition | Add bridging sentence |

### 2. Contradictions Found
| Section A | Section B | Discrepancy |
|-----------|-----------|-------------|
| abstract:L3 | experiments:L45 | "5,000" vs "4,800" |

### 3. Redundancies
| Location 1 | Location 2 | Overlap |
|------------|------------|---------|
| intro:L20 | related:L15 | Same definition of IIS |

### 4. Summary
- Flow score: X/10
- Contradictions: N found
- Redundancies: N found
```

## Checklist Items

### Flow Checklist
- [ ] Abstract → Introduction: smooth transition
- [ ] Introduction → Setup: clear motivation
- [ ] Setup → Construction: logical progression
- [ ] Construction → Training: method connection
- [ ] Training → Experiments: evaluation setup
- [ ] Experiments → Related: positioning
- [ ] Related → Conclusion: synthesis

### Contradiction Checklist
- [ ] Dataset sizes consistent (abstract, setup, experiments)
- [ ] Model counts consistent (abstract, experiments, appendix)
- [ ] Performance numbers match (text vs tables)
- [ ] Reward weights consistent (training, appendix)
- [ ] Contribution claims aligned (intro, conclusion)

### Redundancy Checklist
- [ ] IIS definition appears once (or with clear purpose)
- [ ] Benchmark descriptions not duplicated
- [ ] Training pipeline not over-explained
- [ ] Related work not repeated in introduction

## Output

Generates a comprehensive report with:
1. Specific line references for each issue
2. Severity rating (critical/moderate/minor)
3. Concrete suggestions for fixes
