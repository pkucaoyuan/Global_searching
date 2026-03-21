# Polish Checklist - Detailed Reference

## Level 0: Transitions & Coherence (HIGHEST PRIORITY)

**Core Principle (Gopen & Swan)**: "The misplacement of old and new information is the No. 1 problem in American professional writing today."

### 0.1 Topic Position Check (Sentence Beginning)

Each sentence's opening should contain **old information** that links to previous context.

**Diagnostic**: List the first 5-6 words of consecutive sentences. Do they share a common thread?

| Bad (focus shifts every sentence) | Good (consistent thread) |
|----------------------------------|--------------------------|
| "**Large earthquakes** along a given fault..." | "**Large earthquakes** along a given fault..." |
| "**The rates** at which tectonic plates move..." | "**These intervals** are approximately uniform..." |
| "**Therefore... one** may expect..." | "**Therefore, recurrence** can be predicted..." |
| "**subsequent mainshocks** have different..." | "**However, recurrence times** may vary..." |

**Action**: If topic positions constantly introduce NEW concepts, restructure to place old/linking information first.

### 0.2 Stress Position Check (Sentence End)

The most important NEW information should land at the sentence end (before the period).

| Bad (emphasis wasted) | Good (emphasis on key info) |
|----------------------|----------------------------|
| "The metabolic rate decreased significantly when **the temperature was lowered**." | "Lowering the temperature caused a significant decrease in **the metabolic rate**." |
| "Our algorithm improves throughput, **which is significant**." | "Our algorithm achieves a key improvement: **30% higher throughput**." |

**Action**: Move qualifiers/conditions to sentence beginning; push core findings to sentence end.

### 0.3 Logical Connector Check

Transitions between sentences should be EXPLICIT, not left for readers to infer.

**Connector Palette** (vary these, don't repeat):

| Relationship | Options |
|--------------|---------|
| Contrast | However, Yet, But, In contrast, On the other hand |
| Consequence | Therefore, Thus, Consequently, As a result, Hence |
| Addition | This [noun]..., Building on this..., [No connector needed if flow is clear] |
| Example | For instance, For example, Consider... |
| Concession | Although, While, Despite, Even though |

**Anti-Pattern**: "Furthermore... Moreover... Additionally..." chains signal AI-generated text.

### 0.4 Itemize/Enumerate Overuse Check

**When to use lists**: 4+ distinct items with parallel structure; step-by-step procedures; comparison tables.

**When to use prose**: 2-3 items → inline enumeration; items needing explanation → flowing paragraphs.

```latex
% BAD: Fragmented bullet points
Our contributions are:
\begin{itemize}
    \item We propose a benchmark.
    \item We evaluate 26 models.
    \item We show training improves performance.
\end{itemize}

% GOOD: Flowing prose
We make three contributions. First, we construct OR-Debug-Bench,
a benchmark of 7,200 debugging episodes. Second, we evaluate 26
models spanning... Third, we demonstrate that domain-specific
training improves RR@5 by 9 percentage points.
```

### 0.5 Paragraph Coherence Check

Each paragraph should tell ONE story about ONE subject.

**Diagnostic**: Can you identify whose story this paragraph tells from the topic positions alone?

---

## Level 1: Advisor-Pattern Edits (Prof JD Rules)

**Source**: `writing_references/guides/advisor_editing_rules.md` — must be loaded before polishing.

These rules have the highest return on quality per edit. Apply before lower levels.

### 1.1 Verb Strengthening
Search for `has`, `is`, `are` + abstract noun. Convert noun→verb.

| Before | After |
|--------|-------|
| has emerged as a bottleneck | constrains |
| enables efficient computation | lets X compute efficiently |
| has a structural explanation | reflects the structure |
| our contribution lies in | we [verb] |

### 1.2 Sentence Compression
Every word must earn its place. Cut `incurring`, `inducing`, `resulting in` + noun chains.

| Before | After |
|--------|-------|
| without incurring recomputation costs | without recomputation |
| to develop a more fundamental understanding of | to analyze formally |
| qualitatively different phenomena from X | phenomena absent from X |

### 1.3 Meta-Discourse Removal
Cut phrases that announce content instead of stating it.

| Cut | Replace with |
|-----|-------------|
| The eviction rule deserves comment | Eviction ordering affects... |
| it is worth noting that | [delete, state directly] |
| a concrete reading | a direct interpretation |
| The key [noun] is | The/A [adjective] [noun] is |

### 1.4 Hedging Calibration
Strengthen where math proves; weaken where evidence is thin.

| Before | After | Why |
|--------|-------|-----|
| may be forced to | must | theorem proves it |
| widely observed in practice | [CITE] | needs evidence |

### 1.5 Precision Increase
Replace vague terms only when ambiguity exists.

| Before | After |
|--------|-------|
| heterogeneous systems | systems with job size heterogeneity |
| naive admission policy | naive myopic admission policy |
| homogeneous workloads | homogeneous job sizes |

### 1.6 Paragraph Merging
Two consecutive paragraphs on the same logical point → merge. Short paragraphs in technical writing signal incomplete thought.

---

## Level 2: Subject-Verb Proximity

**Principle**: Readers hold their breath between subject and verb. Long interruptions (>7 words) cause cognitive overload.

| Bad (23 words between S and V) | Good (S-V adjacent) |
|-------------------------------|---------------------|
| "The smallest of the URF's (URFA6L), a 207-nucleotide reading frame overlapping out of phase the NH2-terminal portion of the ATPase subunit 6 gene, **has been identified**..." | "The smallest of the URF's (URFA6L) **has been identified** as the animal equivalent..." |

**Action**: Move long modifiers to separate sentences or to sentence beginning as participial phrases.

---

## Level 3-6: Quick Reference Tables

**Forbidden → Replacement (AI Words):**

| Forbidden | Replacement |
|-----------|-------------|
| significantly | specific number (+9.1%, 3x faster) |
| remarkably, dramatically | delete or state fact |
| novel, comprehensive | delete or specify (26 models, 7,200 samples) |
| state-of-the-art | "among compared models" |
| plays a crucial role | explain the specific role |
| leverages, utilizes | uses, applies |
| It is worth noting | delete, state directly |
| In order to | To |
| Due to the fact that | Because |

**Zombie Noun → Verb Revival:**

| Zombie Noun | Revived Verb |
|-------------|--------------|
| implementation of | implementing |
| utilization of | using |
| investigation of | investigating |
| determination of | determining |
| conduct an analysis | analyze |
