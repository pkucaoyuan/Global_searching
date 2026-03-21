# Academic Writing Guidance for OR/OM Papers

## 📖 Overview

**Purpose**: This guide provides comprehensive academic writing principles with emphasis on de-AI-ing and maintaining rigorous academic standards for operations research and machine learning papers.

**Core Philosophy**: Balance precision with clarity, maintain academic rigor while ensuring natural, readable prose.

**When to Consult This Guide**:
- When you need to improve writing quality and clarity
- When performing de-AI-ing modifications to remove AI-generated patterns
- When uncertain about how to phrase an argument or explanation
- When you want to understand academic writing theoretical foundations
- When reviewing drafts before submission

---

## 🏛️ Level 0: Transitions & Coherence (HIGHEST PRIORITY)

**These are the foundational checks that must pass BEFORE polishing language. They reflect the quality of thinking, not just writing.**

### Core Principle (Gopen & Swan)

> "The misplacement of old and new information is the No. 1 problem in American professional writing today."

Based on Gopen & Swan's "The Science of Scientific Writing" (1990):

1. **Topic Position (Sentence Beginning) → Old Information**: Place known information at the beginning to establish continuity with prior sentences.

2. **Stress Position (Sentence End) → New Information**: Place the key new information at the end, where readers naturally focus most attention.

3. **Subject-Verb Proximity**: Keep subjects and verbs close (≤7 words apart). Long interruptions cause cognitive overload.

### 0.1 Topic Position Diagnostic

Each sentence's opening should contain **old information** that links to previous context.

**How to check**: List the first 5-6 words of consecutive sentences. Do they share a common thread?

| Bad (focus shifts every sentence) | Good (consistent thread) |
|----------------------------------|--------------------------|
| "**Large earthquakes** along a fault..." | "**Large earthquakes** along a fault..." |
| "**The rates** at which plates move..." | "**These intervals** are approximately..." |
| "**Therefore... one** may expect..." | "**Therefore, recurrence** can be predicted..." |

**Fix**: If topic positions constantly introduce NEW concepts, restructure to place old/linking information first.

### 0.2 Stress Position Diagnostic

The most important NEW information should land at the sentence end (before the period).

| Bad (emphasis wasted) | Good (emphasis on key info) |
|----------------------|----------------------------|
| "The rate decreased significantly when **the temperature changed**." | "Changing the temperature caused a significant decrease in **the metabolic rate**." |
| "Our algorithm improves throughput, **which is significant**." | "Our algorithm achieves a key improvement: **30% higher throughput**." |

**Fix**: Move qualifiers/conditions to sentence beginning; push core findings to sentence end.

### 0.3 Logical Connector Diagnostic

Transitions between sentences should be EXPLICIT, not left for readers to infer.

**Missing Connector Symptoms**:
- Two consecutive sentences with no logical relationship word
- Reader must guess: "Is this contrast? Consequence? Example?"

**Connector Palette** (vary these, don't repeat):
| Relationship | Options |
|--------------|---------|
| Contrast | However, Yet, But, In contrast, On the other hand |
| Consequence | Therefore, Thus, Consequently, As a result, Hence |
| Addition | This [noun]..., Building on this..., [No connector if flow is clear] |
| Example | For instance, For example, Consider... |
| Concession | Although, While, Despite, Even though |

**Anti-Pattern**: "Furthermore... Moreover... Additionally..." chains signal AI-generated text.

### 0.4 Subject-Verb Proximity Diagnostic

**Principle**: Readers hold their breath between subject and verb. Long interruptions (>7 words) cause cognitive overload.

| Bad (23 words between S-V) | Good (S-V adjacent) |
|---------------------------|---------------------|
| "The smallest URF, a 207-nucleotide reading frame overlapping out of phase the NH2-terminal portion, **has been identified**..." | "The smallest URF **has been identified** as the animal equivalent..." |

**Fix**: Move long modifiers to separate sentences or to sentence beginning as participial phrases.

### 0.5 Paragraph Coherence Diagnostic

Each paragraph should tell ONE story about ONE subject.

**How to check**: Can you identify whose story this paragraph tells from the topic positions alone?

### 0.6 Itemize/Enumerate Overuse Diagnostic

Excessive bullet points fragment prose and signal AI-generated text.

**When to use lists**:
- ✅ Enumerating 4+ distinct items with parallel structure
- ✅ Step-by-step procedures or algorithms
- ✅ Comparison tables with multiple dimensions

**When to use prose instead**:
- ❌ 2-3 items → use inline enumeration: "first, ...; second, ...; third, ..."
- ❌ Items that need explanation → integrate into flowing paragraphs
- ❌ Consecutive sections of bullet points → convert to prose with transitions

**Transformation example**:
```
BAD (fragmented):
Our contributions are:
• We propose a benchmark.
• We evaluate 26 models.
• We show training improves performance.

GOOD (flowing):
We make three contributions. First, we construct OR-Debug-Bench,
a benchmark of 7,200 debugging episodes. Second, we evaluate 26
models spanning... Third, we demonstrate that domain-specific
training improves RR@5 by 9 percentage points.
```

### OR Paper Logical Flow Checklist

Before any language polishing, verify:

- ✅ **Old → New Information Flow**: Does each sentence begin with information that links to previous context?
- ✅ **Paragraph Topic Sentences**: Does the first sentence of each paragraph clearly state its purpose?
- ✅ **Mathematical Symbol Definitions**: Are all symbols defined when first introduced?
- ✅ **Proof Logic Chain**: Does every "therefore", "thus", "hence" have clear justification from prior statements?
- ✅ **Specific to General**: Do you introduce concepts with examples/special cases before stating general results?

### Quick Self-Check Methods

- **Paragraph Test**: Pick a random paragraph, cover the second half, and see if you can predict the general content from the first half.
- **Theorem Test**: Pick a random theorem: Is the statement self-contained (understandable without flipping back)?
- **Proof Test**: Pick a random proof step: Does it explain "why" this step is taken, not just "what" is done?

---

## 💬 Language Golden Principles

### 1. Verbs Carry Actions

The core action should be expressed by strong verbs.

**Avoid Zombie Nouns (Nominalizations)**: Don't hide actions in nouns.
- ❌ We *conducted an analysis* of the data.
- ✅ We *analyzed* the data.

**Common Zombie Nouns to Eliminate**:
- `perform an analysis` → `analyze`
- `conduct an investigation` → `investigate`
- `provide a description` → `describe`
- `make an assumption` → `assume`
- `reach a conclusion` → `conclude`

### 2. Use Concrete Language

Prefer concrete nouns and active verbs over abstraction and passive voice.

- ❌ *The implementation of the method was completed by the team.*
- ✅ *The team implemented the method.*

### 3. Embrace First Person

Use "We" confidently and appropriately, making arguments more direct and taking authorial responsibility.

- ❌ *It is believed that...*
- ✅ *We argue that...*

**Exception**: Passive voice is acceptable when the actor is unknown or irrelevant, or when focusing on the object.

---

## 🔬 Theory Paper Specific Principles

### 1. Clear Paragraph Functions

Each paragraph should serve one clear function:
- **Define problem**: "We consider a system where..."
- **State result**: "Theorem 1 shows that..."
- **Proof step**: "To prove this, we first establish..."
- **Discuss implications**: "This result implies that..."

### 2. Intuition Before Rigor

Give intuition first, then formal proof.

- ❌ Drop formulas directly: The optimal policy is π* = arg min...
- ✅ Explain intuition first: Intuitively, the optimal policy balances... Formally, π* = arg min...

### 3. Balance Math and Text

Mathematical expressions need textual explanation.

- ❌ Consecutive formulas without transition: Eq(1)... Eq(2)... Eq(3)...
- ✅ Use text to connect: From Eq(1), we derive Eq(2). This leads to Eq(3), which shows...

### 4. Concrete Examples Support Abstract Concepts

- ❌ Pure abstraction: "The threshold mechanism enables decoupling."
- ✅ With example: "The threshold mechanism enables decoupling. For instance, consider a system where..."

### 5. Show Key Calculation Steps

- Not all steps—just the non-trivial transformations
- Use "where", "by", "since" to explain each step
- Example: "By the law of large numbers, X_n/n → λ as n → ∞, where λ is the arrival rate."

---

## 🤖 De-AI-ing Writing Standards

**Apply these checks in order: Level 1 first (easiest to spot and fix), then Level 2, then Level 3.**

### Level 1: Must Delete or Modify — "AI Template Phrases"

❌ **Boosterism (Self-Promotion)**: Delete or replace
- `novel`, `significant`, `remarkable`, `comprehensive`, `groundbreaking`, `indispensable`
- **Fix**: Let results speak for themselves. State facts, not marketing claims.

❌ **Template Phrases**: Rewrite entirely
- `It is worth noting that...` → Delete the phrase, state directly
- `Importantly, ...` / `Crucially, ...` → Overused; state the importance naturally
- `This paper delves into...` / `This study explores...` → Say what the paper **does**
- `The key insight/observation is...` → Too repetitive; state the insight directly

❌ **Vague Verbs**: Replace with specific actions
- `leverages` → `uses`, `applies`, `employs`
- `utilizes` → `uses`
- `employs advanced techniques` → Name the techniques

### Level 2: Watch for "Zombie Nouns" and Passive Voice

❌ **Zombie Nouns (Nominalizations)**: Prefer verb forms
- `perform an analysis` → `analyze`
- `conduct an investigation` → `investigate`
- `provide a description` → `describe`
- `make an assumption` → `assume`
- `reach a conclusion` → `conclude`

❌ **Passive Voice**: Use active voice when possible
- `The data was analyzed...` → `We analyzed the data...`
- `It is shown that...` → `We show that...` or `Theorem 1 shows that...`
- `The algorithm is designed to...` → `We design the algorithm to...`

**Exception**: Passive voice is acceptable when the actor is unknown or irrelevant, or when focusing on the object.

### Level 3: Structural Issues

❌ **New Information Front-Loaded**: Don't introduce complex new concepts at sentence beginnings
- Reorder: Old information → New information

❌ **Subject-Verb Separation**: Check for long insertions between subject and verb
- **Bad**: The algorithm, which incorporates multiple heuristics and is designed to handle large-scale instances, performs well.
- **Good**: The algorithm performs well. It incorporates multiple heuristics and handles large-scale instances.

---

## ⚖️ Polish Philosophy: Balance Over Simplification

**Core Principle**: Don't oversimplify—maintain necessary details and examples. Focus on making transitions natural and the text easy to follow.

### ⚠️ Critical Warning: Avoid Over-Correction

De-AI-ing is NOT about:
- ❌ Removing complete narratives
- ❌ Cutting out necessary context
- ❌ Eliminating explanatory phrases that aid understanding
- ❌ Making the text sparse or telegraphic
- ❌ Removing all transition words

De-AI-ing IS about:
- ✅ Replacing formulaic phrases with natural language
- ✅ Smoothing awkward transitions
- ✅ Making the writing flow more naturally
- ✅ Maintaining completeness while improving readability
- ✅ Preserving the narrative arc

### What to Keep

- ✅ Specific examples and citations
- ✅ Technical details and innovations
- ✅ Numbered/bulleted lists when they aid comprehension
- ✅ Clear contrasts and comparisons with prior work
- ✅ Explanatory examples that illustrate concepts

### What to Improve

- ✅ Opening sentences should be engaging, not formulaic
- ✅ Transition phrases should flow naturally
- ✅ Use colons and dashes to connect explanations smoothly
- ✅ Replace stiff academic phrases with natural alternatives
- ✅ Ensure readers can follow the logical flow effortlessly

---

## 📄 Section-Specific Patterns

### Abstract

**Goals:**
- Be direct and concise
- Avoid superlatives and marketing language
- State contributions clearly
- Include all key results

**Pattern:**
```
[Problem statement in 1-2 sentences]
[Main approach/contributions in 2-3 sentences]
[Key results with specific bounds]
[Validation statement]
```

**Example transformation:**

❌ Before:
```
Dynamic pricing with resource constraints is a critical challenge
in online learning, requiring a delicate balance between exploring
unknown demand patterns and exploiting known information to maximize
revenue. We propose three tailored algorithms to address this problem...
```

✅ After:
```
Dynamic pricing with resource constraints requires balancing exploration
of unknown demand patterns against exploitation of known information to
maximize revenue. We develop three algorithms for this problem across
different information regimes...
```

### Introduction

**Opening paragraphs:**
- Start with the problem, not the field's growth
- Use natural language, not academic clichés
- Build logical flow between paragraphs

**Example:**

❌ Before:
```
Dynamic pricing is a classical problem in online learning and
decision-making. With the growth of e-commerce, there has been
an increasing focus on the development of efficient dynamic
pricing policies...
```

✅ After:
```
Dynamic pricing is a classical problem in online learning and
decision-making. The growth of e-commerce has spurred substantial
recent work on efficient pricing policies...
```

### Related Literature

**Structure:**
Each paragraph should have:
1. Natural opening (not "X is important")
2. Key prior work with specifics
3. Clear statement of how our work differs
4. Preserved technical details

**Good patterns:**

✅ Opening with context:
```
Leveraging offline information for online pricing has received
growing attention.
```

✅ Introducing contrasts:
```
Our work differs in several ways: [specific differences]
```

✅ Listing innovations naturally:
```
We extend their approach through three key innovations:
[innovation 1], [innovation 2], and [innovation 3].
```

✅ Using examples smoothly:
```
Works such as [Author1], [Author2], and [Author3] study...
```

**When to use numbered lists:**
- ✅ When enumerating technical innovations (e.g., "three key contributions")
- ✅ When comparing multiple dimensions (e.g., "differs in three ways")
- ✅ When the structure aids comprehension

**When NOT to use numbered lists:**
- ❌ For simple conjunctions (use commas instead)
- ❌ When flow would be more natural in prose

### Technical Sections

**Proof sketches and intuitions:**
- Explain the difficulty first
- Use specific examples when helpful
- Don't be afraid of informal language in intuition paragraphs
- Use colons and dashes for explanations

**Example:**

✅ Good:
```
The core difficulty is ensuring sufficient exploration. From (1.2),
the parameter estimation error scales inversely with λ_min(P^t), the
minimum eigenvalue of the design matrix. As shown in (1.3), this
eigenvalue grows with the variance of historical prices. If prices
cluster too tightly—always near the unconstrained optimum, for
example—then λ_min(P^t) remains small and estimation stagnates.
```

### Numerical Experiments

**Patterns:**
- State experimental setup clearly
- Report specific numbers (not just "substantial improvement")
- Use natural language to describe observations
- Connect back to theory

**Example:**

✅ Good:
```
With ρ=0.9, the regret decreases by approximately 21% compared to
the baseline, and even a moderately correlated surrogate (ρ=0.5)
provides 8% improvement. This confirms that correlation matters
more than absolute accuracy: the surrogate remains valuable despite
having bias in both α and B.
```

---

## 📖 Examples: Good vs Over-Corrected vs Just Right

### Example 1: Related Literature Opening

❌ **Over-corrected (TOO SPARSE):**
```
BwK has two approaches. First uses policy selection. Second uses Lagrangian dual.
```
*Problem: Lost all context, reads like telegraphic notes.*

⚠️ **Original AI-ish:**
```
Bandits with knapsacks is a critical challenge that has received significant
attention in recent years. Researchers have developed two main approaches.
The first approach, which has proven highly effective, selects...
```
*Problem: Too much filler, formulaic phrases.*

✅ **Just Right:**
```
The BwK framework, introduced by Agrawal (2016), addresses online
decision-making with resource constraints through two main approaches.
The first selects an optimal randomized policy from a finite policy class.
```
*Why it works: Natural flow, preserved context, no filler.*

### Example 2: Explaining Contributions

❌ **Over-corrected (LOSES NARRATIVE):**
```
We do three things: (1) boundary attraction, (2) learning, (3) informed prices.
```
*Problem: Readers don't understand what these mean or why they matter.*

⚠️ **Original AI-ish:**
```
We propose three tailored algorithms to address this critical problem
across varying levels of prior knowledge. Our first contribution is...
Our second contribution is... Our third contribution is...
```
*Problem: Repetitive "Our X contribution is" structure, too formal.*

✅ **Just Right:**
```
We develop three algorithms for this problem across different information regimes.
In the full-information setting, our boundary-attracted re-solve method achieves
O(log T) regret without requiring the non-degeneracy conditions assumed in prior
work. In the no-information setting, our online learning algorithm attains...
```
*Why it works: Complete narrative, clear context for each algorithm, natural enumeration.*

### Example 3: Technical Explanation

❌ **Over-corrected (REMOVES HELPFUL DETAIL):**
```
The estimation error scales inversely with λ_min(P^t).
```
*Problem: Too terse, readers don't understand why this matters.*

⚠️ **Original AI-ish:**
```
It is worth noting that the estimation error, which is a key quantity of
interest in our analysis, scales inversely with λ_min(P^t), which represents
the minimum eigenvalue of the design matrix P^t. This observation is crucial
for understanding the behavior of our algorithm...
```
*Problem: "It is worth noting", "which is a key quantity", excessive signposting.*

✅ **Just Right:**
```
The estimation error scales inversely with λ_min(P^t), the minimum eigenvalue
of the design matrix. As shown in (1.3), this eigenvalue grows with the variance
of historical prices. If prices cluster too tightly—always near the unconstrained
optimum, for example—then λ_min(P^t) remains small and estimation stagnates.
```
*Why it works: Kept the explanation, used natural flow with colons and dashes, removed filler.*

**Key Insight**: The "just right" version maintains complete information and narrative flow while eliminating formulaic AI patterns.

---

## 🚨 Common Anti-Patterns

### AI-Generated Language Markers

**Avoid these patterns:**
- "is a critical challenge"
- "has received increasing attention"
- "there has been an increasing focus"
- "We propose three tailored algorithms"
- "Moreover", "Furthermore" as paragraph starters
- "This work advances X by offering..."
- "Looking ahead" / "In summary"

**Use instead:**
- Direct statements: "Dynamic pricing requires..."
- Natural transitions: "has spurred substantial work"
- Simple enumeration: "We develop three algorithms"
- Integrated transitions without signposting
- Natural conjunctions within sentences

### Overly Formal Structures

**Avoid:**
```
Our work connects to four main research streams:
(i) topic A; (ii) topic B; (iii) topic C; and (iv) topic D.
```

**Use instead:**
```
Our work connects to four research streams:
topic A, topic B, topic C, and topic D.
```

---

## 🔄 Transitions & Coherence Techniques

### The Old→New Principle

Every sentence should flow from OLD information (what reader already knows) to NEW information (what you want to emphasize).

**Pattern**: [Old/Linking Info] → [Core Content] → [New/Emphasized Info]

**Example Flow**:
```
LLMs struggle with solver debugging. [introduces topic]
This debugging challenge... [OLD: links back to "debugging"]
The core difficulty lies in... [OLD: "difficulty" links to "struggle"]
To address this difficulty, we propose... [OLD: "difficulty" links backward]
```

### Connector Palette (Avoid Repetition)

| Relationship | Natural Options | AI-Pattern to AVOID |
|--------------|-----------------|---------------------|
| **Contrast** | However, Yet, But, In contrast | Moreover, Furthermore |
| **Consequence** | Therefore, Thus, Hence, As a result | Additionally, In addition |
| **Addition** | This [noun]..., Building on this... | It is worth noting that |
| **Example** | For instance, Consider..., Take... | Interestingly |
| **Concession** | Although, While, Despite | Importantly |

**Anti-Pattern**: "Furthermore... Moreover... Additionally..." chains signal AI-generated text.

### When NO Connector is Needed

If the Old→New flow is clear, omit explicit connectors:
```
The algorithm terminates in O(n²) iterations.
Each iteration requires O(n) operations.  [No connector needed—"iteration" is the link]
```

### Punctuation as Transitions

**Colons for elaboration:**
```
We observe a clear phase transition: when ε^0 is very small,
the regret grows as O((ε^0)² T).
```

**Dashes for examples/asides:**
```
If prices cluster too tightly—always near the unconstrained
optimum, for example—then estimation stagnates.
```

**Semicolons for parallel ideas:**
```
The first approach uses policy selection; the second uses Lagrangian duality.
```

### Paragraph-Level Coherence

Each paragraph should tell ONE story about ONE subject.

**Diagnostic**: Read only the first sentence of each paragraph in a section. Do they form a coherent outline?

**Good paragraph structure**:
1. **Topic sentence**: States what this paragraph is about (links to previous)
2. **Development**: Evidence, examples, elaboration
3. **Closure**: Optional transition to next topic

---

## ⚠️ Common Writing Problems and Solutions

### Problem 1: Paragraph Lacks Topic Sentence

**Symptom**: Reader can't immediately understand what the paragraph is about.

**Solution**: Start each paragraph with a sentence that summarizes its main point.

### Problem 2: Math Symbols Overwhelming Text

**Symptom**: Too many symbols, not enough explanation.

**Solution**:
- Introduce symbols with text explanation
- Add intuitive description after complex formulas
- Use "where" clauses to define notation inline

### Problem 3: Proof Steps Lack Justification

**Symptom**: Logical jumps in proofs without explanation.

**Solution**:
- Add "by [lemma/theorem]" to justify steps
- Supplement with "since", "because", "as" explanations
- In appendix, provide detailed derivations

### Problem 4: Citations Inappropriate or Excessive

**Symptom**: Original statements have citations, or obvious facts are cited.

**Solution**:
- No citation needed for: your own original work, common knowledge
- Citation required for: key technical results, prior work, established methods

---

## 📚 Recommended Reading

### Academic Writing

- **Gopen, George D., and Judith A. Swan**. "The Science of Scientific Writing." *American Scientist* 78.6 (1990): 550-558.
  - Foundation for Level 0 logical flow principles

- **Sword, Helen**. *Stylish Academic Writing*. Harvard University Press, 2012.
  - Guidance on avoiding zombie nouns and passive voice

- **McEnerney, Larry**. "The Craft of Writing Effectively." YouTube lecture, University of Chicago.
  - Understanding what academic writing should accomplish

### OR/OM Writing Guides

- Check target venue's author guidelines
- Review highly-cited papers in your area for style patterns
- Study how top researchers structure their arguments

---

## 📝 Final Note

**The goal is natural academic writing, not oversimplification.**

When in doubt, ask:
- "Would a colleague write this sentence this way?"
- "Does this help the reader understand, or is it just filler?"
- "Are the transitions smooth, or do they feel forced?"
- "Have I explained the 'why', not just the 'what'?"

**Good academic writing is clear, direct, and flows naturally—but it remains technically precise and appropriately detailed.**

**Remember**: Quality of writing reflects quality of thinking. Master the Level 0 checks first—they matter most.

---

**Document Version**: v2.1 (Enhanced transitions & coherence)
**Created**: 2025-11-30
**Updated**: 2026-01-27
**Maintenance Status**: Active maintenance
**Applicable Projects**: Academic paper writing (OR/OM focus)
