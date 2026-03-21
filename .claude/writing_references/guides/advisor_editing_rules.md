# Advisor Editing Rules

Transformation patterns extracted from Prof JD's line edits on §1–§4 (Feb 2026). These are **before → after** rules grounded in actual edits—not generic advice. Apply these as the highest-priority polish pass.

Source: `draft/sections/01_introduction.tex`, `02_model.tex`, `03_single_class.tex`, `04_multi_class.tex`, `related_work.tex`

---

## Rule 1: Verb Strengthening

Replace weak/vague verbs with precise, assertive ones. Prefer present tense over progressive aspect.

| Before | After | Pattern |
|--------|-------|---------|
| are increasingly deployed as | now serve as | progressive → present tense |
| has consequently emerged as a primary bottleneck in | consequently constrains | zombie noun ("bottleneck") → verb |
| may be forced to terminate | must terminate | weak modal → strong modal |
| This cache enables efficient attention computation | This cache lets the model attend efficiently | nominalization → verb + adverb |
| has a structural explanation | reflects the synchronization structure | "has [noun]" → specific verb |
| admits a constant-coefficient limit | converge to a constant-coefficient limit | vague "admits" → precise dynamics verb |
| Stability is governed by the roots | The roots... govern stability | passive → active (flip subject) |
| Direct computation confirms | Direct computation shows | less emphatic where certainty is implicit |
| Our contribution lies in characterizing | We characterize | meta-statement → direct claim |

**Diagnostic**: Search for `has`, `is`, `are` + noun. Can the noun become a verb?

---

## Rule 2: Sentence Compression

Cut words that add no information. Every word must earn its place.

| Before | After | Words saved |
|--------|-------|-------------|
| that is difficult to preempt or migrate without incurring recomputation costs | that cannot be preempted or migrated without recomputation | 3 |
| Memory has consequently emerged as a primary bottleneck in modern LLM serving systems | Memory consequently constrains modern LLM serving systems | 5 |
| To develop a more fundamental understanding of these dynamics | To analyze these dynamics formally | 5 |
| qualitatively different phenomena from the congestion studied in fluid approximations | phenomena absent from fluid approximations | 6 |
| distill actionable design implications for practitioners | provide practical design guidance for practitioners | 1 + clarity |
| The phenomena we identify arise from first principles | These phenomena arise from first principles | 2 (avoid "we identify" repetition) |
| precipitate throughput collapse | collapse throughput | 1 (verb form more direct) |
| The key tool is the *spread function* | The main tool is the *spread function* | 0 (but "main" is less AI-sounding) |

**Diagnostic**: Can any phrase be cut while preserving meaning? Look for `incurring`, `inducing`, `resulting in` + noun chains.

---

## Rule 3: Paragraph Merging

Two short paragraphs on the same logical point → merge into one. Short paragraphs in technical writing signal incomplete thought development.

**Example**: Two separate paragraphs about (a) nonstationary capacity consumption and (b) eviction not being an artifact were merged because both establish the same point: service itself creates capacity pressure.

**Before** (2 paragraphs):
> ...a request that is feasible at the time of admission may become infeasible solely because it continues to run.
>
> Eviction is not merely an implementation artifact or rare failure mode. Under sustained load, it becomes unavoidable...

**After** (1 paragraph):
> ...a request that fits in memory at admission may become infeasible solely because it continues to run. In this context, eviction is not merely an implementation artifact or rare failure mode. Under sustained load...

**Diagnostic**: If two consecutive paragraphs share the same subject, merge unless the second introduces a genuinely new perspective.

---

## Rule 4: Meta-Discourse Removal

Cut "throat-clearing" phrases that announce what you're about to say instead of saying it.

| Before | After |
|--------|-------|
| The eviction rule deserves comment. A request at stage $j$... | Eviction ordering affects wasted computation. A request at stage $j$... |
| four-step protocol of Section~\ref{...} | protocol of Section~\ref{...} |
| The key asymmetry is the weight gradient | The driving asymmetry is the weight gradient |
| a concrete reading | a direct interpretation |
| Figure~X confirms this: starting from... | Figure~X confirms: starting from... |

**Diagnostic**: Search for "deserves comment", "we note that", "it is worth observing", "we now", "it should be noted". Replace with the content itself.

---

## Rule 5: Hedging Calibration

Strengthen hedges where the math proves the claim. Weaken hedges where the claim exceeds the evidence.

### Strengthen (math proves it)
| Before | After |
|--------|-------|
| may be forced to terminate | must terminate |
| can thus synchronize future memory peaks | can synchronize future memory peaks |

### Weaken (claim needs evidence)
| Before | After |
|--------|-------|
| low compute utilization despite memory pressure | low compute utilization despite memory pressure [CITE] |
| one might hope, could break the synchronization | might break the synchronization |

**Diagnostic**: If the preceding sentence is a theorem or proof, strengthen. If it's an empirical observation, add citation or qualify.

---

## Rule 6: Precision Increase

Replace vague terms with technically precise ones—but only when the precision adds information.

| Before | After | Why |
|--------|-------|-----|
| in heterogeneous systems | in systems with sufficient job size heterogeneity | specifies *what* is heterogeneous |
| homogeneous workloads | homogeneous job sizes | "workloads" is ambiguous (could mean arrival rates) |
| A naive admission policy | A naive myopic admission policy | "myopic" is the technical term for ignoring future |
| the resulting framework departs from | the resulting framework departs from classical queues with fixed capacity consumption | specifies what it departs *from* |

**Diagnostic**: Can a reviewer ask "heterogeneous in what sense?" If yes, specify.

---

## Rule 7: Subsection Title Tightening

Titles should name the content, not describe the content.

| Before | After |
|--------|-------|
| State Dynamics and the Four-Step Protocol | State Dynamics |
| Eviction versus Classical Rejection | (unchanged — already descriptive) |

**Rule**: If the subtitle after "and" is already implied by the section content, cut it.

---

## Rule 8: Related Work Compression

In related work, cut "our contribution lies in" constructions. Instead, use a short clause.

| Before | After |
|--------|-------|
| These works schedule within the feasible region; our contribution lies in characterizing boundary behavior | These works schedule within the feasible region; we characterize boundary behavior |
| we characterize discrete-time limit cycle structure and the role of coprimality in determining stability—qualitatively different phenomena from the congestion studied in fluid approximations | we characterize discrete-time limit cycles and the role of coprimality in determining stability—phenomena absent from fluid approximations |

---

## Rule 9: Opening Verb Modernization

Paper openings should use present tense, not progressive or past tense.

| Before | After | Signal |
|--------|-------|--------|
| LLMs are increasingly deployed as | LLMs now serve as | "increasingly" is vague temporal |
| has been studied since | (unchanged — historical context is fine in past) | |
| has emerged as | constrains / is | "has emerged" is progressive-past → present |

---

## Rule 10: Content Expansion Triggers

Prof expanded continuous batching explanation significantly. The pattern: when the *mechanism that causes the problem* is not explained clearly enough for a non-expert reader, expand with a WHY paragraph:
- WHY does batching exist? (GPU parallelism)
- WHY does batching create memory pressure? (multiple KV caches coexist)
- WHAT happens without batching? (no memory issue, but terrible throughput)

**Diagnostic**: For each key mechanism, check: does the text explain WHY it exists, not just WHAT it does?

---

## Application Priority

When polishing, apply these rules in order:
1. **Rule 1** (Verb strengthening) — highest signal-to-noise improvement
2. **Rule 2** (Compression) — tightest prose
3. **Rule 4** (Meta-discourse removal) — cuts throat-clearing
4. **Rule 6** (Precision) — adds technical clarity
5. **Rule 5** (Hedging calibration) — calibrates confidence
6. **Rule 3** (Paragraph merging) — structural improvement
7. **Rule 7–9** (Titles, related work, openings) — finishing touches
8. **Rule 10** (Expansion triggers) — only when mechanism is under-explained
