# Paper Roadmap Paragraph Templates

Paragraph patterns for describing paper structure and contributions.

---

## [Content Overview] The improvements of this paper mainly concern...

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Section outlining paper contributions

> The improvements of this paper mainly concern the fixed-confidence setting, which will be considered in the next three Sections. We first propose in Section 2 a distribution-dependent lower bound on $\kappa_C(\nu)$ that holds for $m>1$ and for general classes of bandit models (Theorem X). This information-theoretic lower bound permits to interpret the quantity $H(\nu)$ as a subgaussian approximation.
>
> Theorem Y in Section Z proposes a tighter lower bound for general classes of two-armed bandit models, as well as a lower bound on the sample complexity of $\delta$-PAC algorithms using uniform sampling. In Section W we propose, for Gaussian bandits with known variances, an algorithm exactly matching this bound.

**Structure Pattern**:
1. State main focus area
2. For each section: "We [verb] in Section N a [contribution] (Theorem X)"
3. Explain significance of each contribution
4. Mention matching algorithms if applicable

**Tags**: #organization #roadmap #structure #kaufmann

---

## [Two-Ingredient Pattern] To achieve [goal], two ingredients are needed

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Explaining what's needed for a complete result

> To achieve this, two ingredients are needed: a lower bound on the sample complexity of any $\delta$-PAC algorithm and a $\delta$-PAC strategy whose sample complexity attains the lower bound (often referred to as a 'matching' strategy).

**Generic Pattern**:
> To [achieve goal], [N] ingredients are needed:
> - [First component] that [property]
> - [Second component] that [property]
> - ...

**Tags**: #organization #methodology #structure #kaufmann

---

## [Heuristic Interpretation] Heuristically, on the one hand... On the other hand...

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Providing intuition for formal definitions

> Heuristically, on the one hand for a given bandit model $\nu$, and a small value of $\delta$, a fixed-confidence optimal strategy needs an average number of samples of order $\kappa_C(\nu) \log(1/\delta)$ to identify the $m$ best arms with probability at least $1-\delta$. On the other hand, for large values of $t$ the probability of error of a fixed-budget optimal strategy is of order $\exp(-\kappa_B(\nu) t)$.

**Pattern**:
1. Signal informal explanation: "Heuristically"
2. "On the one hand" - first interpretation
3. "On the other hand" - contrasting interpretation
4. Connect back to formal definitions

**Tags**: #exposition #intuition #contrast #kaufmann

---

## [Classification of Results] A particular class of [objects] will be considered

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Narrowing focus to specific cases

> A particular class of algorithms will be considered in the following: those using a uniform sampling strategy, that sample the arms in a round-robin fashion. Whereas it is well known that when $K>2$ uniform sampling is not desirable, it will prove efficient in some examples of two-armed bandits.

**Pattern**:
1. Introduce the special class
2. Acknowledge limitation: "Whereas it is well known that [limitation]"
3. Justify focus: "it will prove [property] in [specific setting]"

**Tags**: #organization #scope #justification #kaufmann

---

## [Connection to Classical Theory] Classical [field] theory provides a first element of comparison

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Connecting to established theory

> Classical sequential testing theory provides a first element of comparison between the fixed-budget and fixed-confidence settings, in the simpler case of fully specified alternatives. Consider for instance the case where [specific setup]...

**Pattern**:
1. Cite classical theory as baseline
2. Specify the simplified setting
3. Walk through the classical result
4. Transition to how your work differs

**Tags**: #exposition #classical #comparison #kaufmann
