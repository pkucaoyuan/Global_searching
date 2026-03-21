# Properties List Paragraph Templates

Paragraph patterns for stating desirable properties and features.

---

## [Enumerated Properties] We develop [objects] that possess the following properties

**Source**: Howard et al. - "Time-uniform confidence sequences" (Annals of Statistics 2021)
**Context**: Formally listing key properties of proposed method

> We develop confidence sequences that possess the following properties:
>
> (P1) **Nonasymptotic and nonparametric**: our confidence sequences offer coverage guarantees for all sample sizes, without exact distributional assumptions or asymptotic approximations.
>
> (P2) **Unbounded sample size**: our methods do not require a final sample size to be chosen ahead of time.
>
> (P3) **Arbitrary stopping rules**: we make no assumptions on the stopping rule used by an experimenter.
>
> (P4) **Asymptotically zero width**: the interval widths shrink towards zero at a $1/\sqrt{t}$ rate.

**Structure Pattern**:
1. Introduce: "We develop [X] that possess the following properties:"
2. Each property: (P#) **Bold title**: explanation
3. Properties should be independent and clearly stated

**Tags**: #properties #enumeration #contribution #howard

---

## [Consequences of Properties] These properties give us [guarantees] and [scope]

**Source**: Howard et al. - Annals of Statistics 2021
**Context**: Explaining implications of properties

> These properties give us strong guarantees and broad applicability. An experimenter may always choose to gather more samples, and may stop at any time according to any rule---the resulting inferential guarantees hold under the stated assumptions without any approximations.

**Tags**: #properties #implications #benefits #howard

---

## [Acknowledging Tradeoffs] Of course, this flexibility comes with a cost

**Source**: Howard et al. - Annals of Statistics 2021
**Context**: Honestly discussing limitations

> Of course, this flexibility comes with a cost: our intervals are wider than those that rely on asymptotics or make stronger assumptions, for example, a known stopping rule.

**Variants**:
- Naturally, these guarantees come at a price:...
- This generality is not free:...
- However, this approach requires...

**Tags**: #tradeoff #honest #limitation #howard

---

## [Surprising Cost Assessment] It is perhaps surprising that [desirable properties] come at a numerical cost of [small amount]

**Source**: Howard et al. - Annals of Statistics 2021
**Context**: Highlighting that cost is smaller than expected

> It is perhaps surprising that these four properties come at a numerical cost of less than doubling the fixed-sample, asymptotic interval width---the discrete mixture bound stays within a factor of two of the fixed-sample CLT bounds over five orders of magnitude in time.

**Pattern**:
1. "It is perhaps surprising that [strong guarantee]"
2. "comes at a numerical cost of [quantified cost]"
3. Concrete comparison with benchmark

**Tags**: #surprising #cost #quantitative #howard
