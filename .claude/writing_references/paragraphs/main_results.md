# Main Results Paragraph Templates

Paragraph patterns for presenting main results in OR/Theory papers.

---

## [Regret Parameterization] We seek [bounds] that are [property] in [parameter], whereas in analyzing [standard methods] one typically expresses [bounds] as [standard form]

**Source**: Badanidiyuru et al. - "Bandits with Knapsacks" (JACM 2018)
**Context**: Explaining why different parameterization is needed

> We seek regret bounds that are sublinear in OPT, whereas in analyzing MAB algorithms one typically expresses regret bounds as a sublinear function of the time horizon $T$. This is because a regret guarantee of the form $o(T)$ may be unacceptably weak for the BwK problem because supply limits prevent the optimal policy from achieving a reward close to $T$.

**Pattern**:
1. State what you seek
2. Contrast with standard approach
3. Explain why standard approach is insufficient

**Tags**: #results #parameterization #regret #bwk #or

---

## [Illustrative Example of Weakness] An illustrative example is [application]: [concrete numbers show problem]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Showing why naive approach fails

> An illustrative example is the dynamic pricing problem with supply $B \ll T$: the seller can only sell $B$ items, each at a price of at most 1, so bounding the regret by any number greater than $B$ is worthless.

**Tags**: #example #weakness #parameterization #bwk #or

---

## [Algorithm Result] We present an algorithm, called [name], whose [metric] is [property]. More precisely, our algorithm's [metric] is [formula]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Stating main algorithmic result

> We present an algorithm, called PD-BwK, whose regret is sublinear in OPT as both OPT and B tend to infinity. More precisely, denoting the number of arms by K, our algorithm's regret is $\tilde{O}(\sqrt{K \cdot OPT} + OPT \sqrt{K/B})$.

**Tags**: #algorithm #result #regret #bwk #or

---

## [Recovering Standard Result] Note that without [constraint], i.e., setting [parameter], we recover [standard result], which is optimal

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Showing result reduces to known optimal in special case

> Note that without resource constraints, i.e., setting $B=T$, we recover regret $\tilde{O}(\sqrt{K \cdot OPT})$, which is optimal up to log factors.

**Tags**: #special_case #reduction #optimality #bwk #or

---

## [Scaling Property] In fact, we prove a slightly stronger [bound] which has an optimal scaling property: if [parameters] are increased by factor $\alpha$, then [bound] scales as [function of $\alpha$]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Stating scaling behavior of bounds

> In fact, we prove a slightly stronger regret bound which has an optimal scaling property: if all budget constraints, including the time horizon, are increased by the factor of $\alpha$, then the regret bound scales as $\sqrt{\alpha}$.

**Tags**: #scaling #property #bound #bwk #or

---

## [Computational Efficiency] The algorithm is computationally efficient, in a strong sense: with machine word size of [bits], the per-round running time is [complexity]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Stating computational complexity

> The algorithm is computationally efficient, in a strong sense: with machine word size of $\log T$ bits or more, the per-round running time is $O(K \cdot d)$.

**Tags**: #efficiency #computation #complexity #bwk #or

---

## [Matching Lower Bound] We provide a matching lower bound: we prove that [bound] is optimal up to [factors]; moreover, this holds for any given tuple of parameters

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Stating information-theoretic optimality

> We provide a matching lower bound: we prove that the regret bound is optimal up to polylogarithmic factors; moreover, this holds for any given tuple of parameters. Specifically, we show that for any given tuple $(K, B, OPT)$, any algorithm must incur regret $\Omega(\min(OPT, OPT\sqrt{K/B} + \sqrt{K \cdot OPT}))$.

**Tags**: #lower_bound #optimality #matching #bwk #or

---

## [Corollaries for Applications] We derive corollaries for the [N] examples outlined in Section [X]:

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Summarizing application-specific results

> We derive corollaries for the three examples outlined in Section 1.1:
> - We obtain regret $\tilde{O}(B^{2/3})$ for the basic version of dynamic pricing. This is optimal for each $(B,T)$ pair.
> - We obtain regret $\tilde{O}(T/B^{1/4})$ for the basic version of dynamic procurement.
> - We obtain regret $\tilde{O}(\sqrt{B})$ for dynamic ad allocation. This is optimal when $B=T$.

**Tags**: #corollaries #applications #results #bwk #or

---

## [Prior Work Comparison] Prior work [citation] achieved [result] w.r.t. [benchmark], and [result] assuming [condition]. The former result is much weaker than ours, and the latter is incomparable.

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Comparing to prior work precisely

> Prior work achieved $\tilde{O}(B^{2/3})$ regret w.r.t. the best fixed price, and $\tilde{O}(\sqrt{B})$ regret assuming "regularity". The former result is much weaker than ours, see Appendix for a simple example, and the latter result is incomparable.

**Tags**: #comparison #prior_work #improvement #bwk #or

---

## [Generality Emphasis] To emphasize the generality of our contributions, we systematically discuss [topic] in Section [X]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Pointing to comprehensive treatment

> To emphasize the generality of our contributions, we systematically discuss applications and corollaries in Section 5. Pointers to prior work on special cases can be found in Section 1.4.

**Tags**: #generality #organization #roadmap #bwk #or
