# Proof Structure Paragraph Templates

Paragraph patterns for presenting proofs and theoretical analysis.

---

## [Change of Measure Technique] The pioneering work of [citation] has popularized the use of [technique] to show [type of bounds]: the idea is to [intuition]. The cost of such [operation] is induced by [measure]: by choosing the most economical [choice], one can prove that [consequence].

**Source**: Garivier & Kaufmann - "Optimal Best Arm Identification with Fixed Confidence" (ALT 2016)
**Context**: Explaining classical proof technique

> The pioneering work of Lai & Robbins has popularized the use of changes of distributions to show problem-dependent lower bounds in bandit problems: the idea is to move the parameters of the arms until a completely different behavior of the algorithm is expected on this alternative bandit model. The cost of such a transportation is induced by the deviations of the arm distributions: by choosing the most economical move, one can prove that the alternative behavior is not too rare in the original model.

**Tags**: #proof #technique #lower_bound #change_of_measure #bai

---

## [Innovation on Classical] Here we go one step further by combining [multiple instances of technique], in the spirit of [citation]. This allows us to prove [stronger result] valid for [general setting].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Explaining technical innovation

> Here we go one step further by combining several changes of measures at the same time, in the spirit of Graves & Lai. This allows us to prove a non-asymptotic lower bound on the sample complexity valid for any $\delta$-PAC algorithm on any bandit model with a unique optimal arm.

**Tags**: #proof #innovation #technique #bai

---

## [Proof Sketch Introduction] Instead of choosing for each [element] a specific instance of [variable] that yields [partial result], we combine here the [constraints] given by all [alternatives].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Explaining proof approach at high level

> Instead of choosing for each arm $a$ a specific instance of $\lambda$ that yields a lower bound on $\mathbb{E}[N_a(\tau)]$, we combine here the inequalities given by all alternatives $\lambda$.

**Tags**: #proof #approach #sketch #bai

---

## [Making Bound Useful] To make this bound useful, it remains to study [quantity A] and [quantity B].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Transition from abstract to concrete

> To make this bound useful, it remains to study $T^*$ and $w^*$.

**Tags**: #proof #transition #analysis #bai

---

## [Optimization Problem Analysis] We study here the optimization problem [reference], so as to better understand [functions] (Proposition [X]), and in order to provide an efficient algorithm for computing first [quantity A] (Theorem [Y]), then [quantity B] (Lemma [Z]). The main ideas are outlined here, while all technical details are postponed to Appendix [A].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Roadmap for optimization problem analysis

> We study here the optimization problem~\eqref{equ:Original}, so as to better understand the function $T^*$ and $w^*$ (Proposition~\ref{prop:PropNu}), and in order to provide an efficient algorithm for computing first $w^*(\bm\mu)$ (Theorem~\ref{thm:ExplicitForm}), then $T^*(\bm\mu)$ (Lemma~\ref{lem:CDinterpretation}). The main ideas are outlined here, while all technical details are postponed to Appendix.

**Tags**: #organization #proof #roadmap #bai

---

## [Simplification Requires] Simplifying [expression] requires the introduction of the following [concept]: for every [parameter], let [definition].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Introducing technical tool for simplification

> Simplifying $T^*(\bm \mu)$ requires the introduction of the following parameterized version of the Jensen-Shannon divergence (which corresponds to $\alpha=1/2$): for every $\alpha\in[0,1]$, let [formula].

**Tags**: #proof #simplification #definition #bai

---

## [Sanity Check Properties] This characterization of [quantity] also permits to obtain a few sanity-check properties, like for example: (1) [property A]; (2) [property B]; (3) [property C].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Listing intuitive properties that verify correctness

> This characterization of $w^*(\bm\mu)$ also permits to obtain a few sanity-check properties, like for example:
> 1. For all $\bm\mu$, for all $a$, $w_a^* \neq 0$.
> 2. $w^*$ is continuous in every $\bm \mu$.
> 3. If $\mu_1>\mu_2 \geq \dots \geq \mu_K$, one has $w_2^* \geq \dots \geq w_K^*$.

**Tags**: #proof #properties #sanity_check #bai

---

## [Particular Cases] In general, it is not possible to give closed-form formulas for [quantities]. But the following particular cases can be mentioned.

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Transitioning to special cases

> In general, it is not possible to give closed-form formulas for $T^*(\bm \mu)$ and $w^*(\bm\mu)$. But the following particular cases can be mentioned.

**Tags**: #special_cases #analysis #closed_form #bai

---

## [Statistical Interpretation] From a statistical point of view, the question of [decision] is a more or less classical [test]: do the past observations allow to assess, with a risk at most $\delta$, that [hypothesis]?

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Statistical interpretation of algorithm component

> From a statistical point of view, the question of stopping at time $t$ is a more or less classical statistical test: do the past observations allow to assess, with a risk at most $\delta$, that one arm is larger than the others?

**Tags**: #interpretation #statistics #test #bai

---

## [Multiple Interpretations] In addition to the [interpretation A] given above, the [component] can be explained in light of [bound/theory]. Indeed, one may write [connection].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Providing multiple viewpoints on same result

> In addition to the testing interpretation given above, the stopping rule can be explained in light of the lower bound $\mathbb{E}[\tau/T^*]\geq \text{kl}(\delta,1-\delta)$. Indeed, one may write [formula showing connection].

**Tags**: #interpretation #multiple_views #connection #bai

---

## [MDL Interpretation] It is also possible to give a Minimum Description Length (MDL) interpretation of [component]. It is well known that choosing the model that gives the shortest description of the data is a provably efficient heuristic. In some sense, [our approach] follows the same principle.

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Information-theoretic interpretation

> It is also possible to give a Minimum Description Length (MDL) interpretation of the stopping rule. It is well known that choosing the model that gives the shortest description of the data is a provably efficient heuristic. In some sense, the stopping rule presented above follows the same principle.

**Tags**: #interpretation #mdl #information_theory #bai

---
