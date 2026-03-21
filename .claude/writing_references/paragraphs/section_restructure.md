# Section Restructure Writing Patterns

Paragraph and sentence patterns for restructuring sections: flattening subsections, promoting/demoting results, writing rate-optimality comparisons, and bridging examples to theory. All patterns extracted from real restructure experience (Section 6, MS journal paper) and top venues.

---

## [Section Opening: Analysis Framing] Having established [property A] and [property B] for [algorithm] ([reference]), we now analyze [metric]. We first derive [bound type A] that [decomposition insight], then establish [bound type B] on [general class]. Comparing the two shows that [algorithm] achieves [optimality claim], with [caveat].

**Source**: Section 6 restructure (MS journal, 2026-02-09)
**Context**: Opening a flat analysis section that replaces multiple subsections. The sentence does three things: (1) links backward to previous section, (2) previews the two main results, (3) states the punchline.

> Having established correctness and a cost-optimal auditing rule for PP-LUCB (Remark~\ref{rem:cost_guarantees}), we now analyze its expected cost. We first derive an upper bound that decomposes PP-LUCB's cost into proxy variance and audit residual variance, then establish an information-theoretic lower bound on the cost of any $\delta$-correct algorithm. Comparing the two shows that PP-LUCB achieves the optimal rate in the confidence and gap parameters, with a discrepancy only in instance-dependent constants.

**Tags**: #section_opening #flat_structure #analysis #rate_optimality

---

## [Bridge Question: Can We Improve?] A natural question is whether this rate can be improved. To answer it, we [approach] using [framework].

**Source**: Section 6 restructure (MS journal, 2026-02-09)
**Context**: Transitioning from an upper bound to a lower bound. The "natural question" framing motivates the lower bound without making it feel like a separate subsection.

> A natural question is whether this rate can be improved. To answer it, we lower-bound the cost of *any* $\delta$-correct algorithm using a Local Asymptotic Normality (LAN) framework.

**Tags**: #transition #bridge_question #upper_to_lower

---

## [Rate Optimality Comparison] Together, [Theorem A] and [Theorem B] show that [algorithm] is **rate-optimal**: both the upper bound ([Theorem A]) and the lower bound ([Theorem B]) scale as [rate expression], so the logarithmic dependence on [parameter] and inverse-quadratic dependence on [parameter] cannot be improved by any [algorithm class]. The gap between the two bounds lies in [instance-dependent constants], not in the [rate].

**Source**: Section 6 restructure (MS journal, 2026-02-09)
**Context**: The key paragraph connecting upper and lower bounds to establish rate optimality. This replaces a subsection-level treatment with a single focused paragraph.

> Together, Theorems~\ref{thm:cost_bound} and~\ref{thm:lower_bound} show that PP-LUCB is **rate-optimal**: both bounds scale as $\log(1/\delta)/\Delta_k^2$, so the logarithmic dependence on the confidence level and the inverse-quadratic dependence on the gap cannot be improved by any $\delta$-correct algorithm. The discrepancy between the upper and lower bounds lies in instance-dependent constants—specifically, the relationship between Neyman-allocated variances and the KL-optimal information allocation—not in the rate.

**Tags**: #rate_optimality #comparison #upper_lower_bound

---

## [Example Bridge: Trade-off Exposition] This decomposition exposes a fundamental trade-off: [increasing X] yields [benefit]—[reducing time/cost for Y]—but [raises cost Z]. The following example makes this trade-off concrete.

**Source**: Section 6 restructure (MS journal, 2026-02-09)
**Context**: Bridging from an abstract KL decomposition formula to a concrete parametric example. The em-dash parenthetical adds operational meaning.

> This decomposition exposes a fundamental trade-off: increasing $\pi_k$ yields more information per sample—reducing the time to distinguish $\nu$ from alternatives—but raises the average audit cost. The following example makes this trade-off concrete.

**Tags**: #bridge #example #trade_off #decomposition

---

## [Example Interpretation: Squared-Sum Structure] The [structural form] shows that no algorithm can avoid [cost A]: [source 1] alone [is insufficient when condition], and [relying solely on source 2] is wasteful when [condition].

**Source**: Section 6 restructure (MS journal, 2026-02-09)
**Context**: Interpreting a formula's functional form. The "shows that" verb drives the claim; the two clauses give operational meaning to each term.

> The squared-sum structure shows that no algorithm can avoid the cost of both information sources: proxy observations alone are insufficient when $\sigma_{R,k}^2 > 0$, and relying solely on audits is wasteful when the proxy carries useful signal.

**Tags**: #interpretation #formula_structure #operational_meaning

---

## [Remark Pointing to Appendix] Closing the gap between the upper and lower bounds requires a different algorithmic approach. Appendix~\ref{sec:proof_optimality} describes [algorithm variant], a [technique]-based alternative that [achieves X] as [condition]. However, [practical limitation], making [original algorithm] the preferred choice in practice.

**Source**: Section 6 restructure (MS journal, 2026-02-09)
**Context**: Demoting a subsection's algorithm to an appendix remark. The remark acknowledges the result exists while explaining why the main algorithm is preferred.

> Closing the gap between the upper and lower bounds requires a different algorithmic approach. Appendix~\ref{sec:proof_optimality} describes PP-Track-and-Audit, a C-tracking-based alternative that achieves $C^*(\nu)\log(1/\delta)(1+o(1))$ as $\delta\to 0$. However, it requires knowledge of instance-dependent constants that are unavailable in practice, making PP-LUCB the preferred choice for deployment.

**Tags**: #appendix_remark #demotion #practical_preference

---

## [Promoting a Result from Appendix] The following result, proved in Appendix~\ref{sec:proof_X}, [bounds/characterizes] the expected cost of [algorithm]. Its proof combines [technique A] with [technique B].

**Source**: Section 6 restructure (MS journal, 2026-02-09)
**Context**: Promoting a theorem from the appendix to the main text. Keep the statement clean; defer the proof.

> The following result, proved in Appendix~\ref{sec:proof_cost_bound}, gives an upper bound on the expected total cost of PP-LUCB. Its proof combines a stopping-time analysis with variance decomposition under the Neyman allocation.

**Tags**: #promotion #theorem_statement #appendix_proof

---

## [Moving Content to Discussion] Beyond the rate analysis above, the cost structure reveals [managerial/operational insight]. [Proposition/Result] formalizes this observation.

**Source**: Section 6 restructure (MS journal, 2026-02-09)
**Context**: Moving comparative statics from a theory subsection to the Discussion/Conclusion. The sentence reframes the technical result as an operational insight.

> Beyond the rate analysis above, the cost structure reveals how the optimal audit intensity responds to changes in the proxy's informativeness. Proposition~\ref{prop:comparative_statics} formalizes this observation: as proxy variance $\sigma_F^2$ decreases, the optimal audit probability $\pi^*$ decreases, and the total cost drops at rate $O(\sigma_F)$.

**Tags**: #discussion #comparative_statics #operational_insight

---

## [Truncated Distribution Motivation] To satisfy [assumption] (which requires [property]), we consider [truncated distribution] with support [interval]. When [condition on parameters], the correction factor [parameter] ≈ 1, recovering the [standard case].

**Source**: Section 6 restructure (MS journal, 2026-02-09)
**Context**: Justifying use of truncated distributions in examples to satisfy bounded-support assumptions.

> To satisfy Assumption~\ref{ass:bounded} (which requires outcomes in $[0,1]$), we consider truncated normals $\text{TN}(\mu, \sigma^2, [0,1])$ rather than unbounded Gaussians. When the mean is well-centered and variance is moderate, the Fisher information correction factor $\kappa \geq 1$ satisfies $\kappa \approx 1$, recovering the standard Gaussian result.

**Tags**: #example #truncation #assumption_satisfaction #bounded_support

---
