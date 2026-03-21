# Control Theory Sentence Templates

Sentence patterns for control theory, online learning, and dynamical systems papers.

---

## [Performance Guarantee Dichotomy] Algorithms for [task] typically attain one of two performance guarantees. For [general setting], [metric] scales as [rate A], and this is tight. However, if [better structure], there exist algorithms that attain [better rate B].

**Source**: Agarwal et al. - "Logarithmic Regret for Online Control" (NeurIPS 2019)
**Context**: Opening with performance dichotomy

> Algorithms for regret minimization typically attain one of two performance guarantees. For general convex losses, regret scales as square root of the number of iterations, and this is tight. However, if the loss function exhibit more curvature, such as quadratic loss functions, there exist algorithms that attain poly-logarithmic regret.

**Tags**: #introduction #dichotomy #rates #control #ml

---

## [Gap Identification] Despite their ubiquitous use in [field A] and [field B], [method] is almost non-existent in [field C]. This can be attributed to [fundamental challenge].

**Source**: Agarwal et al. - NeurIPS 2019
**Context**: Identifying gap between fields

> Despite their ubiquitous use in online learning and statistical estimation, logarithmic regret algorithms are almost non-existent in control of dynamical systems. This can be attributed to fundamental challenges in computing the optimal controller in the presence of noise.

**Tags**: #motivation #gap #challenge #control #ml

---

## [First Result Claim] In this paper we give the first efficient [property] algorithms for [problem]. Our results apply to [general setting], and not only to [special case].

**Source**: Agarwal et al. - NeurIPS 2019
**Context**: Strong contribution claim

> In this paper we give the first efficient poly-logarithmic regret algorithms for controlling a linear dynamical system with noise in the dynamics (i.e. the standard model). Our results apply to general convex loss functions that are strongly convex, and not only to quadratics.

**Tags**: #contribution #first #generality #control #ml

---

## [Prior Approach Limitation] The approach taken by [prior work] is to use [technique]. However, this [removes/destroys] the properties associated with [structure], by [reason].

**Source**: Agarwal et al. - NeurIPS 2019
**Context**: Explaining why prior approaches fail

> The approach taken by [prior work] and other previous works is to use a semi-definite relaxation for the controller. However, this removes the properties associated with the curvature of the loss functions, by reducing the problem to an instance of online linear optimization.

**Tags**: #related_work #limitation #approach #control #ml

---

## [Different Approach] Therefore we take a different approach, initiated by [citation]. We consider [alternative formulation]. While this [admits challenge], it is not a priori clear that [desired property holds]. We demonstrate [conditions under which it does].

**Source**: Agarwal et al. - NeurIPS 2019
**Context**: Introducing novel approach

> Therefore we take a different approach, initiated by [citation]. We consider controllers that depend on the previous noise terms. While this resulting convex relaxation does not remove the curvature altogether, it results in an overparametrized representation, and it is not a priori clear that the loss functions are strongly convex. We demonstrate the appropriate conditions under which the strong convexity is retained.

**Tags**: #approach #methodology #contribution #control #ml

---

## [Two Methods Comparison] Henceforth we present two methods that attain [goal]. They differ in terms of [aspect A] and [aspect B]. The [method A] requires only [simple operation], whereas [method B], in addition, requires [complex operation]. However, [method B] admits [better guarantee].

**Source**: Agarwal et al. - NeurIPS 2019
**Context**: Presenting algorithm variants

> Henceforth we present two methods that attain poly-logarithmic regret. They differ in terms of the regret bounds they afford and the computational cost of their execution. The online gradient descent update requires only gradient computation and update, whereas the online natural gradient update, in addition, requires the computation of the preconditioner. However, the natural gradient update admits an instance-dependent upper bound on the regret.

**Tags**: #algorithm #comparison #tradeoff #control #ml

---

## [Survey Reference] For a survey of [topic], as well as [related problems], see [citation].

**Source**: Agarwal et al. - NeurIPS 2019
**Context**: Brief survey pointer

> For a survey of linear dynamical systems (LDS), as well as learning, prediction and control problems, see [citation].

**Tags**: #related_work #survey #reference #control

---

## [Closest Work] The closest work to ours is that of [citation A] and [citation B], aimed at [goal]. The authors obtain [result], but for [restricted setting]. In contrast, our results apply to [general setting], which presents the main challenges.

**Source**: Agarwal et al. - NeurIPS 2019
**Context**: Positioning relative to closest prior work

> The closest work to ours is that of [citation A] and [citation B], aimed at controlling LDS with adversarial loss functions. The authors obtain a $O(\log^2 T)$ regret algorithm for changing quadratic costs, but for dynamical systems that are noise-free. In contrast, our results apply to the full (noisy) LDS setting, which presents the main challenges.

**Tags**: #related_work #comparison #positioning #control #ml

---
