# Online Learning Sentence Templates

Sentence patterns for online learning and regret minimization papers.

---

## [Problem Statement with Applications] [Problem] is one of the most widely studied problems in [field]. It has been applied successfully to problems in [field A], [field B], [field C] and [field D]. In recent years, it has also received much attention from [community], as increasingly difficult [challenges] have led to demand for [solution type].

**Source**: Cohen et al. - "Online Learning for Online Control" (NeurIPS 2018)
**Context**: Opening with broad applicability

> Linear-quadratic (LQ) control is one of the most widely studied problems in control theory. It has been applied successfully to problems in statistics, econometrics, robotics, social science and physics. In recent years, it has also received much attention from the machine learning community, as increasingly difficult control problems have led to demand for data-driven control systems.

**Tags**: #introduction #applications #broad #control #ml

---

## [Motivation from Applications] This problem may arise in settings such as [application A] in the presence of [complication], due to [cause A] or [cause B].

**Source**: Cohen et al. - NeurIPS 2018
**Context**: Concrete application motivation

> This problem may arise in settings such as building climate control in the presence of time-varying energy costs, due to energy auctions or unexpected demand fluctuations.

**Tags**: #motivation #application #concrete #or

---

## [Regret Definition] To measure how well [system] adapts to [challenge], it is common to consider the notion of regret: the difference between [learner cost] and [benchmark cost].

**Source**: Cohen et al. - NeurIPS 2018
**Context**: Introducing regret metric

> To measure how well a control system adapts to time-varying costs, it is common to consider the notion of regret: the difference between the total cost of the controller, one that is only aware of previously observed costs, and that of the best fixed control policy in hindsight.

**Tags**: #definition #regret #metric #online

---

## [Connection to Online Learning] This notion has been thoroughly studied in the context of [field A], and particularly in that of [field B].

**Source**: Cohen et al. - NeurIPS 2018
**Context**: Connecting to established literature

> This notion has been thoroughly studied in the context of online learning, and particularly in that of online convex optimization.

**Tags**: #connection #literature #online #convex

---

## [Main Results Summary] Our main results are [N] online algorithms that achieve [guarantee], when comparing to [benchmark class].

**Source**: Cohen et al. - NeurIPS 2018
**Context**: Summarizing algorithmic contributions

> Our main results are two online algorithms that achieve $O(\sqrt{T})$ regret, when comparing to any fast mixing linear policy.

**Tags**: #contribution #algorithm #results #online

---

## [Approach Overview] Overall, our approach follows [citation]. We first show how to perform [task] in an "idealized setting," a hypothetical setting in which [simplification]. We proceed to bound the gap between [idealized] and [actual].

**Source**: Cohen et al. - NeurIPS 2018
**Context**: Explaining proof strategy

> Overall, our approach follows [citation]. We first show how to perform online learning in an "idealized setting," a hypothetical setting in which the learner can immediately observe the steady-state cost of any chosen control policy. We proceed to bound the gap between the idealized costs and the actual costs.

**Tags**: #approach #technique #analysis #online

---

## [Conceptual Novelty] Our technique is conceptually different to most [problems]: instead of [standard approach], the learner [novel approach]. Importantly, this view allows us to cast the [problem] as [formulation] which [advantage].

**Source**: Cohen et al. - NeurIPS 2018
**Context**: Highlighting conceptual contribution

> Our technique is conceptually different to most learning problems: instead of predicting a policy and observing its steady-state cost, the learner predicts a steady-state distribution and derives from it a corresponding policy. Importantly, this view allows us to cast the idealized problem as a semidefinite program which minimizes the expected costs.

**Tags**: #novelty #technique #conceptual #online

---

## [Holy Grail Statement] The holy grail of [field] is [grand challenge], and clearly both [model A] and [model B] are well within this mission statement.

**Source**: Cohen et al. - NeurIPS 2018
**Context**: Connecting to broader research agenda

> The holy grail of reinforcement learning is controlling a dynamical stochastic system under uncertainty, and clearly both MDPs and LQ control are well within this mission statement.

**Tags**: #motivation #grand_challenge #connection #rl

---

## [Cross-Fertilization] In this work we are inspired by methodologies from [field A] and [technique B] to derive new results for [field C]. We believe that exploring the interface between the two will be fruitful for both sides.

**Source**: Cohen et al. - NeurIPS 2018
**Context**: Cross-field inspiration

> In this work we are inspired by methodologies from online-MDP and regret minimization to derive new results for LQ control. We believe that exploring the interface between the two will be fruitful for both sides, and holds significant potential for future research agenda.

**Tags**: #cross_field #inspiration #methodology #online

---
