# Algorithm Optimality Sentence Templates

Sentence patterns for presenting optimal algorithms and matching lower bounds.

---

## [Complete Characterization] We give a complete characterization of the complexity of [problem]. We prove a new, tight lower bound on [metric]. We propose [algorithm], which we prove to be asymptotically optimal.

**Source**: Garivier & Kaufmann - "Optimal Best Arm Identification with Fixed Confidence" (ALT 2016)
**Context**: Abstract-level contribution summary

> We give a complete characterization of the complexity of best-arm identification in one-parameter bandit problems. We prove a new, tight lower bound on the sample complexity. We propose the `Track-and-Stop' strategy, which we prove to be asymptotically optimal.

**Tags**: #contribution #optimality #characterization #bai

---

## [Algorithm Components] It consists in a new [component A] (which [achieves goal A]) and in a [component B], for which we give a new analysis.

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Decomposing algorithm into components

> It consists in a new sampling rule (which tracks the optimal proportions of arm draws highlighted by the lower bound) and in a stopping rule named after Chernoff, for which we give a new analysis.

**Tags**: #algorithm #structure #components #bai

---

## [Paradigmatic Framework] A [model] is a paradigmatic framework of [field] made of [components]: at every time step [protocol description].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Formal model introduction

> A multi-armed bandit model is a paradigmatic framework of sequential statistics made of $K$ probability distributions: at every time step one arm is chosen and a new, independent reward is drawn.

**Tags**: #model #setup #framework #bai

---

## [Historical + Modern Interest] Introduced in the [decade] with motivations originally from [field A], [model] has raised a large interest recently as relevant models for [field B] or [field C].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Historical motivation with modern relevance

> Introduced in the 1930s with motivations originally from clinical trials, bandit models have raised a large interest recently as relevant models for interactive learning schemes or recommender systems.

**Tags**: #motivation #history #applications #bai

---

## [Good Understanding → Extensions] A good understanding of this simple model has allowed for efficient strategies in much more elaborate settings, for example including [extension A], [extension B], or [extension C], to name just a few.

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Showing how basic results enable extensions

> A good understanding of this simple model has allowed for efficient strategies in much more elaborate settings, for example including side information, infinitely many arms, or for the search of optimal strategies in games, to name just a few.

**Tags**: #extensions #applications #generalization #bai

---

## [Different Objective] In some of these applications, the real objective is not to [standard goal], but rather to [alternative goal], as fast and accurately as possible, regardless of [tradeoff].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Distinguishing problem variant

> In some of these applications, the real objective is not to maximize the cumulated reward, but rather to identify the arm that yields the largest mean reward, as fast and accurately as possible, regardless of the number of bad arm draws.

**Tags**: #problem #objective #distinction #bai

---

## [Strategy Components] A strategy is then defined by: (i) a [component A], where [property]; (ii) a [component B], which is [property]; and (iii) a [component C].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Formal strategy definition with bullet points

> A strategy is then defined by:
> - a *sampling rule* $(A_t)_t$, where $A_t$ is $\mathcal{F}_{t-1}$-measurable;
> - a *stopping rule* $\tau$, which is a stopping time with respect to $\mathcal{F}_t$;
> - and a $\mathcal{F}_\tau$-measurable *decision rule* $\hat{a}_\tau$.

**Tags**: #definition #strategy #components #bai

---

## [Two Settings] Two settings have been considered in the literature. In the [setting A], [constraint], and one aims at [goal A]. In the [setting B], [constraint], and one looks for [goal B].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Contrasting two problem formulations

> Two settings have been considered in the literature. In the fixed-budget setting, the number of draws is fixed in advance, and one aims at minimizing the probability of error. In the fixed-confidence setting, a maximal risk $\delta$ is fixed, and one looks for a strategy guaranteeing that the error probability is at most $\delta$ while minimizing the expected sample complexity.

**Tags**: #settings #formulations #problem #bai

---

## [Gap Statement] There was still a gap between the lower bounds, involving [complexity term A], and the upper bounds on [metric] for these algorithms, even from an asymptotic point of view.

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Identifying theory-algorithm gap

> There was still a gap between the lower bounds, involving complexity terms reflecting only partially the structure of the problem, and the upper bounds on $\mathbb{E}[\tau]$ for these particular algorithms, even from an asymptotic point of view.

**Tags**: #gap #lower_bound #upper_bound #bai

---

## [First Result] The first result of this paper is a tight, non-asymptotic lower bound on [metric]. This bound involves a [complexity measure] for the problem, which does not take a simple form like [naive expression]. Instead, it appears as the solution of an optimization problem.

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Presenting lower bound contribution

> The first result of this paper is a tight, non-asymptotic lower bound on $\mathbb{E}[\tau]$. This bound involves a `characteristic time' for the problem, depending on the parameters of the arms, which does not take a simple form like a sum of squared inverse gaps. Instead, it appears as the solution of an optimization problem.

**Tags**: #contribution #lower_bound #complexity #bai

---

## [Second Contribution] The second contribution is a new [property] algorithm that asymptotically achieves this lower bound, that we call [name].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Algorithm contribution statement

> The second contribution is a new $\delta$-PAC algorithm that asymptotically achieves this lower bound, that we call the Track-and-Stop strategy.

**Tags**: #contribution #algorithm #optimality #bai

---

## [Algorithm Intuition] In a nutshell, the idea is to [intuitive goal], and to [secondary goal]. The [component] can be interpreted in [N] equivalent ways: [interpretation A]; [interpretation B]; and [interpretation C].

**Source**: Garivier & Kaufmann - ALT 2016
**Context**: Giving high-level algorithm intuition

> In a nutshell, the idea is to sample so as to equalize the probability of all possible wrong decisions, and to stop as soon as possible. The stopping rule can be interpreted in three equivalent ways: in statistical terms, as a generalized likelihood ratio test; in information-theoretic terms, as an application of the Minimal Description Length principle; and in terms of optimal transportation, in light of the lower bound.

**Tags**: #intuition #algorithm #interpretation #bai

---
