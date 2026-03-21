# Problem Setup Sentence Templates

Sentence patterns for formally introducing problem settings.

---

## [Definition] We investigate / We consider

**Source**: Kaufmann, Cappé, Garivier - "On the Complexity of Best-Arm Identification" (JMLR 2016)
**Context**: Opening a technical paper with clear problem statement

> We investigate in this paper the complexity of finding the $m$ best arms in a stochastic multi-armed bandit model.

> We consider in this work the problem of [X] under [setting].

**Variants**:
- We study the problem of...
- We address the challenge of...
- This paper investigates...

**Tags**: #introduction #problem #formal #kaufmann

---

## [Model Definition] A [model] is a [formal description]

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Formally introducing mathematical model

> A bandit model $\nu$ is a collection of $K$ arms, where each arm $\nu_a$ is a probability distribution on $\R$ with expectation $\mu_a$.

**Pattern**:
> A [model name] is a [mathematical object], where [components are defined].

**Tags**: #definition #model #formal #kaufmann

---

## [Agent Action] At each time t, an agent [action] and [observation]

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Describing sequential interaction protocol

> At each time $t=1,2,\dots$, an agent chooses an option $A_t \in \{1,\dots,K\}$ and receives an independent draw $Z_t$ from the corresponding arm.

**Tags**: #setup #protocol #sequential #kaufmann

---

## [Goal Statement] The agent's goal is to [objective]

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Stating the learning objective

> The agent's goal is to identify the $m$ best arms, that is, the set $\cS^*_m$ of indices of the $m$ arms with highest expectation.

**Variants**:
- Our objective is to...
- The learner aims to...
- The goal of the algorithm is to...

**Tags**: #goal #objective #formal #kaufmann

---

## [Strategy Components] Its strategy consists in a triple

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Decomposing an algorithm into components

> More precisely, its strategy consists in a triple $\cA=((A_t),\tau,\hat{S}_m)$ in which:
> - the sampling rule determines...
> - the stopping rule controls...
> - the recommendation rule provides...

**Tags**: #algorithm #decomposition #structure #kaufmann

---

## [Two Settings] In the [field] literature, two different settings have been considered

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Introducing contrasting frameworks

> In the bandit literature, two different settings have been considered. In the fixed-confidence setting, [description]. Alternatively, in the fixed-budget setting, [description].

**Tags**: #framework #contrast #settings #kaufmann

---

## [Unification] In order to unify and compare these approaches

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Motivating a unified treatment

> In order to unify and compare these approaches, we define the complexity $\kappa_C(\nu)$ (resp. $\kappa_B(\nu)$) as follows:

**Variants**:
- To provide a unified view...
- For a comprehensive comparison...
- To bridge these perspectives...

**Tags**: #unification #comparison #motivation #kaufmann

---

## [Paper Aim] In this paper, we aim at [goal]. To achieve this, [ingredients]

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Stating paper's objectives and approach

> In this paper, we aim at evaluating and comparing these two complexities. To achieve this, two ingredients are needed: a lower bound on [X] and a strategy whose [Y] attains the lower bound.

**Tags**: #contribution #structure #roadmap #kaufmann

---

## [Surprise Result] We will show below that this conclusion is not valid anymore when

**Source**: Kaufmann, Cappé, Garivier - JMLR 2016
**Context**: Previewing surprising findings

> We will show below that this conclusion is not valid anymore when the values of $\mu_1$ and $\mu_2$ are not assumed to be known.

**Variants**:
- Surprisingly, this intuition breaks down when...
- Contrary to conventional wisdom, we demonstrate that...
- This familiar behavior no longer holds when...

**Tags**: #results #surprise #preview #kaufmann
