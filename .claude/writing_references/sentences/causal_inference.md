# Causal Inference Sentence Templates

Sentence patterns for causal inference and econometrics papers.

---

## [Broad Motivation] In many applications, we want to use data to draw inferences about [causal effect]: Examples include [list of domains]

**Source**: Wager & Athey - "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (JASA 2018)
**Context**: Opening with broad motivation and diverse applications

> In many applications, we want to use data to draw inferences about the causal effect of a treatment: Examples include medical studies about the effect of a drug on health outcomes, studies of the impact of advertising on consumer purchases, evaluations of the effectiveness of government programs, and "A/B tests" commonly used by technology firms.

**Tags**: #introduction #motivation #causal #wager

---

## [Data Revolution] Historically, most datasets have been too small to [goal]. Recently, however, there has been an explosion of [settings] where it is potentially feasible to [new capability].

**Source**: Wager & Athey - JASA 2018
**Context**: Motivating why new methods are needed now

> Historically, most datasets have been too small to meaningfully explore heterogeneity of treatment effects beyond dividing the sample into a few subgroups. Recently, however, there has been an explosion of empirical settings where it is potentially feasible to customize estimates for individuals.

**Tags**: #motivation #data #opportunity #wager

---

## [Methodological Fear] An impediment to [goal] is the fear that researchers will [bad practice], thus highlighting [spurious result]

**Source**: Wager & Athey - JASA 2018
**Context**: Acknowledging methodological concerns

> An impediment to exploring heterogeneous treatment effects is the fear that researchers will iteratively search for subgroups with high treatment levels, and then report only the results for subgroups with extreme effects, thus highlighting heterogeneity that may be purely spurious.

**Tags**: #motivation #concern #methodology #wager

---

## [Procedural Restrictions] For this reason, [protocols exist]. However, such procedural restrictions can make it difficult to [discover something valuable].

**Source**: Wager & Athey - JASA 2018
**Context**: Acknowledging existing solutions and their limitations

> For this reason, protocols for clinical trials must specify in advance which subgroups will be analyzed. However, such procedural restrictions can make it difficult to discover strong but unexpected treatment effect heterogeneity.

**Tags**: #methodology #limitation #tradeoff #wager

---

## [Paper Goal] In this paper, we seek to address this challenge by developing [method] that yields [desirable property]

**Source**: Wager & Athey - JASA 2018
**Context**: Stating paper's contribution

> In this paper, we seek to address this challenge by developing a powerful, nonparametric method for heterogeneous treatment effect estimation that yields valid asymptotic confidence intervals for the true underlying treatment effect.

**Tags**: #contribution #goal #method #wager

---

## [Classical Methods] Classical approaches to [problem] include [method A], [method B], and [method C]. These methods perform well in [simple setting], but quickly break down as [complexity increases].

**Source**: Wager & Athey - JASA 2018
**Context**: Reviewing classical approaches and their limitations

> Classical approaches to nonparametric estimation of heterogeneous treatment effects include nearest-neighbor matching, kernel methods, and series estimation. These methods perform well in applications with a small number of covariates, but quickly break down as the number of covariates increases.

**Tags**: #related_work #classical #limitation #wager

---

## [ML Connection] In this paper, we explore the use of ideas from the [ML/other field] literature to improve the performance of [classical methods]

**Source**: Wager & Athey - JASA 2018
**Context**: Bridging machine learning and statistics

> In this paper, we explore the use of ideas from the machine learning literature to improve the performance of these classical methods with many covariates.

**Tags**: #contribution #bridge #methodology #wager

---

## [Hurdles] Despite their widespread success at [task], there are important hurdles that need to be cleared before [method] is directly useful to [application]

**Source**: Wager & Athey - JASA 2018
**Context**: Identifying challenges in applying existing methods

> Despite their widespread success at prediction and classification, there are important hurdles that need to be cleared before random forests are directly useful to causal inference.

**Tags**: #challenge #gap #methodology #wager

---

## [Ideal Estimator] Ideally, an estimator should be [property A] with a well-understood [property B], so that a researcher can use it to [practical goal]

**Source**: Wager & Athey - JASA 2018
**Context**: Stating desirable properties for statistical methods

> Ideally, an estimator should be consistent with a well-understood asymptotic sampling distribution, so that a researcher can use it to test hypotheses and establish confidence intervals.

**Tags**: #methodology #desiderata #statistics #wager

---

## [Theory Gap] Yet, the [theoretical property] of [method] has been largely left open, even in [standard context]

**Source**: Wager & Athey - JASA 2018
**Context**: Identifying theoretical gap

> Yet, the asymptotics of random forests have been largely left open, even in the standard regression or classification contexts.

**Tags**: #gap #theory #contribution #wager

---

## [Paper Addresses] This paper addresses these limitations, developing [method] that allows for [theoretical property] and [practical property]

**Source**: Wager & Athey - JASA 2018
**Context**: Summarizing paper's contribution

> This paper addresses these limitations, developing a forest-based method for treatment effect estimation that allows for a tractable asymptotic theory and valid statistical inference.

**Tags**: #contribution #summary #method #wager

---

## [Generality First] In the interest of generality, we begin our theoretical analysis by developing [general result] in the context of [broader setting]

**Source**: Wager & Athey - JASA 2018
**Context**: Explaining proof strategy

> In the interest of generality, we begin our theoretical analysis by developing the desired consistency and asymptotic normality results in the context of regression forests.

**Tags**: #organization #theory #generality #wager
