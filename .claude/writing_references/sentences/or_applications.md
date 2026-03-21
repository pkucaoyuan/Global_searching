# OR Applications Sentence Templates

Sentence patterns for connecting theory to OR/OM applications.

---

## [Classic Problem + Tension] For [time period], [problem] has been the predominant model for [setting] that embodies the tension between [tradeoff]

**Source**: Badanidiyuru, Kleinberg, Slivkins - "Bandits with Knapsacks" (JACM 2018)
**Context**: Introducing a classic problem with its fundamental tradeoff

> For more than fifty years, the multi-armed bandit problem has been the predominant theoretical model for sequential decision problems that embodies the tension between exploration and exploitation, "the conflict between taking actions which yield immediate reward and taking actions whose benefit will come only later," to quote Whittle's apt summary.

**Tags**: #introduction #classic #tradeoff #bwk #or

---

## [Universal Nature] Owing to the universal nature of [phenomenon], it is not surprising that [method] has found diverse applications ranging from [A], to [B], to [C]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Justifying broad relevance

> Owing to the universal nature of this conflict, it is not surprising that MAB algorithms have found diverse applications ranging from medical trials, to communication networks, to Web search and advertising.

**Tags**: #motivation #applications #universal #bwk #or

---

## [Common Feature + Examples] A common feature in many of these application domains is [property]. For example, [example 1]. [Example 2]. [Example 3].

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Identifying shared structure across applications

> A common feature in many of these application domains is the presence of one or more limited-supply resources that are consumed during the decision process. For example, scientists experimenting with alternative medical treatments may be limited not only by the number of patients but also by the cost of materials. A website experimenting with displaying advertisements is constrained not only by users but by advertisers' budgets. A retailer engaging in price experimentation faces inventory limits along with a limited number of consumers.

**Tags**: #motivation #common_feature #examples #bwk #or

---

## [Literature Gap] The literature on [topic] lacks a general model that encompasses [specific problems]. Our paper contributes such a model.

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Identifying literature gap and contribution

> The literature on MAB problems lacks a general model that encompasses these sorts of decision problems with supply limits. Our paper contributes such a model, called bandits with knapsacks.

**Tags**: #gap #contribution #model #bwk #or

---

## [Notable Examples Header] Notable examples.

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Section header for illustrative examples

Use **Notable examples.** or **Illustrative examples.** as a header before walking through concrete instances of the abstract model.

**Pattern**:
> \xhdr{Notable examples.}
> The conventional [problem] naturally fits into this framework. A more interesting example is [application problem]. Modeling this as a [our framework], [mapping explanation].

**Tags**: #examples #applications #or #bwk

---

## [Mapping to Framework] Modeling this as a [framework] problem, [elements] correspond to [abstract concepts]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Showing how an application maps to abstract model

> Modeling this as a BwK problem, rounds correspond to consumers, and arms correspond to the possible prices which may be offered to a consumer. Reward is the revenue from a sale, if any. Resource consumption vectors express the number of items sold and consumers seen, respectively.

**Tags**: #mapping #framework #application #bwk #or

---

## [Dual Problem] A "dual" problem of [X] is [Y], where the algorithm is [action] rather than [opposite action]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Introducing dual/symmetric formulation

> A "dual" problem of dynamic pricing is dynamic procurement, where the algorithm is "dynamically buying" rather than "dynamically selling".

**Tags**: #dual #symmetry #formulation #bwk #or

---

## [Domain Relevance] This problem is also relevant to the domain of [field]: the [elements] then correspond to [concrete objects]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Connecting to additional application domain

> This problem is also relevant to the domain of crowdsourcing: the items "bought" then correspond to microtasks ordered on a crowdsourcing platform such as Amazon Mechanical Turk.

**Tags**: #relevance #domain #application #bwk #or

---

## [Generalization Note] All [N] examples can be easily generalized to [extension]: resp., to [specific 1], [specific 2], and [specific 3]

**Source**: Badanidiyuru et al. - JACM 2018
**Context**: Noting straightforward extensions

> All three examples can be easily generalized to multiple resource constraints: resp., to selling multiple products, procuring different types of goods, and allocating ads from multiple advertisers.

**Tags**: #generalization #extension #examples #bwk #or
