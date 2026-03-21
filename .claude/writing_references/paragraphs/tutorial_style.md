# Tutorial Style Paragraph Templates

Paragraph patterns for pedagogical/tutorial-style writing.

---

## [Game Setup] Consider the following repeated game: In each round...

**Source**: Orabona - "A Modern Introduction to Online Learning" (arXiv 2019)
**Context**: Introducing a framework through a simple game

> Consider the following repeated game:
>
> In each round $t=1,\dots,T$
> - An adversary chooses a real number $y_t \in [0,1]$ and keeps it secret;
> - You try to guess the real number, choosing $x_t \in [0,1]$;
> - The adversary's number is revealed and you pay the squared difference $(x_t-y_t)^2$.
>
> Basically, we want to guess a sequence of numbers as precisely as possible. To make it a game, we must now define a "winning condition".

**Pattern**:
1. "Consider the following repeated game:"
2. Bullet-point protocol for each round
3. Informal summary: "Basically, we want to..."
4. Transition to formalizing: "To make it a game, we must..."

**Tags**: #tutorial #game #setup #orabona #pedagogical

---

## [Easing Assumptions] First, let's make the game easier. Let's assume that [simplification]. However, [adversary freedom remains].

**Source**: Orabona - arXiv 2019
**Context**: Starting with simplified setting before generalization

> First, let's make the game easier for the player. Let's assume that the adversary is drawing the numbers i.i.d. from some fixed distribution over $[0,1]$. However, he is still free to decide which distribution at the beginning of the game.

**Tags**: #tutorial #simplification #progression #orabona

---

## [Benchmarking Intuition] If we knew [information], we could just [optimal action] and we would pay [optimal cost]. We cannot do better than that!

**Source**: Orabona - arXiv 2019
**Context**: Establishing optimal benchmark

> If we knew the distribution, we could just predict each round the mean of the distribution and in expectation we would pay $\sigma^2 T$, where $\sigma^2$ is the variance of the distribution. We cannot do better than that!

**Tags**: #tutorial #benchmark #intuition #orabona

---

## [Natural Measure] Given that we do not know [information], it is natural to benchmark our strategy with respect to [optimal]. That is, it is natural to measure [quantity].

**Source**: Orabona - arXiv 2019
**Context**: Motivating performance metric

> However, given that we do not know the distribution, it is natural to benchmark our strategy with respect to the optimal one. That is, it is natural to measure the quantity [regret formula].

**Tags**: #tutorial #metric #motivation #orabona

---

## [Success Criterion] It would make sense to consider a strategy "successful" if [condition].

**Source**: Orabona - arXiv 2019
**Context**: Defining what it means to win

> It would make sense to consider a strategy "successful" if the difference grows sublinearly over time and, equivalently, if the difference goes to zero as the number of rounds $T$ goes to infinity.

**Tags**: #tutorial #criterion #success #orabona

---

## [Generalization Step] Now, the last step: let's remove the assumption on [X], consider any arbitrary [Y], and let's keep using the same measure of success.

**Source**: Orabona - arXiv 2019
**Context**: Removing assumptions while keeping framework

> Now, the last step: let's remove the assumption on how the data is generated, consider any arbitrary sequence of $y_t$, and let's keep using the same measure of success.

**Tags**: #tutorial #generalization #progression #orabona

---

## [Justification Summary] Our reasoning should provide sufficient justification for this metric, however in the following we will see that this also makes sense from both [perspective A] and [perspective B] point of view.

**Source**: Orabona - arXiv 2019
**Context**: Promising additional justification

> Our reasoning should provide sufficient justification for this metric, however in the following we will see that this also makes sense from both a convex optimization and machine learning point of view.

**Tags**: #tutorial #justification #preview #orabona

---

## [Framework Power] This framework is pretty powerful, and it allows to reformulate a bunch of different problems in [field]. More in general, with [framework] we can analyze situations in which [difficult setting].

**Source**: Orabona - arXiv 2019
**Context**: Highlighting framework generality

> This framework is pretty powerful, and it allows to reformulate a bunch of different problems in machine learning and optimization as similar games. More in general, with the regret framework we can analyze situations in which the data are not independent and identically distributed from a distribution, yet I would like to guarantee that the algorithm is "learning" something.

**Tags**: #tutorial #framework #power #orabona

---

## [Adversarial Intuition] The fact that [property] means that we can immediately rule out [approach]. In fact, it cannot work because [reason].

**Source**: Orabona - arXiv 2019
**Context**: Building intuition for adversarial thinking

> The fact that the numbers are adversarially chosen means that we can immediately rule out any strategy based on any statistical modeling of the data. In fact, it cannot work because the moment we estimate something and act on our estimate, the adversary can immediately change the way he is generating the data, ruining us.

**Tags**: #tutorial #adversarial #intuition #orabona

---

## [Strategy Design] Now, let's try to design a strategy to [achieve goal], regardless of [adversarial freedom]. The first thing to do is to take a look at [benchmark].

**Source**: Orabona - arXiv 2019
**Context**: Transitioning to algorithm design

> Now, let's try to design a strategy to make the regret provably sublinear in time, regardless of how the adversary chooses the numbers. The first thing to do is to take a look at the best strategy in hindsight.

**Tags**: #tutorial #design #transition #orabona
