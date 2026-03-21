# Network Optimization Sentence Templates

Sentence patterns for network flow, graph algorithms, and combinatorial optimization papers.

---

## [Fundamental Theorem] In [theory], the [max-X min-Y theorem] states that [max quantity] equals [min quantity].

**Source**: Max-flow min-cut theorem
**Context**: Stating duality result

> In optimization theory, the max-flow min-cut theorem states that in a flow network, the maximum amount of flow passing from the source to the sink equals the total weight of the edges in a minimum cut.

**Tags**: #theorem #duality #fundamental #network

---

## [Problem Statement] That is, given a network with [elements] that have certain [properties], how much [quantity] can the network [process/support]?

**Source**: Network flow literature
**Context**: Informal problem statement

> That is, given a network with vertices and edges between those vertices that have certain weights, how much "flow" can the network process at a time?

**Tags**: #problem #informal #question #network

---

## [Historical Discovery] It was discovered in [year] by [authors].

**Source**: Ford-Fulkerson algorithm
**Context**: Historical attribution

> It was discovered in 1956 by Ford and Fulkerson, and remains one of the most fundamental algorithms in combinatorial optimization.

**Tags**: #history #discovery #attribution #network

---

## [Algorithm Description] The [algorithm] works as follows. First, we [initialize]. Then we [search for structure]. If [condition], then we can [improve]. We keep on [iterating] until [termination].

**Source**: Ford-Fulkerson method
**Context**: Algorithm description template

> The Ford-Fulkerson method works as follows. First, we set the flow of each edge to zero. Then we look for an augmenting path from s to t. If such a path is found, then we can increase the flow along these edges. We keep on searching for augmenting paths and increasing the flow until no more augmenting paths exist.

**Tags**: #algorithm #description #steps #network

---

## [Key Concept Definition] A [concept] is [definition] where [condition] for all [elements] along that [structure].

**Source**: Augmenting path definition
**Context**: Defining key concept

> An augmenting path is a simple path in the residual graph where residual capacity is positive for all the edges along that path.

**Tags**: #definition #concept #key #network

---

## [Complexity Statement] The complexity of [algorithm] is [bound], where [variable] is the [quantity].

**Source**: Algorithm analysis
**Context**: Stating complexity

> The complexity of Ford-Fulkerson is O(EF), where F is the maximal flow of the network. The Edmonds-Karp variant achieves O(VE²) by using BFS for finding augmenting paths.

**Tags**: #complexity #bound #analysis #network

---

## [Implementation Variant] [Algorithm B] is an implementation of [Algorithm A] that uses [technique] for [subroutine].

**Source**: Edmonds-Karp algorithm
**Context**: Describing algorithm variant

> The Edmonds-Karp algorithm is an implementation of the Ford-Fulkerson method that uses BFS for finding augmenting paths, achieving polynomial time complexity.

**Tags**: #variant #implementation #technique #network

---

## [Independent Discovery] The algorithm was first published by [Author A] in [year], and later independently published by [Authors B] in [year].

**Source**: Algorithm history
**Context**: Noting independent discovery

> The algorithm was first published by Yefim Dinitz in 1970, and later independently published by Jack Edmonds and Richard Karp in 1972.

**Tags**: #history #discovery #independent #network

---

## [Fastest Known] The fastest known [type] algorithm, announced by [author] in [year], runs in [bound] time.

**Source**: Orlin's algorithm (2012)
**Context**: Stating state-of-the-art

> The fastest known purely combinatorial maximum-flow algorithm, announced by James Orlin in 2012, runs in O(VE) time.

**Tags**: #fastest #state_of_art #complexity #network

---

## [Broad Applications] Many important practical problems reduce to [problem]. Examples include [application A], [application B], and [application C].

**Source**: Network flow applications
**Context**: Showing breadth of applications

> Many important practical problems reduce to max flow. Examples include transportation-related problems, network attacks/sabotage problems, and bipartite matching and assignment problems.

**Tags**: #applications #reduction #practical #network

---
