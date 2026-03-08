# Summary: Adaptive Computation Time for Recurrent Neural Networks

**Citation Key:** graves2016adaptive
**arXiv:** 1603.08983
**Authors:** Alex Graves (Google DeepMind)

---

## Setting & Model

### Focus
This paper introduces **Adaptive Computation Time (ACT)**, an algorithm that allows recurrent neural networks to learn how many computational steps to take between receiving an input and emitting an output. The key idea is that computation time should adapt to problem difficulty - some inputs require more "pondering" than others.

### Key Assumptions
1. The amount of computation required varies unpredictably with input difficulty
2. Mean-field updates over multiple computation steps are approximately valid (representations are approximately linear)
3. A differentiable mechanism for halting decisions is preferable to stochastic sampling
4. The network should be parsimonious - minimizing computation while maintaining accuracy

### Notation
- $\rnn$: Recurrent neural network
- $\statef$: State transition model
- $s_t^n$: Intermediate state at timestep $t$, computation step $n$
- $N(t)$: Total number of updates at step $t$
- $h_t^n = \sigma(W_h s_t^n + b_h)$: Halting unit activation
- $p_t^n$: Halting probability
- $R(t) = 1 - \sum_{n=1}^{N(t)-1} h_t^n$: Remainder
- $\rho_t = N(t) + R(t)$: Ponder cost
- $\tau$: Time penalty parameter

---

## Main Results

### 1. ACT Algorithm

**Intermediate Computations:**
For each input step $t$, compute variable number of intermediate states:
$$s_t^n = \begin{cases}\statef(s_{t-1}, x_t^1) & \text{if } n=1\\ \statef(s_t^{n-1}, x_t^n) & \text{otherwise}\end{cases}$$

**Halting Mechanism:**
$$N(t) = \min\{n': \sum_{n=1}^{n'} h_t^n \geq 1-\epsilon\}$$

**Halting Probability Distribution:**
$$p_t^n = \begin{cases} R(t) & \text{if } n=N(t)\\ h_t^n & \text{otherwise}\end{cases}$$

**Mean-Field Updates:**
$$s_t = \sum_{n=1}^{N(t)} p_t^n s_t^n \qquad y_t = \sum_{n=1}^{N(t)} p_t^n y_t^n$$

### 2. Ponder Cost and Training
Loss function with time penalty:
$$\hat{\mathcal{L}}(x, y) = \mathcal{L}(x, y) + \tau \sum_{t=1}^T \rho_t$$

The ponder cost $\rho_t = N(t) + R(t)$ provides an upper bound on total computation.

### 3. Gradient Computation
Treating $N(t)$ as constant:
$$\frac{\partial \hat{\mathcal{L}}}{\partial h_t^n} = \frac{\partial \mathcal{L}}{\partial h_t^n} - \begin{cases} 0 & \text{if } n = N(t)\\ \tau & \text{otherwise}\end{cases}$$

### 4. Experimental Results
- **Parity**: Without ACT ~40% error vs <5% with ACT ($\tau \leq 0.03$)
- **Logic**: Virtually zero error for $\tau \leq 0.01$
- **Addition**: Perfect accuracy with ponder time scaling linearly with digit count
- **Sort**: Error reduced from ~12% to ~6% with substantial computation increase
- **Wikipedia**: Similar error rates, but ponder time reveals data structure (pauses at word/sentence boundaries)

---

## Proof Techniques / Methods

### Theoretical Foundations
1. **Mean-field approximation**: States/outputs are approximately linear, enabling weighted averaging
2. **Differentiable halting**: Using remainder $R(t)$ makes the halting distribution differentiable
3. **Upper bound on computation**: $\mathcal{P}(x)$ bounds $\sum_t N(t)$

### Key Design Choices
- $\epsilon = 0.01$ allows single-step computation when $h_1 \geq 1-\epsilon$
- Maximum computation limit $M$ prevents excessive pondering early in training
- Same state/output parameters shared across intermediate steps

### Empirical Methodology
- Adam optimizer, $lr = 10^{-4}$
- Hogwild! asynchronous training with 16 threads
- Grid search over $\tau \in \{i \times 10^{-j}\}$ for $i \in [1,10]$, $j \in [1,4]$
- 20 random initializations per $\tau$ value

---

## Connection to Our Paper

### Direct Relevance: MODERATE-HIGH

ACT provides foundational insights for our global scheduling framework:

1. **Adaptive Compute per Timestep**: ACT's core idea - learning to allocate variable computation based on difficulty - directly maps to our global scheduler's goal. In diffusion models:
   - ACT timesteps $\leftrightarrow$ Noise levels $\sigma$
   - ACT ponder time $N(t)$ $\leftrightarrow$ Search iterations per timestep
   - ACT halting probability $\leftrightarrow$ Budget allocation decisions

2. **Difficulty-Aware Allocation**: ACT learns that some inputs require more computation. For diffusion:
   - Some noise levels are "harder" than others
   - Boundary regions (high-to-low noise transitions) may need more search
   - Our global scheduler can learn similar difficulty profiles

3. **Ponder Cost as Regularization**: ACT's $\tau \mathcal{P}(x)$ penalty provides a principled way to trade off accuracy vs. computation. We can adapt this:
   $$\hat{\mathcal{L}} = \mathcal{L}_{\text{quality}} + \tau \sum_{\sigma} K(\sigma)$$
   where $K(\sigma)$ is search budget at noise level $\sigma$.

4. **Mean-Field Updates**: ACT's mean-field approximation over intermediate states relates to how we might aggregate multiple search candidates at each timestep.

5. **Wikipedia Insight**: ACT's discovery that networks pause at "boundaries" (spaces, punctuation) suggests diffusion models may benefit from more computation at **transition regions** in the noise schedule.

### Key Differences
- **ACT**: Per-input adaptive computation in sequence models
- **Our Paper**: Per-timestep adaptive computation across the noise trajectory

### Adaptation Opportunities

**Learnable Budget Allocation:**
Inspired by ACT's halting unit, we could add a budget allocation network:
$$K(\sigma) = f_\phi(\sigma, \text{trajectory state})$$

**Differentiable Scheduling:**
ACT's differentiable halting mechanism suggests we could make our scheduler differentiable:
- Soft budget allocation instead of hard discrete choices
- End-to-end training of scheduler with generation quality

**Ponder-Like Metrics:**
Track "ponder profiles" across the noise trajectory to understand which timesteps benefit most from additional search.

---

## Key Quotes

> "The amount of time required to pose a problem and the amount of thought required to solve it are notoriously unrelated."

> "We would like the network to be parsimonious in its use of computation, ideally limiting itself to the minimum number of steps necessary to solve the problem."

> "Finding this limit in its most general form would be equivalent to determining the Kolmogorov complexity of the data (and hence solving the halting problem). We therefore take the more pragmatic approach of adding a time cost to the loss function to encourage faster solutions."

> "The network then has to learn to trade off accuracy against speed, just as a person must when making decisions under time pressure."

> "Character prediction networks trained with ACT consistently pause at spaces between words, and pause for longer at 'boundary' characters such as commas and full stops."

> "This suggests that ACT or other adaptive computation methods could provide a generic method for inferring segment boundaries in sequence data."

> "One weakness of the current algorithm is that it is quite sensitive to the time penalty parameter that controls the relative cost of computation time versus prediction error."

---

## BibTeX

```bibtex
@article{graves2016adaptive,
  title={Adaptive Computation Time for Recurrent Neural Networks},
  author={Graves, Alex},
  journal={arXiv preprint arXiv:1603.08983},
  year={2016}
}
```
