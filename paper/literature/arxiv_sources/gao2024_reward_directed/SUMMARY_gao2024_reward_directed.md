# Gao, Zha, Zhou (2024) --- "Reward-Directed Score-Based Diffusion Models via q-Learning"
ArXiv: 2409.04832

---

## 1. Setting & Model

**Problem.** Train continuous-time score-based diffusion models from scratch to generate samples that (i) maximize a reward function $h$ capturing user preferences, while (ii) keeping the generated distribution close to the unknown target data distribution $p_0$ --- all *without* any pretrained score model.

**Formulation.** The denoising (reverse) process is governed by an SDE whose drift includes the unknown score $\nabla \log p_{T-t}$. The authors treat the score as a *control action* and set up a continuous-time entropy-regularized RL problem:

$$\max_{\mathbf{a}} \left\{ \beta \, \mathbb{E}[h(\mathbf{y}_T^{\mathbf{a}})] - \mathbb{E}\!\int_0^T g^2(T-t)\,\|\nabla\log p_{T-t}(\mathbf{y}_t^{\mathbf{a}}) - a_t\|^2\,dt \right\}$$

subject to the controlled SDE $d\mathbf{y}_t = [f(T-t)\mathbf{y}_t + g^2(T-t)\,a_t]\,dt + g(T-t)\,dW_t$, with $\mathbf{y}_0 \sim \nu$ (Gaussian prior).

The running cost penalizes deviation of the policy from the (unknown) true score and equals the KL divergence between the controlled and true reverse-time path measures (by Girsanov's theorem). This is then lifted to an entropy-regularized exploratory RL problem with temperature $\theta > 0$.

**Key assumptions / design choices:**

1. The forward process follows an Ornstein--Uhlenbeck SDE with known, nonnegative continuous functions $f(t)$ and $g(t)$.
2. **No pretrained model required.** The running reward involves the *true* (unknown) score rather than a pretrained score, making this a genuine RL problem with unknown rewards.
3. The data distribution $p_0$ is unknown but i.i.d. samples from $p_0$ are available.
4. The terminal reward $h$ need not be differentiable nor known in closed form; only noisy observations (reward signals) of $h(\mathbf{y}_T)$ are required.
5. **Score signals via ratio estimator.** The unknown score value at any $(t, \mathbf{x})$ is estimated as a ratio of two expectations w.r.t. the data distribution, using $m$ i.i.d. samples from $p_0$. Remarkably, $m = 1$ suffices for high-dimensional image tasks (CIFAR-10, $d = 3072$).
6. Admissible stochastic policies $\boldsymbol{\pi}(\cdot | t, y)$ must have full support on $\mathbb{R}^d$, satisfy Lipschitz and moment conditions (Definition 1).
7. Temperature $\theta > 0$ controls exploration; weight $\beta \geq 0$ balances reward vs. score-matching fidelity. Setting $\beta = 0$ reduces to pure score matching.

---

## 2. Main Results

### Proposition 1 (Optimal Gaussian Policy)

The optimal stochastic policy $\boldsymbol{\pi}^*(\cdot | t, y)$ for the entropy-regularized RL problem is Gaussian in $\mathbb{R}^d$:

$$\boldsymbol{\pi}^*(\cdot | t, y) \sim \mathcal{N}\!\left(\mu^*(t,y),\; \frac{\theta}{2\,g^2(T-t)} \cdot I_d\right),$$

where the mean is

$$\mu^*(t,y) = \nabla \log p_{T-t}(y) + \tfrac{1}{2}\,J_y^*(t,y),$$

and $J^*$ is the optimal value function. The covariance is *known* and proportional to the temperature $\theta$; it is inversely proportional to $g^2(T-t)$.

**Interpretation.** The optimal mean decomposes into (a) the true score (score-matching component) and (b) a gradient correction $\frac{1}{2} J_y^*$ (reward-seeking component). The exploration noise is smaller when the forward noise $g$ is large, since less additional exploration is needed.

### Proposition 2 (Martingale Characterization)

Let $\hat{J}^* \in C^{1,2}$ with $\hat{J}^*(T,y) = h(y)$ and let $\hat{q}^*$ satisfy $\int \exp(\hat{q}^*/\theta)\,da = 1$. Then:

**(i)** If $\hat{J}^*$ and $\hat{q}^*$ are respectively the optimal value function and $q$-function, then for any admissible policy $\boldsymbol{\pi}$ and any time grid $\mathbb{S}$, the process

$$\hat{J}^*(s, \mathbf{y}_s^{\boldsymbol{\pi}, \mathbb{S}}) + \int_t^s \bigl[r(\tau, \mathbf{y}_\tau, a_\tau) - \hat{q}^*(\tau, \mathbf{y}_\tau, a_\tau)\bigr]\,d\tau$$

is an $\mathcal{F}^{\mathbb{S}}$-martingale.

**(ii)** Conversely, if there exists one admissible policy under which the above process is a martingale, then $\hat{J}^*$ and $\hat{q}^*$ are the optimal value and $q$-functions.

This characterization is the foundation for the actor-critic q-learning algorithm.

### Proposition 3 (SA Convergence)

Under (1) the existence of a twice continuously differentiable Lyapunov function $V$ with $\sup_{\epsilon \leq |(\Theta - \Theta^*, \psi - \psi^*)| \leq 1/\epsilon} \nabla V \cdot e(\Theta, \psi) < 0$ for all $\epsilon > 0$, and bounded second moments of the SA iterates; and (2) the standard Robbins--Monro step-size conditions $\sum_n \alpha(n) = \infty,\; \sum_n \alpha(n)^2 < \infty$:

The parameters $(\Theta_n, \psi_n)$ of the actor-critic q-learning algorithm converge to the optimal $(\Theta^*, \psi^*)$ almost surely.

**Caveat.** The Lyapunov function existence assumption is acknowledged as hard to verify for neural network function approximators and is left for future work.

### Ratio Estimator for Score Values (Eq. 19)

Given $m$ i.i.d. samples $\{\mathbf{x}_0^i\}$ from $p_0$, the score is estimated by

$$\widehat{\nabla_\mathbf{x} \log p_t(\mathbf{x})} = \frac{1}{\sigma_t^2}\left(-\mathbf{x} + \frac{\sum_{i=1}^m p_{t|0}(\mathbf{x}|\mathbf{x}_0^i)\,\mathbf{x}_0^i}{\max\{\sum_{i=1}^m p_{t|0}(\mathbf{x}|\mathbf{x}_0^i),\;m\epsilon\}} \cdot e^{-\int_0^t f(s)\,ds}\right)$$

where $\sigma_t^2 = \int_0^t e^{-2\int_s^t f(v)\,dv}\,g^2(s)\,ds$. This is asymptotically exact as $m \to \infty$ by SLLN. In high dimensions (CIFAR-10), the conditional density $p_{t|0}$ is dominated by the nearest sample, so $m = 1$ works well in practice.

### Loss Function Interpretation (Remark after Algorithm 1)

The actor-critic updates are equivalent to SGD on the mean-square TD loss:

$$L(\Theta, \psi) = \sum_{i=0}^{K-1} \left[J^\Theta(t_{i+1}, y_{t_{i+1}}) - J^\Theta(t_i, y_{t_i}) + r_{t_i}\Delta t - q^\psi(t_i, y_{t_i}, a_{t_i})\Delta t\right]^2$$

In practice, this is optimized with Adam over batches of $B$ trajectories.

---

## 3. Proof Techniques

1. **Girsanov's theorem.** Used to interpret the running penalty $\int g^2 \|\nabla\log p_{T-t} - a_t\|^2\,dt$ as the KL divergence between the path measures of the controlled and true reverse SDEs (Eq. 10). Also used in reverse to connect score matching loss to the KL $\text{KL}(\mathbb{P}^{\mathbf{z}} \| \mathbb{P}^{\mathbf{y}^{\mathbf{a}}})$ (Remark 1).

2. **Hamiltonian analysis from continuous-time stochastic control.** The Hamiltonian $H(t,y,a,p,q) = -g^2(T-t)|s(y) - a|^2 + [f(T-t)y + g^2(T-t)a] \circ p + \frac{1}{2}g^2(T-t) \circ q$ is quadratic in the action $a$, so completing the square directly yields the Gaussian form of the optimal policy.

3. **Exploratory HJB equation.** From the continuous-time RL framework of Wang (2020) and Jia--Zhou (2023): the optimal value function satisfies $\partial_t J^* + \theta \log \int \exp(H/\theta)\,da = 0$, and the optimal policy is $\boldsymbol{\pi}^*(a|t,y) = \exp(q^*/\theta)$.

4. **Martingale orthogonality conditions.** The TD residual process is a martingale at optimality. Test functions $\xi_t = \partial J^\Theta / \partial \Theta$ (TD(0)-like) and $\zeta_t = \partial \log \pi^\psi / \partial \psi$ (policy-gradient-like) yield the stochastic approximation equations for actor-critic updates.

5. **Ratio estimation of score signals.** Expressing $\nabla \log p_t(\mathbf{x}) = \mathbb{E}[\nabla p_{t|0}(\mathbf{x}|\mathbf{x}_0)] / \mathbb{E}[p_{t|0}(\mathbf{x}|\mathbf{x}_0)]$ and using Monte Carlo. In high dimensions, the softmax-like weighting concentrates on the nearest-neighbor data point, explaining why $m=1$ suffices.

6. **Robbins--Monro SA convergence theory** (Benveniste et al., 1990) for the parameter convergence guarantee, contingent on Lyapunov stability.

---

## 4. Connection to Our Paper

Our paper "Where to Search: GAINS" allocates inference-time compute across diffusion timesteps for noise trajectory search under a fixed NFE budget. Key concepts: two-level framework (local search operator + global scheduler), offline profiling + online control, water-filling allocation.

### What We Can Borrow

1. **Time-varying noise sensitivity provides theoretical basis for non-uniform allocation.** Proposition 1 shows the optimal exploration variance is $\theta / (2g^2(T-t))$, inversely proportional to the diffusion coefficient. This gives a principled, closed-form schedule for how much "perturbation" is warranted at each timestep --- directly supporting our premise that compute allocation across timesteps should be non-uniform. Their covariance schedule could serve as a theoretical baseline or initialization for our profiling stage.

2. **Score-matching running cost weight $g^2(T-t)$ as importance weight.** The per-step penalty in their objective is weighted by $g^2(T-t)$, quantifying each timestep's contribution to overall distributional fidelity. This time-varying weight is analogous to our profiling-based importance weights for compute allocation. It suggests a principled, theory-grounded way to initialize our scheduler's allocation before empirical profiling.

3. **Cheap single-sample signals suffice for trajectory-level learning.** Their demonstration that a single-sample ratio estimator ($m=1$) suffices for learning in high dimensions --- because noisy per-step signals average out across many steps and episodes --- supports the viability of our local search approach that uses cheap per-step evaluations. Even very noisy local search signals, when aggregated across the denoising trajectory, can guide effective optimization.

4. **Score estimation accuracy varies by timestep.** Their analysis of the ratio estimator (Table 4) shows MSE decreases as $t \to T$ (the noisy end is easier to estimate). This connects to our finding that early denoising steps (high noise, small $T-t$) are harder and may benefit from more search effort.

5. **ODE vs. SDE quality-efficiency tradeoff under varying NFE.** Section 7.1 systematically compares SDE and ODE samplers (DDIM, ODE-Euler) across $K = 5, 10, 50$ steps. ODE samplers degrade more gracefully with fewer steps for score matching ($\beta=0$), but SDE excels with many steps for reward-directed tasks. This is directly relevant to choosing the base sampler in our NFE-constrained setting.

### How We Differ

1. **Training-time vs. inference-time.** Gao et al. solve a training-time problem: learning a policy (score network) via RL over many episodes (50,000 episodes for 2D, 11,000+ for CIFAR-10). Our GAINS operates at inference time with a fixed pretrained model, allocating search compute per-sample without retraining.

2. **Reward-directed generation vs. unconditional quality maximization.** Their objective involves an explicit terminal reward $h$ and a KL penalty to the true distribution. We optimize sample quality (FID, CLIP score, etc.) by searching over noise realizations from a fixed model, without changing the model or introducing an external reward function.

3. **Continuous-time theory vs. discrete-step resource allocation.** Their analysis is in continuous time (SDEs, HJB equations, martingale theory). Our framework is inherently discrete: given $N$ total NFEs, decide how many to spend per timestep --- a combinatorial resource-allocation problem solved by water-filling.

4. **No NFE budget constraint.** Gao et al. run the full $K$-step denoising process in every episode with no budget constraint on inference cost. Our core contribution is precisely the NFE budget constraint and the principled allocation of limited search effort across steps.

5. **Scope of "search."** Their "exploration" (entropy-regularized policy) is a training mechanism to learn the optimal score-plus-reward policy. Our "search" is an inference-time mechanism to find a high-quality noise trajectory from a fixed, already-trained model without modifying any parameters.

---

## 5. Key Quotes

> "The optimal stochastic policy for our problem is Gaussian with a known covariance matrix and an unknown mean function. This key theoretical result suggests that we need to consider Gaussian policies only and parameterize their mean functions when designing RL algorithms." (Section 1, Introduction)

> "The exploration level is inversely proportional to $g^2(T-t)$. This is intuitive because $g$ represents the strength of the noise we add to blur the original samples. The higher this noise the less *additional* noise we need for exploration." (Section 3, after Proposition 1)

> "Although at a given time the algorithm generates a reward signal (i.e. a ratio estimator) using only one sample at random, at the next time point it generates another signal using another sample. When the number of episodes is large and/or the time step is small, a large number of samples will actually be used to produce these signals during the *entire* learning process. Consequently, all the noises in the reward signals will be eventually averaged out." (Section 6.3, discussing $m=1$)

---

## 6. BibTeX

```bibtex
@article{gao2024reward,
  title={Reward-Directed Score-Based Diffusion Models via q-Learning},
  author={Gao, Xuefeng and Zha, Jiale and Zhou, Xun Yu},
  journal={arXiv preprint arXiv:2409.04832},
  year={2024}
}
```
