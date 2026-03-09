# 修改方案: 技术细节公式化

## 问题描述

当前描述不够清晰:
- 喂给verifier的具体是什么？
- Short rollout是什么意思？
- xt → x0 → xt-1的关系？

## 解决方案

### 1. 明确Verifier输入

在Local search operator部分添加:

```latex
% 替换原有的模糊描述

\paragraph{Local search operator.}
At timestep $t$, given a current latent $x_t$ and candidate noise
$\varepsilon_t$, the diffusion model first predicts the clean image:
\begin{equation}
  \hat{x}_0 = D_\theta(x_t, t, c),
  \label{eq:x0_prediction}
\end{equation}
where $D_\theta$ is the denoising network. The verifier then evaluates
this prediction: $s = v(\hat{x}_0, c)$. The transition to $x_{t-1}$ is
computed from $\hat{x}_0$ and fresh noise:
\begin{equation}
  x_{t-1} = \alpha_{t-1} \hat{x}_0 + \sigma_{t-1} \varepsilon_t,
  \label{eq:transition}
\end{equation}
where $\alpha_{t-1}, \sigma_{t-1}$ are scheduler coefficients.
```

### 2. Short Rollout公式化

```latex
% 详细描述local search过程

A \emph{local search operator} $\mathcal{L}_t$ at timestep $t$ samples
$K$ candidate noises $\{\varepsilon_t^{(k)}\}_{k=1}^K$ and evaluates each:
\begin{equation}
  \hat{x}_0^{(k)} = D_\theta(x_t, t, c), \quad
  s^{(k)} = v(\hat{x}_0^{(k)}, c), \quad k = 1, \ldots, K.
\end{equation}
The best candidate is selected: $k^* = \arg\max_k s^{(k)}$, and the
corresponding noise $\varepsilon_t^{(k^*)}$ is used to compute $x_{t-1}$
via \cref{eq:transition}. If no candidate improves over the current
best score, the original noise is retained.
```

### 3. 符号表更新

需要在symbols.md中添加:
- $D_\theta$: Denoising network (predicts $\hat{x}_0$ from $x_t$)
- $\hat{x}_0$: Predicted clean image
- $\alpha_t, \sigma_t$: Scheduler coefficients
- $K$: Number of candidate noises per local search

### 4. 验证清单

- [ ] 是否明确说明verifier输入是$\hat{x}_0$（不是$x_{t-1}$）
- [ ] 是否有$x_t \to \hat{x}_0 \to x_{t-1}$的公式
- [ ] 是否解释了K个candidates的评估过程
- [ ] 读者能否据此写出代码
