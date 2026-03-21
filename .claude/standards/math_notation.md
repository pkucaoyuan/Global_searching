# 数学符号字典

本文档定义了 LLM-for-BAI 项目中数学符号与代码变量的映射关系。所有代码中的变量命名必须遵循此字典。

---

## 1. Track A: BAI with Biased LLM Judges

### 1.1 核心变量

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $K$ | `K` | `n_arms` | 手臂（选项）数量 |
| $k$ | `k` | `arm_idx` | 手臂索引 |
| $X$ | `X` | `context` | 上下文/特征 |
| $Y$ | `Y$ | `human_label` | 真实标签（人工审计结果） |
| $F$ | `F` | `judge_score` | LLM 判断得分 |
| $R$ | `R = Y - F$ | `residual` | 残差（偏差） |
| $A$ | `A$ | `audit_flag` | 审计指示符（0/1） |
| $\pi_t$ | `\pi_t$ | `audit_prob` | 第 $t$ 轮的审计概率 |
| $N_k(t)$ | `N_k(t)$ | `count_arm[k]` | 手臂 $k$ 到时刻 $t$ 的拉取次数 |
| $n_k(t)$ | `n_k(t)$ | `audit_count[k]` | 手臂 $k$ 到时刻 $t$ 的审计次数 |

### 1.2 估计量

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $\theta_k$ | `\theta_k` | `true_mean[k]` | 手臂 $k$ 的真实均值 $\mathbb{E}[Y \mid k]$ |
| $\mu_{F,k}$ | `\mu_{F,k}` | `judge_mean[k]` | LLM 得分均值 $\mathbb{E}[F \mid k]$ |
| $\mu_{R,k}$ | `\mu_{R,k}` | `residual_mean[k]` | 残差均值 $\mathbb{E}[R \mid k]$ |
| $\hat{\mu}_{F,k}$ | `\hat{\mu}_{F,k}` | `judge_mean_hat[k]` | LLM 得分的样本均值 |
| $\hat{\mu}_{R,k}^{IPW}$ | `\hat{\mu}_{R,k}^{IPW}` | `ipw_estimate[k]` | IPW 残差估计 |
| $\hat{\mu}_{R,k}^{DR}$ | `\hat{\mu}_{R,k}^{DR}` | `dr_estimate[k]` | DR 残差估计 |
| $\hat{\theta}_k$ | `\hat{\theta}_k` | `mean_estimate[k]` | 真实均值估计 $\hat{\mu}_F + \hat{\mu}_R$ |

### 1.3 置信区间

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $L_k(t)$ | `L_k(t)$ | `lower_bound[k]` | 手臂 $k$ 的置信下界 |
| $U_k(t)$ | `U_k(t)$ | `upper_bound[k]` | 手臂 $k$ 的置信上界 |
| $\delta$ | `\delta` | `delta` | 错误概率上界 |
| $\beta(n, \delta)$ | `\beta(n, \delta)$ | `confidence_width` | 置信宽度函数 |

### 1.4 LUCB 算法

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $b_t$ | `b_t$ | `empirical_best` | 经验最优臂 $\arg\max_k \hat{\theta}_k$ |
| $c_t$ | `c_t$ | `challenger` | 挑战者 $\arg\max_{k \neq b_t} U_k(t)$ |
| $\hat{k}$ | `\hat{k}$ | `recommended_arm` | 算法推荐的最优臂 |
| $k^*$ | `k^*` | `true_best_arm` | 真实最优臂 |
| $\Delta_k$ | `\Delta_k$ | `gap[k]` | 次优间隔 $\theta_{k^*} - \theta_k$ |

---

## 2. Track B: Policy Learning with Stochastic Measurement

### 2.1 核心变量

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $D$ | `D` | `doc` | 原始文档 |
| $X$ | `X$ | `label` | 从文档提取的标签（离散） |
| $\mathcal{X}$ | `\mathcal{X}$ | `label_space` | 标签空间 |
| $A$ | `A$ | `action` | 动作/决策 |
| $\mathcal{A}$ | `\mathcal{A}$ | `action_space` | 动作空间 |
| $Y$ | `Y$ | `outcome` | 结果 |
| $\pi$ | `\pi$ | `policy` | 策略 $\pi: \mathcal{X} \to \mathcal{A}$ |
| $p(x)$ | `p(x)$ | `label_prob[x]` | 标签 $x$ 的边际概率 |
| $\mu(x, a)$ | `\mu(x, a)$ | `outcome_mean[x, a]` | 条件均值 $\mathbb{E}[Y \mid X=x, A=a]$ |

### 2.2 策略价值

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $V(\pi)$ | `V(\pi)$ | `policy_value` | 策略 $\pi$ 的价值 |
| $V^*$ | `V^*$ | `optimal_value` | 最优策略价值 |
| $\pi^*$ | `\pi^*$ | `optimal_policy` | 最优策略 |
| $\hat{\pi}$ | `\hat{\pi}$ | `learned_policy` | 学习得到的策略 |
| $\underline{V}(\pi)$ | `\underline{V}(\pi)` | `v_lower` | 策略价值下界 |
| $\bar{V}(\pi)$ | `\bar{V}(\pi)$ | `v_upper` | 策略价值上界 |
| $\underline{V}^*$ | `\underline{V}^*` | `v_star_lower` | 最优价值下界 |
| $\bar{V}^*$ | `\bar{V}^*$ | `v_star_upper` | 最优价值上界 |

### 2.3 计数与估计

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $N_t$ | `N_t$ | `n_samples` | 到时刻 $t$ 的总样本数 |
| $N_t(x)$ | `N_t(x)$ | `count_label[x]` | 标签 $x$ 的观测次数 |
| $N_t(x, a)$ | `N_t(x, a)$ | `count_label_action[x, a]` | $(x, a)$ 对的观测次数 |
| $\hat{p}_t(x)$ | `\hat{p}_t(x)$ | `label_prob_hat[x]` | 标签概率的经验估计 |
| $\hat{\mu}_t(x, a)$ | `\hat{\mu}_t(x, a)` | `outcome_mean_hat[x, a]` | 条件均值的经验估计 |

### 2.4 复制设计

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $R$ | `R$ | `n_replications` | 每个文档的 LLM 查询次数 |
| $D_i$ | `D_i$ | `doc_batch[i]` | 第 $i$ 个文档 |
| $X_{i,r}$ | `X_{i,r}$ | `labels[i, r]` | 文档 $i$ 第 $r$ 次查询的标签 |

---

## 3. 共享数学概念

### 3.1 置信序列（Confidence Sequences）

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $\delta$ | `\delta` | `delta` | 错误概率（$1 - \delta$ 置信水平） |
| $t$ | `t$ | `t` 或 `round_idx` | 时间/轮次索引 |
| $N$ 或 $n$ | `N` | `n_samples` | 样本量 |
| $M_t$ | `M_t$ | `martingale` | Martingale 值 |
| $\Lambda_t$ | `\Lambda_t$ | `likelihood_ratio` | 似然比 |
| $\mathcal{C}_t(\delta)$ | `\mathcal{C}_t(\delta)` | `cs_interval` | 时刻 $t$ 的置信区间 |

### 3.2 Martingale 相关

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $S_t$ | `S_t$ | `sum_t` | 部分和 $\sum_{s=1}^t X_s$ |
| $\bar{X}_t$ | `\bar{X}_t$ | `sample_mean` | 样本均值 |
| $V_t$ | `V_t$ | `variance_sum` | 方差累积和 |
| $\lambda$ | `\lambda$ | `lambda_param` | 混合参数 |
| $\rho$ | `\rho$ | `mixture_density` | 混合密度 |

### 3.3 Dirichlet-Multinomial 混合（Track B 特有）

| 数学符号 | LaTeX | 代码变量 | 描述 |
|----------|-------|----------|------|
| $\alpha$ | `\alpha$ | `alpha` | Dirichlet 参数向量 |
| $\alpha_0$ | `\alpha_0$ | `alpha_sum` | 参数和 $\sum_x \alpha_x$ |
| $\text{Dir}(\alpha)$ | `\text{Dir}(\alpha)$ | `dirichlet_prior` | Dirichlet 先验 |
| $\text{Multi}(N, p)$ | `\text{Multi}(N, p)$ | `multinomial` | 多项分布 |

---

## 4. 代码中的命名约定

### 4.1 前缀约定
- `true_*`: 真实值（如 `true_mean`, `true_best_arm`）
- `*_hat`: 估计值（如 `mean_hat`, `policy_hat`）
- `*_lower`, `*_upper`: 置信区间边界
- `count_*`: 计数（如 `count_arm`, `count_label`）
- `n_*`: 数量参数（如 `n_arms`, `n_samples`）

### 4.2 后缀约定
- `*_idx`: 索引（如 `arm_idx`, `doc_idx`）
- `*_prob`: 概率（如 `audit_prob`, `label_prob`）
- `*_mean`: 均值（如 `judge_mean`, `residual_mean`）
- `*_sum`: 累积和（如 `variance_sum`）

### 4.3 数组 vs 标量
- 单数形式用于标量: `arm_idx`, `outcome`
- 复数形式用于数组: `outcomes`, `audit_probs`

---

## 5. 公式参考

### 5.1 Track A 核心公式

**IPW 估计量：**
$$\hat{\mu}_{R,k}^{IPW} = \frac{1}{N_k} \sum_{s: k_s = k} \frac{A_s}{\pi_s}(Y_s - F_s)$$

**DR 估计量：**
$$\hat{\mu}_{R,k}^{DR} = \frac{1}{N_k} \sum_{s: k_s = k} \left[ \hat{g}_k(X_s) + \frac{A_s}{\pi_s}(R_s - \hat{g}_k(X_s)) \right]$$

**置信区间（Empirical Bernstein）：**
$$\mathcal{C}_t(\delta) = \hat{\mu}_t \pm \sqrt{\frac{2\hat{V}_t \log(2/\delta)}{t}} + \frac{7\log(2/\delta)}{3(t-1)}$$

### 5.2 Track B 核心公式

**策略价值：**
$$V(\pi) = \sum_{x \in \mathcal{X}} p(x) \mu(x, \pi(x))$$

**价值边界（凸优化）：**
$$\underline{V}(\pi) = \min_{p \in \mathcal{P}_t, \mu \in \mathcal{M}_t} \sum_x p(x) \mu(x, \pi(x))$$

**Dirichlet-Multinomial 混合 Martingale：**
$$M_t = \frac{\prod_{x} \Gamma(\alpha_x + N_t(x)) / \Gamma(\alpha_x)}{\Gamma(\alpha_0 + t) / \Gamma(\alpha_0)}$$
