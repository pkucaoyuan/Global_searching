# 学术写作规范

本规范适用于 LLM-for-BAI 项目的所有学术写作，包括论文、技术文档和研究笔记。

---

## 1. 三层写作质量框架

### Level 0: 逻辑流（Logic Flow）

**核心原则：旧信息 → 新信息**

每个句子应从读者已知的信息开始，然后引入新信息：

```
# 错误示例
A novel framework for handling biased LLM judges is proposed in this paper.
（新信息在前，读者不知道你在说什么）

# 正确示例
LLM judges often exhibit systematic biases. We propose a framework to correct for
these biases using selective human auditing.
（先建立背景，再引入新概念）
```

**主语-动词紧邻原则**

避免主语和动词之间插入过长的修饰语：

```
# 错误示例
The estimator, which is based on inverse propensity weighting and has been
shown in prior work to be unbiased under certain conditions, converges
at the optimal rate.
（主语"estimator"和动词"converges"相隔太远）

# 正确示例
The IPW estimator converges at the optimal rate. This follows from prior work
showing that IPW is unbiased under certain conditions.
（拆分成两句，每句主谓紧邻）
```

---

### Level 1: 语言质量（Language Quality）

#### 1.1 消灭僵尸名词（Zombie Nouns）

将弱名词化动词恢复为强动词：

| 僵尸名词形式 | 强动词形式 |
|-------------|-----------|
| perform an analysis of | analyze |
| conduct an investigation of | investigate |
| make an estimation of | estimate |
| provide a demonstration of | demonstrate |
| carry out an evaluation of | evaluate |
| give consideration to | consider |
| reach a conclusion that | conclude |
| make an assumption that | assume |

```
# 错误示例
We perform an analysis of the convergence rate and make an estimation
of the sample complexity.

# 正确示例
We analyze the convergence rate and estimate the sample complexity.
```

#### 1.2 删除冗余表达

| 删除 | 保留 |
|-----|------|
| It is worth noting that | [直接陈述] |
| Importantly, | [直接陈述] |
| It should be mentioned that | [直接陈述] |
| As a matter of fact | [删除] |
| In order to | To |
| Due to the fact that | Because |
| In the event that | If |
| At this point in time | Now |
| For the purpose of | To / For |

```
# 错误示例
It is worth noting that the algorithm terminates in finite time. Importantly,
the sample complexity is optimal.

# 正确示例
The algorithm terminates in finite time, and the sample complexity is optimal.
```

---

### Level 2: 结构润色（Structure Polish）

#### 2.1 句式节奏变化

避免连续使用相同句式：

```
# 单调的句式
We propose an algorithm. We prove its correctness. We show its optimality.
We conduct experiments.

# 有变化的句式
We propose an algorithm and prove its correctness. The optimality result
follows from a novel martingale argument. Experiments on synthetic data
confirm our theoretical predictions.
```

#### 2.2 自然过渡

段落之间需要逻辑连接：

- **对比**: However, In contrast, On the other hand
- **递进**: Furthermore, Moreover, Building on this
- **因果**: Therefore, Consequently, As a result
- **举例**: For instance, To illustrate, Consider

---

## 2. De-AI-ing 规范

### 2.1 必删的 AI 模板词

以下词汇过度使用已成为 AI 生成文本的标志，应避免或替换：

| 禁用词汇 | 替代方案 |
|---------|---------|
| novel | new / our / the proposed |
| significant | [删除，或用具体数字] |
| remarkable | [删除，或用具体描述] |
| comprehensive | thorough / complete / detailed |
| cutting-edge | recent / state-of-the-art |
| groundbreaking | [删除，让结果说话] |
| innovative | [删除，让方法说话] |
| leverages | uses / exploits / builds on |
| utilizes | uses |
| facilitates | enables / allows |
| pivotal | important / key / central |
| delve into | examine / study / explore |
| realm | area / domain / field |
| intricate | complex [也少用] |

### 2.2 禁用的开头模式

```
# 禁用
"In this paper, we propose a novel..."
"This work presents a comprehensive..."
"We leverage the power of..."
"Our innovative approach..."

# 推荐
"We study the problem of..."
"This paper addresses..."
"We develop an algorithm for..."
"We prove that..."
```

### 2.3 数字代替形容词

```
# 模糊的
"Our method achieves significant improvements."

# 具体的
"Our method reduces sample complexity by 40% compared to the baseline."
```

---

## 3. OR/统计论文特有规范

### 3.1 先直觉后形式化

在给出形式化定义或定理之前，先用一两句话解释直觉：

```
# 正确示例
The key insight is that the LLM judge's bias is arm-dependent but predictable.
We can therefore use a small number of human audits to estimate this bias
and correct our estimates.

**Definition 1** (Residual). For arm $k$ and context $x$, define the residual as
$R = Y - F$, where $Y$ is the human label and $F$ is the LLM score.
```

### 3.2 公式配解释

每个重要公式后应有 1-2 句解释：

```
The IPW estimator is given by:
$$\hat{\mu}_{R,k}^{IPW} = \frac{1}{N_k} \sum_{s: k_s = k} \frac{A_s}{\pi_s}(Y_s - F_s)$$
Here, $A_s / \pi_s$ is the importance weight that corrects for the selective
auditing: samples that were less likely to be audited receive higher weight.
```

### 3.3 用例子支撑抽象

引入抽象概念时，提供具体例子：

```
**Example.** Consider a content moderation system with $K=3$ policies:
strict, moderate, and lenient. An LLM judge scores each piece of content,
but tends to rate "strict" policies too harshly. Human auditors can provide
ground truth but are expensive. Our goal is to identify the policy with
the highest true quality score using as few human audits as possible.
```

### 3.4 证明要有 Roadmap

长证明开头应给出思路概述：

```
**Proof of Theorem 1.**
The proof proceeds in three steps. First, we show that the estimator is
unbiased (Lemma 1). Second, we bound its variance using martingale
concentration (Lemma 2). Finally, we combine these to obtain the
confidence sequence via Ville's inequality.

*Step 1: Unbiasedness.* ...
```

---

## 4. 修改分级制度

### Level 1: 直接修改（无需确认）

- 明显的 typo 和语法错误
- AI 模板词替换
- 冗余表达删除
- 僵尸名词转换

### Level 2: 简述后修改

修改前简要说明原因，然后直接修改：

- 语言润色（句式调整、词汇选择）
- 段落流程优化
- 过渡句添加
- 被动语态转主动

示例：
> "将被动语态改为主动以增强表达。"
> 原文: "The algorithm was proposed by Smith et al."
> 改为: "Smith et al. proposed the algorithm."

### Level 3: 确认后修改

涉及以下内容必须先获得用户确认：

- **数学公式修改**：任何对公式的改动
- **证明逻辑调整**：证明步骤的增删或重排
- **关键术语更改**：核心概念的命名
- **结论性陈述**：摘要、贡献声明、结论段落
- **大段删除或重写**：超过一个段落的改动

示例：
> "我注意到公式(3)中的方差项可能需要调整。建议将 $V_t$ 改为 $V_t / t$。
> 这样做的理由是...。请确认是否进行此修改。"

---

## 5. 章节写作模板

### 5.1 Introduction 结构

1. **问题背景**（1-2段）：建立问题的重要性
2. **现有方法的局限**（1段）：为什么需要新方法
3. **本文贡献**（1段）：清晰列出 3-4 个贡献点
4. **论文组织**（可选）：简述各章节内容

### 5.2 Contribution 写法

使用具体、可验证的陈述：

```
# 模糊的
"We propose a novel algorithm that outperforms baselines."

# 具体的
"We develop a LUCB-style algorithm that identifies the best arm with probability
at least $1-\delta$ using $O(H \log(K/\delta))$ human audits, where $H$ is
the problem-dependent complexity."
```

### 5.3 Related Work 写法

- 按主题而非按作者组织
- 明确说明与本文的关系（相似点和不同点）
- 避免仅仅罗列文献

```
# 错误示例
Smith (2020) studied bandit algorithms. Jones (2021) studied LLM evaluation.
Lee (2022) proposed confidence sequences.

# 正确示例
**Confidence sequences and anytime-valid inference.** Our approach builds on
the confidence sequence framework of Howard et al. (2021). Unlike their work,
which focuses on bounded random variables, we extend the theory to handle
inverse propensity weighted observations.
```

---

## 6. 常见错误清单

### 6.1 中式英语

| 中式表达 | 地道表达 |
|---------|---------|
| in recent years | recently |
| more and more | increasingly |
| play an important role | is important / matters |
| with the development of | as X develops / evolves |
| has attracted wide attention | has received attention |

### 6.2 标点符号

- 数学模式中的标点应在数学环境内：$x = 1,$ 而非 $x = 1$,
- 引用前的逗号：See~\cite{smith2020}, 而非 See,~\cite{smith2020}
- 连字符 vs 短破折号 vs 长破折号：
  - 连字符 `-`：复合词 (state-of-the-art)
  - 短破折号 `--`：数字范围 (pages 1--10)
  - 长破折号 `---`：插入语 (The result---which is surprising---shows...)

### 6.3 常见拼写/用法错误

| 错误 | 正确 |
|-----|------|
| indexes | indices (数学语境) |
| which 引导非限制性从句 | that 引导限制性从句 |
| data is | data are (但"this data"可接受) |
| amount of samples | number of samples |

### 6.4 严谨性要求

- **禁止在证明中使用 `\approx`**：所有数学推导必须是精确的。使用不等式（$\le, \ge$）、显式误差项（$+ \epsilon$）或大O符号（$+ O(1/n)$）。
  - 错误：$A \approx B$
  - 正确：$|A - B| \le \epsilon$ 或 $A = B + O(1/n)$

