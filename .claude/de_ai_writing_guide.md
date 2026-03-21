# 去AI化学术写作指南 v4.0 - OR论文混合增强版

本指南专为**Operations Research理论论文**设计，深度整合了通用学术写作原则与OR数学写作的特殊需求。

**理论基础**：
* **逻辑内核**：Gopen & Swan - *The Science of Scientific Writing* (读者认知心理学)
* **风格血肉**：Helen Sword - *Stylish Academic Writing* (声音、动词与叙事)
* **教学灵魂**：Stanford Teaching Handbook (受众意识与大图景)
* **OR特质**：数学严谨性与叙事清晰度的完美平衡

**核心理念**：写作不是数据的堆砌，而是对读者注意力流（Flow of Attention）的精准管理。

---

## 第零部分：OR论文特定原则 (OR-Specific Principles)
*专为Operations Research理论论文的数学写作设计*

### 0.1 段落功能清晰 (One Paragraph, One Function)

**OR论文原理**：数学论文的每个段落都应承担一个明确的功能——定义问题、陈述结果、证明步骤、或讨论意义。混杂多个功能会打乱读者的逻辑线。

**AI的坏习惯**：
* 在同一段落中既定义符号，又陈述定理，还讨论意义
* 段落内部缺乏层次，所有句子都平等重要
* 主题句（topic sentence）缺失或模糊

**人类进阶策略**：
1. **写下功能**：在写一个段落前，先用一句话写下"本段的目的是..."
2. **主题句清晰**：第一句话明确宣告本段功能（"We now prove Theorem 1..."）
3. **检查一致性**：写完后，检查每一句是否都服务于主题句的功能
4. **四种功能模板**：
   - 定义问题："We consider a system where..."
   - 陈述结果："Theorem 1 shows that..."
   - 证明步骤："To prove this, we first establish..."
   - 讨论意义："This result implies that..."

**修改案例**：
* *Bad (AI)*: "We consider a system with memory constraints. Theorem 1 proves stability. This is important because it prevents cascading failures. Let M* denote the equilibrium batch size." (四个功能混杂：定义、陈述、讨论、符号定义)
* *Good (Human)*: 分为两段：
  - 段落1（定义问题）："We consider a system with memory constraints, where M* denotes the equilibrium batch size."
  - 段落2（陈述结果+意义）："Theorem 1 proves stability under these constraints. This result is significant because it prevents cascading failures that plague baseline algorithms."

---

### 0.2 从直觉到严格 (Intuition Before Formalism)

**OR论文原理**：读者需要先理解"为什么"，才能吸收"是什么"。直接抛出形式化定义或定理，会让读者迷失在符号丛林中。

**AI的坏习惯**：
* 开篇就给出复杂公式：π* = arg min E[L(π)]...
* 定理陈述没有前置说明其motivation
* 证明直接开始推导，不解释proof strategy

**人类进阶策略**：
1. **先给直觉句**：在公式/定理前，用一句话说明"直观上..."
2. **然后形式化**：用"Formally,"或"Mathematically,"过渡
3. **Proof sketch优先**：复杂证明前，先用1-2句话描述proof road map
4. **用对比突出**："Unlike baseline FCFS, our threshold mechanism..."

**修改案例**：
* *Bad (AI)*:
  ```
  The optimal policy is:
  π* = arg min_{π∈Π} E[∑ᵢ Lᵢ]
  subject to M(t) ≤ C for all t.
  ```
  (直接扔公式，没有任何铺垫)

* *Good (Human)*:
  ```
  Intuitively, the optimal policy should minimize expected latency while preventing memory overflow.
  Formally, we define:
  π* = arg min_{π∈Π} E[∑ᵢ Lᵢ]
  subject to M(t) ≤ C for all t,
  where M(t) is the memory usage at time t.
  ```

---

### 0.3 数学与文字平衡 (Equations Need Prose)

**OR论文原理**：公式是语言的一部分，不是独立存在的。连续的公式没有文字过渡，等于让读者自己猜测公式之间的逻辑关系。

**AI的坏习惯**：
* 连续3-4个公式，中间没有任何文字连接
* 公式后直接跟下一个公式，不说明"为什么"这样推导
* 把公式当作"证明结束"的标志

**人类进阶策略**：
1. **公式前铺垫**："From Eq(1), we derive..."
2. **公式后解释**："This leads to Eq(3), which shows..."
3. **用连接词**："where", "by", "since", "thus" 解释每一步
4. **黄金比例**：1个公式配1-2句文字说明

**修改案例**：
* *Bad (AI)*:
  ```
  τ = d₀ + d₁M(t)     (1)
  M(t) = ∑ⱼ nⱼ(t)mⱼ    (2)
  λ < 1/τ*            (3)
  ```
  (三个公式零解释)

* *Good (Human)*:
  ```
  The iteration time depends linearly on memory usage:
  τ = d₀ + d₁M(t),    (1)
  where M(t) = ∑ⱼ nⱼ(t)mⱼ is the total KV cache size at time t.

  For stability, the arrival rate must satisfy:
  λ < 1/τ*,           (3)
  where τ* is the equilibrium iteration time.
  ```

---

### 0.4 具体例子支撑抽象概念 (Concrete Examples Anchor Abstraction)

**OR论文原理**：抽象机制（如threshold, coupling, fluid limit）对读者来说是"黑盒"。具体例子能让抽象概念"可视化"。

**AI的坏习惯**：
* 只陈述机制："The threshold mechanism enables decoupling."（什么是decoupling？）
* 不提供toy example或特例说明
* 把intuition留给读者自己"脑补"

**人类进阶策略**：
1. **Mini案例**：用2-3句话描述一个concrete scenario
2. **对比式例子**：展示baseline失败 vs 我们的方法成功
3. **图表辅助**：一张图胜过千言万语
4. **指向具体章节**："For instance, in the FCFS example (Section 4.1)..."

**修改案例**：
* *Bad (AI)*: "Our threshold mechanism prevents cascading failures."

* *Good (Human)*:
  ```
  Our threshold mechanism prevents cascading failures. For instance, consider the FCFS
  example in Section 4.1: when a long prompt enters during high load, FCFS must evict
  multiple short prompts to make room, triggering a cascade where evicted prompts
  re-enter the queue and cause further evictions. In contrast, our WAIT policy holds
  the long prompt at a designated stage until sufficient memory becomes available,
  breaking the cascade.
  ```

---

### 0.5 展示关键计算步骤 (Show Non-Trivial Steps)

**OR论文原理**：证明不是"所有步骤"，而是"关键转换"。跳过trivial步骤可以接受，但non-trivial步骤必须展示。

**AI的坏习惯**：
* "By simple algebra..." 然后跳过5行推导
* 关键不等式没有justification
* 用"it is easy to see"掩盖non-trivial步骤

**人类进阶策略**：
1. **where子句**："By the law of large numbers, X_n/n → λ, where λ is the arrival rate."
2. **by子句**："By Jensen's inequality, E[f(X)] ≥ f(E[X])."
3. **since子句**："Since M(t) is monotone increasing, we have..."
4. **关键步骤单独成行**：不要把3个推导挤在一行

**修改案例**：
* *Bad (AI)*: "By standard arguments, the system is stable when λ < λ*."

* *Good (Human)*:
  ```
  To show stability, we apply Foster-Lyapunov criterion with V(x) = ∑ⱼ xⱼ.
  By the renewal theorem, the drift satisfies:
  E[ΔV | x] = λ - 1/τ(x)
  Since τ(x) is increasing in x and τ* = τ(x*) is the equilibrium iteration time,
  we have E[ΔV | x] < 0 whenever λ < 1/τ*.
  Thus, the system is stable when λ < λ*.
  ```

---

### 0.6 双编码理论 (Dual Coding for Comprehension)

**认知原理**：人类大脑通过两个独立通道处理信息——语言通道和视觉通道。同时激活两个通道能显著提高理解和记忆（Paivio, 1986）。

**AI的坏习惯**：
* 长段文字描述复杂算法，无伪代码或流程图配合
* 公式推导孤立存在，无对应的图形直觉
* 抽象概念只有定义，无具体场景对照

**人类进阶策略**：
1. **公式+图表**：每个核心公式配一张直觉图
   - 例：稳定性条件 λ < λ* 配流量-容量对比图
2. **算法+流程图**：伪代码旁边放决策流程图
   - 例：IIS诊断算法配约束冲突树状图
3. **抽象+场景**：理论机制配实际应用场景
   - 例："decoupling" 配具体的FCFS vs WAIT对比案例

**OR论文应用案例**：

| 抽象概念 | 语言编码 | 视觉编码 |
|----------|----------|----------|
| IIS (最小不可行子集) | 定义 + 性质 | 约束网络图高亮冲突节点 |
| 修复策略 | 算法伪代码 | 决策树流程图 |
| 收敛性证明 | 数学推导 | 误差衰减曲线 |

**修改案例**：
* *Bad (AI)*: [页面充满公式和文字，无任何图表]

* *Good (Human)*:
  ```
  Figure 2 illustrates the IIS structure for a typical infeasible problem.
  The shaded nodes represent constraints in the IIS, while edges show
  variable dependencies. Formally, the IIS satisfies:

  IIS ⊆ C  such that  IIS is infeasible and ∀c ∈ IIS, IIS \ {c} is feasible.  (Eq. 3)

  As the figure shows, removing any single shaded constraint restores feasibility.
  ```

---

## 第一部分：逻辑与结构 (Logic & Structure)
*核心理念：写作即对读者注意力流（Flow of Attention）的管理 —— Gopen & Swan*

### 1.1 句首：语境锚点与旧信息 (Topic Position: The Anchor)

**认知原理**：读者的每一步阅读都需要建立在"已知"的基础上。句子的开头（Topic Position）是读者的"立足点"——必须放置旧信息或连接性信息，降低认知负担。

**AI的坏习惯**：
* 用全新的、复杂的概念开启句子（如 "Recent advancements in deep learning..."）
* 句子之间缺乏逻辑黏连，像在跨栏
* 每句话都从"新主语"开始，迫使读者不断reset

**人类进阶策略**：
1. **向后连接 (Link Backward)**：句子开头应包含上一句已经出现过的信息
2. **向前铺垫 (Context Forward)**：用开头限定本句范围
3. **主题词重复**：不要害怕重复关键术语（如"algorithm", "threshold"）
4. **框架式开头**：新话题用框架句引入（"Regarding X, two issues emerge."）

**修改案例**：
* *Bad (AI)*: "The algorithm was tested on three datasets. An accuracy of 95% was achieved."
  (两句话的主语完全不同，断裂感强)

* *Good (Human)*: "The algorithm was tested on three datasets. **These tests** revealed an accuracy of 95%."
  (用"These tests"承接上文)

**OR论文应用**：
* *Bad*: "Theorem 1 proves stability. The coupling technique is used in the proof."
* *Good*: "Theorem 1 proves stability. **To prove this**, we use a coupling technique."

---

### 1.2 句末：强调区与新信息 (Stress Position: The Stage)

**认知原理**：读者在读到句号时会自然停顿和呼气。这个生理停顿赋予了句末词汇天然的**强调权重**。必须把核心发现放在这里。

**AI的坏习惯**：
* 把重要结论埋在句子中间
* 用无意义的词结尾（"...is shown to be effective."）
* 浪费了最宝贵的"强调区"

**人类进阶策略**：
1. **修剪枝叶**：把状语、引用移到句首或句中
2. **重锤落下**：把核心发现推到句号前
3. **巧用分号/冒号**：创造"次级强调区"
4. **数字后置**：把惊人的数据放在句末

**修改案例**：
* *Bad (AI)*: "The metabolic rate decreased significantly when the temperature was lowered."
  (重点落在"lowered"上，但这只是条件)

* *Good (Human)*: "Lowering the temperature caused a significant decrease in **the metabolic rate**."
  (重点落在核心结果上)

**OR论文应用**：
* *Bad*: "Our algorithm improves throughput, which is significant."
* *Good*: "Our algorithm achieves a key improvement: **30% higher throughput**."

---

### 1.3 主谓紧邻：避免认知中断 (Subject-Verb Proximity)

**认知原理**：主语（S）和动词（V）是句子的引擎。读者读到主语后会悬着一颗心寻找动词。两者距离越远，读者的"工作内存"占用越高，理解越困难。

**AI的坏习惯**：
* 在主谓之间插入长达15-20词的定语从句
* 为了"精确"，堆砌多层修饰
* 让读者在找到动词前已经忘记了主语

**人类进阶策略**：
1. **缝合伤口**：把插入语移到句首做状语
2. **拆分句子**：过长的修饰语独立成句
3. **容忍度**：只有极短且关键的插入语才保留
4. **主谓距离**：理想 ≤ 7个词

**修改案例**：
* *Bad (AI)*: "The model, which utilizes a transformer-based architecture combined with reinforcement learning to optimize query latency, **outperforms** the baseline."
  (主谓之间隔了15个词)

* *Good (Human)*: "**Using** a transformer-based architecture combined with reinforcement learning, the model **outperforms** the baseline by optimizing query latency."
  (动词紧跟主语)

**OR论文应用**：
* *Bad*: "The threshold, which prevents prompts from entering when memory is insufficient, **enables** stability."
* *Good*: "**By preventing** prompts from entering when memory is insufficient, the threshold **enables** stability."

---

### 1.4 逻辑缺口：显性化逻辑连接 (Mind the Gap)

**认知原理**：作者极其熟悉自己的研究，容易产生"知识诅咒"（Curse of Knowledge），假设读者能自动补全A到B的逻辑跳跃。AI也常犯此病，因为它模仿的是成品论文，而非思考过程。

**AI的坏习惯**：
* 从数据直接跳到结论，不说明推理
* 省略"显而易见"的逻辑步骤
* 缺乏逻辑连接词（however, therefore, thus）

**人类进阶策略**：
1. **逻辑连接词**：不要吝啬 *however, therefore, consequently, thus, since*
2. **填补空白**：如果从"数据"跳到"结论"，中间加一句解释"因为这意味着..."
3. **反向检查**：让不熟悉的人读，问他们"为什么这样推导？"
4. **OR论文：每个"thus"都要有justification**

**修改案例**：
* *Bad (AI)*: "The system is stable when λ < λ*. We use a coupling argument."
  (为什么突然用coupling? 逻辑跳跃)

* *Good (Human)*: "The system is stable when λ < λ*. **To prove this**, we use a coupling argument that compares our policy to an idealized system."
  (明确说明coupling的目的)

---

## 第二部分：风格与修辞 (Style & Voice)
*核心理念：学术写作不仅是信息传输，更是风格展示。拒绝"僵尸名词"，拥抱鲜活动词 —— Helen Sword*

### 2.1 拒绝僵尸名词 (Zombie Nouns / Nominalization)

**症状与危害**：把动词变成名词（如 *perform an analysis* vs *analyze*），吸干句子活力，掩盖"谁在做什么"。OR论文尤其易患"名词堆砌症"。

**AI的坏习惯**：
* 极度依赖 -tion, -ment, -ance 结尾的名词
* 名词cluster："memory utilization optimization strategy implementation"
* 配合弱动词：is, was, conduct, perform, make

**人类进阶策略**：
1. **捕猎僵尸名词**：搜索 `-tion`, `-ment`, `-ence`, `-ance`
2. **用动词复活**：investigation → investigate, utilization → utilize → use
3. **以人/物为主语**：不用抽象概念做主语
4. **OR论文：让algorithm, policy, theorem做主语**

**修改案例**：
* *Zombie (AI)*: "We conducted an **investigation** of the system's **performance**."
* *Alive (Human)*: "We **investigated** how the system **performs**."

**OR论文应用**：
* *Zombie*: "The **implementation** of the threshold mechanism results in the **prevention** of cascading failures."
* *Alive*: "**Implementing** the threshold mechanism **prevents** cascading failures."

---

### 2.2 拥抱作者主权 (Claiming Agency with I/We)

**原则**：学术写作是"你"在说话。自信地使用"We"或"I"来申明论点、方法和贡献。被动语态会隐藏研究主体，削弱作者权威。

**AI的坏习惯**：
* 通篇被动语态："It was found that..."
* 无人称结构："This paper argues..."
* 为了"客观"而掩盖作者责任

**人类进阶策略**：
1. **关键处用"We"**：提出假设、解释方法、阐述结论
2. **区分事实与观点**：
   - 客观事实 → 被动/无人称："The solution was heated..."
   - 你的观点 → 主动："**We propose**...", "**We argue**..."
3. **OR论文：在贡献声明处必须用We**

**修改案例**：
* *Passive (AI)*: "It is suggested that this approach is superior."
* *Active (Human)*: "**We suggest** that this approach is superior."

**OR论文应用**：
* *Bad*: "It is shown that the threshold mechanism achieves stability."
* *Good*: "**We show** that the threshold mechanism achieves stability."

---

### 2.3 句子的节奏 (Sentence Rhythm)

**认知原理**：千篇一律的句子长度会催眠读者。节奏变化能保持注意力。短句用于强调，长句用于铺陈。

**AI的坏习惯**：
* 句子长度极其均匀（20-30词）
* 毫无波澜的白噪音
* 缺乏戏剧性的起伏

**人类进阶策略**：
1. **长短结合**：长句解释机制 + 短句强调结论
2. **短句的力量**：关键发现用短句（≤10词）
3. **读出声来**：检查是否monotonous
4. **OR论文：定理陈述后用短句总结意义**

**修改案例**：
* *Monotonous (AI)*:
  ```
  Although the previous methods failed to address the latency issue due to their
  reliance on synchronous processing, our asynchronous approach solves this problem
  effectively by decoupling the prefill and decode stages.
  ```
  (一句话50+词)

* *Rhythmic (Human)*:
  ```
  Previous methods rely on synchronous processing, which fails to address latency.
  Our asynchronous approach decouples prefill and decode stages. **It works.**
  ```
  (长-中-短节奏)

---

### 2.4 叙事钩子 (Narrative Hooks)

**原则**：即使是理论论文，也要"抓住"读者。开篇、标题、章节开头都是设计"钩子"的机会。

**AI的坏习惯**：
* 标题："A Study on [Topic]" 或 "Analysis of [Data]"
* 开篇："In recent years..." 或 "X plays a pivotal role..."
* 毫无吸引力

**人类进阶策略**：
1. **双节棍标题**："Catchy Phrase: Descriptive Subtitle"
2. **开篇策略**：
   - 提问式："Why do algorithms fail when memory is abundant?"
   - 陈述悖论："Counterintuitively, more memory can worsen performance."
   - 故事式："In 2023, a single query consumed energy equivalent to..."
3. **OR论文：用FCFS cascading failure作为motivating example**

**修改案例**：
* *Bad Title*: "Analysis of Scheduling Algorithms for LLM Inference"
* *Good Title*: "Preventing Cascading Failures: Fluid-Based Scheduling for LLM Inference"

---

### 2.5 信息性标题 (Informative Headings)

**原则**：章节标题不仅是导航工具，更是内容的预告和摘要。信息性标题让读者在浏览目录时就能获取核心信息。

**AI的坏习惯**：
* 使用通用、无信息量的标题
* 标题与内容脱节
* 遵循模板化的章节命名

**人类进阶策略**：
1. **结论前置**：把核心发现放入标题
2. **动词优于名词**：标题中使用动词形式
3. **数字具体化**：用数据支撑标题

**标题对比表**：

| 通用标题 (AI风格) | 信息性标题 (Human风格) |
|------------------|----------------------|
| "Results" | "8B Model Surpasses Frontier APIs on RR@5" |
| "Methods" | "Solver-in-the-Loop Evaluation with IIS Feedback" |
| "Related Work" | "From Static Translation to Dynamic Debugging" |
| "Discussion" | "Why Domain-Specific Training Outperforms Scale" |
| "Experiments" | "Evaluating 26 Models Across 7,200 Debug Episodes" |

**OR论文应用**：
* *Bad*: "Section 4: Analysis"
* *Good*: "Section 4: Training Reduces Steps from 3.7 to 2.3 Without Sacrificing Accuracy"

---

## 第三部分：受众意识 (Audience Awareness)
*核心理念：写作即教学。目标不是展示你多聪明，而是帮助读者变聪明 —— Stanford Handbook*

### 3.1 回答"那又怎样？" (The "So What?" Question)

**核心问题**：对于每一个技术细节，都要替读者问："So what? 为什么我要知道这个？这对解决核心问题有何帮助？"

**AI的坏习惯**：
* 罗列技术参数，不解释意义
* 沉浸在细节中，忘记大图景
* 假设读者自己能理解重要性

**人类进阶策略**：
1. **功能性意义**：每个技术细节后加半句话解释其作用
2. **连接大图景**："This is important because..."
3. **OR论文：每个定理后说明implications**

**修改案例**：
* *Bad (AI)*: "We set the learning rate to 0.01."
* *Good (Human)*: "We set the learning rate to 0.01 **to prevent the model from overshooting local minima during early training**."

**OR论文应用**：
* *Bad*: "Theorem 1 shows that E[L] ≤ E[L*] + ε."
* *Good*: "Theorem 1 shows that E[L] ≤ E[L*] + ε, **which means our policy achieves near-optimal latency while preventing evictions**."

---

### 3.2 路标与元话语 (Signposting & Metadiscourse)

**策略**：像导游一样，告诉读者"我们现在在哪里，要去哪里"。OR论文往往很长，路标至关重要。

**AI的坏习惯**：
* 直接开始技术推导，不说明章节roadmap
* 没有"preview"和"recap"
* 读者迷失在细节中

**人类进阶策略**：
1. **章节预告**："This section first addresses X, then analyzes Y, and finally proves Z."
2. **章节回顾**："Having established X, we now turn to Y..."
3. **区分主次**："The key insight is... (The technical details are in Appendix A.)"
4. **OR论文：证明前给proof roadmap**

**修改案例**：
* *Bad (AI)*: [直接开始证明，50行推导]
* *Good (Human)*:
  ```
  We now prove Theorem 1 in three steps. First, we construct a coupling between
  our policy and the idealized system (Lemma 2). Second, we show that the coupling
  gap vanishes asymptotically (Lemma 3). Finally, we combine these results to
  establish stability.

  **Step 1: Coupling Construction.** [证明开始]
  ```

---

### 3.3 祖母测试 (The Grandmother Test)

**原则**：想象向一位聪明但非专业的人（如你的祖母）解释你的研究。如果她能理解这句话的要点，它就通过了"祖母测试"。

**适用场景**：
* Introduction 的前两段
* Abstract 全文
* 每个章节的开头句
* Conclusion 的核心主张

**AI的坏习惯**：
* 在introduction就使用未定义的专业术语
* 假设读者已经知道问题的重要性
* 用复杂句子表达简单意思

**人类进阶策略**：
1. **先日常语言，后专业术语**：
   - ❌ "We address the IIS diagnosis problem in infeasible LPs."
   - ✅ "When a mathematical model has conflicting requirements, we need to find which requirements conflict. This is called IIS diagnosis."

2. **类比优先**：用读者熟悉的概念解释陌生概念
   - "IIS is like finding the smallest set of traffic rules that, together, make it impossible to reach your destination."

3. **一句话总结**：能否用一句话向非专家解释？

**OR论文应用**：

| 专业版本 | 祖母版本 |
|---------|---------|
| "We propose a novel benchmark for LLM-based constraint relaxation in infeasible MILPs." | "We created a test to see if AI can fix mathematical models that have impossible requirements." |
| "The IIS mechanism enables precise root cause identification." | "This tool tells us exactly which rules are fighting each other." |

**测试流程**：
1. 写完一段后，大声读出来
2. 问自己：一个聪明的本科新生能理解吗？
3. 如果有超过2个未解释的术语，重写

---

## 第四部分：AI模式检测 (AI Pattern Detection)
*识别AI生成文本的特征指纹，实现精准去AI化*

### 4.1 词级AI标记 (Word-Level AI Markers)

AI生成的文本有明显的词汇偏好。以下是高频AI标记词及其替换策略：

| 类别 | AI模式词 | 替换策略 |
|------|----------|----------|
| **模糊限定** | potentially, arguably, essentially, fundamentally | 删除，或用具体条件替代 |
| **空洞量词** | significantly, substantially, considerably, remarkably | 用数字："+9.1%", "by 3× speedup" |
| **过度吹嘘** | novel, groundbreaking, state-of-the-art, cutting-edge | "new", "first to our knowledge", "among compared methods" |
| **学术填充** | notably, importantly, interestingly, crucially | 删除，或整合到主句 |
| **假谦虚** | relatively, somewhat, fairly, rather | 删除，直接陈述 |

**OR论文高危词汇**：
```
❌ "Our approach significantly outperforms existing methods."
✅ "Our approach achieves 95.3% RR@5, compared to 82.9% for o1."

❌ "The IIS mechanism plays a crucial role in enabling effective debugging."
✅ "IIS computation identifies the minimal conflicting constraint set in O(n) calls."

❌ "We propose a novel and comprehensive benchmark."
✅ "OR-Debug-Bench includes 5,000+ problems across 9 error types."
```

---

### 4.2 短语级AI模式 (Phrase-Level AI Patterns)

AI生成文本有明显的短语模板。识别并替换这些"AI指纹"：

**开篇废话模式**：
| AI模式 | 问题 | 修复 |
|--------|------|------|
| "In recent years, X has received increasing attention..." | 空洞、老套 | 直接说X是什么，为什么重要 |
| "X plays a pivotal/crucial/vital role in Y..." | 无信息量 | 删除，直接进入主题 |
| "With the rapid development of..." | 陈词滥调 | 用具体里程碑事件 |

**动作描述模式**：
| AI模式 | 问题 | 修复 |
|--------|------|------|
| "This paper delves into / explores / investigates..." | 弱动词 | "We show that..." / "We prove..." |
| "We propose a novel framework that leverages..." | 空洞+吹嘘 | 直接说框架做什么 |
| "Our approach aims to address the challenge of..." | 绕弯子 | "We solve X by Y" |

**结果陈述模式**：
| AI模式 | 问题 | 修复 |
|--------|------|------|
| "The results demonstrate that..." | 被动、弱 | "X achieves Y" |
| "It can be observed that..." | 无主语 | 指明谁观察到什么 |
| "This finding suggests that..." | 回避直接结论 | 直接说结论是什么 |

**OR论文应用**：
```
❌ "In recent years, the application of large language models to operations
    research has received increasing attention from researchers."

✅ "Large language models can now formulate optimization problems from natural
    language, but they cannot debug their own mistakes."
```

**隐蔽AI短语（GPT-5.2审计发现）**：

这些短语不像上面那么明显，但经过GPT-5.2全文审计后确认为AI偏好表达：

| AI短语 | 替换 | 备注 |
|--------|------|------|
| fills this gap | addresses this gap | "填补空白"是经典AI句式 |
| at scale | （删除）或"on many problems" | 工业/AI陈词滥调 |
| enabling X | allowing X | "enabling"是AI样板开头 |
| extends naturally to | can be extended to | "naturally"模糊+AI感 |
| proves decisive | is critical / is key | 戏剧化修辞 |
| lifts X by Y | raises X by Y | "lift"有促销感 |
| outpacing X | exceeding X / beating X | 新闻体 |
| exposes X | reveals X / shows X | 过重 |
| carry/carries implications | inform X / have implications for | "carry"多余；直接说影响什么 |
| training recipe | training procedure | MSOM偏好正式术语 |
| This appendix reports [非实证内容] | presents（用于规格/提示/定义）；reports仅用于实证结果 | 动词选择取决于内容类型 |
| open(s) a new avenue | （删除或用具体描述改写） | 陈词滥调 |
| remarkable ability / impressive capability | （删除或具体说明做了什么） | AI吹嘘模式 |
| The central/key empirical finding [that/---]... | （删除框架；直接陈述发现） | 元话语框架 |
| Following \citet{X}, Y [verb]s... | Y \citep{X} [verb]s...（内联引用，不要宣布它） | 元话语引用导入 |
| X, not merely/just a Y | （解释具体WHY："X isolates A from B"） | 模糊论证 |
| The implication is clear: [claim] | （删除框架；直接陈述） | 元话语框架 |
| with direct implications for [field] | （删除；改写为具体可操作的陈述） | AI空洞收尾 |
| Despite these limitations, X consistently... | （改写为独特的收尾句，非样板） | AI样板结尾 |

**检测命令**：
```bash
grep -Ei "fills (this|the) gap|at scale|enabling|extends naturally|proves decisive|outpacing|lifts.*by|carry.*implications|training recipe|open.*avenue|remarkable ability|impressive capability|implication is clear" paper.tex
```

---

### 4.3 结构性AI指纹 (Structural AI Fingerprints)

AI生成文本的结构也有明显特征：

**1. 句长均匀性 (Uniform Sentence Length)**

| 特征 | AI典型 | Human典型 |
|------|--------|----------|
| 句长标准差 | < 5 词 | > 8 词 |
| 短句 (< 10词) 比例 | < 5% | > 15% |
| 长句 (> 40词) | 常见 | 罕见 |

**检测方法**：计算连续5句的词数，如果差异 < 3词，可能是AI生成。

**2. 单调过渡词 (Monotonous Transitions)**

AI喜欢使用固定的过渡词序列：
```
❌ "Furthermore... Moreover... Additionally... In addition..."
✅ 混合使用：however, yet, but, still, by contrast, on the other hand
```

**检测方法**：如果连续3个段落都以 "Furthermore/Moreover/Additionally" 开头，标记为AI风格。

**3. 三联形容词堆砌 (Triple-Adjective Stacking)**

AI喜欢用三个形容词修饰名词：
```
❌ "a comprehensive, robust, and scalable framework"
❌ "an efficient, accurate, and reliable algorithm"
✅ 只保留最关键的一个形容词，其他用数据证明
```

**4. 编号发现标签 (Numbered Finding Labels)**

AI生成的实验章节几乎总是使用 "Finding N:", "Result N:", "Insight N:" 加粗标签开头，有时还带括号注释如 "(Central result)"。这不是学术论文惯例——这是AI脚手架。

```
❌ AI模式 (Finding标签):
\textbf{Finding 5 (Central result): Trained 8B models outperform all API models.}
The trained Qwen pipeline reaches 81.7% RRR...

\textbf{Finding 6: Phase 1 is the bottleneck.}
API models average only 27.6% RR...

✅ Human模式 (叙事散文):
The trained Qwen pipeline reaches 81.7% RRR, beating
the best API models by 39.5 percentage points.

API models average only 27.6% RR on the supply chain
task versus 97.2% for trained models---a gap of nearly
70 percentage points.
```

**相关模式**：每段都以 `\textbf{加粗断言.}` 开头也是AI指纹。人类写作的段落开头方式应当多样——有的从数据开始，有的从背景开始，有的从对比开始。

**检测命令**：
```bash
grep -En "\\\\textbf\{(Finding|Result|Insight|Observation) [0-9]" paper/*.tex
```

**5. 对称枚举强迫症 (Symmetric Enumeration / Itemize Overuse)**

AI倾向于把所有内容组织成对称的列表，过度使用`itemize`/`enumerate`环境：

```latex
❌ 过度列表化 (AI典型模式):
Our contributions are:
\begin{itemize}
    \item We propose a benchmark.
    \item We evaluate 26 models.
    \item We show training improves performance.
\end{itemize}

✅ 自然段落流 (Human写法):
We make three contributions. First, we construct OR-Debug-Bench,
a benchmark of 7,200 debugging episodes across 9 error types.
Second, we evaluate 26 models spanning four categories...
Third, we demonstrate that domain-specific training improves
RR@5 from 86.2% to 95.3%.
```

**何时使用列表**：
- ✅ 4个以上平行结构的独立项目
- ✅ 算法步骤或操作流程
- ✅ 需要对比的多维度表格

**何时改用prose**：
- ❌ 2-3个项目 → 用inline枚举: "first, ...; second, ...; third, ..."
- ❌ 需要解释的项目 → 融入段落并添加transition
- ❌ 连续多节都是bullet points → 转换为带过渡的段落

---

### 4.4 量化诊断指标 (Quantitative Diagnostic Metrics)

使用以下指标量化文本的"AI程度"：

| 指标 | AI典型值 | Human典型值 | 计算方法 |
|------|---------|------------|----------|
| **句长标准差** | < 5 词 | > 8 词 | std(sentence_lengths) |
| **主动/被动比** | < 0.3 | > 0.5 | active_verbs / total_verbs |
| **模糊词密度** | > 3% | < 1% | hedging_words / total_words |
| **第一人称频率** | < 2/节 | > 5/节 | count("we"/"our") per section |
| **短句比例** | < 5% | > 15% | sentences < 10 words / total |
| **列表密度** | > 2/页 | < 1/页 | itemize_envs / page_count |
| **过渡词重复率** | > 50% | < 30% | repeated_transitions / total_transitions |

**自检脚本逻辑**：
```python
def ai_score(text):
    scores = {
        'sentence_std': 1 if std(sentence_lengths) < 5 else 0,
        'passive_ratio': 1 if passive_verbs/total_verbs > 0.7 else 0,
        'hedging_density': 1 if hedging_words/total_words > 0.03 else 0,
        'first_person': 1 if we_count < 2 else 0,
    }
    return sum(scores.values()) / len(scores)  # 0-1, 越高越像AI
```

**OR论文特定检测**：
```
搜索命令:
grep -E "significant|substantial|novel|comprehensive|crucial|pivotal" paper.tex
grep -E "In recent years|plays a role|delves into|leverages" paper.tex
grep -E "Furthermore.*Moreover.*Additionally" paper.tex
```

---

### 4.5 生僻词/花式词汇检测 (Uncommon Vocabulary Detection)

**核心原则**：目标是**词汇多样性**，而非一律替换。偶尔用一次 "analogous" 没问题——问题是同一个花式词反复出现，或明明有更自然的表达却偏要用拉丁词。

**检测逻辑**：扫描以下词汇。若出现频率高或读起来不自然，替换为更简单的选项：

| 花式词 | 替换建议 | 备注 |
|--------|---------|------|
| taxonomy | classification, types | 用户明确标记：太CS化 |
| pluggable | replaceable, modular | CS系统术语 |
| instantiate | implement, build, apply | 非常CS化 |
| conflates | mixes, blurs | 标准词（"错误地混为一谈"），批评时OK；但prefer simpler |
| curated / curation | filtered, selected / screening | 流行AI buzzword |
| ecological validity | realism | 心理学术语，OR中罕见 |
| screens out | filters out | 两者都是标准词；按上下文选择 |
| synthesized (非化学) | consolidated, compiled | 在非化学语境下略做作 |
| constitutes | is, represents | 标准正式动词；"is"更轻 |
| composed (sequentially) | combined | 花式说法 |
| close the gap/loop, bridge | connect, link | 听起来promotional/不neutral |
| amenable | suitable for, allows | OR标准词（"amenable to analysis"）；prefer lighter替代 |
| delineate | describe, outline | 不常见 |
| paradigm | approach, framework | 过度使用的buzzword |
| juxtapose | compare, contrast | 做作 |
| elucidate | explain, clarify | 做作 |
| preclude | prevent, rule out | OR/数学标准词（"逻辑上不可能"）；语义相同时prefer "prevent" |
| underpins | supports, grounds | 标准学术隐喻；prefer "supports"。注意勿用"enables"（本身被标记） |
| corroborate | confirm, support | 实证章节标准词；prefer "confirm"更轻 |
| pivotal | key, central, important | AI偏好词 |
| operationalize | apply, implement, put into practice | 行话 |
| pinpointing | identifying | 不正式/AI感 |
| outpacing | exceeding, beating | 新闻体 |
| lift/lifts (性能) | raise/raises, increase | 促销感 |
| compresses (隐喻) | distills (ML标准), transfers | 隐喻；"distills"用于SFT更准确 |
| exposes (= reveals) | reveals, shows | 过重 |
| proves decisive | is critical, is key | 戏剧化/修辞感 |
| harness | use | 陈词滥调隐喻 |
| foster | encourage | 花式 |
| myriad | many | 做作 |
| traverse | cross, cover | 花式 |
| embark | start, begin | 花式 |
| encapsulate | capture, contain | 花式 |
| facet | aspect, side | 花式 |
| pipeline（非系统描述） | procedure, approach, method | 描述训练/分析时用procedure；描述系统架构时OK |
| saboteur pipeline | saboteur module | "pipeline"暗示CS基础设施；"module"更适合OM |
| steers the agent | （用主动语态改写："the agent therefore..."） | 模糊agency；视上下文决定 |
| merit（动词） | warrant, deserve | 略正式；名词形式OK |
| aggressive（非技术语境） | larger, stronger | 非技术用法听起来不正式；"aggressive pricing"（OM术语）OK |

**保留不改（OR/学术标准用词）**：
misspecified, perturbation, incurs, domain-agnostic, framework, heuristic, diagnose, operationally rational, marginal, iterate/iterative, benchmark, echelon, infeasible, oracle, formalize, analogous, binding constraint（LP双关）, cascading, self-contained, sharp（如"sharp boundary"）, empirical, research streams, exhibit, deterministic, stochastic, ablation, distillation, rollout, pipeline（当描述实际软件系统时）, saboteur（论文既定术语）, decomposes（标准术语，用于two-phase拆分）

**检测命令**：
```bash
grep -Ei "taxonomy|pluggable|instantiate|conflates|curated|curation|ecological validity|screens out|constitutes|close the (gap|loop)|bridge.*gap|amenable|delineate|paradigm|juxtapose|elucidate|preclude|underpins|corroborate|pivotal|operationalize|pinpointing|outpacing|harness|foster|myriad|traverse|embark|encapsulate|facet|proves decisive" paper.tex
```

---

## 第五部分：OR论文专项案例 (OR-Specific Examples)

### IIS描述的去AI化

**Bad (AI)**:
```
The IIS mechanism plays a crucial role in enabling effective debugging by
providing comprehensive information about the constraint conflicts. This
novel approach significantly enhances the diagnostic capabilities of our
framework, allowing for more robust and efficient problem resolution.
```

**Good (Human)**:
```
IIS computation returns the minimal subset of constraints that cannot be
simultaneously satisfied---exactly what we need to identify the root cause.
For a model with 50 constraints, IIS typically contains 3-5, reducing the
search space by 90%.
```

---

### Benchmark贡献的去AI化

**Bad (AI)**:
```
We propose a novel and comprehensive benchmark that significantly advances
the field of LLM-based optimization debugging. Our groundbreaking approach
encompasses a wide range of problem types and difficulty levels, providing
researchers with an unprecedented resource for evaluation.
```

**Good (Human)**:
```
OR-Debug-Bench includes 5,000+ problems across 9 error types, with solver-
verified ground truth for each. Problems range from simple bound conflicts
(fixable in 1 step) to complex multi-constraint interactions (requiring
5+ steps). We release all code and data at [URL].
```

---

### 实验结果的去AI化

**Bad (AI)**:
```
The experimental results demonstrate that our proposed method significantly
outperforms existing baselines across all metrics. Notably, the improvements
are particularly pronounced on challenging instances, showcasing the
robustness and effectiveness of our approach.
```

**Good (Human)**:
```
Qwen3-8B-GRPO achieves 95.3% RR@5, compared to 82.9% for o1 and 81.8% for
GPT-5.2. The gap widens on hard problems (error types E-I): our model
maintains 89% RR@5 while frontier APIs drop to 65%. Table 3 shows full
results across all 26 models.
```

---

## 终极检查清单 (The Ultimate De-AI Checklist)

在提交任何论文前，请**按顺序**执行以下10步workflow：

### Step 0: OR论文核心检查 ✓
- [ ] **段落功能**：每段只有一个功能（定义/陈述/证明/讨论）？
- [ ] **直觉→严格**：定理/公式前有intuition铺垫吗？
- [ ] **数学文字平衡**：连续公式间有文字连接吗？
- [ ] **具体例子**：抽象机制有concrete example支撑吗？
- [ ] **计算步骤**：non-trivial步骤有where/by/since解释吗？

### Step 0.5: AI模式扫描 ✓ (NEW)
- [ ] **模糊词扫描**：搜索 `significant|substantial|potential|arguably|essential`
- [ ] **废话短语扫描**：搜索 `"worth noting"|"plays a role"|"delves into"|"leverages"`
- [ ] **吹嘘词扫描**：搜索 `novel|remarkable|comprehensive|state-of-the-art|groundbreaking`
- [ ] **句长检查**：是否存在连续3+句子长度相同（±3词）？
- [ ] **过渡词检查**：是否存在连续3段以 Furthermore/Moreover/Additionally 开头？

### Step 1: 逻辑流检查 (Flow Check)
- [ ] **旧信息先行**：每句话开头（前5-6个词）是否与上一句有联系？
- [ ] **新概念铺垫**：如果开头是新概念，是否做了框架式引入？
- [ ] **主题词重复**：关键术语是否适当重复以保持连贯？

### Step 2: 强调点检查 (Stress Check)
- [ ] **新信息断后**：每句话结尾是否是本句最重要的信息？
- [ ] **避免弱结尾**：句末不是"using our method"或"is effective"等弱词？
- [ ] **数字后置**：关键数据放在句末强调位置？

### Step 3: 僵尸词猎杀 (Zombie Hunt)
- [ ] **搜索-tion/-ment**：能否改回动词形式？
- [ ] **名词cluster**：是否存在3个以上名词连用？
- [ ] **弱动词替换**：is/was/conduct/perform能否换成强动词？

### Step 4: 主动权检查 (Agency Check)
- [ ] **搜索被动语态**：was/were + verb中哪些该改主动？
- [ ] **关键处用We**：提出观点、解释方法处是否用了We？
- [ ] **区分事实观点**：客观事实用被动，你的贡献用主动？

### Step 5: 节奏感检查 (Rhythm Check) (ENHANCED)
- [ ] **读出声**：是否存在连续3个长度相似的句子？
- [ ] **句长标准差**：每段的句长std > 5词？
- [ ] **短句存在**：每段至少一句 < 10词？
- [ ] **长句控制**：没有超过40词的句子？
- [ ] **过渡词多样**：没有连续3个 Furthermore/Moreover/Additionally？

### Step 6: 宏观连接检查 (So What Check)
- [ ] **技术细节意义**：每个参数/公式是否说明了"为什么重要"？
- [ ] **定理implications**：每个定理后是否讨论了意义？
- [ ] **大图景连接**：读者能否理解这段对全文的贡献？

### Step 7: 具体化检查 (Concreteness Check)
- [ ] **替换空洞形容词**：efficient/robust/important是否换成了具体描述？
- [ ] **数字支撑**："significantly better"是否改为"30% faster"？
- [ ] **例子补充**：抽象概念是否配有concrete scenario？

### Step 8: 路标完整性 (Signpost Check)
- [ ] **章节预告**：每个section开头是否说明了roadmap？
- [ ] **证明roadmap**：复杂证明前是否有proof sketch？
- [ ] **过渡句**：章节间是否有"Having established X, we now..."？

### Step 9: 双编码检查 (Dual Coding Check)
- [ ] **公式+图表**：核心公式是否有对应的直觉图/示例？
- [ ] **算法+流程**：伪代码是否有可视化流程图？
- [ ] **理论+场景**：抽象概念是否绑定到具体应用场景？
- [ ] **祖母测试**：Introduction前两段能否让非专家理解？

### Step 10: 宏观架构检查 (Macro Architecture Check) (NEW)
- [ ] **符号工程**：符号是否携带语义？风格是否全篇一致？下标是否最小化？
- [ ] **Related Work**：是否分类而非罗列？是否有Comparison Table？你的delta是否显性？
- [ ] **Ablation**：每个组件的贡献是否单独验证？
- [ ] **Sensitivity**：关键参数扰动下结果是否stable？
- [ ] **Failure Cases**：是否诚实讨论了局限性？
- [ ] **Venue匹配**：写作节奏是否符合目标venue（CS快节奏 vs OR深挖掘）？
- [ ] **贡献清单**：每条贡献是否属于Model/Theory/Algorithm/Experiment之一？

---

## 快速检测命令 (Quick Detection Commands)

在终端运行以下命令快速检测AI模式：

```bash
# 模糊词检测
grep -En "significant|substantial|potentially|arguably|essentially" paper/*.tex

# 废话短语检测
grep -En "worth noting|plays a (crucial|vital|key) role|delves into|leverages" paper/*.tex

# 吹嘘词检测
grep -En "novel|groundbreaking|state-of-the-art|comprehensive|unprecedented" paper/*.tex

# 被动语态检测 (粗略)
grep -En "is (shown|demonstrated|proposed|achieved)|are (presented|discussed)" paper/*.tex

# AI开篇模式
grep -En "In recent years|With the (rapid |)development of|has (received|attracted) (increasing |)attention" paper/*.tex
```

---

## 第六部分：宏观架构与审稿人心理 (Macro Architecture & Reviewer Psychology)
*从微观句法到宏观战略：击败90%投稿的秘诀*

### 6.1 符号工程学 (Notation Engineering)

**认知原理**：OR论文伴随厚重的数学符号。每个符号都是读者的"内存占用"——设计不当会让读者在"符号查找表"中迷失。

**三大原则**：

| 原则 | 错误示范 | 正确示范 |
|------|----------|----------|
| **语义记忆 (Semantic Mnemonics)** | 用 x, y, z 表示库存、价格、时间 | I (Inventory), P (Price), T (Time) |
| **层级一致 (Hierarchical Consistency)** | 风格混用 | 集合𝒮,𝒞; 参数α,β; 变量x,y; 随机X,Y |
| **极简主义 (Minimalism)** | x_{i,j,k}^{(t,s)} | 避免双/三下标，除非绝对必要 |

**符号设计清单**：
- [ ] 符号本身是否携带语义？（λ表示arrival rate，μ表示service rate）
- [ ] 全篇符号风格是否统一？（检查：集合、参数、变量、随机变量）
- [ ] 每个下标是否必要？能否合并或省略？
- [ ] 关键符号是否在首次出现时明确定义？

**OR论文案例**：
```
❌ Let x denote the inventory level, y the price, and z the lead time.
   (三个无关联的字母，读者需要不断查表)

✅ Let I denote inventory, P price, and L lead time.
   (语义记忆：I=Inventory, P=Price, L=Lead time)
```

---

### 6.2 Related Work的圈地写法 (Related Work as Land Grab)

**战略定位**：Related Work 不是文献综述——它是**圈地运动**。目标是：(1) 证明你了解领域，(2) 精准定位你的贡献差异。

**Laundry List vs. Clustered Positioning**：

| 新手写法 (Laundry List) | 高手写法 (Clustered) |
|-------------------------|----------------------|
| "Smith did A. Jones did B. Lee did C." | "Approaches fall into three categories..." |
| 逐篇罗列，无组织 | 先分类，再定位 |
| 读者不知道你和他们有何不同 | 明确指出你的 delta |

**模板结构**：
```
[领域分类句] Approaches to dynamic pricing fall into three categories:
strictly parametric [1,2], purely data-driven [3,4], and hybrid models [5,6].

[定位句] Our work belongs to the third category but differs in two key aspects:
(1) we handle non-stationary demand, and (2) we provide regret bounds.

[表格] Table 1 compares our method with the most related works.
```

**Comparison Table 必杀技**：

| Method | Non-stationary | Regret Bound | Real Data | Code |
|--------|----------------|--------------|-----------|------|
| Smith [1] | ✗ | ✓ | ✗ | ✗ |
| Jones [2] | ✓ | ✗ | ✓ | ✗ |
| **Ours** | **✓** | **✓** | **✓** | **✓** |

**原则**：你的那一栏应该是**全勾**（至少在你强调的特性上）。

---

### 6.3 防御性实验设计 (Defense-Oriented Experiments)

**审稿人心理**：Reviewer #2 不会问"你的方法好不好"，而是问"**为什么**好"和"**什么时候**不好"。

**三层防御体系**：

| 实验类型 | 目的 | CS/OR期望 |
|----------|------|-----------|
| **Ablation Studies** (消融实验) | 证明每个组件non-trivial | CS会议**必须** |
| **Sensitivity Analysis** (敏感性分析) | 证明参数扰动下依然robust | OR期刊**期望** |
| **Failure Case Analysis** (失败案例) | 诚实讨论局限性 | 增加可信度 |

**Ablation设计模板**：
```
如果你的算法 = A + B + C，你需要：
- Full model (A+B+C)
- w/o A (B+C)
- w/o B (A+C)
- w/o C (A+B)
- Baseline (none)
```

**Sensitivity Analysis 关键参数**：
- 输入参数估计误差（±10%, ±20%）
- 超参数选择（learning rate, threshold）
- 数据规模（n=100, 1000, 10000）

**失败案例的诚实写法**：
```
❌ "Our method consistently outperforms baselines."
   (过于完美，审稿人不信)

✅ "Our method outperforms baselines on 8 of 9 problem types.
   On Type F (highly non-linear constraints), performance degrades to
   baseline level, likely due to our linear approximation in Step 2.
   Addressing this limitation is left for future work."
   (诚实+解释+未来方向)
```

---

### 6.4 Venue方言切换 (Venue-Specific Dialects)

**核心洞察**：CS会议（NeurIPS, ICML, ICLR）和OR期刊（MS, OR, MSOM）的写作**节奏完全不同**。

**对比表**：

| 维度 | CS会议 (Fast Paced) | OR期刊 (Deep Dive) |
|------|---------------------|---------------------|
| **Hook位置** | 第一页必须讲清Story | 可以慢慢展开motivation |
| **Figure 1** | 系统架构图或核心概念图（必须！） | 可以稍后出现 |
| **证明位置** | 主文只放Proof Sketch，细节扔Appendix | 完整证明expected |
| **强调重点** | 算法性能、收敛速度 | 结构性质(convexity, threshold structure) |
| **商业场景** | 简短 | 可以展开讨论managerial insights |
| **页数限制** | 严格（8-9页+appendix） | 灵活（30-50页常见） |

**CS会议生存法则**：
1. **Figure 1 = 论文灵魂**：很多人只看图，确保Figure 1自成故事
2. **Abstract要有数字**："+9.1% over baseline"比"significantly better"有力
3. **Appendix是救命稻草**：所有细节扔进去，主文保持精炼

**OR期刊生存法则**：
1. **Structural Properties至上**：审稿人想看convexity, submodularity, threshold structure
2. **Managerial Insights必须**：每个理论结果后讨论对管理实践的启示
3. **数值实验要真实数据**：纯合成数据会被质疑relevance

---

### 6.5 贡献声明清单 (Contribution Statement Checklist)

**核心要求**：Abstract和Introduction结尾的贡献列表，每一条必须**可验证**。

**四维度检验**：

| 维度 | 检验问题 | 示例 |
|------|----------|------|
| **Model** | 提出了新问题还是新刻画？ | "First benchmark for LLM debugging in OR" |
| **Theory** | 证明了新Bound还是新结构性质？ | "Prove threshold policy is asymptotically optimal" |
| **Algorithm** | 更快、更准、还是更Robust？ | "3× speedup with same accuracy" |
| **Experiment** | 真实数据验证还是构建新Benchmark？ | "Evaluate 26 models on 7,200 episodes" |

**贡献陈述对比**：

| 弱陈述 (AI风格) | 强陈述 (Human风格) |
|-----------------|---------------------|
| "We propose a novel method" | "We prove that threshold policy minimizes latency under memory constraints" |
| "We conduct extensive experiments" | "We evaluate 26 models across 9 error types on 7,200 debug episodes" |
| "We provide comprehensive analysis" | "We derive closed-form bounds for convergence rate (Theorem 2)" |
| "Our method significantly outperforms" | "Our 8B model achieves 95.3% RR@5, surpassing o1's 82.9%" |

**自检流程**：
1. 写完贡献列表后，对每一条问：**这是Model/Theory/Algorithm/Experiment中的哪一个？**
2. 如果答不上来，说明这条贡献**不够具体**
3. 每一条贡献应该能用**一个数字或一个定理**支撑

---

## 结语

> **"Complexity of thought need not lead to impenetrability of expression."**
> *复杂的思想不代表必须用晦涩的表达。*

优秀的OR论文能做到数学严谨与叙事清晰的完美平衡。使用本指南的20+原则和10步检查清单，让你的论文既有理论深度，又有可读性——这才是真正的学术卓越。

**v5.0 新增内容**：
- 第六部分：宏观架构与审稿人心理
  - 6.1 符号工程学 (Semantic Mnemonics, Hierarchical Consistency)
  - 6.2 Related Work圈地写法 (Comparison Table必杀技)
  - 6.3 防御性实验设计 (Ablation, Sensitivity, Failure Cases)
  - 6.4 Venue方言切换 (CS Conference vs OR Journal)
  - 6.5 贡献声明清单 (Model/Theory/Algorithm/Experiment四维度)

**v4.0 内容**：
- 第四部分：AI模式检测（词/短语/结构/量化指标）
- 0.6 双编码理论
- 2.5 信息性标题
- 3.3 祖母测试
- Step 0.5 AI模式扫描
- Step 9 双编码检查
- OR论文专项案例
- 快速检测命令

---

---

## 第七部分：自动化检测 (Automated Detection via Vale MCP)

以上所有AI模式检测规则已编码为Vale YAML规则，位于 `.vale/styles/AcademicDeAI/`。

### 已编码的10条规则

| 规则文件 | 检测内容 | 对应章节 |
|---------|---------|---------|
| `AIRoadmap.yml` | "We first X, then Y" 等程序化路标 | §4.2 |
| `AIForbiddenWords.yml` | significantly, novel, comprehensive 等 | §4.1 |
| `AIOpenings.yml` | "In recent years", "plays a crucial role" 等 | §4.2 |
| `AIActionVerbs.yml` | "This paper delves into" 等弱动词短语 | §4.2 |
| `AIResults.yml` | "The results demonstrate that" 等被动结果句 | §4.2 |
| `ZombieNouns.yml` | "implementation of" 等僵尸名词+弱动词 | §2.1 |
| `TransitionMonotony.yml` | Furthermore/Moreover/Additionally 连续使用 | §4.3 |
| `TripleAdjective.yml` | "comprehensive, robust, and scalable" 三联形容词 | §4.3 |
| `EmptyHedges.yml` | "It is worth noting that" 等空洞对冲 | §4.1 |
| `FormulaParagraphs.yml` | `\paragraph{Interpretation.}` 等机械化结果解读标题 | §4.3 |

**注意**: `FormulaParagraphs.yml` 检测在每个定理/命题后重复使用 `\paragraph{Interpretation.}` 或 `\paragraph{Discussion.}` 的模式。这种机械化标题是AI生成文本的典型特征。正确做法：用 `\smallskip` + 直接文本替代，让解读自然融入行文。

### 使用方法

**自动调用**：运行 `/polish-paper` 时会在 Step 1.1 自动调用 Vale 进行预扫描。

**手动运行**：
```bash
vale paper/energy_optimal_af/or_draft/sections/*.tex
```

**配置文件**：`.vale.ini` (项目根目录)

### MCP集成

Vale MCP Server 已注册在 `.mcp.json`。详见 `.claude/commands/_shared/mcp_writing_tools.md`。

---

**文档版本**：v8.0 (MSOM Polish Sync Edition)
**更新日期**：2026-02-23
**适用范围**：Operations Research理论论文 + CS/OR交叉领域 (NeurIPS/ICML/MS/OR)

**v8.0 更新**：
- §4.2 新增12个隐蔽AI短语（carry implications, training recipe, This appendix reports, open avenue, remarkable ability, central finding, Following \citet, not merely, implication is clear, direct implications, Despite limitations）
- §4.5 新增5个花式词（pipeline非系统, saboteur pipeline, steers, merit动词, aggressive非技术）
- §4.5 保留不改列表修正：移除"steers"（视上下文），新增pipeline(系统), saboteur, decomposes
- 检测命令扩展

**v7.0 更新**：
- §4.2 新增"隐蔽AI短语"表（GPT-5.2全文审计发现的8个短语）
- §4.5 新增12个花式词（pinpointing, outpacing, lift, compresses, exposes, proves decisive, harness, foster, myriad, traverse, embark, encapsulate, facet）
- §4.5 扩充保留不改列表（+binding constraint, cascading, self-contained, sharp, empirical等）
- 检测命令更新
