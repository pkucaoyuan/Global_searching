# LaTeX 规范

本规范适用于 LLM-for-BAI 项目的所有 LaTeX 文档。

---

## 1. 文件组织

### 1.1 项目结构
```
paper/
├── bai_judge/                    # Track A 论文
│   ├── main.tex                  # 主文件
│   ├── commands.tex              # 自定义宏
│   ├── refs.bib                  # 参考文献
│   └── sections/
│       ├── introduction.tex
│       ├── problem_setup.tex
│       ├── algorithm.tex
│       ├── analysis.tex
│       ├── experiments.tex
│       └── appendix.tex
│
└── policy_learning/              # Track B 论文
    ├── main.tex
    ├── commands.tex
    ├── refs.bib
    └── sections/
        └── ...
```

### 1.2 main.tex 结构
```latex
\documentclass{article}

% 包引入
\usepackage{amsmath,amssymb,amsthm}
\usepackage{algorithm,algorithmic}
\usepackage{graphicx}
\usepackage{hyperref}

% 自定义命令
\input{commands.tex}

% 定理环境
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

\begin{document}

\title{...}
\author{...}
\maketitle

\begin{abstract}
...
\end{abstract}

\input{sections/introduction}
\input{sections/problem_setup}
...

\bibliographystyle{plainnat}
\bibliography{refs}

\appendix
\input{sections/appendix}

\end{document}
```

---

## 2. 自定义命令 (commands.tex)

### 2.1 Track A (BAI Judge) 命令
```latex
% === 数学运算符 ===
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator{\E}{\mathbb{E}}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Cov}{Cov}
\newcommand{\Prob}{\mathbb{P}}

% === 集合与空间 ===
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\cX}{\mathcal{X}}  % 上下文空间
\newcommand{\cK}{\mathcal{K}}  % 手臂集合

% === Track A 专用符号 ===
% 手臂与计数
\newcommand{\arm}{k}           % 手臂索引
\newcommand{\narm}{K}          % 手臂数量
\newcommand{\Nk}{N_k}          % 手臂 k 的拉取次数
\newcommand{\nk}{n_k}          % 手臂 k 的审计次数

% 结果变量
\newcommand{\human}{Y}         % 人工标签
\newcommand{\judge}{F}         % LLM 得分
\newcommand{\residual}{R}      % 残差

% 估计量
\newcommand{\thetak}{\theta_k}         % 真实均值
\newcommand{\muk}{\mu_k}               % 均值
\newcommand{\muF}{\mu_{F,k}}           % LLM 均值
\newcommand{\muR}{\mu_{R,k}}           % 残差均值
\newcommand{\hatmuIPW}{\hat{\mu}_{R,k}^{\text{IPW}}}  % IPW 估计
\newcommand{\hatmuDR}{\hat{\mu}_{R,k}^{\text{DR}}}    % DR 估计

% 置信区间
\newcommand{\Lk}{L_k}          % 下界
\newcommand{\Uk}{U_k}          % 上界

% 审计
\newcommand{\audit}{A}         % 审计指示符
\newcommand{\prop}{\pi}        % 审计概率

% LUCB
\newcommand{\best}{b_t}        % 经验最优
\newcommand{\challenger}{c_t}  % 挑战者
```

### 2.2 Track B (Policy Learning) 命令
```latex
% === Track B 专用符号 ===
% 文档与标签
\newcommand{\doc}{D}           % 文档
\newcommand{\lab}{X}           % 标签
\newcommand{\labspace}{\cX}    % 标签空间

% 策略与动作
\newcommand{\action}{A}        % 动作
\newcommand{\actionspace}{\mathcal{A}}  % 动作空间
\newcommand{\policy}{\pi}      % 策略
\newcommand{\policystar}{\pi^*}% 最优策略
\newcommand{\policyhat}{\hat{\pi}}  % 学习策略

% 价值
\newcommand{\Value}{V}         % 策略价值
\newcommand{\Vstar}{V^*}       % 最优价值
\newcommand{\Vlow}{\underline{V}}   % 价值下界
\newcommand{\Vup}{\bar{V}}     % 价值上界

% 概率与均值
\newcommand{\plabel}{p}        % 标签概率
\newcommand{\outcome}{\mu}     % 条件均值 μ(x,a)

% 复制
\newcommand{\nrep}{R}          % 复制次数
```

### 2.3 共享命令
```latex
% === 置信序列 ===
\newcommand{\CS}{\mathcal{C}}  % 置信序列
\newcommand{\mart}{M}          % Martingale
\newcommand{\LR}{\Lambda}      % 似然比

% === 时间与样本 ===
\newcommand{\tim}{t}           % 时间索引
\newcommand{\nsamp}{N}         % 样本量

% === 置信水平 ===
\newcommand{\conf}{\delta}     % 错误概率

% === 概率陈述 ===
\newcommand{\whp}{with probability at least $1-\delta$}
\newcommand{\wp}{w.p.}

% === 常用缩写 ===
\newcommand{\iid}{i.i.d.}
\newcommand{\wrt}{w.r.t.}
\newcommand{\st}{s.t.}
\newcommand{\eg}{e.g.,}
\newcommand{\ie}{i.e.,}
\newcommand{\cf}{cf.}
\newcommand{\etal}{\textit{et al.}}
```

---

## 3. 数学排版规范

### 3.1 公式环境选择

| 场景 | 环境 | 示例 |
|------|------|------|
| 单行编号公式 | `equation` | 重要结果 |
| 单行不编号 | `equation*` 或 `\[...\]` | 中间步骤 |
| 多行对齐 | `align` | 推导过程 |
| 条件分支 | `cases` | 分段函数 |
| 无编号多行 | `align*` | 辅助计算 |

```latex
% 重要结果（编号）
\begin{equation}
\label{eq:ipw}
\hatmuIPW = \frac{1}{\Nk} \sum_{s: k_s = k} \frac{\audit_s}{\prop_s}(\human_s - \judge_s)
\end{equation}

% 推导过程（对齐）
\begin{align}
\E[\hatmuIPW] &= \E\left[ \frac{\audit}{\prop}(\human - \judge) \right] \\
              &= \E\left[ \E\left[ \frac{\audit}{\prop}(\human - \judge) \mid \human, \judge \right] \right] \\
              &= \E[\human - \judge] = \muR
\end{align}
```

### 3.2 括号大小

使用 `\left` 和 `\right` 自动调整，或手动指定：
```latex
% 自动调整
\left( \frac{1}{n} \sum_{i=1}^n X_i \right)

% 手动指定（推荐，更可控）
\bigl( \frac{1}{n} \sum_{i=1}^n X_i \bigr)

% 大小等级：( < \bigl < \Bigl < \biggl < \Biggl
```

### 3.3 概率与期望

```latex
% 条件期望
\E[\human \mid \arm, \lab]

% 条件概率
\Prob(\audit = 1 \mid \human, \judge)

% 指示函数
\mathbf{1}\{\arm = k\}
```

### 3.4 公式内标点

标点应在数学环境内：
```latex
% 正确
For all $k \in \cK$,
\[
\thetak = \E[\human \mid k].
\]

% 错误
For all $k \in \cK$
\[
\thetak = \E[\human \mid k]
\].
```

---

## 4. 算法排版

### 4.1 使用 algorithm2e 或 algorithmic

```latex
\usepackage{algorithm}
\usepackage{algorithmic}

\begin{algorithm}[t]
\caption{LUCB with Selective Auditing}
\label{alg:lucb}
\begin{algorithmic}[1]
\REQUIRE Arms $\cK$, confidence $\conf$, audit budget $B$
\ENSURE Best arm $\hat{k}$
\STATE Initialize: $\Nk \gets 0$, $\nk \gets 0$ for all $k$
\FOR{$t = 1, 2, \ldots$}
    \STATE Compute UCB/LCB: $\Uk, \Lk$ for all $k$
    \STATE $\best \gets \argmax_k \hat{\theta}_k$
    \STATE $\challenger \gets \argmax_{k \neq \best} \Uk$
    \IF{$L_{\best} > U_{\challenger}$}
        \STATE \textbf{return} $\best$
    \ENDIF
    \STATE Sample arm $k_t \in \{\best, \challenger\}$
    \STATE Observe $\judge_t$
    \STATE With probability $\prop_t$, audit to get $\human_t$
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

### 4.2 算法注释
```latex
\STATE $\best \gets \argmax_k \hat{\theta}_k$ \COMMENT{Empirical best}
```

---

## 5. 图表规范

### 5.1 图片

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\linewidth]{figures/sample_complexity.pdf}
\caption{Sample complexity vs. gap $\Delta$. Our method (blue) achieves
lower sample complexity than the baseline (orange) across all gap values.}
\label{fig:sample_complexity}
\end{figure}
```

### 5.2 表格

```latex
\begin{table}[t]
\centering
\caption{Comparison of estimators. DR achieves lower variance than IPW
while maintaining unbiasedness.}
\label{tab:estimators}
\begin{tabular}{lcc}
\toprule
Estimator & Bias & Variance \\
\midrule
IPW & 0 & High \\
DR  & 0 & Low \\
\bottomrule
\end{tabular}
\end{table}
```

### 5.3 子图

```latex
\usepackage{subcaption}

\begin{figure}[t]
\centering
\begin{subfigure}[b]{0.48\linewidth}
    \includegraphics[width=\linewidth]{fig_a.pdf}
    \caption{Setting A}
    \label{fig:sub_a}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\linewidth}
    \includegraphics[width=\linewidth]{fig_b.pdf}
    \caption{Setting B}
    \label{fig:sub_b}
\end{subfigure}
\caption{Comparison across different settings.}
\label{fig:comparison}
\end{figure}
```

### 5.4 Figure Generation Standards (Python/matplotlib)

**Automated checking**: `/check-figures-tables`

**Font**: CMU Serif → Computer Modern fallback → Times New Roman. Must match LaTeX document.

```python
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
```

**Format**: Always save as PDF (vector). PNG only for photos at ≥300 DPI.

**Legend placement**: Outside by default to avoid obscuring data.
```python
# Below figure (horizontal legend)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

# Right of figure (vertical legend)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
```

**Color palette**: Use semantic colors consistently across all figures.

| Role | Hex | Usage |
|------|-----|-------|
| Success/Best | `#2ecc71` | Neyman allocator, correct results |
| Baseline | `#3498db` | Uniform, fixed methods |
| Failure | `#e74c3c` | Error modes, warnings |
| Neutral | `#95a5a6` | Reference, background |
| Alternative | `#9b59b6` | Secondary methods |
| Accent | `#f39c12` | Highlights |
| Theory | `#34495e` | Theoretical bounds |

**Figure sizes**: Single column `(5, 3.5)`, double column `(10, 3.5)`.

**Caption pattern**: `[What is shown]. [Key takeaway]. [Reference to theorem if applicable].`

---

## 6. 引用规范

### 6.1 文献引用

```latex
\usepackage{natbib}

% 括号引用（句末）
... has been studied extensively~\citep{smith2020, jones2021}.

% 文本引用（句中）
\citet{smith2020} showed that...

% 多作者
\citet{smith2020} 或 Smith et al.~\citep{smith2020}
```

### 6.2 交叉引用

```latex
% 定理/引理
Theorem~\ref{thm:main}
Lemma~\ref{lem:concentration}

% 公式
Equation~\eqref{eq:ipw}  % 自动加括号
\eqref{eq:ipw}           % 简写

% 图表
Figure~\ref{fig:results}
Table~\ref{tab:comparison}

% 章节
Section~\ref{sec:algorithm}

% 附录
Appendix~\ref{app:proofs}
```

### 6.3 使用 cleveref（推荐）

```latex
\usepackage{cleveref}

% 自动添加 "Theorem", "Figure" 等
\cref{thm:main}           % Theorem 1
\Cref{thm:main}           % 句首大写
\cref{thm:main,lem:aux}   % Theorem 1 and Lemma 2
```

---

## 7. 定理环境

### 7.1 定理陈述

```latex
\begin{theorem}[Sample Complexity]
\label{thm:sample}
Let $\conf \in (0,1)$. With probability at least $1 - \conf$,
Algorithm~\ref{alg:lucb} identifies the best arm using at most
\[
\sum_{k \neq k^*} \frac{C}{\Delta_k^2} \log\left( \frac{\narm}{\conf} \right)
\]
samples, where $\Delta_k = \theta_{k^*} - \thetak$ is the suboptimality gap.
\end{theorem}
```

### 7.2 证明

```latex
\begin{proof}
The proof proceeds in two steps.

\textit{Step 1: Concentration.}
By Lemma~\ref{lem:concentration}, for all $t$ and $k$,
\[
\Prob(|\hat{\theta}_k - \thetak| > \beta_t) \leq \frac{\conf}{2\narm t^2}.
\]

\textit{Step 2: Correctness.}
On the event that all confidence intervals are valid...
\end{proof}
```

### 7.3 证明草图

```latex
\begin{proof}[Proof Sketch]
The full proof is in Appendix~\ref{app:proof_main}. The key insight is...
\end{proof}
```

---

## 8. 常见错误与修正

### 8.1 间距问题

```latex
% 错误：双下标
$\mu_R_k$

% 正确
$\mu_{R,k}$

% 错误：没有空格的缩写
i.e.the result

% 正确
i.e., the result
```

### 8.2 数学字体

```latex
% 集合用 \mathcal
\mathcal{X}, \mathcal{A}, \mathcal{K}

% 概率/期望用 \mathbb
\mathbb{P}, \mathbb{E}, \mathbb{R}

% 矩阵/向量用粗体
\mathbf{X}, \boldsymbol{\theta}

% 运算符用 \mathrm 或 \operatorname
\mathrm{Var}, \operatorname{Cov}
```

### 8.3 Overleaf 兼容性

- 使用 `\usepackage[utf8]{inputenc}` 支持中文注释
- 图片格式优先使用 PDF（矢量）
- 避免使用过新的包，检查 CTAN 兼容性

### 8.4 数学环境定界符

**严禁**在 `align`, `equation` 等数学环境内部再次使用 `\[ ... \]` 或 `\begin{equation} ... \end{equation}`。

```latex
% 错误
\begin{align*}
  \E\[ X \] &= \mu  % 错误：在 align 中使用 \[ \]
\end{align*}

% 正确
\begin{align*}
  \E[ X ] &= \mu    % 使用标准括号
  \E\left[ X \right] &= \mu % 或使用 \left[ \right]
\end{align*}
```

同理，在宏定义（如 `\E`, `\Var`）内部也不要包含 `\[` 或 `\]`，除非该宏专门用于显示公式模式。

### 8.5 标点符号

- **引号**：前引号使用 \`\` (两个反引号)，后引号使用 '' (两个单引号)。严禁使用直引号 "。
  - 错误：`"Hello World"`
  - 正确：` ``Hello World'' `

---

## 9. 编译错误排查

### 9.1 致命错误：无法生成 PDF

**症状**：`I can't find file 'main.aux'`, `Emergency stop`, 或 `Text line contains an invalid character`。

**常见原因**：
1.  **隐藏字符污染**：文件中存在不可见的控制字符（如 form feed `0x0c`）或 NULL 字节（`0x00`）。
    *   症状：报错 `! Text line contains an invalid character.`，且通常伴随 weird spacing 或 `^^@` 符号。
2.  **反斜杠丢失**：`\frac` 写成 `rac`（可能是编辑器自动替换导致）。
3.  **输入文件缺失**：`\input{sections/X}` 引用的文件不存在。
4.  **Math Mode 缺失**：在文本模式下使用了数学符号（如 `{`, `}`, `^`, `_` 等）。
    *   错误：`Set A = {x, y}`
    *   正确：`Set A = $\{x, y\}$`

**排查方法**：
```bash
# 检查文件中的非 ASCII 字符 (包括 NULL bytes)
od -c sections/filename.tex | grep -v " \\n"

# 检查是否存在 NULL bytes
python3 -c "print(b'\x00' in open('sections/filename.tex', 'rb').read())"

# 修复方法：完全重写文件内容（推荐使用 Python 脚本写入以确保编码正确），
# 或者使用 tr/perl 清理（风险较高）：
tr -d '\000' < corrupted.tex > clean.tex
```

### 9.2 引用警告

**症状**：`Citation 'xxx' on page Y undefined`

**原因**：
1.  `refs.bib` 中不存在该 key
2.  BibTeX 未运行或运行失败
3.  Key 拼写错误（如 `waudby2024estimating` vs `waudby2024anytime`）

**解决**：
```bash
# 检查 bib 文件中的 key
grep -o '@[a-z]*{[^,]*' refs.bib

# 完整编译流程
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### 9.3 Algorithm 包错误

**症状**：`Undefined control sequence \For` 或 `\EndFor`

**原因**：混淆了 `algorithmic` 和 `algorithm2e` 包的命令

| `algorithmic` 包 | `algorithm2e` 包 |
|------------------|------------------|
| `\STATE` | `\State` |
| `\IF{...}` | `\If{...}` |
| `\ENDIF` | `\EndIf` |
| `\FOR{...}` | `\For{...}` |
| `\ENDFOR` | `\EndFor` |
| `\LOOP` | (无对应) |
| `\ENDLOOP` | (无对应) |

**本项目使用 `algorithmic` 包**，请使用大写命令。

---

## 10. 编译命令

### 10.1 完整编译（含参考文献）
```bash
cd paper/bai_judge && \
pdflatex -interaction=nonstopmode main.tex && \
bibtex main && \
pdflatex -interaction=nonstopmode main.tex && \
pdflatex -interaction=nonstopmode main.tex
```

### 10.2 快速编译（仅预览）
```bash
cd paper/bai_judge && pdflatex -interaction=nonstopmode main.tex
```

### 10.3 清理临时文件
```bash
cd paper/bai_judge && rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot
```

---

## 11. LaTeX 与代码的对应关系

确保论文内容与实现代码保持一致：

| LaTeX Section | 关键内容 | 对应源码 |
|---------------|----------|----------|
| `problem_setup.tex` | $Y_t = F_t + R_t$ 分解 | `src/bai_judge/environments/biased_judge_env.py` |
| `algorithm.tex` | LUCB-Joint 算法、IPW/DR 估计器 | `src/bai_judge/algorithms/lucb_joint.py`, `src/bai_judge/estimators/` |
| `analysis.tex` | Betting CS、正确性定理 | `src/common/confidence_sequences/betting.py` |
| `experiments.tex` | Phase 1 (65% 节省)、Phase 2A/B/C 结果 | `experiments/track_a_*.py` |

### 实验结果对照表

| 方法 | 成本节省 | 准确率 | 审计率 |
|------|----------|--------|--------|
| Phase 1 (UncertaintyWeighted) | 65% | 100% | ~10% |
| Phase 2A (RAG) | +43.3% | 65% | 3.6% |
| Phase 2B (Ensemble) | -- | 75% | 4.8% |
| Phase 2C (Hierarchical) | **90.1%** | **85%** | **0%** |
