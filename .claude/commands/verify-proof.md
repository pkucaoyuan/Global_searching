# Verify Proof - 数学证明验算工具

逐步验算数学证明的每一步，检查逻辑严谨性和数学正确性。

## 参数

- `$ARGUMENTS` - 要验算的文件路径，或特定定理/引理名称

## ⚠️ MANDATORY: Unified Protocol

**EVERY execution of this command MUST follow the unified protocol.**

### Step 0: Read Shared Config & RAG References

**STOP. Before doing ANYTHING else, execute these Read tool calls:**

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/paragraphs/proof_structure.md
Read .claude/writing_references/sentences/algorithm_optimality.md
```

**Why RAG matters for proof verification:**
- `proof_structure.md` contains 11 paragraph templates for how top venues present proofs (change-of-measure, proof sketch, making bounds useful)
- `algorithm_optimality.md` contains 12 sentence patterns for stating optimality results
- When suggesting improvements to proof prose, **ground suggestions in these patterns**

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before verifying ANY proof, you MUST load the paper's structured documentation.

1. Resolve paper name → run `ls docs/paper_state/` to find the actual directory
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/symbols.md        # Symbol definitions (critical for notation)
Read docs/paper_state/{resolved}/results.md        # Theorem registry (labels, locations, statements)
Read docs/paper_state/{resolved}/dependencies.md   # Assumption → Theorem dependency chains
Read docs/paper_state/{resolved}/cross_references.md  # Cross-reference targets
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- symbols.md: [key symbols relevant to this proof, e.g., π_k=audit prob, F_t=proxy score]
- results.md: [the theorem being verified + related theorems it references]
- dependencies.md: [assumptions this theorem depends on, e.g., Assumption 3.1→Theorem 5.1]
- cross_references.md: [other results that cite this theorem]
```

**If you skip this step:**
- You may misinterpret notation (e.g., `b_k` means bias, NOT best arm)
- You may miss assumption dependencies (e.g., Theorem needs Assumption 3.5 but proof doesn't invoke it)
- You may not catch cross-reference inconsistencies

### Step 2: Apply RAG Miss Detection

If when suggesting proof improvements, no good RAG pattern match is found (similarity < 0.7), follow:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

---

## 验算流程

### Step 3: 定位证明内容

1. 如果提供了文件路径，读取该文件
2. 如果提供了定理名称（如 `thm:convergence`, `lem:iterate_error`）：
   - **先查 `results.md`** 获取标签对应的文件和行号
   - 然后在对应文件中搜索并定位该证明
3. 提取完整的证明文本（从 `\begin{proof}` 到 `\end{proof}`）

### Step 4: 解析证明结构

将证明分解为以下组成部分：
- **声明 (Statement)**: 定理/引理的陈述
  - **Cross-validate**: 与 `results.md` 中的注册声明对比，确认一致性
- **假设 (Assumptions)**: 使用了哪些假设条件
  - **Cross-validate**: 与 `dependencies.md` 中的依赖链对比，确认所有必需假设都被引用
- **主要步骤 (Main Steps)**: 证明的关键逻辑步骤
- **结论 (Conclusion)**: 最终结论

### Step 5: 逐步验算

对每一个推导步骤，检查：

#### 5.1 数学正确性
- [ ] 代数运算是否正确（展开、化简、合并同类项）
- [ ] 不等式方向是否正确
- [ ] 极限、期望、求和的交换是否有依据
- [ ] 矩阵/向量运算是否正确（维度匹配、转置、迹）

#### 5.2 逻辑严谨性
- [ ] 每一步是否有充分的理由支撑
- [ ] **是否有跳跃或省略的步骤** ⚠️ 重点检查
  - 从 A 到 B 的推导是否需要中间步骤
  - 是否省略了"显然"但实际需要证明的步骤
  - 不等式链中是否有未解释的跳跃
- [ ] 条件是否满足（如应用某个引理时）
- [ ] 量词（∀, ∃）的使用是否正确

#### 5.3 引用完整性 ⚠️ 重点检查
- [ ] **每个非平凡结论是否有引用支撑**
  - 使用的不等式是否标注来源（如 "by Cauchy-Schwarz", "by Jensen's inequality"）
  - 引用的引理/定理是否给出编号或文献
  - 标准结论是否注明（如 "by standard SVRG analysis"）
- [ ] 禁止出现 "it is easy to see", "clearly", "obviously" 等无依据声明
- [ ] 对于 "following [cite]" 的陈述，验证是否确实与引用一致

#### 5.4 假设使用（cross-validate with dependencies.md）
- [ ] 引用的假设是否已声明
- [ ] **dependencies.md 中列出的所有必需假设是否都在证明中被引用**
- [ ] 假设的条件是否满足
- [ ] 是否有隐含假设未明确说明

#### 5.5 符号一致性（cross-validate with symbols.md）
- [ ] **证明中使用的符号是否与 symbols.md 定义一致**
- [ ] 是否有符号在证明中重新定义但与全文定义冲突
- [ ] 上下标是否与全文一致

#### 5.6 引用验证（cross-validate with cross_references.md）
- [ ] 引用的引理/定理是否正确
- [ ] 引用的公式编号是否正确
- [ ] **cross_references.md 中的引用目标是否与实际一致**
- [ ] 文献引用是否准确

#### 5.7 符号严谨性 🚫 禁止事项
- [ ] **禁止使用 `\approx`** - 必须使用精确符号：
  - 用 `=` 表示相等
  - 用 `\leq`, `\geq` 表示不等式
  - 用 `\lesssim`, `\gtrsim` 表示渐进界（隐藏常数）
  - 用 `= O(\cdot)`, `= \Theta(\cdot)`, `= o(\cdot)` 表示渐进阶
  - 用 `\to` 表示极限收敛
  - 用 `\xrightarrow{p}`, `\xrightarrow{d}` 表示概率/分布收敛
- [ ] 禁止模糊表述如 "approximately equal", "roughly"
- [ ] 所有 $\lesssim$ 必须说明隐藏的常数依赖什么参数

### Step 6: 常见问题检查清单

#### 🚨 高优先级检查项
- [ ] **跳跃步骤 (Jumped Steps)**
  - 寻找 "Thus", "Therefore", "Hence" 后面的大跨度推导
  - 检查从一个公式到下一个公式是否需要多个中间步骤
  - 标记所有 "by similar argument", "analogously" 等省略标记
- [ ] **无引用结论 (Unsubstantiated Claims)**
  - 每个不等式必须有来源
  - 每个极限交换必须说明条件
  - 每个"已知结论"必须给出引用
- [ ] **`\approx` 使用检查**
  - 搜索所有 `\approx` 出现
  - 替换为精确的渐进符号或等式

#### 概率/统计类证明
- [ ] 期望的线性性使用是否正确
- [ ] 条件期望的塔性质 $\E[\E[X|Y]] = \E[X]$
- [ ] 方差分解 $\Var(X) = \E[\Var(X|Y)] + \Var(\E[X|Y])$
- [ ] Markov/Chebyshev/Hoeffding 不等式的条件
- [ ] 大数定律/中心极限定理的条件
- [ ] 鞅差序列的性质

#### 优化类证明
- [ ] 凸性/强凸性的使用
- [ ] Lipschitz 条件的应用
- [ ] 梯度下降收敛性分析
- [ ] Gronwall 不等式的应用条件

#### 矩阵/线性代数
- [ ] 矩阵求逆的条件（可逆性）
- [ ] 特征值/特征向量的性质
- [ ] 范数的次可加性和齐次性
- [ ] Neumann 级数收敛条件 $\|A\| < 1$

### Step 7: 输出验算报告

```markdown
# 验算报告: [定理/引理名称]

## 证明概要
- **文件**: [文件路径]
- **行号**: [起始行-结束行]
- **依赖假设**: [列出使用的假设]
- **State doc cross-validation**: ✅/⚠️ [results.md一致性, dependencies.md完整性, symbols.md符号一致性]

## 逐步验算

### 步骤 1: [步骤描述]
- **原文**: [原始表述]
- **验算**: ✅ 正确 / ⚠️ 需注意 / ❌ 错误
- **说明**: [详细解释]

### 步骤 2: [步骤描述]
...

## State Doc Cross-Validation Results

| Check | Source | Result |
|-------|--------|--------|
| Theorem statement matches results.md | results.md | ✅/❌ |
| All required assumptions cited | dependencies.md | ✅/❌ |
| Symbol usage consistent | symbols.md | ✅/❌ |
| Cross-references valid | cross_references.md | ✅/❌ |

## 问题汇总

| 类型 | 位置 | 问题描述 | 严重程度 |
|------|------|----------|----------|
| 🚨 跳跃步骤 | 步骤3 | 从X直接推到Y，缺少中间步骤 | ❌ 高 |
| 🚨 无引用结论 | 步骤4 | 使用不等式未标注来源 | ❌ 高 |
| 🚨 使用approx | 第15行 | 应替换为 $\lesssim$ 或 $= O(\cdot)$ | ❌ 高 |
| 📋 State不一致 | symbols.md | 证明中b_k≠symbols.md定义 | ⚠️ 中 |
| 计算错误 | 步骤5 | ... | ❌ 高 |

## 改进建议（RAG-Grounded）

For each prose suggestion, cite the RAG pattern used:

1. [建议1] — *Pattern: proof_structure.md "[Change of Measure Technique]"*
2. [建议2] — *Pattern: algorithm_optimality.md "[Optimality Statement]"*

## 结论
- **总体评估**: ✅ 证明正确 / ⚠️ 有小问题 / ❌ 存在严重问题
```

### Step 8: Update State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

After verification, you MUST update the paper state:

```
Edit docs/paper_state/{resolved}/results.md        # Add verification status to the theorem entry
Edit docs/paper_state/{resolved}/changelog.md      # Log: "[date] verify-proof: [theorem] → [status]"
```

**Update rules:**
- In `results.md`: Add `Verified: ✅/⚠️/❌ [date]` to the theorem entry
- In `changelog.md`: Always log the verification result
- If issues found: Also update `consistency_log.md` with the issue list

---

## 使用示例

```bash
# 验算特定文件
/verify-proof appendix/C2_preliminary_lemmas.tex

# 验算特定引理
/verify-proof lem:iterate_error

# 验算特定定理
/verify-proof thm:asymptotic_normal
```

## 注意事项

### 🚨 核心原则（必须遵守）
1. **禁止 `\approx`** - 所有近似必须用精确的渐进符号表示
2. **禁止跳步** - 每一步推导必须清晰可验证，不允许"显然"跳过
3. **禁止无引用结论** - 每个非平凡不等式/结论必须有来源

### 一般注意事项
4. **不要假设正确性** - 即使是已发表的论文也可能有错误
5. **检查边界条件** - 特别注意 $n \to \infty$, $\epsilon \to 0$ 等极限情况
6. **验证常数** - 确保常数的定义和使用一致
7. **追溯依赖** - 如果引用了其他引理，也要验证其正确性
8. **记录疑问** - 对于不确定的步骤，明确标注需要进一步确认

### 常见需要引用的结论
- Cauchy-Schwarz 不等式
- Jensen 不等式
- Markov / Chebyshev 不等式
- Young 不等式: $ab \leq \frac{a^p}{p} + \frac{b^q}{q}$
- Hölder 不等式
- 三角不等式 / 范数次可加性
- Gronwall 不等式
- Neumann 级数: $(I-A)^{-1} = \sum_{k=0}^\infty A^k$ for $\|A\| < 1$
- 矩阵扰动界 (Weyl's inequality)
- 鞅中心极限定理

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: Proof Verification
   Proof: {theorem/lemma name}
   Status: ✅ Correct / ⚠️ Has Issues / ❌ Contains Errors
   State docs updated: ✅ results.md + changelog.md

🔴 IMMEDIATE ACTIONS:
   {If errors found:}
   1. Fix issues at: [locations listed above]
   2. Add missing references/citations
   3. Re-run /verify-proof [name] to verify

   {If correct:}
   ✅ Proof verified. Proceed to next proof or paper check.

🛠️ RECOMMENDED COMMANDS:

   [If errors found:]
   /verify-proof [name]        → Re-verify after fixes

   [When proofs verified:]
   /check-paper-consistency    → Verify notation consistency
   /refine-theory [file]       → Iterative theory refinement

   [Before submission:]
   /paper-pipeline pre-submit MS → Final checklist
```
