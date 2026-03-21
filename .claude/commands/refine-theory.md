# /refine-theory - 理论章节迭代精修

## 用途
对论文的理论章节进行迭代精修，确保：
1. 每个定理/引理都有正确的引用
2. 每个证明步骤都有严格推导，无跳步
3. 符号一致，无歧义
4. 持续迭代直到无法改进

## 输入参数
- `$ARGUMENTS` — 要精修的 LaTeX 文件路径（可选，默认为 `paper/journal/sections/theory_lower_bound.tex`）

## ⚠️ MANDATORY: Unified Protocol

**EVERY execution of this command MUST follow the unified protocol.**

### Step 0: Read Shared Config & RAG References

**STOP. Before doing ANYTHING else, execute these Read tool calls:**

```
Read .claude/commands/_shared/unified_protocol.md
Read .claude/commands/_shared/rag_config.md
Read .claude/writing_references/paragraphs/proof_structure.md
Read .claude/writing_references/sentences/algorithm_optimality.md
Read .claude/writing_references/sentences/problem_setup.md
Read .claude/writing_references/phrases/hedging.md
```

**Why RAG matters for theory refinement:**
- `proof_structure.md` (11 entries): Paragraph templates for proof presentation (change-of-measure, proof sketch intro, making bounds useful, Fisher information arguments)
- `algorithm_optimality.md` (12 entries): Sentence patterns for stating optimality results and complexity bounds
- `problem_setup.md` (9 entries): Patterns for stating assumptions and model definitions
- `hedging.md` (7 entries): Cautious language for claims with conditions

**When rewriting proof prose, you MUST ground every suggestion in a specific RAG pattern.** Do not use generic AI paraphrasing.

### Step 1: Read Paper State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

**STOP.** Before refining ANY theory section, you MUST load the paper's structured documentation.

1. Resolve paper name → run `ls docs/paper_state/` to find the actual directory
2. Read ALL required state files using the resolved path:

```
Read docs/paper_state/{resolved}/symbols.md          # Symbol definitions (prevent conflicts)
Read docs/paper_state/{resolved}/results.md          # Theorem registry (labels, locations, statements)
Read docs/paper_state/{resolved}/dependencies.md     # Assumption → Theorem dependency chains
Read docs/paper_state/{resolved}/framing.md          # Locked terminology
Read docs/paper_state/{resolved}/cross_references.md # What cites this theorem
Read docs/paper_state/{resolved}/changelog.md        # Recent changes (avoid conflicting edits)
```

3. Write a **verification checkpoint** confirming what you loaded:
```
State doc context loaded:
- symbols.md: [key symbols and their locked definitions, e.g., π_k=audit prob]
- results.md: [theorems in target file, their labels and current statements]
- dependencies.md: [assumption→theorem chains for target theorems]
- framing.md: [locked terminology, e.g., "audit" not "review"]
- cross_references.md: [which other sections/results reference these theorems]
- changelog.md: [last change date, recent modifications to watch for]
```

**If you skip this step:**
- You may change notation that is locked by symbols.md
- You may break dependency chains tracked in dependencies.md
- You may contradict terminology decisions in framing.md
- You may re-introduce issues that were already fixed (per changelog.md)

### Step 2: Apply RAG Miss Detection

When rewriting proof text, if no good RAG pattern match is found (similarity < 0.7), follow:
```
Read .claude/commands/_shared/rag_miss_detection.md
```

---

## 执行步骤

### Step 3: 读取当前草稿
```
Read the theory section draft:
- Target file: $ARGUMENTS or paper/journal/sections/theory_lower_bound.tex
- Cross-reference with results.md: verify theorem labels match
- Cross-reference with dependencies.md: verify assumption chain
- Note all theorems, lemmas, propositions
- Count total claims requiring justification
```

### Step 4: 加载参考证明文档

Check if domain-specific proof references exist:
```
Load proof reference documents (if available):
- docs/references/proofs/howard2021_cs_proofs.md (Confidence Sequences)
- docs/references/proofs/kaufmann2016_complexity_proofs.md (BAI Complexity)
- docs/references/proofs/garivier2016_trackandstop_proofs.md (Track-and-Stop)
```

**Note**: If these files do not exist, proceed without them. The RAG writing references (Step 0) provide the prose patterns; these proof reference documents provide the mathematical content.

### Step 5: 迭代精修循环

对每一轮迭代执行以下检查：

#### 5.0 Regression Guard — Load Constraints

**⚠️ MANDATORY — Follow `.claude/commands/_shared/regression_guard.md` Phase 1.**

At the START of each iteration, load the constraint set:
```
Read docs/paper_state/{resolved}/consistency_log.md
```

During refinement (Phase 2): check each edit against constraints — don't change locked notation, don't break dependency chains, don't reintroduce fixed issues.

At the END of each iteration (Phase 3): verify no resolved issues regressed. If regression found → fix immediately and mark as `[REGRESSION]` in the issue list.

#### 5.1 引用验证 (Reference Verification)
- 每个定理的陈述是否与原文一致？
- 条件/假设是否完整陈述？
  - **Cross-validate**: 与 `dependencies.md` 中的依赖链对比
- 引用格式是否正确 (作者, 年份, 定理编号)？
  - **Cross-validate**: 与 `cross_references.md` 对比

#### 5.2 证明完整性 (Proof Step Completeness)
- 是否有未解释的"显然"或"易证"？
- 每个不等式是否有命名的理由？
- 是否有未定义就使用的符号？
  - **Cross-validate**: 与 `symbols.md` 对比

#### 5.3 符号一致性 (Notation Consistency)
- **所有符号是否与 `symbols.md` 定义一致？**
- 是否在定义前使用？
- 上下标是否全文一致？

#### 5.4 数学严谨性 (Mathematical Rigor)
- 量词是否显式 (∀, ∃)？
- 定义域是否指定？
- 边界情况是否处理？

#### 5.5 RAG-Grounded Prose Refinement
When rewriting prose in theorems, remarks, or proof sketches:
1. **Find a matching pattern** from `proof_structure.md` or `algorithm_optimality.md`
2. **Adapt the pattern** to the specific content
3. **Cite the pattern** in the edit comment

**Example:**
```
Original: "We can then show that the lower bound is tight."
RAG match: proof_structure.md "[Making Bound Useful]"
  Pattern: "To make this bound useful, it remains to study [quantity A] and [quantity B]."
Refined: "To make this bound useful, it remains to study the Fisher information I_k^{(F)} and the variance ratio σ²_R/σ²_F."
```

### Step 6: 问题标记与修复
```
For each issue found:
1. Mark location in file (line number)
2. Categorize: [NEEDS REF] / [NEEDS PROOF] / [NOTATION] / [RIGOR] / [STATE_CONFLICT]
3. Propose fix with justification
   - For prose fixes: cite RAG pattern
   - For notation fixes: cite symbols.md definition
   - For dependency fixes: cite dependencies.md chain
4. Apply fix to document
```

### Step 7: 终止判断
```
If issues_found == 0:
    Output final confidence assessment
    Stop iteration
Else:
    Increment iteration counter
    Go to Step 5
```

## 输出格式

### 每轮迭代输出
```markdown
## 精修轮次 [N]

### 发现问题: [count] 个

#### 问题 1: [位置]
- **类型**: [NEEDS REF / NEEDS PROOF / NOTATION / RIGOR / STATE_CONFLICT]
- **描述**: [问题描述]
- **修复**: [如何修复]
- **参考**: [引用来源 — RAG pattern / symbols.md / dependencies.md]

### 本轮修改摘要:
[修改列表]

### State Doc Cross-Validation:
| Check | Source | Result |
|-------|--------|--------|
| Symbols consistent | symbols.md | ✅/❌ |
| Dependencies complete | dependencies.md | ✅/❌ |
| Terminology locked | framing.md | ✅/❌ |
| Cross-refs valid | cross_references.md | ✅/❌ |

### 剩余关注点:
[下轮需检查的问题]
```

### 完成时输出
```markdown
## 精修完成

### 总迭代次数: [N]
### 最终置信度评估:

| 定理 | 置信度 | State doc一致 | 备注 |
|------|--------|--------------|------|
| Theorem 1 (Lower Bound) | High/Medium/Low | ✅/❌ | ... |
| Proposition 1 (Decomposition) | ... | ✅/❌ | ... |
| Proposition 2 (Neyman) | ... | ✅/❌ | ... |
| Theorem 2 (Optimality) | ... | ✅/❌ | ... |

### RAG Patterns Applied:
| Pattern | Source | Applied To |
|---------|--------|-----------|
| [Change of Measure] | proof_structure.md | Thm 1 proof sketch |
| [Optimality Statement] | algorithm_optimality.md | Thm 2 statement |

### 建议后续工作:
[仍需解决的理论问题]
```

### Step 8: Update State Files

**⚠️ MANDATORY — DO NOT SKIP THIS STEP.**

After refining, you MUST update the state docs to reflect all changes:

```
Edit docs/paper_state/{resolved}/results.md        # Update theorem statements if changed
Edit docs/paper_state/{resolved}/symbols.md        # Update if any notation was modified
Edit docs/paper_state/{resolved}/dependencies.md   # Update if dependency chains changed
Edit docs/paper_state/{resolved}/changelog.md      # Log ALL refinements: "[date] refine-theory: [file] Round [N], [count] fixes"
```

**Update rules:**
- **results.md**: If any theorem statement was modified, update the registered statement
- **symbols.md**: If any notation was changed or new symbols introduced
- **dependencies.md**: If assumption→theorem links were added/corrected
- **changelog.md**: ALWAYS update with refinement summary

**If you skip this step, future commands will use stale theorem statements and notation.**

---

## 重点检查区域

### 1. 下界证明 (Theorem 1)
- KL 散度的双源分解是否正确
- 成本加权的运输成本公式
- Alternative set 定义是否完整

### 2. Neyman 分配 (Proposition 2)
- Lagrangian 推导与 KKT 条件
- 边界情况 (π=0 或 π=1) 处理

### 3. 渐近最优性 (Theorem 2)
- 插入估计的收敛性
- 停止规则的有效性
- 强制探索的充分性

### 4. 置信序列构造
- 使用哪种边界类型
- 方差过程 V_t 的定义
- IPW 对 sub-ψ 性质的影响

## 使用示例

```bash
# 精修默认文件
claude /refine-theory

# 精修指定文件
claude /refine-theory paper/journal/sections/theory_delayed_audits.tex
```

## 相关文件

| 文件 | 用途 |
|------|------|
| `paper/journal/prompt_for_gemini_theory_refinement.md` | 完整 prompt 模板 |
| `docs/references/proofs/*.md` | 参考证明文档 |
| `paper/bai_judge/main.tex` | 会议版论文 (符号参考) |
| `.claude/writing_references/paragraphs/proof_structure.md` | RAG: 证明结构模板 |
| `.claude/writing_references/sentences/algorithm_optimality.md` | RAG: 最优性陈述模板 |

---

## MANDATORY: Next Steps Section

**Every output MUST end with this section:**

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

📊 This Check: Theory Refinement
   Iterations: {N}
   Issues Remaining: {M}
   Confidence: {High/Medium/Low}
   State docs updated: ✅ results.md + symbols.md + changelog.md

🔴 IMMEDIATE ACTIONS:
   {If issues remain:}
   1. Address [NEEDS REF] items with proper citations
   2. Fix [NEEDS PROOF] gaps with complete derivations
   3. Re-run /refine-theory to continue iteration

   {If converged:}
   ✅ Theory section refined. Ready for verification.

🛠️ RECOMMENDED COMMANDS:

   [If issues remain:]
   /refine-theory [file]       → Continue refinement

   [When converged:]
   /verify-proof [theorem]     → Verify individual proofs
   /check-paper-consistency    → Check notation consistency

   [Before submission:]
   /paper-pipeline pre-submit MS → Final checklist
```
