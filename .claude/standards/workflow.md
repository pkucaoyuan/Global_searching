# 工作流程规范

本规范定义了 LLM-for-BAI 项目的开发和研究工作流程。

---

## 1. 开发工作流

### 1.1 先读后写原则

在修改任何代码之前：
1. **检查 `docs/plan/`**：确认是否有相关的计划文档
2. **检查 `src/` 接口**：理解现有的模块结构和接口定义
3. **检查相关测试**：了解预期行为

```
# 正确的工作流
1. 阅读 docs/plan/PROJECT_PLAN.md 了解整体架构
2. 阅读相关模块的 __init__.py 了解公共 API
3. 阅读相关测试了解预期行为
4. 实现代码
5. 运行测试验证

# 错误的工作流
1. 直接开始写代码
2. 遇到问题再看文档
```

### 1.2 原子化提交

每次提交应该：
- **单一目的**：一个提交只做一件事
- **可独立回滚**：不破坏其他功能
- **有意义的信息**：遵循 commit message 规范

```
# 正确：分开的原子提交
git commit -m "feat(cs): add empirical bernstein confidence sequence"
git commit -m "test(cs): add coverage probability test for empirical bernstein"
git commit -m "docs(cs): add docstring with math section"

# 错误：混合提交
git commit -m "add empirical bernstein, fix bug in IPW, update docs"
```

### 1.3 不混合重构与新功能

```
# 正确的工作流
# Day 1: 重构
git commit -m "refactor(estimators): extract base class for IPW and DR"

# Day 2: 新功能
git commit -m "feat(estimators): add control variate estimator"

# 错误的工作流
# 一个提交里既重构又加功能
git commit -m "add control variate and refactor estimators"
```

---

## 2. 新功能开发流程

### 2.1 Martingale/CS 实现检查清单

实现新的置信序列时，必须按以下顺序完成：

- [ ] **定义接口**：确定输入/输出类型
- [ ] **编写测试**：先写 coverage probability 测试
- [ ] **实现算法**：参照论文公式
- [ ] **添加 docstring**：包含 Math 部分
- [ ] **运行测试**：确保 coverage ≥ 1 - δ - ε
- [ ] **代码审查**：检查数值稳定性

```python
# 示例：实现 Empirical Bernstein CS 的流程

# Step 1: 定义接口
class EmpiricalBernsteinCS:
    def __init__(self, delta: float): ...
    def update(self, x: float) -> None: ...
    def get_interval(self) -> tuple[float, float]: ...

# Step 2: 编写测试
def test_coverage():
    """Coverage 必须 >= 1 - delta"""
    ...

# Step 3: 实现（参照 Howard et al. 2021）
# Step 4: Docstring with Math section
# Step 5: pytest tests/unit/common/test_confidence_sequences.py
# Step 6: 审查对数/除法等潜在数值问题
```

### 2.2 估计器实现检查清单

- [ ] **无偏性验证**：蒙特卡洛验证 E[estimator] ≈ true_value
- [ ] **方差公式**：确保与理论匹配
- [ ] **边界情况**：propensity → 0, 样本量 = 1
- [ ] **类型标注**：完整的类型提示
- [ ] **文档**：Math section 映射到论文

---

## 3. 实验工作流

### 3.1 实验目录结构

```
experiments/
├── results/
│   └── 2024-01-15/                    # 日期目录
│       ├── bai_gap_sweep/             # 实验名称
│       │   ├── config.yaml            # 实验配置
│       │   ├── results.json           # 结果数据
│       │   └── logs/                  # 日志文件
│       └── policy_bad_stop_demo/
│           └── ...
└── figures/
    └── 2024-01-15/
        ├── sample_complexity.pdf
        └── bad_stop_rate.pdf
```

### 3.2 实验命名规范

使用描述性名称，包含关键参数：
```
# 好的命名
bai_gap_sweep_delta0.05_K10
policy_replication_R1_to_R10
failure_mode_biased_judge

# 不好的命名
exp1
test_run
final_experiment
```

### 3.3 结果保护

**永远不要直接覆盖 `experiments/results/` 中的文件**

```python
# 正确：使用时间戳
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = f"experiments/results/{date}/{exp_name}_{timestamp}"

# 或者：检查是否存在
if result_path.exists():
    raise FileExistsError(f"结果已存在: {result_path}")
```

### 3.4 可复现性要求

每个实验必须保存：
1. **完整配置** (`config.yaml`)
2. **随机种子** (在 config 中)
3. **代码版本** (git commit hash)
4. **依赖版本** (`pip freeze > requirements_snapshot.txt`)

```yaml
# config.yaml 示例
experiment:
  name: bai_gap_sweep
  seed: 42
  git_commit: abc123def

parameters:
  n_arms: 10
  delta: 0.05
  gaps: [0.1, 0.2, 0.3, 0.4, 0.5]

runtime:
  n_trials: 1000
  max_rounds: 100000
```

---

## 4. 代码审查检查清单

### 4.1 数学正确性

- [ ] 公式与论文一致（检查符号、索引）
- [ ] 边界条件正确处理
- [ ] 数值稳定性（log(0), 除以 0, 溢出）

### 4.2 代码质量

- [ ] 类型提示完整
- [ ] Docstring 包含 Math section
- [ ] 无硬编码魔数（使用常量或配置）
- [ ] 遵循命名规范（见 `math_notation.md`）

### 4.3 测试

- [ ] 单元测试存在且通过
- [ ] Coverage probability 测试（对于 CS）
- [ ] 边界情况测试

### 4.4 性能

- [ ] 无不必要的循环
- [ ] 适当使用 NumPy 向量化
- [ ] 大实验有进度显示

---

## 5. 文档更新流程

### 5.1 何时更新文档

- **添加新模块**：更新 `docs/plan/PROJECT_PLAN.md`
- **修改接口**：更新相关 docstring
- **修改数学**：更新 `math_notation.md`
- **添加新命令**：更新 `latex_guide.md` 中的 commands

### 5.2 文档与代码同步

```
# 正确：文档和代码在同一提交
git add src/bai_judge/estimators/new_estimator.py
git add docs/plan/PROJECT_PLAN.md  # 更新架构图
git commit -m "feat(estimators): add new estimator with documentation"

# 错误：文档滞后
git commit -m "feat(estimators): add new estimator"
# ... 一周后 ...
git commit -m "docs: update for new estimator"  # 容易遗忘
```

---

## 6. 问题排查流程

### 6.1 数值问题

```
症状：NaN, Inf, 或结果明显错误

排查步骤：
1. 检查输入数据：是否有 NaN/Inf？
2. 检查除法：是否可能除以 0？
3. 检查对数：是否可能 log(0) 或 log(负数)？
4. 检查累积：是否有数值溢出？
5. 添加断言验证中间结果
```

### 6.2 Coverage 不足

```
症状：置信序列的经验 coverage < 1 - delta

排查步骤：
1. 检查公式实现是否与论文一致
2. 检查方差估计是否正确
3. 检查边界条件（t=1, 小样本）
4. 增加蒙特卡洛试验次数，排除统计波动
5. 对比参考实现
```

### 6.3 算法不收敛

```
症状：LUCB 或 Policy Learning 不停止

排查步骤：
1. 检查置信区间是否收缩（打印宽度）
2. 检查是否有手臂/标签被忽略
3. 检查停止条件的实现
4. 可视化置信区间随时间的变化
```

---

## 7. Git 工作流

### 7.1 分支命名

```
# 功能开发
feature/ipw-estimator
feature/dm-mixture-cs

# Bug 修复
fix/coverage-boundary

# 实验
exp/gap-sweep-simulation

# 文档
docs/update-math-notation
```

### 7.2 Commit Message 格式

```
type(scope): subject

body (optional)

footer (optional)
```

**类型**：
- `feat`: 新功能
- `fix`: Bug 修复
- `math`: 数学/理论修正
- `exp`: 实验相关
- `refactor`: 重构（不改变行为）
- `docs`: 文档
- `test`: 测试
- `chore`: 构建/工具

**示例**：
```
feat(bai_judge): implement doubly robust estimator

Add DR estimator with control variate for residual mean estimation.
Uses linear regression for outcome model.

Math: Eq. (15) in the research note.
```

### 7.3 禁止的操作

- `git push --force` 到 main/master
- `git commit --amend` 已推送的提交
- `git rebase` 公共分支

---

## 8. 发布检查清单

发布新版本前：

- [ ] 所有测试通过：`pytest tests/`
- [ ] 类型检查通过：`mypy src/`
- [ ] 代码格式化：`black src/ tests/`
- [ ] 文档更新完成
- [ ] CHANGELOG 更新
- [ ] 版本号更新 (`pyproject.toml`)
- [ ] 关键实验可复现
