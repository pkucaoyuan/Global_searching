# Python 编码规范

本规范适用于 LLM-for-BAI 项目的所有 Python 代码。

---

## 1. 文件命名规范

### 1.1 禁止使用的命名模式
以下命名模式会导致版本混乱，**严格禁止**：
- `enhanced_*`, `*_v2`, `*_new`, `*_old`
- `*_improved`, `*_better`, `*_final`
- `*_backup`, `*_copy`

### 1.2 推荐的命名方式
使用功能导向的命名：
```
# 正确示例
ipw_estimator.py      # IPW估计器
lucb_sampler.py       # LUCB采样器
empirical_bernstein.py # Empirical Bernstein CS
dm_mixture.py         # Dirichlet-Multinomial混合

# 错误示例
estimator_v2.py
new_sampler.py
improved_bernstein.py
```

---

## 2. 类型提示（Type Hints）

**严格要求**：所有函数和方法必须有完整的类型提示。

### 2.1 基本规范
```python
def compute_cs_width(
    samples: np.ndarray,
    delta: float,
    t: int
) -> float:
    """计算置信序列宽度"""
    ...

def run_lucb(
    env: BiasedJudgeEnv,
    config: LUCBConfig,
    rng: np.random.Generator
) -> tuple[int, int]:
    """返回 (best_arm, total_samples)"""
    ...
```

### 2.2 复杂类型
```python
from typing import Optional, Callable, TypeVar
from collections.abc import Sequence

T = TypeVar('T')

def aggregate_results(
    results: Sequence[dict[str, float]],
    reducer: Callable[[list[float]], float] = np.mean
) -> dict[str, float]:
    ...
```

### 2.3 NumPy 数组形状注释
对于关键数组，在 docstring 中注明形状：
```python
def compute_policy_value(
    outcome_matrix: np.ndarray,  # shape: (n_contexts, n_actions)
    policy: np.ndarray,          # shape: (n_contexts,) dtype=int
    weights: np.ndarray          # shape: (n_contexts,)
) -> float:
    """
    Args:
        outcome_matrix: 结果矩阵，shape (n_contexts, n_actions)
        policy: 策略向量，每个元素是选择的动作索引
        weights: 上下文权重，需满足 sum(weights) == 1
    """
    ...
```

---

## 3. 错误处理（Fail Fast 原则）

### 3.1 使用 `assert` 验证内部不变量
```python
def compute_ipw(residuals: np.ndarray, propensities: np.ndarray) -> float:
    # 内部逻辑不变量
    assert len(residuals) == len(propensities), "长度不匹配"
    assert np.all(propensities > 0), "倾向性得分必须为正"
    assert np.all(propensities <= 1), "倾向性得分必须 <= 1"

    return np.mean(residuals / propensities)
```

### 3.2 使用 `ValueError` 处理用户/配置输入
```python
def create_experiment(config: ExperimentConfig) -> Experiment:
    if config.delta <= 0 or config.delta >= 1:
        raise ValueError(f"delta 必须在 (0, 1) 范围内，收到: {config.delta}")
    if config.n_arms < 2:
        raise ValueError(f"至少需要 2 个手臂，收到: {config.n_arms}")
    ...
```

### 3.3 禁止静默捕获异常
```python
# 错误示例 - 静默吞掉异常
try:
    result = risky_operation()
except Exception:
    result = None  # 隐藏了问题

# 正确示例 - 记录或重新抛出
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"操作失败: {e}")
    raise
```

---

## 4. Docstrings（Google Style + Math Section）

### 4.1 必需的 Math Section
所有涉及数学公式的函数必须包含 `Math:` 部分，映射代码变量到论文符号：

```python
def compute_ipw_residual(
    residuals: np.ndarray,
    propensities: np.ndarray,
    audit_flags: np.ndarray
) -> float:
    """
    计算 IPW 残差估计量。

    Math:
        $$\hat{\mu}_{R,k}^{IPW} = \frac{1}{N_k} \sum_{s:k_s=k} \frac{A_s}{\pi_s}(Y_s - F_s)$$

        其中:
        - $A_s$ = audit_flags[s]（审计指示符）
        - $\pi_s$ = propensities[s]（审计概率）
        - $(Y_s - F_s)$ = residuals[s]（真实值与LLM预测的残差）

    Args:
        residuals: 残差数组 (Y - F)，shape (n_samples,)
        propensities: 审计概率 π_s，shape (n_samples,)
        audit_flags: 二元审计指示符 A_s，shape (n_samples,)

    Returns:
        IPW 残差均值估计

    Raises:
        ValueError: 如果 propensities 包含零值
    """
    ...
```

### 4.2 类的 Docstring
```python
class LUCBAgent:
    """
    LUCB (Lower Upper Confidence Bound) 算法实现。

    Math:
        选择规则：
        - 经验最优臂: $b_t = \arg\max_k \hat{\mu}_k(t)$
        - 挑战者: $c_t = \arg\max_{k \neq b_t} U_k(t)$
        - 停止条件: $L_{b_t}(t) > U_{c_t}(t)$

    Attributes:
        n_arms: 手臂数量 K
        delta: 置信水平 δ
        estimator: 用于计算置信区间的估计器

    Example:
        >>> agent = LUCBAgent(n_arms=5, delta=0.05)
        >>> best_arm = agent.run(env, max_rounds=10000)
    """
```

---

## 5. 配置管理

### 5.1 使用 dataclass 或 pydantic
```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 必需参数
    n_arms: int
    delta: float

    # 带默认值的参数
    estimator_type: Literal["ipw", "dr"] = "dr"
    max_rounds: int = 100_000
    audit_budget_ratio: float = 0.1

    # 可变默认值用 field()
    arm_means: list[float] = field(default_factory=list)

    def __post_init__(self):
        """验证配置"""
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError(f"delta 必须在 (0, 1) 范围内")
```

### 5.2 禁止过长的函数参数列表
如果函数参数超过 5 个，考虑使用配置对象：
```python
# 错误示例
def run_experiment(n_arms, delta, max_rounds, estimator_type,
                   audit_ratio, seed, verbose, log_path, ...):
    ...

# 正确示例
def run_experiment(config: ExperimentConfig, seed: int) -> ExperimentResult:
    ...
```

---

## 6. 随机数管理

### 6.1 禁止全局随机状态
```python
# 错误示例
np.random.seed(42)  # 全局状态，不可复现

# 正确示例
rng = np.random.default_rng(seed=42)
```

### 6.2 传递 RNG 生成器
```python
def sample_arm(
    ucb_values: np.ndarray,
    rng: np.random.Generator
) -> int:
    """随机打破平局"""
    max_val = ucb_values.max()
    candidates = np.where(ucb_values == max_val)[0]
    return rng.choice(candidates)

# 调用
rng = np.random.default_rng(config.seed)
arm = sample_arm(ucb_values, rng)
```

---

## 7. 代码组织

### 7.1 关注点分离
- **Logic（逻辑）**：算法和Agent，在 `algorithms/` 目录
- **Data（数据）**：环境和数据生成，在 `environments/` 目录
- **Math（数学）**：估计器和统计工具，在 `estimators/` 或 `common/` 目录

### 7.2 避免循环导入
```python
# 使用 TYPE_CHECKING 避免运行时导入
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..environments import BiasedJudgeEnv
```

### 7.3 模块的 `__init__.py`
导出公共 API：
```python
# src/bai_judge/estimators/__init__.py
from .ipw_estimator import IPWEstimator
from .dr_estimator import DREstimator

__all__ = ["IPWEstimator", "DREstimator"]
```

---

## 8. 测试规范

### 8.1 测试结构
测试目录结构镜像 `src/`：
```
tests/
├── unit/
│   ├── common/
│   │   └── test_confidence_sequences.py
│   ├── bai_judge/
│   │   └── test_ipw_estimator.py
│   └── policy_learning/
│       └── test_value_bounds.py
└── integration/
    └── test_full_experiment.py
```

### 8.2 Coverage 测试（关键）
每个新的 Martingale/CS 实现**必须**先写 coverage probability 测试：
```python
def test_empirical_bernstein_coverage():
    """验证 CS 的 coverage probability >= 1 - delta"""
    n_trials = 1000
    delta = 0.05
    coverage_count = 0

    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        samples = rng.uniform(0, 1, size=100)
        true_mean = 0.5

        cs = EmpiricalBernsteinCS(delta=delta)
        lower, upper = cs.compute(samples)

        if lower <= true_mean <= upper:
            coverage_count += 1

    empirical_coverage = coverage_count / n_trials
    # 允许一些统计波动
    assert empirical_coverage >= 1 - delta - 0.02, \
        f"Coverage {empirical_coverage} < {1 - delta - 0.02}"
```

### 8.3 最低覆盖率要求
- 核心组件（CS、估计器、算法）：≥ 80%
- 工具函数：≥ 60%

---

## 9. 日志和调试

### 9.1 使用 logging 而非 print
```python
import logging

logger = logging.getLogger(__name__)

def run_experiment(config: ExperimentConfig):
    logger.info(f"开始实验: {config.n_arms} arms, delta={config.delta}")
    ...
    logger.debug(f"Round {t}: best_arm={best}, challenger={challenger}")
```

### 9.2 进度显示
对于长时间运行的实验，使用 tqdm：
```python
from tqdm import tqdm

for t in tqdm(range(max_rounds), desc="LUCB"):
    ...
```

---

## 10. 提交规范

### 10.1 Commit Message 格式
```
type(scope): message
```

**类型**：
- `feat`: 新功能
- `fix`: Bug 修复
- `math`: 数学/理论修正
- `exp`: 实验相关
- `refactor`: 重构
- `docs`: 文档
- `test`: 测试

**示例**：
```
feat(bai_judge): implement DR estimator with asymptotic variance
fix(cs): correct boundary condition in empirical bernstein
math(policy): fix variance term in value bound derivation
exp(bai): add simulation for failure mode demonstration
```
