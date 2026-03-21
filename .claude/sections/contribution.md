# 🤝 贡献指南

## PR流程

**标准流程**:
```
1. Fork仓库 → 创建分支
   ↓
2. 实现功能 → 编写测试
   ↓
3. 提交代码 → 自检清单
   ↓
4. 创建PR → 等待审核
   ↓
5. 修改反馈 → 合并代码
```

**PR模板**: `.github/PULL_REQUEST_TEMPLATE.md`

## 分支命名规范

**分支类型**:
```
feat/[feature-name]       # 新功能
fix/[bug-description]     # Bug修复
docs/[doc-update]         # 文档更新
exp/[experiment-name]     # 实验分支
refactor/[module-name]    # 重构
```

**示例**:
```bash
git checkout -b feat/add-ares-credit-assignment
git checkout -b fix/kl-divergence-explosion
git checkout -b docs/update-training-guide
git checkout -b exp/test-new-reward-function
```

## Commit消息规范

**格式**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型 (type)**:
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式 (不影响功能)
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具链

**示例**:
```
feat(credit_assignment): Add ARES attention-based credit assignment

Implement attention-based credit assignment mechanism as described
in the ARES paper. This addresses the delayed reward problem more
effectively than RUDDER for our scheduling task.

- Add AresModule class
- Integrate with PPOTrainer
- Add configuration parameters
- Update documentation

Closes #123
```

## PR标题规范

**PR模板要求的前缀**:
- `[Bugfix]`: Bug修复
- `[CI/Build]`: CI/CD改进
- `[Doc]`: 文档更新
- `[Model]`: 模型相关
- `[Profiling]`: 性能分析
- `[Core]`: 核心逻辑
- `[Misc]`: 其他

**示例**:
```
[Feat] Add ARES credit assignment mechanism
[Bugfix] Fix KL divergence explosion in high-frequency training
[Doc] Update training monitoring guide with metrics_db examples
```

## Code Review标准

**审核检查项**:
- [ ] 代码符合编程规范 (八荣八耻原则)
- [ ] 通过所有测试和lint检查
- [ ] 新功能有测试覆盖
- [ ] 文档已更新
- [ ] 无明显性能回归
- [ ] Commit历史清晰
- [ ] PR描述清楚功能和动机

**大型PR要求** (> 500 LOC):
- 需要先提交RFC (GitHub Issue)
- 技术设计讨论
- 分阶段提交

## 废弃功能策略

**废弃流程**:
```
1. 标记为deprecated
   ↓
2. 文档记录废弃原因
   ↓
3. 至少保留2个版本
   ↓
4. 迁移指南
   ↓
5. 最终移除
```

**废弃标记**:
```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated and will be removed in v2.0. "
        "Use new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... 旧实现 ...
```

**文档位置**: `docs/deprecated/deprecated_[component].md`

## 新贡献者指南

**第一次贡献**:
1. 阅读 `docs/plan/beating_sqf_updated_plan.md` 了解项目目标
2. 查看 `docs/README.md` 熟悉文档结构
3. 从 "good first issue" 标签的Issue开始
4. 参考现有代码风格

**获取帮助**:
- 技术问题: 提Issue并标记 `question`
- 设计讨论: 提Issue并标记 `RFC`
- 文档问题: 提Issue并标记 `documentation`
