# 🧪 测试与CI/CD规范

## 测试框架

**测试位置**: `tests/`

**测试类型**:
- 单元测试: 测试单个函数/类的功能
- 集成测试: 测试模块间交互
- 梯度测试: 验证反向传播正确性
- 端到端测试: 验证完整训练流程

**现有测试** (16个测试文件):
| 测试类别 | 文件示例 |
|---------|----------|
| 组件测试 | `test_enhanced_state_builder_simple.py` |
| 梯度测试 | `test_ppo_gradients.py`, `test_real_gradients.py` |
| 集成测试 | `test_integration_gradients.py` |
| 监控测试 | `test_tensorboard_*.py`, `test_metrics_exporter.py` |
| 功能测试 | `test_inference_mode.py`, `test_reward_improvements.py` |

## 运行测试

**基本用法**:
```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_ppo_gradients.py

# 运行特定测试函数
pytest tests/test_ppo_gradients.py::test_gradient_flow

# 显示详细输出
pytest tests/ -v

# 显示print输出
pytest tests/ -s
```

**覆盖率要求**:
- 核心模块 (src/core/): 目标覆盖率 ≥ 60%
- 新功能: 必须包含单元测试
- Bug修复: 必须添加回归测试

## CI/CD流程

**GitHub Actions工作流**: `.github/workflows/lint.yml`

**自动检查**:
1. **代码格式检查** (black)
   ```bash
   make lint/black
   ```

2. **Import排序检查** (isort)
   ```bash
   make lint/isort
   ```

3. **触发条件**:
   - Push到main分支
   - Pull request到main分支

**本地预检查**:
```bash
# 自动格式化代码
make format

# 检查格式是否符合规范
make lint

# 单独检查black
make lint/black

# 单独检查isort
make lint/isort
```

## Pre-commit钩子

**推荐配置** (可选但强烈建议):
```bash
# 安装pre-commit
pip install pre-commit

# 配置git hook (如果项目有.pre-commit-config.yaml)
pre-commit install

# 手动运行所有检查
pre-commit run --all-files
```

**最小要求**: 提交前必须运行 `make format` 和 `make lint`

## 测试编写规范

**测试文件命名**:
- `test_[module_name].py` - 模块功能测试
- `test_[feature]_integration.py` - 集成测试
- `debug_[issue].py` - 临时调试脚本（不提交到git）

**测试函数命名**:
```python
def test_[function_name]_[scenario]():
    """Test [specific behavior] when [condition]."""
    pass

# 示例
def test_ppo_update_normal_advantages():
    """Test PPO update with normally distributed advantages."""
    pass
```

**测试结构** (AAA模式):
```python
def test_example():
    # Arrange: 准备测试数据和环境
    config = create_test_config()
    trainer = PPOTrainer(config)

    # Act: 执行被测试的操作
    result = trainer.train_step()

    # Assert: 验证结果
    assert result.loss < 1.0
    assert result.metrics['kl'] < 0.05
```

## 持续集成最佳实践

**提交前检查清单**:
- [ ] 代码已格式化 (`make format`)
- [ ] 通过lint检查 (`make lint`)
- [ ] 相关测试通过 (`pytest tests/test_*.py`)
- [ ] 新功能已添加测试
- [ ] 文档已更新

**CI失败处理**:
1. 查看GitHub Actions日志
2. 本地复现失败 (`make lint`)
3. 修复问题
4. 重新提交
