# 🐛 调试与错误处理规范

## 调试工作流

**系统性调试流程**:
```
1. 识别症状
   ↓
2. 检查metrics数据库 (SQLite)
   ↓
3. 查看TensorBoard曲线
   ↓
4. 分析训练日志
   ↓
5. 代码级调试 (pdb/logging)
   ↓
6. 性能分析 (Profiler)
```

## 常见训练失败诊断

**参考文档**: `docs/research/rl_training_debugging_guide.md`

**症状诊断表**:
| 症状 | 可能原因 | 诊断方法 |
|------|----------|----------|
| 熵爆炸 (entropy → 高) | 奖励信号弱/优势噪声大 | 检查 `explained_var`, 降低 `entropy_coef` |
| KL散度爆炸 (kl > 0.1) | 学习率过大/clip范围过大 | 降低 `lr`, 调整 `clip_epsilon` |
| 价值损失停滞 | Critic无法拟合回报 | 检查状态表示, 增加 `value_loss_coef` |
| 解释方差为0 | 状态-回报相关性弱 | 检查奖励设计, 尝试RUDDER |
| 梯度消失/爆炸 | 网络深度/初始化问题 | 检查梯度范数, 使用 `grad_clip` |
| OOM (内存溢出) | batch_size过大/泄漏 | 降低batch_size, 检查detach() |

## 指标监控检查清单

**健康训练的特征**:
```python
# 正常范围 (参考docs/guides/training_monitoring_guide.md)
explained_var > 0.3      # Critic预测质量
approx_kl < 0.05         # 策略更新幅度
clipfrac < 0.30          # Clip触发比例
entropy > 0.1            # 保持探索
policy_loss: 稳定下降
value_loss: 稳定下降
```

**异常信号**:
- `explained_var < 0`: Critic严重失效
- `approx_kl > 0.1`: 策略更新过于激进
- `clipfrac > 0.5`: Clip失效，学习率过大
- `entropy < 0.01`: 策略过于确定，探索不足
- `entropy > 2.0`: 策略退化为随机

## 日志与监控

**日志级别标准**:
```python
import logging

# 模块级logger
logger = logging.getLogger(__name__)

# 级别使用规范
logger.debug("Detailed state: {state}")      # 详细调试信息
logger.info("Training step 1000 completed")  # 关键流程节点
logger.warning("KL divergence high: {kl}")   # 异常但可恢复
logger.error("Checkpoint load failed")       # 错误但程序继续
logger.critical("CUDA out of memory")        # 致命错误
```

**日志位置**:
- 训练日志: `outputs/runs/ppo_training/YYYYMMDD_HHMMSS/`
- Metrics数据库: `outputs/runs/ppo_training/YYYYMMDD_HHMMSS/metrics.db`
- TensorBoard: `tensorboard --logdir outputs/runs/ppo_training/`

## 错误处理策略

**核心原则** (八荣八耻第3条):
- 🔥 **禁止静默捕获异常** - 让错误traceback显示
- 🔥 **禁止fallback方案** - 直接暴露问题

**正确的错误处理**:
```python
# ✅ 让错误自然抛出 (发现问题根源)
def load_checkpoint(path):
    checkpoint = torch.load(path)  # FileNotFoundError会直接抛出
    return checkpoint

# ❌ 禁止静默fallback
def load_checkpoint_bad(path):
    try:
        checkpoint = torch.load(path)
    except:
        checkpoint = {}  # 隐藏了真实问题！
    return checkpoint
```

**允许的异常处理场景**:
```python
# ✅ 资源清理
try:
    train()
finally:
    cleanup_resources()

# ✅ 重试逻辑 (有明确上限和日志)
for attempt in range(3):
    try:
        result = network_request()
        break
    except NetworkError as e:
        logger.warning(f"Attempt {attempt} failed: {e}")
        if attempt == 2:
            raise  # 最后一次重试失败，抛出原始异常

# ✅ 用户输入验证
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
```

## 调试工具

**Python调试器**:
```python
# 在代码中设置断点
import pdb; pdb.set_trace()

# IPython调试器 (更友好)
import ipdb; ipdb.set_trace()
```

**Profiling性能分析**:
```python
# PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**参考文档**:
- 详细调试指南: `docs/research/rl_training_debugging_guide.md`
- 监控指南: `docs/guides/training_monitoring_guide.md`
- 问题分析: `docs/research/problems/001_ppo_training_pipeline_analysis.md`
