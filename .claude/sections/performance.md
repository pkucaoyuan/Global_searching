# ⚡ 性能优化规范

## Profiling工作流

**性能分析步骤**:
```bash
# 1. 确认性能瓶颈存在
python -m cProfile -o profile.stats train.py
python -m pstats profile.stats

# 2. PyTorch Profiler详细分析
python scripts/profiling/profile_training.py --steps 100

# 3. 查看profiling结果
# TensorBoard可视化
tensorboard --logdir outputs/profiling/

# 4. 内存分析
python -m memory_profiler train.py
```

**参考文档**: `docs/guides/profiling.md`

## 关键性能参数

**训练性能优化**:
| 参数 | 默认值 | 优化建议 | 影响 |
|------|--------|----------|------|
| `batch_size` | 256 | GPU显存允许尽量大 | 吞吐量 ↑, 显存 ↑ |
| `num_rollout_steps` | 512 | 降低可减少延迟 | 延迟 ↓, 样本效率 ↓ |
| `num_workers` | 4 | 设为CPU核心数 | CPU利用率 ↑ |
| `pin_memory` | True | 使用CUDA时必开 | 数据传输 ↑ |
| `use_fused_adam` | True | 推荐 | 优化器速度 ↑ 20% |

**推理性能优化**:
```python
# 推理模式优化
model.eval()
with torch.no_grad():
    action = model(state)

# JIT编译 (首次推理后加速)
model = torch.jit.script(model)

# 半精度推理 (推理速度 ↑ 2x)
model.half()
state = state.half()
```

## 内存优化

**常见内存泄漏原因**:
1. **未detach的hidden states**
   ```python
   # ❌ 错误: 保留计算图
   self.hidden = lstm_output

   # ✅ 正确: 分离计算图
   self.hidden = lstm_output.detach()
   ```

2. **未清理的rollout buffer**
   ```python
   # ✅ 定期清理
   def update(self):
       # ... PPO更新 ...
       self.rollout_buffer.clear()
   ```

3. **TensorBoard过度记录**
   ```python
   # ❌ 每步记录大张量
   writer.add_histogram('states', states, step)

   # ✅ 定期记录
   if step % 100 == 0:
       writer.add_histogram('states', states, step)
   ```

**内存监控**:
```python
import torch

# 查看GPU显存使用
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 清理缓存
torch.cuda.empty_cache()
```

## 分布式训练

**单机多GPU** (DataParallel):
```python
# 简单但不推荐 (性能差)
model = torch.nn.DataParallel(model)
```

**单机多GPU** (DistributedDataParallel, 推荐):
```bash
# 4卡训练
torchrun --nproc_per_node=4 train.py
```

**多机多GPU** (未来扩展):
```bash
# 2台机器，每台4卡
torchrun --nnodes=2 --nproc_per_node=4 --master_addr=<MASTER_IP> train.py
```

## 缩放指南

**垂直扩展** (单机性能提升):
- GPU升级: V100 → A100 → H100
- 内存升级: 支持更大batch_size
- NVMe SSD: 加速数据加载

**水平扩展** (多机分布式):
- 适用场景: 数据集巨大，单机训练时间 > 1周
- 通信开销: 注意梯度同步延迟
- 调试复杂度: 分布式bug难以定位

**何时优化**:
- ✅ 训练时间 > 12小时 → 考虑性能优化
- ✅ GPU利用率 < 70% → Profiling找瓶颈
- ✅ 显存占用 < 50% → 增大batch_size
- ❌ 过早优化 → "不要优化还没写的代码"

**优化优先级**:
```
1. 算法优化 (最大收益)
   - 更好的奖励设计
   - 更高效的状态表示

2. 超参数优化 (中等收益)
   - batch_size, learning_rate

3. 代码优化 (有限收益)
   - Profiling, GPU kernel优化

4. 硬件升级 (最简单但最贵)
   - 更好的GPU, 更多显存
```
