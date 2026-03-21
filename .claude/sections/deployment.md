# 🚀 部署与生产规范

## 生产就绪检查清单

**部署前必须满足**:
- [ ] 所有测试通过 (`pytest tests/`)
- [ ] Lint检查通过 (`make lint`)
- [ ] Checkpoint已验证 (能正常加载和推理)
- [ ] Baseline对比完成 (性能不低于baseline)
- [ ] 配置文件已审核 (超参数合理)
- [ ] 文档已更新 (CHANGELOG, 配置说明)
- [ ] 实验元数据完整 (config.yaml, git_hash.txt)

## 环境管理

**环境分离**:
```
development/     # 开发环境 (本地笔记本)
├── 快速迭代
├── 调试工具完整
└── 可以失败

staging/         # 预发布环境 (测试服务器)
├── 生产配置
├── 完整数据集
└── 性能验证

production/      # 生产环境 (推理服务)
├── 经过验证的checkpoint
├── 监控告警
└── SLA保证
```

**配置隔离**:
```bash
# 开发环境
./scripts/training/ppo/train_ppo_shared.sh --env dev

# 预发布环境
./scripts/training/ppo/train_ppo_shared.sh --env staging

# 生产环境
./scripts/training/ppo/train_ppo_shared.sh --env prod
```

## Checkpoint管理

**Checkpoint版本控制**:
```
outputs/checkpoints/
├── latest.pt                    # 软链接，指向最新checkpoint
├── checkpoint_step_00256099.pt  # 训练中检查点
├── production/                  # 生产环境专用
│   ├── v1.0.0_sqf_baseline.pt
│   └── v1.1.0_ppo_rudder.pt
└── archive/                     # 历史备份
    ├── 2026-01-15/
    └── 2026-01-16/
```

**Checkpoint验证流程**:
```bash
# 1. 加载测试
python -c "import torch; ckpt = torch.load('outputs/checkpoints/latest.pt'); print(ckpt.keys())"

# 2. 推理测试
python scripts/evaluation/test_checkpoint_inference.py --checkpoint outputs/checkpoints/latest.pt

# 3. 性能对比
./scripts/evaluation/compare_with_baseline.sh --checkpoint outputs/checkpoints/latest.pt
```

## 回滚策略

**快速回滚程序**:
```bash
# 1. 停止当前训练
pkill -f train_ppo

# 2. 恢复上一个稳定checkpoint
cp outputs/checkpoints/archive/2026-01-16/checkpoint_step_*.pt outputs/checkpoints/latest.pt

# 3. 从checkpoint恢复训练
./scripts/training/ppo/train_ppo_shared.sh --resume outputs/checkpoints/latest.pt

# 4. 验证恢复成功
sqlite3 outputs/runs/ppo_training/*/metrics.db "SELECT step, ev, kl FROM ppo_metrics ORDER BY step DESC LIMIT 5"
```

**回滚决策标准**:
- 训练指标连续3次评估下降 > 20%
- 出现严重异常 (KL > 0.5, entropy < 0.01)
- 生产推理性能下降 > 10%
- 发现数据/配置错误

## 部署流程

**标准部署流程**:
```
1. 实验阶段
   - 在dev环境验证新功能
   - 记录实验元数据

2. 预发布阶段
   - 在staging环境全量测试
   - 与baseline对比
   - 生成性能报告

3. 生产部署
   - 备份当前checkpoint
   - 部署新checkpoint
   - 监控关键指标
   - 准备回滚方案

4. 验证阶段
   - A/B测试
   - 性能监控 (首24小时)
   - 逐步扩大流量
```

## SLA定义

**关键性能指标**:
| 指标 | 目标值 | 说明 |
|------|--------|------|
| 推理延迟 (p50) | < 10ms | 中位数延迟 |
| 推理延迟 (p99) | < 50ms | 99分位延迟 |
| 吞吐量 | > baseline × 1.1 | 超过baseline 10% |
| 可用性 | 99.9% | 月故障时间 < 43分钟 |
| 调度决策质量 | 平均延迟 < SQF | 超越启发式基线 |

**监控告警**:
```python
# 示例告警规则 (伪代码)
if latency_p99 > 50ms for 5 minutes:
    alert("High latency detected")

if throughput < baseline * 0.9 for 10 minutes:
    alert("Performance degradation")
```

## 灾难恢复

**备份策略**:
- Checkpoint: 每天自动备份到 `outputs/checkpoints/archive/`
- 配置文件: Git版本控制
- Metrics数据库: 每周备份到持久化存储

**恢复测试** (每月执行):
1. 模拟checkpoint丢失
2. 从备份恢复
3. 验证训练可恢复
4. 记录恢复时间
