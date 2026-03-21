# ⚙️ 配置管理规范

## 配置文件结构

**配置位置**: `configs/`

**组织方式**:
```
configs/
├── base/                      # 基础配置
│   ├── ppo_base.yml
│   └── bc_base.yml
├── experiments/               # 实验配置
│   ├── ppo_rudder_mmpp.yml
│   ├── ppo_ares.yml
│   └── comparison_baseline.yml
├── production/                # 生产配置
│   └── ppo_production.yml
└── templates/                 # 配置模板 (未来)
    ├── ppo_stability_focused.yml
    ├── ppo_performance_optimized.yml
    └── ppo_fast_iteration.yml
```

## 配置参数组织

**核心配置类** (定义在 `vidur/config/config.py`):
```python
@dataclass
class PPOGlobalSchedulerModularConfig:
    # 网络架构
    state_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2

    # PPO超参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2

    # 训练参数
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_rollout_steps: int = 512

    # 奖励配置
    reward_mode: str = "differential"
    latency_weight: float = 1.0
    throughput_weight: float = 0.0

    # ... (共50+参数)
```

## 多环境配置策略

**环境变量注入**:
```bash
# 开发环境
export ENV=dev
export CHECKPOINT_DIR=outputs/checkpoints/dev/

# 预发布环境
export ENV=staging
export CHECKPOINT_DIR=outputs/checkpoints/staging/

# 生产环境
export ENV=prod
export CHECKPOINT_DIR=outputs/checkpoints/production/
```

**配置继承** (YAML):
```yaml
# configs/experiments/my_experiment.yml
base_config: configs/base/ppo_base.yml

# 覆盖特定参数
scheduler_config:
  learning_rate: 1e-4
  batch_size: 512

# 环境特定配置
env_overrides:
  dev:
    num_rollout_steps: 128  # 快速迭代
  staging:
    num_rollout_steps: 512  # 完整评估
  prod:
    num_rollout_steps: 1024 # 最佳性能
```

## 配置验证

**运行时验证**:
```python
class ConfigValidator:
    @staticmethod
    def validate_ppo_config(config):
        assert 0 < config.gamma <= 1.0, "gamma must be in (0, 1]"
        assert config.clip_epsilon > 0, "clip_epsilon must be positive"
        assert config.batch_size > 0, "batch_size must be positive"
        # ... 更多验证规则

# 使用
validator = ConfigValidator()
validator.validate_ppo_config(config)
```

**配置一致性检查** (未来工具):
```bash
# 验证配置文件格式正确
python scripts/utils/validate_training_configs.py configs/experiments/my_experiment.yml

# 检查配置参数合理性
python scripts/utils/validate_config_sanity.py configs/experiments/my_experiment.yml
```

## 配置文档化

**参数文档规范**:
```python
@dataclass
class PPOConfig:
    gamma: float = 0.99
    """Discount factor for future rewards.

    Range: (0, 1]
    Typical: 0.95-0.995 for episodic tasks, 0.99-0.999 for continuing tasks
    Impact: Higher values → more long-term focus
    """

    clip_epsilon: float = 0.2
    """PPO clipping range for policy ratio.

    Range: [0.1, 0.3]
    Typical: 0.2
    Impact: Smaller values → more conservative updates
    """
```

**配置引用**:
- 完整参数索引: `docs/guides/configuration_reference.md` (Phase 5待创建)
- 参数优化指南: `docs/guides/hyperparameter_tuning_guide.md` (Phase 3待创建)
- 配置模板: `configs/templates/` (Phase 5待创建)

## 配置版本控制

**Git管理**:
```bash
# 配置文件纳入版本控制
git add configs/experiments/my_experiment.yml

# 实验特定配置自动保存
# 训练脚本会自动保存到 outputs/experiments/{date}/{exp_name}/config.yaml
```

**配置追溯**:
```yaml
# outputs/experiments/2026-01/my_exp/config.yaml
# 自动生成的完整配置快照
experiment_name: my_exp
created: 2026-01-17T12:00:00+00:00
git_hash: 94f8171b22d7f43a
git_branch: toymodel_PPO
config_source: configs/experiments/my_experiment.yml

# ... 完整的参数配置 ...
```
