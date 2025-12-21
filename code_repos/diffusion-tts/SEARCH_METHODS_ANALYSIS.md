# Diffusion-TTS Search Methods 实现分析

本文档详细分析了 `code_repos/diffusion-tts/edm/main.py` 中实现的各种搜索方法。

## 核心组件

### 1. `step` 函数 (82-96行)
所有方法的底层去噪函数：
```python
def step(x_cur, t_cur, t_next, i, eps_i, class_labels_for_step=None):
    # EDM 采样步骤：添加噪声 -> 去噪 -> Heun校正
    # 返回: (x_next, denoised)
```

### 2. 时间步序列 (78-80行)
```python
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * 
          (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
```

### 3. 图像转换 (126行, 869行)
```python
image_for_scoring = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
# 假设 x_next 值域为 [-1, 1]
```

---

## 方法详解

### 1. NAIVE (862-866行)

**最简单的方法**：标准扩散采样

**算法**：
```python
for i, (t_cur, t_next) in enumerate(t_steps[:-1], t_steps[1:]):
    x_cur = x_next
    eps_i = torch.randn_like(x_next)  # 随机噪声
    x_next, _ = step(x_cur, t_cur, t_next, i, eps_i, class_labels)
```

**特点**：
- 每个时间步随机采样噪声
- 无需评分器
- 计算成本最低：`num_steps * 2` NFE（每步两次模型调用）

---

### 2. REJECTION_SAMPLING (101-137行)
**Best-of-N 方法**

**算法**：
1. **扩展初始噪声**：对每个样本生成 N 个候选
   ```python
   x_next_expanded = x_next.repeat_interleave(N, dim=0)  # [batch_size * N, C, H, W]
   class_labels_expanded = class_labels.repeat_interleave(N, dim=0)
   ```

2. **完整采样所有候选**：所有候选运行完整去噪流程
   ```python
   for i, (t_cur, t_next) in enumerate(...):
       x_cur = x_next_expanded
       eps_i = torch.randn_like(x_cur)
       x_next_expanded, _ = step(x_cur, t_cur, t_next, i, eps_i, class_labels_expanded)
   ```

3. **评分并选择最好的**
   ```python
   image_for_scoring = (x_next_expanded * 127.5 + 128).clip(0, 255).to(torch.uint8)
   scores = scorer(image_for_scoring, class_labels_expanded, timesteps)
   scores = scores.view(batch_size, N)  # [batch_size, N]
   best_indices = scores.argmax(dim=1)
   x_next = torch.stack([x_next_reshaped[i, idx] for i, idx in enumerate(best_indices)])
   ```

**特点**：
- 并行生成 N 个完整轨迹
- 在所有轨迹完成后才选择
- **NFE**: `num_steps * N * 2` + `N` (评分)
- **参数**: `N` (候选数量)

---

### 3. BEAM_SEARCH (138-404行)

**算法**：维护 k 个 beam，每步扩展 b 个候选，保留 top k

**关键数据结构**：
```python
x_curs = [(x_next_expanded, None)]  # [(x, x0)] 格式，维护当前所有 beam
sample_indices = torch.arange(batch_size).repeat_interleave(k)  # 跟踪每个beam属于哪个样本
```

**主循环** (每时间步)：
1. **扩展每个 beam**：为每个 beam 生成 b 个候选
   ```python
   x_expanded = x_beam.repeat(b, 1, 1, 1)  # [b, C, H, W]
   eps_i = torch.randn_like(x_expanded)
   x_candidate_beam, x0_candidate_beam = step_with_microbatch(...)
   ```

2. **评分所有候选**（使用 x0 估计）
   ```python
   x_for_scoring = (x0_beam_candidates * 127.5 + 128).clip(0, 255).to(torch.uint8)
   beam_score = scorer(x_for_scoring, class_labels_flat, timesteps)
   ```

3. **选择 top k**
   ```python
   scores = scores.view(batch_size, k * b)  # [batch_size, k*b]
   topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)
   # 收集 top k 候选
   ```

**特点**：
- 使用 `step_with_microbatch` 处理大 batch 以避免 OOM
- 使用 x0（去噪估计）评分，而非最终结果
- **NFE**: `num_steps * (k * b * 2 + k * b)` = `num_steps * k * b * 3`
- **参数**: `b` (branching factor), `k` (beam width)

---

### 4. MCTS (405-713行)

**算法**：Monte Carlo Tree Search with UCB

**关键数据结构**（每个样本独立）：
```python
all_children = [{}]  # 节点的子节点列表
all_parent = [{}]    # 节点的父节点
all_reward = [{}]    # 节点累积奖励
all_visit = [{}]     # 节点访问次数
all_roots = []       # 当前根节点（每步更新）
```

**主循环**（每时间步）：
1. **扩展根节点**：生成 b 个初始候选
   ```python
   for noise_idx in range(b):
       x_next_candidate, _ = step(expansion_batch, t_cur, t_next, i, noise_batch, ...)
       all_children[sample_idx][root_key].append((x_next_candidate, x_next_key))
   ```

2. **N 次模拟**：
   - **Selection**：使用 UCB 选择路径
     ```python
     exploitation = reward[child_key] / visit[child_key]
     exploration = sqrt(2 * log(visit[parent_key]) / visit[child_key])
     ucb = exploitation + exploration
     ```
   - **Expansion**：展开未探索的节点
   - **Simulation**：从当前节点到终点进行确定性采样（零噪声）
     ```python
     temp_x, _ = step(..., torch.zeros_like(...))  # 零噪声 = 确定性
     ```
   - **Backpropagation**：将奖励回传到路径上所有节点
     ```python
     for _, node_key, _, _ in path:
         reward[node_key] += reward_value
         visit[node_key] += 1
     ```

3. **选择最佳子节点**作为下一时间步的根
   ```python
   best_reward = max(avg_reward for all children)
   all_roots[sample_idx] = best_child
   ```

**特点**：
- 使用 UCB 平衡探索与利用
- Simulation 使用零噪声（确定性）而非随机噪声
- 批处理模拟以提高GPU利用率
- **NFE**: `num_steps * (b * 2 + N * simulation_steps * 2)`
- **参数**: `b` (branching factor = N), `S` (simulations = N)

---

### 5. ZERO_ORDER / EPS_GREEDY (714-860行)

**算法**：在每个时间步进行 K 次局部搜索迭代（K 固定）

**主循环**（每时间步）：
1. **初始化 pivot noise**
   ```python
   pivot_noise = torch.randn_like(x_cur)  # 每个时间步重新初始化
   ```

2. **K 次迭代**：
   ```python
   for k in range(K):
       # 生成 N 个候选噪声
       for n in range(N):
           if torch.rand(1) < (1 - eps):  # ZERO_ORDER: eps=0, 总是使用扰动
               # 局部搜索：扰动 pivot
               random_direction = torch.randn_like(base_noise)
               random_direction = random_direction / norm(random_direction)  # 归一化为单位向量
               scale = torch.rand(...) * lambda_param
               candidate_noise = base_noise + scale * random_direction
           else:  # EPS_GREEDY: 概率 eps 使用新噪声
               candidate_noise = torch.randn_like(x_cur)  # 全局探索
       
       # 对所有候选去噪并评分
       all_noises = torch.cat(candidate_noises, dim=0)  # [N*batch_size, C, H, W]
       x_cur_expanded = x_cur.repeat(N, 1, 1, 1)
       x_candidates, x0_candidates = step(x_cur_expanded, t_cur, t_next, i, all_noises, ...)
       
       # 使用 x0 评分
       x_for_scoring = (x0_candidates.reshape(-1, ...) * 127.5 + 128).clip(0, 255).to(torch.uint8)
       scores = scorer(x_for_scoring, scorer_class_labels, timesteps)
       scores = scores.reshape(N, batch_size)
       
       # 选择最好的作为新的 pivot
       best_indices = scores.argmax(dim=0)
       pivot_noise = torch.stack([candidate_noises_batch[best_idx, batch_idx] ...])
   
   # 使用最终 pivot 进行去噪
   x_next, _ = step(x_cur, t_cur, t_next, i, pivot_noise, class_labels)
   ```

**ZERO_ORDER vs EPS_GREEDY**：
- **ZERO_ORDER**: `eps = 0`，总是使用扰动 pivot（纯局部搜索）
- **EPS_GREEDY**: `eps > 0`（默认 0.4），以概率 `eps` 使用新鲜高斯噪声（全局探索）

**关键参数**：
- `lambda_param`: 扰动半径（已缩放 `* sqrt(3 * 64 * 64)`）
- `N`: 每步候选数量
- `K`: 每时间步迭代次数
- `eps`: 探索概率（仅 EPS_GREEDY）

**特点**：
- 使用 x0 估计评分（而非完整采样）
- 每个时间步独立进行局部搜索
- **NFE**: `num_steps * (K * N * 2 + 1 * 2)` = `num_steps * (K * N * 2 + 2)`
- 在极端时间步进行全局探索，在中间时间步进行局部利用

### 6. EPS_GREEDY_1 (新方法，861行后)

**算法**：EPS_GREEDY 的变体，使用**自适应 K 值**

**关键差异**：
- **前一半时间步**：`K = 5`（更多迭代，更精细搜索）
- **后一半时间步**：`K = 3`（较少迭代，快速收敛）

**实现逻辑**：
```python
num_steps_total = len(t_steps) - 1
half_point = num_steps_total // 2

for i, (t_cur, t_next) in enumerate(...):
    K = 5 if i < half_point else 3  # 动态调整K值
    # 其余逻辑与EPS_GREEDY相同
```

**特点**：
- 在去噪早期（噪声较大）进行更充分的搜索（K=5）
- 在去噪后期（接近完成）减少搜索次数（K=3），加速收敛
- 保持 EPS_GREEDY 的所有其他特性（eps 概率、局部+全局探索等）
- **NFE**: `half_steps * (5 * N * 2 + 2) + half_steps * (3 * N * 2 + 2)`
- 相比固定 K=4：前期搜索更充分，后期更高效

---

## 方法对比

| 方法 | NFE 复杂度 | 特点 | 适用场景 |
|------|-----------|------|---------|
| **NAIVE** | `O(steps)` | 无搜索，随机采样 | Baseline |
| **REJECTION** | `O(steps * N)` | 完整轨迹后选择 | 简单但昂贵 |
| **BEAM** | `O(steps * k * b)` | 维护多个候选轨迹 | 平衡性能与成本 |
| **MCTS** | `O(steps * (b + N * sim))` | UCB 树搜索 | 复杂奖励函数 |
| **ZERO_ORDER** | `O(steps * K * N)` | 局部噪声优化 | 论文主要方法 |
| **EPS_GREEDY** | `O(steps * K * N)` | 局部+全局探索 | 论文主要方法（推荐） |
| **EPS_GREEDY_1** | `O(steps * K * N)` | 自适应K值(5→3) | 改进版EPS_GREEDY |

---

## 关键实现细节

### 1. 评分时机
- **REJECTION**: 在完整采样后评分
- **BEAM/MCTS/ZERO_ORDER**: 使用 **x0 估计**（denoised）评分，而非完整采样

### 2. 图像格式转换
所有方法都使用相同的转换：
```python
image_for_scoring = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
# 假设输入在 [-1, 1] 范围内
```

### 3. 内存优化
- **BEAM**: 使用 `step_with_microbatch` 处理大 batch
- **MCTS**: 批处理模拟，mini-batch 处理样本
- 所有方法都使用 `torch.cuda.empty_cache()` 释放内存

### 4. 噪声处理
- **可预计算噪声**：所有方法支持 `precomputed_noise` 参数，用于可重复实验
- **噪声形状**：
  - REJECTION: `[batch_size, max_N, C, H, W]` per timestep
  - BEAM: `[batch_size * k, C, H, W]` per timestep
  - MCTS: `[batch_size, b, C, H, W]` per timestep
  - ZERO_ORDER: `[batch_size, K, N, C, H, W]` per timestep (可选)

---

## 代码位置总结

- **NAIVE**: 862-866行
- **REJECTION_SAMPLING**: 101-137行
- **BEAM_SEARCH**: 138-404行（含 `step_with_microbatch` 161-212行）
- **MCTS**: 405-713行
- **ZERO_ORDER / EPS_GREEDY**: 714-860行
- **EPS_GREEDY_1**: 861行后（新方法）
- **Scorers**: `code_repos/diffusion-tts/edm/scorers.py`

---

## 实验结果

### EPS_GREEDY_1 方法在不同 K 值组合下的性能对比 (ImageNet-64)

| K1 | K2 | ImageNet Score | Compressibility Score |
|----|----|----------------|----------------------|
| 6  | 4  | 0.9223 ± 0.0429 | 0.5969 ± 0.0209 |
| 5  | 5  | 0.9251 ± 0.0341 | 0.5968 ± 0.0166 |
| 4  | 6  | 0.9271 ± 0.0298 | 0.5787 ± 0.0199 |

**观察结果**：
- **ImageNet 分数**：随着 K1 减小、K2 增大（前一半步骤迭代减少，后一半迭代增加），ImageNet 分数呈上升趋势（0.9223 → 0.9251 → 0.9271），且标准差降低（0.0429 → 0.0341 → 0.0298），表明结果更稳定。
- **Compressibility 分数**：K=(5,5) 和 K=(6,4) 的压缩性分数相近（约 0.5968-0.5969），而 K=(4,6) 的压缩性分数明显较低（0.5787），标准差也稍高（0.0199）。
- **结论**：K=(4,6) 配置在 ImageNet 分类任务上表现最佳，但在压缩性指标上略逊于其他配置。K=(5,5) 配置在两类指标上表现均衡。

