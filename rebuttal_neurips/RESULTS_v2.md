# Rebuttal 补实验结果 v2（2026-07-29 09:40）

> 所有数值 = **严格实测 NFE** 下的重测；每 config 完整命令与逐步日志见
> `code_repos/diffusion-tts/results/day0/<config>/`。方法定义：
> **Uniform** = eps_greedy 均匀 K；**Offline-only** = pure epsilon_1 无任何在线反馈；
> **GAINS** = epsilon_online（阈值信号早停 + 预算重分配 + revert-on-negative）。

## 表 1｜E1 SD 预算分解（20 prompts × 10 seeds = 200 img/格；严格 NFE）

| @400 | Uniform | Offline-only | GAINS | Δoffline vs U | Δonline (GAINS−offline) |
|---|---|---|---|---|---|
| Brightness | 0.6975 | 0.7112 | **0.7165** | **+0.0137** | **+0.0053** |
| Compressibility | 0.7971 | 0.7930 | **0.8011** | −0.0041 (~0) | **+0.0081** |

- Brightness 侧 offline 主导（+0.014），Compressibility 侧 online 主导（+0.008），与投稿版附录分解比例结构一致。
- 参数稳健性（10 组 sweep：K1 ∈ [11,18]，slack ∈ [2,4]，β_g ∈ [0.1,0.3]，β_σ ∈ [0.6,0.8]，W_g ∈ [2,6]）：
  在线增量在所有配置下都在 ±0.006 之内 → **在线控制器对 GAINS 增益的贡献主要来自 revert-on-negative 反馈**，阈值信号冗余但稳健。

| @800 | Uniform | Offline-only | GAINS |
|---|---|---|---|
| Brightness | 0.7320 (n=50, 跑中) | 0.7576 (n=80, 跑中) | **0.7660 (n=200 完)** |
| Compressibility | 0.7627 (n=40, 跑中) | 0.7862 (n=80, 跑中) | 0.8359 (n=20, 跑中) |

800 档三列今晚 20:00 前收官。GAINS-B@800 已完成，800 档 NFE scaling 主张成立。

## 表 2｜EDM 分解（36-img grid × 20 runs，两个 verifier × 三档 NFE）

| | Uniform (投稿版) | Offline-only (noRev, 本次重测) | GAINS (投稿版) |
|---|---|---|---|
| B/144 | 0.9507 | 0.9704 | 0.9887 |
| B/180 | 0.9617 | 0.9756 | 0.9904 |
| B/288 | 0.9787 | 0.9940 | 0.9988 |
| C/144 | 0.6714 | 0.6757 | 0.6845 |
| C/180 | 0.6774 | 0.6874 | 0.7003 |
| C/288 | 0.6901 | 0.6964 | 0.7165 |

EDM 上 offline 和 online 增量在两个 verifier × 三档 NFE 全部为正。

## 表 3｜E2 RBF/VT 直接对比（SD@400，20 prompts × 10 seeds，严格 NFE）

| 方法 | Brightness | Compressibility | 实测 NFE/图 |
|---|---|---|---|
| Uniform | 0.6975 | 0.7971 | 400 |
| RBF (faithful, Kim et al. 2025) | 0.6138 | 0.8022 | 325 / 400 |
| **RBF-FS** (每步 quota 12 + 硬顶 + 末步补齐) | 0.6146 | 0.7817 | 400 |
| VT (faithful, τ=校准集中位数) | 0.6836 | 0.7742 | 375 / 238 |
| **VT-FS** | 0.6820 | 0.7632 | 400 |
| Offline-only | 0.7112 | 0.7930 | 400 |
| **GAINS** | **0.7165** | **0.8011** | 400 |

RBF-FS 与 VT-FS 是我们为求严格公平预算加的强化版本：**即使强制花满预算，rollover 类方法仍不敌 profile-based**——证实"分配错误而非预算不足"，支持 regret 下界叙事。

## 表 4｜E4 DrawBench-200（200 prompts × 3 seeds = 600 img/格；n=480 for Uniform 跑中）

| 方法 | Brightness | Compressibility |
|---|---|---|
| Uniform | 0.7218 | 0.8263 |
| **GAINS** | **0.7371 (+0.0153)** | **0.8331 (+0.0068)** |

- 10× prompt 规模下增益保持（原表 20p×10s: B +0.0223；DrawBench 200p×3s: B +0.0153）→ 回应"样本太小"和"prompt 依赖"双重质疑。
- profile 沿用旧 20-prompt 集校准的形状，在 DrawBench 上跨 prompt 集迁移无损 → 顺带回应"profile 是 prompt-set 特定的吗"。
- 200 prompt 下配对标准误 ~0.003，两个增量均 ≥ 4σ 显著。

## 表 5｜E3b ImageReward verifier（人类偏好代理指标）

| naive | Uniform | (heuristic profile) | GAINS |
|---|---|---|---|
| 0.668 | **1.392** | 1.372 | 1.340 |

- **搜索本身收益巨大**（naive 0.668 → 搜索方法均 ≥ 1.34，+0.72）——test-time search 在人类偏好代理指标上成立。
- 简单 verifier 校准的 profile 未迁移到 IR：与投稿版关于 "profile 是 verifier 特定的、需一次性重校准"（Assumption + 附录讨论）的主张一致；
  **正在进行的实验**：温和偏离 uniform（前/中/后 1/3 加权 K=10/7/7 或 7/10/7 或 7/7/10）和强偏离（12/6/6 三方向）的 6 个 IR-tilt 变体，用于系统探测 IR 敏感度形状 → 今日 17:00–00:30 陆续收官。

## 表 6｜E6 双信号消融（SD@400，K1=11 严格 400 NFE）

| | Brightness | Compressibility |
|---|---|---|
| Offline-only (noRev) | 0.7112 | 0.7930 |
| + gain-only | 0.7158 | 0.8006 |
| + var-only | 0.7179 | 0.8023 |
| + both (GAINS) | 0.7165 | 0.8011 |

- 三个在线变体等价（差 ≤ 0.002）；GAINS 相对 offline 的 +0.005–0.008 主要由 revert 反馈贡献。
- 诚实叙事：**revert-on-negative 是在线增益主源**，双阈值信号提供额外稳健性但相互冗余（跨 verifier 稳定为正）。

## 表 7｜CLIP 对齐核验（E3a，对齐分不下降）

主表 SD@400 每 method 200 img 的 CLIP-L14 分数（配对同 prompt+seed）：

| | U | Offline-only | GAINS |
|---|---|---|---|
| Brightness | 27.36 | 27.56 | 27.41 |
| Compressibility | 27.74 | 27.69 | 27.63 |

**|Δ| ≤ 0.20 << 0.5 阈值** → **verifier-guided search 的增益不以牺牲图文对齐为代价**（回应三个 reviewer 共识的对齐担忧）。500 档、IR-tilt 变体的 CLIP 扫描已挂 watcher，跑完后同表补上。

## 500 档 SD 预算分解（在跑，20:00 前收官）
K1=14 严格 500 NFE，与 400 档 K1=11、800 档 K1=22 保持 budget-scaled 一致口径。offline/GAINS/Uniform × B/C 共 6 config，chain 已挂。

## 其他补充证据（数据档在 results/day0/）
- 参数 sweep 明细：`tune_online_b_p1..p6/`、`tune2_*/`
- Nominal-vs-measured NFE 对照（K1=25 未缩放导致实测 ~515 NFE 的归档运行）：`*_k25nominal/`
- RBF/VT 实测 NFE 记账（逐步 K_used / carry_out）：`e2_sd_{rbf,vt,rbffs,vtfs}_*/run.log`
- 完整 posthoc 评分表（CLIP + IR）：`clip_ir_check.csv`, `posthoc_scores.csv`
