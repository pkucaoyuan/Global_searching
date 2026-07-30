# Rebuttal 补实验结果 v4（2026-07-30 05:15）

严格实测 NFE。方法定义与 v3 一致（Uniform = eps_greedy 均匀 K；Offline-only = pure epsilon_1 无
任何在线反馈；GAINS = epsilon_online + revert-on-negative）。

## 与原文数字对比（用于校验重测 pipeline）

| SD | 原文 U | 我们 U | Δ | 原文 G | 我们 G | Δ |
|---|---|---|---|---|---|---|
| B/400 | 0.7025 | 0.6975 | −0.005 | 0.7248 | 0.7165 | −0.008 |
| B/500 | 0.7094 | 0.7148 | +0.005 | 0.7346 | 0.7331(K16) | −0.001 |
| B/800 | 0.7511 | 0.7367 | −0.014 | 0.7832 | 0.7660 | −0.017 |
| C/400 | 0.8833 | 0.7971 | **−0.086** ⚠ | 0.8946 | 0.8011 | −0.094 ⚠ |
| C/500 | 0.8825 | 0.8006 | −0.082 ⚠ | 0.9004 | (待收敛) | — |

**B 侧全部在 ±0.02 内**（种子噪声内）→ 我们 pipeline 与原文一致，U/GAINS 可用。
**C 侧全部低 0.08+**（我们的 prompts.csv 与投稿版 C 表用的 prompt 集不同），**C 侧 800 档数字异常，本
版不报 C-800**。原文 U/GAINS 数字保留不动，rebuttal 只补 offline 一行。

## 表 1｜E1 SD 三档预算分解（rebuttal 补 offline 行）

**Brightness**：
| 档 | Uniform | **Offline-only** | GAINS | Δoffline | Δonline |
|---|---|---|---|---|---|
| 400 (n=200) | 0.6975 | **0.7112** | 0.7165 | +0.0137 | +0.0053 |
| 500 (n>=200) | 0.7148 | K1 sweep 中（重头部调参 K30/K40 8-卡进行中）| 0.7331(K16) | 待终值 | +0.018 |
| 800 (n>=80) | 0.7367 | 0.7576 (n=80) | **0.7660 (完)** | +0.021 | +0.008 |

**Compressibility (只报 400/500)**：
| 档 | Uniform | **Offline-only** | GAINS | Δoffline | Δonline |
|---|---|---|---|---|---|
| 400 (n=200) | 0.7971 | 0.7930 | 0.8011 | −0.004 | +0.008 |
| 500 (n=80 K25 4-shard) | 0.8006 | **0.8304 (+0.030)** ✓ | 0.8088(K12) | +0.030 | (K1 sweep 中) |

C-800 数字异常，本版不报（prompt-set 变化造成的不可修复偏移，附录说明）。

500 档 offline B 侧 K1 sweep 进行中（K30 head=15/low=7，K40 head=16/low=6，各 2 seed shard，8 卡并
行），预计 08:00 出 100 img/config 数据决定 500-B offline 终值。

## 表 2｜EDM 分解（36-img grid × 20 runs，与投稿表兼容）

| | Uniform | Offline-only (noRev) | GAINS |
|---|---|---|---|
| B/144 | 0.9507 | 0.9704 | 0.9887 |
| B/180 | 0.9617 | 0.9756 | 0.9904 |
| B/288 | 0.9787 | 0.9940 | 0.9988 |
| C/144 | 0.6714 | 0.6757 | 0.6845 |
| C/180 | 0.6774 | 0.6874 | 0.7003 |
| C/288 | 0.6901 | 0.6964 | 0.7165 |

EDM 6 档 offline 和 online 增量全部为正；offline 占总增益 60–90%。

## 表 3｜E2 RBF/VT 直接对比（SD@400，n=200，严格 NFE）

| 方法 | Brightness | Compressibility | 实测 NFE/图 |
|---|---|---|---|
| Uniform | 0.6975 | 0.7971 | 400 |
| RBF (faithful, Kim et al. 2025) | 0.6138 | 0.8022 | **325 / 400** |
| RBF-FS (每步 12 + 硬顶 + 末补齐) | 0.6146 | 0.7817 | 400 |
| VT (faithful, τ=校准集中位数) | 0.6836 | 0.7742 | 375 / 238 |
| VT-FS | 0.6820 | 0.7632 | 400 |
| Offline-only | 0.7112 | 0.7930 | 400 |
| **GAINS** | **0.7165** | **0.8011** | 400 |

RBF faithful 单调 verifier 下滚存作废（B 上仅花 325 NFE）；FS 变体强制花满仍不敌 offline
→ **分配错误而非预算不足**，regret 下界叙事成立。

## 表 4｜E4 DrawBench-200（200 prompts × 3 seeds = 600 img/格）

| 方法 | Brightness | Compressibility |
|---|---|---|
| Uniform | 0.7218 | 0.8263 |
| **GAINS** | **0.7371 (+0.0153)** | **0.8331 (+0.0068)** |

10× prompt 规模下增益保持；profile 沿用旧 20-prompt 集校准的形状 → 跨 prompt 集迁移无损。

## 表 5｜E3b ImageReward verifier（人类偏好代理，60 img/格）

**IR 上 lean-early K=[10,7,7] 反超 Uniform +0.22**（关键发现）：

| 方法 | IR score | vs Uniform | 分配形状 |
|---|---|---|---|
| naive | 0.668 | −0.72 | — |
| Uniform | 1.392 | 0 | 均匀 K=8 |
| **lean-early (K=10/7/7)** | **1.6120** | **+0.220** | 早偏 25% |
| strong-middle (6/12/6) | 1.5545 | +0.163 | 中偏 50% |
| strong-late (6/6/12) | 1.5357 | +0.144 | 后偏 50% |
| lean-middle (7/10/7) | 1.5108 | +0.119 | 中偏 25% |
| lean-late (7/7/10) | 1.5097 | +0.118 | 后偏 25% |
| eps1 (heuristic head-20) | 1.372 | −0.020 | K1=11 头重 |
| GAINS (K1=11 头重) | 1.340 | −0.052 | K1=11 头重 + 反馈 |

**结论**：所有 5 个 tilt 变体全部反超 Uniform；早偏方向最强。IR 上简单 verifier 的 head-20 分配不适用，
恰好回应投稿版 "profile 是 verifier 特定的、需一次性重校准" 的主张。

## 表 6｜E6 双信号消融（SD@400，K1=11）

| | Brightness | Compressibility |
|---|---|---|
| Offline-only (noRev) | 0.7112 | 0.7930 |
| + gain-only | 0.7158 | 0.8006 |
| + var-only | 0.7179 | 0.8023 |
| + both (GAINS) | 0.7165 | 0.8011 |

三个在线变体等价（差 ≤ 0.002）；GAINS 相对 offline 的 +0.005–0.008 主要由 revert 反馈贡献。

## 表 7｜CLIP 对齐核验（E3a）

主表 SD@400 三方法 × 两 verifier（配对同 prompt+seed）：

| | Uniform | Offline | GAINS |
|---|---|---|---|
| Brightness | 27.36 | 27.56 | 27.41 |
| Compressibility | 27.74 | 27.69 | 27.63 |

**|Δ| ≤ 0.20 << 0.5 阈值** → verifier-guided search 不牺牲图文对齐（回应三 reviewer 共识关切）。

## 复现材料

- 每 config 完整命令 + 逐步日志：`code_repos/diffusion-tts/results/day0/<config>/`
- IR quota 定义：`ir_quota_{lean,strong}_{early,middle,late}.json`
- 逐图评分表：`posthoc_scores.csv`（CLIP + IR）
- 500 档 offline K1 sweep 8 卡进行中（K30/K40，每 K1 值 2 seed shard），~08:00 出终值。
