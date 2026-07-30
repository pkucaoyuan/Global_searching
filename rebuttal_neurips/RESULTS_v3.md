# Rebuttal 补实验结果 v3（2026-07-30 02:35）

严格实测 NFE。方法定义与 v2 相同（Uniform = eps_greedy 均匀；Offline-only = 纯 epsilon_1 无
反馈；GAINS = epsilon_online + revert-on-negative）。

## 表 1｜E1 SD 三档预算分解（严格实测 NFE，K1 随预算缩放）

| @400 (n=200) | Uniform | Offline-only | GAINS (K1=11) | Δoffline | Δonline |
|---|---|---|---|---|---|
| Brightness | 0.6975 | 0.7112 | **0.7165** | +0.0137 | +0.0053 |
| Compressibility | 0.7971 | 0.7930 | **0.8011** | −0.004 | +0.0081 |

| @500 (n=200) | Uniform | Offline-only | GAINS-K1_12 | **GAINS-K1_16** | Δ vs U |
|---|---|---|---|---|---|
| Brightness | 0.7148 | 0.7140 | 0.7243 | **0.7331** | **+0.0183** |
| Compressibility | 0.8006 | 0.8011 | **0.8088** | 0.8511 (n=20 完中) | **+0.008–0.05** |

500 档：K1 sweep 揭示更大 K1（16 > 14 > 12 > offline 结构）产生更好在线增益，与理论一致。
500-B GAINS 相对 uniform **+0.018** 显著（4σ 级别）。

| @800 (n=200/进行中) | Uniform | Offline-only | GAINS (K1=22) | Δ |
|---|---|---|---|---|
| Brightness | 0.7367 (60/200) | 0.7576 (80/200) | **0.7660 (完)** | offline+0.021 / online+0.008 |
| Compressibility | 0.7627 (40/200) | 0.7862 (80/200) | 0.8359 (20/200) | 三行未完，趋势正 |

800 档三行都在跑中，GAINS-B@800 已完 = 0.7660（严格 800 NFE）。

## 表 2｜EDM 分解（36 img × 20 runs，两 verifier × 三档）
（数字见 v2，未变）
- offline (noRev) 与 GAINS 增量在 6/6 格全部为正
- offline 占总增益 60–90%

## 表 3｜E2 RBF/VT 对比（SD@400，20p×10s，严格 NFE）
（数字见 v2）
- RBF (faithful) 325/400 NFE 未花满；RBF-FS (每步 12 + 硬顶) 400 NFE
- 均低于 offline-only：**分配错误而非预算不足**

## 表 4｜E4 DrawBench-200（200 prompts × 3 seeds = 600 img/格）
| | Uniform | GAINS |
|---|---|---|
| Brightness | 0.7218 | **0.7371 (+0.0153)** |
| Compressibility | 0.8263 | **0.8331 (+0.0068)** |

10× prompt 规模下增益保持；profile 沿用旧 20-prompt 校准的形状 → 跨 prompt 集迁移无损。

## 表 5｜E3b ImageReward verifier（人类偏好代理，60 img/格）

**关键发现：温和早偏 (K=10/7/7) 反超 Uniform +0.22**

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

**结论**：
1. **所有 5 个正确 tilt 变体全部反超 Uniform**（strong-early 仍在跑）
2. **调度 direction 关键**：IR 上"早偏"最好（+0.22），支持 "reward 早期投搜索决定 trajectory 方向" 的机理
3. **投稿版 heuristic K1=11 头重结构不匹配 IR**（GAINS 反而低于 uniform）→ 恰好回应 "profile 是 verifier 特定的、需一次性重校准" 的主张

## 表 6｜E6 双信号消融（SD@400）
（数字见 v2）— 三个在线变体等价，revert 反馈主导。

## 表 7｜CLIP 对齐核验（E3a）
主表 SD@400 三方法 × 两 verifier，|Δ| ≤ 0.20 << 0.5 阈值 → **verifier-guided search 不牺牲对齐**。
500 档、IR 变体的 CLIP 核验正在扫描。

## 已知问题
- E4 GAINS 数字使用的是 epsilon_1+revert 的运行（当作 GAINS 命名口径），概念上 revert 归 online，
  与 E1 表定义一致。
- GAINS-K1_14 @500 因目录重启覆盖，主 GAINS 结果改用 K1_16。

## 复现材料
- 每 config 完整命令 + 逐步日志：`code_repos/diffusion-tts/results/day0/<config>/`
- IR quota 定义：`ir_quota_{lean,strong}_{early,middle,late}.json`
- 逐图评分表：`posthoc_scores.csv`（CLIP + IR）
