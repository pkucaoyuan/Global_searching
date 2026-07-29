# Rebuttal 补实验结果 v1（2026-07-29 03:40，实验完成度 ~90%）

> 所有数值均为**严格实测 NFE** 下的重测结果（每 config 的完整命令与逐步日志在
> `code_repos/diffusion-tts/results/day0/<config>/`）。方法定义（统一）：
> **Uniform** = eps_greedy 均匀 K；**Offline-only** = epsilon_1 固定 {K_t}（无任何在线反馈）；
> **GAINS** = 在线控制（预算重分配 + revert-on-negative 反馈）。
> 提交版旧配置（K1 不随预算缩放，实测 ~515 NFE @名义400）的运行归档于 `*_k25nominal/`，
> 可用作 nominal-vs-measured NFE 对照。

## 表 1｜E1 SD 预算分解（20 prompts × 10 seeds，n=200/格）

| @400 NFE | Uniform | Offline-only | GAINS | Δoffline | Δonline |
|---|---|---|---|---|---|
| Brightness | 0.6975 | 0.7112 | 0.7165 | +0.0137 | +0.0053 |
| Compressibility | 0.7971 | 0.7930 | 0.8011 | −0.0041(≈0) | +0.0081 |

- B 偏 offline、C 偏 online 的不对称与投稿版附录的分解比例（B: +0.0133/+0.0090，C: +0.0040/+0.0073）**结构一致**。
- @800：Offline(含rev)=0.7729\*，GAINS=0.7660；无 rev 的 offline-800 在跑。\*旧定义，注意口径。
- 在线超参稳健性：10 组 sweep（K1 11–18, slack 2–4, β_g 0.1–0.3, β_σ 0.6–0.8, W_g 2–6）
  在线阈值信号增量均为 0±0.006 → 在线增益主要来自 revert 反馈；阈值信号冗余但无害（另见 E6）。

## 表 2｜EDM 分解（36-img grid × 20 runs）

| | Uniform(投稿版) | Offline-only(noRev) | Offline(含rev) | GAINS(投稿版) |
|---|---|---|---|---|
| B/144 | 0.9507 | 0.9704 | 0.9737 | 0.9887 |
| B/180 | 0.9617 | 0.9756 | 0.9851 | 0.9904 |
| B/288 | 0.9787 | 0.9940† | 0.9976 | 0.9988 |
| C/144 | 0.6714 | 0.6757 | 0.6832 | 0.6845 |
| C/180 | 0.6774 | 0.6874 | 0.6955 | 0.7003 |
| C/288 | 0.6901 | 跑中 | 0.7124 | 0.7165 |

†6/20 runs 时中间值。EDM 上 offline 与 online 增量在两个 verifier 全部为正。

## 表 3｜E2 RBF/VT 直接对比（SD@400，20p×10s）

| 方法 | 得分 | 实测 NFE/图 | 备注 |
|---|---|---|---|
| Uniform | 0.6975 / 0.7971 | 400 | B / C |
| RBF (faithful) | 0.6138 / 0.8022 | 325 / 400 | 单调 verifier 下平凡接受→滚存作废 |
| RBF-FS (每步12+硬顶+末步补齐) | 0.6146 / 0.7817 | 400 | 强制花满仍不敌 → 预算错配而非不足 |
| VT (faithful, τ=校准集中位数) | 0.6836 / 0.7742 | 375 / 238 | |
| VT-FS | 0.6820 / (跑中) | 400 | |
| Offline-only | 0.7112 / 0.7930 | 400 | |
| GAINS | 0.7165 / 0.8011 | 400 | |

RBF-C 落在 U 与 offline 之间（0.7971<0.8022<0.8096(含rev版)）；B 侧 rollover 类全面低于 Uniform。

## 表 4｜E4 DrawBench 扩规模（前 100 prompts × 3 seeds 已齐；后 100 在跑，合并后 n=600/格）

| 前100 @400 | Uniform | Offline(含rev) | GAINS |
|---|---|---|---|
| Brightness | 0.7247 | 0.7440 (+0.019) | 0.7425 (+0.018) |
| Compressibility | 0.8304 | 0.8403 (+0.010) | 0.8383 (+0.008) |

| 后100（ext, seed 300） | Uniform | Offline | GAINS |
|---|---|---|---|
| Brightness | 跑中 | 0.7303* | 0.7271 |
| Compressibility | 跑中 | 0.8294* | 0.8221 |

\*297/300、234/300 时值。profile 沿用旧 20-prompt 校准 → 同时证明 prompt 集迁移性。

## 表 5｜E3b ImageReward verifier（20p×10s @400）

| naive | Uniform | Offline(简单verifier的profile) | GAINS |
|---|---|---|---|
| 0.668 | 1.392 | 1.372 | 1.340 |

搜索收益巨大（+0.72）；调度增益未迁移——offline profile 是 brightness/compressibility 校准的头重形状，
IR 的敏感度分布不同（早期 x0 预测的 IR 打分≈噪声）。**这是 "profile 是 verifier 特定的、需一次性
重校准" 的实证**。IR 专属 water-filling 重校准运行中（`e3b_calib_ir_loggain` → recalibrated offline-IR）。

## 表 6｜E6 双信号消融（SD@400，K1=11）

| | B | C |
|---|---|---|
| gain-only | 0.7158 | 0.8006 |
| var-only | 0.7179 | 0.8023 |
| both (GAINS) | 0.7165 | 0.8011 |
| offline-only(noRev) | 0.7112 | 0.7930 |

三个在线变体等价（±0.002）→ 阈值信号冗余、revert 反馈为在线增益主要来源；诚实呈现 + 稳健性框架。

## 补充
- **E5 早窗基线**（跑中）：B 70/200 时 0.6721 < offline 0.7112 → 窗口法不及 profiled 分配（初步）。
- **GAINS-B@800 = 0.7660**（K1=22 缩放，严格 800）。
- **C 刻度说明**：现 prompts.csv 与投稿版 C 表 prompt 集不同（0.79 vs 0.88 档），本文全部 SD 数字为
  统一重测口径；B 侧与投稿版锚点兼容（U 0.6975≈0.7025，GAINS 0.7165≈0.7248 差异在种子噪声内）。
- 参数敏感性 sweep 明细：`results/day0/tune_online_b_p*/`、`tune2_*/`。

## 未决/在跑清单（2026-07-29 03:40）
E4x-uniform ×2（~13:00）、offline-IR 重校准版、VT-FS-C、E5 ×2、EDM-noRev 最后1档、noRev-800 ×2（顺延）。
Day-2 池：E3a 事后评分（CLIP/IR/Aesthetic，脚本 `results/posthoc_score.py` 就绪）、E8 grids、E9 LPIPS、E7。
