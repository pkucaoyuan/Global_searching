# NeurIPS 23990 Rebuttal 补实验计划 v2（8×A800-80GB，保范围压时程）

> 由 `rebuttal_experiment_plan_48h.md`（4×A100 版）改排。**实验范围、规格、数值预期、叙事、降级预案全部不变**，
> 只重排 GPU 排程：4→8 卡，GPU 侧关键路径从 ~2 天压到 ~1 天，Day 2 整天让给分析与写作。
> 各实验（E1–E10）详细规格、数值预期、"若偏离"预案 **一律以原计划 §2 为准**，本文不重复。

## 0. 与原计划的差异摘要

| 项 | 原计划（4×A100） | 本计划（8×A800） |
|----|-----------------|------------------|
| GPU 预算 | 192 GPU·h（用 100–130） | 384 GPU·h（余量 ~3×，砍档预案大概率不触发） |
| E4 / E10 / E5 / E6 | Day 1 才排上 | **Day 0 即入队**（不依赖新开发，方法已实现） |
| E2 / E3b | Day 1 晚开跑 | Day 1 上午开发完立即 8 卡分摊，Day 1 晚前跑完 |
| 事后评分 | Day 2 上午 | Day 1 晚滚动开始（跑完一个 config 扫一个） |
| Day 2 | 半天分析半天写作 | **全天分析 + 写作 + 偏离补跑**（原"若偏离"补充组有卡可跑） |

**硬件差异**：A800 = A100 同架构、NVLink 带宽阉割版。全部实验为单卡推理，不跨卡通信，**吞吐无实质差异**；
标定仍照做（原则不变：一切以实测吞吐为准）。

**前置条件（阻塞）**：8 卡当前被 `precipitation_nowcasting/train_multiseed.py` 占满（每卡 ~63GB、util ~90%）。
下方排程以"卡已释放"为 T0。若只能部分释放，按 GPU0→7 顺序把批次前移合并，关键路径不变（见 §3 降级）。

## 1. 批次排程（T0 = 卡释放时刻）

```
T0 ~ T0+2h —— 全卡环境校验 + 吞吐标定
  每 backend 跑 1 config × 5 图，实测单图耗时 → 确认/修正下方时长估计
  （conda env、模型权重应在 T0 前备好，见 §2 环境清单）

Day 0（T0+2h 起，8 卡满排。除 E2/E3b 外全部 GPU 实验入队）：
  GPU0: E1  SD  Brightness       offline-only × {400,500,800}          (~9–18h)
  GPU1: E1  SD  Compressibility  offline-only × {400,500,800}          (~9–18h)
  GPU2: E1  EDM 全 6 config（快，<4h）→ 接 E10 rejection/beam/mcts × {B,C} @SD/400
  GPU3: E5  窗口 schedule 4 变体 × {B,C} @SD/400（8 config）
  GPU4: E6  双信号消融 gain-only/var-only × {B,C} @SD/400（4 config；both=已有主表数据）
  GPU5: E4  DrawBench-200 Brightness × {Uniform, Offline, GAINS} @400  (~10–20h)
  GPU6: E4  DrawBench-200 Compressibility × 同上                       (~10–20h)
  GPU7: E7  profile 迁移性（eps1_gain_probe.py：对半 A/B、DrawBench-50 子集、
        跨 verifier、B-profile 跑 C 错配组）→ 完成后接 E4 溢出 / E9 补 seed

Day 1 上午（开发窗口，3–4h，与 GPU 上仍在跑的 Day 0 长任务并行）：
  实现 RBF、VT（epsilon_online 框架内变体，先核对 literature/ 原文伪代码）
  实现 ImageRewardScorer（照 CLIPScorer 抄，~40 行）+ 单图 smoke test

Day 1 下午–晚（Day 0 批次此时基本收尾，8 卡重排）：
  GPU0: E2 RBF × Brightness      × {400,500,800}
  GPU1: E2 RBF × Compressibility × {400,500,800}
  GPU2: E2 VT  × Brightness      × {400,500,800}
  GPU3: E2 VT  × Compressibility × {400,500,800}
  GPU4: E3b ImageReward-verifier × {eps_greedy, epsilon_1} × {400,500}
  GPU5: E3b ImageReward-verifier × epsilon_online × {400,500} + naive 基线
  GPU6: E3b 全方法 @800 档
  GPU7: E9 diversity 补 seed（若 E4/主表存档不足）→ 空闲则滚动跑 E3a 事后评分

Day 1 晚（滚动，CPU/单卡轻量）：
  E3a 事后评分（CLIP/ImageReward/Aesthetic）逐 config 扫存档；LPIPS diversity 同步算

Day 2 全天（GPU 基本空闲 = 偏离预案的机动池）：
  上午: 评分收尾、E8 image grids、数据汇总表 vs 原计划 §2 预期逐项核对
  机动: 触发"若偏离"时补跑——GAINS+rollover 杂交(E2)、EDM early-window(E5)、
        DrawBench 重校准 profile(E4)、GAINS-scheduled beam(E10)——8 卡下均 <半天
  下午: 表格生成、rebuttal 文字（W1–W5）、匿名图链接上传
```

### 时长估计依据（先验，以标定为准）

- SD 512px @400 NFE ≈ 1–2 min/图；20 prompts × 10 seeds = 200 图/config ≈ 3–6 GPU·h；
  E1 单卡 3 config（400/500/800 加权 ~×1.4）≈ 9–18h → Day 0 最长批次，Day 1 上午收尾
- E4 单卡 600 图（200 prompts × 3 seeds）× 3 方法 @400 ≈ 10–20h → 同为长批次
- EDM 64px 秒级/图，EDM 全部 < 4 GPU·h
- 总量 ~100–130 GPU·h（与原计划同）对 384 GPU·h 预算，余量 ~3×

## 2. 环境与权重清单（T0 前完成，不占排程）

| 项 | 来源 | 备注 |
|----|------|------|
| conda env `diffusion-tts` | `environment.yml`（py3.10, torch≥2.1, cu121） | |
| SD 1.5 | HF `runwayml/stable-diffusion-v1-5`（代码默认 ID） | 官方已下架，经 hf-mirror 或 `sd-legacy/stable-diffusion-v1-5` 兜底后本地软链 |
| CLIP | HF `openai/clip-vit-large-patch14` | scorers.py 默认 |
| EDM ckpt | `nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl` | ~2GB，直连 CDN |
| ImageNet classifier | `openaipublic.blob.core.windows.net/.../64x64_classifier.pt` | edm/scorers.py 用 |
| ImageReward | `pip install image-reward` + HF `THUDM/ImageReward` (~2GB) | E3 |
| Aesthetic head | LAION aesthetic predictor（CLIP linear head，几 MB） | E3a |
| LPIPS | `pip install lpips`（AlexNet 权重 ~250MB，torch hub） | E9 |
| DrawBench-200 | prompts 表（HF datasets 或论文附录 CSV） | E4 |

网络：`HF_ENDPOINT=https://hf-mirror.com`（已配好，实测可达）；GitHub/直连不稳时走 `http.version=HTTP/1.1` + 重试。

## 3. 风险与降级（在原计划 §4 之上新增/修订）

1. **卡未按时释放 / 只释放一部分**：优先级顺序 = 原计划§4-5 必保序
   （E1 → E2 → E3 → E4 → E8 > E6 > E5 > E10 > E9 > E7），有几张卡就从队首往下排；
   4 卡即退化为原计划排程，仍成立。
2. **吞吐标定超预期**：8 卡余量 ~3×，原"砍 E4/砍 800 档"预案大概率不触发；
   超 3 倍才启用原§4-1。
3. 其余风险（RBF/VT 复现细节、ImageReward 中间步噪声、结果与叙事相反）与原计划 §4 完全一致。

## 4. 执行 checklist

- [ ] T0 前: conda env 创建、全部权重预下载（§2 清单逐项验证可加载）
- [ ] T0 前: 确认 nowcasting 训练释放 8 卡（或确定可用卡数，按 §3-1 调整）
- [ ] T0: 吞吐标定（每 backend 1 config × 5 图）→ 修正排程时长
- [ ] T0+2h: Day 0 八卡批次全部入队；日志与图片存档统一 `results/{exp_id}/{method}_{scorer}_{budget}/`
- [ ] Day 1 AM: RBF / VT / ImageRewardScorer 开发 + smoke test → Day 1 PM 八卡重排入队
- [ ] Day 1 晚: E3a 事后评分滚动开跑；LPIPS 同步
- [ ] Day 2 AM: grids、汇总表 vs 预期核对（偏离项按原计划 §2 各实验预案定叙事 + 机动池补跑）
- [ ] Day 2 PM: rebuttal 撰写（W1–W5）、匿名链接上传
