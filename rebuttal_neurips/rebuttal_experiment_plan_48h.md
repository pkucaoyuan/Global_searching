# NeurIPS 23990 Rebuttal 补实验完整计划（4×A100，48 小时）

> 代码库:`code_repos/diffusion-tts/`(conda env: `diffusion-tts`)
> 提交版 PDF:`rebuttal_neurips/23990_submission_version.pdf`
> 原则:**所有运行保存生成图片 + 每步日志**(image grid、事后指标、diversity 全靠它们,不重跑)

---

## 0. 现有数值锚点(论文提交版)

**SD 主表(Table 1),20 prompts × 10 seeds:**

| NFE | Uniform-B | GAINS-B | Uniform-C | GAINS-C |
|-----|-----------|---------|-----------|---------|
| 400 | 0.7025±.047 | 0.7248±.056 | 0.8833±.017 | 0.8946±.016 |
| 500 | 0.7094±.063 | 0.7346±.060 | 0.8825±.017 | 0.9004±.015 |
| 800 | 0.7511±.062 | 0.7832±.046 | 0.8892±.020 | 0.9008±.012 |

**EDM 主表(Table 2):**

| NFE | Uniform-B | GAINS-B | Uniform-C | GAINS-C |
|-----|-----------|---------|-----------|---------|
| 144 | 0.9507±.015 | 0.9887±.005 | 0.6714±.017 | 0.6845±.007 |
| 180 | 0.9617±.016 | 0.9904±.004 | 0.6774±.012 | 0.7003±.009 |
| 288 | 0.9787±.018 | 0.9988±.001 | 0.6901±.012 | 0.7165±.009 |

**已知分解(SD/400,附录)**:Offline-only 比 Uniform +0.0133(B)/+0.0040(C);online 再 +0.0090/+0.0073。
即 Brightness 上 offline 占总增益 ~60%,Compressibility 上 ~35%。

**Operator 表(SD/400)**:Zero-order U 0.4920→G 0.5080;Random U 0.7382→G 0.7457。

**超参**:EDM(T=18): ε=0.4, λ=0.15, N=4, β_g=0.3, β_σ=0.7, δ=2。SD(T=50): N=4, β_g=0.1, β_σ=0.8, W_g=4, revert-on-negative。

---

## 1. 实验总表与 GPU 排程

### 批次结构(关键路径:E2/E3 的开发在 Day 1 上午完成)

```
Day 0 晚(启动后 2h 内):
  [所有 GPU] 环境校验 + 吞吐量标定(每 backend 跑 1 config 测单图耗时)
  GPU0: Batch A1 = E1 SD Brightness (offline-only × {400,500,800})
  GPU1: Batch A2 = E1 SD Compressibility (同上)
  GPU2: Batch A3 = E1 EDM 全部 6 config(快)→ 完成后接 E10(beam/mcts/rejection)
  GPU3: Batch A4 = E5 窗口 schedule + E6 信号消融(SD/400 × 2 verifiers)

Day 1 上午(人工:实现 RBF、VT、ImageReward scorer,预计 3-4h 开发):
  GPU0: E4 DrawBench-200 × {Uniform, Offline, GAINS} × Brightness @400
  GPU1: E4 同上 × Compressibility @400
  GPU2: (A3 完成后) E7 迁移性 profiling + 交叉评测
  GPU3: 开发调试机 → 调通后跑 E3b ImageReward-as-verifier @{400,500}

Day 1 晚:
  GPU0+1: E2 RBF/VT 对比(SD × 2 verifiers × {400,500,800})
  GPU2: E3b ImageReward @800 + 补漏
  GPU3: E9 多 seed diversity 专用运行(如 E4 留档不够)

Day 2 上午:
  全部 GPU 空出 → 事后评分(CLIP/ImageReward/Aesthetic 扫全部存档图片,快)
  + LPIPS diversity 计算 + image grid 生成
Day 2 下午:
  数据汇总、表格生成、rebuttal 文字撰写、匿名图链接上传
```

### 预算核算(以标定为准,以下是先验估计)

- SD 512px @400 NFE ≈ 1–2 min/图(A100);200 图/config ≈ 3–6 GPU·h
- EDM 64px ≈ 秒级/图,全部 EDM 实验 < 4 GPU·h
- 4×A100×48h = 192 GPU·h,以下全计划约 100–130 GPU·h,有 ~40% 余量
- **若标定超预期**:先砍 E4 到 100 prompts,再砍 E2/E3b 的 800 NFE 档

---

## 2. 各实验详细规格 + 数值预期

> 数值预期 = 内部 sanity check 基准,用于判断"结果是否符合理论叙事"和"哪里出了 bug"。
> 偏离预期 ≠ 失败,每个实验附"若偏离怎么解读"。

### E1. Offline-Only 全表补行 【回应 8PHj 核心要求 + x7rM】

**命令模板**(offline-only = `epsilon_1`,禁用 online):
```bash
python main.py --backend sd --scorer {brightness,compressibility} --method epsilon_1 \
    --total_budget {400,500,800} --prompt_csv prompts.csv --n_runs 10
python main.py --backend edm --scorer {brightness,compressibility} --method epsilon_1 \
    --num_steps 18 --total_budget {144,180,288} --n_runs 20
```

**数值预期**(按已知 offline/online 分解比例外推):

| 设置 | Uniform(已知) | **Offline-only(预期)** | GAINS(已知) |
|------|------|------|------|
| SD-B 400 | 0.7025 | **0.7158**(已实测,校验点) | 0.7248 |
| SD-B 500 | 0.7094 | **0.724 ± 0.010** | 0.7346 |
| SD-B 800 | 0.7511 | **0.770 ± 0.010** | 0.7832 |
| SD-C 400 | 0.8833 | **0.8873**(已实测,校验点) | 0.8946 |
| SD-C 500 | 0.8825 | **0.889 ± 0.005** | 0.9004 |
| SD-C 800 | 0.8892 | **0.893 ± 0.005** | 0.9008 |
| EDM-B 144 | 0.9507 | **0.977 ± 0.008** | 0.9887 |
| EDM-B 180 | 0.9617 | **0.982 ± 0.007** | 0.9904 |
| EDM-B 288 | 0.9787 | **0.993 ± 0.004** | 0.9988 |
| EDM-C 144 | 0.6714 | **0.680 ± 0.006** | 0.6845 |
| EDM-C 180 | 0.6774 | **0.693 ± 0.006** | 0.7003 |
| EDM-C 288 | 0.6901 | **0.709 ± 0.006** | 0.7165 |

(EDM 假设 offline 占比 ~70%,因 σ_t 更分散、平均 profile 更可靠)

**叙事**:offline 捕获大部分增益(证明 water-filling 理论的实用性),online 增量在所有档位稳定为正(Jensen gap,Rmk. online-value)。**若偏离**:若某档 offline-only ≈ GAINS(online 增量≈0),如实报告并解释为该档位 per-prompt 方差小——这正是理论预测 online 价值 = Jensen gap 的场景,不丢人。

### E2. RBF + Verifier Threshold 直接对比 【回应 8PHj 最大的点】

**开发**(Day 1 上午,~2h):在 `epsilon_online` 框架内实现两个变体:
- **RBF**(Kim et al. 2025, rollover budget forcing):每步给基础预算 K_base,若最优候选分数未超过当前 incumbent 则把剩余预算滚到下一步;无 offline profile,无 variance 信号。**实现前先按原文伪代码核对**(`literature/` 有存档)。
- **VT**(Verifier Threshold):每步搜索直到候选分数超过阈值 τ 即停,省下的预算顺延;τ 按原文设定(如校准集分位数)。

**运行**:SD × {brightness, compressibility} × {400,500,800},20 prompts × 10 seeds,与 GAINS 严格同 NFE 记账。

**数值预期**(二者是 online-only 调度,理论上受 regret lower bound 约束,应落在 Uniform 与 GAINS 之间):

| SD/400 | Uniform | VT(预期) | RBF(预期) | GAINS |
|--------|---------|----------|-----------|-------|
| Brightness | 0.7025 | **0.705–0.715** | **0.710–0.720** | 0.7248 |
| Compressibility | 0.8833 | **0.884–0.890** | **0.885–0.892** | 0.8946 |

预期 GAINS 领先最强 online-only 基线 **+0.005–0.015**,且随 NFE 增大领先扩大(offline profile 的复利)。

**叙事**:这是 regret lower bound(Thm. 7)的实证版——纯 online 方法缺 offline 先验,定位慢。**若偏离**:若 RBF 在某档反超,立即补一组 **GAINS+rollover 杂交**(offline profile + rollover 代替早停)——二者正交可组合,叙事从"我们更强"转为"我们的 offline profiling 与它们的 online 规则互补,组合最强",同样成立且更稳。

### E3. 独立质量指标 【回应三人共识,最值钱】

**E3a — 事后监测(零额外生成成本)**:Day 2 用保存图片统一评分:
- CLIP score(`sd/scorers.py` 已有)
- ImageReward(`pip install image-reward`,RM 模型 ~2GB)
- Aesthetic score(LAION aesthetic predictor,linear head on CLIP)

对每个 config 报告:优化 Brightness/Compressibility 时,这三个指标 GAINS vs Uniform 的差。

**数值预期**:
- **GAINS − Uniform 的 CLIP 差:|Δ| < 0.5**(同一 verifier 目标下二者应无系统差异)——这是核心卖点:*GAINS 的 verifier 增益不以牺牲对齐为代价*
- 相对 naive(无搜索),搜索类方法 CLIP 可能整体 −0.5~−1.5(verifier 与对齐轻微 trade-off)——如实报告,归因于 verifier 选择而非调度器

**E3b — ImageReward 作为 verifier(新主实验)**:
```bash
python main.py --backend sd --scorer imagereward --method {eps_greedy,epsilon_1,epsilon_online} \
    --total_budget {400,500,800} --prompt_csv prompts.csv --n_runs 10
```
需要把 ImageReward 接进 Scorer 基类(~40 行,照 CLIPScorer 抄)。

**数值预期**(ImageReward 量纲 ≈ [−2.5, 2],SD1.5 naive 基线 ~0.2):

| NFE | Naive | Uniform(预期) | Offline(预期) | GAINS(预期) |
|-----|-------|--------------|---------------|--------------|
| 400 | ~0.2 | **0.55–0.70** | +0.02–0.05 | **Uniform +0.04–0.10** |
| 800 | ~0.2 | **0.70–0.85** | +0.03–0.06 | **Uniform +0.05–0.12** |

**叙事**:在人类偏好代理指标上,非均匀调度的增益结构与简单 verifier 一致 → 方法不是在 exploit 简单指标。同时验证 "20–50% NFE 节省" 在真实 verifier 上是否复现(预期:GAINS/400 ≥ Uniform/500 → **20% 节省保守成立**;50% 档不强求)。**若偏离**:若增益 < +0.03,检查 ImageReward 的 per-step 噪声(对中间去噪结果打分噪声大),可加 x0-prediction decode 再打分;若仍小,如实报告"增益存在但幅度小于简单 verifier",并把 NFE 节省叙事收窄到 20%。

### E4. DrawBench-200 扩大评测 【回应 GLgQ "20 prompts 太小"】

**运行**:DrawBench 200 prompts × 3 seeds × {Uniform, Offline-only, GAINS} × {B, C} @ 400 NFE(SD)。
offline profile **沿用原 20-prompt 校准的 {K_t} 不重新校准**——顺带证明 profile 迁移性(喂给 E7)。

**数值预期**:绝对分数会变(prompt 分布不同),但**增益结构保持**:
- GAINS − Uniform:Brightness **+0.015–0.030**,Compressibility **+0.008–0.018**
- 均值标准误(n=600)比原来(n=200)缩小 ~40%,显著性 p<0.01(paired t-test per prompt)

**叙事**:10 倍 prompt 规模下增益不变,且 profile 是在旧 prompt 集上校准的 → 同时回应"样本太小"和"profile 是否 prompt-set 特定"。**若偏离**:若增益缩水一半以上,说明 20-prompt 校准集有偏,补一次 DrawBench 上重新校准的 offline profile(增 ~4 GPU·h),报告两版。

### E5. 手调非均匀 schedule 基线 【回应 x7rM】

**运行**:SD/400 × {B, C},四种固定 K_t:
- Early-window:预算集中前 1/3 步;Middle-window:中 1/3;Late-window:后 1/3
- Oracle-profiled:在 held-out 10 prompts 上校准的 profile(≈ offline-only 的"作弊版")

**数值预期**(SD 的敏感度 profile 偏 early,论文已述 "early-spread"):

| SD/400-B | Uniform | Late | Middle | Early | Oracle | Offline | GAINS |
|----------|---------|------|--------|-------|--------|---------|-------|
| 预期 | 0.7025 | **0.69–0.70** | **0.70–0.71** | **0.710–0.718** | **≈0.716** | 0.7158 | 0.7248 |

**叙事**:窗口方向猜对(early)能拿到部分增益,但 profiled 分配 > 最优窗口,full GAINS > 一切固定 schedule → 增益来自 *profile 的形状* + *online 适应*,不是碰运气选对了区间。**若偏离**:若 early-window ≥ offline-only,说明 SD 的 profile 近似阶跃——如实报告并强调 EDM/flow 上 profile 形状不同(补一组 EDM early-window 约 1 GPU·h 佐证,窗口法不跨模型泛化而 profiling 自动适应)。

### E6. Online 双信号消融 【回应 GLgQ】

**运行**:SD/400 × {B, C} × 3 变体:gain-only(`--thresh_var_coef 0`)、var-only(`--thresh_gain_coef 0`)、both(默认)。

**数值预期**(online 总增量:B +0.0090,C +0.0073):

| 变体 | Brightness 预期 | Compressibility 预期 |
|------|-----------------|----------------------|
| offline-only | 0.7158 | 0.8873 |
| + gain-only | **0.720–0.723**(~60–70% 增量) | **0.891–0.893** |
| + var-only | **0.718–0.721**(~40–50% 增量) | **0.890–0.892** |
| + both (GAINS) | 0.7248 | 0.8946 |

**叙事**:两信号互补——gain 检测"已经不涨了"(停),variance 检测"还有的挖"(继续),合起来 > 任一单独。**若偏离**:若单信号 ≈ both,承认冗余、说明保留双信号是为跨 setting 稳健(引 EDM 上的对应消融,若时间够补跑)。

### E7. Offline profile 迁移性 【回应 x7rM + 8PHj 假设质疑】

**运行**(用 `eps1_gain_probe.py`):
1. 20-prompt 集 → 随机对半 A/B,各自跑 profile,报告 per-step gain 曲线的 Spearman 相关
2. 原 20-prompt profile vs DrawBench-50 子集 profile 的相关(E4 已顺带证明端到端迁移)
3. 跨 verifier:Brightness profile 与 Compressibility profile 的相关
4. 交叉评测:用 B-profile 跑 C-优化(错配 profile 的端到端代价)

**数值预期**:
- 同 verifier 跨 prompt 集:Spearman **ρ ≥ 0.8**(prompt-averaged sensitivity 稳定 → 支持 Assumption "prompt 无关 σ_t" 在均值意义下成立)
- 跨 verifier:ρ **0.4–0.7**(中等——profile 是 verifier 特定的,但校准是一次性成本 ~2 GPU·h)
- 错配 profile 端到端:比正确 profile 掉 **30–60% 的 offline 增益**,但仍 ≥ Uniform(early 侧质量主导的共性)

**叙事**:profile 跨 prompt 分布稳定(假设经验成立),跨 verifier 需重校准但成本一次性且低。**若偏离**:若跨 prompt ρ < 0.6,这直接触碰理论假设——如实报告,强调 online controller 正是为此兜底(引 E1 中 online 增量数据),把 Assumption 讨论写进 W2。

### E8. 定性 image grids 【回应 x7rM 硬要求】

Day 2 从存档图片出:
- 4–6 个 prompt × {Uniform, GAINS} @ 同 NFE 同 seed 的并排 grid(B 和 C 各一套)
- 1–2 个"GAINS 改变全局结构"的 case(翻 verifier 分差最大的样本)
- 上传 anonymous.4open.science,rebuttal 贴链接

### E9. Diversity 检查 【回应 x7rM Q6】

E4 每 prompt 3 seeds + 原 20-prompt 10 seeds 存档 → 计算同 prompt 跨 seed pairwise LPIPS:

| | Naive | Uniform | GAINS |
|---|-------|---------|-------|
| LPIPS 预期 | 基准 x | **x − (5–12%)** | **Uniform ± 3%** |

**叙事**:verifier 搜索本身有收敛代价(Uniform 同样有),GAINS 相对 Uniform **不额外损失多样性**——调度改变的是"在哪搜",不是"搜多狠"。**若偏离**:若 GAINS 明显更低,检查是否 early 步重分配导致模式锁定,报告并讨论 λ(扰动半径)的调节作用。

### E10. Particle/tree 方法对比 【回应 GLgQ】

**运行**(方法已实现,纯跑量):SD/400 × {B, C} × {rejection(=per-step BoN), beam, mcts},严格同 NFE 记账。

**数值预期**:

| SD/400-B | rejection | beam | mcts | Uniform ε-greedy | GAINS |
|----------|-----------|------|------|------------------|-------|
| 预期 | **0.69–0.71** | **0.70–0.72** | **0.68–0.71** | 0.7025 | 0.7248 |

(固定预算下 tree 方法把 NFE 花在分支/回溯上,单点质量通常不优;MCTS 开销最大)

**叙事**:同预算下 GAINS ≥ 轨迹搜索方法;且**正交可叠加**——beam/MCTS 是"搜什么"(local operator/轨迹结构),GAINS 是"在哪个 t 花多少"(global schedule),原则上可组合(引 MDP 表)。**若偏离**:若 beam 反超,补 GAINS-scheduled beam(把 beam 宽度按 K_t 调)约 3 GPU·h,叙事转向可组合性,依然是加分项。

---

## 3. 交付物 → Reviewer 映射

| Reviewer | 回应的实验 | 写作件 |
|----------|-----------|--------|
| 8PHj (3, conf 4) | E1(全表 offline 行)、E2(RBF/VT)、E3a、E8 | W2 假设讨论、W4 超链接 |
| GLgQ (2, conf 3) | E4(200 prompts)、E6(双信号消融)、E10(tree 基线)、E3 | W3(4.3 节 intuition)、W1 |
| x7rM (3, conf 2) | E3b(ImageReward verifier)、E5(窗口基线)、E7(迁移)、E8(grids)、E9(diversity) | W1(理论↔heuristic 对应表)、W5 |
| AC | 以上全部 + 一段"理论贡献 + 实证补强"的总回应 | |

写作件(与实验并行,不占 GPU):
- **W1** 理论↔算法对应表:offline water-filling ← Thm 5/6;混合设计必要性 ← regret LB;窗口早停/variance 阈值 = heuristic(标注清楚)
- **W2** 假设与实践:光滑性/小扰动/prompt-无关 σ_t 何时近似成立,失效时 online 兜底(引 E7 数据)
- **W3** Section 4.3 每定理后加 2–3 句 intuition
- **W4** 附录引用 hyperref
- **W5** Limitations 扩写(verifier-dependent 结论边界)

## 4. 风险与降级路线

1. **吞吐标定超预期 2 倍** → E4 降到 100 prompts;E2/E3b 砍 800 NFE 档;E10 砍 mcts
2. **RBF/VT 复现细节不明** → 按论文伪代码实现 + rebuttal 中明确写实现假设(reviewer 通常接受 faithful reimplementation)
3. **ImageReward 中间步打分噪声大** → 用 x0-prediction decode 后打分;不行则退回 E3a(事后监测)+ 承诺 camera-ready 补全
4. **某实验结果与叙事相反** → 一律如实报告 + 解释机制;三个 reviewer 都属于"要证据"型而非"要漂亮数字"型,诚实 + 机制解释比完美数字更加分
5. **时间只够一半** → 必保:E1、E2、E3、E4;其余按 E8(便宜)> E6 > E5 > E10 > E9 > E7 排

## 5. 执行 checklist

- [ ] Day 0: conda env 校验、模型权重预下载(SD1.5 / EDM ckpt / CLIP / ImageReward / LPIPS)
- [ ] Day 0: 吞吐标定(每 backend 1 config × 5 图),据此确认/裁剪排程
- [ ] Day 0: 启动 Batch A(GPU0–3),确认日志和图片存档路径统一:`results/{exp_id}/{method}_{scorer}_{budget}/`
- [ ] Day 1 AM: 实现 RBF、VT、ImageRewardScorer;单图 smoke test 后入队
- [ ] Day 1: E4 / E2 / E3b / E7 依次入队
- [ ] Day 2 AM: 事后评分脚本扫全部存档;LPIPS;grids;上传匿名链接
- [ ] Day 2 PM: 汇总表格 vs 本文预期逐项核对(偏离处按各实验"若偏离"预案定叙事)→ 撰写 rebuttal
