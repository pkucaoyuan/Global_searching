# NeurIPS 23990 Rebuttal 补实验计划

评分:8PHj 3分(conf 4) / GLgQ 2分(conf 3) / x7rM 3分(conf 2)。
三人共识:① baseline 只有 uniform 太窄;② Brightness/Compressibility 撑不起质量主张。
理论无人质疑正确性,全部批评是实验补足类。

现有代码资产(`code_repos/diffusion-tts/`):
- 方法已实现:`naive / rejection / beam / mcts / zero_order / eps_greedy / epsilon_1(GAINS-offline) / epsilon_online(GAINS-full)`
- SD scorers 已有 **CLIP**;EDM 有 ImageNet classifier;flux 只有 brightness/compressibility
- online controller 的 gain/variance 阈值是独立超参(`thresh_gain_coef` / `thresh_var_coef`),消融只需置零
- 本机 GPU:RTX 5070 Ti Laptop 12GB(SD 512px / EDM 64px 可跑但慢;原实验用 A100 80GB)

---

## P0 —— 三人共识,决定 rebuttal 成败(先跑)

### E1. 所有主表加 Offline-Only 行(8PHj 核心要求,x7rM 附议)
- 内容:Table 1 / 2 / 7 每个 NFE budget 跑三行:Uniform vs **GAINS-Offline(`epsilon_1`)** vs Full GAINS(`epsilon_online`)
- 现状:offline-only 只在附录消融出现过,主表没有
- 实现:代码现成,纯跑量
- 交付:更新的 Table 1/2/7 + 一段"online controller 何时有意义、offline 何时已足够"的讨论
- 工作量:小(纯计算)

### E2. RBF 与 Verifier Threshold 直接对比(8PHj 最大的点)
- 内容:实现 RBF(Rollover Budget Forcing, Kim et al. 2025)和 Verifier Threshold,同 NFE budget 下与 GAINS 对比
- 我们附录已把二者分类为 "online rollover / online threshold",却没比——这是 8PHj 认为原创性"在实证层面不清晰"的直接原因
- 实现:两者机制都不复杂(rollover = 分数不达标就把预算滚到下一步;VT = 分数过阈值就提前停),可在现有 `epsilon_online` 框架里改出
- 交付:SD + (可选)flow 上的对比表
- 工作量:中(实现 2 个方法 + 跑量)

### E3. 独立质量指标:优化 verifier 的同时报告未被优化的指标(三人全要)
- 内容:在现有 Brightness/Compressibility 优化实验中,对生成结果**额外报告** CLIP score、ImageReward(或 HPSv2)、Aesthetic score——证明"优化 verifier 没有牺牲整体质量/对齐"
- 升级版(强烈推荐):直接用 **ImageReward 作为 verifier** 跑一组 SD 主实验——正面回应"换成真实质量目标,20-50% NFE 节省还成立吗"(x7rM 的核心质疑)
- 实现:CLIP scorer 已有;ImageReward/HPS 是 pip 包,接进 Scorer 基类约几十行
- 交付:主表加列 + 一组 ImageReward-as-verifier 的新表
- 工作量:中(接指标小,跑量大)

### E4. 扩大 prompt 集(GLgQ:20 prompts × 10 reps 太小)
- 内容:主要 SD 实验扩到 100–200 prompts(DrawBench 200 条或 PartiPrompts 子集),至少覆盖 Uniform / Offline / Full 三行
- 交付:更新的主表 + 均值±标准误,顺带回应统计显著性
- 工作量:中(纯跑量,与 E1/E3 同一批次合并跑)

## P1 —— 单个 reviewer 的硬要求,能做尽做

### E5. 手调非均匀 schedule baselines(x7rM)
- 内容:early-window / middle-window / late-window / oracle-profiled(held-out 集上校准)四种固定 schedule,同 budget 对比
- 实现:只是不同的 K_t 向量,代码几乎不用改
- 顺带的好处:oracle-profiled ≈ offline-only 的上界,可与 E1 呼应
- 工作量:小

### E6. Online controller 双信号消融(GLgQ)
- 内容:gain-only / variance-only / both 三个变体
- 实现:`thresh_gain_coef=0` 或 `thresh_var_coef=0` 即可
- 工作量:小

### E7. Offline profile 迁移性 / 稳定性(x7rM)
- 内容:profile 在 prompt 集 A 校准 → 在集 B 测试;跨 verifier(brightness profile 用于 compressibility);(可选)跨模型
- 回应:"profile 是不是 verifier/数据集特定的" + 顺带回应 8PHj 对 "sensitivity 与 prompt 无关" 假设的质疑(Assumption 与实践的连接)
- 工作量:中

### E8. 定性 image grids(x7rM 硬要求,8PHj 也提)
- 内容:同 prompt 同 NFE budget 下 Uniform vs GAINS 的图片网格;挑几个 GAINS 改变全局结构的例子
- 交付:rebuttal 里放匿名链接(OpenReview 不能贴图时用 anonymous.4open.science / imgur)
- 工作量:小(从 E1/E3 的运行中直接留图)

### E9. Diversity 检查(x7rM Q6)
- 内容:同 prompt 多 seed,报告 pairwise LPIPS(或 CLIP 特征方差),Uniform vs GAINS
- 回应:"优化固定 verifier 是否损害多样性"
- 工作量:小

### E10. Particle / tree-based 对比(GLgQ)
- 内容:同 budget 下 beam / MCTS / rejection(Best-of-N)与 GAINS 对比
- 实现:**方法已全部实现**,纯跑量;注意这些是"轨迹搜索"方法,与 GAINS(调度)正交,表格旁注明 GAINS 可叠加其上
- 工作量:小

## P2 —— 写作类回应(不需要 GPU,与实验并行)

- W1. 理论 ↔ 算法对应表(x7rM):哪些组件有定理支撑(offline water-filling ← Thm 5/6;offline+online 混合 ← regret lower bound),哪些是 heuristic(窗口早停、variance 阈值)——一张表说清
- W2. 理论假设与实践的连接(8PHj):光滑性、小扰动、prompt 无关 sensitivity 各自何时近似成立、失效时算法为何仍稳(online 兜底)+ 引用 E7 的实证
- W3. Section 4.3 加 intuition 段(GLgQ 的 clarity 2 分主因)
- W4. 附录引用加 hyperref 链接(8PHj minor)
- W5. Limitations 扩写:明确 verifier-dependent 结论边界

---

## 执行顺序建议

1. **第一批(立即)**:E1 + E4 + E5 + E6 + E10 合并成一个大跑量批次(全是现成代码,同一批 prompts 上跑全部方法,一次生成、多指标评分),同时留图给 E8
2. **第二批(并行开发)**:E2(实现 RBF/VT)+ E3(接 ImageReward)——开发完插入同一评测框架
3. **第三批**:E7 迁移性 + E9 diversity(用第一批的产出物可算一部分)
4. **全程并行**:W1–W5 写作

## 待确认

- Rebuttal DDL 具体日期?(决定砍哪些)
- 除本机 12GB 外有无 A100/集群可用?(决定 E4 扩到 100 还是 200 prompts、flow 模型要不要覆盖)
- RBF/VT 两篇原文是否已在手?(E2 需要按原文核对机制细节)
