# Advisor Feedback — Prof. Zeyu Zheng (2026-03-20)

**Source**: Meeting recording `GMT20260320-055332_Recording.transcript.vtt`
**Context**: Professor reviewed the GAINS paper introduction (original version) as a simulated reviewer.

---

## Feedback Items (Original Chinese + Translation + Principle)

### F1: Don't use undefined terms the reader doesn't know
> "说到那个inference time, 他其实并不知道inference time是什么啊"

The reader may not know what "inference time" means in the ML sense. An OR reviewer thinks "inference" = statistical estimation. Define the concept before using the term.

**Status**: FIXED — ¶2 now explains "generating each sample requires T steps, each costing one forward pass" before introducing the term "inference-time scaling."

---

### F2: Don't make strong claims without evidence
> "Can substantially improve the quality? 这种是属于一个很强的论述...凭什么你这么说？你这么说的根源是什么呀？...等于说你去做了一个论断，但没给证据"

"Substantially improve" is a strong assertion. The reader asks: by what measure? By how much? Where's the evidence? Don't make claims without backing them up in the same sentence or paragraph.

**Status**: FIXED — Removed "substantially." The abstract now states concrete numbers (20-50% budget reduction).

---

### F3: If something is already achievable, what's your contribution?
> "如果是能substantially，那这个事儿如果是已知的，那你这个工作又做了什么呢？人家既然已经substantially improve了，你是不是得更加substantially"

If you say prior work can already substantially improve quality, then what's new about your work? You undermine your own contribution. Either say what's NOT achievable yet (the gap), or position your work as solving a different problem.

**Status**: FIXED — ¶2 now says "recent work shows that spending more than T NFE can improve quality" (neutral statement) then asks the allocation question as the gap.

---

### F4: Undefined jargon: "noise perturbations"
> "noise perturbations什么是noise perturbations? 他不知道什么是noise perturbations"

The phrase "noise perturbations" is ML jargon. An OR reviewer doesn't know what this means in the diffusion context. Use concrete language: "replacing a noise draw with a better candidate."

**Status**: FIXED — Changed to "replacing a noise draw with a better candidate improves quality far more at some steps than at others."

---

### F5: Optimization framing requires objective and structure
> "two level optimization problem, 大家一说two level optimization problem, 那往往是有自己的想法的, 分别目标函数是什么? 每一层在优化什么? 要说清楚"

When you frame something as an "optimization problem," readers expect: what are the decision variables? What's the objective? What are the constraints? For a "two-level" problem, what does each level optimize? Be explicit.

**Status**: FIXED — ¶5 now states: "given B total NFE across T sequential denoising steps, find the allocation that maximizes output quality." Two levels: local = how to search, global = how to distribute budget.

---

### F6: Comparatives need a metric
> "search for better noise, 什么叫better noise, better是怎么定义的? 说这种相对比较的时候一定得有个metric"

"Better" compared to what? By what measure? Every comparative adjective needs a metric. "Better noise" should be "the noise candidate that scores highest on the verifier."

**Status**: FIXED — Now says "keeping the one that scores highest on a quality metric."

---

### F7: Structural consistency — don't introduce A+B then only use A
> "前面你已经提了有local有global, 怎么就这个方法里就只有global了, 是你不要local了还是怎么样?"

If you introduce "local" and "global" as two components, and then your method name (GAINS = Global Adaptive...) only mentions global, the reader asks: what happened to local? Make sure both components appear in the method description.

**Status**: FIXED — The GAINS description now says "Offline, GAINS profiles verifier sensitivity... Online, GAINS adjusts the search at each step..." Both levels are present in the description.

---

### F8: Define new terms before using them
> "noise scheduling这个事儿, 如果这个词要用到前面, 得定义一下, 什么叫noise scheduling?"

If you introduce a new compound term (e.g., "noise scheduling"), you must define what it means at first use. Don't assume the reader can infer the meaning from the components.

**Status**: FIXED — "noise trajectory search" is now introduced with explicit definition in ¶5.

---

### F9: Don't burden readers with unexplained jargon stacks
> "combines offline profiling, online feedback control, 就是这些词...它很容易就停下来了, 这个是什么意思那个是什么意思...你是在欺负那些真正对你文章感兴趣的读者"

When you write "combines offline profiling with online feedback control," every term is a stop-point. The interested reader stops at "offline profiling" (what does that mean?), then "online feedback control" (what feedback?). This punishes the readers who care most. Either explain each term inline or use plain language.

**Status**: FIXED — Offline and online components are now explained in separate sentences with concrete descriptions.

---

### F10: Serve both audiences
> "对于那些不是很想读你细细读你的文章的人, 那你写那些本质上也没有什么价值"

For the casual reader, dense jargon adds no value (they skip it). For the careful reader, it adds burden (they stop and ask "what does this mean?"). Neither audience is served. Write so that both audiences benefit.

**Status**: FIXED — Mechanism explained intuitively for careful readers; concrete results (20-50% savings) for casual readers.

---

### F11: Writing clarity reflects thinking clarity
> "写东西写清楚这个能力反映了脑子的清晰程度...做的好的人的共性都是表述非常清楚, 能清晰的知道自己说的话对于不同的听众他能不能跟得上"

Clear writing reflects clear thinking. Successful people across all fields share one trait: they can explain their ideas in a way that different audiences can follow. This is a fundamental skill, not just academic writing polish.

**Status**: N/A — This is a meta-principle, not a specific fix.

---

## Summary Checklist for Verification

| # | Check | Principle |
|---|-------|-----------|
| F1 | No undefined terms the reviewer doesn't know | CL-01 |
| F2 | No strong claims without evidence in same paragraph | CL-02 |
| F3 | Don't undermine your own contribution | CL-03 |
| F4 | No ML jargon where plain language works | CL-01/CL-08 |
| F5 | Optimization framing states objective + constraints + variables | CL-05 |
| F6 | Every comparative has a metric | CL-04 |
| F7 | Both sides of a dichotomy must appear in the discussion | CL-06 |
| F8 | New terms defined at first use | CL-07 |
| F9 | No jargon stacks that force readers to stop | CL-08 |
| F10 | Writing serves both careful and casual readers | CL-09 |
| F11 | Writing clarity = thinking clarity | Meta |
