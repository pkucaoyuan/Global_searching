# 会议记录 - 论文修改讨论

**日期**: 2026-03-08
**参会者**: 导师 (说话人01), 学生 (说话人02)

---

## 核心反馈要点

### 1. Flow Model / ODE 兼容性问题 [最重要]

**导师原话**:
> "你要考虑到他读者里面比方说有70%的人，他可能印象里就是SOTA是flow model就是ODE sample，noise只是最开始有个noise，就xT是个大T是个noise，后面每一步其实都是deterministic的"

> "那这样的话，他乍一看你的这个setting啊，就会觉得好像你不是搜他"

**解决方向**:
- 加noise是为了exploration，如果不加noise本质上没得探索
- 参考其他inference-time scaling的paper怎么开头
- 有些公司做RL时也主动加noise

---

### 2. Verifier来源说明

**导师原话**:
> "可能得说一下这个verify是谁给的...verify是take as given的，我们并没有单独去训练verify"

> "你就是说，唉，这个verif是给定的...举个例子是吧。for example...那同时你可以cite一个paper两个paper说，比如说那这两个paper也是use similar这个verify"

**目的**: 避免读者质疑verifier的合理性

---

### 3. 技术细节公式化

**导师原话**:
> "verifier的input是什么？...因为因为它是xt先相当于先proxy x0，然后再加噪，所以说是verify part是对那个proxy"

> "就是根据目前写的这个文案啊，他是能让他比如去直接写code，他是能清晰的知道塞给verify的是什么东西嘛？"

**要求**:
- 说明喂给verifier的是predicted x0
- xt → predicted x0 → xt-1 关系要写清楚
- 写出具体数学公式

---

### 4. 写作风格

**导师原话**:
> "把那个这个半字线，这个叫半字线，就是或者是叫hyphen...这个文章里面你标一下以后都拿掉。因为还有些人觉得是GPT喜欢写这个"

---

### 5. Offline/Online算法细节

**导师原话**:
> "offline这样选出来之后...high value...low...high的就多分"

> "gain也小variance也小...只要有一个稍大点儿就往后走了"

> "variance是衡量我这一步实际上到底重不重要，然后gain是我这步离最优点是不是很近了"

---

### 6. 实验补充 - Flow Model

**导师原话**:
> "实验里面有那个flow base model吗？...没有...我建议就是最后也加一个flow base model"

> "不需要搞个很大的...小的那个flow的也行"

> "这样能呼应到你前面那个...加noise一方面是有的模型本身就是这种带这个noise的，有些模型它其实是这个ODE这种flow加ODE的，那这个我们也可以也适用"

---

### 7. 格式和排版

**导师原话**:
> "你找一个那个archive格式，就是那个单栏的那种conference，就是不带，不需要带那个会议名字"

> "有些时候他那个表啊，比如说这个两列其实也挺好，但他就显得好像东西不多...组合一下，比如说你拿offline和online放在左边，这个NFE多弄一些放右边"

> "视觉效果很重要...影响大家的那个第一印象"

---

## 行动项总结

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | Introduction添加flow model兼容性说明 | TODO |
| P0 | 添加Verifier来源说明+引用 | TODO |
| P0 | 技术细节公式化 | TODO |
| P1 | 添加flow-based model实验 | TODO |
| P2 | 格式改为单栏 | TODO |
| P2 | 去除hyphen | TODO |
| P2 | 表格排版优化 | TODO |
