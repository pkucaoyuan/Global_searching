# Academic Writing Reference Library

Human-authored academic writing references for paper polishing.

## Statistics

| Category | Entries | Files |
|----------|---------|-------|
| **Phrases** | 14 | 2 |
| **Sentences** | 287 | 29 |
| **Paragraphs** | 40 | 5 |
| **Total** | **341** | **36** |

## Structure

```
writing_references/
├── phrases/                    # 短语级别 (2-5 words)
│   ├── transitions.md          # 过渡连接词 (7 entries)
│   └── hedging.md              # 学术谨慎表达 (7 entries)
│
├── sentences/                  # 句子级别 (完整句型模板)
│   ├── introduction.md         # 引言句型 (6)
│   ├── contribution.md         # 贡献声明 (6)
│   ├── motivation.md           # 动机阐述 (8) - Howard
│   ├── problem_setup.md        # 问题设定 (9) - Kaufmann
│   ├── related_work.md         # 相关工作 (8) - Kaufmann
│   ├── citation_patterns.md    # 引用模式 (10) - Howard
│   ├── or_applications.md      # OR应用表达 (9) - BwK
│   ├── causal_inference.md     # 因果推断 (12) - Wager
│   ├── control_theory.md       # 控制理论 (8) - Agarwal
│   ├── algorithm_optimality.md # 算法最优性 (12) - Garivier
│   ├── online_learning.md      # 在线学习 (9) - Cohen
│   ├── dynamic_pricing.md      # 动态定价 (12) - Keskin, Javanmard
│   ├── revenue_management.md   # 收益管理 (12) - Gallego
│   ├── robust_optimization.md  # 鲁棒优化 (12) - Ben-Tal
│   ├── online_matching.md      # 在线匹配 (10) - Karp
│   ├── queueing_theory.md      # 排队论 (10) - Halfin-Whitt
│   ├── inventory_management.md # 库存管理 (11) - Newsvendor
│   ├── stochastic_programming.md # 随机规划 (10) - Birge
│   ├── optimal_stopping.md     # 最优停止 (10) - Prophet
│   ├── approximation_algorithms.md # 近似算法 (10) - PTAS/FPTAS
│   ├── mechanism_design.md     # 机制设计 (10) - Myerson
│   ├── markov_decision_process.md  # MDP (10) - Puterman
│   ├── network_optimization.md # 网络优化 (10) - Ford-Fulkerson
│   ├── game_theory.md          # 博弈论 (10) - Nash
│   ├── supply_chain.md         # 供应链 (10) - Bullwhip
│   ├── learning_theory.md      # 学习理论 (10) - VC/PAC
│   ├── llm_papers.md           # LLM论文 (13) - Transformer, BERT, GPT, RLHF
│   ├── price_of_robustness.md  # 鲁棒性代价 (10) - Bertsimas-Sim
│   └── choice_models.md        # 选择模型 (10) - Talluri-van Ryzin
│
├── paragraphs/                 # 段落级别 (段落结构模板)
│   ├── paper_roadmap.md        # 论文路线图 (5) - Kaufmann
│   ├── properties_list.md      # 属性列举 (4) - Howard
│   ├── main_results.md         # 主要结果 (10) - BwK
│   ├── tutorial_style.md       # 教程风格 (10) - Orabona
│   └── proof_structure.md      # 证明结构 (11) - Garivier
│
├── guides/                     # 写作原则指南
│   └── academic_writing_principles.md
│
├── rag/                        # RAG检索系统
│   ├── __init__.py             # 模块接口
│   ├── entry_parser.py         # Markdown条目解析器
│   ├── writing_store.py        # 嵌入存储与检索
│   └── retrieval.py            # 高级检索接口
│
└── _sources/                   # 论文源码 (LaTeX)
    ├── paper_list.md           # 目标论文清单
    ├── kaufmann2016/           # BAI complexity
    ├── howard2021/             # Confidence sequences
    ├── bwk2013/                # Bandits with Knapsacks
    ├── orabona2019/            # Online learning tutorial
    ├── garivier2016/           # Optimal BAI
    ├── agarwal2019/            # Online control
    └── cohen2018/              # Online LQ control
```

## Source Papers

| Paper | Authors | Venue | Topic |
|-------|---------|-------|-------|
| On the Complexity of BAI | Kaufmann et al. | JMLR 2016 | Bandits |
| Time-uniform CS | Howard et al. | AoS 2021 | Statistics |
| Bandits with Knapsacks | Badanidiyuru et al. | JACM 2018 | OR/ML |
| Online Learning Tutorial | Orabona | arXiv 2019 | ML Tutorial |
| Causal Forests | Wager & Athey | JASA 2018 | Econometrics |
| Online Control | Agarwal et al. | NeurIPS 2019 | Control |
| Optimal BAI | Garivier & Kaufmann | ALT 2016 | Bandits |
| Online LQ Control | Cohen et al. | NeurIPS 2018 | Control |
| Dynamic Pricing | Keskin & Zeevi, Javanmard | MS/arXiv | Pricing |
| Revenue Management | Gallego & Van Ryzin | MS 1994/97 | RM |
| Robust Optimization | Ben-Tal & Nemirovski | MOR 1998 | Robust |
| Online Matching | Karp, Vazirani, Vazirani | STOC 1990 | Matching |
| Queueing Theory | Halfin & Whitt | OR 1981 | Queueing |
| Inventory Management | Arrow et al., Newsvendor | MS/OR | Inventory |
| Stochastic Programming | Birge & Louveaux | Springer | SP |
| Optimal Stopping | Prophet/Secretary | Various | Stopping |
| Approximation Algorithms | Vazirani, Williamson | Various | Approx |
| Mechanism Design | Myerson | Econometrica 1981 | Auctions |
| MDP | Puterman | Wiley 1994 | DP |
| Network Flow | Ford & Fulkerson | 1956 | Graphs |
| Game Theory | Nash | PNAS 1950 | Strategy |
| Supply Chain | Lee et al. | MS 1997 | SCM |
| Learning Theory | Vapnik | Various | ML Theory |
| Attention Is All You Need | Vaswani et al. | NeurIPS 2017 | Transformer |
| BERT | Devlin et al. | NAACL 2019 | Pretraining |
| Language Models are Few-Shot Learners | Brown et al. | NeurIPS 2020 | GPT-3 |
| Training LMs to Follow Instructions | Ouyang et al. | NeurIPS 2022 | RLHF |
| Chain-of-Thought Prompting | Wei et al. | NeurIPS 2022 | Reasoning |
| The Price of Robustness | Bertsimas & Sim | OR 2004 | Robust Opt |
| Choice-Based Revenue Management | Talluri & van Ryzin | MS 2004 | Choice Models |

## Usage

### Adding References

Each reference entry should follow this format:

```markdown
## [Category Tag] Entry Title

**Source**: Paper/Guide name, Author, Year
**Context**: When to use this expression

> The actual phrase/sentence/paragraph

**Pattern** (optional):
- Generalized template with placeholders

**Tags**: #transition #formal #methodology
```

### Searching

```bash
# By tag
grep -r "#hedging" writing_references/

# By keyword
grep -ri "demonstrate" writing_references/sentences/

# By source
grep -r "Source:.*Kaufmann" writing_references/

# By topic
grep -r "#bai" writing_references/
grep -r "#or" writing_references/
```

### Programmatic Access (RAG)

```python
from rag import WritingReferenceRetriever, search_references

# Initialize retriever
retriever = WritingReferenceRetriever()

# Get references for a section
entries = retriever.get_references_for_section("introduction", k=5)

# Search by query
results = retriever.search("how to state contribution", k=3)

# Get references by topic
llm_refs = retriever.get_by_topic("llm")
bai_refs = retriever.get_by_topic("bai")

# Format for LLM prompt
prompt_text = retriever.format_for_prompt(entries)
```

### Verification & Embedding Generation

```bash
# Navigate to writing_references directory
cd .claude/writing_references

# 1. Verify system (keyword search only, no API needed)
python3 rag/verify.py

# 2. Build embeddings (requires Azure OpenAI)
python3 rag/build_embeddings.py

# 3. Force rebuild embeddings
python3 rag/build_embeddings.py --force --verify
```

**Output files:**
- `rag/embeddings_cache.json` - Cached embeddings for fast retrieval

### Integration with /polish-paper

The `/polish-paper` skill will automatically:
1. Read relevant reference files based on the section being polished
2. Use tag-based filtering for context-appropriate suggestions
3. Suggest improvements based on matched references

## Quality Standards

- **Human-authored only**: No AI-generated content
- **Verified sources**: Must cite paper/guide name and venue
- **Context-aware**: Include usage context
- **Domain-specific**: Focus on OR/ML/Statistics/Econometrics

## Tags Reference

| Tag | Description |
|-----|-------------|
| `#introduction` | Opening/motivation patterns |
| `#contribution` | Contribution statements |
| `#methodology` | Method descriptions |
| `#bai` | Best-arm identification |
| `#or` | Operations research |
| `#ml` | Machine learning |
| `#proof` | Proof techniques |
| `#tutorial` | Pedagogical style |
| `#hedging` | Cautious language |
| `#transition` | Connectives |
