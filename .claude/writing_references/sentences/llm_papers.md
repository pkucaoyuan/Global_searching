# Large Language Model Sentence Templates

Sentence patterns from foundational LLM papers for architecture, pretraining, alignment, and reasoning.

---

## [Architecture Limitation] The dominant [models/methods] are based on [architecture type] that include [component A] and [component B]. The best performing models also [enhancement].

**Source**: Vaswani et al. - "Attention Is All You Need" (NeurIPS 2017)
**Context**: Describing limitations of existing approaches before proposing new architecture

> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism.

**Tags**: #architecture #limitation #motivation #llm

---

## [Novel Architecture] We propose a new [simple/novel] [architecture/method], the [Name], based solely on [mechanism], dispensing with [old components] entirely.

**Source**: Vaswani et al. - "Attention Is All You Need" (NeurIPS 2017)
**Context**: Introducing new architecture with key innovation

> We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

**Tags**: #architecture #contribution #innovation #llm

---

## [SOTA Results] Experiments on [task] show these models to be superior in [quality metric] while being more [efficiency property] and requiring significantly less [resource].

**Source**: Vaswani et al. - "Attention Is All You Need" (NeurIPS 2017)
**Context**: Reporting experimental results with multiple benefits

> Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

**Tags**: #results #comparison #efficiency #llm

---

## [Pretraining Contribution] We introduce a new [representation/model] called [Name], which stands for [full name]. Unlike [previous approaches], [Name] is designed to [key innovation] by [mechanism].

**Source**: Devlin et al. - "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)
**Context**: Introducing pretrained language model

> We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

**Tags**: #pretraining #contribution #bidirectional #llm

---

## [Fine-tuning Simplicity] As a result, the pre-trained [model] can be fine-tuned with just [minimal modification] to create state-of-the-art models for [wide range of tasks], without [major changes].

**Source**: Devlin et al. - "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)
**Context**: Highlighting ease of adaptation to downstream tasks

> As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

**Tags**: #finetuning #simplicity #versatility #llm

---

## [Human vs Machine Gap] While [current method] can [capability], humans can generally [superior capability] from only [minimal input]---something which current [systems] still largely struggle to do.

**Source**: Brown et al. - "Language Models are Few-Shot Learners" (NeurIPS 2020)
**Context**: Contrasting human and machine learning capabilities

> While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions---something which current NLP systems still largely struggle to do.

**Tags**: #motivation #human_comparison #few_shot #llm

---

## [Scaling Discovery] We show that scaling up [component] greatly improves [capability], sometimes even reaching competitiveness with prior state-of-the-art [approaches].

**Source**: Brown et al. - "Language Models are Few-Shot Learners" (NeurIPS 2020)
**Context**: Presenting scaling law discovery

> We show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches.

**Tags**: #scaling #discovery #few_shot #llm

---

## [Alignment Problem] Making [models] bigger does not inherently make them better at [desirable property]. For example, [large models] can [undesirable behavior A], [undesirable behavior B], or simply [undesirable behavior C]. In other words, these models are not [aligned] with their [users].

**Source**: Ouyang et al. - "Training language models to follow instructions" (NeurIPS 2022)
**Context**: Motivating alignment research

> Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users.

**Tags**: #alignment #problem #motivation #llm

---

## [RLHF Pipeline] We illustrate a [N]-step method: (1) [step A], (2) [step B], and (3) [step C] on [learned component].

**Source**: Ouyang et al. - "Training language models to follow instructions" (NeurIPS 2022)
**Context**: Describing multi-stage training pipeline

> We illustrate a three-step method: (1) supervised fine-tuning (SFT), (2) reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO) on this reward model.

**Tags**: #methodology #pipeline #rlhf #llm

---

## [Alignment vs Scale] [Alignment method] is very effective at making [models] more [desirable property], more so than a [N]x [scaling dimension] increase. This suggests that [investment recommendation].

**Source**: Ouyang et al. - "Training language models to follow instructions" (NeurIPS 2022)
**Context**: Comparing alignment to scaling

> RLHF is very effective at making language models more helpful to users, more so than a 100x model size increase. This suggests that right now increasing investments in alignment of existing language models is more cost-effective than training larger models.

**Tags**: #alignment #scaling #efficiency #llm

---

## [Emergent Ability] We explore how [technique]---[brief description]---significantly improves the ability of [models] to perform [complex task].

**Source**: Wei et al. - "Chain-of-Thought Prompting Elicits Reasoning" (NeurIPS 2022)
**Context**: Introducing emergent reasoning capability

> We explore how generating a chain of thought---a series of intermediate reasoning steps---significantly improves the ability of large language models to perform complex reasoning.

**Tags**: #reasoning #emergent #chain_of_thought #llm

---

## [Scale Dependence] [Technique] is an emergent ability of model scale. [Technique] does not positively impact performance for [small models], and only yields performance gains when used with models of ~[threshold] parameters.

**Source**: Wei et al. - "Chain-of-Thought Prompting Elicits Reasoning" (NeurIPS 2022)
**Context**: Describing scale-dependent capabilities

> Chain-of-thought prompting is an emergent ability of model scale. Chain-of-thought prompting does not positively impact performance for small models, and only yields performance gains when used with models of ~100B parameters.

**Tags**: #emergent #scaling #threshold #llm

---

## [Striking Results] The empirical gains can be striking. For instance, prompting a [size] model with just [minimal input] achieves state-of-the-art [metric] on [benchmark], surpassing even [stronger baseline].

**Source**: Wei et al. - "Chain-of-Thought Prompting Elicits Reasoning" (NeurIPS 2022)
**Context**: Highlighting surprising empirical results

> The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain of thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.

**Tags**: #results #surprising #benchmark #llm

---
