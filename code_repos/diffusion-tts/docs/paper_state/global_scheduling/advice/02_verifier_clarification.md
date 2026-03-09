# 修改方案: Verifier来源说明

## 问题描述

读者可能会问:
- Verifier是哪来的？
- 是不是需要单独训练？
- 这个设定合理吗？

## 解决方案

### 1. 在Methodology中添加说明

在sec:method-global的verifier定义后添加:

```latex
% 在 "evaluated by a verifier $v(x_0, c)$" 后添加

We emphasize that the verifier $v$ is \emph{taken as given} and is not trained
as part of our method. In practice, verifiers can take various forms depending
on the application: perceptual quality metrics (e.g., CLIP score, aesthetic
predictors), task-specific reward functions (e.g., object detection confidence),
or simple heuristics (e.g., image brightness, compressibility). This setup
mirrors prior work on reward-guided diffusion sampling~\citep{clark2024directly,prabhudesai2023aligning},
where off-the-shelf reward models guide the generation process without
additional training.
```

### 2. 在Experimental Setting中具体化

```latex
% 在sec:exp-setting中添加

Our evaluation uses two lightweight verifiers adopted from prior
noise-trajectory-search benchmarks: \textbf{Brightness}, computed as the
mean perceived luminance of the generated image, and \textbf{Compressibility},
derived from the JPEG file size after maximum compression (normalized to
$[0,1]$). These verifiers are chosen for reproducibility and do not require
human annotation or learned models.
```

### 3. 需要添加的引用

```bibtex
@article{clark2024directly,
  title={Directly fine-tuning diffusion models on differentiable rewards},
  author={Clark, Kevin and others},
  journal={ICLR},
  year={2024}
}

@article{prabhudesai2023aligning,
  title={Aligning text-to-image diffusion models with reward backpropagation},
  author={Prabhudesai, Mihir and others},
  journal={arXiv preprint arXiv:2310.03739},
  year={2023}
}
```

### 4. 验证清单

- [ ] 是否明确说明verifier是"taken as given"
- [ ] 是否给出verifier的具体例子
- [ ] 是否引用使用similar verifier的论文
- [ ] 读者是否不会再质疑verifier的来源
