# Implementation Verification Against Original diffusion-tts

本文档验证我们的实现与原始 `diffusion-tts` 实现的一致性。

## 1. EDM Model Implementation

### ✅ 初始噪声生成
- **原始**: `latents = torch.randn([batch_size, 3, 64, 64])` (标准正态噪声)
- **原始**: `x_next = latents.to(torch.float64) * t_steps[0]`
- **我们的**: `sample_noise()` 返回标准正态噪声，`sample()` 中乘以 `t_steps[0]`
- **状态**: ✅ 已修复，完全一致

### ✅ 去噪步骤 (denoise_step)
- **原始**: 使用 Heun 方法（二阶校正）
- **原始**: `denoised = net(x_hat, t_hat, class_labels_for_step).to(torch.float64)`
- **原始**: `d_cur = (x_hat - denoised) / t_hat`
- **原始**: `x_next = x_hat + (t_next - t_hat) * d_cur`
- **原始**: 如果有下一步，进行二阶校正
- **我们的**: 完全按照原始实现，包括 NaN 检测和除零保护
- **状态**: ✅ 已实现，完全一致

### ✅ 图像归一化
- **原始**: `image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)`
- **我们的**: 使用相同的公式
- **状态**: ✅ 已修复，完全一致

## 2. Best-of-N (Rejection Sampling) Implementation

### ✅ 算法流程
- **原始**: 
  1. `x_next_expanded = x_next.repeat_interleave(N, dim=0)` - 扩展到 N 倍
  2. 对所有 N 个候选运行完整的去噪流程
  3. 最后评分并选择最好的
- **我们的**: 完全按照原始实现，使用 batch 方法
- **状态**: ✅ 已修复，完全一致

### ✅ 评分和选择
- **原始**: `image_for_scoring = (x_next_expanded * 127.5 + 128).clip(0, 255).to(torch.uint8)`
- **原始**: `scores = method_params.scorer(image_for_scoring, class_labels_expanded, timesteps)`
- **原始**: `scores = scores.view(batch_size, N)`
- **原始**: `best_indices = scores.argmax(dim=1)`
- **我们的**: 完全按照原始实现
- **状态**: ✅ 已实现，完全一致

## 3. Zero-Order Search Implementation

### ✅ Lambda 缩放
- **原始**: `lambda_param = method_params.lambda_param * np.sqrt(3 * 64 * 64)`
- **我们的**: 动态计算 `lambda_param * np.sqrt(num_channels * image_size * image_size)`
- **状态**: ✅ 已实现，完全一致

### ✅ 局部搜索
- **原始**: 在每个时间步进行 K 次迭代
- **原始**: 每次迭代生成 N 个候选噪声（通过扰动 pivot）
- **原始**: 对候选进行去噪并评分（使用 x_0 估计）
- **原始**: 选择最佳候选作为新的 pivot
- **我们的**: 逻辑一致，但需要验证评分使用的是 x_0 估计
- **状态**: ⚠️ 需要验证评分逻辑

### ✅ 候选噪声生成
- **原始**: 单位向量归一化，然后随机缩放（0 到 lambda 之间）
- **原始**: `candidate_noise = base_noise + scale * random_direction`
- **我们的**: 相同逻辑
- **状态**: ✅ 已实现

## 4. Scorer Implementation

### ✅ BrightnessScorer
- **原始**: `0.2126*R + 0.7152*G + 0.0722*B`
- **原始**: 接受 uint8 或 float32 [0,1] 输入
- **我们的**: 完全一致
- **状态**: ✅ 已实现，完全一致

### ✅ CompressibilityScorer
- **原始**: JPEG 压缩，计算压缩后大小
- **原始**: `normalized_score = 1.0 - min(1.0, max(0.0, (compressed_size - self.min_size) / (self.max_size - self.min_size)))`
- **原始**: 需要 uint8 格式输入
- **我们的**: 完全一致
- **状态**: ✅ 已修复，完全一致

### ✅ ImageNetScorer
- **原始**: 使用 OpenAI 的 64x64 ImageNet 分类器
- **原始**: 计算目标类别的概率
- **原始**: 需要 one-hot 编码的 class_labels
- **我们的**: 完全一致
- **状态**: ✅ 已实现，完全一致

## 5. Class Labels Generation

### ✅ 生成方式
- **原始**: `class_labels = torch.eye(1000)[torch.randint(1000, size=[g * g])]`
- **我们的**: `all_class_labels = torch.eye(num_classes)[torch.randint(num_classes, size=(num_samples,))]`
- **状态**: ✅ 已实现，完全一致

## 6. 关键修复记录

1. **图像归一化**: 修复为 `(x * 127.5 + 128).clip(0, 255).to(uint8)`
2. **CompressibilityScorer**: 修复为接受 uint8 输入
3. **Best-of-N**: 修复为 batch 方法（repeat_interleave）
4. **sample_noise**: 修复为返回标准正态噪声（不乘以 sigma_max）
5. **EDM sample()**: 修复 initial_noise 处理（乘以 t_steps[0]）
6. **NaN 处理**: 添加了完整的 NaN/Inf 检测和处理
7. **NFE 计数**: 修复了 denoise_step 中的 NFE 计数

## 7. 需要进一步验证的点

1. **Zero-Order Search 评分**: 确认使用的是 x_0 估计（denoised）而不是 x_{t-1}
2. **ε-greedy 实现**: 验证 epsilon 参数的使用（全局探索 vs 局部搜索）
3. **NFE 计数**: 验证所有方法的 NFE 计数是否准确

## 总结

✅ **已完全验证并修复**:
- EDM 模型实现
- Best-of-N (Rejection Sampling) 实现
- 所有 Scorer 实现
- 图像归一化和格式转换

⚠️ **需要进一步验证**:
- Zero-Order Search 的评分逻辑（使用 x_0 vs x_{t-1}）
- ε-greedy 的具体实现细节

