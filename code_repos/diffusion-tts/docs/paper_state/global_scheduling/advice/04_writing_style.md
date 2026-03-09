# 修改方案: 写作风格修正

## 问题描述

文章中有GPT写作痕迹，特别是:
- 过多使用hyphen (半字线)
- 某些表达方式是GPT典型风格

## Hyphen检查清单

### 需要检查的常见hyphen词汇

| 当前写法 | 检查是否必要 | 建议 |
|----------|-------------|------|
| step-aware | 保留（复合形容词）| OK |
| test-time | 保留（复合形容词）| OK |
| inference-time | 保留（复合形容词）| OK |
| best-of-N | 保留（专有名词）| OK |
| offline-to-online | 考虑改为 "offline to online" | 检查 |
| pre-assigned | 可改为 "preassigned" | 检查 |
| per-step | 保留 | OK |
| per-timestep | 保留 | OK |
| fine-grained | 保留 | OK |
| high-value | 保留 | OK |
| low-value | 保留 | OK |

### 规则

1. **保留**: 复合形容词修饰名词时 (e.g., "step-aware scheduling")
2. **保留**: 专有术语 (e.g., "best-of-N")
3. **检查**: 非必要的连接 (e.g., "offline-to-online" 可能改为 "offline to online")
4. **移除**: GPT常加的不必要hyphen

## GPT风格检查

### 常见GPT表达模式

| GPT风格 | 建议替换 |
|---------|----------|
| "it is worth noting that" | 直接陈述 |
| "importantly," | 移除或改为具体说明 |
| "crucially," | 移除或改为具体说明 |
| "significantly" (过度使用) | 减少使用 |
| "dramatically" | 用具体数字 |

### 当前文章中的检查点

1. Line 68: "Importantly, our framework applies..." - 可以保留
2. 检查全文是否有过多的副词修饰

## 执行建议

1. **不要大规模修改**: 先标记，与用户确认后再改
2. **保持学术风格**: hyphen在学术写作中是正常的
3. **重点关注**: 明显的GPT痕迹，而非所有hyphen
