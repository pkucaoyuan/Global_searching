# 修改方案: 格式和排版

## 问题描述

1. 当前是双栏格式，需要改为单栏
2. 表格可能看起来内容不够充实
3. 需要使用archive格式（无会议名）

## 解决方案

### 1. 文档类修改

```latex
% 原来
\documentclass[10pt,twocolumn]{article}

% 改为
\documentclass[11pt]{article}
```

### 2. 页面边距调整

```latex
% 原来
\usepackage[margin=1in]{geometry}

% 保持不变，1inch对单栏是合适的
```

### 3. 表格排版优化

**原则**: 让表格看起来更充实

**策略1**: 合并相关表格
```latex
% 例如：将Table 1 (SD) 和 Table 2 (EDM) 合并为一个大表
% 左边是SD结果，右边是EDM结果
```

**策略2**: 增加表格列
```latex
% 例如：增加更多NFE点
% 或增加更多evaluation metrics
```

**策略3**: 使用更宽的表格
```latex
\begin{table*}[t]  % 使用table*占满页宽
...
\end{table*}
```

### 4. 表格合并示例

```latex
\begin{table}[t]
\centering
\caption{Results across models and NFE budgets.
Left: Stable Diffusion. Right: EDM.}
\label{tab:combined_results}
\begin{tabular}{c|cc|cc}
\toprule
& \multicolumn{2}{c|}{\textbf{Stable Diffusion}} & \multicolumn{2}{c}{\textbf{EDM}} \\
\textbf{NFE} & \textbf{Naive} & \textbf{Ours} & \textbf{Naive} & \textbf{Ours} \\
\midrule
\multicolumn{5}{c}{\textit{Brightness}} \\
\midrule
Low & ... & ... & ... & ... \\
Mid & ... & ... & ... & ... \\
High & ... & ... & ... & ... \\
\midrule
\multicolumn{5}{c}{\textit{Compressibility}} \\
\midrule
Low & ... & ... & ... & ... \\
Mid & ... & ... & ... & ... \\
High & ... & ... & ... & ... \\
\bottomrule
\end{tabular}
\end{table}
```

### 5. 参考ICLR风格

- 检查ICLR 2024/2025上类似工作的表格设计
- 注意他们如何展示实验结果
- 学习视觉呈现技巧

### 6. 验证清单

- [ ] 文档是单栏格式
- [ ] 没有会议名字
- [ ] 表格看起来充实
- [ ] 整体视觉效果良好
