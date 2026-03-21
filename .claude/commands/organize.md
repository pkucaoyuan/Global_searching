# /organize - 代码库结构整理与文档补全

你是代码库整理专家。执行以下任务来整理项目结构和补全文档。

## 执行流程

按顺序执行以下步骤，每步完成后报告结果：

### 步骤 1: 结构扫描

使用 Glob 和 Read 工具扫描项目，检查：

**文件命名问题**（禁用的前缀/后缀）：
- 禁用前缀: `enhanced_`, `integrated_`, `cleaned_`, `improved_`, `final_`, `optimized_`, `new_`, `old_`
- 禁用后缀: `_enhanced`, `_improved`, `_v2`, `_v3`, `_new`, `_old`, `_temp`, `_final`, `_backup`

**文件位置问题**：
- `test_*.py` 或 `*_test.py` → 应在 `tests/` 目录
- `run_*.sh` 或 `monitor_*.sh` → 应在 `scripts/` 目录
- `analyze_*.py` 或 `extract_*.py` → 应在 `scripts/analysis/` 目录

**忽略的目录**：`.git`, `__pycache__`, `node_modules`, `.venv`, `outputs`, `data/`, `chrome_user_data/`

扫描命令示例：
```
Glob: **/*.py 找出所有 Python 文件
Glob: **/*.sh 找出所有 Shell 脚本
```

### 步骤 2: 执行结构整理

如果发现文件位置问题，使用 Bash 工具移动文件：
```bash
mkdir -p tests scripts/analysis
mv path/to/test_*.py tests/
mv path/to/analyze_*.py scripts/analysis/
```

### 步骤 3: 扫描未完成的文档

检查以下目录中的文档完整性：

**进度文档** (`docs/progress/`)：
- 查找包含 `[请填写]`、`[TODO]`、`[具体完成的任务]` 等占位符的文件
- 查找状态为 `[ ] 进行中` 但内容未填写的文件

**模块文档** (`src/*/README.md`)：
- 检查哪些模块缺少 README.md
- 检查 README.md 是否包含 `[请填写]` 占位符

**用户指南** (`docs/guides/`)：
- 检查是否有过时或不完整的文档

### 步骤 4: 补全进度文档

对于未完成的进度文档，执行以下操作：

1. **读取 git 历史**获取实际完成的工作：
```bash
git log --since="YYYY-MM-DD" --until="YYYY-MM-DD" --oneline
git diff --stat HEAD~10..HEAD  # 查看最近变更
```

2. **分析变更内容**，填写：
   - 完成内容（根据 commit messages）
   - 代码变更（根据 git diff）
   - 遇到的问题（如果有相关 commits）

3. **使用 Edit 工具**更新文档，替换占位符为实际内容

### 步骤 5: 补全模块文档

对于缺少或不完整的模块 README：

1. **分析模块代码**：
   - 读取模块中的 Python 文件
   - 提取类定义、函数定义、docstrings
   - 识别主要功能和依赖

2. **生成/更新 README.md**，包含：
   - 模块概述（基于代码分析）
   - 目录结构
   - 核心组件（类和函数列表）
   - 使用示例
   - 依赖模块

### 步骤 6: 生成健康报告

汇总以下指标：

```
📊 项目健康报告
================
📝 文件命名: X 个问题
📁 文件位置: X 个问题
📚 文档覆盖: X/Y 模块有 README
📋 进度文档: X 个未完成
```

---

## 输出格式

每个步骤完成后，输出：

```
✅ 步骤 N 完成
   - 发现 X 个问题
   - 已修复 Y 个
   - 待处理: [列表]
```

最终输出整体摘要。

---

## 注意事项

1. **不要创建新文件**除非必要，优先编辑现有文件
2. **保留用户已填写的内容**，只补全空白部分
3. **使用中文**填写文档内容
4. **引用实际代码和 commits**，不要编造内容
5. **询问用户确认**再执行破坏性操作（删除、大量移动）

---

## 快速命令

用户可以指定子任务：
- `/organize scan` - 仅扫描，不执行
- `/organize fix` - 执行结构整理
- `/organize docs` - 补全文档
- `/organize progress` - 补全进度文档
- `/organize health` - 生成健康报告
- `/organize all` - 执行全部（默认）

---

现在开始执行整理任务。首先进行结构扫描...
