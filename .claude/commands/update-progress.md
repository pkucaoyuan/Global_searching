# /update-progress - 进度文档更新工具

你是进度文档管理专家。帮助用户创建、更新和维护项目进度报告。

**⚠️ 核心要求**:
1. 生成进度报告后，**必须自动更新 CLAUDE.md** 中的相关模块（Code/Scripts/Docs）
2. 使用 Edit 工具直接修改 CLAUDE.md，不要只是提供指导

## 用户输入

进度更新内容或参数：`$ARGUMENTS`

## 参数说明

| 命令 | 功能 |
|------|------|
| `/update-progress` | 自动分析 git 变更，生成新进度报告 |
| `/update-progress <主题>` | 为指定主题创建进度报告 |
| `/update-progress list` | 列出所有现有进度报告 |
| `/update-progress update <文件名>` | 更新现有进度报告 |
| `/update-progress summary` | 生成周/月进度汇总 |
| `/update-progress sync` | **同步更新 CLAUDE.md 模块状态** |
| `/update-progress sync code` | 仅更新代码模块 |
| `/update-progress sync scripts` | 仅更新脚本模块 |
| `/update-progress sync docs` | 仅更新文档模块 |
| `/update-progress full` | 完整流程：生成报告 + 同步 CLAUDE.md |

---

## 执行流程

### 步骤 1: 收集 Git 变更信息

```bash
# 获取最近的进度报告日期
LAST_REPORT=$(ls -t docs/progress/*.md 2>/dev/null | head -1)

# 获取最近的 commits（过去 7 天或自上次报告以来）
git log --since="7 days ago" --oneline --no-merges

# 获取变更统计
git diff --stat HEAD~20..HEAD 2>/dev/null || git diff --stat

# 获取新增文件
git diff --name-only --diff-filter=A HEAD~20..HEAD 2>/dev/null

# 获取修改文件
git diff --name-only --diff-filter=M HEAD~20..HEAD 2>/dev/null

# 获取删除文件
git diff --name-only --diff-filter=D HEAD~20..HEAD 2>/dev/null
```

### 步骤 2: 分析变更类型

根据 git 变更自动分类（通用模式）：

| 文件模式 | 分类 | 说明 |
|---------|------|------|
| `src/**/*.py` | 核心代码 | 主要业务逻辑 |
| `lib/**/*`, `pkg/**/*` | 库代码 | 可复用模块 |
| `scripts/**/*` | 脚本工具 | 自动化脚本 |
| `experiments/**/*` | 实验代码 | 实验和验证 |
| `tests/**/*` | 测试代码 | 单元/集成测试 |
| `docs/**/*` | 文档 | 项目文档 |
| `paper/**/*` | 论文 | 学术论文 |
| `config/**/*`, `*.yaml`, `*.json` | 配置 | 配置文件 |
| `.claude/**/*` | Claude 配置 | AI 助手配置 |
| `data/**/*`, `datasets/**/*` | 数据 | 数据文件 |
| `outputs/**/*`, `results/**/*` | 输出 | 实验结果 |

### 步骤 3: 生成进度报告

**文件命名**: `docs/progress/YYYY_MM_DD_{主题}.md`

**报告模板**:

```markdown
# [功能/主题名称] - YYYY-MM-DD

## 状态
✅ 已完成 / 🚧 进行中 / ❌ 阻塞

## 概要
[1-2 句话总结本次工作的主要成果]

## 完成内容

### 主要功能
- **功能 1**: [描述具体实现]
- **功能 2**: [描述具体实现]

### 改进优化
- [改进点 1]
- [改进点 2]

### Bug 修复
- [修复问题 1]
- [修复问题 2]

## 代码变更

| 类型 | 数量 |
|------|------|
| 新增文件 | X |
| 修改文件 | Y |
| 删除文件 | Z |

### 关键文件变更

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `path/to/file.py` | 新增/修改/删除 | [变更说明] |

## 相关 Commits

```
[commit hash] [commit message]
[commit hash] [commit message]
```

## 遇到的问题与解决

### 问题 1: [问题标题]
**现象**: [描述问题]
**原因**: [分析原因]
**解决**: [解决方案]

## 测试情况

- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试验证

## 下一步计划

- [ ] [待办任务 1]
- [ ] [待办任务 2]
- [ ] [待办任务 3]

## 相关文档

- [相关指南](../guides/xxx.md)
- [相关配置](../../.claude/xxx.md)

---

**作者**: [自动生成]
**审核**: 待审核
```

### 步骤 4: 智能内容填充

根据 git commits 自动提取：

```bash
# 从 commit messages 提取功能点
git log --since="$LAST_REPORT_DATE" --pretty=format:"%s" --no-merges | \
  grep -E "^(feat|add|implement|create)" | head -10

# 从 commit messages 提取修复
git log --since="$LAST_REPORT_DATE" --pretty=format:"%s" --no-merges | \
  grep -E "^(fix|bugfix|resolve|repair)" | head -10

# 从 commit messages 提取改进
git log --since="$LAST_REPORT_DATE" --pretty=format:"%s" --no-merges | \
  grep -E "^(improve|enhance|optimize|refactor|update)" | head -10
```

### 步骤 5: 用户确认与编辑

展示生成的报告预览，询问用户：

```
已生成进度报告预览：

---
[报告内容]
---

请选择：
1. [保存] 保存到 docs/progress/
2. [编辑] 修改后再保存
3. [取消] 放弃本次生成
```

---

## 步骤 6: 自动更新 CLAUDE.md（必须执行）

**⚠️ 强制要求**: 生成进度报告后，你**必须**使用 Edit 工具自动更新 CLAUDE.md。

### 6.1 读取并分析 CLAUDE.md 结构

```bash
# 使用 Read 工具读取 .claude/CLAUDE.md
# 识别以下章节位置：
grep -n "## 2. Architecture" CLAUDE.md  # Directory Structure
grep -n "## 4. Status" CLAUDE.md         # 项目状态
grep -n "### Documentation" CLAUDE.md   # 文档列表
grep -n "### Experiments" CLAUDE.md     # 实验列表
```

### 6.2 根据变更类型更新对应章节

**你必须使用 Edit 工具自动更新以下所有相关章节：**

---

#### A. `src/` 变更 → 更新 `Directory Structure` + `Status`

**1. 更新 Directory Structure（如有新目录/文件）：**
```markdown
## 2. Architecture & conventions
### Directory Structure
src/
├── new_module/              # 🆕 新增模块
│   └── new_file.py          # 功能描述
```

**2. 更新 Status 章节（当前 Phase 下）：**
```markdown
*   **New Components**:
    *   `src/module/new_file.py`: [功能描述]
*   **Updated Files**:
    *   `src/module/existing.py`: [修改说明] (+X lines)
```

---

#### B. `scripts/` 变更 → 更新/创建 Scripts 章节

在 Status 的当前 Phase 下添加或更新：
```markdown
*   **Scripts**:
    *   `scripts/experiments/xxx.py`: [用途]
    *   `scripts/analysis/yyy.py`: [用途]
```

---

#### C. `docs/` 变更 → 更新 Documentation 章节

**1. 进度报告** → 添加到 `#### 进度报告` 或 `*   **Reports**:`
```markdown
*   `docs/progress/YYYY_MM_DD_xxx.md` - [主题]
```

**2. 技术文档** → 添加到相应章节
```markdown
*   **Validation Reports**:
    *   `docs/experiments/new_report.md` (详细说明)
```

---

#### D. `experiments/` 变更 → 更新 Experiments 章节

在 Status 的当前 Phase 下添加：
```markdown
*   **Experiments**:
    *   `experiments/new_exp.py`: [实验说明]
    *   `experiments/theory_validation/exp_xxx.py`: [定理验证]
```

---

#### E. `paper/` 变更 → 更新论文章节

如果 `paper/` 目录有变更：
```markdown
#### 论文结构
paper/journal/main.tex
├── sections/new_section.tex   ← 🆕 新增章节
```

---

#### F. `tests/` 变更 → 更新 Tests 章节

```markdown
*   **Tests**: X unit tests (all passing)
    *   `tests/module/test_xxx.py`: [测试说明]
```

---

#### G. `.claude/` 变更 → 更新 Standards References

```markdown
## 5. Standards References
| [`new_guide.md`](.claude/standards/new_guide.md) | 新指南说明 |
```

---

### 6.3 执行自动更新

**执行顺序：**

1. **Read** `.claude/CLAUDE.md`
2. **分析** git 变更，确定需要更新的章节
3. **Edit** 每个需要更新的章节：
   - 找到章节位置
   - 插入/更新内容
   - 保持格式一致
4. **验证** 更新后的文件

### 6.4 完整更新示例

**git diff 输出：**
```
A  src/bai_judge/audit/new_allocator.py
M  src/bai_judge/algorithms/lucb_joint.py
A  scripts/experiments/run_new_exp.py
A  docs/progress/2026_01_31_feature_update.md
A  experiments/new_experiment.py
M  paper/journal/sections/experiments.tex
```

**自动执行的 Edit 操作：**

```markdown
# Edit 1: 更新 Directory Structure (如果是新目录)
## 2. Architecture & conventions
### Directory Structure
src/
├── bai_judge/
│   ├── audit/
│   │   └── new_allocator.py   # 🆕 新分配器

# Edit 2: 更新当前 Phase Status
#### Phase N: [当前阶段名] (YYYY-MM-DD)
*   **New Components**:
    *   `src/bai_judge/audit/new_allocator.py`: [功能描述]
*   **Updated Files**:
    *   `src/bai_judge/algorithms/lucb_joint.py`: [修改说明] (+X lines)
*   **Scripts**:
    *   `scripts/experiments/run_new_exp.py`: [用途]
*   **Experiments**:
    *   `experiments/new_experiment.py`: [实验说明]
*   **Reports**:
    *   `docs/progress/2026_01_31_feature_update.md`

# Edit 3: 更新论文结构 (如有变更)
#### 论文结构
├── sections/experiments.tex   ← ✏️ 更新实验章节
```

---

### 6.5 验证检查清单

更新后确认：
- [ ] Directory Structure 已更新（新目录/文件）
- [ ] Status 章节的当前 Phase 已更新
- [ ] 新增代码文件已记录
- [ ] 新增脚本已记录
- [ ] 新增文档已记录
- [ ] 新增实验已记录
- [ ] 进度报告链接已添加
- [ ] 日期正确

---

## 进度报告规范

### 命名规范

**格式**: `YYYY_MM_DD_{主题简称}.md`

**示例**:
- `2026_01_28_forum_reply_filter.md`
- `2026_01_28_rag_optimization.md`
- `2026_01_28_weekly_summary.md`

**禁止**:
- ❌ `progress_v2.md`
- ❌ `update_new.md`
- ❌ `latest_changes.md`

### 状态标记

| 标记 | 含义 | 使用场景 |
|------|------|---------|
| ✅ 已完成 | 功能完整、测试通过 | 可以发布/部署的功能 |
| 🚧 进行中 | 开发中、部分完成 | 仍在开发的功能 |
| ❌ 阻塞 | 被阻塞、需要帮助 | 等待依赖或决策 |
| ⏸️ 暂停 | 暂时搁置 | 优先级调整 |

### 内容要求

**必须包含**:
- [ ] 状态标记
- [ ] 概要（1-2 句话）
- [ ] 完成内容列表
- [ ] 代码变更统计
- [ ] 下一步计划

**建议包含**:
- [ ] 关键文件变更表格
- [ ] 相关 commits
- [ ] 遇到的问题与解决
- [ ] 测试情况

**可选包含**:
- [ ] 架构图/流程图
- [ ] 性能指标
- [ ] 相关文档链接

---

## 周/月汇总报告

### 触发命令

```
/update-progress summary          # 生成本周汇总
/update-progress summary week     # 指定周汇总
/update-progress summary month    # 月度汇总
```

### 汇总模板

```markdown
# 进度汇总 - YYYY-MM-DD 至 YYYY-MM-DD

## 总体概况

| 指标 | 数值 |
|------|------|
| 完成功能 | X 项 |
| 修复 Bug | Y 个 |
| 新增代码 | +Z 行 |
| Commits | N 个 |

## 主要成果

### 1. [功能/模块 1]
- 进度报告: [链接]
- 状态: ✅/🚧
- 简述: ...

### 2. [功能/模块 2]
- 进度报告: [链接]
- 状态: ✅/🚧
- 简述: ...

## 关键里程碑

- [x] 里程碑 1
- [x] 里程碑 2
- [ ] 里程碑 3 (进行中)

## 问题与风险

| 问题 | 影响 | 状态 | 负责人 |
|------|------|------|--------|
| [问题1] | 中 | 已解决 | - |
| [问题2] | 高 | 跟进中 | - |

## 下周/下月计划

- [ ] 计划 1
- [ ] 计划 2
- [ ] 计划 3

## 相关进度报告

- [报告1](./YYYY_MM_DD_xxx.md)
- [报告2](./YYYY_MM_DD_xxx.md)
```

---

## 快速操作

### 创建今日进度报告

```bash
# 自动生成文件名
DATE=$(date +%Y_%m_%d)
FILE="docs/progress/${DATE}_update.md"
```

### 查看历史报告

```bash
# 列出最近 10 个报告
ls -lt docs/progress/*.md | head -10

# 按关键词搜索
grep -l "forum" docs/progress/*.md
grep -l "crawler" docs/progress/*.md
```

### 检查报告完整性

```bash
# 检查是否有状态标记
grep -L "## 状态" docs/progress/*.md

# 检查是否有下一步计划
grep -L "## 下一步" docs/progress/*.md
```

---

## 自动化建议

### 触发条件

建议在以下情况创建进度报告：

1. **功能完成** - 完成一个独立功能
2. **里程碑达成** - 达成项目里程碑
3. **重要修复** - 修复重大 bug
4. **每周结束** - 周五进行周汇总
5. **Sprint 结束** - 迭代结束时

### Git Hook 集成（可选）

在 `.git/hooks/post-commit` 中添加提醒：

```bash
#!/bin/bash
# 如果 commits 数量达到阈值，提醒创建进度报告
COMMITS_TODAY=$(git log --since="today 00:00" --oneline | wc -l)
if [ $COMMITS_TODAY -ge 5 ]; then
    echo "提示: 今日已有 $COMMITS_TODAY 个 commits，考虑创建进度报告"
fi
```

---

## 注意事项

1. **保持简洁**: 报告应该在 5 分钟内能读完
2. **聚焦价值**: 强调"做了什么"而非"怎么做的"
3. **可追溯**: 关联相关 commits 和文档
4. **实事求是**: 如实反映进度和问题
5. **及时更新**: 功能完成后尽快记录

---

## 示例

### 示例 1: 自动生成

```
/update-progress
```
→ 自动分析最近 git 变更，生成进度报告

### 示例 2: 指定主题

```
/update-progress 论坛回复过滤器优化
```
→ 为指定主题创建报告，预填充相关变更

### 示例 3: 更新现有报告

```
/update-progress update 2026_01_28_system_improvements.md
```
→ 读取并更新指定报告

### 示例 4: 生成周汇总

```
/update-progress summary week
```
→ 汇总本周所有进度报告

### 示例 5: 同步更新 CLAUDE.md 模块

```
/update-progress sync
```
→ 分析 git 变更，自动更新 CLAUDE.md 中的 code/scripts/docs 模块

### 示例 6: 仅更新代码模块

```
/update-progress sync code
```
→ 仅更新 CLAUDE.md 中的代码相关章节

### 示例 7: 完整流程

```
/update-progress full
```
→ 生成进度报告 + 同步 CLAUDE.md 模块（推荐用于阶段性总结）

---

## Sync 命令详细说明

### `/update-progress sync` 执行流程

1. **分析 git 变更**
   ```bash
   # 获取变更文件列表
   git diff --name-status HEAD~20..HEAD
   ```

2. **分类变更**
   - `src/`, `lib/` → Code 模块
   - `scripts/` → Scripts 模块
   - `docs/` → Documents 模块
   - `experiments/` → Experiments 模块
   - `tests/` → Tests 模块

3. **读取 CLAUDE.md 当前状态**
   ```bash
   # 找到 Status 章节
   grep -n "## Status" CLAUDE.md
   ```

4. **生成更新内容**
   - 新增文件 → 添加到对应模块列表
   - 修改文件 → 更新变更说明
   - 删除文件 → 从列表移除或标记为删除

5. **用户确认**
   ```
   即将更新 CLAUDE.md 以下章节：

   [Code] 新增 3 个文件，修改 2 个文件
   [Scripts] 新增 1 个脚本
   [Docs] 新增 2 个文档

   确认更新？[Y/n]
   ```

6. **写入 CLAUDE.md**

### 输出格式

**Code 模块更新**:
```markdown
#### Code Changes (YYYY-MM-DD)
*   **New**:
    *   `src/module/new_feature.py`: [自动提取的描述]
*   **Modified**:
    *   `src/module/existing.py`: [变更说明] (+X/-Y lines)
*   **Deleted**:
    *   `src/module/deprecated.py`
```

**Scripts 模块更新**:
```markdown
#### Scripts (YYYY-MM-DD)
| 脚本 | 用途 | 状态 |
|------|------|------|
| `scripts/new_script.py` | [描述] | 🆕 |
| `scripts/updated.sh` | [描述] | ✏️ 更新 |
```

**Documents 模块更新**:
```markdown
#### Documentation (YYYY-MM-DD)
*   **Progress**: `docs/progress/YYYY_MM_DD_xxx.md`
*   **Technical**: `docs/experiments/xxx.md`
*   **Guides**: `docs/guides/xxx.md`
```

---

现在开始分析 git 变更并生成进度报告...
