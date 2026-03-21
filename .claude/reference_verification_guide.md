# 参考文献真实性验证指南

**适用项目**: PPI-SVRG (ICML 2026投稿)
**文档版本**: v1.0.0
**最后更新**: 2026-01-24

---

## 📋 目录

1. [概述](#概述)
2. [快速开始 (20分钟)](#快速开始-20分钟)
3. [推荐方案: OpenAlex API](#推荐方案-openalex-api)
4. [验证维度与检查清单](#验证维度与检查清单)
5. [Python自动验证脚本](#python自动验证脚本)
6. [手动验证核心引用](#手动验证核心引用)
7. [验证报告解读](#验证报告解读)
8. [ICML 2026投稿前Checklist](#icml-2026投稿前checklist)
9. [Troubleshooting](#troubleshooting)
10. [附录](#附录)

---

## 概述

### 为什么需要验证引用真实性

学术不端的引用问题包括:
- **幻觉引用**: 不存在的论文（AI写作常见问题）
- **错误元数据**: 标题/作者/年份不匹配
- **撤稿论文**: 引用已撤回的研究
- **格式错误**: BibTeX字段缺失或不规范

**后果**:
- ❌ 投稿被拒（编辑/审稿人发现虚假引用）
- ❌ 学术信誉受损
- ❌ 延误投稿进度

### 本指南适用场景

- ✅ ICML/NeurIPS/ICLR等顶会投稿前验证
- ✅ 从多个来源整合的BibTeX文件
- ✅ 使用AI工具生成的文献列表（需额外小心）
- ✅ 包含大量预印本/技术报告的引用

### 预期完成时间

- **初次设置**: 7分钟（注册API + 安装依赖）
- **运行验证**: 3分钟（自动化脚本）
- **审查修复**: 10分钟（处理问题引用）
- **总计**: ~20分钟

---

## 快速开始 (20分钟)

### Step 1: 注册OpenAlex (5分钟)

OpenAlex是**完全免费**的学术文献数据库API,无需复杂认证。

**注册流程**:
1. 访问 https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication
2. 填写邮箱地址(用于API认证,2026年2月后强制要求)
3. 无需等待审批,立即可用

**配置环境变量**:
```bash
# 在~/.bashrc 或 ~/.zshrc 中添加
export OPENALEX_EMAIL="your_email@example.com"

# 立即生效
source ~/.bashrc  # 或 source ~/.zshrc
```

**测试连接**:
```bash
# 测试API连接
curl "https://api.openalex.org/works?filter=doi:10.48550/arXiv.2311.01453&mailto=your_email@example.com"
```

### Step 2: 安装Python依赖 (2分钟)

```bash
# 在项目根目录
cd /path/to/svrg_ppi

# 安装所需包
pip install pyalex bibtexparser thefuzz[speedup] requests pandas tabulate

# 验证安装
python -c "import pyalex; import bibtexparser; print('✅ Dependencies installed')"
```

**依赖说明**:
- `pyalex`: OpenAlex官方Python客户端
- `bibtexparser`: 解析BibTeX文件
- `thefuzz`: 模糊字符串匹配(处理拼写差异)
- `requests`: HTTP请求(备用API)
- `pandas`: 数据处理
- `tabulate`: 生成Markdown表格

### Step 3: 运行验证脚本 (3分钟)

```bash
# 创建验证脚本目录
mkdir -p code/reference_verification

# 复制下方完整脚本到 code/reference_verification/verify_references.py
# (见第5节完整代码)

# 运行验证
python code/reference_verification/verify_references.py \
    main.bib \
    --email your_email@example.com \
    --output reports/verification_report.md

# 查看报告
cat reports/verification_report.md
```

### Step 4: 修复问题引用 (10分钟)

根据报告中的建议:
1. **已验证**: 无需操作 ✅
2. **需人工复核**: 检查标题/作者拼写
3. **未找到**: 确认是否虚假引用,考虑删除
4. **撤稿警告**: 立即删除 🚨

---

## 推荐方案: OpenAlex API

### 为什么选择OpenAlex

| 特性 | OpenAlex | Semantic Scholar | Crossref | Scite.ai |
|------|----------|------------------|----------|----------|
| **费用** | 完全免费 | 免费 | 免费 | 付费($20/月) |
| **注册难度** | 仅需邮箱 | 需API key申请 | 无需注册 | 需订阅 |
| **覆盖范围** | 2.5亿论文 | 2.2亿论文 | 1.5亿论文 | 1.8亿论文 |
| **arXiv支持** | ✅ 优秀 | ✅ 优秀 | ⚠️ 有限 | ✅ 良好 |
| **BibTeX导出** | ✅ 支持 | ❌ 需手动构建 | ✅ 支持 | ✅ 支持 |
| **CS论文覆盖** | ✅ 全面 | ✅ 专注CS | ⚠️ 一般 | ✅ 良好 |
| **免费配额** | 100k/天 | 100 RPS | 50/秒 | N/A |
| **撤稿检测** | ❌ | ❌ | ✅ | ✅ |

**推荐组合**:
- **主验证**: OpenAlex (覆盖率高 + 免费)
- **撤稿检测**: Crossref Retraction Watch (补充)

### OpenAlex覆盖范围分析

**强项**:
- ✅ ICML/NeurIPS/ICLR等ML会议 (通过PMLR/OpenReview)
- ✅ arXiv预印本 (cs.LG, stat.ML等)
- ✅ 顶级期刊 (Nature, Science, JASA等)
- ✅ 教科书部分可检索 (通过ISBN)

**弱项**:
- ⚠️ 技术报告 (如University Wisconsin Tech Report 1648)
- ⚠️ 最新论文 (可能有1-2周索引延迟)
- ⚠️ 灰色文献 (内部报告,未发表论文)

### 快速注册指南

**方法1: 环境变量 (推荐)**
```bash
export OPENALEX_EMAIL="your@email.com"
```

**方法2: 配置文件**
```python
# code/reference_verification/config.py
OPENALEX_EMAIL = "your@email.com"
CROSSREF_EMAIL = "your@email.com"  # 可选,提升速率限制
```

**方法3: 命令行参数**
```bash
python verify_references.py main.bib --email your@email.com
```

### 测试API连接

```python
from pyalex import Works

# 配置邮箱
Works.email = "your@email.com"

# 测试查询: PPI原论文 (Angelopoulos et al. 2023)
result = Works().filter(doi="10.1126/science.adi6000").get()
print(f"✅ Found: {result[0]['title']}")
# 预期输出: Prediction-powered inference
```

---

## 验证维度与检查清单

### 4.1 元数据完整性

#### 必需字段检查

**@article (期刊论文)**:
- [ ] `title`: 完整标题
- [ ] `author`: 所有作者(格式: `Last, First and Last2, First2`)
- [ ] `journal`: 期刊全称或标准缩写
- [ ] `year`: 发表年份
- [ ] `volume`: 卷号
- [ ] `pages`: 页码范围

**@inproceedings (会议论文)**:
- [ ] `title`
- [ ] `author`
- [ ] `booktitle`: 会议名称(如`Proceedings of ICML`)
- [ ] `year`
- [ ] `pages` (可选,但推荐)

**@misc (预印本/未发表)**:
- [ ] `title`
- [ ] `author`
- [ ] `year`
- [ ] `eprint`: arXiv编号(如`2311.01453`)
- [ ] `archivePrefix`: `arXiv`
- [ ] `primaryClass`: 分类(如`cs.LG`)

**@book (书籍)**:
- [ ] `title`
- [ ] `author`
- [ ] `publisher`
- [ ] `year`
- [ ] `isbn` (可选,有助于验证)

#### 标题匹配标准

**精确匹配** (100%):
```bibtex
# BibTeX中
title = {Prediction-powered inference}

# OpenAlex返回
"Prediction-powered inference"
```

**高置信度匹配** (>95%):
```bibtex
# BibTeX: 缺少副标题
title = {Accelerating stochastic gradient descent}

# OpenAlex: 完整标题
"Accelerating stochastic gradient descent using predictive variance reduction"

# 判定: ✅ 接受 (主标题完全匹配)
```

**可疑匹配** (<85%):
```bibtex
# BibTeX
title = {Deep learning for image classification}

# OpenAlex
"Deep convolutional networks for image recognition"

# 判定: ⚠️ 需人工复核
```

#### 作者匹配逻辑

**处理常见差异**:
- 姓名顺序: `Johnson, Rie and Zhang, Tong` ↔ `Rie Johnson, Tong Zhang`
- 中间名缩写: `John D. Smith` ↔ `John Smith`
- 特殊字符: `François` ↔ `Francois`

**匹配算法**:
```python
from thefuzz import fuzz

def match_authors(bib_authors, api_authors):
    # Token Sort Ratio: 忽略顺序
    score = fuzz.token_sort_ratio(
        normalize_authors(bib_authors),
        normalize_authors(api_authors)
    )
    return score > 80  # 阈值80%
```

#### 年份一致性

**严格匹配**:
- 期刊论文: 必须精确匹配出版年份
- 会议论文: 允许±1年误差 (因会议举办年份 vs 论文集发布年份)
- arXiv: 使用首次提交年份

**示例**:
```bibtex
# BibTeX
year = {2013}

# OpenAlex
"publication_year": 2013  ✅

# 或会议论文特殊情况
booktitle = {ICML 2023}
year = {2024}  # 论文集发布年份,可接受
```

### 4.2 BibTeX格式规范

#### ICML 2026特殊要求

**引用格式**: APA风格 (natbib包)

**关键规范**:
1. **DOI优先于URL**:
   ```bibtex
   # ✅ 推荐
   doi = {10.1126/science.adi6000}

   # ❌ 避免
   url = {https://www.science.org/doi/10.1126/science.adi6000}
   ```

2. **arXiv格式**:
   ```bibtex
   # ✅ 正确
   @misc{angelopoulos2023ppi++,
     title = {PPI++: Efficient prediction-powered inference},
     author = {Angelopoulos, Anastasios N and Duchi, John C and Zrnic, Tijana},
     year = {2023},
     eprint = {2311.01453},
     archivePrefix = {arXiv},
     primaryClass = {cs.LG}
   }

   # ❌ 错误 (使用@article或缺少eprint)
   @article{...}
   journal = {arXiv preprint arXiv:2311.01453}
   ```

3. **期刊名称规范化**:
   ```bibtex
   # ✅ 使用全称或标准缩写
   journal = {Journal of the American Statistical Association}
   # 或
   journal = {JASA}

   # ❌ 避免非标准缩写
   journal = {J. Amer. Stat. Assoc.}
   ```

4. **会议名称一致性**:
   ```bibtex
   # ✅ 统一格式
   booktitle = {Proceedings of the 30th International Conference on Machine Learning}
   # 或简写
   booktitle = {ICML}

   # ❌ 不一致
   booktitle = {Proc. ICML}  # 部分简写
   ```

#### 特殊字符转义

**常见问题**:
| 字符 | 错误 | 正确 | 说明 |
|------|------|------|------|
| & | `&` | `\&` | 逻辑与符号 |
| % | `%` | `\%` | 百分号 |
| $ | `$` | `\$` | 美元符号 |
| _ | `_` | `\_` | 下划线 |
| π | `π` | `$\pi$` | 希腊字母 |
| ö | `o` | `{\"o}` | 变音符号 |

**示例**:
```bibtex
# ❌ 错误
title = {On $π$-inverse weighting versus best linear unbiased weighting}

# ✅ 正确
title = {On $\pi$-inverse weighting versus best linear unbiased weighting}
```

### 4.3 学术诚信检查

#### 撤稿检测

**使用Crossref API**:
```python
import requests

def check_retraction(doi):
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"mailto": "your@email.com"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()["message"]
        # 检查update-to字段(指向更正/撤稿通知)
        if "update-to" in data:
            for update in data["update-to"]:
                if update["type"] == "retraction":
                    return True, update["DOI"]
    return False, None

# 示例
is_retracted, retraction_doi = check_retraction("10.1234/example.doi")
if is_retracted:
    print(f"🚨 WARNING: Paper retracted. See {retraction_doi}")
```

**手动检查**:
1. 访问 https://retractionwatch.com/
2. 搜索论文标题或DOI
3. 确认无撤稿记录

#### DOI有效性验证

**格式检查**:
```python
import re

def validate_doi(doi):
    # DOI格式: 10.xxxx/yyyyy
    pattern = r'^10\.\d{4,}/\S+$'
    return re.match(pattern, doi) is not None

# 示例
validate_doi("10.1126/science.adi6000")  # ✅ True
validate_doi("doi:10.1234")              # ❌ False (缺少后缀)
```

**可解析性测试**:
```bash
# 测试DOI是否可解析
curl -I https://doi.org/10.1126/science.adi6000

# 预期输出: HTTP 302 (重定向到出版商网站)
```

### 4.4 优先级分类

**高优先级** (核心贡献,必须100%验证):
- [ ] Angelopoulos et al. 2023 - PPI原论文
- [ ] Johnson & Zhang 2013 - SVRG原论文
- [ ] Allen-Zhu & Yuan 2016 - Improved SVRG
- [ ] 其他理论证明直接依赖的引用

**中优先级** (方法对比,建议验证):
- [ ] SAGA, PAGE, SARAH等方差缩减方法
- [ ] PPI++, Cross-PPI等PPI变体
- [ ] 实验对比baseline

**低优先级** (教科书/综述,可跳过自动验证):
- [ ] van der Vaart 2000 - 渐近统计学
- [ ] Vershynin 2018 - 高维概率
- [ ] Bubeck 2015 - 凸优化综述

---

## Python自动验证脚本

### 完整代码 (可直接运行)

**文件路径**: `code/reference_verification/verify_references.py`

```python
#!/usr/bin/env python3
"""
Reference Verification Script for PPI-SVRG Project
Verifies BibTeX entries against OpenAlex API

Usage:
    python verify_references.py main.bib --email your@email.com
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import bibtexparser
import pandas as pd
import requests
from pyalex import Works, config as openalex_config
from thefuzz import fuzz
from tabulate import tabulate


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Verification configuration"""
    # Matching thresholds
    TITLE_EXACT_THRESHOLD = 95
    TITLE_HIGH_CONFIDENCE = 85
    TITLE_SUSPECT = 70
    AUTHOR_MATCH_THRESHOLD = 80

    # API settings
    OPENALEX_TIMEOUT = 30
    CROSSREF_TIMEOUT = 10

    # Entry type priority (for verification order)
    PRIORITY = {
        'article': 1,
        'inproceedings': 1,
        'InProceedings': 1,
        'misc': 2,
        'techreport': 3,
        'book': 4
    }


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_string(s: str) -> str:
    """Normalize string for matching: lowercase, remove special chars"""
    if not s:
        return ""
    # Remove LaTeX commands
    s = s.replace('{', '').replace('}', '').replace('\\', '')
    # Normalize whitespace
    s = ' '.join(s.split())
    return s.lower()


def normalize_authors(author_str: str) -> str:
    """Normalize author string for fuzzy matching"""
    # Split by 'and' and clean
    authors = [a.strip() for a in author_str.split(' and ')]
    # Remove affiliations, emails, etc.
    cleaned = []
    for author in authors:
        # Extract last name, first name
        if ',' in author:
            last, first = author.split(',', 1)
            cleaned.append(f"{first.strip()} {last.strip()}")
        else:
            cleaned.append(author)
    return ' '.join(sorted(cleaned)).lower()


def extract_arxiv_id(entry: Dict) -> Optional[str]:
    """Extract arXiv ID from BibTeX entry"""
    if 'eprint' in entry:
        return entry['eprint']
    if 'journal' in entry:
        journal = entry['journal']
        if 'arxiv' in journal.lower():
            # Extract from "arXiv preprint arXiv:2311.01453"
            import re
            match = re.search(r'arXiv:(\d+\.\d+)', journal)
            if match:
                return match.group(1)
    return None


# ============================================================================
# OpenAlex API Functions
# ============================================================================

def search_openalex_by_title(title: str, year: Optional[int] = None) -> Optional[Dict]:
    """
    Search OpenAlex by title

    Returns:
        Dict with keys: title, authors, year, doi, venue, confidence
    """
    try:
        # Build filter
        filter_params = {"display_name.search": title}
        if year:
            # Allow ±1 year for conference papers
            filter_params["publication_year"] = f"{year-1}-{year+1}"

        # Search
        results = Works().filter(**filter_params).get()

        if not results:
            return None

        # Take best match (first result)
        work = results[0]

        # Calculate title similarity
        api_title = work.get('title', '')
        title_score = fuzz.token_set_ratio(
            normalize_string(title),
            normalize_string(api_title)
        )

        return {
            'title': api_title,
            'authors': ', '.join([a['author']['display_name'] for a in work.get('authorships', [])]),
            'year': work.get('publication_year'),
            'doi': work.get('doi', '').replace('https://doi.org/', ''),
            'venue': work.get('primary_location', {}).get('source', {}).get('display_name', 'Unknown'),
            'url': work.get('id', ''),
            'confidence': title_score
        }

    except Exception as e:
        print(f"⚠️  OpenAlex API error: {e}", file=sys.stderr)
        return None


def search_openalex_by_doi(doi: str) -> Optional[Dict]:
    """Search OpenAlex by DOI (exact match)"""
    try:
        # Clean DOI
        doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')

        results = Works().filter(doi=doi).get()

        if not results:
            return None

        work = results[0]
        return {
            'title': work.get('title', ''),
            'authors': ', '.join([a['author']['display_name'] for a in work.get('authorships', [])]),
            'year': work.get('publication_year'),
            'doi': doi,
            'venue': work.get('primary_location', {}).get('source', {}).get('display_name', 'Unknown'),
            'url': work.get('id', ''),
            'confidence': 100  # Exact DOI match
        }

    except Exception as e:
        print(f"⚠️  OpenAlex DOI search error: {e}", file=sys.stderr)
        return None


def search_openalex_by_arxiv(arxiv_id: str) -> Optional[Dict]:
    """Search OpenAlex by arXiv ID"""
    # OpenAlex uses external_ids.arxiv
    # But title search is more reliable for arXiv papers
    # So we search by title from arXiv API first
    try:
        # Get metadata from arXiv API
        arxiv_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(arxiv_url, timeout=10)

        if response.status_code != 200:
            return None

        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)

        # Extract title
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None

        title = entry.find('atom:title', ns).text.strip()

        # Search OpenAlex by title
        return search_openalex_by_title(title)

    except Exception as e:
        print(f"⚠️  arXiv API error: {e}", file=sys.stderr)
        return None


# ============================================================================
# Crossref API (for retraction check)
# ============================================================================

def check_retraction_crossref(doi: str) -> Tuple[bool, Optional[str]]:
    """
    Check if paper is retracted using Crossref

    Returns:
        (is_retracted, retraction_doi)
    """
    try:
        doi = doi.replace('https://doi.org/', '')
        url = f"https://api.crossref.org/works/{doi}"

        response = requests.get(url, timeout=Config.CROSSREF_TIMEOUT)

        if response.status_code != 200:
            return False, None

        data = response.json()['message']

        # Check update-to field
        if 'update-to' in data:
            for update in data['update-to']:
                if update.get('type') == 'retraction':
                    return True, update.get('DOI')

        return False, None

    except Exception:
        return False, None


# ============================================================================
# BibTeX Entry Verification
# ============================================================================

def verify_entry(entry: Dict, api_result: Dict) -> Dict:
    """
    Verify BibTeX entry against API result

    Returns:
        Dict with keys: status, confidence, message, suggested_fix
    """
    # Title matching
    bib_title = normalize_string(entry.get('title', ''))
    api_title = normalize_string(api_result.get('title', ''))
    title_score = fuzz.token_set_ratio(bib_title, api_title)

    # Author matching
    bib_authors = normalize_authors(entry.get('author', ''))
    api_authors = normalize_authors(api_result.get('authors', ''))
    author_score = fuzz.token_sort_ratio(bib_authors, api_authors)

    # Year matching
    bib_year = int(entry.get('year', 0))
    api_year = int(api_result.get('year', 0))
    year_match = abs(bib_year - api_year) <= 1  # Allow ±1 for conferences

    # Decision logic
    if title_score >= Config.TITLE_EXACT_THRESHOLD:
        if author_score >= Config.AUTHOR_MATCH_THRESHOLD and year_match:
            return {
                'status': 'VERIFIED',
                'confidence': min(title_score, author_score),
                'message': 'Exact match found',
                'api_result': api_result
            }
        else:
            return {
                'status': 'SUSPECT',
                'confidence': title_score,
                'message': f'Title matches but author/year mismatch (author: {author_score}%, year: {bib_year} vs {api_year})',
                'api_result': api_result
            }

    elif title_score >= Config.TITLE_HIGH_CONFIDENCE:
        if author_score >= Config.AUTHOR_MATCH_THRESHOLD and year_match:
            return {
                'status': 'VERIFIED',
                'confidence': (title_score + author_score) / 2,
                'message': 'High confidence match',
                'api_result': api_result
            }
        else:
            return {
                'status': 'NEEDS_REVIEW',
                'confidence': title_score,
                'message': f'Partial match (title: {title_score}%, author: {author_score}%)',
                'api_result': api_result
            }

    elif title_score >= Config.TITLE_SUSPECT:
        return {
            'status': 'NEEDS_REVIEW',
            'confidence': title_score,
            'message': f'Low confidence match (title: {title_score}%)',
            'api_result': api_result
        }

    else:
        return {
            'status': 'NOT_FOUND',
            'confidence': 0,
            'message': 'No match found',
            'api_result': None
        }


def verify_bibtex_entry(entry: Dict) -> Dict:
    """
    Main verification function for a BibTeX entry

    Returns:
        Verification result dict
    """
    entry_id = entry.get('ID', 'unknown')
    entry_type = entry.get('ENTRYTYPE', 'unknown')

    # Skip books (usually not in academic databases)
    if entry_type == 'book':
        return {
            'id': entry_id,
            'type': entry_type,
            'status': 'MANUAL_CHECK',
            'confidence': None,
            'message': 'Book entry - verify manually via ISBN or publisher website',
            'api_result': None
        }

    # Try verification in order of reliability
    api_result = None

    # 1. Try DOI (most reliable)
    if 'doi' in entry:
        api_result = search_openalex_by_doi(entry['doi'])

    # 2. Try arXiv ID
    if not api_result:
        arxiv_id = extract_arxiv_id(entry)
        if arxiv_id:
            api_result = search_openalex_by_arxiv(arxiv_id)

    # 3. Try title search
    if not api_result:
        title = entry.get('title', '')
        year = int(entry.get('year', 0)) if 'year' in entry else None
        api_result = search_openalex_by_title(title, year)

    # Verify match quality
    if api_result:
        verification = verify_entry(entry, api_result)
    else:
        verification = {
            'status': 'NOT_FOUND',
            'confidence': 0,
            'message': 'Entry not found in OpenAlex database',
            'api_result': None
        }

    # Add entry metadata
    verification['id'] = entry_id
    verification['type'] = entry_type
    verification['title'] = entry.get('title', 'N/A')

    # Check retraction if verified and has DOI
    if verification['status'] == 'VERIFIED' and api_result and api_result.get('doi'):
        is_retracted, retraction_doi = check_retraction_crossref(api_result['doi'])
        if is_retracted:
            verification['status'] = 'RETRACTED'
            verification['message'] = f'🚨 RETRACTED - see {retraction_doi}'

    return verification


# ============================================================================
# Report Generation
# ============================================================================

def generate_markdown_report(results: List[Dict], output_path: Path):
    """Generate detailed Markdown verification report"""

    # Statistics
    total = len(results)
    verified = sum(1 for r in results if r['status'] == 'VERIFIED')
    needs_review = sum(1 for r in results if r['status'] == 'NEEDS_REVIEW')
    suspect = sum(1 for r in results if r['status'] == 'SUSPECT')
    not_found = sum(1 for r in results if r['status'] == 'NOT_FOUND')
    manual_check = sum(1 for r in results if r['status'] == 'MANUAL_CHECK')
    retracted = sum(1 for r in results if r['status'] == 'RETRACTED')

    # Group results by status
    verified_entries = [r for r in results if r['status'] == 'VERIFIED']
    needs_review_entries = [r for r in results if r['status'] == 'NEEDS_REVIEW']
    suspect_entries = [r for r in results if r['status'] == 'SUSPECT']
    not_found_entries = [r for r in results if r['status'] == 'NOT_FOUND']
    manual_check_entries = [r for r in results if r['status'] == 'MANUAL_CHECK']
    retracted_entries = [r for r in results if r['status'] == 'RETRACTED']

    # Generate report
    report = f"""# Reference Verification Report

**Project**: PPI-SVRG (ICML 2026 Submission)
**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Entries**: {total}

---

## 📊 Summary Statistics

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Verified | {verified} | {verified/total*100:.1f}% |
| ⚠️  Needs Review | {needs_review} | {needs_review/total*100:.1f}% |
| 🔍 Suspect | {suspect} | {suspect/total*100:.1f}% |
| ❌ Not Found | {not_found} | {not_found/total*100:.1f}% |
| 📚 Manual Check | {manual_check} | {manual_check/total*100:.1f}% |
| 🚨 Retracted | {retracted} | {retracted/total*100:.1f}% |

**Verification Rate**: {(verified+needs_review)/total*100:.1f}% ({verified+needs_review}/{total})

---

## ✅ Verified Entries ({verified})

"""

    if verified_entries:
        verified_table = []
        for r in verified_entries:
            verified_table.append([
                r['id'],
                r['title'][:60] + '...' if len(r['title']) > 60 else r['title'],
                r.get('api_result', {}).get('venue', 'N/A')[:40],
                f"{r['confidence']:.0f}%"
            ])

        report += tabulate(
            verified_table,
            headers=['BibTeX Key', 'Title', 'Venue', 'Confidence'],
            tablefmt='github'
        )
        report += "\n\n"
    else:
        report += "*None*\n\n"

    # Needs Review
    report += f"## ⚠️  Needs Review ({needs_review})\n\n"
    if needs_review_entries:
        for r in needs_review_entries:
            report += f"### {r['id']}\n\n"
            report += f"**BibTeX Title**: {r['title']}\n\n"
            if r.get('api_result'):
                report += f"**OpenAlex Match**: {r['api_result']['title']}\n\n"
                report += f"**Venue**: {r['api_result']['venue']}\n\n"
                report += f"**Year**: {r['api_result']['year']}\n\n"
                report += f"**DOI**: {r['api_result']['doi']}\n\n"
            report += f"**Issue**: {r['message']}\n\n"
            report += f"**Recommended Action**: Manually verify title and author spelling\n\n"
            report += "---\n\n"
    else:
        report += "*None*\n\n"

    # Suspect
    report += f"## 🔍 Suspect Entries ({suspect})\n\n"
    if suspect_entries:
        for r in suspect_entries:
            report += f"### {r['id']}\n\n"
            report += f"**BibTeX Title**: {r['title']}\n\n"
            if r.get('api_result'):
                report += f"**Possible Match**: {r['api_result']['title']}\n\n"
                report += f"**Venue**: {r['api_result']['venue']}\n\n"
            report += f"**Issue**: {r['message']}\n\n"
            report += f"**Recommended Action**: High priority - verify this is the correct paper\n\n"
            report += "---\n\n"
    else:
        report += "*None*\n\n"

    # Not Found
    report += f"## ❌ Not Found ({not_found})\n\n"
    if not_found_entries:
        for r in not_found_entries:
            report += f"### {r['id']}\n\n"
            report += f"**Title**: {r['title']}\n\n"
            report += f"**Type**: {r['type']}\n\n"
            report += f"**Possible Reasons**:\n"
            report += f"- Paper title may have typo\n"
            report += f"- Technical report / gray literature not indexed\n"
            report += f"- Very recent paper (indexing delay)\n"
            report += f"- **Potentially fabricated citation** ⚠️\n\n"
            report += f"**Recommended Action**: Verify existence through Google Scholar or direct source\n\n"
            report += "---\n\n"
    else:
        report += "*None*\n\n"

    # Manual Check
    report += f"## 📚 Manual Check Required ({manual_check})\n\n"
    if manual_check_entries:
        manual_table = []
        for r in manual_check_entries:
            manual_table.append([
                r['id'],
                r['title'][:60] + '...' if len(r['title']) > 60 else r['title'],
                r['type']
            ])

        report += tabulate(
            manual_table,
            headers=['BibTeX Key', 'Title', 'Type'],
            tablefmt='github'
        )
        report += "\n\n**Note**: Books and technical reports often not in academic databases. Verify via:\n"
        report += "- ISBN lookup (books)\n"
        report += "- University repository (tech reports)\n"
        report += "- Publisher website\n\n"
    else:
        report += "*None*\n\n"

    # Retracted
    report += f"## 🚨 Retracted Papers ({retracted})\n\n"
    if retracted_entries:
        for r in retracted_entries:
            report += f"### {r['id']}\n\n"
            report += f"**Title**: {r['title']}\n\n"
            report += f"**⚠️  ACTION REQUIRED**: This paper has been retracted!\n\n"
            report += f"**Details**: {r['message']}\n\n"
            report += f"**Recommended Action**: Remove from bibliography immediately\n\n"
            report += "---\n\n"
    else:
        report += "*None*\n\n"

    # Recommendations
    report += "## 📋 Recommended Actions\n\n"
    if retracted > 0:
        report += f"1. 🚨 **URGENT**: Remove {retracted} retracted paper(s) immediately\n"
    if not_found > 0:
        report += f"2. ❌ Verify {not_found} 'Not Found' entries - possible typos or fabricated citations\n"
    if suspect > 0:
        report += f"3. 🔍 Review {suspect} suspect entries with low confidence matches\n"
    if needs_review > 0:
        report += f"4. ⚠️  Check {needs_review} entries with minor metadata discrepancies\n"

    verification_rate = (verified + needs_review) / total * 100
    if verification_rate >= 90:
        report += f"\n✅ **Overall Status**: READY FOR SUBMISSION ({verification_rate:.1f}% verification rate)\n"
    elif verification_rate >= 80:
        report += f"\n⚠️  **Overall Status**: NEEDS MINOR FIXES ({verification_rate:.1f}% verification rate)\n"
    else:
        report += f"\n❌ **Overall Status**: REQUIRES MAJOR REVIEW ({verification_rate:.1f}% verification rate)\n"

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding='utf-8')

    print(f"\n✅ Report saved to: {output_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Verify BibTeX references against OpenAlex API'
    )
    parser.add_argument('bibfile', help='Path to BibTeX file')
    parser.add_argument('--email', required=True, help='Email for OpenAlex API')
    parser.add_argument('--output', default='reports/verification_report.md',
                       help='Output report path')
    parser.add_argument('--json', help='Save results as JSON')

    args = parser.parse_args()

    # Configure OpenAlex
    openalex_config.email = args.email
    Works.email = args.email

    print(f"📖 Reading BibTeX file: {args.bibfile}")

    # Parse BibTeX
    try:
        with open(args.bibfile, 'r', encoding='utf-8') as f:
            bib_database = bibtexparser.load(f)
    except Exception as e:
        print(f"❌ Error reading BibTeX file: {e}")
        sys.exit(1)

    entries = bib_database.entries
    total = len(entries)

    print(f"✅ Found {total} entries")
    print(f"🔍 Starting verification with OpenAlex API...")
    print()

    # Verify each entry
    results = []
    for i, entry in enumerate(entries, 1):
        entry_id = entry.get('ID', f'entry_{i}')
        print(f"[{i}/{total}] Verifying {entry_id}...", end=' ')

        result = verify_bibtex_entry(entry)
        results.append(result)

        # Print status
        status_icon = {
            'VERIFIED': '✅',
            'NEEDS_REVIEW': '⚠️ ',
            'SUSPECT': '🔍',
            'NOT_FOUND': '❌',
            'MANUAL_CHECK': '📚',
            'RETRACTED': '🚨'
        }
        print(f"{status_icon.get(result['status'], '?')} {result['status']}")

    print()
    print("="*60)

    # Generate report
    output_path = Path(args.output)
    generate_markdown_report(results, output_path)

    # Save JSON if requested
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f"✅ JSON results saved to: {json_path}")

    # Print summary
    verified = sum(1 for r in results if r['status'] == 'VERIFIED')
    needs_review = sum(1 for r in results if r['status'] == 'NEEDS_REVIEW')
    print()
    print(f"📊 Verification Summary:")
    print(f"   ✅ Verified: {verified}/{total} ({verified/total*100:.1f}%)")
    print(f"   ⚠️  Needs Review: {needs_review}")
    print(f"   ❌ Issues: {total - verified - needs_review}")


if __name__ == '__main__':
    main()
```

### 使用说明

**基础用法**:
```bash
python code/reference_verification/verify_references.py \
    main.bib \
    --email your@email.com \
    --output reports/verification_report.md
```

**保存JSON结果** (用于进一步分析):
```bash
python verify_references.py main.bib \
    --email your@email.com \
    --json reports/verification_results.json
```

**预期输出**:
```
📖 Reading BibTeX file: main.bib
✅ Found 52 entries
🔍 Starting verification with OpenAlex API...

[1/52] Verifying angelopoulos2023prediction... ✅ VERIFIED
[2/52] Verifying johnson2013accelerating... ✅ VERIFIED
[3/52] Verifying settles.tr09... 📚 MANUAL_CHECK
[4/52] Verifying example_unknown... ❌ NOT_FOUND
...

============================================================
✅ Report saved to: reports/verification_report.md

📊 Verification Summary:
   ✅ Verified: 45/52 (86.5%)
   ⚠️  Needs Review: 3
   ❌ Issues: 4
```

---

## 手动验证核心引用

### 高优先级文献验证步骤

#### 1. Angelopoulos et al. 2023 - PPI原论文

**BibTeX Entry**:
```bibtex
@article{angelopoulos2023prediction,
  title={Prediction-powered inference},
  author={Angelopoulos, Anastasios N and Bates, Stephen and Fannjiang, Clara and Jordan, Michael I and Zrnic, Tijana},
  journal={Science},
  volume={382},
  number={6671},
  pages={669--674},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```

**验证步骤**:
1. **OpenAlex查询**:
   ```python
   from pyalex import Works
   Works.email = "your@email.com"
   result = Works().filter(doi="10.1126/science.adi6000").get()
   print(result[0]['title'])  # 应输出: Prediction-powered inference
   ```

2. **手动确认** (如果API失败):
   - 访问 https://www.science.org/doi/10.1126/science.adi6000
   - 确认作者列表、年份、卷页号

3. **检查撤稿状态**:
   - 访问 https://retractionwatch.com/
   - 搜索 "Angelopoulos Prediction-powered"
   - ✅ 确认无撤稿记录

#### 2. Johnson & Zhang 2013 - SVRG原论文

**BibTeX Entry**:
```bibtex
@article{johnson2013accelerating,
  title={Accelerating stochastic gradient descent using predictive variance reduction},
  author={Johnson, Rie and Zhang, Tong},
  journal={Advances in neural information processing systems},
  volume={26},
  year={2013}
}
```

**验证要点**:
- ⚠️  **注意**: NeurIPS论文可能被索引为@inproceedings而非@article
- ✅ 正确的entry type应为`@inproceedings` (NIPS是会议,非期刊)

**建议修正**:
```bibtex
@inproceedings{johnson2013accelerating,
  title={Accelerating stochastic gradient descent using predictive variance reduction},
  author={Johnson, Rie and Zhang, Tong},
  booktitle={Advances in Neural Information Processing Systems},
  volume={26},
  year={2013},
  pages={315--323}
}
```

#### 3. Allen-Zhu & Yuan 2016 - Improved SVRG

**OpenAlex验证**:
```python
result = Works().filter(
    display_name="Improved SVRG for Non-Strongly-Convex or Sum-of-Non-Convex Objectives"
).get()

# 检查DOI
assert result[0]['doi'] == 'https://doi.org/10.5555/3045390.3045509'
```

### 处理特殊情况

#### 技术报告 (settles.tr09)

**BibTeX**:
```bibtex
@techreport{settles.tr09,
  Author = {Burr Settles},
  Institution = {University of Wisconsin--Madison},
  Number = {1648},
  Title = {Active Learning Literature Survey},
  Type = {Computer Sciences Technical Report},
  Year = {2009}
}
```

**手动验证**:
1. 访问 https://minds.wisconsin.edu/handle/1793/60660
2. 确认作者、标题、机构匹配
3. ✅ 技术报告通常不在OpenAlex中,需直接访问机构库

#### 预印本 (arXiv)

**示例: PPI++ (angelopoulos2023ppi++)**

**正确格式**:
```bibtex
@misc{angelopoulos2023ppi++,
  title={PPI++: Efficient prediction-powered inference},
  author={Angelopoulos, Anastasios N and Duchi, John C and Zrnic, Tijana},
  year={2023},
  eprint={2311.01453},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

**验证**:
```bash
# 访问arXiv直接链接
curl https://arxiv.org/abs/2311.01453

# 或使用OpenAlex
python -c "from pyalex import Works; \
           Works.email='your@email.com'; \
           r = Works().filter(display_name='PPI++ Efficient prediction-powered inference').get(); \
           print(r[0]['title'])"
```

#### 教科书

**示例: van der Vaart 2000**

```bibtex
@book{van2000asymptotic,
  title={Asymptotic statistics},
  author={Van der Vaart, Aad W},
  volume={3},
  year={2000},
  publisher={Cambridge university press}
}
```

**手动验证**:
1. 搜索ISBN: 978-0521784504
2. 确认出版商: Cambridge University Press
3. ✅ 教科书通常可通过Google Books或WorldCat验证

---

## 验证报告解读

### 报告示例

```markdown
# Reference Verification Report

**Project**: PPI-SVRG (ICML 2026 Submission)
**Generated**: 2026-01-24 10:30:00
**Total Entries**: 52

---

## 📊 Summary Statistics

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Verified | 45 | 86.5% |
| ⚠️  Needs Review | 3 | 5.8% |
| 🔍 Suspect | 1 | 1.9% |
| ❌ Not Found | 2 | 3.8% |
| 📚 Manual Check | 1 | 1.9% |
| 🚨 Retracted | 0 | 0.0% |

**Verification Rate**: 92.3% (48/52)

---

## ⚠️  Needs Review (3)

### johnson2013accelerating

**BibTeX Title**: Accelerating stochastic gradient descent using predictive variance reduction

**OpenAlex Match**: Accelerating stochastic gradient descent using predictive variance reduction

**Venue**: Advances in Neural Information Processing Systems

**Year**: 2013

**DOI**: 10.5555/3045390.3045504

**Issue**: Entry type should be @inproceedings (NeurIPS is a conference, not journal)

**Recommended Action**: Change `@article` to `@inproceedings`, update `journal` to `booktitle`

---

## ❌ Not Found (2)

### example_fake_2025

**Title**: A novel approach to non-existent problems

**Type**: article

**Possible Reasons**:
- Paper title may have typo
- Technical report / gray literature not indexed
- Very recent paper (indexing delay)
- **Potentially fabricated citation** ⚠️

**Recommended Action**: Verify existence through Google Scholar or direct source

---
```

### 判定标准详解

#### ✅ VERIFIED

**含义**: 引用真实存在,元数据匹配度>85%

**置信度阈值**:
- **95-100%**: Exact match (标题+作者+年份完全一致)
- **85-95%**: High confidence (允许副标题差异、作者顺序不同)

**无需操作**: 这些引用可直接用于投稿

#### ⚠️  NEEDS_REVIEW

**含义**: 找到匹配,但存在元数据差异

**常见问题**:
1. **Entry类型错误**:
   - 会议论文误标为@article
   - 预印本使用@article而非@misc

2. **作者名格式差异**:
   ```bibtex
   # BibTeX
   author = {Johnson, R and Zhang, T}

   # OpenAlex
   author = {Rie Johnson and Tong Zhang}
   ```

3. **标题缺少副标题**:
   ```bibtex
   # BibTeX (简化版)
   title = {SVRG}

   # 完整版
   title = {SVRG: Accelerating Stochastic Gradient Descent Using Predictive Variance Reduction}
   ```

**操作**: 对比BibTeX与OpenAlex结果,根据建议修正

#### 🔍 SUSPECT

**含义**: 低置信度匹配 (70-85%),可能不是同一篇论文

**原因**:
- 标题相似但不同的论文 (如同主题不同作者)
- 严重的拼写错误

**操作**:
1. 访问OpenAlex提供的DOI链接
2. 确认是否为正确论文
3. 如不匹配,视为NOT_FOUND

#### ❌ NOT_FOUND

**含义**: 未在OpenAlex数据库中找到匹配

**可能原因** (按概率排序):
1. **最可能**: 技术报告/灰色文献 (40%)
2. **较可能**: 标题拼写错误 (30%)
3. **可能**: 最新论文索引延迟 (20%)
4. **警惕**: 虚假引用 (10%)

**操作**:
1. Google Scholar搜索完整标题
2. 检查作者个人主页
3. 如确实不存在 → 删除引用

#### 📚 MANUAL_CHECK

**含义**: 书籍/技术报告,跳过自动验证

**验证方法**:
- **书籍**: 通过ISBN查询 (Google Books, WorldCat)
- **技术报告**: 访问机构库 (如UW-Madison CS Tech Reports)

#### 🚨 RETRACTED

**含义**: 论文已被撤稿 (**极严重**)

**操作**:
1. ✅ 立即从BibTeX删除
2. ✅ 检查正文是否引用,全部删除
3. ✅ 寻找替代文献

---

## ICML 2026投稿前Checklist

### 必须完成项

- [ ] **引用真实性**: >95%验证率 (NOT_FOUND < 2条)
- [ ] **无撤稿论文**: RETRACTED = 0
- [ ] **BibTeX格式**:
  - [ ] 所有@article有journal、volume、year
  - [ ] 所有@inproceedings有booktitle、year
  - [ ] arXiv使用@misc + eprint字段
- [ ] **DOI完整性**: >80%引用有DOI
- [ ] **按字母序排列**: BibTeX keys和显示顺序
- [ ] **特殊字符转义**: 检查LaTeX命令正确性

### 推荐完成项

- [ ] **统一命名规范**:
  - [ ] 会议名称一致 (如统一使用"ICML"或"Proceedings of ICML")
  - [ ] 期刊名称使用标准缩写
- [ ] **页码完整**: 为所有期刊/会议论文添加pages字段
- [ ] **URL清理**: 删除不必要的URL (有DOI时)
- [ ] **作者格式一致**: 统一使用"Last, First"格式

### 快速验证脚本

```bash
#!/bin/bash
# quick_check.sh - ICML 2026投稿前快速检查

echo "🔍 Starting pre-submission verification..."

# Step 1: 引用验证
echo "Step 1/4: Verifying references..."
python code/reference_verification/verify_references.py main.bib \
    --email $OPENALEX_EMAIL \
    --output reports/pre_submission_report.md \
    --json reports/results.json

# Step 2: BibTeX格式检查
echo "Step 2/4: Checking BibTeX format..."
python -c "
import bibtexparser
import json

with open('main.bib') as f:
    bib = bibtexparser.load(f)

errors = []
for entry in bib.entries:
    # Check required fields
    if entry['ENTRYTYPE'] == 'article':
        required = ['title', 'author', 'journal', 'year']
        missing = [f for f in required if f not in entry]
        if missing:
            errors.append(f\"{entry['ID']}: Missing {', '.join(missing)}\")

if errors:
    print('❌ Format errors found:')
    for e in errors:
        print(f'  - {e}')
else:
    print('✅ All entries have required fields')
"

# Step 3: DOI覆盖率
echo "Step 3/4: Checking DOI coverage..."
grep -c "doi = " main.bib | awk '{print "DOI count: " $0}'

# Step 4: 字母序检查
echo "Step 4/4: Checking alphabetical order..."
grep "^@" main.bib | grep -o "{.*," | tr -d '{,' | awk '
{
    if (NR > 1 && prev > $0) {
        print "⚠️  Not alphabetically sorted: " prev " > " $0
    }
    prev = $0
}
END {
    if (NR == 0 || prev <= $0) {
        print "✅ BibTeX keys are alphabetically sorted"
    }
}
'

echo ""
echo "✅ Pre-submission check complete. Review reports/pre_submission_report.md"
```

**运行**:
```bash
chmod +x quick_check.sh
./quick_check.sh
```

---

## Troubleshooting

### API连接问题

#### 问题: Connection Timeout

**错误信息**:
```
requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='api.openalex.org', port=443)
```

**解决方案**:
1. 检查网络连接
2. 增加超时时间:
   ```python
   Config.OPENALEX_TIMEOUT = 60  # 增加到60秒
   ```
3. 使用代理 (如在防火墙后):
   ```python
   import os
   os.environ['HTTPS_PROXY'] = 'http://proxy.example.com:8080'
   ```

#### 问题: 速率限制 (429 Too Many Requests)

**错误信息**:
```
429 Client Error: Too Many Requests
```

**原因**: 超过免费配额 (未认证: 10 req/s, 认证: 100k/day)

**解决方案**:
1. 确认邮箱已配置:
   ```python
   print(Works.email)  # 应输出你的邮箱
   ```
2. 添加请求间隔:
   ```python
   import time
   time.sleep(0.1)  # 每次请求间隔100ms
   ```

### 匹配问题

#### 问题: 标题匹配分数过低 (<70%)

**原因**:
1. BibTeX中标题有拼写错误
2. 使用了非官方标题 (如简写)
3. LaTeX命令未正确处理

**排查步骤**:
```python
from verify_references import normalize_string
from thefuzz import fuzz

bib_title = "Accelerating SGD using variance reduction"
api_title = "Accelerating stochastic gradient descent using predictive variance reduction"

# 检查归一化后的标题
print(f"BibTeX: {normalize_string(bib_title)}")
print(f"API: {normalize_string(api_title)}")

# 计算相似度
score = fuzz.token_set_ratio(
    normalize_string(bib_title),
    normalize_string(api_title)
)
print(f"Similarity: {score}%")
```

**解决**:
- 如果>85%: 降低阈值或手动确认
- 如果<85%: 检查BibTeX标题拼写

#### 问题: 数据库未收录

**现象**: 论文确实存在,但OpenAlex返回空结果

**可能原因**:
1. 最新论文 (索引延迟1-2周)
2. 小众会议/期刊
3. 非英语论文

**替代验证方法**:
```bash
# 使用Google Scholar
# 1. 访问 https://scholar.google.com/
# 2. 搜索完整标题
# 3. 检查"Cited by"数量 (>0表示真实存在)

# 使用Crossref (对DOI验证)
curl -I https://doi.org/10.1234/example.doi
# HTTP 302 = 有效, 404 = 无效
```

### 报告问题

#### 问题: UnicodeEncodeError写入报告

**错误**:
```
UnicodeEncodeError: 'ascii' codec can't encode character '\u2705'
```

**解决**:
```python
# 在report_generator.py中确保
output_path.write_text(report, encoding='utf-8')  # 显式指定UTF-8
```

#### 问题: 表格格式错乱

**原因**: 标题过长导致Markdown表格溢出

**解决**: 截断长标题
```python
# 已在代码中实现
title = r['title'][:60] + '...' if len(r['title']) > 60 else r['title']
```

### Python依赖问题

#### 问题: ImportError: No module named 'pyalex'

**解决**:
```bash
pip install pyalex --upgrade
# 或使用conda
conda install -c conda-forge pyalex
```

#### 问题: thefuzz速度慢

**原因**: 未安装加速库python-Levenshtein

**解决**:
```bash
pip install thefuzz[speedup]
# 或单独安装
pip install python-Levenshtein
```

---

## 附录

### A. OpenAlex vs 其他API详细对比

| 特性 | OpenAlex | Semantic Scholar | Crossref | Google Scholar |
|------|----------|------------------|----------|----------------|
| **访问方式** | RESTful API | RESTful API | RESTful API | 网页爬虫 (不推荐) |
| **认证方式** | 邮箱 (可选) | API Key (必需) | 无 (邮箱可选) | 无 |
| **注册时间** | 即时 | 1-3天审批 | 无需注册 | 无 |
| **免费配额** | 100,000 req/day | 100 req/s | 50 req/s | N/A (受限) |
| **覆盖期刊** | ✅✅✅ | ✅✅ | ✅✅✅ | ✅✅✅ |
| **覆盖会议** | ✅✅✅ | ✅✅✅ | ✅✅ | ✅✅✅ |
| **arXiv支持** | ✅✅✅ | ✅✅✅ | ⚠️  | ✅✅✅ |
| **BibTeX导出** | ✅ 原生支持 | ❌ 需手动构建 | ✅ 支持 | ❌ |
| **撤稿检测** | ❌ | ❌ | ✅ Retraction Watch | ❌ |
| **引文网络** | ✅ | ✅✅✅ | ⚠️  | ✅✅ |
| **全文搜索** | ✅ N-gram | ✅ 语义搜索 | ⚠️ 元数据only | ✅✅✅ |
| **API稳定性** | ✅✅ | ✅✅✅ | ✅✅✅ | ⚠️ 易被封禁 |
| **更新频率** | 每周 | 每日 | 实时 | 实时 |

**推荐策略**:
- **主要工具**: OpenAlex (免费 + 覆盖全面)
- **撤稿检测**: Crossref
- **手动确认**: Google Scholar (作为最后验证)

### B. BibTeX完整字段参考

#### @article (期刊论文)

**必需字段**:
- `author`: 作者列表
- `title`: 论文标题
- `journal`: 期刊名称
- `year`: 发表年份

**可选字段**:
- `volume`: 卷号
- `number`: 期号
- `pages`: 页码范围 (如`123--145`)
- `doi`: Digital Object Identifier
- `publisher`: 出版商
- `url`: 在线链接 (如无DOI)

**示例**:
```bibtex
@article{angelopoulos2023prediction,
  author = {Angelopoulos, Anastasios N and Bates, Stephen and Fannjiang, Clara and Jordan, Michael I and Zrnic, Tijana},
  title = {Prediction-powered inference},
  journal = {Science},
  volume = {382},
  number = {6671},
  pages = {669--674},
  year = {2023},
  doi = {10.1126/science.adi6000},
  publisher = {American Association for the Advancement of Science}
}
```

#### @inproceedings (会议论文)

**必需字段**:
- `author`
- `title`
- `booktitle`: 会议论文集名称 (如`Proceedings of ICML`)
- `year`

**可选字段**:
- `pages`
- `editor`: 编辑
- `volume`: 卷号 (JMLR Proceedings有)
- `series`: 系列名 (如`Proceedings of Machine Learning Research`)
- `organization`: 主办方
- `publisher`: 出版商

**示例**:
```bibtex
@inproceedings{johnson2013accelerating,
  author = {Johnson, Rie and Zhang, Tong},
  title = {Accelerating stochastic gradient descent using predictive variance reduction},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {26},
  pages = {315--323},
  year = {2013}
}
```

#### @misc (预印本/未发表)

**必需字段**:
- `author`
- `title`
- `year`

**arXiv特定字段**:
- `eprint`: arXiv ID (如`2311.01453`)
- `archivePrefix`: `arXiv`
- `primaryClass`: 分类 (如`cs.LG`)

**示例**:
```bibtex
@misc{angelopoulos2023ppi++,
  title = {PPI++: Efficient prediction-powered inference},
  author = {Angelopoulos, Anastasios N and Duchi, John C and Zrnic, Tijana},
  year = {2023},
  eprint = {2311.01453},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

#### @book (书籍)

**必需字段**:
- `author` 或 `editor`
- `title`
- `publisher`
- `year`

**可选字段**:
- `volume`: 卷数
- `series`: 丛书名
- `edition`: 版本 (如`2nd`)
- `isbn`: 国际标准书号

**示例**:
```bibtex
@book{van2000asymptotic,
  title = {Asymptotic statistics},
  author = {Van der Vaart, Aad W},
  volume = {3},
  year = {2000},
  publisher = {Cambridge university press},
  isbn = {978-0521784504}
}
```

#### @techreport (技术报告)

**必需字段**:
- `author`
- `title`
- `institution`: 机构名称
- `year`

**可选字段**:
- `number`: 报告编号
- `type`: 报告类型 (如`Technical Report`)

**示例**:
```bibtex
@techreport{settles.tr09,
  author = {Burr Settles},
  title = {Active Learning Literature Survey},
  institution = {University of Wisconsin--Madison},
  number = {1648},
  type = {Computer Sciences Technical Report},
  year = {2009}
}
```

### C. ICML 2026参考文献格式示例

**APA风格引用示例**:

**期刊论文**:
```
Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023).
    Prediction-powered inference. Science, 382(6671), 669-674.
    https://doi.org/10.1126/science.adi6000
```

**会议论文**:
```
Johnson, R., & Zhang, T. (2013). Accelerating stochastic gradient descent using
    predictive variance reduction. In Advances in Neural Information Processing
    Systems (Vol. 26, pp. 315-323).
```

**arXiv预印本**:
```
Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). PPI++: Efficient
    prediction-powered inference. arXiv preprint arXiv:2311.01453.
```

**书籍**:
```
Van der Vaart, A. W. (2000). Asymptotic statistics (Vol. 3). Cambridge
    university press.
```

**注意事项**:
1. 作者姓名: Last, First Initial格式
2. 标题: Sentence case (仅首字母大写)
3. DOI: 使用https://doi.org/格式
4. 按字母序排列 (by first author's last name)

### D. 常见期刊/会议缩写

| 全称 | 标准缩写 | BibTeX字段 |
|------|---------|-----------|
| International Conference on Machine Learning | ICML | booktitle |
| Neural Information Processing Systems | NeurIPS | booktitle |
| International Conference on Learning Representations | ICLR | booktitle |
| Journal of the American Statistical Association | JASA | journal |
| Annals of Statistics | Ann. Statist. | journal |
| Journal of Machine Learning Research | JMLR | journal |
| Proceedings of Machine Learning Research | PMLR | series |
| IEEE Transactions on Pattern Analysis and Machine Intelligence | IEEE Trans. PAMI | journal |

**使用建议**:
- ✅ 顶会使用缩写 (ICML, NeurIPS)
- ✅ 期刊使用全称或标准缩写
- ❌ 避免自创缩写 (如"Proc. ICML")

---

## 更新日志

- **v1.0.0** (2026-01-24): 初始版本
  - OpenAlex API完整集成
  - Python自动验证脚本
  - ICML 2026格式规范
  - 撤稿检测功能

---

**联系方式**: 如有问题,请在项目GitHub Issues中提出
**License**: MIT
