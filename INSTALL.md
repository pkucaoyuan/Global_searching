# 安装指南

## 正确的安装方法

### 方法1: 可编辑模式安装（推荐）

```bash
# 1. 进入项目目录
cd ~/efs/cy/GLOBAL/Global_searching

# 2. 激活conda环境
conda activate global

# 3. 以可编辑模式安装项目
pip install -e .

# 4. 安装其他依赖（如果需要）
pip install -r requirements.txt
```

**优点**: 
- 修改代码后无需重新安装
- 可以直接使用 `from src.models import ...` 导入

### 方法2: 仅安装依赖 + PYTHONPATH

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置PYTHONPATH（添加到 ~/.bashrc 或每次运行前执行）
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 3. 运行脚本
python scripts/run_diffusion_tts_experiment.py --config configs/imagenet64_diffusion_tts.yaml
```

### 方法3: 使用 setup.py

```bash
# 开发模式（推荐）
python setup.py develop

# 或普通安装
python setup.py install
```

## 常见错误

### ❌ 错误: `pip install src`
这会尝试从PyPI安装名为`src`的包，而不是安装本地项目。

### ✅ 正确: `pip install -e .`
这会以可编辑模式安装当前目录的项目。

## 验证安装

```bash
# 测试导入
python -c "from src.models.edm_model import EDMModel; print('✓ EDM Model')"
python -c "from src.verifiers.scorer_verifier import ScorerVerifier; print('✓ Scorer Verifier')"
python -c "from src.search.diffusion_tts_search import EpsilonGreedySearch; print('✓ Search Methods')"
```

## 完整设置流程

```bash
# 1. 创建conda环境
conda create -n global_searching python=3.10 -y
conda activate global_searching

# 2. 安装PyTorch（根据GPU选择）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 进入项目目录
cd ~/efs/cy/GLOBAL/Global_searching

# 4. 安装项目（可编辑模式）
pip install -e .

# 5. 安装其他依赖
pip install -r requirements.txt

# 6. 克隆外部代码库
mkdir -p code_repos
cd code_repos
git clone https://github.com/rvignav/diffusion-tts.git
cd ..

# 7. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from src.models.edm_model import EDMModel; print('Installation successful!')"
```




