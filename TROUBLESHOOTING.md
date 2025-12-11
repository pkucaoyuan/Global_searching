# 故障排查指南

## ModuleNotFoundError: No module named 'src.models'

### 问题原因

如果出现 `ModuleNotFoundError: No module named 'src.models'` 错误，可能是以下原因：

1. **没有安装项目**：需要先运行 `pip install -e .`
2. **路径问题**：脚本的路径处理可能有问题
3. **工作目录不对**：需要在项目根目录运行脚本

### 解决方案

#### 方案1: 确保已安装项目（推荐）

```bash
# 在项目根目录下
cd ~/efs/cy/GLOBAL/Global_searching

# 确保已安装项目（可编辑模式）
pip install -e .

# 验证安装
python -c "import src; print(src.__file__)"
```

#### 方案2: 使用PYTHONPATH

```bash
# 在项目根目录下
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行脚本
python scripts/run_diffusion_tts_experiment.py --config configs/imagenet64_diffusion_tts.yaml
```

#### 方案3: 使用模块方式运行

```bash
# 在项目根目录下
python -m scripts.run_diffusion_tts_experiment --config configs/imagenet64_diffusion_tts.yaml
```

#### 方案4: 检查路径

```bash
# 运行测试脚本
python test_import.py

# 如果失败，检查：
ls -la src/models/  # 应该能看到 edm_model.py
ls -la src/__init__.py  # 应该存在
```

### 完整设置流程

```bash
# 1. 克隆项目
git clone https://github.com/pkucaoyuan/Global_searching.git
cd Global_searching

# 2. 创建conda环境
conda create -n global_searching python=3.10 -y
conda activate global_searching

# 3. 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 安装项目（重要！）
pip install -e .

# 5. 安装其他依赖
pip install -r requirements.txt

# 6. 验证安装
python test_import.py

# 7. 运行实验
python scripts/run_diffusion_tts_experiment.py --config configs/imagenet64_diffusion_tts.yaml
```

### 调试技巧

如果仍然有问题，可以添加调试信息：

```python
# 在脚本开头添加
import sys
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())
print("Script location:", __file__)

from pathlib import Path
project_root = Path(__file__).parent.parent
print("Project root:", project_root)
print("src directory exists:", (project_root / "src").exists())
```

### 常见错误

1. **错误**: `ModuleNotFoundError: No module named 'src'`
   - **解决**: 运行 `pip install -e .` 或设置 `PYTHONPATH`

2. **错误**: `FileNotFoundError: src directory not found`
   - **解决**: 确保在项目根目录运行脚本

3. **错误**: `ImportError: cannot import name 'EDMModel'`
   - **解决**: 检查 `src/models/edm_model.py` 是否存在且包含 `EDMModel` 类

