#!/usr/bin/env python3
"""
测试导入是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")

try:
    from src.utils.config import Config
    print("✓ src.utils.config")
except ImportError as e:
    print(f"✗ src.utils.config: {e}")

try:
    from src.utils.nfe_counter import NFECounter
    print("✓ src.utils.nfe_counter")
except ImportError as e:
    print(f"✗ src.utils.nfe_counter: {e}")

try:
    from src.models.edm_model import EDMModel
    print("✓ src.models.edm_model")
except ImportError as e:
    print(f"✗ src.models.edm_model: {e}")

try:
    from src.verifiers.scorer_verifier import ScorerVerifier
    print("✓ src.verifiers.scorer_verifier")
except ImportError as e:
    print(f"✗ src.verifiers.scorer_verifier: {e}")

try:
    from src.search.diffusion_tts_search import BestOfNSearch, ZeroOrderSearchTTS, EpsilonGreedySearch
    print("✓ src.search.diffusion_tts_search")
except ImportError as e:
    print(f"✗ src.search.diffusion_tts_search: {e}")

try:
    from src.evaluation.metrics import compute_fid_is
    print("✓ src.evaluation.metrics")
except ImportError as e:
    print(f"✗ src.evaluation.metrics: {e}")

print("\nAll imports tested!")

