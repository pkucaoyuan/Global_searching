"""
NFE (Number of Function Evaluations) 计数器
确保所有方法的NFE计算公平一致
"""

from typing import Optional
import contextlib


class NFECounter:
    """
    统一的NFE计数器
    
    使用方法:
        counter = NFECounter()
        with counter.count():
            # 执行需要计数的操作
            model.forward(...)
        print(counter.total_nfe)
    """
    
    def __init__(self):
        self._count = 0
        self._enabled = True
    
    @contextlib.contextmanager
    def count(self, n: int = 1):
        """上下文管理器，自动计数"""
        if self._enabled:
            self._count += n
        yield
        # 如果需要在异常时回退计数，可以在这里处理
    
    def add(self, n: int = 1):
        """手动增加计数"""
        if self._enabled:
            self._count += n
    
    def reset(self):
        """重置计数"""
        self._count = 0
    
    def disable(self):
        """禁用计数（用于调试）"""
        self._enabled = False
    
    def enable(self):
        """启用计数"""
        self._enabled = True
    
    @property
    def total_nfe(self) -> int:
        """获取总NFE数（已弃用，使用current_nfe）"""
        return self._count
    
    @property
    def current_nfe(self) -> int:
        """获取当前NFE数"""
        return self._count
    
    def increment(self, n: int = 1):
        """增加计数（add的别名）"""
        self.add(n)
    
    def __repr__(self):
        return f"NFECounter(total_nfe={self._count})"


class NFECounterMixin:
    """
    可混入的NFE计数器，方便添加到现有类中
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nfe_counter: Optional[NFECounter] = None
    
    def set_nfe_counter(self, counter: NFECounter):
        """设置NFE计数器"""
        self.nfe_counter = counter
    
    def count_nfe(self, n: int = 1):
        """计数NFE"""
        if self.nfe_counter is not None:
            self.nfe_counter.add(n)

