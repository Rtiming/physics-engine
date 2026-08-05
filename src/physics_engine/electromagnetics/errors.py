"""电磁域的一切失败关闭。

与`optics/errors.py`同形制：一个域一个异常类型，不再细分。
细分要等到调用方真的需要按类型分支处理——今天没有任何调用方这么做。
"""

from __future__ import annotations


class ElectromagneticsError(ValueError):
    """电磁域的一切失败关闭。"""


__all__ = ["ElectromagneticsError"]
