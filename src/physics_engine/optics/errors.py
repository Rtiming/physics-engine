"""光学域的失败关闭类型。

一个域一个异常类型（形制同`materials.MaterialError`/`oracles.OracleError`）：
调用方能按域捕获，而域内各模块不必各造一个。
"""

from __future__ import annotations


class OpticsError(ValueError):
    """光学域的一切失败关闭：域外输入、缺声明、单位制对不上。"""


__all__ = ["OpticsError"]
