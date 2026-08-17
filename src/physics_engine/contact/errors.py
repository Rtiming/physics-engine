"""接触层的失败关闭异常——与`optics/errors.py`、`electromagnetics/errors.py`同形制。

单独一个文件是为了让本子包的其余模块都能import它而**不产生任何环**：
它谁也不依赖。整个接触层的故事在`__init__.py`。
"""

from __future__ import annotations


class ContactError(ValueError):
    """接触层的一切失败关闭。"""

