"""状态形制——spec/12第二节的参考实现（T4第一片）。

三层分家（spec/12第2.1节）：**状态**每步都变、**参考**步内冻结、**上下文**案例内冻结。
本模块只做状态层与它的打包契约；参考与上下文随第一块真实内核搬迁时长出来。

两条硬条款（spec/12第2.2节）：

1. **显式数组不是对象图**——状态是能``pack``成一维向量、能``with_vector``装回来的
   数值容器。理由是硬的：线性求解器与自动微分只认向量；且对象图没法在纯Python
   实现与加速档实现之间逐字节对拍（spec/12第五节）。
2. **打包次序是形制的一部分**——`StateLayout`把次序与语义映射显式写下来并锁进
   `layout_id`。次序换了，梯度与Hessian的索引全错，而多数测试不会发现；
   这是典型的"跑得通但全错"，所以次序必须是**被声明的、可对拍的字节**，
   不是实现里的一个隐含约定。

**真历史 vs 求解器便利**（spec/12第2.2节第三条）：塑性set、粘着锚点是真历史，
必须进状态并随状态被复现；"上一步的切线矩阵"这类只是缓存，不进状态。
本层用`history_fields`把这条分界写成声明——分不清就都当真历史（保守方向）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.canonical import FTS_PROFILE, canonical_bytes, canonical_sha256
from physics_engine.identity import has_unit_suffix


class StateError(ValueError):
    """状态层的一切失败关闭。"""


@dataclass(frozen=True)
class StateField:
    """一个自由度块：名字带单位后缀（轴2），宽度是它占多少个标量。"""

    name: str
    width: int
    is_history: bool = False

    def __post_init__(self) -> None:
        if not has_unit_suffix(self.name):
            raise StateError(
                f"state field must carry a unit suffix (axis 2): {self.name!r}"
            )
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise StateError(f"state field width must be a positive integer: {self.name!r}")


@dataclass(frozen=True)
class StateLayout:
    """打包次序的显式声明。**次序即形制**，改它是破坏性变更。"""

    layout_id: str
    fields: tuple[StateField, ...]

    def __post_init__(self) -> None:
        if not self.layout_id.startswith("layout/"):
            raise StateError("layout_id must be namespaced like 'layout/...'")
        if not self.fields:
            raise StateError("a layout needs at least one field")
        names = [field.name for field in self.fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise StateError(f"duplicate state field names: {duplicates}")

    @property
    def dof_count(self) -> int:
        return sum(field.width for field in self.fields)

    def offset_of(self, name: str) -> int:
        offset = 0
        for field in self.fields:
            if field.name == name:
                return offset
            offset += field.width
        raise StateError(f"unknown state field: {name!r}")

    def history_fields(self) -> tuple[str, ...]:
        """真历史字段名。分不清真历史与缓存时按保守方向声明为真历史。"""

        return tuple(field.name for field in self.fields if field.is_history)

    def to_document(self) -> dict:
        return {
            "layout_id": self.layout_id,
            "dof_count": self.dof_count,
            "fields": [
                {"name": f.name, "width": f.width, "is_history": f.is_history}
                for f in self.fields
            ],
        }

    def fingerprint(self) -> str:
        """次序的内容地址——两个实现是否在同一个打包契约上，比这个就知道。"""

        return canonical_sha256(self.to_document(), FTS_PROFILE)


@dataclass(frozen=True)
class State:
    """一个时刻的状态：布局 + 一维向量。向量是元组——不可变，可逐字节对拍。"""

    layout: StateLayout
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.vector) != self.layout.dof_count:
            raise StateError(
                f"state vector has {len(self.vector)} entries but layout "
                f"{self.layout.layout_id} declares {self.layout.dof_count}"
            )
        if not all(math.isfinite(value) for value in self.vector):
            raise StateError("state vector must be finite — NaN/Inf never enter state")

    def with_vector(self, vector: tuple[float, ...]) -> State:
        return State(layout=self.layout, vector=tuple(vector))

    def block(self, name: str) -> tuple[float, ...]:
        """取一个自由度块。索引由布局算，调用方永不手写偏移量。"""

        offset = self.layout.offset_of(name)
        width = next(f.width for f in self.layout.fields if f.name == name)
        return self.vector[offset : offset + width]

    def pack(self) -> bytes:
        """规范化字节——逐字节对拍与内容寻址都走它。"""

        return canonical_bytes(
            {"layout": self.layout.to_document(), "vector": list(self.vector)},
            FTS_PROFILE,
        )


__all__ = [
    "State",
    "StateError",
    "StateField",
    "StateLayout",
]
